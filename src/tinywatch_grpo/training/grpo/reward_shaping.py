"""Training-only GRPO reward shaping. Eval still uses Reward v1."""

from __future__ import annotations

from tinywatch_grpo.environment.reward import gold_overlap
from tinywatch_grpo.training.grpo.loop import _step_movie_ids


def terminal_reward_v1(trajectory: dict) -> float:
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    if detail.get("reward_valid") is not True:
        return 0.0
    try:
        return float(detail.get("reward", trajectory.get("final_reward") or 0.0))
    except (TypeError, ValueError):
        return float(trajectory.get("final_reward") or 0.0)


def gold_hit_fraction(task: dict, draft: dict) -> float:
    """Fraction of gold ids present in the draft (order-invariant)."""
    gold_ids = set((task.get("gold_watchlist") or {}).get("movie_ids") or [])
    if not gold_ids:
        return 0.0
    draft_ids = set(draft.get("movie_ids") or [])
    return len(gold_ids & draft_ids) / len(gold_ids)


def _ids_from_tools(trajectory: dict, names: set[str]) -> set[str]:
    found: set[str] = set()
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict) or step.get("guard_rejected"):
            continue
        if str(step.get("tool_name") or "") not in names:
            continue
        found.update(_step_movie_ids(step))
    return found


def verified_gold_hit_counts(trajectory: dict, task: dict) -> tuple[int, int]:
    """Gold ids that are in the draft and were inspected (and crewed if required)."""
    gold_ids = [str(item) for item in ((task.get("gold_watchlist") or {}).get("movie_ids") or [])]
    n_gold = len(gold_ids)
    if n_gold == 0:
        return 0, 0
    terminal = trajectory.get("terminal_result") or {}
    draft = set(str(item) for item in ((terminal.get("watchlist") or {}).get("movie_ids") or []))
    inspected = _ids_from_tools(trajectory, {"open", "compare"})
    need_crew = bool((task.get("constraints") or {}).get("directors"))
    crewed = _ids_from_tools(trajectory, {"view_crew"})
    hits = 0
    for movie_id in gold_ids:
        if movie_id not in draft or movie_id not in inspected:
            continue
        if need_crew and movie_id not in crewed:
            continue
        hits += 1
    return hits, n_gold


def verified_gold_hit_fraction(trajectory: dict, task: dict) -> float:
    hits, n_gold = verified_gold_hit_counts(trajectory, task)
    if n_gold <= 0:
        return 0.0
    return hits / n_gold


def shaped_grpo_reward(trajectory: dict, task: dict) -> float:
    """Stretch Reward v1 toward verified gold hits so G=4 groups are less often constant.

    Eval / SFT filtering still read Reward v1 from the environment. This wrapper
    only feeds GRPO advantages.

    v5: n≥2 uses a verified-hit staircase (open/compare, plus view_crew when the
    task has a director constraint) so "right movie + evidence" beats random
    wrong lists. n=1 keeps the v4 Jaccard / crew shaping.
    """
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    draft = terminal.get("watchlist") or {}
    base = terminal_reward_v1(trajectory)
    reward_type = detail.get("reward_type")
    overlap = gold_overlap(task, draft)
    try:
        n_movies = int((task.get("constraints") or {}).get("n_movies") or 1)
    except (TypeError, ValueError):
        n_movies = 1
    hits, n_gold = verified_gold_hit_counts(trajectory, task)
    frac = hits / n_gold if n_gold else 0.0
    if reward_type == "gold_watchlist":
        shaped = 1.0
    elif n_movies >= 2:
        process = 0.4 * frac
        if reward_type == "valid_alternative":
            shaped = min(0.80, 0.60 + 0.20 * overlap + 0.10 * frac)
        elif hits == 0:
            if reward_type in {"wrong_watchlist", "reward_unverifiable"}:
                shaped = -0.50
            else:
                shaped = float(base)
        elif reward_type == "partial":
            shaped = max(float(base), process + 0.15)
        elif reward_type == "wrong_watchlist":
            shaped = process
        else:
            shaped = max(float(base), process)
    elif reward_type == "valid_alternative":
        shaped = 0.60 + 0.35 * overlap
    elif reward_type == "partial":
        shaped = base + 0.25 * overlap
    elif reward_type == "reward_unverifiable":
        shaped = -0.15
    else:
        shaped = base
    directors = list((task.get("constraints") or {}).get("directors") or [])
    if directors and n_movies < 2:
        used_crew = any(
            step.get("tool_name") == "view_crew" for step in trajectory.get("steps") or []
        )
        shaped += 0.02 if used_crew else -0.02
    return round(float(shaped), 4)
