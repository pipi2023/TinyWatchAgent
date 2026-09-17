"""GRPO helpers: advantages, dynamic sampling, PPO clip, collapsed-rollout filter."""

from __future__ import annotations

import math
import random
import re


FORMAT_COLLAPSE_ERRORS = {
    "no_tool_call",
    "invalid_tool_arguments",
    "invalid_tool_arguments_json",
    "tool_call_arguments_invalid",
    "tool_call_arguments_invalid_json",
    "too_many_guard_rejections",
}

IMDB_MOVIE_ID_RE = re.compile(r"^tt\d{7,}$")


def group_advantages(rewards: list[float], *, eps: float = 1e-8) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    variance = sum((item - mean) ** 2 for item in rewards) / len(rewards)
    std = variance ** 0.5
    if std < eps:
        return [0.0 for _ in rewards]
    return [(item - mean) / (std + eps) for item in rewards]


def select_reward_varying_groups(group_ids: list, rewards: list[float]) -> tuple[list[int], dict]:
    buckets: dict = {}
    for index, (group_id, reward) in enumerate(zip(group_ids, rewards)):
        buckets.setdefault(group_id, []).append((index, float(reward)))
    kept = []
    skipped = 0
    for items in buckets.values():
        values = {reward for _, reward in items}
        if len(values) <= 1:
            skipped += 1
            continue
        kept.extend(index for index, _ in items)
    kept.sort()
    return kept, {"kept_group_count": len(buckets) - skipped, "skipped_constant_groups": skipped}


def select_trainable_varying_indices(
    rewards: list[float],
    trainable_mask: list[bool],
    *,
    eps: float = 1e-8,
) -> tuple[list[int], dict]:
    """Keep trainable rollouts whose rewards still vary (drop format-collapsed rows first)."""
    trainable = [index for index, ok in enumerate(trainable_mask) if ok]
    diagnostics = {
        "trainable": len(trainable),
        "dropped_collapsed": sum(1 for ok in trainable_mask if not ok),
        "kept_group_count": 0,
        "skipped_constant_groups": 0,
    }
    if len(trainable) < 2:
        diagnostics["skipped_constant_groups"] = 1
        return [], diagnostics
    values = {round(float(rewards[index]), 6) for index in trainable}
    if len(values) <= 1:
        diagnostics["skipped_constant_groups"] = 1
        return [], diagnostics
    diagnostics["kept_group_count"] = 1
    diagnostics["advantages"] = group_advantages(
        [float(rewards[index]) for index in trainable], eps=eps
    )
    return trainable, diagnostics


def _step_movie_ids(step: dict) -> list[str]:
    params = step.get("parameters") or {}
    ids = []
    if params.get("movie_id") is not None:
        ids.append(str(params.get("movie_id")))
    movie_ids = params.get("movie_ids")
    if isinstance(movie_ids, list):
        ids.extend(str(item) for item in movie_ids if item is not None)
    return ids


def has_illegal_movie_id(trajectory: dict) -> bool:
    """True when a tool used a non-IMDb movie_id (e.g. 1000000001)."""
    for step in trajectory.get("steps") or []:
        if not isinstance(step, dict):
            continue
        for movie_id in _step_movie_ids(step):
            if movie_id and not IMDB_MOVIE_ID_RE.match(movie_id):
                return True
    return False


def is_collapsed_rollout(trajectory: dict) -> bool:
    """True when the trajectory leaked the chat template, never produced a tool call, or used illegal ids."""
    error = trajectory.get("error")
    if error in FORMAT_COLLAPSE_ERRORS:
        return True
    if has_illegal_movie_id(trajectory):
        return True
    messages = trajectory.get("messages") or []
    assistants = [
        message
        for message in messages
        if isinstance(message, dict) and message.get("role") == "assistant"
    ]
    if not assistants:
        return True
    last = assistants[-1]
    content = last.get("content") or ""
    if "<tool_response>" in content:
        return True
    stripped = content.strip()
    if stripped.startswith("user\n"):
        return True
    if content.count("</function>") >= 3:
        return True
    return False


def ppo_clip_loss(
    new_logprob: float,
    old_logprob: float,
    advantage: float,
    clip_ratio: float = 0.2,
) -> float:
    """Sequence-level PPO clip; used by tests. Trainer applies the torch equivalent."""
    ratio = math.exp(new_logprob - old_logprob)
    unclipped = ratio * advantage
    clipped_ratio = min(max(ratio, 1.0 - clip_ratio), 1.0 + clip_ratio)
    clipped = clipped_ratio * advantage
    return -min(unclipped, clipped)


def grpo_policy_loss(logprobs: list[float], advantages: list[float]) -> float:
    if len(logprobs) != len(advantages) or not logprobs:
        raise ValueError("logprobs and advantages must be aligned and non-empty")
    return -sum(lp * adv for lp, adv in zip(logprobs, advantages)) / len(logprobs)


def task_n_movies(task: dict) -> int:
    try:
        return int((task.get("constraints") or {}).get("n_movies") or 1)
    except (TypeError, ValueError):
        return 1


def scale_advantage(advantage: float, *, neg_coef: float = 1.0) -> float:
    """Down-weight negative advantages so long failures do not dominate PPO."""
    if advantage < 0.0 and neg_coef != 1.0:
        return float(advantage) * float(neg_coef)
    return float(advantage)


def n2_policy_rollout_count(group_size: int, n_movies: int, *, oracle_mix: bool) -> int:
    """How many on-policy rollouts to sample. n≥2 groups keep one slot for Oracle."""
    size = max(1, int(group_size))
    if oracle_mix and int(n_movies) >= 2 and size >= 2:
        return size - 1
    return size


def build_task_schedule(
    tasks: list[dict],
    max_steps: int,
    *,
    mode: str = "director_first",
    n2_oversample: float = 1.0,
    seed: int = 42,
) -> list[dict]:
    """Build a length-`max_steps` task list.

    - ``director_first``: v3 order (directors first, then task_id), cycled.
    - ``shuffled_n2``: shuffle with n≥2 tasks oversampled, reshuffle each epoch.
    """
    if not tasks:
        raise ValueError("GRPO train set is empty")
    if max_steps <= 0:
        return []
    if mode == "director_first":
        ordered = sorted(
            tasks,
            key=lambda item: (
                0 if (item.get("constraints") or {}).get("directors") else 1,
                int(item.get("task_id") or 0),
            ),
        )
        return [ordered[step % len(ordered)] for step in range(max_steps)]

    if mode != "shuffled_n2":
        raise ValueError(f"unknown task schedule mode: {mode}")

    rng = random.Random(int(seed))
    single = [task for task in tasks if task_n_movies(task) < 2]
    multi = [task for task in tasks if task_n_movies(task) >= 2]
    pool: list[dict] = list(single)
    oversample = max(1.0, float(n2_oversample))
    if multi:
        pool.extend(multi)
        extra = int(round(len(multi) * (oversample - 1.0)))
        if extra > 0:
            pool.extend(rng.choices(multi, k=extra))
    if not pool:
        pool = list(tasks)
    rng.shuffle(pool)
    schedule: list[dict] = []
    index = 0
    while len(schedule) < max_steps:
        if index > 0 and index % len(pool) == 0:
            rng.shuffle(pool)
        schedule.append(pool[index % len(pool)])
        index += 1
    return schedule
