"""Deterministic four-panel metrics. No LLM Judge on the main path."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from tinywatch_grpo.evaluation.trajectory import NORMALIZED_TRAJECTORY_VERSION
from tinywatch_grpo.environment.tools import REWARD_VERSION


DETERMINISTIC_METRICS_VERSION = "tinywatch-deterministic-metrics-v1"


def _strict_success(normalized: Mapping, reward_detail: Mapping) -> bool:
    terminal = normalized.get("terminal") if isinstance(normalized.get("terminal"), Mapping) else {}
    return (
        reward_detail.get("reward_version") == REWARD_VERSION
        and normalized.get("status") == "done"
        and normalized.get("done") is True
        and terminal.get("done") is True
        and reward_detail.get("reward_type") == "gold_watchlist"
        and reward_detail.get("reward_valid") is True
        and reward_detail.get("watchlist_success") is True
    )


def compute_coded_rubric(task: Mapping | None, reward_detail: Mapping) -> dict:
    constraints = (task or {}).get("constraints") if isinstance(task, Mapping) else {}
    gates = reward_detail.get("hard_gates") if isinstance(reward_detail.get("hard_gates"), Mapping) else {}
    items = [
        {"id": "n_movies", "hard": True, "passed": bool(gates.get("n_ok"))},
        {"id": "runtime", "hard": True, "passed": bool(gates.get("runtime_ok"))},
        {"id": "rating", "hard": True, "passed": bool(gates.get("rating_ok"))},
        {"id": "year", "hard": True, "passed": bool(gates.get("year_ok"))},
        {"id": "genres", "hard": True, "passed": bool(gates.get("genre_ok"))},
        {"id": "directors", "hard": True, "passed": bool(gates.get("director_ok"))},
        {"id": "zh", "hard": True, "passed": bool(gates.get("zh_ok"))},
    ]
    hard_total = len(items)
    hard_passed = sum(1 for item in items if item["passed"])
    return {
        "schema_version": "tinywatch-coded-rubric-v1",
        "n_movies": (constraints or {}).get("n_movies"),
        "items": items,
        "hard_pass_rate": hard_passed / hard_total if hard_total else 0.0,
        "all_hard_passed": bool(gates.get("all_hard")),
    }


def compute_deterministic_metrics(normalized: object, task: Mapping | None = None) -> dict:
    if not isinstance(normalized, Mapping):
        raise TypeError("normalized trajectory must be an object")
    if normalized.get("schema_version") != NORMALIZED_TRAJECTORY_VERSION:
        raise ValueError("unsupported normalized trajectory schema")
    events = [event for event in (normalized.get("events") or []) if isinstance(event, Mapping)]
    executed = [event for event in events if event.get("event_type") == "tool_step"]
    guards = [event for event in events if event.get("event_type") == "guard_rejection"]
    tool_counts = Counter(str(event.get("tool_name") or "unknown") for event in executed)
    signatures = [
        f"{event.get('tool_name')}:{event.get('parameters')}" for event in events
    ]
    consecutive = sum(
        1 for left, right in zip(signatures, signatures[1:]) if left == right
    )
    terminal = normalized.get("terminal") if isinstance(normalized.get("terminal"), Mapping) else {}
    reward_detail = terminal.get("reward_detail") if isinstance(terminal.get("reward_detail"), Mapping) else {}
    infra_error = bool(normalized.get("error"))
    return {
        "schema_version": DETERMINISTIC_METRICS_VERSION,
        "reward_and_outcome": {
            "reward": float(terminal.get("reward") or 0.0),
            "reward_type": reward_detail.get("reward_type"),
            "reward_valid": reward_detail.get("reward_valid"),
            "strict_gold_success": _strict_success(normalized, reward_detail),
            "termination_reason": reward_detail.get("termination_reason"),
        },
        "rubric": compute_coded_rubric(task, reward_detail),
        "behavior": {
            "executed_tool_steps": len(executed),
            "guard_rejections": len(guards),
            "tool_counts": dict(tool_counts),
            "finalize_used": tool_counts.get("finalize_watchlist", 0) > 0,
            "abort_used": tool_counts.get("abort", 0) > 0,
            "search_count": tool_counts.get("search_movies", 0),
            "open_or_compare": tool_counts.get("open", 0) + tool_counts.get("compare", 0),
            "consecutive_duplicate_actions": consecutive,
        },
        "infra": {
            "invalid_trajectory": infra_error,
            "status": normalized.get("status"),
            "error": normalized.get("error"),
        },
    }


def aggregate_run(rows: list[dict]) -> dict:
    success = sum(1 for row in rows if row["deterministic_metrics"]["reward_and_outcome"]["strict_gold_success"])
    reward_types = Counter(
        str(row["deterministic_metrics"]["reward_and_outcome"]["reward_type"]) for row in rows
    )
    executed = sum(row["deterministic_metrics"]["behavior"]["executed_tool_steps"] for row in rows)
    guards = sum(row["deterministic_metrics"]["behavior"]["guard_rejections"] for row in rows)
    rubric_all = sum(1 for row in rows if row["deterministic_metrics"]["rubric"]["all_hard_passed"])
    infra = sum(1 for row in rows if row["deterministic_metrics"]["infra"]["invalid_trajectory"])
    n = len(rows)
    return {
        "schema_version": "tinywatch-eval-summary-v1",
        "trajectory_count": n,
        "panels": {
            "reward": {
                "strict_gold_success_count": success,
                "strict_gold_success_rate": success / n if n else 0.0,
                "reward_type_counts": dict(sorted(reward_types.items())),
            },
            "rubric": {"all_hard_passed_count": rubric_all, "all_hard_passed_rate": rubric_all / n if n else 0.0},
            "behavior": {
                "mean_executed_tool_steps": executed / n if n else 0.0,
                "guard_rejections": guards,
            },
            "infra": {"invalid_count": infra, "invalid_rate": infra / n if n else 0.0},
        },
    }
