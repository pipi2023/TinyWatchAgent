from tinywatch_grpo.evaluation.metrics import compute_deterministic_metrics
from tinywatch_grpo.evaluation.trajectory import normalize_trajectory


def test_four_panels_and_strict_gold():
    raw = {
        "trajectory_id": "t1",
        "task_id": 1,
        "status": "done",
        "done": True,
        "messages": [{"role": "user", "content": "选一部科幻片"}],
        "steps": [
            {"tool_name": "search_movies", "parameters": {"query": "科幻"}, "observation": "1|tt1|x", "done": False},
            {"tool_name": "finalize_watchlist", "parameters": {}, "observation": "end", "done": True},
        ],
        "terminal_result": {
            "done": True,
            "over": True,
            "reward": 1.0,
            "reward_detail": {
                "reward_version": "tinywatch-reward-v1",
                "reward_type": "gold_watchlist",
                "reward_valid": True,
                "watchlist_success": True,
                "termination_reason": "gold_watchlist",
                "hard_gates": {
                    "n_ok": True,
                    "runtime_ok": True,
                    "rating_ok": True,
                    "year_ok": True,
                    "genre_ok": True,
                    "director_ok": True,
                    "zh_ok": True,
                    "has_movies": True,
                    "all_hard": True,
                },
            },
            "watchlist": {"movie_ids": ["tt1"]},
        },
    }
    normalized = normalize_trajectory(raw)
    metrics = compute_deterministic_metrics(normalized, task={"constraints": {"n_movies": 1}})
    assert metrics["reward_and_outcome"]["strict_gold_success"] is True
    assert "rubric" in metrics and "behavior" in metrics and "infra" in metrics
    assert metrics["rubric"]["all_hard_passed"] is True
