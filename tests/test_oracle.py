from tinywatch_grpo.environment.env import TinyWatchEnv
from tinywatch_grpo.generation.oracle import run_oracle
from tinywatch_grpo.generation.task_gen import generate_splits
from tinywatch_grpo.generation.vocab import mini_catalog


def test_oracle_ten_solvable_tasks():
    catalog = mini_catalog()
    splits = generate_splits(catalog, seed=42, sft_pool=30, grpo_train=10, eval_holdout=10)
    solvable = [task for task in splits["sft_pool"] if task.get("solvable")][:10]
    assert len(solvable) == 10
    for task in solvable:
        env = TinyWatchEnv(catalog, task, max_steps=14)
        trajectory = run_oracle(env)
        detail = (trajectory.get("terminal_result") or {}).get("reward_detail") or {}
        assert detail.get("reward_type") == "gold_watchlist", (task["task_id"], detail)
        assert detail.get("reward_valid") is True
        assert detail.get("reward") == 1.0
