#!/usr/bin/env python3
"""Oracle smoke / emergency SFT trajectories. CPU only, no teacher API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinywatch_grpo.collection.reward_filter import build_collection_artifacts, write_jsonl
from tinywatch_grpo.environment.catalog import load_catalog
from tinywatch_grpo.environment.env import TinyWatchEnv
from tinywatch_grpo.generation.oracle import run_oracle


def load_tasks(path: Path, limit: int | None) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/sft_pool.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle-collection"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=14)
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    tasks = load_tasks(args.tasks, args.limit)
    raw = []
    gold = 0
    for task in tasks:
        env = TinyWatchEnv(catalog, task, max_steps=args.max_steps)
        trajectory = run_oracle(env)
        raw.append(trajectory)
        detail = (trajectory.get("terminal_result") or {}).get("reward_detail") or {}
        if detail.get("reward_type") == "gold_watchlist" and detail.get("reward_valid"):
            gold += 1
        print(
            f"task={task['task_id']} solvable={task.get('solvable')} "
            f"type={detail.get('reward_type')} reward={trajectory.get('final_reward')}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    write_jsonl(raw_path, raw)
    summary = {
        "count": len(raw),
        "gold_watchlist": gold,
        "all_gold_when_solvable": gold == sum(1 for task in tasks if task.get("solvable")),
    }
    (args.output_dir / "oracle_smoke.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    build_collection_artifacts(
        raw_path=raw_path,
        output_dir=args.output_dir,
        held_out_task_ids=(),
        validation_ratio=0.2,
        collection_config={"teacher": "oracle", "limit": args.limit},
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.limit <= 50 and gold < sum(1 for task in tasks if task.get("solvable")):
        raise SystemExit("Oracle smoke failed: solvable tasks did not all reach gold_watchlist")


if __name__ == "__main__":
    main()
