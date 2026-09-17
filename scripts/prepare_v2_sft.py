#!/usr/bin/env python3
"""Collect Oracle gold trajectories on the v2 SFT pool and stage data/sft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinywatch_grpo.collection.reward_filter import (
    build_collection_artifacts,
    read_jsonl,
    task_ids_from_jsonl,
    write_jsonl,
)
from tinywatch_grpo.environment.catalog import load_catalog
from tinywatch_grpo.environment.env import TinyWatchEnv
from tinywatch_grpo.generation.oracle import run_oracle


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/sft_pool.jsonl"))
    parser.add_argument("--held-out", type=Path, default=Path("data/evaluation/tasks.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/oracle-v2-collection"))
    parser.add_argument("--sft-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--reward-threshold", type=float, default=0.49)
    parser.add_argument("--copy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_catalog(args.catalog)
    tasks = [json.loads(line) for line in args.tasks.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    raw = []
    gold = 0
    abort = 0
    for index, task in enumerate(tasks, start=1):
        env = TinyWatchEnv(catalog, task, max_steps=args.max_steps)
        trajectory = run_oracle(env)
        detail = (trajectory.get("terminal_result") or {}).get("reward_detail") or {}
        reward_type = detail.get("reward_type")
        if reward_type == "gold_watchlist" and detail.get("reward_valid"):
            gold += 1
        if reward_type == "correct_abort":
            abort += 1
        raw.append(trajectory)
        if index % 50 == 0 or index == len(tasks):
            print(
                f"[oracle-v2] {index}/{len(tasks)} gold={gold} abort={abort} "
                f"last={task['task_id']}:{reward_type}",
                flush=True,
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    write_jsonl(raw_path, raw)
    held_out = task_ids_from_jsonl(args.held_out)
    metadata = build_collection_artifacts(
        raw_path=raw_path,
        output_dir=args.output_dir,
        held_out_task_ids=held_out,
        validation_ratio=0.2,
        seed=42,
        reward_threshold=args.reward_threshold,
        collection_config={"teacher": "oracle", "gold_recipe": "identifiable-v2", "limit": args.limit},
    )
    summary = {
        "collected": len(raw),
        "gold_watchlist": gold,
        "correct_abort": abort,
        "accepted": metadata["accepted"],
        "acceptance_rate": round(metadata["accepted"] / metadata["total"], 4) if metadata["total"] else 0.0,
        "eval_overlap": sorted({int(row["task_id"]) for row in read_jsonl(args.output_dir / "train.jsonl")} & held_out),
    }
    (args.output_dir / "oracle_v2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["eval_overlap"]:
        raise SystemExit("oracle SFT overlaps Final-100")
    min_accepted = 200 if args.limit is None else max(1, int(0.5 * args.limit))
    if metadata["accepted"] < min_accepted:
        raise SystemExit("too few accepted oracle trajectories")
    if args.copy:
        args.sft_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl"):
            src = args.output_dir / name
            dst = args.sft_dir / name
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        (args.sft_dir / "README.md").write_text(
            "# SFT data (v2 Oracle, identifiable gold)\n\n"
            "Staged from `outputs/oracle-v2-collection/` after Reward v1 filtering. "
            "V1 Flash SFT lives in `data/archive/v1-20260911/sft/`.\n",
            encoding="utf-8",
        )
        print(json.dumps({"copied_to": str(args.sft_dir), "train": metadata["train"], "validation": metadata["validation"]}))


if __name__ == "__main__":
    main()
