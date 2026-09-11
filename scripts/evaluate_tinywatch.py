#!/usr/bin/env python3
"""Roll out a student on Final-100 and write four-panel deterministic reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tinywatch_grpo.collection.agent_loop import OpenAIChatClient, rollout_task
from tinywatch_grpo.environment.catalog import load_catalog
from tinywatch_grpo.evaluation.artifacts import write_json_atomic, write_jsonl_atomic
from tinywatch_grpo.evaluation.metrics import aggregate_run, compute_deterministic_metrics
from tinywatch_grpo.evaluation.trajectory import normalize_trajectory


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=Path("data/evaluation/tasks.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("SERVED_MODEL_NAME", "tinywatch-agent"))
    parser.add_argument("--llm-base-url", default=os.environ.get("LLM_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", "EMPTY"))
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.llm_base_url:
        raise SystemExit("LLM_BASE_URL / --llm-base-url is required to evaluate a student")
    catalog = load_catalog(args.catalog)
    tasks = [
        json.loads(line)
        for line in args.benchmark.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit is not None:
        tasks = tasks[: args.limit]
    client = OpenAIChatClient(model=args.model, base_url=args.llm_base_url, api_key=args.api_key)
    rows = []
    raw = []
    for task in tasks:
        trajectory = rollout_task(task, catalog, client, max_steps=args.max_steps, teacher_model=args.model)
        raw.append(trajectory)
        normalized = normalize_trajectory(trajectory)
        metrics = compute_deterministic_metrics(normalized, task=task)
        rows.append(
            {
                "task_id": task["task_id"],
                "trajectory_id": trajectory.get("trajectory_id"),
                "normalized_trajectory": normalized,
                "deterministic_metrics": metrics,
            }
        )
        print(
            f"task={task['task_id']} gold={metrics['reward_and_outcome']['strict_gold_success']} "
            f"type={metrics['reward_and_outcome']['reward_type']}"
        )
    summary = aggregate_run(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(args.output_dir / "trajectories.jsonl", raw, force=args.force)
    write_jsonl_atomic(args.output_dir / "scored.jsonl", rows, force=args.force)
    write_json_atomic(args.output_dir / "summary.json", summary, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
