#!/usr/bin/env python3
"""Run N Flash episodes to estimate acceptance rate and token cost before scaling."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tinywatch_grpo.collection.agent_loop import OpenAIChatClient, rollout_task
from tinywatch_grpo.collection.envfile import load_dotenv
from tinywatch_grpo.collection.reward_filter import acceptance_reasons
from tinywatch_grpo.environment.catalog import load_catalog
from tinywatch_grpo.evaluation.artifacts import append_jsonl


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/sft_pool.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flash-calibrate"))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--reward-threshold", type=float, default=0.55)
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY"))
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    if not args.base_url or not args.api_key:
        raise SystemExit("TEACHER_BASE_URL and TEACHER_API_KEY are required")
    catalog = load_catalog(args.catalog)
    tasks = []
    for line in args.tasks.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tasks.append(json.loads(line))
        if len(tasks) >= args.limit:
            break
    client = OpenAIChatClient(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        thinking=args.thinking,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "raw.jsonl"
    accepted = 0
    tokens = 0
    for task in tasks:
        trajectory = rollout_task(
            task, catalog, client, max_steps=args.max_steps, teacher_model=args.model
        )
        append_jsonl(raw_path, [trajectory])
        ok, _ = acceptance_reasons(trajectory, reward_threshold=args.reward_threshold)
        accepted += int(ok)
        tokens += int((trajectory.get("token_usage") or {}).get("total_tokens") or 0)
    report = {
        "n": len(tasks),
        "accepted": accepted,
        "acceptance_rate": accepted / len(tasks) if tasks else 0.0,
        "total_tokens": tokens,
        "mean_tokens": tokens / len(tasks) if tasks else 0.0,
        "model": args.model,
        "thinking": args.thinking,
        "note": "Unit price is billed by AutoDL; multiply mean_tokens by the console rate.",
    }
    (args.output_dir / "calibrate.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["acceptance_rate"] < 0.25:
        raise SystemExit("acceptance rate <25%; fix schema/prompt/Easy mix before scaling")


if __name__ == "__main__":
    main()
