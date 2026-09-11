#!/usr/bin/env python3
"""Pair Baseline / SFT / GRPO summaries on the same Final-100 task_ids."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--grpo", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/comparison.md"))
    args = parser.parse_args()
    runs = {
        "baseline": load_summary(args.baseline),
        "sft": load_summary(args.sft),
        "grpo": load_summary(args.grpo),
    }
    lines = ["# TinyWatch Final-100 comparison", "", "| model | gold@100 | hard rubric | mean steps | infra invalid |", "|---|---:|---:|---:|---:|"]
    for name, summary in runs.items():
        panels = summary["panels"]
        lines.append(
            f"| {name} | {panels['reward']['strict_gold_success_rate']:.3f} | "
            f"{panels['rubric']['all_hard_passed_rate']:.3f} | "
            f"{panels['behavior']['mean_executed_tool_steps']:.2f} | "
            f"{panels['infra']['invalid_rate']:.3f} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
