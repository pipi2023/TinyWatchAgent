"""CPU/offline utilities: smoke and deterministic evaluation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinywatch_grpo.evaluation.artifacts import iter_jsonl, write_jsonl_atomic
from tinywatch_grpo.evaluation.metrics import aggregate_run, compute_deterministic_metrics
from tinywatch_grpo.evaluation.trajectory import normalize_trajectory
from tinywatch_grpo.smoke import run_cpu_smoke


def _offline_evaluate(args: argparse.Namespace) -> None:
    rows = []
    for raw in iter_jsonl(args.trajectories):
        normalized = normalize_trajectory(raw)
        metrics = compute_deterministic_metrics(normalized)
        rows.append(
            {
                "task_id": normalized["task_id"],
                "trajectory_id": normalized["trajectory_id"],
                "normalized_trajectory": normalized,
                "deterministic_metrics": metrics,
            }
        )
    summary = aggregate_run(rows)
    if args.output:
        write_jsonl_atomic(args.output, rows, force=args.force)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def _smoke(args: argparse.Namespace) -> None:
    result = run_cpu_smoke()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    for check in result["checks"]:
        print(f"[ok] {check}")
    print(f"CPU smoke passed: {len(result['checks'])} checks")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tinywatch-grpo", description="TinyWatch GRPO CPU/offline utilities")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("smoke", help="run pure-CPU contract checks")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_smoke)
    command = commands.add_parser("evaluate", help="summarize saved rollouts with four deterministic panels")
    command.add_argument("trajectories", type=Path)
    command.add_argument("--output", type=Path)
    command.add_argument("--force", action="store_true")
    command.set_defaults(handler=_offline_evaluate)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
