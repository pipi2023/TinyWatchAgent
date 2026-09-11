#!/usr/bin/env python3
"""Audit Flash collection artifacts, then copy filtered SFT rows into data/sft/."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from tinywatch_grpo.collection.reward_filter import (
    build_collection_artifacts,
    read_jsonl,
    task_ids_from_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("outputs/flash-collection/raw.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flash-collection"))
    parser.add_argument("--held-out", type=Path, default=Path("data/evaluation/tasks.jsonl"))
    parser.add_argument("--sft-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--reward-threshold", type=float, default=0.55)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="copy train/validation into data/sft")
    parser.add_argument("--min-accepted", type=int, default=320)
    parser.add_argument("--min-accept-rate", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    held_out = task_ids_from_jsonl(args.held_out)
    metadata = build_collection_artifacts(
        raw_path=args.raw,
        output_dir=args.output_dir,
        held_out_task_ids=held_out,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        reward_threshold=args.reward_threshold,
    )
    train_ids = {int(row["task_id"]) for row in read_jsonl(args.output_dir / "train.jsonl")}
    val_ids = {int(row["task_id"]) for row in read_jsonl(args.output_dir / "validation.jsonl")}
    accepted_ids = train_ids | val_ids
    overlap = sorted(accepted_ids & held_out)
    accept_rate = metadata["accepted"] / metadata["total"] if metadata["total"] else 0.0
    audit = {
        "accepted": metadata["accepted"],
        "total": metadata["total"],
        "acceptance_rate": round(accept_rate, 4),
        "train": metadata["train"],
        "validation": metadata["validation"],
        "held_out_excluded": metadata["held_out_excluded"],
        "duplicate_tasks_excluded": metadata["duplicate_tasks_excluded"],
        "eval_overlap_task_ids": overlap,
        "reject_reasons": metadata["reject_reasons"],
        "pass_min_accepted": metadata["accepted"] >= args.min_accepted,
        "pass_min_accept_rate": accept_rate >= args.min_accept_rate,
        "pass_no_eval_overlap": not overlap,
    }
    audit_path = args.output_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not (audit["pass_min_accepted"] and audit["pass_min_accept_rate"] and audit["pass_no_eval_overlap"]):
        raise SystemExit("audit failed; refusing to copy SFT data")
    if args.copy:
        args.sft_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.jsonl", "validation.jsonl"):
            src = args.output_dir / name
            dst = args.sft_dir / name
            shutil.copy2(src, dst)
        readme = args.sft_dir / "README.md"
        readme.write_text(
            "# SFT data (Flash-accepted)\n\n"
            "Copied from `outputs/flash-collection/train.jsonl` and `validation.jsonl` "
            "after collection audit. These rows are action-only sanitized messages "
            "(thinking stripped). Do not train on `raw.jsonl` or full teacher replies.\n",
            encoding="utf-8",
        )
        print(json.dumps({"copied_to": str(args.sft_dir), "train": metadata["train"], "validation": metadata["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
