#!/usr/bin/env python3
"""Generate disjoint SFT pool / GRPO train / Eval-100 task files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinywatch_grpo.environment.catalog import load_catalog
from tinywatch_grpo.generation.task_gen import assert_disjoint, generate_splits, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft-pool", type=int, default=1000)
    parser.add_argument("--grpo-train", type=int, default=400)
    parser.add_argument("--eval-holdout", type=int, default=100)
    parser.add_argument("--sft-out", type=Path, default=Path("data/tasks/sft_pool.jsonl"))
    parser.add_argument("--grpo-out", type=Path, default=Path("data/grpo/train.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/evaluation/tasks.jsonl"))
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    splits = generate_splits(
        catalog,
        seed=args.seed,
        sft_pool=args.sft_pool,
        grpo_train=args.grpo_train,
        eval_holdout=args.eval_holdout,
    )
    assert_disjoint(splits)
    metadata = {
        "sft_pool": write_jsonl(args.sft_out, splits["sft_pool"]),
        "grpo_train": write_jsonl(args.grpo_out, splits["grpo_train"]),
        "evaluation": write_jsonl(args.eval_out, splits["evaluation"]),
        "seed": args.seed,
        "disjoint": True,
        "gold_recipe": "identifiable-v2",
        "notes": "Easy/hard tasks expose director + year so gold is recoverable from the public query.",
    }
    Path("data/tasks/metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path("data/evaluation/metadata.json").write_text(
        json.dumps({"evaluation": metadata["evaluation"], "seed": args.seed}, indent=2) + "\n",
        encoding="utf-8",
    )
    Path("data/grpo/metadata.json").write_text(
        json.dumps({"grpo_train": metadata["grpo_train"], "seed": args.seed}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
