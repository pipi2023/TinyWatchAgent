#!/usr/bin/env python3
"""Lightweight GRPO: G=4 online rollouts scored by Reward v1, PEFT LoRA updates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinywatch_grpo.environment.catalog import load_catalog
from tinywatch_grpo.training.grpo.loop import group_advantages, select_reward_varying_groups


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("outputs/models/sft-merged"))
    parser.add_argument("--train", type=Path, default=Path("data/grpo/train.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/models/grpo"))
    parser.add_argument("--merged-output", type=Path, default=Path("outputs/models/grpo-merged"))
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-environment-steps", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--merge", action="store_true", help="merge LoRA adapter after training")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--student-base-url",
        default=None,
        help="Unused for in-process GRPO; kept for CLI compatibility with docs/grpo.md",
    )
    parser.add_argument("--student-api-key", default="EMPTY")
    parser.add_argument("--student-model", default="tinywatch-agent")
    return parser.parse_args()


def main():
    args = parse_args()
    catalog = load_catalog(args.catalog)
    tasks = [
        json.loads(line)
        for line in args.train.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    config = {
        "model": str(args.model),
        "n_tasks": len(tasks),
        "n_movies": len(catalog["movies"]),
        "group_size": args.group_size,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "max_environment_steps": args.max_environment_steps,
        "dynamic_sampling": True,
        "mode": "in_process_peft",
    }
    if args.dry_run:
        dummy = [0.0, 1.0, 0.6, 0.6]
        kept, diagnostics = select_reward_varying_groups(["t"] * 4, dummy)
        config["dummy_advantages"] = group_advantages(dummy)
        config["dynamic_sampling_demo"] = diagnostics
        config["kept_indices"] = kept
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return

    from tinywatch_grpo.training.grpo.trainer import merge_grpo_adapter, run_lightweight_grpo

    summary = run_lightweight_grpo(
        model_path=args.model,
        tasks=tasks,
        catalog=catalog,
        output_dir=args.output,
        group_size=args.group_size,
        max_steps=args.max_steps,
        max_environment_steps=args.max_environment_steps,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        dtype_name=args.dtype,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.merge:
        manifest = merge_grpo_adapter(
            base_model=args.model,
            adapter=Path(summary["adapter"]),
            output=args.merged_output,
            bf16=args.dtype == "bf16",
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
