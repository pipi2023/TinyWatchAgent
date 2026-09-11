#!/usr/bin/env python3
"""Merge LoRA adapter into a standalone checkpoint for GRPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = {
        "operation": "peft_merge_and_unload",
        "source": {"base_model": args.base_model, "adapter": str(args.adapter)},
        "output": str(args.output),
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录：{args.output}")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor
    except ImportError as exc:
        raise SystemExit("缺少合并依赖；pip install -e '.[sft]'") from exc
    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    model_class = (
        AutoModelForMultimodalLM
        if str(getattr(config, "model_type", "")).startswith("qwen3_5")
        else AutoModelForCausalLM
    )
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    base = model_class.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True)
    AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(str(args.output))
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
