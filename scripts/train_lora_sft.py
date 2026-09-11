#!/usr/bin/env python3
"""Minimal LoRA SFT for accepted TinyWatch tool-calling trajectories."""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

from tinywatch_grpo.training.sft.dataset import load_supervised_examples


DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=list(DEFAULT_TARGET_MODULES))
    parser.add_argument("--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--liger-kernel",
        action="store_true",
        help="Fused LM-head + CE via liger-kernel; avoids materializing full vocab logits",
    )
    parser.add_argument("--attention-implementation", choices=("auto", "sdpa"), default="sdpa")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _loss_only_eval_trainer_class(trainer_base, enable_skip_logits):
    """Skip full-vocab logits during loss-only eval when Liger supports skip_logits."""

    class LossOnlyEvalTrainer(trainer_base):
        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if enable_skip_logits and prediction_loss_only and inputs.get("labels") is not None:
                inputs = dict(inputs)
                inputs["skip_logits"] = True
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only,
                ignore_keys=ignore_keys,
            )

    return LossOnlyEvalTrainer


def main():
    args = parse_args()
    if args.dry_run:
        payload = {
            "model": args.model,
            "train": str(args.train),
            "validation": str(args.validation) if args.validation else None,
            "output": str(args.output),
            "lora_r": args.lora_r,
            "epochs": args.epochs,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    try:
        import torch
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForMultimodalLM,
            AutoProcessor,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit("缺少训练依赖。请在 conda 环境中安装: pip install -e '.[sft]'") from exc
    if args.liger_kernel:
        try:
            import liger_kernel  # noqa: F401
        except ImportError as exc:
            raise SystemExit("--liger-kernel 需要: pip install liger-kernel") from exc

    load_kwargs = {"trust_remote_code": True}
    config = AutoConfig.from_pretrained(args.model, **load_kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(args.model, **load_kwargs)
        tokenizer = processor.tokenizer
        chat_template = processor
        model_class = AutoModelForMultimodalLM
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
        chat_template = tokenizer
        model_class = AutoModelForCausalLM
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_examples, train_stats = load_supervised_examples(
        args.train, tokenizer=tokenizer, chat_template=chat_template, max_length=args.max_length
    )
    print("train_data=", train_stats)
    if not train_examples:
        raise SystemExit("训练集没有可用样本")
    validation_examples = []
    if args.validation:
        validation_examples, validation_stats = load_supervised_examples(
            args.validation, tokenizer=tokenizer, chat_template=chat_template, max_length=args.max_length
        )
        print("validation_data=", validation_stats)

    if args.dtype == "auto":
        dtype_name = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp32"
        if dtype_name != "bf16" and torch.cuda.is_available():
            dtype_name = "fp16"
    else:
        dtype_name = args.dtype
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    model_kwargs = {"dtype": dtype, "trust_remote_code": True}
    if args.attention_implementation != "auto":
        model_kwargs["attn_implementation"] = args.attention_implementation
    model = model_class.from_pretrained(args.model, **model_kwargs)
    if args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=list(args.target_modules),
        ),
    )
    model.print_trainable_parameters()
    args.output.mkdir(parents=True, exist_ok=True)

    class TokenizedDataset(torch.utils.data.Dataset):
        def __len__(self):
            return len(self.rows)

        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, index):
            example = self.rows[index]
            return {
                "input_ids": torch.tensor(example["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(example["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(example["labels"], dtype=torch.long),
            }

    def collate(batch):
        max_length = max(item["input_ids"].size(0) for item in batch)
        pad = tokenizer.pad_token_id
        input_ids = torch.full((len(batch), max_length), pad, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), max_length), dtype=torch.long)
        labels = torch.full((len(batch), max_length), -100, dtype=torch.long)
        for row, item in enumerate(batch):
            length = item["input_ids"].size(0)
            input_ids[row, :length] = item["input_ids"]
            attention_mask[row, :length] = item["attention_mask"]
            labels[row, :length] = item["labels"]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=dtype_name == "bf16",
        fp16=dtype_name == "fp16",
        gradient_checkpointing=args.gradient_checkpointing,
        use_liger_kernel=args.liger_kernel,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch" if validation_examples else "no",
        report_to="none",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        remove_unused_columns=False,
        seed=args.seed,
    )
    trainer_class = _loss_only_eval_trainer_class(
        Trainer,
        enable_skip_logits=args.liger_kernel and is_multimodal,
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=TokenizedDataset(train_examples),
        eval_dataset=TokenizedDataset(validation_examples) if validation_examples else None,
        data_collator=collate,
    )
    started = time.time()
    result = trainer.train()
    trainer.save_model(str(args.output))
    chat_template.save_pretrained(str(args.output))
    summary = {
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "train_loss": result.training_loss,
        "seconds": round(time.time() - started, 1),
    }
    (args.output / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
