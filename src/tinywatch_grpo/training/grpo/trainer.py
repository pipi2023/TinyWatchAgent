"""In-process lightweight GRPO with PEFT LoRA updates."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor

from tinywatch_grpo.collection.agent_loop import rollout_task
from tinywatch_grpo.collection.local_client import TransformersToolClient
from tinywatch_grpo.environment.tools import TINYWATCH_TOOL_SCHEMAS
from tinywatch_grpo.training.grpo.loop import group_advantages, select_reward_varying_groups
from tinywatch_grpo.training.sft.dataset import IGNORE_INDEX, build_supervised_example


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


def load_policy_model(
    model_path: str | Path,
    *,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules=None,
    dtype_name: str = "bf16",
):
    model_path = str(model_path)
    load_kwargs = {"trust_remote_code": True}
    config = AutoConfig.from_pretrained(model_path, **load_kwargs)
    is_multimodal = str(getattr(config, "model_type", "")).startswith("qwen3_5")
    processor = AutoProcessor.from_pretrained(model_path, **load_kwargs)
    tokenizer = getattr(processor, "tokenizer", processor)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[dtype_name]
    model_class = AutoModelForMultimodalLM if is_multimodal else AutoModelForCausalLM
    model = model_class.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        **load_kwargs,
    )
    if torch.cuda.is_available():
        model = model.cuda()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=list(target_modules or DEFAULT_TARGET_MODULES),
        ),
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.train()
    return model, processor, tokenizer


def trajectory_reward(trajectory: dict) -> float:
    terminal = trajectory.get("terminal_result") or {}
    detail = terminal.get("reward_detail") or {}
    if detail.get("reward_valid") is not True:
        return 0.0
    try:
        return float(detail.get("reward", trajectory.get("final_reward") or 0.0))
    except (TypeError, ValueError):
        return float(trajectory.get("final_reward") or 0.0)


def sequence_logprob(model, input_ids, labels, attention_mask=None) -> torch.Tensor:
    """Mean token log-prob over supervised (non-IGNORE) positions."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = torch.log_softmax(shift_logits.float(), dim=-1)
    token_logprob = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(IGNORE_INDEX)
    if not mask.any():
        return input_ids.new_zeros(()).float()
    return (token_logprob * mask).sum() / mask.sum().clamp(min=1)


def prepare_example_tensors(example: dict, device) -> dict:
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([example["attention_mask"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def run_lightweight_grpo(
    *,
    model_path: str | Path,
    tasks: list[dict],
    catalog: dict,
    output_dir: str | Path,
    group_size: int = 4,
    max_steps: int = 100,
    max_environment_steps: int = 14,
    learning_rate: float = 1e-5,
    lora_r: int = 16,
    lora_alpha: int = 32,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
    max_length: int = 4096,
    dtype_name: str = "bf16",
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not tasks:
        raise ValueError("GRPO train set is empty")

    model, processor, tokenizer = load_policy_model(
        model_path,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        dtype_name=dtype_name,
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=learning_rate,
    )
    client = TransformersToolClient(
        model,
        processor,
        temperature=temperature,
        top_p=0.9,
        max_new_tokens=max_new_tokens,
        enable_thinking=False,
    )
    device = next(model.parameters()).device
    history = []
    started = time.time()
    optimizer_steps = 0
    skipped_groups = 0

    for step in range(max_steps):
        task = tasks[step % len(tasks)]
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trajectories = []
        rewards = []
        for _ in range(group_size):
            trajectory = rollout_task(
                task,
                catalog,
                client,
                max_steps=max_environment_steps,
                teacher_model=str(model_path),
            )
            trajectories.append(trajectory)
            rewards.append(trajectory_reward(trajectory))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        group_ids = [task["task_id"]] * group_size
        kept, diagnostics = select_reward_varying_groups(group_ids, rewards)
        advantages = group_advantages(rewards)
        record = {
            "step": step,
            "task_id": task["task_id"],
            "rewards": rewards,
            "advantages": advantages,
            "mean_reward": sum(rewards) / len(rewards),
            "kept_indices": kept,
            "dynamic_sampling": diagnostics,
        }

        if not kept:
            skipped_groups += 1
            record["optimizer_step"] = False
            history.append(record)
            print(
                f"[grpo {step + 1}/{max_steps}] task={task['task_id']} "
                f"mean_r={record['mean_reward']:.3f} skipped_constant_group",
                flush=True,
            )
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        trainable_kept = 0
        for index in kept:
            trajectory = trajectories[index]
            advantage = float(advantages[index])
            if abs(advantage) < 1e-8:
                continue
            example = build_supervised_example(
                messages=trajectory.get("messages") or [],
                tools=trajectory.get("tools") or TINYWATCH_TOOL_SCHEMAS,
                tokenizer=tokenizer,
                max_length=max_length,
                chat_template=processor,
            )
            if example is None:
                continue
            tensors = prepare_example_tensors(example, device)
            try:
                logprob = sequence_logprob(model, **tensors)
                loss = -(advantage * logprob)
                loss.backward()
                losses.append(float(loss.detach().cpu()))
                trainable_kept += 1
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            finally:
                del tensors
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not trainable_kept:
            skipped_groups += 1
            record["optimizer_step"] = False
            history.append(record)
            print(
                f"[grpo {step + 1}/{max_steps}] task={task['task_id']} "
                f"mean_r={record['mean_reward']:.3f} no_trainable_examples",
                flush=True,
            )
            continue

        torch.nn.utils.clip_grad_norm_(
            [param for param in model.parameters() if param.requires_grad],
            1.0,
        )
        optimizer.step()
        optimizer_steps += 1
        record["optimizer_step"] = True
        record["loss"] = sum(losses) / len(losses)
        history.append(record)
        print(
            f"[grpo {step + 1}/{max_steps}] task={task['task_id']} "
            f"mean_r={record['mean_reward']:.3f} loss={record['loss']:.4f}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    summary = {
        "steps": len(history),
        "optimizer_steps": optimizer_steps,
        "skipped_constant_groups": skipped_groups,
        "mean_reward": (
            sum(item["mean_reward"] for item in history) / len(history) if history else 0.0
        ),
        "seconds": round(time.time() - started, 1),
        "adapter": str(adapter_dir),
        "config": {
            "model": str(model_path),
            "group_size": group_size,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "max_environment_steps": max_environment_steps,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "dynamic_sampling": True,
        },
    }
    (output_dir / "history.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in history) + ("\n" if history else ""),
        encoding="utf-8",
    )
    (output_dir / "grpo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def merge_grpo_adapter(
    *,
    base_model: str | Path,
    adapter: str | Path,
    output: str | Path,
    bf16: bool = True,
) -> dict:
    """Merge a GRPO LoRA adapter into a standalone checkpoint."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录：{output}")
    dtype = torch.bfloat16 if bf16 else torch.float32
    config = AutoConfig.from_pretrained(str(base_model), trust_remote_code=True)
    model_class = (
        AutoModelForMultimodalLM
        if str(getattr(config, "model_type", "")).startswith("qwen3_5")
        else AutoModelForCausalLM
    )
    base = model_class.from_pretrained(
        str(base_model), torch_dtype=dtype, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output), safe_serialization=True)
    AutoProcessor.from_pretrained(str(base_model), trust_remote_code=True).save_pretrained(
        str(output)
    )
    manifest = {
        "operation": "peft_merge_and_unload",
        "source": {"base_model": str(base_model), "adapter": str(adapter)},
        "output": str(output),
    }
    (output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
