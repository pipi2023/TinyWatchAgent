"""In-process lightweight GRPO with PEFT LoRA updates."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForMultimodalLM, AutoProcessor

from tinywatch_grpo.collection.agent_loop import rollout_task
from tinywatch_grpo.collection.local_client import TransformersToolClient
from tinywatch_grpo.environment.tools import TINYWATCH_TOOL_SCHEMAS
from tinywatch_grpo.generation.oracle import oracle_rollout
from tinywatch_grpo.training.grpo.loop import (
    build_task_schedule,
    group_advantages,
    is_collapsed_rollout,
    n2_policy_rollout_count,
    scale_advantage,
    select_trainable_varying_indices,
    task_n_movies,
)
from tinywatch_grpo.training.grpo.reward_shaping import (
    shaped_grpo_reward,
    terminal_reward_v1,
    verified_gold_hit_fraction,
)
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


def trajectory_reward(trajectory: dict, task: dict | None = None, *, shaped: bool = True) -> float:
    if shaped:
        return shaped_grpo_reward(trajectory, task or {})
    return terminal_reward_v1(trajectory)


def sequence_logprob(model, input_ids, labels, attention_mask=None, reduction="sum") -> torch.Tensor:
    """Token log-prob over supervised (non-IGNORE) positions.

    v3 used sum (raw sequence log-prob). v4 defaults to mean so long tool
    trajectories do not explode PPO ratios. Do not pair mean with unclipped
    REINFORCE (that collapsed v2).
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    log_probs = torch.log_softmax(logits, dim=-1)
    token_logprob = log_probs.gather(-1, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = shift_labels.ne(IGNORE_INDEX)
    zero = logits.new_zeros(()).float()
    if not mask.any():
        return zero
    summed = (token_logprob.float() * mask).sum()
    if reduction == "mean":
        return summed / mask.sum().clamp(min=1)
    return summed


def prepare_example_tensors(example: dict, device) -> dict:
    input_ids = torch.tensor([example["input_ids"]], dtype=torch.long, device=device)
    attention_mask = torch.tensor([example["attention_mask"]], dtype=torch.long, device=device)
    labels = torch.tensor([example["labels"]], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _window_rates(history: list[dict], start: int, end: int) -> dict:
    chunk = history[start:end]
    policy_types = []
    n1_types = []
    n2_types = []
    n2_verified = []
    n2_null = 0
    for row in chunk:
        n_movies = int(row.get("n_movies") or 1)
        flags = row.get("oracle_flags") or [False] * len(row.get("reward_types") or [])
        fracs = row.get("verified_hit_fracs") or []
        types = row.get("reward_types") or []
        for index, reward_type in enumerate(types):
            if index < len(flags) and flags[index]:
                continue
            policy_types.append(reward_type)
            if n_movies >= 2:
                n2_types.append(reward_type)
                if index < len(fracs):
                    n2_verified.append(float(fracs[index] or 0.0))
                if reward_type is None:
                    n2_null += 1
            else:
                n1_types.append(reward_type)
    n = len(policy_types) or 1
    gold = sum(1 for item in policy_types if item == "gold_watchlist")
    null = sum(1 for item in policy_types if item is None)
    n1_n = len(n1_types)
    n2_n = len(n2_types)
    n1_gold = sum(1 for item in n1_types if item == "gold_watchlist")
    n2_gold = sum(1 for item in n2_types if item == "gold_watchlist")
    gold_rate = gold / n
    n1_gold_rate = (n1_gold / n1_n) if n1_n else 0.0
    n2_gold_rate = (n2_gold / n2_n) if n2_n else 0.0
    n2_verified_hit_rate = (sum(n2_verified) / len(n2_verified)) if n2_verified else 0.0
    n2_null_rate = (n2_null / n2_n) if n2_n else 0.0
    select_score = n2_verified_hit_rate if n2_n else gold_rate
    return {
        "gold_rate": gold_rate,
        "n1_gold_rate": n1_gold_rate,
        "n2_gold_rate": n2_gold_rate,
        "n2_verified_hit_rate": n2_verified_hit_rate,
        "n2_null_rate": n2_null_rate,
        "null_rate": null / n,
        "select_score": select_score,
        "mean_reward": (
            sum(float(row.get("mean_reward") or 0.0) for row in chunk) / len(chunk) if chunk else 0.0
        ),
        "optimizer_steps": sum(1 for row in chunk if row.get("optimizer_step")),
        "collapsed_count": sum(int(row.get("collapsed_count") or 0) for row in chunk),
    }


def _save_adapter(model, processor, path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    processor.save_pretrained(str(path))


def run_lightweight_grpo(
    *,
    model_path: str | Path,
    tasks: list[dict],
    catalog: dict,
    output_dir: str | Path,
    group_size: int = 4,
    max_steps: int = 100,
    max_environment_steps: int = 14,
    learning_rate: float = 1e-6,
    lora_r: int = 16,
    lora_alpha: int = 32,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    max_length: int = 4096,
    dtype_name: str = "bf16",
    shaped_reward: bool = True,
    resample_attempts: int = 2,
    clip_ratio: float = 0.2,
    save_every: int = 25,
    logprob_reduction: str = "sum",
    neg_advantage_coef: float = 1.0,
    task_schedule: str = "director_first",
    n2_oversample: float = 1.0,
    schedule_seed: int = 42,
    kl_coefficient: float = 0.0,
    n2_temperature: float | None = None,
    oracle_mix: bool = False,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not tasks:
        raise ValueError("GRPO train set is empty")
    reduction = str(logprob_reduction or "sum")
    if reduction not in {"sum", "mean"}:
        raise ValueError(f"unsupported logprob_reduction: {reduction}")
    neg_coef = max(0.0, float(neg_advantage_coef))
    kl_coef = max(0.0, float(kl_coefficient))
    n2_temp = float(temperature if n2_temperature is None else n2_temperature)

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
    checkpoints = []
    started = time.time()
    optimizer_steps = 0
    skipped_groups = 0
    resampled_groups = 0
    dropped_collapsed = 0
    scheduled_tasks = build_task_schedule(
        tasks,
        max_steps,
        mode=task_schedule,
        n2_oversample=n2_oversample,
        seed=schedule_seed,
    )

    def prepare_rollout(trajectory: dict) -> dict:
        if is_collapsed_rollout(trajectory):
            return {"collapsed": True, "example": None, "old_logprob": None}
        example = build_supervised_example(
            messages=trajectory.get("messages") or [],
            tools=trajectory.get("tools") or TINYWATCH_TOOL_SCHEMAS,
            tokenizer=tokenizer,
            max_length=max_length,
            chat_template=processor,
            overflow="truncate",
        )
        if example is None:
            return {"collapsed": False, "example": None, "old_logprob": None}
        tensors = prepare_example_tensors(example, device)
        try:
            with torch.inference_mode():
                old_logprob = float(
                    sequence_logprob(model, **tensors, reduction=reduction).detach().cpu()
                )
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"collapsed": False, "example": None, "old_logprob": None}
        finally:
            del tensors
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {"collapsed": False, "example": example, "old_logprob": old_logprob}

    def rollout_group(task: dict) -> tuple[list[dict], list[float], list[dict]]:
        trajectories = []
        rewards = []
        artifacts = []
        n_movies = task_n_movies(task)
        policy_n = n2_policy_rollout_count(group_size, n_movies, oracle_mix=oracle_mix)
        previous_temp = client.temperature
        if n_movies >= 2:
            client.temperature = n2_temp
        try:
            for _ in range(policy_n):
                trajectory = rollout_task(
                    task,
                    catalog,
                    client,
                    max_steps=max_environment_steps,
                    teacher_model=str(model_path),
                )
                trajectories.append(trajectory)
                rewards.append(trajectory_reward(trajectory, task, shaped=shaped_reward))
                artifacts.append(prepare_rollout(trajectory))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            client.temperature = previous_temp
        if policy_n < group_size:
            trajectory = oracle_rollout(task, catalog, max_steps=max_environment_steps)
            trajectories.append(trajectory)
            rewards.append(trajectory_reward(trajectory, task, shaped=shaped_reward))
            artifacts.append(prepare_rollout(trajectory))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return trajectories, rewards, artifacts

    extra_attempts = max(0, int(resample_attempts))
    clip = max(0.0, float(clip_ratio))
    history_path = output_dir / "history.jsonl"
    history_path.write_text("", encoding="utf-8")
    for step in range(max_steps):
        task = scheduled_tasks[step]
        n_movies = task_n_movies(task)
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        trajectories, rewards, artifacts = [], [], []
        kept = []
        diagnostics = {}
        attempts_used = 0
        for attempt in range(1 + extra_attempts):
            trajectories, rewards, artifacts = rollout_group(task)
            trainable_mask = [
                (item.get("example") is not None and item.get("old_logprob") is not None)
                for item in artifacts
            ]
            kept, diagnostics = select_trainable_varying_indices(rewards, trainable_mask)
            attempts_used = attempt + 1
            if kept:
                if attempt:
                    resampled_groups += 1
                break
        dropped_collapsed += sum(1 for item in artifacts if item.get("collapsed"))
        trainable_rewards = [float(rewards[index]) for index in kept]
        advantages_kept = [
            scale_advantage(value, neg_coef=neg_coef)
            for value in (group_advantages(trainable_rewards) if kept else [])
        ]
        advantage_by_index = dict(zip(kept, advantages_kept))
        record = {
            "step": step,
            "task_id": task["task_id"],
            "n_movies": n_movies,
            "rewards": rewards,
            "advantages": [advantage_by_index.get(index) for index in range(len(rewards))],
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "kept_indices": kept,
            "dynamic_sampling": diagnostics,
            "resample_attempts_used": attempts_used,
            "collapsed_count": sum(1 for item in artifacts if item.get("collapsed")),
            "reward_types": [
                ((item.get("terminal_result") or {}).get("reward_detail") or {}).get("reward_type")
                for item in trajectories
            ],
            "oracle_flags": [bool(item.get("oracle")) for item in trajectories],
            "verified_hit_fracs": [
                verified_gold_hit_fraction(item, task) for item in trajectories
            ],
        }

        if not kept:
            skipped_groups += 1
            record["optimizer_step"] = False
            history.append(record)
            _append_history(history_path, record)
            print(
                f"[grpo {step + 1}/{max_steps}] task={task['task_id']} "
                f"mean_r={record['mean_reward']:.3f} skipped_constant_or_collapsed "
                f"attempts={attempts_used} collapsed={record['collapsed_count']}",
                flush=True,
            )
            _maybe_checkpoint(
                model,
                processor,
                output_dir,
                history,
                checkpoints,
                step,
                max_steps,
                save_every,
            )
            continue

        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        trainable_kept = 0
        for index in kept:
            artifact = artifacts[index]
            advantage = float(advantage_by_index[index])
            if abs(advantage) < 1e-8:
                continue
            example = artifact["example"]
            old_logprob = float(artifact["old_logprob"])
            tensors = prepare_example_tensors(example, device)
            try:
                ref_logprob = None
                if kl_coef > 0:
                    disable = getattr(model, "disable_adapter", None)
                    with torch.inference_mode():
                        if disable is not None:
                            with model.disable_adapter():
                                ref_value = sequence_logprob(
                                    model, **tensors, reduction=reduction
                                )
                        else:
                            ref_value = tensors["input_ids"].new_zeros(())
                    ref_logprob = float(ref_value.detach().cpu())
                new_logprob = sequence_logprob(model, **tensors, reduction=reduction)
                ratio = torch.exp(new_logprob - old_logprob)
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                loss = -torch.min(ratio * advantage, clipped * advantage)
                if ref_logprob is not None:
                    loss = loss + kl_coef * (new_logprob - ref_logprob)
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
            _append_history(history_path, record)
            print(
                f"[grpo {step + 1}/{max_steps}] task={task['task_id']} "
                f"mean_r={record['mean_reward']:.3f} no_trainable_examples",
                flush=True,
            )
            _maybe_checkpoint(
                model,
                processor,
                output_dir,
                history,
                checkpoints,
                step,
                max_steps,
                save_every,
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
        _append_history(history_path, record)
        print(
            f"[grpo {step + 1}/{max_steps}] task={task['task_id']} "
            f"mean_r={record['mean_reward']:.3f} loss={record['loss']:.4f} "
            f"kept={len(kept)} collapsed={record['collapsed_count']}",
            flush=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _maybe_checkpoint(
            model,
            processor,
            output_dir,
            history,
            checkpoints,
            step,
            max_steps,
            save_every,
        )

    adapter_dir = output_dir / "adapter"
    selected = _select_checkpoint(checkpoints)
    if selected is not None:
        _save_adapter(model, processor, output_dir / "checkpoints" / "last")
        source = Path(selected["path"])
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(source, adapter_dir)
        print(
            f"[grpo] selected checkpoint step={selected['step']} "
            f"gold_rate={selected['gold_rate']:.3f} "
            f"n2_gold_rate={selected.get('n2_gold_rate', 0.0):.3f} "
            f"n2_verified={selected.get('n2_verified_hit_rate', 0.0):.3f} "
            f"n2_null={selected.get('n2_null_rate', 0.0):.3f} "
            f"null_rate={selected['null_rate']:.3f}",
            flush=True,
        )
    else:
        _save_adapter(model, processor, adapter_dir)

    summary = {
        "steps": len(history),
        "optimizer_steps": optimizer_steps,
        "skipped_constant_groups": skipped_groups,
        "resampled_groups": resampled_groups,
        "dropped_collapsed": dropped_collapsed,
        "selected_checkpoint": selected,
        "checkpoints": checkpoints,
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
            "shaped_reward": shaped_reward,
            "resample_attempts": extra_attempts,
            "temperature": temperature,
            "clip_ratio": clip,
            "save_every": save_every,
            "overflow": "truncate",
            "logprob_reduction": reduction,
            "neg_advantage_coef": neg_coef,
            "task_schedule": task_schedule,
            "n2_oversample": float(n2_oversample),
            "schedule_seed": int(schedule_seed),
            "kl_coefficient": kl_coef,
            "n2_temperature": n2_temp,
            "oracle_mix": bool(oracle_mix),
        },
    }
    history_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in history) + ("\n" if history else ""),
        encoding="utf-8",
    )
    (output_dir / "grpo_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _append_history(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _maybe_checkpoint(
    model,
    processor,
    output_dir: Path,
    history: list[dict],
    checkpoints: list[dict],
    step: int,
    max_steps: int,
    save_every: int,
) -> None:
    step_one = step + 1
    if save_every <= 0:
        return
    if step_one % save_every != 0 and step_one != max_steps:
        return
    start = max(0, step_one - save_every)
    stats = _window_rates(history, start, step_one)
    path = output_dir / "checkpoints" / f"step-{step_one}"
    _save_adapter(model, processor, path)
    record = {"step": step_one, "path": str(path), **stats}
    checkpoints.append(record)
    print(
        f"[grpo] saved {path} gold_rate={stats['gold_rate']:.3f} "
        f"n2_gold_rate={stats['n2_gold_rate']:.3f} "
        f"n2_verified={stats.get('n2_verified_hit_rate', 0.0):.3f} "
        f"n2_null={stats.get('n2_null_rate', 0.0):.3f} "
        f"null_rate={stats['null_rate']:.3f}",
        flush=True,
    )


def _select_checkpoint(checkpoints: list[dict]) -> dict | None:
    if not checkpoints:
        return None
    return max(
        checkpoints,
        key=lambda item: (
            float(item.get("select_score") if item.get("select_score") is not None else item.get("n2_verified_hit_rate") or item.get("gold_rate") or 0.0),
            float(item.get("n2_gold_rate") or 0.0),
            float(item.get("n1_gold_rate") or 0.0),
            -float(item.get("n2_null_rate") or item.get("null_rate") or 0.0),
            -int(item["step"]),
        ),
    )


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
