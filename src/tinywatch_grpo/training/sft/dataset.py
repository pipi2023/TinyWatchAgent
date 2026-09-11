"""Action-only SFT labels: mask user and tool observations."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


IGNORE_INDEX = -100


def _token_ids(tokenizer, text):
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _common_prefix_length(left, right):
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def normalize_messages_for_chat_template(messages):
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            function["arguments"] = parsed
    return normalized


def build_supervised_example(messages, tools, tokenizer, max_length=8192, chat_template=None):
    template = chat_template or tokenizer
    rendered_messages = normalize_messages_for_chat_template(messages)
    if rendered_messages is None:
        return None
    assistant_indices = [
        index for index, message in enumerate(rendered_messages) if message.get("role") == "assistant"
    ]
    if not assistant_indices:
        return None
    try:
        full_text = template.apply_chat_template(
            rendered_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
        )
        input_ids = _token_ids(tokenizer, full_text)
    except Exception:
        return None
    if len(input_ids) > int(max_length):
        return None
    labels = [IGNORE_INDEX] * len(input_ids)
    for index in assistant_indices:
        try:
            prefix_text = template.apply_chat_template(
                rendered_messages[:index],
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
            )
            through_assistant_text = template.apply_chat_template(
                rendered_messages[: index + 1],
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_ids = _token_ids(tokenizer, prefix_text)
            through_assistant_ids = _token_ids(tokenizer, through_assistant_text)
        except Exception:
            return None
        start = _common_prefix_length(prefix_ids, through_assistant_ids)
        end = len(through_assistant_ids)
        if start >= end or end > len(input_ids):
            return None
        if input_ids[:end] != through_assistant_ids:
            return None
        labels[start:end] = input_ids[start:end]
    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def load_supervised_examples(path, tokenizer, max_length=8192, chat_template=None, task_ids=None):
    examples = []
    stats = {"total": 0, "kept": 0, "dropped": 0}
    requested_ids = {int(task_id) for task_id in task_ids} if task_ids is not None else None
    if requested_ids is not None:
        stats["filtered_out"] = 0
        stats["matched"] = 0
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    stats["total"] = len(lines)
    for line in lines:
        try:
            row = json.loads(line)
            if requested_ids is not None and int(row["task_id"]) not in requested_ids:
                stats["filtered_out"] += 1
                continue
            if requested_ids is not None:
                stats["matched"] += 1
            example = build_supervised_example(
                messages=row["messages"],
                tools=row.get("tools") or [],
                tokenizer=tokenizer,
                max_length=max_length,
                chat_template=chat_template,
            )
        except (KeyError, TypeError, json.JSONDecodeError):
            example = None
        if example is None:
            stats["dropped"] += 1
            continue
        example["task_id"] = row.get("task_id")
        example["trajectory_id"] = row.get("trajectory_id")
        examples.append(example)
        stats["kept"] += 1
    return examples, stats


def split_rows_by_task(rows, validation_ratio=0.2, seed=42):
    ratio = float(validation_ratio)
    if not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    task_ids = {row.get("task_id") for row in rows}
    if ratio == 0 or len(task_ids) < 2:
        return list(rows), []

    def stable_key(task_id):
        return hashlib.sha256(f"{seed}:{task_id}".encode("utf-8")).hexdigest()

    ordered_ids = sorted(task_ids, key=stable_key)
    validation_count = max(1, round(len(ordered_ids) * ratio))
    validation_count = min(validation_count, len(ordered_ids) - 1)
    validation_ids = set(ordered_ids[:validation_count])
    validation_rows = [row for row in rows if row.get("task_id") in validation_ids]
    train_rows = [row for row in rows if row.get("task_id") not in validation_ids]
    return train_rows, validation_rows
