"""Accept Reward v1 trajectories into action-only SFT JSONL."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

from tinywatch_grpo.environment.tools import TINYWATCH_TOOL_SCHEMAS
from tinywatch_grpo.training.sft.dataset import split_rows_by_task


COLLECTION_SCHEMA_VERSION = "tinywatch-sft-collection-v1"
ALLOWED_MESSAGE_KEYS = {"role", "content", "tool_calls", "tool_call_id", "name"}
ALLOWED_TOOL_CALL_KEYS = {"id", "type", "function"}
ALLOWED_FUNCTION_KEYS = {"name", "arguments"}
DEFAULT_REWARD_THRESHOLD = 0.55


def acceptance_reasons(trajectory: dict, *, reward_threshold: float = DEFAULT_REWARD_THRESHOLD) -> tuple[bool, list[str]]:
    reasons = []
    terminal = trajectory.get("terminal_result") or {}
    reward = terminal.get("reward_detail") or {}
    if trajectory.get("error"):
        reasons.append("has_error")
    if trajectory.get("status") != "done":
        reasons.append("status_not_done")
    if trajectory.get("done") is not True:
        reasons.append("trajectory_not_done")
    if not any(step.get("tool_name") == "finalize_watchlist" for step in trajectory.get("steps") or []):
        if reward.get("reward_type") != "correct_abort":
            reasons.append("missing_finalize")
    if reward.get("reward_version") != "tinywatch-reward-v1":
        reasons.append("reward_v1_required")
    if reward.get("reward_valid") is not True:
        reasons.append("reward_invalid")
    try:
        if float(reward.get("reward", 0.0)) < float(reward_threshold):
            reasons.append("reward_below_threshold")
    except (TypeError, ValueError):
        reasons.append("reward_unreadable")
    for index, message in enumerate(trajectory.get("messages") or []):
        if message.get("role") == "assistant" and len(message.get("tool_calls") or []) > 1:
            reasons.append(f"message_{index}.multiple_tool_calls")
    return not reasons, reasons


def _sanitize_messages(messages: list) -> list:
    cleaned = []
    for message in messages:
        row = {key: message[key] for key in ALLOWED_MESSAGE_KEYS if key in message}
        if row.get("role") == "assistant":
            row.pop("reasoning_content", None)
            row["content"] = row.get("content") or ""
            calls = []
            for call in row.get("tool_calls") or []:
                function = {
                    key: (call.get("function") or {}).get(key)
                    for key in ALLOWED_FUNCTION_KEYS
                    if (call.get("function") or {}).get(key) is not None
                }
                calls.append({key: call[key] for key in ALLOWED_TOOL_CALL_KEYS if key in call} | {"function": function})
            if calls:
                row["tool_calls"] = calls
        cleaned.append(row)
    return cleaned


def build_sft_row(trajectory: dict) -> dict:
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "task_id": int(trajectory["task_id"]),
        "messages": _sanitize_messages(trajectory.get("messages") or []),
        "tools": deepcopy(TINYWATCH_TOOL_SCHEMAS),
    }


def read_jsonl(path: str | Path):
    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def task_ids_from_jsonl(path: str | Path) -> set[int]:
    path = Path(path)
    if not path.exists():
        return set()
    return {int(row["task_id"]) for row in read_jsonl(path)}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_collection_artifacts(
    *,
    raw_path: str | Path,
    output_dir: str | Path,
    held_out_task_ids: Iterable[int] = (),
    validation_ratio: float = 0.2,
    seed: int = 42,
    reward_threshold: float = DEFAULT_REWARD_THRESHOLD,
    collection_config: dict | None = None,
) -> dict:
    output_dir = Path(output_dir)
    held_out = {int(task_id) for task_id in held_out_task_ids}
    accepted_trajectories = []
    rejected_rows = []
    sft_rows = []
    accepted_task_ids = set()
    reject_reasons = Counter()
    total = 0
    held_out_excluded = 0
    duplicate_tasks_excluded = 0
    quality_rejected = 0

    for trajectory in read_jsonl(raw_path):
        total += 1
        accepted, reasons = acceptance_reasons(trajectory, reward_threshold=reward_threshold)
        task_id = int(trajectory["task_id"])
        if accepted and task_id in held_out:
            accepted = False
            reasons = ["held_out_task"]
            held_out_excluded += 1
        elif accepted and task_id in accepted_task_ids:
            accepted = False
            reasons = ["duplicate_task"]
            duplicate_tasks_excluded += 1
        elif not accepted:
            quality_rejected += 1
        if accepted:
            accepted_task_ids.add(task_id)
            accepted_trajectories.append(trajectory)
            sft_rows.append(build_sft_row(trajectory))
            continue
        reject_reasons.update(reasons)
        rejected_rows.append(
            {
                "trajectory_id": trajectory.get("trajectory_id"),
                "task_id": task_id,
                "status": trajectory.get("status"),
                "reject_reasons": reasons,
            }
        )

    train_rows, validation_rows = split_rows_by_task(
        sft_rows, validation_ratio=validation_ratio, seed=seed
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "accepted": output_dir / "accepted.jsonl",
        "rejected": output_dir / "rejected.jsonl",
        "sft": output_dir / "sft.jsonl",
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "stats": output_dir / "reject_stats.json",
        "metadata": output_dir / "metadata.json",
    }
    write_jsonl(paths["accepted"], accepted_trajectories)
    write_jsonl(paths["rejected"], rejected_rows)
    write_jsonl(paths["sft"], sft_rows)
    write_jsonl(paths["train"], train_rows)
    write_jsonl(paths["validation"], validation_rows)
    summary = {
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "total": total,
        "accepted": len(sft_rows),
        "rejected": quality_rejected,
        "held_out_excluded": held_out_excluded,
        "duplicate_tasks_excluded": duplicate_tasks_excluded,
        "train": len(train_rows),
        "validation": len(validation_rows),
        "reward_threshold": reward_threshold,
        "reject_reasons": dict(sorted(reject_reasons.items())),
        "hashes": {name: sha256_file(path) for name, path in paths.items() if path.suffix == ".jsonl"},
    }
    paths["stats"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metadata = {
        **summary,
        "environment": "tinywatch-environment-v1",
        "reward": "tinywatch-reward-v1",
        "collection_config": collection_config or {},
    }
    paths["metadata"].write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata
