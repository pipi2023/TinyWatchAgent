#!/usr/bin/env python3
"""Resumable DeepSeek-V4-Flash collection. Does not start unless invoked explicitly."""

from __future__ import annotations

import argparse
import json
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tinywatch_grpo.collection.agent_loop import OpenAIChatClient, rollout_task
from tinywatch_grpo.collection.envfile import load_dotenv
from tinywatch_grpo.collection.reward_filter import (
    acceptance_reasons,
    build_collection_artifacts,
    read_jsonl,
    task_ids_from_jsonl,
)
from tinywatch_grpo.environment.catalog import load_catalog
from tinywatch_grpo.evaluation.artifacts import append_jsonl


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("data/tasks/sft_pool.jsonl"))
    parser.add_argument("--catalog", type=Path, default=Path("data/catalog/catalog.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/flash-collection"))
    parser.add_argument("--held-out-tasks", type=Path, default=Path("data/evaluation/tasks.jsonl"))
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-accepted", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=14)
    parser.add_argument("--reward-threshold", type=float, default=0.55)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.environ.get("TEACHER_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--base-url", default=os.environ.get("TEACHER_BASE_URL"))
    parser.add_argument("--api-key", default=os.environ.get("TEACHER_API_KEY"))
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=30.0,
        help="Seconds between heartbeat accepted/pending prints. 0 disables the timer.",
    )
    return parser.parse_args()


def _completed_ids(path: Path) -> set[int]:
    return {int(row["task_id"]) for row in read_jsonl(path)} if path.exists() else set()


def _accepted_count(path: Path, held_out: set[int], threshold: float) -> int:
    count = 0
    seen = set()
    for row in read_jsonl(path):
        ok, _reasons = acceptance_reasons(row, reward_threshold=threshold)
        task_id = int(row["task_id"])
        if ok and task_id not in held_out and task_id not in seen:
            seen.add(task_id)
            count += 1
    return count


def _raw_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in read_jsonl(path))


def progress_message(
    *,
    accepted: int,
    target: int,
    pending: int,
    written: int,
    queued: int,
) -> str:
    return (
        f"accepted={accepted}/{target} pending={pending} "
        f"written={written} queued={queued}"
    )


class CollectionProgress:
    """Thread-safe raw.jsonl writer plus accepted/pending heartbeats."""

    def __init__(
        self,
        *,
        raw_path: Path,
        target: int,
        accepted: int,
        written: int,
        queued: int,
        interval: float,
    ):
        self.raw_path = Path(raw_path)
        self.target = int(target)
        self.interval = float(interval)
        self.lock = threading.Lock()
        self.accepted = int(accepted)
        self.written = int(written)
        self.queued = int(queued)
        self.pending = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                "accepted": self.accepted,
                "target": self.target,
                "pending": self.pending,
                "written": self.written,
                "queued": self.queued,
            }

    def format_progress(self) -> str:
        return progress_message(**self.snapshot())

    def print_progress(self, extra: str = "") -> None:
        line = self.format_progress()
        if extra:
            line = f"{line} {extra}"
        print(line, flush=True)

    def append_raw(self, trajectory: dict) -> None:
        with self.lock:
            append_jsonl(self.raw_path, [trajectory])
            self.written += 1

    def mark_submitted(self) -> None:
        with self.lock:
            self.queued -= 1
            self.pending += 1

    def mark_finished(self, *, accepted: bool) -> None:
        with self.lock:
            self.pending -= 1
            if accepted:
                self.accepted += 1

    def start(self) -> None:
        self.print_progress()
        if self.interval <= 0:
            return
        self._thread = threading.Thread(target=self._heartbeat, name="collect-progress", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            self._thread = None
        self.print_progress()

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            self.print_progress()


def main() -> None:
    args = parse_args()
    raw_path = args.output_dir / "raw.jsonl"
    held_out = task_ids_from_jsonl(args.held_out_tasks)
    if args.build_only:
        metadata = build_collection_artifacts(
            raw_path=raw_path,
            output_dir=args.output_dir,
            held_out_task_ids=held_out,
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            reward_threshold=args.reward_threshold,
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if not args.base_url or not args.api_key:
        raise SystemExit("TEACHER_BASE_URL and TEACHER_API_KEY are required (do not commit secrets)")
    catalog = load_catalog(args.catalog)
    tasks = []
    for line in args.tasks.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task = json.loads(line)
        if int(task["task_id"]) in held_out:
            continue
        tasks.append(task)
        if args.limit is not None and len(tasks) >= args.limit:
            break
    completed = _completed_ids(raw_path)
    pending_tasks = [task for task in tasks if int(task["task_id"]) not in completed]
    accepted = _accepted_count(raw_path, held_out, args.reward_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress = CollectionProgress(
        raw_path=raw_path,
        target=int(args.target_accepted),
        accepted=accepted,
        written=_raw_row_count(raw_path),
        queued=len(pending_tasks),
        interval=args.progress_interval,
    )

    def client_factory():
        return OpenAIChatClient(
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            thinking=args.thinking,
            timeout=args.timeout,
            max_tokens=args.max_tokens,
        )

    def collect_one(task: dict) -> dict:
        trajectory = rollout_task(
            task,
            catalog,
            client_factory(),
            max_steps=args.max_steps,
            teacher_model=args.model,
        )
        progress.append_raw(trajectory)
        return trajectory

    workers = max(1, int(args.workers))
    pending = {}
    iterator = iter(pending_tasks)

    def submit(executor):
        remaining = int(args.target_accepted) - progress.snapshot()["accepted"]
        if remaining <= 0:
            return
        while len(pending) < min(workers, remaining):
            try:
                task = next(iterator)
            except StopIteration:
                return
            future = executor.submit(collect_one, task)
            pending[future] = task
            progress.mark_submitted()

    progress.start()
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            submit(executor)
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    task = pending.pop(future)
                    accepted_row = False
                    extra = f"task_id={task['task_id']}"
                    try:
                        trajectory = future.result()
                        ok, _reasons = acceptance_reasons(
                            trajectory, reward_threshold=args.reward_threshold
                        )
                        accepted_row = bool(
                            ok and int(trajectory["task_id"]) not in held_out
                        )
                        extra = f"task_id={task['task_id']} ok={int(accepted_row)}"
                    finally:
                        progress.mark_finished(accepted=accepted_row)
                    progress.print_progress(extra=extra)
                    submit(executor)
    finally:
        progress.stop()

    metadata = build_collection_artifacts(
        raw_path=raw_path,
        output_dir=args.output_dir,
        held_out_task_ids=held_out,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
        reward_threshold=args.reward_threshold,
        collection_config={
            "model": args.model,
            "thinking": args.thinking,
            "workers": args.workers,
            "max_steps": args.max_steps,
            "reward_threshold": args.reward_threshold,
            "progress_interval": args.progress_interval,
        },
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
