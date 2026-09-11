import importlib.util
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def _load_collect_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "collect_flash_trajectories.py"
    spec = importlib.util.spec_from_file_location("collect_flash_trajectories", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collect = _load_collect_module()


def test_progress_message_includes_accepted_and_pending():
    line = collect.progress_message(
        accepted=27, target=400, pending=4, written=50, queued=691
    )
    assert line == "accepted=27/400 pending=4 written=50 queued=691"


def test_append_raw_is_thread_safe(tmp_path):
    raw_path = tmp_path / "raw.jsonl"
    progress = collect.CollectionProgress(
        raw_path=raw_path,
        target=100,
        accepted=0,
        written=0,
        queued=0,
        interval=0,
    )
    workers = 8
    per_worker = 25

    def write_batch(worker_id):
        for index in range(per_worker):
            progress.append_raw({"task_id": worker_id * per_worker + index, "worker": worker_id})

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(write_batch, range(workers)))

    lines = raw_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == workers * per_worker
    rows = [json.loads(line) for line in lines]
    assert all(isinstance(row, dict) for row in rows)
    assert {row["task_id"] for row in rows} == set(range(workers * per_worker))
    assert progress.snapshot()["written"] == workers * per_worker


def test_progress_heartbeat_prints(tmp_path, capsys):
    progress = collect.CollectionProgress(
        raw_path=tmp_path / "raw.jsonl",
        target=400,
        accepted=13,
        written=20,
        queued=100,
        interval=0.05,
    )
    progress.pending = 4
    started = threading.Event()

    original_print = progress.print_progress

    def print_and_signal(extra=""):
        original_print(extra)
        started.set()

    progress.print_progress = print_and_signal
    progress.start()
    assert started.wait(timeout=2)
    progress.stop()
    output = capsys.readouterr().out
    assert "accepted=13/400" in output
    assert "pending=4" in output
