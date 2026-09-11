#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
cd "$ROOT"
exec "$PY" scripts/train_grpo.py --max-steps "${GRPO_STEPS:-100}" --merge "$@"
