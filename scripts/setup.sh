#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/autodl-tmp/miniconda/tinywatch/bin/python}"

cd "$ROOT"
"$PY" -m pip install -e .
if [[ "${TINYWATCH_MINI_CATALOG:-0}" == "1" ]]; then
  "$PY" scripts/import_imdb_catalog.py --mini
else
  "$PY" scripts/import_imdb_catalog.py
fi
"$PY" scripts/generate_tasks.py
"$PY" scripts/smoke_tinywatch.py
echo "TinyWatch CPU environment is ready."
echo "Python: $PY"
