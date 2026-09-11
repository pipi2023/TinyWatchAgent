#!/usr/bin/env bash
# Serve a TinyWatch student checkpoint with an OpenAI-compatible /v1 API.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
MODEL="${1:-${SERVED_MODEL_PATH:-$ROOT/outputs/models/sft-merged}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-tinywatch-agent}"
LLM_PORT="${LLM_PORT:-8000}"
HOST="${LLM_HOST:-127.0.0.1}"

cd "$ROOT"
exec "$PY" scripts/serve_openai_model.py \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$LLM_PORT"
