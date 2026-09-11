#!/usr/bin/env bash
# Serve a checkpoint, run Final-100 deterministic eval, then stop the server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
LABEL="${1:?usage: $0 <label> <model_path>}"
MODEL="${2:?usage: $0 <label> <model_path>}"
PORT="${LLM_PORT:-8000}"
HOST="${LLM_HOST:-127.0.0.1}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-tinywatch-agent}"
OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$ROOT/outputs/evaluation/$LABEL}"
LOG="$ROOT/outputs/evaluation/${LABEL}-serve.log"

mkdir -p "$ROOT/outputs/evaluation"
cd "$ROOT"

# Free any leftover server on this port.
if command -v fuser >/dev/null 2>&1; then
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
fi
pkill -f "scripts/serve_openai_model.py" >/dev/null 2>&1 || true
sleep 2

echo "[eval] serving $MODEL as $SERVED_MODEL_NAME on :$PORT" | tee "$LOG"
"$PY" -u scripts/serve_openai_model.py \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  >>"$LOG" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$ROOT/outputs/evaluation/${LABEL}-serve.pid"

cleanup() {
  kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait "$SERVER_PID" 2>/dev/null || true
  rm -f "$ROOT/outputs/evaluation/${LABEL}-serve.pid"
}
trap cleanup EXIT

# Wait for health.
for _ in $(seq 1 120); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "serve process died; see $LOG" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "http://${HOST}:${PORT}/health" >/dev/null

export LLM_BASE_URL="http://${HOST}:${PORT}/v1"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export SERVED_MODEL_NAME
export EVAL_OUTPUT_DIR="$OUTPUT_DIR"
bash scripts/evaluate.sh "$LABEL"
echo "[eval] done $LABEL -> $OUTPUT_DIR/summary.json"
