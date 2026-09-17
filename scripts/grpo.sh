#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
cd "$ROOT"
exec "$PY" scripts/train_grpo.py \
  --max-steps "${GRPO_STEPS:-100}" \
  --max-environment-steps "${GRPO_MAX_ENV_STEPS:-14}" \
  --learning-rate "${GRPO_LR:-1e-6}" \
  --temperature "${GRPO_TEMPERATURE:-0.7}" \
  --resample-attempts "${GRPO_RESAMPLE_ATTEMPTS:-2}" \
  --max-length "${GRPO_MAX_LENGTH:-4096}" \
  --clip-ratio "${GRPO_CLIP_RATIO:-0.2}" \
  --save-every "${GRPO_SAVE_EVERY:-25}" \
  --merge "$@"
