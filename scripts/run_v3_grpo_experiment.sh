#!/usr/bin/env bash
# v3: shopping-aligned GRPO from frozen v2 SFT. Does not regenerate tasks or SFT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
export TINYWATCH_PYTHON="$PY"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

SFT_MERGED="${SFT_MERGED_DIR:-$ROOT/outputs/models/v2/sft-merged}"
GRPO_OUT="${GRPO_OUTPUT_DIR:-$ROOT/outputs/models/v3/grpo}"
GRPO_MERGED="${GRPO_MERGED_DIR:-$ROOT/outputs/models/v3/grpo-merged}"

if [[ ! -d "$SFT_MERGED" ]]; then
  echo "missing v2 SFT merged checkpoint: $SFT_MERGED" >&2
  exit 1
fi

echo "[v3] archiving frozen v2"
bash scripts/archive_v2.sh v2-20260914

mkdir -p outputs/models/v3 experiments/v3 outputs/logs

echo "[v3] GRPO from $SFT_MERGED"
"$PY" -u scripts/train_grpo.py \
  --model "$SFT_MERGED" \
  --output "$GRPO_OUT" \
  --merged-output "$GRPO_MERGED" \
  --max-steps "${GRPO_STEPS:-100}" \
  --max-environment-steps "${GRPO_MAX_ENV_STEPS:-14}" \
  --learning-rate "${GRPO_LR:-1e-6}" \
  --temperature "${GRPO_TEMPERATURE:-0.7}" \
  --resample-attempts "${GRPO_RESAMPLE_ATTEMPTS:-2}" \
  --max-length "${GRPO_MAX_LENGTH:-4096}" \
  --clip-ratio "${GRPO_CLIP_RATIO:-0.2}" \
  --save-every "${GRPO_SAVE_EVERY:-25}" \
  --merge

echo "[v3] eval selected GRPO checkpoint"
bash scripts/run_eval_checkpoint.sh v3-grpo "$GRPO_MERGED"

"$PY" scripts/build_comparison_report.py \
  --baseline "$ROOT/outputs/evaluation/v2-baseline/summary.json" \
  --sft "$ROOT/outputs/evaluation/v2-sft/summary.json" \
  --grpo "$ROOT/outputs/evaluation/v3-grpo/summary.json" \
  --output "$ROOT/experiments/v3/comparison.md"

mkdir -p "$ROOT/experiments/v3/sft" "$ROOT/experiments/v3/grpo" "$ROOT/experiments/v3/baseline"
cp -a "$ROOT/outputs/evaluation/v2-sft/summary.json" "$ROOT/experiments/v3/sft/summary.json"
cp -a "$ROOT/outputs/evaluation/v3-grpo/summary.json" "$ROOT/experiments/v3/grpo/summary.json"
cp -a "$ROOT/outputs/evaluation/v2-baseline/summary.json" "$ROOT/experiments/v3/baseline/summary.json"
echo "[v3] ALL_EVAL_DONE -> $ROOT/experiments/v3/comparison.md"
