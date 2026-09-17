#!/usr/bin/env bash
# v5: Oracle-mix GRPO from frozen v2 SFT. Archives v4 first. No new tasks/SFT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
export TINYWATCH_PYTHON="$PY"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

SFT_MERGED="${SFT_MERGED_DIR:-$ROOT/outputs/models/v2/sft-merged}"
GRPO_OUT="${GRPO_OUTPUT_DIR:-$ROOT/outputs/models/v5/grpo}"
GRPO_MERGED="${GRPO_MERGED_DIR:-$ROOT/outputs/models/v5/grpo-merged}"
LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR" outputs/models/v5 experiments/v5

if [[ ! -d "$SFT_MERGED" ]]; then
  echo "missing v2 SFT merged checkpoint: $SFT_MERGED" >&2
  exit 1
fi

echo "[v5] archiving frozen v4"
bash scripts/archive_v4.sh v4-20260916

echo "[v5] GRPO from $SFT_MERGED (oracle mix + verified-hit shaping + KL)"
"$PY" -u scripts/train_grpo.py \
  --model "$SFT_MERGED" \
  --output "$GRPO_OUT" \
  --merged-output "$GRPO_MERGED" \
  --max-steps "${GRPO_STEPS:-100}" \
  --max-environment-steps "${GRPO_MAX_ENV_STEPS:-14}" \
  --learning-rate "${GRPO_LR:-1e-6}" \
  --temperature "${GRPO_TEMPERATURE:-0.7}" \
  --n2-temperature "${GRPO_N2_TEMPERATURE:-1.0}" \
  --resample-attempts "${GRPO_RESAMPLE_ATTEMPTS:-2}" \
  --max-length "${GRPO_MAX_LENGTH:-4096}" \
  --clip-ratio "${GRPO_CLIP_RATIO:-0.2}" \
  --save-every "${GRPO_SAVE_EVERY:-25}" \
  --logprob-reduction "${GRPO_LOGPROB_REDUCTION:-mean}" \
  --neg-advantage-coef "${GRPO_NEG_ADV_COEF:-0.5}" \
  --n2-oversample "${GRPO_N2_OVERSAMPLE:-1.0}" \
  --kl-coefficient "${GRPO_KL:-0.02}" \
  --schedule-seed "${GRPO_SCHEDULE_SEED:-42}" \
  --task-schedule shuffled_n2 \
  --oracle-mix \
  --merge

echo "[v5] eval selected GRPO checkpoint"
bash scripts/run_eval_checkpoint.sh v5-grpo "$GRPO_MERGED"

"$PY" scripts/build_comparison_report.py \
  --baseline "$ROOT/outputs/evaluation/v2-baseline/summary.json" \
  --sft "$ROOT/outputs/evaluation/v2-sft/summary.json" \
  --grpo "$ROOT/outputs/evaluation/v5-grpo/summary.json" \
  --output "$ROOT/experiments/v5/comparison.md"

mkdir -p "$ROOT/experiments/v5/sft" "$ROOT/experiments/v5/grpo" "$ROOT/experiments/v5/baseline"
cp -a "$ROOT/outputs/evaluation/v2-sft/summary.json" "$ROOT/experiments/v5/sft/summary.json"
cp -a "$ROOT/outputs/evaluation/v5-grpo/summary.json" "$ROOT/experiments/v5/grpo/summary.json"
cp -a "$ROOT/outputs/evaluation/v2-baseline/summary.json" "$ROOT/experiments/v5/baseline/summary.json"
echo "[v5] ALL_EVAL_DONE -> $ROOT/experiments/v5/comparison.md"
