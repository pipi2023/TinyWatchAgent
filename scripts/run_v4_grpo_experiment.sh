#!/usr/bin/env bash
# v4: n2-focused GRPO from frozen v2 SFT. Archives v3 first. No new tasks/SFT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
export TINYWATCH_PYTHON="$PY"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
cd "$ROOT"

SFT_MERGED="${SFT_MERGED_DIR:-$ROOT/outputs/models/v2/sft-merged}"
GRPO_OUT="${GRPO_OUTPUT_DIR:-$ROOT/outputs/models/v4/grpo}"
GRPO_MERGED="${GRPO_MERGED_DIR:-$ROOT/outputs/models/v4/grpo-merged}"
LOG_DIR="$ROOT/outputs/logs"
mkdir -p "$LOG_DIR" outputs/models/v4 experiments/v4

if [[ ! -d "$SFT_MERGED" ]]; then
  echo "missing v2 SFT merged checkpoint: $SFT_MERGED" >&2
  exit 1
fi

echo "[v4] archiving frozen v3"
bash scripts/archive_v3.sh v3-20260915

echo "[v4] GRPO from $SFT_MERGED (n2 oversample + length-norm PPO)"
"$PY" -u scripts/train_grpo.py \
  --model "$SFT_MERGED" \
  --output "$GRPO_OUT" \
  --merged-output "$GRPO_MERGED" \
  --max-steps "${GRPO_STEPS:-200}" \
  --max-environment-steps "${GRPO_MAX_ENV_STEPS:-16}" \
  --learning-rate "${GRPO_LR:-1e-6}" \
  --temperature "${GRPO_TEMPERATURE:-0.7}" \
  --resample-attempts "${GRPO_RESAMPLE_ATTEMPTS:-2}" \
  --max-length "${GRPO_MAX_LENGTH:-4096}" \
  --clip-ratio "${GRPO_CLIP_RATIO:-0.2}" \
  --save-every "${GRPO_SAVE_EVERY:-25}" \
  --logprob-reduction "${GRPO_LOGPROB_REDUCTION:-mean}" \
  --neg-advantage-coef "${GRPO_NEG_ADV_COEF:-0.5}" \
  --n2-oversample "${GRPO_N2_OVERSAMPLE:-2.0}" \
  --schedule-seed "${GRPO_SCHEDULE_SEED:-42}" \
  --task-schedule shuffled_n2 \
  --merge

echo "[v4] eval selected GRPO checkpoint"
bash scripts/run_eval_checkpoint.sh v4-grpo "$GRPO_MERGED"

"$PY" scripts/build_comparison_report.py \
  --baseline "$ROOT/outputs/evaluation/v2-baseline/summary.json" \
  --sft "$ROOT/outputs/evaluation/v2-sft/summary.json" \
  --grpo "$ROOT/outputs/evaluation/v4-grpo/summary.json" \
  --output "$ROOT/experiments/v4/comparison.md"

mkdir -p "$ROOT/experiments/v4/sft" "$ROOT/experiments/v4/grpo" "$ROOT/experiments/v4/baseline"
cp -a "$ROOT/outputs/evaluation/v2-sft/summary.json" "$ROOT/experiments/v4/sft/summary.json"
cp -a "$ROOT/outputs/evaluation/v4-grpo/summary.json" "$ROOT/experiments/v4/grpo/summary.json"
cp -a "$ROOT/outputs/evaluation/v2-baseline/summary.json" "$ROOT/experiments/v4/baseline/summary.json"
echo "[v4] ALL_EVAL_DONE -> $ROOT/experiments/v4/comparison.md"
