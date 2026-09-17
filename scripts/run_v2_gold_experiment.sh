#!/usr/bin/env bash
# v2 gold@100 experiment: identifiable-gold tasks, Oracle SFT, shaped GRPO, Final-100.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
BASE_MODEL="${BASE_MODEL:-${LOCAL_BASE_MODEL:-/root/autodl-tmp/models/Qwen3.5-2B}}"
export TINYWATCH_PYTHON="$PY"
export BASE_MODEL
export SFT_ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/v2/sft-lora}"
export SFT_MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/v2/sft-merged}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
cd "$ROOT"
mkdir -p outputs/models/v2 experiments/v2

echo "[v2] generating identifiable-gold splits"
"$PY" scripts/generate_tasks.py

echo "[v2] collecting Oracle SFT trajectories"
"$PY" -u scripts/prepare_v2_sft.py --copy

echo "[v2] Action-only LoRA SFT"
bash scripts/sft.sh

echo "[v2] eval SFT on new Final-100"
bash scripts/run_eval_checkpoint.sh v2-sft "$SFT_MERGED_DIR"

echo "[v2] shaped GRPO from SFT"
"$PY" -u scripts/train_grpo.py \
  --model "$SFT_MERGED_DIR" \
  --output "$ROOT/outputs/models/v2/grpo" \
  --merged-output "$ROOT/outputs/models/v2/grpo-merged" \
  --max-steps "${GRPO_STEPS:-100}" \
  --max-environment-steps "${GRPO_MAX_ENV_STEPS:-14}" \
  --temperature "${GRPO_TEMPERATURE:-1.0}" \
  --resample-attempts "${GRPO_RESAMPLE_ATTEMPTS:-1}" \
  --max-length "${GRPO_MAX_LENGTH:-4096}" \
  --merge

echo "[v2] eval GRPO"
bash scripts/run_eval_checkpoint.sh v2-grpo "$ROOT/outputs/models/v2/grpo-merged"

echo "[v2] eval baseline on the same new Final-100"
bash scripts/run_eval_checkpoint.sh v2-baseline "$BASE_MODEL"

"$PY" scripts/build_comparison_report.py \
  --baseline "$ROOT/outputs/evaluation/v2-baseline/summary.json" \
  --sft "$ROOT/outputs/evaluation/v2-sft/summary.json" \
  --grpo "$ROOT/outputs/evaluation/v2-grpo/summary.json" \
  --output "$ROOT/experiments/v2/comparison.md"

mkdir -p "$ROOT/experiments/v2/sft" "$ROOT/experiments/v2/grpo" "$ROOT/experiments/v2/baseline"
cp -a "$ROOT/outputs/evaluation/v2-sft/summary.json" "$ROOT/experiments/v2/sft/summary.json"
cp -a "$ROOT/outputs/evaluation/v2-grpo/summary.json" "$ROOT/experiments/v2/grpo/summary.json"
cp -a "$ROOT/outputs/evaluation/v2-baseline/summary.json" "$ROOT/experiments/v2/baseline/summary.json"
echo "[v2] ALL_EVAL_DONE -> $ROOT/experiments/v2/comparison.md"
