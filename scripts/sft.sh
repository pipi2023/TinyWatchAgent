#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${TINYWATCH_PYTHON:-/root/miniconda3/bin/python}"
BASE_MODEL="${BASE_MODEL:-${LOCAL_BASE_MODEL:-/root/autodl-tmp/models/Qwen3.5-2B}}"
if [[ ! -d "$BASE_MODEL" && -d /root/autodl-tmp/modelscope-cache ]]; then
  # Prefer a completed ModelScope snapshot when HF mirror download is incomplete.
  CANDIDATE="$(find /root/autodl-tmp/modelscope-cache -type d -name 'Qwen3.5-2B' 2>/dev/null | head -1)"
  if [[ -n "${CANDIDATE:-}" && -f "$CANDIDATE/config.json" ]]; then
    BASE_MODEL="$CANDIDATE"
  fi
fi
ADAPTER_DIR="${SFT_ADAPTER_DIR:-$ROOT/outputs/models/sft-lora}"
MERGED_DIR="${SFT_MERGED_DIR:-$ROOT/outputs/models/sft-merged}"

cd "$ROOT"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"$PY" scripts/train_lora_sft.py \
  --model "$BASE_MODEL" \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output "$ADAPTER_DIR" \
  --dtype auto \
  --gradient-checkpointing \
  --liger-kernel \
  --attention-implementation sdpa

exec "$PY" scripts/merge_lora_adapter.py \
  --base-model "$BASE_MODEL" \
  --adapter "$ADAPTER_DIR" \
  --output "$MERGED_DIR" \
  --bf16
