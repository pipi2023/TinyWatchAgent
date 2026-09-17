#!/usr/bin/env bash
# Freeze v2 (identifiable-gold + collapsed GRPO) without duplicating SFT weights.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-v2-20260914}"
EXP_FREEZE="$ROOT/experiments/$STAMP"
OUT_FREEZE="$ROOT/outputs/archive/$STAMP"

if [[ -d "$EXP_FREEZE" && -f "$EXP_FREEZE/comparison.md" ]]; then
  echo "[archive] experiments/$STAMP already exists, skipping experiment copy"
else
  mkdir -p "$ROOT/experiments"
  cp -a "$ROOT/experiments/v2" "$EXP_FREEZE"
  echo "[archive] copied experiments/v2 -> $EXP_FREEZE"
fi

mkdir -p "$OUT_FREEZE/models" "$OUT_FREEZE/evaluation" "$OUT_FREEZE/oracle-v2-collection"

copy_or_link() {
  local src="$1" dest="$2"
  if [[ -e "$dest" ]]; then
    echo "[archive] keep existing $dest"
    return
  fi
  if [[ -d "$src" ]]; then
    cp -a "$src" "$dest"
  fi
}

# Evaluation dumps are small; copy.
for label in v2-baseline v2-sft v2-grpo; do
  if [[ -d "$ROOT/outputs/evaluation/$label" ]]; then
    copy_or_link "$ROOT/outputs/evaluation/$label" "$OUT_FREEZE/evaluation/$label"
  fi
  if [[ -f "$ROOT/outputs/evaluation/${label}-serve.log" ]]; then
    cp -a "$ROOT/outputs/evaluation/${label}-serve.log" "$OUT_FREEZE/evaluation/"
  fi
done

# SFT stays in outputs/models/v2 for v3; archive via hardlinks when possible.
if [[ -d "$ROOT/outputs/models/v2/sft-merged" && ! -e "$OUT_FREEZE/models/sft-merged" ]]; then
  cp -al "$ROOT/outputs/models/v2/sft-merged" "$OUT_FREEZE/models/sft-merged" 2>/dev/null \
    || cp -a "$ROOT/outputs/models/v2/sft-merged" "$OUT_FREEZE/models/sft-merged"
fi
if [[ -d "$ROOT/outputs/models/v2/sft-lora" && ! -e "$OUT_FREEZE/models/sft-lora" ]]; then
  cp -al "$ROOT/outputs/models/v2/sft-lora" "$OUT_FREEZE/models/sft-lora" 2>/dev/null \
    || cp -a "$ROOT/outputs/models/v2/sft-lora" "$OUT_FREEZE/models/sft-lora"
fi

# Move collapsed GRPO weights out of the live v2 tree to free space for v3.
if [[ -d "$ROOT/outputs/models/v2/grpo" && ! -e "$OUT_FREEZE/models/grpo" ]]; then
  mv "$ROOT/outputs/models/v2/grpo" "$OUT_FREEZE/models/grpo"
  echo "[archive] moved v2 GRPO adapter -> $OUT_FREEZE/models/grpo"
elif [[ -d "$ROOT/outputs/models/v2/grpo" ]]; then
  echo "[archive] freeze already has grpo; leaving live copy in place"
fi
if [[ -d "$ROOT/outputs/models/v2/grpo-merged" && ! -e "$OUT_FREEZE/models/grpo-merged" ]]; then
  mv "$ROOT/outputs/models/v2/grpo-merged" "$OUT_FREEZE/models/grpo-merged"
  echo "[archive] moved v2 GRPO merged -> $OUT_FREEZE/models/grpo-merged"
fi

if [[ -d "$ROOT/outputs/oracle-v2-collection" && ! -e "$OUT_FREEZE/oracle-v2-collection/oracle_v2_summary.json" ]]; then
  cp -a "$ROOT/outputs/oracle-v2-collection/"*.json "$OUT_FREEZE/oracle-v2-collection/" 2>/dev/null || true
  cp -a "$ROOT/outputs/oracle-v2-collection/"*.jsonl "$OUT_FREEZE/oracle-v2-collection/" 2>/dev/null || true
fi

cat > "$OUT_FREEZE/README.md" << EOF
# $STAMP archive

Frozen identifiable-gold pipeline. Do not train from the GRPO adapter here.

- SFT merged (live, hardlinked): \`outputs/models/v2/sft-merged\`
- Collapsed GRPO last-100: \`models/grpo-merged\` (gold@100 = 0)
- Eval dumps: \`evaluation/\`
- Experiment table: \`experiments/$STAMP/comparison.md\`

v3 GRPO restarts from the v2 SFT merged checkpoint with a shopping-aligned recipe.
EOF

echo "[archive] done -> $EXP_FREEZE and $OUT_FREEZE"
df -h /root/autodl-tmp | tail -1
