#!/usr/bin/env bash
# Freeze v3 (shopping-aligned GRPO ≤ SFT). Keep v2 SFT live for v4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-v3-20260915}"
EXP_FREEZE="$ROOT/experiments/$STAMP"
OUT_FREEZE="$ROOT/outputs/archive/$STAMP"

if [[ -d "$EXP_FREEZE" && -f "$EXP_FREEZE/comparison.md" ]]; then
  echo "[archive] experiments/$STAMP already exists, skipping experiment copy"
else
  mkdir -p "$ROOT/experiments"
  cp -a "$ROOT/experiments/v3" "$EXP_FREEZE"
  echo "[archive] copied experiments/v3 -> $EXP_FREEZE"
fi

mkdir -p "$OUT_FREEZE/models" "$OUT_FREEZE/evaluation"

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

for label in v3-grpo; do
  if [[ -d "$ROOT/outputs/evaluation/$label" ]]; then
    copy_or_link "$ROOT/outputs/evaluation/$label" "$OUT_FREEZE/evaluation/$label"
  fi
  if [[ -f "$ROOT/outputs/evaluation/${label}-serve.log" ]]; then
    cp -a "$ROOT/outputs/evaluation/${label}-serve.log" "$OUT_FREEZE/evaluation/"
  fi
done

# Move v3 GRPO weights out of the live tree to free space for v4.
if [[ -d "$ROOT/outputs/models/v3/grpo" && ! -e "$OUT_FREEZE/models/grpo" ]]; then
  mv "$ROOT/outputs/models/v3/grpo" "$OUT_FREEZE/models/grpo"
  echo "[archive] moved v3 GRPO adapter -> $OUT_FREEZE/models/grpo"
elif [[ -d "$ROOT/outputs/models/v3/grpo" ]]; then
  echo "[archive] freeze already has grpo; leaving live copy in place"
fi
if [[ -d "$ROOT/outputs/models/v3/grpo-merged" && ! -e "$OUT_FREEZE/models/grpo-merged" ]]; then
  mv "$ROOT/outputs/models/v3/grpo-merged" "$OUT_FREEZE/models/grpo-merged"
  echo "[archive] moved v3 GRPO merged -> $OUT_FREEZE/models/grpo-merged"
fi

cat > "$OUT_FREEZE/README.md" << EOF
# $STAMP archive

Frozen shopping-aligned GRPO (v3-ppo-clip). Do not train from this adapter.

- Init SFT (still live): \`outputs/models/v2/sft-merged\`
- V3 GRPO selected ckpt: \`models/grpo-merged\` (Final-100 gold = 0.40, ≤ SFT 0.41)
- Eval dump: \`evaluation/v3-grpo/\`
- Experiment table: \`experiments/$STAMP/comparison.md\`

v4 GRPO restarts from the same v2 SFT with n2-focused sampling and length-norm PPO.
EOF

# Pointer conclude on live v3 if missing.
if [[ ! -f "$ROOT/experiments/v3/conclude.md" ]]; then
  cat > "$ROOT/experiments/v3/conclude.md" << EOF
# v3 freeze (2026-09-15)

Shopping-aligned GRPO completed. **Do not overwrite these files.** Next recipe is v4 from frozen v2 SFT.

| model | gold@100 | hard rubric |
|---|---:|---:|
| baseline | 0.250 | 0.260 |
| sft | 0.410 | 0.490 |
| grpo (v3) | 0.400 | 0.450 |

Stable (no format collapse) but did not beat SFT; 2-movie gold stayed at 0%.

Archive: \`outputs/archive/$STAMP/\`. Next: \`bash scripts/run_v4_grpo_experiment.sh\`.
EOF
fi

if [[ ! -f "$EXP_FREEZE/conclude.md" ]]; then
  cp -a "$ROOT/experiments/v3/conclude.md" "$EXP_FREEZE/conclude.md"
fi

# Refresh live v3 README pointer.
cat > "$ROOT/experiments/v3/README.md" << EOF
# v3: shopping-aligned GRPO on frozen v2 SFT

Frozen copy: \`experiments/$STAMP/\`. Weights: \`outputs/archive/$STAMP/\`. Do not overwrite.

V3 fixed v2 format collapse but Final-100 stayed ≤ SFT (gold 0.40 vs 0.41). See \`conclude.md\`.

Next recipe: \`bash scripts/run_v4_grpo_experiment.sh\`.
EOF
cp -a "$ROOT/experiments/v3/README.md" "$EXP_FREEZE/README.md"

echo "[archive] done -> $EXP_FREEZE and $OUT_FREEZE"
df -h /root/autodl-tmp | tail -1
