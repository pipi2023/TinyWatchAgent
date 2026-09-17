#!/usr/bin/env bash
# Freeze v4 (n2-focused GRPO: beat SFT on 1-movie, 2-movie gold stayed 0%).
# Keep v2 SFT live for v5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${1:-v4-20260916}"
EXP_FREEZE="$ROOT/experiments/$STAMP"
OUT_FREEZE="$ROOT/outputs/archive/$STAMP"

if [[ -d "$EXP_FREEZE" && -f "$EXP_FREEZE/comparison.md" ]]; then
  echo "[archive] experiments/$STAMP already exists, skipping experiment copy"
else
  mkdir -p "$ROOT/experiments"
  cp -a "$ROOT/experiments/v4" "$EXP_FREEZE"
  echo "[archive] copied experiments/v4 -> $EXP_FREEZE"
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

for label in v4-grpo; do
  if [[ -d "$ROOT/outputs/evaluation/$label" ]]; then
    copy_or_link "$ROOT/outputs/evaluation/$label" "$OUT_FREEZE/evaluation/$label"
  fi
  if [[ -f "$ROOT/outputs/evaluation/${label}-serve.log" ]]; then
    cp -a "$ROOT/outputs/evaluation/${label}-serve.log" "$OUT_FREEZE/evaluation/"
  fi
done

if [[ -d "$ROOT/outputs/models/v4/grpo" && ! -e "$OUT_FREEZE/models/grpo" ]]; then
  mv "$ROOT/outputs/models/v4/grpo" "$OUT_FREEZE/models/grpo"
  echo "[archive] moved v4 GRPO adapter -> $OUT_FREEZE/models/grpo"
elif [[ -d "$ROOT/outputs/models/v4/grpo" ]]; then
  echo "[archive] freeze already has grpo; leaving live copy in place"
fi
if [[ -d "$ROOT/outputs/models/v4/grpo-merged" && ! -e "$OUT_FREEZE/models/grpo-merged" ]]; then
  mv "$ROOT/outputs/models/v4/grpo-merged" "$OUT_FREEZE/models/grpo-merged"
  echo "[archive] moved v4 GRPO merged -> $OUT_FREEZE/models/grpo-merged"
fi

cat > "$OUT_FREEZE/README.md" << EOF
# $STAMP archive

Frozen n2-focused GRPO (v4). Do not train from this adapter.

- Init SFT (still live): \`outputs/models/v2/sft-merged\`
- V4 GRPO selected ckpt: \`models/grpo-merged\` (Final-100 gold = 0.46 vs SFT 0.41; 2-movie gold = 0%)
- Eval dump: \`evaluation/v4-grpo/\`
- Experiment table: \`experiments/$STAMP/comparison.md\`

v5 GRPO restarts from the same v2 SFT with Oracle mix, verified-hit shaping, and KL to SFT.
EOF

if [[ ! -f "$ROOT/experiments/v4/conclude.md" ]]; then
  cat > "$ROOT/experiments/v4/conclude.md" << 'EOF'
# v4 freeze (2026-09-16)

n2-focused GRPO completed. **Do not overwrite these files.** Next recipe is v5 from frozen v2 SFT.

| model | gold@100 | hard rubric | mean steps | infra invalid |
|---|---:|---:|---:|---:|
| baseline | 0.250 | 0.260 | 5.16 | 0.220 |
| sft | 0.410 | 0.490 | 7.32 | 0.150 |
| grpo (v4) | 0.460 | 0.500 | 6.77 | 0.230 |

Beat SFT on overall gold (+5pt) via 1-movie (65.6%→75.4%). 2-movie gold stayed at 0%; 2-movie invalid rose to 41% (`too_many_guard_rejections`). Online: 1 n2 gold in 448 rollouts.

Archive: `outputs/archive/v4-20260916/`. Next: `bash scripts/run_v5_grpo_experiment.sh`.
EOF
fi

if [[ ! -f "$EXP_FREEZE/conclude.md" ]]; then
  cp -a "$ROOT/experiments/v4/conclude.md" "$EXP_FREEZE/conclude.md"
fi

cat > "$ROOT/experiments/v4/README.md" << EOF
# v4: n2-focused GRPO on frozen v2 SFT

Frozen copy: \`experiments/$STAMP/\`. Weights: \`outputs/archive/$STAMP/\`. Do not overwrite.

V4 beat SFT on 1-movie gold (overall 0.46 vs 0.41) but 2-movie gold stayed 0%. See \`conclude.md\`.

Next recipe: \`bash scripts/run_v5_grpo_experiment.sh\`.
EOF
cp -a "$ROOT/experiments/v4/README.md" "$EXP_FREEZE/README.md"

echo "[archive] done -> $EXP_FREEZE and $OUT_FREEZE"
df -h /root/autodl-tmp | tail -1
