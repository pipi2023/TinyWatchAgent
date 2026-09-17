# v2: identifiable gold + shaped GRPO (FROZEN 2026-09-14)

Frozen copy: `experiments/v2-20260914/`. Collapsed GRPO weights: `outputs/archive/v2-20260914/`. Do not overwrite. v3 GRPO restarts from `outputs/models/v2/sft-merged`.

See `conclude.md` for the last-100 collapse. New recipe: `bash scripts/run_v3_grpo_experiment.sh`.

V1 (`experiments/v1-20260911/`) is frozen. gold@100 stayed at 2% because the
public query did not identify the hidden gold ids (easy tasks had a median of
~400 feasible movies). Shopping GRPO only adds about +1.5pt *after* SFT already
reaches ~60% strict success; it cannot invent a unique target that the query
does not determine.

v2 therefore changes the **learnability** of gold, not the Reward v1 eval rule:

1. Easy/hard tasks put **director + year** in the public constraints/query so
   gold is typically unique (shopping analog: brand + model in the user request).
2. Search accepts English genre aliases (`War` → `战争`).
3. Agent loop retries once on `no_tool_call` (v1 eval lost 35/100 this way).
4. SFT is Oracle on the new pool (fair: teacher searches public director/genre/year).
5. GRPO keeps Reward v1 for the environment, but advantages use gold-Jaccard
   shaping + one bounded resample, like shopping dynamic sampling.

Eval is a **new** Final-100 from the same catalog and seed machinery. Do not mix
v1 and v2 gold@100 numbers.

```bash
bash scripts/run_v2_gold_experiment.sh
```
