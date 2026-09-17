# Experiments

- `v1-20260911/` — frozen first full pipeline (Flash SFT → light GRPO, gold@100 = 2%).
- `v2-20260914/` — frozen identifiable-gold recipe. SFT gold@100 = 41%; last-100 GRPO collapsed to 0%.
- `v2/` — pointer at the v2 freeze (`conclude.md`). Do not overwrite.
- `v3-20260915/` — frozen shopping-aligned GRPO (stable, gold@100 = 40%, ≤ SFT).
- `v3/` — pointer at the v3 freeze. Do not overwrite.
- `v4/` — frozen n2-focused GRPO (gold@100 = 46%, 2-movie gold = 0%). Do not overwrite.
- `v5/` — Oracle-mix GRPO restart from v2 SFT. Do not overwrite v1–v4.
- `baseline/` `sft/` `grpo/` — copies of the latest completed comparison.
