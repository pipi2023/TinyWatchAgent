# V1 frozen experiment (2026-09-10 ~ 09-11)

Complete first pipeline on Qwen3.5-2B:

Flash collection → Action-only LoRA SFT → lightweight GRPO → Final-100.

| model | gold@100 | hard rubric | infra invalid |
|---|---:|---:|---:|
| baseline | 0.010 | 0.180 | 0.470 |
| sft | 0.020 | 0.470 | 0.430 |
| grpo | 0.020 | 0.500 | 0.390 |

Do not overwrite these files. New gold@100 work lives under `experiments/v2-*` and `outputs/models/v2/`.

Artifacts:

- `data/archive/v1-20260911/` — tasks, SFT jsonl, GRPO/eval splits
- `outputs/archive/v1-20260911/` — merged models, eval dumps, Flash collection
- `conclude.md` — full writeup copied from `docs/conclude.md` at freeze time
