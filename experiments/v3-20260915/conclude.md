# v3 freeze (2026-09-15)

Shopping-aligned GRPO completed. **Do not overwrite these files.** Next recipe is v4 from frozen v2 SFT.

| model | gold@100 | hard rubric |
|---|---:|---:|
| baseline | 0.250 | 0.260 |
| sft | 0.410 | 0.490 |
| grpo (v3) | 0.400 | 0.450 |

Stable (no format collapse) but did not beat SFT; 2-movie gold stayed at 0%.

Archive: `outputs/archive/v3-20260915/`. Next: `bash scripts/run_v4_grpo_experiment.sh`.
