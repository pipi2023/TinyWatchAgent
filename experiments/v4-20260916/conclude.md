# v4 freeze (2026-09-16)

n2-focused GRPO completed. **Do not overwrite these files.** Next recipe is v5 from frozen v2 SFT.

| model | gold@100 | hard rubric | mean steps | infra invalid |
|---|---:|---:|---:|---:|
| baseline | 0.250 | 0.260 | 5.16 | 0.220 |
| sft | 0.410 | 0.490 | 7.32 | 0.150 |
| grpo (v4) | 0.460 | 0.500 | 6.77 | 0.230 |

Beat SFT on overall gold (+5pt) via 1-movie (65.6%→75.4%). 2-movie gold stayed at 0%; 2-movie invalid rose to 41% (`too_many_guard_rejections`). Online: 1 n2 gold in 448 rollouts.

Archive: `outputs/archive/v4-20260916/`. Next: `bash scripts/run_v5_grpo_experiment.sh`.
