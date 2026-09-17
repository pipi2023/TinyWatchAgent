# v2 freeze (2026-09-14)

Identified-gold recipe completed. **Do not overwrite these files.** GRPO recipe work continues as v3 from the frozen v2 SFT merged checkpoint.

| model | gold@100 | hard rubric | mean steps | infra invalid |
|---|---:|---:|---:|---:|
| baseline | 0.250 | 0.260 | 5.16 | 0.220 |
| sft | 0.410 | 0.490 | 7.32 | 0.150 |
| grpo (last-100) | 0.000 | 0.010 | 2.05 | 0.980 |

## What worked

Director + year in the public query made gold learnable: baseline 1%→25%, SFT 2%→41% versus v1. SFT gold is 66% on 1-movie tasks and 3% on 2-movie tasks.

## Why GRPO last-100 is not a policy

Online gold stayed ~30–42% through step 50, then format-collapsed: eval 91/100 `no_tool_call` with chat-template leakage. Causes: LR 1e-5 without PPO clip, temperature 1.0, `max_length=4096` dropping long groups, last checkpoint only.

Archive: `outputs/archive/v2-20260914/`. Next run: `bash scripts/run_v3_grpo_experiment.sh`.
