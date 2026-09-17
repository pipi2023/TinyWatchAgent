# v5: Oracle-mix GRPO on frozen v2 SFT

V4 (`experiments/v4/`) is frozen. n2-focused oversampling beat SFT on 1-movie
gold but **2-movie gold stayed 0%** and 2-movie invalid rose to 41%. v5 does
**not** regenerate tasks or SFT. It restarts GRPO from `outputs/models/v2/sft-merged`.

Changes versus v4:

| knob | v4 | v5 |
|---|---|---|
| n=2 sampling | 2× oversample | **natural mix** (`n2_oversample=1`) |
| n=2 group | G=4 policy | **3 policy + 1 Oracle gold** |
| collapse | format / no_tool_call | **+ guard-spam + illegal movie_id** |
| shaping | Jaccard + hit-fraction | **verified-hit staircase** on n≥2 |
| KL | unwired (0) | **0.02 to SFT** (`disable_adapter`) |
| n=2 temperature | 0.7 | **1.0** (n=1 stays 0.7) |
| steps / env | 200 / 16 | **100 / 14** (align with eval) |
| ckpt select | 0.5·gold + 0.5·n2_gold | **n2 verified-hit** (oracle rows ignored) |

```bash
bash scripts/run_v5_grpo_experiment.sh
```
