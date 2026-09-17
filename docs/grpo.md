# Lightweight GRPO

SFT 热启动后，每个 prompt 在 TinyWatch 里打 G=4 条轨迹，用 Reward v1 + shaping 做 group-relative advantage。不依赖 veRL。

- **v3**（已冻结）：director-first、sequence-sum PPO clip、100 step。稳定但 ≤ SFT。
- **v4**（已冻结）：`shuffled_n2` 2× 过采样、token-mean、负优势 ×0.5、200 step。总分 gold 0.46 > SFT 0.41，但 2-movie gold 仍为 0%。
- **v5**：2 部组 **3 policy + 1 Oracle**、核验命中阶梯 shaping、守卫炸/假 ID 当 collapse、KL→SFT 0.02、不过采样。ckpt 看 **n2 verified-hit**（不含 Oracle 行）。

```bash
bash scripts/grpo.sh --dry-run
# v4 冻结存档
# bash scripts/archive_v4.sh v4-20260916
# v5：从冻结的 v2 SFT 重开 GRPO（会先 archive v4）
bash scripts/run_v5_grpo_experiment.sh
```

动态采样丢掉组内可训 reward 全相同的 group。评测前用 `bash scripts/serve_model.sh outputs/models/v5/grpo-merged` 起 OpenAI 兼容接口。
