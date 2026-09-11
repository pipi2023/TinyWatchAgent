# Lightweight GRPO

SFT 热启动后，每个 prompt 在 TinyWatch 里打 G=4 条轨迹，用 Reward v1 做 group-relative advantage。默认 100 step。不依赖 veRL。

```bash
bash scripts/grpo.sh --dry-run
# 进程内 G=4 rollout + PEFT LoRA 更新（默认 100 step，结束后合并到 grpo-merged）
bash scripts/grpo.sh
```

动态采样丢掉组内 reward 全相同的 group。评测前用 `bash scripts/serve_model.sh outputs/models/grpo-merged` 起 OpenAI 兼容接口。
