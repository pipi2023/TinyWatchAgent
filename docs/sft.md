# LoRA SFT

Base：`Qwen/Qwen3.5-2B`（关 thinking）。只在 assistant / tool_call token 上计算 loss。

```bash
bash scripts/sft.sh
```

默认 LoRA rank 16、alpha 32，启用 `--liger-kernel` 避免 Qwen3.5 大词表 logits OOM。合并后再作为 GRPO 起点：

```text
outputs/models/sft-lora/   adapter
outputs/models/sft-merged/ standalone checkpoint
```

未明确要求不要启动训练或 merge。
