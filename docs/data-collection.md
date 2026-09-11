# Data collection

Flash 是主教师。Oracle 只用于 M0 冒烟和 API 不可用时的应急。

```bash
python scripts/collect_flash_trajectories.py \
  --tasks data/tasks/sft_pool.jsonl \
  --output-dir outputs/flash-collection \
  --target-accepted 400 \
  --workers 4
```

产物：

```text
outputs/flash-collection/
  raw.jsonl
  accepted.jsonl
  rejected.jsonl
  train.jsonl / validation.jsonl
  metadata.json   # 含 SHA-256
```

验收：`reward_valid=true` 且 Reward ≥ 0.55；去掉 thinking；与 Eval-100 `task_id` 零重叠。
