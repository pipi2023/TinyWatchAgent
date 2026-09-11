# AutoDL 接入

教师 API 走控制台的 **OpenAI 兼容** 接口。密钥只放环境变量或未入库的 `.env`。

`calibrate_teacher_cost.py` 和 `collect_flash_trajectories.py` 会读取仓库根目录 `.env`（已 gitignore）。已 export 的环境变量优先，不会被 `.env` 覆盖。

```bash
cp .env.example .env
# 编辑 .env：
# TEACHER_BASE_URL=https://<AutoDL 兼容 endpoint>/v1
# TEACHER_API_KEY=<平台密钥>
# TEACHER_MODEL=deepseek-v4-flash
```

也可以不用文件，直接 export：

```bash
export TEACHER_BASE_URL=<AutoDL 兼容 endpoint，通常含 /v1>
export TEACHER_API_KEY=<平台密钥>
export TEACHER_MODEL=deepseek-v4-flash
```

采集与校准：

```bash
python scripts/calibrate_teacher_cost.py --limit 50
python scripts/collect_flash_trajectories.py \
  --tasks data/tasks/sft_pool.jsonl \
  --output-dir outputs/flash-collection \
  --target-accepted 400 \
  --workers 4 \
  --max-steps 14
```

约定：

- 默认关闭 thinking（`thinking: disabled`）。
- 采集用无卡实例；不要空烧训练 GPU。
- `raw.jsonl` 可断点续跑，按 `task_id` 去重。
- 接受阈值默认 Reward ≥ 0.55 且 `reward_valid=true`。
- 接受率 <25% 时先修 schema/prompt/Easy 比例，不要放量。
