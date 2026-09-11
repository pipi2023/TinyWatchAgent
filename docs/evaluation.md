# Evaluation

Final-100 只跑确定性四面板，不调用 DeepSeek-V4-Pro，也不默认用 Flash 逐条打分。

1. Reward / 终局（`gold_watchlist` + `reward_valid`）
2. 代码化 Rubric（部数/片长/评分/年份/类型/导演）
3. Behavior（步数、守卫拒绝、finalize/abort）
4. Infra（轨迹是否因接口/解析失败无效）

```bash
bash scripts/evaluate.sh sft
python scripts/build_comparison_report.py \
  --baseline outputs/evaluation/baseline/summary.json \
  --sft outputs/evaluation/sft/summary.json \
  --grpo outputs/evaluation/grpo/summary.json
```
