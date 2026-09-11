# Reward v1

确定性终局奖励，只看片单里的影片与环境证据，不用 LLM Judge。

## 硬门

部数、总片长 ≤ 预算、每部评分、年份窗口、类型覆盖、导演硬约束、可选「须有中文译名」。

## 档位

| 结局 | Reward | valid |
|---|---:|---|
| `gold_watchlist` | 1.00 | true |
| `valid_alternative` | 0.60 | true |
| `partial` | 0–0.25 | true |
| `correct_abort` | 0.50 | true |
| `early_abort` | -0.35 | true |
| `max_steps` | -0.50 | true |
| `loop` | -0.65 | true |
| `wrong_watchlist` / `incomplete_watchlist` | -0.85 | true |
| 关键影片未 open/compare（或导演约束未 view_crew） | 0.00 | **false** |

`reward_valid=false` 的轨迹不进 SFT 正样本。严格评测成功还要求 `finalize_watchlist` 且 `reward_type=gold_watchlist`。
