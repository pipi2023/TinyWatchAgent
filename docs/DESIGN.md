# TinyWatch-GRPO 设计说明

方法学对齐 shopping-grpo-longhorizon：`Baseline → SFT → GRPO → Evaluation`。  
基座 Qwen3.5-2B，教师 AutoDL DeepSeek-V4-Flash，评测只用确定性四面板。

## 为什么不用旅游

旅游旧版（已归档为 TinyTripAgent-travel）验证过 OSM 可以拿到真实 POI 名和坐标，但：

- 房价、门票、库存几乎不在 OSM 里，只能按星级/主题**推断**
- 携程/Booking 等 OTA 接口收费或禁止批量抓取

这会让「预算」这个核心硬门变成假数据，和购物项目里真实价格不对齐。

## 为什么用电影

| 需求 | IMDb 非商业 TSV |
|---|---|
| 费用 | 官方 `datasets.imdbws.com`，无 API key |
| 真实性 | 片名、年份、片长、评分、票数、类型、导演均为公开事实 |
| 中文 | `title.akas` 含 CN/TW/HK 译名 |
| 冻结 | 一次下载后离线，训练不再访问网络 |
| 结构 | 检索 → 打开详情 → 核验导演 → 比较 → 写入片单 → 终止，对齐购物长程工具链 |

片长总和 ≈ 购物预算；导演 ≈ 品牌；类型 ≈ 品类；评分 ≈ 质量门槛。

放弃过的低成本候选：OpenAlex 论文（更学术、中文查询弱）、Open Food Facts（仍像购物）、MusicBrainz（约束偏少）。

## 环境摘要

- 目录：约 5k 部高票电影（`numVotes≥8000`，1970–2024，片长 70–210 分钟）
- 工具 8 个，`max_steps=14`
- Reward v1：部数 / 总片长 / 评分 / 年份 / 类型 / 导演 / 中文译名
- 严格成功：`gold_watchlist` + `reward_valid=true`
- 未核验（未 open/compare，或有导演约束却未 view_crew）→ `reward_valid=false`，不进 SFT
