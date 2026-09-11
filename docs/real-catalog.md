# 真实目录（IMDb）

默认 `data/catalog/catalog.json` 由官方非商业 TSV 派生。CPU 单测使用仓库内 50 部真实名作 fixture（`mini_catalog()`），不访问网络。

## 数据源

- [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)
- 文件：`title.basics` / `title.ratings` / `title.akas` / `title.crew` / `name.basics`
- 用途：研究用 Agent 环境，不转售、不镜像全量 dump

## 什么是真的

| 字段 | 来源 |
|---|---|
| movie_id、英文/原名、年份、片长、类型 | title.basics |
| 评分、票数 | title.ratings |
| 中文译名 | title.akas（region CN/TW/HK/MO 或 language zh） |
| 导演 | title.crew + name.basics |

没有虚构价格。总片长预算是对目录内真实 `runtimeMinutes` 求和。

## 导入

```bash
python scripts/import_imdb_catalog.py \
  --output data/catalog/catalog.json \
  --raw-dir outputs/imdb-raw \
  --max-movies 5000

# 断网重放（dump 已在 raw-dir）
python scripts/import_imdb_catalog.py --skip-download

# 仅 fixture，给 CI / 无网机器
python scripts/import_imdb_catalog.py --mini
```
