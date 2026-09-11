# Data

The production catalog is a **frozen derived subset** of the IMDb non-commercial TSV dumps. Raw dumps stay in `outputs/imdb-raw/` and are gitignored. Generated trajectories stay in `outputs/`.

| Split | Path | Notes |
|---|---|---|
| Catalog | `catalog/catalog.json` | 5000 IMDb movies, all with Chinese akas |
| Mini fixture | in-code `mini_catalog()` | 50 well-known films for CPU tests |
| Flash/SFT pool | `tasks/sft_pool.jsonl` | 1000 |
| GRPO train | `grpo/train.jsonl` | 400 |
| Eval holdout | `evaluation/tasks.jsonl` | 100 |

All splits are `task_id` disjoint. Copy Flash-accepted `train.jsonl` / `validation.jsonl` into `sft/` only after reviewing the collection audit. Never mix evaluation task IDs into SFT or GRPO.

IMDb license: [non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/). Do not republish the raw dumps.
