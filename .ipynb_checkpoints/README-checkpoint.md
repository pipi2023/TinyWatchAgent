# TinyWatch-GRPO

面向**约束选片 / 片单规划 Agent** 的低成本后训练与评测仓库。

方法学对齐 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon)：基座 **Qwen3.5-2B**，教师 **AutoDL DeepSeek-V4-Flash**，评测只用确定性四面板、**不用 LLM Judge**。

电影目录来自 **IMDb 官方非商业 TSV**（一次下载、零 API 密钥）。类型、年份、片长、评分、导演、中文译名都是真实字段；片长总和对应购物任务里的预算，导演对应品牌，类型对应品类，评分对应质量门槛。

```text
IMDb 冻结目录 → DeepSeek-V4-Flash 采轨迹 → Reward v1 过滤
→ Action-only SFT → Qwen3.5-2B LoRA → 轻量 GRPO → Final-100
```

运行时契约：TinyWatch Environment v1 · Reward v1 · observation v1 · tool schema v1。

更细的实验记录见 [docs/conclude.md](docs/conclude.md)，设计取舍见 [docs/DESIGN.md](docs/DESIGN.md)。

---

## 项目目的

训练一个能走**真实工具链**的片单 Agent，而不是只会生成推荐文本。

用户给出自然语言约束（类型、年份、最低评分、总片长预算、导演、是否需要中文译名等）。Agent 必须在冻结的 IMDb 目录上检索、打开详情、核验导演、比较、写入草稿，最后调用 `finalize_watchlist` 或 `abort`。

核心目标：

- 用真实目录把购物项目的长程工具链迁移到电影选片：检索 → 核验 → 写入 → 终止。
- 严格成功 = `finalize_watchlist` 得到 `gold_watchlist` 且 `reward_valid=true`。
- 训练数据与 `data/evaluation/tasks.jsonl`（Final-100）**零重叠**。
- 评测全程确定性，不调用 DeepSeek-V4-Pro / Flash 当 Judge。

为什么不用旅游：OSM 能给真实 POI，但房价/门票几乎只能推断，预算硬门会变成假数据。IMDb 非商业 TSV 能一次下载、离线冻结，且字段都是公开事实。详见 [docs/DESIGN.md](docs/DESIGN.md)。

---

## 环境与奖励

目录约 **5000** 部高票电影（`numVotes≥8000`，1970–2024，片长 70–210 分钟，全部带中文译名）。检索为 BM25，observation 截断，`max_steps=14`。

| 工具 | 作用 |
|---|---|
| `search_movies` | 按关键词 / 类型 / 年份 / 评分检索 |
| `open` | 打开 observation 中出现的影片详情 |
| `view_crew` | 核验导演（有导演硬约束时必须调用） |
| `compare` | 并排比较 2–4 部已出现的影片 |
| `add_to_watchlist` / `view_draft` | 写入 / 查看草稿片单 |
| `finalize_watchlist` / `abort` | 合法终止 |

Reward v1 只看片单与环境证据，硬门包括：部数、总片长 ≤ 预算、每部评分、年份窗口、类型覆盖、导演、可选中文译名。

| 结局 | Reward | 说明 |
|---|---:|---|
| `gold_watchlist` | 1.00 | 与金标片单一致且证据充分 |
| `valid_alternative` | 0.60 | 硬门满足、可核验，但不是精确金标集合 |
| `partial` | 0–0.25 | 部分满足 |
| `correct_abort` | 0.50 | 无解题正确放弃 |
| 其它错误终止 / 坏片单 | 负分居多 | `early_abort` / `loop` / `max_steps` / `wrong_watchlist` 等 |
| 未 open/compare（或缺 `view_crew`） | 0.00，`reward_valid=false` | 不计严格成功、不进 SFT |

`valid_alternative` **可进 SFT**，但 **不算** 严格成功。这是「训练看起来像学会了、评测 gold 仍很低」的核心口径差。完整规则见 [docs/reward-v1.md](docs/reward-v1.md)。

---

## 实验过程

本轮主流程于 2026-09-10～09-11 在 AutoDL（RTX 5090）上跑通。下面按数据、训练、评测说明实际做了什么。

### 1. 数据集如何收集

#### 1.1 冻结 IMDb 目录

从 [IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/) 下载 `title.basics` / `title.ratings` / `title.akas` / `title.crew` / `name.basics`，派生 `data/catalog/catalog.json`。原始 dump 只放在 `outputs/imdb-raw/`（gitignore），训练不再访问网络。

```bash
python scripts/import_imdb_catalog.py          # 全量冻结
python scripts/import_imdb_catalog.py --mini   # CI / 无网 fixture
```

#### 1.2 合成互斥任务

从冻结目录合成三份 `task_id` 互斥题集（seed=42）：

| 划分 | 路径 | 条数 | 用途 |
|---|---|---:|---|
| SFT / Flash 池 | `data/tasks/sft_pool.jsonl` | 1000 | 教师采集 |
| GRPO 训题 | `data/grpo/train.jsonl` | 400 | 在线强化学习 |
| Eval holdout | `data/evaluation/tasks.jsonl` | 100 | Final-100，永不进训练 |

每题含自然语言 `query`、结构化约束、`gold_watchlist`、难度（Easy / Medium / Hard）以及是否可解。Final-100 约为 Easy 50 / Medium 30 / Hard 20，其中可解 99、无解 1（应 `abort`）。

```bash
python scripts/generate_tasks.py
```

#### 1.3 教师轨迹（Flash 为主）

Oracle 只用于 M0 冒烟或 API 不可用时的应急。正式 SFT 数据来自 **DeepSeek-V4-Flash**（thinking 关闭）在 TinyWatch 环境里的真实 rollout。

```bash
# 先估接受率（建议 ≥25% 再放量）
python scripts/calibrate_teacher_cost.py --limit 50

# 本轮实际采集：目标接受 540，workers=4
python scripts/collect_flash_trajectories.py \
  --tasks data/tasks/sft_pool.jsonl \
  --output-dir outputs/flash-collection \
  --target-accepted 540 --workers 4 --max-steps 14
```

接受线：`reward_valid=true` 且 Reward ≥ 0.55；去掉 thinking；与 Eval-100 撞号的轨迹一律丢弃。本轮结果：

| 指标 | 数值 |
|---|---|
| raw 条数 | 840 |
| 接受 | **540（64.29%）** |
| 拒绝 | 300 |
| train / validation | 432 / 108 |
| 与 Eval-100 重叠 | **0** |

主要拒绝原因（可叠加）：`reward_below_threshold` 300、`missing_finalize` 258、`reward_invalid` 107。校准期曾因教师一轮多工具导致接受率约 26%，后来只保留第一个 tool call，接受率升至约 54% 再放量。

审计通过后才入库，**不用 raw 教师回复当训练集**：

```bash
python scripts/audit_and_stage_sft.py --copy --min-accepted 400
```

产物进入 `data/sft/`，消息已做成 action-only（只在 assistant / tool_call token 上算 SFT loss）。

### 2. 如何训练

流水线是 **Baseline → Action-only LoRA SFT → 轻量 GRPO**。基座权重用 ModelScope 下载 `Qwen/Qwen3.5-2B`，放到 `/root/autodl-tmp/models/Qwen3.5-2B`。

#### 2.1 Action-only LoRA SFT

只监督合法工具协议：一回合一个工具、先搜再核验、会 finalize。默认 LoRA rank 16、alpha 32，启用 `--liger-kernel` + gradient checkpointing，避免 Qwen3.5 大词表 logits OOM。

```bash
bash scripts/sft.sh
# BASE_MODEL 默认 /root/autodl-tmp/models/Qwen3.5-2B
```

本轮：432 train / 108 val 全部可 tokenize；可训参数约 16.8M（0.75%）；`train_loss=0.1317`；约 38 分钟。输出 `outputs/models/sft-lora`，合并为 `outputs/models/sft-merged`。

#### 2.2 轻量 GRPO

从 `sft-merged` 热启动。每个 prompt 在环境里打 **G=4** 条轨迹，用 Reward v1 做 group-relative advantage；组内 reward 全相同则跳过（动态采样）。不依赖 veRL。

```bash
python -u scripts/train_grpo.py \
  --max-steps 100 --max-length 4096 \
  --max-environment-steps 12 --temperature 1.0 --merge
# 或：bash scripts/grpo.sh
```

本轮：100 step 中有效优化 **43** 步（57 次常数组跳过）；在线 mean_reward ≈ 0.189；约 6.3 小时。输出 `outputs/models/grpo/adapter`，合并为 `outputs/models/grpo-merged`。

### 3. 如何测评

Final-100（`data/evaluation/tasks.jsonl`）对每个 checkpoint 用 OpenAI 兼容接口 serve，在环境里逐题 rollout（`max_steps=14`），环境真实执行工具并打 Reward v1。

**严格成功（gold@100）** 必须同时满足：轨迹 `status=done` 且无 infra error；终止动作为 `finalize_watchlist`；`reward_type == gold_watchlist`；`reward_valid == true`。

四面板：

| 面板 | 看什么 | 主指标 |
|---|---|---|
| Reward / 终局 | 是否严格 gold、各类 `reward_type` | `strict_gold_success_rate`（gold@100） |
| Rubric | 部数 / 片长 / 评分 / 年份 / 类型 / 导演 / 中文译名是否全过 | `all_hard_passed_rate` |
| Behavior | 步数、守卫拒绝、finalize/abort、`view_crew` | mean steps、guard 次数 |
| Infra | 接口失败、解析失败、空终局 | `invalid_rate` |

```bash
bash scripts/run_eval_checkpoint.sh baseline /root/autodl-tmp/models/Qwen3.5-2B
bash scripts/run_eval_checkpoint.sh sft outputs/models/sft-merged
bash scripts/run_eval_checkpoint.sh grpo outputs/models/grpo-merged
python scripts/build_comparison_report.py \
  --baseline outputs/evaluation/baseline/summary.json \
  --sft outputs/evaluation/sft/summary.json \
  --grpo outputs/evaluation/grpo/summary.json
```

划分校验：`sft ∩ eval = 0`，`grpo ∩ eval = 0`。对比表见 `experiments/comparison.md`。

---

## 实验结果与分析

### Final-100 三模型对比

| model | gold@100 | hard rubric | mean steps | infra invalid |
|---|---:|---:|---:|---|
| baseline | **0.010** | 0.180 | 3.88 | 0.470 |
| sft | **0.020** | 0.470 | 5.36 | 0.430 |
| grpo | **0.020** | **0.500** | 5.60 | **0.390** |

终局与行为补充：

| 指标 | baseline | sft | grpo |
|---|---:|---:|---:|
| `gold_watchlist` | 1 | 2 | 2 |
| `valid_alternative` | 17 | 45 | 48 |
| `None`（常与 infra invalid 重叠） | 47 | 43 | 39 |
| `reward_unverifiable` | 18 | 2 | 0 |
| 使用 `finalize_watchlist` 的题数 | 37 | 51 | 51 |
| 使用 `abort` 的题数 | 8 | 0 | 0 |
| 使用过 `view_crew` 的题数 | 72 | 21 | 12 |
| guard 拒绝总次数 | 72 | 42 | 43 |

要把两把尺子分开看：`gold@100` 要求精确命中金标 ID 集合；`hard rubric` 只要求约束门过关。本轮进步主要落在后者。

### SFT：阶段预期大体达成，严格 gold 未跃升

SFT 的设计目标是教合法工具协议、抬高可核验硬门通过率，而不是直接把 gold 打到两位数。

达到的部分：

- hard rubric：0.18 → **0.47**（+29pt），更能交一份硬门过关的片单。
- `valid_alternative` 17 → 45；`finalize` 题数 37 → 51；guard 72 → 42。
- `reward_unverifiable` 18 → 2，「瞎 finalize、证据不足」明显减少。

未达到的部分：gold@100 仅 1% → **2%**（多 1 题）。原因：

1. **训练目标与评测目标错位**：SFT 接受线是 Reward ≥ 0.55；合格轨迹绝大多数是 `valid_alternative`（0.60），很少是 `gold_watchlist`（1.00）。模型学的是「能过硬门的合法工具链」，不是「撞上唯一金标 ID」。
2. **评测尺子更严**：gold 要求片单与金标完全一致；hard rubric 只要求约束门。
3. **基座与题难**：Final-100 含 30% Medium + 20% Hard；2B + 单次 LoRA 很难从近零 gold 跳到两位数。
4. **行为副作用**：`view_crew` 72 → 21、`abort` 8 → 0——更敢 finalize、更少核验导演/放弃，有利于凑出 `valid_alternative`，不利于精确 gold 与无解题。

### GRPO：工程跑通，效果仅边际改善

相对 SFT，gold@100 **仍为 0.02**。hard 0.47 → 0.50，infra 0.43 → 0.39，`valid_alternative` 45 → 48。同时 `view_crew` 降到 12，`loop` 从 5 升到 10。

未达「再上一层」的原因：

1. **有效更新太少**：100 step 里 57 次因组内 reward 全相同被丢掉。
2. **奖励仍不指向精确金标**：在线信号大量是 0 / 0.60 / 负分，优化更偏向「多打出 valid_alternative、少崩」。
3. **组内方差不足**：温度 1.0 仍常出现四条轨迹同为 0.60 或同为 0。
4. **轻量配方**：G=4、`max_env_steps=12`、串行 rollout，墙钟约 6 小时，不足以扭转 gold。

因此 **Baseline → SFT → GRPO** 在行为 / 硬门上有递进，在严格 gold 上尚未形成递进。

| 期望 | SFT | GRPO（相对 SFT） |
|---|---|---|
| 工具协议可用 | 明显改善 | 维持 |
| 硬门通过率 | 0.18 → 0.47 | 0.47 → 0.50 |
| infra / 守卫 | 部分变好 | infra 略好 |
| 严格 gold@100 明显上升 | 仅 +1 题 | 持平 |

### 建议的下一步（本轮未执行）

1. 按 Final-100 badcase 拆 gold 失败：差一部、导演未 `view_crew`、年份/类型门、过早 finalize、无解题未 abort。
2. 抬高 SFT 中 `gold_watchlist` 占比（或单独一条 gold-only 课程）。
3. 加强 GRPO 有效信号：降常数组比例、加长有效 step。
4. 压 infra invalid（仍约 39–47%）：优先修空终局与解析失败。

问题与修复（教师多工具、权重下载、SFT/GRPO OOM、chat template 等）见 [docs/conclude.md](docs/conclude.md) §3。

---

## 如何运行

未明确要求时，本仓库**不会**启动训练、merge、Flash 采集或 100 题评测。下面命令都需要你自己执行。

### 环境

默认复用 `/root/autodl-tmp/miniconda/tinywatch` 的 Python 3.12。训练脚本也接受 `/root/miniconda3/bin/python`（本轮 SFT/GRPO 实际用的环境，含 CUDA torch）。统一用环境变量指定：

```bash
export TINYWATCH_PYTHON=/root/autodl-tmp/miniconda/tinywatch/bin/python
# 训练 / 评测若要用已装好 CUDA 的解释器：
# export TINYWATCH_PYTHON=/root/miniconda3/bin/python

cd /root/autodl-tmp/TinyWatchAgent
bash scripts/setup.sh
```

`setup.sh` 会 `pip install -e .`、导入目录、生成任务、跑 CPU smoke。只跑 CPU 单测、不下载 IMDb 全量时：

```bash
TINYWATCH_MINI_CATALOG=1 bash scripts/setup.sh
```

教师密钥用环境变量或仓库根目录 `.env`（已 gitignore），见 [docs/autodl.md](docs/autodl.md)：

```bash
cp .env.example .env
# 编辑 TEACHER_BASE_URL / TEACHER_API_KEY / TEACHER_MODEL=deepseek-v4-flash
```

基座权重（SFT/GRPO 需要）：

```bash
# 本轮用 ModelScope，避免 HF 镜像大文件失败
# snapshot_download('Qwen/Qwen3.5-2B') → /root/autodl-tmp/models/Qwen3.5-2B
export BASE_MODEL=/root/autodl-tmp/models/Qwen3.5-2B
```

可选依赖：`pip install -e '.[sft]'` 或 `.[grpo]`。本机若已有匹配的 CUDA torch，安装 transformers/peft/accelerate 时建议 `--no-deps`，避免把 torch 升坏。

CPU 契约检查与单测：

```bash
python scripts/smoke_tinywatch.py
python -m pytest tests/ -q
```

### 端到端流水线

| 阶段 | 入口 | 说明 |
|---|---|---|
| 目录 | `python scripts/import_imdb_catalog.py` | IMDb 真实目录；`--skip-download` 可断网重放 |
| 任务 | `python scripts/generate_tasks.py` | 互斥 SFT / GRPO / Eval 划分 |
| Oracle 冒烟 | `python scripts/collect_oracle_trajectories.py --limit 10` | M0，CPU，¥0 |
| Flash 校准 | `python scripts/calibrate_teacher_cost.py --limit 50` | 需 API，先估接受率 |
| Flash 采集 | `python scripts/collect_flash_trajectories.py --target-accepted 400` | 可断点续跑；未明确要求不要跑 |
| 审计入库 | `python scripts/audit_and_stage_sft.py --copy --min-accepted 400` | 合格才复制到 `data/sft/` |
| Baseline | `bash scripts/run_eval_checkpoint.sh baseline $BASE_MODEL` | 评基座工具能力 |
| SFT | `bash scripts/sft.sh` | Action-only LoRA + merge |
| GRPO | `bash scripts/grpo.sh` | G=4，默认 100 step + merge |
| Eval-100 | `bash scripts/run_eval_checkpoint.sh NAME MODEL_PATH` | 无 Pro Judge |
| 对比报告 | `python scripts/build_comparison_report.py ...` | 写 `experiments/comparison.md` |

已有合并权重、只评测时，也可先手动 serve 再评：

```bash
bash scripts/serve_model.sh outputs/models/grpo-merged
# 另一终端
bash scripts/evaluate.sh grpo
```

`run_eval_checkpoint.sh` 会自动起服、跑完 Final-100、再停服，一般优先用它。

### 关键产物

| 类别 | 路径 |
|---|---|
| 采集 raw / 审计 | `outputs/flash-collection/` |
| SFT 数据 | `data/sft/train.jsonl`, `data/sft/validation.jsonl` |
| SFT 模型 | `outputs/models/sft-lora`, `outputs/models/sft-merged` |
| GRPO | `outputs/models/grpo/{adapter,history.jsonl,grpo_summary.json}`, `outputs/models/grpo-merged` |
| 评测 | `outputs/evaluation/{baseline,sft,grpo}/summary.json` |
| 对比与归档 | `experiments/comparison.md`, `experiments/{baseline,sft,grpo}/` |

---

## 费用锚点（设计）

- IMDb 目录：官方 TSV，一次下载，之后离线冻结。
- Flash 教师软顶 ¥30–60；先 50 条校准。
- GPU Baseline→SFT→GRPO100→Eval 约 ¥20–40（按 4090 估；本轮实际在 RTX 5090 上完成，SFT 显存约 8GB，瓶颈是串行 rollout 墙钟而不是显存）。
- 评测不用 DeepSeek-V4-Pro。

本轮墙钟大约：Flash 采集数小时；SFT+merge ~0.6 h；GRPO+merge ~6.3 h；Final-100 ×3 ~4.3 h。

---

## 文档

1. [主题与数据](docs/DESIGN.md) — 为什么用 IMDb
2. [真实目录](docs/real-catalog.md)
3. [数据采集](docs/data-collection.md)
4. [SFT](docs/sft.md)
5. [GRPO](docs/grpo.md)
6. [评测](docs/evaluation.md)
7. [Reward v1](docs/reward-v1.md)
8. [AutoDL](docs/autodl.md)
9. [实验结论](docs/conclude.md) — 完整跑通记录、指标、问题与修复

IMDb 许可：[non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)。不要再发布原始 dump。
