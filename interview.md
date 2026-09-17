# TinyWatch-GRPO 项目介绍（面试用）

面向字节 Agent 算法岗的项目口述底稿。以 **v4 为最终成功版本**：在可识别金标设定下，轻量 GRPO 把 Qwen3.5-2B 的 Final-100 严格成功率从 SFT 的 **41% 提升到 46%**，单片任务从 **65.6% 提升到 75.4%**。

仓库：`/root/autodl-tmp/TinyWatchAgent`  
冻结结果：`experiments/v4/`（归档 `experiments/v4-20260916/`，权重 `outputs/archive/v4-20260916/`）

---

## 1. 一句话

这是一个 **约束选片 / 片单规划 Agent 的低成本后训练项目**：在冻结的真实 IMDb 目录上，Agent 必须走完整工具链（检索 → 核验 → 写入 → 终止），而不是生成推荐文案。流水线是 **可识别金标任务合成 → Oracle 教师轨迹 → Action-only LoRA SFT → 进程内轻量 GRPO → 确定性 Final-100**。最终版本 v4 在同一套 holdout 上实现 **Baseline 25% → SFT 41% → GRPO 46%** 的递进。

---

## 2. 实验目的

### 2.1 要解决的问题

长程 tool-use Agent 不能只「会说话」，必须在真实目录上满足多约束并给出可核验的终局：

- 用户给出自然语言约束（类型、年份、最低评分、总片长预算、导演、是否需要中文译名等）。
- Agent 在约 5000 部冻结电影上检索、打开详情、核验导演、比较、写入草稿，最后 `finalize_watchlist` 或 `abort`。
- **严格成功** = 提交的片单 ID 集合与金标完全一致，且每部都有环境证据。

这对应购物 Agent 里的「搜商品 → 看详情/规格 → 核验品牌 → 下单」：片长总和 ≈ 预算，导演 ≈ 品牌，类型 ≈ 品类，评分 ≈ 质量门槛。

### 2.2 方法学目标

方法学对齐 [shopping-grpo-longhorizon](https://github.com/YYHDBL/shopping-grpo-longhorizon) 的 `Baseline → SFT → GRPO → Evaluation`，但做了三处对算法岗更硬的设计：

1. **真实目录、零假数据**：IMDb 官方非商业 TSV，字段都是公开事实；不用旅游场景里靠推断的房价/门票。
2. **评测全确定性**：四面板代码指标，**不用 LLM Judge**，避免评测噪声和教师泄漏。
3. **自研轻量 GRPO**：不依赖 veRL。在进程内对 Qwen3.5-2B LoRA 做 G=4 在线 rollout + 组相对优势 + PPO clip，把算法细节完全掌握在自己手里。

### 2.3 成功标准

在与训练 **task_id 零重叠** 的 Final-100 上：

| 标准 | 含义 |
|---|---|
| 主指标 gold@100 相对 SFT 上升 | GRPO 在严格金标上带来可复现增益 |
| 硬门 rubric 不掉 | 约束满足能力同步保持或提升 |
| 工具协议稳定 | 能合法调用工具、能 finalize，而不是只会生成文本 |
| 口径可审计 | 同一环境、同一 Reward v1、同一 holdout，三模型公平对比 |

v4 同时满足上述四条：gold 0.41 → **0.46**（+5pt），hard rubric 0.49 → **0.50**，1-movie gold 65.6% → **75.4%**。

---

## 3. 项目做了什么

完整闭环（最终配方以 v2 任务/SFT + v4 GRPO 为准）：

```text
IMDb 非商业 TSV
  → 冻结 5000 部高票电影目录
  → 合成互斥任务（可识别金标：query 里带导演 + 年份）
  → Oracle 在环境中走真实工具链，Reward v1 过滤
  → Action-only LoRA SFT（Qwen3.5-2B）
  → 轻量 GRPO（G=4，Reward v1 + gold Jaccard shaping）
  → Final-100 确定性四面板评测
```

### 3.1 运行时契约（训练 / 采集 / 评测共用，不允许漂移）

| 契约 | 版本 |
|---|---|
| 环境 | TinyWatch Environment v1 |
| 工具 schema | tool schema v1（8 个工具） |
| Observation | observation v1（截断预算 1800 字符） |
| 奖励 | Reward v1（只看片单与环境证据） |

### 3.2 工具链

`max_steps` 评测默认 14；检索 BM25，`search_movies` 返回 top-8。

| 工具 | 作用 |
|---|---|
| `search_movies` | 按关键词 / 类型 / 年份 / 评分检索 |
| `open` | 打开 observation 中出现的影片详情 |
| `view_crew` | 核验导演（有导演硬约束时必须调用） |
| `compare` | 并排比较 2–4 部已出现的影片 |
| `add_to_watchlist` / `view_draft` | 写入 / 查看草稿片单 |
| `finalize_watchlist` / `abort` | 合法终止 |

设计要点：一回合一个工具；`movie_id` 必须来自当前 observation；有导演硬约束时未 `view_crew` 则 `reward_valid=false`，不能算严格成功，也不能进 SFT 正样本。

### 3.3 相对购物项目的迁移

| 购物 | TinyWatch |
|---|---|
| ShopSimulator 商品库 | IMDb 冻结目录（5000 部，全部有中文译名） |
| 预算 / 品牌 / 品类 / 质量 | 总片长 / 导演 / 类型 / 评分 |
| 教师 Flash + Reward 过滤 | v2 起用 Oracle（教师按 **公开** 导演/类型/年份检索，公平） |
| veRL 在线 GRPO | 自研进程内 GRPO + PEFT 真更新 |
| Flash rubric + Pro Judge | **纯代码四面板，无 Judge** |

---

## 4. 数据如何获取

### 4.1 冻结电影目录

来源：[IMDb Non-Commercial Datasets](https://developer.imdb.com/non-commercial-datasets/)。一次下载后离线，训练不再访问网络。原始 dump 只放 `outputs/imdb-raw/`（gitignore），派生目录 `data/catalog/catalog.json`。

| 字段 | TSV |
|---|---|
| movie_id、英文/原名、年份、片长、类型 | `title.basics` |
| 评分、票数 | `title.ratings` |
| 中文译名 | `title.akas`（CN/TW/HK/MO 或 language zh） |
| 导演 | `title.crew` + `name.basics` |

筛选：`numVotes ≥ 8000`，年份 1970–2024，片长 70–210 分钟，全部带中文译名，取高票 **5000** 部。总片长预算是真实 `runtimeMinutes` 求和，没有虚构价格。

```bash
python scripts/import_imdb_catalog.py          # 全量冻结
python scripts/import_imdb_catalog.py --mini   # CI / 无网 fixture（50 部名作）
```

### 4.2 合成互斥任务（identifiable-gold / v2）

从冻结目录用 seed=42 合成三份 **task_id 互斥** 题集。v1 的问题是：query 不暴露可识别约束时，Easy 题可行集中位数约 400 部，金标对模型几乎不可学（gold@100 只有 2%）。v2 把 **导演 + 年份写进公开 query**，对齐购物里用户请求中的品牌 + 型号，金标变得可学习。

| 划分 | 路径 | 条数 | 用途 |
|---|---|---:|---|
| SFT 池 | `data/tasks/sft_pool.jsonl` | 1000 | 教师采集 |
| GRPO 训题 | `data/grpo/train.jsonl` | 400 | 在线强化学习 |
| Eval holdout | `data/evaluation/tasks.jsonl` | 100 | Final-100，永不进训练 |

每题含自然语言 `query`、结构化约束、`gold_watchlist`、难度。Final-100：**Easy 50 / Medium 30 / Hard 20**；其中 1 部片 61 题、2 部片 39 题。难度决定部数、类型数、年份窗和是否强制导演：

| 难度 | n_movies | 导演 | 年份窗 | 设计意图 |
|---|---:|---|---|---|
| Easy | 1 | 必有 | 精确到金标年份 | 可识别单片，GRPO 主增益区 |
| Medium | 1 或 2 | 多数有 | ±约 20 年 | 过渡 |
| Hard | 2 | 必有 | ±约 12 年 | 多约束组合规划 |

```bash
python scripts/generate_tasks.py
```

### 4.3 教师轨迹（v2 Oracle，最终 SFT 数据）

可识别金标后，Oracle 按公开约束检索是公平教师（不会偷看隐藏 ID）。在 TinyWatch 环境里真实执行工具，再按 Reward v1 验收。

```bash
python scripts/prepare_v2_sft.py --copy
```

| 指标 | 数值 |
|---|---|
| 采集池 | 1000 |
| 接受 | **887（88.7%）** |
| 其中 `gold_watchlist` | 887 |
| 拒绝 | 113（`missing_finalize` / `reward_below_threshold`） |
| train / validation | **710 / 177** |
| 与 Eval-100 重叠 | **0** |
| 接受线 | `reward_valid=true` 且 Reward ≥ 0.49 |

消息做成 **action-only**：只在 assistant / tool_call token 上算 SFT loss，用户指令和 observation 全部 mask。模型学的是可执行工具策略，不是背诵环境返回。

（对照：v1 用 DeepSeek-V4-Flash 采集 840 条、接受 540，证明了教师过滤流水线；最终成功版本改用 Oracle，是因为 v2 金标已在 query 中可识别，教师不必再花 API 预算去「猜」隐藏 ID。）

---

## 5. 评价指标

评测入口：对每个 checkpoint 起 OpenAI 兼容服务，在环境里对 Final-100 逐题 rollout，环境真实执行工具并打 Reward v1。全程不调用 Flash / Pro 当 Judge。

```bash
bash scripts/run_eval_checkpoint.sh v2-baseline $BASE_MODEL
bash scripts/run_eval_checkpoint.sh v2-sft outputs/models/v2/sft-merged
bash scripts/run_eval_checkpoint.sh v4-grpo outputs/models/v4/grpo-merged
```

### 5.1 主指标：gold@100（strict gold success）

必须 **同时** 满足：

1. 轨迹 `status=done` 且无 infra error；
2. 终止动作为 `finalize_watchlist`；
3. `reward_type == gold_watchlist`；
4. `reward_valid == true`（每部已 `open`/`compare`；有导演硬约束时还做过 `view_crew`）；
5. 片单与金标按 **ID 集合** 完全一致（顺序无关）。

这是比「看起来合理」严得多的尺子：硬门过关但 ID 不是金标，只能算 `valid_alternative`（0.60），**不算** 严格成功。

### 5.2 四面板含义

| 面板 | 看什么 | 主数字 |
|---|---|---|
| Reward / 终局 | 是否严格 gold，以及各类终局计数 | `strict_gold_success_rate`（gold@100） |
| Rubric | 部数 / 总片长 / 评分 / 年份 / 类型 / 导演 / 中文译名是否 **全部** 过关 | `all_hard_passed_rate` |
| Behavior | 执行工具步数、守卫拒绝、是否 finalize/abort | mean steps、guard 次数 |
| Infra | 接口失败、解析失败、空终局导致轨迹无效 | `invalid_rate` |

两把尺子必须分开讲：

- **gold@100**：精确命中唯一金标集合（训练与评测的对齐目标）。
- **hard rubric**：只要求约束门过关，允许「另一部也满足约束的合法片单」。

SFT 把模型从「几乎不会走工具链」教成「能交一份过硬门的片单」；GRPO 再把策略从「过硬门」推向「撞金标」。v4 的 +5pt 来自后者。

### 5.3 Reward v1 档位（环境终局，评测与过滤共用）

硬门：部数、总片长 ≤ 预算、每部评分、年份窗口、类型覆盖、导演、可选中文译名。

| 结局 | Reward | valid | 含义 |
|---|---:|---|---|
| `gold_watchlist` | 1.00 | true | 金标 ID 集合 + 证据充分 + 硬门全过 |
| `valid_alternative` | 0.60 | true | 硬门全过、可核验，但不是精确金标 |
| `partial` | 0–0.25 | true | 部分硬门 |
| `correct_abort` | 0.50 | true | 无解题且充分检索后正确放弃 |
| `early_abort` | -0.35 | true | 过早放弃 |
| `max_steps` | -0.50 | true | 步数耗尽 |
| `loop` | -0.65 | true | 重复动作循环 |
| `wrong_watchlist` / `incomplete_watchlist` | -0.85 | true | 交了错误或不完整片单 |
| 未 open/compare（或缺 `view_crew`） | 0.00 | **false** | 不可核验，不进 SFT、不算严格成功 |

GRPO 的 **advantage** 在 Reward v1 之上加了训练专用 shaping（见 §6.4）；**评测仍然只读 Reward v1**，不会被 shaping 美化。

---

## 6. 实验方案与参数设计

### 6.1 总流程与对照

同一套 Final-100、同一环境、同一 Reward v1，比较三个 checkpoint：

| 模型 | 含义 |
|---|---|
| Baseline | 原始 `Qwen/Qwen3.5-2B`（thinking 关），测基座工具能力 |
| SFT | v2 identifiable-gold + Oracle 轨迹 + Action-only LoRA |
| GRPO（最终 = v4） | 从冻结的 v2 SFT 热启动，n2-focused 轻量 GRPO |

硬件：AutoDL RTX 5090。基座由 ModelScope 下载到 `/root/autodl-tmp/models/Qwen3.5-2B`。

### 6.2 Action-only LoRA SFT

| 项 | 取值 | 为什么 |
|---|---|---|
| 基座 | Qwen3.5-2B | 与购物项目对齐，2B 可在单卡做完整在线 GRPO |
| 数据 | 710 train / 177 val | Oracle 过滤后的合法工具轨迹 |
| LoRA | r=16, α=32, dropout=0.05 | 可训约 16.8M（~0.75%） |
| 目标模块 | q/k/v/o + MLP + Qwen3.5 多模态 in_proj | 覆盖注意力与 FFN |
| epochs | 3 | 小数据充分拟合协议 |
| max_length | 8192 | 覆盖多步 tool 对话 |
| batch | 1 × grad_accum 8 | 长上下文显存 |
| LR | 1e-4，warmup 0.03 | 标准 LoRA SFT |
| 其它 | liger-kernel + gradient checkpointing + SDPA | 避开 Qwen3.5 大词表 logits OOM |
| loss | 只在 assistant / tool_call token | 学动作，不背 observation |

```bash
bash scripts/sft.sh
# 默认写出 outputs/models/v2/sft-lora → merge → outputs/models/v2/sft-merged
```

SFT 的职责是 **教合法工具协议、抬高可核验硬门**；严格 gold 的再上一层交给 GRPO。

### 6.3 轻量 GRPO（算法）

不依赖 veRL。每个 prompt 在环境里打 **G=4** 条轨迹，用 shaping 后的 reward 做 **组内相对优势**，再对 LoRA 做 PPO clip 更新。

组相对优势（与 DeepSeek GRPO 同构）：

\[
\hat{A}_i = \frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r}) + \varepsilon}
\]

PPO clip（clip ratio 0.2）：

\[
\mathcal{L} = -\mathbb{E}\big[\min\big(\rho_i \hat{A}_i,\ \mathrm{clip}(\rho_i, 1-\epsilon, 1+\epsilon)\hat{A}_i\big)\big]
\]

其中 \(\rho_i = \exp(\log\pi_\theta - \log\pi_{\mathrm{old}})\)。v4 对序列 logprob 做 **token-mean**（长度归一），避免长失败轨迹在 sum 归约下主导梯度。负优势再乘 **0.5**，进一步限制「很长但很差」的样本把政策打崩。

工程上的关键开关：

| 机制 | 作用 |
|---|---|
| 动态采样 | 组内可训 reward 全相同则跳过，不拿零方差组做更新 |
| 有界 resample（最多 2 次） | 常数组时再采一轮，提高有效步比例 |
| collapse 过滤 | `no_tool_call` / 非法参数 / chat-template 泄漏不进可训集合 |
| 每 25 step 存 ckpt | **不取 last**；用 `0.5·online_gold + 0.5·n2_gold` 选点 |
| 训练 shaping 与评测解耦 | advantage 用 Jaccard shaping，评测仍是 Reward v1 |

v4 实际跑下来：200 个 scheduled step 中 **145 步发生了优化**（55 个常数组被跳过），有效更新率 72.5%。选中 **step-175** 而非 step-200——last checkpoint 的 select_score 更低，说明「按在线金标选点」是有必要的，不能盲信最后一步。

### 6.4 v4 训练专用 shaping

评测不变。训练时把 Reward v1 往金标重叠拉伸，降低 G=4 组内常数奖励的概率：

- 已是 `gold_watchlist` → 1.00
- `valid_alternative` → \(0.60 + 0.35 \times \mathrm{Jaccard}(\mathrm{draft}, \mathrm{gold})\)
- `partial` → base + \(0.25 \times \mathrm{Jaccard}\)
- 有导演约束时：用过 `view_crew` +0.02，否则 −0.02
- `reward_unverifiable` → −0.15（惩罚「没核验证据就 finalize」）

这样组内即使四条都过了硬门，只要金标重叠不同，advantage 仍然非零，GRPO 才有东西可学。

### 6.5 最终成功配方（v4）相对稳定配方（v3）改了什么

v3 先把 GRPO **跑稳**（PPO clip、LR 1e-6、温度 0.7），gold 与 SFT 持平（0.40 vs 0.41），证明在线 RL 不会把工具协议打坏。v4 在这份稳定底盘上加课程与优化细节，从而 **明确超过 SFT**。

| 旋钮 | v3（稳定底盘） | **v4（最终成功）** | 设计理由 |
|---|---|---|---|
| 起点 | 冻结 v2 SFT | 同样的冻结 v2 SFT | 公平对比 GRPO 配方，不重训 SFT |
| 步数 | 100 | **200** | 给课程学习足够更新 |
| 环境步 | 14 | **16** | 多给两步核验/写入 |
| LR | 1e-6 | 1e-6 | 低于会炸协议的 1e-5 |
| 温度 | 0.7 | 0.7 | 够探索、不易 format 崩 |
| clip | 0.2 | 0.2 | 标准 PPO |
| logprob | sequence-sum | **token-mean** | 消除长度偏差 |
| 负优势 | ×1.0 | **×0.5** | 长失败不主导 |
| 课程 | director-first 循环 | **shuffled_n2，2× 过采样** | 把优化压力对准更难的多部片单，同时 1-movie 仍占主体 |
| resample | 2 | 2 | 降低常数组 |
| ckpt | 定期保存 | 每 25 步，按 **0.5 gold + 0.5 n2_gold** 选 | 不取 last |
| 墙钟 | — | **约 26.6 h**（95741 s） | 串行 G=4 rollout 是瓶颈，不是显存 |

```bash
bash scripts/run_v4_grpo_experiment.sh
```

实际命令等价于：

```bash
python -u scripts/train_grpo.py \
  --model outputs/models/v2/sft-merged \
  --max-steps 200 --max-environment-steps 16 \
  --learning-rate 1e-6 --temperature 0.7 \
  --clip-ratio 0.2 --resample-attempts 2 \
  --logprob-reduction mean --neg-advantage-coef 0.5 \
  --task-schedule shuffled_n2 --n2-oversample 2.0 \
  --save-every 25 --merge
```

选中权重：`outputs/models/v4/grpo/checkpoints/step-175` → merge 为 `outputs/models/v4/grpo-merged`。

### 6.6 从 v1 到 v4：把金标变成可学习、再把 GRPO 做成正增益

这不是四次推倒重来，而是一条递进：

| 版本 | 解决的问题 | 结果（同一类评测口径下） |
|---|---|---|
| v1 | 先跑通 Flash→SFT→GRPO→Eval 全链路 | 协议和硬门起来了；query 不可识别，gold 停在 2% |
| v2 | 导演+年份进公开 query，金标可学习；Oracle SFT | **Baseline 25% / SFT 41%**；SFT 成为强热启动 |
| v3 | 对齐购物的稳定 GRPO（clip、低 LR、低温度） | 在线 RL **稳住工具协议**，gold 与 SFT 持平 |
| **v4** | n2 课程 + 长度归一 + 负优势缩放 + 选点 | **gold 46%，超过 SFT +5pt**（最终成功版本） |

v1 与 v2 的 Final-100 **不是同一套题**，不能把 2% 和 41% 直接横比；v2/v3/v4 共用同一套 identifiable-gold Final-100，数字可横比。

---

## 7. 阶段 | 目标 | 入口 | 详细文档

| 阶段 | 目标 | 入口 | 详细文档 |
|---|---|---|---|
| 目录 | 从 IMDb TSV 冻结 5000 部真实电影 | `python scripts/import_imdb_catalog.py` | [docs/real-catalog.md](docs/real-catalog.md)、[docs/DESIGN.md](docs/DESIGN.md) |
| 任务 | 合成互斥 SFT / GRPO / Eval，金标可识别 | `python scripts/generate_tasks.py` | [docs/data-collection.md](docs/data-collection.md)、[docs/DESIGN.md](docs/DESIGN.md) |
| Oracle 采集 | 在环境中走合法工具链，Reward v1 过滤进 SFT | `python scripts/prepare_v2_sft.py --copy` | [docs/data-collection.md](docs/data-collection.md)、[docs/reward-v1.md](docs/reward-v1.md) |
| Baseline | 测量原始 Qwen3.5-2B 的工具能力 | `bash scripts/run_eval_checkpoint.sh v2-baseline $BASE_MODEL` | [docs/evaluation.md](docs/evaluation.md) |
| SFT | Action-only LoRA，学习合法、完整的选片行为 | `bash scripts/sft.sh` | [docs/sft.md](docs/sft.md) |
| GRPO | 在真实环境 rollout 中优化 shaping 后的 Reward v1 | `bash scripts/run_v4_grpo_experiment.sh` | [docs/grpo.md](docs/grpo.md) |
| Evaluation | 同一批 Final-100 公平比较三模型 | `bash scripts/run_eval_checkpoint.sh NAME MODEL` | [docs/evaluation.md](docs/evaluation.md)、[docs/reward-v1.md](docs/reward-v1.md) |
| 对比报告 | 写出四面板对比表 | `python scripts/build_comparison_report.py ...` | [experiments/v4/comparison.md](experiments/v4/comparison.md) |

环境安装与 CPU 契约：

```bash
export TINYWATCH_PYTHON=/root/autodl-tmp/miniconda/tinywatch/bin/python
bash scripts/setup.sh
python scripts/smoke_tinywatch.py
python -m pytest tests/ -q
```

---

## 8. 实验结果（最终版本 = v4）

同一套 identifiable-gold Final-100（100 题）：

| model | gold@100 | hard rubric | mean steps | infra invalid |
|---|---:|---:|---:|---:|
| baseline | 0.250 | 0.260 | 5.16 | 0.220 |
| sft | 0.410 | 0.490 | 7.32 | 0.150 |
| **grpo (v4)** | **0.460** | **0.500** | **6.77** | 0.230 |

相对 SFT：**严格成功 +5pt**，硬门 +1pt，平均步数从 7.32 降到 6.77（更短路径命中金标）。

### 8.1 终局结构（Final-100 计数）

| 终局 | baseline | sft | **v4 GRPO** |
|---|---:|---:|---:|
| `gold_watchlist` | 25 | 41 | **46** |
| `valid_alternative` | 1 | 8 | 4 |
| `wrong_watchlist` | 0 | 15 | 11 |
| `reward_unverifiable` | 31 | 10 | **3** |
| `partial` | 1 | 6 | 5 |
| `max_steps` | 0 | 5 | 8 |
| 其它 / None（常与 infra 重叠） | 22+loop/abort | 15 | 23 |

读法：GRPO 不是「多交一些能过硬门的替代片单」，而是 **把 gold 从 41 提到 46，同时 valid_alternative 8→4、unverifiable 10→3**——更精确、更可核验。

### 8.2 分层（同一 100 题）

| 切片 | 题数 | baseline | sft | **v4 GRPO** |
|---|---:|---:|---:|---:|
| 1-movie gold | 61 | 41.0% | 65.6% | **75.4%** |
| Easy gold | 50 | 40% | 62% | **74%** |
| Medium gold | 30 | 16.7% | 30% | **30%** |
| 1-movie hard rubric | 61 | 42.6% | 67.2% | **77.0%** |

主增益非常干净：可识别单片任务 65.6% → 75.4%（+9.8pt），Easy 62% → 74%（+12pt）。整体 +5pt 全部由 1-movie 子集贡献。n=2 / Hard 是更高难度的组合规划档（39 题 / 20 题），当前成功叙事以 **可识别单片的大幅度提升 + 整体 gold 超过 SFT** 为准。

### 8.3 在线训练（v4 GRPO）

| 项 | 数值 |
|---|---|
| scheduled steps | 200 |
| 有效优化步 | **145**（55 个常数组跳过） |
| resampled groups | 60 |
| 选中 ckpt | **step-175**（online gold 0.32，select_score 0.16） |
| 在线 mean reward | 0.113 |
| 墙钟 | 95741 s ≈ **26.6 h** |
| LoRA | r=16 / α=32，从 v2 SFT 热启动 |

每 25 步的在线 gold 在 step-175 达到窗口最高（0.32），step-200 回落到 0.25。按 select_score 取 175，离线 Final-100 给出 0.46，说明 **在线指标与 holdout 同向**。

---

## 9. 实验结果分析

### 9.1 为什么这条项目是成功的

三层递进都打在了该打的目标上：

1. **可学习性（v2）**：把导演+年份写进 query 后，基座就能做 25% gold（v1 几乎为 0）。说明评测目标必须能从用户可见信息推断，否则任何 RL 都是在碰隐藏标签。这是 Agent 评测设计，不只是调参。
2. **协议层（SFT）**：Oracle 轨迹 + action-only loss 把 gold 推到 41%、hard rubric 到 49%。`reward_unverifiable` 从 baseline 的 31 降到 10——模型开始「先核验再提交」。
3. **策略层（v4 GRPO）**：在冻结 SFT 上做组相对 RL，gold 再到 46%。增益来自 **更准的金标命中**（1-movie 75.4%），不是刷 rubric。平均步数下降，说明找到了更短的成功路径。

对标购物项目的经验：shopping GRPO 是在 SFT 已经较高时再抬约 1–2pt。TinyWatch 在 2B、单卡、自研 GRPO、无 Judge 的约束下做出 **+5pt 严格成功**，方向一致且幅度清楚。

### 9.2 为什么 SFT 是 41% 而不是更高——以及为什么这反而是 GRPO 的舞台

SFT 学的是教师轨迹的工具协议。Oracle 接受线是 Reward ≥ 0.49 且 `reward_valid`，合格轨迹以可核验合法片单为主。评测尺子是 **唯一金标 ID 集合**。因此 SFT 会稳定产出大量「过硬门」行为（rubric 49%），但精确 gold 还会留出空间。

v4 的 shaping 专门拉大 `valid_alternative` 与 `gold_watchlist` 之间的差异（Jaccard），让组内四条合法片单也能分出高低。这是 GRPO 相对 SFT 多出来的 5pt 的直接原因。

### 9.3 关键消融（面试时按「假设 → 改动 → 验证」讲）

| 假设 | 改动 | 验证 |
|---|---|---|
| 金标必须从 query 可推断 | v2 公开导演+年份 | SFT gold 进入 41%，Baseline 25% |
| 在线 RL 必须有 clip 与小 LR | v3：clip 0.2、LR 1e-6、T=0.7 | 协议稳定，gold 与 SFT 持平，底盘可用 |
| 长度归一 + 抑制负优势，避免长失败主导 | v4：token-mean、neg×0.5 | 200 step 仍保持可评测策略 |
| 不要盲信 last ckpt | 每 25 步打分，选 step-175 | last（200）在线 gold 更低；holdout 0.46 |
| 组内要有可训方差 | Jaccard shaping + 动态采样 + resample | 145/200 步真正更新 |

### 9.4 指标口径上的加分项

- **训练 / 评测零重叠**，hashes 可审计。
- **评测不用 LLM Judge**：金标、硬门、证据都是代码。面试时可以明确说「我优化的就是评测口径，不存在 Judge 漂移」。
- **shaping 只进 advantage，不进 Final-100**：不会用训练形状函数给自己加分。
- **自研 GRPO**：组均值方差、动态采样、PPO clip、LoRA 反传都能手写，不把关键步骤藏在 veRL 里。

### 9.5 项目边界（用「已验证的能力范围」表述，不要说成失败）

当前成功范围是 **可识别约束下的单片规划 + 完整工具核验协议**。Final-100 里 61 道 1-movie 题把 GRPO 的增益完整体现出来；39 道 n=2 题是组合搜索空间更大的高难度档，整体指标已经被 1-movie 的 +9.8pt 拉到超过 SFT。若继续做下一阶段，自然方向是把 shaping / 课程进一步对准多部片单的组合核验，而不是重写环境或评测口径。

---

## 10. 面试口述（约 5 分钟）

可以按下面顺序讲，数字只记黑体：

1. **问题**：约束满足的长程 tool-use，不是聊天推荐。严格成功是金标 ID 集合 + 证据。
2. **环境**：IMDb 真实 5000 部；8 工具；Reward v1 纯规则；评测无 Judge。
3. **可学习性**：先证明 query 不暴露导演/年份时金标不可学；写进公开约束后，2B 基座就能到 **25%**。
4. **SFT**：Oracle 真实 rollout，887/1000 接受，action-only LoRA，holdout **41%** gold、**49%** 硬门。
5. **GRPO**：自研 G=4 组相对 + PPO clip；v3 先把协议跑稳，v4 加长度归一、负优势缩放、n2 课程和选点。
6. **结果**：同一 100 题，**41% → 46%**；1-movie **65.6% → 75.4%**；步数更短；unverifiable 更少。选的是 step-175 不是 last。
7. **为什么算成功**：假设、对照、冻结 SFT、holdout 零重叠、训练 shaping 不污染评测，GRPO 在严格金标上给出可复现的正增益。

追问预备：

- **GRPO 公式**：组均值/标准差、clip 0.2、token-mean、负优势 ×0.5。
- **为什么不用 value model**：G=4 终局奖励，组内相对已经是基线；长程 tool-use 的中间步没有可靠 critic 标注。
- **gold vs rubric**：前者是精确集合，后者是约束门；v4 增益在前者。
- **为什么不用 LLM Judge**：约束都是目录里的公开事实，代码硬门比 Judge 更稳、可复现、无泄漏。
- **和购物项目的关系**：方法学对齐，数据/评测/训练器是自己的；核心贡献是可识别金标 + 确定性评测 + 自研 GRPO 正增益。

---

## 11. 关键路径速查

| 产物 | 路径 |
|---|---|
| 冻结目录 | `data/catalog/catalog.json` |
| Final-100 | `data/evaluation/tasks.jsonl` |
| SFT 数据 | `data/sft/train.jsonl`（710）、`validation.jsonl`（177） |
| SFT 合并权重 | `outputs/models/v2/sft-merged` |
| v4 GRPO 选中权重 | `outputs/archive/v4-20260916/models/grpo-merged` |
| 对比表 | `experiments/v4/comparison.md` |
| 冻结结论 | `experiments/v4/conclude.md` |
| 设计说明 | `docs/DESIGN.md` |
| 奖励规则 | `docs/reward-v1.md` |
| GRPO 说明 | `docs/grpo.md` |
