# TinyWatch-GRPO 实验结论文档

本文记录 2026-09-10～09-11 在 AutoDL（RTX 5090）上完成的一轮完整主流程：

```text
Flash 采集 → 审计入库 → Action-only LoRA SFT → 合并
→ 轻量 GRPO → 合并 → Final-100 确定性评测
```

基座 **Qwen3.5-2B**，教师 **DeepSeek-V4-Flash**（thinking 关），评测**不用 LLM Judge**。  
方法学对齐 `shopping-grpo-longhorizon`，运行时契约为 TinyWatch Environment v1 / Reward v1。

---

## 1. 目标与约束

### 目标

- 用真实 IMDb 目录训练一个能走工具链的片单 Agent（检索 → 核验 → 写入 → `finalize_watchlist`/`abort`）。
- 严格成功 = `gold_watchlist` 且 `reward_valid=true`。
- 训练数据与 `data/evaluation/tasks.jsonl`（Final-100）**零重叠**。

### 本轮执行约定

- 采集结束后先审计：合格率、拒绝原因、是否与评测题撞号。
- 只把过滤后的 `train.jsonl` / `validation.jsonl` 收进 `data/sft/`，**不用 raw 教师回复当训练集**。
- 主流程：SFT → merge → 轻量 GRPO → Final-100；不上 LLM Judge。

---

## 2. 逐步做了什么、结果如何

### 2.1 环境与代码准备

| 项 | 结果 |
|---|---|
| Python | `/root/miniconda3/bin/python`（含 CUDA torch 2.8.0+cu128） |
| 训练依赖 | `transformers` / `peft` / `accelerate` / `liger-kernel` / `fastapi` 等 |
| 基座权重 | ModelScope 下载 `Qwen/Qwen3.5-2B` → `/root/autodl-tmp/models/Qwen3.5-2B` |
| GPU | RTX 5090 约 32GB；**无需 RTX 6000** |

补齐了仓库原先缺的链路：

- `scripts/serve_openai_model.py` + `scripts/serve_model.sh`：OpenAI 兼容 serve
- `src/tinywatch_grpo/collection/local_client.py`：本地 tool-calling client
- `src/tinywatch_grpo/training/grpo/trainer.py`：进程内 G=4 + PEFT 真更新权重
- `scripts/audit_and_stage_sft.py`：采集审计并复制 SFT 数据
- `scripts/run_eval_checkpoint.sh`：serve → Eval-100 → 停服

### 2.2 Flash 教师采集

命令（目标接受 540，workers=4）：

```bash
python scripts/collect_flash_trajectories.py \
  --tasks data/tasks/sft_pool.jsonl \
  --output-dir outputs/flash-collection \
  --target-accepted 540 --workers 4 --max-steps 14
```

| 指标 | 数值 |
|---|---|
| raw 条数 | 840 |
| 接受 | **540（64.29%）** |
| 拒绝 | 300 |
| train / validation | 432 / 108 |
| held-out 撞号 | **0** |
| 与 Eval-100 重叠 | **无** |

主要拒绝原因（可叠加）：

| 原因 | 次数 |
|---|---:|
| `reward_below_threshold` | 300 |
| `missing_finalize` | 258 |
| `reward_invalid` | 107 |
| `has_error` / `status_not_done` / `trajectory_not_done` / `reward_v1_required` | 各 71 |

审计通过后执行：

```bash
python scripts/audit_and_stage_sft.py --copy --min-accepted 400
```

产物进入 `data/sft/`（action-only 清洗消息，已去掉 thinking）。

### 2.3 Action-only LoRA SFT + 合并

```bash
bash scripts/sft.sh
# BASE_MODEL=/root/autodl-tmp/models/Qwen3.5-2B
# 启用 --liger-kernel + gradient checkpointing
```

| 项 | 结果 |
|---|---|
| 训练样本 | 432 train / 108 val（全部可 tokenize） |
| 可训参数 | ~16.8M（约 0.75%） |
| train_loss | **0.1317** |
| 耗时 | ~38 分钟（2279 s） |
| 输出 | `outputs/models/sft-lora` → merge → `outputs/models/sft-merged` |

### 2.4 轻量 GRPO + 合并

从 `sft-merged` 热启动，进程内 G=4、Reward v1、动态采样丢掉组内 reward 全相同的 group：

```bash
python -u scripts/train_grpo.py \
  --max-steps 100 --max-length 4096 \
  --max-environment-steps 12 --temperature 1.0 --merge
```

| 项 | 结果 |
|---|---|
| steps | 100 |
| 有效优化步 | **43**（其余 57 为常数组跳过） |
| mean_reward（在线） | 0.189 |
| 耗时 | ~6.3 小时（22815 s） |
| 输出 | `outputs/models/grpo/adapter` → `outputs/models/grpo-merged` |

### 2.5 Final-100 确定性评测

对 base / SFT / GRPO 依次 serve + 评测（无 Judge）：

```bash
bash scripts/run_eval_checkpoint.sh baseline /root/autodl-tmp/models/Qwen3.5-2B
bash scripts/run_eval_checkpoint.sh sft outputs/models/sft-merged
bash scripts/run_eval_checkpoint.sh grpo outputs/models/grpo-merged
python scripts/build_comparison_report.py ...
```

训练/评测划分校验：`sft ∩ eval = 0`，`grpo ∩ eval = 0`。

#### 2.5.1 Final-100 是什么、怎么评

**题集**：`data/evaluation/tasks.jsonl` 共 **100** 道留出题，与 SFT 池 / GRPO 训题 `task_id` 互斥。难度约为 Easy 50 / Medium 30 / Hard 20；其中可解 99、无解 1（应用 `abort`）。

**跑法**：每个 checkpoint 用 OpenAI 兼容接口 serve，在 TinyWatch 环境里逐题 rollout（`max_steps=14`），环境真实执行工具并打 Reward v1。全程**不调用** DeepSeek-V4-Pro / Flash 当 Judge。

**严格成功（表头 gold@100）** 必须同时满足：

1. 轨迹 `status=done` 且无 infra error；
2. 终止动作为 `finalize_watchlist`；
3. `reward_type == gold_watchlist`；
4. `reward_valid == true`（关键影片已 open/compare；有导演硬约束时还做过 `view_crew`）。

注意：`valid_alternative`（Reward 0.60）**算有效终局、可进 SFT**，但**不算**严格成功。这是本轮「训练看起来像学会了、评测 gold 仍很低」的核心口径差。

**四面板含义**：

| 面板 | 看什么 | 本轮主指标 |
|---|---|---|
| Reward / 终局 | 是否严格 gold、各类 `reward_type` 计数 | `strict_gold_success_rate`（gold@100） |
| Rubric | 部数 / 总片长 / 评分 / 年份 / 类型 / 导演 / 中文译名等硬门是否全过 | `all_hard_passed_rate` |
| Behavior | 执行工具步数、守卫拒绝、是否 finalize/abort、是否 view_crew 等 | mean steps、guard 次数 |
| Infra | 接口失败、解析失败、空终局等导致轨迹无效 | `invalid_rate` |

常见 `reward_type` 直觉：

| 类型 | 含义 | Reward（示意） |
|---|---|---|
| `gold_watchlist` | 与金标片单一致且证据充分 | 1.00 |
| `valid_alternative` | 硬门满足、可核验，但不是精确金标集合 | 0.60 |
| `partial` | 部分满足 | 0–0.25 |
| `early_abort` / `loop` / `max_steps` / `wrong_watchlist` 等 | 错误终止或坏片单 | 负分居多 |
| `reward_unverifiable` / `reward_valid=false` | 片单可能「看起来对」，但缺 open/compare（或缺 view_crew） | 不计严格成功、不进 SFT |
| `None` | 没形成可读终局（常与 infra invalid 重叠） | — |

#### 2.5.2 本轮三模型数字

汇总（见 `experiments/comparison.md`）：

| model | gold@100 | hard rubric | mean steps | infra invalid |
|---|---:|---:|---:|---:|
| baseline | **0.010** | 0.180 | 3.88 | 0.470 |
| sft | **0.020** | 0.470 | 5.36 | 0.430 |
| grpo | **0.020** | **0.500** | 5.60 | **0.390** |

**上表各列含义：**

| 指标 | 全称 / 代码字段 | 含义 | 怎么读 |
|---|---|---|---|
| `model` | 被评 checkpoint | `baseline`=原始 Qwen3.5-2B；`sft`=SFT 合并模型；`grpo`=GRPO 后再合并 | — |
| **gold@100** | `strict_gold_success_rate` | Final-100 上严格成功比例：必须 `finalize` 出与题目 `gold_watchlist` **完全一致**的片单，且 `reward_valid=true` | **越高越好**；0.02 = 100 题里仅 2 题严格成功 |
| **hard rubric** | `all_hard_passed_rate` | 硬约束代码化检查全过的比例（部数、总片长≤预算、每部评分、年份、类型覆盖、导演、中文译名等），**不要求**片单等于金标 ID 集合 | **越高越好**；可高于 gold（「约束满足但不撞金标」） |
| **mean steps** | `mean_executed_tool_steps` | 平均每题实际执行的工具步数（不含被守卫直接挡掉前的无效意图时，以环境记到的执行为准） | 过低常表示早停/崩掉；适度升高通常表示更完整地走完检索→写入→终止；不是越大越好 |
| **infra invalid** | `invalid_rate` | 因接口失败、工具参数解析失败、轨迹 error 等导致**整题无效**的比例 | **越低越好**；与「答错但轨迹完整」不同，这里是基础设施/格式层失败 |

终局类型与行为补充：

| 指标 | baseline | sft | grpo |
|---|---:|---:|---:|
| `gold_watchlist` | 1 | 2 | 2 |
| `valid_alternative` | 17 | 45 | 48 |
| `None` | 47 | 43 | 39 |
| `reward_unverifiable` | 18 | 2 | 0 |
| 使用 `finalize_watchlist` 的题数 | 37 | 51 | 51 |
| 使用 `abort` 的题数 | 8 | 0 | 0 |
| 使用过 `view_crew` 的题数 | 72 | 21 | 12 |
| guard 拒绝总次数 | 72 | 42 | 43 |

**上表各行含义：**

| 指标 | 含义 | 怎么读 |
|---|---|---|
| **`gold_watchlist`** | 终局类型为精确金标成功的**题数**（与 gold@100×100 一致） | 越高越好；本轮主成功口径 |
| **`valid_alternative`** | 硬门满足、证据可核验，但片单 ≠ 金标集合的题数（Reward 通常 0.60） | 上升说明「会交合法片单」；**不能**当成严格成功 |
| **`None`** | 没有形成有效 `reward_type`（常伴随轨迹未正常结束 / infra 问题） | 越低越好 |
| **`reward_unverifiable`** | 交了片单但关键影片未充分核验（缺 `open`/`compare`，或有导演约束却未 `view_crew`）→ `reward_valid=false` | 越低越好；SFT 后从 18→2 是重要改善 |
| **使用 `finalize_watchlist` 的题数** | 100 题里至少调用过一次「提交片单」终止工具的题数 | 过低=不敢/不会收尾；过高但 gold 低=乱交卷 |
| **使用 `abort` 的题数** | 至少调用过 `abort`（声明无可行片单）的题数 | 无解题应 abort；可解题乱 abort 则差。本轮 SFT/GRPO 降到 0，说明几乎不再放弃 |
| **使用过 `view_crew` 的题数** | 轨迹中出现过「查看导演/演职员」工具的题数 | 有导演硬约束时应更高；本轮 SFT/GRPO 反而下降，可能伤害精确 gold |
| **guard 拒绝总次数** | 环境动作守卫拒绝非法操作的累计次数（如用不存在的 `movie_id`、重复非法动作等） | **越低越好**；下降说明更少「瞎调工具」 |

**和 gold / hard 易混的三点：**

1. **hard rubric 高 ≠ gold 高**：硬门全过即可计 hard；gold 还要求片单 ID 集合与金标一致。  
2. **`valid_alternative` 多 ≠ 评测胜利**：训练接受线 ≥0.55 会大量收入这类轨迹；评测主表仍看 gold@100。  
3. **infra invalid 与答错分开**：答错但轨迹完整会进 `wrong_watchlist`/`partial`/`valid_alternative` 等；infra 是「题根本没评成」。

---

## 2.6 两个训练是否达到预期、为什么

仓库对两段训练的**设计预期**并不相同，应用不同尺子量：

| 阶段 | 设计预期（本仓库口径） | 本轮是否达到 |
|---|---|---|
| **Action-only SFT** | 学会合法工具协议：一回合一个工具、先搜再核验、会 finalize；抬高可核验硬门通过与有效终局占比 | **大体达到（行为/硬门），未达到严格 gold 跃升** |
| **轻量 GRPO** | 在 SFT 热启动上用在线 Reward v1 拉开组内优劣，进一步改善约束满足与终止；默认约 100 step | **弱达到（hard/infra 略升），严格 gold 未达到「再上一层」的预期** |

### 2.6.1 SFT：预期部分达成

**达到的部分（符合「热启动 / 教协议」预期）**

- hard rubric：0.18 → **0.47**（+29pt），说明模型更能交一份「硬门过关」的片单。
- `valid_alternative`：17 → **45**；`finalize` 题数：37 → **51**；guard：72 → **42**。
- `reward_unverifiable`：18 → **2**，说明「瞎 finalize、证据不足」大幅减少——这正是 Action-only SFT + Reward 过滤要教的。
- train_loss 降到 ~0.13，540 条过滤轨迹可稳定拟合。

**未达到的部分（若把预期理解成「Final-100 严格成功明显上升」）**

- gold@100 仅 1% → **2%**（多 1 题），统计上几乎持平。
- **原因**：
  1. **训练目标与评测目标错位**：SFT 接受线是 Reward ≥ 0.55 且 `reward_valid`；校准与放量后合格轨迹里绝大多数是 `valid_alternative`（0.60），不是 `gold_watchlist`（1.00）。模型学的是「能过硬门的合法工具链」，不是「撞上唯一金标 ID 集合」。
  2. **评测尺子更严**：gold 要求片单与 `gold_watchlist` 完全一致；hard rubric 只要求约束门过关。本轮 SFT 主要抬的是后者。
  3. **基座能力与题难**：Final-100 含 30% Medium + 20% Hard；2B + 单次 LoRA 很难从近零 gold 跳到两位数。
  4. **行为副作用**：SFT 后 `view_crew` 题数从 72 降到 21、`abort` 从 8 降到 0——更敢 finalize、更少核验导演/放弃，有利于凑出 `valid_alternative`，不利于精确 gold 与无解题。

**一句话**：SFT **达到了「工具可用 + 硬门可过」的阶段预期**，**没有达到「严格金标成功率显著提升」的终局预期**——后者本来就更偏 GRPO / 数据对齐要解决的问题，且本轮 SFT 数据本身就很少教 gold。

### 2.6.2 GRPO：预期基本未达成（仅边际改善）

**设计上希望看到的**：相对 SFT，gold@100 或至少 hard/终止质量再上一截；在线 mean reward 有可用的组内对比信号。

**实际**：

- gold@100：**仍为 0.02**（与 SFT 相同，仍是 2 题）。
- hard：0.47 → **0.50**（+3pt）；infra：0.43 → **0.39**；mean steps：5.36 → 5.60。
- `valid_alternative`：45 → 48；`view_crew` 进一步降到 12；`loop` 从 5 升到 10。
- 训练过程：100 step 里仅 **43** 次有效优化，**57** 次因组内 reward 全相同被动态采样丢掉；在线 mean_reward ≈ 0.19。

**为什么没达预期**：

1. **有效更新太少**：过半 step 无梯度，等价于远弱于「满血 100 step」的轻量 GRPO。
2. **奖励与评测仍不对齐**：在线信号大量是 0 / 0.60 / 负分；优化更偏向「多打出 valid_alternative、少崩」，对「精确命中 gold ID」几乎无监督。
3. **从 SFT 起点方差不足**：温度 1.0 仍常出现四条轨迹同为 0.60 或同为 0 → 被跳过。
4. **算力与步数预算轻**：~6 小时墙钟、G=4、max_env_steps=12、无 veRL 大批量并行；相对购物项目里更完整的 GRPO 配方，本轮是「能跑通的轻量版」，不足以扭转 gold。
5. **可能强化了错误偏好**：更少 `view_crew`、更多 `loop`，说明策略在「快 finalize / 重复试」上有漂移，不利于严格成功。

**一句话**：GRPO **完成了工程预期（真更新权重、可合并、可评测）**，**未完成效果预期（严格成功相对 SFT 再提升）**；观测到的是 hard/infra 的边际收益，以及 gold 平台期。

### 2.6.3 对照总表

| 期望 | SFT | GRPO（相对 SFT） |
|---|---|---|
| 工具协议可用 | ✅ 明显 | ≈ 维持 |
| 硬门通过率上升 | ✅ 0.18→0.47 | △ 0.47→0.50 |
| infra / 守卫变好 | ✅ 部分 | △ infra 略好 |
| 严格 gold@100 明显上升 | ❌ 仅 +1 题 | ❌ 持平 |
| 与设计文档「Baseline→SFT→GRPO 递进」一致 | 行为递进成立 | gold 递进不成立 |

---


## 3. 中间遇到的问题与解决办法

### 3.1 教师一轮多工具导致采集接受率偏低（校准期）

- **现象**：早期校准接受率约 26%，大量 `multiple_tool_calls`。
- **原因**：Flash 常并行打多个 tool call；旧 loop 直接整条判失败。
- **解决**：对齐 shopping 项目，`_enforce_serial_tool_call` 只保留第一个、其余记入 `tool_call_truncations`。重校准接受率升至约 **54%**，随后放量采集。

### 3.2 Hugging Face 下载权重失败

- **现象**：`HF_ENDPOINT=hf-mirror` + xet 报 401；禁用 xet 后大文件仍卡住。
- **解决**：改用 **ModelScope** `snapshot_download('Qwen/Qwen3.5-2B')`，软链到 `/root/autodl-tmp/models/Qwen3.5-2B`。

### 3.3 pip 误升级 torch，破坏 5090 CUDA

- **现象**：装 peft 时拉到 CPU/不匹配的 torch 2.14。
- **解决**：`--no-deps` 安装 transformers/peft/accelerate，固定已有 `torch 2.8.0+cu128`。

### 3.4 SFT 长序列 OOM（词表 ~248k）

- **现象**：约第 54/162 step，`logits.float()` 再申请约 28GB 失败。
- **解决**：对齐 shopping，启用 **`--liger-kernel`**（融合 LM-head+CE）+ `LossOnlyEvalTrainer(skip_logits)` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。重跑后显存约 8GB，SFT 完成。

### 3.5 可选加速 kernel 装不上

- **现象**：`causal-conv1d` / `flash-linear-attention` 因本机 CUDA 12.8 与 PyTorch 编译 CUDA 13.0 不匹配编不过。
- **解决**：放弃；继续用参考实现。不影响正确性，只是慢一些。

### 3.6 本地 GRPO / serve 多轮模板报错

- **现象**：`TypeError: Can only get item pairs from a mapping`（第二轮起）。
- **原因**：Qwen chat template 对 `arguments|items`，但消息里是 OpenAI 风格 JSON **字符串**。
- **解决**：`messages_for_chat_template()` 在 `apply_chat_template` 前把 arguments 解析成 dict；消息落盘仍保持字符串以兼容 OpenAI schema。

### 3.7 GRPO 训练反传 OOM

- **现象**：前几步后 30GB+ OOM。
- **解决**：
  - `max_length=4096`、`max_environment_steps=12`
  - LoRA 开启 gradient checkpointing、`use_cache=False`
  - 逐步 `backward` + `torch.cuda.empty_cache`
  - rollout / train 之间清缓存  
  之后 100 step 跑通。

### 3.8 GRPO 日志缓冲与常数组过多

- **现象**：nohup 长时间看不到 `[grpo]`；大量 `skipped_constant_group`。
- **解决**：`PYTHONUNBUFFERED=1` + `print(..., flush=True)`；动态采样按设计跳过全相同 reward；`temperature=1.0` 增加组内方差。最终有效优化步 43/100。

### 3.9 仓库原 GRPO 脚本只记 reward、不更新权重

- **现象**：`train_grpo.py` 结尾注明需 endpoint logprobs 才接 PEFT。
- **解决**：实现进程内 `TransformersToolClient` rollout + `sequence_logprob` + AdamW 更新 LoRA，并支持 `--merge`。

### 3.10 是否要租 RTX 6000

- **结论：不需要。** 5090 上 SFT ~8GB、GRPO rollout ~5–6GB。瓶颈是 **串行环境 rollout 的墙钟时间**，不是显存。

---

## 4. 关键产物路径

| 类别 | 路径 |
|---|---|
| 采集 raw / 审计 | `outputs/flash-collection/{raw,accepted,rejected,train,validation,audit,metadata}.jsonl|json` |
| SFT 数据 | `data/sft/train.jsonl`, `data/sft/validation.jsonl` |
| SFT 模型 | `outputs/models/sft-lora`, `outputs/models/sft-merged` |
| GRPO | `outputs/models/grpo/{adapter,history.jsonl,grpo_summary.json}`, `outputs/models/grpo-merged` |
| 评测 | `outputs/evaluation/{baseline,sft,grpo}/summary.json` |
| 对比与归档 | `experiments/comparison.md`, `experiments/{baseline,sft,grpo}/` |

---

## 5. 结论与后续建议

### 结论

1. **端到端主流程已跑通**：真实 IMDb 目录 + Flash 过滤轨迹 + Action-only SFT + 轻量 GRPO + Final-100 四面板确定性评测（无 LLM Judge）。
2. **Final-100 要分清两把尺子**：`gold@100` 要求精确金标；`hard rubric` 只要求硬门。本轮进步主要落在后者。详见 §2.5、§2.6。
3. **SFT：阶段预期大体达成，终局 gold 预期未达成**——工具协议与硬门（0.18→0.47）明显变好，但 gold 仅 1%→2%；根因是合格数据多为 `valid_alternative`，与评测严格成功不对齐。
4. **GRPO：工程跑通，效果预期基本未达成**——相对 SFT，gold 持平；hard/infra 仅边际改善；100 step 中 57 次常数组跳过，有效优化不足，且奖励仍不指向精确金标。
5. 因此「Baseline → SFT → GRPO」在**行为/硬门**上有递进，在**严格 gold**上尚未形成递进。

### 建议的下一步（未在本轮执行）

1. **按 Final-100 badcase 拆 gold 失败**：差一部、导演未 `view_crew`、年份/类型门、过早 finalize、无解题未 abort。
2. **抬高 SFT 中 `gold_watchlist` 占比**（或单独一条 gold-only 课程），减少「只学 0.60 凑单」。
3. **加强 GRPO 有效信号**：降常数组比例（难度混合、温度/采样）、加长有效 step，必要时对齐购物项目更完整配方。
4. **压 infra invalid**（仍约 39–47%）：优先修空终局与解析失败。
5. 装齐线性注意力 CUDA 扩展后，缩短 SFT/GRPO 墙钟时间。

---

## 6. 时间线（摘要）

| 阶段 | 大约耗时 | 状态 |
|---|---|---|
| Flash 采集（540 accepted） | 数小时 | 完成 |
| 审计 + 入库 SFT | 分钟级 | 完成 |
| LoRA SFT + merge | ~0.6 h | 完成 |
| 轻量 GRPO + merge | ~6.3 h | 完成 |
| Final-100 ×3 | ~4.3 h | 完成 |

整体于 2026-09-11 凌晨完成对比报告生成（`ALL_EVAL_DONE`）。
