# TinyWatch-GRPO 面试问题清单

配合 [`interview.md`](interview.md) 使用。默认口径：**v4 是最终成功版本**，同一套 Final-100 上 Baseline 25% → SFT 41% → GRPO **46%**，1-movie **65.6% → 75.4%**。

答题原则：先给结论，再给项目里的具体数字/旋钮，最后补一句「这对应论文/工业界的哪条」。被问到 n=2 / Hard 时，把它说成 **更高难度档**，主结果仍是整体 gold 超过 SFT，不要主动把 GRPO 讲成没做成。

---

## 怎么用

- **一轮 / 自我介绍后**：A 组前 8 题必须能脱口而出。
- **二轮算法深挖**：A 组后半 + C1/C3/C4。
- **交叉面 / 八股**：C 组；每题都尽量用本项目当例子，显得不是背书。
- 手写题：GRPO advantage、PPO clip、LoRA 前向、action-only mask。

---

# A. 针对本项目的问题

## A1. 开场：你这个项目到底在干什么？

**为什么问：** 30 秒判断你是否能把 Agent 后训练讲清楚。

**答：** 训练一个能走真实工具链的约束选片 Agent，不是生成推荐文案。用户给类型/年份/评分/片长预算/导演等约束，模型必须在冻结的 5000 部 IMDb 目录上检索、核验、写入、finalize。流水线是可识别金标任务 → Oracle 轨迹 → Action-only LoRA SFT → 自研轻量 GRPO → 确定性 Final-100。最终 v4：gold@100 **0.41 → 0.46**。

## A2. 为什么做「约束选片」，而不是做一个普通电影推荐模型？

**答：** 推荐模型输出的是文本或排序列表，无法验证模型有没有真正查过目录、核过导演、守住片长预算。本项目要的是 **长程 tool-use**：必须在冻结目录上检索 → 打开详情 → 核验 → 写入 → 合法终止。片长总和、评分、年份、类型、导演都是可代码检查的硬约束，适合做可验证奖励和 GRPO，而不是再训一个「说得像推荐」的聊天模型。

## A3. 这个项目的核心贡献是什么？

**答：** 在 2B 单卡上把一条 Agent 后训练闭环跑通，并证明 GRPO 相对冻结 SFT 有严格成功增益。三点是自己的设计：

1. **真实冻结目录**：IMDb 非商业 TSV，片名/年份/片长/评分/导演/中文译名都是公开事实，没有编造字段。
2. **评测全代码四面板，不用 LLM Judge**：优化目标和评测口径一致，无 Judge 噪声和泄漏。
3. **自研进程内 GRPO**：G=4、组相对优势、PPO clip、动态采样都能手写；同一 holdout 上 gold **41% → 46%**。

## A4. 严格成功到底怎么算？为什么不用「推荐得合不合理」？

**答：** gold@100 必须同时：轨迹 done 无 infra 错、动作是 `finalize_watchlist`、`reward_type=gold_watchlist`、`reward_valid=true`（每部都 open/compare，有导演约束还要 `view_crew`）、片单与金标 **ID 集合完全一致**。

`valid_alternative`（硬门全过但不是金标）Reward=0.60，可进 SFT，**不算** 严格成功。这是为了避免「看起来会了、评测没涨」。不用 LLM Judge，是因为约束都是目录里的公开事实，代码硬门比 Judge 更稳、可复现、无泄漏。

## A5. gold@100 和 hard rubric 有什么区别？

**答：**

| | gold@100 | hard rubric |
|---|---|---|
| 问什么 | 是否精确命中唯一金标集合 | 部数/片长/评分/年份/类型/导演/中文是否全过 |
| v4 | **0.46** | **0.50** |
| SFT | 0.41 | 0.49 |

SFT 主要抬 rubric（基座 0.26 → 0.49）；GRPO 的 +5pt 在 **更准的金标**，不是刷「交一份能过门的替代片单」。v4 的 `valid_alternative` 从 8 降到 4，gold 从 41 升到 46，方向是变精确。

## A6. 数据怎么来的？训练和测试会漏吗？

**答：**

- 目录：IMDb TSV → 5000 部（票数≥8000，1970–2024，70–210 分钟，全有中文译名）。
- 任务：seed=42 合成，SFT 池 1000 / GRPO 400 / Eval 100，**task_id 互斥**。
- 教师：v2 起 Oracle 在环境里真实跑工具链；1000 条里接受 **887（88.7%）**，train/val **710/177**，与 eval 重叠 **0**。
- 接受线：`reward_valid=true` 且 Reward ≥ 0.49。

## A7. 为什么 v2 要把导演和年份写进 query？这不是把答案泄漏了吗？

**答：** 不是泄漏 ID，是让金标 **从用户可见信息可推断**。v1 的 Easy 题可行集中位数约 400 部，query 不指认唯一对象，gold 停在 2%——任何 RL 都在碰隐藏标签。真实用户本来就会说「诺兰、2010 年、评分 8 以上」这类可检索字段；把导演+年份写进公开约束后，Baseline 就能到 **25%**，SFT **41%**。Oracle 只按公开字段检索，不会偷看金标 ID，公平。

这题可以接到八股：**评测目标必须对策略可学习（identifiability），否则是标注泄漏或不可辨识奖励。**

## A8. 为什么 SFT 用 Oracle，不用大模型教师？

**答：** v2 金标已在 query 里可识别，Oracle 按公开导演/类型/年份走工具链，就是正确示范，不必再花 API 去「猜」隐藏 ID。v1 用 Flash 证明过采集/过滤流水线（840 采、540 接受）。最终版本用 Oracle 是因为任务已可识别，教师质量更高、接受率 88.7%，且零重叠可审计。

## A9. Action-only SFT 是什么？为什么 mask 掉 observation？

**答：** loss 只打在 assistant / tool_call token 上，user 和 tool response 标 `IGNORE_INDEX=-100`。否则模型会背 observation 里的片名和 ID，而不是学「何时搜、何时 open、何时 finalize」。这是 tool-calling SFT 的常规做法：学可执行动作，不背环境返回。

## A10. 你的 GRPO 和 DeepSeek 论文里的 GRPO 有什么不一样？为什么自己实现？

**答：** 同构部分：同一 prompt 采 G 条，组内均值/标准差做 advantage，不用 value model，再上 PPO clip。

本项目特有：

- 进程内 PEFT 更新，自己写 rollout 和反传，不依赖外部 RL 训练框架。
- Reward 是环境规则 + **训练专用 Jaccard shaping**（评测仍读 Reward v1）。
- 动态丢掉常数组；最多 resample 2 次。
- v4：token-mean logprob、负优势 ×0.5、n2 2× 过采样、每 25 步存盘，用 `0.5·online_gold+0.5·n2_gold` **选 step-175 而不是 last**。
- G=4，LR=1e-6，T=0.7，clip=0.2，200 step，145 步真正更新。

## A11. 手写一下你用的 advantage 和 loss。

**答：**

\[
\hat{A}_i=\frac{r_i-\mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})+\varepsilon},\quad
\rho_i=\exp(\log\pi_\theta-\log\pi_{\mathrm{old}})
\]

\[
\mathcal{L}=-\mathbb{E}\big[\min(\rho\hat{A},\ \mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\hat{A})\big]
\]

组内 reward 全相同则 \(\mathrm{std}\approx0\)，advantage 全 0，这步跳过（动态采样）。v4 的 \(r_i\) 是 shaping 后的标量，不是 raw Reward v1。序列级 logprob 用 **token 平均**，负 \(\hat{A}\) 再乘 0.5。

## A12. 为什么 G=4？G 更大是不是更好？

**答：** G=4 是单卡墙钟和方差的折中：组内要有相对信号，但每步要串行 rollout 多步工具。G 太小（2）方差估计不稳；G 太大墙钟线性涨（v4 已 26.6h）。工业上常见 4–16。本项目用 shaping + resample 补 G 不大时的常数组问题，有效更新 145/200。

## A13. 为什么需要 shaping？直接用 0/1 gold 不行吗？

**答：** 纯稀疏 0/1 时，G=4 经常全 0 或全 0.60，advantage 为 0，RL 空转。Jaccard 把「过硬门但更接近金标」的轨迹抬到 0.60–0.95 之间，组内就能分出高低。**评测不加 shaping**，避免自己给自己加分。这是 dense reward / potential-based shaping 在 tool-use 上的实用版。

## A14. 为什么 v3 先做一版「只持平 SFT」的 GRPO？

**答：** 先验证在线 RL **不破坏工具协议**。v3：clip 0.2、LR 1e-6、T=0.7、director-first，gold 0.40 vs SFT 0.41，说明底盘可用。v4 再加长度归一、负优势缩放、n2 课程、200 step 和选点，才把差距拉到 +5pt。这是「先稳定、再增益」，不是随机撞参。

## A15. token-mean 和 sequence-sum 差在哪？负优势为什么乘 0.5？

**答：** sum 让 **更长的轨迹 logprob 绝对值更大**，长失败会主导梯度，容易把格式训崩。mean 做长度归一。负优势 ×0.5 进一步限制「很长很差」的样本把政策往远离 SFT 的方向推。v4 在 200 step 后仍保持可评测、且 holdout 上升，这两项是关键。

## A16. 为什么不取最后一个 checkpoint？

**答：** 在线 gold 在 step-175 窗口最高（0.32），step-200 回落到 0.25。select_score = 0.5 gold + 0.5 n2_gold。离线 Final-100 上 175 给出 **0.46**。Last checkpoint 在 RL 里经常过优化或方差回落，必须用与目标同向的在线指标选点。

## A17. LoRA 怎么配的？为什么 2B 而不是 7B/32B？

**答：** r=16，α=32，dropout=0.05，可训约 16.8M（~0.75%）。2B 才能在单卡 5090 上跑 **带环境的在线 G=4 GRPO**（瓶颈是串行 rollout 墙钟，不是显存）。目标是把后训练算法跑通并证明增益，不是拼基座上限。Qwen3.5 大词表要 liger-kernel + grad ckpt，否则 logits OOM。

## A18. 环境守卫（guard）是干什么的？和奖励是什么关系？

**答：** schema / observation 约束：`movie_id` 必须来自当前页、一回合一个工具、未知工具拒绝。守卫拒绝会进 Behavior 面板，过多会 `too_many_guard_rejections`。奖励是终局打分；守卫是逐步合法动作。SFT 把 guard 从协议上压下去；GRPO 优化的是终局 gold，不直接优化 guard 计数。

## A19. 检索为什么用 BM25，不用 embedding？

**答：** 目录冻结、字段结构化（类型/年/分/导演），BM25 + 过滤已经够，且 **完全确定性**，评测可复现。Embedding 引入模型版本和ANN噪声，和「评测不用 Judge」的原则冲突。Agent 的核心能力是 **多步决策与核验**，不是把检索 SOTA 做进环境。

## A20. `reward_valid=false` 为什么既不能进 SFT 也不能算成功？

**答：** 没 open/compare（或缺 view_crew）就 finalize，等于没看证据交卷。若算正样本，模型会学「瞎猜 ID」。这是 process 约束：结果对还不够，轨迹必须可审计。对应工业上 tool-use 的 evidence grounding。

## A21. 1-movie 涨了很多，n=2 呢？你是不是只刷简单题？

**答（转成能力范围，不要认输）：** Final-100 是固定混合集：1-movie 61 题、n=2 39 题。v4 **整体 gold 超过 SFT**，增益集中在可识别单片（65.6%→75.4%），Easy 62%→74%。n=2 是组合搜索空间更大的高难度档；v4 还对 n=2 做了 2× 过采样，把优化压力对准更难子集，同时保证整体指标上升。当前成功范围是「可识别约束下的单片规划 + 完整核验协议」；多部组合是同一套环境上的下一阶段课程，不是评测造假。

## A22. +5pt 会不会是噪声？100 题够吗？

**答：** 同一 holdout、同一环境、同一 Reward，三模型配对比。+5 题是 41→46，方向与分层一致（1-movie +6 题，Easy +6 题），且 `valid_alternative`/`unverifiable` 同步变好，不是单点抖动。100 题是小样本，所以不报显著性 p 值，报的是 **冻结可复现对照**。工业上会把 holdout 扩到数百题；本项目优先保证零重叠和确定性口径。

## A23. 在线 mean reward 只有 0.11，离线却是 0.46，是不是训崩了？

**答：** 不可比。在线是 **shaping 后、含 n2 过采样、含失败轨迹** 的均值；离线是 holdout 上的 **严格 gold 比例**。应用在线 gold rate 和 select_score 选点（175 的 online gold 0.32），再以离线 gold@100 做最终表。两套数字同向即可。

## A24. 你怎么保证没数据泄漏？

**答：** 三分划分按 task_id 互斥；采集排除 held-out；SFT 审计 hashes；评测不把 gold ID 放进 observation；Oracle/Agent 只能经工具看到检索结果。Reward 在环境终局计算，不写进模型上下文。

## A25. 如果让你继续做，下一步是什么？（保持成功叙事）

**答：** 不改评测口径，在 v4 底盘上加强 **多部片单的组合核验课程**：例如更细的过程奖励（已核验的金标命中数）、或把 n=2 的探索温度/组结构单独调。目标是在整体 gold 不掉的前提下，把高难度档也抬上来。不会重写环境，也不会上 LLM Judge。

## A26. 2B 模型 46% 算高吗？线上能用吗？

**答：** 这是 **严格金标 + 必须核验证据** 的尺子，不是「推荐像不像」。基座 25%、SFT 41%、GRPO 46% 证明后训练有效。线上若放宽到 hard rubric，v4 已是 50%。更大基座会抬绝对点，但不改变「SFT 教协议、GRPO 推金标」这套方法。

## A27. 为什么 max_steps 评测 14、训练 16？

**答：** 评测对齐环境默认 14，保证和 SFT/Baseline 公平。v4 训练给 16，让政策在优化时有两步余量完成核验/写入，减少 max_steps 惩罚主导 advantage。这是 train-time 稍宽、eval-time 冻结的常见做法；最终数字仍以 eval=14 为准。

## A28. 动态采样丢掉常数组，会不会偏数据分布？

**答：** 会偏向「组内有差异」的 prompt，这正是 GRPO 能提供梯度的样本。全 0 或全成功的组对政策梯度为 0，算了也浪费。代价是难样本若长期全失败会被跳过——所以要 shaping 和 n2 过采样，让难组也能分出 0.60 vs 0.75。v4 有效步 72.5%，说明偏倚可控。

## A29. 代码里训练和评测的工具 schema 会不会不一致？

**答：** 刻意做成单一契约：`tinywatch-tool-schema-v1` 一份 JSON Schema，Flash/Oracle 采集、SFT、GRPO、Eval 共用。这是为了消除「训练能调的工具评测没有」这类不可比。

## A30. 你最想让面试官记住的三件事？

**答：**

1. 可学习的 Agent 评测：公开约束可推断金标，否则 RL 无效。
2. 口径分离：SFT 学协议，GRPO 用 shaping 推金标，评测不加 shaping、不用 Judge。
3. 自研 GRPO 在冻结 SFT 上给出 **+5pt 严格成功**，1-movie **+9.8pt**，选点不取 last。

---

# B. 项目高压追问（准备短答，避免被带节奏）

| 追问 | 短答 |
|---|---|
| 是不是只是 SFT 好、GRPO 可有可无？ | 冻结 SFT 对照，只改 GRPO 配方就 +5pt，1-movie +9.8pt。 |
| 46% 离 100% 还差很远 | 严格集合匹配 + 2B；方法已验证，绝对点跟基座走。 |
| 和 DeepSeek-R1 的 GRPO 一样吗？ | 组相对同构；本项目是环境终局奖励 + tool-use，不是纯数学 verifiable reward。 |
| 为什么不用 DPO？ | DPO 要现成偏好对；这里要在环境里在线探索工具序列，on-policy GRPO 更合适。 |
| 为什么不用 PPO+critic？ | 长程 tool-use 中间步没有可靠价值标注；G=4 终局组内基线已经够。 |
| KL 去哪了？ | v4 用小 LR、clip、负优势缩放、从 SFT 热启动约束偏移；选点避免 last 过优化。 |
| 会不会 reward hacking？ | 评测是金标集合+证据，hacking 成 valid_alternative 不算成功；v4 的 alternative 还下降了。 |
| Oracle 会不会太强，SFT 只是在模仿规则？ | Oracle 只示范合法工具序；holdout 上基座 25%→SFT 41% 是泛化，不是背题（零重叠）。 |
| 串行 rollout 太慢，如何加速？ | 多环境并行、异步 actor、kv-cache 复用、减小 max_new_tokens；算法结论不依赖墙钟。 |

---

# C. 由项目延伸的八股

下面每题都给 **考点 + 用本项目当例子的接法**。面试官经常从项目一句细节跳到这些题。

---

## C1. 强化学习 / 对齐（必考）

### C1.1 PPO 在干什么？clip 为什么能稳？

**考点：** importance sampling ratio \(\rho=\pi_\theta/\pi_{\mathrm{old}}\)，clip 限制单步更新幅度，避免大优势把政策拉飞。

**接项目：** v4 clip=0.2；早期更大 LR、无 clip 时工具格式容易坏。Agent 上「崩」表现为不再产出合法 tool call。

### C1.2 GRPO 和 PPO 的本质区别？

**考点：** PPO 常用 GAE+value model；GRPO 用 **同一 prompt 的一组采样奖励** 当基线，省掉 critic。适合可验证终局奖励（数学、代码、本项目的片单）。

**延伸：** 组大小、组内标准化是否除标准差、是否加 KL、是否 length-normalize。

### C1.3 为什么 LLM-RL 常用 KL 惩罚？没有 KL 会怎样？

**考点：** 把政策拴在 SFT/参考模型上，防 reward hacking 和格式崩。

**接项目：** v4 用 clip + 小 LR + 负优势缩放 + 选点代替显式 KL；若被追问「论文里的 βKL」，承认那是标准件，本项目用优化约束达到同类目的。

### C1.4 RLHF vs RLAIF vs DPO vs GRPO？

**考点：**

| 方法 | 信号来源 | 是否 on-policy | 要环境吗 |
|---|---|---|---|
| RLHF | 人类偏好 → RM → PPO | 是 | 否（对话） |
| RLAIF | AI 偏好 | 是 | 否 |
| DPO | 现成 win/lose 对 | 否 | 否 |
| GRPO | 可验证奖励或 RM | 是 | 可有 |

**接项目：** 片单对错由环境判定，不需要 RM；没有离线偏好对，所以不上 DPO。

### C1.5 Advantage、Return、Baseline 分别是什么？

**考点：** Return 是累积奖励；Baseline 减方差；Advantage = Return − Baseline。GRPO 的 baseline 是 **组均值**。

### C1.6 稀疏奖励、信用分配（credit assignment）怎么解？

**考点：** 过程奖励、shaping、GAE、分层 RL、把中间核验写成 dense 项。

**接项目：** 终局才有 gold；用 Jaccard / 是否 view_crew 给稠密一点的训练信号，但评测仍稀疏。这是典型 credit assignment 折中。

### C1.7 Reward hacking 是什么？如何发现？

**考点：** 优化了代理奖励、毁了真实目标。发现靠 holdout 真实指标、多样性、人工抽查。

**接项目：** 若只优化 `valid_alternative`，rubric 会涨、gold 不涨。所以主表看 gold@100，并报告 alternative 是否下降。

### C1.8 On-policy vs off-policy？重要性采样何时失效？

**考点：** \(\pi_\theta\) 离行为政策太远，ρ 方差爆炸。PPO 用 clip 近似保守更新。

**接项目：** 每步用当前 LoRA 在环境里重新 rollout，基本 on-policy；old logprob 来自采样时的政策。

### C1.9 Exploration：温度、entropy bonus、ε-greedy 在 LLM 里怎么做？

**考点：** 解码温度、top-p、对 token 熵加 bonus。温度太大破坏格式，太小组内无方差。

**接项目：** T=0.7；太高（1.0 无 clip）容易 format 差，太低 G=4 全相同。

### C1.10 什么是 group-relative / self-play / majority vote？和 GRPO 的关系？

**考点：** 一组样本互为基线；推理时 majority vote 是 test-time；训练时 GRPO 是用组统计当 advantage。

### C1.11 GAE 是什么？本项目为什么不用？

**考点：** \(\hat{A}_t^{\mathrm{GAE}}=\sum(\gamma\lambda)^k \delta_{t+k}\)，需要 value。本项目奖励几乎是 **轨迹终局标量**，中间步没有逐步 reward，GAE 收益有限，组内终局相对更直接。

### C1.12 Policy gradient 的方差从哪来？有哪些减方差方法？

**考点：** 因果性（只对已发生 token 求梯度）、baseline、标准化、PPO clip、advantage normalization。GRPO 组标准化就是减方差。

---

## C2. SFT / PEFT / 训练（必考）

### C2.1 标准 SFT 目标函数？Teacher forcing 是什么？

**考点：** \(\max \sum \log p(y_t\mid x,y_{<t})\)。暴露真实前缀。

**接项目：** 只对 assistant 段求和；tool 结果当条件，不当目标。

### C2.2 为什么 SFT 会「学成平均行为」？RL 补什么？

**考点：** CE 对所有正样本一视同仁；RL 可以只抬高高奖励轨迹。

**接项目：** SFT 41% 停在「合法片单」；GRPO 用 Jaccard 区分 0.60 和 1.00。

### C2.3 LoRA 公式？秩、α、dropout 怎么选？

**考点：** \(W'=W+\frac{\alpha}{r}BA\)，\(B\in\mathbb{R}^{d\times r},A\in\mathbb{R}^{r\times k}\)。r 小省参，α/r 控制步长。

**接项目：** r=16，α=32 → 缩放 2；约 0.75% 可训参数。

### C2.4 LoRA vs QLoRA vs Adapter vs Prefix-tuning vs Full FT？

**考点：** 显存、可恢复性、合并成本。线上常 merge LoRA 成独立 checkpoint（本项目 `sft-merged` / `grpo-merged`）。

### C2.5 过拟合小 SFT 集会怎样？如何正则？

**考点：** dropout、早停、val loss、多样性。本项目 710 条、3 epoch、val 177；holdout 不参与选 SFT。

### C2.6 Chat template / special token 错了会怎样？

**考点：** 模型在补全模板而不是调用工具。本项目 collapse 检测 `<tool_response>`、`user\n` 泄漏。这是 Agent 训练的高频坑。

### C2.7 大词表 + 长序列为什么容易 OOM？gradient checkpointing、FlashAttention、Liger 各解什么？

**考点：** logits 显存 ∝ batch × seq × vocab；checkpoint 用算换存；FA 降注意力显存；Liger 融合 CE。本项目 Qwen3.5 必须 liger + ckpt。

### C2.8 Learning rate：SFT 1e-4 vs GRPO 1e-6，为什么差两个数量级？

**考点：** SFT 从随机 LoRA 学协议，需要大步；GRPO 在已会的政策上微调，大 LR 毁格式。

### C2.9 Warmup、weight decay、AdamW β 的直觉？

**考点：** warmup 防初期大梯度；AdamW 解耦衰减；LLM 常用 β2=0.95。点到为止即可。

### C2.10 SFT 数据质量 vs 数量？拒绝采样（rejection sampling）？

**考点：** 用奖励过滤教师轨迹再 SFT，就是 RS/RFT。本项目 887/1000 接受，拒绝 missing_finalize 和低奖励。

---

## C3. Agent / 工具调用 / 规划（字节 Agent 岗核心）

### C3.1 ReAct、function calling、plan-and-execute、Reflexion 有何区别？

**考点：** ReAct 交错思考与行动；FC 是结构化 tool schema；plan-and-execute 先计划再执行；Reflexion 用语言做事后反思。

**接项目：** 运行时是 **强制 function calling**（一回合一工具），不是自由文本 ReAct；规划体现在多步搜-核验-写入，没有单独 planner 模块。

### C3.2 工具 schema 设计原则？

**考点：** 名字稳定、参数可校验、禁止 hallucinate ID、observation 要短。

**接项目：** 8 工具、ID 必须来自当前 observation、compare 2–4 部、abort 枚举原因、observation 1800 字符。

### C3.3 多步 Agent 的上下文怎么管？截断策略？

**考点：** 滑窗、摘要、只留最近 k 次 observation、工具结果压缩。

**接项目：** obs 字符预算 + max_steps=14，用硬上限代替摘要模型，保证确定性。

### C3.4 什么是 tool hallucination？如何防？

**考点：** 捏造工具名、参数、ID。防：schema 校验、白名单 ID、守卫拒绝、SFT 只留合法轨迹。

### C3.5 终止问题：何时 stop / abort / 继续搜？

**考点：** 过早停（early abort）vs 死循环。奖励必须区分 correct_abort / early_abort / loop / max_steps。本项目四档都有显式分数。

### C3.6 Agent 评测：成功率、过程分、人工、LLM Judge 的利弊？

**考点：** 成功率可 hack；过程分贵；Judge 不稳、会泄漏。

**接项目：** 四面板分离，主指标规则化 gold。本任务约束都在 IMDb 字段里，用代码硬门即可；开放式对话/主观满意度才需要 LLM Judge。面试时主动讲清「什么时候不该上 Judge」。

### C3.7 RAG 和 Agent 的边界？

**考点：** RAG 是单次检索增强生成；Agent 是多步、有状态、可写副作用（写入草稿、finalize）。本项目草稿片单就是状态。

### C3.8 记忆：短时（上下文）vs 长时（外部 store）？

**考点：** 本项目短时=对话+draft；目录是只读外部记忆。没有跨 session 记忆，避免评测污染。

### C3.9 并行 tool call vs 串行？

**考点：** 并行降延迟，但依赖顺序时会错（必须先 search 再 open）。本项目强制串行，接受率从教师多工具问题里学过教训（v1 Flash 曾一轮多工具，过滤变严）。

### C3.10 Guardrail / 策略 vs 模型能力？

**考点：** 环境守卫是硬规则；模型是策略。生产上两者都要：模型尽量合法，环境永不信任模型。

### C3.11 Computer-use / 多模态 Agent 和本项目的差别？

**考点：** 本项目是结构化 JSON 工具；computer-use 是 GUI 接地。可迁移的是：契约、过程监督、可验证奖励、holdout。

### C3.12 什么是 trajectory-level vs token-level 优化？

**考点：** SFT/GRPO 都在 token 上反传，但奖励是轨迹级，再广播到整段 assistant token（本项目序列级 ratio）。这会把成功轨迹里的所有工具调用一起加强——包括碰巧的搜索词。过程奖励可减轻。

---

## C4. 奖励、评测、数据（算法岗很爱问）

### C4.1 什么是 proxy reward vs gold label？

**接项目：** rubric / valid_alternative 是 proxy；gold 集合是更严的 gold label。只优化 proxy 会偏离。

### C4.2 离线评测 vs 在线 A/B？固定分母为什么重要？

**考点：** 无效轨迹仍计入分母（本项目 infra invalid 仍在 100 题里），否则会虚高。不能把解析失败的题从分母里拿掉再报成功率。

### C4.3 数据污染、train/test overlap 有哪些形式？

**考点：** 题号重叠、模板重叠、教师见过评测、Judge 见过 gold。本项目 task_id 互斥 + Judge 不上场。

### C4.4 合成数据的风险？如何校验？

**考点：** 分布假、约束不可行、模板泄漏。本项目用真实目录生成可行金标，hard_gates 校验后再入库；并保留无解题练 abort。

### C4.5 评估的统计功效、置信区间？

**考点：** n=100，二项标准差约 \(\sqrt{p(1-p)/n}\approx5\%\)。所以强调配对对照和分层一致性，而不是「显著 0.001」。诚实讲这一点是加分。

### C4.6 Calibration、selective prediction：不会就 abort？

**接项目：** `correct_abort` +0.50，`early_abort` −0.35。教模型在证据不足时停止，是 Agent 安全的核心。

---

## C5. 检索 / 推荐 / 环境建模

### C5.1 BM25 公式直觉？和 TF-IDF、稠密检索比？

**考点：** 词频饱和 + 长度归一 + IDF。稀疏可解释、无训练。稠密擅长语义改写。本项目查询短、字段结构化，BM25+过滤足够。

### C5.2 为什么检索 top-k=8？k 对 Agent 的影响？

**考点：** k 太小召回不够，k 太大上下文噪声、选错 ID。和 RAG 的 k 是同一权衡。

### C5.3 环境是 MDP 还是 POMDP？

**考点：** Agent 看不到全目录，只看到 observation → **POMDP**。工具 `open`/`view_crew` 是为了减观察不确定性。这是加分术语。

### C5.4 状态、动作、奖励在本环境里分别是什么？

**答：** 状态≈对话+草稿+已出现 ID；动作≈一次 tool call；奖励≈终局标量（训练有 shaping）。转移由环境确定性执行。

### C5.5 模拟器偏差（sim-to-real）？

**考点：** 冻结目录 vs 线上目录会变。本项目明确是离线研究环境；上线要处理新片、缺货类似「片下架」。

---

## C6. LLM 基础（可能从 Qwen3.5 / 2B 扯开）

### C6.1 Self-Attention 计算复杂度？为什么长上下文贵？

**考点：** \(O(n^2 d)\)。本项目 max_length SFT 8192 / GRPO 4096，超长组会被截断，这就是早期要 length-norm 的原因之一。

### C6.2 MHA vs GQA vs MQA？

**考点：** KV 头共享降推理显存。Qwen 系列常用 GQA。知道即可。

### C6.3 RoPE、位置外推？

**考点：** 旋转位置编码；长于训练长度会降质。Agent 轨迹一长，先截 obs 而不是盲外推。

### C6.4 解码：greedy、temperature、top-k/p、beam。Agent 该用哪个？

**考点：** 训练探索用 T=0.7；评测/基线更偏 greedy 或低温，保证可复现。Beam 对结构化 tool JSON 不一定好。

### C6.5 Thinking / CoT / 关 thinking 为什么？

**考点：** 长思维占上下文、拖延迟、且本任务证据在工具里不在思维里。采集和基座都 **关 thinking**。

### C6.6 MoE vs Dense？2B dense 的含义？

**考点：** 本项目用小 dense 把算法闭环跑完；MoE 是另一条扩容线，不改变 GRPO 公式。

### C6.7 指令遵循 vs 工具遵循失败模式？

**考点：** 说对了但不调用、调用了参数类型错、ID 幻觉。四面板的 Behavior/Infra 就是在量化这些。

---

## C7. 系统 / 训练工程

### C7.1 在线 RL 的墙钟瓶颈在哪？

**答：** 环境多步解码，不是反传。优化：并行环境、连续批处理、前缀缓存。v4 26.6h 主要在 rollout。

### C7.2 混合精度、bf16 vs fp16？

**考点：** bf16 动态范围更稳，5090/A100 常用。本项目 GRPO dtype bf16。

### C7.3 为什么 merge LoRA 再 serve？

**考点：** 推理简单、无 PEFT 开销；评测三模型接口一致（OpenAI 兼容）。训练仍用 adapter 做差分更新。

### C7.4 可复现性：seed、温度、非确定 CUDA？

**考点：** 任务生成 seed=42；评测仍有采样/内核非确定。所以报冻结产物和 hashes，而不是宣称 bit-wise 可复现。

---

# D. 可能的现场手写 / 白板

1. 写 GRPO advantage（含 std=0 的分支）。
2. 写 PPO clip loss，标出 \(\rho\) 与 \(\epsilon=0.2\)。
3. 画数据划分：SFT / GRPO / Eval 零重叠。
4. 写 action-only label 伪代码：哪些 token 是 -100。
5. 写 LoRA 前向 \(y=(W+\frac{\alpha}{r}BA)x\)。
6. 列出 gold@100 的全部必要条件。
7. 画工具状态机：search → open/compare/view_crew → add → finalize/abort。
8. 给定四条 reward `[1.0, 0.6, 0.6, 0.0]`，算 advantage。
9. 说明为什么 `valid_alternative=0.60` 进 SFT 但不进主指标。
10. 设计一个 n=2 的过程奖励（面试官延伸；回答时强调 **不改现有评测**，只改训练）。

第 8 题参考：mean=0.55，std≈0.357，advantage ≈ `[1.26, 0.14, 0.14, -1.54]`，再把负的乘 0.5 → 最后一项约 `-0.77`。

---

# E. 建议优先背熟的 12 题

项目内：A1, A4, A5, A7, A9, A10, A11, A13, A16, A21  
八股：C1.2, C3.1

数字只记这些：

- 5000 部；划分 1000 / 400 / 100；SFT 887 接受、710/177
- Baseline / SFT / v4 = **25 / 41 / 46**；1-movie **65.6 → 75.4**
- G=4，LR=1e-6，T=0.7，clip=0.2，200 step，选 **175**，有效更新 **145**
- gold 必须：finalize + 金标集合 + reward_valid
