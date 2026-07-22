# LLM Judge 跨 Domain × Rubric 数据集与 Ground Truth 设计

> 目标：给定任务、候选回答 `X` 和评分标准 `rubric`，让小模型 Judge 输出 `Z`，并检验 `Z` 是否接近人类或指定强模型提供的参考标签 `Y`。
>
> 本文只回答两个问题：**现成 Ground Truth 能否直接使用；如果不完整，最少需要补什么标签。**
>
> 样例说明：每个数据集展示一条真实样本的核心字段。对于数千字报告，只节选候选回答正文，但保留真实样本 ID、rubric 和全部标签。

> **完整矩阵硬性要求（2026-07-22 修订）：**一旦选定 `m` 个 domain 和 `n` 个
> task，正式 B-space 必须覆盖每个保留回答与全部 `n` 个 task 的组合，即
> `B = A x n`。矩阵中九格或十六格“都有若干样本”仍不够；每个
> `input_document_id` 都必须有全部任务的合法标签和 B hidden state。原始数据缺失的
> 回答-任务标签必须列为“待补评分”，不能复制其他 rubric 的标签或只改 task 名称。

## 0. 当前最前置结论：缺什么数据和 Hidden State

五个原始数据集文件均已下载；当前主要缺口不是继续找数据，而是补齐任务标签、
修正适配器并生成完整 B-space。详细逐格审计见
`benchmark_ground_truth_matrix_audit.md`。

| 数据集 | 完整目标 | 当前状态 | 下一步 |
|---|---:|---|---|
| DeepResearch-Bench | `4 x 4`，`A=200, B=800` | A/B 未运行；当前适配器错误压成一个加权任务 | 用原生四维人工分数展开四条 B，无需补评分 |
| RealDR | `3 x 3`，`A=640, B=1,920` | A/B 未运行；当前适配器错误压成一个加权任务 | 用原生三维人工分数展开三条 B，无需补评分 |
| RuVerBench | 两个完整 `1 x 4`；共享 domain-shift 版本为 `2 x 4` | Deep Research 旧 Colab 产物未归档；Agentic Coding 未接入；每回答四任务覆盖未审计 | 先做 taxonomy 和覆盖审计，缺失回答-任务对补评分后再提取 |
| BiGGen-Bench | `3 x 3`，`A=1,080, B=3,240` | 已有 A=1,080；旧 B=1,080 只有三个对角格 | 补 2,160 个回答-任务评分，按冻结的三套 rubric 重建 B |
| FLASK | `3 x 3`，`A=1,691, B=5,073` | 已有 A=1,691；旧 3,576 条 B 只有 2,792 个唯一回答-任务对 | 去除 784 条重复对，补 2,281 个唯一回答-任务评分并重建 B |
| Prometheus | `3 x 3`，`A=N, B=3N` | 完整原始文件已下载，A/B 未运行 | 分类 domain，固定三套 rubric，为每个回答生成三项 Teacher 评分后提取 |

BiGGen-Bench 和 FLASK 的现有归档只能标为旧的稀疏 B-space 产物，不能作为完整
`3 x 3` 已完成的证据。当前五个数据集中，还没有任何一个通过上述 dense B-space
验收。

## 目录

1. [统一实验形式](#1-统一实验形式)
2. [Ground Truth 总表](#2-ground-truth-总表)
3. [LongJudgeBench](#3-longjudgebench)
4. [RuVerBench](#4-ruverbench)
5. [BiGGen-Bench](#5-biggen-bench)
6. [FLASK](#6-flask)
7. [Prometheus](#7-prometheus)
8. [最终推荐](#8-最终推荐)

---

## 1. 统一实验形式

每条数据统一为：

```text
Benchmark/domain: A、B 或 C
Task/rubric: Q、W 或 E
Question/Instruction: q
Candidate response: X
Reference label: Y
Small judge prediction: Z
```

若选定三个任务，则同一候选回答必须产生三条 Judge 输入：

```text
(domain, task 1, rubric 1, instruction, response) -> Y_1 -> B_1
(domain, task 2, rubric 2, instruction, response) -> Y_2 -> B_2
(domain, task 3, rubric 3, instruction, response) -> Y_3 -> B_3
```

A-space 对该回答只提取一次。B-space 必须按任务分别提取，标签不进入 prompt。

建议固定四种测试：

```text
Training: A + Q
Test 1:  B + Q   # 只改变 domain
Test 2:  A + W   # 只改变 rubric
Test 3:  B + W   # domain 和 rubric 同时变化
```

Ground Truth 优先级：

1. 人工或专家逐条标签；
2. 经人工验证的高置信仲裁标签；
3. 明确作为 teacher label 的强模型评分。

---

## 2. Ground Truth 总表

| 数据集 | 参考标签 `Y` | 是否符合要求 | 最少需要补什么 |
|---|---|---:|---|
| **LongJudgeBench** | DeepResearch-Bench 与 RealDR 的人工多维 pointwise 分数 | **标签齐，构造未完成** | 展开为 DeepResearch-Bench `4 x 4` 和 RealDR `3 x 3`；不纳入二分类 VerifyBench-Hard |
| **RuVerBench** | 人工 rubric-level 成功/失败标签 | **尚未证明每回答四任务齐全** | 接入 Agentic Coding，完成 taxonomy 和逐回答覆盖审计；缺项补人工评分 |
| **BiGGen-Bench human_eval** | 每个回答仅一个 instance-specific `human_score` 1–5 | **不完整** | `3 x 3` 缺 2,160 个回答-任务人工分数；GPT-4 分数不能冒充人工标签 |
| **FLASK** | 公开主结果为 GPT-4 的逐 skill 1–5 分 | **Teacher 标签不完整** | `3 x 3` 缺 2,281 个唯一回答-任务 Teacher 分数；另建人工测试子集 |
| **Prometheus** | GPT-4 生成的 custom-rubric 1–5 分 | **每回答仅一个 rubric，不完整** | 固定三套 rubric 后为每个回答重新生成三项 Teacher 分数，并增加人工测试子集 |

核心区别：

- **BiGGen-Bench 中只有 `human_score` 是本文需要的 Ground Truth。**
- `gpt4_score`、`claude_score`、`prometheus_score` 是其他 Judge 的输出，不能当作 human Ground Truth。
- FLASK 和 Prometheus 的 GPT-4 分数可以用于“大模型 Teacher → 小模型 Judge”的任务，但不能写成人工 Ground Truth。

---

## 3. LongJudgeBench

### 3.1 Domain 与 Task

只使用两个单回答、多维 pointwise 子集，分别构造矩阵；不能把子集名称直接当作
两个 domain，再把全部维度压成一个任务。

**DeepResearch-Bench：完整 `4 x 4`**

```text
Domain 1 = Science and Technology
Domain 2 = Business and Finance
Domain 3 = Society and Public Policy
Domain 4 = Humanities and Culture

Task 1 = Comprehensiveness
Task 2 = Insight / Depth
Task 3 = Instruction Following
Task 4 = Readability
```

**RealDR：完整 `3 x 3`**

```text
Domain 1 = Science and Engineering
Domain 2 = Business and Applied Studies
Domain 3 = Humanities and Social Sciences

Task 1 = Logical Structure
Task 2 = Presentation Form
Task 3 = Bias Checking
```

domain 聚合必须保存明确映射并抽样复核。`verify_bench_hard` 是二分类正确性数据，
本流程不下载、不准备、也不提取它的 hidden states。

### 3.2 Ground Truth

- `deepresearch_bench` 的每个候选报告都有四个维度的人类聚合分数和
  `weighted_total`。完整 B 使用四个**独立维度分数**，不是只使用总分。
- `realdr` 的每个候选文档都有逻辑结构、表达形式、偏见检查三维人工分数和
  `weighted_total`。完整 B 使用三个**独立维度分数**。
- `weighted_total` 可以保留为辅助总体质量标签，但不能替代多任务矩阵中的独立
  task 标签，也不能作为唯一 B-space 任务。

### 3.3 真实样例

```yaml
dataset: deepresearch_bench
domain: 由该样本原始 topic 映射到四个领域之一
tasks:
  - Comprehensiveness
  - Insight / Depth
  - Instruction Following
  - Readability

rubric_dimensions:
  readability: 0.14
  insight: 0.36
  comprehensiveness: 0.30
  instruction_following: 0.20

reference_labels_Y:
  type: pointwise_multi_dimension
  each_dimension: 使用原始四维人工聚合分数
  weighted_total: 79.3  # 仅作辅助总体标签
  annotators: 3

ground_truth_source: human aggregated multi-dimension score
```

同一候选报告生成四条 B-space Judge 输入，每条只包含当前 task 的 rubric、原始
instruction 和候选回答；人工分数不进入 prompt。RealDR 同理，每个候选文档生成
三条 B。

### 3.4 是否需要补标签

- 本次两个多维 pointwise 子集：**不需要补标签，但必须修改适配器按维度展开**。
- DeepResearch-Bench 正确数量为 `A=200, B=800`；RealDR 正确数量为
  `A=640, B=1,920`。
- 当前只生成加权总任务的配置不满足完整矩阵要求，不能继续用于正式提取。
- 二分类 VerifyBench-Hard 不属于本次 hidden-state 提取范围。

---

## 4. RuVerBench

### 4.1 Domain 与 Task

RuVerBench 原生只有两个真实文本分布：

```text
Domain 1 = Deep Research：长报告
Domain 2 = Agentic Coding：代理执行轨迹
```

“Agentic Coding 任务结果正确性”和“Agentic Coding 流程合规性”是 task，不是两个
domain，不能为了凑 `3 x 3` 把它们放到 domain 轴。

原生可先形成两个完整四任务面板：

```text
Deep Research:
  Format / Numbers / Logic / Facts

Agentic Coding:
  Task Completion / Planning / Tool Use / Rules and Compliance
```

若要构造可严格解释为纯 Domain-OOD 的共享 `2 x 4`，还需冻结四个跨域同义 rubric，
并确保两个 domain 的每个回答都有全部四项标签。

### 4.2 Ground Truth

每条 rubric/check 都有人工参考结果：

```text
success / fail
covered = true / false
```

可以直接作为二分类 `Y`。

### 4.3 真实样例

```yaml
id: DRB2_1
benchmark: A = Deep Research
task: Q = Coverage

question: |
  撰写一份关于中国地方“土地财政”模式转型的研究报告，
  包含当前模式困境、1998—2021 年数据、转型机制及四类房地产税制比较表。

candidate_response_X_excerpt: |
  报告解释了 1994 年分税制改革、1998 年住房制度改革、
  2021 年土地出让收入及财政依赖，并提出增量与存量协同改革。

rubrics_and_labels_Y:
  - rubric: 将土地出让金征收对象从普通商品房转向“改善型住房”
    covered: true
  - rubric: 比较表包含“对建筑物征税”一行
    covered: true
  - rubric: 该行明确写出缺点为“社会总福利损失最大”
    covered: false
  - rubric: 比较表包含“对土地和建筑物统一征税”一行
    covered: true
  - rubric: 解释 1994 年分税制改革与地方依赖土地财政的因果关系
    covered: true

ground_truth_source: human rubric-level labels
```

原始文件可以先把一条长回答拆成五条 rubric-check 样本：

```text
(question, response X, rubric_1) -> true
(question, response X, rubric_2) -> true
(question, response X, rubric_3) -> false
...
```

但这五条 check 不自动等于完整四任务 B-space。必须先把 check 映射到 Format、
Numbers、Logic、Facts，并逐个 `input_document_id` 检查四类是否齐全。缺少任一类时，
该回答对应的 task 仍是待补评分。

### 4.4 是否需要补标签

当前不能写成“不需要重新标注”：

- Deep Research 已有 284 个唯一回答和 1,615 条原始 check，但尚未完成每回答四任务
  覆盖审计；按每回答每任务一条的目标至少应有 `B=284 x 4=1,136` 个唯一组合。
- Agentic Coding 四份官方文件已下载，但尚未接入适配器，也没有得到唯一回答数和
  每回答四任务覆盖率。
- 原始 check 能支持的标签可以复用；某个回答缺失的任务必须补人工评分。
- 如果采用共享 `2 x 4` rubric，原生 taxonomy 与共享 rubric 不等价的部分也必须
  重新评分，不能只改类别名称。

---

## 5. BiGGen-Bench

### 5.1 Benchmark 与 Task

```text
Benchmark A = Grounding
Benchmark B = Reasoning
Benchmark C = Planning

Task Q = Requirement/Grounding Satisfaction
Task W = Correctness
Task E = Completeness
```

BiGGen 原始 rubric 是 instance-specific，需要先把 rubric 映射到 Q/W/E。只保留映射明确的样本。

正式目标不是三个对角单元，而是每个保留回答都按三套冻结 rubric 评分：

| Domain | Requirement/Grounding Satisfaction | Correctness | Completeness |
|---|---:|---:|---:|
| Grounding | 400 个已有原生评分 | 400 个待补评分 | 400 个待补评分 |
| Reasoning | 400 个待补评分 | 400 个已有原生评分 | 400 个待补评分 |
| Planning | 280 个待补评分 | 280 个待补评分 | 280 个已有原生评分 |

因此 `A=1,080` 时，完整 B 应为 `3,240`，当前旧 B 只有 `1,080`，缺
`2,160` 个回答-任务评分和对应 hidden state。

### 5.2 Ground Truth

必须使用：

```text
split = human_eval 或 multilingual_human_eval
Y = human_score
```

不能使用：

```text
gpt4_score
claude_score
prometheus_score
```

这些字段是候选 Judge 的预测，正是需要与人工分数比较的对象。

### 5.3 真实样例

```yaml
id: biggen_bench/grounding_demo_vs_instruction_1
benchmark: A = Grounding
task: Q = Requirement/Grounding Satisfaction

problem: |
  Sort the numbers in ascending order.
  Demonstrations misleadingly show descending outputs.
  Final input: 2911 2 98 -33

candidate_response_X: |
  Output: -33 2 98 2911
  However, I would recommend using a built-in sorting method for large data...

reference_answer: "Output: -33 2 98 2911"

human_reference_Y:
  human_score: 3
  final_answer_correct: false

ground_truth_source: human evaluator
```

该行的公开人工参考分数是 `human_score=3`。实验必须直接学习这一人工分数，不能根据 `final_answer_correct` 自行改写标签，也不能用任何 GPT-4 评分字段替换它。

### 5.4 是否需要补标签

- 当前 1,080 个原生 `human_score` 只能用于它实际对应的 instance-specific rubric，
  不能复制到同一回答的另外两个任务。
- 对每个回答缺失的另外两个任务，按冻结的 Requirement/Grounding Satisfaction、
  Correctness、Completeness rubric 由两名人工独立评分并仲裁，共需补 2,160 项。
- 不允许用 `gpt4_score` 填补 human_score 缺口后再声称是人工 Ground Truth。
- 若为节省成本改用固定强模型补分，必须另建 Teacher-Label track，并重建完整 B
  cache；不能与 Human-GT track 混合报告。

---

## 6. FLASK

### 6.1 Benchmark 与 Task

FLASK 原生提供 domain 和部分 skill 字段。目标固定为完整 `3 x 3`：

```text
Benchmark A = Humanities
Benchmark B = Coding
Benchmark C = Math

Task Q = Logical Correctness
Task W = Comprehension
Task E = Conciseness
```

可增加 Readability、Factuality、Completeness 等任务，但主实验先保留三个。

“某个 domain-task 格子有样本”不代表完成。主实验保留的每个回答都必须同时具有
Logical Correctness、Comprehension 和 Conciseness 三项评分及三条 B。

### 6.2 Ground Truth

公开评分文件中最完整的是 GPT-4 的逐 skill 1–5 分，可作为“大模型 Ground Truth/Teacher Label”。

```text
Y_Q = GPT-4 Logical Correctness score
Y_W = GPT-4 Comprehension score
Y_E = GPT-4 Conciseness score
```

若论文结论要写成“接近人类评分”，必须增加人工测试集。

### 6.3 真实样例

```yaml
question_id: 1
benchmark: A = Humanities

instruction: |
  Rewrite the sentence to make it clearer and more concise:
  "If you have any questions about my rate or if you find it necessary
  to increase or decrease the scope for this project, please let me know."

reference_answer: |
  If you have any questions about my rate or find it necessary to increase
  or decrease this project's scope, please let me know.

candidate_response_X: |
  Please let me know if you have any questions about my rate
  or if you need to adjust the scope of this project.

skills_and_labels_Y:
  Readability: 5
  Logical Correctness: 5
  Conciseness: 5

ground_truth_source: GPT-4 skill-level evaluation
```

该原始样本目前只能拆出它实际带标签的三条：

```text
Humanities + Readability          -> 5
Humanities + Logical Correctness  -> 5
Humanities + Conciseness          -> 5
```

其中 Readability 不属于当前冻结的三任务集合，而 Comprehension 在该样本中缺失。
所以该样本要进入完整 `3 x 3` 主实验，还必须补一项 Comprehension Teacher 评分；
不能把 Readability 分数改名为 Comprehension。

### 6.4 是否需要补标签

当前归档有 1,691 个唯一 A 和 3,576 条旧 B 行，但按
`(input_document_id, task)` 去重后只有 2,792 个唯一回答-任务对。完整目标为：

```text
A = 1,691
B = 1,691 x 3 = 5,073
缺失唯一回答-任务评分 = 2,281
旧 B 中重复回答-任务行 = 784
```

处理步骤：

1. 对旧 prepared 数据按 `(input_document_id, task)` 去重并审计重复标签是否一致；
2. 使用同一冻结版本的强 Teacher 和三套固定 rubric 补齐 2,281 项；
3. 为完整 5,073 个回答-任务对生成新版 B cache，A cache 可以复用；
4. 从 Humanities、Coding、Math 的九格抽取平衡测试集，对三个任务分别做双人人工
   1--5 分标注与仲裁；
5. 分别报告 `small judge vs Teacher` 和 `small judge vs human`。

---

## 7. Prometheus

### 7.1 Benchmark 与 Task

Prometheus 没有统一 domain 字段，需要根据真实 instruction 建立确定性分类：

```text
Benchmark A = Communication and Advice
Benchmark B = Knowledge/Technical Explanation
Benchmark C = Writing and Revision

Task Q = Correctness/Factuality
Task W = Completeness/Instruction Fulfillment
Task E = Style/Context Adaptation
```

上述 domain 是实验派生字段，需要保存分类规则和人工抽样复核结果。

### 7.2 Ground Truth

Feedback Collection/FeedbackBench 中的反馈和 1–5 分主要由 GPT-4 生成，因此：

```text
Y = GPT-4 custom-rubric score
```

它可以作为“大模型评分 Ground Truth”，但不是人工 Ground Truth。

### 7.3 真实样例

```yaml
benchmark: A = Communication and Advice
task: E = Style/Context Adaptation

instruction: |
  A Japanese professional is preparing for a business meeting with an Irish partner
  and asks about etiquette, suitable conversation topics and cultural pitfalls.

criterion: |
  Is the model culturally aware and does it modify its reaction
  based on the user's cultural background?

candidate_response_X_excerpt: |
  Irish business culture can be more informal than Japanese business culture,
  but punctuality remains valued. Humor is common, while overly personal topics
  should be avoided. Gift-giving is generally not expected.

reference_answer_excerpt: |
  Understand the more relaxed Irish business style, remain punctual,
  use clear communication, avoid stereotypes and adapt respectfully.

reference_label_Y: 4
feedback_excerpt: |
  The response shows good cultural sensitivity and identifies punctuality,
  humor and communication differences, but could provide more detailed guidance.

ground_truth_source: GPT-4-generated teacher score and feedback
```

### 7.4 是否需要补标签

原始样本通常只有一个 custom rubric 分数，不能自动得到完整的 Q/W/E。

最小补充方案：

1. 保留现有回答 `X`；
2. 固定三套统一 rubric：Correctness、Completeness、Style；
3. 用同一版本的强模型重新给每个 `X` 打三个 1–5 分；
4. 从每个 `domain × rubric` 抽取平衡测试样本；
5. 对测试样本做双人人工标注与仲裁。

因此训练可以使用大模型标签，最终结论由人工测试集验证。

正式 prepare 得到三个选中 domain 中的唯一回答数 `N` 后，必须验收为
`A=N, B=3N`。现有 custom-rubric 分数只有在语义与某套冻结 rubric 一致时才能
复用；其余任务仍需重新评分。

---

## 8. 最终推荐

### 8.1 主实验数据

优先顺序：

```text
1. LongJudgeBench  原生多维人工标签，先完成 4 x 4 和 3 x 3 展开
2. BiGGen human_eval  人工 1–5 分，补齐完整 3 x 3 后使用
3. RuVerBench      人工 rubric-level 二分类，完成四任务覆盖审计后使用
```

这三个数据集在各自完整 B-space 建成后，可以支撑“**小模型 Judge 是否接近人类
评分**”。当前都不能按完整矩阵状态直接进入主实验。

### 8.2 大模型 Ground Truth 实验

```text
4. FLASK           补齐每回答三项 GPT-4/固定 Teacher skill 分数
5. Prometheus      为每回答生成三项固定 rubric Teacher 分数
```

这两个数据集完成 dense Teacher 标签后可以支撑“**小模型 Judge 是否接近强模型
评分**”。为了进一步证明其与人类一致，需要补充一个分层、平衡的人工测试集；
人工测试集不能替代训练矩阵中缺失的 Teacher 评分。

### 8.3 最终统一实验结构

```text
Human-GT track:
  RuVerBench + BiGGen human_eval + LongJudgeBench

LLM-teacher track:
  FLASK + Prometheus
```

所有结果必须按 Ground Truth 类型分别报告，不能混合成一个总准确率。

---

## 数据来源

- LongJudgeBench：官方 GitHub 数据文件 `data_standardized/deepresearch_bench.jsonl` 与 `data_standardized/realdr.jsonl`
- RuVerBench：官方 GitHub `deepresearch_dataset.json`、`deepresearch_responses.json`、`deepresearch_labels.json`
- BiGGen-Bench：`human_eval` / `multilingual_human_eval` 与 `outcome_meta_evaluation`
- FLASK：官方 `flask_evaluation.jsonl` 与 `gpt_review/outputs/*.jsonl`
- Prometheus：Feedback Collection / FeedbackBench 官方数据
