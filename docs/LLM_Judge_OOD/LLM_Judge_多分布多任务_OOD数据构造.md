# LLM Judge 多分布 × 多任务 OOD 数据构造与 Ground Truth 方案

> 目标：利用现有数据集中的原生 Ground Truth，构造“多个 Benchmark/domain × 多个 Task/rubric”的 OOD 实验。
>
> 核心原则：矩阵形状可以是 `3 x 3`、`4 x 4` 或其他预先冻结的 `m x n`，但
> 一旦形状确定，每个保留回答必须拥有全部 `n` 个任务标签和对应 B-space。
> 原始数据可以稀疏；正式 OOD 主实验不能稀疏。空缺必须补评分并记录，不能复制
> 其他任务标签、补零或只改变 task 名称。
>
> 本文将参考标签统一记为 `Y`，但必须在 metadata 和结果中严格区分
> Human-GT 与 Teacher Label，二者不能混合报告。

---

## 1. 最终结论

只有 LongJudgeBench 的 DeepResearch-Bench 和 RealDR 原生对每个回答提供全部目标
维度，不需要新增评分。RuVerBench、BiGGen-Bench、FLASK 和 Prometheus 的原始标签
都是部分任务覆盖；要满足完整 B-space，必须先审计每个 `(input_document_id, task)`，
再为缺项补人工或固定 Teacher 评分。

最终建议如下：

| 数据集 | 正式目标 | B-space 验收 | 是否需要补评分 |
|---|---:|---|---|
| LongJudgeBench / DeepResearch-Bench | `4 x 4` | `A=200, B=800` | 不需要；原生四维人工标签齐全 |
| LongJudgeBench / RealDR | `3 x 3` | `A=640, B=1,920` | 不需要；原生三维人工标签齐全 |
| RuVerBench | 两个完整 `1 x 4`；共享任务时为 `2 x 4` | 每个回答均有四项 B | 覆盖审计后按回答补缺项 |
| BiGGen-Bench | 选定 Grounding、Reasoning、Planning 的 `3 x 3` | `A=1,080, B=3,240` | 需要补 2,160 项 |
| FLASK | Humanities、Coding、Math x 三任务的 `3 x 3` | `A=1,691, B=5,073` | 需要补 2,281 个唯一回答-任务对 |
| Prometheus | 三个派生 domain x 三套固定 rubric 的 `3 x 3` | prepare 后为 `A=N, B=3N` | 每回答原生通常只有一项，其他项需 Teacher 评分 |

| 数据集                | 多 Domain 文本分布                                                                                                                                           | 多任务类型分布                                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DeepResearchBench** | 官方有 **22 个研究领域**。实验中聚合为：① Science and Technology；② Business and Finance；③ Society and Public Policy；④ Humanities and Culture | 原生 4 个任务：**Comprehensiveness、Insight/Depth、Instruction Following、Readability**。([GitHub][1])                                                                                                                        |
| **RealDR**            | 官方包含 **12 个学科领域**。可聚合为：① Science & Engineering；② Business & Applied Studies；③ Humanities & Social Sciences                                  | 原生 3 个任务：**Logical Structure、Presentation Form、Bias Checking**。LongJudgeBench 将 RealDR 定义为三维评分数据。([Emergent Mind][2])                                                                                     |
| **RuVerBench**        | 2 个原生文本分布：① **DeepResearch** 长报告；② **AgenticCoding** 代码代理轨迹                                                                                | DeepResearch：**Format、Numbers、Logic、Facts**；AgenticCoding：**Task Completion、Planning、Tool Use、Rules/Compliance**。两个 Domain 的任务类型不同，因此属于异构多任务                                                     |
| **BiGGen-Bench**      | 9 个 capability 分布：**Instruction Following、Grounding、Reasoning、Planning、Refinement、Multilingual、Safety、Theory of Mind、Tool Usage**                | 77 个具体任务，例如 personal assistant、compositional planning 等。任务隶属于 capability，是层级结构，不能直接视为完整 `9×77` 或 `9×9` 矩阵。([Hugging Face][3])                                                              |
| **FLASK**             | 10 个原生 Domain：**Language、Culture、Health、History、Natural Science、Math、Social Science、Technology、Coding、Humanities**。                            | 12 个原生 Skill：**Logical Robustness、Logical Correctness、Logical Efficiency、Commonsense Understanding、Factuality、Metacognition、Insightfulness、Completeness、Comprehension、Conciseness、Readability、Harmlessness**。 |
| **Prometheus**        | 没有原生统一 Domain。当前根据 instruction 派生：① Communication and Advice；② Knowledge/Technical Explanation；③ Writing and Revision | 没有原生统一任务类型。当前固定三套 rubric：① Correctness/Factuality；② Completeness/Instruction Fulfillment；③ Style/Context Adaptation |

最清楚的分类是：

- **原生标签已经完整**：DeepResearchBench、RealDR。
- **原生多 Domain + 多任务但每回答标签稀疏，必须补齐**：FLASK。
- **capability 与原生 rubric 绑定，必须交叉补齐三任务**：BiGGen-Bench。
- **两个异构 Domain，各有自己的任务体系，必须先做覆盖审计**：RuVerBench。
- **需要自己分类并重新做三任务 Teacher 评分**：Prometheus。

---

# 2. “多×多”

设：

```text
Domain/Benchmark 轴：D1, D2, ..., Dm
Task/Rubric 轴：    T1, T2, ..., Tn
```

则整个数据集形成：

\[
m \times n
\]

矩阵。

每个单元包含：

```text
(domain = Di, task = Tj) 的 Judge 样本
```

每条 Judge 样本统一为：

```yaml
sample_id: 原始文本 ID
domain: Di
task: Tj
instruction: 原始问题或指令
candidate_response: 待评分文本
rubric: 当前评分标准
judge_ground_truth: 原始 Ground Truth
```

需要区分两类矩阵。

## 2.1 完整多任务矩阵

同一篇文本原生具有多个任务标签：

```text
同一篇文本
├── T1 → Y1
├── T2 → Y2
├── T3 → Y3
└── T4 → Y4
```

只要把文本按 domain 分组，就天然形成完整的 `m×n`。

代表数据：

```text
DeepResearchBench
RealDR
```

## 2.2 稀疏原始标签与待补矩阵

每篇文本只覆盖部分任务，甚至只有一个任务标签：

```text
文本 1：D1 + T1 → Y
文本 2：D1 + T3 → Y
文本 3：D2 + T1 → Y
```

整个原始数据集可能在汇总层面覆盖多个 domain 和多个 task，但并非每篇文本都有
全部标签。

代表数据：

```text
RuVerBench
BiGGen-Bench
FLASK
Prometheus
```

这种稀疏结构只能用于审计标签来源和估算补分成本，不能直接作为本文正式的
Domain-OOD、Task-OOD 和 Joint-OOD 主矩阵。即使所有 cell 都非空，也必须逐回答
补齐所选任务，直到 `B=A x n`。

---

# 3. 统一 OOD 构造流程

所有数据集统一执行下面六步。

## 3.1 转成长表

最终每一行只表示一个任务和一个 Ground Truth：

```yaml
sample_id: report_001
domain: A
task: Q
rubric: Comprehensiveness
label: 8.4
```

若同一文本有四个任务标签，就展开成四行。

## 3.2 构造 cell count matrix

统计：

\[
N\_{d,t}
=
\#\{x:\operatorname{domain}(x)=d,\operatorname{task}(x)=t\}
\]

得到真实的 `domain × task` 样本量矩阵。

## 3.3 选择目标子矩阵并补齐回答-任务对

设置最低样本数：

```text
建议初筛：每个 cell ≥ 50
正式实验：每个 cell 最好 ≥ 100
```

先选择算力和标注预算可承受的：

```text
K 个 domains × L 个 tasks
```

不要求是正方形，例如以下都成立：

```text
4×4
3×5
6×4
8×3
```

关键是：

- K ≥ 2；
- L ≥ 2；
- 每个保留 cell 都有足够 Ground Truth；
- 每个保留回答都有全部 `L` 个 task 标签，而不只是每格非空；
- 缺失标签通过合法人工或冻结 Teacher 重新评分补齐；
- 绝不复制其他任务标签。

完成后还要验证：

```text
unique(input_document_id, task) = unique(input_document_id) x L
```

## 3.4 按原始 sample_id 划分

必须按原始文本 ID 分组切分：

```text
同一篇文本的所有 task rows
只能全部进入 train、validation 或 test 中的一个集合
```

否则同一文本的一个评分维度可能进入训练，另一个评分维度进入 OOD 测试，造成内容泄漏。

## 3.5 定义 ID 与 OOD

假设训练 cell 是：

```text
ID Train = A + Q
```

则：

```text
ID Test:
  held-out A + Q

Domain OOD:
  B + Q
  C + Q
  ...

Task OOD:
  A + W
  A + E
  ...

Joint OOD:
  B + W
  C + E
  ...
```

OOD Ground Truth 自动生成：

```text
A + Q held-out → ood_label = 0
其他选定 cell → ood_label = 1
```

这不是人工标注，是实验定义直接产生的标签。

## 3.6 保存两类 Ground Truth

每条数据同时保存：

```yaml
judge_ground_truth: 回答质量的原始 Ground Truth
ood_ground_truth: 0 或 1
```

两者作用不同：

- `judge_ground_truth`：判断小 Judge 是否接近参考 Ground Truth；
- `ood_ground_truth`：计算 AUROC、AUPR、FPR95 等 OOD 指标。

---

# 4. LongJudgeBench

只使用两个多维 pointwise 子集：

```text
DeepResearchBench
RealDR
```

---

## 4.1 DeepResearchBench：完整 `4×4`

DeepResearchBench 的每篇报告原生具有四个独立评分维度：

```text
Task Q = Comprehensiveness
Task W = Insight / Depth
Task E = Instruction Following
Task R = Readability
```

每篇文本拥有：

```text
Y_Q
Y_W
Y_E
Y_R
```

因此任务轴原生就是四任务，不需要重新打分。

### Domain 轴

根据原始 topic/domain 字段，把研究主题确定性聚合成四个上层分布：

```text
Domain A = Science and Technology
Domain B = Business and Finance
Domain C = Society and Public Policy
Domain D = Humanities and Culture
```

正式映射必须保存为配置文件：

```json
{
  "physics": "A",
  "computer_science": "A",
  "finance": "B",
  "business": "B",
  "law": "C",
  "public_policy": "C",
  "history": "D",
  "culture": "D"
}
```

实际类别名称以原始文件为准。无法明确归类的主题删除，不要强行映射。

### 最终结构

```text
4 domains × 4 tasks
= 完整 4×4
```

由于每篇文本原生有四个分数，只要某个 domain 中有报告，该 domain 的四个 task cell 都有 Ground Truth。

### OOD 构造

```text
ID Train:
A + Q

Domain OOD:
B + Q
C + Q
D + Q

Task OOD:
A + W
A + E
A + R

Joint OOD:
B + W
C + E
D + R
```

### 是否需要新增 Ground Truth

```text
完全不需要。
```

只需要：

- domain 聚合；
- 宽表转长表；
- 按 sample_id 划分。

---

## 4.2 RealDR：完整 `3×3`

RealDR 的每篇报告原生具有三个独立评分维度：

```text
Task Q = Logical Structure
Task W = Presentation Form
Task E = Bias Checking
```

每篇文本原生拥有：

```text
Y_Q
Y_W
Y_E
```

### Domain 轴

根据原始 discipline/topic 字段聚合：

```text
Domain A = Science and Engineering
Domain B = Business and Applied Studies
Domain C = Humanities and Social Sciences
```

### 最终结构

```text
3 domains × 3 tasks
= 完整 3×3
```

### OOD 构造

```text
ID Train:
A + Q

Domain OOD:
B + Q
C + Q

Task OOD:
A + W
A + E

Joint OOD:
B + W
C + E
```

### 是否需要新增 Ground Truth

```text
完全不需要。
```

---

## 4.3 VerifyBench-Hard：当前流程排除

VerifyBench-Hard 原生只有：

```text
Task = Correctness Verification
Label = correct / incorrect
```

它无法在自身内部构造多 domain × 多 task。

它可以在其他实验中作为单任务 external far-OOD，但不属于本次五数据集 dense
B-space 流程，不下载、不准备、不提取：

```text
本次状态 = excluded
```

例如：

```yaml
source_benchmark: VerifyBench-Hard
shift_type: cross_benchmark_far_ood
ood_label: 1
judge_ground_truth: correct / incorrect
```

### 是否需要新增 Ground Truth

```text
不需要。
```

但它不能承担完整的多×多主实验。

---

# 5. RuVerBench

RuVerBench 有两个原生 domain：

```text
Domain A = DeepResearch
Domain B = AgenticCoding
```

每个 rubric/check 都有人工二分类 Ground Truth：

```text
success / fail
covered / not covered
```

## 5.1 DeepResearch：`1×4`

原生 taxonomy：

```text
Task Q = Format
Task W = Numbers
Task E = Logic
Task R = Facts
```

目标形成：

```text
1 domain × 4 tasks
```

但“原始 check 汇总后四类都出现过”不等于完整。已知 284 个 DeepResearch 回答时，
必须逐回答确认 Format、Numbers、Logic、Facts 四项都存在；规范化为每回答每任务
一条时，目标为 `A=284, B=1,136`。缺项要补评分。

## 5.2 AgenticCoding：`1×4`

原生 taxonomy：

```text
Task Q = Task Completion
Task W = Planning
Task E = Tool Use
Task R = Rules / Compliance
```

目标形成：

```text
1 domain × 4 tasks
```

Agentic Coding 必须先接入已下载的 dataset、labels、taxonomy 和 trajectories 四份
文件，再统计唯一回答数及每回答四任务覆盖。

## 5.3 能否构造 `2×4`

数据结构上可以合并为：

```text
2 domains × 4 task positions
= 两个任务语义不同的面板
```

但两个 domain 的任务语义不同：

```text
DeepResearch:
Format / Numbers / Logic / Facts

AgenticCoding:
Task / Planning / Tools / Rules
```

因此：

```text
A + Q → B + Q
```

并非严格的“只改变 domain”，而是 domain 与 rubric 语义同时变化。

### 推荐用法

在各自回答四任务都补齐后，可分别做：

```text
DeepResearch 1×4：只研究 task shift
AgenticCoding 1×4：只研究 task shift
```

再把两个 domain 之间的变化作为：

```text
cross-domain + cross-task comprehensive OOD
```

### 是否需要新增 Ground Truth

需要先逐回答审计。原生 check 已覆盖的回答-任务对可复用，缺失项必须补人工评分。
若要求两个 domain 共享完全相同的四个 task，还必须冻结跨域同义 rubric，并对与
共享 rubric 不等价的原生项重新标注。未完成该步骤时不能声称纯 Domain-OOD。

---

# 6. BiGGen-Bench

BiGGen-Bench 原生有：

```text
9 capabilities
77 concrete tasks
765 benchmark instances
每个 instance 一个 instance-specific score rubric
```

Evaluation 结果中，每条候选回答通常只有：

```text
一个 rubric
一个 human_score
```

## 6.1 为什么不是自动 `9×9`

九个 capability 是样本分布类别：

```text
Instruction Following
Grounding
Reasoning
Planning
Refinement
Multilingual
Safety
Theory of Mind
Tool Usage
```

它们不是同一回答的九个评价维度。

不能把一个 Planning 回答的 `human_score` 同时当作：

```text
Planning score
Safety score
Grounding score
Tool Usage score
...
```

因此原生结构不是 dense `9×9`。

## 6.2 当前正式目标：完整 `3 x 3`

### Domain 轴

当前选择三个原生 capability 作为 domain：

```text
D1 = Grounding
D2 = Reasoning
D3 = Planning
```

### Task 轴

冻结三套跨 domain 通用 rubric：

```text
T1 = Requirement / Grounding Satisfaction
T2 = Correctness
T3 = Completeness
```

原始每条样本只能先支持一个最主要的 task：

```yaml
capability: Planning
rubric_family: Completeness
human_score: 4
```

不能把同一个 `human_score` 复制到多个 rubric family。

### 映射方法

按以下顺序分类：

1. 用 rubric 文本中的明确关键词和定义规则分类；
2. 规则无法判断时，使用固定版本分类模型；
3. 保存分类置信度；
4. 低置信样本删除；
5. 抽样检查分类质量。

这一步只确定现有 `human_score` 属于哪个 task，不能产生同一回答另外两个 task 的
标签。

### 统计和筛选

建立：

\[
N\_{\text{capability},\text{rubric family}}
\]

当前旧 prepared 数据是三个对角单元：

| Domain | Requirement/Grounding Satisfaction | Correctness | Completeness |
|---|---:|---:|---:|
| Grounding | 400 | 0 | 0 |
| Reasoning | 0 | 400 | 0 |
| Planning | 0 | 0 | 280 |

旧结果为 `A=1,080, B=1,080`。完整目标必须是 `B=1,080 x 3=3,240`，
因此缺少 2,160 个回答-任务标签和对应 B hidden state。

### OOD 构造示例

完整 `3 x 3` 的构造：

```text
ID Train: Grounding + Requirement/Grounding Satisfaction
Domain OOD: Reasoning/Planning + Requirement/Grounding Satisfaction
Task OOD: Grounding + Correctness/Completeness
Joint OOD: Reasoning/Planning + Correctness/Completeness
```

### 是否需要新增 Ground Truth

需要。若保持 Human-GT track，三个非空对角单元的 1,080 个 `human_score` 可复用，
其余六个单元共 2,160 项必须由人工按冻结 rubric 评分。若使用固定强模型补齐，
则整个补齐版本必须标为 Teacher-Label track，不能与人工分数混称 Human-GT。

---

# 7. FLASK

FLASK 原生提供：

```text
10 domains
12 skills
5 difficulty levels
```

每条实例会从 12 个 skills 中选择若干最相关 skills，官方评分结果对这些 skill 分别给出 1–5 分。

因此 FLASK 天然覆盖：

```text
10 domains × 12 skills
```

但这是稀疏矩阵。

## 7.1 Domain 轴

直接使用原生 10 个 domain。

## 7.2 Task 轴

直接使用原生 12 个 skill：

```text
Logical Correctness
Logical Robustness
Logical Efficiency
Factuality
Commonsense Understanding
Comprehension
Insightfulness
Completeness
Metacognition
Readability
Conciseness
Harmlessness
```

## 7.3 展开方式

一条原始数据例如：

```yaml
sample_id: 001
domain:
  - Humanities
skills:
  - Readability
  - Logical Correctness
  - Conciseness
scores:
  - 5
  - 5
  - 4
```

展开为三行：

```text
Humanities + Readability → 5
Humanities + Logical Correctness → 5
Humanities + Conciseness → 4
```

## 7.4 多 domain 样本处理

若一条数据有多个 domain：

```text
[Math, Technology]
```

推荐两种做法之一：

```text
方案 A：只保留单 domain 样本；
方案 B：使用明确的 primary domain。
```

不要把同一文本复制到多个 domain，否则容易在训练和测试之间形成重复内容。

## 7.5 正式目标 `3 x 3`

当前冻结：

```text
Domains = Humanities / Coding / Math
Tasks = Logical Correctness / Comprehension / Conciseness
```

旧归档包含 1,691 个唯一回答、3,576 条原始 B 行，但只有 2,792 个唯一
`(input_document_id, task)`；另有 784 条重复回答-任务行。完整目标为：

```text
A = 1,691
B = 1,691 x 3 = 5,073
缺失唯一回答-任务对 = 2,281
```

九个 cell 当前都非空并不等于完成，因为 1,169 个回答只覆盖一个或两个目标任务，
只有 522 个回答覆盖全部三个任务。

## 7.6 OOD 构造

完整 `3 x 3`：

```text
ID Train: Humanities + Logical Correctness
Domain OOD: Coding/Math + Logical Correctness
Task OOD: Humanities + Comprehension/Conciseness
Joint OOD: Coding/Math + Comprehension/Conciseness
```

必须按原始 sample_id 分组切分。

### 是否需要新增 Ground Truth

需要补 2,281 个唯一回答-任务 Teacher 分数。现有 GPT-4 skill score 可用于已覆盖
的组合；缺项必须使用同一冻结 Teacher 和相同 rubric 重新评分。若需要 Human-GT，
再从九格抽取平衡测试集做双人评分与仲裁。

---

# 8. Prometheus / Feedback Collection

Feedback Collection 原生约包含：

```text
99,952 rows
996 custom criteria
每条数据一个 response
一个 custom rubric
一个 1–5 Ground Truth score
```

它没有统一 domain 字段。正式目标是派生三个 domain，并为每个保留回答重新生成
三套固定任务评分，得到完整 `3 x 3`。

## 8.1 Domain 轴

根据 `orig_instruction` 做固定 taxonomy 分类，当前保留三类：

```text
D1 = Communication and Advice
D2 = Knowledge and Technical Explanation
D3 = Writing and Revision
```

分类步骤：

1. 固定类别定义；
2. 规则或固定模型分类；
3. 保存置信度；
4. 低置信样本删除；
5. 抽样复核。

## 8.2 Task 轴

原始约 996 个 criteria 只用于判断现有分数与哪套目标 rubric 最接近。正式任务固定为：

```text
T1 = Correctness / Factuality
T2 = Completeness / Instruction Fulfillment
T3 = Style / Context Adaptation
```

每条原始数据通常只能直接支持一个最主要的 task；另外两个 task 必须重新评分。

## 8.3 候选矩阵

正式目标：

```text
3 domains × 3 fixed rubrics
= complete 3×3
```

完整 prepare 后先确定三个 domain 中的唯一回答数 `N`。必须为每个回答生成三项
Teacher 分数和三条 B，验收数量为 `A=N, B=3N`。可先冻结分层抽样子集控制成本，
但抽样子集内部仍必须每回答三任务齐全。

## 8.4 OOD 构造

完整 `3 x 3`：

```text
ID Train: Communication and Advice + Correctness/Factuality
Domain OOD: 其他两个 domain + Correctness/Factuality
Task OOD: Communication and Advice + 其他两个 task
Joint OOD: 其他两个 domain + 其他两个 task
```

## 8.5 是否需要新增 Ground Truth

需要。原始 `orig_score` 和 `orig_feedback` 只对应原 `orig_criteria`；仅当它与某套
冻结 rubric 语义一致时才能复用。其余回答-任务对必须由同一版本强 Teacher 评分，
并保存模型 revision、模板 SHA-256 和评分时间。人工 Ground Truth 测试集另行分层
抽取。

---

# 9. 最终实验分层

## 9.1 完整多任务主实验

```text
DeepResearchBench:
4×4 complete matrix

RealDR:
3×3 complete matrix
```

这两个数据最适合验证：

```text
domain shift
task shift
domain + task joint shift
```

## 9.2 需要补齐后才能进入主实验

```text
RuVerBench:
两个完整 1×4
或共享 rubric 的完整 2×4

BiGGen-Bench:
Grounding / Reasoning / Planning
× Requirement Satisfaction / Correctness / Completeness
= 完整 3×3
```

## 9.3 大规模 Teacher-Label 完整矩阵

```text
FLASK:
Humanities / Coding / Math
× Logical Correctness / Comprehension / Conciseness
= 完整 3×3

Prometheus:
三个派生 domain × 三套固定 rubric
= 完整 3×3
```

## 9.4 当前排除项

```text
VerifyBench-Hard：二分类单任务，不进入当前 hidden-state 运行清单
```

---

# 10. 哪些数据需要补 Ground Truth

## 10.1 当前推荐方案

结论：只有 LongJudgeBench 两个多维子集不需要新增评分；其余数据必须按每回答
每任务审计并补齐。

```text
DeepResearchBench：
直接使用四个原生 Ground Truth

RealDR：
直接使用三个原生 Ground Truth

VerifyBench-Hard：
当前流程已排除

RuVerBench：
复用已有 rubric-level 二分类；缺失回答-任务对补人工评分

BiGGen-Bench：
复用 1,080 个原生 human_score；补 2,160 个回答-任务分数

FLASK：
复用已有唯一回答-任务 Teacher 分数；补 2,281 个缺失对

Prometheus：
原始 custom-rubric 分数按语义复用；其余任务重新 Teacher 评分
```

## 10.2 仍然需要做的工作

除补评分外，仍需完成：

```text
1. domain 聚合；
2. rubric taxonomy 映射；
3. 宽表转长表；
4. cell count 统计；
5. 目标完整子矩阵选择；
6. OOD label 自动生成；
7. 按 sample_id 防泄漏切分。
```

## 10.3 完整性验收

对冻结 `n` 个任务的每个数据集必须同时满足：

```text
unique A = N
unique (input_document_id, task) labels = N x n
unique B = N x n
每个 input_document_id 的 task_count = n
```

只统计 cell 非零、只生成 task metadata 或只保存原始 sparse B 都不能通过验收。

---

# 11. 推荐执行顺序

```text
第一阶段：
DeepResearchBench 4×4
RealDR 3×3

第二阶段：
BiGGen-Bench 补齐完整 3×3
FLASK 补齐完整 3×3

第三阶段：
RuVerBench 两个完整 1×4；需要纯 domain shift 时补共享 2×4
Prometheus 完整 3×3
```

这样先用无需新增评分的 LongJudgeBench 验证算法，再处理需要额外人工或 Teacher
评分的数据，避免继续生成无法用于正式 dense OOD 实验的稀疏 B cache。

---

## 12. 数据来源

- LongJudgeBench official repository:  
  https://github.com/cjj826/LongJudgeBench
- BiGGen-Bench:  
  https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench
- BiGGen-Bench Results:  
  https://huggingface.co/datasets/prometheus-eval/BiGGen-Bench-Results
- FLASK official repository:  
  https://github.com/kaistAI/FLASK
- Prometheus Feedback Collection:  
  https://huggingface.co/datasets/prometheus-eval/Feedback-Collection
