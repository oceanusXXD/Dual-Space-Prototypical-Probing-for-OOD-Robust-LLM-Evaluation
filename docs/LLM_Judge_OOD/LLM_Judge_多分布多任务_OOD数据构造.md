# LLM Judge 多分布 × 多任务 OOD 数据构造与 Ground Truth 方案

> 目标：利用现有数据集中的原生 Ground Truth，构造“多个 Benchmark/domain × 多个 Task/rubric”的 OOD 实验。
>
> 核心原则：**不要求所有数据集统一成 3×3，也不要求每篇文本拥有所有任务标签。**  
> 只要数据集中存在多个分布和多个评分任务，并且每个保留的 `domain × task` 单元有足够的原生 Ground Truth，就可以构造多×多 OOD 实验。
>
> 本文统一将数据集提供的参考标签或参考分数记为 **Ground Truth**，不再区分其生成来源。

---

## 1. 最终结论

在下面三个条件成立时，**不需要新增任何 Ground Truth**：

1. 不强行补齐原始数据中不存在的 `domain × task` 单元；
2. 不把一个已有分数复制成多个不同任务的标签；
3. FLASK 和 Prometheus 接受原始 评分作为 Ground Truth，而不是声称为Ground Truth。

需要新增的只是：

```text
domain_code
task_code
shift_type
ood_label
```

这些是实验元数据，不是新的Ground Truth。

最终建议如下：

| 数据集                             |                       可用结构 | 矩阵类型              |       是否新增 Ground Truth |
| ---------------------------------- | -----------------------------: | --------------------- | --------------------------: |
| LongJudgeBench / DeepResearchBench |                          `4×4` | 完整多任务矩阵        |                      不需要 |
| LongJudgeBench / RealDR            |                          `3×3` | 完整多任务矩阵        |                      不需要 |
| LongJudgeBench / VerifyBench-Hard  |                   外部 far-OOD | 单任务数据            | 不需要，但不能独立构造多×多 |
| RuVerBench                         |   两个 `1×4`；或异构稀疏 `2×4` | rubric-level 稀疏矩阵 |                      不需要 |
| BiGGen-Bench                       | `9×K`，K 由 rubric family 决定 | 稀疏矩阵              |                      不需要 |
| FLASK                              |               原生稀疏 `10×12` | 稀疏矩阵              |                      不需要 |
| Prometheus                         |                     派生 `M×N` | 稀疏矩阵              |                      不需要 |

| 数据集                | 多 Domain 文本分布                                                                                                                                           | 多任务类型分布                                                                                                                                                                                                                |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DeepResearchBench** | 官方有 **22 个研究领域**。实验中可聚合为：① Science & Technology；② Business & Finance；③ Software & Internet；④ Humanities / Society / Other Applied Fields | 原生 4 个任务：**Comprehensiveness、Insight/Depth、Instruction Following、Readability**。([GitHub][1])                                                                                                                        |
| **RealDR**            | 官方包含 **12 个学科领域**。可聚合为：① Science & Engineering；② Business & Applied Studies；③ Humanities & Social Sciences                                  | 原生 3 个任务：**Logical Structure、Presentation Form、Bias Checking**。LongJudgeBench 将 RealDR 定义为三维评分数据。([Emergent Mind][2])                                                                                     |
| **VerifyBench-Hard**  | 没有适合主实验的多 Domain 划分，只能按题型或错误类型形成若干子分布                                                                                           | 只有一个任务：**Correctness Verification**，因此不能单独构造多 Domain × 多 Task                                                                                                                                               |
| **RuVerBench**        | 2 个原生文本分布：① **DeepResearch** 长报告；② **AgenticCoding** 代码代理轨迹                                                                                | DeepResearch：**Format、Numbers、Logic、Facts**；AgenticCoding：**Task Completion、Planning、Tool Use、Rules/Compliance**。两个 Domain 的任务类型不同，因此属于异构多任务                                                     |
| **BiGGen-Bench**      | 9 个 capability 分布：**Instruction Following、Grounding、Reasoning、Planning、Refinement、Multilingual、Safety、Theory of Mind、Tool Usage**                | 77 个具体任务，例如 personal assistant、compositional planning 等。任务隶属于 capability，是层级结构，不能直接视为完整 `9×77` 或 `9×9` 矩阵。([Hugging Face][3])                                                              |
| **FLASK**             | 10 个原生 Domain：**Language、Culture、Health、History、Natural Science、Math、Social Science、Technology、Coding、Humanities**。                            | 12 个原生 Skill：**Logical Robustness、Logical Correctness、Logical Efficiency、Commonsense Understanding、Factuality、Metacognition、Insightfulness、Completeness、Comprehension、Conciseness、Readability、Harmlessness**。 |
| **Prometheus**        | 没有原生统一 Domain。需要根据 instruction 自己分类，例如：① Knowledge/Technical；② Reasoning；③ Writing；④ Advice/Communication；⑤ Creative；⑥ Coding        | 没有原生统一任务类型。需要根据 criterion 分类，例如：① Correctness；② Instruction Fulfillment；③ Completeness；④ Clarity/Coherence；⑤ Style；⑥ Safety                                                                         |

最清楚的分类是：

- **完整多 Domain + 多任务**：DeepResearchBench、RealDR。
- **原生多 Domain + 原生多任务，但矩阵稀疏**：FLASK。
- **多 capability + 多下属任务**：BiGGen-Bench。
- **两个异构 Domain，各有自己的任务体系**：RuVerBench。
- **需要自己分类**：Prometheus。
- **只有单任务**：VerifyBench-Hard。

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

## 2.2 稀疏矩阵

每篇文本只覆盖部分任务，甚至只有一个任务标签：

```text
文本 1：D1 + T1 → Y
文本 2：D1 + T3 → Y
文本 3：D2 + T1 → Y
```

整个数据集仍可能覆盖多个 domain 和多个 task，但并非每篇文本都有全部标签。

代表数据：

```text
RuVerBench
BiGGen-Bench
FLASK
Prometheus
```

稀疏矩阵同样可以构造 OOD，但必须先统计每个 cell 的样本量，再选择样本充分的最大子矩阵。

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

## 3.3 选择最大可用子矩阵

设置最低样本数：

```text
建议初筛：每个 cell ≥ 50
正式实验：每个 cell 最好 ≥ 100
```

选择最大的：

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
- 不为补齐矩阵伪造或重新复制标签。

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

只使用三个 pointwise 子集：

```text
DeepResearchBench
RealDR
VerifyBench-Hard
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

## 4.3 VerifyBench-Hard：只作为 external far-OOD

VerifyBench-Hard 原生只有：

```text
Task = Correctness Verification
Label = correct / incorrect
```

它无法在自身内部构造多 domain × 多 task。

正确用法是：

```text
DeepResearchBench 或 RealDR 作为 ID 主矩阵
VerifyBench-Hard 作为 external far-OOD
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

形成：

```text
1 domain × 4 tasks
```

## 5.2 AgenticCoding：`1×4`

原生 taxonomy：

```text
Task Q = Task Completion
Task W = Planning
Task E = Tool Use
Task R = Rules / Compliance
```

形成：

```text
1 domain × 4 tasks
```

## 5.3 能否构造 `2×4`

数据结构上可以合并为：

```text
2 domains × 4 task positions
= 异构稀疏 2×4
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

主报告分别做：

```text
DeepResearch 1×4：只研究 task shift
AgenticCoding 1×4：只研究 task shift
```

再把两个 domain 之间的变化作为：

```text
cross-domain + cross-task comprehensive OOD
```

### 是否需要新增 Ground Truth

```text
完全不需要。
```

若强行要求两个 domain 共享完全相同的四个 task，则需要重新定义和重新标注；当前方案不这样做。

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

## 6.2 无需新 GT 的正确构造：稀疏 `9×K`

### Domain 轴

直接使用九个原生 capability：

```text
D1 ... D9 = 9 capabilities
```

### Task 轴

对 `score_rubric.criteria` 做确定性 taxonomy 映射。建议先使用五个跨 capability 的 rubric families：

```text
T1 = Requirement / Instruction Satisfaction
T2 = Correctness / Validity
T3 = Completeness / Coverage
T4 = Coherence / Overall Quality
T5 = Safety / Constraint Compliance
```

每条样本只分配到一个最主要的 task family：

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

这一步只是生成 task metadata，不改变原始 `human_score`。

### 统计和筛选

建立：

\[
N\_{\text{capability},\text{rubric family}}
\]

候选矩阵为：

```text
9×5 sparse matrix
```

随后选择所有 cell 都达到最低样本数的最大子矩阵，例如：

```text
6×4
5×5
7×3
```

最终矩阵大小由真实计数决定，不能预先声称一定是 `9×5` 或 `9×9`。

### OOD 构造示例

若最终保留 `6×4`：

```text
ID Train:
Grounding + Requirement Satisfaction

Domain OOD:
Reasoning + Requirement Satisfaction
Planning + Requirement Satisfaction
...

Task OOD:
Grounding + Correctness
Grounding + Completeness
Grounding + Coherence

Joint OOD:
Reasoning + Correctness
Planning + Completeness
...
```

### 是否需要新增 Ground Truth

```text
完全不需要。
```

前提是：

- 只使用已有 `human_score` 作为 Ground Truth；
- 不补空 cell；
- 不要求同一文本具有多个 rubric family 分数。

如果要求真正 dense `9×9`，才需要对每篇回答重新进行九次评分；本方案不做。

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

## 7.5 构造最大稠密子矩阵

先建立真实 `10×12` count matrix：

\[
N\_{d,s}
\]

再选择满足最低样本数的最大子矩阵：

```text
K domains × L skills
```

结果可能是：

```text
7×5
6×6
5×8
```

不需要强行降为 `4×4`。

## 7.6 OOD 构造

假设最终得到 `6×6`：

```text
ID Train:
Domain A + Logical Correctness

Domain OOD:
其他 5 个 domain + Logical Correctness

Task OOD:
Domain A + 其他 5 个 skills

Joint OOD:
其他 domain + 其他 skill
```

必须按原始 sample_id 分组切分。

### 是否需要新增 Ground Truth

```text
完全不需要。
```

前提是接受 原始 skill score 作为 Ground Truth。

若要求所有 10×12 cell 都是评分，或要求每篇文本都有全部 12 个 skill score，才需要重新标注；当前方案不要求。

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

它没有统一 domain 字段，但数据规模足够大，可以派生多×多稀疏矩阵。

## 8.1 Domain 轴

根据 `orig_instruction` 做固定 taxonomy 分类。建议初始设置六类：

```text
D1 = Knowledge and Technical Explanation
D2 = Reasoning and Problem Solving
D3 = Writing and Revision
D4 = Communication and Advice
D5 = Creative Generation
D6 = Coding and Structured Tasks
```

分类步骤：

1. 固定类别定义；
2. 规则或固定模型分类；
3. 保存置信度；
4. 低置信样本删除；
5. 抽样复核。

## 8.2 Task 轴

根据 `orig_criteria` 将约 996 个 rubric 归为六类：

```text
T1 = Correctness / Factuality
T2 = Instruction Fulfillment
T3 = Completeness / Coverage
T4 = Clarity / Coherence
T5 = Style / Context Adaptation
T6 = Safety / Appropriateness
```

每条数据只分配到一个最主要的 task family。

## 8.3 候选矩阵

初始候选：

```text
6 domains × 6 rubric families
= sparse 6×6
```

随后统计真实 cell 数量，并删除过少的 domain 或 task，得到最大可用子矩阵。

由于总数据量接近 10 万，Prometheus 比 BiGGen-Bench 更可能形成较大的稠密子矩阵。

## 8.4 OOD 构造

例如最终保留 `5×6`：

```text
ID Train:
Knowledge Explanation + Correctness

Domain OOD:
其他 domain + Correctness

Task OOD:
Knowledge Explanation + 其他 rubric families

Joint OOD:
其他 domain + 其他 rubric families
```

## 8.5 是否需要新增 Ground Truth

```text
完全不需要。
```

保留每条数据原有的：

```text
orig_score
orig_feedback
```

domain 和 task family 只是派生元数据。

如果要求同一回答同时拥有六个 rubric 分数，才需要重新 teacher 评分；当前方案不要求。

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

## 9.2 稀疏矩阵实验

```text
RuVerBench:
两个 1×4
或异构 2×4

BiGGen-Bench:
9×K sparse matrix
最终使用最大稠密子矩阵
```

## 9.3 大规模稀疏矩阵实验

```text
FLASK:
原生 sparse 10×12
最终使用最大稠密子矩阵

Prometheus:
派生 sparse M×N
最终使用最大稠密子矩阵
```

## 9.4 External far-OOD

```text
VerifyBench-Hard
```

---

# 10. 是否完全不需要补 Ground Truth

## 10.1 当前推荐方案

结论：

```text
是，完全不需要新增 Ground Truth。
```

具体包括：

```text
DeepResearchBench：
直接使用四个原生 Ground Truth

RealDR：
直接使用三个原生 Ground Truth

VerifyBench-Hard：
直接使用二分类 Ground Truth

RuVerBench：
直接使用rubric-level 二分类 Ground Truth

BiGGen-Bench：
直接使用 human_score

FLASK：
直接使用原始 skill-level Ground Truth

Prometheus：
直接使用原始 custom-rubric Ground Truth
```

## 10.2 仍然需要做的工作

以下工作不属于 Ground Truth 标注：

```text
1. domain 聚合；
2. rubric taxonomy 映射；
3. 宽表转长表；
4. cell count 统计；
5. 最大稠密子矩阵选择；
6. OOD label 自动生成；
7. 按 sample_id 防泄漏切分。
```

## 10.3 哪些要求会迫使你补 Ground Truth

只有提出以下额外要求时才需要补标：

```text
1. 每篇文本必须拥有所有 task 的分数；
2. 所有理论 cell 必须被填满；
3. BiGGen-Bench 必须做 dense 9×9；
4. FLASK 每篇文本必须有全部 12 个 skill 分数；
5. Prometheus 每篇文本必须有全部 rubric family 分数；
6. FLASK 和 Prometheus 必须全部改成Ground Truth。
```

当前实验不需要这些要求，因此不需要补 Ground Truth。

---

# 11. 推荐执行顺序

```text
第一阶段：
DeepResearchBench 4×4
RealDR 3×3

第二阶段：
FLASK sparse 10×12 → 最大稠密子矩阵
BiGGen-Bench 9×K → 最大稠密子矩阵

第三阶段：
Prometheus M×N → 最大稠密子矩阵
RuVerBench 两个 1×4

外部测试：
VerifyBench-Hard far-OOD
```

这样能够先用两个完整人工多任务数据验证算法，再用更大的稀疏矩阵测试泛化能力。

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
