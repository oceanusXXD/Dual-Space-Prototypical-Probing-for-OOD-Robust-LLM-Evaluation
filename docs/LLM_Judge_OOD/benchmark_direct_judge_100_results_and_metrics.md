# Judge 原始任务评测结果

## 结果速览

**主指标为 Balanced ACC 与 Macro-F1。** 数值越高越好；MAE 越低越好；QWK 越接近 1 越好。

| 数据集 | 任务形式 | 标签来源 | Balanced ACC | Macro-F1 | MAE | QWK |
|---|---|---|---:|---:|---:|---:|
| **RuVerBench** | 覆盖判定，二分类 | 人工 | **66.0%** | **64.3%** | -- | -- |
| **Prometheus** | 回答质量，1--5 分 | GPT-4 teacher | **52.7%** | **50.9%** | **0.75 / 4** | **0.673** |
| **BiGGen-Bench** | rubric 质量，1--5 分 | 人工 | 32.0% | 28.1% | 1.18 / 4 | 0.468 |
| **LongJudgeBench** | 长报告质量，0--10 分 | 人工 | 30.7% | 28.2% | 2.94 / 10 | -0.018 |
| **FLASK** | skill 质量，1--5 分 | GPT-4 teacher | 29.3% | 21.1% | 1.51 / 4 | 0.260 |

### 一句话结论

- **二分类覆盖判定**：RuVerBench 最好，但模型偏向预测“已覆盖”（预测 1 占 72.0%）。
- **五级打分**：Prometheus 明显优于其余五级任务；平均只差 0.75 个等级，QWK 为 0.673。
- **人工长文打分**：LongJudgeBench 很弱。模型分数与人工分数几乎没有有序关系，Spearman 为 -0.028，QWK 为 -0.018。
- **低分辨别**：FLASK 和 BiGGen-Bench 都存在偏向高分的现象，尤其 FLASK 有 109/150 条被预测为 5 分。

---

## 原始任务与当前运行的输入输出对照

当前运行是：让 **Qwen3.5-4B 充当 Judge**，直接输出预测标签，
再与公开参考标签比较。对于 BiGGen-Bench、FLASK 这类先生成候选回答、再进行
质量评价的 benchmark，当前实验只测试了后面的 **Judge 评分环节**

### RuVerBench：逐 rubric 覆盖判定

**原始任务输入**

```text
question：原始研究问题
response：一篇完整候选长报告
rubric：该问题对应的一组具体检查点
```

**原始任务输出**

```text
对每个 rubric point 输出 covered = true / false
```

公开数据在一条记录中保存整个 `coverage_results` 列表，但实际计分单元是单个
`(question, response, rubric point)`。同一篇报告对应多少个 rubric point，就有
多少个二分类判断。

**当前运行输入**

```text
Dataset / domain：RuVerBench / Deep Research
Task：Coverage
Question/Instruction：原始问题
Rubric/check：从原 rubric 列表取出的一条检查点
Candidate response：完整候选报告
```

**当前运行输出**

```json
{"label": 0}
```

其中 `1` 表示覆盖，`0` 表示未覆盖；不输出理由。当前把原始记录中的多个
rubric point 展开为多条独立样本。

**一致性判断：高度一致。** 输入信息和二分类标签含义均一致；差别主要是把
列表式输出展开成单条 JSON，并且当前只跑了 Deep Research，没有跑 RuVerBench
的 Agentic Coding 部分。

### Prometheus：自定义 rubric 反馈与评分

**原始任务输入**

```text
Instruction to evaluate：原始用户要求
Response to evaluate：候选回答
Reference answer：一份应获得 5 分的参考答案
Score rubric：本样本的自定义 criteria
Score descriptions：1、2、3、4、5 分分别应满足的描述
```

**原始任务输出要求**

```text
Feedback: 针对该 criteria 的详细评价 [RESULT] 1--5 的整数分数
```

也就是说，原始 Prometheus 同时要求 **生成反馈** 和 **给出分数**，而且明确
要求评分必须依据给定的参考答案和五档描述。

**当前运行输入**

```text
实验映射的 domain：Communication and Advice、
                    Knowledge/Technical Explanation 或 Writing and Revision
实验映射的 task：Correctness/Factuality、Completeness/Instruction Fulfillment
                 或 Style/Context Adaptation
Question/Instruction：orig_instruction
Rubric/check：orig_criteria
Candidate response：orig_response
```
**当前运行输出**

```json
{"score": 3}
```

只允许输出 1--5 的整数，不生成反馈。

**一致性判断：只与原始评分子任务一致，不是完整原任务。** 当前预测的仍是同一
个 `orig_score`，但输入删去了参考答案和五档评分说明，输出也删去了详细反馈。

### BiGGen-Bench：实例专属 rubric 的人工评分

BiGGen-Bench 最初首先是一个生成能力 benchmark：被测生成模型接收
`input/instruction`，输出 `response`。当前实验没有重测这个生成任务，而是使用
已经生成好的 response，测试 Qwen3.5-4B 能否复现其人工评分。

**原始人工评分环节输入**

```text
Input/Instruction：原始生成任务
Candidate response：待评价回答
Reference answer：参考回答
Score rubric criteria：该实例专属的评价标准
Score 1--5 descriptions：五个分数档各自的详细定义
```

**原始人工评分环节输出**

```text
human_score：1--5 的整数
```

部分原始记录还有 `final_answer_correct` 等辅助字段，但当前结果以
`human_score` 为唯一参考标签。

**当前运行输入**

```text
实验映射的 domain：Grounding、Reasoning 或 Planning
实验映射的 task：Requirement/Grounding Satisfaction、Correctness 或 Completeness
Question/Instruction：原始 input
Rubric/check：仅 score_rubric.criteria 主标准
Candidate response：原始 response
```

当前没有放入 reference answer，也没有放入 1--5 分的逐档详细描述；三组
domain/task 是实验映射，不是 BiGGen-Bench 原始输出类别。

**当前运行输出**

```json
{"score": 1}
```

只允许输出 1--5 的整数，不输出评分理由或其他辅助判断。

**一致性判断：人工分数标签一致，但评分条件被简化。** 当前可以解释为
“复现 `human_score` 的简化 Judge 任务”，不能解释为 Qwen3.5-4B 在原始
BiGGen 生成任务上的效果。

### LongJudgeBench：七个长报告质量维度

当前按实验要求只使用 DeepResearchBench 和 RealDR，不包含 VerifyBench-Hard。

**原始 DeepResearchBench 输入与输出**

```text
输入：instruction + 完整候选报告 + 每个维度的详细 criteria
输出：comprehensiveness、insight、instruction_following、readability 四项分数，
      以及 overall / weighted_total
原生聚合分数范围：0--100
```

**原始 RealDR 输入与输出**

```text
输入：instruction + 完整候选报告 + 多维评价要求
输出：logical_structure、expression、bias_check 三项分数，
      以及 weighted_total
原生分数范围：0--10
```

**当前运行输入**

每次只评价七个维度中的一个：

```text
Dataset：DeepResearchBench 或 RealDR
Evaluation task/dimension：当前单个维度
Original task instruction：原始问题
Scoring rubric：当前维度的 rubric
Candidate report to evaluate：完整候选报告
```

DeepResearchBench 使用该维度的逐项 criteria 与权重；RealDR 使用当前维度的
定义以及 `9--10 excellent / 7--8 good / 5--6 pass / 0--4 major deficiencies`
分档说明。入选提示词不做文本截断；受 T4 显存限制，只从完整提示词不超过
5,500 tokens 的候选池中抽样。

**当前运行输出**

```json
{"score": 8.5}
```

允许输出 0--10 的整数或小数，不输出解释。DeepResearchBench 的人工分数先从
0--100 除以 10，统一到 0--10。模型输出后，才为 Balanced ACC 和 Macro-F1
额外转换成三档：`<7`、`[7, 9)`、`>=9`；三档不是模型被要求输出的内容。

**一致性判断：单维分数含义一致，输入输出组织方式不完全一致。** 当前将原始
一次多维评价拆成七种单维任务，并统一到 0--10；它符合本次“七项任务分别测”
的要求，但不是原 benchmark 的一次性多维输出或加权总分任务。

### FLASK：逐 skill 回答质量评价

FLASK 先要求生成模型根据 `text` 产生回答；随后评价器从多个 skill 角度评价
该回答。当前实验只测试后面的评分环节。

**原始评价输入**

```text
text：原始问题或指令
answer：参考答案；部分样本可能没有
target_txt：候选回答
metrics：本样本需要评价的多个 skill
metric_explanation：各 skill 对本样本具体检查什么
```

一条样本可能同时包含 Readability、Logical Correctness、Conciseness 等多个
skill。

**原始评价输出**

```text
review：逐 skill 的文字评价和 Score
score：{skill_1: 1--5, skill_2: 1--5, ...}
```

**当前运行输入**

```text
实验映射的 domain：Humanities、Coding 或 Math
Task/rubric：Logical Correctness、Comprehension 或 Conciseness 中的一项
Rubric/check：当前 skill 名称
Question/Instruction：text
Candidate response：target_txt
```

当前将一条多 skill 记录展开为多条单 skill 样本，没有放入 `answer` 参考答案，
也没有放入原始 `metric_explanation` 的详细检查说明。

**当前运行输出**

```json
{"score": 5}
```

只允许输出当前一个 skill 的 1--5 整数，不生成逐 skill review。

**一致性判断：单 skill 标签含义一致，但原任务被拆分并简化。** 当前结果衡量
Qwen3.5-4B 复现 GPT-4 逐 skill 分数的能力，不是模型在 FLASK 原始回答生成
任务上的效果，也不是完整的多 skill 反馈生成效果。

### 对照结论

| 数据集 | 当前输出与原始标签是否同义 | 端到端输入输出是否一致 |
|---|---|---|
| RuVerBench | 是，`0/1` 对应 `covered=false/true` | **基本一致**；仅展开 rubric，并少了其他子集 |
| Prometheus | 是，预测同一个 1--5 分 | **不一致**；缺参考答案、五档说明和反馈输出 |
| BiGGen-Bench | 是，预测 `human_score` | **部分一致**；是简化评分，不是原始回答生成任务 |
| LongJudgeBench | 是，缩放后对应原始单维人工分 | **部分一致**；多维输出被拆成单维，分类档位为派生结果 |
| FLASK | 是，对应原始单 skill GPT-4 分数 | **部分一致**；多 skill 被拆分，缺参考答案、详细说明和 review |

---

## 各数据集在测什么

### RuVerBench

- **领域**：Deep Research 长报告。
- **任务**：候选报告是否覆盖某一个明确的 rubric 要求。
- **标签**：人工 `covered`，`1` 为覆盖、`0` 为未覆盖。
- **平衡样本**：150 条，`0/1 = 75/75`。
- **结果**：Balanced ACC 66.0%，Macro-F1 64.3%。

### Prometheus Feedback Collection

- **领域**：建议、技术解释、写作和修订。
- **任务**：Correctness、Completeness、Style/Context Adaptation 等自定义 rubric 打分。
- **标签**：GPT-4 custom-rubric teacher score，1--5 分，**不是人工金标**。
- **平衡样本**：150 条，1--5 分各 30 条。
- **结果**：Balanced ACC 52.7%，Macro-F1 50.9%，MAE 0.75，QWK 0.673。

### BiGGen-Bench

- **领域**：Grounding、Reasoning、Planning。
- **任务**：Requirement-Grounding Satisfaction、Correctness、Completeness 等 instance-specific rubric 判定。
- **标签**：人工 `human_score`，1--5 分；每条样本使用自身的评分 rubric。
- **平衡样本**：150 条，1--5 分各 30 条。
- **结果**：Balanced ACC 32.0%，Macro-F1 28.1%，MAE 1.18，QWK 0.468；预测明显偏向 5 分。

### LongJudgeBench

- **领域**：长篇研究报告质量评估。
- **任务**：只包含 DeepResearchBench 的 `comprehensiveness`、`insight`、`instruction_following`、`readability`，以及 RealDR 的 `logical_structure`、`expression`、`bias_check`。
- **不包含**：VerifyBench-Hard 二分类正确性任务。
- **标签**：人工连续 0--10 分。为计算分类主指标，映射为低 `<7`、中 `[7, 9)`、高 `>=9` 三档。
- **平衡样本**：150 条，低/中/高各 50 条；七项任务均有覆盖。
- **结果**：Balanced ACC 30.7%，Macro-F1 28.2%，MAE 2.94 分，QWK -0.018，Spearman -0.028。

### FLASK

- **领域**：Humanities、Coding、Math 等指令任务。
- **任务**：Logical Correctness、Comprehension、Conciseness 等 skill 判定。
- **标签**：公开 GPT-4 逐 skill teacher score，1--5 分，**不是人工金标**。
- **平衡样本**：150 条，1--5 分各 30 条。
- **结果**：Balanced ACC 29.3%，Macro-F1 21.1%，MAE 1.51，QWK 0.260；模型几乎总给高分。

---

## 抽样与完整文本协议

| 项目 | 规则 |
|---|---|
| 类别平衡 | 五级任务每一分数各 30 条；二分类各 75 条；LongJudge 三个分数段各 50 条 |


---

## 指标怎么读

### ACC

完全匹配比例：

```text
ACC = (1 / N) * sum 1[prediction_i == reference_i]
```

真值 4、预测 5 也算错。ACC 在类别不均衡时会被多数类抬高，因此本报告不把它作为主比较指标。

### Balanced ACC

每一类 recall 的平均值；二分类时：

```text
Balanced ACC = (TPR + TNR) / 2
```

它让各类别同权，适合判断模型是否真的会识别少数类。

### Macro-F1

先对每个类别计算 F1，再取平均。它同时要求 precision 与 recall 都好，能发现模型是否只偏向输出某一个类别或高分档。

### MAE

只用于有明确分数距离的 1--5 或 0--10 任务：

```text
MAE = (1 / N) * sum |prediction_i - reference_i|
```

它表示平均差多少分。BiGGen 的 1.18 表示平均偏离人工分 1.18 个等级；LongJudge 的 2.94 表示平均偏离人工 2.94 分。二分类不报告 MAE。

### QWK

Quadratic Weighted Kappa 衡量有序评分的一致性，对远距离错误的惩罚更大。`1` 表示完全一致，`0` 约等于随机一致，负值表示比该随机基线更差。QWK 能区分“4 预测成 5”和“1 预测成 5”这两种错误；不用于 RuVerBench 二分类。

---
