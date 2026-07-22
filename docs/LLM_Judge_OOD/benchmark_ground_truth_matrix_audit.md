# LLM Judge OOD 完整 Domain x Task 矩阵与 B-space 缺口审计

本文记录五个 benchmark-ground-truth 数据集的目标矩阵、合法 Ground Truth、
A/B hidden state 和当前缺口。所有领域与任务均写出完整名称，不使用单独的
`A/B/C`、`Q/W/E` 代称。

## 0. 必须遵守的验收口径

### 0.1 目标矩阵必须完整

一旦为某个实验冻结了 `m` 个 domain 和 `n` 个 task，正式数据必须是完整的
`m x n`，不能用稀疏对角线代替：

```text
每个保留的候选回答 X
  x 每个冻结的任务 rubric T_j
  -> 一条有合法标签 Y 的 Judge 样本
  -> 一个对应的 B-space hidden state
```

因此，完整性不是“矩阵中的每格至少偶然出现过一些样本”，而是：

1. 每个保留的 `input_document_id` 都有全部目标任务的标签；
2. 每个 `(input_document_id, task)` 只保留一条规范化 Judge 样本；
3. 每条规范化 Judge 样本都有对应 B-space hidden state；
4. 同一回答跨任务共用同一个 A-space hidden state；
5. 缺失标签必须记为“待补评分”，不能复制其他任务的标签、补零或只改任务名。

对于一个含 `N` 个唯一候选回答、冻结 `n` 个任务的数据集，验收数量应满足：

```text
A-space 唯一向量数 = N
B-space 唯一向量数 = N x n
```

原始数据可以是稀疏的，但稀疏原始数据只是标签来源，不代表正式 OOD 矩阵已经
完成。没有补齐前，该数据集只能用于数据审计或探索实验，不能作为完整
Domain-OOD、Task-OOD 和 Joint-OOD 主实验。

### 0.2 A-space 与 B-space

- **A-space**：仅输入原始候选回答，按 `input_document_id` 提取一次，不含 domain、
  task、rubric、split 或标签。
- **B-space**：对每个 `(input_document_id, task)` 单独输入数据集、所属 domain、
  当前 task、当前 rubric、原始 instruction 和候选回答。
- Ground Truth、ID/OOD 标记和 train/test split 都只保存在 metadata 中，不进入
  B-space prompt。
- 如果 task 定义、rubric 文本或 prompt 模板改变，A 可以复用，B 必须按新模板
  重新生成。

### 0.3 当前总状态

状态快照：**2026-07-22**。

| 数据集 | 冻结目标 | A-space | 当前 B-space | 完整矩阵状态 | 还缺什么 |
|---|---:|---:|---:|---|---|
| LongJudgeBench / DeepResearch-Bench | `4 x 4` | 未运行，目标 200 | 未运行，目标 800 | 未完成 | domain 聚合、四维宽表转长表、A/B 提取；原生人工四维标签已存在 |
| LongJudgeBench / RealDR | `3 x 3` | 未运行，目标 640 | 未运行，目标 1,920 | 未完成 | domain 聚合、三维宽表转长表、A/B 提取；原生人工三维标签已存在 |
| RuVerBench | 两个完整 `1 x 4`；若做纯 domain shift，需另建共享 `2 x 4` | Deep Research 284 个旧 Colab A 未归档；Agentic Coding 未接入 | Deep Research 1,615 条旧 rubric-check 分片未归档；Agentic Coding 未接入 | 未完成 | taxonomy、每回答四任务覆盖审计、缺项补评分、Agentic Coding 适配和全部 A/B 提取 |
| BiGGen-Bench | `3 x 3` | 已有 1,080，可复用 | 已有 1,080 条旧对角线 B；完整目标 3,240 | 未完成 | 2,160 个回答-任务标签；冻结统一 rubric 后生成完整 B |
| FLASK | `3 x 3` | 已有 1,691，可复用 | 已有 3,576 条原始 B 行，但只有 2,792 个唯一回答-任务对；完整目标 5,073 | 未完成 | 去除 784 条重复回答-任务行，补 2,281 个唯一回答-任务评分并生成完整 B |
| Prometheus | `3 x 3` | 未运行，正式 prepare 后确定 N | 未运行，目标为 `3N` | 未完成 | domain 分类、固定三套 rubric、每回答三任务 Teacher 评分、A/B 提取 |

本机现有两个归档仍保留，但应标注为 **legacy sparse B-space**：

```text
/tmp/llm-judge-ood-local/biggen_bench/
  biggen_bench_benchmark_gt_hiddenstate_ab.tar.gz

/tmp/llm-judge-ood-colab/flask/
  flask_benchmark_gt_hiddenstate_ab.tar.gz
```

它们证明提取程序可以运行，不证明目标 dense B-space 已完成。

## 1. LongJudgeBench

LongJudgeBench 的两个多分类、多维 pointwise 子集应分别建矩阵；不纳入二分类
`VerifyBench-Hard`。

### 1.1 DeepResearch-Bench：完整 4 x 4

领域：

1. Science and Technology（科学与技术）
2. Business and Finance（商业与金融）
3. Society and Public Policy（社会与公共政策）
4. Humanities and Culture（人文与文化）

任务：

1. Comprehensiveness（全面性）
2. Insight / Depth（洞察与深度）
3. Instruction Following（指令遵循）
4. Readability（可读性）

每个候选报告原生具有四个维度的人工聚合分数，因此每个回答应展开成四条 B：

| 领域/分布 | 全面性 | 洞察与深度 | 指令遵循 | 可读性 |
|---|---|---|---|---|
| 科学与技术 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 |
| 商业与金融 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 |
| 社会与公共政策 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 |
| 人文与文化 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 |

当前适配器把四维分数加权成一个总任务，只会得到 200 条 B。该实现不满足目标；
正确目标是 `A=200, B=200 x 4=800`。不缺人工评分，只缺正确展开和提取。

### 1.2 RealDR：完整 3 x 3

领域：

1. Science and Engineering（科学与工程）
2. Business and Applied Studies（商业与应用研究）
3. Humanities and Social Sciences（人文与社会科学）

任务：

1. Logical Structure（逻辑结构）
2. Presentation Form（表达形式）
3. Bias Checking（偏见检查）

| 领域/分布 | 逻辑结构 | 表达形式 | 偏见检查 |
|---|---|---|---|
| 科学与工程 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 |
| 商业与应用研究 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 |
| 人文与社会科学 | 待生成；原生标签已存在 | 待生成；原生标签已存在 | 待生成；原生标签已存在 |

当前适配器只会生成 640 条加权总分 B，不满足目标。正确目标是
`A=640, B=640 x 3=1,920`。两个 LongJudgeBench 子集合计目标为
`A=840, B=2,720`。

## 2. RuVerBench

RuVerBench 有两个真实文本分布：Deep Research 长报告和 Agentic Coding 代理轨迹。
两部分原生 taxonomy 不同，不能把“Agentic Coding 正确性”和“Agentic Coding
合规性”伪装成两个 domain。

### 2.1 当前可辩护的目标

先分别构造两个完整的四任务面板：

| 文本分布 | 四个原生任务 |
|---|---|
| Deep Research | Format（格式）、Numbers（数值）、Logic（逻辑）、Facts（事实） |
| Agentic Coding | Task Completion（任务完成）、Planning（规划）、Tool Use（工具使用）、Rules / Compliance（规则与合规） |

这里的“完整”同样要求每个回答在四个任务上都有 B，而不是只把已有 check 分到
四类后看见四列非空。Deep Research 已知 284 个唯一回答，因此若规范化为每回答
每任务一条，目标至少是 `A=284, B=1,136`；当前 1,615 条 rubric-check 只是原始
检查项行数，尚未证明 284 个回答都覆盖四类。

Agentic Coding 的四份原始文件已经下载，但适配器、回答数和四任务覆盖尚未审计，
因此不能填写完成数量。

### 2.2 共享 2 x 4 的限制

如果要把两个文本分布放进同一个 `2 x 4` 做纯 Domain-OOD，必须先冻结四个跨域
同义任务，例如：

1. Requirement / Task Fulfillment（要求或任务完成）
2. Correctness / Factuality（正确性或事实性）
3. Reasoning / Planning Quality（推理或规划质量）
4. Format / Tool / Rule Compliance（格式、工具与规则合规）

然后对两个 domain 的每个回答都按这四个 rubric 评分并生成 B。原生 check 能直接
支持的标签可复用；没有对应 check 的回答-任务对必须补人工评分。完成该步骤之前，
RuVerBench 只能称为两个异构 `1 x 4`，不能声称完成纯 Domain-OOD。

## 3. BiGGen-Bench：目标 3 x 3

领域：Grounding（事实依据与约束绑定）、Reasoning（推理）、Planning（规划）。

任务：Requirement/Grounding Satisfaction（要求与事实依据满足度）、
Correctness（正确性）、Completeness（完整性）。

### 3.1 当前旧矩阵

| 领域/分布 | 要求与事实依据满足度 | 正确性 | 完整性 | 唯一回答数 |
|---|---:|---:|---:|---:|
| Grounding（事实依据与约束绑定） | 400 | 0 | 0 | 400 |
| Reasoning（推理） | 0 | 400 | 0 | 400 |
| Planning（规划） | 0 | 0 | 280 | 280 |

`A=1,080, B=1,080` 的原因是旧适配器把 capability 同时映射为 domain 和 task，
只生成了三个对角格。每个回答原生只有一个 instance-specific rubric 和一个
`human_score`，所以旧数据不是完整 `3 x 3`。

### 3.2 必须补齐的 B-space

| 领域/分布 | 要求与事实依据满足度 | 正确性 | 完整性 | 缺失 B |
|---|---|---|---|---:|
| Grounding（事实依据与约束绑定） | 已有 400 个原生评分 | 400 个待补评分 | 400 个待补评分 | 800 |
| Reasoning（推理） | 400 个待补评分 | 已有 400 个原生评分 | 400 个待补评分 | 800 |
| Planning（规划） | 280 个待补评分 | 280 个待补评分 | 已有 280 个原生评分 | 560 |

完整目标是：

```text
A = 1,080
B = 1,080 x 3 = 3,240
缺失标签和 B = 2,160
```

若继续保持 Human-GT track，2,160 个缺失分数必须由人工按冻结的三套 rubric 补标。
若改用强模型补分，则必须单独标成 Teacher-Label track，不能与原 `human_score`
混称人工 Ground Truth。三套 rubric 冻结后应生成一个新版本的完整 B cache；旧
1,080 条 B 仅保留作稀疏流程记录。

## 4. FLASK：目标 3 x 3

领域：Humanities（人文学科）、Coding（编程）、Math（数学）。

任务：Logical Correctness（逻辑正确性）、Comprehension（理解能力）、
Conciseness（简洁性）。

FLASK 当前归档有 `A=1,691` 和 3,576 条 B 行，但 3,576 中包含同一
`(input_document_id, task)` 的重复行。按唯一回答-任务对审计后，只有 2,792 对；
完整目标应为 5,073 对。

| 领域/分布 | 唯一回答 | 已有逻辑正确性 | 已有理解能力 | 已有简洁性 | 完整 B 目标 | 缺失唯一回答-任务对 |
|---|---:|---:|---:|---:|---:|---:|
| Humanities（人文学科） | 1,337 | 988 | 1,003 | 273 | 4,011 | 1,747 |
| Coding（编程） | 159 | 143 | 73 | 20 | 477 | 241 |
| Math（数学） | 195 | 190 | 71 | 31 | 585 | 293 |
| 合计 | 1,691 | 1,321 | 1,147 | 324 | 5,073 | 2,281 |

另有 784 条重复回答-任务行不能用于证明任务覆盖，应在新 prepared contract 中按
`(input_document_id, task)` 规范化。缺失的 2,281 对需要使用同一版本强 Teacher
按冻结 rubric 补评分；若论文要报告 Human-GT，还需从九格抽取平衡测试集做双人
评分和仲裁。A cache 可以复用，完整 B 应生成新 cache。

## 5. Prometheus：目标 3 x 3

领域：

1. Communication and Advice（沟通与建议）
2. Knowledge/Technical Explanation（知识与技术解释）
3. Writing and Revision（写作与修订）

任务：

1. Correctness/Factuality（正确性与事实性）
2. Completeness/Instruction Fulfillment（完整性与指令满足）
3. Style/Context Adaptation（风格与上下文适配）

| 领域/分布 | 正确性与事实性 | 完整性与指令满足 | 风格与上下文适配 |
|---|---|---|---|
| 沟通与建议 | 待完整数据分类与三任务评分 | 待完整数据分类与三任务评分 | 待完整数据分类与三任务评分 |
| 知识与技术解释 | 待完整数据分类与三任务评分 | 待完整数据分类与三任务评分 | 待完整数据分类与三任务评分 |
| 写作与修订 | 待完整数据分类与三任务评分 | 待完整数据分类与三任务评分 | 待完整数据分类与三任务评分 |

完整官方文件约有 99,952 条原始记录。原始每条记录通常只有一个 custom rubric
及一个 GPT-4 分数，因此不能通过 rubric 分类直接得到每回答三任务。正式 prepare
得到选中三个 domain 的唯一回答数 `N` 后，目标必须是 `A=N, B=3N`。每个回答
缺少的任务分数都要由同一冻结版本的强 Teacher 重新评分；人工测试集另行分层抽取。

## 6. 后续执行顺序

1. 先修改 LongJudgeBench 适配器，直接利用原生多维标签生成 DeepResearch-Bench
   `4 x 4` 和 RealDR `3 x 3`；这是唯一不需要补评分即可完成 dense B 的数据。
2. 为 BiGGen-Bench 冻结三套跨 domain 通用 rubric，决定采用 Human-GT 还是
   Teacher-Label 补齐 2,160 个分数，然后生成 3,240 条 B。
3. 为 FLASK 补齐 2,281 个唯一回答-任务 Teacher 分数并去重，生成 5,073 条 B。
4. 接入 RuVerBench Agentic Coding，审计两个分布的每回答四任务覆盖，再决定分别
   报告两个 `1 x 4`，还是补成共享 `2 x 4`。
5. Prometheus 最后处理；先冻结抽样规模和三套 rubric，避免直接对约十万回答进行
   三任务评分造成不受控成本。

在这些步骤完成前，当前可用 hidden state 只有 BiGGen-Bench 和 FLASK 的旧稀疏
版本；没有任何一个五数据集实验已经通过本文定义的完整 B-space 验收。
