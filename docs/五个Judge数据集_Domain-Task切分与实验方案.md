# 五个 Judge 数据集最终 Domain–Task 划分表

## 1. 最终总表

| 数据集 | 最终 Domain | 最终 Task | Ground Truth | A-space 数量 | B-space / 评分行数量 | 支持的 OOD |
|---|---|---|---|---:|---:|---|
| LongJudge–DeepResearch | 固定为 `longjudge_deepresearch` | 4 个官方评分维度 | 三名人工评分的维度均值，0–100 | 200 | 800 | Task-OOD |
| LongJudge–RealDR | 固定为 `longjudge_realdr` | 3 个官方评分维度 | 1–2 名人工评分的维度均值，0–10 | 640 | 1,920 | Task-OOD |
| RuVerBench | `deepresearch`、`agentic_coding` | 统一为 `rubric_verification` | 人工二分类，0/1 | 494 | 2,458 | Domain-OOD |
| FLASK | 10 个官方 Domains | 12 个官方 Skills | GPT-4 的 1–5 分 | 25,500 | 76,004 条唯一有效 B；展开后属于 103,850 个 Domain–Skill cells | Domain/Task/Joint OOD |
| BiGGen-Bench | 9 个官方 Capabilities | Capability 下的 77 个官方 Tasks | `human_score`，1–5 分 | 3,196 | 3,196 | Capability-OOD；域内 Task-OOD |
| Prometheus | 固定为 `prometheus` | 1,000 个 Custom Rubrics | GPT-4 的 `orig_score`，1–5 分 | 99,952 | 99,952 | Rubric/Task-OOD |

## 2. A-space 与 B-space 的计数规则

| 空间 | 一条数据是什么 | 唯一键 |
|---|---|---|
| A-space | 只输入一条 `candidate_response` | `response_id` |
| B-space | 输入 instruction、reference、rubric、score guide 和 candidate response | `response_id + task_id` |

| 情况 | 最终处理 |
|---|---|
| 同一 response 有 4 个维度分数 | 生成 1 条 A、4 条 B |
| 同一 response 有 3 个 skill 分数 | 生成 1 条 A、3 条 B |
| 同一 response 只有自己的一个 rubric 分数 | 生成 1 条 A、1 条 B |
| 某个 response–task 没有 GT | 不创建该 B 行 |

## 3. LongJudgeBench

### 3.1 DeepResearch

| 项目 | 最终划分 |
|---|---|
| 原始结构 | 50 个 queries × 每题 4 个模型 responses = 200 个 responses |
| Domain | 全部固定为 `longjudge_deepresearch` |
| Generator | `openai-deepresearch`、`gemini-2.5-pro-deepresearch`、`grok-deeper-search`、`perplexity-Research` |
| Generator 是否是 Domain | 否，只保存为 `generator_id` |
| Task 数 | 4 |
| 每条 response 是否有全部 Task GT | 是 |
| 最终数量 | 200 条 A；200 × 4 = 800 条 B |

| `task_id` | 原始 GT 字段 | GT |
|---|---|---|
| `comprehensiveness` | `ground_truth.scores[model].dimensions.comprehensiveness` | 三名人工均分，0–100 |
| `insight` | `ground_truth.scores[model].dimensions.insight` | 三名人工均分，0–100 |
| `instruction_following` | `ground_truth.scores[model].dimensions.instruction_following` | 三名人工均分，0–100 |
| `readability` | `ground_truth.scores[model].dimensions.readability` | 三名人工均分，0–100 |

| 原始字段 | 标准字段 |
|---|---|
| `id` | `base_id=query_id` |
| `instruction` | `instruction` |
| `responses[].model` | `generator_id` |
| `responses[].content` | `candidate_response` |
| `dimensions` 中当前维度名 | `task_id` |
| 当前模型、当前维度的分数 | `ground_truth` |

### 3.2 RealDR

| 项目 | 最终划分 |
|---|---|
| 原始结构 | 640 个 records，每条只有 1 个 response |
| Domain | 全部固定为 `longjudge_realdr` |
| Task 数 | 3 |
| 每条 response 是否有全部 Task GT | 是 |
| 最终数量 | 640 条 A；640 × 3 = 1,920 条 B |

| `task_id` | 原始维度 | GT |
|---|---|---|
| `logical_structure` | `逻辑结构` | 人工分数，0–10 |
| `presentation_form` | `表达形式` | 人工分数，0–10 |
| `bias_checking` | `偏见检查` | 人工分数，0–10 |

| 原始字段 | 标准字段 |
|---|---|
| `id` | `base_id` |
| `instruction` | `instruction` |
| `responses[0].model` | `generator_id` |
| `responses[0].content` | `candidate_response` |
| `ground_truth.dimensions` 中当前维度名 | `task_id` |
| `ground_truth.dimensions[当前维度]` | `ground_truth` |

## 4. RuVerBench

| Domain | Cases/A-space | Rubric checks/B-space | Task | Label |
|---|---:|---:|---|---|
| `deepresearch` | 284 | 1,615 | `rubric_verification` | `covered=true/false` → 1/0 |
| `agentic_coding` | 210 | 843 | `rubric_verification` | `result=success/fail` → 1/0 |
| **合计** | **494** | **2,458** | **同一个二分类任务** | **人工 GT** |

| Domain | Candidate response | Rubric/check | GT 字段 |
|---|---|---|---|
| DeepResearch | `deepresearch_responses[].response` | `deepresearch_dataset[].rubric[].point` | `result.coverage_results[].covered` |
| AgenticCoding | 对应 case 的 `messages`/trajectory | `results[].*.checks[].description` | `results[].*.checks[].result` |

| 最终规则 | 处理方式 |
|---|---|
| rubric 文本不同 | 作为 B-space 的当前评分标准 |
| rubric 文本是否分别成为 Task | 否，Task 统一为二分类 `rubric_verification` |
| 同一 case 有多个 checks | 共享 1 条 A，每个 check 各生成 1 条 B |
| 切分单位 | `case_id/instance_id`，同一 case 的全部 checks 必须在同一 split |

## 5. FLASK

### 5.1 官方 Domains

| Domain ID | Domain ID | Domain ID | Domain ID | Domain ID |
|---|---|---|---|---|
| Humanities | Language | Social Science | History | Culture |
| Technology | Coding | Math | Natural Science | Health |

### 5.2 官方 Tasks/Skills

| `task_id` | `task_id` | `task_id` |
|---|---|---|
| Comprehension | Factuality | Logical Correctness |
| Commonsense Understanding | Completeness | Insightfulness |
| Metacognition | Readability | Conciseness |
| Harmlessness | Logical Robustness | Logical Efficiency |

### 5.3 最终展开

| 项目 | 数量/处理 |
|---|---|
| 已评分 prompts | 1,700 |
| 每个 prompt 的 candidate models | 15 |
| A-space | 1,700 × 15 = 25,500 条 responses |
| 每条 response 的 Tasks | 只使用 `metrics` 中实际选择的 3 个 Skills |
| 原始 score slots | 25,500 × 3 = 76,500 |
| `N/A` score | 478 |
| 超出官方 1–5 范围的 score | 18，包括 17 个 `0` 和 1 个 `-1` |
| 最终唯一有效 B-space | 76,500 − 478 − 18 = 76,004 |
| 可能的 Domain–Skill cells | 10 × 12 = 120 |
| 全量数据实际有数据的 cells | 120/120 |
| 展开 `domain_labeled` 后的 cell 成员行 | 103,850；多 Domain 数据会在多个 Domain 下重复归属 |

| 一条原始数据 | 唯一 A | 唯一 B | Domain–Skill cell 成员数 |
|---|---:|---:|---:|
| 1 个 response、1 个 Domain、3 个 Skills | 1 | 3 | 3 |
| 1 个 response、2 个 Domains、3 个 Skills | 1 | 3 | 6 |
| 1 个 prompt、15 个 responses、1 个 Domain、3 个 Skills | 15 | 45 | 45 |

### 5.4 全量 10 × 12 Domain–Skill 评分行

下表每个数字都是该 cell 中合法的 1–5 分 `response–skill` 评分数量。多 Domain response 会同时计入其所有 Domain。

| 缩写 | Skill | 缩写 | Skill |
|---|---|---|---|
| Cmp | Comprehension | Fact | Factuality |
| LC | Logical Correctness | CU | Commonsense Understanding |
| Comp | Completeness | Ins | Insightfulness |
| Meta | Metacognition | Read | Readability |
| Conc | Conciseness | Safe | Harmlessness |
| LR | Logical Robustness | LE | Logical Efficiency |

| Domain | Cmp | Fact | LC | CU | Comp | Ins | Meta | Read | Conc | Safe | LR | LE | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Humanities | 4,139 | 1,158 | 825 | 2,706 | 1,800 | 1,139 | 740 | 1,439 | 825 | 1,005 | 330 | 75 | **16,181** |
| Language | 2,204 | 925 | 705 | 1,989 | 811 | 1,064 | 234 | 1,110 | 1,035 | 195 | 90 | 60 | **10,422** |
| Social Science | 3,388 | 1,752 | 644 | 1,314 | 1,065 | 643 | 564 | 1,049 | 838 | 599 | 90 | 45 | **11,991** |
| History | 990 | 800 | 225 | 507 | 450 | 283 | 90 | 360 | 210 | 30 | 15 | 15 | **3,975** |
| Culture | 4,259 | 2,301 | 780 | 2,555 | 1,381 | 1,644 | 442 | 1,874 | 1,317 | 420 | 254 | 195 | **17,422** |
| Technology | 3,164 | 1,454 | 479 | 1,540 | 1,425 | 1,018 | 246 | 1,365 | 733 | 253 | 210 | 210 | **12,097** |
| Coding | 1,513 | 750 | 1,168 | 763 | 646 | 314 | 73 | 660 | 420 | 45 | 1,229 | 1,590 | **9,171** |
| Math | 1,215 | 469 | 2,655 | 1,587 | 495 | 164 | 162 | 510 | 540 | 15 | 1,379 | 627 | **9,818** |
| Natural Science | 1,799 | 1,382 | 808 | 973 | 900 | 285 | 134 | 480 | 300 | 90 | 135 | 75 | **7,361** |
| Health | 1,470 | 689 | 210 | 704 | 645 | 330 | 164 | 390 | 390 | 210 | 90 | 120 | **5,412** |
| **Skill 合计** | **24,141** | **11,680** | **8,499** | **14,638** | **9,618** | **6,884** | **2,849** | **9,237** | **6,608** | **2,862** | **3,822** | **3,012** | **103,850** |

### 5.5 单一 Domain 子集的 10 × 12 评分行

该表删除所有多 Domain prompts，用于互斥的 Domain-OOD。此时共有 48,158 条有效 B，118/120 个非空 cells。

| Domain | Cmp | Fact | LC | CU | Comp | Ins | Meta | Read | Conc | Safe | LR | LE | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Humanities | 1,484 | 365 | 210 | 941 | 735 | 360 | 176 | 585 | 285 | 420 | 75 | **0** | **5,636** |
| Language | 1,140 | 570 | 465 | 1,017 | 375 | 540 | 117 | 630 | 630 | 90 | 30 | 30 | **5,634** |
| Social Science | 1,228 | 809 | 329 | 373 | 315 | 104 | 194 | 419 | 344 | 134 | 30 | 15 | **4,294** |
| History | 255 | 296 | 135 | 90 | 120 | 30 | 15 | 90 | 60 | 15 | **0** | 15 | **1,121** |
| Culture | 1,979 | 1,341 | 570 | 1,272 | 601 | 476 | 204 | 795 | 582 | 105 | 104 | 90 | **8,119** |
| Technology | 975 | 521 | 150 | 495 | 375 | 209 | 72 | 465 | 224 | 58 | 90 | 60 | **3,694** |
| Coding | 960 | 482 | 959 | 435 | 375 | 90 | 30 | 465 | 270 | 15 | 1,140 | 1,457 | **6,678** |
| Math | 780 | 277 | 2,070 | 1,122 | 315 | 60 | 117 | 360 | 390 | 15 | 960 | 434 | **6,900** |
| Natural Science | 974 | 914 | 538 | 463 | 510 | 45 | 60 | 225 | 180 | 30 | 90 | 30 | **4,059** |
| Health | 510 | 343 | 75 | 240 | 285 | 90 | 30 | 165 | 165 | 60 | 15 | 45 | **2,023** |
| **Skill 合计** | **10,285** | **5,918** | **5,501** | **6,448** | **4,006** | **2,004** | **1,015** | **4,199** | **3,130** | **942** | **2,534** | **2,176** | **48,158** |

两个空 cell 是 `Humanities × Logical Efficiency` 和 `History × Logical Robustness`。

### 5.6 字段映射与使用范围

| 原始字段 | 标准字段 |
|---|---|
| `question_id` | `base_id` |
| `text` | `instruction` |
| `answer` | `reference_answer` |
| `target_txt` | `candidate_response` |
| review 文件名 | `generator_id` |
| `domain_labeled` | `domain_ids` |
| `metrics[i]` | `task_id` |
| `metrics[i]` 对应的官方 skill description | `rubric/score_guide` |
| `metric_explanation` | 仅保留为 Skill 选择说明，不作为评分 GT |
| `score[当前 metric]` | `ground_truth` |

| FLASK 子集 | Prompts | A-space | 有效 B-space | 用途 |
|---|---:|---:|---:|---|
| 全部数据 | 1,700 | 25,500 | 76,004 条唯一 B；103,850 个 cell 成员 | Direct Judge、分类头、全量 cell 汇总 |
| 单一 Domain 数据 | 1,077 | 16,155 | 48,158 | 互斥 Domain/Task/Joint OOD |

| 多 Domain 规则 | 最终处理 |
|---|---|
| 基础评分实验 | 保留 `domain_labeled` 中全部官方 Domains |
| Domain-OOD 实验 | 只使用恰好有一个官方 Domain 的 prompts |
| 多 Domain 的相同 B hidden state | 只提取一次；汇总 cell 时分配到其所有 Domains |
| 未被选中的 Skill | 没有 GT，不创建该 response–skill 行 |
| train/validation/test | 先按 `question_id` 切分，再展开 Skills 和 Domain memberships |

## 6. BiGGen-Bench

### 6.1 官方 Capability–Task 划分

| `domain_id=capability` | 官方 Task 数 | 原始评分行 | 删除 `human_score=-1` | 最终有效行 |
|---|---:|---:|---:|---:|
| `grounding` | 10 | 400 | 3 | 397 |
| `instruction_following` | 10 | 400 | 0 | 400 |
| `multilingual` | 7 | 420 | 0 | 420 |
| `planning` | 7 | 280 | 0 | 280 |
| `reasoning` | 10 | 400 | 0 | 400 |
| `refinement` | 8 | 304 | 0 | 304 |
| `safety` | 8 | 316 | 0 | 316 |
| `theory_of_mind` | 10 | 400 | 0 | 400 |
| `tool_usage` | 7 | 280 | 1 | 279 |
| **合计** | **77** | **3,200** | **4** | **3,196** |

| 原始字段 | 标准字段 |
|---|---|
| `query` | `instruction`、`base_id` 的组成部分 |
| `response` | `candidate_response` |
| `reference_answer` | `reference_answer` |
| `rubric` | `score_guide` |
| `natural_unit_test` | 当前 instance 的 `rubric/evaluation criterion` |
| `capability` | `domain_id` |
| `task` | `task_id` |
| `human_score` / `label` | 主要 `ground_truth` |
| `gpt4_score` | 辅助 teacher score |

| 最终规则 | 处理方式 |
|---|---|
| `human_score` 与 `gpt4_score` | 保留在同一条 judgment row 中，不拆成两个 Tasks |
| 分类头 GT | 使用 `human_score`；`label` 与 `human_score` 相同 |
| `gpt4_score` | 只作为辅助字段，不与 human score 求平均 |
| `human_score=-1` | 删除；对应空 response 或生成失败 |
| A/B 数量 | 每条 response 只有一个 instance-specific rubric，因此 A=3,196、B=3,196 |
| 切分单位 | 相同 `query/instance` 的全部 candidate responses 必须在同一 split |

## 7. Prometheus Feedback Collection

| 项目 | 最终划分 |
|---|---|
| Domain | 全部固定为 `prometheus` |
| Task | 每套完整 Custom Rubric 作为一个 `task_id` |
| 官方 Rubrics | 1,000 套 |
| Instructions/reference answers | 20,000 |
| 最终评分行 | 99,952 |
| A-space | 99,952 |
| B-space | 99,952 |
| GT | GPT-4 `orig_score`，1–5 |

| 原始字段 | 标准字段 |
|---|---|
| `orig_instruction` | `instruction` |
| `orig_response` | `candidate_response` |
| `orig_reference_answer` | `reference_answer` |
| `orig_criteria` | rubric 的评价标准 |
| `orig_score1_description` 至 `orig_score5_description` | `score_guide` |
| 完整 rubric 内容的 hash | `task_id=rubric_id` |
| `orig_score` | `ground_truth` |

| 最终规则 | 处理方式 |
|---|---|
| 一个 rubric 的组成 | `orig_criteria + 五档 score descriptions` |
| 同一 rubric 下的多条 responses | Domain 相同、Task 相同，response_id 不同 |
| 一条 response 是否扩展到其他 rubrics | 否，只保留原始有 GT 的 rubric |
| OOD 切分 | 按 `rubric_id` 整体划分 seen/unseen；同一 rubric 不跨两侧 |

## 8. 最终统一输出字段

| 字段 | 内容 |
|---|---|
| `dataset_id` | 数据集名称 |
| `base_id` | 原始 query/question/case/instance ID |
| `response_id` | 当前 candidate response 的唯一 ID |
| `domain_id` / `domain_ids` | 官方 Domain；没有官方 Domain 时使用固定数据集 Domain |
| `task_id` | 官方 dimension、skill、task 或 rubric ID |
| `instruction` | 原始问题/指令 |
| `candidate_response` | 被评分回答 |
| `reference_answer` | 数据集有则保留 |
| `rubric` | 当前评分标准 |
| `score_guide` | 分档说明；数据集有则保留 |
| `ground_truth` | 当前 response–task 的真实分数 |
| `label_source` | `human` 或 `gpt4_teacher` |
| `generator_id` | response 的生成模型；只作 metadata |

## 9. 最终分组键

| 数据集 | 所有派生行必须共同切分的 group key |
|---|---|
| LongJudge–DeepResearch | `query_id` |
| LongJudge–RealDR | `record_id` |
| RuVerBench | `case_id/instance_id` |
| FLASK | `question_id` |
| BiGGen-Bench | `query/instance_id` |
| Prometheus | `rubric_id + orig_instruction` |

## 10. 官方来源

| 数据集 | 官方来源 |
|---|---|
| LongJudgeBench | https://github.com/cjj826/LongJudgeBench |
| RuVerBench | https://github.com/THU-KEG/RuVerBench |
| FLASK | https://github.com/kaistAI/FLASK |
| BiGGen-Bench | https://huggingface.co/datasets/ContextualAI/BiGGenBench |
| Prometheus | https://huggingface.co/datasets/prometheus-eval/Feedback-Collection |
