# FLASK 完整 Domain–Task 划分表

## 1. 一句话结论

FLASK 可以构造成 10 个官方 Domain × 12 个官方 Skill 的评测视图。每条候选回答只对原始 `metrics` 中选中的 Skills 有 GPT-4 分数；多 Domain 样本可以同时归入多个 `Domain × Skill` cell，分数沿用同一条 `response × skill` Ground Truth。

## 2. 核心概念

| 概念 | 含义 |
|---|---|
| Domain | prompt 的领域标签，来自 `domain_labeled` |
| Task / Skill | 当前评分维度，来自 `metrics[i]` |
| A-space | 一条候选回答 `candidate_response` |
| B-space | 一条 `response × skill` 评分行 |
| Cell membership | 一条 B 行在某个 `Domain × Skill` 子集中的归属 |
| Ground Truth | GPT-4 对该 `response × skill` 的分数 |
| 切分 group key | `question_id`，同题的 15 个模型回答及所有 Skill 行一起切分 |

## 3. 原始规模与清洗口径

| 项目 | 数量 / 规则 |
|---|---:|
| prompts | 1,700 |
| candidate models | 15 |
| A-space responses | 25,500 |
| 原始 score slots | 76,499 |
| 合法数值分数 | 75,977 |
| `N/A` | 476 |
| 空值 | 28 |
| 1–5 范围外分数 | 18 |
| 清洗后的唯一 B-space | 75,977 |

清洗规则：保留 1 到 5 之间的数值分数，包括小数；删除 `N/A`、空值和范围外分数。未出现在 `metrics` 中的 Skill 没有 GT，不生成 B 行。

## 4. Domain 与 Skill 分数的关系

FLASK 的 `score` 定义在 `response × skill` 层级。Domain 描述样本领域归属；同一条 `response × skill` B 行进入多个 Domain membership 时，Ground Truth 保持一致。

真实双 Domain 样例，来自官方 FLASK `gpt4_review.jsonl` 第 19 条记录：

```json
"question_id": 19,
"domain_labeled": ["Culture", "Technology"],
"metrics": ["Commonsense Understanding", "Comprehension", "Conciseness"],
"score": {
  "commonsense understanding": 5,
  "comprehension": 5,
  "conciseness": 4
}
```

这条候选回答先形成 3 条唯一 B 行：

| Skill | Ground Truth | domain_ids |
|---|---:|---|
| Commonsense Understanding | 5 | Culture, Technology |
| Comprehension | 5 | Culture, Technology |
| Conciseness | 4 | Culture, Technology |

展开为 `Domain × Skill` memberships 后，用于 6 个 cell：

| Domain | Skill | Ground Truth |
|---|---|---:|
| Culture | Commonsense Understanding | 5 |
| Technology | Commonsense Understanding | 5 |
| Culture | Comprehension | 5 |
| Technology | Comprehension | 5 |
| Culture | Conciseness | 4 |
| Technology | Conciseness | 4 |

## 5. 两种数据视图

| 视图 | 使用范围 | Prompts | A-space | 清洗后 B-space / memberships |
|---|---|---:|---:|---:|
| 全量视图 | Direct Judge、分类头、完整 10×12 cell 统计 | 1,700 | 25,500 | 75,977 条唯一 B；103,812 个 memberships |
| 单一 Domain 子集 | 严格 Domain-OOD、严格 Joint-OOD、需要互斥领域的实验 | 1,077 | 16,155 | 48,142 条 B / memberships |

单一 Domain 子集保留 `domain_labeled` 长度为 1 的 prompts。它的用途是让每条样本只归属一个 Domain，从而在 Domain-OOD 中把训练 Domain 和测试 Domain 分开。多 Domain 样本继续用于全量视图；严格互斥的 Domain-OOD 视图使用单一 Domain 子集。

例子：`["Culture", "Technology"]` 的样本可进入全量 `Culture × Skill` 和 `Technology × Skill` cell 统计。若实验把 Technology 作为 held-out Domain，单一 Domain 子集会移除这类跨 Domain 样本，使 held-out 测试更干净。

## 6. 官方 Domains

| Domain | Domain | Domain | Domain | Domain |
|---|---|---|---|---|
| Humanities | Language | Social Science | History | Culture |
| Technology | Coding | Math | Natural Science | Health |

## 7. 官方 Tasks / Skills

| Skill | Skill | Skill |
|---|---|---|
| Comprehension | Factuality | Logical Correctness |
| Commonsense Understanding | Completeness | Insightfulness |
| Metacognition | Readability | Conciseness |
| Harmlessness | Logical Robustness | Logical Efficiency |

## 8. 全量 10 × 12 Domain–Skill 评分行

每个数字表示该 cell 中清洗后的 `response × skill` membership 数。多 Domain 样本会同时计入它的所有 Domain。

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
| Humanities | 4,139 | 1,158 | 825 | 2,700 | 1,800 | 1,139 | 740 | 1,439 | 825 | 1,004 | 330 | 75 | **16,174** |
| Language | 2,204 | 925 | 705 | 1,986 | 810 | 1,064 | 234 | 1,110 | 1,035 | 195 | 90 | 60 | **10,418** |
| Social Science | 3,388 | 1,752 | 644 | 1,313 | 1,065 | 643 | 564 | 1,049 | 838 | 599 | 90 | 45 | **11,990** |
| History | 990 | 799 | 225 | 503 | 449 | 283 | 90 | 360 | 210 | 30 | 15 | 15 | **3,969** |
| Culture | 4,258 | 2,301 | 780 | 2,553 | 1,380 | 1,644 | 442 | 1,874 | 1,317 | 420 | 254 | 195 | **17,418** |
| Technology | 3,163 | 1,453 | 479 | 1,535 | 1,423 | 1,018 | 246 | 1,364 | 733 | 253 | 210 | 210 | **12,087** |
| Coding | 1,513 | 750 | 1,168 | 761 | 645 | 314 | 73 | 660 | 420 | 45 | 1,229 | 1,590 | **9,168** |
| Math | 1,215 | 469 | 2,655 | 1,587 | 495 | 164 | 162 | 510 | 540 | 15 | 1,379 | 627 | **9,818** |
| Natural Science | 1,799 | 1,382 | 808 | 973 | 900 | 285 | 134 | 480 | 300 | 90 | 135 | 75 | **7,361** |
| Health | 1,469 | 689 | 210 | 704 | 645 | 330 | 164 | 389 | 389 | 210 | 90 | 120 | **5,409** |
| **Skill 合计** | **24,138** | **11,678** | **8,499** | **14,615** | **9,612** | **6,884** | **2,849** | **9,235** | **6,607** | **2,861** | **3,822** | **3,012** | **103,812** |

## 9. 单一 Domain 子集的 10 × 12 评分行

该视图用于严格互斥的 Domain-OOD / Joint-OOD。保留单一 Domain prompt 后，每条 B 行只有一个 Domain membership。当前有 118 个非空 cell，两个空 cell 是 `Humanities × Logical Efficiency` 和 `History × Logical Robustness`。

| Domain | Cmp | Fact | LC | CU | Comp | Ins | Meta | Read | Conc | Safe | LR | LE | 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Humanities | 1,484 | 365 | 210 | 937 | 735 | 360 | 176 | 585 | 285 | 419 | 75 | 0 | **5,631** |
| Language | 1,140 | 570 | 465 | 1,015 | 375 | 540 | 117 | 630 | 630 | 90 | 30 | 30 | **5,632** |
| Social Science | 1,228 | 809 | 329 | 373 | 315 | 104 | 194 | 419 | 344 | 134 | 30 | 15 | **4,294** |
| History | 255 | 296 | 135 | 89 | 120 | 30 | 15 | 90 | 60 | 15 | 0 | 15 | **1,120** |
| Culture | 1,978 | 1,341 | 570 | 1,272 | 600 | 476 | 204 | 795 | 582 | 105 | 104 | 90 | **8,117** |
| Technology | 974 | 521 | 150 | 495 | 374 | 209 | 72 | 464 | 224 | 58 | 90 | 60 | **3,691** |
| Coding | 960 | 482 | 959 | 435 | 375 | 90 | 30 | 465 | 270 | 15 | 1,140 | 1,457 | **6,678** |
| Math | 780 | 277 | 2,070 | 1,122 | 315 | 60 | 117 | 360 | 390 | 15 | 960 | 434 | **6,900** |
| Natural Science | 974 | 914 | 538 | 463 | 510 | 45 | 60 | 225 | 180 | 30 | 90 | 30 | **4,059** |
| Health | 509 | 343 | 75 | 240 | 285 | 90 | 30 | 164 | 164 | 60 | 15 | 45 | **2,020** |
| **Skill 合计** | **10,282** | **5,918** | **5,501** | **6,441** | **4,004** | **2,004** | **1,015** | **4,197** | **3,129** | **941** | **2,534** | **2,176** | **48,142** |

## 10. 字段映射

| 原始字段 | 标准字段 |
|---|---|
| `question_id` | `base_id` / split group key |
| `text` | `instruction` |
| `answer` | `reference_answer` |
| `target_txt` | `candidate_response` |
| review 文件名 | `generator_id` |
| `domain_labeled` | `domain_ids` |
| `metrics[i]` | `task_id` |
| 当前 Skill 的说明 | `rubric` / `score_guide` |
| `metric_explanation` | Skill 选择说明 |
| `score[metric.lower()]` | `ground_truth` |

## 11. 推荐实验用法

| 实验 | 使用视图 | 切分要求 |
|---|---|---|
| Direct Judge | 全量视图 | 全部有效 B 行直接推理；提示词格式如需选择，按 `question_id` 划分验证集 |
| 分类头 / 回归头 | 全量视图 | 按 `question_id` 切分 |
| 10×12 cell 分析 | 全量视图 | 多 Domain 样本展开 memberships |
| 严格 Domain-OOD | 单一 Domain 子集 | held-out Domain 只出现在测试侧 |
| Task-OOD | 全量视图或单一 Domain 子集 | held-out Skill 只出现在测试侧 |
| Joint-OOD | 单一 Domain 子集 | held-out Domain 和 held-out Skill 同时留出 |

## 12. Qwen3.5 Direct Judge 实验

### 12.1 目标

Direct Judge 评测 Qwen3.5 对 FLASK Skill 分数的直接预测能力。模型输出 1–5
分，并与该 B 行的 GPT-4 `ground_truth` 比较。该实验独立报告 Judge 评分质量。

### 12.2 输入与标签

| 用途 | 字段 |
|---|---|
| Qwen 输入 | 当前 Skill 的 rubric / score guide、原始 instruction、reference answer（有则提供）、candidate response |
| 评测标签 | 当前 `response × skill` 的 `ground_truth` |
| 分组报告 | `domain_ids`、`task_id`、`generator_id` |

`ground_truth` 和原始 GPT-4 `review` 在 Qwen 推理时不提供。输出格式固定为一个
可解析的 1–5 分数。

### 12.3 统计单位

全局结果以唯一 B 行 `response_id + task_id` 为单位，每条评分行只计算一次。
按 `Domain × Task` 汇总时，该 B 行会出现在它所属的每个 Domain cell 中；这些
cell 指标用于观察不同领域和 Skill 上的评分表现。

### 12.4 结果指标

| 指标 | 含义 |
|---|---|
| MAE | 预测分数与 GPT-4 分数的平均绝对误差 |
| Exact Accuracy | 预测分数与标签完全相同的比例 |
| ±1 Accuracy | 预测分数与标签相差至多 1 分的比例 |
| Quadratic Weighted Kappa | 有序 1–5 分数的一致性 |
