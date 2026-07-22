# LLM Judge 的 OOD 检测与受控更新：任务定义、数据构造和实验表格

# 目录与章节说明

本文按照“定义任务 → 构造数据 → 确定真值 → 设计评测 → 比较方法 → 形成论文表格”的顺序组织。

- [1. 什么是 OOD](#1-什么是-ood)：说明 OOD 的基本概念、ID 与 OOD 的区别，以及 background shift、semantic shift、Near/Far 和 Harmful/Benign 的正确关系。
- [2. 本文的正式任务定义](#2-本文的正式任务定义)：将研究拆分为样本级 OOD 检测、窗口级漂移检测、性能影响确认和受控更新四个独立任务。
- [3. 数据集选择](#3-数据集选择)：说明四个核心数据集和一个可选补充数据集分别用于什么实验。
- [4. 数据集字段、样本和 OOD 构造](#4-数据集字段样本和-ood-构造)：逐个介绍 ELLIPSE、ASAP-AES、CLINC150、ROSTD 和 AG News 的原始字段、样本形式、ID/OOD 构造和适用实验。
- [5. Ground Truth 到底有哪些](#5-ground-truth-到底有哪些)：区分 OOD ground truth、任务标签 ground truth、窗口漂移 ground truth 和更新效果 ground truth。
- [6. 统一测试流程](#6-统一测试流程)：说明如何隔离数据，并保证不同 baseline 公平比较。
- [7. Baseline、指标及其实际含义](#7-baseline指标及其实际含义)：通俗解释每组 baseline 比较什么、指标是什么意思，以及如何读结果。
- [8. 论文最终表格及读表方式](#8-论文最终表格及读表方式)：给出论文最终六张主表的结构和读表规则。
- [9. 现有结果应如何放入新结构](#9-现有结果应如何放入新结构)：说明已有实验结果分别进入正文表格还是附录。
- [10. 最终任务陈述](#10-最终任务陈述)：用一段话概括完整研究任务和五个核心问题。
- [参考文献](#参考文献)：列出数据集、OOD 检测、漂移检验和更新方法的主要来源。

---

**版本：** 精简标准版  
**目标：** 明确本文究竟研究什么 OOD 任务、数据如何产生 OOD ground truth、不同 baseline 比较什么，以及论文最终需要哪些结果表。

---

# 1. 什么是 OOD

OOD（out-of-distribution）指的是：**部署时出现的数据，与模型训练和校准阶段见过的数据来源不一致。**

例如，一个作文评分模型只在若干固定写作题目上训练。部署后出现一个训练阶段从未见过的新题目，这些作文就可以被定义为 OOD。这个定义来自题目来源，而不是检测器分数。

本文区分两种常见变化：

1. **背景变化（background shift）**  
   任务没有改变，但写作题目、学生群体、表达风格、文本长度或语料来源发生变化。

2. **语义变化（semantic shift）**  
   输入出现训练阶段没有覆盖的新类别或新意图。例如，CLINC150 中原系统只支持一组固定意图，而用户提出了系统范围之外的问题。

Near OOD 和 Far OOD 只表示 OOD 与训练数据的接近程度：

- Near OOD：和训练数据较相似，通常更难检测；
- Far OOD：差异更明显，通常更容易检测。

Near/Far 没有跨数据集统一的界线，因此必须在实验前根据 prompt、domain、class 或 corpus 来源确定，不能看完检测结果后再划分。

Harmful、Benign 和 Uncertain 描述的是**分布变化对模型性能的影响**：

- **Harmful**：人工标签证明模型性能明显下降；
- **Benign**：人工标签证明性能下降仍在允许范围内；
- **Uncertain**：现有标签不足以确定是否有害。

因此，Near/Far 是 OOD 难度，Harmful/Benign 是性能后果，两者不是一套四分类。

---

# 2. 本文的正式任务定义

本文包含三个 OOD 评测任务，以及一个检测后的更新任务。

## 2.1 Task A：样本级 OOD 检测

输入是一篇新文档，检测器输出一个异常分数。分数越高，表示该文档越不像训练数据。

离线测试时，每篇文档都有明确真值：

- 来自预先规定的训练数据源：ID；
- 来自预先规定的未见数据源：OOD。

评价单位是**单篇文档**。主要比较 AUROC、AUPR-OOD 和 FPR95。

这一任务回答：

> 检测器能否把 OOD 文档排在 ID 文档前面？

## 2.2 Task B：窗口级分布漂移检测

系统连续接收文档。Task B 不再判断某一篇文档，而是比较：

- 一批历史正常文档；
- 最近到达的一批文档。

如果两批文档整体差异明显，系统报告窗口漂移。

评价单位是**一个窗口或到达批次**。主要比较：

- 没有漂移时误报多少；
- 有漂移时能检出多少；
- 少量 OOD 混入时是否仍能检测；
- 需要多少文档；
- 多久才能确认；
- 检测需要多少时间。

这一任务回答：

> 最近一批输入是否已经整体偏离源分布？

## 2.3 Task C：性能影响确认

检测到 OOD 或窗口漂移，只能说明输入发生变化，不能直接说明 Judge 已经失效。

系统需要从报警数据中抽取少量文档，获得人工标签，再比较目标域和源域的平均误差。

正文只保留一条必要关系：

```text
风险变化 = 目标域平均误差 - 源域平均误差
```

- 风险明显上升：Harmful；
- 风险被证明没有超过允许范围：Benign；
- 证据不足：Uncertain。

Task C 的真实结论来自完整人工标签，而不是 ViM 分数、MMD p 值或模型置信度。

## 2.4 Task D：受控更新与安全上线

只有当系统确认漂移具有实际性能影响时，才训练候选更新模型。

所有更新方法使用相同的目标标签预算。候选模型先通过独立 Gate 数据检查，再使用完全独立的 Future 数据评价最终效果。

主要检查：

- 目标域性能是否改善；
- 源域性能是否下降；
- 更新是否制造新的错误；
- Gate 是否错误放行坏更新；
- 使用了多少标签、参数、时间和显存。

---

# 3. 数据集选择

为了同时覆盖 LLM Judge 生命周期和标准文本 OOD benchmark，正文优先使用四个核心数据集；AG News 作为可选补充 benchmark，不属于论文必须完成的数据集。不同数据集承担不同任务，不能全部放入同一类更新实验。

| 数据集 | 主要用途 | OOD ground truth 类型 | 支持的任务 |
|---|---|---|---|
| **ELLIPSE** | 完整作文 Judge 生命周期主数据集 | 完整留出 prompt，真值来自 `prompt` membership | Task A–D |
| **ASAP-AES** | 第二个作文评分与跨 prompt compound OOD | 完整留出 essay set，真值来自 `essay_set` membership | Task A–D 复现 |
| **CLINC150** | 官方 OOS 的标准 semantic OOD benchmark | 官方 in-scope/OOS split | Task A–B |
| **ROSTD** | 人工编写的真实 task-oriented-dialog OOD | 官方 ID/OOD 来源 | Task A–B |
| **AG News（可选补充）** | 类别留出的 controlled semantic OOD | 官方主题标签与预注册 held-out class | Task A–B；不进入最低必需实验集 |

五个数据集分成两组：

1. **完整生命周期组：ELLIPSE、ASAP-AES。**  
   两者具有人工评分标签，可以继续评价 Judge、Probe、Adapt 和 Gate。

2. **标准 OOD 检测组：CLINC150、ROSTD。**  
   两者用于比较 Residual-only ViM、Mahalanobis、RMD、kNN、MSP、Energy、MaxLogit、Full ViM 以及窗口漂移方法。它们不用于证明作文 Judge 的更新有效。

3. **可选补充：AG News。**  
   AG News 用于增加一个类别留出的 controlled semantic OOD 场景，主要检查检测器是否依赖某个特定 OOS 数据集。若时间或算力有限，完成 CLINC150 和 ROSTD 后即可形成标准文本 OOD 证据，AG News 可放在补充实验或后续扩展中。

其中 CLINC150 和 ROSTD 提供自然或官方 OOS 数据，应作为标准文本 OOD 的核心实验。AG News 没有官方 OOS split，需要采用预注册的 leave-one-topic-out 构造，因此证据强度低于 CLINC150 和 ROSTD，定位为可选 controlled semantic OOD 补充。

SummEval 不进入主 OOD 实验，因为随机 held-out article 仍属于同一 CNN/DailyMail 分布，原数据没有自然 OOD 标记。LLMBar 属于 pairwise evaluator challenge benchmark，与 pointwise 作文评分和标准 OOS detection 不是同一任务，也不进入主表。

### 最低必需数据集与补充数据集

论文最低必需组合为：

```text
ELLIPSE + ASAP-AES + CLINC150 + ROSTD
```

其中：

- ELLIPSE 和 ASAP-AES 支撑 LLM Judge 的完整生命周期；
- CLINC150 和 ROSTD 支撑标准文本 OOD baseline 比较；
- AG News 只用于增加 controlled class-holdout semantic OOD，不影响核心结论是否成立。

因此，AG News 可以在以下情况下省略：

- CLINC150 和 ROSTD 已完成全部样本级和窗口级 baseline；
- 论文篇幅有限；
- 四折训练成本较高；
- 主要贡献集中在 Judge 监控与更新，而非建立新的通用文本 OOD benchmark。


---

# 4. 数据集字段、样本和 OOD 构造

## 4.1 ELLIPSE：完整生命周期主数据集

ELLIPSE 包含约 6,500 篇英语学习者作文，覆盖 29 个独立 prompts。每篇作文由两名训练过的人工评分员评分，提供 Overall 以及六个分析维度。

### 4.1.1 原始字段

```text
text_id_kaggle,
full_text,
gender,
grade,
race_ethnicity,
num_words,
num_words2,
num_words3,
num_sent,
num_para,
num_word_div_para,
MTLD,
TTR,
Type,
Token,
task,
SES,
prompt,
Overall,
Cohesion,
Syntax,
Vocabulary,
Phraseology,
Grammar,
Conventions,
set
```

其中实际模型主输入和标签为：

- 输入：`full_text`，必要时加 `prompt`；
- 主标签：`Overall`；
- 辅助标签：六个 analytic scores；
- OOD provenance：`prompt`、`task`；
- 审计字段：`grade`、`gender`、`race_ethnicity`、`SES`；
- `num_words`、`MTLD` 等统计量只能用于分析，不能作为主 OOD detector 的特殊提示信息。

### 4.1.2 一条原始样本

```json
{
  "text_id_kaggle": "5661280443",
  "full_text": "Imagine if you could prove other people that you are a good problem solver ...",
  "gender": "Male",
  "grade": 8,
  "race_ethnicity": "Hispanic/Latino",
  "num_words": 420,
  "num_sent": 18,
  "num_para": 4,
  "MTLD": 28.6263,
  "TTR": 25.2900,
  "task": "Independent",
  "SES": "Economically disadvantaged",
  "prompt": "Benefits of a problem",
  "Overall": 4.0,
  "Cohesion": 3.5,
  "Syntax": 4.0,
  "Vocabulary": 3.5,
  "Phraseology": 3.5,
  "Grammar": 4.0,
  "Conventions": 4.0,
  "set": "train"
}
```

### 4.1.3 标准 OOD 构造

采用 **prompt-disjoint split**：

1. 按 `prompt` 将 29 个 prompts 分成 source prompts 和 held-out prompts；
2. source prompts 内再按 `text_id_kaggle` 划分 Judge-train、reference、calibration、ID-test 和 source-guard；
3. held-out prompts 完全不参与 Judge、OOD detector、阈值和超参数训练；
4. held-out prompts 再按文档划分 Probe、Adapt-validation、Gate 和 Future。

单篇 OOD 真值直接由作文的 `prompt` 是否属于 source prompts 决定。

Near/Far 只做辅助分层。根据 `task`、grade、prompt 类型和预先计算的文本语义距离，将 held-out prompts 分为 Near 和 Far；主要结果同时报告全部 held-out prompts，避免 Near/Far 人为定义影响总结果。

窗口 OOD ground truth 通过按已知比例混合 source 和 held-out prompt 文档得到，例如 0%、5%、10%、20%、50% 和 100%。

该数据集的优势是：不同 prompts 共享 Overall 和六个分析维度的评分体系，因此比跨语料强行映射标签更适合完成 Probe、更新和 Gate。

## 4.2 ASAP-AES：跨 Prompt Compound OOD

ASAP-AES 包含 8 个 essay sets，每个 set 对应不同 prompt、作文类型和评分量表，共约 12,978 篇人工评分作文。

### 4.2.1 原始完整字段

```text
essay_id,
essay_set,
essay,
rater1_domain1,
rater2_domain1,
rater3_domain1,
domain1_score,
rater1_domain2,
rater2_domain2,
domain2_score,
rater1_trait1-rater1_trait6,
rater2_trait1-rater2_trait6,
rater3_trait1-rater3_trait6
```

共 28 个字段。`domain1_score` 是主要 resolved score；部分 essay sets 还提供第二评分维度和 trait scores。

### 4.2.2 一条原始样本

```json
{
  "essay_id": 21603,
  "essay_set": 8,
  "essay": "Those eyes, it was like I was looking out into a warm Caribbean sunset ...",
  "rater1_domain1": 17,
  "rater2_domain1": 18,
  "rater3_domain1": null,
  "domain1_score": 35,
  "rater1_domain2": null,
  "rater2_domain2": null,
  "domain2_score": null,
  "rater1_trait1": 4,
  "rater2_trait1": 4
}
```

未展示的 trait 字段保留在原始表中，缺失位置为 null。

### 4.2.3 标准 OOD 构造

采用 leave-one-prompt-out 或 multi-source-to-held-out-prompt：

- source sets：训练 Judge 和 detector；
- source held-out documents：ID test；
- 完整未见 essay set：OOD test；
- OOD ground truth：`essay_set` 是否属于 source set。

由于各 essay set 的题目、rubric 和分数范围可能同时变化，本文将其称为：

> cross-prompt compound OOD

而不称为纯 covariate shift。

评分结果按每个 essay set 独立计算 QWK，并将 raw score 按该 set 的官方最小/最大值归一化后计算 MAE。不同 set 的 raw scores 不直接合并成一个分类标签。

ASAP 主要用于验证检测结论能否在第二个作文语料上复现。完整更新主结论优先使用 ELLIPSE。

## 4.3 CLINC150：标准 Semantic OOD Benchmark

CLINC150 包含 150 个 in-scope intent classes，覆盖 10 个领域，并提供独立 OOS utterances。

### 4.3.1 原始结构

```json
{
  "train": [],
  "val": [],
  "test": [],
  "oos_train": [],
  "oos_val": [],
  "oos_test": []
}
```

in-scope 单条样本为：

```json
[
  "can i make a reservation for redrobin",
  "accept_reservations"
]
```

标准化后字段为：

```text
utterance,
intent_name,
intent_id,
domain_name,
split,
is_oos
```

其中 `intent_id`、`domain_name` 和 `is_oos` 可由官方 intent catalog 和 split 派生。

### 4.3.2 标准 OOD 协议

主协议只使用 in-scope train 训练 classifier 和 detector：

- ID：in-scope `test`；
- OOD：官方 `oos_test`；
- OOD ground truth：样本是否来自 `oos_test`；
- `oos_test` 不参与选型。

官方主版本包含每个 in-scope intent 100 train、20 validation、30 test，以及 1,000 个 OOS test utterances。

CLINC150 只验证标准 semantic OOD 检测和窗口漂移，不用于证明作文 Judge 的更新有效。

---


## 4.4 ROSTD：人工编写的真实对话 OOD

ROSTD（Real Out-of-Domain Sentences From Task-oriented Dialog）是专门用于自然语言 OOD detection 的数据集。它以 Schuster et al. 的 task-oriented-dialog 数据为 ID 基础；原 ID 数据包含英语任务型指令，主要覆盖 weather、alarm 和 reminder 等领域。ROSTD 另外提供约 4,000 条由人工标注者在预先了解支持域边界后编写的 OOD utterances。

ROSTD 的关键价值在于：OOD 数据不是通过随机留出某几个 intent 人工合成，而是由标注者明确编写为不属于系统已支持任务域的输入。因此它比单纯 class holdout 更接近真实部署中的 out-of-scope request。

### 4.4.1 标准化字段

不同公开镜像的文件名和存储格式可能不同。统一读取后，实验表应至少保留：

```text
utterance,
intent_name,
domain_name,
split,
is_ood,
source_dataset
```

字段含义为：

- `utterance`：用户输入文本；
- `intent_name`：ID 样本的原始 intent；OOD 样本可设为 `OOS` 或 null；
- `domain_name`：weather、alarm、reminder 等 ID domain；OOD 样本设为 `OOS`；
- `split`：train、validation 或 test；
- `is_ood`：官方 ID/OOD ground truth；
- `source_dataset`：用于防止与 CLINC150 等数据混淆。

`intent_name` 和 `domain_name` 只用于训练 ID classifier 或离线审计，不能作为 OOD detector 的直接输入提示。

### 4.4.2 公开样本例子

论文公开的 ROSTD 样本包括：

```json
{
  "utterance": "Should I be expecting rain today",
  "domain_name": "weather",
  "is_ood": 0
}
```

以及：

```json
{
  "utterance": "Why do people watch television",
  "domain_name": "OOS",
  "is_ood": 1
}
```

其他公开 OOD 示例包括：

```text
Where do pineapples grow
Tell me how to install a pool
Transfer my PayPal balance to my bank
```

这些句子不属于 weather、alarm 或 reminder 等被支持任务域，因此具有直接的 OOD ground truth。

### 4.4.3 标准 OOD 协议

主协议采用 **ID-only post-hoc detection**：

1. 只使用官方 ID train 训练 intent classifier；
2. 只使用 ID train/reference 拟合 OOD detector；
3. ID benchmark 使用独立 ID test；
4. OOD benchmark 使用官方 ROSTD OOD test；
5. OOD ground truth 直接来自官方 ID/OOD 来源；
6. OOD test 不参与阈值选择或超参数搜索。

如果某个发布版本包含 OOD train 或 OOD validation，它们不能进入主 post-hoc track。使用这些 OOD 样本训练或调阈值的方法应单独标记为 **supervised-OOD track**，不能与 ID-only 方法混表。

窗口实验按 0%、5%、10%、20%、50% 和 100% 的比例，将 ROSTD OOD utterances 混入 ID test stream。实验生成器记录每个窗口中的真实 OOD 数量，因此窗口 power、n@80% 和 delay 都有确定 ground truth。

ROSTD 适合：

- 样本级 OOD detection；
- 低 OOD prevalence 下的 AUPR 和 FPR95；
- 窗口漂移与序贯检测；
- 验证模型能否识别真实人工 OOS，而非只识别被留出的已知类别。

ROSTD 不适合直接进入作文 Judge 的 Probe、Adapt 和 Gate，因为其任务标签与作文人工评分不兼容。

## 4.5 AG News：可选的类别留出 Controlled Semantic OOD

AG News 是英语新闻主题分类数据集，由四个主题组成：

```text
World
Sports
Business
Sci/Tech
```

标准版本包含 120,000 条训练样本和 7,600 条测试样本；每类分别有 30,000 条训练样本和 1,900 条测试样本。

AG News 本身没有官方 out-of-scope split。它在本文中仅作为 **可选 controlled semantic OOD benchmark**：训练阶段完整留出一个主题类别，将该类别视为测试时未见语义类别。该实验可以增加类别留出场景，但不是论文最低必需实验。

### 4.5.1 原始字段

原始 CSV 通常包含：

```text
class_index,
title,
description
```

Hugging Face 等标准化版本常表示为：

```text
text,
label
```

其中：

- `class_index` 或 `label`：四个主题之一；
- `title`：新闻标题；
- `description`：新闻摘要；
- `text`：通常为 title 与 description 的拼接。

正式实验应固定一种输入格式，例如：

```text
[TITLE] {title} [DESCRIPTION] {description}
```

所有 baseline 使用完全相同的文本输入，不能有的方法只看标题、另一些方法读取标题和描述。

### 4.5.2 一条公开样本

```json
{
  "title": "Wall St. Bears Claw Back Into the Black (Reuters)",
  "description": "Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
  "label": "Business"
}
```

标准化后可写为：

```json
{
  "text": "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
  "label_id": 2,
  "label_name": "Business"
}
```

不同数据加载器可能使用 0–3 或 1–4 的 label ID，因此实验代码必须先映射到统一的 `label_name`，不能直接根据原始整数猜测类别。

### 4.5.3 标准 OOD 构造

采用四折 **leave-one-topic-out**：

| Fold | ID classes | OOD class |
|---|---|---|
| 1 | Sports、Business、Sci/Tech | World |
| 2 | World、Business、Sci/Tech | Sports |
| 3 | World、Sports、Sci/Tech | Business |
| 4 | World、Sports、Business | Sci/Tech |

每个 fold 的协议为：

1. 只使用三个 ID classes 的官方 train 数据训练 classifier；
2. OOD detector 只读取三个 ID classes 的 train/reference；
3. ID benchmark 为这三个类别的官方 test 样本；
4. OOD benchmark 为完整 held-out class 的官方 test 样本；
5. OOD 真值由样本是否属于 held-out class 决定；
6. 四个 fold 分别报告结果，主表报告 四折平均 和标准差。

这种设置是标准的 **class-holdout semantic OOD**。它具有明确 ground truth，但 OOD 是根据已知类别标签构造的，因此不能描述成 AG News 官方提供的自然 OOS 数据。

由于 AG News 只有四个较粗粒度主题，本文不再强行把 held-out classes 划分成 Near 和 Far。四个 fold 共同评价检测器对不同未见语义类别的稳定性。

窗口实验在每个 fold 中分别构造，例如窗口大小 50：

```text
0% OOD:   50 ID + 0 held-out-topic
10% OOD:  45 ID + 5 held-out-topic
20% OOD:  40 ID + 10 held-out-topic
```

窗口 ground truth 由 held-out-topic 的真实数量给出。

在核心数据集实验完成后，AG News 可用于：

- 样本级 semantic OOD；
- 检查检测结果是否依赖某一个特定 held-out class；
- 窗口级低比例语义变化；
- 评价分类头置信度类 baseline，如 MSP、MaxLogit、Energy；
- 评价特征空间方法，如 Mahalanobis、RMD、kNN 和 ViM。

AG News 不用于作文 Judge 的 Probe、Adapt 或 Gate。若正文篇幅、时间或算力受限，可完全不运行 AG News；也可以只在附录报告四折平均结果。将 ROSTD 句子直接作为 AG News 的外部 OOD 会形成过于容易的 cross-task Far OOD，只建议作为额外压力测试，不作为主结论。

---

# 5. Ground Truth 到底有哪些

本文需要四种不同的真值。它们不能互相替代。

| Ground truth | 从哪里获得 | 用于评价什么 |
|---|---|---|
| **ID/OOD 真值** | prompt、essay set、corpus、held-out class 或官方 OOS split | 单篇 OOD 检测 |
| **窗口组成真值** | 构造窗口时记录实际混入多少 OOD | 窗口 Power、Delay |
| **任务标签真值** | 人工作文分数、intent label、topic label | Judge 性能和漂移是否有害 |
| **更新效果真值** | 独立 source guard、Gate 和 Future 数据 | Adapt 和 Gate |

必须遵守三条规则：

1. OOD 真值不能由检测器自己的分数产生；
2. Near/Far 不能根据最终 AUROC 反向划分；
3. Harmful 不能由 MMD p 值或 OOD 分数代替，必须查看人工标签下的实际性能变化。

---

# 6. 统一测试流程

## 6.1 先冻结数据划分

在训练前就固定：

1. Judge 训练集；
2. source reference；
3. detector calibration；
4. ID test；
5. OOD test；
6. 窗口模拟数据；
7. Probe；
8. adaptation validation；
9. Gate；
10. Future。

同一篇原始文档及其派生记录只能出现在一个集合中。

## 6.2 样本级检测怎么比

- 所有方法使用同一个 Judge 和同一层 hidden representation；
- 所有检测器只使用 ID 训练数据拟合；
- 目标 OOD test 不用于训练、选阈值或调参数；
- 使用过 OOD validation 的方法必须单独列为 supervised-OOD track。

这样才能公平比较 Residual-only ViM、Mahalanobis、RMD、kNN、MSP、Energy 等方法。

## 6.3 窗口检测怎么比

所有方法使用：

- 相同 source reference；
- 相同窗口大小；
- 相同 OOD 混入比例；
- 相同重复次数；
- 相同 arrival block 定义。

正式统计检验和只输出诊断分数的方法需要分开说明，不能混写相同的误报保证。

## 6.4 Probe 和更新怎么比

- 所有 Probe 方法使用相同最大标签预算；
- 所有更新方法使用同一批目标标签；
- replay 方法使用相同数量的源域标签；
- Gate 和 Future 不能参与训练或调参；
- Full-label 只作为理想上界。

---

# 7. Baseline、指标及其实际含义

本章先规定每个实验究竟比较什么，再介绍对应 baseline。所有方法只能在相同数据、相同 Judge、相同标签预算和相同测试划分下比较。

## 7.1 六类实验的统一定义

| 实验 | 评价单位 | 输入 | 方法输出 | Ground truth | 主指标 | 使用的数据集 |
|---|---|---|---|---|---|---|
| Judge 能力 | 单篇文档 | 文本或作文 | 任务预测结果 | 人工分数或类别标签 | QWK；分类任务用 Accuracy/Macro-F1 | ELLIPSE、ASAP-AES；文本分类数据用于训练分类头 |
| 单篇 OOD 检测 | 单篇文档 | hidden state 或 logits | 连续 OOD 分数 | 官方 OOS 标签，或预注册的 prompt/class membership | AUROC | 全部核心数据集 |
| 窗口漂移检测 | 一批文档 | source reference 与当前窗口 | p 值或报警结果 | 窗口中实际混入的 OOD 比例 | Type-I Error、Power | 全部核心数据集 |
| Harmful 判断 | 一个报警窗口 | 少量带任务标签的 Probe 样本 | Harmful、Benign 或 Uncertain | 使用该窗口全部已有任务标签得到的 oracle 状态 | Status Macro-F1、Average Labels | ELLIPSE、ASAP-AES |
| 受控更新 | 一个目标域 | Probe/Adapt 标签与源域 replay | 候选更新模型 | 独立 Future 和 source guard 标签 | Target Gain、Source Drop、NFR | ELLIPSE、ASAP-AES |
| Gate | 一个候选更新 | Gate 数据上的目标与源域结果 | Accept、Reject 或 Defer | 独立 Future 判断该更新最终是好还是坏 | Bad-update Acceptance、Good-update Rejection | ELLIPSE、ASAP-AES |

其中：

- **单篇 OOD 检测**回答“这一篇是否异常”；
- **窗口漂移检测**回答“最近这一批数据是否整体变化”；
- **Harmful 判断**回答“这种变化是否真的使任务性能下降”；
- **Adapt 和 Gate**回答“如何更新，以及该更新是否可以安全上线”。

这些任务的 ground truth 不相同，不能用一个 OOD 分数同时代替全部真值。

---

## 7.2 Judge Baseline：先确认任务模型本身有效

Judge baseline 只比较任务预测能力，不比较 OOD 检测能力。

### 7.2.1 方法表

| 方法 | 输入 | 是否训练 | 使用的监督信息 | 输出 | 主要作用 |
|---|---|---:|---|---|---|
| EASE-SVR | 人工设计的作文特征 | 是 | 人工作文分数 | 连续分数 | 传统 AES 基线 |
| EASE-BLRR | 人工设计特征 | 是 | 人工作文分数 | 分数分布或连续分数 | 贝叶斯回归基线 |
| GPT Pointwise Judge | 作文、题目和 rubric | 否或少量提示调节 | rubric；不使用测试标签 | 直接评分 | 强 LLM Judge 基线 |
| Frozen LLM + Linear Head | 冻结 LLM hidden state | 只训练线性头 | source-domain 人工分数 | 分数或分数等级 | 本文统一 Judge |
| Frozen LLM + MLP Head（附录） | 冻结 LLM hidden state | 训练 MLP | source-domain 人工分数 | 分数或分数等级 | 检查分类头容量影响 |

PandaLM 是 pairwise evaluator，输出“回答 A 还是回答 B 更好”，与 pointwise 作文评分的输出形式不同，只能放入单独的 pairwise appendix，不能直接与上表排名。

### 7.2.2 指标

| 指标 | 实际含义 | 例子 | 方向 | 是否主指标 |
|---|---|---|---|---|
| QWK | 模型评分等级与人工评分的一致性，并对大幅错分给予更大惩罚 | 人工 5 分、模型 1 分比模型 4 分受到更重惩罚 | 越高越好 | **是** |
| MAE | 模型平均与人工相差多少分 | MAE=0.6 表示平均相差 0.6 分 | 越低越好 | 是 |
| Spearman | 模型能否正确排序作文质量 | A>B>C 的排序是否与人工一致 | 越高越好 | 次要 |
| Accuracy/Macro-F1 | 文本分类任务中类别是否预测正确 | 用于 CLINC150、ROSTD、AG News 的分类头 | 越高越好 | 仅用于分类任务 |

### 7.2.3 公平比较要求

- ELLIPSE 和 ASAP-AES 使用相同 train/validation/test 划分；
- 所有方法使用相同的分数映射；
- GPT Judge 的 prompt、rubric 和输出解析规则必须固定；
- 不允许在 OOD test 上选择 Judge、层或分类头；
- 主表报告均值和标准差，至少使用多个随机种子或多个预注册 split。

---

## 7.3 样本级 OOD Baseline：比较单篇文档能否识别

### 7.3.1 方法表

| 方法 | 使用的信息 | 是否额外拟合 | 是否需要 ID 类别标签 | 输出 | 主要比较点 |
|---|---|---:|---:|---|---|
| MSP | softmax 最大概率 | 否 | 否 | OOD 分数 | 最简单置信度基线 |
| MaxLogit | 最大原始 logit | 否 | 否 | OOD 分数 | 避免 softmax 归一化影响 |
| Energy | 全部 logits | 否 | 否 | OOD 分数 | 综合所有类别分数 |
| Mahalanobis | hidden feature 与类别中心的距离 | 是 | 是 | OOD 分数 | 类条件距离 |
| RMD | 类条件距离减去全局背景距离 | 是 | 是 | OOD 分数 | 改善 Near-OOD |
| kNN | 与最近 ID 特征的距离 | 建索引 | 否 | OOD 分数 | 非参数局部距离 |
| Full ViM | residual 与 logits | 是 | 是 | OOD 分数 | 标准 ViM |
| **Residual-only ViM** | 主子空间外的 residual 大小 | 是 | 否 | OOD 分数 | 本文主检测器 |

### 7.3.2 指标

| 指标 | 实际含义 | 例子 | 方向 | 备注 |
|---|---|---|---|---|
| **AUROC** | 随机取一篇 ID 和一篇 OOD，方法把 OOD 排在前面的概率 | AUROC=0.90 表示约 90% 的随机配对顺序正确 | 越高越好 | 主指标，不依赖固定阈值 |
| AUPR-OOD | OOD 较少时，报警的准确性和 OOD 找回率的综合表现 | 适合 5% 或 10% OOD prevalence | 越高越好 | 必须说明测试集 OOD 比例 |
| FPR95 | 找回 95% OOD 时，有多少 ID 被误报 | FPR95=0.10 表示 10% ID 被误报 | 越低越好 | 体现高召回下的误报代价 |
| Runtime | 计算每篇 OOD 分数所需时间 | 毫秒/文档 | 越低越好 | kNN 还应报告索引内存 |
| AUROC 标准差 | 不同种子或 split 下结果是否稳定 | 0.91±0.02 | 越低越稳定 | 必须与均值一起报告 |

普通 Accuracy 不作为主指标，因为它依赖阈值和 ID/OOD 比例，容易在 ID 占多数时产生误导。

### 7.3.3 统一协议

- 所有方法使用相同 backbone、层、pooling 和分类头；
- 主赛道只允许使用 ID train/reference 拟合检测器；
- OOD validation 不参与主赛道的参数选择；
- 阈值只能由 source calibration 或预注册规则确定；
- ELLIPSE、ASAP-AES、CLINC150、ROSTD 分别报告结果；
- AG News 为可选四折 leave-one-topic-out；
- Near/Far 只作为预注册子分析，不根据结果反向划分。

---

## 7.4 窗口漂移 Baseline：比较最近一批数据是否变化

### 7.4.1 方法表

| 方法 | 比较的表示 | 是否训练额外模型 | 是否输出正式 p 值 | 主要优点 | 主要限制 |
|---|---|---:|---:|---|---|
| A-space MMD | 原始文档 embedding | 否 | 是 | 通用两样本检验 | 可能检测到与 Judge 无关的变化 |
| **B-space MMD** | residual vector | 否 | 是 | 更接近 Judge 表征异常 | 需要固定核、带宽和置换方案 |
| C2ST-permutation | source/target 可分性 | 是 | 是 | 对复杂差异有较高功效 | 计算成本高，需防止过拟合 |
| KS | residual norm | 否 | 是 | 简单、快速 | 只检查一维变化 |
| BBSDs | softmax 概率向量 | 否 | 是，需多重校正 | 直接监测预测分布 | 依赖分类头质量 |
| BBSDh | 最终预测类别频率 | 否 | 是 | 计算最简单 | 信息损失最大 |

### 7.4.2 单窗口指标

| 指标 | 实际含义 | 例子 | 方向 |
|---|---|---|---|
| **Type-I Error** | 纯 ID 窗口被误报的比例 | 1,000 个 ID 窗口中误报 45 个，即 4.5% | 应接近且不超过预设水平 |
| **Power@5/10/20% OOD** | 窗口真实混入对应比例 OOD 时成功检出的比例 | 10% OOD 的 500 个窗口中检出 400 个，Power=80% | 越高越好 |
| n@80% | 达到 80% Power 所需的最小窗口样本数 | n@80%=100 表示至少需要约 100 篇文档 | 越低越好 |
| Runtime | 每次窗口检验的运行时间 | 秒/窗口 | 越低越好 |

### 7.4.3 连续监控指标

| 指标 | 实际含义 | 例子 | 方向 |
|---|---|---|---|
| **FWER** | 一个完整纯 ID episode 中至少误报一次的概率 | 1,000 个 episode 中 20 个出现过误报，即 2% | 越低越好 |
| Detection Rate | 漂移 episode 最终被确认的比例 | 100 个漂移 episode 中确认 90 个 | 越高越好 |
| Delay | 漂移开始到正式确认需要多少窗口或文档 | 平均 2.4 个窗口 | 越低越好 |
| Transient false persistence | 短暂变化是否被错误判断为持续漂移 | 2 个异常窗口后恢复正常，但系统仍确认持续漂移 | 越低越好 |

### 7.4.4 统一协议

每个数据集至少测试：

- 0% OOD：检查 Type-I Error；
- 5% OOD：困难低比例场景；
- 10% OOD：主要 Power 指标；
- 20% OOD：中等强度场景；
- 100% OOD：仅作为强漂移 sanity check，不作为主要结论。

所有方法使用相同：

- source reference；
- 窗口大小；
- OOD 混入比例；
- 重复次数；
- arrival blocks；
- 显著性水平；
- 序贯 persistence 或 alpha-spending 规则。

单窗口检验和序贯监控必须分开报告。不能因为单窗口 p 值显著，就直接声称整个连续监控过程控制了 FWER。

---

## 7.5 Probe Baseline：判断漂移是否真的 Harmful

Probe 的 ground truth 来自目标窗口的全部已有任务标签。实验时只向方法公开少量 Probe 标签，再将其判断与 full-label oracle 比较。

### 7.5.1 方法表

| 方法 | 使用标签数 | 样本选择方式 | 是否容易产生抽样偏差 | 是否自然提供不确定性 | 输出 |
|---|---:|---|---:|---:|---|
| ATC | 0 | 不抽样，使用置信度阈值估计性能 | 是 | 否 | 性能估计或状态 |
| DoC | 0 | 使用平均置信度变化估计性能变化 | 是 | 否 | 性能变化估计 |
| Fixed Random Probe | 固定 B | 均匀随机 | 否 | 可以 | 状态和风险变化 |
| Confidence Sampling | 固定 B | 优先选择低置信度文档 | 是；需加权校正 | 可以 | 状态和风险变化 |
| Residual-stratified Probe | 固定 B | 按 residual 区间分层抽样 | 低；需使用分层权重 | 可以 | 状态和风险变化 |
| **Sequential Random Probe** | 最多 B | 先少量随机抽样，证据不足再追加 | 否 | 是 | Harmful/Benign/Uncertain |
| Full-label Oracle | 全部 | 使用目标窗口全部标签 | 否 | 不需要 | 实验 ground truth |

ATC 和 DoC 属于 zero-label performance estimation；随机 Probe 属于少量人工标签估计。两类方法可以放在同一结果表中比较最终状态，但必须清楚标注它们使用的监督资源不同。

### 7.5.2 状态定义

- **Harmful**：完整标签显示目标域任务误差明显超过源域或超过允许退化阈值；
- **Benign**：完整标签能够证明性能退化没有超过允许范围；
- **Uncertain**：现有 Probe 标签不足以支持前两种结论。

允许退化阈值必须在实验前固定，不能根据结果调整。

### 7.5.3 指标

| 指标 | 实际含义 | 方向 | 必须报告的补充项 |
|---|---|---|---|
| **Status Macro-F1** | Harmful、Benign、Uncertain 三种状态的平均识别质量 | 越高越好 | 每类 F1 |
| Harmful Recall | 所有真实 Harmful 窗口中成功识别多少 | 越高越好 | 避免漏掉危险漂移 |
| Benign Recall | 所有真实 Benign 窗口中成功识别多少 | 越高越好 | 避免无谓更新 |
| Risk-change Error | 少量标签估计的性能变化与 full-label oracle 相差多少 | 越低越好 | Bias 和 RMSE 可放附录 |
| CI Coverage | 声称的置信区间实际覆盖真值的比例 | 接近预设水平 | 同时报告平均区间宽度 |
| Average Labels | 平均实际使用多少人工标签 | 越低越好 | 同时报告最大标签数 |
| Uncertain Rate | 有多少窗口最终无法判断 | 视任务而定 | 不能只靠大量 Uncertain 获得低错误率 |

---

## 7.6 Adapt Baseline：在相同标签预算下更新模型

### 7.6.1 方法表

| 方法 | 更新参数 | 使用目标标签 | 使用 source replay | 稳定机制 | 作用 |
|---|---|---:|---:|---|---|
| No Update | 无 | 0 | 否 | 无 | 不更新基线 |
| Target-only Head FT | 分类/回归头 | B | 否 | 无 | 检查最简单目标域微调 |
| Head + Replay | 分类/回归头 | B | 是 | replay | 降低遗忘 |
| **Head + Replay + Anchor** | 分类/回归头 | B | 是 | L2-SP 或等价锚定 | 本文主更新方法 |
| EWC-style Head（可选） | 分类/回归头 | B | 可选 | 参数重要性约束 | 额外抗遗忘基线 |
| LoRA + Replay | LoRA 参数 | B | 是 | replay | 更高容量轻量更新 |
| Full Fine-tuning（可选） | 全部模型参数 | B | 是 | replay/regularization | 高成本上界 |
| Full-label Target Oracle | 预注册参数范围 | 全部目标标签 | 按协议 | 按协议 | 标签上界，不参与公平预算排名 |

### 7.6.2 指标

| 指标 | 实际含义 | 方向 |
|---|---|---|
| Target Future Gain | 更新后在独立目标 Future 数据上改善多少 | 越高越好 |
| Target Future QWK/MAE | 更新后的目标域绝对性能 | QWK 越高、MAE 越低 |
| Source QWK Drop | 更新前后源域 QWK 下降多少 | 越低越好 |
| Source MAE Increase | 更新前后源域 MAE 上升多少 | 越低越好 |
| NFR | 原模型正确、更新后变错的源域样本比例 | 越低越好 |
| Trainable Parameters | 实际更新多少参数 | 越低越轻量 |
| GPU Time/Memory | 训练时间与峰值显存 | 越低越好 |
| Seed Stability | 多个随机种子下结果是否稳定 | 方差越低越好 |

所有方法必须使用：

- 相同目标标签预算 B；
- 相同 target Adapt split；
- 相同 source replay 上限；
- 相同 Gate 和 Future 数据；
- 相同候选学习率搜索预算。

Full-label oracle 只表示理想上界，不能与预算为 B 的方法直接宣称公平胜负。

---

## 7.7 Gate Baseline：决定候选更新能否上线

Gate 和 Adapt 必须分开比较。Adapt 负责产生候选模型；Gate 负责决定 Accept、Reject 或 Defer。

### 7.7.1 Gate 组件消融

| Gate 方法 | 检查目标改善 | 检查不确定性 | 检查源域 NFR | 检查源域 QWK/MAE | 输出 |
|---|---:|---:|---:|---:|---|
| No Gate | 否 | 否 | 否 | 否 | 所有候选都上线 |
| Gain-only | 是 | 否 | 否 | 否 | Accept/Reject |
| Gain + Confidence Bound | 是 | 是 | 否 | 否 | Accept/Reject/Defer |
| Gain + Confidence + NFR | 是 | 是 | 是 | 否 | Accept/Reject/Defer |
| **Full Gate** | 是 | 是 | 是 | 是 | Accept/Reject/Defer |

### 7.7.2 Gate ground truth

使用完全独立的 Future 和 source guard 数据，将候选更新定义为：

- **Good update**：目标域达到最低改善要求，同时源域退化和 NFR 都在允许范围内；
- **Bad update**：目标域没有达到改善要求，或违反任一源域安全限制；
- **Borderline update**：结果接近边界，统计证据不足。

Gate 阈值只能在独立 candidate-development 数据上确定，不能看完 Future 结果后回调。

### 7.7.3 指标

| 指标 | 实际含义 | 方向 |
|---|---|---|
| **Bad-update Acceptance** | 所有真实坏更新中，被 Gate 错误放行的比例 | 越低越好 |
| Good-update Rejection | 所有真实好更新中，被 Gate 错误拒绝的比例 | 越低越好 |
| Defer Rate | 有多少候选被暂缓，需要更多标签 | 需要与安全性一起解释 |
| Accepted Target Gain | 真正上线模型的平均目标域改善 | 越高越好 |
| Accepted Source Drop | 真正上线模型的平均源域退化 | 越低越好 |
| Accepted NFR | 真正上线模型的平均 NFR | 越低越好 |
| Decision Coverage | Gate 能直接 Accept 或 Reject 的比例 | 越高越好，但不能牺牲安全性 |

---

# 8. 论文最终实验表格

完整实验建议使用八张表。正文篇幅有限时，可将 Table 1、Table 7 和部分详细成本列放入附录，但实验本身仍应完成。

## Table 1：数据集与 OOD Protocol

| Dataset | Role | ID 定义 | OOD 定义 | OOD 真值来源 | 任务标签 | Natural/Constructed | 用于哪些任务 |
|---|---|---|---|---|---|---|---|
| ELLIPSE | 核心生命周期 | source prompts | held-out prompts | prompt membership | Overall + six traits | Constructed prompt OOD | Judge、Sample、Window、Probe、Adapt、Gate |
| ASAP-AES | 核心复现 | source essay sets | held-out essay sets | essay_set membership | resolved score | Constructed cross-prompt OOD | Judge、Sample、Window、Probe、Adapt、Gate |
| CLINC150 | 标准文本 OOD | in-scope intents | official OOS | 官方 split | intent/OOS label | Natural official OOS | Judge classifier、Sample、Window |
| ROSTD | 标准文本 OOD | supported task domains | annotator-written OOD | 官方 ID/OOD 来源 | intent/OOS label | Natural annotator-written OOD | Judge classifier、Sample、Window |
| AG News（optional） | 补充类别留出 | 每折三个 topics | held-out topic | topic membership | topic label | Constructed class-holdout OOD | Judge classifier、Sample、Window |

表下注明每个数据集的 train、calibration、ID test 和 OOD test 样本数；不能只写总样本量。

---

## Table 2：Judge / Classifier ID Performance

| Dataset | Method | Train labels | QWK ↑ | MAE ↓ | Spearman ↑ | Accuracy ↑ | Macro-F1 ↑ | Mean±Std | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| ELLIPSE | EASE-SVR | source all | TBD | TBD | TBD | -- | -- | TBD | Overall |
| ELLIPSE | EASE-BLRR | source all | TBD | TBD | TBD | -- | -- | TBD | Overall |
| ELLIPSE | GPT Pointwise Judge | 0 test labels | TBD | TBD | TBD | -- | -- | TBD | fixed rubric |
| ELLIPSE | Ours Frozen LLM + Linear Head | source all | TBD | TBD | TBD | -- | -- | TBD | main Judge |
| ASAP-AES | EASE-SVR | source all | TBD | TBD | TBD | -- | -- | TBD | prompt-wise |
| ASAP-AES | EASE-BLRR | source all | TBD | TBD | TBD | -- | -- | TBD | prompt-wise |
| ASAP-AES | GPT Pointwise Judge | 0 test labels | TBD | TBD | TBD | -- | -- | TBD | fixed rubric |
| ASAP-AES | Ours Frozen LLM + Linear Head | source all | TBD | TBD | TBD | -- | -- | TBD | prompt-wise QWK |
| CLINC150 | Frozen LLM + Linear Head | ID train | -- | -- | -- | TBD | TBD | TBD | in-scope only |
| ROSTD | Frozen LLM + Linear Head | ID train | -- | -- | -- | TBD | TBD | TBD | supported intents |
| AG News（optional） | Frozen LLM + Linear Head | 3-class train | -- | -- | -- | TBD | TBD | TBD | 4-fold average |

主表可以将作文评分和文本分类拆成两个 panel，避免不同指标直接比较。

---

## Table 3：Sample-level OOD Detection

单元格记为 `AUROC / AUPR-OOD / FPR95`，并报告均值±标准差。

| Method | Extra OOD data | ELLIPSE | ASAP-AES | CLINC150 | ROSTD | AG News（optional） | Core Mean Rank ↓ | ms/doc ↓ |
|---|---:|---|---|---|---|---|---:|---:|
| MSP | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MaxLogit | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Energy | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Mahalanobis | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RMD | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| kNN | No | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Full ViM | No | TBD | current ASAP result available | TBD | TBD | TBD | TBD | TBD |
| **Residual-only ViM** | No | TBD | current ASAP result available | TBD | TBD | TBD | TBD | TBD |

补充规则：

- `Core Mean Rank` 只使用 ELLIPSE、ASAP-AES、CLINC150 和 ROSTD；
- AG News 报告四折平均和标准差；
- Near/Far、不同 OOD prevalence 和不同层的结果放附录；
- 使用 OOD validation 的方法单列为 supervised-OOD track，不能混入本表。

---

## Table 4：Window-level Shift Detection

每个核心数据集均重复下表。正文可以分成四个 panel，也可以只在正文展示 Power@10%，完整曲线放附录。

| Dataset | Method | Type-I@0.05 ↓ | FWER ↓ | Power@5% ↑ | Power@10% ↑ | Power@20% ↑ | n@80% ↓ | Delay ↓ | Runtime/window ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ELLIPSE | A-MMD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | **B-MMD** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | C2ST-permutation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | KS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | BBSDs | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | BBSDh | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ASAP-AES | A-MMD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ASAP-AES | **B-MMD** | current result available | current result available | TBD | current result available | current result available | TBD | current result available | TBD |
| ASAP-AES | C2ST-permutation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ASAP-AES | KS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ASAP-AES | BBSDs | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ASAP-AES | BBSDh | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLINC150 | A-MMD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLINC150 | **B-MMD** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLINC150 | C2ST-permutation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLINC150 | KS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLINC150 | BBSDs | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CLINC150 | BBSDh | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROSTD | A-MMD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROSTD | **B-MMD** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROSTD | C2ST-permutation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROSTD | KS | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROSTD | BBSDs | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROSTD | BBSDh | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

还应在附录报告：

- 100% OOD sanity check；
- 不同窗口大小；
- 短暂 1–2 窗口变化；
- 持续变化；
- 未检出 episode 的处理方式；
- kernel、bandwidth 和 permutation 数量。

---

## Table 5：Probe / Harmfulness Estimation

该表只在 ELLIPSE 和 ASAP-AES 上运行。每个数据集至少分别报告 Near、Far、Benign 和 Harmful 场景。

| Dataset | Scenario | Method | Max labels | Avg labels ↓ | Status Macro-F1 ↑ | Harmful Recall ↑ | Benign Recall ↑ | Risk Error ↓ | CI Coverage | Uncertain Rate |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ELLIPSE | Near/Far | ATC | 0 | 0 | TBD | TBD | TBD | TBD | -- | TBD |
| ELLIPSE | Near/Far | DoC | 0 | 0 | TBD | TBD | TBD | TBD | -- | TBD |
| ELLIPSE | Near/Far | Fixed Random Probe | B | B | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near/Far | Confidence Sampling | B | B | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near/Far | Residual-stratified Probe | B | B | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near/Far | **Sequential Random Probe** | B | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near/Far | Full-label Oracle | all | all | 1.000 | 1.000 | 1.000 | 0 | -- | 0 |
| ASAP-AES | Near/Far | ATC | 0 | 0 | TBD | TBD | TBD | TBD | -- | TBD |
| ASAP-AES | Near/Far | DoC | 0 | 0 | TBD | TBD | TBD | TBD | -- | TBD |
| ASAP-AES | Near/Far | Fixed Random Probe | B | B | current result available | TBD | TBD | current result available | current result available | TBD |
| ASAP-AES | Near/Far | Confidence Sampling | B | B | current result available | TBD | TBD | current result available | TBD | TBD |
| ASAP-AES | Near/Far | Residual-stratified Probe | B | B | current result available | TBD | TBD | current result available | current result available | TBD |
| ASAP-AES | Near/Far | **Sequential Random Probe** | B | current result available | current result available | TBD | TBD | current result available | current result available | TBD |
| ASAP-AES | Near/Far | Full-label Oracle | all | all | 1.000 | 1.000 | 1.000 | 0 | -- | 0 |

主表中的 `Scenario` 可以拆成 Near 和 Far 两个 panel；Benign/Harmful 的具体构造与 oracle 判断规则必须在表注中写明。

---

## Table 6：Controlled Adaptation

| Dataset | Shift | Method | Target labels | Replay labels | Target Before | Target After | Target Gain ↑ | Source Drop ↓ | NFR ↓ | Trainable Params ↓ | GPU Time ↓ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ELLIPSE | Near | No Update | 0 | 0 | TBD | same | 0 | 0 | 0 | 0 | 0 |
| ELLIPSE | Near | Target-only Head FT | B | 0 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near | Head + Replay | B | R | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near | **Head + Replay + Anchor** | B | R | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near | LoRA + Replay | B | R | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ELLIPSE | Near | Full-label Oracle | all | R | TBD | TBD | upper bound | TBD | TBD | TBD | TBD |
| ELLIPSE | Far | same method block |  |  |  |  |  |  |  |  |  |
| ASAP-AES | Near | same method block |  |  | current result available | current result available | current result available | current result available | current result available | TBD | TBD |
| ASAP-AES | Far | same method block |  |  | current result available | current result available | current result available | current result available | current result available | TBD | TBD |

说明：

- `Target Before/After` 对作文数据优先报告 QWK 和 MAE；
- `Target Gain` 必须在独立 Future 上计算；
- `Source Drop` 同时报告 QWK drop 和 MAE increase；
- 每个方法至少运行多个随机种子；
- 若 Far OOD 下所有 head-only 方法都失败，应将结果解释为容量限制，而不是只继续调学习率。

---

## Table 7：Gate Safety Evaluation

| Gate | Candidate set | Bad updates | Good updates | Bad-update Acceptance ↓ | Good-update Rejection ↓ | Defer Rate | Accepted Target Gain ↑ | Accepted Source Drop ↓ | Accepted NFR ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No Gate | TBD | TBD | TBD | TBD | 0 | 0 | TBD | TBD | TBD |
| Gain-only | TBD | TBD | TBD | TBD | TBD | 0 | TBD | TBD | TBD |
| Gain + Confidence Bound | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Gain + Confidence + NFR | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **Full Gate** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

必须使用两个独立候选集合：

1. development candidate set：确定 Gate 阈值；
2. evaluation candidate set：正式计算 Table 7。

不能在同一批候选更新上既选阈值又报告最终 Gate 结果。

---

## Table 8：End-to-end Equal-label-budget Comparison

该表用于证明完整系统，而不是只证明单个模块。

| System | OOD detector | Window detector | Probe | Adapt | Gate | Total target labels | FWER ↓ | Harmful Recall ↑ | Target Future Gain ↑ | Source Drop ↓ | Bad update deployed ↓ | Total Runtime ↓ |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| No Monitoring / No Update | -- | -- | -- | -- | -- | 0 | 0 | 0 | 0 | 0 | 0 | lowest |
| Sample OOD only | Best sample baseline | -- | fixed random | target-only FT | no gate | B | -- | TBD | TBD | TBD | TBD | TBD |
| MMD + Fixed Probe | -- | B-MMD | fixed random | Head + Replay | gain-only | B | TBD | TBD | TBD | TBD | TBD | TBD |
| Strong Baseline Pipeline | Best non-ours sample detector | best non-ours window detector | best non-ours Probe | best non-ours Adapt | Full Gate | B | TBD | TBD | TBD | TBD | TBD | TBD |
| **Full Proposed System** | Residual-only ViM | B-MMD + sequential rule | Sequential Random Probe | Head + Replay + Anchor | Full Gate | B | TBD | TBD | TBD | TBD | TBD | TBD |
| Full-label Oracle | oracle OOD membership | oracle shift time | all labels | full-label update | oracle | all | 0 | 1.000 | upper bound | TBD | 0 | highest |

端到端表必须保证：

- 所有非 oracle 系统使用相同目标标签总预算；
- 不能给某个系统更多 Probe 标签或更多 Adapt 标签；
- 未检测到漂移时不允许偷偷使用目标标签；
- 所有系统在相同 episode、相同 arrival order 和相同 Future 数据上运行；
- 同时报告检测安全性、标签成本、更新收益和源域损害。

---

# 9. 现有结果应如何放入新结构

现有阶段性结果可保留，但只迁移到对应任务：

- Residual-only ViM、Mahalanobis、Full ViM、kNN、RMD、Energy、MSP 的数值进入 Table 3 的 ASAP 列；补跑 MaxLogit。随后优先在 CLINC150 和 ROSTD 上运行完全相同的 detector 集合。AG News 四折协议仅在资源允许时补充。
- A/B MMD、C2ST、KS、BBSDs、BBSDh 进入 Table 4；强漂移和低比例漂移分开报告。
- Layer、分类头、ViM rank、threshold、kernel 和 bandwidth 全部移到附录。
- 聚类和 Probe 可作为方法消融附录；正文只需说明 Probe 如何产生 Harmful/Benign/Uncertain。
- 查看 Future 后选择出的 Adapt 学习率、radius 和 Gate 阈值不能作为正式结果，必须在独立 split 上冻结后再进入 Table 5。

---

# 10. 最终任务陈述

本文研究一个连续的部署问题：

> 一个 LLM Judge 在源域人工评分数据上训练完成后，系统首先判断单篇新文档是否偏离训练分布，再判断最近一批文档是否发生持续漂移。检测到漂移后，系统使用少量人工标签判断这种变化是否真的导致 Judge 性能下降。只有确认有害时，才进行轻量更新，并通过独立 Gate 决定是否上线。

最终需要回答五个问题：

1. Residual-only ViM 能否在未见 prompt、官方 OOS 和人工编写 OOD 上稳定优于常见 OOD 分数？
2. residual-vector MMD 能否在控制误报的同时检测少量 OOD 混入？
3. 少量随机 Probe 标签能否正确区分 Harmful、Benign 和 Uncertain？
4. 在相同标签预算下，Replay + Anchor 是否比简单微调更稳定？
5. Safety Gate 能否减少坏更新上线，同时保留真正有效的更新？

AG News 只作为可选补充，用于检查上述检测结论能否扩展到 held-out news topic。

---

# 参考文献

1. Arora, U., Huang, W. and He, H. (2021). *Types of Out-of-Distribution Texts and How to Detect Them*. EMNLP.
2. Larson, S. et al. (2019). *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*. EMNLP-IJCNLP.
3. Gangal, V., Arora, A., Einolghozati, A. and Gupta, S. (2020). *Likelihood Ratios and Generative Classifiers for Unsupervised Out-of-Domain Detection in Task Oriented Dialog*. AAAI.
4. Schuster, S., Gupta, S., Shah, R. and Lewis, M. (2019). *Cross-lingual Transfer Learning for Multilingual Task Oriented Dialog*. NAACL-HLT.
5. Zhang, X., Zhao, J. and LeCun, Y. (2015). *Character-level Convolutional Networks for Text Classification*. NeurIPS.
6. Crossley, S. A. et al. (2023). *The English Language Learner Insight, Proficiency and Skills Evaluation (ELLIPSE) Corpus*. International Journal of Learner Corpus Research.
7. Wang, H. et al. (2022). *ViM: Out-of-Distribution with Virtual-logit Matching*. CVPR.
8. Ren, J. et al. (2021). *A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection*.
9. Sun, Y. et al. (2022). *Out-of-Distribution Detection with Deep Nearest Neighbors*. ICML.
7. Liu, W. et al. (2020). *Energy-based Out-of-distribution Detection*. NeurIPS.
8. Gretton, A. et al. (2012). *A Kernel Two-Sample Test*. JMLR.
9. Lopez-Paz, D. and Oquab, M. (2017). *Revisiting Classifier Two-Sample Tests*. ICLR.
10. Rabanser, S., Günnemann, S. and Lipton, Z. (2019). *Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift*. NeurIPS.
