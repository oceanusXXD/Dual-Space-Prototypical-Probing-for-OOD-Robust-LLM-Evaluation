# Algorithm Flow

本文档按当前已经迁移到 `src/algorithm` 的实际代码说明算法流程。它不是旧 numbered script 的运行说明，而是当前核心包的算法结构、数据契约和 CLI 链路。

## 1. 总体链路

当前算法主链路固定为：

```text
原始文本
-> Hidden-state 表征
-> 评分分类器
-> 错误风险检测器
-> WSR 阈值认证
-> 逐条 accept/reject
-> 窗口级失效确认
-> 模型更新
-> 重新认证
```

代码里的权威链路定义在：

- `src.algorithm.data.flow.ALGORITHM_CHAIN`
- `src.algorithm.pipeline.algorithm_chain()`

链路中的每一步都对应一个清晰的输入/输出契约，避免旧脚本之间通过路径和临时字段互相调用。

## 2. 原始文本

入口数据通过 `src.common.schema.JudgeRecord` 表示。每条记录同时区分：

- `input_document_text`: A-space 使用的原始输入文档文本。
- `judge_input_text`: B-space 使用的冻结 judge prompt / scoring context。
- `sample_id`: 与 hidden-state cache 对齐的行 ID。
- `query_id`: 多任务或多 rubric 场景下的评分维度。
- `label` / `split`: 分类器训练、校准、监控和评估使用。

相关模块：

- `src/common/schema.py`
- `src/common/feature_store.py`
- `src/common/validation.py`

## 3. Hidden-state 表征

Hidden-state 是算法第一环，负责把原始文本转成 A/B 两个空间的 frozen representation。

实际代码位置：

- `src/algorithm/hidden_state/extract.py`
- `src/algorithm/hidden_state/qwen_hidden.py`
- `src/algorithm/hidden_state/views.py`
- `src/algorithm/hidden_state/layer_selection.py`
- `src/algorithm/hidden_state/pooling.py`
- `src/algorithm/hidden_state/score_token.py`
- `src/algorithm/hidden_state/contract.py`

当前支持的核心能力：

- A-space view: `input_document_masked_mean`
- B-space view: `judge_prompt_masked_mean`
- span/decision-state 预留 view: `candidate_span_mean`, `rubric_task_span_mean`, `pre_score_token`, `pre_label_token`
- layer selection: 单层、多层、last layer、all layers、layer mean
- pooling: `masked_mean`, `last_token`, `span_mean`, pre-answer/pre-score/pre-label token hidden state
- cache metadata: `space`, `feature_scope`, `layers`, `pooling`, `model_id`, `revision`, `prompt_template`, `max_length`, `view`

CLI:

```bash
python -m src.algorithm extract \
  --records rows.jsonl \
  --space b \
  --view pre_score_token \
  --layers -1 \
  --model-path /path/to/qwen \
  --output b_hidden.npz
```

输出是 hidden-state feature cache，供后续 classifier 使用。

## 4. 评分分类器

评分分类器在冻结 hidden-state 上训练，不再直接依赖旧脚本。分类器输出统一走 `JudgeHeadOutput`，供 detector 使用。

实际代码位置：

- `src/algorithm/classifier/base.py`
- `src/algorithm/classifier/output.py`
- `src/algorithm/classifier/train.py`
- `src/algorithm/classifier/predict.py`
- `src/algorithm/classifier/selection.py`
- `src/algorithm/classifier/linear_softmax.py`
- `src/algorithm/classifier/coral_ordinal.py`
- `src/algorithm/classifier/ridge.py`
- `src/algorithm/classifier/logistic.py`

当前 CLI 暴露的分类器：

- `linear`
- `linear_softmax`
- `coral`
- `ridge`
- `logistic`

主要输出：

- predicted score / class
- probability
- logits
- penultimate representation
- optional exact affine head parameters, used by full ViM / OpenOOD-style methods

CLI:

```bash
python -m src.algorithm train-classifier \
  --features b_hidden.npz \
  --records rows.jsonl \
  --train-split training_train \
  --val-split training_validation \
  --classifier coral \
  --output classifier.joblib
```

## 5. 错误风险检测器

检测器消费 classifier 的 penultimate/logits/prediction，输出每条样本的 risk / OOD score。它不负责 WSR 阈值认证，也不负责模型更新。

实际代码位置：

- `src/algorithm/detector/residual_vim.py`
- `src/algorithm/detector/full_vim.py`
- `src/algorithm/detector/mahalanobis.py`
- `src/algorithm/detector/rmd.py`
- `src/algorithm/detector/knn.py`
- `src/algorithm/detector/openood.py`
- `src/algorithm/detector/selection.py`
- `src/algorithm/detector/score.py`

当前 CLI 支持的检测器：

- `residual_vim`
- `full_vim`
- `mahalanobis`
- `rmd`
- `knn`
- `msp`
- `maxlogit`
- `energy`
- `gen`
- `kl_matching`
- `gradnorm`
- `odin`
- `react`
- `dice`
- `ash`

说明：

- `residual_vim` 是 residual-only ViM，不需要 logits。
- `full_vim` 需要 classifier logits 和 exact affine head parameters。
- `odin`, `react`, `dice`, `ash` 等 head-transforming OpenOOD-style 方法也需要 compatible affine head parameters。
- `msp`, `maxlogit`, `energy`, `gen`, `kl_matching`, `gradnorm` 使用 logits / probabilities。

CLI:

```bash
python -m src.algorithm detect \
  --features b_hidden.npz \
  --records rows.jsonl \
  --classifier classifier.joblib \
  --fit-split training_train \
  --calibration-split training_calibration \
  --detectors residual_vim,mahalanobis,rmd,knn,msp,energy \
  --output detector_scores.jsonl
```

输出是 `detector_scores.jsonl`，每行至少包括 `sample_id`, `query_id`, `split`, `label`, `prediction`, `detector`, `score`。

## 6. WSR 阈值认证

WSR 只处理 bounded loss、detector score 和 calibration indices。它不 import classifier / detector 实现，避免风险认证层反向依赖上游模型细节。

实际代码位置：

- `src/algorithm/wsr/certification.py`
- `src/algorithm/wsr/betting.py`
- `src/algorithm/wsr/bounds.py`
- `src/algorithm/wsr/threshold_grid.py`
- `src/algorithm/wsr/selective_risk.py`

主要能力：

- normalized MAE loss
- quantile threshold grid
- WSR betting log capital
- finite-population mean upper bound
- certified threshold selection

CLI:

```bash
python -m src.algorithm certify-wsr \
  --scores detector_scores.jsonl \
  --predictions predictions.jsonl \
  --detector residual_vim \
  --risk-loss normalized_mae \
  --risk-bound 0.125 \
  --delta 0.05 \
  --output thresholds.json
```

## 7. 逐条 accept/reject

WSR 输出 threshold 后，逐条决策层将 detector score 转成 deployment-time accept/reject。

实际代码位置：

- `src/algorithm/data/decisions.py`
- `src/algorithm/data/thresholds.py`

规则：

- 默认 `score <= threshold` 为 accept。
- 可通过 CLI 的 `--accept-above` 反转方向。
- 输出 `DecisionRow`: `sample_id`, `detector`, `score`, `threshold`, `decision`, `accepted`。

CLI:

```bash
python -m src.algorithm apply-threshold \
  --scores detector_scores.jsonl \
  --thresholds thresholds.json \
  --detector residual_vim \
  --output decisions.jsonl
```

## 8. 窗口级失效确认

窗口级确认把逐条 reject 聚合成 stream/window 级别的 failure signal。当前实现是一个明确的数据契约和基础判定函数；更复杂的 MMD/C2ST/KS 监控逻辑在 `update` 层。

实际代码位置：

- `src/algorithm/data/monitoring.py`
- `src/algorithm/update/drift.py`
- `src/algorithm/update/mmd.py`
- `src/algorithm/update/c2st.py`
- `src/algorithm/update/ks.py`
- `src/algorithm/update/sequential.py`

当前窗口确认 CLI：

```bash
python -m src.algorithm confirm-window \
  --decisions decisions.jsonl \
  --window-size 200 \
  --min-reject-rate 0.25 \
  --output monitoring.json
```

`update/drift.py` 中保留了更完整的 drift monitoring 组件：

- `MMDPermutationTest`
- `BlockAwareC2ST`
- `ScalarKSTest`
- `WindowDriftConfig`

## 9. 模型更新

更新层消费上游输出，不被上游模块反向 import。

实际代码位置：

- `src/algorithm/update/head_update.py`
- `src/algorithm/update/affine_update.py`
- `src/algorithm/update/tent_update.py`
- `src/algorithm/update/gate.py`
- `src/algorithm/update/probe.py`
- `src/algorithm/update/clustering.py`
- `src/algorithm/update/geometry_check.py`

当前 CLI:

```bash
python -m src.algorithm update-adapt \
  --features b_hidden.npz \
  --classifier classifier.joblib \
  --labels probe_labels.jsonl \
  --mode affine \
  --gate-split deployment_gate \
  --output adapted_classifier.joblib
```

说明：

- `HeadAdapter` / affine update 的可复用逻辑已经迁移到 `update`。
- CLI 当前是保守入口：读取 features/classifier/probe labels 并写出 adapted classifier artifact metadata。
- 后续若要做完整在线更新，应在该层继续扩展，而不是让 detector/classifier 反向依赖 update。

## 10. 重新认证

模型更新后，需要重新生成 predictions / detector scores，再跑 WSR。CLI 提供 `recertify-wsr` 作为显式入口，使链路末端和首次认证区分开。

```bash
python -m src.algorithm recertify-wsr \
  --scores detector_scores_after_update.jsonl \
  --predictions predictions_after_update.jsonl \
  --detector residual_vim \
  --risk-loss normalized_mae \
  --risk-bound 0.125 \
  --delta 0.05 \
  --output thresholds_after_update.json
```

## 11. 当前可验证的最小链路

当前代码已经通过 `/tmp` smoke 覆盖以下链路：

```text
train-classifier
-> detect
-> certify-wsr
-> apply-threshold
-> confirm-window
```

该 smoke 覆盖了 logistic classifier，以及 15 个 detector score 输出：

- residual/full ViM
- Mahalanobis/RMD/kNN
- OpenOOD-style MSP/MaxLogit/Energy/GEN/KL Matching/GradNorm/ODIN/ReAct/DICE/ASH

## 12. 不再保留的旧结构

旧的 numbered runner、config 驱动和笔记本不再作为核心代码树保留。它们的可复用逻辑已经迁移到：

- `src/algorithm`
- `src/common`
- `experiments`

根目录 `datasets/` 和 `artifacts/` 是本地运行数据，不属于源代码；不要把它们作为迁移或清理目标。
