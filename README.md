# Dual-Space Prototypical Probing for OOD-Robust LLM Evaluation

This repository implements a layered pipeline for evaluating LLM judges under
distribution shift. The core idea is to keep two representation spaces explicit:

- **A-space**: hidden-state representations of the original input document.
- **B-space**: hidden-state representations of the frozen judge prompt / scoring context.

The reusable algorithm code lives under `src/algorithm`; experiment-specific
dataset preparation, reporting, and study runners live under `experiments`.

## Main Chain

The intended end-to-end flow is fixed as one chain:

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

The code-level contract for this chain is
`src.algorithm.data.flow.ALGORITHM_CHAIN`. The detailed algorithm flow is in
[`docs/ALGORITHM_FLOW.md`](docs/ALGORITHM_FLOW.md).

## Repository Layout

```text
src/
  algorithm/
    hidden_state/   A/B hidden-state extraction, layer selection, pooling, views
    classifier/     frozen-feature scoring heads and prediction contracts
    detector/       OOD / error-risk detectors over classifier representations
    confidence/     ECDF confidence, logit uncertainty, disagreement, fusion
    wsr/            WSR betting bounds, threshold grids, selective-risk certs
    update/         drift monitoring, probes, gates, and adaptation helpers
    data/           stable prediction, score, threshold, decision, monitoring schemas
  common/           IO, records, feature stores, metrics, Qwen loading, stats
experiments/
  flask/            FLASK-specific data, prompts, splits, LoRA/reporting helpers
  asap/             ASAP-specific data, prompts, splits, baselines
  benchmark/        benchmark-ground-truth data and reporting helpers
  studies/          detector suite, fusion, RQ4/RQ5, table builders
```

`src/algorithm/data` is source code. Root-level `artifacts/` is reserved for
generated experiment outputs and is ignored by Git.

## Dependency Rules

- `src.common` does not import `src.algorithm` or `experiments`.
- `hidden_state` only depends on `src.common` and model-loading utilities.
- `classifier` consumes hidden-state caches and emits prediction contracts.
- `detector` consumes classifier outputs / feature matrices and emits scores.
- `confidence` and `wsr` consume scores and predictions.
- `update` consumes upstream outputs for monitoring and adaptation.
- `experiments` may import `src.common` and `src.algorithm`; core code does not import experiments.

## Supported Methods

Classifier methods exposed by the CLI:

- `linear` / `linear_softmax`
- `coral`
- `ridge`
- `logistic`

Detector methods exposed by the CLI:

- `residual_vim`
- `full_vim`
- `mahalanobis`
- `rmd`
- `knn`
- OpenOOD-style: `msp`, `maxlogit`, `energy`, `gen`, `kl_matching`, `gradnorm`, `odin`, `react`, `dice`, `ash`

Some OpenOOD-style methods require classifier logits and exact affine head
parameters; the CLI validates this at runtime.

## CLI Overview

```bash
python -m src.algorithm --help
python -m src.algorithm extract --help
python -m src.algorithm train-classifier --help
python -m src.algorithm detect --help
python -m src.algorithm certify-wsr --help
python -m src.algorithm apply-threshold --help
python -m src.algorithm confirm-window --help
python -m src.algorithm update-monitor --help
python -m src.algorithm update-adapt --help
python -m src.algorithm recertify-wsr --help
python -m experiments.cli --help
```

### 1. Extract Hidden States

```bash
python -m src.algorithm extract \
  --records rows.jsonl \
  --space a \
  --view input_document_masked_mean \
  --layers -10 -1 \
  --pooling masked_mean \
  --model-path /path/to/qwen \
  --output a_hidden.npz
```

```bash
python -m src.algorithm extract \
  --records rows.jsonl \
  --space b \
  --view pre_score_token \
  --layers -1 \
  --model-path /path/to/qwen \
  --output b_hidden.npz
```

### 2. Train Scoring Classifier

```bash
python -m src.algorithm train-classifier \
  --features b_hidden.npz \
  --records rows.jsonl \
  --train-split training_train \
  --val-split training_validation \
  --classifier coral \
  --output classifier.joblib
```

### 3. Score Error Risk / OOD

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

### 4. Certify WSR Threshold

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

### 5. Apply Accept / Reject

```bash
python -m src.algorithm apply-threshold \
  --scores detector_scores.jsonl \
  --thresholds thresholds.json \
  --detector residual_vim \
  --output decisions.jsonl
```

### 6. Confirm Window-Level Failure

```bash
python -m src.algorithm confirm-window \
  --decisions decisions.jsonl \
  --window-size 200 \
  --min-reject-rate 0.25 \
  --output monitoring.json
```

### 7. Update and Re-Certify

```bash
python -m src.algorithm update-adapt \
  --features b_hidden.npz \
  --classifier classifier.joblib \
  --labels probe_labels.jsonl \
  --mode affine \
  --gate-split deployment_gate \
  --output adapted_classifier.joblib

python -m src.algorithm recertify-wsr \
  --scores detector_scores_after_update.jsonl \
  --predictions predictions_after_update.jsonl \
  --detector residual_vim \
  --risk-loss normalized_mae \
  --risk-bound 0.125 \
  --delta 0.05 \
  --output thresholds_after_update.json
```

## Data Policy

Keep generated data, hidden-state caches, model checkpoints, logs, and experiment
outputs outside Git. The root-level `datasets/` and `artifacts/` paths are local
runtime data locations and should not be treated as source code.

## Verification

This repository intentionally uses lightweight verification for normal
development. Do not add regression-test scaffolding unless explicitly requested.

Recommended checks:

```bash
python -m compileall src experiments
python -m src.algorithm --help
python -m experiments.cli --help
```

For CLI changes, also run each subcommand with `--help` and, when a small local
cache is available, run a smoke chain:

```text
train-classifier -> detect -> certify-wsr -> apply-threshold -> confirm-window
```
