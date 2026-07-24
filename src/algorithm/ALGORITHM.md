# Algorithm Package

This package keeps the reusable algorithm stack separate from experiment
drivers and generated artifacts.

## Layers

- `hidden_state`: frozen model hidden-state extraction, layer selection, views,
  pooling, score-token/prelogit capture, and cache metadata.
- `classifier`: frozen-feature Judge heads and classifier output contracts.
- `detector`: OOD and novelty scores over classifier or document features.
- `confidence`: post-detector uncertainty transforms and score fusion.
- `wsr`: finite-population WSR threshold certification and selective risk.
- `update`: drift monitoring, probes, gates, and adaptation mechanisms.
- `data`: stable JSON/JSONL/NPZ output, threshold, decision, monitoring, and
  metadata schemas.

## Main Chain

`src.algorithm.data.flow.ALGORITHM_CHAIN` preserves the intended flow:

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

Classifier implementations include linear softmax, CORAL ordinal, ridge, and
logistic heads. Detector implementations include residual ViM, full ViM,
Mahalanobis, RMD, kNN, and OpenOOD-style post-hoc methods.

`src.common` contains only reusable utilities and does not import this package.
`experiments` may import `src.algorithm` and `src.common`; the reverse direction
is intentionally disallowed.
