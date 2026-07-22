# ViM 与 Mahalanobis：检测器审计、任务适配与最终算法

本文合并并更新以下两份实验记录：

- `ViM与Mahalanobis检测器审计及冻结实验.md`
- `ViM与Mahalanobis融合优化及任务适配分析.md`

目标是回答一个完整问题：**为什么原始 Full ViM 在部分任务上明显输给 Mahalanobis，怎样修正和适配两个检测器，最终应选择哪套可复现的 OOD 检测算法。**

## 目录

1. [问题现象](#1-问题现象)
2. [原始检测器结构](#2-原始检测器结构)
3. [实现审计与原因分析](#3-实现审计与原因分析)
4. [提出的适配与融合算法](#4-提出的适配与融合算法)
5. [实验协议](#5-实验协议)
6. [算法验证结果](#6-算法验证结果)
7. [最终选择的算法](#7-最终选择的算法)
8. [适用边界与最终结论](#8-适用边界与最终结论)

## 1. 问题现象

最初结果中，ASAP-AES 的 residual-only ViM AUROC 接近 1，但 Full ViM 只有约 0.87；ELLIPSE 上 Mahalanobis 也明显强于 Full ViM。这不符合“Full ViM 在 residual 上增加 logits 后应该更强”的直觉，因此需要先判断：

1. Full ViM 是否实现错误；
2. 作文评分 head 是否不适合 ViM；
3. PCA rank、residual 或 logits 融合是否需要适配；
4. Mahalanobis 是否确实更适合作文 prompt shift；
5. 能否将 Mahalanobis、RMD 和 ViM 融合成更稳定的检测器。

最终发现，问题同时包含一个小的实现错误和一个更重要的任务适配问题：

- 旧 Full ViM 的中心使用了 source feature mean，而标准 ViM 应使用分类器 affine origin；这个错误已经修复，但修复只提升不到 0.003 AUROC，不是主要原因。
- 作文 OOD 是“prompt 变了，但评分等级仍合法”的 label-preserving shift。Judge 即使面对 OOD 作文，也可能非常自信地输出 `1..5` 中的合法分数，因此 logits 不一定变弱。
- CLINC150 和 ROSTD 的 OOD 是训练 intent 之外的 semantic OOS，更符合 ViM 的原始假设；类别条件 residual、logits 和 RMD 在这里更有效。

所以不能简单得出“ViM 错了”或“Mahalanobis 永远更强”。真正结论是：**不同 OOD 定义需要不同的几何信号。**

## 2. 原始检测器结构

所有检测器使用同一份冻结的 Judge hidden representation。设 Judge penultimate feature 为 `h(x)`，分类 logits 为 `z(x)`。

### 2.1 Mahalanobis

Mahalanobis 为每个 ID 类别计算中心 `mu_c`，再用 source ID 的协方差消除不同方向的尺度差异：

```text
d_c(x) = (h(x)-mu_c)^T Sigma^{-1} (h(x)-mu_c)
s_M(x) = min_c d_c(x)
```

分数越大，样本越不像任何已知类别。它的优势是能识别“虽然 Judge 仍给出合法标签，但 feature 已远离 source 分布”的样本。

### 2.2 RMD

RMD 可以理解为“相对 Mahalanobis 距离”：先计算样本到类别分布的距离，再扣除到全局背景分布的距离。这样可减少文本长度、写作风格等公共方向的影响，更突出类别相关偏离。它在 CLINC150、ROSTD 这类 intent/OOS 任务上通常较强。

### 2.3 Residual-only ViM

ViM 先在 source feature 上拟合 PCA 主子空间。主子空间表示训练数据最常见的变化方向，正交补 residual 表示样本无法被 source 主结构解释的部分：

```text
r(x) = (I - PP^T)(h(x)-mu_ID)
s_res(x) = ||r(x)||_2
```

其中 `mu_ID` 是 source feature mean。Residual-only ViM 只使用 `s_res`，不使用 logits。它对明显离开 source feature manifold 的样本敏感。

### 2.4 Full ViM

标准 Full ViM 同时使用 residual 和分类 logits。分类器 origin 为：

```text
o = -pinv(W^T)b
```

用 source ID 计算尺度 `alpha`，构造 virtual logit：

```text
v(x) = alpha * ||r(x)||_2
```

再将 `v(x)` 与原分类 logits 一起做 softmax，virtual class 的概率作为 OOD 分数。直观上，residual 越大、已知类别 logits 越弱，样本越可能是 OOD。

问题在于：作文 prompt OOD 仍然可以被高置信映射到合法评分等级，已知类别 logits 不一定减弱。此时 logit 项会抵消 residual 提供的有效 OOD 信号。

## 3. 实现审计与原因分析

### 3.1 审计结果

检查项包括分数方向、`alpha` fit scope、hiddenstate 行对齐、softmax 维度、等级映射、PCA 中心、分类器 origin，以及 residual/full 是否使用同一层、pooling 和 rank。

| 检查项 | 结果 |
|---|---|
| OOD 分数方向 | 正确；取反后 AUROC 明显低于 0.5 |
| `alpha` 数据范围 | 只使用 source training/reference |
| hiddenstate 与 logits 行对齐 | 通过 |
| softmax 维度 | 按 class dimension，行和误差约 `1.19e-7` |
| 作文 head | ELLIPSE 为 9 类，ASAP 为 5 类，不是单值回归 logits |
| 分数等级映射 | ELLIPSE 与 ASAP 各自映射一致 |
| residual/full 表示 | 使用相同 feature、layer、pooling 和 rank |
| ViM origin | 旧实现错误，已改为 classifier affine origin |

单独修复 origin 后：

| Dataset | 修复前独立复算 | 修复后 | 提升 |
|---|---:|---:|---:|
| ELLIPSE | 0.5773 | 0.5799 | +0.0026 |
| ASAP-AES | 0.8652 | 0.8671 | +0.0019 |

因此实现错误存在，但不足以解释 residual 与 Full ViM 的巨大差距。

### 3.2 分类头不是唯一原因

比较 Linear、MLP、Ordinal 和 Regression head 后发现，Judge 的 QWK 提升并不会稳定带来 Full ViM 提升。例如 ASAP 上 Linear QWK 为 0.7059、Full ViM 为 0.9045；MLP QWK 更低，为 0.6052，但 Full ViM 反而达到 0.9992。

这说明 OOD 检测效果取决于整个 penultimate geometry，不只是 Judge 预测精度或 logits 质量。单纯换一个更强分类头不能解决问题。

### 3.3 数据任务才是主要原因

| 数据类型 | OOD 含义 | 最可靠信号 |
|---|---|---|
| ELLIPSE cross-prompt | prompt 改变，评分标签空间不变 | 协方差距离、prompt/document 表示 |
| ASAP cross-set | rubric、任务和体裁变化明显 | residual 或 Mahalanobis，任务接近饱和 |
| CLINC150 OOS | 新 utterance 不属于任何训练 intent | RMD、类别条件 residual、弱 logit fusion |
| ROSTD OOS | 新 utterance 不属于训练 intent | 类别条件 adapted ViM、RMD |

作文任务中，“Judge 还能高置信评分”与“作文来自新 prompt”可以同时成立；intent OOS 中，“不属于任何已知类”与低类别匹配度更一致。这是 Full ViM 跨任务表现不同的根本原因。

## 4. 提出的适配与融合算法

### 4.1 Adapted ViM

不再固定使用标准 Full ViM，而是在 source-only pseudo-OOD 上选择：

- Linear、MLP、Ordinal head；
- layer 23 或 last layer；
- raw、L2-normalized、whitened、class-conditional residual；
- explained-variance rank 或固定 residual dimension；
- residual/logit 权重 `lambda`；
- logits temperature。

冻结结果体现了任务差异：

| Dataset | Residual 形式 | Rank | `lambda / T` | 含义 |
|---|---|---|---|---|
| ELLIPSE | L2-normalized | EV 97% | `1 / 10` | 强降温 logits，但仍不稳定 |
| ASAP-AES | Raw | EV 80% | `0 / 1` | 完全拒绝 logits，退化为 residual-only |
| CLINC150 | Class-conditional | per-class EV 90% | `0.25 / 0.5` | residual 为主，弱 logits 辅助 |
| ROSTD | Class-conditional | per-class EV 90% | `1 / 0.5` | residual 与 logits 均有效 |

### 4.2 强 Mahalanobis 变体

比较了 shared covariance、shrinkage、diagonal、class-balanced 和 RMD。另提出 spectral Mahalanobis：

```text
d_gamma,c(x) = sum_j <h(x)-mu_c, v_j>^2 / (lambda_j + gamma * mean(lambda))
```

其中 `gamma` 控制对低方差方向的正则强度。再比较三种类别聚合：

- nearest-class：取最近类别距离；
- predicted-class：使用 Judge 预测类别的距离；
- posterior-weighted：按 Judge posterior 加权所有类别距离。

posterior-weighted spectral Mahalanobis 将类别几何和 Judge posterior 在距离层结合，比在最终两个标量上硬相加更有结构。

### 4.3 ECDF 三路融合

Mahalanobis、RMD 和 ViM 的原始数值尺度不同，不能直接相加。先用 source ID calibration 将每个分数转换成 ID 分位数：

```text
u_j(x) = F_hat_j,ID(s_j(x))
```

`u_j` 越接近 1，表示该检测器认为样本比绝大多数 source ID 更异常。然后融合：

```text
S(x) = w_M u_M(x) + w_R u_R(x) + w_V u_V(x)
w_M,w_R,w_V >= 0,  w_M+w_R+w_V = 1
```

默认权重候选使用 0.1 步长，共 66 组。权重只在 source-derived pseudo-OOD 上选择，official OOD 不参与选权重。

### 4.4 其他权重算法

为确认粗网格不是偶然最优，还测试了：

| 方法 | 核心思想 |
|---|---|
| Equal Weight | 三个检测器各占 `1/3` |
| RRF | 融合三个检测器相对 ID calibration 的倒数排名 |
| Fine Grid | 将权重步长从 0.1 缩小到 0.02 |
| Reliability Softmax | 根据 `AUROC-gamma*FPR95` 计算平滑权重 |
| Nonnegative Logistic | 三个 ECDF 分数作为输入，学习非负二级模型 |
| Shrinkage-LDA/Fisher | 使用收缩协方差处理检测器相关性 |
| Robust CV Mean-Std | 最大化多折 `MeanAUROC-beta*StdAUROC` |
| Robust CV Maximin | 最大化最差 pseudo-shift AUROC |

这些方法都只使用 source-only pseudo-OOD；多折版本让每个 source prompt/domain/intent 恰好作为 pseudo-OOD 一次。

## 5. 实验协议

- 数据集：ELLIPSE、ASAP-AES、CLINC150、ROSTD；暂不包含 AG News。
- seed：42；复用现有 Qwen3.5-4B hiddenstate，不做 backbone forward。
- 可用表示：layer 23、last layer、masked-mean pooling。
- 基础检测器、PCA、head 只在 source training 上拟合。
- ECDF 只使用 source validation ID calibration。
- pseudo-OOD：从 source prompt、essay set、domain 或 intent 中留出 group。
- 所有配置先写入 `frozen_selection.json`，再计算 official OOD。
- 指标：AUROC、AUPR-OOD、FPR95，并使用配对 bootstrap 检查差值。

正式 OOD test 不参与 head、layer、rank、residual、Mahalanobis family、融合权重、temperature 或正则强度的选择。正式 test oracle 只用于分析互补上限，不作为可部署结果。

## 6. 算法验证结果

### 6.1 单检测器审计结果

下表使用统一 source-fitted PCA/head 管线。单元格为 `AUROC / AUPR-OOD / FPR95`。Strong Mahalanobis 是用于判断方法上限的强实现对照；最终部署使用的 Mahalanobis family 仍由 source-only selection 冻结，不能按正式结果事后选择。

| Dataset | Adapted ViM | Standard Full ViM | Residual-only | Strong Mahalanobis | RMD |
|---|---|---|---|---|---|
| ELLIPSE | 0.5939 / 0.6452 / 0.9269 | 0.5035 / 0.5657 / 0.9615 | 0.5882 / 0.6297 / 0.9192 | **0.6721 / 0.7003 / 0.8115** | 0.5289 / 0.5994 / 0.9462 |
| ASAP-AES | 0.9984 / 0.9996 / 0.0047 | 0.9045 / 0.9786 / 0.8279 | 0.9984 / 0.9996 / 0.0047 | **0.9999 / 1.0000 / 0.0000** | 0.3957 / 0.8249 / 0.9953 |
| CLINC150 | 0.9369 / 0.7498 / 0.2304 | 0.9278 / 0.6998 / 0.2571 | 0.6244 / 0.2761 / 0.8993 | 0.8854 / 0.5966 / 0.3924 | **0.9461 / 0.7690 / 0.2069** |
| ROSTD | **0.9859 / 0.9582 / 0.0605** | 0.9668 / 0.9067 / 0.1313 | 0.8779 / 0.7106 / 0.4390 | 0.9214 / 0.8033 / 0.3153 | 0.9760 / 0.9292 / 0.0952 |

主要结论：

- origin 修复和 ViM 适配有效，但没有形成统一最强的单检测器；
- ELLIPSE 仍明显偏向 Mahalanobis；
- ASAP 的 residual 与 Mahalanobis 都接近饱和，标准 Full ViM 的 logits 明显有害；
- CLINC150 最强为 RMD，ROSTD 最强为 class-conditional adapted ViM。

### 6.2 最终三路融合结果

| Method | Mean AUROC | Mean AUPR-OOD | Mean FPR95 |
|---|---:|---:|---:|
| Equal Weight | 0.8471 | 0.8388 | 0.3712 |
| RRF Fusion | 0.8824 | 0.8497 | 0.3130 |
| Best Single Detector | 0.8999 | 0.8553 | 0.2692 |
| **ECDF + 0.1 Grid Weight** | **0.9030** | **0.8671** | **0.2618** |

逐数据集：

| Dataset | 最佳单检测器 AUROC | ECDF Grid AUROC | ECDF Grid AUPR | ECDF Grid FPR95 |
|---|---:|---:|---:|---:|
| ELLIPSE | 0.6681 | **0.6747** | 0.7172 | 0.7962 |
| ASAP-AES | **0.9997** | 0.9994 | 0.9998 | 0.0047 |
| CLINC150 | 0.9461 | **0.9495** | 0.7862 | 0.1960 |
| ROSTD | 0.9859 | **0.9883** | 0.9654 | 0.0502 |

融合相对最佳单检测器的平均 AUROC 提升为 `+0.0030`，平均 FPR95 改善约 `0.0075`。提升很小，因此应表述为稳定的软路由收益，不能声称显著创造了新的统一 SOTA。

### 6.3 更复杂权重算法没有稳定胜出

| 权重算法 | Mean AUROC | Mean AUPR | Mean FPR95 | 判断 |
|---|---:|---:|---:|---|
| Coarse Grid 0.1 | 0.9030 | 0.8671 | 0.2618 | 默认方案 |
| Fine Grid 0.02 | 0.9033 | 0.8670 | 0.2619 | 点估计略高，但提升不显著 |
| Shrinkage Fisher | 0.9031 | 0.8698 | 0.2615 | AUPR/FPR 略好，AUROC 无显著提升 |
| Reliability Softmax | 0.9026 | 0.8672 | 0.2757 | 没有超过粗网格 |
| Nonnegative Logistic | 0.8895 | 0.8588 | 0.2861 | 单切分过拟合明显 |
| Robust CV Maximin | 0.8983 | 0.8587 | 0.2686 | 多折更保守，但正式均值下降 |
| CV Shrinkage-LDA | 0.8892 | 0.8585 | 0.2851 | covariance shift 下不稳定 |

Fine Grid 相对 Coarse Grid 只提升 `+0.00033`，95% CI `[-0.00088, 0.00164]`；Shrinkage Fisher 相对 Coarse Grid 只提升 `+0.00016`，95% CI `[-0.00252, 0.00239]`。两者都没有显著优势。

多折 Robust CV 在 CLINC150 有帮助，但在 ELLIPSE 上更稳定地选择了错误方向。这说明当前瓶颈不是权重优化器，而是 source prompt 间 pseudo-shift 是否能代表正式 held-out prompt shift。

## 7. 最终选择的算法

最终选择：**source-ID ECDF 校准 + dataset-aware 0.1 粗网格三路融合**。

它可以理解为一个软 task-aware router：不强制三个检测器平均工作，而是根据 source-only pseudo-OOD 决定当前任务主要相信 Mahalanobis、RMD 还是 adapted ViM。

### 7.1 三个输入分数

1. `s_M`：该数据集 source-only 选出的强 Mahalanobis family；
2. `s_R`：RMD 分数；
3. `s_V`：source-only 冻结的 adapted ViM 分数。

对应基础检测器策略：

| Dataset | Mahalanobis family | Adapted ViM 重点 |
|---|---|---|
| ELLIPSE | Class-balanced Mahalanobis | L2-normalized residual，高温 logits |
| ASAP-AES | Shrinkage Mahalanobis | Raw residual，`lambda=0` |
| CLINC150 | Posterior-weighted spectral Mahalanobis | Class-conditional residual，弱 logits |
| ROSTD | Posterior-weighted spectral Mahalanobis | Class-conditional residual |

### 7.2 拟合和选权重

对每个数据集独立执行：

```text
1. 从 source groups 中留出 pseudo-OOD group。
2. 用其余 source training 拟合 PCA、head、Mahalanobis、RMD 和 adapted ViM。
3. 用独立 source validation ID 拟合三个 ECDF。
4. 将 pseudo ID/OOD 分数转换为 u_M、u_R、u_V。
5. 遍历 66 组非负 0.1-grid 权重。
6. 先选 pseudo AUROC 最高者；并列时比较 AUPR、FPR95，再偏向简单均衡权重。
7. 将 detector 配置、ECDF 规则和权重写入 frozen_selection.json。
8. 在全部 source training 上重拟合基础检测器，official OOD 只评测一次。
```

### 7.3 冻结权重

| Dataset | `w_M` | `w_R` | `w_V` | 实际含义 |
|---|---:|---:|---:|---|
| ELLIPSE | 0.5 | 0.1 | 0.4 | Mahalanobis 为主，保留 ViM 互补信号 |
| ASAP-AES | 0.5 | 0.0 | 0.5 | Mahalanobis 与 residual ViM 一致性融合 |
| CLINC150 | 0.0 | 0.9 | 0.1 | 基本由 RMD 主导 |
| ROSTD | 0.0 | 0.1 | 0.9 | 基本由 adapted ViM 主导 |

这些权重不是从 official OOD test 反推出来的，也不是一组全局固定权重。数据集/任务类型在部署前已知，因此这种 dataset-aware policy 不构成 test-time selection。

### 7.4 推理过程

对新样本 `x`：

```text
s_M, s_R, s_V = three_detectors(x)
u_M = ECDF_M_ID(s_M)
u_R = ECDF_R_ID(s_R)
u_V = ECDF_V_ID(s_V)
S   = w_M*u_M + w_R*u_R + w_V*u_V
```

`S` 越大越像 OOD。部署阈值只能由 source ID calibration quantile 确定，不能查看正式 OOD 后再移动阈值。

### 7.5 为什么选择它

- 它在当前四数据集冻结实验中取得最高的实用综合结果：Mean AUROC `0.9030`、Mean AUPR `0.8671`、Mean FPR95 `0.2618`。
- 它比最佳单检测器略好，同时不会像 Equal Weight 那样强迫弱检测器参与。
- 它保留 ECDF 的可解释性：每个输入都是“相对 source ID 有多异常”的分位数。
- 只有 66 组候选，不需要训练复杂二级模型，过拟合面较小。
- Fine Grid、Logistic、Reliability Softmax、Fisher/LDA 和 Robust CV 都没有给出稳定且显著更好的四数据集结果。

选择依据不是“0.9030 显著高于所有方法”，而是：**结果处于最佳档、FPR95 最低档、实现简单、冻结协议清楚，并且能自然表达不同任务应依赖不同检测器。**

## 8. 适用边界与最终结论

最终算法不是一个脱离任务定义的万能 detector。推荐解释为：

```text
label-preserving prompt/background shift
    -> covariance/residual detector 为主

label-expanding semantic OOS
    -> RMD/class-conditional adapted ViM 为主

ECDF coarse-grid fusion
    -> 在 source-only 条件下把上述选择平滑化
```

必须保留以下边界：

1. ELLIPSE 的 source prompt pseudo-OOD 与正式 held-out prompt 并不完全同分布，多折 CV 也不能解决这种代表性问题；后续更值得加入 A-space prompt/document 表示，而不是继续细调三个权重。
2. ASAP 只有两个 source prompts，且正式检测接近饱和，不能把小数点后的差异解释成算法突破。
3. CLINC150、ROSTD 支持 class-conditional ViM/RMD 的有效性，但目前仍主要是 seed 42 结果。
4. 三路融合相对最佳单检测器提升很小，不能宣称统计显著统一胜出。
5. 当前实验未包含 AG News，也没有测试缓存中不存在的 pooling 和中间层组合。

最终可使用的论文表述是：

> Full ViM 的弱点主要来自任务适配，而不是分数方向或数据对齐错误。修正 classifier origin 并对 residual、rank 和 logits 融合做 source-only 适配后，ViM 在 semantic OOS 上表现很强，但作文 cross-prompt shift 仍更依赖协方差距离。最终系统使用 source-ID ECDF 将 Mahalanobis、RMD 和 adapted ViM 统一到同一尺度，再通过 source-only 0.1 粗网格选择 dataset-aware 权重。该方法在四个非 AG News 数据集上取得 0.9030 平均 AUROC、0.8671 平均 AUPR-OOD 和 0.2618 平均 FPR95；其收益应解释为可复现的软任务路由，而不是一个显著超过所有单检测器的统一新 SOTA。

主要可复现产物：

- 检测器审计：`artifacts/docs_experiments/vim_mahalanobis_study_seed42/`
- Mahalanobis/ViM 结构融合：`artifacts/docs_experiments/vim_mahalanobis_fusion_seed42/`
- 最终三路 ECDF 融合：`artifacts/docs_experiments/three_way_ecdf_fusion_seed42/`
- 权重算法对比：`artifacts/docs_experiments/fusion_weighting_study_seed42/`
- 多折稳健性审计：`artifacts/docs_experiments/robust_cv_fusion_seed42/`

本系列实验未重新抽取 hiddenstate，backbone forward pass 为 0，API 调用为 0，GPU 使用为 0。
