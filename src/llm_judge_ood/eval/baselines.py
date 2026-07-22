"""Reproducible evaluation-only baselines for the document OOD protocol.

The production pipeline owns the detector/lifecycle decision.  This module
keeps the paper baselines in a separate, label-free evaluation surface:
representations are fitted on source features, target labels are never read by
the detector, and every sample-size point is repeated with explicit seeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import SparseRandomProjection
from sklearn.metrics import roc_auc_score

from src.llm_judge_ood.lifecycle.drift import MMDPermutationTest, WindowDriftConfig


DETECTION_BASELINES = ("NoRed", "PCA", "SRP", "TAE", "UAE", "BBSDs", "BBSDh", "Classif")
OPERATIONAL_BASELINES = ("no-monitor", "always-retrain", "confidence-only", "final-accuracy-only")


@dataclass(frozen=True)
class DetectionBaselineConfig:
    sample_sizes: tuple[int, ...] = (10, 20, 50, 100, 200, 500, 1000, 10000)
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    pca_dim: int = 128
    srp_dim: int = 128
    tae_dim: int = 128
    uae_dim: int = 128
    autoencoder_hidden_dim: int = 256
    autoencoder_epochs: int = 100
    autoencoder_batch_size: int = 256
    autoencoder_learning_rate: float = 1e-3
    autoencoder_weight_decay: float = 1e-5
    mmd_permutations: int = 1000
    alpha: float = 0.05
    classifier_max_iter: int = 500

    def __post_init__(self) -> None:
        if not self.sample_sizes or any(int(size) < 2 for size in self.sample_sizes):
            raise ValueError("sample_sizes must contain values >= 2")
        if len(set(int(seed) for seed in self.seeds)) != len(self.seeds):
            raise ValueError("detection baseline seeds must be unique")
        if int(self.mmd_permutations) < 19:
            raise ValueError("mmd_permutations must be at least 19")
        if min(int(self.pca_dim), int(self.srp_dim), int(self.tae_dim), int(self.uae_dim)) < 1:
            raise ValueError("detection baseline representation dimensions must be positive")
        if int(self.autoencoder_hidden_dim) < 1 or int(self.autoencoder_epochs) < 1:
            raise ValueError("autoencoder hidden dimension and epochs must be positive")
        if int(self.autoencoder_batch_size) < 1 or float(self.autoencoder_learning_rate) <= 0.0:
            raise ValueError("autoencoder batch size and learning rate must be positive")
        if float(self.autoencoder_weight_decay) < 0.0:
            raise ValueError("autoencoder weight decay must be non-negative")
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError("alpha must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_detection_baselines(
    source_features: np.ndarray,
    target_features: np.ndarray,
    *,
    source_logits: np.ndarray | None = None,
    target_logits: np.ndarray | None = None,
    config: DetectionBaselineConfig | None = None,
    baselines: Sequence[str] = DETECTION_BASELINES,
) -> dict[str, Any]:
    """Report detection rate as a function of target sample size.

    ``source_features`` and ``target_features`` may be ``[N,D]`` or
    ``[N,L,D]``.  All transforms are fitted once on source rows.  BBSDs/h and
    Classif uses the input representation directly. BBSDs/h use Judge logits
    and therefore fail explicitly when logits are omitted.
    """

    cfg = config or DetectionBaselineConfig()
    source = _matrix(source_features, "source_features")
    target = _matrix(target_features, "target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source_features and target_features must have the same dimension")
    requested = tuple(str(name) for name in baselines)
    unknown = sorted(set(requested) - set(DETECTION_BASELINES))
    if unknown:
        raise ValueError(f"unknown detection baselines: {unknown}")
    needs_logits = bool(set(requested) & {"BBSDs", "BBSDh"})
    if needs_logits and (source_logits is None or target_logits is None):
        raise ValueError("BBSDs and BBSDh require source_logits and target_logits")
    source_views, target_views, fit_metadata = _fit_views(source, target, cfg, requested)
    if needs_logits:
        source_logit_values = _matrix(source_logits, "source_logits")
        target_logit_values = _matrix(target_logits, "target_logits")
        if source_logit_values.shape[0] != len(source) or target_logit_values.shape[0] != len(target):
            raise ValueError("source_logits and target_logits must align with feature rows")
        source_probabilities = softmax(source_logit_values, axis=1)
        target_probabilities = softmax(target_logit_values, axis=1)
        if "BBSDs" in requested:
            source_views["BBSDs"], target_views["BBSDs"] = source_probabilities, target_probabilities
        if "BBSDh" in requested:
            source_views["BBSDh"] = np.eye(source_probabilities.shape[1], dtype=np.float64)[np.argmax(source_probabilities, axis=1)]
            target_views["BBSDh"] = np.eye(target_probabilities.shape[1], dtype=np.float64)[np.argmax(target_probabilities, axis=1)]
        fit_metadata["logits"] = {
            "source_rows": int(len(source_logit_values)),
            "target_rows": int(len(target_logit_values)),
            "class_count": int(source_logit_values.shape[1]),
            "probability_transform": "softmax",
        }
    rows: list[dict[str, Any]] = []
    for baseline in requested:
        source_view = source_views[baseline]
        target_view = target_views[baseline]
        for sample_size in cfg.sample_sizes:
            n = int(sample_size)
            if n > len(target):
                continue
            if baseline == "Classif" and n > len(source):
                continue
            for seed in cfg.seeds:
                rng = np.random.default_rng(int(seed) + _stable_int(baseline))
                selected = rng.choice(len(target), size=n, replace=False)
                if baseline == "Classif":
                    metric = _classifier_test(source_view, target_view[selected], cfg, seed=int(seed))
                elif baseline == "BBSDs":
                    metric = _bbsds_test(source_view, target_view[selected])
                elif baseline == "BBSDh":
                    metric = _bbsdh_test(source_view, target_view[selected])
                else:
                    metric = _mmd_test(source_view, target_view[selected], cfg, seed=int(seed))
                rows.append(
                    {
                        "baseline": baseline,
                        "sample_size": n,
                        "seed": int(seed),
                        "detected": bool(float(metric["p_value"]) <= float(cfg.alpha)),
                        **metric,
                    }
                )
    aggregate = _aggregate_detection_rows(rows)
    return {
        "artifact_type": "llm_judge_ood_detection_baseline_sweep",
        "protocol": {
            "representation_fit_scope": "source_only",
            "classif_fit_scope": "balanced_source_target_domain_membership_train_split",
            "target_task_labels_used": False,
            "metric": "representation_specific_two_sample_test",
            "baseline_tests": {
                "BBSDs": "per_probability_dimension_KS_with_Bonferroni",
                "BBSDh": "Pearson_chi_square_on_predicted_class_counts",
                "Classif": "independent_holdout_domain_classifier_label_permutation",
                "default": "block_permutation_RBF_MMD",
            },
            "sample_size_axis": "target_rows",
            "seeds": [int(seed) for seed in cfg.seeds],
            "baselines": list(requested),
        },
        "config": cfg.to_dict(),
        "fit_metadata": fit_metadata,
        "rows": rows,
        "aggregate": aggregate,
    }


def evaluate_operational_baselines(
    *,
    ood_scores: Sequence[float],
    harmful_windows: Sequence[bool],
    source_ood_scores: Sequence[float] | None = None,
    confidence: Sequence[float] | None = None,
    final_accuracy: Sequence[float] | None = None,
    source_confidence: Sequence[float] | None = None,
    source_accuracy: Sequence[float] | None = None,
    confidence_quantile: float = 0.10,
    accuracy_drop_tolerance: float = 0.05,
) -> dict[str, Any]:
    """Compare deployable strategy baselines on a fixed window schedule.

    ``harmful_windows`` is evaluation truth only.  The strategy rules consume
    the supplied unlabeled score/confidence signals, or delayed final accuracy
    when that baseline is selected.  Pass ``source_ood_scores`` when available
    so the OOD-only threshold is calibrated outside the deployment schedule.
    """

    scores = np.asarray(ood_scores, dtype=np.float64)
    truth = np.asarray(harmful_windows, dtype=bool)
    if scores.ndim != 1 or truth.shape != scores.shape or scores.size == 0:
        raise ValueError("ood_scores and harmful_windows must be aligned non-empty vectors")
    n = int(scores.size)
    confidence_values = None if confidence is None else _aligned_vector(confidence, n, "confidence")
    accuracy_values = None if final_accuracy is None else _aligned_vector(final_accuracy, n, "final_accuracy")
    source_conf = (
        np.asarray(source_confidence, dtype=np.float64)
        if source_confidence is not None
        else np.asarray([], dtype=np.float64)
    )
    source_acc = (
        np.asarray(source_accuracy, dtype=np.float64)
        if source_accuracy is not None
        else np.asarray([], dtype=np.float64)
    )
    source_scores = (
        _vector(source_ood_scores, "source_ood_scores")
        if source_ood_scores is not None
        else scores[: max(1, n // 4)]
    )
    score_threshold = float(np.quantile(source_scores, 0.95))
    confidence_threshold = (
        float(np.quantile(source_conf, float(confidence_quantile)))
        if source_conf.size
        else float(np.quantile(confidence_values, float(confidence_quantile)))
        if confidence_values is not None
        else float("nan")
    )
    baseline_accuracy = float(source_acc.mean()) if source_acc.size else float(np.nanmean(accuracy_values)) if accuracy_values is not None else float("nan")
    actions: dict[str, np.ndarray] = {
        "no-monitor": np.zeros(n, dtype=bool),
        "always-retrain": np.ones(n, dtype=bool),
        "confidence-only": (
            np.isfinite(confidence_values) & (confidence_values <= confidence_threshold)
            if confidence_values is not None and np.isfinite(confidence_threshold)
            else np.zeros(n, dtype=bool)
        ),
        "final-accuracy-only": (
            np.isfinite(accuracy_values) & (accuracy_values < baseline_accuracy - float(accuracy_drop_tolerance))
            if accuracy_values is not None and np.isfinite(baseline_accuracy)
            else np.zeros(n, dtype=bool)
        ),
    }
    # OOD threshold is retained as a useful diagnostic alongside the named
    # strategy baselines, but is not presented as the proposed policy.
    actions["ood-score-only"] = scores >= score_threshold
    methods = [_summarize_operational(name, action, truth) for name, action in actions.items()]
    return {
        "artifact_type": "llm_judge_ood_operational_baseline_comparison",
        "protocol": {
            "truth_used_only_for_evaluation": True,
            "window_count": n,
            "ood_score_threshold": score_threshold,
            "ood_score_threshold_fit_scope": "source_ood_scores" if source_ood_scores is not None else "initial_schedule_proxy",
            "confidence_threshold": confidence_threshold,
            "source_accuracy_reference": baseline_accuracy,
            "action_is_update_or_alarm": True,
            "label_cost_proxy": "one unit per retrain/final-accuracy action; confidence and OOD-only actions are unlabeled",
        },
        "methods": methods,
    }


def evaluate_label_cost_curve(
    *,
    judge_predictions: Sequence[float],
    labels: Sequence[float],
    rater_scores: Sequence[Sequence[float]],
    reference_excess_human_error: float,
    budgets: Sequence[int] = (2, 5, 10, 20),
    confidence: Sequence[float] | None = None,
    seeds: Sequence[int] = (42, 43, 44, 45, 46),
) -> dict[str, Any]:
    """Report harmfulness-estimation error versus human-label budget.

    The target quantity is the paired excess-human-error delta.  Random and
    confidence-prioritized selection are compared with a full-label upper
    bound.  Selection order never changes the estimator itself.
    """

    predictions = _vector(judge_predictions, "judge_predictions")
    truth = _vector(labels, "labels")
    if predictions.shape != truth.shape:
        raise ValueError("judge_predictions and labels must be aligned")
    raters = np.asarray(rater_scores, dtype=object)
    if len(raters) != len(predictions):
        raise ValueError("rater_scores must align with judge_predictions")
    g = np.asarray(
        [
            abs(float(prediction) - float(label))
            - abs(float(np.asarray(pair, dtype=np.float64)[0]) - float(np.asarray(pair, dtype=np.float64)[1]))
            if len(pair) >= 2
            else np.nan
            for prediction, label, pair in zip(predictions, truth, raters, strict=True)
        ],
        dtype=np.float64,
    )
    valid = np.isfinite(g)
    if not valid.any():
        raise ValueError("at least one row needs two finite rater scores")
    full_delta = float(np.nanmean(g) - float(reference_excess_human_error))
    confidence_values = None if confidence is None else _aligned_vector(confidence, len(predictions), "confidence")
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        requested = int(budget)
        if requested < 1:
            raise ValueError("label budgets must be positive")
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            for method in ("random", "confidence-only", "full-label"):
                if method != "random" and int(seed) != int(seeds[0]):
                    continue
                if method == "full-label":
                    selected = np.flatnonzero(valid)
                elif method == "confidence-only" and confidence_values is not None:
                    selected = np.flatnonzero(valid)[np.argsort(confidence_values[valid], kind="stable")]
                else:
                    valid_indices = np.flatnonzero(valid)
                    selected = rng.permutation(valid_indices)
                if method != "full-label":
                    selected = selected[: min(requested, selected.size)]
                selected = np.asarray(selected, dtype=int)
                estimate = float(np.mean(g[selected]) - float(reference_excess_human_error)) if selected.size else float("nan")
                rows.append(
                    {
                        "method": method,
                        "budget": requested,
                        "seed": int(seed),
                        "labels_used": int(selected.size),
                        "estimate": estimate,
                        "true_full_delta": full_delta,
                        "absolute_estimation_error": abs(estimate - full_delta) if np.isfinite(estimate) else float("nan"),
                    }
                )
    return {
        "artifact_type": "llm_judge_ood_harmfulness_label_cost_curve",
        "protocol": {
            "estimator": "mean(|y_pred-y_true|-|rater1-rater2|)-reference_excess_human_error",
            "target_labels_used_only_in_evaluation": True,
            "selection_methods": ["random", "confidence-only", "full-label"],
            "reference_excess_human_error": float(reference_excess_human_error),
        },
        "rows": rows,
        "aggregate": _aggregate_cost_rows(rows),
    }


def _fit_views(source: np.ndarray, target: np.ndarray, cfg: DetectionBaselineConfig, requested: Sequence[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    source_views: dict[str, np.ndarray] = {}
    target_views: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    if "NoRed" in requested:
        source_views["NoRed"], target_views["NoRed"] = source, target
        metadata["NoRed"] = {
            "representation": "caller_supplied_unreduced_features",
            "input_shape": [int(source.shape[0]), int(source.shape[1])],
            "input_semantics": "must_be_recorded_by_caller_in_experiment_manifest",
            "fit_rows": int(len(source)),
        }
    def add_projection(name: str, estimator: Any, requested_dim: int, fit_scope: str) -> None:
        estimator.fit(source)
        source_views[name] = np.asarray(estimator.transform(source), dtype=np.float32)
        target_views[name] = np.asarray(estimator.transform(target), dtype=np.float32)
        metadata[name] = {
            "representation": fit_scope,
            "fit_rows": int(len(source)),
            "requested_dim": int(requested_dim),
            "effective_dim": int(source_views[name].shape[1]),
        }
    if "PCA" in requested:
        add_projection("PCA", PCA(n_components=min(int(cfg.pca_dim), source.shape[0], source.shape[1]), random_state=42), int(cfg.pca_dim), "source_fitted_pca")
    if "SRP" in requested:
        add_projection("SRP", SparseRandomProjection(n_components=min(int(cfg.srp_dim), source.shape[1]), density="auto", random_state=42), int(cfg.srp_dim), "source_fitted_sparse_random_projection")
    if "TAE" in requested:
        tae = _AutoencoderProjection(
            bottleneck_dim=int(cfg.tae_dim),
            hidden_dim=int(cfg.autoencoder_hidden_dim),
            trained=True,
            epochs=int(cfg.autoencoder_epochs),
            batch_size=int(cfg.autoencoder_batch_size),
            learning_rate=float(cfg.autoencoder_learning_rate),
            weight_decay=float(cfg.autoencoder_weight_decay),
            seed=42,
        ).fit(source)
        source_views["TAE"] = tae.transform(source)
        target_views["TAE"] = tae.transform(target)
        metadata["TAE"] = tae.to_metadata()
    if "UAE" in requested:
        uae = _AutoencoderProjection(
            bottleneck_dim=int(cfg.uae_dim),
            hidden_dim=int(cfg.autoencoder_hidden_dim),
            trained=False,
            epochs=int(cfg.autoencoder_epochs),
            batch_size=int(cfg.autoencoder_batch_size),
            learning_rate=float(cfg.autoencoder_learning_rate),
            weight_decay=float(cfg.autoencoder_weight_decay),
            seed=42,
        ).fit(source)
        source_views["UAE"] = uae.transform(source)
        target_views["UAE"] = uae.transform(target)
        metadata["UAE"] = uae.to_metadata()
    if "Classif" in requested:
        source_views["Classif"], target_views["Classif"] = source, target
        metadata["Classif"] = {
            "representation": "raw_input_features_for_domain_classifier",
            "fit_scope": "balanced_independent_train_holdout_split_per_repetition",
            "source_transform_fit_rows": 0,
        }
    if {"BBSDs", "BBSDh"} & set(requested):
        # Logits are attached after the representation views are built by the
        # caller; placeholders make the metadata explicit here.
        metadata["logit_views"] = {"fit_scope": "source_only_head_outputs"}
    return source_views, target_views, metadata


class _AutoencoderProjection:
    """Source-standardized nonlinear encoder for the TAE/UAE baselines."""

    def __init__(
        self,
        *,
        bottleneck_dim: int,
        hidden_dim: int,
        trained: bool,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        self.bottleneck_dim = int(bottleneck_dim)
        self.hidden_dim = int(hidden_dim)
        self.trained = bool(trained)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.seed = int(seed)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.encoder_: nn.Sequential | None = None
        self.initial_loss_: float | None = None
        self.final_loss_: float | None = None
        self.fit_rows_: int = 0
        self.input_dim_: int = 0
        self.output_dim_: int = 0

    def fit(self, source: np.ndarray) -> "_AutoencoderProjection":
        matrix = np.asarray(source, dtype=np.float32)
        if matrix.ndim != 2 or len(matrix) < 2 or not np.isfinite(matrix).all():
            raise ValueError("autoencoder baseline requires finite source [N,D] features")
        self.fit_rows_ = int(len(matrix))
        self.input_dim_ = int(matrix.shape[1])
        self.output_dim_ = min(int(self.bottleneck_dim), int(self.input_dim_))
        effective_hidden = max(
            self.output_dim_,
            min(int(self.hidden_dim), max(self.input_dim_, self.output_dim_ * 2)),
        )
        self.mean_ = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        self.scale_ = np.maximum(scale, 1e-6)
        standardized = (matrix - self.mean_) / self.scale_
        torch.manual_seed(int(self.seed))
        encoder = nn.Sequential(
            nn.Linear(self.input_dim_, effective_hidden),
            nn.ReLU(),
            nn.Linear(effective_hidden, self.output_dim_),
        )
        decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.output_dim_, effective_hidden),
            nn.ReLU(),
            nn.Linear(effective_hidden, self.input_dim_),
        )
        values = torch.as_tensor(standardized, dtype=torch.float32)
        with torch.no_grad():
            self.initial_loss_ = float(nn.functional.mse_loss(decoder(encoder(values)), values))
        if self.trained:
            parameters = [*encoder.parameters(), *decoder.parameters()]
            optimizer = torch.optim.AdamW(
                parameters,
                lr=float(self.learning_rate),
                weight_decay=float(self.weight_decay),
            )
            rng = np.random.default_rng(int(self.seed))
            for _ in range(int(self.epochs)):
                for start in range(0, len(values), int(self.batch_size)):
                    order = rng.permutation(len(values)) if start == 0 else order
                    batch = torch.as_tensor(
                        order[start : start + int(self.batch_size)], dtype=torch.long
                    )
                    reconstruction = decoder(encoder(values[batch]))
                    loss = nn.functional.mse_loss(reconstruction, values[batch])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            self.final_loss_ = float(nn.functional.mse_loss(decoder(encoder(values)), values))
        self.encoder_ = encoder
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.encoder_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("autoencoder projection is not fitted")
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.input_dim_ or not np.isfinite(matrix).all():
            raise ValueError("autoencoder transform features do not match the source space")
        standardized = (matrix - self.mean_) / self.scale_
        with torch.no_grad():
            encoded = self.encoder_(torch.as_tensor(standardized, dtype=torch.float32))
        return encoded.numpy().astype(np.float32)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "representation": (
                "source_trained_nonlinear_autoencoder_encoder"
                if self.trained
                else "source_standardized_untrained_nonlinear_autoencoder_encoder"
            ),
            "fit_scope": "source_only",
            "fit_rows": int(self.fit_rows_),
            "input_dim": int(self.input_dim_),
            "requested_dim": int(self.bottleneck_dim),
            "effective_dim": int(self.output_dim_),
            "weights_trained": bool(self.trained),
            "epochs": int(self.epochs) if self.trained else 0,
            "initial_reconstruction_loss": self.initial_loss_,
            "final_reconstruction_loss": self.final_loss_,
            "normalization_fit_scope": "source_only",
            "seed": int(self.seed),
        }


def _mmd_test(source: np.ndarray, target: np.ndarray, cfg: DetectionBaselineConfig, *, seed: int) -> dict[str, Any]:
    test = MMDPermutationTest(
        WindowDriftConfig(
            mmd_permutations=int(cfg.mmd_permutations),
            reference_max_samples=max(len(source), 2),
            minimum_window_documents=2,
            window_size=max(len(target), 2),
            power_enabled=False,
            seed=int(seed),
        )
    ).fit(source, block_ids=np.asarray([f"source-{index}" for index in range(len(source))], dtype=str))
    return dict(
        test.test(
            target,
            seed=int(seed) + 17,
            block_ids=np.asarray([f"target-{index}" for index in range(len(target))], dtype=str),
        )
    )


def _bbsds_test(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Failing Loudly BBSDs: per-softmax-dimension KS with Bonferroni."""

    source_values = _matrix(source, "BBSDs source probabilities")
    target_values = _matrix(target, "BBSDs target probabilities")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("BBSDs source and target class dimensions must match")
    statistics: list[float] = []
    p_values: list[float] = []
    for class_index in range(source_values.shape[1]):
        result = ks_2samp(
            source_values[:, class_index],
            target_values[:, class_index],
            alternative="two-sided",
            method="auto",
        )
        statistics.append(float(result.statistic))
        p_values.append(float(result.pvalue))
    corrected = min(1.0, float(min(p_values)) * len(p_values))
    return {
        "statistic": float(max(statistics)),
        "p_value": corrected,
        "test": "per_dimension_ks_bonferroni",
        "dimension_statistics": statistics,
        "dimension_p_values": p_values,
        "multiple_testing_correction": "bonferroni",
        "source_rows": int(len(source_values)),
        "target_rows": int(len(target_values)),
    }


def _bbsdh_test(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Failing Loudly BBSDh: Pearson chi-square on predicted labels."""

    source_values = _matrix(source, "BBSDh source labels")
    target_values = _matrix(target, "BBSDh target labels")
    if source_values.shape[1] != target_values.shape[1]:
        raise ValueError("BBSDh source and target class dimensions must match")
    source_counts = np.bincount(
        np.argmax(source_values, axis=1), minlength=source_values.shape[1]
    )
    target_counts = np.bincount(
        np.argmax(target_values, axis=1), minlength=target_values.shape[1]
    )
    observed = np.vstack([source_counts, target_counts])
    observed = observed[:, observed.sum(axis=0) > 0]
    if observed.shape[1] < 2:
        statistic, p_value, degrees_of_freedom = 0.0, 1.0, 0
    else:
        statistic, p_value, degrees_of_freedom, _ = chi2_contingency(
            observed,
            correction=False,
        )
    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "test": "pearson_chi_square_predicted_class_counts",
        "degrees_of_freedom": int(degrees_of_freedom),
        "source_class_counts": source_counts.astype(int).tolist(),
        "target_class_counts": target_counts.astype(int).tolist(),
        "source_rows": int(len(source_values)),
        "target_rows": int(len(target_values)),
    }


def _classifier_test(source: np.ndarray, target: np.ndarray, cfg: DetectionBaselineConfig, *, seed: int) -> dict[str, Any]:
    balanced_rows = min(len(source), len(target))
    if balanced_rows < 2:
        raise ValueError("Classif needs at least two source and two target rows")
    rng = np.random.default_rng(int(seed) + 29)
    source_used = source[rng.choice(len(source), size=balanced_rows, replace=False)]
    target_used = target[rng.choice(len(target), size=balanced_rows, replace=False)]
    values = np.vstack([source_used, target_used])
    labels = np.concatenate(
        [np.zeros(balanced_rows, dtype=int), np.ones(balanced_rows, dtype=int)]
    )
    source_rows = np.flatnonzero(labels == 0)
    target_rows = np.flatnonzero(labels == 1)
    heldout_per_domain = min(
        balanced_rows - 1,
        max(1, int(round(0.30 * balanced_rows))),
    )
    source_order = rng.permutation(source_rows)
    target_order = rng.permutation(target_rows)
    heldout_indices = np.concatenate(
        [source_order[:heldout_per_domain], target_order[:heldout_per_domain]]
    )
    train_indices = np.concatenate(
        [source_order[heldout_per_domain:], target_order[heldout_per_domain:]]
    )
    classifier = LogisticRegression(
        C=1.0,
        max_iter=int(cfg.classifier_max_iter),
        class_weight="balanced",
        random_state=int(seed),
    ).fit(values[train_indices], labels[train_indices])
    heldout_labels = labels[heldout_indices]
    heldout_probabilities = classifier.predict_proba(values[heldout_indices])[:, 1]
    predictions = (heldout_probabilities >= 0.5).astype(int)
    correct = int(np.sum(predictions == heldout_labels))
    auc = float(roc_auc_score(heldout_labels, heldout_probabilities))
    permutation_exceedances = 0
    for _ in range(int(cfg.mmd_permutations)):
        permuted_labels = rng.permutation(heldout_labels)
        permutation_exceedances += int(np.sum(predictions == permuted_labels) >= correct)
    p_value = float(
        (1 + permutation_exceedances) / (int(cfg.mmd_permutations) + 1)
    )
    return {
        "statistic": float(correct / len(heldout_labels)),
        "accuracy": float(correct / len(heldout_labels)),
        "auroc": auc,
        "p_value": p_value,
        "source_rows": int(len(source_used)),
        "target_rows": int(len(target_used)),
        "source_rows_before_balance": int(len(source)),
        "target_rows_before_balance": int(len(target)),
        "domain_rows_balanced": True,
        "training_rows": int(len(train_indices)),
        "heldout_rows": int(len(heldout_indices)),
        "holdout_fraction_per_domain": float(heldout_per_domain / balanced_rows),
        "train_holdout_independent": True,
        "permutation_unit": "row",
        "classifier": "logistic_regression_independent_holdout",
        "p_value_method": "conservative_heldout_domain_label_permutation_conditional_on_fitted_classifier",
        "permutations": int(cfg.mmd_permutations),
    }


def _aggregate_detection_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["baseline"]), int(row["sample_size"])), []).append(row)
    return [
        {
            "baseline": key[0],
            "sample_size": key[1],
            "repetitions": len(values),
            "detection_rate": float(np.mean([bool(value["detected"]) for value in values])),
            "mean_statistic": float(np.mean([float(value["statistic"]) for value in values])),
            "mean_p_value": float(np.mean([float(value["p_value"]) for value in values])),
        }
        for key, values in sorted(grouped.items())
    ]


def _summarize_operational(name: str, action: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    harmful = np.flatnonzero(truth)
    benign = np.flatnonzero(~truth)
    alarms = np.flatnonzero(action)
    false_alarm_positions = alarms[~truth[alarms]] if alarms.size else np.zeros(0, dtype=int)
    detected_harm = harmful[action[harmful]] if harmful.size else np.zeros(0, dtype=int)
    first_harm = int(harmful[0]) if harmful.size else None
    first_detected = int(detected_harm[0]) if detected_harm.size else None
    delay = float(first_detected - first_harm) if first_harm is not None and first_detected is not None else float("nan")
    # With one finite schedule, ARL is the first false-alarm run length; the
    # right-censored value n+1 records that no false alarm was observed.
    average_run_length = float(false_alarm_positions[0] + 1) if false_alarm_positions.size else float(len(truth) + 1)
    return {
        "method": name,
        "implementation_kind": "window_policy_proxy_no_model_retraining",
        "display_name": (
            "always-update-policy-proxy" if name == "always-retrain" else name
        ),
        "alarm_windows": alarms.astype(int).tolist(),
        "harmful_detection_recall": float(detected_harm.size > 0) if harmful.size else float("nan"),
        "benign_specificity": float(np.mean(~action[benign])) if benign.size else float("nan"),
        "false_alarm_rate": float(np.mean(action[benign])) if benign.size else float("nan"),
        "wrong_update_rate": float(np.mean(~truth[alarms])) if alarms.size else 0.0,
        "first_detection_delay_windows": delay,
        "average_run_length_windows": average_run_length,
        "label_cost": int(alarms.size) if name in {"always-retrain", "final-accuracy-only"} else 0,
    }


def _aggregate_cost_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        value = float(row["absolute_estimation_error"])
        if np.isfinite(value):
            grouped.setdefault((str(row["method"]), int(row["budget"])), []).append(value)
    return [
        {"method": key[0], "budget": key[1], "repetitions": len(values), "mean_absolute_estimation_error": float(np.mean(values)), "std_absolute_estimation_error": float(np.std(values))}
        for key, values in sorted(grouped.items())
    ]


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 3:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D] or [N,L,D] with N>=2")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _vector(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _aligned_vector(values: Sequence[float], length: int, name: str) -> np.ndarray:
    array = _vector(values, name)
    if len(array) != int(length):
        raise ValueError(f"{name} must align with the window vector")
    return array


def _stable_int(value: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(str(value))) % 100003
