from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.llm_judge_ood.scores.knn import Thresholds, calibrate_thresholds


@dataclass
class ViMScorer:
    """PCA residual scorer using the feature-space component of ViM."""

    rank: int
    epsilon: float = 1e-12
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    thresholds_: Thresholds | None = None
    fit_rows_: int = 0

    def fit(self, penultimate: np.ndarray) -> "ViMScorer":
        h = _validate_penultimate(penultimate, minimum_rows=3)
        mean = h.mean(axis=0)
        centered = h - mean
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        return self.fit_from_svd(
            h,
            source_mean=mean,
            right_singular_vectors=right,
        )

    def fit_from_svd(
        self,
        penultimate: np.ndarray,
        *,
        source_mean: np.ndarray,
        right_singular_vectors: np.ndarray,
    ) -> "ViMScorer":
        """Fit from a source SVD shared by a ViM rank-selection grid.

        The source mean and right singular vectors fully determine every rank
        candidate. Reusing them avoids recomputing the same SVD for each rank.
        """

        h = _validate_penultimate(penultimate, minimum_rows=3)
        max_rank = min(h.shape[0] - 1, h.shape[1])
        if self.rank < 1 or self.rank >= max_rank:
            raise ValueError(
                "ViM rank must be within 1..min(source_rows - 1, penultimate_dim) - 1; "
                f"got rank={self.rank}, source_rows={h.shape[0]}, penultimate_dim={h.shape[1]}"
            )
        mean = np.asarray(source_mean, dtype=np.float64)
        right = np.asarray(right_singular_vectors, dtype=np.float64)
        if mean.shape != (h.shape[1],):
            raise ValueError("ViM source mean does not match the penultimate dimension")
        if right.ndim != 2 or right.shape[1] != h.shape[1] or right.shape[0] < int(self.rank):
            raise ValueError("ViM right singular vectors do not support the requested rank")
        centered = h - mean
        components = right[: int(self.rank)].T
        residuals = _residual_norms(centered, components)
        if float(residuals.sum()) <= float(self.epsilon):
            raise ValueError("ViM residual norm is zero; choose a lower rank or non-degenerate Judge features")
        self.mean_ = mean.astype(np.float64)
        self.components_ = components.astype(np.float64)
        self.fit_rows_ = int(h.shape[0])
        return self

    def residual_norm(self, penultimate: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.residual_features(penultimate), axis=1)

    def residual_features(self, penultimate: np.ndarray) -> np.ndarray:
        """Return the source-subspace residual vector used by downstream MMD."""

        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("ViMScorer is not fitted")
        values = np.asarray(penultimate, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean_.shape[0] or not np.isfinite(values).all():
            raise ValueError("ViM penultimate features do not match the fitted source space")
        centered = values - self.mean_
        return (centered - centered @ self.components_ @ self.components_.T).astype(np.float64)

    def score(self, penultimate: np.ndarray) -> np.ndarray:
        return self.residual_norm(penultimate)

    def calibrate(
        self,
        penultimate: np.ndarray,
        *,
        soft_q: float = 0.90,
        hard_q: float = 0.95,
    ) -> Thresholds:
        self.thresholds_ = calibrate_thresholds(self.score(penultimate), soft_q=soft_q, hard_q=hard_q)
        return self.thresholds_

    def labels(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("ViMScorer thresholds are not calibrated")
        values = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(["id"] * len(values), dtype=object)
        labels[values >= self.thresholds_.soft] = "soft_ood"
        labels[values >= self.thresholds_.hard] = "hard_ood"
        return labels

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("ViMScorer is not fitted")
        return {
            "source_mean": self.mean_.astype(np.float32),
            "principal_components": self.components_.astype(np.float32),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scorer": "vim",
            "feature_scope": "judge_behavior",
            "detection_unit": "judge_record",
            "score_variant": "residual_only",
            "uses_logits": False,
            "rank": int(self.rank),
            "fit_rows": int(self.fit_rows_),
            "thresholds": self.thresholds_.to_dict() if self.thresholds_ is not None else None,
        }


@dataclass
class FullViMScorer:
    """Original virtual-logit ViM using the deployed affine-head origin."""

    rank: int
    epsilon: float = 1e-12
    origins_: np.ndarray | None = None
    query_ids_: tuple[str, ...] = ()
    components_: np.ndarray | None = None
    alpha_: float | None = None
    thresholds_: Thresholds | None = None
    fit_rows_: int = 0
    class_count_: int | None = None

    def fit(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        *,
        head_weight: np.ndarray,
        head_bias: np.ndarray,
        query_ids: np.ndarray | None = None,
        head_query_ids: np.ndarray | None = None,
    ) -> "FullViMScorer":
        h, values = _validate_full_vim_inputs(penultimate, logits, minimum_rows=3)
        origins, fitted_query_ids = _classifier_origins(
            head_weight,
            head_bias,
            feature_dim=h.shape[1],
            class_count=values.shape[1],
            head_query_ids=head_query_ids,
        )
        positions = _query_positions(query_ids, len(h), fitted_query_ids)
        centered = h - origins[positions]
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        return self.fit_from_svd(
            h,
            values,
            classifier_origins=origins,
            source_query_ids=query_ids,
            head_query_ids=np.asarray(fitted_query_ids, dtype=str),
            right_singular_vectors=right,
        )

    def fit_from_svd(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        *,
        classifier_origins: np.ndarray,
        source_query_ids: np.ndarray | None,
        head_query_ids: np.ndarray,
        right_singular_vectors: np.ndarray,
    ) -> "FullViMScorer":
        h, values = _validate_full_vim_inputs(penultimate, logits, minimum_rows=3)
        max_rank = min(h.shape[0] - 1, h.shape[1])
        if self.rank < 1 or self.rank >= max_rank:
            raise ValueError(
                "Full ViM rank must be within 1..min(source_rows - 1, penultimate_dim) - 1"
            )
        origins = np.asarray(classifier_origins, dtype=np.float64)
        fitted_query_ids = tuple(np.asarray(head_query_ids).astype(str).tolist())
        right = np.asarray(right_singular_vectors, dtype=np.float64)
        if origins.shape != (len(fitted_query_ids), h.shape[1]):
            raise ValueError("Full ViM classifier origins do not match the fitted head and feature space")
        if right.ndim != 2 or right.shape[1] != h.shape[1] or right.shape[0] < int(self.rank):
            raise ValueError("Full ViM right singular vectors do not support the requested rank")
        positions = _query_positions(source_query_ids, len(h), fitted_query_ids)
        centered = h - origins[positions]
        components = right[: int(self.rank)].T
        residuals = _residual_norms(centered, components)
        denominator = float(residuals.sum())
        if denominator <= float(self.epsilon):
            raise ValueError("Full ViM source residual norm is zero")
        alpha = float(np.max(values, axis=1).sum()) / denominator
        if not np.isfinite(alpha) or alpha <= 0.0:
            raise ValueError("Full ViM alpha must be finite and positive")
        self.origins_ = origins.copy()
        self.query_ids_ = fitted_query_ids
        self.components_ = components.astype(np.float64)
        self.alpha_ = alpha
        self.fit_rows_ = int(len(h))
        self.class_count_ = int(values.shape[1])
        return self

    def residual_norm(
        self,
        penultimate: np.ndarray,
        query_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.origins_ is None or self.components_ is None:
            raise RuntimeError("FullViMScorer is not fitted")
        h = _validate_penultimate(penultimate, minimum_rows=1)
        if h.shape[1] != self.origins_.shape[1]:
            raise ValueError("Full ViM penultimate dimension does not match the source space")
        positions = _query_positions(query_ids, len(h), self.query_ids_)
        return _residual_norms(h - self.origins_[positions], self.components_)

    def score(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        query_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.alpha_ is None or self.class_count_ is None:
            raise RuntimeError("FullViMScorer is not fitted")
        h, values = _validate_full_vim_inputs(penultimate, logits, minimum_rows=1)
        if values.shape[1] != int(self.class_count_):
            raise ValueError("Full ViM logits class count does not match the fitted Judge head")
        virtual_logits = float(self.alpha_) * self.residual_norm(h, query_ids)
        combined = np.concatenate([virtual_logits[:, None], values], axis=1)
        maxima = np.max(combined, axis=1, keepdims=True)
        normalized = np.exp(combined - maxima)
        return (normalized[:, 0] / normalized.sum(axis=1)).astype(np.float64)

    def calibrate(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        *,
        query_ids: np.ndarray | None = None,
        soft_q: float = 0.90,
        hard_q: float = 0.95,
    ) -> Thresholds:
        self.thresholds_ = calibrate_thresholds(
            self.score(penultimate, logits, query_ids), soft_q=soft_q, hard_q=hard_q
        )
        return self.thresholds_

    def labels(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("FullViMScorer thresholds are not calibrated")
        values = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(["id"] * len(values), dtype=object)
        labels[values >= self.thresholds_.soft] = "soft_ood"
        labels[values >= self.thresholds_.hard] = "hard_ood"
        return labels

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        if self.origins_ is None or self.components_ is None or self.alpha_ is None:
            raise RuntimeError("FullViMScorer is not fitted")
        return {
            "classifier_origins": self.origins_.astype(np.float32),
            "head_query_ids": np.asarray(self.query_ids_, dtype=str),
            "principal_components": self.components_.astype(np.float32),
            "alpha": np.asarray(self.alpha_, dtype=np.float64),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scorer": "full_vim",
            "feature_scope": "judge_behavior",
            "detection_unit": "judge_record",
            "score_variant": "virtual_logit_softmax",
            "uses_logits": True,
            "center": "classifier_origin_-pinv(W.T)@b",
            "rank": int(self.rank),
            "fit_rows": int(self.fit_rows_),
            "class_count": self.class_count_,
            "alpha": self.alpha_,
            "thresholds": self.thresholds_.to_dict() if self.thresholds_ is not None else None,
        }


def _classifier_origins(
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    *,
    feature_dim: int,
    class_count: int,
    head_query_ids: np.ndarray | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    weights = np.asarray(head_weight, dtype=np.float64)
    biases = np.asarray(head_bias, dtype=np.float64)
    if weights.ndim == 2:
        weights = weights[None, :, :]
    if biases.ndim == 1:
        biases = biases[None, :]
    if weights.shape != (weights.shape[0], int(feature_dim), int(class_count)):
        raise ValueError("Full ViM head weights must have shape [Q, D, K] or [D, K]")
    if biases.shape != (weights.shape[0], int(class_count)):
        raise ValueError("Full ViM head biases must have shape [Q, K] or [K]")
    if not np.isfinite(weights).all() or not np.isfinite(biases).all():
        raise ValueError("Full ViM head parameters must be finite")
    if head_query_ids is None:
        if weights.shape[0] != 1:
            raise ValueError("Multiple Full ViM heads require head_query_ids")
        fitted_query_ids = ("__global__",)
    else:
        fitted_query_ids = tuple(np.asarray(head_query_ids).astype(str).tolist())
        if len(fitted_query_ids) != weights.shape[0] or len(set(fitted_query_ids)) != len(fitted_query_ids):
            raise ValueError("Full ViM head_query_ids must uniquely align with head parameters")
    origins = np.stack(
        [-np.linalg.pinv(weights[index].T) @ biases[index] for index in range(weights.shape[0])]
    )
    return origins.astype(np.float64), fitted_query_ids


def _query_positions(
    query_ids: np.ndarray | None,
    row_count: int,
    fitted_query_ids: tuple[str, ...],
) -> np.ndarray:
    if len(fitted_query_ids) == 1 and query_ids is None:
        return np.zeros(int(row_count), dtype=int)
    if query_ids is None:
        raise ValueError("Multiple Full ViM heads require row-aligned query_ids")
    queries = np.asarray(query_ids).astype(str)
    if queries.shape != (int(row_count),):
        raise ValueError("Full ViM query_ids must align with penultimate rows")
    lookup = {query: index for index, query in enumerate(fitted_query_ids)}
    unknown = sorted(set(queries.tolist()) - set(lookup))
    if unknown:
        raise ValueError(f"Full ViM has no classifier origin for query_ids={unknown}")
    return np.asarray([lookup[query] for query in queries.tolist()], dtype=int)


def _validate_penultimate(
    penultimate: np.ndarray,
    *,
    minimum_rows: int,
) -> np.ndarray:
    h = np.asarray(penultimate, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("ViM residual scoring requires [N, D] penultimate features")
    if h.shape[0] < int(minimum_rows) or h.shape[1] < 2:
        raise ValueError(
            f"ViM residual scoring requires at least {int(minimum_rows)} rows and two feature dimensions"
        )
    if not np.isfinite(h).all():
        raise ValueError("ViM residual features contain non-finite values")
    return h


def _validate_full_vim_inputs(
    penultimate: np.ndarray,
    logits: np.ndarray,
    *,
    minimum_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    h = _validate_penultimate(penultimate, minimum_rows=minimum_rows)
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(h) or values.shape[1] < 2:
        raise ValueError("Full ViM requires aligned [N,D] features and [N,K] logits")
    if not np.isfinite(values).all():
        raise ValueError("Full ViM logits contain non-finite values")
    return h, values


def _residual_norms(centered: np.ndarray, components: np.ndarray) -> np.ndarray:
    projected = centered @ components @ components.T
    return np.linalg.norm(centered - projected, axis=1)
