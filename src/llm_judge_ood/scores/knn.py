from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class Thresholds:
    """Global score thresholds calibrated on a held-out reference split."""

    soft: float
    hard: float
    soft_quantile: float
    hard_quantile: float
    calibration_count: int
    quantile_resolution: float

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "calibration_scope": "global_held_out_reference"}


def calibrate_thresholds(scores: np.ndarray, *, soft_q: float, hard_q: float) -> Thresholds:
    if not (0.0 <= float(soft_q) <= float(hard_q) <= 1.0):
        raise ValueError("Threshold quantiles must satisfy 0 <= soft_q <= hard_q <= 1")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Calibration scores must be a non-empty 1D array")
    return Thresholds(
        soft=float(np.quantile(values, soft_q)),
        hard=float(np.quantile(values, hard_q)),
        soft_quantile=float(soft_q),
        hard_quantile=float(hard_q),
        calibration_count=int(values.size),
        quantile_resolution=1.0 / float(values.size),
    )


class KNNScorer:
    def __init__(self, k: int = 10, metric: str = "euclidean", *, normalize: bool = False) -> None:
        self.k = int(k)
        self.metric = str(metric)
        self.normalize = bool(normalize)
        self.bank_: np.ndarray | None = None
        self.nn_: NearestNeighbors | None = None
        self.thresholds_: Thresholds | None = None

    def fit(self, training_features: np.ndarray) -> "KNNScorer":
        bank = np.asarray(training_features, dtype=np.float32)
        if bank.ndim != 2 or bank.shape[0] < 2:
            raise ValueError("training_features must be a 2D bank with at least two rows")
        if self.k < 1:
            raise ValueError("k must be at least 1")
        if self.k > bank.shape[0]:
            raise ValueError(
                f"k={self.k} exceeds the global training document bank size ({bank.shape[0]})"
            )
        if self.normalize:
            bank = _l2_normalize_rows(bank, name="training_features")
        self.bank_ = bank
        self.nn_ = NearestNeighbors(n_neighbors=self.k, metric=self.metric, n_jobs=-1)
        self.nn_.fit(bank)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.score_at_ks(features, (self.k,))[int(self.k)]

    def score_at_ks(self, features: np.ndarray, ks: tuple[int, ...] | list[int]) -> dict[int, np.ndarray]:
        """Return k-th-neighbor distances from one shared nearest-neighbor query."""

        if self.nn_ is None:
            raise RuntimeError("KNNScorer is not fitted")
        values = np.asarray(features, dtype=np.float32)
        if self.normalize:
            values = _l2_normalize_rows(values, name="features")
        requested = tuple(sorted(set(int(value) for value in ks)))
        if not requested or requested[0] < 1:
            raise ValueError("kNN scores require one or more positive k values")
        if self.bank_ is None or requested[-1] > int(self.bank_.shape[0]):
            raise ValueError("Requested k exceeds the fitted kNN reference bank")
        distances, _ = self.nn_.kneighbors(
            values,
            n_neighbors=int(requested[-1]),
            return_distance=True,
        )
        return {
            int(k): distances[:, int(k) - 1].astype(np.float64)
            for k in requested
        }

    def calibrate(self, calibration_features: np.ndarray, *, soft_q: float = 0.90, hard_q: float = 0.95) -> Thresholds:
        self.thresholds_ = calibrate_thresholds(self.score(calibration_features), soft_q=soft_q, hard_q=hard_q)
        return self.thresholds_

    def labels(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("KNNScorer thresholds are not calibrated")
        values = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(["id"] * values.shape[0], dtype=object)
        labels[values >= self.thresholds_.soft] = "soft_ood"
        labels[values >= self.thresholds_.hard] = "hard_ood"
        return labels

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scorer": "knn",
            "k": self.k,
            "metric": self.metric,
            "l2_normalized": self.normalize,
            "bank_rows": int(self.bank_.shape[0]) if self.bank_ is not None else None,
            "thresholds": self.thresholds_.to_dict() if self.thresholds_ else None,
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        if self.bank_ is None:
            raise RuntimeError("KNNScorer is not fitted")
        return {"reference_bank": self.bank_.astype(np.float32)}


class DocumentKNNScorer(KNNScorer):
    """Global kNN scorer for unique deployment input documents."""

    def to_metadata(self) -> dict[str, Any]:
        return {
            **super().to_metadata(),
            "feature_scope": "input_document",
            "retrieval_scope": "global_training_document_bank",
        }


def _l2_normalize_rows(values: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 2D matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-norm row that cannot be L2-normalized")
    return (matrix / norms).astype(np.float32)
