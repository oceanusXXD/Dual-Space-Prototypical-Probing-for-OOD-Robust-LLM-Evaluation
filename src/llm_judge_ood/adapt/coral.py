from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CoralConfig:
    regularization: float = 1e-3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CoralAligner:
    """Unlabeled CORAL alignment baseline.

    The aligner learns source and target first/second moments, then maps target
    features into the source covariance geometry. It is used only as a baseline;
    the main adaptation path remains few-label head/source-replay training.
    """

    def __init__(self, config: CoralConfig | None = None) -> None:
        self.config = config or CoralConfig()
        self.source_mean_: np.ndarray | None = None
        self.target_mean_: np.ndarray | None = None
        self.source_sqrt_: np.ndarray | None = None
        self.target_inv_sqrt_: np.ndarray | None = None

    def fit(self, source_features: np.ndarray, target_features: np.ndarray) -> "CoralAligner":
        source = np.asarray(source_features, dtype=np.float64)
        target = np.asarray(target_features, dtype=np.float64)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("CORAL expects 2D source and target feature matrices")
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and target dimensions must match for CORAL")
        self.source_mean_ = source.mean(axis=0)
        self.target_mean_ = target.mean(axis=0)
        self.source_sqrt_ = _matrix_power(_cov(source, self.config.regularization), 0.5)
        self.target_inv_sqrt_ = _matrix_power(_cov(target, self.config.regularization), -0.5)
        return self

    def transform_target(self, target_features: np.ndarray) -> np.ndarray:
        if self.source_mean_ is None or self.target_mean_ is None or self.source_sqrt_ is None or self.target_inv_sqrt_ is None:
            raise RuntimeError("CORAL aligner is not fitted")
        target = np.asarray(target_features, dtype=np.float64)
        aligned = (target - self.target_mean_) @ self.target_inv_sqrt_ @ self.source_sqrt_ + self.source_mean_
        return aligned.astype(np.float32)

    def fit_transform_target(self, source_features: np.ndarray, target_features: np.ndarray) -> np.ndarray:
        return self.fit(source_features, target_features).transform_target(target_features)

    def to_metadata(self) -> dict[str, Any]:
        return {"baseline": "coral", "config": self.config.to_dict(), "fitted": self.source_mean_ is not None}


def nearest_centroid_predict(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_features: np.ndarray,
) -> np.ndarray:
    labels = np.asarray(source_labels)
    centroids: list[np.ndarray] = []
    classes: list[Any] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        centroids.append(np.asarray(source_features, dtype=np.float32)[mask].mean(axis=0))
        classes.append(label)
    centroid_matrix = np.vstack(centroids)
    distances = ((np.asarray(target_features, dtype=np.float32)[:, None, :] - centroid_matrix[None, :, :]) ** 2).sum(axis=2)
    return np.asarray(classes, dtype=object)[np.argmin(distances, axis=1)]


def _cov(values: np.ndarray, regularization: float) -> np.ndarray:
    centered = values - values.mean(axis=0)
    covariance = np.cov(centered, rowvar=False)
    covariance = np.atleast_2d(covariance).astype(np.float64)
    return covariance + float(regularization) * np.eye(covariance.shape[0])


def _matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ np.diag(eigvals**power) @ eigvecs.T
