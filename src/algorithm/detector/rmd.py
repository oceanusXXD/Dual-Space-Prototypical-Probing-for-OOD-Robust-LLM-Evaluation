from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from src.algorithm.detector.knn import Thresholds, calibrate_thresholds


@dataclass
class GaussianStats:
    mean: np.ndarray
    precision: np.ndarray

    def md(self, values: np.ndarray) -> np.ndarray:
        centered = np.asarray(values, dtype=np.float64) - self.mean
        return np.einsum("ij,jk,ik->i", centered, self.precision, centered)


class MahalanobisScorer:
    def __init__(self, regularization: float = 1e-5) -> None:
        self.regularization = float(regularization)
        self.global_: GaussianStats | None = None

    def fit(self, training_features: np.ndarray) -> "MahalanobisScorer":
        self.global_ = _fit_gaussian(np.asarray(training_features, dtype=np.float64), self.regularization)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.global_ is None:
            raise RuntimeError("MahalanobisScorer is not fitted")
        return self.global_.md(np.asarray(features, dtype=np.float64)).astype(np.float64)

    def to_metadata(self) -> dict[str, Any]:
        return {"scorer": "mahalanobis", "regularization": self.regularization}


class RMDScorer:
    """Label-conditioned B-space RMD scorer for Judge-record diagnostics."""

    def __init__(self, regularization: float = 1e-5) -> None:
        self.regularization = float(regularization)
        self.global_: GaussianStats | None = None
        self.class_stats_: dict[str, GaussianStats] = {}
        self.thresholds_: Thresholds | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "RMDScorer":
        values = np.asarray(features, dtype=np.float64)
        label_values = np.asarray(labels).astype(str)
        if values.ndim != 2 or values.shape[0] < 2 or len(values) != len(label_values):
            raise ValueError("RMDScorer requires aligned [N, D] features and labels with at least two rows")
        self.global_ = _fit_gaussian(values, self.regularization)
        self.class_stats_ = {}
        class_means = {
            label: values[label_values == label].mean(axis=0)
            for label in sorted(set(label_values.tolist()))
        }
        residuals = np.vstack(
            [values[index] - class_means[str(label)] for index, label in enumerate(label_values.tolist())]
        )
        shared_precision = _fit_precision(residuals, self.regularization)
        self.class_stats_ = {
            label: GaussianStats(mean=mean, precision=shared_precision)
            for label, mean in class_means.items()
        }
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.global_ is None or not self.class_stats_:
            raise RuntimeError("RMDScorer is not fitted")
        values = np.asarray(features, dtype=np.float64)
        global_scores = self.global_.md(values)
        class_scores = np.vstack([stats.md(values) for stats in self.class_stats_.values()])
        return (np.min(class_scores, axis=0) - global_scores).astype(np.float64)

    def calibrate(
        self,
        features: np.ndarray,
        *,
        soft_q: float = 0.90,
        hard_q: float = 0.95,
    ) -> Thresholds:
        self.thresholds_ = calibrate_thresholds(self.score(features), soft_q=soft_q, hard_q=hard_q)
        return self.thresholds_

    def labels(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("RMDScorer thresholds are not calibrated")
        values = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(["id"] * len(values), dtype=object)
        labels[values >= self.thresholds_.soft] = "soft_ood"
        labels[values >= self.thresholds_.hard] = "hard_ood"
        return labels

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        if self.global_ is None or not self.class_stats_:
            raise RuntimeError("RMDScorer is not fitted")
        classes = sorted(self.class_stats_)
        return {
            "global_mean": self.global_.mean.astype(np.float32),
            "global_precision": self.global_.precision.astype(np.float32),
            "class_names": np.asarray(classes),
            "class_means": np.stack([self.class_stats_[name].mean for name in classes]).astype(np.float32),
            "class_precisions": np.stack([self.class_stats_[name].precision for name in classes]).astype(np.float32),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scorer": "rmd",
            "regularization": self.regularization,
            "classes": sorted(self.class_stats_),
            "class_covariance": "shared_shrinkage_gda",
            "covariance_estimator": "LedoitWolf_plus_numerical_ridge",
            "feature_scope": "judge_behavior",
            "detection_unit": "judge_record",
            "thresholds": self.thresholds_.to_dict() if self.thresholds_ is not None else None,
        }


class DocumentGaussianScorer:
    """Global label-free Mahalanobis scorer for input-document OOD."""

    def __init__(self, regularization: float = 1e-5) -> None:
        self.regularization = float(regularization)
        self.scorer = MahalanobisScorer(regularization=self.regularization)
        self.thresholds_: Thresholds | None = None
        self.metric = "mahalanobis"
        self.k: int | None = None

    def fit(self, training_features: np.ndarray) -> "DocumentGaussianScorer":
        self.scorer.fit(training_features)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.scorer.score(features)

    def calibrate(
        self,
        calibration_features: np.ndarray,
        *,
        soft_q: float = 0.90,
        hard_q: float = 0.95,
    ) -> Thresholds:
        self.thresholds_ = calibrate_thresholds(self.score(calibration_features), soft_q=soft_q, hard_q=hard_q)
        return self.thresholds_

    def labels(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("DocumentGaussianScorer thresholds are not calibrated")
        values = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(["id"] * values.shape[0], dtype=object)
        labels[values >= self.thresholds_.soft] = "soft_ood"
        labels[values >= self.thresholds_.hard] = "hard_ood"
        return labels

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scorer": "mahalanobis",
            "metric": self.metric,
            "k": self.k,
            "regularization": self.regularization,
            "feature_scope": "input_document",
            "retrieval_scope": "global_training_document_bank",
            "thresholds": self.thresholds_.to_dict() if self.thresholds_ is not None else None,
        }


def _fit_gaussian(values: np.ndarray, regularization: float) -> GaussianStats:
    mean = values.mean(axis=0)
    centered = values - mean
    precision = _fit_precision(centered, regularization)
    return GaussianStats(mean=mean, precision=precision)


def _fit_precision(centered: np.ndarray, regularization: float) -> np.ndarray:
    values = np.asarray(centered, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Gaussian covariance needs at least two feature rows")
    covariance = np.atleast_2d(LedoitWolf().fit(values).covariance_).astype(np.float64)
    covariance += float(regularization) * np.eye(covariance.shape[0])
    return np.linalg.pinv(covariance)
