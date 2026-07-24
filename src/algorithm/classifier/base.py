from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from torch import nn

from src.algorithm.classifier.output import JudgeHeadOutput, stable_softmax
from src.common.metrics import _quadratic_weighted_kappa, normalize_label_array
from src.common.representation import RepresentationSpec, RepresentationTransform


@dataclass(frozen=True)
class LinearJudgeConfig:
    method: str = "ridge"
    alpha: float = 10.0
    c: float = 0.1
    max_iter: int = 500
    representation: str = "last_layer"
    pca_dim: int = 48
    class_values: tuple[Any, ...] = ()
    seed: int = 42
    class_weight: str | None = "balanced"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    batch_size: int = 256
    patience: int = 6
    device: str = "cpu"
    head_sharing: str = "shared"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LinearSoftmaxClassifier:
    """Sklearn-shaped pure linear softmax head trained with the specified AdamW recipe."""

    def __init__(self, config: LinearJudgeConfig, classes: np.ndarray) -> None:
        self.config = config
        self.classes_ = np.asarray(classes)
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.best_epoch_: int | None = None
        self.best_validation_qwk_: float | None = None
        self.epochs_ran_: int = 0
        self.device_: str = "cpu"

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> "LinearSoftmaxClassifier":
        matrix = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels)
        if matrix.ndim != 2 or len(matrix) != len(targets) or len(matrix) == 0:
            raise ValueError("Linear softmax training requires aligned non-empty [N,D] features")
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        if any(value not in class_to_index for value in targets.tolist()):
            raise ValueError("Linear softmax training labels fall outside the configured classes")
        target_indices = np.asarray([class_to_index[value] for value in targets], dtype=np.int64)
        requested_device = str(self.config.device).lower()
        device = torch.device(
            requested_device
            if not requested_device.startswith("cuda") or torch.cuda.is_available()
            else "cpu"
        )
        self.device_ = str(device)
        torch.manual_seed(int(self.config.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.config.seed))
        head = nn.Linear(matrix.shape[1], len(self.classes_), bias=True).to(device)
        with torch.no_grad():
            head.weight.zero_()
            head.bias.zero_()
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        counts = np.bincount(target_indices, minlength=len(self.classes_)).astype(np.float32)
        present = counts > 0
        class_weights = np.zeros(len(self.classes_), dtype=np.float32)
        if str(self.config.class_weight).lower() == "balanced":
            class_weights[present] = len(target_indices) / (present.sum() * counts[present])
        else:
            class_weights[present] = 1.0
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        y = torch.as_tensor(target_indices, dtype=torch.long, device=device)
        validation = _linear_validation_payload(
            validation_features=validation_features,
            validation_labels=validation_labels,
            classes=self.classes_,
            class_to_index=class_to_index,
            device=device,
        )
        rng = np.random.default_rng(int(self.config.seed))
        best_state: dict[str, torch.Tensor] | None = None
        best_qwk = -float("inf")
        best_epoch = 0
        stale_epochs = 0
        epochs = max(1, int(self.config.epochs))
        batch_size = max(1, int(self.config.batch_size))
        for epoch in range(epochs):
            head.train()
            order = rng.permutation(len(matrix))
            for start in range(0, len(order), batch_size):
                batch = torch.as_tensor(
                    order[start : start + batch_size],
                    dtype=torch.long,
                    device=device,
                )
                logits = head(x[batch])
                loss = nn.functional.cross_entropy(logits, y[batch], weight=weight_tensor)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.epochs_ran_ = epoch + 1
            if validation is None:
                continue
            validation_qwk = _linear_validation_qwk(head, validation, self.classes_)
            if validation_qwk > best_qwk + 1e-12:
                best_qwk = validation_qwk
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in head.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= max(1, int(self.config.patience)):
                    break
        if best_state is not None:
            head.load_state_dict(best_state)
            self.best_epoch_ = int(best_epoch)
            self.best_validation_qwk_ = float(best_qwk)
        else:
            self.best_epoch_ = int(self.epochs_ran_)
            self.best_validation_qwk_ = None
        head.eval()
        self.coef_ = head.weight.detach().cpu().numpy().astype(np.float64)
        self.intercept_ = head.bias.detach().cpu().numpy().astype(np.float64)
        return self

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return (
            np.asarray(features, dtype=np.float64) @ self.coef_.T
            + self.intercept_[None, :]
        )

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return stable_softmax(self.decision_function(features))

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(features), axis=1)]

    def optimization_metadata(self) -> dict[str, Any]:
        return {
            "optimizer": "AdamW",
            "learning_rate": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
            "batch_size": int(self.config.batch_size),
            "maximum_epochs": int(self.config.epochs),
            "epochs_ran": int(self.epochs_ran_),
            "early_stopping_metric": "validation_qwk",
            "patience": int(self.config.patience),
            "best_epoch": self.best_epoch_,
            "best_validation_qwk": self.best_validation_qwk_,
            "device": self.device_,
        }

    def _require_fitted(self) -> None:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("Linear softmax classifier is not fitted")


class _CoralLinearHead(nn.Module):
    def __init__(self, input_dim: int, threshold_count: int) -> None:
        super().__init__()
        self.score = nn.Linear(int(input_dim), 1, bias=True)
        self.first_threshold = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.threshold_deltas = nn.Parameter(
            torch.zeros(max(int(threshold_count) - 1, 0), dtype=torch.float32)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.score(values) - self.thresholds().unsqueeze(0)

    def thresholds(self) -> torch.Tensor:
        if self.threshold_deltas.numel() == 0:
            return self.first_threshold.unsqueeze(0)
        increments = nn.functional.softplus(self.threshold_deltas) + 1e-4
        return torch.cat(
            [
                self.first_threshold.unsqueeze(0),
                self.first_threshold + torch.cumsum(increments, dim=0),
            ],
            dim=0,
        )


def _inverse_softplus(values: np.ndarray) -> np.ndarray:
    clipped = np.maximum(np.asarray(values, dtype=np.float64), 1e-6)
    return np.log(np.expm1(clipped))


class CoralOrdinalClassifier:
    """Linear ordinal CORAL head over frozen features.

    The head uses one shared linear direction and one bias per ordinal threshold.
    For K ordered labels it optimizes K-1 Bernoulli targets ``label > threshold``.
    """

    def __init__(self, config: LinearJudgeConfig, classes: np.ndarray) -> None:
        self.config = config
        self.classes_ = np.asarray(classes)
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.score_bias_: float | None = None
        self.thresholds_: np.ndarray | None = None
        self.feature_mean_: np.ndarray | None = None
        self.feature_scale_: np.ndarray | None = None
        self.best_epoch_: int | None = None
        self.best_validation_qwk_: float | None = None
        self.best_validation_mae_: float | None = None
        self.epochs_ran_: int = 0
        self.device_: str = "cpu"

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        *,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> "CoralOrdinalClassifier":
        matrix = np.asarray(features, dtype=np.float32)
        targets = np.asarray(labels)
        if matrix.ndim != 2 or len(matrix) != len(targets) or len(matrix) == 0:
            raise ValueError("CORAL training requires aligned non-empty [N,D] features")
        if len(self.classes_) < 2:
            raise ValueError("CORAL training requires at least two ordinal classes")
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        if any(value not in class_to_index for value in targets.tolist()):
            raise ValueError("CORAL training labels fall outside the configured classes")
        target_indices = np.asarray([class_to_index[value] for value in targets], dtype=np.int64)
        level_targets = _coral_levels(target_indices, len(self.classes_))
        self.feature_mean_ = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale_ = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        self.feature_scale_ = np.where(self.feature_scale_ < 1e-6, 1.0, self.feature_scale_).astype(np.float32)
        matrix = self._standardize(matrix)
        validation_features = (
            self._standardize(np.asarray(validation_features, dtype=np.float32))
            if validation_features is not None
            else None
        )
        requested_device = str(self.config.device).lower()
        device = torch.device(
            requested_device
            if not requested_device.startswith("cuda") or torch.cuda.is_available()
            else "cpu"
        )
        self.device_ = str(device)
        torch.manual_seed(int(self.config.seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.config.seed))
        head = _CoralLinearHead(matrix.shape[1], len(self.classes_) - 1).to(device)
        with torch.no_grad():
            head.score.weight.zero_()
            head.score.bias.zero_()
            priors = np.clip(level_targets.mean(axis=0), 1e-4, 1.0 - 1e-4)
            initial_thresholds = -np.log(priors / (1.0 - priors))
            initial_thresholds = np.maximum.accumulate(initial_thresholds)
            if len(initial_thresholds) > 1:
                for idx in range(1, len(initial_thresholds)):
                    if initial_thresholds[idx] <= initial_thresholds[idx - 1]:
                        initial_thresholds[idx] = initial_thresholds[idx - 1] + 1e-3
            head.first_threshold.copy_(torch.as_tensor(initial_thresholds[0], dtype=torch.float32, device=device))
            if head.threshold_deltas.numel():
                deltas = np.diff(initial_thresholds) - 1e-4
                head.threshold_deltas.copy_(
                    torch.as_tensor(_inverse_softplus(deltas), dtype=torch.float32, device=device)
                )
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        positive = level_targets.sum(axis=0).astype(np.float32)
        negative = float(len(level_targets)) - positive
        if str(self.config.class_weight).lower() == "balanced":
            pos_weight = negative / np.maximum(positive, 1.0)
        else:
            pos_weight = np.ones_like(positive, dtype=np.float32)
        pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
        x = torch.as_tensor(matrix, dtype=torch.float32, device=device)
        y = torch.as_tensor(level_targets, dtype=torch.float32, device=device)
        validation = _linear_validation_payload(
            validation_features=validation_features,
            validation_labels=validation_labels,
            classes=self.classes_,
            class_to_index=class_to_index,
            device=device,
        )
        rng = np.random.default_rng(int(self.config.seed))
        best_state: dict[str, torch.Tensor] | None = None
        best_qwk = -float("inf")
        best_mae = float("inf")
        best_epoch = 0
        stale_epochs = 0
        epochs = max(1, int(self.config.epochs))
        batch_size = max(1, int(self.config.batch_size))
        for epoch in range(epochs):
            head.train()
            order = rng.permutation(len(matrix))
            for start in range(0, len(order), batch_size):
                batch = torch.as_tensor(
                    order[start : start + batch_size],
                    dtype=torch.long,
                    device=device,
                )
                logits = head(x[batch])
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    y[batch],
                    pos_weight=pos_weight_tensor,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            self.epochs_ran_ = epoch + 1
            if validation is None:
                continue
            validation_qwk, validation_mae = _coral_validation_metrics(head, validation, self.classes_)
            improved = validation_qwk > best_qwk + 1e-12 or (
                abs(validation_qwk - best_qwk) <= 1e-12 and validation_mae < best_mae - 1e-12
            )
            if improved:
                best_qwk = validation_qwk
                best_mae = validation_mae
                best_epoch = epoch + 1
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in head.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= max(1, int(self.config.patience)):
                    break
        if best_state is not None:
            head.load_state_dict(best_state)
            self.best_epoch_ = int(best_epoch)
            self.best_validation_qwk_ = float(best_qwk)
            self.best_validation_mae_ = float(best_mae)
        else:
            self.best_epoch_ = int(self.epochs_ran_)
            self.best_validation_qwk_ = None
            self.best_validation_mae_ = None
        head.eval()
        self.coef_ = head.score.weight.detach().cpu().numpy().astype(np.float64)
        self.score_bias_ = float(head.score.bias.detach().cpu().item())
        self.thresholds_ = head.thresholds().detach().cpu().numpy().astype(np.float64)
        self.intercept_ = self.score_bias_ - self.thresholds_
        return self

    def threshold_logits(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("CORAL threshold logits require [N,D] features")
        standardized = self._standardize(matrix)
        return standardized @ self.coef_.T + self.intercept_[None, :]

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        probabilities = np.clip(self.predict_proba(features), 1e-12, 1.0)
        return np.log(probabilities)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self.threshold_logits(features)
        survival = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
        survival = np.minimum.accumulate(survival, axis=1)
        probabilities = np.zeros((survival.shape[0], len(self.classes_)), dtype=np.float64)
        probabilities[:, 0] = 1.0 - survival[:, 0]
        if len(self.classes_) > 2:
            probabilities[:, 1:-1] = survival[:, :-1] - survival[:, 1:]
        probabilities[:, -1] = survival[:, -1]
        probabilities = np.clip(probabilities, 0.0, 1.0)
        row_sums = probabilities.sum(axis=1, keepdims=True)
        return probabilities / np.maximum(row_sums, 1e-12)

    def predict(self, features: np.ndarray) -> np.ndarray:
        logits = self.threshold_logits(features)
        predicted_indices = (logits > 0.0).sum(axis=1)
        return self.classes_[predicted_indices]

    def optimization_metadata(self) -> dict[str, Any]:
        return {
            "optimizer": "AdamW",
            "learning_rate": float(self.config.learning_rate),
            "weight_decay": float(self.config.weight_decay),
            "regularization": "L2_weight_decay",
            "feature_preprocessing": "source_train_z_score",
            "batch_size": int(self.config.batch_size),
            "maximum_epochs": int(self.config.epochs),
            "epochs_ran": int(self.epochs_ran_),
            "early_stopping_metric": "validation_qwk_then_mae",
            "patience": int(self.config.patience),
            "best_epoch": self.best_epoch_,
            "best_validation_qwk": self.best_validation_qwk_,
            "best_validation_mae": self.best_validation_mae_,
            "device": self.device_,
            "loss": "unweighted_CORAL_ordinal_cumulative_binary_cross_entropy"
            if str(self.config.class_weight).lower() != "balanced"
            else "threshold_balanced_CORAL_ordinal_cumulative_binary_cross_entropy",
        }

    def _standardize(self, features: np.ndarray) -> np.ndarray:
        if self.feature_mean_ is None or self.feature_scale_ is None:
            raise RuntimeError("CORAL feature standardization parameters are not fitted")
        return ((np.asarray(features, dtype=np.float32) - self.feature_mean_) / self.feature_scale_).astype(np.float32)

    def _require_fitted(self) -> None:
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.thresholds_ is None
            or self.feature_mean_ is None
            or self.feature_scale_ is None
        ):
            raise RuntimeError("CORAL ordinal classifier is not fitted")


class PerQueryLinearJudge:
    """Frozen-feature linear Judge with explicit shared or per-query head ownership."""

    def __init__(self, config: LinearJudgeConfig | None = None) -> None:
        self.config = config or LinearJudgeConfig()
        self.models_: dict[str, Ridge | LogisticRegression | LinearSoftmaxClassifier | CoralOrdinalClassifier | None] = {}
        self.majority_by_query_: dict[str, Any] = {}
        self.residual_scale_by_query_: dict[str, float] = {}
        self.classes_: np.ndarray | None = None
        self.query_ids_: tuple[str, ...] = ()
        self.representation_: RepresentationTransform | None = None
        self.feature_shape_: tuple[int, int] | None = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
        *,
        train_mask: np.ndarray,
        validation_mask: np.ndarray | None = None,
    ) -> "PerQueryLinearJudge":
        values = _as_layers(features)
        targets = normalize_label_array(labels)
        queries = np.asarray(query_ids).astype(str)
        mask = np.asarray(train_mask, dtype=bool)
        validation = (
            np.zeros(len(values), dtype=bool)
            if validation_mask is None
            else np.asarray(validation_mask, dtype=bool)
        )
        if len(values) != len(targets) or len(values) != len(queries) or mask.shape != (len(values),):
            raise ValueError("features, labels, query_ids, and train_mask must be aligned")
        if validation.shape != mask.shape or np.any(mask & validation):
            raise ValueError("linear Judge train and validation masks must align and be disjoint")
        if not mask.any():
            raise ValueError("train_mask selects no rows")
        configured = tuple(self.config.class_values)
        self.classes_ = np.asarray(configured if configured else np.unique(targets[mask]))
        if len(self.classes_) < 2:
            raise ValueError("linear judge requires at least two classes")
        self.query_ids_ = tuple(sorted(set(queries[mask].tolist())))
        self.feature_shape_ = (int(values.shape[1]), int(values.shape[2]))
        spec = RepresentationSpec(kind=str(self.config.representation), pca_dim=int(self.config.pca_dim))
        self.representation_ = RepresentationTransform(spec, random_state=int(self.config.seed)).fit(values[mask])
        matrix = self.representation_.transform(values)
        method = str(self.config.method).lower()
        if method not in {"ridge", "linear", "logistic", "coral"}:
            raise ValueError("linear judge method must be 'ridge', 'linear', 'logistic', or 'coral'")
        head_sharing = str(self.config.head_sharing).lower()
        if head_sharing not in {"shared", "per_query"}:
            raise ValueError("linear Judge head_sharing must be 'shared' or 'per_query'")
        self.models_ = {}
        self.majority_by_query_ = {}
        self.residual_scale_by_query_ = {}
        if head_sharing == "shared":
            train_labels = targets[mask]
            values_unique, counts = np.unique(train_labels, return_counts=True)
            majority = values_unique[int(np.argmax(counts))]
            if len(values_unique) < 2:
                shared_model: Ridge | LogisticRegression | LinearSoftmaxClassifier | None = None
                residual_scale = 1.0
            elif method == "ridge":
                shared_model = Ridge(alpha=float(self.config.alpha))
                shared_model.fit(matrix[mask], train_labels.astype(float))
                residual = train_labels.astype(float) - np.asarray(
                    shared_model.predict(matrix[mask]), dtype=float
                )
                residual_scale = max(float(np.std(residual)), 0.5)
            elif method == "linear":
                shared_model = LinearSoftmaxClassifier(self.config, self.classes_).fit(
                    matrix[mask],
                    train_labels,
                    validation_features=matrix[validation] if validation.any() else None,
                    validation_labels=targets[validation] if validation.any() else None,
                )
                residual_scale = 1.0
            elif method == "coral":
                shared_model = CoralOrdinalClassifier(self.config, self.classes_).fit(
                    matrix[mask],
                    train_labels,
                    validation_features=matrix[validation] if validation.any() else None,
                    validation_labels=targets[validation] if validation.any() else None,
                )
                residual_scale = 1.0
            else:
                shared_model = LogisticRegression(
                    C=float(self.config.c),
                    max_iter=int(self.config.max_iter),
                    class_weight=self.config.class_weight,
                    random_state=int(self.config.seed),
                    solver="lbfgs",
                ).fit(matrix[mask], train_labels)
                residual_scale = 1.0
            self.models_ = {query_id: shared_model for query_id in self.query_ids_}
            self.majority_by_query_ = {query_id: majority for query_id in self.query_ids_}
            self.residual_scale_by_query_ = {
                query_id: residual_scale for query_id in self.query_ids_
            }
            return self

        for query_id in self.query_ids_:
            local_train = mask & (queries == query_id)
            local_validation = validation & (queries == query_id)
            train_labels = targets[local_train]
            values_unique, counts = np.unique(train_labels, return_counts=True)
            majority = values_unique[int(np.argmax(counts))]
            self.majority_by_query_[query_id] = majority
            if len(values_unique) < 2:
                self.models_[query_id] = None
                self.residual_scale_by_query_[query_id] = 1.0
                continue
            if method == "ridge":
                model: Ridge | LogisticRegression | LinearSoftmaxClassifier | CoralOrdinalClassifier = Ridge(
                    alpha=float(self.config.alpha)
                )
                model.fit(matrix[local_train], train_labels.astype(float))
                residual = train_labels.astype(float) - np.asarray(
                    model.predict(matrix[local_train]), dtype=float
                )
                self.residual_scale_by_query_[query_id] = max(float(np.std(residual)), 0.5)
            elif method == "linear":
                model = LinearSoftmaxClassifier(self.config, self.classes_).fit(
                    matrix[local_train],
                    train_labels,
                    validation_features=(matrix[local_validation] if local_validation.any() else None),
                    validation_labels=(targets[local_validation] if local_validation.any() else None),
                )
                self.residual_scale_by_query_[query_id] = 1.0
            elif method == "coral":
                model = CoralOrdinalClassifier(self.config, self.classes_).fit(
                    matrix[local_train],
                    train_labels,
                    validation_features=(matrix[local_validation] if local_validation.any() else None),
                    validation_labels=(targets[local_validation] if local_validation.any() else None),
                )
                self.residual_scale_by_query_[query_id] = 1.0
            else:
                model = LogisticRegression(
                    C=float(self.config.c),
                    max_iter=int(self.config.max_iter),
                    class_weight=self.config.class_weight,
                    random_state=int(self.config.seed),
                    solver="lbfgs",
                ).fit(matrix[local_train], train_labels)
                self.residual_scale_by_query_[query_id] = 1.0
            self.models_[query_id] = model
        return self

    def transform_u(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.representation_ is not None
        return self.representation_.transform(features)

    def predict(self, features: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        method = str(self.config.method).lower()
        if method in {"linear", "logistic"}:
            output = self.predict_output(features, query_ids)
            return output.classes[np.argmax(output.probabilities, axis=1)]
        self._require_fitted()
        assert self.classes_ is not None
        matrix = self.transform_u(features)
        queries = np.asarray(query_ids).astype(str)
        if len(matrix) != len(queries):
            raise ValueError("features and query_ids must be aligned")
        default = self.classes_[int(len(self.classes_) // 2)]
        out = np.asarray([self.majority_by_query_.get(query, default) for query in queries], dtype=self.classes_.dtype)
        for query_id, model in self.models_.items():
            local = queries == query_id
            if not local.any() or model is None:
                continue
            if method == "coral":
                if not isinstance(model, CoralOrdinalClassifier):
                    raise RuntimeError("CORAL Judge has no fitted CORAL classifier")
                out[local] = np.asarray(model.predict(matrix[local]))
                continue
            raw = np.asarray(model.predict(matrix[local]))
            if method == "ridge":
                raw_numeric = raw.astype(float)
                distances = np.abs(raw_numeric[:, None] - self.classes_.astype(float)[None, :])
                out[local] = self.classes_[np.argmin(distances, axis=1)]
            else:
                out[local] = raw
        return out

    def predict_proba(self, features: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        if str(self.config.method).lower() in {"linear", "logistic", "coral"}:
            return self.predict_output(features, query_ids).probabilities
        self._require_fitted()
        assert self.classes_ is not None
        matrix = self.transform_u(features)
        queries = np.asarray(query_ids).astype(str)
        probabilities = np.full((len(matrix), len(self.classes_)), 1.0 / len(self.classes_), dtype=np.float32)
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        for query_id, model in self.models_.items():
            local = queries == query_id
            if not local.any():
                continue
            if model is None:
                probabilities[local] = 0.0
                probabilities[local, class_to_index[self.majority_by_query_[query_id]]] = 1.0
            elif str(self.config.method).lower() == "ridge":
                prediction = np.asarray(model.predict(matrix[local]), dtype=float)
                scale = float(self.residual_scale_by_query_[query_id])
                logits = -0.5 * (
                    (prediction[:, None] - self.classes_.astype(float)[None, :]) / max(scale, 1e-6)
                ) ** 2
                logits -= logits.max(axis=1, keepdims=True)
                local_probability = np.exp(logits)
                probabilities[local] = local_probability / local_probability.sum(axis=1, keepdims=True)
            else:
                local_probability = np.asarray(model.predict_proba(matrix[local]), dtype=np.float32)
                mapped = np.zeros((int(local.sum()), len(self.classes_)), dtype=np.float32)
                for local_index, value in enumerate(model.classes_.tolist()):
                    mapped[:, class_to_index[value]] = local_probability[:, local_index]
                probabilities[local] = mapped
        return probabilities

    def predict_output(self, features: np.ndarray, query_ids: np.ndarray) -> JudgeHeadOutput:
        self._require_fitted()
        if str(self.config.method).lower() not in {"linear", "logistic", "coral"}:
            raise RuntimeError("Ridge is a quality-only baseline and cannot provide classification logits")
        assert self.classes_ is not None
        matrix = self.transform_u(features)
        queries = np.asarray(query_ids).astype(str)
        if len(matrix) != len(queries):
            raise ValueError("features and query_ids must be aligned")
        logits = np.full((len(matrix), len(self.classes_)), -np.inf, dtype=np.float64)
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        for query_id, model in self.models_.items():
            local = queries == query_id
            if not local.any():
                continue
            if model is None:
                logits[local, class_to_index[self.majority_by_query_[query_id]]] = 0.0
                continue
            if not isinstance(model, (LogisticRegression, LinearSoftmaxClassifier, CoralOrdinalClassifier)):
                raise RuntimeError("Only linear-softmax/CORAL heads can provide classification logits")
            if isinstance(model, CoralOrdinalClassifier):
                local_probability = np.asarray(model.predict_proba(matrix[local]), dtype=np.float64)
                logits[local] = np.log(np.clip(local_probability, 1e-30, 1.0))
                continue
            decision = np.asarray(model.decision_function(matrix[local]), dtype=np.float64)
            if decision.ndim == 1:
                decision = np.column_stack([np.zeros(len(decision), dtype=np.float64), decision])
            for local_column, class_value in enumerate(model.classes_.tolist()):
                logits[local, class_to_index[class_value]] = decision[:, local_column]
        unknown = sorted(set(queries.tolist()) - set(self.models_))
        if unknown:
            raise ValueError(f"Judge has no fitted classifier for query_ids={unknown}")
        if not np.isfinite(np.max(logits, axis=1)).all():
            raise RuntimeError("Logistic Judge did not produce a valid logit for every row")
        finite_logits = np.where(np.isneginf(logits), -1e30, logits)
        return JudgeHeadOutput(
            penultimate=matrix.astype(np.float32),
            logits=finite_logits.astype(np.float32),
            probabilities=stable_softmax(finite_logits),
            classes=self.classes_.copy(),
        )

    def to_metadata(self) -> dict[str, Any]:
        optimization = next(
            (
                model.optimization_metadata()
                for model in self.models_.values()
                if isinstance(model, (LinearSoftmaxClassifier, CoralOrdinalClassifier))
            ),
            None,
        )
        method = str(self.config.method).lower()
        if method == "coral":
            architecture = (
                "single_shared_linear_coral_ordinal"
                if str(self.config.head_sharing).lower() == "shared"
                else "per_query_linear_coral_ordinal"
            )
            objective = "l2_regularized_coral_loss"
            prediction_rule = "count_threshold_logits_above_zero"
            probability_semantics = (
                "ordinal_class_probabilities_from_cumulative_threshold_survival; "
                "argmax_probability_is_not_the_CORAL_prediction_rule"
            )
        else:
            architecture = (
                "single_shared_pure_linear_softmax"
                if str(self.config.head_sharing).lower() == "shared"
                else "per_query_pure_linear_softmax"
            )
            objective = "class_weighted_multinomial_cross_entropy"
            prediction_rule = "argmax_softmax_probability"
            probability_semantics = "softmax_class_probabilities"
        return {
            "artifact_type": "linear_judge",
            "config": self.config.to_dict(),
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "query_ids": list(self.query_ids_),
            "feature_shape": list(self.feature_shape_) if self.feature_shape_ is not None else None,
            "majority_by_query": {
                query: _json_scalar(value) for query, value in sorted(self.majority_by_query_.items())
            },
            "representation": self.representation_.to_metadata() if self.representation_ is not None else {},
            "device": optimization["device"] if optimization is not None else "cpu",
            "optimization": optimization,
            "head_contract": {
                "architecture": architecture,
                "shared_across_queries": bool(
                    str(self.config.head_sharing).lower() == "shared"
                ),
                "penultimate": "frozen_judge_representation",
                "penultimate_equals_z_a": None,
                "feature_preprocessing": "source_train_z_score" if method == "coral" else "representation_transform_only",
                "threshold_parameterization": "strictly_increasing_softplus_deltas" if method == "coral" else None,
                "logit_count": int(len(self.classes_)) if self.classes_ is not None else None,
                "threshold_count": int(len(self.classes_) - 1) if method == "coral" and self.classes_ is not None else None,
                "objective": objective,
                "class_weight": self.config.class_weight,
                "prediction_rule": prediction_rule,
                "probability_semantics": probability_semantics,
            },
        }

    def affine_head_parameters(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return exact per-query affine maps in OpenOOD's ``[Q,D,K]`` layout."""

        self._require_fitted()
        if str(self.config.method).lower() not in {"linear", "logistic"} or self.classes_ is None:
            raise RuntimeError("Exact affine parameters require the deployed logistic Judge")
        weights_by_query: list[np.ndarray] = []
        biases_by_query: list[np.ndarray] = []
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        query_ids = np.asarray(self.query_ids_, dtype=str)
        for query_id in query_ids.tolist():
            model = self.models_.get(query_id)
            if not isinstance(model, (LogisticRegression, LinearSoftmaxClassifier)):
                raise RuntimeError(
                    f"Deployed logistic Judge has no fitted affine parameters for query {query_id!r}"
                )
            input_dim = int(model.coef_.shape[1])
            weights = np.zeros((input_dim, len(self.classes_)), dtype=np.float64)
            biases = np.full(len(self.classes_), -1e30, dtype=np.float64)
            local_classes = np.asarray(model.classes_)
            if model.coef_.shape[0] == 1 and len(local_classes) == 2:
                first = class_to_index[local_classes[0]]
                second = class_to_index[local_classes[1]]
                biases[first] = 0.0
                weights[:, second] = model.coef_[0]
                biases[second] = model.intercept_[0]
            else:
                if model.coef_.shape[0] != len(local_classes):
                    raise RuntimeError("Unexpected sklearn multiclass coefficient shape")
                for row, value in enumerate(local_classes.tolist()):
                    destination = class_to_index[value]
                    weights[:, destination] = model.coef_[row]
                    biases[destination] = model.intercept_[row]
            weights_by_query.append(weights)
            biases_by_query.append(biases)
        return (
            np.stack(weights_by_query, axis=0),
            np.stack(biases_by_query, axis=0),
            query_ids,
        )

    def save(self, path: str | Path) -> str:
        self._require_fitted()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output)
        return str(output)

    def _require_fitted(self) -> None:
        if self.classes_ is None or self.representation_ is None:
            raise RuntimeError("PerQueryLinearJudge must be fitted before use")


def _linear_validation_payload(
    *,
    validation_features: np.ndarray | None,
    validation_labels: np.ndarray | None,
    classes: np.ndarray,
    class_to_index: dict[Any, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if validation_features is None and validation_labels is None:
        return None
    if validation_features is None or validation_labels is None:
        raise ValueError("Linear softmax validation features and labels must be provided together")
    matrix = np.asarray(validation_features, dtype=np.float32)
    labels = np.asarray(validation_labels)
    if matrix.ndim != 2 or len(matrix) != len(labels) or len(matrix) == 0:
        raise ValueError("Linear softmax validation data must be aligned and non-empty")
    if any(value not in class_to_index for value in labels.tolist()):
        raise ValueError("Linear softmax validation labels fall outside the configured classes")
    indices = np.asarray([class_to_index[value] for value in labels], dtype=np.int64)
    return (
        torch.as_tensor(matrix, dtype=torch.float32, device=device),
        torch.as_tensor(indices, dtype=torch.long, device=device),
    )


def _linear_validation_qwk(
    head: nn.Linear,
    validation: tuple[torch.Tensor, torch.Tensor],
    classes: np.ndarray,
) -> float:
    x, y = validation
    head.eval()
    with torch.no_grad():
        predicted_indices = torch.argmax(head(x), dim=1).detach().cpu().numpy()
    true_indices = y.detach().cpu().numpy()
    true = np.asarray(classes)[true_indices]
    predicted = np.asarray(classes)[predicted_indices]
    if len(np.unique(true)) < 2 and len(np.unique(predicted)) < 2:
        return float(np.mean(true == predicted))
    score = float(
        _quadratic_weighted_kappa(
            true,
            predicted,
            class_values=np.asarray(classes),
        )
    )
    return score if np.isfinite(score) else float(np.mean(true == predicted))


def _coral_levels(target_indices: np.ndarray, class_count: int) -> np.ndarray:
    indices = np.asarray(target_indices, dtype=np.int64)
    thresholds = np.arange(int(class_count) - 1, dtype=np.int64)
    return (indices[:, None] > thresholds[None, :]).astype(np.float32)


def _coral_validation_metrics(
    head: _CoralLinearHead,
    validation: tuple[torch.Tensor, torch.Tensor],
    classes: np.ndarray,
) -> tuple[float, float]:
    x, y = validation
    head.eval()
    with torch.no_grad():
        predicted_indices = (head(x) > 0.0).sum(dim=1).detach().cpu().numpy()
    true_indices = y.detach().cpu().numpy()
    true = np.asarray(classes)[true_indices]
    predicted = np.asarray(classes)[predicted_indices]
    mae = float(np.mean(np.abs(predicted.astype(float) - true.astype(float))))
    if len(np.unique(true)) < 2 and len(np.unique(predicted)) < 2:
        return float(np.mean(true == predicted)), mae
    score = float(
        _quadratic_weighted_kappa(
            true,
            predicted,
            class_values=np.asarray(classes),
        )
    )
    qwk = score if np.isfinite(score) else float(np.mean(true == predicted))
    return qwk, mae


def _as_layers(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"Expected [N,L,D] or [N,D], got {values.shape}")
    return values


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
