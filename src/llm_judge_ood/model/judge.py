from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from src.llm_judge_ood.model.output import JudgeHeadOutput, stable_softmax
from src.llm_judge_ood.shared.metrics import macro_query_judge_metrics, normalize_label_array


@dataclass(frozen=True)
class JudgeTrainingConfig:
    """Configuration for the frozen-feature shared judge.

    The input feature extractor and whitening are intentionally outside this
    class. This module only learns the shared representation and query heads.
    """

    hidden_dim: int = 96
    output_dim: int = 48
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    batch_size: int = 128
    patience: int = 4
    seed: int = 42
    device: str = "cpu"
    loss: str = "ce"
    class_weight: str | None = None
    class_values: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(self.class_values)
        if len(values) != len(set(values)):
            raise ValueError("class_values must not contain duplicates")
        if str(self.loss).lower() != "ce":
            raise ValueError("Only CE classification heads are supported by the deployed Judge contract")
        if self.class_weight is not None and str(self.class_weight).lower() != "balanced":
            raise ValueError("class_weight must be None or 'balanced'")
        object.__setattr__(self, "class_values", values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SharedFeatureBackbone(nn.Module):
    """Learnable layer mixing followed by a compact shared MLP."""

    def __init__(self, *, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        if num_layers < 1 or input_dim < 1:
            raise ValueError("num_layers and input_dim must be positive")
        self.layer_weights = nn.Parameter(torch.zeros(int(num_layers)))
        self.gamma = nn.Parameter(torch.ones(()))
        self.network = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(output_dim)),
            nn.LayerNorm(int(output_dim)),
        )

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"expected [batch, layers, dim], got shape {tuple(features.shape)}")
        weights = torch.softmax(self.layer_weights, dim=0)
        mixed = (features * weights.view(1, -1, 1)).sum(dim=1) * self.gamma
        return self.network(mixed)


def _class_weight_tensor(
    targets: np.ndarray,
    *,
    num_classes: int,
    mode: str | None,
    device: torch.device,
) -> Tensor | None:
    if mode is None or str(mode).lower() != "balanced":
        return None
    counts = np.bincount(np.asarray(targets, dtype=np.int64), minlength=int(num_classes)).astype(np.float32)
    present = counts > 0
    weights = np.zeros(int(num_classes), dtype=np.float32)
    if present.any():
        weights[present] = float(len(targets)) / (float(present.sum()) * counts[present])
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _classification_loss(
    logits: Tensor,
    targets: Tensor,
    loss: str,
    num_classes: int,
    class_weights: Tensor | None = None,
) -> Tensor:
    if str(loss).lower() != "ce":
        raise ValueError("Only CE classification heads are supported")
    if logits.shape[-1] != int(num_classes):
        raise ValueError(f"CE head must have {int(num_classes)} outputs, got {logits.shape[-1]}")
    return F.cross_entropy(logits, targets, weight=class_weights)


def pooled_source_space(features: np.ndarray) -> np.ndarray:
    """Mean-pool layer features for the source-space OOD detectors."""

    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"expected [samples, layers, dim], got shape {values.shape}")
    return values.mean(axis=1).astype(np.float32)


class SharedBackboneJudge:
    """Shared feature backbone with an independent head for every query."""

    def __init__(self, config: JudgeTrainingConfig | None = None) -> None:
        self.config = config or JudgeTrainingConfig()
        if str(self.config.loss).lower() != "ce":
            raise ValueError("loss must be 'ce'")
        self.backbone: SharedFeatureBackbone | None = None
        self.heads: nn.ModuleDict = nn.ModuleDict()
        self.classes_: np.ndarray | None = None
        self.query_ids_: tuple[str, ...] = ()
        self.feature_shape_: tuple[int, int] | None = None
        self.device_: torch.device | None = None
        self.history_: list[dict[str, Any]] = []
        self.best_state_: dict[str, Any] | None = None
        self.last_state_: dict[str, Any] | None = None
        self.best_epoch_: int | None = None
        self.last_epoch_: int | None = None
        self.best_validation_: dict[str, float] = {}
        self.class_weights_: Tensor | None = None

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
        *,
        train_mask: np.ndarray,
        validation_mask: np.ndarray,
    ) -> "SharedBackboneJudge":
        values = _as_layer_features(features)
        normalized_labels = normalize_label_array(labels)
        queries = np.asarray(query_ids).astype(str)
        train_mask = np.asarray(train_mask, dtype=bool)
        validation_mask = np.asarray(validation_mask, dtype=bool)
        if len(values) != len(normalized_labels) or len(values) != len(queries):
            raise ValueError("features, labels, and query_ids must have the same length")
        if train_mask.shape != (len(values),) or validation_mask.shape != (len(values),):
            raise ValueError("train_mask and validation_mask must be one-dimensional and aligned")
        if not train_mask.any():
            raise ValueError("train_mask selects no records")

        self._initialize(values, normalized_labels, queries, train_mask)
        assert self.backbone is not None
        assert self.classes_ is not None
        assert self.device_ is not None

        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        unknown_labels = sorted({value for value in normalized_labels.tolist() if value not in class_to_index}, key=str)
        if unknown_labels:
            raise ValueError(
                "labels outside the configured judge class vocabulary: "
                f"{unknown_labels}; configure JudgeTrainingConfig.class_values explicitly"
            )
        encoded = np.asarray([class_to_index[value] for value in normalized_labels], dtype=np.int64)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        self.class_weights_ = _class_weight_tensor(
            encoded[train_indices],
            num_classes=len(self.classes_),
            mode=self.config.class_weight,
            device=self.device_,
        )
        optimizer = torch.optim.AdamW(
            list(self.backbone.parameters()) + list(self.heads.parameters()),
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        generator = torch.Generator().manual_seed(int(self.config.seed))
        best_key = (float("-inf"), float("-inf"), float("-inf"))
        stale_epochs = 0
        self.history_ = []
        self.best_state_ = None
        self.last_state_ = None
        self.best_epoch_ = None
        self.last_epoch_ = None
        self.best_validation_ = {}

        x_all = torch.as_tensor(values, dtype=torch.float32, device=self.device_)
        y_all = torch.as_tensor(encoded, dtype=torch.long, device=self.device_)
        q_all = queries
        for epoch in range(max(1, int(self.config.epochs))):
            self.backbone.train()
            for head in self.heads.values():
                head.train()
            permutation = torch.randperm(len(train_indices), generator=generator).numpy()
            train_loss_total = 0.0
            train_batches = 0
            batch_size = max(1, int(self.config.batch_size))
            for start in range(0, len(permutation), batch_size):
                batch_indices = train_indices[permutation[start : start + batch_size]]
                batch_x = x_all[batch_indices]
                batch_y = y_all[batch_indices]
                batch_q = q_all[batch_indices]
                shared = self.backbone(batch_x)
                losses: list[Tensor] = []
                for query_id in sorted(set(batch_q.tolist())):
                    mask = torch.as_tensor(batch_q == query_id, dtype=torch.bool, device=self.device_)
                    head = self.heads[query_id]
                    losses.append(_classification_loss(
                        head(shared[mask]), batch_y[mask], self.config.loss,
                        len(self.classes_), self.class_weights_,
                    ))
                if not losses:
                    continue
                loss = torch.stack(losses).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.backbone.parameters()) + list(self.heads.parameters()), max_norm=5.0
                )
                optimizer.step()
                train_loss_total += float(loss.detach().cpu())
                train_batches += 1

            self.backbone.eval()
            for head in self.heads.values():
                head.eval()
            # Record train-set metrics after the parameter update as well as
            # validation metrics.  The optimizer's minibatch loss alone cannot
            # distinguish a model that has not learned from one that has
            # memorized Source Train and immediately fails on Source
            # Validation.  These metrics are diagnostic only: early stopping
            # and checkpoint selection remain validation-only below.
            train_loss = self._loss_on_indices(x_all, y_all, q_all, train_indices)
            train_probabilities = self.predict_proba(values[train_indices], queries[train_indices])
            train_predictions = self.predict(values[train_indices], queries[train_indices])
            train_metrics = macro_query_judge_metrics(
                normalized_labels[train_indices],
                train_predictions,
                queries[train_indices],
                probabilities=train_probabilities,
                class_values=self.classes_,
            )["macro"]
            validation_loss = self._loss_on_indices(x_all, y_all, q_all, validation_indices)
            if not np.isfinite(validation_loss):
                validation_loss = train_loss_total / max(train_batches, 1)
            monitor_indices = validation_indices if len(validation_indices) else train_indices
            validation_probabilities = self.predict_proba(values[monitor_indices], queries[monitor_indices])
            validation_predictions = self.predict(values[monitor_indices], queries[monitor_indices])
            validation_metrics = macro_query_judge_metrics(
                normalized_labels[monitor_indices],
                validation_predictions,
                queries[monitor_indices],
                probabilities=validation_probabilities,
                class_values=self.classes_,
            )["macro"]
            selection_key = (
                float(validation_metrics["qwk"]),
                -float(validation_metrics["mae"]),
                -float(validation_loss),
            )
            self.history_.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": train_loss_total / max(train_batches, 1),
                    "train_evaluation_loss": float(train_loss),
                    "train_macro_qwk": float(train_metrics["qwk"]),
                    "train_macro_mae": float(train_metrics["mae"]),
                    "train_macro_accuracy": float(train_metrics["accuracy"]),
                    "validation_loss": float(validation_loss),
                    "validation_macro_qwk": float(validation_metrics["qwk"]),
                    "validation_macro_mae": float(validation_metrics["mae"]),
                    "validation_macro_accuracy": float(validation_metrics["accuracy"]),
                }
            )
            self.last_state_ = {
                "backbone": deepcopy(self.backbone.state_dict()),
                "heads": deepcopy(self.heads.state_dict()),
            }
            self.last_epoch_ = int(epoch + 1)
            if _selection_key_is_better(selection_key, best_key):
                best_key = selection_key
                self.best_state_ = {
                    "backbone": deepcopy(self.backbone.state_dict()),
                    "heads": deepcopy(self.heads.state_dict()),
                }
                self.best_epoch_ = int(epoch + 1)
                self.best_validation_ = {
                    "macro_qwk": float(validation_metrics["qwk"]),
                    "macro_mae": float(validation_metrics["mae"]),
                    "loss": float(validation_loss),
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= max(1, int(self.config.patience)):
                    break

        if self.best_state_ is not None:
            self.backbone.load_state_dict(self.best_state_["backbone"])
            self.heads.load_state_dict(self.best_state_["heads"])
        self.backbone.eval()
        for head in self.heads.values():
            head.eval()
        return self

    def transform_u(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        values = _as_layer_features(features)
        assert self.backbone is not None
        assert self.device_ is not None
        self.backbone.eval()
        with torch.no_grad():
            output = self.backbone(torch.as_tensor(values, dtype=torch.float32, device=self.device_))
        return output.detach().cpu().numpy().astype(np.float32)

    def predict_output(self, features: np.ndarray, query_ids: np.ndarray) -> JudgeHeadOutput:
        self._require_fitted()
        values = _as_layer_features(features)
        queries = np.asarray(query_ids).astype(str)
        if len(values) != len(queries):
            raise ValueError("features and query_ids must have the same length")
        assert self.backbone is not None
        assert self.classes_ is not None
        assert self.device_ is not None
        shared = self._transform_tensor(values)
        logits = torch.empty((len(values), len(self.classes_)), dtype=torch.float32, device=self.device_)
        with torch.no_grad():
            for query_id in sorted(set(queries.tolist())):
                mask = torch.as_tensor(queries == query_id, dtype=torch.bool, device=self.device_)
                if query_id not in self.heads:
                    raise ValueError(f"Judge head has no fitted classifier for query_id={query_id!r}")
                logits[mask] = self.heads[query_id](shared[mask])
        h = shared.detach().cpu().numpy().astype(np.float32)
        raw_logits = logits.detach().cpu().numpy().astype(np.float32)
        return JudgeHeadOutput(
            penultimate=h,
            logits=raw_logits,
            probabilities=stable_softmax(raw_logits),
            classes=self.classes_.copy(),
        )

    def predict_proba(self, features: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        return self.predict_output(features, query_ids).probabilities

    def predict(self, features: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
        output = self.predict_output(features, query_ids)
        return output.classes[np.argmax(output.probabilities, axis=1)]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "artifact_type": "shared_backbone_per_query_judge",
            "config": self.config.to_dict(),
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "query_ids": list(self.query_ids_),
            "feature_shape": list(self.feature_shape_) if self.feature_shape_ is not None else None,
            "fitted": self.backbone is not None and self.classes_ is not None,
            "device": str(self.device_) if self.device_ is not None else None,
            "history": self.history_,
            "early_stopping_metric": "macro_per_query_validation_qwk",
            "early_stopping_tiebreakers": ["macro_per_query_validation_mae", "validation_loss"],
            "best_epoch": self.best_epoch_,
            "last_epoch": self.last_epoch_,
            "best_validation": self.best_validation_,
            "class_weights": (
                self.class_weights_.detach().cpu().tolist()
                if self.class_weights_ is not None else None
            ),
        }

    def save_checkpoints(self, output_dir: str | Path) -> dict[str, str]:
        self._require_fitted()
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for name, state, epoch in (
            ("best", self.best_state_, self.best_epoch_),
            ("last", self.last_state_, self.last_epoch_),
        ):
            if state is None:
                continue
            path = root / f"judge_{name}.pt"
            torch.save(self._checkpoint_payload(state=state, epoch=epoch, checkpoint_kind=name), path)
            paths[name] = str(path)
        return paths

    def _checkpoint_payload(
        self,
        *,
        state: dict[str, Any],
        epoch: int | None,
        checkpoint_kind: str,
    ) -> dict[str, Any]:
        return {
            "artifact_type": "shared_backbone_judge_checkpoint",
            "checkpoint_kind": checkpoint_kind,
            "epoch": epoch,
            "config": self.config.to_dict(),
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "query_ids": list(self.query_ids_),
            "feature_shape": list(self.feature_shape_) if self.feature_shape_ is not None else None,
            "backbone_state_dict": state["backbone"],
            "heads_state_dict": state["heads"],
            "best_validation": self.best_validation_,
            "history": self.history_,
        }

    def _initialize(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        queries: np.ndarray,
        train_mask: np.ndarray,
    ) -> None:
        num_layers, input_dim = int(features.shape[1]), int(features.shape[2])
        configured_classes = tuple(self.config.class_values)
        classes = np.asarray(configured_classes if configured_classes else np.unique(labels[train_mask]))
        if len(classes) < 2:
            raise ValueError("source training data must contain at least two labels")
        query_ids = tuple(sorted(set(queries[train_mask].tolist())))
        if not query_ids:
            raise ValueError("source training data must contain at least one query")
        self.classes_ = classes
        self.query_ids_ = query_ids
        self.feature_shape_ = (num_layers, input_dim)
        self.device_ = _resolve_device(self.config.device)
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if self.device_.type == "cuda":
            torch.cuda.manual_seed_all(int(self.config.seed))
        self.backbone = SharedFeatureBackbone(
            num_layers=num_layers,
            input_dim=input_dim,
            hidden_dim=int(self.config.hidden_dim),
            output_dim=int(self.config.output_dim),
        ).to(self.device_)
        self.heads = nn.ModuleDict(
            {query_id: nn.Linear(int(self.config.output_dim), len(classes)) for query_id in query_ids}
        ).to(self.device_)

    def _loss_on_indices(
        self,
        x_all: Tensor,
        y_all: Tensor,
        q_all: np.ndarray,
        indices: np.ndarray,
    ) -> float:
        if len(indices) == 0:
            return float("nan")
        assert self.backbone is not None
        assert self.classes_ is not None
        with torch.no_grad():
            shared = self.backbone(x_all[indices])
            losses: list[Tensor] = []
            local_queries = q_all[indices]
            for query_id in sorted(set(local_queries.tolist())):
                if query_id not in self.heads:
                    continue
                mask = torch.as_tensor(local_queries == query_id, dtype=torch.bool, device=self.device_)
                losses.append(
                    _classification_loss(
                        self.heads[query_id](shared[mask]),
                        y_all[indices][mask],
                        self.config.loss,
                        len(self.classes_),
                        self.class_weights_,
                    )
                )
        return float(torch.stack(losses).mean().cpu()) if losses else float("nan")

    def _transform_tensor(self, features: np.ndarray) -> Tensor:
        assert self.backbone is not None
        assert self.device_ is not None
        self.backbone.eval()
        with torch.no_grad():
            return self.backbone(torch.as_tensor(features, dtype=torch.float32, device=self.device_))

    def _require_fitted(self) -> None:
        if self.backbone is None or self.classes_ is None or self.device_ is None:
            raise RuntimeError("SharedBackboneJudge must be fitted before prediction")


def _as_layer_features(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"expected [samples, layers, dim], got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("features must contain at least one sample")
    if not np.isfinite(values).all():
        raise ValueError("features contain non-finite values")
    return values


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested.startswith("cuda") and torch.cuda.is_available():
        return torch.device(requested)
    if requested.startswith("cuda"):
        return torch.device("cpu")
    return torch.device(requested)


def _selection_key_is_better(
    candidate: tuple[float, float, float],
    incumbent: tuple[float, float, float],
    *,
    tolerance: float = 1e-8,
) -> bool:
    for candidate_value, incumbent_value in zip(candidate, incumbent, strict=True):
        if candidate_value > incumbent_value + tolerance:
            return True
        if candidate_value < incumbent_value - tolerance:
            return False
    return False
