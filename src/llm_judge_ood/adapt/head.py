from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn

from src.llm_judge_ood.model.baselines import LinearSoftmaxClassifier, PerQueryLinearJudge
from src.llm_judge_ood.model.judge import (
    SharedBackboneJudge,
    _classification_loss,
)
from src.llm_judge_ood.shared.metrics import normalize_label_array


@dataclass(frozen=True)
class HeadAdaptConfig:
    epochs: int = 25
    patience: int = 5
    deployment_validation_fraction: float = 0.2
    minimum_validation_rows: int = 2
    holdout_deployment_for_early_stopping: bool = False
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    training_replay_weight: float = 1.0
    deployment_weight: float = 1.0
    anchor_weight: float = 1e-2
    seed: int = 42
    # Retained for the explicit no-Judge fallback only.
    max_iter: int = 500
    c: float = 1.0

    def __post_init__(self) -> None:
        if int(self.epochs) < 1 or int(self.patience) < 1:
            raise ValueError("Head adaptation epochs and patience must be positive")
        if not 0.0 <= float(self.deployment_validation_fraction) < 1.0:
            raise ValueError("deployment_validation_fraction must be in [0, 1)")
        if int(self.minimum_validation_rows) < 1:
            raise ValueError("minimum_validation_rows must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeadAdapter:
    """Type-I head-only adaptation with training replay and an anchor to old heads.

    Each neural or logistic Judge head starts from the deployed per-query
    parameters. The shared backbone is never part of the optimizer, and both
    head types receive the same squared-distance anchor.
    """

    def __init__(self, config: HeadAdaptConfig | None = None) -> None:
        self.config = config or HeadAdaptConfig()
        self.models_: dict[str, LogisticRegression] = {}
        self.heads_: dict[str, nn.Module] = {}
        self.classes_: np.ndarray | None = None
        self.loss_: str | None = None
        self.device_: torch.device | None = None
        self.mode_: str = "unfitted"
        self.training_rows_: dict[str, dict[str, int]] = {}
        self.initialization_: dict[str, str] = {}
        self.optimization_: dict[str, dict[str, Any]] = {}

    def fit(
        self,
        *,
        u_features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
        deployment_indices: np.ndarray,
        training_replay_indices: np.ndarray,
        deployment_cluster_ids: np.ndarray | None = None,
        class_values: np.ndarray | None = None,
        judge: SharedBackboneJudge | PerQueryLinearJudge | None = None,
    ) -> "HeadAdapter":
        labels = normalize_label_array(labels)
        queries = np.asarray(query_ids).astype(str)
        deployment = np.asarray(deployment_indices, dtype=int)
        training_replay = np.asarray(training_replay_indices, dtype=int)
        deployment_clusters = (
            np.full(len(labels), "deployment_cluster", dtype=str)
            if deployment_cluster_ids is None
            else np.asarray(deployment_cluster_ids).astype(str)
        )
        if deployment.size == 0:
            raise ValueError("HeadAdapter needs at least one deployment Adapt row")
        if (
            np.any(deployment < 0)
            or np.any(training_replay < 0)
            or np.any(deployment >= len(labels))
            or np.any(training_replay >= len(labels))
        ):
            raise ValueError("adaptation indices are out of bounds")
        if deployment_clusters.shape != (len(labels),):
            raise ValueError("deployment_cluster_ids must align with labels")
        self.classes_ = normalize_label_array(class_values) if class_values is not None else np.unique(labels)
        self.models_ = {}
        self.heads_ = {}
        self.training_rows_ = {}
        self.initialization_ = {}
        self.optimization_ = {}
        if isinstance(judge, SharedBackboneJudge) and judge.backbone is not None and judge.classes_ is not None:
            self._fit_neural(
                u_features=np.asarray(u_features, dtype=np.float32),
                labels=labels,
                query_ids=queries,
                deployment_indices=deployment,
                training_replay_indices=training_replay,
                deployment_cluster_ids=deployment_clusters,
                judge=judge,
            )
        elif isinstance(judge, PerQueryLinearJudge):
            self._fit_linear_copy(
                u_features=np.asarray(u_features, dtype=np.float32),
                labels=labels,
                query_ids=queries,
                deployment_indices=deployment,
                training_replay_indices=training_replay,
                deployment_cluster_ids=deployment_clusters,
                judge=judge,
            )
        else:
            self._fit_linear(
                u_features=np.asarray(u_features, dtype=np.float32),
                labels=labels,
                query_ids=queries,
                deployment_indices=deployment,
                training_replay_indices=training_replay,
            )
        return self

    def _fit_neural(
        self,
        *,
        u_features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
        deployment_indices: np.ndarray,
        training_replay_indices: np.ndarray,
        deployment_cluster_ids: np.ndarray,
        judge: SharedBackboneJudge,
    ) -> None:
        assert judge.classes_ is not None
        self.classes_ = np.asarray(judge.classes_)
        self.loss_ = str(judge.config.loss).lower()
        self.device_ = next(judge.heads.parameters()).device
        self.mode_ = "neural_head_copy"
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        if self.device_.type == "cuda":
            torch.cuda.manual_seed_all(int(self.config.seed))
        rng = np.random.default_rng(int(self.config.seed))

        deployment_queries = sorted(set(query_ids[deployment_indices].tolist()))
        for query_id in deployment_queries:
            if query_id not in judge.heads:
                self.training_rows_[query_id] = {"training_replay": 0, "deployment": 0}
                self.initialization_[query_id] = "missing_training_head"
                continue
            training_replay = training_replay_indices[query_ids[training_replay_indices] == query_id]
            deployment = deployment_indices[query_ids[deployment_indices] == query_id]
            usable_training_replay = np.asarray(
                [index for index in training_replay if labels[index] in class_to_index], dtype=int
            )
            usable_deployment = np.asarray(
                [index for index in deployment if labels[index] in class_to_index], dtype=int
            )
            deployment_train, deployment_validation = (
                _split_deployment_train_validation(
                    usable_deployment,
                    cluster_ids=deployment_cluster_ids,
                    validation_fraction=float(self.config.deployment_validation_fraction),
                    minimum_validation_rows=int(self.config.minimum_validation_rows),
                    rng=rng,
                )
                if bool(self.config.holdout_deployment_for_early_stopping)
                else (usable_deployment.copy(), np.zeros(0, dtype=int))
            )
            usable_training_replay = usable_training_replay[: len(deployment_train)]
            self.training_rows_[query_id] = {
                "training_replay": int(usable_training_replay.size),
                "deployment": int(usable_deployment.size),
                "deployment_train": int(deployment_train.size),
                "deployment_validation": int(deployment_validation.size),
            }
            if deployment_train.size == 0:
                self.initialization_[query_id] = "missing_usable_deployment_labels"
                continue

            head = deepcopy(judge.heads[query_id]).to(self.device_)
            self.optimization_[query_id] = _optimize_copied_head(
                head=head,
                u_features=u_features,
                labels=labels,
                class_to_index=class_to_index,
                source_indices=usable_training_replay,
                deployment_train_indices=deployment_train,
                deployment_validation_indices=deployment_validation,
                loss_name=self.loss_,
                class_count=len(self.classes_),
                config=self.config,
                device=self.device_,
                validation_cluster_ids=deployment_cluster_ids[deployment_validation],
            )
            self.heads_[query_id] = head
            self.initialization_[query_id] = "copied_deployed_head"

    def _fit_linear_copy(
        self,
        *,
        u_features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
        deployment_indices: np.ndarray,
        training_replay_indices: np.ndarray,
        deployment_cluster_ids: np.ndarray,
        judge: PerQueryLinearJudge,
    ) -> None:
        """Adapt deployed logistic heads from their exact coefficients with anchoring."""

        if str(judge.config.method).lower() not in {"linear", "logistic"} or judge.classes_ is None:
            raise ValueError("Head-only adaptation requires a fitted linear-softmax Judge")
        self.classes_ = np.asarray(judge.classes_)
        self.loss_ = "ce"
        self.device_ = torch.device("cpu")
        self.mode_ = "linear_head_copy"
        class_to_index = {value: index for index, value in enumerate(self.classes_.tolist())}
        random.seed(int(self.config.seed))
        np.random.seed(int(self.config.seed))
        torch.manual_seed(int(self.config.seed))
        rng = np.random.default_rng(int(self.config.seed))

        for query_id in sorted(set(query_ids[deployment_indices].tolist())):
            model = judge.models_.get(query_id)
            training_replay = training_replay_indices[query_ids[training_replay_indices] == query_id]
            deployment = deployment_indices[query_ids[deployment_indices] == query_id]
            usable_training_replay = np.asarray(
                [index for index in training_replay if labels[index] in class_to_index], dtype=int
            )
            usable_deployment = np.asarray(
                [index for index in deployment if labels[index] in class_to_index], dtype=int
            )
            deployment_train, deployment_validation = (
                _split_deployment_train_validation(
                    usable_deployment,
                    cluster_ids=deployment_cluster_ids,
                    validation_fraction=float(self.config.deployment_validation_fraction),
                    minimum_validation_rows=int(self.config.minimum_validation_rows),
                    rng=rng,
                )
                if bool(self.config.holdout_deployment_for_early_stopping)
                else (usable_deployment.copy(), np.zeros(0, dtype=int))
            )
            usable_training_replay = usable_training_replay[: len(deployment_train)]
            self.training_rows_[query_id] = {
                "training_replay": int(usable_training_replay.size),
                "deployment": int(usable_deployment.size),
                "deployment_train": int(deployment_train.size),
                "deployment_validation": int(deployment_validation.size),
            }
            if deployment_train.size == 0:
                self.initialization_[query_id] = "missing_usable_deployment_labels"
                continue
            if not isinstance(model, (LogisticRegression, LinearSoftmaxClassifier)):
                self.initialization_[query_id] = "missing_deployed_logistic_head"
                continue

            head = _copy_logistic_head(
                model,
                classes=self.classes_,
                input_dim=int(u_features.shape[1]),
            ).to(self.device_)
            self.optimization_[query_id] = _optimize_copied_head(
                head=head,
                u_features=u_features,
                labels=labels,
                class_to_index=class_to_index,
                source_indices=usable_training_replay,
                deployment_train_indices=deployment_train,
                deployment_validation_indices=deployment_validation,
                loss_name="ce",
                class_count=len(self.classes_),
                config=self.config,
                device=self.device_,
                validation_cluster_ids=deployment_cluster_ids[deployment_validation],
            )
            self.heads_[query_id] = head
            self.initialization_[query_id] = "copied_deployed_logistic_head"

    def _fit_linear(
        self,
        *,
        u_features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
        deployment_indices: np.ndarray,
        training_replay_indices: np.ndarray,
    ) -> None:
        self.mode_ = "linear_refit_fallback"
        self.loss_ = None
        train_indices = np.concatenate([training_replay_indices, deployment_indices])
        weights = np.concatenate(
            [
                np.full(len(training_replay_indices), float(self.config.training_replay_weight), dtype=float),
                np.full(len(deployment_indices), float(self.config.deployment_weight), dtype=float),
            ]
        )
        for query_id in sorted(set(query_ids[deployment_indices].tolist())):
            training_mask = query_ids[training_replay_indices] == query_id
            deployment_mask = query_ids[deployment_indices] == query_id
            local = train_indices[query_ids[train_indices] == query_id]
            local_weights = weights[query_ids[train_indices] == query_id]
            self.training_rows_[query_id] = {
                "training_replay": int(training_mask.sum()),
                "deployment": int(deployment_mask.sum()),
            }
            self.initialization_[query_id] = "linear_refit_no_head_state"
            if len(local) < 2 or len(np.unique(labels[local])) < 2:
                continue
            model = LogisticRegression(
                max_iter=int(self.config.max_iter),
                C=float(self.config.c),
                random_state=int(self.config.seed),
            )
            model.fit(u_features[local], labels[local], sample_weight=local_weights)
            self.models_[query_id] = model

    def predict(self, *, u_features: np.ndarray, query_ids: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_ids).astype(str)
        out = np.asarray(fallback).copy()
        if self.mode_ in {"neural_head_copy", "linear_head_copy"}:
            assert self.classes_ is not None
            assert self.device_ is not None
            values = torch.as_tensor(np.asarray(u_features, dtype=np.float32), dtype=torch.float32, device=self.device_)
            with torch.no_grad():
                for query_id, head in self.heads_.items():
                    mask = queries == query_id
                    if not mask.any():
                        continue
                    local = torch.as_tensor(mask, dtype=torch.bool, device=self.device_)
                    logits = head(values[local])
                    out[mask] = self.classes_[torch.argmax(logits, dim=1).detach().cpu().numpy()]
            return out
        for query_id, model in self.models_.items():
            mask = queries == query_id
            if mask.any():
                out[mask] = model.predict(np.asarray(u_features, dtype=np.float32)[mask])
        return out

    def predict_proba(self, *, u_features: np.ndarray, query_ids: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_ids).astype(str)
        out = np.asarray(fallback, dtype=np.float32).copy()
        if self.classes_ is None or len(self.classes_) != out.shape[1]:
            return out
        if self.mode_ in {"neural_head_copy", "linear_head_copy"}:
            assert self.device_ is not None
            values = torch.as_tensor(np.asarray(u_features, dtype=np.float32), dtype=torch.float32, device=self.device_)
            with torch.no_grad():
                for query_id, head in self.heads_.items():
                    mask = queries == query_id
                    if not mask.any():
                        continue
                    local = torch.as_tensor(mask, dtype=torch.bool, device=self.device_)
                    logits = head(values[local])
                    probabilities = torch.softmax(logits, dim=1)
                    out[mask] = probabilities.detach().cpu().numpy().astype(np.float32)
            return out
        global_columns = {value: index for index, value in enumerate(self.classes_.tolist())}
        for query_id, model in self.models_.items():
            mask = queries == query_id
            if not mask.any():
                continue
            proba = model.predict_proba(np.asarray(u_features, dtype=np.float32)[mask])
            mapped = np.zeros((int(mask.sum()), out.shape[1]), dtype=np.float32)
            for local_column, class_value in enumerate(model.classes_.tolist()):
                global_column = global_columns.get(class_value)
                if global_column is not None:
                    mapped[:, global_column] = proba[:, local_column]
            row_sums = mapped.sum(axis=1, keepdims=True)
            valid = row_sums[:, 0] > 0.0
            if valid.any():
                mapped[valid] /= row_sums[valid]
                local_rows = np.flatnonzero(mask)
                out[local_rows[valid]] = mapped[valid]
        return out

    def predict_logits(self, *, u_features: np.ndarray, query_ids: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Return full-class logits from adapted copied heads, with old-head fallback."""

        queries = np.asarray(query_ids).astype(str)
        out = np.asarray(fallback, dtype=np.float32).copy()
        if self.mode_ not in {"neural_head_copy", "linear_head_copy"}:
            return out
        if self.classes_ is None or len(self.classes_) != out.shape[1] or self.device_ is None:
            return out
        values = torch.as_tensor(
            np.asarray(u_features, dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
        with torch.no_grad():
            for query_id, head in self.heads_.items():
                mask = queries == query_id
                if not mask.any():
                    continue
                local = torch.as_tensor(mask, dtype=torch.bool, device=self.device_)
                out[mask] = head(values[local]).detach().cpu().numpy().astype(np.float32)
        return out

    def save_checkpoint(self, path: str | Path) -> str:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if self.mode_ not in {"neural_head_copy", "linear_head_copy"}:
            raise RuntimeError("Only deployed-head-copy adapters have a torch checkpoint")
        torch.save(
            {
                "artifact_type": "llm_judge_ood_head_adapter",
                "config": self.config.to_dict(),
                "mode": self.mode_,
                "loss": self.loss_,
                "classes": self.classes_.tolist() if self.classes_ is not None else [],
                "heads_state_dict": {query: head.state_dict() for query, head in self.heads_.items()},
                "training_rows": self.training_rows_,
                "initialization": self.initialization_,
                "optimization": self.optimization_,
            },
            output,
        )
        return str(output)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "mode": self.mode_,
            "adapted_query_ids": sorted(set(self.models_) | set(self.heads_)),
            "classes": self.classes_.tolist() if self.classes_ is not None else [],
            "training_rows_by_query": self.training_rows_,
            "initialization_by_query": self.initialization_,
            "optimization_by_query": self.optimization_,
            "early_stopping": {
                "metric": (
                    "deployment_cluster_validation_class_balanced_loss"
                    if bool(self.config.holdout_deployment_for_early_stopping)
                    else None
                ),
                "scope": (
                    "held_out_within_confirmed_harmful_probe"
                    if bool(self.config.holdout_deployment_for_early_stopping)
                    else "disabled_all_confirmed_harmful_probe_labels_used_for_fixed_epochs"
                ),
                "gate_rows_used": False,
            },
            "anchor": (
                "sum_over_parameters_squared_distance_to_deployed_head"
                if self.mode_ in {"neural_head_copy", "linear_head_copy"}
                else None
            ),
            "objective": (
                "training_replay_weight*source_loss + deployment_weight*deployment_loss "
                "+ anchor_weight*sum((theta-theta0)^2)"
            ),
            "loss_balancing": "class_balanced_terms_with_documented_source_and_deployment_weights",
        }


def _split_deployment_train_validation(
    deployment_indices: np.ndarray,
    *,
    cluster_ids: np.ndarray,
    validation_fraction: float,
    minimum_validation_rows: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(deployment_indices, dtype=int)
    if values.size == 0 or float(validation_fraction) <= 0.0:
        return values.copy(), np.zeros(0, dtype=int)
    train: list[int] = []
    validation: list[int] = []
    for cluster_id in sorted(set(np.asarray(cluster_ids)[values].astype(str).tolist())):
        local = values[np.asarray(cluster_ids)[values].astype(str) == cluster_id]
        if local.size <= int(minimum_validation_rows):
            train.extend(local.tolist())
            continue
        requested = max(
            int(minimum_validation_rows),
            int(round(float(validation_fraction) * int(local.size))),
        )
        validation_count = min(requested, int(local.size) - 1)
        order = rng.permutation(local)
        validation.extend(order[:validation_count].tolist())
        train.extend(order[validation_count:].tolist())
    return np.asarray(train, dtype=int), np.asarray(validation, dtype=int)


def _optimize_copied_head(
    *,
    head: nn.Module,
    u_features: np.ndarray,
    labels: np.ndarray,
    class_to_index: dict[Any, int],
    source_indices: np.ndarray,
    deployment_train_indices: np.ndarray,
    deployment_validation_indices: np.ndarray,
    loss_name: str,
    class_count: int,
    config: HeadAdaptConfig,
    device: torch.device,
    validation_cluster_ids: np.ndarray,
) -> dict[str, Any]:
    source = np.asarray(source_indices, dtype=int)
    deployment_train = np.asarray(deployment_train_indices, dtype=int)
    deployment_validation = np.asarray(deployment_validation_indices, dtype=int)
    indices = np.concatenate([source, deployment_train])
    x = torch.as_tensor(u_features[indices], dtype=torch.float32, device=device)
    y = torch.as_tensor(
        [class_to_index[labels[index]] for index in indices],
        dtype=torch.long,
        device=device,
    )
    source_count = int(source.size)
    source_mask = torch.zeros(len(indices), dtype=torch.bool, device=device)
    source_mask[:source_count] = True
    deployment_mask = ~source_mask
    validation_x = (
        torch.as_tensor(u_features[deployment_validation], dtype=torch.float32, device=device)
        if deployment_validation.size
        else None
    )
    validation_y = (
        torch.as_tensor(
            [class_to_index[labels[index]] for index in deployment_validation],
            dtype=torch.long,
            device=device,
        )
        if deployment_validation.size
        else None
    )
    anchor = {name: value.detach().clone() for name, value in head.named_parameters()}
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_loss = float("inf")
    best_epoch: int | None = None
    stale_epochs = 0
    epochs_ran = 0
    for epoch in range(max(1, int(config.epochs))):
        head.train()
        logits = head(x)
        data_loss = torch.zeros((), dtype=logits.dtype, device=device)
        if bool(source_mask.any()):
            source_loss = (
                _class_balanced_classification_loss(
                    logits[source_mask], y[source_mask], loss_name, class_count
                )
            )
            data_loss = data_loss + float(config.training_replay_weight) * source_loss
        deployment_loss = _class_balanced_classification_loss(
            logits[deployment_mask], y[deployment_mask], loss_name, class_count
        )
        data_loss = data_loss + float(config.deployment_weight) * deployment_loss
        loss = data_loss
        if float(config.anchor_weight) > 0.0:
            anchor_penalty = sum(
                (parameter - anchor[name]).pow(2).sum()
                for name, parameter in head.named_parameters()
            )
            loss = loss + float(config.anchor_weight) * anchor_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=5.0)
        optimizer.step()
        epochs_ran = epoch + 1
        if validation_x is None or validation_y is None:
            continue
        head.eval()
        with torch.no_grad():
            validation_loss = float(
                _class_balanced_classification_loss(
                    head(validation_x), validation_y, loss_name, class_count
                ).item()
            )
        if validation_loss < best_validation_loss - 1e-12:
            best_validation_loss = validation_loss
            best_epoch = epoch + 1
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in head.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.patience):
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return {
        "optimizer": "AdamW",
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "anchor_weight": float(config.anchor_weight),
        "objective": (
            "training_replay_weight*source_loss + deployment_weight*deployment_loss "
            "+ anchor_weight*sum((theta-theta0)^2)"
            + (
                " + optimizer_weight_decay*sum(theta^2)"
                if float(config.weight_decay) > 0.0
                else ""
            )
        ),
        "maximum_epochs": int(config.epochs),
        "epochs_ran": int(epochs_ran),
        "patience": int(config.patience),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_deployment_validation_loss": (
            float(best_validation_loss) if np.isfinite(best_validation_loss) else None
        ),
        "early_stopping_metric": (
            "deployment_cluster_validation_class_balanced_loss"
            if bool(config.holdout_deployment_for_early_stopping)
            else None
        ),
        "validation_scope": (
            "held_out_within_confirmed_harmful_probe"
            if bool(config.holdout_deployment_for_early_stopping)
            else "disabled_all_confirmed_harmful_probe_labels_used_for_fixed_epochs"
        ),
        "validation_clusters": sorted(set(np.asarray(validation_cluster_ids).astype(str).tolist())),
        "gate_rows_used": False,
        "source_replay_indices": source.astype(int).tolist(),
        "deployment_train_indices": deployment_train.astype(int).tolist(),
        "deployment_validation_indices": deployment_validation.astype(int).tolist(),
        "source_replay_to_deployment_train_ratio": float(source.size / max(deployment_train.size, 1)),
    }


def _copy_logistic_head(
    model: LogisticRegression | LinearSoftmaxClassifier,
    *,
    classes: np.ndarray,
    input_dim: int,
) -> nn.Linear:
    """Represent sklearn's per-query logistic logits as a K-class torch head."""

    if model.coef_.shape[1] != int(input_dim):
        raise ValueError("deployed logistic head does not match the supplied Judge penultimate dimension")
    global_index = {value: index for index, value in enumerate(np.asarray(classes).tolist())}
    head = nn.Linear(int(input_dim), len(classes), bias=True)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.fill_(-30.0)
        local_classes = np.asarray(model.classes_)
        if model.coef_.shape[0] == 1 and len(local_classes) == 2:
            first = global_index.get(local_classes[0])
            second = global_index.get(local_classes[1])
            if first is None or second is None:
                raise ValueError("deployed logistic head contains classes outside the Judge vocabulary")
            head.bias[int(first)] = 0.0
            head.weight[int(first)].zero_()
            head.weight[int(second)].copy_(torch.as_tensor(model.coef_[0], dtype=torch.float32))
            head.bias[int(second)] = float(model.intercept_[0])
        else:
            if model.coef_.shape[0] != len(local_classes):
                raise ValueError("unexpected multiclass logistic coefficient shape")
            for row, value in enumerate(local_classes.tolist()):
                destination = global_index.get(value)
                if destination is None:
                    raise ValueError("deployed logistic head contains classes outside the Judge vocabulary")
                head.weight[int(destination)].copy_(torch.as_tensor(model.coef_[row], dtype=torch.float32))
                head.bias[int(destination)] = float(model.intercept_[row])
    return head


def _class_balanced_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_name: str,
    num_classes: int,
) -> torch.Tensor:
    if str(loss_name).lower() != "ce":
        return _classification_loss(logits, targets, loss_name, num_classes)
    counts = torch.bincount(targets, minlength=int(num_classes)).to(dtype=logits.dtype)
    present = counts > 0
    weights = torch.zeros(int(num_classes), dtype=logits.dtype, device=logits.device)
    weights[present] = targets.numel() / (
        present.sum().to(dtype=logits.dtype) * counts[present]
    )
    return nn.functional.cross_entropy(logits, targets, weight=weights)


class NewQueryHeadTrainer:
    """Type-II baseline: fit a scratch linear head on frozen shared features."""

    def __init__(self, config: HeadAdaptConfig | None = None) -> None:
        self.config = config or HeadAdaptConfig()
        self.models_: dict[str, LogisticRegression] = {}
        self.training_rows_: dict[str, int] = {}

    def fit(
        self,
        *,
        u_features: np.ndarray,
        labels: np.ndarray,
        query_ids: np.ndarray,
        new_query_indices: np.ndarray,
    ) -> "NewQueryHeadTrainer":
        values = normalize_label_array(labels)
        queries = np.asarray(query_ids).astype(str)
        indices = np.asarray(new_query_indices, dtype=int)
        self.models_ = {}
        self.training_rows_ = {}
        for query_id in sorted(set(queries[indices].tolist())):
            local = indices[queries[indices] == query_id]
            self.training_rows_[query_id] = int(len(local))
            if len(local) < 2 or len(np.unique(values[local])) < 2:
                continue
            model = LogisticRegression(
                max_iter=int(self.config.max_iter),
                C=float(self.config.c),
                random_state=int(self.config.seed),
            )
            model.fit(np.asarray(u_features, dtype=np.float32)[local], values[local])
            self.models_[query_id] = model
        return self

    def predict(self, *, u_features: np.ndarray, query_ids: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        queries = np.asarray(query_ids).astype(str)
        out = np.asarray(fallback).copy()
        for query_id, model in self.models_.items():
            mask = queries == query_id
            if mask.any():
                out[mask] = model.predict(np.asarray(u_features, dtype=np.float32)[mask])
        return out

    def to_metadata(self) -> dict[str, Any]:
        return {
            "path": "type2_scratch_linear_head",
            "config": self.config.to_dict(),
            "trained_query_ids": sorted(self.models_),
            "training_rows_by_query": self.training_rows_,
        }
