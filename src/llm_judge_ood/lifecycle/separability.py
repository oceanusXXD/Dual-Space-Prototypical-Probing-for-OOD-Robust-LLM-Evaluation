from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SeparabilityConfig:
    """Development-only layer diagnostic for document covariate shifts."""

    enabled: bool = True
    apply_selected_layer: bool = True
    detectable_auroc: float = 0.65
    undetectable_auroc: float = 0.55
    maximum_cv_folds: int = 5
    minimum_documents_per_class: int = 2
    logistic_c: float = 1.0
    logistic_max_iter: int = 500
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.5 <= float(self.undetectable_auroc) <= float(self.detectable_auroc) <= 1.0:
            raise ValueError(
                "separability thresholds must satisfy "
                "0.5 <= undetectable_auroc <= detectable_auroc <= 1"
            )
        if int(self.maximum_cv_folds) < 2:
            raise ValueError("maximum_cv_folds must be at least two")
        if int(self.minimum_documents_per_class) < 2:
            raise ValueError("minimum_documents_per_class must be at least two")
        if float(self.logistic_c) <= 0.0 or int(self.logistic_max_iter) < 1:
            raise ValueError("separability logistic-regression parameters must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_layer_separability(
    *,
    raw_document_features: np.ndarray,
    input_document_ids: np.ndarray,
    source_validation_mask: np.ndarray,
    development_mask: np.ndarray,
    document_shift_types: np.ndarray,
    config: SeparabilityConfig | None = None,
) -> dict[str, Any]:
    """Select a monitoring layer using only labeled development environments.

    Judge rows are collapsed to one row per input document before any fit or
    metric. Source-validation documents provide the ID class; known near/far
    development shifts provide the OOD class. Deployment records are never read.
    """

    cfg = config or SeparabilityConfig()
    values = _as_layers(raw_document_features)
    document_ids = np.asarray(input_document_ids).astype(str)
    source = np.asarray(source_validation_mask, dtype=bool)
    development = np.asarray(development_mask, dtype=bool)
    shifts = np.asarray(document_shift_types).astype(str)
    if not (
        len(values)
        == len(document_ids)
        == len(source)
        == len(development)
        == len(shifts)
    ):
        raise ValueError("separability inputs must align")
    if not bool(cfg.enabled):
        return _unavailable(cfg, "disabled")

    first_indices = _unique_document_indices(
        document_ids=document_ids,
        source_mask=source,
        development_mask=development,
        shift_types=shifts,
    )
    unique_values = values[first_indices]
    unique_source = source[first_indices]
    unique_development = development[first_indices]
    unique_shifts = shifts[first_indices]
    development_shift_types = sorted(
        shift
        for shift in set(unique_shifts[unique_development].tolist())
        if not _is_id_shift(shift)
    )
    if int(unique_source.sum()) < int(cfg.minimum_documents_per_class):
        return _unavailable(cfg, "insufficient_source_validation_documents")
    if not development_shift_types:
        return _unavailable(cfg, "development_has_no_explicit_covariate_shift_types")

    evaluation_groups = ["all_covariate_shifts", *development_shift_types]
    layer_rows: list[dict[str, Any]] = []
    for layer_index in range(unique_values.shape[1]):
        by_shift: dict[str, dict[str, Any]] = {}
        for shift_type in evaluation_groups:
            target = unique_development & (
                np.isin(unique_shifts, development_shift_types)
                if shift_type == "all_covariate_shifts"
                else unique_shifts == shift_type
            )
            by_shift[shift_type] = _layer_shift_auroc(
                source_values=unique_values[unique_source, layer_index, :],
                target_values=unique_values[target, layer_index, :],
                config=cfg,
                seed=int(cfg.seed) + layer_index * 101 + len(by_shift),
            )
        available_aurocs = [
            float(row["auroc"])
            for row in by_shift.values()
            if row.get("auroc") is not None
        ]
        primary = by_shift["all_covariate_shifts"].get("auroc")
        selection_auroc = (
            float(primary)
            if primary is not None
            else float(np.mean(available_aurocs)) if available_aurocs else None
        )
        layer_rows.append(
            {
                "layer_index": int(layer_index),
                "selection_auroc": selection_auroc,
                "by_shift": by_shift,
            }
        )
    selectable = [row for row in layer_rows if row["selection_auroc"] is not None]
    if not selectable:
        return {
            **_unavailable(cfg, "insufficient_documents_for_cross_validated_auroc"),
            "layers": layer_rows,
            "development_shift_types": development_shift_types,
        }
    selected = max(
        selectable,
        key=lambda row: (float(row["selection_auroc"]), int(row["layer_index"])),
    )
    shift_summaries: dict[str, dict[str, Any]] = {}
    for shift_type in evaluation_groups:
        values_by_layer = [
            (int(row["layer_index"]), row["by_shift"][shift_type].get("auroc"))
            for row in layer_rows
        ]
        available = [(layer, float(auroc)) for layer, auroc in values_by_layer if auroc is not None]
        best_layer, best_auroc = max(available, key=lambda item: (item[1], item[0]))
        shift_summaries[shift_type] = {
            "best_layer_index": int(best_layer),
            "best_auroc": float(best_auroc),
            "detectable": bool(best_auroc >= float(cfg.detectable_auroc)),
            "declared_undetectable": bool(best_auroc < float(cfg.undetectable_auroc)),
            "routing_when_undetectable": "fixed_budget_label_safety_net",
        }
    return {
        "enabled": True,
        "available": True,
        "unavailable_reason": None,
        "config": cfg.to_dict(),
        "selected_layer_index": int(selected["layer_index"]),
        "selected_layer_auroc": float(selected["selection_auroc"]),
        "selection_metric": "out_of_fold_logistic_regression_auroc",
        "selection_scope": "training_validation_id_vs_development_known_shifts",
        "selection_used_deployment_records": False,
        "selection_used_quality_labels": False,
        "document_unit": "unique_input_document",
        "source_validation_document_count": int(unique_source.sum()),
        "development_document_count": int(unique_development.sum()),
        "development_shift_types": development_shift_types,
        "by_shift": shift_summaries,
        "layers": layer_rows,
    }


def _layer_shift_auroc(
    *,
    source_values: np.ndarray,
    target_values: np.ndarray,
    config: SeparabilityConfig,
    seed: int,
) -> dict[str, Any]:
    source = np.asarray(source_values, dtype=np.float64)
    target = np.asarray(target_values, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or source.shape[1:] != target.shape[1:]:
        raise ValueError("source and target layer features must be aligned matrices")
    minimum = int(config.minimum_documents_per_class)
    if len(source) < minimum or len(target) < minimum:
        return {
            "auroc": None,
            "source_documents": int(len(source)),
            "target_documents": int(len(target)),
            "reason": "insufficient_documents_per_class",
        }
    labels = np.concatenate(
        [np.zeros(len(source), dtype=int), np.ones(len(target), dtype=int)]
    )
    features = np.vstack([source, target])
    folds = min(int(config.maximum_cv_folds), len(source), len(target))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(seed))
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=float(config.logistic_c),
            max_iter=int(config.logistic_max_iter),
            class_weight="balanced",
            random_state=int(seed),
            solver="lbfgs",
        ),
    )
    probabilities = cross_val_predict(
        model,
        features,
        labels,
        cv=splitter,
        method="predict_proba",
    )[:, 1]
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "source_documents": int(len(source)),
        "target_documents": int(len(target)),
        "cv_folds": int(folds),
        "evaluation": "out_of_fold",
    }


def _unique_document_indices(
    *,
    document_ids: np.ndarray,
    source_mask: np.ndarray,
    development_mask: np.ndarray,
    shift_types: np.ndarray,
) -> np.ndarray:
    selected: list[int] = []
    seen: dict[str, tuple[bool, bool, str]] = {}
    for index, document_id in enumerate(document_ids.tolist()):
        state = (
            bool(source_mask[index]),
            bool(development_mask[index]),
            str(shift_types[index]),
        )
        key = str(document_id)
        previous = seen.get(key)
        if previous is not None and previous != state:
            raise ValueError(f"Input document {document_id!r} has inconsistent diagnostic roles")
        if previous is None:
            seen[key] = state
            selected.append(index)
    return np.asarray(selected, dtype=int)


def _is_id_shift(value: str) -> bool:
    return str(value).strip().lower() in {"id", "source", "training", "none"}


def _as_layers(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] == 0:
        raise ValueError("separability features must be non-empty [N,L,D] or [N,D]")
    if not np.isfinite(values).all():
        raise ValueError("separability features contain NaN or inf")
    return values


def _unavailable(config: SeparabilityConfig, reason: str) -> dict[str, Any]:
    return {
        "enabled": bool(config.enabled),
        "available": False,
        "unavailable_reason": str(reason),
        "config": config.to_dict(),
        "selected_layer_index": None,
        "selection_scope": "training_validation_id_vs_development_known_shifts",
        "selection_used_deployment_records": False,
        "selection_used_quality_labels": False,
        "document_unit": "unique_input_document",
        "layers": [],
    }
