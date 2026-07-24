from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from src.algorithm.classifier.base import LinearJudgeConfig, PerQueryLinearJudge
from src.algorithm.classifier.train import JudgeTrainingConfig, SharedBackboneJudge
from src.algorithm.classifier.output import JudgeHeadOutput
from src.common.metrics import macro_query_judge_metrics, normalize_label_array
from src.common.stats import LayerPreprocessor


@dataclass(frozen=True)
class JudgeSelectionConfig:
    preprocess_methods: tuple[str, ...] = ("pca_whiten",)
    neural_losses: tuple[str, ...] = ("ce",)
    neural_seeds: tuple[int, ...] = (13, 42, 73)
    ridge_alphas: tuple[float, ...] = (1.0, 10.0, 100.0)
    linear_learning_rates: tuple[float, ...] = (1e-3,)
    linear_cs: tuple[float, ...] = ()
    baseline_representation: str = "last_layer"
    pca_dim: int = 48
    include_linear: bool = True
    include_neural_ablation: bool = False
    deployment_policy: str = "linear_specification"
    force_neural: bool = False
    linear_head_sharing: str = "shared"

    def __post_init__(self) -> None:
        if str(self.deployment_policy) not in {
            "linear_specification",
            "performance_selection",
            "neural_ablation",
        }:
            raise ValueError(
                "deployment_policy must be 'linear_specification', "
                "'performance_selection', or 'neural_ablation'"
            )
        if bool(self.force_neural) and str(self.deployment_policy) == "linear_specification":
            raise ValueError(
                "force_neural is incompatible with deployment_policy='linear_specification'; "
                "use deployment_policy='neural_ablation'"
            )
        if str(self.linear_head_sharing).lower() not in {"shared", "per_query"}:
            raise ValueError("linear_head_sharing must be 'shared' or 'per_query'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeSelectionResult:
    model: SharedBackboneJudge | PerQueryLinearJudge
    preprocessor: LayerPreprocessor
    processed_features: np.ndarray
    selected_candidate: dict[str, Any]
    summary: dict[str, Any]
    ensemble_predictions: np.ndarray
    ensemble_candidate_names: tuple[str, ...]
    representative_neural_model: SharedBackboneJudge | None = None
    candidate_models: dict[str, SharedBackboneJudge | PerQueryLinearJudge] = field(
        default_factory=dict,
        repr=False,
    )
    preprocessors_by_method: dict[str, LayerPreprocessor] = field(
        default_factory=dict,
        repr=False,
    )

    @property
    def is_neural(self) -> bool:
        return isinstance(self.model, SharedBackboneJudge)

    @property
    def classes_(self) -> np.ndarray:
        assert self.model.classes_ is not None
        return self.model.classes_

    def predict(self, query_ids: np.ndarray) -> np.ndarray:
        return self.model.predict(self.processed_features, query_ids)

    def predict_proba(self, query_ids: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.processed_features, query_ids)

    def predict_output(self, query_ids: np.ndarray) -> JudgeHeadOutput:
        return self.model.predict_output(self.processed_features, query_ids)

    def transform_u(self) -> np.ndarray:
        return self.model.transform_u(self.processed_features)

    def refreshed(
        self,
        raw_features: np.ndarray,
        query_ids: np.ndarray,
    ) -> "JudgeSelectionResult":
        """Apply cached fitted candidates to the current full record matrix."""

        if not self.candidate_models or not self.preprocessors_by_method:
            raise RuntimeError("Cached Judge selection is missing fitted candidate state")
        values = _as_layers(raw_features)
        queries = np.asarray(query_ids).astype(str)
        if len(values) != len(queries):
            raise ValueError("Judge refresh features and query IDs must align")
        processed_by_method = {
            method: preprocessor.transform(values)
            for method, preprocessor in self.preprocessors_by_method.items()
        }
        candidates = {
            str(row["name"]): str(row["preprocess_method"])
            for row in self.summary.get("candidate_results", [])
        }
        missing = sorted(set(self.ensemble_candidate_names) - set(candidates))
        if missing:
            raise RuntimeError(f"Cached Judge selection is missing candidate metadata: {missing}")
        ensemble = np.column_stack(
            [
                self.candidate_models[name].predict(
                    processed_by_method[candidates[name]],
                    queries,
                )
                for name in self.ensemble_candidate_names
            ]
        )
        selected_method = str(self.selected_candidate["preprocess_method"])
        return replace(
            self,
            processed_features=processed_by_method[selected_method],
            ensemble_predictions=ensemble,
        )

    def save_model_artifacts(self, output_dir: str | Path) -> dict[str, str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        if isinstance(self.model, SharedBackboneJudge):
            for kind, path in self.model.save_checkpoints(root / "selected_neural").items():
                paths[f"selected_{kind}"] = path
        else:
            paths["selected"] = self.model.save(root / "selected_linear_judge.joblib")
        if self.representative_neural_model is not None and self.representative_neural_model is not self.model:
            for kind, path in self.representative_neural_model.save_checkpoints(
                root / "best_neural_candidate"
            ).items():
                paths[f"best_neural_candidate_{kind}"] = path
        return paths

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate,
            "preprocessor": self.preprocessor.to_metadata(),
            "model": self.model.to_metadata(),
            "selection": self.summary,
            "agreement_ensemble": {
                "candidate_count": int(self.ensemble_predictions.shape[1]),
                "candidate_names": list(self.ensemble_candidate_names),
            },
        }


def select_source_judge(
    *,
    raw_features: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    base_config: JudgeTrainingConfig,
    selection_config: JudgeSelectionConfig | None = None,
) -> JudgeSelectionResult:
    selection = selection_config or JudgeSelectionConfig()
    if any(str(loss).lower() != "ce" for loss in selection.neural_losses):
        raise ValueError("CORN is not supported by the deployed K-class Judge and post-hoc detector contract")
    values = _as_layers(raw_features)
    targets = normalize_label_array(labels)
    queries = np.asarray(query_ids).astype(str)
    train = np.asarray(train_mask, dtype=bool)
    validation = np.asarray(validation_mask, dtype=bool)
    if not train.any() or not validation.any():
        raise ValueError("Judge selection requires non-empty Source Train and Source Validation masks")
    if not (len(values) == len(targets) == len(queries) == len(train) == len(validation)):
        raise ValueError("Judge selection inputs must be aligned")

    preprocessor_by_method: dict[str, LayerPreprocessor] = {}
    processed_by_method: dict[str, np.ndarray] = {}
    for method in selection.preprocess_methods:
        normalized_method = str(method).lower()
        preprocessor = LayerPreprocessor(
            method=normalized_method,
            pca_components=int(selection.pca_dim),
            random_state=int(base_config.seed),
        ).fit(values[train])
        preprocessor_by_method[normalized_method] = preprocessor
        processed_by_method[normalized_method] = preprocessor.transform(values)

    majority = _majority_baseline(targets, queries, train, validation, base_config.class_values)
    candidate_rows: list[dict[str, Any]] = []
    candidate_models: dict[str, SharedBackboneJudge | PerQueryLinearJudge] = {}

    for method, processed in processed_by_method.items():
        for alpha in selection.ridge_alphas:
            config = LinearJudgeConfig(
                method="ridge",
                alpha=float(alpha),
                representation=str(selection.baseline_representation),
                pca_dim=int(selection.pca_dim),
                class_values=tuple(base_config.class_values),
                seed=int(base_config.seed),
                head_sharing=str(selection.linear_head_sharing),
            )
            model = PerQueryLinearJudge(config).fit(
                processed,
                targets,
                queries,
                train_mask=train,
            )
            name = f"ridge::{method}::alpha={float(alpha):g}"
            row = _evaluate_candidate(
                name=name,
                family="ridge",
                preprocess_method=method,
                seed=int(base_config.seed),
                model=model,
                features=processed,
                labels=targets,
                query_ids=queries,
                validation_mask=validation,
                config=config.to_dict(),
            )
            candidate_rows.append(row)
            candidate_models[name] = model
        if selection.include_linear:
            for learning_rate in selection.linear_learning_rates:
                config = LinearJudgeConfig(
                    method="linear",
                    representation=str(selection.baseline_representation),
                    pca_dim=int(selection.pca_dim),
                    class_values=tuple(base_config.class_values),
                    seed=int(base_config.seed),
                    learning_rate=float(learning_rate),
                    weight_decay=float(base_config.weight_decay),
                    epochs=int(base_config.epochs),
                    batch_size=int(base_config.batch_size),
                    patience=int(base_config.patience),
                    device=str(base_config.device),
                    head_sharing=str(selection.linear_head_sharing),
                )
                model = PerQueryLinearJudge(config).fit(
                    processed,
                    targets,
                    queries,
                    train_mask=train,
                    validation_mask=validation,
                )
                name = f"linear::{method}::lr={float(learning_rate):g}"
                row = _evaluate_candidate(
                    name=name,
                    family="linear",
                    preprocess_method=method,
                    seed=int(base_config.seed),
                    model=model,
                    features=processed,
                    labels=targets,
                    query_ids=queries,
                    validation_mask=validation,
                    config=config.to_dict(),
                )
                candidate_rows.append(row)
                candidate_models[name] = model
            for c_value in selection.linear_cs:
                config = LinearJudgeConfig(
                    method="logistic",
                    c=float(c_value),
                    max_iter=500,
                    representation=str(selection.baseline_representation),
                    pca_dim=int(selection.pca_dim),
                    class_values=tuple(base_config.class_values),
                    seed=int(base_config.seed),
                    class_weight="balanced",
                    head_sharing=str(selection.linear_head_sharing),
                )
                model = PerQueryLinearJudge(config).fit(
                    processed,
                    targets,
                    queries,
                    train_mask=train,
                )
                name = f"logistic::{method}::c={float(c_value):g}"
                row = _evaluate_candidate(
                    name=name,
                    family="linear",
                    preprocess_method=method,
                    seed=int(base_config.seed),
                    model=model,
                    features=processed,
                    labels=targets,
                    query_ids=queries,
                    validation_mask=validation,
                    config=config.to_dict(),
                )
                candidate_rows.append(row)
                candidate_models[name] = model

    neural_groups: dict[str, list[dict[str, Any]]] = {}
    if bool(selection.include_neural_ablation):
        for method, processed in processed_by_method.items():
            for loss_name in selection.neural_losses:
                group_name = f"neural_ablation::{method}::{str(loss_name).lower()}"
                neural_groups[group_name] = []
                for seed in selection.neural_seeds:
                    config = replace(base_config, loss=str(loss_name).lower(), seed=int(seed))
                    model = SharedBackboneJudge(config).fit(
                        processed,
                        targets,
                        queries,
                        train_mask=train,
                        validation_mask=validation,
                    )
                    name = f"{group_name}::seed={int(seed)}"
                    row = _evaluate_candidate(
                        name=name,
                        family="neural_ablation",
                        preprocess_method=method,
                        seed=int(seed),
                        model=model,
                        features=processed,
                        labels=targets,
                        query_ids=queries,
                        validation_mask=validation,
                        config=config.to_dict(),
                    )
                    row["group"] = group_name
                    candidate_rows.append(row)
                    neural_groups[group_name].append(row)
                    candidate_models[name] = model

    ridge_rows = [row for row in candidate_rows if row["family"] == "ridge"]
    deployable_linear_rows = [row for row in candidate_rows if row["family"] == "linear"]
    best_linear = max(deployable_linear_rows, key=_candidate_key) if deployable_linear_rows else None
    neural_group_rows = [_summarize_neural_group(name, rows) for name, rows in neural_groups.items() if rows]
    best_neural_group = max(neural_group_rows, key=_neural_group_key) if neural_group_rows else None
    representative_neural: dict[str, Any] | None = None
    if best_neural_group is not None:
        rows = neural_groups[str(best_neural_group["group"])]
        representative_neural = min(
            rows,
            key=lambda row: (
                abs(float(row["macro_qwk"]) - float(best_neural_group["mean_macro_qwk"])),
                float(row["macro_mae"]),
                int(row["seed"]),
            ),
        )
    neural_group_beats_baseline = bool(
        best_neural_group is not None
        and (best_linear is None or _neural_group_candidate_key(best_neural_group) > _candidate_key(best_linear))
    )
    representative_neural_beats_baseline = bool(
        representative_neural is not None
        and (best_linear is None or _candidate_key(representative_neural) > _candidate_key(best_linear))
    )
    # The deployment candidate must win both as a three-seed configuration and
    # as the representative checkpoint.  Otherwise a lucky seed can bypass the
    # strong-baseline fallback even when the neural family is worse on average.
    neural_beats_baseline = _stable_neural_beats_baseline(
        neural_group=best_neural_group,
        representative_neural=representative_neural,
        baseline=best_linear,
    )
    deployment_policy = (
        "neural_ablation" if bool(selection.force_neural) else str(selection.deployment_policy)
    )
    if deployment_policy == "neural_ablation":
        if representative_neural is None:
            raise RuntimeError(
                "deployment_policy='neural_ablation' requires include_neural_ablation=True"
            )
        selected = representative_neural
        selection_policy = "explicit_neural_ablation"
    elif deployment_policy == "linear_specification":
        if best_linear is None:
            raise RuntimeError(
                "linear_specification requires include_linear=True and at least one linear learning rate"
            )
        selected = best_linear
        selection_policy = "document_specified_pure_linear_5_logit_head"
    elif neural_beats_baseline:
        selected = representative_neural
        selection_policy = "stable_neural_then_logistic_fallback"
    elif best_linear is not None:
        selected = best_linear
        selection_policy = "stable_neural_then_logistic_fallback"
    elif representative_neural is not None:
        selected = representative_neural
        selection_policy = "only_neural_classifier_candidate"
    else:
        raise RuntimeError("Judge selection produced no deployable CE or logistic classifier")
    assert selected is not None
    selected_method = str(selected["preprocess_method"])
    candidate_rows_by_name = {str(row["name"]): row for row in candidate_rows}
    ensemble_candidate_names = tuple(str(row["name"]) for row in candidate_rows)
    ensemble_predictions = np.column_stack(
        [
            candidate_models[name].predict(
                processed_by_method[
                    str(candidate_rows_by_name[name]["preprocess_method"])
                ],
                queries,
            )
            for name in ensemble_candidate_names
        ]
    )
    summary = {
        "config": selection.to_dict(),
        "selection_metric": "macro_per_query_training_validation_qwk",
        "tiebreakers": ["macro_per_query_mae", "validation_log_loss"],
        "majority_baseline": majority,
        "candidate_results": candidate_rows,
        "neural_group_results": neural_group_rows,
        "best_linear": best_linear,
        "ridge_quality_baselines": ridge_rows,
        "best_neural_group": best_neural_group,
        "representative_neural": representative_neural,
        "neural_group_beats_baseline": neural_group_beats_baseline,
        "representative_neural_beats_baseline": representative_neural_beats_baseline,
        "neural_beats_baseline": neural_beats_baseline,
        "force_neural": bool(selection.force_neural),
        "deployment_policy": deployment_policy,
        "selection_policy": selection_policy,
        "selected_name": selected["name"],
        "fallback_reason": (
            "explicit_neural_ablation_despite_baseline_comparison"
            if deployment_policy == "neural_ablation" and not neural_beats_baseline
            else _fallback_reason(
                representative_neural=representative_neural,
                neural_group_beats_baseline=neural_group_beats_baseline,
                representative_neural_beats_baseline=representative_neural_beats_baseline,
            )
        ),
        "baseline_gate": "three_seed_mean_and_representative_checkpoint_must_both_beat_logistic_classifier",
        "deployment_candidate_families": ["linear"],
        "ablation_candidate_families": ["ridge", "neural_ablation"],
        "head_contract": {
            "architecture": "pure_linear_5_logit",
            "shared_across_queries": bool(
                str(selection.linear_head_sharing).lower() == "shared"
            ),
            "penultimate_equals_z_a": None,
            "penultimate": "selected_frozen_judge_representation",
            "mlp_is_ablation_only": True,
        },
        "vim_residual_compatible": True,
        "posthoc_logit_compatible": True,
        "agreement_ensemble": {
            "candidate_count": int(ensemble_predictions.shape[1]),
            "candidate_names": list(ensemble_candidate_names),
            "prediction_scope": "all_records_for_unlabeled_target_agreement",
        },
        "selected_beats_majority_qwk": float(selected["macro_qwk"]) > float(majority["macro_qwk"]),
        "preprocessing_fit_rows": {
            method: int(preprocessor.fit_rows_) for method, preprocessor in preprocessor_by_method.items()
        },
    }
    return JudgeSelectionResult(
        model=candidate_models[str(selected["name"])],
        preprocessor=preprocessor_by_method[selected_method],
        processed_features=processed_by_method[selected_method],
        selected_candidate=selected,
        summary=summary,
        ensemble_predictions=ensemble_predictions,
        ensemble_candidate_names=ensemble_candidate_names,
        representative_neural_model=(
            candidate_models[str(representative_neural["name"])]
            if representative_neural is not None
            and isinstance(candidate_models[str(representative_neural["name"])], SharedBackboneJudge)
            else None
        ),
        candidate_models=dict(candidate_models),
        preprocessors_by_method=dict(preprocessor_by_method),
    )


def _evaluate_candidate(
    *,
    name: str,
    family: str,
    preprocess_method: str,
    seed: int,
    model: SharedBackboneJudge | PerQueryLinearJudge,
    features: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    validation_mask: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    local_features = features[validation_mask]
    local_queries = query_ids[validation_mask]
    probabilities = model.predict_proba(local_features, local_queries)
    predictions = model.predict(local_features, local_queries)
    metrics = macro_query_judge_metrics(
        labels[validation_mask],
        predictions,
        local_queries,
        probabilities=probabilities,
        class_values=model.classes_,
    )
    validation_loss = _safe_log_loss(labels[validation_mask], probabilities, model.classes_)
    return {
        "name": name,
        "family": family,
        "preprocess_method": preprocess_method,
        "seed": int(seed),
        "macro_qwk": float(metrics["macro"]["qwk"]),
        "macro_mae": float(metrics["macro"]["mae"]),
        "macro_accuracy": float(metrics["macro"]["accuracy"]),
        "validation_log_loss": validation_loss,
        "metrics_by_query": metrics["by_query"],
        "config": config,
        "device": model.to_metadata().get("device"),
    }


def _summarize_neural_group(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    qwk = np.asarray([float(row["macro_qwk"]) for row in rows], dtype=float)
    mae = np.asarray([float(row["macro_mae"]) for row in rows], dtype=float)
    validation_log_loss = np.asarray(
        [float(row["validation_log_loss"]) for row in rows], dtype=float
    )
    return {
        "group": group,
        "num_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "mean_macro_qwk": float(qwk.mean()),
        "std_macro_qwk": float(qwk.std(ddof=0)),
        "min_macro_qwk": float(qwk.min()),
        "max_macro_qwk": float(qwk.max()),
        "mean_macro_mae": float(mae.mean()),
        "mean_validation_log_loss": float(validation_log_loss.mean()),
    }


def _majority_baseline(
    labels: np.ndarray,
    query_ids: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    class_values: tuple[Any, ...],
) -> dict[str, Any]:
    classes = np.asarray(class_values if class_values else np.unique(labels[train_mask]))
    queries = np.asarray(query_ids).astype(str)
    majority_by_query: dict[str, Any] = {}
    for query_id in sorted(set(queries[train_mask].tolist())):
        values, counts = np.unique(labels[train_mask & (queries == query_id)], return_counts=True)
        majority_by_query[query_id] = values[int(np.argmax(counts))]
    default = classes[int(len(classes) // 2)]
    predictions = np.asarray(
        [majority_by_query.get(query, default) for query in queries[validation_mask]],
        dtype=labels.dtype,
    )
    metrics = macro_query_judge_metrics(
        labels[validation_mask],
        predictions,
        queries[validation_mask],
        class_values=classes,
    )
    return {
        "macro_qwk": float(metrics["macro"]["qwk"]),
        "macro_mae": float(metrics["macro"]["mae"]),
        "macro_accuracy": float(metrics["macro"]["accuracy"]),
        "metrics_by_query": metrics["by_query"],
        "majority_by_query": {
            query: _json_scalar(value) for query, value in majority_by_query.items()
        },
    }


def _candidate_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["macro_qwk"]),
        -float(row["macro_mae"]),
        -float(row["validation_log_loss"]),
    )


def _neural_group_candidate_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["mean_macro_qwk"]),
        -float(row["mean_macro_mae"]),
        -float(row["mean_validation_log_loss"]),
    )


def _stable_neural_beats_baseline(
    *,
    neural_group: dict[str, Any] | None,
    representative_neural: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> bool:
    return bool(
        neural_group is not None
        and representative_neural is not None
        and (
            baseline is None
            or (
                _neural_group_candidate_key(neural_group) > _candidate_key(baseline)
                and _candidate_key(representative_neural) > _candidate_key(baseline)
            )
        )
    )


def _fallback_reason(
    *,
    representative_neural: dict[str, Any] | None,
    neural_group_beats_baseline: bool,
    representative_neural_beats_baseline: bool,
) -> str | None:
    if neural_group_beats_baseline and representative_neural_beats_baseline:
        return None
    if representative_neural is None:
        return "no_neural_judge_candidate"
    if not neural_group_beats_baseline:
        return "neural_three_seed_mean_did_not_beat_strong_baseline"
    return "representative_neural_checkpoint_did_not_beat_strong_baseline"


def _neural_group_key(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(row["mean_macro_qwk"]),
        -float(row["std_macro_qwk"]),
        -float(row["mean_macro_mae"]),
    )


def _safe_log_loss(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray | None) -> float:
    if classes is None:
        return float("inf")
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    if any(value not in class_to_index for value in labels.tolist()):
        return float("inf")
    encoded = np.asarray([class_to_index[value] for value in labels.tolist()], dtype=int)
    try:
        probability_values = np.asarray(probabilities, dtype=np.float64)
        probability_values = np.clip(probability_values, 1e-12, 1.0)
        probability_values /= probability_values.sum(axis=1, keepdims=True).clip(min=1e-12)
        return float(
            log_loss(
                encoded,
                probability_values,
                labels=np.arange(len(classes)),
            )
        )
    except ValueError:
        return float("inf")


def _as_layers(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"Expected [N,L,D] or [N,D], got {values.shape}")
    return values


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
