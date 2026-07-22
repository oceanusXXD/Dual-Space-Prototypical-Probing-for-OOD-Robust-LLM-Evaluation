from __future__ import annotations

import json
import hashlib
import math
from itertools import product
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import log_loss

from src.common.io import ensure_dir, write_json, write_jsonl
from src.llm_judge_ood.adapt.coral import CoralAligner, nearest_centroid_predict
from src.llm_judge_ood.adapt.head import HeadAdaptConfig, HeadAdapter
from src.llm_judge_ood.eval.monitoring import MonitoringBaselineConfig, evaluate_monitoring_baselines
from src.llm_judge_ood.eval.tables import build_result_tables
from src.llm_judge_ood.lifecycle.cluster import ClusterConfig
from src.llm_judge_ood.lifecycle.drift import (
    AlphaSpendingTracker,
    BehaviorMainRepresentation,
    BlockAwareC2ST,
    MMDPermutationTest,
    WindowDriftConfig,
    cluster_persistent_documents,
    derive_effective_sequential_config,
    ordered_calibration_document_indices,
    run_dual_space_drift_monitor,
    wilson_interval,
)
from src.llm_judge_ood.lifecycle.persistence import PersistenceConfig
from src.llm_judge_ood.lifecycle.probe import (
    estimate_excess_human_error_reference,
    estimate_human_ceiling,
    harmfulness_probe,
    paired_excess_human_error_probe,
)
from src.llm_judge_ood.lifecycle.sampling import stratified_random_sample
from src.llm_judge_ood.lifecycle.separability import (
    SeparabilityConfig,
    diagnose_layer_separability,
)
from src.llm_judge_ood.lifecycle.warning import BehaviorWarningCalibrator, BehaviorWarningConfig
from src.llm_judge_ood.model.judge import JudgeTrainingConfig, SharedBackboneJudge
from src.llm_judge_ood.model.baselines import PerQueryLinearJudge
from src.llm_judge_ood.model.selection import (
    JudgeSelectionConfig,
    JudgeSelectionResult,
    select_source_judge,
)
from src.llm_judge_ood.scores.selection import (
    OODSelectionConfig,
    OODSelectionResult,
    select_document_ood_detector,
)
from src.llm_judge_ood.scores.judge_selection import (
    JudgeOODSelectionConfig,
    JudgeOODSelectionResult,
    refit_selected_judge_ood_detector,
    select_judge_ood_detector,
)
from src.llm_judge_ood.shared.metrics import (
    confusion_matrix_report,
    judge_metrics,
    macro_query_judge_metrics,
    normalize_label_array,
    ood_metrics,
)
from src.llm_judge_ood.shared.feature_store import load_hidden_feature_store, record_fingerprint
from src.llm_judge_ood.shared.schema import JudgeRecord, limit_input_document_records, load_judge_records
from src.llm_judge_ood.shared.static_cache import (
    load_or_create_static_cache,
    static_cache_signature,
)
from src.llm_judge_ood.shared.whitening import LayerPreprocessor
from src.models.extract_hidden import (
    QWEN3_5_4B_HIDDEN_SIZE,
    QWEN3_5_4B_MODEL_ID,
    QWEN3_5_4B_NUM_LAYERS,
    QWEN3_5_4B_REVISION,
)


@dataclass(frozen=True)
class SampleOODConfig:
    input_paths: tuple[str, ...] = ("artifacts/splits/all_local_prepared.jsonl",)
    judge_hidden_feature_path: str | None = None
    document_hidden_feature_path: str | None = None
    backbone_model_id: str = QWEN3_5_4B_MODEL_ID
    hidden_pooling: str = "masked_mean"
    hidden_max_length: int = 2048
    output_dir: str = "artifacts/llm_judge_ood_smoke"
    ood_definition: str = "document_distribution"
    judge_feature_scope: str = "input_document"
    judge_prompt_template_version: str | None = None
    judge_prompt_template_sha256: str | None = None
    training_document_train_splits: tuple[str, ...] = ("training_train",)
    training_document_drift_reference_splits: tuple[str, ...] = ()
    require_independent_drift_reference: bool = False
    training_document_calibration_splits: tuple[str, ...] = ("training_calibration",)
    training_document_validation_splits: tuple[str, ...] = ("training_validation",)
    training_document_guard_splits: tuple[str, ...] = ("training_guard",)
    training_document_test_splits: tuple[str, ...] = ("training_test",)
    development_document_splits: tuple[str, ...] = ("development",)
    benchmark_document_splits: tuple[str, ...] = ()
    deployment_document_stream_splits: tuple[str, ...] = ("deployment_stream",)
    deployment_document_ood_evaluation_splits: tuple[str, ...] = ()
    # Deprecated compatibility alias. These rows are offline OOD evaluation
    # only; the real Probe pool is the observed persistent stream below.
    deployment_document_probe_splits: tuple[str, ...] = ()
    deployment_document_adapt_splits: tuple[str, ...] = ("deployment_adapt",)
    deployment_document_gate_splits: tuple[str, ...] = ("deployment_gate",)
    deployment_document_future_splits: tuple[str, ...] = ("deployment_future_test",)
    deployment_document_evaluation_splits: tuple[str, ...] = (
        "deployment_stream",
        "deployment_ood_evaluation",
        "deployment_adapt",
        "deployment_gate",
        "deployment_future_test",
    )
    max_input_documents: int = 1200
    static_reference_cache_dir: str | None = None
    judge: JudgeTrainingConfig = field(default_factory=lambda: JudgeTrainingConfig(epochs=12, patience=3, hidden_dim=96, output_dim=48))
    judge_selection: JudgeSelectionConfig = field(
        default_factory=lambda: JudgeSelectionConfig(
            preprocess_methods=("pca_whiten",),
            neural_losses=("ce",),
            neural_seeds=(42,),
            ridge_alphas=(10.0,),
            linear_learning_rates=(1e-3,),
        )
    )
    ood_selection: OODSelectionConfig = field(
        default_factory=lambda: OODSelectionConfig(
            preprocess_methods=("pca_whiten",),
            representations=("last_layer",),
            metrics=("cosine",),
            k_values=(10,),
        )
    )
    judge_ood_selection: JudgeOODSelectionConfig = field(default_factory=JudgeOODSelectionConfig)
    window_drift: WindowDriftConfig = field(default_factory=WindowDriftConfig)
    behavior_warning: BehaviorWarningConfig = field(default_factory=BehaviorWarningConfig)
    separability: SeparabilityConfig = field(default_factory=SeparabilityConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)
    tune_lifecycle_on_development: bool = False
    lifecycle_window_sizes: tuple[int, ...] = ()
    lifecycle_minimum_consecutive_windows: tuple[int, ...] = ()
    lifecycle_alpha_fwers: tuple[float, ...] = ()
    lifecycle_alpha_spendings: tuple[str, ...] = ()
    probe_budget: int = 20
    probe_requires_behavior_warning: bool = False
    probe_min_documents: int = 4
    probe_min_documents_per_query: int = 2
    safety_net_period_documents: int = 1000
    safety_net_labels_per_period: int = 10
    safety_net_initial_documents: int = 0
    safety_net_initial_labels: int = 0
    gate_budget: int = 20
    gate_min_documents: int = 3
    training_replay_budget: int = 20
    training_drop_tolerance: float = 0.02
    harm_tolerance: float = 0.15
    harm_fdr_alpha: float = 0.10
    human_ceiling_metadata_key: str = "rater_scores"
    require_human_ceiling: bool = False
    permutation_block_metadata_key: str = "arrival_batch_id"
    bootstrap_samples: int = 1000
    probe_metric: str = "qwk"
    paired_harmfulness_mode: str = "auto"
    gate_min_excess_error_improvement: float = 0.10
    gate_max_negative_flip_rate: float = 0.05
    gate_bootstrap_samples: int = 500
    wide_harmful_document_share: float = 0.5
    reference_full_recalibration_update_interval: int = 5
    reference_full_recalibration_shift_sigma: float = 1.0
    document_cluster_routing_requires_ood: bool = True
    document_cluster_routing_distance_quantile: float = 0.95
    monitoring_include_training_test_preamble: bool = False
    monitoring_window_ood_rate_threshold: float | None = None
    head_adapt: HeadAdaptConfig = field(default_factory=HeadAdaptConfig)
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        payload = _json_ready(asdict(self))
        payload["judge"] = self.judge.to_dict()
        payload["judge_selection"] = self.judge_selection.to_dict()
        payload["ood_selection"] = self.ood_selection.to_dict()
        payload["judge_ood_selection"] = self.judge_ood_selection.to_dict()
        payload["window_drift"] = self.window_drift.to_dict()
        payload["behavior_warning"] = self.behavior_warning.to_dict()
        payload["separability"] = self.separability.to_dict()
        payload["cluster"] = self.cluster.to_dict()
        payload["persistence"] = self.persistence.to_dict()
        payload["head_adapt"] = self.head_adapt.to_dict()
        return payload


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def run_sample_ood_pipeline(config: SampleOODConfig) -> dict[str, Any]:
    if config.ood_definition != "document_distribution":
        raise ValueError("Only ood_definition='document_distribution' is supported by the production pipeline")
    if str(config.judge_feature_scope) not in {"input_document", "judge_input"}:
        raise ValueError("judge_feature_scope must be 'input_document' or 'judge_input'")
    if str(config.backbone_model_id) != QWEN3_5_4B_MODEL_ID:
        raise ValueError("The final protocol requires backbone_model_id='Qwen/Qwen3.5-4B'")
    if str(config.hidden_pooling) != "masked_mean":
        raise ValueError("The final protocol requires hidden_pooling='masked_mean'")
    if int(config.hidden_max_length) != 2048:
        raise ValueError("The final protocol requires hidden_max_length=2048")
    if config.monitoring_include_training_test_preamble:
        raise ValueError(
            "Final-design window monitoring accepts deployment documents only; "
            "training-test preambles are evaluation baselines, not production window input"
        )
    if int(config.safety_net_period_documents) < 1:
        raise ValueError("safety_net_period_documents must be positive")
    if not 0 <= int(config.safety_net_labels_per_period) <= int(config.safety_net_period_documents):
        raise ValueError("safety_net_labels_per_period must be between zero and the period size")
    if not 0 <= int(config.safety_net_initial_labels) <= int(config.safety_net_initial_documents):
        raise ValueError("safety_net_initial_labels must be between zero and initial_documents")
    if int(config.safety_net_initial_labels) > int(config.safety_net_labels_per_period):
        raise ValueError("front-loaded safety-net labels cannot exceed the period budget")
    if not 0 <= int(config.probe_budget) <= 20:
        raise ValueError("probe_budget must be in [0, 20] under the small-sample harmfulness design")
    if not 0 <= int(config.gate_budget) <= 20:
        raise ValueError("gate_budget must be in [0, 20] under the independent per-cluster gate budget")
    if not 0 <= int(config.training_replay_budget) <= 20:
        raise ValueError("training_replay_budget must be in [0, 20] under the 1:1 source replay budget")
    if int(config.probe_min_documents) < 2 or int(config.probe_min_documents_per_query) < 2:
        raise ValueError("Probe document minimums must both be at least two")
    if int(config.gate_min_documents) < 2:
        raise ValueError("gate_min_documents must be at least two")
    if not 0.0 <= float(config.wide_harmful_document_share) <= 1.0:
        raise ValueError("wide_harmful_document_share must be in [0, 1]")
    if not 0.0 < float(config.document_cluster_routing_distance_quantile) <= 1.0:
        raise ValueError("document_cluster_routing_distance_quantile must be in (0, 1]")
    if not bool(config.document_cluster_routing_requires_ood):
        raise ValueError("document_cluster_routing_requires_ood must remain true in the final protocol")
    if bool(config.head_adapt.holdout_deployment_for_early_stopping):
        raise ValueError(
            "The final small-sample protocol reuses every confirmed harmful Probe label "
            "for Adapt; choose fixed development-selected epochs instead of a target holdout"
        )
    if int(config.reference_full_recalibration_update_interval) < 1:
        raise ValueError("reference_full_recalibration_update_interval must be positive")
    if float(config.reference_full_recalibration_shift_sigma) <= 0.0:
        raise ValueError("reference_full_recalibration_shift_sigma must be positive")
    if not 0.0 < float(config.harm_fdr_alpha) <= 1.0:
        raise ValueError("harm_fdr_alpha must be in (0, 1]")
    if str(config.paired_harmfulness_mode) not in {"auto", "required", "disabled"}:
        raise ValueError("paired_harmfulness_mode must be 'auto', 'required', or 'disabled'")
    output_dir = ensure_dir(config.output_dir)
    records = load_judge_records(config.input_paths)
    records = limit_input_document_records(records, config.max_input_documents, seed=config.seed)
    audit_document_groups = np.asarray([record.audit_document_group_id for record in records]).astype(str)
    document_roles = np.asarray([record.document_distribution_role for record in records]).astype(str)
    labels = normalize_label_array([record.label for record in records])
    splits = np.asarray([record.split for record in records]).astype(str)
    query_ids = np.asarray([record.query_id for record in records]).astype(str)
    document_ids = np.asarray([record.input_document_id for record in records]).astype(str)
    stream_orders = np.asarray([record.stream_order for record in records], dtype=object)
    permutation_block_ids = _permutation_block_ids(records, key=config.permutation_block_metadata_key)
    rater_scores = _rater_scores(records, key=config.human_ceiling_metadata_key)
    document_ood_truth, document_shift_types = _document_ood_ground_truth(records)
    input_document_ids = document_ids
    _validate_document_distribution_contract(
        records,
        document_ids=document_ids,
        document_roles=document_roles,
        stream_orders=stream_orders,
    )
    _validate_judge_input_template_contract(records=records, config=config, defer_records=False)
    training_role_mask = document_roles == "training"
    development_role_mask = document_roles == "development"
    benchmark_role_mask = document_roles == "benchmark"
    deployment_role_mask = document_roles == "deployment"
    training_train_mask = training_role_mask & np.isin(
        splits, config.training_document_train_splits
    )
    training_drift_reference_mask = training_role_mask & np.isin(
        splits, config.training_document_drift_reference_splits
    )
    training_calibration_mask = training_role_mask & np.isin(
        splits, config.training_document_calibration_splits
    )
    training_validation_mask = training_role_mask & np.isin(
        splits, config.training_document_validation_splits
    )
    training_guard_mask = training_role_mask & np.isin(
        splits, config.training_document_guard_splits
    )
    training_test_mask = training_role_mask & np.isin(
        splits, config.training_document_test_splits
    )
    development_document_mask = development_role_mask & np.isin(
        splits, config.development_document_splits
    )
    benchmark_document_mask = benchmark_role_mask & np.isin(
        splits, config.benchmark_document_splits
    )
    deployment_stream_mask = deployment_role_mask & np.isin(
        splits, config.deployment_document_stream_splits
    )
    offline_ood_splits = (
        config.deployment_document_ood_evaluation_splits
        if config.deployment_document_ood_evaluation_splits
        else config.deployment_document_probe_splits
    )
    if config.deployment_document_probe_splits and config.deployment_document_ood_evaluation_splits:
        if tuple(config.deployment_document_probe_splits) != tuple(config.deployment_document_ood_evaluation_splits):
            raise ValueError(
                "deployment_document_probe_splits is deprecated and must match "
                "deployment_document_ood_evaluation_splits when both are supplied"
            )
    deployment_offline_ood_mask = deployment_role_mask & np.isin(
        splits, offline_ood_splits
    )
    deployment_adapt_mask = deployment_role_mask & np.isin(
        splits, config.deployment_document_adapt_splits
    )
    deployment_gate_mask = deployment_role_mask & np.isin(
        splits, config.deployment_document_gate_splits
    )
    deployment_future_mask = deployment_role_mask & np.isin(
        splits, config.deployment_document_future_splits
    )
    deployment_evaluation_mask = deployment_role_mask & np.isin(
        splits, config.deployment_document_evaluation_splits
    )
    if not training_train_mask.any():
        raise ValueError("No training document train records selected")
    if bool(config.require_independent_drift_reference) and not config.training_document_drift_reference_splits:
        raise ValueError(
            "Formal monitoring requires non-empty training_document_drift_reference_splits"
        )
    if config.training_document_drift_reference_splits:
        _require_selected(
            training_drift_reference_mask,
            "training-document drift reference",
            config.training_document_drift_reference_splits,
        )
    _require_selected(training_calibration_mask, "training-document calibration", config.training_document_calibration_splits)
    _require_selected(training_validation_mask, "training-document validation", config.training_document_validation_splits)
    _require_selected(training_guard_mask, "training-document guard", config.training_document_guard_splits)
    _require_selected(training_test_mask, "training-document test", config.training_document_test_splits)
    _require_selected(
        development_document_mask,
        "development documents",
        config.development_document_splits,
    )
    if benchmark_role_mask.any() and not config.benchmark_document_splits:
        raise ValueError(
            "Benchmark-role documents are present but benchmark_document_splits is empty; "
            "declare the independent confirmation split instead of silently ignoring it"
        )
    if config.benchmark_document_splits:
        _require_selected(
            benchmark_document_mask,
            "benchmark-test documents",
            config.benchmark_document_splits,
        )
    _require_selected(deployment_stream_mask, "deployment document stream", config.deployment_document_stream_splits)
    if offline_ood_splits:
        _require_selected(
            deployment_offline_ood_mask,
            "deployment offline OOD evaluation",
            offline_ood_splits,
        )
    if config.deployment_document_adapt_splits:
        _require_selected(deployment_adapt_mask, "deployment document adapt", config.deployment_document_adapt_splits)
    _require_selected(deployment_gate_mask, "deployment document gate", config.deployment_document_gate_splits)
    _require_selected(deployment_future_mask, "deployment document future", config.deployment_document_future_splits)
    _require_selected(
        deployment_evaluation_mask,
        "deployment document OOD evaluation",
        config.deployment_document_evaluation_splits,
    )
    _validate_disjoint_masks(
        (
            training_train_mask,
            training_drift_reference_mask,
            training_calibration_mask,
            training_validation_mask,
            training_guard_mask,
            training_test_mask,
        ),
        ("train", "drift_reference", "calibration", "validation", "guard", "test"),
    )
    _validate_disjoint_document_masks(
        document_ids,
        (
            training_train_mask,
            training_drift_reference_mask,
            training_calibration_mask,
            training_validation_mask,
            training_guard_mask,
            training_test_mask,
        ),
        ("train", "drift_reference", "calibration", "validation", "guard", "test"),
    )
    _validate_disjoint_masks(
        (
            deployment_stream_mask,
            deployment_offline_ood_mask,
            deployment_adapt_mask,
            deployment_gate_mask,
            deployment_future_mask,
        ),
        ("stream", "offline_ood_evaluation", "adapt", "gate", "future_test"),
    )
    _validate_disjoint_document_masks(
        document_ids,
        (
            deployment_stream_mask,
            deployment_offline_ood_mask,
            deployment_adapt_mask,
            deployment_gate_mask,
            deployment_future_mask,
        ),
        ("stream", "offline_ood_evaluation", "adapt", "gate", "future_test"),
    )
    _validate_document_isolation(
        document_ids=document_ids,
        source_mask=training_train_mask | training_drift_reference_mask | training_calibration_mask | training_validation_mask | training_guard_mask | training_test_mask,
        development_mask=development_document_mask,
        benchmark_mask=benchmark_document_mask,
        final_mask=deployment_evaluation_mask,
    )

    document_raw_features, document_feature_metadata = _load_or_extract_features(
        config,
        records,
        training_train_mask,
        feature_scope="input_document",
    )
    static_cache_audit: dict[str, dict[str, Any]] = {}
    effective_separability_config = replace(config.separability, seed=int(config.seed))
    separability_rows = training_validation_mask | development_document_mask
    separability_signature = static_cache_signature(
        "separability_v1",
        config=effective_separability_config.to_dict(),
        arrays=(("features", document_raw_features[separability_rows]),),
        string_arrays=(
            ("document_ids", document_ids[separability_rows]),
            ("roles", np.where(training_validation_mask[separability_rows], "validation", "development")),
            ("shift_types", document_shift_types[separability_rows]),
        ),
    )
    separability_cache = load_or_create_static_cache(
        cache_dir=config.static_reference_cache_dir,
        namespace="separability_v1",
        signature=separability_signature,
        create=lambda: diagnose_layer_separability(
            raw_document_features=document_raw_features,
            input_document_ids=document_ids,
            source_validation_mask=training_validation_mask,
            development_mask=development_document_mask,
            document_shift_types=document_shift_types,
            config=effective_separability_config,
        ),
        validate=lambda value: isinstance(value, dict),
    )
    separability = dict(separability_cache.value)
    static_cache_audit["separability"] = separability_cache.to_metadata()
    separability_artifact = output_dir / "representation_separability.json"
    write_json(separability_artifact, separability)
    selected_monitoring_layer = separability.get("selected_layer_index")
    if (
        bool(config.separability.apply_selected_layer)
        and separability.get("available")
        and selected_monitoring_layer is not None
    ):
        layer_index = int(selected_monitoring_layer)
        document_raw_features = document_raw_features[:, layer_index : layer_index + 1, :]
        document_feature_metadata = {
            **document_feature_metadata,
            "separability_selected_layer_index": layer_index,
            "monitoring_layer_application": "input_document_A_space",
        }
    if str(config.judge_feature_scope) == "input_document":
        judge_raw_features = document_raw_features.copy()
        judge_feature_metadata = {
            **document_feature_metadata,
            "feature_scope": "input_document",
            "judge_A_space_contract": "same_frozen_input_document_representation",
            "shared_with_document_monitoring": True,
        }
    else:
        judge_raw_features, judge_feature_metadata = _load_or_extract_features(
            config,
            records,
            training_train_mask,
            feature_scope="judge_input",
        )
        judge_feature_metadata = {
            **judge_feature_metadata,
            "judge_A_space_contract": "separate_judge_input_representation_A_space_remains_input_document_only",
            "shared_with_document_monitoring": False,
        }
    judge_selection_rows = training_train_mask | training_validation_mask
    judge_selection_signature = static_cache_signature(
        "judge_selection_v1",
        config={
            "judge": config.judge.to_dict(),
            "selection": config.judge_selection.to_dict(),
        },
        arrays=(
            ("features", judge_raw_features[judge_selection_rows]),
            ("labels", labels[judge_selection_rows]),
        ),
        string_arrays=(
            ("query_ids", query_ids[judge_selection_rows]),
            (
                "roles",
                np.where(training_train_mask[judge_selection_rows], "train", "validation"),
            ),
        ),
    )
    judge_selection_cache = load_or_create_static_cache(
        cache_dir=config.static_reference_cache_dir,
        namespace="judge_selection_v1",
        signature=judge_selection_signature,
        create=lambda: _compact_judge_selection(
            select_source_judge(
                raw_features=judge_raw_features,
                labels=labels,
                query_ids=query_ids,
                train_mask=training_train_mask,
                validation_mask=training_validation_mask,
                base_config=config.judge,
                selection_config=config.judge_selection,
            )
        ),
        validate=lambda value: isinstance(value, JudgeSelectionResult)
        and bool(value.candidate_models)
        and bool(value.preprocessors_by_method),
    )
    judge_selection = judge_selection_cache.value.refreshed(judge_raw_features, query_ids)
    static_cache_audit["judge_selection"] = judge_selection_cache.to_metadata()
    judge = judge_selection.model
    if not isinstance(judge, PerQueryLinearJudge):
        raise RuntimeError(
            "The final protocol requires a deployed pure-linear Judge head"
        )
    judge_features = judge_selection.processed_features
    judge_output = judge.predict_output(judge_features, query_ids)
    predictions = judge_output.classes[np.argmax(judge_output.probabilities, axis=1)]
    probabilities = judge_output.probabilities
    u_space = judge_output.penultimate
    head_weight, head_bias, head_query_ids = judge.affine_head_parameters()

    reference_metrics = judge_metrics(
        labels[training_validation_mask],
        predictions[training_validation_mask],
        probabilities=probabilities[training_validation_mask],
        class_values=judge.classes_,
    )
    training_reference_macro = macro_query_judge_metrics(
        labels[training_validation_mask],
        predictions[training_validation_mask],
        query_ids[training_validation_mask],
        probabilities=probabilities[training_validation_mask],
        class_values=judge.classes_,
    )
    if config.probe_metric not in training_reference_macro["macro"]:
        raise ValueError(f"Unsupported probe metric: {config.probe_metric!r}")
    reference_human_ceiling = estimate_human_ceiling(
        rater_scores[training_validation_mask],
        metric_name=config.probe_metric,
        query_ids=query_ids[training_validation_mask],
        class_values=judge.classes_,
    )
    reference_excess_human_error = estimate_excess_human_error_reference(
        y_true=labels[training_guard_mask],
        y_pred=predictions[training_guard_mask],
        rater_scores=rater_scores[training_guard_mask],
        groups=input_document_ids[training_guard_mask],
    )
    effective_ood_config = replace(config.ood_selection, seed=int(config.seed))
    ood_selection_rows = (
        training_train_mask | training_calibration_mask | development_document_mask
    )
    ood_selection_signature = static_cache_signature(
        "document_ood_selection_v1",
        config=effective_ood_config.to_dict(),
        arrays=(("features", document_raw_features[ood_selection_rows]),),
        string_arrays=(
            ("document_ids", document_ids[ood_selection_rows]),
            (
                "roles",
                np.select(
                    [
                        training_train_mask[ood_selection_rows],
                        training_calibration_mask[ood_selection_rows],
                    ],
                    ["train", "calibration"],
                    default="development",
                ),
            ),
        ),
    )
    reusable_preprocessors = (
        judge_selection.preprocessors_by_method
        if str(config.judge_feature_scope) == "input_document"
        else None
    )
    ood_selection_cache = load_or_create_static_cache(
        cache_dir=config.static_reference_cache_dir,
        namespace="document_ood_selection_v1",
        signature=ood_selection_signature,
        create=lambda: _compact_ood_selection(
            select_document_ood_detector(
                raw_features=document_raw_features,
                input_document_ids=document_ids,
                training_document_mask=training_train_mask,
                calibration_document_mask=training_calibration_mask,
                development_document_mask=development_document_mask,
                config=effective_ood_config,
                prefitted_preprocessors=reusable_preprocessors,
            )
        ),
        validate=lambda value: isinstance(value, OODSelectionResult),
    )
    ood_selection = ood_selection_cache.value.refreshed(
        document_raw_features,
        document_ids,
    )
    static_cache_audit["document_ood_selection"] = ood_selection_cache.to_metadata()
    knn_scores = ood_selection.scores
    score_labels = ood_selection.score_labels
    thresholds = ood_selection.thresholds
    judge_ood_rows = (
        training_train_mask
        | training_calibration_mask
        | development_document_mask
        | benchmark_document_mask
    )
    judge_ood_signature = static_cache_signature(
        "judge_ood_selection_v4_classifier_origin_full_vim",
        config=config.judge_ood_selection.to_dict(),
        arrays=(
            ("penultimate", judge_output.penultimate[judge_ood_rows]),
            ("logits", judge_output.logits[judge_ood_rows]),
            ("labels", labels[judge_ood_rows]),
            ("document_ood_truth", document_ood_truth[judge_ood_rows].astype(np.int8)),
            ("head_weight", head_weight),
            ("head_bias", head_bias),
        ),
        string_arrays=(
            ("document_ids", document_ids[judge_ood_rows]),
            ("query_ids", query_ids[judge_ood_rows]),
            ("shift_types", document_shift_types[judge_ood_rows]),
            ("head_query_ids", head_query_ids),
            (
                "roles",
                np.select(
                    [
                        training_train_mask[judge_ood_rows],
                        training_calibration_mask[judge_ood_rows],
                        development_document_mask[judge_ood_rows],
                    ],
                    ["train", "calibration", "development"],
                    default="benchmark",
                ),
            ),
        ),
    )
    judge_ood_cache = load_or_create_static_cache(
        cache_dir=config.static_reference_cache_dir,
        namespace="judge_ood_selection_v4_classifier_origin_full_vim",
        signature=judge_ood_signature,
        create=lambda: _compact_judge_ood_selection(
            select_judge_ood_detector(
                penultimate=judge_output.penultimate,
                logits=judge_output.logits,
                labels=labels,
                class_values=judge_output.classes,
                training_mask=training_train_mask,
                calibration_mask=training_calibration_mask,
                development_mask=development_document_mask,
                benchmark_mask=benchmark_document_mask,
                document_ood_labels=document_ood_truth,
                document_ids=document_ids,
                shift_types=document_shift_types,
                query_ids=query_ids,
                head_weight=head_weight,
                head_bias=head_bias,
                head_query_ids=head_query_ids,
                config=config.judge_ood_selection,
            )
        ),
        validate=lambda value: isinstance(value, JudgeOODSelectionResult),
    )
    judge_behavior_ood = judge_ood_cache.value.refreshed(
        judge_output.penultimate,
        judge_output.logits,
        query_ids,
    )
    static_cache_audit["judge_ood_selection"] = judge_ood_cache.to_metadata()
    judge_behavior_scores = judge_behavior_ood.scores
    judge_behavior_labels = judge_behavior_ood.score_labels
    judge_behavior_document_labels = _broadcast_document_ood_labels(
        judge_behavior_labels,
        document_ids,
    )
    deployment_ood_evaluation_mask = deployment_stream_mask | deployment_offline_ood_mask | deployment_adapt_mask | deployment_gate_mask | deployment_future_mask
    deployment_ood_target_mask = deployment_ood_evaluation_mask & document_ood_truth
    deployment_id_control_mask = deployment_ood_evaluation_mask & ~document_ood_truth
    final_ood_report = _document_ood_metrics(
        document_ids=document_ids,
        id_mask=training_calibration_mask | deployment_id_control_mask,
        target_mask=deployment_ood_target_mask,
        scores=knn_scores,
    )
    development_ood_by_shift = {
        shift_type: _document_ood_metrics(
            document_ids=document_ids,
            id_mask=training_calibration_mask,
            target_mask=development_document_mask & (document_shift_types == shift_type),
            scores=knn_scores,
        )
        for shift_type in ("near", "far")
    }
    benchmark_ood_by_shift = {
        shift_type: _document_ood_metrics(
            document_ids=document_ids,
            id_mask=benchmark_document_mask & ~document_ood_truth,
            target_mask=benchmark_document_mask
            & document_ood_truth
            & (document_shift_types == shift_type),
            scores=knn_scores,
        )
        for shift_type in ("near", "far")
    } if benchmark_document_mask.any() else {}
    benchmark_ood_report = (
        _document_ood_metrics(
            document_ids=document_ids,
            id_mask=benchmark_document_mask & ~document_ood_truth,
            target_mask=benchmark_document_mask & document_ood_truth,
            scores=knn_scores,
        )
        if benchmark_document_mask.any()
        else {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    )
    static_ood = {
        "selected_development": dict(ood_selection.development_metrics),
        "selected_development_by_shift": development_ood_by_shift,
        "independent_benchmark_test": benchmark_ood_report,
        "independent_benchmark_test_by_shift": benchmark_ood_by_shift,
        "selected_deployment": dict(final_ood_report),
    }
    y_ood = document_ood_truth.astype(int)
    judge_artifacts = judge_selection.save_model_artifacts(output_dir / "judge_checkpoints")
    judge_fingerprint = _judge_fingerprint(judge_selection, artifact_paths=judge_artifacts)
    judge_behavior_ood_artifact = judge_behavior_ood.save_artifact(
        output_dir,
        judge_fingerprint=judge_fingerprint,
    )
    judge_preprocessor_artifact = output_dir / "judge_preprocessor.npz"
    _save_layer_preprocessor(judge_selection.preprocessor, judge_preprocessor_artifact)
    ood_artifacts = ood_selection.save_artifacts(output_dir)
    judge_diagnostics = _judge_diagnostics(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        query_ids=query_ids,
        train_mask=training_train_mask,
        validation_mask=training_validation_mask,
        class_values=judge.classes_,
        majority_by_query=judge_selection.summary["majority_baseline"]["majority_by_query"],
    )
    write_json(output_dir / "judge_selection.json", judge_selection.summary)
    write_json(output_dir / "judge_diagnostics.json", judge_diagnostics)
    confusion_artifacts = _write_confusion_artifacts(judge_diagnostics, output_dir)
    if str(judge_behavior_ood.selected_candidate.get("detector", "")).lower() != "vim":
        raise ValueError("Formal B-MMD requires the selected residual-only ViM detector")
    behavior_main_signature = static_cache_signature(
        "behavior_main_representation_v2",
        config={
            "representation": "vim_source_subspace_residual_vector",
            "vim_rank": int(judge_behavior_ood.selected_candidate["rank"]),
        },
        arrays=(
            ("penultimate", judge_output.penultimate[training_train_mask]),
            *tuple(judge_behavior_ood.scorer.artifact_arrays().items()),
        ),
    )
    behavior_main_cache = load_or_create_static_cache(
        cache_dir=config.static_reference_cache_dir,
        namespace="behavior_main_representation_v2",
        signature=behavior_main_signature,
        create=lambda: BehaviorMainRepresentation(
            rank=int(judge_behavior_ood.selected_candidate["rank"]),
            random_state=int(config.seed),
        ).fit(
            judge_output.penultimate[training_train_mask],
            scorer=judge_behavior_ood.scorer,
        ),
        validate=lambda value: (
            isinstance(value, BehaviorMainRepresentation)
            and value.to_metadata().get("representation")
            == "vim_source_subspace_residual_vector"
        ),
    )
    behavior_main_representation = behavior_main_cache.value
    static_cache_audit["behavior_main_representation"] = behavior_main_cache.to_metadata()
    behavior_drift_embeddings = behavior_main_representation.transform(
        judge_output.penultimate
    )
    document_drift_embeddings = np.asarray(ood_selection.embeddings, dtype=np.float64)
    cluster_routing_embeddings, cluster_space_metadata = _cluster_routing_embeddings(
        config.cluster,
        document_embeddings=document_drift_embeddings,
        behavior_embeddings=behavior_drift_embeddings,
        document_ids=document_ids,
    )
    behavior_main_artifact = output_dir / "behavior_main_representation.npz"
    np.savez(
        behavior_main_artifact,
        **behavior_main_representation.artifact_arrays(),
        metadata_json=np.asarray(
            json.dumps(behavior_main_representation.to_metadata(), ensure_ascii=False)
        ),
    )
    if bool(config.require_independent_drift_reference):
        drift_reference_mask = training_drift_reference_mask
    else:
        drift_reference_mask = (
            training_drift_reference_mask
            if training_drift_reference_mask.any()
            else training_train_mask
        )
    source_document_indices = _first_indices_by_document(
        np.where(drift_reference_mask)[0], document_ids
    )
    calibration_document_indices = _first_indices_by_document(np.where(training_calibration_mask)[0], document_ids)
    development_document_indices = _first_indices_by_document(np.where(development_document_mask)[0], document_ids)
    source_behavior_indices = np.where(drift_reference_mask)[0]
    calibration_behavior_indices = np.where(training_calibration_mask)[0]
    development_behavior_indices = np.where(development_document_mask)[0]
    lifecycle_document_indices = np.concatenate(
        [
            source_document_indices,
            calibration_document_indices,
            development_document_indices,
        ]
    )
    lifecycle_behavior_indices = np.concatenate(
        [
            source_behavior_indices,
            calibration_behavior_indices,
            development_behavior_indices,
        ]
    )
    lifecycle_selection_arrays: tuple[tuple[str, np.ndarray], ...] = (
        ("document_embeddings", document_drift_embeddings[lifecycle_document_indices]),
        ("behavior_embeddings", behavior_drift_embeddings[lifecycle_behavior_indices]),
        ("behavior_ood_scores", judge_behavior_scores[lifecycle_behavior_indices]),
    )
    lifecycle_selection_strings: tuple[tuple[str, np.ndarray], ...] = (
        ("document_ids", document_ids[lifecycle_document_indices]),
        ("behavior_document_ids", document_ids[lifecycle_behavior_indices]),
        (
            "document_roles",
            np.concatenate(
                [
                    np.full(source_document_indices.size, "source"),
                    np.full(calibration_document_indices.size, "calibration"),
                    np.full(development_document_indices.size, "development"),
                ]
            ),
        ),
        (
            "behavior_roles",
            np.concatenate(
                [
                    np.full(source_behavior_indices.size, "source"),
                    np.full(calibration_behavior_indices.size, "calibration"),
                    np.full(development_behavior_indices.size, "development"),
                ]
            ),
        ),
        (
            "permutation_block_ids",
            (
                np.asarray(permutation_block_ids).astype(str)[lifecycle_behavior_indices]
                if permutation_block_ids is not None
                else document_ids[lifecycle_behavior_indices]
            ),
        ),
    )
    lifecycle_selection_signature = static_cache_signature(
        "lifecycle_selection_v1",
        config={
            "window_drift": replace(config.window_drift, seed=int(config.seed)).to_dict(),
            "tune_lifecycle_on_development": bool(config.tune_lifecycle_on_development),
            "lifecycle_window_sizes": list(config.lifecycle_window_sizes),
            "lifecycle_minimum_consecutive_windows": list(
                config.lifecycle_minimum_consecutive_windows
            ),
            "lifecycle_alpha_fwers": list(config.lifecycle_alpha_fwers),
            "lifecycle_alpha_spendings": list(config.lifecycle_alpha_spendings),
            "seed": int(config.seed),
        },
        arrays=lifecycle_selection_arrays,
        string_arrays=lifecycle_selection_strings,
    )
    lifecycle_selection_cache = load_or_create_static_cache(
        cache_dir=config.static_reference_cache_dir,
        namespace="lifecycle_selection_v1",
        signature=lifecycle_selection_signature,
        create=lambda: _select_window_drift_configuration(
            config=config,
            document_embeddings=document_drift_embeddings,
            behavior_embeddings=behavior_drift_embeddings,
            document_ids=document_ids,
            source_document_indices=source_document_indices,
            calibration_document_indices=calibration_document_indices,
            development_document_indices=development_document_indices,
            source_behavior_indices=source_behavior_indices,
            calibration_behavior_indices=calibration_behavior_indices,
            permutation_block_ids=permutation_block_ids,
            behavior_ood_scores=judge_behavior_scores,
        ),
        validate=_valid_lifecycle_selection_cache,
    )
    (
        effective_window_drift_config,
        window_drift_selection,
        selected_drift_reference,
    ) = lifecycle_selection_cache.value
    static_cache_audit["lifecycle_selection"] = lifecycle_selection_cache.to_metadata()
    monitoring_stream_indices, monitoring_stream_metadata = _ordered_monitoring_document_indices(
        training_indices=np.zeros(0, dtype=int),
        deployment_indices=np.where(deployment_stream_mask)[0],
        input_document_ids=document_ids,
        window_size=int(effective_window_drift_config.window_size),
        stream_orders=stream_orders,
    )
    dual_space_drift = run_dual_space_drift_monitor(
        document_embeddings=document_drift_embeddings,
        behavior_embeddings=behavior_drift_embeddings,
        document_ids=document_ids,
        source_document_indices=source_document_indices,
        calibration_document_indices=calibration_document_indices,
        stream_document_indices=monitoring_stream_indices,
        source_behavior_indices=source_behavior_indices,
        calibration_behavior_indices=calibration_behavior_indices,
        config=effective_window_drift_config,
        permutation_block_ids=permutation_block_ids,
        behavior_ood_scores=judge_behavior_scores,
        reference=selected_drift_reference,
    )
    write_jsonl(output_dir / "window_drift.jsonl", dual_space_drift.window_rows)
    episode = dual_space_drift.first_persistent_episode
    episode_segment_indices = np.asarray(
        episode.get("active_rejection_segment_document_indices", [])
        if episode is not None
        else [],
        dtype=int,
    )
    episode_visible_indices = np.asarray(
        episode.get("visible_document_indices", [])
        if episode is not None
        else monitoring_stream_indices,
        dtype=int,
    )
    localized_contributor_mask = np.zeros(len(document_ids), dtype=bool)
    if episode_segment_indices.size:
        localized_contributor_mask[episode_segment_indices] = np.isin(
            judge_behavior_document_labels[episode_segment_indices],
            ("soft_ood", "hard_ood"),
        )
    lifecycle_rows, candidate_document_indices, cluster_labels = cluster_persistent_documents(
        document_embeddings=cluster_routing_embeddings,
        persistent_document_indices=episode_segment_indices,
        window_rows=dual_space_drift.window_rows,
        config=config.cluster,
        localization_mask=localized_contributor_mask,
        cluster_space=str(cluster_space_metadata["name"]),
    )
    lifecycle_selection = {
        **window_drift_selection,
        "selection_used_deployment_documents": False,
        "window_drift": effective_window_drift_config.to_dict(),
        "first_persistent_episode": {
            "confirmed": episode is not None,
            "confirmation_window": (
                int(episode["confirmation_window"]) if episode is not None else None
            ),
            "visible_document_count": int(episode_visible_indices.size),
            "rejection_segment_document_count": int(episode_segment_indices.size),
            "localized_contributor_document_count": int(
                localized_contributor_mask.sum()
            ),
            "localization_rule": (
                "first_persistent_rejection_segment_and_vim_residual_in_{soft_ood,hard_ood}"
            ),
        },
        "cluster": config.cluster.to_dict(),
        "persistence": {
            "role": "legacy_cluster_metadata_only_routing_uses_frozen_distance_quantile",
            "config": config.persistence.to_dict(),
        },
    }
    candidate_document_indices = candidate_document_indices.astype(int)

    persistent_prototypes = _build_persistent_document_cluster_prototypes(
        lifecycle_rows,
        cluster_routing_embeddings,
        routing_distance_quantile=float(config.document_cluster_routing_distance_quantile),
    )
    persistent_indices = _persistent_document_indices(lifecycle_rows, candidate_document_indices, cluster_labels)
    has_persistent_candidate = bool(persistent_prototypes)
    persistent_predicted_document_cluster_ids = tuple(sorted(persistent_prototypes))
    audit_persistent_document_group_ids = (
        tuple(sorted(set(audit_document_groups[persistent_indices].astype(str).tolist()))) if persistent_indices.size else ()
    )
    contributor_cluster_mask = np.isin(
        cluster_labels.astype(str),
        np.asarray(persistent_predicted_document_cluster_ids, dtype=str),
    )
    probe_allowed = candidate_document_indices[contributor_cluster_mask]
    probe_predicted_document_clusters = cluster_labels[contributor_cluster_mask].astype(
        str
    )
    probe_routing = {
        "source": "observed_persistent_B_contributor_documents",
        "pool_rows": int(candidate_document_indices.size),
        "ood_document_rows": int(candidate_document_indices.size),
        "assigned_rows": int(probe_allowed.size),
        "rejected_non_ood_rows": 0,
        "rejected_outside_radius_rows": 0,
        "assigned_by_predicted_document_cluster": {
            cluster_id: int(np.sum(probe_predicted_document_clusters == cluster_id))
            for cluster_id in sorted(set(probe_predicted_document_clusters.tolist()))
        },
        "accepted_distance_mean": 0.0 if probe_allowed.size else None,
        "accepted_distance_max": 0.0 if probe_allowed.size else None,
        "uses_independent_gate_or_future_rows": False,
    }
    gate_allowed, gate_predicted_document_clusters, gate_routing = _route_to_persistent_document_clusters(
        allowed_indices=np.where(deployment_gate_mask)[0],
        embeddings=cluster_routing_embeddings,
        score_labels=judge_behavior_document_labels,
        prototypes=persistent_prototypes,
        requires_ood=True,
    )
    future_allowed, future_predicted_document_clusters, future_routing = _route_to_persistent_document_clusters(
        allowed_indices=np.where(deployment_future_mask)[0],
        embeddings=cluster_routing_embeddings,
        score_labels=judge_behavior_document_labels,
        prototypes=persistent_prototypes,
        requires_ood=True,
    )
    document_cluster_by_document_id: dict[str, str] = {}
    for member_index, predicted_document_cluster_id in zip(candidate_document_indices.tolist(), cluster_labels.tolist()):
        if str(predicted_document_cluster_id) in persistent_prototypes:
            document_cluster_by_document_id[str(document_ids[int(member_index)])] = str(predicted_document_cluster_id)
    for routed_indices, routed_document_clusters in (
        (probe_allowed, probe_predicted_document_clusters),
        (gate_allowed, gate_predicted_document_clusters),
        (future_allowed, future_predicted_document_clusters),
    ):
        for document_index, document_cluster_id in zip(routed_indices.tolist(), routed_document_clusters.tolist()):
            document_cluster_by_document_id[str(document_ids[int(document_index)])] = str(document_cluster_id)
    predicted_document_cluster_ids = np.asarray(
        [document_cluster_by_document_id.get(str(document_id), "-1") for document_id in document_ids],
        dtype=object,
    )
    agreement_fit_mask = training_calibration_mask | training_validation_mask
    agreement_validation_mask = development_document_mask
    agreement_environment_ids = np.asarray(
        [
            f"{role}::{split}::{query_id}"
            for role, split, query_id in zip(
                document_roles.tolist(),
                splits.tolist(),
                query_ids.tolist(),
                strict=True,
            )
        ],
        dtype=str,
    )
    warning_calibrator = BehaviorWarningCalibrator(config.behavior_warning).fit(
        source_probabilities=probabilities[training_train_mask],
        source_labels=labels[training_train_mask],
        source_predictions=predictions[training_train_mask],
        calibration_probabilities=probabilities[training_calibration_mask],
        calibration_logits=judge_output.logits[training_calibration_mask],
        calibration_ood_scores=judge_behavior_scores[training_calibration_mask],
        atc_validation_probabilities=probabilities[training_validation_mask],
        atc_validation_labels=labels[training_validation_mask],
        atc_validation_predictions=predictions[training_validation_mask],
        atc_validation_environment_ids=agreement_environment_ids[
            training_validation_mask
        ],
        agreement_predictions=judge_selection.ensemble_predictions[agreement_fit_mask],
        agreement_labels=labels[agreement_fit_mask],
        agreement_environment_ids=agreement_environment_ids[agreement_fit_mask],
        agreement_validation_predictions=judge_selection.ensemble_predictions[
            agreement_validation_mask
        ],
        agreement_validation_labels=labels[agreement_validation_mask],
        agreement_validation_environment_ids=agreement_environment_ids[
            agreement_validation_mask
        ],
    )
    behavior_warnings = _behavior_warnings_by_document_cluster(
        prototypes=persistent_prototypes,
        document_ids=document_ids,
        predicted_document_cluster_ids=predicted_document_cluster_ids,
        record_mask=deployment_stream_mask,
        probabilities=probabilities,
        logits=judge_output.logits,
        ood_scores=judge_behavior_scores,
        ensemble_predictions=judge_selection.ensemble_predictions,
        calibrator=warning_calibrator,
    )
    warning_predicted_document_cluster_ids = tuple(
        document_cluster_id
        for document_cluster_id, warning in behavior_warnings.items()
        if bool(warning.get("triggered"))
    )
    # Warnings can prioritize a persistent cluster, but ASAP may disable them
    # as a hard gate when their source-validation audit is unavailable.  Every
    # route remains OOD-localized and label-free before Probe.
    warning_probe_allowed, _ = _restrict_routed_indices(
        probe_allowed,
        probe_predicted_document_clusters,
        allowed_document_cluster_ids=warning_predicted_document_cluster_ids,
    )
    safety_net_indices, safety_net_sampling = _sample_safety_net_by_rate(
        ordered_document_indices=episode_visible_indices,
        stream_mask=deployment_stream_mask,
        document_ids=document_ids,
        query_ids=query_ids,
        period_documents=int(config.safety_net_period_documents),
        labels_per_period=int(config.safety_net_labels_per_period),
        initial_documents=int(config.safety_net_initial_documents),
        initial_labels=int(config.safety_net_initial_labels),
        window_size=int(effective_window_drift_config.window_size),
        seed=int(config.seed) + 101,
    )
    safety_net = _run_safety_net(
        safety_indices=safety_net_indices,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        query_ids=query_ids,
        input_document_ids=input_document_ids,
        rater_scores=rater_scores,
        reference_metric=float(training_reference_macro["macro"][config.probe_metric]),
        reference_human_ceiling=reference_human_ceiling,
        reference_excess_human_error=reference_excess_human_error,
        tolerance=float(config.harm_tolerance),
        metric_name=config.probe_metric,
        class_values=judge.classes_,
        require_human_ceiling=bool(config.require_human_ceiling),
        paired_harmfulness_mode=str(config.paired_harmfulness_mode),
        minimum_documents=int(config.probe_min_documents),
        minimum_documents_per_query=int(config.probe_min_documents_per_query),
        expected_query_ids=np.unique(query_ids),
        sampling_metadata=safety_net_sampling,
        n_boot=int(config.bootstrap_samples),
        seed=int(config.seed) + 101,
    )
    safety_route = _safety_route_from_predicted_clusters(
        safety_net=safety_net,
        safety_indices=safety_net_indices,
        sampling_metadata=safety_net_sampling,
        document_ids=document_ids,
        predicted_document_cluster_ids=predicted_document_cluster_ids,
    )
    safety_route_cluster_ids = tuple(
        str(value) for value in safety_route.get("predicted_document_cluster_ids", [])
    )
    probe_cluster_assignments = predicted_document_cluster_ids.copy()
    safety_probe_allowed, _ = _restrict_routed_indices(
        probe_allowed,
        probe_predicted_document_clusters,
        allowed_document_cluster_ids=safety_route_cluster_ids,
    )
    probe_pool = (
        np.asarray(
            sorted(set(safety_probe_allowed.tolist()) | set(warning_probe_allowed.tolist())),
            dtype=int,
        )
        if bool(config.probe_requires_behavior_warning)
        else np.asarray(probe_allowed, dtype=int)
    )
    probe_indices = _sample_probe_indices(
        probe_allowed=probe_pool,
        predicted_document_cluster_ids=probe_cluster_assignments,
        document_ids=document_ids,
        query_ids=query_ids,
        budget=int(config.probe_budget),
        seed=config.seed,
        budget_per_cluster=True,
    )
    probe = _probe_persistent_document_clusters(
        probe_indices=probe_indices,
        predicted_document_cluster_ids=probe_cluster_assignments,
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        query_ids=query_ids,
        input_document_ids=input_document_ids,
        rater_scores=rater_scores,
        reference_metric=float(training_reference_macro["macro"][config.probe_metric]),
        reference_human_ceiling=reference_human_ceiling,
        reference_excess_human_error=reference_excess_human_error,
        tolerance=float(config.harm_tolerance),
        metric_name=config.probe_metric,
        class_values=judge.classes_,
        require_human_ceiling=bool(config.require_human_ceiling),
        paired_harmfulness_mode=str(config.paired_harmfulness_mode),
        minimum_documents=int(config.probe_min_documents),
        minimum_documents_per_query=int(config.probe_min_documents_per_query),
        expected_query_ids=np.unique(query_ids),
        fdr_alpha=float(config.harm_fdr_alpha),
        n_boot=int(config.bootstrap_samples),
        seed=int(config.seed),
    )
    harmful_predicted_document_cluster_ids = tuple(probe["harmful_predicted_document_cluster_ids"])
    has_probe_candidate = bool(has_persistent_candidate or safety_route_cluster_ids)
    adaptation_skip_reason = _adaptation_skip_reason(has_probe_candidate, probe)
    harmful_persistent_clusters = set(harmful_predicted_document_cluster_ids) & set(
        persistent_predicted_document_cluster_ids
    )
    harmful_member_indices = np.asarray(
        sorted(
            {
                int(member_index)
                for row in lifecycle_rows
                if str(row.get("document_cluster_id")) in harmful_persistent_clusters
                for member_index in row.get("member_indices", [])
            }
        ),
        dtype=int,
    )
    harmful_document_share = float(
        len(set(document_ids[harmful_member_indices].astype(str).tolist()))
        / max(len(set(document_ids[episode_visible_indices].astype(str).tolist())), 1)
    )
    pre_gate_retraining_reasons = _pre_gate_retraining_reasons(
        safety_status=str(safety_net.get("status")),
        safety_route_localized=bool(safety_route_cluster_ids),
        harmful_persistent_cluster_count=len(harmful_persistent_clusters),
        harmful_document_share=harmful_document_share,
        wide_harmful_document_share=float(config.wide_harmful_document_share),
    )
    safety_harm_unlocalized = bool(
        str(safety_net.get("status")) == "harmful" and not safety_route_cluster_ids
    )
    update_requested = bool(
        adaptation_skip_reason == "harmful_probe" and not safety_harm_unlocalized
    )
    if safety_harm_unlocalized:
        adaptation_skip_reason = "harmful_but_unlocalized_manual_review"
    full_retraining_requested = bool(update_requested and pre_gate_retraining_reasons)
    full_retraining_requested_pre_gate = bool(full_retraining_requested)
    adaptation_allowed = bool(update_requested)
    head_adaptation_allowed = bool(update_requested and not full_retraining_requested)

    actionable_gate_documents, _ = _restrict_routed_indices(
        gate_allowed,
        gate_predicted_document_clusters,
        allowed_document_cluster_ids=harmful_predicted_document_cluster_ids,
    )
    actionable_future_documents, _ = _restrict_routed_indices(
        future_allowed,
        future_predicted_document_clusters,
        allowed_document_cluster_ids=harmful_predicted_document_cluster_ids,
    )
    actionable_gate_allowed = _expand_document_indices_to_records(
        document_indices=actionable_gate_documents,
        document_ids=document_ids,
        pool_mask=deployment_gate_mask,
    )
    actionable_future_allowed = _expand_document_indices_to_records(
        document_indices=actionable_future_documents,
        document_ids=document_ids,
        pool_mask=deployment_future_mask,
    )

    action_cluster_assignments = predicted_document_cluster_ids.copy()
    if update_requested:
        adapt_indices = np.asarray(
            [
                int(index)
                for index in probe_indices.tolist()
                if str(action_cluster_assignments[int(index)])
                in harmful_predicted_document_cluster_ids
            ],
            dtype=int,
        )
        gate_indices = _sample_probe_indices(
            probe_allowed=actionable_gate_allowed,
            predicted_document_cluster_ids=action_cluster_assignments,
            document_ids=document_ids,
            query_ids=query_ids,
            budget=config.gate_budget,
            seed=config.seed + 1,
            budget_per_cluster=True,
        )
    else:
        adapt_indices = np.zeros(0, dtype=int)
        gate_indices = np.zeros(0, dtype=int)
    probe_label_document_count = len(set(document_ids[probe_indices].astype(str).tolist()))
    safety_label_document_count = len(set(document_ids[safety_net_indices].astype(str).tolist()))
    gate_label_document_count = len(set(document_ids[gate_indices].astype(str).tolist()))
    requested_target_label_document_count = len(
        set(
            document_ids[
                np.concatenate(
                    [
                        np.asarray(probe_indices, dtype=int),
                        np.asarray(safety_net_indices, dtype=int),
                        np.asarray(gate_indices, dtype=int),
                    ]
                )
            ].astype(str).tolist()
        )
    )
    training_replay_indices = _stratified_source_replay_sample(
        source_indices=np.where(training_train_mask)[0],
        query_ids=query_ids,
        labels=labels,
        budget=(
            0
            if full_retraining_requested
            else min(int(config.training_replay_budget), int(adapt_indices.size))
        ),
        seed=config.seed,
    )
    adapter = HeadAdapter(config.head_adapt)
    full_retraining = None
    full_retraining_artifacts: dict[str, str] = {}
    candidate_penultimate = u_space
    candidate_logits = judge_output.logits
    head_candidate_predictions = predictions.copy()
    head_candidate_probabilities = probabilities.copy()
    candidate_class_values = judge.classes_.copy()
    candidate_head_weight = head_weight.copy()
    candidate_head_bias = head_bias.copy()
    candidate_head_query_ids = head_query_ids.copy()
    candidate_model_artifact: str | None = None
    candidate_model_metadata: dict[str, Any] | None = None
    update_mode = "none"
    if adapt_indices.size and full_retraining_requested:
        full_retraining, full_retraining_artifacts = _fit_full_retraining_candidate(
            raw_features=judge_raw_features,
            labels=labels,
            query_ids=query_ids,
            source_train_mask=training_train_mask,
            source_validation_mask=training_validation_mask,
            deployment_adapt_indices=adapt_indices,
            base_config=config.judge,
            selection_config=config.judge_selection,
            output_dir=output_dir / "full_retraining_candidate",
        )
        full_output = full_retraining.predict_output(query_ids)
        candidate_penultimate = full_output.penultimate
        candidate_logits = full_output.logits
        head_candidate_probabilities = full_output.probabilities
        candidate_class_values = full_output.classes
        head_candidate_predictions = full_output.classes[
            np.argmax(full_output.probabilities, axis=1)
        ]
        candidate_head_weight, candidate_head_bias, candidate_head_query_ids = (
            full_retraining.model.affine_head_parameters()
        )
        candidate_model_artifact = full_retraining_artifacts["selected"]
        candidate_model_metadata = full_retraining.to_metadata()
        update_mode = "full_retraining"
    elif adapt_indices.size:
        adapter.fit(
            u_features=u_space,
            labels=labels,
            query_ids=query_ids,
            deployment_indices=adapt_indices,
            training_replay_indices=training_replay_indices,
            deployment_cluster_ids=action_cluster_assignments,
            class_values=judge.classes_,
            judge=judge,
        )
        head_candidate_predictions = adapter.predict(
            u_features=u_space, query_ids=query_ids, fallback=predictions
        )
        head_candidate_probabilities = adapter.predict_proba(
            u_features=u_space, query_ids=query_ids, fallback=probabilities
        )
        candidate_logits = adapter.predict_logits(
            u_features=u_space, query_ids=query_ids, fallback=judge_output.logits
        )
        candidate_head_weight, candidate_head_bias, candidate_head_query_ids = (
            _adapter_affine_head_parameters(
                adapter=adapter,
                base_weight=head_weight,
                base_bias=head_bias,
                base_query_ids=head_query_ids,
            )
        )
        candidate_model_metadata = adapter.to_metadata()
        update_mode = "head_adaptation"
    gate = _gate_decision(
        labels=labels,
        old_predictions=predictions,
        new_predictions=head_candidate_predictions,
        old_probabilities=probabilities,
        new_probabilities=head_candidate_probabilities,
        class_values=candidate_class_values,
        training_validation_mask=training_guard_mask,
        gate_indices=gate_indices,
        training_drop_tolerance=config.training_drop_tolerance,
        query_ids=query_ids,
        groups=input_document_ids,
        expected_query_ids=np.unique(query_ids[adapt_indices]),
        gate_min_excess_error_improvement=config.gate_min_excess_error_improvement,
        gate_max_negative_flip_rate=config.gate_max_negative_flip_rate,
        minimum_documents=int(config.gate_min_documents),
        bootstrap_samples=config.gate_bootstrap_samples,
        seed=config.seed,
    )
    retraining_reasons = list(pre_gate_retraining_reasons)
    head_only_gate = gate if update_mode == "head_adaptation" else None
    retraining_required = bool(retraining_reasons)
    retraining_executed = update_mode == "full_retraining"
    effective_training_replay_indices = np.asarray(
        sorted(
            {
                int(index)
                for metadata in adapter.optimization_.values()
                for index in metadata.get("source_replay_indices", [])
            }
        ),
        dtype=int,
    )
    deployed_predictions = head_candidate_predictions if gate["accepted"] else predictions
    deployed_probabilities = head_candidate_probabilities if gate["accepted"] else probabilities
    refresh_candidate_kwargs = (
        {
            "candidate_penultimate": candidate_penultimate,
            "candidate_logits": candidate_logits,
            "candidate_probabilities": head_candidate_probabilities,
            "candidate_class_values": candidate_class_values,
            "candidate_model_artifact": candidate_model_artifact,
            "candidate_model_metadata": candidate_model_metadata,
        }
        if update_mode == "full_retraining"
        else {}
    )
    post_update_reference = _refresh_b_reference_after_update(
        accepted=bool(gate["accepted"]),
        output_dir=output_dir,
        base_judge_fingerprint=judge_fingerprint,
        adapter=adapter,
        penultimate=u_space,
        old_logits=judge_output.logits,
        old_probabilities=probabilities,
        labels=labels,
        query_ids=query_ids,
        training_guard_mask=training_guard_mask,
        training_calibration_mask=training_calibration_mask,
        adapt_indices=adapt_indices,
        gate_indices=gate_indices,
        selected_candidate=judge_behavior_ood.selected_candidate,
        config=config.judge_ood_selection,
        window_drift_config=effective_window_drift_config,
        behavior_main_representation=behavior_main_representation,
        permutation_block_ids=permutation_block_ids,
        head_weight=candidate_head_weight,
        head_bias=candidate_head_bias,
        head_query_ids=candidate_head_query_ids,
        full_recalibration_update_interval=int(
            config.reference_full_recalibration_update_interval
        ),
        full_recalibration_shift_sigma=float(
            config.reference_full_recalibration_shift_sigma
        ),
        **refresh_candidate_kwargs,
    )
    future_indices = np.where(deployment_future_mask)[0]
    future_before = judge_metrics(
        labels[future_indices],
        predictions[future_indices],
        probabilities=probabilities[future_indices],
        class_values=judge.classes_,
    )
    future_candidate = judge_metrics(
        labels[future_indices],
        head_candidate_predictions[future_indices],
        probabilities=head_candidate_probabilities[future_indices],
        class_values=judge.classes_,
    )
    future_after = judge_metrics(
        labels[future_indices],
        deployed_predictions[future_indices],
        probabilities=deployed_probabilities[future_indices],
        class_values=judge.classes_,
    )
    future_routed_before = (
        judge_metrics(
            labels[actionable_future_allowed],
            predictions[actionable_future_allowed],
            probabilities=probabilities[actionable_future_allowed],
            class_values=judge.classes_,
        )
        if actionable_future_allowed.size
        else {}
    )
    future_routed_after = (
        judge_metrics(
            labels[actionable_future_allowed],
            deployed_predictions[actionable_future_allowed],
            probabilities=deployed_probabilities[actionable_future_allowed],
            class_values=judge.classes_,
        )
        if actionable_future_allowed.size
        else {}
    )
    future_before_macro = macro_query_judge_metrics(
        labels[future_indices],
        predictions[future_indices],
        query_ids[future_indices],
        probabilities=probabilities[future_indices],
        class_values=judge.classes_,
    )
    future_after_macro = macro_query_judge_metrics(
        labels[future_indices],
        deployed_predictions[future_indices],
        query_ids[future_indices],
        probabilities=deployed_probabilities[future_indices],
        class_values=judge.classes_,
    )
    future_group_metrics_before = _audit_judge_metrics_by_document_group(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        query_ids=query_ids,
        audit_document_group_ids=audit_document_groups,
        mask=deployment_future_mask,
        class_values=judge.classes_,
    )
    future_group_metrics_after = _audit_judge_metrics_by_document_group(
        labels=labels,
        predictions=deployed_predictions,
        probabilities=deployed_probabilities,
        query_ids=query_ids,
        audit_document_group_ids=audit_document_groups,
        mask=deployment_future_mask,
        class_values=judge.classes_,
    )
    source_guard_bwt_proxy = float(
        gate.get("new_training", {}).get("qwk", float("nan"))
        - gate.get("old_training", {}).get("qwk", float("nan"))
    )
    coral = CoralAligner()
    coral_fit_indices = adapt_indices
    coral_head_metrics: dict[str, float] = {}
    if future_indices.size and coral_fit_indices.size:
        coral.fit(u_space[training_train_mask], u_space[coral_fit_indices])
        coral_target_u = coral.transform_target(u_space[future_indices])
        coral_predictions = nearest_centroid_predict(
            source_features=u_space[training_train_mask],
            source_labels=labels[training_train_mask],
            target_features=coral_target_u,
        )
        coral_metrics = judge_metrics(
            labels[future_indices], coral_predictions, class_values=judge.classes_
        )
        coral_head_predictions = adapter.predict(
            u_features=coral_target_u,
            query_ids=query_ids[future_indices],
            fallback=coral_predictions,
        )
        coral_head_metrics = judge_metrics(
            labels[future_indices], coral_head_predictions, class_values=judge.classes_
        )
    else:
        coral_metrics = {}
    recovery = _recovery(float(future_before["accuracy"]), float(future_after["accuracy"]), float(reference_metrics["accuracy"]))

    audit_document_group_harmfulness = _audit_document_group_harmfulness(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
        query_ids=query_ids,
        audit_document_group_ids=audit_document_groups,
        input_document_ids=input_document_ids,
        rater_scores=rater_scores,
        evaluation_mask=deployment_evaluation_mask,
        reference_metric=float(training_reference_macro["macro"][config.probe_metric]),
        reference_human_ceiling=reference_human_ceiling,
        metric_name=config.probe_metric,
        tolerance=float(config.harm_tolerance),
        class_values=judge.classes_,
        require_human_ceiling=bool(config.require_human_ceiling),
        n_boot=int(config.bootstrap_samples),
        seed=int(config.seed) + 401,
    )
    clustering_monitoring_events, persistence_monitoring_events = _lifecycle_monitoring_events(
        lifecycle_rows=lifecycle_rows,
        stream_indices=monitoring_stream_indices,
    )
    full_action_monitoring_events = [
        event
        for event in persistence_monitoring_events
        if str(event.get("predicted_document_cluster_id")) in set(harmful_predicted_document_cluster_ids)
    ]
    monitoring_baselines = evaluate_monitoring_baselines(
        stream_indices=monitoring_stream_indices,
        score_labels=score_labels,
        embeddings=document_drift_embeddings,
        audit_document_group_ids=audit_document_groups,
        audit_document_group_harmfulness={
            document_group_id: payload["status"]
            for document_group_id, payload in audit_document_group_harmfulness.items()
        },
        config=MonitoringBaselineConfig(
            window_size=int(effective_window_drift_config.window_size),
            ood_rate_threshold=(
                float(config.monitoring_window_ood_rate_threshold)
                if config.monitoring_window_ood_rate_threshold is not None
                else float(config.persistence.min_share)
            ),
            cluster=config.cluster,
            persistence=config.persistence,
        ),
        clustering_detection_events=clustering_monitoring_events,
        persistence_detection_events=persistence_monitoring_events,
        full_detection_events=persistence_monitoring_events,
        full_action_events=full_action_monitoring_events,
        full_label_cost=int(probe_label_document_count + safety_label_document_count),
    )

    sample_rows = _sample_score_rows(
        records,
        knn_scores,
        score_labels,
        judge_behavior_scores,
        judge_behavior_labels,
        str(judge_behavior_ood.selected_candidate["detector"]),
        predictions,
        probabilities,
        y_ood,
        predicted_document_cluster_ids,
    )
    write_jsonl(output_dir / "sample_ood_scores.jsonl", sample_rows)
    write_jsonl(output_dir / "document_cluster_lifecycle.jsonl", lifecycle_rows)
    write_json(output_dir / "monitoring_baselines.json", monitoring_baselines)
    write_jsonl(
        output_dir / "label_cost_ledger.jsonl",
        _label_cost_rows(
            probe_indices,
            safety_net_indices,
            adapt_indices,
            gate_indices,
            document_ids,
            predicted_document_cluster_ids,
        ),
    )
    summary = {
        "artifact_type": "llm_judge_ood_sample_pipeline_summary",
        "ood_definition": "document_distribution",
        "config": config.to_dict(),
        "feature_extractors": {
            "judge_input": judge_feature_metadata,
            "input_document_A_space": document_feature_metadata,
            "shared_frozen_qwen_cache": bool(
                str(config.judge_feature_scope) == "input_document"
            ),
        },
        "preprocessing": {
            "judge": judge_selection.preprocessor.to_metadata(),
            "ood": ood_selection.preprocessor.to_metadata(),
        },
        "whitening": judge_selection.preprocessor.to_metadata(),
        "judge": judge.to_metadata(),
        "judge_selection": judge_selection.summary,
        "judge_fingerprint": judge_fingerprint,
        "judge_behavior_ood": judge_behavior_ood.to_metadata(),
        "representation_separability": separability,
        "static_reference_cache": {
            "cache_dir": config.static_reference_cache_dir,
            "components": static_cache_audit,
            "selection_used_deployment_records": False,
            "cached_payload_scope": "fitted_source_calibration_development_state_only",
            "current_scenario_dynamic_arrays_recomputed": True,
        },
        "post_update_reference": post_update_reference,
        "dual_space_drift": dual_space_drift.to_metadata(),
        "behavior_main_representation": behavior_main_representation.to_metadata(),
        "behavior_warning": {
            "calibration": warning_calibrator.to_metadata(),
            "by_predicted_document_cluster": behavior_warnings,
            "warning_predicted_document_cluster_ids": list(warning_predicted_document_cluster_ids),
            "probe_requires_behavior_warning": bool(
                config.probe_requires_behavior_warning
            ),
            "decision_boundary": (
                "warning_or_safety_route_is_required_before_persistent_cluster_probe;"
                "neither_confirms_harmfulness_without_probe"
                if bool(config.probe_requires_behavior_warning)
                else "all_persistent_ood_localized_clusters_receive_bounded_probe;"
                "warning_is_reporting_and_prioritization_only"
            ),
        },
        "human_ceiling_reference": reference_human_ceiling,
        "paired_excess_human_error_reference": reference_excess_human_error,
        "safety_net": safety_net,
        "safety_route": safety_route,
        "judge_diagnostics": judge_diagnostics,
        "document_distribution_roles": ["training", "development", "benchmark", "deployment"],
        "document_shift_types": sorted(set(document_shift_types.tolist())),
        "document_ood_ground_truth": "explicit_is_document_ood_or_legacy_deployment_fallback",
        "audit_document_group_ids": sorted(set(audit_document_groups.tolist())),
        "persistent_document_cluster_ids": list(persistent_predicted_document_cluster_ids),
        "audit_persistent_document_group_ids": list(audit_persistent_document_group_ids),
        "num_judge_records": int(len(records)),
        "num_unique_input_documents": int(len(set(document_ids.tolist()))),
        "num_training_train_documents": _unique_document_count(document_ids, training_train_mask),
        "num_training_drift_reference_documents": _unique_document_count(
            document_ids, training_drift_reference_mask
        ),
        "num_training_calibration_documents": _unique_document_count(document_ids, training_calibration_mask),
        "num_training_validation_documents": _unique_document_count(document_ids, training_validation_mask),
        "num_training_guard_documents": _unique_document_count(document_ids, training_guard_mask),
        "num_training_test_documents": _unique_document_count(document_ids, training_test_mask),
        "num_development_documents": _unique_document_count(document_ids, development_document_mask),
        "num_benchmark_test_documents": _unique_document_count(document_ids, benchmark_document_mask),
        "num_deployment_ood_evaluation_documents": _unique_document_count(document_ids, deployment_evaluation_mask),
        "num_deployment_stream_documents": _unique_document_count(document_ids, deployment_stream_mask),
        "num_deployment_offline_ood_documents": _unique_document_count(document_ids, deployment_offline_ood_mask),
        "num_deployment_adapt_documents": _unique_document_count(document_ids, deployment_adapt_mask),
        "num_deployment_gate_documents": _unique_document_count(document_ids, deployment_gate_mask),
        "num_deployment_future_documents": _unique_document_count(document_ids, deployment_future_mask),
        "num_monitoring_document_events": int(monitoring_stream_indices.size),
        "training_reference_judge_metrics": reference_metrics,
        "training_reference_judge_macro_metrics": training_reference_macro["macro"],
        "training_reference_judge_metrics_by_query": training_reference_macro["by_query"],
        "training_test_judge_metrics": judge_metrics(
            labels[training_test_mask],
            predictions[training_test_mask],
            probabilities=probabilities[training_test_mask],
            class_values=judge.classes_,
        ),
        "development_document_judge_metrics": judge_metrics(
            labels[development_document_mask],
            predictions[development_document_mask],
            probabilities=probabilities[development_document_mask],
            class_values=judge.classes_,
        ),
        "development_document_judge_macro": macro_query_judge_metrics(
            labels[development_document_mask],
            predictions[development_document_mask],
            query_ids[development_document_mask],
            probabilities=probabilities[development_document_mask],
            class_values=judge.classes_,
        ),
        "benchmark_test_judge_metrics": (
            judge_metrics(
                labels[benchmark_document_mask],
                predictions[benchmark_document_mask],
                probabilities=probabilities[benchmark_document_mask],
                class_values=judge.classes_,
            )
            if benchmark_document_mask.any()
            else None
        ),
        "benchmark_test_evidence_level": (
            "independent_confirmation" if benchmark_document_mask.any() else "unavailable"
        ),
        "deployment_before_adaptation": future_before,
        "deployment_before_adaptation_macro": future_before_macro,
        "deployment_candidate_after_adaptation": future_candidate,
        "deployment_after_adaptation": future_after,
        "deployment_after_adaptation_macro": future_after_macro,
        "deployment_routed_before_adaptation": future_routed_before,
        "deployment_routed_after_adaptation": future_routed_after,
        "audit_deployment_future_judge_metrics_by_document_group": future_group_metrics_after,
        "future_worst_group_accuracy": {
            "before": _worst_group_accuracy(future_group_metrics_before),
            "after": _worst_group_accuracy(future_group_metrics_after),
        },
        "recovery_accuracy": recovery,
        "thresholds": thresholds.to_dict(),
        "ood_metrics": static_ood,
        "ood_development": {
            "aggregate": ood_selection.development_metrics,
            "by_shift": development_ood_by_shift,
        },
        "ood_deployment": final_ood_report,
        "judge_behavior_ood_deployment": ood_metrics(
            deployment_ood_evaluation_mask[
                training_calibration_mask | deployment_ood_evaluation_mask
            ].astype(int),
            judge_behavior_scores[training_calibration_mask | deployment_ood_evaluation_mask],
        ),
        "ood_selection": ood_selection.to_metadata(),
        "lifecycle_selection": lifecycle_selection,
        "lifecycle": _lifecycle_summary(lifecycle_rows),
        "monitoring_stream": monitoring_stream_metadata,
        "audit_document_group_harmfulness": audit_document_group_harmfulness,
        "monitoring_baselines": monitoring_baselines,
        "detection_space": {
            "name": ood_selection.representation.spec.name,
            "preprocess_method": ood_selection.preprocessor.method,
            "metric": ood_selection.scorer.metric,
            "k": ood_selection.scorer.k,
            "retrieval_scope": "global_training_document_bank",
            "cluster_space": str(cluster_space_metadata["name"]),
            "cluster_dim": int(cluster_routing_embeddings.shape[1]),
            "cluster_space_metadata": cluster_space_metadata,
        },
        "document_cluster_routing": {
            "routing_uses_audit_document_group_ids": False,
            "predicted_cluster_routing_uses_audit_document_group_ids": False,
            "safety_domain_routing_uses_audit_document_group_ids": False,
            "eligible_ood_statuses": ["soft_ood", "hard_ood"],
            "probe_source": "observed_persistent_B_contributor_documents",
            "distance": f"euclidean_in_{cluster_space_metadata['name']}",
            "embedding_space": cluster_space_metadata,
            "prototypes": _persistent_prototype_metadata(persistent_prototypes),
            "probe": probe_routing,
            "adapt": {
                "source": "confirmed_harmful_probe_labels",
                "assigned_rows": int(adapt_indices.size),
                "uses_separate_deployment_adapt_labels": False,
                "deployment_adapt_role": "reserved_ood_evaluation_split_not_label_source",
            },
            "gate": gate_routing,
            "future_test": future_routing,
            "harmful_predicted_document_cluster_ids": list(harmful_predicted_document_cluster_ids),
            "actionable_rows": {
                "adapt": int(adapt_indices.size),
                "gate": int(actionable_gate_allowed.size),
                "future_test": int(actionable_future_allowed.size),
            },
        },
        "probe": probe,
        "adaptation": {
            "persistent_document_cluster_confirmed": has_persistent_candidate,
            "safety_domain_route_opened": bool(safety_route_cluster_ids),
            "update_requested": update_requested,
            "update_mode": update_mode,
            "candidate_trained": update_mode != "none",
            "candidate_deployed": bool(gate.get("accepted")) and update_mode != "none",
            "adaptation_allowed": adaptation_allowed,
            "head_adaptation_allowed": head_adaptation_allowed,
            "head_adaptation_attempted": head_only_gate is not None,
            "adaptation_skip_reason": (
                None
                if adaptation_allowed
                else adaptation_skip_reason
            ),
            "retraining_required": retraining_required,
            "full_retraining_requested": full_retraining_requested,
            "full_retraining_requested_pre_gate": full_retraining_requested_pre_gate,
            "full_retraining_executed": retraining_executed,
            "full_retraining_artifacts": full_retraining_artifacts,
            "automatic_retraining_scope": (
                "wide_covariate_drift_linear_refit_with_frozen_Qwen_only"
            ),
            "semantic_drift_automatic_diagnosis": False,
            "candidate_model_artifact": candidate_model_artifact,
            "candidate_model_metadata": candidate_model_metadata,
            "retraining_reasons": retraining_reasons,
            "harmful_document_share": harmful_document_share,
            "wide_harmful_document_share_threshold": float(config.wide_harmful_document_share),
            "multi_episode_escalation": {
                "supported": False,
                "reason": (
                    "This pipeline stops after one persistent episode and does not "
                    "persist Gate failures or diagnose semantic drift."
                ),
                "required_action_after_rejected_gate": "manual_review_or_separate_retraining_workflow",
            },
            "harmful_predicted_document_cluster_ids": list(harmful_predicted_document_cluster_ids),
            "adapter": adapter.to_metadata(),
            "coral_baseline": {
                **coral.to_metadata(),
                "fit_split": "confirmed_harmful_probe",
                "deployment_metrics": coral_metrics,
                "coral_plus_head_deployment_metrics": coral_head_metrics,
            },
            "type2_new_query": {"path": "separate_type2_pipeline", "trained_query_ids": [], "training_rows_by_query": {}},
            "gate": gate,
            "head_only_gate": head_only_gate,
            "source_guard_bwt_proxy": {
                "value": source_guard_bwt_proxy if np.isfinite(source_guard_bwt_proxy) else None,
                "definition": "single_source_task_new_guard_qwk_minus_old_guard_qwk",
                "note": "A full multi-task BWT average requires more than one completed adaptation episode.",
            },
            "training_replay": _source_replay_metadata(
                effective_training_replay_indices,
                query_ids=query_ids,
                labels=labels,
                adapt_rows=int(adapt_indices.size),
                configured_budget=int(config.training_replay_budget),
                selected_rows=int(training_replay_indices.size),
            ),
            "requested_probe_labels": int(probe_label_document_count),
            "probe_query_ratings_collected": int(probe_indices.size),
            "requested_safety_net_labels": int(safety_label_document_count),
            "safety_net_query_ratings_collected": int(safety_net_indices.size),
            "requested_adapt_labels": 0,
            "reused_probe_labels_for_adapt": int(
                len(set(document_ids[adapt_indices].astype(str).tolist()))
            ),
            "requested_gate_labels": int(gate_label_document_count),
            "gate_query_ratings_collected": int(gate_indices.size),
            "requested_total_labels": int(requested_target_label_document_count),
        },
        "outputs": {
            "scores": str(output_dir / "sample_ood_scores.jsonl"),
            "window_drift": str(output_dir / "window_drift.jsonl"),
            "lifecycle": str(output_dir / "document_cluster_lifecycle.jsonl"),
            "label_cost": str(output_dir / "label_cost_ledger.jsonl"),
            "monitoring_baselines": str(output_dir / "monitoring_baselines.json"),
            "judge_preprocessor": str(judge_preprocessor_artifact),
            "whitening": str(judge_preprocessor_artifact),
            "pca": ood_artifacts["representation"],
            "ood_preprocessor": ood_artifacts["preprocessor"],
            "ood_representation": ood_artifacts["representation"],
            "thresholds": ood_artifacts["thresholds"],
            "judge_behavior_ood": judge_behavior_ood_artifact,
            "post_update_b_reference": post_update_reference.get("reference_artifact"),
            "post_update_judge_behavior_ood": post_update_reference.get("detector_artifact"),
            "behavior_main_representation": str(behavior_main_artifact),
            "representation_separability": str(separability_artifact),
            "judge_selection": str(output_dir / "judge_selection.json"),
            "judge_diagnostics": str(output_dir / "judge_diagnostics.json"),
            "confusion_matrices": confusion_artifacts,
            "judge_checkpoints": judge_artifacts,
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    summary["outputs"]["tables"] = build_result_tables(summary, output_dir=output_dir / "tables")
    write_json(output_dir / "summary.json", summary)
    return summary


def _require_selected(mask: np.ndarray, role: str, splits: tuple[str, ...]) -> None:
    if not mask.any():
        raise ValueError(f"No {role} records selected; configured splits={list(splits)}")


def _validate_disjoint_masks(masks: tuple[np.ndarray, ...], names: tuple[str, ...]) -> None:
    for left in range(len(masks)):
        for right in range(left + 1, len(masks)):
            overlap = np.asarray(masks[left], dtype=bool) & np.asarray(masks[right], dtype=bool)
            if overlap.any():
                raise ValueError(f"Configured LLM Judge pools overlap: {names[left]} vs {names[right]}")


def _validate_disjoint_document_masks(
    document_ids: np.ndarray,
    masks: tuple[np.ndarray, ...],
    names: tuple[str, ...],
) -> None:
    documents = np.asarray(document_ids).astype(str)
    pools = [set(documents[np.asarray(mask, dtype=bool)].tolist()) for mask in masks]
    for left in range(len(pools)):
        for right in range(left + 1, len(pools)):
            overlap = pools[left] & pools[right]
            if overlap:
                raise ValueError(
                    "Configured LLM Judge document pools overlap: "
                    f"{names[left]} vs {names[right]}, first={sorted(overlap)[:5]}"
                )


def _validate_document_distribution_contract(
    records: list[JudgeRecord],
    *,
    document_ids: np.ndarray,
    document_roles: np.ndarray,
    stream_orders: np.ndarray,
) -> None:
    """Reject ambiguous document identities before any OOD split is formed."""

    for record in records:
        if not record.input_document_contract_explicit:
            raise ValueError(
                "Document OOD requires explicit input_document_id, input_document_text, and "
                "document_distribution_role fields; legacy Judge-context rows are not valid OOD input"
            )
    allowed_roles = {"training", "development", "benchmark", "deployment"}
    roles = np.asarray(document_roles).astype(str)
    invalid_roles = sorted(set(roles.tolist()) - allowed_roles)
    if invalid_roles:
        raise ValueError(
            "Document OOD requires document_distribution_role in "
            f"{sorted(allowed_roles)}, got {invalid_roles}"
        )
    arrivals = np.asarray(stream_orders, dtype=object)
    if arrivals.shape != (len(records),):
        raise ValueError("stream_orders must align with Judge records")
    by_document: dict[str, tuple[str, str, int | None]] = {}
    for record, document_id, role, stream_order in zip(
        records,
        document_ids.tolist(),
        roles.tolist(),
        arrivals.tolist(),
        strict=True,
    ):
        current = (str(role), str(record.input_document_text), None if stream_order is None else int(stream_order))
        previous = by_document.setdefault(str(document_id), current)
        if previous != current:
            raise ValueError(
                f"Input document {document_id!r} must have one distribution role and one input text across Judge rows"
            )


def _validate_judge_input_template_contract(
    *, records: list[JudgeRecord], config: SampleOODConfig, defer_records: bool = False
) -> None:
    """Bind Judge-input text and cache metadata to one frozen template."""

    if defer_records:
        return
    if str(config.judge_feature_scope) != "judge_input":
        if config.judge_prompt_template_version is not None or config.judge_prompt_template_sha256 is not None:
            raise ValueError(
                "judge_prompt_template_* may be set only when judge_feature_scope='judge_input'"
            )
        return
    configured = (
        config.judge_prompt_template_version,
        config.judge_prompt_template_sha256,
    )
    if (configured[0] is None) != (configured[1] is None):
        raise ValueError(
            "judge_prompt_template_version and judge_prompt_template_sha256 must be provided together"
        )
    observed = {
        (
            str(record.metadata.get("prompt_template_version") or ""),
            str(record.metadata.get("prompt_template_sha256") or ""),
        )
        for record in records
    }
    if len(observed) != 1:
        raise ValueError(
            "Judge-input records must have exactly one prompt-template version/hash pair; "
            f"observed={sorted(observed)}"
        )
    version, digest = next(iter(observed))
    if not version or not digest:
        raise ValueError(
            "Judge-input records are missing prompt_template_version or prompt_template_sha256"
        )
    if configured[0] is not None and (version, digest) != (str(configured[0]), str(configured[1])):
        raise ValueError(
            "Prepared Judge-input template does not match the frozen configuration: "
            f"records={(version, digest)}, config={configured}"
        )


def _validate_document_isolation(
    *,
    document_ids: np.ndarray,
    source_mask: np.ndarray,
    development_mask: np.ndarray,
    benchmark_mask: np.ndarray,
    final_mask: np.ndarray,
) -> None:
    documents = np.asarray(document_ids).astype(str)
    pools = {
        "training": set(documents[np.asarray(source_mask, dtype=bool)].tolist()),
        "development": set(documents[np.asarray(development_mask, dtype=bool)].tolist()),
        "benchmark": set(documents[np.asarray(benchmark_mask, dtype=bool)].tolist()),
        "deployment": set(documents[np.asarray(final_mask, dtype=bool)].tolist()),
    }
    for left, right in (
        ("training", "development"),
        ("training", "benchmark"),
        ("training", "deployment"),
        ("development", "benchmark"),
        ("development", "deployment"),
        ("benchmark", "deployment"),
    ):
        overlap = pools[left] & pools[right]
        if overlap:
            raise ValueError(
                f"Input-document leakage between {left} and {right} pools: {sorted(overlap)[:5]} "
                f"({len(overlap)} overlapping documents)"
            )


def _permutation_block_ids(records: list[JudgeRecord], *, key: str) -> np.ndarray:
    """Use explicit arrival/source blocks when available, otherwise documents."""

    block_ids: list[str] = []
    for record in records:
        value = record.metadata.get(str(key))
        block_ids.append(str(value) if value not in (None, "") else str(record.input_document_id))
    return np.asarray(block_ids, dtype=str)


def _rater_scores(records: list[JudgeRecord], *, key: str) -> np.ndarray:
    return np.asarray([record.metadata.get(str(key)) for record in records], dtype=object)


def _document_ood_ground_truth(records: list[JudgeRecord]) -> tuple[np.ndarray, np.ndarray]:
    """Read explicit controlled-shift truth, with legacy deployment fallback."""

    explicit = [record.metadata.get("is_document_ood") for record in records]
    has_explicit = [value is not None for value in explicit]
    if any(has_explicit) and not all(has_explicit):
        raise ValueError("is_document_ood must be present for every row or omitted for every row")
    if all(has_explicit):
        truth = np.asarray([_as_bool(value, name="is_document_ood") for value in explicit], dtype=bool)
        shifts = np.asarray(
            [str(record.metadata.get("document_shift_type") or ("far" if value else "id")) for record, value in zip(records, truth, strict=True)],
            dtype=str,
        )
    else:
        truth = np.asarray([record.document_distribution_role == "deployment" for record in records], dtype=bool)
        shifts = np.where(truth, "legacy_deployment_ood", "id").astype(str)
    by_document: dict[str, tuple[bool, str]] = {}
    for record, is_ood, shift_type in zip(records, truth.tolist(), shifts.tolist(), strict=True):
        value = (bool(is_ood), str(shift_type))
        previous = by_document.setdefault(str(record.input_document_id), value)
        if previous != value:
            raise ValueError(f"Input document {record.input_document_id!r} has inconsistent OOD truth")
    return truth, shifts


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{name} must be boolean, got {value!r}")


def _document_ood_metrics(
    *,
    document_ids: np.ndarray,
    id_mask: np.ndarray,
    target_mask: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    ids = np.asarray(document_ids).astype(str)
    id_values = np.asarray(id_mask, dtype=bool)
    target_values = np.asarray(target_mask, dtype=bool)
    score_values = np.asarray(scores, dtype=np.float64)
    if np.any(id_values & target_values):
        raise ValueError("Document OOD ID and target evaluation masks must be disjoint")
    unique_ids, first_indices, inverse, counts = np.unique(
        ids,
        return_index=True,
        return_inverse=True,
        return_counts=True,
    )
    id_counts = np.bincount(inverse, weights=id_values.astype(np.int64))
    target_counts = np.bincount(inverse, weights=target_values.astype(np.int64))
    incompatible = (
        ((id_counts != 0) & (id_counts != counts))
        | ((target_counts != 0) & (target_counts != counts))
    )
    if incompatible.any():
        document_id = str(unique_ids[int(np.flatnonzero(incompatible)[0])])
        raise ValueError(f"Input document {document_id!r} spans incompatible OOD evaluation pools")
    references = score_values[first_indices][inverse]
    score_matches = np.isclose(score_values, references, rtol=1e-6, atol=1e-6)
    if not score_matches.all():
        document_id = str(ids[int(np.flatnonzero(~score_matches)[0])])
        raise ValueError(f"Input document {document_id!r} has inconsistent OOD scores")
    selected_groups = (id_counts > 0) | (target_counts > 0)
    selected = first_indices[selected_groups]
    if selected.size == 0:
        return {"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")}
    return ood_metrics(target_values[selected].astype(int), score_values[selected])


def _save_layer_preprocessor(preprocessor: LayerPreprocessor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **preprocessor.artifact_arrays(),
        metadata_json=np.asarray(json.dumps(preprocessor.to_metadata(), ensure_ascii=False)),
    )


def _compact_judge_selection(result: JudgeSelectionResult) -> JudgeSelectionResult:
    return replace(
        result,
        processed_features=np.zeros((0, 0, 0), dtype=np.float32),
        ensemble_predictions=np.zeros(
            (0, len(result.ensemble_candidate_names)),
            dtype=np.int64,
        ),
    )


def _compact_ood_selection(result: OODSelectionResult) -> OODSelectionResult:
    return replace(
        result,
        embeddings=np.zeros((0, 0), dtype=np.float32),
        scores=np.zeros(0, dtype=np.float64),
        score_labels=np.zeros(0, dtype=object),
        input_document_ids=np.zeros(0, dtype=str),
        unique_document_count=0,
    )


def _compact_judge_ood_selection(
    result: JudgeOODSelectionResult,
) -> JudgeOODSelectionResult:
    return replace(
        result,
        scores=np.zeros(0, dtype=np.float64),
        score_labels=np.zeros(0, dtype=object),
    )


def _valid_lifecycle_selection_cache(value: Any) -> bool:
    if not isinstance(value, tuple) or len(value) != 3:
        return False
    selected_config, selection, reference = value
    if not isinstance(selected_config, WindowDriftConfig) or not isinstance(selection, dict):
        return False
    if bool(selection.get("selection_used_deployment_documents", True)):
        return False
    if reference is None:
        return not bool(selection.get("enabled"))
    return hasattr(reference, "signature") and hasattr(reference, "calibration_rows")


def _judge_diagnostics(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    query_ids: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    class_values: np.ndarray,
    majority_by_query: dict[str, Any],
) -> dict[str, Any]:
    queries = np.asarray(query_ids).astype(str)
    validation = np.asarray(validation_mask, dtype=bool)
    classes = np.asarray(class_values)
    default = classes[int(len(classes) // 2)]
    majority_predictions = np.asarray(
        [majority_by_query.get(query, default) for query in queries[validation]],
        dtype=classes.dtype,
    )
    selected_by_query: dict[str, Any] = {}
    majority_by_query_report: dict[str, Any] = {}
    for query_id in sorted(set(queries[validation].tolist())):
        local = validation & (queries == query_id)
        selected_by_query[query_id] = confusion_matrix_report(
            labels[local], predictions[local], class_values=classes
        )
        local_majority = np.asarray(
            [majority_by_query.get(query_id, default)] * int(local.sum()),
            dtype=classes.dtype,
        )
        majority_by_query_report[query_id] = confusion_matrix_report(
            labels[local], local_majority, class_values=classes
        )
    return {
        "fit_rows": int(np.asarray(train_mask, dtype=bool).sum()),
        "validation_rows": int(validation.sum()),
        "selected": {
            "overall": confusion_matrix_report(
                labels[validation], predictions[validation], class_values=classes
            ),
            "by_query": selected_by_query,
            "mean_confidence": float(np.max(probabilities[validation], axis=1).mean()),
        },
        "majority_baseline": {
            "overall": confusion_matrix_report(
                labels[validation], majority_predictions, class_values=classes
            ),
            "by_query": majority_by_query_report,
        },
    }


def _write_confusion_artifacts(diagnostics: dict[str, Any], output_dir: Path) -> dict[str, str]:
    import pandas as pd

    paths: dict[str, str] = {}
    for name, payload in (
        ("selected", diagnostics["selected"]["overall"]),
        ("majority", diagnostics["majority_baseline"]["overall"]),
    ):
        classes = [str(value) for value in payload["classes"]]
        path = output_dir / f"confusion_matrix_{name}.csv"
        pd.DataFrame(payload["matrix"], index=classes, columns=classes).to_csv(path, index_label="true\\pred")
        paths[f"{name}_csv"] = str(path)
    distribution_rows: list[dict[str, Any]] = []
    selected = diagnostics["selected"]["overall"]
    majority = diagnostics["majority_baseline"]["overall"]
    for class_value in selected["classes"]:
        key = str(class_value)
        distribution_rows.append(
            {
                "class": class_value,
                "true_count": selected["true_distribution"].get(key, 0),
                "selected_prediction_count": selected["prediction_distribution"].get(key, 0),
                "majority_prediction_count": majority["prediction_distribution"].get(key, 0),
            }
        )
    distribution_path = output_dir / "prediction_distribution.csv"
    pd.DataFrame(distribution_rows).to_csv(distribution_path, index=False)
    paths["prediction_distribution"] = str(distribution_path)
    try:
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for axis, title, payload in (
            (axes[0], "Selected Judge", selected),
            (axes[1], "Per-query majority", majority),
        ):
            matrix = np.asarray(payload["matrix"], dtype=int)
            image = axis.imshow(matrix, cmap="Blues")
            axis.set_title(title)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
            axis.set_xticks(range(len(payload["classes"])), payload["classes"])
            axis.set_yticks(range(len(payload["classes"])), payload["classes"])
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure_path = output_dir / "confusion_matrices.png"
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
        paths["figure"] = str(figure_path)
    except Exception as error:
        paths["figure_skipped_reason"] = f"{type(error).__name__}: {error}"
    return paths


def _audit_judge_metrics_by_document_group(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    query_ids: np.ndarray,
    audit_document_group_ids: np.ndarray,
    mask: np.ndarray,
    class_values: np.ndarray,
) -> dict[str, Any]:
    audit_groups = np.asarray(audit_document_group_ids).astype(str)
    selected = np.asarray(mask, dtype=bool)
    return {
        document_group_id: {
            "overall": judge_metrics(
                labels[selected & (audit_groups == document_group_id)],
                predictions[selected & (audit_groups == document_group_id)],
                probabilities=probabilities[selected & (audit_groups == document_group_id)],
                class_values=class_values,
            ),
            "macro_query": macro_query_judge_metrics(
                labels[selected & (audit_groups == document_group_id)],
                predictions[selected & (audit_groups == document_group_id)],
                query_ids[selected & (audit_groups == document_group_id)],
                probabilities=probabilities[selected & (audit_groups == document_group_id)],
                class_values=class_values,
            ),
        }
        for document_group_id in sorted(set(audit_groups[selected].tolist()))
    }


def _worst_group_accuracy(metrics_by_group: dict[str, Any]) -> dict[str, Any]:
    candidates = [
        (str(document_group_id), float(payload.get("overall", {}).get("accuracy", float("nan"))))
        for document_group_id, payload in metrics_by_group.items()
    ]
    finite = [(document_group_id, value) for document_group_id, value in candidates if np.isfinite(value)]
    if not finite:
        return {"audit_document_group_id": None, "accuracy": None}
    document_group_id, accuracy = min(finite, key=lambda item: (item[1], item[0]))
    return {"audit_document_group_id": document_group_id, "accuracy": accuracy}


def _select_window_drift_configuration(
    *,
    config: SampleOODConfig,
    document_embeddings: np.ndarray,
    behavior_embeddings: np.ndarray,
    document_ids: np.ndarray,
    source_document_indices: np.ndarray,
    calibration_document_indices: np.ndarray,
    development_document_indices: np.ndarray,
    source_behavior_indices: np.ndarray,
    calibration_behavior_indices: np.ndarray,
    permutation_block_ids: np.ndarray | None = None,
    behavior_ood_scores: np.ndarray | None = None,
) -> tuple[WindowDriftConfig, dict[str, Any], Any | None]:
    """Select sequential settings on Calibration+Development, never deployment.

    Detector families are already selected upstream on Development. This stage
    selects window length, alpha-spending budget, and persistence length by
    enforcing an ID-calibration false-alert constraint, then preferring B-space
    persistent detection on independent Development documents.
    """

    base = replace(config.window_drift, seed=int(config.seed))
    if not config.tune_lifecycle_on_development:
        return (
            base,
            {
                "enabled": False,
                "selection_scope": "fixed_final_design_window_config",
                "selection_used_deployment_documents": False,
                "candidate_results": [],
            },
            None,
        )
    candidate_windows = (
        tuple(sorted(set(int(value) for value in config.lifecycle_window_sizes)))
        or (int(base.window_size),)
    )
    candidate_minimums = (
        tuple(sorted(set(int(value) for value in config.lifecycle_minimum_consecutive_windows)))
        or (int(base.minimum_consecutive_windows),)
    )
    candidate_fwers = (
        tuple(sorted(set(float(value) for value in config.lifecycle_alpha_fwers)))
        or (float(base.alpha_fwer),)
    )
    candidate_spendings = (
        tuple(sorted(set(str(value) for value in config.lifecycle_alpha_spendings)))
        or (str(base.alpha_spending),)
    )
    block_ids = (
        np.asarray(permutation_block_ids).astype(str)
        if permutation_block_ids is not None
        else np.asarray(document_ids).astype(str)
    )
    rows: list[dict[str, Any]] = []
    references_by_window: dict[int, Any] = {}
    chosen: WindowDriftConfig | None = None
    chosen_key: tuple[bool, float, float, float, int, int, float, str] | None = None
    for window_size in candidate_windows:
        # Selection needs every independent Development window.  Deployment
        # itself still stops at the first persistent episode.
        mmd_config = replace(
            base,
            window_size=int(window_size),
            stop_after_first_persistent=False,
            # Power is a source/calibration-only diagnostic for the selected
            # final lifecycle configuration.  It cannot affect candidate
            # eligibility or the selection objective below, so running it for
            # every candidate only repeats the same expensive simulation.
            # ``chosen`` is still built from ``base`` and the final deployment
            # monitor therefore computes and reports the requested analysis.
            power_enabled=False,
        )
        preflight = _block_calibration_preflight(
            config=mmd_config,
            document_ids=document_ids,
            source_document_indices=source_document_indices,
            calibration_document_indices=calibration_document_indices,
            source_behavior_indices=source_behavior_indices,
            calibration_behavior_indices=calibration_behavior_indices,
            block_ids=block_ids,
        )
        if not bool(preflight["formal_valid"]) and not bool(
            mmd_config.allow_nominal_fallback_for_smoke
        ):
            for minimum_windows, alpha_fwer, alpha_spending in product(
                candidate_minimums,
                candidate_fwers,
                candidate_spendings,
            ):
                candidate = replace(
                    base,
                    window_size=int(window_size),
                    minimum_consecutive_windows=int(minimum_windows),
                    alpha_fwer=float(alpha_fwer),
                    alpha_spending=str(alpha_spending),
                )
                rows.append(
                    {
                        "window_drift": candidate.to_dict(),
                        "effective_sequential_config": None,
                        "calibration_hard_alert_rate": None,
                        "calibration_window_count": int(
                            min(
                                int(preflight["a_valid_windows"]),
                                int(preflight["b_valid_windows"]),
                            )
                        ),
                        "calibration_failure_reasons": list(preflight["failure_reasons"]),
                        "development_hard_drift_window_rate": None,
                        "development_persistent_document_coverage": None,
                        "calibration_constraint_satisfied": False,
                        "formal_candidate_eligible": False,
                        "block_calibration_audit": preflight,
                        "selection_used_deployment_documents": False,
                    }
                )
            continue
        result = run_dual_space_drift_monitor(
            document_embeddings=document_embeddings,
            behavior_embeddings=behavior_embeddings,
            document_ids=document_ids,
            source_document_indices=source_document_indices,
            calibration_document_indices=calibration_document_indices,
            stream_document_indices=development_document_indices,
            source_behavior_indices=source_behavior_indices,
            calibration_behavior_indices=calibration_behavior_indices,
            config=mmd_config,
            permutation_block_ids=permutation_block_ids,
            behavior_ood_scores=behavior_ood_scores,
        )
        references_by_window[int(window_size)] = result.reference
        block_calibration_audit = _observed_block_calibration_audit(result, preflight)
        calibration_pvalues = np.asarray(
            [
                row["B"]["p_value"]
                for row in result.calibration_rows
                if "B" in row and row["B"].get("p_value") is not None
            ],
            dtype=float,
        )
        development_rows = [row for row in result.window_rows if "B" in row]
        development_hard_rate = (
            float(np.mean([row["B"]["status"] == "hard_drift" for row in development_rows]))
            if development_rows
            else 0.0
        )
        for minimum_windows, alpha_fwer, alpha_spending in product(
            candidate_minimums,
            candidate_fwers,
            candidate_spendings,
        ):
            candidate = replace(
                base,
                window_size=int(window_size),
                minimum_consecutive_windows=int(minimum_windows),
                alpha_fwer=float(alpha_fwer),
                alpha_spending=str(alpha_spending),
            )
            effective = derive_effective_sequential_config(
                candidate, result.calibrated_thresholds
            )
            strict_sequential_calibration = bool(
                candidate.require_sequential_fwer_calibration
            )
            sequential_audit = dict(
                result.calibrated_thresholds.get("B", {}).get(
                    "sequential_fwer_audit", {}
                )
            )
            if strict_sequential_calibration:
                calibration_hard_alert_rate = float(
                    sequential_audit.get(
                        "window_false_positive_rate_alpha_0_01", float("nan")
                    )
                )
                calibration_hard_alert_count = int(
                    round(
                        calibration_hard_alert_rate
                        * int(sequential_audit.get("window_count", 0))
                    )
                )
                calibration_hard_alert_ci95 = sequential_audit.get(
                    "window_false_positive_rate_alpha_0_01_ci95"
                )
                calibration_hard_alert_limit = float(candidate.hard_alpha)
            else:
                calibration_hard_alert_rate = (
                    float(
                        np.mean(
                            calibration_pvalues
                            <= float(effective.calibrated_b_hard_alpha)
                        )
                    )
                    if calibration_pvalues.size
                    else float("nan")
                )
                calibration_hard_alert_count = int(
                    np.sum(
                        calibration_pvalues
                        <= float(effective.calibrated_b_hard_alpha)
                    )
                )
                calibration_hard_alert_ci95 = (
                    wilson_interval(
                        calibration_hard_alert_count, int(calibration_pvalues.size)
                    )
                    if calibration_pvalues.size
                    else None
                )
                calibration_hard_alert_limit = max(
                    float(candidate.hard_alpha),
                    1.0 / float(calibration_pvalues.size)
                    if calibration_pvalues.size
                    else 1.0,
                )
            development_persistent_coverage = _development_persistent_coverage(
                result.window_rows,
                config=effective.tracker_config(candidate),
                development_document_count=len(development_document_indices),
            )
            if strict_sequential_calibration:
                false_alert_constraint_satisfied = bool(sequential_audit.get("valid"))
                calibration_hard_alert_limit_rule = (
                    "episode_fwer_wilson_95_upper_bound_lte_alpha_fwer"
                )
            else:
                false_alert_constraint_satisfied = bool(
                    np.isfinite(calibration_hard_alert_rate)
                    and calibration_hard_alert_rate
                    <= float(calibration_hard_alert_limit) + 1e-12
                )
                calibration_hard_alert_limit_rule = (
                    "development_selection_max(nominal_hard_alpha,1/N_valid_windows)"
                )
            calibration_valid = bool(
                effective.calibration_valid
                and false_alert_constraint_satisfied
                and bool(block_calibration_audit["formal_valid"])
            )
            row = {
                "window_drift": candidate.to_dict(),
                "effective_sequential_config": effective.to_dict(),
                "calibration_hard_alert_rate": calibration_hard_alert_rate,
                "calibration_hard_alert_count": calibration_hard_alert_count,
                "calibration_hard_alert_rate_ci95": calibration_hard_alert_ci95,
                "calibration_hard_alert_limit": float(calibration_hard_alert_limit),
                "calibration_hard_alert_limit_rule": calibration_hard_alert_limit_rule,
                "sequential_fwer_audit": sequential_audit,
                "calibration_window_count": int(
                    effective.calibration_window_count
                ),
                "calibration_failure_reasons": list(
                    effective.calibration_failure_reasons
                ),
                "development_hard_drift_window_rate": development_hard_rate,
                "development_persistent_document_coverage": development_persistent_coverage,
                "calibration_constraint_satisfied": calibration_valid,
                "formal_candidate_eligible": bool(block_calibration_audit["formal_valid"]),
                "block_calibration_audit": block_calibration_audit,
                "selection_used_deployment_documents": False,
            }
            rows.append(row)
            key = (
                calibration_valid,
                development_persistent_coverage,
                development_hard_rate,
                -calibration_hard_alert_rate,
                -int(window_size),
                -int(minimum_windows),
                -float(alpha_fwer),
                str(alpha_spending),
            )
            if chosen_key is None or key > chosen_key:
                chosen = candidate
                chosen_key = key
    if chosen is None:
        raise ValueError(
            "No lifecycle window candidate has enough independent calibration blocks; "
            "remove invalid window sizes or provide additional training_calibration data."
        )
    selected = next(row for row in rows if row["window_drift"] == chosen.to_dict())
    if bool(chosen.require_sequential_fwer_calibration) and not bool(
        selected.get("calibration_constraint_satisfied")
    ):
        raise ValueError(
            "No lifecycle window candidate passed independent H0 calibration; "
            "formal deployment decisions are disabled until calibration passes."
        )
    return (
        chosen,
        {
            "enabled": True,
            "selection_scope": "training_calibration_and_development_only",
            "objective": [
                "calibration_hard_alert_constraint",
                "development_persistent_B_coverage",
                "development_B_hard_drift_rate",
                "negative_calibration_hard_alert_rate",
                "window_size_W_min_alpha_spending_selected_on_development",
                "smaller_window_tiebreak",
            ],
            "selected_metrics": selected,
            "candidate_results": rows,
            "selection_used_deployment_documents": False,
            "final_monitor_reuses_selected_source_calibration_reference": True,
        },
        references_by_window.get(int(chosen.window_size)),
    )


def _block_calibration_preflight(
    *,
    config: WindowDriftConfig,
    document_ids: np.ndarray,
    source_document_indices: np.ndarray,
    calibration_document_indices: np.ndarray,
    source_behavior_indices: np.ndarray,
    calibration_behavior_indices: np.ndarray,
    block_ids: np.ndarray,
) -> dict[str, Any]:
    """Audit block capacity before a candidate can use Development data.

    The formal protocol treats a document as the smallest ASAP arrival block.
    A candidate with insufficient calibration blocks is recorded but is never
    evaluated on Development or deployed.  The smoke-only fallback remains an
    explicitly non-formal path in the monitor itself.
    """

    blocks = np.asarray(block_ids).astype(str)
    source_documents = np.asarray(source_document_indices, dtype=int)
    calibration_documents = ordered_calibration_document_indices(
        np.asarray(calibration_document_indices, dtype=int),
        np.asarray(document_ids).astype(str),
        config,
    )
    source_behavior = np.asarray(source_behavior_indices, dtype=int)
    calibration_behavior = np.asarray(calibration_behavior_indices, dtype=int)
    required_blocks = max(5, int(config.c2st_folds))
    source_a_blocks = int(len(np.unique(blocks[source_documents])))
    source_b_blocks = int(len(np.unique(blocks[source_behavior])))
    calibration_blocks = int(len(np.unique(blocks[calibration_documents])))
    calibration_behavior_blocks = int(len(np.unique(blocks[calibration_behavior])))
    a_valid_windows = 0
    b_valid_windows = 0
    target_block_counts: list[int] = []
    insufficient_target_blocks = False
    for start in range(0, int(calibration_documents.size), int(config.window_size)):
        documents = calibration_documents[start : start + int(config.window_size)]
        if documents.size < int(config.minimum_window_documents):
            continue
        target_blocks = int(len(np.unique(blocks[documents])))
        target_block_counts.append(target_blocks)
        if target_blocks < required_blocks:
            insufficient_target_blocks = True
            continue
        a_valid_windows += 1
        # B-space can have multiple Judge records per document, but their
        # permutation blocks remain document-level and therefore share this
        # independently auditable count.
        b_valid_windows += 1
    actual_c2st_folds = (
        int(
            min(
                int(config.c2st_folds), source_a_blocks, source_b_blocks, min(target_block_counts)
            )
        )
        if bool(config.c2st_enabled) and target_block_counts
        else None
    )
    failures: list[str] = []
    if source_a_blocks < required_blocks or source_b_blocks < required_blocks:
        failures.append("insufficient_source_blocks")
    if a_valid_windows < int(config.minimum_valid_calibration_windows):
        failures.append("A_insufficient_valid_calibration_windows")
    if b_valid_windows < int(config.minimum_valid_calibration_windows):
        failures.append("B_insufficient_valid_calibration_windows")
    if bool(config.c2st_enabled) and actual_c2st_folds != int(config.c2st_folds):
        failures.append("c2st_folds_below_configured")
    if insufficient_target_blocks:
        failures.append("insufficient_target_blocks")
    return {
        "window_size": int(config.window_size),
        "source_blocks": int(min(source_a_blocks, source_b_blocks)),
        "source_A_blocks": source_a_blocks,
        "source_B_blocks": source_b_blocks,
        "calibration_documents": int(calibration_documents.size),
        "calibration_blocks": calibration_blocks,
        "calibration_B_blocks": calibration_behavior_blocks,
        "a_valid_windows": int(a_valid_windows),
        "b_valid_windows": int(b_valid_windows),
        "actual_c2st_folds": actual_c2st_folds,
        "configured_c2st_folds": int(config.c2st_folds) if config.c2st_enabled else None,
        "insufficient_target_blocks": bool(insufficient_target_blocks),
        "minimum_valid_calibration_windows": int(config.minimum_valid_calibration_windows),
        "nominal_fallback_for_smoke": False,
        "formal_valid": not failures,
        "failure_reasons": failures,
    }


def _observed_block_calibration_audit(
    result: Any,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Replace preflight counts with the calibration results actually used."""

    rows = list(result.calibration_rows)
    a_valid_windows = sum(
        1 for row in rows if isinstance(row.get("A"), dict) and row["A"].get("p_value") is not None
    )
    b_valid_windows = sum(
        1 for row in rows if isinstance(row.get("B"), dict) and row["B"].get("p_value") is not None
    )
    folds = [
        int(space["c2st"]["folds"])
        for row in rows
        for space in (row.get("A", {}), row.get("B", {}))
        if isinstance(space, dict)
        and isinstance(space.get("c2st"), dict)
        and space["c2st"].get("folds") is not None
    ]
    insufficient_target_blocks = any(
        row.get("status") == "insufficient_target_blocks"
        or row.get("A", {}).get("status") == "insufficient_target_blocks"
        or row.get("B", {}).get("status") == "insufficient_target_blocks"
        for row in rows
    )
    audit = {
        **preflight,
        "a_valid_windows": int(a_valid_windows),
        "b_valid_windows": int(b_valid_windows),
        "actual_c2st_folds": min(folds) if folds else preflight["actual_c2st_folds"],
        "insufficient_target_blocks": bool(insufficient_target_blocks),
        "nominal_fallback_for_smoke": bool(
            result.effective_sequential_config.nominal_fallback_for_smoke
        ),
    }
    failures = [
        reason
        for reason in preflight["failure_reasons"]
        if reason
        not in {
            "A_insufficient_valid_calibration_windows",
            "B_insufficient_valid_calibration_windows",
            "insufficient_target_blocks",
            "c2st_folds_below_configured",
        }
    ]
    if int(audit["a_valid_windows"]) < int(audit["minimum_valid_calibration_windows"]):
        failures.append("A_insufficient_valid_calibration_windows")
    if int(audit["b_valid_windows"]) < int(audit["minimum_valid_calibration_windows"]):
        failures.append("B_insufficient_valid_calibration_windows")
    if bool(result.config.c2st_enabled) and audit["actual_c2st_folds"] != int(
        result.config.c2st_folds
    ):
        failures.append("c2st_folds_below_configured")
    if bool(audit["insufficient_target_blocks"]):
        failures.append("insufficient_target_blocks")
    if bool(audit["nominal_fallback_for_smoke"]):
        failures.append("nominal_fallback_for_smoke")
    audit["failure_reasons"] = failures
    audit["formal_valid"] = not failures and bool(
        result.effective_sequential_config.calibration_valid
    )
    return audit


def _development_persistent_coverage(
    window_rows: list[dict[str, Any]],
    *,
    config: WindowDriftConfig,
    development_document_count: int,
) -> float:
    """Replay only alpha spending; MMD p-values do not depend on this setting."""

    tracker = AlphaSpendingTracker(config)
    active_segment: list[list[int]] = []
    persistent_documents: set[int] = set()
    for row in window_rows:
        b_result = row.get("B")
        p_value = b_result.get("p_value") if isinstance(b_result, dict) else None
        if p_value is None or not np.isfinite(float(p_value)):
            tracker.break_consecutive_segment()
            active_segment = []
            continue
        sequential = tracker.update(
            window_index=int(row["window_index"]),
            p_value=float(p_value),
        )
        if sequential["b_sequential_reject"]:
            active_segment.append([int(index) for index in row.get("document_indices", [])])
        else:
            active_segment = []
        if sequential["persistent_b_drift"]:
            persistent_documents.update(index for segment in active_segment for index in segment)
    return float(len(persistent_documents) / max(int(development_document_count), 1))


def _ordered_monitoring_document_indices(
    *,
    training_indices: np.ndarray,
    deployment_indices: np.ndarray,
    input_document_ids: np.ndarray,
    window_size: int,
    stream_orders: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a deterministic monitoring chronology with one event per document."""

    training = np.asarray(training_indices, dtype=int)
    deployment = np.asarray(deployment_indices, dtype=int)
    document_ids = np.asarray(input_document_ids).astype(str)
    arrivals = (
        np.asarray(stream_orders, dtype=object)
        if stream_orders is not None
        else np.asarray([None] * len(document_ids), dtype=object)
    )
    if arrivals.shape != (len(document_ids),):
        raise ValueError("stream_orders must align with input_document_ids")
    if np.intersect1d(training, deployment).size:
        raise ValueError("Training preamble and deployment stream overlap")

    training_unique = _first_indices_by_document(training, document_ids)
    deployment_unique = _first_indices_by_document(deployment, document_ids)
    if set(document_ids[training_unique].tolist()) & set(document_ids[deployment_unique].tolist()):
        raise ValueError("Monitoring training and deployment pools share input documents")
    training_order = sorted(training_unique.tolist(), key=lambda index: (document_ids[int(index)], int(index)))
    deployment_arrivals = [arrivals[int(index)] for index in deployment_unique.tolist()]
    if all(value is not None for value in deployment_arrivals):
        deployment_documents = [
            str(document_ids[int(index)])
            for index in sorted(
                deployment_unique.tolist(),
                key=lambda index: (int(arrivals[int(index)]), str(document_ids[int(index)]), int(index)),
            )
        ]
        deployment_order_source = "explicit_stream_order_or_arrival_index"
    elif any(value is not None for value in deployment_arrivals):
        raise ValueError("deployment stream ordering is mixed: every document needs stream_order/arrival_index")
    else:
        deployment_documents = sorted(set(document_ids[deployment_unique].tolist()))
        deployment_order_source = "input_document_id_lexicographic_fallback"
    documents_per_block = max(1, int(window_size))
    deployment_by_document = {document_ids[int(index)]: int(index) for index in deployment_unique.tolist()}
    deployment_order: list[int] = []
    for start in range(0, len(deployment_documents), documents_per_block):
        for document_id in deployment_documents[start : start + documents_per_block]:
            deployment_order.append(deployment_by_document[document_id])
    order = np.asarray(training_order + deployment_order, dtype=int)
    expected = set(training_unique.tolist()) | set(deployment_unique.tolist())
    if len(order) != len(expected) or set(order.tolist()) != expected:
        raise RuntimeError("Monitoring document ordering dropped or duplicated input documents")
    return order, {
        "ordering": "training_document_preamble_then_deployment_document_blocks",
        "training_preamble_documents": int(training_unique.size),
        "deployment_stream_documents": int(deployment_unique.size),
        "input_document_events": int(order.size),
        "document_block_size": int(documents_per_block),
        "window_size": int(window_size),
        "deployment_order_source": deployment_order_source,
    }


def _first_indices_by_document(indices: np.ndarray, document_ids: np.ndarray) -> np.ndarray:
    selected: list[int] = []
    seen: set[str] = set()
    for index in np.asarray(indices, dtype=int).tolist():
        document_id = str(document_ids[int(index)])
        if document_id not in seen:
            selected.append(int(index))
            seen.add(document_id)
    return np.asarray(selected, dtype=int)


def _broadcast_document_ood_labels(
    score_labels: np.ndarray,
    document_ids: np.ndarray,
) -> np.ndarray:
    """Aggregate record-level ViM status to documents and broadcast it to rows."""

    labels = np.asarray(score_labels).astype(str)
    documents = np.asarray(document_ids).astype(str)
    if labels.shape != documents.shape:
        raise ValueError("Judge behavior labels and document IDs must align")
    severity = {"id": 0, "soft_ood": 1, "hard_ood": 2}
    unknown = sorted(set(labels.tolist()) - set(severity))
    if unknown:
        raise ValueError(f"Unknown Judge behavior OOD labels: {unknown}")
    maximum_by_document: dict[str, int] = {}
    for document_id, label in zip(documents.tolist(), labels.tolist(), strict=True):
        maximum_by_document[document_id] = max(
            maximum_by_document.get(document_id, 0),
            severity[label],
        )
    by_severity = np.asarray(["id", "soft_ood", "hard_ood"], dtype=object)
    return np.asarray(
        [by_severity[maximum_by_document[document_id]] for document_id in documents.tolist()],
        dtype=object,
    )


def _cluster_routing_embeddings(
    cluster_config: ClusterConfig,
    *,
    document_embeddings: np.ndarray,
    behavior_embeddings: np.ndarray,
    document_ids: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one frozen geometry for clustering and every downstream route."""

    document_values = np.asarray(document_embeddings, dtype=np.float64)
    behavior_values = np.asarray(behavior_embeddings, dtype=np.float64)
    documents = np.asarray(document_ids).astype(str)
    if document_values.ndim != 2 or behavior_values.ndim != 2:
        raise ValueError("Cluster embedding inputs must be two-dimensional")
    if len(document_values) != len(documents) or len(behavior_values) != len(documents):
        raise ValueError("Cluster embedding inputs must align with document IDs")
    if not np.all(np.isfinite(document_values)) or not np.all(np.isfinite(behavior_values)):
        raise ValueError("Cluster embedding inputs must be finite")
    if cluster_config.method not in {"hybrid", "hdbscan_knn_expand"}:
        return document_values, {
            "name": "A_input_document_embedding",
            "family": "input_document_A_space",
            "document_aggregation": "feature_store_document_embedding",
            "normalization": "source_fitted_ood_representation",
            "dimension": int(document_values.shape[1]),
        }

    # The latest hybrid was selected in B residual direction space.  SummEval
    # can have several Judge rows per document, so aggregate its unit residual
    # directions once and broadcast the document vector to every row before
    # clustering or Gate/Future routing.
    unit = behavior_values / np.maximum(
        np.linalg.norm(behavior_values, axis=1, keepdims=True), 1e-12
    )
    summed_by_document: dict[str, np.ndarray] = {}
    for document_id, vector in zip(documents.tolist(), unit, strict=True):
        if document_id not in summed_by_document:
            summed_by_document[document_id] = np.zeros(behavior_values.shape[1], dtype=np.float64)
        summed_by_document[document_id] += vector
    document_directions = {
        document_id: vector / max(float(np.linalg.norm(vector)), 1e-12)
        for document_id, vector in summed_by_document.items()
    }
    values = np.stack(
        [document_directions[document_id] for document_id in documents.tolist()]
    )
    return values, {
        "name": "B_vim_residual_direction_document_mean",
        "family": "judge_behavior_B_space",
        "document_aggregation": "mean_unit_residual_direction_then_l2_normalize",
        "normalization": "l2_per_record_then_l2_per_document",
        "source_representation": "vim_source_subspace_residual_vector",
        "dimension": int(values.shape[1]),
        "hybrid_radius_multiplier": float(cluster_config.hybrid_radius_multiplier),
        "hybrid_radius_quantile": float(cluster_config.hybrid_radius_quantile),
    }


def _behavior_drift_embeddings(
    *,
    penultimate: np.ndarray,
    scorer: ViMScorer,
    probabilities: np.ndarray | None = None,
    ood_scores: np.ndarray | None = None,
) -> np.ndarray:
    """Compatibility helper for the residual-vector B-main view."""

    del probabilities, ood_scores
    return BehaviorMainRepresentation(rank=int(scorer.rank)).fit(
        penultimate,
        scorer=scorer,
    ).transform(penultimate)


def _refresh_b_reference_after_update(
    *,
    accepted: bool,
    output_dir: Path,
    base_judge_fingerprint: str,
    adapter: HeadAdapter,
    penultimate: np.ndarray,
    old_logits: np.ndarray,
    old_probabilities: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    training_guard_mask: np.ndarray,
    training_calibration_mask: np.ndarray,
    adapt_indices: np.ndarray,
    gate_indices: np.ndarray,
    selected_candidate: dict[str, Any],
    config: JudgeOODSelectionConfig,
    window_drift_config: WindowDriftConfig,
    behavior_main_representation: BehaviorMainRepresentation,
    permutation_block_ids: np.ndarray,
    full_recalibration_update_interval: int = 5,
    full_recalibration_shift_sigma: float = 1.0,
    head_weight: np.ndarray | None = None,
    head_bias: np.ndarray | None = None,
    head_query_ids: np.ndarray | None = None,
    candidate_penultimate: np.ndarray | None = None,
    candidate_logits: np.ndarray | None = None,
    candidate_probabilities: np.ndarray | None = None,
    candidate_class_values: np.ndarray | None = None,
    candidate_model_artifact: str | None = None,
    candidate_model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal next-episode B reference after an accepted head update."""

    history_path = output_dir / "b_reference_history.json"
    history = _load_b_reference_history(history_path)
    if not accepted:
        return {
            "status": "not_refreshed_gate_rejected",
            "a_reference": "unchanged_frozen_backbone",
            "b_reference": "unchanged_deployed_head",
            "reference_artifact": None,
            "detector_artifact": None,
            "candidate_model_artifact": None,
            "history_artifact": str(history_path) if history_path.exists() else None,
            "cumulative_accepted_updates": int(history["cumulative_accepted_updates"]),
            "full_recalibration_required": False,
        }
    cumulative_updates = int(history["cumulative_accepted_updates"]) + 1
    version = int(history["latest_version"]) + 1
    external_candidate = candidate_logits is not None
    if candidate_logits is None:
        candidate_logits = adapter.predict_logits(
            u_features=penultimate,
            query_ids=query_ids,
            fallback=old_logits,
        )
    if candidate_probabilities is None:
        candidate_probabilities = adapter.predict_proba(
            u_features=penultimate,
            query_ids=query_ids,
            fallback=old_probabilities,
        )
    updated_penultimate = np.asarray(
        penultimate if candidate_penultimate is None else candidate_penultimate,
        dtype=np.float64,
    )
    if updated_penultimate.ndim != 2 or len(updated_penultimate) != len(labels):
        raise ValueError("Accepted candidate penultimate features must be aligned [N,D]")
    # Recompute all source and calibration behavior with the accepted head.
    # The adapter fallback preserves old outputs for query dimensions that were
    # not part of this adaptation episode.
    updated_logits = np.asarray(candidate_logits).copy()
    updated_probabilities = np.asarray(candidate_probabilities).copy()
    reference_mask = np.asarray(training_guard_mask, dtype=bool).copy()
    reference_mask[np.asarray(adapt_indices, dtype=int)] = True
    calibration_mask = np.asarray(training_calibration_mask, dtype=bool).copy()
    refreshed = refit_selected_judge_ood_detector(
        selected_candidate=selected_candidate,
        penultimate=updated_penultimate,
        logits=updated_logits,
        labels=labels,
        class_values=np.asarray(
            adapter.classes_ if candidate_class_values is None else candidate_class_values
        ),
        reference_mask=reference_mask,
        calibration_mask=calibration_mask,
        query_ids=query_ids,
        head_weight=head_weight,
        head_bias=head_bias,
        head_query_ids=head_query_ids,
        config=config,
    )
    updated_behavior_representation = BehaviorMainRepresentation(
        rank=int(refreshed.selected_candidate["rank"]),
        random_state=int(behavior_main_representation.random_state),
        fit_scope="post_update_guard_plus_confirmed_harmful_probe_candidate",
    ).fit(
        updated_penultimate[reference_mask],
        scorer=refreshed.scorer,
    )
    updated_behavior = updated_behavior_representation.transform(
        updated_penultimate
    )
    fingerprint_payload = {
        "base_judge_fingerprint": str(base_judge_fingerprint),
        "candidate_model": candidate_model_metadata or adapter.to_metadata(),
        "selected_detector": refreshed.selected_candidate,
    }
    if candidate_model_artifact is None:
        candidate_model_artifact = adapter.save_checkpoint(
            output_dir / f"judge_head_adapter_v{version}.pt"
        )
    candidate_model_path = Path(candidate_model_artifact)
    if not candidate_model_path.exists():
        raise ValueError(f"Accepted candidate model artifact does not exist: {candidate_model_path}")
    fingerprint_digest = hashlib.sha256(
        json.dumps(fingerprint_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    )
    with candidate_model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            fingerprint_digest.update(chunk)
    updated_judge_fingerprint = fingerprint_digest.hexdigest()
    detector_artifact = refreshed.save_artifact(
        output_dir,
        judge_fingerprint=updated_judge_fingerprint,
        filename=f"judge_behavior_ood_scorer_v{version}.npz",
    )
    behavior_artifact = output_dir / f"behavior_main_representation_v{version}.npz"
    np.savez(
        behavior_artifact,
        **updated_behavior_representation.artifact_arrays(),
        metadata_json=np.asarray(
            json.dumps(updated_behavior_representation.to_metadata(), ensure_ascii=False)
        ),
    )
    reference_indices = np.flatnonzero(reference_mask).astype(int)
    calibration_indices = np.flatnonzero(calibration_mask).astype(int)
    two_sample_calibration = _post_update_b_mmd_calibration(
        reference_features=updated_behavior[reference_indices],
        calibration_features=updated_behavior[calibration_indices],
        reference_block_ids=np.asarray(permutation_block_ids)[reference_indices],
        calibration_block_ids=np.asarray(permutation_block_ids)[calibration_indices],
        config=window_drift_config,
    )
    previous_reference = _load_previous_b_reference(history, output_dir=output_dir)
    reference_features = updated_behavior[reference_indices].astype(np.float32)
    baseline_shift = _reference_baseline_shift(previous_reference, reference_features)
    recalibration_reasons = _reference_recalibration_reasons(
        cumulative_accepted_updates=cumulative_updates,
        baseline_shift=baseline_shift,
        update_interval=int(full_recalibration_update_interval),
        shift_sigma_threshold=float(full_recalibration_shift_sigma),
    )
    reference_artifact = output_dir / f"b_reference_v{version}.npz"
    metadata = {
        "artifact_type": "llm_judge_ood_b_reference",
        "version": int(version),
        "previous_version": int(history["latest_version"]) if history["latest_version"] else None,
        "cumulative_accepted_updates": int(cumulative_updates),
        "base_judge_fingerprint": str(base_judge_fingerprint),
        "judge_fingerprint": updated_judge_fingerprint,
        "selected_candidate_fixed_from_pre_update": dict(selected_candidate),
        "a_reference": "unchanged_frozen_backbone",
        "b_reference": "recomputed_with_accepted_head",
        "deployment_routing_scope": (
            "accepted_wide_linear_refit_for_all_records"
            if external_candidate
            else "accepted_head_for_all_records_of_adapted_queries"
        ),
        "reference_scope": "training_guard_plus_accepted_confirmed_harmful_probe",
        "calibration_scope": "source_training_calibration_only",
        "reference_calibration_exchangeable": False,
        "next_episode_monitoring_enabled": False,
        "next_episode_monitoring_status": (
            "disabled_pending_independent_post_update_reference_and_calibration"
        ),
        "reference_rows": int(reference_indices.size),
        "calibration_rows": int(calibration_indices.size),
        "thresholds": refreshed.thresholds.to_dict(),
        "two_sample_calibration": two_sample_calibration,
        "mmd_calibration": two_sample_calibration,
        "baseline_shift_from_previous": baseline_shift,
        "full_recalibration_policy": {
            "accepted_update_interval": int(full_recalibration_update_interval),
            "mean_shift_sigma_threshold": float(full_recalibration_shift_sigma),
        },
        "full_recalibration_required": True,
        "full_recalibration_performed": False,
        "full_recalibration_evidence": None,
        "full_recalibration_reasons": recalibration_reasons,
        "behavior_representation_artifact": str(behavior_artifact),
        "candidate_model_artifact": str(candidate_model_path),
    }
    np.savez(
        reference_artifact,
        reference_features=reference_features,
        calibration_features=updated_behavior[calibration_indices].astype(np.float32),
        reference_indices=reference_indices,
        calibration_indices=calibration_indices,
        reference_block_ids=np.asarray(permutation_block_ids)[reference_indices].astype(str),
        calibration_block_ids=np.asarray(permutation_block_ids)[calibration_indices].astype(str),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    history_entry = {
        "version": int(version),
        "judge_fingerprint": updated_judge_fingerprint,
        "reference_artifact": str(reference_artifact),
        "detector_artifact": str(detector_artifact),
        "candidate_model_artifact": str(candidate_model_path),
        "thresholds": refreshed.thresholds.to_dict(),
        "baseline_shift_from_previous": baseline_shift,
        "full_recalibration_required": bool(metadata["full_recalibration_required"]),
        "full_recalibration_performed": bool(metadata["full_recalibration_performed"]),
        "full_recalibration_reasons": recalibration_reasons,
    }
    updated_history = {
        "artifact_type": "llm_judge_ood_b_reference_history",
        "latest_version": int(version),
        "cumulative_accepted_updates": int(cumulative_updates),
        "full_recalibration_policy": metadata["full_recalibration_policy"],
        "versions": [*history["versions"], history_entry],
    }
    write_json(history_path, updated_history)
    return {
        "status": "saved_pending_independent_post_update_calibration",
        **metadata,
        "reference_artifact": str(reference_artifact),
        "detector_artifact": str(detector_artifact),
        "candidate_model_artifact": str(candidate_model_path),
        "behavior_representation_artifact": str(behavior_artifact),
        "history_artifact": str(history_path),
    }


def _adapter_affine_head_parameters(
    *,
    adapter: HeadAdapter,
    base_weight: np.ndarray,
    base_bias: np.ndarray,
    base_query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Overlay exact adapted torch coefficients on the deployed affine heads."""

    weights = np.asarray(base_weight, dtype=np.float64).copy()
    biases = np.asarray(base_bias, dtype=np.float64).copy()
    queries = np.asarray(base_query_ids).astype(str)
    mapping = {query: index for index, query in enumerate(queries.tolist())}
    for query_id, head in adapter.heads_.items():
        if query_id not in mapping:
            raise ValueError(f"Adapted head has unknown query_id={query_id!r}")
        position = mapping[query_id]
        state = head.state_dict()
        weight = np.asarray(state["weight"].detach().cpu(), dtype=np.float64).T
        bias = np.asarray(state["bias"].detach().cpu(), dtype=np.float64)
        if weight.shape != weights[position].shape or bias.shape != biases[position].shape:
            raise ValueError("Adapted affine head dimensions differ from the deployed Judge")
        weights[position] = weight
        biases[position] = bias
    return weights, biases, queries


def _fit_full_retraining_candidate(
    *,
    raw_features: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    source_train_mask: np.ndarray,
    source_validation_mask: np.ndarray,
    deployment_adapt_indices: np.ndarray,
    base_config: JudgeTrainingConfig,
    selection_config: JudgeSelectionConfig,
    output_dir: Path,
):
    """Refit preprocessing and a linear Judge for a wide covariate-drift episode."""

    adapt_indices = np.asarray(deployment_adapt_indices, dtype=int)
    if adapt_indices.size == 0:
        raise ValueError("Wide covariate refit requires confirmed deployment Adapt labels")
    train_mask = np.asarray(source_train_mask, dtype=bool).copy()
    validation_mask = np.asarray(source_validation_mask, dtype=bool)
    if train_mask.shape != validation_mask.shape or len(train_mask) != len(raw_features):
        raise ValueError("Wide covariate refit source masks must align with frozen features")
    if np.any(adapt_indices < 0) or np.any(adapt_indices >= len(train_mask)):
        raise ValueError("Wide covariate refit deployment Adapt indices are out of bounds")
    train_mask[adapt_indices] = True
    if np.any(train_mask & validation_mask):
        raise ValueError("Wide covariate refit fit and validation rows must remain disjoint")
    locked_selection = replace(
        selection_config,
        include_linear=True,
        include_neural_ablation=False,
        deployment_policy="linear_specification",
        force_neural=False,
    )
    result = select_source_judge(
        raw_features=raw_features,
        labels=labels,
        query_ids=query_ids,
        train_mask=train_mask,
        validation_mask=validation_mask,
        base_config=base_config,
        selection_config=locked_selection,
    )
    if not isinstance(result.model, PerQueryLinearJudge):
        raise RuntimeError("Wide covariate refit must produce a pure-linear Judge")
    output = ensure_dir(output_dir)
    artifacts = result.save_model_artifacts(output / "judge_checkpoints")
    _save_layer_preprocessor(result.preprocessor, output / "judge_preprocessor.npz")
    metadata = {
        "artifact_type": "llm_judge_ood_wide_covariate_linear_refit_candidate",
        "backbone": "frozen_Qwen_Qwen3_5_4B",
        "scope_boundary": (
            "wide_covariate_drift_only; not an automatic semantic-drift diagnosis "
            "or full-backbone retraining"
        ),
        "fit_scope": "source_training_train_plus_confirmed_harmful_probe",
        "fit_rows": int(train_mask.sum()),
        "source_rows": int(np.asarray(source_train_mask, dtype=bool).sum()),
        "confirmed_harmful_probe_rows": int(adapt_indices.size),
        "validation_scope": "source_training_validation_only",
        "validation_rows": int(validation_mask.sum()),
        "model": result.to_metadata(),
        "artifacts": artifacts,
    }
    write_json(output / "full_retraining_candidate.json", metadata)
    return result, artifacts


def _load_b_reference_history(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "latest_version": 1,
            "cumulative_accepted_updates": 0,
            "versions": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid B-reference history at {path}: {exc}") from exc
    versions = payload.get("versions")
    if not isinstance(versions, list):
        raise ValueError("B-reference history versions must be a list")
    return {
        "latest_version": int(payload.get("latest_version", 1)),
        "cumulative_accepted_updates": int(
            payload.get("cumulative_accepted_updates", len(versions))
        ),
        "versions": versions,
    }


def _load_previous_b_reference(
    history: dict[str, Any],
    *,
    output_dir: Path,
) -> np.ndarray | None:
    versions = list(history.get("versions", []))
    if not versions:
        legacy = output_dir / "b_reference_v2.npz"
        return _load_reference_features(legacy) if legacy.exists() else None
    path = Path(str(versions[-1].get("reference_artifact", "")))
    if not path.exists() and not path.is_absolute():
        path = output_dir / path.name
    return _load_reference_features(path) if path.exists() else None


def _load_reference_features(path: Path) -> np.ndarray:
    try:
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["reference_features"], dtype=np.float64)
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError(f"Could not load prior B reference {path}: {exc}") from exc
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"Prior B reference {path} is not a finite [N,D] matrix")
    return values


def _reference_baseline_shift(
    previous: np.ndarray | None,
    current: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(current, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError("Current B reference must be a finite [N,D] matrix")
    if previous is None:
        return {
            "available": False,
            "reason": "no_previous_reference_artifact",
            "standardized_mean_shift": None,
        }
    old = np.asarray(previous, dtype=np.float64)
    if old.shape[1] != values.shape[1]:
        return {
            "available": False,
            "reason": "reference_dimension_changed_requires_full_recalibration",
            "previous_dimension": int(old.shape[1]),
            "current_dimension": int(values.shape[1]),
            "standardized_mean_shift": None,
        }
    mean_distance = float(np.linalg.norm(values.mean(axis=0) - old.mean(axis=0)))
    reference_sigma = float(np.sqrt(np.mean(np.var(old, axis=0))))
    standardized = mean_distance / max(reference_sigma, 1e-12)
    return {
        "available": True,
        "mean_l2_distance": mean_distance,
        "previous_reference_rms_sigma": reference_sigma,
        "standardized_mean_shift": float(standardized),
        "previous_rows": int(len(old)),
        "current_rows": int(len(values)),
    }


def _reference_recalibration_reasons(
    *,
    cumulative_accepted_updates: int,
    baseline_shift: dict[str, Any],
    update_interval: int,
    shift_sigma_threshold: float,
) -> list[str]:
    if int(cumulative_accepted_updates) < 1 or int(update_interval) < 1:
        raise ValueError("reference update counts and intervals must be positive")
    if float(shift_sigma_threshold) <= 0.0:
        raise ValueError("reference shift sigma threshold must be positive")
    reasons: list[str] = []
    if int(cumulative_accepted_updates) % int(update_interval) == 0:
        reasons.append("accepted_update_interval_reached")
    dimension_changed = (
        baseline_shift.get("reason")
        == "reference_dimension_changed_requires_full_recalibration"
    )
    if dimension_changed or (
        bool(baseline_shift.get("available"))
        and float(baseline_shift["standardized_mean_shift"])
        > float(shift_sigma_threshold)
    ):
        reasons.append("b_reference_mean_shift_exceeded_sigma")
    return reasons


def _post_update_b_mmd_calibration(
    *,
    reference_features: np.ndarray,
    calibration_features: np.ndarray,
    reference_block_ids: np.ndarray,
    calibration_block_ids: np.ndarray,
    config: WindowDriftConfig,
) -> dict[str, Any]:
    """Audit nominal B-space thresholds after an accepted update."""

    reference_blocks = np.asarray(reference_block_ids).astype(str)
    calibration_blocks = np.asarray(calibration_block_ids).astype(str)
    unique_calibration_blocks = list(dict.fromkeys(calibration_blocks.tolist()))
    if len(set(reference_blocks.tolist())) < 2 or len(unique_calibration_blocks) < 2:
        return {
            "status": "insufficient_blocks",
            "reference_blocks": int(len(set(reference_blocks.tolist()))),
            "calibration_blocks": int(len(unique_calibration_blocks)),
            "formal_calibration_valid": False,
            "monitoring_enabled": False,
        }
    mmd_test = MMDPermutationTest(config).fit(
        reference_features,
        block_ids=reference_blocks,
    )
    c2st_test = (
        BlockAwareC2ST(config).fit(
            reference_features,
            block_ids=reference_blocks,
        )
        if config.c2st_enabled
        else None
    )
    window_size = min(int(config.window_size), len(unique_calibration_blocks))
    windows: list[dict[str, Any]] = []
    for window_index, start in enumerate(range(0, len(unique_calibration_blocks), window_size)):
        block_window = unique_calibration_blocks[start : start + window_size]
        if len(block_window) < 2:
            continue
        mask = np.isin(calibration_blocks, np.asarray(block_window, dtype=str))
        transformed = np.asarray(calibration_features)[mask]
        mmd_result = mmd_test.test(
            transformed,
            block_ids=calibration_blocks[mask],
            seed=int(config.seed) + 700 + window_index,
        )
        c2st_result = (
            c2st_test.test(
                transformed,
                block_ids=calibration_blocks[mask],
                seed=int(config.seed) + 900 + window_index,
            )
            if c2st_test is not None
            else None
        )
        windows.append(
            {
                "mmd": mmd_result,
                "c2st": c2st_result if c2st_result is not None else {"enabled": False},
            }
        )
    by_test: dict[str, dict[str, Any]] = {}
    for test_name in ("mmd", "c2st"):
        p_values = np.asarray(
            [
                float(row[test_name]["p_value"])
                for row in windows
                if row[test_name].get("p_value") is not None
            ],
            dtype=float,
        )
        soft_rejections = int(np.sum(p_values <= float(config.soft_alpha)))
        hard_rejections = int(np.sum(p_values <= float(config.hard_alpha)))
        by_test[test_name] = {
            "enabled": bool(test_name == "mmd" or config.c2st_enabled),
            "soft_false_alert_rate": (
                float(soft_rejections / p_values.size) if p_values.size else None
            ),
            "hard_false_alert_rate": (
                float(hard_rejections / p_values.size) if p_values.size else None
            ),
            "soft_false_alert_rate_ci95": (
                wilson_interval(soft_rejections, int(p_values.size)) if p_values.size else None
            ),
            "hard_false_alert_rate_ci95": (
                wilson_interval(hard_rejections, int(p_values.size)) if p_values.size else None
            ),
        }
    primary_name = str(config.primary_test).lower()
    primary_p_values = np.asarray(
        [
            float(row[primary_name]["p_value"])
            for row in windows
            if isinstance(row.get(primary_name), dict)
            and row[primary_name].get("p_value") is not None
            and np.isfinite(float(row[primary_name]["p_value"]))
        ],
        dtype=float,
    )
    valid_window_count = primary_p_values.size >= int(config.minimum_valid_calibration_windows)
    fallback_allowed = bool(config.allow_nominal_fallback_for_smoke)
    if valid_window_count:
        soft_alpha = float(config.soft_alpha)
        hard_alpha = float(config.hard_alpha)
        status = "diagnostic_only_nonexchangeable_post_update_calibration"
    else:
        soft_alpha = float(config.soft_alpha)
        hard_alpha = float(config.hard_alpha)
        status = "diagnostic_only_insufficient_nonexchangeable_post_update_calibration"
    return {
        "status": status,
        "scope": "mixed_reference_guard_plus_adapt_vs_legacy_source_calibration",
        "primary_test": primary_name,
        "soft_alpha": float(soft_alpha),
        "hard_alpha": float(hard_alpha),
        "formal_calibration_valid": False,
        "diagnostic_window_count_valid": bool(valid_window_count),
        "reference_calibration_exchangeable": False,
        "nominal_fallback_for_smoke": bool(not valid_window_count and fallback_allowed),
        "monitoring_enabled": False,
        "minimum_valid_calibration_windows": int(config.minimum_valid_calibration_windows),
        "valid_calibration_window_count": int(primary_p_values.size),
        "threshold_rule": "diagnostic_nominal_thresholds_not_deployable",
        "window_count": int(len(windows)),
        "by_test": by_test,
        "windows": windows,
    }


def _expand_document_indices_to_records(
    *,
    document_indices: np.ndarray,
    document_ids: np.ndarray,
    pool_mask: np.ndarray,
) -> np.ndarray:
    """Expand routed document anchors to all Judge records in a disjoint label pool."""

    anchors = np.asarray(document_indices, dtype=int)
    ids = np.asarray(document_ids).astype(str)
    mask = np.asarray(pool_mask, dtype=bool)
    if mask.shape != (len(ids),):
        raise ValueError("pool_mask must align with document_ids")
    if anchors.size == 0:
        return np.zeros(0, dtype=int)
    if np.any(anchors < 0) or np.any(anchors >= len(ids)):
        raise ValueError("document routing indices are out of bounds")
    selected_ids = np.asarray(sorted(set(ids[anchors].tolist())), dtype=str)
    return np.flatnonzero(mask & np.isin(ids, selected_ids)).astype(int)


def _behavior_warnings_by_document_cluster(
    *,
    prototypes: dict[str, dict[str, Any]],
    document_ids: np.ndarray,
    predicted_document_cluster_ids: np.ndarray,
    record_mask: np.ndarray,
    probabilities: np.ndarray,
    logits: np.ndarray,
    ood_scores: np.ndarray,
    ensemble_predictions: np.ndarray | None = None,
    calibrator: BehaviorWarningCalibrator,
) -> dict[str, dict[str, Any]]:
    ids = np.asarray(document_ids).astype(str)
    clusters = np.asarray(predicted_document_cluster_ids).astype(str)
    mask = np.asarray(record_mask, dtype=bool)
    if len(ids) != len(clusters) or mask.shape != (len(ids),):
        raise ValueError("document IDs and predicted document clusters must align")
    result: dict[str, dict[str, Any]] = {}
    for document_cluster_id in sorted(prototypes):
        record_indices = np.flatnonzero(mask & (clusters == str(document_cluster_id))).astype(int)
        if record_indices.size == 0:
            result[str(document_cluster_id)] = {
                "triggered": False,
                "reason": "insufficient_cluster_records",
                "record_count": 0,
                "ranked_record_indices": [],
                "predicted_document_cluster_id": str(document_cluster_id),
                "cluster_document_count": 0,
            }
            continue
        warning = calibrator.evaluate(
            probabilities=np.asarray(probabilities)[record_indices],
            logits=np.asarray(logits)[record_indices],
            ood_scores=np.asarray(ood_scores)[record_indices],
            record_indices=record_indices,
            agreement_predictions=(
                np.asarray(ensemble_predictions)[record_indices]
                if ensemble_predictions is not None
                else None
            ),
        )
        result[str(document_cluster_id)] = {
            **warning,
            "predicted_document_cluster_id": str(document_cluster_id),
            "cluster_document_count": int(len(set(ids[record_indices].tolist()))),
        }
    return result


def _unique_document_count(document_ids: np.ndarray, mask: np.ndarray) -> int:
    ids = np.asarray(document_ids).astype(str)
    selected = np.asarray(mask, dtype=bool)
    return int(len(set(ids[selected].tolist())))


def _audit_document_group_harmfulness(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    query_ids: np.ndarray,
    audit_document_group_ids: np.ndarray,
    input_document_ids: np.ndarray,
    rater_scores: np.ndarray,
    evaluation_mask: np.ndarray,
    reference_metric: float,
    reference_human_ceiling: dict[str, Any],
    metric_name: str,
    tolerance: float,
    class_values: np.ndarray | None,
    require_human_ceiling: bool,
    n_boot: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Evaluation-only document-group truth from fully labeled deployment data.

    This function is deliberately called only after the model, detector, and
    monitoring decisions have been fixed. It is not the deployable Probe,
    which samples only observed contributors to a persistent stream cluster.
    """

    audit_groups = np.asarray(audit_document_group_ids).astype(str)
    mask = np.asarray(evaluation_mask, dtype=bool)
    output: dict[str, dict[str, Any]] = {}
    for offset, document_group_id in enumerate(sorted(set(audit_groups[mask].tolist()))):
        indices = np.flatnonzero(mask & (audit_groups == document_group_id))
        if not indices.size:
            output[document_group_id] = {"status": "uncertain", "n_evaluation": 0}
            continue
        target_human_ceiling = estimate_human_ceiling(
            rater_scores[indices],
            metric_name=metric_name,
            query_ids=query_ids[indices],
            class_values=class_values,
        )
        result = harmfulness_probe(
            y_true=labels[indices],
            y_pred=predictions[indices],
            reference_metric=float(reference_metric),
            tolerance=float(tolerance),
            metric_name=metric_name,
            query_ids=query_ids[indices],
            probabilities=probabilities[indices],
            class_values=class_values,
            groups=input_document_ids[indices],
            reference_human_ceiling=reference_human_ceiling.get("value"),
            target_human_ceiling=target_human_ceiling.get("value"),
            require_human_ceiling=bool(require_human_ceiling),
            n_boot=int(n_boot),
            seed=int(seed) + int(offset),
        )
        output[document_group_id] = {
            "status": str(result.get("status", "uncertain")),
            "n_evaluation": int(indices.size),
            "result": result,
            "human_ceiling_target": target_human_ceiling,
        }
    return output


def _lifecycle_summary(lifecycle_rows: list[dict[str, Any]]) -> dict[str, Any]:
    confirmations: dict[str, dict[str, Any]] = {}
    for row in lifecycle_rows:
        document_cluster_id = str(row.get("document_cluster_id", ""))
        confirmation_window = row.get("confirmation_window")
        if not document_cluster_id or confirmation_window is None:
            continue
        current = confirmations.get(document_cluster_id)
        if current is None or int(confirmation_window) < int(current["confirmation_window"]):
            confirmations[document_cluster_id] = {
                "confirmation_window": int(confirmation_window),
                "confirmation_latency_windows": int(row["confirmation_latency_windows"]),
                "confirmation_latency_samples": int(row["confirmation_latency_samples"]),
            }
    latency_windows = np.asarray(
        [item["confirmation_latency_windows"] for item in confirmations.values()], dtype=float
    )
    latency_samples = np.asarray(
        [item["confirmation_latency_samples"] for item in confirmations.values()], dtype=float
    )
    return {
        "confirmed_document_clusters": dict(sorted(confirmations.items())),
        "num_confirmed_document_clusters": int(len(confirmations)),
        "confirmation_latency_windows": {
            "mean": float(latency_windows.mean()) if latency_windows.size else None,
            "min": float(latency_windows.min()) if latency_windows.size else None,
            "max": float(latency_windows.max()) if latency_windows.size else None,
        },
        "confirmation_latency_samples": {
            "mean": float(latency_samples.mean()) if latency_samples.size else None,
            "min": float(latency_samples.min()) if latency_samples.size else None,
            "max": float(latency_samples.max()) if latency_samples.size else None,
        },
    }


def _lifecycle_monitoring_events(
    *,
    lifecycle_rows: list[dict[str, Any]],
    stream_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn deployed lifecycle records into stream-relative alarm events.

    A document cluster can contain members from earlier overlapping context,
    but it becomes actionable only at the end of its *confirmation* window.
    The returned persistence event therefore carries the confirmation position,
    not the position of its earliest historical member.
    """

    stream = np.asarray(stream_indices, dtype=int)
    stream_members = set(stream.tolist())
    clustering: list[dict[str, Any]] = []
    persistence: list[dict[str, Any]] = []
    for row in lifecycle_rows:
        members = sorted(
            {
                int(index)
                for index in row.get("member_indices", [])
                if int(index) in stream_members
            }
        )
        if not members:
            continue
        window_stop = int(row.get("window_stop", 0))
        position = min(max(window_stop - 1, 0), max(len(stream) - 1, 0))
        event = {
            "position": position,
            "members": members,
            "window_index": int(row.get("window_index", 0)),
            "cluster_id": int(row.get("cluster_id", -1)),
            "predicted_document_cluster_id": str(row.get("document_cluster_id", "")),
        }
        clustering.append(event)
        if row.get("confirmation_window") == row.get("window_index"):
            persistence.append(event)
    return clustering, persistence


def _load_or_extract_features(
    config: SampleOODConfig,
    records: list[JudgeRecord],
    training_train_mask: np.ndarray,
    *,
    feature_scope: str = "input_document",
) -> tuple[np.ndarray, dict[str, Any]]:
    if feature_scope not in {"input_document", "judge_input"}:
        raise ValueError("feature_scope must be 'input_document' or 'judge_input'")
    hidden_feature_path = (
        config.document_hidden_feature_path
        if feature_scope == "input_document"
        else config.judge_hidden_feature_path
    )
    if hidden_feature_path:
        store = load_hidden_feature_store(hidden_feature_path)
        cache_metadata = store.metadata.get("cache_metadata", {})
        _validate_frozen_hidden_cache_contract(
            cache_metadata=cache_metadata,
            feature_scope=feature_scope,
            expected_model_id=str(config.backbone_model_id),
            expected_pooling=str(config.hidden_pooling),
            expected_max_length=int(config.hidden_max_length),
            expected_prompt_template_version=(
                config.judge_prompt_template_version if feature_scope == "judge_input" else None
            ),
            expected_prompt_template_sha256=(
                config.judge_prompt_template_sha256 if feature_scope == "judge_input" else None
            ),
        )
        cache_scope = cache_metadata.get("feature_scope") if isinstance(cache_metadata, dict) else None
        if cache_scope != feature_scope:
            raise ValueError(
                f"Hidden cache has feature_scope={cache_scope!r}, expected {feature_scope!r}"
            )
        cache_fingerprint = cache_metadata.get("dataset_fingerprint") if isinstance(cache_metadata, dict) else None
        expected_fingerprint = record_fingerprint(records, feature_scope=feature_scope)
        if cache_fingerprint is None:
            raise ValueError(
                "Frozen Qwen feature cache is missing its dataset_fingerprint; "
                "re-extract the cache instead of accepting an unbound artifact"
            )
        if cache_fingerprint is not None and str(cache_fingerprint) != expected_fingerprint:
            raise ValueError(
                "Hidden feature cache fingerprint does not match the current records; "
                "re-extract features instead of reusing a stale or superset cache"
            )
        aligned = (
            _align_document_hidden_features(store=store, records=records)
            if feature_scope == "input_document"
            else _align_judge_hidden_features(store=store, records=records)
        )
        return aligned, {
            "extractor": "hidden_feature_store",
            "feature_scope": feature_scope,
            **store.metadata,
            "record_fingerprint_validation": "exact_match" if cache_fingerprint is not None else "metadata_not_available",
            "expected_record_fingerprint": expected_fingerprint,
        }
    raise ValueError(
        f"Missing required frozen Qwen hidden cache for feature_scope={feature_scope!r}. "
        "The TF-IDF/SVD fallback has been removed. Extract the cache with "
        f"scripts/llm_judge_ood/20_prepare_llm_judge_ood_hidden.py --model-path {config.backbone_model_id} "
        f"--feature-scope {feature_scope} --pooling masked_mean "
        f"--max-length {int(config.hidden_max_length)}."
    )


def _validate_frozen_hidden_cache_contract(
    *,
    cache_metadata: Any,
    feature_scope: str,
    expected_model_id: str,
    expected_pooling: str,
    expected_max_length: int,
    expected_prompt_template_version: str | None,
    expected_prompt_template_sha256: str | None,
) -> None:
    if not isinstance(cache_metadata, dict):
        raise ValueError("Frozen hidden cache is missing structured metadata")
    required = {
        "artifact_type": "llm_judge_ood_frozen_qwen_hidden_features",
        "feature_scope": str(feature_scope),
        "model_id": str(expected_model_id),
        "model_revision": QWEN3_5_4B_REVISION,
        "model_type": "qwen3_5_text",
        "num_model_layers": QWEN3_5_4B_NUM_LAYERS,
        "model_hidden_size": QWEN3_5_4B_HIDDEN_SIZE,
        "hidden_state_count": QWEN3_5_4B_NUM_LAYERS + 1,
        "embedding_state_included": True,
        "model_revision_requested": QWEN3_5_4B_REVISION,
        "model_eval": True,
        "requires_grad": False,
        "backbone_frozen": True,
        "max_length": int(expected_max_length),
        "pooling": str(expected_pooling),
        "pooling_scope": str(feature_scope),
        "pooling_formula": "sum(hidden_state * attention_mask) / sum(attention_mask)",
        "pooling_mask_source": "tokenizer_attention_mask",
        "pooling_excludes_padding": True,
        "prompt_template_version": (
            str(expected_prompt_template_version)
            if expected_prompt_template_version is not None
            else _feature_prompt_template_version(feature_scope)
        ),
        "labels_in_prompt": False,
    }
    if expected_prompt_template_sha256 is not None:
        required["prompt_template_sha256"] = str(expected_prompt_template_sha256)
    mismatches = {
        key: {"expected": value, "actual": cache_metadata.get(key)}
        for key, value in required.items()
        if cache_metadata.get(key) != value
    }
    identity = cache_metadata.get("model_identity_evidence")
    if (
        not isinstance(identity, dict)
        or identity.get("kind") not in {
            "huggingface_repo_id",
            "verified_local_huggingface_readme",
            "verified_local_git_snapshot",
        }
        or identity.get("repo_id") != QWEN3_5_4B_MODEL_ID
    ):
        mismatches["model_identity_evidence"] = {
            "expected": {
                "kind": (
                    "huggingface_repo_id, verified_local_huggingface_readme, "
                    "or verified_local_git_snapshot"
                ),
                "repo_id": QWEN3_5_4B_MODEL_ID,
                "revision": QWEN3_5_4B_REVISION,
            },
            "actual": identity,
        }
    elif identity.get("revision") != QWEN3_5_4B_REVISION:
        mismatches["model_identity_evidence.revision"] = {
            "expected": QWEN3_5_4B_REVISION,
            "actual": identity.get("revision"),
        }
    if mismatches:
        raise ValueError(f"Frozen hidden cache violates the final Qwen feature contract: {mismatches}")


def _unique_input_document_records(records: list[JudgeRecord]) -> list[JudgeRecord]:
    unique: dict[str, JudgeRecord] = {}
    for record in records:
        existing = unique.get(record.input_document_id)
        if existing is not None and existing.input_document_text != record.input_document_text:
            raise ValueError(f"Input document id {record.input_document_id!r} has inconsistent text")
        unique.setdefault(record.input_document_id, record)
    return list(unique.values())


def _align_document_hidden_features(*, store: Any, records: list[JudgeRecord]) -> np.ndarray:
    if store.input_document_ids is None:
        raise ValueError("Document OOD hidden cache is missing input_document_ids")
    cached_document_ids = np.asarray(store.input_document_ids).astype(str)
    if len(cached_document_ids) != len(store.features):
        raise ValueError("Document OOD hidden cache input_document_ids do not align with features")
    row_by_document_id: dict[str, int] = {}
    for index, document_id in enumerate(cached_document_ids.tolist()):
        existing = row_by_document_id.setdefault(document_id, index)
        if existing != index and not np.allclose(store.features[existing], store.features[index], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Document OOD hidden cache has inconsistent features for {document_id!r}")
    missing = [record.input_document_id for record in records if record.input_document_id not in row_by_document_id]
    if missing:
        raise ValueError(f"Document OOD hidden cache is missing {len(set(missing))} input documents, first={missing[:5]}")
    return np.stack(
        [store.features[row_by_document_id[record.input_document_id]] for record in records], axis=0
    ).astype(np.float32)


def _align_judge_hidden_features(*, store: Any, records: list[JudgeRecord]) -> np.ndarray:
    cached_sample_ids = np.asarray(store.sample_ids).astype(str)
    if len(cached_sample_ids) != len(store.features):
        raise ValueError("Judge-input hidden cache sample_ids do not align with features")
    row_by_sample_id: dict[str, int] = {}
    for index, sample_id in enumerate(cached_sample_ids.tolist()):
        if sample_id in row_by_sample_id:
            raise ValueError(f"Judge-input hidden cache has duplicate sample_id {sample_id!r}")
        row_by_sample_id[sample_id] = int(index)
    missing = [record.sample_id for record in records if record.sample_id not in row_by_sample_id]
    if missing:
        raise ValueError(
            f"Judge-input hidden cache is missing {len(set(missing))} Judge rows, first={missing[:5]}"
        )
    if store.query_ids is not None:
        cached_queries = np.asarray(store.query_ids).astype(str)
        for record in records:
            cached = cached_queries[row_by_sample_id[record.sample_id]]
            if cached != str(record.query_id):
                raise ValueError(
                    f"Judge-input cache query ID disagrees for sample {record.sample_id!r}"
                )
    if store.input_document_ids is not None:
        cached_documents = np.asarray(store.input_document_ids).astype(str)
        for record in records:
            cached = cached_documents[row_by_sample_id[record.sample_id]]
            if cached != str(record.input_document_id):
                raise ValueError(
                    f"Judge-input cache document ID disagrees for sample {record.sample_id!r}"
                )
    return np.stack(
        [store.features[row_by_sample_id[record.sample_id]] for record in records], axis=0
    ).astype(np.float32)


def _feature_prompt_template_version(feature_scope: str) -> str:
    if feature_scope == "input_document":
        return "raw_input_document_v1"
    if feature_scope == "judge_input":
        return "judge_input_query_document_candidate_v1"
    raise ValueError("feature_scope must be 'input_document' or 'judge_input'")


def _persistent_document_indices(
    lifecycle_rows: list[dict[str, Any]],
    document_indices: np.ndarray,
    document_cluster_labels: np.ndarray,
) -> np.ndarray:
    persistent_document_cluster_ids = {
        str(row["document_cluster_id"])
        for row in lifecycle_rows
        if row.get("status") == "persistent_document_cluster"
    }
    if not persistent_document_cluster_ids:
        return np.zeros(0, dtype=int)
    persistent_members: set[int] = set()
    for row in lifecycle_rows:
        if str(row.get("document_cluster_id")) in persistent_document_cluster_ids:
            persistent_members.update(int(idx) for idx in row.get("member_indices", []))
    if not persistent_members:
        return np.zeros(0, dtype=int)
    return np.asarray(
        [
            idx
            for idx, predicted_document_cluster_id in zip(document_indices.tolist(), document_cluster_labels.tolist())
            if int(idx) in persistent_members and str(predicted_document_cluster_id) in persistent_document_cluster_ids
        ],
        dtype=int,
    )


def _build_persistent_document_cluster_prototypes(
    lifecycle_rows: list[dict[str, Any]],
    embeddings: np.ndarray,
    *,
    routing_distance_quantile: float,
) -> dict[str, dict[str, Any]]:
    persistent_document_cluster_ids = {
        str(row["document_cluster_id"])
        for row in lifecycle_rows
        if row.get("status") == "persistent_document_cluster"
    }
    members_by_document_cluster: dict[str, set[int]] = {
        document_cluster_id: set() for document_cluster_id in persistent_document_cluster_ids
    }
    for row in lifecycle_rows:
        document_cluster_id = str(row.get("document_cluster_id"))
        if document_cluster_id in members_by_document_cluster:
            members_by_document_cluster[document_cluster_id].update(
                int(index) for index in row.get("member_indices", [])
            )
    values = np.asarray(embeddings, dtype=np.float32)
    prototypes: dict[str, dict[str, Any]] = {}
    for document_cluster_id in sorted(members_by_document_cluster):
        member_indices = np.asarray(sorted(members_by_document_cluster[document_cluster_id]), dtype=int)
        if member_indices.size == 0:
            continue
        member_values = values[member_indices]
        centroid = member_values.mean(axis=0)
        distances = np.linalg.norm(member_values - centroid, axis=1)
        observed_radius = float(np.max(distances)) if distances.size else 0.0
        routing_radius = float(
            np.quantile(
                distances,
                float(routing_distance_quantile),
                method="linear",
            )
        ) if distances.size else 0.0
        prototypes[document_cluster_id] = {
            "centroid": centroid.astype(np.float32),
            "observed_radius": observed_radius,
            "routing_distance_quantile": float(routing_distance_quantile),
            "routing_radius": routing_radius,
            "routing_radius_policy": "frozen_distance_quantile",
            "member_indices": member_indices,
        }
    return prototypes


def _route_to_persistent_document_clusters(
    *,
    allowed_indices: np.ndarray,
    embeddings: np.ndarray,
    score_labels: np.ndarray,
    prototypes: dict[str, dict[str, Any]],
    requires_ood: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    allowed_indices = np.asarray(allowed_indices, dtype=int)
    values = np.asarray(embeddings, dtype=np.float32)
    statuses = np.asarray(score_labels).astype(str)
    if len(values) != len(statuses):
        raise ValueError("embeddings and score_labels must be aligned")
    eligible_statuses = {"soft_ood", "hard_ood"}
    assigned_indices: list[int] = []
    assigned_document_clusters: list[str] = []
    rejected_non_ood = 0
    rejected_outside_radius = 0
    accepted_distances: list[float] = []
    for index in allowed_indices.tolist():
        if requires_ood and statuses[int(index)] not in eligible_statuses:
            rejected_non_ood += 1
            continue
        if not prototypes:
            rejected_outside_radius += 1
            continue
        distances = {
            document_cluster_id: float(np.linalg.norm(values[int(index)] - np.asarray(prototype["centroid"])))
            for document_cluster_id, prototype in prototypes.items()
        }
        nearest_document_cluster = min(distances, key=distances.get)
        nearest_distance = distances[nearest_document_cluster]
        if nearest_distance > float(prototypes[nearest_document_cluster]["routing_radius"]):
            rejected_outside_radius += 1
            continue
        assigned_indices.append(int(index))
        assigned_document_clusters.append(str(nearest_document_cluster))
        accepted_distances.append(float(nearest_distance))
    counts = {
        document_cluster_id: int(sum(value == document_cluster_id for value in assigned_document_clusters))
        for document_cluster_id in sorted(set(assigned_document_clusters))
    }
    report = {
        "pool_rows": int(allowed_indices.size),
        "ood_document_rows": int(allowed_indices.size - rejected_non_ood),
        "assigned_rows": int(len(assigned_indices)),
        "rejected_non_ood_rows": int(rejected_non_ood),
        "rejected_outside_radius_rows": int(rejected_outside_radius),
        "assigned_by_predicted_document_cluster": counts,
        "accepted_distance_mean": float(np.mean(accepted_distances)) if accepted_distances else None,
        "accepted_distance_max": float(np.max(accepted_distances)) if accepted_distances else None,
    }
    return (
        np.asarray(assigned_indices, dtype=int),
        np.asarray(assigned_document_clusters, dtype=str),
        report,
    )


def _persistent_prototype_metadata(prototypes: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        document_cluster_id: {
            "member_count": int(len(prototype["member_indices"])),
            "observed_radius": float(prototype["observed_radius"]),
            "routing_distance_quantile": float(prototype["routing_distance_quantile"]),
            "routing_radius_policy": str(prototype["routing_radius_policy"]),
            "routing_radius": float(prototype["routing_radius"]),
        }
        for document_cluster_id, prototype in sorted(prototypes.items())
    }


def _stratified_source_replay_sample(
    *,
    source_indices: np.ndarray,
    query_ids: np.ndarray,
    labels: np.ndarray,
    budget: int,
    seed: int,
) -> np.ndarray:
    source_indices = np.asarray(source_indices, dtype=int)
    queries = np.asarray(query_ids).astype(str)
    values = normalize_label_array(labels)
    strata = np.asarray(
        [f"{query_id}\u241f{label}" for query_id, label in zip(queries.tolist(), values.tolist())],
        dtype=str,
    )
    return stratified_random_sample(source_indices, strata, budget=int(budget), seed=int(seed))


def _source_replay_metadata(
    indices: np.ndarray,
    *,
    query_ids: np.ndarray,
    labels: np.ndarray,
    adapt_rows: int | None = None,
    configured_budget: int | None = None,
    selected_rows: int | None = None,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=int)
    queries = np.asarray(query_ids).astype(str)
    values = normalize_label_array(labels)
    counts: dict[str, int] = {}
    for index in selected.tolist():
        key = f"{queries[int(index)]}|{values[int(index)]}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "rows": int(selected.size),
        "selected_rows_before_target_holdout": (
            int(selected_rows) if selected_rows is not None else int(selected.size)
        ),
        "sampling": "stratified_by_query_id_and_label",
        "ratio_contract": "at_most_one_source_replay_row_per_confirmed_harmful_probe_label",
        "adapt_rows": int(adapt_rows) if adapt_rows is not None else None,
        "configured_budget_cap": (
            int(configured_budget) if configured_budget is not None else None
        ),
        "strata_counts": dict(sorted(counts.items())),
    }


def _adaptation_skip_reason(has_persistent_candidate: bool, probe: dict[str, Any]) -> str:
    if not has_persistent_candidate:
        return "no_persistent_document_cluster"
    status = str(probe.get("status", "unknown"))
    if status == "harmful":
        return "harmful_probe"
    if status == "no_probe_candidates":
        return "no_probe_candidates"
    return f"probe_status_{status}"


def _sample_probe_indices(
    *,
    probe_allowed: np.ndarray,
    predicted_document_cluster_ids: np.ndarray,
    document_ids: np.ndarray,
    query_ids: np.ndarray | None = None,
    budget: int,
    seed: int,
    budget_per_cluster: bool = False,
) -> np.ndarray:
    """Budget unique documents, then collect at most one row per document/query."""

    pool = np.asarray(probe_allowed, dtype=int)
    clusters = np.asarray(predicted_document_cluster_ids).astype(str)
    ids = np.asarray(document_ids).astype(str)
    if pool.size == 0 or int(budget) <= 0:
        return np.zeros(0, dtype=int)
    if len(clusters) != len(ids):
        raise ValueError("predicted document clusters and document IDs must align")
    unique_documents = _random_record_per_document(pool, ids, seed=int(seed))
    if budget_per_cluster:
        selected_documents = _stratified_random_sample_per_cluster(
            unique_documents,
            clusters,
            budget_per_cluster=int(budget),
            seed=int(seed) + 1,
        )
    else:
        selected_documents = stratified_random_sample(
            unique_documents,
            clusters,
            budget=int(budget),
            seed=int(seed) + 1,
        ).astype(int)
    if query_ids is None:
        return selected_documents.astype(int)
    queries = np.asarray(query_ids).astype(str)
    if len(queries) != len(ids):
        raise ValueError("query IDs and document IDs must align")
    selected_document_ids = ids[selected_documents]
    selected_pool = pool[np.isin(ids[pool], selected_document_ids)]
    return _random_record_per_document_query(
        selected_pool,
        ids,
        queries,
        seed=int(seed) + 2,
    ).astype(int)


def _stratified_random_sample_per_cluster(
    indices: np.ndarray,
    clusters: np.ndarray,
    *,
    budget_per_cluster: int,
    seed: int,
) -> np.ndarray:
    pool = np.asarray(indices, dtype=int)
    values = np.asarray(clusters).astype(str)
    selected: list[int] = []
    for position, cluster_id in enumerate(sorted(set(values[pool].tolist()))):
        local = pool[values[pool] == cluster_id]
        selected.extend(
            stratified_random_sample(
                local,
                values,
                budget=int(budget_per_cluster),
                seed=int(seed) + position,
            ).astype(int).tolist()
        )
    return np.asarray(selected, dtype=int)


def _sample_safety_net_indices(
    *,
    stream_indices: np.ndarray,
    document_ids: np.ndarray,
    query_ids: np.ndarray | None = None,
    budget: int,
    seed: int,
) -> np.ndarray:
    """Randomly sample stream documents without consulting any drift signal."""

    pool = np.asarray(stream_indices, dtype=int)
    ids = np.asarray(document_ids).astype(str)
    if int(budget) <= 0 or pool.size == 0:
        return np.zeros(0, dtype=int)
    clusters = np.full(len(ids), "deployment_stream", dtype=str)
    return _sample_probe_indices(
        probe_allowed=pool,
        predicted_document_cluster_ids=clusters,
        document_ids=ids,
        query_ids=query_ids,
        budget=int(budget),
        seed=int(seed),
    ).astype(int)


def _sample_safety_net_by_rate(
    *,
    ordered_document_indices: np.ndarray,
    stream_mask: np.ndarray,
    document_ids: np.ndarray,
    query_ids: np.ndarray,
    period_documents: int,
    labels_per_period: int,
    initial_documents: int = 0,
    initial_labels: int = 0,
    window_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Spend a cumulative document budget, optionally front-loaded, independent of alarms."""

    ordered = np.asarray(ordered_document_indices, dtype=int)
    mask = np.asarray(stream_mask, dtype=bool)
    ids = np.asarray(document_ids).astype(str)
    if int(window_size) < 1 or int(period_documents) < 1:
        raise ValueError("Safety-net window and period sizes must be positive")
    if not 0 <= int(labels_per_period) <= int(period_documents):
        raise ValueError("Safety-net labels per period must be in [0, period_documents]")
    if not 0 <= int(initial_labels) <= int(initial_documents):
        raise ValueError("Safety-net initial labels must be in [0, initial_documents]")
    if mask.shape != (len(ids),):
        raise ValueError("Safety-net stream mask must align with document IDs")
    selected: list[int] = []
    windows: list[dict[str, Any]] = []
    for window_index, start in enumerate(range(0, len(ordered), int(window_size))):
        document_window = ordered[start : start + int(window_size)]
        window_document_ids = ids[document_window]
        pool = np.flatnonzero(mask & np.isin(ids, window_document_ids))
        observed_documents = min(start + len(document_window), len(ordered))
        rate_entitlement = int(
            math.floor(
                observed_documents
                * int(labels_per_period)
                / int(period_documents)
            )
        )
        initial_entitlement = (
            int(
                math.floor(
                    min(observed_documents, int(initial_documents))
                    * int(initial_labels)
                    / int(initial_documents)
                )
            )
            if int(initial_documents) > 0
            else 0
        )
        cumulative_entitlement = max(rate_entitlement, initial_entitlement)
        selected_document_count = len(set(ids[np.asarray(selected, dtype=int)].tolist()))
        local_budget = max(0, cumulative_entitlement - selected_document_count)
        local = _sample_safety_net_indices(
            stream_indices=pool,
            document_ids=ids,
            query_ids=query_ids,
            budget=int(local_budget),
            seed=int(seed) + window_index,
        )
        selected.extend(local.astype(int).tolist())
        windows.append(
            {
                "window_index": int(window_index),
                "document_count": int(len(document_window)),
                "document_indices": document_window.astype(int).tolist(),
                "requested_labels": int(local_budget),
                "sampled_documents": int(len(set(ids[local].tolist()))),
                "query_ratings_collected": int(len(local)),
                "sampled_record_indices": local.astype(int).tolist(),
            }
        )
    total_documents = int(len(set(ids[np.asarray(selected, dtype=int)].tolist())))
    return np.asarray(selected, dtype=int), {
        "scheme": "fixed_rate_per_stream_documents",
        "period_documents": int(period_documents),
        "labels_per_period": int(labels_per_period),
        "initial_documents": int(initial_documents),
        "initial_labels": int(initial_labels),
        "budget_unit": "unique_input_document",
        "query_ratings_collected": int(len(selected)),
        "unique_documents_labeled": total_documents,
        "execution_mode": "offline_replay_over_episode_visible_prefix",
        "sampling_rate": float(labels_per_period / period_documents),
        "window_count": int(len(windows)),
        "requested_labels": total_documents,
        "sampled_labels": total_documents,
        "sampled_query_ratings": int(len(selected)),
        "windows": windows,
    }


def _safety_route_from_predicted_clusters(
    *,
    safety_net: dict[str, Any],
    safety_indices: np.ndarray,
    sampling_metadata: dict[str, Any],
    document_ids: np.ndarray,
    predicted_document_cluster_ids: np.ndarray,
) -> dict[str, Any]:
    """Route a harmful safety window only through observable cluster assignments."""

    ids = np.asarray(document_ids).astype(str)
    predicted_clusters = np.asarray(predicted_document_cluster_ids).astype(str)
    if predicted_clusters.shape != ids.shape:
        raise ValueError("Safety routing cluster assignments and document IDs must align")
    if str(safety_net.get("status")) != "harmful":
        return {
            "status": "not_triggered",
            "predicted_document_cluster_ids": [],
            "basis": "safety_net_not_harmful",
        }
    harmful_windows = {
        int(value) for value in safety_net.get("harmful_window_indices", [])
    }
    selected_window: dict[str, Any] | None = None
    for window in reversed(list(sampling_metadata.get("windows", []))):
        if int(window.get("window_index", -1)) in harmful_windows:
            selected_window = window
            break
    if selected_window is not None:
        route_indices = np.asarray(selected_window.get("document_indices", []), dtype=int)
        basis = "latest_harmful_safety_window_all_documents"
        window_index: int | None = int(selected_window["window_index"])
    else:
        route_indices = np.asarray(safety_indices, dtype=int)
        basis = "aggregate_harmful_safety_sample"
        window_index = None
    if route_indices.size == 0:
        return {
            "status": "harmful_but_unlocalized",
            "predicted_document_cluster_ids": [],
            "basis": basis,
            "window_index": window_index,
        }
    unique_indices = _first_indices_by_document(route_indices, ids)
    cluster_ids = sorted(
        {
            str(cluster_id)
            for cluster_id in predicted_clusters[unique_indices].tolist()
            if str(cluster_id) not in {"", "-1", "none", "null"}
        }
    )
    if not cluster_ids:
        return {
            "status": "harmful_but_unlocalized_manual_review",
            "predicted_document_cluster_ids": [],
            "basis": basis,
            "window_index": window_index,
            "window_document_count": int(len(unique_indices)),
            "manual_review_required": True,
        }
    counts = {
        cluster_id: int(np.sum(predicted_clusters[unique_indices] == cluster_id))
        for cluster_id in cluster_ids
    }
    return {
        "status": "probe_required_for_predicted_clusters",
        "predicted_document_cluster_ids": cluster_ids,
        "basis": basis,
        "window_index": window_index,
        "window_document_count": int(len(unique_indices)),
        "predicted_cluster_document_counts": counts,
        "probe_is_final_harmfulness_authority": True,
    }


def _pre_gate_retraining_reasons(
    *,
    safety_status: str,
    safety_route_localized: bool,
    harmful_persistent_cluster_count: int,
    harmful_document_share: float,
    wide_harmful_document_share: float,
) -> list[str]:
    """Return only the document-specified pre-gate escalation triggers."""

    reasons: list[str] = []
    if str(safety_status) == "harmful" and not bool(safety_route_localized):
        reasons.append("safety_net_confirmed_unlocalized_harm")
    if int(harmful_persistent_cluster_count) > 1:
        reasons.append("multiple_harmful_persistent_clusters")
    if float(harmful_document_share) > float(wide_harmful_document_share):
        reasons.append("wide_harmful_document_share_exceeded")
    return reasons


def _random_record_per_document(
    indices: np.ndarray,
    document_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    pool = np.asarray(indices, dtype=int)
    ids = np.asarray(document_ids).astype(str)
    if np.any(pool < 0) or np.any(pool >= len(ids)):
        raise ValueError("record sampling indices are out of bounds")
    records_by_document: dict[str, list[int]] = {}
    for index in pool.tolist():
        records_by_document.setdefault(str(ids[int(index)]), []).append(int(index))
    rng = np.random.default_rng(int(seed))
    return np.asarray(
        [int(rng.choice(records)) for records in records_by_document.values()],
        dtype=int,
    )


def _random_record_per_document_query(
    indices: np.ndarray,
    document_ids: np.ndarray,
    query_ids: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    pool = np.asarray(indices, dtype=int)
    ids = np.asarray(document_ids).astype(str)
    queries = np.asarray(query_ids).astype(str)
    records_by_unit: dict[tuple[str, str], list[int]] = {}
    for index in pool.tolist():
        records_by_unit.setdefault((str(ids[index]), str(queries[index])), []).append(int(index))
    rng = np.random.default_rng(int(seed))
    return np.asarray(
        [int(rng.choice(records)) for records in records_by_unit.values()],
        dtype=int,
    )


def _run_safety_net(
    *,
    safety_indices: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    query_ids: np.ndarray,
    input_document_ids: np.ndarray,
    rater_scores: np.ndarray,
    reference_metric: float,
    reference_human_ceiling: dict[str, Any],
    reference_excess_human_error: dict[str, Any] | None = None,
    tolerance: float,
    metric_name: str,
    class_values: np.ndarray,
    require_human_ceiling: bool,
    paired_harmfulness_mode: str = "auto",
    minimum_documents: int,
    minimum_documents_per_query: int,
    expected_query_ids: np.ndarray,
    sampling_metadata: dict[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    selected = np.asarray(safety_indices, dtype=int)
    if selected.size == 0:
        return {
            "status": "disabled_or_no_stream_documents",
            "n_probe": 0,
            "sampling": "uniform_random_deployment_stream_documents",
            "sampling_metadata": sampling_metadata,
            "independent_of_drift_and_warning": True,
        }

    def evaluate(indices: np.ndarray, *, evaluation_seed: int) -> dict[str, Any]:
        local = np.asarray(indices, dtype=int)
        target_human_ceiling = estimate_human_ceiling(
            rater_scores[local],
            metric_name=metric_name,
            query_ids=query_ids[local],
            class_values=class_values,
        )
        result = _evaluate_harmfulness_mode(
            y_true=labels[local],
            y_pred=predictions[local],
            rater_scores=rater_scores[local],
            groups=input_document_ids[local],
            reference_excess_human_error=reference_excess_human_error,
            paired_harmfulness_mode=str(paired_harmfulness_mode),
            reference_metric=float(reference_metric),
            tolerance=float(tolerance),
            metric_name=metric_name,
            query_ids=query_ids[local],
            probabilities=probabilities[local],
            class_values=class_values,
            reference_human_ceiling=reference_human_ceiling.get("value"),
            target_human_ceiling=target_human_ceiling.get("value"),
            require_human_ceiling=bool(require_human_ceiling),
            minimum_documents=int(minimum_documents),
            minimum_documents_per_query=int(minimum_documents_per_query),
            expected_query_ids=expected_query_ids,
            n_boot=int(n_boot),
            seed=int(evaluation_seed),
        )
        return {**result, "human_ceiling_target": target_human_ceiling}

    aggregate_result = evaluate(selected, evaluation_seed=int(seed))
    query_count = max(1, len(set(np.asarray(expected_query_ids).astype(str).tolist())))
    rolling_record_limit = 2 * max(
        int(minimum_documents),
        int(minimum_documents_per_query) * query_count,
    )
    rolling: list[int] = []
    window_results: list[dict[str, Any]] = []
    for window in sampling_metadata.get("windows", []):
        local = [int(index) for index in window.get("sampled_record_indices", [])]
        rolling.extend(local)
        rolling = rolling[-rolling_record_limit:]
        if not rolling:
            continue
        result = evaluate(
            np.asarray(rolling, dtype=int),
            evaluation_seed=int(seed) + int(window["window_index"]) + 1,
        )
        window_results.append(
            {
                **result,
                "window_index": int(window["window_index"]),
                "new_sample_count": int(len(local)),
                "rolling_sample_count": int(len(rolling)),
            }
        )
    harmful_windows = [
        int(result["window_index"])
        for result in window_results
        if str(result.get("status")) == "harmful"
    ]
    status = "harmful" if harmful_windows else str(aggregate_result.get("status", "uncertain"))
    return {
        **aggregate_result,
        "status": status,
        "sampling": "uniform_random_deployment_stream_documents_per_window",
        "sampling_metadata": sampling_metadata,
        "independent_of_drift_and_warning": True,
        "decision_schedule": "rolling_evaluation_after_each_monitoring_window",
        "rolling_record_limit": int(rolling_record_limit),
        "by_window": window_results,
        "harmful_window_indices": harmful_windows,
        "aggregate_result": aggregate_result,
        "action": "manual_review_required" if status == "harmful" else "monitor_only",
    }


def _probe_persistent_document_clusters(
    *,
    probe_indices: np.ndarray,
    predicted_document_cluster_ids: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    query_ids: np.ndarray,
    input_document_ids: np.ndarray,
    rater_scores: np.ndarray | None = None,
    reference_metric: float,
    reference_human_ceiling: dict[str, Any] | None = None,
    reference_excess_human_error: dict[str, Any] | None = None,
    tolerance: float,
    metric_name: str,
    class_values: np.ndarray,
    require_human_ceiling: bool = False,
    paired_harmfulness_mode: str = "auto",
    minimum_documents: int = 4,
    minimum_documents_per_query: int = 2,
    expected_query_ids: np.ndarray | None = None,
    fdr_alpha: float = 0.05,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Run an independent harmfulness Probe for every predicted document cluster.

    The returned top-level status is an operational summary: it is ``harmful``
    when at least one document cluster is harmful. Per-cluster results remain the source
    of truth for routing Adapt, Gate, and Future-Test actions.
    """

    selected = np.asarray(probe_indices, dtype=int)
    predicted = np.asarray(predicted_document_cluster_ids).astype(str)
    if selected.size == 0:
        return {
            "status": "no_probe_candidates",
            "n_probe": 0,
            "budget_scope": "maximum_20_records_per_predicted_document_cluster",
            "probe_sampling_unit": "one_record_per_input_document_and_query",
            "by_predicted_document_cluster": {},
            "harmful_predicted_document_cluster_ids": [],
            "benign_predicted_document_cluster_ids": [],
            "uncertain_predicted_document_cluster_ids": [],
        }
    if np.any(selected < 0) or np.any(selected >= len(predicted)):
        raise ValueError("probe indices are out of bounds")
    if len(np.unique(selected)) != len(selected):
        raise ValueError("probe indices must not contain duplicates")

    rater_values = np.asarray(rater_scores, dtype=object) if rater_scores is not None else None
    reference_ceiling = (reference_human_ceiling or {}).get("value")
    by_document_cluster: dict[str, dict[str, Any]] = {}
    for offset, document_cluster_id in enumerate(sorted(set(predicted[selected].tolist()))):
        local = selected[predicted[selected] == document_cluster_id]
        if int(local.size) > 20:
            raise ValueError("Probe exceeds the final per-cluster budget of 20 records")
        target_human_ceiling = estimate_human_ceiling(
            rater_values[local] if rater_values is not None else None,
            metric_name=metric_name,
            query_ids=query_ids[local],
            class_values=class_values,
        )
        result = _evaluate_harmfulness_mode(
            y_true=labels[local],
            y_pred=predictions[local],
            rater_scores=rater_values[local] if rater_values is not None else None,
            groups=input_document_ids[local],
            reference_excess_human_error=reference_excess_human_error,
            paired_harmfulness_mode=str(paired_harmfulness_mode),
            reference_metric=float(reference_metric),
            tolerance=float(tolerance),
            metric_name=metric_name,
            query_ids=query_ids[local],
            probabilities=probabilities[local],
            class_values=class_values,
            reference_human_ceiling=reference_ceiling,
            target_human_ceiling=target_human_ceiling.get("value"),
            require_human_ceiling=bool(require_human_ceiling),
            minimum_documents=int(minimum_documents),
            minimum_documents_per_query=int(minimum_documents_per_query),
            expected_query_ids=expected_query_ids,
            n_boot=int(n_boot),
            seed=int(seed) + offset,
        )
        by_document_cluster[str(document_cluster_id)] = {
            **dict(result),
            "predicted_document_cluster_id": str(document_cluster_id),
            "n_probe": int(local.size),
            "human_ceiling_target": target_human_ceiling,
        }

    correction = _benjamini_hochberg(
        {
            document_cluster_id: _harmfulness_p_value(result)
            for document_cluster_id, result in by_document_cluster.items()
        },
        alpha=float(fdr_alpha),
    )
    for document_cluster_id, result in by_document_cluster.items():
        fdr = correction[document_cluster_id]
        result["unadjusted_status"] = str(result.get("status", "uncertain"))
        result["harmfulness_fdr_alpha"] = float(fdr_alpha)
        result["harmfulness_fdr_adjusted_p_value"] = fdr["adjusted_p_value"]
        result["harmfulness_fdr_rejected"] = fdr["rejected"]
        if result["unadjusted_status"] == "harmful" and not fdr["rejected"]:
            result["status"] = "uncertain"

    harmful = sorted(
        document_cluster_id
        for document_cluster_id, result in by_document_cluster.items()
        if str(result.get("status")) == "harmful"
    )
    benign = sorted(
        document_cluster_id
        for document_cluster_id, result in by_document_cluster.items()
        if str(result.get("status")) == "benign"
    )
    uncertain = sorted(
        document_cluster_id
        for document_cluster_id, result in by_document_cluster.items()
        if str(result.get("status")) == "uncertain"
    )
    if harmful:
        status = "harmful"
    elif uncertain:
        status = "uncertain"
    else:
        status = "benign"
    summary: dict[str, Any] = {
        "status": status,
        "n_probe": int(selected.size),
        "budget_scope": "maximum_20_records_per_predicted_document_cluster",
        "probe_sampling_unit": "one_record_per_input_document_and_query",
        "by_predicted_document_cluster": by_document_cluster,
        "harmful_predicted_document_cluster_ids": harmful,
        "benign_predicted_document_cluster_ids": benign,
        "uncertain_predicted_document_cluster_ids": uncertain,
    }
    # Retain flat metric fields in the single-cluster case for
    # existing reports, while avoiding a misleading aggregate metric when
    # multiple document clusters were probed independently.
    if len(by_document_cluster) == 1:
        only_result = next(iter(by_document_cluster.values()))
        summary = {**only_result, **summary}
    else:
        summary["metric"] = None
        summary["metric_name"] = metric_name
    return summary


def _harmfulness_p_value(result: dict[str, Any]) -> float:
    value = result.get("harmfulness_p_value")
    if value is not None and np.isfinite(value):
        return float(np.clip(value, 0.0, 1.0))
    return 0.0 if str(result.get("status")) == "harmful" else 1.0


def _evaluate_harmfulness_mode(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rater_scores: np.ndarray | None,
    groups: np.ndarray,
    reference_excess_human_error: dict[str, Any] | None,
    paired_harmfulness_mode: str,
    reference_metric: float,
    tolerance: float,
    metric_name: str,
    query_ids: np.ndarray,
    probabilities: np.ndarray,
    class_values: np.ndarray,
    reference_human_ceiling: float | None,
    target_human_ceiling: float | None,
    require_human_ceiling: bool,
    minimum_documents: int,
    minimum_documents_per_query: int,
    expected_query_ids: np.ndarray | None,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    mode = str(paired_harmfulness_mode)
    if mode not in {"auto", "required", "disabled"}:
        raise ValueError("paired_harmfulness_mode must be 'auto', 'required', or 'disabled'")
    paired_reference_available = bool(
        reference_excess_human_error
        and reference_excess_human_error.get("available")
        and reference_excess_human_error.get("value") is not None
    )
    if mode == "required" or (mode == "auto" and paired_reference_available):
        result = paired_excess_human_error_probe(
            y_true=y_true,
            y_pred=y_pred,
            rater_scores=rater_scores,
            reference=reference_excess_human_error,
            tolerance=float(tolerance),
            groups=groups,
            minimum_documents=int(minimum_documents),
            n_boot=int(n_boot),
            confidence=0.95,
            seed=int(seed),
        )
        result["mode_selection"] = {
            "configured": mode,
            "selected": "paired_excess_human_error",
            "paired_reference_available": paired_reference_available,
            "fallback_used": False,
        }
        return result
    result = harmfulness_probe(
        y_true=y_true,
        y_pred=y_pred,
        reference_metric=float(reference_metric),
        tolerance=float(tolerance),
        metric_name=metric_name,
        query_ids=query_ids,
        probabilities=probabilities,
        class_values=class_values,
        groups=groups,
        reference_human_ceiling=reference_human_ceiling,
        target_human_ceiling=target_human_ceiling,
        require_human_ceiling=bool(require_human_ceiling),
        minimum_documents=int(minimum_documents),
        minimum_documents_per_query=int(minimum_documents_per_query),
        expected_query_ids=expected_query_ids,
        n_boot=int(n_boot),
        seed=int(seed),
    )
    result["mode_selection"] = {
        "configured": mode,
        "selected": "general_metric_human_ceiling",
        "paired_reference_available": paired_reference_available,
        "fallback_used": mode == "auto",
        "fallback_reason": (
            "paired_excess_human_error_reference_unavailable" if mode == "auto" else "paired_mode_disabled"
        ),
    }
    return result


def _benjamini_hochberg(p_values: dict[str, float], *, alpha: float) -> dict[str, dict[str, float | bool]]:
    """Return BH adjusted p-values and rejection decisions keyed by cluster."""

    if not 0.0 < float(alpha) <= 1.0:
        raise ValueError("BH alpha must be in (0, 1]")
    ordered = sorted(
        (float(np.clip(value, 0.0, 1.0)), str(key))
        for key, value in p_values.items()
    )
    count = len(ordered)
    if not count:
        return {}
    rejected_rank = 0
    for rank, (p_value, _) in enumerate(ordered, start=1):
        if p_value <= float(alpha) * rank / count:
            rejected_rank = rank
    adjusted: list[float] = [0.0] * count
    running = 1.0
    for index in range(count - 1, -1, -1):
        p_value, _ = ordered[index]
        running = min(running, p_value * count / (index + 1))
        adjusted[index] = running
    return {
        key: {
            "adjusted_p_value": float(adjusted[index]),
            "rejected": bool(index + 1 <= rejected_rank),
        }
        for index, (_, key) in enumerate(ordered)
    }


def _restrict_routed_indices(
    indices: np.ndarray,
    routed_document_clusters: np.ndarray,
    *,
    allowed_document_cluster_ids: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only routes whose independently probed document cluster is harmful."""

    routed_indices = np.asarray(indices, dtype=int)
    routed = np.asarray(routed_document_clusters).astype(str)
    if routed_indices.shape != routed.shape:
        raise ValueError("routed indices and document clusters must be aligned")
    if not allowed_document_cluster_ids:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=str)
    mask = np.isin(routed, np.asarray(allowed_document_cluster_ids, dtype=str))
    return routed_indices[mask], routed[mask]


def _gate_decision(
    *,
    labels: np.ndarray,
    old_predictions: np.ndarray,
    new_predictions: np.ndarray,
    old_probabilities: np.ndarray | None = None,
    new_probabilities: np.ndarray | None = None,
    class_values: np.ndarray | None = None,
    training_validation_mask: np.ndarray,
    gate_indices: np.ndarray,
    training_drop_tolerance: float,
    query_ids: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    expected_query_ids: np.ndarray | None = None,
    gate_min_excess_error_improvement: float = 0.10,
    gate_max_negative_flip_rate: float = 0.05,
    minimum_documents: int = 2,
    bootstrap_samples: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    values = normalize_label_array(labels)
    old = normalize_label_array(old_predictions)
    new = normalize_label_array(new_predictions)
    training_mask = np.asarray(training_validation_mask, dtype=bool)
    gate_indices = np.asarray(gate_indices, dtype=int)
    queries = np.asarray(query_ids).astype(str) if query_ids is not None else None
    group_values = np.asarray(groups).astype(str) if groups is not None else None
    if len(values) != len(old) or len(values) != len(new) or training_mask.shape != (len(values),):
        raise ValueError("Gate labels, predictions, and training mask must be aligned")
    if queries is not None and len(queries) != len(values):
        raise ValueError("Gate query_ids must align with labels")
    if group_values is not None and len(group_values) != len(values):
        raise ValueError("Gate groups must align with labels")
    if np.any(gate_indices < 0) or np.any(gate_indices >= len(values)):
        raise ValueError("Gate indices are out of bounds")
    if not 0.0 <= float(gate_max_negative_flip_rate) <= 1.0:
        raise ValueError("gate_max_negative_flip_rate must be in [0, 1]")
    if int(minimum_documents) < 2:
        raise ValueError("Gate minimum_documents must be at least two")

    def metrics(indices: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None) -> dict[str, float]:
        local_probabilities = probabilities[indices] if probabilities is not None else None
        if queries is None:
            return judge_metrics(
                values[indices],
                predictions[indices],
                probabilities=local_probabilities,
                class_values=class_values,
            )
        return macro_query_judge_metrics(
            values[indices],
            predictions[indices],
            queries[indices],
            probabilities=local_probabilities,
            class_values=class_values,
        )["macro"]

    old_training = metrics(np.flatnonzero(training_mask), old, old_probabilities)
    new_training = metrics(np.flatnonzero(training_mask), new, new_probabilities)
    if gate_indices.size:
        old_gate = metrics(gate_indices, old, old_probabilities)
        new_gate = metrics(gate_indices, new, new_probabilities)
        old_log_loss = _multiclass_log_loss(
            values[gate_indices],
            old_probabilities[gate_indices] if old_probabilities is not None else None,
            class_values,
        )
        new_log_loss = _multiclass_log_loss(
            values[gate_indices],
            new_probabilities[gate_indices] if new_probabilities is not None else None,
            class_values,
        )
    else:
        return {
            "accepted": False,
            "failure_reasons": ["no_gate_samples"],
            "old_gate": {},
            "new_gate": {},
            "old_training": old_training,
            "new_training": new_training,
            "training_drop_tolerance": float(training_drop_tolerance),
            "metric_name": "paired_excess_human_error_improvement",
            "minimum_excess_error_improvement": float(gate_min_excess_error_improvement),
            "maximum_source_negative_flip_rate": float(gate_max_negative_flip_rate),
            "source_guard_negative_flip_rate": None,
            "source_guard_qwk_drop": None,
            "source_guard_protected": False,
            "paired_excess_error_improvement": None,
            "old_gate_log_loss": None,
            "new_gate_log_loss": None,
        }
    effective_groups = (
        group_values
        if group_values is not None
        else np.asarray([f"row-{index}" for index in range(len(values))], dtype=str)
    )
    source_guard_indices = np.flatnonzero(training_mask)
    gate_document_count = int(len(np.unique(effective_groups[gate_indices])))
    source_document_count = int(len(np.unique(effective_groups[source_guard_indices])))
    identifiability_reason = _gate_identifiability_reason(
        query_ids=queries,
        expected_query_ids=expected_query_ids,
        groups=effective_groups,
        indices=gate_indices,
        minimum_documents=int(minimum_documents),
    )
    if source_document_count < int(minimum_documents):
        identifiability_reason = "insufficient_independent_source_guard_documents"
    if identifiability_reason is not None:
        return {
            "accepted": False,
            "failure_reasons": [identifiability_reason],
            "old_gate": old_gate,
            "new_gate": new_gate,
            "old_training": old_training,
            "new_training": new_training,
            "training_drop_tolerance": float(training_drop_tolerance),
            "metric_name": "paired_excess_human_error_improvement",
            "minimum_excess_error_improvement": float(gate_min_excess_error_improvement),
            "minimum_independent_documents": int(minimum_documents),
            "gate_independent_documents": gate_document_count,
            "source_guard_independent_documents": source_document_count,
            "maximum_source_negative_flip_rate": float(gate_max_negative_flip_rate),
            "source_guard_negative_flip_rate": None,
            "source_guard_qwk_drop": None,
            "source_guard_protected": False,
            "paired_excess_error_improvement": None,
            "old_gate_log_loss": float(old_log_loss) if np.isfinite(old_log_loss) else None,
            "new_gate_log_loss": float(new_log_loss) if np.isfinite(new_log_loss) else None,
        }
    paired_improvement = _paired_excess_error_improvement(
        labels=values[gate_indices],
        old_predictions=old[gate_indices],
        new_predictions=new[gate_indices],
        groups=effective_groups[gate_indices],
        n_boot=int(bootstrap_samples),
        seed=int(seed),
    )
    improvement = float(paired_improvement["improvement"])
    improvement_lcb = float(paired_improvement["ci95"][0])
    point_improved = bool(
        np.isfinite(improvement)
        and improvement >= float(gate_min_excess_error_improvement)
    )
    lcb_positive = bool(np.isfinite(improvement_lcb) and improvement_lcb > 0.0)
    improved = bool(point_improved and lcb_positive)
    source_nfr = float(
        np.mean((old[source_guard_indices] == values[source_guard_indices]) & (new[source_guard_indices] != values[source_guard_indices]))
    ) if source_guard_indices.size else float("nan")
    source_qwk_drop = float(old_training["qwk"] - new_training["qwk"])
    source_protected_by_nfr = bool(np.isfinite(source_nfr) and source_nfr <= float(gate_max_negative_flip_rate))
    source_protected_by_qwk = bool(
        np.isfinite(source_qwk_drop) and source_qwk_drop <= float(training_drop_tolerance)
    )
    source_guard_protected = bool(source_protected_by_nfr and source_protected_by_qwk)
    # Promotion requires a practically meaningful target gain with positive
    # paired-bootstrap evidence and both source-domain protections.
    accepted = bool(improved and source_guard_protected)
    return {
        "accepted": accepted,
        "failure_reasons": ([] if accepted else [
            reason for reason, failed in {
                "gate_excess_error_improvement_below_threshold": not point_improved,
                "gate_excess_error_improvement_lcb_not_positive": not lcb_positive,
                "source_guard_nfr_protection_failed": not source_protected_by_nfr,
                "source_guard_qwk_protection_failed": not source_protected_by_qwk,
            }.items() if failed
        ]),
        "old_gate": old_gate,
        "new_gate": new_gate,
        "old_training": old_training,
        "new_training": new_training,
        "training_drop_tolerance": float(training_drop_tolerance),
        "metric_name": "paired_excess_human_error_improvement",
        "minimum_excess_error_improvement": float(gate_min_excess_error_improvement),
        "minimum_improvement_lcb": 0.0,
        "minimum_independent_documents": int(minimum_documents),
        "gate_independent_documents": gate_document_count,
        "source_guard_independent_documents": source_document_count,
        "maximum_source_negative_flip_rate": float(gate_max_negative_flip_rate),
        "source_guard_negative_flip_rate": source_nfr if np.isfinite(source_nfr) else None,
        "source_guard_qwk_drop": source_qwk_drop if np.isfinite(source_qwk_drop) else None,
        "source_guard_protected": source_guard_protected,
        "source_guard_protection_rule": "nfr_and_qwk_drop_both_required",
        "promotion_rule": "point_improvement_and_positive_lcb_and_nfr_and_qwk",
        "source_guard_protection_path": {
            "nfr": source_protected_by_nfr,
            "qwk_drop": source_protected_by_qwk,
        },
        "paired_excess_error_improvement": paired_improvement,
        "old_gate_log_loss": float(old_log_loss) if np.isfinite(old_log_loss) else None,
        "new_gate_log_loss": float(new_log_loss) if np.isfinite(new_log_loss) else None,
    }


def _gate_identifiability_reason(
    *,
    query_ids: np.ndarray | None,
    expected_query_ids: np.ndarray | None,
    groups: np.ndarray,
    indices: np.ndarray,
    minimum_documents: int,
) -> str | None:
    if len(np.unique(groups[indices])) < int(minimum_documents):
        return "insufficient_independent_gate_documents"
    if query_ids is None:
        return None
    expected = (
        sorted(set(np.asarray(expected_query_ids).astype(str).tolist()))
        if expected_query_ids is not None
        else sorted(set(query_ids[indices].tolist()))
    )
    for query_id in expected:
        local = indices[query_ids[indices] == query_id]
        if len(np.unique(groups[local])) < 2:
            return f"insufficient_gate_documents_for_query:{query_id}"
    return None


def _paired_excess_error_improvement(
    *,
    labels: np.ndarray,
    old_predictions: np.ndarray,
    new_predictions: np.ndarray,
    groups: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Estimate old-minus-new excess human error on paired gate documents.

    The human-disagreement term is identical for the old and new Judge on a
    given document, so it cancels exactly. The paired quantity is therefore
    ``|old-y| - |new-y|``, bootstrapped over independent documents.
    """

    truth = np.asarray(labels, dtype=np.float64)
    old = np.asarray(old_predictions, dtype=np.float64)
    new = np.asarray(new_predictions, dtype=np.float64)
    document_ids = np.asarray(groups).astype(str)
    if not (truth.shape == old.shape == new.shape == document_ids.shape):
        raise ValueError("Paired gate labels, predictions, and document groups must align")
    if truth.ndim != 1 or truth.size < 2 or int(n_boot) < 1:
        raise ValueError("Paired gate improvement needs at least two rows and one bootstrap draw")
    per_row = np.abs(old - truth) - np.abs(new - truth)
    unique_documents = np.unique(document_ids)
    per_document = np.asarray(
        [per_row[document_ids == document_id].mean() for document_id in unique_documents],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(
        per_document,
        size=(int(n_boot), len(per_document)),
        replace=True,
    ).mean(axis=1)
    return {
        "definition": "mean_document((abs(old-y)-e_H) - (abs(new-y)-e_H))",
        "human_error_cancellation": "exact_within_document",
        "improvement": float(per_document.mean()),
        "ci95": [float(value) for value in np.quantile(draws, [0.025, 0.975])],
        "independent_documents": int(len(unique_documents)),
        "bootstrap_unit": "input_document",
        "bootstrap_samples": int(n_boot),
    }


def _multiclass_log_loss(
    labels: np.ndarray,
    probabilities: np.ndarray | None,
    class_values: np.ndarray | None,
) -> float:
    if probabilities is None or class_values is None:
        return float("nan")
    classes = np.asarray(class_values)
    values = np.asarray(labels)
    index = {value: idx for idx, value in enumerate(classes.tolist())}
    if any(value not in index for value in values.tolist()):
        return float("nan")
    encoded = np.asarray([index[value] for value in values.tolist()], dtype=int)
    try:
        probs = np.asarray(probabilities, dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-12)
        return float(log_loss(encoded, probs, labels=np.arange(len(classes))))
    except ValueError:
        return float("nan")


def _sample_score_rows(
    records: list[JudgeRecord],
    scores: np.ndarray,
    labels: np.ndarray,
    judge_behavior_scores: np.ndarray,
    judge_behavior_labels: np.ndarray,
    judge_behavior_detector: str,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    y_ood: np.ndarray,
    predicted_document_cluster_ids: np.ndarray,
) -> list[dict[str, Any]]:
    predicted_document_cluster_assignments = np.asarray(predicted_document_cluster_ids).astype(str)
    rows: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        predicted_document_cluster_id = str(predicted_document_cluster_assignments[idx])
        rows.append(
            {
                "sample_id": record.sample_id,
                "input_document_id": record.input_document_id,
                "audit_document_group_id": record.audit_document_group_id,
                "document_distribution_role": record.document_distribution_role,
                "document_split": record.split,
                "document_ood_score": float(scores[idx]),
                "document_ood_status": str(labels[idx]),
                "judge_behavior_ood_detector": judge_behavior_detector,
                "judge_behavior_ood_score": float(judge_behavior_scores[idx]),
                "judge_behavior_ood_status": str(judge_behavior_labels[idx]),
                "is_deployment_document_ood_eval": int(y_ood[idx]),
                "predicted_document_cluster_id": None if predicted_document_cluster_id == "-1" else predicted_document_cluster_id,
                "judge_query_id": record.query_id,
                "judge_prediction": _json_scalar(predictions[idx]),
                "judge_confidence": float(np.max(probabilities[idx])),
            }
        )
    return rows


def _judge_fingerprint(judge_selection: Any, *, artifact_paths: dict[str, str] | None = None) -> str:
    payload = {
        "selected_candidate": judge_selection.selected_candidate,
        "model": judge_selection.model.to_metadata(),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    )
    for kind, raw_path in sorted((artifact_paths or {}).items()):
        path = Path(raw_path)
        digest.update(str(kind).encode("utf-8"))
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _metrics_by_query(
    labels: np.ndarray,
    predictions: np.ndarray,
    query_ids: np.ndarray,
    probabilities: np.ndarray | None = None,
    class_values: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    queries = np.asarray(query_ids).astype(str)
    return {
        query_id: judge_metrics(
            labels[queries == query_id],
            predictions[queries == query_id],
            probabilities=probabilities[queries == query_id] if probabilities is not None else None,
            class_values=class_values,
        )
        for query_id in sorted(set(queries.tolist()))
    }


def _label_cost_rows(
    probe_indices: np.ndarray,
    safety_net_indices: np.ndarray,
    adapt_indices: np.ndarray,
    gate_indices: np.ndarray,
    document_ids: np.ndarray,
    predicted_document_cluster_ids: np.ndarray,
) -> list[dict[str, Any]]:
    document_ids = np.asarray(document_ids).astype(str)
    predicted = np.asarray(predicted_document_cluster_ids).astype(str)

    def row(
        stage: str,
        indices: np.ndarray,
        *,
        requested_labels: int | None = None,
        reused_labels: int = 0,
    ) -> dict[str, Any]:
        selected = np.asarray(indices, dtype=int)
        document_cluster_values = predicted[selected].tolist()
        return {
            "stage": stage,
            "requested_labels": (
                int(len(set(document_ids[selected].tolist())))
                if requested_labels is None
                else int(requested_labels)
            ),
            "reused_previously_requested_labels": int(reused_labels),
            "indices": selected.tolist(),
            "predicted_document_cluster_ids": document_cluster_values,
            "labels_by_predicted_document_cluster": {
                document_cluster_id: int(sum(value == document_cluster_id for value in document_cluster_values))
                for document_cluster_id in sorted(set(document_cluster_values))
            },
        }

    rows = [
        row("probe", probe_indices),
        row("safety_net", safety_net_indices),
        row(
            "adapt_reuses_confirmed_harmful_probe",
            adapt_indices,
            requested_labels=0,
            reused_labels=int(len(set(document_ids[np.asarray(adapt_indices, dtype=int)].tolist()))),
        ),
        row("gate", gate_indices),
    ]
    rows.append(
        {
            "stage": "total",
            "requested_labels": int(
                len(
                    set(
                        document_ids[
                            np.concatenate(
                                [
                                    np.asarray(probe_indices, dtype=int),
                                    np.asarray(safety_net_indices, dtype=int),
                                    np.asarray(gate_indices, dtype=int),
                                ]
                            )
                        ].tolist()
                    )
                )
            ),
            "reused_previously_requested_labels": int(
                len(set(document_ids[np.asarray(adapt_indices, dtype=int)].tolist()))
            ),
            "indices": [],
            "predicted_document_cluster_ids": [],
            "labels_by_predicted_document_cluster": {},
        }
    )
    return rows


def _recovery(before: float, after: float, reference: float) -> float | None:
    denominator = float(reference - before)
    if denominator <= 1e-12:
        return None
    return float((after - before) / denominator)


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value
