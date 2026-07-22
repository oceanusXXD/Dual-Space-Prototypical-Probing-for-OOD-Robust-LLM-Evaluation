"""Configuration loading helpers shared by LLM Judge OOD entrypoints."""

from __future__ import annotations

from typing import Any

from src.llm_judge_ood.adapt.head import HeadAdaptConfig
from src.llm_judge_ood.lifecycle.cluster import ClusterConfig
from src.llm_judge_ood.lifecycle.drift import WindowDriftConfig
from src.llm_judge_ood.lifecycle.persistence import PersistenceConfig
from src.llm_judge_ood.lifecycle.separability import SeparabilityConfig
from src.llm_judge_ood.lifecycle.warning import BehaviorWarningConfig
from src.llm_judge_ood.model.judge import JudgeTrainingConfig
from src.llm_judge_ood.model.selection import JudgeSelectionConfig
from src.llm_judge_ood.pipelines.sample_ood import SampleOODConfig
from src.llm_judge_ood.scores.judge_selection import JudgeOODSelectionConfig
from src.llm_judge_ood.scores.selection import OODSelectionConfig


_NESTED_CONFIGS = {
    "judge": JudgeTrainingConfig,
    "judge_selection": JudgeSelectionConfig,
    "ood_selection": OODSelectionConfig,
    "judge_ood_selection": JudgeOODSelectionConfig,
    "window_drift": WindowDriftConfig,
    "behavior_warning": BehaviorWarningConfig,
    "separability": SeparabilityConfig,
    "cluster": ClusterConfig,
    "persistence": PersistenceConfig,
    "head_adapt": HeadAdaptConfig,
}

_NESTED_SEQUENCE_FIELDS = {
    "judge_selection": (
        "preprocess_methods",
        "neural_losses",
        "neural_seeds",
        "ridge_alphas",
        "linear_learning_rates",
        "linear_cs",
    ),
    "ood_selection": (
        "preprocess_methods",
        "representations",
        "detectors",
        "metrics",
        "k_values",
    ),
    "judge_ood_selection": (
        "detectors",
        "vim_ranks",
        "knn_ks",
        "react_quantiles",
    ),
    "window_drift": ("power_effect_sizes", "power_window_sizes"),
}

_SAMPLE_SEQUENCE_FIELDS = (
    "input_paths",
    "training_document_train_splits",
    "training_document_drift_reference_splits",
    "training_document_calibration_splits",
    "training_document_validation_splits",
    "training_document_guard_splits",
    "training_document_test_splits",
    "development_document_splits",
    "benchmark_document_splits",
    "deployment_document_stream_splits",
    "deployment_document_ood_evaluation_splits",
    "deployment_document_probe_splits",
    "deployment_document_adapt_splits",
    "deployment_document_gate_splits",
    "deployment_document_future_splits",
    "deployment_document_evaluation_splits",
    "lifecycle_window_sizes",
    "lifecycle_minimum_consecutive_windows",
    "lifecycle_alpha_fwers",
    "lifecycle_alpha_spendings",
)


def config_from_mapping(payload: dict[str, Any]) -> SampleOODConfig:
    """Build a pipeline config from a JSON-compatible mapping.

    JSON arrays are converted to tuples for immutable dataclass fields. Unknown
    keys fail early, which keeps stale experiment configs from being silently
    accepted by the document-distribution pipeline.
    """
    known = set(SampleOODConfig.__dataclass_fields__) - set(_NESTED_CONFIGS)
    unexpected = sorted(set(payload) - known - set(_NESTED_CONFIGS))
    if unexpected:
        raise ValueError(
            "Unsupported document-level OOD config keys: "
            f"{unexpected}. Regenerate config with document-distribution terminology."
        )

    nested_configs = {}
    for name, config_type in _NESTED_CONFIGS.items():
        nested_payload = _tuple_fields(
            dict(payload.get(name, {})),
            _NESTED_SEQUENCE_FIELDS.get(name, ()),
        )
        nested_configs[name] = config_type(**nested_payload)
    kwargs = _tuple_fields(
        {key: value for key, value in payload.items() if key in known},
        _SAMPLE_SEQUENCE_FIELDS,
    )
    return SampleOODConfig(**kwargs, **nested_configs)


def _tuple_fields(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    for field in fields:
        if field in payload:
            payload[field] = tuple(payload[field])
    return payload
