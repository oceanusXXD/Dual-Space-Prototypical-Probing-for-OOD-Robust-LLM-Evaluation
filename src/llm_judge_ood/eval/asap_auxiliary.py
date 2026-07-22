from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.common.io import read_jsonl
from src.llm_judge_ood.lifecycle.drift import (
    MMDPermutationTest,
    WindowDriftConfig,
    run_dual_space_drift_monitor,
)
from src.llm_judge_ood.lifecycle.probe import paired_excess_human_error_probe
from src.llm_judge_ood.scores.knn import DocumentKNNScorer
from src.llm_judge_ood.scores.rmd import DocumentGaussianScorer
from src.llm_judge_ood.shared.feature_store import (
    load_hidden_feature_store,
    record_fingerprint,
)
from src.llm_judge_ood.shared.metrics import ood_metrics
from src.llm_judge_ood.shared.schema import JudgeRecord, load_judge_records


def evaluate_asap_auxiliary_benchmarks(
    *,
    reference_run: dict[str, Any],
    drift_config: WindowDriftConfig,
    within_input: Path,
    within_document_cache: Path,
    within_judge_cache: Path,
    semantic_input: Path,
    semantic_document_cache: Path,
    semantic_judge_cache: Path,
    probe_budget: int,
    probe_min_documents: int,
    harm_tolerance: float,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate frozen ASAP sidecars without admitting them to model selection."""

    within = _load_scenario(
        input_path=within_input,
        document_cache_path=within_document_cache,
        judge_cache_path=within_judge_cache,
    )
    semantic = _load_scenario(
        input_path=semantic_input,
        document_cache_path=semantic_document_cache,
        judge_cache_path=semantic_judge_cache,
    )
    _validate_sidecar_contract(within, scenario="within_prompt_covariate")
    _validate_sidecar_contract(semantic, scenario="semantic_task_shift")

    within_values = _frozen_representations(reference_run, within)
    semantic_values = _frozen_representations(reference_run, semantic)
    document_scorer = _fit_frozen_document_scorer(reference_run)
    _calibrate_frozen_vim(reference_run)

    rows: list[dict[str, Any]] = []
    scenario_audits: dict[str, Any] = {}
    for scenario_name, scenario, values in (
        ("within_prompt_covariate", within, within_values),
        ("semantic_task_shift", semantic, semantic_values),
    ):
        base_indices = _paired_base_indices(reference_run, scenario["rows"])
        a_scores = document_scorer.score(values["a"])
        b_scores = reference_run["vim"].score(values["u"])
        base_a_scores = document_scorer.score(reference_run["a"][base_indices])
        base_b_scores = reference_run["vim"].score(reference_run["output"].penultimate[base_indices])
        for space, base_scores, shifted_scores in (
            ("A_input_document", base_a_scores, a_scores),
            ("B_vim_residual", base_b_scores, b_scores),
        ):
            metrics = ood_metrics(
                np.concatenate(
                    [np.zeros(len(base_scores), dtype=int), np.ones(len(shifted_scores), dtype=int)]
                ),
                np.concatenate([base_scores, shifted_scores]),
            )
            rows.append(
                {
                    "scenario": scenario_name,
                    "stage": "detector",
                    "space": space,
                    "status": "ok",
                    "documents": int(len(shifted_scores)),
                    "auroc": float(metrics["auroc"]),
                    "aupr": float(metrics["aupr"]),
                    "fpr95": float(metrics["fpr95"]),
                    "evaluation_scope": "paired_independent_benchmark_test_id_controls",
                    "selection_scope": "development_only_frozen_before_sidecar",
                }
            )
        mmd_rows, mmd_audit = _whole_pool_mmd_rows(
            reference_run=reference_run,
            scenario_name=scenario_name,
            a_values=values["a"],
            b_values=values["b"],
            drift_config=drift_config,
            document_ids=np.asarray([str(row["input_document_id"]) for row in scenario["rows"]]),
        )
        rows.extend(mmd_rows)
        scenario_audits[scenario_name] = {
            "documents": int(len(scenario["rows"])),
            "mmd": mmd_audit,
        }

    monitor = _within_prompt_monitor(
        reference_run=reference_run,
        values=within_values,
        rows=within["rows"],
        drift_config=drift_config,
    )
    window_rows = monitor.window_rows
    rows.append(
        {
            "scenario": "within_prompt_covariate",
            "stage": "sequential",
            "space": "A_and_B",
            "status": "ok",
            "documents": int(len(within["rows"])),
            "windows": int(len(window_rows)),
            "valid_windows": int(sum("B" in row for row in window_rows)),
            "b_rejection_windows": int(sum(bool(row.get("b_sequential_reject")) for row in window_rows)),
            "persistent_b_drift": bool(monitor.first_persistent_episode is not None),
            "persistent_documents": int(monitor.persistent_document_indices.size),
            "first_persistent_window": (
                monitor.first_persistent_episode.get("window_index")
                if monitor.first_persistent_episode is not None
                else None
            ),
            "selection_scope": "frozen_source_and_calibration_reference",
        }
    )
    probe_row, probe_audit = _within_prompt_probe(
        reference_run=reference_run,
        scenario=within,
        values=within_values,
        monitor=monitor,
        probe_budget=int(probe_budget),
        probe_min_documents=int(probe_min_documents),
        harm_tolerance=float(harm_tolerance),
        bootstrap_samples=int(bootstrap_samples),
    )
    rows.append(probe_row)
    scenario_audits["within_prompt_covariate"].update(
        {
            "sequential": monitor.to_metadata(),
            "probe": probe_audit,
            "label_contract": "non_whitespace_token_sequence_exactly_equal",
        }
    )

    semantic_fail_closed = _semantic_fail_closed_audit(semantic)
    rows.append(
        {
            "scenario": "semantic_task_shift",
            "stage": "fail_closed",
            "space": "B_judge_task_input",
            "status": "passed" if semantic_fail_closed["passed"] else "failed",
            "documents": int(len(semantic["rows"])),
            "label_available": False,
            "judge_metric_attempted": False,
            "probe_attempted": False,
            "adapt_attempted": False,
            "gate_attempted": False,
            "required_action": "manual_review_or_separate_labeled_task_workflow",
        }
    )
    scenario_audits["semantic_task_shift"].update(
        {"fail_closed": semantic_fail_closed}
    )
    audit = {
        "artifact_type": "llm_judge_ood_asap_auxiliary_benchmark_v1",
        "status": "complete",
        "selection_used_auxiliary_documents": False,
        "frozen_reference_run": str(reference_run["name"]),
        "scenarios": scenario_audits,
        "claim_boundary": {
            "within_prompt_covariate": "format-only paired benchmark with labels",
            "semantic_task_shift": "detection and fail-closed only",
            "cross_prompt": "compound shift; never described as pure covariate shift",
        },
    }
    return rows, audit


def _load_scenario(
    *,
    input_path: Path,
    document_cache_path: Path,
    judge_cache_path: Path,
) -> dict[str, Any]:
    rows = read_jsonl(input_path)
    records = load_judge_records([input_path])
    if len(rows) != len(records):
        raise RuntimeError(f"Sidecar record parsing changed row count for {input_path}")
    return {
        "input_path": str(input_path),
        "rows": rows,
        "records": records,
        "document_raw": _aligned_cache_features(
            records=records,
            cache_path=document_cache_path,
            feature_scope="input_document",
        ),
        "judge_raw": _aligned_cache_features(
            records=records,
            cache_path=judge_cache_path,
            feature_scope="judge_input",
        ),
    }


def _aligned_cache_features(
    *,
    records: list[JudgeRecord],
    cache_path: Path,
    feature_scope: str,
) -> np.ndarray:
    store = load_hidden_feature_store(cache_path)
    metadata = store.metadata.get("cache_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"Auxiliary cache has no structured metadata: {cache_path}")
    if str(metadata.get("feature_scope")) != str(feature_scope):
        raise ValueError(
            f"Auxiliary cache {cache_path} has feature_scope={metadata.get('feature_scope')!r}, "
            f"expected {feature_scope!r}"
        )
    expected_fingerprint = record_fingerprint(records, feature_scope=feature_scope)
    if str(metadata.get("dataset_fingerprint")) != expected_fingerprint:
        raise ValueError(f"Auxiliary cache fingerprint does not match {feature_scope}: {cache_path}")
    if feature_scope == "input_document":
        if store.input_document_ids is None:
            raise ValueError(f"Input-document cache has no input_document_ids: {cache_path}")
        cache_ids = store.input_document_ids.astype(str)
        target_ids = np.asarray([record.input_document_id for record in records], dtype=str)
    else:
        cache_ids = store.sample_ids.astype(str)
        target_ids = np.asarray([record.sample_id for record in records], dtype=str)
    if len(cache_ids) != len(set(cache_ids.tolist())):
        raise ValueError(f"Auxiliary cache identifiers are not unique: {cache_path}")
    by_id = {identifier: index for index, identifier in enumerate(cache_ids.tolist())}
    missing = sorted(set(target_ids.tolist()) - set(by_id))
    extra = sorted(set(by_id) - set(target_ids.tolist()))
    if missing or extra:
        raise ValueError(
            f"Auxiliary cache identifiers do not exactly match input: missing={missing[:5]}, extra={extra[:5]}"
        )
    return np.stack([store.features[by_id[identifier]] for identifier in target_ids.tolist()])


def _validate_sidecar_contract(payload: dict[str, Any], *, scenario: str) -> None:
    rows = payload["rows"]
    if not rows:
        raise ValueError(f"ASAP auxiliary benchmark {scenario!r} is empty")
    lineages = [str(row.get("lineage_document_id") or "") for row in rows]
    if not all(lineages) or len(lineages) != len(set(lineages)):
        raise ValueError(f"ASAP auxiliary benchmark {scenario!r} needs unique explicit lineages")
    if any(str(row.get("shift_taxonomy")) != "id_h0" for row in rows):
        # Sidecar rows replace shift_taxonomy with their scenario, so verify
        # purity through the original prompt family instead.
        if any(int(row.get("asap_prompt_id", -1)) not in {1, 2} for row in rows):
            raise ValueError(f"ASAP auxiliary benchmark {scenario!r} is not within an ID prompt")
    if scenario == "within_prompt_covariate":
        if any(row.get("label") is None for row in rows):
            raise ValueError("Within-prompt benchmark requires preserved labels")
        if any(str(row.get("split")) != "benchmark_within_prompt" for row in rows):
            raise ValueError("Within-prompt benchmark has an invalid split")
    else:
        if any(row.get("label") is not None for row in rows):
            raise ValueError("Semantic/task benchmark must not expose labels")
        if any(str(row.get("document_distribution_role")) != "diagnostic" for row in rows):
            raise ValueError("Semantic/task benchmark must use the diagnostic role")


def _frozen_representations(
    reference_run: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, np.ndarray]:
    summary = reference_run["summary"]
    extractors = summary.get("feature_extractors", {})
    judge_extractor = extractors.get("judge_input", {})
    if str(judge_extractor.get("feature_scope")) != "judge_input":
        raise RuntimeError(
            "Formal auxiliary evaluation requires a v6 reference run with a separate judge_input cache"
        )
    a_layer = int(
        extractors.get("input_document_A_space", {}).get(
            "separability_selected_layer_index", 0
        )
    )
    judge_layer = int(judge_extractor.get("separability_selected_layer_index", a_layer))
    a = _pca_whitened_layer(
        scenario["document_raw"],
        layer_index=a_layer,
        artifact=Path(reference_run["result_dir"]) / "ood_preprocessor.npz",
    )
    selected_representation = (
        summary.get("ood_selection", {}).get("selected_candidate", {}).get("representation_name")
    )
    if str(selected_representation) != "last_layer":
        raise RuntimeError(
            "ASAP auxiliary evaluator currently requires the pre-registered last_layer A representation"
        )
    judge_features_2d = _pca_whitened_layer(
        scenario["judge_raw"],
        layer_index=judge_layer,
        artifact=Path(reference_run["result_dir"]) / "judge_preprocessor.npz",
    )
    judge_features = judge_features_2d[:, None, :].astype(np.float32)
    u = reference_run["model"].transform_u(judge_features)
    b = reference_run["vim"].residual_features(u)
    return {"a": a, "judge_features": judge_features, "u": u, "b": b}


def _pca_whitened_layer(
    raw_features: np.ndarray,
    *,
    layer_index: int,
    artifact: Path,
) -> np.ndarray:
    values = np.asarray(raw_features, dtype=np.float64)
    if values.ndim != 3 or not 0 <= int(layer_index) < values.shape[1]:
        raise ValueError(f"Invalid auxiliary hidden feature shape/layer: {values.shape}, {layer_index}")
    payload = np.load(artifact, allow_pickle=False)
    matrix = values[:, int(layer_index), :]
    return (
        (matrix - np.asarray(payload["pca_means"])[0])
        @ np.asarray(payload["components"])[0].T
        / np.sqrt(np.maximum(np.asarray(payload["explained_variance"])[0], 1e-5))
    )


def _paired_base_indices(reference_run: dict[str, Any], rows: list[dict[str, Any]]) -> np.ndarray:
    by_id = {
        str(row["input_document_id"]): index
        for index, row in enumerate(reference_run["rows"])
        if str(row.get("split")) == "benchmark_test"
        and str(row.get("document_shift_type")) == "id"
    }
    lineages = [str(row["lineage_document_id"]) for row in rows]
    missing = sorted(set(lineages) - set(by_id))
    if missing:
        raise ValueError(f"Auxiliary sidecar lineages are absent from benchmark_test ID controls: {missing[:5]}")
    return np.asarray([by_id[lineage] for lineage in lineages], dtype=int)


def _fit_frozen_document_scorer(reference_run: dict[str, Any]) -> Any:
    rows = reference_run["rows"]
    split = np.asarray([str(row["split"]) for row in rows])
    train = split == "training_train"
    calibration = split == "training_calibration"
    selected = reference_run["summary"]["ood_selection"]["selected_candidate"]
    config = reference_run["summary"]["ood_selection"]["config"]
    if str(selected["detector"]) == "knn":
        scorer: Any = DocumentKNNScorer(
            k=int(selected["k"]),
            metric=str(selected.get("metric") or "euclidean"),
            normalize=False,
        ).fit(reference_run["a"][train])
    elif str(selected["detector"]) == "mahalanobis":
        scorer = DocumentGaussianScorer(
            regularization=float(config.get("gaussian_regularization", 1e-5))
        ).fit(reference_run["a"][train])
    else:
        raise RuntimeError(f"Unsupported frozen A detector: {selected['detector']!r}")
    scorer.calibrate(
        reference_run["a"][calibration],
        soft_q=float(config["soft_quantile"]),
        hard_q=float(config["hard_quantile"]),
    )
    return scorer


def _calibrate_frozen_vim(reference_run: dict[str, Any]) -> None:
    split = np.asarray([str(row["split"]) for row in reference_run["rows"]])
    calibration = split == "training_calibration"
    config = reference_run["summary"]["judge_behavior_ood"]["config"]
    reference_run["vim"].calibrate(
        reference_run["output"].penultimate[calibration],
        soft_q=float(config["soft_quantile"]),
        hard_q=float(config["hard_quantile"]),
    )


def _whole_pool_mmd_rows(
    *,
    reference_run: dict[str, Any],
    scenario_name: str,
    a_values: np.ndarray,
    b_values: np.ndarray,
    drift_config: WindowDriftConfig,
    document_ids: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = np.asarray([str(row["split"]) for row in reference_run["rows"]])
    source = split == "training_drift_reference"
    source_blocks = np.asarray(
        [str(row.get("arrival_batch_id", row["input_document_id"])) for row in reference_run["rows"]]
    )[source]
    output: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    for offset, (space, source_values, target_values) in enumerate(
        (
            ("A_input_document", reference_run["a"][source], a_values),
            ("B_vim_residual", reference_run["b"][source], b_values),
        )
    ):
        result = MMDPermutationTest(drift_config).fit(
            source_values,
            block_ids=source_blocks,
        ).test(
            target_values,
            seed=int(drift_config.seed) + 700 + offset,
            block_ids=document_ids,
        )
        output.append(
            {
                "scenario": scenario_name,
                "stage": "mmd",
                "space": space,
                "status": str(result.get("status", "ok")),
                "documents": int(len(target_values)),
                "mmd2": result.get("mmd2"),
                "conservative_p_value": result.get("conservative_p_value"),
                "p_value": result.get("p_value"),
                "alpha_005_reject": (
                    float(result.get("conservative_p_value", 1.0)) <= 0.05
                ),
                "selection_scope": "frozen_source_drift_reference",
            }
        )
        audit[space] = result
    return output, audit


def _within_prompt_monitor(
    *,
    reference_run: dict[str, Any],
    values: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    drift_config: WindowDriftConfig,
) -> Any:
    base_rows = reference_run["rows"]
    split = np.asarray([str(row["split"]) for row in base_rows])
    base_count = len(base_rows)
    combined_ids = np.concatenate(
        [
            np.asarray([str(row["input_document_id"]) for row in base_rows]),
            np.asarray([str(row["input_document_id"]) for row in rows]),
        ]
    )
    combined_blocks = np.concatenate(
        [
            np.asarray(
                [str(row.get("arrival_batch_id", row["input_document_id"])) for row in base_rows]
            ),
            np.asarray([str(row["input_document_id"]) for row in rows]),
        ]
    )
    combined_a = np.vstack([reference_run["a"], values["a"]])
    combined_b = np.vstack([reference_run["b"], values["b"]])
    combined_b_scores = np.concatenate(
        [reference_run["b_score"], reference_run["vim"].score(values["u"])]
    )
    local_config = replace(drift_config, power_enabled=False)
    return run_dual_space_drift_monitor(
        document_embeddings=combined_a,
        behavior_embeddings=combined_b,
        document_ids=combined_ids,
        source_document_indices=np.flatnonzero(split == "training_drift_reference"),
        calibration_document_indices=np.flatnonzero(split == "training_calibration"),
        stream_document_indices=np.arange(base_count, base_count + len(rows), dtype=int),
        source_behavior_indices=np.flatnonzero(split == "training_drift_reference"),
        calibration_behavior_indices=np.flatnonzero(split == "training_calibration"),
        config=local_config,
        permutation_block_ids=combined_blocks,
        behavior_ood_scores=combined_b_scores,
    )


def _within_prompt_probe(
    *,
    reference_run: dict[str, Any],
    scenario: dict[str, Any],
    values: dict[str, np.ndarray],
    monitor: Any,
    probe_budget: int,
    probe_min_documents: int,
    harm_tolerance: float,
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_count = len(reference_run["rows"])
    persistent = np.asarray(monitor.persistent_document_indices, dtype=int) - int(base_count)
    persistent = persistent[(persistent >= 0) & (persistent < len(scenario["rows"]))]
    b_labels = reference_run["vim"].labels(reference_run["vim"].score(values["u"]))
    eligible = persistent[np.isin(b_labels[persistent], ["soft_ood", "hard_ood"])]
    if len(eligible) < int(probe_min_documents):
        audit = {
            "status": "not_triggered",
            "reason": "no_persistent_localized_stream_pool_with_minimum_documents",
            "persistent_documents": int(len(persistent)),
            "eligible_documents": int(len(eligible)),
            "source": "observed_persistent_B_contributor_documents",
        }
        return (
            {
                "scenario": "within_prompt_covariate",
                "stage": "benign_probe",
                "space": "persistent_B_contributors",
                "status": "not_triggered",
                "documents": int(len(eligible)),
                "probe_source": "observed_persistent_B_contributor_documents",
            },
            audit,
        )
    rng = np.random.default_rng(42)
    selected = np.asarray(
        rng.choice(
            eligible,
            size=min(int(probe_budget), len(eligible)),
            replace=False,
        ),
        dtype=int,
    )
    query_ids = np.asarray([str(row["query_id"]) for row in scenario["rows"]])
    output = reference_run["model"].predict_output(values["judge_features"], query_ids)
    predictions = output.classes[np.argmax(output.probabilities, axis=1)]
    labels = np.asarray([row["label"] for row in scenario["rows"]])
    raters = [row.get("rater_scores") for row in scenario["rows"]]
    groups = np.asarray([str(row["input_document_id"]) for row in scenario["rows"]])
    result = paired_excess_human_error_probe(
        y_true=labels[selected],
        y_pred=predictions[selected],
        rater_scores=[raters[index] for index in selected.tolist()],
        reference=reference_run["summary"]["paired_excess_human_error_reference"],
        tolerance=float(harm_tolerance),
        groups=groups[selected],
        minimum_documents=int(probe_min_documents),
        n_boot=int(bootstrap_samples),
        seed=42,
    )
    result = {
        **result,
        "source": "observed_persistent_B_contributor_documents",
        "budget": int(probe_budget),
        "selected_indices": selected.astype(int).tolist(),
    }
    return (
        {
            "scenario": "within_prompt_covariate",
            "stage": "benign_probe",
            "space": "persistent_B_contributors",
            "status": result.get("status"),
            "documents": int(len(selected)),
            "harm_delta": result.get("harm_delta"),
            "harm_delta_lcb": result.get("harm_delta_lcb"),
            "harm_delta_ucb": result.get("harm_delta_ucb"),
            "probe_source": "observed_persistent_B_contributor_documents",
        },
        result,
    )


def _semantic_fail_closed_audit(scenario: dict[str, Any]) -> dict[str, Any]:
    rows = scenario["rows"]
    checks = {
        "all_labels_absent": all(row.get("label") is None for row in rows),
        "all_groundtruth_absent": all(row.get("groundtruth") is None for row in rows),
        "all_rater_scores_absent": all(row.get("rater_scores") is None for row in rows),
        "all_marked_diagnostic_only": all(
            str(row.get("selection_role")) == "diagnostic_only_no_selection" for row in rows
        ),
        "all_require_fail_closed": all(
            str(row.get("required_behavior"))
            == "fail_closed_no_qwk_mae_or_harmfulness_claim"
            for row in rows
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "forbidden_outputs": ["qwk", "mae", "harmfulness", "probe", "adapt", "gate", "future"],
        "executed_outputs": ["A_detector", "B_detector", "A_MMD", "B_MMD"],
    }
