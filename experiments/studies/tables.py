from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.common.io import ensure_dir, write_json


def build_result_tables(summary: dict[str, Any], *, output_dir: str | Path) -> dict[str, str]:
    """Write compact CSV tables matching the LLM Judge OOD acceptance surfaces."""

    root = ensure_dir(output_dir)
    paths = {
        "block_calibration_audit": root / "table0_block_calibration_audit.csv",
        "judge": root / "table1_training_judge.csv",
        "judge_candidates": root / "table1b_judge_candidates.csv",
        "separability": root / "table1c_representation_separability.csv",
        "ood": root / "table2_document_ood.csv",
        "judge_ood_candidates": root / "table2b_judge_behavior_ood.csv",
        "ood_search": root / "table2c_document_ood_search.csv",
        "drift": root / "table3a_dual_space_drift.csv",
        "power": root / "table3c_power_analysis.csv",
        "warnings": root / "table3b_behavior_warnings.csv",
        "lifecycle": root / "table3_lifecycle.csv",
        "adaptation": root / "table4_adaptation.csv",
        "probe_acceptance": root / "table4a_probe_acceptance.csv",
        "acceptance": root / "table6_acceptance.csv",
        "label_cost": root / "table7_label_cost.csv",
        "monitoring": root / "table5_monitoring_baselines.csv",
        "manifest": root / "table_manifest.json",
    }
    pd.DataFrame(_block_calibration_audit_rows(summary)).to_csv(
        paths["block_calibration_audit"], index=False
    )
    pd.DataFrame([_judge_row(summary)]).to_csv(paths["judge"], index=False)
    pd.DataFrame(summary.get("judge_selection", {}).get("candidate_results", [])).to_csv(
        paths["judge_candidates"], index=False
    )
    pd.DataFrame(_separability_rows(summary)).to_csv(paths["separability"], index=False)
    pd.DataFrame(_ood_rows(summary)).to_csv(paths["ood"], index=False)
    pd.DataFrame(summary.get("judge_behavior_ood", {}).get("candidate_results", [])).to_csv(
        paths["judge_ood_candidates"], index=False
    )
    pd.DataFrame(_ood_search_rows(summary)).to_csv(paths["ood_search"], index=False)
    pd.DataFrame(_drift_rows(summary)).to_csv(paths["drift"], index=False)
    pd.DataFrame(_power_rows(summary)).to_csv(paths["power"], index=False)
    pd.DataFrame(_warning_rows(summary)).to_csv(paths["warnings"], index=False)
    pd.DataFrame([_lifecycle_row(summary)]).to_csv(paths["lifecycle"], index=False)
    pd.DataFrame([_adaptation_row(summary)]).to_csv(paths["adaptation"], index=False)
    pd.DataFrame(_probe_acceptance_rows(summary)).to_csv(paths["probe_acceptance"], index=False)
    pd.DataFrame(_acceptance_rows(summary)).to_csv(paths["acceptance"], index=False)
    pd.DataFrame(_label_cost_rows(summary)).to_csv(paths["label_cost"], index=False)
    pd.DataFrame(_monitoring_rows(summary)).to_csv(paths["monitoring"], index=False)
    write_json(paths["manifest"], {key: str(value) for key, value in paths.items()})
    return {key: str(value) for key, value in paths.items()}


def _block_calibration_audit_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """One automatic audit row per candidate window size.

    Candidate audits are created before Development selection.  Consequently a
    missing/invalid window is visible in the result artifact instead of being
    silently replaced by a nominal threshold or a deployment-derived choice.
    """

    selection = summary.get("lifecycle_selection", {})
    selected_window = selection.get("selected_metrics", {}).get("window_drift", {}).get("window_size")
    by_window: dict[int, dict[str, Any]] = {}
    for candidate in selection.get("candidate_results", []):
        config = candidate.get("window_drift", {})
        window_size = config.get("window_size")
        audit = candidate.get("block_calibration_audit")
        if window_size is None or not isinstance(audit, dict):
            continue
        window = int(window_size)
        current = by_window.get(window)
        # All W_min/alpha variants of one window share the same block audit.
        # Prefer the formally eligible row if there is more than one.
        if current is None or bool(audit.get("formal_valid")):
            by_window[window] = {
                "window_size": window,
                "source_blocks": audit.get("source_blocks"),
                "calibration_documents": audit.get("calibration_documents"),
                "calibration_blocks": audit.get("calibration_blocks"),
                "A_valid_windows": audit.get("a_valid_windows"),
                "B_valid_windows": audit.get("b_valid_windows"),
                "actual_C2ST_folds": audit.get("actual_c2st_folds"),
                "formal_valid": audit.get("formal_valid"),
                "nominal_fallback_for_smoke": audit.get("nominal_fallback_for_smoke"),
                "insufficient_target_blocks": audit.get("insufficient_target_blocks"),
                "minimum_valid_calibration_windows": audit.get(
                    "minimum_valid_calibration_windows"
                ),
                "formal_candidate_eligible": candidate.get("formal_candidate_eligible"),
                "selected_for_lifecycle": window == selected_window,
                "selection_used_deployment_documents": candidate.get(
                    "selection_used_deployment_documents"
                ),
                "failure_reasons": ";".join(audit.get("failure_reasons", [])),
            }
    if by_window:
        return [by_window[window] for window in sorted(by_window)]

    drift = summary.get("dual_space_drift", {})
    source = drift.get("source", {})
    calibration = drift.get("calibration", {})
    config = drift.get("config", {})
    a_source = source.get("A", {})
    b_source = source.get("B", {})
    results = drift.get("calibration_results", [])
    folds = [
        int(space.get("c2st", {}).get("folds"))
        for row in results
        for space in (row.get("A", {}), row.get("B", {}))
        if isinstance(space.get("c2st"), dict) and space["c2st"].get("folds") is not None
    ]
    a_thresholds = calibration.get("calibrated_thresholds", {}).get("A", {})
    b_thresholds = calibration.get("calibrated_thresholds", {}).get("B", {})
    return [
        {
            "window_size": config.get("window_size"),
            "source_blocks": min(
                value
                for value in (
                    a_source.get("mmd", {}).get("source_blocks"),
                    b_source.get("mmd", {}).get("source_blocks"),
                )
                if value is not None
            )
            if any(
                value is not None
                for value in (
                    a_source.get("mmd", {}).get("source_blocks"),
                    b_source.get("mmd", {}).get("source_blocks"),
                )
            )
            else None,
            "calibration_documents": a_source.get("calibration_documents"),
            "calibration_blocks": None,
            "A_valid_windows": a_thresholds.get("window_count"),
            "B_valid_windows": b_thresholds.get("window_count"),
            "actual_C2ST_folds": min(folds) if folds else None,
            "formal_valid": calibration.get("formal_calibration_valid"),
            "nominal_fallback_for_smoke": calibration.get("nominal_fallback_for_smoke"),
            "insufficient_target_blocks": any(
                row.get("status") == "insufficient_target_blocks"
                or row.get("A", {}).get("status") == "insufficient_target_blocks"
                or row.get("B", {}).get("status") == "insufficient_target_blocks"
                for row in results
            ),
            "minimum_valid_calibration_windows": config.get(
                "minimum_valid_calibration_windows"
            ),
            "formal_candidate_eligible": calibration.get("formal_calibration_valid"),
            "selected_for_lifecycle": True,
            "selection_used_deployment_documents": drift.get(
                "selection_used_deployment_records"
            ),
            "failure_reasons": ";".join(
                calibration.get("calibration_failure_reasons", [])
            ),
        }
    ]


def _judge_row(summary: dict[str, Any]) -> dict[str, Any]:
    training = summary.get("training_reference_judge_metrics", {})
    source_test = summary.get("training_test_judge_metrics", {})
    before = summary.get("deployment_before_adaptation", {})
    after = summary.get("deployment_after_adaptation", {})
    return {
        "method": summary.get("judge_selection", {}).get("selected_name", "unknown"),
        "loss": summary.get("judge", {}).get("config", {}).get("loss"),
        "training_accuracy": training.get("accuracy"),
        "training_qwk": training.get("qwk"),
        "training_macro_qwk": summary.get("training_reference_judge_macro_metrics", {}).get("qwk"),
        "majority_macro_qwk": summary.get("judge_selection", {}).get("majority_baseline", {}).get("macro_qwk"),
        "neural_beats_baseline": summary.get("judge_selection", {}).get("neural_beats_baseline"),
        "training_spearman": training.get("spearman"),
        "source_test_accuracy": source_test.get("accuracy"),
        "source_test_qwk": source_test.get("qwk"),
        "source_test_mae": source_test.get("mae"),
        "source_test_spearman": source_test.get("spearman"),
        "deployment_accuracy_before": before.get("accuracy"),
        "deployment_accuracy_after": after.get("accuracy"),
        "recovery_accuracy": summary.get("recovery_accuracy"),
    }


def _ood_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, metrics in sorted(summary.get("ood_metrics", {}).items()):
        if not isinstance(metrics, dict):
            continue
        if "auroc" not in metrics:
            for subgroup, subgroup_metrics in sorted(metrics.items()):
                if not isinstance(subgroup_metrics, dict) or "auroc" not in subgroup_metrics:
                    continue
                rows.append(
                    {
                        "detector": f"{name}_{subgroup}",
                        "auroc": subgroup_metrics.get("auroc"),
                        "aupr": subgroup_metrics.get("aupr"),
                        "fpr95": subgroup_metrics.get("fpr95"),
                        "soft_threshold": None,
                        "hard_threshold": None,
                    }
                )
            continue
        rows.append(
            {
                "detector": name,
                "auroc": metrics.get("auroc"),
                "aupr": metrics.get("aupr"),
                "fpr95": metrics.get("fpr95"),
                "soft_threshold": summary.get("thresholds", {}).get("soft") if name.startswith("selected_") else None,
                "hard_threshold": summary.get("thresholds", {}).get("hard") if name.startswith("selected_") else None,
            }
        )
    return rows


def _separability_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostic = summary.get("representation_separability", {})
    rows: list[dict[str, Any]] = []
    for layer in diagnostic.get("layers", []):
        base = {
            "available": diagnostic.get("available"),
            "selected_layer_index": diagnostic.get("selected_layer_index"),
            "layer_index": layer.get("layer_index"),
            "selection_auroc": layer.get("selection_auroc"),
        }
        by_shift = layer.get("by_shift", {})
        if not by_shift:
            rows.append(base)
            continue
        for shift_type, metrics in sorted(by_shift.items()):
            rows.append(
                {
                    **base,
                    "shift_type": shift_type,
                    "auroc": metrics.get("auroc"),
                    "source_documents": metrics.get("source_documents"),
                    "target_documents": metrics.get("target_documents"),
                    "cv_folds": metrics.get("cv_folds"),
                }
            )
    if not rows:
        rows.append(
            {
                "available": diagnostic.get("available"),
                "unavailable_reason": diagnostic.get("unavailable_reason"),
                "selected_layer_index": diagnostic.get("selected_layer_index"),
            }
        )
    return rows


def _ood_search_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate in summary.get("ood_selection", {}).get("candidate_results", []):
        rows.append(
            {
                key: value
                for key, value in candidate.items()
                if key != "representation"
            }
        )
    return rows


def _lifecycle_row(summary: dict[str, Any]) -> dict[str, Any]:
    probe = summary.get("probe", {})
    lifecycle = summary.get("lifecycle", {})
    latency_windows = lifecycle.get("confirmation_latency_windows", {})
    latency_samples = lifecycle.get("confirmation_latency_samples", {})
    return {
        "probe_status": probe.get("status"),
        "probe_metric": probe.get("metric"),
        "n_probe": probe.get("n_probe"),
        "num_confirmed_document_clusters": lifecycle.get("num_confirmed_document_clusters"),
        "confirmation_latency_windows_mean": latency_windows.get("mean"),
        "confirmation_latency_samples_mean": latency_samples.get("mean"),
    }


def _drift_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in summary.get("dual_space_drift", {}).get("window_results", []):
        a = row.get("A", {})
        b = row.get("B", {})
        a_mmd = a.get("mmd", {})
        b_mmd = b.get("mmd", {})
        a_c2st = a.get("c2st", {})
        b_c2st = b.get("c2st", {})
        b_ks = b.get("ks", {})
        rows.append(
            {
                "window_index": row.get("window_index"),
                "document_count": row.get("document_count"),
                "judge_record_count": row.get("judge_record_count"),
                "primary_test": a.get("primary_test"),
                "a_primary_p_value": a.get("p_value"),
                "a_status": a.get("status"),
                "b_primary_p_value": b.get("p_value"),
                "b_status": b.get("status"),
                "a_mmd_statistic": a_mmd.get("statistic"),
                "a_mmd_p_value": a_mmd.get("p_value"),
                "b_mmd_statistic": b_mmd.get("statistic"),
                "b_mmd_p_value": b_mmd.get("p_value"),
                "a_c2st_accuracy": a_c2st.get("accuracy"),
                "a_c2st_p_value": a_c2st.get("p_value"),
                "b_c2st_accuracy": b_c2st.get("accuracy"),
                "b_c2st_p_value": b_c2st.get("p_value"),
                "b_ks_statistic": b_ks.get("statistic"),
                "b_ks_p_value": b_ks.get("p_value"),
                "alpha_t": row.get("alpha_t"),
                "b_sequential_reject": row.get("b_sequential_reject"),
                "persistent_b_drift": row.get("persistent_b_drift"),
                "quadrant": row.get("quadrant"),
                "status": row.get("status"),
            }
        )
    return rows


def _power_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    power = summary.get("dual_space_drift", {}).get("power_analysis", {})
    rows: list[dict[str, Any]] = []
    for space, payload in sorted(power.get("spaces", {}).items()):
        for row in payload.get("results", []):
            rows.append({"space": space, **row})
    return rows


def _warning_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    warnings = summary.get("behavior_warning", {}).get("by_predicted_document_cluster", {})
    return [
        {
            "predicted_document_cluster_id": cluster_id,
            "triggered": warning.get("triggered"),
            "record_count": warning.get("record_count"),
            "cluster_document_count": warning.get("cluster_document_count"),
            "atc_estimated_accuracy": warning.get("atc_estimated_accuracy"),
            "doc_estimated_accuracy": warning.get("doc_estimated_accuracy"),
            "atc_estimated_accuracy_drop": warning.get("atc_estimated_accuracy_drop"),
            "doc_estimated_accuracy_drop": warning.get("doc_estimated_accuracy_drop"),
            "agreement_available": warning.get("agreement_on_the_line", {}).get("available"),
            "agreement": warning.get("agreement_on_the_line", {}).get("agreement"),
            "agreement_estimated_accuracy": warning.get("agreement_on_the_line", {}).get(
                "estimated_accuracy"
            ),
            "agreement_estimated_accuracy_drop": warning.get("agreement_on_the_line", {}).get(
                "estimated_accuracy_drop"
            ),
            "mean_confidence": warning.get("mean_confidence"),
            "mean_margin": warning.get("mean_margin"),
            "mean_energy": warning.get("mean_energy"),
            "mean_ood_score": warning.get("mean_ood_score"),
        }
        for cluster_id, warning in sorted(warnings.items())
    ]


def _adaptation_row(summary: dict[str, Any]) -> dict[str, Any]:
    adaptation = summary.get("adaptation", {})
    gate = adaptation.get("gate", {})
    type2 = adaptation.get("type2_new_query", {})
    coral = adaptation.get("coral_baseline", {})
    coral_metrics = coral.get("deployment_metrics", {})
    gate_difference = gate.get("paired_excess_error_improvement") or {}
    gate_interval = gate_difference.get("ci95", [None, None])
    return {
        "head_adapted_query_ids": ",".join(adaptation.get("adapter", {}).get("adapted_query_ids", [])),
        "coral_deployment_accuracy": coral_metrics.get("accuracy"),
        "coral_deployment_qwk": coral_metrics.get("qwk"),
        "type2_trained_query_ids": ",".join(type2.get("trained_query_ids", [])),
        "gate_accepted": gate.get("accepted"),
        "gate_failure_reasons": ",".join(gate.get("failure_reasons", [])),
        "gate_excess_error_improvement": gate_difference.get("improvement"),
        "gate_excess_error_improvement_ci95_low": gate_interval[0] if len(gate_interval) > 0 else None,
        "gate_excess_error_improvement_ci95_high": gate_interval[1] if len(gate_interval) > 1 else None,
        "gate_source_guard_negative_flip_rate": gate.get("source_guard_negative_flip_rate"),
        "gate_source_guard_qwk_drop": gate.get("source_guard_qwk_drop"),
        "gate_source_guard_protected": gate.get("source_guard_protected"),
        "source_guard_bwt_proxy": adaptation.get("source_guard_bwt_proxy", {}).get("value"),
        "future_worst_group_accuracy_before": summary.get("future_worst_group_accuracy", {}).get("before", {}).get("accuracy"),
        "future_worst_group_accuracy_after": summary.get("future_worst_group_accuracy", {}).get("after", {}).get("accuracy"),
        "requested_probe_labels": adaptation.get("requested_probe_labels"),
        "requested_deployment_adapt_labels": adaptation.get("requested_adapt_labels"),
        "requested_deployment_gate_labels": adaptation.get("requested_gate_labels"),
        "requested_total_labels": adaptation.get("requested_total_labels"),
    }


def _probe_acceptance_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    probe = summary.get("probe", {})
    warnings = summary.get("behavior_warning", {}).get("by_predicted_document_cluster", {})
    rows: list[dict[str, Any]] = []
    for cluster_id, result in sorted(probe.get("by_predicted_document_cluster", {}).items()):
        warning = warnings.get(cluster_id, {})
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": warning.get("cluster_document_count"),
                "warning": warning.get("triggered"),
                "n_probe": result.get("n_probe"),
                "harm_delta": result.get("harm_delta"),
                "LCB": result.get("harm_delta_lcb"),
                "bootstrap_p": result.get("harmfulness_p_value"),
                "BH_adjusted_p": result.get("harmfulness_fdr_adjusted_p_value"),
                "BH_FDR_rejected": result.get("harmfulness_fdr_rejected"),
                "status": result.get("status"),
                "selection_used_probe_labels": result.get("selection", {}).get("used_labels"),
                "all_probe_labels_used_for_estimation": result.get("selection", {}).get(
                    "all_probe_labels_used_for_estimation"
                ),
            }
        )
    return rows


def _acceptance_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """A compact actual-vs-threshold surface for static and update claims."""

    source_test = summary.get("training_test_judge_metrics", {})
    majority = summary.get("judge_selection", {}).get("majority_baseline", {})
    separability = _separability_rows(summary)
    ood = _ood_rows(summary)
    adaptation = summary.get("adaptation", {})
    gate = adaptation.get("gate", {})
    future_before = summary.get("deployment_before_adaptation", {})
    future_after = summary.get("deployment_after_adaptation", {})
    judge_behavior = summary.get("judge_behavior_ood", {})
    selected_vim = judge_behavior.get("selected_candidate", {})
    if str(selected_vim.get("detector", "")).lower() != "vim":
        selected_vim = next(
            (
                candidate
                for candidate in judge_behavior.get("candidate_results", [])
                if str(candidate.get("detector", "")).lower() == "vim"
            ),
            {},
        )
    rows: list[dict[str, Any]] = [
        {
            "area": "static",
            "metric": "Judge source test QWK",
            "value": source_test.get("qwk"),
            "threshold": "> 0 and > majority",
            "passed": (
                source_test.get("qwk") is not None
                and source_test.get("qwk") > 0
                and source_test.get("qwk") > majority.get("macro_qwk", float("inf"))
            ),
        },
        {
            "area": "static",
            "metric": "Judge source test MAE",
            "value": source_test.get("mae"),
            "threshold": "reported",
            "passed": source_test.get("mae") is not None,
        },
        {
            "area": "static",
            "metric": "ViM residual B-space AUROC",
            "value": selected_vim.get("development_auroc") if selected_vim else None,
            "threshold": "> 0.50; report RMD/kNN/MSP controls",
            "passed": bool(selected_vim and selected_vim.get("development_auroc", 0) > 0.50),
        },
        {
            "area": "adaptation",
            "metric": "requested_adapt_labels",
            "value": adaptation.get("requested_adapt_labels"),
            "threshold": "= 0",
            "passed": adaptation.get("requested_adapt_labels") == 0,
        },
        {
            "area": "adaptation",
            "metric": "Gate acceptance condition",
            "value": gate.get("accepted"),
            "threshold": (
                "target improvement >= 0.10 AND bootstrap LCB > 0 AND "
                "NFR <= 0.05 AND guard QWK drop <= 0.02"
            ),
            "passed": gate.get("accepted"),
        },
        {
            "area": "future",
            "metric": "Future MAE improvement",
            "value": _difference(future_before.get("mae"), future_after.get("mae")),
            "threshold": "after < before",
            "passed": _lower_is_better_improved(future_before.get("mae"), future_after.get("mae")),
        },
        {
            "area": "future",
            "metric": "Future QWK improvement",
            "value": _difference(future_after.get("qwk"), future_before.get("qwk")),
            "threshold": "after > before",
            "passed": _higher_is_better_improved(future_before.get("qwk"), future_after.get("qwk")),
        },
    ]
    for row in separability:
        shift = row.get("shift_type")
        if shift not in {"near", "far"}:
            continue
        threshold = 0.65 if shift == "near" else 0.80
        rows.append(
            {
                "area": "static",
                "metric": f"{shift} representation AUROC",
                "value": row.get("auroc"),
                "threshold": f">= {threshold:.2f}",
                "passed": row.get("auroc") is not None and row["auroc"] >= threshold,
            }
        )
    for row in ood:
        detector = str(row.get("detector", ""))
        for shift, threshold in (("near", 0.65), ("far", 0.80)):
            if shift not in detector.lower():
                continue
            rows.append(
                {
                    "area": "static",
                    "metric": f"{shift} document OOD AUROC ({detector})",
                    "value": row.get("auroc"),
                    "threshold": f">= {threshold:.2f}",
                    "passed": row.get("auroc") is not None and row["auroc"] >= threshold,
                }
            )
    return rows


def _label_cost_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    adaptation = summary.get("adaptation", {})
    return [
        {
            "stage": "Probe",
            "requested_labels": adaptation.get("requested_probe_labels"),
            "reused_labels": 0,
            "limit": "<= 20 / predicted cluster",
        },
        {
            "stage": "Adapt",
            "requested_labels": adaptation.get("requested_adapt_labels"),
            "reused_labels": adaptation.get("reused_probe_labels_for_adapt"),
            "limit": "requested = 0; reuse harmful Probe labels",
        },
        {
            "stage": "Gate",
            "requested_labels": adaptation.get("requested_gate_labels"),
            "reused_labels": 0,
            "limit": "<= 20 / harmful predicted cluster; independent",
        },
        {
            "stage": "Safety net",
            "requested_labels": adaptation.get("requested_safety_net_labels"),
            "reused_labels": 0,
            "limit": "account separately",
        },
        {
            "stage": "Total",
            "requested_labels": adaptation.get("requested_total_labels"),
            "reused_labels": adaptation.get("reused_probe_labels_for_adapt"),
            "limit": "single-cluster main path normally <= 40 excluding safety net",
        },
    ]


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _lower_is_better_improved(before: Any, after: Any) -> bool | None:
    if before is None or after is None:
        return None
    return bool(after < before)


def _higher_is_better_improved(before: Any, after: Any) -> bool | None:
    if before is None or after is None:
        return None
    return bool(after > before)


def _monitoring_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    monitoring = summary.get("monitoring_baselines", {})
    rows: list[dict[str, Any]] = []
    for method in monitoring.get("methods", []):
        rows.append({key: value for key, value in method.items() if key != "events"})
    return rows
