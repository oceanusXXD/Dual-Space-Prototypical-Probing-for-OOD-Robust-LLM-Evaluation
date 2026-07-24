from __future__ import annotations

from typing import Any

import numpy as np

from src.common.metrics import (
    bootstrap_judge_metric_draws,
    judge_metrics,
    macro_query_judge_metrics,
)


LOWER_IS_BETTER_METRICS = frozenset({"mae", "normalized_mae", "ordinal_log_loss"})


def estimate_excess_human_error_reference(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rater_scores: np.ndarray | list[object] | tuple[object, ...] | None,
    groups: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate fixed guard-set mean g = |prediction-label| - |rater1-rater2|."""

    paired = _paired_document_errors(
        y_true=y_true,
        y_pred=y_pred,
        rater_scores=rater_scores,
        groups=groups,
    )
    if not paired["available"]:
        return {
            **paired,
            "artifact_type": "paired_excess_human_error_reference",
            "scope": "training_guard",
        }
    g_values = np.asarray(paired.pop("g_values"), dtype=np.float64)
    judge_errors = np.asarray(paired.pop("judge_errors"), dtype=np.float64)
    human_errors = np.asarray(paired.pop("human_errors"), dtype=np.float64)
    return {
        **paired,
        "artifact_type": "paired_excess_human_error_reference",
        "scope": "training_guard",
        "value": float(g_values.mean()),
        "mean_excess_human_error": float(g_values.mean()),
        "mean_judge_absolute_error": float(judge_errors.mean()),
        "mean_human_absolute_error": float(human_errors.mean()),
        "formula": "mean(|y_pred-y_true|-|rater1-rater2|)",
        "reference_is_fixed_during_target_bootstrap": True,
    }


def paired_excess_human_error_probe(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rater_scores: np.ndarray | list[object] | tuple[object, ...] | None,
    reference: dict[str, Any] | float | None,
    tolerance: float = 0.15,
    groups: np.ndarray | None = None,
    minimum_documents: int = 4,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Small-sample harmfulness estimator specified in document section 21.

    Selection is assumed to be entirely unlabeled.  All labeled target
    documents are therefore used here, and the bootstrap resamples paired
    document-level excess-human-error values without a data split.
    """

    if float(tolerance) < 0.0:
        raise ValueError("Paired harmfulness tolerance must be non-negative")
    if int(minimum_documents) < 2:
        raise ValueError("Paired harmfulness minimum_documents must be at least two")
    if int(n_boot) < 1:
        raise ValueError("Paired harmfulness n_boot must be positive")
    if not 0.5 < float(confidence) < 1.0:
        raise ValueError("Paired harmfulness confidence must be in (0.5, 1)")

    reference_value: float | None
    reference_metadata: dict[str, Any]
    if isinstance(reference, dict):
        reference_metadata = dict(reference)
        raw_reference = reference.get("value")
        reference_value = (
            float(raw_reference)
            if raw_reference is not None and np.isfinite(raw_reference)
            else None
        )
    elif reference is not None and np.isfinite(reference):
        reference_value = float(reference)
        reference_metadata = {
            "available": True,
            "value": reference_value,
            "scope": "precomputed_fixed_reference",
        }
    else:
        reference_value = None
        reference_metadata = {"available": False, "reason": "reference_not_provided"}
    if reference_value is None or not bool(reference_metadata.get("available", True)):
        return _paired_uncertain_result(
            reason="paired_excess_human_error_reference_unavailable",
            tolerance=float(tolerance),
            minimum_documents=int(minimum_documents),
            n_boot=int(n_boot),
            reference=reference_metadata,
        )

    paired = _paired_document_errors(
        y_true=y_true,
        y_pred=y_pred,
        rater_scores=rater_scores,
        groups=groups,
    )
    if not paired["available"]:
        return _paired_uncertain_result(
            reason=str(paired.get("reason", "paired_rater_scores_unavailable")),
            tolerance=float(tolerance),
            minimum_documents=int(minimum_documents),
            n_boot=int(n_boot),
            reference=reference_metadata,
            paired=paired,
        )
    g_values = np.asarray(paired.pop("g_values"), dtype=np.float64)
    judge_errors = np.asarray(paired.pop("judge_errors"), dtype=np.float64)
    human_errors = np.asarray(paired.pop("human_errors"), dtype=np.float64)
    if len(g_values) < int(minimum_documents):
        return _paired_uncertain_result(
            reason="insufficient_independent_probe_documents",
            tolerance=float(tolerance),
            minimum_documents=int(minimum_documents),
            n_boot=int(n_boot),
            reference=reference_metadata,
            paired={**paired, "document_count": int(len(g_values))},
        )

    target_mean = float(g_values.mean())
    harm_estimate = float(target_mean - reference_value)
    rng = np.random.default_rng(int(seed))
    sample_indices = rng.integers(0, len(g_values), size=(int(n_boot), len(g_values)))
    target_draws = g_values[sample_indices].mean(axis=1)
    harm_draws = target_draws - float(reference_value)
    alpha = 1.0 - float(confidence)
    centered = harm_draws - harm_estimate
    harm_lower = float(harm_estimate - np.quantile(centered, float(confidence)))
    harm_upper = float(harm_estimate - np.quantile(centered, alpha))
    if harm_estimate > float(tolerance) and harm_lower > float(tolerance):
        status = "harmful"
    elif harm_upper <= float(tolerance):
        status = "benign"
    else:
        status = "uncertain"
    p_value = float(
        (1 + np.sum(centered >= harm_estimate - float(tolerance)))
        / (1 + len(centered))
    )
    return {
        "status": status,
        "reason": "paired_excess_human_error_estimated",
        "mode": "paired_excess_human_error",
        "metric_name": "paired_excess_human_error",
        "metric": target_mean,
        "reference_metric": float(reference_value),
        "tolerance": float(tolerance),
        "threshold": float(tolerance),
        "harm_delta": harm_estimate,
        "harm_delta_lcb": harm_lower,
        "harm_delta_ucb": harm_upper,
        "harmfulness_p_value": p_value,
        "harmfulness_p_value_method": "centered_paired_document_bootstrap_one_sided",
        "n_probe": int(len(g_values)),
        "n_probe_rows": int(paired["row_count"]),
        "paired_errors": {
            "mean_judge_absolute_error": float(judge_errors.mean()),
            "mean_human_absolute_error": float(human_errors.mean()),
            "mean_excess_human_error": target_mean,
            "reference_mean_excess_human_error": float(reference_value),
            "formula_e_j": "abs(y_pred-y_true)",
            "formula_e_h": "abs(rater1-rater2)",
            "formula_g": "e_j-e_h",
            "formula_delta": "mean(g_target)-mean(g_training_guard)",
        },
        "reference": reference_metadata,
        "selection": {
            "used_labels": False,
            "data_split_required": False,
            "all_probe_labels_used_for_estimation": True,
            "rationale": "cluster_selection_and_warning_are_unlabeled",
        },
        "bootstrap": {
            "scheme": "paired_document",
            "samples": int(n_boot),
            "confidence": float(confidence),
            "interval": "basic_one_sided_bounds_from_centered_bootstrap",
            "reference_resampled": False,
        },
        "identifiability": {
            "identifiable": True,
            "reason": "paired_document_errors_available",
            "minimum_documents": int(minimum_documents),
            "independent_documents": int(len(g_values)),
        },
    }


def estimate_human_ceiling(
    rater_scores: np.ndarray | list[object] | tuple[object, ...] | None,
    *,
    metric_name: str,
    query_ids: np.ndarray | None = None,
    class_values: np.ndarray | None = None,
) -> dict[str, Any]:
    """Estimate a domain's human ceiling from per-item, ordered rater scores.

    For higher-is-better metrics this is an inter-rater ceiling. For error
    metrics it is the corresponding inter-rater error floor. Rows without at
    least two numeric rater scores are excluded.
    """

    if rater_scores is None:
        return {"available": False, "reason": "rater_scores_not_provided", "value": None}
    rows = list(np.asarray(rater_scores, dtype=object).tolist())
    queries = np.asarray(query_ids).astype(str) if query_ids is not None else None
    if queries is not None and len(queries) != len(rows):
        raise ValueError("query_ids must align with rater_scores")
    normalized: list[list[float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple, np.ndarray)):
            normalized.append([])
            continue
        values: list[float] = []
        for value in row:
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(parsed):
                values.append(parsed)
        normalized.append(values)
    max_raters = max((len(row) for row in normalized), default=0)
    pair_values: list[float] = []
    for left in range(max_raters):
        for right in range(left + 1, max_raters):
            indices = np.asarray(
                [index for index, row in enumerate(normalized) if len(row) > right], dtype=int
            )
            if indices.size < 2:
                continue
            true = np.asarray([normalized[index][left] for index in indices])
            pred = np.asarray([normalized[index][right] for index in indices])
            if queries is None:
                metrics = judge_metrics(true, pred, class_values=class_values)
            else:
                metrics = macro_query_judge_metrics(
                    true,
                    pred,
                    queries[indices],
                    class_values=class_values,
                )["macro"]
            value = metrics.get(str(metric_name))
            if value is not None and np.isfinite(value):
                pair_values.append(float(value))
    if not pair_values:
        return {"available": False, "reason": "insufficient_aligned_rater_scores", "value": None}
    return {
        "available": True,
        "value": float(np.mean(pair_values)),
        "metric_name": str(metric_name),
        "pair_count": int(len(pair_values)),
        "rater_count_max": int(max_raters),
        "direction": "lower_is_better" if metric_name in LOWER_IS_BETTER_METRICS else "higher_is_better",
    }


def harmfulness_probe(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    reference_metric: float,
    tolerance: float = 0.05,
    metric_name: str = "qwk",
    query_ids: np.ndarray | None = None,
    probabilities: np.ndarray | None = None,
    class_values: np.ndarray | None = None,
    groups: np.ndarray | None = None,
    reference_human_ceiling: float | None = None,
    target_human_ceiling: float | None = None,
    require_human_ceiling: bool = False,
    minimum_documents: int = 4,
    minimum_documents_per_query: int = 2,
    expected_query_ids: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    queries = np.asarray(query_ids).astype(str) if query_ids is not None else None
    probabilities = np.asarray(probabilities) if probabilities is not None else None
    if queries is None:
        metrics = judge_metrics(
            y_true,
            y_pred,
            probabilities=probabilities,
            class_values=class_values,
        )
    else:
        metrics = macro_query_judge_metrics(
            y_true,
            y_pred,
            queries,
            probabilities=probabilities,
            class_values=class_values,
        )["macro"]
    identifiability = _probe_identifiability(
        y_true=np.asarray(y_true),
        metric_name=metric_name,
        query_ids=queries,
        groups=groups,
        minimum_documents=int(minimum_documents),
        minimum_documents_per_query=int(minimum_documents_per_query),
        expected_query_ids=expected_query_ids,
    )
    if not identifiability["identifiable"]:
        return {
            "status": "uncertain",
            "reason": identifiability["reason"],
            "metric_name": metric_name,
            "metric": metrics.get(metric_name),
            "reference_metric": float(reference_metric),
            "tolerance": float(tolerance),
            "n_probe": int(len(y_true)),
            "metrics": {key: float(value) for key, value in metrics.items() if np.isscalar(value)},
            "identifiability": identifiability,
            "human_ceiling": {
                "mode": "not_evaluated_until_probe_is_identifiable",
                "reference": reference_human_ceiling,
                "target": target_human_ceiling,
            },
            "harmfulness_p_value": None,
            "bootstrap": {
                "scheme": "article_group" if groups is not None else "query_label_stratified_rows",
                "samples": 0,
            },
        }
    current, bootstrap_values = bootstrap_judge_metric_draws(
        y_true,
        y_pred,
        metric_name=metric_name,
        query_ids=queries,
        probabilities=probabilities,
        class_values=class_values,
        groups=groups,
        rng=np.random.default_rng(int(seed)),
        n_boot=int(n_boot),
    )
    lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
    has_human_ceiling = (
        reference_human_ceiling is not None
        and target_human_ceiling is not None
        and np.isfinite(reference_human_ceiling)
        and np.isfinite(target_human_ceiling)
    )
    if require_human_ceiling and not has_human_ceiling:
        return {
            "status": "uncertain",
            "reason": "human_ceiling_required_but_unavailable",
            "metric_name": metric_name,
            "metric": current,
            "reference_metric": float(reference_metric),
            "tolerance": float(tolerance),
            "n_probe": int(len(y_true)),
            "metrics": {key: float(value) for key, value in metrics.items() if np.isscalar(value)},
            "human_ceiling": {
                "mode": "required_but_unavailable",
                "reference": reference_human_ceiling,
                "target": target_human_ceiling,
            },
            "bootstrap": {"scheme": "article_group" if groups is not None else "query_label_stratified_rows", "samples": int(len(bootstrap_values))},
        }
    if has_human_ceiling:
        reference_gap = (
            float(reference_metric) - float(reference_human_ceiling)
            if lower_is_better
            else float(reference_human_ceiling) - float(reference_metric)
        )
        current_gap = (
            float(current) - float(target_human_ceiling)
            if lower_is_better
            else float(target_human_ceiling) - float(current)
        )
        harm_values = (
            bootstrap_values - float(target_human_ceiling) - reference_gap
            if lower_is_better
            else float(target_human_ceiling) - bootstrap_values - reference_gap
        )
        harm_estimate = float(current_gap - reference_gap)
        mode = "relative_human_ceiling"
        threshold = float(tolerance)
    else:
        harm_values = (
            bootstrap_values - float(reference_metric)
            if lower_is_better
            else float(reference_metric) - bootstrap_values
        )
        harm_estimate = (
            float(current) - float(reference_metric)
            if lower_is_better
            else float(reference_metric) - float(current)
        )
        mode = "raw_reference_metric_fallback"
        threshold = float(tolerance)
    if not harm_values.size:
        status = "uncertain"
        harm_lower = float("nan")
        harm_upper = float("nan")
    else:
        harm_lower = float(np.quantile(harm_values, 0.025))
        harm_upper = float(np.quantile(harm_values, 0.975))
        if harm_estimate > threshold and harm_lower > threshold:
            status = "harmful"
        elif harm_upper <= threshold:
            status = "benign"
        else:
            status = "uncertain"
    if harm_values.size:
        centered_harm = harm_values - harm_estimate
        harmfulness_p_value = float(
            (1 + np.sum(centered_harm >= harm_estimate - threshold))
            / (1 + len(centered_harm))
        )
    else:
        harmfulness_p_value = None
    metric_lower = float(np.quantile(bootstrap_values, 0.025)) if bootstrap_values.size else float("nan")
    metric_upper = float(np.quantile(bootstrap_values, 0.975)) if bootstrap_values.size else float("nan")
    return {
        "status": status,
        "metric_name": metric_name,
        "metric": current,
        "reference_metric": float(reference_metric),
        "tolerance": float(tolerance),
        "threshold": threshold,
        "metric_direction": "lower_is_better" if lower_is_better else "higher_is_better",
        "bootstrap_lcb": metric_lower,
        "bootstrap_ucb": metric_upper,
        "harm_delta": harm_estimate,
        "harm_delta_lcb": harm_lower,
        "harm_delta_ucb": harm_upper,
        "harmfulness_p_value": harmfulness_p_value,
        "harmfulness_p_value_method": "centered_bootstrap_one_sided",
        "n_probe": int(len(y_true)),
        "metrics": {key: float(value) for key, value in metrics.items() if np.isscalar(value)},
        "human_ceiling": {
            "mode": mode,
            "reference": float(reference_human_ceiling) if has_human_ceiling else None,
            "target": float(target_human_ceiling) if has_human_ceiling else None,
        },
        "bootstrap": {
            "scheme": "article_group" if groups is not None else "query_label_stratified_rows",
            "samples": int(len(bootstrap_values)),
        },
        "identifiability": identifiability,
    }


def _probe_identifiability(
    *,
    y_true: np.ndarray,
    metric_name: str,
    query_ids: np.ndarray | None,
    groups: np.ndarray | None,
    minimum_documents: int,
    minimum_documents_per_query: int,
    expected_query_ids: np.ndarray | None,
) -> dict[str, Any]:
    if int(minimum_documents) < 2 or int(minimum_documents_per_query) < 2:
        raise ValueError("Probe document minimums must both be at least two")
    labels = np.asarray(y_true)
    group_values = (
        np.asarray(groups).astype(str)
        if groups is not None
        else np.asarray([f"row-{index}" for index in range(len(labels))], dtype=str)
    )
    if len(group_values) != len(labels):
        raise ValueError("Probe groups must align with labels")
    document_count = int(len(np.unique(group_values)))
    details: dict[str, Any] = {
        "identifiable": False,
        "minimum_documents": int(minimum_documents),
        "minimum_documents_per_query": int(minimum_documents_per_query),
        "independent_documents": document_count,
        "documents_by_query": {},
    }
    if document_count < int(minimum_documents):
        return {**details, "reason": "insufficient_independent_probe_documents"}
    if query_ids is None:
        if str(metric_name) == "qwk" and len(np.unique(labels)) < 2:
            return {**details, "reason": "qwk_true_labels_not_identifiable"}
        return {**details, "identifiable": True, "reason": "identifiable"}

    queries = np.asarray(query_ids).astype(str)
    if len(queries) != len(labels):
        raise ValueError("Probe query IDs must align with labels")
    expected = (
        sorted(set(np.asarray(expected_query_ids).astype(str).tolist()))
        if expected_query_ids is not None
        else sorted(set(queries.tolist()))
    )
    for query_id in expected:
        mask = queries == query_id
        count = int(len(np.unique(group_values[mask])))
        details["documents_by_query"][query_id] = count
        if count < int(minimum_documents_per_query):
            return {**details, "reason": f"insufficient_probe_documents_for_query:{query_id}"}
        if str(metric_name) == "qwk" and len(np.unique(labels[mask])) < 2:
            return {**details, "reason": f"qwk_true_labels_not_identifiable_for_query:{query_id}"}
    return {**details, "identifiable": True, "reason": "identifiable"}


def _paired_document_errors(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rater_scores: np.ndarray | list[object] | tuple[object, ...] | None,
    groups: np.ndarray | None,
) -> dict[str, Any]:
    true = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if true.ndim != 1 or pred.ndim != 1 or len(true) != len(pred):
        raise ValueError("Paired harmfulness labels and predictions must be aligned vectors")
    if not np.all(np.isfinite(true)) or not np.all(np.isfinite(pred)):
        raise ValueError("Paired harmfulness labels and predictions must be finite")
    if rater_scores is None:
        return {
            "available": False,
            "reason": "paired_rater_scores_not_provided",
            "row_count": int(len(true)),
            "valid_rater_rows": 0,
        }
    rows = list(np.asarray(rater_scores, dtype=object).tolist())
    if len(rows) != len(true):
        raise ValueError("Paired harmfulness rater scores must align with labels")
    group_values = (
        np.asarray(groups).astype(str)
        if groups is not None
        else np.asarray([f"row-{index}" for index in range(len(true))], dtype=str)
    )
    if len(group_values) != len(true):
        raise ValueError("Paired harmfulness groups must align with labels")

    by_group: dict[str, dict[str, list[float]]] = {}
    invalid_rows: list[int] = []
    for index, row in enumerate(rows):
        if not isinstance(row, (list, tuple, np.ndarray)) or len(row) < 2:
            invalid_rows.append(int(index))
            continue
        try:
            rater1 = float(row[0])
            rater2 = float(row[1])
        except (TypeError, ValueError):
            invalid_rows.append(int(index))
            continue
        if not np.isfinite(rater1) or not np.isfinite(rater2):
            invalid_rows.append(int(index))
            continue
        judge_error = float(abs(pred[index] - true[index]))
        human_error = float(abs(rater1 - rater2))
        bucket = by_group.setdefault(
            str(group_values[index]), {"judge": [], "human": [], "g": []}
        )
        bucket["judge"].append(judge_error)
        bucket["human"].append(human_error)
        bucket["g"].append(judge_error - human_error)
    if invalid_rows:
        return {
            "available": False,
            "reason": "paired_rater_scores_missing_or_invalid",
            "row_count": int(len(true)),
            "valid_rater_rows": int(len(true) - len(invalid_rows)),
            "invalid_rater_rows": int(len(invalid_rows)),
            "invalid_rater_row_indices": invalid_rows[:20],
        }
    if not by_group:
        return {
            "available": False,
            "reason": "no_paired_document_errors",
            "row_count": int(len(true)),
            "valid_rater_rows": 0,
        }
    group_ids = sorted(by_group)
    judge_errors = np.asarray(
        [np.mean(by_group[group_id]["judge"]) for group_id in group_ids], dtype=np.float64
    )
    human_errors = np.asarray(
        [np.mean(by_group[group_id]["human"]) for group_id in group_ids], dtype=np.float64
    )
    g_values = np.asarray(
        [np.mean(by_group[group_id]["g"]) for group_id in group_ids], dtype=np.float64
    )
    return {
        "available": True,
        "reason": "paired_document_errors_available",
        "row_count": int(len(true)),
        "valid_rater_rows": int(len(true)),
        "document_count": int(len(group_ids)),
        "grouping": "input_document" if groups is not None else "row_as_document",
        "rater_pair": "first_two_ordered_raters",
        "g_values": g_values,
        "judge_errors": judge_errors,
        "human_errors": human_errors,
    }


def _paired_uncertain_result(
    *,
    reason: str,
    tolerance: float,
    minimum_documents: int,
    n_boot: int,
    reference: dict[str, Any],
    paired: dict[str, Any] | None = None,
) -> dict[str, Any]:
    details = dict(paired or {})
    for key in ("g_values", "judge_errors", "human_errors"):
        details.pop(key, None)
    return {
        "status": "uncertain",
        "reason": str(reason),
        "mode": "paired_excess_human_error",
        "metric_name": "paired_excess_human_error",
        "metric": None,
        "reference_metric": reference.get("value"),
        "tolerance": float(tolerance),
        "threshold": float(tolerance),
        "harm_delta": None,
        "harm_delta_lcb": None,
        "harm_delta_ucb": None,
        "harmfulness_p_value": None,
        "n_probe": int(details.get("document_count", 0)),
        "reference": reference,
        "paired_errors": details,
        "selection": {
            "used_labels": False,
            "data_split_required": False,
            "all_probe_labels_used_for_estimation": True,
        },
        "bootstrap": {
            "scheme": "paired_document",
            "samples": 0,
            "requested_samples": int(n_boot),
            "interval": "one_sided_lower_bound",
            "reference_resampled": False,
        },
        "identifiability": {
            "identifiable": False,
            "reason": str(reason),
            "minimum_documents": int(minimum_documents),
            "independent_documents": int(details.get("document_count", 0)),
        },
    }
