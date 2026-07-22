from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def normalize_label_array(labels: Any) -> np.ndarray:
    values = np.asarray(labels, dtype=object)
    converted: list[float] = []
    for value in values:
        try:
            converted.append(float(value))
        except (TypeError, ValueError):
            return values.astype(str)
    numeric = np.asarray(converted, dtype=np.float64)
    if np.all(np.isfinite(numeric)) and np.all(numeric == np.floor(numeric)):
        return numeric.astype(np.int64)
    return numeric


def judge_metrics(
    y_true: Any,
    y_pred: Any,
    *,
    probabilities: np.ndarray | None = None,
    class_values: Any | None = None,
) -> dict[str, float]:
    true = normalize_label_array(y_true)
    pred = normalize_label_array(y_pred)
    out = {
        "accuracy": float(accuracy_score(_class_labels(true), _class_labels(pred))),
        "mae": float(np.mean(np.abs(true.astype(float) - pred.astype(float)))) if _numeric(true, pred) else float("nan"),
    }
    if _numeric(true, pred):
        if class_values is not None:
            scale_values = normalize_label_array(class_values).astype(float)
        else:
            scale_values = np.concatenate([true.astype(float), pred.astype(float)])
        scale = float(np.max(scale_values) - np.min(scale_values)) if scale_values.size else 0.0
        out["normalized_mae"] = float(out["mae"] / scale) if scale > 0.0 else 0.0
    else:
        out["normalized_mae"] = float("nan")
    try:
        if len(np.unique(true)) < 2 and len(np.unique(pred)) < 2:
            out["qwk"] = 0.0
        else:
            qwk = float(_quadratic_weighted_kappa(true, pred, class_values=class_values))
            out["qwk"] = 0.0 if not np.isfinite(qwk) else qwk
    except Exception:
        out["qwk"] = 0.0
    try:
        corr = _spearman(true.astype(float), pred.astype(float))
        out["spearman"] = float(0.0 if corr is None or np.isnan(corr) else corr)
    except Exception:
        out["spearman"] = float("nan")
    if probabilities is not None:
        probability_values = np.asarray(probabilities, dtype=np.float64)
        out["mean_confidence"] = float(np.max(probability_values, axis=1).mean())
        if class_values is not None:
            out["ordinal_log_loss"] = ordinal_log_loss(
                true,
                probability_values,
                class_values=class_values,
            )
    return out


def macro_query_judge_metrics(
    y_true: Any,
    y_pred: Any,
    query_ids: Any,
    *,
    probabilities: np.ndarray | None = None,
    class_values: Any | None = None,
) -> dict[str, Any]:
    true = normalize_label_array(y_true)
    pred = normalize_label_array(y_pred)
    queries = np.asarray(query_ids).astype(str)
    if len(true) != len(pred) or len(true) != len(queries):
        raise ValueError("labels, predictions, and query_ids must be aligned")
    by_query: dict[str, dict[str, float]] = {}
    for query_id in sorted(set(queries.tolist())):
        mask = queries == query_id
        by_query[query_id] = judge_metrics(
            true[mask],
            pred[mask],
            probabilities=np.asarray(probabilities)[mask] if probabilities is not None else None,
            class_values=class_values,
        )
    keys = ("accuracy", "mae", "normalized_mae", "qwk", "spearman")
    if probabilities is not None and class_values is not None:
        keys = (*keys, "ordinal_log_loss")
    macro = {
        key: float(np.mean([metrics[key] for metrics in by_query.values()])) if by_query else float("nan")
        for key in keys
    }
    return {"macro": macro, "by_query": by_query, "num_queries": len(by_query)}


def ordinal_log_loss(y_true: Any, probabilities: np.ndarray, *, class_values: Any) -> float:
    """Ordinal cumulative log loss for ordered class probabilities.

    For every ordinal threshold, this evaluates the Bernoulli event ``y > r``
    against the cumulative model probability.  It works for both CE and CORN
    heads once their predictions have been converted to class probabilities.
    """

    true = normalize_label_array(y_true)
    classes = normalize_label_array(class_values)
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[0] != len(true) or probs.shape[1] != len(classes):
        return float("nan")
    if len(classes) < 2 or not _numeric(true, classes):
        return float("nan")
    class_to_index = {value: index for index, value in enumerate(classes.tolist())}
    if any(value not in class_to_index for value in true.tolist()):
        return float("nan")
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    encoded = np.asarray([class_to_index[value] for value in true.tolist()], dtype=int)
    losses: list[np.ndarray] = []
    for threshold in range(len(classes) - 1):
        target = (encoded > threshold).astype(np.float64)
        survival = probs[:, threshold + 1 :].sum(axis=1)
        losses.append(-(target * np.log(survival) + (1.0 - target) * np.log1p(-survival)))
    return float(np.concatenate(losses).mean()) if losses else 0.0


def bootstrap_judge_metric_interval(
    y_true: Any,
    y_pred: Any,
    *,
    metric_name: str = "qwk",
    query_ids: Any | None = None,
    probabilities: np.ndarray | None = None,
    class_values: Any | None = None,
    groups: Any | None = None,
    rng: np.random.Generator | None = None,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return observed metric and a bootstrap confidence interval.

    Row draws are stratified by ``query_id`` and ordinal label so small Probe
    sets preserve the score/query composition.  When article-level groups are
    available, groups are resampled instead to respect paired rows.
    """

    observed, values = bootstrap_judge_metric_draws(
        y_true,
        y_pred,
        metric_name=metric_name,
        query_ids=query_ids,
        probabilities=probabilities,
        class_values=class_values,
        groups=groups,
        rng=rng,
        n_boot=n_boot,
    )
    if not values.size:
        return float(observed), float("nan"), float("nan")
    return (
        float(observed),
        float(np.quantile(values, float(alpha) / 2.0)),
        float(np.quantile(values, 1.0 - float(alpha) / 2.0)),
    )


def bootstrap_judge_metric_draws(
    y_true: Any,
    y_pred: Any,
    *,
    metric_name: str = "qwk",
    query_ids: Any | None = None,
    probabilities: np.ndarray | None = None,
    class_values: Any | None = None,
    groups: Any | None = None,
    rng: np.random.Generator | None = None,
    n_boot: int = 500,
) -> tuple[float, np.ndarray]:
    """Return an observed Judge metric and grouped bootstrap replicates."""

    true = normalize_label_array(y_true)
    pred = normalize_label_array(y_pred)
    queries = np.asarray(query_ids).astype(str) if query_ids is not None else None
    probs = np.asarray(probabilities) if probabilities is not None else None
    if len(true) != len(pred) or (queries is not None and len(queries) != len(true)):
        raise ValueError("bootstrap metric inputs must be aligned")
    if probs is not None and len(probs) != len(true):
        raise ValueError("probabilities must align with labels")
    generator = rng or np.random.default_rng(42)
    observed = _judge_metric_value(
        true,
        pred,
        metric_name=metric_name,
        query_ids=queries,
        probabilities=probs,
        class_values=class_values,
    )
    draws: list[float] = []
    index_sampler = _BootstrapIndexSampler.build(
        true,
        query_ids=queries,
        groups=groups,
    )
    for _ in range(max(1, int(n_boot))):
        indices = index_sampler.draw(generator)
        value = _judge_metric_value(
            true[indices],
            pred[indices],
            metric_name=metric_name,
            query_ids=queries[indices] if queries is not None else None,
            probabilities=probs[indices] if probs is not None else None,
            class_values=class_values,
        )
        if np.isfinite(value):
            draws.append(float(value))
    return float(observed), np.asarray(draws, dtype=np.float64)


def paired_bootstrap_metric_difference(
    y_true: Any,
    old_predictions: Any,
    new_predictions: Any,
    *,
    metric_name: str = "qwk",
    query_ids: Any | None = None,
    old_probabilities: np.ndarray | None = None,
    new_probabilities: np.ndarray | None = None,
    class_values: Any | None = None,
    groups: Any | None = None,
    rng: np.random.Generator | None = None,
    n_boot: int = 500,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired bootstrap for an update's new-minus-old metric difference."""

    true = normalize_label_array(y_true)
    old = normalize_label_array(old_predictions)
    new = normalize_label_array(new_predictions)
    queries = np.asarray(query_ids).astype(str) if query_ids is not None else None
    old_probs = np.asarray(old_probabilities) if old_probabilities is not None else None
    new_probs = np.asarray(new_probabilities) if new_probabilities is not None else None
    if len(true) != len(old) or len(true) != len(new):
        raise ValueError("paired bootstrap labels and predictions must be aligned")
    if queries is not None and len(queries) != len(true):
        raise ValueError("query_ids must align with paired bootstrap rows")
    generator = rng or np.random.default_rng(42)

    def metric(indices: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None) -> float:
        return _judge_metric_value(
            true[indices],
            predictions[indices],
            metric_name=metric_name,
            query_ids=queries[indices] if queries is not None else None,
            probabilities=probabilities[indices] if probabilities is not None else None,
            class_values=class_values,
        )

    observed_old = metric(np.arange(len(true), dtype=int), old, old_probs)
    observed_new = metric(np.arange(len(true), dtype=int), new, new_probs)
    draws: list[float] = []
    index_sampler = _BootstrapIndexSampler.build(
        true,
        query_ids=queries,
        groups=groups,
    )
    for _ in range(max(1, int(n_boot))):
        indices = index_sampler.draw(generator)
        value = metric(indices, new, new_probs) - metric(indices, old, old_probs)
        if np.isfinite(value):
            draws.append(float(value))
    values = np.asarray(draws, dtype=np.float64)
    return {
        "metric_name": str(metric_name),
        "old": float(observed_old),
        "new": float(observed_new),
        "improvement": float(observed_new - observed_old),
        "ci95": (
            [float(np.quantile(values, float(alpha) / 2.0)), float(np.quantile(values, 1.0 - float(alpha) / 2.0))]
            if values.size
            else [float("nan"), float("nan")]
        ),
        "bootstrap_samples": int(len(draws)),
    }


def _judge_metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_name: str,
    query_ids: np.ndarray | None,
    probabilities: np.ndarray | None,
    class_values: Any | None,
) -> float:
    name = str(metric_name)
    if name in {"accuracy", "mae", "normalized_mae", "qwk", "spearman", "ordinal_log_loss"}:
        if query_ids is None:
            return _single_judge_metric_value(
                y_true,
                y_pred,
                metric_name=name,
                probabilities=probabilities,
                class_values=class_values,
            )
        queries = np.asarray(query_ids).astype(str)
        values = [
            _single_judge_metric_value(
                y_true[queries == query],
                y_pred[queries == query],
                metric_name=name,
                probabilities=(
                    probabilities[queries == query] if probabilities is not None else None
                ),
                class_values=class_values,
            )
            for query in np.unique(queries)
        ]
        return float(np.mean(values)) if values else float("nan")
    if query_ids is None:
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
            query_ids,
            probabilities=probabilities,
            class_values=class_values,
        )["macro"]
    value = metrics.get(name)
    return float(value) if value is not None else float("nan")


def _single_judge_metric_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    metric_name: str,
    probabilities: np.ndarray | None,
    class_values: Any | None,
) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if metric_name == "accuracy":
        return float(np.mean(true == pred)) if len(true) else float("nan")
    if metric_name in {"mae", "normalized_mae"}:
        if not _numeric(true, pred) or not len(true):
            return float("nan")
        mae = float(np.mean(np.abs(true.astype(float) - pred.astype(float))))
        if metric_name == "mae":
            return mae
        scale_values = (
            normalize_label_array(class_values).astype(float)
            if class_values is not None
            else np.concatenate([true.astype(float), pred.astype(float)])
        )
        scale = float(np.max(scale_values) - np.min(scale_values)) if scale_values.size else 0.0
        return float(mae / scale) if scale > 0.0 else 0.0
    if metric_name == "qwk":
        return _quadratic_weighted_kappa(true, pred, class_values=class_values)
    if metric_name == "spearman":
        if not _numeric(true, pred):
            return float("nan")
        value = _spearman(true.astype(float), pred.astype(float))
        return float(0.0 if not np.isfinite(value) else value)
    if metric_name == "ordinal_log_loss":
        if probabilities is None or class_values is None:
            return float("nan")
        return ordinal_log_loss(true, probabilities, class_values=class_values)
    return float("nan")


def _quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_values: Any | None = None,
) -> float:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if len(true) == 0 or len(true) != len(pred):
        return float("nan")
    if class_values is None:
        observed = normalize_label_array(np.concatenate([true, pred]))
        if (
            observed.dtype.kind in "iu"
            and observed.size
            and int(observed.max()) - int(observed.min()) <= 1000
        ):
            labels = np.arange(int(observed.min()), int(observed.max()) + 1, dtype=observed.dtype)
            label_to_index = {value: index for index, value in enumerate(labels.tolist())}
            try:
                true_encoded = np.asarray([label_to_index[value] for value in true.tolist()], dtype=int)
                pred_encoded = np.asarray([label_to_index[value] for value in pred.tolist()], dtype=int)
            except (KeyError, TypeError):
                return float("nan")
        else:
            labels, encoded = np.unique(observed, return_inverse=True)
            true_encoded = encoded[: len(true)]
            pred_encoded = encoded[len(true) :]
    else:
        labels = normalize_label_array(class_values)
        if labels.ndim != 1 or len(labels) < 2:
            return float("nan")
        if len(set(labels.tolist())) != len(labels):
            return float("nan")
        label_to_index = {value: index for index, value in enumerate(labels.tolist())}
        try:
            true_encoded = np.asarray([label_to_index[value] for value in true.tolist()], dtype=int)
            pred_encoded = np.asarray([label_to_index[value] for value in pred.tolist()], dtype=int)
        except (KeyError, TypeError):
            return float("nan")
    if len(labels) < 2:
        return 0.0
    n_classes = len(labels)
    confusion = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(confusion, (true_encoded, pred_encoded), 1.0)
    true_histogram = confusion.sum(axis=1)
    pred_histogram = confusion.sum(axis=0)
    expected = np.outer(true_histogram, pred_histogram) / float(len(true))
    positions = np.arange(n_classes, dtype=np.float64)
    weights = np.square(positions[:, None] - positions[None, :])
    denominator = float(np.sum(weights * expected))
    if denominator <= 0.0:
        return 0.0
    value = 1.0 - float(np.sum(weights * confusion)) / denominator
    return float(value) if np.isfinite(value) else 0.0


@dataclass(frozen=True)
class _BootstrapIndexSampler:
    """Pre-index bootstrap strata so every draw avoids full-array scans."""

    grouped: bool
    values: np.ndarray
    members: tuple[np.ndarray, ...]

    @classmethod
    def build(
        cls,
        labels: np.ndarray,
        *,
        query_ids: np.ndarray | None,
        groups: Any | None,
    ) -> "_BootstrapIndexSampler":
        if len(labels) == 0:
            return cls(False, np.zeros(0, dtype=str), ())
        if groups is not None:
            group_values = np.asarray(groups).astype(str)
            if len(group_values) != len(labels):
                raise ValueError("groups must align with bootstrap labels")
            unique, members = _group_member_indices(group_values)
            return cls(True, unique, members)
        if query_ids is None:
            strata = np.asarray([str(value) for value in labels.tolist()], dtype=str)
        else:
            strata = np.asarray(
                [
                    f"{query}\u241f{label}"
                    for query, label in zip(query_ids.tolist(), labels.tolist(), strict=True)
                ],
                dtype=str,
            )
        unique, members = _group_member_indices(strata)
        return cls(False, unique, members)

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        if not self.members:
            return np.zeros(0, dtype=int)
        if self.grouped:
            sampled = rng.choice(self.values, size=len(self.values), replace=True)
            positions = np.searchsorted(self.values, sampled)
            return np.concatenate([self.members[int(position)] for position in positions]).astype(int)
        indices = np.concatenate(
            [rng.choice(member, size=len(member), replace=True) for member in self.members]
        ).astype(int)
        return indices[rng.permutation(len(indices))]


def _bootstrap_indices(
    labels: np.ndarray,
    *,
    query_ids: np.ndarray | None,
    groups: Any | None,
    rng: np.random.Generator,
) -> np.ndarray:
    return _BootstrapIndexSampler.build(
        labels,
        query_ids=query_ids,
        groups=groups,
    ).draw(rng)


def _group_member_indices(values: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Return sorted unique string values and their stable member indices."""

    groups = np.asarray(values).astype(str)
    if groups.ndim != 1:
        raise ValueError("bootstrap groups must be one-dimensional")
    if groups.size == 0:
        return np.zeros(0, dtype=str), ()
    order = np.argsort(groups, kind="stable")
    ordered = groups[order]
    starts = np.concatenate(
        [np.asarray([0], dtype=np.intp), np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    )
    stops = np.concatenate([starts[1:], np.asarray([len(order)], dtype=np.intp)])
    return (
        ordered[starts],
        tuple(order[start:stop].astype(int) for start, stop in zip(starts, stops, strict=True)),
    )


def confusion_matrix_report(y_true: Any, y_pred: Any, *, class_values: Any) -> dict[str, Any]:
    classes = normalize_label_array(class_values)
    true = normalize_label_array(y_true)
    pred = normalize_label_array(y_pred)
    matrix = confusion_matrix(
        _class_labels(true),
        _class_labels(pred),
        labels=_class_labels(classes),
    )
    return {
        "classes": [_json_scalar(value) for value in classes],
        "matrix": matrix.astype(int).tolist(),
        "true_distribution": _value_counts(true),
        "prediction_distribution": _value_counts(pred),
    }


def ood_metrics(y_true_ood: Any, scores: Any) -> dict[str, float]:
    y = np.asarray(y_true_ood, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    out: dict[str, float] = {}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, s))
        out["aupr"] = float(average_precision_score(y, s))
        out["fpr95"] = fpr_at_tpr(y, s, target_tpr=0.95)
    else:
        out.update({"auroc": float("nan"), "aupr": float("nan"), "fpr95": float("nan")})
    return out


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, *, target_tpr: float = 0.95) -> float:
    order = np.argsort(-scores, kind="stable")
    y = y_true[order].astype(bool)
    positives = max(int(y.sum()), 1)
    negatives = max(int((~y).sum()), 1)
    tp = 0
    fp = 0
    best = 1.0
    for is_pos in y:
        if is_pos:
            tp += 1
        else:
            fp += 1
        if tp / positives >= target_tpr:
            best = min(best, fp / negatives)
    return float(best)


def binary_prf(y_true: Any, y_pred: Any) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        average="binary",
        zero_division=0,
    )
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def bootstrap_interval(values: np.ndarray, *, rng: np.random.Generator, n_boot: int = 500, alpha: float = 0.05) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return (float("nan"), float("nan"))
    samples = np.empty(int(n_boot), dtype=np.float64)
    for idx in range(int(n_boot)):
        draw = rng.choice(array, size=array.size, replace=True)
        samples[idx] = float(draw.mean())
    return (float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1.0 - alpha / 2)))


def bootstrap_binary_metric_interval(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    metric: str = "auroc",
) -> tuple[float, float]:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return (float("nan"), float("nan"))
    draws: list[float] = []
    if groups is None:
        class_indices = [np.where(y == value)[0] for value in (0, 1)]
        for _ in range(int(n_boot)):
            index = np.concatenate([rng.choice(local, size=len(local), replace=True) for local in class_indices])
            draws.append(_binary_metric(metric, y[index], s[index]))
    else:
        group_values = np.asarray(groups).astype(str)
        unique_groups, group_members = _group_member_indices(group_values)
        members_by_group = {
            str(group): members
            for group, members in zip(unique_groups.tolist(), group_members, strict=True)
        }
        labels_by_group = {
            group: np.unique(y[members_by_group[str(group)]]).tolist()
            for group in unique_groups.tolist()
        }
        mixed = any(len(values) > 1 for values in labels_by_group.values())
        grouped_by_class = {
            value: np.asarray(
                [group for group, labels in labels_by_group.items() if labels == [value]],
                dtype=str,
            )
            for value in (0, 1)
        }
        for _ in range(int(n_boot)):
            if mixed or any(len(grouped_by_class[value]) == 0 for value in (0, 1)):
                sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)
            else:
                sampled_groups = np.concatenate(
                    [
                        rng.choice(grouped_by_class[value], size=len(grouped_by_class[value]), replace=True)
                        for value in (0, 1)
                    ]
                )
            indices = np.concatenate(
                [members_by_group[str(group)] for group in sampled_groups]
            )
            if len(np.unique(y[indices])) == 2:
                draws.append(_binary_metric(metric, y[indices], s[indices]))
    finite = np.asarray([value for value in draws if np.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(finite, alpha / 2)),
        float(np.quantile(finite, 1.0 - alpha / 2)),
    )


def _numeric(*arrays: np.ndarray) -> bool:
    try:
        for array in arrays:
            np.asarray(array).astype(float)
        return True
    except Exception:
        return False


def _class_labels(values: np.ndarray) -> np.ndarray:
    return np.asarray([str(_json_scalar(value)) for value in np.asarray(values).tolist()], dtype=str)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_rank = _rankdata(np.asarray(left, dtype=np.float64))
    right_rank = _rankdata(np.asarray(right, dtype=np.float64))
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denom <= 1e-12:
        return 0.0
    return float((left_centered @ right_centered) / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.shape[0]:
        end = start + 1
        while end < values.shape[0] and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _binary_metric(name: str, y: np.ndarray, scores: np.ndarray) -> float:
    if name == "auroc":
        return float(roc_auc_score(y, scores))
    if name == "aupr":
        return float(average_precision_score(y, scores))
    raise ValueError(f"Unsupported bootstrap metric: {name}")


def _value_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values, return_counts=True)
    return {str(_json_scalar(value)): int(count) for value, count in zip(unique, counts, strict=True)}


def _json_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value
