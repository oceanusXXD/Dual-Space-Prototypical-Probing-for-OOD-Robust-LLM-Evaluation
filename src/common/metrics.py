from __future__ import annotations
from typing import Any
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score

def normalize_labels(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=object)
    numeric: list[float] = []
    for value in values:
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, str):
            value = value.strip()
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            return np.asarray([str(item) for item in values])
    numeric_array = np.asarray(numeric, dtype=np.float64)
    if np.all(np.isfinite(numeric_array)) and np.all(numeric_array == np.floor(numeric_array)):
        return numeric_array.astype(np.int64)
    return numeric_array

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *, y_score: np.ndarray | None=None, positive_label: Any | None=None) -> dict[str, float]:
    metrics = {'accuracy': float(accuracy_score(y_true, y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)), 'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 'weighted_f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0))}
    labels = np.unique(y_true)
    if len(labels) == 2 and y_score is not None:
        pos = positive_label if positive_label is not None else labels[-1]
        try:
            y_binary = (np.asarray(y_true) == pos).astype(int)
            metrics['roc_auc'] = float(roc_auc_score(y_binary, np.asarray(y_score)))
        except ValueError:
            pass
    return metrics
