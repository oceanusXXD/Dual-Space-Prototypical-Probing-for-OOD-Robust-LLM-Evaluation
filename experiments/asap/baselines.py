from __future__ import annotations

from typing import Any

import numpy as np

from src.common.metrics import judge_metrics


def majority_baseline_metrics(labels: np.ndarray) -> dict[str, Any]:
    values, counts = np.unique(labels, return_counts=True)
    majority = values[int(np.argmax(counts))]
    predictions = np.asarray([majority] * len(labels))
    return {"majority_label": majority.item() if hasattr(majority, "item") else majority, **judge_metrics(labels, predictions)}
