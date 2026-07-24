from __future__ import annotations

import numpy as np


def weighted_score_fusion(scores: dict[str, np.ndarray], weights: dict[str, float] | None = None) -> np.ndarray:
    if not scores:
        raise ValueError("at least one score vector is required")
    names = sorted(scores)
    arrays = [np.asarray(scores[name], dtype=np.float64) for name in names]
    length = arrays[0].shape
    if any(array.shape != length for array in arrays):
        raise ValueError("all score vectors must have the same shape")
    raw_weights = np.asarray(
        [float((weights or {}).get(name, 1.0)) for name in names],
        dtype=np.float64,
    )
    if np.any(raw_weights < 0.0) or raw_weights.sum() <= 0.0:
        raise ValueError("fusion weights must be non-negative and not all zero")
    normalized = raw_weights / raw_weights.sum()
    return np.sum([weight * array for weight, array in zip(normalized, arrays, strict=True)], axis=0)
