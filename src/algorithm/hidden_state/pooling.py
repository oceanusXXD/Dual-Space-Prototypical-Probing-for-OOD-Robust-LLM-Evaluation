from __future__ import annotations

import numpy as np


def masked_mean(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float32)
    weights = np.asarray(mask, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("hidden must have shape [N, T, D]")
    if weights.shape != values.shape[:2]:
        raise ValueError("mask must have shape [N, T]")
    numerator = (values * weights[:, :, None]).sum(axis=1)
    denominator = weights.sum(axis=1, keepdims=True).clip(min=1.0)
    return (numerator / denominator).astype(np.float32)


def last_token(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float32)
    weights = np.asarray(mask)
    if values.ndim != 3 or weights.shape != values.shape[:2]:
        raise ValueError("hidden and mask must align as [N, T, D] and [N, T]")
    positions = np.maximum(weights.sum(axis=1).astype(int) - 1, 0)
    return values[np.arange(values.shape[0]), positions].astype(np.float32)


def span_mean(hidden: np.ndarray, spans: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float32)
    ranges = np.asarray(spans, dtype=int)
    if values.ndim != 3 or ranges.shape != (values.shape[0], 2):
        raise ValueError("hidden must be [N, T, D] and spans must be [N, 2]")
    rows: list[np.ndarray] = []
    for index, (start, stop) in enumerate(ranges.tolist()):
        lo = max(0, int(start))
        hi = min(values.shape[1], int(stop))
        if hi <= lo:
            rows.append(values[index, max(0, min(values.shape[1] - 1, lo))])
        else:
            rows.append(values[index, lo:hi].mean(axis=0))
    return np.stack(rows).astype(np.float32)
