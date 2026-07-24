from __future__ import annotations

import numpy as np


def empirical_tail_probability(reference_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference_scores, dtype=np.float64))
    values = np.asarray(scores, dtype=np.float64)
    if reference.ndim != 1 or values.ndim != 1 or reference.size == 0:
        raise ValueError("reference_scores and scores must be one-dimensional and non-empty")
    ranks = np.searchsorted(reference, values, side="right")
    return (1.0 - ranks / float(reference.size)).astype(np.float64)
