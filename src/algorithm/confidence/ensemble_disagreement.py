from __future__ import annotations

import numpy as np


def vote_disagreement(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions)
    if values.ndim != 2:
        raise ValueError("predictions must have shape [N, M]")
    out = []
    for row in values:
        _, counts = np.unique(row, return_counts=True)
        out.append(1.0 - float(counts.max()) / float(row.size))
    return np.asarray(out, dtype=np.float64)
