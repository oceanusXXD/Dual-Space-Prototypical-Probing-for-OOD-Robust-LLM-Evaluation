from __future__ import annotations

import numpy as np


def max_softmax_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [N, K]")
    return (1.0 - np.max(probs, axis=1)).astype(np.float64)


def entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape [N, K]")
    entropy = -(probs * np.log(probs)).sum(axis=1)
    return (entropy / max(np.log(probs.shape[1]), 1e-12)).astype(np.float64)
