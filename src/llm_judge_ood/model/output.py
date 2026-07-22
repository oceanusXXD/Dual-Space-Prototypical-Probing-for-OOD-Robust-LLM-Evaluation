from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JudgeHeadOutput:
    """Stable classifier-head contract used by B-space OOD detectors."""

    penultimate: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    classes: np.ndarray

    def __post_init__(self) -> None:
        h = np.asarray(self.penultimate)
        logits = np.asarray(self.logits)
        probabilities = np.asarray(self.probabilities)
        classes = np.asarray(self.classes)
        if h.ndim != 2 or logits.ndim != 2 or probabilities.ndim != 2:
            raise ValueError("Judge head outputs must be two-dimensional matrices")
        if len(h) != len(logits) or len(h) != len(probabilities):
            raise ValueError("Judge head output rows must align")
        if logits.shape[1] != len(classes) or probabilities.shape != logits.shape:
            raise ValueError("Judge logits, probabilities, and class vocabulary must align")
        if not np.isfinite(h).all() or not np.isfinite(logits).all() or not np.isfinite(probabilities).all():
            raise ValueError("Judge head outputs contain non-finite values")
        if np.any(probabilities < 0.0) or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Judge probabilities must be normalized")


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits must have shape [N, K]")
    maxima = np.max(values, axis=1, keepdims=True)
    shifted = values - maxima
    numerator = np.exp(shifted)
    return (numerator / numerator.sum(axis=1, keepdims=True)).astype(np.float32)
