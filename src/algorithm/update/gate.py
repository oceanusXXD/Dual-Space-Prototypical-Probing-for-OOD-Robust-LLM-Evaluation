from __future__ import annotations

import numpy as np


def threshold_gate(scores: np.ndarray, threshold: float, *, accept_below: bool = True) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    return values <= float(threshold) if accept_below else values >= float(threshold)
