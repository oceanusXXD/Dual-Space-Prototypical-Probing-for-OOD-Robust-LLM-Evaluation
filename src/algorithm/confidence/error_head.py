from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class ErrorProbabilityHead:
    def __init__(self, *, c: float = 1.0, max_iter: int = 500, seed: int = 42) -> None:
        self.model = LogisticRegression(C=float(c), max_iter=int(max_iter), random_state=int(seed))

    def fit(self, features: np.ndarray, errors: np.ndarray) -> "ErrorProbabilityHead":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(errors, dtype=int)
        if x.ndim != 2 or y.shape != (x.shape[0],):
            raise ValueError("features and errors must align")
        self.model.fit(x, y)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(features, dtype=np.float64))[:, -1]
