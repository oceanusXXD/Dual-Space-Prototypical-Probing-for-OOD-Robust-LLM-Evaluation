from __future__ import annotations

from typing import Protocol

import numpy as np


class Detector(Protocol):
    def fit(self, features: np.ndarray, *args, **kwargs):
        ...

    def score(self, features: np.ndarray, *args, **kwargs) -> np.ndarray:
        ...
