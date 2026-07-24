from __future__ import annotations

from typing import Any

import numpy as np

from src.algorithm.classifier.output import JudgeHeadOutput


def predict_output(model: Any, features: np.ndarray, query_ids: np.ndarray) -> JudgeHeadOutput:
    """Return the standard classifier output contract for a fitted Judge head."""

    if not hasattr(model, "predict_output"):
        raise TypeError("model must expose predict_output(features, query_ids)")
    return model.predict_output(features, query_ids)


def predict_labels(model: Any, features: np.ndarray, query_ids: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict"):
        return np.asarray(model.predict(features, query_ids))
    output = predict_output(model, features, query_ids)
    return output.classes[np.argmax(output.probabilities, axis=1)]


__all__ = ["predict_labels", "predict_output"]
