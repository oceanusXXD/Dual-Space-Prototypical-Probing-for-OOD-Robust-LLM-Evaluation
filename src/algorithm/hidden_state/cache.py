from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.common.feature_store import HiddenFeatureStore, load_hidden_feature_store, save_hidden_feature_store


def load_cache(path: str | Path) -> HiddenFeatureStore:
    return load_hidden_feature_store(path)


def save_cache(
    path: str | Path,
    *,
    features: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray | None = None,
    query_ids: np.ndarray | None = None,
    input_document_ids: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    return save_hidden_feature_store(
        path,
        features=features,
        sample_ids=sample_ids,
        labels=labels,
        query_ids=query_ids,
        input_document_ids=input_document_ids,
        metadata=metadata,
    )
