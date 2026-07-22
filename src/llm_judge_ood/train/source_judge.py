from __future__ import annotations

import numpy as np

from src.llm_judge_ood.model.judge import JudgeTrainingConfig, SharedBackboneJudge
from src.llm_judge_ood.shared.metrics import normalize_label_array


def train_source_judge(
    *,
    features: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    config: JudgeTrainingConfig | None = None,
) -> SharedBackboneJudge:
    """Train the shared backbone plus per-query heads with the documented contract."""

    return SharedBackboneJudge(config or JudgeTrainingConfig()).fit(
        features,
        normalize_label_array(labels),
        np.asarray(query_ids).astype(str),
        train_mask=np.asarray(train_mask, dtype=bool),
        validation_mask=np.asarray(validation_mask, dtype=bool),
    )

