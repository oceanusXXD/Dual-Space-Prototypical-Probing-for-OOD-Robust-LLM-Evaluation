from __future__ import annotations

import numpy as np


def pre_answer_token_positions(attention_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(attention_mask)
    if mask.ndim != 2:
        raise ValueError("attention_mask must have shape [N, T]")
    return np.maximum(mask.sum(axis=1).astype(int) - 1, 0)


def gather_pre_answer_hidden(hidden: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    values = np.asarray(hidden, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("hidden must have shape [N, T, D]")
    positions = pre_answer_token_positions(attention_mask)
    return values[np.arange(values.shape[0]), positions].astype(np.float32)


pre_score_hidden = gather_pre_answer_hidden
pre_label_hidden = gather_pre_answer_hidden
