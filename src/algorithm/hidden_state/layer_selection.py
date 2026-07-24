from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LayerSelection:
    mode: str = "layer_index"
    layers: tuple[int, ...] = (-1,)
    pca_dim: int | None = None

    @classmethod
    def from_layers(cls, layers: Iterable[int]) -> "LayerSelection":
        return cls(mode="layer_index", layers=tuple(int(value) for value in layers))


def resolve_layers(num_hidden_states: int, requested: Iterable[int] | None) -> list[int]:
    count = int(num_hidden_states)
    if count < 2:
        raise ValueError("num_hidden_states must include embeddings plus transformer states")
    values = list(requested or [-1])
    resolved: list[int] = []
    for value in values:
        raw = int(value)
        index = raw if raw >= 0 else count + raw
        if index <= 0 or index >= count:
            raise ValueError(
                f"layer {raw} resolves to {index}; valid transformer states are 1..{count - 1}"
            )
        if index in resolved:
            raise ValueError(f"duplicate resolved layer: {index}")
        resolved.append(index)
    return resolved


def select_layer_matrix(hidden_states: np.ndarray, selection: LayerSelection) -> np.ndarray:
    values = np.asarray(hidden_states)
    if values.ndim != 3:
        raise ValueError("hidden_states must have shape [N, L, D]")
    mode = str(selection.mode).lower()
    if mode in {"last_layer", "layer_index"}:
        indices = resolve_layers(values.shape[1], selection.layers)
        return values[:, indices, :]
    if mode == "all_layers":
        return values[:, 1:, :]
    if mode == "layer_mean":
        return values[:, 1:, :].mean(axis=1, keepdims=True)
    raise ValueError(f"unsupported layer selection mode: {selection.mode}")
