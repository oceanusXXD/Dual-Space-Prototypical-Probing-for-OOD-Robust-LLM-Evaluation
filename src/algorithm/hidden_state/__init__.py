"""Hidden-state extraction contracts and helpers."""

from __future__ import annotations

from src.algorithm.hidden_state.contract import HiddenStateCacheMetadata
from src.algorithm.hidden_state.layer_selection import LayerSelection, resolve_layers
from src.algorithm.hidden_state.pooling import masked_mean
from src.algorithm.hidden_state.views import HiddenStateView, resolve_view_texts

__all__ = [
    "HiddenStateCacheMetadata",
    "HiddenStateView",
    "LayerSelection",
    "masked_mean",
    "resolve_layers",
    "resolve_view_texts",
]
