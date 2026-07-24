from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import torch

from src.algorithm.hidden_state.layer_selection import resolve_layers
from src.algorithm.hidden_state.score_token import gather_pre_answer_hidden


def extract_qwen_hidden_features(
    *,
    tokenizer,
    model,
    texts: Sequence[str],
    layers: Iterable[int],
    pooling: str = "masked_mean",
    batch_size: int = 1,
    max_length: int = 2048,
    device: torch.device | str | None = None,
) -> np.ndarray:
    if int(batch_size) < 1 or int(max_length) < 1:
        raise ValueError("batch_size and max_length must be positive")
    resolved_device = torch.device(device) if device is not None else next(model.parameters()).device
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(texts), int(batch_size)):
            batch_texts = [str(value) for value in texts[start : start + int(batch_size)]]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=int(max_length),
                return_tensors="pt",
            )
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
            result = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden_states = result.hidden_states
            selected = resolve_layers(len(hidden_states), layers)
            attention_mask = encoded["attention_mask"].detach().cpu().numpy()
            per_layer: list[np.ndarray] = []
            for layer in selected:
                hidden = hidden_states[layer].detach().float().cpu().numpy()
                if pooling in {"pre_score_token", "pre_label_token", "pre_answer_token"}:
                    pooled = gather_pre_answer_hidden(hidden, attention_mask)
                elif pooling == "masked_mean":
                    weights = attention_mask.astype(np.float32)
                    pooled = (hidden * weights[:, :, None]).sum(axis=1) / weights.sum(
                        axis=1, keepdims=True
                    ).clip(min=1.0)
                elif pooling == "last_token":
                    positions = np.maximum(attention_mask.sum(axis=1).astype(int) - 1, 0)
                    pooled = hidden[np.arange(hidden.shape[0]), positions]
                else:
                    raise ValueError(f"unsupported pooling: {pooling}")
                per_layer.append(pooled.astype(np.float32))
            outputs.append(np.stack(per_layer, axis=1))
    return np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.empty((0, 0, 0), dtype=np.float32)
