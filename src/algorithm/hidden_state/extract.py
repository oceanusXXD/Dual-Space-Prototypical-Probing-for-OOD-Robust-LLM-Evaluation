from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.algorithm.hidden_state.cache import save_cache
from src.algorithm.hidden_state.contract import HiddenStateCacheMetadata
from src.algorithm.hidden_state.qwen_hidden import extract_qwen_hidden_features
from src.algorithm.hidden_state.views import get_view, resolve_view_texts
from src.common.qwen import load_qwen_model
from src.common.schema import load_judge_records


def extract_hidden_states(
    *,
    records: list[str | Path],
    output: str | Path,
    model_path: str | Path,
    space: str,
    view: str,
    layers: tuple[int, ...],
    pooling: str | None = None,
    revision: str | None = None,
    model_id: str | None = None,
    batch_size: int = 1,
    max_length: int = 2048,
    device: str = "cuda",
    torch_dtype: str = "auto",
    attn_implementation: str = "sdpa",
    local_files_only: bool = False,
) -> dict[str, Any]:
    rows = load_judge_records(records)
    view_spec = get_view(view)
    if str(space).lower() != view_spec.space:
        raise ValueError(f"view {view!r} belongs to space {view_spec.space!r}")
    resolved_pooling = str(pooling or view_spec.pooling)
    texts = resolve_view_texts(rows, view)
    tokenizer, model, resolved_device = load_qwen_model(
        model_path,
        revision=revision,
        device=device,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
        local_files_only=local_files_only,
    )
    features = extract_qwen_hidden_features(
        tokenizer=tokenizer,
        model=model,
        texts=texts,
        layers=layers,
        pooling=resolved_pooling,
        batch_size=batch_size,
        max_length=max_length,
        device=resolved_device,
    )
    metadata = HiddenStateCacheMetadata(
        space=str(space).lower(),
        feature_scope=view_spec.feature_scope,
        layers=tuple(int(value) for value in layers),
        pooling=resolved_pooling,
        model_id=str(model_id or model_path),
        revision=revision,
        prompt_template=None,
        max_length=int(max_length),
        view=view,
    ).to_dict()
    save_cache(
        output,
        features=features,
        sample_ids=np.asarray([record.sample_id for record in rows]),
        labels=np.asarray([record.label for record in rows], dtype=object),
        query_ids=np.asarray([record.query_id for record in rows]),
        input_document_ids=np.asarray([record.input_document_id for record in rows]),
        metadata=metadata,
    )
    return {**metadata, "rows": len(rows), "shape": list(features.shape), "output": str(output)}
