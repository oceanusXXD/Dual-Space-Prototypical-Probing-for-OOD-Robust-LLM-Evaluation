from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class HiddenFeatureStore:
    features: np.ndarray
    sample_ids: np.ndarray
    labels: np.ndarray | None
    query_ids: np.ndarray | None
    audit_document_group_ids: np.ndarray | None
    input_document_ids: np.ndarray | None
    metadata: dict[str, Any]


def load_hidden_feature_store(path: str | Path) -> HiddenFeatureStore:
    """Load a frozen hidden-state cache using the OOD package's public contract.

    Supported feature keys:
    - `feat`, `features`, `hidden`, `hidden_states`: `[N, L, D]` or `[N, D]`
    - `X`: `[N, D]`, promoted to one layer

    Optional alignment keys include `sample_ids`/`ids`, `labels`/`y`, `query_ids`,
    `audit_document_group_ids`, and `input_document_ids`. Legacy document-group
    keys are read only as explicit compatibility fallbacks; they are audit metadata,
    not OOD inputs.
    """

    payload = np.load(Path(path), allow_pickle=True)
    feature_key = _first_existing(payload.files, ["feat", "features", "hidden", "hidden_states", "X"])
    if feature_key is None:
        raise ValueError(f"{path} has no supported feature key")
    features = np.asarray(payload[feature_key], dtype=np.float32)
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3:
        raise ValueError(f"Expected [N,L,D] or [N,D] features, got {features.shape}")
    sample_key = _first_existing(payload.files, ["sample_ids", "ids", "id"])
    sample_ids = np.asarray(payload[sample_key]).astype(str) if sample_key else np.asarray([f"row-{i}" for i in range(features.shape[0])])
    labels = _optional_array(payload, ["labels", "y", "label"])
    query_ids = _optional_array(payload, ["query_ids", "query_id"])
    audit_document_group_key = _first_existing(
        payload.files,
        ["audit_document_group_ids", "document_group_ids", "reporting_group_ids", "domain_ids", "domain_id"],
    )
    audit_document_group_ids = np.asarray(payload[audit_document_group_key]) if audit_document_group_key else None
    input_document_ids = _optional_array(payload, ["input_document_ids", "input_document_id"])
    metadata: dict[str, Any] = {
        "path": str(path),
        "feature_key": feature_key,
        "num_samples": int(features.shape[0]),
        "num_layers": int(features.shape[1]),
        "dim": int(features.shape[2]),
    }
    if "metadata_json" in payload.files:
        try:
            raw_metadata = np.asarray(payload["metadata_json"]).item()
            parsed = json.loads(str(raw_metadata))
            if isinstance(parsed, dict):
                metadata["cache_metadata"] = parsed
        except (TypeError, ValueError, json.JSONDecodeError):
            metadata["cache_metadata_parse_error"] = True
    if audit_document_group_key in {"domain_ids", "domain_id"}:
        metadata["audit_document_group_ids_source"] = "legacy_domain_ids"
    elif audit_document_group_key in {"document_group_ids", "reporting_group_ids"}:
        metadata["audit_document_group_ids_source"] = "legacy_document_group_ids"
    elif audit_document_group_key is not None:
        metadata["audit_document_group_ids_source"] = "audit_document_group_ids"
    return HiddenFeatureStore(
        features=features,
        sample_ids=sample_ids,
        labels=labels,
        query_ids=query_ids,
        audit_document_group_ids=audit_document_group_ids,
        input_document_ids=input_document_ids,
        metadata=metadata,
    )


def record_fingerprint(records: Sequence[Any], *, feature_scope: str) -> str:
    """Fingerprint the exact text rows consumed by frozen Qwen."""

    digest = hashlib.sha256()
    scope = str(feature_scope).strip().lower()
    if scope not in {"input_document", "judge_input"}:
        raise ValueError("feature_scope must be 'input_document' or 'judge_input'")
    digest.update(f"feature_scope={scope}\n".encode("utf-8"))
    if scope == "input_document":
        # A fingerprint: raw document identity + raw input_document_text.
        rows: dict[str, str] = {}
        for record in records:
            document_id = str(record.input_document_id)
            text = str(record.input_document_text)
            previous = rows.setdefault(document_id, text)
            if previous != text:
                raise ValueError(
                    f"Input document {document_id!r} has inconsistent text"
                )
    else:
        # B fingerprint: Judge-row identity + query + frozen judge_input_text.
        rows = {}
        for record in records:
            sample_id = str(record.sample_id)
            payload = f"{record.query_id}\0{record.judge_input_text}"
            previous = rows.setdefault(sample_id, payload)
            if previous != payload:
                raise ValueError(
                    f"Judge input {sample_id!r} has inconsistent query or text"
                )
    for identifier in sorted(rows):
        payload = f"{identifier}\0{rows[identifier]}\n"
        digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def input_document_fingerprint(records: Sequence[Any]) -> str:
    return record_fingerprint(records, feature_scope="input_document")


def save_hidden_feature_store(
    path: str | Path,
    *,
    features: np.ndarray,
    sample_ids: np.ndarray,
    labels: np.ndarray | None = None,
    query_ids: np.ndarray | None = None,
    audit_document_group_ids: np.ndarray | None = None,
    input_document_ids: np.ndarray | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"features": np.asarray(features, dtype=np.float32), "sample_ids": np.asarray(sample_ids)}
    if labels is not None:
        payload["labels"] = np.asarray(labels, dtype=object)
    if query_ids is not None:
        payload["query_ids"] = np.asarray(query_ids)
    if audit_document_group_ids is not None:
        payload["audit_document_group_ids"] = np.asarray(audit_document_group_ids)
    if input_document_ids is not None:
        payload["input_document_ids"] = np.asarray(input_document_ids)
    if metadata is not None:
        payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False, default=str))
    np.savez_compressed(path, **payload)
    return path


def _first_existing(keys: list[str], candidates: list[str]) -> str | None:
    available = set(keys)
    for candidate in candidates:
        if candidate in available:
            return candidate
    return None


def _optional_array(payload: Any, candidates: list[str]) -> np.ndarray | None:
    key = _first_existing(payload.files, candidates)
    if key is None:
        return None
    return np.asarray(payload[key])
