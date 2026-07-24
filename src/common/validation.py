from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.common.io import write_json, write_jsonl


def template_sha256(template: str) -> str:
    """Hash the frozen template with its document placeholder still present."""

    return hashlib.sha256(str(template).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_sha256(path: str | Path, expected: str | None) -> str:
    actual = file_sha256(path)
    if expected and actual != str(expected):
        raise ValueError(
            f"Source checksum mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def stable_rank(values: Iterable[str], *, seed: int) -> list[str]:
    return sorted(
        (str(value) for value in values),
        key=lambda value: hashlib.sha256(f"{int(seed)}::{value}".encode("utf-8")).hexdigest(),
    )


def stable_partition(
    identifiers: Sequence[str],
    fractions: Mapping[str, float],
    *,
    seed: int,
) -> dict[str, str]:
    names = tuple(str(name) for name in fractions)
    weights = [float(fractions[name]) for name in names]
    if not names or any(weight < 0.0 for weight in weights):
        raise ValueError("Partition fractions must be non-negative and non-empty")
    total = sum(weights)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Partition fractions must sum to 1.0, got {total}")

    ranked = stable_rank(identifiers, seed=int(seed))
    exact = [len(ranked) * weight for weight in weights]
    counts = [int(value) for value in exact]
    for index in sorted(
        range(len(names)),
        key=lambda item: (exact[item] - counts[item], -item),
        reverse=True,
    )[: len(ranked) - sum(counts)]:
        counts[index] += 1

    assignments: dict[str, str] = {}
    offset = 0
    for name, count in zip(names, counts, strict=True):
        for identifier in ranked[offset : offset + count]:
            assignments[identifier] = name
        offset += count
    return assignments


def build_prepared_record(
    *,
    dataset: str,
    sample_id: str,
    raw_text: str,
    judge_input_text: str,
    query_id: str,
    query_text: str,
    label: Any,
    split: str,
    document_distribution_role: str,
    audit_document_group_id: str,
    document_shift_type: str,
    is_document_ood: bool,
    prompt_template_version: str,
    prompt_template_sha256: str,
    input_document_id: str | None = None,
    base_document_id: str | None = None,
    stream_order: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    record_id = str(sample_id)
    document_id = str(input_document_id or sample_id)
    base_id = str(base_document_id or document_id)
    text = str(raw_text)
    if not record_id or not document_id or not text.strip():
        raise ValueError("Prepared rows require a non-empty sample id, document id, and raw text")
    if not str(judge_input_text).strip():
        raise ValueError(f"Prepared row {document_id} has an empty Judge input")

    row: dict[str, Any] = {
        "sample_id": record_id,
        "id": record_id,
        "dataset": str(dataset),
        "query_id": str(query_id),
        "query_text": str(query_text),
        "document_text": text,
        "label": label,
        "groundtruth": label,
        "split": str(split),
        "judge_provenance_id": f"{dataset}::{query_id}",
        "base_document_id": base_id,
        # A-space contract: the monitored input is only the untouched essay,
        # utterance, or article. No instruction, rubric, label, or OOD flag.
        "input_document_id": document_id,
        "input_document_text": text,
        # B/Judge-space contract: this is the frozen, label-free task template
        # plus the same raw text. The extractor reads this field only for B.
        "judge_input_text": str(judge_input_text),
        "document_distribution_role": str(document_distribution_role),
        "audit_document_group_id": str(audit_document_group_id),
        "document_shift_type": str(document_shift_type),
        "is_document_ood": bool(is_document_ood),
        "arrival_batch_id": document_id,
        "stream_order": stream_order,
        "prompt_template_version": str(prompt_template_version),
        "prompt_template_sha256": str(prompt_template_sha256),
    }
    extra = dict(metadata or {})
    overlap = sorted(set(row) & set(extra))
    if overlap:
        raise ValueError(f"Audit metadata cannot replace prepared-contract fields: {overlap}")
    row.update(extra)
    return row


def validate_prepared_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Prepared dataset is empty")
    sample_ids = [str(row.get("sample_id") or "") for row in rows]
    document_ids = [str(row.get("input_document_id") or "") for row in rows]
    if any(not value for value in sample_ids + document_ids):
        raise ValueError("Prepared rows contain empty sample/document ids")
    duplicate_samples = [key for key, count in Counter(sample_ids).items() if count != 1]
    if duplicate_samples:
        raise ValueError(f"Prepared sample ids must be unique; examples={duplicate_samples[:5]}")
    document_text_by_id: dict[str, str] = {}
    for row in rows:
        document_id = str(row.get("input_document_id") or "")
        text = str(row.get("input_document_text") or "")
        previous = document_text_by_id.get(document_id)
        if previous is not None and previous != text:
            raise ValueError(f"Prepared input-document id {document_id!r} has inconsistent text")
        document_text_by_id[document_id] = text

    template_pairs = {
        (
            str(row.get("prompt_template_version") or ""),
            str(row.get("prompt_template_sha256") or ""),
        )
        for row in rows
    }
    if len(template_pairs) != 1 or any(not value for value in next(iter(template_pairs))):
        raise ValueError(f"Prepared rows require one complete B-template identity, got {template_pairs}")
    return {
        "row_count": len(rows),
        "unique_sample_ids": len(set(sample_ids)),
        "unique_input_document_ids": len(set(document_ids)),
        "split_counts": dict(sorted(Counter(str(row["split"]) for row in rows).items())),
        "document_shift_counts": dict(
            sorted(Counter(str(row["document_shift_type"]) for row in rows).items())
        ),
        "prompt_template_version": next(iter(template_pairs))[0],
        "prompt_template_sha256": next(iter(template_pairs))[1],
    }


def write_prepared_contract(
    output_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_path)
    audit = validate_prepared_rows(rows)
    write_jsonl(output, [dict(row) for row in rows])
    payload = {
        **dict(metadata),
        **audit,
        "output_path": str(output),
        "prepared_sha256": file_sha256(output),
    }
    write_json(output.with_suffix(".metadata.json"), payload)
    return payload
