from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.common.io import read_jsonl


@dataclass(frozen=True)
class JudgeRecord:
    sample_id: str
    query_id: str
    query_text: str
    document_text: str
    label: Any | None
    split: str
    judge_provenance_id: str
    base_document_id: str
    metadata: dict[str, Any]
    # A-space identity/text: the raw monitored document only. The A extractor
    # keys by input_document_id and never reads judge_input_text.
    input_document_id: str = ""
    input_document_text: str = ""
    # B/Judge-space text: frozen task template plus the raw document. The B
    # extractor keys by sample_id and never reads input_document_text.
    judge_input_text: str = ""
    document_distribution_role: str = "unassigned"
    audit_document_group_id: str = ""
    stream_order: int | None = None
    input_document_contract_explicit: bool | None = None

    def __post_init__(self) -> None:
        explicit_contract = (
            bool(self.input_document_id)
            and bool(self.input_document_text)
            and str(self.document_distribution_role or "unassigned") != "unassigned"
            if self.input_document_contract_explicit is None
            else bool(self.input_document_contract_explicit)
        )
        object.__setattr__(self, "input_document_contract_explicit", explicit_contract)
        object.__setattr__(self, "input_document_id", str(self.input_document_id or self.base_document_id))
        object.__setattr__(self, "input_document_text", str(self.input_document_text or self.document_text))
        object.__setattr__(self, "judge_input_text", str(self.judge_input_text or self.document_text))
        object.__setattr__(self, "document_distribution_role", str(self.document_distribution_role or "unassigned"))
        default_group = self.document_distribution_role if self.document_distribution_role != "unassigned" else "unassigned"
        object.__setattr__(self, "audit_document_group_id", str(self.audit_document_group_id or default_group))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def record_from_mapping(row: dict[str, Any], *, default_judge_provenance: str = "unassigned") -> JudgeRecord:
    sample_id = str(row.get("sample_id") or row.get("id") or row.get("document_id") or row.get("review_id"))
    if not sample_id or sample_id == "None":
        raise ValueError(f"Record is missing a usable sample id: {row}")
    query_id = str(row.get("query_id") or row.get("query") or "global")
    query_text = str(row.get("query_text") or row.get("query") or query_id)
    document_text = str(row.get("document_text") or row.get("document") or row.get("response") or "")
    if not document_text:
        raise ValueError(f"Record {sample_id} is missing document text")
    label = row.get("label", row.get("groundtruth", row.get("score")))
    split = str(row.get("split") or row.get("legacy_split") or "unassigned")
    judge_provenance_id = str(
        row.get("judge_provenance_id")
        or row.get("domain_id")
        or row.get("system_id")
        or row.get("dataset")
        or default_judge_provenance
    )
    base_document_id = str(row.get("base_document_id") or row.get("document_id") or sample_id)
    has_explicit_document_contract = all(
        str(row.get(field) or "")
        for field in ("input_document_id", "input_document_text", "document_distribution_role")
    )
    input_document_id = str(
        row.get("input_document_id")
        or row.get("ood_document_id")
        or row.get("article_id")
        or row.get("document_id")
        or base_document_id
    )
    input_document_text = str(
        row.get("input_document_text")
        or row.get("ood_document_text")
        or row.get("source_document_text")
        or document_text
    )
    judge_input_text = str(row.get("judge_input_text") or document_text)
    document_distribution_role = str(
        row.get("document_distribution_role")
        or row.get("document_role")
        or row.get("ood_role")
        or "unassigned"
    )
    audit_document_group_id = str(
        row.get("audit_document_group_id")
        or row.get("document_group_id")
        or row.get("reporting_group_id")
        or document_distribution_role
    )
    raw_stream_order = row.get("stream_order", row.get("arrival_index"))
    if raw_stream_order in (None, ""):
        stream_order = None
    else:
        try:
            stream_order = int(raw_stream_order)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Record {sample_id} has an invalid stream_order/arrival_index") from error
    metadata = {
        key: value
        for key, value in row.items()
        if key
        not in {
            "sample_id",
            "id",
            "query_id",
            "query_text",
            "query",
            "document_text",
            "document",
            "response",
            "label",
            "groundtruth",
            "score",
            "split",
            "legacy_split",
            "domain_id",
            "judge_provenance_id",
            "system_id",
            "base_document_id",
            "document_id",
            "input_document_id",
            "ood_document_id",
            "input_document_text",
            "judge_input_text",
            "ood_document_text",
            "source_document_text",
            "document_distribution_role",
            "document_role",
            "ood_role",
            "audit_document_group_id",
            "document_group_id",
            "reporting_group_id",
            "stream_order",
            "arrival_index",
        }
    }
    return JudgeRecord(
        sample_id=sample_id,
        query_id=query_id,
        query_text=query_text,
        document_text=document_text,
        label=label,
        split=split,
        judge_provenance_id=judge_provenance_id,
        base_document_id=base_document_id,
        metadata=metadata,
        input_document_id=input_document_id,
        input_document_text=input_document_text,
        judge_input_text=judge_input_text,
        document_distribution_role=document_distribution_role,
        audit_document_group_id=audit_document_group_id,
        stream_order=stream_order,
        input_document_contract_explicit=has_explicit_document_contract,
    )


def load_judge_records(
    paths: Iterable[str | Path], *, default_judge_provenance: str = "unassigned"
) -> list[JudgeRecord]:
    records: list[JudgeRecord] = []
    for path in paths:
        for row in read_jsonl(path):
            records.append(record_from_mapping(row, default_judge_provenance=default_judge_provenance))
    if not records:
        raise ValueError("No judge records were loaded")
    return records


def records_to_frame(records: list[JudgeRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.to_dict() for record in records])


def input_document_text(records: list[JudgeRecord]) -> list[str]:
    return [record.input_document_text for record in records]


def limit_input_document_records(
    records: list[JudgeRecord],
    max_input_documents: int,
    *,
    seed: int,
) -> list[JudgeRecord]:
    """Keep complete Judge-row groups selected only by input-document identity."""

    if max_input_documents <= 0:
        return records
    document_ids = list(dict.fromkeys(record.input_document_id for record in records))
    if len(document_ids) <= max_input_documents:
        return records
    selected = set(
        sorted(
            document_ids,
            key=lambda document_id: sha256(f"{seed}::{document_id}".encode("utf-8")).hexdigest(),
        )[: int(max_input_documents)]
    )
    return [record for record in records if record.input_document_id in selected]
