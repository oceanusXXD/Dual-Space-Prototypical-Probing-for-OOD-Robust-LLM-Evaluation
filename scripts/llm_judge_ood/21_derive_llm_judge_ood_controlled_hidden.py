#!/usr/bin/env python
"""Compose a controlled-flow cache from verified base and event caches."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json, write_jsonl
from src.llm_judge_ood.shared.feature_store import load_hidden_feature_store, record_fingerprint
from src.llm_judge_ood.shared.schema import JudgeRecord, load_judge_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive an exact controlled-flow Qwen cache without re-embedding unchanged documents."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--base-input", required=True)
    parser.add_argument("--base-cache", required=True)
    parser.add_argument("--event-cache", default=None)
    parser.add_argument("--event-cache-input", nargs="*", default=None)
    parser.add_argument("--reusable-event-input", action="append", default=[])
    parser.add_argument("--reusable-event-cache", action="append", default=[])
    parser.add_argument("--missing-event-output", default=None)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _unique_documents(records: list[JudgeRecord]) -> list[JudgeRecord]:
    unique: dict[str, JudgeRecord] = {}
    for record in records:
        previous = unique.setdefault(record.input_document_id, record)
        if previous.input_document_text != record.input_document_text:
            raise ValueError(f"Input document {record.input_document_id!r} has inconsistent text")
    return list(unique.values())


def _features_by_document_id(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    store = load_hidden_feature_store(path)
    if store.input_document_ids is None:
        raise ValueError(f"{path} is missing input_document_ids")
    metadata = store.metadata.get("cache_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} is missing structured cache metadata")
    document_ids = store.input_document_ids.astype(str).tolist()
    if len(document_ids) != len(set(document_ids)):
        raise ValueError(f"{path} has duplicate input_document_ids")
    if metadata.get("feature_scope") != "input_document":
        raise ValueError(f"{path} has incompatible feature_scope={metadata.get('feature_scope')!r}")
    return {
        document_id: np.asarray(feature, dtype=np.float16)
        for document_id, feature in zip(document_ids, store.features, strict=True)
    }, metadata


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    base_input_path = Path(args.base_input)
    base_cache_path = Path(args.base_cache)
    output = Path(args.output)
    records = load_judge_records([input_path])
    base_records = _unique_documents(load_judge_records([base_input_path]))
    documents = _unique_documents(records)
    base_by_id = {record.input_document_id: record for record in base_records}
    base_features, base_metadata = _features_by_document_id(base_cache_path)
    expected_base_fingerprint = record_fingerprint(base_records, feature_scope="input_document")
    if base_metadata.get("dataset_fingerprint") != expected_base_fingerprint:
        raise ValueError("Base cache fingerprint does not match --base-input")

    event_documents = [record for record in documents if record.input_document_id not in base_by_id]
    event_features: dict[str, np.ndarray] = {}
    event_metadata: dict[str, object] | None = None
    if args.event_cache is not None:
        event_features, event_metadata = _features_by_document_id(Path(args.event_cache))
        event_cache_documents = (
            _unique_documents(load_judge_records(args.event_cache_input))
            if args.event_cache_input
            else event_documents
        )
        expected_event_fingerprint = record_fingerprint(
            event_cache_documents, feature_scope="input_document"
        )
        if event_metadata.get("dataset_fingerprint") != expected_event_fingerprint:
            raise ValueError("Event cache fingerprint does not match the controlled-flow event documents")
        if set(event_features) != {
            record.input_document_id for record in event_cache_documents
        }:
            raise ValueError("Event cache document IDs do not exactly match controlled-flow event documents")

    if len(args.reusable_event_input) != len(args.reusable_event_cache):
        raise ValueError(
            "--reusable-event-input and --reusable-event-cache must be supplied in pairs"
        )
    reusable_by_lineage: dict[str, tuple[str, np.ndarray]] = {}
    for reusable_input, reusable_cache in zip(
        args.reusable_event_input, args.reusable_event_cache, strict=True
    ):
        reusable_documents = _unique_documents(load_judge_records([reusable_input]))
        reusable_features, reusable_metadata = _features_by_document_id(
            Path(reusable_cache)
        )
        expected_fingerprint = record_fingerprint(
            reusable_documents, feature_scope="input_document"
        )
        if reusable_metadata.get("dataset_fingerprint") != expected_fingerprint:
            raise ValueError(
                "Reusable event cache fingerprint does not match its declared input"
            )
        if set(reusable_features) != {
            record.input_document_id for record in reusable_documents
        }:
            raise ValueError(
                "Reusable event cache document IDs do not exactly match its declared input"
            )
        for record in reusable_documents:
            lineage_id = str(
                record.metadata.get("lineage_document_id")
                or record.input_document_id
            )
            candidate = (
                record.input_document_text,
                reusable_features[record.input_document_id],
            )
            previous = reusable_by_lineage.setdefault(lineage_id, candidate)
            if previous[0] != candidate[0] or not np.array_equal(previous[1], candidate[1]):
                raise ValueError(
                    f"Reusable lineage {lineage_id!r} has inconsistent text or hidden features"
                )

    missing_events = []
    for record in event_documents:
        if record.input_document_id in event_features:
            continue
        lineage_id = str(record.metadata.get("lineage_document_id") or "")
        reusable = reusable_by_lineage.get(lineage_id)
        if reusable is None or reusable[0] != record.input_document_text:
            missing_events.append(record)
    if missing_events and args.missing_event_output is not None:
        missing_output = Path(args.missing_event_output)
        missing_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(missing_output, [record.to_dict() for record in missing_events])
        print(
            json.dumps(
                {
                    "status": "missing_event_subset_written",
                    "output": str(missing_output),
                    "missing_event_documents": len(missing_events),
                    "reusable_event_documents": len(event_documents) - len(missing_events),
                },
                indent=2,
            )
        )
        return

    features: list[np.ndarray] = []
    reused_base = 0
    reused_lineage = 0
    reused_event_lineage = 0
    extracted_events = 0
    for record in documents:
        base_record = base_by_id.get(record.input_document_id)
        if base_record is not None:
            if base_record.input_document_text != record.input_document_text:
                raise ValueError(f"Base document text changed for {record.input_document_id!r}")
            features.append(base_features[record.input_document_id])
            reused_base += 1
            continue
        lineage_id = str(record.metadata.get("lineage_document_id", ""))
        lineage_record = base_by_id.get(lineage_id)
        if lineage_record is not None and lineage_record.input_document_text == record.input_document_text:
            features.append(base_features[lineage_id])
            reused_lineage += 1
            continue
        if record.input_document_id not in event_features:
            lineage_id = str(record.metadata.get("lineage_document_id") or "")
            reusable = reusable_by_lineage.get(lineage_id)
            if reusable is None or reusable[0] != record.input_document_text:
                raise ValueError(
                    f"Event {record.input_document_id!r} changed text and has no verified event-cache feature"
                )
            features.append(reusable[1])
            reused_event_lineage += 1
        else:
            features.append(event_features[record.input_document_id])
            extracted_events += 1

    matrix = np.stack(features, axis=0).astype(np.float16)
    metadata = dict(base_metadata)
    metadata.update(
        {
            "dataset_fingerprint": record_fingerprint(records, feature_scope="input_document"),
            "num_records": len(documents),
            "num_input_documents": len(documents),
            "shape": list(matrix.shape),
            "feature_storage_dtype": "float16",
            "derivation": {
                "method": "verified_base_and_lineage_cache_composition_v1",
                "base_input": str(base_input_path),
                "base_cache": str(base_cache_path),
                "event_cache": str(args.event_cache) if args.event_cache is not None else None,
                "reused_base_documents": reused_base,
                "reused_lineage_events": reused_lineage,
                "reused_cross_flow_event_lineage": reused_event_lineage,
                "extracted_event_documents": extracted_events,
            },
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=matrix,
        sample_ids=np.asarray([record.sample_id for record in documents]),
        labels=np.asarray([record.label for record in documents], dtype=object),
        query_ids=np.asarray([record.query_id for record in documents]),
        audit_document_group_ids=np.asarray([record.audit_document_group_id for record in documents]),
        input_document_ids=np.asarray([record.input_document_id for record in documents]),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, default=str)),
    )
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(json.dumps({"output": str(output), "shape": list(matrix.shape), "derivation": metadata["derivation"]}, indent=2))


if __name__ == "__main__":
    main()
