#!/usr/bin/env python3
"""Build the deduplicated FLASK A-space contract from scored B-space rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json, write_jsonl


DOMAINS = (
    "Humanities",
    "Language",
    "Social Science",
    "History",
    "Culture",
)
SKILLS = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Deduplicate the completed 5x6 FLASK B-space into one raw candidate "
            "response A-space record per response_id."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge/"
            "b_space_with_direct_judge.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge/a_space_contract.jsonl"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = list(load_jsonl(args.input))
    records = build_a_space(rows)
    if args.output.exists() and not args.overwrite:
        existing = list(load_jsonl(args.output))
        if existing != records:
            raise ValueError("A-space output differs; pass --overwrite to replace it")
    else:
        write_jsonl(args.output, records)
    metadata = {
        "artifact_type": "flask_5x6_a_space_contract_v1",
        "source_b_space": str(args.input),
        "a_space_records": len(records),
        "domains": list(DOMAINS),
        "source_b_rows": len(rows),
        "excluded_empty_candidate_response_records": sum(
            not str(row.get("candidate_response") or "").strip() for row in rows
        ),
        "input_document_policy": "raw_candidate_response_only",
        "deduplication_key": "response_id",
    }
    write_json(args.output.with_suffix(".metadata.json"), metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing B-space result: {path}")
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_a_space(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    source_count = 0
    for row in rows:
        source_count += 1
        domains = list(row.get("domain_ids") or [])
        domain = str(domains[0]) if len(domains) == 1 else ""
        task = str(row.get("task_id") or "")
        if domain not in DOMAINS or task not in SKILLS:
            raise ValueError(f"B-space row is outside the documented 5x6 scope: {row.get('b_id')}")
        response_id = str(row.get("response_id") or "")
        candidate = str(row.get("candidate_response") or "")
        if not response_id:
            raise ValueError(f"B-space row lacks response_id: {row.get('b_id')}")
        if not candidate.strip():
            # The documented model-input scope has already excluded these
            # rows.  Keeping the guard here makes A-space robust to raw input.
            continue
        candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        record = {
            # Fields below are understood by 20_prepare_llm_judge_ood_hidden.py.
            "sample_id": response_id,
            "query_id": str(row.get("base_id")),
            "query_text": str(row.get("instruction") or ""),
            "document_text": candidate,
            "label": None,
            "split": "all",
            "judge_provenance_id": domain,
            "base_document_id": response_id,
            "input_document_id": response_id,
            "input_document_text": candidate,
            "document_distribution_role": domain,
            "audit_document_group_id": str(row.get("base_id")),
            "response_id": response_id,
            "base_id": row.get("base_id"),
            "domain_id": domain,
            "generator_id": str(row.get("generator_id") or ""),
            "candidate_response_sha256": candidate_hash,
        }
        previous = unique.get(response_id)
        if previous is None:
            unique[response_id] = record
            continue
        comparable = (
            "input_document_text",
            "query_id",
            "domain_id",
            "generator_id",
            "candidate_response_sha256",
        )
        mismatched = [key for key in comparable if previous[key] != record[key]]
        if mismatched:
            raise ValueError(
                f"response_id={response_id!r} is inconsistent across B-space rows: {mismatched}"
            )
    if not source_count:
        raise ValueError("No B-space rows were loaded")
    records = list(unique.values())
    records.sort(
        key=lambda record: (
            DOMAINS.index(str(record["domain_id"])),
            int(record["base_id"]),
            str(record["generator_id"]),
        )
    )
    return records


if __name__ == "__main__":
    main()
