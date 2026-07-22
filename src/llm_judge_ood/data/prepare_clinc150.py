from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.llm_judge_ood.data.hidden_contract import (
    build_prepared_record,
    require_file_sha256,
    template_sha256,
    write_prepared_contract,
)


CLINC150_QUERY_ID = "clinc150_intent"
CLINC150_QUERY_TEXT = "Classify an in-scope CLINC150 user utterance by intent."
CLINC150_TEMPLATE_VERSION = "clinc150_intent_input_v1"
CLINC150_JUDGE_TEMPLATE = (
    "Task: CLINC150 in-scope intent classification.\n"
    "User utterance:\n"
    "{utterance}"
)
CLINC150_TEMPLATE_SHA256 = template_sha256(CLINC150_JUDGE_TEMPLATE)

_EXPECTED_COUNTS = {
    "train": 15000,
    "val": 3000,
    "test": 4500,
    "oos_train": 100,
    "oos_val": 100,
    "oos_test": 1000,
}
_SPLIT_ORDER = ("train", "val", "test", "oos_train", "oos_val", "oos_test")


def prepare_clinc150(
    *,
    data_path: str | Path,
    domains_path: str | Path,
    output_path: str | Path,
    expected_data_sha256: str | None = None,
    expected_domains_sha256: str | None = None,
) -> dict[str, Any]:
    data_file = Path(data_path)
    domains_file = Path(domains_path)
    source_hashes = {
        "data_full": require_file_sha256(data_file, expected_data_sha256),
        "domains": require_file_sha256(domains_file, expected_domains_sha256),
    }
    data = json.loads(data_file.read_text(encoding="utf-8"))
    domains = json.loads(domains_file.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(domains, dict):
        raise ValueError("CLINC150 data_full.json and domains.json must both contain JSON objects")
    actual_counts = {name: len(data.get(name, ())) for name in _SPLIT_ORDER}
    if actual_counts != _EXPECTED_COUNTS:
        raise ValueError(f"Unexpected CLINC150 official split counts: {actual_counts}")

    intent_to_domain: dict[str, str] = {}
    for domain_name, intent_names in domains.items():
        for intent_name in intent_names:
            intent = str(intent_name)
            if intent in intent_to_domain:
                raise ValueError(f"CLINC150 intent {intent!r} appears in multiple domains")
            intent_to_domain[intent] = str(domain_name)
    intent_catalog = sorted(intent_to_domain)
    if len(intent_catalog) != 150:
        raise ValueError(f"CLINC150 domains.json must define 150 intents, got {len(intent_catalog)}")
    intent_to_id = {name: index for index, name in enumerate(intent_catalog)}

    rows: list[dict[str, Any]] = []
    for official_split in _SPLIT_ORDER:
        is_oos = official_split.startswith("oos_")
        for row_index, item in enumerate(data[official_split]):
            if not isinstance(item, list) or len(item) != 2:
                raise ValueError(
                    f"CLINC150 {official_split}[{row_index}] must be [utterance, intent]"
                )
            utterance, intent_name = str(item[0]), str(item[1])
            if not utterance.strip():
                raise ValueError(f"CLINC150 {official_split}[{row_index}] has empty utterance")
            if is_oos:
                if intent_name != "oos":
                    raise ValueError(
                        f"CLINC150 {official_split}[{row_index}] must use intent='oos'"
                    )
                intent_id: int | None = None
                domain_name = "oos"
            else:
                if intent_name not in intent_to_id:
                    raise ValueError(f"Unknown CLINC150 intent {intent_name!r}")
                intent_id = intent_to_id[intent_name]
                domain_name = intent_to_domain[intent_name]

            document_id = f"clinc150::{official_split}::{row_index:05d}"
            rows.append(
                build_prepared_record(
                    dataset="clinc150",
                    sample_id=document_id,
                    # A = the raw user utterance only.
                    raw_text=utterance,
                    # B = one frozen task instruction plus the same utterance.
                    judge_input_text=CLINC150_JUDGE_TEMPLATE.format(utterance=utterance),
                    query_id=CLINC150_QUERY_ID,
                    query_text=CLINC150_QUERY_TEXT,
                    label=intent_id,
                    split=official_split,
                    document_distribution_role=_document_role(official_split),
                    audit_document_group_id=("oos" if is_oos else f"id::{domain_name}"),
                    document_shift_type="official_oos" if is_oos else "id",
                    is_document_ood=is_oos,
                    prompt_template_version=CLINC150_TEMPLATE_VERSION,
                    prompt_template_sha256=CLINC150_TEMPLATE_SHA256,
                    metadata={
                        "source_dataset": "clinc150",
                        "official_split": official_split,
                        "official_row_id": row_index,
                        "intent_name": None if is_oos else intent_name,
                        "intent_id": intent_id,
                        "domain_name": domain_name,
                        "is_oos": is_oos,
                        "classifier_fit_eligible": official_split == "train",
                        "main_ood_benchmark_eligible": official_split == "oos_test",
                    },
                )
            )

    return write_prepared_contract(
        output_path,
        rows,
        {
            "artifact_type": "llm_judge_ood_clinc150_prepared_metadata",
            "dataset_source": "clinc/oos-eval data/data_full.json",
            "source_paths": {"data_full": str(data_file), "domains": str(domains_file)},
            "source_sha256": source_hashes,
            "official_split_counts": actual_counts,
            "intent_count": len(intent_catalog),
            "intent_catalog": intent_catalog,
            "intent_id_contract": "zero_based_lexicographic_intent_name_v1",
        },
    )


def _document_role(official_split: str) -> str:
    if official_split == "train":
        return "training"
    if official_split == "val":
        return "development"
    if official_split in {"test", "oos_test"}:
        return "benchmark"
    return "excluded"
