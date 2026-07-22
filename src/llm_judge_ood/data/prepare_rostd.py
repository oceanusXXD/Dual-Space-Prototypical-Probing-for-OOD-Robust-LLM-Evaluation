from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from src.llm_judge_ood.data.hidden_contract import (
    build_prepared_record,
    require_file_sha256,
    template_sha256,
    write_prepared_contract,
)


ROSTD_QUERY_ID = "rostd_supported_intent"
ROSTD_QUERY_TEXT = "Classify a user utterance into a supported task-oriented intent."
ROSTD_TEMPLATE_VERSION = "rostd_intent_input_v1"
ROSTD_JUDGE_TEMPLATE = (
    "Task: supported task-oriented intent classification.\n"
    "User utterance:\n"
    "{utterance}"
)
ROSTD_TEMPLATE_SHA256 = template_sha256(ROSTD_JUDGE_TEMPLATE)

_EXPECTED_SPLIT_COUNTS = {"train": 30521, "eval": 5681, "test": 11711}
_EXPECTED_OOD_COUNTS = {"train": 0, "eval": 1500, "test": 3090}
_OOD_LABEL = "outOfDomain"


def prepare_rostd(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    test_path: str | Path,
    ood_release_path: str | Path,
    output_path: str | Path,
    expected_sha256: dict[str, str] | None = None,
) -> dict[str, Any]:
    paths = {
        "train": Path(train_path),
        "eval": Path(validation_path),
        "test": Path(test_path),
        "ood_release": Path(ood_release_path),
    }
    expected = dict(expected_sha256 or {})
    source_hashes = {
        name: require_file_sha256(path, expected.get(name)) for name, path in paths.items()
    }
    raw_splits = {
        name: _load_tsv(paths[name], split=name) for name in ("train", "eval", "test")
    }
    split_counts = {name: len(rows) for name, rows in raw_splits.items()}
    if split_counts != _EXPECTED_SPLIT_COUNTS:
        raise ValueError(f"Unexpected canonical ROSTD split counts: {split_counts}")
    ood_counts = {
        name: sum(row["intent_name"] == _OOD_LABEL for row in rows)
        for name, rows in raw_splits.items()
    }
    if ood_counts != _EXPECTED_OOD_COUNTS:
        raise ValueError(f"Unexpected canonical ROSTD OOD counts: {ood_counts}")

    release_rows = _load_tsv(paths["ood_release"], split="ood_release")
    if len(release_rows) != 4590 or any(row["intent_name"] != _OOD_LABEL for row in release_rows):
        raise ValueError("Canonical ROSTD OODrelease.tsv must contain 4,590 OOD rows")
    released_text = Counter(row["utterance"].strip() for row in release_rows)
    split_ood_text = Counter(
        row["utterance"].strip()
        for split_rows in raw_splits.values()
        for row in split_rows
        if row["intent_name"] == _OOD_LABEL
    )
    if released_text != split_ood_text:
        raise ValueError("ROSTD eval/test OOD rows do not exactly match OODrelease.tsv")

    intent_catalog = sorted(
        {
            row["intent_name"]
            for split_rows in raw_splits.values()
            for row in split_rows
            if row["intent_name"] != _OOD_LABEL
        }
    )
    if len(intent_catalog) != 12:
        raise ValueError(f"Canonical ROSTD must contain 12 supported intents, got {len(intent_catalog)}")
    intent_to_id = {name: index for index, name in enumerate(intent_catalog)}

    rows: list[dict[str, Any]] = []
    for official_split in ("train", "eval", "test"):
        for raw in raw_splits[official_split]:
            is_ood = raw["intent_name"] == _OOD_LABEL
            intent_name = None if is_ood else raw["intent_name"]
            intent_id = None if is_ood else intent_to_id[raw["intent_name"]]
            domain_name = "oos" if is_ood else str(raw["intent_name"]).split("/", 1)[0]
            utterance = raw["utterance"]
            document_id = f"rostd::{official_split}::{raw['row_id']:05d}"
            rows.append(
                build_prepared_record(
                    dataset="rostd",
                    sample_id=document_id,
                    # A = the raw task-oriented or human-written OOD utterance.
                    raw_text=utterance,
                    # B = one frozen supported-intent instruction plus that utterance.
                    judge_input_text=ROSTD_JUDGE_TEMPLATE.format(utterance=utterance),
                    query_id=ROSTD_QUERY_ID,
                    query_text=ROSTD_QUERY_TEXT,
                    label=intent_id,
                    split=official_split,
                    document_distribution_role=_document_role(official_split, is_ood=is_ood),
                    audit_document_group_id="oos" if is_ood else f"id::{domain_name}",
                    document_shift_type="official_human_ood" if is_ood else "id",
                    is_document_ood=is_ood,
                    prompt_template_version=ROSTD_TEMPLATE_VERSION,
                    prompt_template_sha256=ROSTD_TEMPLATE_SHA256,
                    metadata={
                        "source_dataset": "rostd_fbrelease",
                        "official_split": official_split,
                        "official_row_id": raw["row_id"],
                        "intent_name": intent_name,
                        "intent_id": intent_id,
                        "domain_name": domain_name,
                        "is_ood": is_ood,
                        "classifier_fit_eligible": official_split == "train" and not is_ood,
                        "main_ood_benchmark_eligible": official_split == "test" and is_ood,
                    },
                )
            )

    return write_prepared_contract(
        output_path,
        rows,
        {
            "artifact_type": "llm_judge_ood_rostd_prepared_metadata",
            "dataset_source": "vgtomahawk/LR_GC_OOD canonical fbrelease",
            "source_paths": {name: str(path) for name, path in paths.items()},
            "source_sha256": source_hashes,
            "official_split_counts": split_counts,
            "official_ood_counts": ood_counts,
            "ood_release_count": len(release_rows),
            "intent_count": len(intent_catalog),
            "intent_catalog": intent_catalog,
            "intent_id_contract": "zero_based_lexicographic_supported_intent_v1",
        },
    )


def _load_tsv(path: Path, *, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for row_id, line in enumerate(handle):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError(f"ROSTD {path}:{row_id + 1} must contain four TSV fields")
            utterance = fields[2]
            if not utterance.strip():
                raise ValueError(f"ROSTD {path}:{row_id + 1} has empty utterance")
            rows.append(
                {
                    "row_id": row_id,
                    "split": split,
                    "intent_name": fields[0],
                    "utterance": utterance,
                }
            )
    return rows


def _document_role(official_split: str, *, is_ood: bool) -> str:
    if official_split == "train":
        return "training"
    if official_split == "eval" and not is_ood:
        return "development"
    if official_split == "test":
        return "benchmark"
    return "excluded"
