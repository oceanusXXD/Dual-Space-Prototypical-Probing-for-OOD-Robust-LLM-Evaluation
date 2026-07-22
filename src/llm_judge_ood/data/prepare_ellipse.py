from __future__ import annotations

import csv
import io
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.llm_judge_ood.data.hidden_contract import (
    build_prepared_record,
    require_file_sha256,
    stable_partition,
    stable_rank,
    template_sha256,
    write_prepared_contract,
)


ELLIPSE_QUERY_ID = "overall_language_proficiency"
ELLIPSE_QUERY_TEXT = "Assess overall English-language proficiency on the official 1-5 scale."
ELLIPSE_TEMPLATE_VERSION = "ellipse_overall_proficiency_v1"
ELLIPSE_RUBRIC_SOURCE_SHA256 = "0c0ac5dfa6ae89c9c99ffe483c13ea4a57e8152a1472797e1fe361c2c203efdc"

# This is the Overall column of the official ELLIPSE rubric, transcribed into
# plain text. Analytic trait descriptions and every observed score stay out of
# the B input; they remain audit/label fields in the prepared JSONL.
ELLIPSE_OVERALL_RUBRIC = (
    "Score 5: Native-like facility with syntactic variety, appropriate words and phrases, "
    "well-controlled organization, precise grammar and conventions, and only rare inaccuracies "
    "that do not impede communication.\n"
    "Score 4: Facility with syntactic variety and a range of words and phrases, controlled "
    "organization, accurate grammar and conventions, and occasional inaccuracies that rarely "
    "impede communication.\n"
    "Score 3: Facility limited to common structures and generic vocabulary; organization is "
    "generally controlled, with errors in grammar, syntax, and usage that sometimes impede "
    "communication.\n"
    "Score 2: Inconsistent sentence formation, word choice, mechanics, and only partially "
    "developed organization; language inaccuracies impede communication in many instances.\n"
    "Score 1: A limited range of familiar words or phrases loosely strung together, with frequent "
    "grammar, syntax, and usage errors that impede communication in most cases."
)
ELLIPSE_JUDGE_TEMPLATE = (
    "Task: ELLIPSE overall English-language proficiency assessment.\n"
    "Evaluation rubric:\n"
    f"{ELLIPSE_OVERALL_RUBRIC}\n"
    "Essay to assess:\n"
    "{essay_text}\n"
    "Output contract: one score from 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, or 5."
)
ELLIPSE_TEMPLATE_SHA256 = template_sha256(ELLIPSE_JUDGE_TEMPLATE)

SOURCE_SPLIT_FRACTIONS = {
    "training_train": 0.23,
    "training_drift_reference": 0.10,
    "training_calibration": 0.29,
    "training_validation": 0.08,
    "training_guard": 0.06,
    "training_test": 0.06,
    "benchmark_test": 0.06,
    "deployment_stream": 0.12,
}
TARGET_SPLIT_FRACTIONS = {
    "development": 0.15,
    "benchmark_test": 0.15,
    "deployment_stream": 0.28,
    "deployment_ood_evaluation": 0.105,
    "deployment_adapt": 0.105,
    "deployment_gate": 0.105,
    "deployment_future_test": 0.105,
}

_REQUIRED_COLUMNS = {
    "text_id_kaggle",
    "full_text",
    "task",
    "prompt",
    "Overall",
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
    "set",
}
_ANALYTIC_COLUMNS = (
    "Cohesion",
    "Syntax",
    "Vocabulary",
    "Phraseology",
    "Grammar",
    "Conventions",
)


def prepare_ellipse(
    *,
    train_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    rubric_path: str | Path | None = None,
    seed: int = 42,
    source_prompt_count: int = 30,
    test_zip_password: str = "ellipse_test",
    expected_train_sha256: str | None = None,
    expected_test_sha256: str | None = None,
    expected_rubric_sha256: str | None = None,
) -> dict[str, Any]:
    train = Path(train_path)
    test = Path(test_path)
    source_hashes = {
        "train": require_file_sha256(train, expected_train_sha256),
        "test": require_file_sha256(test, expected_test_sha256),
    }
    rubric = Path(rubric_path) if rubric_path is not None else None
    if rubric is not None:
        source_hashes["rubric"] = require_file_sha256(rubric, expected_rubric_sha256)
    raw_rows = _load_csv(train) + _load_test(test, password=test_zip_password)
    if len(raw_rows) != 6482:
        raise ValueError(f"ELLIPSE official train+test must contain 6,482 rows, got {len(raw_rows)}")

    normalized = [_normalize_row(row) for row in raw_rows]
    document_ids = [row["text_id_kaggle"] for row in normalized]
    if len(set(document_ids)) != 6482:
        duplicates = [key for key, count in Counter(document_ids).items() if count > 1]
        raise ValueError(f"ELLIPSE text_id_kaggle must be unique; examples={duplicates[:5]}")
    prompts = sorted({row["prompt"] for row in normalized})
    if len(prompts) != 44:
        raise ValueError(f"ELLIPSE official release must contain 44 prompts, got {len(prompts)}")
    if not 1 <= int(source_prompt_count) < len(prompts):
        raise ValueError("source_prompt_count must leave at least one source and one held-out prompt")

    # The primary protocol is prompt-disjoint. Prompt roles are frozen before
    # any labels or HiddenStates are inspected by hashing prompt names at seed.
    prompt_order = stable_rank(prompts, seed=int(seed))
    source_prompts = set(prompt_order[: int(source_prompt_count)])
    held_out_prompts = set(prompt_order[int(source_prompt_count) :])
    source_ids = [row["text_id_kaggle"] for row in normalized if row["prompt"] in source_prompts]
    target_ids = [row["text_id_kaggle"] for row in normalized if row["prompt"] in held_out_prompts]
    source_assignments = stable_partition(source_ids, SOURCE_SPLIT_FRACTIONS, seed=int(seed) + 101)
    target_assignments = stable_partition(target_ids, TARGET_SPLIT_FRACTIONS, seed=int(seed) + 202)

    stream_ids = [
        identifier
        for identifier in document_ids
        if source_assignments.get(identifier) == "deployment_stream"
        or target_assignments.get(identifier) == "deployment_stream"
    ]
    stream_order = {
        identifier: index
        for index, identifier in enumerate(stable_rank(stream_ids, seed=int(seed) + 303))
    }

    rows: list[dict[str, Any]] = []
    for raw in normalized:
        identifier = raw["text_id_kaggle"]
        is_ood = raw["prompt"] in held_out_prompts
        split = target_assignments[identifier] if is_ood else source_assignments[identifier]
        role = _role_for_split(split, is_ood=is_ood)
        text = raw["full_text"]
        metadata = {
            "ellipse_text_id": identifier,
            "ellipse_prompt": raw["prompt"],
            "ellipse_task": raw["task"],
            "ellipse_official_set": raw["set"],
            "raw_overall_score": raw["Overall"],
            "analytic_scores": {name: raw[name] for name in _ANALYTIC_COLUMNS},
            "label_scale": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
            "rubric_source_sha256": ELLIPSE_RUBRIC_SOURCE_SHA256,
            "prompt_role": "held_out" if is_ood else "source",
            "auxiliary_near_far_status": "not_assigned; primary OOD is all held-out prompts",
        }
        for audit_field in ("gender", "grade", "race_ethnicity", "SES"):
            if audit_field in raw:
                metadata[f"audit_{audit_field}"] = raw[audit_field]
        rows.append(
            build_prepared_record(
                dataset="ellipse",
                sample_id=f"ellipse::{identifier}",
                # A = the original essay only.
                raw_text=text,
                # B = the frozen Overall rubric plus that same essay.
                judge_input_text=ELLIPSE_JUDGE_TEMPLATE.format(essay_text=text),
                query_id=ELLIPSE_QUERY_ID,
                query_text=ELLIPSE_QUERY_TEXT,
                label=raw["Overall"],
                split=split,
                document_distribution_role=role,
                audit_document_group_id=("held_out" if is_ood else "source")
                + f"::{raw['prompt']}",
                document_shift_type="held_out_prompt" if is_ood else "id",
                is_document_ood=is_ood,
                prompt_template_version=ELLIPSE_TEMPLATE_VERSION,
                prompt_template_sha256=ELLIPSE_TEMPLATE_SHA256,
                stream_order=stream_order.get(identifier),
                metadata=metadata,
            )
        )

    return write_prepared_contract(
        output_path,
        rows,
        {
            "artifact_type": "llm_judge_ood_ellipse_prepared_metadata",
            "dataset_source": "scrosseye/ELLIPSE-Corpus",
            "source_paths": {
                "train": str(train),
                "test": str(test),
                "rubric": None if rubric is None else str(rubric),
            },
            "source_sha256": source_hashes,
            "seed": int(seed),
            "prompt_split_contract": "sha256_ranked_prompt_disjoint_primary_v1",
            "source_prompt_count": len(source_prompts),
            "held_out_prompt_count": len(held_out_prompts),
            "source_prompts": sorted(source_prompts),
            "held_out_prompts": sorted(held_out_prompts),
            "official_row_count": 6482,
            "official_prompt_count": 44,
            "rubric_source_sha256": ELLIPSE_RUBRIC_SOURCE_SHA256,
        },
    )


def _load_test(path: Path, *, password: str) -> list[dict[str, str]]:
    if path.suffix.lower() != ".zip":
        return _load_csv(path)
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"ELLIPSE test zip must contain exactly one CSV, got {csv_names}")
        payload = archive.read(csv_names[0], pwd=password.encode("utf-8"))
    return _parse_csv_bytes(payload, source=str(path))


def _load_csv(path: Path) -> list[dict[str, str]]:
    return _parse_csv_bytes(path.read_bytes(), source=str(path))


def _parse_csv_bytes(payload: bytes, *, source: str) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or not _REQUIRED_COLUMNS.issubset(reader.fieldnames):
        missing = sorted(_REQUIRED_COLUMNS - set(reader.fieldnames or ()))
        raise ValueError(f"ELLIPSE source {source} is missing columns {missing}")
    rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"ELLIPSE source is empty: {source}")
    return rows


def _normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    identifier = str(row.get("text_id_kaggle") or "").strip()
    text = str(row.get("full_text") or "")
    prompt = str(row.get("prompt") or "").strip()
    if not identifier or not text.strip() or not prompt:
        raise ValueError("ELLIPSE rows require text_id_kaggle, full_text, and prompt")
    normalized = dict(row)
    normalized.update(
        {
            "text_id_kaggle": identifier,
            "full_text": text,
            "prompt": prompt,
            "task": str(row.get("task") or ""),
            "set": str(row.get("set") or ""),
        }
    )
    for name in ("Overall", *_ANALYTIC_COLUMNS):
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"ELLIPSE row {identifier} has invalid score {name}") from error
        if value not in {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0}:
            raise ValueError(f"ELLIPSE row {identifier} has out-of-contract {name}={value}")
        normalized[name] = value
    return normalized


def _role_for_split(split: str, *, is_ood: bool) -> str:
    if str(split).startswith("training_"):
        return "training"
    if split == "development":
        return "development"
    if split == "benchmark_test":
        return "benchmark"
    if str(split).startswith("deployment_"):
        return "deployment"
    raise ValueError(f"Unknown ELLIPSE split {split!r} (is_ood={is_ood})")
