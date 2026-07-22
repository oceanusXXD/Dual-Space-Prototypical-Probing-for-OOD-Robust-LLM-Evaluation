from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.common.io import write_json, write_jsonl

SUMMEVAL_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")
SUMMEVAL_QUERY_TEXT = {
    "coherence": "Rate the coherence of the candidate summary on a 1-5 scale.",
    "consistency": "Rate the factual consistency of the candidate summary with the source document on a 1-5 scale.",
    "fluency": "Rate the fluency and grammatical quality of the candidate summary on a 1-5 scale.",
    "relevance": "Rate how well the candidate summary captures important source content on a 1-5 scale.",
}
DEFAULT_DOCUMENT_PARTITION_FRACTIONS = {
    "training": 0.60,
    "development": 0.20,
    "deployment": 0.20,
}
DEFAULT_TRAINING_DOCUMENT_SPLIT_FRACTIONS = {
    "training_train": 0.50,
    "training_validation": 0.15,
    "training_calibration": 0.15,
    "training_guard": 0.10,
    "training_test": 0.10,
}
DEFAULT_DEPLOYMENT_DOCUMENT_SPLIT_FRACTIONS = {
    "deployment_stream": 0.40,
    "deployment_probe": 0.15,
    "deployment_adapt": 0.15,
    "deployment_gate": 0.15,
    "deployment_future_test": 0.15,
}
SUMMEVAL_SHIFT_TYPES = ("id", "near", "far")
_FORMAT_MARKER = "NEWS ARTICLE FORMAT"


def prepare_summeval_rows_from_dataset(
    dataset_rows: Iterable[dict[str, Any]],
    *,
    document_partition_fractions: dict[str, float] | None = None,
    training_document_split_fractions: dict[str, float] | None = None,
    deployment_document_split_fractions: dict[str, float] | None = None,
    candidate_system_ids: Sequence[str] | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Expand SummEval's article x system x dimension arrays into JudgeRecord rows.

    The HF `mteb/summeval` dataset stores one row per article with 16 machine
    summaries and four parallel score arrays. Document partitions are assigned
    before candidate expansion, so candidate-system provenance cannot define
    the production OOD task.
    """

    partition_fractions = _resolve_fraction_mapping(
        document_partition_fractions,
        default=DEFAULT_DOCUMENT_PARTITION_FRACTIONS,
        name="document_partition_fractions",
    )
    training_fractions = _resolve_fraction_mapping(
        training_document_split_fractions,
        default=DEFAULT_TRAINING_DOCUMENT_SPLIT_FRACTIONS,
        name="training_document_split_fractions",
    )
    deployment_fractions = _resolve_fraction_mapping(
        deployment_document_split_fractions,
        default=DEFAULT_DEPLOYMENT_DOCUMENT_SPLIT_FRACTIONS,
        name="deployment_document_split_fractions",
    )
    selected_candidate_systems = (
        None if candidate_system_ids is None else {str(value) for value in candidate_system_ids}
    )
    raw_rows = list(dataset_rows)
    valid_articles: list[tuple[int, dict[str, Any], str]] = []
    for article_index, row in enumerate(raw_rows):
        source_text = str(row.get("text") or row.get("input") or "")
        summaries = list(row.get("machine_summaries") or [])
        if not source_text or not summaries:
            continue
        valid_articles.append((article_index, row, str(row.get("id") or f"article_{article_index:04d}")))
    document_partitions = _stable_partition(
        [article_id for _, _, article_id in valid_articles],
        partition_fractions,
        seed=seed,
    )
    training_splits = _stable_partition(
        [
            article_id
            for _, _, article_id in valid_articles
            if document_partitions[article_id] == "training"
        ],
        training_fractions,
        seed=seed + 1,
    )
    deployment_splits = _stable_partition(
        [
            article_id
            for _, _, article_id in valid_articles
            if document_partitions[article_id] == "deployment"
        ],
        deployment_fractions,
        seed=seed + 2,
    )
    document_shift_types, stream_order_by_document = _covariate_shift_plan(
        document_partitions=document_partitions,
        deployment_splits=deployment_splits,
        seed=seed + 3,
    )
    rows: list[dict[str, Any]] = []
    for article_index, row, article_id in valid_articles:
        source_text = str(row.get("text") or row.get("input") or "")
        shift_type = document_shift_types[article_id]
        shifted_source_text = _apply_covariate_shift(source_text, shift_type=shift_type)
        summaries = list(row.get("machine_summaries") or [])
        document_role = document_partitions[article_id]
        if document_role == "training":
            split = training_splits[article_id]
            selection_role = "training_fit" if split == "training_train" else "training_holdout"
        elif document_role == "development":
            split = "development"
            selection_role = "development_only"
        else:
            split = deployment_splits[article_id]
            selection_role = "deployment_blind"
        for candidate_index, summary in enumerate(summaries):
            candidate_system_id = f"candidate_system_{candidate_index:02d}"
            if selected_candidate_systems is not None and candidate_system_id not in selected_candidate_systems:
                continue
            for dimension in SUMMEVAL_DIMENSIONS:
                scores = row.get(dimension)
                if scores is None or candidate_index >= len(scores):
                    continue
                label = _round_score_to_ordinal(scores[candidate_index])
                sample_id = f"summeval::{article_id}::{candidate_system_id}::{dimension}"
                judge_input_text = (
                    f"Evaluation dimension: {dimension}\n"
                    f"Instruction: {SUMMEVAL_QUERY_TEXT[dimension]}\n"
                    f"Input document: {shifted_source_text}\n"
                    f"Candidate response: {summary}"
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "id": sample_id,
                        "dataset": "summeval",
                        "query_id": dimension,
                        "query_text": SUMMEVAL_QUERY_TEXT[dimension],
                        "document_text": judge_input_text,
                        "judge_input_text": judge_input_text,
                        "label": label,
                        "groundtruth": label,
                        "raw_score": float(scores[candidate_index]),
                        "split": split,
                        "judge_provenance_id": "summeval_judge",
                        "base_document_id": article_id,
                        "input_document_id": article_id,
                        "input_document_text": shifted_source_text,
                        "document_distribution_role": document_role,
                        "audit_document_group_id": shift_type,
                        "document_shift_type": shift_type,
                        "is_document_ood": shift_type != "id",
                        "shift_construction": "label_preserving_format_and_length_transform",
                        "stream_order": stream_order_by_document.get(article_id),
                        "candidate_system_id": candidate_system_id,
                        "candidate_system_index": candidate_index,
                        "selection_role": selection_role,
                        "prompt_template_version": "summeval_query_source_candidate_v3",
                    }
                )
    return rows


def load_summeval_hf(dataset_name: str = "mteb/summeval", split: str = "test") -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the `datasets` package to download SummEval from Hugging Face.") from exc
    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def write_summeval_prepared(
    *,
    output_path: str | Path,
    dataset_name: str = "mteb/summeval",
    split: str = "test",
    document_partition_fractions: dict[str, float] | None = None,
    training_document_split_fractions: dict[str, float] | None = None,
    deployment_document_split_fractions: dict[str, float] | None = None,
    candidate_system_ids: Sequence[str] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    raw_rows = load_summeval_hf(dataset_name=dataset_name, split=split)
    prepared_all = prepare_summeval_rows_from_dataset(
        raw_rows,
        document_partition_fractions=document_partition_fractions,
        training_document_split_fractions=training_document_split_fractions,
        deployment_document_split_fractions=deployment_document_split_fractions,
        candidate_system_ids=candidate_system_ids,
        seed=seed,
    )
    prepared = prepared_all
    output = Path(output_path)
    write_jsonl(output, prepared)
    metadata = _metadata(
        raw_rows=raw_rows,
        prepared=prepared,
        output_path=output,
        dataset_name=dataset_name,
        split=split,
        document_partition_fractions=_resolve_fraction_mapping(
            document_partition_fractions,
            default=DEFAULT_DOCUMENT_PARTITION_FRACTIONS,
            name="document_partition_fractions",
        ),
        training_document_split_fractions=_resolve_fraction_mapping(
            training_document_split_fractions,
            default=DEFAULT_TRAINING_DOCUMENT_SPLIT_FRACTIONS,
            name="training_document_split_fractions",
        ),
        deployment_document_split_fractions=_resolve_fraction_mapping(
            deployment_document_split_fractions,
            default=DEFAULT_DEPLOYMENT_DOCUMENT_SPLIT_FRACTIONS,
            name="deployment_document_split_fractions",
        ),
        candidate_system_ids=candidate_system_ids,
        seed=seed,
    )
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare HF SummEval for the standalone LLM Judge OOD pipeline.")
    parser.add_argument("--dataset", default="mteb/summeval")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="artifacts/llm_judge_ood_summeval/summeval_prepared.jsonl")
    parser.add_argument(
        "--candidate-system-ids",
        nargs="*",
        default=None,
        help="Optional candidate systems retained for Judge evaluation; never an OOD partition key.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-config",
        default=None,
        help=(
            "Optional JSON file with document_partition_fractions, "
            "training_document_split_fractions, and deployment_document_split_fractions. "
            "Values may be fractions or relative integer weights."
        ),
    )
    args = parser.parse_args(argv)
    split_config = _load_split_config(args.split_config)
    metadata = write_summeval_prepared(
        output_path=args.output,
        dataset_name=args.dataset,
        split=args.split,
        document_partition_fractions=split_config.get("document_partition_fractions"),
        training_document_split_fractions=split_config.get("training_document_split_fractions"),
        deployment_document_split_fractions=split_config.get("deployment_document_split_fractions"),
        candidate_system_ids=tuple(args.candidate_system_ids) if args.candidate_system_ids is not None else None,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def _round_score_to_ordinal(value: Any) -> int:
    score = float(value)
    return int(np.clip(np.rint(score), 1, 5))


def _stable_split(key: str, fractions: dict[str, float], *, seed: int) -> str:
    total = float(sum(fractions.values()))
    if total <= 0:
        raise ValueError("Split fractions must sum to a positive value")
    digest = hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest()
    value = int(digest[:16], 16) / float(16**16)
    cumulative = 0.0
    last_name = next(reversed(fractions))
    for name, fraction in fractions.items():
        cumulative += float(fraction) / total
        if value <= cumulative:
            return str(name)
    return str(last_name)


def _stable_partition(keys: Sequence[str], fractions: dict[str, float], *, seed: int) -> dict[str, str]:
    unique_keys = list(dict.fromkeys(str(key) for key in keys))
    if len(unique_keys) != len(keys):
        raise ValueError("Input document ids must be unique for deterministic grouped splitting")
    total = float(sum(fractions.values()))
    if total <= 0:
        raise ValueError("Split fractions must sum to a positive value")
    names = list(fractions)
    raw_sizes = np.asarray([float(fractions[name]) / total * len(unique_keys) for name in names])
    sizes = np.floor(raw_sizes).astype(int)
    remainder = int(len(unique_keys) - sizes.sum())
    if remainder > 0:
        order = np.argsort(-(raw_sizes - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    ranked = sorted(
        unique_keys,
        key=lambda key: hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest(),
    )
    output: dict[str, str] = {}
    start = 0
    for name, size in zip(names, sizes.tolist(), strict=True):
        for key in ranked[start : start + size]:
            output[key] = str(name)
        start += size
    return output


def _covariate_shift_plan(
    *,
    document_partitions: dict[str, str],
    deployment_splits: dict[str, str],
    seed: int,
) -> tuple[dict[str, str], dict[str, int]]:
    """Create an auditable ID -> near -> far stream without reusing articles."""

    shifts = {document_id: "id" for document_id in document_partitions}
    development = _stable_rank(
        [document_id for document_id, role in document_partitions.items() if role == "development"],
        seed=seed,
    )
    development_groups = np.array_split(np.asarray(development, dtype=object), 2)
    for document_id in development_groups[0].tolist():
        shifts[str(document_id)] = "near"
    for document_id in development_groups[1].tolist():
        shifts[str(document_id)] = "far"

    stream = _stable_rank(
        [document_id for document_id, split in deployment_splits.items() if split == "deployment_stream"],
        seed=seed + 1,
    )
    stream_groups = np.array_split(np.asarray(stream, dtype=object), 4)
    for stage, group in zip(("id", "near", "far", "far"), stream_groups, strict=True):
        for document_id in group.tolist():
            shifts[str(document_id)] = stage
    stream_order = {str(document_id): index for index, document_id in enumerate(stream)}

    for document_id, split in deployment_splits.items():
        if split != "deployment_stream":
            shifts[str(document_id)] = "far"
    return shifts, stream_order


def _stable_rank(keys: Sequence[str], *, seed: int) -> list[str]:
    return sorted(
        (str(key) for key in keys),
        key=lambda key: hashlib.sha256(f"{seed}::{key}".encode("utf-8")).hexdigest(),
    )


def _apply_covariate_shift(source_text: str, *, shift_type: str) -> str:
    """Apply a label-preserving synthetic style/length shift for baseline runs."""

    normalized = str(source_text).strip()
    if shift_type == "id":
        return f"{_FORMAT_MARKER}\n{normalized}"
    if shift_type == "near":
        return f"{(_FORMAT_MARKER + ' | ') * 4}\n{normalized}"
    if shift_type == "far":
        marker = (_FORMAT_MARKER + " | ") * 24
        return f"{marker}\n{normalized}\n{normalized}"
    raise ValueError(f"Unsupported SummEval shift type: {shift_type!r}")


def _metadata(
    *,
    raw_rows: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    output_path: Path,
    dataset_name: str,
    split: str,
    document_partition_fractions: dict[str, float],
    training_document_split_fractions: dict[str, float],
    deployment_document_split_fractions: dict[str, float],
    candidate_system_ids: Sequence[str] | None,
    seed: int,
) -> dict[str, Any]:
    split_counts = _counts(row["split"] for row in prepared)
    query_counts = _counts(row["query_id"] for row in prepared)
    audit_document_group_counts = _counts(row["audit_document_group_id"] for row in prepared)
    label_counts = _counts(str(row["label"]) for row in prepared)
    return {
        "artifact_type": "llm_judge_ood_summeval_prepared_metadata",
        "dataset_name": dataset_name,
        "source": "Hugging Face public dataset",
        "split": split,
        "output_path": str(output_path),
        "raw_input_documents": len(raw_rows),
        "prepared_rows": len(prepared),
        "quality_dimensions": list(SUMMEVAL_DIMENSIONS),
        "ood_definition": "document_distribution",
        "ood_protocol": "controlled_label_preserving_covariate_shift",
        "shift_types": list(SUMMEVAL_SHIFT_TYPES),
        "shift_type_counts": _counts(
            {str(row["input_document_id"]): str(row["document_shift_type"]) for row in prepared}.values()
        ),
        "stream_protocol": "ID_then_near_then_two_far_stages",
        "candidate_system_ids": list(candidate_system_ids) if candidate_system_ids is not None else "all_available",
        "candidate_system_note": "Candidate systems are Judge provenance and never define OOD partitions.",
        "label_rule": "Mean human score per system and dimension, rounded to ordinal 1-5.",
        "prompt_template_version": "summeval_source_candidate_v2",
        "seed": int(seed),
        "document_partition_fractions": document_partition_fractions,
        "training_document_split_fractions": training_document_split_fractions,
        "deployment_document_split_fractions": deployment_document_split_fractions,
        "split_counts": split_counts,
        "query_counts": query_counts,
        "audit_document_group_counts": audit_document_group_counts,
        "label_counts": label_counts,
        "sample_id_overlap_checks": _overlap_checks(prepared),
        "input_document_id_overlap_checks": _input_document_overlap_checks(prepared),
        "document_partition_counts": _counts(
            {str(row["input_document_id"]): str(row["document_distribution_role"]) for row in prepared}.values()
        ),
        "selection_role_counts": _counts(row["selection_role"] for row in prepared),
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        key = str(value)
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


def _overlap_checks(rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    by_split: dict[str, set[str]] = {}
    for row in rows:
        by_split.setdefault(str(row["split"]), set()).add(str(row["sample_id"]))
    checks: dict[str, bool] = {}
    names = sorted(by_split)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            checks[f"{left}__{right}"] = bool(by_split[left] & by_split[right])
    return checks


def _input_document_overlap_checks(rows: Sequence[dict[str, Any]]) -> dict[str, bool]:
    primary_splits = {
        "training_train",
        "training_validation",
        "training_calibration",
        "training_guard",
        "training_test",
        "development",
        "deployment_stream",
        "deployment_probe",
        "deployment_adapt",
        "deployment_gate",
        "deployment_future_test",
    }
    by_split: dict[str, set[str]] = {}
    for row in rows:
        split = str(row["split"])
        if split in primary_splits:
            by_split.setdefault(split, set()).add(str(row["input_document_id"]))
    checks: dict[str, bool] = {}
    names = sorted(by_split)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            checks[f"{left}__{right}"] = bool(by_split[left] & by_split[right])
    return checks


def _resolve_fraction_mapping(
    supplied: dict[str, float] | None,
    *,
    default: dict[str, float],
    name: str,
) -> dict[str, float]:
    values = dict(default if supplied is None else supplied)
    if set(values) != set(default):
        missing = sorted(set(default) - set(values))
        unexpected = sorted(set(values) - set(default))
        raise ValueError(f"{name} must use exactly {sorted(default)}; missing={missing}, unexpected={unexpected}")
    normalized = {str(key): float(value) for key, value in values.items()}
    if any(value <= 0.0 for value in normalized.values()):
        raise ValueError(f"{name} values must all be positive")
    return normalized


def _load_split_config(path: str | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("split config must be a JSON object")
    allowed = {
        "document_partition_fractions",
        "training_document_split_fractions",
        "deployment_document_split_fractions",
    }
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"split config has unsupported keys: {unexpected}")
    output: dict[str, dict[str, float]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            raise ValueError(f"split config field {key!r} must be an object")
        output[str(key)] = {str(name): float(weight) for name, weight in value.items()}
    return output


if __name__ == "__main__":
    main(sys.argv[1:])
