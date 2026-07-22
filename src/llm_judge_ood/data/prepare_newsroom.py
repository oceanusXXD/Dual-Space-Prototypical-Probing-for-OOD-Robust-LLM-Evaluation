from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from src.common.io import write_json, write_jsonl

NEWSROOM_DIMENSIONS = ("coherence", "fluency", "informativeness", "relevance")
NEWSROOM_RATING_COLUMNS = {
    "coherence": "CoherenceRating",
    "fluency": "FluencyRating",
    "informativeness": "InformativenessRating",
    "relevance": "RelevanceRating",
}
NEWSROOM_QUERY_TEXT = {
    "coherence": "Rate the coherence of the candidate summary on a 1-5 scale.",
    "fluency": "Rate the fluency and grammatical quality of the candidate summary on a 1-5 scale.",
    "informativeness": "Rate how informative the candidate summary is about the source article on a 1-5 scale.",
    "relevance": "Rate how relevant the candidate summary is to the source article on a 1-5 scale.",
}

DEFAULT_DOCUMENT_PARTITION_FRACTIONS = {
    "training": 52,
    "development": 14,
    "deployment": 34,
}
DEFAULT_TRAINING_DOCUMENT_SPLIT_FRACTIONS = {
    "training_train": 35,
    "training_validation": 4,
    "training_calibration": 4,
    "training_guard": 3,
    "training_test": 6,
}
DEFAULT_DEPLOYMENT_DOCUMENT_SPLIT_FRACTIONS = {
    "deployment_stream": 16,
    "deployment_probe": 6,
    "deployment_adapt": 4,
    "deployment_gate": 4,
    "deployment_future_test": 4,
}
PRIMARY_SPLITS = (
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
)


def load_newsroom_human_eval_csv(input_path: str | Path) -> list[dict[str, str]]:
    """Load the 3-rater Newsroom human-evaluation CSV."""

    with Path(input_path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Newsroom human-evaluation CSV is empty: {input_path}")
    required = {
        "ArticleID",
        "System",
        "ArticleText",
        "SystemSummary",
        "ArticleTitle",
        *NEWSROOM_RATING_COLUMNS.values(),
    }
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Newsroom human-evaluation CSV is missing columns: {missing}")
    return rows


def prepare_newsroom_rows_from_human_eval(
    annotation_rows: Iterable[dict[str, Any]],
    *,
    document_partition_fractions: dict[str, float] | None = None,
    training_document_split_fractions: dict[str, float] | None = None,
    deployment_document_split_fractions: dict[str, float] | None = None,
    seed: int = 42,
    expected_raters_per_system: int = 3,
) -> list[dict[str, Any]]:
    """Aggregate Newsroom human ratings and assign document-level OOD splits.

    The public human-evaluation mirror contains one row per rater. Ratings are
    averaged by article, summary system, and dimension, then rounded to the
    ordinal 1-5 label used by the Judge. Splits are assigned to article IDs
    before expanding the four dimensions, so no article crosses a split.
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

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in annotation_rows:
        article_id = str(raw.get("ArticleID") or "").strip()
        system = str(raw.get("System") or "").strip()
        article_text = _clean_text(raw.get("ArticleText"))
        summary = _clean_text(raw.get("SystemSummary"))
        title = _clean_text(raw.get("ArticleTitle"))
        if not article_id or not system or not article_text or not summary:
            raise ValueError(f"Newsroom row is missing ArticleID/System/text/summary: {raw}")
        normalized = {
            "article_id": article_id,
            "system": system,
            "article_text": article_text,
            "summary": summary,
            "title": title,
            "source_row": raw,
        }
        for dimension, column in NEWSROOM_RATING_COLUMNS.items():
            try:
                rating = float(raw[column])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid {column} for article={article_id}, system={system}: {raw.get(column)!r}") from exc
            if not 1.0 <= rating <= 5.0:
                raise ValueError(f"{column} must be in [1, 5], got {rating} for article={article_id}, system={system}")
            normalized[dimension] = rating
        grouped[(article_id, system)].append(normalized)

    if not grouped:
        raise ValueError("No Newsroom annotation groups were found")
    for key, rows in grouped.items():
        if len(rows) != expected_raters_per_system:
            raise ValueError(
                f"Expected exactly {expected_raters_per_system} raters for article/system={key}, got {len(rows)}"
            )
        if len({row["article_text"] for row in rows}) != 1 or len({row["summary"] for row in rows}) != 1:
            raise ValueError(f"Article text or system summary differs across raters for article/system={key}")

    article_ids = sorted({article_id for article_id, _ in grouped})
    document_roles = _stable_partition(article_ids, partition_fractions, seed=seed)
    training_ids = [article_id for article_id in article_ids if document_roles[article_id] == "training"]
    deployment_ids = [article_id for article_id in article_ids if document_roles[article_id] == "deployment"]
    training_splits = _stable_partition(training_ids, training_fractions, seed=seed + 1)
    deployment_splits = _stable_partition(deployment_ids, deployment_fractions, seed=seed + 2)

    rows: list[dict[str, Any]] = []
    for (article_id, system), annotations in sorted(grouped.items()):
        role = document_roles[article_id]
        if role == "training":
            split = training_splits[article_id]
            selection_role = "training_fit" if split == "training_train" else "training_holdout"
        elif role == "development":
            split = "development"
            selection_role = "development_only"
        else:
            split = deployment_splits[article_id]
            selection_role = "deployment_blind"
        representative = annotations[0]
        candidate_system_id = f"newsroom_system_{system}"
        for dimension in NEWSROOM_DIMENSIONS:
            ratings = [float(row[dimension]) for row in annotations]
            mean_rating = float(np.mean(ratings))
            label = _round_score_to_ordinal(mean_rating)
            sample_id = f"newsroom::{article_id}::{system}::{dimension}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "id": sample_id,
                    "dataset": "newsroom",
                    "dataset_source": "KnutJaegersberg/newsroom-human-eval",
                    "query_id": dimension,
                    "query_text": NEWSROOM_QUERY_TEXT[dimension],
                    "document_text": f"Input document: {representative['article_text']}\nCandidate response: {representative['summary']}",
                    "label": label,
                    "groundtruth": label,
                    "raw_score": mean_rating,
                    "rater_scores": ratings,
                    "rater_count": len(ratings),
                    "split": split,
                    "judge_provenance_id": "newsroom_human_eval",
                    "base_document_id": article_id,
                    "input_document_id": article_id,
                    "input_document_text": representative["article_text"],
                    "document_distribution_role": role,
                    "audit_document_group_id": role,
                    "candidate_system_id": candidate_system_id,
                    "candidate_system_name": system,
                    "article_title": representative["title"],
                    "selection_role": selection_role,
                    "prompt_template_version": "newsroom_source_candidate_v1",
                }
            )
    return rows


def write_newsroom_prepared(
    *,
    input_path: str | Path,
    output_path: str | Path,
    split_config_path: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    config = _load_split_config(split_config_path)
    annotation_rows = load_newsroom_human_eval_csv(input_path)
    prepared = prepare_newsroom_rows_from_human_eval(annotation_rows, seed=seed, **config)
    output = Path(output_path)
    write_jsonl(output, prepared)
    split_paths: dict[str, str] = {}
    split_dir = output.parent / f"{output.stem}_splits"
    for split in PRIMARY_SPLITS:
        split_output = split_dir / f"{split}.jsonl"
        split_rows = (row for row in prepared if row["split"] == split)
        write_jsonl(split_output, split_rows)
        split_paths[split] = str(split_output)
    metadata = _metadata(
        input_path=Path(input_path),
        output_path=output,
        annotation_rows=annotation_rows,
        prepared=prepared,
        split_paths=split_paths,
        seed=seed,
        document_partition_fractions=_resolve_fraction_mapping(
            config.get("document_partition_fractions"),
            default=DEFAULT_DOCUMENT_PARTITION_FRACTIONS,
            name="document_partition_fractions",
        ),
        training_document_split_fractions=_resolve_fraction_mapping(
            config.get("training_document_split_fractions"),
            default=DEFAULT_TRAINING_DOCUMENT_SPLIT_FRACTIONS,
            name="training_document_split_fractions",
        ),
        deployment_document_split_fractions=_resolve_fraction_mapping(
            config.get("deployment_document_split_fractions"),
            default=DEFAULT_DEPLOYMENT_DOCUMENT_SPLIT_FRACTIONS,
            name="deployment_document_split_fractions",
        ),
    )
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare Newsroom human-evaluation rows for LLM Judge document OOD.")
    parser.add_argument(
        "--input",
        default="datasets/raw/newsroom/human_eval/newsroom_human_eval.csv",
        help="Newsroom human-evaluation CSV (three ratings per article/system).",
    )
    parser.add_argument(
        "--output",
        default="artifacts/llm_judge_ood_newsroom/newsroom_prepared_document_ood_v1.jsonl",
    )
    parser.add_argument(
        "--split-config",
        default="configs/llm_judge_ood/newsroom_document_split_profile.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    metadata = write_newsroom_prepared(
        input_path=args.input,
        output_path=args.output,
        split_config_path=args.split_config,
        seed=args.seed,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def _clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<p\s*/?>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _round_score_to_ordinal(value: float) -> int:
    return int(np.clip(np.rint(float(value)), 1, 5))


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


def _load_split_config(path: str | Path | None) -> dict[str, dict[str, float]]:
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
    return {str(key): {str(name): float(value) for name, value in value.items()} for key, value in payload.items()}


def _metadata(
    *,
    input_path: Path,
    output_path: Path,
    annotation_rows: Sequence[dict[str, Any]],
    prepared: Sequence[dict[str, Any]],
    split_paths: dict[str, str],
    seed: int,
    document_partition_fractions: dict[str, float],
    training_document_split_fractions: dict[str, float],
    deployment_document_split_fractions: dict[str, float],
) -> dict[str, Any]:
    split_counts = _counts(row["split"] for row in prepared)
    document_counts = {
        split: len({str(row["input_document_id"]) for row in prepared if row["split"] == split})
        for split in PRIMARY_SPLITS
    }
    by_split = {
        split: {str(row["input_document_id"]) for row in prepared if row["split"] == split}
        for split in PRIMARY_SPLITS
    }
    overlap_checks = {
        f"{left}__{right}": bool(by_split[left] & by_split[right])
        for index, left in enumerate(PRIMARY_SPLITS)
        for right in PRIMARY_SPLITS[index + 1 :]
    }
    return {
        "artifact_type": "llm_judge_ood_newsroom_prepared_metadata",
        "dataset_name": "Newsroom human evaluation",
        "dataset_source": "KnutJaegersberg/newsroom-human-eval (HF mirror of LS-Score data)",
        "official_raw_source": "https://lil.nlp.cornell.edu/resources/newsroom/newsroom-thin.tar.bz2",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "split_paths": split_paths,
        "raw_annotation_rows": len(annotation_rows),
        "prepared_rows": len(prepared),
        "input_documents": len({str(row["input_document_id"]) for row in prepared}),
        "candidate_systems": sorted({str(row["candidate_system_name"]) for row in prepared}),
        "quality_dimensions": list(NEWSROOM_DIMENSIONS),
        "ood_definition": "document_distribution",
        "label_rule": "Mean of three human ratings per article/system/dimension, rounded to ordinal 1-5.",
        "label_semantics": "human_quality_rating",
        "candidate_system_note": "Candidate systems are Judge provenance and never define OOD partitions.",
        "seed": int(seed),
        "document_partition_fractions": document_partition_fractions,
        "training_document_split_fractions": training_document_split_fractions,
        "deployment_document_split_fractions": deployment_document_split_fractions,
        "split_counts": split_counts,
        "document_counts": document_counts,
        "input_document_id_overlap_checks": overlap_checks,
        "document_partition_counts": _counts(
            {str(row["input_document_id"]): str(row["document_distribution_role"]) for row in prepared}.values()
        ),
        "annotation_rater_count_counts": _counts(row["rater_count"] for row in prepared),
        "source_sha256": _sha256(input_path),
    }


def _counts(values: Iterable[Any]) -> dict[str, int]:
    output: dict[str, int] = {}
    for value in values:
        key = str(value)
        output[key] = output.get(key, 0) + 1
    return dict(sorted(output.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main(sys.argv[1:])
