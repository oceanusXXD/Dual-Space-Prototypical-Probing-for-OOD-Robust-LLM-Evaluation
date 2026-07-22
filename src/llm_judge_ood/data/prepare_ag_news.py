from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.llm_judge_ood.data.hidden_contract import (
    build_prepared_record,
    file_sha256,
    template_sha256,
    write_prepared_contract,
)


AG_NEWS_QUERY_ID = "ag_news_topic"
AG_NEWS_QUERY_TEXT = "Classify a news article by topic."
AG_NEWS_TEMPLATE_VERSION = "ag_news_topic_input_v1"
AG_NEWS_JUDGE_TEMPLATE = "Task: news topic classification.\nNews article:\n{text}"
AG_NEWS_TEMPLATE_SHA256 = template_sha256(AG_NEWS_JUDGE_TEMPLATE)
AG_NEWS_TOPICS = ("World", "Sports", "Business", "Sci/Tech")
_EXPECTED_COUNTS = {"train": 120000, "test": 7600}


def prepare_ag_news(
    *,
    train_paths: Sequence[str | Path],
    test_paths: Sequence[str | Path],
    output_path_template: str | Path,
    folds: Sequence[int] = (0, 1, 2, 3),
    expected_sha256: dict[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    sources = {
        "train": [Path(path) for path in train_paths],
        "test": [Path(path) for path in test_paths],
    }
    if not sources["train"] or not sources["test"]:
        raise ValueError("AG News requires at least one local train and test parquet file")
    frames = {
        split: _load_parquet_files(paths, split=split) for split, paths in sources.items()
    }
    actual_counts = {split: len(frame) for split, frame in frames.items()}
    if actual_counts != _EXPECTED_COUNTS:
        raise ValueError(f"Unexpected official AG News split counts: {actual_counts}")

    requested_folds = tuple(int(fold) for fold in folds)
    if not requested_folds or any(fold not in range(4) for fold in requested_folds):
        raise ValueError("AG News folds must be selected from 0, 1, 2, 3")
    if len(set(requested_folds)) != len(requested_folds):
        raise ValueError("AG News folds must be unique")

    output_template = str(output_path_template)
    if "{fold}" not in output_template:
        raise ValueError("AG News output_path_template must include the {fold} placeholder")
    source_hashes = {
        split: {str(path): file_sha256(path) for path in paths}
        for split, paths in sources.items()
    }
    expected = dict(expected_sha256 or {})
    for split, paths in sources.items():
        configured = tuple(str(value) for value in expected.get(split, ()))
        actual = tuple(source_hashes[split][str(path)] for path in paths)
        if configured and configured != actual:
            raise ValueError(
                f"AG News {split} parquet checksums do not match the pinned config: {actual}"
            )
    fold_metadata: dict[str, Any] = {}
    for held_out_topic in requested_folds:
        rows: list[dict[str, Any]] = []
        for official_split in ("train", "test"):
            for row_index, raw in frames[official_split].iterrows():
                label = int(raw["label"])
                text = str(raw["text"])
                is_ood = label == held_out_topic
                document_id = f"ag_news::{official_split}::{int(row_index):06d}"
                rows.append(
                    build_prepared_record(
                        dataset="ag_news",
                        sample_id=document_id,
                        # A = the exact public article text only.
                        raw_text=text,
                        # B = the frozen topic task instruction plus that article.
                        judge_input_text=AG_NEWS_JUDGE_TEMPLATE.format(text=text),
                        query_id=AG_NEWS_QUERY_ID,
                        query_text=AG_NEWS_QUERY_TEXT,
                        label=label,
                        split=_prepared_split(official_split, is_ood=is_ood),
                        document_distribution_role=_document_role(
                            official_split, is_ood=is_ood
                        ),
                        audit_document_group_id=("held_out" if is_ood else "source")
                        + f"::{AG_NEWS_TOPICS[label]}",
                        document_shift_type="held_out_topic" if is_ood else "id",
                        is_document_ood=is_ood,
                        prompt_template_version=AG_NEWS_TEMPLATE_VERSION,
                        prompt_template_sha256=AG_NEWS_TEMPLATE_SHA256,
                        metadata={
                            "source_dataset": "fancyzhx/ag_news",
                            "official_split": official_split,
                            "official_row_id": int(row_index),
                            "class_index": label,
                            "label_name": AG_NEWS_TOPICS[label],
                            "held_out_topic_index": held_out_topic,
                            "held_out_topic_name": AG_NEWS_TOPICS[held_out_topic],
                            "fold_id": held_out_topic + 1,
                            "classifier_fit_eligible": official_split == "train" and not is_ood,
                            "main_ood_benchmark_eligible": official_split == "test" and is_ood,
                        },
                    )
                )
        output_path = Path(output_template.format(fold=held_out_topic + 1))
        fold_metadata[str(held_out_topic + 1)] = write_prepared_contract(
            output_path,
            rows,
            {
                "artifact_type": "llm_judge_ood_ag_news_prepared_metadata",
                "dataset_source": "Hugging Face fancyzhx/ag_news public parquet",
                "source_paths": {
                    split: [str(path) for path in paths] for split, paths in sources.items()
                },
                "source_sha256": source_hashes,
                "official_split_counts": actual_counts,
                "fold_id": held_out_topic + 1,
                "held_out_topic_index": held_out_topic,
                "held_out_topic_name": AG_NEWS_TOPICS[held_out_topic],
                "cache_reuse_contract": (
                    "all folds have identical sample_id/input_document_text/judge_input_text; "
                    "only labels, roles, and OOD audit metadata vary"
                ),
            },
        )

    return {
        "artifact_type": "llm_judge_ood_ag_news_four_fold_metadata",
        "folds": fold_metadata,
        "prompt_template_version": AG_NEWS_TEMPLATE_VERSION,
        "prompt_template_sha256": AG_NEWS_TEMPLATE_SHA256,
    }


def _load_parquet_files(paths: Sequence[Path], *, split: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if not {"text", "label"}.issubset(frame.columns):
            raise ValueError(f"AG News parquet {path} must contain text and label columns")
        frames.append(frame[["text", "label"]])
    combined = pd.concat(frames, ignore_index=True)
    if combined["text"].isna().any() or combined["label"].isna().any():
        raise ValueError(f"AG News {split} contains missing text or label values")
    labels = {int(value) for value in combined["label"].unique().tolist()}
    if labels != {0, 1, 2, 3}:
        raise ValueError(f"AG News {split} labels must be 0,1,2,3; got {sorted(labels)}")
    if any(not str(text).strip() for text in combined["text"].tolist()):
        raise ValueError(f"AG News {split} contains empty article text")
    return combined


def _prepared_split(official_split: str, *, is_ood: bool) -> str:
    if official_split == "train":
        return "held_out_train_excluded" if is_ood else "train"
    return "ood_test" if is_ood else "test"


def _document_role(official_split: str, *, is_ood: bool) -> str:
    if official_split == "train":
        return "excluded" if is_ood else "training"
    return "benchmark"
