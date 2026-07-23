#!/usr/bin/env python3
"""Train four source-cell FLASK classification heads and evaluate 4×4."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_judge_ood.flask_comparison import (
    CLASSES,
    cell_id,
    cell_sort_key,
    integer_score,
    metrics_from_predictions,
    read_jsonl,
    row_cell,
    slug,
    write_csv,
    write_json,
    write_jsonl,
)
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/direct_and_features/b_space_with_direct_judge.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/split_manifest.json"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/direct_and_features/strict_final_prelogit_features.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/classification_head"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--pca-dim", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.rows, args.split_manifest)
    features = load_aligned_features(args.features, rows)
    labels = np.asarray([integer_score(row["ground_truth"]) for row in rows], dtype=np.int8)
    splits = np.asarray([str(row["split"]) for row in rows])
    cells = tuple(sorted({row_cell(row) for row in rows}, key=cell_sort_key))
    if len(cells) != 4:
        raise ValueError(f"Expected exactly four source/target cells, got {len(cells)}")

    all_predictions: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    model_summaries: list[dict[str, Any]] = []
    for source_cell in cells:
        source_id = cell_id(*source_cell)
        source_mask = np.asarray([row_cell(row) == source_cell for row in rows], dtype=bool)
        train_mask = source_mask & (splits == "train")
        validation_mask = source_mask & (splits == "validation")
        if not train_mask.any() or not validation_mask.any():
            raise ValueError(f"Source cell {source_id} is missing train or validation rows")
        train_labels = labels[train_mask]
        if len(set(train_labels.tolist())) < 2:
            raise ValueError(f"Source cell {source_id} has fewer than two labels in train split")
        query_ids = np.full(len(rows), source_id, dtype=object)
        config = LinearJudgeConfig(
            method="linear",
            representation="last_layer",
            pca_dim=int(args.pca_dim),
            class_values=CLASSES,
            seed=int(args.seed),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            patience=int(args.patience),
            device=str(args.device),
            class_weight="balanced",
            head_sharing="shared",
        )
        model = PerQueryLinearJudge(config).fit(
            features,
            labels,
            query_ids,
            train_mask=train_mask,
            validation_mask=validation_mask,
        )
        output = model.predict_output(features, query_ids)
        predictions = output.classes[np.argmax(output.probabilities, axis=1)].astype(np.int8)
        source_dir = args.output_dir / f"source_{slug(source_cell[0])}__{slug(source_cell[1])}"
        source_dir.mkdir(parents=True, exist_ok=True)
        model_path = source_dir / "classification_head.joblib"
        if model_path.exists() and not args.overwrite:
            raise FileExistsError(f"Model already exists: {model_path}; pass --overwrite")
        model.save(model_path)
        source_predictions = []
        for row, prediction, probabilities in zip(rows, predictions.tolist(), output.probabilities.tolist(), strict=True):
            item = {
                "method": "classification_head",
                "source_cell_id": source_id,
                "target_cell_id": cell_id(*row_cell(row)),
                "b_id": row["b_id"],
                "split": row["split"],
                "ground_truth": integer_score(row["ground_truth"]),
                "predicted_score": int(prediction),
                "probabilities": probabilities,
            }
            source_predictions.append(item)
            all_predictions.append(item)
        write_jsonl(source_dir / "predictions.jsonl", source_predictions)

        validation_metric = metrics_from_predictions(
            labels[validation_mask].tolist(),
            predictions[validation_mask].tolist(),
        )
        model_summaries.append(
            {
                "source_cell_id": source_id,
                "model_path": str(model_path),
                "train_rows": int(train_mask.sum()),
                "validation_rows": int(validation_mask.sum()),
                "train_label_counts": label_counts(labels[train_mask]),
                "validation_metrics": validation_metric,
                "model_metadata": model.to_metadata(),
            }
        )
        for target_cell in cells:
            target_id = cell_id(*target_cell)
            target_test_mask = np.asarray(
                [row_cell(row) == target_cell and str(row["split"]) == "test" for row in rows],
                dtype=bool,
            )
            metric = metrics_from_predictions(
                labels[target_test_mask].tolist(),
                predictions[target_test_mask].tolist(),
            )
            metrics_rows.append(
                {
                    "method": "classification_head",
                    "source_cell_id": source_id,
                    "target_cell_id": target_id,
                    "split": "test",
                    **metric,
                }
            )

    write_jsonl(args.output_dir / "classification_head_predictions.jsonl", all_predictions)
    write_csv(args.output_dir / "classification_head_4x4_metrics.csv", metrics_rows)
    summary = {
        "artifact_type": "flask_comparison_four_source_classification_heads_v1",
        "source_rows": str(args.rows),
        "split_manifest": str(args.split_manifest),
        "features": str(args.features),
        "cells": [cell_id(*cell) for cell in cells],
        "head_count": len(model_summaries),
        "test_evaluations": len(metrics_rows),
        "models": model_summaries,
        "metrics": metrics_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "test_evaluations": len(metrics_rows)}, ensure_ascii=False, indent=2))


def load_rows(rows_path: Path, split_manifest_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(rows_path)
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    row_splits = {str(key): str(value) for key, value in manifest.get("row_splits", {}).items()}
    if set(row_splits) != {str(row["b_id"]) for row in rows}:
        raise ValueError("Split manifest row ids do not match --rows")
    return [{**row, "split": row_splits[str(row["b_id"])]} for row in rows]


def load_aligned_features(path: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float16)
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
    index = {value: idx for idx, value in enumerate(sample_ids.tolist())}
    missing = [str(row["b_id"]) for row in rows if str(row["b_id"]) not in index]
    if missing:
        raise ValueError(f"Feature file is missing {len(missing)} selected rows")
    return features[np.asarray([index[str(row["b_id"])] for row in rows], dtype=np.int64)]


def label_counts(values: np.ndarray) -> dict[str, int]:
    return {str(label): int(np.sum(values == label)) for label in CLASSES}


if __name__ == "__main__":
    main()

