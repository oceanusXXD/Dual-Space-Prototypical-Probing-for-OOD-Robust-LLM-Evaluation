#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_json, write_json
from src.llm_judge_ood.model.judge import JudgeTrainingConfig
from src.llm_judge_ood.model.selection import JudgeSelectionConfig, select_source_judge
from src.llm_judge_ood.shared.feature_store import load_hidden_feature_store, record_fingerprint
from src.llm_judge_ood.shared.metrics import macro_query_judge_metrics, normalize_label_array
from src.llm_judge_ood.shared.schema import load_judge_records
from src.models.extract_hidden import (
    QWEN3_5_4B_HIDDEN_SIZE,
    QWEN3_5_4B_MODEL_ID,
    QWEN3_5_4B_NUM_LAYERS,
    QWEN3_5_4B_REVISION,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare raw-document frozen Qwen hidden caches on source/development splits."
    )
    parser.add_argument("--config", default="configs/llm_judge_ood/llm_judge_ood_summeval_gpu.json")
    parser.add_argument("--input", default=None)
    parser.add_argument("--cache", action="append", required=True, help="NAME=PATH; repeat for every cache")
    parser.add_argument("--baseline", required=True, help="Cache NAME used as the input-document baseline")
    parser.add_argument("--promotion-threshold", type=float, default=0.02)
    parser.add_argument("--output-dir", default="artifacts/llm_judge_ood_summeval/input_document_model_comparison")
    args = parser.parse_args()

    payload = read_json(args.config)
    input_path = args.input or payload["input_paths"][0]
    records = load_judge_records([input_path])
    training_fit_splits = {
        str(value) for value in payload.get("training_document_train_splits", ("training_train",))
    }
    training_validation_splits = {
        str(value) for value in payload.get("training_document_validation_splits", ("training_validation",))
    }
    development_splits = {
        str(value) for value in payload.get("development_document_splits", ("development",))
    }
    record_splits = np.asarray([record.split for record in records]).astype(str)
    record_roles = np.asarray([record.document_distribution_role for record in records]).astype(str)
    training_role_mask = record_roles == "training"
    development_role_mask = record_roles == "development"
    training_fit_mask = training_role_mask & np.isin(record_splits, sorted(training_fit_splits))
    training_validation_mask = training_role_mask & np.isin(record_splits, sorted(training_validation_splits))
    development_mask = development_role_mask & np.isin(record_splits, sorted(development_splits))
    if not training_fit_mask.any() or not training_validation_mask.any() or not development_mask.any():
        raise ValueError(
            "Model comparison requires non-empty training fit, training validation, and development document pools"
        )
    selected_indices = np.flatnonzero(training_fit_mask | training_validation_mask | development_mask)
    selected_records = [records[index] for index in selected_indices.tolist()]
    labels = normalize_label_array([record.label for record in selected_records])
    query_ids = np.asarray([record.query_id for record in selected_records]).astype(str)
    train_mask = training_fit_mask[selected_indices]
    validation_mask = training_validation_mask[selected_indices]
    development_mask = development_mask[selected_indices]
    judge_config = JudgeTrainingConfig(**payload.get("judge", {}))
    selection_payload = dict(payload.get("judge_selection", {}))
    for key in (
        "preprocess_methods",
        "neural_losses",
        "neural_seeds",
        "ridge_alphas",
        "linear_learning_rates",
        "linear_cs",
    ):
        if key in selection_payload:
            selection_payload[key] = tuple(selection_payload[key])
    selection_config = JudgeSelectionConfig(**selection_payload)
    rows: list[dict[str, Any]] = []
    expected_fingerprint = record_fingerprint(records, feature_scope="input_document")
    for cache_spec in args.cache:
        name, path = _parse_cache(cache_spec)
        store = load_hidden_feature_store(path)
        cache_metadata = store.metadata.get("cache_metadata")
        required_metadata = {
            "artifact_type": "llm_judge_ood_frozen_qwen_hidden_features",
            "feature_scope": "input_document",
            "model_id": QWEN3_5_4B_MODEL_ID,
            "model_revision": QWEN3_5_4B_REVISION,
            "model_revision_requested": QWEN3_5_4B_REVISION,
            "model_type": "qwen3_5_text",
            "num_model_layers": QWEN3_5_4B_NUM_LAYERS,
            "model_hidden_size": QWEN3_5_4B_HIDDEN_SIZE,
            "hidden_state_count": QWEN3_5_4B_NUM_LAYERS + 1,
            "embedding_state_included": True,
            "max_length": 2048,
            "pooling": "masked_mean",
            "pooling_scope": "input_document",
            "pooling_formula": "sum(hidden_state * attention_mask) / sum(attention_mask)",
            "pooling_mask_source": "tokenizer_attention_mask",
            "pooling_excludes_padding": True,
            "prompt_template_version": "raw_input_document_v1",
            "labels_in_prompt": False,
            "model_eval": True,
            "requires_grad": False,
            "backbone_frozen": True,
            "dataset_fingerprint": expected_fingerprint,
        }
        mismatches = {
            key: {"expected": value, "actual": cache_metadata.get(key) if isinstance(cache_metadata, dict) else None}
            for key, value in required_metadata.items()
            if not isinstance(cache_metadata, dict) or cache_metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Cache {name!r} violates the frozen raw-document Qwen contract: {mismatches}"
            )
        identity = cache_metadata.get("model_identity_evidence")
        if (
            not isinstance(identity, dict)
            or identity.get("repo_id") != QWEN3_5_4B_MODEL_ID
            or identity.get("revision") != QWEN3_5_4B_REVISION
        ):
            raise ValueError(
                f"Cache {name!r} lacks exact Qwen/Qwen3.5-4B revision identity evidence"
            )
        if store.input_document_ids is None:
            raise ValueError(f"Cache {name!r} is missing input_document_ids")
        row_by_id = {
            str(document_id): index
            for index, document_id in enumerate(np.asarray(store.input_document_ids).astype(str).tolist())
        }
        if len(row_by_id) != len(store.input_document_ids):
            raise ValueError(f"Cache {name!r} has duplicate input_document_ids")
        missing = [
            record.input_document_id
            for record in selected_records
            if record.input_document_id not in row_by_id
        ]
        if missing:
            raise ValueError(
                f"Cache {name!r} misses {len(set(missing))} required input documents, first={missing[:5]}"
            )
        features = np.stack(
            [store.features[row_by_id[record.input_document_id]] for record in selected_records],
            axis=0,
        ).astype(np.float32)
        selection = select_source_judge(
            raw_features=features,
            labels=labels,
            query_ids=query_ids,
            train_mask=train_mask,
            validation_mask=validation_mask,
            base_config=judge_config,
            selection_config=selection_config,
        )
        predictions = selection.model.predict(selection.processed_features, query_ids)
        probabilities = selection.model.predict_proba(selection.processed_features, query_ids)
        validation = macro_query_judge_metrics(
            labels[validation_mask],
            predictions[validation_mask],
            query_ids[validation_mask],
            probabilities=probabilities[validation_mask],
            class_values=selection.model.classes_,
        )
        development = macro_query_judge_metrics(
            labels[development_mask],
            predictions[development_mask],
            query_ids[development_mask],
            probabilities=probabilities[development_mask],
            class_values=selection.model.classes_,
        )
        training_validation_qwk = float(validation["macro"]["qwk"])
        development_qwk = float(development["macro"]["qwk"])
        rows.append(
            {
                "name": name,
                "path": path,
                "shape": list(features.shape),
                "selected_judge": selection.selected_candidate["name"],
                "training_validation_macro_qwk": training_validation_qwk,
                "development_macro_qwk": development_qwk,
                "mean_training_validation_development_qwk": float(
                    (training_validation_qwk + development_qwk) / 2.0
                ),
                "training_validation_macro_mae": float(validation["macro"]["mae"]),
                "development_macro_mae": float(development["macro"]["mae"]),
                "neural_beats_baseline": bool(selection.summary["neural_beats_baseline"]),
                "device": selection.model.to_metadata().get("device"),
                "selection": selection.summary,
            }
        )
    by_name = {str(row["name"]): row for row in rows}
    if args.baseline not in by_name:
        raise ValueError(f"Baseline cache {args.baseline!r} was not supplied")
    baseline_score = float(by_name[args.baseline]["mean_training_validation_development_qwk"])
    decisions: dict[str, Any] = {}
    for name, row in by_name.items():
        improvement = float(row["mean_training_validation_development_qwk"]) - baseline_score
        decisions[name] = {
            "improvement_over_baseline": improvement,
            "promote_to_full_extraction": bool(
                name != args.baseline and improvement >= float(args.promotion_threshold)
            ),
        }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "artifact_type": "llm_judge_ood_input_document_model_comparison",
        "input": str(input_path),
        "selection_document_splits": {
            "training_fit": sorted(training_fit_splits),
            "training_validation": sorted(training_validation_splits),
            "development": sorted(development_splits),
        },
        "deployment_documents_used": False,
        "baseline": args.baseline,
        "promotion_threshold": float(args.promotion_threshold),
        "results": rows,
        "decisions": decisions,
    }
    write_json(output_dir / "summary.json", summary)
    flat_rows = [
        {key: value for key, value in row.items() if key != "selection"}
        | decisions[str(row["name"])]
        for row in rows
    ]
    pd.DataFrame(flat_rows).to_csv(output_dir / "model_comparison.csv", index=False)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "summary.json"),
                "results": flat_rows,
                "decisions": decisions,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _parse_cache(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError("--cache values must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError("--cache values must use a non-empty NAME=PATH")
    return name, path


if __name__ == "__main__":
    main()
