#!/usr/bin/env python3
"""Extract 3x3 score-position B-space features and train nine linear heads.

This is the fast follow-up to the single-cell score-token diagnostic:

* filter the documented 3x3 source cells;
* teacher-force each existing Direct-Judge completion;
* capture hidden states at the token immediately before the score digit and at
  the score digit itself for layers 23 and 32;
* train the same nine question-group linear classification heads for both
  feature variants against GPT-4 ground-truth labels.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.shared.metrics import judge_metrics
from src.models.extract_hidden import QWEN3_5_4B_HIDDEN_SIZE, load_qwen_model


SOURCE_DOMAINS = ("Humanities", "Language", "Social Science")
SOURCE_SKILLS = ("Comprehension", "Factuality", "Logical Correctness")
CLASSES = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b-space",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_with_direct_judge.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/score_token_hidden_3x3_heads"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("/home/zeus/models/qwen3.5-4b"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="flash_attention_2")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-question-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        row for row in load_jsonl(args.b_space)
        if str(row["domain_ids"][0]) in SOURCE_DOMAINS and str(row["task_id"]) in SOURCE_SKILLS
    ]
    if not rows:
        raise ValueError("3x3 B-space filter returned no rows")
    direct = load_module("flask_direct_judge_helpers", ROOT / "scripts/llm_judge_ood/49_run_flask_minimal_direct_judge.py")
    one_cell_probe = load_module("score_token_cell_probe", ROOT / "scripts/llm_judge_ood/58_score_token_hidden_cell_probe.py")

    tokenizer, model, device = load_qwen_model(
        args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = one_cell_probe.prepare_teacher_forcing_records(
        rows=rows,
        tokenizer=tokenizer,
        direct=direct,
        max_prompt_length=int(args.max_prompt_length),
    )
    variants = one_cell_probe.extract_features(
        records=records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_blocks=(22, 31),
        batch_size=int(args.batch_size),
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    labels = np.asarray([int(row["ground_truth"]) for row in rows], dtype=np.int8)
    direct_scores = np.asarray([int(row["direct_score"]) for row in rows], dtype=np.int8)
    sample_ids = np.asarray([str(row["b_id"]) for row in rows])
    question_ids = np.asarray([str(row["base_id"]) for row in rows])
    domain_ids = np.asarray([str(row["domain_ids"][0]) for row in rows])
    task_ids = np.asarray([str(row["task_id"]) for row in rows])

    feature_paths: dict[str, str] = {}
    train_summaries: dict[str, Any] = {}
    train_module = load_module("flask_3x3_cpu_heads", ROOT / "scripts/llm_judge_ood/53_train_flask_3x3_cpu_heads.py")
    train_args = argparse.Namespace(
        seed=int(args.seed),
        train_question_fraction=float(args.train_question_fraction),
        epochs=int(args.epochs),
        batch_size=int(args.head_batch_size),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        overwrite=bool(args.overwrite),
    )

    for variant_name, features in variants.items():
        feature_path = args.output_dir / f"{variant_name}_b_space_features.npz"
        if feature_path.exists() and not args.overwrite:
            raise FileExistsError(f"Feature file already exists: {feature_path}; pass --overwrite")
        np.savez_compressed(
            feature_path,
            features=features.astype(np.float16),
            sample_ids=sample_ids,
            labels=labels,
            direct_scores=direct_scores,
            query_ids=question_ids,
            domain_ids=domain_ids,
            task_ids=task_ids,
            metadata_json=np.asarray(json.dumps({
                "artifact_type": f"flask_3x3_{variant_name}_b_space_hidden_v1",
                "feature_scope": variant_name,
                "source_domains": SOURCE_DOMAINS,
                "source_skills": SOURCE_SKILLS,
                "layers": [23, 32],
                "shape": [len(rows), 2, QWEN3_5_4B_HIDDEN_SIZE],
                "teacher_forced_existing_direct_judge_completion": True,
                "score_digit_position": "located in direct_judge_raw_completion",
            }, ensure_ascii=False)),
        )
        feature_paths[variant_name] = str(feature_path)
        variant_dir = args.output_dir / f"{variant_name}_heads"
        variant_dir.mkdir(parents=True, exist_ok=True)
        train_args.output_dir = variant_dir
        head_summaries = []
        for domain in SOURCE_DOMAINS:
            for skill in SOURCE_SKILLS:
                head_summaries.append(train_module.train_one_head(
                    rows=rows,
                    features=features,
                    domain=domain,
                    skill=skill,
                    args=train_args,
                ))
        aggregate = {
            "artifact_type": f"flask_3x3_{variant_name}_linear_heads_v1",
            "source_b_space": str(args.b_space),
            "source_features": str(feature_path),
            "feature_scope": variant_name,
            "heads": head_summaries,
            "head_count": len(head_summaries),
            "macro_id_test": macro_id_test(head_summaries),
            "direct_judge_same_splits": direct_metrics_same_splits(head_summaries, rows),
            "device": "cpu",
        }
        write_json(variant_dir / "summary.json", aggregate)
        train_summaries[variant_name] = aggregate

    summary = {
        "artifact_type": "flask_3x3_score_token_hidden_heads_run_v1",
        "rows": len(rows),
        "source_domains": SOURCE_DOMAINS,
        "source_skills": SOURCE_SKILLS,
        "feature_paths": feature_paths,
        "variants": {
            name: {
                "summary_path": str(args.output_dir / f"{name}_heads" / "summary.json"),
                "macro_id_test": value["macro_id_test"],
                "direct_judge_same_splits": value["direct_judge_same_splits"],
            }
            for name, value in train_summaries.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def macro_id_test(head_summaries: list[dict[str, Any]]) -> dict[str, float]:
    rows = np.asarray([item["id_test_metrics"]["rows"] for item in head_summaries], dtype=float)
    total_rows = float(rows.sum())
    output: dict[str, float] = {}
    for key in ("mae", "exact_accuracy", "plus_minus_1_accuracy"):
        output[key] = float(sum(item["id_test_metrics"][key] * item["id_test_metrics"]["rows"] for item in head_summaries) / total_rows)
    output["quadratic_weighted_kappa_macro"] = float(np.mean([item["id_test_metrics"]["quadratic_weighted_kappa"] for item in head_summaries]))
    output["rows"] = int(total_rows)
    return output


def direct_metrics_same_splits(head_summaries: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, float]:
    rows_by_cell = {
        (str(row["domain_ids"][0]), str(row["task_id"])): [] for row in rows
    }
    for row in rows:
        rows_by_cell[(str(row["domain_ids"][0]), str(row["task_id"]))].append(row)
    weighted: dict[str, float] = {}
    total_rows = 0
    qwk_values = []
    for item in head_summaries:
        cell_rows = rows_by_cell[(item["source_domain"], item["source_skill"])]
        test_questions = set(str(value) for value in item["split"]["test_question_ids"])
        test_rows = [row for row in cell_rows if str(row["base_id"]) in test_questions]
        labels = np.asarray([int(row["ground_truth"]) for row in test_rows], dtype=np.int8)
        direct_scores = np.asarray([int(row["direct_score"]) for row in test_rows], dtype=np.int8)
        metrics = metric_row(labels, direct_scores)
        total_rows += metrics["rows"]
        qwk_values.append(metrics["quadratic_weighted_kappa"])
        for key in ("mae", "exact_accuracy", "plus_minus_1_accuracy"):
            weighted[key] = weighted.get(key, 0.0) + metrics[key] * metrics["rows"]
    for key in ("mae", "exact_accuracy", "plus_minus_1_accuracy"):
        weighted[key] = float(weighted[key] / total_rows)
    weighted["quadratic_weighted_kappa_macro"] = float(np.mean(qwk_values))
    weighted["rows"] = int(total_rows)
    return weighted


def metric_row(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    value = judge_metrics(labels, predictions, class_values=CLASSES)
    return {
        "rows": int(len(labels)),
        "mae": float(value["mae"]),
        "exact_accuracy": float(value["accuracy"]),
        "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
        "quadratic_weighted_kappa": float(value["qwk"]),
    }


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
