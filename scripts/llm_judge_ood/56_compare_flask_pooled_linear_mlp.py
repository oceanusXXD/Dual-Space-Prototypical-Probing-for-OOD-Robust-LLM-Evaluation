#!/usr/bin/env python3
"""Compare the pooled 3x3 linear head with the repository MLP on one split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge
from src.llm_judge_ood.model.judge import JudgeTrainingConfig, SharedBackboneJudge
from src.llm_judge_ood.shared.metrics import judge_metrics


SOURCE_DOMAINS = ("Humanities", "Language", "Social Science")
SOURCE_SKILLS = ("Comprehension", "Factuality", "Logical Correctness")
TARGET_DOMAINS = (*SOURCE_DOMAINS, "History", "Culture")
TARGET_SKILLS = (*SOURCE_SKILLS, "Commonsense Understanding", "Completeness", "Insightfulness")
CLASSES = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b-space", type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_with_direct_judge.jsonl"
        ),
    )
    parser.add_argument(
        "--features", type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_hidden_states.npz"
        ),
    )
    parser.add_argument(
        "--pooled-summary", type=Path,
        default=Path("artifacts/flask_minimal_validation/pooled_3x3_15pct_head/summary.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/flask_minimal_validation/pooled_3x3_15pct_weighted_mlp_comparison"),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    pooled = json.loads(args.pooled_summary.read_text(encoding="utf-8"))
    rows = load_jsonl(args.b_space)
    features = load_aligned_features(args.features, rows)
    domains = np.asarray([str(row["domain_ids"][0]) for row in rows], dtype=str)
    skills = np.asarray([str(row["task_id"]) for row in rows], dtype=str)
    questions = np.asarray([str(row["base_id"]) for row in rows], dtype=str)
    labels = np.asarray([int(row["ground_truth"]) for row in rows], dtype=np.int8)
    source = np.isin(domains, SOURCE_DOMAINS) & np.isin(skills, SOURCE_SKILLS)
    train_pool = set(str(value) for value in pooled["train_question_ids"])
    validation_questions = select_validation_questions(
        questions=questions[source], labels=labels[source], train_questions=train_pool,
        fraction=float(args.validation_fraction), seed=int(args.seed),
    )
    fit_questions = train_pool.difference(validation_questions)
    fit_mask = source & np.isin(questions, np.asarray(sorted(fit_questions), dtype=str))
    validation_mask = source & np.isin(questions, np.asarray(sorted(validation_questions), dtype=str))
    source_id_mask = source & ~np.isin(questions, np.asarray(sorted(train_pool), dtype=str))
    all_test_mask = ~np.isin(questions, np.asarray(sorted(train_pool), dtype=str))
    query_ids = np.full(int(source.sum()), "pooled_3x3", dtype=object)
    all_query_ids = np.full(len(rows), "pooled_3x3", dtype=object)

    linear = PerQueryLinearJudge(LinearJudgeConfig(
        method="linear", representation="last_layer", pca_dim=2560,
        class_values=CLASSES, seed=int(args.seed), learning_rate=1e-3,
        weight_decay=1e-4, epochs=int(args.epochs), batch_size=int(args.batch_size),
        patience=6, device="cpu", class_weight="balanced", head_sharing="shared",
    )).fit(
        features[source], labels[source], query_ids,
        train_mask=fit_mask[source], validation_mask=validation_mask[source],
    )
    linear_output = linear.predict_output(features, all_query_ids)
    linear_predictions = linear_output.classes[np.argmax(linear_output.probabilities, axis=1)].astype(np.int8)
    linear.save(args.output_dir / "linear_validation_split.joblib")

    mlp = SharedBackboneJudge(JudgeTrainingConfig(
        hidden_dim=96, output_dim=48, learning_rate=1e-3, weight_decay=1e-4,
        epochs=int(args.epochs), batch_size=int(args.batch_size), patience=6,
        seed=int(args.seed), device="cpu", loss="ce", class_weight="balanced",
        class_values=CLASSES,
    )).fit(
        features[source], labels[source], query_ids,
        train_mask=fit_mask[source], validation_mask=validation_mask[source],
    )
    mlp_output = mlp.predict_output(features, all_query_ids)
    mlp_predictions = mlp_output.classes[np.argmax(mlp_output.probabilities, axis=1)].astype(np.int8)
    checkpoint_paths = mlp.save_checkpoints(args.output_dir / "mlp_checkpoints")

    metrics_by_model = {}
    for name, prediction in (("linear", linear_predictions), ("mlp", mlp_predictions)):
        metrics_by_model[name] = {
            "fit": metrics(labels[fit_mask], prediction[fit_mask]),
            "validation": metrics(labels[validation_mask], prediction[validation_mask]),
            "source_id": metrics(labels[source_id_mask], prediction[source_id_mask]),
            "all_5x6": metrics(labels[all_test_mask], prediction[all_test_mask]),
            "cell_results": cell_metrics(
                domains=domains, skills=skills, labels=labels, predictions=prediction,
                test_mask=all_test_mask,
            ),
        }
    direct_prediction, direct_parsed = parsed_direct_scores(rows)
    direct_source_id_mask = source_id_mask & direct_parsed
    metrics_by_model["direct_same_source_id"] = {
        "source_id": metrics(labels[direct_source_id_mask], direct_prediction[direct_source_id_mask]),
        "parsed_rows": int(direct_source_id_mask.sum()),
        "unparsed_rows": int((source_id_mask & ~direct_parsed).sum()),
    }
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        sample_ids=np.asarray([str(row["b_id"]) for row in rows], dtype=str)[all_test_mask],
        question_ids=questions[all_test_mask], labels=labels[all_test_mask],
        linear_predictions=linear_predictions[all_test_mask], mlp_predictions=mlp_predictions[all_test_mask],
        domain_ids=domains[all_test_mask], task_ids=skills[all_test_mask],
    )
    write_cell_csv(args.output_dir / "cell_results.csv", metrics_by_model)
    summary = {
        "artifact_type": "flask_5x6_pooled_3x3_15pct_linear_vs_mlp_v1",
        "source_b_space": str(args.b_space), "source_features": str(args.features),
        "pooled_summary": str(args.pooled_summary), "train_question_ids": sorted(train_pool),
        "validation_question_ids": sorted(validation_questions),
        "fit_question_ids": sorted(fit_questions),
        "fit_rows": int(fit_mask.sum()), "validation_rows": int(validation_mask.sum()),
        "source_id_rows": int(source_id_mask.sum()), "all_5x6_test_rows": int(all_test_mask.sum()),
        "model_metadata": {"linear": linear.to_metadata(), "mlp": mlp.to_metadata()},
        "mlp_checkpoint_paths": checkpoint_paths,
        "metrics": metrics_by_model,
        "train_question_ids_excluded_from_all_test": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "fit_rows": summary["fit_rows"], "validation_rows": summary["validation_rows"],
        "source_id_rows": summary["source_id_rows"], "all_5x6_test_rows": summary["all_5x6_test_rows"],
        "linear_source_id": metrics_by_model["linear"]["source_id"],
        "mlp_source_id": metrics_by_model["mlp"]["source_id"],
        "linear_all_5x6": metrics_by_model["linear"]["all_5x6"],
        "mlp_all_5x6": metrics_by_model["mlp"]["all_5x6"],
        "direct_same_source_id": metrics_by_model["direct_same_source_id"]["source_id"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


def select_validation_questions(
    *, questions: np.ndarray, labels: np.ndarray, train_questions: set[str],
    fraction: float, seed: int,
) -> set[str]:
    available = sorted(train_questions)
    n_validation = max(1, min(int(round(len(available) * fraction)), len(available) - 1))
    by_question = {
        question: np.asarray([
            np.sum(labels[questions == question] == value) for value in CLASSES
        ], dtype=float)
        for question in available
    }
    pool = np.isin(questions, np.asarray(available, dtype=str))
    target = np.asarray([np.sum(labels[pool] == value) for value in CLASSES], dtype=float) * fraction
    selected: set[str] = set()
    current = np.zeros(len(CLASSES), dtype=float)
    candidates = set(available)
    while len(selected) < n_validation:
        def score(question: str) -> tuple[float, str]:
            proposal = current + by_question[question]
            distance = float(np.sum(((proposal - target) / np.maximum(target, 1.0)) ** 2))
            tie = hashlib.sha256(f"{seed}::validation::{question}".encode("utf-8")).hexdigest()
            return distance, tie
        choice = min(candidates, key=score)
        selected.add(choice); candidates.remove(choice); current += by_question[choice]
    return selected


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        raise ValueError("Cannot calculate metrics on an empty label array")
    value = judge_metrics(labels, predictions, class_values=CLASSES)
    return {
        "rows": int(len(labels)), "mae": float(value["mae"]),
        "exact_accuracy": float(value["accuracy"]),
        "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
        "quadratic_weighted_kappa": float(value["qwk"]),
    }


def parsed_direct_scores(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(rows), dtype=np.int8)
    parsed = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        value = row.get("direct_score")
        try:
            score = int(value)
        except (TypeError, ValueError):
            continue
        if score in CLASSES:
            scores[index] = score
            parsed[index] = True
    return scores, parsed


def cell_metrics(
    *, domains: np.ndarray, skills: np.ndarray, labels: np.ndarray,
    predictions: np.ndarray, test_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in TARGET_DOMAINS:
        for skill in TARGET_SKILLS:
            mask = test_mask & (domains == domain) & (skills == skill)
            if not mask.any():
                raise RuntimeError(f"No test rows for {domain} x {skill}")
            rows.append({"target_domain": domain, "target_skill": skill, **metrics(labels[mask], predictions[mask])})
    return rows


def write_cell_csv(path: Path, model_rows: dict[str, Any]) -> None:
    fields = (
        "model", "target_domain", "target_skill", "rows", "mae", "exact_accuracy",
        "plus_minus_1_accuracy", "quadratic_weighted_kappa",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for model in ("linear", "mlp"):
            for row in model_rows[model]["cell_results"]:
                writer.writerow({"model": model, **row})


def load_aligned_features(path: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    row_ids = np.asarray([str(row["b_id"]) for row in rows], dtype=str)
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float16)
        feature_ids = np.asarray(payload["sample_ids"]).astype(str)
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or features.shape[0] != len(feature_ids):
        raise ValueError(f"Unexpected feature cache shape: {features.shape}")
    if set(row_ids.tolist()) != set(feature_ids.tolist()):
        raise ValueError("B-space rows and feature cache ids differ")
    index = {value: i for i, value in enumerate(feature_ids.tolist())}
    return features[np.asarray([index[value] for value in row_ids], dtype=np.int64)]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
