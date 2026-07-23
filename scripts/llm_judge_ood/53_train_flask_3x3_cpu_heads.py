#!/usr/bin/env python3
"""Train the documented nine FLASK 3x3 B-space linear heads on CPU."""

from __future__ import annotations

import argparse
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
from src.llm_judge_ood.shared.metrics import judge_metrics


SOURCE_DOMAINS = ("Humanities", "Language", "Social Science")
SOURCE_SKILLS = ("Comprehension", "Factuality", "Logical Correctness")
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
        "--output-dir", type=Path,
        default=Path("artifacts/flask_minimal_validation/cpu_3x3_heads"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-question-fraction", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.train_question_fraction < 1.0:
        raise ValueError("--train-question-fraction must be in (0, 1)")
    rows = load_jsonl(args.b_space)
    features, feature_ids = load_features(args.features)
    row_ids = [str(row["b_id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError("B-space rows contain duplicate b_id values")
    if set(row_ids) != set(feature_ids.tolist()):
        raise ValueError("B-space rows and feature cache do not contain the same ids")
    feature_index = {value: index for index, value in enumerate(feature_ids.tolist())}
    aligned_features = features[np.asarray([feature_index[value] for value in row_ids], dtype=np.int64)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    for domain in SOURCE_DOMAINS:
        for skill in SOURCE_SKILLS:
            summary = train_one_head(
                rows=rows, features=aligned_features, domain=domain, skill=skill,
                args=args,
            )
            summaries.append(summary)
    aggregate = {
        "artifact_type": "flask_5x6_cpu_3x3_linear_heads_v1",
        "source_b_space": str(args.b_space),
        "source_features": str(args.features),
        "heads": summaries,
        "head_count": len(summaries),
        "elapsed_seconds": time.perf_counter() - started,
        "device": "cpu",
    }
    write_json(args.output_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def train_one_head(*, rows: list[dict[str, Any]], features: np.ndarray, domain: str, skill: str, args: argparse.Namespace) -> dict[str, Any]:
    indices = np.asarray([
        index for index, row in enumerate(rows)
        if str(row["domain_ids"][0]) == domain and str(row["task_id"]) == skill
    ], dtype=np.int64)
    if not len(indices):
        raise ValueError(f"Source cell is empty: {domain} x {skill}")
    cell_rows = [rows[index] for index in indices]
    labels = np.asarray([int(row["ground_truth"]) for row in cell_rows], dtype=int)
    question_ids = np.asarray([str(row["base_id"]) for row in cell_rows])
    train_questions = stratified_question_selection(
        labels=labels, question_ids=question_ids,
        fraction=float(args.train_question_fraction), seed=int(args.seed),
        namespace=f"{domain}::{skill}",
    )
    train_mask = np.isin(question_ids, np.asarray(sorted(train_questions)))
    test_mask = ~train_mask
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(f"Question split failed for {domain} x {skill}")
    query_ids = np.full(len(indices), f"{domain}::{skill}", dtype=str)
    config = LinearJudgeConfig(
        method="linear", representation="last_layer", pca_dim=2560,
        class_values=CLASSES, seed=int(args.seed), learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay), epochs=int(args.epochs),
        batch_size=int(args.batch_size), patience=6, device="cpu",
        class_weight="balanced", head_sharing="shared",
    )
    head = PerQueryLinearJudge(config).fit(
        features[indices], labels, query_ids, train_mask=train_mask,
        validation_mask=np.zeros(len(indices), dtype=bool),
    )
    output = head.predict_output(features[indices], query_ids)
    predictions = output.classes[np.argmax(output.probabilities, axis=1)].astype(int)
    test_metrics = metrics(labels[test_mask], predictions[test_mask], output.probabilities[test_mask])
    train_metrics = metrics(labels[train_mask], predictions[train_mask], output.probabilities[train_mask])
    name = f"{slug(domain)}__{slug(skill)}"
    model_path = args.output_dir / f"{name}.joblib"
    if model_path.exists() and not args.overwrite:
        raise FileExistsError(f"Head already exists: {model_path}; pass --overwrite")
    head.save(model_path)
    split = {
        "source_domain": domain, "source_skill": skill,
        "train_question_ids": sorted(train_questions),
        "test_question_ids": sorted(set(question_ids.tolist()).difference(train_questions)),
        "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
        "train_label_counts": counts(labels[train_mask]), "test_label_counts": counts(labels[test_mask]),
    }
    write_json(args.output_dir / f"{name}.split.json", split)
    result = {
        "head_id": f"H({domain}, {skill})", "source_domain": domain,
        "source_skill": skill, "model_path": str(model_path),
        "split_path": str(args.output_dir / f"{name}.split.json"),
        "model_metadata": head.to_metadata(), "split": split,
        "train_metrics": train_metrics, "id_test_metrics": test_metrics,
    }
    write_json(args.output_dir / f"{name}.summary.json", result)
    return result


def stratified_question_selection(*, labels: np.ndarray, question_ids: np.ndarray, fraction: float, seed: int, namespace: str) -> set[str]:
    """Greedily match 10% row-level score proportions while splitting by question."""

    unique_questions = sorted(set(question_ids.tolist()))
    n_train = max(1, int(round(len(unique_questions) * fraction)))
    n_train = min(n_train, len(unique_questions) - 1)
    by_question: dict[str, np.ndarray] = {
        question: np.asarray([np.sum(labels[question_ids == question] == value) for value in CLASSES], dtype=float)
        for question in unique_questions
    }
    target = np.asarray([np.sum(labels == value) for value in CLASSES], dtype=float) * fraction
    selected: set[str] = set(); current = np.zeros(len(CLASSES), dtype=float)
    candidates = set(unique_questions)
    while len(selected) < n_train:
        def score(question: str) -> tuple[float, str]:
            proposal = current + by_question[question]
            distance = float(np.sum(((proposal - target) / np.maximum(target, 1.0)) ** 2))
            tie = hashlib.sha256(f"{seed}::{namespace}::{question}".encode("utf-8")).hexdigest()
            return distance, tie
        choice = min(candidates, key=score)
        selected.add(choice); candidates.remove(choice); current += by_question[choice]
    return selected


def metrics(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    value = judge_metrics(labels, predictions, probabilities=probabilities, class_values=CLASSES)
    return {
        "rows": int(len(labels)), "mae": float(value["mae"]),
        "exact_accuracy": float(value["accuracy"]),
        "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
        "quadratic_weighted_kappa": float(value["qwk"]),
    }


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float16)
        ids = np.asarray(payload["sample_ids"]).astype(str)
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or features.shape[0] != len(ids):
        raise ValueError(f"Unexpected B-space feature cache shape: {features.shape}")
    return features, ids


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No B-space rows in {path}")
    return rows


def counts(values: np.ndarray) -> dict[str, int]:
    counter = Counter(int(value) for value in values.tolist())
    return {str(value): int(counter.get(value, 0)) for value in CLASSES}


def slug(value: str) -> str:
    return "_".join(value.lower().split()).replace("-", "_")


if __name__ == "__main__":
    main()
