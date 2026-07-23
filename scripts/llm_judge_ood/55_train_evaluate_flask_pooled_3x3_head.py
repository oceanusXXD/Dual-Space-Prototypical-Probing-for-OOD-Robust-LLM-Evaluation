#!/usr/bin/env python3
"""Train one pooled 3x3 FLASK linear head and evaluate the documented 5x6 grid.

The left-upper 3x3 cells are pooled into one source dataset.  A global,
question-group split selects approximately the requested fraction in *every*
source cell while keeping every occurrence of a question id on one split side.
The resulting single head predicts all 5x6 B-space cells without re-running
Qwen.
"""

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
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge
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
        "--output-dir", type=Path,
        default=Path("artifacts/flask_minimal_validation/pooled_3x3_15pct_head"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-question-fraction", type=float, default=0.15)
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
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    rows = load_jsonl(args.b_space)
    features = load_aligned_features(args.features, rows)
    domains = np.asarray([str(row["domain_ids"][0]) for row in rows], dtype=str)
    skills = np.asarray([str(row["task_id"]) for row in rows], dtype=str)
    question_ids = np.asarray([str(row["base_id"]) for row in rows], dtype=str)
    labels = np.asarray([int(row["ground_truth"]) for row in rows], dtype=np.int8)
    source_mask = np.isin(domains, SOURCE_DOMAINS) & np.isin(skills, SOURCE_SKILLS)
    if not source_mask.any():
        raise RuntimeError("The documented 3x3 pooled source is empty")

    train_questions, split_audit = select_pooled_train_questions(
        domains=domains,
        skills=skills,
        question_ids=question_ids,
        labels=labels,
        source_mask=source_mask,
        fraction=float(args.train_question_fraction),
        seed=int(args.seed),
    )
    train_question_array = np.asarray(sorted(train_questions), dtype=str)
    train_mask = source_mask & np.isin(question_ids, train_question_array)
    source_test_mask = source_mask & ~np.isin(question_ids, train_question_array)
    global_test_mask = ~np.isin(question_ids, train_question_array)
    if not train_mask.any() or not source_test_mask.any():
        raise RuntimeError("The pooled source split has an empty train or held-out set")
    if np.isin(question_ids[global_test_mask], train_question_array).any():
        raise RuntimeError("A pooled train question leaked into the 5x6 test set")

    query_ids = np.full(int(source_mask.sum()), "pooled_3x3", dtype=object)
    config = LinearJudgeConfig(
        method="linear", representation="last_layer", pca_dim=2560,
        class_values=CLASSES, seed=int(args.seed), learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay), epochs=int(args.epochs),
        batch_size=int(args.batch_size), patience=6, device="cpu",
        class_weight="balanced", head_sharing="shared",
    )
    head = PerQueryLinearJudge(config).fit(
        features[source_mask], labels[source_mask], query_ids,
        train_mask=train_mask[source_mask],
        validation_mask=np.zeros(int(source_mask.sum()), dtype=bool),
    )
    model_path = args.output_dir / "pooled_3x3_linear_head.joblib"
    head.save(model_path)

    routed_queries = np.full(len(features), "pooled_3x3", dtype=object)
    output = head.predict_output(features, routed_queries)
    predictions = output.classes[np.argmax(output.probabilities, axis=1)].astype(np.int8)
    cell_results = evaluate_cells(
        domains=domains, skills=skills, labels=labels, predictions=predictions,
        global_test_mask=global_test_mask,
    )
    source_train_metrics = metrics(labels[train_mask], predictions[train_mask])
    source_id_metrics = metrics(labels[source_test_mask], predictions[source_test_mask])
    direct_scores, direct_parsed = parsed_direct_scores(rows)
    direct_source_id_mask = source_test_mask & direct_parsed
    direct_source_id_metrics = metrics(
        labels[direct_source_id_mask],
        direct_scores[direct_source_id_mask],
    )
    all_test_metrics = metrics(labels[global_test_mask], predictions[global_test_mask])
    np.savez_compressed(
        args.output_dir / "test_predictions.npz",
        sample_ids=np.asarray([str(row["b_id"]) for row in rows], dtype=str)[global_test_mask],
        question_ids=question_ids[global_test_mask], labels=labels[global_test_mask],
        predictions=predictions[global_test_mask], domain_ids=domains[global_test_mask],
        task_ids=skills[global_test_mask],
    )
    write_cell_csv(args.output_dir / "cell_results.csv", cell_results)
    write_cell_markdown(args.output_dir / "cell_results.md", cell_results)
    summary = {
        "artifact_type": "flask_5x6_pooled_3x3_15pct_linear_head_v1",
        "source_b_space": str(args.b_space),
        "source_features": str(args.features),
        "model_path": str(model_path),
        "seed": int(args.seed),
        "source_cells": [f"{domain}::{skill}" for domain in SOURCE_DOMAINS for skill in SOURCE_SKILLS],
        "train_question_fraction_requested": float(args.train_question_fraction),
        "train_question_ids": sorted(train_questions),
        "split": split_audit,
        "model_metadata": head.to_metadata(),
        "source_train_rows": int(train_mask.sum()),
        "source_id_rows": int(source_test_mask.sum()),
        "all_5x6_test_rows": int(global_test_mask.sum()),
        "source_train_metrics": source_train_metrics,
        "source_id_metrics": source_id_metrics,
        "direct_judge_on_same_source_id_metrics": direct_source_id_metrics,
        "direct_judge_on_same_source_id_parsed_rows": int(direct_source_id_mask.sum()),
        "direct_judge_on_same_source_id_unparsed_rows": int((source_test_mask & ~direct_parsed).sum()),
        "all_5x6_micro_metrics": all_test_metrics,
        "cell_results": cell_results,
        "all_train_question_ids_excluded_from_5x6_test": True,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "source_train_rows": summary["source_train_rows"],
        "source_id_rows": summary["source_id_rows"],
        "all_5x6_test_rows": summary["all_5x6_test_rows"],
        "source_id_metrics": source_id_metrics,
        "direct_judge_on_same_source_id_metrics": direct_source_id_metrics,
        "all_5x6_micro_metrics": all_test_metrics,
        "elapsed_seconds": summary["elapsed_seconds"],
    }, ensure_ascii=False, indent=2))


def select_pooled_train_questions(
    *, domains: np.ndarray, skills: np.ndarray, question_ids: np.ndarray, labels: np.ndarray,
    source_mask: np.ndarray, fraction: float, seed: int,
) -> tuple[set[str], dict[str, Any]]:
    """Choose globally disjoint question groups with each source cell at 15%."""

    cells = [(domain, skill) for domain in SOURCE_DOMAINS for skill in SOURCE_SKILLS]
    train_questions: set[str] = set()
    objective = 0.0
    audit_by_cell: dict[tuple[str, str], dict[str, Any]] = {}
    # The selected B-space is single-domain.  Its source question groups are
    # therefore disjoint across these three domains, so solving three smaller
    # MILPs is exactly equivalent to one large grouped split and much faster.
    for domain in SOURCE_DOMAINS:
        domain_cells = [(domain, skill) for skill in SOURCE_SKILLS]
        domain_source_mask = source_mask & (domains == domain)
        source_questions = sorted(set(question_ids[domain_source_mask].tolist()))
        question_index = {value: index for index, value in enumerate(source_questions)}
        cell_question_counts = np.zeros((len(domain_cells), len(source_questions)), dtype=np.float64)
        cell_label_counts = np.zeros(
            (len(domain_cells), len(CLASSES), len(source_questions)), dtype=np.float64
        )
        for row_index in np.flatnonzero(domain_source_mask):
            cell_index = domain_cells.index((domains[row_index], skills[row_index]))
            question_index_value = question_index[question_ids[row_index]]
            class_index = CLASSES.index(int(labels[row_index]))
            cell_question_counts[cell_index, question_index_value] = 1.0
            cell_label_counts[cell_index, class_index, question_index_value] += 1.0
        cell_totals = cell_question_counts.sum(axis=1).astype(int)
        targets = np.asarray([
            max(1, min(int(round(total * fraction)), int(total) - 1)) for total in cell_totals
        ], dtype=float)
        selected, local_objective = solve_group_split(
            cell_question_counts=cell_question_counts,
            cell_label_counts=cell_label_counts,
            targets=targets,
            fraction=fraction,
            seed=seed,
        )
        objective += local_objective
        train_questions.update(source_questions[index] for index in np.flatnonzero(selected))
        for cell_index, (cell_domain, skill) in enumerate(domain_cells):
            cell_question_mask = cell_question_counts[cell_index].astype(bool)
            selected_count = int(selected[cell_question_mask].sum())
            available = int(cell_question_mask.sum())
            if selected_count != int(targets[cell_index]):
                raise RuntimeError(f"Global split missed the requested count for {cell_domain} x {skill}")
            all_labels = cell_label_counts[cell_index].sum(axis=1)
            train_labels = cell_label_counts[cell_index][:, selected].sum(axis=1)
            audit_by_cell[(cell_domain, skill)] = {
                "source_domain": cell_domain,
                "source_skill": skill,
                "available_questions": available,
                "train_questions": selected_count,
                "train_question_fraction": float(selected_count / available),
                "target_train_label_counts": {
                    str(value): float(all_labels[offset] * fraction)
                    for offset, value in enumerate(CLASSES)
                },
                "actual_train_label_counts": {
                    str(value): int(train_labels[offset]) for offset, value in enumerate(CLASSES)
                },
            }
    audit_cells: list[dict[str, Any]] = []
    for cell in cells:
        audit_cells.append(audit_by_cell[cell])
    return train_questions, {
        "method": "global_question_group_milp_label_stratification",
        "objective": float(objective),
        "global_source_questions": len(set(question_ids[source_mask].tolist())),
        "global_train_questions": len(train_questions),
        "source_cells": audit_cells,
    }


def solve_group_split(
    *, cell_question_counts: np.ndarray, cell_label_counts: np.ndarray,
    targets: np.ndarray, fraction: float, seed: int,
) -> tuple[np.ndarray, float]:
    """Solve exact per-cell question counts while minimizing label-count deviation."""

    cell_count, class_count, question_count = cell_label_counts.shape
    slack_count = cell_count * class_count
    variable_count = question_count + slack_count
    objective = np.zeros(variable_count, dtype=float)
    label_targets = cell_label_counts.sum(axis=2) * float(fraction)
    objective[question_count:] = 1.0 / np.maximum(label_targets.reshape(-1), 1.0)
    tie_break = np.asarray([
        int(hashlib.sha256(f"{seed}::{index}".encode("utf-8")).hexdigest()[:12], 16)
        for index in range(question_count)
    ], dtype=float)
    objective[:question_count] = tie_break / max(float(tie_break.max()), 1.0) * 1e-8

    equality = np.zeros((cell_count, variable_count), dtype=float)
    equality[:, :question_count] = cell_question_counts
    inequalities: list[np.ndarray] = []
    upper_bounds: list[float] = []
    for cell_index in range(cell_count):
        for class_index in range(class_count):
            slack_index = question_count + cell_index * class_count + class_index
            values = cell_label_counts[cell_index, class_index]
            target = float(label_targets[cell_index, class_index])
            positive = np.zeros(variable_count, dtype=float)
            positive[:question_count] = values
            positive[slack_index] = -1.0
            inequalities.append(positive); upper_bounds.append(target)
            negative = np.zeros(variable_count, dtype=float)
            negative[:question_count] = -values
            negative[slack_index] = -1.0
            inequalities.append(negative); upper_bounds.append(-target)
    result = milp(
        c=objective,
        integrality=np.concatenate([np.ones(question_count, dtype=int), np.zeros(slack_count, dtype=int)]),
        bounds=Bounds(
            lb=np.zeros(variable_count, dtype=float),
            ub=np.concatenate([np.ones(question_count, dtype=float), np.full(slack_count, np.inf)]),
        ),
        constraints=(
            LinearConstraint(equality, lb=targets, ub=targets),
            LinearConstraint(np.vstack(inequalities), lb=-np.inf, ub=np.asarray(upper_bounds)),
        ),
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"Could not construct a globally grouped 3x3 split: {result.message}")
    selected = np.asarray(result.x[:question_count] >= 0.5, dtype=bool)
    if not np.array_equal(cell_question_counts @ selected.astype(float), targets):
        raise RuntimeError("MILP split does not meet all requested per-cell question counts")
    return selected, float(result.fun)


def evaluate_cells(
    *, domains: np.ndarray, skills: np.ndarray, labels: np.ndarray,
    predictions: np.ndarray, global_test_mask: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in TARGET_DOMAINS:
        for skill in TARGET_SKILLS:
            cell = (domains == domain) & (skills == skill)
            test = cell & global_test_mask
            if not test.any():
                raise RuntimeError(f"No held-out rows for {domain} x {skill}")
            rows.append({
                "target_domain": domain,
                "target_skill": skill,
                "rows": int(test.sum()),
                "excluded_train_question_rows": int((cell & ~global_test_mask).sum()),
                **metrics(labels[test], predictions[test]),
            })
    return rows


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        raise ValueError("Cannot calculate metrics on an empty label array")
    value = judge_metrics(labels, predictions, class_values=CLASSES)
    return {
        "mae": float(value["mae"]),
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


def load_aligned_features(path: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    row_ids = np.asarray([str(row["b_id"]) for row in rows], dtype=str)
    if len(row_ids) != len(set(row_ids.tolist())):
        raise ValueError("B-space rows contain duplicate b_id values")
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float16)
        feature_ids = np.asarray(payload["sample_ids"]).astype(str)
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or features.shape[0] != len(feature_ids):
        raise ValueError(f"Unexpected B-space feature cache shape: {features.shape}")
    if set(row_ids.tolist()) != set(feature_ids.tolist()):
        raise ValueError("B-space rows and feature cache do not contain the same ids")
    feature_index = {value: index for index, value in enumerate(feature_ids.tolist())}
    return features[np.asarray([feature_index[value] for value in row_ids], dtype=np.int64)]


def write_cell_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "target_domain", "target_skill", "rows", "excluded_train_question_rows",
        "mae", "exact_accuracy", "plus_minus_1_accuracy", "quadratic_weighted_kappa",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def write_cell_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "| target Domain | target Skill | B 行数 | MAE | Exact Accuracy | ±1 Accuracy | QWK |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['target_domain']} | {row['target_skill']} | {row['rows']} | "
                f"{row['mae']:.4f} | {row['exact_accuracy']:.4f} | "
                f"{row['plus_minus_1_accuracy']:.4f} | "
                f"{row['quadratic_weighted_kappa']:.4f} |\n"
            )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No B-space rows in {path}")
    return rows


if __name__ == "__main__":
    main()
