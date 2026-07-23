#!/usr/bin/env python3
"""Run basic residual-only ViM on cached FLASK B-space features."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_judge_ood.scores.vim import ViMScorer
from src.llm_judge_ood.shared.metrics import ood_metrics


SOURCE_DOMAINS = ("Humanities", "Language", "Social Science")
SOURCE_SKILLS = ("Comprehension", "Factuality", "Logical Correctness")
CLASS_VALUES = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-question-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-question-fraction", type=float, default=0.10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--layer",
        default="last",
        help="Feature layer to score: last, all, or a zero-based layer index.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = load_feature_payload(args.features, layer=str(args.layer))

    target_domains = ordered_values(payload["domains"], SOURCE_DOMAINS)
    target_skills = ordered_values(payload["skills"], SOURCE_SKILLS)
    pair_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []

    for source_domain in SOURCE_DOMAINS:
        for source_skill in SOURCE_SKILLS:
            source_mask = (payload["domains"] == source_domain) & (payload["skills"] == source_skill)
            if not source_mask.any():
                continue
            source_indices = np.flatnonzero(source_mask)
            source_labels = payload["labels"][source_indices]
            source_questions = payload["question_ids"][source_indices]
            train_questions = stratified_question_selection(
                labels=source_labels,
                question_ids=source_questions,
                fraction=float(args.train_question_fraction),
                seed=int(args.seed),
                namespace=f"{source_domain}::{source_skill}",
            )
            remaining_questions = sorted(set(source_questions.tolist()).difference(train_questions))
            calibration_questions = stable_question_sample(
                remaining_questions,
                fraction=float(args.calibration_question_fraction),
                seed=int(args.seed),
                namespace=f"{source_domain}::{source_skill}::vim_calibration",
            )
            id_questions = sorted(set(remaining_questions).difference(calibration_questions))
            if not train_questions or not calibration_questions or not id_questions:
                raise RuntimeError(f"ViM split failed for {source_domain} x {source_skill}")

            train_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
            calibration_mask = source_mask & np.isin(
                payload["question_ids"], np.asarray(sorted(calibration_questions))
            )
            id_mask = source_mask & np.isin(payload["question_ids"], np.asarray(id_questions))
            train_rows = int(train_mask.sum())
            rank = min(int(args.rank), train_rows - 2, payload["features"].shape[1] - 1)
            if rank < 1:
                raise RuntimeError(f"Not enough train rows for ViM: {source_domain} x {source_skill}")

            scorer = fit_vim_with_rank(payload["features"][train_mask], rank)
            rank = int(scorer.rank)
            calibration_scores = scorer.score(payload["features"][calibration_mask])
            id_scores = scorer.score(payload["features"][id_mask])
            hard_threshold = float(np.quantile(calibration_scores, 0.95))
            soft_threshold = float(np.quantile(calibration_scores, 0.90))
            id_fpr_hard = float(np.mean(id_scores >= hard_threshold))
            reference_rows.append(
                {
                    "source_domain": source_domain,
                    "source_skill": source_skill,
                    "train_questions": len(train_questions),
                    "calibration_questions": len(calibration_questions),
                    "id_questions": len(id_questions),
                    "train_rows": train_rows,
                    "calibration_rows": int(calibration_mask.sum()),
                    "id_rows": int(id_mask.sum()),
                    "rank": int(rank),
                    "soft_threshold_q90": soft_threshold,
                    "hard_threshold_q95": hard_threshold,
                    "id_fpr_at_hard_threshold": id_fpr_hard,
                    "id_score_mean": float(np.mean(id_scores)),
                    "id_score_std": float(np.std(id_scores)),
                }
            )

            for target_domain in target_domains:
                for target_skill in target_skills:
                    if target_domain == source_domain and target_skill == source_skill:
                        continue
                    target_mask = (
                        (payload["domains"] == target_domain)
                        & (payload["skills"] == target_skill)
                        & ~np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
                    )
                    if not target_mask.any():
                        continue
                    ood_scores_values = scorer.score(payload["features"][target_mask])
                    y = np.concatenate(
                        [
                            np.zeros(len(id_scores), dtype=np.int64),
                            np.ones(len(ood_scores_values), dtype=np.int64),
                        ]
                    )
                    scores = np.concatenate([id_scores, ood_scores_values]).astype(np.float64)
                    metrics = ood_metrics(y, scores)
                    pair_rows.append(
                        {
                            "source_domain": source_domain,
                            "source_skill": source_skill,
                            "target_domain": target_domain,
                            "target_skill": target_skill,
                            "shift_type": shift_type(source_domain, source_skill, target_domain, target_skill),
                            "id_rows": int(id_mask.sum()),
                            "ood_rows": int(target_mask.sum()),
                            "rank": int(rank),
                            "auroc": metrics["auroc"],
                            "aupr_ood": metrics["aupr"],
                            "fpr95": metrics["fpr95"],
                            "id_fpr_at_hard_threshold": id_fpr_hard,
                            "ood_tpr_at_hard_threshold": float(np.mean(ood_scores_values >= hard_threshold)),
                            "id_score_mean": float(np.mean(id_scores)),
                            "ood_score_mean": float(np.mean(ood_scores_values)),
                        }
                    )

    if not pair_rows:
        raise RuntimeError("No ViM head-target results were produced")
    write_csv(args.output_dir / "head_cell_results.csv", pair_rows)
    write_csv(args.output_dir / "source_id_reference.csv", reference_rows)
    summary = build_summary(
        args=args,
        payload=payload,
        pair_rows=pair_rows,
        reference_rows=reference_rows,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_feature_payload(path: Path, *, layer: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float32)
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        labels = np.asarray(cache["labels"], dtype=int)
        domains = np.asarray(cache["domain_ids"]).astype(str)
        skills = np.asarray(cache["task_ids"]).astype(str)
        question_ids = np.asarray(cache["query_ids"]).astype(str)
        metadata_json = str(np.asarray(cache["metadata_json"]).item()) if "metadata_json" in cache.files else "{}"
    if features.ndim == 2:
        matrix = features
    elif features.ndim == 3:
        if layer == "last":
            matrix = features[:, -1, :]
        elif layer == "all":
            matrix = features.reshape(features.shape[0], features.shape[1] * features.shape[2])
        else:
            index = int(layer)
            matrix = features[:, index, :]
    else:
        raise ValueError(f"Unsupported feature shape: {features.shape}")
    expected = len(sample_ids)
    for name, values in {
        "labels": labels,
        "domain_ids": domains,
        "task_ids": skills,
        "query_ids": question_ids,
    }.items():
        if len(values) != expected:
            raise ValueError(f"{name} does not align with sample_ids")
    if not np.isfinite(matrix).all():
        raise ValueError("Feature matrix contains non-finite values")
    return {
        "features": matrix.astype(np.float32, copy=False),
        "sample_ids": sample_ids,
        "labels": labels,
        "domains": domains,
        "skills": skills,
        "question_ids": question_ids,
        "metadata_json": metadata_json,
    }


def stratified_question_selection(
    *,
    labels: np.ndarray,
    question_ids: np.ndarray,
    fraction: float,
    seed: int,
    namespace: str,
) -> set[str]:
    unique_questions = sorted(set(question_ids.tolist()))
    n_train = max(1, int(round(len(unique_questions) * fraction)))
    n_train = min(n_train, len(unique_questions) - 1)
    by_question = {
        question: np.asarray([np.sum(labels[question_ids == question] == value) for value in CLASS_VALUES], dtype=float)
        for question in unique_questions
    }
    target = np.asarray([np.sum(labels == value) for value in CLASS_VALUES], dtype=float) * fraction
    selected: set[str] = set()
    current = np.zeros(len(CLASS_VALUES), dtype=float)
    candidates = set(unique_questions)
    while len(selected) < n_train:
        def score(question: str) -> tuple[float, str]:
            proposal = current + by_question[question]
            distance = float(np.sum(((proposal - target) / np.maximum(target, 1.0)) ** 2))
            tie = hashlib.sha256(f"{seed}::{namespace}::{question}".encode("utf-8")).hexdigest()
            return distance, tie

        choice = min(candidates, key=score)
        selected.add(choice)
        candidates.remove(choice)
        current += by_question[choice]
    return selected


def stable_question_sample(questions: list[str], *, fraction: float, seed: int, namespace: str) -> set[str]:
    if not questions:
        return set()
    n = max(1, int(round(len(questions) * fraction)))
    n = min(n, len(questions))
    ordered = sorted(
        questions,
        key=lambda question: hashlib.sha256(f"{seed}::{namespace}::{question}".encode("utf-8")).hexdigest(),
    )
    return set(ordered[:n])


def ordered_values(values: np.ndarray, preferred: tuple[str, ...]) -> list[str]:
    present = set(values.astype(str).tolist())
    return [value for value in preferred if value in present] + sorted(present.difference(preferred))


def shift_type(source_domain: str, source_skill: str, target_domain: str, target_skill: str) -> str:
    if source_domain == target_domain and source_skill == target_skill:
        return "ID test"
    if source_domain != target_domain and source_skill == target_skill:
        return "Domain shift"
    if source_domain == target_domain and source_skill != target_skill:
        return "Task shift"
    return "Joint shift"


def fit_vim_with_rank(features: np.ndarray, rank: int) -> ViMScorer:
    errors: list[str] = []
    for candidate in range(int(rank), 0, -1):
        try:
            return ViMScorer(rank=candidate).fit(features)
        except ValueError as exc:
            errors.append(f"rank={candidate}: {exc}")
    raise RuntimeError("Could not fit non-degenerate ViM scorer; " + " | ".join(errors[-3:]))


def build_summary(
    *,
    args: argparse.Namespace,
    payload: dict[str, np.ndarray],
    pair_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        keys = ("auroc", "aupr_ood", "fpr95", "id_fpr_at_hard_threshold", "ood_tpr_at_hard_threshold")
        return {
            "result_count": len(rows),
            **{key: float(np.mean([float(row[key]) for row in rows])) for key in keys},
            "id_rows_mean": float(np.mean([int(row["id_rows"]) for row in rows])),
            "ood_rows_mean": float(np.mean([int(row["ood_rows"]) for row in rows])),
        }

    by_shift = defaultdict(list)
    for row in pair_rows:
        by_shift[str(row["shift_type"])].append(row)
    return {
        "artifact_type": "flask_basic_residual_vim_v1",
        "source_features": str(args.features),
        "feature_rows": int(len(payload["sample_ids"])),
        "feature_dim": int(payload["features"].shape[1]),
        "layer": str(args.layer),
        "seed": int(args.seed),
        "train_question_fraction": float(args.train_question_fraction),
        "calibration_question_fraction_of_remaining": float(args.calibration_question_fraction),
        "requested_rank": int(args.rank),
        "head_count": len(reference_rows),
        "head_cell_result_count": len(pair_rows),
        "macro_metrics": mean_metrics(pair_rows),
        "shift_type_macro_metrics": {key: mean_metrics(value) for key, value in sorted(by_shift.items())},
        "source_id_reference": {
            "mean_id_fpr_at_hard_threshold": float(
                np.mean([float(row["id_fpr_at_hard_threshold"]) for row in reference_rows])
            ),
            "mean_train_rows": float(np.mean([int(row["train_rows"]) for row in reference_rows])),
            "mean_calibration_rows": float(np.mean([int(row["calibration_rows"]) for row in reference_rows])),
            "mean_id_rows": float(np.mean([int(row["id_rows"]) for row in reference_rows])),
        },
        "result_csv": str(args.output_dir / "head_cell_results.csv"),
        "source_id_reference_csv": str(args.output_dir / "source_id_reference.csv"),
        "elapsed_seconds": float(elapsed_seconds),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
