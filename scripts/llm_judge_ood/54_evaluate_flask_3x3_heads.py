#!/usr/bin/env python3
"""Evaluate the nine saved FLASK 3x3 heads on every documented 5x6 cell.

Each source head predicts the frozen Direct-Judge B-space once.  Its source
training question ids are then excluded from every target cell before the
per-cell metrics are calculated.  The independent source-head jobs run in
parallel and never invoke Qwen again.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# One BLAS thread per source-head process avoids oversubscribing the CPU when
# the nine independent source heads are evaluated concurrently.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.shared.metrics import judge_metrics


DOMAINS = ("Humanities", "Language", "Social Science", "History", "Culture")
SKILLS = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
)
CLASSES = (1, 2, 3, 4, 5)

# These objects are populated once by the parent, then inherited read-only by
# forked workers.  It keeps the 18428 x 2 x 2560 cache from being copied into
# every submitted job.
ROWS: list[dict[str, Any]] = []
FEATURES: np.ndarray | None = None
ROW_IDS: np.ndarray | None = None
QUESTION_IDS: np.ndarray | None = None
LABELS: np.ndarray | None = None
DOMAIN_IDS: np.ndarray | None = None
TASK_IDS: np.ndarray | None = None


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
        "--features",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_hidden_states.npz"
        ),
    )
    parser.add_argument(
        "--heads-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/cpu_3x3_heads"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/cpu_3x3_head_evaluation"),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(9, os.cpu_count() or 1),
        help="Concurrent source-head evaluations (default: all nine heads).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "prediction_parts"
    parts_dir.mkdir(exist_ok=True)

    load_inputs(args.b_space, args.features)
    jobs = load_jobs(args.heads_dir, parts_dir)
    if len(jobs) != 9:
        raise ValueError(f"Expected exactly nine trained source heads, found {len(jobs)}")

    started = time.perf_counter()
    results_by_head: list[dict[str, Any]] = []
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=min(int(args.workers), len(jobs)), mp_context=context
    ) as executor:
        futures = {executor.submit(evaluate_head, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            result = future.result()
            results_by_head.append(result)
            print(
                f"completed {job['head_id']}: {len(result['cell_results'])} cells, "
                f"{result['eligible_rows']} held-out rows",
                flush=True,
            )

    cell_results = sorted(
        (row for head in results_by_head for row in head["cell_results"]),
        key=lambda row: (
            DOMAINS.index(row["source_domain"]),
            SKILLS.index(row["source_skill"]),
            DOMAINS.index(row["target_domain"]),
            SKILLS.index(row["target_skill"]),
        ),
    )
    if len(cell_results) != 270:
        raise RuntimeError(f"Expected 270 source-head/target-cell rows, got {len(cell_results)}")
    if not all(row["train_question_ids_excluded"] for row in cell_results):
        raise RuntimeError("A test cell contains a question id used to train its source head")

    micro_metrics, total_predictions = micro_metrics_from_parts(results_by_head)
    target_cell_results = target_cell_micro_metrics_from_parts(results_by_head)
    if len(target_cell_results) != len(DOMAINS) * len(SKILLS):
        raise RuntimeError("Target-cell aggregation does not cover the complete documented 5x6 grid")
    head_shift_results, shift_type_micro_results = shift_type_metrics_from_parts(results_by_head)
    shift_type_macro_results = shift_type_macro_metrics(cell_results)
    elapsed = time.perf_counter() - started
    write_csv(args.output_dir / "head_cell_results.csv", cell_results)
    write_markdown_table(args.output_dir / "head_cell_results.md", cell_results)
    write_target_cell_csv(args.output_dir / "target_cell_results.csv", target_cell_results)
    write_target_cell_markdown(args.output_dir / "target_cell_results.md", target_cell_results)
    write_head_shift_csv(args.output_dir / "head_shift_results.csv", head_shift_results)
    write_shift_type_summary_csv(
        args.output_dir / "shift_type_summary.csv", shift_type_micro_results, shift_type_macro_results
    )
    summary = {
        "artifact_type": "flask_5x6_3x3_linear_head_evaluation_v1",
        "source_b_space": str(args.b_space),
        "source_features": str(args.features),
        "heads_dir": str(args.heads_dir),
        "head_count": len(results_by_head),
        "head_cell_result_count": len(cell_results),
        "workers": min(int(args.workers), len(jobs)),
        "device": "cpu",
        "prediction_rows": total_predictions,
        "micro_metrics": micro_metrics,
        "target_cell_micro_metrics": target_cell_results,
        "head_shift_metrics": head_shift_results,
        "shift_type_micro_metrics": shift_type_micro_results,
        "shift_type_macro_cell_metrics": shift_type_macro_results,
        "train_question_ids_excluded_from_every_test_cell": True,
        "result_csv": str(args.output_dir / "head_cell_results.csv"),
        "result_markdown": str(args.output_dir / "head_cell_results.md"),
        "target_cell_result_csv": str(args.output_dir / "target_cell_results.csv"),
        "target_cell_result_markdown": str(args.output_dir / "target_cell_results.md"),
        "head_shift_result_csv": str(args.output_dir / "head_shift_results.csv"),
        "shift_type_summary_csv": str(args.output_dir / "shift_type_summary.csv"),
        "prediction_parts_dir": str(parts_dir),
        "elapsed_seconds": elapsed,
        "heads": sorted(results_by_head, key=lambda row: row["head_id"]),
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({
        "head_cell_result_count": len(cell_results),
        "prediction_rows": total_predictions,
        "micro_metrics": micro_metrics,
        "elapsed_seconds": elapsed,
    }, ensure_ascii=False, indent=2))


def load_inputs(b_space_path: Path, features_path: Path) -> None:
    global ROWS, FEATURES, ROW_IDS, QUESTION_IDS, LABELS, DOMAIN_IDS, TASK_IDS
    ROWS = load_jsonl(b_space_path)
    row_ids = np.asarray([str(row["b_id"]) for row in ROWS], dtype=str)
    if len(row_ids) != len(set(row_ids.tolist())):
        raise ValueError("B-space rows contain duplicate b_id values")
    with np.load(features_path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float16)
        feature_ids = np.asarray(payload["sample_ids"]).astype(str)
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or features.shape[0] != len(feature_ids):
        raise ValueError(f"Unexpected B-space feature cache shape: {features.shape}")
    if set(row_ids.tolist()) != set(feature_ids.tolist()):
        raise ValueError("B-space rows and feature cache do not contain the same ids")
    feature_index = {value: index for index, value in enumerate(feature_ids.tolist())}
    FEATURES = features[np.asarray([feature_index[value] for value in row_ids], dtype=np.int64)]
    ROW_IDS = row_ids
    QUESTION_IDS = np.asarray([str(row["base_id"]) for row in ROWS], dtype=str)
    LABELS = np.asarray([int(row["ground_truth"]) for row in ROWS], dtype=np.int8)
    DOMAIN_IDS = np.asarray([str(row["domain_ids"][0]) for row in ROWS], dtype=str)
    TASK_IDS = np.asarray([str(row["task_id"]) for row in ROWS], dtype=str)
    if not set(DOMAIN_IDS.tolist()).issubset(DOMAINS) or not set(TASK_IDS.tolist()).issubset(SKILLS):
        raise ValueError("B-space has an unexpected Domain or Skill outside the documented 5x6 scope")


def load_jobs(heads_dir: Path, parts_dir: Path) -> list[dict[str, Any]]:
    summaries = sorted(heads_dir.glob("*.summary.json"))
    jobs: list[dict[str, Any]] = []
    for summary_path in summaries:
        with summary_path.open(encoding="utf-8") as handle:
            summary = json.load(handle)
        source_domain = str(summary["source_domain"])
        source_skill = str(summary["source_skill"])
        split_path = Path(summary["split_path"])
        if not split_path.is_absolute():
            split_path = ROOT / split_path
        with split_path.open(encoding="utf-8") as handle:
            split = json.load(handle)
        model_path = Path(summary["model_path"])
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        jobs.append({
            "head_id": str(summary["head_id"]),
            "source_domain": source_domain,
            "source_skill": source_skill,
            "model_path": str(model_path),
            "train_question_ids": [str(value) for value in split["train_question_ids"]],
            "part_path": str(parts_dir / f"{slug(source_domain)}__{slug(source_skill)}.npz"),
        })
    return jobs


def evaluate_head(job: dict[str, Any]) -> dict[str, Any]:
    if any(value is None for value in (FEATURES, ROW_IDS, QUESTION_IDS, LABELS, DOMAIN_IDS, TASK_IDS)):
        raise RuntimeError("Worker did not receive the shared B-space inputs")
    features = FEATURES
    row_ids = ROW_IDS
    question_ids = QUESTION_IDS
    labels = LABELS
    domain_ids = DOMAIN_IDS
    task_ids = TASK_IDS
    model = joblib.load(job["model_path"])
    query_ids = tuple(getattr(model, "query_ids_", ()))
    if len(query_ids) != 1:
        raise ValueError(f"{job['head_id']} must contain exactly one trained query id, got {query_ids}")
    # The source head owns a single classifier.  This routing id selects that
    # classifier only; it is never concatenated to the frozen feature vectors.
    routed_queries = np.full(len(features), query_ids[0], dtype=object)
    output = model.predict_output(features, routed_queries)
    predictions = output.classes[np.argmax(output.probabilities, axis=1)].astype(np.int8)

    train_questions = np.asarray(job["train_question_ids"], dtype=str)
    allowed = ~np.isin(question_ids, train_questions)
    if not allowed.any():
        raise RuntimeError(f"{job['head_id']} excludes every B-space row")
    np.savez_compressed(
        job["part_path"],
        sample_ids=row_ids[allowed],
        question_ids=question_ids[allowed],
        labels=labels[allowed],
        predictions=predictions[allowed],
        domain_ids=domain_ids[allowed],
        task_ids=task_ids[allowed],
    )

    cell_results: list[dict[str, Any]] = []
    for target_domain in DOMAINS:
        for target_skill in SKILLS:
            cell = (domain_ids == target_domain) & (task_ids == target_skill)
            test = cell & allowed
            if not test.any():
                raise RuntimeError(
                    f"No held-out rows for {job['head_id']} -> {target_domain} x {target_skill}"
                )
            if np.isin(question_ids[test], train_questions).any():
                raise RuntimeError("Training question id leaked into a test cell")
            value = metrics(labels[test], predictions[test])
            cell_results.append({
                "head_id": job["head_id"],
                "source_domain": job["source_domain"],
                "source_skill": job["source_skill"],
                "target_domain": target_domain,
                "target_skill": target_skill,
                "test_type": shift_type(
                    job["source_domain"], job["source_skill"], target_domain, target_skill
                ),
                "rows": int(test.sum()),
                "excluded_train_question_rows": int((cell & ~allowed).sum()),
                "train_question_ids_excluded": True,
                **value,
                "acceptance": "passed",
            })
    return {
        "head_id": job["head_id"],
        "source_domain": job["source_domain"],
        "source_skill": job["source_skill"],
        "train_question_count": len(train_questions),
        "eligible_rows": int(allowed.sum()),
        "excluded_train_question_rows": int((~allowed).sum()),
        "prediction_part": job["part_path"],
        "cell_results": cell_results,
    }


def micro_metrics_from_parts(head_results: list[dict[str, Any]]) -> tuple[dict[str, float], int]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for result in head_results:
        with np.load(result["prediction_part"], allow_pickle=False) as payload:
            labels.append(np.asarray(payload["labels"], dtype=np.int8))
            predictions.append(np.asarray(payload["predictions"], dtype=np.int8))
    truth = np.concatenate(labels)
    predicted = np.concatenate(predictions)
    return metrics(truth, predicted), int(len(truth))


def target_cell_micro_metrics_from_parts(head_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    domain_ids: list[np.ndarray] = []
    task_ids: list[np.ndarray] = []
    for result in head_results:
        with np.load(result["prediction_part"], allow_pickle=False) as payload:
            labels.append(np.asarray(payload["labels"], dtype=np.int8))
            predictions.append(np.asarray(payload["predictions"], dtype=np.int8))
            domain_ids.append(np.asarray(payload["domain_ids"]).astype(str))
            task_ids.append(np.asarray(payload["task_ids"]).astype(str))
    truth = np.concatenate(labels)
    predicted = np.concatenate(predictions)
    domains = np.concatenate(domain_ids)
    skills = np.concatenate(task_ids)
    rows: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for skill in SKILLS:
            mask = (domains == domain) & (skills == skill)
            if not mask.any():
                raise RuntimeError(f"Target-cell aggregation has no predictions for {domain} x {skill}")
            rows.append({
                "target_domain": domain,
                "target_skill": skill,
                "head_prediction_rows": int(mask.sum()),
                "source_head_count": len(head_results),
                **metrics(truth[mask], predicted[mask]),
            })
    return rows


def shift_type_metrics_from_parts(
    head_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Calculate per-head and all-head micro metrics for the four shift types."""

    per_head: list[dict[str, Any]] = []
    combined: dict[str, dict[str, list[np.ndarray]]] = {
        value: {"labels": [], "predictions": []}
        for value in ("ID test", "Domain shift", "Task shift", "Joint shift")
    }
    for result in head_results:
        with np.load(result["prediction_part"], allow_pickle=False) as payload:
            labels = np.asarray(payload["labels"], dtype=np.int8)
            predictions = np.asarray(payload["predictions"], dtype=np.int8)
            domains = np.asarray(payload["domain_ids"]).astype(str)
            skills = np.asarray(payload["task_ids"]).astype(str)
        masks = shift_type_masks(
            result["source_domain"], result["source_skill"], domains, skills
        )
        for test_type, mask in masks.items():
            if not mask.any():
                raise RuntimeError(f"{result['head_id']} has no rows for {test_type}")
            per_head.append({
                "head_id": result["head_id"],
                "source_domain": result["source_domain"],
                "source_skill": result["source_skill"],
                "test_type": test_type,
                "target_cell_count": int(
                    sum(
                        shift_type(result["source_domain"], result["source_skill"], domain, skill)
                        == test_type
                        for domain in DOMAINS for skill in SKILLS
                    )
                ),
                "head_prediction_rows": int(mask.sum()),
                **metrics(labels[mask], predictions[mask]),
            })
            combined[test_type]["labels"].append(labels[mask])
            combined[test_type]["predictions"].append(predictions[mask])
    aggregate: list[dict[str, Any]] = []
    for test_type, values in combined.items():
        truth = np.concatenate(values["labels"])
        predicted = np.concatenate(values["predictions"])
        aggregate.append({
            "test_type": test_type,
            "head_cell_result_count": int(sum(
                row["target_cell_count"] for row in per_head if row["test_type"] == test_type
            )),
            "head_prediction_rows": int(len(truth)),
            **metrics(truth, predicted),
        })
    return per_head, aggregate


def shift_type_masks(
    source_domain: str, source_skill: str, domains: np.ndarray, skills: np.ndarray
) -> dict[str, np.ndarray]:
    same_domain = domains == source_domain
    same_skill = skills == source_skill
    return {
        "ID test": same_domain & same_skill,
        "Domain shift": ~same_domain & same_skill,
        "Task shift": same_domain & ~same_skill,
        "Joint shift": ~same_domain & ~same_skill,
    }


def shift_type_macro_metrics(cell_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregate: list[dict[str, Any]] = []
    metric_names = ("mae", "exact_accuracy", "plus_minus_1_accuracy", "quadratic_weighted_kappa")
    for test_type in ("ID test", "Domain shift", "Task shift", "Joint shift"):
        selected = [row for row in cell_results if row["test_type"] == test_type]
        if not selected:
            raise RuntimeError(f"No head-cell rows found for {test_type}")
        aggregate.append({
            "test_type": test_type,
            "head_cell_result_count": len(selected),
            "head_prediction_rows": int(sum(row["rows"] for row in selected)),
            **{name: float(np.mean([row[name] for row in selected])) for name in metric_names},
        })
    return aggregate


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    value = judge_metrics(labels, predictions, class_values=CLASSES)
    return {
        "mae": float(value["mae"]),
        "exact_accuracy": float(value["accuracy"]),
        "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
        "quadratic_weighted_kappa": float(value["qwk"]),
    }


def shift_type(source_domain: str, source_skill: str, target_domain: str, target_skill: str) -> str:
    if target_domain == source_domain and target_skill == source_skill:
        return "ID test"
    if target_domain != source_domain and target_skill == source_skill:
        return "Domain shift"
    if target_domain == source_domain and target_skill != source_skill:
        return "Task shift"
    return "Joint shift"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "head_id", "source_domain", "source_skill", "target_domain", "target_skill", "test_type",
        "rows", "excluded_train_question_rows", "mae", "exact_accuracy",
        "plus_minus_1_accuracy", "quadratic_weighted_kappa",
        "train_question_ids_excluded", "acceptance",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "| source head | target Domain | target Skill | 测试类型 | B 行数 | MAE | "
            "Exact Accuracy | ±1 Accuracy | QWK | 验收 |\n"
        )
        handle.write("|---|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                f"| `{row['head_id']}` | {row['target_domain']} | {row['target_skill']} | "
                f"{row['test_type']} | {row['rows']} | {row['mae']:.4f} | "
                f"{row['exact_accuracy']:.4f} | {row['plus_minus_1_accuracy']:.4f} | "
                f"{row['quadratic_weighted_kappa']:.4f} | 通过 |\n"
            )


def write_target_cell_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "target_domain", "target_skill", "head_prediction_rows", "source_head_count",
        "mae", "exact_accuracy", "plus_minus_1_accuracy", "quadratic_weighted_kappa",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_target_cell_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write(
            "| target Domain | target Skill | head×B 行数 | source heads | MAE | "
            "Exact Accuracy | ±1 Accuracy | QWK |\n"
        )
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| {row['target_domain']} | {row['target_skill']} | "
                f"{row['head_prediction_rows']} | {row['source_head_count']} | "
                f"{row['mae']:.4f} | {row['exact_accuracy']:.4f} | "
                f"{row['plus_minus_1_accuracy']:.4f} | "
                f"{row['quadratic_weighted_kappa']:.4f} |\n"
            )


def write_head_shift_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "head_id", "source_domain", "source_skill", "test_type", "target_cell_count",
        "head_prediction_rows", "mae", "exact_accuracy", "plus_minus_1_accuracy",
        "quadratic_weighted_kappa",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_shift_type_summary_csv(
    path: Path, micro_rows: list[dict[str, Any]], macro_rows: list[dict[str, Any]]
) -> None:
    macro_by_type = {row["test_type"]: row for row in macro_rows}
    fields = (
        "test_type", "head_cell_result_count", "head_prediction_rows",
        "micro_mae", "micro_exact_accuracy", "micro_plus_minus_1_accuracy", "micro_quadratic_weighted_kappa",
        "macro_cell_mae", "macro_cell_exact_accuracy", "macro_cell_plus_minus_1_accuracy",
        "macro_cell_quadratic_weighted_kappa",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for micro in micro_rows:
            macro = macro_by_type[micro["test_type"]]
            writer.writerow({
                "test_type": micro["test_type"],
                "head_cell_result_count": micro["head_cell_result_count"],
                "head_prediction_rows": micro["head_prediction_rows"],
                **{f"micro_{name}": micro[name] for name in (
                    "mae", "exact_accuracy", "plus_minus_1_accuracy", "quadratic_weighted_kappa"
                )},
                **{f"macro_cell_{name}": macro[name] for name in (
                    "mae", "exact_accuracy", "plus_minus_1_accuracy", "quadratic_weighted_kappa"
                )},
            })


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No B-space rows in {path}")
    return rows


def slug(value: str) -> str:
    return "_".join(value.lower().split()).replace("-", "_")


if __name__ == "__main__":
    main()
