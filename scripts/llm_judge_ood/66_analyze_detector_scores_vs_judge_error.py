#!/usr/bin/env python3
"""Compare sample-level OOD detector scores with Judge scoring quality.

MAE has a row-level primitive (absolute error), so the script reports its
Spearman association with each detector score. QWK is only defined for a set
of predictions, so its association is measured over 90 head-by-risk-decile
groups. All detector directions follow the existing FLASK detector suite:
higher means more OOD.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


CLASSES = (1, 2, 3, 4, 5)
CALIBRATION_BANDS = (
    (0.00, 0.50, "p00_p50"),
    (0.50, 0.75, "p50_p75"),
    (0.75, 0.90, "p75_p90"),
    (0.90, 0.95, "p90_p95"),
    (0.95, 1.01, "p95_p100"),
)
DECILE_BANDS = tuple(
    (i / 10.0, (i + 1) / 10.0 if i < 9 else 1.01, f"d{i + 1:02d}")
    for i in range(10)
)
SHIFT_ORDER = ("ID test", "Domain shift", "Task shift", "Joint shift")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/"
            "qwen35_08b_5x6_digit_direct_judge_strict_prelogit_bspace/"
            "strict_final_prelogit_b_space_features.npz"
        ),
    )
    parser.add_argument(
        "--heads-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/qwen35_08b_strict_prelogit_3x3_heads"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/"
            "detector_scores_vs_error_08b_5x6_prelogit"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-question-fraction", type=float, default=0.10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--detectors",
        default="all",
        help="all, basic, or comma-separated detector ids from the FLASK detector suite.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir is non-empty: {args.output_dir}; pass --overwrite")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    suite = load_detector_suite()
    detector_ids = suite.selected_detectors(str(args.detectors))
    payload = suite.load_feature_payload(args.features)
    head_specs = suite.load_head_specs(args.heads_dir)

    chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reference_rows: list[dict[str, Any]] = []
    for head_spec in head_specs:
        print(f"analyzing {head_spec['head_id']}", flush=True)
        run_head(
            spec=head_spec,
            payload=payload,
            args=args,
            suite=suite,
            detector_ids=detector_ids,
            chunks=chunks,
            reference_rows=reference_rows,
        )

    summary_rows: list[dict[str, Any]] = []
    calibration_bin_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    head_decile_rows: list[dict[str, Any]] = []
    shift_bin_rows: list[dict[str, Any]] = []
    shift_association_rows: list[dict[str, Any]] = []
    for detector in detector_ids:
        detector_chunks = chunks.get(detector, [])
        if not detector_chunks:
            summary_rows.append(failure_summary(detector, suite.DISPLAY_NAME[detector], reference_rows))
            continue
        result = analyze_detector(
            detector=detector,
            detector_name=suite.DISPLAY_NAME[detector],
            chunks=detector_chunks,
        )
        summary_rows.append(result["summary"])
        calibration_bin_rows.extend(result["calibration_bins"])
        decile_rows.extend(result["deciles"])
        head_decile_rows.extend(result["head_deciles"])
        shift_bin_rows.extend(result["shift_bins"])
        shift_association_rows.extend(result["shift_associations"])

    summary_rows.sort(key=lambda row: correlation_sort_value(row), reverse=True)
    write_csv(args.output_dir / "detector_error_associations.csv", summary_rows)
    write_csv(args.output_dir / "detector_calibration_percentile_bins.csv", calibration_bin_rows)
    write_csv(args.output_dir / "detector_within_head_deciles.csv", decile_rows)
    write_csv(args.output_dir / "detector_head_decile_groups.csv", head_decile_rows)
    write_csv(args.output_dir / "detector_shift_calibration_bins.csv", shift_bin_rows)
    write_csv(args.output_dir / "detector_shift_error_associations.csv", shift_association_rows)
    write_csv(args.output_dir / "detector_fit_reference.csv", reference_rows)

    summary = {
        "artifact_type": "flask_detector_score_vs_judge_error_v1",
        "source_features": str(args.features),
        "heads_dir": str(args.heads_dir),
        "feature_rows": int(len(payload["sample_ids"])),
        "feature_shape": list(payload["features"].shape),
        "head_count": len(head_specs),
        "detectors_requested": list(detector_ids),
        "detectors_completed": [row["detector"] for row in summary_rows if row["status"] == "ok"],
        "evaluation_unit": "head_by_nontraining_row",
        "qwk_association_unit": "head_by_within_head_detector_decile",
        "rank_requested": int(args.rank),
        "calibration_question_fraction": float(args.calibration_question_fraction),
        "detector_error_associations": summary_rows,
        "detector_shift_error_associations": shift_association_rows,
        "elapsed_seconds": float(time.perf_counter() - started),
        "outputs": {
            "associations": str(args.output_dir / "detector_error_associations.csv"),
            "calibration_bins": str(args.output_dir / "detector_calibration_percentile_bins.csv"),
            "within_head_deciles": str(args.output_dir / "detector_within_head_deciles.csv"),
            "head_decile_groups": str(args.output_dir / "detector_head_decile_groups.csv"),
            "shift_bins": str(args.output_dir / "detector_shift_calibration_bins.csv"),
            "shift_associations": str(args.output_dir / "detector_shift_error_associations.csv"),
            "fit_reference": str(args.output_dir / "detector_fit_reference.csv"),
        },
    }
    write_json(args.output_dir / "summary.json", summary)
    write_markdown(
        args.output_dir / "detector_scores_vs_error.md", summary_rows, shift_association_rows
    )
    print(json.dumps(compact_summary(summary), ensure_ascii=False, indent=2))


def run_head(
    *,
    spec: dict[str, Any],
    payload: dict[str, np.ndarray],
    args: argparse.Namespace,
    suite: ModuleType,
    detector_ids: tuple[str, ...],
    chunks: dict[str, list[dict[str, Any]]],
    reference_rows: list[dict[str, Any]],
) -> None:
    model = joblib.load(spec["model_path"])
    query_ids = tuple(getattr(model, "query_ids_", ()))
    if len(query_ids) != 1:
        raise ValueError(f"{spec['head_id']} must contain exactly one query id, got {query_ids}")
    routed_queries = np.full(len(payload["features"]), query_ids[0], dtype=object)
    output = model.predict_output(payload["features"], routed_queries)
    h = np.asarray(output.penultimate, dtype=np.float32)
    logits = np.asarray(output.logits, dtype=np.float32)
    probabilities = np.asarray(output.probabilities, dtype=np.float64)
    predictions = output.classes[np.argmax(probabilities, axis=1)].astype(int)

    source_domain = str(spec["source_domain"])
    source_skill = str(spec["source_skill"])
    train_questions = set(str(value) for value in spec["split"]["train_question_ids"])
    source_mask = (payload["domains"] == source_domain) & (payload["skills"] == source_skill)
    train_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
    remaining_questions = sorted(set(payload["question_ids"][source_mask].tolist()).difference(train_questions))
    calibration_questions = suite.stable_question_sample(
        remaining_questions,
        fraction=float(args.calibration_question_fraction),
        seed=int(args.seed),
        namespace=f"{source_domain}::{source_skill}::detector_calibration",
    )
    id_questions = sorted(set(remaining_questions).difference(calibration_questions))
    calibration_mask = source_mask & np.isin(
        payload["question_ids"], np.asarray(sorted(calibration_questions))
    )
    id_mask = source_mask & np.isin(payload["question_ids"], np.asarray(id_questions))
    if not train_mask.any() or not calibration_mask.any() or not id_mask.any():
        raise RuntimeError(f"Detector split failed for {spec['head_id']}")

    weights, biases, head_query_ids = model.affine_head_parameters()
    fitted = suite.fit_detectors(
        h=h,
        logits=logits,
        labels=payload["labels"],
        train_mask=train_mask,
        rank=min(int(args.rank), int(train_mask.sum()) - 2, h.shape[1] - 1),
        routed_queries=routed_queries,
        weights=weights,
        biases=biases,
        head_query_ids=head_query_ids,
        detector_ids=detector_ids,
    )
    eval_mask = ~train_mask
    eval_indices = np.flatnonzero(eval_mask)
    shifts = np.asarray(
        [
            suite.shift_type(source_domain, source_skill, str(payload["domains"][i]), str(payload["skills"][i]))
            for i in eval_indices
        ],
        dtype=object,
    )
    labels = payload["labels"][eval_indices].astype(int)
    preds = predictions[eval_indices].astype(int)
    errors = np.abs(labels - preds).astype(float)

    for detector in detector_ids:
        detector_fit = fitted[detector]
        reference = {
            "detector": detector,
            "detector_name": suite.DISPLAY_NAME[detector],
            "head_id": str(spec["head_id"]),
            "source_domain": source_domain,
            "source_skill": source_skill,
            "status": detector_fit["status"],
            "train_rows": int(train_mask.sum()),
            "calibration_rows": int(calibration_mask.sum()),
            "id_rows": int(id_mask.sum()),
            "eval_rows": int(eval_mask.sum()),
            "error": detector_fit.get("error", ""),
        }
        reference_rows.append(reference)
        if detector_fit["status"] != "ok":
            continue
        all_scores = np.asarray(
            detector_fit["score_fn"](np.ones(len(h), dtype=bool)), dtype=np.float64
        )
        if not np.isfinite(all_scores).all():
            reference["status"] = "not_applicable"
            reference["error"] = "detector produced non-finite scores"
            continue
        calibration_scores = all_scores[calibration_mask]
        eval_scores = all_scores[eval_indices]
        chunks[detector].append(
            {
                "head_id": str(spec["head_id"]),
                "source_domain": source_domain,
                "source_skill": source_skill,
                "labels": labels,
                "predictions": preds,
                "errors": errors,
                "raw_scores": eval_scores,
                "calibration_percentiles": calibration_ecdf_percentile(calibration_scores, eval_scores),
                "within_percentiles": within_vector_percentile(eval_scores),
                "shifts": shifts,
            }
        )


def analyze_detector(
    *, detector: str, detector_name: str, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    labels = np.concatenate([chunk["labels"] for chunk in chunks])
    predictions = np.concatenate([chunk["predictions"] for chunk in chunks])
    errors = np.concatenate([chunk["errors"] for chunk in chunks])
    raw_scores = np.concatenate([chunk["raw_scores"] for chunk in chunks])
    calibration_percentiles = np.concatenate([chunk["calibration_percentiles"] for chunk in chunks])
    within_percentiles = np.concatenate([chunk["within_percentiles"] for chunk in chunks])
    shifts = np.concatenate([chunk["shifts"] for chunk in chunks])

    calibration_bins = aggregate_array_bins(
        detector, detector_name, labels, predictions, raw_scores, calibration_percentiles,
        CALIBRATION_BANDS, "calibration_band"
    )
    deciles = aggregate_array_bins(
        detector, detector_name, labels, predictions, raw_scores, within_percentiles,
        DECILE_BANDS, "within_head_decile"
    )
    shift_bins: list[dict[str, Any]] = []
    shift_associations: list[dict[str, Any]] = []
    for shift in SHIFT_ORDER:
        shift_mask = shifts == shift
        local = aggregate_array_bins(
            detector,
            detector_name,
            labels[shift_mask],
            predictions[shift_mask],
            raw_scores[shift_mask],
            calibration_percentiles[shift_mask],
            CALIBRATION_BANDS,
            "calibration_band",
        )
        for row in local:
            row["shift_type"] = shift
        shift_bins.extend(local)
        shift_associations.append(
            analyze_shift_association(
                detector=detector,
                detector_name=detector_name,
                shift=shift,
                chunks=chunks,
                labels=labels[shift_mask],
                predictions=predictions[shift_mask],
                errors=errors[shift_mask],
                raw_scores=raw_scores[shift_mask],
                calibration_percentiles=calibration_percentiles[shift_mask],
                within_percentiles=within_percentiles[shift_mask],
            )
        )

    head_deciles: list[dict[str, Any]] = []
    for chunk in chunks:
        rows = aggregate_array_bins(
            detector,
            detector_name,
            chunk["labels"],
            chunk["predictions"],
            chunk["raw_scores"],
            chunk["within_percentiles"],
            DECILE_BANDS,
            "within_head_decile",
        )
        for index, row in enumerate(rows, start=1):
            row.update(
                {
                    "head_id": chunk["head_id"],
                    "source_domain": chunk["source_domain"],
                    "source_skill": chunk["source_skill"],
                    "decile_index": index,
                }
            )
        head_deciles.extend(rows)

    group_risk = np.asarray([float(row["decile_index"]) for row in head_deciles])
    group_mae = np.asarray([float(row["mae"]) for row in head_deciles])
    group_qwk = np.asarray([float(row["qwk"]) for row in head_deciles])
    group_mae_assoc = safe_spearman(group_risk, group_mae)
    group_qwk_assoc = safe_spearman(group_risk, group_qwk)
    raw_assoc = safe_spearman(raw_scores, errors)
    calibration_assoc = safe_spearman(calibration_percentiles, errors)
    within_assoc = safe_spearman(within_percentiles, errors)
    low = next(row for row in deciles if row["within_head_decile"] == "d01")
    high = next(row for row in deciles if row["within_head_decile"] == "d10")

    summary = {
        "detector": detector,
        "detector_name": detector_name,
        "status": "ok",
        "head_count": len(chunks),
        "row_records": len(labels),
        "sample_rho_raw_score_abs_error": raw_assoc["rho"],
        "sample_p_raw_score_abs_error": raw_assoc["p_value"],
        "sample_rho_calibrated_percentile_abs_error": calibration_assoc["rho"],
        "sample_p_calibrated_percentile_abs_error": calibration_assoc["p_value"],
        "sample_rho_within_head_percentile_abs_error": within_assoc["rho"],
        "sample_p_within_head_percentile_abs_error": within_assoc["p_value"],
        "group_rho_risk_decile_mae": group_mae_assoc["rho"],
        "group_p_risk_decile_mae": group_mae_assoc["p_value"],
        "group_rho_risk_decile_qwk": group_qwk_assoc["rho"],
        "group_p_risk_decile_qwk": group_qwk_assoc["p_value"],
        "low_decile_mae": low["mae"],
        "high_decile_mae": high["mae"],
        "high_minus_low_mae": float(high["mae"] - low["mae"]),
        "low_decile_qwk": low["qwk"],
        "high_decile_qwk": high["qwk"],
        "high_minus_low_qwk": float(high["qwk"] - low["qwk"]),
        "low_decile_plus_minus_1_accuracy": low["plus_minus_1_accuracy"],
        "high_decile_plus_minus_1_accuracy": high["plus_minus_1_accuracy"],
    }
    return {
        "summary": summary,
        "calibration_bins": calibration_bins,
        "deciles": deciles,
        "head_deciles": head_deciles,
        "shift_bins": shift_bins,
        "shift_associations": shift_associations,
    }


def analyze_shift_association(
    *,
    detector: str,
    detector_name: str,
    shift: str,
    chunks: list[dict[str, Any]],
    labels: np.ndarray,
    predictions: np.ndarray,
    errors: np.ndarray,
    raw_scores: np.ndarray,
    calibration_percentiles: np.ndarray,
    within_percentiles: np.ndarray,
) -> dict[str, Any]:
    pooled_deciles = aggregate_array_bins(
        detector,
        detector_name,
        labels,
        predictions,
        raw_scores,
        within_percentiles,
        DECILE_BANDS,
        "within_head_decile",
    )
    group_rows: list[dict[str, Any]] = []
    for chunk in chunks:
        local_mask = chunk["shifts"] == shift
        if not local_mask.any():
            continue
        rows = aggregate_array_bins(
            detector,
            detector_name,
            chunk["labels"][local_mask],
            chunk["predictions"][local_mask],
            chunk["raw_scores"][local_mask],
            chunk["within_percentiles"][local_mask],
            DECILE_BANDS,
            "within_head_decile",
        )
        for index, row in enumerate(rows, start=1):
            if int(row["rows"]) > 0:
                row["decile_index"] = index
                group_rows.append(row)

    group_risk = np.asarray([float(row["decile_index"]) for row in group_rows])
    group_mae = np.asarray([float(row["mae"]) for row in group_rows])
    group_qwk = np.asarray([float(row["qwk"]) for row in group_rows])
    raw_assoc = safe_spearman(raw_scores, errors)
    calibration_assoc = safe_spearman(calibration_percentiles, errors)
    within_assoc = safe_spearman(within_percentiles, errors)
    group_mae_assoc = safe_spearman(group_risk, group_mae)
    group_qwk_assoc = safe_spearman(group_risk, group_qwk)
    low = next(row for row in pooled_deciles if row["within_head_decile"] == "d01")
    high = next(row for row in pooled_deciles if row["within_head_decile"] == "d10")
    return {
        "detector": detector,
        "detector_name": detector_name,
        "shift_type": shift,
        "rows": int(len(labels)),
        "sample_rho_raw_score_abs_error": raw_assoc["rho"],
        "sample_p_raw_score_abs_error": raw_assoc["p_value"],
        "sample_rho_calibrated_percentile_abs_error": calibration_assoc["rho"],
        "sample_p_calibrated_percentile_abs_error": calibration_assoc["p_value"],
        "sample_rho_within_head_percentile_abs_error": within_assoc["rho"],
        "sample_p_within_head_percentile_abs_error": within_assoc["p_value"],
        "head_decile_groups": len(group_rows),
        "finite_qwk_groups": int(np.isfinite(group_qwk).sum()),
        "group_rho_risk_decile_mae": group_mae_assoc["rho"],
        "group_p_risk_decile_mae": group_mae_assoc["p_value"],
        "group_rho_risk_decile_qwk": group_qwk_assoc["rho"],
        "group_p_risk_decile_qwk": group_qwk_assoc["p_value"],
        "low_decile_rows": low["rows"],
        "high_decile_rows": high["rows"],
        "low_decile_mae": low["mae"],
        "high_decile_mae": high["mae"],
        "high_minus_low_mae": float(high["mae"] - low["mae"]),
        "low_decile_qwk": low["qwk"],
        "high_decile_qwk": high["qwk"],
        "high_minus_low_qwk": float(high["qwk"] - low["qwk"]),
    }


def aggregate_array_bins(
    detector: str,
    detector_name: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    raw_scores: np.ndarray,
    risk_percentiles: np.ndarray,
    bands: tuple[tuple[float, float, str], ...],
    band_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for low, high, name in bands:
        mask = (risk_percentiles >= low) & (risk_percentiles < high)
        local_labels = labels[mask]
        local_predictions = predictions[mask]
        local_errors = np.abs(local_labels - local_predictions)
        rows.append(
            {
                "detector": detector,
                "detector_name": detector_name,
                band_key: name,
                "rows": int(mask.sum()),
                "mae": float(np.mean(local_errors)) if mask.any() else float("nan"),
                "exact_accuracy": float(np.mean(local_errors == 0)) if mask.any() else float("nan"),
                "plus_minus_1_accuracy": float(np.mean(local_errors <= 1)) if mask.any() else float("nan"),
                "qwk": qwk(local_labels, local_predictions),
                "score_mean": float(np.mean(raw_scores[mask])) if mask.any() else float("nan"),
                "risk_percentile_mean": float(np.mean(risk_percentiles[mask])) if mask.any() else float("nan"),
            }
        )
    return rows


def failure_summary(
    detector: str, detector_name: str, references: list[dict[str, Any]]
) -> dict[str, Any]:
    failures = [row["error"] for row in references if row["detector"] == detector and row["error"]]
    return {
        "detector": detector,
        "detector_name": detector_name,
        "status": "not_applicable",
        "head_count": 0,
        "row_records": 0,
        "error": " | ".join(sorted(set(failures))),
    }


def calibration_ecdf_percentile(reference_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference_scores, dtype=np.float64))
    if reference.size == 0:
        raise ValueError("Calibration scores are empty")
    return np.searchsorted(reference, np.asarray(values, dtype=np.float64), side="right") / float(reference.size)


def within_vector_percentile(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = (np.arange(len(x), dtype=np.float64) + 1.0) / max(1.0, float(len(x)))
    return ranks


def qwk(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    value = cohen_kappa_score(labels, predictions, labels=list(CLASSES), weights="quadratic")
    return float(value) if np.isfinite(value) else float("nan")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    values_x = np.asarray(x, dtype=np.float64)
    values_y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(values_x) & np.isfinite(values_y)
    values_x = values_x[finite]
    values_y = values_y[finite]
    if len(values_x) < 2 or float(np.std(values_x)) <= 1e-12 or float(np.std(values_y)) <= 1e-12:
        return {"rho": float("nan"), "p_value": float("nan")}
    result = spearmanr(values_x, values_y)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def load_detector_suite() -> ModuleType:
    path = ROOT / "scripts/llm_judge_ood/64_run_flask_detector_suite.py"
    spec = importlib.util.spec_from_file_location("flask_detector_suite", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load detector suite from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def correlation_sort_value(row: dict[str, Any]) -> float:
    try:
        return float(row["sample_rho_within_head_percentile_abs_error"])
    except (KeyError, TypeError, ValueError):
        return float("-inf")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_rows": summary["feature_rows"],
        "head_count": summary["head_count"],
        "detectors_completed": len(summary["detectors_completed"]),
        "elapsed_seconds": summary["elapsed_seconds"],
        "ranked_associations": [
            {
                "detector": row["detector_name"],
                "rho_abs_error": row.get("sample_rho_within_head_percentile_abs_error"),
                "rho_group_qwk": row.get("group_rho_risk_decile_qwk"),
                "delta_mae_d10_d01": row.get("high_minus_low_mae"),
                "delta_qwk_d10_d01": row.get("high_minus_low_qwk"),
            }
            for row in summary["detector_error_associations"]
        ],
    }


def write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    shift_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Detector OOD scores vs Judge-head scoring error",
        "",
        "MAE association is sample-level Spearman correlation between within-head detector percentile and absolute error. QWK association is Spearman correlation across 90 head-by-decile groups.",
        "",
        "| Detector | rho(score, abs error) | rho(risk decile, group MAE) | rho(risk decile, group QWK) | D1 MAE | D10 MAE | Delta MAE | D1 QWK | D10 QWK | Delta QWK |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "ok":
            continue
        lines.append(
            "| {detector_name} | {sample_rho_within_head_percentile_abs_error:.4f} | "
            "{group_rho_risk_decile_mae:.4f} | {group_rho_risk_decile_qwk:.4f} | "
            "{low_decile_mae:.3f} | {high_decile_mae:.3f} | {high_minus_low_mae:+.3f} | "
            "{low_decile_qwk:.3f} | {high_decile_qwk:.3f} | {high_minus_low_qwk:+.3f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Positive MAE rho/delta and negative QWK rho/delta mean that higher detector risk is associated with worse scoring quality.",
            "",
            "## Within-shift sample-level error associations",
            "",
            "| Detector | ID rho | Domain rho | Task rho | Joint rho |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    by_detector_shift = {
        (row["detector"], row["shift_type"]): row for row in shift_rows
    }
    for row in rows:
        if row["status"] != "ok":
            continue
        values = [by_detector_shift[(row["detector"], shift)] for shift in SHIFT_ORDER]
        lines.append(
            "| {} | {} |".format(
                row["detector_name"],
                " | ".join(
                    f"{float(value['sample_rho_within_head_percentile_abs_error']):.4f}"
                    for value in values
                ),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
