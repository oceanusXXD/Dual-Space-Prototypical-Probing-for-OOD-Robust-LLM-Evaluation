#!/usr/bin/env python3
"""Analyze how ViM OOD score relates to Judge-head scoring quality.

For each saved FLASK source-cell head, this script:

1. routes every cached B-space row through that head;
2. fits residual-only ViM on the head's source training rows;
3. scores every non-training row with ViM;
4. bins rows by the ViM score percentile calibrated on held-out source-ID rows;
5. reports MAE and quadratic weighted kappa (QWK) inside each bin.

The main table answers: when ViM says a row is more OOD, does the frozen
linear Judge head become less accurate?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.scores.vim import ViMScorer


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


@dataclass(frozen=True)
class Payload:
    features: np.ndarray
    sample_ids: np.ndarray
    labels: np.ndarray
    domains: np.ndarray
    skills: np.ndarray
    question_ids: np.ndarray
    metadata_json: str


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
        default=Path("artifacts/flask_minimal_validation/vim_score_vs_error_08b_5x6_prelogit"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-question-fraction", type=float, default=0.10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--row-output",
        action="store_true",
        help="Also write the row-level score/error table. This is large but useful for plotting.",
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
    payload = load_feature_payload(args.features)
    head_specs = load_head_specs(args.heads_dir)

    row_records: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for spec in head_specs:
        print(f"analyzing {spec['head_id']}", flush=True)
        result = analyze_head(spec=spec, payload=payload, args=args)
        row_records.extend(result["rows"])
        reference_rows.append(result["reference"])

    calibration_bin_rows = aggregate_bins(row_records, key="calibration_band")
    decile_rows = aggregate_bins(row_records, key="within_head_decile")
    shift_bin_rows = aggregate_bins(row_records, key=("shift_type", "calibration_band"))
    head_bin_rows = aggregate_bins(row_records, key=("head_id", "source_domain", "source_skill", "calibration_band"))
    shift_decile_rows = aggregate_bins(row_records, key=("shift_type", "within_head_decile"))

    write_csv(args.output_dir / "vim_calibration_percentile_bins.csv", calibration_bin_rows)
    write_csv(args.output_dir / "vim_within_head_deciles.csv", decile_rows)
    write_csv(args.output_dir / "vim_shift_calibration_bins.csv", shift_bin_rows)
    write_csv(args.output_dir / "vim_head_calibration_bins.csv", head_bin_rows)
    write_csv(args.output_dir / "vim_shift_deciles.csv", shift_decile_rows)
    write_csv(args.output_dir / "vim_source_reference.csv", reference_rows)
    if args.row_output:
        write_csv(args.output_dir / "vim_row_scores.csv", row_records)

    summary = build_summary(
        args=args,
        payload=payload,
        head_specs=head_specs,
        row_records=row_records,
        reference_rows=reference_rows,
        calibration_bin_rows=calibration_bin_rows,
        decile_rows=decile_rows,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(args.output_dir / "summary.json", summary)
    write_markdown(args.output_dir / "vim_score_vs_error.md", summary, calibration_bin_rows, shift_bin_rows)
    print(json.dumps(compact_summary(summary), ensure_ascii=False, indent=2))


def analyze_head(*, spec: dict[str, Any], payload: Payload, args: argparse.Namespace) -> dict[str, Any]:
    model = joblib.load(spec["model_path"])
    query_ids = tuple(getattr(model, "query_ids_", ()))
    if len(query_ids) != 1:
        raise ValueError(f"{spec['head_id']} must contain exactly one query id, got {query_ids}")

    routed_queries = np.full(len(payload.sample_ids), query_ids[0], dtype=object)
    output = model.predict_output(payload.features, routed_queries)
    h = np.asarray(output.penultimate, dtype=np.float32)
    probabilities = np.asarray(output.probabilities, dtype=np.float64)
    predictions = output.classes[np.argmax(probabilities, axis=1)].astype(int)

    source_domain = str(spec["source_domain"])
    source_skill = str(spec["source_skill"])
    train_questions = set(str(value) for value in spec["split"]["train_question_ids"])
    source_mask = (payload.domains == source_domain) & (payload.skills == source_skill)
    train_mask = source_mask & np.isin(payload.question_ids, np.asarray(sorted(train_questions)))
    remaining_questions = sorted(set(payload.question_ids[source_mask].tolist()).difference(train_questions))
    calibration_questions = stable_question_sample(
        remaining_questions,
        fraction=float(args.calibration_question_fraction),
        seed=int(args.seed),
        namespace=f"{source_domain}::{source_skill}::vim_score_vs_error_calibration",
    )
    id_questions = sorted(set(remaining_questions).difference(calibration_questions))
    calibration_mask = source_mask & np.isin(payload.question_ids, np.asarray(sorted(calibration_questions)))
    id_mask = source_mask & np.isin(payload.question_ids, np.asarray(id_questions))
    if not train_mask.any() or not calibration_mask.any() or not id_mask.any():
        raise RuntimeError(f"Split failed for {spec['head_id']}")

    rank = min(int(args.rank), int(train_mask.sum()) - 2, int(h.shape[1]) - 1)
    scorer = fit_vim_with_rank(h[train_mask], rank)
    vim_scores = np.asarray(scorer.score(h), dtype=np.float64)
    calibration_scores = vim_scores[calibration_mask]
    id_scores = vim_scores[id_mask]
    q90 = float(np.quantile(calibration_scores, 0.90))
    q95 = float(np.quantile(calibration_scores, 0.95))

    eval_mask = ~train_mask
    eval_indices = np.flatnonzero(eval_mask)
    eval_scores = vim_scores[eval_indices]
    calibration_percentiles = calibration_ecdf_percentile(calibration_scores, eval_scores)
    within_percentiles = within_vector_percentile(eval_scores)

    rows: list[dict[str, Any]] = []
    for local_index, index in enumerate(eval_indices.tolist()):
        label = int(payload.labels[index])
        pred = int(predictions[index])
        shift = shift_type(source_domain, source_skill, str(payload.domains[index]), str(payload.skills[index]))
        calibration_percentile = float(calibration_percentiles[local_index])
        within_percentile = float(within_percentiles[local_index])
        rows.append(
            {
                "head_id": str(spec["head_id"]),
                "source_domain": source_domain,
                "source_skill": source_skill,
                "sample_id": str(payload.sample_ids[index]),
                "question_id": str(payload.question_ids[index]),
                "target_domain": str(payload.domains[index]),
                "target_skill": str(payload.skills[index]),
                "shift_type": shift,
                "label": label,
                "prediction": pred,
                "absolute_error": abs(label - pred),
                "vim_score": float(vim_scores[index]),
                "vim_calibration_percentile": calibration_percentile,
                "vim_within_head_percentile": within_percentile,
                "calibration_band": assign_band(calibration_percentile, CALIBRATION_BANDS),
                "within_head_decile": assign_band(within_percentile, DECILE_BANDS),
                "above_calibration_q90": bool(vim_scores[index] >= q90),
                "above_calibration_q95": bool(vim_scores[index] >= q95),
            }
        )

    reference = {
        "head_id": str(spec["head_id"]),
        "source_domain": source_domain,
        "source_skill": source_skill,
        "rank": int(scorer.rank),
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "id_rows": int(id_mask.sum()),
        "eval_rows": int(eval_mask.sum()),
        "calibration_q90": q90,
        "calibration_q95": q95,
        "id_fpr_q95": float(np.mean(id_scores >= q95)),
        "id_score_mean": float(np.mean(id_scores)),
        "id_score_std": float(np.std(id_scores)),
    }
    return {"rows": rows, "reference": reference}


def aggregate_bins(rows: list[dict[str, Any]], *, key: str | tuple[str, ...]) -> list[dict[str, Any]]:
    keys = (key,) if isinstance(key, str) else tuple(key)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(row[item] for item in keys)
        groups.setdefault(group_key, []).append(row)

    out: list[dict[str, Any]] = []
    for group_key, members in groups.items():
        labels = np.asarray([int(row["label"]) for row in members], dtype=int)
        predictions = np.asarray([int(row["prediction"]) for row in members], dtype=int)
        scores = np.asarray([float(row["vim_score"]) for row in members], dtype=float)
        errors = np.asarray([float(row["absolute_error"]) for row in members], dtype=float)
        percentiles = np.asarray([float(row["vim_calibration_percentile"]) for row in members], dtype=float)
        record = {keys[index]: group_key[index] for index in range(len(keys))}
        record.update(
            {
                "rows": int(len(members)),
                "unique_questions": int(len(set(str(row["question_id"]) for row in members))),
                "mae": float(np.mean(errors)) if len(errors) else float("nan"),
                "exact_accuracy": float(np.mean(labels == predictions)) if len(labels) else float("nan"),
                "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)) if len(labels) else float("nan"),
                "quadratic_weighted_kappa": qwk(labels, predictions),
                "vim_score_mean": float(np.mean(scores)) if len(scores) else float("nan"),
                "vim_score_median": float(np.median(scores)) if len(scores) else float("nan"),
                "vim_calibration_percentile_mean": float(np.mean(percentiles)) if len(percentiles) else float("nan"),
                "mean_absolute_error_std": float(np.std(errors)) if len(errors) else float("nan"),
            }
        )
        out.append(record)
    return sorted(out, key=sort_key)


def build_summary(
    *,
    args: argparse.Namespace,
    payload: Payload,
    head_specs: list[dict[str, Any]],
    row_records: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    calibration_bin_rows: list[dict[str, Any]],
    decile_rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    labels = np.asarray([int(row["label"]) for row in row_records], dtype=int)
    predictions = np.asarray([int(row["prediction"]) for row in row_records], dtype=int)
    errors = np.asarray([float(row["absolute_error"]) for row in row_records], dtype=float)
    raw_scores = np.asarray([float(row["vim_score"]) for row in row_records], dtype=float)
    calibration_percentiles = np.asarray([float(row["vim_calibration_percentile"]) for row in row_records], dtype=float)
    within_percentiles = np.asarray([float(row["vim_within_head_percentile"]) for row in row_records], dtype=float)
    spearman_raw = safe_spearman(raw_scores, errors)
    spearman_calibrated = safe_spearman(calibration_percentiles, errors)
    spearman_within = safe_spearman(within_percentiles, errors)
    low_band = next((row for row in calibration_bin_rows if row["calibration_band"] == "p00_p50"), None)
    high_band = next((row for row in calibration_bin_rows if row["calibration_band"] == "p95_p100"), None)
    trend = {}
    if low_band and high_band:
        trend = {
            "high_minus_low_mae": float(high_band["mae"] - low_band["mae"]),
            "high_minus_low_qwk": float(high_band["quadratic_weighted_kappa"] - low_band["quadratic_weighted_kappa"]),
            "high_band": high_band["calibration_band"],
            "low_band": low_band["calibration_band"],
        }
    return {
        "artifact_type": "flask_vim_score_vs_judge_error_v1",
        "source_features": str(args.features),
        "heads_dir": str(args.heads_dir),
        "feature_shape": list(payload.features.shape),
        "feature_rows": int(len(payload.sample_ids)),
        "head_count": int(len(head_specs)),
        "row_records": int(len(row_records)),
        "rank_requested": int(args.rank),
        "calibration_question_fraction": float(args.calibration_question_fraction),
        "overall_metrics": {
            "rows": int(len(row_records)),
            "mae": float(np.mean(errors)),
            "exact_accuracy": float(np.mean(labels == predictions)),
            "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
            "quadratic_weighted_kappa": qwk(labels, predictions),
        },
        "vim_error_association": {
            "spearman_raw_vim_score_vs_abs_error": spearman_raw,
            "spearman_calibrated_percentile_vs_abs_error": spearman_calibrated,
            "spearman_within_head_percentile_vs_abs_error": spearman_within,
            **trend,
        },
        "calibration_bins": calibration_bin_rows,
        "within_head_deciles": decile_rows,
        "reference_by_head": reference_rows,
        "outputs": {
            "calibration_bins": str(args.output_dir / "vim_calibration_percentile_bins.csv"),
            "within_head_deciles": str(args.output_dir / "vim_within_head_deciles.csv"),
            "shift_calibration_bins": str(args.output_dir / "vim_shift_calibration_bins.csv"),
            "head_calibration_bins": str(args.output_dir / "vim_head_calibration_bins.csv"),
            "source_reference": str(args.output_dir / "vim_source_reference.csv"),
            "markdown": str(args.output_dir / "vim_score_vs_error.md"),
        },
        "elapsed_seconds": float(elapsed_seconds),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_records": summary["row_records"],
        "overall_metrics": summary["overall_metrics"],
        "vim_error_association": summary["vim_error_association"],
        "outputs": summary["outputs"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


def write_markdown(
    path: Path,
    summary: dict[str, Any],
    calibration_bin_rows: list[dict[str, Any]],
    shift_bin_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# ViM OOD score vs Judge-head scoring error",
        "",
        "Rows are binned by each source head's residual-only ViM score percentile calibrated on held-out source-ID rows.",
        "",
        "## Overall calibration-percentile bins",
        "",
        "| ViM calibration bin | Rows | MAE ↓ | Exact ↑ | ±1 Acc ↑ | QWK ↑ | Mean ViM percentile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in calibration_bin_rows:
        lines.append(
            "| {calibration_band} | {rows} | {mae:.3f} | {exact_accuracy:.1%} | {plus_minus_1_accuracy:.1%} | {quadratic_weighted_kappa:.3f} | {vim_calibration_percentile_mean:.3f} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Shift-type calibration-percentile bins",
            "",
            "| Shift | ViM calibration bin | Rows | MAE ↓ | Exact ↑ | ±1 Acc ↑ | QWK ↑ |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in shift_bin_rows:
        lines.append(
            "| {shift_type} | {calibration_band} | {rows} | {mae:.3f} | {exact_accuracy:.1%} | {plus_minus_1_accuracy:.1%} | {quadratic_weighted_kappa:.3f} |".format(
                **row
            )
        )
    assoc = summary["vim_error_association"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Spearman(raw ViM score, absolute error): {assoc['spearman_raw_vim_score_vs_abs_error']['rho']:.4f} (p={assoc['spearman_raw_vim_score_vs_abs_error']['p_value']:.3g}).",
            f"- Spearman(calibrated ViM percentile, absolute error): {assoc['spearman_calibrated_percentile_vs_abs_error']['rho']:.4f} (p={assoc['spearman_calibrated_percentile_vs_abs_error']['p_value']:.3g}).",
            f"- Top calibration band minus bottom calibration band MAE: {assoc.get('high_minus_low_mae', float('nan')):.3f}.",
            f"- Top calibration band minus bottom calibration band QWK: {assoc.get('high_minus_low_qwk', float('nan')):.3f}.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def load_feature_payload(path: Path) -> Payload:
    with np.load(path, allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float16)
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        labels = np.asarray(cache["labels"], dtype=int)
        domains = np.asarray(cache["domain_ids"]).astype(str)
        skills = np.asarray(cache["task_ids"]).astype(str)
        question_ids = np.asarray(cache["query_ids"]).astype(str)
        metadata_json = str(np.asarray(cache["metadata_json"]).item()) if "metadata_json" in cache.files else "{}"
    expected = len(sample_ids)
    for name, values in {
        "labels": labels,
        "domain_ids": domains,
        "task_ids": skills,
        "query_ids": question_ids,
    }.items():
        if len(values) != expected:
            raise ValueError(f"{name} does not align with sample_ids")
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3:
        raise ValueError(f"Expected [N,L,D] or [N,D] feature cache, got {features.shape}")
    return Payload(
        features=features,
        sample_ids=sample_ids,
        labels=labels,
        domains=domains,
        skills=skills,
        question_ids=question_ids,
        metadata_json=metadata_json,
    )


def load_head_specs(path: Path) -> list[dict[str, Any]]:
    summary_path = path / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing head summary: {summary_path}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    heads = payload.get("heads")
    if not isinstance(heads, list) or not heads:
        raise ValueError(f"No heads found in {summary_path}")
    return sorted(heads, key=lambda row: (str(row["source_domain"]), str(row["source_skill"])))


def fit_vim_with_rank(features: np.ndarray, rank: int) -> ViMScorer:
    errors: list[str] = []
    for candidate in range(max(1, int(rank)), 0, -1):
        try:
            return ViMScorer(rank=candidate).fit(features)
        except ValueError as exc:
            errors.append(f"rank={candidate}: {exc}")
    raise RuntimeError("Could not fit residual ViM; " + " | ".join(errors[-3:]))


def calibration_ecdf_percentile(calibration_scores: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    if reference.size == 0:
        raise ValueError("Calibration scores are empty")
    return np.searchsorted(reference, np.asarray(values, dtype=np.float64), side="right") / float(reference.size)


def within_vector_percentile(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = (np.arange(len(x), dtype=np.float64) + 1.0) / max(1.0, float(len(x)))
    return ranks


def assign_band(value: float, bands: tuple[tuple[float, float, str], ...]) -> str:
    for low, high, name in bands:
        if low <= float(value) < high:
            return name
    if float(value) >= bands[-1][1]:
        return bands[-1][2]
    return bands[0][2]


def stable_question_sample(questions: list[str], *, fraction: float, seed: int, namespace: str) -> set[str]:
    import hashlib

    if not questions:
        return set()
    n = max(1, int(round(len(questions) * fraction)))
    n = min(n, len(questions))
    ordered = sorted(
        questions,
        key=lambda item: hashlib.sha256(f"{seed}::{namespace}::{item}".encode("utf-8")).hexdigest(),
    )
    return set(ordered[:n])


def shift_type(source_domain: str, source_skill: str, target_domain: str, target_skill: str) -> str:
    if target_domain == source_domain and target_skill == source_skill:
        return "ID test"
    if target_domain != source_domain and target_skill != source_skill:
        return "Joint shift"
    if target_domain != source_domain:
        return "Domain shift"
    return "Task shift"


def qwk(labels: np.ndarray, predictions: np.ndarray) -> float:
    if len(labels) == 0:
        return float("nan")
    try:
        value = cohen_kappa_score(labels, predictions, labels=list(CLASSES), weights="quadratic")
    except ValueError:
        return float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def safe_spearman(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return {"rho": float("nan"), "p_value": float("nan")}
    result = spearmanr(x, y)
    return {"rho": float(result.statistic), "p_value": float(result.pvalue)}


def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    order = {
        "ID test": 0,
        "Domain shift": 1,
        "Task shift": 2,
        "Joint shift": 3,
        "p00_p50": 0,
        "p50_p75": 1,
        "p75_p90": 2,
        "p90_p95": 3,
        "p95_p100": 4,
    }
    values = []
    for key in ("shift_type", "head_id", "source_domain", "source_skill", "calibration_band", "within_head_decile"):
        if key in row:
            values.append(order.get(str(row[key]), row[key]))
    return tuple(values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
