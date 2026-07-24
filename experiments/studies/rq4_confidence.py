#!/usr/bin/env python3
"""Run FLASK RQ4: three-leg confidence with conformal selective risk control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
from scipy.stats import beta
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, cohen_kappa_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.algorithm.wsr.certification import certify_wsr_thresholds, normalized_absolute_error
from src.algorithm.detector.residual_vim import ViMScorer


SOURCE_DOMAINS = ("Humanities", "Language", "Social Science")
SOURCE_SKILLS = ("Comprehension", "Factuality", "Logical Correctness")
TARGET_DOMAINS = ("Humanities", "Language", "Social Science", "History", "Culture")
TARGET_SKILLS = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
)
CLASSES = (1, 2, 3, 4, 5)
PRIMARY_LOSSES = ("pm1_error", "exact_error")


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
        default=Path("artifacts/flask_minimal_validation/rq4_conformal_08b_5x6_prelogit"),
    )
    parser.add_argument("--primary-loss", choices=PRIMARY_LOSSES, default="pm1_error")
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument(
        "--alpha-grid",
        default="",
        help="Optional comma-separated alpha values. If set, overrides --alpha for threshold sweeps.",
    )
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument(
        "--wsr-normalized-mae-bound",
        type=float,
        default=0.125,
        help="Selective normalized MAE bound b for finite-population WSR certification.",
    )
    parser.add_argument("--wsr-max-threshold-candidates", type=int, default=32)
    parser.add_argument(
        "--wsr-calibration-fraction",
        type=float,
        default=0.10,
        help="SRSWOR row fraction drawn from each certified finite population.",
    )
    parser.add_argument("--wsr-calibration-size", type=int, default=0)
    parser.add_argument("--calibration-question-fraction", type=float, default=0.10)
    parser.add_argument(
        "--error-probe-question-fraction",
        type=float,
        default=0.20,
        help="Held-out source questions used only to train g(h); disjoint from conformal calibration.",
    )
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument(
        "--selective-coverages",
        default="1.0,0.9,0.8",
        help="Comma-separated fixed coverages for selective-scoring evaluation.",
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--knn-k", type=int, default=10)
    parser.add_argument("--density-ratio-clip", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--max-target-cells", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def alpha_values(args: argparse.Namespace) -> list[float]:
    raw = str(getattr(args, "alpha_grid", "") or "").strip()
    if not raw:
        return [float(args.alpha)]
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        return [float(args.alpha)]
    return sorted(set(values))


def parse_float_grid(raw: str) -> list[float]:
    values = sorted(set(float(item.strip()) for item in str(raw).split(",") if item.strip()))
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("coverage grid values must be in (0, 1]")
    return values


def main() -> None:
    args = parse_args()
    alphas = alpha_values(args)
    if any(not 0.0 < value < 1.0 for value in alphas):
        raise ValueError("--alpha/--alpha-grid values must be in (0, 1)")
    if not 0.0 < float(args.delta) < 1.0:
        raise ValueError("--delta must be in (0, 1)")
    if not 0.0 < float(args.wsr_normalized_mae_bound) < 1.0:
        raise ValueError("--wsr-normalized-mae-bound must be in (0, 1)")
    if int(args.wsr_max_threshold_candidates) < 1:
        raise ValueError("--wsr-max-threshold-candidates must be positive")
    if int(args.wsr_calibration_size) < 0:
        raise ValueError("--wsr-calibration-size must be non-negative")
    if not 0.0 < float(args.wsr_calibration_fraction) <= 1.0:
        raise ValueError("--wsr-calibration-fraction must be in (0, 1]")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    payload = load_feature_payload(args.features)
    head_specs = load_head_specs(args.heads_dir)
    if int(args.max_heads) > 0:
        head_specs = head_specs[: int(args.max_heads)]

    conformal_rows: list[dict[str, Any]] = []
    wsr_rows: list[dict[str, Any]] = []
    shift_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    hard_benign_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    for spec in head_specs:
        print(f"running RQ4 {spec['head_id']}", flush=True)
        result = run_head(spec, payload, args)
        conformal_rows.extend(result["conformal_rows"])
        wsr_rows.extend(result["wsr_rows"])
        shift_rows.extend(result["shift_rows"])
        diagnostic_rows.extend(result["diagnostic_rows"])
        bin_rows.extend(result["bin_rows"])
        hard_benign_rows.extend(result["hard_benign_rows"])
        selective_rows.extend(result["selective_rows"])

    write_csv(args.output_dir / "conformal_head_cell_results.csv", conformal_rows)
    write_csv(args.output_dir / "wsr_normalized_mae_results.csv", wsr_rows)
    write_csv(args.output_dir / "conformal_shift_summary.csv", shift_rows)
    write_csv(args.output_dir / "score_error_diagnostics.csv", diagnostic_rows)
    write_csv(args.output_dir / "score_bin_calibration.csv", bin_rows)
    write_csv(args.output_dir / "hard_id_benign_ood_summary.csv", hard_benign_rows)
    write_csv(args.output_dir / "selective_scoring.csv", selective_rows)
    summary = build_summary(
        args=args,
        payload=payload,
        head_specs=head_specs,
        conformal_rows=conformal_rows,
        wsr_rows=wsr_rows,
        shift_rows=shift_rows,
        diagnostic_rows=diagnostic_rows,
        selective_rows=selective_rows,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(args.output_dir / "summary.json", clean_json(summary))
    print(json.dumps(compact_summary(summary), ensure_ascii=False, indent=2))


def run_head(
    spec: dict[str, Any],
    payload: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    model = joblib.load(spec["model_path"])
    query_ids = tuple(getattr(model, "query_ids_", ()))
    if len(query_ids) != 1:
        raise ValueError(f"{spec['head_id']} must contain exactly one query id, got {query_ids}")
    routed_queries = np.full(len(payload["features"]), query_ids[0], dtype=object)
    output = model.predict_output(payload["features"], routed_queries)
    h = np.asarray(output.penultimate, dtype=np.float32)
    logits = np.asarray(output.logits, dtype=np.float32)
    probabilities = np.asarray(output.probabilities, dtype=np.float32)
    classes = np.asarray(output.classes)
    predictions = classes[np.argmax(probabilities, axis=1)].astype(int)
    labels = payload["labels"].astype(int)
    exact_loss = (predictions != labels).astype(int)
    pm1_loss = (np.abs(predictions - labels) > 1).astype(int)
    primary_loss = pm1_loss if str(args.primary_loss) == "pm1_error" else exact_loss
    normalized_mae = normalized_absolute_error(
        predictions,
        labels,
        y_min=float(min(CLASSES)),
        y_max=float(max(CLASSES)),
    )

    source_domain = str(spec["source_domain"])
    source_skill = str(spec["source_skill"])
    train_questions = set(str(value) for value in spec["split"]["train_question_ids"])
    source_mask = (payload["domains"] == source_domain) & (payload["skills"] == source_skill)
    train_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
    remaining_questions = sorted(set(payload["question_ids"][source_mask].tolist()).difference(train_questions))
    probe_questions, calibration_questions, id_questions = disjoint_question_partition(
        remaining_questions,
        probe_fraction=float(args.error_probe_question_fraction),
        calibration_fraction=float(args.calibration_question_fraction),
        seed=int(args.seed),
        namespace=f"{source_domain}::{source_skill}::rq4",
    )
    probe_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(probe_questions)))
    calibration_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(calibration_questions)))
    id_mask = source_mask & np.isin(payload["question_ids"], np.asarray(id_questions))
    if not train_mask.any() or not probe_mask.any() or not calibration_mask.any() or not id_mask.any():
        raise RuntimeError(f"RQ4 split failed for {spec['head_id']}")

    score_bundle = build_risk_scores(
        h=h,
        logits=logits,
        probabilities=probabilities,
        labels=labels,
        train_mask=train_mask,
        probe_mask=probe_mask,
        question_ids=payload["question_ids"],
        calibration_mask=calibration_mask,
        primary_loss=primary_loss,
        rank=min(int(args.rank), int(train_mask.sum()) - 2, h.shape[1] - 1),
        knn_k=int(args.knn_k),
        seed=int(args.seed),
        ensemble_size=int(args.ensemble_size),
    )
    target_cells = target_cell_masks(
        payload=payload,
        source_domain=source_domain,
        source_skill=source_skill,
        train_questions=train_questions,
    )
    if int(args.max_target_cells) > 0:
        target_cells = target_cells[: int(args.max_target_cells)]

    conformal_rows: list[dict[str, Any]] = []
    wsr_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    hard_benign_rows: list[dict[str, Any]] = []
    selective_rows: list[dict[str, Any]] = []
    alphas = alpha_values(args)
    thresholds: dict[tuple[str, float], dict[str, Any]] = {}
    for score_id, scores in score_bundle["scores"].items():
        wsr_rows.append(
            wsr_certification_row(
                spec=spec,
                score_id=score_id,
                score_family=score_bundle["score_families"][score_id],
                scores=scores,
                normalized_mae=normalized_mae,
                exact_loss=exact_loss,
                pm1_loss=pm1_loss,
                mask=id_mask,
                split="source_id_test",
                target_domain=source_domain,
                target_skill=source_skill,
                shift="ID test",
                args=args,
            )
        )
        for alpha in alphas:
            threshold = calibrate_unweighted(
                scores=scores[calibration_mask],
                losses=primary_loss[calibration_mask],
                alpha=float(alpha),
                delta=float(args.delta),
            )
            thresholds[(score_id, float(alpha))] = threshold
            conformal_rows.append(
                {
                    **head_prefix(spec),
                    "conformal_mode": "source_unweighted_cp_ucb",
                    "score_id": score_id,
                    "score_family": score_bundle["score_families"][score_id],
                    "split": "source_id_test",
                    "target_domain": source_domain,
                    "target_skill": source_skill,
                    "shift_type": "ID test",
                    "alpha": float(alpha),
                    "delta": float(args.delta),
                    **threshold_fields(threshold),
                    **evaluation_fields(
                        scores=scores,
                        mask=id_mask,
                        threshold=threshold.get("threshold"),
                        primary_loss=primary_loss,
                        exact_loss=exact_loss,
                        pm1_loss=pm1_loss,
                        prefix="eval",
                    ),
                    "density_ratio_status": "",
                    "density_ratio_effective_calibration_n": "",
                    "density_ratio_mean_weight": "",
                    "density_ratio_max_weight": "",
                }
            )
        bin_rows.extend(
            calibration_bin_rows(
                spec=spec,
                score_id=score_id,
                scores=scores[calibration_mask],
                primary_loss=primary_loss[calibration_mask],
                exact_loss=exact_loss[calibration_mask],
                pm1_loss=pm1_loss[calibration_mask],
            )
        )
        diagnostic_rows.extend(
            score_diagnostic_rows(
                spec=spec,
                score_id=score_id,
                scores=scores,
                primary_loss=primary_loss,
                exact_loss=exact_loss,
                pm1_loss=pm1_loss,
                masks={
                    "source_error_probe": probe_mask,
                    "source_calibration": calibration_mask,
                    "source_id_test": id_mask,
                    "all_target": combine_masks([cell["mask"] for cell in target_cells], len(scores)),
                },
            )
        )
        selective_rows.extend(
            selective_scoring_rows(
                spec=spec,
                score_id=score_id,
                scores=scores,
                labels=labels,
                predictions=predictions,
                masks={
                    "ID test": id_mask,
                    "Domain shift": combine_masks(
                        [cell["mask"] for cell in target_cells if cell["shift_type"] == "Domain shift"],
                        len(scores),
                    ),
                    "Task shift": combine_masks(
                        [cell["mask"] for cell in target_cells if cell["shift_type"] == "Task shift"],
                        len(scores),
                    ),
                    "Joint shift": combine_masks(
                        [cell["mask"] for cell in target_cells if cell["shift_type"] == "Joint shift"],
                        len(scores),
                    ),
                },
                coverages=parse_float_grid(args.selective_coverages),
            )
        )

    for cell in target_cells:
        target_mask = cell["mask"]
        weights, weight_meta = estimate_density_ratio_weights(
            calibration_features=score_bundle["density_features"][calibration_mask],
            target_features=score_bundle["density_features"][target_mask],
            clip=float(args.density_ratio_clip),
            seed=int(args.seed),
        )
        for score_id, scores in score_bundle["scores"].items():
            wsr_rows.append(
                wsr_certification_row(
                    spec=spec,
                    score_id=score_id,
                    score_family=score_bundle["score_families"][score_id],
                    scores=scores,
                    normalized_mae=normalized_mae,
                    exact_loss=exact_loss,
                    pm1_loss=pm1_loss,
                    mask=target_mask,
                    split="target_cell",
                    target_domain=cell["target_domain"],
                    target_skill=cell["target_skill"],
                    shift=cell["shift_type"],
                    args=args,
                )
            )
            for alpha in alphas:
                unweighted = thresholds[(score_id, float(alpha))]
                conformal_rows.append(
                    {
                        **head_prefix(spec),
                        "conformal_mode": "source_unweighted_cp_ucb",
                        "score_id": score_id,
                        "score_family": score_bundle["score_families"][score_id],
                        "split": "target_cell",
                        "target_domain": cell["target_domain"],
                        "target_skill": cell["target_skill"],
                        "shift_type": cell["shift_type"],
                        "alpha": float(alpha),
                        "delta": float(args.delta),
                        **threshold_fields(unweighted),
                        **evaluation_fields(
                            scores=scores,
                            mask=target_mask,
                            threshold=unweighted.get("threshold"),
                            primary_loss=primary_loss,
                            exact_loss=exact_loss,
                            pm1_loss=pm1_loss,
                            prefix="eval",
                        ),
                        "density_ratio_status": "",
                        "density_ratio_effective_calibration_n": "",
                        "density_ratio_mean_weight": "",
                        "density_ratio_max_weight": "",
                    }
                )
                weighted = calibrate_weighted(
                    scores=scores[calibration_mask],
                    losses=primary_loss[calibration_mask],
                    weights=weights,
                    alpha=float(alpha),
                    delta=float(args.delta),
                )
                conformal_rows.append(
                    {
                        **head_prefix(spec),
                        "conformal_mode": "target_weighted_effective_n_ucb",
                        "score_id": score_id,
                        "score_family": score_bundle["score_families"][score_id],
                        "split": "target_cell",
                        "target_domain": cell["target_domain"],
                        "target_skill": cell["target_skill"],
                        "shift_type": cell["shift_type"],
                        "alpha": float(alpha),
                        "delta": float(args.delta),
                        **threshold_fields(weighted),
                        **evaluation_fields(
                            scores=scores,
                            mask=target_mask,
                            threshold=weighted.get("threshold"),
                            primary_loss=primary_loss,
                            exact_loss=exact_loss,
                            pm1_loss=pm1_loss,
                            prefix="eval",
                        ),
                        "density_ratio_status": weight_meta["status"],
                        "density_ratio_effective_calibration_n": weight_meta["effective_n"],
                        "density_ratio_mean_weight": weight_meta["mean_weight"],
                        "density_ratio_max_weight": weight_meta["max_weight"],
                    }
                )

    hard_benign_rows.extend(
        hard_id_benign_rows_for_head(
            spec=spec,
            score_bundle=score_bundle,
            thresholds=thresholds,
            alphas=alphas,
            id_mask=id_mask,
            target_cells=target_cells,
            primary_loss=primary_loss,
        )
    )
    shift_rows = aggregate_shift_rows(conformal_rows)
    return {
        "conformal_rows": conformal_rows,
        "wsr_rows": wsr_rows,
        "shift_rows": shift_rows,
        "diagnostic_rows": diagnostic_rows,
        "bin_rows": bin_rows,
        "hard_benign_rows": hard_benign_rows,
        "selective_rows": selective_rows,
    }


def build_risk_scores(
    *,
    h: np.ndarray,
    logits: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    probe_mask: np.ndarray,
    question_ids: np.ndarray,
    calibration_mask: np.ndarray,
    primary_loss: np.ndarray,
    rank: int,
    knn_k: int,
    seed: int,
    ensemble_size: int,
) -> dict[str, Any]:
    eps = 1e-12
    sorted_probs = np.sort(probabilities, axis=1)
    max_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2] if probabilities.shape[1] > 1 else np.zeros_like(max_prob)
    margin = np.clip(max_prob - second_prob, 0.0, 1.0)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, eps, 1.0)), axis=1)
    entropy = entropy / max(math.log(probabilities.shape[1]), eps)
    maxprob_risk = 1.0 - max_prob
    margin_risk = 1.0 - margin
    logit_uncertainty = np.mean(np.stack([maxprob_risk, margin_risk, entropy], axis=1), axis=1)

    train_h = h[train_mask]
    train_labels = labels[train_mask]
    residual_vim = fit_residual_vim(train_h, max(1, int(rank))).score(h)
    distance_scorer = LowRankClassDistanceScorer(regularization=1e-5).fit(train_h, train_labels)
    mahalanobis = distance_scorer.mahalanobis_score(h)
    rmd = distance_scorer.rmd_score(h)
    knn = fit_knn_score(h, train_mask=train_mask, k=knn_k)

    raw_error_features = np.column_stack(
        [
            residual_vim,
            mahalanobis,
            rmd,
            knn,
            maxprob_risk,
            margin_risk,
            entropy,
        ]
    ).astype(np.float32)
    g_error_probability, g_meta = fit_error_head(
        features=raw_error_features,
        train_mask=probe_mask,
        losses=primary_loss,
        seed=seed,
    )
    ensemble_disagreement, ensemble_meta = fit_linear_bootstrap_ensemble_disagreement(
        h=h,
        labels=labels,
        train_mask=train_mask,
        question_ids=question_ids,
        ensemble_size=ensemble_size,
        seed=seed,
    )

    vim_ecdf = ecdf_transform(residual_vim[calibration_mask], residual_vim)
    mahal_ecdf = ecdf_transform(mahalanobis[calibration_mask], mahalanobis)
    rmd_ecdf = ecdf_transform(rmd[calibration_mask], rmd)
    knn_ecdf = ecdf_transform(knn[calibration_mask], knn)
    uncertainty_ecdf = ecdf_transform(logit_uncertainty[calibration_mask], logit_uncertainty)
    g_ecdf = ecdf_transform(g_error_probability[calibration_mask], g_error_probability)
    ensemble_ecdf = ecdf_transform(
        ensemble_disagreement[calibration_mask], ensemble_disagreement
    )
    novelty_ecdf = np.mean(np.stack([vim_ecdf, mahal_ecdf, rmd_ecdf, knn_ecdf], axis=1), axis=1)
    vim_g = np.mean(np.stack([novelty_ecdf, g_ecdf], axis=1), axis=1)
    fusion_equal = np.mean(np.stack([novelty_ecdf, g_ecdf, ensemble_ecdf], axis=1), axis=1)
    fusion_guarded = np.maximum(g_ecdf, 0.5 * novelty_ecdf + 0.5 * ensemble_ecdf)
    fusion_four_leg = np.mean(
        np.stack([novelty_ecdf, uncertainty_ecdf, g_ecdf, ensemble_ecdf], axis=1), axis=1
    )
    density_features = np.column_stack(
        [
            vim_ecdf,
            mahal_ecdf,
            rmd_ecdf,
            knn_ecdf,
            uncertainty_ecdf,
            g_ecdf,
            ensemble_ecdf,
            fusion_equal,
        ]
    ).astype(np.float32)

    scores = {
        "residual_vim": residual_vim,
        "mahalanobis": mahalanobis,
        "rmd": rmd,
        "knn": knn,
        "logit_uncertainty": logit_uncertainty,
        "g_error_probability": g_error_probability,
        "ensemble_disagreement": ensemble_disagreement,
        "vim_g_ecdf": vim_g,
        "fusion_equal_ecdf": fusion_equal,
        "fusion_guarded_ecdf": fusion_guarded,
        "fusion_four_leg_ecdf": fusion_four_leg,
    }
    families = {
        "residual_vim": "novelty",
        "mahalanobis": "novelty",
        "rmd": "novelty",
        "knn": "novelty",
        "logit_uncertainty": "logit_confidence",
        "g_error_probability": "supervised_error_head",
        "ensemble_disagreement": "bootstrap_ensemble_disagreement",
        "vim_g_ecdf": "novelty_plus_error_head",
        "fusion_equal_ecdf": "three_leg_fusion",
        "fusion_guarded_ecdf": "three_leg_fusion",
        "fusion_four_leg_ecdf": "four_leg_fusion_with_logit_confidence",
    }
    return {
        "scores": {key: np.asarray(value, dtype=np.float64) for key, value in scores.items()},
        "score_families": families,
        "density_features": density_features,
        "g_error_head": g_meta,
        "ensemble": ensemble_meta,
        "residual_vim": residual_vim,
    }


def fit_error_head(
    *,
    features: np.ndarray,
    train_mask: np.ndarray,
    losses: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_y = np.asarray(losses[train_mask], dtype=int)
    if len(np.unique(train_y)) < 2:
        prior = float(np.mean(train_y)) if len(train_y) else 0.0
        return (
            np.full(len(features), prior, dtype=np.float64),
            {"status": "constant_prior", "train_rows": int(train_mask.sum()), "train_error_rate": prior},
        )
    scaler = StandardScaler().fit(features[train_mask])
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit(scaler.transform(features[train_mask]), train_y)
    probabilities = model.predict_proba(scaler.transform(features))[:, 1]
    return probabilities.astype(np.float64), {
        "status": "ok",
        "train_rows": int(train_mask.sum()),
        "train_error_rate": float(np.mean(train_y)),
        "model": "balanced_logistic_regression",
    }


def fit_linear_bootstrap_ensemble_disagreement(
    *,
    h: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    question_ids: np.ndarray,
    ensemble_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Train source-specific bootstrap linear heads and return vote disagreement.

    Bootstrap sampling is performed at question-group level so the roughly fifteen
    responses derived from one FLASK question are never treated as independent
    bootstrap units. The ensemble is used only as a risk signal; the deployed
    Judge prediction still comes from the persisted source head.
    """

    train_indices = np.flatnonzero(train_mask)
    train_questions = np.asarray(question_ids)[train_indices].astype(str)
    unique_questions = np.asarray(sorted(set(train_questions.tolist())))
    member_count = max(1, int(ensemble_size))
    predictions: list[np.ndarray] = []
    statuses: list[str] = []
    for member in range(member_count):
        rng = np.random.default_rng(int(seed) + 7919 * (member + 1))
        sampled_questions = rng.choice(unique_questions, size=len(unique_questions), replace=True)
        sampled_indices = np.concatenate(
            [train_indices[train_questions == question] for question in sampled_questions]
        )
        local_y = np.asarray(labels)[sampled_indices].astype(int)
        if len(np.unique(local_y)) < 2:
            predictions.append(np.full(len(h), int(local_y[0]), dtype=int))
            statuses.append("constant_member")
            continue
        scaler = StandardScaler().fit(h[sampled_indices])
        c_value = float(10.0 ** np.linspace(-0.35, 0.35, member_count)[member])
        member_model = LogisticRegression(
            class_weight="balanced",
            C=c_value,
            max_iter=1000,
            random_state=int(seed) + member,
            solver="lbfgs",
        )
        member_model.fit(scaler.transform(h[sampled_indices]), local_y)
        predictions.append(member_model.predict(scaler.transform(h)).astype(int))
        statuses.append("ok")
    votes = np.stack(predictions, axis=1)
    majority_fraction = np.max(
        np.stack([(votes == value).mean(axis=1) for value in CLASSES], axis=1), axis=1
    )
    return (1.0 - majority_fraction).astype(np.float64), {
        "status": "ok" if all(value == "ok" for value in statuses) else "partial_constant",
        "model": "question_group_bootstrap_linear_softmax",
        "members": member_count,
        "train_rows": int(len(train_indices)),
        "train_question_groups": int(len(unique_questions)),
        "member_statuses": statuses,
    }


def calibrate_unweighted(
    *,
    scores: np.ndarray,
    losses: np.ndarray,
    alpha: float,
    delta: float,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    loss = np.asarray(losses, dtype=int)
    candidates = sorted_unique_finite(values)
    delta_per = float(delta) / max(len(candidates), 1)
    selected: dict[str, Any] | None = None
    for threshold in candidates:
        accepted = values <= threshold
        n = int(accepted.sum())
        k = int(loss[accepted].sum())
        risk = float(k / n) if n else None
        upper = clopper_pearson_upper(k, n, delta_per)
        if n > 0 and upper <= float(alpha):
            selected = {
                "threshold": float(threshold),
                "calibration_accepted_rows": n,
                "calibration_error_rows": k,
                "calibration_empirical_risk": risk,
                "calibration_risk_ucb": upper,
                "candidate_count": len(candidates),
                "delta_per_candidate": delta_per,
                "selection_status": "ok",
            }
    if selected is not None:
        return selected
    return {
        "threshold": None,
        "calibration_accepted_rows": 0,
        "calibration_error_rows": 0,
        "calibration_empirical_risk": None,
        "calibration_risk_ucb": None,
        "candidate_count": len(candidates),
        "delta_per_candidate": delta_per,
        "selection_status": "no_nonzero_coverage_threshold_satisfies_alpha",
    }


def calibrate_weighted(
    *,
    scores: np.ndarray,
    losses: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    delta: float,
) -> dict[str, Any]:
    values = np.asarray(scores, dtype=np.float64)
    loss = np.asarray(losses, dtype=float)
    weight = np.asarray(weights, dtype=np.float64)
    candidates = sorted_unique_finite(values)
    delta_per = float(delta) / max(len(candidates), 1)
    selected: dict[str, Any] | None = None
    for threshold in candidates:
        accepted = values <= threshold
        if not accepted.any():
            continue
        local_w = weight[accepted]
        w_sum = float(local_w.sum())
        if w_sum <= 0.0:
            continue
        risk = float(np.sum(local_w * loss[accepted]) / w_sum)
        n_eff = effective_sample_size(local_w)
        upper = weighted_effective_n_upper(risk, n_eff, delta_per)
        if upper <= float(alpha):
            selected = {
                "threshold": float(threshold),
                "calibration_accepted_rows": int(accepted.sum()),
                "calibration_error_rows": float(np.sum(local_w * loss[accepted])),
                "calibration_empirical_risk": risk,
                "calibration_risk_ucb": upper,
                "calibration_effective_n": n_eff,
                "candidate_count": len(candidates),
                "delta_per_candidate": delta_per,
                "selection_status": "ok",
            }
    if selected is not None:
        return selected
    return {
        "threshold": None,
        "calibration_accepted_rows": 0,
        "calibration_error_rows": 0.0,
        "calibration_empirical_risk": None,
        "calibration_risk_ucb": None,
        "calibration_effective_n": 0.0,
        "candidate_count": len(candidates),
        "delta_per_candidate": delta_per,
        "selection_status": "no_nonzero_weighted_coverage_threshold_satisfies_alpha",
    }


def clopper_pearson_upper(k: int, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - float(delta), int(k) + 1, int(n) - int(k)))


def weighted_effective_n_upper(risk: float, n_eff: float, delta: float) -> float:
    if n_eff <= 0.0:
        return 1.0
    k_eff = min(max(float(risk) * float(n_eff), 0.0), float(n_eff))
    if k_eff >= n_eff:
        return 1.0
    return float(beta.ppf(1.0 - float(delta), k_eff + 1.0, n_eff - k_eff))


def stable_u32_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}::{namespace}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


def srswor_local_indices(population_size: int, *, args: argparse.Namespace, namespace: str) -> np.ndarray:
    n = int(population_size)
    if n <= 0:
        return np.asarray([], dtype=int)
    requested = int(getattr(args, "wsr_calibration_size", 0) or 0)
    if requested > 0:
        sample_size = min(requested, n)
    else:
        sample_size = max(1, int(round(n * float(args.wsr_calibration_fraction))))
        sample_size = min(sample_size, n)
    rng = np.random.default_rng(stable_u32_seed(int(args.seed), namespace))
    return rng.permutation(n)[:sample_size].astype(int)


def wsr_certification_row(
    *,
    spec: dict[str, Any],
    score_id: str,
    score_family: str,
    scores: np.ndarray,
    normalized_mae: np.ndarray,
    exact_loss: np.ndarray,
    pm1_loss: np.ndarray,
    mask: np.ndarray,
    split: str,
    target_domain: str,
    target_skill: str,
    shift: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    population_indices = np.flatnonzero(selected)
    population_scores = np.asarray(scores, dtype=np.float64)[population_indices]
    population_losses = np.asarray(normalized_mae, dtype=np.float64)[population_indices]
    namespace = "::".join(
        [
            str(spec["head_id"]),
            str(score_id),
            str(split),
            str(target_domain),
            str(target_skill),
            "wsr_normalized_mae",
        ]
    )
    if population_indices.size == 0:
        certification: dict[str, Any] = {
            "selection_status": "empty_population",
            "threshold": None,
            "candidate_count": 0,
            "delta_per_candidate": None,
            "risk_bound": float(args.wsr_normalized_mae_bound),
            "certified_candidate_count": 0,
        }
        calibration_global_indices = np.asarray([], dtype=int)
    else:
        calibration_local_indices = srswor_local_indices(len(population_indices), args=args, namespace=namespace)
        calibration_global_indices = population_indices[calibration_local_indices]
        certification = certify_wsr_thresholds(
            scores=population_scores,
            losses=population_losses,
            calibration_indices=calibration_local_indices,
            risk_bound=float(args.wsr_normalized_mae_bound),
            delta=float(args.delta),
            max_candidates=int(args.wsr_max_threshold_candidates),
        )

    threshold = certification.get("threshold")
    if threshold is None:
        accepted_global = np.zeros_like(selected, dtype=bool)
    else:
        accepted_global = selected & (np.asarray(scores, dtype=np.float64) <= float(threshold))
    accepted_rows = int(np.sum(accepted_global, dtype=np.int64))
    population_rows = int(np.sum(selected, dtype=np.int64))
    norm_mae = mean_or_none(np.asarray(normalized_mae, dtype=np.float64)[accepted_global])
    score_range = float(max(CLASSES) - min(CLASSES))
    return {
        **head_prefix(spec),
        "certification_mode": "finite_population_wsr_normalized_mae",
        "score_id": score_id,
        "score_family": score_family,
        "split": split,
        "target_domain": target_domain,
        "target_skill": target_skill,
        "shift_type": shift,
        "delta": float(args.delta),
        "risk_bound": certification.get("risk_bound", float(args.wsr_normalized_mae_bound)),
        "risk_bound_original_mae": float(args.wsr_normalized_mae_bound) * score_range,
        "score_min": float(min(CLASSES)),
        "score_max": float(max(CLASSES)),
        "candidate_count": certification.get("candidate_count"),
        "certified_candidate_count": certification.get("certified_candidate_count"),
        "delta_per_candidate": certification.get("delta_per_candidate"),
        "selection_status": certification.get("selection_status"),
        "threshold": threshold,
        "population_rows": certification.get("population_rows", population_rows),
        "population_accepted_rows": certification.get("population_accepted_rows", accepted_rows),
        "coverage": certification.get(
            "coverage",
            float(accepted_rows / population_rows) if population_rows else None,
        ),
        "calibration_rows": certification.get("calibration_rows", int(len(calibration_global_indices))),
        "calibration_accepted_rows": certification.get("calibration_accepted_rows"),
        "calibration_mean_accept_loss": certification.get("calibration_mean_accept_loss"),
        "calibration_selective_risk": certification.get("calibration_selective_risk"),
        "target_population_loss_mean": certification.get("target_population_loss_mean"),
        "log_capital_at_target": certification.get("log_capital_at_target"),
        "wsr_population_loss_ucb": certification.get("wsr_population_loss_ucb"),
        "wsr_selective_risk_ucb": certification.get("wsr_selective_risk_ucb"),
        "wsr_selective_risk_ucb_original_mae": (
            None
            if certification.get("wsr_selective_risk_ucb") is None
            else float(certification["wsr_selective_risk_ucb"]) * score_range
        ),
        "eval_rows": population_rows,
        "eval_accepted_rows": accepted_rows,
        "eval_coverage": float(accepted_rows / population_rows) if population_rows else None,
        "eval_selective_normalized_mae": norm_mae,
        "eval_selective_original_mae": None if norm_mae is None else float(norm_mae) * score_range,
        "eval_exact_error_rate": mean_or_none(exact_loss[accepted_global]),
        "eval_pm1_error_rate": mean_or_none(pm1_loss[accepted_global]),
        "eval_all_rows_normalized_mae": mean_or_none(np.asarray(normalized_mae)[selected]),
        "eval_all_rows_exact_error_rate": mean_or_none(exact_loss[selected]),
        "eval_all_rows_pm1_error_rate": mean_or_none(pm1_loss[selected]),
        "calibration_sample_seed": stable_u32_seed(int(args.seed), namespace),
    }


def estimate_density_ratio_weights(
    *,
    calibration_features: np.ndarray,
    target_features: np.ndarray,
    clip: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    cal = np.asarray(calibration_features, dtype=np.float32)
    target = np.asarray(target_features, dtype=np.float32)
    if len(cal) < 5 or len(target) < 5:
        weights = np.ones(len(cal), dtype=np.float64)
        return weights, density_weight_meta(weights, status="fallback_too_few_rows")
    x = np.vstack([cal, target])
    y = np.concatenate([np.zeros(len(cal), dtype=int), np.ones(len(target), dtype=int)])
    scaler = StandardScaler().fit(x)
    try:
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=int(seed))
        model.fit(scaler.transform(x), y)
        p_target = model.predict_proba(scaler.transform(cal))[:, 1]
        p_target = np.clip(p_target, 1e-6, 1.0 - 1e-6)
        odds = p_target / (1.0 - p_target)
        weights = odds * (float(len(cal)) / float(len(target)))
        weights = np.clip(weights, 1e-6, float(clip))
        weights = weights / max(float(np.mean(weights)), 1e-12)
        return weights.astype(np.float64), density_weight_meta(weights, status="ok")
    except Exception as exc:  # noqa: BLE001 - weighted conformal is a diagnostic layer.
        weights = np.ones(len(cal), dtype=np.float64)
        meta = density_weight_meta(weights, status="fallback_logistic_failed")
        meta["error"] = str(exc)
        return weights, meta


def density_weight_meta(weights: np.ndarray, *, status: str) -> dict[str, Any]:
    values = np.asarray(weights, dtype=np.float64)
    return {
        "status": status,
        "effective_n": effective_sample_size(values),
        "mean_weight": float(np.mean(values)) if len(values) else 0.0,
        "max_weight": float(np.max(values)) if len(values) else 0.0,
    }


def evaluation_fields(
    *,
    scores: np.ndarray,
    mask: np.ndarray,
    threshold: float | None,
    primary_loss: np.ndarray,
    exact_loss: np.ndarray,
    pm1_loss: np.ndarray,
    prefix: str,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    total = int(selected.sum())
    if threshold is None:
        accepted = np.zeros_like(selected, dtype=bool)
    else:
        accepted = selected & (np.asarray(scores, dtype=np.float64) <= float(threshold))
    accepted_rows = int(accepted.sum())
    return {
        f"{prefix}_rows": total,
        f"{prefix}_accepted_rows": accepted_rows,
        f"{prefix}_coverage": float(accepted_rows / total) if total else None,
        f"{prefix}_rejected_rows": int(total - accepted_rows),
        f"{prefix}_primary_risk": mean_or_none(primary_loss[accepted]),
        f"{prefix}_exact_error_rate": mean_or_none(exact_loss[accepted]),
        f"{prefix}_pm1_error_rate": mean_or_none(pm1_loss[accepted]),
        f"{prefix}_all_rows_primary_risk": mean_or_none(primary_loss[selected]),
        f"{prefix}_all_rows_exact_error_rate": mean_or_none(exact_loss[selected]),
        f"{prefix}_all_rows_pm1_error_rate": mean_or_none(pm1_loss[selected]),
    }


def threshold_fields(threshold: dict[str, Any]) -> dict[str, Any]:
    return {
        "selection_status": threshold.get("selection_status"),
        "threshold": threshold.get("threshold"),
        "candidate_count": threshold.get("candidate_count"),
        "delta_per_candidate": threshold.get("delta_per_candidate"),
        "calibration_accepted_rows": threshold.get("calibration_accepted_rows"),
        "calibration_error_rows": threshold.get("calibration_error_rows"),
        "calibration_empirical_risk": threshold.get("calibration_empirical_risk"),
        "calibration_risk_ucb": threshold.get("calibration_risk_ucb"),
        "calibration_effective_n": threshold.get("calibration_effective_n", ""),
    }


def score_diagnostic_rows(
    *,
    spec: dict[str, Any],
    score_id: str,
    scores: np.ndarray,
    primary_loss: np.ndarray,
    exact_loss: np.ndarray,
    pm1_loss: np.ndarray,
    masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, mask in masks.items():
        selected = np.asarray(mask, dtype=bool)
        rows.append(
            {
                **head_prefix(spec),
                "score_id": score_id,
                "split": split,
                "rows": int(selected.sum()),
                "primary_loss_rate": mean_or_none(primary_loss[selected]),
                "exact_error_rate": mean_or_none(exact_loss[selected]),
                "pm1_error_rate": mean_or_none(pm1_loss[selected]),
                "primary_loss_auroc": binary_score_metric(primary_loss[selected], scores[selected], "auroc"),
                "primary_loss_average_precision": binary_score_metric(
                    primary_loss[selected], scores[selected], "average_precision"
                ),
                "aurc": area_under_risk_coverage(primary_loss[selected], scores[selected]),
                "top_20pct_primary_loss_rate": tail_loss_rate(
                    primary_loss[selected], scores[selected], high_risk=True
                ),
                "bottom_20pct_primary_loss_rate": tail_loss_rate(
                    primary_loss[selected], scores[selected], high_risk=False
                ),
            }
        )
    return rows


def selective_scoring_rows(
    *,
    spec: dict[str, Any],
    score_id: str,
    scores: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    masks: dict[str, np.ndarray],
    coverages: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split, mask in masks.items():
        indices = np.flatnonzero(np.asarray(mask, dtype=bool))
        if len(indices) == 0:
            continue
        ordered = indices[np.argsort(np.asarray(scores)[indices], kind="stable")]
        for requested in coverages:
            accepted_n = min(len(ordered), max(1, int(math.ceil(len(ordered) * requested))))
            accepted = ordered[:accepted_n]
            y = np.asarray(labels)[accepted].astype(int)
            pred = np.asarray(predictions)[accepted].astype(int)
            rows.append(
                {
                    **head_prefix(spec),
                    "score_id": score_id,
                    "split": split,
                    "requested_coverage": float(requested),
                    "rows": int(len(indices)),
                    "accepted_rows": int(accepted_n),
                    "empirical_coverage": float(accepted_n / len(indices)),
                    "mae": float(np.mean(np.abs(pred - y))),
                    "exact_error_rate": float(np.mean(pred != y)),
                    "pm1_error_rate": float(np.mean(np.abs(pred - y) > 1)),
                    "quadratic_weighted_kappa": safe_qwk(y, pred),
                }
            )
    return rows


def calibration_bin_rows(
    *,
    spec: dict[str, Any],
    score_id: str,
    scores: np.ndarray,
    primary_loss: np.ndarray,
    exact_loss: np.ndarray,
    pm1_loss: np.ndarray,
    bins: int = 10,
) -> list[dict[str, Any]]:
    values = np.asarray(scores, dtype=np.float64)
    if len(values) == 0:
        return []
    order = np.argsort(values)
    chunks = np.array_split(order, min(int(bins), len(order)))
    rows: list[dict[str, Any]] = []
    for index, local in enumerate(chunks, start=1):
        if len(local) == 0:
            continue
        rows.append(
            {
                **head_prefix(spec),
                "score_id": score_id,
                "bin_index": index,
                "bin_count": int(len(local)),
                "score_min": float(np.min(values[local])),
                "score_max": float(np.max(values[local])),
                "score_mean": float(np.mean(values[local])),
                "primary_loss_rate": mean_or_none(primary_loss[local]),
                "exact_error_rate": mean_or_none(exact_loss[local]),
                "pm1_error_rate": mean_or_none(pm1_loss[local]),
            }
        )
    return rows


def hard_id_benign_rows_for_head(
    *,
    spec: dict[str, Any],
    score_bundle: dict[str, Any],
    thresholds: dict[tuple[str, float], dict[str, Any]],
    alphas: list[float],
    id_mask: np.ndarray,
    target_cells: list[dict[str, Any]],
    primary_loss: np.ndarray,
) -> list[dict[str, Any]]:
    residual_vim = np.asarray(score_bundle["residual_vim"], dtype=np.float64)
    source_scores = residual_vim[id_mask]
    if len(source_scores):
        hard_cutoff = float(np.quantile(source_scores, 0.75))
    else:
        hard_cutoff = float("nan")
    hard_id_mask = id_mask & (residual_vim <= hard_cutoff) & (primary_loss.astype(bool))
    target_all = combine_masks([cell["mask"] for cell in target_cells], len(primary_loss))
    target_scores = residual_vim[target_all]
    if len(target_scores):
        benign_cutoff = float(np.quantile(target_scores, 0.90))
    else:
        benign_cutoff = float("nan")
    benign_all = target_all & (residual_vim >= benign_cutoff) & (~primary_loss.astype(bool))
    rows: list[dict[str, Any]] = []
    for score_id, scores in score_bundle["scores"].items():
        for alpha in alphas:
            threshold = thresholds[(score_id, float(alpha))].get("threshold")
            rows.append(
                {
                    **head_prefix(spec),
                    "alpha": float(alpha),
                    "score_id": score_id,
                    "subset_type": "hard_id_low_vim_wrong",
                    "shift_type": "ID test",
                    "rows": int(hard_id_mask.sum()),
                    "reject_rate": reject_rate(scores, hard_id_mask, threshold),
                    "vim_cutoff": hard_cutoff,
                }
            )
            rows.append(
                {
                    **head_prefix(spec),
                    "alpha": float(alpha),
                    "score_id": score_id,
                    "subset_type": "benign_ood_high_vim_correct",
                    "shift_type": "All target shifts",
                    "rows": int(benign_all.sum()),
                    "reject_rate": reject_rate(scores, benign_all, threshold),
                    "vim_cutoff": benign_cutoff,
                }
            )
            for shift in ("Domain shift", "Task shift", "Joint shift"):
                shift_mask = combine_masks(
                    [cell["mask"] for cell in target_cells if cell["shift_type"] == shift],
                    len(primary_loss),
                )
                benign = shift_mask & (residual_vim >= benign_cutoff) & (~primary_loss.astype(bool))
                rows.append(
                    {
                        **head_prefix(spec),
                        "alpha": float(alpha),
                        "score_id": score_id,
                        "subset_type": "benign_ood_high_vim_correct",
                        "shift_type": shift,
                        "rows": int(benign.sum()),
                        "reject_rate": reject_rate(scores, benign, threshold),
                        "vim_cutoff": benign_cutoff,
                    }
                )
    return rows


def aggregate_shift_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, float, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["conformal_mode"]),
                float(row["alpha"]),
                str(row["score_id"]),
                str(row["score_family"]),
                str(row["shift_type"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (mode, alpha, score_id, family, shift), selected in sorted(groups.items()):
        total = sum(int(row["eval_rows"]) for row in selected)
        accepted = sum(int(row["eval_accepted_rows"]) for row in selected)
        out.append(
            {
                "conformal_mode": mode,
                "alpha": float(alpha),
                "score_id": score_id,
                "score_family": family,
                "shift_type": shift,
                "result_rows": len(selected),
                "eval_rows": total,
                "eval_accepted_rows": accepted,
                "eval_coverage_micro": float(accepted / total) if total else None,
                "eval_primary_risk_macro": safe_mean([row["eval_primary_risk"] for row in selected]),
                "eval_exact_error_rate_macro": safe_mean([row["eval_exact_error_rate"] for row in selected]),
                "eval_pm1_error_rate_macro": safe_mean([row["eval_pm1_error_rate"] for row in selected]),
                "eval_all_rows_primary_risk_macro": safe_mean(
                    [row["eval_all_rows_primary_risk"] for row in selected]
                ),
                "calibration_risk_ucb_macro": safe_mean([row["calibration_risk_ucb"] for row in selected]),
                "mean_density_ratio_effective_calibration_n": safe_mean(
                    [row["density_ratio_effective_calibration_n"] for row in selected]
                ),
                "nonempty_acceptance_result_rows": int(
                    sum(row["eval_primary_risk"] not in ("", None) for row in selected)
                ),
                "empirical_risk_pass_rows": int(
                    sum(
                        row["eval_primary_risk"] not in ("", None)
                        and float(row["eval_primary_risk"]) <= float(alpha)
                        for row in selected
                    )
                ),
            }
        )
    return out


def fit_residual_vim(features: np.ndarray, rank: int) -> ViMScorer:
    errors: list[str] = []
    for candidate in range(max(1, int(rank)), 0, -1):
        try:
            return ViMScorer(rank=candidate).fit(features)
        except ValueError as exc:
            errors.append(f"rank={candidate}: {exc}")
    raise RuntimeError("Could not fit residual ViM; " + " | ".join(errors[-3:]))


def fit_knn_score(h: np.ndarray, *, train_mask: np.ndarray, k: int) -> np.ndarray:
    train_h = np.asarray(h[train_mask], dtype=np.float32)
    values = np.asarray(h, dtype=np.float32)
    train_norm = l2_normalize(train_h)
    value_norm = l2_normalize(values)
    local_k = min(max(1, int(k)), max(1, len(train_h)))
    nn = NearestNeighbors(n_neighbors=local_k, metric="euclidean")
    nn.fit(train_norm)
    distances, _ = nn.kneighbors(value_norm)
    return distances[:, -1].astype(np.float64)


class LowRankPrecision:
    """Ledoit-Wolf shrinkage precision applied through the Woodbury identity."""

    def __init__(self, regularization: float = 1e-5) -> None:
        self.regularization = float(regularization)
        self.a_: float | None = None
        self.c_: float | None = None
        self.z_: np.ndarray | None = None
        self.inv_inner_: np.ndarray | None = None

    def fit(self, samples: np.ndarray) -> "LowRankPrecision":
        values = np.asarray(samples, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
            raise ValueError("LowRankPrecision requires a non-empty [N,D] matrix")
        lw = LedoitWolf().fit(values)
        centered = values - np.asarray(lw.location_, dtype=np.float64)
        n_rows, n_dim = centered.shape
        shrinkage = float(lw.shrinkage_)
        c_value = max(0.0, 1.0 - shrinkage)
        mu = float(np.sum(centered * centered) / (float(n_rows) * float(n_dim)))
        a_value = float(shrinkage * mu + self.regularization)
        if not np.isfinite(a_value) or a_value <= 0.0:
            raise ValueError("Ledoit-Wolf shrinkage produced a non-positive diagonal term")
        z = centered / np.sqrt(float(n_rows))
        self.a_ = a_value
        self.c_ = c_value
        self.z_ = z.astype(np.float64, copy=False)
        if c_value > 1e-12:
            inner = np.eye(n_rows, dtype=np.float64) + (c_value / a_value) * (self.z_ @ self.z_.T)
            self.inv_inner_ = np.linalg.pinv(inner)
        else:
            self.inv_inner_ = None
        return self

    def md_from_centered(self, centered: np.ndarray, *, batch_size: int = 4096) -> np.ndarray:
        if self.a_ is None or self.c_ is None or self.z_ is None:
            raise RuntimeError("LowRankPrecision is not fitted")
        values = np.asarray(centered, dtype=np.float64)
        out = np.empty(values.shape[0], dtype=np.float64)
        a_value = float(self.a_)
        c_value = float(self.c_)
        for start in range(0, values.shape[0], int(batch_size)):
            batch = values[start : start + int(batch_size)]
            norm = np.sum(batch * batch, axis=1) / a_value
            if self.inv_inner_ is None or c_value <= 1e-12:
                out[start : start + len(batch)] = norm
                continue
            projected = batch @ self.z_.T
            correction = np.sum((projected @ self.inv_inner_) * projected, axis=1)
            out[start : start + len(batch)] = norm - (c_value / (a_value * a_value)) * correction
        return np.maximum(out, 0.0)


class LowRankClassDistanceScorer:
    def __init__(self, regularization: float = 1e-5) -> None:
        self.regularization = float(regularization)
        self.global_mean_: np.ndarray | None = None
        self.global_precision_: LowRankPrecision | None = None
        self.class_means_: list[np.ndarray] = []
        self.class_precision_: LowRankPrecision | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LowRankClassDistanceScorer":
        values = np.asarray(features, dtype=np.float64)
        label_values = np.asarray(labels).astype(str)
        if values.ndim != 2 or values.shape[0] < 2 or len(values) != len(label_values):
            raise ValueError("LowRankClassDistanceScorer requires aligned [N,D] features and labels")
        self.global_mean_ = values.mean(axis=0)
        self.global_precision_ = LowRankPrecision(self.regularization).fit(values - self.global_mean_)
        self.class_means_ = [
            values[label_values == label].mean(axis=0)
            for label in sorted(set(label_values.tolist()))
        ]
        residuals = np.vstack(
            [
                values[label_values == label] - values[label_values == label].mean(axis=0)
                for label in sorted(set(label_values.tolist()))
            ]
        )
        self.class_precision_ = LowRankPrecision(self.regularization).fit(residuals)
        return self

    def mahalanobis_score(self, features: np.ndarray) -> np.ndarray:
        if self.class_precision_ is None or not self.class_means_:
            raise RuntimeError("LowRankClassDistanceScorer is not fitted")
        values = np.asarray(features, dtype=np.float64)
        distances = [
            self.class_precision_.md_from_centered(values - mean)
            for mean in self.class_means_
        ]
        return np.min(np.stack(distances, axis=1), axis=1)

    def rmd_score(self, features: np.ndarray) -> np.ndarray:
        if self.global_mean_ is None or self.global_precision_ is None:
            raise RuntimeError("LowRankClassDistanceScorer is not fitted")
        values = np.asarray(features, dtype=np.float64)
        class_score = self.mahalanobis_score(values)
        global_score = self.global_precision_.md_from_centered(values - self.global_mean_)
        return class_score - global_score


def target_cell_masks(
    *,
    payload: dict[str, np.ndarray],
    source_domain: str,
    source_skill: str,
    train_questions: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_question_array = np.asarray(sorted(train_questions))
    for target_domain in ordered_targets(payload["domains"], TARGET_DOMAINS):
        for target_skill in ordered_targets(payload["skills"], TARGET_SKILLS):
            if target_domain == source_domain and target_skill == source_skill:
                continue
            mask = (
                (payload["domains"] == target_domain)
                & (payload["skills"] == target_skill)
                & ~np.isin(payload["question_ids"], train_question_array)
            )
            if not mask.any():
                continue
            rows.append(
                {
                    "target_domain": str(target_domain),
                    "target_skill": str(target_skill),
                    "shift_type": shift_type(source_domain, source_skill, str(target_domain), str(target_skill)),
                    "mask": mask,
                }
            )
    return rows


def load_feature_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float16)
        if features.ndim == 2:
            features = features[:, None, :]
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        labels = np.asarray(cache["labels"], dtype=int)
        domains = np.asarray(cache["domain_ids"]).astype(str)
        skills = np.asarray(cache["task_ids"]).astype(str)
        question_ids = np.asarray(cache["query_ids"]).astype(str)
    if features.ndim != 3:
        raise ValueError(f"Expected [N,L,D] B-space features, got {features.shape}")
    expected = len(sample_ids)
    for name, values in {
        "labels": labels,
        "domain_ids": domains,
        "task_ids": skills,
        "query_ids": question_ids,
    }.items():
        if len(values) != expected:
            raise ValueError(f"{name} does not align with sample_ids")
    return {
        "features": features,
        "sample_ids": sample_ids,
        "labels": labels,
        "domains": domains,
        "skills": skills,
        "question_ids": question_ids,
    }


def load_head_specs(heads_dir: Path) -> list[dict[str, Any]]:
    summary_path = heads_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    for head in summary.get("heads", []):
        split_path = resolve_repo_path(head["split_path"])
        model_path = resolve_repo_path(head["model_path"])
        split = json.loads(split_path.read_text(encoding="utf-8"))
        specs.append(
            {
                "head_id": str(head["head_id"]),
                "source_domain": str(head["source_domain"]),
                "source_skill": str(head["source_skill"]),
                "model_path": model_path,
                "split_path": split_path,
                "split": split,
            }
        )
    order = {
        (domain, skill): i
        for i, (domain, skill) in enumerate((d, s) for d in SOURCE_DOMAINS for s in SOURCE_SKILLS)
    }
    specs.sort(key=lambda item: order[(item["source_domain"], item["source_skill"])])
    if len(specs) != 9:
        raise ValueError(f"Expected nine heads in {heads_dir}, found {len(specs)}")
    return specs


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


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


def disjoint_question_partition(
    questions: list[str],
    *,
    probe_fraction: float,
    calibration_fraction: float,
    seed: int,
    namespace: str,
) -> tuple[set[str], set[str], list[str]]:
    if not 0.0 < probe_fraction < 1.0 or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("probe/calibration fractions must lie in (0, 1)")
    if probe_fraction + calibration_fraction >= 1.0:
        raise ValueError("probe and calibration fractions must leave a non-empty ID-test fraction")
    ordered = sorted(
        questions,
        key=lambda question: hashlib.sha256(
            f"{seed}::{namespace}::{question}".encode("utf-8")
        ).hexdigest(),
    )
    if len(ordered) < 3:
        raise ValueError("RQ4 needs at least three held-out source question groups")
    probe_n = max(1, int(round(len(ordered) * probe_fraction)))
    calibration_n = max(1, int(round(len(ordered) * calibration_fraction)))
    if probe_n + calibration_n >= len(ordered):
        overflow = probe_n + calibration_n - (len(ordered) - 1)
        calibration_n = max(1, calibration_n - overflow)
    probe = set(ordered[:probe_n])
    calibration = set(ordered[probe_n : probe_n + calibration_n])
    test = ordered[probe_n + calibration_n :]
    return probe, calibration, test


def ordered_targets(values: np.ndarray, preferred: tuple[str, ...]) -> list[str]:
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


def ecdf_transform(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    if len(reference) == 0:
        return np.zeros(len(scores), dtype=np.float64)
    values = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(reference, values, side="right").astype(np.float64) / float(len(reference))


def sorted_unique_finite(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return []
    return sorted(set(float(value) for value in finite.tolist()))


def l2_normalize(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(values * values))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(values) ** 2 / denom)


def mean_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def safe_mean(values: list[Any]) -> float | None:
    clean: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            clean.append(numeric)
    return float(np.mean(clean)) if clean else None


def binary_score_metric(labels: np.ndarray, scores: np.ndarray, metric: str) -> float | None:
    y = np.asarray(labels, dtype=int)
    if y.size == 0 or len(np.unique(y)) < 2:
        return None
    try:
        if metric == "auroc":
            return float(roc_auc_score(y, scores))
        if metric == "average_precision":
            return float(average_precision_score(y, scores))
    except ValueError:
        return None
    raise ValueError(metric)


def area_under_risk_coverage(losses: np.ndarray, scores: np.ndarray) -> float | None:
    loss = np.asarray(losses, dtype=np.float64)
    if len(loss) == 0:
        return None
    ordered_loss = loss[np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")]
    cumulative_risk = np.cumsum(ordered_loss) / np.arange(1, len(ordered_loss) + 1)
    return float(np.mean(cumulative_risk))


def tail_loss_rate(losses: np.ndarray, scores: np.ndarray, *, high_risk: bool) -> float | None:
    loss = np.asarray(losses, dtype=np.float64)
    if len(loss) == 0:
        return None
    count = max(1, int(math.ceil(0.20 * len(loss))))
    order = np.argsort(np.asarray(scores, dtype=np.float64), kind="stable")
    selected = order[-count:] if high_risk else order[:count]
    return float(np.mean(loss[selected]))


def safe_qwk(labels: np.ndarray, predictions: np.ndarray) -> float | None:
    if len(labels) < 2:
        return None
    try:
        value = float(cohen_kappa_score(labels, predictions, labels=list(CLASSES), weights="quadratic"))
    except ValueError:
        return None
    return value if np.isfinite(value) else None


def reject_rate(scores: np.ndarray, mask: np.ndarray, threshold: float | None) -> float | None:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any() or threshold is None:
        return None
    return float(np.mean(np.asarray(scores)[selected] > float(threshold)))


def combine_masks(masks: list[np.ndarray], length: int) -> np.ndarray:
    out = np.zeros(int(length), dtype=bool)
    for mask in masks:
        out |= np.asarray(mask, dtype=bool)
    return out


def head_prefix(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "head_id": spec["head_id"],
        "source_domain": spec["source_domain"],
        "source_skill": spec["source_skill"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def build_summary(
    *,
    args: argparse.Namespace,
    payload: dict[str, np.ndarray],
    head_specs: list[dict[str, Any]],
    conformal_rows: list[dict[str, Any]],
    wsr_rows: list[dict[str, Any]],
    shift_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
    selective_rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    target_rows = [row for row in conformal_rows if row.get("split") == "target_cell"]
    ranked = sorted(
        [
            {
                "conformal_mode": row["conformal_mode"],
                "alpha": row["alpha"],
                "score_id": row["score_id"],
                "shift_type": row["shift_type"],
                "eval_rows": row["eval_rows"],
                "eval_coverage_micro": row["eval_coverage_micro"],
                "eval_primary_risk_macro": row["eval_primary_risk_macro"],
                "eval_all_rows_primary_risk_macro": row["eval_all_rows_primary_risk_macro"],
            }
            for row in shift_rows
            if row["shift_type"] != "ID test" and row["eval_primary_risk_macro"] is not None
        ],
        key=lambda row: (
            float(row["eval_primary_risk_macro"]),
            -float(row["eval_coverage_micro"] or 0.0),
        ),
    )
    best_diagnostics = sorted(
        [
            row
            for row in diagnostic_rows
            if row.get("split") == "all_target" and row.get("primary_loss_auroc") not in ("", None)
        ],
        key=lambda row: float(row["primary_loss_auroc"]),
        reverse=True,
    )[:20]
    return {
        "artifact_type": "flask_rq4_conformal_failure_risk_v2",
        "source_features": str(args.features),
        "heads_dir": str(args.heads_dir),
        "feature_rows": int(len(payload["sample_ids"])),
        "feature_shape": list(payload["features"].shape),
        "head_count": len(head_specs),
        "primary_loss": str(args.primary_loss),
        "alpha": float(args.alpha),
        "alpha_grid": alpha_values(args),
        "delta": float(args.delta),
        "calibration_question_fraction": float(args.calibration_question_fraction),
        "error_probe_question_fraction": float(args.error_probe_question_fraction),
        "ensemble_size": int(args.ensemble_size),
        "selective_coverages": parse_float_grid(args.selective_coverages),
        "wsr_normalized_mae": {
            "risk_bound": float(args.wsr_normalized_mae_bound),
            "risk_bound_original_mae": float(args.wsr_normalized_mae_bound)
            * float(max(CLASSES) - min(CLASSES)),
            "max_threshold_candidates": int(args.wsr_max_threshold_candidates),
            "calibration_fraction": float(args.wsr_calibration_fraction),
            "calibration_size": int(args.wsr_calibration_size),
            "threshold_rule": "accept if detector score <= threshold",
            "selection_rule": "flat Bonferroni over fixed threshold grid; choose certified threshold with maximum coverage",
        },
        "data_split_contract": (
            "Judge-train, error-head probe, conformal calibration, and ID test are disjoint at "
            "FLASK question-group level. Target evaluations also exclude Judge-train question IDs."
        ),
        "fusion_contract": {
            "three_leg": [
                "novelty ensemble (ViM/Mahalanobis/RMD/KNN)",
                "held-out supervised error head g(h)",
                "source-specific question-bootstrap linear-head disagreement",
            ],
            "head_logit_uncertainty": "reported separately and only added in fusion_four_leg_ecdf",
        },
        "conformal_algorithm": {
            "unweighted": (
                "For each risk score s, accept rows with s <= t. Candidate thresholds are the "
                "source-calibration scores. The selected threshold is the largest t whose "
                "Clopper-Pearson upper confidence bound on accepted calibration loss is <= alpha, "
                "with delta Bonferroni-spent across tested thresholds."
            ),
            "weighted": (
                "For each target cell, a logistic density-ratio model estimates p_target(x)/p_cal(x) "
                "from score features. Calibration losses are weighted by the clipped ratio; threshold "
                "selection uses the same accept-low-risk rule with an effective-sample-size beta UCB."
            ),
            "weighted_caveat": (
                "The effective-n beta UCB is an explicit covariate-shift diagnostic, not an exact "
                "finite-sample weighted-conformal theorem. Its validity depends on the density-ratio "
                "model, clipping, independence assumptions, and reported effective sample size."
            ),
            "dependence_caveat": (
                "All splits are question-group disjoint, but CP operates on accepted response rows. "
                "Because multiple responses share a question, the nominal row-IID guarantee should "
                "be interpreted conservatively; empirical pass rates are reported separately."
            ),
        },
        "score_ids": sorted(set(row["score_id"] for row in conformal_rows)),
        "conformal_result_rows": len(conformal_rows),
        "wsr_result_rows": len(wsr_rows),
        "target_result_rows": len(target_rows),
        "selective_result_rows": len(selective_rows),
        "ranked_shift_rows_by_primary_risk": ranked[:30],
        "top_target_error_auroc_scores": best_diagnostics,
        "outputs": {
            "conformal_head_cell_results_csv": str(args.output_dir / "conformal_head_cell_results.csv"),
            "wsr_normalized_mae_results_csv": str(args.output_dir / "wsr_normalized_mae_results.csv"),
            "conformal_shift_summary_csv": str(args.output_dir / "conformal_shift_summary.csv"),
            "score_error_diagnostics_csv": str(args.output_dir / "score_error_diagnostics.csv"),
            "score_bin_calibration_csv": str(args.output_dir / "score_bin_calibration.csv"),
            "hard_id_benign_ood_summary_csv": str(args.output_dir / "hard_id_benign_ood_summary.csv"),
            "selective_scoring_csv": str(args.output_dir / "selective_scoring.csv"),
        },
        "elapsed_seconds": float(elapsed_seconds),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    ranked = summary.get("ranked_shift_rows_by_primary_risk", [])
    return {
        "artifact_type": summary["artifact_type"],
        "head_count": summary["head_count"],
        "primary_loss": summary["primary_loss"],
        "alpha_grid": summary.get("alpha_grid", [summary["alpha"]]),
        "delta": summary["delta"],
        "result_rows": summary["conformal_result_rows"],
        "wsr_result_rows": summary.get("wsr_result_rows", 0),
        "top_ranked_rows": ranked[:5],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


if __name__ == "__main__":
    main()
