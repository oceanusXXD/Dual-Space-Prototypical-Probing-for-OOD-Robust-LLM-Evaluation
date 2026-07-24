#!/usr/bin/env python3
"""Run the documented FLASK 3x3 -> 5x6 OOD detector suite."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.algorithm.detector.knn import KNNScorer
from src.algorithm.detector.openood import OpenOODPosthocScorer
from src.algorithm.detector.residual_vim import FullViMScorer, ViMScorer
from src.common.metrics import ood_metrics


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

OPENOOD_DETECTORS = (
    "msp",
    "maxlogit",
    "energy",
    "gen",
    "gradnorm",
    "kl_matching",
    "odin",
    "react",
    "dice",
    "ash",
)
ALL_DETECTORS = (
    "msp",
    "maxlogit",
    "energy",
    "gen",
    "gradnorm",
    "kl_matching",
    "mahalanobis",
    "rmd",
    "knn",
    "residual_vim",
    "full_vim",
    "odin",
    "react",
    "dice",
    "ash",
)
BASIC_DETECTORS = (
    "msp",
    "maxlogit",
    "energy",
    "gen",
    "gradnorm",
    "kl_matching",
    "knn",
    "residual_vim",
)
DISPLAY_NAME = {
    "msp": "MSP",
    "maxlogit": "MaxLogit",
    "energy": "Energy",
    "gen": "GEN",
    "gradnorm": "GradNorm",
    "kl_matching": "KL Matching",
    "mahalanobis": "Mahalanobis",
    "rmd": "RMD",
    "knn": "kNN",
    "residual_vim": "Residual-only ViM",
    "full_vim": "Full ViM",
    "odin": "ODIN",
    "react": "ReAct",
    "dice": "DICE",
    "ash": "ASH-B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--heads-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--calibration-question-fraction", type=float, default=0.10)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--detectors",
        default="all",
        help="all, basic, or comma-separated detector ids.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_detectors(value: str) -> tuple[str, ...]:
    raw = value.strip().lower()
    if raw == "all":
        return ALL_DETECTORS
    if raw == "basic":
        return BASIC_DETECTORS
    detectors = tuple(
        item.strip().lower().replace("-", "_")
        for item in raw.split(",")
        if item.strip()
    )
    unknown = sorted(set(detectors).difference(ALL_DETECTORS))
    if unknown:
        raise ValueError(f"Unknown detectors: {unknown}")
    if not detectors:
        raise ValueError("No detectors selected")
    return detectors


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output dir is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload = load_feature_payload(args.features)
    head_specs = load_head_specs(args.heads_dir)
    detector_ids = selected_detectors(str(args.detectors))

    all_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for spec in head_specs:
        print(f"running {spec['head_id']}", flush=True)
        head_result = run_head(spec, payload, args, detector_ids)
        all_rows.extend(head_result["cell_rows"])
        reference_rows.extend(head_result["reference_rows"])

    write_csv(args.output_dir / "detector_head_cell_results.csv", all_rows)
    write_csv(args.output_dir / "detector_source_id_reference.csv", reference_rows)
    summary = build_summary(
        args=args,
        payload=payload,
        cell_rows=all_rows,
        reference_rows=reference_rows,
        detector_ids=detector_ids,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(compact_print_summary(summary), ensure_ascii=False, indent=2))


def run_head(
    spec: dict[str, Any],
    payload: dict[str, np.ndarray],
    args: argparse.Namespace,
    detector_ids: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    model = joblib.load(spec["model_path"])
    query_ids = tuple(getattr(model, "query_ids_", ()))
    if len(query_ids) != 1:
        raise ValueError(f"{spec['head_id']} must contain exactly one query id, got {query_ids}")
    routed_queries = np.full(len(payload["features"]), query_ids[0], dtype=object)
    output = model.predict_output(payload["features"], routed_queries)
    h = np.asarray(output.penultimate, dtype=np.float32)
    logits = np.asarray(output.logits, dtype=np.float32)
    labels = payload["labels"]
    train_questions = set(str(value) for value in spec["split"]["train_question_ids"])
    source_domain = str(spec["source_domain"])
    source_skill = str(spec["source_skill"])
    source_mask = (payload["domains"] == source_domain) & (payload["skills"] == source_skill)
    train_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
    remaining_questions = sorted(set(payload["question_ids"][source_mask].tolist()).difference(train_questions))
    calibration_questions = stable_question_sample(
        remaining_questions,
        fraction=float(args.calibration_question_fraction),
        seed=int(args.seed),
        namespace=f"{source_domain}::{source_skill}::detector_calibration",
    )
    id_questions = sorted(set(remaining_questions).difference(calibration_questions))
    calibration_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(calibration_questions)))
    id_mask = source_mask & np.isin(payload["question_ids"], np.asarray(id_questions))
    if not train_mask.any() or not calibration_mask.any() or not id_mask.any():
        raise RuntimeError(f"Detector split failed for {spec['head_id']}")

    weights, biases, head_query_ids = model.affine_head_parameters()
    scorers = fit_detectors(
        h=h,
        logits=logits,
        labels=labels,
        train_mask=train_mask,
        rank=min(int(args.rank), int(train_mask.sum()) - 2, h.shape[1] - 1),
        routed_queries=routed_queries,
        weights=weights,
        biases=biases,
        head_query_ids=head_query_ids,
        detector_ids=detector_ids,
    )
    cell_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    for detector in detector_ids:
        fitted = scorers[detector]
        if fitted["status"] != "ok":
            reference_rows.append(
                reference_failure_row(spec, detector, train_mask, calibration_mask, id_mask, fitted["error"])
            )
            cell_rows.extend(
                failure_cell_rows(
                    spec,
                    detector,
                    payload,
                    id_mask,
                    train_questions,
                    fitted["error"],
                )
            )
            continue
        score_fn = fitted["score_fn"]
        calibration_scores = score_fn(calibration_mask)
        id_scores = score_fn(id_mask)
        soft_threshold = float(np.quantile(calibration_scores, 0.90))
        hard_threshold = float(np.quantile(calibration_scores, 0.95))
        id_fpr_hard = float(np.mean(id_scores >= hard_threshold))
        reference_rows.append(
            {
                "detector": detector,
                "detector_name": DISPLAY_NAME[detector],
                "source_domain": source_domain,
                "source_skill": source_skill,
                "status": "ok",
                "train_rows": int(train_mask.sum()),
                "calibration_rows": int(calibration_mask.sum()),
                "id_rows": int(id_mask.sum()),
                "rank": fitted.get("rank", ""),
                "soft_threshold_q90": soft_threshold,
                "hard_threshold_q95": hard_threshold,
                "id_fpr_at_hard_threshold": id_fpr_hard,
                "id_score_mean": float(np.mean(id_scores)),
                "id_score_std": float(np.std(id_scores)),
                "error": "",
            }
        )
        for target_domain in ordered_targets(payload["domains"], TARGET_DOMAINS):
            for target_skill in ordered_targets(payload["skills"], TARGET_SKILLS):
                if target_domain == source_domain and target_skill == source_skill:
                    continue
                target_mask = (
                    (payload["domains"] == target_domain)
                    & (payload["skills"] == target_skill)
                    & ~np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
                )
                if not target_mask.any():
                    continue
                ood_scores = score_fn(target_mask)
                truth = np.concatenate(
                    [
                        np.zeros(len(id_scores), dtype=np.int64),
                        np.ones(len(ood_scores), dtype=np.int64),
                    ]
                )
                scores = np.concatenate([id_scores, ood_scores])
                metrics = ood_metrics(truth, scores)
                cell_rows.append(
                    {
                        "detector": detector,
                        "detector_name": DISPLAY_NAME[detector],
                        "source_domain": source_domain,
                        "source_skill": source_skill,
                        "target_domain": str(target_domain),
                        "target_skill": str(target_skill),
                        "shift_type": shift_type(source_domain, source_skill, str(target_domain), str(target_skill)),
                        "status": "ok",
                        "id_rows": int(id_mask.sum()),
                        "ood_rows": int(target_mask.sum()),
                        "rank": fitted.get("rank", ""),
                        "auroc": metrics["auroc"],
                        "aupr_ood": metrics["aupr"],
                        "fpr95": metrics["fpr95"],
                        "id_fpr_at_hard_threshold": id_fpr_hard,
                        "ood_tpr_at_hard_threshold": float(np.mean(ood_scores >= hard_threshold)),
                        "id_score_mean": float(np.mean(id_scores)),
                        "ood_score_mean": float(np.mean(ood_scores)),
                        "error": "",
                    }
                )
    return {"cell_rows": cell_rows, "reference_rows": reference_rows}


def fit_detectors(
    *,
    h: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    train_mask: np.ndarray,
    rank: int,
    routed_queries: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    head_query_ids: np.ndarray,
    detector_ids: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    train_h = h[train_mask]
    train_logits = logits[train_mask]
    train_labels = labels[train_mask]
    train_queries = routed_queries[train_mask]
    for detector in [value for value in OPENOOD_DETECTORS if value in detector_ids]:
        try:
            scorer = OpenOODPosthocScorer(method=detector).fit(
                train_h,
                train_logits,
                labels=train_labels,
                query_ids=train_queries,
                head_weight=weights,
                head_bias=biases,
                head_query_ids=head_query_ids,
            )
            scores = scorer.score(h, logits, routed_queries)
            result[detector] = {
                "status": "ok",
                "score_fn": lambda mask, scores=scores: scores[mask],
            }
        except Exception as exc:  # noqa: BLE001 - record detector applicability.
            result[detector] = {"status": "not_applicable", "error": str(exc)}
    if "mahalanobis" in detector_ids or "rmd" in detector_ids:
        try:
            scorer = LowRankClassDistanceScorer(regularization=1e-5).fit(train_h, train_labels)
            if "mahalanobis" in detector_ids:
                mahalanobis_scores = scorer.mahalanobis_score(h)
                result["mahalanobis"] = {
                    "status": "ok",
                    "score_fn": lambda mask, scores=mahalanobis_scores: scores[mask],
                }
            if "rmd" in detector_ids:
                rmd_scores = scorer.rmd_score(h)
                result["rmd"] = {"status": "ok", "score_fn": lambda mask, scores=rmd_scores: scores[mask]}
        except Exception as exc:  # noqa: BLE001
            if "mahalanobis" in detector_ids:
                result["mahalanobis"] = {"status": "not_applicable", "error": str(exc)}
            if "rmd" in detector_ids:
                result["rmd"] = {"status": "not_applicable", "error": str(exc)}
    if "knn" in detector_ids:
        try:
            k = min(10, int(train_h.shape[0]) - 1)
            scorer = KNNScorer(k=k, metric="euclidean", normalize=True).fit(train_h)
            scores = scorer.score(h)
            result["knn"] = {
                "status": "ok",
                "rank": "",
                "score_fn": lambda mask, scores=scores: scores[mask],
            }
        except Exception as exc:  # noqa: BLE001
            result["knn"] = {"status": "not_applicable", "error": str(exc)}
    if "residual_vim" in detector_ids:
        try:
            scorer = fit_residual_vim(train_h, int(rank))
            scores = scorer.score(h)
            result["residual_vim"] = {
                "status": "ok",
                "rank": int(scorer.rank),
                "score_fn": lambda mask, scores=scores: scores[mask],
            }
        except Exception as exc:  # noqa: BLE001
            result["residual_vim"] = {"status": "not_applicable", "error": str(exc)}
    if "full_vim" in detector_ids:
        try:
            scorer = fit_full_vim(
                train_h,
                train_logits,
                rank=int(rank),
                train_queries=train_queries,
                weights=weights,
                biases=biases,
                head_query_ids=head_query_ids,
            )
            scores = scorer.score(h, logits, routed_queries)
            result["full_vim"] = {
                "status": "ok",
                "rank": int(scorer.rank),
                "score_fn": lambda mask, scores=scores: scores[mask],
            }
        except Exception as exc:  # noqa: BLE001
            result["full_vim"] = {"status": "not_applicable", "error": str(exc)}
    return result


def fit_residual_vim(features: np.ndarray, rank: int) -> ViMScorer:
    errors: list[str] = []
    for candidate in range(max(1, int(rank)), 0, -1):
        try:
            return ViMScorer(rank=candidate).fit(features)
        except ValueError as exc:
            errors.append(f"rank={candidate}: {exc}")
    raise RuntimeError("Could not fit residual ViM; " + " | ".join(errors[-3:]))


def fit_full_vim(
    features: np.ndarray,
    logits: np.ndarray,
    *,
    rank: int,
    train_queries: np.ndarray,
    weights: np.ndarray,
    biases: np.ndarray,
    head_query_ids: np.ndarray,
) -> FullViMScorer:
    errors: list[str] = []
    for candidate in range(max(1, int(rank)), 0, -1):
        try:
            return FullViMScorer(rank=candidate).fit(
                features,
                logits,
                head_weight=weights,
                head_bias=biases,
                query_ids=train_queries,
                head_query_ids=head_query_ids,
            )
        except ValueError as exc:
            errors.append(f"rank={candidate}: {exc}")
    raise RuntimeError("Could not fit Full ViM; " + " | ".join(errors[-3:]))


class LowRankPrecision:
    """Ledoit-Wolf shrinkage precision applied through Woodbury identity."""

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
        if values.ndim != 2 or values.shape[1] != self.z_.shape[1]:
            raise ValueError("Mahalanobis score rows do not match fitted precision")
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
    specs = []
    for head in summary.get("heads", []):
        split_path = Path(head["split_path"])
        split = json.loads(split_path.read_text(encoding="utf-8"))
        specs.append(
            {
                "head_id": head["head_id"],
                "source_domain": head["source_domain"],
                "source_skill": head["source_skill"],
                "model_path": Path(head["model_path"]),
                "split_path": split_path,
                "split": split,
            }
        )
    order = {(domain, skill): i for i, (domain, skill) in enumerate((d, s) for d in SOURCE_DOMAINS for s in SOURCE_SKILLS)}
    specs.sort(key=lambda item: order[(item["source_domain"], item["source_skill"])])
    if len(specs) != 9:
        raise ValueError(f"Expected nine heads in {heads_dir}, found {len(specs)}")
    return specs


def stable_question_sample(questions: list[str], *, fraction: float, seed: int, namespace: str) -> set[str]:
    if not questions:
        return set()
    n = max(1, int(round(len(questions) * fraction)))
    n = min(n, len(questions))
    import hashlib

    ordered = sorted(
        questions,
        key=lambda question: hashlib.sha256(f"{seed}::{namespace}::{question}".encode("utf-8")).hexdigest(),
    )
    return set(ordered[:n])


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


def reference_failure_row(
    spec: dict[str, Any],
    detector: str,
    train_mask: np.ndarray,
    calibration_mask: np.ndarray,
    id_mask: np.ndarray,
    error: str,
) -> dict[str, Any]:
    return {
        "detector": detector,
        "detector_name": DISPLAY_NAME[detector],
        "source_domain": spec["source_domain"],
        "source_skill": spec["source_skill"],
        "status": "not_applicable",
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "id_rows": int(id_mask.sum()),
        "rank": "",
        "soft_threshold_q90": "",
        "hard_threshold_q95": "",
        "id_fpr_at_hard_threshold": "",
        "id_score_mean": "",
        "id_score_std": "",
        "error": error,
    }


def failure_cell_rows(
    spec: dict[str, Any],
    detector: str,
    payload: dict[str, np.ndarray],
    id_mask: np.ndarray,
    train_questions: set[str],
    error: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_domain = str(spec["source_domain"])
    source_skill = str(spec["source_skill"])
    for target_domain in ordered_targets(payload["domains"], TARGET_DOMAINS):
        for target_skill in ordered_targets(payload["skills"], TARGET_SKILLS):
            if target_domain == source_domain and target_skill == source_skill:
                continue
            target_mask = (
                (payload["domains"] == target_domain)
                & (payload["skills"] == target_skill)
                & ~np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
            )
            if not target_mask.any():
                continue
            rows.append(
                {
                    "detector": detector,
                    "detector_name": DISPLAY_NAME[detector],
                    "source_domain": source_domain,
                    "source_skill": source_skill,
                    "target_domain": str(target_domain),
                    "target_skill": str(target_skill),
                    "shift_type": shift_type(source_domain, source_skill, str(target_domain), str(target_skill)),
                    "status": "not_applicable",
                    "id_rows": int(id_mask.sum()),
                    "ood_rows": int(target_mask.sum()),
                    "rank": "",
                    "auroc": "",
                    "aupr_ood": "",
                    "fpr95": "",
                    "id_fpr_at_hard_threshold": "",
                    "ood_tpr_at_hard_threshold": "",
                    "id_score_mean": "",
                    "ood_score_mean": "",
                    "error": error,
                }
            )
    return rows


def build_summary(
    *,
    args: argparse.Namespace,
    payload: dict[str, np.ndarray],
    cell_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    detector_ids: tuple[str, ...],
    elapsed_seconds: float,
) -> dict[str, Any]:
    ok_rows = [row for row in cell_rows if row["status"] == "ok"]
    detector_metrics = {}
    detector_shift_metrics = {}
    for detector in detector_ids:
        rows = [row for row in ok_rows if row["detector"] == detector]
        detector_metrics[detector] = mean_cell_metrics(rows)
        shifts = defaultdict(list)
        for row in rows:
            shifts[str(row["shift_type"])].append(row)
        detector_shift_metrics[detector] = {
            shift: mean_cell_metrics(shift_rows) for shift, shift_rows in sorted(shifts.items())
        }
    reference_by_detector = {}
    for detector in detector_ids:
        rows = [row for row in reference_rows if row["detector"] == detector]
        ok_reference = [row for row in rows if row["status"] == "ok"]
        reference_by_detector[detector] = {
            "head_count": len(rows),
            "ok_head_count": len(ok_reference),
            "mean_id_fpr_at_hard_threshold": safe_mean(
                [row["id_fpr_at_hard_threshold"] for row in ok_reference]
            ),
        }
    ranked = sorted(
        [
            {
                "detector": detector,
                "detector_name": DISPLAY_NAME[detector],
                **metrics,
            }
            for detector, metrics in detector_metrics.items()
            if metrics["result_count"]
        ],
        key=lambda row: (float(row["auroc"]), float(row["aupr_ood"]), -float(row["fpr95"])),
        reverse=True,
    )
    return {
        "artifact_type": "flask_3x3_to_5x6_detector_suite_v1",
        "source_features": str(args.features),
        "heads_dir": str(args.heads_dir),
        "feature_rows": int(len(payload["sample_ids"])),
        "feature_shape": list(payload["features"].shape),
        "head_count": 9,
        "detectors": list(detector_ids),
        "detector_count": len(detector_ids),
        "cell_result_rows": len(cell_rows),
        "ok_cell_result_rows": len(ok_rows),
        "detector_macro_metrics": detector_metrics,
        "detector_shift_macro_metrics": detector_shift_metrics,
        "source_id_reference_by_detector": reference_by_detector,
        "ranked_by_macro_auroc": ranked,
        "result_csv": str(args.output_dir / "detector_head_cell_results.csv"),
        "source_id_reference_csv": str(args.output_dir / "detector_source_id_reference.csv"),
        "elapsed_seconds": float(elapsed_seconds),
    }


def mean_cell_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("auroc", "aupr_ood", "fpr95", "id_fpr_at_hard_threshold", "ood_tpr_at_hard_threshold")
    return {
        "result_count": len(rows),
        **{key: safe_mean([row[key] for row in rows]) for key in keys},
    }


def safe_mean(values: list[Any]) -> float:
    parsed: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            parsed.append(number)
    return float(np.mean(parsed)) if parsed else float("nan")


def compact_print_summary(summary: dict[str, Any]) -> dict[str, Any]:
    top = summary["ranked_by_macro_auroc"][:5]
    return {
        "feature_rows": summary["feature_rows"],
        "ok_cell_result_rows": summary["ok_cell_result_rows"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "top5_by_macro_auroc": [
            {
                "detector": row["detector_name"],
                "auroc": row["auroc"],
                "aupr_ood": row["aupr_ood"],
                "fpr95": row["fpr95"],
                "ood_tpr_at_hard_threshold": row["ood_tpr_at_hard_threshold"],
            }
            for row in top
        ],
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
