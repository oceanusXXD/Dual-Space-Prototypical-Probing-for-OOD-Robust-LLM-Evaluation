#!/usr/bin/env python
"""Source-only ECDF fusion of Mahalanobis, RMD, and adapted ViM.

All 66 non-negative 0.1-grid weights are selected on source pseudo-OOD.
Official OOD metrics are computed only after dataset-aware and global weights
have been frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import scipy
import sklearn
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.shared.metrics import ood_metrics


DATASETS = ("ellipse", "asap", "clinc150", "rostd")
COMPONENTS = ("mahalanobis", "rmd", "vim")
WEIGHT_GRID = tuple(
    (m / 10.0, r / 10.0, (10 - m - r) / 10.0)
    for m in range(11)
    for r in range(11 - m)
)
EQUAL_WEIGHT = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
RRF_K = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion-root",
        type=Path,
        default=Path("artifacts/docs_experiments/vim_mahalanobis_fusion_seed42"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/docs_experiments/three_way_ecdf_fusion_seed42"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path.read_text(encoding="utf-8"))
        return
    root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    study = _load_module("study39_three_way", ROOT / "scripts/llm_judge_ood/39_run_vim_mahalanobis_study.py")
    fusion = _load_module("fusion41_three_way", ROOT / "scripts/llm_judge_ood/41_run_vim_mahalanobis_fusion.py")

    pseudo_metrics: dict[str, dict[tuple[float, float, float], dict[str, float]]] = {}
    component_metrics: dict[str, dict[str, dict[str, float]]] = {}
    dataset_selections: dict[str, dict[str, Any]] = {}

    # Selection phase. No official benchmark score is evaluated in this loop.
    for dataset in DATASETS:
        payload = _fit_components(
            dataset,
            phase="pseudo",
            args=args,
            study=study,
            fusion=fusion,
        )
        evaluation = payload["evaluation"]
        calibration = payload["calibration"]
        truth = payload["truth"]
        calibrated = {
            name: _ecdf(payload["scores"][name], payload["scores"][name][calibration])
            for name in COMPONENTS
        }
        rows = []
        by_weight: dict[tuple[float, float, float], dict[str, float]] = {}
        for weights in WEIGHT_GRID:
            score = _weighted(calibrated, weights)
            metrics = ood_metrics(truth, score[evaluation])
            by_weight[weights] = metrics
            rows.append(_weight_row(dataset, weights, metrics))
        pseudo_metrics[dataset] = by_weight
        _write_csv(root / dataset / "pseudo_weight_grid.csv", rows)

        local_components = {
            name: ood_metrics(truth, calibrated[name][evaluation]) for name in COMPONENTS
        }
        component_metrics[dataset] = local_components
        best_single = max(
            COMPONENTS,
            key=lambda name: _metric_key(local_components[name]) + (_single_simplicity(name),),
        )
        best_grid = _select_dataset_weight(by_weight)
        equal_score = _weighted(calibrated, EQUAL_WEIGHT)
        rrf_score = _rrf(
            [payload["scores"][name] for name in COMPONENTS],
            [payload["scores"][name][calibration] for name in COMPONENTS],
        )
        baseline_rows = [
            {
                "dataset": dataset,
                "method": f"Single {name}",
                **local_components[name],
            }
            for name in COMPONENTS
        ]
        baseline_rows.extend(
            [
                {
                    "dataset": dataset,
                    "method": "Equal Weight",
                    **ood_metrics(truth, equal_score[evaluation]),
                },
                {
                    "dataset": dataset,
                    "method": "RRF Fusion",
                    **ood_metrics(truth, rrf_score[evaluation]),
                },
                {
                    "dataset": dataset,
                    "method": "Best Grid Weight",
                    **by_weight[best_grid],
                },
                {
                    "dataset": dataset,
                    "method": "Best Single Detector",
                    **local_components[best_single],
                },
            ]
        )
        _write_csv(root / dataset / "pseudo_baselines.csv", baseline_rows)
        dataset_selections[dataset] = {
            "mahalanobis_method": payload["mahalanobis_method"],
            "best_grid_weight": _weight_dict(best_grid),
            "best_grid_metrics": by_weight[best_grid],
            "best_single_detector": best_single,
            "best_single_metrics": local_components[best_single],
            "pseudo_held_groups": payload["held_groups"],
            "pseudo_fit_rows": int(payload["fit_train"].sum()),
            "pseudo_ecdf_calibration_rows": int(calibration.sum()),
            "pseudo_id_eval_rows": int((evaluation & ~payload["pseudo_ood"]).sum()),
            "pseudo_ood_eval_rows": int(payload["pseudo_ood"].sum()),
        }
        del payload, calibrated

    global_rows, global_selection = _select_global_weight(pseudo_metrics, component_metrics)
    _write_csv(root / "pseudo_global_weight_grid.csv", global_rows)
    frozen = {
        "selection_scope": "four source-only pseudo-OOD tasks",
        "official_ood_used_for_selection": False,
        "weight_grid_size": len(WEIGHT_GRID),
        "weight_step": 0.1,
        "component_order": list(COMPONENTS),
        "ecdf_fit_scope": "per-dataset held-out source validation ID calibration split",
        "pseudo_eval_scope": "independent source validation ID split vs held validation pseudo-OOD groups",
        "dataset_aware": dataset_selections,
        "global": global_selection,
        "constraints": {
            "mean_fpr95_not_worse_than_best_global_single": True,
            "per_task_auroc_drop_from_task_best_single_max": 0.02,
        },
    }
    write_json(root / "frozen_selection.json", frozen)

    # Formal phase. The frozen file above exists before any official metrics.
    formal_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        payload = _fit_components(
            dataset,
            phase="formal",
            args=args,
            study=study,
            fusion=fusion,
        )
        benchmark = payload["evaluation"]
        calibration = payload["calibration"]
        truth = payload["truth"]
        calibrated = {
            name: _ecdf(payload["scores"][name], payload["scores"][name][calibration])
            for name in COMPONENTS
        }
        dataset_weight = _weight_tuple(dataset_selections[dataset]["best_grid_weight"])
        global_weight = _weight_tuple(global_selection["selected_weight"])
        best_single = str(dataset_selections[dataset]["best_single_detector"])
        local_scores = {
            "Equal Weight": _weighted(calibrated, EQUAL_WEIGHT),
            "Best Grid Weight": _weighted(calibrated, dataset_weight),
            "Global Grid Weight": _weighted(calibrated, global_weight),
            "Best Single Detector": calibrated[best_single],
            "Single Mahalanobis": calibrated["mahalanobis"],
            "Single RMD": calibrated["rmd"],
            "Single ViM": calibrated["vim"],
        }
        local_scores["RRF Fusion"] = _rrf(
            [payload["scores"][name] for name in COMPONENTS],
            [payload["scores"][name][calibration] for name in COMPONENTS],
        )
        benchmark_truth = truth.astype(int)
        for method, score in local_scores.items():
            metrics = ood_metrics(benchmark_truth, score[benchmark])
            formal_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **metrics,
                    "selected_on": "source_only_pseudo_ood",
                }
            )
        bootstrap_rows.extend(
            _bootstrap_ci(
                dataset,
                benchmark_truth,
                {name: score[benchmark] for name, score in local_scores.items()},
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
        )
        _write_csv(root / dataset / "formal_ood_results.csv", [row for row in formal_rows if row["dataset"] == dataset])
        np.savez_compressed(
            root / dataset / "formal_sample_scores.npz",
            sample_ids=payload["sample_ids"],
            truth=payload["full_truth"].astype(np.int8),
            benchmark_mask=benchmark.astype(np.int8),
            equal_weight=local_scores["Equal Weight"].astype(np.float32),
            best_grid_weight=local_scores["Best Grid Weight"].astype(np.float32),
            global_grid_weight=local_scores["Global Grid Weight"].astype(np.float32),
            best_single=local_scores["Best Single Detector"].astype(np.float32),
        )
        del payload, calibrated, local_scores

    _write_csv(root / "formal_all_datasets.csv", formal_rows)
    _write_csv(root / "bootstrap_95ci.csv", bootstrap_rows)
    summary_rows = _formal_summary(formal_rows)
    _write_csv(root / "formal_four_dataset_summary.csv", summary_rows)

    manifest = {
        "artifact_type": "three_way_source_only_ecdf_fusion_v1",
        "datasets": list(DATASETS),
        "seed": int(args.seed),
        "hiddenstate_forward_passes": 0,
        "gpu_required": False,
        "api_calls": 0,
        "weight_grid_size": len(WEIGHT_GRID),
        "rrf_k": RRF_K,
        "frozen_selection": frozen,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "files": {
            "frozen": str(root / "frozen_selection.json"),
            "pseudo_global_grid": str(root / "pseudo_global_weight_grid.csv"),
            "formal": str(root / "formal_all_datasets.csv"),
            "summary": str(root / "formal_four_dataset_summary.csv"),
            "bootstrap": str(root / "bootstrap_95ci.csv"),
        },
        "command": [sys.executable, *sys.argv],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _fit_components(
    dataset: str,
    *,
    phase: str,
    args: argparse.Namespace,
    study: ModuleType,
    fusion: ModuleType,
) -> dict[str, Any]:
    spec = study.SPECS[dataset]
    rows = read_jsonl(spec.prepared)
    with np.load(spec.hidden, allow_pickle=True) as cache:
        hidden = np.asarray(cache["features"], dtype=np.float32)
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        metadata = json.loads(str(np.asarray(cache["metadata_json"]).item()))
    expected_ids = np.asarray([str(row["sample_id"]) for row in rows])
    if not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError(f"Hiddenstate row alignment failed for {dataset}")
    labels = np.asarray([str(row["label"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    full_truth = np.asarray(
        [bool(row.get("is_document_ood", row.get("is_ood", False))) for row in rows]
    )
    masks = study._masks(spec, splits, full_truth)
    prior = json.loads(
        (Path(args.fusion_root) / dataset / "frozen_selection.json").read_text(encoding="utf-8")
    )
    layer = str(prior["layer"])
    layer_map = dict(study._available_layers(hidden, metadata))
    raw = study._layer_values(hidden, int(layer_map[layer]))
    del hidden

    groups, held = study._pseudo_groups(spec, rows, masks["train"], int(args.seed))
    if phase == "pseudo":
        fit_train = masks["train"] & ~np.isin(groups, held)
        validation_id = masks["validation"] & ~np.isin(groups, held)
        pseudo_ood = masks["validation"] & np.isin(groups, held)
        calibration, id_eval = _split_validation_id(
            validation_id,
            labels,
            sample_ids,
            int(args.seed),
        )
        evaluation = id_eval | pseudo_ood
        local_truth = pseudo_ood[evaluation].astype(int)
    elif phase == "formal":
        fit_train = masks["train"]
        calibration = masks["validation"]
        pseudo_ood = np.zeros(len(full_truth), dtype=bool)
        evaluation = masks["benchmark"]
        local_truth = full_truth[evaluation].astype(int)
    else:
        raise ValueError(f"Unknown phase {phase!r}")
    if not fit_train.any():
        raise RuntimeError(f"{dataset} {phase} detector-fit split is empty")
    if not calibration.any():
        raise RuntimeError(f"{dataset} {phase} ECDF calibration split is empty")
    if phase == "pseudo" and not pseudo_ood.any():
        raise RuntimeError(f"{dataset} validation pseudo-OOD split is empty")
    if not evaluation.any():
        raise RuntimeError(f"{dataset} {phase} evaluation split is empty")

    pca = study._fit_pca(raw[fit_train], int(args.pca_dim), int(args.seed))
    values = pca.transform(raw).astype(np.float64)
    del raw
    head = study._fit_head(
        str(prior["head"]),
        values[fit_train],
        labels[fit_train],
        values,
        ordered=bool(spec.ordered),
        seed=int(args.seed),
    )
    all_scores = fusion._detector_scores(study, head, labels, fit_train, dict(prior["vim"]))
    maha_method = str(prior["mahalanobis"])
    return {
        "scores": {
            "mahalanobis": all_scores[maha_method],
            "rmd": all_scores["RMD"],
            "vim": all_scores["Selected adapted ViM"],
        },
        "mahalanobis_method": maha_method,
        "fit_train": fit_train,
        "train": fit_train,
        "calibration": calibration,
        "evaluation": evaluation,
        "pseudo_ood": pseudo_ood,
        "truth": local_truth,
        "full_truth": full_truth,
        "sample_ids": sample_ids,
        "held_groups": held.tolist(),
    }


def _ecdf(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    if len(ordered) == 0:
        raise ValueError("ECDF reference is empty")
    ranks = np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="right")
    return (ranks + 0.5) / (len(ordered) + 1.0)


def _weighted(
    calibrated: dict[str, np.ndarray],
    weights: tuple[float, float, float],
) -> np.ndarray:
    return sum(float(weight) * calibrated[name] for name, weight in zip(COMPONENTS, weights))


def _split_validation_id(
    validation_id: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    calibration = np.zeros(len(validation_id), dtype=bool)
    evaluation = np.zeros(len(validation_id), dtype=bool)
    for label in sorted(set(labels[validation_id].tolist())):
        indices = np.flatnonzero(validation_id & (labels == label))
        ordered = sorted(
            indices.tolist(),
            key=lambda index: _stable_hash(str(sample_ids[index]), int(seed)),
        )
        if len(ordered) == 1:
            calibration[ordered[0]] = True
            continue
        split = min(max(1, len(ordered) // 2), len(ordered) - 1)
        calibration[ordered[:split]] = True
        evaluation[ordered[split:]] = True

    if calibration.any() and evaluation.any():
        return calibration, evaluation

    indices = sorted(
        np.flatnonzero(validation_id).tolist(),
        key=lambda index: _stable_hash(str(sample_ids[index]), int(seed)),
    )
    if len(indices) < 2:
        raise RuntimeError("Need at least two source validation ID rows for calibration/evaluation split")
    split = min(max(1, len(indices) // 2), len(indices) - 1)
    calibration[:] = False
    evaluation[:] = False
    calibration[indices[:split]] = True
    evaluation[indices[split:]] = True
    return calibration, evaluation


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{value}".encode("utf-8")).hexdigest()


def _rrf(score_arrays: list[np.ndarray], reference_arrays: list[np.ndarray]) -> np.ndarray:
    if len(score_arrays) != len(reference_arrays):
        raise ValueError("RRF score/reference array count mismatch")
    length = len(score_arrays[0])
    fused = np.zeros(length, dtype=np.float64)
    for scores, reference in zip(score_arrays, reference_arrays):
        local_scores = np.asarray(scores, dtype=np.float64)
        if len(local_scores) != length:
            raise ValueError("RRF score arrays must have equal length")
        ordered_reference = np.sort(np.asarray(reference, dtype=np.float64))
        if len(ordered_reference) == 0:
            raise ValueError("RRF reference is empty")
        ranks = len(ordered_reference) - np.searchsorted(
            ordered_reference,
            local_scores,
            side="left",
        ) + 1
        fused += 1.0 / (RRF_K + ranks)
    return fused


def _select_dataset_weight(
    metrics: dict[tuple[float, float, float], dict[str, float]],
) -> tuple[float, float, float]:
    candidates = list(metrics)
    best_auroc = max(float(metrics[weights]["auroc"]) for weights in candidates)
    candidates = [
        weights for weights in candidates
        if float(metrics[weights]["auroc"]) >= best_auroc - 1e-12
    ]
    best_aupr = max(float(metrics[weights]["aupr"]) for weights in candidates)
    candidates = [
        weights for weights in candidates
        if float(metrics[weights]["aupr"]) >= best_aupr - 1e-12
    ]
    best_fpr = min(float(metrics[weights]["fpr95"]) for weights in candidates)
    candidates = [
        weights for weights in candidates
        if float(metrics[weights]["fpr95"]) <= best_fpr + 1e-12
    ]
    return max(candidates, key=_balance)


def _select_global_weight(
    pseudo_metrics: dict[str, dict[tuple[float, float, float], dict[str, float]]],
    component_metrics: dict[str, dict[str, dict[str, float]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    single_weights = {
        "mahalanobis": (1.0, 0.0, 0.0),
        "rmd": (0.0, 1.0, 0.0),
        "vim": (0.0, 0.0, 1.0),
    }
    single_aggregate = {}
    for name, weight in single_weights.items():
        values = [pseudo_metrics[dataset][weight] for dataset in DATASETS]
        single_aggregate[name] = _aggregate(values)
    best_global_single = max(
        COMPONENTS,
        key=lambda name: _metric_key(single_aggregate[name]) + (_single_simplicity(name),),
    )
    best_single_fpr = float(single_aggregate[best_global_single]["fpr95"])
    task_best_auroc = {
        dataset: max(float(component_metrics[dataset][name]["auroc"]) for name in COMPONENTS)
        for dataset in DATASETS
    }

    rows = []
    eligible_weights = []
    for weights in WEIGHT_GRID:
        local = [pseudo_metrics[dataset][weights] for dataset in DATASETS]
        aggregate = _aggregate(local)
        drops = [
            task_best_auroc[dataset] - float(pseudo_metrics[dataset][weights]["auroc"])
            for dataset in DATASETS
        ]
        fpr_ok = float(aggregate["fpr95"]) <= best_single_fpr + 1e-12
        drop_ok = max(drops) <= 0.02 + 1e-12
        eligible = bool(fpr_ok and drop_ok)
        if eligible:
            eligible_weights.append(weights)
        rows.append(
            {
                **_weight_dict(weights),
                "mean_auroc": aggregate["auroc"],
                "mean_aupr": aggregate["aupr"],
                "mean_fpr95": aggregate["fpr95"],
                "worst_task_auroc": min(
                    float(pseudo_metrics[dataset][weights]["auroc"]) for dataset in DATASETS
                ),
                "max_task_drop_from_best_single": max(drops),
                "fpr95_constraint_pass": fpr_ok,
                "task_drop_constraint_pass": drop_ok,
                "eligible": eligible,
            }
        )
    row_by_weight = {
        _weight_tuple(row): row for row in rows
    }
    selection_pool = eligible_weights if eligible_weights else list(WEIGHT_GRID)
    selected = max(
        selection_pool,
        key=lambda weight: (
            float(row_by_weight[weight]["mean_auroc"]),
            float(row_by_weight[weight]["worst_task_auroc"]),
            float(row_by_weight[weight]["mean_aupr"]),
            -float(row_by_weight[weight]["mean_fpr95"]),
            _balance(weight),
        ),
    )
    return rows, {
        "selected_weight": _weight_dict(selected),
        "selected_pseudo_metrics": row_by_weight[selected],
        "best_global_single_detector": best_global_single,
        "best_global_single_mean_metrics": single_aggregate[best_global_single],
        "best_global_single_mean_fpr95": best_single_fpr,
        "eligible_weight_count": len(eligible_weights),
        "constraints_satisfied": bool(eligible_weights),
        "selection_rule": (
            "constrained_source_only_grid"
            if eligible_weights
            else "unconstrained_source_only_grid_no_weight_satisfied_constraints"
        ),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        "auroc": float(np.mean([float(row["auroc"]) for row in rows])),
        "aupr": float(np.mean([float(row["aupr"]) for row in rows])),
        "fpr95": float(np.mean([float(row["fpr95"]) for row in rows])),
    }


def _metric_key(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(metrics["auroc"]),
        float(metrics["aupr"]),
        -float(metrics["fpr95"]),
    )


def _single_simplicity(name: str) -> int:
    return {"mahalanobis": 3, "rmd": 2, "vim": 1}[name]


def _balance(weights: tuple[float, float, float]) -> float:
    return -float(sum((weight - 1.0 / 3.0) ** 2 for weight in weights))


def _weight_dict(weights: tuple[float, float, float]) -> dict[str, float]:
    return {"w_mahalanobis": weights[0], "w_rmd": weights[1], "w_vim": weights[2]}


def _weight_tuple(values: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(values["w_mahalanobis"]),
        float(values["w_rmd"]),
        float(values["w_vim"]),
    )


def _weight_row(
    dataset: str,
    weights: tuple[float, float, float],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {"dataset": dataset, **_weight_dict(weights), **metrics}


def _formal_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    output = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        ranks = []
        for dataset in DATASETS:
            dataset_rows = [row for row in rows if row["dataset"] == dataset]
            value = next(float(row["auroc"]) for row in dataset_rows if row["method"] == method)
            greater = sum(float(row["auroc"]) > value + 1e-12 for row in dataset_rows)
            tied = sum(abs(float(row["auroc"]) - value) <= 1e-12 for row in dataset_rows)
            ranks.append(1.0 + greater + 0.5 * (tied - 1))
        output.append(
            {
                "method": method,
                "mean_auroc": float(np.mean([float(row["auroc"]) for row in local])),
                "mean_aupr": float(np.mean([float(row["aupr"]) for row in local])),
                "mean_fpr95": float(np.mean([float(row["fpr95"]) for row in local])),
                "mean_rank": float(np.mean(ranks)),
            }
        )
    return output


def _bootstrap_ci(
    dataset: str,
    truth: np.ndarray,
    scores: dict[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    id_indices = np.flatnonzero(truth == 0)
    ood_indices = np.flatnonzero(truth == 1)
    values = {name: [] for name in scores}
    for _ in range(int(bootstrap)):
        selected = np.concatenate(
            [
                rng.choice(id_indices, len(id_indices), replace=True),
                rng.choice(ood_indices, len(ood_indices), replace=True),
            ]
        )
        local_truth = truth[selected]
        for name, score in scores.items():
            values[name].append(float(roc_auc_score(local_truth, score[selected])))
    return [
        {
            "dataset": dataset,
            "method": name,
            "auroc": float(roc_auc_score(truth, scores[name])),
            "ci_low": float(np.quantile(values[name], 0.025)),
            "ci_high": float(np.quantile(values[name], 0.975)),
            "bootstrap_samples": int(bootstrap),
        }
        for name in scores
    ]


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
