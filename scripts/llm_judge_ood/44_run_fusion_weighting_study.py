#!/usr/bin/env python
"""Compare source-only weighting algorithms for three calibrated OOD scores.

The component detectors and ECDF protocol are shared with experiment 43. Every
weight is estimated using source-only pseudo-OOD, written to a frozen selection
file, and only then evaluated on the official OOD benchmark.
"""

from __future__ import annotations

import argparse
import csv
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
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.shared.metrics import ood_metrics


DATASETS = ("ellipse", "asap", "clinc150", "rostd")
COMPONENTS = ("mahalanobis", "rmd", "vim")
FINE_STEP = 0.02
FINE_UNITS = int(round(1.0 / FINE_STEP))
FINE_GRID = tuple(
    (m / FINE_UNITS, r / FINE_UNITS, (FINE_UNITS - m - r) / FINE_UNITS)
    for m in range(FINE_UNITS + 1)
    for r in range(FINE_UNITS + 1 - m)
)
SOFTMAX_TEMPERATURE = 0.05
LOGISTIC_L2 = 0.01
ONE_SE_BOOTSTRAPS = 300


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
        default=Path("artifacts/docs_experiments/fusion_weighting_study_seed42"),
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
    base = _load_module(
        "fusion43_weight_study",
        ROOT / "scripts/llm_judge_ood/43_run_three_way_ecdf_fusion.py",
    )
    study = _load_module(
        "study39_weight_study",
        ROOT / "scripts/llm_judge_ood/39_run_vim_mahalanobis_study.py",
    )
    fusion = _load_module(
        "fusion41_weight_study",
        ROOT / "scripts/llm_judge_ood/41_run_vim_mahalanobis_fusion.py",
    )

    selections: dict[str, dict[str, Any]] = {}
    fine_metrics_by_dataset: dict[
        str, dict[tuple[float, float, float], dict[str, float]]
    ] = {}

    # Selection phase: only source validation ID and source-held pseudo-OOD.
    for dataset in DATASETS:
        payload = base._fit_components(
            dataset, phase="pseudo", args=args, study=study, fusion=fusion
        )
        truth, matrix = _calibrated_evaluation(base, payload)
        fine_metrics = _grid_metrics(base, truth, matrix, FINE_GRID)
        fine_metrics_by_dataset[dataset] = fine_metrics
        component_metrics = {
            name: ood_metrics(truth, matrix[:, index])
            for index, name in enumerate(COMPONENTS)
        }
        weights = _select_weights(
            base,
            truth,
            matrix,
            fine_metrics,
            component_metrics,
            seed=int(args.seed),
        )
        method_rows = []
        method_payload: dict[str, Any] = {}
        for method, weight in weights.items():
            metrics = ood_metrics(truth, matrix @ np.asarray(weight))
            method_rows.append(
                {"dataset": dataset, "method": method, **_weight_dict(weight), **metrics}
            )
            method_payload[method] = {
                "weight": _weight_dict(weight),
                "pseudo_metrics": metrics,
            }
        best_single = max(
            COMPONENTS,
            key=lambda name: base._metric_key(component_metrics[name])
            + (base._single_simplicity(name),),
        )
        method_payload["Best Single Detector"] = {
            "detector": best_single,
            "weight": _weight_dict(_unit_weight(best_single)),
            "pseudo_metrics": component_metrics[best_single],
        }
        method_rows.append(
            {
                "dataset": dataset,
                "method": "Best Single Detector",
                **_weight_dict(_unit_weight(best_single)),
                **component_metrics[best_single],
            }
        )
        _write_csv(root / dataset / "pseudo_weighting_results.csv", method_rows)
        selections[dataset] = {
            "methods": method_payload,
            "component_metrics": component_metrics,
            "pseudo_held_groups": payload["held_groups"],
            "pseudo_fit_rows": int(payload["fit_train"].sum()),
            "pseudo_ecdf_calibration_rows": int(payload["calibration"].sum()),
            "pseudo_evaluation_rows": int(len(truth)),
        }

    global_weight = _select_global_fine_weight(base, fine_metrics_by_dataset)
    global_pseudo = {
        dataset: fine_metrics_by_dataset[dataset][global_weight] for dataset in DATASETS
    }
    frozen = {
        "official_ood_used_for_selection": False,
        "selection_scope": "source-only pseudo-OOD",
        "component_order": list(COMPONENTS),
        "score_calibration": "per-detector source-ID ECDF",
        "dataset_aware": selections,
        "global_fine_grid": {
            "weight": _weight_dict(global_weight),
            "per_dataset_pseudo_metrics": global_pseudo,
            "mean_pseudo_auroc": float(
                np.mean([row["auroc"] for row in global_pseudo.values()])
            ),
        },
        "algorithms": {
            "Coarse Grid 0.1": "maximize pseudo AUROC on the 66-point simplex",
            "Fine Grid 0.02": "maximize pseudo AUROC on the 1326-point simplex",
            "One-SE Sparse Grid": (
                "choose the sparsest fine-grid weight within one bootstrap SE "
                "of the best pseudo AUROC"
            ),
            "AUROC Softmax": (
                f"softmax of component pseudo AUROCs with T={SOFTMAX_TEMPERATURE}"
            ),
            "Nonnegative Logistic": (
                f"balanced pseudo-OOD logistic stacking with nonnegative coefficients "
                f"and L2={LOGISTIC_L2}"
            ),
            "Shrinkage Fisher": (
                "nonnegative simplex Fisher criterion with Ledoit-Wolf within-class covariance"
            ),
        },
        "fine_grid_step": FINE_STEP,
        "fine_grid_size": len(FINE_GRID),
        "one_se_bootstraps": ONE_SE_BOOTSTRAPS,
    }
    write_json(root / "frozen_selection.json", frozen)

    # Formal phase starts only after all algorithm choices and weights are frozen.
    formal_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        payload = base._fit_components(
            dataset, phase="formal", args=args, study=study, fusion=fusion
        )
        truth, matrix = _calibrated_evaluation(base, payload)
        local_scores: dict[str, np.ndarray] = {}
        for method, details in selections[dataset]["methods"].items():
            weight = _weight_tuple(details["weight"])
            local_scores[method] = matrix @ np.asarray(weight)
        local_scores["Global Fine Grid"] = matrix @ np.asarray(global_weight)
        local_scores["Equal Weight"] = matrix.mean(axis=1)
        raw_scores = [payload["scores"][name] for name in COMPONENTS]
        references = [
            payload["scores"][name][payload["calibration"]] for name in COMPONENTS
        ]
        local_scores["RRF Fusion"] = base._rrf(raw_scores, references)[payload["evaluation"]]

        for method, score in local_scores.items():
            formal_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **ood_metrics(truth, score),
                    "selected_on": "source_only_pseudo_ood",
                }
            )
        bootstrap_rows.extend(
            base._bootstrap_ci(
                dataset,
                truth,
                local_scores,
                bootstrap=int(args.bootstrap),
                seed=int(args.seed),
            )
        )
        _write_csv(
            root / dataset / "formal_weighting_results.csv",
            [row for row in formal_rows if row["dataset"] == dataset],
        )
        np.savez_compressed(
            root / dataset / "formal_weighting_scores.npz",
            truth=truth.astype(np.int8),
            **{_safe_name(name): score.astype(np.float32) for name, score in local_scores.items()},
        )

    _write_csv(root / "formal_all_datasets.csv", formal_rows)
    _write_csv(root / "formal_four_dataset_summary.csv", base._formal_summary(formal_rows))
    _write_csv(root / "bootstrap_95ci.csv", bootstrap_rows)
    manifest = {
        "artifact_type": "source_only_fusion_weighting_study_v1",
        "datasets": list(DATASETS),
        "seed": int(args.seed),
        "hiddenstate_forward_passes": 0,
        "gpu_required": False,
        "api_calls": 0,
        "elapsed_seconds": float(time.perf_counter() - started),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "files": {
            "frozen": str(root / "frozen_selection.json"),
            "formal": str(root / "formal_all_datasets.csv"),
            "summary": str(root / "formal_four_dataset_summary.csv"),
            "bootstrap": str(root / "bootstrap_95ci.csv"),
        },
        "command": [sys.executable, *sys.argv],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _calibrated_evaluation(
    base: ModuleType, payload: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    calibrated = {
        name: base._ecdf(
            payload["scores"][name], payload["scores"][name][payload["calibration"]]
        )
        for name in COMPONENTS
    }
    evaluation = payload["evaluation"]
    matrix = np.column_stack([calibrated[name][evaluation] for name in COMPONENTS])
    return np.asarray(payload["truth"], dtype=int), matrix


def _grid_metrics(
    base: ModuleType,
    truth: np.ndarray,
    matrix: np.ndarray,
    grid: tuple[tuple[float, float, float], ...],
) -> dict[tuple[float, float, float], dict[str, float]]:
    return {
        weight: ood_metrics(truth, matrix @ np.asarray(weight, dtype=np.float64))
        for weight in grid
    }


def _select_weights(
    base: ModuleType,
    truth: np.ndarray,
    matrix: np.ndarray,
    fine_metrics: dict[tuple[float, float, float], dict[str, float]],
    component_metrics: dict[str, dict[str, float]],
    *,
    seed: int,
) -> dict[str, tuple[float, float, float]]:
    coarse_metrics = {weight: fine_metrics[weight] for weight in base.WEIGHT_GRID}
    coarse = base._select_dataset_weight(coarse_metrics)
    fine = base._select_dataset_weight(fine_metrics)
    one_se = _one_se_sparse_weight(truth, matrix, fine, fine_metrics, seed=seed)
    softmax = _auroc_softmax_weight(component_metrics)
    logistic = _nonnegative_logistic_weight(truth, matrix)
    fisher = _shrinkage_fisher_weight(truth, matrix, softmax)
    return {
        "Coarse Grid 0.1": coarse,
        "Fine Grid 0.02": fine,
        "One-SE Sparse Grid": one_se,
        "AUROC Softmax": softmax,
        "Nonnegative Logistic": logistic,
        "Shrinkage Fisher": fisher,
    }


def _one_se_sparse_weight(
    truth: np.ndarray,
    matrix: np.ndarray,
    best: tuple[float, float, float],
    metrics: dict[tuple[float, float, float], dict[str, float]],
    *,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    id_indices = np.flatnonzero(truth == 0)
    ood_indices = np.flatnonzero(truth == 1)
    score = matrix @ np.asarray(best)
    bootstrap_values = []
    for _ in range(ONE_SE_BOOTSTRAPS):
        selected = np.concatenate(
            [
                rng.choice(id_indices, len(id_indices), replace=True),
                rng.choice(ood_indices, len(ood_indices), replace=True),
            ]
        )
        bootstrap_values.append(float(roc_auc_score(truth[selected], score[selected])))
    standard_error = float(np.std(bootstrap_values, ddof=1))
    threshold = float(metrics[best]["auroc"]) - standard_error
    eligible = [
        weight for weight, row in metrics.items() if float(row["auroc"]) >= threshold
    ]
    return max(
        eligible,
        key=lambda weight: (
            -sum(value > 1e-12 for value in weight),
            sum(value * value for value in weight),
            float(metrics[weight]["auroc"]),
            float(metrics[weight]["aupr"]),
            -float(metrics[weight]["fpr95"]),
        ),
    )


def _auroc_softmax_weight(
    component_metrics: dict[str, dict[str, float]],
) -> tuple[float, float, float]:
    values = np.asarray([component_metrics[name]["auroc"] for name in COMPONENTS])
    logits = (values - values.max()) / SOFTMAX_TEMPERATURE
    weights = np.exp(logits)
    weights /= weights.sum()
    return tuple(float(value) for value in weights)


def _nonnegative_logistic_weight(
    truth: np.ndarray, matrix: np.ndarray
) -> tuple[float, float, float]:
    class_weight = np.where(
        truth == 1,
        0.5 / max(1, int((truth == 1).sum())),
        0.5 / max(1, int((truth == 0).sum())),
    )

    def objective(params: np.ndarray) -> float:
        coefficients = params[:3]
        intercept = params[3]
        logits = matrix @ coefficients + intercept
        losses = np.logaddexp(0.0, logits) - truth * logits
        return float(np.sum(class_weight * losses) + LOGISTIC_L2 * np.sum(coefficients**2))

    result = minimize(
        objective,
        x0=np.asarray([1.0, 1.0, 1.0, -1.5]),
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None), (0.0, None), (None, None)],
    )
    coefficients = np.maximum(np.asarray(result.x[:3], dtype=np.float64), 0.0)
    if not result.success or coefficients.sum() <= 1e-12:
        return (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
    coefficients /= coefficients.sum()
    return tuple(float(value) for value in coefficients)


def _shrinkage_fisher_weight(
    truth: np.ndarray,
    matrix: np.ndarray,
    initial: tuple[float, float, float],
) -> tuple[float, float, float]:
    id_values = matrix[truth == 0]
    ood_values = matrix[truth == 1]
    delta = ood_values.mean(axis=0) - id_values.mean(axis=0)
    centered = np.vstack(
        [id_values - id_values.mean(axis=0), ood_values - ood_values.mean(axis=0)]
    )
    covariance = LedoitWolf().fit(centered).covariance_

    def objective(weight: np.ndarray) -> float:
        variance = max(float(weight @ covariance @ weight), 1e-12)
        return -float(delta @ weight) / np.sqrt(variance)

    result = minimize(
        objective,
        x0=np.asarray(initial),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 3,
        constraints=[{"type": "eq", "fun": lambda weight: float(weight.sum() - 1.0)}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        return initial
    weight = np.maximum(np.asarray(result.x), 0.0)
    weight /= weight.sum()
    return tuple(float(value) for value in weight)


def _select_global_fine_weight(
    base: ModuleType,
    metrics: dict[str, dict[tuple[float, float, float], dict[str, float]]],
) -> tuple[float, float, float]:
    def key(weight: tuple[float, float, float]) -> tuple[float, float, float, float]:
        rows = [metrics[dataset][weight] for dataset in DATASETS]
        return (
            float(np.mean([row["auroc"] for row in rows])),
            float(min(row["auroc"] for row in rows)),
            float(np.mean([row["aupr"] for row in rows])),
            base._balance(weight),
        )

    return max(FINE_GRID, key=key)


def _unit_weight(name: str) -> tuple[float, float, float]:
    index = COMPONENTS.index(name)
    return tuple(1.0 if position == index else 0.0 for position in range(3))


def _weight_dict(weight: tuple[float, float, float]) -> dict[str, float]:
    return {
        "w_mahalanobis": float(weight[0]),
        "w_rmd": float(weight[1]),
        "w_vim": float(weight[2]),
    }


def _weight_tuple(values: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(values["w_mahalanobis"]),
        float(values["w_rmd"]),
        float(values["w_vim"]),
    )


def _safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace(".", "_")


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
