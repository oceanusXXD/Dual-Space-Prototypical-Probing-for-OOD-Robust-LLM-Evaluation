#!/usr/bin/env python
"""Fast grouped source-only CV study for three-detector fusion weights."""

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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.shared.metrics import ood_metrics


DATASETS = ("ellipse", "asap", "clinc150", "rostd")
COMPONENTS = ("mahalanobis", "rmd", "vim")
GRID = tuple(
    (m / 10.0, r / 10.0, (10 - m - r) / 10.0)
    for m in range(11)
    for r in range(11 - m)
)
TEMPERATURES = (0.05, 0.1, 0.2, 0.5, 1.0)
L2_VALUES = (0.0, 0.001, 0.01, 0.1, 1.0)
EQUAL_WEIGHT = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fusion-root",
        type=Path,
        default=Path("artifacts/docs_experiments/vim_mahalanobis_fusion_seed42"),
    )
    parser.add_argument(
        "--single-split-root",
        type=Path,
        default=Path("artifacts/docs_experiments/three_way_ecdf_fusion_seed42"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/docs_experiments/robust_cv_fusion_seed42"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--max-folds", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.1)
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
        "fusion43_robust_cv",
        ROOT / "scripts/llm_judge_ood/43_run_three_way_ecdf_fusion.py",
    )
    weighting = _load_module(
        "fusion44_robust_cv",
        ROOT / "scripts/llm_judge_ood/44_run_fusion_weighting_study.py",
    )
    study = _load_module(
        "study39_robust_cv",
        ROOT / "scripts/llm_judge_ood/39_run_vim_mahalanobis_study.py",
    )
    fusion = _load_module(
        "fusion41_robust_cv",
        ROOT / "scripts/llm_judge_ood/41_run_vim_mahalanobis_fusion.py",
    )
    single_split = json.loads(
        (Path(args.single_split_root) / "frozen_selection.json").read_text(encoding="utf-8")
    )["dataset_aware"]

    frozen_datasets: dict[str, Any] = {}
    contexts: dict[str, dict[str, Any]] = {}
    for dataset in DATASETS:
        context = _load_context(dataset, args, study)
        contexts[dataset] = context
        held_folds = _group_folds(
            context["groups"][context["masks"]["train"]],
            max_folds=int(args.max_folds),
            seed=int(args.seed),
            stable_hash=base._stable_hash,
        )
        fold_payloads = []
        for fold_index, held in enumerate(held_folds):
            fold_payloads.append(
                _fit_fold(
                    context,
                    held,
                    fold_index=fold_index,
                    args=args,
                    base=base,
                    study=study,
                    fusion=fusion,
                )
            )

        selected, fold_rows = _select_methods(
            fold_payloads,
            beta=float(args.beta),
            gamma=float(args.gamma),
            seed=int(args.seed),
            weighting=weighting,
        )
        single_weight = _weight_tuple(single_split[dataset]["best_grid_weight"])
        selected["Single-Split Grid"] = {
            "kind": "linear",
            "weight": single_weight,
            "selection": "previous one-fold source-only grid",
            "cv_metrics": _fixed_weight_cv(fold_payloads, single_weight),
        }
        selected["Equal Weight"] = {
            "kind": "linear",
            "weight": EQUAL_WEIGHT,
            "selection": "fixed",
            "cv_metrics": _fixed_weight_cv(fold_payloads, EQUAL_WEIGHT),
        }
        _write_csv(root / dataset / "pseudo_fold_results.csv", fold_rows)
        frozen_datasets[dataset] = {
            "fold_count": len(held_folds),
            "held_groups_by_fold": [held.tolist() for held in held_folds],
            "methods": {
                method: _json_method(details) for method, details in selected.items()
            },
            "head": context["prior"]["head"],
            "layer": context["prior"]["layer"],
            "mahalanobis": context["prior"]["mahalanobis"],
            "vim": context["prior"]["vim"],
        }
        context["selected"] = selected
        del fold_payloads

    frozen = {
        "official_ood_used_for_selection": False,
        "selection_scope": "grouped source-only pseudo-OOD cross-validation",
        "component_order": list(COMPONENTS),
        "score_calibration": "per-fold non-held source validation ID ECDF",
        "max_folds": int(args.max_folds),
        "robust_objective": f"mean_AUROC - {float(args.beta)} * std_AUROC",
        "maximin_objective": "minimum fold AUROC",
        "reliability": f"mean(AUROC - {float(args.gamma)} * FPR95) - {float(args.beta)} * std",
        "temperature_candidates": list(TEMPERATURES),
        "logistic_l2_candidates": list(L2_VALUES),
        "datasets": frozen_datasets,
    }
    write_json(root / "frozen_selection.json", frozen)

    formal_rows = []
    for dataset in DATASETS:
        context = contexts[dataset]
        truth, matrix = _fit_formal(
            context, args=args, base=base, study=study, fusion=fusion
        )
        score_payload = {"truth": truth.astype(np.int8)}
        for method, details in context["selected"].items():
            score = _apply_method(matrix, details)
            formal_rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    **ood_metrics(truth, score),
                    "selected_on": "grouped_source_only_cv",
                }
            )
            score_payload[_safe_name(method)] = score.astype(np.float32)
        _write_csv(
            root / dataset / "formal_results.csv",
            [row for row in formal_rows if row["dataset"] == dataset],
        )
        np.savez_compressed(root / dataset / "formal_scores.npz", **score_payload)
        del contexts[dataset]

    _write_csv(root / "formal_all_datasets.csv", formal_rows)
    _write_csv(root / "formal_four_dataset_summary.csv", _formal_summary(formal_rows))
    manifest = {
        "artifact_type": "grouped_source_only_robust_cv_fusion_v1",
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
        },
        "command": [sys.executable, *sys.argv],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _load_context(dataset: str, args: argparse.Namespace, study: ModuleType) -> dict[str, Any]:
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
    truth = np.asarray(
        [bool(row.get("is_document_ood", row.get("is_ood", False))) for row in rows]
    )
    masks = study._masks(spec, splits, truth)
    groups, _ = study._pseudo_groups(spec, rows, masks["train"], int(args.seed))
    prior = json.loads(
        (Path(args.fusion_root) / dataset / "frozen_selection.json").read_text(encoding="utf-8")
    )
    layer_map = dict(study._available_layers(hidden, metadata))
    raw = study._layer_values(hidden, int(layer_map[str(prior["layer"])]))
    del hidden
    return {
        "dataset": dataset,
        "spec": spec,
        "labels": labels,
        "truth": truth,
        "masks": masks,
        "groups": groups,
        "sample_ids": sample_ids,
        "raw": raw,
        "prior": prior,
    }


def _group_folds(
    train_groups: np.ndarray,
    *,
    max_folds: int,
    seed: int,
    stable_hash: Any,
) -> list[np.ndarray]:
    available = sorted(
        set(train_groups.tolist()), key=lambda value: stable_hash(str(value), seed)
    )
    fold_count = min(max(2, int(max_folds)), len(available))
    return [np.asarray(values, dtype=str) for values in np.array_split(available, fold_count)]


def _fit_fold(
    context: dict[str, Any],
    held: np.ndarray,
    *,
    fold_index: int,
    args: argparse.Namespace,
    base: ModuleType,
    study: ModuleType,
    fusion: ModuleType,
) -> dict[str, Any]:
    groups = context["groups"]
    masks = context["masks"]
    labels = context["labels"]
    fit_train = masks["train"] & ~np.isin(groups, held)
    validation_id = masks["validation"] & ~np.isin(groups, held)
    pseudo_ood = masks["validation"] & np.isin(groups, held)
    calibration, id_eval = base._split_validation_id(
        validation_id,
        labels,
        context["sample_ids"],
        int(args.seed) + int(fold_index),
    )
    evaluation = id_eval | pseudo_ood
    truth = pseudo_ood[evaluation].astype(int)
    matrix = _fit_matrix(
        context,
        fit_train,
        calibration,
        evaluation,
        args=args,
        base=base,
        study=study,
        fusion=fusion,
    )
    return {
        "fold": fold_index,
        "held_groups": held.tolist(),
        "truth": truth,
        "matrix": matrix,
    }


def _fit_formal(
    context: dict[str, Any],
    *,
    args: argparse.Namespace,
    base: ModuleType,
    study: ModuleType,
    fusion: ModuleType,
) -> tuple[np.ndarray, np.ndarray]:
    masks = context["masks"]
    matrix = _fit_matrix(
        context,
        masks["train"],
        masks["validation"],
        masks["benchmark"],
        args=args,
        base=base,
        study=study,
        fusion=fusion,
    )
    return context["truth"][masks["benchmark"]].astype(int), matrix


def _fit_matrix(
    context: dict[str, Any],
    fit_train: np.ndarray,
    calibration: np.ndarray,
    evaluation: np.ndarray,
    *,
    args: argparse.Namespace,
    base: ModuleType,
    study: ModuleType,
    fusion: ModuleType,
) -> np.ndarray:
    pca = study._fit_pca(context["raw"][fit_train], int(args.pca_dim), int(args.seed))
    values = pca.transform(context["raw"]).astype(np.float64)
    prior = context["prior"]
    head = study._fit_head(
        str(prior["head"]),
        values[fit_train],
        context["labels"][fit_train],
        values,
        ordered=bool(context["spec"].ordered),
        seed=int(args.seed),
    )
    scores = fusion._detector_scores(
        study, head, context["labels"], fit_train, dict(prior["vim"])
    )
    selected = {
        "mahalanobis": scores[str(prior["mahalanobis"])],
        "rmd": scores["RMD"],
        "vim": scores["Selected adapted ViM"],
    }
    return np.column_stack(
        [base._ecdf(selected[name], selected[name][calibration])[evaluation] for name in COMPONENTS]
    )


def _select_methods(
    folds: list[dict[str, Any]],
    *,
    beta: float,
    gamma: float,
    seed: int,
    weighting: ModuleType,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    grid_stats = {weight: _fixed_weight_cv(folds, weight) for weight in GRID}
    mean_std_weight = max(
        GRID,
        key=lambda weight: (
            _robust_key(grid_stats[weight], beta),
            grid_stats[weight]["mean_aupr"],
            -grid_stats[weight]["mean_fpr95"],
            _balance(weight),
        ),
    )
    maximin_weight = max(
        GRID,
        key=lambda weight: (
            grid_stats[weight]["min_auroc"],
            grid_stats[weight]["mean_auroc"],
            -grid_stats[weight]["std_auroc"],
            _balance(weight),
        ),
    )
    reliability_weight, temperature, reliability_cv = _select_reliability(
        folds, beta=beta, gamma=gamma
    )
    logistic = _select_nested_logistic(folds, beta=beta)
    lda = _cross_validated_lda(folds, beta=beta, weighting=weighting)

    vertices = {name: _unit_weight(name) for name in COMPONENTS}
    best_single_name = max(
        COMPONENTS,
        key=lambda name: (
            _robust_key(grid_stats[vertices[name]], beta),
            grid_stats[vertices[name]]["min_auroc"],
        ),
    )
    selected = {
        "Robust CV Mean-Std": {
            "kind": "linear",
            "weight": mean_std_weight,
            "selection": f"maximize mean_AUROC - {beta} * std_AUROC",
            "cv_metrics": grid_stats[mean_std_weight],
        },
        "Robust CV Maximin": {
            "kind": "linear",
            "weight": maximin_weight,
            "selection": "maximize minimum fold AUROC",
            "cv_metrics": grid_stats[maximin_weight],
        },
        "Reliability Softmax": {
            "kind": "linear",
            "weight": reliability_weight,
            "temperature": temperature,
            "selection": "robust q=AUROC-gamma*FPR95 softmax",
            "cv_metrics": reliability_cv,
        },
        "Nested Nonnegative Logistic": logistic,
        "CV Shrinkage-LDA": lda,
        "Best Robust Single": {
            "kind": "linear",
            "weight": vertices[best_single_name],
            "detector": best_single_name,
            "selection": "best single by robust CV objective",
            "cv_metrics": grid_stats[vertices[best_single_name]],
        },
    }
    fold_rows = []
    for method, details in selected.items():
        if "fold_metrics" in details:
            local_metrics = details["fold_metrics"]
        else:
            local_metrics = _fold_metrics(folds, details["weight"])
        for fold, metrics in zip(folds, local_metrics):
            fold_rows.append(
                {
                    "fold": fold["fold"],
                    "held_groups": " | ".join(fold["held_groups"]),
                    "method": method,
                    **metrics,
                }
            )
    return selected, fold_rows


def _select_reliability(
    folds: list[dict[str, Any]], *, beta: float, gamma: float
) -> tuple[tuple[float, float, float], float, dict[str, float]]:
    reliability = []
    for component_index in range(3):
        values = []
        for fold in folds:
            metrics = ood_metrics(fold["truth"], fold["matrix"][:, component_index])
            values.append(float(metrics["auroc"]) - gamma * float(metrics["fpr95"]))
        reliability.append(float(np.mean(values) - beta * np.std(values, ddof=0)))
    reliability_array = np.asarray(reliability)
    candidates = []
    for temperature in TEMPERATURES:
        logits = (reliability_array - reliability_array.max()) / temperature
        weight_values = np.exp(logits)
        weight_values /= weight_values.sum()
        weight = tuple(float(value) for value in weight_values)
        candidates.append((temperature, weight, _fixed_weight_cv(folds, weight)))
    temperature, weight, metrics = max(
        candidates,
        key=lambda item: (
            _robust_key(item[2], beta),
            item[2]["min_auroc"],
            -item[2]["mean_fpr95"],
            item[0],
        ),
    )
    return weight, float(temperature), metrics


def _select_nested_logistic(
    folds: list[dict[str, Any]], *, beta: float
) -> dict[str, Any]:
    candidates = []
    for l2 in L2_VALUES:
        fold_metrics = []
        for held_index, held_fold in enumerate(folds):
            train_x = np.vstack(
                [fold["matrix"] for index, fold in enumerate(folds) if index != held_index]
            )
            train_y = np.concatenate(
                [fold["truth"] for index, fold in enumerate(folds) if index != held_index]
            )
            coefficients, intercept = _fit_logistic(train_y, train_x, l2=l2)
            score = held_fold["matrix"] @ coefficients + intercept
            fold_metrics.append(ood_metrics(held_fold["truth"], score))
        summary = _summarize_metrics(fold_metrics)
        candidates.append((l2, fold_metrics, summary))
    l2, fold_metrics, summary = max(
        candidates,
        key=lambda item: (
            _robust_key(item[2], beta),
            item[2]["min_auroc"],
            -item[2]["mean_fpr95"],
            item[0],
        ),
    )
    all_x = np.vstack([fold["matrix"] for fold in folds])
    all_y = np.concatenate([fold["truth"] for fold in folds])
    coefficients, intercept = _fit_logistic(all_y, all_x, l2=l2)
    return {
        "kind": "logistic",
        "coefficients": tuple(float(value) for value in coefficients),
        "intercept": float(intercept),
        "l2": float(l2),
        "selection": "leave-one-pseudo-fold-out robust AUROC",
        "cv_metrics": summary,
        "fold_metrics": fold_metrics,
    }


def _cross_validated_lda(
    folds: list[dict[str, Any]], *, beta: float, weighting: ModuleType
) -> dict[str, Any]:
    fold_metrics = []
    for held_index, held_fold in enumerate(folds):
        train_x = np.vstack(
            [fold["matrix"] for index, fold in enumerate(folds) if index != held_index]
        )
        train_y = np.concatenate(
            [fold["truth"] for index, fold in enumerate(folds) if index != held_index]
        )
        weight = weighting._shrinkage_fisher_weight(train_y, train_x, EQUAL_WEIGHT)
        fold_metrics.append(
            ood_metrics(held_fold["truth"], held_fold["matrix"] @ np.asarray(weight))
        )
    all_x = np.vstack([fold["matrix"] for fold in folds])
    all_y = np.concatenate([fold["truth"] for fold in folds])
    final_weight = weighting._shrinkage_fisher_weight(all_y, all_x, EQUAL_WEIGHT)
    return {
        "kind": "linear",
        "weight": final_weight,
        "selection": "Ledoit-Wolf nonnegative Fisher fit on pooled out-of-fold pseudo scores",
        "cv_metrics": _summarize_metrics(fold_metrics),
        "fold_metrics": fold_metrics,
        "robust_objective": _robust_key(_summarize_metrics(fold_metrics), beta),
    }


def _fit_logistic(
    truth: np.ndarray, matrix: np.ndarray, *, l2: float
) -> tuple[np.ndarray, float]:
    sample_weight = np.where(
        truth == 1,
        0.5 / max(1, int((truth == 1).sum())),
        0.5 / max(1, int((truth == 0).sum())),
    )

    def objective(params: np.ndarray) -> float:
        coefficients = params[:3]
        logits = matrix @ coefficients + params[3]
        loss = np.logaddexp(0.0, logits) - truth * logits
        return float(np.sum(sample_weight * loss) + l2 * np.sum(coefficients**2))

    result = minimize(
        objective,
        x0=np.asarray([1.0, 1.0, 1.0, -1.5]),
        method="L-BFGS-B",
        bounds=[(0.0, None), (0.0, None), (0.0, None), (None, None)],
    )
    coefficients = np.maximum(np.asarray(result.x[:3], dtype=np.float64), 0.0)
    if not result.success or coefficients.sum() <= 1e-12:
        coefficients = np.asarray(EQUAL_WEIGHT)
    return coefficients, float(result.x[3])


def _fixed_weight_cv(
    folds: list[dict[str, Any]], weight: tuple[float, float, float]
) -> dict[str, float]:
    return _summarize_metrics(_fold_metrics(folds, weight))


def _fold_metrics(
    folds: list[dict[str, Any]], weight: tuple[float, float, float]
) -> list[dict[str, float]]:
    values = np.asarray(weight, dtype=np.float64)
    return [ood_metrics(fold["truth"], fold["matrix"] @ values) for fold in folds]


def _summarize_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    aurocs = np.asarray([row["auroc"] for row in rows], dtype=np.float64)
    return {
        "mean_auroc": float(aurocs.mean()),
        "std_auroc": float(aurocs.std(ddof=0)),
        "min_auroc": float(aurocs.min()),
        "mean_aupr": float(np.mean([row["aupr"] for row in rows])),
        "mean_fpr95": float(np.mean([row["fpr95"] for row in rows])),
    }


def _robust_key(metrics: dict[str, float], beta: float) -> float:
    return float(metrics["mean_auroc"] - beta * metrics["std_auroc"])


def _apply_method(matrix: np.ndarray, details: dict[str, Any]) -> np.ndarray:
    if details["kind"] == "logistic":
        return matrix @ np.asarray(details["coefficients"]) + float(details["intercept"])
    return matrix @ np.asarray(details["weight"])


def _unit_weight(name: str) -> tuple[float, float, float]:
    index = COMPONENTS.index(name)
    return tuple(1.0 if position == index else 0.0 for position in range(3))


def _weight_tuple(values: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float(values["w_mahalanobis"]),
        float(values["w_rmd"]),
        float(values["w_vim"]),
    )


def _balance(weight: tuple[float, float, float]) -> float:
    return -float(sum((value - 1.0 / 3.0) ** 2 for value in weight))


def _json_method(details: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in details.items() if key != "fold_metrics"}
    if "weight" in output:
        output["weight"] = {
            name: float(value) for name, value in zip(COMPONENTS, output["weight"])
        }
    if "coefficients" in output:
        output["coefficients"] = {
            name: float(value) for name, value in zip(COMPONENTS, output["coefficients"])
        }
    return output


def _formal_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = list(dict.fromkeys(row["method"] for row in rows))
    output = []
    for method in methods:
        local = [row for row in rows if row["method"] == method]
        output.append(
            {
                "method": method,
                "mean_auroc": float(np.mean([row["auroc"] for row in local])),
                "mean_aupr": float(np.mean([row["aupr"] for row in local])),
                "mean_fpr95": float(np.mean([row["fpr95"] for row in local])),
            }
        )
    return output


def _safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


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
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
