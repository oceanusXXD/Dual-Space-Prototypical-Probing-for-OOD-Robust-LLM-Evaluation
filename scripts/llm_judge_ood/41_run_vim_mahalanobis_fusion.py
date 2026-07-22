#!/usr/bin/env python
"""Fuse frozen ViM with source-selected Mahalanobis on cached hidden states.

The detector family, score calibration, and fusion rule are selected only on
source-side pseudo-OOD groups. Official OOD rows are evaluated after the
selection has been written to disk.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy.stats import norm, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.shared.metrics import ood_metrics


STUDY_ROOT = Path("artifacts/docs_experiments/vim_mahalanobis_study_seed42")
OUTPUT_ROOT = Path("artifacts/docs_experiments/vim_mahalanobis_fusion_seed42")
SPECTRAL_GAMMAS = (0.0, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0)
VIM_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
CALIBRATIONS = ("empirical_normal", "robust_z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ellipse", "asap", "clinc150", "rostd"), required=True)
    parser.add_argument("--study-root", type=Path, default=STUDY_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    study = _load_study_module()
    spec = study.SPECS[str(args.dataset)]
    output = Path(args.output_root) / spec.name
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path.read_text(encoding="utf-8"))
        return
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    rows = read_jsonl(spec.prepared)
    cache = np.load(spec.hidden, allow_pickle=True)
    hidden = np.asarray(cache["features"], dtype=np.float32)
    sample_ids = np.asarray(cache["sample_ids"]).astype(str)
    expected_ids = np.asarray([str(row["sample_id"]) for row in rows])
    if not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError(f"Hiddenstate row alignment failed for {spec.name}")

    labels = np.asarray([str(row["label"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    truth = np.asarray(
        [bool(row.get("is_document_ood", row.get("is_ood", False))) for row in rows]
    )
    masks = study._masks(spec, splits, truth)
    pseudo_groups, held_groups = study._pseudo_groups(spec, rows, masks["train"], int(args.seed))
    pseudo_train = masks["train"] & ~np.isin(pseudo_groups, held_groups)
    pseudo_ood = masks["train"] & np.isin(pseudo_groups, held_groups)
    pseudo_eval = pseudo_train | pseudo_ood
    if not pseudo_train.any() or not pseudo_ood.any():
        raise RuntimeError("Source-only pseudo-OOD split is empty")

    prior_path = Path(args.study_root) / spec.name / "frozen_selection.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    selected_head = str(prior["head"])
    selected_layer = str(prior["layer"])
    selected_vim = dict(prior["vim"])

    metadata = json.loads(str(np.asarray(cache["metadata_json"]).item()))
    layer_map = dict(study._available_layers(hidden, metadata))
    if selected_layer not in layer_map:
        raise RuntimeError(f"Selected layer {selected_layer!r} is not in hiddenstate cache")
    raw = study._layer_values(hidden, int(layer_map[selected_layer]))

    # Phase 1: all detector and fusion choices use only source training groups.
    pseudo_pca = study._fit_pca(raw[pseudo_train], int(args.pca_dim), int(args.seed))
    pseudo_x = pseudo_pca.transform(raw).astype(np.float64)
    pseudo_head = study._fit_head(
        selected_head,
        pseudo_x[pseudo_train],
        labels[pseudo_train],
        pseudo_x,
        ordered=bool(spec.ordered),
        seed=int(args.seed),
    )
    pseudo_scores = _detector_scores(
        study,
        pseudo_head,
        labels,
        pseudo_train,
        selected_vim,
    )
    pseudo_y = pseudo_ood[pseudo_eval].astype(int)
    candidate_rows = _candidate_rows(
        spec.name,
        pseudo_scores,
        pseudo_train,
        pseudo_eval,
        pseudo_y,
    )
    _write_csv(output / "pseudo_mahalanobis_candidates.csv", candidate_rows)
    selected_maha = _select_mahalanobis(candidate_rows)

    fusion_rows, pseudo_diagnostics = _fusion_grid(
        spec.name,
        pseudo_scores["Selected adapted ViM"],
        pseudo_scores[selected_maha],
        pseudo_train,
        pseudo_eval,
        pseudo_y,
        selected_maha=selected_maha,
    )
    _write_csv(output / "pseudo_fusion_grid.csv", fusion_rows)
    selected_fusion = _select_fusion(fusion_rows)
    frozen = {
        "dataset": spec.name,
        "head": selected_head,
        "layer": selected_layer,
        "vim": selected_vim,
        "mahalanobis": selected_maha,
        "fusion": selected_fusion,
        "selection_scope": "source-only pseudo-OOD groups",
        "pseudo_held_groups": held_groups.tolist(),
        "official_ood_used_for_selection": False,
    }
    write_json(output / "frozen_selection.json", frozen)

    # Phase 2: refit the frozen configuration on all source training rows.
    formal_pca = study._fit_pca(raw[masks["train"]], int(args.pca_dim), int(args.seed))
    formal_x = formal_pca.transform(raw).astype(np.float64)
    del raw, pseudo_x
    formal_head = study._fit_head(
        selected_head,
        formal_x[masks["train"]],
        labels[masks["train"]],
        formal_x,
        ordered=bool(spec.ordered),
        seed=int(args.seed),
    )
    formal_scores = _detector_scores(
        study,
        formal_head,
        labels,
        masks["train"],
        selected_vim,
    )
    fusion_score = _apply_fusion(
        formal_scores["Selected adapted ViM"],
        formal_scores[selected_maha],
        masks["train"],
        selected_fusion,
    )

    benchmark = masks["benchmark"]
    y_benchmark = truth[benchmark].astype(int)
    report_scores = {
        "Frozen adapted ViM": formal_scores["Selected adapted ViM"],
        "Frozen selected Mahalanobis": formal_scores[selected_maha],
        "Frozen ViM-Mahalanobis fusion": fusion_score,
        "RMD": formal_scores["RMD"],
    }
    for name, score in formal_scores.items():
        if name.startswith("Mahalanobis") or name.startswith("Spectral Mahalanobis"):
            report_scores[name] = score
    formal_rows = [
        {
            "dataset": spec.name,
            "method": name,
            **ood_metrics(y_benchmark, score[benchmark]),
            "selected_on": "source_only_pseudo_ood",
        }
        for name, score in report_scores.items()
    ]
    _write_csv(output / "formal_ood_results.csv", formal_rows)

    formal_diagnostics = _complementarity(
        formal_scores["Selected adapted ViM"],
        formal_scores[selected_maha],
        masks["train"],
        benchmark,
        y_benchmark,
    )
    diagnostics = {
        "pseudo": pseudo_diagnostics,
        "formal": formal_diagnostics,
        "selected_mahalanobis": selected_maha,
        "selected_fusion": selected_fusion,
    }
    write_json(output / "complementarity.json", diagnostics)

    per_group = study._per_group_rows(spec, rows, truth, benchmark, report_scores)
    _write_csv(output / "formal_per_prompt_domain.csv", per_group)
    ci_rows, paired_rows = _bootstrap_rows(
        spec.name,
        y_benchmark,
        {
            "Frozen adapted ViM": report_scores["Frozen adapted ViM"][benchmark],
            "Frozen selected Mahalanobis": report_scores["Frozen selected Mahalanobis"][benchmark],
            "Frozen ViM-Mahalanobis fusion": report_scores["Frozen ViM-Mahalanobis fusion"][benchmark],
        },
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    _write_csv(output / "bootstrap_95ci.csv", ci_rows)
    _write_csv(output / "paired_fusion_comparisons.csv", paired_rows)
    np.savez_compressed(
        output / "formal_sample_scores.npz",
        sample_ids=sample_ids,
        truth=truth.astype(np.int8),
        benchmark_mask=benchmark.astype(np.int8),
        vim=report_scores["Frozen adapted ViM"].astype(np.float32),
        mahalanobis=report_scores["Frozen selected Mahalanobis"].astype(np.float32),
        fusion=report_scores["Frozen ViM-Mahalanobis fusion"].astype(np.float32),
    )

    manifest = {
        "artifact_type": "source_only_vim_mahalanobis_fusion_v1",
        "dataset": spec.name,
        "seed": int(args.seed),
        "hiddenstate_forward_passes": 0,
        "gpu_required": False,
        "api_calls": 0,
        "prior_study": str(prior_path),
        "prepared": spec.prepared,
        "hidden": spec.hidden,
        "hidden_shape": list(hidden.shape),
        "pooling": metadata.get("pooling", metadata.get("pooling_method", "masked_mean")),
        "pseudo_train_rows": int(pseudo_train.sum()),
        "pseudo_ood_rows": int(pseudo_ood.sum()),
        "formal_id_rows": int((benchmark & ~truth).sum()),
        "formal_ood_rows": int((benchmark & truth).sum()),
        "frozen_selection": frozen,
        "spectral_definition": (
            "shared within-class eigenspectrum: sum_j <x-mu_c,v_j>^2/"
            "(eigenvalue_j+gamma*mean_eigenvalue); compare min-class, "
            "Judge-predicted-class, and posterior-weighted aggregation"
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "files": {
            "pseudo_mahalanobis": str(output / "pseudo_mahalanobis_candidates.csv"),
            "pseudo_fusion": str(output / "pseudo_fusion_grid.csv"),
            "frozen": str(output / "frozen_selection.json"),
            "formal": str(output / "formal_ood_results.csv"),
            "complementarity": str(output / "complementarity.json"),
            "per_group": str(output / "formal_per_prompt_domain.csv"),
            "bootstrap": str(output / "bootstrap_95ci.csv"),
            "paired": str(output / "paired_fusion_comparisons.csv"),
            "scores": str(output / "formal_sample_scores.npz"),
        },
        "command": [sys.executable, *sys.argv],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _load_study_module() -> ModuleType:
    path = ROOT / "scripts/llm_judge_ood/39_run_vim_mahalanobis_study.py"
    spec = importlib.util.spec_from_file_location("vim_mahalanobis_study", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _detector_scores(
    study: ModuleType,
    head: Any,
    labels: np.ndarray,
    train: np.ndarray,
    selected_vim: dict[str, Any],
) -> dict[str, np.ndarray]:
    scores = study._formal_scores(head, labels, train, selected_vim)
    scores.update(
        _spectral_mahalanobis_scores(
            np.asarray(head.penultimate[train], dtype=np.float64),
            np.asarray(head.penultimate, dtype=np.float64),
            labels[train],
            np.asarray(head.predictions).astype(str),
            np.asarray(head.classes).astype(str),
            np.asarray(head.probabilities, dtype=np.float64),
        )
    )
    return scores


def _spectral_mahalanobis_scores(
    train_values: np.ndarray,
    all_values: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    head_classes: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, np.ndarray]:
    classes = sorted(set(labels.tolist()))
    means = np.stack([train_values[labels == label].mean(axis=0) for label in classes])
    residuals = np.vstack(
        [train_values[labels == label] - means[index] for index, label in enumerate(classes)]
    )
    covariance = residuals.T @ residuals / max(len(residuals) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    scale = max(float(np.mean(eigenvalues)), 1e-12)
    floor = max(scale * 1e-8, 1e-12)
    output: dict[str, np.ndarray] = {}
    class_index = {label: index for index, label in enumerate(classes)}
    predicted_index = np.asarray([class_index[value] for value in predictions], dtype=int)
    probability_index = np.asarray([list(head_classes).index(label) for label in classes], dtype=int)
    ordered_probabilities = probabilities[:, probability_index]
    for gamma in SPECTRAL_GAMMAS:
        inverse = 1.0 / np.maximum(eigenvalues + float(gamma) * scale, floor)
        precision = (eigenvectors * inverse[None, :]) @ eigenvectors.T
        distances = _mahalanobis_distances(all_values, means, precision)
        output[f"Spectral Mahalanobis gamma={gamma:g}"] = np.min(distances, axis=1)
        output[f"Spectral Mahalanobis predicted-class gamma={gamma:g}"] = distances[
            np.arange(len(distances)), predicted_index
        ]
        output[f"Spectral Mahalanobis posterior-weighted gamma={gamma:g}"] = np.sum(
            ordered_probabilities * distances, axis=1
        )
    return output


def _min_mahalanobis(values: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    return np.min(_mahalanobis_distances(values, means, precision), axis=1)


def _mahalanobis_distances(
    values: np.ndarray,
    means: np.ndarray,
    precision: np.ndarray,
) -> np.ndarray:
    projected = values @ precision
    value_term = np.sum(projected * values, axis=1)
    mean_term = np.sum((means @ precision) * means, axis=1)
    distances = value_term[:, None] - 2.0 * projected @ means.T + mean_term[None, :]
    return np.maximum(distances, 0.0)


def _candidate_rows(
    dataset: str,
    scores: dict[str, np.ndarray],
    train: np.ndarray,
    evaluation: np.ndarray,
    truth: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for name, score in scores.items():
        if not (name.startswith("Mahalanobis") or name.startswith("Spectral Mahalanobis")):
            continue
        rows.append(
            {
                "dataset": dataset,
                "method": name,
                **ood_metrics(truth, score[evaluation]),
                "source_score_median": float(np.median(score[train])),
                "source_score_q95": float(np.quantile(score[train], 0.95)),
            }
        )
    return rows


def _select_mahalanobis(rows: list[dict[str, Any]]) -> str:
    return str(
        max(
            rows,
            key=lambda row: (
                float(row["auroc"]),
                float(row["aupr"]),
                -float(row["fpr95"]),
                _mahalanobis_simplicity(str(row["method"])),
            ),
        )["method"]
    )


def _mahalanobis_simplicity(method: str) -> int:
    preference = {
        "Mahalanobis shrinkage": 4,
        "Mahalanobis shared empirical": 3,
        "Mahalanobis diagonal": 2,
        "Mahalanobis class-balanced": 1,
    }
    return preference.get(method, 0)


def _fusion_grid(
    dataset: str,
    vim: np.ndarray,
    maha: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    truth: np.ndarray,
    *,
    selected_maha: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for calibration in CALIBRATIONS:
        vim_cal = _calibrate(vim, vim[train], calibration)
        maha_cal = _calibrate(maha, maha[train], calibration)
        for weight in VIM_WEIGHTS:
            fused = float(weight) * vim_cal + (1.0 - float(weight)) * maha_cal
            rows.append(
                _fusion_row(
                    dataset,
                    selected_maha,
                    calibration,
                    "convex_sum",
                    weight,
                    truth,
                    fused[evaluation],
                )
            )
        for rule, fused in (
            ("max", np.maximum(vim_cal, maha_cal)),
            ("min", np.minimum(vim_cal, maha_cal)),
        ):
            rows.append(
                _fusion_row(
                    dataset,
                    selected_maha,
                    calibration,
                    rule,
                    None,
                    truth,
                    fused[evaluation],
                )
            )
        vim_tail = _upper_tail_probability(vim, vim[train])
        maha_tail = _upper_tail_probability(maha, maha[train])
        fisher = -2.0 * (np.log(vim_tail) + np.log(maha_tail))
        rows.append(
            _fusion_row(
                dataset,
                selected_maha,
                calibration,
                "fisher_tail",
                None,
                truth,
                fisher[evaluation],
            )
        )
    return rows, _complementarity(vim, maha, train, evaluation, truth)


def _fusion_row(
    dataset: str,
    maha: str,
    calibration: str,
    rule: str,
    vim_weight: float | None,
    truth: np.ndarray,
    score: np.ndarray,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "mahalanobis": maha,
        "calibration": calibration,
        "rule": rule,
        "vim_weight": "" if vim_weight is None else float(vim_weight),
        **ood_metrics(truth, score),
    }


def _select_fusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = max(
        rows,
        key=lambda row: (
            float(row["auroc"]),
            float(row["aupr"]),
            -float(row["fpr95"]),
            _simplicity_key(row),
        ),
    )
    return dict(selected)


def _simplicity_key(row: dict[str, Any]) -> tuple[float, float]:
    if str(row["rule"]) != "convex_sum":
        return (-1.0, -1.0)
    weight = float(row["vim_weight"])
    return (1.0, -abs(weight - round(weight)))


def _apply_fusion(
    vim: np.ndarray,
    maha: np.ndarray,
    train: np.ndarray,
    selection: dict[str, Any],
) -> np.ndarray:
    calibration = str(selection["calibration"])
    rule = str(selection["rule"])
    vim_cal = _calibrate(vim, vim[train], calibration)
    maha_cal = _calibrate(maha, maha[train], calibration)
    if rule == "convex_sum":
        weight = float(selection["vim_weight"])
        return weight * vim_cal + (1.0 - weight) * maha_cal
    if rule == "max":
        return np.maximum(vim_cal, maha_cal)
    if rule == "min":
        return np.minimum(vim_cal, maha_cal)
    if rule == "fisher_tail":
        return -2.0 * (
            np.log(_upper_tail_probability(vim, vim[train]))
            + np.log(_upper_tail_probability(maha, maha[train]))
        )
    raise ValueError(f"Unknown fusion rule {rule!r}")


def _calibrate(values: np.ndarray, reference: np.ndarray, method: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if method == "empirical_normal":
        ordered = np.sort(reference)
        probabilities = (np.searchsorted(ordered, values, side="right") + 0.5) / (len(ordered) + 1.0)
        bound = 0.5 / (len(ordered) + 1.0)
        return norm.ppf(np.clip(probabilities, bound, 1.0 - bound))
    if method == "robust_z":
        center = float(np.median(reference))
        mad = float(np.median(np.abs(reference - center))) * 1.4826
        scale = mad if mad > 1e-12 else max(float(np.std(reference, ddof=1)), 1e-12)
        return (values - center) / scale
    raise ValueError(f"Unknown calibration {method!r}")


def _upper_tail_probability(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(reference, dtype=np.float64))
    ranks = np.searchsorted(ordered, np.asarray(values, dtype=np.float64), side="left")
    tail = (len(ordered) - ranks + 0.5) / (len(ordered) + 1.0)
    floor = 0.5 / (len(ordered) + 1.0)
    return np.clip(tail, floor, 1.0)


def _complementarity(
    vim: np.ndarray,
    maha: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    truth: np.ndarray,
) -> dict[str, Any]:
    local_vim = vim[evaluation]
    local_maha = maha[evaluation]
    local_ood = truth.astype(bool)
    vim_z = _calibrate(vim, vim[train], "empirical_normal")[evaluation]
    maha_z = _calibrate(maha, maha[train], "empirical_normal")[evaluation]
    return {
        "id_spearman": _safe_spearman(local_vim[~local_ood], local_maha[~local_ood]),
        "ood_spearman": _safe_spearman(local_vim[local_ood], local_maha[local_ood]),
        "source_fit_spearman": _safe_spearman(vim[train], maha[train]),
        "calibrated_difference_std": float(np.std(vim_z - maha_z)),
        "vim_q95_exceedance_ood": float(np.mean(local_vim[local_ood] > np.quantile(vim[train], 0.95))),
        "mahalanobis_q95_exceedance_ood": float(np.mean(local_maha[local_ood] > np.quantile(maha[train], 0.95))),
    }


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return float("nan")
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else float("nan")


def _bootstrap_rows(
    dataset: str,
    truth: np.ndarray,
    methods: dict[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    id_indices = np.flatnonzero(truth == 0)
    ood_indices = np.flatnonzero(truth == 1)
    values = {name: [] for name in methods}
    for _ in range(int(bootstrap)):
        selected = np.concatenate(
            [
                rng.choice(id_indices, len(id_indices), replace=True),
                rng.choice(ood_indices, len(ood_indices), replace=True),
            ]
        )
        local_truth = truth[selected]
        for name, score in methods.items():
            values[name].append(float(roc_auc_score(local_truth, score[selected])))
    ci_rows = [
        {
            "dataset": dataset,
            "method": name,
            "auroc": float(roc_auc_score(truth, methods[name])),
            "ci_low": float(np.quantile(values[name], 0.025)),
            "ci_high": float(np.quantile(values[name], 0.975)),
            "bootstrap_samples": int(bootstrap),
        }
        for name in methods
    ]
    paired_rows = []
    fusion_name = "Frozen ViM-Mahalanobis fusion"
    for baseline in ("Frozen adapted ViM", "Frozen selected Mahalanobis"):
        differences = np.asarray(values[fusion_name]) - np.asarray(values[baseline])
        p_value = 2.0 * min(
            (float(np.sum(differences <= 0.0)) + 1.0) / (len(differences) + 1.0),
            (float(np.sum(differences >= 0.0)) + 1.0) / (len(differences) + 1.0),
        )
        paired_rows.append(
            {
                "dataset": dataset,
                "method_a": fusion_name,
                "method_b": baseline,
                "auroc_difference": float(
                    roc_auc_score(truth, methods[fusion_name])
                    - roc_auc_score(truth, methods[baseline])
                ),
                "difference_ci_low": float(np.quantile(differences, 0.025)),
                "difference_ci_high": float(np.quantile(differences, 0.975)),
                "paired_bootstrap_p_two_sided": min(float(p_value), 1.0),
                "bootstrap_samples": int(bootstrap),
            }
        )
    return ci_rows, paired_rows


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
