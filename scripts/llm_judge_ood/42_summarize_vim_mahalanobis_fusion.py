#!/usr/bin/env python
"""Aggregate frozen ViM/Mahalanobis fusion results and oracle diagnostics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.shared.metrics import ood_metrics


DATASETS = ("ellipse", "asap", "clinc150", "rostd")
METHOD_MAP = {
    "Frozen adapted ViM": "Adapted ViM",
    "Frozen selected Mahalanobis": "Source-selected Mahalanobis family",
    "Frozen ViM-Mahalanobis fusion": "Source-selected fusion/router",
    "RMD": "RMD",
}
ORACLE_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/docs_experiments/vim_mahalanobis_fusion_seed42"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    study = _load_module("study39", ROOT / "scripts/llm_judge_ood/39_run_vim_mahalanobis_study.py")
    fusion = _load_module("fusion41", ROOT / "scripts/llm_judge_ood/41_run_vim_mahalanobis_fusion.py")

    frozen_rows: list[dict[str, Any]] = []
    structural_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    oracle_best: list[dict[str, Any]] = []
    configurations: dict[str, Any] = {}
    for dataset in DATASETS:
        dataset_root = root / dataset
        formal = _read_csv(dataset_root / "formal_ood_results.csv")
        frozen = json.loads((dataset_root / "frozen_selection.json").read_text(encoding="utf-8"))
        configurations[dataset] = frozen
        by_method = {str(row["method"]): row for row in formal}
        for stored_name, display_name in METHOD_MAP.items():
            row = by_method[stored_name]
            frozen_rows.append(
                {
                    "dataset": dataset,
                    "method": display_name,
                    "auroc": float(row["auroc"]),
                    "aupr": float(row["aupr"]),
                    "fpr95": float(row["fpr95"]),
                }
            )

        conventional = [row for row in formal if str(row["method"]).startswith("Mahalanobis ")]
        structural = [row for row in formal if str(row["method"]).startswith("Spectral Mahalanobis ")]
        best_conventional = _best(conventional)
        best_structural = _best(structural)
        selected_name = str(frozen["mahalanobis"])
        selected_row = by_method[selected_name]
        structural_rows.append(
            {
                "dataset": dataset,
                "best_conventional_method_official_diagnostic": best_conventional["method"],
                "best_conventional_auroc": float(best_conventional["auroc"]),
                "best_structural_method_official_diagnostic": best_structural["method"],
                "best_structural_auroc": float(best_structural["auroc"]),
                "structural_minus_conventional": float(best_structural["auroc"])
                - float(best_conventional["auroc"]),
                "source_selected_method": selected_name,
                "source_selected_auroc": float(selected_row["auroc"]),
                "official_test_used_for_method_selection": False,
            }
        )

        manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
        spec = study.SPECS[dataset]
        prepared = read_jsonl(manifest["prepared"])
        splits = np.asarray([str(row["split"]) for row in prepared])
        truth = np.asarray(
            [bool(row.get("is_document_ood", row.get("is_ood", False))) for row in prepared]
        )
        masks = study._masks(spec, splits, truth)
        scores = np.load(dataset_root / "formal_sample_scores.npz")
        vim_score = np.asarray(scores["vim"], dtype=np.float64)
        maha_score = np.asarray(scores["mahalanobis"], dtype=np.float64)
        benchmark = masks["benchmark"]
        benchmark_truth = truth[benchmark].astype(int)
        for calibration in fusion.CALIBRATIONS:
            calibrated_vim = fusion._calibrate(vim_score, vim_score[masks["train"]], calibration)
            calibrated_maha = fusion._calibrate(maha_score, maha_score[masks["train"]], calibration)
            for weight in ORACLE_WEIGHTS:
                combined = float(weight) * calibrated_vim + (1.0 - float(weight)) * calibrated_maha
                oracle_rows.append(
                    {
                        "dataset": dataset,
                        "calibration": calibration,
                        "vim_weight": float(weight),
                        **ood_metrics(benchmark_truth, combined[benchmark]),
                        "selection_scope": "official_test_oracle_diagnostic_only",
                    }
                )
        local_oracle = [row for row in oracle_rows if row["dataset"] == dataset]
        best_oracle = _best(local_oracle)
        oracle_best.append(dict(best_oracle))

    _write_csv(root / "formal_frozen_all_datasets.csv", frozen_rows)
    _write_csv(root / "formal_structural_mahalanobis_diagnostic.csv", structural_rows)
    _write_csv(root / "official_oracle_fusion_weight_diagnostic.csv", oracle_rows)
    _write_csv(root / "official_oracle_fusion_best_diagnostic.csv", oracle_best)

    means = []
    for method in METHOD_MAP.values():
        local = [row for row in frozen_rows if row["method"] == method]
        means.append(
            {
                "method": method,
                "mean_auroc": float(np.mean([row["auroc"] for row in local])),
                "mean_aupr": float(np.mean([row["aupr"] for row in local])),
                "mean_fpr95": float(np.mean([row["fpr95"] for row in local])),
                "datasets": len(local),
            }
        )
    _write_csv(root / "formal_four_dataset_mean.csv", means)

    by_dataset_method = {
        (row["dataset"], row["method"]): float(row["auroc"]) for row in frozen_rows
    }
    fusion_values = np.asarray(
        [by_dataset_method[(dataset, "Source-selected fusion/router")] for dataset in DATASETS]
    )
    cross_significance = {}
    for baseline in ("Adapted ViM", "Source-selected Mahalanobis family", "RMD"):
        baseline_values = np.asarray(
            [by_dataset_method[(dataset, baseline)] for dataset in DATASETS]
        )
        cross_significance[baseline] = {
            "fusion_minus_baseline": (fusion_values - baseline_values).tolist(),
            "mean_difference": float(np.mean(fusion_values - baseline_values)),
            "wilcoxon_two_sided_p": _safe_wilcoxon(fusion_values, baseline_values),
        }

    aggregate = {
        "artifact_type": "source_only_vim_mahalanobis_fusion_aggregate_v1",
        "datasets": list(DATASETS),
        "hiddenstate_forward_passes": 0,
        "api_calls": 0,
        "selection_rule": "all frozen choices use source-only pseudo-OOD",
        "official_oracle_warning": (
            "official oracle weight and best-structural columns are diagnostics only and must not be deployed"
        ),
        "frozen_configurations": configurations,
        "cross_dataset_significance": cross_significance,
        "files": {
            "formal": str(root / "formal_frozen_all_datasets.csv"),
            "mean": str(root / "formal_four_dataset_mean.csv"),
            "structural_diagnostic": str(root / "formal_structural_mahalanobis_diagnostic.csv"),
            "oracle_grid": str(root / "official_oracle_fusion_weight_diagnostic.csv"),
            "oracle_best": str(root / "official_oracle_fusion_best_diagnostic.csv"),
        },
    }
    write_json(root / "aggregate_manifest.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (
            float(row["auroc"]),
            float(row["aupr"]),
            -float(row["fpr95"]),
        ),
    )


def _safe_wilcoxon(left: np.ndarray, right: np.ndarray) -> float:
    if np.allclose(left, right):
        return 1.0
    return float(wilcoxon(left, right).pvalue)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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
