#!/usr/bin/env python
"""Paired-bootstrap comparisons for the source-only fusion weighting study."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


DATASETS = ("ellipse", "asap", "clinc150", "rostd")
COMPARISONS = (
    ("shrinkage_fisher", "coarse_grid_0_1"),
    ("fine_grid_0_02", "coarse_grid_0_1"),
    ("shrinkage_fisher", "best_single_detector"),
)
DISPLAY = {
    "shrinkage_fisher": "Shrinkage Fisher",
    "coarse_grid_0_1": "Coarse Grid 0.1",
    "fine_grid_0_02": "Fine Grid 0.02",
    "best_single_detector": "Best Single Detector",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("artifacts/docs_experiments/fusion_weighting_study_seed42"),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    rows = []
    for dataset_index, dataset in enumerate(DATASETS):
        path = args.root / dataset / "formal_weighting_scores.npz"
        with np.load(path) as data:
            truth = np.asarray(data["truth"], dtype=int)
            scores = {name: np.asarray(data[name], dtype=np.float64) for name in data.files}
        samples[dataset] = {}
        for comparison_index, (candidate, baseline) in enumerate(COMPARISONS):
            differences = _bootstrap_differences(
                truth,
                scores[candidate],
                scores[baseline],
                bootstrap=int(args.bootstrap),
                seed=int(args.seed) + 100 * dataset_index + comparison_index,
            )
            samples[dataset][(candidate, baseline)] = differences
            rows.append(
                _row(
                    dataset,
                    candidate,
                    baseline,
                    float(
                        roc_auc_score(truth, scores[candidate])
                        - roc_auc_score(truth, scores[baseline])
                    ),
                    differences,
                    int(args.bootstrap),
                )
            )

    for candidate, baseline in COMPARISONS:
        mean_differences = np.mean(
            [samples[dataset][(candidate, baseline)] for dataset in DATASETS], axis=0
        )
        observed = float(
            np.mean(
                [
                    next(
                        row["auroc_difference"]
                        for row in rows
                        if row["dataset"] == dataset
                        and row["candidate"] == DISPLAY[candidate]
                        and row["baseline"] == DISPLAY[baseline]
                    )
                    for dataset in DATASETS
                ]
            )
        )
        rows.append(
            _row(
                "four_dataset_mean",
                candidate,
                baseline,
                observed,
                mean_differences,
                int(args.bootstrap),
            )
        )

    _write_csv(args.root / "paired_bootstrap_differences.csv", rows)
    for row in rows:
        print(
            f"{row['dataset']}: {row['candidate']} - {row['baseline']} = "
            f"{row['auroc_difference']:.6f}, "
            f"95% CI [{row['ci_low']:.6f}, {row['ci_high']:.6f}], "
            f"p={row['two_sided_p']:.4f}"
        )


def _bootstrap_differences(
    truth: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    id_indices = np.flatnonzero(truth == 0)
    ood_indices = np.flatnonzero(truth == 1)
    differences = np.empty(bootstrap, dtype=np.float64)
    for index in range(bootstrap):
        selected = np.concatenate(
            [
                rng.choice(id_indices, len(id_indices), replace=True),
                rng.choice(ood_indices, len(ood_indices), replace=True),
            ]
        )
        local_truth = truth[selected]
        differences[index] = float(
            roc_auc_score(local_truth, candidate[selected])
            - roc_auc_score(local_truth, baseline[selected])
        )
    return differences


def _row(
    dataset: str,
    candidate: str,
    baseline: str,
    observed: float,
    differences: np.ndarray,
    bootstrap: int,
) -> dict[str, object]:
    lower = (float(np.sum(differences <= 0.0)) + 1.0) / (len(differences) + 1.0)
    upper = (float(np.sum(differences >= 0.0)) + 1.0) / (len(differences) + 1.0)
    return {
        "dataset": dataset,
        "candidate": DISPLAY[candidate],
        "baseline": DISPLAY[baseline],
        "auroc_difference": observed,
        "ci_low": float(np.quantile(differences, 0.025)),
        "ci_high": float(np.quantile(differences, 0.975)),
        "two_sided_p": min(1.0, 2.0 * min(lower, upper)),
        "bootstrap_samples": bootstrap,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
