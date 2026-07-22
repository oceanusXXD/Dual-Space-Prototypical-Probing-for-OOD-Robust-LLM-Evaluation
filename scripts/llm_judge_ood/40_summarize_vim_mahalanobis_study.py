#!/usr/bin/env python
"""Aggregate the cached-hiddenstate ViM/Mahalanobis study and plot rank curves."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageDraw
from scipy.special import logsumexp
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.scores.vim import FullViMScorer
from src.llm_judge_ood.shared.metrics import ood_metrics


DATASETS = ("ellipse", "asap", "clinc150", "rostd")
ESSAY_AUDIT = {
    "ellipse": {
        "prepared": "artifacts/llm_judge_ood_ellipse/ellipse_prepared_contract_v1.jsonl",
        "hidden": "hiddenstates/ellipse/qwen3_5_4b_judge_input_overall_v1.npz",
        "run": "artifacts/docs_experiments/ellipse/seed_42",
    },
    "asap": {
        "prepared": "artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl",
        "hidden": "hiddenstates/asap_aes/qwen3_5_4b_judge_input_asap_rubric_v1.npz",
        "run": "artifacts/docs_experiments/asap/seed_42",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-root",
        default="artifacts/docs_experiments/vim_mahalanobis_study_seed42",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.study_root)
    missing = [name for name in DATASETS if not (root / name / "manifest.json").exists()]
    if missing:
        raise RuntimeError(f"Incomplete detector study; missing manifests for {missing}")
    audit = {name: _deployed_head_audit(name, payload) for name, payload in ESSAY_AUDIT.items()}
    write_json(root / "full_vim_implementation_audit.json", audit)
    selected_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        manifest = json.loads((root / dataset / "manifest.json").read_text(encoding="utf-8"))
        selected_head = str(manifest["frozen_selection"]["head"])
        formal = _read_csv(root / dataset / "formal_ood_results.csv")
        for row in formal:
            if row["head"] == selected_head:
                selected_rows.append(
                    {
                        **row,
                        "auroc": float(row["auroc"]),
                        "aupr": float(row["aupr"]),
                        "fpr95": float(row["fpr95"]),
                    }
                )
        curve = _rank_curve_rows(root / dataset, manifest)
        _write_csv(root / dataset / "pseudo_rank_curve.csv", curve)
        _plot_rank_curve(root / dataset / "pseudo_rank_curve.png", dataset, curve)
    _write_csv(root / "formal_selected_head_all_datasets.csv", selected_rows)
    summary = _aggregate_rows(selected_rows)
    _write_csv(root / "formal_four_dataset_summary.csv", summary)
    significance = _cross_dataset_significance(selected_rows)
    write_json(root / "cross_dataset_significance.json", significance)
    write_json(
        root / "aggregate_manifest.json",
        {
            "artifact_type": "vim_mahalanobis_cached_hiddenstate_study_aggregate_v1",
            "datasets": list(DATASETS),
            "hiddenstate_forward_passes": 0,
            "implementation_audit": str(root / "full_vim_implementation_audit.json"),
            "formal_results": str(root / "formal_selected_head_all_datasets.csv"),
            "summary": str(root / "formal_four_dataset_summary.csv"),
            "significance": str(root / "cross_dataset_significance.json"),
            "rank_plots": {
                name: str(root / name / "pseudo_rank_curve.png") for name in DATASETS
            },
        },
    )


def _deployed_head_audit(name: str, paths: dict[str, str]) -> dict[str, Any]:
    rows = read_jsonl(paths["prepared"])
    cache = np.load(paths["hidden"], allow_pickle=True)
    raw = np.asarray(cache["features"], dtype=np.float64)
    run = Path(paths["run"])
    pre = np.load(run / "judge_preprocessor.npz", allow_pickle=False)
    processed = []
    for layer in range(raw.shape[1]):
        processed.append(
            (raw[:, layer] - pre["pca_means"][layer])
            @ pre["components"][layer].T
            / np.sqrt(np.maximum(pre["explained_variance"][layer], 1e-5))
        )
    features = np.stack(processed, axis=1).astype(np.float32)
    model = joblib.load(run / "judge_checkpoints" / "selected_linear_judge.joblib")
    queries = np.asarray([str(row["query_id"]) for row in rows])
    output = model.predict_output(features, queries)
    h = np.asarray(output.penultimate, dtype=np.float64)
    logits = np.asarray(output.logits, dtype=np.float64)
    split = np.asarray([str(row["split"]) for row in rows])
    truth = np.asarray([bool(row.get("is_document_ood", False)) for row in rows])
    train = split == "training_train"
    benchmark = split == "benchmark_test"
    rank = int(json.loads((run / "summary.json").read_text(encoding="utf-8"))["behavior_main_representation"]["vim_rank"])
    mean = h[train].mean(axis=0)
    _, _, right = np.linalg.svd(h[train] - mean, full_matrices=False)
    components = right[:rank].T
    train_residual = _residual(h[train] - mean, components)
    legacy_alpha = float(np.max(logits[train], axis=1).mean()) / float(train_residual.mean())
    legacy_score = legacy_alpha * _residual(h - mean, components) - logsumexp(logits, axis=1)
    weight, bias, head_queries = model.affine_head_parameters()
    corrected = FullViMScorer(rank=rank).fit(
        h[train],
        logits[train],
        head_weight=weight,
        head_bias=bias,
        query_ids=queries[train],
        head_query_ids=head_queries,
    )
    corrected_score = corrected.score(h, logits, queries)
    y = truth[benchmark].astype(int)
    stored = np.load(run / "judge_behavior_ood_scorer.npz", allow_pickle=False)
    stored_metadata = json.loads(str(stored["metadata_json"].item()))
    stored_full = next(
        row for row in stored_metadata["candidate_results"] if row["detector"] == "full_vim" and int(row["rank"]) == rank
    )
    origin = corrected.origins_[0]
    return {
        "dataset": name,
        "fit_rows": int(train.sum()),
        "benchmark_rows": int(benchmark.sum()),
        "rank": rank,
        "classes": output.classes.tolist(),
        "logit_shape": list(logits.shape),
        "row_alignment": bool(
            np.array_equal(np.asarray(cache["sample_ids"]).astype(str), np.asarray([row["sample_id"] for row in rows]))
        ),
        "softmax_max_row_sum_error": float(np.max(np.abs(output.probabilities.sum(axis=1) - 1.0))),
        "legacy_center": "source_feature_mean",
        "correct_center": "classifier_origin_-pinv(W.T)@b",
        "source_mean_norm": float(np.linalg.norm(mean)),
        "classifier_origin_norm": float(np.linalg.norm(origin)),
        "center_difference_norm": float(np.linalg.norm(mean - origin)),
        "legacy_alpha": legacy_alpha,
        "corrected_alpha": float(corrected.alpha_),
        "stored_pipeline_legacy_metrics": {
            "auroc": float(stored_full["benchmark_test_auroc"]),
            "aupr": float(stored_full["benchmark_test_aupr"]),
            "fpr95": float(stored_full["benchmark_test_fpr95"]),
        },
        "independent_legacy_recompute": ood_metrics(y, legacy_score[benchmark]),
        "classifier_origin_corrected": ood_metrics(y, corrected_score[benchmark]),
        "corrected_score_negated_auroc": float(ood_metrics(y, -corrected_score[benchmark])["auroc"]),
        "conclusion": "classifier-origin correction is valid but too small to explain the residual/full gap",
    }


def _residual(centered: np.ndarray, components: np.ndarray) -> np.ndarray:
    return np.linalg.norm(centered - centered @ components @ components.T, axis=1)


def _rank_curve_rows(dataset_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    frozen = manifest["frozen_selection"]
    rows = _read_csv(dataset_root / "pseudo_ood_vim_grid.csv")
    selected = [
        row
        for row in rows
        if row["head"] == frozen["head"] and row["layer"] == frozen["layer"]
    ]
    curves = {
        "raw residual": ("raw_residual", 0.0, 1.0),
        "L2-normalized residual": ("l2_normalized_residual", 0.0, 1.0),
        "whitened residual": ("whitened_residual", 0.0, 1.0),
        "standard Full ViM": ("origin_raw_standard_vim", 1.0, 1.0),
    }
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for curve, (variant, lam, temperature) in curves.items():
        for row in selected:
            if (
                row["variant"] != variant
                or float(row["lambda"]) != lam
                or float(row["temperature"]) != temperature
            ):
                continue
            key = (curve, int(row["rank"]))
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    "dataset": manifest["dataset"],
                    "curve": curve,
                    "rank": int(row["rank"]),
                    "rank_policy": row["rank_policy"],
                    "auroc": float(row["auroc"]),
                    "fpr95": float(row["fpr95"]),
                }
            )
    return sorted(output, key=lambda row: (row["curve"], row["rank"]))


def _plot_rank_curve(path: Path, dataset: str, rows: list[dict[str, Any]]) -> None:
    width, height = 1500, 570
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    colors = {
        "raw residual": "#2563eb",
        "L2-normalized residual": "#059669",
        "whitened residual": "#dc2626",
        "standard Full ViM": "#7c3aed",
    }
    panels = ((70, 75, 705, 470, "AUROC"), (800, 75, 1435, 470, "FPR95"))
    ranks = [int(row["rank"]) for row in rows]
    x_min, x_max = min(ranks), max(ranks)
    for left, top, right, bottom, metric in panels:
        draw.rectangle((left, top, right, bottom), outline="#111827", width=2)
        for tick in range(6):
            value = tick / 5.0
            y = int(bottom - value * (bottom - top))
            draw.line((left, y, right, y), fill="#e5e7eb", width=1)
            draw.text((left - 44, y - 7), f"{value:.1f}", fill="#374151")
        unique_ranks = sorted(set(ranks))
        label_indices = {
            int(round(index))
            for index in np.linspace(0, len(unique_ranks) - 1, min(6, len(unique_ranks)))
        }
        for index, rank in enumerate(unique_ranks):
            x = _plot_x(rank, x_min, x_max, left, right)
            draw.line((x, bottom, x, bottom + 5), fill="#111827", width=1)
            if index in label_indices:
                draw.text((x - 10, bottom + 8), str(rank), fill="#374151")
        draw.text((left, 28), f"{dataset}: source-only pseudo-OOD {metric}", fill="#111827")
        draw.text(((left + right) // 2 - 55, bottom + 28), "Retained PCA rank", fill="#374151")
        for curve in sorted(set(row["curve"] for row in rows)):
            local = sorted((row for row in rows if row["curve"] == curve), key=lambda row: row["rank"])
            points = [
                (
                    _plot_x(int(row["rank"]), x_min, x_max, left, right),
                    int(bottom - float(row[metric.lower()]) * (bottom - top)),
                )
                for row in local
            ]
            if len(points) > 1:
                draw.line(points, fill=colors[curve], width=4)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[curve])
    legend_x, legend_y = 70, 515
    for curve in colors:
        draw.line((legend_x, legend_y + 7, legend_x + 28, legend_y + 7), fill=colors[curve], width=4)
        draw.text((legend_x + 36, legend_y), curve, fill="#111827")
        legend_x += 320
    image.save(path)


def _plot_x(rank: int, minimum: int, maximum: int, left: int, right: int) -> int:
    if maximum == minimum:
        return (left + right) // 2
    return int(left + (rank - minimum) * (right - left) / (maximum - minimum))


def _aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods = sorted(set(row["method"] for row in rows))
    datasets = sorted(set(row["dataset"] for row in rows))
    by_key = {(row["dataset"], row["method"]): row for row in rows}
    ranks: dict[tuple[str, str], float] = {}
    for dataset in datasets:
        values = np.asarray([by_key[(dataset, method)]["auroc"] for method in methods])
        local_ranks = rankdata(-values, method="average")
        for method, rank in zip(methods, local_ranks.tolist()):
            ranks[(dataset, method)] = float(rank)
    output = []
    for method in methods:
        local = [by_key[(dataset, method)] for dataset in datasets]
        output.append(
            {
                "method": method,
                "mean_auroc": float(np.mean([row["auroc"] for row in local])),
                "mean_aupr": float(np.mean([row["aupr"] for row in local])),
                "mean_fpr95": float(np.mean([row["fpr95"] for row in local])),
                "mean_rank": float(np.mean([ranks[(dataset, method)] for dataset in datasets])),
                **{f"{dataset}_auroc": by_key[(dataset, method)]["auroc"] for dataset in datasets},
            }
        )
    return sorted(output, key=lambda row: (-row["mean_auroc"], row["mean_rank"]))


def _cross_dataset_significance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, float]] = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], {})[row["method"]] = float(row["auroc"])
    vim = np.asarray([by_dataset[name]["Selected adapted ViM"] for name in DATASETS])
    shrinkage = np.asarray([by_dataset[name]["Mahalanobis shrinkage"] for name in DATASETS])
    best_maha = np.asarray(
        [
            max(value for method, value in by_dataset[name].items() if method.startswith("Mahalanobis"))
            for name in DATASETS
        ]
    )
    return {
        "dataset_order": list(DATASETS),
        "selected_vim_auroc": vim.tolist(),
        "shrinkage_mahalanobis_auroc": shrinkage.tolist(),
        "best_strong_mahalanobis_auroc": best_maha.tolist(),
        "vim_minus_shrinkage": (vim - shrinkage).tolist(),
        "vim_minus_best_strong_mahalanobis": (vim - best_maha).tolist(),
        "wilcoxon_vim_vs_shrinkage_two_sided": float(wilcoxon(vim, shrinkage).pvalue),
        "wilcoxon_vim_vs_best_strong_mahalanobis_two_sided": float(wilcoxon(vim, best_maha).pvalue),
        "datasets_vim_not_worse_than_best_mahalanobis": int(np.sum(vim >= best_maha)),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
