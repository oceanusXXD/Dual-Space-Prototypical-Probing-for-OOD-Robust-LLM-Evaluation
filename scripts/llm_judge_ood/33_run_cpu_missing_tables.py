#!/usr/bin/env python
"""Fill remaining CPU-only experiment-table cells from frozen hidden caches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import scipy
import sklearn
import torch
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.adapt.head import HeadAdaptConfig, HeadAdapter
from src.llm_judge_ood.lifecycle.cluster import ClusterConfig, DocumentClusterer
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge
from src.llm_judge_ood.model.judge import JudgeTrainingConfig, SharedBackboneJudge
from src.llm_judge_ood.scores.openood import OpenOODPosthocScorer
from src.llm_judge_ood.scores.vim import ViMScorer
from src.llm_judge_ood.shared.metrics import judge_metrics, ood_metrics


DEFAULT_OUTPUT = Path("artifacts/llm_judge_ood_asap/cpu_missing_tables/acceptance")
MAIN_CACHE = Path("artifacts/llm_judge_ood_asap/qwen3_5_4b_input_document_masked_mean_v1.npz")
MAIN_INPUT = Path("artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl")
FORMAL_INPUTS = Path("artifacts/llm_judge_ood_asap/formal_acceptance_v1/inputs")
FORMAL_CACHES = Path("artifacts/llm_judge_ood_asap/formal_acceptance_v1/caches")
FORMAL_RESULTS = Path("artifacts/llm_judge_ood_asap/formal_acceptance_v5/results")
RUN_NAMES = tuple(
    f"abrupt_{shift}_seed_{seed}"
    for shift in ("near", "far")
    for seed in (42, 43, 44)
) + tuple(f"harmless_seed_{seed}" for seed in (42, 43, 44))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--bbds-trials", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cpu = _load_module("cpu_supplementary", ROOT / "scripts/llm_judge_ood/32_run_cpu_supplementary.py")
    formal = cpu._load_formal_module()
    runs = {
        name: formal._load_run(name, FORMAL_INPUTS, FORMAL_CACHES, FORMAL_RESULTS)
        for name in RUN_NAMES
    }
    base = runs["abrupt_near_seed_42"]
    main_rows, main_features = _load_main_cache()
    main_judge_features = _apply_frozen_preprocessor(base, main_features)
    main_queries = np.asarray([str(row["query_id"]) for row in main_rows])
    main_output = base["model"].predict_output(main_judge_features, main_queries)

    stages: dict[str, dict[str, Any]] = {}
    stages["judge"] = _cached_csv(
        output / "cpu12_benchmark_judge.csv", args.force,
        lambda: _judge_rows(base, main_rows, main_judge_features, main_output),
    )
    stages["detector"] = _cached_csv(
        output / "cpu13_benchmark_detector.csv", args.force,
        lambda: _detector_rows(base, main_rows, main_output, int(args.bootstrap_samples)),
    )
    stages["pca"] = _cached_csv(
        output / "cpu14_pca_preprocessing.csv", args.force,
        lambda: _pca_rows(base),
    )
    stages["heads"] = _cached_csv(
        output / "cpu15_head_extensions.csv", args.force,
        lambda: _head_rows(base),
    )
    stages["bbsd"] = _cached_csv(
        output / "cpu16_bbsd.csv", args.force,
        lambda: _bbsd_rows(base, trials=int(args.bbds_trials)),
    )
    stages["dbscan"] = _cached_csv(
        output / "cpu17_dbscan.csv", args.force,
        lambda: _dbscan_rows(runs, formal),
    )
    stages["probe"] = _cached_csv(
        output / "cpu18_probe_clusters.csv", args.force,
        lambda: _probe_cluster_rows(runs, cpu),
    )
    stages["adapt"] = _cached_csv(
        output / "cpu19_adapt_missing.csv", args.force,
        lambda: _adapt_rows(runs, cpu),
    )

    manifest = {
        "artifact_type": "llm_judge_ood_cpu_missing_tables_v1",
        "protocol": "shared_input_document_cache_cpu_completion",
        "formal_eligibility": False,
        "qwen_forward_passes": 0,
        "main_cache": str(MAIN_CACHE),
        "main_cache_sha256": _sha256(MAIN_CACHE),
        "main_input": str(MAIN_INPUT),
        "main_input_sha256": _sha256(MAIN_INPUT),
        "records": len(main_rows),
        "cache_feature_shape": list(main_features.shape),
        "cached_layers_resolved": [23, 32],
        "cached_pooling": "masked_mean",
        "a_b_cache_policy": "A and B both derived from the input-document cache as explicitly authorized",
        "stages": stages,
        "parameters": {
            "bootstrap_samples": int(args.bootstrap_samples),
            "bbsd_trials": int(args.bbds_trials),
            "seeds": [42, 43, 44],
        },
        "git_revision": _git("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git("status", "--porcelain")),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "elapsed_seconds": float(time.perf_counter() - started),
        "notes": [
            "This is a shared-cache CPU development completion, not rubric-aware Formal v4/v6.",
            "No Qwen forward pass or GPU operation is performed.",
            "Rows that require a new judge-input cache, a new episode, or an undefined protocol remain explicitly deferred.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_main_cache() -> tuple[list[dict[str, Any]], np.ndarray]:
    rows = read_jsonl(MAIN_INPUT)
    cache = np.load(MAIN_CACHE, allow_pickle=True)
    features = np.asarray(cache["features"], dtype=np.float32)
    sample_ids = np.asarray(cache["sample_ids"]).astype(str)
    row_ids = np.asarray([str(row["sample_id"]) for row in rows])
    if len(rows) != len(features) or not np.array_equal(sample_ids, row_ids):
        raise RuntimeError("Main cache and contract rows are not aligned")
    return rows, features


def _apply_frozen_preprocessor(run: dict[str, Any], features: np.ndarray) -> np.ndarray:
    payload = np.load(Path(run["result_dir"]) / "judge_preprocessor.npz", allow_pickle=False)
    layer = int(run["summary"]["feature_extractors"]["judge_input"].get("separability_selected_layer_index", 1))
    matrix = np.asarray(features[:, layer, :], dtype=np.float64)
    transformed = (
        (matrix - np.asarray(payload["pca_means"])[0])
        @ np.asarray(payload["components"])[0].T
        / np.sqrt(np.maximum(np.asarray(payload["explained_variance"])[0], 1e-5))
    )
    return transformed[:, None, :].astype(np.float32)


def _judge_rows(
    base: dict[str, Any], rows: list[dict[str, Any]], features: np.ndarray, output: Any,
) -> list[dict[str, Any]]:
    split = np.asarray([str(row["split"]) for row in rows])
    prompt = np.asarray([str(row.get("asap_prompt_id")) for row in rows])
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    queries = np.asarray([str(row["query_id"]) for row in rows])
    classes = np.asarray(output.classes)
    prediction = classes[np.argmax(output.probabilities, axis=1)]
    benchmark = split == "benchmark_test"
    train = split == "training_train"
    validation = split == "training_validation"
    majority_value = int(np.bincount(labels[train]).argmax())
    mean_value = int(np.clip(np.rint(labels[train].mean()), classes.min(), classes.max()))
    result = [
        _judge_metric_row("Majority", labels, np.full(len(labels), majority_value), None, benchmark, rows),
        _judge_metric_row("Mean score", labels, np.full(len(labels), mean_value), None, benchmark, rows),
        _judge_metric_row("Frozen LLM + Linear Head", labels, prediction, output.probabilities, benchmark, rows),
    ]
    mlp = SharedBackboneJudge(
        JudgeTrainingConfig(
            hidden_dim=96, output_dim=48, learning_rate=1e-3, weight_decay=1e-4,
            epochs=50, batch_size=256, patience=6, seed=42, device="cpu",
            loss="ce", class_values=tuple(classes.tolist()),
        )
    ).fit(features, labels, queries, train_mask=train, validation_mask=validation)
    mlp_output = mlp.predict_output(features, queries)
    mlp_prediction = classes[np.argmax(mlp_output.probabilities, axis=1)]
    result.append(_judge_metric_row(
        "Frozen LLM + MLP", labels, mlp_prediction, mlp_output.probabilities, benchmark, rows
    ))
    for prompt_id in (1, 2, 3, 4, 7, 8):
        mask = benchmark & (prompt == str(prompt_id))
        metric = _judge_metric_row(
            f"Prompt {prompt_id}", labels, prediction, output.probabilities, mask, rows
        )
        metric["experiment"] = "prompt"
        metric["prompt_id"] = prompt_id
        result.append(metric)
    return result


def _judge_metric_row(
    name: str, labels: np.ndarray, prediction: np.ndarray,
    probabilities: np.ndarray | None, mask: np.ndarray, rows: list[dict[str, Any]],
) -> dict[str, Any]:
    classes = np.asarray([1, 2, 3, 4, 5])
    metrics = judge_metrics(labels[mask], prediction[mask], class_values=classes)
    by_prompt = []
    for prompt_id in np.unique([str(rows[index].get("asap_prompt_id")) for index in np.flatnonzero(mask)]):
        local = mask & np.asarray([str(row.get("asap_prompt_id")) == prompt_id for row in rows])
        by_prompt.append(judge_metrics(labels[local], prediction[local], class_values=classes)["qwk"])
    result: dict[str, Any] = {
        "experiment": "judge_baseline", "method": name, "documents": int(mask.sum()),
        "qwk": metrics["qwk"], "macro_qwk": float(np.mean(by_prompt)),
        "mae": metrics["mae"], "accuracy": metrics["accuracy"],
        "macro_f1": float(f1_score(labels[mask], prediction[mask], labels=classes, average="macro", zero_division=0)),
    }
    if probabilities is not None:
        result["nll"] = float(log_loss(labels[mask], probabilities[mask], labels=classes))
        result["ece"] = _ece(labels[mask], prediction[mask], probabilities[mask])
    rater_pairs = [rows[index].get("rater_scores") for index in np.flatnonzero(mask)]
    valid = [(labels[index], pair) for index, pair in zip(np.flatnonzero(mask), rater_pairs, strict=True) if pair and len(pair) >= 2]
    if valid:
        truth = np.asarray([value for value, _ in valid])
        human = np.asarray([int(np.clip(np.rint(np.mean(pair[:2])), 1, 5)) for _, pair in valid])
        human_metrics = judge_metrics(truth, human, class_values=classes)
        result["human_qwk"] = human_metrics["qwk"]
        result["human_mae"] = human_metrics["mae"]
    return result


def _detector_rows(
    base: dict[str, Any], rows: list[dict[str, Any]], output: Any, bootstrap_samples: int,
) -> list[dict[str, Any]]:
    split = np.asarray([str(row["split"]) for row in rows])
    shift = np.asarray([str(row.get("document_shift_type", "id")) for row in rows])
    benchmark = split == "benchmark_test"
    labels = np.asarray([row["label"] for row in rows])
    queries = np.asarray([str(row["query_id"]) for row in rows])
    old_masks = _old_masks(base)
    old_h = np.asarray(base["output"].penultimate)
    old_logits = np.asarray(base["output"].logits)
    old_labels = np.asarray([row["label"] for row in base["rows"]])
    old_queries = np.asarray([str(row["query_id"]) for row in base["rows"]])
    h = np.asarray(output.penultimate)
    logits = np.asarray(output.logits)

    scorers: dict[str, Callable[[], np.ndarray]] = {
        "Residual-only ViM": lambda: base["vim"].score(h),
        "MaxLogit": lambda: OpenOODPosthocScorer("maxlogit").fit(
            old_h[old_masks["train"]], old_logits[old_masks["train"]],
            old_labels[old_masks["train"]], old_queries[old_masks["train"]],
        ).score(h, logits, queries),
        "kNN": lambda: OpenOODPosthocScorer("knn").fit(
            old_h[old_masks["train"]], old_logits[old_masks["train"]],
            old_labels[old_masks["train"]], old_queries[old_masks["train"]],
        ).score(h, logits, queries),
    }
    output_rows: list[dict[str, Any]] = []
    scores_by_method: dict[str, np.ndarray] = {}
    for method, build in scorers.items():
        try:
            scores = np.asarray(build(), dtype=np.float64)
        except ValueError:
            if method != "kNN":
                raise
            from src.llm_judge_ood.scores.knn import KNNScorer
            scorer = KNNScorer(k=28, metric="euclidean", normalize=True).fit(old_h[old_masks["train"]])
            scores = scorer.score(h)
        scores_by_method[method] = scores
        output_rows.append(_benchmark_ood_row(method, scores, benchmark, shift))
    for right in ("kNN",):
        output_rows.append(_paired_bootstrap_row(
            "Residual-only ViM", right, scores_by_method["Residual-only ViM"],
            scores_by_method[right], benchmark, shift, bootstrap_samples,
        ))
    return output_rows


def _benchmark_ood_row(method: str, scores: np.ndarray, benchmark: np.ndarray, shift: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {"experiment": "benchmark_detector", "method": method}
    for name, mask in (
        ("overall", benchmark),
        ("near", benchmark & np.isin(shift, ["id", "near"])),
        ("far", benchmark & np.isin(shift, ["id", "far"])),
    ):
        truth = (shift[mask] != "id").astype(int)
        metric = ood_metrics(truth, scores[mask])
        for key in ("auroc", "aupr", "fpr95"):
            result[f"{name}_{key}"] = metric[key]
    return result


def _paired_bootstrap_row(
    left_name: str, right_name: str, left: np.ndarray, right: np.ndarray,
    benchmark: np.ndarray, shift: np.ndarray, samples: int,
) -> dict[str, Any]:
    strata = [np.flatnonzero(benchmark & (shift == name)) for name in ("id", "near", "far")]
    rng = np.random.default_rng(20260720)
    diffs = []
    for _ in range(samples):
        selected = [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        indices = np.concatenate(selected)
        truth = (shift[indices] != "id").astype(int)
        diffs.append(roc_auc_score(truth, left[indices]) - roc_auc_score(truth, right[indices]))
    indices = np.flatnonzero(benchmark)
    truth = (shift[indices] != "id").astype(int)
    left_metrics = ood_metrics(truth, left[indices])
    right_metrics = ood_metrics(truth, right[indices])
    return {
        "experiment": "paired_bootstrap", "method": left_name, "comparison": right_name,
        "auroc_difference": left_metrics["auroc"] - right_metrics["auroc"],
        "auroc_ci95": np.quantile(diffs, [0.025, 0.975]).tolist(),
        "aupr_difference": left_metrics["aupr"] - right_metrics["aupr"],
        "fpr95_difference": left_metrics["fpr95"] - right_metrics["fpr95"],
        "bootstrap_samples": samples,
    }


def _pca_rows(base: dict[str, Any]) -> list[dict[str, Any]]:
    raw = np.asarray(base["features"][:, 1, :], dtype=np.float32)
    masks = _old_masks(base)
    output: list[dict[str, Any]] = []
    for label, n_components, whiten in (
        ("64", 64, True), ("128", 128, True), ("256", 256, True), ("512", 512, True),
        ("90% variance", 0.90, True), ("95% variance", 0.95, True),
    ):
        pca = PCA(n_components=n_components, whiten=whiten, svd_solver="full", random_state=42).fit(raw[masks["train"]])
        values = pca.transform(raw).astype(np.float32)[:, None, :]
        row = _fit_linear_variant(base, values, seed=42)
        row.update({
            "experiment": "pca_dimension", "variant": label,
            "output_dimension": int(values.shape[-1]),
            "retained_variance": float(np.sum(pca.explained_variance_ratio_)),
            "preprocess": "PCA + Whitening",
        })
        output.append(row)
    centered = raw - raw[masks["train"]].mean(axis=0)
    scale = np.maximum(raw[masks["train"]].std(axis=0), 1e-5)
    pca_plain = PCA(n_components=226, whiten=False, svd_solver="full", random_state=42).fit(raw[masks["train"]])
    pca_white = PCA(n_components=226, whiten=True, svd_solver="full", random_state=42).fit(raw[masks["train"]])
    variants = {
        "无预处理": raw,
        "标准化": centered / scale,
        "PCA": pca_plain.transform(raw),
        "PCA + Whitening": pca_white.transform(raw),
    }
    for label, values in variants.items():
        row = _fit_linear_variant(base, np.asarray(values, dtype=np.float32)[:, None, :], seed=42)
        row.update({"experiment": "preprocessing", "variant": label})
        output.append(row)
    return output


def _fit_linear_variant(base: dict[str, Any], features: np.ndarray, *, seed: int) -> dict[str, Any]:
    masks = _old_masks(base)
    labels = np.asarray([row["label"] for row in base["rows"]])
    queries = np.asarray([str(row["query_id"]) for row in base["rows"]])
    judge = PerQueryLinearJudge(
        LinearJudgeConfig(
            method="linear", representation="last_layer", pca_dim=features.shape[-1],
            class_values=(1, 2, 3, 4, 5), seed=seed, learning_rate=1e-3,
            weight_decay=1e-4, epochs=50, batch_size=256, patience=6,
            device="cpu", class_weight="balanced", head_sharing="shared",
        )
    ).fit(features, labels, queries, train_mask=masks["train"], validation_mask=masks["validation"])
    result = judge.predict_output(features, queries)
    prediction = result.classes[np.argmax(result.probabilities, axis=1)]
    id_metrics = judge_metrics(labels[masks["test"]], prediction[masks["test"]], class_values=result.classes)
    vim = ViMScorer(rank=min(64, result.penultimate.shape[1] - 1)).fit(result.penultimate[masks["train"]])
    score = vim.score(result.penultimate)
    truth = np.concatenate([np.zeros(masks["calibration"].sum()), np.ones(masks["development"].sum())])
    detector_score = np.concatenate([score[masks["calibration"]], score[masks["development"]]])
    ood = ood_metrics(truth.astype(int), detector_score)
    return {"id_qwk": id_metrics["qwk"], "id_mae": id_metrics["mae"], "vim_auroc": ood["auroc"], "vim_fpr95": ood["fpr95"]}


def _head_rows(base: dict[str, Any]) -> list[dict[str, Any]]:
    masks = _old_masks(base)
    labels = np.asarray([row["label"] for row in base["rows"]])
    queries = np.asarray([str(row["query_id"]) for row in base["rows"]])
    features = np.asarray(base["judge_features"], dtype=np.float32)
    output: list[dict[str, Any]] = []
    for seed in (42, 43, 44):
        linear = _fit_linear_variant(base, features, seed=seed)
        linear.update({"experiment": "head_seed", "head": "Linear", "seed": seed})
        output.append(linear)
        mlp = SharedBackboneJudge(
            JudgeTrainingConfig(
                hidden_dim=96, output_dim=48, learning_rate=1e-3, weight_decay=1e-4,
                epochs=50, batch_size=256, patience=6, seed=seed, device="cpu",
                loss="ce", class_values=(1, 2, 3, 4, 5),
            )
        ).fit(features, labels, queries, train_mask=masks["train"], validation_mask=masks["validation"])
        mlp_output = mlp.predict_output(features, queries)
        row = _head_output_metrics(base, mlp_output)
        row.update({"experiment": "head_seed", "head": "MLP", "seed": seed})
        output.append(row)
    for name, hidden_dim, output_dim in (("MLP 1-layer 128", 128, 128), ("MLP 2-layer 128/64", 128, 64)):
        model = SharedBackboneJudge(
            JudgeTrainingConfig(
                hidden_dim=hidden_dim, output_dim=output_dim, learning_rate=1e-3,
                weight_decay=1e-4, epochs=50, batch_size=256, patience=6,
                seed=42, device="cpu", loss="ce", class_values=(1, 2, 3, 4, 5),
            )
        ).fit(features, labels, queries, train_mask=masks["train"], validation_mask=masks["validation"])
        result = model.predict_output(features, queries)
        row = _head_output_metrics(base, result)
        row.update({
            "experiment": "head_architecture", "head": name,
            "hidden_dim": hidden_dim, "output_dim": output_dim,
            "parameters": sum(value.numel() for value in model.backbone.parameters()) + sum(value.numel() for value in model.heads.parameters()),
        })
        output.append(row)
    majority = int(np.bincount(labels[masks["train"]].astype(int)).argmax())
    prediction = np.full(len(labels), majority)
    metrics = judge_metrics(labels[masks["test"]], prediction[masks["test"]], class_values=np.asarray([1, 2, 3, 4, 5]))
    output.append({
        "experiment": "head_architecture", "head": "Majority baseline", "parameters": 0,
        "id_qwk": metrics["qwk"], "id_mae": metrics["mae"], "accuracy": metrics["accuracy"],
        "macro_f1": f1_score(labels[masks["test"]], prediction[masks["test"]], labels=[1, 2, 3, 4, 5], average="macro", zero_division=0),
    })
    return output


def _head_output_metrics(base: dict[str, Any], result: Any) -> dict[str, Any]:
    masks = _old_masks(base)
    labels = np.asarray([row["label"] for row in base["rows"]])
    prediction = result.classes[np.argmax(result.probabilities, axis=1)]
    id_metric = judge_metrics(labels[masks["test"]], prediction[masks["test"]], class_values=result.classes)
    near_metric = judge_metrics(labels[masks["near"]], prediction[masks["near"]], class_values=result.classes)
    far_metric = judge_metrics(labels[masks["far"]], prediction[masks["far"]], class_values=result.classes)
    vim = ViMScorer(rank=min(64, result.penultimate.shape[1] - 1)).fit(result.penultimate[masks["train"]])
    score = vim.score(result.penultimate)
    truth = np.concatenate([np.zeros(masks["calibration"].sum()), np.ones(masks["development"].sum())]).astype(int)
    detector_score = np.concatenate([score[masks["calibration"]], score[masks["development"]]])
    return {
        "id_qwk": id_metric["qwk"], "id_mae": id_metric["mae"], "accuracy": id_metric["accuracy"],
        "macro_f1": f1_score(labels[masks["test"]], prediction[masks["test"]], labels=result.classes, average="macro", zero_division=0),
        "nll": log_loss(labels[masks["test"]], result.probabilities[masks["test"]], labels=result.classes),
        "ece": _ece(labels[masks["test"]], prediction[masks["test"]], result.probabilities[masks["test"]]),
        "near_qwk": near_metric["qwk"], "far_qwk": far_metric["qwk"],
        "ood_auroc": roc_auc_score(truth, detector_score),
    }


def _bbsd_rows(base: dict[str, Any], *, trials: int) -> list[dict[str, Any]]:
    masks = _old_masks(base)
    probabilities = np.asarray(base["output"].probabilities)
    predictions = np.argmax(probabilities, axis=1)
    rng = np.random.default_rng(20260720)
    calibration = np.flatnonzero(masks["calibration"])
    near = np.flatnonzero(masks["near"])
    far = np.flatnonzero(masks["far"])
    output: list[dict[str, Any]] = []
    for method in ("BBSDs", "BBSDh"):
        row: dict[str, Any] = {"method": method, "trials": trials, "window_documents": 50}
        for scenario, target in (("h0", calibration), ("near", near), ("far", far)):
            reject_005 = reject_001 = 0
            pvalues = []
            for _ in range(trials):
                ref = rng.choice(calibration, size=50, replace=False)
                if scenario == "h0":
                    pool = np.setdiff1d(calibration, ref)
                    sample = rng.choice(pool, size=50, replace=False)
                else:
                    shifted = rng.choice(target, size=10, replace=False)
                    clean = rng.choice(calibration, size=40, replace=False)
                    sample = np.concatenate([clean, shifted])
                p = _bbsd_pvalue(method, probabilities, predictions, ref, sample)
                pvalues.append(p)
                reject_005 += int(p <= 0.05)
                reject_001 += int(p <= 0.01)
            row[f"{scenario}_rate_005"] = reject_005 / trials
            row[f"{scenario}_rate_001"] = reject_001 / trials
            row[f"{scenario}_median_p"] = float(np.median(pvalues))
        output.append(row)
    return output


def _bbsd_pvalue(method: str, probabilities: np.ndarray, predictions: np.ndarray, ref: np.ndarray, sample: np.ndarray) -> float:
    if method == "BBSDs":
        pvalues = [ks_2samp(probabilities[ref, column], probabilities[sample, column]).pvalue for column in range(probabilities.shape[1])]
        return float(min(1.0, min(pvalues) * probabilities.shape[1]))
    classes = np.arange(probabilities.shape[1])
    table = np.stack([
        np.bincount(predictions[ref], minlength=len(classes)),
        np.bincount(predictions[sample], minlength=len(classes)),
    ])
    keep = table.sum(axis=0) > 0
    if keep.sum() < 2:
        return 1.0
    return float(chi2_contingency(table[:, keep], correction=False).pvalue)


def _dbscan_rows(runs: dict[str, dict[str, Any]], formal: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run_name, run in sorted(runs.items()):
        if run_name.startswith("harmless"):
            continue
        candidates = formal._persistent_candidates(run)
        values = np.asarray(run["b"])[candidates]
        values = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
        truth_values = np.asarray([str(run["rows"][index]["audit_document_group_id"]) for index in candidates])
        _, truth = np.unique(truth_values, return_inverse=True)
        for eps in (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0):
            for min_samples in (3, 4, 5, 10):
                labels, _ = DocumentClusterer(ClusterConfig(
                    method="dbscan", dbscan_eps=eps, dbscan_min_samples=min_samples,
                    min_cluster_size=min_samples,
                )).fit_predict(values)
                precision, recall = formal._routing_metrics(run, candidates, labels, embeddings=np.asarray(run["b"]))
                output.append({
                    "run": run_name, "scenario": "near" if "near" in run_name else "far",
                    "eps": eps, "min_samples": min_samples,
                    "precision": precision, "recall": recall,
                    "purity": formal._purity(truth, labels),
                    "nmi": sklearn.metrics.normalized_mutual_info_score(truth, labels),
                    "ari": sklearn.metrics.adjusted_rand_score(truth, labels),
                    "noise_rate": float(np.mean(labels < 0)),
                    "clusters": len(set(labels[labels >= 0].tolist())),
                })
    return output


def _probe_cluster_rows(runs: dict[str, dict[str, Any]], cpu: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run_name in ("abrupt_near_seed_42", "abrupt_far_seed_42", "harmless_seed_42"):
        run = runs[run_name]
        by_cluster = run["summary"]["probe"]["by_predicted_document_cluster"]
        for lifecycle in run["lifecycle"]:
            cluster = str(lifecycle["document_cluster_id"])
            if cluster not in by_cluster:
                continue
            members = np.asarray(lifecycle.get("member_indices", []), dtype=int)
            full = cpu._probe_result_local(run, members, seed=42, n_boot=1000)
            selected = cpu._probe_sample_local(
                run, members, strategy="random", count=min(20, len(members)),
                rng=np.random.default_rng(cpu._stable_seed(f"missing-table:{run_name}:{cluster}")),
            )
            probe = cpu._probe_result_local(run, selected, seed=42, n_boot=1000)
            historical = by_cluster[cluster]
            output.append({
                "run": run_name, "scenario": "near" if "near" in run_name else "far" if "far" in run_name else "harmless",
                "cluster": cluster, "cluster_size": len(members), "probe_documents": len(selected),
                "full_delta": full.get("harm_delta"), "probe_delta": probe.get("harm_delta"),
                "probe_lcb": probe.get("harm_delta_lcb"), "probe_ucb": probe.get("harm_delta_ucb"),
                "p_value": probe.get("harmfulness_p_value"),
                "fdr_adjusted_p_value": historical.get("harmfulness_fdr_adjusted_p_value"),
                "fdr_rejected": historical.get("harmfulness_fdr_rejected"),
                "full_status": cpu._three_way_probe_status(full, benign_margin=0.1),
                "probe_status": cpu._three_way_probe_status(probe, benign_margin=0.1),
            })
    return output


def _adapt_rows(runs: dict[str, dict[str, Any]], cpu: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run_name, run in sorted(runs.items()):
        if run_name.startswith("harmless"):
            continue
        harmful = set(run["summary"]["probe"]["harmful_predicted_document_cluster_ids"])
        if not harmful:
            continue
        labels = np.asarray([row["label"] for row in run["rows"]])
        queries = np.asarray([str(row["query_id"]) for row in run["rows"]])
        split = np.asarray([str(row["split"]) for row in run["rows"]])
        old = np.asarray([score["judge_prediction"] for score in run["scores"]])
        cluster = np.asarray([str(score.get("predicted_document_cluster_id", "")) for score in run["scores"]])
        pool = np.asarray(sorted({
            int(index) for lifecycle in run["lifecycle"]
            if str(lifecycle.get("document_cluster_id")) in harmful
            for index in lifecycle.get("member_indices", [])
        }), dtype=int)
        future = np.flatnonzero((split == "deployment_future_test") & np.isin(cluster, list(harmful)))
        gate = np.flatnonzero((split == "deployment_gate") & np.isin(cluster, list(harmful)))
        guard = np.flatnonzero(split == "training_guard")
        if not len(pool) or not len(future) or not len(gate):
            continue
        u = run["model"].transform_u(run["judge_features"])
        metadata = run["summary"]["adaptation"].get("adapter") or {}
        base = HeadAdaptConfig(**(metadata.get("config") or {}))
        for candidate, config in (
            ("naive_head_ft", replace(base, training_replay_weight=0.0, anchor_weight=0.0)),
            ("lr_0.003", replace(base, learning_rate=3e-3)),
        ):
            rng = np.random.default_rng(cpu._stable_seed(f"missing-adapt:{run_name}:{candidate}"))
            target = rng.choice(pool, size=min(20, len(pool)), replace=False)
            adapter = HeadAdapter(config).fit(
                u_features=u, labels=labels, query_ids=queries,
                deployment_indices=target, training_replay_indices=np.asarray([], dtype=int),
                class_values=run["model"].classes_, judge=run["model"],
            ) if candidate == "naive_head_ft" else HeadAdapter(config).fit(
                u_features=u, labels=labels, query_ids=queries,
                deployment_indices=target,
                training_replay_indices=np.flatnonzero(split == "training_train")[:len(target)],
                class_values=run["model"].classes_, judge=run["model"],
            )
            prediction = adapter.predict(u_features=u, query_ids=queries, fallback=old)
            before = judge_metrics(labels[future], old[future], class_values=run["model"].classes_)
            after = judge_metrics(labels[future], prediction[future], class_values=run["model"].classes_)
            source_before = judge_metrics(labels[guard], old[guard], class_values=run["model"].classes_)
            source_after = judge_metrics(labels[guard], prediction[guard], class_values=run["model"].classes_)
            nfr = float(np.mean((prediction[guard] != labels[guard]) & (old[guard] == labels[guard])))
            gate_gain = np.abs(old[gate].astype(float) - labels[gate]) - np.abs(prediction[gate].astype(float) - labels[gate])
            output.append({
                "run": run_name, "scenario": "near" if "near" in run_name else "far",
                "candidate": candidate, "learning_rate": config.learning_rate,
                "epochs": config.epochs, "replay_weight": config.training_replay_weight,
                "anchor_weight": config.anchor_weight, "target_documents": len(target),
                "future_mae_before": before["mae"], "future_mae_after": after["mae"],
                "future_mae_gain": before["mae"] - after["mae"],
                "future_qwk_before": before["qwk"], "future_qwk_after": after["qwk"],
                "future_qwk_gain": after["qwk"] - before["qwk"],
                "source_nfr": nfr, "source_qwk_drop": source_before["qwk"] - source_after["qwk"],
                "gate_gain": float(np.mean(gate_gain)),
            })
    return output


def _old_masks(run: dict[str, Any]) -> dict[str, np.ndarray]:
    split = np.asarray([str(row["split"]) for row in run["rows"]])
    shift = np.asarray([str(row.get("document_shift_type", "id")) for row in run["rows"]])
    return {
        "train": split == "training_train", "validation": split == "training_validation",
        "calibration": split == "training_calibration", "test": split == "training_test",
        "development": split == "development",
        "near": (split == "development") & (shift == "near"),
        "far": (split == "development") & (shift == "far"),
    }


def _ece(labels: np.ndarray, prediction: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    confidence = np.max(probabilities, axis=1)
    correct = prediction == labels
    total = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return total


def _cached_csv(path: Path, force: bool, builder: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
    started = time.perf_counter()
    if path.exists() and not force:
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        return {"path": str(path), "rows": len(rows), "reused": True, "elapsed_seconds": 0.0}
    rows = builder()
    _write_csv(path, rows)
    return {"path": str(path), "rows": len(rows), "reused": False, "elapsed_seconds": time.perf_counter() - started}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


if __name__ == "__main__":
    main()
