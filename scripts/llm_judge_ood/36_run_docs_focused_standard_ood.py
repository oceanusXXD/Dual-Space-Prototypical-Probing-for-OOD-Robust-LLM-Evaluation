#!/usr/bin/env python
"""Run docs Table 2-4 focused experiments for standard text OOD datasets.

This runner consumes prepared JSONL contracts and frozen hidden-state caches.
It performs no model forward passes and writes only experiment artifacts under
``artifacts/docs_experiments``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy
import sklearn
import torch
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.lifecycle.drift import (
    BlockAwareC2ST,
    MMDPermutationTest,
    ScalarKSTest,
    WindowDriftConfig,
)
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge
from src.llm_judge_ood.scores.knn import KNNScorer
from src.llm_judge_ood.scores.openood import OpenOODPosthocScorer
from src.llm_judge_ood.scores.rmd import RMDScorer
from src.llm_judge_ood.scores.vim import FullViMScorer, ViMScorer
from src.llm_judge_ood.shared.metrics import judge_metrics, ood_metrics


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    prepared_path: Path
    document_hidden_path: Path
    judge_hidden_path: Path
    query_id: str
    train_split: str
    calibration_split: str
    id_test_split: str
    ood_test_split: str
    notes: str


DATASETS: dict[str, DatasetSpec] = {
    "clinc150": DatasetSpec(
        name="clinc150",
        prepared_path=Path("artifacts/llm_judge_ood_clinc150/clinc150_prepared_contract_v1.jsonl"),
        document_hidden_path=Path("hiddenstates/clinc150/qwen3_5_4b_input_document_masked_mean_v1.npz"),
        judge_hidden_path=Path("hiddenstates/clinc150/qwen3_5_4b_judge_input_intent_v1.npz"),
        query_id="clinc150_intent",
        train_split="train",
        calibration_split="val",
        id_test_split="test",
        ood_test_split="oos_test",
        notes="official in-scope / OOS splits; oos_train and oos_val are excluded",
    ),
    "rostd": DatasetSpec(
        name="rostd",
        prepared_path=Path("artifacts/llm_judge_ood_rostd/rostd_prepared_contract_v1.jsonl"),
        document_hidden_path=Path("hiddenstates/rostd/qwen3_5_4b_input_document_masked_mean_v1.npz"),
        judge_hidden_path=Path("hiddenstates/rostd/qwen3_5_4b_judge_input_intent_v1.npz"),
        query_id="rostd_supported_intent",
        train_split="train",
        calibration_split="eval",
        id_test_split="test",
        ood_test_split="test",
        notes="official train ID, eval ID calibration, test ID/OOD benchmark; eval OOD is excluded",
    ),
}


METHOD_ORDER = (
    "MSP",
    "MaxLogit",
    "Energy",
    "Mahalanobis",
    "RMD",
    "kNN",
    "Full ViM",
    "Residual-only ViM",
)

WINDOW_METHOD_ORDER = (
    "A-MMD",
    "B-MMD",
    "C2ST",
    "KS",
    "BBSDs",
    "BBSDh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument("--output-root", default="artifacts/docs_experiments")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--classifier-pca-dim", type=int, default=128)
    parser.add_argument("--a-pca-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-trials", type=int, default=40)
    parser.add_argument("--mmd-permutations", type=int, default=99)
    parser.add_argument("--window-horizon", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = DATASETS[str(args.dataset)]
    dataset_root = Path(args.output_root) / spec.name
    dataset_root.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(spec.prepared_path)
    document_cache = _load_cache(spec.document_hidden_path, rows, "document")
    judge_cache = _load_cache(spec.judge_hidden_path, rows, "judge")
    masks = _dataset_masks(spec, rows)
    _validate_masks(masks)
    seed_summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        seed_dir = dataset_root / f"seed_{int(seed)}_focused"
        summary_path = seed_dir / "summary.json"
        if summary_path.exists() and not args.force:
            seed_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        seed_summary = _run_seed(
            args=args,
            spec=spec,
            rows=rows,
            document_features=document_cache["features"],
            judge_features=judge_cache["features"],
            masks=masks,
            seed=int(seed),
            output_dir=seed_dir,
        )
        seed_summaries.append(seed_summary)
    aggregate = _write_aggregate_tables(
        dataset_root=dataset_root,
        spec=spec,
        seed_summaries=seed_summaries,
        args=args,
        rows=rows,
        masks=masks,
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


def _run_seed(
    *,
    args: argparse.Namespace,
    spec: DatasetSpec,
    rows: list[dict[str, Any]],
    document_features: np.ndarray,
    judge_features: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Standard OOD datasets include held-out OOS rows whose label is a text
    # token such as "oos".  Keep the classification vocabulary string-typed so
    # source class IDs and benchmark labels stay aligned inside the shared
    # PerQueryLinearJudge normalization path.
    labels = np.asarray([row["label"] for row in rows]).astype(str)
    query_ids = np.asarray([str(row["query_id"]) for row in rows])
    class_values = np.unique(labels[masks["train"]])
    judge = PerQueryLinearJudge(
        LinearJudgeConfig(
            method="linear",
            representation="pca",
            pca_dim=int(args.classifier_pca_dim),
            class_values=tuple(class_values.tolist()),
            seed=int(seed),
            learning_rate=float(args.learning_rate),
            weight_decay=1e-4,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            patience=6,
            device="cpu",
            class_weight="balanced",
            head_sharing="shared",
        )
    ).fit(
        judge_features,
        labels,
        query_ids,
        train_mask=masks["train"],
        validation_mask=masks["calibration"],
    )
    judge_output = judge.predict_output(judge_features, query_ids)
    head_weight, head_bias, head_query_ids = judge.affine_head_parameters()
    predictions = judge_output.classes[np.argmax(judge_output.probabilities, axis=1)]
    table2 = [_classifier_row(spec, labels, predictions, judge_output.probabilities, masks, seed)]
    scores = _sample_ood_scores(
        rows=rows,
        labels=labels,
        query_ids=query_ids,
        masks=masks,
        output=judge_output,
        head_weight=head_weight,
        head_bias=head_bias,
        head_query_ids=head_query_ids,
        seed=seed,
    )
    table3 = _sample_ood_rows(scores, masks, seed)
    a_space = _source_pca_space(
        document_features,
        fit_mask=masks["train"],
        pca_dim=int(args.a_pca_dim),
        seed=seed,
    )
    residual_scorer = ViMScorer(rank=_vim_rank(judge_output.penultimate)).fit(judge_output.penultimate[masks["train"]])
    b_space = residual_scorer.residual_features(judge_output.penultimate)
    table4 = _window_rows(
        args=args,
        masks=masks,
        a_space=a_space,
        b_space=b_space,
        residual_scores=residual_scorer.score(judge_output.penultimate),
        probabilities=judge_output.probabilities,
        seed=seed,
    )
    _write_csv(output_dir / "table2_classifier_id_performance.csv", table2)
    _write_csv(output_dir / "table3_sample_ood_detection.csv", table3)
    _write_csv(output_dir / "table4_window_shift_detection.csv", table4)
    summary = {
        "artifact_type": "docs_focused_standard_ood_seed_v1",
        "dataset": spec.name,
        "seed": int(seed),
        "tables": {
            "table2": str(output_dir / "table2_classifier_id_performance.csv"),
            "table3": str(output_dir / "table3_sample_ood_detection.csv"),
            "table4": str(output_dir / "table4_window_shift_detection.csv"),
        },
        "split_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "classifier": judge.to_metadata(),
        "hiddenstate": {
            "prepared_path": str(spec.prepared_path),
            "prepared_sha256": _sha256(spec.prepared_path),
            "document_hidden_path": str(spec.document_hidden_path),
            "document_hidden_sha256": _sha256(spec.document_hidden_path),
            "judge_hidden_path": str(spec.judge_hidden_path),
            "judge_hidden_sha256": _sha256(spec.judge_hidden_path),
            "features_shape": list(judge_features.shape),
            "alignment": "sample_ids exactly match prepared sample_id order",
        },
        "protocol": {
            "train_scope": spec.train_split,
            "calibration_scope": spec.calibration_split,
            "id_test_scope": spec.id_test_split,
            "ood_test_scope": spec.ood_test_split,
            "classifier_representation": f"source-fitted PCA({int(args.classifier_pca_dim)}) over judge-input cached layers",
            "a_space": f"source-fitted PCA({int(args.a_pca_dim)}) over input-document cached layers",
            "b_space": "source-fitted residual-only ViM residual vectors over classifier penultimate space",
            "window_size": int(args.window_size),
            "window_trials": int(args.window_trials),
            "mmd_permutations": int(args.mmd_permutations),
            "window_horizon": int(args.window_horizon),
            "uses_ood_validation": False,
            "qwen_forward_passes": 0,
            "notes": spec.notes,
        },
        "table2": table2,
        "table3": table3,
        "table4": table4,
        "git_revision": _git("rev-parse", "HEAD"),
        "git_worktree_dirty": bool(_git("status", "--porcelain")),
        "command": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def _load_cache(path: Path, rows: list[dict[str, Any]], role: str) -> dict[str, np.ndarray]:
    cache = np.load(path, allow_pickle=True)
    features = np.asarray(cache["features"], dtype=np.float32)
    sample_ids = np.asarray(cache["sample_ids"]).astype(str)
    expected = np.asarray([str(row["sample_id"]) for row in rows])
    if len(features) != len(rows) or not np.array_equal(sample_ids, expected):
        raise RuntimeError(f"{role} hidden cache is not aligned with prepared rows: {path}")
    labels = np.asarray(cache["labels"])
    expected_labels = np.asarray([row["label"] for row in rows])
    if not np.array_equal(labels.astype(str), expected_labels.astype(str)):
        raise RuntimeError(f"{role} hidden cache labels do not match prepared rows: {path}")
    if features.ndim != 3 or features.shape[1] < 1 or features.shape[2] < 2:
        raise RuntimeError(f"{role} hidden cache has invalid feature shape: {features.shape}")
    return {"features": features, "sample_ids": sample_ids, "labels": labels}


def _dataset_masks(spec: DatasetSpec, rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    split = np.asarray([str(row["split"]) for row in rows])
    is_ood = np.asarray([bool(row.get("is_document_ood", row.get("is_ood", False))) for row in rows])
    if spec.name == "rostd":
        train = (split == "train") & ~is_ood
        calibration = (split == "eval") & ~is_ood
        id_test = (split == "test") & ~is_ood
        ood_test = (split == "test") & is_ood
        excluded = ((split == "eval") & is_ood)
    else:
        train = (split == "train") & ~is_ood
        calibration = (split == "val") & ~is_ood
        id_test = (split == "test") & ~is_ood
        ood_test = (split == "oos_test") & is_ood
        excluded = is_ood & ~ood_test
    benchmark = id_test | ood_test
    return {
        "train": train,
        "calibration": calibration,
        "id_test": id_test,
        "ood_test": ood_test,
        "benchmark": benchmark,
        "excluded": excluded,
    }


def _validate_masks(masks: dict[str, np.ndarray]) -> None:
    required = ("train", "calibration", "id_test", "ood_test")
    missing = [name for name in required if not bool(masks[name].any())]
    if missing:
        raise RuntimeError(f"Empty required split mask(s): {missing}")
    if np.any(masks["train"] & masks["calibration"]) or np.any(masks["train"] & masks["benchmark"]):
        raise RuntimeError("Train/calibration/benchmark masks must be disjoint")
    if np.any(masks["calibration"] & masks["benchmark"]):
        raise RuntimeError("Calibration and benchmark masks must be disjoint")


def _classifier_row(
    spec: DatasetSpec,
    labels: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    mask = masks["id_test"]
    classes = np.unique(labels[masks["train"]])
    metrics = judge_metrics(
        labels[mask],
        predictions[mask],
        probabilities=probabilities[mask],
        class_values=classes,
    )
    return {
        "dataset": spec.name,
        "method": "Ours Frozen LLM + Linear Head",
        "seed": int(seed),
        "train_labels": int(masks["train"].sum()),
        "id_test_records": int(mask.sum()),
        "qwk": "",
        "mae": "",
        "spearman": "",
        "accuracy": metrics["accuracy"],
        "macro_f1": float(
            f1_score(labels[mask], predictions[mask], labels=classes, average="macro", zero_division=0)
        ),
        "mean_confidence": metrics.get("mean_confidence", float("nan")),
        "notes": "in-scope benchmark only" if spec.name == "clinc150" else "supported-intent benchmark only",
    }


def _sample_ood_scores(
    *,
    rows: list[dict[str, Any]],
    labels: np.ndarray,
    query_ids: np.ndarray,
    masks: dict[str, np.ndarray],
    output: Any,
    head_weight: np.ndarray,
    head_bias: np.ndarray,
    head_query_ids: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    h = np.asarray(output.penultimate, dtype=np.float64)
    logits = np.asarray(output.logits, dtype=np.float64)
    probs = np.asarray(output.probabilities, dtype=np.float64)
    train = masks["train"]
    rank = _vim_rank(h)
    vim = ViMScorer(rank=rank).fit(h[train])
    full_vim = FullViMScorer(rank=rank).fit(
        h[train],
        logits[train],
        head_weight=head_weight,
        head_bias=head_bias,
        query_ids=query_ids[train],
        head_query_ids=head_query_ids,
    )
    mahal = OpenOODPosthocScorer("mahalanobis").fit(
        h[train],
        logits[train],
        labels[train],
        query_ids[train],
    )
    rmd = RMDScorer().fit(h[train], labels[train])
    knn = KNNScorer(k=min(10, int(train.sum())), metric="euclidean", normalize=True).fit(h[train])
    energy = OpenOODPosthocScorer("energy").fit(h[train], logits[train], labels[train], query_ids[train])
    maxlogit = OpenOODPosthocScorer("maxlogit").fit(h[train], logits[train], labels[train], query_ids[train])
    return {
        "MSP": 1.0 - np.max(probs, axis=1),
        "MaxLogit": maxlogit.score(h, logits, query_ids),
        "Energy": energy.score(h, logits, query_ids),
        "Mahalanobis": mahal.score(h, logits, query_ids),
        "RMD": rmd.score(h),
        "kNN": knn.score(h),
        "Full ViM": full_vim.score(h, logits, query_ids),
        "Residual-only ViM": vim.score(h),
    }


def _sample_ood_rows(scores: dict[str, np.ndarray], masks: dict[str, np.ndarray], seed: int) -> list[dict[str, Any]]:
    truth = np.zeros(len(next(iter(scores.values()))), dtype=int)
    truth[masks["ood_test"]] = 1
    benchmark = masks["benchmark"]
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        metrics = ood_metrics(truth[benchmark], scores[method][benchmark])
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "extra_ood_data": "No",
                "benchmark_records": int(benchmark.sum()),
                "id_records": int(masks["id_test"].sum()),
                "ood_records": int(masks["ood_test"].sum()),
                "auroc": metrics["auroc"],
                "aupr_ood": metrics["aupr"],
                "fpr95": metrics["fpr95"],
            }
        )
    return rows


def _source_pca_space(features: np.ndarray, *, fit_mask: np.ndarray, pca_dim: int, seed: int) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32).reshape(features.shape[0], -1)
    n_components = min(int(pca_dim), int(fit_mask.sum()) - 1, values.shape[1])
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed)).fit(values[fit_mask])
    return pca.transform(values).astype(np.float32)


def _window_rows(
    *,
    args: argparse.Namespace,
    masks: dict[str, np.ndarray],
    a_space: np.ndarray,
    b_space: np.ndarray,
    residual_scores: np.ndarray,
    probabilities: np.ndarray,
    seed: int,
) -> list[dict[str, Any]]:
    cfg = WindowDriftConfig(
        window_size=int(args.window_size),
        mmd_permutations=int(args.mmd_permutations),
        reference_max_samples=int(args.window_size),
        reference_subsample_threshold=int(args.window_size),
        c2st_max_iter=200,
        c2st_regularization=1.0,
        soft_alpha=0.05,
        hard_alpha=0.01,
        seed=int(seed),
    )
    source_indices = np.flatnonzero(masks["calibration"])
    id_indices = np.flatnonzero(masks["id_test"])
    ood_indices = np.flatnonzero(masks["ood_test"])
    block_ids = np.arange(len(a_space), dtype=int).astype(str)
    methods = tuple(str(method) for method in getattr(args, "window_methods", WINDOW_METHOD_ORDER))
    unknown = sorted(set(methods) - set(WINDOW_METHOD_ORDER))
    if unknown:
        raise ValueError(f"Unknown window methods: {unknown}")
    testers: dict[str, Any] = {}
    if "A-MMD" in methods:
        testers["A-MMD"] = MMDPermutationTest(cfg).fit(a_space[source_indices], block_ids=block_ids[source_indices])
    if "B-MMD" in methods:
        testers["B-MMD"] = MMDPermutationTest(cfg).fit(b_space[source_indices], block_ids=block_ids[source_indices])
    if "C2ST" in methods:
        testers["C2ST"] = BlockAwareC2ST(cfg).fit(b_space[source_indices], block_ids=block_ids[source_indices])
    if "KS" in methods:
        testers["KS"] = ScalarKSTest().fit(residual_scores[source_indices])
    rng = np.random.default_rng(int(seed) + 20260721)
    rates = tuple(float(rate) for rate in getattr(args, "ood_rates", (0.0, 0.05, 0.10, 0.20)))
    if 0.0 not in rates:
        raise ValueError("Window grid must include a 0% OOD null condition")
    pvalues: dict[str, dict[float, list[float]]] = {
        method: {rate: [] for rate in rates}
        for method in methods
    }
    runtimes: dict[str, list[float]] = {method: [] for method in methods}
    for rate in rates:
        for trial in range(int(args.window_trials)):
            target = _mixed_window_indices(
                id_indices=id_indices,
                ood_indices=ood_indices,
                rate=rate,
                window_size=int(args.window_size),
                rng=rng,
            )
            for method in methods:
                start = time.perf_counter()
                if method == "A-MMD":
                    p_value = float(
                        testers[method].test(
                            a_space[target],
                            seed=int(seed) + trial + int(rate * 1000),
                            block_ids=block_ids[target],
                        )["p_value"]
                    )
                elif method == "B-MMD":
                    p_value = float(
                        testers[method].test(
                            b_space[target],
                            seed=int(seed) + trial + int(rate * 1000) + 17,
                            block_ids=block_ids[target],
                        )["p_value"]
                    )
                elif method == "C2ST":
                    p_value = float(
                        testers[method].test(
                            b_space[target],
                            seed=int(seed) + trial + int(rate * 1000) + 31,
                            block_ids=block_ids[target],
                        )["p_value"]
                    )
                elif method == "KS":
                    p_value = float(testers[method].test(residual_scores[target])["p_value"])
                elif method in {"BBSDs", "BBSDh"}:
                    reference = rng.choice(source_indices, size=int(args.window_size), replace=False)
                    p_value = _bbsd_pvalue(method, probabilities, reference, target)
                else:  # pragma: no cover - method order is fixed
                    raise RuntimeError(method)
                runtimes[method].append(float(time.perf_counter() - start))
                pvalues[method][rate].append(p_value)
    rows: list[dict[str, Any]] = []
    for method in methods:
        h0 = np.asarray(pvalues[method][0.0], dtype=np.float64)
        powers = {
            rate: float(np.mean(np.asarray(pvalues[method][rate], dtype=np.float64) <= 0.05))
            for rate in rates if rate > 0.0
        }
        row = {
                "method": method,
                "seed": int(seed),
                "window_size": int(args.window_size),
                "trials_per_rate": int(args.window_trials),
                "alpha": 0.05,
                "type_i_at_0_05": float(np.mean(h0 <= 0.05)),
                "fwer": _episode_fwer(h0, horizon=int(args.window_horizon), alpha=0.05),
                "power_5pct": powers.get(0.05, ""),
                "power_10pct": powers.get(0.10, ""),
                "power_20pct": powers.get(0.20, ""),
                "power_100pct": powers.get(1.0, ""),
                "n_at_80": _n_at_80(
                    {rate: powers.get(rate, 0.0) for rate in (0.05, 0.10, 0.20)},
                    window_size=int(args.window_size),
                ),
                "delay_windows": _delay_windows(
                    {rate: powers.get(rate, 0.0) for rate in (0.05, 0.10, 0.20)},
                ),
                "runtime_sec_per_window": float(np.mean(runtimes[method])),
                "p_value_method": _p_value_method(method),
            }
        if bool(getattr(args, "formal_sequential", False)):
            sequential = _formal_episode_metrics(
                pvalues[method],
                horizon=int(args.window_horizon),
                episodes=int(getattr(args, "episode_trials", 1000)),
                alpha=0.05,
                minimum_consecutive=int(getattr(args, "minimum_consecutive_windows", 2)),
                seed=int(seed) + 7919 * (WINDOW_METHOD_ORDER.index(method) + 1),
            )
            row.update(sequential)
        rows.append(row)
    return rows


def _formal_episode_metrics(
    pvalues: dict[float, list[float]], *, horizon: int, episodes: int,
    alpha: float, minimum_consecutive: int, seed: int,
) -> dict[str, Any]:
    """Evaluate a frozen alpha-spending persistence rule on simulated episodes."""
    if horizon < 1 or episodes < 1 or minimum_consecutive < 1:
        raise ValueError("Sequential episode settings must be positive")
    null = np.asarray(pvalues[0.0], dtype=np.float64)
    if null.size == 0:
        raise ValueError("Sequential evaluation needs null-window p-values")
    rng = np.random.default_rng(int(seed))
    per_look_alpha = float(alpha) / float(horizon)

    def sample(pool: np.ndarray, count: int) -> np.ndarray:
        return rng.choice(pool, size=(episodes, count), replace=True)

    def decisions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hits = values <= per_look_alpha
        confirmed = np.zeros(episodes, dtype=bool)
        delays = np.full(episodes, np.nan, dtype=float)
        run = np.zeros(episodes, dtype=int)
        for look in range(horizon):
            run = np.where(hits[:, look], run + 1, 0)
            new = (~confirmed) & (run >= minimum_consecutive)
            delays[new] = look + 1
            confirmed |= new
        return confirmed, delays

    null_decision, _ = decisions(sample(null, horizon))
    output: dict[str, Any] = {
        "sequential_rule": f"alpha/horizon spending; {minimum_consecutive} consecutive windows",
        "sequential_fwer": float(np.mean(null_decision)),
        "episode_trials": int(episodes),
        "minimum_consecutive_windows": int(minimum_consecutive),
    }
    for rate, label in ((0.05, "5pct"), (0.10, "10pct"), (0.20, "20pct"), (1.0, "100pct")):
        pool = np.asarray(pvalues.get(rate, []), dtype=np.float64)
        if pool.size == 0:
            continue
        confirmed, delays = decisions(sample(pool, horizon))
        output[f"detection_rate_{label}"] = float(np.mean(confirmed))
        output[f"delay_windows_{label}"] = (
            float(np.nanmean(delays)) if bool(confirmed.any()) else ""
        )
    transient_pool = np.asarray(pvalues.get(0.20, []), dtype=np.float64)
    if transient_pool.size:
        transient = np.concatenate(
            [sample(transient_pool, min(2, horizon)), sample(null, max(0, horizon - 2))],
            axis=1,
        )
        transient_decision, _ = decisions(transient)
        output["transient_false_persistence"] = float(np.mean(transient_decision))
    return output


def _mixed_window_indices(
    *,
    id_indices: np.ndarray,
    ood_indices: np.ndarray,
    rate: float,
    window_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    ood_count = int(round(float(rate) * int(window_size)))
    id_count = int(window_size) - ood_count
    if id_count > len(id_indices) or ood_count > len(ood_indices):
        raise RuntimeError("Not enough benchmark rows for requested mixed window")
    selected = [
        rng.choice(id_indices, size=id_count, replace=False),
        rng.choice(ood_indices, size=ood_count, replace=False) if ood_count else np.asarray([], dtype=int),
    ]
    out = np.concatenate(selected)
    rng.shuffle(out)
    return out


def _bbsd_pvalue(method: str, probabilities: np.ndarray, reference: np.ndarray, target: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    if method == "BBSDs":
        pvalues = [
            ks_2samp(probs[reference, column], probs[target, column]).pvalue
            for column in range(probs.shape[1])
        ]
        return float(min(1.0, min(pvalues) * probs.shape[1]))
    predictions = np.argmax(probs, axis=1)
    classes = np.arange(probs.shape[1])
    table = np.stack(
        [
            np.bincount(predictions[reference], minlength=len(classes)),
            np.bincount(predictions[target], minlength=len(classes)),
        ]
    )
    keep = table.sum(axis=0) > 0
    if int(keep.sum()) < 2:
        return 1.0
    return float(chi2_contingency(table[:, keep], correction=False).pvalue)


def _vim_rank(penultimate: np.ndarray) -> int:
    values = np.asarray(penultimate)
    return max(1, min(64, int(values.shape[0]) - 2, int(values.shape[1]) - 1))


def _n_at_80(powers: dict[float, float], *, window_size: int) -> Any:
    for rate in (0.05, 0.10, 0.20):
        if powers[rate] >= 0.80:
            return int(window_size)
    return ""


def _delay_windows(powers: dict[float, float]) -> Any:
    for rate in (0.05, 0.10, 0.20):
        if powers[rate] >= 0.80:
            return 1
    return ""


def _episode_fwer(pvalues: np.ndarray, *, horizon: int, alpha: float) -> float:
    values = np.asarray(pvalues, dtype=np.float64)
    episodes = len(values) // int(horizon)
    if episodes < 1:
        return float("nan")
    trimmed = values[: episodes * int(horizon)].reshape(episodes, int(horizon))
    return float(np.mean(np.any(trimmed <= float(alpha), axis=1)))


def _p_value_method(method: str) -> str:
    return {
        "A-MMD": "RBF MMD permutation",
        "B-MMD": "RBF MMD permutation on residual vectors",
        "C2ST": "five-fold logistic C2ST with binomial block tail",
        "KS": "two-sided KS on residual norm",
        "BBSDs": "Bonferroni KS over probability columns",
        "BBSDh": "chi-square on predicted-class counts",
    }[method]


def _write_aggregate_tables(
    *,
    dataset_root: Path,
    spec: DatasetSpec,
    seed_summaries: list[dict[str, Any]],
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
) -> dict[str, Any]:
    tables_dir = dataset_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    table2_rows = [row for summary in seed_summaries for row in summary["table2"]]
    table3_rows = [row for summary in seed_summaries for row in summary["table3"]]
    table4_rows = [row for summary in seed_summaries for row in summary["table4"]]
    table2_summary = _summarize_table2(table2_rows)
    table3_summary = _summarize_by_key(table3_rows, "method", ("auroc", "aupr_ood", "fpr95"))
    table4_summary = _summarize_by_key(
        table4_rows,
        "method",
        (
            "type_i_at_0_05",
            "fwer",
            "power_5pct",
            "power_10pct",
            "power_20pct",
            "runtime_sec_per_window",
        ),
    )
    _write_csv(tables_dir / "table2_classifier_id_performance.csv", table2_summary)
    _write_csv(tables_dir / "table3_sample_ood_detection.csv", table3_summary)
    _write_csv(tables_dir / "table4_window_shift_detection.csv", table4_summary)
    seed_summary_path = dataset_root / f"{spec.name}_seed_{'_'.join(str(s['seed']) for s in seed_summaries)}_focused_summary.csv"
    _write_csv(
        seed_summary_path,
        [
            {
                "dataset": spec.name,
                "seed": summary["seed"],
                "accuracy": summary["table2"][0]["accuracy"],
                "macro_f1": summary["table2"][0]["macro_f1"],
                "best_sample_ood_auroc": max(row["auroc"] for row in summary["table3"]),
                "best_sample_ood_method": max(summary["table3"], key=lambda row: row["auroc"])["method"],
                "best_window_power_10pct": max(row["power_10pct"] for row in summary["table4"]),
                "best_window_method_10pct": max(summary["table4"], key=lambda row: row["power_10pct"])["method"],
                "elapsed_seconds": summary["elapsed_seconds"],
            }
            for summary in seed_summaries
        ],
    )
    manifest = {
        "artifact_type": "docs_focused_standard_ood_dataset_v1",
        "dataset": spec.name,
        "seeds": [int(summary["seed"]) for summary in seed_summaries],
        "tables": {
            "table2": str(tables_dir / "table2_classifier_id_performance.csv"),
            "table3": str(tables_dir / "table3_sample_ood_detection.csv"),
            "table4": str(tables_dir / "table4_window_shift_detection.csv"),
            "seed_summary": str(seed_summary_path),
        },
        "split_counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "records": len(rows),
        "hiddenstate_verified": True,
        "protocol": {
            "classifier_pca_dim": int(args.classifier_pca_dim),
            "a_pca_dim": int(args.a_pca_dim),
            "window_size": int(args.window_size),
            "window_trials": int(args.window_trials),
            "mmd_permutations": int(args.mmd_permutations),
            "window_horizon": int(args.window_horizon),
            "uses_ood_validation": False,
            "qwen_forward_passes": 0,
        },
    }
    write_json(tables_dir / "table_manifest.json", manifest)
    return manifest


def _summarize_table2(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    accuracy = np.asarray([float(row["accuracy"]) for row in rows], dtype=np.float64)
    macro_f1 = np.asarray([float(row["macro_f1"]) for row in rows], dtype=np.float64)
    base = dict(rows[0])
    base.update(
        {
            "seed": "mean",
            "accuracy_mean": float(accuracy.mean()),
            "accuracy_std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
            "macro_f1_mean": float(macro_f1.mean()),
            "macro_f1_std": float(macro_f1.std(ddof=1)) if len(macro_f1) > 1 else 0.0,
            "mean_std": _fmt_mean_std(accuracy, macro_f1),
        }
    )
    return rows + [base]


def _summarize_by_key(rows: list[dict[str, Any]], key: str, metrics: Iterable[str]) -> list[dict[str, Any]]:
    out = list(rows)
    keys = list(dict.fromkeys(str(row[key]) for row in rows))
    for value in keys:
        local = [row for row in rows if str(row[key]) == value]
        summary = {key: value, "seed": "mean", "n_seeds": len(local)}
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in local], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        if {"auroc", "aupr_ood", "fpr95"}.issubset(set(metrics)):
            summary["cell"] = (
                f"{summary['auroc_mean']:.4f} / "
                f"{summary['aupr_ood_mean']:.4f} / "
                f"{summary['fpr95_mean']:.4f}"
            )
        out.append(summary)
    return out


def _fmt_mean_std(accuracy: np.ndarray, macro_f1: np.ndarray) -> str:
    return (
        f"Acc {accuracy.mean():.4f}±{(accuracy.std(ddof=1) if len(accuracy) > 1 else 0.0):.4f}; "
        f"Macro-F1 {macro_f1.mean():.4f}±{(macro_f1.std(ddof=1) if len(macro_f1) > 1 else 0.0):.4f}"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
