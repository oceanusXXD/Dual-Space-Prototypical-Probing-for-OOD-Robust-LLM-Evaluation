#!/usr/bin/env python
"""Run CPU-only supplementary experiments from frozen Qwen hidden caches.

The runner deliberately separates long stages into independently resumable CSV/JSON
artifacts.  It never invokes the Qwen backbone.  Formal results in group1--group6
remain untouched; this script supplies the development ablations requested by the
short report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable

import joblib
import numpy as np
import torch
from scipy.stats import genpareto
from sklearn.cluster import AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.adapt.head import HeadAdaptConfig, HeadAdapter
from src.llm_judge_ood.lifecycle.cluster import ClusterConfig, DocumentClusterer
from src.llm_judge_ood.lifecycle.drift import (
    AlphaSpendingTracker,
    MMDPermutationTest,
    WindowDriftConfig,
    wilson_interval,
)
from src.llm_judge_ood.lifecycle.probe import paired_excess_human_error_probe
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge
from src.llm_judge_ood.model.judge import JudgeTrainingConfig, SharedBackboneJudge
from src.llm_judge_ood.scores.openood import OpenOODPosthocScorer
from src.llm_judge_ood.scores.vim import ViMScorer
from src.llm_judge_ood.shared.metrics import judge_metrics, ood_metrics
from src.llm_judge_ood.shared.whitening import LayerPreprocessor


DEFAULT_OUTPUT = Path("artifacts/llm_judge_ood_asap/cpu_supplementary/acceptance")
DEFAULT_INPUTS = Path("artifacts/llm_judge_ood_asap/formal_acceptance_v1/inputs")
DEFAULT_CACHES = Path("artifacts/llm_judge_ood_asap/formal_acceptance_v1/caches")
DEFAULT_RESULTS = Path("artifacts/llm_judge_ood_asap/formal_acceptance_v5/results")
BASE_CACHE = Path("artifacts/llm_judge_ood_asap/qwen3_5_4b_input_document_masked_mean_v1.npz")
FLOW_DIR = Path("artifacts/llm_judge_ood_asap/asap_prepared_flows")
RUN_NAMES = tuple(
    f"abrupt_{shift}_seed_{seed}"
    for shift in ("near", "far")
    for seed in (42, 43, 44)
) + tuple(f"harmless_seed_{seed}" for seed in (42, 43, 44))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run resumable CPU-only supplementary LLM Judge OOD experiments."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--inputs-dir", default=str(DEFAULT_INPUTS))
    parser.add_argument("--caches-dir", default=str(DEFAULT_CACHES))
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid-trials", type=int, default=100)
    parser.add_argument("--grid-permutations", type=int, default=199)
    parser.add_argument("--probe-repetitions", type=int, default=100)
    parser.add_argument("--sequential-trials", type=int, default=1000)
    parser.add_argument("--sequential-permutations", type=int, default=999)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=(
            "representation", "detector", "mmd", "sequential", "flows",
            "clustering", "probe", "adaptation", "gate", "all",
        ),
        default=("all",),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stages = {
        "representation", "detector", "mmd", "sequential", "flows",
        "clustering", "probe", "adaptation", "gate",
    } if "all" in args.stages else set(args.stages)
    formal = _load_formal_module()
    runs = {
        name: formal._load_run(
            name,
            Path(args.inputs_dir),
            Path(args.caches_dir),
            Path(args.results_dir),
        )
        for name in RUN_NAMES
    }
    base = runs["abrupt_near_seed_42"]
    stage_results: dict[str, Any] = {}

    if "representation" in stages:
        stage_results["representation"] = _cached_stage(
            output / "cpu1_representation_and_heads.csv",
            args.force,
            lambda: _representation_and_head_rows(base),
        )
    if "detector" in stages:
        stage_results["detector"] = _cached_stage(
            output / "cpu2_detector_rank_threshold.csv",
            args.force,
            lambda: _detector_rows(base),
        )
    if "mmd" in stages:
        stage_results["mmd"] = _mmd_stage(
            base,
            output=output,
            trials=int(args.grid_trials),
            permutations=int(args.grid_permutations),
            workers=int(args.workers),
            force=bool(args.force),
        )
    if "sequential" in stages:
        stage_results["sequential"] = _sequential_stage(
            base,
            output=output,
            trials=int(args.sequential_trials),
            permutations=int(args.sequential_permutations),
            workers=int(args.workers),
            force=bool(args.force),
        )
    if "flows" in stages:
        stage_results["flows"] = _cached_stage(
            output / "cpu7_flow_monitoring.csv",
            args.force,
            lambda: _flow_monitoring_rows(base, permutations=999),
        )
    if "clustering" in stages:
        stage_results["clustering"] = _cached_stage(
            output / "cpu8_clustering_grid.csv",
            args.force,
            lambda: _clustering_rows(runs, formal),
        )
    if "probe" in stages:
        stage_results["probe"] = _cached_stage(
            output / "cpu9_probe_budget.csv",
            args.force,
            lambda: _probe_rows(runs, repetitions=int(args.probe_repetitions)),
        )
        stage_results["probe_localization"] = _cached_stage(
            output / "cpu9_probe_localization_sensitivity.csv",
            args.force,
            lambda: _probe_localization_rows(
                runs, formal, repetitions=int(args.probe_repetitions)
            ),
        )
    if "adaptation" in stages or "gate" in stages:
        adaptation_path = output / "cpu10_adaptation_grid.csv"
        adaptation_reused = adaptation_path.exists() and not args.force
        if adaptation_reused:
            adaptation_rows = _read_csv(adaptation_path)
        else:
            adaptation_rows = _adaptation_rows(runs)
            _write_csv(adaptation_path, adaptation_rows)
        stage_results["adaptation"] = {
            "path": str(adaptation_path),
            "rows": len(adaptation_rows),
            "reused": adaptation_reused,
        }
        if "gate" in stages:
            stage_results["gate"] = _cached_stage(
                output / "cpu11_gate_evaluation.csv",
                args.force,
                lambda: _gate_rows(adaptation_rows),
            )

    scope_rows = _scope_rows()
    _write_csv(output / "cpu0_scope.csv", scope_rows)
    manifest = {
        "artifact_type": "llm_judge_ood_cpu_supplementary_v1",
        "protocol": "cached_hidden_cpu_only_development_ablations",
        "qwen_forward_passes": 0,
        "base_cache": str(BASE_CACHE),
        "cached_layers_resolved": [23, 32],
        "cached_pooling": "masked_mean",
        "formal_runs": list(RUN_NAMES),
        "stages": stage_results,
        "parameters": {
            "grid_trials": int(args.grid_trials),
            "grid_permutations": int(args.grid_permutations),
            "probe_repetitions": int(args.probe_repetitions),
            "sequential_trials": int(args.sequential_trials),
            "sequential_permutations": int(args.sequential_permutations),
            "workers": int(args.workers),
        },
        "python": platform.python_version(),
        "numpy": np.__version__,
        "elapsed_seconds": float(time.perf_counter() - started),
        "notes": [
            "Long stages are cached independently and MMD condition grids resume by condition.",
            "Development grids use the requested repeated estimates with Wilson intervals.",
            "The formal 200x1000 group1--group6 artifacts remain the acceptance source.",
            "Sequential Probe early stopping is an unadjusted diagnostic replay and is not formal inference.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _load_formal_module() -> Any:
    path = ROOT / "scripts" / "llm_judge_ood" / "31_run_formal_acceptance.py"
    spec = importlib.util.spec_from_file_location("llm_judge_ood_formal", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cached_stage(path: Path, force: bool, builder: Callable[[], list[dict[str, Any]]]) -> dict[str, Any]:
    if path.exists() and not force:
        return {"path": str(path), "rows": len(_read_csv(path)), "reused": True}
    rows = builder()
    _write_csv(path, rows)
    return {"path": str(path), "rows": len(rows), "reused": False}


def _scope_rows() -> list[dict[str, Any]]:
    return [
        {"experiment": "16-A cached layer selection", "cpu_status": "completed", "reason": "layers 23 and 32 already cached; mean/concatenation are CPU transforms"},
        {"experiment": "16-B pooling", "cpu_status": "deferred_gpu", "reason": "cache contains masked-mean only; last-token/attention/max require Qwen forward"},
        {"experiment": "17 head selection", "cpu_status": "completed_partial_contract", "reason": "linear CE, shallow MLP, ordinal-penalty, CORAL and CORN evaluated on frozen features"},
        {"experiment": "18 uncertainty/rank/threshold", "cpu_status": "completed", "reason": "paired bootstrap, feasible ranks and calibration threshold sweep"},
        {"experiment": "19 MMD difficulty/input/kernel/permutation", "cpu_status": "completed", "reason": "condition-cached CPU permutation grids"},
        {"experiment": "19 block sensitivity", "cpu_status": "not_identifiable_asap", "reason": "ASAP has exactly one query row and one arrival block per document; row/document/arrival block are identical"},
        {"experiment": "19 sequential/gradual/5-seed", "cpu_status": "completed", "reason": "one H=12 H0 p-value bank is replayed for all spending/horizon/consecutive variants"},
        {"experiment": "20 clustering/routing", "cpu_status": "completed", "reason": "spaces, algorithms, hyperparameters, hybrid noise expansion and threshold linkage"},
        {
            "experiment": "21 Probe",
            "cpu_status": "completed_with_diagnostic_only_sequential_replay",
            "reason": (
                "fixed budgets, benign margin and sampling strategies are development analyses; "
                "uncorrected sequential early stopping is not formal inference"
            ),
        },
        {
            "experiment": "22 head Adapt/Gate",
            "cpu_status": "completed_retrospective_diagnostic_only",
            "reason": (
                "replay/anchor/lr/epoch/budget/oracle and gate sweeps on frozen features; "
                "future labels are used only for retrospective truth and cannot select a formal candidate"
            ),
        },
        {"experiment": "22 adapter/LoRA", "cpu_status": "deferred_gpu", "reason": "changes the 4B backbone representation and requires model training/forward"},
        {"experiment": "multi-episode post-update recalibration", "cpu_status": "deferred_new_data", "reason": "requires an independent post-update reference/calibration episode not present in current artifacts"},
    ]


def _masks(run: dict[str, Any]) -> dict[str, np.ndarray]:
    split = np.asarray([str(row["split"]) for row in run["rows"]])
    shift = np.asarray([str(row.get("document_shift_type", "id")) for row in run["rows"]])
    return {
        "train": split == "training_train",
        "validation": split == "training_validation",
        "calibration": split == "training_calibration",
        "reference": split == "training_drift_reference",
        "test": split == "training_test",
        "guard": split == "training_guard",
        "development": split == "development",
        "near": (split == "development") & (shift == "near"),
        "far": (split == "development") & (shift == "far"),
    }


def _representation_and_head_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    masks = _masks(run)
    labels = np.asarray([row["label"] for row in run["rows"]])
    queries = np.asarray([str(row["query_id"]) for row in run["rows"]])
    features = np.asarray(run["features"], dtype=np.float32)
    representations = {
        "layer23_cached": features[:, 0:1, :],
        "layer32_cached": features[:, 1:2, :],
        "layer23_32_mean": features.mean(axis=1, keepdims=True),
        "layer23_32_concat": features.reshape(len(features), 1, -1),
    }
    rows: list[dict[str, Any]] = []
    for name, raw in representations.items():
        preprocessor = LayerPreprocessor(
            method="pca_whiten", pca_components=512, pca_variance_target=0.95, random_state=42
        ).fit(raw[masks["train"]])
        transformed = preprocessor.transform(raw)
        judge = PerQueryLinearJudge(
            LinearJudgeConfig(
                method="linear", representation="last_layer", pca_dim=512,
                class_values=(1, 2, 3, 4, 5), seed=42, learning_rate=1e-3,
                weight_decay=1e-4, epochs=50, batch_size=256, patience=6,
                device="cpu", class_weight="balanced", head_sharing="shared",
            )
        ).fit(
            transformed, labels, queries,
            train_mask=masks["train"], validation_mask=masks["validation"],
        )
        output = judge.predict_output(transformed, queries)
        row = _head_metrics_row(
            run, labels, output.classes[np.argmax(output.probabilities, axis=1)],
            output.penultimate, output.logits, name=f"linear_ce::{name}", family="representation",
        )
        row["pca_dimension"] = int(transformed.shape[-1])
        row["cached_layer_indices"] = "23" if name == "layer23_cached" else "32" if name == "layer32_cached" else "23+32"
        rows.append(row)

    base_x = np.asarray(run["judge_features"], dtype=np.float32)
    base_dim = int(base_x.shape[-1])
    classes = np.asarray([1, 2, 3, 4, 5])
    head_specs = (
        ("linear_ce_production", "production"),
        ("shallow_mlp_ce", "mlp"),
        ("linear_ce_ordinal_penalty", "ordinal_penalty"),
        ("coral", "coral"),
        ("corn", "corn"),
    )
    for name, kind in head_specs:
        if kind == "production":
            output = run["output"]
            prediction = output.classes[np.argmax(output.probabilities, axis=1)]
            h, logits = output.penultimate, output.logits
        elif kind == "mlp":
            model = SharedBackboneJudge(
                JudgeTrainingConfig(
                    hidden_dim=96, output_dim=48, learning_rate=1e-3,
                    weight_decay=1e-4, epochs=50, batch_size=256, patience=6,
                    seed=42, device="cpu", loss="ce", class_values=(1, 2, 3, 4, 5),
                )
            ).fit(
                base_x, labels, queries,
                train_mask=masks["train"], validation_mask=masks["validation"],
            )
            output = model.predict_output(base_x, queries)
            prediction = output.classes[np.argmax(output.probabilities, axis=1)]
            h, logits = output.penultimate, output.logits
        else:
            ordinal = _TorchOrdinalHead(kind=kind, input_dim=base_dim, classes=classes, seed=42)
            ordinal.fit(
                base_x[:, 0, :], labels,
                train_mask=masks["train"], validation_mask=masks["validation"],
            )
            probabilities = ordinal.predict_proba(base_x[:, 0, :])
            prediction = classes[np.argmax(probabilities, axis=1)]
            h = base_x[:, 0, :]
            logits = np.log(np.clip(probabilities, 1e-12, 1.0))
        rows.append(_head_metrics_row(
            run, labels, prediction, h, logits, name=name, family="head",
            frozen_vim=run["vim"] if kind == "production" else None,
        ))
    return rows


class _TorchOrdinalHead:
    """Small CPU-only ordinal ablation; it is not a production model contract."""

    def __init__(self, *, kind: str, input_dim: int, classes: np.ndarray, seed: int) -> None:
        self.kind = str(kind)
        self.input_dim = int(input_dim)
        self.classes = np.asarray(classes)
        self.seed = int(seed)
        self.state: dict[str, torch.Tensor] | None = None
        torch.manual_seed(self.seed)
        if self.kind == "ordinal_penalty":
            self.model: nn.Module = nn.Linear(self.input_dim, len(self.classes))
        elif self.kind == "coral":
            self.model = _CoralModule(self.input_dim, len(self.classes) - 1)
        elif self.kind == "corn":
            self.model = nn.Linear(self.input_dim, len(self.classes) - 1)
        else:
            raise ValueError(self.kind)

    def fit(self, x: np.ndarray, y: np.ndarray, *, train_mask: np.ndarray, validation_mask: np.ndarray) -> None:
        matrix = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        class_to_index = {value: index for index, value in enumerate(self.classes.tolist())}
        target = torch.as_tensor([class_to_index[value] for value in y.tolist()], dtype=torch.long)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        train_indices = np.flatnonzero(train_mask)
        validation_indices = np.flatnonzero(validation_mask)
        rng = np.random.default_rng(self.seed)
        best_qwk = -float("inf")
        stale = 0
        for _epoch in range(50):
            self.model.train()
            for start in range(0, len(train_indices), 256):
                batch = train_indices[rng.permutation(len(train_indices))[start:start + 256]]
                logits = self.model(matrix[batch])
                loss = self._loss(logits, target[batch])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            prediction = self.classes[np.argmax(self.predict_proba(x[validation_indices]), axis=1)]
            score = float(
                judge_metrics(
                    y[validation_indices],
                    prediction,
                    class_values=self.classes,
                )["qwk"]
            )
            if score > best_qwk + 1e-12:
                best_qwk = score
                self.state = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 6:
                    break
        if self.state is not None:
            self.model.load_state_dict(self.state)

    def _loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.kind == "ordinal_penalty":
            ce = nn.functional.cross_entropy(logits, target)
            probabilities = torch.softmax(logits, dim=1)
            distance = torch.abs(
                torch.arange(len(self.classes), dtype=torch.float32)[None, :] - target[:, None]
            )
            return ce + 0.25 * torch.mean(torch.sum(probabilities * distance, dim=1))
        thresholds = torch.arange(len(self.classes) - 1)[None, :]
        if self.kind == "coral":
            binary = (target[:, None] > thresholds).float()
            return nn.functional.binary_cross_entropy_with_logits(logits, binary)
        losses: list[torch.Tensor] = []
        for threshold in range(len(self.classes) - 1):
            eligible = target >= threshold
            binary = (target[eligible] > threshold).float()
            losses.append(nn.functional.binary_cross_entropy_with_logits(logits[eligible, threshold], binary))
        return torch.stack(losses).mean()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.as_tensor(np.asarray(x), dtype=torch.float32))
            conditional = torch.sigmoid(logits).cpu().numpy()
        cumulative = conditional if self.kind == "coral" else np.cumprod(conditional, axis=1)
        probabilities = np.empty((len(cumulative), len(self.classes)), dtype=np.float64)
        probabilities[:, 0] = 1.0 - cumulative[:, 0]
        for index in range(1, len(self.classes) - 1):
            probabilities[:, index] = cumulative[:, index - 1] - cumulative[:, index]
        probabilities[:, -1] = cumulative[:, -1]
        probabilities = np.maximum(probabilities, 0.0)
        return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


class _CoralModule(nn.Module):
    def __init__(self, input_dim: int, thresholds: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(input_dim))
        self.base_bias = nn.Parameter(torch.zeros(()))
        self.gaps = nn.Parameter(torch.zeros(thresholds - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.gaps):
            bias = torch.cat([
                self.base_bias[None],
                self.base_bias[None] - torch.cumsum(nn.functional.softplus(self.gaps), dim=0),
            ])
        else:
            bias = self.base_bias[None]
        return x @ self.weight[:, None] + bias[None, :]


def _head_metrics_row(
    run: dict[str, Any], labels: np.ndarray, prediction: np.ndarray,
    penultimate: np.ndarray, logits: np.ndarray, *, name: str, family: str,
    frozen_vim: ViMScorer | None = None,
) -> dict[str, Any]:
    masks = _masks(run)
    if frozen_vim is None:
        rank = min(64, int(penultimate.shape[1]) - 1)
        vim = ViMScorer(rank=rank).fit(np.asarray(penultimate)[masks["train"]])
        scorer_parameter_source = "head_specific_source_refit"
    else:
        rank = int(frozen_vim.rank)
        vim = frozen_vim
        scorer_parameter_source = "serialized_selected_judge_behavior_ood_scorer"
        if int(penultimate.shape[1]) != int(vim.components_.shape[0]):
            raise RuntimeError(
                "Frozen ViM scorer dimension does not match the production head: "
                f"penultimate={penultimate.shape[1]}, scorer={vim.components_.shape[0]}"
            )
    scores = vim.score(penultimate)
    detector_labels = np.concatenate([
        np.zeros(int(masks["calibration"].sum()), dtype=int),
        np.ones(int(masks["development"].sum()), dtype=int),
    ])
    detector_scores = np.concatenate([scores[masks["calibration"]], scores[masks["development"]]])
    class_values = run["model"].classes_
    id_metrics = judge_metrics(
        labels[masks["test"]], prediction[masks["test"]], class_values=class_values
    )
    near_metrics = judge_metrics(
        labels[masks["near"]], prediction[masks["near"]], class_values=class_values
    )
    far_metrics = judge_metrics(
        labels[masks["far"]], prediction[masks["far"]], class_values=class_values
    )
    mmd_metrics = _head_residual_mmd_metrics(run, vim.residual_features(penultimate))
    return {
        "family": family,
        "variant": name,
        "penultimate_dimension": int(penultimate.shape[1]),
        "vim_rank": int(rank),
        "id_qwk": id_metrics["qwk"], "id_mae": id_metrics["mae"],
        "near_qwk": near_metrics["qwk"], "near_mae": near_metrics["mae"],
        "far_qwk": far_metrics["qwk"], "far_mae": far_metrics["mae"],
        "ood_auroc": roc_auc_score(detector_labels, detector_scores),
        **mmd_metrics,
        "logit_dimension": int(np.asarray(logits).shape[1]),
        "vim_parameter_source": scorer_parameter_source,
        "selection_scope": "source_train_validation_and_development_only",
    }


def _head_residual_mmd_metrics(
    run: dict[str, Any], residual: np.ndarray,
) -> dict[str, Any]:
    spaces = {"head_residual": np.asarray(residual, dtype=np.float64)}
    results: dict[str, dict[str, Any]] = {}
    for scenario in ("H0", "Near", "Far"):
        results[scenario] = _mmd_condition(
            run,
            spaces,
            {
                "grid": "head_residual_W50_p20",
                "space": "head_residual",
                "scenario": scenario,
                "window_documents": 50,
                "shift_proportion": 0.0 if scenario == "H0" else 0.20,
                "permutations": 199,
                "bandwidth_multiplier": None,
            },
            trials=100,
        )
    return {
        "mmd_h0_fpr_005": results["H0"]["fpr_or_power_005"],
        "mmd_near_power_005": results["Near"]["fpr_or_power_005"],
        "mmd_far_power_005": results["Far"]["fpr_or_power_005"],
        "mmd_window_documents": 50,
        "mmd_shift_proportion": 0.20,
        "mmd_trials": 100,
        "mmd_permutations": 199,
    }


def _detector_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    masks = _masks(run)
    rows: list[dict[str, Any]] = []
    for candidate in run["summary"]["judge_behavior_ood"]["candidate_results"]:
        if candidate.get("detector") != "vim":
            continue
        rows.append({
            "experiment": "vim_rank",
            "method": "residual-only ViM",
            "rank": candidate.get("rank"),
            "overall_auroc": candidate.get("development_auroc"),
            "overall_aupr": candidate.get("development_aupr"),
            "overall_fpr95": candidate.get("development_fpr95"),
            "near_auroc": candidate.get("development_by_shift", {}).get("near", {}).get("auroc"),
            "far_auroc": candidate.get("development_by_shift", {}).get("far", {}).get("auroc"),
            "status": "ok",
        })
    h = np.asarray(run["output"].penultimate)
    source = masks["train"]
    # Rank 32 was requested by the report but was absent from the original
    # pipeline grid.  It is a true prefix of the serialized rank-64 SVD basis,
    # so it can be evaluated without an ambiguous SVD refit.
    vim32 = ViMScorer(rank=32)
    vim32.mean_ = np.asarray(run["vim"].mean_, dtype=np.float64).copy()
    vim32.components_ = np.asarray(run["vim"].components_[:, :32], dtype=np.float64).copy()
    vim32.fit_rows_ = int(source.sum())
    score32 = vim32.score(h)
    calibration_indices = np.flatnonzero(masks["calibration"])
    near_indices = np.flatnonzero(masks["near"])
    far_indices = np.flatnonzero(masks["far"])

    def rank_metrics(indices: np.ndarray) -> dict[str, float]:
        truth = np.concatenate([
            np.zeros(len(calibration_indices), dtype=int),
            np.ones(len(indices), dtype=int),
        ])
        scores = np.concatenate([score32[calibration_indices], score32[indices]])
        return ood_metrics(truth, scores)

    overall32 = rank_metrics(np.concatenate([near_indices, far_indices]))
    near32 = rank_metrics(near_indices)
    far32 = rank_metrics(far_indices)
    rows.append({
        "experiment": "vim_rank",
        "method": "residual-only ViM",
        "rank": 32,
        "overall_auroc": overall32["auroc"],
        "overall_aupr": overall32["aupr"],
        "overall_fpr95": overall32["fpr95"],
        "near_auroc": near32["auroc"],
        "far_auroc": far32["auroc"],
        "status": "ok_serialized_rank64_prefix",
    })
    vim = run["vim"]
    residual_scores = vim.score(h)
    mahal = OpenOODPosthocScorer("mahalanobis").fit(
        h[source], run["output"].logits[source],
        np.asarray([row["label"] for row in run["rows"]])[source],
        np.asarray([row["query_id"] for row in run["rows"]])[source],
    )
    mahal_scores = mahal.score(h, run["output"].logits, np.asarray([row["query_id"] for row in run["rows"]]))
    rng = np.random.default_rng(20260720)
    strata = [np.flatnonzero(masks[key]) for key in ("calibration", "near", "far")]
    differences: list[float] = []
    for _ in range(2000):
        selected = [rng.choice(indices, size=len(indices), replace=True) for indices in strata]
        ids, near, far = selected
        ood = np.concatenate([near, far])
        truth = np.concatenate([np.zeros(len(ids), dtype=int), np.ones(len(ood), dtype=int)])
        left = np.concatenate([residual_scores[ids], residual_scores[ood]])
        right = np.concatenate([mahal_scores[ids], mahal_scores[ood]])
        differences.append(float(roc_auc_score(truth, left) - roc_auc_score(truth, right)))
    point_truth = np.concatenate([
        np.zeros(int(masks["calibration"].sum()), dtype=int),
        np.ones(int(masks["development"].sum()), dtype=int),
    ])
    point_residual = np.concatenate([residual_scores[masks["calibration"]], residual_scores[masks["development"]]])
    point_mahal = np.concatenate([mahal_scores[masks["calibration"]], mahal_scores[masks["development"]]])
    diff = np.asarray(differences)
    rows.append({
        "experiment": "paired_bootstrap",
        "method": "Residual-only minus Mahalanobis AUROC",
        "overall_auroc": roc_auc_score(point_truth, point_residual),
        "comparison_auroc": roc_auc_score(point_truth, point_mahal),
        "auroc_difference": float(roc_auc_score(point_truth, point_residual) - roc_auc_score(point_truth, point_mahal)),
        "difference_ci95": json.dumps(np.quantile(diff, [0.025, 0.975]).tolist()),
        "bootstrap_samples": 2000,
        "status": "ok",
    })
    calibration = residual_scores[masks["calibration"]]
    test_id = residual_scores[masks["test"]]
    near = residual_scores[masks["near"]]
    far = residual_scores[masks["far"]]
    for quantile in (0.90, 0.95, 0.975, 0.99, 0.995):
        threshold = float(np.quantile(calibration, quantile))
        rows.append(_threshold_row("empirical_quantile", quantile, threshold, test_id, near, far))
    base_q = 0.90
    base_threshold = float(np.quantile(calibration, base_q))
    excess = calibration[calibration > base_threshold] - base_threshold
    shape, _, scale = genpareto.fit(excess, floc=0.0)
    for tail_probability in (0.05, 0.01, 0.005):
        conditional_tail = tail_probability / (1.0 - base_q)
        threshold = base_threshold + float(genpareto.ppf(1.0 - conditional_tail, shape, loc=0.0, scale=scale))
        rows.append(_threshold_row("evt_gpd_u90", 1.0 - tail_probability, threshold, test_id, near, far))
    development = residual_scores[masks["development"]]
    combined = np.concatenate([calibration, development])
    truth = np.concatenate([np.zeros(len(calibration), dtype=int), np.ones(len(development), dtype=int)])
    candidates = np.unique(combined)
    for fp_cost, fn_cost in ((1.0, 1.0), (5.0, 1.0), (10.0, 1.0)):
        losses = [
            fp_cost * np.mean((combined >= threshold) & (truth == 0))
            + fn_cost * np.mean((combined < threshold) & (truth == 1))
            for threshold in candidates
        ]
        threshold = float(candidates[int(np.argmin(losses))])
        row = _threshold_row("development_cost_sensitive", None, threshold, test_id, near, far)
        row["false_positive_cost"] = fp_cost
        row["false_negative_cost"] = fn_cost
        rows.append(row)
    rows.extend([
        {"experiment": "vim_rank", "method": "residual-only ViM", "rank": 256, "status": "invalid", "reason": "penultimate dimension is 226; residual complement would be empty"},
        {"experiment": "vim_rank", "method": "residual-only ViM", "rank": 512, "status": "invalid", "reason": "penultimate dimension is 226; residual complement would be empty"},
    ])
    return rows


def _threshold_row(method: str, quantile: float | None, threshold: float, id_scores: np.ndarray, near: np.ndarray, far: np.ndarray) -> dict[str, Any]:
    return {
        "experiment": "threshold_calibration", "method": method, "quantile": quantile,
        "threshold": threshold, "id_test_fpr": float(np.mean(id_scores >= threshold)),
        "near_tpr": float(np.mean(near >= threshold)), "far_tpr": float(np.mean(far >= threshold)),
        "status": "ok",
    }


def _mmd_stage(
    run: dict[str, Any], *, output: Path, trials: int, permutations: int,
    workers: int, force: bool,
) -> dict[str, Any]:
    spaces = _mmd_spaces(run)
    masks = _masks(run)
    conditions: list[dict[str, Any]] = []
    for space in ("B_residual", "A_embedding"):
        for scenario in ("H0", "Near", "Far"):
            for window in (10, 20, 30, 40, 50):
                conditions.append({
                    "grid": "window_full_shift", "space": space, "scenario": scenario,
                    "window_documents": window,
                    "shift_proportion": 0.0 if scenario == "H0" else 1.0,
                    "permutations": permutations, "bandwidth_multiplier": None,
                })
            if scenario == "H0":
                continue
            for proportion in (0.05, 0.10, 0.20, 0.50):
                conditions.append({
                    "grid": "mixture_W50", "space": space, "scenario": scenario,
                    "window_documents": 50, "shift_proportion": proportion,
                    "permutations": permutations, "bandwidth_multiplier": None,
                })
    for space in spaces:
        for scenario in ("H0", "Near", "Far"):
            conditions.append({
                "grid": "input_ablation_W50_p10", "space": space, "scenario": scenario,
                "window_documents": 50,
                "shift_proportion": 0.0 if scenario == "H0" else 0.10,
                "permutations": permutations, "bandwidth_multiplier": None,
            })
    base_bandwidth = _pooled_median_bandwidth(
        np.vstack([spaces["B_residual"][masks["reference"]], spaces["B_residual"][masks["calibration"]]])
    )
    for multiplier in (0.5, 1.0, 2.0):
        for scenario in ("H0", "Near", "Far"):
            conditions.append({
                "grid": "bandwidth_W50_p20", "space": "B_residual", "scenario": scenario,
                "window_documents": 50, "shift_proportion": 0.0 if scenario == "H0" else 0.20,
                "permutations": permutations, "bandwidth_multiplier": multiplier,
                "fixed_bandwidth": base_bandwidth * multiplier,
            })
    for count in (199, 499, 999, 4999):
        for scenario in ("H0", "Near", "Far"):
            conditions.append({
                "grid": "permutation_count_W50_p20", "space": "B_residual", "scenario": scenario,
                "window_documents": 50, "shift_proportion": 0.0 if scenario == "H0" else 0.20,
                "permutations": count, "bandwidth_multiplier": None,
            })
    path = output / "cpu3_6_mmd_grids.csv"
    if force and path.exists():
        path.unlink()
    key_fields = (
        "grid", "space", "scenario", "window_documents", "shift_proportion",
        "permutations", "bandwidth_multiplier",
    )
    existing = {_row_key(row, key_fields): row for row in _read_csv(path)} if path.exists() else {}
    pending = [condition for condition in conditions if _row_key(condition, key_fields) not in existing]
    rows = list(existing.values())

    def execute(condition: dict[str, Any]) -> dict[str, Any]:
        return _mmd_condition(
            run, spaces, condition, trials=trials,
        )

    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {executor.submit(execute, condition): condition for condition in pending}
            for future in as_completed(futures):
                rows.append(future.result())
                rows.sort(key=lambda row: _row_key(row, key_fields))
                _write_csv(path, rows)
    # Also canonicalize/deduplicate a fully reused table produced by an older
    # runner version; this keeps resume semantics exact across numeric CSV text.
    rows.sort(key=lambda row: _row_key(row, key_fields))
    _write_csv(path, rows)
    kernel_path = output / "cpu6_mmd_kernel.csv"
    kernel_rows = _read_csv(kernel_path) if kernel_path.exists() and not force else _kernel_rows(
        run, spaces["B_residual"], trials=trials, permutations=permutations
    )
    if not kernel_path.exists() or force:
        _write_csv(kernel_path, kernel_rows)
    return {
        "grid_path": str(path), "grid_rows": len(rows), "conditions_computed": len(pending),
        "conditions_reused": len(existing), "kernel_path": str(kernel_path),
        "kernel_rows": len(kernel_rows),
    }


def _mmd_spaces(run: dict[str, Any]) -> dict[str, np.ndarray]:
    masks = _masks(run)
    h = np.asarray(run["output"].penultimate, dtype=np.float64)
    logits = np.asarray(run["output"].logits, dtype=np.float64)
    a = np.asarray(run["a"], dtype=np.float64)
    b = np.asarray(run["b"], dtype=np.float64)
    source = masks["train"]

    def scale(values: np.ndarray) -> np.ndarray:
        scaler = StandardScaler().fit(values[source])
        return scaler.transform(values).astype(np.float64)

    pca = PCA(n_components=min(128, int(source.sum()) - 1, h.shape[1]), random_state=42).fit(h[source])
    pca_h = pca.transform(h)
    return {
        "B_residual": b,
        "B_residual_norm": np.linalg.norm(b, axis=1, keepdims=True),
        "A_embedding": a,
        "logits_only": scale(logits),
        "PCA_h_plus_logits": np.concatenate([scale(pca_h), scale(logits)], axis=1),
        "A_plus_residual": np.concatenate([scale(a), scale(b)], axis=1),
    }


def _mmd_condition(
    run: dict[str, Any], spaces: dict[str, np.ndarray], condition: dict[str, Any], *, trials: int,
) -> dict[str, Any]:
    values = spaces[str(condition["space"])]
    masks = _masks(run)
    scenario = str(condition["scenario"])
    window = int(condition["window_documents"])
    proportion = float(condition["shift_proportion"])
    permutations = int(condition["permutations"])
    source_indices = np.flatnonzero(masks["reference"])
    id_indices = np.flatnonzero(masks["calibration"])
    drift_indices = np.flatnonzero(masks["near"] if scenario == "Near" else masks["far"])
    config = WindowDriftConfig(
        window_size=window, minimum_window_documents=max(2, window),
        mmd_permutations=permutations, c2st_enabled=False,
        reference_max_samples=2000, reference_subsample_threshold=5000,
        kernel_bandwidth=condition.get("fixed_bandwidth"), seed=42,
    )
    test = MMDPermutationTest(config).fit(
        values[source_indices], block_ids=np.asarray([f"source::{index}" for index in source_indices])
    )
    stable = _stable_seed(json.dumps(condition, sort_keys=True))
    rng = np.random.default_rng(stable)
    rejection_005 = rejection_001 = 0
    p_values: list[float] = []
    realized_shift = 0
    for trial in range(trials):
        if scenario == "H0":
            selected = rng.choice(id_indices, size=window, replace=False)
            shifted = 0
        else:
            shifted = min(window, max(1, int(round(window * proportion))))
            normal = window - shifted
            chosen_shift = rng.choice(drift_indices, size=shifted, replace=False)
            chosen_id = rng.choice(id_indices, size=normal, replace=False) if normal else np.zeros(0, dtype=int)
            selected = np.concatenate([chosen_shift, chosen_id])
            rng.shuffle(selected)
        realized_shift = shifted
        result = test.test(
            values[selected], seed=stable + trial + 1000,
            block_ids=np.asarray([f"target::{trial}::{index}" for index in selected]),
        )
        p = float(result["conservative_p_value"])
        p_values.append(p)
        rejection_005 += int(p <= 0.05)
        rejection_001 += int(p <= 0.01)
    return {
        **condition,
        "trials": trials,
        "realized_shift_documents": realized_shift,
        "realized_shift_proportion": realized_shift / window,
        "fpr_or_power_005": rejection_005 / trials,
        "fpr_or_power_005_ci95": json.dumps(wilson_interval(rejection_005, trials)),
        "fpr_or_power_001": rejection_001 / trials,
        "fpr_or_power_001_ci95": json.dumps(wilson_interval(rejection_001, trials)),
        "median_p": float(np.median(p_values)),
        "p_value_resolution": 1.0 / (permutations + 1),
        "status": "ok",
    }


def _pooled_median_bandwidth(values: np.ndarray) -> float:
    matrix = np.asarray(values, dtype=np.float64)
    if len(matrix) > 256:
        indices = np.linspace(0, len(matrix) - 1, 256, dtype=int)
        matrix = matrix[indices]
    squared = np.maximum(
        np.sum(matrix * matrix, axis=1)[:, None]
        + np.sum(matrix * matrix, axis=1)[None, :]
        - 2.0 * matrix @ matrix.T,
        0.0,
    )
    distances = np.sqrt(squared[np.triu_indices(len(matrix), k=1)])
    positive = distances[distances > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def _kernel_rows(
    run: dict[str, Any], values: np.ndarray, *, trials: int, permutations: int,
) -> list[dict[str, Any]]:
    masks = _masks(run)
    source = values[masks["reference"]]
    id_indices = np.flatnonzero(masks["calibration"])
    pools = {
        "H0": id_indices,
        "Near": np.flatnonzero(masks["near"]),
        "Far": np.flatnonzero(masks["far"]),
    }
    rows: list[dict[str, Any]] = []
    for kernel_name in (
        "rbf_median", "rbf_multi_0.5_1_2", "linear", "polynomial_degree2",
    ):
        for scenario, pool in pools.items():
            rng = np.random.default_rng(_stable_seed(f"kernel::{kernel_name}::{scenario}"))
            rejections = 0
            p_values: list[float] = []
            for trial in range(trials):
                if scenario == "H0":
                    selected = rng.choice(pool, size=50, replace=False)
                else:
                    shift = rng.choice(pool, size=10, replace=False)
                    normal = rng.choice(id_indices, size=40, replace=False)
                    selected = np.concatenate([shift, normal])
                pooled = np.vstack([source, values[selected]])
                kernel = _kernel_matrix(pooled, kernel_name)
                p = _kernel_permutation_p(
                    kernel, n_source=len(source), permutations=permutations,
                    seed=_stable_seed(f"{kernel_name}:{scenario}:{trial}"),
                )
                p_values.append(p)
                rejections += int(p <= 0.05)
            rows.append({
                "kernel": kernel_name, "scenario": scenario, "window_documents": 50,
                "shift_proportion": 0.0 if scenario == "H0" else 0.20,
                "trials": trials, "permutations": permutations,
                "fpr_or_power_005": rejections / trials,
                "fpr_or_power_005_ci95": json.dumps(wilson_interval(rejections, trials)),
                "median_p": float(np.median(p_values)), "status": "ok",
            })
    return rows


def _kernel_matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if name == "linear":
        scaled = StandardScaler().fit_transform(matrix)
        return (scaled @ scaled.T) / max(scaled.shape[1], 1)
    if name == "polynomial_degree2":
        scaled = StandardScaler().fit_transform(matrix)
        return np.square(1.0 + (scaled @ scaled.T) / max(scaled.shape[1], 1))
    squared = np.maximum(
        np.sum(matrix * matrix, axis=1)[:, None]
        + np.sum(matrix * matrix, axis=1)[None, :]
        - 2.0 * matrix @ matrix.T,
        0.0,
    )
    bandwidth = _pooled_median_bandwidth(matrix)
    if name == "rbf_median":
        return np.exp(-squared / (2.0 * bandwidth**2))
    return sum(
        np.exp(-squared / (2.0 * (bandwidth * multiplier) ** 2))
        for multiplier in (0.5, 1.0, 2.0)
    ) / 3.0


def _kernel_permutation_p(kernel: np.ndarray, *, n_source: int, permutations: int, seed: int) -> float:
    total = int(kernel.shape[0])
    source = np.arange(n_source)
    target = np.arange(n_source, total)
    observed = _v_mmd(kernel, source, target)
    rng = np.random.default_rng(seed)
    strict = 0
    ties = 1
    target_count = total - n_source
    kernel_total = float(kernel.sum())
    row_sums = kernel.sum(axis=1)
    batch_size = min(64, permutations)
    for start in range(0, permutations, batch_size):
        count = min(batch_size, permutations - start)
        chosen = np.stack([rng.permutation(total)[n_source:] for _ in range(count)])
        target_self = kernel[chosen[:, :, None], chosen[:, None, :]].sum(axis=(1, 2))
        target_total = row_sums[chosen].sum(axis=1)
        source_self = kernel_total - 2.0 * target_total + target_self
        cross = target_total - target_self
        statistics = (
            source_self / float(n_source**2)
            + target_self / float(target_count**2)
            - 2.0 * cross / float(n_source * target_count)
        )
        strict += int(np.sum(statistics > observed + 1e-12))
        ties += int(np.sum(np.abs(statistics - observed) <= 1e-12))
    return float((strict + ties) / (permutations + 1))


def _v_mmd(kernel: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    return float(
        kernel[np.ix_(left, left)].mean()
        + kernel[np.ix_(right, right)].mean()
        - 2.0 * kernel[np.ix_(left, right)].mean()
    )


def _sequential_stage(
    run: dict[str, Any], *, output: Path, trials: int, permutations: int,
    workers: int, force: bool,
) -> dict[str, Any]:
    bank_path = output / "sequential_h0_H12_bank.json"
    if bank_path.exists() and not force:
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
    else:
        masks = _masks(run)
        reference = np.flatnonzero(masks["reference"])
        calibration = np.flatnonzero(masks["calibration"])
        ids = np.asarray([str(row["input_document_id"]) for row in run["rows"]])
        blocks = np.asarray([str(row.get("arrival_batch_id", row["input_document_id"])) for row in run["rows"]])
        config = WindowDriftConfig(
            window_size=50, minimum_window_documents=50,
            mmd_permutations=permutations, c2st_enabled=False,
            reference_max_samples=2000, reference_subsample_threshold=5000,
            alpha_fwer=0.05, alpha_spending="pocock", pocock_horizon=12,
            minimum_consecutive_windows=3, sequential_calibration_trials=trials,
            sequential_calibration_seed=20260720, seed=42,
        )
        test = MMDPermutationTest(config).fit(run["b"][reference], block_ids=blocks[reference])
        bank = _parallel_sequential_bank(
            test=test, values=run["b"], calibration=calibration,
            blocks=blocks, config=config, workers=workers,
        )
        write_json(bank_path, bank)
    rows: list[dict[str, Any]] = []
    episodes = bank.get("episodes", [])
    for horizon in (4, 6, 8, 10, 12):
        for minimum in (2, 3, 4, 5):
            if minimum > horizon:
                continue
            for spending in ("pocock", "bonferroni", "harmonic"):
                false_alerts = 0
                run_lengths: list[int] = []
                for episode in episodes:
                    persistent, first = _replay_sequence(
                        episode["p_values"][:horizon], horizon=horizon,
                        minimum=minimum, spending=spending,
                    )
                    false_alerts += int(persistent)
                    run_lengths.append(first if first is not None else horizon + 1)
                count = len(episodes)
                rows.append({
                    "spending": spending, "horizon": horizon,
                    "minimum_consecutive_windows": minimum,
                    "trials": count, "false_alert_count": false_alerts,
                    "episode_fwer": false_alerts / max(count, 1),
                    "episode_fwer_ci95": json.dumps(wilson_interval(false_alerts, count)),
                    "censored_arl": float(np.mean(run_lengths)),
                    "mmd_permutations": permutations,
                    "p_value_resolution": 1.0 / (permutations + 1),
                    "minimum_alpha_allocation": _minimum_allocation(spending, horizon),
                    "resolution_reaches_all_allocations": bool(
                        1.0 / (permutations + 1) <= _minimum_allocation(spending, horizon) + 1e-15
                    ),
                })
    path = output / "cpu7_sequential_h0.csv"
    _write_csv(path, rows)
    return {
        "h0_bank": str(bank_path), "h0_episodes": len(episodes),
        "table": str(path), "rows": len(rows),
        "bank_elapsed_seconds": bank.get("elapsed_seconds"),
    }


def _parallel_sequential_bank(
    *, test: MMDPermutationTest, values: np.ndarray, calibration: np.ndarray,
    blocks: np.ndarray, config: WindowDriftConfig, workers: int,
) -> dict[str, Any]:
    """Build one reusable p-value bank with episode-level CPU parallelism."""

    horizon = int(config.pocock_horizon)
    window = int(config.window_size)
    trials = int(config.sequential_calibration_trials)
    required = horizon * window
    rng = np.random.default_rng(int(config.sequential_calibration_seed))
    sampled = [rng.choice(calibration, size=required, replace=False) for _ in range(trials)]
    started = time.perf_counter()

    def run_trial(trial: int) -> dict[str, Any]:
        p_values: list[float] = []
        for window_index in range(horizon):
            indices = sampled[trial][window_index * window:(window_index + 1) * window]
            result = test.test(
                values[indices],
                seed=_stable_seed(f"sequential-bank:{config.sequential_calibration_seed}:{trial}:{window_index}"),
                block_ids=blocks[indices],
            )
            p_values.append(float(result["conservative_p_value"]))
        persistent, first = _replay_sequence(
            p_values, horizon=horizon,
            minimum=int(config.minimum_consecutive_windows), spending="pocock",
        )
        return {
            "trial": trial, "persistent_false_alert": persistent,
            "first_persistent_window": first, "p_values": p_values,
        }

    episodes: list[dict[str, Any]] = []
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = [executor.submit(run_trial, trial) for trial in range(trials)]
            for future in as_completed(futures):
                episodes.append(future.result())
    episodes.sort(key=lambda row: int(row["trial"]))
    p_values = np.asarray([p for episode in episodes for p in episode["p_values"]])
    false_alerts = sum(bool(episode["persistent_false_alert"]) for episode in episodes)
    run_lengths = [
        int(episode["first_persistent_window"])
        if episode["first_persistent_window"] is not None else horizon + 1
        for episode in episodes
    ]
    return {
        "artifact_type": "llm_judge_ood_cpu_parallel_sequential_h0_bank_v1",
        "scope": "conditional_monte_carlo_over_fixed_independent_training_calibration_pool",
        "sampling": "without_replacement_within_episode_reuse_allowed_across_trials",
        "trials": trials, "episode_count": trials, "horizon": horizon,
        "window_documents": window, "calibration_document_count": len(calibration),
        "minimum_consecutive_windows": int(config.minimum_consecutive_windows),
        "false_alert_count": false_alerts,
        "episode_false_alert_rate": false_alerts / trials,
        "episode_false_alert_rate_ci95": wilson_interval(false_alerts, trials),
        "average_run_length_censored_at_horizon_plus_one": float(np.mean(run_lengths)),
        "window_false_positive_rate_alpha_0_05": float(np.mean(p_values <= 0.05)),
        "window_false_positive_rate_alpha_0_01": float(np.mean(p_values <= 0.01)),
        "permutations": int(config.mmd_permutations),
        "conservative_p_value_resolution": 1.0 / (int(config.mmd_permutations) + 1),
        "workers": max(1, workers), "episodes": episodes,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _replay_sequence(
    p_values: Iterable[float], *, horizon: int, minimum: int, spending: str,
) -> tuple[bool, int | None]:
    consecutive = 0
    for index, p_value in enumerate(p_values):
        if spending in {"pocock", "harmonic"}:
            config = WindowDriftConfig(
                window_size=50, minimum_window_documents=50, mmd_permutations=999,
                c2st_enabled=False, alpha_fwer=0.05,
                alpha_spending=spending, pocock_horizon=horizon,
                minimum_consecutive_windows=minimum,
            )
            allocation = AlphaSpendingTracker(config).allocation(index)
        else:
            allocation = 0.05 / horizon
        consecutive = consecutive + 1 if float(p_value) <= allocation else 0
        if consecutive >= minimum:
            return True, index + 1
    return False, None


def _minimum_allocation(spending: str, horizon: int) -> float:
    if spending == "bonferroni":
        return 0.05 / horizon
    config = WindowDriftConfig(
        window_size=50, minimum_window_documents=50, mmd_permutations=999,
        c2st_enabled=False, alpha_fwer=0.05, alpha_spending=spending,
        pocock_horizon=horizon, minimum_consecutive_windows=1,
    )
    tracker = AlphaSpendingTracker(config)
    return min(tracker.allocation(index) for index in range(horizon))


def _flow_monitoring_rows(run: dict[str, Any], *, permutations: int) -> list[dict[str, Any]]:
    base_payload = np.load(BASE_CACHE, allow_pickle=True)
    base_ids = base_payload["input_document_ids"].astype(str)
    base_features = np.asarray(base_payload["features"], dtype=np.float32)
    base_feature_by_id = {document_id: base_features[index] for index, document_id in enumerate(base_ids)}
    base_rows = {
        str(row["input_document_id"]): row
        for row in read_jsonl("artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl")
    }
    reusable: dict[tuple[str, str], np.ndarray] = {}
    for seed in (42, 43, 44):
        name = f"harmless_seed_{seed}"
        input_rows = read_jsonl(DEFAULT_INPUTS / f"{name}.jsonl")
        cache = np.load(DEFAULT_CACHES / f"{name}.npz", allow_pickle=True)
        feature_by_id = {
            str(document_id): np.asarray(feature, dtype=np.float32)
            for document_id, feature in zip(cache["input_document_ids"], cache["features"], strict=True)
        }
        for row in input_rows:
            if row["split"] != "deployment_stream":
                continue
            lineage = str(row.get("lineage_document_id", ""))
            reusable[(lineage, str(row["input_document_text"]))] = feature_by_id[str(row["input_document_id"])]

    result_dir = Path(run["result_dir"])
    judge_preprocessor = np.load(result_dir / "judge_preprocessor.npz")
    ood_preprocessor = np.load(result_dir / "ood_preprocessor.npz")
    source_indices = np.flatnonzero(_masks(run)["reference"])
    config8 = WindowDriftConfig(
        window_size=50, minimum_window_documents=50, mmd_permutations=permutations,
        c2st_enabled=False, alpha_fwer=0.05, alpha_spending="pocock",
        pocock_horizon=8, minimum_consecutive_windows=3,
        reference_max_samples=2000, reference_subsample_threshold=5000, seed=42,
    )
    mmd = MMDPermutationTest(config8).fit(
        run["b"][source_indices],
        block_ids=np.asarray([f"source::{index}" for index in source_indices]),
    )
    rows: list[dict[str, Any]] = []
    for flow_type in ("abrupt_near", "abrupt_far", "gradual_near", "gradual_far", "harmless"):
        for seed in (42, 43, 44, 45, 46):
            name = f"{flow_type}_seed_{seed}"
            flow_path = FLOW_DIR / f"{name}.jsonl"
            flow_rows = read_jsonl(flow_path)
            raw_features: list[np.ndarray] = []
            missing = 0
            for row in flow_rows:
                lineage = str(row.get("lineage_document_id", ""))
                base_row = base_rows.get(lineage)
                if base_row is not None and str(base_row["input_document_text"]) == str(row["input_document_text"]):
                    raw_features.append(base_feature_by_id[lineage])
                elif (lineage, str(row["input_document_text"])) in reusable:
                    raw_features.append(reusable[(lineage, str(row["input_document_text"]))])
                else:
                    missing += 1
                    raw_features.append(np.full((2, 2560), np.nan, dtype=np.float32))
            if missing:
                rows.append({
                    "flow": name, "status": "deferred_gpu_missing_hidden",
                    "missing_hidden_documents": missing,
                })
                continue
            features = np.stack(raw_features)
            layer = features[:, 0, :]
            judge_matrix = (
                (layer - judge_preprocessor["pca_means"][0])
                @ judge_preprocessor["components"][0].T
                / np.sqrt(np.maximum(judge_preprocessor["explained_variance"][0], 1e-5))
            )
            judge_features = judge_matrix[:, None, :].astype(np.float32)
            judge_output = run["model"].predict_output(
                judge_features, np.asarray([str(row["query_id"]) for row in flow_rows])
            )
            behavior = run["vim"].residual_features(judge_output.penultimate)
            # Compute A as an audit as well; the sequential decision remains B-only.
            _ = (
                (layer - ood_preprocessor["pca_means"][0])
                @ ood_preprocessor["components"][0].T
                / np.sqrt(np.maximum(ood_preprocessor["explained_variance"][0], 1e-5))
            )
            horizon = 12 if flow_type.startswith("gradual") else 8
            tracker_config = replace(config8, pocock_horizon=horizon)
            tracker = AlphaSpendingTracker(tracker_config)
            window_rows: list[dict[str, Any]] = []
            for window_index in sorted({int(row["flow_window_index"]) for row in flow_rows}):
                indices = np.asarray([
                    index for index, row in enumerate(flow_rows)
                    if int(row["flow_window_index"]) == window_index
                ], dtype=int)
                result = mmd.test(
                    behavior[indices], seed=_stable_seed(f"flow::{name}::{window_index}"),
                    block_ids=np.asarray([str(flow_rows[index]["arrival_batch_id"]) for index in indices]),
                )
                decision = tracker.update(window_index=window_index, p_value=float(result["conservative_p_value"]))
                window_rows.append({
                    "window": window_index + 1,
                    "p": float(result["conservative_p_value"]),
                    "target_proportion": float(flow_rows[int(indices[0])].get("flow_target_proportion", 1.0)),
                    **decision,
                })
            first_raw = next((row for row in window_rows if row["p"] <= 0.05), None)
            first_reject = next((row for row in window_rows if row["b_sequential_reject"]), None)
            first_persistent = next((row for row in window_rows if row["persistent_b_drift"]), None)
            rows.append({
                "flow": name, "flow_type": flow_type, "seed": seed, "horizon": horizon,
                "windows": len(window_rows),
                "first_raw_p_le_005_window": first_raw["window"] if first_raw else None,
                "first_sequential_reject_window": first_reject["window"] if first_reject else None,
                "persistent": bool(first_persistent),
                "persistent_window": first_persistent["window"] if first_persistent else None,
                "persistent_target_proportion": first_persistent["target_proportion"] if first_persistent else None,
                "minimum_p": min(row["p"] for row in window_rows),
                "evaluation": "primary_flow",
                "minimum_consecutive_windows": 3,
                "spending": "pocock",
                "status": "ok",
            })
            for minimum in (2, 3, 4, 5):
                for spending in ("pocock", "bonferroni", "harmonic"):
                    persistent, first = _replay_sequence(
                        [row["p"] for row in window_rows],
                        horizon=horizon,
                        minimum=minimum,
                        spending=spending,
                    )
                    rows.append({
                        "flow": name,
                        "flow_type": flow_type,
                        "seed": seed,
                        "horizon": horizon,
                        "windows": len(window_rows),
                        "first_raw_p_le_005_window": first_raw["window"] if first_raw else None,
                        "persistent": persistent,
                        "persistent_window": first,
                        "persistent_target_proportion": (
                            window_rows[first - 1]["target_proportion"] if first is not None else None
                        ),
                        "minimum_p": min(row["p"] for row in window_rows),
                        "evaluation": "sequential_grid",
                        "minimum_consecutive_windows": minimum,
                        "spending": spending,
                        "status": "ok_replayed_p_values",
                    })
    # The prepared flow files cover abrupt/gradual/harmless streams.  Reuse the
    # same frozen ID and shifted features to isolate temporal patterns that need
    # no new Qwen forward: a one-window burst, a two-window transient, and an
    # alternating shift.  Three consecutive rejections should suppress all of
    # them even when each shifted window is individually obvious.
    masks = _masks(run)
    id_pool = np.flatnonzero(masks["calibration"])
    shift_pools = {
        "near": np.flatnonzero(masks["near"]),
        "far": np.flatnonzero(masks["far"]),
    }
    profiles = {
        "single_burst": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "transient_two_window": (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        "alternating": (0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
    }
    for shift_name, shift_pool in shift_pools.items():
        for profile_name, proportions in profiles.items():
            for seed in (42, 43, 44, 45, 46):
                name = f"{profile_name}_{shift_name}_seed_{seed}"
                rng = np.random.default_rng(_stable_seed(f"temporal::{name}"))
                tracker = AlphaSpendingTracker(config8)
                window_rows: list[dict[str, Any]] = []
                for window_index, proportion in enumerate(proportions):
                    shifted = int(round(50 * proportion))
                    normal = 50 - shifted
                    indices = np.concatenate([
                        rng.choice(shift_pool, size=shifted, replace=False),
                        rng.choice(id_pool, size=normal, replace=False),
                    ])
                    rng.shuffle(indices)
                    result = mmd.test(
                        run["b"][indices],
                        seed=_stable_seed(f"temporal-window::{name}::{window_index}"),
                        block_ids=np.asarray([
                            f"temporal::{name}::{window_index}::{index}" for index in indices
                        ]),
                    )
                    decision = tracker.update(
                        window_index=window_index,
                        p_value=float(result["conservative_p_value"]),
                    )
                    window_rows.append({
                        "window": window_index + 1,
                        "p": float(result["conservative_p_value"]),
                        "target_proportion": proportion,
                        **decision,
                    })
                first_raw = next((row for row in window_rows if row["p"] <= 0.05), None)
                first_reject = next((row for row in window_rows if row["b_sequential_reject"]), None)
                first_persistent = next((row for row in window_rows if row["persistent_b_drift"]), None)
                rows.append({
                    "flow": name,
                    "flow_type": f"{profile_name}_{shift_name}",
                    "seed": seed,
                    "horizon": 8,
                    "windows": len(window_rows),
                    "first_raw_p_le_005_window": first_raw["window"] if first_raw else None,
                    "first_sequential_reject_window": first_reject["window"] if first_reject else None,
                    "persistent": bool(first_persistent),
                    "persistent_window": first_persistent["window"] if first_persistent else None,
                    "persistent_target_proportion": (
                        first_persistent["target_proportion"] if first_persistent else None
                    ),
                    "minimum_p": min(row["p"] for row in window_rows),
                    "evaluation": "primary_flow",
                    "minimum_consecutive_windows": 3,
                    "spending": "pocock",
                    "status": "ok_synthetic_cached_features",
                })
                for minimum in (2, 3, 4, 5):
                    for spending in ("pocock", "bonferroni", "harmonic"):
                        persistent, first = _replay_sequence(
                            [row["p"] for row in window_rows],
                            horizon=8,
                            minimum=minimum,
                            spending=spending,
                        )
                        rows.append({
                            "flow": name,
                            "flow_type": f"{profile_name}_{shift_name}",
                            "seed": seed,
                            "horizon": 8,
                            "windows": len(window_rows),
                            "first_raw_p_le_005_window": (
                                first_raw["window"] if first_raw else None
                            ),
                            "persistent": persistent,
                            "persistent_window": first,
                            "persistent_target_proportion": (
                                window_rows[first - 1]["target_proportion"]
                                if first is not None else None
                            ),
                            "minimum_p": min(row["p"] for row in window_rows),
                            "evaluation": "sequential_grid",
                            "minimum_consecutive_windows": minimum,
                            "spending": spending,
                            "status": "ok_synthetic_replayed_p_values",
                        })
    return rows


def _clustering_rows(runs: dict[str, dict[str, Any]], formal: Any) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run_name, run in sorted(runs.items()):
        candidates = formal._persistent_candidates(run)
        if len(candidates) < 2:
            continue
        truth_values = np.asarray([
            str(run["rows"][index]["audit_document_group_id"]) for index in candidates
        ])
        _, truth = np.unique(truth_values, return_inverse=True)
        b = np.asarray(run["b"], dtype=np.float64)
        b_direction = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
        combined = np.concatenate([
            StandardScaler().fit_transform(run["a"]),
            StandardScaler().fit_transform(b),
        ], axis=1)
        pca = PCA(n_components=min(20, len(candidates) - 1, run["a"].shape[1]), random_state=42)
        pca_values = np.zeros((len(run["a"]), pca.n_components), dtype=np.float64)
        pca_values[candidates] = pca.fit_transform(run["a"][candidates])
        spaces = {
            "A_embedding": np.asarray(run["a"]),
            "B_residual": b,
            "B_residual_direction": b_direction,
            "A_plus_B_standardized": combined,
            "A_PCA20": pca_values,
        }
        methods: list[tuple[str, dict[str, Any]]] = []
        for size in (5, 10, 20):
            methods.append(("hdbscan", {"min_cluster_size": size}))
        for k in (5, 10, 20, 30):
            methods.append(("mutual_knn", {"mutual_k": k, "min_cluster_size": 5}))
        for radius in (1.0, 1.5, 2.0, float("inf")):
            methods.append(("hdbscan_knn_expand", {"min_cluster_size": 10, "radius_multiplier": radius}))
        methods.extend([
            ("agglomerative_k2_oracle_count_diagnostic", {}),
            ("spectral_k2_oracle_count_diagnostic", {}),
        ])
        for space_name, all_values in spaces.items():
            values = np.asarray(all_values[candidates], dtype=np.float64)
            for method, parameters in methods:
                labels = _cluster_labels(values, method, parameters)
                nonnoise = labels >= 0
                routing_precision, routing_recall = formal._routing_metrics(
                    run,
                    candidates,
                    labels,
                    embeddings=np.asarray(all_values, dtype=np.float64),
                )
                output.append({
                    "run": run_name, "space": space_name, "method": method,
                    "parameters": json.dumps(parameters, sort_keys=True),
                    "candidate_documents": len(candidates),
                    "clusters": len(set(labels[labels >= 0].tolist())),
                    "noise_rate": float(1.0 - nonnoise.mean()),
                    "assigned_coverage": float(nonnoise.mean()),
                    "purity": formal._purity(truth, labels),
                    "nmi": float(normalized_mutual_info_score(truth, labels)),
                    "ari": float(adjusted_rand_score(truth, labels)),
                    "routing_precision": routing_precision,
                    "routing_recall": routing_recall,
                    "evaluation_scope": "current_threshold_end_to_end_routing",
                })

        segment = _persistent_segment_indices(run)
        calibration = np.flatnonzero(_masks(run)["calibration"])
        for quantile in (0.80, 0.85, 0.90, 0.95, 0.99):
            threshold = float(np.quantile(run["b_score"][calibration], quantile))
            linked = np.asarray([index for index in segment if run["b_score"][index] >= threshold], dtype=int)
            if len(linked) < 5:
                continue
            linked_truth_values = np.asarray([
                str(run["rows"][index]["audit_document_group_id"]) for index in linked
            ])
            _, linked_truth = np.unique(linked_truth_values, return_inverse=True)
            labels = _cluster_labels(
                np.asarray(run["a"])[linked], "hdbscan_knn_expand",
                {"min_cluster_size": 5, "radius_multiplier": 2.0},
            )
            output.append({
                "run": run_name, "space": "A_embedding", "method": "threshold_linked_hybrid",
                "parameters": json.dumps({"soft_quantile": quantile, "radius_multiplier": 2.0}),
                "candidate_documents": len(linked),
                "clusters": len(set(labels[labels >= 0].tolist())),
                "noise_rate": float(np.mean(labels < 0)),
                "assigned_coverage": float(np.mean(labels >= 0)),
                "purity": formal._purity(linked_truth, labels),
                "nmi": float(normalized_mutual_info_score(linked_truth, labels)),
                "ari": float(adjusted_rand_score(linked_truth, labels)),
                "evaluation_scope": "ood_threshold_clustering_linkage_on_rejection_segment",
            })
    return output


def _cluster_labels(values: np.ndarray, method: str, parameters: dict[str, Any]) -> np.ndarray:
    if method == "hdbscan":
        config = ClusterConfig(
            method="hdbscan", min_cluster_size=int(parameters["min_cluster_size"]),
            hdbscan_allow_single_cluster=True,
        )
        return DocumentClusterer(config).fit_predict(values)[0]
    if method == "mutual_knn":
        config = ClusterConfig(
            method="mutual_knn", min_cluster_size=int(parameters["min_cluster_size"]),
            mutual_k=int(parameters["mutual_k"]), min_similarity=0.0,
        )
        return DocumentClusterer(config).fit_predict(values)[0]
    if method == "hdbscan_knn_expand":
        config = ClusterConfig(
            method="hybrid",
            min_cluster_size=int(parameters["min_cluster_size"]),
            hdbscan_allow_single_cluster=True,
            hybrid_radius_multiplier=float(parameters["radius_multiplier"]),
            hybrid_radius_quantile=0.95,
        )
        return DocumentClusterer(config).fit_predict(values)[0]
    if method.startswith("agglomerative"):
        return AgglomerativeClustering(n_clusters=min(2, len(values))).fit_predict(values)
    if method.startswith("spectral"):
        if len(values) < 3:
            return np.zeros(len(values), dtype=int)
        return SpectralClustering(
            n_clusters=2, affinity="nearest_neighbors",
            n_neighbors=min(10, len(values) - 1), random_state=42,
        ).fit_predict(values)
    raise ValueError(method)


def _persistent_segment_indices(run: dict[str, Any]) -> np.ndarray:
    rows = [
        json.loads(line) for line in
        (Path(run["result_dir"]) / "window_drift.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    active: list[int] = []
    for row in rows:
        if row.get("b_sequential_reject"):
            active.extend(row.get("document_indices", []))
        if row.get("persistent_b_drift"):
            break
    return np.asarray(sorted(set(active)), dtype=int)


def _probe_rows(
    runs: dict[str, dict[str, Any]], *, repetitions: int,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for run_name, run in sorted(runs.items()):
        for lifecycle in run["lifecycle"]:
            members = np.asarray(lifecycle.get("member_indices", []), dtype=int)
            if len(members) < 4:
                continue
            cluster = str(lifecycle["document_cluster_id"])
            full = _probe_result_local(run, members, seed=42, n_boot=1000)
            for benign_margin in (0.05, 0.10, 0.15):
                full_status = _three_way_probe_status(full, benign_margin=benign_margin)
                for strategy in ("random", "confidence", "half_random_confidence", "residual_stratified", "core_boundary"):
                    for budget in (5, 10, 15, 20, 30, 40):
                        count = min(budget, len(members))
                        local_repetitions = repetitions if strategy in {"random", "half_random_confidence", "residual_stratified", "core_boundary"} else 1
                        correct = harmful_called = benign_called = uncertain = 0
                        delta_errors: list[float] = []
                        coverage = 0
                        costs: list[int] = []
                        for repetition in range(local_repetitions):
                            selected = _probe_sample_local(
                                run, members, strategy=strategy, count=count,
                                rng=np.random.default_rng(_stable_seed(f"{run_name}:{cluster}:{strategy}:{budget}:{repetition}")),
                            )
                            result = _probe_result_local(
                                run, selected,
                                seed=_stable_seed(f"probe:{run_name}:{cluster}:{strategy}:{budget}:{repetition}"),
                                n_boot=500,
                            )
                            status = _three_way_probe_status(result, benign_margin=benign_margin)
                            correct += int(status == full_status)
                            harmful_called += int(status == "harmful")
                            benign_called += int(status == "benign")
                            uncertain += int(status == "uncertain")
                            delta_errors.append(abs(float(result["harm_delta"]) - float(full["harm_delta"])))
                            coverage += int(
                                float(result["harm_delta_lcb"]) <= float(full["harm_delta"])
                                <= float(result["harm_delta_ucb"])
                            )
                            costs.append(len(selected))
                        output.append({
                            "run": run_name, "cluster": cluster, "strategy": strategy,
                            "budget": budget, "effective_budget": count,
                            "benign_margin": benign_margin, "full_label_status": full_status,
                            "classification_accuracy": correct / local_repetitions,
                            "harmful_call_rate": harmful_called / local_repetitions,
                            "benign_call_rate": benign_called / local_repetitions,
                            "uncertain_rate": uncertain / local_repetitions,
                            "delta_abs_error_mean": float(np.mean(delta_errors)),
                            "ci_coverage": coverage / local_repetitions,
                            "mean_labels": float(np.mean(costs)),
                            "repetitions": local_repetitions, "bootstrap_samples": 500,
                            "experiment": "fixed_budget",
                        })

                # Repeated ordinary bootstrap intervals are not anytime-valid.
                # Keep this replay only to estimate a candidate label cost; it
                # must not be promoted to a formal sequential Probe decision.
                stop_costs: list[int] = []
                stop_correct = 0
                for repetition in range(repetitions):
                    order = np.random.default_rng(
                        _stable_seed(f"sequential-probe:{run_name}:{cluster}:{benign_margin}:{repetition}")
                    ).permutation(members)
                    status = "uncertain"
                    cost = min(40, len(members))
                    for budget in (10, 20, 30, 40):
                        count = min(budget, len(members))
                        result = _probe_result_local(run, order[:count], seed=42 + repetition + budget, n_boot=500)
                        status = _three_way_probe_status(result, benign_margin=benign_margin)
                        cost = count
                        if status != "uncertain" or count == len(members):
                            break
                    stop_costs.append(cost)
                    stop_correct += int(status == full_status)
                output.append({
                    "run": run_name, "cluster": cluster,
                    "strategy": "sequential_random_unadjusted_diagnostic_10_20_30_40",
                    "budget": 40, "effective_budget": max(stop_costs),
                    "benign_margin": benign_margin, "full_label_status": full_status,
                    "classification_accuracy": stop_correct / repetitions,
                    "mean_labels": float(np.mean(stop_costs)),
                    "repetitions": repetitions, "bootstrap_samples": 500,
                    "experiment": "sequential_budget_diagnostic",
                    "evaluation_scope": "development_cost_replay_only",
                    "formal_inference_eligible": False,
                    "early_stopping_correction": "none",
                    "inference_warning": (
                        "ordinary_bootstrap_intervals_reused_at_multiple_looks_are_not_anytime_valid"
                    ),
                })
    return output


def _probe_localization_rows(
    runs: dict[str, dict[str, Any]], formal: Any, *, repetitions: int,
) -> list[dict[str, Any]]:
    """Compare the same budget on production clusters and audit-only oracle groups."""

    output: list[dict[str, Any]] = []
    for run_name, run in sorted(runs.items()):
        cluster_sets: dict[str, list[tuple[str, np.ndarray]]] = {
            "production_lifecycle": [
                (
                    str(row["document_cluster_id"]),
                    np.asarray(row.get("member_indices", []), dtype=int),
                )
                for row in run["lifecycle"]
            ],
            "oracle_audit_group_diagnostic": [],
        }
        candidates = formal._persistent_candidates(run)
        truth = np.asarray([
            str(run["rows"][index]["audit_document_group_id"]) for index in candidates
        ])
        for group in sorted(set(truth.tolist())):
            cluster_sets["oracle_audit_group_diagnostic"].append(
                (group, candidates[truth == group])
            )
        for localization, clusters in cluster_sets.items():
            for cluster, members in clusters:
                if len(members) < 4:
                    continue
                full = _probe_result_local(run, members, seed=42, n_boot=1000)
                full_status = _three_way_probe_status(full, benign_margin=0.10)
                correct = harmful = benign = uncertain = coverage = 0
                errors: list[float] = []
                costs: list[int] = []
                for repetition in range(repetitions):
                    rng = np.random.default_rng(
                        _stable_seed(
                            f"probe-localization:{run_name}:{localization}:{cluster}:{repetition}"
                        )
                    )
                    selected = rng.choice(
                        members, size=min(20, len(members)), replace=False
                    )
                    result = _probe_result_local(
                        run,
                        selected,
                        seed=_stable_seed(
                            f"probe-localization-result:{run_name}:{localization}:{cluster}:{repetition}"
                        ),
                        n_boot=500,
                    )
                    status = _three_way_probe_status(result, benign_margin=0.10)
                    correct += int(status == full_status)
                    harmful += int(status == "harmful")
                    benign += int(status == "benign")
                    uncertain += int(status == "uncertain")
                    errors.append(abs(float(result["harm_delta"]) - float(full["harm_delta"])))
                    coverage += int(
                        float(result["harm_delta_lcb"]) <= float(full["harm_delta"])
                        <= float(result["harm_delta_ucb"])
                    )
                    costs.append(len(selected))
                output.append({
                    "run": run_name,
                    "localization": localization,
                    "cluster": cluster,
                    "cluster_documents": len(members),
                    "budget": 20,
                    "benign_margin": 0.10,
                    "full_label_status": full_status,
                    "classification_accuracy": correct / repetitions,
                    "harmful_call_rate": harmful / repetitions,
                    "benign_call_rate": benign / repetitions,
                    "uncertain_rate": uncertain / repetitions,
                    "delta_abs_error_mean": float(np.mean(errors)),
                    "ci_coverage": coverage / repetitions,
                    "mean_labels": float(np.mean(costs)),
                    "repetitions": repetitions,
                    "bootstrap_samples": 500,
                    "evaluation_scope": (
                        "audit_only_bottleneck_diagnostic"
                        if localization == "oracle_audit_group_diagnostic"
                        else "deployable_localization"
                    ),
                })
    return output


def _probe_result_local(
    run: dict[str, Any], indices: np.ndarray, *, seed: int, n_boot: int,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=int)
    return paired_excess_human_error_probe(
        y_true=np.asarray([run["rows"][index]["label"] for index in selected]),
        y_pred=np.asarray([run["scores"][index]["judge_prediction"] for index in selected]),
        rater_scores=[run["rows"][index].get("rater_scores") for index in selected],
        reference=run["summary"]["paired_excess_human_error_reference"],
        tolerance=0.15,
        groups=np.asarray([run["rows"][index]["input_document_id"] for index in selected]),
        minimum_documents=4, n_boot=n_boot, seed=seed,
    )


def _three_way_probe_status(result: dict[str, Any], *, benign_margin: float) -> str:
    if result.get("harm_delta") is None:
        return "uncertain"
    if float(result["harm_delta"]) > 0.15 and float(result["harm_delta_lcb"]) > 0.15:
        return "harmful"
    if float(result["harm_delta_ucb"]) < float(benign_margin):
        return "benign"
    return "uncertain"


def _probe_sample_local(
    run: dict[str, Any], members: np.ndarray, *, strategy: str, count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    members = np.asarray(members, dtype=int)
    count = min(int(count), len(members))
    confidence = np.asarray([float(run["scores"][index]["judge_confidence"]) for index in members])
    residual = np.asarray(run["b_score"])[members]
    if strategy == "random":
        return np.asarray(rng.choice(members, size=count, replace=False), dtype=int)
    if strategy == "confidence":
        return members[np.argsort(confidence)[:count]]
    if strategy == "half_random_confidence":
        active_count = count // 2
        active = members[np.argsort(confidence)[:active_count]]
        remaining = np.setdiff1d(members, active)
        random = rng.choice(remaining, size=count - active_count, replace=False)
        return np.concatenate([active, random]).astype(int)
    if strategy == "residual_stratified":
        order = np.argsort(residual)
        bins = np.array_split(members[order], min(4, count))
        selected: list[int] = []
        while len(selected) < count:
            for bucket in bins:
                available = np.setdiff1d(bucket, np.asarray(selected, dtype=int))
                if len(available) and len(selected) < count:
                    selected.append(int(rng.choice(available)))
        return np.asarray(selected, dtype=int)
    values = np.asarray(run["a"])[members]
    centroid = values.mean(axis=0)
    distance = np.linalg.norm(values - centroid, axis=1)
    core_count = count // 2
    core = members[np.argsort(distance)[:core_count]]
    boundary_pool = members[np.argsort(distance)[::-1]]
    boundary = np.asarray([index for index in boundary_pool if index not in set(core.tolist())])[:count - core_count]
    return np.concatenate([core, boundary]).astype(int)


def _adaptation_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
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
        shift = np.asarray([str(row.get("document_shift_type", "id")) for row in run["rows"]])
        old = np.asarray([score["judge_prediction"] for score in run["scores"]])
        predicted_cluster = np.asarray([
            str(score.get("predicted_document_cluster_id", "")) for score in run["scores"]
        ])
        production_pool = np.asarray(sorted({
            int(index)
            for lifecycle in run["lifecycle"]
            if str(lifecycle.get("document_cluster_id")) in harmful
            for index in lifecycle.get("member_indices", [])
        }), dtype=int)
        scenario = "near" if "near" in run_name else "far"
        oracle_pool = np.flatnonzero(
            np.isin(split, ["deployment_ood_evaluation", "deployment_probe"])
            & (shift == scenario)
        )
        routed_future = np.flatnonzero(
            (split == "deployment_future_test") & np.isin(predicted_cluster, list(harmful))
        )
        routed_gate = np.flatnonzero(
            (split == "deployment_gate") & np.isin(predicted_cluster, list(harmful))
        )
        oracle_future = np.flatnonzero((split == "deployment_future_test") & (shift == scenario))
        oracle_gate = np.flatnonzero((split == "deployment_gate") & (shift == scenario))
        guard = np.flatnonzero(split == "training_guard")
        train = np.flatnonzero(split == "training_train")
        u = run["model"].transform_u(run["judge_features"])
        adapter_metadata = run["summary"]["adaptation"].get("adapter") or {}
        base_config = HeadAdaptConfig(**(adapter_metadata.get("config") or {}))
        metadata_replay = np.asarray(sorted({
            int(index)
            for metadata in (adapter_metadata.get("optimization_by_query") or {}).values()
            for index in metadata.get("source_replay_indices", [])
        }), dtype=int)
        replay = metadata_replay if len(metadata_replay) else train[:40]
        specs = _adaptation_specs(base_config)
        for spec in specs:
            mode = str(spec["target_mode"])
            pool = oracle_pool if mode == "oracle" else production_pool
            future = oracle_future if mode == "oracle" else routed_future
            gate = oracle_gate if mode == "oracle" else routed_gate
            if len(pool) == 0 or len(future) == 0 or len(gate) == 0:
                output.append({
                    "run": run_name, "candidate": spec["candidate"],
                    "target_mode": mode, "status": "unavailable_empty_route",
                })
                continue
            budget = min(int(spec["target_budget"]), len(pool))
            rng = np.random.default_rng(_stable_seed(f"adapt-target:{run_name}:{spec['candidate']}"))
            target = np.asarray(rng.choice(pool, size=budget, replace=False), dtype=int)
            replay_count = min(max(budget, 1), len(replay))
            selected_replay = replay[:replay_count]
            if spec["candidate"] == "source_only":
                prediction = old.copy()
            else:
                adapter = HeadAdapter(spec["config"]).fit(
                    u_features=u, labels=labels, query_ids=queries,
                    deployment_indices=target,
                    training_replay_indices=selected_replay,
                    class_values=run["model"].classes_, judge=run["model"],
                )
                prediction = adapter.predict(u_features=u, query_ids=queries, fallback=old)
            class_values = run["model"].classes_
            source_old = judge_metrics(labels[guard], old[guard], class_values=class_values)
            source_new = judge_metrics(
                labels[guard], prediction[guard], class_values=class_values
            )
            future_old = judge_metrics(labels[future], old[future], class_values=class_values)
            future_new = judge_metrics(
                labels[future], prediction[future], class_values=class_values
            )
            gate_gain_values = (
                np.abs(old[gate].astype(float) - labels[gate].astype(float))
                - np.abs(prediction[gate].astype(float) - labels[gate].astype(float))
            )
            gate_gain = float(np.mean(gate_gain_values))
            gate_lcb = _mean_lcb(gate_gain_values, seed=_stable_seed(f"gate-lcb:{run_name}:{spec['candidate']}"))
            nfr = float(np.mean(
                (prediction[guard] != labels[guard]) & (old[guard] == labels[guard])
            ))
            qwk_drop = float(source_old["qwk"] - source_new["qwk"])
            future_gain = float(future_old["mae"] - future_new["mae"])
            groups = np.asarray([str(run["rows"][index]["audit_document_group_id"]) for index in future])
            worst_accuracy = min(
                float(np.mean(prediction[future][groups == group] == labels[future][groups == group]))
                for group in np.unique(groups)
            )
            row: dict[str, Any] = {
                "run": run_name, "scenario": scenario, "candidate": spec["candidate"],
                "target_mode": mode, "target_budget": int(spec["target_budget"]),
                "target_documents": len(target), "replay_documents": len(selected_replay),
                "training_replay_weight": spec["config"].training_replay_weight,
                "anchor_weight": spec["config"].anchor_weight,
                "learning_rate": spec["config"].learning_rate,
                "epochs": spec["config"].epochs,
                "future_documents": len(future),
                "future_mae_before": future_old["mae"], "future_mae_after": future_new["mae"],
                "future_mae_gain": future_gain,
                "future_qwk_before": future_old["qwk"], "future_qwk_after": future_new["qwk"],
                "future_worst_group_accuracy": worst_accuracy,
                "gate_documents": len(gate), "gate_gain": gate_gain, "gate_gain_lcb": gate_lcb,
                "source_nfr": nfr, "source_qwk_drop": qwk_drop,
                "oracle_good_candidate": bool(future_gain >= 0.1 and nfr <= 0.05 and qwk_drop <= 0.02),
                "full_gate_accept": bool(gate_gain >= 0.1 and gate_lcb > 0.0 and nfr <= 0.05 and qwk_drop <= 0.02),
                "evaluation_scope": "retrospective_future_label_diagnostic_only",
                "formal_candidate_eligible": False,
                "formal_gate_eligible": False,
                "uses_future_labels_for_candidate_truth": True,
                "status": "ok",
            }
            for sample_size in (10, 20, 30, 50):
                row[f"gate_accept_rate_n{sample_size}"] = _gate_sample_accept_rate(
                    gate_gain_values, sample_size=sample_size,
                    source_nfr=nfr, source_qwk_drop=qwk_drop,
                    seed=_stable_seed(f"gate-size:{run_name}:{spec['candidate']}:{sample_size}"),
                )
            output.append(row)
    return output


def _adaptation_specs(base: HeadAdaptConfig) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"candidate": "source_only", "config": base, "target_budget": 20, "target_mode": "production"},
        {"candidate": "baseline_replay_anchor", "config": base, "target_budget": 20, "target_mode": "production"},
        {"candidate": "no_replay", "config": replace(base, training_replay_weight=0.0), "target_budget": 20, "target_mode": "production"},
    ]
    for weight in (0.25, 0.5, 2.0, 4.0):
        specs.append({
            "candidate": f"replay_weight_{weight:g}",
            "config": replace(base, training_replay_weight=weight),
            "target_budget": 20, "target_mode": "production",
        })
    for weight in (0.0, 1e-4, 1e-3, 0.1, 1.0):
        specs.append({
            "candidate": f"anchor_{weight:g}", "config": replace(base, anchor_weight=weight),
            "target_budget": 20, "target_mode": "production",
        })
    for learning_rate in (1e-5, 3e-5, 1e-4, 1e-3):
        specs.append({
            "candidate": f"lr_{learning_rate:g}", "config": replace(base, learning_rate=learning_rate),
            "target_budget": 20, "target_mode": "production",
        })
    for epochs in (10, 50):
        specs.append({
            "candidate": f"epochs_{epochs}", "config": replace(base, epochs=epochs),
            "target_budget": 20, "target_mode": "production",
        })
    for budget in (10, 30, 40):
        specs.append({
            "candidate": f"probe_budget_{budget}", "config": base,
            "target_budget": budget, "target_mode": "production",
        })
    for budget in (20, 40):
        specs.append({
            "candidate": f"oracle_localization_{budget}", "config": base,
            "target_budget": budget, "target_mode": "oracle",
        })
    return specs


def _mean_lcb(values: np.ndarray, *, seed: int, samples: int = 1000) -> float:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return float(np.quantile(draws, 0.025))


def _gate_sample_accept_rate(
    values: np.ndarray, *, sample_size: int, source_nfr: float,
    source_qwk_drop: float, seed: int,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    count = min(int(sample_size), len(array))
    if count == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    accepted = 0
    repetitions = 30
    for repetition in range(repetitions):
        selected = rng.choice(array, size=count, replace=False)
        gain = float(np.mean(selected))
        lcb = _mean_lcb(selected, seed=seed + repetition + 100, samples=200)
        accepted += int(
            gain >= 0.1 and lcb > 0.0
            and source_nfr <= 0.05 and source_qwk_drop <= 0.02
        )
    return accepted / repetitions


def _gate_rows(adaptation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [row for row in adaptation_rows if row.get("status") == "ok"]
    output: list[dict[str, Any]] = []
    variants = (
        "no_gate", "target_gain", "target_gain_lcb", "gain_lcb_plus_qwk",
        "gain_lcb_plus_nfr", "full_four_gate",
    )
    for variant in variants:
        output.append(_gate_summary(candidates, variant=variant, epsilon=0.1, eta=0.05, gamma=0.02))
    for epsilon in (0.05, 0.10, 0.15):
        for eta in (0.03, 0.05, 0.08):
            for gamma in (0.01, 0.02, 0.03):
                row = _gate_summary(
                    candidates, variant="full_four_gate_threshold_grid",
                    epsilon=epsilon, eta=eta, gamma=gamma,
                )
                row["experiment"] = "threshold_sensitivity"
                output.append(row)
    for sample_size in (10, 20, 30, 50):
        rates = np.asarray([
            _as_float(row.get(f"gate_accept_rate_n{sample_size}")) for row in candidates
        ])
        full = np.asarray([_as_bool(row.get("full_gate_accept")) for row in candidates])
        output.append({
            "experiment": "gate_sample_size", "variant": "full_four_gate",
            "gate_sample_size": sample_size, "candidates": len(candidates),
            "mean_accept_probability": float(np.nanmean(rates)),
            "mean_absolute_decision_instability": float(np.nanmean(np.abs(rates - full.astype(float)))),
            "evaluation_scope": "retrospective_future_label_diagnostic_only",
            "formal_gate_eligible": False,
            "uses_future_labels_for_candidate_truth": True,
        })
    return output


def _gate_summary(
    candidates: list[dict[str, Any]], *, variant: str,
    epsilon: float, eta: float, gamma: float,
) -> dict[str, Any]:
    truth = np.asarray([_as_bool(row.get("oracle_good_candidate")) for row in candidates])
    accepted: list[bool] = []
    gains: list[float] = []
    for row in candidates:
        gain = _as_float(row.get("gate_gain"))
        lcb = _as_float(row.get("gate_gain_lcb"))
        nfr = _as_float(row.get("source_nfr"))
        qwk = _as_float(row.get("source_qwk_drop"))
        if variant == "no_gate":
            decision = True
        elif variant == "target_gain":
            decision = gain >= epsilon
        elif variant == "target_gain_lcb":
            decision = gain >= epsilon and lcb > 0.0
        elif variant == "gain_lcb_plus_qwk":
            decision = gain >= epsilon and lcb > 0.0 and qwk <= gamma
        elif variant == "gain_lcb_plus_nfr":
            decision = gain >= epsilon and lcb > 0.0 and nfr <= eta
        else:
            decision = gain >= epsilon and lcb > 0.0 and nfr <= eta and qwk <= gamma
        accepted.append(decision)
        gains.append(_as_float(row.get("future_mae_gain")))
    prediction = np.asarray(accepted, dtype=bool)
    bad = ~truth
    false_accepts = int(np.sum(prediction & bad))
    false_rejects = int(np.sum(~prediction & truth))
    return {
        "experiment": "gate_ablation", "variant": variant,
        "epsilon": epsilon, "eta_nfr": eta, "gamma_qwk": gamma,
        "candidates": len(candidates), "good_candidates": int(truth.sum()),
        "bad_candidates": int(bad.sum()), "accepted": int(prediction.sum()),
        "false_accept_count": false_accepts,
        "false_accept_rate": false_accepts / max(int(bad.sum()), 1),
        "false_reject_count": false_rejects,
        "false_reject_rate": false_rejects / max(int(truth.sum()), 1),
        "accepted_future_gain_mean": (
            float(np.mean(np.asarray(gains)[prediction])) if prediction.any() else None
        ),
        "evaluation_scope": "retrospective_future_label_diagnostic_only",
        "formal_gate_eligible": False,
        "uses_future_labels_for_candidate_truth": True,
    }


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _row_key(row: dict[str, Any], fields: Iterable[str]) -> tuple[str, ...]:
    def normalize(value: Any) -> str:
        if value is None or value == "":
            return ""
        return str(value)
    return tuple(normalize(row.get(field)) for field in fields)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


if __name__ == "__main__":
    main()
