#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import ks_2samp
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, read_json, write_json
from src.llm_judge_ood.adapt.head import HeadAdaptConfig, HeadAdapter
from src.llm_judge_ood.eval.asap_auxiliary import evaluate_asap_auxiliary_benchmarks
from src.llm_judge_ood.lifecycle.cluster import ClusterConfig, DocumentClusterer
from src.llm_judge_ood.lifecycle.drift import (
    BehaviorMainRepresentation,
    BlockAwareC2ST,
    MMDPermutationTest,
    WindowDriftConfig,
    _cached_sequential_monte_carlo_audit,
    _document_to_indices,
    ordered_calibration_document_indices,
    run_dual_space_drift_monitor,
    wilson_interval,
)
from src.llm_judge_ood.lifecycle.probe import paired_excess_human_error_probe
from src.llm_judge_ood.scores.vim import ViMScorer
from src.llm_judge_ood.shared.metrics import judge_metrics


def main() -> None:
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run the six-group formal acceptance protocol from cached hidden states.")
    parser.add_argument("--config", default="configs/llm_judge_ood/llm_judge_ood_asap.json")
    parser.add_argument("--inputs-dir", default="artifacts/llm_judge_ood_asap/formal_acceptance_v6/inputs")
    parser.add_argument("--caches-dir", default="artifacts/llm_judge_ood_asap/formal_acceptance_v6/caches")
    parser.add_argument("--results-dir", default="artifacts/llm_judge_ood_asap/formal_acceptance_v6/results")
    parser.add_argument("--output-dir", default="artifacts/llm_judge_ood_asap/formal_acceptance_v6/acceptance")
    parser.add_argument(
        "--within-prompt-input",
        default="artifacts/llm_judge_ood_asap/asap_prepared_contract_v1_within_prompt_covariate_v1.jsonl",
    )
    parser.add_argument(
        "--within-prompt-document-cache",
        default="artifacts/llm_judge_ood_asap/asap_within_prompt_input_document_v1.npz",
    )
    parser.add_argument(
        "--within-prompt-judge-cache",
        default="artifacts/llm_judge_ood_asap/asap_within_prompt_judge_input_v1.npz",
    )
    parser.add_argument(
        "--semantic-task-input",
        default="artifacts/llm_judge_ood_asap/asap_prepared_contract_v1_semantic_task_shift_v1.jsonl",
    )
    parser.add_argument(
        "--semantic-task-document-cache",
        default="artifacts/llm_judge_ood_asap/asap_semantic_task_input_document_v1.npz",
    )
    parser.add_argument(
        "--semantic-task-judge-cache",
        default="artifacts/llm_judge_ood_asap/asap_semantic_task_judge_input_v1.npz",
    )
    parser.add_argument("--skip-auxiliary-benchmarks", action="store_true")
    parser.add_argument("--sequential-h0-path", default=None)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    parser.add_argument("--skip-power", action="store_true")
    args = parser.parse_args()
    if int(args.trials) < 200:
        raise ValueError("Formal acceptance requires at least 200 trials")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = read_json(args.config)
    drift_config = WindowDriftConfig(**payload.get("window_drift", {}))
    drift_config = replace(drift_config, mmd_permutations=int(args.permutations))
    runs = {
        name: _load_run(name, Path(args.inputs_dir), Path(args.caches_dir), Path(args.results_dir))
        for name in (
            "abrupt_near_seed_42", "abrupt_near_seed_43", "abrupt_near_seed_44",
            "abrupt_far_seed_42", "abrupt_far_seed_43", "abrupt_far_seed_44",
            "harmless_seed_42", "harmless_seed_43", "harmless_seed_44",
        )
    }
    detector_rows = _detector_rows(runs["abrupt_near_seed_42"]["summary"])
    _write_csv(output / "group1_detector.csv", detector_rows)
    auxiliary_path = output / "group1b_auxiliary_benchmarks.csv"
    auxiliary_audit_path = output / "group1b_auxiliary_benchmarks.json"
    auxiliary_inputs = {
        "within_prompt_input": Path(args.within_prompt_input),
        "within_prompt_document_cache": Path(args.within_prompt_document_cache),
        "within_prompt_judge_cache": Path(args.within_prompt_judge_cache),
        "semantic_task_input": Path(args.semantic_task_input),
        "semantic_task_document_cache": Path(args.semantic_task_document_cache),
        "semantic_task_judge_cache": Path(args.semantic_task_judge_cache),
    }
    if args.skip_auxiliary_benchmarks:
        auxiliary_rows = [{
            "scenario": "auxiliary_benchmarks",
            "stage": "not_run",
            "status": "explicitly_skipped",
        }]
        auxiliary_audit = {
            "status": "explicitly_skipped",
            "formal_acceptance_complete": False,
        }
    else:
        missing_auxiliary = [str(path) for path in auxiliary_inputs.values() if not path.exists()]
        if missing_auxiliary:
            raise FileNotFoundError(
                "Formal auxiliary benchmark artifacts are required; missing="
                f"{missing_auxiliary}. Use --skip-auxiliary-benchmarks only for an explicitly incomplete run."
            )
        auxiliary_rows, auxiliary_audit = evaluate_asap_auxiliary_benchmarks(
            reference_run=runs["abrupt_near_seed_42"],
            drift_config=drift_config,
            within_input=auxiliary_inputs["within_prompt_input"],
            within_document_cache=auxiliary_inputs["within_prompt_document_cache"],
            within_judge_cache=auxiliary_inputs["within_prompt_judge_cache"],
            semantic_input=auxiliary_inputs["semantic_task_input"],
            semantic_document_cache=auxiliary_inputs["semantic_task_document_cache"],
            semantic_judge_cache=auxiliary_inputs["semantic_task_judge_cache"],
            probe_budget=int(payload.get("probe_budget", 20)),
            probe_min_documents=int(payload.get("probe_min_documents", 4)),
            harm_tolerance=float(payload.get("harm_tolerance", 0.15)),
            bootstrap_samples=int(payload.get("bootstrap_samples", 1000)),
        )
    _write_csv(auxiliary_path, auxiliary_rows)
    write_json(auxiliary_audit_path, auxiliary_audit)
    power_path = output / "group2_mmd_power.csv"
    if not args.skip_power:
        power_rows, power_execution = _power_rows(
            runs["abrupt_near_seed_42"],
            drift_config,
            int(args.trials),
            int(args.workers),
            existing_path=power_path,
        )
        _write_csv(power_path, power_rows)
    elif not power_path.exists():
        raise FileNotFoundError("--skip-power requires an existing group2_mmd_power.csv")
    else:
        power_execution = {
            "mode": "explicit_skip_with_existing_csv",
            "conditions_total": None,
            "conditions_reused": None,
            "conditions_computed": 0,
            "existing_path": str(power_path),
        }
    sequential_h0_path = Path(
        args.sequential_h0_path
        or drift_config.sequential_calibration_cache_path
        or "artifacts/llm_judge_ood_asap/shared/formal_sequential_h0_v1.json"
    )
    sequential_h0_audit = _ensure_sequential_h0_audit(
        runs["abrupt_near_seed_42"], drift_config, sequential_h0_path
    )
    sequential_rows = _sequential_rows(
        runs,
        drift_config,
        sequential_h0_path,
        h0_audit=sequential_h0_audit,
    )
    _write_csv(output / "group3_sequential.csv", sequential_rows)
    cluster_rows = _cluster_rows(runs)
    _write_csv(output / "group4_clustering.csv", cluster_rows)
    probe_rows = _probe_rows(runs, int(args.trials))
    _write_csv(output / "group5_probe.csv", probe_rows)
    adaptation_rows = _production_adaptation_rows(runs)
    adaptation_rows.extend(_adaptation_rows(runs["abrupt_far_seed_42"]))
    _write_csv(output / "group6_adaptation.csv", adaptation_rows)
    revision, worktree_dirty = _git_state()
    group_paths = {
        "1_detector": output / "group1_detector.csv",
        "1b_auxiliary_benchmarks": auxiliary_path,
        "2_mmd_power": power_path,
        "3_sequential": output / "group3_sequential.csv",
        "4_clustering": output / "group4_clustering.csv",
        "5_probe": output / "group5_probe.csv",
        "6_adaptation": output / "group6_adaptation.csv",
    }
    manifest = {
        "artifact_type": "llm_judge_ood_formal_acceptance_v4",
        "protocol_version": "residual_vim_mmd_six_group_plus_auxiliary_benchmark_v4",
        "config": str(Path(args.config)),
        "config_sha256": _sha256_file(Path(args.config)),
        "config_snapshot": payload,
        "results_dir": str(args.results_dir),
        "auxiliary_benchmark": {
            "complete": not bool(args.skip_auxiliary_benchmarks),
            "audit_path": str(auxiliary_audit_path),
            "audit_sha256": _sha256_file(auxiliary_audit_path),
            "inputs": {key: str(path) for key, path in auxiliary_inputs.items()},
            "input_sha256": (
                {key: _sha256_file(path) for key, path in auxiliary_inputs.items()}
                if not args.skip_auxiliary_benchmarks
                else {}
            ),
        },
        "input_sha256": {
            name: _sha256_file(Path(run["input_path"]))
            for name, run in sorted(runs.items())
        },
        "input_metadata_sha256": {
            name: _sha256_file(Path(run["input_metadata_path"]))
            for name, run in sorted(runs.items())
        },
        "hidden_cache_sha256": {
            name: _sha256_file(Path(run["cache_path"]))
            for name, run in sorted(runs.items())
        },
        "hidden_cache_metadata_sha256": {
            name: _sha256_file(Path(run["cache_metadata_path"]))
            for name, run in sorted(runs.items())
        },
        "result_summary_sha256": {
            name: _sha256_file(Path(run["result_dir"]) / "summary.json")
            for name, run in sorted(runs.items())
        },
        "judge_behavior_ood_scorer_sha256": {
            name: _sha256_file(Path(run["result_dir"]) / "judge_behavior_ood_scorer.npz")
            for name, run in sorted(runs.items())
        },
        "split_independence_audit": {
            name: run["split_independence_audit"]
            for name, run in sorted(runs.items())
        },
        "behavior_main_representation_audit": {
            name: run["behavior_main_representation_audit"]
            for name, run in sorted(runs.items())
        },
        "drift_reference_signature": {
            name: run["summary"]["dual_space_drift"]["reference_cache"]["signature"]
            for name, run in sorted(runs.items())
        },
        "formal_mmd_decision_contract": {
            "primary_test": str(drift_config.primary_test),
            "p_value_method": "conservative_permutation_rank",
            "p_value_resolution": 1.0 / (int(args.permutations) + 1),
            "randomized_p_value_role": "diagnostic_only_never_used_for_decisions",
        },
        "trials": int(args.trials),
        "mmd_permutations": int(args.permutations),
        "workers": int(args.workers),
        "power_execution": power_execution,
        "sequential_h0_artifact": {
            "path": str(sequential_h0_path),
            "sha256": _sha256_file(sequential_h0_path),
            "signature": sequential_h0_audit.get("signature"),
            "cache": sequential_h0_audit.get("cache"),
        },
        "seeds": [42, 43, 44],
        "groups": {key: str(path) for key, path in group_paths.items()},
        "group_sha256": {key: _sha256_file(path) for key, path in group_paths.items()},
        "code_revision": revision,
        "code_worktree_dirty": worktree_dirty,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "command": [sys.executable, *sys.argv],
        "elapsed_seconds": float(time.perf_counter() - started),
        "cache_reuse": (
            "Qwen hidden caches and frozen pipeline outputs; no new model forward pass; "
            "power-condition reuse is recorded in power_execution"
        ),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _load_run(name: str, inputs_dir: Path, caches_dir: Path, results_dir: Path) -> dict[str, Any]:
    result = results_dir / name
    input_path = inputs_dir / f"{name}.jsonl"
    input_metadata_path = inputs_dir / f"{name}.metadata.json"
    cache_path = caches_dir / f"{name}.npz"
    cache_metadata_path = caches_dir / f"{name}.metadata.json"
    rows = read_jsonl(input_path)
    scores = read_jsonl(result / "sample_ood_scores.jsonl")
    cache = np.load(cache_path, allow_pickle=True)
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    lifecycle = read_jsonl(result / "document_cluster_lifecycle.jsonl")
    ledger = read_jsonl(result / "label_cost_ledger.jsonl")
    features = np.asarray(cache["features"], dtype=np.float64)
    model = joblib.load(result / "judge_checkpoints" / "selected_linear_judge.joblib")
    query_ids = np.asarray([str(row["query_id"]) for row in rows])
    judge_preprocessor = np.load(result / "judge_preprocessor.npz")
    layer_index = int(summary.get("feature_extractors", {}).get("judge_input", {}).get("separability_selected_layer_index", 0))
    judge_last_layer = features[:, layer_index, :]
    judge_matrix = (
        (judge_last_layer - judge_preprocessor["pca_means"][0])
        @ judge_preprocessor["components"][0].T
        / np.sqrt(np.maximum(judge_preprocessor["explained_variance"][0], 1e-5))
    )
    judge_features = judge_matrix[:, None, :].astype(np.float32)
    output = model.predict_output(judge_features, query_ids)
    source_mask = np.asarray([row["split"] == "training_train" for row in rows])
    behavior_metadata = summary["behavior_main_representation"]
    rank = int(behavior_metadata["vim_rank"])
    # The formal acceptance protocol must consume the exact detector selected
    # and serialized by the frozen pipeline.  Re-running SVD here can rotate a
    # truncated subspace at near-degenerate singular values, so a source refit
    # is not an identity-preserving way to reconstruct the deployed residual.
    vim_artifact_path = result / "judge_behavior_ood_scorer.npz"
    vim_artifact = np.load(vim_artifact_path, allow_pickle=False)
    vim = ViMScorer(rank=rank)
    vim.mean_ = np.asarray(vim_artifact["source_mean"], dtype=np.float64)
    vim.components_ = np.asarray(vim_artifact["principal_components"], dtype=np.float64)
    vim.fit_rows_ = int(source_mask.sum())
    if vim.components_.shape != (output.penultimate.shape[1], rank):
        raise RuntimeError(
            "Serialized ViM components do not match the selected penultimate/rank: "
            f"components={vim.components_.shape}, expected={(output.penultimate.shape[1], rank)}"
        )
    behavior_main = BehaviorMainRepresentation(rank=rank).fit(
        output.penultimate[source_mask],
        scorer=vim,
    )
    b = behavior_main.transform(output.penultimate)
    canonical_b = vim.residual_features(output.penultimate)
    residual_max_abs_error = float(np.max(np.abs(b - canonical_b)))
    if residual_max_abs_error > 1e-12:
        raise RuntimeError(
            "BehaviorMainRepresentation differs from ViMScorer.residual_features: "
            f"max_abs_error={residual_max_abs_error}"
        )
    behavior_audit = behavior_main.to_metadata()
    behavior_audit["max_abs_error_vs_vim_residual_features"] = residual_max_abs_error
    behavior_audit["vim_parameter_source"] = "serialized_selected_judge_behavior_ood_scorer"
    behavior_audit["vim_artifact_path"] = str(vim_artifact_path)
    if behavior_audit["representation"] != "vim_source_subspace_residual_vector":
        raise RuntimeError("Formal B-main is not the ViM residual vector")
    if bool(behavior_audit["uses_logits"]):
        raise RuntimeError("Formal B-main must not consume logits")
    b_score = vim.score(output.penultimate)
    preprocessor = np.load(result / "ood_preprocessor.npz")
    last_layer = features[:, layer_index, :]
    a = (
        (last_layer - preprocessor["pca_means"][0])
        @ preprocessor["components"][0].T
        / np.sqrt(np.maximum(preprocessor["explained_variance"][0], 1e-5))
    )
    return {
        "name": name,
        "input_path": str(input_path),
        "input_metadata_path": str(input_metadata_path),
        "cache_path": str(cache_path),
        "cache_metadata_path": str(cache_metadata_path),
        "rows": rows,
        "scores": scores,
        "summary": summary,
        "lifecycle": lifecycle,
        "ledger": ledger,
        "features": features,
        "judge_features": judge_features,
        "model": model,
        "output": output,
        "vim": vim,
        "b": b,
        "b_score": b_score,
        "a": a,
        "result_dir": str(result),
        "split_independence_audit": _split_independence_audit(rows),
        "behavior_main_representation_audit": behavior_audit,
    }


def _split_independence_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_by_document: dict[str, set[str]] = {}
    for row in rows:
        split_by_document.setdefault(str(row["input_document_id"]), set()).add(
            str(row["split"])
        )
    conflicts = {
        document_id: sorted(splits)
        for document_id, splits in split_by_document.items()
        if len(splits) != 1
    }
    if conflicts:
        examples = list(sorted(conflicts.items()))[:5]
        raise RuntimeError(
            "Formal input documents must belong to exactly one split; "
            f"examples={examples}"
        )
    split_counts: dict[str, int] = {}
    for splits in split_by_document.values():
        split = next(iter(splits))
        split_counts[split] = split_counts.get(split, 0) + 1
    required = (
        "training_train",
        "training_drift_reference",
        "training_calibration",
    )
    missing = [split for split in required if split_counts.get(split, 0) == 0]
    if missing:
        raise RuntimeError(f"Formal split independence audit is missing roles: {missing}")
    return {
        "document_count": int(len(split_by_document)),
        "split_document_counts": dict(sorted(split_counts.items())),
        "documents_with_multiple_splits": 0,
        "training_train_drift_reference_calibration_pairwise_overlap": 0,
        "all_split_document_ids_disjoint": True,
    }


def _detector_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = summary["judge_behavior_ood"]["candidate_results"]
    labels = {
        "vim": "residual-only ViM",
        "full_vim": "Full ViM",
        "mahalanobis": "Mahalanobis",
        "rmd": "RMD",
        "knn": "kNN",
        "energy": "Energy",
        "msp": "MSP",
    }
    rows: list[dict[str, Any]] = []
    for detector, label in labels.items():
        choices = [row for row in candidates if row.get("detector") == detector]
        if not choices:
            continue
        selected = max(choices, key=lambda row: float(row.get("development_auroc", -np.inf)))
        if selected.get("benchmark_test_evidence_level") != "independent_confirmation":
            raise RuntimeError(
                f"Frozen detector {detector!r} has no independent benchmark_test metrics; "
                "rerun the v6 pipeline instead of reporting development results"
            )
        for scope in ("overall", "near", "far"):
            metrics = (
                {
                    "auroc": selected.get("benchmark_test_auroc"),
                    "aupr": selected.get("benchmark_test_aupr"),
                    "fpr95": selected.get("benchmark_test_fpr95"),
                }
                if scope == "overall"
                else selected.get("benchmark_test_by_shift", {}).get(scope, {})
            )
            rows.append({
                "method": label,
                "detector": detector,
                "scope": scope,
                "selection_rank_or_k": selected.get("rank", selected.get("k")),
                "auroc": metrics.get("auroc"),
                "aupr": metrics.get("aupr"),
                "fpr95": metrics.get("fpr95"),
                "selection_scope": "development_only",
                "evaluation_scope": "independent_benchmark_test_only",
            })
    return rows


def _power_rows(
    run: dict[str, Any],
    config: WindowDriftConfig,
    trials: int,
    workers: int,
    *,
    existing_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = run["rows"]
    split = np.asarray([str(row["split"]) for row in rows])
    shift = np.asarray([str(row.get("document_shift_type", "id")) for row in rows])
    block_ids = np.asarray([str(row.get("arrival_batch_id", row["input_document_id"])) for row in rows])
    source = split == "training_drift_reference"
    calibration = split == "training_calibration"
    spaces = {"B_residual": run["b"], "A_embedding": run["a"]}
    pools = {
        "H0": calibration,
        "Near": (shift == "near") & (split == "development"),
        "Far": (shift == "far") & (split == "development"),
    }
    sizes = (50, 100, 200, 500)
    tasks: list[tuple[str, str, int]] = []
    for space in spaces:
        for scenario in pools:
            for size in ((50,) if scenario == "H0" else sizes):
                tasks.append((space, scenario, size))
    existing: dict[tuple[str, str, int], dict[str, Any]] = {}
    if existing_path.exists():
        with existing_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "ok":
                    expected_scope = (
                        "training_calibration_only"
                        if str(row.get("scenario")) == "H0"
                        else "development_only"
                    )
                    if str(row.get("power_pool_scope")) == expected_scope:
                        existing[(str(row["space"]), str(row["scenario"]), int(row["window_documents"]))] = dict(row)
    def run_task(task: tuple[str, str, int]) -> list[dict[str, Any]]:
        space_name, scenario, size = task
        values = spaces[space_name]
        source_values = values[source]
        source_blocks = block_ids[source]
        pool_indices = np.flatnonzero(pools[scenario])
        if len(pool_indices) < size:
            return [{
                "space": space_name,
                "scenario": scenario,
                "window_documents": size,
                "status": "unavailable",
                "pool_rows": int(len(pool_indices)),
                "power_pool_scope": "training_calibration_only" if scenario == "H0" else "development_only",
            }]
        local_config = replace(config, mmd_permutations=int(config.mmd_permutations), c2st_enabled=True)
        mmd = MMDPermutationTest(local_config).fit(source_values, block_ids=source_blocks)
        c2st = (
            BlockAwareC2ST(local_config).fit(source_values, block_ids=source_blocks)
            if space_name == "B_residual"
            else None
        )
        stable_seed = sum((index + 1) * ord(char) for index, char in enumerate(f"{space_name}:{scenario}:{size}"))
        rng = np.random.default_rng(42 + stable_seed % 100000)
        mmd05 = mmd01 = c2st05 = c2st01 = ks05 = ks01 = 0
        p_values: list[float] = []
        for trial in range(int(trials)):
            selected = rng.choice(pool_indices, size=int(size), replace=False)
            target = values[selected]
            target_blocks = block_ids[selected]
            mmd_result = mmd.test(target, seed=trial + 101, block_ids=target_blocks)
            c2st_result = (
                c2st.test(target, seed=trial + 1001, block_ids=target_blocks)
                if c2st is not None
                else None
            )
            p_values.append(float(mmd_result["conservative_p_value"]))
            mmd05 += int(float(mmd_result["conservative_p_value"]) <= 0.05)
            mmd01 += int(float(mmd_result["conservative_p_value"]) <= 0.01)
            if c2st_result is not None:
                c2st05 += int(float(c2st_result["p_value"]) <= 0.05)
                c2st01 += int(float(c2st_result["p_value"]) <= 0.01)
                ks_p = float(ks_2samp(run["b_score"][source], run["b_score"][selected], alternative="two-sided", method="auto").pvalue)
                ks05 += int(ks_p <= 0.05)
                ks01 += int(ks_p <= 0.01)
        return [{
            "space": space_name,
            "scenario": scenario,
            "window_documents": int(size),
            "trials": int(trials),
            "pool_rows": int(len(pool_indices)),
            "mmd_fpr_or_power_alpha_005": float(mmd05 / trials),
            "mmd_fpr_or_power_alpha_001": float(mmd01 / trials),
            "c2st_fpr_or_power_alpha_005": float(c2st05 / trials) if c2st is not None else None,
            "c2st_fpr_or_power_alpha_001": float(c2st01 / trials) if c2st is not None else None,
            "ks_fpr_or_power_alpha_005": float(ks05 / trials) if c2st is not None else None,
            "ks_fpr_or_power_alpha_001": float(ks01 / trials) if c2st is not None else None,
            "mmd_conservative_p_median": float(np.median(p_values)),
            "mmd_p_value_resolution": 1.0 / (int(config.mmd_permutations) + 1),
            "power_pool_scope": "training_calibration_only" if scenario == "H0" else "development_only",
            "status": "ok",
        }]
    output: list[dict[str, Any]] = list(existing.values())
    pending = [task for task in tasks if task not in existing]
    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = [executor.submit(run_task, task) for task in pending]
            for future in as_completed(futures):
                output.extend(future.result())
    for row in output:
        if row.get("status") != "ok":
            continue
        count = int(row["trials"])
        for key in (
            "mmd_fpr_or_power_alpha_005",
            "mmd_fpr_or_power_alpha_001",
            "c2st_fpr_or_power_alpha_005",
            "c2st_fpr_or_power_alpha_001",
            "ks_fpr_or_power_alpha_005",
            "ks_fpr_or_power_alpha_001",
        ):
            value = row.get(key)
            if value not in (None, ""):
                successes = int(round(float(value) * count))
                row[f"{key}_ci95"] = json.dumps(wilson_interval(successes, count))
    ordered = sorted(
        output,
        key=lambda row: (row["space"], row["scenario"], int(row["window_documents"])),
    )
    return ordered, {
        "mode": "condition_level_resume",
        "conditions_total": int(len(tasks)),
        "conditions_reused": int(len(existing)),
        "conditions_computed": int(len(pending)),
        "existing_path": str(existing_path),
    }


def _ensure_sequential_h0_audit(
    run: dict[str, Any],
    config: WindowDriftConfig,
    h0_path: Path,
) -> dict[str, Any]:
    """Bind the H0 audit to the exact serialized ViM residual geometry."""

    local_config = replace(config, sequential_calibration_cache_path=str(h0_path))
    rows = run["rows"]
    split = np.asarray([str(row["split"]) for row in rows])
    document_ids = np.asarray([str(row["input_document_id"]) for row in rows])
    block_ids = np.asarray([
        str(row.get("arrival_batch_id", row["input_document_id"])) for row in rows
    ])
    source = np.flatnonzero(split == "training_drift_reference")
    calibration = ordered_calibration_document_indices(
        np.flatnonzero(split == "training_calibration"),
        document_ids,
        local_config,
    )
    test = MMDPermutationTest(local_config).fit(
        run["b"][source], block_ids=block_ids[source]
    )
    return _cached_sequential_monte_carlo_audit(
        b_test=test,
        b_values=run["b"],
        document_ids=document_ids,
        document_to_records=_document_to_indices(document_ids),
        calibration_documents=calibration,
        block_ids=block_ids,
        config=local_config,
    )


def _sequential_rows(
    runs: dict[str, dict[str, Any]],
    config: WindowDriftConfig,
    h0_path: Path,
    *,
    h0_audit: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if h0_audit is not None or h0_path.exists():
        audit = (
            h0_audit
            if h0_audit is not None
            else json.loads(h0_path.read_text(encoding="utf-8")).get("audit", {})
        )
        rows.append({
            "scenario": "ID_H0_MonteCarlo",
            "single_window_fpr_005": audit.get("window_false_positive_rate_alpha_0_05"),
            "single_window_fpr_001": audit.get("window_false_positive_rate_alpha_0_01"),
            "episode_fwer": audit.get("episode_false_alert_rate"),
            "episode_fwer_wilson_upper": (audit.get("episode_false_alert_rate_ci95") or [None, None])[1],
            "censored_arl": audit.get("average_run_length_censored_at_horizon_plus_one"),
            "trials": audit.get("trials"),
            "persistent_rate": None,
        })
    for name, run in sorted(runs.items()):
        windows = [
            json.loads(line)
            for line in (Path(run["result_dir"]) / "window_drift.jsonl").read_text().splitlines()
            if line
        ]
        single = sum(float(row["B"].get("conservative_p_value", 1.0)) <= 0.05 for row in windows) / max(len(windows), 1)
        persistent = [row for row in windows if row.get("persistent_b_drift")]
        delay = None
        if persistent:
            rejects = [row for row in windows if row.get("b_sequential_reject")]
            delay = int(persistent[0]["window_index"] - rejects[0]["window_index"]) if rejects else None
        rows.append({
            "scenario": name,
            "single_window_fpr_or_detection_rate": float(single),
            "persistent_rate": float(bool(persistent)),
            "persistent_window": persistent[0]["window_index"] if persistent else None,
            "detection_delay_windows": delay,
            "episode_fwer": None,
            "episode_ar1_reference": "H0 artifact above",
            "mmd_primary_test": config.primary_test,
            "p_value_method": "conservative_permutation_rank",
        })
    return rows


def _persistent_candidates(run: dict[str, Any]) -> np.ndarray:
    windows = [
        json.loads(line)
        for line in (Path(run["result_dir"]) / "window_drift.jsonl").read_text().splitlines()
        if line
    ]
    active: list[int] = []
    for row in windows:
        active = active + list(row.get("document_indices", [])) if row.get("b_sequential_reject") else []
        if row.get("persistent_b_drift"):
            break
    scores = run["scores"]
    return np.asarray(sorted(set(index for index in active if scores[index]["judge_behavior_ood_status"] in {"soft_ood", "hard_ood"})), dtype=int)


def _cluster_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, run in sorted(runs.items()):
        candidates = _persistent_candidates(run)
        if len(candidates) < 2:
            continue
        truth_values = np.asarray([str(run["rows"][index]["audit_document_group_id"]) for index in candidates])
        _, truth = np.unique(truth_values, return_inverse=True)
        a_values = np.asarray(run["a"], dtype=np.float64)
        b_values = np.asarray(run["b"], dtype=np.float64)
        b_direction = b_values / np.maximum(
            np.linalg.norm(b_values, axis=1, keepdims=True), 1e-12
        )
        candidates_by_method = (
            (
                "hdbscan",
                "A_input_document_embedding",
                a_values,
                ClusterConfig(method="hdbscan", min_cluster_size=10, hdbscan_allow_single_cluster=True),
            ),
            (
                "mutual_knn",
                "A_input_document_embedding",
                a_values,
                ClusterConfig(method="mutual_knn", min_cluster_size=10, hdbscan_allow_single_cluster=True),
            ),
            (
                "hybrid",
                "B_vim_residual_direction",
                b_direction,
                ClusterConfig(
                    method="hybrid",
                    min_cluster_size=10,
                    hdbscan_allow_single_cluster=True,
                    hybrid_radius_multiplier=1.5,
                    hybrid_radius_quantile=0.95,
                ),
            ),
        )
        for method, cluster_space, values, config in candidates_by_method:
            labels, summaries = DocumentClusterer(config).fit_predict(values[candidates])
            nonnoise = labels >= 0
            predicted = labels.copy()
            purity = _purity(truth, predicted)
            routing_precision, routing_recall = _routing_metrics(
                run, candidates, labels, embeddings=values
            )
            rows.append({
                "run": name,
                "method": method,
                "cluster_space": cluster_space,
                "config": json.dumps(config.to_dict(), sort_keys=True),
                "candidate_documents": int(len(candidates)),
                "clusters": int(len(summaries)),
                "noise_rate": float(1.0 - nonnoise.mean()),
                "purity": purity,
                "nmi": float(normalized_mutual_info_score(truth, predicted)),
                "ari": float(adjusted_rand_score(truth, predicted)),
                "routing_precision": routing_precision,
                "routing_recall": routing_recall,
                "evaluation_scope": "density_clusters_before_production_review_stratum",
            })
    return rows


def _routing_metrics(
    run: dict[str, Any],
    candidates: np.ndarray,
    labels: np.ndarray,
    *,
    embeddings: np.ndarray,
) -> tuple[float | None, float | None]:
    cluster_ids = sorted(set(labels[labels >= 0].tolist()))
    if not cluster_ids:
        return None, None
    prototypes: list[np.ndarray] = []
    radii: list[float] = []
    majority_groups: list[str] = []
    for cluster_id in cluster_ids:
        members = candidates[labels == cluster_id]
        member_values = np.asarray(embeddings)[members]
        centroid = member_values.mean(axis=0)
        prototypes.append(centroid)
        radii.append(float(np.quantile(np.linalg.norm(member_values - centroid, axis=1), 0.95)))
        groups = [str(run["rows"][index]["audit_document_group_id"]) for index in members]
        values, counts = np.unique(groups, return_counts=True)
        majority_groups.append(str(values[int(np.argmax(counts))]))
    pool = np.asarray([
        index for index, row in enumerate(run["rows"])
        if row["split"] in {"deployment_gate", "deployment_future_test"}
        and run["scores"][index]["judge_behavior_ood_status"] in {"soft_ood", "hard_ood"}
    ], dtype=int)
    if pool.size == 0:
        return None, None
    distances = np.linalg.norm(
        np.asarray(embeddings)[pool, None, :] - np.stack(prototypes)[None, :, :], axis=2
    )
    nearest = np.argmin(distances, axis=1)
    accepted = distances[np.arange(len(pool)), nearest] <= np.asarray(radii)[nearest]
    routed = pool[accepted]
    routed_clusters = nearest[accepted]
    correct = sum(
        str(run["rows"][index]["audit_document_group_id"]) == majority_groups[int(cluster)]
        for index, cluster in zip(routed.tolist(), routed_clusters.tolist())
    )
    relevant_groups = set(majority_groups)
    relevant = sum(str(run["rows"][index]["audit_document_group_id"]) in relevant_groups for index in pool.tolist())
    precision = float(correct / len(routed)) if len(routed) else 0.0
    recall = float(correct / relevant) if relevant else None
    return precision, recall


def _purity(truth: np.ndarray, predicted: np.ndarray) -> float:
    assigned = predicted >= 0
    if not assigned.any():
        return 0.0
    return float(sum(np.max(np.bincount(truth[predicted == label])) for label in set(predicted[assigned].tolist())) / assigned.sum())


def _probe_rows(runs: dict[str, dict[str, Any]], trials: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, run in sorted(runs.items()):
        for lifecycle in run["lifecycle"]:
            members = np.asarray(lifecycle.get("member_indices", []), dtype=int)
            if len(members) < 2:
                continue
            cluster_id = str(lifecycle["document_cluster_id"])
            full = _probe_result(run, members, seed=42)
            for strategy, repetitions in (
                ("random", int(trials)),
                ("confidence_only", 1),
                ("full_label", 1),
            ):
                deltas: list[float] = []
                coverage = 0
                correct = 0
                costs: list[int] = []
                stable_seed = sum((index + 1) * ord(char) for index, char in enumerate(f"{name}:{cluster_id}:{strategy}"))
                rng = np.random.default_rng(1000 + stable_seed % 100000)
                for repetition in range(repetitions):
                    selected = _probe_sample(run, members, strategy, rng)
                    result = _probe_result(
                        run,
                        selected,
                        seed=42 + repetition + stable_seed % 100000,
                    )
                    if result["status"] == full["status"]:
                        correct += 1
                    coverage += int(full["harm_delta"] >= result["harm_delta_lcb"] - 1e-12 and full["harm_delta"] <= result["harm_delta_ucb"] + 1e-12)
                    deltas.append(abs(float(result["harm_delta"]) - float(full["harm_delta"])))
                    costs.append(int(len(selected)))
                rows.append({
                    "run": name,
                    "cluster": cluster_id,
                    "strategy": strategy,
                    "strategy_role": "label_based_probe_estimator",
                    "full_label_status": full["status"],
                    "classification_accuracy": float(correct / repetitions),
                    "delta_abs_error_mean": float(np.mean(deltas)),
                    "ci_coverage": float(coverage / repetitions),
                    "mean_unique_documents_labeled": float(np.mean(costs)),
                    "max_unique_documents_labeled": int(max(costs)),
                    "bootstrap_samples": 1000,
                    "repetitions": repetitions,
                })
            warning = (
                run["summary"]
                .get("behavior_warning", {})
                .get("by_predicted_document_cluster", {})
                .get(cluster_id, {})
            )
            for strategy, drop_key, available in (
                ("ATC", "atc_estimated_accuracy_drop", bool(warning.get("atc_trigger_eligible"))),
                ("DoC", "doc_estimated_accuracy_drop", warning.get("doc_estimated_accuracy_drop") is not None),
            ):
                drop = warning.get(drop_key)
                proxy_status = (
                    "harmful"
                    if available and float(drop) >= 0.05
                    else "benign"
                    if available
                    else "unavailable"
                )
                rows.append({
                    "run": name,
                    "cluster": cluster_id,
                    "strategy": strategy,
                    "strategy_role": "unlabeled_accuracy_drop_diagnostic_not_delta_estimator",
                    "full_label_status": full["status"],
                    "proxy_status": proxy_status,
                    "classification_accuracy": (
                        float(proxy_status == full["status"]) if available else None
                    ),
                    "delta_abs_error_mean": None,
                    "ci_coverage": None,
                    "mean_unique_documents_labeled": 0.0,
                    "max_unique_documents_labeled": 0,
                    "bootstrap_samples": 0,
                    "repetitions": 1,
                    "available": available,
                    "estimated_accuracy_drop": drop,
                    "decision_threshold": 0.05,
                    "unavailable_reason": (
                        warning.get("atc_unavailable_reason")
                        if strategy == "ATC" and not available
                        else None
                    ),
                })
    return rows


def _probe_sample(run: dict[str, Any], members: np.ndarray, strategy: str, rng: np.random.Generator) -> np.ndarray:
    count = min(20, len(members))
    if strategy == "full_label":
        return members.copy()
    confidence = np.asarray([float(run["scores"][index]["judge_confidence"]) for index in members])
    if strategy == "confidence_only":
        return members[np.argsort(confidence)[:count]]
    return np.asarray(rng.choice(members, size=count, replace=False), dtype=int)


def _probe_result(run: dict[str, Any], indices: np.ndarray, *, seed: int) -> dict[str, Any]:
    labels = np.asarray([run["rows"][index]["label"] for index in indices])
    predictions = np.asarray([run["scores"][index]["judge_prediction"] for index in indices])
    raters = [run["rows"][index].get("rater_scores") for index in indices]
    reference = run["summary"]["paired_excess_human_error_reference"]
    return paired_excess_human_error_probe(
        y_true=labels,
        y_pred=predictions,
        rater_scores=raters,
        reference=reference,
        tolerance=0.15,
        groups=np.asarray([run["rows"][index]["input_document_id"] for index in indices]),
        minimum_documents=4,
        n_boot=1000,
        seed=int(seed),
    )


def _adaptation_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    name = run["name"]
    lifecycle_harmful = set(run["summary"]["probe"]["harmful_predicted_document_cluster_ids"])
    ledger_probe = next((row for row in run["ledger"] if row.get("stage") == "probe"), {"indices": []})
    target = np.asarray([index for index, cluster in zip(ledger_probe.get("indices", []), ledger_probe.get("predicted_document_cluster_ids", [])) if cluster in lifecycle_harmful], dtype=int)
    rows = run["rows"]
    labels = np.asarray([row["label"] for row in rows])
    query_ids = np.asarray([str(row["query_id"]) for row in rows])
    old = np.asarray([score["judge_prediction"] for score in run["scores"]])
    source_guard = np.asarray([index for index, row in enumerate(rows) if row["split"] == "training_guard"], dtype=int)
    future = np.asarray([
        index for index, row in enumerate(rows)
        if row["split"] == "deployment_future_test"
        and run["scores"][index]["predicted_document_cluster_id"] in lifecycle_harmful
    ], dtype=int)
    if target.size == 0:
        return [{"run": name, "status": "unavailable", "reason": "no_confirmed_harmful_probe_indices"}]
    u = run["model"].transform_u(run["judge_features"])
    adapter_metadata = run["summary"]["adaptation"]["adapter"]
    replay = np.asarray(sorted({
        int(index)
        for metadata in adapter_metadata.get("optimization_by_query", {}).values()
        for index in metadata.get("source_replay_indices", [])
    }), dtype=int)
    base_adapt_config = HeadAdaptConfig(**adapter_metadata["config"])
    variants = {
        "source_only": None,
        "naive_finetuning": replace(base_adapt_config, training_replay_weight=0.0, anchor_weight=0.0),
        "no_replay": replace(base_adapt_config, training_replay_weight=0.0),
        "no_anchor": replace(base_adapt_config, anchor_weight=0.0),
        "full_replay_anchor": base_adapt_config,
        "full_without_gate": base_adapt_config,
    }
    output: list[dict[str, Any]] = []
    for variant, adapt_config in variants.items():
        if adapt_config is None:
            prediction = old.copy()
        else:
            adapter = HeadAdapter(adapt_config).fit(
                u_features=u,
                labels=labels,
                query_ids=query_ids,
                deployment_indices=target,
                training_replay_indices=replay,
                class_values=run["model"].classes_,
                judge=run["model"],
            )
            prediction = adapter.predict(u_features=u, query_ids=query_ids, fallback=old)
        class_values = run["model"].classes_
        source_old = judge_metrics(
            labels[source_guard], old[source_guard], class_values=class_values
        )
        source_new = judge_metrics(
            labels[source_guard], prediction[source_guard], class_values=class_values
        )
        future_old = judge_metrics(labels[future], old[future], class_values=class_values)
        future_new = judge_metrics(
            labels[future], prediction[future], class_values=class_values
        )
        nfr = float(np.mean((prediction[source_guard] != labels[source_guard]) & (old[source_guard] == labels[source_guard])))
        improvement = float(future_old["mae"] - future_new["mae"])
        gate = bool(improvement >= 0.1 and nfr <= 0.05 and source_old["qwk"] - source_new["qwk"] <= 0.02)
        output.append({
            "run": name,
            "variant": variant,
            "target_documents": int(len(target)),
            "deployed_future_mae_before": float(future_old["mae"]),
            "deployed_future_mae_after": float(future_new["mae"]),
            "deployed_future_mae_improvement": improvement,
            "future_qwk_before": float(future_old["qwk"]),
            "future_qwk_after": float(future_new["qwk"]),
            "source_nfr": nfr,
            "source_qwk_drop": float(source_old["qwk"] - source_new["qwk"]),
            "point_source_gate_proxy_would_accept": gate,
            "deployed_if_variant": bool(gate or variant == "full_without_gate"),
            "is_production_gate": False,
            "evaluation_scope": "retrospective_future_label_diagnostic_only",
            "formal_candidate_eligible": False,
            "uses_future_labels_for_candidate_truth": True,
            "source_only_or_ablation": True,
        })
    return output


def _production_adaptation_rows(runs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for name, run in sorted(runs.items()):
        summary = run["summary"]
        adaptation = summary["adaptation"]
        gate = adaptation["gate"]
        improvement = gate.get("paired_excess_error_improvement") or {}
        before = summary.get("deployment_routed_before_adaptation", {})
        after = summary.get("deployment_routed_after_adaptation", {})
        before_mae = before.get("mae")
        after_mae = after.get("mae")
        output.append({
            "run": name,
            "variant": "production_full_method",
            "target_documents": adaptation.get("reused_probe_labels_for_adapt", 0),
            "deployed_future_mae_before": before_mae,
            "deployed_future_mae_after": after_mae,
            "deployed_future_mae_improvement": (
                float(before_mae) - float(after_mae)
                if before_mae is not None and after_mae is not None
                else None
            ),
            "gate_target_excess_error_improvement": improvement.get("improvement"),
            "gate_target_improvement_lcb": (
                (improvement.get("ci95") or [None])[0]
            ),
            "candidate_gate_mae_before": (gate.get("old_gate") or {}).get("mae"),
            "candidate_gate_mae_after": (gate.get("new_gate") or {}).get("mae"),
            "future_qwk_before": before.get("qwk"),
            "future_qwk_after": after.get("qwk"),
            "source_nfr": gate.get("source_guard_negative_flip_rate"),
            "source_qwk_drop": gate.get("source_guard_qwk_drop"),
            "formal_gate_accepted": bool(adaptation.get("candidate_deployed")),
            "deployed_if_variant": bool(adaptation.get("candidate_deployed")),
            "is_production_gate": True,
            "evaluation_scope": "formal_3x3_end_to_end",
            "formal_candidate_eligible": True,
            "uses_future_labels_for_candidate_truth": False,
            "future_metrics_reporting_only": True,
            "source_only_or_ablation": False,
            "update_mode": adaptation.get("update_mode"),
            "probe_status": summary.get("probe", {}).get("status"),
            "gate_failure_reasons": json.dumps(gate.get("failure_reasons", []), ensure_ascii=False),
            "manual_labels_total": adaptation.get("requested_total_labels"),
        })
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, bool(status.strip())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nnot_run\n", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
