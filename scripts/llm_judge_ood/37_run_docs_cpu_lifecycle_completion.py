#!/usr/bin/env python3
"""Complete lifecycle-table CPU experiments from frozen hiddenstate caches.

This runner is intentionally limited to local, reproducible CPU work.  It does
not call an external judge API and does not execute the Qwen backbone.  The
outputs complement the seed-level lifecycle artifacts with the missing window,
Probe, Adapt, Gate, and end-to-end comparison grids used by the docs tables.
"""

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
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import joblib
import numpy as np
import scipy
import sklearn
import torch
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.adapt.head import HeadAdaptConfig, HeadAdapter
from src.llm_judge_ood.scores.vim import ViMScorer
from src.llm_judge_ood.shared.metrics import judge_metrics


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    prepared_path: Path
    hidden_path: Path
    document_hidden_path: Path
    lifecycle_root: Path
    scenarios: tuple[str, ...]


DATASETS = {
    "asap": DatasetSpec(
        name="asap",
        prepared_path=Path("artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl"),
        hidden_path=Path("hiddenstates/asap_aes/qwen3_5_4b_judge_input_asap_rubric_v1.npz"),
        document_hidden_path=Path("hiddenstates/asap_aes/qwen3_5_4b_input_document_masked_mean_v1.npz"),
        lifecycle_root=Path("artifacts/docs_experiments/asap"),
        scenarios=("near", "far"),
    ),
    "ellipse": DatasetSpec(
        name="ellipse",
        prepared_path=Path("artifacts/llm_judge_ood_ellipse/ellipse_prepared_contract_v1.jsonl"),
        hidden_path=Path("hiddenstates/ellipse/qwen3_5_4b_judge_input_overall_v1.npz"),
        document_hidden_path=Path("hiddenstates/ellipse/qwen3_5_4b_input_document_masked_mean_v1.npz"),
        lifecycle_root=Path("artifacts/docs_experiments/ellipse"),
        scenarios=("held_out_prompt",),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=["asap", "ellipse"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-root", default="artifacts/docs_experiments")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--window-trials", type=int, default=40)
    parser.add_argument("--mmd-permutations", type=int, default=99)
    parser.add_argument("--probe-repetitions", type=int, default=30)
    parser.add_argument("--probe-bootstrap", type=int, default=300)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    focused = _load_module(
        "docs_focused_standard_ood",
        ROOT / "scripts/llm_judge_ood/36_run_docs_focused_standard_ood.py",
    )
    supplementary = _load_module(
        "cpu_supplementary_current",
        ROOT / "scripts/llm_judge_ood/32_run_cpu_supplementary.py",
    )
    manifests: list[dict[str, Any]] = []
    for dataset in args.datasets:
        manifests.append(
            _run_dataset(
                args=args,
                spec=DATASETS[str(dataset)],
                focused=focused,
                supplementary=supplementary,
            )
        )
    combined_adaptation: list[dict[str, Any]] = []
    for manifest in manifests:
        combined_adaptation.extend(_read_csv(Path(manifest["tables"]["table6_adaptation"])))
    combined_gate = supplementary._gate_rows(combined_adaptation)
    combined_root = Path(args.output_root) / "lifecycle_cpu_completion"
    combined_root.mkdir(parents=True, exist_ok=True)
    _write_csv(combined_root / "table7_gate_combined.csv", combined_gate)
    write_json(
        combined_root / "manifest.json",
        {
            "artifact_type": "docs_lifecycle_cpu_completion_combined_v1",
            "datasets": [manifest["dataset"] for manifest in manifests],
            "seeds": [int(seed) for seed in args.seeds],
            "qwen_forward_passes": 0,
            "external_api_calls": 0,
            "hiddenstate_verified": all(bool(manifest["hiddenstate_verified"]) for manifest in manifests),
            "table7_gate_combined": str(combined_root / "table7_gate_combined.csv"),
        },
    )
    print(json.dumps({"datasets": manifests}, indent=2, ensure_ascii=False))


def _run_dataset(*, args: argparse.Namespace, spec: DatasetSpec, focused: Any, supplementary: Any) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(args.output_root) / spec.name / "cpu_completion"
    output.mkdir(parents=True, exist_ok=True)
    all_window: list[dict[str, Any]] = []
    all_probe: list[dict[str, Any]] = []
    all_adapt: list[dict[str, Any]] = []
    all_e2e: list[dict[str, Any]] = []
    run_audits: list[dict[str, Any]] = []
    for seed in args.seeds:
        run = _load_current_run(spec, int(seed))
        run_audits.append(run["audit"])
        seed_dir = output / f"seed_{int(seed)}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        window_path = seed_dir / "table4_window_complete.csv"
        if window_path.exists() and not args.force:
            window_rows = _read_csv(window_path)
        else:
            window_args = SimpleNamespace(
                window_size=int(args.window_size),
                window_trials=int(args.window_trials),
                mmd_permutations=int(args.mmd_permutations),
                window_horizon=8,
            )
            window_rows = focused._window_rows(
                args=window_args,
                masks=_window_masks(run["rows"]),
                a_space=run["a"],
                b_space=run["b"],
                residual_scores=run["b_score"],
                probabilities=run["output"].probabilities,
                seed=int(seed),
            )
            for row in window_rows:
                row.update({"dataset": spec.name, "seed": int(seed)})
            _write_csv(window_path, window_rows)
        all_window.extend(window_rows)

        probe_path = seed_dir / "table5_probe_complete.csv"
        if probe_path.exists() and not args.force:
            probe_rows = _read_csv(probe_path)
        else:
            probe_rows = _probe_rows(
                run,
                spec=spec,
                repetitions=int(args.probe_repetitions),
                bootstrap_samples=int(args.probe_bootstrap),
                supplementary=supplementary,
            )
            _write_csv(probe_path, probe_rows)
        all_probe.extend(probe_rows)

        adapt_path = seed_dir / "table6_adaptation_complete.csv"
        if adapt_path.exists() and not args.force:
            adapt_rows = _read_csv(adapt_path)
        else:
            adapt_rows = _adaptation_rows(run, spec=spec, supplementary=supplementary)
            _write_csv(adapt_path, adapt_rows)
        all_adapt.extend(adapt_rows)

        e2e_rows = _end_to_end_rows(run, spec=spec, seed=int(seed))
        _write_csv(seed_dir / "table8_end_to_end.csv", e2e_rows)
        all_e2e.extend(e2e_rows)

    gate_rows = supplementary._gate_rows(all_adapt)
    for row in gate_rows:
        row["dataset"] = spec.name
    _write_csv(output / "table4_window_complete.csv", _aggregate_rows(all_window, "method", _window_metrics()))
    _write_csv(
        output / "table5_probe_complete.csv",
        _aggregate_rows(
            all_probe,
            ("localization", "scenario", "strategy"),
            ("classification_accuracy", "harmful_call_rate", "benign_call_rate", "uncertain_rate", "ci_coverage", "mean_labels"),
        ),
    )
    _write_csv(output / "table6_adaptation_complete.csv", all_adapt)
    _write_csv(output / "table7_gate_complete.csv", gate_rows)
    _write_csv(
        output / "table8_end_to_end.csv",
        _aggregate_rows(
            all_e2e,
            ("method", "stage"),
            (
                "harmful_detection_recall", "harmful_action_recall", "benign_specificity",
                "false_alarm_rate_per_100_stream_rows", "wrong_update_rate",
                "mean_detection_delay_samples", "label_cost",
            ),
        ),
    )
    baseline_path = output / "table2_local_text_baselines.csv"
    if spec.name == "ellipse" and (args.force or not baseline_path.exists()):
        _write_csv(baseline_path, _ellipse_text_baselines(spec, seeds=args.seeds))
    if spec.name == "asap" and (args.force or not baseline_path.exists()):
        _write_csv(baseline_path, _asap_text_baselines(spec, seeds=args.seeds))
    manifest = {
        "artifact_type": "docs_lifecycle_cpu_completion_v1",
        "dataset": spec.name,
        "seeds": [int(seed) for seed in args.seeds],
        "qwen_forward_passes": 0,
        "external_api_calls": 0,
        "hiddenstate_verified": all(bool(row["hiddenstate_verified"]) for row in run_audits),
        "run_audits": run_audits,
        "tables": {
            "table2_local_text_baselines": str(baseline_path) if baseline_path.exists() else None,
            "table4_window": str(output / "table4_window_complete.csv"),
            "table5_probe": str(output / "table5_probe_complete.csv"),
            "table6_adaptation": str(output / "table6_adaptation_complete.csv"),
            "table7_gate": str(output / "table7_gate_complete.csv"),
            "table8_end_to_end": str(output / "table8_end_to_end.csv"),
        },
        "parameters": {
            "window_size": int(args.window_size),
            "window_trials": int(args.window_trials),
            "mmd_permutations": int(args.mmd_permutations),
            "probe_repetitions": int(args.probe_repetitions),
            "probe_bootstrap": int(args.probe_bootstrap),
        },
        "protocol_notes": [
            "All completion experiments consume frozen hiddenstate caches.",
            "Adaptation is head-only; backbone LoRA is outside the frozen-cache CPU protocol.",
            "External API judges are excluded by scope.",
            "Oracle-localization rows are retrospective diagnostics and are marked as such.",
        ],
        "elapsed_seconds": float(time.perf_counter() - started),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "git_revision": _git("rev-parse", "HEAD"),
    }
    write_json(output / "manifest.json", manifest)
    return manifest


def _load_current_run(spec: DatasetSpec, seed: int) -> dict[str, Any]:
    result = spec.lifecycle_root / f"seed_{int(seed)}"
    rows = read_jsonl(spec.prepared_path)
    scores = read_jsonl(result / "sample_ood_scores.jsonl")
    lifecycle = read_jsonl(result / "document_cluster_lifecycle.jsonl")
    ledger = read_jsonl(result / "label_cost_ledger.jsonl")
    summary = json.loads((result / "summary.json").read_text(encoding="utf-8"))
    cache = np.load(spec.hidden_path, allow_pickle=True)
    features = np.asarray(cache["features"], dtype=np.float32)
    sample_ids = np.asarray(cache["sample_ids"]).astype(str)
    document_cache = np.load(spec.document_hidden_path, allow_pickle=True)
    document_features = np.asarray(document_cache["features"], dtype=np.float32)
    document_sample_ids = np.asarray(document_cache["sample_ids"]).astype(str)
    expected_ids = np.asarray([str(row["sample_id"]) for row in rows])
    if len(rows) != len(features) or not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError(f"Judge hiddenstate alignment failed for {spec.name}/seed_{seed}")
    if len(rows) != len(document_features) or not np.array_equal(document_sample_ids, expected_ids):
        raise RuntimeError(f"Document hiddenstate alignment failed for {spec.name}/seed_{seed}")
    if len(scores) != len(rows):
        raise RuntimeError(f"Score alignment failed for {spec.name}/seed_{seed}")
    layer = int(
        summary.get("feature_extractors", {})
        .get("judge_input", {})
        .get("separability_selected_layer_index", 0)
    )
    model = joblib.load(result / "judge_checkpoints/selected_linear_judge.joblib")
    query_ids = np.asarray([str(row["query_id"]) for row in rows])
    judge_preprocessor = np.load(result / "judge_preprocessor.npz", allow_pickle=False)
    judge_layers = []
    for layer_index in range(features.shape[1]):
        judge_layers.append(
            (
                (np.asarray(features[:, layer_index, :], dtype=np.float64) - judge_preprocessor["pca_means"][layer_index])
                @ judge_preprocessor["components"][layer_index].T
                / np.sqrt(np.maximum(judge_preprocessor["explained_variance"][layer_index], 1e-5))
            )
        )
    judge_features = np.stack(judge_layers, axis=1).astype(np.float32)
    output = model.predict_output(judge_features, query_ids)
    vim_payload = np.load(result / "judge_behavior_ood_scorer.npz", allow_pickle=False)
    rank = int(summary["behavior_main_representation"]["vim_rank"])
    vim = ViMScorer(rank=rank)
    vim.mean_ = np.asarray(vim_payload["source_mean"], dtype=np.float64)
    vim.components_ = np.asarray(vim_payload["principal_components"], dtype=np.float64)
    vim.fit_rows_ = int(sum(str(row["split"]) == "training_train" for row in rows))
    b = vim.residual_features(output.penultimate)
    b_score = vim.score(output.penultimate)
    ood_preprocessor = np.load(result / "ood_preprocessor.npz", allow_pickle=False)
    a = (
        (np.asarray(document_features[:, layer, :], dtype=np.float64) - ood_preprocessor["pca_means"][0])
        @ ood_preprocessor["components"][0].T
        / np.sqrt(np.maximum(ood_preprocessor["explained_variance"][0], 1e-5))
    ).astype(np.float32)
    return {
        "name": f"{spec.name}_seed_{seed}",
        "dataset": spec.name,
        "seed": int(seed),
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
        "a": a,
        "b": b,
        "b_score": b_score,
        "result_dir": str(result),
        "audit": {
            "seed": int(seed),
            "records": len(rows),
            "hiddenstate_verified": True,
            "hiddenstate_path": str(spec.hidden_path),
            "hiddenstate_sha256": _sha256(spec.hidden_path),
            "document_hiddenstate_path": str(spec.document_hidden_path),
            "document_hiddenstate_sha256": _sha256(spec.document_hidden_path),
            "prepared_sha256": _sha256(spec.prepared_path),
            "selected_layer_index": layer,
            "feature_shape": list(features.shape),
        },
    }


def _window_masks(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    split = np.asarray([str(row["split"]) for row in rows])
    is_ood = np.asarray([bool(row.get("is_document_ood", False)) for row in rows])
    calibration = (split == "training_calibration") & ~is_ood
    id_test = (split == "benchmark_test") & ~is_ood
    ood_test = (split == "benchmark_test") & is_ood
    if min(int(calibration.sum()), int(id_test.sum()), int(ood_test.sum())) < 100:
        raise RuntimeError("Lifecycle benchmark needs at least 100 calibration/ID/OOD rows")
    return {
        "train": split == "training_train",
        "calibration": calibration,
        "id_test": id_test,
        "ood_test": ood_test,
        "benchmark": id_test | ood_test,
        "excluded": np.zeros(len(rows), dtype=bool),
    }


def _probe_rows(
    run: dict[str, Any], *, spec: DatasetSpec, repetitions: int,
    bootstrap_samples: int, supplementary: Any,
) -> list[dict[str, Any]]:
    groups: list[tuple[str, str, str, np.ndarray]] = []
    for lifecycle in run["lifecycle"]:
        members = np.asarray(lifecycle.get("member_indices", []), dtype=int)
        if len(members) >= 4:
            cluster = str(lifecycle["document_cluster_id"])
            shift = _majority([str(run["rows"][index].get("document_shift_type")) for index in members])
            groups.append(("production_cluster", shift, cluster, members))
    stream = np.asarray([str(row["split"]) == "deployment_stream" for row in run["rows"]])
    is_ood = np.asarray([bool(row.get("is_document_ood", False)) for row in run["rows"]])
    audit_ids = np.asarray([str(row.get("audit_document_group_id")) for row in run["rows"]])
    for group in sorted(set(audit_ids[stream & is_ood].tolist())):
        members = np.flatnonzero(stream & is_ood & (audit_ids == group))
        if len(members) >= 4:
            shift = _majority([str(run["rows"][index].get("document_shift_type")) for index in members])
            groups.append(("oracle_audit_group", shift, group, members))
    output: list[dict[str, Any]] = []
    for localization, scenario, group, members in groups:
        full = supplementary._probe_result_local(
            run, members, seed=run["seed"] + 700, n_boot=max(bootstrap_samples, 500)
        )
        if localization == "oracle_audit_group":
            audit_truth = run["summary"].get("audit_document_group_harmfulness", {}).get(group, {})
            full_status = str(audit_truth.get("status") or "uncertain")
        else:
            full_status = supplementary._three_way_probe_status(full, benign_margin=0.10)
        strategies = (
            ("Fixed Random Probe", "random", repetitions, 20),
            ("Confidence Sampling", "confidence", 1, 20),
            ("Residual-stratified Probe", "residual_stratified", repetitions, 20),
            ("Sequential Random Probe", "random", repetitions, 40),
            ("Full-label Oracle", "full", 1, len(members)),
        )
        for display, strategy, local_repetitions, budget in strategies:
            correct = harmful = benign = uncertain = coverage = 0
            errors: list[float] = []
            costs: list[int] = []
            for repetition in range(local_repetitions):
                rng = np.random.default_rng(_stable_seed(f"{run['name']}:{localization}:{group}:{display}:{repetition}"))
                if strategy == "full":
                    selected = members
                elif display == "Sequential Random Probe":
                    order = rng.permutation(members)
                    selected = order[: min(40, len(order))]
                    for look in (10, 20, 30, 40):
                        candidate = order[: min(look, len(order))]
                        result = supplementary._probe_result_local(
                            run, candidate,
                            seed=_stable_seed(f"seq:{run['name']}:{group}:{repetition}:{look}"),
                            n_boot=bootstrap_samples,
                        )
                        selected = candidate
                        if supplementary._three_way_probe_status(result, benign_margin=0.10) != "uncertain" or len(candidate) == len(order):
                            break
                else:
                    selected = supplementary._probe_sample_local(
                        run, members, strategy=strategy,
                        count=min(budget, len(members)), rng=rng,
                    )
                result = supplementary._probe_result_local(
                    run, selected,
                    seed=_stable_seed(f"probe:{run['name']}:{group}:{display}:{repetition}"),
                    n_boot=bootstrap_samples,
                )
                status = supplementary._three_way_probe_status(result, benign_margin=0.10)
                if strategy == "full":
                    status = full_status
                correct += int(status == full_status)
                harmful += int(status == "harmful")
                benign += int(status == "benign")
                uncertain += int(status == "uncertain")
                if result.get("harm_delta") is not None and full.get("harm_delta") is not None:
                    errors.append(abs(float(result["harm_delta"]) - float(full["harm_delta"])))
                if (
                    result.get("harm_delta_lcb") is not None
                    and result.get("harm_delta_ucb") is not None
                    and full.get("harm_delta") is not None
                ):
                    coverage += int(
                        float(result["harm_delta_lcb"])
                        <= float(full["harm_delta"])
                        <= float(result["harm_delta_ucb"])
                    )
                costs.append(len(selected))
            output.append({
                "dataset": spec.name, "seed": run["seed"], "localization": localization,
                "scenario": scenario, "group": group, "strategy": display,
                "full_label_status": full_status, "group_documents": len(members),
                "classification_accuracy": correct / local_repetitions,
                "harmful_call_rate": harmful / local_repetitions,
                "benign_call_rate": benign / local_repetitions,
                "uncertain_rate": uncertain / local_repetitions,
                "delta_abs_error_mean": float(np.mean(errors)) if errors else "",
                "ci_coverage": coverage / local_repetitions,
                "mean_labels": float(np.mean(costs)),
                "repetitions": local_repetitions,
                "evaluation_scope": "deployable" if localization == "production_cluster" else "oracle_localization_diagnostic",
            })
        if localization == "production_cluster":
            warning = run["summary"].get("behavior_warning", {}).get("by_predicted_document_cluster", {}).get(group, {})
            for strategy, key, available in (
                ("ATC", "atc_estimated_accuracy_drop", bool(warning.get("atc_trigger_eligible"))),
                ("DoC", "doc_estimated_accuracy_drop", warning.get("doc_estimated_accuracy_drop") is not None),
            ):
                drop = warning.get(key)
                status = "harmful" if available and float(drop) >= 0.05 else "benign" if available else "uncertain"
                output.append({
                    "dataset": spec.name, "seed": run["seed"], "localization": localization,
                    "scenario": scenario, "group": group, "strategy": strategy,
                    "full_label_status": full_status, "group_documents": len(members),
                    "classification_accuracy": float(status == full_status),
                    "harmful_call_rate": float(status == "harmful"),
                    "benign_call_rate": float(status == "benign"),
                    "uncertain_rate": float(status == "uncertain"),
                    "ci_coverage": "", "mean_labels": 0.0, "repetitions": 1,
                    "estimated_accuracy_drop": drop if available else "",
                    "evaluation_scope": "unlabeled_proxy_diagnostic",
                })
    return output


def _adaptation_rows(run: dict[str, Any], *, spec: DatasetSpec, supplementary: Any) -> list[dict[str, Any]]:
    rows = run["rows"]
    split = np.asarray([str(row["split"]) for row in rows])
    shift = np.asarray([str(row.get("document_shift_type")) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    queries = np.asarray([str(row["query_id"]) for row in rows])
    old = np.asarray([row["judge_prediction"] for row in run["scores"]])
    guard = np.flatnonzero(split == "training_guard")
    train = np.flatnonzero(split == "training_train")
    u = run["model"].transform_u(run["judge_features"])
    metadata = run["summary"].get("adaptation", {}).get("adapter") or {}
    base = HeadAdaptConfig(**(metadata.get("config") or {}))
    variants = (
        ("No Update", None, 0, False),
        ("Target-only Head FT", replace(base, training_replay_weight=0.0, anchor_weight=0.0), 20, False),
        ("Head + Replay", replace(base, anchor_weight=0.0), 20, True),
        ("Head + Replay + Anchor", base, 20, True),
        ("Full-label Oracle Head", base, -1, True),
    )
    output: list[dict[str, Any]] = []
    for scenario in spec.scenarios:
        adapt_pool = np.flatnonzero((split == "deployment_adapt") & (shift == scenario))
        gate = np.flatnonzero((split == "deployment_gate") & (shift == scenario))
        future = np.flatnonzero((split == "deployment_future_test") & (shift == scenario))
        if min(len(adapt_pool), len(gate), len(future), len(guard)) == 0:
            raise RuntimeError(f"Empty Adapt/Gate/Future scope for {spec.name}/{scenario}/seed_{run['seed']}")
        for name, config, requested_budget, use_replay in variants:
            if config is None:
                prediction = old.copy()
                target = np.asarray([], dtype=int)
                replay = np.asarray([], dtype=int)
            else:
                count = len(adapt_pool) if requested_budget < 0 else min(requested_budget, len(adapt_pool))
                rng = np.random.default_rng(_stable_seed(f"adapt:{run['name']}:{scenario}:{name}"))
                target = np.asarray(rng.choice(adapt_pool, size=count, replace=False), dtype=int)
                replay_count = min(max(count, 20), len(train)) if use_replay else 0
                replay = np.asarray(rng.choice(train, size=replay_count, replace=False), dtype=int) if replay_count else np.asarray([], dtype=int)
                adapter = HeadAdapter(config).fit(
                    u_features=u, labels=labels, query_ids=queries,
                    deployment_indices=target, training_replay_indices=replay,
                    class_values=run["model"].classes_, judge=run["model"],
                )
                prediction = adapter.predict(u_features=u, query_ids=queries, fallback=old)
            class_values = run["model"].classes_
            before = judge_metrics(labels[future], old[future], class_values=class_values)
            after = judge_metrics(labels[future], prediction[future], class_values=class_values)
            source_before = judge_metrics(labels[guard], old[guard], class_values=class_values)
            source_after = judge_metrics(labels[guard], prediction[guard], class_values=class_values)
            gate_values = np.abs(old[gate].astype(float) - labels[gate].astype(float)) - np.abs(prediction[gate].astype(float) - labels[gate].astype(float))
            gate_gain = float(np.mean(gate_values))
            gate_lcb = supplementary._mean_lcb(
                gate_values, seed=_stable_seed(f"gate:{run['name']}:{scenario}:{name}"), samples=1000
            )
            nfr = float(np.mean((prediction[guard] != labels[guard]) & (old[guard] == labels[guard])))
            qwk_drop = float(source_before["qwk"] - source_after["qwk"])
            future_gain = float(before["mae"] - after["mae"])
            status = "baseline" if config is None else "ok"
            output.append({
                "dataset": spec.name, "seed": run["seed"], "run": run["name"],
                "scenario": scenario, "candidate": name, "target_mode": "oracle_shift_scope",
                "target_documents": len(target), "replay_documents": len(replay),
                "future_documents": len(future), "future_mae_before": before["mae"],
                "future_mae_after": after["mae"], "future_mae_gain": future_gain,
                "future_qwk_before": before["qwk"], "future_qwk_after": after["qwk"],
                "gate_documents": len(gate), "gate_gain": gate_gain, "gate_gain_lcb": gate_lcb,
                "source_nfr": nfr, "source_qwk_drop": qwk_drop,
                "oracle_good_candidate": bool(future_gain >= 0.1 and nfr <= 0.05 and qwk_drop <= 0.02),
                "full_gate_accept": bool(gate_gain >= 0.1 and gate_lcb > 0.0 and nfr <= 0.05 and qwk_drop <= 0.02),
                "status": status, "formal_candidate_eligible": False,
                "formal_gate_eligible": False,
                "uses_future_labels_for_candidate_truth": True,
                "evaluation_scope": "retrospective_oracle_shift_ablation",
                **{
                    f"gate_accept_rate_n{size}": supplementary._gate_sample_accept_rate(
                        gate_values, sample_size=size, source_nfr=nfr, source_qwk_drop=qwk_drop,
                        seed=_stable_seed(f"gate-size:{run['name']}:{scenario}:{name}:{size}"),
                    )
                    for size in (10, 20, 30, 50)
                },
            })
    return output


def _end_to_end_rows(run: dict[str, Any], *, spec: DatasetSpec, seed: int) -> list[dict[str, Any]]:
    path = Path(run["result_dir"]) / "tables" / "table5_monitoring_baselines.csv"
    rows = _read_csv(path)
    for row in rows:
        row.update({"dataset": spec.name, "seed": int(seed)})
    return rows


def _ellipse_text_baselines(spec: DatasetSpec, *, seeds: Iterable[int]) -> list[dict[str, Any]]:
    rows = read_jsonl(spec.prepared_path)
    split = np.asarray([str(row["split"]) for row in rows])
    train = np.flatnonzero(split == "training_train")
    validation = np.flatnonzero(split == "training_validation")
    test = np.flatnonzero(split == "training_test")
    texts = [str(row["input_document_text"]) for row in rows]
    labels = np.asarray([float(row["label"]) for row in rows])
    classes = np.unique(labels[train])
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, max_features=6000,
        sublinear_tf=True, strip_accents="unicode",
    )
    x_train_sparse = vectorizer.fit_transform([texts[index] for index in train])
    x_validation_sparse = vectorizer.transform([texts[index] for index in validation])
    x_test_sparse = vectorizer.transform([texts[index] for index in test])
    length_train = _length_features([texts[index] for index in train])
    length_validation = _length_features([texts[index] for index in validation])
    length_test = _length_features([texts[index] for index in test])
    x_train = sparse.hstack([x_train_sparse, sparse.csr_matrix(length_train)], format="csr")
    x_validation = sparse.hstack([x_validation_sparse, sparse.csr_matrix(length_validation)], format="csr")
    x_test = sparse.hstack([x_test_sparse, sparse.csr_matrix(length_test)], format="csr")
    best: tuple[float, float, float] | None = None
    for c in (0.01, 0.1, 1.0, 10.0):
        for epsilon in (0.0, 0.1, 0.25):
            model = SVR(kernel="linear", C=c, epsilon=epsilon).fit(x_train, labels[train])
            pred = _nearest_class(model.predict(x_validation), classes)
            mae = float(np.mean(np.abs(pred - labels[validation])))
            candidate = (mae, c, epsilon)
            if best is None or candidate < best:
                best = candidate
    assert best is not None
    svr = SVR(kernel="linear", C=best[1], epsilon=best[2]).fit(x_train, labels[train])
    svr_pred = _nearest_class(svr.predict(x_test), classes)
    dense_train = x_train.toarray().astype(np.float32)
    dense_test = x_test.toarray().astype(np.float32)
    blrr = BayesianRidge().fit(dense_train, labels[train])
    blrr_pred = _nearest_class(blrr.predict(dense_test), classes)
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for method, pred, details in (
            ("EASE-style SVR", svr_pred, {"selected_c": best[1], "selected_epsilon": best[2]}),
            ("EASE-style BLRR", blrr_pred, {}),
        ):
            metrics = judge_metrics(labels[test], pred, class_values=classes)
            output.append({
                "dataset": spec.name, "seed": int(seed), "method": method,
                "train_documents": len(train), "validation_documents": len(validation),
                "test_documents": len(test), "qwk": metrics["qwk"],
                "mae": metrics["mae"], "spearman": metrics["spearman"],
                "accuracy": metrics["accuracy"],
                "protocol": "source-only TF-IDF 1-2gram + length features; validation-selected",
                **details,
            })
    return output


def _asap_text_baselines(spec: DatasetSpec, *, seeds: Iterable[int]) -> list[dict[str, Any]]:
    source_path = spec.lifecycle_root / "judge_baselines" / "judge_baselines.csv"
    rows = _read_csv(source_path)
    selected = [
        row for row in rows
        if str(row.get("scope")) == "pooled_test"
        and str(row.get("status")) == "complete"
        and str(row.get("method")) in {"ease_svr", "ease_blrr", "qwen_frozen_linear_judge"}
    ]
    if len(selected) != 3:
        raise RuntimeError(f"Expected three completed ASAP local Judge baselines, got {len(selected)}")
    output: list[dict[str, Any]] = []
    for seed in seeds:
        for row in selected:
            output.append({
                "dataset": spec.name,
                "seed": int(seed),
                "method": row["method"],
                "test_documents": int(row["documents"]),
                "qwk": float(row["qwk"]),
                "mae": float(row["mae"]),
                "spearman": float(row["spearman"]),
                "protocol": "matched source train/validation/test; local CPU only",
            })
    return output


def _length_features(texts: list[str]) -> np.ndarray:
    values = []
    for text in texts:
        words = text.split()
        sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
        unique = len(set(word.lower() for word in words))
        values.append((len(words), len(text), unique, len(words) / sentences))
    matrix = np.asarray(values, dtype=np.float64)
    mean = np.maximum(matrix.mean(axis=0), 1.0)
    return np.log1p(matrix) / np.log1p(mean)


def _nearest_class(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return classes[np.argmin(np.abs(array[:, None] - classes[None, :]), axis=1)]


def _aggregate_rows(rows: list[dict[str, Any]], keys: str | tuple[str, ...], metrics: Iterable[str]) -> list[dict[str, Any]]:
    key_tuple = (keys,) if isinstance(keys, str) else keys
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row.get(key, "")) for key in key_tuple), []).append(row)
    output: list[dict[str, Any]] = []
    for values, local in sorted(groups.items()):
        summary: dict[str, Any] = {key: value for key, value in zip(key_tuple, values, strict=True)}
        summary["n_rows"] = len(local)
        summary["seeds"] = sorted({int(float(row["seed"])) for row in local if str(row.get("seed", "")).strip()})
        for metric in metrics:
            numeric = [_to_float(row.get(metric)) for row in local]
            numeric = [value for value in numeric if value is not None and np.isfinite(value)]
            if numeric:
                array = np.asarray(numeric, dtype=float)
                summary[f"{metric}_mean"] = float(array.mean())
                summary[f"{metric}_std"] = float(array.std(ddof=1)) if len(array) > 1 else 0.0
        output.append(summary)
    return output


def _window_metrics() -> tuple[str, ...]:
    return (
        "type_i_at_0_05", "fwer", "power_5pct", "power_10pct",
        "power_20pct", "runtime_sec_per_window",
    )


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Refusing to write empty result table: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _to_float(value: Any) -> float | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _majority(values: list[str]) -> str:
    unique, counts = np.unique(np.asarray(values).astype(str), return_counts=True)
    return str(unique[int(np.argmax(counts))])


def _stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


if __name__ == "__main__":
    main()
