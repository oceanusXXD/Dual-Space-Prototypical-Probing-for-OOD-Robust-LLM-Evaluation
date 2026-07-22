#!/usr/bin/env python3
"""Run the non-API, frozen-hiddenstate Table 4-8 formal CPU protocol.

The runner keeps candidate-development and candidate-evaluation documents
disjoint, freezes Gate thresholds on development candidates, and evaluates the
final Gate only on evaluation candidates.  It never executes the Qwen backbone.
True backbone LoRA is intentionally outside this CPU/cache protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.adapt.head import HeadAdaptConfig, HeadAdapter
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge
from src.llm_judge_ood.scores.vim import ViMScorer
from src.llm_judge_ood.shared.metrics import judge_metrics


STATUS_ORDER = ("harmful", "benign", "uncertain")
GATE_VARIANTS = (
    "no_gate",
    "gain_only",
    "gain_plus_confidence",
    "gain_confidence_nfr",
    "full_gate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["ellipse", "asap", "clinc150", "rostd"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-root", default="artifacts/docs_experiments/formal_tables4_8")
    parser.add_argument("--window-sizes", nargs="+", type=int, default=[50, 100, 200])
    parser.add_argument("--window-methods", nargs="+", choices=["A-MMD", "B-MMD", "C2ST", "KS", "BBSDs", "BBSDh"], default=["A-MMD", "B-MMD"])
    parser.add_argument("--window-trials", type=int, default=60)
    parser.add_argument("--mmd-permutations", type=int, default=99)
    parser.add_argument("--episode-trials", type=int, default=1000)
    parser.add_argument("--probe-repetitions", type=int, default=100)
    parser.add_argument("--probe-budget", type=int, default=20)
    parser.add_argument("--adapt-budget", type=int, default=20)
    parser.add_argument("--only-table4", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    focused = _load_module("docs_focused_formal", ROOT / "scripts/llm_judge_ood/36_run_docs_focused_standard_ood.py")
    lifecycle = _load_module("docs_lifecycle_formal", ROOT / "scripts/llm_judge_ood/37_run_docs_cpu_lifecycle_completion.py")
    supplementary = _load_module("docs_supplementary_formal", ROOT / "scripts/llm_judge_ood/32_run_cpu_supplementary.py")
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    table4 = _run_table4(args, output, focused, lifecycle)
    if args.only_table4:
        print(json.dumps({"table4": str(output / "table4_window_formal.csv"), "rows": len(table4)}, indent=2))
        return
    lifecycle_names = [name for name in ("ellipse", "asap") if name in args.datasets]
    runs = {
        name: [lifecycle._load_current_run(lifecycle.DATASETS[name], seed) for seed in args.seeds]
        for name in lifecycle_names
    }
    table5_detail, table5 = _run_table5(args, output, runs, supplementary)
    table6 = _run_table6(args, output, runs, supplementary)
    thresholds, table7 = _run_table7(output, table6)
    table8 = _run_table8(args, output, table4, table5, table6, thresholds)

    manifest = {
        "artifact_type": "docs_tables4_8_formal_cpu_v1",
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "qwen_forward_passes": 0,
        "external_api_calls": 0,
        "ag_news_included": False,
        "hiddenstate_verified": True,
        "formal_candidate_protocol": True,
        "candidate_development_evaluation_disjoint": True,
        "gate_thresholds_frozen_before_evaluation": True,
        "equal_label_budget_non_oracle": int(args.probe_budget + args.adapt_budget),
        "backbone_lora_included": False,
        "backbone_lora_reason": "requires backbone training and cannot be derived from frozen hiddenstates",
        "parameters": vars(args),
        "tables": {
            "table4": str(output / "table4_window_formal.csv"),
            "table5_detail": str(output / "table5_probe_detail.csv"),
            "table5": str(output / "table5_probe_formal.csv"),
            "table6": str(output / "table6_adaptation_formal.csv"),
            "table6_summary": str(output / "table6_adaptation_summary.csv"),
            "table7_thresholds": str(output / "table7_frozen_thresholds.json"),
            "table7": str(output / "table7_gate_formal.csv"),
            "table8": str(output / "table8_equal_budget_formal.csv"),
        },
        "row_counts": {
            "table4": len(table4), "table5_detail": len(table5_detail),
            "table5": len(table5), "table6": len(table6),
            "table7": len(table7), "table8": len(table8),
        },
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def _run_table4(args: argparse.Namespace, output: Path, focused: Any, lifecycle: Any) -> list[dict[str, Any]]:
    path = output / "table4_window_formal.csv"
    if path.exists() and not args.force:
        return lifecycle._read_csv(path)
    detail: list[dict[str, Any]] = []
    checkpoint_root = output / "table4_checkpoints"

    def checked_window_grid(
        dataset: str, seed: int, masks: dict[str, np.ndarray], a_space: np.ndarray,
        b_space: np.ndarray, residual: np.ndarray, probabilities: np.ndarray,
    ) -> list[dict[str, Any]]:
        local_rows: list[dict[str, Any]] = []
        for size in args.window_sizes:
            checkpoint = checkpoint_root / f"{dataset}_seed_{int(seed)}_n{int(size)}.csv"
            if checkpoint.exists() and not args.force:
                values = lifecycle._read_csv(checkpoint)
            else:
                local_args = SimpleNamespace(**vars(args))
                local_args.window_sizes = [int(size)]
                values = _window_grid(
                    local_args, focused, masks, a_space, b_space, residual,
                    probabilities, dataset, int(seed),
                )
                lifecycle._write_csv(checkpoint, values)
            local_rows.extend(values)
        return local_rows

    for dataset in args.datasets:
        expected_checkpoints = [
            checkpoint_root / f"{dataset}_seed_{int(seed)}_n{int(size)}.csv"
            for seed in args.seeds for size in args.window_sizes
        ]
        if not args.force and expected_checkpoints and all(path.exists() for path in expected_checkpoints):
            for checkpoint in expected_checkpoints:
                detail.extend(lifecycle._read_csv(checkpoint))
            continue
        if dataset in lifecycle.DATASETS:
            for seed in args.seeds:
                run = lifecycle._load_current_run(lifecycle.DATASETS[dataset], int(seed))
                detail.extend(checked_window_grid(
                    dataset, int(seed), lifecycle._window_masks(run["rows"]),
                    run["a"], run["b"], run["b_score"], run["output"].probabilities,
                ))
        elif dataset in focused.DATASETS:
            spec = focused.DATASETS[dataset]
            rows = read_jsonl(spec.prepared_path)
            document = focused._load_cache(spec.document_hidden_path, rows, "document")["features"]
            judge_features = focused._load_cache(spec.judge_hidden_path, rows, "judge")["features"]
            masks = focused._dataset_masks(spec, rows)
            focused._validate_masks(masks)
            labels = np.asarray([row["label"] for row in rows]).astype(str)
            query_ids = np.asarray([str(row["query_id"]) for row in rows])
            classes = np.unique(labels[masks["train"]])
            for seed in args.seeds:
                judge = PerQueryLinearJudge(LinearJudgeConfig(
                    method="linear", representation="pca", pca_dim=128,
                    class_values=tuple(classes.tolist()), seed=int(seed),
                    learning_rate=1e-3, weight_decay=1e-4, epochs=35,
                    batch_size=512, patience=6, device="cpu",
                    class_weight="balanced", head_sharing="shared",
                )).fit(judge_features, labels, query_ids, train_mask=masks["train"], validation_mask=masks["calibration"])
                judge_output = judge.predict_output(judge_features, query_ids)
                a_space = focused._source_pca_space(document, fit_mask=masks["train"], pca_dim=128, seed=int(seed))
                scorer = ViMScorer(rank=focused._vim_rank(judge_output.penultimate)).fit(judge_output.penultimate[masks["train"]])
                detail.extend(checked_window_grid(
                    dataset, int(seed), masks, a_space,
                    scorer.residual_features(judge_output.penultimate),
                    scorer.score(judge_output.penultimate), judge_output.probabilities,
                ))
        else:
            raise ValueError(f"Unsupported Table 4 dataset: {dataset}")
    summary = _summarize_table4(detail)
    lifecycle._write_csv(output / "table4_window_detail.csv", detail)
    lifecycle._write_csv(path, summary)
    return summary


def _window_grid(
    args: argparse.Namespace, focused: Any, masks: dict[str, np.ndarray],
    a_space: np.ndarray, b_space: np.ndarray, residual: np.ndarray,
    probabilities: np.ndarray, dataset: str, seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for size in args.window_sizes:
        local = SimpleNamespace(
            window_size=int(size), window_trials=int(args.window_trials),
            mmd_permutations=int(args.mmd_permutations), window_horizon=8,
            ood_rates=(0.0, 0.05, 0.10, 0.20, 1.0), formal_sequential=True,
            episode_trials=int(args.episode_trials), minimum_consecutive_windows=2,
            window_methods=tuple(args.window_methods),
        )
        values = focused._window_rows(
            args=local, masks=masks, a_space=a_space, b_space=b_space,
            residual_scores=residual, probabilities=probabilities, seed=int(seed),
        )
        for row in values:
            row.update({"dataset": dataset, "seed": int(seed), "formal_protocol": True})
        rows.extend(values)
    return rows


def _summarize_table4(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = (
        "type_i_at_0_05", "sequential_fwer", "power_5pct", "power_10pct",
        "power_20pct", "power_100pct", "detection_rate_5pct",
        "detection_rate_10pct", "detection_rate_20pct", "delay_windows_10pct",
        "transient_false_persistence", "runtime_sec_per_window",
    )
    normalized: list[dict[str, Any]] = []
    for item in rows:
        row = dict(item)
        row.update(_persistence_metrics_from_rates(
            null_rate=float(row["type_i_at_0_05"]),
            rates={
                "5pct": float(row["power_5pct"]), "10pct": float(row["power_10pct"]),
                "20pct": float(row["power_20pct"]), "100pct": float(row["power_100pct"]),
            },
            horizon=8, minimum_consecutive=2,
        ))
        normalized.append(row)
    output: list[dict[str, Any]] = []
    for key, local in _group(normalized, ("dataset", "method", "window_size")).items():
        row: dict[str, Any] = {"dataset": key[0], "method": key[1], "window_size": int(float(key[2])), "seeds": sorted({int(item["seed"]) for item in local})}
        for metric in metrics:
            values = _numeric(item.get(metric) for item in local)
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        row["formal_protocol"] = True
        row["sequential_rule"] = "p<=0.05 for two consecutive windows"
        output.append(row)
    for row in output:
        same = [item for item in output if item["dataset"] == row["dataset"] and item["method"] == row["method"]]
        for label in ("5pct", "10pct", "20pct"):
            eligible = [item["window_size"] for item in same if float(item.get(f"power_{label}_mean", 0.0)) >= 0.8]
            row[f"n_at_80_{label}"] = min(eligible) if eligible else "not_achieved_within_grid"
    return output


def _persistence_metrics_from_rates(
    *, null_rate: float, rates: dict[str, float], horizon: int, minimum_consecutive: int,
) -> dict[str, float]:
    """Propagate empirical per-window rates through a fixed persistence rule."""
    def episode(probabilities: list[float]) -> tuple[float, float]:
        state = np.zeros(minimum_consecutive, dtype=float)
        state[0] = 1.0
        confirmed = 0.0
        weighted_delay = 0.0
        for look, probability in enumerate(probabilities, start=1):
            probability = min(max(float(probability), 0.0), 1.0)
            next_state = np.zeros_like(state)
            next_state[0] += float(state.sum()) * (1.0 - probability)
            for run_length, mass in enumerate(state):
                if run_length + 1 >= minimum_consecutive:
                    new = float(mass) * probability
                    confirmed += new
                    weighted_delay += look * new
                else:
                    next_state[run_length + 1] += float(mass) * probability
            state = next_state
        delay = weighted_delay / confirmed if confirmed > 0.0 else float("nan")
        return confirmed, delay

    fwer, _ = episode([null_rate] * horizon)
    output: dict[str, float] = {"sequential_fwer": fwer}
    for label, rate in rates.items():
        detection, delay = episode([rate] * horizon)
        output[f"detection_rate_{label}"] = detection
        output[f"delay_windows_{label}"] = delay
    transient, _ = episode([rates["20pct"]] * min(2, horizon) + [null_rate] * max(0, horizon - 2))
    output["transient_false_persistence"] = transient
    return output


def _run_table5(
    args: argparse.Namespace, output: Path, runs: dict[str, list[dict[str, Any]]], supplementary: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_path = output / "table5_probe_detail.csv"
    summary_path = output / "table5_probe_formal.csv"
    if detail_path.exists() and summary_path.exists() and not args.force:
        module = _load_module("docs_lifecycle_reader", ROOT / "scripts/llm_judge_ood/37_run_docs_cpu_lifecycle_completion.py")
        return module._read_csv(detail_path), module._read_csv(summary_path)
    detail: list[dict[str, Any]] = []
    for dataset, dataset_runs in runs.items():
        for run in dataset_runs:
            detail.extend(_probe_detail_rows(args, run, dataset))
    summary = _probe_summary(detail)
    lifecycle = _load_module("docs_lifecycle_writer", ROOT / "scripts/llm_judge_ood/37_run_docs_cpu_lifecycle_completion.py")
    lifecycle._write_csv(detail_path, detail)
    lifecycle._write_csv(summary_path, summary)
    return detail, summary


def _probe_detail_rows(args: argparse.Namespace, run: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    rows = run["rows"]
    split = np.asarray([str(row["split"]) for row in rows])
    is_ood = np.asarray([bool(row.get("is_document_ood", False)) for row in rows])
    groups = np.asarray([str(row.get("audit_document_group_id")) for row in rows])
    labels = np.asarray([float(row["label"]) for row in rows])
    predictions = np.asarray([float(row["judge_prediction"]) for row in run["scores"]])
    losses = np.abs(predictions - labels)
    confidence = np.asarray([float(row["judge_confidence"]) for row in run["scores"]])
    residual = np.asarray(run["b_score"], dtype=float)
    guard = np.flatnonzero(split == "training_guard")
    source_risk = float(np.mean(losses[guard]))
    source_confidence = float(np.mean(confidence[guard]))
    source_correct = predictions[guard] == labels[guard]
    atc_threshold = float(np.quantile(confidence[guard], max(0.0, min(1.0, 1.0 - float(np.mean(source_correct))))))
    slope = float(np.polyfit(confidence[guard], losses[guard], 1)[0]) if len(guard) > 2 else -1.0
    candidates: list[tuple[str, np.ndarray]] = []
    stream = (split == "deployment_stream") & is_ood
    for group in sorted(set(groups[stream].tolist())):
        members = np.flatnonzero(stream & (groups == group))
        if len(members) >= 4:
            candidates.append((group, members))
    scenario_by_group = _probe_scenarios(dataset, run, candidates)
    output: list[dict[str, Any]] = []
    looks = (5, 10, int(args.probe_budget))
    for group, members in candidates:
        true_delta = float(np.mean(losses[members]) - source_risk)
        truth = _risk_status(true_delta, true_delta, true_delta)
        scenario = scenario_by_group[group]
        for strategy in ("ATC", "DoC", "Fixed Random Probe", "Confidence Sampling", "Residual-stratified Probe", "Sequential Random Probe", "Full-label Oracle"):
            repetitions = 1 if strategy in {"ATC", "DoC", "Full-label Oracle"} else int(args.probe_repetitions)
            for repetition in range(repetitions):
                rng = np.random.default_rng(_stable_seed(f"probe:{run['name']}:{group}:{strategy}:{repetition}"))
                if strategy == "ATC":
                    estimated = float(np.mean(confidence[members] < atc_threshold) - np.mean(confidence[guard] < atc_threshold))
                    low = high = estimated
                    used = 0
                elif strategy == "DoC":
                    estimated = float(-slope * (source_confidence - float(np.mean(confidence[members]))))
                    low = high = estimated
                    used = 0
                elif strategy == "Full-label Oracle":
                    estimated = low = high = true_delta
                    used = len(members)
                elif strategy == "Sequential Random Probe":
                    order = rng.permutation(members)
                    estimated = low = high = float("nan")
                    used = 0
                    for look_index, count in enumerate(looks):
                        selected = order[: min(count, len(order))]
                        estimated, low, high, _ = _risk_interval(losses, selected, source_risk, alpha=0.05 / len(looks))
                        used = len(selected)
                        if _risk_status(estimated, low, high) != "uncertain" or used == len(order):
                            break
                elif strategy == "Fixed Random Probe":
                    selected = rng.choice(members, size=min(int(args.probe_budget), len(members)), replace=False)
                    estimated, low, high, _ = _risk_interval(losses, selected, source_risk)
                    used = len(selected)
                else:
                    values = confidence if strategy == "Confidence Sampling" else residual
                    allocation = (0.40, 0.30, 0.20, 0.10) if strategy == "Confidence Sampling" else (0.25, 0.25, 0.25, 0.25)
                    estimated, low, high, used = _poststratified_interval(
                        losses, members, values, source_risk, int(args.probe_budget), allocation, rng,
                    )
                predicted = truth if strategy == "Full-label Oracle" else _risk_status(estimated, low, high)
                output.append({
                    "dataset": dataset, "seed": run["seed"], "group": group,
                    "scenario": scenario, "oracle_status": truth, "strategy": strategy,
                    "estimated_status": predicted, "true_risk_delta": true_delta,
                    "estimated_risk_delta": estimated, "ci_low": low, "ci_high": high,
                    "ci_covered": float(low <= true_delta <= high), "labels_used": int(used),
                    "group_documents": len(members), "weighted_correction": strategy in {"Confidence Sampling", "Residual-stratified Probe"},
                    "anytime_correction": "Bonferroni three-look confidence sequence" if strategy == "Sequential Random Probe" else "",
                    "formal_protocol": True,
                })
    return output


def _probe_scenarios(dataset: str, run: dict[str, Any], groups: list[tuple[str, np.ndarray]]) -> dict[str, str]:
    if dataset == "asap":
        return {group: _majority([str(run["rows"][index].get("document_shift_type")) for index in members]) for group, members in groups}
    split = np.asarray([str(row["split"]) for row in run["rows"]])
    source = np.flatnonzero(split == "training_calibration")
    center = np.mean(run["a"][source], axis=0)
    distances = {group: float(np.linalg.norm(np.mean(run["a"][members], axis=0) - center)) for group, members in groups}
    median = float(np.median(list(distances.values())))
    return {group: "near" if value <= median else "far" for group, value in distances.items()}


def _risk_interval(losses: np.ndarray, selected: np.ndarray, source_risk: float, alpha: float = 0.05) -> tuple[float, float, float, float]:
    values = np.asarray(losses[selected], dtype=float)
    estimate = float(np.mean(values) - source_risk)
    se = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("inf")
    radius = float(norm.ppf(1.0 - alpha / 2.0) * se) if np.isfinite(se) else float("inf")
    return estimate, estimate - radius, estimate + radius, se


def _poststratified_interval(
    losses: np.ndarray, members: np.ndarray, scores: np.ndarray, source_risk: float,
    budget: int, allocation: tuple[float, ...], rng: np.random.Generator,
) -> tuple[float, float, float, int]:
    order = members[np.argsort(scores[members])]
    strata = [np.asarray(part, dtype=int) for part in np.array_split(order, len(allocation)) if len(part)]
    remaining = min(int(budget), len(members))
    counts = [min(len(stratum), max(1, int(round(remaining * allocation[index])))) for index, stratum in enumerate(strata)]
    while sum(counts) > remaining:
        index = int(np.argmax(counts))
        if counts[index] > 1:
            counts[index] -= 1
        else:
            break
    while sum(counts) < remaining:
        choices = [index for index, stratum in enumerate(strata) if counts[index] < len(stratum)]
        if not choices:
            break
        counts[choices[0]] += 1
    estimate = 0.0
    variance = 0.0
    used = 0
    for stratum, count in zip(strata, counts, strict=True):
        selected = rng.choice(stratum, size=count, replace=False)
        weight = len(stratum) / len(members)
        values = losses[selected]
        estimate += weight * float(np.mean(values))
        if count > 1:
            variance += weight * weight * float(np.var(values, ddof=1)) / count
        used += count
    delta = float(estimate - source_risk)
    radius = float(norm.ppf(0.975) * math.sqrt(max(variance, 0.0)))
    return delta, delta - radius, delta + radius, used


def _risk_status(estimate: float, low: float, high: float) -> str:
    if low > 0.15:
        return "harmful"
    if high < 0.10:
        return "benign"
    return "uncertain"


def _probe_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for key, local in _group(rows, ("dataset", "scenario", "strategy")).items():
        truth = [str(row["oracle_status"]) for row in local]
        predicted = [str(row["estimated_status"]) for row in local]
        labels = list(STATUS_ORDER)
        errors = np.asarray([float(row["estimated_risk_delta"]) - float(row["true_risk_delta"]) for row in local])
        widths = np.asarray([float(row["ci_high"]) - float(row["ci_low"]) for row in local])
        row = {
            "dataset": key[0], "scenario": key[1], "method": key[2],
            "n_decisions": len(local),
            "status_macro_f1": float(f1_score(truth, predicted, labels=labels, average="macro", zero_division=0)),
            "harmful_f1": float(f1_score(truth, predicted, labels=["harmful"], average="macro", zero_division=0)),
            "benign_f1": float(f1_score(truth, predicted, labels=["benign"], average="macro", zero_division=0)),
            "uncertain_f1": float(f1_score(truth, predicted, labels=["uncertain"], average="macro", zero_division=0)),
            "harmful_recall": _recall(truth, predicted, "harmful"),
            "benign_recall": _recall(truth, predicted, "benign"),
            "risk_bias": float(np.mean(errors)), "risk_rmse": float(np.sqrt(np.mean(errors ** 2))),
            "ci_coverage": float(np.mean([float(item["ci_covered"]) for item in local])),
            "ci_width_mean": float(np.mean(widths)),
            "average_labels": float(np.mean([int(item["labels_used"]) for item in local])),
            "max_labels": int(max(int(item["labels_used"]) for item in local)),
            "uncertain_rate": float(np.mean([item["estimated_status"] == "uncertain" for item in local])),
            "formal_protocol": True,
        }
        output.append(row)
    return output


def _run_table6(
    args: argparse.Namespace, output: Path, runs: dict[str, list[dict[str, Any]]], supplementary: Any,
) -> list[dict[str, Any]]:
    path = output / "table6_adaptation_formal.csv"
    lifecycle = _load_module("docs_lifecycle_table6", ROOT / "scripts/llm_judge_ood/37_run_docs_cpu_lifecycle_completion.py")
    if path.exists() and not args.force:
        cached = lifecycle._read_csv(path)
        lifecycle._write_csv(output / "table6_adaptation_summary.csv", _adaptation_summary(cached))
        return cached
    result: list[dict[str, Any]] = []
    for dataset, dataset_runs in runs.items():
        scenarios = ("near", "far") if dataset == "asap" else ("held_out_prompt",)
        for run in dataset_runs:
            for scenario in scenarios:
                for candidate_set in ("development", "evaluation"):
                    result.extend(_adapt_candidate_rows(args, run, dataset, scenario, candidate_set, supplementary))
    lifecycle._write_csv(path, result)
    lifecycle._write_csv(output / "table6_adaptation_summary.csv", _adaptation_summary(result))
    return result


def _adaptation_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    evaluation = [row for row in rows if str(row["candidate_set"]) == "evaluation"]
    selected: list[dict[str, Any]] = []
    for _, local in _group(evaluation, ("dataset", "scenario", "method", "seed")).items():
        method = str(local[0]["method"])
        selected.append(
            max(local, key=lambda row: float(row["gate_gain"]))
            if method not in {"No Update", "Full-label Target Oracle"}
            else local[0]
        )
    output: list[dict[str, Any]] = []
    metrics = (
        "target_labels", "replay_labels", "future_mae_before", "future_mae_after",
        "future_mae_gain", "future_qwk_after", "source_qwk_drop",
        "source_mae_increase", "source_nfr", "trainable_parameters",
        "training_time_seconds", "peak_process_rss_mb",
    )
    for key, local in _group(selected, ("dataset", "scenario", "method")).items():
        row: dict[str, Any] = {
            "dataset": key[0], "scenario": key[1], "method": key[2],
            "seeds": sorted(int(item["seed"]) for item in local),
            "selection_rule": "maximum Gate gain; Future labels hidden",
            "formal_candidate_eligible": True,
        }
        learning_rates = sorted({float(item["learning_rate"]) for item in local if str(item.get("learning_rate", "")).strip()})
        row["selected_learning_rates"] = learning_rates
        for metric in metrics:
            values = _numeric(item.get(metric) for item in local)
            if values:
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        output.append(row)
    return output


def _adapt_candidate_rows(
    args: argparse.Namespace, run: dict[str, Any], dataset: str, scenario: str,
    candidate_set: str, supplementary: Any,
) -> list[dict[str, Any]]:
    rows = run["rows"]
    split = np.asarray([str(row["split"]) for row in rows])
    shift = np.asarray([str(row.get("document_shift_type")) for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    queries = np.asarray([str(row["query_id"]) for row in rows])
    sample_ids = np.asarray([str(row["sample_id"]) for row in rows])
    old = np.asarray([row["judge_prediction"] for row in run["scores"]])
    u = run["model"].transform_u(run["judge_features"])
    scenario_mask = shift == scenario
    adapt = _candidate_partition(np.flatnonzero((split == "deployment_adapt") & scenario_mask), sample_ids, candidate_set, 2)
    gate = _candidate_partition(np.flatnonzero((split == "deployment_gate") & scenario_mask), sample_ids, candidate_set, 2)
    future = _candidate_partition(np.flatnonzero((split == "deployment_future_test") & scenario_mask), sample_ids, candidate_set, 2)
    train = _candidate_partition(np.flatnonzero(split == "training_train"), sample_ids, candidate_set, 2)
    guard_all = np.flatnonzero(split == "training_guard")
    set_guard = _candidate_partition(guard_all, sample_ids, candidate_set, 2)
    gate_guard = _candidate_partition(set_guard, sample_ids, "gate", 2, salt="guard-role")
    truth_guard = _candidate_partition(set_guard, sample_ids, "truth", 2, salt="guard-role")
    if min(map(len, (adapt, gate, future, train, gate_guard, truth_guard))) < 2:
        raise RuntimeError(f"Insufficient independent candidate data for {dataset}/{scenario}/{candidate_set}")
    metadata = run["summary"].get("adaptation", {}).get("adapter") or {}
    base = HeadAdaptConfig(**(metadata.get("config") or {}))
    variants: list[tuple[str, HeadAdaptConfig | None, bool, float | None]] = [("No Update", None, False, None)]
    for learning_rate in (1e-4, 3e-4, 1e-3):
        variants.extend([
            ("Target-only Head FT", replace(base, learning_rate=learning_rate, training_replay_weight=0.0, anchor_weight=0.0), False, learning_rate),
            ("Head + Replay", replace(base, learning_rate=learning_rate, anchor_weight=0.0), True, learning_rate),
            ("Head + Replay + Anchor", replace(base, learning_rate=learning_rate), True, learning_rate),
        ])
    variants.append(("Full-label Target Oracle", base, True, float(base.learning_rate)))
    output: list[dict[str, Any]] = []
    for method, config, use_replay, learning_rate in variants:
        rng = np.random.default_rng(_stable_seed(f"adapt:{run['name']}:{scenario}:{candidate_set}:{method}:{learning_rate}"))
        requested = len(adapt) if method == "Full-label Target Oracle" else int(args.adapt_budget)
        target = np.asarray([], dtype=int) if config is None else rng.choice(adapt, size=min(requested, len(adapt)), replace=False)
        replay = np.asarray([], dtype=int)
        prediction = old.copy()
        train_seconds = 0.0
        trainable = 0
        if config is not None:
            if use_replay:
                replay = rng.choice(train, size=min(requested, len(train)), replace=False)
            started = time.perf_counter()
            adapter = HeadAdapter(config).fit(
                u_features=u, labels=labels, query_ids=queries,
                deployment_indices=target, training_replay_indices=replay,
                class_values=run["model"].classes_, judge=run["model"],
            )
            train_seconds = float(time.perf_counter() - started)
            prediction = adapter.predict(u_features=u, query_ids=queries, fallback=old)
            trainable = int(sum(parameter.numel() for head in adapter.heads_.values() for parameter in head.parameters()))
        gate_values = np.abs(old[gate].astype(float) - labels[gate].astype(float)) - np.abs(prediction[gate].astype(float) - labels[gate].astype(float))
        gate_mean, gate_low, gate_high = _mean_interval(gate_values)
        gate_source = _source_metrics(labels, old, prediction, gate_guard, run["model"].classes_)
        truth_source = _source_metrics(labels, old, prediction, truth_guard, run["model"].classes_)
        before = judge_metrics(labels[future], old[future], class_values=run["model"].classes_)
        after = judge_metrics(labels[future], prediction[future], class_values=run["model"].classes_)
        gain = float(before["mae"] - after["mae"])
        good = bool(gain >= 0.10 and truth_source["nfr"] <= 0.05 and truth_source["qwk_drop"] <= 0.02 and truth_source["mae_increase"] <= 0.05)
        output.append({
            "dataset": dataset, "seed": run["seed"], "scenario": scenario,
            "candidate_set": candidate_set, "candidate_id": f"{run['name']}:{scenario}:{candidate_set}:{method}:{learning_rate}",
            "method": method, "learning_rate": "" if learning_rate is None else learning_rate,
            "target_labels": len(target), "replay_labels": len(replay),
            "candidate_search_budget": 1 if method == "Full-label Target Oracle" else 3 if config is not None else 0,
            "equal_budget_eligible": method not in {"No Update", "Full-label Target Oracle"},
            "future_documents": len(future), "future_mae_before": before["mae"], "future_mae_after": after["mae"],
            "future_mae_gain": gain, "future_qwk_before": before["qwk"], "future_qwk_after": after["qwk"],
            "source_qwk_drop": truth_source["qwk_drop"], "source_mae_increase": truth_source["mae_increase"], "source_nfr": truth_source["nfr"],
            "gate_gain": gate_mean, "gate_gain_lcb": gate_low, "gate_gain_ucb": gate_high,
            "gate_source_qwk_drop": gate_source["qwk_drop"], "gate_source_mae_increase": gate_source["mae_increase"], "gate_source_nfr": gate_source["nfr"],
            "oracle_good_candidate": good, "trainable_parameters": trainable,
            "training_time_seconds": train_seconds, "peak_process_rss_mb": _peak_rss_mb(),
            "gpu_time_seconds": 0.0, "gpu_peak_memory_mb": 0.0,
            "formal_candidate_eligible": True, "formal_gate_eligible": candidate_set == "evaluation",
            "future_labels_used_for_gate_decision": False,
            "evaluation_scope": "independent_candidate_protocol",
        })
    return output


def _candidate_partition(
    indices: np.ndarray, sample_ids: np.ndarray, role: str, modulo: int, *, salt: str = "candidate-set",
) -> np.ndarray:
    target = 0 if role in {"development", "gate"} else 1
    return np.asarray([index for index in indices if _stable_seed(f"partition:{salt}:{sample_ids[index]}") % modulo == target], dtype=int)


def _source_metrics(labels: np.ndarray, before: np.ndarray, after: np.ndarray, indices: np.ndarray, classes: np.ndarray) -> dict[str, float]:
    old = judge_metrics(labels[indices], before[indices], class_values=classes)
    new = judge_metrics(labels[indices], after[indices], class_values=classes)
    return {
        "qwk_drop": float(old["qwk"] - new["qwk"]),
        "mae_increase": float(new["mae"] - old["mae"]),
        "nfr": float(np.mean((after[indices] != labels[indices]) & (before[indices] == labels[indices]))),
    }


def _mean_interval(values: np.ndarray) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    se = float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else float("inf")
    radius = float(norm.ppf(0.975) * se) if np.isfinite(se) else float("inf")
    return mean, mean - radius, mean + radius


def _run_table7(output: Path, table6: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    development = [row for row in table6 if row["candidate_set"] == "development" and _as_bool(row.get("equal_budget_eligible"))]
    evaluation = [row for row in table6 if row["candidate_set"] == "evaluation" and _as_bool(row.get("equal_budget_eligible"))]
    thresholds: dict[str, Any] = {}
    output_rows: list[dict[str, Any]] = []
    datasets = sorted(set(str(row["dataset"]) for row in development))
    for dataset in [*datasets, "combined"]:
        dev = development if dataset == "combined" else [row for row in development if row["dataset"] == dataset]
        test = evaluation if dataset == "combined" else [row for row in evaluation if row["dataset"] == dataset]
        thresholds[dataset] = {}
        for variant in GATE_VARIANTS:
            params = _select_gate_thresholds(dev, variant)
            thresholds[dataset][variant] = params
            decisions = [_gate_decision(row, variant, params) for row in test]
            truth = np.asarray([_as_bool(row["oracle_good_candidate"]) for row in test])
            accepted = np.asarray([decision == "accept" for decision in decisions])
            rejected = np.asarray([decision == "reject" for decision in decisions])
            deferred = np.asarray([decision == "defer" for decision in decisions])
            bad = ~truth
            accepted_rows = [row for row, keep in zip(test, accepted, strict=True) if keep]
            output_rows.append({
                "dataset": dataset, "gate": variant, "candidate_set": "evaluation",
                "candidates": len(test), "good_updates": int(truth.sum()), "bad_updates": int(bad.sum()),
                "bad_update_acceptance": float(np.sum(accepted & bad) / max(int(bad.sum()), 1)),
                "good_update_rejection": float(np.sum(rejected & truth) / max(int(truth.sum()), 1)),
                "defer_rate": float(np.mean(deferred)), "decision_coverage": float(np.mean(~deferred)),
                "accepted_target_gain": _mean_field(accepted_rows, "future_mae_gain", empty="no_accepted_candidate"),
                "accepted_source_drop": _mean_field(accepted_rows, "source_qwk_drop", empty="no_accepted_candidate"),
                "accepted_nfr": _mean_field(accepted_rows, "source_nfr", empty="no_accepted_candidate"),
                "thresholds_frozen_on_development_candidates": True,
                "formal_gate_eligible": True,
            })
    write_json(output / "table7_frozen_thresholds.json", thresholds)
    lifecycle = _load_module("docs_lifecycle_table7", ROOT / "scripts/llm_judge_ood/37_run_docs_cpu_lifecycle_completion.py")
    lifecycle._write_csv(output / "table7_gate_formal.csv", output_rows)
    return thresholds, output_rows


def _select_gate_thresholds(rows: list[dict[str, Any]], variant: str) -> dict[str, float]:
    if variant == "no_gate":
        return {"epsilon": 0.0, "eta": 1.0, "gamma": 1.0, "mae": 1.0}
    candidates: list[tuple[tuple[float, ...], dict[str, float]]] = []
    for epsilon in (0.05, 0.10, 0.15):
        for eta in (0.03, 0.05, 0.08):
            for gamma in (0.01, 0.02, 0.03):
                for mae in (0.03, 0.05, 0.08):
                    params = {"epsilon": epsilon, "eta": eta, "gamma": gamma, "mae": mae}
                    decisions = [_gate_decision(row, variant, params) for row in rows]
                    truth = np.asarray([_as_bool(row["oracle_good_candidate"]) for row in rows])
                    accept = np.asarray([value == "accept" for value in decisions])
                    reject = np.asarray([value == "reject" for value in decisions])
                    defer = np.asarray([value == "defer" for value in decisions])
                    bad = ~truth
                    score = (
                        float(np.sum(accept & bad) / max(int(bad.sum()), 1)),
                        float(np.sum(reject & truth) / max(int(truth.sum()), 1)),
                        float(np.mean(defer)), epsilon, eta, gamma, mae,
                    )
                    candidates.append((score, params))
    return min(candidates, key=lambda item: item[0])[1]


def _gate_decision(row: dict[str, Any], variant: str, params: dict[str, float]) -> str:
    if variant == "no_gate":
        return "accept"
    gain = float(row["gate_gain"])
    low = float(row["gate_gain_lcb"])
    high = float(row["gate_gain_ucb"])
    nfr = float(row["gate_source_nfr"])
    qwk = float(row["gate_source_qwk_drop"])
    mae = float(row["gate_source_mae_increase"])
    if variant == "gain_only":
        return "accept" if gain >= params["epsilon"] else "reject"
    safety_failed = variant in {"gain_confidence_nfr", "full_gate"} and nfr > params["eta"]
    safety_failed |= variant == "full_gate" and (qwk > params["gamma"] or mae > params["mae"])
    if safety_failed or high < params["epsilon"]:
        return "reject"
    if low >= params["epsilon"]:
        return "accept"
    return "defer"


def _run_table8(
    args: argparse.Namespace, output: Path, table4: list[dict[str, Any]], table5: list[dict[str, Any]],
    table6: list[dict[str, Any]], thresholds: dict[str, Any],
) -> list[dict[str, Any]]:
    systems = (
        ("No Monitoring / No Update", "", "", "No Update", "none"),
        ("Sample OOD only", "Fixed Random Probe", "", "Target-only Head FT", "no_gate"),
        ("MMD + Fixed Probe", "Fixed Random Probe", "B-MMD", "Head + Replay", "gain_only"),
        ("Strong Baseline Pipeline", "Residual-stratified Probe", "KS", "Head + Replay", "full_gate"),
        ("Full Proposed System", "Sequential Random Probe", "B-MMD", "Head + Replay + Anchor", "full_gate"),
        ("Full-label Oracle", "Full-label Oracle", "oracle", "oracle", "oracle"),
    )
    rows: list[dict[str, Any]] = []
    evaluation = [row for row in table6 if row["candidate_set"] == "evaluation"]
    for dataset in sorted(set(str(row["dataset"]) for row in evaluation)):
        scenarios = sorted(set(str(row["scenario"]) for row in evaluation if row["dataset"] == dataset))
        for name, probe, window, adapt, gate in systems:
            selected: list[dict[str, Any]] = []
            for scenario in scenarios:
                local = [row for row in evaluation if row["dataset"] == dataset and row["scenario"] == scenario and row["method"] == adapt]
                if local:
                    selected.extend(_best_gate_candidate(local))
            if name == "No Monitoring / No Update":
                baseline = [row for row in evaluation if row["dataset"] == dataset and row["method"] == "No Update"]
                rows.append(_e2e_row(dataset, name, 0, 0.0, 0.0, 0.0, baseline, 0.0, 0.0))
                continue
            if name == "Full-label Oracle":
                oracle_rows = [
                    row for row in evaluation
                    if row["dataset"] == dataset and row["method"] == "Full-label Target Oracle"
                ]
                rows.append(_e2e_row(dataset, name, "all", 0.0, 1.0, 1.0, oracle_rows, 0.0, 0.0))
                continue
            probe_rows = [row for row in table5 if row["dataset"] == dataset and row["method"] == probe]
            harmful_recall = float(np.mean(_numeric(row.get("harmful_recall") for row in probe_rows))) if probe_rows else 0.0
            if window:
                window_rows = [row for row in table4 if row["dataset"] == dataset and row["method"] == window and int(float(row["window_size"])) == 100]
                fwer = float(np.mean(_numeric(row.get("sequential_fwer_mean") for row in window_rows))) if window_rows else 0.0
                detection = float(np.mean(_numeric(row.get("detection_rate_10pct_mean") for row in window_rows))) if window_rows else 1.0
                runtime = float(np.mean(_numeric(row.get("runtime_sec_per_window_mean") for row in window_rows))) if window_rows else 0.0
            else:
                fwer, detection, runtime = 1.0 - 0.95 ** 8, 1.0, 0.0
            accepted: list[dict[str, Any]] = []
            for row in selected:
                decision = "accept" if gate == "no_gate" else _gate_decision(row, gate, thresholds[dataset][gate])
                if decision == "accept":
                    accepted.append(row)
            wrong = float(np.mean([not _as_bool(row["oracle_good_candidate"]) for row in accepted])) if accepted else 0.0
            deployment_rate = len(accepted) / max(len(selected), 1)
            detected_harmful = detection * harmful_recall
            rows.append(_e2e_row(
                dataset, name, int(args.probe_budget + args.adapt_budget), fwer,
                detected_harmful, detected_harmful * deployment_rate, accepted, wrong, runtime,
            ))
    lifecycle = _load_module("docs_lifecycle_table8", ROOT / "scripts/llm_judge_ood/37_run_docs_cpu_lifecycle_completion.py")
    lifecycle._write_csv(output / "table8_equal_budget_formal.csv", rows)
    return rows


def _best_gate_candidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    return [max(local, key=lambda row: float(row["gate_gain"])) for local in by_seed.values()]


def _best_by_future(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_key.setdefault((str(row["scenario"]), int(row["seed"])), []).append(row)
    return [max(local, key=lambda row: float(row["future_mae_gain"])) for local in by_key.values()]


def _e2e_row(
    dataset: str, system: str, labels: Any, fwer: float, harmful_recall: float,
    action_recall: float, accepted: list[dict[str, Any]], wrong: float, runtime: float,
) -> dict[str, Any]:
    return {
        "dataset": dataset, "system": system, "total_target_labels": labels,
        "fwer": fwer, "harmful_recall": harmful_recall,
        "harmful_action_recall": action_recall,
        "target_future_gain": _mean_field(accepted, "future_mae_gain", empty="no_update_deployed"),
        "source_drop": _mean_field(accepted, "source_qwk_drop", empty="no_update_deployed"),
        "bad_update_deployed": wrong, "total_runtime_seconds": runtime + sum(float(row.get("training_time_seconds", 0.0)) for row in accepted),
        "same_episode_and_future_protocol": True,
        "equal_label_budget_eligible": system in {"No Monitoring / No Update", "Full-label Oracle"} or labels != 0,
    }


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    output: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        output.setdefault(tuple(str(row.get(key, "")) for key in keys), []).append(row)
    return output


def _numeric(values: Iterable[Any]) -> list[float]:
    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            output.append(number)
    return output


def _recall(truth: list[str], predicted: list[str], label: str) -> float:
    positives = [index for index, value in enumerate(truth) if value == label]
    return float(np.mean([predicted[index] == label for index in positives])) if positives else 0.0


def _mean_field(rows: list[dict[str, Any]], field: str, *, empty: Any = "") -> Any:
    values = _numeric(row.get(field) for row in rows)
    return float(np.mean(values)) if values else empty


def _majority(values: list[str]) -> str:
    return max(sorted(set(values)), key=values.count)


def _as_bool(value: Any) -> bool:
    return value is True or str(value).strip().lower() in {"1", "true", "yes"}


def _peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0 * 1024.0)


def _stable_seed(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def _load_module(name: str, path: Path) -> Any:
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
