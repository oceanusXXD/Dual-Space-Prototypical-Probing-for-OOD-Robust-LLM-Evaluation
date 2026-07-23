#!/usr/bin/env python3
"""Run FLASK RQ5: low-confidence-triggered adaptation with rollback gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.adapt.head import HeadAdaptConfig, HeadAdapter
from src.llm_judge_ood.scores.vim import ViMScorer
from src.llm_judge_ood.shared.metrics import judge_metrics


SOURCE_DOMAINS = ("Humanities", "Language", "Social Science")
SOURCE_SKILLS = ("Comprehension", "Factuality", "Logical Correctness")
TARGET_DOMAINS = ("Humanities", "Language", "Social Science", "History", "Culture")
TARGET_SKILLS = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
)
CLASSES = (1, 2, 3, 4, 5)
GATE_METRICS = ("plus_minus_1_accuracy", "exact_accuracy", "quadratic_weighted_kappa", "mae")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/"
            "qwen35_08b_5x6_digit_direct_judge_strict_prelogit_bspace/"
            "strict_final_prelogit_b_space_features.npz"
        ),
    )
    parser.add_argument(
        "--heads-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/qwen35_08b_strict_prelogit_3x3_heads"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/rq5_gated_adaptation_08b_5x6_prelogit"),
    )
    parser.add_argument("--adapt-question-fraction", type=float, default=0.15)
    parser.add_argument("--validation-question-fraction", type=float, default=0.25)
    parser.add_argument("--test-question-fraction", type=float, default=0.25)
    parser.add_argument(
        "--label-budgets",
        default="1,3,5,10,20",
        help="Comma-separated target question-group budgets for supervised L0/L1.",
    )
    parser.add_argument("--ablation-label-budget", type=int, default=5)
    parser.add_argument("--calibration-question-fraction", type=float, default=0.10)
    parser.add_argument("--gate-metric", choices=GATE_METRICS, default="plus_minus_1_accuracy")
    parser.add_argument("--min-target-improvement", type=float, default=0.01)
    parser.add_argument("--max-source-nfr", type=float, default=0.05)
    parser.add_argument("--max-source-qwk-drop", type=float, default=0.02)
    parser.add_argument("--bootstrap-samples", type=int, default=300)
    parser.add_argument("--l0-epochs", type=int, default=12)
    parser.add_argument("--l1-epochs", type=int, default=40)
    parser.add_argument("--l2-epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--source-replay-weight", type=float, default=1.0)
    parser.add_argument("--target-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=1e-2)
    parser.add_argument("--geometry-weight", type=float, default=1e-2)
    parser.add_argument("--max-geometry-auroc-drop", type=float, default=0.05)
    parser.add_argument("--min-geometry-mmd-ratio", type=float, default=0.80)
    parser.add_argument("--collapse-max-class-fraction", type=float, default=0.95)
    parser.add_argument("--disable-l2-tent", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--max-target-cells", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is non-empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    payload = load_feature_payload(args.features)
    specs = load_head_specs(args.heads_dir)
    if int(args.max_heads) > 0:
        specs = specs[: int(args.max_heads)]

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"running RQ5 {spec['head_id']}", flush=True)
        result = run_head(spec, payload, args)
        candidate_rows.extend(result["candidate_rows"])
        selected_rows.extend(result["selected_rows"])
        trigger_rows.extend(result["trigger_rows"])

    level_rows = rq5_level_rows(args)
    gate_rows = build_gate_summary(candidate_rows)
    write_csv(args.output_dir / "adaptation_cell_results.csv", candidate_rows)
    write_csv(args.output_dir / "selected_cell_results.csv", selected_rows)
    write_csv(args.output_dir / "gate_summary.csv", gate_rows)
    write_csv(args.output_dir / "low_confidence_trigger_summary.csv", trigger_rows)
    write_csv(args.output_dir / "rq5_level_summary.csv", level_rows)
    summary = build_summary(
        args=args,
        payload=payload,
        specs=specs,
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        gate_rows=gate_rows,
        level_rows=level_rows,
        elapsed_seconds=time.perf_counter() - started,
    )
    write_json(args.output_dir / "summary.json", clean_json(summary))
    print(json.dumps(compact_summary(summary), ensure_ascii=False, indent=2))


def run_head(
    spec: dict[str, Any],
    payload: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, Any]]]:
    model = joblib.load(spec["model_path"])
    query_ids = tuple(getattr(model, "query_ids_", ()))
    if len(query_ids) != 1:
        raise ValueError(f"{spec['head_id']} must contain exactly one query id, got {query_ids}")
    routed_queries = np.full(len(payload["features"]), query_ids[0], dtype=object)
    output = model.predict_output(payload["features"], routed_queries)
    h = np.asarray(output.penultimate, dtype=np.float32)
    logits = np.asarray(output.logits, dtype=np.float32)
    probabilities = np.asarray(output.probabilities, dtype=np.float32)
    classes = np.asarray(output.classes)
    baseline_predictions = classes[np.argmax(probabilities, axis=1)].astype(int)
    labels = payload["labels"].astype(int)
    pm1_loss = (np.abs(baseline_predictions - labels) > 1).astype(int)

    source_domain = str(spec["source_domain"])
    source_skill = str(spec["source_skill"])
    train_questions = set(str(value) for value in spec["split"]["train_question_ids"])
    source_mask = (payload["domains"] == source_domain) & (payload["skills"] == source_skill)
    train_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(train_questions)))
    remaining_questions = sorted(set(payload["question_ids"][source_mask].tolist()).difference(train_questions))
    calibration_questions = stable_question_sample(
        remaining_questions,
        fraction=float(args.calibration_question_fraction),
        seed=int(args.seed),
        namespace=f"{source_domain}::{source_skill}::rq5_trigger_calibration",
    )
    source_guard_questions = sorted(set(remaining_questions).difference(calibration_questions))
    calibration_mask = source_mask & np.isin(payload["question_ids"], np.asarray(sorted(calibration_questions)))
    source_guard_mask = source_mask & np.isin(payload["question_ids"], np.asarray(source_guard_questions))
    if not train_mask.any() or not source_guard_mask.any():
        raise RuntimeError(f"RQ5 split failed for {spec['head_id']}")

    trigger_score, trigger_meta = build_trigger_score(
        h=h,
        probabilities=probabilities,
        train_mask=train_mask,
        calibration_mask=calibration_mask,
        primary_loss=pm1_loss,
        seed=int(args.seed),
    )
    weights, biases, head_query_ids = model.affine_head_parameters()
    head_index = int(np.flatnonzero(np.asarray(head_query_ids).astype(str) == str(query_ids[0]))[0])
    affine_weight = np.asarray(weights[head_index], dtype=np.float32)
    affine_bias = np.asarray(biases[head_index], dtype=np.float32)
    target_cells = target_cell_masks(
        payload=payload,
        source_domain=source_domain,
        source_skill=source_skill,
        train_questions=train_questions,
    )
    if int(args.max_target_cells) > 0:
        target_cells = target_cells[: int(args.max_target_cells)]

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    label_budgets = parse_int_grid(args.label_budgets)
    for cell in target_cells:
        splits = budgeted_question_splits(
            question_ids=payload["question_ids"],
            target_mask=cell["mask"],
            trigger_score=trigger_score,
            validation_fraction=float(args.validation_question_fraction),
            test_fraction=float(args.test_question_fraction),
            seed=int(args.seed),
            namespace=f"{spec['head_id']}::{cell['target_domain']}::{cell['target_skill']}",
        )
        trigger_rows.append(
            {
                **head_prefix(spec),
                "target_domain": cell["target_domain"],
                "target_skill": cell["target_skill"],
                "shift_type": cell["shift_type"],
                "trigger_score": trigger_meta["score_id"],
                "available_adapt_questions": len(splits["adapt_pool_questions"]),
                "supported_label_budgets": ";".join(
                    str(value) for value in label_budgets if value <= len(splits["adapt_pool_questions"])
                ),
                "validation_questions": len(splits["validation_questions"]),
                "test_questions": len(splits["test_questions"]),
                "adapt_pool_rows": int(splits["adapt_pool_mask"].sum()),
                "validation_rows": int(splits["validation_mask"].sum()),
                "test_rows": int(splits["test_mask"].sum()),
                "adapt_trigger_score_mean": mean_or_none(trigger_score[splits["adapt_pool_mask"]]),
                "validation_trigger_score_mean": mean_or_none(trigger_score[splits["validation_mask"]]),
                "test_trigger_score_mean": mean_or_none(trigger_score[splits["test_mask"]]),
            }
        )
        if (
            not splits["adapt_pool_mask"].any()
            or not splits["validation_mask"].any()
            or not splits["test_mask"].any()
        ):
            selected_rows.append(skip_selected_row(spec, cell, "insufficient_target_split"))
            continue

        baseline_context = baseline_metrics_context(
            labels=labels,
            baseline_predictions=baseline_predictions,
            target_validation_mask=splits["validation_mask"],
            target_test_mask=splits["test_mask"],
            source_guard_mask=source_guard_mask,
        )
        candidates: list[dict[str, Any]] = []
        if not bool(args.disable_l2_tent):
            row, candidate = evaluate_candidate_method(
                method_id="L2_tent_unlabeled_diagonal",
                prediction_builder=lambda: predict_diagonal_affine(
                    h=h,
                    labels=labels,
                    classes=classes,
                    weight=affine_weight,
                    bias=affine_bias,
                    source_mask=train_mask,
                    target_mask=splits["adapt_pool_mask"],
                    args=args,
                    mode="tent_unlabeled",
                ),
                label_budget=0,
                ablation_id="primary",
                spec=spec,
                cell=cell,
                features=h,
                question_ids=payload["question_ids"],
                labels=labels,
                baseline_predictions=baseline_predictions,
                baseline_context=baseline_context,
                source_train_mask=train_mask,
                target_validation_mask=splits["validation_mask"],
                target_test_mask=splits["test_mask"],
                source_guard_mask=source_guard_mask,
                args=args,
            )
            candidate_rows.append(row)
            candidates.append(candidate)

        for budget in label_budgets:
            if budget > len(splits["adapt_pool_questions"]):
                continue
            adapt_questions = set(splits["adapt_pool_questions"][:budget])
            adapt_mask = cell["mask"] & np.isin(
                payload["question_ids"], np.asarray(sorted(adapt_questions))
            )
            primary_builders = (
                (
                    "L0_head_only",
                    lambda: predict_l0_head_adapter(
                        model=model,
                        h=h,
                        labels=labels,
                        classes=classes,
                        routed_queries=routed_queries,
                        baseline_predictions=baseline_predictions,
                        baseline_probabilities=probabilities,
                        train_mask=train_mask,
                        adapt_mask=adapt_mask,
                        args=args,
                    ),
                ),
                (
                    "L1_diagonal_feature_alignment",
                    lambda: predict_diagonal_affine(
                        h=h,
                        labels=labels,
                        classes=classes,
                        weight=affine_weight,
                        bias=affine_bias,
                        source_mask=train_mask,
                        target_mask=adapt_mask,
                        args=args,
                        mode="supervised_alignment",
                    ),
                ),
            )
            for method_id, prediction_builder in primary_builders:
                row, candidate = evaluate_candidate_method(
                    method_id=method_id,
                    prediction_builder=prediction_builder,
                    label_budget=budget,
                    ablation_id="primary",
                    spec=spec,
                    cell=cell,
                    features=h,
                    question_ids=payload["question_ids"],
                    labels=labels,
                    baseline_predictions=baseline_predictions,
                    baseline_context=baseline_context,
                    source_train_mask=train_mask,
                    target_validation_mask=splits["validation_mask"],
                    target_test_mask=splits["test_mask"],
                    source_guard_mask=source_guard_mask,
                    args=args,
                )
                candidate_rows.append(row)
                candidates.append(candidate)

            if budget == int(args.ablation_label_budget):
                for ablation_id, replay_weight, anchor_weight, geometry_weight in (
                    ("no_replay_no_anchor_no_geometry", 0.0, 0.0, 0.0),
                    ("replay_only", float(args.source_replay_weight), 0.0, 0.0),
                    (
                        "replay_parameter_anchor",
                        float(args.source_replay_weight),
                        float(args.anchor_weight),
                        0.0,
                    ),
                ):
                    row, _candidate = evaluate_candidate_method(
                        method_id="L1_diagonal_feature_alignment",
                        prediction_builder=lambda rw=replay_weight, aw=anchor_weight, gw=geometry_weight: predict_diagonal_affine(
                            h=h,
                            labels=labels,
                            classes=classes,
                            weight=affine_weight,
                            bias=affine_bias,
                            source_mask=train_mask,
                            target_mask=adapt_mask,
                            args=args,
                            mode="supervised_alignment",
                            source_replay_weight=rw,
                            anchor_weight=aw,
                            geometry_weight=gw,
                        ),
                        label_budget=budget,
                        ablation_id=ablation_id,
                        spec=spec,
                        cell=cell,
                        features=h,
                        question_ids=payload["question_ids"],
                        labels=labels,
                        baseline_predictions=baseline_predictions,
                        baseline_context=baseline_context,
                        source_train_mask=train_mask,
                        target_validation_mask=splits["validation_mask"],
                        target_test_mask=splits["test_mask"],
                        source_guard_mask=source_guard_mask,
                        args=args,
                    )
                    candidate_rows.append(row)

        selected_rows.append(
            selected_result_row(
                spec=spec,
                cell=cell,
                candidates=candidates,
                labels=labels,
                baseline_predictions=baseline_predictions,
                baseline_context=baseline_context,
                target_test_mask=splits["test_mask"],
            )
        )
    return {
        "candidate_rows": candidate_rows,
        "selected_rows": selected_rows,
        "trigger_rows": trigger_rows,
    }


def predict_l0_head_adapter(
    *,
    model: Any,
    h: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    routed_queries: np.ndarray,
    baseline_predictions: np.ndarray,
    baseline_probabilities: np.ndarray,
    train_mask: np.ndarray,
    adapt_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    config = HeadAdaptConfig(
        epochs=int(args.l0_epochs),
        patience=max(2, min(5, int(args.l0_epochs))),
        deployment_validation_fraction=0.0,
        learning_rate=float(args.learning_rate),
        training_replay_weight=float(args.source_replay_weight),
        deployment_weight=float(args.target_weight),
        anchor_weight=float(args.anchor_weight),
        seed=int(args.seed),
    )
    adapter = HeadAdapter(config)
    adapter.fit(
        u_features=np.asarray(h, dtype=np.float32),
        labels=labels,
        query_ids=routed_queries,
        deployment_indices=np.flatnonzero(adapt_mask),
        training_replay_indices=np.flatnonzero(train_mask),
        class_values=classes,
        judge=model,
    )
    predictions = adapter.predict(
        u_features=np.asarray(h, dtype=np.float32),
        query_ids=routed_queries,
        fallback=baseline_predictions,
    ).astype(int)
    return predictions, {"adapter": adapter.to_metadata()}


def predict_diagonal_affine(
    *,
    h: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    args: argparse.Namespace,
    mode: str,
    source_replay_weight: float | None = None,
    anchor_weight: float | None = None,
    geometry_weight: float | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    result = fit_diagonal_affine(
        h=np.asarray(h, dtype=np.float32),
        labels=labels,
        classes=classes,
        weight=np.asarray(weight, dtype=np.float32),
        bias=np.asarray(bias, dtype=np.float32),
        source_indices=np.flatnonzero(source_mask),
        target_indices=np.flatnonzero(target_mask),
        mode=mode,
        args=args,
        source_replay_weight=source_replay_weight,
        anchor_weight=anchor_weight,
        geometry_weight=geometry_weight,
    )
    logits = diagonal_affine_logits(
        np.asarray(h, dtype=np.float32),
        result["log_scale"],
        result["shift"],
        np.asarray(weight, dtype=np.float32),
        np.asarray(bias, dtype=np.float32),
    )
    predictions = np.asarray(classes)[np.argmax(logits, axis=1)].astype(int)
    return predictions, {"adapter": result["metadata"]}, {
        "kind": "diagonal_affine",
        "log_scale": result["log_scale"],
        "shift": result["shift"],
    }


def fit_diagonal_affine(
    *,
    h: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    source_indices: np.ndarray,
    target_indices: np.ndarray,
    mode: str,
    args: argparse.Namespace,
    source_replay_weight: float | None = None,
    anchor_weight: float | None = None,
    geometry_weight: float | None = None,
) -> dict[str, Any]:
    device = torch.device("cpu")
    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    train_indices = np.unique(np.concatenate([source_indices, target_indices]))
    local_source = np.searchsorted(train_indices, source_indices)
    local_target = np.searchsorted(train_indices, target_indices)
    x = torch.as_tensor(h[train_indices], dtype=torch.float32, device=device)
    w = torch.as_tensor(weight, dtype=torch.float32, device=device)
    b = torch.as_tensor(bias, dtype=torch.float32, device=device)
    labels_array = np.asarray(labels)
    class_to_index = {value: index for index, value in enumerate(np.asarray(classes).tolist())}
    y_all = torch.as_tensor(
        [class_to_index[value] for value in labels_array[train_indices].tolist()],
        dtype=torch.long,
        device=device,
    )
    source = torch.as_tensor(local_source, dtype=torch.long, device=device)
    target = torch.as_tensor(local_target, dtype=torch.long, device=device)
    feature_std = torch.as_tensor(
        np.maximum(np.std(h[train_indices], axis=0), 1e-3),
        dtype=torch.float32,
        device=device,
    )
    log_scale = nn.Parameter(torch.zeros(h.shape[1], dtype=torch.float32, device=device))
    shift_raw = nn.Parameter(torch.zeros(h.shape[1], dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(
        [log_scale, shift_raw],
        lr=float(args.learning_rate),
        weight_decay=0.0,
    )
    epochs = int(args.l2_epochs if mode == "tent_unlabeled" else args.l1_epochs)
    replay_weight = (
        float(args.source_replay_weight) if source_replay_weight is None else float(source_replay_weight)
    )
    parameter_anchor_weight = float(args.anchor_weight) if anchor_weight is None else float(anchor_weight)
    geometry_anchor_weight = (
        float(args.geometry_weight) if geometry_weight is None else float(geometry_weight)
    )
    loss_history: list[float] = []
    for _epoch in range(max(1, epochs)):
        transformed = x * torch.exp(log_scale)[None, :] + shift_raw[None, :] * feature_std[None, :]
        logits = transformed @ w + b
        source_loss = class_balanced_ce(logits[source], y_all[source], len(classes))
        if mode == "tent_unlabeled":
            target_prob = torch.softmax(logits[target], dim=1)
            target_loss = -torch.sum(target_prob * torch.log(torch.clamp(target_prob, min=1e-8)), dim=1).mean()
        else:
            target_loss = class_balanced_ce(logits[target], y_all[target], len(classes))
        parameter_anchor = torch.mean(log_scale.pow(2)) + torch.mean(shift_raw.pow(2))
        source_original_normalized = nn.functional.normalize(x[source], dim=1)
        source_transformed_normalized = nn.functional.normalize(transformed[source], dim=1)
        # Deterministic O(ND) pair sample: preserve source cosine geometry without
        # materializing an O(N^2) Gram matrix for every adaptation epoch.
        original_pair_cosine = torch.sum(
            source_original_normalized * torch.roll(source_original_normalized, shifts=1, dims=0),
            dim=1,
        )
        transformed_pair_cosine = torch.sum(
            source_transformed_normalized * torch.roll(source_transformed_normalized, shifts=1, dims=0),
            dim=1,
        )
        geometry_anchor = torch.mean((transformed_pair_cosine - original_pair_cosine).pow(2))
        loss = (
            replay_weight * source_loss
            + float(args.target_weight) * target_loss
            + parameter_anchor_weight * parameter_anchor
            + geometry_anchor_weight * geometry_anchor
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([log_scale, shift_raw], max_norm=5.0)
        optimizer.step()
        loss_history.append(float(loss.detach().cpu().item()))
    return {
        "log_scale": log_scale.detach().cpu().numpy().astype(np.float32),
        "shift": (shift_raw.detach().cpu().numpy() * feature_std.detach().cpu().numpy()).astype(np.float32),
        "metadata": {
            "mode": mode,
            "epochs": epochs,
            "optimizer": "AdamW",
            "learning_rate": float(args.learning_rate),
            "source_replay_weight": replay_weight,
            "target_weight": float(args.target_weight),
            "anchor_weight": parameter_anchor_weight,
            "geometry_weight": geometry_anchor_weight,
            "source_rows": int(len(source_indices)),
            "target_rows": int(len(target_indices)),
            "final_loss": loss_history[-1] if loss_history else None,
            "objective": (
                "source replay CE + target supervised CE + parameter anchor + source cosine-geometry anchor"
                if mode == "supervised_alignment"
                else "source replay CE + unlabeled target entropy + parameter anchor + source cosine-geometry anchor"
            ),
        },
    }


def diagonal_affine_logits(
    h: np.ndarray,
    log_scale: np.ndarray,
    shift: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    transformed = np.asarray(h, dtype=np.float32) * np.exp(np.asarray(log_scale, dtype=np.float32))[None, :]
    transformed = transformed + np.asarray(shift, dtype=np.float32)[None, :]
    return (transformed @ np.asarray(weight, dtype=np.float32) + np.asarray(bias, dtype=np.float32)).astype(np.float32)


def evaluate_candidate_method(
    *,
    method_id: str,
    prediction_builder: Any,
    label_budget: int,
    ablation_id: str,
    spec: dict[str, Any],
    cell: dict[str, Any],
    features: np.ndarray,
    question_ids: np.ndarray,
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    baseline_context: dict[str, Any],
    source_train_mask: np.ndarray,
    target_validation_mask: np.ndarray,
    target_test_mask: np.ndarray,
    source_guard_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        built = prediction_builder()
        candidate_predictions, metadata = built[:2]
        feature_transform = built[2] if len(built) > 2 else None
        status = "ok"
        error = ""
    except Exception as exc:  # noqa: BLE001 - one failed adaptation candidate should not kill the grid.
        candidate_predictions = baseline_predictions.copy()
        metadata = {}
        feature_transform = None
        status = "failed"
        error = str(exc)

    gate = gate_decision(
        labels=labels,
        baseline_predictions=baseline_predictions,
        candidate_predictions=candidate_predictions,
        validation_question_ids=question_ids[target_validation_mask],
        target_validation_mask=target_validation_mask,
        source_guard_mask=source_guard_mask,
        gate_metric=str(args.gate_metric),
        min_target_improvement=float(args.min_target_improvement),
        max_source_nfr=float(args.max_source_nfr),
        max_source_qwk_drop=float(args.max_source_qwk_drop),
        bootstrap_samples=int(args.bootstrap_samples),
        seed=int(args.seed),
        force_fail=status != "ok",
    )
    collapse = prediction_collapse_diagnostics(
        candidate_predictions[target_validation_mask],
        max_class_fraction=float(args.collapse_max_class_fraction),
    )
    if method_id == "L2_tent_unlabeled_diagonal" and collapse["collapsed"]:
        gate["failure_reasons"].append("target_prediction_collapse")
        gate["accepted"] = False

    geometry = {
        "status": "not_evaluated_preliminary_gate_failed",
        "baseline_vim_auroc": None,
        "candidate_vim_auroc": None,
        "vim_auroc_drop": None,
        "baseline_mmd2": None,
        "candidate_mmd2": None,
        "mmd_ratio": None,
    }
    if gate["accepted"]:
        geometry = geometry_diagnostics(
            features=features,
            source_train_mask=source_train_mask,
            source_guard_mask=source_guard_mask,
            target_mask=target_validation_mask,
            transform=feature_transform,
            seed=int(args.seed),
        )
        if (
            geometry["vim_auroc_drop"] is not None
            and float(geometry["vim_auroc_drop"]) > float(args.max_geometry_auroc_drop)
        ):
            gate["failure_reasons"].append("ood_vim_auroc_drop_above_threshold")
        if (
            geometry["mmd_ratio"] is not None
            and float(geometry["mmd_ratio"]) < float(args.min_geometry_mmd_ratio)
        ):
            gate["failure_reasons"].append("ood_mmd_ratio_below_threshold")
        gate["accepted"] = not gate["failure_reasons"]
    target_test_metrics = metrics(labels[target_test_mask], candidate_predictions[target_test_mask])
    row = {
        **head_prefix(spec),
        "target_domain": cell["target_domain"],
        "target_skill": cell["target_skill"],
        "shift_type": cell["shift_type"],
        "method_id": method_id,
        "label_budget_questions": int(label_budget),
        "ablation_id": ablation_id,
        "status": status,
        "error": error,
        "gate_metric": str(args.gate_metric),
        "gate_accepted": gate["accepted"],
        "gate_failure_reasons": ";".join(gate["failure_reasons"]),
        "target_validation_rows": int(target_validation_mask.sum()),
        "target_test_rows": int(target_test_mask.sum()),
        "source_guard_rows": int(source_guard_mask.sum()),
        "target_validation_improvement": gate["target_improvement"],
        "target_validation_improvement_ci_low": gate["ci_low"],
        "target_validation_improvement_ci_high": gate["ci_high"],
        "source_pm1_nfr": gate["source_pm1_nfr"],
        "source_exact_nfr": gate["source_exact_nfr"],
        "source_qwk_drop": gate["source_qwk_drop"],
        "collapse_unique_classes": collapse["unique_classes"],
        "collapse_max_class_fraction": collapse["max_class_fraction"],
        "collapse_detected": collapse["collapsed"],
        "geometry_status": geometry["status"],
        "baseline_vim_auroc": geometry["baseline_vim_auroc"],
        "candidate_vim_auroc": geometry["candidate_vim_auroc"],
        "vim_auroc_drop": geometry["vim_auroc_drop"],
        "baseline_mmd2": geometry["baseline_mmd2"],
        "candidate_mmd2": geometry["candidate_mmd2"],
        "mmd_ratio": geometry["mmd_ratio"],
        **prefixed_metrics("baseline_validation", baseline_context["target_validation_metrics"]),
        **prefixed_metrics("candidate_validation", gate["candidate_validation_metrics"]),
        **prefixed_metrics("baseline_test", baseline_context["target_test_metrics"]),
        **prefixed_metrics("candidate_test", target_test_metrics),
        **prefixed_metrics("baseline_source_guard", baseline_context["source_guard_metrics"]),
        **prefixed_metrics("candidate_source_guard", gate["candidate_source_guard_metrics"]),
        "adapter_metadata_json": json.dumps(clean_json(metadata), ensure_ascii=False),
    }
    candidate = {
        "method_id": method_id,
        "label_budget": int(label_budget),
        "ablation_id": ablation_id,
        "accepted": bool(gate["accepted"]),
        "validation_improvement": gate["target_improvement"],
        "source_qwk_drop": gate["source_qwk_drop"],
        "predictions": candidate_predictions,
        "row": row,
    }
    return row, candidate


def gate_decision(
    *,
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    validation_question_ids: np.ndarray,
    target_validation_mask: np.ndarray,
    source_guard_mask: np.ndarray,
    gate_metric: str,
    min_target_improvement: float,
    max_source_nfr: float,
    max_source_qwk_drop: float,
    bootstrap_samples: int,
    seed: int,
    force_fail: bool,
) -> dict[str, Any]:
    validation_labels = labels[target_validation_mask]
    validation_base = baseline_predictions[target_validation_mask]
    validation_candidate = candidate_predictions[target_validation_mask]
    source_labels = labels[source_guard_mask]
    source_base = baseline_predictions[source_guard_mask]
    source_candidate = candidate_predictions[source_guard_mask]
    base_validation_metrics = metrics(validation_labels, validation_base)
    candidate_validation_metrics = metrics(validation_labels, validation_candidate)
    base_source_metrics = metrics(source_labels, source_base)
    candidate_source_metrics = metrics(source_labels, source_candidate)
    improvement = metric_improvement(
        validation_labels,
        validation_base,
        validation_candidate,
        gate_metric,
    )
    ci_low, ci_high = cluster_bootstrap_improvement_ci(
        validation_labels,
        validation_base,
        validation_candidate,
        validation_question_ids,
        gate_metric=gate_metric,
        samples=bootstrap_samples,
        seed=seed,
    )
    source_pm1_nfr = float(
        np.mean((np.abs(source_base - source_labels) <= 1) & (np.abs(source_candidate - source_labels) > 1))
    )
    source_exact_nfr = float(np.mean((source_base == source_labels) & (source_candidate != source_labels)))
    source_qwk_drop = float(
        base_source_metrics["quadratic_weighted_kappa"] - candidate_source_metrics["quadratic_weighted_kappa"]
    )
    reasons: list[str] = []
    if force_fail:
        reasons.append("candidate_failed")
    if improvement < float(min_target_improvement):
        reasons.append("target_improvement_below_minimum")
    if ci_low is None or ci_low <= 0.0:
        reasons.append("target_improvement_ci_lower_not_positive")
    if source_pm1_nfr > float(max_source_nfr):
        reasons.append("source_nfr_above_threshold")
    if source_qwk_drop > float(max_source_qwk_drop):
        reasons.append("source_qwk_drop_above_threshold")
    return {
        "accepted": not reasons,
        "failure_reasons": reasons,
        "target_improvement": improvement,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "source_pm1_nfr": source_pm1_nfr,
        "source_exact_nfr": source_exact_nfr,
        "source_qwk_drop": source_qwk_drop,
        "candidate_validation_metrics": candidate_validation_metrics,
        "candidate_source_guard_metrics": candidate_source_metrics,
    }


def selected_result_row(
    *,
    spec: dict[str, Any],
    cell: dict[str, Any],
    candidates: list[dict[str, Any]],
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    baseline_context: dict[str, Any],
    target_test_mask: np.ndarray,
) -> dict[str, Any]:
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    accepted.sort(
        key=lambda item: (
            float(item["validation_improvement"]),
            -float(item["source_qwk_drop"]),
            -int(item["label_budget"]),
        ),
        reverse=True,
    )
    if accepted:
        selected = accepted[0]
        selected_method = str(selected["method_id"])
        selected_label_budget = int(selected["label_budget"])
        selected_predictions = selected["predictions"]
        rollback = False
    else:
        selected_method = "baseline_rollback"
        selected_label_budget = 0
        selected_predictions = baseline_predictions
        rollback = True
    selected_test_metrics = metrics(labels[target_test_mask], selected_predictions[target_test_mask])
    return {
        **head_prefix(spec),
        "target_domain": cell["target_domain"],
        "target_skill": cell["target_skill"],
        "shift_type": cell["shift_type"],
        "selected_method": selected_method,
        "selected_label_budget_questions": selected_label_budget,
        "rollback_to_baseline": rollback,
        "accepted_candidate_count": len(accepted),
        "target_test_rows": int(target_test_mask.sum()),
        **prefixed_metrics("baseline_test", baseline_context["target_test_metrics"]),
        **prefixed_metrics("selected_test", selected_test_metrics),
        "selected_test_pm1_improvement": (
            selected_test_metrics["plus_minus_1_accuracy"]
            - baseline_context["target_test_metrics"]["plus_minus_1_accuracy"]
        ),
        "selected_test_qwk_improvement": (
            selected_test_metrics["quadratic_weighted_kappa"]
            - baseline_context["target_test_metrics"]["quadratic_weighted_kappa"]
        ),
    }


def skip_selected_row(spec: dict[str, Any], cell: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        **head_prefix(spec),
        "target_domain": cell["target_domain"],
        "target_skill": cell["target_skill"],
        "shift_type": cell["shift_type"],
        "selected_method": "not_run",
        "selected_label_budget_questions": 0,
        "rollback_to_baseline": True,
        "accepted_candidate_count": 0,
        "target_test_rows": 0,
        "skip_reason": reason,
    }


def baseline_metrics_context(
    *,
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    target_validation_mask: np.ndarray,
    target_test_mask: np.ndarray,
    source_guard_mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "target_validation_metrics": metrics(labels[target_validation_mask], baseline_predictions[target_validation_mask]),
        "target_test_metrics": metrics(labels[target_test_mask], baseline_predictions[target_test_mask]),
        "source_guard_metrics": metrics(labels[source_guard_mask], baseline_predictions[source_guard_mask]),
    }


def build_trigger_score(
    *,
    h: np.ndarray,
    probabilities: np.ndarray,
    train_mask: np.ndarray,
    calibration_mask: np.ndarray,
    primary_loss: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    eps = 1e-12
    sorted_probs = np.sort(probabilities, axis=1)
    max_prob = sorted_probs[:, -1]
    second_prob = sorted_probs[:, -2] if probabilities.shape[1] > 1 else np.zeros_like(max_prob)
    margin = np.clip(max_prob - second_prob, 0.0, 1.0)
    entropy = -np.sum(probabilities * np.log(np.clip(probabilities, eps, 1.0)), axis=1)
    entropy = entropy / max(math.log(probabilities.shape[1]), eps)
    uncertainty = np.mean(np.stack([1.0 - max_prob, 1.0 - margin, entropy], axis=1), axis=1)
    vim = fit_residual_vim(h[train_mask], rank=min(16, int(train_mask.sum()) - 2, h.shape[1] - 1)).score(h)
    raw = np.column_stack([vim, uncertainty]).astype(np.float32)
    g_probability, g_status = fit_error_head(raw, train_mask=train_mask, losses=primary_loss, seed=seed)
    vim_ecdf = ecdf_transform(vim[calibration_mask], vim)
    uncertainty_ecdf = ecdf_transform(uncertainty[calibration_mask], uncertainty)
    g_ecdf = ecdf_transform(g_probability[calibration_mask], g_probability)
    trigger = np.mean(np.stack([vim_ecdf, uncertainty_ecdf, g_ecdf], axis=1), axis=1)
    return trigger.astype(np.float64), {
        "score_id": "rq4_style_fusion_low_confidence",
        "components": ["residual_vim_ecdf", "logit_uncertainty_ecdf", "g_error_probability_ecdf"],
        "error_head_status": g_status,
    }


def parse_int_grid(raw: str) -> list[int]:
    values = sorted(set(int(item.strip()) for item in str(raw).split(",") if item.strip()))
    if not values or any(value < 1 for value in values):
        raise ValueError("--label-budgets must contain positive integers")
    return values


def budgeted_question_splits(
    *,
    question_ids: np.ndarray,
    target_mask: np.ndarray,
    trigger_score: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    """Create one fixed group-disjoint adapt pool/validation/test split per cell."""

    target_questions = sorted(set(np.asarray(question_ids)[target_mask].astype(str).tolist()))
    if len(target_questions) < 3:
        return empty_budget_split(len(question_ids))
    if validation_fraction <= 0.0 or test_fraction <= 0.0 or validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation/test fractions must be positive and leave an adaptation pool")
    heldout_order = sorted(
        target_questions,
        key=lambda question: stable_hash(seed, namespace + "::heldout", question),
    )
    validation_n = max(1, int(round(len(target_questions) * validation_fraction)))
    test_n = max(1, int(round(len(target_questions) * test_fraction)))
    while validation_n + test_n >= len(target_questions):
        if test_n >= validation_n and test_n > 1:
            test_n -= 1
        elif validation_n > 1:
            validation_n -= 1
        else:
            return empty_budget_split(len(question_ids))
    validation_questions = set(heldout_order[:validation_n])
    test_questions = set(heldout_order[validation_n : validation_n + test_n])
    adapt_pool = [
        question
        for question in target_questions
        if question not in validation_questions and question not in test_questions
    ]
    question_score = {
        question: float(np.mean(trigger_score[target_mask & (question_ids == question)]))
        for question in adapt_pool
    }
    adapt_pool.sort(
        key=lambda question: (-question_score[question], stable_hash(seed, namespace, question))
    )
    return {
        "adapt_pool_questions": adapt_pool,
        "validation_questions": sorted(validation_questions),
        "test_questions": sorted(test_questions),
        "adapt_pool_mask": target_mask & np.isin(question_ids, np.asarray(adapt_pool)),
        "validation_mask": target_mask
        & np.isin(question_ids, np.asarray(sorted(validation_questions))),
        "test_mask": target_mask & np.isin(question_ids, np.asarray(sorted(test_questions))),
    }


def empty_budget_split(length: int) -> dict[str, Any]:
    empty = np.zeros(int(length), dtype=bool)
    return {
        "adapt_pool_questions": [],
        "validation_questions": [],
        "test_questions": [],
        "adapt_pool_mask": empty,
        "validation_mask": empty,
        "test_mask": empty,
    }


def low_confidence_question_splits(
    *,
    question_ids: np.ndarray,
    target_mask: np.ndarray,
    trigger_score: np.ndarray,
    adapt_fraction: float,
    validation_fraction: float,
    seed: int,
    namespace: str,
) -> dict[str, Any]:
    target_questions = sorted(set(np.asarray(question_ids)[target_mask].astype(str).tolist()))
    if len(target_questions) < 3:
        return empty_split(len(question_ids))
    question_score = {
        question: float(np.mean(trigger_score[target_mask & (question_ids == question)]))
        for question in target_questions
    }
    ordered_by_risk = sorted(
        target_questions,
        key=lambda question: (-question_score[question], stable_hash(seed, namespace, question)),
    )
    adapt_n = max(1, int(round(len(target_questions) * float(adapt_fraction))))
    adapt_n = min(adapt_n, len(target_questions) - 2)
    adapt_questions = set(ordered_by_risk[:adapt_n])
    remaining = [question for question in target_questions if question not in adapt_questions]
    remaining = sorted(remaining, key=lambda question: stable_hash(seed, namespace + "::gate", question))
    validation_n = max(1, int(round(len(remaining) * float(validation_fraction))))
    validation_n = min(validation_n, len(remaining) - 1)
    validation_questions = set(remaining[:validation_n])
    test_questions = set(remaining[validation_n:])
    return {
        "adapt_questions": sorted(adapt_questions),
        "validation_questions": sorted(validation_questions),
        "test_questions": sorted(test_questions),
        "adapt_mask": target_mask & np.isin(question_ids, np.asarray(sorted(adapt_questions))),
        "validation_mask": target_mask & np.isin(question_ids, np.asarray(sorted(validation_questions))),
        "test_mask": target_mask & np.isin(question_ids, np.asarray(sorted(test_questions))),
    }


def empty_split(length: int) -> dict[str, Any]:
    empty = np.zeros(int(length), dtype=bool)
    return {
        "adapt_questions": [],
        "validation_questions": [],
        "test_questions": [],
        "adapt_mask": empty,
        "validation_mask": empty,
        "test_mask": empty,
    }


def metric_improvement(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, metric_name: str) -> float:
    base_metrics = metrics(labels, baseline)
    candidate_metrics = metrics(labels, candidate)
    if metric_name == "mae":
        return float(base_metrics["mae"] - candidate_metrics["mae"])
    return float(candidate_metrics[metric_name] - base_metrics[metric_name])


def bootstrap_improvement_ci(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    *,
    gate_metric: str,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    n = len(labels)
    if n < 2 or int(samples) < 1:
        return None, None
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        local = rng.integers(0, n, size=n)
        values[index] = metric_improvement(labels[local], baseline[local], candidate[local], gate_metric)
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def cluster_bootstrap_improvement_ci(
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    question_ids: np.ndarray,
    *,
    gate_metric: str,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    """Paired bootstrap with FLASK question groups as the resampling unit."""

    groups = np.asarray(question_ids).astype(str)
    unique_groups = np.asarray(sorted(set(groups.tolist())))
    if len(unique_groups) < 2 or int(samples) < 1:
        return None, None
    indices_by_group = [np.flatnonzero(groups == group) for group in unique_groups]
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        sampled_groups = rng.integers(0, len(indices_by_group), size=len(indices_by_group))
        local = np.concatenate([indices_by_group[position] for position in sampled_groups])
        values[index] = metric_improvement(
            labels[local], baseline[local], candidate[local], gate_metric
        )
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def prediction_collapse_diagnostics(
    predictions: np.ndarray, *, max_class_fraction: float
) -> dict[str, Any]:
    values = np.asarray(predictions).astype(int)
    if len(values) == 0:
        return {"unique_classes": 0, "max_class_fraction": None, "collapsed": True}
    _, counts = np.unique(values, return_counts=True)
    maximum = float(np.max(counts) / len(values))
    return {
        "unique_classes": int(len(counts)),
        "max_class_fraction": maximum,
        "collapsed": bool(len(counts) < 2 or maximum >= float(max_class_fraction)),
    }


def geometry_diagnostics(
    *,
    features: np.ndarray,
    source_train_mask: np.ndarray,
    source_guard_mask: np.ndarray,
    target_mask: np.ndarray,
    transform: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """Revalidate residual-ViM separation and residual-vector MMD for a gate candidate."""

    h = np.asarray(features, dtype=np.float32)
    train = h[np.asarray(source_train_mask, dtype=bool)]
    source = stable_subsample(h[np.asarray(source_guard_mask, dtype=bool)], 128, seed + 11)
    target = stable_subsample(h[np.asarray(target_mask, dtype=bool)], 128, seed + 17)
    if len(train) < 3 or len(source) < 2 or len(target) < 2:
        return {
            "status": "insufficient_rows",
            "baseline_vim_auroc": None,
            "candidate_vim_auroc": None,
            "vim_auroc_drop": None,
            "baseline_mmd2": None,
            "candidate_mmd2": None,
            "mmd_ratio": None,
        }
    rank = min(16, len(train) - 2, train.shape[1] - 1)
    baseline_scorer = fit_residual_vim(train, rank)
    baseline_source_score = baseline_scorer.score(source)
    baseline_target_score = baseline_scorer.score(target)
    baseline_auroc = ood_auroc(baseline_source_score, baseline_target_score)
    baseline_source_residual = baseline_scorer.residual_features(source)
    baseline_target_residual = baseline_scorer.residual_features(target)
    baseline_mmd = rbf_mmd2(baseline_source_residual, baseline_target_residual)

    candidate_train = apply_feature_transform(train, transform)
    candidate_source = apply_feature_transform(source, transform)
    candidate_target = apply_feature_transform(target, transform)
    candidate_scorer = fit_residual_vim(candidate_train, rank)
    candidate_source_score = candidate_scorer.score(candidate_source)
    candidate_target_score = candidate_scorer.score(candidate_target)
    candidate_auroc = ood_auroc(candidate_source_score, candidate_target_score)
    candidate_mmd = rbf_mmd2(
        candidate_scorer.residual_features(candidate_source),
        candidate_scorer.residual_features(candidate_target),
    )
    ratio = 1.0 if baseline_mmd <= 1e-12 else float(candidate_mmd / baseline_mmd)
    return {
        "status": "ok",
        "baseline_vim_auroc": baseline_auroc,
        "candidate_vim_auroc": candidate_auroc,
        "vim_auroc_drop": float(baseline_auroc - candidate_auroc),
        "baseline_mmd2": baseline_mmd,
        "candidate_mmd2": candidate_mmd,
        "mmd_ratio": ratio,
    }


def apply_feature_transform(features: np.ndarray, transform: dict[str, Any] | None) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if not transform:
        return values
    if transform.get("kind") != "diagonal_affine":
        raise ValueError(f"Unsupported feature transform: {transform.get('kind')}")
    scale = np.exp(np.asarray(transform["log_scale"], dtype=np.float32))
    shift = np.asarray(transform["shift"], dtype=np.float32)
    return values * scale[None, :] + shift[None, :]


def stable_subsample(values: np.ndarray, maximum: int, seed: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if len(matrix) <= int(maximum):
        return matrix
    rng = np.random.default_rng(int(seed))
    selected = np.sort(rng.choice(len(matrix), size=int(maximum), replace=False))
    return matrix[selected]


def ood_auroc(source_scores: np.ndarray, target_scores: np.ndarray) -> float:
    labels = np.concatenate(
        [np.zeros(len(source_scores), dtype=int), np.ones(len(target_scores), dtype=int)]
    )
    scores = np.concatenate([source_scores, target_scores])
    return float(roc_auc_score(labels, scores))


def rbf_mmd2(source: np.ndarray, target: np.ndarray) -> float:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    joined = np.vstack([x, y])
    squared = pairwise_squared_distances(joined, joined)
    positive = squared[np.triu_indices_from(squared, k=1)]
    positive = positive[positive > 0.0]
    bandwidth2 = float(np.median(positive)) if len(positive) else 1.0
    gamma = 1.0 / max(2.0 * bandwidth2, 1e-12)
    kxx = np.exp(-gamma * pairwise_squared_distances(x, x))
    kyy = np.exp(-gamma * pairwise_squared_distances(y, y))
    kxy = np.exp(-gamma * pairwise_squared_distances(x, y))
    return float(max(np.mean(kxx) + np.mean(kyy) - 2.0 * np.mean(kxy), 0.0))


def pairwise_squared_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    return np.maximum(left_norm + right_norm - 2.0 * (left @ right.T), 0.0)


def metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        return {
            "mae": float("nan"),
            "exact_accuracy": float("nan"),
            "plus_minus_1_accuracy": float("nan"),
            "quadratic_weighted_kappa": float("nan"),
        }
    value = judge_metrics(labels, predictions, class_values=CLASSES)
    return {
        "mae": float(value["mae"]),
        "exact_accuracy": float(value["accuracy"]),
        "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
        "quadratic_weighted_kappa": float(value["qwk"]),
    }


def prefixed_metrics(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def class_balanced_ce(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=int(num_classes)).to(dtype=logits.dtype)
    present = counts > 0
    weights = torch.zeros(int(num_classes), dtype=logits.dtype, device=logits.device)
    weights[present] = targets.numel() / (present.sum().to(dtype=logits.dtype) * counts[present])
    return nn.functional.cross_entropy(logits, targets, weight=weights)


def fit_residual_vim(features: np.ndarray, rank: int) -> ViMScorer:
    errors: list[str] = []
    for candidate in range(max(1, int(rank)), 0, -1):
        try:
            return ViMScorer(rank=candidate).fit(features)
        except ValueError as exc:
            errors.append(f"rank={candidate}: {exc}")
    raise RuntimeError("Could not fit residual ViM; " + " | ".join(errors[-3:]))


def fit_error_head(
    features: np.ndarray,
    *,
    train_mask: np.ndarray,
    losses: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, str]:
    y = np.asarray(losses[train_mask], dtype=int)
    if len(np.unique(y)) < 2:
        prior = float(np.mean(y)) if len(y) else 0.0
        return np.full(len(features), prior, dtype=np.float64), "constant_prior"
    scaler = StandardScaler().fit(features[train_mask])
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=int(seed))
    model.fit(scaler.transform(features[train_mask]), y)
    return model.predict_proba(scaler.transform(features))[:, 1].astype(np.float64), "ok"


def ecdf_transform(calibration_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(calibration_scores, dtype=np.float64))
    if len(reference) == 0:
        return np.zeros(len(scores), dtype=np.float64)
    return np.searchsorted(reference, np.asarray(scores, dtype=np.float64), side="right") / float(len(reference))


def target_cell_masks(
    *,
    payload: dict[str, np.ndarray],
    source_domain: str,
    source_skill: str,
    train_questions: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_question_array = np.asarray(sorted(train_questions))
    for target_domain in ordered_targets(payload["domains"], TARGET_DOMAINS):
        for target_skill in ordered_targets(payload["skills"], TARGET_SKILLS):
            if target_domain == source_domain and target_skill == source_skill:
                continue
            mask = (
                (payload["domains"] == target_domain)
                & (payload["skills"] == target_skill)
                & ~np.isin(payload["question_ids"], train_question_array)
            )
            if not mask.any():
                continue
            rows.append(
                {
                    "target_domain": str(target_domain),
                    "target_skill": str(target_skill),
                    "shift_type": shift_type(source_domain, source_skill, str(target_domain), str(target_skill)),
                    "mask": mask,
                }
            )
    return rows


def load_feature_payload(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as cache:
        features = np.asarray(cache["features"], dtype=np.float16)
        if features.ndim == 2:
            features = features[:, None, :]
        sample_ids = np.asarray(cache["sample_ids"]).astype(str)
        labels = np.asarray(cache["labels"], dtype=int)
        domains = np.asarray(cache["domain_ids"]).astype(str)
        skills = np.asarray(cache["task_ids"]).astype(str)
        question_ids = np.asarray(cache["query_ids"]).astype(str)
    if features.ndim != 3:
        raise ValueError(f"Expected [N,L,D] B-space features, got {features.shape}")
    return {
        "features": features,
        "sample_ids": sample_ids,
        "labels": labels,
        "domains": domains,
        "skills": skills,
        "question_ids": question_ids,
    }


def load_head_specs(heads_dir: Path) -> list[dict[str, Any]]:
    summary_path = heads_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    specs: list[dict[str, Any]] = []
    for head in summary.get("heads", []):
        split_path = resolve_repo_path(head["split_path"])
        model_path = resolve_repo_path(head["model_path"])
        split = json.loads(split_path.read_text(encoding="utf-8"))
        specs.append(
            {
                "head_id": str(head["head_id"]),
                "source_domain": str(head["source_domain"]),
                "source_skill": str(head["source_skill"]),
                "model_path": model_path,
                "split_path": split_path,
                "split": split,
            }
        )
    order = {
        (domain, skill): i
        for i, (domain, skill) in enumerate((d, s) for d in SOURCE_DOMAINS for s in SOURCE_SKILLS)
    }
    specs.sort(key=lambda item: order[(item["source_domain"], item["source_skill"])])
    if len(specs) != 9:
        raise ValueError(f"Expected nine heads in {heads_dir}, found {len(specs)}")
    return specs


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    return raw if raw.is_absolute() else ROOT / raw


def stable_question_sample(questions: list[str], *, fraction: float, seed: int, namespace: str) -> set[str]:
    if not questions:
        return set()
    n = max(1, int(round(len(questions) * fraction)))
    n = min(n, len(questions))
    ordered = sorted(
        questions,
        key=lambda question: hashlib.sha256(f"{seed}::{namespace}::{question}".encode("utf-8")).hexdigest(),
    )
    return set(ordered[:n])


def stable_hash(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}::{namespace}::{value}".encode("utf-8")).hexdigest()


def ordered_targets(values: np.ndarray, preferred: tuple[str, ...]) -> list[str]:
    present = set(values.astype(str).tolist())
    return [value for value in preferred if value in present] + sorted(present.difference(preferred))


def shift_type(source_domain: str, source_skill: str, target_domain: str, target_skill: str) -> str:
    if source_domain == target_domain and source_skill == target_skill:
        return "ID test"
    if source_domain != target_domain and source_skill == target_skill:
        return "Domain shift"
    if source_domain == target_domain and source_skill != target_skill:
        return "Task shift"
    return "Joint shift"


def mean_or_none(values: np.ndarray) -> float | None:
    arr = np.asarray(values)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def head_prefix(spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "head_id": spec["head_id"],
        "source_domain": spec["source_domain"],
        "source_skill": spec["source_skill"],
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fields})


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def rq5_level_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "level": "L0",
            "method_id": "L0_head_only",
            "status": "implemented",
            "description": "Copied deployed head adapted on low-confidence target probe labels plus source replay.",
        },
        {
            "level": "L1",
            "method_id": "L1_diagonal_feature_alignment",
            "status": "implemented",
            "description": "Frozen classifier with learned diagonal affine feature alignment and source geometry anchor.",
        },
        {
            "level": "L2",
            "method_id": "L2_tent_unlabeled_diagonal",
            "status": "disabled_by_flag" if bool(args.disable_l2_tent) else "implemented",
            "description": "TENT-style unlabeled target entropy minimization on diagonal affine parameters with source replay.",
        },
        {
            "level": "L3",
            "method_id": "L3_lora",
            "status": "not_applicable_cached_prelogit_features",
            "description": "LoRA needs base LLM weights and hidden-state extraction, not only cached strict prelogit B-space.",
        },
    ]


def build_gate_summary(candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = {}
    for row in candidate_rows:
        key = (
            str(row["method_id"]),
            int(row.get("label_budget_questions", 0)),
            str(row.get("ablation_id", "primary")),
            str(row["shift_type"]),
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (method_id, budget, ablation_id, shift), rows in sorted(groups.items()):
        out.append(
            {
                "method_id": method_id,
                "label_budget_questions": budget,
                "ablation_id": ablation_id,
                "shift_type": shift,
                "candidate_rows": len(rows),
                "accepted_rows": int(sum(bool(row["gate_accepted"]) for row in rows)),
                "acceptance_rate": float(np.mean([bool(row["gate_accepted"]) for row in rows])) if rows else None,
                "mean_target_validation_improvement": safe_mean(
                    [row["target_validation_improvement"] for row in rows]
                ),
                "mean_source_pm1_nfr": safe_mean([row["source_pm1_nfr"] for row in rows]),
                "mean_source_qwk_drop": safe_mean([row["source_qwk_drop"] for row in rows]),
                "collapse_rate": float(np.mean([bool(row["collapse_detected"]) for row in rows])),
                "mean_vim_auroc_drop_evaluated": safe_mean([row["vim_auroc_drop"] for row in rows]),
                "mean_mmd_ratio_evaluated": safe_mean([row["mmd_ratio"] for row in rows]),
                "mean_candidate_test_pm1": safe_mean([row["candidate_test_plus_minus_1_accuracy"] for row in rows]),
                "mean_baseline_test_pm1": safe_mean([row["baseline_test_plus_minus_1_accuracy"] for row in rows]),
                "mean_candidate_test_qwk": safe_mean([row["candidate_test_quadratic_weighted_kappa"] for row in rows]),
                "mean_baseline_test_qwk": safe_mean([row["baseline_test_quadratic_weighted_kappa"] for row in rows]),
            }
        )
    return out


def safe_mean(values: list[Any]) -> float | None:
    clean: list[float] = []
    for value in values:
        if value in ("", None):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            clean.append(numeric)
    return float(np.mean(clean)) if clean else None


def build_summary(
    *,
    args: argparse.Namespace,
    payload: dict[str, np.ndarray],
    specs: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
    level_rows: list[dict[str, Any]],
    elapsed_seconds: float,
) -> dict[str, Any]:
    selected_by_shift: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        selected_by_shift.setdefault(str(row["shift_type"]), []).append(row)
    selected_summary = []
    for shift, rows in sorted(selected_by_shift.items()):
        selected_summary.append(
            {
                "shift_type": shift,
                "cell_rows": len(rows),
                "rollback_rate": float(np.mean([bool(row["rollback_to_baseline"]) for row in rows])),
                "mean_selected_test_pm1_improvement": safe_mean(
                    [row.get("selected_test_pm1_improvement") for row in rows]
                ),
                "mean_selected_test_qwk_improvement": safe_mean(
                    [row.get("selected_test_qwk_improvement") for row in rows]
                ),
            }
        )
    return {
        "artifact_type": "flask_rq5_gated_adaptation_v2",
        "source_features": str(args.features),
        "heads_dir": str(args.heads_dir),
        "feature_rows": int(len(payload["sample_ids"])),
        "feature_shape": list(payload["features"].shape),
        "head_count": len(specs),
        "gate": {
            "metric": str(args.gate_metric),
            "min_target_improvement": float(args.min_target_improvement),
            "ci_lower_must_be_positive": True,
            "max_source_pm1_nfr": float(args.max_source_nfr),
            "max_source_qwk_drop": float(args.max_source_qwk_drop),
            "bootstrap_samples": int(args.bootstrap_samples),
            "bootstrap_unit": "FLASK question group (paired cluster bootstrap)",
            "max_geometry_auroc_drop": float(args.max_geometry_auroc_drop),
            "min_geometry_mmd_ratio": float(args.min_geometry_mmd_ratio),
            "collapse_max_class_fraction": float(args.collapse_max_class_fraction),
        },
        "trigger": {
            "strategy": "select target adaptation questions by highest RQ4-style fusion risk",
            "adapt_question_fraction": float(args.adapt_question_fraction),
            "validation_question_fraction": float(args.validation_question_fraction),
            "test_question_fraction": float(args.test_question_fraction),
            "label_budgets": parse_int_grid(args.label_budgets),
            "selection_contract": (
                "Validation/test question groups are held out first; the disjoint adaptation pool "
                "is then ranked by RQ4-style risk. Budgets are exact question-group counts."
            ),
        },
        "levels": level_rows,
        "candidate_result_rows": len(candidate_rows),
        "selected_result_rows": len(selected_rows),
        "gate_summary": gate_rows,
        "selected_summary_by_shift": selected_summary,
        "outputs": {
            "adaptation_cell_results_csv": str(args.output_dir / "adaptation_cell_results.csv"),
            "selected_cell_results_csv": str(args.output_dir / "selected_cell_results.csv"),
            "gate_summary_csv": str(args.output_dir / "gate_summary.csv"),
            "low_confidence_trigger_summary_csv": str(args.output_dir / "low_confidence_trigger_summary.csv"),
            "rq5_level_summary_csv": str(args.output_dir / "rq5_level_summary.csv"),
        },
        "elapsed_seconds": float(elapsed_seconds),
    }


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact_type": summary["artifact_type"],
        "head_count": summary["head_count"],
        "candidate_result_rows": summary["candidate_result_rows"],
        "selected_result_rows": summary["selected_result_rows"],
        "gate": summary["gate"],
        "selected_summary_by_shift": summary["selected_summary_by_shift"],
        "elapsed_seconds": summary["elapsed_seconds"],
    }


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    return value


if __name__ == "__main__":
    main()
