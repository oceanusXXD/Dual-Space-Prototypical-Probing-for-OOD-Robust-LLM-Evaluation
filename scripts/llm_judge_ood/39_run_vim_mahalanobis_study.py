#!/usr/bin/env python
"""Audit and compare ViM/Mahalanobis using only frozen hidden-state caches.

Hyperparameters are selected on source-only pseudo-OOD groups. Official OOD
rows are scored only after the selected configuration has been frozen.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import scipy
import sklearn
import torch
from scipy.special import logsumexp
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json
from src.llm_judge_ood.shared.metrics import ood_metrics


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    prepared: str
    hidden: str
    ordered: bool
    train_split: str
    validation_split: str
    id_test_split: str
    ood_test_split: str


SPECS = {
    "ellipse": DatasetSpec(
        "ellipse",
        "artifacts/llm_judge_ood_ellipse/ellipse_prepared_contract_v1.jsonl",
        "hiddenstates/ellipse/qwen3_5_4b_judge_input_overall_v1.npz",
        True,
        "training_train",
        "training_validation",
        "benchmark_test",
        "benchmark_test",
    ),
    "asap": DatasetSpec(
        "asap",
        "artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl",
        "hiddenstates/asap_aes/qwen3_5_4b_judge_input_asap_rubric_v1.npz",
        True,
        "training_train",
        "training_validation",
        "benchmark_test",
        "benchmark_test",
    ),
    "clinc150": DatasetSpec(
        "clinc150",
        "artifacts/llm_judge_ood_clinc150/clinc150_prepared_contract_v1.jsonl",
        "hiddenstates/clinc150/qwen3_5_4b_judge_input_intent_v1.npz",
        False,
        "train",
        "val",
        "test",
        "oos_test",
    ),
    "rostd": DatasetSpec(
        "rostd",
        "artifacts/llm_judge_ood_rostd/rostd_prepared_contract_v1.jsonl",
        "hiddenstates/rostd/qwen3_5_4b_judge_input_intent_v1.npz",
        False,
        "train",
        "eval",
        "test",
        "test",
    ),
}

HEADS = ("linear_softmax", "mlp_softmax", "ordinal", "regression")
LAMBDAS = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
TEMPERATURES = (0.5, 1.0, 2.0, 5.0, 10.0)
VARIANCE_TARGETS = (0.80, 0.85, 0.90, 0.95, 0.97, 0.99)
RESIDUAL_DIMS = (32, 64, 128, 256)


@dataclass
class HeadOutput:
    penultimate: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    weight: np.ndarray | None
    bias: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), required=True)
    parser.add_argument(
        "--output-root",
        default="artifacts/docs_experiments/vim_mahalanobis_study_seed42",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-dim", type=int, default=128)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = SPECS[str(args.dataset)]
    output = Path(args.output_root) / spec.name
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(manifest_path.read_text(encoding="utf-8"))
        return
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows = read_jsonl(spec.prepared)
    cache = np.load(spec.hidden, allow_pickle=True)
    hidden = np.asarray(cache["features"], dtype=np.float32)
    sample_ids = np.asarray(cache["sample_ids"]).astype(str)
    expected_ids = np.asarray([str(row["sample_id"]) for row in rows])
    if not np.array_equal(sample_ids, expected_ids):
        raise RuntimeError(f"Hiddenstate row alignment failed for {spec.name}")
    metadata = json.loads(str(np.asarray(cache["metadata_json"]).item()))
    labels = np.asarray([str(row["label"]) for row in rows])
    splits = np.asarray([str(row["split"]) for row in rows])
    truth = np.asarray(
        [bool(row.get("is_document_ood", row.get("is_ood", False))) for row in rows]
    )
    masks = _masks(spec, splits, truth)
    pseudo_groups, held_groups = _pseudo_groups(spec, rows, masks["train"], int(args.seed))
    pseudo_train = masks["train"] & ~np.isin(pseudo_groups, held_groups)
    pseudo_ood = masks["train"] & np.isin(pseudo_groups, held_groups)
    if not pseudo_train.any() or not pseudo_ood.any():
        raise RuntimeError("Source-only pseudo-OOD split is empty")
    layers = _available_layers(hidden, metadata)

    audit = _audit_payload(spec, hidden, cache, rows, masks, metadata, held_groups)
    write_json(output / "audit.json", audit)

    pseudo_rows: list[dict[str, Any]] = []
    head_rows: list[dict[str, Any]] = []
    best_by_head_layer: dict[tuple[str, str], dict[str, Any]] = {}
    official_models: dict[tuple[str, str], tuple[PCA, HeadOutput]] = {}
    head_names = HEADS if spec.ordered else HEADS[:2]
    for layer_name, layer_index in layers:
        raw = _layer_values(hidden, layer_index)
        pseudo_pca = _fit_pca(raw[pseudo_train], int(args.pca_dim), int(args.seed))
        pseudo_x = pseudo_pca.transform(raw).astype(np.float64)
        formal_pca = _fit_pca(raw[masks["train"]], int(args.pca_dim), int(args.seed))
        formal_x = formal_pca.transform(raw).astype(np.float64)
        del raw
        for head_name in head_names:
            pseudo_head = _fit_head(
                head_name,
                pseudo_x[pseudo_train],
                labels[pseudo_train],
                pseudo_x,
                ordered=spec.ordered,
                seed=int(args.seed),
            )
            if head_name != "regression":
                candidates = _vim_grid(
                    pseudo_head,
                    labels,
                    pseudo_train,
                    pseudo_ood,
                    dataset=spec.name,
                    head=head_name,
                    layer=layer_name,
                )
                pseudo_rows.extend(candidates)
                best_by_head_layer[(head_name, layer_name)] = _best_vim(candidates)

            formal_head = _fit_head(
                head_name,
                formal_x[masks["train"]],
                labels[masks["train"]],
                formal_x,
                ordered=spec.ordered,
                seed=int(args.seed),
            )
            official_models[(head_name, layer_name)] = (formal_pca, formal_head)
            head_rows.append(
                _head_metrics_row(
                    spec,
                    head_name,
                    layer_name,
                    labels[masks["validation"]],
                    formal_head.predictions[masks["validation"]],
                )
            )

    _write_csv(output / "head_comparison_source_id.csv", head_rows)
    _write_csv(output / "pseudo_ood_vim_grid.csv", pseudo_rows)
    selected_head = _select_head(head_rows, ordered=spec.ordered)
    selected_layer, selected_vim = _select_layer_and_vim(
        selected_head, layers, best_by_head_layer
    )
    frozen = {
        "head": selected_head,
        "layer": selected_layer,
        "vim": selected_vim,
        "selection_data": "source-only pseudo-OOD; official OOD test not read for selection",
        "pseudo_held_groups": held_groups.tolist(),
    }
    write_json(output / "frozen_selection.json", frozen)

    formal_rows: list[dict[str, Any]] = []
    sample_scores: dict[str, np.ndarray] = {}
    benchmark = masks["benchmark"]
    y_benchmark = truth[benchmark].astype(int)
    for head_name in (name for name in head_names if name != "regression"):
        layer_name, candidate = _best_layer_for_head(head_name, layers, best_by_head_layer)
        head = official_models[(head_name, layer_name)][1]
        methods = _formal_scores(head, labels, masks["train"], candidate)
        for method, scores in methods.items():
            metrics = ood_metrics(y_benchmark, scores[benchmark])
            formal_rows.append(
                {
                    "dataset": spec.name,
                    "head": head_name,
                    "layer": layer_name,
                    "method": method,
                    **metrics,
                    "selected_on": "source_only_pseudo_ood",
                }
            )
            sample_scores[f"{head_name}::{method}"] = scores
    _write_csv(output / "formal_ood_results.csv", formal_rows)

    selected_methods = {
        key.split("::", 1)[1]: value
        for key, value in sample_scores.items()
        if key.startswith(f"{selected_head}::")
    }
    np.savez_compressed(
        output / "formal_sample_scores.npz",
        sample_ids=sample_ids,
        truth=truth.astype(np.int8),
        benchmark_mask=benchmark.astype(np.int8),
        **{_safe_key(name): score.astype(np.float32) for name, score in selected_methods.items()},
    )
    per_group = _per_group_rows(spec, rows, truth, benchmark, selected_methods)
    _write_csv(output / "formal_per_prompt_domain.csv", per_group)
    ci_rows, paired_rows = _uncertainty_rows(
        spec.name,
        truth[benchmark].astype(int),
        {name: score[benchmark] for name, score in selected_methods.items()},
        bootstrap=int(args.bootstrap),
        seed=int(args.seed),
    )
    _write_csv(output / "bootstrap_95ci.csv", ci_rows)
    _write_csv(output / "paired_vim_vs_mahalanobis.csv", paired_rows)
    manifest = {
        "artifact_type": "vim_mahalanobis_cached_hiddenstate_study_v1",
        "dataset": spec.name,
        "seed": int(args.seed),
        "hiddenstate_forward_passes": 0,
        "prepared": spec.prepared,
        "prepared_sha256": _sha256(Path(spec.prepared)),
        "hidden": spec.hidden,
        "hidden_sha256": _sha256(Path(spec.hidden)),
        "hidden_shape": list(hidden.shape),
        "hidden_metadata": metadata,
        "available_layers": [name for name, _ in layers],
        "pooling": metadata.get("pooling", metadata.get("pooling_method", "masked_mean")),
        "unavailable_requested_representations": [
            "last_token",
            "judge_or_rubric_token",
            "layer_-2",
            "layer_-4",
            "last_4_layer_average",
        ],
        "pseudo_train_rows": int(pseudo_train.sum()),
        "pseudo_ood_rows": int(pseudo_ood.sum()),
        "frozen_selection": frozen,
        "files": {
            "audit": str(output / "audit.json"),
            "head_comparison": str(output / "head_comparison_source_id.csv"),
            "pseudo_grid": str(output / "pseudo_ood_vim_grid.csv"),
            "formal": str(output / "formal_ood_results.csv"),
            "per_group": str(output / "formal_per_prompt_domain.csv"),
            "bootstrap": str(output / "bootstrap_95ci.csv"),
            "paired": str(output / "paired_vim_vs_mahalanobis.csv"),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "elapsed_seconds": float(time.perf_counter() - started),
        "command": [sys.executable, *sys.argv],
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def _masks(spec: DatasetSpec, splits: np.ndarray, truth: np.ndarray) -> dict[str, np.ndarray]:
    train = (splits == spec.train_split) & ~truth
    validation = (splits == spec.validation_split) & ~truth
    id_test = (splits == spec.id_test_split) & ~truth
    ood_test = (splits == spec.ood_test_split) & truth
    return {
        "train": train,
        "validation": validation,
        "id_test": id_test,
        "ood_test": ood_test,
        "benchmark": id_test | ood_test,
    }


def _pseudo_groups(
    spec: DatasetSpec,
    rows: list[dict[str, Any]],
    train: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if spec.name in {"ellipse", "asap"}:
        groups = np.asarray([str(row["audit_document_group_id"]) for row in rows])
        source_group_count = len(set(groups[train].tolist()))
        count = (
            max(1, int(math.ceil(source_group_count * 0.20)))
            if spec.name == "ellipse"
            else 1
        )
    elif spec.name == "clinc150":
        groups = np.asarray([str(row.get("domain_name", "unknown")) for row in rows])
        count = 1
    else:
        groups = np.asarray([str(row.get("intent_name", row["label"])) for row in rows])
        count = max(1, int(math.ceil(len(set(groups[train].tolist())) * 0.20)))
    available = sorted(set(groups[train].tolist()), key=lambda x: _stable_hash(x, seed))
    return groups, np.asarray(available[:count], dtype=str)


def _available_layers(hidden: np.ndarray, metadata: dict[str, Any]) -> list[tuple[str, int]]:
    resolved = (
        metadata.get("layers_resolved")
        or metadata.get("resolved_layer_indices")
        or metadata.get("layers")
        or []
    )
    if len(resolved) != hidden.shape[1]:
        resolved = list(range(hidden.shape[1]))
    names = []
    for index, layer in enumerate(resolved):
        name = "last_layer" if index == hidden.shape[1] - 1 else f"layer_{layer}"
        names.append((name, index))
    return names


def _layer_values(hidden: np.ndarray, layer_index: int) -> np.ndarray:
    return np.asarray(hidden[:, int(layer_index), :], dtype=np.float32)


def _fit_pca(values: np.ndarray, requested_dim: int, seed: int) -> PCA:
    dim = min(int(requested_dim), values.shape[0] - 1, values.shape[1])
    return PCA(
        n_components=dim,
        whiten=True,
        svd_solver="randomized",
        random_state=int(seed),
    ).fit(np.asarray(values, dtype=np.float32))


def _fit_head(
    name: str,
    train_x: np.ndarray,
    train_labels: np.ndarray,
    all_x: np.ndarray,
    *,
    ordered: bool,
    seed: int,
) -> HeadOutput:
    classes = _sorted_classes(train_labels, numeric=ordered)
    index = {value: position for position, value in enumerate(classes.tolist())}
    y_index = np.asarray([index[value] for value in train_labels.tolist()], dtype=int)
    if name == "linear_softmax":
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=250,
            solver="lbfgs",
            random_state=int(seed),
        ).fit(train_x, train_labels)
        logits = np.asarray(model.decision_function(all_x), dtype=np.float64)
        if logits.ndim == 1:
            logits = np.column_stack([np.zeros(len(logits)), logits])
        model_classes = np.asarray(model.classes_).astype(str)
        predictions = model_classes[np.argmax(logits, axis=1)]
        weight = np.asarray(model.coef_, dtype=np.float64).T
        bias = np.asarray(model.intercept_, dtype=np.float64)
        if weight.shape[1] == 1:
            weight = np.column_stack([np.zeros(weight.shape[0]), weight[:, 0]])
            bias = np.asarray([0.0, bias[0]])
        return HeadOutput(all_x, logits, _softmax(logits), predictions, model_classes, weight, bias)
    if name == "mlp_softmax":
        model = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            alpha=1e-4,
            batch_size=min(200, len(train_x)),
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=6,
            random_state=int(seed),
        ).fit(train_x, train_labels)
        penultimate = np.maximum(0.0, all_x @ model.coefs_[0] + model.intercepts_[0])
        logits = penultimate @ model.coefs_[1] + model.intercepts_[1]
        model_classes = np.asarray(model.classes_).astype(str)
        predictions = model_classes[np.argmax(logits, axis=1)]
        return HeadOutput(
            penultimate,
            logits,
            _softmax(logits),
            predictions,
            model_classes,
            np.asarray(model.coefs_[1], dtype=np.float64),
            np.asarray(model.intercepts_[1], dtype=np.float64),
        )
    if not ordered:
        raise ValueError(f"{name} is only defined for ordered score labels")
    numeric_classes = classes.astype(float)
    y_numeric = train_labels.astype(float)
    if name == "regression":
        model = Ridge(alpha=1.0).fit(train_x, y_numeric)
        scalar = np.asarray(model.predict(all_x), dtype=np.float64)
        scale = max(float(np.std(model.predict(train_x) - y_numeric)), 0.25)
        logits = -0.5 * ((scalar[:, None] - numeric_classes[None, :]) / scale) ** 2
        predictions = classes[np.argmax(logits, axis=1)]
        return HeadOutput(all_x, logits, _softmax(logits), predictions, classes, None, None)
    if name == "ordinal":
        return _fit_ordinal_softmax_head(train_x, y_index, all_x, classes, seed)
    raise ValueError(f"Unknown head {name}")


def _fit_ordinal_softmax_head(
    train_x: np.ndarray,
    y_index: np.ndarray,
    all_x: np.ndarray,
    classes: np.ndarray,
    seed: int,
) -> HeadOutput:
    torch.manual_seed(int(seed))
    x_tensor = torch.as_tensor(train_x, dtype=torch.float32)
    y_tensor = torch.as_tensor(y_index, dtype=torch.long)
    linear = torch.nn.Linear(train_x.shape[1], len(classes))
    counts = torch.bincount(y_tensor, minlength=len(classes)).float()
    weights = torch.sqrt(counts.sum() / torch.clamp(counts * len(classes), min=1.0))
    optimizer = torch.optim.AdamW(linear.parameters(), lr=3e-3, weight_decay=1e-4)
    rng = np.random.default_rng(int(seed))
    for _ in range(80):
        order = rng.permutation(len(train_x))
        for start in range(0, len(order), 512):
            batch = torch.as_tensor(order[start : start + 512], dtype=torch.long)
            logits = linear(x_tensor[batch])
            probabilities = torch.softmax(logits, dim=1)
            predicted_cdf = torch.cumsum(probabilities, dim=1)[:, :-1]
            target_cdf = (
                torch.arange(len(classes) - 1)[None, :] >= y_tensor[batch, None]
            ).float()
            ce = torch.nn.functional.cross_entropy(logits, y_tensor[batch], weight=weights)
            ordinal = torch.mean((predicted_cdf - target_cdf) ** 2)
            loss = ce + ordinal
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    weight = linear.weight.detach().cpu().numpy().T.astype(np.float64)
    bias = linear.bias.detach().cpu().numpy().astype(np.float64)
    logits = all_x @ weight + bias
    predictions = classes[np.argmax(logits, axis=1)]
    return HeadOutput(all_x, logits, _softmax(logits), predictions, classes, weight, bias)


def _vim_grid(
    output: HeadOutput,
    labels: np.ndarray,
    train: np.ndarray,
    pseudo_ood: np.ndarray,
    *,
    dataset: str,
    head: str,
    layer: str,
) -> list[dict[str, Any]]:
    h_train = np.asarray(output.penultimate[train], dtype=np.float64)
    h_eval = np.asarray(output.penultimate[train | pseudo_ood], dtype=np.float64)
    logits_train = np.asarray(output.logits[train], dtype=np.float64)
    logits_eval = np.asarray(output.logits[train | pseudo_ood], dtype=np.float64)
    y = pseudo_ood[train | pseudo_ood].astype(int)
    source_local = train[train | pseudo_ood]
    rows: list[dict[str, Any]] = []
    variants = _residual_variants(h_train, h_eval, output.weight, output.bias)
    for variant, payload in variants.items():
        for policy, rank in _rank_policies(payload["singular_values"], h_train.shape[1]):
            residual = _residual_score(payload, rank)
            source_residual = residual[source_local]
            alpha = float(np.max(logits_train, axis=1).mean()) / max(float(source_residual.mean()), 1e-12)
            evidence = alpha * residual
            for temperature in TEMPERATURES:
                logit_evidence = float(temperature) * logsumexp(
                    logits_eval / float(temperature), axis=1
                )
                for lam in LAMBDAS:
                    if lam == 0.0 and temperature != 1.0:
                        continue
                    score = evidence - float(lam) * logit_evidence
                    metrics = ood_metrics(y, score)
                    rows.append(
                        {
                            "dataset": dataset,
                            "head": head,
                            "layer": layer,
                            "variant": variant,
                            "rank_policy": policy,
                            "rank": int(rank),
                            "lambda": float(lam),
                            "temperature": float(temperature),
                            "alpha": alpha,
                            **metrics,
                        }
                    )
    class_rows = _class_conditional_candidates(
        h_train,
        h_eval,
        labels[train],
        output.classes,
        logits_train,
        logits_eval,
        y,
        source_local,
        dataset=dataset,
        head=head,
        layer=layer,
    )
    rows.extend(class_rows)
    return rows


def _residual_variants(
    h_train: np.ndarray,
    h_eval: np.ndarray,
    weight: np.ndarray | None,
    bias: np.ndarray | None,
) -> dict[str, dict[str, np.ndarray]]:
    variants: dict[str, dict[str, np.ndarray]] = {}
    mean = h_train.mean(axis=0)
    variants["raw_residual"] = _svd_payload(h_train - mean, h_eval - mean, whiten=False)
    variants["whitened_residual"] = _svd_payload(h_train - mean, h_eval - mean, whiten=True)
    norms_train = np.maximum(np.linalg.norm(h_train, axis=1, keepdims=True), 1e-12)
    norms_eval = np.maximum(np.linalg.norm(h_eval, axis=1, keepdims=True), 1e-12)
    normalized_train = h_train / norms_train
    normalized_eval = h_eval / norms_eval
    normalized_mean = normalized_train.mean(axis=0)
    variants["l2_normalized_residual"] = _svd_payload(
        normalized_train - normalized_mean,
        normalized_eval - normalized_mean,
        whiten=False,
    )
    if weight is not None and bias is not None:
        origin = -np.linalg.pinv(np.asarray(weight, dtype=np.float64).T) @ np.asarray(
            bias, dtype=np.float64
        )
        variants["origin_raw_standard_vim"] = _svd_payload(
            h_train - origin,
            h_eval - origin,
            whiten=False,
        )
    return variants


def _svd_payload(train_centered: np.ndarray, eval_centered: np.ndarray, *, whiten: bool) -> dict[str, np.ndarray]:
    _, singular, right = np.linalg.svd(train_centered, full_matrices=False)
    return {
        "train_centered": train_centered,
        "eval_centered": eval_centered,
        "right": right,
        "singular_values": singular,
        "whiten": np.asarray(int(whiten)),
    }


def _rank_policies(singular: np.ndarray, dim: int) -> list[tuple[str, int]]:
    variance = singular**2
    cumulative = np.cumsum(variance) / max(float(variance.sum()), 1e-12)
    values: list[tuple[str, int]] = []
    for target in VARIANCE_TARGETS:
        rank = min(int(np.searchsorted(cumulative, target) + 1), dim - 1)
        values.append((f"explained_variance_{target:.2f}", rank))
    for residual_dim in RESIDUAL_DIMS:
        if residual_dim < dim:
            values.append((f"fixed_residual_dim_{residual_dim}", dim - residual_dim))
    seen: set[tuple[str, int]] = set()
    return [item for item in values if not (item in seen or seen.add(item))]


def _resolve_rank(policy: str, singular: np.ndarray, dim: int) -> int:
    if policy.startswith("explained_variance_"):
        target = float(policy.rsplit("_", 1)[1])
        cumulative = np.cumsum(singular**2) / max(float(np.sum(singular**2)), 1e-12)
        return min(int(np.searchsorted(cumulative, target) + 1), dim - 1)
    residual_dim = int(policy.rsplit("_", 1)[1])
    return max(1, dim - min(residual_dim, dim - 1))


def _residual_score(payload: dict[str, np.ndarray], rank: int) -> np.ndarray:
    right = payload["right"]
    residual_basis = right[int(rank) :]
    projection = payload["eval_centered"] @ residual_basis.T
    if bool(payload["whiten"]):
        source_projection = payload["train_centered"] @ residual_basis.T
        scale = np.std(source_projection, axis=0, ddof=1)
        projection = projection / np.maximum(scale, 1e-6)
    return np.linalg.norm(projection, axis=1)


def _class_conditional_candidates(
    h_train: np.ndarray,
    h_eval: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    logits_train: np.ndarray,
    logits_eval: np.ndarray,
    y: np.ndarray,
    source_local: np.ndarray,
    *,
    dataset: str,
    head: str,
    layer: str,
) -> list[dict[str, Any]]:
    scores = np.full(len(h_eval), np.inf, dtype=np.float64)
    predicted = np.asarray(classes).astype(str)[np.argmax(logits_eval, axis=1)]
    for label in sorted(set(labels.tolist())):
        local = h_train[labels == label]
        if len(local) < 4:
            continue
        target = predicted == str(label)
        if not target.any():
            continue
        mean = local.mean(axis=0)
        _, singular, right = np.linalg.svd(local - mean, full_matrices=False)
        variance = np.cumsum(singular**2) / max(float(np.sum(singular**2)), 1e-12)
        rank = min(int(np.searchsorted(variance, 0.90) + 1), right.shape[0] - 1)
        centered = h_eval[target] - mean
        residual = centered - centered @ right[:rank].T @ right[:rank]
        scores[target] = np.linalg.norm(residual, axis=1)
    if not np.isfinite(scores).all():
        return []
    alpha = float(np.max(logits_train, axis=1).mean()) / max(
        float(scores[source_local].mean()), 1e-12
    )
    rows = []
    for temperature in TEMPERATURES:
        logit_evidence = float(temperature) * logsumexp(logits_eval / float(temperature), axis=1)
        for lam in LAMBDAS:
            if lam == 0.0 and temperature != 1.0:
                continue
            metrics = ood_metrics(y, alpha * scores - float(lam) * logit_evidence)
            rows.append(
                {
                    "dataset": dataset,
                    "head": head,
                    "layer": layer,
                    "variant": "class_conditional_residual",
                    "rank_policy": "per_class_explained_variance_0.90",
                    "rank": -1,
                    "lambda": float(lam),
                    "temperature": float(temperature),
                    "alpha": alpha,
                    **metrics,
                }
            )
    return rows


def _best_vim(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        max(
            rows,
            key=lambda row: (
                float(row["auroc"]),
                float(row["aupr"]),
                -float(row["fpr95"]),
                -float(row["lambda"]),
            ),
        )
    )


def _head_metrics_row(
    spec: DatasetSpec,
    head: str,
    layer: str,
    truth: np.ndarray,
    prediction: np.ndarray,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": spec.name,
        "head": head,
        "layer": layer,
        "validation_rows": len(truth),
        "accuracy": float(accuracy_score(truth, prediction)),
    }
    if spec.ordered:
        y = truth.astype(float)
        p = prediction.astype(float)
        levels = sorted(set(y.tolist()) | set(p.tolist()))
        level_index = {value: index for index, value in enumerate(levels)}
        y_qwk = np.asarray([level_index[value] for value in y], dtype=int)
        p_qwk = np.asarray([level_index[value] for value in p], dtype=int)
        row.update(
            {
                "qwk": float(cohen_kappa_score(y_qwk, p_qwk, weights="quadratic")),
                "mae": float(np.mean(np.abs(y - p))),
                "spearman": float(spearmanr(y, p).statistic),
            }
        )
    else:
        row.update({"qwk": "", "mae": "", "spearman": ""})
    return row


def _select_head(rows: list[dict[str, Any]], *, ordered: bool) -> str:
    candidates = [row for row in rows if row["layer"] == "last_layer" and row["head"] != "regression"]
    key = "qwk" if ordered else "accuracy"
    return str(max(candidates, key=lambda row: float(row[key]))["head"])


def _select_layer_and_vim(
    head: str,
    layers: list[tuple[str, int]],
    best: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    layer = max(
        (name for name, _ in layers),
        key=lambda name: (
            float(best[(head, name)]["auroc"]),
            float(best[(head, name)]["aupr"]),
            -float(best[(head, name)]["fpr95"]),
        ),
    )
    return layer, best[(head, layer)]


def _best_layer_for_head(
    head: str,
    layers: list[tuple[str, int]],
    best: dict[tuple[str, str], dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    return _select_layer_and_vim(head, layers, best)


def _formal_scores(
    output: HeadOutput,
    labels: np.ndarray,
    train: np.ndarray,
    selected: dict[str, Any],
) -> dict[str, np.ndarray]:
    h_train = np.asarray(output.penultimate[train], dtype=np.float64)
    h_all = np.asarray(output.penultimate, dtype=np.float64)
    logits_train = np.asarray(output.logits[train], dtype=np.float64)
    variants = _residual_variants(h_train, h_all, output.weight, output.bias)
    variant = str(selected["variant"])
    if variant == "class_conditional_residual":
        selected_score = _class_conditional_score(
            h_train, h_all, labels[train], output.predictions
        )
        rank = None
    else:
        payload = variants[variant]
        rank = _resolve_rank(str(selected["rank_policy"]), payload["singular_values"], h_train.shape[1])
        selected_score = _residual_score(payload, rank)
    alpha = float(np.max(logits_train, axis=1).mean()) / max(
        float(selected_score[train].mean()), 1e-12
    )
    temperature = float(selected["temperature"])
    lam = float(selected["lambda"])
    selected_fused = alpha * selected_score - lam * temperature * logsumexp(
        output.logits / temperature, axis=1
    )
    raw_payload = variants["raw_residual"]
    raw_rank = rank if rank is not None else _resolve_rank("explained_variance_0.90", raw_payload["singular_values"], h_train.shape[1])
    raw_residual = _residual_score(raw_payload, raw_rank)
    origin_payload = variants.get("origin_raw_standard_vim")
    methods = {
        "Selected adapted ViM": selected_fused,
        "Residual-only ViM": raw_residual,
    }
    if origin_payload is not None:
        origin_rank = (
            _resolve_rank(
                str(selected["rank_policy"]),
                origin_payload["singular_values"],
                h_train.shape[1],
            )
            if variant != "class_conditional_residual"
            else _resolve_rank(
                "explained_variance_0.90",
                origin_payload["singular_values"],
                h_train.shape[1],
            )
        )
        origin_residual = _residual_score(origin_payload, origin_rank)
        origin_alpha = float(np.max(logits_train, axis=1).mean()) / max(
            float(origin_residual[train].mean()), 1e-12
        )
        methods["Standard Full ViM"] = origin_alpha * origin_residual - logsumexp(
            output.logits, axis=1
        )
    methods.update(_mahalanobis_scores(h_train, h_all, labels[train]))
    return methods


def _class_conditional_score(
    h_train: np.ndarray,
    h_all: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    scores = np.full(len(h_all), np.inf, dtype=np.float64)
    for label in sorted(set(labels.tolist())):
        local = h_train[labels == label]
        if len(local) < 4:
            continue
        target = np.asarray(predictions).astype(str) == str(label)
        if not target.any():
            continue
        mean = local.mean(axis=0)
        _, singular, right = np.linalg.svd(local - mean, full_matrices=False)
        cumulative = np.cumsum(singular**2) / max(float(np.sum(singular**2)), 1e-12)
        rank = min(int(np.searchsorted(cumulative, 0.90) + 1), right.shape[0] - 1)
        centered = h_all[target] - mean
        residual = centered - centered @ right[:rank].T @ right[:rank]
        scores[target] = np.linalg.norm(residual, axis=1)
    return scores


def _mahalanobis_scores(
    h_train: np.ndarray,
    h_all: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    classes = sorted(set(labels.tolist()))
    means = np.stack([h_train[labels == label].mean(axis=0) for label in classes])
    residuals = np.vstack([h_train[labels == label] - means[index] for index, label in enumerate(classes)])
    dim = h_train.shape[1]
    empirical_cov = residuals.T @ residuals / max(len(residuals) - 1, 1)
    empirical_precision = np.linalg.pinv(empirical_cov + 1e-5 * np.eye(dim))
    shrink_cov = LedoitWolf().fit(residuals).covariance_ + 1e-5 * np.eye(dim)
    shrink_precision = np.linalg.pinv(shrink_cov)
    diagonal = np.maximum(np.var(residuals, axis=0, ddof=1), 1e-5)
    class_covariances = []
    for index, label in enumerate(classes):
        local = h_train[labels == label] - means[index]
        class_covariances.append(local.T @ local / max(len(local) - 1, 1))
    balanced_cov = np.mean(class_covariances, axis=0) + 1e-5 * np.eye(dim)
    balanced_precision = np.linalg.pinv(balanced_cov)
    global_mean = h_train.mean(axis=0)
    global_cov = LedoitWolf().fit(h_train - global_mean).covariance_ + 1e-5 * np.eye(dim)
    global_precision = np.linalg.pinv(global_cov)
    shared_empirical = _min_mahalanobis(h_all, means, empirical_precision)
    shared_shrinkage = _min_mahalanobis(h_all, means, shrink_precision)
    diagonal_score = _min_diagonal(h_all, means, diagonal)
    class_balanced = _min_mahalanobis(h_all, means, balanced_precision)
    global_score = _quadratic(h_all - global_mean, global_precision)
    return {
        "Mahalanobis shared empirical": shared_empirical,
        "Mahalanobis shrinkage": shared_shrinkage,
        "Mahalanobis diagonal": diagonal_score,
        "Mahalanobis class-balanced": class_balanced,
        "RMD": shared_shrinkage - global_score,
    }


def _min_mahalanobis(values: np.ndarray, means: np.ndarray, precision: np.ndarray) -> np.ndarray:
    projected = values @ precision
    value_term = np.sum(projected * values, axis=1)
    mean_term = np.sum((means @ precision) * means, axis=1)
    distances = value_term[:, None] - 2.0 * projected @ means.T + mean_term[None, :]
    return np.min(distances, axis=1)


def _min_diagonal(values: np.ndarray, means: np.ndarray, variance: np.ndarray) -> np.ndarray:
    precision = 1.0 / variance
    projected = values * precision
    value_term = np.sum(projected * values, axis=1)
    mean_term = np.sum(means * means * precision, axis=1)
    distances = value_term[:, None] - 2.0 * projected @ means.T + mean_term[None, :]
    return np.min(distances, axis=1)


def _quadratic(centered: np.ndarray, precision: np.ndarray) -> np.ndarray:
    return np.einsum("ij,jk,ik->i", centered, precision, centered)


def _per_group_rows(
    spec: DatasetSpec,
    rows: list[dict[str, Any]],
    truth: np.ndarray,
    benchmark: np.ndarray,
    methods: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    if spec.name in {"ellipse", "asap"}:
        groups = np.asarray([str(row["audit_document_group_id"]) for row in rows])
    else:
        groups = np.asarray([str(row.get("domain_name", row["audit_document_group_id"])) for row in rows])
    ood = benchmark & truth
    output = []
    for group in sorted(set(groups[benchmark & ~truth].tolist())):
        local = ood | (benchmark & ~truth & (groups == group))
        if len(np.unique(truth[local])) < 2:
            continue
        for method, scores in methods.items():
            output.append(
                {
                    "dataset": spec.name,
                    "id_prompt_or_domain": group,
                    "method": method,
                    "id_rows": int((local & ~truth).sum()),
                    "ood_rows": int((local & truth).sum()),
                    **ood_metrics(truth[local].astype(int), scores[local]),
                }
            )
    return output


def _uncertainty_rows(
    dataset: str,
    truth: np.ndarray,
    methods: dict[str, np.ndarray],
    *,
    bootstrap: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(seed))
    id_indices = np.where(truth == 0)[0]
    ood_indices = np.where(truth == 1)[0]
    boot = {method: [] for method in methods}
    for _ in range(int(bootstrap)):
        indices = np.concatenate(
            [rng.choice(id_indices, len(id_indices), replace=True), rng.choice(ood_indices, len(ood_indices), replace=True)]
        )
        local_truth = truth[indices]
        for method, scores in methods.items():
            boot[method].append(float(roc_auc_score(local_truth, scores[indices])))
    ci_rows = []
    for method, values in boot.items():
        ci_rows.append(
            {
                "dataset": dataset,
                "method": method,
                "auroc": float(ood_metrics(truth, methods[method])["auroc"]),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "bootstrap_samples": int(bootstrap),
            }
        )
    vim_name = "Selected adapted ViM"
    maha_names = [name for name in methods if name.startswith("Mahalanobis")]
    best_maha = max(maha_names, key=lambda name: ood_metrics(truth, methods[name])["auroc"])
    differences = np.asarray(boot[vim_name]) - np.asarray(boot[best_maha])
    p_value = float(2.0 * min(np.mean(differences <= 0.0), np.mean(differences >= 0.0)))
    paired = [
        {
            "dataset": dataset,
            "method_a": vim_name,
            "method_b": best_maha,
            "auroc_difference": float(
                ood_metrics(truth, methods[vim_name])["auroc"]
                - ood_metrics(truth, methods[best_maha])["auroc"]
            ),
            "difference_ci_low": float(np.quantile(differences, 0.025)),
            "difference_ci_high": float(np.quantile(differences, 0.975)),
            "paired_bootstrap_p_two_sided": min(p_value, 1.0),
            "bootstrap_samples": int(bootstrap),
        }
    ]
    return ci_rows, paired


def _audit_payload(
    spec: DatasetSpec,
    hidden: np.ndarray,
    cache: Any,
    rows: list[dict[str, Any]],
    masks: dict[str, np.ndarray],
    metadata: dict[str, Any],
    held_groups: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(cache["labels"]).astype(str)
    expected_labels = np.asarray([str(row["label"]) for row in rows])
    return {
        "dataset": spec.name,
        "score_direction": "higher_is_ood; verified by ood_metrics contract",
        "alpha_fit_scope": "source training rows only",
        "row_alignment": bool(
            np.array_equal(np.asarray(cache["sample_ids"]).astype(str), np.asarray([row["sample_id"] for row in rows]))
        ),
        "label_alignment": bool(np.array_equal(labels, expected_labels)),
        "hidden_shape": list(hidden.shape),
        "logit_contract": "classification heads emit K>=2 class logits; regression is Judge-only control",
        "pca_center": "residual-only uses source mean; standard Full ViM uses -pinv(W.T)@b",
        "softmax_axis": 1,
        "fit_rows": int(masks["train"].sum()),
        "id_test_rows": int(masks["id_test"].sum()),
        "ood_test_rows": int(masks["ood_test"].sum()),
        "pseudo_held_groups": held_groups.tolist(),
        "hidden_metadata": metadata,
    }


def _sorted_classes(labels: np.ndarray, *, numeric: bool) -> np.ndarray:
    values = sorted(set(labels.astype(str).tolist()), key=(lambda x: float(x)) if numeric else None)
    return np.asarray(values, dtype=str)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def _stable_hash(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}::{value}".encode("utf-8")).hexdigest()


def _safe_key(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        writer.writerows(rows)


if __name__ == "__main__":
    main()
