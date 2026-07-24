from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from src.algorithm.classifier.base import LinearJudgeConfig, PerQueryLinearJudge
from src.algorithm.data.decisions import decision_rows_from_scores
from src.algorithm.data.monitoring import confirm_window_failures
from src.algorithm.detector.knn import KNNScorer
from src.algorithm.detector.openood import OPENOOD_POSTHOC_METHODS, OpenOODPosthocScorer
from src.algorithm.detector.residual_vim import FullViMScorer, ViMScorer
from src.algorithm.detector.rmd import MahalanobisScorer, RMDScorer
from src.algorithm.hidden_state.extract import extract_hidden_states
from src.algorithm.wsr.certification import certify_wsr_thresholds, normalized_absolute_error
from src.common.feature_store import HiddenFeatureStore, load_hidden_feature_store
from src.common.io import read_jsonl, write_json, write_jsonl
from src.common.metrics import normalize_label_array
from src.common.schema import JudgeRecord, load_judge_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.algorithm")
    subcommands = parser.add_subparsers(dest="command", required=True)

    extract = subcommands.add_parser("extract", help="Extract A/B hidden-state caches")
    extract.add_argument("--records", nargs="+", required=True)
    extract.add_argument("--space", choices=("a", "b"), required=True)
    extract.add_argument("--view", required=True)
    extract.add_argument("--layers", nargs="+", type=int, default=[-1])
    extract.add_argument("--pooling", default=None)
    extract.add_argument("--model-path", required=True)
    extract.add_argument("--model-id", default=None)
    extract.add_argument("--revision", default=None)
    extract.add_argument("--output", required=True)
    extract.add_argument("--batch-size", type=int, default=1)
    extract.add_argument("--max-length", type=int, default=2048)
    extract.add_argument("--device", default="cuda")
    extract.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    extract.add_argument("--attn-implementation", choices=("eager", "sdpa", "flash_attention_2"), default="sdpa")
    extract.add_argument("--local-files-only", action="store_true")
    extract.set_defaults(func=extract_command)

    train = subcommands.add_parser("train-classifier", help="Train a classifier over hidden-state caches")
    train.add_argument("--features", required=True)
    train.add_argument("--records", required=True)
    train.add_argument("--train-split", required=True)
    train.add_argument("--val-split", default=None)
    train.add_argument("--classifier", choices=("linear", "linear_softmax", "coral", "ridge", "logistic"), default="coral")
    train.add_argument("--output", required=True)
    train.add_argument("--representation", default="last_layer")
    train.add_argument("--pca-dim", type=int, default=48)
    train.add_argument("--alpha", type=float, default=10.0)
    train.add_argument("--c", type=float, default=0.1)
    train.add_argument("--max-iter", type=int, default=500)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--head-sharing", choices=("shared", "per_query"), default="shared")
    train.set_defaults(func=train_classifier_command)

    detect = subcommands.add_parser("detect", help="Fit and score OOD detectors")
    detect.add_argument("--features", required=True)
    detect.add_argument("--records", required=True)
    detect.add_argument("--classifier", required=True)
    detect.add_argument("--fit-split", required=True)
    detect.add_argument("--calibration-split", default=None)
    detect.add_argument("--detectors", default="residual_vim,mahalanobis,rmd,knn")
    detect.add_argument("--output", required=True)
    detect.add_argument("--vim-rank", type=int, default=64)
    detect.add_argument("--knn-k", type=int, default=10)
    detect.add_argument("--soft-quantile", type=float, default=0.90)
    detect.add_argument("--hard-quantile", type=float, default=0.95)
    detect.set_defaults(func=detect_command)

    certify = subcommands.add_parser("certify-wsr", help="Certify selective-risk thresholds")
    _add_wsr_arguments(certify)
    certify.set_defaults(func=certify_wsr_command)

    apply_threshold = subcommands.add_parser("apply-threshold", help="Apply a WSR threshold row by row")
    apply_threshold.add_argument("--scores", required=True)
    apply_threshold.add_argument("--thresholds", required=True)
    apply_threshold.add_argument("--output", required=True)
    apply_threshold.add_argument("--detector", default=None)
    apply_threshold.add_argument("--threshold", type=float, default=None)
    apply_threshold.add_argument("--accept-above", action="store_true")
    apply_threshold.set_defaults(func=apply_threshold_command)

    confirm_window = subcommands.add_parser("confirm-window", help="Confirm window-level failure from decisions")
    confirm_window.add_argument("--decisions", required=True)
    confirm_window.add_argument("--window-size", type=int, required=True)
    confirm_window.add_argument("--min-reject-rate", type=float, required=True)
    confirm_window.add_argument("--output", required=True)
    confirm_window.set_defaults(func=confirm_window_command)

    monitor = subcommands.add_parser("update-monitor", help="Summarize update-monitor inputs")
    monitor.add_argument("--a-features", required=True)
    monitor.add_argument("--b-features", required=True)
    monitor.add_argument("--classifier", required=True)
    monitor.add_argument("--stream-split", required=True)
    monitor.add_argument("--mmd-permutations", type=int, default=1000)
    monitor.add_argument("--output", required=True)
    monitor.set_defaults(func=update_monitor_command)

    adapt = subcommands.add_parser("update-adapt", help="Create an adapted classifier artifact")
    adapt.add_argument("--features", required=True)
    adapt.add_argument("--classifier", required=True)
    adapt.add_argument("--labels", required=True)
    adapt.add_argument("--mode", choices=("head", "affine", "tent"), default="affine")
    adapt.add_argument("--gate-split", default=None)
    adapt.add_argument("--output", required=True)
    adapt.set_defaults(func=update_adapt_command)

    recertify = subcommands.add_parser("recertify-wsr", help="Re-run WSR certification after model update")
    _add_wsr_arguments(recertify)
    recertify.set_defaults(func=recertify_wsr_command)
    return parser


def _add_wsr_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scores", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--risk-loss", choices=("normalized_mae",), default="normalized_mae")
    parser.add_argument("--risk-bound", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--detector", default=None)
    parser.add_argument("--calibration-split", default="training_calibration")
    parser.add_argument("--max-candidates", type=int, default=32)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)


def extract_command(args: argparse.Namespace) -> dict[str, Any]:
    return extract_hidden_states(
        records=[Path(value) for value in args.records],
        output=args.output,
        model_path=args.model_path,
        model_id=args.model_id,
        revision=args.revision,
        space=args.space,
        view=args.view,
        layers=tuple(args.layers),
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=bool(args.local_files_only),
    )


def train_classifier_command(args: argparse.Namespace) -> dict[str, Any]:
    store = load_hidden_feature_store(args.features)
    records = _align_records(load_judge_records([args.records]), store)
    labels = normalize_label_array([record.label for record in records])
    queries = np.asarray([record.query_id for record in records]).astype(str)
    splits = np.asarray([record.split for record in records]).astype(str)
    train_mask = splits == str(args.train_split)
    validation_mask = np.zeros(len(records), dtype=bool) if args.val_split is None else splits == str(args.val_split)
    if not train_mask.any():
        raise ValueError(f"train split {args.train_split!r} selected no rows")
    method = _classifier_method(args.classifier)
    config = LinearJudgeConfig(
        method=method,
        alpha=float(args.alpha),
        c=float(args.c),
        max_iter=int(args.max_iter),
        representation=str(args.representation),
        pca_dim=int(args.pca_dim),
        class_values=tuple(np.unique(labels[train_mask]).tolist()),
        seed=int(args.seed),
        head_sharing=str(args.head_sharing),
    )
    model = PerQueryLinearJudge(config).fit(
        store.features,
        labels,
        queries,
        train_mask=train_mask,
        validation_mask=validation_mask,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    metadata = {
        "artifact_type": "classifier",
        "classifier": args.classifier,
        "method": method,
        "output": str(output),
        "rows": len(records),
        "train_rows": int(train_mask.sum()),
        "validation_rows": int(validation_mask.sum()),
        "model": model.to_metadata(),
    }
    write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    return metadata


def detect_command(args: argparse.Namespace) -> dict[str, Any]:
    store = load_hidden_feature_store(args.features)
    records = _align_records(load_judge_records([args.records]), store)
    classifier = _load_classifier(args.classifier)
    labels = normalize_label_array([record.label for record in records])
    queries = np.asarray([record.query_id for record in records]).astype(str)
    splits = np.asarray([record.split for record in records]).astype(str)
    fit_mask = splits == str(args.fit_split)
    calibration_mask = fit_mask if args.calibration_split is None else splits == str(args.calibration_split)
    if not fit_mask.any():
        raise ValueError(f"fit split {args.fit_split!r} selected no rows")
    penultimate, logits, predictions = _classifier_outputs(classifier, store.features, queries)
    detector_names = [item.strip().lower() for item in str(args.detectors).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "artifact_type": "detector_scores",
        "detectors": detector_names,
        "fit_rows": int(fit_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "candidates": [],
    }
    for name in detector_names:
        scorer, canonical = _fit_detector(
            name,
            penultimate=penultimate,
            logits=logits,
            labels=labels,
            query_ids=queries,
            fit_mask=fit_mask,
            classifier=classifier,
            vim_rank=int(args.vim_rank),
            knn_k=int(args.knn_k),
        )
        scores = _score_detector(
            scorer,
            canonical=canonical,
            penultimate=penultimate,
            logits=logits,
            query_ids=queries,
        )
        if calibration_mask.any() and hasattr(scorer, "calibrate"):
            try:
                _calibrate_detector(
                    scorer,
                    canonical=canonical,
                    penultimate=penultimate[calibration_mask],
                    logits=logits[calibration_mask] if logits is not None else None,
                    query_ids=queries[calibration_mask],
                    soft_q=float(args.soft_quantile),
                    hard_q=float(args.hard_quantile),
                )
            except Exception as error:
                metadata["candidates"].append(
                    {"detector": canonical, "calibration_error": str(error)}
                )
        metadata["candidates"].append(
            {"detector": canonical, "score_min": float(scores.min()), "score_max": float(scores.max())}
        )
        for index, score in enumerate(scores.tolist()):
            rows.append(
                {
                    "sample_id": str(store.sample_ids[index]),
                    "query_id": str(queries[index]),
                    "split": str(splits[index]),
                    "label": _json_scalar(labels[index]),
                    "prediction": _json_scalar(predictions[index]),
                    "detector": canonical,
                    "score": float(score),
                }
            )
    write_jsonl(args.output, rows)
    metadata["output"] = str(args.output)
    write_json(Path(args.output).with_suffix(Path(args.output).suffix + ".metadata.json"), metadata)
    return metadata


def certify_wsr_command(args: argparse.Namespace) -> dict[str, Any]:
    score_rows = read_jsonl(args.scores)
    if not score_rows:
        raise ValueError("scores file is empty")
    detector = args.detector or str(score_rows[0].get("detector") or "score")
    selected_score_rows = [row for row in score_rows if str(row.get("detector") or "score") == detector]
    if not selected_score_rows:
        raise ValueError(f"no score rows found for detector {detector!r}")
    predictions_by_id = {str(row["sample_id"]): row for row in read_jsonl(args.predictions)}
    predictions: list[float] = []
    labels: list[float] = []
    scores: list[float] = []
    calibration_indices: list[int] = []
    for index, row in enumerate(selected_score_rows):
        sample_id = str(row["sample_id"])
        payload = predictions_by_id.get(sample_id, row)
        if "prediction" not in payload or "label" not in payload:
            raise ValueError(f"prediction row for sample {sample_id!r} lacks prediction/label")
        predictions.append(float(payload["prediction"]))
        labels.append(float(payload["label"]))
        scores.append(float(row["score"]))
        split = str(row.get("split") or payload.get("split") or "")
        if split == str(args.calibration_split):
            calibration_indices.append(index)
    if not calibration_indices:
        calibration_indices = list(range(len(scores)))
    y_values = np.asarray(predictions + labels, dtype=np.float64)
    y_min = float(args.y_min) if args.y_min is not None else float(np.min(y_values))
    y_max = float(args.y_max) if args.y_max is not None else float(np.max(y_values))
    losses = normalized_absolute_error(predictions, labels, y_min=y_min, y_max=y_max)
    result = certify_wsr_thresholds(
        scores=scores,
        losses=losses,
        calibration_indices=calibration_indices,
        risk_bound=float(args.risk_bound),
        delta=float(args.delta),
        max_candidates=int(args.max_candidates),
    )
    result = {
        **result,
        "artifact_type": "wsr_thresholds",
        "detector": detector,
        "risk_loss": args.risk_loss,
        "score_rows": len(scores),
        "calibration_rows": len(calibration_indices),
        "y_min": y_min,
        "y_max": y_max,
    }
    write_json(args.output, result)
    return result


def apply_threshold_command(args: argparse.Namespace) -> dict[str, Any]:
    score_rows = read_jsonl(args.scores)
    thresholds = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else _threshold_from_payload(thresholds, detector=args.detector)
    )
    detector = args.detector or thresholds.get("detector")
    rows = decision_rows_from_scores(
        score_rows,
        threshold=threshold,
        detector=detector,
        accept_below=not bool(args.accept_above),
    )
    write_jsonl(args.output, rows)
    metadata = {
        "artifact_type": "accept_reject_decisions",
        "scores": str(args.scores),
        "thresholds": str(args.thresholds),
        "detector": detector,
        "threshold": threshold,
        "rows": len(rows),
        "accepted_rows": int(sum(1 for row in rows if row["accepted"])),
        "rejected_rows": int(sum(1 for row in rows if not row["accepted"])),
        "output": str(args.output),
    }
    write_json(Path(args.output).with_suffix(Path(args.output).suffix + ".metadata.json"), metadata)
    return metadata


def confirm_window_command(args: argparse.Namespace) -> dict[str, Any]:
    decision_rows = read_jsonl(args.decisions)
    windows = confirm_window_failures(
        decision_rows,
        window_size=int(args.window_size),
        min_reject_rate=float(args.min_reject_rate),
    )
    write_json(args.output, {"artifact_type": "window_failure_confirmation", "windows": windows})
    return {
        "artifact_type": "window_failure_confirmation",
        "decisions": str(args.decisions),
        "window_size": int(args.window_size),
        "min_reject_rate": float(args.min_reject_rate),
        "window_count": len(windows),
        "failed_window_count": int(sum(1 for row in windows if row["failure_confirmed"])),
        "output": str(args.output),
    }


def recertify_wsr_command(args: argparse.Namespace) -> dict[str, Any]:
    result = certify_wsr_command(args)
    result["artifact_type"] = "wsr_recertification"
    result["recertification"] = True
    write_json(args.output, result)
    return result


def update_monitor_command(args: argparse.Namespace) -> dict[str, Any]:
    a_store = load_hidden_feature_store(args.a_features)
    b_store = load_hidden_feature_store(args.b_features)
    _load_classifier(args.classifier)
    metadata = {
        "artifact_type": "update_monitoring_summary",
        "status": "input_summary_written",
        "a_feature_shape": list(a_store.features.shape),
        "b_feature_shape": list(b_store.features.shape),
        "stream_split": str(args.stream_split),
        "mmd_permutations": int(args.mmd_permutations),
        "classifier": str(args.classifier),
    }
    write_json(args.output, metadata)
    return metadata


def update_adapt_command(args: argparse.Namespace) -> dict[str, Any]:
    load_hidden_feature_store(args.features)
    classifier = _load_classifier(args.classifier)
    label_rows = read_jsonl(args.labels)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, output)
    metadata = {
        "artifact_type": "adapted_classifier",
        "status": "copied_source_classifier_with_probe_metadata",
        "mode": str(args.mode),
        "gate_split": args.gate_split,
        "probe_label_rows": len(label_rows),
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".metadata.json"), metadata)
    return metadata


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = args.func(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=_json_scalar))


def _align_records(records: list[JudgeRecord], store: HiddenFeatureStore) -> list[JudgeRecord]:
    by_id = {str(record.sample_id): record for record in records}
    aligned: list[JudgeRecord] = []
    missing: list[str] = []
    for sample_id in np.asarray(store.sample_ids).astype(str).tolist():
        record = by_id.get(sample_id)
        if record is None:
            missing.append(sample_id)
        else:
            aligned.append(record)
    if missing:
        raise ValueError(f"records file is missing feature sample ids: {missing[:5]}")
    return aligned


def _classifier_method(name: str) -> str:
    normalized = str(name).lower()
    if normalized in {"linear", "linear_softmax"}:
        return "linear"
    if normalized in {"coral", "ridge", "logistic"}:
        return normalized
    raise ValueError(f"unsupported classifier: {name}")


def _load_classifier(path: str | Path) -> Any:
    payload = joblib.load(path)
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    return payload


def _classifier_outputs(
    classifier: Any,
    features: np.ndarray,
    query_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    try:
        output = classifier.predict_output(features, query_ids)
        predictions = output.classes[np.argmax(output.probabilities, axis=1)]
        return output.penultimate, output.logits, predictions
    except Exception:
        penultimate = classifier.transform_u(features)
        predictions = classifier.predict(features, query_ids)
        return penultimate, None, predictions


def _fit_detector(
    name: str,
    *,
    penultimate: np.ndarray,
    logits: np.ndarray | None,
    labels: np.ndarray,
    query_ids: np.ndarray,
    fit_mask: np.ndarray,
    classifier: Any,
    vim_rank: int,
    knn_k: int,
) -> tuple[Any, str]:
    canonical = "residual_vim" if name in {"vim", "residual_vim"} else name
    fit_features = np.asarray(penultimate, dtype=np.float64)[fit_mask]
    if canonical == "residual_vim":
        max_rank = max(1, min(fit_features.shape[0] - 2, fit_features.shape[1] - 1))
        rank = max(1, min(int(vim_rank), max_rank))
        return ViMScorer(rank=rank).fit(fit_features), canonical
    if canonical == "mahalanobis":
        return MahalanobisScorer().fit(fit_features), canonical
    if canonical == "rmd":
        return RMDScorer().fit(fit_features, labels[fit_mask]), canonical
    if canonical == "knn":
        k = max(1, min(int(knn_k), int(fit_features.shape[0])))
        return KNNScorer(k=k, normalize=True).fit(fit_features), canonical
    if canonical == "full_vim":
        if logits is None:
            raise ValueError("full_vim requires classifier logits")
        head = _affine_head_parameters(classifier)
        if head is None:
            raise ValueError("full_vim requires classifier.affine_head_parameters()")
        weights, biases, head_query_ids = head
        max_rank = max(1, min(fit_features.shape[0] - 2, fit_features.shape[1] - 1))
        rank = max(1, min(int(vim_rank), max_rank))
        return (
            FullViMScorer(rank=rank).fit(
                fit_features,
                logits[fit_mask],
                head_weight=weights,
                head_bias=biases,
                query_ids=query_ids[fit_mask],
                head_query_ids=head_query_ids,
            ),
            canonical,
        )
    openood_method = _openood_method(canonical)
    if openood_method is not None:
        if logits is None:
            raise ValueError(f"{canonical} requires classifier logits")
        scorer = OpenOODPosthocScorer(method=openood_method)
        head = _affine_head_parameters(classifier)
        head_kwargs: dict[str, Any] = {}
        if head is not None:
            weights, biases, head_query_ids = head
            head_kwargs = {
                "head_weight": weights,
                "head_bias": biases,
                "head_query_ids": head_query_ids,
            }
        scorer.fit(
            fit_features,
            logits[fit_mask],
            labels=labels[fit_mask],
            query_ids=query_ids[fit_mask],
            **head_kwargs,
        )
        return scorer, f"openood_{openood_method}"
    raise ValueError(f"unsupported detector: {name}")


def _score_detector(
    scorer: Any,
    *,
    canonical: str,
    penultimate: np.ndarray,
    logits: np.ndarray | None,
    query_ids: np.ndarray,
) -> np.ndarray:
    if isinstance(scorer, (OpenOODPosthocScorer, FullViMScorer)):
        if logits is None:
            raise ValueError(f"{canonical} requires classifier logits")
        return np.asarray(scorer.score(penultimate, logits, query_ids), dtype=np.float64)
    return np.asarray(scorer.score(penultimate), dtype=np.float64)


def _calibrate_detector(
    scorer: Any,
    *,
    canonical: str,
    penultimate: np.ndarray,
    logits: np.ndarray | None,
    query_ids: np.ndarray,
    soft_q: float,
    hard_q: float,
) -> Any:
    if isinstance(scorer, (OpenOODPosthocScorer, FullViMScorer)):
        if logits is None:
            raise ValueError(f"{canonical} requires classifier logits")
        return scorer.calibrate(penultimate, logits, query_ids=query_ids, soft_q=soft_q, hard_q=hard_q)
    return scorer.calibrate(penultimate, soft_q=soft_q, hard_q=hard_q)


def _openood_method(name: str) -> str | None:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized.startswith("openood_"):
        normalized = normalized.removeprefix("openood_")
    if normalized in OPENOOD_POSTHOC_METHODS and normalized != "mahalanobis":
        return normalized
    return None


def _affine_head_parameters(classifier: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not hasattr(classifier, "affine_head_parameters"):
        return None
    try:
        return classifier.affine_head_parameters()
    except Exception:
        return None


def _threshold_from_payload(payload: dict[str, Any], *, detector: str | None) -> float:
    if payload.get("threshold") is not None:
        return float(payload["threshold"])
    candidates = payload.get("candidates")
    if isinstance(candidates, list):
        certified = [row for row in candidates if row.get("certified")]
        if detector is not None:
            certified = [row for row in certified if str(row.get("detector") or detector) == str(detector)]
        if certified:
            return float(max(certified, key=lambda row: float(row.get("coverage", 0.0)))["threshold"])
    raise ValueError("threshold payload does not contain a selected threshold")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
