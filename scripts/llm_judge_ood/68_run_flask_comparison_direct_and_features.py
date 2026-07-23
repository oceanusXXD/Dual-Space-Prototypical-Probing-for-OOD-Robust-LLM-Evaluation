#!/usr/bin/env python3
"""Run FLASK Direct Judge and capture strict final-prelogit features.

The input rows must already carry the shared split produced by script 67. This
single pass writes Direct Judge predictions and the frozen hidden features used
by the classification-head baseline.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_judge_ood.flask_comparison import (
    DIRECT_JUDGE_TEMPLATE_VERSION,
    apply_chat_template,
    cell_id,
    cell_sort_key,
    head_tail,
    integer_score,
    metrics_from_predictions,
    parse_direct_score,
    read_jsonl,
    row_cell,
    slug,
    stable_rank,
    write_csv,
    write_json,
    write_jsonl,
)
from src.models.extract_hidden import (
    load_qwen_model,
    qwen_text_config,
)

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DEFAULT_LOCAL_MODEL_PATH = Path("models/qwen3.5-0.8b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/comparison_rows.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/split_manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/direct_and_features"),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-batch-tokens", type=int, default=196608)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--pad-to-multiple-of", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    validate_runtime(args)
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.output_dir / "strict_final_prelogit.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "direct_judge_predictions.jsonl"
    scored_rows_path = args.output_dir / "b_space_with_direct_judge.jsonl"
    features_path = args.output_dir / "strict_final_prelogit_features.npz"
    summary_path = args.output_dir / "summary.json"

    rows = load_rows(args.rows, args.split_manifest)
    selected_ids = [str(row["b_id"]) for row in rows]
    prediction_rows = read_jsonl(predictions_path) if predictions_path.exists() else []
    completed_predictions = {str(row["b_id"]) for row in prediction_rows}
    completed_features = feature_part_ids(parts_dir)
    if completed_predictions != completed_features:
        raise ValueError("Prediction rows and feature parts are not aligned; pass --overwrite.")
    pending = [row for row in rows if str(row["b_id"]) not in completed_predictions]

    runtime: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(args.model_path),
        "prompt_template_version": DIRECT_JUDGE_TEMPLATE_VERSION,
        "feature_scope": "strict_final_prelogit",
        "feature_definition": "final-normalized decoder state at last prompt token before first generated score digit",
        "rows": len(rows),
        "batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "max_length": int(args.max_length),
    }

    if pending:
        tokenizer, model, device = load_qwen_model(
            args.model_path,
            revision=MODEL_REVISION,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            tf32=True,
            local_files_only=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        text_config = qwen_text_config(model)
        hidden_size = int(getattr(text_config, "hidden_size", 0))
        num_layers = int(getattr(text_config, "num_hidden_layers", 0))
        if hidden_size <= 0 or num_layers <= 0:
            raise RuntimeError("Loaded Qwen text config has invalid layer metadata")
        capture = StrictFinalPrelogitCapture(
            getattr(model, "model", model),
            hidden_size=hidden_size,
            num_layers=num_layers,
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        runtime.update(
            generate_pending(
                rows=pending,
                tokenizer=tokenizer,
                model=model,
                device=device,
                capture=capture,
                parts_dir=parts_dir,
                predictions_path=predictions_path,
                hidden_size=hidden_size,
                batch_size=int(args.batch_size),
                max_batch_tokens=int(args.max_batch_tokens),
                max_length=int(args.max_length),
                max_new_tokens=int(args.max_new_tokens),
                pad_to_multiple_of=int(args.pad_to_multiple_of),
            )
        )
        capture.close()
        if torch.cuda.is_available():
            runtime["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        runtime["resumed_without_generation"] = True

    prediction_rows = read_jsonl(predictions_path)
    by_id = {str(row["b_id"]): row for row in prediction_rows}
    if set(by_id) != set(selected_ids):
        raise RuntimeError("Direct Judge predictions do not cover exactly the selected rows")
    metadata = merge_feature_parts(
        parts_dir=parts_dir,
        output=features_path,
        rows=rows,
        settings={**runtime, "elapsed_generation_seconds": time.perf_counter() - started},
    )
    scored_rows = [
        {
            **row,
            "direct_score": by_id[str(row["b_id"])].get("predicted_score"),
            "direct_judge_raw_completion": by_id[str(row["b_id"])].get("raw_completion", ""),
            "judge_model_id": MODEL_ID,
            "judge_model_revision": MODEL_REVISION,
            "b_space_feature_index": index,
        }
        for index, row in enumerate(rows)
    ]
    write_jsonl(scored_rows_path, scored_rows)
    metrics_rows = direct_test_metrics(scored_rows)
    write_csv(args.output_dir / "direct_judge_metrics.csv", metrics_rows)
    summary = {
        "artifact_type": "flask_comparison_direct_judge_strict_final_prelogit_v1",
        "rows": len(rows),
        "source_rows": str(args.rows),
        "split_manifest": str(args.split_manifest),
        "scored_rows": str(scored_rows_path),
        "features": str(features_path),
        "metadata": metadata,
        "metrics": metrics_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(summary_path, summary)
    materialize_cell_feature_files(args.output_dir / "cells", scored_rows, features_path)
    print(json.dumps({"summary": str(summary_path), "rows": len(rows)}, ensure_ascii=False, indent=2))


def validate_runtime(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_path}. Pass --model-path to the local Qwen3.5-0.8B checkout."
        )
    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        print(
            "WARNING: --device requested CUDA but torch.cuda.is_available() is false; "
            "load_qwen_model will fall back to CPU. Full FLASK Direct Judge/feature extraction "
            "is expected to be very slow without a GPU.",
            file=sys.stderr,
            flush=True,
        )


def load_rows(rows_path: Path, split_manifest_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(rows_path)
    if not rows:
        raise ValueError(f"No rows in {rows_path}")
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    row_splits = {str(key): str(value) for key, value in manifest.get("row_splits", {}).items()}
    if set(row_splits) != {str(row["b_id"]) for row in rows}:
        raise ValueError("Split manifest row ids do not match --rows")
    return [{**row, "split": row_splits[str(row["b_id"])]} for row in rows]


def generate_pending(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    capture: "StrictFinalPrelogitCapture",
    parts_dir: Path,
    predictions_path: Path,
    hidden_size: int,
    batch_size: int,
    max_batch_tokens: int,
    max_length: int,
    max_new_tokens: int,
    pad_to_multiple_of: int,
) -> dict[str, Any]:
    prompts = [apply_chat_template(row, tokenizer) for row in rows]
    token_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    order = sorted(range(len(rows)), key=lambda index: (len(token_ids[index]), stable_rank(rows[index]["b_id"])))
    batches = make_batches(
        order,
        token_ids,
        max_documents=batch_size,
        max_tokens=max_batch_tokens,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    next_part = next_part_number(parts_dir)
    effective: list[int] = []
    progress = tqdm(total=len(rows), desc="FLASK comparison Direct Judge")
    try:
        while batches:
            indices = batches.pop(0)
            try:
                batch_rows = [rows[index] for index in indices]
                batch_ids = [head_tail(token_ids[index], max_length) for index in indices]
                encoded = tokenizer.pad(
                    [{"input_ids": ids} for ids in batch_ids],
                    padding=True,
                    pad_to_multiple_of=pad_to_multiple_of,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    capture.begin(encoded["attention_mask"])
                    generated = model.generate(
                        **encoded,
                        do_sample=False,
                        num_beams=1,
                        use_cache=True,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    features = capture.features()
                prompt_width = int(encoded["input_ids"].shape[1])
                completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
                write_feature_part(
                    parts_dir / f"part-{next_part:06d}.npz",
                    rows=batch_rows,
                    features=features,
                    hidden_size=hidden_size,
                    completions=completions,
                    attention_mask=encoded["attention_mask"],
                )
                next_part += 1
                append_predictions(predictions_path, batch_rows, completions, batch_ids, token_ids, indices)
                effective.append(len(indices))
                progress.update(len(indices))
                del generated, features, encoded
                capture.clear()
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                recoverable = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
                capture.clear()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if not recoverable or len(indices) == 1:
                    raise
                midpoint = len(indices) // 2
                batches.insert(0, indices[midpoint:])
                batches.insert(0, indices[:midpoint])
    finally:
        progress.close()
    return {"generated_rows": len(rows), "batch_count": len(effective), "effective_batch_sizes": sorted(set(effective))}


class StrictFinalPrelogitCapture:
    def __init__(self, base_model: torch.nn.Module, *, hidden_size: int, num_layers: int) -> None:
        layers = getattr(base_model, "layers", None)
        if layers is None or len(layers) != num_layers:
            raise ValueError("Unexpected Qwen decoder layer layout")
        norm = getattr(base_model, "norm", None)
        if norm is None:
            raise ValueError("Qwen final normalization layer is missing")
        self.hidden_size = int(hidden_size)
        self.attention_mask: torch.Tensor | None = None
        self.expected_shape: tuple[int, int] | None = None
        self.value: torch.Tensor | None = None
        self.handle = norm.register_forward_hook(self._hook)

    def begin(self, attention_mask: torch.Tensor) -> None:
        self.clear()
        self.attention_mask = attention_mask
        self.expected_shape = (int(attention_mask.shape[0]), int(attention_mask.shape[1]))

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if self.attention_mask is None or self.expected_shape is None or not isinstance(hidden, torch.Tensor):
            return
        batch, length = self.expected_shape
        if tuple(hidden.shape[:2]) != (batch, length):
            return
        if self.value is not None:
            raise RuntimeError("Captured strict final prelogit twice")
        positions = torch.arange(length, device=self.attention_mask.device).unsqueeze(0)
        last = (self.attention_mask.to(dtype=torch.long) * positions).max(dim=1).values.to(hidden.device)
        row_indices = torch.arange(batch, device=hidden.device)
        value = hidden[row_indices, last].float()
        if tuple(value.shape) != (batch, self.hidden_size):
            raise RuntimeError("Unexpected strict final prelogit shape")
        if not torch.isfinite(value).all():
            raise RuntimeError("Non-finite strict final prelogit")
        self.value = value

    def features(self) -> torch.Tensor:
        if self.value is None:
            raise RuntimeError("Strict final prelogit was not captured")
        return self.value.unsqueeze(1)

    def clear(self) -> None:
        self.attention_mask = None
        self.expected_shape = None
        self.value = None

    def close(self) -> None:
        self.handle.remove()
        self.clear()


def make_batches(
    order: list[int],
    token_ids: list[list[int]],
    *,
    max_documents: int,
    max_tokens: int,
    max_length: int,
    pad_to_multiple_of: int,
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for index in order:
        bounded = min(len(token_ids[index]), max_length)
        rounded = min(max_length, ((bounded + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of)
        candidate_width = max(width, rounded)
        if current and (len(current) + 1 > max_documents or candidate_width * (len(current) + 1) > max_tokens):
            batches.append(current)
            current = []
            width = 0
        current.append(index)
        width = max(width, rounded)
    if current:
        batches.append(current)
    return batches


def write_feature_part(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    features: torch.Tensor,
    hidden_size: int,
    completions: list[str],
    attention_mask: torch.Tensor,
) -> None:
    if tuple(features.shape) != (len(rows), 1, hidden_size):
        raise RuntimeError(f"Invalid feature shape: {tuple(features.shape)}")
    np.savez(
        path,
        features=features.detach().cpu().numpy().astype(np.float16),
        sample_ids=np.asarray([str(row["b_id"]) for row in rows]),
        labels=np.asarray([integer_score(row["ground_truth"]) for row in rows], dtype=np.int8),
        query_ids=np.asarray([cell_id(*row_cell(row)) for row in rows]),
        split=np.asarray([str(row["split"]) for row in rows]),
        domain_ids=np.asarray([row_cell(row)[0] for row in rows]),
        task_ids=np.asarray([row_cell(row)[1] for row in rows]),
        direct_scores=np.asarray([parse_direct_score(text) or -1 for text in completions], dtype=np.int8),
        direct_completions=np.asarray(completions),
        active_tokens=np.asarray(int(attention_mask.sum().item()), dtype=np.int64),
        padded_tokens=np.asarray(int(attention_mask.numel()), dtype=np.int64),
    )


def append_predictions(
    path: Path,
    rows: list[dict[str, Any]],
    completions: list[str],
    batch_ids: list[list[int]],
    token_ids: list[list[int]],
    indices: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row, completion, ids, index in zip(rows, completions, batch_ids, indices, strict=True):
            out = {
                "b_id": row["b_id"],
                "response_id": row.get("response_id"),
                "base_id": row.get("base_id"),
                "domain_id": row_cell(row)[0],
                "task_id": row_cell(row)[1],
                "split": row["split"],
                "generator_id": row.get("generator_id"),
                "ground_truth": integer_score(row.get("ground_truth")),
                "predicted_score": parse_direct_score(completion),
                "raw_completion": completion,
                "input_token_count": len(ids),
                "truncated": len(token_ids[index]) > len(ids),
            }
            handle.write(json.dumps(out, ensure_ascii=False, sort_keys=True) + "\n")


def feature_part_ids(parts_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["sample_ids"]).astype(str).tolist()
        if ids.intersection(values):
            raise ValueError("Duplicate ids across feature parts")
        ids.update(values)
    return ids


def next_part_number(parts_dir: Path) -> int:
    numbers: list[int] = []
    for path in parts_dir.glob("part-*.npz"):
        try:
            numbers.append(int(path.stem.split("-")[-1]))
        except ValueError:
            continue
    return max(numbers, default=-1) + 1


def merge_feature_parts(
    *,
    parts_dir: Path,
    output: Path,
    rows: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    fields = ("features", "sample_ids", "labels", "query_ids", "split", "domain_ids", "task_ids", "direct_scores", "direct_completions")
    pieces: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    active_tokens = 0
    padded_tokens = 0
    for part in sorted(parts_dir.glob("part-*.npz")):
        with np.load(part, allow_pickle=False) as payload:
            for field in fields:
                pieces[field].append(np.asarray(payload[field]))
            active_tokens += int(payload["active_tokens"])
            padded_tokens += int(payload["padded_tokens"])
    if not pieces["features"]:
        raise RuntimeError("No feature parts found")
    merged = {field: np.concatenate(values, axis=0) for field, values in pieces.items()}
    ids = merged["sample_ids"].astype(str)
    wanted = [str(row["b_id"]) for row in rows]
    if set(ids.tolist()) != set(wanted):
        raise RuntimeError("Feature parts do not cover selected rows")
    order_index = {value: index for index, value in enumerate(ids.tolist())}
    order = np.asarray([order_index[value] for value in wanted], dtype=np.int64)
    ordered = {field: values[order] for field, values in merged.items()}
    metadata = {
        **settings,
        "shape": list(ordered["features"].shape),
        "num_records": len(rows),
        "active_tokens": active_tokens,
        "padded_tokens": padded_tokens,
        "padding_fraction": 1.0 - active_tokens / max(padded_tokens, 1),
        "feature_storage_dtype": "float16",
        "direct_judge_parsed_rows": int(np.sum((ordered["direct_scores"] >= 1) & (ordered["direct_scores"] <= 5))),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **ordered, metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    return metadata


def materialize_cell_feature_files(output_dir: Path, rows: list[dict[str, Any]], features_path: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(features_path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"])
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
    index = {value: idx for idx, value in enumerate(sample_ids.tolist())}
    for cell in sorted({row_cell(row) for row in rows}):
        cell_rows = [row for row in rows if row_cell(row) == cell]
        ids = [str(row["b_id"]) for row in cell_rows]
        indices = np.asarray([index[value] for value in ids], dtype=np.int64)
        np.savez_compressed(
            output_dir / f"{slug(cell[0])}__{slug(cell[1])}.npz",
            features=features[indices],
            sample_ids=np.asarray(ids),
            labels=np.asarray([integer_score(row["ground_truth"]) for row in cell_rows], dtype=np.int8),
            query_ids=np.asarray([cell_id(*cell)] * len(cell_rows)),
            split=np.asarray([str(row["split"]) for row in cell_rows]),
            direct_scores=np.asarray([row.get("direct_score") or -1 for row in cell_rows], dtype=np.int8),
            metadata_json=np.asarray(json.dumps({"cell_id": cell_id(*cell), "rows": len(cell_rows)}, ensure_ascii=False)),
        )


def direct_test_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    cells = tuple(sorted({row_cell(row) for row in rows}, key=cell_sort_key))
    for target_cell in cells:
        target_rows = [
            row for row in rows
            if row_cell(row) == target_cell and str(row.get("split")) == "test"
        ]
        metric = metrics_from_predictions(
            [row["ground_truth"] for row in target_rows],
            [row.get("direct_score") for row in target_rows],
        )
        metrics.append(
            {
                "method": "direct_judge",
                "source_cell_id": "",
                "target_cell_id": cell_id(*target_cell),
                "split": "test",
                **metric,
            }
        )
    return metrics


if __name__ == "__main__":
    main()
