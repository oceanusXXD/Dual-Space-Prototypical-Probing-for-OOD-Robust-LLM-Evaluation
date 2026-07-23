#!/usr/bin/env python3
"""Run Direct Judge and save B-space features from the same Qwen generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json, write_jsonl
from src.models.extract_hidden import (
    QWEN3_5_4B_HIDDEN_SIZE,
    QWEN3_5_4B_MODEL_ID,
    QWEN3_5_4B_NUM_LAYERS,
    QWEN3_5_4B_REVISION,
    load_qwen_model,
    qwen_text_config,
)


DOMAINS = (
    "Humanities",
    "Language",
    "Social Science",
    "History",
    "Culture",
)
SKILLS = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
)
SEED = 42
DIRECT_JUDGE_TEMPLATE_VERSION = "flask_minimal_direct_judge_digit_v1"
DIRECT_JUDGE_TEMPLATE = (
    "You are an evaluator.\n\n"
    "Skill and rubric:\n{rubric}\n\n"
    "Instruction:\n{instruction}\n\n"
    "Reference answer:\n{reference_answer}\n\n"
    "Candidate response:\n{candidate_response}\n\n"
    "Return exactly one digit only: 1, 2, 3, 4, or 5.\n"
    "Do not output JSON, markdown, explanations, labels, spaces, or any other text."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qwen3.5-4B Direct Judge on the FLASK minimal 5x6 experiment."
    )
    parser.add_argument(
        "--b-space",
        type=Path,
        default=Path("datasets/processed/flask_domain_task_v1/b_space_single_domain.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/direct_judge"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("/home/zeus/models/qwen3.5-4b"))
    parser.add_argument(
        "--per-cell",
        type=int,
        default=10,
        help="Stratified rows per cell unless --all-rows is set.",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Score every valid integer-label B row in the 5x6 scope.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
        help="Maximum documents per dynamic GPU batch.",
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=196608,
        help=(
            "Maximum padded prompt tokens per dynamic GPU batch. This is below "
            "Qwen3.5's Conv1d 32-bit indexing limit; failing batches are split."
        ),
    )
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=[-10, -1],
        help="Transformer hidden-state indices to mean-pool from the Judge prompt.",
    )
    parser.add_argument(
        "--hidden-dtype",
        choices=["float16", "float32"],
        default="float16",
    )
    parser.add_argument(
        "--b-space-features",
        type=Path,
        default=None,
        help="Output NPZ for pooled B-space Judge features (default: output-dir/b_space_hidden_states.npz).",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_cell < 1 or args.batch_size < 1 or args.max_batch_tokens < 1:
        raise ValueError("--per-cell, --batch-size, and --max-batch-tokens must be positive")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "direct_judge_rows.jsonl"
    predictions_path = output_dir / "direct_judge_predictions.jsonl"
    b_space_scored_path = output_dir / "b_space_with_direct_judge.jsonl"
    features_path = args.b_space_features or output_dir / "b_space_hidden_states.npz"
    feature_metadata_path = features_path.with_suffix(".metadata.json")
    feature_parts_dir = features_path.with_name(f"{features_path.stem}.parts")
    summary_path = output_dir / "summary.json"

    selected = select_rows(
        load_jsonl(args.b_space),
        per_cell=args.per_cell,
        all_rows=bool(args.all_rows),
    )
    selected_layers = resolve_layers(QWEN3_5_4B_NUM_LAYERS + 1, list(args.layers))
    write_jsonl(sample_path, selected)

    if args.overwrite:
        for path in (
            predictions_path,
            b_space_scored_path,
            features_path,
            feature_metadata_path,
            summary_path,
        ):
            if path.exists():
                path.unlink()
        if feature_parts_dir.exists():
            shutil.rmtree(feature_parts_dir)
    feature_parts_dir.mkdir(parents=True, exist_ok=True)
    completed = {
        str(row["b_id"])
        for row in load_jsonl(predictions_path)
        if row.get("predicted_score") is not None
    }
    completed_feature_ids = _feature_part_ids(feature_parts_dir)
    if completed != completed_feature_ids:
        raise ValueError(
            "Direct Judge predictions and B-space feature parts differ; use --overwrite "
            "to restart this coupled single-inference run."
        )
    pending = [row for row in selected if str(row["b_id"]) not in completed]

    started = time.perf_counter()
    runtime: dict[str, Any] = {
        "model_id": QWEN3_5_4B_MODEL_ID,
        "revision": QWEN3_5_4B_REVISION,
        "model_path": str(args.model_path),
        "attn_implementation_requested": args.attn_implementation,
        "initial_batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "max_length": int(args.max_length),
        "max_new_tokens": int(args.max_new_tokens),
        "b_space_features": str(features_path),
        "b_space_hidden_layers_requested": list(args.layers),
        "b_space_hidden_layers_resolved": selected_layers,
        "b_space_hidden_dtype": args.hidden_dtype,
        "b_space_and_direct_judge_single_generation": True,
    }
    if pending:
        tokenizer, model, device = load_qwen_model(
            args.model_path,
            revision=QWEN3_5_4B_REVISION,
            device="cuda",
            torch_dtype="bfloat16",
            attn_implementation=args.attn_implementation,
            tf32=True,
            local_files_only=True,
        )
        if device.type != "cuda":
            raise RuntimeError("Direct Judge requires an attached CUDA GPU")
        tokenizer.padding_side = "left"
        text_config = qwen_text_config(model)
        model_selected_layers = resolve_layers(
            int(getattr(text_config, "num_hidden_layers", 0)) + 1,
            list(args.layers),
        )
        if model_selected_layers != selected_layers:
            raise RuntimeError("Loaded Qwen layer layout does not match the pinned protocol")
        if int(getattr(text_config, "hidden_size", 0)) != QWEN3_5_4B_HIDDEN_SIZE:
            raise RuntimeError("Unexpected Qwen3.5 hidden size")
        capture = GeneratePromptLayerCapture(getattr(model, "model", model), selected_layers)
        generated = generate_predictions(
            pending,
            tokenizer=tokenizer,
            model=model,
            device=device,
            predictions_path=predictions_path,
            feature_parts_dir=feature_parts_dir,
            capture=capture,
            initial_batch_size=int(args.batch_size),
            max_batch_tokens=int(args.max_batch_tokens),
            max_length=int(args.max_length),
            max_new_tokens=int(args.max_new_tokens),
            hidden_dtype=args.hidden_dtype,
        )
        capture.close()
        runtime.update(generated)
        runtime["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        del model
        torch.cuda.empty_cache()
    else:
        runtime["resumed_without_generation"] = True
    all_predictions = load_jsonl(predictions_path)
    predictions_by_id = {str(row["b_id"]): row for row in all_predictions}
    if {str(row["b_id"]) for row in selected} != set(predictions_by_id):
        missing = len(selected) - len(predictions_by_id)
        raise RuntimeError(f"Direct Judge is incomplete: {max(missing, 0)} B rows are missing")
    feature_metadata = merge_b_space_feature_parts(
        feature_parts_dir,
        features_path,
        selected=selected,
        settings={
            "artifact_type": "flask_5x6_b_space_hidden_states_with_direct_judge_v1",
            "feature_scope": "judge_input",
            "source_b_space": str(args.b_space),
            "prompt_template_version": DIRECT_JUDGE_TEMPLATE_VERSION,
            "prompt_template_sha256": hashlib.sha256(
                DIRECT_JUDGE_TEMPLATE.encode("utf-8")
            ).hexdigest(),
            "pooling": "masked_mean",
            "pooling_formula": "sum(hidden_state * attention_mask) / sum(attention_mask)",
            "pooling_mask_source": "tokenizer_attention_mask",
            "pooling_excludes_padding": True,
            "layers_requested": list(args.layers),
            "layers_resolved": selected_layers,
            "hidden_dtype": args.hidden_dtype,
            "model_id": QWEN3_5_4B_MODEL_ID,
            "model_revision": QWEN3_5_4B_REVISION,
            "model_path": str(args.model_path),
            "attn_implementation_requested": args.attn_implementation,
            "single_generation_with_direct_judge": True,
        },
    )
    write_json(feature_metadata_path, feature_metadata)
    write_scored_b_space(selected, predictions_by_id, b_space_scored_path)
    summary = summarize(
        selected=selected,
        predictions=all_predictions,
        selection={
            "source_b_space": str(args.b_space),
            "scope": "all_rows" if args.all_rows else f"{args.per_cell}_stratified_rows_per_cell",
            "ground_truth_filter": "integer scores in [1, 5]",
            "dropped_noninteger_ground_truth_rows": _noninteger_score_count(
                load_jsonl(args.b_space)
            ),
            "dropped_empty_candidate_response_rows": _empty_candidate_count(
                load_jsonl(args.b_space)
            ),
            "scored_b_space": str(b_space_scored_path),
            "b_space_features": str(features_path),
        },
        runtime={
            **runtime,
            "elapsed_seconds": time.perf_counter() - started,
            "b_space_feature_shape": feature_metadata["shape"],
        },
    )
    write_json(summary_path, summary)
    print(json.dumps(summary["global_metrics"], ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_rows(
    rows: Iterable[dict[str, Any]], *, per_cell: int, all_rows: bool
) -> list[dict[str, Any]]:
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {
        (domain, skill): [] for domain in DOMAINS for skill in SKILLS
    }
    selected_b_ids: set[str] = set()
    for row in rows:
        domains = list(row.get("domain_ids") or [])
        if len(domains) != 1:
            continue
        domain = str(domains[0])
        task = str(row.get("task_id") or "")
        if domain not in DOMAINS or task not in SKILLS:
            continue
        if _integer_score(row.get("ground_truth")) is None:
            continue
        if not str(row.get("candidate_response") or "").strip():
            # Current FLASK protocol keeps these in raw score-cleaning counts,
            # but excludes them from all model inputs.
            continue
        b_id = str(row["b_id"])
        if b_id in selected_b_ids:
            raise ValueError(f"Unexpected duplicate B-space row: {b_id}")
        selected_b_ids.add(b_id)
        cells[(domain, task)].append(row)

    selected: list[dict[str, Any]] = []
    for cell, candidates in cells.items():
        if not candidates:
            raise ValueError(f"Cell {cell} has no usable B rows")
        if not all_rows and len(candidates) < per_cell:
            raise ValueError(f"Cell {cell} has only {len(candidates)} rows, need {per_cell}")
        selected.extend(candidates if all_rows else stratified_sample(candidates, count=per_cell))
    selected.sort(key=lambda row: (DOMAINS.index(row["domain_ids"][0]), SKILLS.index(row["task_id"]), _rank(row["b_id"])))
    if not all_rows and len(selected) != len(cells) * per_cell:
        raise RuntimeError("Direct Judge sample has an unexpected row count")
    return selected


def stratified_sample(rows: list[dict[str, Any]], *, count: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["ground_truth"])].append(row)
    chosen: list[dict[str, Any]] = []
    for label in sorted(groups, key=_numeric_label_key):
        chosen.append(sorted(groups[label], key=lambda row: _rank(row["b_id"]))[0])
        if len(chosen) == count:
            return chosen
    chosen_ids = {str(row["b_id"]) for row in chosen}
    rest = sorted(
        (row for row in rows if str(row["b_id"]) not in chosen_ids),
        key=lambda row: _rank(row["b_id"]),
    )
    chosen.extend(rest[: count - len(chosen)])
    return chosen


def generate_predictions(
    rows: list[dict[str, Any]],
    *,
    tokenizer: Any,
    model: Any,
    device: torch.device,
    predictions_path: Path,
    feature_parts_dir: Path,
    capture: "GeneratePromptLayerCapture",
    initial_batch_size: int,
    max_batch_tokens: int,
    max_length: int,
    max_new_tokens: int,
    hidden_dtype: str,
) -> dict[str, Any]:
    prompts = [_chat_prompt(row, tokenizer) for row in rows]
    token_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    order = sorted(range(len(rows)), key=lambda index: (len(token_ids[index]), _rank(rows[index]["b_id"])))
    batches = _length_bucketed_batches(
        order,
        token_ids,
        maximum_documents=int(initial_batch_size),
        maximum_tokens=int(max_batch_tokens),
        max_length=int(max_length),
    )
    output_rows: list[dict[str, Any]] = []
    batch_sizes: list[int] = []
    next_part_number = _next_feature_part_number(feature_parts_dir)
    progress = tqdm(total=len(rows), desc="FLASK Direct Judge")
    try:
        while batches:
            indices = batches.pop(0)
            try:
                batch_rows = [rows[index] for index in indices]
                batch_ids = [_head_tail(ids=token_ids[index], max_length=max_length) for index in indices]
                encoded = tokenizer.pad(
                    [{"input_ids": ids} for ids in batch_ids],
                    padding=True,
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
                        max_new_tokens=int(max_new_tokens),
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                    features = capture.features()
                part = feature_parts_dir / f"part-{next_part_number:06d}.npz"
                write_b_space_feature_part(
                    part,
                    features=features,
                    rows=batch_rows,
                    indices=indices,
                    attention_mask=encoded["attention_mask"],
                    hidden_dtype=hidden_dtype,
                    max_length=max_length,
                    original_token_ids=[token_ids[index] for index in indices],
                    truncated_token_ids=batch_ids,
                )
                next_part_number += 1
                input_width = encoded["input_ids"].shape[1]
                completions = tokenizer.batch_decode(
                    generated[:, input_width:], skip_special_tokens=True
                )
                for row, completion, original_ids, ids in zip(
                    batch_rows,
                    completions,
                    (token_ids[index] for index in indices),
                    batch_ids,
                    strict=True,
                ):
                    predicted = parse_score(completion)
                    output_rows.append(
                        {
                            "b_id": row["b_id"],
                            "response_id": row["response_id"],
                            "base_id": row["base_id"],
                            "domain_id": row["domain_ids"][0],
                            "task_id": row["task_id"],
                            "generator_id": row["generator_id"],
                            "ground_truth": row["ground_truth"],
                            "predicted_score": predicted,
                            "raw_completion": completion,
                            "input_token_count": len(ids),
                            "truncated": len(original_ids) > len(ids),
                        }
                    )
                write_jsonl_append(predictions_path, output_rows)
                output_rows.clear()
                batch_sizes.append(len(indices))
                progress.update(len(indices))
                del generated, features, encoded
                capture.clear()
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                recoverable = isinstance(exc, torch.OutOfMemoryError) or (
                    "canUse32BitIndexMath" in str(exc)
                    or "out of memory" in str(exc).lower()
                )
                if not recoverable:
                    raise
                torch.cuda.empty_cache()
                capture.clear()
                if len(indices) == 1:
                    raise
                midpoint = len(indices) // 2
                batches.insert(0, indices[midpoint:])
                batches.insert(0, indices[:midpoint])
                print(
                    f"CUDA OOM: split one Direct Judge batch from {len(indices)} "
                    f"to {midpoint}+{len(indices) - midpoint}",
                    flush=True,
                )
    finally:
        progress.close()
    return {
        "effective_batch_sizes": sorted(set(batch_sizes)),
        "batch_count": len(batch_sizes),
        "generated_rows": len(rows),
    }


def _chat_prompt(row: dict[str, Any], tokenizer: Any) -> str:
    prompt = DIRECT_JUDGE_TEMPLATE.format(
        rubric=str(row.get("rubric") or ""),
        instruction=str(row.get("instruction") or ""),
        reference_answer=str(row.get("reference_answer") or "(not provided)"),
        candidate_response=str(row.get("candidate_response") or ""),
    )
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _head_tail(*, ids: list[int], max_length: int) -> list[int]:
    if len(ids) <= max_length:
        return list(ids)
    head = (max_length + 1) // 2
    return [*ids[:head], *ids[-(max_length - head) :]]


def parse_score(text: str) -> int | None:
    stripped = text.strip()
    return int(stripped) if re.fullmatch(r"[1-5]", stripped) else None


def write_jsonl_append(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(
    *,
    selected: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    selection: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    by_id = {str(row["b_id"]): row for row in predictions}
    rows = [by_id[str(row["b_id"])] for row in selected if str(row["b_id"]) in by_id]
    summary = {
        "artifact_type": "flask_minimal_5x6_direct_judge_v1",
        "domains": list(DOMAINS),
        "skills": list(SKILLS),
        "selected_rows": len(selected),
        "prediction_rows": len(rows),
        "selection": selection,
        "runtime": runtime,
        "global_metrics": metrics(rows),
        "cell_metrics": {},
    }
    for domain in DOMAINS:
        for skill in SKILLS:
            cell = [
                row
                for row in rows
                if row["domain_id"] == domain and row["task_id"] == skill
            ]
            summary["cell_metrics"][f"{domain}::{skill}"] = metrics(cell)
    return summary


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = [
        row
        for row in rows
        if row.get("predicted_score") is not None and _integer_score(row["ground_truth"]) is not None
    ]
    if not parsed:
        return {
            "rows": len(rows),
            "parsed_rows": 0,
            "parse_rate": 0.0,
            "mae": None,
            "exact_accuracy": None,
            "plus_minus_1_accuracy": None,
            "quadratic_weighted_kappa": None,
        }
    truth = np.asarray([_integer_score(row["ground_truth"]) for row in parsed], dtype=int)
    pred = np.asarray([int(row["predicted_score"]) for row in parsed], dtype=int)
    return {
        "rows": len(rows),
        "parsed_rows": len(parsed),
        "parse_rate": len(parsed) / max(len(rows), 1),
        "mae": float(np.mean(np.abs(pred - truth))),
        "exact_accuracy": float(np.mean(pred == truth)),
        "plus_minus_1_accuracy": float(np.mean(np.abs(pred - truth) <= 1)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(truth, pred, labels=[1, 2, 3, 4, 5], weights="quadratic")
        ),
    }


def _integer_score(value: Any) -> int | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    rounded = int(round(score))
    return rounded if abs(score - rounded) < 1e-8 and 1 <= rounded <= 5 else None


def _rank(value: Any) -> str:
    return hashlib.sha256(f"{SEED}::{value}".encode("utf-8")).hexdigest()


def _numeric_label_key(value: str) -> tuple[float, str]:
    try:
        return float(value), value
    except ValueError:
        return float("inf"), value


def _noninteger_score_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if len(list(row.get("domain_ids") or [])) == 1
        and row["domain_ids"][0] in DOMAINS
        and row.get("task_id") in SKILLS
        and _integer_score(row.get("ground_truth")) is None
    )


def _empty_candidate_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for row in rows
        if len(list(row.get("domain_ids") or [])) == 1
        and row["domain_ids"][0] in DOMAINS
        and row.get("task_id") in SKILLS
        and _integer_score(row.get("ground_truth")) is not None
        and not str(row.get("candidate_response") or "").strip()
    )


def resolve_layers(num_states: int, requested: list[int]) -> list[int]:
    """Resolve protocol layer indices; state zero is the embedding state."""

    resolved: list[int] = []
    for value in requested:
        index = int(value) if int(value) >= 0 else int(num_states) + int(value)
        if index <= 0 or index >= int(num_states):
            raise ValueError(
                f"Layer {value} resolves to {index}; valid transformer outputs are "
                f"1..{int(num_states) - 1}"
            )
        if index in resolved:
            raise ValueError(f"Duplicate resolved layer: {index}")
        resolved.append(index)
    return resolved


class GeneratePromptLayerCapture:
    """Mean-pool selected prompt states while ``generate`` performs its prefill.

    Qwen generation first runs the full prompt (prefill), then one-token decode
    steps. The hook only accepts tensors matching the prefill shape, so it
    retains the B-space prompt representation rather than a generated token.
    Pooling in the hook avoids materialising all 33 model hidden-state tensors.
    """

    def __init__(self, base_model: torch.nn.Module, state_indices: list[int]) -> None:
        layers = getattr(base_model, "layers", None)
        if layers is None or len(layers) != QWEN3_5_4B_NUM_LAYERS:
            raise ValueError("Qwen3.5 decoder layers were not found for B-space capture")
        self.state_indices = list(state_indices)
        self._attention_mask: torch.Tensor | None = None
        self._expected_shape: tuple[int, int] | None = None
        self._features: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        for state_index in self.state_indices:
            if int(state_index) == QWEN3_5_4B_NUM_LAYERS:
                module = getattr(base_model, "norm", None)
                if module is None:
                    raise ValueError("Qwen3.5 final norm was not found for B-space capture")
            else:
                module = layers[int(state_index) - 1]
            self._handles.append(
                module.register_forward_hook(self._hook_for_state(int(state_index)))
            )

    def begin(self, attention_mask: torch.Tensor) -> None:
        if attention_mask.ndim != 2:
            raise ValueError("Expected a rank-2 tokenizer attention mask")
        self.clear()
        self._attention_mask = attention_mask
        self._expected_shape = (int(attention_mask.shape[0]), int(attention_mask.shape[1]))

    def _hook_for_state(self, state_index: int):
        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    f"Decoder output for B-space state {state_index} is not a tensor"
                )
            if self._expected_shape is None or self._attention_mask is None:
                return
            expected_batch, expected_length = self._expected_shape
            if tuple(output.shape[:2]) != (expected_batch, expected_length):
                # ``generate`` calls the decoder again for one-token decode steps.
                return
            if state_index in self._features:
                raise RuntimeError(
                    f"B-space prefill state {state_index} was captured more than once"
                )
            weights = self._attention_mask.to(dtype=torch.float32).unsqueeze(-1)
            pooled = (output.float() * weights).sum(dim=1)
            pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
            if tuple(pooled.shape) != (expected_batch, QWEN3_5_4B_HIDDEN_SIZE):
                raise RuntimeError(
                    f"B-space pooled state has shape {tuple(pooled.shape)}, expected "
                    f"({expected_batch}, {QWEN3_5_4B_HIDDEN_SIZE})"
                )
            if not torch.isfinite(pooled).all():
                raise RuntimeError(f"B-space state {state_index} contains a non-finite value")
            self._features[state_index] = pooled

        return capture

    def features(self) -> torch.Tensor:
        missing = [index for index in self.state_indices if index not in self._features]
        if missing:
            raise RuntimeError(
                "Qwen generation did not expose the B-space prompt hidden state(s): "
                f"{missing}"
            )
        return torch.stack([self._features[index] for index in self.state_indices], dim=1)

    def clear(self) -> None:
        self._features.clear()
        self._attention_mask = None
        self._expected_shape = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()


def _feature_part_ids(parts_dir: Path) -> set[str]:
    identifiers: set[str] = set()
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            if "sample_ids" not in payload.files:
                raise ValueError(f"B-space feature part is missing sample_ids: {path}")
            ids = np.asarray(payload["sample_ids"]).astype(str).tolist()
        duplicate = identifiers.intersection(ids)
        if duplicate:
            raise ValueError(f"Duplicate B-space feature ids in parts: {sorted(duplicate)[:3]}")
        identifiers.update(ids)
    return identifiers


def _next_feature_part_number(parts_dir: Path) -> int:
    numbers: list[int] = []
    for path in parts_dir.glob("part-*.npz"):
        match = re.fullmatch(r"part-(\d+)\.npz", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=-1) + 1


def write_b_space_feature_part(
    path: Path,
    *,
    features: torch.Tensor,
    rows: list[dict[str, Any]],
    indices: list[int],
    attention_mask: torch.Tensor,
    hidden_dtype: str,
    max_length: int,
    original_token_ids: list[list[int]],
    truncated_token_ids: list[list[int]],
) -> None:
    """Persist one completed Direct Judge batch before its prediction records."""

    if path.exists():
        raise FileExistsError(f"B-space feature part already exists: {path}")
    expected_shape = (len(rows), features.shape[1], QWEN3_5_4B_HIDDEN_SIZE)
    if tuple(features.shape) != expected_shape:
        raise RuntimeError(
            f"B-space feature shape {tuple(features.shape)} does not match {expected_shape}"
        )
    dtype = np.float16 if hidden_dtype == "float16" else np.float32
    labels = [_integer_score(row.get("ground_truth")) for row in rows]
    if any(label is None for label in labels):
        raise ValueError("B-space feature batch includes a non-integer label")
    np.savez(
        path,
        features=features.cpu().numpy().astype(dtype),
        indices=np.asarray(indices, dtype=np.int64),
        sample_ids=np.asarray([str(row["b_id"]) for row in rows]),
        labels=np.asarray(labels, dtype=np.int8),
        query_ids=np.asarray([str(row["base_id"]) for row in rows]),
        audit_document_group_ids=np.asarray([str(row["base_id"]) for row in rows]),
        input_document_ids=np.asarray([str(row["response_id"]) for row in rows]),
        domain_ids=np.asarray([str(row["domain_ids"][0]) for row in rows]),
        task_ids=np.asarray([str(row["task_id"]) for row in rows]),
        active_tokens=np.asarray(int(attention_mask.sum().item()), dtype=np.int64),
        padded_tokens=np.asarray(int(attention_mask.numel()), dtype=np.int64),
        pooled_tokens=np.asarray(int(attention_mask.sum().item()), dtype=np.int64),
        truncated_records=np.asarray(
            sum(len(original) > len(truncated) for original, truncated in zip(
                original_token_ids, truncated_token_ids, strict=True
            )),
            dtype=np.int64,
        ),
        max_unpadded_tokens=np.asarray(
            max((len(ids) for ids in truncated_token_ids), default=0), dtype=np.int64
        ),
        max_length=np.asarray(int(max_length), dtype=np.int64),
    )


def merge_b_space_feature_parts(
    parts_dir: Path,
    output: Path,
    *,
    selected: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Merge batch parts into the standard feature-cache shape, ordered by B row."""

    fields = (
        "features",
        "indices",
        "sample_ids",
        "labels",
        "query_ids",
        "audit_document_group_ids",
        "input_document_ids",
        "domain_ids",
        "task_ids",
    )
    pieces: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    active_tokens = padded_tokens = pooled_tokens = truncated_records = 0
    max_unpadded_tokens = 0
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            missing = [field for field in fields if field not in payload.files]
            if missing:
                raise ValueError(f"B-space feature part {path} is missing {missing}")
            for field in fields:
                pieces[field].append(np.asarray(payload[field]))
            active_tokens += int(payload["active_tokens"])
            padded_tokens += int(payload["padded_tokens"])
            pooled_tokens += int(payload["pooled_tokens"])
            truncated_records += int(payload["truncated_records"])
            max_unpadded_tokens = max(max_unpadded_tokens, int(payload["max_unpadded_tokens"]))
    if not pieces["features"]:
        raise RuntimeError("No B-space feature parts were written")
    merged = {field: np.concatenate(values, axis=0) for field, values in pieces.items()}
    sample_ids = merged["sample_ids"].astype(str)
    if len(set(sample_ids.tolist())) != len(sample_ids):
        raise RuntimeError("B-space feature parts contain duplicate sample ids")
    selected_ids = [str(row["b_id"]) for row in selected]
    if set(sample_ids.tolist()) != set(selected_ids):
        missing = len(set(selected_ids).difference(sample_ids.tolist()))
        extra = len(set(sample_ids.tolist()).difference(selected_ids))
        raise RuntimeError(f"B-space feature coverage mismatch: missing={missing}, extra={extra}")
    source_index = {value: index for index, value in enumerate(sample_ids.tolist())}
    order = np.asarray([source_index[value] for value in selected_ids], dtype=np.int64)
    ordered = {field: value[order] for field, value in merged.items() if field != "indices"}
    metadata = {
        **settings,
        "shape": list(ordered["features"].shape),
        "num_records": len(selected_ids),
        "active_tokens": active_tokens,
        "padded_tokens": padded_tokens,
        "pooled_tokens": pooled_tokens,
        "truncated_records": truncated_records,
        "max_unpadded_tokens": max_unpadded_tokens,
        "padding_fraction": 1.0 - active_tokens / max(padded_tokens, 1),
        "feature_storage_dtype": np.dtype(ordered["features"].dtype).name,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **ordered,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def write_scored_b_space(
    selected: list[dict[str, Any]],
    predictions_by_id: dict[str, dict[str, Any]],
    path: Path,
) -> None:
    """Write the selected B rows with their Direct Judge answer on the same row."""

    rows: list[dict[str, Any]] = []
    for feature_index, row in enumerate(selected):
        prediction = predictions_by_id[str(row["b_id"])]
        rows.append(
            {
                **row,
                "direct_score": prediction.get("predicted_score"),
                "direct_judge_raw_completion": prediction.get("raw_completion", ""),
                "direct_judge_input_token_count": prediction.get("input_token_count"),
                "direct_judge_truncated": prediction.get("truncated"),
                "b_space_feature_index": feature_index,
                "judge_prompt_template_version": DIRECT_JUDGE_TEMPLATE_VERSION,
            }
        )
    write_jsonl(path, rows)


def _length_bucketed_batches(
    order: list[int],
    token_ids: list[list[int]],
    *,
    maximum_documents: int,
    maximum_tokens: int,
    max_length: int,
) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    current_max = 0
    for index in order:
        length = min(len(token_ids[index]), int(max_length))
        candidate_max = max(current_max, length)
        if current and (
            len(current) + 1 > int(maximum_documents)
            or candidate_max * (len(current) + 1) > int(maximum_tokens)
        ):
            batches.append(current)
            current = []
            current_max = 0
        current.append(index)
        current_max = max(current_max, length)
    if current:
        batches.append(current)
    if sum(len(batch) for batch in batches) != len(order):
        raise RuntimeError("Direct Judge batch plan is incomplete")
    return batches


if __name__ == "__main__":
    main()
