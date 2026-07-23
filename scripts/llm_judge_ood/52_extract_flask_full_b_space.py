#!/usr/bin/env python3
"""Extract full 10x12 FLASK B-space caches without re-running completed prompts.

The 5x6 Direct Judge cache already contains Qwen prefill features for a subset
of B prompts.  This runner reuses those vectors, forwards only previously
unseen B ids, then materializes one feature cache for every full-view
``Domain x Skill`` membership cell.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.models.extract_hidden import (
    QWEN3_5_4B_HIDDEN_SIZE,
    QWEN3_5_4B_MODEL_ID,
    QWEN3_5_4B_NUM_LAYERS,
    QWEN3_5_4B_REVISION,
    load_qwen_model,
    qwen_text_config,
)


DOMAINS = (
    "Humanities", "Language", "Social Science", "History", "Culture",
    "Technology", "Coding", "Math", "Natural Science", "Health",
)
SKILLS = (
    "Comprehension", "Factuality", "Logical Correctness",
    "Commonsense Understanding", "Completeness", "Insightfulness",
    "Metacognition", "Readability", "Conciseness", "Harmlessness",
    "Logical Robustness", "Logical Efficiency",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b-space", type=Path,
        default=Path("datasets/processed/flask_domain_task_v1/b_space_full.jsonl"),
    )
    parser.add_argument(
        "--cells-dir", type=Path,
        default=Path("datasets/processed/flask_domain_task_v1/cells_full"),
    )
    parser.add_argument(
        "--reuse-features", type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_hidden_states.npz"
        ),
    )
    parser.add_argument(
        "--reuse-direct-judge", type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_with_direct_judge.jsonl"
        ),
        help="Completed 5x6 B rows carrying reusable direct_score values.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/flask_full_b_space"),
    )
    parser.add_argument("--model-path", type=Path, default=Path("/home/zeus/models/qwen3.5-4b"))
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-batch-tokens", type=int, default=262144)
    parser.add_argument("--tokenization-chunk-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--pad-to-multiple-of", type=int, default=128)
    parser.add_argument("--layers", nargs="+", type=int, default=[-10, -1])
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.max_batch_tokens, args.tokenization_chunk_size, args.max_length, args.pad_to_multiple_of) < 1:
        raise ValueError("batch, token, length, and padding settings must be positive")
    if args.max_length % args.pad_to_multiple_of:
        raise ValueError("--max-length must be divisible by --pad-to-multiple-of")
    output_dir: Path = args.output_dir
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "unique_parts"
    parts_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    settings = {
        "artifact_type": "flask_full_10x12_b_space_hidden_v1",
        "source_b_space": str(args.b_space),
        "source_cells_dir": str(args.cells_dir),
        "reuse_features": str(args.reuse_features),
        "reuse_direct_judge": str(args.reuse_direct_judge),
        "model_id": QWEN3_5_4B_MODEL_ID,
        "model_revision": QWEN3_5_4B_REVISION,
        "model_path": str(args.model_path),
        "prompt_template_version": "flask_minimal_direct_judge_digit_v1",
        "layers_requested": list(args.layers),
        "pooling": "masked_mean",
        "feature_dtype": "float16",
        "integer_score_only": True,
        "nonempty_candidate_response_only": True,
        "batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "max_length": int(args.max_length),
        "max_new_tokens": int(args.max_new_tokens),
        "pad_to_multiple_of": int(args.pad_to_multiple_of),
    }
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key, value in settings.items():
            if existing_manifest.get(key) != value:
                raise ValueError(f"Existing full B-space setting {key!r} differs; use --overwrite")
    else:
        write_json(manifest_path, {**settings, "complete": False})

    direct = _direct_judge_module()
    existing = load_existing_features(args.reuse_features, args.reuse_direct_judge)
    part_ids = load_part_ids(parts_dir)
    if set(existing["id_to_index"]).intersection(part_ids):
        raise ValueError("Reused 5x6 and new feature parts overlap")
    known_ids = set(existing["id_to_index"]).union(part_ids)
    eligible_ids, extraction_stats = scan_b_space(
        args.b_space, known_ids=known_ids, extract=False
    )
    if not eligible_ids:
        raise ValueError("No eligible full FLASK B-space rows were found")
    pending_total = len(set(eligible_ids).difference(known_ids))
    started = time.perf_counter()
    runtime: dict[str, Any] = {
        "eligible_unique_b_rows": len(eligible_ids),
        "reused_5x6_unique_b_rows": len(existing["id_to_index"]),
        "resumed_new_unique_b_rows": len(part_ids),
        "new_unique_b_rows_pending": pending_total,
        "cleaning": extraction_stats,
    }
    if pending_total:
        tokenizer, model, device = load_qwen_model(
            args.model_path, revision=QWEN3_5_4B_REVISION, device="cuda",
            torch_dtype="bfloat16", attn_implementation=args.attn_implementation,
            tf32=True, local_files_only=True,
        )
        if device.type != "cuda":
            raise RuntimeError("Full B-space extraction requires CUDA")
        tokenizer.padding_side = "left"
        text_config = qwen_text_config(model)
        selected_layers = direct.resolve_layers(
            int(getattr(text_config, "num_hidden_layers", 0)) + 1, list(args.layers)
        )
        if int(getattr(text_config, "hidden_size", 0)) != QWEN3_5_4B_HIDDEN_SIZE:
            raise RuntimeError("Unexpected Qwen hidden size")
        capture = direct.GeneratePromptLayerCapture(getattr(model, "model", model), selected_layers)
        torch.cuda.reset_peak_memory_stats()
        generated = extract_pending_rows(
            b_space=args.b_space, known_ids=known_ids, parts_dir=parts_dir,
            tokenizer=tokenizer, model=model,
            capture=capture, direct=direct, device=device, args=args,
        )
        capture.close()
        runtime.update(generated)
        runtime["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        del model
        torch.cuda.empty_cache()
    else:
        runtime["resumed_without_qwen_forward"] = True
    runtime["elapsed_seconds"] = time.perf_counter() - started

    merged = merge_unique_features(
        existing=existing, parts_dir=parts_dir, eligible_ids=eligible_ids,
        output=output_dir / "unique_b_space_hidden_states.npz", settings=settings,
    )
    direct_prediction_summary = write_unique_direct_predictions(
        output=output_dir / "unique_direct_judge_predictions.jsonl",
        b_space=args.b_space, merged=merged,
    )
    cell_summary = materialize_cells(
        cells_dir=args.cells_dir, output_dir=output_dir / "cells", merged=merged,
        expected_ids=set(eligible_ids), settings=settings,
    )
    summary = {**settings, **runtime, **merged["metadata"], **direct_prediction_summary, **cell_summary}
    write_json(output_dir / "summary.json", summary)
    write_json(manifest_path, {**settings, "complete": True, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def scan_b_space(
    path: Path, *, known_ids: set[str], extract: bool
) -> tuple[list[str], dict[str, int]]:
    identifiers: list[str] = []
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    for row in iter_jsonl(path):
        stats["raw_unique_b_rows"] += 1
        if integer_score(row.get("ground_truth")) is None:
            stats["dropped_noninteger_score"] += 1
            continue
        if not str(row.get("candidate_response") or "").strip():
            stats["dropped_empty_candidate_response"] += 1
            continue
        b_id = str(row["b_id"])
        if b_id in seen:
            raise ValueError(f"Duplicate B id in full source: {b_id}")
        seen.add(b_id)
        identifiers.append(b_id)
        if b_id in known_ids:
            stats["already_cached"] += 1
        else:
            stats["needs_qwen"] += 1
    return identifiers, dict(stats)


def extract_pending_rows(
    *, b_space: Path, known_ids: set[str], parts_dir: Path, tokenizer: Any,
    model: Any, capture: Any, direct: Any, device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    next_part = next_part_number(parts_dir)
    written = 0
    effective_batches: list[int] = []
    chunk: list[dict[str, Any]] = []
    progress = tqdm(desc="Full FLASK B-space", unit="prompt")
    try:
        for row in iter_jsonl(b_space):
            if integer_score(row.get("ground_truth")) is None or not str(row.get("candidate_response") or "").strip():
                continue
            if str(row["b_id"]) in known_ids:
                continue
            chunk.append(row)
            if len(chunk) >= int(args.tokenization_chunk_size):
                next_part, count, batches = extract_chunk(
                    chunk, next_part=next_part, tokenizer=tokenizer, model=model,
                    capture=capture, direct=direct, device=device, parts_dir=parts_dir, args=args,
                )
                written += count; effective_batches.extend(batches); progress.update(count); chunk = []
        if chunk:
            next_part, count, batches = extract_chunk(
                chunk, next_part=next_part, tokenizer=tokenizer, model=model,
                capture=capture, direct=direct, device=device, parts_dir=parts_dir, args=args,
            )
            written += count; effective_batches.extend(batches); progress.update(count)
    finally:
        progress.close()
    return {
        "new_qwen_unique_b_rows": written,
        "batch_count": len(effective_batches),
        "effective_batch_sizes": sorted(set(effective_batches)),
    }


def extract_chunk(
    rows: list[dict[str, Any]], *, next_part: int, tokenizer: Any, model: Any,
    capture: Any, direct: Any, device: torch.device, parts_dir: Path, args: argparse.Namespace,
) -> tuple[int, int, list[int]]:
    prompts = [direct._chat_prompt(row, tokenizer) for row in rows]
    token_ids = tokenizer(
        prompts, add_special_tokens=False, padding=False, truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    order = sorted(range(len(rows)), key=lambda index: (len(token_ids[index]), str(rows[index]["b_id"])))
    batches = length_batches(order, token_ids, args=args)
    successful: list[int] = []
    count = 0
    while batches:
        indices = batches.pop(0)
        try:
            batch_ids = [direct._head_tail(ids=token_ids[index], max_length=int(args.max_length)) for index in indices]
            encoded = tokenizer.pad(
                [{"input_ids": ids} for ids in batch_ids], padding=True,
                pad_to_multiple_of=int(args.pad_to_multiple_of),
                return_attention_mask=True, return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                capture.begin(encoded["attention_mask"])
                generated = model.generate(
                    **encoded, do_sample=False, num_beams=1, use_cache=True,
                    max_new_tokens=int(args.max_new_tokens),
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
                features = capture.features()
            input_width = int(encoded["input_ids"].shape[1])
            completions = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
            direct_scores = [direct.parse_score(completion) for completion in completions]
            part = parts_dir / f"part-{next_part:06d}.npz"
            np.savez(
                part,
                features=features.cpu().numpy().astype(np.float16),
                sample_ids=np.asarray([str(rows[index]["b_id"]) for index in indices]),
                labels=np.asarray([integer_score(rows[index]["ground_truth"]) for index in indices], dtype=np.int8),
                query_ids=np.asarray([str(rows[index]["base_id"]) for index in indices]),
                direct_scores=np.asarray([score if score is not None else -1 for score in direct_scores], dtype=np.int8),
                direct_completions=np.asarray(completions),
                active_tokens=np.asarray(int(encoded["attention_mask"].sum().item()), dtype=np.int64),
                padded_tokens=np.asarray(int(encoded["attention_mask"].numel()), dtype=np.int64),
                truncated_records=np.asarray(sum(len(token_ids[index]) > len(ids) for index, ids in zip(indices, batch_ids, strict=True)), dtype=np.int64),
            )
            next_part += 1; count += len(indices); successful.append(len(indices))
            del generated, features, encoded
            capture.clear()
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            recoverable = isinstance(exc, torch.OutOfMemoryError) or any(
                token in str(exc).lower()
                for token in ("out of memory", "canuse32bitindexmath", "triton error [cuda]: invalid argument")
            )
            capture.clear(); torch.cuda.empty_cache()
            if not recoverable or len(indices) == 1:
                raise
            midpoint = len(indices) // 2
            batches.insert(0, indices[midpoint:]); batches.insert(0, indices[:midpoint])
            print(f"B-space batch split {len(indices)} -> {midpoint}+{len(indices)-midpoint}", flush=True)
    return next_part, count, successful


def length_batches(order: list[int], token_ids: list[list[int]], *, args: argparse.Namespace) -> list[list[int]]:
    batches: list[list[int]] = []; current: list[int] = []; width = 0
    for index in order:
        bounded = min(len(token_ids[index]), int(args.max_length))
        rounded = min(int(args.max_length), ((bounded + int(args.pad_to_multiple_of) - 1) // int(args.pad_to_multiple_of)) * int(args.pad_to_multiple_of))
        candidate_width = max(width, rounded)
        if current and (len(current) + 1 > int(args.batch_size) or candidate_width * (len(current) + 1) > int(args.max_batch_tokens)):
            batches.append(current); current = []; width = 0
        current.append(index); width = max(width, rounded)
    if current:
        batches.append(current)
    return batches


def load_existing_features(path: Path, scored_b_space: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing completed 5x6 B-space cache: {path}")
    with np.load(path, allow_pickle=False) as payload:
        ids = np.asarray(payload["sample_ids"]).astype(str)
        features = np.asarray(payload["features"], dtype=np.float16)
        labels = np.asarray(payload["labels"], dtype=np.int8)
        query_ids = np.asarray(payload["query_ids"]).astype(str)
        metadata = json.loads(str(payload["metadata_json"].item()))
    if features.shape != (len(ids), 2, QWEN3_5_4B_HIDDEN_SIZE) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Completed 5x6 B-space cache has an invalid shape or duplicate ids")
    if metadata.get("prompt_template_version") != "flask_minimal_direct_judge_digit_v1":
        raise ValueError("5x6 B-space cache was not built with the reusable Direct Judge prompt")
    direct_by_id = {
        str(row["b_id"]): row
        for row in iter_jsonl(scored_b_space)
    }
    if set(direct_by_id) != set(ids.tolist()):
        raise ValueError("Reusable 5x6 Direct Judge rows are not aligned with its feature cache")
    direct_scores = np.asarray([
        int(direct_by_id[b_id].get("direct_score"))
        if direct_by_id[b_id].get("direct_score") is not None else -1
        for b_id in ids.tolist()
    ], dtype=np.int8)
    direct_completions = np.asarray([
        str(direct_by_id[b_id].get("direct_judge_raw_completion") or "")
        for b_id in ids.tolist()
    ])
    return {
        "sample_ids": ids, "features": features, "labels": labels,
        "query_ids": query_ids, "direct_scores": direct_scores,
        "direct_completions": direct_completions, "metadata": metadata,
        "id_to_index": {value: index for index, value in enumerate(ids.tolist())},
    }


def load_part_ids(parts_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            part_ids = np.asarray(payload["sample_ids"]).astype(str).tolist()
        overlap = ids.intersection(part_ids)
        if overlap:
            raise ValueError(f"Duplicate B id in new parts: {next(iter(overlap))}")
        ids.update(part_ids)
    return ids


def next_part_number(parts_dir: Path) -> int:
    numbers = [int(match.group(1)) for path in parts_dir.glob("part-*.npz") if (match := re.fullmatch(r"part-(\d+)\.npz", path.name))]
    return max(numbers, default=-1) + 1


def merge_unique_features(*, existing: dict[str, Any], parts_dir: Path, eligible_ids: list[str], output: Path, settings: dict[str, Any]) -> dict[str, Any]:
    new_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("features", "sample_ids", "labels", "query_ids", "direct_scores", "direct_completions")}
    active_tokens = padded_tokens = truncated_records = 0
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            for key in new_arrays:
                new_arrays[key].append(np.asarray(payload[key]))
            active_tokens += int(payload["active_tokens"]); padded_tokens += int(payload["padded_tokens"]); truncated_records += int(payload["truncated_records"])
    new = {key: np.concatenate(value, axis=0) if value else np.asarray([], dtype=np.float16) for key, value in new_arrays.items()}
    if new["features"].size == 0:
        new["features"] = np.empty((0, 2, QWEN3_5_4B_HIDDEN_SIZE), dtype=np.float16)
    new_ids = new["sample_ids"].astype(str)
    if len(set(new_ids.tolist())) != len(new_ids):
        raise ValueError("New B-space feature parts have duplicate ids")
    new_index = {value: index for index, value in enumerate(new_ids.tolist())}
    expected = set(eligible_ids)
    combined_ids = set(existing["id_to_index"]).union(new_index)
    if combined_ids != expected:
        raise RuntimeError(f"Unique B-space cache coverage mismatch: missing={len(expected-combined_ids)}, extra={len(combined_ids-expected)}")
    ordered_features: list[np.ndarray] = []; ordered_labels: list[int] = []; ordered_query_ids: list[str] = []; ordered_direct_scores: list[int] = []; ordered_direct_completions: list[str] = []
    for b_id in eligible_ids:
        if b_id in existing["id_to_index"]:
            index = existing["id_to_index"][b_id]
            ordered_features.append(existing["features"][index]); ordered_labels.append(int(existing["labels"][index])); ordered_query_ids.append(str(existing["query_ids"][index])); ordered_direct_scores.append(int(existing["direct_scores"][index])); ordered_direct_completions.append(str(existing["direct_completions"][index]))
        else:
            index = new_index[b_id]
            ordered_features.append(new["features"][index]); ordered_labels.append(int(new["labels"][index])); ordered_query_ids.append(str(new["query_ids"][index])); ordered_direct_scores.append(int(new["direct_scores"][index])); ordered_direct_completions.append(str(new["direct_completions"][index]))
    features = np.stack(ordered_features).astype(np.float16)
    metadata = {
        **settings,
        "num_records": len(eligible_ids), "shape": list(features.shape),
        "reused_5x6_unique_b_rows": len(existing["id_to_index"]), "new_qwen_unique_b_rows": len(new_ids),
        "active_tokens_new_qwen": active_tokens, "padded_tokens_new_qwen": padded_tokens,
        "truncated_records_new_qwen": truncated_records,
        "direct_judge_parsed_rows": int(sum(1 <= value <= 5 for value in ordered_direct_scores)),
        "direct_judge_parse_rate": float(sum(1 <= value <= 5 for value in ordered_direct_scores) / max(len(ordered_direct_scores), 1)),
        "feature_storage_dtype": "float16", "reuse_policy": "identical_b_id_and_direct_judge_prompt_only",
    }
    np.savez_compressed(output, features=features, sample_ids=np.asarray(eligible_ids), labels=np.asarray(ordered_labels, dtype=np.int8), query_ids=np.asarray(ordered_query_ids), direct_scores=np.asarray(ordered_direct_scores, dtype=np.int8), direct_completions=np.asarray(ordered_direct_completions), metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)))
    return {"features": features, "sample_ids": np.asarray(eligible_ids), "labels": np.asarray(ordered_labels, dtype=np.int8), "query_ids": np.asarray(ordered_query_ids), "direct_scores": np.asarray(ordered_direct_scores, dtype=np.int8), "direct_completions": np.asarray(ordered_direct_completions), "id_to_index": {value: index for index, value in enumerate(eligible_ids)}, "metadata": metadata}


def materialize_cells(*, cells_dir: Path, output_dir: Path, merged: dict[str, Any], expected_ids: set[str], settings: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    total = 0; cell_count = 0; references: Counter[str] = Counter()
    for domain in DOMAINS:
        for skill in SKILLS:
            source = cells_dir / slug(domain) / f"{slug(skill)}.jsonl"
            rows = [row for row in iter_jsonl(source) if integer_score(row.get("ground_truth")) is not None and str(row.get("candidate_response") or "").strip()]
            if not rows:
                raise ValueError(f"Full 10x12 cell unexpectedly empty after model-input filtering: {domain} x {skill}")
            ids = [str(row["b_id"]) for row in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate b_id in cell {domain} x {skill}")
            missing = [value for value in ids if value not in merged["id_to_index"]]
            if missing:
                raise RuntimeError(f"Cell {domain} x {skill} references missing B id {missing[0]}")
            indices = np.asarray([merged["id_to_index"][value] for value in ids], dtype=np.int64)
            output = output_dir / f"{slug(domain)}__{slug(skill)}.npz"
            metadata = {**settings, "cell_id": f"{slug(domain)}__{slug(skill)}", "domain_id": domain, "task_id": skill, "num_records": len(rows), "shape": [len(rows), 2, QWEN3_5_4B_HIDDEN_SIZE]}
            np.savez_compressed(
                output,
                features=merged["features"][indices], sample_ids=np.asarray(ids),
                labels=np.asarray([integer_score(row["ground_truth"]) for row in rows], dtype=np.int8),
                query_ids=np.asarray([str(row["base_id"]) for row in rows]),
                domain_ids=np.asarray([domain] * len(rows)), task_ids=np.asarray([skill] * len(rows)),
                membership_ids=np.asarray([str(row["membership_id"]) for row in rows]),
                direct_scores=merged["direct_scores"][indices],
                direct_completions=merged["direct_completions"][indices],
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
            )
            total += len(rows); cell_count += 1; references.update(ids)
    if set(references) != expected_ids:
        raise RuntimeError(f"Cell materialization coverage mismatch: missing={len(expected_ids-set(references))}, extra={len(set(references)-expected_ids)}")
    return {"cell_feature_files": cell_count, "cell_membership_rows": total, "unique_b_rows_referenced": len(references), "duplicate_membership_references": total - len(references)}


def write_unique_direct_predictions(*, output: Path, b_space: Path, merged: dict[str, Any]) -> dict[str, Any]:
    """Persist one Direct Judge record per unique B prompt, not per membership."""

    source_by_id = {
        str(row["b_id"]): row
        for row in iter_jsonl(b_space)
        if integer_score(row.get("ground_truth")) is not None
        and str(row.get("candidate_response") or "").strip()
    }
    ids = merged["sample_ids"].astype(str).tolist()
    if set(source_by_id) != set(ids):
        raise RuntimeError("Direct Judge source B rows are not aligned with merged hidden states")
    parsed = 0
    with output.open("w", encoding="utf-8") as handle:
        for index, b_id in enumerate(ids):
            score = int(merged["direct_scores"][index])
            parsed += int(1 <= score <= 5)
            row = source_by_id[b_id]
            payload = {
                "b_id": b_id,
                "response_id": row["response_id"],
                "base_id": row["base_id"],
                "task_id": row["task_id"],
                "ground_truth": integer_score(row["ground_truth"]),
                "direct_score": score if 1 <= score <= 5 else None,
                "direct_judge_raw_completion": str(merged["direct_completions"][index]),
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "unique_direct_judge_predictions": len(ids),
        "unique_direct_judge_parsed_rows": parsed,
        "unique_direct_judge_parse_rate": parsed / max(len(ids), 1),
    }


def _direct_judge_module():
    path = ROOT / "scripts/llm_judge_ood/49_run_flask_minimal_direct_judge.py"
    spec = importlib.util.spec_from_file_location("flask_direct_judge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Direct Judge helpers from {path}")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def integer_score(value: Any) -> int | None:
    try: score = float(value)
    except (TypeError, ValueError): return None
    rounded = int(round(score))
    return rounded if abs(score-rounded) < 1e-8 and 1 <= rounded <= 5 else None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists(): raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip(): yield json.loads(line)


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


if __name__ == "__main__":
    main()
