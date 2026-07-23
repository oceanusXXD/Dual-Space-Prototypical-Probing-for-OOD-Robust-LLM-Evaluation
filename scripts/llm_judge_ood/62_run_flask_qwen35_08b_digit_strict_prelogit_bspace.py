#!/usr/bin/env python3
"""Run Qwen3.5-0.8B FLASK Direct Judge and strict final-prelogit B-space."""

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
from sklearn.metrics import cohen_kappa_score
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json, write_jsonl
from src.models.extract_hidden import load_qwen_model, qwen_text_config

MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
DOMAINS_5X6 = ("Humanities", "Language", "Social Science", "History", "Culture")
SKILLS_5X6 = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
)
DOMAINS_3X3 = DOMAINS_5X6[:3]
SKILLS_3X3 = SKILLS_5X6[:3]
DOMAINS_10X12 = (
    "Humanities", "Language", "Social Science", "History", "Culture",
    "Technology", "Coding", "Math", "Natural Science", "Health",
)
SKILLS_10X12 = (
    "Comprehension", "Factuality", "Logical Correctness", "Commonsense Understanding",
    "Completeness", "Insightfulness", "Readability", "Conciseness",
    "Logical Robustness", "Logical Efficiency", "Metacognition", "Harmlessness",
)
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("3x3", "5x6", "10x12"), default="5x6")
    parser.add_argument(
        "--b-space",
        type=Path,
        default=Path("datasets/processed/flask_domain_task_v1/b_space_single_domain.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/"
            "qwen35_08b_5x6_digit_direct_judge_strict_prelogit_bspace"
        ),
    )
    parser.add_argument("--model-path", type=Path, default=Path("/home/zeus/models/qwen3.5-0.8b"))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-batch-tokens", type=int, default=1572864)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--pad-to-multiple-of", type=int, default=128)
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("sdpa", "flash_attention_2"),
        default="flash_attention_2",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.batch_size, args.max_batch_tokens, args.max_length, args.pad_to_multiple_of) < 1:
        raise ValueError("batch, token, length, and padding settings must be positive")
    if args.max_length % args.pad_to_multiple_of:
        raise ValueError("--max-length must be divisible by --pad-to-multiple-of")

    domains, skills = scope_values(args.scope)
    if args.scope == "10x12" and args.b_space == Path(
        "datasets/processed/flask_domain_task_v1/b_space_single_domain.jsonl"
    ):
        args.b_space = Path("datasets/processed/flask_domain_task_v1/b_space_full.jsonl")
    if args.scope == "10x12" and args.output_dir == Path(
        "artifacts/flask_minimal_validation/"
        "qwen35_08b_5x6_digit_direct_judge_strict_prelogit_bspace"
    ):
        args.output_dir = Path(
            "artifacts/flask_full_validation/"
            "qwen35_08b_10x12_digit_direct_judge_strict_prelogit_bspace"
        )
    selected, cleaning = select_rows(args.b_space, domains, skills)
    output = args.output_dir
    if args.overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    parts_dir = output / "strict_final_prelogit_b_space.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output / "direct_judge_rows.jsonl"
    predictions_path = output / "direct_judge_predictions.jsonl"
    scored_path = output / "b_space_with_direct_judge.jsonl"
    features_path = output / "strict_final_prelogit_b_space_features.npz"
    metadata_path = output / "strict_final_prelogit_b_space_features.metadata.json"
    summary_path = output / "summary.json"
    write_jsonl(selected_path, selected)

    prediction_rows = load_jsonl(predictions_path)
    completed_predictions = {str(row["b_id"]) for row in prediction_rows}
    completed_features = feature_part_ids(parts_dir)
    if completed_predictions != completed_features:
        raise ValueError(
            "Prediction and strict-prelogit parts are not aligned; use --overwrite."
        )
    pending = [row for row in selected if str(row["b_id"]) not in completed_predictions]

    started = time.perf_counter()
    runtime: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(args.model_path),
        "scope": args.scope,
        "domains": list(domains),
        "skills": list(skills),
        "initial_batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "max_length": int(args.max_length),
        "max_new_tokens": int(args.max_new_tokens),
        "attn_implementation_requested": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "feature_scope": "strict_final_prelogit",
        "feature_definition": (
            "final-normalized decoder state at the last prompt token, "
            "immediately before lm_head predicts the first generated score digit"
        ),
        "single_generation_for_direct_judge_and_b_space": True,
    }

    if pending:
        tokenizer, model, device = load_qwen_model(
            args.model_path,
            revision=MODEL_REVISION,
            device="cuda",
            torch_dtype=args.torch_dtype,
            attn_implementation=args.attn_implementation,
            tf32=True,
            local_files_only=True,
        )
        if device.type != "cuda":
            raise RuntimeError("Qwen3.5-0.8B requires CUDA")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        text_config = qwen_text_config(model)
        hidden_size = int(getattr(text_config, "hidden_size", 0))
        num_layers = int(getattr(text_config, "num_hidden_layers", 0))
        if hidden_size <= 0 or num_layers <= 0:
            raise RuntimeError("Qwen text config has invalid hidden size or layer count")
        runtime.update({
            "model_type": str(getattr(text_config, "model_type", "unknown")),
            "model_hidden_size": hidden_size,
            "num_model_layers": num_layers,
        })
        torch.cuda.reset_peak_memory_stats()
        capture = StrictFinalPrelogitCapture(
            getattr(model, "model", model),
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        runtime.update(
            generate_rows(
                rows=pending,
                tokenizer=tokenizer,
                model=model,
                device=device,
                predictions_path=predictions_path,
                parts_dir=parts_dir,
                capture=capture,
                hidden_size=hidden_size,
                batch_size=int(args.batch_size),
                max_batch_tokens=int(args.max_batch_tokens),
                max_length=int(args.max_length),
                max_new_tokens=int(args.max_new_tokens),
                pad_to_multiple_of=int(args.pad_to_multiple_of),
                scope=args.scope,
            )
        )
        capture.close()
        runtime["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        runtime["peak_cuda_memory_gib"] = float(torch.cuda.max_memory_allocated() / 1024**3)
        del model
        torch.cuda.empty_cache()
    else:
        with np.load(features_path, allow_pickle=False) as payload:
            feature_metadata = json.loads(str(payload["metadata_json"].item()))
        runtime["resumed_without_generation"] = True
        hidden_size = int(feature_metadata["model_hidden_size"])
        num_layers = int(feature_metadata["num_model_layers"])

    prediction_rows = load_jsonl(predictions_path)
    by_id = {str(row["b_id"]): row for row in prediction_rows}
    selected_ids = {str(row["b_id"]) for row in selected}
    if set(by_id) != selected_ids:
        raise RuntimeError(
            f"Direct Judge incomplete: missing={len(selected_ids - set(by_id))}, "
            f"extra={len(set(by_id) - selected_ids)}"
        )

    metadata = merge_parts(
        parts_dir=parts_dir,
        output=features_path,
        selected=selected,
        hidden_size=hidden_size,
        settings={
            "artifact_type": "flask_qwen35_08b_digit_direct_judge_strict_final_prelogit_bspace_v1",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_path": str(args.model_path),
            "model_hidden_size": hidden_size,
            "num_model_layers": num_layers,
            "scope": args.scope,
            "domains": list(domains),
            "skills": list(skills),
            "feature_scope": "strict_final_prelogit",
            "position": "last_prompt_token_before_first_generated_digit",
            "pooling": None,
            "pooling_formula": None,
            "does_not_use_layer_mean": True,
            "does_not_include_generated_score_token": True,
            "prompt_template_version": direct_helpers().DIRECT_JUDGE_TEMPLATE_VERSION,
            "prompt_template_sha256": hashlib.sha256(
                direct_helpers().DIRECT_JUDGE_TEMPLATE.encode("utf-8")
            ).hexdigest(),
            "source_b_space": str(args.b_space),
            "feature_dtype": "float16",
        },
    )
    write_json(metadata_path, metadata)
    write_scored_b_space(selected, by_id, scored_path)
    cell_summary = materialize_cells(
        output / "cells", selected, by_id, features_path, hidden_size, domains, skills
    )
    summary = summarize(
        selected=selected,
        predictions=prediction_rows,
        domains=domains,
        skills=skills,
        cleaning=cleaning,
        runtime={**runtime, "elapsed_seconds": time.perf_counter() - started},
        metadata=metadata,
        cell_summary=cell_summary,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary["global_metrics"], ensure_ascii=False, indent=2))


def scope_values(scope: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if scope == "3x3":
        return DOMAINS_3X3, SKILLS_3X3
    if scope == "10x12":
        return DOMAINS_10X12, SKILLS_10X12
    return DOMAINS_5X6, SKILLS_5X6


def select_rows(
    path: Path, domains: tuple[str, ...], skills: tuple[str, ...]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in iter_jsonl(path):
        stats["raw_rows"] += 1
        domain_ids = list(row.get("domain_ids") or [])
        matched_domains = tuple(domain for domain in domain_ids if str(domain) in domains)
        if not matched_domains:
            stats["outside_scope"] += 1
            continue
        skill = str(row.get("task_id") or "")
        if skill not in skills:
            stats["outside_scope"] += 1
            continue
        if integer_score(row.get("ground_truth")) is None:
            stats["dropped_noninteger_score"] += 1
            continue
        if not str(row.get("candidate_response") or "").strip():
            stats["dropped_empty_candidate_response"] += 1
            continue
        b_id = str(row["b_id"])
        if b_id in seen:
            raise ValueError(f"Duplicate B id: {b_id}")
        seen.add(b_id)
        rows.append(row)
    rows.sort(
        key=lambda row: (
            min(domains.index(str(domain)) for domain in row["domain_ids"] if str(domain) in domains),
            skills.index(str(row["task_id"])),
            rank(row["b_id"]),
        )
    )
    for domain in domains:
        for skill in skills:
            count = sum(
                domain in tuple(str(value) for value in row["domain_ids"])
                and str(row["task_id"]) == skill
                for row in rows
            )
            if count == 0:
                raise ValueError(f"Empty cell after cleaning: {domain} x {skill}")
            stats[f"cell::{domain}::{skill}"] = int(count)
    stats["selected_rows"] = len(rows)
    return rows, dict(stats)


def generate_rows(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    predictions_path: Path,
    parts_dir: Path,
    capture: "StrictFinalPrelogitCapture",
    hidden_size: int,
    batch_size: int,
    max_batch_tokens: int,
    max_length: int,
    max_new_tokens: int,
    pad_to_multiple_of: int,
    scope: str,
) -> dict[str, Any]:
    direct = direct_helpers()
    prompts = [direct._chat_prompt(row, tokenizer) for row in rows]
    token_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    order = sorted(range(len(rows)), key=lambda i: (len(token_ids[i]), rank(rows[i]["b_id"])))
    batches = make_batches(
        order,
        token_ids,
        max_documents=batch_size,
        max_tokens=max_batch_tokens,
        max_length=max_length,
        pad_to_multiple_of=pad_to_multiple_of,
    )
    next_part = next_part_number(parts_dir)
    progress = tqdm(total=len(rows), desc=f"Qwen3.5-0.8B strict {scope} Direct Judge")
    effective: list[int] = []
    try:
        while batches:
            indices = batches.pop(0)
            try:
                batch_rows = [rows[i] for i in indices]
                batch_ids = [
                    direct._head_tail(ids=token_ids[i], max_length=max_length)
                    for i in indices
                ]
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
                completions = tokenizer.batch_decode(
                    generated[:, prompt_width:], skip_special_tokens=True
                )
                part = parts_dir / f"part-{next_part:06d}.npz"
                write_part(
                    part,
                    rows=batch_rows,
                    indices=indices,
                    features=features,
                    hidden_size=hidden_size,
                    attention_mask=encoded["attention_mask"],
                    completions=completions,
                )
                next_part += 1
                append_jsonl(
                    predictions_path,
                    [
                        {
                            "b_id": row["b_id"],
                            "response_id": row["response_id"],
                            "base_id": row["base_id"],
                            "domain_id": row["domain_ids"][0],
                            "domain_ids": [str(value) for value in row["domain_ids"]],
                            "domain_id_set": domain_id_key(row),
                            "task_id": row["task_id"],
                            "generator_id": row["generator_id"],
                            "ground_truth": integer_score(row["ground_truth"]),
                            "predicted_score": direct.parse_score(completion),
                            "raw_completion": completion,
                            "input_token_count": len(ids),
                            "truncated": len(original) > len(ids),
                        }
                        for row, completion, original, ids in zip(
                            batch_rows,
                            completions,
                            (token_ids[i] for i in indices),
                            batch_ids,
                            strict=True,
                        )
                    ],
                )
                progress.update(len(indices))
                effective.append(len(indices))
                del generated, features, encoded
                capture.clear()
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                recoverable = isinstance(exc, torch.OutOfMemoryError) or any(
                    token in str(exc).lower()
                    for token in (
                        "out of memory",
                        "canuse32bitindexmath",
                        "triton error [cuda]: invalid argument",
                        "non-finite strict final prelogit",
                    )
                )
                capture.clear()
                torch.cuda.empty_cache()
                if not recoverable or len(indices) == 1:
                    raise
                midpoint = len(indices) // 2
                batches.insert(0, indices[midpoint:])
                batches.insert(0, indices[:midpoint])
                print(
                    f"strict-prelogit batch split {len(indices)} -> "
                    f"{midpoint}+{len(indices) - midpoint}",
                    flush=True,
                )
    finally:
        progress.close()
    return {
        "generated_rows": len(rows),
        "batch_count": len(effective),
        "effective_batch_sizes": sorted(set(effective)),
    }


class StrictFinalPrelogitCapture:
    def __init__(
        self, base_model: torch.nn.Module, *, num_layers: int, hidden_size: int
    ) -> None:
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
        self.expected_shape = (
            int(attention_mask.shape[0]),
            int(attention_mask.shape[1]),
        )

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        if not isinstance(hidden, torch.Tensor):
            raise RuntimeError("Qwen final norm output is not a tensor")
        if self.attention_mask is None or self.expected_shape is None:
            return
        batch, length = self.expected_shape
        if tuple(hidden.shape[:2]) != (batch, length):
            return
        if self.value is not None:
            raise RuntimeError("Captured strict final prelogit twice")
        positions = torch.arange(length, device=self.attention_mask.device).unsqueeze(0)
        last = (
            self.attention_mask.to(dtype=torch.long) * positions
        ).max(dim=1).values.to(hidden.device)
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


def write_part(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    indices: list[int],
    features: torch.Tensor,
    hidden_size: int,
    attention_mask: torch.Tensor,
    completions: list[str],
) -> None:
    if tuple(features.shape) != (len(rows), 1, hidden_size):
        raise RuntimeError(f"Invalid feature shape: {tuple(features.shape)}")
    np.savez(
        path,
        features=features.cpu().numpy().astype(np.float16),
        indices=np.asarray(indices, dtype=np.int64),
        sample_ids=np.asarray([str(row["b_id"]) for row in rows]),
        labels=np.asarray([integer_score(row["ground_truth"]) for row in rows], dtype=np.int8),
        query_ids=np.asarray([str(row["base_id"]) for row in rows]),
        audit_document_group_ids=np.asarray([str(row["base_id"]) for row in rows]),
        input_document_ids=np.asarray([str(row["response_id"]) for row in rows]),
        domain_ids=np.asarray([str(row["domain_ids"][0]) for row in rows]),
        domain_id_sets=np.asarray([domain_id_key(row) for row in rows]),
        task_ids=np.asarray([str(row["task_id"]) for row in rows]),
        direct_scores=np.asarray(
            [direct_score(text) for text in completions], dtype=np.int8
        ),
        direct_completions=np.asarray(completions),
        active_tokens=np.asarray(int(attention_mask.sum().item()), dtype=np.int64),
        padded_tokens=np.asarray(int(attention_mask.numel()), dtype=np.int64),
    )


def merge_parts(
    *,
    parts_dir: Path,
    output: Path,
    selected: list[dict[str, Any]],
    hidden_size: int,
    settings: dict[str, Any],
) -> dict[str, Any]:
    fields = (
        "features",
        "sample_ids",
        "labels",
        "query_ids",
        "audit_document_group_ids",
        "input_document_ids",
        "domain_ids",
        "domain_id_sets",
        "task_ids",
        "direct_scores",
        "direct_completions",
    )
    pieces: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    active_tokens = padded_tokens = 0
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            for field in fields:
                pieces[field].append(np.asarray(payload[field]))
            active_tokens += int(payload["active_tokens"])
            padded_tokens += int(payload["padded_tokens"])
    if not pieces["features"]:
        raise RuntimeError("No strict final prelogit parts found")
    merged = {field: np.concatenate(values, axis=0) for field, values in pieces.items()}
    ids = merged["sample_ids"].astype(str)
    selected_ids = [str(row["b_id"]) for row in selected]
    if len(set(ids.tolist())) != len(ids) or set(ids.tolist()) != set(selected_ids):
        raise RuntimeError("Strict final prelogit sample coverage is invalid")
    source_index = {value: index for index, value in enumerate(ids.tolist())}
    order = np.asarray([source_index[value] for value in selected_ids], dtype=np.int64)
    ordered = {field: value[order] for field, value in merged.items()}
    if tuple(ordered["features"].shape) != (len(selected), 1, hidden_size):
        raise RuntimeError(f"Unexpected merged feature shape {ordered['features'].shape}")
    metadata = {
        **settings,
        "shape": list(ordered["features"].shape),
        "num_records": len(selected),
        "active_tokens": active_tokens,
        "padded_tokens": padded_tokens,
        "padding_fraction": 1.0 - active_tokens / max(padded_tokens, 1),
        "feature_storage_dtype": "float16",
        "direct_judge_parsed_rows": int(np.sum((ordered["direct_scores"] >= 1) & (ordered["direct_scores"] <= 5))),
    }
    np.savez_compressed(
        output,
        **ordered,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def write_scored_b_space(
    selected: list[dict[str, Any]], by_id: dict[str, dict[str, Any]], path: Path
) -> None:
    write_jsonl(
        path,
        [
            {
                **row,
                "direct_score": by_id[str(row["b_id"])].get("predicted_score"),
                "direct_judge_raw_completion": by_id[str(row["b_id"])].get("raw_completion", ""),
                "judge_prompt_template_version": direct_helpers().DIRECT_JUDGE_TEMPLATE_VERSION,
                "judge_model_id": MODEL_ID,
                "b_space_feature_index": index,
            }
            for index, row in enumerate(selected)
        ],
    )


def materialize_cells(
    output_dir: Path,
    selected: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
    features_path: Path,
    hidden_size: int,
    domains: tuple[str, ...],
    skills: tuple[str, ...],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(features_path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"])
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
    id_to_index = {value: index for index, value in enumerate(sample_ids.tolist())}
    count = 0
    for domain in domains:
        for skill in skills:
            rows = [
                row for row in selected
                if domain in domain_values(row) and str(row["task_id"]) == skill
            ]
            if not rows:
                continue
            ids = [str(row["b_id"]) for row in rows]
            indices = np.asarray([id_to_index[value] for value in ids], dtype=np.int64)
            np.savez_compressed(
                output_dir / f"{slug(domain)}__{slug(skill)}.npz",
                features=features[indices],
                sample_ids=np.asarray(ids),
                labels=np.asarray([integer_score(row["ground_truth"]) for row in rows], dtype=np.int8),
                query_ids=np.asarray([str(row["base_id"]) for row in rows]),
                domain_ids=np.asarray([domain] * len(rows)),
                source_domain_id_sets=np.asarray([domain_id_key(row) for row in rows]),
                task_ids=np.asarray([skill] * len(rows)),
                direct_scores=np.asarray([
                    int(by_id[value]["predicted_score"])
                    if by_id[value].get("predicted_score") is not None else -1
                    for value in ids
                ], dtype=np.int8),
                direct_completions=np.asarray([
                    str(by_id[value].get("raw_completion") or "") for value in ids
                ]),
                metadata_json=np.asarray(json.dumps({
                    "artifact_type": "flask_qwen35_08b_cell_strict_final_prelogit_bspace_v1",
                    "model_id": MODEL_ID,
                    "domain_id": domain,
                    "task_id": skill,
                    "shape": [len(rows), 1, hidden_size],
                    "source_rows_are_membership_filtered": True,
                    "feature_scope": "strict_final_prelogit",
                }, ensure_ascii=False)),
            )
            count += 1
    return {"cell_feature_files": count}


def summarize(
    *,
    selected: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    domains: tuple[str, ...],
    skills: tuple[str, ...],
    cleaning: dict[str, int],
    runtime: dict[str, Any],
    metadata: dict[str, Any],
    cell_summary: dict[str, Any],
) -> dict[str, Any]:
    by_id = {str(row["b_id"]): row for row in predictions}
    rows = [by_id[str(row["b_id"])] for row in selected]
    return {
        "artifact_type": "flask_qwen35_08b_digit_direct_judge_strict_final_prelogit_bspace_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "domains": list(domains),
        "skills": list(skills),
        "selected_rows": len(selected),
        "prediction_rows": len(rows),
        "cleaning": cleaning,
        "runtime": runtime,
        "metadata": metadata,
        "cell_summary": cell_summary,
        "global_metrics": metrics(rows),
        "cell_metrics": {
            f"{domain}::{skill}": metrics([
                by_id[str(row["b_id"])] for row in selected
                if domain in domain_values(row) and str(row["task_id"]) == skill
            ])
            for domain in domains
            for skill in skills
        },
    }


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row for row in rows
        if row.get("predicted_score") is not None
        and integer_score(row.get("ground_truth")) is not None
    ]
    if not valid:
        return {"rows": len(rows), "parsed_rows": 0, "parse_rate": 0.0}
    truth = np.asarray([integer_score(row["ground_truth"]) for row in valid], dtype=int)
    pred = np.asarray([int(row["predicted_score"]) for row in valid], dtype=int)
    return {
        "rows": len(rows),
        "parsed_rows": len(valid),
        "parse_rate": len(valid) / max(len(rows), 1),
        "mae": float(np.mean(np.abs(pred - truth))),
        "exact_accuracy": float(np.mean(pred == truth)),
        "plus_minus_1_accuracy": float(np.mean(np.abs(pred - truth) <= 1)),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(truth, pred, labels=[1, 2, 3, 4, 5], weights="quadratic")
        ),
    }


def direct_score(text: str) -> int:
    value = str(text).strip()
    return int(value) if re.fullmatch(r"[1-5]", value) else -1


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
        rounded = min(
            max_length,
            ((bounded + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of,
        )
        candidate = max(width, rounded)
        if current and (
            len(current) + 1 > max_documents
            or candidate * (len(current) + 1) > max_tokens
        ):
            batches.append(current)
            current = []
            width = 0
        current.append(index)
        width = max(width, rounded)
    if current:
        batches.append(current)
    return batches


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
    numbers = []
    for path in parts_dir.glob("part-*.npz"):
        match = re.fullmatch(r"part-(\d+)\.npz", path.name)
        if match:
            numbers.append(int(match.group(1)))
    return max(numbers, default=-1) + 1


def direct_helpers() -> Any:
    path = ROOT / "scripts/llm_judge_ood/49_run_flask_minimal_direct_judge.py"
    spec = importlib.util.spec_from_file_location("flask_direct_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import Direct Judge helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def integer_score(value: Any) -> int | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    rounded = int(round(score))
    return rounded if abs(score - rounded) < 1e-8 and 1 <= rounded <= 5 else None


def rank(value: Any) -> str:
    return hashlib.sha256(f"{SEED}::{value}".encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def domain_values(row: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(value) for value in row.get("domain_ids") or ())


def domain_id_key(row: dict[str, Any]) -> str:
    return "|".join(domain_values(row))


if __name__ == "__main__":
    main()
