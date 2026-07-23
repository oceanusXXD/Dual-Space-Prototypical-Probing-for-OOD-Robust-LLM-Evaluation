#!/usr/bin/env python3
"""Extract the complete cleaned FLASK A-space without re-forwarding 5x6 rows.

A-space contains the raw non-empty candidate answer only.  This runner selects
one answer for every response represented by at least one formal full-view
B-space row (integer 1--5 label), reuses the already extracted 5x6 A-space
vectors when their response text is identical, and forwards only new answers.
It also writes lightweight per-domain index files that reference the unified
unique A-space cache rather than duplicating hidden-state tensors.
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

QWEN3_5_08B_MODEL_ID = "Qwen/Qwen3.5-0.8B"
QWEN3_5_08B_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
QWEN3_5_08B_HIDDEN_SIZE = 1024

DOMAINS = (
    "Humanities", "Language", "Social Science", "History", "Culture",
    "Technology", "Coding", "Math", "Natural Science", "Health",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b-space", type=Path,
        default=Path("datasets/processed/flask_domain_task_v1/b_space_full.jsonl"),
    )
    parser.add_argument(
        "--reuse-features", type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "a_space_hidden_states.npz"
        ),
    )
    parser.add_argument(
        "--reuse-contract", type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "a_space_contract.jsonl"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/flask_full_a_space"))
    parser.add_argument("--model-path", type=Path, default=Path("/home/zeus/models/qwen3.5-4b"))
    parser.add_argument("--model-id", default=QWEN3_5_4B_MODEL_ID)
    parser.add_argument("--model-revision", default=QWEN3_5_4B_REVISION)
    parser.add_argument("--expected-hidden-size", type=int, default=QWEN3_5_4B_HIDDEN_SIZE)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--max-batch-tokens", type=int, default=262144)
    parser.add_argument("--tokenization-chunk-size", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--pad-to-multiple-of", type=int, default=128)
    parser.add_argument("--layers", nargs="+", type=int, default=[-10, -1])
    parser.add_argument(
        "--attn-implementation", choices=["sdpa", "flash_attention_2"],
        default="flash_attention_2",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--no-reuse-features",
        action="store_true",
        help="Do not reuse the 4B 5x6 A-space cache; required for Qwen3.5-0.8B.",
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
    parts_dir = output_dir / "new_parts"
    parts_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    settings = {
        "artifact_type": "flask_full_10x12_a_space_hidden_v1",
        "source_b_space": str(args.b_space),
        "reuse_features": None if args.no_reuse_features else str(args.reuse_features),
        "reuse_contract": None if args.no_reuse_features else str(args.reuse_contract),
        "model_id": str(args.model_id),
        "model_revision": str(args.model_revision),
        "model_path": str(args.model_path),
        "model_hidden_size": int(args.expected_hidden_size),
        "feature_scope": "input_document",
        "prompt_template_version": "raw_input_document_v1",
        "layers_requested": list(args.layers),
        "pooling": "masked_mean",
        "feature_dtype": "float16",
        "integer_score_only": True,
        "nonempty_candidate_response_only": True,
        "batch_size": int(args.batch_size),
        "max_batch_tokens": int(args.max_batch_tokens),
        "max_length": int(args.max_length),
        "pad_to_multiple_of": int(args.pad_to_multiple_of),
        "attn_implementation": str(args.attn_implementation),
        "torch_dtype": str(args.torch_dtype),
    }
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete"):
            print(json.dumps(manifest.get("summary", {}), ensure_ascii=False, indent=2))
            return
        for key, value in settings.items():
            if manifest.get(key) != value:
                raise ValueError(f"Existing A-space setting {key!r} differs; use --overwrite")
    else:
        write_json(manifest_path, {**settings, "complete": False})

    records, cleaning = load_full_a_records(args.b_space)
    write_jsonl(output_dir / "a_space_contract.jsonl", [record["contract"] for record in records])
    existing = load_existing_a_space(
        args.reuse_features,
        args.reuse_contract,
        records,
        enabled=not args.no_reuse_features,
        model_id=str(args.model_id),
        model_revision=str(args.model_revision),
        hidden_size=int(args.expected_hidden_size),
        num_feature_layers=len(args.layers),
    )
    part_ids = load_part_ids(parts_dir)
    if set(existing["id_to_index"]).intersection(part_ids):
        raise ValueError("Reused 5x6 A-space and new A-space parts overlap")
    known_ids = set(existing["id_to_index"]).union(part_ids)
    pending = [record for record in records if record["response_id"] not in known_ids]

    started = time.perf_counter()
    runtime: dict[str, Any] = {
        "cleaning": cleaning,
        "eligible_unique_a_rows": len(records),
        "reused_5x6_unique_a_rows": len(existing["id_to_index"]),
        "resumed_new_unique_a_rows": len(part_ids),
        "new_unique_a_rows_pending": len(pending),
    }
    if pending:
        tokenizer, model, device = load_qwen_model(
            args.model_path, revision=str(args.model_revision), device="cuda",
            torch_dtype=args.torch_dtype, attn_implementation=args.attn_implementation,
            tf32=True, local_files_only=True,
        )
        if device.type != "cuda":
            raise RuntimeError("Full A-space extraction requires CUDA")
        tokenizer.padding_side = "right"
        text_config = qwen_text_config(model)
        hidden_size = int(getattr(text_config, "hidden_size", 0))
        num_layers = int(getattr(text_config, "num_hidden_layers", 0))
        selected_layers = resolve_layers(
            int(getattr(text_config, "num_hidden_layers", 0)) + 1, list(args.layers)
        )
        if hidden_size != int(args.expected_hidden_size):
            raise RuntimeError(
                f"Unexpected Qwen hidden size {hidden_size}; expected {args.expected_hidden_size}"
            )
        if num_layers <= 0:
            raise RuntimeError("Unexpected Qwen layer count")
        capture = MaskedMeanPromptLayerCapture(
            getattr(model, "model", model),
            selected_layers,
            num_layers=num_layers,
            hidden_size=hidden_size,
        )
        torch.cuda.reset_peak_memory_stats()
        generated = extract_pending(
            pending=pending, parts_dir=parts_dir, tokenizer=tokenizer, model=model,
            device=device, capture=capture, args=args,
        )
        capture.close()
        runtime.update({
            **generated,
            "model_hidden_size": hidden_size,
            "num_model_layers": num_layers,
            "layers_resolved": selected_layers,
        })
        runtime["peak_cuda_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        del model
        torch.cuda.empty_cache()
    else:
        runtime["resumed_without_qwen_forward"] = True
    runtime["elapsed_seconds"] = time.perf_counter() - started

    merged = merge_features(
        records=records, existing=existing, parts_dir=parts_dir,
        output=output_dir / "a_space_hidden_states.npz", settings=settings,
        num_feature_layers=len(args.layers),
        hidden_size=int(args.expected_hidden_size),
    )
    domain_summary = write_domain_indexes(
        records=records, merged=merged, output_dir=output_dir / "domain_indexes",
        settings=settings,
    )
    summary = {**settings, **runtime, **merged["metadata"], **domain_summary}
    write_json(output_dir / "summary.json", summary)
    write_json(manifest_path, {**settings, "complete": True, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_full_a_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_response: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    for row in iter_jsonl(path):
        stats["source_b_rows"] += 1
        if integer_score(row.get("ground_truth")) is None:
            stats["dropped_noninteger_score"] += 1
            continue
        candidate = str(row.get("candidate_response") or "")
        if not candidate.strip():
            stats["dropped_empty_candidate_response"] += 1
            continue
        response_id = str(row["response_id"])
        domains = tuple(str(value) for value in row.get("domain_ids") or ())
        if not domains or any(value not in DOMAINS for value in domains):
            raise ValueError(f"Response {response_id!r} has invalid domain ids")
        previous = by_response.get(response_id)
        if previous is not None:
            if previous["candidate_response"] != candidate:
                raise ValueError(f"Response {response_id!r} has inconsistent candidate text")
            if previous["base_id"] != str(row["base_id"]) or previous["generator_id"] != str(row.get("generator_id") or ""):
                raise ValueError(f"Response {response_id!r} has inconsistent response metadata")
            if previous["domain_ids"] != domains:
                raise ValueError(f"Response {response_id!r} has inconsistent domain ids")
            continue
        contract = {
            "sample_id": response_id,
            "query_id": str(row["base_id"]),
            "query_text": str(row.get("instruction") or ""),
            "document_text": candidate,
            "label": None,
            "split": "all",
            "judge_provenance_id": domains[0],
            "base_document_id": response_id,
            "input_document_id": response_id,
            "input_document_text": candidate,
            "document_distribution_role": domains[0],
            "audit_document_group_id": str(row["base_id"]),
            "response_id": response_id,
            "base_id": row["base_id"],
            "domain_ids": list(domains),
            "generator_id": str(row.get("generator_id") or ""),
            "candidate_response_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        }
        by_response[response_id] = {
            "response_id": response_id,
            "base_id": str(row["base_id"]),
            "generator_id": str(row.get("generator_id") or ""),
            "candidate_response": candidate,
            "domain_ids": domains,
            "contract": contract,
        }
    records = sorted(
        by_response.values(),
        key=lambda record: (int(record["base_id"]) if record["base_id"].isdigit() else record["base_id"], record["generator_id"], record["response_id"]),
    )
    stats["eligible_unique_a_rows"] = len(records)
    return records, dict(stats)


def load_existing_a_space(
    features_path: Path,
    contract_path: Path,
    records: list[dict[str, Any]],
    *,
    enabled: bool,
    model_id: str,
    model_revision: str,
    hidden_size: int,
    num_feature_layers: int,
) -> dict[str, Any]:
    if not enabled:
        return {
            "features": np.empty((0, num_feature_layers, hidden_size), dtype=np.float16),
            "sample_ids": np.asarray([], dtype=str),
            "id_to_index": {},
            "metadata": {},
        }
    if not features_path.exists() or not contract_path.exists():
        raise FileNotFoundError("The completed 5x6 A-space feature cache and contract are both required")
    full_by_id = {record["response_id"]: record for record in records}
    with np.load(features_path, allow_pickle=True) as payload:
        features = np.asarray(payload["features"], dtype=np.float16)
        ids = np.asarray(payload["sample_ids"]).astype(str)
        metadata = json.loads(str(payload["metadata_json"].item()))
    if features.shape != (len(ids), num_feature_layers, hidden_size) or len(set(ids.tolist())) != len(ids):
        raise ValueError("Completed 5x6 A-space cache has an invalid shape or duplicate ids")
    required_metadata = {
        "model_id": model_id,
        "model_revision": model_revision,
        "feature_scope": "input_document",
        "prompt_template_version": "raw_input_document_v1",
        "pooling": "masked_mean",
    }
    for key, value in required_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(f"Reusable A-space metadata {key!r} is incompatible")
    old_contract = {str(row["sample_id"]): row for row in iter_jsonl(contract_path)}
    if set(old_contract) != set(ids.tolist()):
        raise ValueError("Reusable 5x6 A-space contract is not aligned with its feature cache")
    missing = [response_id for response_id in ids.tolist() if response_id not in full_by_id]
    if missing:
        raise ValueError(f"Reusable A-space id is absent from the full formal scope: {missing[0]}")
    for response_id in ids.tolist():
        old_text = str(old_contract[response_id].get("input_document_text") or old_contract[response_id].get("document_text") or "")
        if old_text != full_by_id[response_id]["candidate_response"]:
            raise ValueError(f"Reusable A-space text differs for {response_id}")
    return {
        "features": features,
        "sample_ids": ids,
        "id_to_index": {value: index for index, value in enumerate(ids.tolist())},
        "metadata": metadata,
    }


def extract_pending(*, pending: list[dict[str, Any]], parts_dir: Path, tokenizer: Any, model: Any, device: torch.device, capture: Any, args: argparse.Namespace) -> dict[str, Any]:
    token_ids: list[list[int]] = []
    for start in tqdm(range(0, len(pending), int(args.tokenization_chunk_size)), desc="Full FLASK A-space tokenization"):
        texts = [record["candidate_response"] for record in pending[start:start + int(args.tokenization_chunk_size)]]
        encoded = tokenizer(texts, add_special_tokens=True, padding=False, truncation=False, return_attention_mask=False)
        token_ids.extend([list(map(int, value)) for value in encoded["input_ids"]])
    if len(token_ids) != len(pending) or any(not value for value in token_ids):
        raise RuntimeError("A-space tokenization produced an incomplete batch")
    order = sorted(range(len(pending)), key=lambda index: (len(token_ids[index]), pending[index]["response_id"]))
    batches = length_batches(order, token_ids, args=args)
    next_part = next_part_number(parts_dir)
    written = 0
    effective_batches: list[int] = []
    progress = tqdm(total=len(pending), desc="Full FLASK A-space", unit="answer")
    try:
        while batches:
            indices = batches.pop(0)
            try:
                bounded_ids = [head_tail(token_ids[index], max_length=int(args.max_length)) for index in indices]
                encoded = tokenizer.pad(
                    [{"input_ids": value} for value in bounded_ids], padding=True,
                    pad_to_multiple_of=int(args.pad_to_multiple_of), return_attention_mask=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                with torch.inference_mode():
                    capture.begin(encoded["attention_mask"])
                    outputs = getattr(model, "model", model)(
                        **encoded, output_hidden_states=False, use_cache=False, return_dict=True,
                    )
                    features = capture.features()
                part = parts_dir / f"part-{next_part:06d}.npz"
                temporary = part.with_name(f"{part.stem}.tmp.npz")
                np.savez(
                    temporary,
                    features=features.cpu().numpy().astype(np.float16),
                    sample_ids=np.asarray([pending[index]["response_id"] for index in indices]),
                    query_ids=np.asarray([pending[index]["base_id"] for index in indices]),
                    base_ids=np.asarray([pending[index]["base_id"] for index in indices]),
                    domain_ids_json=np.asarray([json.dumps(pending[index]["domain_ids"], ensure_ascii=False) for index in indices]),
                    active_tokens=np.asarray(int(encoded["attention_mask"].sum().item()), dtype=np.int64),
                    padded_tokens=np.asarray(int(encoded["attention_mask"].numel()), dtype=np.int64),
                    truncated_records=np.asarray(sum(len(token_ids[index]) > len(value) for index, value in zip(indices, bounded_ids, strict=True)), dtype=np.int64),
                )
                temporary.replace(part)
                next_part += 1
                written += len(indices)
                effective_batches.append(len(indices))
                progress.update(len(indices))
                del outputs, features, encoded
                capture.clear()
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                recoverable = isinstance(exc, torch.OutOfMemoryError) or any(
                    value in str(exc).lower()
                    for value in ("out of memory", "canuse32bitindexmath", "triton error [cuda]: invalid argument")
                )
                capture.clear()
                torch.cuda.empty_cache()
                if not recoverable or len(indices) == 1:
                    raise
                midpoint = len(indices) // 2
                batches.insert(0, indices[midpoint:])
                batches.insert(0, indices[:midpoint])
                print(f"A-space batch split {len(indices)} -> {midpoint}+{len(indices) - midpoint}", flush=True)
    finally:
        progress.close()
    return {
        "new_qwen_unique_a_rows": written,
        "batch_count": len(effective_batches),
        "effective_batch_sizes": sorted(set(effective_batches)),
    }


def length_batches(order: list[int], token_ids: list[list[int]], *, args: argparse.Namespace) -> list[list[int]]:
    batches: list[list[int]] = []
    current: list[int] = []
    width = 0
    for index in order:
        bounded = min(len(token_ids[index]), int(args.max_length))
        rounded = min(int(args.max_length), ((bounded + int(args.pad_to_multiple_of) - 1) // int(args.pad_to_multiple_of)) * int(args.pad_to_multiple_of))
        candidate_width = max(width, rounded)
        if current and (len(current) + 1 > int(args.batch_size) or candidate_width * (len(current) + 1) > int(args.max_batch_tokens)):
            batches.append(current)
            current = []
            width = 0
        current.append(index)
        width = max(width, rounded)
    if current:
        batches.append(current)
    return batches


def merge_features(
    *,
    records: list[dict[str, Any]],
    existing: dict[str, Any],
    parts_dir: Path,
    output: Path,
    settings: dict[str, Any],
    num_feature_layers: int,
    hidden_size: int,
) -> dict[str, Any]:
    new_features: list[np.ndarray] = []
    new_ids: list[str] = []
    active_tokens = padded_tokens = truncated_records = 0
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            new_features.append(np.asarray(payload["features"], dtype=np.float16))
            new_ids.extend(np.asarray(payload["sample_ids"]).astype(str).tolist())
            active_tokens += int(payload["active_tokens"])
            padded_tokens += int(payload["padded_tokens"])
            truncated_records += int(payload["truncated_records"])
    if len(new_ids) != len(set(new_ids)):
        raise ValueError("New A-space feature parts contain duplicate ids")
    new_matrix = (
        np.concatenate(new_features, axis=0)
        if new_features
        else np.empty((0, num_feature_layers, hidden_size), dtype=np.float16)
    )
    if new_matrix.shape[1:] != (num_feature_layers, hidden_size):
        raise RuntimeError(f"Unexpected A-space feature shape {new_matrix.shape}")
    new_index = {value: index for index, value in enumerate(new_ids)}
    expected_ids = [record["response_id"] for record in records]
    expected = set(expected_ids)
    combined = set(existing["id_to_index"]).union(new_index)
    if combined != expected:
        raise RuntimeError(f"A-space feature coverage mismatch: missing={len(expected - combined)}, extra={len(combined - expected)}")
    features: list[np.ndarray] = []
    for response_id in expected_ids:
        if response_id in existing["id_to_index"]:
            features.append(existing["features"][existing["id_to_index"][response_id]])
        else:
            features.append(new_matrix[new_index[response_id]])
    matrix = np.stack(features).astype(np.float16)
    sample_ids = np.asarray(expected_ids)
    query_ids = np.asarray([record["base_id"] for record in records])
    metadata = {
        **settings,
        "num_records": len(records),
        "num_input_documents": len(records),
        "shape": list(matrix.shape),
        "reused_5x6_unique_a_rows": len(existing["id_to_index"]),
        "new_qwen_unique_a_rows": len(new_ids),
        "active_tokens_new_qwen": active_tokens,
        "padded_tokens_new_qwen": padded_tokens,
        "truncated_records_new_qwen": truncated_records,
        "feature_storage_dtype": "float16",
        "reuse_policy": "identical_response_id_and_raw_candidate_response_only",
    }
    np.savez_compressed(
        output, features=matrix, sample_ids=sample_ids, labels=np.asarray([None] * len(records), dtype=object),
        query_ids=query_ids, audit_document_group_ids=query_ids, input_document_ids=sample_ids,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return {
        "features": matrix,
        "sample_ids": sample_ids,
        "id_to_index": {value: index for index, value in enumerate(expected_ids)},
        "metadata": metadata,
    }


def write_domain_indexes(*, records: list[dict[str, Any]], merged: dict[str, Any], output_dir: Path, settings: dict[str, Any]) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    memberships = 0
    for domain in DOMAINS:
        indices = np.asarray([index for index, record in enumerate(records) if domain in record["domain_ids"]], dtype=np.int64)
        if not len(indices):
            raise ValueError(f"Full A-space domain unexpectedly has no answers: {domain}")
        np.savez_compressed(
            output_dir / f"{slug(domain)}.npz",
            indices=indices, sample_ids=merged["sample_ids"][indices],
            query_ids=np.asarray([records[index]["base_id"] for index in indices]),
            domain_ids=np.asarray([domain] * len(indices)),
            metadata_json=np.asarray(json.dumps({**settings, "domain_id": domain, "num_records": len(indices), "feature_cache": "../a_space_hidden_states.npz"}, ensure_ascii=False)),
        )
        memberships += len(indices)
    return {"domain_index_files": len(DOMAINS), "domain_a_memberships": memberships}


def resolve_layers(num_states: int, requested: list[int]) -> list[int]:
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


class MaskedMeanPromptLayerCapture:
    def __init__(
        self,
        base_model: torch.nn.Module,
        state_indices: list[int],
        *,
        num_layers: int,
        hidden_size: int,
    ) -> None:
        layers = getattr(base_model, "layers", None)
        if layers is None or len(layers) != int(num_layers):
            raise ValueError("Qwen decoder layers were not found for A-space capture")
        self.state_indices = list(state_indices)
        self.num_layers = int(num_layers)
        self.hidden_size = int(hidden_size)
        self._attention_mask: torch.Tensor | None = None
        self._expected_shape: tuple[int, int] | None = None
        self._features: dict[int, torch.Tensor] = {}
        self._handles: list[Any] = []
        for state_index in self.state_indices:
            if int(state_index) == self.num_layers:
                module = getattr(base_model, "norm", None)
                if module is None:
                    raise ValueError("Qwen final norm was not found for A-space capture")
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
            output: Any,
        ) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if not isinstance(hidden, torch.Tensor):
                raise RuntimeError(f"Decoder output for A-space state {state_index} is not a tensor")
            if self._expected_shape is None or self._attention_mask is None:
                return
            expected_batch, expected_length = self._expected_shape
            if tuple(hidden.shape[:2]) != (expected_batch, expected_length):
                return
            if state_index in self._features:
                raise RuntimeError(f"A-space state {state_index} was captured more than once")
            weights = self._attention_mask.to(dtype=torch.float32).unsqueeze(-1)
            pooled = (hidden.float() * weights).sum(dim=1)
            pooled = pooled / weights.sum(dim=1).clamp_min(1.0)
            if tuple(pooled.shape) != (expected_batch, self.hidden_size):
                raise RuntimeError(
                    f"A-space pooled state has shape {tuple(pooled.shape)}, expected "
                    f"({expected_batch}, {self.hidden_size})"
                )
            if not torch.isfinite(pooled).all():
                raise RuntimeError(f"A-space state {state_index} contains a non-finite value")
            self._features[state_index] = pooled

        return capture

    def features(self) -> torch.Tensor:
        missing = [index for index in self.state_indices if index not in self._features]
        if missing:
            raise RuntimeError(f"Qwen forward did not expose A-space prompt states: {missing}")
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


def load_part_ids(parts_dir: Path) -> set[str]:
    ids: set[str] = set()
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=False) as payload:
            values = np.asarray(payload["sample_ids"]).astype(str).tolist()
        overlap = ids.intersection(values)
        if overlap:
            raise ValueError(f"Duplicate A-space id in new parts: {next(iter(overlap))}")
        ids.update(values)
    return ids


def next_part_number(parts_dir: Path) -> int:
    numbers = [int(match.group(1)) for path in parts_dir.glob("part-*.npz") if (match := re.fullmatch(r"part-(\d+)\.npz", path.name))]
    return max(numbers, default=-1) + 1


def head_tail(ids: list[int], *, max_length: int) -> list[int]:
    if len(ids) <= max_length:
        return list(ids)
    head = (max_length + 1) // 2
    return [*ids[:head], *ids[-(max_length - head):]]


def integer_score(value: Any) -> int | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    rounded = int(round(score))
    return rounded if abs(score - rounded) < 1e-8 and 1 <= rounded <= 5 else None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _direct_judge_module():
    path = ROOT / "scripts/llm_judge_ood/49_run_flask_minimal_direct_judge.py"
    spec = importlib.util.spec_from_file_location("flask_direct_judge_for_a_space", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import Qwen layer capture helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
