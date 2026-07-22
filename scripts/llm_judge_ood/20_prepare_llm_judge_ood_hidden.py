#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.shared.feature_store import record_fingerprint
from src.llm_judge_ood.shared.schema import JudgeRecord, limit_input_document_records, load_judge_records
from src.models.extract_hidden import (
    QWEN3_5_4B_HIDDEN_SIZE,
    QWEN3_5_4B_MODEL_ID,
    QWEN3_5_4B_NUM_LAYERS,
    QWEN3_5_4B_REVISION,
    QWEN3_5_TEXT_MODEL_TYPE,
    local_git_revision,
    load_qwen_model,
    qwen_text_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen Qwen3.5-4B hidden states for document OOD or Judge inputs."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--model-path", default=QWEN3_5_4B_MODEL_ID)
    parser.add_argument(
        "--model-id",
        default=QWEN3_5_4B_MODEL_ID,
        choices=[QWEN3_5_4B_MODEL_ID],
        help="Canonical Hugging Face identity recorded in the cache metadata.",
    )
    parser.add_argument(
        "--revision",
        default=QWEN3_5_4B_REVISION,
        choices=[QWEN3_5_4B_REVISION],
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--parts-dir", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=[-10, -1])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument(
        "--truncation-strategy",
        choices=["right", "head_tail"],
        default="right",
        help="Token-window policy. head_tail preserves both prompt context and response tail.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--hidden-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--attn-implementation", choices=["eager", "sdpa", "flash_attention_2"], default="sdpa")
    parser.add_argument(
        "--feature-scope",
        choices=["input_document", "judge_input"],
        default="input_document",
        help=(
            "A=input_document reads only raw input_document_text; "
            "B=judge_input reads only frozen, label-free judge_input_text."
        ),
    )
    parser.add_argument(
        "--pooling",
        choices=["masked_mean"],
        default="masked_mean",
        help="Mean over tokenizer attention-mask tokens; padding is excluded.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-input-documents", type=int, default=0)
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Capture only the requested transformer layers with forward hooks, "
            "pre-tokenize once, and use length-bucketed dynamic batches."
        ),
    )
    parser.add_argument(
        "--max-batch-tokens",
        type=int,
        default=0,
        help="Fast-mode cap for padded tokens per batch; 0 disables the token cap.",
    )
    parser.add_argument(
        "--tokenization-batch-size",
        type=int,
        default=256,
        help="Number of documents per CPU pre-tokenization chunk in fast mode.",
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=1,
        help="Round fast-mode padded sequence lengths to this multiple to reuse GPU kernels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
    return (hidden.float() * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def resolve_layers(num_states: int, requested: list[int]) -> list[int]:
    resolved: list[int] = []
    for value in requested:
        index = value if value >= 0 else num_states + value
        if index <= 0 or index >= num_states:
            raise ValueError(f"Layer {value} resolves to {index}; valid transformer outputs are 1..{num_states - 1}")
        if index in resolved:
            raise ValueError(f"Duplicate resolved layer: {index}")
        resolved.append(index)
    return resolved


def main() -> None:
    args = parse_args()
    if (
        args.batch_size < 1
        or args.max_length < 1
        or args.tokenization_batch_size < 1
        or args.pad_to_multiple_of < 1
    ):
        raise ValueError(
            "batch-size, max-length, tokenization-batch-size, and pad-to-multiple-of must be positive"
        )
    if args.max_length % args.pad_to_multiple_of != 0:
        raise ValueError("max-length must be divisible by pad-to-multiple-of")
    if args.max_batch_tokens < 0:
        raise ValueError("max-batch-tokens must be non-negative")
    if args.pooling != "masked_mean":
        raise ValueError("The final protocol requires pooling='masked_mean'")
    pooling = "masked_mean"
    records = load_judge_records(args.input)
    if args.max_input_documents > 0:
        records = limit_input_document_records(records, args.max_input_documents, seed=args.seed)
    records = _records_for_feature_scope(records, feature_scope=str(args.feature_scope))
    template_identity = _prompt_template_identity(
        records, feature_scope=str(args.feature_scope)
    )
    output = Path(args.output)
    parts_dir = Path(args.parts_dir) if args.parts_dir else output.with_name(f"{output.stem}.parts")
    dataset_hash = record_fingerprint(records, feature_scope=args.feature_scope)
    model_identity = _model_identity_evidence(
        model_path=str(args.model_path),
        model_id=str(args.model_id),
        revision=str(args.revision),
    )
    settings = {
        "dataset_fingerprint": dataset_hash,
        "num_records": len(records),
        "num_input_documents": len({record.input_document_id for record in records}),
        "model_id": str(args.model_id),
        "model_source": str(args.model_path),
        "model_identity_evidence": model_identity,
        "model_revision_requested": str(args.revision),
        "model_type": QWEN3_5_TEXT_MODEL_TYPE,
        "num_model_layers": QWEN3_5_4B_NUM_LAYERS,
        "model_hidden_size": QWEN3_5_4B_HIDDEN_SIZE,
        "hidden_state_count": QWEN3_5_4B_NUM_LAYERS + 1,
        "embedding_state_included": True,
        "layers_requested": list(args.layers),
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "truncation_strategy": str(args.truncation_strategy),
        "torch_dtype": args.torch_dtype,
        "hidden_dtype": args.hidden_dtype,
        "attn_implementation": args.attn_implementation,
        "feature_scope": str(args.feature_scope),
        "pooling": pooling,
        "pooling_scope": str(args.feature_scope),
        "pooling_formula": "sum(hidden_state * attention_mask) / sum(attention_mask)",
        "pooling_mask_source": "tokenizer_attention_mask",
        "pooling_excludes_padding": True,
        "seed": args.seed,
        "prompt_template_version": template_identity["version"],
        "labels_in_prompt": False,
    }
    if template_identity["sha256"] is not None:
        settings["prompt_template_sha256"] = template_identity["sha256"]
    if args.fast:
        settings.update(
            {
                "extraction_engine": "transformers_selected_layer_hooks_v1",
                "batching_policy": "length_bucketed_dynamic_v1",
                "maximum_batch_documents": int(args.batch_size),
                "max_batch_tokens": int(args.max_batch_tokens),
                "tokenization_batch_size": int(args.tokenization_batch_size),
                "pad_to_multiple_of": int(args.pad_to_multiple_of),
                "tokenization_passes": 1,
            }
        )
    if output.exists() and not args.overwrite:
        print(json.dumps({"output": str(output), "reused": True}, indent=2))
        return
    if args.overwrite:
        if output.exists():
            output.unlink()
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
    prepare_parts(parts_dir, settings)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer, model, device = load_qwen_model(
        args.model_path,
        revision=args.revision,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        tf32=True,
        local_files_only=bool(args.local_files_only),
    )
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Frozen extractor invariant failed")
    text_config = qwen_text_config(model)
    if str(args.model_id) == QWEN3_5_4B_MODEL_ID:
        if str(getattr(text_config, "model_type", "")) != QWEN3_5_TEXT_MODEL_TYPE:
            raise ValueError(
                "Qwen/Qwen3.5-4B must load a model_type='qwen3_5_text' text backbone"
            )
        if int(getattr(text_config, "hidden_size", 0)) != QWEN3_5_4B_HIDDEN_SIZE:
            raise ValueError("Qwen/Qwen3.5-4B must expose the official hidden_size=2560")
        if int(getattr(text_config, "num_hidden_layers", 0)) != QWEN3_5_4B_NUM_LAYERS:
            raise ValueError("Qwen/Qwen3.5-4B must expose the official 32 transformer layers")
    base_model = getattr(model, "model", model)
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    started = time.perf_counter()
    hidden_states_validated = False
    selected_layers = resolve_layers(
        int(getattr(text_config, "num_hidden_layers", 0)) + 1,
        list(args.layers),
    )
    tokenized_input_ids: list[list[int]] | None = None
    token_lengths: list[int] | None = None
    capture: SelectedLayerCapture | None = None
    if args.fast:
        tokenized_input_ids, token_lengths = _pretokenize_records(
            tokenizer,
            records,
            chunk_size=int(args.tokenization_batch_size),
            feature_scope=str(args.feature_scope),
        )
        batch_plan = _length_bucketed_batch_plan(
            token_lengths,
            maximum_documents=int(args.batch_size),
            maximum_tokens=int(args.max_batch_tokens),
            max_length=int(args.max_length),
            pad_to_multiple_of=int(args.pad_to_multiple_of),
        )
        capture = SelectedLayerCapture(base_model, selected_layers)
    else:
        batch_plan = [
            list(range(start, min(start + int(args.batch_size), len(records))))
            for start in range(0, len(records), int(args.batch_size))
        ]

    for batch_number, batch_indices in enumerate(
        tqdm(batch_plan, desc="Qwen hidden extraction")
    ):
        batch = [records[index] for index in batch_indices]
        part_number = batch_number if args.fast else batch_indices[0]
        part = parts_dir / f"part-{part_number:06d}.npz"
        if part.exists():
            validate_part(part, batch, batch_indices)
            continue
        prompt_texts = [
            _feature_text(record, feature_scope=str(args.feature_scope)) for record in batch
        ]
        if tokenized_input_ids is not None and token_lengths is not None:
            untruncated_lengths = [token_lengths[index] for index in batch_indices]
            encoded = tokenizer.pad(
                [
                    {
                        "input_ids": _truncate_input_ids(
                            tokenized_input_ids[index],
                            max_length=int(args.max_length),
                            strategy=str(args.truncation_strategy),
                        )
                    }
                    for index in batch_indices
                ],
                padding=True,
                pad_to_multiple_of=(
                    int(args.pad_to_multiple_of)
                    if int(args.pad_to_multiple_of) > 1
                    else None
                ),
                return_attention_mask=True,
                return_tensors="pt",
            )
        else:
            untruncated_lengths = [
                len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
                for text in prompt_texts
            ]
            unpadded = tokenizer(
                prompt_texts,
                add_special_tokens=True,
                padding=False,
                truncation=False,
                return_attention_mask=False,
            )["input_ids"]
            encoded = tokenizer.pad(
                [
                    {
                        "input_ids": _truncate_input_ids(
                            list(map(int, input_ids)),
                            max_length=int(args.max_length),
                            strategy=str(args.truncation_strategy),
                        )
                    }
                    for input_ids in unpadded
                ],
                padding=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
        truncated_records = [length > int(args.max_length) for length in untruncated_lengths]
        attention_mask = encoded["attention_mask"]
        pooling_mask = attention_mask.to(dtype=torch.bool)
        model_inputs = {key: value.to(device) for key, value in encoded.items() if key in {"input_ids", "attention_mask"}}
        pooling_mask = pooling_mask.to(device)
        with torch.inference_mode():
            if capture is not None:
                capture.clear()
                outputs = base_model(
                    **model_inputs,
                    output_hidden_states=False,
                    use_cache=False,
                    return_dict=True,
                )
                selected_hidden = capture.selected(
                    batch_size=len(batch),
                    sequence_length=int(model_inputs["input_ids"].shape[1]),
                )
                if not hidden_states_validated:
                    for state_index, hidden in zip(
                        selected_layers,
                        selected_hidden,
                        strict=True,
                    ):
                        if not torch.isfinite(hidden).all():
                            raise RuntimeError(
                                f"Captured hidden state {state_index} contains a non-finite value"
                            )
                    hidden_states_validated = True
            else:
                outputs = base_model(
                    **model_inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                if not hidden_states_validated:
                    _validate_hidden_states(
                        outputs.hidden_states,
                        batch_size=len(batch),
                        sequence_length=int(model_inputs["input_ids"].shape[1]),
                    )
                    hidden_states_validated = True
                selected_hidden = [outputs.hidden_states[index] for index in selected_layers]
            features = torch.stack(
                [masked_mean(hidden, pooling_mask) for hidden in selected_hidden], dim=1
            )
        active_tokens = int(model_inputs["attention_mask"].sum().item())
        padded_tokens = int(model_inputs["attention_mask"].numel())
        pooled_tokens = int(pooling_mask.sum().item())
        max_unpadded_tokens = int(attention_mask.sum(dim=1).max().item())
        if max_unpadded_tokens > int(args.max_length):
            raise RuntimeError("Length-bounded prompt exceeded max_length")
        np.savez(
            part,
            features=features.cpu().numpy().astype(args.hidden_dtype),
            indices=np.asarray(batch_indices, dtype=np.int64),
            sample_ids=np.asarray([record.sample_id for record in batch]),
            labels=np.asarray([record.label for record in batch], dtype=object),
            query_ids=np.asarray([record.query_id for record in batch]),
            audit_document_group_ids=np.asarray([record.audit_document_group_id for record in batch]),
            input_document_ids=np.asarray([record.input_document_id for record in batch]),
            active_tokens=np.asarray(active_tokens),
            padded_tokens=np.asarray(padded_tokens),
            pooled_tokens=np.asarray(pooled_tokens),
            truncated_records=np.asarray(sum(truncated_records)),
            max_unpadded_tokens=np.asarray(max_unpadded_tokens),
        )
        del outputs, features, encoded, model_inputs

    if capture is not None:
        capture.close()

    if device.type == "cuda":
        torch.cuda.synchronize()
    metadata = merge_parts(
        parts_dir,
        output,
        settings={
            **settings,
            "artifact_type": "llm_judge_ood_frozen_qwen_hidden_features",
            "model_name": str(getattr(model.config, "_name_or_path", args.model_path)),
            "model_id": str(args.model_id),
            "model_source": str(args.model_path),
            "model_revision": str(
                getattr(model.config, "_commit_hash", None)
                or model_identity.get("revision")
                or args.revision
            ),
            "model_type": str(getattr(text_config, "model_type", "unknown")),
            "num_model_layers": int(getattr(text_config, "num_hidden_layers", 0)),
            "model_hidden_size": int(getattr(text_config, "hidden_size", 0)),
            "hidden_state_count": int(getattr(text_config, "num_hidden_layers", 0)) + 1,
            "embedding_state_included": True,
            "layers_resolved": selected_layers,
            "model_class": type(model).__name__,
            "backbone_class": type(base_model).__name__,
            "feature_scope": str(args.feature_scope),
            "pooling": pooling,
            "pooling_formula": "sum(hidden_state * attention_mask) / sum(attention_mask)",
            "pooling_mask_source": "tokenizer_attention_mask",
            "pooling_excludes_padding": True,
            "model_eval": True,
            "requires_grad": False,
            "backbone_frozen": True,
            "labels_in_prompt": False,
            "device": str(device),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        },
    )
    write_json(output.with_suffix(".metadata.json"), metadata)
    write_json(parts_dir / "manifest.json", {**settings, "complete": True})
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def _records_for_feature_scope(
    records: list[JudgeRecord],
    *,
    feature_scope: str,
) -> list[JudgeRecord]:
    """Apply B row identity or A document identity before tokenization."""

    if feature_scope == "judge_input":
        # B/Judge space is one representation per Judge row, aligned by sample_id.
        seen: set[str] = set()
        for record in records:
            if record.sample_id in seen:
                raise ValueError(
                    f"Judge-input extraction requires unique sample_id values; duplicate={record.sample_id!r}"
                )
            seen.add(record.sample_id)
        return records
    if feature_scope != "input_document":
        raise ValueError("feature_scope must be 'input_document' or 'judge_input'")
    # A space is one representation per raw document, aligned by input_document_id.
    unique: dict[str, JudgeRecord] = {}
    for record in records:
        previous = unique.get(record.input_document_id)
        if previous is not None and previous.input_document_text != record.input_document_text:
            raise ValueError(
                f"Input document {record.input_document_id!r} has inconsistent text"
            )
        unique.setdefault(record.input_document_id, record)
    return list(unique.values())


def _feature_text(record: JudgeRecord, *, feature_scope: str) -> str:
    if feature_scope == "input_document":
        # A: raw monitored document, never the Judge task template.
        return str(record.input_document_text)
    if feature_scope == "judge_input":
        # B: frozen Judge task template, never the bare A-space field.
        return str(record.judge_input_text)
    raise ValueError("feature_scope must be 'input_document' or 'judge_input'")


def _prompt_template_version(feature_scope: str) -> str:
    if feature_scope == "input_document":
        return "raw_input_document_v1"
    if feature_scope == "judge_input":
        return "judge_input_query_document_candidate_v1"
    raise ValueError("feature_scope must be 'input_document' or 'judge_input'")


def _prompt_template_identity(
    records: list[JudgeRecord], *, feature_scope: str
) -> dict[str, str | None]:
    """Require a single prepared Judge template when text includes task context."""

    if feature_scope == "input_document":
        return {"version": _prompt_template_version(feature_scope), "sha256": None}
    if feature_scope != "judge_input":
        raise ValueError("feature_scope must be 'input_document' or 'judge_input'")
    pairs = {
        (
            str(record.metadata.get("prompt_template_version") or ""),
            str(record.metadata.get("prompt_template_sha256") or ""),
        )
        for record in records
    }
    if pairs == {("", "")}:
        return {"version": _prompt_template_version(feature_scope), "sha256": None}
    if len(pairs) != 1:
        raise ValueError(f"Judge-input extraction requires one template version/hash pair, got {sorted(pairs)}")
    version, digest = next(iter(pairs))
    if not version or not digest:
        raise ValueError(
            "Judge-input extraction requires both prompt_template_version and prompt_template_sha256"
        )
    return {"version": version, "sha256": digest}


def _validate_hidden_states(
    hidden_states: tuple[torch.Tensor, ...],
    *,
    batch_size: int,
    sequence_length: int,
) -> None:
    expected_count = QWEN3_5_4B_NUM_LAYERS + 1
    if len(hidden_states) != expected_count:
        raise RuntimeError(
            f"Expected embedding plus 32 transformer hidden states, got {len(hidden_states)}"
        )
    expected_shape = (batch_size, sequence_length, QWEN3_5_4B_HIDDEN_SIZE)
    for index, hidden in enumerate(hidden_states):
        if tuple(hidden.shape) != expected_shape:
            raise RuntimeError(
                f"Hidden state {index} has shape {tuple(hidden.shape)}, expected {expected_shape}"
            )
        if not torch.isfinite(hidden).all():
            raise RuntimeError(f"Hidden state {index} contains a non-finite value")


class SelectedLayerCapture:
    """Capture decoder outputs matching Transformers hidden-state indices."""

    def __init__(self, base_model: torch.nn.Module, state_indices: list[int]) -> None:
        layers = getattr(base_model, "layers", None)
        if layers is None or len(layers) != QWEN3_5_4B_NUM_LAYERS:
            raise ValueError("Fast extraction requires the Qwen text backbone decoder layers")
        self.state_indices = list(state_indices)
        self.values: dict[int, torch.Tensor] = {}
        self.handles = []
        for state_index in self.state_indices:
            if int(state_index) == QWEN3_5_4B_NUM_LAYERS:
                module = getattr(base_model, "norm", None)
                if module is None:
                    raise ValueError("Fast extraction requires the Qwen final normalization layer")
            else:
                module = layers[int(state_index) - 1]
            self.handles.append(
                module.register_forward_hook(self._hook_for_state(int(state_index)))
            )

    def _hook_for_state(self, state_index: int):
        def capture(
            _module: torch.nn.Module,
            _inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(
                    f"Decoder layer for hidden state {state_index} returned a non-tensor"
                )
            self.values[state_index] = output

        return capture

    def clear(self) -> None:
        self.values.clear()

    def selected(self, *, batch_size: int, sequence_length: int) -> list[torch.Tensor]:
        missing = [index for index in self.state_indices if index not in self.values]
        if missing:
            raise RuntimeError(f"Fast extraction did not capture hidden states {missing}")
        expected_shape = (int(batch_size), int(sequence_length), QWEN3_5_4B_HIDDEN_SIZE)
        output = [self.values[index] for index in self.state_indices]
        for state_index, hidden in zip(self.state_indices, output, strict=True):
            if tuple(hidden.shape) != expected_shape:
                raise RuntimeError(
                    f"Captured hidden state {state_index} has shape {tuple(hidden.shape)}, "
                    f"expected {expected_shape}"
                )
        return output

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.values.clear()


def _pretokenize_records(
    tokenizer: object,
    records: list[JudgeRecord],
    *,
    chunk_size: int,
    feature_scope: str,
) -> tuple[list[list[int]], list[int]]:
    tokenized: list[list[int]] = []
    for start in tqdm(
        range(0, len(records), int(chunk_size)),
        desc="Qwen pre-tokenization",
    ):
        texts = [
            _feature_text(record, feature_scope=feature_scope)
            for record in records[start : start + int(chunk_size)]
        ]
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )
        rows = encoded["input_ids"]
        if len(rows) != len(texts):
            raise RuntimeError("Fast tokenizer returned an unexpected number of rows")
        tokenized.extend([list(map(int, row)) for row in rows])
    if len(tokenized) != len(records) or any(not row for row in tokenized):
        raise RuntimeError("Fast pre-tokenization produced incomplete or empty rows")
    return tokenized, [len(row) for row in tokenized]


def _truncate_input_ids(
    input_ids: list[int],
    *,
    max_length: int,
    strategy: str,
) -> list[int]:
    if len(input_ids) <= int(max_length):
        return input_ids
    if strategy == "right":
        return input_ids[: int(max_length)]
    if strategy != "head_tail":
        raise ValueError(f"Unsupported truncation strategy: {strategy!r}")
    head_length = (int(max_length) + 1) // 2
    tail_length = int(max_length) - head_length
    if not tail_length:
        return input_ids[:head_length]
    return [*input_ids[:head_length], *input_ids[-tail_length:]]


def _length_bucketed_batch_plan(
    token_lengths: list[int],
    *,
    maximum_documents: int,
    maximum_tokens: int,
    max_length: int,
    pad_to_multiple_of: int = 1,
) -> list[list[int]]:
    def padded_length(length: int) -> int:
        bounded = min(int(length), int(max_length))
        multiple = int(pad_to_multiple_of)
        return min(int(max_length), ((bounded + multiple - 1) // multiple) * multiple)

    ranked = sorted(
        range(len(token_lengths)),
        key=lambda index: (padded_length(token_lengths[index]), int(index)),
    )
    batches: list[list[int]] = []
    current: list[int] = []
    current_max_length = 0
    for index in ranked:
        bounded_length = padded_length(token_lengths[index])
        candidate_max_length = max(current_max_length, bounded_length)
        candidate_documents = len(current) + 1
        exceeds_documents = candidate_documents > int(maximum_documents)
        exceeds_tokens = (
            int(maximum_tokens) > 0
            and candidate_max_length * candidate_documents > int(maximum_tokens)
        )
        if current and (exceeds_documents or exceeds_tokens):
            batches.append(current)
            current = []
            current_max_length = 0
        current.append(int(index))
        current_max_length = max(current_max_length, bounded_length)
    if current:
        batches.append(current)
    if sorted(index for batch in batches for index in batch) != list(range(len(token_lengths))):
        raise RuntimeError("Length-bucketed batch plan is incomplete or duplicated")
    return batches


def _model_identity_evidence(
    *,
    model_path: str,
    model_id: str,
    revision: str = QWEN3_5_4B_REVISION,
) -> dict[str, object]:
    source = Path(model_path).expanduser()
    if not source.exists():
        if str(model_path) != str(model_id):
            raise ValueError(
                "A remote model source must equal the canonical --model-id; "
                "use a verified local snapshot when the load path differs"
            )
        if str(model_id) == QWEN3_5_4B_MODEL_ID and str(revision) != QWEN3_5_4B_REVISION:
            raise ValueError(
                f"The final Qwen3.5-4B protocol is pinned to revision {QWEN3_5_4B_REVISION}"
            )
        return {
            "kind": "huggingface_repo_id",
            "value": str(model_id),
            "repo_id": str(model_id),
            "revision": str(revision),
        }
    if str(model_id) != QWEN3_5_4B_MODEL_ID:
        return {"kind": "declared_local_snapshot", "value": str(source.resolve())}
    readme = source / "README.md"
    if not readme.exists():
        raise ValueError("Local Qwen/Qwen3.5-4B snapshot is missing README.md identity evidence")
    marker = "https://huggingface.co/Qwen/Qwen3.5-4B/blob/"
    if marker not in readme.read_text(encoding="utf-8", errors="replace"):
        raise ValueError(
            "Local model is not the exact Qwen/Qwen3.5-4B snapshot; "
            "Qwen3-4B-Instruct-2507 is a different checkpoint"
        )
    evidence: dict[str, object] = {
        "kind": "verified_local_huggingface_readme",
        "value": str(source.resolve()),
        "repo_id": "Qwen/Qwen3.5-4B",
    }
    download_metadata = (
        source / ".cache" / "huggingface" / "download" / "config.json.metadata"
    )
    if download_metadata.exists():
        lines = download_metadata.read_text(encoding="utf-8").splitlines()
        if lines and len(lines[0]) == 40:
            evidence["revision"] = lines[0]
    git_revision = local_git_revision(source)
    if git_revision is not None:
        evidence.update(
            {
                "kind": "verified_local_git_snapshot",
                "revision": git_revision,
            }
        )
    resolved_revision = str(evidence.get("revision", ""))
    if resolved_revision != QWEN3_5_4B_REVISION:
        raise ValueError(
            "Local Qwen/Qwen3.5-4B snapshot revision is missing or does not match "
            f"the pinned revision {QWEN3_5_4B_REVISION}"
        )
    if str(revision) != QWEN3_5_4B_REVISION:
        raise ValueError(
            f"The final Qwen3.5-4B protocol is pinned to revision {QWEN3_5_4B_REVISION}"
        )
    return evidence


def prepare_parts(parts_dir: Path, settings: dict[str, object]) -> None:
    manifest = parts_dir / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        for key, value in settings.items():
            if existing.get(key) != value:
                raise ValueError(f"Parts setting {key!r} changed; use --overwrite")
        return
    if parts_dir.exists() and any(parts_dir.iterdir()):
        raise ValueError(f"Non-empty parts directory has no manifest: {parts_dir}")
    parts_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest, {**settings, "complete": False})


def validate_part(path: Path, records: list[JudgeRecord], indices: list[int]) -> None:
    with np.load(path, allow_pickle=True) as payload:
        ids = np.asarray(payload["sample_ids"]).astype(str).tolist()
        stored_indices = np.asarray(payload["indices"]).astype(int).tolist()
    if ids != [record.sample_id for record in records] or stored_indices != list(indices):
        raise ValueError(f"Existing part does not match current input: {path}")


def merge_parts(parts_dir: Path, output: Path, *, settings: dict[str, object]) -> dict[str, object]:
    arrays: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "features",
            "indices",
            "sample_ids",
            "labels",
            "query_ids",
            "audit_document_group_ids",
            "input_document_ids",
        )
    }
    active_tokens = 0
    padded_tokens = 0
    pooled_tokens = 0
    truncated_records = 0
    max_unpadded_tokens = 0
    for path in sorted(parts_dir.glob("part-*.npz")):
        with np.load(path, allow_pickle=True) as payload:
            for key in arrays:
                source_key = key
                if key == "audit_document_group_ids" and key not in payload.files:
                    source_key = "document_group_ids"
                if source_key not in payload.files:
                    raise ValueError(f"Hidden feature part is missing {key!r}: {path}")
                arrays[key].append(np.asarray(payload[source_key]))
            active_tokens += int(payload["active_tokens"])
            padded_tokens += int(payload["padded_tokens"])
            pooled_tokens += int(payload["pooled_tokens"]) if "pooled_tokens" in payload.files else int(payload["active_tokens"])
            truncated_records += int(
                payload["truncated_records"]
                if "truncated_records" in payload.files
                else payload["input_document_truncated_records"]
            )
            max_unpadded_tokens = max(
                max_unpadded_tokens,
                int(payload["max_unpadded_tokens"]) if "max_unpadded_tokens" in payload.files else 0,
            )
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    order = np.argsort(merged["indices"], kind="stable")
    if not np.array_equal(merged["indices"][order], np.arange(len(order))):
        raise ValueError("Feature parts are incomplete or duplicated")
    output.parent.mkdir(parents=True, exist_ok=True)
    feature_dtype = np.float16 if str(settings.get("hidden_dtype", "float16")) == "float16" else np.float32
    metadata = {
        **settings,
        "shape": list(merged["features"].shape),
        "active_tokens": active_tokens,
        "padded_tokens": padded_tokens,
        "pooled_tokens": pooled_tokens,
        "truncated_records": truncated_records,
        "truncated_input_document_records": (
            truncated_records if str(settings.get("feature_scope")) == "input_document" else 0
        ),
        "max_unpadded_tokens": max_unpadded_tokens,
        "feature_storage_dtype": np.dtype(feature_dtype).name,
        "padding_fraction": 1.0 - active_tokens / max(padded_tokens, 1),
    }
    np.savez_compressed(
        output,
        features=merged["features"][order].astype(feature_dtype),
        sample_ids=merged["sample_ids"][order],
        labels=merged["labels"][order],
        query_ids=merged["query_ids"][order],
        audit_document_group_ids=merged["audit_document_group_ids"][order],
        input_document_ids=merged["input_document_ids"][order],
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


if __name__ == "__main__":
    main()
