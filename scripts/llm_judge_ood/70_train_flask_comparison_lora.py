#!/usr/bin/env python3
"""Train four source-cell Qwen3.5-0.8B LoRA adapters and evaluate 4×4."""

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
    CLASSES,
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
from src.models.extract_hidden import load_qwen_model


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
        default=Path("artifacts/flask_direct_head_lora_comparison/lora"),
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_MODEL_PATH)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="sdpa")
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    validate_runtime(args)
    if args.overwrite and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.rows, args.split_manifest)
    cells = tuple(sorted({row_cell(row) for row in rows}, key=cell_sort_key))
    if len(cells) != 4:
        raise ValueError(f"Expected exactly four source/target cells, got {len(cells)}")
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    all_predictions: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    adapter_summaries: list[dict[str, Any]] = []
    for source_cell in cells:
        source_id = cell_id(*source_cell)
        source_rows = [row for row in rows if row_cell(row) == source_cell]
        train_rows = [row for row in source_rows if str(row["split"]) == "train"]
        validation_rows = [row for row in source_rows if str(row["split"]) == "validation"]
        if not train_rows or not validation_rows:
            raise ValueError(f"Source cell {source_id} is missing train or validation rows")
        if len({integer_score(row["ground_truth"]) for row in train_rows}) < 2:
            raise ValueError(f"Source cell {source_id} has fewer than two train labels")
        source_dir = args.output_dir / f"source_{slug(source_cell[0])}__{slug(source_cell[1])}"
        source_dir.mkdir(parents=True, exist_ok=True)
        tokenizer, model, device = load_lora_model(args)
        train_loss = train_adapter(
            model=model,
            tokenizer=tokenizer,
            device=device,
            train_rows=train_rows,
            validation_rows=validation_rows,
            source_id=source_id,
            args=args,
        )
        adapter_path = source_dir / "adapter"
        model.save_pretrained(adapter_path)
        tokenizer.save_pretrained(adapter_path)
        validation_predictions = predict_rows(
            model=model,
            tokenizer=tokenizer,
            device=device,
            rows=validation_rows,
            batch_size=int(args.eval_batch_size),
            max_prompt_length=int(args.max_prompt_length),
        )
        adapter_summaries.append(
            {
                "source_cell_id": source_id,
                "adapter_path": str(adapter_path),
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
                "train_label_counts": label_counts(train_rows),
                "train_loss_last": train_loss,
                "validation_metrics": metrics_from_predictions(
                    [row["ground_truth"] for row in validation_rows],
                    [validation_predictions[str(row["b_id"])] for row in validation_rows],
                ),
            }
        )
        for target_cell in cells:
            target_id = cell_id(*target_cell)
            target_test_rows = [
                row for row in rows
                if row_cell(row) == target_cell and str(row["split"]) == "test"
            ]
            predictions = predict_rows(
                model=model,
                tokenizer=tokenizer,
                device=device,
                rows=target_test_rows,
                batch_size=int(args.eval_batch_size),
                max_prompt_length=int(args.max_prompt_length),
            )
            metric = metrics_from_predictions(
                [row["ground_truth"] for row in target_test_rows],
                [predictions[str(row["b_id"])] for row in target_test_rows],
            )
            metrics_rows.append(
                {
                    "method": "lora",
                    "source_cell_id": source_id,
                    "target_cell_id": target_id,
                    "split": "test",
                    **metric,
                }
            )
            for row in target_test_rows:
                all_predictions.append(
                    {
                        "method": "lora",
                        "source_cell_id": source_id,
                        "target_cell_id": target_id,
                        "b_id": row["b_id"],
                        "split": row["split"],
                        "ground_truth": integer_score(row["ground_truth"]),
                        "predicted_score": predictions[str(row["b_id"])],
                    }
                )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_jsonl(args.output_dir / "lora_predictions.jsonl", all_predictions)
    write_csv(args.output_dir / "lora_4x4_metrics.csv", metrics_rows)
    summary = {
        "artifact_type": "flask_comparison_four_source_lora_adapters_v1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(args.model_path),
        "source_rows": str(args.rows),
        "split_manifest": str(args.split_manifest),
        "cells": [cell_id(*cell) for cell in cells],
        "adapter_count": len(adapter_summaries),
        "test_evaluations": len(metrics_rows),
        "training": {
            "epochs": int(args.epochs),
            "train_batch_size": int(args.train_batch_size),
            "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "lora_r": int(args.lora_r),
            "lora_alpha": int(args.lora_alpha),
            "lora_dropout": float(args.lora_dropout),
            "target_modules": list(args.target_modules),
        },
        "adapters": adapter_summaries,
        "metrics": metrics_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps({"summary": str(args.output_dir / "summary.json"), "test_evaluations": len(metrics_rows)}, ensure_ascii=False, indent=2))


def validate_runtime(args: argparse.Namespace) -> None:
    if not args.model_path.exists():
        raise FileNotFoundError(
            f"Model path does not exist: {args.model_path}. Pass --model-path to the local Qwen3.5-0.8B checkout."
        )
    if str(args.device).lower().startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "LoRA training was requested on CUDA, but torch.cuda.is_available() is false. "
            "Run this script in a GPU runtime, or pass --device cpu only for a tiny debug run; "
            "full 0.8B LoRA training on CPU is not practical."
        )
    try:
        import peft  # noqa: F401
    except ImportError as error:
        raise RuntimeError("LoRA training requires peft. Install requirements.txt before running this step.") from error


def load_rows(rows_path: Path, split_manifest_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(rows_path)
    manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    row_splits = {str(key): str(value) for key, value in manifest.get("row_splits", {}).items()}
    if set(row_splits) != {str(row["b_id"]) for row in rows}:
        raise ValueError("Split manifest row ids do not match --rows")
    return [{**row, "split": row_splits[str(row["b_id"])]} for row in rows]


def load_lora_model(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as error:
        raise RuntimeError("LoRA training requires peft. Install requirements.txt or pip install peft.") from error
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
    tokenizer.padding_side = "right"
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=int(args.lora_r),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        target_modules=list(args.target_modules),
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return tokenizer, model, device


def train_adapter(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    source_id: str,
    args: argparse.Namespace,
) -> float:
    del validation_rows
    model.train()
    examples = [supervised_example(row, tokenizer, int(args.max_prompt_length)) for row in train_rows]
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    accumulation = max(1, int(args.gradient_accumulation_steps))
    last_loss = 0.0
    global_step = 0
    for epoch in range(max(1, int(args.epochs))):
        order = sorted(range(len(examples)), key=lambda index: stable_rank(f"{args.seed}::{source_id}::{epoch}::{examples[index]['b_id']}"))
        progress = tqdm(range(0, len(order), int(args.train_batch_size)), desc=f"LoRA train {source_id} epoch {epoch + 1}")
        optimizer.zero_grad(set_to_none=True)
        for start in progress:
            batch_indices = order[start : start + int(args.train_batch_size)]
            batch = collate_examples([examples[index] for index in batch_indices], tokenizer, device)
            output = model(**batch)
            loss = output.loss / accumulation
            loss.backward()
            last_loss = float(loss.detach().cpu().item() * accumulation)
            global_step += 1
            if global_step % accumulation == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            progress.set_postfix(loss=f"{last_loss:.4f}")
        if global_step % accumulation:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    return last_loss


def supervised_example(row: dict[str, Any], tokenizer: Any, max_prompt_length: int) -> dict[str, Any]:
    prompt_ids = tokenizer(apply_chat_template(row, tokenizer), add_special_tokens=False).input_ids
    prompt_ids = head_tail(prompt_ids, max_prompt_length)
    score = str(integer_score(row["ground_truth"]))
    completion_ids = tokenizer(score, add_special_tokens=False).input_ids
    if not completion_ids:
        raise RuntimeError(f"Could not tokenize score for row {row['b_id']}")
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [int(tokenizer.eos_token_id)]
    input_ids = list(prompt_ids) + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids
    return {"b_id": str(row["b_id"]), "input_ids": input_ids, "labels": labels}


def collate_examples(examples: list[dict[str, Any]], tokenizer: Any, device: torch.device) -> dict[str, torch.Tensor]:
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
    width = max(len(example["input_ids"]) for example in examples)
    input_ids = torch.full((len(examples), width), pad_id, dtype=torch.long, device=device)
    labels = torch.full((len(examples), width), -100, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(examples), width), dtype=torch.long, device=device)
    for row_index, example in enumerate(examples):
        values = torch.as_tensor(example["input_ids"], dtype=torch.long, device=device)
        target = torch.as_tensor(example["labels"], dtype=torch.long, device=device)
        input_ids[row_index, : len(values)] = values
        labels[row_index, : len(target)] = target
        attention_mask[row_index, : len(values)] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict_rows(
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_prompt_length: int,
) -> dict[str, int | None]:
    model.eval()
    tokenizer.padding_side = "left"
    prompts = [apply_chat_template(row, tokenizer) for row in rows]
    token_ids = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )["input_ids"]
    predictions: dict[str, int | None] = {}
    order = sorted(range(len(rows)), key=lambda index: (len(token_ids[index]), stable_rank(rows[index]["b_id"])))
    for start in tqdm(range(0, len(order), max(1, batch_size)), desc="LoRA eval"):
        indices = order[start : start + max(1, batch_size)]
        batch_ids = [head_tail(token_ids[index], max_prompt_length) for index in indices]
        encoded = tokenizer.pad(
            [{"input_ids": ids} for ids in batch_ids],
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = int(encoded["input_ids"].shape[1])
        completions = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for index, completion in zip(indices, completions, strict=True):
            predictions[str(rows[index]["b_id"])] = parse_direct_score(completion)
    return predictions


def label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    values = [integer_score(row["ground_truth"]) for row in rows]
    return {str(label): int(sum(value == label for value in values)) for label in CLASSES}


if __name__ == "__main__":
    main()
