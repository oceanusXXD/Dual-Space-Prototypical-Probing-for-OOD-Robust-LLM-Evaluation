#!/usr/bin/env python3
"""Extract score-token hidden features for one FLASK cell and probe GT labels.

This is intentionally a focused diagnostic, not a full 5x6 production run. It
uses the existing Direct-Judge completions, teacher-forces prompt + completion,
captures two transformer-block outputs around the generated score digit, and
then trains the same linear 5-class head on the existing question-group split.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.model.baselines import LinearJudgeConfig, PerQueryLinearJudge
from src.llm_judge_ood.shared.metrics import judge_metrics
from src.models.extract_hidden import QWEN3_5_4B_HIDDEN_SIZE, load_qwen_model


CLASSES = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b-space",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_with_direct_judge.jsonl"
        ),
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/cpu_3x3_heads/humanities__comprehension.split.json"),
    )
    parser.add_argument(
        "--prompt-mean-features",
        type=Path,
        default=Path(
            "artifacts/flask_minimal_validation/direct_judge_model_inputs/"
            "b_space_hidden_states.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/score_token_hidden_humanities_comprehension"),
    )
    parser.add_argument("--domain", default="Humanities")
    parser.add_argument("--skill", default="Comprehension")
    parser.add_argument("--model-path", type=Path, default=Path("/home/zeus/models/qwen3.5-4b"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", choices=("auto", "bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2"), default="flash_attention_2")
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {args.output_dir}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    direct = load_direct_helpers()
    rows = [
        row for row in load_jsonl(args.b_space)
        if str(row["domain_ids"][0]) == args.domain and str(row["task_id"]) == args.skill
    ]
    if not rows:
        raise ValueError(f"Empty cell: {args.domain} x {args.skill}")
    tokenizer, model, device = load_qwen_model(
        args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    target_blocks = (22, 31)  # Documented layers 23 and 32.
    records = prepare_teacher_forcing_records(
        rows=rows,
        tokenizer=tokenizer,
        direct=direct,
        max_prompt_length=int(args.max_prompt_length),
    )
    variants = extract_features(
        records=records,
        model=model,
        tokenizer=tokenizer,
        device=device,
        target_blocks=target_blocks,
        batch_size=int(args.batch_size),
    )
    sample_ids = np.asarray([str(row["b_id"]) for row in rows])
    labels = np.asarray([int(row["ground_truth"]) for row in rows], dtype=np.int8)
    question_ids = np.asarray([str(row["base_id"]) for row in rows])
    direct_scores = np.asarray([int(row["direct_score"]) for row in rows], dtype=np.int8)
    for name, features in variants.items():
        np.savez_compressed(
            args.output_dir / f"{name}_features.npz",
            features=features.astype(np.float16),
            sample_ids=sample_ids,
            labels=labels,
            query_ids=question_ids,
            direct_scores=direct_scores,
            domain_ids=np.asarray([args.domain] * len(rows)),
            task_ids=np.asarray([args.skill] * len(rows)),
            metadata_json=np.asarray(json.dumps({
                "artifact_type": f"flask_cell_{name}_score_token_hidden_v1",
                "domain": args.domain,
                "skill": args.skill,
                "feature_scope": name,
                "layers": [23, 32],
                "shape": [len(rows), 2, QWEN3_5_4B_HIDDEN_SIZE],
                "teacher_forced_existing_direct_judge_completion": True,
                "score_digit_position": "located in direct_judge_raw_completion",
            }, ensure_ascii=False)),
        )
    prompt_mean = load_prompt_mean_cell_features(args.prompt_mean_features, rows)
    summary = {
        "artifact_type": "flask_cell_score_token_hidden_probe_v1",
        "domain": args.domain,
        "skill": args.skill,
        "rows": len(rows),
        "features": {name: str(args.output_dir / f"{name}_features.npz") for name in variants},
        "prompt_mean_reference": str(args.prompt_mean_features),
        "metrics": evaluate_variants(
            rows=rows,
            labels=labels,
            direct_scores=direct_scores,
            question_ids=question_ids,
            split_path=args.split,
            feature_variants={**variants, "prompt_mean_reference": prompt_mean},
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))


def prepare_teacher_forcing_records(*, rows: list[dict[str, Any]], tokenizer: Any, direct: Any, max_prompt_length: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        prompt = direct._chat_prompt(row, tokenizer)
        completion = str(row.get("direct_judge_raw_completion") or "")
        score = str(int(row["direct_score"]))
        match = re.search(rf"(?<!\d){re.escape(score)}(?!\d)", completion)
        if match is None:
            completion = f'{{"score": {score}}}'
            match = re.search(rf"(?<!\d){re.escape(score)}(?!\d)", completion)
        assert match is not None
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        prompt_ids = direct._head_tail(ids=prompt_ids, max_length=max_prompt_length)
        completion_payload = tokenizer(completion, add_special_tokens=False, return_offsets_mapping=True)
        completion_ids = list(completion_payload.input_ids)
        offsets = list(completion_payload.offset_mapping)
        score_char = match.start()
        local_score_index = None
        for token_index, (start, end) in enumerate(offsets):
            if start <= score_char < end:
                local_score_index = token_index
                break
        if local_score_index is None:
            raise RuntimeError(f"Could not locate score digit token for {row['b_id']}: {completion!r}")
        input_ids = prompt_ids + completion_ids
        score_index = len(prompt_ids) + int(local_score_index)
        if score_index <= 0:
            raise RuntimeError("Score token index cannot be the first token")
        records.append({
            "input_ids": input_ids,
            "pre_score_index": score_index - 1,
            "score_index": score_index,
            "score_token": completion_ids[local_score_index],
            "b_id": row["b_id"],
        })
    return records


def extract_features(
    *, records: list[dict[str, Any]], model: Any, tokenizer: Any, device: torch.device,
    target_blocks: tuple[int, int], batch_size: int,
) -> dict[str, np.ndarray]:
    pre = np.empty((len(records), len(target_blocks), QWEN3_5_4B_HIDDEN_SIZE), dtype=np.float16)
    score = np.empty_like(pre)
    captured: dict[int, torch.Tensor] = {}
    handles = []
    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers")
    for block in target_blocks:
        def make_hook(index: int):
            def hook(_module: Any, _inputs: Any, output: Any) -> None:
                hidden = output[0] if isinstance(output, tuple) else output
                captured[index] = hidden.detach()
            return hook
        handles.append(layers[block].register_forward_hook(make_hook(block)))
    try:
        pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
        for start in tqdm(range(0, len(records), batch_size), desc="score-token hidden"):
            batch = records[start:start + batch_size]
            max_len = max(len(record["input_ids"]) for record in batch)
            ids = torch.full((len(batch), max_len), pad, dtype=torch.long, device=device)
            mask = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
            for row_index, record in enumerate(batch):
                values = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
                ids[row_index, :len(values)] = values
                mask[row_index, :len(values)] = 1
            captured.clear()
            with torch.no_grad():
                model(input_ids=ids, attention_mask=mask, use_cache=False)
            for layer_pos, block in enumerate(target_blocks):
                hidden = captured[block]
                for row_index, record in enumerate(batch):
                    pre[start + row_index, layer_pos] = hidden[row_index, record["pre_score_index"]].detach().cpu().to(torch.float16).numpy()
                    score[start + row_index, layer_pos] = hidden[row_index, record["score_index"]].detach().cpu().to(torch.float16).numpy()
    finally:
        for handle in handles:
            handle.remove()
    return {"pre_score_token": pre, "score_token": score}


def evaluate_variants(
    *, rows: list[dict[str, Any]], labels: np.ndarray, direct_scores: np.ndarray,
    question_ids: np.ndarray, split_path: Path, feature_variants: dict[str, np.ndarray],
) -> dict[str, Any]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_questions = set(str(value) for value in split["train_question_ids"])
    train_mask = np.isin(question_ids, np.asarray(sorted(train_questions)))
    test_mask = ~train_mask
    output: dict[str, Any] = {
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "train_label_counts": counts(labels[train_mask]),
        "test_label_counts": counts(labels[test_mask]),
        "direct_judge_same_test": metric_row(labels[test_mask], direct_scores[test_mask]),
    }
    query_ids = np.full(len(labels), "score_token_probe", dtype=object)
    for name, features in feature_variants.items():
        cfg = LinearJudgeConfig(
            method="linear", representation="last_layer", pca_dim=2560,
            class_values=CLASSES, seed=42, learning_rate=1e-3, weight_decay=1e-4,
            epochs=50, batch_size=256, patience=6, device="cpu",
            class_weight="balanced", head_sharing="shared",
        )
        model = PerQueryLinearJudge(cfg).fit(
            features, labels, query_ids, train_mask=train_mask,
            validation_mask=np.zeros(len(labels), dtype=bool),
        )
        pred_output = model.predict_output(features, query_ids)
        predictions = pred_output.classes[np.argmax(pred_output.probabilities, axis=1)].astype(np.int8)
        output[name] = {
            "train": metric_row(labels[train_mask], predictions[train_mask]),
            "test": metric_row(labels[test_mask], predictions[test_mask]),
            "test_prediction_counts": counts(predictions[test_mask]),
        }
        model.save(Path("artifacts/flask_minimal_validation/score_token_hidden_humanities_comprehension") / f"{name}_linear_head.joblib")
    return output


def load_prompt_mean_cell_features(path: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    row_ids = [str(row["b_id"]) for row in rows]
    with np.load(path, allow_pickle=False) as payload:
        features = np.asarray(payload["features"], dtype=np.float16)
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
    index = {value: idx for idx, value in enumerate(sample_ids.tolist())}
    return features[np.asarray([index[value] for value in row_ids], dtype=np.int64)]


def metric_row(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    value = judge_metrics(labels, predictions, class_values=CLASSES)
    return {
        "rows": int(len(labels)),
        "mae": float(value["mae"]),
        "exact_accuracy": float(value["accuracy"]),
        "plus_minus_1_accuracy": float(np.mean(np.abs(labels - predictions) <= 1)),
        "quadratic_weighted_kappa": float(value["qwk"]),
    }


def counts(values: np.ndarray) -> dict[str, int]:
    counter = Counter(int(value) for value in values.tolist())
    return {str(value): int(counter.get(value, 0)) for value in CLASSES}


def load_direct_helpers() -> Any:
    path = ROOT / "scripts/llm_judge_ood/49_run_flask_minimal_direct_judge.py"
    spec = importlib.util.spec_from_file_location("flask_direct_judge_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Direct Judge helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


if __name__ == "__main__":
    main()
