#!/usr/bin/env python3
"""Materialize the documented non-empty FLASK model-input scope from one run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a completed Direct Judge/B-space coupled run to the current "
            "FLASK model-input policy: valid integer labels and non-empty candidate responses."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/direct_judge"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_minimal_validation/direct_judge_model_inputs"),
    )
    parser.add_argument("--expected-rows", type=int, default=18428)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir: Path = args.source_dir
    output_dir: Path = args.output_dir
    source_rows = load_jsonl(source_dir / "b_space_with_direct_judge.jsonl")
    selected = [
        row
        for row in source_rows
        if str(row.get("candidate_response") or "").strip()
        and _integer_score(row.get("ground_truth")) is not None
    ]
    if len(selected) != int(args.expected_rows):
        raise ValueError(
            f"Filtered B-space has {len(selected)} rows; expected {args.expected_rows}"
        )
    selected_ids = [str(row["b_id"]) for row in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Filtered B-space has duplicate b_id values")
    predictions = load_jsonl(source_dir / "direct_judge_predictions.jsonl")
    by_id = {str(row["b_id"]): row for row in predictions}
    if set(by_id) != {str(row["b_id"]) for row in source_rows}:
        raise ValueError("Source Direct Judge prediction coverage does not match source B-space")
    selected_predictions = [by_id[b_id] for b_id in selected_ids]

    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "b_space_with_direct_judge.jsonl", selected)
    write_jsonl(output_dir / "direct_judge_rows.jsonl", selected)
    write_jsonl(output_dir / "direct_judge_predictions.jsonl", selected_predictions)
    feature_metadata = filter_features(
        source_dir / "b_space_hidden_states.npz",
        output_dir / "b_space_hidden_states.npz",
        selected_ids=selected_ids,
        source_rows=len(source_rows),
    )
    write_json(output_dir / "b_space_hidden_states.metadata.json", feature_metadata)

    direct = _direct_judge_module()
    source_summary = json.loads((source_dir / "summary.json").read_text(encoding="utf-8"))
    summary = direct.summarize(
        selected=selected,
        predictions=selected_predictions,
        selection={
            "source_coupled_run": str(source_dir),
            "scope": "all_nonempty_candidate_response_rows",
            "ground_truth_filter": "integer scores in [1, 5]",
            "source_integer_b_rows": len(source_rows),
            "excluded_empty_candidate_response_rows": len(source_rows) - len(selected),
            "scored_b_space": str(output_dir / "b_space_with_direct_judge.jsonl"),
            "b_space_features": str(output_dir / "b_space_hidden_states.npz"),
        },
        runtime={
            **source_summary["runtime"],
            "formal_model_input_rows": len(selected),
            "source_coupled_generation_rows": len(source_rows),
            "postprocessed_without_additional_qwen_inference": True,
            "b_space_feature_shape": feature_metadata["shape"],
        },
    )
    summary["artifact_type"] = "flask_5x6_direct_judge_model_input_scope_v1"
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["global_metrics"], ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_features(
    source: Path,
    output: Path,
    *,
    selected_ids: list[str],
    source_rows: int,
) -> dict[str, Any]:
    if not source.exists():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as payload:
        source_ids = np.asarray(payload["sample_ids"]).astype(str)
        if len(source_ids) != int(source_rows):
            raise ValueError("B-space feature cache is not aligned with source B-space")
        if len(set(source_ids.tolist())) != len(source_ids):
            raise ValueError("B-space feature cache has duplicate sample ids")
        source_index = {sample_id: index for index, sample_id in enumerate(source_ids.tolist())}
        missing = [sample_id for sample_id in selected_ids if sample_id not in source_index]
        if missing:
            raise ValueError(f"B-space feature cache misses selected id {missing[0]!r}")
        order = np.asarray([source_index[sample_id] for sample_id in selected_ids], dtype=np.int64)
        arrays: dict[str, np.ndarray] = {}
        for key in payload.files:
            if key == "metadata_json":
                continue
            value = np.asarray(payload[key])
            arrays[key] = value[order] if value.ndim and len(value) == len(source_ids) else value
        old_metadata = json.loads(str(payload["metadata_json"].item()))
    metadata = {
        **old_metadata,
        "artifact_type": "flask_5x6_b_space_hidden_states_model_input_scope_v1",
        "source_feature_cache": str(source),
        "num_records": len(selected_ids),
        "shape": list(arrays["features"].shape),
        "excluded_empty_candidate_response_rows": int(source_rows) - len(selected_ids),
        "postprocessed_without_additional_qwen_inference": True,
    }
    np.savez_compressed(
        output,
        **arrays,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return metadata


def _direct_judge_module():
    path = ROOT / "scripts/llm_judge_ood/49_run_flask_minimal_direct_judge.py"
    spec = importlib.util.spec_from_file_location("flask_direct_judge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Direct Judge helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _integer_score(value: Any) -> int | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    rounded = int(round(score))
    return rounded if abs(score - rounded) < 1e-8 and 1 <= rounded <= 5 else None


if __name__ == "__main__":
    main()
