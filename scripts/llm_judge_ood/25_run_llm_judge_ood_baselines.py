#!/usr/bin/env python3
"""Run evaluation-only OOD, operational, and label-cost baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json
from src.llm_judge_ood.eval.baselines import (
    DetectionBaselineConfig,
    evaluate_detection_baselines,
    evaluate_label_cost_curve,
    evaluate_operational_baselines,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone LLM Judge OOD baseline sweeps.")
    parser.add_argument("--features", required=True, help="NPZ with source_features and target_features arrays.")
    parser.add_argument("--output", required=True, help="JSON output path.")
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100, 200, 500, 1000, 10000],
    )
    parser.add_argument("--mmd-permutations", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--operational-json", default=None, help="Optional JSON with ood_scores/harmful_windows and signals.")
    parser.add_argument("--harmfulness-json", default=None, help="Optional JSON with judge_predictions/labels/rater_scores/reference_excess_human_error.")
    args = parser.parse_args()

    with np.load(args.features, allow_pickle=False) as payload:
        if "source_features" not in payload or "target_features" not in payload:
            raise ValueError("features NPZ must contain source_features and target_features")
        kwargs = {
            "source_logits": payload["source_logits"] if "source_logits" in payload else None,
            "target_logits": payload["target_logits"] if "target_logits" in payload else None,
        }
        detection = evaluate_detection_baselines(
            payload["source_features"],
            payload["target_features"],
            **kwargs,
            config=DetectionBaselineConfig(
                sample_sizes=tuple(args.sample_sizes),
                mmd_permutations=int(args.mmd_permutations),
                alpha=float(args.alpha),
            ),
        )
    result: dict[str, object] = {"detection": detection}
    if args.operational_json:
        operational = _read_json(args.operational_json)
        result["operational"] = evaluate_operational_baselines(**operational)
    if args.harmfulness_json:
        harmfulness = _read_json(args.harmfulness_json)
        result["harmfulness"] = evaluate_label_cost_curve(**harmfulness)
    write_json(args.output, result)
    print(json.dumps({"output": str(args.output), "sections": sorted(result)}, indent=2))


def _read_json(path: str) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {path}")
    return payload


if __name__ == "__main__":
    main()
