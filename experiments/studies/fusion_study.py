from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.algorithm.confidence.fusion import weighted_score_fusion
from src.common.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fuse detector score columns from JSONL rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", nargs="+", required=True)
    parser.add_argument("--weights", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    weights = _parse_weights(args.weights)
    fused = weighted_score_fusion(
        {column: np.asarray([float(row[column]) for row in rows]) for column in args.columns},
        weights=weights,
    )
    output_rows = [{**row, "fused_score": float(fused[index])} for index, row in enumerate(rows)]
    write_jsonl(args.output, output_rows)
    print(json.dumps({"rows": len(output_rows), "output": str(args.output)}, indent=2))


def _parse_weights(items: list[str] | None) -> dict[str, float] | None:
    if not items:
        return None
    weights: dict[str, float] = {}
    for item in items:
        name, value = str(item).split("=", 1)
        weights[name] = float(value)
    return weights


if __name__ == "__main__":
    main()
