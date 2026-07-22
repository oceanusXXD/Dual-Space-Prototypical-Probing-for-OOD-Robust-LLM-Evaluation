#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter prepared LLM Judge records by split without changing ids.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    args = parser.parse_args()
    requested = {str(value) for value in args.splits}
    rows = [row for row in read_jsonl(args.input) if str(row.get("split")) in requested]
    if not rows:
        raise ValueError(f"No rows matched splits={sorted(requested)}")
    output = Path(args.output)
    write_jsonl(output, rows)
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split"))
        counts[split] = counts.get(split, 0) + 1
    metadata = {
        "artifact_type": "llm_judge_ood_split_subset",
        "input_path": str(args.input),
        "output": str(output),
        "splits": sorted(requested),
        "rows": len(rows),
        "split_counts": dict(sorted(counts.items())),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(metadata)


if __name__ == "__main__":
    main()
