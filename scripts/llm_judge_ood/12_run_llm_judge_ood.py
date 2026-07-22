#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_json
from src.llm_judge_ood.pipelines.config import config_from_mapping
from src.llm_judge_ood.pipelines.sample_ood import run_sample_ood_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone document-level LLM Judge OOD lifecycle pipeline.")
    parser.add_argument("--config", default="configs/llm_judge_ood/llm_judge_ood_smoke.json")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--input", nargs="*", default=None, help="Override input-document JSONL path(s).")
    parser.add_argument("--max-input-documents", type=int, default=None)
    args = parser.parse_args()
    config = config_from_mapping(read_json(args.config))
    if args.output_dir:
        config = replace(config, output_dir=args.output_dir)
    if args.input:
        config = replace(config, input_paths=tuple(args.input))
    if args.max_input_documents is not None:
        config = replace(config, max_input_documents=int(args.max_input_documents))
    summary = run_sample_ood_pipeline(config)
    print(json.dumps({"summary": summary["outputs"]["summary"], "ood_metrics": summary["ood_metrics"], "adaptation": summary["adaptation"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
