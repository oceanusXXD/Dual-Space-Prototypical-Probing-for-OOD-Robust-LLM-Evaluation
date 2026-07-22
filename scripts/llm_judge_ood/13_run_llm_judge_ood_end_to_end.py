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
from src.llm_judge_ood.eval import build_result_tables
from src.llm_judge_ood.pipelines.config import config_from_mapping
from src.llm_judge_ood.pipelines.sample_ood import run_sample_ood_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run document-level LLM Judge OOD pipeline and export compact result tables.")
    parser.add_argument("--config", default="configs/llm_judge_ood/llm_judge_ood_smoke.json")
    parser.add_argument("--output-dir", default="artifacts/llm_judge_ood_end_to_end")
    parser.add_argument("--input", nargs="+", default=None)
    parser.add_argument("--max-input-documents", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the top-level lifecycle seed while retaining the fixed model-selection settings.",
    )
    parser.add_argument("--judge-hidden-feature-path", default=None)
    parser.add_argument("--document-hidden-feature-path", default=None)
    args = parser.parse_args()

    config = replace(config_from_mapping(read_json(args.config)), output_dir=args.output_dir)
    if args.input is not None:
        config = replace(config, input_paths=tuple(str(path) for path in args.input))
    if args.max_input_documents is not None:
        config = replace(config, max_input_documents=int(args.max_input_documents))
    if args.seed is not None:
        config = replace(config, seed=int(args.seed))
    if args.judge_hidden_feature_path is not None:
        config = replace(config, judge_hidden_feature_path=str(args.judge_hidden_feature_path))
    if args.document_hidden_feature_path is not None:
        config = replace(config, document_hidden_feature_path=str(args.document_hidden_feature_path))
    summary = run_sample_ood_pipeline(config)
    table_paths = build_result_tables(summary, output_dir=Path(args.output_dir) / "tables")
    print(json.dumps({"summary": summary["outputs"]["summary"], "tables": table_paths}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
