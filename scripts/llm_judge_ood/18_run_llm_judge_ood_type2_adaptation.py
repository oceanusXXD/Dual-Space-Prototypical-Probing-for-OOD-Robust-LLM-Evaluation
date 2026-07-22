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
from src.llm_judge_ood.pipelines.type2 import Type2Config, run_type2_new_query_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Type-II Judge adaptation for new audit queries.")
    parser.add_argument("--config", default="configs/llm_judge_ood/llm_judge_ood_smoke.json")
    parser.add_argument("--output-dir", default="artifacts/llm_judge_ood_type2_adaptation")
    parser.add_argument("--max-input-documents", type=int, default=None)
    parser.add_argument("--new-query-id", action="append", default=[])
    parser.add_argument("--label-split", action="append", default=None)
    parser.add_argument("--eval-split", action="append", default=None)
    parser.add_argument("--budget", action="append", type=int, default=None)
    args = parser.parse_args()
    sample_config = config_from_mapping(read_json(args.config))
    if args.max_input_documents is not None:
        sample_config = replace(sample_config, max_input_documents=int(args.max_input_documents))
    config = Type2Config(
        sample=sample_config,
        new_query_ids=tuple(args.new_query_id),
        label_splits=tuple(args.label_split or ("deployment_adapt",)),
        eval_splits=tuple(args.eval_split or ("deployment_future_test",)),
        budgets=tuple(args.budget or (8, 16, 32, 64, 128)),
        output_dir=args.output_dir,
    )
    summary = run_type2_new_query_pipeline(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
