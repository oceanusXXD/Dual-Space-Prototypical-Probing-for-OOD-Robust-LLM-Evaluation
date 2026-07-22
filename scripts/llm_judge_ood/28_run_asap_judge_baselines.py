#!/usr/bin/env python3
"""Run matched-split ASAP Judge baselines and prepare external-model inputs."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_jsonl, write_json, write_jsonl
from src.llm_judge_ood.eval.judge_baselines import (
    EaseBaselineConfig,
    build_external_score_manifest,
    build_pandalm_pairwise_manifests,
    evaluate_pandalm_predictions,
    run_asap_judge_baselines,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full-data ASAP Judge baselines.")
    parser.add_argument("--input", default="artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl")
    parser.add_argument(
        "--current-scores",
        default=None,
        help="Optional frozen-Qwen scores. Omit while the new Judge-input cache is pending GPU extraction.",
    )
    parser.add_argument("--external-predictions", action="append", default=[])
    parser.add_argument("--pandalm-predictions", action="append", default=[])
    parser.add_argument(
        "--output-dir",
        default="artifacts/llm_judge_ood_asap/judge_baselines_contract_v1",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    current = _current_predictions(args.current_scores) if args.current_scores else None
    external = [row for path in args.external_predictions for row in read_jsonl(path)]
    summary = run_asap_judge_baselines(
        records,
        current_judge_predictions=current,
        external_predictions=external,
        config=EaseBaselineConfig(bootstrap_samples=int(args.bootstrap_samples)),
    )
    summary["run_metadata"] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "current_scores": str(args.current_scores) if args.current_scores else None,
        "current_scores_sha256": _sha256(args.current_scores) if args.current_scores else None,
        "python": platform.python_version(),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "nltk": importlib.metadata.version("nltk"),
        "seed": 42,
        "bootstrap_samples": int(args.bootstrap_samples),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    score_manifest = build_external_score_manifest(records)
    pair_manifest, pair_truth = build_pandalm_pairwise_manifests(records)
    write_jsonl(output_dir / "external_single_score_manifest.jsonl", score_manifest)
    write_jsonl(output_dir / "pandalm_pairwise_manifest.jsonl", pair_manifest)
    write_jsonl(output_dir / "pandalm_pairwise_truth.jsonl", pair_truth)
    panda_predictions = [row for path in args.pandalm_predictions for row in read_jsonl(path)]
    summary["pandalm"] = {
        "protocol": {
            "native_task": "pairwise preference",
            "not_comparable_to_qwk_mae": True,
            "pairs": len(pair_manifest),
            "test_documents_covered_at_most_once_per_prompt_pairing": True,
        },
        "results": evaluate_pandalm_predictions(pair_truth, panda_predictions),
        "status": "pending_external_inference" if not panda_predictions else "evaluated",
    }
    summary["external_manifests"] = {
        "single_score": str(output_dir / "external_single_score_manifest.jsonl"),
        "single_score_rows": len(score_manifest),
        "pandalm_pairs": str(output_dir / "pandalm_pairwise_manifest.jsonl"),
        "pandalm_truth": str(output_dir / "pandalm_pairwise_truth.jsonl"),
        "pandalm_pair_rows": len(pair_manifest),
        "labels_exposed_in_inference_manifests": False,
    }
    write_json(output_dir / "summary.json", summary)
    table = _result_rows(summary)
    pd.DataFrame(table).to_csv(output_dir / "judge_baselines.csv", index=False)
    print(
        json.dumps(
            {
                "summary": str(output_dir / "summary.json"),
                "table": str(output_dir / "judge_baselines.csv"),
                "methods": sorted(summary["methods"]),
                "inter_human_macro_prompt_qwk": summary["inter_human"]["macro_prompt_qwk"],
                "pandalm_status": summary["pandalm"]["status"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _current_predictions(path: str) -> dict[str, int]:
    predictions: dict[str, int] = {}
    for row in read_jsonl(path):
        sample_id = str(row["sample_id"])
        if sample_id in predictions:
            raise ValueError(f"Duplicate current Judge prediction for {sample_id}")
        predictions[sample_id] = int(row["judge_prediction"])
    return predictions


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, result in summary["methods"].items():
        if result.get("status") != "complete":
            rows.append(
                {
                    "method": method,
                    "scope": "test",
                    "status": result.get("status"),
                    "documents": result.get("coverage", 0),
                }
            )
            continue
        pooled = result["pooled"]
        macro = result["macro_prompt"]
        rows.append(
            {
                "method": method,
                "scope": "pooled_test",
                "status": "complete",
                "documents": result["coverage"],
                "qwk": pooled["qwk"],
                "spearman": pooled["spearman"],
                "mae": pooled["mae"],
                "qwk_ci95_low": None,
                "qwk_ci95_high": None,
            }
        )
        rows.append(
            {
                "method": method,
                "scope": "macro_prompt_test",
                "status": "complete",
                "documents": result["coverage"],
                "qwk": macro["qwk"],
                "spearman": macro["spearman"],
                "mae": macro["mae"],
                "qwk_ci95_low": result["macro_prompt_qwk_ci95"][0],
                "qwk_ci95_high": result["macro_prompt_qwk_ci95"][1],
            }
        )
        for prompt_id, metrics in result["by_prompt"].items():
            rows.append(
                {
                    "method": method,
                    "scope": f"prompt_{prompt_id}_test",
                    "status": "complete",
                    "documents": sum(
                        1
                        for row in read_jsonl(summary["external_manifests"]["single_score"])
                        if str(row["split"]) == "training_test"
                        and str(row["asap_prompt_id"]) == str(prompt_id)
                    )
                    if "external_manifests" in summary
                    else None,
                    "qwk": metrics["qwk"],
                    "spearman": metrics["spearman"],
                    "mae": metrics["mae"],
                    "qwk_ci95_low": None,
                    "qwk_ci95_high": None,
                }
            )
    human = summary["inter_human"]
    rows.append(
        {
            "method": "inter_human_raw_raters",
            "scope": "macro_prompt_test",
            "status": "complete",
            "documents": human["documents"],
            "qwk": human["macro_prompt_qwk"],
            "spearman": None,
            "mae": None,
            "qwk_ci95_low": None,
            "qwk_ci95_high": None,
        }
    )
    for prompt_id, metrics in human["by_prompt"].items():
        rows.append(
            {
                "method": "inter_human_raw_raters",
                "scope": f"prompt_{prompt_id}_test",
                "status": "complete",
                "documents": metrics["documents"],
                "qwk": metrics["qwk"],
                "spearman": metrics["spearman"],
                "mae": metrics["mae"],
                "qwk_ci95_low": metrics["qwk_ci95"][0],
                "qwk_ci95_high": metrics["qwk_ci95"][1],
            }
        )
    return rows


if __name__ == "__main__":
    main()
