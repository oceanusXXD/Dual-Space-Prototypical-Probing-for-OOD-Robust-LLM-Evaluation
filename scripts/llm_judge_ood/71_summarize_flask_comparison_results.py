#!/usr/bin/env python3
"""Combine Direct Judge, classification-head, and LoRA FLASK comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_judge_ood.flask_comparison import markdown_table, write_csv, write_json


METRIC_COLUMNS = [
    "method",
    "source_cell_id",
    "target_cell_id",
    "split",
    "rows",
    "parsed_rows",
    "parse_rate",
    "mae",
    "exact_accuracy",
    "plus_minus_1_accuracy",
    "quadratic_weighted_kappa",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison/performance_summary.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/FLASK_DirectJudge_Head_LoRA_对比实验结果.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_rows = read_template_rows(args.base_dir / "result_template.csv")
    rows: list[dict[str, Any]] = []
    rows.extend(
        read_metric_file(
            args.base_dir / "direct_and_features/direct_judge_metrics.csv",
            "direct_judge",
            template_rows,
        )
    )
    rows.extend(
        read_metric_file(
            args.base_dir / "classification_head/classification_head_4x4_metrics.csv",
            "classification_head",
            template_rows,
        )
    )
    rows.extend(read_metric_file(args.base_dir / "lora/lora_4x4_metrics.csv", "lora", template_rows))
    rows = normalize_rows(rows)
    if not rows:
        raise FileNotFoundError(f"No metric files found under {args.base_dir}")
    write_csv(args.output_csv, rows, fieldnames=METRIC_COLUMNS)
    summary = {
        "artifact_type": "flask_direct_head_lora_performance_summary_v1",
        "base_dir": str(args.base_dir),
        "rows": len(rows),
        "methods_present": sorted({row["method"] for row in rows}),
        "output_csv": str(args.output_csv),
        "output_md": str(args.output_md),
        "pipeline_preflight": read_json(args.base_dir / "pipeline_preflight.json"),
    }
    write_json(args.base_dir / "performance_summary.json", summary)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def read_metric_file(path: Path, method: str, template_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path.exists():
        pending = [row for row in template_rows if row.get("method") == method]
        if not pending:
            pending = [{"method": method, "source_cell_id": "", "target_cell_id": "", "split": "test"}]
        return [{**row, "parsed_rows": "", "status": f"missing: {path}"} for row in pending]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{**row, "method": row.get("method") or method, "status": "complete"} for row in reader]


def read_template_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        item = {key: row.get(key, "") for key in METRIC_COLUMNS}
        if not item["source_cell_id"] and row.get("cell_id"):
            item["target_cell_id"] = row.get("cell_id", "")
        out.append(item)
    order = {"direct_judge": 0, "classification_head": 1, "lora": 2}
    return sorted(
        out,
        key=lambda row: (
            order.get(row["method"], 99),
            row["source_cell_id"],
            row["target_cell_id"],
        ),
    )


def render_markdown(rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    complete_rows = [row for row in rows if row.get("status") == "complete"]
    pending_rows = [row for row in rows if row.get("status") != "complete"]
    display_columns = [
        "method",
        "source_cell_id",
        "target_cell_id",
        "rows",
        "mae",
        "exact_accuracy",
        "plus_minus_1_accuracy",
        "quadratic_weighted_kappa",
        "status",
    ]
    pending = ""
    if pending_rows:
        pending = (
            "\n\n## Pending\n\n"
            + markdown_table(pending_rows, ["method", "source_cell_id", "target_cell_id", "status"])
        )
    runtime = runtime_section(summary.get("pipeline_preflight"))
    return f"""# FLASK Direct Judge / 分类头 / LoRA Performance Summary

## Scope

- 数据集：FLASK single-domain 2×2 transfer grid。
- 模型：Direct Judge 与 LoRA 使用 Qwen3.5-0.8B；分类头使用同一批 strict final-prelogit hidden features。
- Split：所有方法共享 question_id 分组后的 60% train / 10% validation / 30% test；Direct Judge 只在 test 上评估。
- Output CSV：{summary['output_csv']}
{runtime}

## Results

{markdown_table(complete_rows, display_columns)}
{pending}
"""


def runtime_section(preflight: dict[str, Any] | None) -> str:
    if not preflight:
        return ""
    errors = preflight.get("errors") or []
    status = "ready" if not errors else "blocked"
    rows = [
        {
            "Item": "status",
            "Value": status,
        },
        {
            "Item": "local_deps",
            "Value": preflight.get("local_deps", ""),
        },
        {
            "Item": "torch_cuda_available",
            "Value": preflight.get("torch_cuda_available", ""),
        },
        {
            "Item": "errors",
            "Value": "; ".join(str(error) for error in errors) if errors else "",
        },
    ]
    return "\n## Runtime\n\n" + markdown_table(rows, ["Item", "Value"])


if __name__ == "__main__":
    main()
