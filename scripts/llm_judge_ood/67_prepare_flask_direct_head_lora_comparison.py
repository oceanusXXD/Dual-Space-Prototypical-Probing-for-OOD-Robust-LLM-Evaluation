#!/usr/bin/env python3
"""Prepare shared FLASK rows/splits for Direct Judge vs head vs LoRA.

Use --counts-only before the FLASK dataset is downloaded. Once
b_space_single_domain.jsonl exists, run it without that flag to materialize the
exact row ids used by all three methods.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_judge_ood.flask_comparison import (
    SPLIT_NAMES,
    attach_splits,
    build_shared_group_split,
    cell_id,
    filter_selected_rows,
    markdown_table,
    parse_cell_id,
    read_jsonl,
    result_template_rows,
    selected_cell_records,
    top_transfer_grid,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--b-space",
        type=Path,
        default=Path("datasets/processed/flask_domain_task_v1/b_space_single_domain.jsonl"),
        help="Prepared FLASK single-domain B-space JSONL from script 48.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/flask_direct_head_lora_comparison"),
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=Path("docs/FLASK_DirectJudge_Head_LoRA_对比实验表.md"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional explicit cells in 'Domain::Task' form. Defaults to a top 2-domain × 2-skill grid.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.60)
    parser.add_argument("--validation-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.30)
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Write the count-based plan/doc even if the B-space dataset is not present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = {
        "train": float(args.train_ratio),
        "validation": float(args.validation_ratio),
        "test": float(args.test_ratio),
    }
    if args.cells:
        cells = tuple(parse_cell_id(value) for value in args.cells)
    else:
        cells = top_transfer_grid()
    if len(cells) != 4:
        raise ValueError("This comparison expects exactly four Domain::Task cells")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    planned_records = selected_cell_records(cells)
    write_csv(args.output_dir / "selected_cells_from_doc_counts.csv", planned_records)
    write_json(args.output_dir / "selected_cells_from_doc_counts.json", planned_records)
    write_csv(args.output_dir / "result_template.csv", result_template_rows(cells))

    split_manifest: dict[str, Any] | None = None
    selected_rows_with_split: list[dict[str, Any]] = []
    if args.b_space.exists():
        selected_rows, selection_audit = filter_selected_rows(read_jsonl(args.b_space), cells)
        row_split, split_audit = build_shared_group_split(
            selected_rows,
            seed=int(args.seed),
            ratios=ratios,
        )
        selected_rows_with_split = attach_splits(selected_rows, row_split)
        write_jsonl(args.output_dir / "comparison_rows.jsonl", selected_rows_with_split)
        for split in SPLIT_NAMES:
            write_jsonl(
                args.output_dir / f"{split}.jsonl",
                [row for row in selected_rows_with_split if row["split"] == split],
            )
        split_manifest = {
            "artifact_type": "flask_direct_head_lora_shared_split_v1",
            "source_b_space": str(args.b_space),
            "seed": int(args.seed),
            "selected_cells": [cell_id(domain, task) for domain, task in cells],
            "ratios_requested": ratios,
            "selection_audit": selection_audit,
            "split_audit": split_audit,
            "row_splits": {str(row["b_id"]): str(row["split"]) for row in selected_rows_with_split},
        }
        write_json(args.output_dir / "split_manifest.json", split_manifest)
        write_csv(args.output_dir / "split_audit_by_cell.csv", split_audit["cells"])
    elif not args.counts_only:
        raise FileNotFoundError(
            f"{args.b_space} does not exist. Run script 48 after downloading FLASK, "
            "or pass --counts-only to write only the planning tables."
        )

    doc_text = render_doc(
        cells=cells,
        planned_records=planned_records,
        output_dir=args.output_dir,
        split_manifest=split_manifest,
        selected_rows=selected_rows_with_split,
        ratios=ratios,
    )
    args.doc.parent.mkdir(parents=True, exist_ok=True)
    args.doc.write_text(doc_text, encoding="utf-8")
    print(
        json.dumps(
            {
                "doc": str(args.doc),
                "output_dir": str(args.output_dir),
                "selected_cells": [cell_id(domain, task) for domain, task in cells],
                "planned_rows": sum(int(row["doc_rows"]) for row in planned_records),
                "materialized_rows": len(selected_rows_with_split),
                "has_split_manifest": split_manifest is not None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def render_doc(
    *,
    cells: tuple[tuple[str, str], ...],
    planned_records: list[dict[str, Any]],
    output_dir: Path,
    split_manifest: dict[str, Any] | None,
    selected_rows: list[dict[str, Any]],
    ratios: dict[str, float],
) -> str:
    planned_total = sum(int(row["doc_rows"]) for row in planned_records)
    cell_rows = [
        {
            "Cell": row["cell_id"],
            "Doc rows": row["doc_rows"],
            "Train 60%": row["target_train_rows"],
            "Val 10%": row["target_validation_rows"],
            "Test 30%": row["target_test_rows"],
        }
        for row in planned_records
    ]
    source_rows = source_training_rows(planned_records, split_manifest)
    train_rows = sum(int(row["Head/LoRA train rows"]) for row in source_rows)
    validation_rows = sum(int(row["Head/LoRA validation rows"]) for row in source_rows)
    test_rows = sum(int(row["Own-cell test rows"]) for row in source_rows)
    method_rows = [
        {
            "Method": "Direct Judge",
            "Train rows": 0,
            "Validation rows": "not used",
            "Test rows": test_rows,
            "Notes": "No supervised training; evaluate once on each target test cell.",
        },
        {
            "Method": "Classification head",
            "Train rows": f"{train_rows} total; see source table",
            "Validation rows": f"{validation_rows} total; see source table",
            "Test rows": f"{test_rows} rows × 4 heads = {test_rows * len(cells)} eval rows",
            "Notes": "Train 4 separate heads; each source head is evaluated on all 4 target test cells.",
        },
        {
            "Method": "LoRA",
            "Train rows": f"{train_rows} total; see source table",
            "Validation rows": f"{validation_rows} total; see source table",
            "Test rows": f"{test_rows} rows × 4 adapters = {test_rows * len(cells)} eval rows",
            "Notes": "Train 4 separate LoRA adapters; each adapter is evaluated on all 4 target test cells.",
        },
    ]
    if split_manifest is not None:
        audit = split_manifest["split_audit"]
        materialized = "\n\n## 已生成的实际 split\n\n"
        materialized += f"- 总行数：{len(selected_rows)}\n"
        materialized += f"- Train / Val / Test：{audit['rows_by_split']}\n"
        materialized += f"- 共享集合文件：'{output_dir / 'comparison_rows.jsonl'}'\n"
        materialized += f"- Split manifest：'{output_dir / 'split_manifest.json'}'\n"
        materialized += f"- Cell audit CSV：'{output_dir / 'split_audit_by_cell.csv'}'\n"
    else:
        materialized = (
            "\n\n## 当前状态\n\n"
            "- 数据集尚未落地，所以这里只写入文档计数下的目标 split；等 FLASK 下载并运行 script 48 后，再运行本脚本生成真实 row ids。\n"
        )

    result_columns = [
        "method",
        "source_cell_id",
        "target_cell_id",
        "split",
        "rows",
        "parse_rate",
        "mae",
        "exact_accuracy",
        "plus_minus_1_accuracy",
        "quadratic_weighted_kappa",
        "notes",
    ]
    template_rows = result_template_rows(cells)
    command_block = f"""~~~bash
# 推荐：在 GPU 环境一条命令跑完整 pipeline
python scripts/llm_judge_ood/72_run_flask_direct_head_lora_pipeline.py

# 手动分步：
# 1) 下载/准备 FLASK domain-task B-space（数据下载完成后）
python scripts/llm_judge_ood/48_prepare_flask_domain_task_splits.py

# 2) 固定共同 row ids：question_id 分组，60/10/30
python scripts/llm_judge_ood/67_prepare_flask_direct_head_lora_comparison.py

# 3) 0.8B Direct Judge + strict final-prelogit features（同一批 rows）
python scripts/llm_judge_ood/68_run_flask_comparison_direct_and_features.py \\
  --rows {output_dir / 'comparison_rows.jsonl'} \\
  --split-manifest {output_dir / 'split_manifest.json'}

# 4) 训练分类头（同一 train/validation/test）
python scripts/llm_judge_ood/69_train_flask_comparison_head.py \\
  --rows {output_dir / 'direct_and_features/b_space_with_direct_judge.jsonl'} \\
  --split-manifest {output_dir / 'split_manifest.json'} \\
  --features {output_dir / 'direct_and_features/strict_final_prelogit_features.npz'}

# 5) 训练 LoRA（同一 train/validation/test）
python scripts/llm_judge_ood/70_train_flask_comparison_lora.py \\
  --rows {output_dir / 'comparison_rows.jsonl'} \\
  --split-manifest {output_dir / 'split_manifest.json'}

# 6) 汇总三种方法的 performance 表
python scripts/llm_judge_ood/71_summarize_flask_comparison_results.py
~~~"""
    return f"""# FLASK Direct Judge / 分类头 / LoRA 对比实验表

## 结论先写

本轮按你的要求把一个 Domain × 一个 Skill 当成一个数据集，选择单一 Domain 视图里样本量最大的 2×2 transfer grid：Language / Culture × Comprehension / Commonsense Understanding。这样得到 4 个数据集，避免多 Domain membership 带来的跨 cell 重复。文档计数合计 {planned_total} 条 B-space 评分行；真实可用行数是 {len(selected_rows) if selected_rows else '待生成'}，每个 cell 内按 question_id 分组切成 train {ratios['train']:.0%} / validation {ratios['validation']:.0%} / test {ratios['test']:.0%}。

Direct Judge 没有训练集，训练数据量记为 0，只在四个 target cell 的 test split 上跑。分类头和 LoRA 各训练 4 个模型：每个 source cell 训练 1 个分类头和 1 个 LoRA，然后分别测试四个 target cell，所以分类头是 4×4 次测试，LoRA 也是 4×4 次测试。两种训练方法使用完全相同的 row ids。

## 选中的大样本 cells

{markdown_table(cell_rows, ["Cell", "Doc rows", "Train 60%", "Val 10%", "Test 30%"])}

## 三方法数据量

{markdown_table(method_rows, ["Method", "Train rows", "Validation rows", "Test rows", "Notes"])}

## 4 个 source 训练集

{markdown_table(source_rows, ["Source cell", "Head/LoRA train rows", "Head/LoRA validation rows", "Own-cell test rows"])}
{materialized}

## 运行顺序

{command_block}

## 结果表模板

实际训练/测试完成后，用 71_summarize_flask_comparison_results.py 填充下面字段；空表 CSV 已写到 '{output_dir / 'result_template.csv'}'。Direct Judge 行没有 source_cell_id；分类头和 LoRA 行用 source_cell_id → target_cell_id 表示 4×4 迁移测试。

{markdown_table(template_rows, result_columns)}
"""


def source_training_rows(
    planned_records: list[dict[str, Any]],
    split_manifest: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if split_manifest is None:
        return [
            {
                "Source cell": row["cell_id"],
                "Head/LoRA train rows": row["target_train_rows"],
                "Head/LoRA validation rows": row["target_validation_rows"],
                "Own-cell test rows": row["target_test_rows"],
            }
            for row in planned_records
        ]
    by_cell: dict[str, dict[str, int]] = defaultdict(dict)
    for row in split_manifest["split_audit"]["cells"]:
        by_cell[str(row["cell_id"])][str(row["split"])] = int(row["rows"])
    return [
        {
            "Source cell": row["cell_id"],
            "Head/LoRA train rows": by_cell[row["cell_id"]]["train"],
            "Head/LoRA validation rows": by_cell[row["cell_id"]]["validation"],
            "Own-cell test rows": by_cell[row["cell_id"]]["test"],
        }
        for row in planned_records
    ]


if __name__ == "__main__":
    main()
