from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score


DOMAINS: tuple[str, ...] = (
    "Humanities",
    "Language",
    "Social Science",
    "History",
    "Culture",
    "Technology",
    "Coding",
    "Math",
    "Natural Science",
    "Health",
)

SKILLS: tuple[str, ...] = (
    "Comprehension",
    "Factuality",
    "Logical Correctness",
    "Commonsense Understanding",
    "Completeness",
    "Insightfulness",
    "Metacognition",
    "Readability",
    "Conciseness",
    "Harmlessness",
    "Logical Robustness",
    "Logical Efficiency",
)

CLASSES: tuple[int, ...] = (1, 2, 3, 4, 5)
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")
DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.60, "validation": 0.10, "test": 0.30}

FULL_DOMAIN_TASK_COUNTS: dict[str, list[int]] = {
    "Humanities": [4139, 1158, 825, 2700, 1800, 1139, 740, 1439, 825, 1004, 330, 75],
    "Language": [2204, 925, 705, 1986, 810, 1064, 234, 1110, 1035, 195, 90, 60],
    "Social Science": [3388, 1752, 644, 1313, 1065, 643, 564, 1049, 838, 599, 90, 45],
    "History": [990, 799, 225, 503, 449, 283, 90, 360, 210, 30, 15, 15],
    "Culture": [4258, 2301, 780, 2553, 1380, 1644, 442, 1874, 1317, 420, 254, 195],
    "Technology": [3163, 1453, 479, 1535, 1423, 1018, 246, 1364, 733, 253, 210, 210],
    "Coding": [1513, 750, 1168, 761, 645, 314, 73, 660, 420, 45, 1229, 1590],
    "Math": [1215, 469, 2655, 1587, 495, 164, 162, 510, 540, 15, 1379, 627],
    "Natural Science": [1799, 1382, 808, 973, 900, 285, 134, 480, 300, 90, 135, 75],
    "Health": [1469, 689, 210, 704, 645, 330, 164, 389, 389, 210, 90, 120],
}

SINGLE_DOMAIN_TASK_COUNTS: dict[str, list[int]] = {
    "Humanities": [1484, 365, 210, 937, 735, 360, 176, 585, 285, 419, 75, 0],
    "Language": [1140, 570, 465, 1015, 375, 540, 117, 630, 630, 90, 30, 30],
    "Social Science": [1228, 809, 329, 373, 315, 104, 194, 419, 344, 134, 30, 15],
    "History": [255, 296, 135, 89, 120, 30, 15, 90, 60, 15, 0, 15],
    "Culture": [1978, 1341, 570, 1272, 600, 476, 204, 795, 582, 105, 104, 90],
    "Technology": [974, 521, 150, 495, 374, 209, 72, 464, 224, 58, 90, 60],
    "Coding": [960, 482, 959, 435, 375, 90, 30, 465, 270, 15, 1140, 1457],
    "Math": [780, 277, 2070, 1122, 315, 60, 117, 360, 390, 15, 960, 434],
    "Natural Science": [974, 914, 538, 463, 510, 45, 60, 225, 180, 30, 90, 30],
    "Health": [509, 343, 75, 240, 285, 90, 30, 164, 164, 60, 15, 45],
}

DEFAULT_TRANSFER_DOMAINS: tuple[str, str] = ("Language", "Culture")
DEFAULT_TRANSFER_SKILLS: tuple[str, str] = (
    "Comprehension",
    "Commonsense Understanding",
)
DEFAULT_SELECTED_CELLS: tuple[tuple[str, str], ...] = tuple(
    (domain, skill)
    for domain in DEFAULT_TRANSFER_DOMAINS
    for skill in DEFAULT_TRANSFER_SKILLS
)

DIRECT_JUDGE_TEMPLATE_VERSION = "flask_comparison_direct_judge_digit_v1"
DIRECT_JUDGE_TEMPLATE = (
    "You are an evaluator.\n\n"
    "Skill and rubric:\n{rubric}\n\n"
    "Instruction:\n{instruction}\n\n"
    "Reference answer:\n{reference_answer}\n\n"
    "Candidate response:\n{candidate_response}\n\n"
    "Return exactly one digit only: 1, 2, 3, 4, or 5.\n"
    "Do not output JSON, markdown, explanations, labels, spaces, or any other text."
)


def selected_cell_records(
    cells: Sequence[tuple[str, str]] = DEFAULT_SELECTED_CELLS,
    *,
    count_view: str = "single_domain",
) -> list[dict[str, Any]]:
    counts = count_matrix(count_view)
    records: list[dict[str, Any]] = []
    for domain, skill in cells:
        records.append(
            {
                "cell_id": cell_id(domain, skill),
                "domain": domain,
                "task": skill,
                "doc_count_view": count_view,
                "doc_rows": int(counts[domain][SKILLS.index(skill)]),
                **split_targets(int(counts[domain][SKILLS.index(skill)])),
            }
        )
    return records


def count_matrix(view: str) -> dict[str, list[int]]:
    if view == "single_domain":
        return SINGLE_DOMAIN_TASK_COUNTS
    if view == "full":
        return FULL_DOMAIN_TASK_COUNTS
    raise ValueError("count view must be 'single_domain' or 'full'")


def top_cells(*, k: int, count_view: str = "single_domain") -> tuple[tuple[str, str], ...]:
    counts = count_matrix(count_view)
    ranked: list[tuple[int, str, str]] = []
    for domain in DOMAINS:
        for skill, count in zip(SKILLS, counts[domain], strict=True):
            if count > 0:
                ranked.append((int(count), domain, skill))
    ranked.sort(key=lambda item: (-item[0], DOMAINS.index(item[1]), SKILLS.index(item[2])))
    return tuple((domain, skill) for _, domain, skill in ranked[:k])


def top_transfer_grid(
    *,
    domain_count: int = 2,
    skill_count: int = 2,
    count_view: str = "single_domain",
) -> tuple[tuple[str, str], ...]:
    """Return a dense Domain×Skill grid with large per-cell counts.

    For the current FLASK single-domain table this selects Language/Culture ×
    Comprehension/Commonsense Understanding. The objective is to maximize the
    weakest cell first, then total rows, so every source/target cell has enough
    train, validation, and test rows.
    """

    if domain_count != 2 or skill_count != 2:
        raise ValueError("Only the 2×2 transfer grid is currently supported")
    counts = count_matrix(count_view)
    best: tuple[int, int, str, str, str, str] | None = None
    for domain_i, domain_a in enumerate(DOMAINS):
        for domain_b in DOMAINS[domain_i + 1 :]:
            for skill_i, skill_a in enumerate(SKILLS):
                for skill_b in SKILLS[skill_i + 1 :]:
                    values = [
                        counts[domain][SKILLS.index(skill)]
                        for domain in (domain_a, domain_b)
                        for skill in (skill_a, skill_b)
                    ]
                    if min(values) <= 0:
                        continue
                    candidate = (
                        min(values),
                        sum(values),
                        domain_a,
                        domain_b,
                        skill_a,
                        skill_b,
                    )
                    if best is None or candidate > best:
                        best = candidate
    if best is None:
        raise RuntimeError("No non-empty transfer grid found")
    _, _, domain_a, domain_b, skill_a, skill_b = best
    return tuple((domain, skill) for domain in (domain_a, domain_b) for skill in (skill_a, skill_b))


def split_targets(rows: int, ratios: dict[str, float] | None = None) -> dict[str, int]:
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    train = int(round(rows * ratios["train"]))
    validation = int(round(rows * ratios["validation"]))
    test = int(rows - train - validation)
    return {"target_train_rows": train, "target_validation_rows": validation, "target_test_rows": test}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_no}")
                rows.append(value)
    return rows


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")
    return path


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=_json_default))
            handle.write("\n")
    return path


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def filter_selected_rows(
    rows: Iterable[dict[str, Any]],
    cells: Sequence[tuple[str, str]],
    *,
    require_single_domain: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wanted = set(cells)
    selected: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    for row in rows:
        stats["raw_rows"] += 1
        score = integer_score(row.get("ground_truth"))
        if score is None:
            stats["dropped_noninteger_score"] += 1
            continue
        if not str(row.get("candidate_response") or "").strip():
            stats["dropped_empty_candidate_response"] += 1
            continue
        domains = tuple(str(value) for value in row.get("domain_ids") or ())
        if require_single_domain and len(domains) != 1:
            stats["dropped_non_single_domain"] += 1
            continue
        matched = [(domain, str(row.get("task_id") or "")) for domain in domains if (domain, str(row.get("task_id") or "")) in wanted]
        if not matched:
            stats["outside_selected_cells"] += 1
            continue
        b_id = str(row["b_id"])
        if b_id in seen:
            raise ValueError(f"Duplicate selected B id: {b_id}")
        seen.add(b_id)
        selected.append({**row, "ground_truth": score, "comparison_cell_id": cell_id(*matched[0])})
    selected.sort(key=lambda row: (cell_sort_key(row_cell(row)), stable_rank(row["b_id"])))
    by_cell = Counter(row["comparison_cell_id"] for row in selected)
    stats["selected_rows"] = len(selected)
    stats["selected_cells"] = len(by_cell)
    return selected, {"stats": dict(stats), "rows_by_cell": dict(sorted(by_cell.items()))}


def build_shared_group_split(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
    group_key: str = "base_id",
) -> tuple[dict[str, str], dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot split an empty selected row set")
    ratios = normalize_ratios(ratios or DEFAULT_SPLIT_RATIOS)
    split_names = tuple(ratios)
    cells = tuple(sorted({row_cell(row) for row in rows}, key=cell_sort_key))
    cell_index = {cell: index for index, cell in enumerate(cells)}
    class_index = {value: index for index, value in enumerate(CLASSES)}
    groups = sorted({str(row[group_key]) for row in rows})
    group_index = {group: index for index, group in enumerate(groups)}
    group_counts = np.zeros((len(groups), len(cells)), dtype=np.float64)
    group_label_counts = np.zeros((len(groups), len(cells), len(CLASSES)), dtype=np.float64)
    for row in rows:
        group = group_index[str(row[group_key])]
        cell = cell_index[row_cell(row)]
        label = class_index[int(row["ground_truth"])]
        group_counts[group, cell] += 1.0
        group_label_counts[group, cell, label] += 1.0
    totals = group_counts.sum(axis=0)
    label_totals = group_label_counts.sum(axis=0)
    target_counts = np.asarray([[totals[c] * ratios[name] for c in range(len(cells))] for name in split_names])
    target_label_counts = np.asarray(
        [
            [[label_totals[c, k] * ratios[name] for k in range(len(CLASSES))] for c in range(len(cells))]
            for name in split_names
        ]
    )
    current_counts = np.zeros_like(target_counts)
    current_label_counts = np.zeros_like(target_label_counts)
    assignment: dict[str, str] = {}
    order = sorted(
        range(len(groups)),
        key=lambda idx: (-float(group_counts[idx].sum()), stable_rank(f"{seed}::{groups[idx]}")),
    )
    for group_pos in order:
        best_split = min(
            split_names,
            key=lambda name: (
                _trial_score(
                    split_names.index(name),
                    group_counts[group_pos],
                    group_label_counts[group_pos],
                    current_counts,
                    current_label_counts,
                    target_counts,
                    target_label_counts,
                ),
                stable_rank(f"{seed}::{groups[group_pos]}::{name}"),
            ),
        )
        split_idx = split_names.index(best_split)
        current_counts[split_idx] += group_counts[group_pos]
        current_label_counts[split_idx] += group_label_counts[group_pos]
        assignment[groups[group_pos]] = best_split
    row_split = {str(row["b_id"]): assignment[str(row[group_key])] for row in rows}
    audit = split_audit(
        rows,
        row_split,
        cells=cells,
        ratios=ratios,
        group_key=group_key,
    )
    _validate_split(rows, row_split, cells)
    return row_split, audit


def normalize_ratios(ratios: dict[str, float]) -> dict[str, float]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Split ratios must contain exactly {SPLIT_NAMES}")
    total = sum(float(value) for value in ratios.values())
    if total <= 0.0:
        raise ValueError("Split ratios must have positive sum")
    return {name: float(ratios[name]) / total for name in SPLIT_NAMES}


def _trial_score(
    split_idx: int,
    group_counts: np.ndarray,
    group_label_counts: np.ndarray,
    current_counts: np.ndarray,
    current_label_counts: np.ndarray,
    target_counts: np.ndarray,
    target_label_counts: np.ndarray,
) -> float:
    trial_counts = current_counts.copy()
    trial_labels = current_label_counts.copy()
    trial_counts[split_idx] += group_counts
    trial_labels[split_idx] += group_label_counts
    count_error = ((trial_counts - target_counts) / np.maximum(target_counts, 1.0)) ** 2
    label_error = ((trial_labels - target_label_counts) / np.maximum(target_label_counts, 1.0)) ** 2
    return float(count_error.sum() + 0.15 * label_error.sum())


def split_audit(
    rows: Sequence[dict[str, Any]],
    row_split: dict[str, str],
    *,
    cells: Sequence[tuple[str, str]] | None = None,
    ratios: dict[str, float] | None = None,
    group_key: str = "base_id",
) -> dict[str, Any]:
    ratios = ratios or DEFAULT_SPLIT_RATIOS
    cells = tuple(cells or sorted({row_cell(row) for row in rows}, key=cell_sort_key))
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        by_split[row_split[str(row["b_id"])]].append(row)
    cell_rows: list[dict[str, Any]] = []
    for domain, skill in cells:
        for split in SPLIT_NAMES:
            subset = [row for row in by_split[split] if row_cell(row) == (domain, skill)]
            cell_total = sum(row_cell(row) == (domain, skill) for row in rows)
            cell_rows.append(
                {
                    "cell_id": cell_id(domain, skill),
                    "domain": domain,
                    "task": skill,
                    "split": split,
                    "rows": len(subset),
                    "groups": len({str(row[group_key]) for row in subset}),
                    "target_rows": cell_total * ratios[split],
                    "row_fraction": len(subset) / max(cell_total, 1),
                    "label_counts": label_counts([row["ground_truth"] for row in subset]),
                }
            )
    return {
        "split_strategy": "shared_question_group_greedy_count_label_balance_v1",
        "group_key": group_key,
        "ratios": ratios,
        "rows_total": len(rows),
        "groups_total": len({str(row[group_key]) for row in rows}),
        "rows_by_split": {name: len(by_split[name]) for name in SPLIT_NAMES},
        "groups_by_split": {name: len({str(row[group_key]) for row in by_split[name]}) for name in SPLIT_NAMES},
        "label_counts_by_split": {
            name: label_counts([row["ground_truth"] for row in by_split[name]])
            for name in SPLIT_NAMES
        },
        "cells": cell_rows,
    }


def _validate_split(
    rows: Sequence[dict[str, Any]],
    row_split: dict[str, str],
    cells: Sequence[tuple[str, str]],
) -> None:
    if set(row_split) != {str(row["b_id"]) for row in rows}:
        raise RuntimeError("Split map does not cover exactly the selected row ids")
    for split in SPLIT_NAMES:
        if not any(row_split[str(row["b_id"])] == split for row in rows):
            raise RuntimeError(f"Split {split!r} is empty")
    for cell in cells:
        for split in SPLIT_NAMES:
            if not any(row_cell(row) == cell and row_split[str(row["b_id"])] == split for row in rows):
                raise RuntimeError(f"Cell {cell_id(*cell)} has no {split} rows")


def attach_splits(rows: Sequence[dict[str, Any]], row_split: dict[str, str]) -> list[dict[str, Any]]:
    return [{**row, "split": row_split[str(row["b_id"])]} for row in rows]


def row_cell(row: dict[str, Any]) -> tuple[str, str]:
    domains = tuple(str(value) for value in row.get("domain_ids") or ())
    domain = domains[0] if domains else str(row.get("domain_id") or "")
    return domain, str(row.get("task_id") or "")


def cell_id(domain: str, task: str) -> str:
    return f"{domain}::{task}"


def parse_cell_id(value: str) -> tuple[str, str]:
    if "::" not in value:
        raise ValueError(f"Cell id must use 'Domain::Task': {value!r}")
    domain, task = value.split("::", 1)
    return domain, task


def cell_sort_key(cell: tuple[str, str]) -> tuple[int, int, str, str]:
    domain, skill = cell
    domain_pos = DOMAINS.index(domain) if domain in DOMAINS else len(DOMAINS)
    skill_pos = SKILLS.index(skill) if skill in SKILLS else len(SKILLS)
    return domain_pos, skill_pos, domain, skill


def integer_score(value: Any) -> int | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    rounded = int(round(score))
    return rounded if abs(score - rounded) < 1e-8 and 1 <= rounded <= 5 else None


def parse_direct_score(text: str) -> int | None:
    stripped = str(text).strip()
    return int(stripped) if re.fullmatch(r"[1-5]", stripped) else None


def build_direct_judge_prompt(row: dict[str, Any]) -> str:
    return DIRECT_JUDGE_TEMPLATE.format(
        rubric=str(row.get("rubric") or ""),
        instruction=str(row.get("instruction") or ""),
        reference_answer=str(row.get("reference_answer") or "(not provided)"),
        candidate_response=str(row.get("candidate_response") or ""),
    )


def apply_chat_template(row: dict[str, Any], tokenizer: Any) -> str:
    messages = [{"role": "user", "content": build_direct_judge_prompt(row)}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def head_tail(ids: Sequence[int], max_length: int) -> list[int]:
    values = list(ids)
    if len(values) <= max_length:
        return values
    head = (max_length + 1) // 2
    return [*values[:head], *values[-(max_length - head) :]]


def metrics_from_predictions(labels: Sequence[Any], predictions: Sequence[Any]) -> dict[str, Any]:
    valid_labels: list[int] = []
    valid_predictions: list[int] = []
    for label, prediction in zip(labels, predictions, strict=True):
        label_value = integer_score(label)
        pred_value = integer_score(prediction)
        if label_value is None or pred_value is None:
            continue
        valid_labels.append(label_value)
        valid_predictions.append(pred_value)
    if not valid_labels:
        return {
            "rows": len(list(labels)),
            "parsed_rows": 0,
            "parse_rate": 0.0,
            "mae": None,
            "exact_accuracy": None,
            "plus_minus_1_accuracy": None,
            "quadratic_weighted_kappa": None,
        }
    truth = np.asarray(valid_labels, dtype=int)
    pred = np.asarray(valid_predictions, dtype=int)
    qwk = cohen_kappa_score(truth, pred, labels=list(CLASSES), weights="quadratic")
    return {
        "rows": len(list(labels)),
        "parsed_rows": len(valid_labels),
        "parse_rate": len(valid_labels) / max(len(list(labels)), 1),
        "mae": float(np.mean(np.abs(pred - truth))),
        "exact_accuracy": float(np.mean(pred == truth)),
        "plus_minus_1_accuracy": float(np.mean(np.abs(pred - truth) <= 1)),
        "quadratic_weighted_kappa": float(qwk) if np.isfinite(qwk) else 0.0,
    }


def summarize_method_predictions(
    rows: Sequence[dict[str, Any]],
    predictions_by_id: dict[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        split_rows = [row for row in rows if str(row.get("split")) == split]
        metric = metrics_from_predictions(
            [row["ground_truth"] for row in split_rows],
            [predictions_by_id.get(str(row["b_id"])) for row in split_rows],
        )
        output_rows.append({"method": method, "cell_id": "ALL", "split": split, **metric})
        for cell in sorted({row_cell(row) for row in split_rows}, key=cell_sort_key):
            cell_rows = [row for row in split_rows if row_cell(row) == cell]
            metric = metrics_from_predictions(
                [row["ground_truth"] for row in cell_rows],
                [predictions_by_id.get(str(row["b_id"])) for row in cell_rows],
            )
            output_rows.append({"method": method, "cell_id": cell_id(*cell), "split": split, **metric})
    return {"method": method, "metrics": output_rows}


def label_counts(values: Iterable[Any]) -> dict[str, int]:
    counter = Counter(int(value) for value in values if integer_score(value) is not None)
    return {str(value): int(counter.get(value, 0)) for value in CLASSES}


def stable_rank(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def result_template_rows(cells: Sequence[tuple[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for target_domain, target_task in cells:
        rows.append(_blank_result_row("direct_judge", "", cell_id(target_domain, target_task), "test"))
    for method in ("classification_head", "lora"):
        for source_domain, source_task in cells:
            source = cell_id(source_domain, source_task)
            for target_domain, target_task in cells:
                rows.append(_blank_result_row(method, source, cell_id(target_domain, target_task), "test"))
    return rows


def _blank_result_row(method: str, source_cell: str, target_cell: str, split: str) -> dict[str, Any]:
    return {
        "method": method,
        "source_cell_id": source_cell,
        "target_cell_id": target_cell,
        "split": split,
        "rows": "",
        "parse_rate": "",
        "mae": "",
        "exact_accuracy": "",
        "plus_minus_1_accuracy": "",
        "quadratic_weighted_kappa": "",
        "notes": "",
    }


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
