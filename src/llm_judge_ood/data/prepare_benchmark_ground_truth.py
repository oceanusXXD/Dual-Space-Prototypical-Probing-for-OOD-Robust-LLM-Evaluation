from __future__ import annotations

import csv
import glob
import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from src.llm_judge_ood.data.hidden_contract import (
    build_prepared_record,
    file_sha256,
    template_sha256,
    write_prepared_contract,
)


def _template(dataset_name: str) -> str:
    return (
        f"Dataset: {dataset_name}\n"
        "Benchmark/domain: {benchmark_label}\n"
        "Task/rubric: {task_label}\n"
        "Rubric/check:\n{rubric}\n\n"
        "Question/Instruction:\n{instruction}\n\n"
        "Candidate response:\n{response}"
    )


LONGJUDGEBENCH_TEMPLATE_VERSION = "longjudgebench_multidim_pointwise_judge_v3"
LONGJUDGEBENCH_JUDGE_TEMPLATE = _template("LongJudgeBench")
LONGJUDGEBENCH_TEMPLATE_SHA256 = template_sha256(LONGJUDGEBENCH_JUDGE_TEMPLATE)

RUVERBENCH_TEMPLATE_VERSION = "ruverbench_rubric_check_judge_v2"
RUVERBENCH_JUDGE_TEMPLATE = _template("RuVerBench")
RUVERBENCH_TEMPLATE_SHA256 = template_sha256(RUVERBENCH_JUDGE_TEMPLATE)

BIGGEN_BENCH_TEMPLATE_VERSION = "biggen_bench_human_score_judge_v2"
BIGGEN_BENCH_JUDGE_TEMPLATE = _template("BiGGen-Bench")
BIGGEN_BENCH_TEMPLATE_SHA256 = template_sha256(BIGGEN_BENCH_JUDGE_TEMPLATE)

FLASK_TEMPLATE_VERSION = "flask_gpt4_skill_score_judge_v2"
FLASK_JUDGE_TEMPLATE = _template("FLASK")
FLASK_TEMPLATE_SHA256 = template_sha256(FLASK_JUDGE_TEMPLATE)

PROMETHEUS_TEMPLATE_VERSION = "prometheus_custom_rubric_teacher_judge_v2"
PROMETHEUS_JUDGE_TEMPLATE = _template("Prometheus")
PROMETHEUS_TEMPLATE_SHA256 = template_sha256(PROMETHEUS_JUDGE_TEMPLATE)

BENCHMARK_GT_TEMPLATE_IDENTITIES = {
    "longjudgebench": (LONGJUDGEBENCH_TEMPLATE_VERSION, LONGJUDGEBENCH_TEMPLATE_SHA256),
    "ruverbench": (RUVERBENCH_TEMPLATE_VERSION, RUVERBENCH_TEMPLATE_SHA256),
    "biggen_bench": (BIGGEN_BENCH_TEMPLATE_VERSION, BIGGEN_BENCH_TEMPLATE_SHA256),
    "flask": (FLASK_TEMPLATE_VERSION, FLASK_TEMPLATE_SHA256),
    "prometheus": (PROMETHEUS_TEMPLATE_VERSION, PROMETHEUS_TEMPLATE_SHA256),
}

_COMMON_ID_ALIASES = (
    "_parent_id",
    "sample_id",
    "id",
    "question_id",
    "example_id",
    "instance_id",
    "uid",
    "task_id",
    "uuid",
)
_COMMON_INSTRUCTION_ALIASES = (
    "instruction",
    "question",
    "prompt",
    "problem",
    "query",
    "input",
    "task",
)
_COMMON_RESPONSE_ALIASES = (
    "target_txt",
    "orig_response",
    "content",
    "candidate_response_X",
    "candidate_response",
    "response",
    "answer",
    "model_response",
    "model_output",
    "completion",
    "output",
    "prediction",
    "summary",
)
_COMMON_RUBRIC_ALIASES = (
    "orig_criteria",
    "point",
    "rubric",
    "criterion",
    "criteria",
    "constraint",
    "check",
    "requirement",
    "skill",
    "dimension",
)


_PROFILES: dict[str, dict[str, Any]] = {
    "ruverbench": {
        "source_name": "RuVerBench",
        "template": RUVERBENCH_JUDGE_TEMPLATE,
        "template_version": RUVERBENCH_TEMPLATE_VERSION,
        "template_sha256": RUVERBENCH_TEMPLATE_SHA256,
        "track": "human_gt",
        "ground_truth_source": "human_rubric_level_label",
        "label_kind": "binary",
        "nested_aliases": (
            "result.coverage_results",
            "rubrics_and_labels_Y",
            "rubrics_and_labels",
            "checks",
            "rubrics",
        ),
        "rubric_aliases": ("point",),
        "label_aliases": ("covered", "success", "passed", "label", "result"),
        "benchmark_aliases": ("benchmark", "domain", "category", "task_family"),
        "task_aliases": ("task", "rubric_category", "check_type", "category"),
        "default_benchmark": ("A", "Deep Research"),
        "default_task": ("Q", "Coverage"),
        "benchmark_map": {
            "deep_research": ("A", "Deep Research"),
            "deepresearch": ("A", "Deep Research"),
            "agentic_coding": ("B", "Agentic Coding"),
            "agentic_coding_correctness": ("B", "Agentic Coding: task correctness"),
            "agentic_coding_compliance": ("C", "Agentic Coding: process compliance"),
            "tool_compliance": ("C", "Agentic Coding: process compliance"),
        },
        "task_map": {
            "coverage": ("Q", "Coverage"),
            "correctness": ("W", "Correctness"),
            "compliance": ("E", "Compliance"),
        },
    },
    "biggen_bench": {
        "source_name": "BiGGen-Bench",
        "template": BIGGEN_BENCH_JUDGE_TEMPLATE,
        "template_version": BIGGEN_BENCH_TEMPLATE_VERSION,
        "template_sha256": BIGGEN_BENCH_TEMPLATE_SHA256,
        "track": "human_gt",
        "ground_truth_source": "human_eval_human_score",
        "label_kind": "numeric",
        "label_aliases": ("human_score", "human_reference_Y.human_score"),
        "instruction_aliases": ("input",),
        "response_aliases": ("response",),
        "rubric_aliases": ("score_rubric.criteria",),
        "source_id_aliases": ("uuid", "id"),
        "benchmark_aliases": ("benchmark", "capability", "domain", "category"),
        "task_aliases": ("capability", "task", "rubric_category", "metric"),
        "default_benchmark": ("A", "Grounding"),
        "default_task": ("Q", "Requirement/Grounding Satisfaction"),
        "benchmark_map": {
            "grounding": ("A", "Grounding"),
            "reasoning": ("B", "Reasoning"),
            "planning": ("C", "Planning"),
        },
        "task_map": {
            "grounding": ("Q", "Requirement/Grounding Satisfaction"),
            "reasoning": ("W", "Correctness"),
            "planning": ("E", "Completeness"),
            "requirement": ("Q", "Requirement/Grounding Satisfaction"),
            "correctness": ("W", "Correctness"),
            "completeness": ("E", "Completeness"),
        },
    },
    "flask": {
        "source_name": "FLASK",
        "template": FLASK_JUDGE_TEMPLATE,
        "template_version": FLASK_TEMPLATE_VERSION,
        "template_sha256": FLASK_TEMPLATE_SHA256,
        "track": "llm_teacher",
        "ground_truth_source": "gpt4_skill_level_teacher_score",
        "label_kind": "numeric",
        "nested_aliases": (
            "score",
            "skills_and_labels_Y",
            "skills_and_labels",
            "skill_scores",
            "scores",
            "gpt4_scores",
        ),
        "instruction_aliases": ("text",),
        "response_aliases": ("target_txt",),
        "rubric_aliases": ("skill", "metric"),
        "label_aliases": ("label", "value", "gpt4_score", "score"),
        "benchmark_aliases": (
            "benchmark",
            "domain_labeled",
            "domain_review",
            "domain",
            "category",
        ),
        "task_aliases": ("task", "skill", "dimension", "metric"),
        "source_id_includes_source_path": True,
        "default_benchmark": ("A", "Humanities"),
        "default_task": ("Q", "Logical Correctness"),
        "selected_task_codes": ("Q", "W", "E"),
        "benchmark_map": {
            "humanities": ("A", "Humanities"),
            "coding": ("B", "Coding"),
            "code": ("B", "Coding"),
            "math": ("C", "Math"),
            "mathematics": ("C", "Math"),
        },
        "task_map": {
            "logical_correctness": ("Q", "Logical Correctness"),
            "correctness": ("Q", "Logical Correctness"),
            "comprehension": ("W", "Comprehension"),
            "conciseness": ("E", "Conciseness"),
            "readability": ("R", "Readability"),
            "factuality": ("F", "Factuality"),
            "completeness": ("C", "Completeness"),
        },
    },
    "prometheus": {
        "source_name": "Prometheus",
        "template": PROMETHEUS_JUDGE_TEMPLATE,
        "template_version": PROMETHEUS_TEMPLATE_VERSION,
        "template_sha256": PROMETHEUS_TEMPLATE_SHA256,
        "track": "llm_teacher",
        "ground_truth_source": "gpt4_custom_rubric_teacher_score",
        "label_kind": "numeric",
        "instruction_aliases": ("orig_instruction",),
        "response_aliases": ("orig_response",),
        "rubric_aliases": ("orig_criteria",),
        "label_aliases": (
            "orig_score",
            "reference_label_Y",
            "score",
            "label",
            "rating",
            "gpt4_score",
        ),
        "benchmark_aliases": ("benchmark", "domain", "category"),
        "task_aliases": ("task", "dimension", "criterion_type", "rubric_category"),
        "default_benchmark": ("A", "Communication and Advice"),
        "default_task": ("W", "Completeness/Instruction Fulfillment"),
        "benchmark_map": {
            "communication_and_advice": ("A", "Communication and Advice"),
            "communication": ("A", "Communication and Advice"),
            "advice": ("A", "Communication and Advice"),
            "knowledge_technical_explanation": ("B", "Knowledge/Technical Explanation"),
            "technical": ("B", "Knowledge/Technical Explanation"),
            "writing_and_revision": ("C", "Writing and Revision"),
            "writing": ("C", "Writing and Revision"),
            "revision": ("C", "Writing and Revision"),
        },
        "task_map": {
            "correctness": ("Q", "Correctness/Factuality"),
            "factuality": ("Q", "Correctness/Factuality"),
            "completeness": ("W", "Completeness/Instruction Fulfillment"),
            "instruction_fulfillment": ("W", "Completeness/Instruction Fulfillment"),
            "style": ("E", "Style/Context Adaptation"),
            "context_adaptation": ("E", "Style/Context Adaptation"),
        },
    },
}


def prepare_longjudgebench(
    *,
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    records_path: str | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    seed: int = 42,
    max_records: int = 0,
    merge_records_by_id: bool = False,
    selected_task_codes: Sequence[str] | None = None,
    selected_benchmark_codes: Sequence[str] | None = None,
    benchmark_map: Mapping[str, Sequence[str]] | None = None,
    task_map: Mapping[str, Sequence[str]] | None = None,
    training_benchmark: str = "A",
    training_task: str = "Q",
) -> dict[str, Any]:
    del merge_records_by_id, benchmark_map, task_map
    paths = _expand_paths(input_paths)
    source_hashes = _source_hashes(paths, expected_sha256 or {})
    raw_records = _load_records(paths, records_path=records_path)
    if max_records and int(max_records) > 0:
        raw_records = raw_records[: int(max_records)]

    selected_tasks = {str(value) for value in (selected_task_codes or ())}
    selected_benchmarks = {str(value) for value in (selected_benchmark_codes or ())}
    benchmark_specs = {
        "deepresearch_bench": ("A", "DeepResearch-Bench"),
        "realdr": ("B", "RealDR"),
    }
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    for raw_index, raw in enumerate(raw_records):
        dataset_name = _norm_key(raw.get("dataset"))
        benchmark = benchmark_specs.get(dataset_name)
        if benchmark is None:
            _increment(skipped, "unsupported_subset")
            continue
        benchmark_code, benchmark_label = benchmark
        task_code = "Q"
        task_label = "Weighted Multi-Dimension Quality"
        if selected_tasks and task_code not in selected_tasks:
            _increment(skipped, "task_not_selected")
            continue
        if selected_benchmarks and benchmark_code not in selected_benchmarks:
            _increment(skipped, "benchmark_not_selected")
            continue

        instruction = str(raw.get("instruction") or "").strip()
        responses = raw.get("responses")
        if not isinstance(responses, list):
            _increment(skipped, "missing_responses")
            continue
        rubric, score_lookup, score_scale = _longjudgebench_subset_fields(raw, dataset_name)
        if score_lookup is None:
            _increment(skipped, "missing_reference_label")
            continue

        source_record_id = str(raw.get("id") or raw_index)
        for response_index, response_item in enumerate(responses):
            if not isinstance(response_item, Mapping):
                _increment(skipped, "invalid_response_record")
                continue
            response = str(response_item.get("content") or "").strip()
            model_name = str(response_item.get("model") or response_index)
            if not response:
                _increment(skipped, "missing_candidate_response")
                continue
            raw_label = score_lookup.get(model_name)
            if raw_label is None:
                _increment(skipped, "missing_response_label")
                continue
            try:
                label = _normalize_label(float(raw_label) / score_scale, kind="numeric")
            except (TypeError, ValueError):
                _increment(skipped, "invalid_reference_label")
                continue

            source_id = f"{dataset_name}::{source_record_id}::{model_name}"
            input_document_id = f"longjudgebench::{_slug(source_id)}"
            sample_id = f"{input_document_id}::{task_code}::{raw_index:06d}_{response_index:03d}"
            split, role, shift_type = _split_role_shift(
                benchmark_code=benchmark_code,
                task_code=task_code,
                training_benchmark=str(training_benchmark),
                training_task=str(training_task),
            )
            judge_input_text = LONGJUDGEBENCH_JUDGE_TEMPLATE.format(
                benchmark_label=f"{benchmark_code} = {benchmark_label}",
                task_label=f"{task_code} = {task_label}",
                instruction=instruction or "(not provided)",
                rubric=rubric,
                response=response,
            )
            rows.append(
                build_prepared_record(
                    dataset="longjudgebench",
                    sample_id=sample_id,
                    raw_text=response,
                    judge_input_text=judge_input_text,
                    query_id="q_weighted_multidim_quality",
                    query_text=task_label,
                    label=label,
                    split=split,
                    document_distribution_role=role,
                    audit_document_group_id=f"{benchmark_code}::{task_code}::{shift_type}",
                    document_shift_type=shift_type,
                    is_document_ood=shift_type != "id",
                    prompt_template_version=LONGJUDGEBENCH_TEMPLATE_VERSION,
                    prompt_template_sha256=LONGJUDGEBENCH_TEMPLATE_SHA256,
                    input_document_id=input_document_id,
                    base_document_id=input_document_id,
                    metadata={
                        "source_dataset": "LongJudgeBench",
                        "source_subset": dataset_name,
                        "source_path": str(raw.get("_source_path") or ""),
                        "source_row_index": raw.get("_source_row_index"),
                        "source_id": source_id,
                        "source_response_model": model_name,
                        "benchmark_code": benchmark_code,
                        "benchmark_label": benchmark_label,
                        "task_code": task_code,
                        "task_label": task_label,
                        "ground_truth_track": "human_gt",
                        "ground_truth_source": "human_multi_dimension_weighted_score",
                        "raw_label": raw_label,
                        "label_scale_normalized_to": 10,
                    },
                )
            )
    if not rows:
        raise ValueError(f"LongJudgeBench adapter produced no prepared rows; skipped={skipped}")
    return write_prepared_contract(
        output_path,
        rows,
        {
            "artifact_type": "llm_judge_ood_longjudgebench_prepared_metadata",
            "dataset_source": "LongJudgeBench",
            "source_subsets": ["deepresearch_bench", "realdr"],
            "source_paths": [str(path) for path in paths],
            "source_sha256": source_hashes,
            "ground_truth_track": "human_gt",
            "ground_truth_source": "human_multi_dimension_weighted_score",
            "raw_record_count": len(raw_records),
            "skipped_record_counts": dict(sorted(skipped.items())),
            "template_version": LONGJUDGEBENCH_TEMPLATE_VERSION,
            "template_sha256": LONGJUDGEBENCH_TEMPLATE_SHA256,
            "training_cell": {
                "benchmark": str(training_benchmark),
                "task": str(training_task),
            },
            "seed": int(seed),
        },
    )


def _longjudgebench_subset_fields(
    raw: Mapping[str, Any], dataset_name: str
) -> tuple[str, Mapping[str, Any] | None, float]:
    ground_truth = raw.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        return ("", None, 1.0)
    if dataset_name == "deepresearch_bench":
        scores = ground_truth.get("scores")
        if not isinstance(scores, Mapping):
            return ("", None, 1.0)
        dimensions = ground_truth.get("dimension_weights")
        if not isinstance(dimensions, Mapping):
            return ("", None, 1.0)
        labels = {
            str(model): values.get("weighted_total")
            for model, values in scores.items()
            if isinstance(values, Mapping) and values.get("weighted_total") is not None
        }
        rubric = _longjudgebench_weighted_rubric(
            dimensions,
            score_range="0-10 weighted report-quality score",
        )
        return (rubric, labels, 10.0)
    if dataset_name == "realdr":
        weighted_total = ground_truth.get("weighted_total")
        dimensions = ground_truth.get("weights")
        responses = raw.get("responses")
        if weighted_total is None or not isinstance(dimensions, Mapping) or not isinstance(responses, list):
            return ("", None, 1.0)
        labels = {
            str(item.get("model") or index): weighted_total
            for index, item in enumerate(responses)
            if isinstance(item, Mapping)
        }
        rubric = _longjudgebench_weighted_rubric(
            dimensions,
            score_range="0-10 weighted document-quality score",
        )
        return (rubric, labels, 1.0)
    return ("", None, 1.0)


def _longjudgebench_weighted_rubric(
    dimensions: Mapping[str, Any], *, score_range: str
) -> str:
    weights = []
    for name, weight in dimensions.items():
        try:
            weights.append(f"- {name}: weight {float(weight):g}")
        except (TypeError, ValueError):
            weights.append(f"- {name}: weight {weight}")
    return "Evaluate the candidate across the following weighted dimensions:\n" + "\n".join(
        weights
    ) + f"\nReport a {score_range}."


def prepare_ruverbench(**kwargs: Any) -> dict[str, Any]:
    return _prepare_profile("ruverbench", **kwargs)


def prepare_biggen_bench(**kwargs: Any) -> dict[str, Any]:
    return _prepare_profile("biggen_bench", **kwargs)


def prepare_flask(**kwargs: Any) -> dict[str, Any]:
    return _prepare_profile("flask", **kwargs)


def prepare_prometheus(**kwargs: Any) -> dict[str, Any]:
    return _prepare_profile("prometheus", **kwargs)


def _prepare_profile(
    profile_name: str,
    *,
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    records_path: str | None = None,
    expected_sha256: Mapping[str, str] | None = None,
    seed: int = 42,
    max_records: int = 0,
    merge_records_by_id: bool = False,
    selected_task_codes: Sequence[str] | None = None,
    selected_benchmark_codes: Sequence[str] | None = None,
    benchmark_map: Mapping[str, Sequence[str]] | None = None,
    task_map: Mapping[str, Sequence[str]] | None = None,
    training_benchmark: str = "A",
    training_task: str = "Q",
) -> dict[str, Any]:
    profile = _PROFILES[profile_name]
    paths = _expand_paths(input_paths)
    source_hashes = _source_hashes(paths, expected_sha256 or {})
    raw_records = _load_records(paths, records_path=records_path)
    if merge_records_by_id:
        raw_records = _merge_records_by_id(raw_records)
    if max_records and int(max_records) > 0:
        raw_records = raw_records[: int(max_records)]

    benchmark_mapping = _mapping(profile.get("benchmark_map", {}), benchmark_map)
    task_mapping = _mapping(profile.get("task_map", {}), task_map)
    selected_tasks = {
        str(value)
        for value in (
            selected_task_codes
            if selected_task_codes is not None
            else profile.get("selected_task_codes", ())
        )
    }
    selected_benchmarks = {str(value) for value in (selected_benchmark_codes or ())}

    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for raw_index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            _increment(skipped, "non_object_record")
            continue
        for item_index, row in enumerate(_expand_nested_items(raw, profile)):
            label = _extract_label(row, profile)
            if label is None:
                _increment(
                    skipped,
                    "invalid_reference_label"
                    if "_invalid_label" in row
                    else "missing_reference_label",
                )
                continue
            response = _first_text(row, profile, "response_aliases", _COMMON_RESPONSE_ALIASES)
            if not response:
                _increment(skipped, "missing_candidate_response")
                continue
            instruction = _first_text(
                row, profile, "instruction_aliases", _COMMON_INSTRUCTION_ALIASES
            )
            rubric = _first_text(row, profile, "rubric_aliases", _COMMON_RUBRIC_ALIASES)
            if not rubric:
                rubric = str(profile["default_task"][1])
            benchmark_code, benchmark_label = _benchmark_pair(
                row, profile, benchmark_mapping, instruction=instruction
            )
            task_code, task_label = _task_pair(row, profile, task_mapping, rubric=rubric)
            if selected_tasks and task_code not in selected_tasks:
                _increment(skipped, "task_not_selected")
                continue
            if selected_benchmarks and benchmark_code not in selected_benchmarks:
                _increment(skipped, "benchmark_not_selected")
                continue

            source_id = _source_id(
                row,
                instruction=instruction,
                response=response,
                include_source_path=bool(profile.get("source_id_includes_source_path", False)),
                nested_items_are_documents=bool(profile.get("nested_items_are_documents", False)),
                id_aliases=tuple(profile.get("source_id_aliases", _COMMON_ID_ALIASES)),
            )
            input_document_id = f"{profile_name}::{_slug(source_id)}"
            sample_id = (
                f"{profile_name}::{_slug(source_id)}::{_slug(task_code)}::"
                f"{raw_index:06d}_{item_index:03d}"
            )
            split, role, shift_type = _split_role_shift(
                benchmark_code=benchmark_code,
                task_code=task_code,
                training_benchmark=str(training_benchmark),
                training_task=str(training_task),
            )
            judge_input_text = str(profile["template"]).format(
                benchmark_label=f"{benchmark_code} = {benchmark_label}",
                task_label=f"{task_code} = {task_label}",
                instruction=instruction or "(not provided)",
                rubric=rubric,
                response=response,
            )
            rows.append(
                build_prepared_record(
                    dataset=profile_name,
                    sample_id=sample_id,
                    raw_text=response,
                    judge_input_text=judge_input_text,
                    query_id=f"{task_code.lower()}_{_slug(task_label, max_length=48)}",
                    query_text=str(task_label),
                    label=label,
                    split=split,
                    document_distribution_role=role,
                    audit_document_group_id=f"{benchmark_code}::{task_code}::{shift_type}",
                    document_shift_type=shift_type,
                    is_document_ood=shift_type != "id",
                    prompt_template_version=str(profile["template_version"]),
                    prompt_template_sha256=str(profile["template_sha256"]),
                    input_document_id=input_document_id,
                    base_document_id=input_document_id,
                    metadata={
                        "source_dataset": str(profile["source_name"]),
                        "source_path": str(row.get("_source_path") or ""),
                        "source_row_index": row.get("_source_row_index"),
                        "source_id": source_id,
                        "benchmark_code": benchmark_code,
                        "benchmark_label": benchmark_label,
                        "task_code": task_code,
                        "task_label": task_label,
                        "ground_truth_track": str(profile["track"]),
                        "ground_truth_source": str(profile["ground_truth_source"]),
                        "raw_label": row.get("_raw_label"),
                    },
                )
            )
    if not rows:
        raise ValueError(
            f"{profile['source_name']} adapter produced no prepared rows; skipped={skipped}"
        )
    return write_prepared_contract(
        output_path,
        rows,
        {
            "artifact_type": f"llm_judge_ood_{profile_name}_prepared_metadata",
            "dataset_source": str(profile["source_name"]),
            "source_paths": [str(path) for path in paths],
            "source_sha256": source_hashes,
            "ground_truth_track": str(profile["track"]),
            "ground_truth_source": str(profile["ground_truth_source"]),
            "raw_record_count": len(raw_records),
            "skipped_record_counts": dict(sorted(skipped.items())),
            "template_version": str(profile["template_version"]),
            "template_sha256": str(profile["template_sha256"]),
            "training_cell": {
                "benchmark": str(training_benchmark),
                "task": str(training_task),
            },
            "seed": int(seed),
        },
    )


def _expand_paths(patterns: Sequence[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        raw = str(pattern)
        matches = sorted(glob.glob(raw)) if any(char in raw for char in "*?[]") else [raw]
        if not matches:
            raise FileNotFoundError(f"No files matched {raw}")
        for match in matches:
            path = Path(match)
            if not path.is_file():
                raise FileNotFoundError(path)
            paths.append(path)
    if not paths:
        raise ValueError("At least one input path is required")
    return paths


def _source_hashes(paths: Sequence[Path], expected: Mapping[str, str]) -> dict[str, str]:
    actual = {str(path): file_sha256(path) for path in paths}
    for key, expected_hash in expected.items():
        matches = [value for path, value in actual.items() if path == key or Path(path).name == key]
        if not matches:
            raise ValueError(f"Configured checksum key {key!r} did not match an input path")
        if str(expected_hash) not in matches:
            raise ValueError(f"Checksum mismatch for {key}: expected {expected_hash}, got {matches}")
    return actual


def _load_records(paths: Sequence[Path], *, records_path: str | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        loaded = _load_one_path(path, records_path=records_path)
        for index, record in enumerate(loaded):
            item = dict(record)
            item.setdefault("_source_path", str(path))
            item.setdefault("_source_row_index", index)
            records.append(item)
    return records


def _load_one_path(path: Path, *, records_path: str | None) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        selected = _lookup(payload, records_path) if records_path else payload
        return _records_from_json(selected)
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    raise ValueError(f"Unsupported benchmark source format: {path}")


def _merge_records_by_id(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in records:
        key = _first_value(row, _COMMON_ID_ALIASES)
        if key in (None, ""):
            anonymous.append(dict(row))
            continue
        text_key = str(key)
        target = merged.setdefault(text_key, {})
        for field, value in row.items():
            if value in (None, ""):
                continue
            current = target.get(field)
            if current in (None, ""):
                target[field] = value
            elif current == value:
                continue
            elif isinstance(current, list) and isinstance(value, list):
                target[field] = [*current, *value]
            elif field.startswith("_source_"):
                target[field] = f"{current};{value}"
            else:
                target.setdefault(f"{field}__additional", value)
        target.setdefault("_parent_id", text_key)
    return [*merged.values(), *anonymous]


def _records_from_json(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("records", "data", "examples", "items", "rows", "human_eval"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [dict(item) for item in nested if isinstance(item, dict)]
        rows: list[dict[str, Any]] = []
        for key, nested in value.items():
            if isinstance(nested, list):
                for item in nested:
                    if isinstance(item, dict):
                        row = dict(item)
                        row.setdefault("source_split", str(key))
                        rows.append(row)
        if rows:
            return rows
    raise ValueError("JSON benchmark source must contain a list of record objects")


def _expand_nested_items(row: Mapping[str, Any], profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    for alias in profile.get("nested_aliases", ()):
        nested = _lookup(row, str(alias))
        if isinstance(nested, list):
            expanded: list[dict[str, Any]] = []
            parent_id = _first_value(row, _COMMON_ID_ALIASES)
            for index, item in enumerate(nested):
                merged = dict(row)
                merged["_parent_id"] = parent_id
                merged["_nested_index"] = index
                if isinstance(item, dict):
                    merged.update(item)
                elif item not in (None, ""):
                    merged["rubric"] = str(item)
                expanded.append(merged)
            return expanded
        if isinstance(nested, dict):
            expanded = []
            parent_id = _first_value(row, _COMMON_ID_ALIASES)
            for index, (key, value) in enumerate(nested.items()):
                merged = dict(row)
                merged["_parent_id"] = parent_id
                merged["_nested_index"] = index
                if isinstance(value, dict):
                    merged.update(value)
                    merged.setdefault("rubric", str(key))
                    merged.setdefault("skill", str(key))
                else:
                    merged["rubric"] = str(key)
                    merged["skill"] = str(key)
                    merged["label"] = value
                expanded.append(merged)
            return expanded
    return [dict(row)]


def _extract_label(row: dict[str, Any], profile: Mapping[str, Any]) -> Any | None:
    for alias in profile.get("label_aliases", ()):
        value = _lookup(row, str(alias))
        if value not in (None, ""):
            row["_raw_label"] = value
            try:
                return _normalize_label(value, kind=str(profile["label_kind"]))
            except (TypeError, ValueError):
                row["_invalid_label"] = value
                return None
    return None


def _normalize_label(value: Any, *, kind: str) -> Any:
    if kind == "numeric":
        if isinstance(value, bool):
            return int(value)
        number = float(str(value).strip())
        return int(number) if number.is_integer() else number
    if kind == "binary":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)) and float(value) in {0.0, 1.0}:
            return int(value)
        normalized = _norm_key(value)
        true_values = {"true", "yes", "y", "1", "success", "pass", "passed", "covered"}
        false_values = {"false", "no", "n", "0", "fail", "failed", "not_covered"}
        if normalized in true_values:
            return 1
        if normalized in false_values:
            return 0
    raise ValueError(f"Unsupported or invalid {kind} reference label: {value!r}")


def _benchmark_pair(
    row: Mapping[str, Any],
    profile: Mapping[str, Any],
    mapping: Mapping[str, tuple[str, str]],
    *,
    instruction: str,
) -> tuple[str, str]:
    raw = _first_value(row, profile.get("benchmark_aliases", ()))
    if raw not in (None, ""):
        found = mapping.get(_norm_key(_first_scalar(raw)))
        if found:
            return found
        if profile["source_name"] == "BiGGen-Bench":
            return ("U", f"Unmapped capability: {_first_scalar(raw)}")
    if profile["source_name"] == "Prometheus":
        inferred = _infer_prometheus_benchmark(instruction)
        if inferred:
            return inferred
    return tuple(profile["default_benchmark"])  # type: ignore[return-value]


def _task_pair(
    row: Mapping[str, Any],
    profile: Mapping[str, Any],
    mapping: Mapping[str, tuple[str, str]],
    *,
    rubric: str,
) -> tuple[str, str]:
    raw = _first_value(row, profile.get("task_aliases", ()))
    if raw not in (None, ""):
        found = mapping.get(_norm_key(_first_scalar(raw)))
        if found:
            return found
    found = mapping.get(_norm_key(rubric))
    if found:
        return found
    if profile["source_name"] == "Prometheus":
        return _infer_prometheus_task(rubric)
    return tuple(profile["default_task"])  # type: ignore[return-value]


def _mapping(
    defaults: Mapping[str, Sequence[str]],
    overrides: Mapping[str, Sequence[str]] | None,
) -> dict[str, tuple[str, str]]:
    merged = {str(key): tuple(value) for key, value in defaults.items()}
    for key, value in dict(overrides or {}).items():
        if len(value) != 2:
            raise ValueError(f"Mapping override for {key!r} must be [code, label]")
        merged[str(key)] = (str(value[0]), str(value[1]))
    return {_norm_key(key): (str(value[0]), str(value[1])) for key, value in merged.items()}


def _infer_prometheus_benchmark(instruction: str) -> tuple[str, str] | None:
    normalized = _norm_key(instruction)
    if any(key in normalized for key in ("email", "advice", "meeting", "culture", "etiquette")):
        return ("A", "Communication and Advice")
    if any(key in normalized for key in ("code", "technical", "explain", "knowledge", "science")):
        return ("B", "Knowledge/Technical Explanation")
    if any(key in normalized for key in ("write", "rewrite", "revise", "story", "essay")):
        return ("C", "Writing and Revision")
    return None


def _infer_prometheus_task(rubric: str) -> tuple[str, str]:
    normalized = _norm_key(rubric)
    if any(key in normalized for key in ("correct", "factual", "accurate")):
        return ("Q", "Correctness/Factuality")
    if any(key in normalized for key in ("complete", "fulfill", "requirement")):
        return ("W", "Completeness/Instruction Fulfillment")
    if any(key in normalized for key in ("style", "tone", "context", "adapt")):
        return ("E", "Style/Context Adaptation")
    return ("W", "Completeness/Instruction Fulfillment")


def _split_role_shift(
    *,
    benchmark_code: str,
    task_code: str,
    training_benchmark: str,
    training_task: str,
) -> tuple[str, str, str]:
    same_benchmark = benchmark_code == training_benchmark
    same_task = task_code == training_task
    if same_benchmark and same_task:
        return ("training_train", "training", "id")
    if not same_benchmark and same_task:
        return ("benchmark_domain_shift", "benchmark", "domain")
    if same_benchmark and not same_task:
        return ("benchmark_rubric_shift", "benchmark", "rubric")
    return ("benchmark_domain_rubric_shift", "benchmark", "domain_rubric")


def _first_text(
    row: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_key: str,
    defaults: Sequence[str],
) -> str:
    aliases = tuple(profile.get(profile_key, ())) + tuple(defaults)
    value = _first_value(row, aliases)
    return "" if value in (None, "") else str(value).strip()


def _first_value(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for alias in aliases:
        value = _lookup(row, str(alias))
        if value not in (None, ""):
            return value
    return None


def _first_scalar(value: Any) -> Any:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return next((item for item in value if item not in (None, "")), "")
    return value


def _lookup(value: Any, path: str | None) -> Any:
    if not path:
        return value
    if isinstance(value, Mapping) and path in value:
        return value[path]
    current = value
    for part in str(path).split("."):
        if isinstance(current, Mapping):
            current = current.get(part)
        elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
        if current is None:
            return None
    return current


def _source_id(
    row: Mapping[str, Any],
    *,
    instruction: str,
    response: str,
    include_source_path: bool = False,
    nested_items_are_documents: bool = False,
    id_aliases: Sequence[str] = _COMMON_ID_ALIASES,
) -> str:
    value = _first_value(row, id_aliases)
    if value not in (None, ""):
        source_id = str(value)
    else:
        digest = sha256(f"{instruction}\0{response}".encode("utf-8")).hexdigest()[:16]
        source_id = f"row_{digest}"
    if include_source_path:
        source_path = str(row.get("_source_path") or "")
        if source_path:
            source_id = f"{Path(source_path.split(';', 1)[0]).stem}::{source_id}"
    if nested_items_are_documents and row.get("_nested_index") is not None:
        source_id = f"{source_id}::candidate_{int(row['_nested_index'])}"
    return source_id


def _slug(value: Any, *, max_length: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    if not slug:
        slug = "unassigned"
    if len(slug) <= max_length:
        return slug
    digest = sha256(slug.encode("utf-8")).hexdigest()[:10]
    return f"{slug[: max_length - 11]}_{digest}"


def _norm_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _increment(counter: dict[str, int], key: str) -> None:
    counter[key] = counter.get(key, 0) + 1
