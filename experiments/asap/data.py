from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from src.common.io import write_json, write_jsonl
from experiments.asap.prompts import (
    ASAP_JUDGE_TEMPLATE_VERSION,
    build_asap_judge_input,
    prompt_catalog_metadata,
)


ASAP_SOURCE_PROMPTS = (1, 2)
ASAP_NEAR_PROMPTS = (3, 4)
ASAP_FAR_PROMPTS = (7, 8)
ASAP_USED_PROMPTS = ASAP_SOURCE_PROMPTS + ASAP_NEAR_PROMPTS + ASAP_FAR_PROMPTS
ASAP_QUERY_ID = "overall_quality"
ASAP_QUERY_TEXT = "Rate the overall quality of this essay on a 1-5 scale."

# Official ASAP-AES rubric ranges.  These are fixed before any document split
# is created, so score conversion cannot inspect development, deployment, or
# future labels.  Resolved scores and individual raters have different raw
# units for several prompts (for example, prompts 1 and 7 use a two-rater
# aggregate), but each has an explicit rubric range on the common semantic
# quality scale.
ASAP_OFFICIAL_SCORE_RANGES = {
    1: {"resolved": (2.0, 12.0), "rater1": (1.0, 6.0), "rater2": (1.0, 6.0)},
    2: {"resolved": (1.0, 6.0), "rater1": (1.0, 6.0), "rater2": (1.0, 6.0)},
    3: {"resolved": (0.0, 3.0), "rater1": (0.0, 3.0), "rater2": (0.0, 3.0)},
    4: {"resolved": (0.0, 3.0), "rater1": (0.0, 3.0), "rater2": (0.0, 3.0)},
    # Prompt 7 uses four 0–3 traits, with Ideas doubled: 0–12 per rater and
    # 0–24 after resolution.  Prompt 8's reported composite is 5–30 per
    # rater and 10–60 after resolution.  These are score-formula ranges, not
    # empirical min/max values learned from the TSV.
    7: {"resolved": (0.0, 24.0), "rater1": (0.0, 12.0), "rater2": (0.0, 12.0)},
    8: {"resolved": (10.0, 60.0), "rater1": (5.0, 30.0), "rater2": (5.0, 30.0)},
}

SOURCE_SPLIT_FRACTIONS = {
    # A formal 50-document candidate requires at least 20 independent
    # calibration windows.  With ASAP prompts 1/2, the former 15% allocation
    # could not supply that audit capacity. Keep calibration at 29%, and split
    # the former 33% fit pool so drift tests never reuse in-sample residuals as
    # their two-sample reference distribution.
    "training_train": 0.23,
    "training_drift_reference": 0.10,
    "training_calibration": 0.29,
    "training_validation": 0.08,
    "training_guard": 0.06,
    "training_test": 0.06,
    "benchmark_test": 0.06,
    "deployment_stream": 0.12,
}

# Target documents are separated before any lifecycle evaluation.  The 20/80
# development/deployment division and the deployment 40/15/15/15/15 division
# match the repository's established document-level data contract.
TARGET_SPLIT_FRACTIONS = {
    "development": 0.15,
    "benchmark_test": 0.15,
    "deployment_stream": 0.28,
    "deployment_ood_evaluation": 0.105,
    "deployment_adapt": 0.105,
    "deployment_gate": 0.105,
    "deployment_future_test": 0.105,
}

DEFAULT_FLOW_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_GRADUAL_PROPORTIONS = tuple(float(value) / 10.0 for value in range(11))

_COLUMN_ALIASES = {
    "essay_id": ("essay_id", "id", "document_id"),
    "prompt_id": ("essay_set", "prompt_id", "prompt", "essay_prompt"),
    "essay_text": ("essay", "essay_text", "text", "document_text"),
    "rater1": ("rater1_domain1", "rater1_score", "rater1", "rater1_overall"),
    "rater2": ("rater2_domain1", "rater2_score", "rater2", "rater2_overall"),
    "resolved": ("domain1_score", "resolved_score", "resolved", "overall_score", "score"),
}


@dataclass(frozen=True)
class ASAPFlowConfig:
    window_size: int = 50
    abrupt_pre_shift_windows: int = 3
    abrupt_post_shift_windows: int = 5
    gradual_proportions: tuple[float, ...] = DEFAULT_GRADUAL_PROPORTIONS
    harmless_windows: int = 8
    seeds: tuple[int, ...] = DEFAULT_FLOW_SEEDS

    def __post_init__(self) -> None:
        if int(self.window_size) < 1:
            raise ValueError("ASAP flow window_size must be positive")
        if int(self.abrupt_pre_shift_windows) < 1 or int(self.abrupt_post_shift_windows) < 1:
            raise ValueError("ASAP abrupt flow requires pre- and post-shift windows")
        if int(self.harmless_windows) < 1:
            raise ValueError("ASAP harmless flow requires at least one window")
        if not self.gradual_proportions:
            raise ValueError("ASAP gradual flow proportions cannot be empty")
        if any(not 0.0 <= float(value) <= 1.0 for value in self.gradual_proportions):
            raise ValueError("ASAP gradual flow proportions must be in [0, 1]")
        if tuple(sorted(float(value) for value in self.gradual_proportions)) != tuple(
            float(value) for value in self.gradual_proportions
        ):
            raise ValueError("ASAP gradual flow proportions must be non-decreasing")
        if len(set(int(seed) for seed in self.seeds)) != len(self.seeds):
            raise ValueError("ASAP flow seeds must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def prepare_asap_rows_from_dataset(
    dataset_rows: Iterable[Mapping[str, Any]],
    *,
    seed: int = 42,
    source_split_fractions: Mapping[str, float] | None = None,
    target_split_fractions: Mapping[str, float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Normalize official ASAP-AES rows into the document OOD schema.

    Raw scores are converted with fixed, prompt-specific official rubric
    ranges.  The range mapping is applied separately to resolved, rater-1,
    and rater-2 because those columns do not always share raw units.  No label
    statistics are estimated from any split, so deployment/future labels
    cannot affect the mapping or the human-error baseline.
    """

    source_fractions = _validate_fractions(
        source_split_fractions or SOURCE_SPLIT_FRACTIONS,
        expected_names=tuple(SOURCE_SPLIT_FRACTIONS),
        name="source_split_fractions",
    )
    target_fractions = _validate_fractions(
        target_split_fractions or TARGET_SPLIT_FRACTIONS,
        expected_names=tuple(TARGET_SPLIT_FRACTIONS),
        name="target_split_fractions",
    )
    normalized = [_normalize_raw_row(dict(row)) for row in dataset_rows]
    used = [row for row in normalized if int(row["prompt_id"]) in ASAP_USED_PROMPTS]
    if not used:
        raise ValueError("No ASAP rows from prompts 1, 2, 3, 4, 7, or 8 were found")
    essay_ids = [str(row["essay_id"]) for row in used]
    if len(set(essay_ids)) != len(essay_ids):
        duplicates = sorted(key for key, count in Counter(essay_ids).items() if count > 1)
        raise ValueError(f"ASAP essay ids must be unique; duplicates include {duplicates[:5]}")

    score_mappings = _fit_prompt_score_mappings(used)
    prompt_metadata = prompt_catalog_metadata()
    source_assignments = _stable_partition(
        [str(row["essay_id"]) for row in used if int(row["prompt_id"]) in ASAP_SOURCE_PROMPTS],
        source_fractions,
        seed=int(seed),
    )
    target_assignments: dict[str, str] = {}
    # Partition near and far independently so both shifts are represented in
    # every lifecycle pool whenever their sample counts permit it.
    for shift_name, prompt_ids in (("near", ASAP_NEAR_PROMPTS), ("far", ASAP_FAR_PROMPTS)):
        assignments = _stable_partition(
            [str(row["essay_id"]) for row in used if int(row["prompt_id"]) in prompt_ids],
            target_fractions,
            seed=int(seed) + (101 if shift_name == "near" else 202),
        )
        target_assignments.update(assignments)

    stream_ids = [
        str(row["essay_id"])
        for row in used
        if (
            source_assignments.get(str(row["essay_id"])) == "deployment_stream"
            or target_assignments.get(str(row["essay_id"])) == "deployment_stream"
        )
    ]
    stream_order = {
        essay_id: index
        for index, essay_id in enumerate(_stable_rank(stream_ids, seed=int(seed) + 303))
    }

    prepared: list[dict[str, Any]] = []
    for row in used:
        prompt_id = int(row["prompt_id"])
        essay_id = str(row["essay_id"])
        mapping = score_mappings[str(prompt_id)]
        label = _apply_score_mapping(float(row["resolved"]), mapping, field="resolved")
        rater1_label = _apply_score_mapping(float(row["rater1"]), mapping, field="rater1")
        rater2_label = _apply_score_mapping(float(row["rater2"]), mapping, field="rater2")
        if prompt_id in ASAP_SOURCE_PROMPTS:
            shift_type = "id"
            split = source_assignments[essay_id]
            role = (
                "deployment"
                if split == "deployment_stream"
                else "benchmark"
                if split == "benchmark_test"
                else "training"
            )
            selection_role = (
                "deployment_blind"
                if role == "deployment"
                else "independent_confirmation"
                if role == "benchmark"
                else "training_fit" if split == "training_train" else "training_holdout"
            )
        else:
            shift_type = "near" if prompt_id in ASAP_NEAR_PROMPTS else "far"
            split = target_assignments[essay_id]
            role = (
                "development"
                if split == "development"
                else "benchmark"
                if split == "benchmark_test"
                else "deployment"
            )
            selection_role = (
                "development_only"
                if role == "development"
                else "independent_confirmation"
                if role == "benchmark"
                else "deployment_blind"
            )
        document_id = f"asap::{essay_id}"
        judge_input = build_asap_judge_input(prompt_id=prompt_id, essay_text=str(row["essay_text"]))
        prepared.append(
            {
                "sample_id": document_id,
                "id": document_id,
                "dataset": "asap_aes",
                "query_id": ASAP_QUERY_ID,
                "query_text": ASAP_QUERY_TEXT,
                "document_text": str(row["essay_text"]),
                # B/Judge space: prompt-specific frozen task/rubric plus essay.
                "judge_input_text": judge_input,
                "label": int(label),
                "groundtruth": int(label),
                "split": split,
                "judge_provenance_id": "asap_aes_overall_judge",
                "base_document_id": document_id,
                # A space: the raw ASAP essay only, with no prompt/rubric/score.
                "input_document_id": document_id,
                "input_document_text": str(row["essay_text"]),
                "document_distribution_role": role,
                "audit_document_group_id": f"{shift_type}_prompt_{prompt_id}",
                "document_shift_type": shift_type,
                "shift_taxonomy": "id_h0" if shift_type == "id" else "cross_prompt_compound",
                "shift_construction": (
                    "same_prompt_independent_document"
                    if shift_type == "id"
                    else "controlled_cross_prompt_compound_shift"
                ),
                "allowed_shift_claim": (
                    "ID/H0 only" if shift_type == "id" else "controlled cross-prompt compound document-domain shift"
                ),
                "prohibited_shift_claim": (
                    "not an OOD target" if shift_type == "id" else "not a pure covariate shift and does not assert P(Y|X) invariance"
                ),
                "is_document_ood": shift_type != "id",
                "asap_essay_id": essay_id,
                "asap_prompt_id": prompt_id,
                "raw_score": float(row["resolved"]),
                "raw_resolved_score": float(row["resolved"]),
                "raw_rater1_score": float(row["rater1"]),
                "raw_rater2_score": float(row["rater2"]),
                "resolved_score": int(label),
                "rater1_score": int(rater1_label),
                "rater2_score": int(rater2_label),
                "rater_scores": [int(rater1_label), int(rater2_label)],
                "score_mapping_prompt_id": prompt_id,
                "score_mapping_method": "official_rubric_range_equal_width_buckets",
                "score_mapping_source": "asap_official_rubric_ranges_pre_split",
                # ASAP-AES has no observed arrival batches.  An essay is
                # therefore the smallest independent permutation unit; prompt
                # provenance stays in audit_document_group_id only.
                "arrival_batch_id": document_id,
                "selection_role": selection_role,
                "stream_order": stream_order.get(essay_id),
                "prompt_template_version": ASAP_JUDGE_TEMPLATE_VERSION,
                "prompt_template_sha256": prompt_metadata["template_sha256"],
                "prompt_rubric_catalog_version": prompt_metadata["catalog_version"],
                "prompt_rubric_catalog_sha256": prompt_metadata["catalog_sha256"],
            }
        )

    mapping_audit = _validate_mapped_score_distributions(prepared)
    metadata = _asap_metadata(
        raw_rows=normalized,
        prepared=prepared,
        score_mappings=score_mappings,
        seed=int(seed),
        source_split_fractions=source_fractions,
        target_split_fractions=target_fractions,
        mapping_audit=mapping_audit,
        prompt_metadata=prompt_metadata,
    )
    return prepared, metadata


def build_asap_deployment_flows(
    prepared_rows: Sequence[Mapping[str, Any]],
    *,
    config: ASAPFlowConfig | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Build abrupt, gradual, and harmless deployment streams for five seeds.

    Flow rows are simulation events with unique event ids.  `lineage_document_id`
    preserves the original ASAP identity, making template reuse explicit and
    allowing downstream audits to distinguish events from independent essays.
    """

    flow_config = config or ASAPFlowConfig()
    source_pool = [
        dict(row)
        for row in prepared_rows
        if str(row.get("document_shift_type")) == "id"
        and str(row.get("split")) == "deployment_stream"
    ]
    target_pools = {
        "near": [
            dict(row)
            for row in prepared_rows
            if str(row.get("document_shift_type")) == "near"
            and str(row.get("split")) == "deployment_stream"
        ],
        "far": [
            dict(row)
            for row in prepared_rows
            if str(row.get("document_shift_type")) == "far"
            and str(row.get("split")) == "deployment_stream"
        ],
    }
    if not source_pool or any(not values for values in target_pools.values()):
        raise ValueError(
            "ASAP deployment flows require non-empty ID/near/far deployment_stream pools"
        )

    flows: dict[str, list[dict[str, Any]]] = {}
    flow_summaries: list[dict[str, Any]] = []
    for seed in flow_config.seeds:
        for shift_type, target_pool in target_pools.items():
            abrupt_proportions = (
                (0.0,) * int(flow_config.abrupt_pre_shift_windows)
                + (1.0,) * int(flow_config.abrupt_post_shift_windows)
            )
            for flow_type, proportions in (
                ("abrupt", abrupt_proportions),
                ("gradual", flow_config.gradual_proportions),
            ):
                flow_id = f"{flow_type}_{shift_type}_seed_{int(seed)}"
                events = _mixture_flow(
                    source_pool=source_pool,
                    target_pool=target_pool,
                    flow_id=flow_id,
                    flow_type=flow_type,
                    target_shift_type=shift_type,
                    target_proportions=proportions,
                    window_size=int(flow_config.window_size),
                    seed=int(seed),
                )
                flows[flow_id] = events
                flow_summaries.append(_flow_summary(flow_id, events))

        harmless_id = f"harmless_seed_{int(seed)}"
        harmless_events = _harmless_flow(
            source_pool=source_pool,
            flow_id=harmless_id,
            window_size=int(flow_config.window_size),
            window_count=int(flow_config.harmless_windows),
            seed=int(seed),
        )
        flows[harmless_id] = harmless_events
        flow_summaries.append(_flow_summary(harmless_id, harmless_events))

    metadata = {
        "artifact_type": "llm_judge_ood_asap_deployment_flows_metadata",
        "config": flow_config.to_dict(),
        "flow_count": len(flows),
        "flow_types": ["abrupt", "gradual", "harmless"],
        "shift_types": ["near", "far"],
        "seeds": [int(seed) for seed in flow_config.seeds],
        "event_identity_contract": "unique_simulation_event_id_with_explicit_asap_lineage",
        "sampling_contract": "deterministic_seeded_without_replacement_within_each_flow",
        "pool_scope": "deployment_stream_only; development/probe/adapt/gate/future excluded",
        "permutation_block_contract": "one_simulation_event_per_arrival_block",
        "flow_summaries": flow_summaries,
    }
    return flows, metadata


def load_asap_table(path: str | Path, *, delimiter: str | None = None) -> list[dict[str, Any]]:
    input_path = Path(path)
    payload = input_path.read_bytes()
    try:
        text = payload.decode("utf-8-sig")
        encoding = "utf-8-sig"
    except UnicodeDecodeError:
        text = payload.decode("latin-1")
        encoding = "latin-1"
    separator = delimiter or ("\t" if input_path.suffix.lower() in {".tsv", ".txt"} else ",")
    rows = [dict(row) for row in csv.DictReader(io.StringIO(text), delimiter=separator)]
    if not rows:
        raise ValueError(f"ASAP input table is empty: {input_path}")
    for row in rows:
        row["_source_encoding"] = encoding
    return rows


def write_asap_prepared(
    *,
    input_path: str | Path,
    output_path: str | Path,
    seed: int = 42,
    write_flows: bool = False,
    flow_config: ASAPFlowConfig | None = None,
) -> dict[str, Any]:
    raw_rows = load_asap_table(input_path)
    prepared, metadata = prepare_asap_rows_from_dataset(raw_rows, seed=int(seed))
    output = Path(output_path)
    write_jsonl(output, prepared)
    benchmark_sidecars = _write_asap_benchmark_sidecars(
        prepared_rows=prepared,
        output=output,
        prompt_metadata=prompt_catalog_metadata(),
    )
    metadata.update(
        {
            "input_path": str(input_path),
            "output_path": str(output),
            "source_sha256": _sha256(Path(input_path)),
            "prepared_sha256": _sha256(output),
            "benchmark_sidecars": benchmark_sidecars,
        }
    )
    if write_flows:
        flows, flow_metadata = build_asap_deployment_flows(prepared, config=flow_config)
        flow_root = output.parent / f"{output.stem}_flows"
        flow_paths: dict[str, str] = {}
        for flow_id, rows in flows.items():
            flow_path = flow_root / f"{flow_id}.jsonl"
            write_jsonl(flow_path, rows)
            flow_paths[flow_id] = str(flow_path)
        flow_metadata["flow_paths"] = flow_paths
        write_json(flow_root / "metadata.json", flow_metadata)
        metadata["deployment_flows"] = flow_metadata
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def _write_asap_benchmark_sidecars(
    *,
    prepared_rows: Sequence[Mapping[str, Any]],
    output: Path,
    prompt_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Write paired benchmark controls without changing base labels.

    The ordinary ``benchmark_test`` rows remain the independent detector
    confirmation pool. The sidecars are paired transformations of its ID/H0
    documents and are evaluated only after model and detector selection.
    """

    benchmark_rows = [
        dict(row)
        for row in prepared_rows
        if str(row.get("split")) == "benchmark_test"
        and str(row.get("document_shift_type")) == "id"
    ]
    within_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    scenario_counts: dict[str, int] = {
        "detector_confirmation_id_controls": len(benchmark_rows),
        "within_prompt": 0,
        "semantic_task": 0,
    }
    for row in benchmark_rows:
        transformed = _format_only_within_prompt_transform(str(row["input_document_text"]))
        if re.findall(r"\S+", transformed) != re.findall(r"\S+", str(row["input_document_text"])):
            raise RuntimeError("Within-prompt benchmark transform changed non-whitespace content")
        variant = dict(row)
        variant_id = f"asap-within-format-v1::{row['asap_essay_id']}"
        variant.update(
            {
                "sample_id": variant_id,
                "id": variant_id,
                "base_document_id": variant_id,
                "input_document_id": variant_id,
                "document_text": transformed,
                "input_document_text": transformed,
                "judge_input_text": build_asap_judge_input(
                    prompt_id=int(row["asap_prompt_id"]), essay_text=transformed
                ),
                "split": "benchmark_within_prompt",
                "document_distribution_role": "benchmark",
                "selection_role": "independent_confirmation_auxiliary",
                "audit_document_group_id": f"within_prompt_format_prompt_{int(row['asap_prompt_id'])}",
                "document_shift_type": "within_prompt_covariate",
                "shift_taxonomy": "within_prompt_covariate",
                "shift_construction": "deterministic_whitespace_and_paragraph_layout_only",
                "is_document_ood": True,
                "lineage_document_id": str(row["input_document_id"]),
                "lineage_reuse_is_simulation_only": False,
                "label_preservation_evidence": "non_whitespace_token_sequence_exactly_equal",
                "prompt_template_version": ASAP_JUDGE_TEMPLATE_VERSION,
                "prompt_template_sha256": prompt_metadata["template_sha256"],
            }
        )
        within_rows.append(variant)
        scenario_counts["within_prompt"] += 1
        semantic = dict(row)
        semantic_id = f"asap-semantic-task-v1::{row['asap_essay_id']}"
        semantic.update(
            {
                "sample_id": semantic_id,
                "id": semantic_id,
                "base_document_id": semantic_id,
                "input_document_id": semantic_id,
                "query_id": "semantic_task_shift_no_label",
                "query_text": "Semantic/task-shift fail-closed diagnostic; no compatible ASAP label exists.",
                "judge_input_text": _semantic_task_shift_input(
                    prompt_id=int(row["asap_prompt_id"]), essay_text=str(row["input_document_text"])
                ),
                "label": None,
                "groundtruth": None,
                "rater_scores": None,
                "resolved_score": None,
                "rater1_score": None,
                "rater2_score": None,
                "split": "diagnostic_semantic_task",
                "document_distribution_role": "diagnostic",
                "selection_role": "diagnostic_only_no_selection",
                "audit_document_group_id": f"semantic_task_prompt_{int(row['asap_prompt_id'])}",
                "document_shift_type": "semantic_task_shift",
                "shift_taxonomy": "semantic_task",
                "shift_construction": "alternate_scoring_construct_without_compatible_label",
                "is_document_ood": True,
                "lineage_document_id": str(row["input_document_id"]),
                "lineage_reuse_is_simulation_only": False,
                "label_available": False,
                "required_behavior": "fail_closed_no_qwk_mae_or_harmfulness_claim",
                "prompt_template_version": "asap_semantic_task_shift_v1",
                "prompt_template_sha256": _semantic_template_sha256(),
            }
        )
        semantic_rows.append(semantic)
        scenario_counts["semantic_task"] += 1

    within_path = output.with_name(f"{output.stem}_within_prompt_covariate_v1.jsonl")
    semantic_path = output.with_name(f"{output.stem}_semantic_task_shift_v1.jsonl")
    card_path = output.with_name(f"{output.stem}_benchmark_card_v1.json")
    write_jsonl(within_path, within_rows)
    write_jsonl(semantic_path, semantic_rows)
    card = {
        "artifact_type": "llm_judge_ood_asap_benchmark_card",
        "version": "asap_benchmark_contract_v2",
        "base_pool": "benchmark_test ID/H0 documents only",
        "evaluation_design": "paired transformations scored only after development selection is frozen",
        "scenario_document_counts": scenario_counts,
        "scenarios": {
            "id_h0": {
                "rows": "base benchmark_test rows with document_shift_type=id",
                "claim": "ID/H0 calibration and Judge quality only",
            },
            "within_prompt_covariate": {
                "path": str(within_path),
                "documents": len(within_rows),
                "construction": "same prompt/rubric; deterministic whitespace and paragraph layout only",
                "label_contract": "original score and rater labels are carried only because the non-whitespace token sequence is identical",
                "claim": "narrow format/style-only covariate perturbation; not broad benign equivalence",
                "formal_path": "detector, MMD, sequential monitoring, and bounded benign Probe",
            },
            "cross_prompt_compound": {
                "rows": "base benchmark_test rows with document_shift_type=near or far",
                "claim": "controlled cross-prompt compound document-domain shift only",
            },
            "semantic_task": {
                "path": str(semantic_path),
                "documents": len(semantic_rows),
                "label_available": False,
                "claim": "fail-closed diagnostic only; no QWK, MAE, Probe, Adapt, Gate, or Future claim is allowed",
            },
        },
        "prompt_rubric_contract": dict(prompt_metadata),
    }
    write_json(card_path, card)
    return {
        "within_prompt_covariate_path": str(within_path),
        "within_prompt_covariate_sha256": _sha256(within_path),
        "semantic_task_shift_path": str(semantic_path),
        "semantic_task_shift_sha256": _sha256(semantic_path),
        "benchmark_card_path": str(card_path),
        "benchmark_card_sha256": _sha256(card_path),
        "scenario_document_counts": scenario_counts,
    }


def _format_only_within_prompt_transform(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", str(text).strip())
    return re.sub(r"(?<=[.!?]) +", "\n", normalized)


def _semantic_task_shift_input(*, prompt_id: int, essay_text: str) -> str:
    return (
        "ASAP-AES semantic/task-shift diagnostic\n"
        "template_version: asap_semantic_task_shift_v1\n"
        f"original_prompt_id: {int(prompt_id)}\n\n"
        "TASK: Instead of the official assignment rubric, assess only whether this response discusses a newly introduced "
        "meta-criterion: the writer's awareness of automated scoring. This construct has no compatible human label in "
        "ASAP-AES. Do not output an ordinal quality score.\n\n"
        f"ESSAY:\n{str(essay_text).strip()}\n\n"
        "OUTPUT CONTRACT: Return the literal token UNAVAILABLE."
    )


def _semantic_template_sha256() -> str:
    return hashlib.sha256(
        b"asap_semantic_task_shift_v1::automated_scoring_awareness::UNAVAILABLE"
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare ASAP-AES for the LLM Judge OOD lifecycle.")
    parser.add_argument("--input", required=True, help="Official ASAP-AES TSV/CSV training table.")
    parser.add_argument("--output", default="artifacts/llm_judge_ood_asap/asap_prepared_contract_v1.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write-flows", action="store_true")
    parser.add_argument("--flow-window-size", type=int, default=50)
    args = parser.parse_args(argv)
    metadata = write_asap_prepared(
        input_path=args.input,
        output_path=args.output,
        seed=int(args.seed),
        write_flows=bool(args.write_flows),
        flow_config=ASAPFlowConfig(window_size=int(args.flow_window_size)),
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


def _normalize_raw_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for field, aliases in _COLUMN_ALIASES.items():
        value = next((row.get(alias) for alias in aliases if row.get(alias) not in (None, "")), None)
        if value is None:
            raise ValueError(f"ASAP row is missing {field}; accepted columns are {aliases}")
        normalized[field] = value
    normalized["essay_id"] = str(normalized["essay_id"])
    try:
        normalized["prompt_id"] = int(float(normalized["prompt_id"]))
        for field in ("rater1", "rater2", "resolved"):
            normalized[field] = float(normalized[field])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ASAP row {normalized['essay_id']} has a non-numeric prompt or score") from exc
    if not str(normalized["essay_text"]).strip():
        raise ValueError(f"ASAP row {normalized['essay_id']} has empty essay text")
    if any(not np.isfinite(float(normalized[field])) for field in ("rater1", "rater2", "resolved")):
        raise ValueError(f"ASAP row {normalized['essay_id']} has a non-finite overall score")
    normalized["essay_text"] = str(normalized["essay_text"])
    normalized["source_encoding"] = row.get("_source_encoding")
    return normalized


def _fit_prompt_score_mappings(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return fixed ASAP rubric mappings; no label fitting is performed.

    The historical function name is retained for compatibility with callers,
    but ``rows`` is used only to report prompt-level row counts in metadata.
    """
    mappings: dict[str, dict[str, Any]] = {}
    for prompt_id in ASAP_USED_PROMPTS:
        prompt_rows = [row for row in rows if int(row["prompt_id"]) == prompt_id]
        if not prompt_rows:
            continue
        ranges = ASAP_OFFICIAL_SCORE_RANGES[prompt_id]
        mappings[str(prompt_id)] = {
            "prompt_id": int(prompt_id),
            "method": "official_rubric_range_equal_width_buckets",
            "source": "asap_official_rubric_ranges_pre_split",
            "target_scale": [1, 2, 3, 4, 5],
            "field_ranges": {
                field: {"minimum": float(bounds[0]), "maximum": float(bounds[1])}
                for field, bounds in ranges.items()
            },
            "fit_documents": 0,
            "audit_documents": int(len(prompt_rows)),
            "tie_rule": "bucket_boundary_is_assigned_to_the_higher_bucket",
        }
    return mappings


def _apply_score_mapping(value: float, mapping: Mapping[str, Any], *, field: str) -> int:
    field_ranges = mapping.get("field_ranges", {})
    if field not in field_ranges:
        raise ValueError(f"ASAP score mapping has no range for field {field!r}")
    minimum = float(field_ranges[field]["minimum"])
    maximum = float(field_ranges[field]["maximum"])
    if not np.isfinite(value) or maximum <= minimum:
        raise ValueError(f"Invalid ASAP score mapping range for field {field!r}")
    if float(value) < minimum or float(value) > maximum:
        raise ValueError(
            f"ASAP {field} score {value} is outside the official prompt range "
            f"[{minimum}, {maximum}]"
        )
    normalized = (float(value) - minimum) / (maximum - minimum)
    # Equal-width buckets on [0, 1].  ``searchsorted(..., side='right')``
    # assigns an exact boundary to the higher ordinal bucket.
    boundaries = np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float64)
    return int(np.clip(1 + np.searchsorted(boundaries, normalized, side="right"), 1, 5))


def _validate_mapped_score_distributions(
    prepared: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit a fixed conversion without fitting or selecting on labels."""

    # This is a post-conversion integrity audit, not a mapping fit or model
    # selection step.  It therefore records the whole immutable prepared
    # dataset, including deployment rows, while making the zero-fit contract
    # explicit in the metadata.
    audit: dict[str, Any] = {}
    for prompt_id in ASAP_USED_PROMPTS:
        rows = [
            row
            for row in prepared
            if int(row["asap_prompt_id"]) == prompt_id
        ]
        if not rows:
            continue
        prompt_audit: dict[str, Any] = {"documents": int(len(rows)), "fields": {}}
        for field, raw_field in (
            ("resolved_score", "raw_resolved_score"),
            ("rater1_score", "raw_rater1_score"),
            ("rater2_score", "raw_rater2_score"),
        ):
            values = [int(row[field]) for row in rows]
            unique = sorted(set(values))
            if len(unique) < 2:
                raise ValueError(
                    f"ASAP mapped {field} collapsed to fewer than two levels for prompt {prompt_id}"
                )
            prompt_audit["fields"][field] = {
                "raw_observed_range": [
                    float(min(float(row[raw_field]) for row in rows)),
                    float(max(float(row[raw_field]) for row in rows)),
                ],
                "unique_values": unique,
                "counts": {str(value): int(values.count(value)) for value in unique},
                "collapsed": False,
            }
        if prompt_id in {1, 7, 8}:
            resolved = np.asarray([float(row["raw_resolved_score"]) for row in rows])
            rater_sum = np.asarray(
                [float(row["raw_rater1_score"]) + float(row["raw_rater2_score"]) for row in rows]
            )
            prompt_audit["resolved_rater_sum_audit"] = {
                "rows_equal_to_rater_sum": int(np.isclose(resolved, rater_sum).sum()),
                "rows_not_equal_to_rater_sum": int((~np.isclose(resolved, rater_sum)).sum()),
                "maximum_absolute_difference": float(np.max(np.abs(resolved - rater_sum))),
                "interpretation": "Disagreement can be caused by an official third-reader resolution; it is audited, not overwritten.",
            }
        audit[str(prompt_id)] = prompt_audit
    return {
        "method": "mapped_label_distribution_audit",
        "scope_splits": "all_prepared_rows_integrity_audit_only",
        "mapping_fit_documents": 0,
        "model_selection_uses_audit": False,
        "minimum_unique_levels": 2,
        "by_prompt": audit,
        "passed": True,
    }


def _validate_fractions(
    values: Mapping[str, float], *, expected_names: tuple[str, ...], name: str
) -> dict[str, float]:
    if tuple(values) != expected_names:
        raise ValueError(f"{name} keys must be ordered as {expected_names}")
    output = {str(key): float(value) for key, value in values.items()}
    if any(value < 0.0 for value in output.values()) or sum(output.values()) <= 0.0:
        raise ValueError(f"{name} must contain non-negative values with a positive sum")
    return output


def _stable_partition(keys: Sequence[str], fractions: Mapping[str, float], *, seed: int) -> dict[str, str]:
    unique_keys = list(dict.fromkeys(str(key) for key in keys))
    if len(unique_keys) != len(keys):
        raise ValueError("ASAP partition keys must be unique")
    if not unique_keys:
        return {}
    names = list(fractions)
    total = float(sum(fractions.values()))
    raw_sizes = np.asarray([float(fractions[name]) / total * len(unique_keys) for name in names])
    sizes = np.floor(raw_sizes).astype(int)
    remainder = int(len(unique_keys) - sizes.sum())
    if remainder:
        order = np.argsort(-(raw_sizes - sizes), kind="stable")
        sizes[order[:remainder]] += 1
    ranked = _stable_rank(unique_keys, seed=int(seed))
    output: dict[str, str] = {}
    start = 0
    for name, size in zip(names, sizes.tolist(), strict=True):
        for key in ranked[start : start + int(size)]:
            output[key] = str(name)
        start += int(size)
    return output


def _stable_rank(keys: Sequence[str], *, seed: int) -> list[str]:
    return sorted(
        (str(key) for key in keys),
        key=lambda key: hashlib.sha256(f"{int(seed)}::{key}".encode("utf-8")).hexdigest(),
    )


def _mixture_flow(
    *,
    source_pool: Sequence[dict[str, Any]],
    target_pool: Sequence[dict[str, Any]],
    flow_id: str,
    flow_type: str,
    target_shift_type: str,
    target_proportions: Sequence[float],
    window_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + _stable_int(flow_id))
    source_order = rng.permutation(len(source_pool)).tolist()
    target_order = rng.permutation(len(target_pool)).tolist()
    source_cursor = 0
    target_cursor = 0
    events: list[dict[str, Any]] = []
    for window_index, proportion in enumerate(target_proportions):
        target_count = int(round(float(proportion) * int(window_size)))
        source_count = int(window_size) - target_count
        selected_source, source_cursor = _cycle_take(source_pool, source_order, source_cursor, source_count)
        selected_target, target_cursor = _cycle_take(target_pool, target_order, target_cursor, target_count)
        members = [(row, False) for row in selected_source] + [(row, True) for row in selected_target]
        rng.shuffle(members)
        for row, is_shifted in members:
            events.append(
                _flow_event(
                    row,
                    flow_id=flow_id,
                    flow_type=flow_type,
                    target_shift_type=target_shift_type,
                    target_proportion=float(proportion),
                    window_index=int(window_index),
                    position=len(events),
                    is_shifted=bool(is_shifted),
                    harmless_transform=False,
                    seed=int(seed),
                )
            )
    return events


def _harmless_flow(
    *, source_pool: Sequence[dict[str, Any]], flow_id: str, window_size: int, window_count: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + _stable_int(flow_id))
    order = rng.permutation(len(source_pool)).tolist()
    cursor = 0
    events: list[dict[str, Any]] = []
    for window_index in range(int(window_count)):
        selected, cursor = _cycle_take(source_pool, order, cursor, int(window_size))
        for row in selected:
            events.append(
                _flow_event(
                    row,
                    flow_id=flow_id,
                    flow_type="harmless",
                    target_shift_type="harmless",
                    target_proportion=1.0,
                    window_index=int(window_index),
                    position=len(events),
                    is_shifted=True,
                    harmless_transform=True,
                    seed=int(seed),
                )
            )
    return events


def _cycle_take(
    pool: Sequence[dict[str, Any]], order: list[int], cursor: int, count: int
) -> tuple[list[dict[str, Any]], int]:
    if count == 0:
        return [], int(cursor)
    if int(cursor) + int(count) > len(order):
        raise ValueError(
            "ASAP flow would reuse a deployment document within one run; "
            "provide a larger independent deployment_stream pool or shorten the flow"
        )
    selected = [dict(pool[order[(int(cursor) + offset) % len(order)]]) for offset in range(int(count))]
    return selected, int(cursor) + int(count)


def _flow_event(
    row: dict[str, Any], *, flow_id: str, flow_type: str, target_shift_type: str,
    target_proportion: float, window_index: int, position: int, is_shifted: bool,
    harmless_transform: bool, seed: int
) -> dict[str, Any]:
    lineage_id = str(row["input_document_id"])
    event_id = f"asap-flow::{flow_id}::{int(position):06d}"
    text = str(row["input_document_text"])
    if harmless_transform:
        text = _label_preserving_style_transform(text)
    event = dict(row)
    event.update(
        {
            "sample_id": event_id,
            "id": event_id,
            "base_document_id": event_id,
            "input_document_id": event_id,
            "document_text": text,
            "input_document_text": text,
            "judge_input_text": build_asap_judge_input(
                prompt_id=int(row["asap_prompt_id"]), essay_text=text
            ),
            "split": "deployment_stream",
            "document_distribution_role": "deployment",
            "audit_document_group_id": (
                f"{'harmless' if harmless_transform else target_shift_type if is_shifted else 'id'}"
                f"_prompt_{int(row['asap_prompt_id'])}"
            ),
            "document_shift_type": (
                "harmless" if harmless_transform else target_shift_type if is_shifted else "id"
            ),
            "is_document_ood": bool(is_shifted),
            "stream_order": int(position),
            # The simulation event has a fresh document identity, so it is
            # also its own arrival/permutation block.  Do not recover prompt,
            # window, or injection-role batches here: those would collapse
            # independent essays into a pseudo-block.
            "arrival_batch_id": event_id,
            "flow_id": flow_id,
            "flow_type": flow_type,
            "flow_seed": int(seed),
            "flow_window_index": int(window_index),
            "flow_target_shift_type": target_shift_type,
            "flow_target_proportion": float(target_proportion),
            "lineage_document_id": lineage_id,
            "lineage_asap_essay_id": row.get("asap_essay_id"),
            "lineage_reuse_is_simulation_only": True,
            "shift_construction": (
                "label_preserving_light_formatting"
                if harmless_transform
                else "controlled_cross_prompt_compound_shift"
            ),
        }
    )
    return event


def _label_preserving_style_transform(text: str) -> str:
    normalized = re.sub(r"[ \t]+", " ", str(text).strip())
    normalized = re.sub(r"(?<=[.!?]) +", "\n", normalized)
    return f"Essay body (format-normalized):\n{normalized}"


def _flow_summary(flow_id: str, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_window: dict[int, list[Mapping[str, Any]]] = {}
    for row in events:
        by_window.setdefault(int(row["flow_window_index"]), []).append(row)
    return {
        "flow_id": flow_id,
        "events": len(events),
        "windows": len(by_window),
        "window_target_proportions": [
            float(rows[0]["flow_target_proportion"]) for _, rows in sorted(by_window.items())
        ],
        "realized_shift_proportions": [
            float(np.mean([bool(row["is_document_ood"]) for row in rows]))
            for _, rows in sorted(by_window.items())
        ],
    }


def _asap_metadata(
    *, raw_rows: Sequence[Mapping[str, Any]], prepared: Sequence[Mapping[str, Any]],
    score_mappings: Mapping[str, Mapping[str, Any]], seed: int,
    source_split_fractions: Mapping[str, float], target_split_fractions: Mapping[str, float],
    mapping_audit: Mapping[str, Any], prompt_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    split_ids: dict[str, list[str]] = {}
    for row in prepared:
        split_ids.setdefault(str(row["split"]), []).append(str(row["input_document_id"]))
    overlap_checks = {
        f"{left}__{right}": bool(set(split_ids[left]) & set(split_ids[right]))
        for index, left in enumerate(sorted(split_ids))
        for right in sorted(split_ids)[index + 1 :]
    }
    ignored_prompts = Counter(
        int(row["prompt_id"]) for row in raw_rows if int(row["prompt_id"]) not in ASAP_USED_PROMPTS
    )
    return {
        "artifact_type": "llm_judge_ood_asap_aes_prepared_metadata",
        "dataset_name": "ASAP-AES",
        "dataset_source": "Kaggle Automated Student Assessment Prize training set",
        "raw_rows": len(raw_rows),
        "prepared_rows": len(prepared),
        "used_prompts": list(ASAP_USED_PROMPTS),
        "ignored_prompt_counts": {str(key): int(value) for key, value in sorted(ignored_prompts.items())},
        "prompt_roles": {
            "id": list(ASAP_SOURCE_PROMPTS),
            "near": list(ASAP_NEAR_PROMPTS),
            "far": list(ASAP_FAR_PROMPTS),
        },
        "shift_taxonomy": {
            "id_h0": {
                "construction": "same_prompt_same_template_independent_documents",
                "supported_claim": "ID quality and Type-I/H0 calibration",
            },
            "within_prompt_covariate": {
                "construction": "separate deterministic text-layout variants; emitted in benchmark sidecar",
                "supported_claim": "format/style-only label-preserving covariate perturbation",
                "limitation": "does not establish broad benign equivalence",
            },
            "cross_prompt_compound": {
                "construction": "prompts 1/2 to prompts 3/4 (near) or 7/8 (far)",
                "supported_claim": "controlled cross-prompt compound document-domain shift",
                "prohibited_claim": "pure covariate shift or P(Y|X) invariance",
            },
            "semantic_task": {
                "construction": "alternate scoring construct with labels intentionally unavailable; emitted in benchmark sidecar",
                "supported_claim": "fail-closed semantic/task-shift diagnostic only",
            },
        },
        "prompt_rubric_contract": dict(prompt_metadata),
        "score_mapping": {
            "fit_field": None,
            "applied_fields": ["resolved", "rater1", "rater2"],
            "mapping_source": "asap_official_rubric_ranges_pre_split",
            "mapping_method": "official_rubric_range_equal_width_buckets",
            "target_scale": [1, 2, 3, 4, 5],
            "by_prompt": dict(score_mappings),
            "distribution_audit": dict(mapping_audit),
        },
        "source_split_fractions": dict(source_split_fractions),
        "target_split_fractions": dict(target_split_fractions),
        "seed": int(seed),
        "split_counts": {key: len(value) for key, value in sorted(split_ids.items())},
        "split_document_ids": {key: sorted(value) for key, value in sorted(split_ids.items())},
        "split_overlap_checks": overlap_checks,
        "all_primary_splits_document_disjoint": not any(overlap_checks.values()),
        "split_role_contract": {
            "training_train": "train Judge/head, whitening, PCA, ViM only",
            "training_drift_reference": "MMD reference only",
            "training_calibration": "single-window and episode H0 calibration only",
            "training_validation": "training early stopping/source hyperparameters only",
            "training_guard": "fixed source safety baseline only",
            "training_test": "Judge eligibility only",
            "development": "candidate selection only",
            "benchmark_test": "independent detector confirmation only",
            "deployment_stream": "controlled online stream only",
            "deployment_ood_evaluation": "offline OOD reporting only; never a Probe input",
            "deployment_adapt": "confirmed-harmful update training only",
            "deployment_gate": "frozen Gate only",
            "deployment_future_test": "one-time final report only",
        },
        "rater_contract": "raw_and_prompt_mapped_rater1_rater2_and_resolved_preserved",
        "multi_trait_policy": "overall_domain1_only_for_prompts_7_and_8",
    }


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:8], 16)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
