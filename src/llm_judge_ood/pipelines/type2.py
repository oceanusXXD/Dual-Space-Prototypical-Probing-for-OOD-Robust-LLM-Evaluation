from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.common.io import ensure_dir, write_json
from src.llm_judge_ood.adapt.head import NewQueryHeadTrainer
from src.llm_judge_ood.model.judge import JudgeTrainingConfig, SharedBackboneJudge
from src.llm_judge_ood.pipelines.sample_ood import SampleOODConfig, _load_or_extract_features
from src.llm_judge_ood.shared.metrics import judge_metrics, normalize_label_array
from src.llm_judge_ood.shared.schema import load_judge_records
from src.llm_judge_ood.shared.whitening import LayerWhitening


@dataclass(frozen=True)
class Type2Config:
    sample: SampleOODConfig
    new_query_ids: tuple[str, ...] = ()
    label_splits: tuple[str, ...] = ("deployment_adapt",)
    eval_splits: tuple[str, ...] = ("deployment_future_test",)
    budgets: tuple[int, ...] = (8, 16, 32, 64, 128)
    output_dir: str = "artifacts/llm_judge_ood_type2"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["sample"] = self.sample.to_dict()
        return payload


def run_type2_new_query_pipeline(config: Type2Config) -> dict[str, Any]:
    """Run explicit unseen-query adaptation without routing through OOD detection."""

    records = load_judge_records(config.sample.input_paths)
    document_roles = np.asarray([record.document_distribution_role for record in records]).astype(str)
    labels = normalize_label_array([record.label for record in records])
    splits = np.asarray([record.split for record in records]).astype(str)
    query_ids = np.asarray([record.query_id for record in records]).astype(str)
    invalid_roles = sorted(set(document_roles.tolist()) - {"training", "development", "deployment"})
    if invalid_roles:
        raise ValueError(f"Type-II adaptation requires explicit document_distribution_role values, got {invalid_roles}")
    training_train_mask = (document_roles == "training") & np.isin(
        splits, config.sample.training_document_train_splits
    )
    training_validation_mask = (document_roles == "training") & np.isin(
        splits, config.sample.training_document_validation_splits
    )
    if not training_train_mask.any():
        raise ValueError("No training document train records selected for Type-II adaptation")
    if not training_validation_mask.any():
        training_validation_mask = training_train_mask.copy()

    known_queries = set(query_ids[training_train_mask].tolist())
    requested_queries = set(str(value) for value in config.new_query_ids)
    if requested_queries:
        unknown_requested = requested_queries & known_queries
        if unknown_requested:
            raise ValueError(f"Type-II query IDs are already known in Source Train: {sorted(unknown_requested)}")
        new_queries = tuple(sorted(requested_queries))
    else:
        new_queries = tuple(sorted(set(query_ids.tolist()) - known_queries))
    output_dir = ensure_dir(config.output_dir)
    if not new_queries:
        summary = {
            "artifact_type": "llm_judge_ood_type2_summary",
            "status": "no_new_queries",
            "known_query_ids": sorted(known_queries),
            "new_query_ids": [],
            "config": config.to_dict(),
        }
        write_json(output_dir / "summary.json", summary)
        return summary

    raw_features, feature_metadata = _load_or_extract_features(
        config.sample,
        records,
        training_train_mask,
        feature_scope=str(config.sample.judge_feature_scope),
    )
    whitened = LayerWhitening().fit(raw_features[training_train_mask]).transform(raw_features)
    judge = SharedBackboneJudge(config.sample.judge).fit(
        whitened,
        labels,
        query_ids,
        train_mask=training_train_mask,
        validation_mask=training_validation_mask,
    )
    u_space = judge.transform_u(whitened)
    curves: list[dict[str, Any]] = []
    for query_id in new_queries:
        label_pool = np.flatnonzero((query_ids == query_id) & np.isin(splits, config.label_splits))
        eval_indices = np.flatnonzero((query_ids == query_id) & np.isin(splits, config.eval_splits))
        query_rows: list[dict[str, Any]] = []
        if not label_pool.size or not eval_indices.size:
            curves.append({"query_id": query_id, "status": "missing_label_or_eval_pool", "budgets": query_rows})
            continue
        for budget in sorted(set(max(1, int(value)) for value in config.budgets)):
            selected = label_pool[: min(int(budget), len(label_pool))]
            trainer = NewQueryHeadTrainer()
            trainer.fit(
                u_features=u_space,
                labels=labels,
                query_ids=query_ids,
                new_query_indices=selected,
            )
            fallback = np.full(eval_indices.size, judge.classes_[0], dtype=judge.classes_.dtype)
            predictions = trainer.predict(
                u_features=u_space[eval_indices],
                query_ids=query_ids[eval_indices],
                fallback=fallback,
            )
            query_rows.append(
                {
                    "budget": int(budget),
                    "used_labels": int(len(selected)),
                    "metrics": judge_metrics(
                        labels[eval_indices],
                        predictions,
                        class_values=judge.classes_,
                    ),
                    "trained_query_ids": sorted(trainer.models_),
                }
            )
        curves.append({"query_id": query_id, "status": "completed", "budgets": query_rows})
    summary = {
        "artifact_type": "llm_judge_ood_type2_summary",
        "status": "completed",
        "known_query_ids": sorted(known_queries),
        "new_query_ids": list(new_queries),
        "feature_extractor": feature_metadata,
        "curves": curves,
        "config": config.to_dict(),
        "outputs": {"summary": str(output_dir / "summary.json")},
    }
    write_json(output_dir / "summary.json", summary)
    return summary
