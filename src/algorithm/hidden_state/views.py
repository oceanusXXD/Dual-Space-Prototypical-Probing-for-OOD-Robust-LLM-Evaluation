from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.common.schema import JudgeRecord


@dataclass(frozen=True)
class HiddenStateView:
    name: str
    space: str
    feature_scope: str
    pooling: str


VIEWS: dict[str, HiddenStateView] = {
    "input_document_masked_mean": HiddenStateView(
        "input_document_masked_mean", "a", "input_document", "masked_mean"
    ),
    "judge_prompt_masked_mean": HiddenStateView(
        "judge_prompt_masked_mean", "b", "judge_input", "masked_mean"
    ),
    "candidate_span_mean": HiddenStateView("candidate_span_mean", "a", "input_document", "span_mean"),
    "rubric_task_span_mean": HiddenStateView("rubric_task_span_mean", "b", "judge_input", "span_mean"),
    "pre_score_token": HiddenStateView("pre_score_token", "b", "judge_input", "pre_score_token"),
    "pre_label_token": HiddenStateView("pre_label_token", "b", "judge_input", "pre_label_token"),
}


def get_view(name: str) -> HiddenStateView:
    try:
        return VIEWS[str(name)]
    except KeyError as error:
        raise ValueError(f"unknown hidden-state view: {name}") from error


def resolve_view_texts(records: Sequence[JudgeRecord], view: str) -> list[str]:
    spec = get_view(view)
    if spec.feature_scope == "input_document":
        return [record.input_document_text for record in records]
    if spec.feature_scope == "judge_input":
        return [record.judge_input_text for record in records]
    raise ValueError(f"unsupported feature scope: {spec.feature_scope}")
