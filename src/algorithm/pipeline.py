from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.algorithm.cli import (
    apply_threshold_command,
    certify_wsr_command,
    confirm_window_command,
    detect_command,
    extract_command,
    recertify_wsr_command,
    train_classifier_command,
    update_adapt_command,
    update_monitor_command,
)
from src.algorithm.data.flow import ALGORITHM_CHAIN, AlgorithmFlowStep, validate_algorithm_chain


@dataclass(frozen=True)
class PipelineStepResult:
    command: str
    output: Path | None
    metadata: dict[str, Any]


def algorithm_chain() -> tuple[AlgorithmFlowStep, ...]:
    validate_algorithm_chain()
    return ALGORITHM_CHAIN


def run_extract(args: Any) -> PipelineStepResult:
    metadata = extract_command(args)
    return PipelineStepResult("extract", Path(args.output), metadata)


def run_train_classifier(args: Any) -> PipelineStepResult:
    metadata = train_classifier_command(args)
    return PipelineStepResult("train-classifier", Path(args.output), metadata)


def run_detect(args: Any) -> PipelineStepResult:
    metadata = detect_command(args)
    return PipelineStepResult("detect", Path(args.output), metadata)


def run_certify_wsr(args: Any) -> PipelineStepResult:
    metadata = certify_wsr_command(args)
    return PipelineStepResult("certify-wsr", Path(args.output), metadata)


def run_apply_threshold(args: Any) -> PipelineStepResult:
    metadata = apply_threshold_command(args)
    return PipelineStepResult("apply-threshold", Path(args.output), metadata)


def run_confirm_window(args: Any) -> PipelineStepResult:
    metadata = confirm_window_command(args)
    return PipelineStepResult("confirm-window", Path(args.output), metadata)


def run_update_monitor(args: Any) -> PipelineStepResult:
    metadata = update_monitor_command(args)
    return PipelineStepResult("update-monitor", Path(args.output), metadata)


def run_update_adapt(args: Any) -> PipelineStepResult:
    metadata = update_adapt_command(args)
    return PipelineStepResult("update-adapt", Path(args.output), metadata)


def run_recertify_wsr(args: Any) -> PipelineStepResult:
    metadata = recertify_wsr_command(args)
    return PipelineStepResult("recertify-wsr", Path(args.output), metadata)
