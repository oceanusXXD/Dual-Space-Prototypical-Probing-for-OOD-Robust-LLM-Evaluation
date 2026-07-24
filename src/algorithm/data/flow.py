from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AlgorithmFlowStep:
    key: str
    label: str
    package: str
    output_contract: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


ALGORITHM_CHAIN: tuple[AlgorithmFlowStep, ...] = (
    AlgorithmFlowStep("raw_text", "原始文本", "src.common.schema", "JudgeRecord"),
    AlgorithmFlowStep("hidden_state", "Hidden-state 表征", "src.algorithm.hidden_state", "HiddenFeatureStore"),
    AlgorithmFlowStep("classifier", "评分分类器", "src.algorithm.classifier", "PredictionRow/JudgeHeadOutput"),
    AlgorithmFlowStep("detector", "错误风险检测器", "src.algorithm.detector", "DetectorScoreRow"),
    AlgorithmFlowStep("wsr_certification", "WSR 阈值认证", "src.algorithm.wsr", "ThresholdArtifact"),
    AlgorithmFlowStep("row_decision", "逐条 accept/reject", "src.algorithm.data.decisions", "DecisionRow"),
    AlgorithmFlowStep("window_failure", "窗口级失效确认", "src.algorithm.data.monitoring", "WindowFailureArtifact"),
    AlgorithmFlowStep("model_update", "模型更新", "src.algorithm.update", "adapted classifier artifact"),
    AlgorithmFlowStep("recertification", "重新认证", "src.algorithm.wsr", "ThresholdArtifact"),
)


def validate_algorithm_chain() -> None:
    keys = [step.key for step in ALGORITHM_CHAIN]
    expected = [
        "raw_text",
        "hidden_state",
        "classifier",
        "detector",
        "wsr_certification",
        "row_decision",
        "window_failure",
        "model_update",
        "recertification",
    ]
    if keys != expected:
        raise RuntimeError(f"algorithm chain changed unexpectedly: {keys}")
