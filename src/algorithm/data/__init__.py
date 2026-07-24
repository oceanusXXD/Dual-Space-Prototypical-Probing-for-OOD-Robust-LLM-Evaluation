"""Stable data contracts for algorithm inputs, outputs, and decisions."""

from __future__ import annotations

from src.algorithm.data.decisions import DecisionRow, apply_accept_reject
from src.algorithm.data.detector_scores import DetectorScoreRow
from src.algorithm.data.flow import ALGORITHM_CHAIN, AlgorithmFlowStep
from src.algorithm.data.monitoring import MonitoringArtifact, WindowFailureArtifact
from src.algorithm.data.predictions import PredictionRow
from src.algorithm.data.thresholds import ThresholdArtifact

__all__ = [
    "ALGORITHM_CHAIN",
    "AlgorithmFlowStep",
    "DecisionRow",
    "DetectorScoreRow",
    "MonitoringArtifact",
    "PredictionRow",
    "ThresholdArtifact",
    "WindowFailureArtifact",
    "apply_accept_reject",
]
