"""Few-label adaptation helpers for LLM Judge OOD."""

from src.llm_judge_ood.adapt.backbone import LightBackboneAdaptResult, light_backbone_update
from src.llm_judge_ood.adapt.coral import CoralAligner, nearest_centroid_predict
from src.llm_judge_ood.adapt.head import HeadAdapter, NewQueryHeadTrainer

__all__ = [
    "CoralAligner",
    "HeadAdapter",
    "LightBackboneAdaptResult",
    "NewQueryHeadTrainer",
    "light_backbone_update",
    "nearest_centroid_predict",
]
