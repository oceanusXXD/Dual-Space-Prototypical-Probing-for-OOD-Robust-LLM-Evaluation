from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from src.llm_judge_ood.model.judge import SharedBackboneJudge, _classification_loss


@dataclass(frozen=True)
class LightBackboneAdaptConfig:
    epochs: int = 5
    backbone_learning_rate: float = 1e-4
    head_learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_deployment_rows: int = 8
    anchor_weight: float = 1e-4
    training_replay_weight: float = 1.0
    deployment_weight: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 < float(self.backbone_learning_rate) < float(self.head_learning_rate):
            raise ValueError("backbone_learning_rate must be positive and lower than head_learning_rate")
        if int(self.epochs) < 1:
            raise ValueError("epochs must be positive")
        if int(self.min_deployment_rows) < 1:
            raise ValueError("min_deployment_rows must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LightBackboneAdaptResult:
    attempted: bool
    updated: bool
    reason: str
    train_rows: int
    config: dict[str, Any]


def light_backbone_update(
    *,
    judge: SharedBackboneJudge,
    features: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    deployment_indices: np.ndarray,
    training_replay_indices: np.ndarray,
    config: LightBackboneAdaptConfig | None = None,
) -> LightBackboneAdaptResult:
    """Level-2 path: lightly update the existing shared backbone and heads.

    This keeps the frozen feature extractor and whitening fixed. It is intentionally
    conservative and only runs when enough deployment rows exist; otherwise the pipeline
    records an explicit skip instead of pretending the upgrade happened.
    """

    cfg = config or LightBackboneAdaptConfig()
    deployment_indices = np.asarray(deployment_indices, dtype=int)
    training_replay_indices = np.asarray(training_replay_indices, dtype=int)
    train_indices = np.concatenate([training_replay_indices, deployment_indices])
    if deployment_indices.size < int(cfg.min_deployment_rows):
        return LightBackboneAdaptResult(False, False, "insufficient_deployment_rows", int(deployment_indices.size), cfg.to_dict())
    if judge.backbone is None or judge.classes_ is None:
        return LightBackboneAdaptResult(False, False, "judge_not_fitted", int(deployment_indices.size), cfg.to_dict())
    class_to_index = {value: idx for idx, value in enumerate(judge.classes_.tolist())}
    usable = [idx for idx in train_indices.tolist() if labels[idx] in class_to_index]
    if len(usable) < 2:
        return LightBackboneAdaptResult(False, False, "insufficient_usable_rows", int(len(usable)), cfg.to_dict())
    device = next(judge.backbone.parameters()).device
    x = torch.as_tensor(np.asarray(features, dtype=np.float32)[usable], dtype=torch.float32, device=device)
    y = torch.as_tensor([class_to_index[labels[idx]] for idx in usable], dtype=torch.long, device=device)
    q = np.asarray(query_ids).astype(str)[usable]
    training_index_set = set(training_replay_indices.tolist())
    is_training = np.asarray([index in training_index_set for index in usable], dtype=bool)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(judge.backbone.parameters()),
                "lr": float(cfg.backbone_learning_rate),
            },
            {
                "params": list(judge.heads.parameters()),
                "lr": float(cfg.head_learning_rate),
            },
        ],
        weight_decay=float(cfg.weight_decay),
    )
    anchor_parameters = {
        name: parameter.detach().clone()
        for name, parameter in [
            *judge.backbone.named_parameters(),
            *[(f"head.{query_id}.{name}", parameter) for query_id, head in judge.heads.items() for name, parameter in head.named_parameters()],
        ]
    }
    judge.backbone.train()
    for head in judge.heads.values():
        head.train()
    for _ in range(int(cfg.epochs)):
        shared = judge.backbone(x)
        losses: list[torch.Tensor] = []
        loss_weights: list[float] = []
        for query_id in sorted(set(q.tolist())):
            if query_id not in judge.heads:
                continue
            mask = torch.as_tensor(q == query_id, dtype=torch.bool, device=device)
            if not mask.any():
                continue
            logits = judge.heads[query_id](shared[mask])
            local_indices = np.flatnonzero(q == query_id)
            training_local = torch.as_tensor(
                is_training[local_indices],
                dtype=torch.bool,
                device=device,
            )
            deployment_local = ~training_local
            if bool(training_local.any()):
                losses.append(
                    _classification_loss(
                        logits[training_local],
                        y[mask][training_local],
                        judge.config.loss,
                        len(judge.classes_),
                    )
                )
                loss_weights.append(float(cfg.training_replay_weight))
            if bool(deployment_local.any()):
                losses.append(
                    _classification_loss(
                        logits[deployment_local],
                        y[mask][deployment_local],
                        judge.config.loss,
                        len(judge.classes_),
                    )
                )
                loss_weights.append(float(cfg.deployment_weight))
        if not losses:
            return LightBackboneAdaptResult(False, False, "no_known_query_head", int(len(usable)), cfg.to_dict())
        weights = torch.as_tensor(loss_weights, dtype=torch.float32, device=device)
        loss = (torch.stack(losses) * weights).sum() / weights.sum().clamp_min(1e-12)
        if float(cfg.anchor_weight) > 0.0:
            current_parameters = [
                *judge.backbone.named_parameters(),
                *[(f"head.{query_id}.{name}", parameter) for query_id, head in judge.heads.items() for name, parameter in head.named_parameters()],
            ]
            anchor_penalty = torch.stack(
                [(parameter - anchor_parameters[name]).pow(2).mean() for name, parameter in current_parameters]
            ).mean()
            loss = loss + float(cfg.anchor_weight) * anchor_penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    judge.backbone.eval()
    for head in judge.heads.values():
        head.eval()
    return LightBackboneAdaptResult(True, True, "updated", int(len(usable)), cfg.to_dict())
