from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PersistenceConfig:
    window_size: int = 64
    min_share: float = 0.05
    history_windows: int = 3
    min_passing_windows: int = 2
    centroid_match_max_distance: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentClusterState:
    document_cluster_id: str
    centroid: np.ndarray
    first_window: int
    last_window: int
    shares: list[float] = field(default_factory=list)
    status: str = "candidate_document_cluster"
    total_members: int = 0
    consecutive_passing_windows: int = 0
    confirmation_window: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_cluster_id": self.document_cluster_id,
            "first_window": self.first_window,
            "last_window": self.last_window,
            "shares": [float(value) for value in self.shares],
            "status": self.status,
            "total_members": int(self.total_members),
            "consecutive_passing_windows": int(self.consecutive_passing_windows),
            "confirmation_window": self.confirmation_window,
            "confirmation_latency_windows": (
                int(self.confirmation_window - self.first_window)
                if self.confirmation_window is not None
                else None
            ),
            "centroid": self.centroid.astype(float).tolist(),
        }


class DocumentClusterTracker:
    def __init__(self, config: PersistenceConfig | None = None) -> None:
        self.config = config or PersistenceConfig()
        if int(self.config.history_windows) < 1:
            raise ValueError("history_windows must be positive")
        if not 1 <= int(self.config.min_passing_windows) <= int(self.config.history_windows):
            raise ValueError("min_passing_windows must be in [1, history_windows]")
        self.states: dict[str, DocumentClusterState] = {}
        self._next_id = 1

    def update(self, *, window_index: int, window_size: int, cluster_summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        touched: set[str] = set()
        rows: list[dict[str, Any]] = []
        for cluster in cluster_summaries:
            centroid = np.asarray(cluster["centroid"], dtype=np.float32)
            document_cluster_id = self._match(centroid, excluded=touched)
            share = float(cluster["size"]) / max(int(window_size), 1)
            if document_cluster_id is None:
                document_cluster_id = f"C{self._next_id:04d}"
                self._next_id += 1
                self.states[document_cluster_id] = DocumentClusterState(
                    document_cluster_id=document_cluster_id,
                    centroid=centroid,
                    first_window=int(window_index),
                    last_window=int(window_index),
                )
            state = self.states[document_cluster_id]
            state.centroid = 0.7 * state.centroid + 0.3 * centroid
            state.last_window = int(window_index)
            state.shares.append(share)
            state.shares = state.shares[-int(self.config.history_windows) :]
            state.total_members += int(cluster["size"])
            state.consecutive_passing_windows = (
                state.consecutive_passing_windows + 1
                if share >= float(self.config.min_share)
                else 0
            )
            if (
                state.confirmation_window is None
                and state.consecutive_passing_windows >= int(self.config.min_passing_windows)
            ):
                state.confirmation_window = int(window_index)
            state.status = self._status(state)
            touched.add(document_cluster_id)
            rows.append({**cluster, **state.to_dict(), "window_share": share})
        for document_cluster_id, state in self.states.items():
            if document_cluster_id in touched:
                continue
            state.shares.append(0.0)
            state.shares = state.shares[-int(self.config.history_windows) :]
            state.consecutive_passing_windows = 0
            if int(window_index) - state.last_window >= int(self.config.history_windows):
                state.status = "expired_document_cluster"
            else:
                state.status = self._status(state)
        return rows

    def _match(self, centroid: np.ndarray, *, excluded: set[str]) -> str | None:
        best_id: str | None = None
        best_distance = float("inf")
        for document_cluster_id, state in self.states.items():
            if state.status == "expired_document_cluster" or document_cluster_id in excluded:
                continue
            distance = float(np.linalg.norm(centroid - state.centroid))
            if distance < best_distance:
                best_distance = distance
                best_id = document_cluster_id
        if best_id is not None and best_distance <= float(self.config.centroid_match_max_distance):
            return best_id
        return None

    def _status(self, state: DocumentClusterState) -> str:
        if not state.shares:
            return "isolated_anomaly"
        if state.confirmation_window is not None:
            return "persistent_document_cluster"
        if state.total_members <= 1:
            return "isolated_anomaly"
        return "candidate_document_cluster"

    def snapshot(self) -> list[dict[str, Any]]:
        return [state.to_dict() for state in sorted(self.states.values(), key=lambda item: item.document_cluster_id)]
