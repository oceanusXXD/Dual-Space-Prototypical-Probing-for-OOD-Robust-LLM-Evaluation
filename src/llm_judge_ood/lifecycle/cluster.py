from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class ClusterConfig:
    method: str = "hdbscan"
    min_cluster_size: int = 10
    hdbscan_allow_single_cluster: bool = True
    mutual_k: int = 5
    min_similarity: float = 0.0
    dbscan_eps: float = 1.2
    dbscan_min_samples: int = 4
    hybrid_radius_multiplier: float = 1.5
    hybrid_radius_quantile: float = 0.95

    def __post_init__(self) -> None:
        if self.method not in {"hdbscan", "mutual_knn", "dbscan", "hybrid", "hdbscan_knn_expand"}:
            raise ValueError(
                "Cluster method must be 'hdbscan', 'mutual_knn', 'dbscan', or 'hybrid'"
            )
        if int(self.min_cluster_size) < 2:
            raise ValueError("min_cluster_size must be at least two")
        if int(self.mutual_k) < 1:
            raise ValueError("mutual_k must be positive")
        if float(self.dbscan_eps) <= 0.0 or int(self.dbscan_min_samples) < 1:
            raise ValueError("DBSCAN eps and min_samples must be positive")
        if float(self.hybrid_radius_multiplier) <= 0.0:
            raise ValueError("Hybrid radius multiplier must be positive")
        if not 0.0 < float(self.hybrid_radius_quantile) <= 1.0:
            raise ValueError("Hybrid radius quantile must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DocumentClusterer:
    def __init__(self, config: ClusterConfig | None = None) -> None:
        self.config = config or ClusterConfig()

    def fit_predict(self, embeddings: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
        values = np.asarray(embeddings, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("Document embeddings must be a 2D matrix")
        if values.shape[0] == 0:
            return np.zeros(0, dtype=int), []
        if values.shape[0] < int(self.config.min_cluster_size):
            # No retained cluster can meet the protocol's minimum size.  This
            # also prevents HDBSCAN from rejecting a candidate window whose
            # effective min_samples exceeds its row count.
            return np.full(values.shape[0], -1, dtype=int), []
        if self.config.method == "dbscan":
            raw = DBSCAN(eps=self.config.dbscan_eps, min_samples=self.config.dbscan_min_samples).fit_predict(values)
        elif self.config.method == "hdbscan":
            raw = hdbscan_labels(values, self.config)
        elif self.config.method in {"hybrid", "hdbscan_knn_expand"}:
            raw = hdbscan_labels(values, self.config)
            labels, expansion = expand_hdbscan_noise(
                values,
                raw,
                radius_multiplier=float(self.config.hybrid_radius_multiplier),
                radius_quantile=float(self.config.hybrid_radius_quantile),
            )
            summaries = summarize_clusters(values, labels)
            for summary in summaries:
                cluster_id = int(summary["cluster_id"])
                summary.update(
                    {
                        "cluster_origin": "hdbscan_core_radius_expansion",
                        "density_cluster_found": True,
                        "raw_core_size": int(np.sum(raw == cluster_id)),
                        "expanded_noise_count": int(expansion["expanded_by_cluster"].get(cluster_id, 0)),
                        "raw_noise_count": int(expansion["raw_noise_count"]),
                        "hybrid_radius_multiplier": float(self.config.hybrid_radius_multiplier),
                        "hybrid_radius_quantile": float(self.config.hybrid_radius_quantile),
                    }
                )
            return labels, summaries
        elif self.config.method == "mutual_knn":
            raw = mutual_knn_components(values, k=self.config.mutual_k, min_similarity=self.config.min_similarity)
        else:
            raise RuntimeError(f"Unsupported validated cluster method: {self.config.method}")
        labels = (
            np.asarray(raw, dtype=int)
            if self.config.method == "hdbscan"
            else filter_small_clusters(raw, min_size=self.config.min_cluster_size)
        )
        summaries = summarize_clusters(values, labels)
        return labels, summaries


def hdbscan_labels(values: np.ndarray, config: ClusterConfig) -> np.ndarray:
    model = HDBSCAN(
        min_cluster_size=int(config.min_cluster_size),
        allow_single_cluster=False,
        copy=False,
    )
    raw = model.fit_predict(values)
    if bool(config.hdbscan_allow_single_cluster) and not np.any(np.asarray(raw, dtype=int) >= 0):
        raw = HDBSCAN(
            min_cluster_size=int(config.min_cluster_size),
            allow_single_cluster=True,
            copy=False,
        ).fit_predict(values)
    return np.asarray(raw, dtype=int)


def expand_hdbscan_noise(
    values: np.ndarray,
    core_labels: np.ndarray,
    *,
    radius_multiplier: float,
    radius_quantile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Attach HDBSCAN noise to its nearest core only within that core's frozen radius."""

    output = np.asarray(core_labels, dtype=int).copy()
    clusters = sorted(set(output[output >= 0].tolist()))
    raw_noise_count = int(np.sum(output < 0))
    if not clusters:
        return output, {"raw_noise_count": raw_noise_count, "expanded_by_cluster": {}}

    centroids = np.stack([values[output == cluster].mean(axis=0) for cluster in clusters])
    radii = np.asarray(
        [
            np.quantile(
                np.linalg.norm(values[output == cluster] - centroid, axis=1),
                float(radius_quantile),
                method="linear",
            )
            for cluster, centroid in zip(clusters, centroids, strict=True)
        ],
        dtype=np.float64,
    )
    noise = np.flatnonzero(output < 0)
    if noise.size == 0:
        return output, {"raw_noise_count": raw_noise_count, "expanded_by_cluster": {}}

    distances = np.linalg.norm(values[noise, None, :] - centroids[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    accepted = distances[np.arange(noise.size), nearest] <= (
        radii[nearest] * float(radius_multiplier)
    )
    output[noise[accepted]] = np.asarray(clusters, dtype=int)[nearest[accepted]]
    expanded_by_cluster = {
        int(cluster): int(np.sum(accepted & (nearest == offset)))
        for offset, cluster in enumerate(clusters)
    }
    return output, {
        "raw_noise_count": raw_noise_count,
        "expanded_by_cluster": expanded_by_cluster,
    }


def mutual_knn_components(values: np.ndarray, *, k: int, min_similarity: float) -> np.ndarray:
    n_rows = int(values.shape[0])
    if n_rows == 1:
        return np.asarray([0], dtype=int)
    normalized = values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    k_eff = min(max(1, int(k)), n_rows - 1)
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="cosine")
    nn.fit(normalized)
    _, indices = nn.kneighbors(normalized)
    neighbor_sets = [set(row[1:].tolist()) for row in indices]
    parent = list(range(n_rows))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    similarities = normalized @ normalized.T
    for i in range(n_rows):
        for j in neighbor_sets[i]:
            if i in neighbor_sets[j] and float(similarities[i, j]) >= float(min_similarity):
                union(i, int(j))
    roots = [find(i) for i in range(n_rows)]
    remap = {root: idx for idx, root in enumerate(sorted(set(roots)))}
    return np.asarray([remap[root] for root in roots], dtype=int)


def filter_small_clusters(labels: np.ndarray, *, min_size: int) -> np.ndarray:
    out = np.asarray(labels, dtype=int).copy()
    next_label = 0
    remap: dict[int, int] = {}
    for label in sorted(set(out.tolist())):
        if label < 0:
            continue
        mask = out == label
        if int(mask.sum()) < int(min_size):
            out[mask] = -1
            continue
        remap[label] = next_label
        next_label += 1
    for old, new in remap.items():
        out[labels == old] = new
    return out


def summarize_clusters(values: np.ndarray, labels: np.ndarray) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        if int(label) < 0:
            continue
        mask = labels == label
        centroid = values[mask].mean(axis=0)
        compactness = float(np.mean(np.linalg.norm(values[mask] - centroid, axis=1)))
        summaries.append(
            {
                "cluster_id": int(label),
                "size": int(mask.sum()),
                "compactness": compactness,
                "centroid": centroid.astype(float).tolist(),
            }
        )
    return summaries
