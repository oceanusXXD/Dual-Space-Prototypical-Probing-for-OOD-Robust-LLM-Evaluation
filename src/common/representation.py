from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from sklearn.decomposition import PCA

from src.common.stats import flatten_layers, mean_pool_layers


@dataclass(frozen=True)
class RepresentationSpec:
    kind: str = "last_layer"
    layer_index: int | None = None
    pca_dim: int = 48

    @property
    def name(self) -> str:
        if self.kind == "u":
            return "learned_u"
        if self.kind == "layer":
            return f"layer_{self.layer_index}"
        if self.kind == "pca":
            return f"pca_{self.pca_dim}"
        return self.kind

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RepresentationTransform:
    """Turn a source-fitted ``[N,L,D]`` cache into a 2-D detector space."""

    def __init__(self, spec: RepresentationSpec, *, random_state: int = 42) -> None:
        self.spec = spec
        self.random_state = int(random_state)
        self.pca_: PCA | None = None
        self.input_shape_: tuple[int, int] | None = None
        self.fit_rows_: int = 0

    def fit(self, source_features: np.ndarray) -> "RepresentationTransform":
        values = _as_layers(source_features)
        self.input_shape_ = (int(values.shape[1]), int(values.shape[2]))
        self.fit_rows_ = int(values.shape[0])
        kind = str(self.spec.kind).lower()
        if kind not in {"last_layer", "layer_mean", "layer", "pca", "u"}:
            raise ValueError("representation kind must be last_layer, layer_mean, layer, pca, or u")
        if kind == "layer":
            _resolve_layer_index(self.spec.layer_index, values.shape[1])
        elif kind == "pca":
            matrix = flatten_layers(values)
            n_components = min(int(self.spec.pca_dim), matrix.shape[0], matrix.shape[1])
            if n_components <= 0:
                raise ValueError("PCA representation requires a non-empty source matrix")
            solver = "randomized" if n_components < min(matrix.shape) else "full"
            self.pca_ = PCA(
                n_components=n_components,
                svd_solver=solver,
                random_state=self.random_state,
            ).fit(matrix)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = _as_layers(features)
        if self.input_shape_ is None:
            raise RuntimeError("RepresentationTransform is not fitted")
        if tuple(values.shape[1:]) != self.input_shape_:
            raise ValueError(f"Expected layer shape {self.input_shape_}, got {tuple(values.shape[1:])}")
        kind = str(self.spec.kind).lower()
        if kind in {"last_layer", "u"}:
            return values[:, -1, :].astype(np.float32)
        if kind == "layer_mean":
            return mean_pool_layers(values)
        if kind == "layer":
            index = _resolve_layer_index(self.spec.layer_index, values.shape[1])
            return values[:, index, :].astype(np.float32)
        if self.pca_ is None:
            raise RuntimeError("PCA representation is not fitted")
        return self.pca_.transform(flatten_layers(values)).astype(np.float32)

    def fit_transform(self, source_features: np.ndarray) -> np.ndarray:
        return self.fit(source_features).transform(source_features)

    def to_metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "spec": self.spec.to_dict(),
            "name": self.spec.name,
            "random_state": self.random_state,
            "fit_rows": self.fit_rows_,
            "input_shape": list(self.input_shape_) if self.input_shape_ is not None else None,
        }
        if self.pca_ is not None:
            payload.update(
                {
                    "output_dim": int(self.pca_.n_components_),
                    "explained_variance_ratio_sum": float(self.pca_.explained_variance_ratio_.sum()),
                }
            )
        elif self.input_shape_ is not None:
            payload["output_dim"] = int(self.input_shape_[1])
        return payload

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        if self.pca_ is None:
            return {}
        return {
            "components": self.pca_.components_.astype(np.float32),
            "mean": self.pca_.mean_.astype(np.float32),
            "explained_variance": self.pca_.explained_variance_.astype(np.float32),
            "explained_variance_ratio": self.pca_.explained_variance_ratio_.astype(np.float32),
        }


def expand_representation_specs(
    kinds: tuple[str, ...] | list[str],
    *,
    num_layers: int,
    pca_dim: int,
) -> list[RepresentationSpec]:
    specs: list[RepresentationSpec] = []
    for kind_value in kinds:
        kind = str(kind_value).lower()
        if kind == "all_layers":
            specs.extend(RepresentationSpec(kind="layer", layer_index=index) for index in range(num_layers))
        elif kind == "pca":
            specs.append(RepresentationSpec(kind="pca", pca_dim=int(pca_dim)))
        elif kind == "u":
            specs.append(RepresentationSpec(kind="u", pca_dim=int(pca_dim)))
        else:
            specs.append(RepresentationSpec(kind=kind, pca_dim=int(pca_dim)))
    deduplicated: list[RepresentationSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.name not in seen:
            deduplicated.append(spec)
            seen.add(spec.name)
    return deduplicated


def _resolve_layer_index(index: int | None, num_layers: int) -> int:
    if index is None:
        raise ValueError("layer representation requires layer_index")
    resolved = int(index) if int(index) >= 0 else int(num_layers) + int(index)
    if resolved < 0 or resolved >= int(num_layers):
        raise ValueError(f"layer_index {index} is invalid for {num_layers} layers")
    return resolved


def _as_layers(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"Expected [N,L,D] or [N,D], got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Representation features contain NaN or inf")
    return values
