from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf, ledoit_wolf_shrinkage
from sklearn.decomposition import PCA


@dataclass
class LayerWhitening:
    epsilon: float = 1e-5
    means_: list[np.ndarray] = field(default_factory=list)
    matrices_: list[np.ndarray] = field(default_factory=list)

    def fit(self, features: np.ndarray) -> "LayerWhitening":
        values = _check_layers(features)
        self.means_ = []
        self.matrices_ = []
        for layer in range(values.shape[1]):
            x = values[:, layer, :].astype(np.float64)
            mean = x.mean(axis=0)
            centered = x - mean
            covariance = LedoitWolf().fit(centered).covariance_ if x.shape[0] > 2 else np.cov(centered, rowvar=False)
            covariance = np.atleast_2d(covariance).astype(np.float64)
            eigvals, eigvecs = np.linalg.eigh(covariance)
            eigvals = np.maximum(eigvals, self.epsilon)
            matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            self.means_.append(mean.astype(np.float32))
            self.matrices_.append(matrix.astype(np.float32))
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = _check_layers(features)
        if not self.means_ or len(self.means_) != values.shape[1]:
            raise RuntimeError("LayerWhitening is not fitted for this layer count")
        out = np.empty_like(values, dtype=np.float32)
        for layer, (mean, matrix) in enumerate(zip(self.means_, self.matrices_, strict=True)):
            out[:, layer, :] = (values[:, layer, :] - mean) @ matrix
        return out

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def to_metadata(self) -> dict[str, Any]:
        return {"epsilon": self.epsilon, "num_layers": len(self.means_), "dim": int(self.means_[0].shape[0]) if self.means_ else None}


@dataclass
class SourcePCA:
    n_components: int = 32
    random_state: int = 42
    pca_: PCA | None = None

    def fit(self, features: np.ndarray) -> "SourcePCA":
        matrix = flatten_layers(features)
        n_components = min(int(self.n_components), matrix.shape[0], matrix.shape[1])
        if n_components <= 0:
            raise ValueError("PCA needs a non-empty feature matrix")
        self.pca_ = PCA(n_components=n_components, random_state=self.random_state)
        self.pca_.fit(matrix)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.pca_ is None:
            raise RuntimeError("SourcePCA is not fitted")
        return self.pca_.transform(flatten_layers(features)).astype(np.float32)


@dataclass
class LayerPreprocessor:
    """Configurable preprocessing fitted exclusively on Source Train rows."""

    method: str = "none"
    epsilon: float = 1e-5
    pca_components: int = 48
    pca_variance_target: float = 0.95
    random_state: int = 42
    means_: list[np.ndarray] = field(default_factory=list)
    scales_: list[np.ndarray] = field(default_factory=list)
    matrices_: list[np.ndarray] = field(default_factory=list)
    zca_bases_: list[np.ndarray] = field(default_factory=list)
    zca_deltas_: list[np.ndarray] = field(default_factory=list)
    zca_residual_scales_: list[float] = field(default_factory=list)
    pcas_: list[PCA] = field(default_factory=list)
    fit_rows_: int = 0
    input_shape_: tuple[int, int] | None = None
    input_stats_: list[dict[str, float]] = field(default_factory=list)
    output_stats_: list[dict[str, float]] = field(default_factory=list)

    def fit(self, features: np.ndarray) -> "LayerPreprocessor":
        values = _check_layers(features)
        method = str(self.method).lower()
        if method not in {"none", "diagonal", "zca", "pca_whiten"}:
            raise ValueError("method must be one of: none, diagonal, zca, pca_whiten")
        if not 0.0 < float(self.pca_variance_target) <= 1.0:
            raise ValueError("pca_variance_target must be in (0, 1]")
        self.method = method
        self.means_ = []
        self.scales_ = []
        self.matrices_ = []
        self.zca_bases_ = []
        self.zca_deltas_ = []
        self.zca_residual_scales_ = []
        self.pcas_ = []
        self.input_stats_ = []
        self.output_stats_ = []
        self.fit_rows_ = int(values.shape[0])
        self.input_shape_ = (int(values.shape[1]), int(values.shape[2]))
        pca_required_components_by_layer: list[int] = []
        pca_full_models: list[PCA] = []
        if method == "pca_whiten":
            for layer in range(values.shape[1]):
                x = values[:, layer, :].astype(np.float64)
                maximum_components = min(x.shape[0], x.shape[1])
                if maximum_components <= 0:
                    raise ValueError("pca_whiten requires a non-empty source matrix")
                # A full PCA is needed to find the variance rank.  Its leading
                # components are also the final whitening basis, so retain and
                # truncate it below instead of fitting the source matrix twice.
                target_pca = PCA(
                    n_components=None,
                    whiten=True,
                    svd_solver="full",
                    random_state=int(self.random_state),
                ).fit(x)
                cumulative_variance = np.cumsum(
                    np.asarray(target_pca.explained_variance_ratio_, dtype=np.float64)
                )
                required_components = (
                    int(maximum_components)
                    if float(self.pca_variance_target) >= 1.0
                    else int(
                        np.searchsorted(
                            cumulative_variance,
                            float(self.pca_variance_target),
                            side="left",
                        )
                        + 1
                    )
                )
                required_components = min(required_components, int(maximum_components))
                pca_required_components_by_layer.append(required_components)
                pca_full_models.append(target_pca)
            # The final protocol retains the smallest source-only PCA basis
            # that reaches the variance target. A fixed dimensional request
            # must not silently enlarge that basis beyond the documented 95%.
            pca_output_components = min(
                values.shape[0],
                values.shape[2],
                max(pca_required_components_by_layer),
            )
        else:
            pca_output_components = 0
        for layer in range(values.shape[1]):
            x = values[:, layer, :].astype(np.float64)
            mean = x.mean(axis=0)
            centered = x - mean
            variances = centered.var(axis=0)
            positive = variances[variances > float(self.epsilon)]
            diagonal_condition = (
                float(variances.max() / max(float(positive.min()), float(self.epsilon)))
                if variances.size and positive.size
                else 1.0
            )
            self.input_stats_.append(
                {
                    "variance_mean": float(variances.mean()),
                    "variance_min": float(variances.min()) if variances.size else 0.0,
                    "variance_max": float(variances.max()) if variances.size else 0.0,
                    "condition_number": diagonal_condition,
                }
            )
            self.means_.append(mean.astype(np.float32))
            if method == "diagonal":
                scale = np.sqrt(np.maximum(variances, float(self.epsilon)))
                self.scales_.append(scale.astype(np.float32))
            elif method == "zca":
                if x.shape[1] > x.shape[0] and x.shape[0] > 2:
                    # The Ledoit-Wolf covariance is an isotropic shrinkage of
                    # X^T X.  In the high-dimensional regime its eigensystem can
                    # be applied in the n-dimensional data span, avoiding a
                    # cubic d x d eigendecomposition for 4B hidden states.
                    shrinkage = float(ledoit_wolf_shrinkage(centered, assume_centered=True))
                    mu = float(np.square(centered).sum() / max(x.shape[0] * x.shape[1], 1))
                    residual_eigenvalue = max(shrinkage * mu, float(self.epsilon))
                    _, singular_values, basis_t = np.linalg.svd(centered, full_matrices=False)
                    span_eigenvalues = (
                        (1.0 - shrinkage) * np.square(singular_values) / x.shape[0]
                        + residual_eigenvalue
                    )
                    span_eigenvalues = np.maximum(span_eigenvalues, float(self.epsilon))
                    base_scale = float(1.0 / np.sqrt(residual_eigenvalue))
                    deltas = 1.0 / np.sqrt(span_eigenvalues) - base_scale
                    self.zca_bases_.append(basis_t.astype(np.float32))
                    self.zca_deltas_.append(deltas.astype(np.float32))
                    self.zca_residual_scales_.append(base_scale)
                    self.input_stats_[-1]["condition_number"] = float(
                        span_eigenvalues.max() / residual_eigenvalue
                    )
                else:
                    covariance = (
                        LedoitWolf().fit(centered).covariance_
                        if x.shape[0] > 2
                        else np.atleast_2d(np.cov(centered, rowvar=False))
                    )
                    covariance = np.atleast_2d(covariance).astype(np.float64)
                    eigvals, eigvecs = np.linalg.eigh(covariance)
                    eigvals = np.maximum(eigvals, float(self.epsilon))
                    self.input_stats_[-1]["condition_number"] = float(eigvals.max() / eigvals.min())
                    matrix = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                    self.matrices_.append(matrix.astype(np.float32))
            elif method == "pca_whiten":
                maximum_components = min(x.shape[0], x.shape[1])
                if maximum_components <= 0:
                    raise ValueError("pca_whiten requires a non-empty source matrix")
                # One common dimension keeps [N,L,D'] tensors and serialized
                # component arrays regular; it is the maximum source-only
                # requirement across layers.
                required_components = int(pca_required_components_by_layer[layer])
                n_components = min(maximum_components, int(pca_output_components))
                pca = _truncate_fitted_pca(
                    pca_full_models[layer],
                    n_components=int(n_components),
                )
                explained = np.maximum(np.asarray(pca.explained_variance_, dtype=np.float64), self.epsilon)
                self.input_stats_[-1]["condition_number"] = float(explained.max() / explained.min())
                self.input_stats_[-1]["pca_required_components_for_target"] = float(required_components)
                self.input_stats_[-1]["pca_explained_variance_ratio_sum"] = float(
                    np.asarray(pca.explained_variance_ratio_, dtype=np.float64).sum()
                )
                self.input_stats_[-1]["pca_variance_target_met"] = float(
                    np.asarray(pca.explained_variance_ratio_, dtype=np.float64).sum()
                ) >= float(self.pca_variance_target)
                self.pcas_.append(pca)
        transformed = self.transform(values)
        for layer in range(transformed.shape[1]):
            variances = transformed[:, layer, :].astype(np.float64).var(axis=0)
            self.output_stats_.append(
                {
                    "variance_mean": float(variances.mean()),
                    "variance_min": float(variances.min()) if variances.size else 0.0,
                    "variance_max": float(variances.max()) if variances.size else 0.0,
                }
            )
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = _check_layers(features)
        if self.input_shape_ is None or not self.means_:
            raise RuntimeError("LayerPreprocessor is not fitted")
        if tuple(values.shape[1:]) != self.input_shape_:
            raise ValueError(f"Expected layer shape {self.input_shape_}, got {tuple(values.shape[1:])}")
        if self.method == "none":
            return values.astype(np.float32, copy=True)
        if self.method == "pca_whiten":
            output = [
                pca.transform(values[:, layer, :].astype(np.float64)).astype(np.float32)
                for layer, pca in enumerate(self.pcas_)
            ]
            return np.stack(output, axis=1)
        out = np.empty_like(values, dtype=np.float32)
        for layer, mean in enumerate(self.means_):
            centered = values[:, layer, :] - mean
            if self.method == "diagonal":
                out[:, layer, :] = centered / self.scales_[layer]
            elif self.method == "zca":
                if self.zca_bases_:
                    basis = self.zca_bases_[layer]
                    projection = centered @ basis.T
                    out[:, layer, :] = (
                        centered * self.zca_residual_scales_[layer]
                        + (projection * self.zca_deltas_[layer]) @ basis
                    )
                else:
                    out[:, layer, :] = centered @ self.matrices_[layer]
        return out

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        return self.fit(features).transform(features)

    def to_metadata(self) -> dict[str, Any]:
        output_dim = None
        if self.input_shape_ is not None:
            output_dim = (
                int(self.pcas_[0].n_components_)
                if self.method == "pca_whiten" and self.pcas_
                else self.input_shape_[1]
            )
        return {
            "method": self.method,
            "epsilon": float(self.epsilon),
            "pca_components": int(self.pca_components),
            "pca_dimension_rule": "minimum_source_train_components_reaching_variance_target",
            "pca_variance_target": float(self.pca_variance_target),
            "random_state": int(self.random_state),
            "fit_rows": int(self.fit_rows_),
            "input_shape": list(self.input_shape_) if self.input_shape_ is not None else None,
            "output_dim": output_dim,
            "input_stats_by_layer": self.input_stats_,
            "output_stats_by_layer": self.output_stats_,
            "condition_number_max": (
                float(max(row["condition_number"] for row in self.input_stats_))
                if self.input_stats_
                else None
            ),
            "zca_solver": (
                "dual_low_rank" if self.method == "zca" and self.zca_bases_ else
                "full_eigendecomposition" if self.method == "zca" else None
            ),
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        if self.input_shape_ is None:
            raise RuntimeError("Cannot serialize an unfitted LayerPreprocessor")
        payload: dict[str, np.ndarray] = {"means": np.stack(self.means_, axis=0).astype(np.float32)}
        if self.scales_:
            payload["scales"] = np.stack(self.scales_, axis=0).astype(np.float32)
        if self.matrices_:
            payload["matrices"] = np.stack(self.matrices_, axis=0).astype(np.float32)
        if self.zca_bases_:
            payload["zca_bases"] = np.stack(self.zca_bases_, axis=0).astype(np.float32)
            payload["zca_deltas"] = np.stack(self.zca_deltas_, axis=0).astype(np.float32)
            payload["zca_residual_scales"] = np.asarray(
                self.zca_residual_scales_, dtype=np.float32
            )
        if self.pcas_:
            payload.update(
                {
                    "components": np.stack([pca.components_ for pca in self.pcas_], axis=0).astype(np.float32),
                    "pca_means": np.stack([pca.mean_ for pca in self.pcas_], axis=0).astype(np.float32),
                    "explained_variance": np.stack(
                        [pca.explained_variance_ for pca in self.pcas_], axis=0
                    ).astype(np.float32),
                    "explained_variance_ratio": np.stack(
                        [pca.explained_variance_ratio_ for pca in self.pcas_], axis=0
                    ).astype(np.float32),
                }
            )
        return payload


def _truncate_fitted_pca(pca: PCA, *, n_components: int) -> PCA:
    """Retain the leading fitted PCA basis without another decomposition."""

    available = int(pca.components_.shape[0])
    count = int(n_components)
    if count < 1 or count > available:
        raise ValueError(
            f"PCA truncation must be within 1..{available}, got {count}"
        )
    if count == available:
        return pca
    pca.components_ = pca.components_[:count].copy()
    pca.explained_variance_ = pca.explained_variance_[:count].copy()
    pca.explained_variance_ratio_ = pca.explained_variance_ratio_[:count].copy()
    pca.singular_values_ = pca.singular_values_[:count].copy()
    pca.n_components_ = count
    pca.n_components = count
    return pca


def flatten_layers(features: np.ndarray) -> np.ndarray:
    values = _check_layers(features)
    return values.reshape(values.shape[0], values.shape[1] * values.shape[2]).astype(np.float32)


def mean_pool_layers(features: np.ndarray) -> np.ndarray:
    return _check_layers(features).mean(axis=1).astype(np.float32)


def _check_layers(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"Expected [N,L,D] or [N,D] features, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Feature matrix contains NaN or inf")
    return values
