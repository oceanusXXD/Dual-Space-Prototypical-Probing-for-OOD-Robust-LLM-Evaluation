from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.covariance import LedoitWolf

from src.algorithm.detector.knn import Thresholds, calibrate_thresholds


OPENOOD_POSTHOC_METHODS = (
    "msp",
    "odin",
    "energy",
    "maxlogit",
    "mahalanobis",
    "react",
    "dice",
    "ash",
    "gen",
    "kl_matching",
    "gradnorm",
)

AFFINE_HEAD_METHODS = frozenset({"odin", "react", "dice", "ash"})


@dataclass
class OpenOODPosthocScorer:
    """OpenOOD mechanism ports over Judge penultimate features and logits.

    Head-transforming methods receive the deployed Judge's exact affine
    coefficients. ODIN's perturbation is applied at the postprocessor input
    (the penultimate representation), the feature-space analogue of its image
    input perturbation.
    """

    method: str
    regularization: float = 1e-5
    temperature: float = 1.0
    odin_temperature: float = 1000.0
    odin_epsilon: float = 0.0014
    react_quantile: float = 0.90
    dice_sparsity: float = 0.90
    ash_percentile: float = 65.0
    gen_gamma: float = 0.10
    thresholds_: Thresholds | None = None
    head_weight_: np.ndarray | None = None
    head_bias_: np.ndarray | None = None
    head_reconstruction_rmse_: float | None = None
    class_means_: np.ndarray | None = None
    shared_precision_: np.ndarray | None = None
    probability_prototypes_: np.ndarray | None = None
    react_threshold_: float | None = None
    dice_mask_: np.ndarray | None = None
    fit_rows_: int = 0
    class_count_: int = 0
    query_ids_: tuple[str, ...] = ()
    feature_dim_: int = 0
    head_parameter_source_: str | None = None

    def __post_init__(self) -> None:
        self.method = normalize_openood_method(self.method)
        if self.method not in OPENOOD_POSTHOC_METHODS:
            raise ValueError(f"Unsupported OpenOOD post-hoc method: {self.method!r}")
        if float(self.regularization) <= 0.0:
            raise ValueError("regularization must be positive")
        if float(self.temperature) <= 0.0 or float(self.odin_temperature) <= 0.0:
            raise ValueError("score temperatures must be positive")
        if float(self.odin_epsilon) < 0.0:
            raise ValueError("odin_epsilon must be non-negative")
        if not 0.0 < float(self.react_quantile) < 1.0:
            raise ValueError("react_quantile must be in (0, 1)")
        if not 0.0 <= float(self.dice_sparsity) < 1.0:
            raise ValueError("dice_sparsity must be in [0, 1)")
        if not 0.0 <= float(self.ash_percentile) < 100.0:
            raise ValueError("ash_percentile must be in [0, 100)")
        if float(self.gen_gamma) <= 0.0:
            raise ValueError("gen_gamma must be positive")

    def fit(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        labels: np.ndarray | None = None,
        query_ids: np.ndarray | None = None,
        *,
        head_weight: np.ndarray | None = None,
        head_bias: np.ndarray | None = None,
        head_query_ids: np.ndarray | None = None,
    ) -> "OpenOODPosthocScorer":
        h, raw_logits = _inputs(penultimate, logits)
        queries = _queries(query_ids, len(h))
        self.fit_rows_ = int(len(h))
        self.class_count_ = int(raw_logits.shape[1])
        self.query_ids_ = tuple(sorted(set(queries.tolist())))
        self.feature_dim_ = int(h.shape[1])
        if head_weight is not None or head_bias is not None or head_query_ids is not None:
            self._set_exact_head(
                h=h,
                logits=raw_logits,
                queries=queries,
                head_weight=head_weight,
                head_bias=head_bias,
                head_query_ids=head_query_ids,
            )
        elif self.method in AFFINE_HEAD_METHODS:
            raise ValueError(
                f"{self.method} requires exact deployed Judge head_weight/head_bias; "
                "least-squares head reconstruction is forbidden"
            )

        target = np.asarray(labels) if labels is not None else None
        if self.method == "mahalanobis":
            if target is None or len(target) != len(h):
                raise ValueError("Mahalanobis requires aligned source labels")
            classes = np.unique(target)
            self.class_means_ = np.stack([h[target == value].mean(axis=0) for value in classes])
            residuals = np.vstack(
                [h[target == value] - h[target == value].mean(axis=0) for value in classes]
            )
            covariance = np.atleast_2d(
                LedoitWolf().fit(residuals).covariance_
            ).astype(np.float64)
            covariance += float(self.regularization) * np.eye(covariance.shape[0])
            self.shared_precision_ = np.linalg.pinv(covariance)
        elif self.method == "kl_matching":
            probabilities = _softmax(raw_logits)
            predictions = np.argmax(raw_logits, axis=1)
            prototypes: list[np.ndarray] = []
            for class_index in range(self.class_count_):
                local = predictions == class_index
                if local.any():
                    prototypes.append(probabilities[local].mean(axis=0))
                else:
                    prototypes.append(np.eye(self.class_count_, dtype=np.float64)[class_index])
            self.probability_prototypes_ = np.stack(prototypes)
        elif self.method == "react":
            self.react_threshold_ = float(np.quantile(h, float(self.react_quantile)))
        elif self.method == "dice":
            masks: list[np.ndarray] = []
            for query_index, query in enumerate(self.query_ids_):
                contribution = h[queries == query].mean(axis=0)[:, None] * self.head_weight_[query_index]
                threshold = float(np.quantile(contribution, float(self.dice_sparsity)))
                masks.append((contribution >= threshold).astype(np.float64))
            self.dice_mask_ = np.stack(masks)
        return self

    def score(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        query_ids: np.ndarray | None = None,
    ) -> np.ndarray:
        h, raw_logits = _inputs(penultimate, logits)
        queries = _queries(query_ids, len(h))
        query_positions = self._require_fitted(h, raw_logits, queries)
        method = self.method
        if method == "msp":
            scores = 1.0 - np.max(_softmax(raw_logits), axis=1)
        elif method == "odin":
            probabilities = _softmax(raw_logits / float(self.odin_temperature))
            predicted = np.argmax(probabilities, axis=1)
            one_hot = np.eye(raw_logits.shape[1], dtype=np.float64)[predicted]
            assert self.head_weight_ is not None
            row_weights = self.head_weight_[query_positions]
            gradient = np.einsum("nk,ndk->nd", probabilities - one_hot, row_weights)
            perturbed = h - float(self.odin_epsilon) * np.sign(gradient)
            scores = 1.0 - np.max(
                _softmax(self._head_logits(perturbed, query_positions) / float(self.odin_temperature)),
                axis=1,
            )
        elif method == "energy":
            scores = _energy(raw_logits, float(self.temperature))
        elif method == "maxlogit":
            scores = -np.max(raw_logits, axis=1)
        elif method == "mahalanobis":
            assert self.class_means_ is not None and self.shared_precision_ is not None
            distances = []
            for mean in self.class_means_:
                centered = h - mean
                distances.append(
                    np.einsum("ij,jk,ik->i", centered, self.shared_precision_, centered)
                )
            scores = np.min(np.stack(distances, axis=1), axis=1)
        elif method == "react":
            assert self.react_threshold_ is not None
            scores = _energy(
                self._head_logits(np.minimum(h, float(self.react_threshold_)), query_positions),
                float(self.temperature),
            )
        elif method == "dice":
            assert self.head_weight_ is not None and self.head_bias_ is not None and self.dice_mask_ is not None
            weights = (self.head_weight_ * self.dice_mask_)[query_positions]
            logits = np.einsum("nd,ndk->nk", h, weights) + self.head_bias_[query_positions]
            scores = _energy(logits, float(self.temperature))
        elif method == "ash":
            keep = max(
                1,
                h.shape[1] - int(round(h.shape[1] * float(self.ash_percentile) / 100.0)),
            )
            top_indices = np.argpartition(h, kth=h.shape[1] - keep, axis=1)[:, -keep:]
            retained = np.zeros_like(h)
            fill = h.sum(axis=1) / float(keep)
            np.put_along_axis(retained, top_indices, fill[:, None], axis=1)
            scores = _energy(
                self._head_logits(retained, query_positions),
                float(self.temperature),
            )
        elif method == "gen":
            probabilities = _softmax(raw_logits)
            scores = np.sum(
                (probabilities**float(self.gen_gamma))
                * ((1.0 - probabilities) ** float(self.gen_gamma)),
                axis=1,
            )
        elif method == "kl_matching":
            assert self.probability_prototypes_ is not None
            probabilities = np.clip(_softmax(raw_logits), 1e-12, 1.0)
            prototypes = np.clip(self.probability_prototypes_, 1e-12, 1.0)
            divergence = np.sum(
                probabilities[:, None, :]
                * (np.log(probabilities[:, None, :]) - np.log(prototypes[None, :, :])),
                axis=2,
            )
            scores = np.min(divergence, axis=1)
        elif method == "gradnorm":
            probabilities = np.clip(_softmax(raw_logits), 1e-12, 1.0)
            log_probabilities = np.log(probabilities)
            expected_log_probability = np.sum(
                probabilities * log_probabilities,
                axis=1,
                keepdims=True,
            )
            # For KL(p || Uniform), dL/dlogit_j =
            # p_j * (log(p_j) - E_p[log p]). The affine-head parameter
            # gradient is its outer product with [h, 1] (bias included).
            logit_gradient = probabilities * (
                log_probabilities - expected_log_probability
            )
            gradient_l1 = np.sum(np.abs(logit_gradient), axis=1) * (
                np.sum(np.abs(h), axis=1) + 1.0
            )
            scores = -gradient_l1
        else:  # pragma: no cover - guarded by construction
            raise RuntimeError(f"Unhandled OpenOOD method: {method}")
        result = np.asarray(scores, dtype=np.float64)
        if result.shape != (len(h),) or not np.isfinite(result).all():
            raise RuntimeError(f"{self.method} produced invalid OOD scores")
        return result

    def calibrate(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        query_ids: np.ndarray | None = None,
        *,
        soft_q: float = 0.90,
        hard_q: float = 0.95,
    ) -> Thresholds:
        self.thresholds_ = calibrate_thresholds(
            self.score(penultimate, logits, query_ids),
            soft_q=soft_q,
            hard_q=hard_q,
        )
        return self.thresholds_

    def labels(self, scores: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("OpenOOD post-hoc thresholds are not calibrated")
        values = np.asarray(scores, dtype=np.float64)
        result = np.asarray(["id"] * len(values), dtype=object)
        result[values >= self.thresholds_.soft] = "soft_ood"
        result[values >= self.thresholds_.hard] = "hard_ood"
        return result

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        if self.head_weight_ is not None and self.head_bias_ is not None:
            arrays["head_weight"] = np.asarray(self.head_weight_, dtype=np.float32)
            arrays["head_bias"] = np.asarray(self.head_bias_, dtype=np.float32)
        for name, value in (
            ("class_means", self.class_means_),
            ("shared_precision", self.shared_precision_),
            ("probability_prototypes", self.probability_prototypes_),
            ("dice_mask", self.dice_mask_),
        ):
            if value is not None:
                arrays[name] = np.asarray(value, dtype=np.float32)
        if self.react_threshold_ is not None:
            arrays["react_threshold"] = np.asarray(self.react_threshold_, dtype=np.float64)
        return arrays

    def to_metadata(self) -> dict[str, Any]:
        return {
            "scorer": self.method,
            "suite": "openood_style_posthoc_mechanism_ports",
            "official_openood_numerical_equivalence_claimed": False,
            "primary_source_verification_status": (
                "pending_per_protocol_document"
                if self.method in {"dice", "ash", "gen", "kl_matching"}
                else "mechanism_port_with_repository_formula_tests"
            ),
            "feature_scope": "judge_behavior",
            "detection_unit": "judge_record",
            "fit_rows": int(self.fit_rows_),
            "class_count": int(self.class_count_),
            "query_ids": list(self.query_ids_),
            "score_direction": "higher_is_more_ood",
            "head_parameter_source": self.head_parameter_source_,
            "head_affine_consistency_rmse": self.head_reconstruction_rmse_,
            "covariance_estimator": (
                "shared_class_residual_ledoit_wolf"
                if self.method == "mahalanobis"
                else None
            ),
            "odin_perturbation_space": "judge_penultimate_feature" if self.method == "odin" else None,
            "ash_variant": "ASH-B" if self.method == "ash" else None,
            "parameters": {
                "regularization": float(self.regularization),
                "temperature": float(self.temperature),
                "odin_temperature": float(self.odin_temperature),
                "odin_epsilon": float(self.odin_epsilon),
                "react_quantile": float(self.react_quantile),
                "dice_sparsity": float(self.dice_sparsity),
                "ash_percentile": float(self.ash_percentile),
                "gen_gamma": float(self.gen_gamma),
            },
            "thresholds": self.thresholds_.to_dict() if self.thresholds_ is not None else None,
        }

    def _head_logits(self, features: np.ndarray, query_positions: np.ndarray) -> np.ndarray:
        self._require_fitted_arrays()
        assert self.head_weight_ is not None and self.head_bias_ is not None
        weights = self.head_weight_[np.asarray(query_positions, dtype=int)]
        return np.einsum("nd,ndk->nk", np.asarray(features, dtype=np.float64), weights) + self.head_bias_[
            np.asarray(query_positions, dtype=int)
        ]

    def _require_fitted(
        self,
        h: np.ndarray,
        logits: np.ndarray,
        queries: np.ndarray,
    ) -> np.ndarray:
        if self.fit_rows_ < 1:
            raise RuntimeError("OpenOODPosthocScorer is not fitted")
        if h.shape[1] != int(self.feature_dim_) or logits.shape[1] != int(self.class_count_):
            raise ValueError("OpenOOD features/logits do not match the fitted Judge head")
        mapping = {query: index for index, query in enumerate(self.query_ids_)}
        unknown = sorted(set(queries.tolist()) - set(mapping))
        if unknown:
            raise ValueError(f"OpenOOD scorer has no fitted head for query_ids={unknown}")
        return np.asarray([mapping[value] for value in queries.tolist()], dtype=int)

    def _set_exact_head(
        self,
        *,
        h: np.ndarray,
        logits: np.ndarray,
        queries: np.ndarray,
        head_weight: np.ndarray | None,
        head_bias: np.ndarray | None,
        head_query_ids: np.ndarray | None,
    ) -> None:
        if head_weight is None or head_bias is None:
            raise ValueError("Exact Judge head requires both head_weight and head_bias")
        weights = np.asarray(head_weight, dtype=np.float64)
        biases = np.asarray(head_bias, dtype=np.float64)
        if weights.ndim == 2:
            weights = weights[None, :, :]
        if biases.ndim == 1:
            biases = biases[None, :]
        parameter_queries = (
            np.asarray(head_query_ids).astype(str)
            if head_query_ids is not None
            else np.asarray(self.query_ids_, dtype=str)
        )
        if (
            weights.ndim != 3
            or biases.ndim != 2
            or weights.shape[0] != biases.shape[0]
            or weights.shape[0] != len(parameter_queries)
            or weights.shape[1] != h.shape[1]
            or weights.shape[2] != logits.shape[1]
            or biases.shape[1] != logits.shape[1]
            or len(set(parameter_queries.tolist())) != len(parameter_queries)
        ):
            raise ValueError("Exact Judge affine parameters have incompatible dimensions")
        parameter_map = {query: index for index, query in enumerate(parameter_queries.tolist())}
        unknown = sorted(set(queries.tolist()) - set(parameter_map))
        if unknown:
            raise ValueError(f"Exact Judge affine parameters are missing query_ids={unknown}")
        positions = np.asarray([parameter_map[value] for value in queries.tolist()], dtype=int)
        reconstructed = np.einsum("nd,ndk->nk", h, weights[positions]) + biases[positions]
        rmse = float(np.sqrt(np.mean((reconstructed - logits) ** 2)))
        scale = max(1.0, float(np.sqrt(np.mean(logits**2))))
        if rmse > 1e-5 * scale:
            raise ValueError(
                "Supplied Judge affine parameters do not reproduce the exposed logits "
                f"(relative RMSE={rmse / scale:.3e})"
            )
        ordered_positions = np.asarray([parameter_map[value] for value in self.query_ids_], dtype=int)
        self.head_weight_ = weights[ordered_positions]
        self.head_bias_ = biases[ordered_positions]
        self.head_reconstruction_rmse_ = rmse
        self.head_parameter_source_ = "exact_deployed_judge_affine_coefficients"

    def _require_fitted_arrays(self) -> None:
        if self.head_weight_ is None or self.head_bias_ is None:
            raise RuntimeError("OpenOODPosthocScorer is not fitted")


def normalize_openood_method(method: str) -> str:
    return str(method).strip().lower().replace("-", "_")


def _inputs(penultimate: np.ndarray, logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.asarray(penultimate, dtype=np.float64)
    raw_logits = np.asarray(logits, dtype=np.float64)
    if (
        h.ndim != 2
        or raw_logits.ndim != 2
        or len(h) != len(raw_logits)
        or len(h) == 0
        or h.shape[1] == 0
        or raw_logits.shape[1] < 2
        or not np.isfinite(h).all()
        or not np.isfinite(raw_logits).all()
    ):
        raise ValueError("OpenOOD requires aligned finite [N, D] features and [N, K] logits")
    return h, raw_logits


def _queries(query_ids: np.ndarray | None, rows: int) -> np.ndarray:
    if query_ids is None:
        return np.asarray(["__global__"] * int(rows), dtype=str)
    queries = np.asarray(query_ids).astype(str)
    if queries.shape != (int(rows),):
        raise ValueError("query_ids must align with OpenOOD feature rows")
    return queries


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    shifted = values - np.max(values, axis=1, keepdims=True)
    numerator = np.exp(shifted)
    return numerator / numerator.sum(axis=1, keepdims=True)


def _energy(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    maxima = np.max(scaled, axis=1, keepdims=True)
    return -float(temperature) * (
        maxima[:, 0] + np.log(np.exp(scaled - maxima).sum(axis=1))
    )
