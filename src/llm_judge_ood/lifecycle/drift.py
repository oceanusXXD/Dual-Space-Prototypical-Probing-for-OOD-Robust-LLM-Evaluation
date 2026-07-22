from __future__ import annotations

import hashlib
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import ks_2samp

from src.llm_judge_ood.lifecycle.cluster import ClusterConfig, DocumentClusterer
from src.llm_judge_ood.scores.vim import ViMScorer


@dataclass(frozen=True)
class WindowDriftConfig:
    """Configuration for the final-design dual-space drift decision chain."""

    window_size: int = 200
    minimum_window_documents: int = 50
    mmd_permutations: int = 1000
    primary_test: str = "mmd"
    c2st_enabled: bool = True
    c2st_folds: int = 5
    c2st_max_iter: int = 200
    c2st_regularization: float = 1.0
    reference_max_samples: int = 2000
    reference_subsample_threshold: int = 5000
    soft_alpha: float = 0.05
    hard_alpha: float = 0.01
    alpha_fwer: float = 0.05
    alpha_spending: str = "harmonic"
    pocock_horizon: int = 1000
    minimum_consecutive_windows: int = 3
    kernel_bandwidth: float | None = None
    minimum_valid_calibration_windows: int = 2
    require_sequential_fwer_calibration: bool = False
    sequential_calibration_trials: int = 200
    sequential_calibration_seed: int = 42
    sequential_calibration_cache_path: str | None = None
    allow_nominal_fallback_for_smoke: bool = False
    stop_after_first_persistent: bool = True
    power_enabled: bool = False
    power_trials: int = 20
    power_permutations: int = 49
    power_effect_sizes: tuple[float, ...] = (0.0, 0.25, 0.50, 1.0, 2.0)
    power_window_sizes: tuple[int, ...] = ()
    power_target: float = 0.80
    power_reference_max_samples: int = 512
    power_seed: int | None = None
    power_analysis_cache_path: str | None = None
    reference_cache_dir: str | None = None
    calibration_order_policy: str = "as_provided"
    seed: int = 42

    def __post_init__(self) -> None:
        if int(self.window_size) < 1 or int(self.minimum_window_documents) < 2:
            raise ValueError("window_size must be positive and minimum_window_documents must be at least two")
        if int(self.mmd_permutations) < 19:
            raise ValueError("mmd_permutations must be at least 19 for a usable empirical p-value")
        if str(self.primary_test).lower() not in {"mmd", "c2st", "ks"}:
            raise ValueError("primary_test must be 'mmd', 'c2st', or 'ks'")
        if str(self.primary_test).lower() == "c2st" and not bool(self.c2st_enabled):
            raise ValueError("primary_test='c2st' requires c2st_enabled=True")
        if int(self.c2st_folds) != 5:
            raise ValueError("The final C2ST protocol requires c2st_folds=5")
        if int(self.c2st_max_iter) < 1 or float(self.c2st_regularization) <= 0.0:
            raise ValueError("C2ST max_iter and regularization must be positive")
        if int(self.reference_max_samples) < 2:
            raise ValueError("reference_max_samples must be at least two")
        if int(self.reference_subsample_threshold) < int(self.reference_max_samples):
            raise ValueError(
                "reference_subsample_threshold must be at least reference_max_samples"
            )
        if not 0.0 < float(self.hard_alpha) <= float(self.soft_alpha) < 1.0:
            raise ValueError("drift thresholds must satisfy 0 < hard_alpha <= soft_alpha < 1")
        if not 0.0 < float(self.alpha_fwer) < 1.0:
            raise ValueError("alpha_fwer must be in (0, 1)")
        if str(self.alpha_spending) not in {"harmonic", "pocock"}:
            raise ValueError("alpha_spending must be 'harmonic' or 'pocock'")
        if int(self.pocock_horizon) < 1:
            raise ValueError("pocock_horizon must be positive")
        if int(self.minimum_consecutive_windows) < 1:
            raise ValueError("minimum_consecutive_windows must be positive")
        if str(self.alpha_spending) == "pocock" and int(self.minimum_consecutive_windows) > int(
            self.pocock_horizon
        ):
            raise ValueError("minimum_consecutive_windows cannot exceed pocock_horizon")
        if bool(self.require_sequential_fwer_calibration) and str(self.alpha_spending) != "pocock":
            raise ValueError("Formal sequential FWER calibration requires a finite Pocock horizon")
        if (
            bool(self.require_sequential_fwer_calibration)
            and str(self.primary_test).lower() == "mmd"
            and str(self.alpha_spending) == "pocock"
        ):
            horizon = int(self.pocock_horizon)
            cumulative_before_last = float(self.alpha_fwer) * math.log(
                1.0 + (math.e - 1.0) * float(horizon - 1) / float(horizon)
            )
            minimum_alpha = float(self.alpha_fwer) - cumulative_before_last
            resolution = 1.0 / float(int(self.mmd_permutations) + 1)
            if minimum_alpha + 1e-15 < resolution:
                raise ValueError(
                    "MMD conservative p-value resolution cannot reach every planned Pocock "
                    f"threshold: minimum_alpha={minimum_alpha:.8g}, resolution={resolution:.8g}"
                )
        if int(self.minimum_valid_calibration_windows) < 1:
            raise ValueError("minimum_valid_calibration_windows must be positive")
        if int(self.sequential_calibration_trials) < 1:
            raise ValueError("sequential_calibration_trials must be positive")
        int(self.sequential_calibration_seed)
        if self.sequential_calibration_cache_path is not None and not str(
            self.sequential_calibration_cache_path
        ).strip():
            raise ValueError(
                "sequential_calibration_cache_path must be a non-empty path when configured"
            )
        if int(self.power_trials) < 1 or int(self.power_permutations) < 19:
            raise ValueError("power_trials must be positive and power_permutations must be at least 19")
        if not self.power_effect_sizes or any(float(value) < 0.0 for value in self.power_effect_sizes):
            raise ValueError("power_effect_sizes must be non-empty and non-negative")
        if any(int(value) < 2 for value in self.power_window_sizes):
            raise ValueError("power_window_sizes must contain values of at least two")
        if not 0.0 < float(self.power_target) <= 1.0:
            raise ValueError("power_target must be in (0, 1]")
        if int(self.power_reference_max_samples) < 2:
            raise ValueError("power_reference_max_samples must be at least two")
        if self.power_seed is not None:
            int(self.power_seed)
        if self.power_analysis_cache_path is not None and not str(self.power_analysis_cache_path).strip():
            raise ValueError("power_analysis_cache_path must be a non-empty path when configured")
        if self.reference_cache_dir is not None and not str(self.reference_cache_dir).strip():
            raise ValueError("reference_cache_dir must be a non-empty path when configured")
        if str(self.calibration_order_policy) not in {"as_provided", "seeded_document_hash"}:
            raise ValueError(
                "calibration_order_policy must be 'as_provided' or 'seeded_document_hash'"
            )
        if self.kernel_bandwidth is not None and float(self.kernel_bandwidth) <= 0.0:
            raise ValueError("kernel_bandwidth must be positive when configured")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EffectiveSequentialConfig:
    """The calibrated alpha contract shared by selection and deployment."""

    calibrated_a_soft_alpha: float
    calibrated_a_hard_alpha: float
    calibrated_b_soft_alpha: float
    calibrated_b_hard_alpha: float
    sequential_scale: float
    effective_alpha_fwer: float
    alpha_spending: str
    minimum_consecutive_windows: int
    calibration_valid: bool
    calibration_window_count: int
    calibration_failure_reasons: tuple[str, ...]
    nominal_fallback_for_smoke: bool

    def tracker_config(self, base: WindowDriftConfig) -> WindowDriftConfig:
        return replace(base, alpha_fwer=float(self.effective_alpha_fwer))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_effective_sequential_config(
    config: WindowDriftConfig,
    calibrated_thresholds: dict[str, Any],
) -> EffectiveSequentialConfig:
    """Validate independent calibration and derive the deployed alpha budget.

    A formal run must have the configured number of valid windows in both
    spaces.  The nominal fallback remains available only for explicitly named
    smoke configurations, and is surfaced in metadata rather than silently
    promoted to a calibrated result.
    """

    spaces = {space: dict(calibrated_thresholds.get(space, {})) for space in ("A", "B")}
    failures: list[str] = []
    counts: list[int] = []
    for space, payload in spaces.items():
        count = int(payload.get("window_count", 0))
        counts.append(count)
        if (
            payload.get("status")
            != "nominal_thresholds_h0_audited_on_independent_training_calibration"
        ):
            failures.append(f"{space}_calibration_status={payload.get('status', 'missing')}")
        if count < int(config.minimum_valid_calibration_windows):
            failures.append(
                f"{space}_valid_calibration_windows={count}<minimum={int(config.minimum_valid_calibration_windows)}"
            )
        for threshold_name in ("soft_alpha", "hard_alpha"):
            value = payload.get(threshold_name)
            if value is None or not np.isfinite(float(value)) or not 0.0 < float(value) < 1.0:
                failures.append(f"{space}_{threshold_name}_invalid")
    if bool(config.require_sequential_fwer_calibration):
        sequential_audit = dict(spaces["B"].get("sequential_fwer_audit", {}))
        if not sequential_audit:
            b_p_values = np.asarray(
                spaces["B"].get("calibration_p_values", []), dtype=np.float64
            )
            sequential_audit = _sequential_fwer_audit(b_p_values, config)
        spaces["B"]["sequential_fwer_audit"] = sequential_audit
        if not bool(sequential_audit.get("valid")):
            failures.append(
                "B_sequential_fwer_calibration_invalid="
                + str(sequential_audit.get("reason", "wilson_upper_exceeds_alpha_fwer"))
            )
    calibration_valid = not failures
    smoke_fallback = bool(failures and config.allow_nominal_fallback_for_smoke)
    if failures and not smoke_fallback:
        raise ValueError(
            "Invalid independent calibration blocks deployment drift decisions: "
            + "; ".join(failures)
        )

    a_soft = float(spaces["A"].get("soft_alpha", config.soft_alpha))
    a_hard = float(spaces["A"].get("hard_alpha", config.hard_alpha))
    b_soft = float(spaces["B"].get("soft_alpha", config.soft_alpha))
    b_hard = float(spaces["B"].get("hard_alpha", config.hard_alpha))
    if smoke_fallback:
        a_soft = b_soft = float(config.soft_alpha)
        a_hard = b_hard = float(config.hard_alpha)
    sequential_scale = min(1.0, b_hard / max(float(config.hard_alpha), 1e-12))
    return EffectiveSequentialConfig(
        calibrated_a_soft_alpha=a_soft,
        calibrated_a_hard_alpha=a_hard,
        calibrated_b_soft_alpha=b_soft,
        calibrated_b_hard_alpha=b_hard,
        sequential_scale=float(sequential_scale),
        effective_alpha_fwer=float(config.alpha_fwer) * float(sequential_scale),
        alpha_spending=str(config.alpha_spending),
        minimum_consecutive_windows=int(config.minimum_consecutive_windows),
        calibration_valid=bool(calibration_valid),
        calibration_window_count=min(counts) if counts else 0,
        calibration_failure_reasons=tuple(failures),
        nominal_fallback_for_smoke=bool(smoke_fallback),
    )


@dataclass
class BehaviorMainRepresentation:
    """B-main = residual vectors from the source-fitted residual-only ViM."""

    rank: int = 128
    random_state: int = 42
    fit_scope: str = "training_train_judge_records_only"
    mean_: np.ndarray | None = None
    components_: np.ndarray | None = None
    fit_rows_: int = 0
    input_dim_: int | None = None

    def fit(
        self,
        penultimate: np.ndarray,
        *,
        scorer: ViMScorer,
    ) -> "BehaviorMainRepresentation":
        h = _matrix(penultimate, "B-main source penultimate")
        fitted = scorer
        if fitted.mean_ is None or fitted.components_ is None:
            raise ValueError("B-main requires a fitted residual-only ViM scorer")
        if fitted.mean_.shape != (h.shape[1],) or fitted.components_.shape[0] != h.shape[1]:
            raise ValueError("B-main ViM scorer does not match the source penultimate dimension")
        self.rank = int(fitted.components_.shape[1])
        self.mean_ = np.asarray(fitted.mean_, dtype=np.float64).copy()
        self.components_ = np.asarray(fitted.components_, dtype=np.float64).copy()
        self.fit_rows_ = int(fitted.fit_rows_)
        self.input_dim_ = int(h.shape[1])
        return self

    def transform(
        self,
        penultimate: np.ndarray,
    ) -> np.ndarray:
        if self.mean_ is None or self.components_ is None or self.input_dim_ is None:
            raise RuntimeError("BehaviorMainRepresentation is not fitted")
        h = _matrix(penultimate, "B-main penultimate")
        return self.transform_penultimate(h)

    def transform_penultimate(self, penultimate: np.ndarray) -> np.ndarray:
        """Project penultimate features onto the ViM residual complement."""

        if self.mean_ is None or self.components_ is None or self.input_dim_ is None:
            raise RuntimeError("BehaviorMainRepresentation is not fitted")
        h = _matrix(penultimate, "B-main penultimate")
        if h.shape[1] != int(self.input_dim_):
            raise ValueError("B-main penultimate dimension differs from the source-fitted ViM")
        centered = h - self.mean_
        projected = centered @ self.components_ @ self.components_.T
        return (centered - projected).astype(np.float64)

    def to_metadata(self) -> dict[str, Any]:
        residual_complement_dimension = (
            int(self.input_dim_) - int(self.components_.shape[1])
            if self.input_dim_ is not None and self.components_ is not None
            else None
        )
        return {
            "artifact_type": "llm_judge_ood_behavior_main_representation",
            "representation": "vim_source_subspace_residual_vector",
            "fit_scope": str(self.fit_scope),
            "fit_rows": int(self.fit_rows_),
            "penultimate_input_dim": self.input_dim_,
            "vim_rank": int(self.components_.shape[1]) if self.components_ is not None else None,
            "output_dim": int(self.input_dim_) if self.input_dim_ is not None else None,
            "residual_complement_dimension": residual_complement_dimension,
            "uses_logits": False,
            "excluded_from_main": [
                "raw_logits",
                "probabilities",
                "maximum_confidence",
                "ood_score",
            ],
            "auxiliary_scalar_test": "vim_residual_norm_via_KS",
        }

    def artifact_arrays(self) -> dict[str, np.ndarray]:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("Cannot serialize an unfitted B-main representation")
        return {
            "source_mean": self.mean_.astype(np.float32),
            "principal_components": self.components_.astype(np.float32),
        }


@dataclass
class MMDPermutationTest:
    """RBF-kernel MMD with the documented V-statistic and permutation p-value."""

    config: WindowDriftConfig
    reference_: np.ndarray | None = None
    reference_block_ids_: np.ndarray | None = None
    bandwidth_: float | None = None
    last_bandwidth_: float | None = None
    source_rows_: int = 0
    source_blocks_: int = 0

    def fit(self, source_features: np.ndarray, *, block_ids: np.ndarray | None = None) -> "MMDPermutationTest":
        source = _matrix(source_features, "MMD source features")
        source_blocks = _normalized_block_ids(source, block_ids, name="MMD source block IDs")
        self.source_rows_ = int(source.shape[0])
        self.source_blocks_ = int(len(np.unique(source_blocks)))
        self.reference_, self.reference_block_ids_ = _sample_rows(
            source,
            source_blocks,
            maximum_rows=int(self.config.reference_max_samples),
            subsample_threshold=int(self.config.reference_subsample_threshold),
            seed=int(self.config.seed),
        )
        if len(np.unique(self.reference_block_ids_)) < 2:
            raise ValueError("MMD requires at least two source blocks")
        self.bandwidth_ = (
            float(self.config.kernel_bandwidth)
            if self.config.kernel_bandwidth is not None
            else None
        )
        self.last_bandwidth_ = None
        return self

    def test(
        self,
        target_features: np.ndarray,
        *,
        seed: int,
        block_ids: np.ndarray | None = None,
    ) -> dict[str, float | int | str]:
        if self.reference_ is None or self.reference_block_ids_ is None:
            raise RuntimeError("MMDPermutationTest is not fitted")
        target = _matrix(target_features, "MMD target features")
        target_blocks = _normalized_block_ids(target, block_ids, name="MMD target block IDs")
        if target.shape[1] != self.reference_.shape[1]:
            raise ValueError("MMD target features do not match the fitted source dimension")
        if len(np.unique(target_blocks)) < 2:
            raise ValueError("MMD requires at least two target blocks")
        values = np.vstack([self.reference_, target])
        bandwidth = (
            float(self.bandwidth_)
            if self.bandwidth_ is not None
            else _median_bandwidth(values)
        )
        self.last_bandwidth_ = float(bandwidth)
        kernel = _rbf_kernel(values, bandwidth=float(bandwidth))
        n_source_rows = len(self.reference_)
        source_block_labels = np.asarray(
            [f"source::{value}" for value in self.reference_block_ids_.tolist()], dtype=str
        )
        target_block_labels = np.asarray(
            [f"target::{value}" for value in target_blocks.tolist()], dtype=str
        )
        pooled_blocks = np.concatenate([source_block_labels, target_block_labels])
        unique_blocks = np.unique(pooled_blocks)
        n_source_blocks = len(np.unique(source_block_labels))
        source_indices = np.arange(n_source_rows, dtype=int)
        target_indices = np.arange(n_source_rows, len(values), dtype=int)
        observed = _mmd_from_kernel(kernel, source_indices, target_indices)
        rng = np.random.default_rng(int(seed))
        strict_exceedances = 0
        ties = 1  # Include the observed assignment in the randomized rank.
        # The permutation unit is a block.  Aggregate the fixed row kernel to
        # that unit once, then evaluate each draw from the smaller assigned
        # domain.  This leaves the 1,000 random block partitions and the
        # V-statistic unchanged while avoiding an O(permutations * N^2) dense
        # matrix multiply for every window.
        _, block_indices = np.unique(pooled_blocks, return_inverse=True)
        block_kernel, block_sizes = _aggregate_kernel_by_block(kernel, block_indices)
        kernel_total = float(block_kernel.sum())
        block_row_sums = block_kernel.sum(axis=1)
        permutation_count = int(self.config.mmd_permutations)
        n_target_blocks = len(unique_blocks) - n_source_blocks
        select_source = n_source_blocks <= n_target_blocks
        selected_block_count = n_source_blocks if select_source else n_target_blocks
        # Bound the advanced-indexing temporary used for a batch of quadratic
        # forms.  The cap is independent of feature dimensionality and keeps
        # the formal 1,000 permutations viable on CPU hosts.
        batch_size = min(
            64,
            permutation_count,
            max(1, int(4_000_000 // max(selected_block_count**2, 1))),
        )
        for batch_start in range(0, permutation_count, batch_size):
            batch_stop = min(batch_start + batch_size, permutation_count)
            batch_count = int(batch_stop - batch_start)
            selected_blocks = np.empty(
                (batch_count, selected_block_count), dtype=np.intp
            )
            for row_index in range(batch_count):
                permutation = rng.permutation(len(unique_blocks))
                selected_blocks[row_index] = (
                    permutation[:n_source_blocks]
                    if select_source
                    else permutation[n_source_blocks:]
                )
            selected_self = block_kernel[
                selected_blocks[:, :, None], selected_blocks[:, None, :]
            ].sum(axis=(1, 2))
            selected_total = block_row_sums[selected_blocks].sum(axis=1)
            selected_counts = block_sizes[selected_blocks].sum(axis=1, dtype=np.float64)
            other_counts = float(len(values)) - selected_counts
            other_self = kernel_total - 2.0 * selected_total + selected_self
            cross = selected_total - selected_self
            source_self = selected_self if select_source else other_self
            target_self = other_self if select_source else selected_self
            source_counts = selected_counts if select_source else other_counts
            target_counts = other_counts if select_source else selected_counts
            statistics = (
                source_self / np.square(source_counts)
                + target_self / np.square(target_counts)
                - 2.0 * cross / (source_counts * target_counts)
            )
            strict_exceedances += int(np.sum(statistics > observed + 1e-12))
            ties += int(np.sum(np.abs(statistics - observed) <= 1e-12))
        denominator = int(self.config.mmd_permutations) + 1
        randomized_p_value = float((strict_exceedances + rng.random() * ties) / denominator)
        conservative_p_value = float((strict_exceedances + ties) / denominator)
        return {
            "statistic": float(observed),
            # Every formal consumer reads the top-level p_value. Randomized tie
            # breaking is retained only as an explicitly named diagnostic.
            "p_value": conservative_p_value,
            "conservative_p_value": conservative_p_value,
            "randomized_p_value": randomized_p_value,
            "p_value_method": "conservative_permutation_rank",
            "randomized_p_value_method": "randomized_permutation_rank_with_tie_breaking_diagnostic_only",
            "source_rows": int(n_source_rows),
            "target_rows": int(target.shape[0]),
            "source_blocks": int(n_source_blocks),
            "target_blocks": int(len(np.unique(target_block_labels))),
            "permutation_unit": "arrival_block" if block_ids is not None else "row",
            "permutations": int(self.config.mmd_permutations),
            "bandwidth": float(bandwidth),
            "bandwidth_policy": (
                "configured_fixed"
                if self.bandwidth_ is not None
                else "median_pairwise_distance_on_pooled_source_and_window"
            ),
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "test": "rbf_mmd_v_statistic_permutation",
            "source_rows": int(self.source_rows_),
            "source_blocks": int(self.source_blocks_),
            "reference_rows": int(self.reference_.shape[0]) if self.reference_ is not None else None,
            "reference_blocks": (
                int(len(np.unique(self.reference_block_ids_)))
                if self.reference_block_ids_ is not None
                else None
            ),
            "reference_max_rows": int(self.config.reference_max_samples),
            "reference_subsample_threshold": int(self.config.reference_subsample_threshold),
            "reference_subsample_threshold_role": "legacy_compatibility_only",
            "reference_sampling": "full_source_if_rows_at_most_maximum_block_complete_subsample_above_maximum",
            "configured_bandwidth": self.bandwidth_,
            "last_test_bandwidth": self.last_bandwidth_,
            "bandwidth_policy": (
                "configured_fixed"
                if self.bandwidth_ is not None
                else "median_pairwise_distance_on_pooled_source_and_window"
            ),
            "permutations": int(self.config.mmd_permutations),
            "p_value_method": "conservative_permutation_rank",
            "randomized_p_value_role": "diagnostic_only_never_used_for_decisions",
            "conservative_p_value_resolution": 1.0 / float(int(self.config.mmd_permutations) + 1),
        }


@dataclass
class BlockAwareC2ST:
    """Five-fold logistic C2ST with a Binomial reference tail probability."""

    config: WindowDriftConfig
    reference_: np.ndarray | None = None
    reference_block_ids_: np.ndarray | None = None
    source_rows_: int = 0
    source_blocks_: int = 0

    def fit(self, source_features: np.ndarray, *, block_ids: np.ndarray | None = None) -> "BlockAwareC2ST":
        source = _matrix(source_features, "C2ST source features")
        blocks = _normalized_block_ids(source, block_ids, name="C2ST source block IDs")
        self.source_rows_ = int(source.shape[0])
        self.source_blocks_ = int(len(np.unique(blocks)))
        self.reference_, self.reference_block_ids_ = _sample_rows(
            source,
            blocks,
            maximum_rows=int(self.config.reference_max_samples),
            subsample_threshold=int(self.config.reference_subsample_threshold),
            seed=int(self.config.seed) + 37,
        )
        if len(np.unique(self.reference_block_ids_)) < 2:
            raise ValueError("C2ST requires at least two source blocks")
        return self

    def test(
        self,
        target_features: np.ndarray,
        *,
        seed: int,
        block_ids: np.ndarray | None = None,
    ) -> dict[str, float | int | str | bool]:
        if self.reference_ is None or self.reference_block_ids_ is None:
            raise RuntimeError("BlockAwareC2ST is not fitted")
        target = _matrix(target_features, "C2ST target features")
        target_blocks = _normalized_block_ids(target, block_ids, name="C2ST target block IDs")
        if target.shape[1] != self.reference_.shape[1]:
            raise ValueError("C2ST target features do not match the fitted source dimension")
        source_block_values = np.unique(self.reference_block_ids_)
        target_block_values = np.unique(target_blocks)
        balanced_block_count = min(len(source_block_values), len(target_block_values))
        rng = np.random.default_rng(int(seed) + 5)
        selected_source_blocks = np.sort(
            rng.choice(source_block_values, size=balanced_block_count, replace=False)
        )
        selected_target_blocks = np.sort(
            rng.choice(target_block_values, size=balanced_block_count, replace=False)
        )
        source_mask = np.isin(self.reference_block_ids_, selected_source_blocks)
        target_mask = np.isin(target_blocks, selected_target_blocks)
        source = self.reference_[source_mask]
        source_blocks_used = self.reference_block_ids_[source_mask]
        target_used = target[target_mask]
        target_blocks_used = target_blocks[target_mask]
        values = np.vstack([source, target_used])
        truth = np.concatenate(
            [
                np.zeros(len(source), dtype=int),
                np.ones(len(target_used), dtype=int),
            ]
        )
        source_groups = np.asarray(
            [f"source::{value}" for value in source_blocks_used.tolist()], dtype=str
        )
        target_groups = np.asarray(
            [f"target::{value}" for value in target_blocks_used.tolist()], dtype=str
        )
        groups = np.concatenate([source_groups, target_groups])
        effective_folds = min(
            int(self.config.c2st_folds),
            int(len(np.unique(source_groups))),
            int(len(np.unique(target_groups))),
        )
        if effective_folds < 2:
            raise ValueError("C2ST needs at least two independent blocks from each domain")
        target_probability = np.empty(len(values), dtype=np.float64)
        folds = StratifiedGroupKFold(
            n_splits=int(effective_folds),
            shuffle=True,
            random_state=int(seed) + 11,
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            folds.split(values, truth, groups=groups)
        ):
            classifier = LogisticRegression(
                C=float(self.config.c2st_regularization),
                max_iter=int(self.config.c2st_max_iter),
                class_weight="balanced",
                random_state=int(seed) + fold_index,
                solver="lbfgs",
            ).fit(values[train_indices], truth[train_indices])
            target_probability[test_indices] = classifier.predict_proba(values[test_indices])[:, 1]
        predicted = (target_probability >= 0.5).astype(int)
        row_correct = int(np.sum(predicted == truth))
        unique_groups = np.unique(groups)
        block_truth = np.asarray([truth[groups == group][0] for group in unique_groups], dtype=int)
        block_prediction = np.asarray(
            [int(target_probability[groups == group].mean() >= 0.5) for group in unique_groups],
            dtype=int,
        )
        correct = int(np.sum(block_prediction == block_truth))
        test_blocks = int(len(unique_groups))
        p_value = _binomial_upper_tail(correct, test_blocks, probability=0.5)
        return {
            "statistic": float(correct / test_blocks),
            "accuracy": float(correct / test_blocks),
            "block_accuracy": float(correct / test_blocks),
            "row_accuracy": float(row_correct / len(truth)),
            "correct_blocks": correct,
            "test_blocks": test_blocks,
            "correct_rows": row_correct,
            "test_rows": int(len(truth)),
            "folds": int(effective_folds),
            "configured_folds": int(self.config.c2st_folds),
            "p_value": float(p_value),
            "conservative_p_value": float(p_value),
            "p_value_method": "approximate_one_sided_binomial_reference_on_balanced_oof_blocks",
            "p_value_role": "diagnostic_only_for_formal_mmd_protocol",
            "source_rows": int(len(source)),
            "target_rows": int(len(target_used)),
            "source_rows_before_block_balance": int(len(self.reference_)),
            "target_rows_before_block_balance": int(len(target)),
            "source_blocks": int(balanced_block_count),
            "target_blocks": int(balanced_block_count),
            "source_blocks_before_balance": int(len(source_block_values)),
            "target_blocks_before_balance": int(len(target_block_values)),
            "domain_blocks_balanced_for_binomial_null": True,
            "classifier": "logistic_regression",
            "selection_used_deployment_labels": False,
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "test": "five_fold_logistic_c2st",
            "source_rows": int(self.source_rows_),
            "source_blocks": int(self.source_blocks_),
            "reference_rows": int(self.reference_.shape[0]) if self.reference_ is not None else None,
            "reference_blocks": (
                int(len(np.unique(self.reference_block_ids_)))
                if self.reference_block_ids_ is not None
                else None
            ),
            "reference_max_rows": int(self.config.reference_max_samples),
            "reference_subsample_threshold": int(self.config.reference_subsample_threshold),
            "reference_subsample_threshold_role": "legacy_compatibility_only",
            "reference_sampling": "full_source_if_rows_at_most_maximum_block_complete_subsample_above_maximum",
            "folds": int(self.config.c2st_folds),
            "classifier": "logistic_regression",
            "heldout_unit": "arrival_block",
            "p_value_method": "approximate_one_sided_binomial_reference_on_balanced_oof_blocks",
            "p_value_role": "diagnostic_only_for_formal_mmd_protocol",
        }


@dataclass
class ScalarKSTest:
    """Two-sided KS auxiliary test for the selected scalar B-space OOD score."""

    reference_: np.ndarray | None = None

    def fit(self, source_scores: np.ndarray) -> "ScalarKSTest":
        self.reference_ = _score_vector(source_scores, "KS source scores")
        return self

    def test(self, target_scores: np.ndarray) -> dict[str, float | int | str]:
        if self.reference_ is None:
            raise RuntimeError("ScalarKSTest is not fitted")
        target = _score_vector(target_scores, "KS target scores")
        result = ks_2samp(self.reference_, target, alternative="two-sided", method="auto")
        return {
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "conservative_p_value": float(result.pvalue),
            "p_value_method": "scipy_two_sided_ks_2samp_auto",
            "source_rows": int(len(self.reference_)),
            "target_rows": int(len(target)),
            "input": "selected_judge_behavior_ood_scalar_score",
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "test": "two_sided_ks_2samp",
            "reference_rows": int(len(self.reference_)) if self.reference_ is not None else None,
            "input": "selected_judge_behavior_ood_scalar_score",
            "role": "B_space_auxiliary_test",
        }

class AlphaSpendingTracker:
    """Sequential B-space tracker with a strict, auditable FWER spending cap."""

    def __init__(self, config: WindowDriftConfig) -> None:
        self.config = config
        self.consecutive_rejections = 0
        self.wealth = float(config.alpha_fwer)
        self.spent = 0.0

    def allocation(self, window_index: int) -> float:
        if str(self.config.alpha_spending) == "harmonic":
            # A telescoping gamma sequence keeps sum_t alpha_t <= alpha_FWER
            # over an unbounded stream. Rejections never mint extra FWER.
            t = int(window_index) + 1
            return float(self.config.alpha_fwer) / float(t * (t + 1))
        t = int(window_index) + 1
        horizon = int(self.config.pocock_horizon)
        information_now = min(float(t) / horizon, 1.0)
        information_before = min(float(t - 1) / horizon, 1.0)
        cumulative_now = float(self.config.alpha_fwer) * math.log(
            1.0 + (math.e - 1.0) * information_now
        )
        cumulative_before = float(self.config.alpha_fwer) * math.log(
            1.0 + (math.e - 1.0) * information_before
        )
        return max(0.0, float(cumulative_now - cumulative_before))

    def update(self, *, window_index: int, p_value: float) -> dict[str, Any]:
        wealth_before = float(self.wealth)
        alpha_t = self.allocation(int(window_index))
        horizon_exhausted = bool(
            str(self.config.alpha_spending) == "pocock"
            and int(window_index) >= int(self.config.pocock_horizon)
        )
        rejected = bool(
            not horizon_exhausted and np.isfinite(p_value) and float(p_value) <= alpha_t
        )
        self.spent = min(float(self.config.alpha_fwer), self.spent + alpha_t)
        self.wealth = max(0.0, float(self.config.alpha_fwer) - self.spent)
        self.consecutive_rejections = self.consecutive_rejections + 1 if rejected else 0
        return {
            "alpha_t": float(alpha_t),
            "alpha_wealth_before": wealth_before,
            "alpha_wealth_after": float(self.wealth),
            "alpha_spent_cumulative": float(self.spent),
            "b_sequential_reject": rejected,
            "monitoring_horizon_exhausted": horizon_exhausted,
            "planned_monitoring_horizon": (
                int(self.config.pocock_horizon)
                if str(self.config.alpha_spending) == "pocock"
                else None
            ),
            "consecutive_b_rejections": int(self.consecutive_rejections),
            "persistent_b_drift": bool(
                self.consecutive_rejections >= int(self.config.minimum_consecutive_windows)
            ),
        }

    def break_consecutive_segment(self) -> None:
        self.consecutive_rejections = 0


@dataclass
class DualSpaceDriftReference:
    """Source/calibration-only state reusable across stream replays.

    The object contains fitted sklearn objects and no development/deployment
    rows.  Its signature binds disk and process-local reuse to the exact
    source/calibration arrays, records, blocks, and test settings.
    """

    signature: str
    a_test: MMDPermutationTest
    b_test: MMDPermutationTest
    a_c2st: BlockAwareC2ST | None
    b_c2st: BlockAwareC2ST | None
    b_ks: ScalarKSTest
    calibration_rows: list[dict[str, Any]]
    calibrated_thresholds: dict[str, Any]
    cache_metadata: dict[str, Any]


@dataclass
class DualSpaceDriftResult:
    window_rows: list[dict[str, Any]]
    calibration_rows: list[dict[str, Any]]
    persistent_document_indices: np.ndarray
    source_metadata: dict[str, Any]
    power_analysis: dict[str, Any]
    calibrated_thresholds: dict[str, Any]
    effective_sequential_config: EffectiveSequentialConfig
    first_persistent_episode: dict[str, Any] | None
    config: WindowDriftConfig
    reference: DualSpaceDriftReference

    def to_metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "tests": ["rbf_mmd_v_statistic_permutation", "two_sided_ks_2samp_B_auxiliary"] + (
                ["five_fold_block_grouped_logistic_c2st"] if self.config.c2st_enabled else []
            ),
            "primary_test": str(self.config.primary_test).lower(),
            "spaces": {
                "A": "input_document_embedding",
                "B": "source_fitted_vim_residual_vector",
            },
            "detection_unit": {"A": "input_document", "B": "judge_record"},
            "decision_rule": {
                "window": f"two_sample_{str(self.config.primary_test).lower()}_p_values_in_A_and_B",
                "persistence": "consecutive_B_rejections_under_alpha_spending",
                "clustering": "localization_only_after_persistent_B_drift",
            },
            "source": self.source_metadata,
            "calibration_window_count": int(len(self.calibration_rows)),
            "calibration": {
                **_calibration_summary(self.calibration_rows, self.config),
                "calibrated_thresholds": self.calibrated_thresholds,
                "formal_calibration_valid": bool(self.effective_sequential_config.calibration_valid),
                "calibration_failure_reasons": list(
                    self.effective_sequential_config.calibration_failure_reasons
                ),
                "nominal_fallback_for_smoke": bool(
                    self.effective_sequential_config.nominal_fallback_for_smoke
                ),
            },
            "reference_cache": dict(self.reference.cache_metadata),
            "effective_sequential_config": self.effective_sequential_config.to_dict(),
            "stream_window_count": int(len(self.window_rows)),
            "persistent_document_count": int(self.persistent_document_indices.size),
            "first_persistent_episode": self.first_persistent_episode,
            "window_results": [_window_metadata(row) for row in self.window_rows],
            "calibration_results": [_calibration_metadata(row) for row in self.calibration_rows],
            "power_analysis": self.power_analysis,
            "selection_used_deployment_records": False,
        }


def ordered_calibration_document_indices(
    indices: np.ndarray,
    document_ids: np.ndarray,
    config: WindowDriftConfig,
) -> np.ndarray:
    """Return the configured, reproducible calibration arrival order."""

    documents = np.asarray(indices, dtype=int)
    policy = str(config.calibration_order_policy)
    if policy == "as_provided":
        return documents.copy()
    ids = np.asarray(document_ids).astype(str)
    if documents.size and (int(documents.min()) < 0 or int(documents.max()) >= len(ids)):
        raise ValueError("calibration document indices are out of bounds")
    ordered = sorted(
        documents.tolist(),
        key=lambda index: (
            hashlib.sha256(
                f"{int(config.seed)}::calibration::{ids[int(index)]}".encode("utf-8")
            ).digest(),
            ids[int(index)],
        ),
    )
    return np.asarray(ordered, dtype=int)


def _drift_reference_signature(
    *,
    config: WindowDriftConfig,
    document_ids: np.ndarray,
    source_document_indices: np.ndarray,
    calibration_document_indices: np.ndarray,
    source_behavior_indices: np.ndarray,
    calibration_behavior_indices: np.ndarray,
    block_ids: np.ndarray,
    document_embeddings: np.ndarray,
    behavior_embeddings: np.ndarray,
    behavior_ood_scores: np.ndarray,
) -> str:
    """Bind reusable reference state to every input that can affect it."""

    ids = np.asarray(document_ids).astype(str)
    blocks = np.asarray(block_ids).astype(str)
    source_documents = np.asarray(source_document_indices, dtype=int)
    calibration_documents = np.asarray(calibration_document_indices, dtype=int)
    source_behavior = np.asarray(source_behavior_indices, dtype=int)
    calibration_behavior = np.asarray(calibration_behavior_indices, dtype=int)
    digest = hashlib.sha256()
    digest.update(b"dual_space_drift_reference_v6_sequential_mc_hard_reference_cap\0")
    reference_config = {
        "window_size": int(config.window_size),
        "minimum_window_documents": int(config.minimum_window_documents),
        "mmd_permutations": int(config.mmd_permutations),
        "primary_test": str(config.primary_test),
        "c2st_enabled": bool(config.c2st_enabled),
        "c2st_folds": int(config.c2st_folds),
        "c2st_max_iter": int(config.c2st_max_iter),
        "c2st_regularization": float(config.c2st_regularization),
        "reference_max_samples": int(config.reference_max_samples),
        "reference_subsample_threshold": int(config.reference_subsample_threshold),
        "soft_alpha": float(config.soft_alpha),
        "hard_alpha": float(config.hard_alpha),
        "minimum_valid_calibration_windows": int(config.minimum_valid_calibration_windows),
        "require_sequential_fwer_calibration": bool(
            config.require_sequential_fwer_calibration
        ),
        "sequential_calibration_trials": int(config.sequential_calibration_trials),
        "sequential_calibration_seed": int(config.sequential_calibration_seed),
        "sequential_calibration_cache_path": config.sequential_calibration_cache_path,
        "alpha_fwer": float(config.alpha_fwer),
        "alpha_spending": str(config.alpha_spending),
        "pocock_horizon": int(config.pocock_horizon),
        "minimum_consecutive_windows": int(config.minimum_consecutive_windows),
        "allow_nominal_fallback_for_smoke": bool(config.allow_nominal_fallback_for_smoke),
        "calibration_order_policy": str(config.calibration_order_policy),
        "kernel_bandwidth": (
            None if config.kernel_bandwidth is None else float(config.kernel_bandwidth)
        ),
        "seed": int(config.seed),
    }
    digest.update(
        json.dumps(reference_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values in (
        ("a_source", document_embeddings[source_documents]),
        ("a_calibration", document_embeddings[calibration_documents]),
        ("b_source", behavior_embeddings[source_behavior]),
        ("b_calibration", behavior_embeddings[calibration_behavior]),
        ("b_score_source", behavior_ood_scores[source_behavior]),
        ("b_score_calibration", behavior_ood_scores[calibration_behavior]),
    ):
        _update_array_digest(digest, name=name, values=values)
    for name, values in (
        ("source_document_ids", ids[source_documents]),
        ("calibration_document_ids", ids[calibration_documents]),
        ("source_behavior_ids", ids[source_behavior]),
        ("calibration_behavior_ids", ids[calibration_behavior]),
        ("source_document_blocks", blocks[source_documents]),
        ("calibration_document_blocks", blocks[calibration_documents]),
        ("source_behavior_blocks", blocks[source_behavior]),
        ("calibration_behavior_blocks", blocks[calibration_behavior]),
    ):
        _update_string_array_digest(digest, name=name, values=values)
    return digest.hexdigest()


def run_dual_space_drift_monitor(
    *,
    document_embeddings: np.ndarray,
    behavior_embeddings: np.ndarray,
    document_ids: np.ndarray,
    source_document_indices: np.ndarray,
    calibration_document_indices: np.ndarray,
    stream_document_indices: np.ndarray,
    source_behavior_indices: np.ndarray,
    calibration_behavior_indices: np.ndarray,
    config: WindowDriftConfig,
    permutation_block_ids: np.ndarray | None = None,
    behavior_ood_scores: np.ndarray | None = None,
    reference: DualSpaceDriftReference | None = None,
) -> DualSpaceDriftResult:
    """Run A/B two-sample windows without fitting or tuning on deployment records.

    ``stream_document_indices`` contains one canonical row per document. B-space
    windows expand each document to all of its Judge records before testing.
    """

    a_values = _matrix(document_embeddings, "A-space embeddings")
    b_values = _matrix(behavior_embeddings, "B-space embeddings")
    ids = np.asarray(document_ids).astype(str)
    if len(a_values) != len(b_values) or len(a_values) != len(ids):
        raise ValueError("A-space, B-space, and document IDs must align")
    source_documents = _unique_indices(source_document_indices, ids, "source_document_indices")
    calibration_documents = ordered_calibration_document_indices(
        _unique_indices(calibration_document_indices, ids, "calibration_document_indices"),
        ids,
        config,
    )
    stream_documents = _unique_indices(stream_document_indices, ids, "stream_document_indices")
    source_behavior = _indices(source_behavior_indices, len(ids), "source_behavior_indices")
    calibration_behavior = _indices(calibration_behavior_indices, len(ids), "calibration_behavior_indices")
    if source_documents.size < 2 or calibration_documents.size < 2 or stream_documents.size < 1:
        raise ValueError("dual-space drift monitoring needs source, calibration, and stream documents")
    if source_behavior.size < 2 or calibration_behavior.size < 2:
        raise ValueError("dual-space drift monitoring needs source and calibration Judge records")
    source_document_ids = set(ids[source_documents].tolist())
    calibration_document_ids = set(ids[calibration_documents].tolist())
    stream_document_ids = set(ids[stream_documents].tolist())
    if source_document_ids & calibration_document_ids:
        raise ValueError("source and calibration document sets must be disjoint")
    if source_document_ids & stream_document_ids:
        raise ValueError("source and stream document sets must be disjoint")
    if calibration_document_ids & stream_document_ids:
        raise ValueError("calibration and stream document sets must be disjoint")

    source_behavior_set = set(source_behavior.tolist())
    calibration_behavior_set = set(calibration_behavior.tolist())
    if source_behavior_set & calibration_behavior_set:
        raise ValueError("source and calibration behavior records must be disjoint")
    source_behavior_documents = set(ids[source_behavior].tolist())
    calibration_behavior_documents = set(ids[calibration_behavior].tolist())
    if source_behavior_documents != source_document_ids:
        raise ValueError("source behavior records must cover exactly the source documents")
    if calibration_behavior_documents != calibration_document_ids:
        raise ValueError("calibration behavior records must cover exactly the calibration documents")
    document_to_records = _document_to_indices(ids)
    stream_behavior = _records_for_documents(stream_documents, ids, document_to_records)
    if source_behavior_set & set(stream_behavior.tolist()):
        raise ValueError("source behavior records overlap stream documents")
    if calibration_behavior_set & set(stream_behavior.tolist()):
        raise ValueError("calibration behavior records overlap stream documents")

    blocks = np.asarray(permutation_block_ids).astype(str) if permutation_block_ids is not None else ids
    if blocks.shape != ids.shape:
        raise ValueError("permutation_block_ids must align with document IDs")
    b_scores = (
        _score_vector(behavior_ood_scores, "B-space OOD scores")
        if behavior_ood_scores is not None
        else np.asarray(b_values[:, -1], dtype=np.float64)
    )
    if b_scores.shape != (len(ids),):
        raise ValueError("behavior_ood_scores must align with document IDs")
    expected_reference_signature = _drift_reference_signature(
        config=config,
        document_ids=ids,
        source_document_indices=source_documents,
        calibration_document_indices=calibration_documents,
        source_behavior_indices=source_behavior,
        calibration_behavior_indices=calibration_behavior,
        block_ids=blocks,
        document_embeddings=a_values,
        behavior_embeddings=b_values,
        behavior_ood_scores=b_scores,
    )
    reference_was_provided = reference is not None
    if reference is not None and reference.signature != expected_reference_signature:
        raise ValueError(
            "Dual-space drift reference does not match the source/calibration records or test settings"
        )
    reference_cache_path = (
        Path(config.reference_cache_dir) / f"{expected_reference_signature}.joblib"
        if config.reference_cache_dir is not None
        else None
    )
    if reference is None and reference_cache_path is not None:
        reference_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_file_lock(reference_cache_path):
            reference = _load_drift_reference_cache(
                reference_cache_path,
                signature=expected_reference_signature,
            )
            if reference is None:
                reference = _fit_drift_reference(
                    signature=expected_reference_signature,
                    a_values=a_values,
                    b_values=b_values,
                    b_scores=b_scores,
                    ids=ids,
                    blocks=blocks,
                    source_documents=source_documents,
                    calibration_documents=calibration_documents,
                    source_behavior=source_behavior,
                    calibration_behavior=calibration_behavior,
                    document_to_records=document_to_records,
                    config=config,
                    cache_metadata={
                        "status": "miss_created",
                        "path": str(reference_cache_path),
                        "signature": expected_reference_signature,
                        "selection_used_development_or_deployment_records": False,
                    },
                )
                _write_drift_reference_cache(reference_cache_path, reference=reference)
            else:
                reference.cache_metadata = {
                    "status": "disk_hit",
                    "path": str(reference_cache_path),
                    "signature": expected_reference_signature,
                    "selection_used_development_or_deployment_records": False,
                }
    if reference is None:
        reference = _fit_drift_reference(
            signature=expected_reference_signature,
            a_values=a_values,
            b_values=b_values,
            b_scores=b_scores,
            ids=ids,
            blocks=blocks,
            source_documents=source_documents,
            calibration_documents=calibration_documents,
            source_behavior=source_behavior,
            calibration_behavior=calibration_behavior,
            document_to_records=document_to_records,
            config=config,
            cache_metadata={
                "status": "disabled",
                "path": None,
                "signature": expected_reference_signature,
                "selection_used_development_or_deployment_records": False,
            },
        )
    elif reference_was_provided:
        origin_status = reference.cache_metadata.get("status")
        reference.cache_metadata = {
            **reference.cache_metadata,
            "status": "in_process_hit",
            "origin_status": origin_status,
        }
    a_test = reference.a_test
    b_test = reference.b_test
    a_c2st = reference.a_c2st
    b_c2st = reference.b_c2st
    b_ks = reference.b_ks
    calibration_rows = reference.calibration_rows
    calibrated_thresholds = reference.calibrated_thresholds
    effective_sequential_config = derive_effective_sequential_config(
        config, calibrated_thresholds
    )
    power_analysis = _dual_space_power_analysis(
        a_source=a_values[source_documents],
        a_calibration=a_values[calibration_documents],
        a_source_documents=ids[source_documents],
        a_calibration_documents=ids[calibration_documents],
        a_source_blocks=blocks[source_documents],
        a_calibration_blocks=blocks[calibration_documents],
        b_source=b_values[source_behavior],
        b_calibration=b_values[calibration_behavior],
        b_source_documents=ids[source_behavior],
        b_calibration_documents=ids[calibration_behavior],
        b_source_blocks=blocks[source_behavior],
        b_calibration_blocks=blocks[calibration_behavior],
        config=config,
    )
    tracker = AlphaSpendingTracker(effective_sequential_config.tracker_config(config))
    rows: list[dict[str, Any]] = []
    persistent_documents: set[int] = set()
    active_rejection_segment: list[np.ndarray] = []
    first_persistent_episode: dict[str, Any] | None = None
    for window_index, start in enumerate(range(0, len(stream_documents), int(config.window_size))):
        if (
            str(config.alpha_spending) == "pocock"
            and int(window_index) >= int(config.pocock_horizon)
        ):
            break
        stop = min(start + int(config.window_size), len(stream_documents))
        document_window = stream_documents[start:stop]
        if document_window.size < int(config.minimum_window_documents):
            rows.append(
                {
                    "window_index": int(window_index),
                    "window_start": int(start),
                    "window_stop": int(stop),
                    "document_indices": document_window.astype(int).tolist(),
                    "document_count": int(document_window.size),
                    "status": "insufficient_window_documents",
                    "selection_used_deployment_records": False,
                }
            )
            tracker.break_consecutive_segment()
            active_rejection_segment = []
            continue
        record_window = _records_for_documents(document_window, ids, document_to_records)
        # Block-aware tests cannot form a valid permutation/classifier split
        # when a small window contains only one arrival/source block.  Keep
        # the window in the audit trail and defer the decision instead of
        # silently falling back to row-level permutations.
        a_block_count = len(np.unique(blocks[document_window]))
        b_block_count = len(np.unique(blocks[record_window]))
        if a_block_count < 2 or b_block_count < 2:
            rows.append(
                {
                    "window_index": int(window_index),
                    "window_start": int(start),
                    "window_stop": int(stop),
                    "document_indices": document_window.astype(int).tolist(),
                    "document_count": int(document_window.size),
                    "judge_record_count": int(record_window.size),
                    "status": "insufficient_target_blocks",
                    "A": {
                        "status": "insufficient_target_blocks",
                        "target_blocks": int(a_block_count),
                    },
                    "B": {
                        "status": "insufficient_target_blocks",
                        "target_blocks": int(b_block_count),
                    },
                    "selection_used_deployment_records": False,
                }
            )
            tracker.break_consecutive_segment()
            active_rejection_segment = []
            continue
        a_mmd_result = a_test.test(
            a_values[document_window],
            seed=_seed(config.seed, window_index, 11),
            block_ids=blocks[document_window],
        )
        b_mmd_result = b_test.test(
            b_values[record_window],
            seed=_seed(config.seed, window_index, 29),
            block_ids=blocks[record_window],
        )
        a_c2st_result = (
            a_c2st.test(
                a_values[document_window],
                seed=_seed(config.seed, window_index, 41),
                block_ids=blocks[document_window],
            )
            if a_c2st is not None
            else None
        )
        b_c2st_result = (
            b_c2st.test(
                b_values[record_window],
                seed=_seed(config.seed, window_index, 59),
                block_ids=blocks[record_window],
            )
            if b_c2st is not None
            else None
        )
        b_ks_result = b_ks.test(b_scores[record_window])
        a_result = _combined_test_result(a_mmd_result, a_c2st_result, config)
        b_result = _combined_test_result(
            b_mmd_result,
            b_c2st_result,
            config,
            ks_result=b_ks_result,
        )
        a_status = _window_status(
            float(a_result["p_value"]),
            soft_alpha=float(effective_sequential_config.calibrated_a_soft_alpha),
            hard_alpha=float(effective_sequential_config.calibrated_a_hard_alpha),
        )
        b_status = _window_status(
            float(b_result["p_value"]),
            soft_alpha=float(effective_sequential_config.calibrated_b_soft_alpha),
            hard_alpha=float(effective_sequential_config.calibrated_b_hard_alpha),
        )
        sequential = tracker.update(window_index=window_index, p_value=float(b_result["p_value"]))
        if sequential["b_sequential_reject"]:
            active_rejection_segment.append(document_window.copy())
        else:
            active_rejection_segment = []
        if sequential["persistent_b_drift"]:
            persistent_documents.update(
                int(index)
                for segment in active_rejection_segment
                for index in segment.tolist()
            )
        row = {
            "window_index": int(window_index),
            "window_start": int(start),
            "window_stop": int(stop),
            "document_indices": document_window.astype(int).tolist(),
            "document_count": int(document_window.size),
            "judge_record_count": int(record_window.size),
            "A": {**a_result, "status": a_status},
            "B": {**b_result, "status": b_status},
            "quadrant": _quadrant(a_status, b_status),
            **sequential,
            "status": "persistent_b_drift" if sequential["persistent_b_drift"] else b_status,
            "selection_used_deployment_records": False,
        }
        rows.append(row)
        if sequential["persistent_b_drift"] and first_persistent_episode is None:
            active_indices = np.asarray(
                [
                    int(index)
                    for segment in active_rejection_segment
                    for index in segment.tolist()
                ],
                dtype=int,
            )
            first_persistent_episode = {
                "confirmation_window": int(window_index),
                "confirmation_window_start": int(start),
                "confirmation_window_stop": int(stop),
                "active_rejection_segment_document_indices": active_indices.tolist(),
                "visible_document_indices": stream_documents[:stop].astype(int).tolist(),
                "alpha_spending_state": {
                    key: value
                    for key, value in sequential.items()
                    if key
                    in {
                        "alpha_t",
                        "alpha_wealth_before",
                        "alpha_wealth_after",
                        "alpha_spent_cumulative",
                        "b_sequential_reject",
                        "consecutive_b_rejections",
                        "persistent_b_drift",
                    }
                },
                "effective_sequential_config": effective_sequential_config.to_dict(),
            }
            if bool(config.stop_after_first_persistent):
                break
    return DualSpaceDriftResult(
        window_rows=rows,
        calibration_rows=calibration_rows,
        persistent_document_indices=np.asarray(sorted(persistent_documents), dtype=int),
        source_metadata={
            "A": {
                "source_documents": int(source_documents.size),
                "calibration_documents": int(calibration_documents.size),
                "calibration_order_policy": str(config.calibration_order_policy),
                "input_preprocessing": "none_after_documented_shared_pca",
                "mmd": a_test.to_metadata(),
                "c2st": a_c2st.to_metadata() if a_c2st is not None else {"enabled": False},
            },
            "B": {
                "source_judge_records": int(source_behavior.size),
                "calibration_judge_records": int(calibration_behavior.size),
                "calibration_order_policy": str(config.calibration_order_policy),
                "input_preprocessing": "source_fitted_vim_residual_projection_no_logits",
                "mmd": b_test.to_metadata(),
                "c2st": b_c2st.to_metadata() if b_c2st is not None else {"enabled": False},
                "ks": b_ks.to_metadata(),
            },
        },
        power_analysis=power_analysis,
        calibrated_thresholds=calibrated_thresholds,
        effective_sequential_config=effective_sequential_config,
        first_persistent_episode=first_persistent_episode,
        config=config,
        reference=reference,
    )


def _fit_drift_reference(
    *,
    signature: str,
    a_values: np.ndarray,
    b_values: np.ndarray,
    b_scores: np.ndarray,
    ids: np.ndarray,
    blocks: np.ndarray,
    source_documents: np.ndarray,
    calibration_documents: np.ndarray,
    source_behavior: np.ndarray,
    calibration_behavior: np.ndarray,
    document_to_records: dict[str, np.ndarray],
    config: WindowDriftConfig,
    cache_metadata: dict[str, Any],
) -> DualSpaceDriftReference:
    """Fit the source/calibration-only state once for replay and disk reuse."""

    a_test = MMDPermutationTest(config).fit(
        a_values[source_documents], block_ids=blocks[source_documents]
    )
    b_test = MMDPermutationTest(config).fit(
        b_values[source_behavior], block_ids=blocks[source_behavior]
    )
    a_c2st = (
        BlockAwareC2ST(config).fit(
            a_values[source_documents],
            block_ids=blocks[source_documents],
        )
        if config.c2st_enabled
        else None
    )
    b_c2st = (
        BlockAwareC2ST(config).fit(
            b_values[source_behavior],
            block_ids=blocks[source_behavior],
        )
        if config.c2st_enabled
        else None
    )
    b_ks = ScalarKSTest().fit(b_scores[source_behavior])
    calibration_rows = _evaluate_id_calibration(
        a_test=a_test,
        b_test=b_test,
        a_c2st=a_c2st,
        b_c2st=b_c2st,
        b_ks=b_ks,
        a_values=a_values,
        b_values=b_values,
        b_scores=b_scores,
        document_ids=ids,
        document_to_records=document_to_records,
        calibration_documents=calibration_documents,
        block_ids=blocks,
        config=config,
    )
    sequential_fwer_audit = (
        _cached_sequential_monte_carlo_audit(
            b_test=b_test,
            b_values=b_values,
            document_ids=ids,
            document_to_records=document_to_records,
            calibration_documents=calibration_documents,
            block_ids=blocks,
            config=config,
        )
        if bool(config.require_sequential_fwer_calibration)
        else None
    )
    calibrated_thresholds = _calibrate_p_value_thresholds(
        calibration_rows,
        config,
        sequential_fwer_audit=sequential_fwer_audit,
    )
    return DualSpaceDriftReference(
        signature=signature,
        a_test=a_test,
        b_test=b_test,
        a_c2st=a_c2st,
        b_c2st=b_c2st,
        b_ks=b_ks,
        calibration_rows=calibration_rows,
        calibrated_thresholds=calibrated_thresholds,
        cache_metadata=cache_metadata,
    )


def _load_drift_reference_cache(
    cache_path: Path,
    *,
    signature: str,
) -> DualSpaceDriftReference | None:
    try:
        payload = joblib.load(cache_path)
    except (OSError, EOFError, ValueError, TypeError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    reference = payload.get("reference")
    if (
        payload.get("artifact_type") != "llm_judge_ood_dual_space_drift_reference"
        or payload.get("signature") != signature
        or not isinstance(reference, DualSpaceDriftReference)
        or reference.signature != signature
    ):
        return None
    return reference


def _write_drift_reference_cache(
    cache_path: Path,
    *,
    reference: DualSpaceDriftReference,
) -> None:
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    joblib.dump(
        {
            "artifact_type": "llm_judge_ood_dual_space_drift_reference",
            "signature": reference.signature,
            "reference": reference,
        },
        temporary,
        compress=3,
    )
    os.replace(temporary, cache_path)


def cluster_persistent_documents(
    *,
    document_embeddings: np.ndarray,
    persistent_document_indices: np.ndarray,
    window_rows: list[dict[str, Any]],
    config: ClusterConfig,
    localization_mask: np.ndarray | None = None,
    cluster_space: str = "A_input_document_embedding",
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    """Cluster only after persistence is established; clusters never create it."""

    indices = np.asarray(persistent_document_indices, dtype=int)
    if indices.size == 0:
        return [], np.zeros(0, dtype=int), np.zeros(0, dtype=str)
    if not str(cluster_space):
        raise ValueError("cluster_space must be a non-empty audit label")
    values = _matrix(document_embeddings, "document embeddings")
    if np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("persistent document indices are out of bounds")
    if localization_mask is not None:
        candidates = np.asarray(localization_mask, dtype=bool)
        if candidates.shape != (len(values),):
            raise ValueError("localization_mask must align with document embeddings")
        indices = indices[candidates[indices]]
        if indices.size == 0:
            return [], np.zeros(0, dtype=int), np.zeros(0, dtype=str)
    labels, summaries = DocumentClusterer(config).fit_predict(values[indices])
    density_cluster_found = bool(summaries)
    if summaries and np.any(labels < 0) and int(np.sum(labels < 0)) >= int(config.min_cluster_size):
        noise_mask = labels < 0
        noise_indices = indices[noise_mask]
        labels = labels.copy()
        review_id = int(max(summary["cluster_id"] for summary in summaries) + 1)
        labels[noise_mask] = review_id
        centroid = values[noise_indices].mean(axis=0)
        summaries.append(
            {
                "cluster_id": review_id,
                "size": int(noise_indices.size),
                "compactness": float(
                    np.mean(np.linalg.norm(values[noise_indices] - centroid, axis=1))
                ),
                "centroid": centroid.astype(float).tolist(),
                "cluster_origin": f"{config.method}_noise_review_stratum",
                "density_cluster_found": False,
                "raw_noise_count": int(noise_indices.size),
            }
        )
    elif not summaries and indices.size >= int(config.min_cluster_size):
        # Persistent group drift can be diffuse in the localization space even
        # when every contributor is a B-space residual outlier. Preserve one explicit
        # review stratum so Probe can fail closed instead of silently dropping
        # the episode. This is not reported as an HDBSCAN density cluster.
        labels = np.zeros(indices.size, dtype=int)
        centroid = values[indices].mean(axis=0)
        summaries = [
            {
                "cluster_id": 0,
                "size": int(indices.size),
                "compactness": float(
                    np.mean(np.linalg.norm(values[indices] - centroid, axis=1))
                ),
                "centroid": centroid.astype(float).tolist(),
                "cluster_origin": f"{config.method}_all_noise_unclustered_review_stratum",
                "density_cluster_found": False,
                "raw_noise_count": int(indices.size),
            }
        ]
    first_rejection_windows: dict[int, tuple[int, int, int]] = {}
    confirmation_windows: dict[int, tuple[int, int, int]] = {}
    active_rejection_segment: set[int] = set()
    for row in window_rows:
        if bool(row.get("b_sequential_reject")):
            window = (
                int(row["window_index"]),
                int(row["window_start"]),
                int(row["window_stop"]),
            )
            for index in row.get("document_indices", []):
                document_index = int(index)
                active_rejection_segment.add(document_index)
                first_rejection_windows.setdefault(document_index, window)
        else:
            active_rejection_segment.clear()
        if bool(row.get("persistent_b_drift")):
            confirmation = (
                int(row["window_index"]),
                int(row["window_start"]),
                int(row["window_stop"]),
            )
            for index in active_rejection_segment:
                confirmation_windows.setdefault(index, confirmation)
    rows: list[dict[str, Any]] = []
    assigned = np.full(indices.shape[0], "-1", dtype=object)
    for summary in summaries:
        local_id = int(summary["cluster_id"])
        members = indices[labels == local_id]
        document_cluster_id = f"C{local_id + 1:04d}"
        assigned[labels == local_id] = document_cluster_id
        missing_windows = [
            int(index)
            for index in members.tolist()
            if int(index) not in first_rejection_windows
            or int(index) not in confirmation_windows
        ]
        if missing_windows:
            raise RuntimeError(
                "Persistent cluster members are missing alpha-spending lifecycle windows"
            )
        member_first_windows = [
            first_rejection_windows[int(index)] for index in members.tolist()
        ]
        member_windows = [confirmation_windows[int(index)] for index in members.tolist()]
        first_window, first_window_start, first_window_stop = min(member_first_windows)
        confirmation_window, window_start, window_stop = min(member_windows) if member_windows else (0, 0, 0)
        latency_windows = max(0, int(confirmation_window) - int(first_window))
        latency_samples = max(0, int(window_stop) - int(first_window_stop))
        rows.append(
            {
                **summary,
                "cluster_origin": summary.get(
                    "cluster_origin", f"{config.method}_density_cluster"
                ),
                "density_cluster_found": bool(
                    summary.get("density_cluster_found", density_cluster_found)
                ),
                "raw_noise_count": int(summary.get("raw_noise_count", 0)),
                "document_cluster_id": document_cluster_id,
                "status": "persistent_document_cluster",
                "cluster_space": str(cluster_space),
                "decision_basis": "persistent_B_space_alpha_spending",
                "window_index": int(confirmation_window),
                "window_start": int(window_start),
                "window_stop": int(window_stop),
                "first_window": int(first_window),
                "first_window_start": int(first_window_start),
                "first_window_stop": int(first_window_stop),
                "confirmation_window": int(confirmation_window),
                "confirmation_latency_windows": int(latency_windows),
                "confirmation_latency_samples": int(latency_samples),
                "member_indices": members.astype(int).tolist(),
                "window_share": float(len(members) / max(len(indices), 1)),
            }
        )
    return rows, indices, assigned


def _evaluate_id_calibration(
    *,
    a_test: MMDPermutationTest,
    b_test: MMDPermutationTest,
    a_c2st: BlockAwareC2ST | None,
    b_c2st: BlockAwareC2ST | None,
    b_ks: ScalarKSTest,
    a_values: np.ndarray,
    b_values: np.ndarray,
    b_scores: np.ndarray,
    document_ids: np.ndarray,
    document_to_records: dict[str, np.ndarray],
    calibration_documents: np.ndarray,
    block_ids: np.ndarray,
    config: WindowDriftConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block_index, start in enumerate(range(0, len(calibration_documents), int(config.window_size))):
        stop = min(start + int(config.window_size), len(calibration_documents))
        documents = calibration_documents[start:stop]
        if documents.size < int(config.minimum_window_documents):
            continue
        records = _records_for_documents(documents, document_ids, document_to_records)
        a_block_count = len(np.unique(block_ids[documents]))
        b_block_count = len(np.unique(block_ids[records]))
        if a_block_count < 2 or b_block_count < 2:
            rows.append(
                {
                    "calibration_window_index": int(block_index),
                    "document_count": int(documents.size),
                    "judge_record_count": int(records.size),
                    "status": "insufficient_target_blocks",
                    "A": {"status": "insufficient_target_blocks", "target_blocks": int(a_block_count)},
                    "B": {"status": "insufficient_target_blocks", "target_blocks": int(b_block_count)},
                    "scope": "independent_training_calibration",
                    "calibration_order_policy": str(config.calibration_order_policy),
                }
            )
            continue
        a_mmd_result = a_test.test(
            a_values[documents],
            seed=_seed(config.seed, block_index, 101),
            block_ids=block_ids[documents],
        )
        b_mmd_result = b_test.test(
            b_values[records],
            seed=_seed(config.seed, block_index, 211),
            block_ids=block_ids[records],
        )
        a_c2st_result = (
            a_c2st.test(
                a_values[documents],
                seed=_seed(config.seed, block_index, 307),
                block_ids=block_ids[documents],
            )
            if a_c2st is not None
            else None
        )
        b_c2st_result = (
            b_c2st.test(
                b_values[records],
                seed=_seed(config.seed, block_index, 401),
                block_ids=block_ids[records],
            )
            if b_c2st is not None
            else None
        )
        b_ks_result = b_ks.test(b_scores[records])
        rows.append(
            {
                "calibration_window_index": int(block_index),
                "document_count": int(documents.size),
                "judge_record_count": int(records.size),
                "A": _combined_test_result(a_mmd_result, a_c2st_result, config),
                "B": _combined_test_result(
                    b_mmd_result,
                    b_c2st_result,
                    config,
                    ks_result=b_ks_result,
                ),
                "scope": "independent_training_calibration",
                "calibration_order_policy": str(config.calibration_order_policy),
            }
        )
    return rows


def _combined_test_result(
    mmd_result: dict[str, Any],
    c2st_result: dict[str, Any] | None,
    config: WindowDriftConfig,
    *,
    ks_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    primary_name = str(config.primary_test).lower()
    if primary_name == "ks" and ks_result is None:
        # KS is a scalar B-space auxiliary test; A-space remains multivariate MMD.
        primary_name = "mmd"
    primary = {
        "mmd": mmd_result,
        "c2st": c2st_result,
        "ks": ks_result,
    }[primary_name]
    if primary is None:
        raise RuntimeError(f"Configured primary two-sample test {primary_name!r} is unavailable")
    return {
        **primary,
        "primary_test": primary_name,
        "configured_primary_test": str(config.primary_test).lower(),
        "mmd": dict(mmd_result),
        "c2st": dict(c2st_result) if c2st_result is not None else {"enabled": False},
        "ks": dict(ks_result) if ks_result is not None else {"enabled": False},
    }


def _dual_space_power_analysis(
    *,
    a_source: np.ndarray,
    a_calibration: np.ndarray,
    a_source_documents: np.ndarray,
    a_calibration_documents: np.ndarray,
    a_source_blocks: np.ndarray,
    a_calibration_blocks: np.ndarray,
    b_source: np.ndarray,
    b_calibration: np.ndarray,
    b_source_documents: np.ndarray,
    b_calibration_documents: np.ndarray,
    b_source_blocks: np.ndarray,
    b_calibration_blocks: np.ndarray,
    config: WindowDriftConfig,
) -> dict[str, Any]:
    if not bool(config.power_enabled):
        return {
            "enabled": False,
            "scope": "source_and_independent_training_calibration_only",
            "selection_used_deployment_records": False,
        }
    power_seed = int(config.power_seed) if config.power_seed is not None else int(config.seed)
    cache_path = (
        Path(config.power_analysis_cache_path)
        if config.power_analysis_cache_path is not None
        else None
    )
    signature = _power_analysis_signature(
        a_source=a_source,
        a_calibration=a_calibration,
        a_source_documents=a_source_documents,
        a_calibration_documents=a_calibration_documents,
        a_source_blocks=a_source_blocks,
        a_calibration_blocks=a_calibration_blocks,
        b_source=b_source,
        b_calibration=b_calibration,
        b_source_documents=b_source_documents,
        b_calibration_documents=b_calibration_documents,
        b_source_blocks=b_source_blocks,
        b_calibration_blocks=b_calibration_blocks,
        config=config,
        power_seed=power_seed,
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_file_lock(cache_path):
            cached = _load_power_analysis_cache(cache_path, signature=signature)
            if cached is not None:
                return {
                    **cached,
                    "cache": {
                        "status": "hit",
                        "path": str(cache_path),
                        "signature": signature,
                        "power_seed": power_seed,
                    },
                }
            analysis = _compute_dual_space_power_analysis(
                a_source=a_source,
                a_calibration=a_calibration,
                a_source_documents=a_source_documents,
                a_calibration_documents=a_calibration_documents,
                a_source_blocks=a_source_blocks,
                a_calibration_blocks=a_calibration_blocks,
                b_source=b_source,
                b_calibration=b_calibration,
                b_source_documents=b_source_documents,
                b_calibration_documents=b_calibration_documents,
                b_source_blocks=b_source_blocks,
                b_calibration_blocks=b_calibration_blocks,
                config=config,
                power_seed=power_seed,
            )
            _write_power_analysis_cache(cache_path, signature=signature, analysis=analysis)
        return {
            **analysis,
            "cache": {
                "status": "miss_created",
                "path": str(cache_path),
                "signature": signature,
                "power_seed": power_seed,
            },
        }
    return _compute_dual_space_power_analysis(
        a_source=a_source,
        a_calibration=a_calibration,
        a_source_documents=a_source_documents,
        a_calibration_documents=a_calibration_documents,
        a_source_blocks=a_source_blocks,
        a_calibration_blocks=a_calibration_blocks,
        b_source=b_source,
        b_calibration=b_calibration,
        b_source_documents=b_source_documents,
        b_calibration_documents=b_calibration_documents,
        b_source_blocks=b_source_blocks,
        b_calibration_blocks=b_calibration_blocks,
        config=config,
        power_seed=power_seed,
    )


def _compute_dual_space_power_analysis(
    *,
    a_source: np.ndarray,
    a_calibration: np.ndarray,
    a_source_documents: np.ndarray,
    a_calibration_documents: np.ndarray,
    a_source_blocks: np.ndarray,
    a_calibration_blocks: np.ndarray,
    b_source: np.ndarray,
    b_calibration: np.ndarray,
    b_source_documents: np.ndarray,
    b_calibration_documents: np.ndarray,
    b_source_blocks: np.ndarray,
    b_calibration_blocks: np.ndarray,
    config: WindowDriftConfig,
    power_seed: int,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "scope": "source_and_independent_training_calibration_only",
        "effect_definition": "mean_shift_in_source_standard_deviation_units_along_source_PC1",
        "alpha": float(config.hard_alpha),
        "target_power": float(config.power_target),
        "trials": int(config.power_trials),
        "spaces": {
            "A": _space_power_analysis(
                source=a_source,
                calibration=a_calibration,
                source_documents=a_source_documents,
                calibration_documents=a_calibration_documents,
                source_blocks=a_source_blocks,
                calibration_blocks=a_calibration_blocks,
                config=config,
                seed=int(power_seed) + 701,
            ),
            "B": _space_power_analysis(
                source=b_source,
                calibration=b_calibration,
                source_documents=b_source_documents,
                calibration_documents=b_calibration_documents,
                source_blocks=b_source_blocks,
                calibration_blocks=b_calibration_blocks,
                config=config,
                seed=int(power_seed) + 907,
            ),
        },
        "selection_used_deployment_records": False,
        "selection_used_deployment_labels": False,
    }


def _update_array_digest(
    digest: Any,
    *,
    name: str,
    values: np.ndarray,
) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(memoryview(array).cast("B"))


def _update_string_array_digest(
    digest: Any,
    *,
    name: str,
    values: np.ndarray,
) -> None:
    encoded = json.dumps(
        np.asarray(values).astype(str).tolist(),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(name.encode("utf-8") + b"\0" + encoded)


def _power_analysis_signature(
    *,
    a_source: np.ndarray,
    a_calibration: np.ndarray,
    a_source_documents: np.ndarray,
    a_calibration_documents: np.ndarray,
    a_source_blocks: np.ndarray,
    a_calibration_blocks: np.ndarray,
    b_source: np.ndarray,
    b_calibration: np.ndarray,
    b_source_documents: np.ndarray,
    b_calibration_documents: np.ndarray,
    b_source_blocks: np.ndarray,
    b_calibration_blocks: np.ndarray,
    config: WindowDriftConfig,
    power_seed: int,
) -> str:
    """Fingerprint every source/calibration-only power-analysis input."""

    digest = hashlib.sha256()
    digest.update(b"dual_space_power_analysis_v3_hard_reference_cap\0")
    power_config = {
        "mmd_permutations": int(config.power_permutations),
        "c2st_enabled": bool(config.c2st_enabled),
        "c2st_folds": int(config.c2st_folds),
        "c2st_max_iter": int(config.c2st_max_iter),
        "c2st_regularization": float(config.c2st_regularization),
        "reference_max_samples": int(config.reference_max_samples),
        "reference_subsample_threshold": int(
            min(config.reference_max_samples, config.power_reference_max_samples)
        ),
        "kernel_bandwidth": (
            None if config.kernel_bandwidth is None else float(config.kernel_bandwidth)
        ),
        "hard_alpha": float(config.hard_alpha),
        "power_trials": int(config.power_trials),
        "power_effect_sizes": [float(value) for value in config.power_effect_sizes],
        "power_window_sizes": [int(value) for value in config.power_window_sizes],
        "power_target": float(config.power_target),
        "power_reference_max_samples": int(config.power_reference_max_samples),
        "power_seed": int(power_seed),
    }
    digest.update(
        json.dumps(power_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name, values in (
        ("a_source", a_source),
        ("a_calibration", a_calibration),
        ("b_source", b_source),
        ("b_calibration", b_calibration),
    ):
        matrix = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(matrix.shape).encode("ascii") + b"\0")
        digest.update(memoryview(matrix).cast("B"))
    for name, values in (
        ("a_source_documents", a_source_documents),
        ("a_calibration_documents", a_calibration_documents),
        ("a_source_blocks", a_source_blocks),
        ("a_calibration_blocks", a_calibration_blocks),
        ("b_source_documents", b_source_documents),
        ("b_calibration_documents", b_calibration_documents),
        ("b_source_blocks", b_source_blocks),
        ("b_calibration_blocks", b_calibration_blocks),
    ):
        encoded = json.dumps(
            np.asarray(values).astype(str).tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(name.encode("utf-8") + b"\0" + encoded)
    return digest.hexdigest()


@contextmanager
def _exclusive_file_lock(cache_path: Path):
    """Serialize cache creation without making a valid cache mandatory."""

    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX local fallback.
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _load_power_analysis_cache(cache_path: Path, *, signature: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    analysis = payload.get("power_analysis")
    if (
        payload.get("artifact_type") != "llm_judge_ood_dual_space_power_analysis"
        or payload.get("signature") != signature
        or not isinstance(analysis, dict)
        or not bool(analysis.get("enabled"))
    ):
        return None
    return analysis


def _write_power_analysis_cache(
    cache_path: Path,
    *,
    signature: str,
    analysis: dict[str, Any],
) -> None:
    payload = {
        "artifact_type": "llm_judge_ood_dual_space_power_analysis",
        "signature": signature,
        "power_analysis": analysis,
    }
    temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, cache_path)


def _space_power_analysis(
    *,
    source: np.ndarray,
    calibration: np.ndarray,
    source_documents: np.ndarray,
    calibration_documents: np.ndarray,
    source_blocks: np.ndarray,
    calibration_blocks: np.ndarray,
    config: WindowDriftConfig,
    seed: int,
) -> dict[str, Any]:
    source_values = _matrix(source, "power source features")
    calibration_values = _matrix(calibration, "power calibration features")
    source_block_ids = _normalized_block_ids(
        source_values, source_blocks, name="power source block IDs"
    )
    calibration_block_ids = _normalized_block_ids(
        calibration_values, calibration_blocks, name="power calibration block IDs"
    )
    source_document_ids = np.asarray(source_documents).astype(str)
    calibration_document_ids = np.asarray(calibration_documents).astype(str)
    if source_document_ids.shape != (len(source_values),) or calibration_document_ids.shape != (
        len(calibration_values),
    ):
        raise ValueError("Power-analysis document IDs must align with feature rows")
    unique_calibration_documents = np.unique(calibration_document_ids)
    if len(unique_calibration_documents) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer_than_two_independent_calibration_documents",
        }
    centered = source_values - source_values.mean(axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    direction = right[0]
    direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
    pc1_scale = max(float(np.std(centered @ direction)), 1e-12)
    power_reference_max_samples = min(
        int(config.reference_max_samples), int(config.power_reference_max_samples)
    )
    power_config = replace(
        config,
        seed=int(seed),
        mmd_permutations=int(config.power_permutations),
        reference_max_samples=power_reference_max_samples,
        # ``_sample_rows`` intentionally keeps a full source below this
        # threshold.  Power analysis has an explicit smaller reference budget,
        # so its threshold must be lowered with the cap rather than inheriting
        # the main-monitoring 5,000-row threshold.
        reference_subsample_threshold=power_reference_max_samples,
        power_enabled=False,
    )
    mmd = MMDPermutationTest(power_config).fit(source_values, block_ids=source_block_ids)
    c2st = (
        BlockAwareC2ST(power_config).fit(source_values, block_ids=source_block_ids)
        if config.c2st_enabled
        else None
    )
    window_sizes = (
        tuple(int(value) for value in config.power_window_sizes)
        if config.power_window_sizes
        else (int(config.window_size),)
    )
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    document_rows = {
        document_id: np.flatnonzero(calibration_document_ids == document_id)
        for document_id in unique_calibration_documents.tolist()
    }
    for window_size in window_sizes:
        for effect_size in sorted(set(float(value) for value in config.power_effect_sizes)):
            trial_inputs: list[tuple[np.ndarray, np.ndarray, int]] = []
            for trial in range(int(config.power_trials)):
                target, target_blocks = _bootstrap_documents(
                    calibration_values,
                    calibration_document_ids,
                    document_count=int(window_size),
                    rng=rng,
                    document_rows=document_rows,
                )
                shifted = target + float(effect_size) * pc1_scale * direction[None, :]
                trial_seed = _seed(seed, window_size * 10_000 + trial, int(round(effect_size * 10_000)))
                trial_inputs.append((shifted, target_blocks, trial_seed))
            if joblib.effective_n_jobs(-1) > 1:
                with joblib.parallel_backend("loky", inner_max_num_threads=1):
                    outcomes = joblib.Parallel(n_jobs=-1)(
                        joblib.delayed(_power_trial_rejections)(
                            mmd=mmd,
                            c2st=c2st,
                            shifted=shifted,
                            target_blocks=target_blocks,
                            trial_seed=trial_seed,
                            hard_alpha=float(config.hard_alpha),
                        )
                        for shifted, target_blocks, trial_seed in trial_inputs
                    )
            else:
                outcomes = [
                    _power_trial_rejections(
                        mmd=mmd,
                        c2st=c2st,
                        shifted=shifted,
                        target_blocks=target_blocks,
                        trial_seed=trial_seed,
                        hard_alpha=float(config.hard_alpha),
                    )
                    for shifted, target_blocks, trial_seed in trial_inputs
                ]
            mmd_rejections = sum(int(outcome[0]) for outcome in outcomes)
            c2st_rejections = sum(int(outcome[1]) for outcome in outcomes)
            rows.append(
                {
                    "window_documents": int(window_size),
                    "effect_size": float(effect_size),
                    "trials": int(config.power_trials),
                    "mmd_power": float(mmd_rejections / int(config.power_trials)),
                    "mmd_power_ci95": wilson_interval(
                        mmd_rejections,
                        int(config.power_trials),
                    ),
                    "c2st_power": (
                        float(c2st_rejections / int(config.power_trials))
                        if c2st is not None
                        else None
                    ),
                    "c2st_power_ci95": (
                        wilson_interval(c2st_rejections, int(config.power_trials))
                        if c2st is not None
                        else None
                    ),
                }
            )
    return {
        "status": "ok",
        "results": rows,
        "minimum_detectable_effect": {
            "mmd": _minimum_detectable_effect(rows, "mmd_power", float(config.power_target)),
            "c2st": (
                _minimum_detectable_effect(rows, "c2st_power", float(config.power_target))
                if c2st is not None
                else None
            ),
        },
        "calibration_documents": int(len(unique_calibration_documents)),
        "source_pc1_standard_deviation": float(pc1_scale),
        "power_permutations": int(config.power_permutations),
        "parallel_jobs": int(joblib.effective_n_jobs(-1)),
        "power_reference_max_samples": int(
            min(int(config.reference_max_samples), int(config.power_reference_max_samples))
        ),
    }


def _power_trial_rejections(
    *,
    mmd: MMDPermutationTest,
    c2st: BlockAwareC2ST | None,
    shifted: np.ndarray,
    target_blocks: np.ndarray,
    trial_seed: int,
    hard_alpha: float,
) -> tuple[bool, bool]:
    mmd_result = mmd.test(shifted, seed=trial_seed, block_ids=target_blocks)
    mmd_rejected = bool(float(mmd_result["p_value"]) <= float(hard_alpha))
    if c2st is None:
        return mmd_rejected, False
    c2st_result = c2st.test(
        shifted,
        seed=trial_seed + 1,
        block_ids=target_blocks,
    )
    return mmd_rejected, bool(float(c2st_result["p_value"]) <= float(hard_alpha))


def _minimum_detectable_effect(
    rows: list[dict[str, Any]],
    key: str,
    target_power: float,
) -> dict[str, float] | None:
    eligible = [
        row
        for row in rows
        if float(row["effect_size"]) > 0.0
        and row.get(key) is not None
        and float(row[key]) >= float(target_power)
    ]
    if not eligible:
        return None
    selected = min(eligible, key=lambda row: (float(row["effect_size"]), int(row["window_documents"])))
    return {
        "effect_size": float(selected["effect_size"]),
        "window_documents": int(selected["window_documents"]),
        "estimated_power": float(selected[key]),
    }


def wilson_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> list[float]:
    if int(trials) < 1 or not 0 <= int(successes) <= int(trials):
        raise ValueError("Wilson interval requires 0 <= successes <= trials and trials >= 1")
    n = float(trials)
    proportion = float(successes) / n
    denominator = 1.0 + float(z) ** 2 / n
    center = (proportion + float(z) ** 2 / (2.0 * n)) / denominator
    half_width = (
        float(z)
        * math.sqrt(proportion * (1.0 - proportion) / n + float(z) ** 2 / (4.0 * n**2))
        / denominator
    )
    return [float(max(0.0, center - half_width)), float(min(1.0, center + half_width))]


def _window_status(p_value: float, *, soft_alpha: float, hard_alpha: float) -> str:
    if p_value <= float(hard_alpha):
        return "hard_drift"
    if p_value <= float(soft_alpha):
        return "soft_drift"
    return "in_distribution"


def _calibrate_p_value_thresholds(
    rows: list[dict[str, Any]],
    config: WindowDriftConfig,
    *,
    sequential_fwer_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit nominal p-value cutoffs on independent ID windows.

    A small calibration pool cannot estimate a 1% null quantile. The formal
    thresholds therefore remain the pre-registered nominal levels from a valid
    test; calibration estimates their realized H0 false-alert rates only.
    """

    result: dict[str, Any] = {}
    for space in ("A", "B"):
        p_values = np.asarray(
            [
                float(row[space]["p_value"])
                for row in rows
                if isinstance(row.get(space), dict)
                and row[space].get("p_value") is not None
                and np.isfinite(row[space]["p_value"])
            ],
            dtype=np.float64,
        )
        if p_values.size < int(config.minimum_valid_calibration_windows):
            fallback_allowed = bool(config.allow_nominal_fallback_for_smoke)
            result[space] = {
                "status": (
                    "nominal_fallback_smoke_only"
                    if fallback_allowed
                    else "invalid_insufficient_calibration_windows"
                ),
                "window_count": int(p_values.size),
                "soft_alpha": float(config.soft_alpha),
                "hard_alpha": float(config.hard_alpha),
                "soft_false_alert_rate": None,
                "hard_false_alert_rate": None,
                "soft_false_alert_rate_ci95": None,
                "hard_false_alert_rate_ci95": None,
                "minimum_valid_calibration_windows": int(
                    config.minimum_valid_calibration_windows
                ),
                "fallback_allowed_for_smoke": fallback_allowed,
                "rule": "pre_registered_nominal_p_value_thresholds_with_h0_audit",
                "calibration_p_values": p_values.astype(float).tolist(),
            }
            continue
        soft_rejections = int(np.sum(p_values <= float(config.soft_alpha)))
        hard_rejections = int(np.sum(p_values <= float(config.hard_alpha)))
        result[space] = {
            "status": "nominal_thresholds_h0_audited_on_independent_training_calibration",
            "window_count": int(p_values.size),
            "soft_alpha": float(config.soft_alpha),
            "hard_alpha": float(config.hard_alpha),
            "soft_false_alert_count": soft_rejections,
            "hard_false_alert_count": hard_rejections,
            "soft_false_alert_rate": float(soft_rejections / p_values.size),
            "hard_false_alert_rate": float(hard_rejections / p_values.size),
            "soft_false_alert_rate_ci95": wilson_interval(
                soft_rejections, int(p_values.size)
            ),
            "hard_false_alert_rate_ci95": wilson_interval(
                hard_rejections, int(p_values.size)
            ),
            "rule": "pre_registered_nominal_p_value_thresholds_with_h0_audit",
            "calibration_p_values": p_values.astype(float).tolist(),
        }
        if space == "B" and bool(config.require_sequential_fwer_calibration):
            result[space]["sequential_fwer_audit"] = (
                dict(sequential_fwer_audit)
                if sequential_fwer_audit is not None
                else _sequential_fwer_audit(p_values, config)
            )
    return result


def _cached_sequential_monte_carlo_audit(
    *,
    b_test: MMDPermutationTest,
    b_values: np.ndarray,
    document_ids: np.ndarray,
    document_to_records: dict[str, np.ndarray],
    calibration_documents: np.ndarray,
    block_ids: np.ndarray,
    config: WindowDriftConfig,
) -> dict[str, Any]:
    """Estimate complete-episode H0 FWER on the fixed independent ID pool."""

    signature = _sequential_monte_carlo_signature(
        b_test=b_test,
        b_values=b_values,
        document_ids=document_ids,
        document_to_records=document_to_records,
        calibration_documents=calibration_documents,
        block_ids=block_ids,
        config=config,
    )
    cache_path = (
        Path(config.sequential_calibration_cache_path)
        if config.sequential_calibration_cache_path is not None
        else None
    )
    if cache_path is None:
        return _compute_sequential_monte_carlo_audit(
            b_test=b_test,
            b_values=b_values,
            document_ids=document_ids,
            document_to_records=document_to_records,
            calibration_documents=calibration_documents,
            block_ids=block_ids,
            config=config,
            signature=signature,
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_file_lock(cache_path):
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("artifact_type")
            == "llm_judge_ood_sequential_h0_monte_carlo"
            and payload.get("signature") == signature
            and isinstance(payload.get("audit"), dict)
        ):
            return {
                **payload["audit"],
                "cache": {
                    "status": "hit",
                    "path": str(cache_path),
                    "signature": signature,
                },
            }
        audit = _compute_sequential_monte_carlo_audit(
            b_test=b_test,
            b_values=b_values,
            document_ids=document_ids,
            document_to_records=document_to_records,
            calibration_documents=calibration_documents,
            block_ids=block_ids,
            config=config,
            signature=signature,
        )
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "artifact_type": "llm_judge_ood_sequential_h0_monte_carlo",
                    "signature": signature,
                    "audit": audit,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, cache_path)
        return {
            **audit,
            "cache": {
                "status": "miss_created",
                "path": str(cache_path),
                "signature": signature,
            },
        }


def _sequential_monte_carlo_signature(
    *,
    b_test: MMDPermutationTest,
    b_values: np.ndarray,
    document_ids: np.ndarray,
    document_to_records: dict[str, np.ndarray],
    calibration_documents: np.ndarray,
    block_ids: np.ndarray,
    config: WindowDriftConfig,
) -> str:
    if b_test.reference_ is None or b_test.reference_block_ids_ is None:
        raise RuntimeError("Sequential H0 calibration requires a fitted B-space MMD test")
    ids = np.asarray(document_ids).astype(str)
    calibration = np.asarray(calibration_documents, dtype=int)
    calibration_records = _records_for_documents(calibration, ids, document_to_records)
    blocks = np.asarray(block_ids).astype(str)
    digest = hashlib.sha256()
    digest.update(b"sequential_h0_monte_carlo_v1_conservative_mmd\0")
    digest.update(
        json.dumps(
            {
                "window_size": int(config.window_size),
                "minimum_window_documents": int(config.minimum_window_documents),
                "mmd_permutations": int(config.mmd_permutations),
                "kernel_bandwidth": config.kernel_bandwidth,
                "alpha_fwer": float(config.alpha_fwer),
                "alpha_spending": str(config.alpha_spending),
                "pocock_horizon": int(config.pocock_horizon),
                "minimum_consecutive_windows": int(config.minimum_consecutive_windows),
                "trials": int(config.sequential_calibration_trials),
                "seed": int(config.sequential_calibration_seed),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _update_array_digest(digest, name="b_reference", values=b_test.reference_)
    _update_array_digest(
        digest, name="b_calibration", values=np.asarray(b_values)[calibration_records]
    )
    _update_string_array_digest(
        digest, name="b_reference_blocks", values=b_test.reference_block_ids_
    )
    for name, values in (
        ("calibration_document_ids", ids[calibration]),
        ("calibration_record_ids", ids[calibration_records]),
        ("calibration_record_blocks", blocks[calibration_records]),
    ):
        _update_string_array_digest(digest, name=name, values=values)
    return digest.hexdigest()


def _compute_sequential_monte_carlo_audit(
    *,
    b_test: MMDPermutationTest,
    b_values: np.ndarray,
    document_ids: np.ndarray,
    document_to_records: dict[str, np.ndarray],
    calibration_documents: np.ndarray,
    block_ids: np.ndarray,
    config: WindowDriftConfig,
    signature: str,
) -> dict[str, Any]:
    horizon = int(config.pocock_horizon)
    window_size = int(config.window_size)
    trials = int(config.sequential_calibration_trials)
    required_documents = horizon * window_size
    ids = np.asarray(document_ids).astype(str)
    blocks = np.asarray(block_ids).astype(str)
    pool = np.asarray(calibration_documents, dtype=int)
    if str(config.alpha_spending) != "pocock":
        return {
            "valid": False,
            "reason": "finite_pocock_horizon_required",
            "signature": signature,
            "episode_count": 0,
        }
    if pool.size < required_documents:
        return {
            "valid": False,
            "reason": "insufficient_unique_calibration_documents_for_episode",
            "signature": signature,
            "episode_count": 0,
            "calibration_document_count": int(pool.size),
            "required_documents_per_episode": int(required_documents),
        }
    rng = np.random.default_rng(int(config.sequential_calibration_seed))
    window_p_values: list[float] = []
    episode_rows: list[dict[str, Any]] = []
    false_alerts = 0
    run_lengths: list[int] = []
    started = time.perf_counter()
    for trial in range(trials):
        sampled = rng.choice(pool, size=required_documents, replace=False)
        tracker = AlphaSpendingTracker(config)
        episode_p_values: list[float] = []
        persistent = False
        first_persistent_window: int | None = None
        for window_index in range(horizon):
            documents = sampled[
                window_index * window_size : (window_index + 1) * window_size
            ]
            records = _records_for_documents(documents, ids, document_to_records)
            result = b_test.test(
                np.asarray(b_values)[records],
                seed=_seed(
                    int(config.sequential_calibration_seed),
                    trial,
                    window_index + 10_000,
                ),
                block_ids=blocks[records],
            )
            p_value = float(result["conservative_p_value"])
            episode_p_values.append(p_value)
            window_p_values.append(p_value)
            decision = tracker.update(window_index=window_index, p_value=p_value)
            if bool(decision["persistent_b_drift"]) and not persistent:
                persistent = True
                first_persistent_window = int(window_index + 1)
        false_alerts += int(persistent)
        run_lengths.append(
            int(first_persistent_window) if first_persistent_window is not None else horizon + 1
        )
        episode_rows.append(
            {
                "trial": int(trial),
                "persistent_false_alert": bool(persistent),
                "first_persistent_window": first_persistent_window,
                "p_values": episode_p_values,
            }
        )
    p_values = np.asarray(window_p_values, dtype=np.float64)
    soft_rejections = int(np.sum(p_values <= float(config.soft_alpha)))
    hard_rejections = int(np.sum(p_values <= float(config.hard_alpha)))
    episode_interval = wilson_interval(false_alerts, trials)
    valid = bool(float(episode_interval[1]) <= float(config.alpha_fwer) + 1e-12)
    histogram, edges = np.histogram(p_values, bins=np.linspace(0.0, 1.0, 11))
    return {
        "valid": valid,
        "reason": None if valid else "episode_fwer_wilson_upper_exceeds_alpha_fwer",
        "signature": signature,
        "scope": "conditional_monte_carlo_over_fixed_independent_training_calibration_pool",
        "sampling": "without_replacement_within_episode_reuse_allowed_across_trials",
        "p_value_method": "conservative_permutation_rank",
        "trials": trials,
        "episode_count": trials,
        "horizon": horizon,
        "window_documents": window_size,
        "required_documents_per_episode": required_documents,
        "calibration_document_count": int(pool.size),
        "minimum_consecutive_windows": int(config.minimum_consecutive_windows),
        "false_alert_count": int(false_alerts),
        "episode_false_alert_rate": float(false_alerts / trials),
        "episode_false_alert_rate_ci95": episode_interval,
        "alpha_fwer": float(config.alpha_fwer),
        "acceptance_rule": "episode_fwer_wilson_95_upper_bound_lte_alpha_fwer",
        "window_count": int(p_values.size),
        "window_false_positive_rate_alpha_0_05": float(soft_rejections / p_values.size),
        "window_false_positive_rate_alpha_0_05_ci95": wilson_interval(
            soft_rejections, int(p_values.size)
        ),
        "window_false_positive_rate_alpha_0_01": float(hard_rejections / p_values.size),
        "window_false_positive_rate_alpha_0_01_ci95": wilson_interval(
            hard_rejections, int(p_values.size)
        ),
        "average_run_length_censored_at_horizon_plus_one": float(np.mean(run_lengths)),
        "p_value_quantiles": {
            str(quantile): float(np.quantile(p_values, quantile))
            for quantile in (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
        },
        "p_value_histogram": {
            "bin_edges": edges.astype(float).tolist(),
            "counts": histogram.astype(int).tolist(),
        },
        "window_p_values": p_values.astype(float).tolist(),
        "episodes": episode_rows,
        "permutations": int(config.mmd_permutations),
        "conservative_p_value_resolution": 1.0
        / float(int(config.mmd_permutations) + 1),
        "seed": int(config.sequential_calibration_seed),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def _sequential_fwer_audit(
    p_values: np.ndarray,
    config: WindowDriftConfig,
) -> dict[str, Any]:
    """Replay complete finite monitoring episodes on independent H0 windows."""

    values = np.asarray(p_values, dtype=np.float64)
    if str(config.alpha_spending) != "pocock":
        return {
            "valid": False,
            "reason": "finite_horizon_required",
            "episode_count": 0,
        }
    horizon = int(config.pocock_horizon)
    complete_episode_count = int(values.size // horizon)
    if complete_episode_count < 1:
        return {
            "valid": False,
            "reason": "no_complete_independent_calibration_episode",
            "episode_count": 0,
            "horizon": horizon,
            "available_window_count": int(values.size),
        }
    alerts = 0
    episode_rows: list[dict[str, Any]] = []
    for episode_index in range(complete_episode_count):
        tracker = AlphaSpendingTracker(config)
        episode_values = values[
            episode_index * horizon : (episode_index + 1) * horizon
        ]
        persistent = False
        for window_index, p_value in enumerate(episode_values.tolist()):
            decision = tracker.update(window_index=window_index, p_value=float(p_value))
            persistent = persistent or bool(decision["persistent_b_drift"])
        alerts += int(persistent)
        episode_rows.append(
            {
                "episode_index": int(episode_index),
                "persistent_false_alert": bool(persistent),
            }
        )
    interval = wilson_interval(alerts, complete_episode_count)
    valid = bool(float(interval[1]) <= float(config.alpha_fwer) + 1e-12)
    return {
        "valid": valid,
        "reason": None if valid else "episode_fwer_wilson_upper_exceeds_alpha_fwer",
        "horizon": horizon,
        "minimum_consecutive_windows": int(config.minimum_consecutive_windows),
        "episode_count": int(complete_episode_count),
        "unused_incomplete_window_count": int(values.size % horizon),
        "false_alert_count": int(alerts),
        "episode_false_alert_rate": float(alerts / complete_episode_count),
        "episode_false_alert_rate_ci95": interval,
        "alpha_fwer": float(config.alpha_fwer),
        "acceptance_rule": "wilson_95_upper_bound_lte_alpha_fwer",
        "episodes": episode_rows,
    }


def _quadrant(a_status: str, b_status: str) -> str:
    a_drift = a_status != "in_distribution"
    b_drift = b_status != "in_distribution"
    if a_drift and b_drift:
        return "A_and_B_drift"
    if b_drift:
        return "B_only_drift"
    if a_drift:
        return "A_only_drift"
    return "neither_space_drift"


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} contain non-finite values")
    return matrix


def _normalized_block_ids(values: np.ndarray, block_ids: np.ndarray | None, *, name: str) -> np.ndarray:
    matrix = _matrix(values, "MMD block features")
    if block_ids is None:
        return np.asarray([f"row::{index}" for index in range(len(matrix))], dtype=str)
    blocks = np.asarray(block_ids).astype(str)
    if blocks.shape != (len(matrix),):
        raise ValueError(f"{name} must align with MMD features")
    return blocks


def _score_vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size < 2 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector with at least two values")
    return vector


def _sample_rows(
    values: np.ndarray,
    block_ids: np.ndarray,
    *,
    maximum_rows: int,
    subsample_threshold: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use all rows up to the hard cap, then select complete source blocks."""

    matrix = _matrix(values, "MMD block features")
    blocks = np.asarray(block_ids).astype(str)
    if blocks.shape != (len(matrix),):
        raise ValueError("MMD block IDs must align with features")
    if int(maximum_rows) < 2 or int(subsample_threshold) < int(maximum_rows):
        raise ValueError("invalid reference subsampling limits")
    if len(matrix) <= int(maximum_rows):
        return matrix, blocks
    rng = np.random.default_rng(int(seed))
    order = np.argsort(blocks, kind="stable")
    ordered_blocks = blocks[order]
    starts = np.concatenate(
        [np.asarray([0], dtype=np.intp), np.flatnonzero(ordered_blocks[1:] != ordered_blocks[:-1]) + 1]
    )
    block_values = ordered_blocks[starts]
    stops = np.concatenate([starts[1:], np.asarray([len(order)], dtype=np.intp)])
    by_block = {
        str(block): order[start:stop]
        for block, start, stop in zip(block_values.tolist(), starts.tolist(), stops.tolist(), strict=True)
    }
    oversized = [block for block, rows in by_block.items() if rows.size > int(maximum_rows)]
    if oversized:
        raise ValueError(
            "Reference block exceeds reference_max_samples; split or reduce the block before fitting: "
            f"{oversized[:3]}"
        )
    block_order = rng.permutation(np.asarray(sorted(by_block), dtype=str)).tolist()
    selected_blocks: list[str] = []
    selected_rows = 0
    for block in block_order:
        block_rows = int(by_block[block].size)
        if selected_blocks and selected_rows + block_rows > int(maximum_rows):
            continue
        selected_blocks.append(str(block))
        selected_rows += block_rows
    if len(selected_blocks) < 2:
        smallest = sorted(by_block, key=lambda block: (int(by_block[block].size), str(block)))
        selected_blocks = [str(block) for block in smallest[:2]]
        selected_row_count = sum(int(by_block[block].size) for block in selected_blocks)
        if len(selected_blocks) < 2 or selected_row_count > int(maximum_rows):
            raise ValueError(
                "reference_max_samples is too small to retain two complete reference blocks"
            )
    indices = np.sort(
        np.concatenate([by_block[block] for block in selected_blocks]).astype(int)
    )
    return matrix[indices], blocks[indices]


def _balanced_domain_block_split(
    source_block_ids: np.ndarray,
    target_block_ids: np.ndarray,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source_blocks = np.unique(np.asarray(source_block_ids).astype(str))
    target_blocks = np.unique(np.asarray(target_block_ids).astype(str))
    if len(source_blocks) < 2 or len(target_blocks) < 2:
        raise ValueError("Block-aware train/test splitting requires at least two blocks per domain")
    rng = np.random.default_rng(int(seed))
    shuffled_source = rng.permutation(source_blocks)
    shuffled_target = rng.permutation(target_blocks)
    minimum_domain_blocks = min(len(source_blocks), len(target_blocks))
    test_count = min(
        minimum_domain_blocks - 1,
        max(1, int(round(minimum_domain_blocks * float(test_fraction)))),
    )
    return (
        shuffled_source[test_count:],
        shuffled_source[:test_count],
        shuffled_target[test_count:],
        shuffled_target[:test_count],
    )


def _binomial_upper_tail(successes: int, trials: int, *, probability: float) -> float:
    if int(trials) < 1 or not 0 <= int(successes) <= int(trials):
        raise ValueError("Binomial successes and trials are invalid")
    p = float(probability)
    if not 0.0 <= p <= 1.0:
        raise ValueError("Binomial probability must be in [0, 1]")
    return float(
        sum(
            math.comb(int(trials), value)
            * (p**value)
            * ((1.0 - p) ** (int(trials) - value))
            for value in range(int(successes), int(trials) + 1)
        )
    )


def _bootstrap_documents(
    values: np.ndarray,
    document_ids: np.ndarray,
    *,
    document_count: int,
    rng: np.random.Generator,
    document_rows: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _matrix(values, "power calibration features")
    documents = np.asarray(document_ids).astype(str)
    unique_documents = np.unique(documents)
    if int(document_count) < 2 or len(unique_documents) < 2:
        raise ValueError("Power analysis requires at least two target documents")
    sampled = rng.choice(unique_documents, size=int(document_count), replace=True)
    row_index = document_rows or {
        document_id: np.flatnonzero(documents == document_id)
        for document_id in unique_documents.tolist()
    }
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for replicate, document_id in enumerate(sampled.tolist()):
        local = matrix[row_index[str(document_id)]]
        rows.append(local)
        labels.extend([f"bootstrap_document::{replicate}::{document_id}"] * len(local))
    return np.vstack(rows), np.asarray(labels, dtype=str)


def _indices(indices: np.ndarray, total: int, name: str) -> np.ndarray:
    result = np.asarray(indices, dtype=int)
    if result.ndim != 1 or result.size == 0 or np.any(result < 0) or np.any(result >= int(total)):
        raise ValueError(f"{name} must be non-empty and in bounds")
    return result


def _unique_indices(indices: np.ndarray, document_ids: np.ndarray, name: str) -> np.ndarray:
    result = _indices(indices, len(document_ids), name)
    selected: list[int] = []
    seen: set[str] = set()
    for index in result.tolist():
        document_id = str(document_ids[int(index)])
        if document_id not in seen:
            selected.append(int(index))
            seen.add(document_id)
    return np.asarray(selected, dtype=int)


def _document_to_indices(document_ids: np.ndarray) -> dict[str, np.ndarray]:
    result: dict[str, list[int]] = {}
    for index, document_id in enumerate(np.asarray(document_ids).astype(str).tolist()):
        result.setdefault(document_id, []).append(int(index))
    return {document_id: np.asarray(indices, dtype=int) for document_id, indices in result.items()}


def _records_for_documents(
    document_indices: np.ndarray,
    document_ids: np.ndarray,
    document_to_records: dict[str, np.ndarray],
) -> np.ndarray:
    arrays = [document_to_records[str(document_ids[int(index)])] for index in np.asarray(document_indices, dtype=int).tolist()]
    if not arrays:
        return np.zeros(0, dtype=int)
    return np.concatenate(arrays).astype(int)


def _deterministic_subsample(values: np.ndarray, *, maximum: int, seed: int) -> np.ndarray:
    if len(values) <= int(maximum):
        return values.copy()
    rng = np.random.default_rng(int(seed))
    chosen = np.sort(rng.choice(len(values), size=int(maximum), replace=False))
    return values[chosen]


def _median_bandwidth(values: np.ndarray) -> float:
    sample = _matrix(_deterministic_subsample(values, maximum=256, seed=0), "bandwidth features")
    squared_norms = np.einsum("ij,ij->i", sample, sample)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (sample @ sample.T)
    np.maximum(squared, 0.0, out=squared)
    upper = np.sqrt(squared[np.triu_indices_from(squared, k=1)])
    positive = upper[upper > 1e-12]
    return float(np.median(positive)) if positive.size else 1.0


def _rbf_kernel(values: np.ndarray, *, bandwidth: float) -> np.ndarray:
    matrix = _matrix(values, "RBF-kernel features")
    # ``values[:, None, :] - values[None, :, :]`` creates an N×N×D
    # temporary.  The Gram identity gives the same squared Euclidean
    # distances with only N×N storage, which is essential when an offline
    # source reference contains thousands of 128-dimensional embeddings.
    squared_norms = np.einsum("ij,ij->i", matrix, matrix)
    squared = squared_norms[:, None] + squared_norms[None, :] - 2.0 * (matrix @ matrix.T)
    np.maximum(squared, 0.0, out=squared)
    return np.exp(-squared / (2.0 * float(bandwidth) ** 2))


def _aggregate_kernel_by_block(
    kernel: np.ndarray,
    block_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact block-level kernel sums and their row counts.

    ``np.add.reduceat`` groups rows and columns without constructing an
    N×N×B indicator tensor.  The result is sufficient for every block
    assignment used by the MMD V-statistic.
    """

    values = np.asarray(kernel, dtype=np.float64)
    indices = np.asarray(block_indices, dtype=np.intp)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("MMD kernel must be square")
    if indices.shape != (values.shape[0],) or indices.size == 0:
        raise ValueError("MMD block indices must align with the kernel")
    order = np.argsort(indices, kind="stable")
    ordered_blocks = indices[order]
    starts = np.concatenate(
        [np.asarray([0], dtype=np.intp), np.flatnonzero(np.diff(ordered_blocks)) + 1]
    )
    row_grouped = np.add.reduceat(values[order], starts, axis=0)
    block_kernel = np.add.reduceat(row_grouped[:, order], starts, axis=1)
    block_sizes = np.diff(
        np.concatenate([starts, np.asarray([len(indices)], dtype=np.intp)])
    ).astype(np.float64)
    return block_kernel, block_sizes


def _mmd_from_kernel(kernel: np.ndarray, source_indices: np.ndarray, target_indices: np.ndarray) -> float:
    x = np.asarray(source_indices, dtype=int)
    y = np.asarray(target_indices, dtype=int)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("MMD needs at least two rows per sample")
    k_xx = kernel[np.ix_(x, x)]
    k_yy = kernel[np.ix_(y, y)]
    k_xy = kernel[np.ix_(x, y)]
    term_xx = k_xx.mean()
    term_yy = k_yy.mean()
    term_xy = 2.0 * k_xy.mean()
    return float(term_xx + term_yy - term_xy)


def _seed(base: int, first: int, second: int) -> int:
    return int((int(base) * 1_000_003 + int(first) * 9_176 + int(second)) % (2**32 - 1))


def _window_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in {"document_indices"}
    }


def _calibration_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def _calibration_summary(
    rows: list[dict[str, Any]],
    config: WindowDriftConfig,
) -> dict[str, Any]:
    """Report independent-calibration false-alert estimates with bootstrap CIs."""

    summary: dict[str, Any] = {
        "scope": "independent_training_calibration",
        "window_count": int(len(rows)),
        "completed_window_count": int(
            sum(1 for row in rows if row.get("status") not in {"insufficient_target_blocks", "insufficient_window_documents"})
        ),
        "skipped_window_count": int(
            sum(1 for row in rows if row.get("status") in {"insufficient_target_blocks", "insufficient_window_documents"})
        ),
        "document_count": int(sum(int(row.get("document_count", 0)) for row in rows)),
        "judge_record_count": int(sum(int(row.get("judge_record_count", 0)) for row in rows)),
        "primary_test": str(config.primary_test).lower(),
        "permutations_per_window": int(config.mmd_permutations),
        "tests": ["mmd"] + (["c2st"] if config.c2st_enabled else []),
        "conservative_p_value_resolution": 1.0 / float(int(config.mmd_permutations) + 1),
        "configured_thresholds": {
            "soft_alpha": float(config.soft_alpha),
            "hard_alpha": float(config.hard_alpha),
            "alpha_fwer": float(config.alpha_fwer),
            "alpha_spending": str(config.alpha_spending),
        },
        "false_alert_interval_method": "wilson_score_95",
        "by_test": {},
    }
    for space in ("A", "B"):
        p_values = np.asarray(
            [
                float(row[space]["p_value"])
                for row in rows
                if space in row and row[space].get("p_value") is not None
            ],
            dtype=np.float64,
        )
        summary[space] = _calibration_p_value_summary(
            p_values,
            soft_alpha=float(config.soft_alpha),
            hard_alpha=float(config.hard_alpha),
        )
    for test_name in summary["tests"]:
        summary["by_test"][test_name] = {}
        for space in ("A", "B"):
            p_values = np.asarray(
                [
                    float(row[space][test_name]["p_value"])
                    for row in rows
                    if space in row and row[space].get(test_name, {}).get("p_value") is not None
                ],
                dtype=np.float64,
            )
            summary["by_test"][test_name][space] = _calibration_p_value_summary(
                p_values,
                soft_alpha=float(config.soft_alpha),
                hard_alpha=float(config.hard_alpha),
            )
    return summary


def _calibration_p_value_summary(
    p_values: np.ndarray,
    *,
    soft_alpha: float,
    hard_alpha: float,
) -> dict[str, Any]:
    values = np.asarray(p_values, dtype=np.float64)
    if values.size == 0:
        return {
            "p_value_window_count": 0,
            "soft_false_alert_rate": None,
            "hard_false_alert_rate": None,
            "soft_false_alert_rate_ci95": None,
            "hard_false_alert_rate_ci95": None,
        }
    soft_rejections = int(np.sum(values <= float(soft_alpha)))
    hard_rejections = int(np.sum(values <= float(hard_alpha)))
    return {
        "p_value_window_count": int(values.size),
        "soft_false_alert_rate": float(soft_rejections / values.size),
        "hard_false_alert_rate": float(hard_rejections / values.size),
        "soft_false_alert_rate_ci95": wilson_interval(soft_rejections, int(values.size)),
        "hard_false_alert_rate_ci95": wilson_interval(hard_rejections, int(values.size)),
    }
