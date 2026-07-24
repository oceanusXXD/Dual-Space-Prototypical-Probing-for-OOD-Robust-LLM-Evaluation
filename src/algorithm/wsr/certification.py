"""Finite-population WSR certification for detector-gated selective prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np


_EPS = 1e-12


def _bounded_sample(values: Iterable[float], *, name: str = "WSR sample") -> np.ndarray:
    sample = (
        values.astype(np.float64, copy=False)
        if isinstance(values, np.ndarray)
        else np.asarray(list(values), dtype=np.float64)
    )
    if sample.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if sample.size and not np.all(np.isfinite(sample)):
        raise ValueError(f"{name} must be finite")
    if sample.size and (np.min(sample) < -1e-10 or np.max(sample) > 1.0 + 1e-10):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(sample, 0.0, 1.0)


def _unit_interval(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return min(1.0, max(0.0, resolved))


def wsr_betting_log_capital(
    values: Iterable[float],
    *,
    population_mean_null: float,
    population_size: int,
    betting_clip: float = 0.5,
) -> float:
    """Compute one-sided without-replacement betting log capital.

    The sample values must be bounded in [0, 1]. Larger capital rejects the null
    that the finite-population mean is at least ``population_mean_null``.
    """

    sample = _bounded_sample(values)
    sample_size = int(sample.size)
    population_n = int(population_size)
    if population_n <= 0:
        raise ValueError("WSR population size must be positive")
    if sample_size > population_n:
        raise ValueError("WSR sample cannot exceed the population")
    if sample_size == 0:
        return 0.0

    u = _unit_interval(population_mean_null, name="WSR null population mean")
    times = np.arange(1, sample_size + 1, dtype=np.float64)
    remaining = float(population_n) - times + 1.0
    prefix = np.concatenate(([0.0], np.cumsum(sample, dtype=np.float64)[:-1]))
    q = np.clip((float(population_n) * u - prefix) / remaining, 0.0, 1.0)

    residual_squared = (sample - q) ** 2
    residual_prefix = np.concatenate(([0.0], np.cumsum(residual_squared, dtype=np.float64)[:-1]))
    variance = np.maximum((0.25 + residual_prefix) / times, _EPS)
    estimated_mean = (0.5 + prefix) / times
    raw_bets = (q - estimated_mean) / variance

    clip_value = min(1.0, max(0.0, float(betting_clip)))
    upper = np.full_like(q, np.inf, dtype=np.float64)
    bounded = q < 1.0 - _EPS
    upper[bounded] = clip_value / np.maximum(1.0 - q[bounded], _EPS)
    bets = np.clip(np.where(np.isfinite(raw_bets), raw_bets, 0.0), 0.0, upper)
    factors = 1.0 + bets * (q - sample)
    if np.any(factors <= 0.0):
        return -math.inf
    return float(np.sum(np.log(np.maximum(factors, _EPS)), dtype=np.float64))


def wsr_population_mean_upper_bound(
    values: Iterable[float],
    *,
    population_size: int,
    alpha: float,
    tolerance: float = 1e-6,
    max_iterations: int = 80,
) -> float:
    """Invert WSR tests to upper-bound a [0, 1] finite-population mean."""

    sample = _bounded_sample(values)
    population_n = int(population_size)
    if population_n <= 0:
        raise ValueError("WSR population size must be positive")
    if sample.size > population_n:
        raise ValueError("WSR sample cannot exceed the population")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 < alpha_value <= 1.0:
        raise ValueError("WSR alpha must be within (0, 1]")
    if sample.size == 0:
        return 1.0

    log_boundary = math.log(1.0 / alpha_value)

    def rejected(null_mean: float) -> bool:
        log_capital = wsr_betting_log_capital(
            sample,
            population_mean_null=float(null_mean),
            population_size=population_n,
        )
        return bool(log_capital >= log_boundary)

    if not rejected(1.0):
        return 1.0
    if rejected(0.0):
        return 0.0

    lo = 0.0
    hi = 1.0
    tol = max(float(tolerance), _EPS)
    for _ in range(max(1, int(max_iterations))):
        mid = 0.5 * (lo + hi)
        if rejected(mid):
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol:
            break
    return float(min(1.0, max(0.0, hi)))


def normalized_absolute_error(
    predictions: Sequence[float],
    labels: Sequence[float],
    *,
    y_min: float,
    y_max: float,
    clip_predictions: bool = True,
) -> np.ndarray:
    """Return clipped normalized absolute errors in [0, 1]."""

    pred = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError("predictions and labels must align")
    low = float(y_min)
    high = float(y_max)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("score range must satisfy y_max > y_min")
    if clip_predictions:
        pred = np.clip(pred, low, high)
    losses = np.abs(pred - truth) / (high - low)
    return _bounded_sample(losses, name="normalized absolute errors")


def quantile_threshold_grid(
    scores: Sequence[float],
    *,
    max_candidates: int = 32,
    quantiles: Sequence[float] | None = None,
) -> np.ndarray:
    """Build a finite threshold grid before calibration labels are used."""

    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("detector scores must be one-dimensional")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.asarray([], dtype=np.float64)
    unique = np.unique(values)
    if quantiles is None and unique.size <= max(1, int(max_candidates)):
        return unique.astype(np.float64)
    qs = (
        np.asarray(list(quantiles), dtype=np.float64)
        if quantiles is not None
        else np.linspace(0.0, 1.0, max(2, int(max_candidates)), dtype=np.float64)
    )
    qs = qs[np.isfinite(qs)]
    if qs.size == 0:
        return np.asarray([], dtype=np.float64)
    qs = np.clip(qs, 0.0, 1.0)
    return np.unique(np.quantile(values, qs)).astype(np.float64)


@dataclass(frozen=True)
class WSRThresholdCandidate:
    threshold: float
    population_rows: int
    population_accepted_rows: int
    coverage: float
    calibration_rows: int
    calibration_accepted_rows: int
    calibration_mean_accept_loss: float
    calibration_selective_risk: float | None
    target_population_loss_mean: float
    log_capital_at_target: float
    wsr_population_loss_ucb: float
    wsr_selective_risk_ucb: float
    delta_per_candidate: float
    risk_bound: float
    certified: bool


def certify_wsr_thresholds(
    *,
    scores: Sequence[float],
    losses: Sequence[float],
    calibration_indices: Sequence[int],
    risk_bound: float,
    delta: float,
    thresholds: Sequence[float] | None = None,
    max_candidates: int = 32,
    bonferroni_count: int | None = None,
) -> dict[str, object]:
    """Certify detector thresholds for selective normalized MAE.

    ``scores`` and ``losses`` describe the current finite population. Only
    ``losses[calibration_indices]`` are used for WSR certification.
    """

    score_values = np.asarray(scores, dtype=np.float64)
    loss_values = _bounded_sample(losses, name="normalized losses")
    if score_values.ndim != 1:
        raise ValueError("detector scores must be one-dimensional")
    if score_values.shape != loss_values.shape:
        raise ValueError("scores and losses must align")
    if not np.all(np.isfinite(score_values)):
        raise ValueError("detector scores must be finite")
    population_n = int(score_values.size)
    if population_n <= 0:
        raise ValueError("WSR population must not be empty")

    risk_target = _unit_interval(risk_bound, name="selective risk bound")
    delta_value = float(delta)
    if not math.isfinite(delta_value) or not 0.0 < delta_value <= 1.0:
        raise ValueError("delta must be within (0, 1]")

    calibration = np.asarray(list(calibration_indices), dtype=int)
    if calibration.ndim != 1:
        raise ValueError("calibration_indices must be one-dimensional")
    if calibration.size == 0:
        raise ValueError("WSR calibration sample must not be empty")
    if np.any(calibration < 0) or np.any(calibration >= population_n):
        raise ValueError("calibration_indices must refer to population rows")
    if np.unique(calibration).size != calibration.size:
        raise ValueError("calibration_indices must be sampled without replacement")

    grid = (
        np.asarray(list(thresholds), dtype=np.float64)
        if thresholds is not None
        else quantile_threshold_grid(score_values, max_candidates=max_candidates)
    )
    grid = np.unique(grid[np.isfinite(grid)])
    candidates_with_coverage = []
    for threshold in grid.tolist():
        accepted = score_values <= float(threshold)
        if np.any(accepted):
            candidates_with_coverage.append(float(threshold))
    grid = np.asarray(candidates_with_coverage, dtype=np.float64)
    if grid.size == 0:
        return {
            "selection_status": "no_nonzero_coverage_thresholds",
            "threshold": None,
            "candidate_count": 0,
            "delta_per_candidate": None,
            "risk_bound": risk_target,
            "certified_candidate_count": 0,
            "candidates": [],
        }

    correction_count = int(bonferroni_count) if bonferroni_count is not None else int(grid.size)
    correction_count = max(1, correction_count)
    beta = delta_value / float(correction_count)
    log_boundary = math.log(1.0 / beta)

    candidate_rows: list[WSRThresholdCandidate] = []
    for threshold in grid.tolist():
        accepted = score_values <= float(threshold)
        accepted_rows = int(np.sum(accepted, dtype=np.int64))
        coverage = float(accepted_rows / population_n)
        accepted_calibration = accepted[calibration]
        calibration_losses = accepted_calibration.astype(np.float64) * loss_values[calibration]
        calibration_accepted_rows = int(np.sum(accepted_calibration, dtype=np.int64))
        target_mean = float(risk_target * coverage)
        log_capital = wsr_betting_log_capital(
            calibration_losses,
            population_mean_null=target_mean,
            population_size=population_n,
        )
        upper_bound = wsr_population_mean_upper_bound(
            calibration_losses,
            population_size=population_n,
            alpha=beta,
        )
        selective_upper = float(upper_bound / coverage) if coverage > 0.0 else 1.0
        calibration_loss_sum = float(np.sum(calibration_losses, dtype=np.float64))
        calibration_selective = (
            float(calibration_loss_sum / calibration_accepted_rows)
            if calibration_accepted_rows > 0
            else None
        )
        certified = bool(
            log_capital >= log_boundary
            and selective_upper <= risk_target + 1e-10
        )
        candidate_rows.append(
            WSRThresholdCandidate(
                threshold=float(threshold),
                population_rows=population_n,
                population_accepted_rows=accepted_rows,
                coverage=coverage,
                calibration_rows=int(calibration.size),
                calibration_accepted_rows=calibration_accepted_rows,
                calibration_mean_accept_loss=float(calibration_loss_sum / calibration.size),
                calibration_selective_risk=calibration_selective,
                target_population_loss_mean=target_mean,
                log_capital_at_target=float(log_capital),
                wsr_population_loss_ucb=float(upper_bound),
                wsr_selective_risk_ucb=selective_upper,
                delta_per_candidate=float(beta),
                risk_bound=risk_target,
                certified=certified,
            )
        )

    certified_rows = [row for row in candidate_rows if row.certified]
    selected = (
        max(
            certified_rows,
            key=lambda row: (
                float(row.coverage),
                -float(row.wsr_selective_risk_ucb),
                float(row.threshold),
            ),
        )
        if certified_rows
        else None
    )
    result = {
        "selection_status": "ok" if selected is not None else "no_threshold_satisfies_wsr_risk_bound",
        "threshold": None if selected is None else float(selected.threshold),
        "candidate_count": int(len(candidate_rows)),
        "delta_per_candidate": float(beta),
        "risk_bound": risk_target,
        "certified_candidate_count": int(len(certified_rows)),
        "candidates": [asdict(row) for row in candidate_rows],
    }
    if selected is not None:
        result.update(asdict(selected))
    return result


__all__ = [
    "WSRThresholdCandidate",
    "certify_wsr_thresholds",
    "normalized_absolute_error",
    "quantile_threshold_grid",
    "wsr_betting_log_capital",
    "wsr_population_mean_upper_bound",
]
