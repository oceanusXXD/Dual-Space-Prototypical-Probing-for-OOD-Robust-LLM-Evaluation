from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BehaviorWarningConfig:
    """Unlabeled B-space early-warning thresholds calibrated on held-out ID rows."""

    accuracy_drop_tolerance: float = 0.05
    confidence_quantile: float = 0.10
    margin_quantile: float = 0.10
    energy_quantile: float = 0.90
    ood_quantile: float = 0.90
    minimum_records: int = 2
    atc_maximum_validation_mae: float = 0.05
    agreement_enabled: bool = True
    agreement_minimum_environments: int = 3
    agreement_minimum_records_per_environment: int = 2
    agreement_minimum_r2: float = 0.0

    def __post_init__(self) -> None:
        if float(self.accuracy_drop_tolerance) < 0.0:
            raise ValueError("accuracy_drop_tolerance must be non-negative")
        for name in ("confidence_quantile", "margin_quantile", "energy_quantile", "ood_quantile"):
            value = float(getattr(self, name))
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be in (0, 1)")
        if int(self.minimum_records) < 1:
            raise ValueError("minimum_records must be positive")
        if not 0.0 <= float(self.atc_maximum_validation_mae) <= 1.0:
            raise ValueError("atc_maximum_validation_mae must be in [0, 1]")
        if int(self.agreement_minimum_environments) < 2:
            raise ValueError("agreement_minimum_environments must be at least two")
        if int(self.agreement_minimum_records_per_environment) < 1:
            raise ValueError("agreement_minimum_records_per_environment must be positive")
        if not -1.0 <= float(self.agreement_minimum_r2) <= 1.0:
            raise ValueError("agreement_minimum_r2 must be in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgreementOnLineCalibrator:
    minimum_environments: int = 3
    minimum_records_per_environment: int = 2
    minimum_r2: float = 0.0
    slope_: float | None = None
    intercept_: float | None = None
    r2_: float | None = None
    mae_: float | None = None
    fit_r2_: float | None = None
    fit_mae_: float | None = None
    reference_accuracy_: float | None = None
    environment_rows_: list[dict[str, Any]] | None = None
    fit_environment_rows_: list[dict[str, Any]] | None = None
    validation_environment_rows_: list[dict[str, Any]] | None = None
    validation_scope_: str | None = None
    candidate_count_: int = 0
    available_: bool = False
    unavailable_reason_: str | None = None

    def fit(
        self,
        *,
        predictions: np.ndarray,
        labels: np.ndarray,
        environment_ids: np.ndarray,
        validation_predictions: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
        validation_environment_ids: np.ndarray | None = None,
    ) -> "AgreementOnLineCalibrator":
        matrix = np.asarray(predictions)
        target = np.asarray(labels)
        environments = np.asarray(environment_ids).astype(str)
        if matrix.ndim != 2 or matrix.shape[1] < 2:
            self.unavailable_reason_ = "fewer_than_two_judge_candidates"
            return self
        if not (len(matrix) == len(target) == len(environments)):
            raise ValueError("Agreement predictions, labels, and environments must align")
        self.candidate_count_ = int(matrix.shape[1])
        fit_rows = _agreement_environment_rows(
            matrix,
            target,
            environments,
            minimum_records=int(self.minimum_records_per_environment),
            role="fit",
        )
        self.fit_environment_rows_ = fit_rows
        if len(fit_rows) < int(self.minimum_environments):
            self.unavailable_reason_ = "insufficient_labeled_calibration_environments"
            self.environment_rows_ = fit_rows
            return self
        fit_agreements = np.asarray(
            [float(row["agreement"]) for row in fit_rows], dtype=np.float64
        )
        fit_accuracies = np.asarray(
            [float(row["accuracy"]) for row in fit_rows], dtype=np.float64
        )
        if float(np.var(fit_agreements)) <= 1e-12:
            self.unavailable_reason_ = "agreement_has_no_environment_variation"
            self.environment_rows_ = fit_rows
            return self
        design = np.column_stack(
            [fit_agreements, np.ones(len(fit_agreements), dtype=np.float64)]
        )
        coefficients, _, _, _ = np.linalg.lstsq(design, fit_accuracies, rcond=None)
        self.slope_ = float(coefficients[0])
        self.intercept_ = float(coefficients[1])
        fitted = design @ coefficients
        self.fit_r2_, self.fit_mae_ = _line_validation_metrics(fit_accuracies, fitted)

        validation_inputs = (
            validation_predictions,
            validation_labels,
            validation_environment_ids,
        )
        if all(value is None for value in validation_inputs):
            validation_rows = fit_rows
            self.validation_scope_ = "in_sample_fallback"
        elif any(value is None for value in validation_inputs):
            raise ValueError("Agreement validation inputs must be provided together")
        else:
            validation_matrix = np.asarray(validation_predictions)
            validation_target = np.asarray(validation_labels)
            validation_environments = np.asarray(validation_environment_ids).astype(str)
            if validation_matrix.ndim != 2 or validation_matrix.shape[1] != self.candidate_count_:
                raise ValueError("Agreement validation predictions must match the fitted ensemble")
            if not (
                len(validation_matrix)
                == len(validation_target)
                == len(validation_environments)
            ):
                raise ValueError("Agreement validation predictions, labels, and environments must align")
            validation_rows = _agreement_environment_rows(
                validation_matrix,
                validation_target,
                validation_environments,
                minimum_records=int(self.minimum_records_per_environment),
                role="validation",
            )
            self.validation_scope_ = "independent_labeled_development_environments"
        self.validation_environment_rows_ = validation_rows
        self.environment_rows_ = (
            fit_rows if self.validation_scope_ == "in_sample_fallback" else fit_rows + validation_rows
        )
        if len(validation_rows) < int(self.minimum_environments):
            self.unavailable_reason_ = "insufficient_labeled_validation_environments"
            return self
        validation_agreements = np.asarray(
            [float(row["agreement"]) for row in validation_rows], dtype=np.float64
        )
        validation_accuracies = np.asarray(
            [float(row["accuracy"]) for row in validation_rows], dtype=np.float64
        )
        if float(np.var(validation_agreements)) <= 1e-12:
            self.unavailable_reason_ = "validation_agreement_has_no_environment_variation"
            return self
        validation_fitted = self.slope_ * validation_agreements + self.intercept_
        self.r2_, self.mae_ = _line_validation_metrics(
            validation_accuracies,
            validation_fitted,
        )
        self.reference_accuracy_ = float(np.mean(validation_accuracies))
        self.available_ = bool(self.r2_ >= float(self.minimum_r2))
        if not self.available_:
            self.unavailable_reason_ = "accuracy_on_the_line_validation_failed"
        return self

    def estimate(self, predictions: np.ndarray) -> dict[str, Any]:
        matrix = np.asarray(predictions)
        if matrix.ndim != 2 or matrix.shape[1] != int(self.candidate_count_):
            raise ValueError("Target agreement predictions do not match the calibrated ensemble")
        agreement = _mean_pairwise_agreement(matrix)
        result: dict[str, Any] = {
            "available": bool(self.available_),
            "agreement": float(agreement),
            "candidate_count": int(self.candidate_count_),
            "unavailable_reason": self.unavailable_reason_,
        }
        if self.available_:
            assert self.slope_ is not None and self.intercept_ is not None
            estimated_accuracy = float(np.clip(self.slope_ * agreement + self.intercept_, 0.0, 1.0))
            result.update(
                {
                    "estimated_accuracy": estimated_accuracy,
                    "reference_accuracy": self.reference_accuracy_,
                    "estimated_accuracy_drop": float(
                        float(self.reference_accuracy_) - estimated_accuracy
                    ),
                }
            )
        return result

    def to_metadata(self) -> dict[str, Any]:
        return {
            "method": "Agreement-on-the-Line",
            "available": bool(self.available_),
            "unavailable_reason": self.unavailable_reason_,
            "candidate_count": int(self.candidate_count_),
            "environment_count": int(len(self.environment_rows_ or [])),
            "fit_environment_count": int(len(self.fit_environment_rows_ or [])),
            "validation_environment_count": int(
                len(self.validation_environment_rows_ or [])
            ),
            "minimum_environments": int(self.minimum_environments),
            "minimum_records_per_environment": int(self.minimum_records_per_environment),
            "minimum_r2": float(self.minimum_r2),
            "slope": self.slope_,
            "intercept": self.intercept_,
            "r2": self.r2_,
            "validation_mae": self.mae_,
            "fit_r2": self.fit_r2_,
            "fit_mae": self.fit_mae_,
            "reference_accuracy": self.reference_accuracy_,
            "environments": self.environment_rows_ or [],
            "fit_environments": self.fit_environment_rows_ or [],
            "validation_environments": self.validation_environment_rows_ or [],
            "fit_scope": "labeled_training_calibration_and_validation_environments_only",
            "validation_scope": self.validation_scope_,
            "selection_used_deployment_labels": False,
        }


@dataclass
class BehaviorWarningCalibrator:
    config: BehaviorWarningConfig
    source_accuracy_: float | None = None
    source_confidence_: float | None = None
    calibration_thresholds_: dict[str, float] | None = None
    atc_validation_rows_: list[dict[str, Any]] | None = None
    atc_validation_mae_: float | None = None
    atc_trigger_eligible_: bool = False
    atc_unavailable_reason_: str | None = None
    agreement_: AgreementOnLineCalibrator | None = None

    def fit(
        self,
        *,
        source_probabilities: np.ndarray,
        source_labels: np.ndarray,
        source_predictions: np.ndarray,
        calibration_probabilities: np.ndarray,
        calibration_logits: np.ndarray,
        calibration_ood_scores: np.ndarray,
        atc_validation_probabilities: np.ndarray | None = None,
        atc_validation_labels: np.ndarray | None = None,
        atc_validation_predictions: np.ndarray | None = None,
        atc_validation_environment_ids: np.ndarray | None = None,
        agreement_predictions: np.ndarray | None = None,
        agreement_labels: np.ndarray | None = None,
        agreement_environment_ids: np.ndarray | None = None,
        agreement_validation_predictions: np.ndarray | None = None,
        agreement_validation_labels: np.ndarray | None = None,
        agreement_validation_environment_ids: np.ndarray | None = None,
    ) -> "BehaviorWarningCalibrator":
        source_probs = _probabilities(source_probabilities, "source probabilities")
        calibration_probs = _probabilities(calibration_probabilities, "calibration probabilities")
        calibration_logits = _matrix(calibration_logits, "calibration logits")
        calibration_ood = _vector(calibration_ood_scores, "calibration OOD scores")
        labels = np.asarray(source_labels)
        predictions = np.asarray(source_predictions)
        if len(source_probs) != len(labels) or len(labels) != len(predictions):
            raise ValueError("source labels, predictions, and probabilities must align")
        if len(calibration_probs) != len(calibration_logits) or len(calibration_probs) != len(calibration_ood):
            raise ValueError("calibration B-space signals must align")
        self.source_accuracy_ = float(np.mean(predictions == labels))
        source_confidence = np.max(source_probs, axis=1)
        self.source_confidence_ = float(source_confidence.mean())
        source_error_rate = float(1.0 - self.source_accuracy_)
        atc_threshold = float(np.quantile(source_confidence, source_error_rate))
        calibration_confidence = np.max(calibration_probs, axis=1)
        calibration_margin = _margin(calibration_probs)
        calibration_energy = _energy(calibration_logits)
        self.calibration_thresholds_ = {
            "atc_confidence_threshold": atc_threshold,
            "confidence_floor": float(np.quantile(calibration_confidence, self.config.confidence_quantile)),
            "margin_floor": float(np.quantile(calibration_margin, self.config.margin_quantile)),
            "energy_ceiling": float(np.quantile(calibration_energy, self.config.energy_quantile)),
            "ood_score_ceiling": float(np.quantile(calibration_ood, self.config.ood_quantile)),
            "calibration_confidence_mean": float(calibration_confidence.mean()),
        }
        atc_validation_inputs = (
            atc_validation_probabilities,
            atc_validation_labels,
            atc_validation_predictions,
            atc_validation_environment_ids,
        )
        if all(value is None for value in atc_validation_inputs):
            self.atc_unavailable_reason_ = "independent_atc_validation_not_provided"
        elif any(value is None for value in atc_validation_inputs):
            raise ValueError("ATC validation inputs must be provided together")
        else:
            validation_probs = _probabilities(
                atc_validation_probabilities,
                "ATC validation probabilities",
            )
            validation_labels = np.asarray(atc_validation_labels)
            validation_predictions = np.asarray(atc_validation_predictions)
            validation_environments = np.asarray(
                atc_validation_environment_ids
            ).astype(str)
            if not (
                len(validation_probs)
                == len(validation_labels)
                == len(validation_predictions)
                == len(validation_environments)
            ):
                raise ValueError("ATC validation inputs must align")
            self.atc_validation_rows_ = _atc_environment_rows(
                probabilities=validation_probs,
                labels=validation_labels,
                predictions=validation_predictions,
                environment_ids=validation_environments,
                confidence_threshold=atc_threshold,
                minimum_records=int(self.config.minimum_records),
            )
            if not self.atc_validation_rows_:
                self.atc_unavailable_reason_ = "insufficient_atc_validation_environments"
            else:
                self.atc_validation_mae_ = float(
                    np.mean(
                        [
                            abs(
                                float(row["estimated_accuracy"])
                                - float(row["actual_accuracy"])
                            )
                            for row in self.atc_validation_rows_
                        ]
                    )
                )
                self.atc_trigger_eligible_ = bool(
                    self.atc_validation_mae_
                    <= float(self.config.atc_maximum_validation_mae)
                )
                if not self.atc_trigger_eligible_:
                    self.atc_unavailable_reason_ = (
                        "accuracy_on_the_line_validation_mae_exceeded"
                    )
        if bool(self.config.agreement_enabled):
            self.agreement_ = AgreementOnLineCalibrator(
                minimum_environments=int(self.config.agreement_minimum_environments),
                minimum_records_per_environment=int(
                    self.config.agreement_minimum_records_per_environment
                ),
                minimum_r2=float(self.config.agreement_minimum_r2),
            )
            if (
                agreement_predictions is None
                or agreement_labels is None
                or agreement_environment_ids is None
            ):
                self.agreement_.unavailable_reason_ = "agreement_calibration_inputs_not_provided"
            else:
                self.agreement_.fit(
                    predictions=agreement_predictions,
                    labels=agreement_labels,
                    environment_ids=agreement_environment_ids,
                    validation_predictions=agreement_validation_predictions,
                    validation_labels=agreement_validation_labels,
                    validation_environment_ids=agreement_validation_environment_ids,
                )
        return self

    def evaluate(
        self,
        *,
        probabilities: np.ndarray,
        logits: np.ndarray,
        ood_scores: np.ndarray,
        record_indices: np.ndarray,
        agreement_predictions: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self.source_accuracy_ is None or self.source_confidence_ is None or self.calibration_thresholds_ is None:
            raise RuntimeError("BehaviorWarningCalibrator is not fitted")
        probs = _probabilities(probabilities, "cluster probabilities")
        values = _matrix(logits, "cluster logits")
        ood = _vector(ood_scores, "cluster OOD scores")
        indices = np.asarray(record_indices, dtype=int)
        if not (len(probs) == len(values) == len(ood) == len(indices)):
            raise ValueError("cluster B-space signals and record indices must align")
        if len(indices) < int(self.config.minimum_records):
            return {
                "triggered": False,
                "reason": "insufficient_cluster_records",
                "record_count": int(len(indices)),
                "ranked_record_indices": indices.astype(int).tolist(),
            }
        confidence = np.max(probs, axis=1)
        margin = _margin(probs)
        energy = _energy(values)
        atc_accuracy = float(np.mean(confidence >= self.calibration_thresholds_["atc_confidence_threshold"]))
        doc_accuracy = float(
            np.clip(
                self.source_accuracy_ + (float(confidence.mean()) - self.source_confidence_),
                0.0,
                1.0,
            )
        )
        atc_drop = float(self.source_accuracy_ - atc_accuracy)
        doc_drop = float(self.source_accuracy_ - doc_accuracy)
        harm_risk_components = {
            "atc_accuracy_drop": bool(
                self.atc_trigger_eligible_
                and atc_drop >= float(self.config.accuracy_drop_tolerance)
            ),
            "doc_accuracy_drop": doc_drop >= float(self.config.accuracy_drop_tolerance),
            "confidence_below_id_floor": float(confidence.mean()) < self.calibration_thresholds_["confidence_floor"],
            "margin_below_id_floor": float(margin.mean()) < self.calibration_thresholds_["margin_floor"],
        }
        shift_evidence_components = {
            "energy_above_id_ceiling": float(energy.mean()) > self.calibration_thresholds_["energy_ceiling"],
            "ood_score_above_id_ceiling": float(ood.mean()) > self.calibration_thresholds_["ood_score_ceiling"],
        }
        agreement_result: dict[str, Any] = {
            "available": False,
            "unavailable_reason": "agreement_disabled_or_not_calibrated",
        }
        if self.agreement_ is not None:
            if agreement_predictions is None:
                agreement_result = {
                    "available": False,
                    "unavailable_reason": "target_ensemble_predictions_not_provided",
                }
            else:
                target_agreement = np.asarray(agreement_predictions)
                if target_agreement.shape[0] != len(indices):
                    raise ValueError("target agreement predictions must align with cluster records")
                agreement_result = self.agreement_.estimate(target_agreement)
                harm_risk_components["agreement_estimated_accuracy_drop"] = bool(
                    agreement_result.get("available")
                    and float(agreement_result.get("estimated_accuracy_drop", 0.0))
                    >= float(self.config.accuracy_drop_tolerance)
                )
        risk = _risk_score(
            confidence=confidence,
            margin=margin,
            energy=energy,
            ood=ood,
            thresholds=self.calibration_thresholds_,
        )
        order = np.argsort(-risk, kind="stable")
        return {
            "triggered": bool(any(harm_risk_components.values())),
            "reason": "B_space_unlabeled_harm_risk_warning",
            "record_count": int(len(indices)),
            "source_accuracy": float(self.source_accuracy_),
            "atc_estimated_accuracy": atc_accuracy,
            "doc_estimated_accuracy": doc_accuracy,
            "atc_estimated_accuracy_drop": atc_drop,
            "atc_trigger_eligible": bool(self.atc_trigger_eligible_),
            "atc_unavailable_reason": self.atc_unavailable_reason_,
            "doc_estimated_accuracy_drop": doc_drop,
            "agreement_on_the_line": agreement_result,
            "mean_confidence": float(confidence.mean()),
            "mean_margin": float(margin.mean()),
            "mean_energy": float(energy.mean()),
            "mean_ood_score": float(ood.mean()),
            "harm_risk_components": harm_risk_components,
            "shift_evidence_components": shift_evidence_components,
            "trigger_components": harm_risk_components,
            "shift_evidence_present": bool(any(shift_evidence_components.values())),
            "ranked_record_indices": indices[order].astype(int).tolist(),
            "ranking_signal": "calibrated_B_space_risk_composite",
        }

    def to_metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "fit_scope": "source_train_thresholds_training_calibration_B_signals_and_source_validation",
            "source_accuracy": self.source_accuracy_,
            "thresholds": self.calibration_thresholds_,
            "atc_accuracy_on_the_line": {
                "trigger_eligible": bool(self.atc_trigger_eligible_),
                "unavailable_reason": self.atc_unavailable_reason_,
                "validation_mae": self.atc_validation_mae_,
                "maximum_validation_mae": float(
                    self.config.atc_maximum_validation_mae
                ),
                "validation_environment_count": int(
                    len(self.atc_validation_rows_ or [])
                ),
                "validation_environments": self.atc_validation_rows_ or [],
                "failed_validation_policy": "ranking_and_reporting_only_no_trigger",
            },
            "agreement_on_the_line": self.agreement_.to_metadata() if self.agreement_ is not None else {
                "available": False,
                "unavailable_reason": "disabled",
            },
            "decision_boundary": "harm_risk_warning_is_advisory_and_never_required_for_probe",
            "shift_evidence_policy": "ranking_and_reporting_only_never_triggers_harm_warning_alone",
        }


def _atc_environment_rows(
    *,
    probabilities: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    environment_ids: np.ndarray,
    confidence_threshold: float,
    minimum_records: int,
) -> list[dict[str, Any]]:
    confidence = np.max(np.asarray(probabilities, dtype=np.float64), axis=1)
    rows: list[dict[str, Any]] = []
    for environment in sorted(set(environment_ids.tolist())):
        local = environment_ids == environment
        if int(local.sum()) < int(minimum_records):
            continue
        estimated = float(np.mean(confidence[local] >= float(confidence_threshold)))
        actual = float(np.mean(predictions[local] == labels[local]))
        rows.append(
            {
                "environment_id": str(environment),
                "record_count": int(local.sum()),
                "estimated_accuracy": estimated,
                "actual_accuracy": actual,
                "absolute_error": float(abs(estimated - actual)),
            }
        )
    return rows


def _mean_pairwise_agreement(predictions: np.ndarray) -> float:
    matrix = np.asarray(predictions)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("Pairwise agreement requires a non-empty [N, M] matrix with M >= 2")
    agreements = [
        float(np.mean(matrix[:, left] == matrix[:, right]))
        for left in range(matrix.shape[1])
        for right in range(left + 1, matrix.shape[1])
    ]
    return float(np.mean(agreements))


def _agreement_environment_rows(
    predictions: np.ndarray,
    labels: np.ndarray,
    environment_ids: np.ndarray,
    *,
    minimum_records: int,
    role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for environment in sorted(set(environment_ids.tolist())):
        local = environment_ids == environment
        if int(local.sum()) < int(minimum_records):
            continue
        local_predictions = predictions[local]
        rows.append(
            {
                "environment_id": environment,
                "environment_role": str(role),
                "record_count": int(local.sum()),
                "agreement": _mean_pairwise_agreement(local_predictions),
                "accuracy": float(np.mean(local_predictions == labels[local, None])),
            }
        )
    return rows


def _line_validation_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> tuple[float, float]:
    residual = float(np.sum((observed - predicted) ** 2))
    total = float(np.sum((observed - observed.mean()) ** 2))
    r2 = float(1.0 - residual / total) if total > 1e-12 else 0.0
    mae = float(np.mean(np.abs(observed - predicted)))
    return r2, mae


def _probabilities(values: np.ndarray, name: str) -> np.ndarray:
    matrix = _matrix(values, name)
    if np.any(matrix < 0.0) or not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"{name} must be normalized probability rows")
    return matrix


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2 or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite non-empty [N, K] matrix")
    return matrix


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite non-empty vector")
    return vector


def _margin(probabilities: np.ndarray) -> np.ndarray:
    ordered = np.sort(probabilities, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _energy(logits: np.ndarray) -> np.ndarray:
    maxima = np.max(logits, axis=1, keepdims=True)
    return -(maxima[:, 0] + np.log(np.exp(logits - maxima).sum(axis=1)))


def _risk_score(
    *,
    confidence: np.ndarray,
    margin: np.ndarray,
    energy: np.ndarray,
    ood: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    scales = np.asarray(
        [
            max(abs(float(thresholds["confidence_floor"])), 1e-6),
            max(abs(float(thresholds["margin_floor"])), 1e-6),
            max(abs(float(thresholds["energy_ceiling"])), 1e-6),
            max(abs(float(thresholds["ood_score_ceiling"])), 1e-6),
        ]
    )
    parts = np.column_stack(
        [
            (float(thresholds["confidence_floor"]) - confidence) / scales[0],
            (float(thresholds["margin_floor"]) - margin) / scales[1],
            (energy - float(thresholds["energy_ceiling"])) / scales[2],
            (ood - float(thresholds["ood_score_ceiling"])) / scales[3],
        ]
    )
    return parts.mean(axis=1)
