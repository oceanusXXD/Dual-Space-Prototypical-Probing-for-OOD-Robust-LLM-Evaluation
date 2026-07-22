from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.llm_judge_ood.scores.knn import DocumentKNNScorer, Thresholds
from src.llm_judge_ood.scores.rmd import DocumentGaussianScorer
from src.llm_judge_ood.shared.metrics import ood_metrics
from src.llm_judge_ood.shared.representation import (
    RepresentationSpec,
    RepresentationTransform,
    expand_representation_specs,
)
from src.llm_judge_ood.shared.whitening import LayerPreprocessor


DocumentDetector = DocumentKNNScorer | DocumentGaussianScorer


@dataclass(frozen=True)
class OODSelectionConfig:
    preprocess_methods: tuple[str, ...] = ("pca_whiten",)
    representations: tuple[str, ...] = ("last_layer", "all_layers", "layer_mean", "pca")
    detectors: tuple[str, ...] = ("knn", "mahalanobis")
    metrics: tuple[str, ...] = ("cosine", "euclidean")
    k_values: tuple[int, ...] = (5, 10, 20, 50)
    pca_dim: int = 48
    gaussian_max_dim: int = 128
    gaussian_regularization: float = 1e-5
    soft_quantile: float = 0.90
    hard_quantile: float = 0.95
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OODSelectionResult:
    preprocessor: LayerPreprocessor
    representation: RepresentationTransform
    scorer: DocumentDetector
    embeddings: np.ndarray
    scores: np.ndarray
    score_labels: np.ndarray
    thresholds: Thresholds
    selected_candidate: dict[str, Any]
    candidate_results: list[dict[str, Any]]
    development_metrics: dict[str, Any]
    config: OODSelectionConfig
    input_document_ids: np.ndarray
    unique_document_count: int

    def refreshed(
        self,
        raw_features: np.ndarray,
        input_document_ids: np.ndarray,
    ) -> "OODSelectionResult":
        """Score a new full matrix with the cached source-fitted detector."""

        values = _as_layers(raw_features)
        document_ids = np.asarray(input_document_ids).astype(str)
        if len(values) != len(document_ids):
            raise ValueError("Document OOD refresh features and document IDs must align")
        processed = self.preprocessor.transform(values)
        embeddings = self.representation.transform(processed)
        scores = self.scorer.score(embeddings)
        return replace(
            self,
            embeddings=embeddings,
            scores=scores,
            score_labels=self.scorer.labels(scores),
            input_document_ids=document_ids,
            unique_document_count=int(len(set(document_ids.tolist()))),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "ood_definition": "document_distribution",
            "selection_scope": "development_documents_only",
            "feature_scope": "input_document",
            "unique_document_count": int(self.unique_document_count),
            "selected_candidate": self.selected_candidate,
            "development_metrics": self.development_metrics,
            "preprocessor": self.preprocessor.to_metadata(),
            "representation": self.representation.to_metadata(),
            "scorer": self.scorer.to_metadata(),
            "candidate_results": self.candidate_results,
            "candidate_tie_break": "detector_then_lower_knn_k_then_preprocess_representation_metric",
            "execution": {
                "preprocessing": "cpu_sklearn_numpy",
                "representation": "cpu_sklearn_numpy",
                "detectors": ["sklearn_knn", "global_mahalanobis"],
            },
        }

    def save_artifacts(self, output_dir: str | Path) -> dict[str, str]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        preprocessor_path = root / "ood_preprocessor.npz"
        representation_path = root / "ood_representation.npz"
        threshold_path = root / "ood_thresholds.json"
        np.savez(
            preprocessor_path,
            **self.preprocessor.artifact_arrays(),
            metadata_json=np.asarray(json.dumps(self.preprocessor.to_metadata(), ensure_ascii=False)),
        )
        np.savez(
            representation_path,
            **self.representation.artifact_arrays(),
            metadata_json=np.asarray(json.dumps(self.representation.to_metadata(), ensure_ascii=False)),
        )
        threshold_path.write_text(
            json.dumps(
                {
                    **self.thresholds.to_dict(),
                    "selected_detector": self.selected_candidate["detector"],
                    "selected_space": self.selected_candidate["space"],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {
            "preprocessor": str(preprocessor_path),
            "representation": str(representation_path),
            "thresholds": str(threshold_path),
        }


def select_document_ood_detector(
    *,
    raw_features: np.ndarray,
    input_document_ids: np.ndarray,
    training_document_mask: np.ndarray,
    calibration_document_mask: np.ndarray,
    development_document_mask: np.ndarray,
    config: OODSelectionConfig | None = None,
    prefitted_preprocessors: dict[str, LayerPreprocessor] | None = None,
) -> OODSelectionResult:
    """Select a label-free OOD detector from input-document features only.

    Repeated Judge rows for one input document are collapsed before fitting,
    calibration, candidate selection, metric computation, and scoring. Scores
    are expanded back to the original record order only for joined artifacts.
    """

    cfg = config or OODSelectionConfig()
    if any(str(kind).lower() == "u" for kind in cfg.representations):
        raise ValueError("Document OOD cannot use Judge-derived representation 'u'")
    unsupported = sorted({str(detector).lower() for detector in cfg.detectors} - {"knn", "mahalanobis"})
    if unsupported:
        raise ValueError(f"Document OOD detectors must be label-free; unsupported={unsupported}")

    values = _as_layers(raw_features)
    document_ids = np.asarray(input_document_ids).astype(str)
    training = np.asarray(training_document_mask, dtype=bool)
    calibration = np.asarray(calibration_document_mask, dtype=bool)
    development = np.asarray(development_document_mask, dtype=bool)
    if not (
        len(values)
        == len(document_ids)
        == len(training)
        == len(calibration)
        == len(development)
    ):
        raise ValueError("Document OOD inputs must be aligned")

    (
        unique_features,
        unique_document_ids,
        unique_training,
        unique_calibration,
        unique_development,
        row_to_document,
    ) = _collapse_document_rows(
        features=values,
        document_ids=document_ids,
        training_mask=training,
        calibration_mask=calibration,
        development_mask=development,
    )
    if not unique_training.any() or not unique_calibration.any() or not unique_development.any():
        raise ValueError("Document OOD selection requires training, calibration, and development documents")
    if np.any(unique_training & (unique_calibration | unique_development)) or np.any(unique_calibration & unique_development):
        raise ValueError("Document OOD training, calibration, and development masks must be disjoint")
    _validate_knn_grid(cfg, fit_documents=int(unique_training.sum()))

    evaluation_mask = unique_calibration | unique_development
    raw_specs = expand_representation_specs(
        cfg.representations,
        num_layers=int(unique_features.shape[1]),
        pca_dim=int(cfg.pca_dim),
    )
    candidate_results: list[dict[str, Any]] = []
    preprocessors_by_method: dict[str, LayerPreprocessor] = {}
    processed_by_method: dict[str, np.ndarray] = {}
    representations_by_key: dict[tuple[str, str], RepresentationTransform] = {}
    for preprocess_method in cfg.preprocess_methods:
        normalized_method = str(preprocess_method).lower()
        preprocessor = (
            prefitted_preprocessors.get(normalized_method)
            if prefitted_preprocessors is not None
            else None
        )
        if preprocessor is None:
            preprocessor = LayerPreprocessor(
                method=normalized_method,
                pca_components=int(cfg.pca_dim),
                random_state=int(cfg.seed),
            ).fit(unique_features[unique_training])
        elif (
            str(preprocessor.method) != normalized_method
            or int(preprocessor.fit_rows_) != int(unique_training.sum())
            or preprocessor.input_shape_ != tuple(unique_features.shape[1:])
        ):
            raise ValueError("Prefitted document OOD preprocessor does not match source inputs")
        processed = preprocessor.transform(unique_features)
        preprocessors_by_method[preprocessor.method] = preprocessor
        processed_by_method[preprocessor.method] = processed
        space = "h" if preprocessor.method == "none" else "h_tilde"
        for spec in raw_specs:
            representation = RepresentationTransform(spec, random_state=int(cfg.seed)).fit(processed[unique_training])
            representations_by_key[(preprocessor.method, spec.name)] = representation
            training_embeddings = representation.transform(processed[unique_training])
            evaluation_embeddings = representation.transform(processed[evaluation_mask])
            candidate_results.extend(
                _score_document_candidate_space(
                    cfg=cfg,
                    training_embeddings=training_embeddings,
                    evaluation_embeddings=evaluation_embeddings,
                    calibration_mask=unique_calibration[evaluation_mask],
                    development_mask=unique_development[evaluation_mask],
                    space=space,
                    preprocess_method=preprocessor.method,
                    representation=spec,
                    condition_number_max=preprocessor.to_metadata()["condition_number_max"],
                    fit_documents=int(unique_training.sum()),
                    embedding_dim=int(training_embeddings.shape[1]),
                )
            )
    if not candidate_results:
        raise RuntimeError("Document OOD search produced no candidate configurations")

    selected = min(candidate_results, key=_candidate_sort_key)
    selected_method = str(selected["preprocess_method"])
    selected_spec = RepresentationSpec(**dict(selected["representation"]))
    selected_preprocessor = preprocessors_by_method[selected_method]
    selected_input = processed_by_method[selected_method]
    selected_representation = representations_by_key[(selected_method, selected_spec.name)]
    unique_embeddings = selected_representation.transform(selected_input)
    scorer = _fit_document_detector(
        detector=str(selected["detector"]),
        training_embeddings=unique_embeddings[unique_training],
        metric=selected.get("metric"),
        k=selected.get("k"),
        regularization=float(cfg.gaussian_regularization),
    )
    thresholds = scorer.calibrate(
        unique_embeddings[unique_calibration],
        soft_q=float(cfg.soft_quantile),
        hard_q=float(cfg.hard_quantile),
    )
    unique_scores = scorer.score(unique_embeddings)
    unique_score_labels = scorer.labels(unique_scores)
    development_metrics = ood_metrics(
        unique_development[evaluation_mask].astype(int),
        unique_scores[evaluation_mask],
    )
    return OODSelectionResult(
        preprocessor=selected_preprocessor,
        representation=selected_representation,
        scorer=scorer,
        embeddings=unique_embeddings[row_to_document],
        scores=unique_scores[row_to_document],
        score_labels=unique_score_labels[row_to_document],
        thresholds=thresholds,
        selected_candidate=selected,
        candidate_results=candidate_results,
        development_metrics=development_metrics,
        config=cfg,
        input_document_ids=document_ids,
        unique_document_count=int(len(unique_document_ids)),
    )


def _collapse_document_rows(
    *,
    features: np.ndarray,
    document_ids: np.ndarray,
    training_mask: np.ndarray,
    calibration_mask: np.ndarray,
    development_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    first_indices: list[int] = []
    row_to_document = np.empty(len(document_ids), dtype=int)
    by_document: dict[str, list[int]] = {}
    for index, document_id in enumerate(document_ids.tolist()):
        by_document.setdefault(str(document_id), []).append(index)
    for document_index, (document_id, indices) in enumerate(by_document.items()):
        reference = indices[0]
        document_features = features[np.asarray(indices, dtype=int)]
        if not np.allclose(document_features, features[reference], rtol=1e-6, atol=1e-6):
            raise ValueError(f"Input-document features differ across Judge rows for {document_id!r}")
        for name, mask in (
            ("training", training_mask),
            ("calibration", calibration_mask),
            ("development", development_mask),
        ):
            if len(set(mask[np.asarray(indices, dtype=int)].tolist())) != 1:
                raise ValueError(f"Input-document {name} membership differs across Judge rows for {document_id!r}")
        first_indices.append(reference)
        row_to_document[np.asarray(indices, dtype=int)] = document_index
    selected = np.asarray(first_indices, dtype=int)
    return (
        features[selected],
        document_ids[selected],
        training_mask[selected],
        calibration_mask[selected],
        development_mask[selected],
        row_to_document,
    )


def _score_document_candidate_space(
    *,
    cfg: OODSelectionConfig,
    training_embeddings: np.ndarray,
    evaluation_embeddings: np.ndarray,
    calibration_mask: np.ndarray,
    development_mask: np.ndarray,
    space: str,
    preprocess_method: str,
    representation: RepresentationSpec,
    condition_number_max: float | None,
    fit_documents: int,
    embedding_dim: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for detector in tuple(str(value).lower() for value in cfg.detectors):
        if detector == "knn":
            candidates = [
                (str(metric), int(k))
                for metric in cfg.metrics
                for k in cfg.k_values
            ]
            for metric in cfg.metrics:
                metric_ks = tuple(int(k) for candidate_metric, k in candidates if candidate_metric == str(metric))
                scorer = DocumentKNNScorer(
                    k=max(metric_ks),
                    metric=str(metric),
                ).fit(training_embeddings)
                scores_by_k = scorer.score_at_ks(evaluation_embeddings, metric_ks)
                for k in metric_ks:
                    rows.append(
                        _document_candidate_row(
                            detector="knn",
                            metric=str(metric),
                            k=int(k),
                            scores=scores_by_k[int(k)],
                            calibration_mask=calibration_mask,
                            development_mask=development_mask,
                            space=space,
                            preprocess_method=preprocess_method,
                            representation=representation,
                            condition_number_max=condition_number_max,
                            fit_documents=fit_documents,
                            embedding_dim=embedding_dim,
                        )
                    )
        elif detector == "mahalanobis":
            if training_embeddings.shape[1] > int(cfg.gaussian_max_dim):
                continue
            scorer = DocumentGaussianScorer(regularization=float(cfg.gaussian_regularization)).fit(training_embeddings)
            rows.append(
                _document_candidate_row(
                    detector="mahalanobis",
                    metric="mahalanobis",
                    k=None,
                    scores=scorer.score(evaluation_embeddings),
                    calibration_mask=calibration_mask,
                    development_mask=development_mask,
                    space=space,
                    preprocess_method=preprocess_method,
                    representation=representation,
                    condition_number_max=condition_number_max,
                    fit_documents=fit_documents,
                    embedding_dim=embedding_dim,
                )
            )
        else:
            raise ValueError(f"Document OOD detector must be label-free, got {detector!r}")
    return rows


def _validate_knn_grid(cfg: OODSelectionConfig, *, fit_documents: int) -> None:
    if "knn" not in {str(value).lower() for value in cfg.detectors}:
        return
    values = tuple(int(value) for value in cfg.k_values)
    if not values:
        raise ValueError("Document kNN selection requires at least one k value")
    if len(set(values)) != len(values):
        raise ValueError("Document kNN k_values must not contain duplicates")
    invalid = [value for value in values if value < 1 or value > fit_documents]
    if invalid:
        raise ValueError(
            "Document kNN k_values must be within 1..fit_documents "
            f"(fit_documents={fit_documents}, invalid={invalid})"
        )


def _document_candidate_row(
    *,
    detector: str,
    metric: str,
    k: int | None,
    scores: np.ndarray,
    calibration_mask: np.ndarray,
    development_mask: np.ndarray,
    space: str,
    preprocess_method: str,
    representation: RepresentationSpec,
    condition_number_max: float | None,
    fit_documents: int,
    embedding_dim: int,
) -> dict[str, Any]:
    if len(scores) != len(calibration_mask) or len(scores) != len(development_mask):
        raise ValueError("Document OOD candidate scores and selection masks must align")
    if np.any(calibration_mask & development_mask):
        raise ValueError("Document OOD calibration and development masks must be disjoint")
    metrics = ood_metrics(development_mask.astype(int), scores)
    return {
        "detector": detector,
        "space": space,
        "preprocess_method": str(preprocess_method),
        "representation": representation.to_dict(),
        "representation_name": representation.name,
        "metric": metric,
        "k": int(k) if k is not None else None,
        "development_auroc": float(metrics["auroc"]),
        "development_aupr": float(metrics["aupr"]),
        "development_fpr95": float(metrics["fpr95"]),
        "condition_number_max": condition_number_max,
        "fit_rows": int(fit_documents),
        "fit_documents": int(fit_documents),
        "representation_output_dim": int(embedding_dim),
        "calibration_documents": int(calibration_mask.sum()),
        "development_documents": int(development_mask.sum()),
        "retrieval_scope": "global_training_document_bank",
        "selection_used_deployment_documents": False,
    }


def _fit_document_detector(
    *,
    detector: str,
    training_embeddings: np.ndarray,
    metric: Any,
    k: Any,
    regularization: float,
) -> DocumentDetector:
    if detector == "knn":
        return DocumentKNNScorer(k=int(k), metric=str(metric)).fit(training_embeddings)
    if detector == "mahalanobis":
        return DocumentGaussianScorer(regularization=regularization).fit(training_embeddings)
    raise ValueError(f"Document OOD detector must be label-free, got {detector!r}")


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str, int, str, str, str]:
    return (
        -float(row["development_auroc"]),
        -float(row["development_aupr"]),
        float(row["development_fpr95"]),
        str(row["detector"]),
        int(row["k"]) if row["k"] is not None else -1,
        str(row["preprocess_method"]),
        str(row["representation_name"]),
        str(row["metric"]),
    )


def _as_layers(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim == 2:
        values = values[:, None, :]
    if values.ndim != 3:
        raise ValueError(f"Expected [N,L,D] or [N,D], got {values.shape}")
    return values
