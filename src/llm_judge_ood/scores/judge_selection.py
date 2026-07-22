from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from src.llm_judge_ood.scores.knn import KNNScorer, Thresholds
from src.llm_judge_ood.scores.openood import (
    OPENOOD_POSTHOC_METHODS,
    OpenOODPosthocScorer,
    normalize_openood_method,
)
from src.llm_judge_ood.scores.rmd import RMDScorer
from src.llm_judge_ood.scores.vim import FullViMScorer, ViMScorer
from src.llm_judge_ood.shared.metrics import ood_metrics


JudgeBehaviorDetector = ViMScorer | FullViMScorer | RMDScorer | KNNScorer | OpenOODPosthocScorer


@dataclass(frozen=True)
class JudgeOODSelectionConfig:
    detectors: tuple[str, ...] = ("vim", "rmd", "knn")
    vim_score_variant: str = "residual_only"
    vim_ranks: tuple[int, ...] = (64, 128, 256, 512)
    vim_include_variance_rank: bool = True
    vim_variance_target: float = 0.90
    vim_rank_cap: int = 512
    rmd_regularization: float = 1e-5
    knn_ks: tuple[int, ...] = (10, 20, 50, 100)
    knn_include_sqrt_k: bool = True
    knn_metric: str = "euclidean"
    knn_normalize: bool = True
    openood_temperature: float = 1.0
    odin_temperature: float = 1000.0
    odin_epsilon: float = 0.0014
    react_quantiles: tuple[float, ...] = (0.85, 0.90, 0.95)
    dice_sparsity: float = 0.90
    ash_percentile: float = 65.0
    gen_gamma: float = 0.10
    soft_quantile: float = 0.90
    hard_quantile: float = 0.95

    def __post_init__(self) -> None:
        if not self.detectors:
            raise ValueError("Judge OOD selection requires at least one detector")
        if str(self.vim_score_variant).lower() != "residual_only":
            raise ValueError("vim_score_variant must be 'residual_only'")
        if any(int(value) < 1 for value in self.vim_ranks):
            raise ValueError("vim_ranks must be positive")
        if not 0.0 < float(self.vim_variance_target) <= 1.0:
            raise ValueError("vim_variance_target must be in (0, 1]")
        if int(self.vim_rank_cap) < 1:
            raise ValueError("vim_rank_cap must be positive")
        if any(int(value) < 1 for value in self.knn_ks):
            raise ValueError("knn_ks must be positive")
        if float(self.rmd_regularization) <= 0.0:
            raise ValueError("rmd_regularization must be positive")
        if not 0.0 <= float(self.soft_quantile) <= float(self.hard_quantile) <= 1.0:
            raise ValueError("score quantiles must satisfy 0 <= soft <= hard <= 1")
        if float(self.openood_temperature) <= 0.0 or float(self.odin_temperature) <= 0.0:
            raise ValueError("OpenOOD temperatures must be positive")
        if float(self.odin_epsilon) < 0.0:
            raise ValueError("odin_epsilon must be non-negative")
        if not self.react_quantiles or any(
            not 0.0 < float(value) < 1.0 for value in self.react_quantiles
        ):
            raise ValueError("react_quantiles must be non-empty values in (0, 1)")
        if not 0.0 <= float(self.dice_sparsity) < 1.0:
            raise ValueError("dice_sparsity must be in [0, 1)")
        if not 0.0 <= float(self.ash_percentile) < 100.0:
            raise ValueError("ash_percentile must be in [0, 100)")
        if float(self.gen_gamma) <= 0.0:
            raise ValueError("gen_gamma must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeOODSelectionResult:
    scorer: JudgeBehaviorDetector
    scores: np.ndarray
    score_labels: np.ndarray
    thresholds: Thresholds
    selected_candidate: dict[str, Any]
    candidate_results: list[dict[str, Any]]
    selection_decision: dict[str, Any]
    development_metrics: dict[str, float]
    config: JudgeOODSelectionConfig
    class_values: np.ndarray
    selection_scope: str = "training_calibration_id_plus_development_ood_documents"
    reference_scope: str = "training_train"

    def refreshed(
        self,
        penultimate: np.ndarray,
        logits: np.ndarray,
        query_ids: np.ndarray,
    ) -> "JudgeOODSelectionResult":
        """Score current records with a cached source-fitted B-space detector."""

        h = np.asarray(penultimate, dtype=np.float64)
        raw_logits = np.asarray(logits, dtype=np.float64)
        queries = np.asarray(query_ids).astype(str)
        if len(h) != len(raw_logits) or len(h) != len(queries):
            raise ValueError("Judge OOD refresh inputs must align")
        scores = _selected_scores(self.scorer, h, raw_logits, queries)
        return replace(
            self,
            scores=scores,
            score_labels=self.scorer.labels(scores),
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "feature_scope": "judge_behavior",
            "detection_unit": "judge_record",
            "selection_scope": self.selection_scope,
            "reference_scope": self.reference_scope,
            "selected_candidate": self.selected_candidate,
            "candidate_results": self.candidate_results,
            "selection_decision": self.selection_decision,
            "baseline_comparison": _baseline_comparison(
                self.candidate_results,
                self.selected_candidate,
            ),
            "development_metrics": self.development_metrics,
            "scorer": self.scorer.to_metadata(),
            "class_values": self.class_values.tolist(),
        }

    def save_artifact(
        self,
        output_dir: str | Path,
        *,
        judge_fingerprint: str,
        filename: str = "judge_behavior_ood_scorer.npz",
    ) -> str:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        path = root / str(filename)
        metadata = {
            **self.to_metadata(),
            "judge_fingerprint": str(judge_fingerprint),
        }
        arrays = self.scorer.artifact_arrays()
        np.savez(
            path,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
        )
        return str(path)


def select_judge_ood_detector(
    *,
    penultimate: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    class_values: np.ndarray,
    training_mask: np.ndarray,
    calibration_mask: np.ndarray,
    development_mask: np.ndarray,
    benchmark_mask: np.ndarray | None = None,
    document_ood_labels: np.ndarray | None = None,
    document_ids: np.ndarray | None = None,
    shift_types: np.ndarray | None = None,
    query_ids: np.ndarray | None = None,
    head_weight: np.ndarray | None = None,
    head_bias: np.ndarray | None = None,
    head_query_ids: np.ndarray | None = None,
    config: JudgeOODSelectionConfig | None = None,
) -> JudgeOODSelectionResult:
    """Select a B-space detector without fitting or tuning on deployment rows."""

    cfg = config or JudgeOODSelectionConfig()
    h = np.asarray(penultimate, dtype=np.float64)
    raw_logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels)
    classes = np.asarray(class_values)
    train = np.asarray(training_mask, dtype=bool)
    calibration = np.asarray(calibration_mask, dtype=bool)
    development = np.asarray(development_mask, dtype=bool)
    benchmark = (
        np.zeros(len(h), dtype=bool)
        if benchmark_mask is None
        else np.asarray(benchmark_mask, dtype=bool)
    )
    ood_truth = (
        np.zeros(len(h), dtype=bool)
        if document_ood_labels is None
        else np.asarray(document_ood_labels, dtype=bool)
    )
    documents = (
        np.asarray(document_ids).astype(str)
        if document_ids is not None
        else np.asarray([f"row_{index}" for index in range(len(h))], dtype=str)
    )
    queries = (
        np.asarray(query_ids).astype(str)
        if query_ids is not None
        else np.asarray(["__global__"] * len(h), dtype=str)
    )
    shifts = (
        np.asarray(shift_types).astype(str)
        if shift_types is not None
        else np.asarray(["unspecified"] * len(h), dtype=str)
    )
    if h.ndim != 2 or raw_logits.ndim != 2 or h.shape[0] != raw_logits.shape[0]:
        raise ValueError("Judge B-space selection requires aligned [N, D] penultimate features and [N, K] logits")
    if raw_logits.shape[1] != len(classes):
        raise ValueError("Judge B-space logits and class vocabulary must align")
    if not (
        len(h)
        == len(target)
        == len(train)
        == len(calibration)
        == len(development)
        == len(benchmark)
        == len(ood_truth)
        == len(documents)
        == len(queries)
        == len(shifts)
    ):
        raise ValueError("Judge B-space selection inputs must align")
    if not train.any() or not calibration.any() or not development.any():
        raise ValueError("Judge B-space selection requires training, calibration, and development records")
    if (
        np.any(train & (calibration | development | benchmark))
        or np.any(calibration & (development | benchmark))
        or np.any(development & benchmark)
    ):
        raise ValueError(
            "Judge B-space training, calibration, development, and benchmark masks must be disjoint"
        )
    if benchmark.any() and document_ood_labels is None:
        raise ValueError("Benchmark evaluation requires explicit document_ood_labels")
    requested = tuple(normalize_openood_method(detector) for detector in cfg.detectors)
    supported = {"vim", "full_vim", "rmd", "knn", *OPENOOD_POSTHOC_METHODS}
    unsupported = sorted(set(requested) - supported)
    if unsupported:
        raise ValueError(f"Unsupported Judge B-space detector(s): {unsupported}")
    missing_main = sorted({"vim", "rmd", "knn"} - set(requested))
    if missing_main:
        raise ValueError(
            "The final Judge B-space protocol requires ViM, RMD, and kNN candidates; "
            f"missing {missing_main}"
        )

    evaluation_mask = calibration | development | benchmark
    rows: list[dict[str, Any]] = []
    vim_scorers: dict[int, ViMScorer] = {}
    vim_source_mean: np.ndarray | None = None
    vim_right_singular_vectors: np.ndarray | None = None
    vim_singular_values: np.ndarray | None = None
    if "vim" in requested or "full_vim" in requested:
        source_centered = h[train] - h[train].mean(axis=0)
        _, vim_singular_values, vim_right_singular_vectors = np.linalg.svd(
            source_centered,
            full_matrices=False,
        )
        vim_source_mean = h[train].mean(axis=0)
    for detector in requested:
        if detector == "vim":
            if (
                vim_source_mean is None
                or vim_right_singular_vectors is None
                or vim_singular_values is None
            ):
                raise RuntimeError("ViM source decomposition was not initialized")
            for rank, rank_policy, explained_variance in _vim_candidate_ranks(
                h[train],
                cfg,
                singular_values=vim_singular_values,
            ):
                scorer = ViMScorer(rank=int(rank)).fit_from_svd(
                    h[train],
                    source_mean=vim_source_mean,
                    right_singular_vectors=vim_right_singular_vectors,
                )
                vim_scorers[int(rank)] = scorer
                scores = scorer.score(h[evaluation_mask])
                rows.append(
                    _candidate_row(
                        detector="vim",
                        rank=int(rank),
                        rank_policy=rank_policy,
                        rank_explained_variance=explained_variance,
                        scores=scores,
                        calibration_mask=calibration[evaluation_mask],
                        development_mask=development[evaluation_mask],
                        benchmark_mask=benchmark[evaluation_mask],
                        document_ood_labels=ood_truth[evaluation_mask],
                        document_ids=documents[evaluation_mask],
                        shift_types=shifts[evaluation_mask],
                        fit_rows=int(train.sum()),
                    )
                )
        elif detector == "full_vim":
            if head_weight is None or head_bias is None:
                raise ValueError("Full ViM requires the deployed Judge's exact affine head")
            for rank, rank_policy, explained_variance in _vim_candidate_ranks(
                h[train], cfg, singular_values=vim_singular_values
            ):
                scorer = FullViMScorer(rank=int(rank)).fit(
                    h[train],
                    raw_logits[train],
                    head_weight=head_weight,
                    head_bias=head_bias,
                    query_ids=queries[train],
                    head_query_ids=head_query_ids,
                )
                scores = scorer.score(
                    h[evaluation_mask], raw_logits[evaluation_mask], queries[evaluation_mask]
                )
                rows.append(
                    _candidate_row(
                        detector="full_vim",
                        rank=int(rank),
                        rank_policy=rank_policy,
                        rank_explained_variance=explained_variance,
                        scores=scores,
                        calibration_mask=calibration[evaluation_mask],
                        development_mask=development[evaluation_mask],
                        benchmark_mask=benchmark[evaluation_mask],
                        document_ood_labels=ood_truth[evaluation_mask],
                        document_ids=documents[evaluation_mask],
                        shift_types=shifts[evaluation_mask],
                        fit_rows=int(train.sum()),
                    )
                )
        elif detector == "rmd":
            scorer = RMDScorer(regularization=float(cfg.rmd_regularization)).fit(h[train], target[train])
            scores = scorer.score(h[evaluation_mask])
            rows.append(
                _candidate_row(
                    detector="rmd",
                    rank=None,
                    scores=scores,
                    calibration_mask=calibration[evaluation_mask],
                    development_mask=development[evaluation_mask],
                    benchmark_mask=benchmark[evaluation_mask],
                    document_ood_labels=ood_truth[evaluation_mask],
                    document_ids=documents[evaluation_mask],
                    shift_types=shifts[evaluation_mask],
                    fit_rows=int(train.sum()),
                )
            )
        elif detector == "knn":
            knn_candidates = _knn_candidate_ks(int(train.sum()), cfg)
            scorer = KNNScorer(
                k=max(int(k) for k, _ in knn_candidates),
                metric=str(cfg.knn_metric),
                normalize=bool(cfg.knn_normalize),
            ).fit(h[train])
            scores_by_k = scorer.score_at_ks(
                h[evaluation_mask],
                tuple(int(k) for k, _ in knn_candidates),
            )
            for k, k_policy in knn_candidates:
                rows.append(
                    _candidate_row(
                        detector="knn",
                        rank=None,
                        k=int(k),
                        k_policy=k_policy,
                        scores=scores_by_k[int(k)],
                        calibration_mask=calibration[evaluation_mask],
                        development_mask=development[evaluation_mask],
                        benchmark_mask=benchmark[evaluation_mask],
                        document_ood_labels=ood_truth[evaluation_mask],
                        document_ids=documents[evaluation_mask],
                        shift_types=shifts[evaluation_mask],
                        fit_rows=int(train.sum()),
                    )
                )
        elif detector in OPENOOD_POSTHOC_METHODS:
            react_quantiles: tuple[float | None, ...] = (
                tuple(float(value) for value in cfg.react_quantiles)
                if detector == "react"
                else (None,)
            )
            for react_quantile in react_quantiles:
                scorer = _openood_scorer(
                    detector,
                    cfg,
                    react_quantile=react_quantile,
                ).fit(
                    h[train],
                    raw_logits[train],
                    target[train],
                    queries[train],
                    head_weight=head_weight,
                    head_bias=head_bias,
                    head_query_ids=head_query_ids,
                )
                scores = scorer.score(
                    h[evaluation_mask],
                    raw_logits[evaluation_mask],
                    queries[evaluation_mask],
                )
                rows.append(
                    _candidate_row(
                        detector=detector,
                        rank=None,
                        react_quantile=react_quantile,
                        scores=scores,
                        calibration_mask=calibration[evaluation_mask],
                        development_mask=development[evaluation_mask],
                        benchmark_mask=benchmark[evaluation_mask],
                        document_ood_labels=ood_truth[evaluation_mask],
                        document_ids=documents[evaluation_mask],
                        shift_types=shifts[evaluation_mask],
                        fit_rows=int(train.sum()),
                    )
                )
    if not rows:
        raise ValueError("Judge B-space selection requires at least one detector candidate")
    selected_index, selection_decision = _select_protocol_candidate(
        rows=rows,
    )
    selected = dict(rows[selected_index])
    scorer = (
        vim_scorers[int(selected["rank"])]
        if selected["detector"] == "vim"
        else _fit_selected(
            selected=selected,
            h=h[train],
            logits=raw_logits[train],
            labels=target[train],
            query_ids=queries[train],
            head_weight=head_weight,
            head_bias=head_bias,
            head_query_ids=head_query_ids,
            config=cfg,
        )
    )
    if isinstance(scorer, ViMScorer):
        thresholds = scorer.calibrate(
            h[calibration],
            soft_q=cfg.soft_quantile,
            hard_q=cfg.hard_quantile,
        )
        scores = scorer.score(h)
    elif isinstance(scorer, OpenOODPosthocScorer):
        thresholds = scorer.calibrate(
            h[calibration],
            raw_logits[calibration],
            queries[calibration],
            soft_q=cfg.soft_quantile,
            hard_q=cfg.hard_quantile,
        )
        scores = scorer.score(h, raw_logits, queries)
    else:
        thresholds = scorer.calibrate(h[calibration], soft_q=cfg.soft_quantile, hard_q=cfg.hard_quantile)
        scores = scorer.score(h)
    return JudgeOODSelectionResult(
        scorer=scorer,
        scores=scores,
        score_labels=scorer.labels(scores),
        thresholds=thresholds,
        selected_candidate=selected,
        candidate_results=rows,
        selection_decision=selection_decision,
        development_metrics=_document_level_ood_metrics(
            development[calibration | development],
            _selected_scores(
                scorer,
                h[calibration | development],
                raw_logits[calibration | development],
                queries[calibration | development],
            ),
            documents[calibration | development],
        ),
        config=cfg,
        class_values=classes.copy(),
    )


def refit_selected_judge_ood_detector(
    *,
    selected_candidate: dict[str, Any],
    penultimate: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    class_values: np.ndarray,
    reference_mask: np.ndarray,
    calibration_mask: np.ndarray,
    config: JudgeOODSelectionConfig,
    query_ids: np.ndarray | None = None,
    head_weight: np.ndarray | None = None,
    head_bias: np.ndarray | None = None,
    head_query_ids: np.ndarray | None = None,
) -> JudgeOODSelectionResult:
    """Refit a fixed B-space detector after an accepted Judge-head update."""

    h = np.asarray(penultimate, dtype=np.float64)
    raw_logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(labels)
    reference = np.asarray(reference_mask, dtype=bool)
    calibration = np.asarray(calibration_mask, dtype=bool)
    queries = (
        np.asarray(query_ids).astype(str)
        if query_ids is not None
        else np.asarray(["__global__"] * len(h), dtype=str)
    )
    if h.ndim != 2 or raw_logits.ndim != 2 or len(h) != len(raw_logits):
        raise ValueError("B-space refresh requires aligned penultimate features and logits")
    if not (len(h) == len(target) == len(reference) == len(calibration) == len(queries)):
        raise ValueError("B-space refresh inputs must align")
    if not reference.any() or not calibration.any() or np.any(reference & calibration):
        raise ValueError("B-space refresh needs non-empty, disjoint reference and calibration rows")
    selected = dict(selected_candidate)
    scorer = _fit_selected(
        selected=selected,
        h=h[reference],
        logits=raw_logits[reference],
        labels=target[reference],
        query_ids=queries[reference],
        head_weight=head_weight,
        head_bias=head_bias,
        head_query_ids=head_query_ids,
        config=config,
    )
    if isinstance(scorer, ViMScorer):
        thresholds = scorer.calibrate(
            h[calibration],
            soft_q=float(config.soft_quantile),
            hard_q=float(config.hard_quantile),
        )
    elif isinstance(scorer, OpenOODPosthocScorer):
        thresholds = scorer.calibrate(
            h[calibration],
            raw_logits[calibration],
            queries[calibration],
            soft_q=float(config.soft_quantile),
            hard_q=float(config.hard_quantile),
        )
    else:
        thresholds = scorer.calibrate(
            h[calibration],
            soft_q=float(config.soft_quantile),
            hard_q=float(config.hard_quantile),
        )
    scores = _selected_scores(scorer, h, raw_logits, queries)
    return JudgeOODSelectionResult(
        scorer=scorer,
        scores=scores,
        score_labels=scorer.labels(scores),
        thresholds=thresholds,
        selected_candidate=selected,
        candidate_results=[selected],
        selection_decision={
            "policy": "fixed_pre_update_candidate",
            "selected_detector": str(selected.get("detector")),
            "selection_repeated": False,
        },
        development_metrics={},
        config=config,
        class_values=np.asarray(class_values).copy(),
        selection_scope="fixed_pre_update_candidate",
        reference_scope="training_train_plus_accepted_adapt",
    )


def _fit_selected(
    *,
    selected: dict[str, Any],
    h: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    query_ids: np.ndarray,
    head_weight: np.ndarray | None,
    head_bias: np.ndarray | None,
    head_query_ids: np.ndarray | None,
    config: JudgeOODSelectionConfig,
) -> JudgeBehaviorDetector:
    if selected["detector"] == "vim":
        return ViMScorer(rank=int(selected["rank"])).fit(h)
    if selected["detector"] == "rmd":
        return RMDScorer(regularization=float(config.rmd_regularization)).fit(h, labels)
    if selected["detector"] == "knn":
        return KNNScorer(
            k=int(selected["k"]),
            metric=str(config.knn_metric),
            normalize=bool(config.knn_normalize),
        ).fit(h)
    if selected["detector"] in OPENOOD_POSTHOC_METHODS:
        return _openood_scorer(
            str(selected["detector"]),
            config,
            react_quantile=(
                float(selected["react_quantile"])
                if selected.get("react_quantile") is not None
                else None
            ),
        ).fit(
            h,
            logits,
            labels,
            query_ids,
            head_weight=head_weight,
            head_bias=head_bias,
            head_query_ids=head_query_ids,
        )
    raise ValueError(f"Unsupported selected Judge B-space detector: {selected['detector']!r}")


def _selected_scores(
    scorer: JudgeBehaviorDetector,
    h: np.ndarray,
    logits: np.ndarray,
    query_ids: np.ndarray,
) -> np.ndarray:
    if isinstance(scorer, (OpenOODPosthocScorer, FullViMScorer)):
        if isinstance(scorer, FullViMScorer):
            return scorer.score(h, logits)
        return scorer.score(h, logits, query_ids)
    return scorer.score(h)


def _openood_scorer(
    method: str,
    config: JudgeOODSelectionConfig,
    *,
    react_quantile: float | None = None,
) -> OpenOODPosthocScorer:
    return OpenOODPosthocScorer(
        method=method,
        regularization=float(config.rmd_regularization),
        temperature=float(config.openood_temperature),
        odin_temperature=float(config.odin_temperature),
        odin_epsilon=float(config.odin_epsilon),
        react_quantile=(
            float(react_quantile)
            if react_quantile is not None
            else float(config.react_quantiles[0])
        ),
        dice_sparsity=float(config.dice_sparsity),
        ash_percentile=float(config.ash_percentile),
        gen_gamma=float(config.gen_gamma),
    )


def _candidate_row(
    *,
    detector: str,
    rank: int | None,
    rank_policy: str | None = None,
    rank_explained_variance: float | None = None,
    k: int | None = None,
    k_policy: str | None = None,
    react_quantile: float | None = None,
    scores: np.ndarray,
    calibration_mask: np.ndarray,
    development_mask: np.ndarray,
    benchmark_mask: np.ndarray | None = None,
    document_ood_labels: np.ndarray | None = None,
    document_ids: np.ndarray,
    shift_types: np.ndarray,
    fit_rows: int,
) -> dict[str, Any]:
    calibration = np.asarray(calibration_mask, dtype=bool)
    development = np.asarray(development_mask, dtype=bool)
    benchmark = (
        np.zeros(len(calibration), dtype=bool)
        if benchmark_mask is None
        else np.asarray(benchmark_mask, dtype=bool)
    )
    ood_truth = (
        benchmark.copy()
        if document_ood_labels is None
        else np.asarray(document_ood_labels, dtype=bool)
    )
    if len(scores) != len(calibration) or len(scores) != len(development) or len(scores) != len(benchmark):
        raise ValueError("Judge detector candidate masks and scores must align")
    development_subset = calibration | development
    metrics = _document_level_ood_metrics(
        development[development_subset],
        np.asarray(scores)[development_subset],
        np.asarray(document_ids)[development_subset],
    )
    shifts = np.asarray(shift_types).astype(str)
    by_shift: dict[str, dict[str, float]] = {}
    for shift in ("near", "far"):
        positives = development & (shifts == shift)
        if not positives.any():
            continue
        subset = calibration | positives
        by_shift[shift] = _document_level_ood_metrics(
            positives[subset],
            np.asarray(scores)[subset],
            np.asarray(document_ids)[subset],
        )
    benchmark_metrics = None
    benchmark_by_shift: dict[str, dict[str, float]] = {}
    if benchmark.any():
        benchmark_metrics = _document_level_ood_metrics(
            ood_truth[benchmark],
            np.asarray(scores)[benchmark],
            np.asarray(document_ids)[benchmark],
        )
        for shift in ("near", "far"):
            target = benchmark & ood_truth & (shifts == shift)
            if not target.any():
                continue
            subset = benchmark & ((~ood_truth) | (shifts == shift))
            benchmark_by_shift[shift] = _document_level_ood_metrics(
                ood_truth[subset],
                np.asarray(scores)[subset],
                np.asarray(document_ids)[subset],
            )
    return {
        "detector": detector,
        "rank": rank,
        "rank_policy": rank_policy,
        "rank_explained_variance": rank_explained_variance,
        "k": k,
        "k_policy": k_policy,
        "react_quantile": react_quantile,
        "development_auroc": float(metrics["auroc"]),
        "development_aupr": float(metrics["aupr"]),
        "development_fpr95": float(metrics["fpr95"]),
        "development_by_shift": by_shift,
        "benchmark_test_auroc": (
            float(benchmark_metrics["auroc"]) if benchmark_metrics is not None else None
        ),
        "benchmark_test_aupr": (
            float(benchmark_metrics["aupr"]) if benchmark_metrics is not None else None
        ),
        "benchmark_test_fpr95": (
            float(benchmark_metrics["fpr95"]) if benchmark_metrics is not None else None
        ),
        "benchmark_test_by_shift": benchmark_by_shift,
        "fit_rows": int(fit_rows),
        "calibration_records": int(calibration_mask.sum()),
        "development_records": int(development_mask.sum()),
        "benchmark_test_records": int(benchmark.sum()),
        "benchmark_test_evidence_level": (
            "independent_confirmation" if benchmark.any() else "unavailable"
        ),
        "feature_scope": "judge_behavior",
        "detection_unit": "judge_record",
        "selection_metric_unit": "input_document",
        "selection_used_deployment_records": False,
        "score_variant": (
            "residual_only"
            if detector == "vim"
            else "virtual_logit_softmax"
            if detector == "full_vim"
            else None
        ),
    }


def _vim_candidate_ranks(
    source_penultimate: np.ndarray,
    config: JudgeOODSelectionConfig,
    *,
    singular_values: np.ndarray | None = None,
) -> list[tuple[int, str, float | None]]:
    values = np.asarray(source_penultimate, dtype=np.float64)
    max_rank = min(values.shape[0] - 1, values.shape[1]) - 1
    if max_rank < 1:
        raise ValueError("ViM needs at least two non-residual source dimensions")
    candidates: list[tuple[int, str, float | None]] = []
    if bool(config.vim_include_variance_rank):
        spectrum = (
            np.asarray(singular_values, dtype=np.float64)
            if singular_values is not None
            else np.linalg.svd(values - values.mean(axis=0), compute_uv=False)
        )
        explained = spectrum**2
        total = float(explained.sum())
        if total <= 0.0:
            raise ValueError("ViM source features have zero total PCA variance")
        cumulative = np.cumsum(explained) / total
        target_rank = int(np.searchsorted(cumulative, float(config.vim_variance_target), side="left") + 1)
        rank = min(target_rank, int(config.vim_rank_cap), int(max_rank))
        candidates.append((rank, "source_pca_variance_target", float(cumulative[rank - 1])))
    for configured_rank in config.vim_ranks:
        rank = int(configured_rank)
        if rank > int(max_rank):
            continue
        candidates.append((rank, "configured_ablation_grid", None))
    deduplicated: list[tuple[int, str, float | None]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate[0] not in seen:
            deduplicated.append(candidate)
            seen.add(candidate[0])
    if not deduplicated:
        raise ValueError("ViM selection requires an automatic or configured rank candidate")
    return deduplicated


def _knn_candidate_ks(
    source_rows: int,
    config: JudgeOODSelectionConfig,
) -> list[tuple[int, str]]:
    if int(source_rows) < 2:
        raise ValueError("kNN selection needs at least two source rows")
    candidates: list[tuple[int, str]] = []
    if bool(config.knn_include_sqrt_k):
        candidates.append((max(1, int(np.floor(np.sqrt(source_rows)))), "floor_sqrt_source_rows"))
    for configured_k in config.knn_ks:
        k = int(configured_k)
        if k > int(source_rows):
            raise ValueError(
                f"Configured kNN k={k} exceeds the training Judge feature bank size "
                f"({source_rows})"
            )
        candidates.append((k, "configured_ablation_grid"))
    deduplicated: list[tuple[int, str]] = []
    seen: set[int] = set()
    for candidate in candidates:
        if candidate[0] not in seen:
            deduplicated.append(candidate)
            seen.add(candidate[0])
    if not deduplicated:
        raise ValueError("kNN selection requires an automatic or configured k candidate")
    return deduplicated


def _candidate_sort_key(
    row: dict[str, Any],
) -> tuple[float, float, float, str, int, int, float]:
    return (
        -float(row["development_auroc"]),
        -float(row["development_aupr"]),
        float(row["development_fpr95"]),
        str(row["detector"]),
        int(row["rank"]) if row["rank"] is not None else -1,
        int(row.get("k")) if row.get("k") is not None else -1,
        float(row.get("react_quantile") or -1.0),
    )


def _select_protocol_candidate(
    *,
    rows: list[dict[str, Any]],
) -> tuple[int, dict[str, Any]]:
    best_indices: dict[str, int] = {}
    for index, row in enumerate(rows):
        detector = str(row["detector"])
        current = best_indices.get(detector)
        if current is None or _candidate_sort_key(row) < _candidate_sort_key(rows[current]):
            best_indices[detector] = int(index)
    if "vim" not in best_indices:
        raise ValueError("Fixed ViM deployment requires at least one ViM candidate")

    selected_index = best_indices["vim"]
    return selected_index, {
        "policy": "fixed_vim_residual_primary",
        "selection_unit": "input_document",
        "vim_score_variant": "residual_only",
        "vim_variant_selected_by": ["development_auroc", "development_aupr", "development_fpr95"],
        "paired_bootstrap_performed": False,
        "paired_bootstrap_reason": "deployment_detector_is_fixed_to_vim",
        "best_candidate_by_detector": {
            detector: dict(rows[index]) for detector, index in sorted(best_indices.items())
        },
        "selected_vim_candidate": dict(rows[selected_index]),
        "selected_detector": str(rows[selected_index]["detector"]),
        "non_vim_detectors_are_baselines_only": True,
    }


def _document_level_ood_metrics(
    development_mask: np.ndarray,
    scores: np.ndarray,
    document_ids: np.ndarray,
) -> dict[str, float]:
    labels, values, _ = _document_level_binary_scores(
        development_mask,
        scores,
        document_ids,
    )
    return {key: float(value) for key, value in ood_metrics(labels, values).items()}


def _document_level_binary_scores(
    development_mask: np.ndarray,
    scores: np.ndarray,
    document_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(development_mask, dtype=bool)
    values = np.asarray(scores, dtype=np.float64)
    documents = np.asarray(document_ids).astype(str)
    if labels.shape != values.shape or labels.shape != documents.shape:
        raise ValueError("Document-level OOD metric inputs must align")
    unique_documents, inverse, counts = np.unique(
        documents,
        return_inverse=True,
        return_counts=True,
    )
    label_counts = np.bincount(inverse, weights=labels.astype(np.int64))
    inconsistent = (label_counts != 0) & (label_counts != counts)
    if inconsistent.any():
        document = str(unique_documents[int(np.flatnonzero(inconsistent)[0])])
        raise ValueError(f"Document {document!r} spans ID and OOD selection roles")
    score_sums = np.bincount(inverse, weights=values)
    return (
        (label_counts > 0).astype(int),
        np.asarray(score_sums / counts, dtype=np.float64),
        unique_documents,
    )


def _baseline_comparison(
    rows: list[dict[str, Any]],
    selected: dict[str, Any],
) -> dict[str, Any]:
    best_by_detector: dict[str, dict[str, Any]] = {}
    for row in rows:
        detector = str(row.get("detector"))
        current = best_by_detector.get(detector)
        if current is None or _candidate_sort_key(row) < _candidate_sort_key(current):
            best_by_detector[detector] = dict(row)
    return {
        "protocol": "training_calibration_id_vs_development_ood_documents",
        "selection_metric_unit": "input_document",
        "metrics": ["auroc", "aupr", "fpr95"],
        "selected_detector": str(selected.get("detector")),
        "best_by_detector": best_by_detector,
    }
