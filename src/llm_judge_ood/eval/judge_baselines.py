"""Matched-split baselines for the ASAP ordinal Judge experiment."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import nltk
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from src.llm_judge_ood.shared.metrics import (
    bootstrap_judge_metric_interval,
    judge_metrics,
    macro_query_judge_metrics,
)
from src.llm_judge_ood.data.asap_prompting import ASAP_PROMPT_CATALOG, build_asap_judge_input


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|@[A-Za-z0-9]+")
_STEMMER = PorterStemmer()
_ASAP_CLASS_VALUES = np.arange(1, 6, dtype=int)

# Prompt text is absent from training_set_rel3.tsv.  The frozen catalog is
# shared with the Qwen Judge template; EASE only consumes ``task`` for its
# documented prompt-overlap feature and never uses labels to construct it.
ASAP_SOURCE_PROMPT_DESCRIPTIONS = {
    prompt_id: str(ASAP_PROMPT_CATALOG[prompt_id]["task"])
    for prompt_id in (1, 2)
}


@dataclass(frozen=True)
class EaseBaselineConfig:
    useful_ngrams_per_view: int = 201
    minimum_ngram_document_frequency: int = 2
    minimum_good_pos_ngram_frequency: int = 3
    svr_cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    svr_epsilons: tuple[float, ...] = (0.0, 0.1, 0.25)
    seed: int = 42
    bootstrap_samples: int = 1000

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_asap_judge_baselines(
    records: Sequence[Mapping[str, Any]],
    *,
    current_judge_predictions: Mapping[str, int] | None = None,
    external_predictions: Sequence[Mapping[str, Any]] = (),
    config: EaseBaselineConfig | None = None,
) -> dict[str, Any]:
    """Run source-only Judge baselines on the fixed ASAP split contract."""

    cfg = config or EaseBaselineConfig()
    source = [
        dict(row)
        for row in records
        if str(row.get("document_distribution_role")) == "training"
        and str(row.get("split"))
        in {"training_train", "training_validation", "training_test"}
    ]
    prompts = sorted({int(row["asap_prompt_id"]) for row in source})
    if prompts != [1, 2]:
        raise ValueError(f"ASAP source Judge baselines require prompts 1 and 2, got {prompts}")
    for split in ("training_train", "training_validation", "training_test"):
        for prompt_id in prompts:
            if not any(
                str(row["split"]) == split and int(row["asap_prompt_id"]) == prompt_id
                for row in source
            ):
                raise ValueError(f"Missing {split} rows for ASAP prompt {prompt_id}")

    ease_predictions: dict[str, dict[str, int]] = {"ease_svr": {}, "ease_blrr": {}}
    ease_details: dict[str, Any] = {}
    for prompt_id in prompts:
        prompt_rows = [row for row in source if int(row["asap_prompt_id"]) == prompt_id]
        result = _run_prompt_ease(
            prompt_rows,
            prompt_id=prompt_id,
            prompt_text=ASAP_SOURCE_PROMPT_DESCRIPTIONS[prompt_id],
            config=cfg,
        )
        ease_details[str(prompt_id)] = result["details"]
        for method in ease_predictions:
            ease_predictions[method].update(result["predictions"][method])

    methods: dict[str, Any] = {}
    test_rows = [row for row in source if str(row["split"]) == "training_test"]
    for method, predictions in ease_predictions.items():
        methods[method] = _evaluate_predictions(
            test_rows,
            predictions,
            bootstrap_samples=cfg.bootstrap_samples,
        )
    if current_judge_predictions is not None:
        methods["qwen_frozen_linear_judge"] = _evaluate_predictions(
            test_rows,
            current_judge_predictions,
            bootstrap_samples=cfg.bootstrap_samples,
        )
    methods.update(
        evaluate_external_score_predictions(
            test_rows,
            external_predictions,
            bootstrap_samples=cfg.bootstrap_samples,
        )
    )
    human = evaluate_inter_human_ceiling(test_rows, bootstrap_samples=cfg.bootstrap_samples)
    return {
        "artifact_type": "llm_judge_ood_asap_judge_baselines",
        "status": "complete_local_external_pending" if not external_predictions else "complete_with_imported_external",
        "protocol": {
            "dataset": "ASAP-AES",
            "source_prompts": prompts,
            "train_split": "training_train",
            "selection_split": "training_validation",
            "report_split": "training_test",
            "test_documents": len(test_rows),
            "test_labels_used_for_selection": False,
            "metric_policy": "pooled and macro-per-prompt QWK/Spearman/MAE",
            "human_ceiling_policy": "raw rater1/rater2 scores, macro per prompt",
        },
        "ease_reproduction": {
            "source": "Phandi et al. 2015 / EASE feature categories",
            "clean_room": True,
            "matched_split": True,
            "feature_contract": [
                "six length features",
                "source-high-score POS-ngram anomaly count and proportion",
                "direct prompt-token overlap count and proportion",
                "Fisher-selected unstemmed and Porter-stemmed unigram/bigram counts",
            ],
            "documented_deviations": [
                "ASAP prompt text is absent from training_set_rel3.tsv; compact public task descriptions are used for direct overlap",
                "WordNet synonym overlap is omitted",
                "aspell correction is omitted before the stemmed n-gram view",
                "the external EASE grammar corpus is replaced by source-only high-score POS n-grams",
                "results are matched to this repository's fixed split and are not the published 5-fold EASE numbers",
            ],
            "config": cfg.to_dict(),
            "svr_estimator": "sklearn.svm.SVR(kernel='linear')",
            "by_prompt": ease_details,
        },
        "methods": methods,
        "inter_human": human,
    }


def evaluate_inter_human_ceiling(
    test_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Compute a matched-split ceiling without mixing prompt score scales."""

    by_prompt: dict[str, Any] = {}
    qwk_values: list[float] = []
    for prompt_id in sorted({int(row["asap_prompt_id"]) for row in test_rows}):
        rows = [row for row in test_rows if int(row["asap_prompt_id"]) == prompt_id]
        rater1 = np.asarray([float(row["raw_rater1_score"]) for row in rows])
        rater2 = np.asarray([float(row["raw_rater2_score"]) for row in rows])
        raw_rater_class_values = np.arange(1, 7, dtype=int)
        document_groups = np.asarray([str(row["input_document_id"]) for row in rows])
        metrics = judge_metrics(
            rater1,
            rater2,
            class_values=raw_rater_class_values,
        )
        observed, low, high = bootstrap_judge_metric_interval(
            rater1,
            rater2,
            metric_name="qwk",
            class_values=raw_rater_class_values,
            groups=document_groups,
            rng=np.random.default_rng(4200 + prompt_id),
            n_boot=bootstrap_samples,
        )
        by_prompt[str(prompt_id)] = {
            "documents": len(rows),
            "raw_score_scale_rater1": [float(rater1.min()), float(rater1.max())],
            "raw_score_scale_rater2": [float(rater2.min()), float(rater2.max())],
            **metrics,
            "qwk_ci95": [low, high],
            "qwk_bootstrap_unit": "input_document_id",
            "qwk_bootstrap_samples": int(bootstrap_samples),
        }
        qwk_values.append(observed)
    mapped_unique = {
        str(prompt_id): {
            "rater1": sorted(
                {int(row["rater1_score"]) for row in test_rows if int(row["asap_prompt_id"]) == prompt_id}
            ),
            "rater2": sorted(
                {int(row["rater2_score"]) for row in test_rows if int(row["asap_prompt_id"]) == prompt_id}
            ),
        }
        for prompt_id in sorted({int(row["asap_prompt_id"]) for row in test_rows})
    }
    return {
        "documents": len(test_rows),
        "macro_prompt_qwk": float(np.mean(qwk_values)),
        "by_prompt": by_prompt,
        "mapped_rater_score_diagnostic": {
            "formal_ceiling_uses_mapped_scores": False,
            "reason": "resolved and individual-rater raw scales differ for some ASAP prompts",
            "unique_values": mapped_unique,
        },
    }


def build_external_score_manifest(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return label-free held-out test inputs for single-answer LLM judges."""

    return [
        {
            "sample_id": str(row["sample_id"]),
            "dataset": "asap_aes",
            "split": str(row["split"]),
            "asap_prompt_id": int(row["asap_prompt_id"]),
            "task": ASAP_SOURCE_PROMPT_DESCRIPTIONS[int(row["asap_prompt_id"])],
            "judge_input": build_asap_judge_input(
                prompt_id=int(row["asap_prompt_id"]),
                essay_text=str(row["input_document_text"]),
            ),
            "essay": str(row["input_document_text"]),
            "output_contract": {"score": "integer from 1 through 5"},
        }
        for row in records
        if str(row.get("document_distribution_role")) == "training"
        and str(row.get("split")) == "training_test"
        and int(row.get("asap_prompt_id", -1)) in ASAP_SOURCE_PROMPT_DESCRIPTIONS
    ]


def build_pandalm_pairwise_manifests(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair nearly every source-test essay once for PandaLM preference scoring."""

    rng = np.random.default_rng(seed)
    manifest: list[dict[str, Any]] = []
    truth: list[dict[str, Any]] = []
    for prompt_id in sorted(ASAP_SOURCE_PROMPT_DESCRIPTIONS):
        rows = [
            dict(row)
            for row in records
            if str(row.get("split")) == "training_test"
            and int(row.get("asap_prompt_id", -1)) == prompt_id
        ]
        rows.sort(key=lambda row: (int(row["label"]), str(row["sample_id"])))
        low, high = 0, len(rows) - 1
        pair_index = 0
        while low < high:
            left, right = rows[low], rows[high]
            low += 1
            high -= 1
            if int(left["label"]) == int(right["label"]):
                continue
            if bool(rng.integers(0, 2)):
                response1, response2 = right, left
                expected = 1
            else:
                response1, response2 = left, right
                expected = 2
            pair_id = f"asap-p{prompt_id}-pair-{pair_index:04d}"
            pair_index += 1
            manifest.append(
                {
                    "pair_id": pair_id,
                    "asap_prompt_id": prompt_id,
                    "instruction": "Choose the higher-quality student essay for the assignment.",
                    "input": ASAP_SOURCE_PROMPT_DESCRIPTIONS[prompt_id],
                    "response1_sample_id": str(response1["sample_id"]),
                    "response2_sample_id": str(response2["sample_id"]),
                    "response1": str(response1["input_document_text"]),
                    "response2": str(response2["input_document_text"]),
                    "output_contract": "1, 2, or Tie",
                }
            )
            truth.append(
                {
                    "pair_id": pair_id,
                    "asap_prompt_id": prompt_id,
                    "expected_preference": expected,
                    "score_gap": abs(int(response1["label"]) - int(response2["label"])),
                }
            )
    return manifest, truth


def evaluate_external_score_predictions(
    test_rows: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    expected_ids = {str(row["sample_id"]) for row in test_rows}
    by_model: dict[str, dict[str, int]] = {}
    for row in predictions:
        model = str(row.get("model") or row.get("method") or "external")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or "score" not in row:
            continue
        score = int(row["score"])
        if score not in {1, 2, 3, 4, 5}:
            raise ValueError(f"External prediction {model}/{sample_id} has score {score}, expected 1-5")
        model_rows = by_model.setdefault(model, {})
        if sample_id in model_rows:
            raise ValueError(f"Duplicate external prediction for {model}/{sample_id}")
        model_rows[sample_id] = score
    results: dict[str, Any] = {}
    for model, values in sorted(by_model.items()):
        covered = expected_ids & set(values)
        if covered != expected_ids:
            results[f"external::{model}"] = {
                "status": "incomplete",
                "coverage": len(covered),
                "expected": len(expected_ids),
                "missing_sample_ids": sorted(expected_ids - covered)[:20],
            }
            continue
        results[f"external::{model}"] = _evaluate_predictions(
            test_rows,
            values,
            bootstrap_samples=bootstrap_samples,
        )
    return results


def evaluate_pandalm_predictions(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    truth = {str(row["pair_id"]): int(row["expected_preference"]) for row in truth_rows}
    by_model: dict[str, dict[str, int]] = {}
    for row in prediction_rows:
        model = str(row.get("model") or "PandaLM")
        pair_id = str(row.get("pair_id") or "")
        preference = int(row.get("preference", -1))
        if preference not in {0, 1, 2}:
            raise ValueError(f"Invalid PandaLM preference for {pair_id}: {preference}")
        by_model.setdefault(model, {})[pair_id] = preference
    out: dict[str, Any] = {}
    for model, values in sorted(by_model.items()):
        covered = set(truth) & set(values)
        correct = sum(values[pair_id] == truth[pair_id] for pair_id in covered)
        out[model] = {
            "status": "complete" if covered == set(truth) else "incomplete",
            "coverage": len(covered),
            "expected": len(truth),
            "pairwise_accuracy": float(correct / len(covered)) if covered else float("nan"),
            "tie_prediction_rate": (
                float(sum(values[pair_id] == 0 for pair_id in covered) / len(covered))
                if covered
                else float("nan")
            ),
        }
    return out


class _EaseFeatureExtractor:
    def __init__(self, config: EaseBaselineConfig, prompt_text: str) -> None:
        self.config = config
        self.prompt_tokens = set(_tokens(prompt_text))
        self.normal_vectorizer: CountVectorizer | None = None
        self.stem_vectorizer: CountVectorizer | None = None
        self.normal_indices: np.ndarray | None = None
        self.stem_indices: np.ndarray | None = None
        self.good_pos_ngrams: set[str] = set()
        self.scaler = MinMaxScaler()

    def fit(self, texts: Sequence[str], labels: np.ndarray) -> "_EaseFeatureExtractor":
        self.normal_vectorizer, self.normal_indices = _fit_useful_ngram_view(
            texts,
            labels,
            analyzer=_word_ngram_analyzer,
            config=self.config,
        )
        self.stem_vectorizer, self.stem_indices = _fit_useful_ngram_view(
            texts,
            labels,
            analyzer=_stem_ngram_analyzer,
            config=self.config,
        )
        tagged = _pos_tags(texts)
        good_mask = labels.astype(float) >= float(np.mean(labels.astype(float)))
        counts: Counter[str] = Counter()
        for tags, is_good in zip(tagged, good_mask, strict=True):
            if is_good:
                counts.update(_ngrams(tags, 2, 4))
        self.good_pos_ngrams = {
            value
            for value, count in counts.items()
            if count >= self.config.minimum_good_pos_ngram_frequency
        }
        self.scaler.fit(self._handcrafted(texts, tagged))
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if (
            self.normal_vectorizer is None
            or self.stem_vectorizer is None
            or self.normal_indices is None
            or self.stem_indices is None
        ):
            raise RuntimeError("EASE feature extractor is not fitted")
        tagged = _pos_tags(texts)
        handcrafted = self.scaler.transform(self._handcrafted(texts, tagged))
        normal = self.normal_vectorizer.transform(texts)[:, self.normal_indices]
        stemmed = self.stem_vectorizer.transform(texts)[:, self.stem_indices]
        bag = np.log1p(np.concatenate([normal.toarray(), stemmed.toarray()], axis=1))
        return np.concatenate([handcrafted, bag], axis=1).astype(np.float64)

    def _handcrafted(self, texts: Sequence[str], tagged: Sequence[Sequence[str]]) -> np.ndarray:
        rows: list[list[float]] = []
        for text, tags in zip(texts, tagged, strict=True):
            tokens = _tokens(text)
            word_count = max(1, len(tokens))
            pos_values = _ngrams(tags, 2, 4)
            bad_pos = sum(value not in self.good_pos_ngrams for value in pos_values)
            overlap = sum(token in self.prompt_tokens for token in tokens)
            rows.append(
                [
                    float(len(text)),
                    float(word_count),
                    float(text.count(",")),
                    float(text.count("'")),
                    float(text.count(".") + text.count("?") + text.count("!")),
                    float(len(text) / word_count),
                    float(bad_pos),
                    float(bad_pos / word_count),
                    float(overlap),
                    float(overlap / word_count),
                ]
            )
        return np.asarray(rows, dtype=np.float64)


def _run_prompt_ease(
    rows: Sequence[Mapping[str, Any]],
    *,
    prompt_id: int,
    prompt_text: str,
    config: EaseBaselineConfig,
) -> dict[str, Any]:
    by_split = {
        split: [dict(row) for row in rows if str(row["split"]) == split]
        for split in ("training_train", "training_validation", "training_test")
    }
    train = by_split["training_train"]
    validation = by_split["training_validation"]
    test = by_split["training_test"]
    train_text = [str(row["input_document_text"]) for row in train]
    validation_text = [str(row["input_document_text"]) for row in validation]
    train_y = np.asarray([int(row["label"]) for row in train])
    validation_y = np.asarray([int(row["label"]) for row in validation])
    extractor = _EaseFeatureExtractor(config, prompt_text).fit(train_text, train_y)
    x_train = extractor.transform(train_text)
    x_validation = extractor.transform(validation_text)
    scaled_train_y = _scale_scores(train_y)

    candidates: list[dict[str, Any]] = []
    for c_value in config.svr_cs:
        for epsilon in config.svr_epsilons:
            model = SVR(
                C=float(c_value),
                epsilon=float(epsilon),
                kernel="linear",
                cache_size=512,
            ).fit(x_train, scaled_train_y)
            predictions = _unscale_scores(model.predict(x_validation))
            metrics = judge_metrics(
                validation_y,
                predictions,
                class_values=_ASAP_CLASS_VALUES,
            )
            candidates.append(
                {
                    "C": float(c_value),
                    "epsilon": float(epsilon),
                    "validation_qwk": float(metrics["qwk"]),
                    "validation_mae": float(metrics["mae"]),
                }
            )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["validation_qwk"]),
            -float(row["validation_mae"]),
            -float(row["C"]),
            -float(row["epsilon"]),
        ),
    )

    combined = train + validation
    combined_text = [str(row["input_document_text"]) for row in combined]
    combined_y = np.asarray([int(row["label"]) for row in combined])
    final_extractor = _EaseFeatureExtractor(config, prompt_text).fit(combined_text, combined_y)
    x_combined = final_extractor.transform(combined_text)
    x_test = final_extractor.transform([str(row["input_document_text"]) for row in test])
    svr = SVR(
        C=float(selected["C"]),
        epsilon=float(selected["epsilon"]),
        kernel="linear",
        cache_size=512,
    ).fit(x_combined, _scale_scores(combined_y))
    blrr = BayesianRidge(
        alpha_1=1e-6,
        alpha_2=1e-6,
        lambda_1=1e-6,
        lambda_2=1e-6,
    ).fit(x_combined, _scale_scores(combined_y))
    svr_predictions = _unscale_scores(svr.predict(x_test))
    blrr_predictions = _unscale_scores(blrr.predict(x_test))
    test_y = np.asarray([int(row["label"]) for row in test])
    return {
        "predictions": {
            "ease_svr": {
                str(row["sample_id"]): int(value)
                for row, value in zip(test, svr_predictions, strict=True)
            },
            "ease_blrr": {
                str(row["sample_id"]): int(value)
                for row, value in zip(test, blrr_predictions, strict=True)
            },
        },
        "details": {
            "prompt_id": prompt_id,
            "documents": {key: len(value) for key, value in by_split.items()},
            "selected_svr": selected,
            "svr_candidates": candidates,
            "final_feature_count": int(x_combined.shape[1]),
            "test_svr": judge_metrics(
                test_y, svr_predictions, class_values=_ASAP_CLASS_VALUES
            ),
            "test_blrr": judge_metrics(
                test_y, blrr_predictions, class_values=_ASAP_CLASS_VALUES
            ),
        },
    }


def _fit_useful_ngram_view(
    texts: Sequence[str],
    labels: np.ndarray,
    *,
    analyzer: Any,
    config: EaseBaselineConfig,
) -> tuple[CountVectorizer, np.ndarray]:
    vectorizer = CountVectorizer(
        analyzer=analyzer,
        lowercase=False,
        min_df=config.minimum_ngram_document_frequency,
    )
    matrix = vectorizer.fit_transform(texts).astype(np.float64)
    good = labels.astype(float) >= float(np.mean(labels.astype(float)))
    if not good.any() or good.all():
        raise ValueError("Fisher n-gram selection requires both high- and low-score source essays")
    score = _fisher_score(matrix, good)
    count = min(config.useful_ngrams_per_view, matrix.shape[1])
    indices = np.argsort(score, kind="stable")[-count:]
    return vectorizer, np.asarray(sorted(indices.tolist()), dtype=np.int64)


def _fisher_score(matrix: Any, positive: np.ndarray) -> np.ndarray:
    pos = matrix[positive]
    neg = matrix[~positive]
    pos_mean = np.asarray(pos.mean(axis=0)).ravel()
    neg_mean = np.asarray(neg.mean(axis=0)).ravel()
    pos_var = np.maximum(np.asarray(pos.power(2).mean(axis=0)).ravel() - pos_mean**2, 0.0)
    neg_var = np.maximum(np.asarray(neg.power(2).mean(axis=0)).ravel() - neg_mean**2, 0.0)
    return (pos_mean - neg_mean) ** 2 / (pos_var / pos.shape[0] + neg_var / neg.shape[0] + 1e-12)


def _evaluate_predictions(
    test_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, int],
    *,
    bootstrap_samples: int,
) -> dict[str, Any]:
    missing = [str(row["sample_id"]) for row in test_rows if str(row["sample_id"]) not in predictions]
    if missing:
        return {
            "status": "incomplete",
            "coverage": len(test_rows) - len(missing),
            "expected": len(test_rows),
            "missing_sample_ids": missing[:20],
        }
    labels = np.asarray([int(row["label"]) for row in test_rows])
    values = np.asarray([int(predictions[str(row["sample_id"])]) for row in test_rows])
    prompt_ids = np.asarray([str(row["asap_prompt_id"]) for row in test_rows])
    document_groups = np.asarray([str(row["input_document_id"]) for row in test_rows])
    pooled = judge_metrics(labels, values, class_values=_ASAP_CLASS_VALUES)
    macro = macro_query_judge_metrics(
        labels,
        values,
        prompt_ids,
        class_values=_ASAP_CLASS_VALUES,
    )
    observed, low, high = bootstrap_judge_metric_interval(
        labels,
        values,
        metric_name="qwk",
        query_ids=prompt_ids,
        class_values=_ASAP_CLASS_VALUES,
        groups=document_groups,
        rng=np.random.default_rng(42),
        n_boot=bootstrap_samples,
    )
    return {
        "status": "complete",
        "coverage": len(test_rows),
        "expected": len(test_rows),
        "pooled": pooled,
        "macro_prompt": macro["macro"],
        "by_prompt": macro["by_query"],
        "macro_prompt_qwk_observed": observed,
        "macro_prompt_qwk_ci95": [low, high],
        "qwk_bootstrap": {
            "unit": "input_document_id",
            "samples": int(bootstrap_samples),
            "interval": "two_sided_percentile_95",
        },
    }


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(str(text))]


def _stem_tokens(text: str) -> list[str]:
    return [_STEMMER.stem(token) for token in _tokens(text)]


def _word_ngram_analyzer(text: str) -> list[str]:
    tokens = _tokens(text)
    return tokens + _ngrams(tokens, 2, 2)


def _stem_ngram_analyzer(text: str) -> list[str]:
    tokens = _stem_tokens(text)
    return tokens + _ngrams(tokens, 2, 2)


def _pos_tags(texts: Sequence[str]) -> list[list[str]]:
    token_rows = [_tokens(text) for text in texts]
    try:
        return [[tag for _, tag in row] for row in nltk.pos_tag_sents(token_rows, lang="eng")]
    except LookupError as error:
        raise RuntimeError(
            "EASE POS features require NLTK averaged_perceptron_tagger_eng; set NLTK_DATA to its parent directory"
        ) from error


def _ngrams(tokens: Sequence[str], minimum: int, maximum: int) -> list[str]:
    values: list[str] = []
    for width in range(minimum, maximum + 1):
        values.extend(" ".join(tokens[index : index + width]) for index in range(len(tokens) - width + 1))
    return values


def _scale_scores(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float64) - 3.0) / 2.0


def _unscale_scores(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, dtype=np.float64) * 2.0 + 3.0), 1, 5).astype(np.int64)
