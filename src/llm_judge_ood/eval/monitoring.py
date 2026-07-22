"""Stage-wise monitoring baselines for the OOD lifecycle.

The OOD score itself is held fixed.  These baselines answer a different
question from sample-level AUROC: whether adding aggregation, clustering,
persistence, and a harmfulness probe improves operational decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.llm_judge_ood.lifecycle.cluster import DocumentClusterer, ClusterConfig
from src.llm_judge_ood.lifecycle.persistence import DocumentClusterTracker, PersistenceConfig


OOD_STATUSES = frozenset({"soft_ood", "hard_ood"})
HARM_STATUSES = frozenset({"harmful", "benign", "uncertain"})


@dataclass(frozen=True)
class MonitoringBaselineConfig:
    """Fixed protocol shared by all deployable monitoring baselines."""

    window_size: int = 64
    ood_rate_threshold: float = 0.05
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    persistence: PersistenceConfig = field(default_factory=PersistenceConfig)

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "cluster": self.cluster.to_dict(),
            "persistence": self.persistence.to_dict(),
        }


def evaluate_monitoring_baselines(
    *,
    stream_indices: Sequence[int] | np.ndarray,
    score_labels: Sequence[str] | np.ndarray,
    embeddings: np.ndarray,
    audit_document_group_ids: Sequence[str] | np.ndarray,
    audit_document_group_harmfulness: Mapping[str, str],
    config: MonitoringBaselineConfig,
    clustering_detection_events: Sequence[Mapping[str, Any]] | None = None,
    persistence_detection_events: Sequence[Mapping[str, Any]] | None = None,
    full_detection_events: Sequence[Mapping[str, Any]] | None = None,
    full_action_events: Sequence[Mapping[str, Any]] | None = None,
    full_detection_indices: Sequence[int] | np.ndarray = (),
    full_action_indices: Sequence[int] | np.ndarray = (),
    full_label_cost: int = 0,
) -> dict[str, Any]:
    """Compare the five monitoring stages using a common OOD score stream.

    ``audit_document_group_harmfulness`` is evaluation-only ground truth derived after the
    run from fully labeled deployment data. It is never used by the
    deployable baselines themselves; the full method still uses only its
    separate random Probe pool for its action decision.
    ``clustering_detection_events`` and ``persistence_detection_events`` let
    the caller pass the exact events produced by the deployed lifecycle. This
    is important for a fair stage-wise comparison: clustering, persistence,
    and Persistence+Probe must share the same clustering and arrival
    semantics, then differ only by the additional stage being tested.

    ``full_detection_events`` and ``full_action_events`` are the preferred
    representation for the complete method.  The legacy ``*_indices``
    arguments remain supported for small fixtures and callers that have only
    a single event; they should not be used when a confirmation time is
    available, because an old member of a confirmed document cluster is not its alarm
    time.
    """

    stream = np.asarray(stream_indices, dtype=int)
    labels = np.asarray(score_labels).astype(str)
    values = np.asarray(embeddings, dtype=np.float32)
    audit_document_groups = np.asarray(audit_document_group_ids).astype(str)
    _validate_inputs(
        stream=stream,
        score_labels=labels,
        embeddings=values,
        audit_document_group_ids=audit_document_groups,
        config=config,
    )
    truth = _normalize_harmfulness(audit_document_group_harmfulness)
    candidate_mask = np.isin(labels, sorted(OOD_STATUSES))

    immediate_events = [
        _event("single_ood_immediate_alarm", position, [int(index)])
        for position, index in enumerate(stream.tolist())
        if bool(candidate_mask[int(index)])
    ]
    window_events = _window_rate_events(
        stream=stream,
        candidate_mask=candidate_mask,
        window_size=int(config.window_size),
        threshold=float(config.ood_rate_threshold),
    )
    recomputed_clustering_events, recomputed_persistence_events = _cluster_and_persistence_events(
        stream=stream,
        candidate_mask=candidate_mask,
        embeddings=values,
        config=config,
    )
    clustering_events = (
        _normalize_external_events(
            clustering_detection_events,
            name="ood_plus_clustering",
            stream=stream,
        )
        if clustering_detection_events is not None
        else recomputed_clustering_events
    )
    persistence_events = (
        _normalize_external_events(
            persistence_detection_events,
            name="ood_clustering_persistence",
            stream=stream,
        )
        if persistence_detection_events is not None
        else recomputed_persistence_events
    )
    full_detection = (
        _normalize_external_events(
            full_detection_events,
            name="persistence_plus_probe_detection",
            stream=stream,
        )
        if full_detection_events is not None
        else _single_event_from_indices(
            "persistence_plus_probe_detection",
            stream=stream,
            indices=np.asarray(full_detection_indices, dtype=int),
        )
    )
    full_action = (
        _normalize_external_events(
            full_action_events,
            name="persistence_plus_probe_action",
            stream=stream,
        )
        if full_action_events is not None
        else _single_event_from_indices(
            "persistence_plus_probe_action",
            stream=stream,
            indices=np.asarray(full_action_indices, dtype=int),
        )
    )
    oracle_detection, oracle_action = _audit_oracle_events(
        stream=stream,
        audit_document_groups=audit_document_groups,
        truth=truth,
    )

    methods = [
        _summarize_method(
            name="single_ood_immediate_alarm",
            stage="single_sample_ood",
            deployable=True,
            detection_events=immediate_events,
            action_events=immediate_events,
            stream=stream,
            audit_document_groups=audit_document_groups,
            truth=truth,
            label_cost=0,
        ),
        _summarize_method(
            name="window_ood_rate",
            stage="window_aggregation",
            deployable=True,
            detection_events=window_events,
            action_events=window_events,
            stream=stream,
            audit_document_groups=audit_document_groups,
            truth=truth,
            label_cost=0,
        ),
        _summarize_method(
            name="ood_plus_clustering",
            stage="within_window_clustering",
            deployable=True,
            detection_events=clustering_events,
            action_events=clustering_events,
            stream=stream,
            audit_document_groups=audit_document_groups,
            truth=truth,
            label_cost=0,
        ),
        _summarize_method(
            name="ood_clustering_persistence",
            stage="cross_window_persistence",
            deployable=True,
            detection_events=persistence_events,
            action_events=persistence_events,
            stream=stream,
            audit_document_groups=audit_document_groups,
            truth=truth,
            label_cost=0,
        ),
        _summarize_method(
            name="full_persistence_probe",
            stage="persistence_plus_harmfulness_probe",
            deployable=True,
            detection_events=full_detection,
            action_events=full_action,
            stream=stream,
            audit_document_groups=audit_document_groups,
            truth=truth,
            label_cost=max(0, int(full_label_cost)),
        ),
        _summarize_method(
            name="audit_oracle_document_group",
            stage="audit_oracle_upper_bound",
            deployable=False,
            detection_events=oracle_detection,
            action_events=oracle_action,
            stream=stream,
            audit_document_groups=audit_document_groups,
            truth=truth,
            label_cost=0,
        ),
    ]
    return {
        "protocol": {
            "ood_score_is_fixed": True,
            "ood_statuses_that_trigger_single_sample_alarm": sorted(OOD_STATUSES),
            "clustering_baseline_uses_deployed_lifecycle_events": clustering_detection_events is not None,
            "persistence_baseline_uses_deployed_lifecycle_events": persistence_detection_events is not None,
            "full_method_uses_explicit_confirmation_events": full_detection_events is not None,
            "full_method_action_rule": "persistent_document_cluster_and_harmful_probe",
            "harmfulness_ground_truth": "fully_labeled_deployment_documents_for_evaluation_only",
            "metric_definitions": {
                "harmful_detection_recall": "fraction of harmful audit document groups with at least one detection event",
                "harmful_action_recall": "fraction of harmful audit document groups that reach an alarm/update action",
                "benign_specificity": "fraction of benign audit document groups with no alarm/update action",
                "false_alarm_rate_per_100_stream_rows": "non-harmful action events divided by stream rows, times 100",
                "wrong_update_rate": "non-harmful action events divided by all action events",
                "mean_detection_delay_samples": "first detection position minus first appearance of each detected harmful audit document group",
                "label_cost": "human labels consumed by the monitoring decision; only Probe uses labels",
            },
        },
        "config": config.to_dict(),
        "audit_document_group_harmfulness": {key: truth[key] for key in sorted(truth)},
        "stream_rows": int(stream.size),
        "methods": methods,
    }


def _validate_inputs(
    *,
    stream: np.ndarray,
    score_labels: np.ndarray,
    embeddings: np.ndarray,
    audit_document_group_ids: np.ndarray,
    config: MonitoringBaselineConfig,
) -> None:
    if stream.ndim != 1 or stream.size == 0:
        raise ValueError("stream_indices must be a non-empty one-dimensional array")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [N,D]")
    if len(score_labels) != len(embeddings) or len(audit_document_group_ids) != len(embeddings):
        raise ValueError("score_labels, embeddings, and audit_document_group_ids must be aligned")
    if np.any(stream < 0) or np.any(stream >= len(embeddings)):
        raise ValueError("stream_indices are out of bounds")
    if len(set(stream.tolist())) != len(stream):
        raise ValueError("stream_indices must contain each arrival at most once")
    if int(config.window_size) < 1:
        raise ValueError("monitoring window_size must be positive")
    if not 0.0 <= float(config.ood_rate_threshold) <= 1.0:
        raise ValueError("monitoring ood_rate_threshold must lie in [0,1]")


def _normalize_harmfulness(audit_document_group_harmfulness: Mapping[str, str]) -> dict[str, str]:
    output = {str(key): str(value) for key, value in audit_document_group_harmfulness.items()}
    invalid = {key: value for key, value in output.items() if value not in HARM_STATUSES}
    if invalid:
        raise ValueError(f"audit document-group harmfulness must be one of {sorted(HARM_STATUSES)}: {invalid}")
    return output


def _window_rate_events(
    *,
    stream: np.ndarray,
    candidate_mask: np.ndarray,
    window_size: int,
    threshold: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for start in range(0, len(stream), max(1, int(window_size))):
        stop = min(start + max(1, int(window_size)), len(stream))
        window = stream[start:stop]
        candidates = window[candidate_mask[window]]
        rate = float(len(candidates)) / max(len(window), 1)
        if rate >= float(threshold):
            # A rate threshold can be evaluated only after all arrivals in the
            # window have been observed, so its alarm time is the window end.
            events.append(_event("window_ood_rate", stop - 1, candidates.tolist(), rate=rate))
    return events


def _cluster_and_persistence_events(
    *,
    stream: np.ndarray,
    candidate_mask: np.ndarray,
    embeddings: np.ndarray,
    config: MonitoringBaselineConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clusterer = DocumentClusterer(config.cluster)
    tracker = DocumentClusterTracker(config.persistence)
    clustering_events: list[dict[str, Any]] = []
    persistence_events: list[dict[str, Any]] = []
    window_size = max(1, int(config.window_size))
    for window_index, start in enumerate(range(0, len(stream), window_size)):
        stop = min(start + window_size, len(stream))
        window = stream[start:stop]
        candidates = window[candidate_mask[window]]
        # A window smaller than the configured cluster minimum cannot produce
        # a retained document cluster.  In particular, sklearn's HDBSCAN
        # rejects this input because its effective ``min_samples`` exceeds the
        # number of candidate rows.  Treat it as an empty clustering result so
        # the persistence baseline still receives an explicit no-cluster
        # update for this arrival window.
        if candidates.size < int(config.cluster.min_cluster_size):
            tracker.update(window_index=window_index, window_size=int(window.size), cluster_summaries=[])
            continue
        labels, summaries = clusterer.fit_predict(embeddings[candidates])
        members_by_cluster = {
            int(cluster_id): candidates[labels == int(cluster_id)].astype(int).tolist()
            for cluster_id in sorted(set(labels.tolist()))
            if int(cluster_id) >= 0
        }
        for summary in summaries:
            cluster_id = int(summary["cluster_id"])
            clustering_events.append(
                _event(
                    "ood_plus_clustering",
                    stop - 1,
                    members_by_cluster.get(cluster_id, []),
                    window_index=window_index,
                    cluster_id=cluster_id,
                )
            )
        tracked = tracker.update(
            window_index=window_index,
            window_size=int(window.size),
            cluster_summaries=summaries,
        )
        for row in tracked:
            if row.get("confirmation_window") != window_index:
                continue
            cluster_id = int(row["cluster_id"])
            persistence_events.append(
                _event(
                    "ood_clustering_persistence",
                    stop - 1,
                    members_by_cluster.get(cluster_id, []),
                    window_index=window_index,
                    predicted_document_cluster_id=str(row["document_cluster_id"]),
                )
            )
    return clustering_events, persistence_events


def _single_event_from_indices(name: str, *, stream: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
    selected = np.asarray(indices, dtype=int)
    selected = selected[np.isin(selected, stream)]
    if selected.size == 0:
        return []
    positions = {int(index): position for position, index in enumerate(stream.tolist())}
    ordered = sorted(set(selected.tolist()), key=lambda index: positions[int(index)])
    return [_event(name, positions[int(ordered[0])], ordered)]


def _normalize_external_events(
    events: Sequence[Mapping[str, Any]],
    *,
    name: str,
    stream: np.ndarray,
) -> list[dict[str, Any]]:
    """Validate and normalize event records supplied by the deployed flow.

    Positions are stream-relative arrival positions; members are global record
    indices.  Keeping those two coordinate systems explicit prevents the
    common error of using the first historical member of a confirmed document cluster as
    its detection timestamp.
    """

    stream_members = set(int(index) for index in stream.tolist())
    output: list[dict[str, Any]] = []
    for event in events:
        if "position" not in event:
            raise ValueError("external monitoring event is missing position")
        position = int(event["position"])
        if position < 0 or position >= len(stream):
            raise ValueError("external monitoring event position is outside the stream")
        members = [int(value) for value in event.get("members", [])]
        if not members:
            raise ValueError("external monitoring event must contain at least one member")
        if not set(members).issubset(stream_members):
            raise ValueError("external monitoring event members must belong to the monitoring stream")
        output.append(
            _event(
                name,
                position,
                members,
                **{
                    str(key): value
                    for key, value in event.items()
                    if key not in {"name", "position", "members"}
                },
            )
        )
    return output


def _audit_oracle_events(
    *,
    stream: np.ndarray,
    audit_document_groups: np.ndarray,
    truth: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positions = {int(index): position for position, index in enumerate(stream.tolist())}
    detection: list[dict[str, Any]] = []
    action: list[dict[str, Any]] = []
    for audit_document_group_id in sorted({str(audit_document_groups[int(index)]) for index in stream.tolist()}):
        status = truth.get(audit_document_group_id)
        if status is None:
            continue
        members = [
            int(index)
            for index in stream.tolist()
            if str(audit_document_groups[int(index)]) == audit_document_group_id
        ]
        if not members:
            continue
        event = _event(
            "audit_oracle_document_group",
            positions[members[0]],
            members,
            audit_oracle_document_group_id=audit_document_group_id,
        )
        detection.append(event)
        if status == "harmful":
            action.append(event)
    return detection, action


def _event(name: str, position: int, members: Sequence[int], **metadata: Any) -> dict[str, Any]:
    return {
        "name": name,
        "position": int(position),
        "members": [int(value) for value in members],
        **metadata,
    }


def _summarize_method(
    *,
    name: str,
    stage: str,
    deployable: bool,
    detection_events: list[dict[str, Any]],
    action_events: list[dict[str, Any]],
    stream: np.ndarray,
    audit_document_groups: np.ndarray,
    truth: Mapping[str, str],
    label_cost: int,
) -> dict[str, Any]:
    target_audit_document_groups = sorted(
        {
            str(audit_document_groups[int(index)])
            for index in stream.tolist()
            if str(audit_document_groups[int(index)]) in truth
        }
    )
    harmful = [audit_document_group for audit_document_group in target_audit_document_groups if truth[audit_document_group] == "harmful"]
    benign = [audit_document_group for audit_document_group in target_audit_document_groups if truth[audit_document_group] == "benign"]
    detection_by_audit_document_group = _earliest_event_by_audit_document_group(
        detection_events,
        audit_document_groups=audit_document_groups,
    )
    action_by_audit_document_group = _earliest_event_by_audit_document_group(
        action_events,
        audit_document_groups=audit_document_groups,
    )
    stream_first = _first_stream_position_by_audit_document_group(
        stream,
        audit_document_groups=audit_document_groups,
    )
    harmful_detection = [audit_document_group for audit_document_group in harmful if audit_document_group in detection_by_audit_document_group]
    harmful_action = [audit_document_group for audit_document_group in harmful if audit_document_group in action_by_audit_document_group]
    benign_without_action = [audit_document_group for audit_document_group in benign if audit_document_group not in action_by_audit_document_group]
    false_action_events = [
        event
        for event in action_events
        if truth.get(_dominant_audit_document_group(event, audit_document_groups=audit_document_groups), "id") != "harmful"
    ]
    delays = [
        int(detection_by_audit_document_group[audit_document_group]["position"]) - int(stream_first[audit_document_group])
        for audit_document_group in harmful_detection
        if audit_document_group in stream_first
    ]
    return {
        "method": name,
        "stage": stage,
        "deployable": bool(deployable),
        "detection_event_count": int(len(detection_events)),
        "action_event_count": int(len(action_events)),
        "harmful_audit_document_group_count": int(len(harmful)),
        "benign_audit_document_group_count": int(len(benign)),
        "uncertain_audit_document_group_count": int(
            sum(truth[audit_document_group] == "uncertain" for audit_document_group in target_audit_document_groups)
        ),
        "harmful_detection_recall": _ratio(len(harmful_detection), len(harmful)),
        "harmful_action_recall": _ratio(len(harmful_action), len(harmful)),
        "benign_specificity": _ratio(len(benign_without_action), len(benign)),
        "false_alarm_event_count": int(len(false_action_events)),
        "false_alarm_rate_per_100_stream_rows": float(100.0 * len(false_action_events) / max(len(stream), 1)),
        "wrong_update_rate": _ratio(len(false_action_events), len(action_events)),
        "mean_detection_delay_samples": float(np.mean(delays)) if delays else None,
        "label_cost": int(label_cost),
        "detected_harmful_audit_document_groups": harmful_detection,
        "actioned_harmful_audit_document_groups": harmful_action,
        "benign_audit_document_groups_without_action": benign_without_action,
        "events": {
            "detection": _serialize_events(detection_events, audit_document_groups=audit_document_groups),
            "action": _serialize_events(action_events, audit_document_groups=audit_document_groups),
        },
    }


def _earliest_event_by_audit_document_group(
    events: Sequence[dict[str, Any]],
    *,
    audit_document_groups: np.ndarray,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for event in events:
        audit_document_group = _dominant_audit_document_group(
            event,
            audit_document_groups=audit_document_groups,
        )
        if audit_document_group is None:
            continue
        prior = output.get(audit_document_group)
        if prior is None or int(event["position"]) < int(prior["position"]):
            output[audit_document_group] = event
    return output


def _first_stream_position_by_audit_document_group(
    stream: np.ndarray,
    *,
    audit_document_groups: np.ndarray,
) -> dict[str, int]:
    output: dict[str, int] = {}
    for position, index in enumerate(stream.tolist()):
        output.setdefault(str(audit_document_groups[int(index)]), int(position))
    return output


def _dominant_audit_document_group(
    event: Mapping[str, Any],
    *,
    audit_document_groups: np.ndarray,
) -> str | None:
    members = [int(value) for value in event.get("members", [])]
    if not members:
        return None
    counts: dict[str, int] = {}
    for index in members:
        audit_document_group = str(audit_document_groups[index])
        counts[audit_document_group] = counts.get(audit_document_group, 0) + 1
    return min(counts, key=lambda audit_document_group: (-counts[audit_document_group], audit_document_group))


def _serialize_events(
    events: Sequence[dict[str, Any]],
    *,
    audit_document_groups: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            **{key: value for key, value in event.items() if key != "members"},
            "member_count": int(len(event.get("members", []))),
            "audit_document_group_id": _dominant_audit_document_group(
                event,
                audit_document_groups=audit_document_groups,
            ),
        }
        for event in events
    ]


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None
