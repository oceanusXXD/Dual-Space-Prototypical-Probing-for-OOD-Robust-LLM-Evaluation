from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np


@dataclass(frozen=True)
class DecisionRow:
    sample_id: str
    detector: str
    score: float
    threshold: float
    decision: str
    accepted: bool
    split: str | None = None
    query_id: str | None = None
    label: Any | None = None
    prediction: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def apply_accept_reject(
    scores: Iterable[float],
    *,
    threshold: float,
    accept_below: bool = True,
) -> np.ndarray:
    values = np.asarray(list(scores), dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a finite one-dimensional sequence")
    cutoff = float(threshold)
    if not np.isfinite(cutoff):
        raise ValueError("threshold must be finite")
    return values <= cutoff if accept_below else values >= cutoff


def decision_rows_from_scores(
    score_rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    detector: str | None = None,
    accept_below: bool = True,
) -> list[dict[str, Any]]:
    rows = [dict(row) for row in score_rows]
    selected = [
        row
        for row in rows
        if detector is None or str(row.get("detector") or "") == str(detector)
    ]
    accepted = apply_accept_reject(
        [float(row["score"]) for row in selected],
        threshold=float(threshold),
        accept_below=accept_below,
    )
    output: list[dict[str, Any]] = []
    for row, keep in zip(selected, accepted.tolist(), strict=True):
        output.append(
            DecisionRow(
                sample_id=str(row["sample_id"]),
                detector=str(row.get("detector") or detector or "score"),
                score=float(row["score"]),
                threshold=float(threshold),
                decision="accept" if bool(keep) else "reject",
                accepted=bool(keep),
                split=str(row["split"]) if row.get("split") is not None else None,
                query_id=str(row["query_id"]) if row.get("query_id") is not None else None,
                label=row.get("label"),
                prediction=row.get("prediction"),
            ).to_dict()
        )
    return output
