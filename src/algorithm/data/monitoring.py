from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class MonitoringArtifact:
    status: str
    stream_rows: int
    drift_score: float | None = None
    p_value: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WindowFailureArtifact:
    window_index: int
    start_row: int
    end_row: int
    rows: int
    reject_rows: int
    reject_rate: float
    failure_confirmed: bool
    min_reject_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def confirm_window_failures(
    decision_rows: Iterable[Mapping[str, object]],
    *,
    window_size: int,
    min_reject_rate: float,
) -> list[dict[str, object]]:
    rows = [dict(row) for row in decision_rows]
    size = int(window_size)
    threshold = float(min_reject_rate)
    if size < 1:
        raise ValueError("window_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("min_reject_rate must be in [0, 1]")
    windows: list[dict[str, object]] = []
    for window_index, start in enumerate(range(0, len(rows), size)):
        stop = min(start + size, len(rows))
        window = rows[start:stop]
        reject_rows = sum(1 for row in window if not bool(row.get("accepted", False)))
        reject_rate = float(reject_rows / len(window)) if window else 0.0
        windows.append(
            WindowFailureArtifact(
                window_index=window_index,
                start_row=start,
                end_row=stop,
                rows=len(window),
                reject_rows=reject_rows,
                reject_rate=reject_rate,
                failure_confirmed=bool(reject_rate >= threshold),
                min_reject_rate=threshold,
            ).to_dict()
        )
    return windows
