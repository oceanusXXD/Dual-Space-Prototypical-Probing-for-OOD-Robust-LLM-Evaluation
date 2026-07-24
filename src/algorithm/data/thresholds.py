from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ThresholdArtifact:
    detector: str
    threshold: float | None
    risk_bound: float
    delta: float
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
