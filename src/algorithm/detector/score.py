from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DetectorScore:
    sample_id: str
    detector: str
    score: float
    split: str | None = None
    label: Any | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
