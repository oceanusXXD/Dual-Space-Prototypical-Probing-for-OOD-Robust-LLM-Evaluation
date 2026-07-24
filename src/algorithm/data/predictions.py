from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionRow:
    sample_id: str
    prediction: Any
    label: Any | None = None
    split: str | None = None
    query_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
