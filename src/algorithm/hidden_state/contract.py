from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HiddenStateCacheMetadata:
    space: str
    feature_scope: str
    layers: tuple[int, ...]
    pooling: str
    model_id: str
    revision: str | None
    prompt_template: str | None
    max_length: int
    view: str

    def __post_init__(self) -> None:
        if self.space not in {"a", "b"}:
            raise ValueError("space must be 'a' or 'b'")
        if not self.feature_scope:
            raise ValueError("feature_scope is required")
        if not self.layers:
            raise ValueError("at least one layer is required")
        if not self.pooling:
            raise ValueError("pooling is required")
        if not self.model_id:
            raise ValueError("model_id is required")
        if int(self.max_length) < 1:
            raise ValueError("max_length must be positive")
        if not self.view:
            raise ValueError("view is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
