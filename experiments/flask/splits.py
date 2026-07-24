from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence


def split_counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("split") or "unassigned") for row in rows).items()))
