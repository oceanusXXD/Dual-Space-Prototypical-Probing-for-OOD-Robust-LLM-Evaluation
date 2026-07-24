from __future__ import annotations

import re


def parse_direct_score(text: str) -> int | None:
    match = re.search(r"-?\d+", str(text))
    return int(match.group(0)) if match else None
