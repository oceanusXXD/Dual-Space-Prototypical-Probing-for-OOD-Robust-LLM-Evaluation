from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io import read_jsonl


def load_flask_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)
