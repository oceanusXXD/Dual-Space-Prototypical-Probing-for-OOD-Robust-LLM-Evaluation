from __future__ import annotations

from pathlib import Path
from typing import Any

from src.common.io import read_json


def load_lora_summary(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("LoRA summary must be a JSON object")
    return payload
