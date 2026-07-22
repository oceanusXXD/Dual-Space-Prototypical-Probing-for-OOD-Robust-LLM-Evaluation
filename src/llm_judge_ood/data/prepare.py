from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.common.io import write_jsonl
from src.llm_judge_ood.shared.schema import load_judge_records


def prepare_records(input_paths: Iterable[str | Path], *, output_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Normalize JSONL Query-Document rows into the public LLM Judge OOD schema."""

    records = [record.to_dict() for record in load_judge_records(input_paths)]
    if output_path is not None:
        write_jsonl(output_path, records)
    return records
