from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable
import numpy as np
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def resolve_path(path: str | Path, *, must_exist: bool=False, bases: Iterable[Path] | None=None) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        if must_exist and (not raw.exists()):
            raise FileNotFoundError(raw)
        return raw
    search_bases = list(bases or (Path.cwd(), WORKSPACE_ROOT, PROJECT_ROOT))
    candidates = [base / raw for base in search_bases]
    if must_exist:
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        raise FileNotFoundError(f'Could not resolve {path!s}. Tried: ' + ', '.join((str(candidate) for candidate in candidates)))
    return candidates[0]

def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')

def read_json(path: str | Path) -> Any:
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)

def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write('\n')
    return path

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSONL at {path}:{line_no}: {exc}') from exc
            if not isinstance(item, dict):
                raise ValueError(f'Expected JSON object at {path}:{line_no}')
            records.append(item)
    return records

def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open('w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default))
            handle.write('\n')
    return path

def atomic_write_text(path: str | Path, text: str) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)
    return path
