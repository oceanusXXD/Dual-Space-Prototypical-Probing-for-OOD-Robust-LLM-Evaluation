from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np


@dataclass(frozen=True)
class StaticCacheResult:
    value: Any
    status: str
    path: str | None
    signature: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "path": self.path,
            "signature": self.signature,
            "selection_used_deployment_records": False,
        }


def static_cache_signature(
    namespace: str,
    *,
    config: dict[str, Any],
    arrays: tuple[tuple[str, np.ndarray], ...] = (),
    string_arrays: tuple[tuple[str, np.ndarray], ...] = (),
) -> str:
    """Hash every source-side value that can affect a reusable fitted object."""

    digest = hashlib.sha256()
    digest.update(f"llm_judge_ood_static_cache::{namespace}::v1\0".encode("utf-8"))
    digest.update(
        json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )
    for name, values in arrays:
        array = np.ascontiguousarray(np.asarray(values))
        digest.update(str(name).encode("utf-8") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(json.dumps(array.shape, separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(memoryview(array).cast("B"))
    for name, values in string_arrays:
        encoded = json.dumps(
            np.asarray(values).astype(str).tolist(),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(str(name).encode("utf-8") + b"\0" + encoded)
    return digest.hexdigest()


def load_or_create_static_cache(
    *,
    cache_dir: str | None,
    namespace: str,
    signature: str,
    create: Callable[[], Any],
    validate: Callable[[Any], bool],
) -> StaticCacheResult:
    if cache_dir is None:
        return StaticCacheResult(
            value=create(),
            status="disabled",
            path=None,
            signature=signature,
        )
    cache_path = Path(cache_dir) / namespace / f"{signature}.joblib"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_file_lock(cache_path):
        value = _load(cache_path, namespace=namespace, signature=signature)
        if value is not None and validate(value):
            return StaticCacheResult(
                value=value,
                status="disk_hit",
                path=str(cache_path),
                signature=signature,
            )
        value = create()
        if not validate(value):
            raise TypeError(f"Static cache creator returned an invalid {namespace} value")
        temporary = cache_path.with_name(f".{cache_path.name}.{os.getpid()}.tmp")
        joblib.dump(
            {
                "artifact_type": "llm_judge_ood_static_reference",
                "namespace": namespace,
                "signature": signature,
                "value": value,
            },
            temporary,
            compress=0,
        )
        os.replace(temporary, cache_path)
        return StaticCacheResult(
            value=value,
            status="miss_created",
            path=str(cache_path),
            signature=signature,
        )


def _load(cache_path: Path, *, namespace: str, signature: str) -> Any | None:
    try:
        payload = joblib.load(cache_path)
    except (OSError, EOFError, ValueError, TypeError, AttributeError):
        return None
    if not isinstance(payload, dict):
        return None
    if (
        payload.get("artifact_type") != "llm_judge_ood_static_reference"
        or payload.get("namespace") != namespace
        or payload.get("signature") != signature
    ):
        return None
    return payload.get("value")


@contextmanager
def _exclusive_file_lock(cache_path: Path):
    lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover
            fcntl = None
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
