#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import read_json
from src.llm_judge_ood.data.asap_prompting import (
    ASAP_JUDGE_TEMPLATE_VERSION,
    prompt_catalog_metadata,
)
from src.llm_judge_ood.data.prepare_benchmark_ground_truth import (
    BENCHMARK_GT_TEMPLATE_IDENTITIES,
    prepare_biggen_bench,
    prepare_flask,
    prepare_longjudgebench,
    prepare_prometheus,
    prepare_ruverbench,
)
from src.llm_judge_ood.data.prepare_ag_news import (
    AG_NEWS_TEMPLATE_SHA256,
    AG_NEWS_TEMPLATE_VERSION,
    prepare_ag_news,
)
from src.llm_judge_ood.data.prepare_asap import ASAP_USED_PROMPTS, write_asap_prepared
from src.llm_judge_ood.data.prepare_clinc150 import (
    CLINC150_TEMPLATE_SHA256,
    CLINC150_TEMPLATE_VERSION,
    prepare_clinc150,
)
from src.llm_judge_ood.data.prepare_ellipse import (
    ELLIPSE_TEMPLATE_SHA256,
    ELLIPSE_TEMPLATE_VERSION,
    prepare_ellipse,
)
from src.llm_judge_ood.data.prepare_rostd import (
    ROSTD_TEMPLATE_SHA256,
    ROSTD_TEMPLATE_VERSION,
    prepare_rostd,
)


DEFAULT_CONFIG = "configs/llm_judge_ood/llm_judge_ood_hiddenstate_datasets.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen A/B HiddenState contracts for every registered dataset."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dataset",
        nargs="+",
        default=["all"],
        help="Dataset key(s) from the config, or 'all'.",
    )
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="When --dataset all is used, skip optional AG News.",
    )
    args = parser.parse_args(argv)

    config_path = _path(args.config)
    config = read_json(config_path)
    datasets = config.get("datasets") if isinstance(config, dict) else None
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("HiddenState dataset config must define a non-empty datasets object")
    selected = _selected_datasets(
        datasets,
        requested=[str(value) for value in args.dataset],
        required_only=bool(args.required_only),
    )
    outputs: dict[str, Any] = {}
    for name in selected:
        spec = datasets[name]
        if not isinstance(spec, dict):
            raise ValueError(f"Dataset config {name!r} must be a JSON object")
        outputs[name] = prepare_from_spec(name, spec)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


def prepare_from_spec(name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    adapter = str(spec.get("adapter") or name)
    _validate_template_identity(adapter, spec)
    if adapter == "ellipse":
        return prepare_ellipse(
            train_path=_path(spec["train_path"]),
            test_path=_path(spec["test_path"]),
            output_path=_path(spec["prepared_path"]),
            rubric_path=_path(spec["rubric_path"]),
            seed=int(spec.get("seed", 42)),
            source_prompt_count=int(spec.get("source_prompt_count", 30)),
            test_zip_password=str(spec.get("test_zip_password", "ellipse_test")),
            expected_train_sha256=_optional_string(spec.get("expected_train_sha256")),
            expected_test_sha256=_optional_string(spec.get("expected_test_sha256")),
            expected_rubric_sha256=_optional_string(spec.get("expected_rubric_sha256")),
        )
    if adapter == "asap_aes":
        configured_prompts = tuple(int(value) for value in spec.get("used_prompt_ids", ()))
        if configured_prompts != tuple(ASAP_USED_PROMPTS):
            raise ValueError(
                f"ASAP config used_prompt_ids must be {list(ASAP_USED_PROMPTS)}, got {configured_prompts}"
            )
        return write_asap_prepared(
            input_path=_path(spec["input_path"]),
            output_path=_path(spec["prepared_path"]),
            seed=int(spec.get("seed", 42)),
        )
    if adapter == "clinc150":
        return prepare_clinc150(
            data_path=_path(spec["data_path"]),
            domains_path=_path(spec["domains_path"]),
            output_path=_path(spec["prepared_path"]),
            expected_data_sha256=_optional_string(spec.get("expected_data_sha256")),
            expected_domains_sha256=_optional_string(spec.get("expected_domains_sha256")),
        )
    if adapter == "rostd":
        return prepare_rostd(
            train_path=_path(spec["train_path"]),
            validation_path=_path(spec["validation_path"]),
            test_path=_path(spec["test_path"]),
            ood_release_path=_path(spec["ood_release_path"]),
            output_path=_path(spec["prepared_path"]),
            expected_sha256={
                str(key): str(value)
                for key, value in dict(spec.get("expected_sha256") or {}).items()
            },
        )
    if adapter == "ag_news":
        return prepare_ag_news(
            train_paths=_expand_globs(spec.get("train_paths", ())),
            test_paths=_expand_globs(spec.get("test_paths", ())),
            output_path_template=_path_template(spec["output_path_template"]),
            folds=tuple(int(value) for value in spec.get("folds", (0, 1, 2, 3))),
            expected_sha256={
                str(key): tuple(str(item) for item in value)
                for key, value in dict(spec.get("expected_sha256") or {}).items()
            },
        )
    benchmark_adapters = {
        "longjudgebench": prepare_longjudgebench,
        "ruverbench": prepare_ruverbench,
        "biggen_bench": prepare_biggen_bench,
        "flask": prepare_flask,
        "prometheus": prepare_prometheus,
    }
    if adapter in benchmark_adapters:
        return benchmark_adapters[adapter](**_benchmark_adapter_kwargs(spec))
    raise ValueError(f"Unsupported HiddenState dataset adapter {adapter!r} for {name!r}")


def _validate_template_identity(adapter: str, spec: Mapping[str, Any]) -> None:
    asap_metadata = prompt_catalog_metadata()
    expected = {
        "ellipse": (ELLIPSE_TEMPLATE_VERSION, ELLIPSE_TEMPLATE_SHA256),
        "asap_aes": (ASAP_JUDGE_TEMPLATE_VERSION, asap_metadata["template_sha256"]),
        "clinc150": (CLINC150_TEMPLATE_VERSION, CLINC150_TEMPLATE_SHA256),
        "rostd": (ROSTD_TEMPLATE_VERSION, ROSTD_TEMPLATE_SHA256),
        "ag_news": (AG_NEWS_TEMPLATE_VERSION, AG_NEWS_TEMPLATE_SHA256),
        **BENCHMARK_GT_TEMPLATE_IDENTITIES,
    }
    if adapter not in expected:
        return
    configured = (
        str(spec.get("prompt_template_version") or ""),
        str(spec.get("prompt_template_sha256") or ""),
    )
    if configured != expected[adapter]:
        raise ValueError(
            f"{adapter} config template identity {configured} does not match code {expected[adapter]}"
        )


def _selected_datasets(
    datasets: Mapping[str, Any],
    *,
    requested: Sequence[str],
    required_only: bool,
) -> list[str]:
    if "all" in requested:
        if len(requested) != 1:
            raise ValueError("Use --dataset all by itself")
        return [
            name
            for name, spec in datasets.items()
            if not required_only or bool(dict(spec).get("required", False))
        ]
    missing = sorted(set(requested) - set(datasets))
    if missing:
        raise ValueError(f"Unknown configured datasets: {missing}")
    return list(dict.fromkeys(requested))


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def _path_template(value: Any) -> str:
    path = Path(str(value)).expanduser()
    return str(path if path.is_absolute() else ROOT / path)


def _benchmark_adapter_kwargs(spec: Mapping[str, Any]) -> dict[str, Any]:
    input_paths = spec.get("input_paths")
    if input_paths is None:
        input_path = spec.get("input_path")
        if input_path is None:
            raise ValueError("Benchmark HiddenState adapters require input_path or input_paths")
        input_paths = [input_path]
    return {
        "input_paths": [_path_template(value) for value in input_paths],
        "output_path": _path(spec["prepared_path"]),
        "records_path": _optional_string(spec.get("records_path")),
        "expected_sha256": {
            str(key): str(value)
            for key, value in dict(spec.get("expected_sha256") or {}).items()
        },
        "seed": int(spec.get("seed", 42)),
        "max_records": int(spec.get("max_records", 0)),
        "merge_records_by_id": bool(spec.get("merge_records_by_id", False)),
        "selected_task_codes": _optional_sequence(spec, "selected_task_codes"),
        "selected_benchmark_codes": _optional_sequence(spec, "selected_benchmark_codes"),
        "benchmark_map": {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(spec.get("benchmark_map") or {}).items()
        },
        "task_map": {
            str(key): tuple(str(item) for item in value)
            for key, value in dict(spec.get("task_map") or {}).items()
        },
        "training_benchmark": str(spec.get("training_benchmark", "A")),
        "training_task": str(spec.get("training_task", "Q")),
    }


def _expand_globs(patterns: Sequence[Any]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        resolved = _path_template(pattern)
        matches = [Path(path) for path in sorted(glob.glob(resolved))]
        if not matches:
            raise FileNotFoundError(f"No files matched {resolved}")
        paths.extend(matches)
    return paths


def _optional_string(value: Any) -> str | None:
    return None if value in (None, "") else str(value)


def _optional_sequence(spec: Mapping[str, Any], key: str) -> tuple[str, ...] | None:
    if key not in spec:
        return None
    return tuple(str(value) for value in spec.get(key, ()))


if __name__ == "__main__":
    main()
