#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.llm_judge_ood.shared.feature_store import record_fingerprint
from src.llm_judge_ood.shared.schema import limit_input_document_records, load_judge_records


EXTRACTOR = ROOT / "scripts/llm_judge_ood/20_prepare_llm_judge_ood_hidden.py"
DEFAULT_CONFIG = ROOT / "configs/llm_judge_ood/llm_judge_ood_hiddenstate_datasets.json"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract the configured A and B HiddenState caches for all prepared datasets."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset", nargs="+", default=["all"])
    parser.add_argument("--required-only", action="store_true")
    parser.add_argument(
        "--include-auxiliary",
        action="store_true",
        help="Also extract configured post-selection prompt variants, currently the two ASAP sidecars.",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batch-tokens", type=int, default=None)
    parser.add_argument("--pad-to-multiple-of", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Store all final caches and resumable parts under this dedicated directory.",
    )
    parser.add_argument("--max-input-documents", type=int, default=0)
    parser.add_argument(
        "--smoke-output-dir",
        default="artifacts/llm_judge_ood_hidden_smoke",
        help="Used instead of formal cache paths whenever max-input-documents is positive.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model = config.get("model")
    datasets = config.get("datasets")
    if not isinstance(model, dict) or not isinstance(datasets, dict):
        raise ValueError("HiddenState config must define model and datasets objects")
    selected = _selected_datasets(
        datasets,
        requested=[str(value) for value in args.dataset],
        required_only=bool(args.required_only),
    )

    completed: list[dict[str, str]] = []
    for dataset_name in selected:
        spec = datasets[dataset_name]
        completed.extend(
            _extract_pair(
                dataset_name=dataset_name,
                dataset_spec=spec,
                model=model,
                args=args,
            )
        )
        if args.include_auxiliary:
            auxiliary = spec.get("auxiliary_hiddenstate_inputs", {})
            if not isinstance(auxiliary, dict):
                raise ValueError(
                    f"auxiliary_hiddenstate_inputs for {dataset_name} must be an object"
                )
            for auxiliary_name, auxiliary_spec in auxiliary.items():
                if not isinstance(auxiliary_spec, dict):
                    raise ValueError(
                        f"Auxiliary HiddenState spec {dataset_name}/{auxiliary_name} must be an object"
                    )
                completed.extend(
                    _extract_pair(
                        dataset_name=f"{dataset_name}__{auxiliary_name}",
                        dataset_spec=auxiliary_spec,
                        model=model,
                        args=args,
                    )
                )
    print(json.dumps({"completed": completed}, indent=2, ensure_ascii=False))


def _extract_pair(
    *,
    dataset_name: str,
    dataset_spec: Mapping[str, Any],
    model: Mapping[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, str]]:
    prepared_path = _path(dataset_spec["prepared_path"])
    if not prepared_path.is_file():
        raise FileNotFoundError(
            f"Prepared input for {dataset_name} is missing: {prepared_path}. Run script 34 first."
        )
    records = load_judge_records([prepared_path])
    records = limit_input_document_records(
        records,
        int(args.max_input_documents),
        seed=int(model.get("seed", 42)),
    )
    expected_a_fingerprint = record_fingerprint(records, feature_scope="input_document")
    expected_b_fingerprint = record_fingerprint(records, feature_scope="judge_input")
    expected_a_records = len({record.input_document_id for record in records})
    expected_b_records = len(records)
    del records
    a_output_path = _scope_output_path(
        dataset_name=dataset_name,
        formal_path=_path(dataset_spec["a_cache_path"]),
        space="A",
        args=args,
    )
    b_output_path = _scope_output_path(
        dataset_name=dataset_name,
        formal_path=_path(dataset_spec["b_cache_path"]),
        space="B",
        args=args,
    )

    # A-space reads only input_document_text: raw essay, utterance, or article.
    _extract_scope(
        dataset_name=dataset_name,
        feature_scope="input_document",
        input_path=prepared_path,
        output_path=a_output_path,
        model=model,
        args=args,
    )
    _validate_extracted_cache(
        output_path=a_output_path,
        feature_scope="input_document",
        expected_fingerprint=expected_a_fingerprint,
        expected_records=expected_a_records,
        dataset_spec=dataset_spec,
        model=model,
    )

    # B-space reads only judge_input_text: frozen label-free task prompt plus document.
    _extract_scope(
        dataset_name=dataset_name,
        feature_scope="judge_input",
        input_path=prepared_path,
        output_path=b_output_path,
        model=model,
        args=args,
    )
    _validate_extracted_cache(
        output_path=b_output_path,
        feature_scope="judge_input",
        expected_fingerprint=expected_b_fingerprint,
        expected_records=expected_b_records,
        dataset_spec=dataset_spec,
        model=model,
    )
    return [
        {
            "dataset": dataset_name,
            "space": "A",
            "feature_scope": "input_document",
            "output": str(a_output_path),
        },
        {
            "dataset": dataset_name,
            "space": "B",
            "feature_scope": "judge_input",
            "output": str(b_output_path),
        },
    ]


def _extract_scope(
    *,
    dataset_name: str,
    feature_scope: str,
    input_path: Path,
    output_path: Path,
    model: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    batch_size = int(args.batch_size or model.get("batch_size", 1))
    maximum_tokens = int(
        args.max_batch_tokens
        if args.max_batch_tokens is not None
        else model.get("max_batch_tokens", 0)
    )
    command = [
        sys.executable,
        str(EXTRACTOR),
        "--input",
        str(input_path),
        "--model-path",
        str(args.model_path),
        "--model-id",
        str(model["model_id"]),
        "--revision",
        str(model["revision"]),
        "--output",
        str(output_path),
        "--parts-dir",
        str(output_path.with_name(f"{output_path.stem}.parts")),
        "--layers",
        *[str(value) for value in model.get("layers", (-10, -1))],
        "--batch-size",
        str(batch_size),
        "--max-batch-tokens",
        str(maximum_tokens),
        "--tokenization-batch-size",
        str(int(model.get("tokenization_batch_size", 256))),
        "--pad-to-multiple-of",
        str(
            int(args.pad_to_multiple_of)
            if args.pad_to_multiple_of is not None
            else int(model.get("pad_to_multiple_of", 1))
        ),
        "--max-length",
        str(int(model.get("max_length", 2048))),
        "--truncation-strategy",
        str(model.get("truncation_strategy", "right")),
        "--device",
        str(args.device),
        "--torch-dtype",
        str(model.get("torch_dtype", "bfloat16")),
        "--hidden-dtype",
        str(model.get("hidden_dtype", "float16")),
        "--attn-implementation",
        str(model.get("attn_implementation", "sdpa")),
        "--feature-scope",
        feature_scope,
        "--pooling",
        str(model.get("pooling", "masked_mean")),
        "--seed",
        str(int(model.get("seed", 42))),
        "--max-input-documents",
        str(int(args.max_input_documents)),
        "--fast",
    ]
    if args.local_files_only:
        command.append("--local-files-only")
    if args.overwrite:
        command.append("--overwrite")
    print(
        json.dumps(
            {
                "dataset": dataset_name,
                "space": "A" if feature_scope == "input_document" else "B",
                "feature_scope": feature_scope,
                "input": str(input_path),
                "output": str(output_path),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    subprocess.run(command, cwd=ROOT, check=True)


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


def _validate_extracted_cache(
    *,
    output_path: Path,
    feature_scope: str,
    expected_fingerprint: str,
    expected_records: int,
    dataset_spec: Mapping[str, Any],
    model: Mapping[str, Any],
) -> None:
    metadata_path = output_path.with_suffix(".metadata.json")
    if not output_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"HiddenState cache or metadata is missing for {output_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "feature_scope": feature_scope,
        "dataset_fingerprint": expected_fingerprint,
        "num_records": int(expected_records),
        "model_id": str(model["model_id"]),
        "model_revision": str(model["revision"]),
        "pooling": str(model.get("pooling", "masked_mean")),
        "truncation_strategy": str(model.get("truncation_strategy", "right")),
        "labels_in_prompt": False,
    }
    if feature_scope == "judge_input":
        expected.update(
            {
                "prompt_template_version": str(dataset_spec["prompt_template_version"]),
                "prompt_template_sha256": str(dataset_spec["prompt_template_sha256"]),
            }
        )
    else:
        expected["prompt_template_version"] = "raw_input_document_v1"
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"HiddenState cache metadata mismatch for {output_path}: {mismatches}")


def _scope_output_path(
    *,
    dataset_name: str,
    formal_path: Path,
    space: str,
    args: argparse.Namespace,
) -> Path:
    if int(args.max_input_documents) <= 0:
        if args.output_dir:
            return _path(args.output_dir) / dataset_name / formal_path.name
        return formal_path
    smoke_root = _path(args.smoke_output_dir)
    return smoke_root / dataset_name / f"smoke_{space.lower()}_{formal_path.name}"


def _path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    main()
