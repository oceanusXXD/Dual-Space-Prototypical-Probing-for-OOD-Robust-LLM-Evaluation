#!/usr/bin/env python3
"""Run the FLASK Direct Judge vs head vs LoRA comparison pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LOCAL_DEPS = ROOT / ".deps" / "python_min"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path("artifacts/flask_direct_head_lora_comparison"))
    parser.add_argument("--model-path", type=Path, default=Path("models/qwen3.5-0.8b"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite-direct", action="store_true")
    parser.add_argument("--overwrite-head", action="store_true")
    parser.add_argument("--overwrite-lora", action="store_true")
    parser.add_argument("--direct-batch-size", type=int, default=512)
    parser.add_argument("--direct-max-batch-tokens", type=int, default=196608)
    parser.add_argument("--head-epochs", type=int, default=50)
    parser.add_argument("--lora-epochs", type=int, default=1)
    parser.add_argument("--lora-train-batch-size", type=int, default=2)
    parser.add_argument("--lora-eval-batch-size", type=int, default=16)
    parser.add_argument("--lora-gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--ignore-preflight",
        action="store_true",
        help="Run commands even if package/GPU preflight fails.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    args.base_dir.mkdir(parents=True, exist_ok=True)
    preflight = build_preflight(args)
    write_json(args.base_dir / "pipeline_preflight.json", preflight)
    if preflight["errors"] and not args.ignore_preflight:
        raise RuntimeError(
            "Pipeline preflight failed; see "
            f"{args.base_dir / 'pipeline_preflight.json'}: {preflight['errors']}"
        )

    run([sys.executable, "scripts/llm_judge_ood/67_prepare_flask_direct_head_lora_comparison.py"])

    direct_summary = args.base_dir / "direct_and_features/summary.json"
    direct_cmd = [
        sys.executable,
        "scripts/llm_judge_ood/68_run_flask_comparison_direct_and_features.py",
        "--rows",
        str(args.base_dir / "comparison_rows.jsonl"),
        "--split-manifest",
        str(args.base_dir / "split_manifest.json"),
        "--output-dir",
        str(args.base_dir / "direct_and_features"),
        "--model-path",
        str(args.model_path),
        "--device",
        str(args.device),
        "--batch-size",
        str(args.direct_batch_size),
        "--max-batch-tokens",
        str(args.direct_max_batch_tokens),
    ]
    if args.overwrite_direct:
        direct_cmd.append("--overwrite")
    if args.overwrite_direct or not direct_summary.exists():
        run(direct_cmd)

    head_summary = args.base_dir / "classification_head/summary.json"
    head_cmd = [
        sys.executable,
        "scripts/llm_judge_ood/69_train_flask_comparison_head.py",
        "--rows",
        str(args.base_dir / "direct_and_features/b_space_with_direct_judge.jsonl"),
        "--split-manifest",
        str(args.base_dir / "split_manifest.json"),
        "--features",
        str(args.base_dir / "direct_and_features/strict_final_prelogit_features.npz"),
        "--output-dir",
        str(args.base_dir / "classification_head"),
        "--epochs",
        str(args.head_epochs),
    ]
    if args.overwrite_head:
        head_cmd.append("--overwrite")
    if args.overwrite_head or not head_summary.exists():
        run(head_cmd)

    lora_summary = args.base_dir / "lora/summary.json"
    lora_cmd = [
        sys.executable,
        "scripts/llm_judge_ood/70_train_flask_comparison_lora.py",
        "--rows",
        str(args.base_dir / "comparison_rows.jsonl"),
        "--split-manifest",
        str(args.base_dir / "split_manifest.json"),
        "--output-dir",
        str(args.base_dir / "lora"),
        "--model-path",
        str(args.model_path),
        "--device",
        str(args.device),
        "--epochs",
        str(args.lora_epochs),
        "--train-batch-size",
        str(args.lora_train_batch_size),
        "--eval-batch-size",
        str(args.lora_eval_batch_size),
        "--gradient-accumulation-steps",
        str(args.lora_gradient_accumulation_steps),
    ]
    if args.overwrite_lora:
        lora_cmd.append("--overwrite")
    if args.overwrite_lora or not lora_summary.exists():
        run(lora_cmd)

    run([sys.executable, "scripts/llm_judge_ood/71_summarize_flask_comparison_results.py"])
    write_json(
        args.base_dir / "pipeline_summary.json",
        {
            "artifact_type": "flask_direct_head_lora_pipeline_summary_v1",
            "base_dir": str(args.base_dir),
            "elapsed_seconds": time.perf_counter() - started,
            "outputs": {
                "direct": str(direct_summary),
                "classification_head": str(head_summary),
                "lora": str(lora_summary),
                "performance_summary": str(args.base_dir / "performance_summary.csv"),
            },
        },
    )


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "transformers",
            "accelerate",
            "peft",
            "datasets",
            "bitsandbytes",
            "scikit-learn",
            "joblib",
        )
    }
    errors: list[str] = []
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_count = int(torch.cuda.device_count())
        torch_cuda = torch.version.cuda
    except Exception as error:  # pragma: no cover - environment diagnostic path.
        cuda_available = False
        cuda_count = 0
        torch_cuda = f"error: {error!r}"
        errors.append(f"torch import/cuda check failed: {error!r}")
    if args.require_gpu and not cuda_available:
        errors.append("CUDA is required for the full Direct Judge + LoRA run, but torch cannot see a GPU.")
    if not args.model_path.exists():
        errors.append(f"model path does not exist: {args.model_path}")
    for package in ("transformers", "accelerate", "peft", "scikit-learn", "joblib"):
        if packages.get(package) is None:
            errors.append(f"missing required package: {package}")
    return {
        "artifact_type": "flask_direct_head_lora_pipeline_preflight_v1",
        "base_dir": str(args.base_dir),
        "model_path": str(args.model_path),
        "local_deps": str(LOCAL_DEPS) if LOCAL_DEPS.exists() else "",
        "device": str(args.device),
        "require_gpu": bool(args.require_gpu),
        "packages": packages,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": cuda_count,
        "torch_cuda_runtime": torch_cuda,
        "errors": errors,
    }


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    env = None
    if LOCAL_DEPS.exists():
        import os

        env = os.environ.copy()
        current = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(LOCAL_DEPS) if not current else f"{LOCAL_DEPS}:{current}"
    subprocess.run(command, cwd=ROOT, check=True, env=env)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
