#!/usr/bin/env python3
"""Validate and record the pinned LLM-Judge-OOD runtime without inference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io import write_json


_PACKAGES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "joblib",
    "nltk",
    "torch",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "sentencepiece",
    "protobuf",
    "safetensors",
    "datasets",
)

_OPTIONAL_ACCELERATION_PACKAGES = {
    "flash-attn": "flash_attn",
    "triton": "triton",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check the frozen runtime without inference. --require-gpu verifies an "
            "attached CUDA device; --require-flash-attn also checks the optional "
            "FlashAttention acceleration package."
        )
    )
    parser.add_argument("--output", default=None, help="Optional JSON evidence path.")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--require-flash-attn",
        action="store_true",
        help=(
            "Require the optional flash-attn package used only with "
            "--attn-implementation flash_attention_2."
        ),
    )
    args = parser.parse_args()

    versions = {package: importlib.metadata.version(package) for package in _PACKAGES}
    acceleration_versions = {
        package: _optional_package_version(package)
        for package in _OPTIONAL_ACCELERATION_PACKAGES
    }
    import torch
    import transformers
    from src.llm_judge_ood.data.asap_prompting import prompt_catalog_metadata
    from src.llm_judge_ood.pipelines.sample_ood import SampleOODConfig

    # Import the formal config surface rather than executing a model or an
    # extractor.  This is intentionally a CPU-only prerequisite check.
    _ = (torch.__version__, transformers.__version__, SampleOODConfig)
    payload: dict[str, object] = {
        "artifact_type": "llm_judge_ood_environment_validation",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "torch_version": torch.__version__,
        "cuda_runtime_reported_by_torch": torch.version.cuda,
        "torch_cxx11_abi": torch.compiled_with_cxx11_abi(),
        "optional_acceleration_packages": acceleration_versions,
        "catalog": prompt_catalog_metadata(),
        "cpu_import_validation": "passed",
        "gpu_validation": "not_requested",
    }
    if args.require_flash_attn:
        try:
            import flash_attn
        except ImportError as exc:
            raise RuntimeError(
                "FlashAttention was requested but flash-attn is not importable. "
                "Install a wheel matching the current CUDA, PyTorch, Python, and C++ ABI."
            ) from exc
        payload["flash_attention_2_validation"] = {
            "status": "package_import_passed",
            "version": flash_attn.__version__,
            "note": (
                "CUDA kernel execution is intentionally not run by this no-inference "
                "environment check."
            ),
        }
    else:
        payload["flash_attention_2_validation"] = {"status": "not_requested"}
    if args.require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU readiness requested but torch.cuda.is_available() is false")
        payload["gpu_validation"] = "passed_no_inference"
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        payload["cuda_device_count"] = torch.cuda.device_count()
        payload["cuda_device_capability"] = list(torch.cuda.get_device_capability(0))
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            )
            payload["nvidia_driver_version"] = result.stdout.strip().splitlines()[0]
        except (FileNotFoundError, subprocess.CalledProcessError):
            payload["nvidia_driver_version"] = "unavailable"
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _optional_package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    main()
