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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the frozen CPU environment; --require-gpu is reserved for a later GPU readiness check."
    )
    parser.add_argument("--output", default=None, help="Optional JSON evidence path.")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    versions = {package: importlib.metadata.version(package) for package in _PACKAGES}
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
        "catalog": prompt_catalog_metadata(),
        "cpu_import_validation": "passed",
        "gpu_validation": "not_requested",
    }
    if args.require_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU readiness requested but torch.cuda.is_available() is false")
        payload["gpu_validation"] = "passed_no_inference"
        payload["cuda_device_name"] = torch.cuda.get_device_name(0)
        payload["cuda_device_count"] = torch.cuda.device_count()
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


if __name__ == "__main__":
    main()
