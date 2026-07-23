from __future__ import annotations

import os
from pathlib import Path
import subprocess
import importlib.util

import torch
import transformers.utils as transformers_utils
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForCausalLM, AutoTokenizer


QWEN3_5_4B_MODEL_ID = "Qwen/Qwen3.5-4B"
QWEN3_5_4B_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
QWEN3_5_4B_HIDDEN_SIZE = 2560
QWEN3_5_4B_NUM_LAYERS = 32
QWEN3_5_TEXT_MODEL_TYPE = "qwen3_5_text"


def load_qwen_model(
    model_path: str | Path,
    *,
    revision: str | None = None,
    device: str = "cuda",
    torch_dtype: str = "auto",
    attn_implementation: str = "sdpa",
    tf32: bool = True,
    local_files_only: bool = False,
):
    """Load a local or Hugging Face Qwen checkpoint for frozen inference."""

    # This extractor is text-only. Some GPU images ship a torchvision build
    # that is incompatible with their newer PyTorch build; keep that optional
    # vision dependency out of Transformers' lazy Qwen import path.
    transformers_utils.is_torchvision_available = lambda: False
    transformers_import_utils.is_torchvision_available = lambda: False
    requested = str(model_path)
    path = Path(requested).expanduser()
    source = str(path.resolve()) if path.exists() else requested
    requested_device = str(device).lower()
    resolved_device = torch.device(requested_device if not requested_device.startswith("cuda") or torch.cuda.is_available() else "cpu")
    dtype = _resolve_dtype(torch_dtype, resolved_device)
    if resolved_device.type == "cuda":
        disable_cudnn = os.environ.get("LLM_JUDGE_OOD_DISABLE_CUDNN_CONV1D", "").strip().lower()
        if disable_cudnn in {"1", "true", "yes", "on"}:
            # Some hosted T4/cuDNN images cannot select an engine for Qwen3.5's
            # linear-attention Conv1d. PyTorch's CUDA fallback remains available.
            torch.backends.cudnn.enabled = False
            print("Disabled cuDNN for Qwen3.5 linear-attention Conv1d.", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.backends.cudnn.allow_tf32 = bool(tf32)
    tokenizer = AutoTokenizer.from_pretrained(
        source,
        revision=revision,
        local_files_only=bool(local_files_only),
        use_fast=True,
    )
    model_kwargs = {
        "revision": revision,
        "local_files_only": bool(local_files_only),
        "attn_implementation": attn_implementation,
        "low_cpu_mem_usage": importlib.util.find_spec("accelerate") is not None,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype, **model_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(source, torch_dtype=dtype, **model_kwargs)
    model = model.to(resolved_device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return tokenizer, model, resolved_device


def qwen_text_config(model: torch.nn.Module):
    """Return the text config from either a composite or text-only Qwen model."""

    config = model.config
    return getattr(config, "text_config", None) or config


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = str(name).lower()
    if normalized == "float16":
        return torch.float16
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float32":
        return torch.float32
    if normalized == "auto":
        if device.type == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name!r}")


def local_git_revision(path: str | Path) -> str | None:
    """Return the checked-out commit for a local model cloned with Git."""

    source = Path(path).expanduser()
    if not (source / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and len(revision) == 40 else None
