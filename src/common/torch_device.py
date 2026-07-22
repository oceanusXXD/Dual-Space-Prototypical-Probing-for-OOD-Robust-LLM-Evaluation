from __future__ import annotations
import torch

def resolve_torch_device(device: str | torch.device | None=None, *, prefer_cuda_when_unspecified: bool=False) -> torch.device:
    if device is None:
        requested = 'cuda' if prefer_cuda_when_unspecified and torch.cuda.is_available() else 'cpu'
    else:
        requested = str(device)
    resolved = torch.device(requested)
    if resolved.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA device requested but torch.cuda.is_available() is False. Use --device cpu, or install/run with a CUDA-enabled PyTorch environment.')
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {resolved.index} requested, but only {torch.cuda.device_count()} CUDA device(s) are visible.')
    return resolved
