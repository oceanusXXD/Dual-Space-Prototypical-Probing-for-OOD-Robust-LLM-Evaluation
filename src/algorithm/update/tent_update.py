from __future__ import annotations


class TentUpdateNotConfigured(RuntimeError):
    """Raised when test-time entropy minimization is requested without a model adapter."""


def require_tent_adapter() -> None:
    raise TentUpdateNotConfigured("TENT update requires an explicit model adapter implementation")
