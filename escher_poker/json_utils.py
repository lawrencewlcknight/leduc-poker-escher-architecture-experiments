"""JSON conversion helpers without solver-framework dependencies."""

from __future__ import annotations

from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    """Convert NumPy containers and non-finite floats for strict JSON output."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        result = float(value)
        return None if not np.isfinite(result) else result
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


__all__ = ["json_safe"]
