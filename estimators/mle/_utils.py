"""Shared input validation for MLE estimators."""

from collections.abc import Iterable
import numpy as np


def observations(values: Iterable[float], *, name: str = "observations") -> np.ndarray:
    """Convert observations to a finite, non-empty one-dimensional float array."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of numbers")
    try:
        array = np.asarray(list(values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain only numeric values") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def parameter(value: float, *, name: str) -> float:
    """Validate and return a finite scalar parameter."""
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result
