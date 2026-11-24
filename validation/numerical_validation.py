"""Numerical comparison and MLE validation functions."""

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from estimators.mle import estimate_bernoulli_mle, estimate_normal_mle, estimate_poisson_mle


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of comparing an AI estimate with an independent estimate."""

    ai_estimate: Any
    independent_estimate: Any
    absolute_error: Any
    relative_error: Any
    tolerance: float
    is_valid: bool

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-friendly dictionary."""
        return {"ai_estimate": self.ai_estimate, "independent_estimate": self.independent_estimate,
                "absolute_error": self.absolute_error, "relative_error": self.relative_error,
                "tolerance": self.tolerance, "is_valid": self.is_valid}


def compare_estimates(ai_estimate: Any, independent_estimate: Any, *, tolerance: float = 1e-8) -> ValidationResult:
    """Compare scalar or same-shaped vector estimates with absolute tolerance."""
    if not isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("tolerance must be a finite, non-negative number")
    ai = np.asarray(ai_estimate, dtype=float)
    expected = np.asarray(independent_estimate, dtype=float)
    if ai.shape != expected.shape:
        raise ValueError("estimates must have matching shapes")
    if not np.all(np.isfinite(ai)) or not np.all(np.isfinite(expected)):
        raise ValueError("estimates must be finite")
    absolute = np.abs(ai - expected)
    relative = np.divide(absolute, np.abs(expected), out=np.zeros_like(absolute), where=expected != 0)
    valid = bool(np.all(np.isclose(ai, expected, rtol=tolerance, atol=tolerance)))
    def clean(value: np.ndarray) -> Any:
        return float(value) if value.ndim == 0 else value.tolist()
    return ValidationResult(clean(ai), clean(expected), clean(absolute), clean(relative), float(tolerance), valid)


def validate_mle(method: str, data: Any, ai_estimate: Any, *, tolerance: float = 1e-8) -> ValidationResult:
    """Independently calculate and compare a Bernoulli, Poisson, or Normal MLE."""
    estimators = {"bernoulli": estimate_bernoulli_mle, "poisson": estimate_poisson_mle, "normal": estimate_normal_mle}
    try:
        independent = estimators[method.lower()](data)
    except KeyError as exc:
        raise ValueError("method must be 'bernoulli', 'poisson', or 'normal'") from exc
    return compare_estimates(ai_estimate, independent, tolerance=tolerance)
