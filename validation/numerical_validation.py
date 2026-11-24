"""Independent numerical comparison and MLE validation functions."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np

from .reference_calculations import bernoulli_reference, normal_reference, poisson_reference


class ValidationStatus(str, Enum):
    """Possible outcomes of numerical validation."""

    PASS = "PASS"
    FAIL = "FAIL"
    INVALID = "INVALID"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class ValidationResult:
    """Outcome of comparing an AI estimate with an independent estimate."""

    ai_estimate: Any
    independent_estimate: Any
    absolute_error: Any
    relative_error: Any
    tolerance: float
    is_valid: bool
    status: ValidationStatus = ValidationStatus.INVALID
    explanation: str = ""
    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-8

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a JSON-friendly dictionary."""
        return {"ai_estimate": self.ai_estimate, "independent_estimate": self.independent_estimate,
                "absolute_error": self.absolute_error, "relative_error": self.relative_error,
                "tolerance": self.tolerance, "absolute_tolerance": self.absolute_tolerance,
                "relative_tolerance": self.relative_tolerance, "is_valid": self.is_valid,
                "validation_status": self.status.value, "explanation": self.explanation}


def _invalid(message: str, ai: Any = None, expected: Any = None,
             absolute_tolerance: float = 1e-8, relative_tolerance: float = 1e-8) -> ValidationResult:
    """Create a consistent invalid result."""
    return ValidationResult(ai, expected, None, None, absolute_tolerance, False,
                            ValidationStatus.INVALID, message, absolute_tolerance,
                            relative_tolerance)


def compare_numeric_estimates(ai_estimate: Any, independent_estimate: Any, *,
                              absolute_tolerance: float = 1e-8,
                              relative_tolerance: float = 1e-8) -> ValidationResult:
    """Compare numeric estimates and return PASS, FAIL, or INVALID."""
    try:
        if (not isfinite(float(absolute_tolerance)) or absolute_tolerance < 0
                or not isfinite(float(relative_tolerance)) or relative_tolerance < 0):
            return _invalid("Absolute and relative tolerances must be finite and non-negative.",
                            ai_estimate, independent_estimate, float(absolute_tolerance),
                            float(relative_tolerance))
    except (TypeError, ValueError):
        return _invalid("Absolute and relative tolerances must be numeric.", ai_estimate,
                        independent_estimate)
    if ai_estimate is None or independent_estimate is None:
        return _invalid("Both estimates are required.", ai_estimate, independent_estimate,
                        absolute_tolerance, relative_tolerance)
    try:
        ai = np.asarray(ai_estimate, dtype=float)
        expected = np.asarray(independent_estimate, dtype=float)
    except (TypeError, ValueError):
        return _invalid("Estimates must be numeric.", ai_estimate, independent_estimate,
                        absolute_tolerance, relative_tolerance)
    if ai.shape != expected.shape:
        return _invalid("Estimates must have matching shapes.", ai_estimate, independent_estimate,
                        absolute_tolerance, relative_tolerance)
    if not np.all(np.isfinite(ai)) or not np.all(np.isfinite(expected)):
        return _invalid("Estimates must be finite; NaN and infinity are not valid.",
                        ai_estimate, independent_estimate, absolute_tolerance, relative_tolerance)
    absolute = np.abs(ai - expected)
    relative = np.divide(absolute, np.abs(expected), out=np.zeros_like(absolute), where=expected != 0)
    passed = bool(np.all(np.isclose(ai, expected, atol=absolute_tolerance, rtol=relative_tolerance)))
    status = ValidationStatus.PASS if passed else ValidationStatus.FAIL
    explanation = ("The estimates agree within the configured numerical tolerances."
                   if passed else "The AI-generated estimate differs beyond the configured numerical tolerances.")

    def clean(value: np.ndarray) -> Any:
        return float(value) if value.ndim == 0 else value.tolist()

    return ValidationResult(clean(ai), clean(expected), clean(absolute), clean(relative),
                            absolute_tolerance, passed, status, explanation,
                            absolute_tolerance, relative_tolerance)


def compare_estimates(ai_estimate: Any, independent_estimate: Any, *, tolerance: float = 1e-8) -> ValidationResult:
    """Backward-compatible strict comparison using one tolerance for both metrics."""
    result = compare_numeric_estimates(ai_estimate, independent_estimate,
                                       absolute_tolerance=tolerance, relative_tolerance=tolerance)
    if result.status is ValidationStatus.INVALID and result.explanation in {
            "Estimates must have matching shapes.",
            "Absolute and relative tolerances must be finite and non-negative.",
            "Absolute and relative tolerances must be numeric."}:
        raise ValueError(result.explanation)
    return result


def _reference(method: str, data: Any):
    """Dispatch to an independent reference calculation."""
    try:
        return {"bernoulli": bernoulli_reference, "poisson": poisson_reference,
                "normal": normal_reference}[method.lower()](data)
    except (AttributeError, KeyError) as exc:
        raise ValueError("method must be 'bernoulli', 'poisson', or 'normal'") from exc


def validate_independent_mle(method: str, data: Any, ai_estimate: Any, *,
                             absolute_tolerance: float = 1e-8,
                             relative_tolerance: float = 1e-8) -> dict[str, ValidationResult]:
    """Validate each parameter against an independent MLE reference result."""
    reference = _reference(method, data)
    if isinstance(ai_estimate, dict):
        submitted = ai_estimate
    elif method.lower() == "normal":
        try:
            submitted_values = list(ai_estimate)
            submitted = (dict(zip(("mean", "variance"), submitted_values))
                         if len(submitted_values) == 2 else {})
        except (TypeError, ValueError):
            submitted = {}
    else:
        submitted = {next(iter(reference.estimates)): ai_estimate}
    unexpected = set(submitted) - set(reference.estimates)
    results = {name: compare_numeric_estimates(submitted.get(name), expected,
                                                absolute_tolerance=absolute_tolerance,
                                                relative_tolerance=relative_tolerance)
               for name, expected in reference.estimates.items()}
    if unexpected:
        message = "Unexpected parameter name(s): " + ", ".join(sorted(unexpected)) + "."
        for name in results:
            results[name] = _invalid(message, submitted.get(name), reference.estimates[name],
                                     absolute_tolerance, relative_tolerance)
    return results


def validate_mle(method: str, data: Any, ai_estimate: Any, *, tolerance: float = 1e-8) -> ValidationResult:
    """Backward-compatible validation returning one result for an MLE call."""
    results = validate_independent_mle(method, data, ai_estimate,
                                       absolute_tolerance=tolerance, relative_tolerance=tolerance)
    if len(results) == 1:
        return next(iter(results.values()))
    values = list(results.values())
    passed = all(result.status is ValidationStatus.PASS for result in values)
    return ValidationResult(ai_estimate,
                            {key: result.independent_estimate for key, result in results.items()},
                            {key: result.absolute_error for key, result in results.items()},
                            {key: result.relative_error for key, result in results.items()},
                            tolerance, passed, ValidationStatus.PASS if passed else ValidationStatus.FAIL,
                            "All Normal parameters agree within tolerance." if passed else
                            "One or more Normal parameters differ beyond tolerance.", tolerance, tolerance)
