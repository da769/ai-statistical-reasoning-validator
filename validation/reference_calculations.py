"""Independent reference calculations for supported MLE distributions."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any


@dataclass(frozen=True)
class ReferenceCalculation:
    """Result of an independent reference calculation."""

    distribution: str
    parameter_estimates: dict[str, float]
    calculation_method: str
    numerical_precision: dict[str, Any]

    @property
    def estimates(self) -> dict[str, float]:
        """Return independently calculated parameter estimates."""
        return self.parameter_estimates

    def as_dict(self) -> dict[str, Any]:
        """Return the calculation as a serializable dictionary."""
        return {"distribution": self.distribution, "parameter_estimates": self.parameter_estimates.copy(),
                "calculation_method": self.calculation_method,
                "numerical_precision": self.numerical_precision.copy()}


def _values(data: Iterable[float]) -> list[float]:
    """Convert data to a non-empty finite list without estimator helpers."""
    if isinstance(data, (str, bytes)):
        raise TypeError("data must be an iterable of numbers")
    try:
        values = [float(value) for value in data]
    except (TypeError, ValueError) as exc:
        raise ValueError("data must contain only numeric values") from exc
    if not values:
        raise ValueError("data must be non-empty")
    if any(not isfinite(value) for value in values):
        raise ValueError("data must contain only finite values")
    return values


def bernoulli_reference(data: Iterable[float]) -> ReferenceCalculation:
    """Calculate Bernoulli ``p`` as successes divided by observations."""
    values = _values(data)
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("Bernoulli observations must be 0 or 1")
    estimate = sum(value == 1.0 for value in values) / len(values)
    return ReferenceCalculation("bernoulli", {"p": float(estimate)},
                                "count successes / count observations",
                                {"arithmetic": "integer count divided by n", "variance_denominator": None})


def poisson_reference(data: Iterable[float]) -> ReferenceCalculation:
    """Calculate Poisson ``lambda`` as the arithmetic mean."""
    values = _values(data)
    if any(value < 0 or value != int(value) for value in values):
        raise ValueError("Poisson observations must be non-negative integers")
    estimate = fsum(values) / len(values)
    return ReferenceCalculation("poisson", {"lambda": float(estimate)},
                                "arithmetic sum / count", {"arithmetic": "math.fsum divided by n"})


def normal_reference(data: Iterable[float]) -> ReferenceCalculation:
    """Calculate Normal mean and MLE variance using denominator ``n``."""
    values = _values(data)
    mean = fsum(values) / len(values)
    variance = fsum((value - mean) ** 2 for value in values) / len(values)
    return ReferenceCalculation("normal", {"mean": float(mean), "variance": float(variance)},
                                "arithmetic mean and sum of squared deviations / n",
                                {"arithmetic": "math.fsum", "variance_denominator": "n"})


reference_bernoulli_mle = bernoulli_reference
reference_poisson_mle = poisson_reference
reference_normal_mle = normal_reference
