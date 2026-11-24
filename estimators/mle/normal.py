"""Normal-distribution maximum-likelihood estimation."""

import math
from collections.abc import Iterable

import numpy as np

from ._utils import observations, parameter


def estimate_normal_mle(data: Iterable[float]) -> tuple[float, float]:
    """Return ``(mu, variance)`` using the Normal MLE variance denominator ``n``."""
    values = observations(data)
    mean = float(np.mean(values))
    variance = float(np.mean((values - mean) ** 2))
    return mean, variance


def normal_log_likelihood(data: Iterable[float], mean: float, variance: float) -> float:
    """Return the Normal log-likelihood for a mean and population variance."""
    values = observations(data)
    mu = parameter(mean, name="mean")
    var = parameter(variance, name="variance")
    if var <= 0:
        raise ValueError("variance must be positive for a Normal likelihood")
    return float(-0.5 * values.size * math.log(2 * math.pi * var)
                 - np.sum((values - mu) ** 2) / (2 * var))


def normal_likelihood(data: Iterable[float], mean: float, variance: float) -> float:
    """Return the Normal likelihood for a mean and population variance."""
    return float(np.exp(normal_log_likelihood(data, mean, variance)))
