"""Poisson maximum-likelihood estimation."""

import math
from collections.abc import Iterable

import numpy as np
from scipy.special import gammaln

from ._utils import observations, parameter


def estimate_poisson_mle(data: Iterable[float]) -> float:
    """Return the Poisson MLE for the rate parameter, the sample mean."""
    values = observations(data)
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("Poisson observations must be non-negative integers")
    return float(np.mean(values))


def poisson_log_likelihood(data: Iterable[float], rate: float) -> float:
    """Return ``log L(rate)`` for independent Poisson observations."""
    values = observations(data)
    lam = parameter(rate, name="rate")
    if lam < 0:
        raise ValueError("rate must be non-negative")
    if np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("Poisson observations must be non-negative integers")
    if lam == 0:
        return 0.0 if np.all(values == 0) else float("-inf")
    return float(np.sum(values * math.log(lam) - lam - gammaln(values + 1)))


def poisson_likelihood(data: Iterable[float], rate: float) -> float:
    """Return the Poisson likelihood ``L(rate)`` for the observations."""
    return float(np.exp(poisson_log_likelihood(data, rate)))
