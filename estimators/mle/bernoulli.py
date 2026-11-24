"""Bernoulli maximum-likelihood estimation."""

import math
from collections.abc import Iterable

import numpy as np

from ._utils import observations, parameter


def estimate_bernoulli_mle(data: Iterable[float]) -> float:
    """Return the Bernoulli MLE, the sample proportion of ones."""
    values = observations(data)
    if not np.all(np.isin(values, (0.0, 1.0))):
        raise ValueError("Bernoulli observations must be 0 or 1")
    return float(np.mean(values))


def bernoulli_log_likelihood(data: Iterable[float], p: float) -> float:
    """Return ``log L(p)`` for independent Bernoulli observations.

    At the boundary, impossible observations correctly produce ``-inf``.
    """
    values = observations(data)
    probability = parameter(p, name="p")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("p must be between 0 and 1")
    if not np.all(np.isin(values, (0.0, 1.0))):
        raise ValueError("Bernoulli observations must be 0 or 1")
    successes = int(np.sum(values))
    failures = values.size - successes
    if (probability == 0 and successes) or (probability == 1 and failures):
        return float("-inf")
    result = (successes * math.log(probability) if successes else 0.0)
    result += (failures * math.log1p(-probability) if failures else 0.0)
    return float(result)


def bernoulli_likelihood(data: Iterable[float], p: float) -> float:
    """Return the Bernoulli likelihood ``L(p)`` for the observations."""
    log_likelihood = bernoulli_log_likelihood(data, p)
    return float(np.exp(log_likelihood))
