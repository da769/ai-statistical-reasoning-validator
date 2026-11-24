import math

import numpy as np
import pytest

from estimators.mle.bernoulli import (
    bernoulli_likelihood, bernoulli_log_likelihood, estimate_bernoulli_mle,
)
from estimators.mle.normal import estimate_normal_mle
from estimators.mle.poisson import (
    estimate_poisson_mle, poisson_likelihood, poisson_log_likelihood,
)
from reporting.report_generator import generate_mle_report
from validation.numerical_validation import compare_estimates, validate_mle


def test_bernoulli_mle_and_likelihoods():
    data = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    assert estimate_bernoulli_mle(data) == pytest.approx(0.7)
    assert bernoulli_likelihood(data, 0.7) == pytest.approx(0.7**7 * 0.3**3)
    assert bernoulli_log_likelihood(data, 0.7) == pytest.approx(math.log(0.7**7 * 0.3**3))


@pytest.mark.parametrize("data, expected", [([0, 0], 0.0), ([1, 1], 1.0)])
def test_bernoulli_boundary_mles(data, expected):
    assert estimate_bernoulli_mle(data) == expected


def test_poisson_mle_and_likelihoods():
    data = [0, 1, 2, 3]
    assert estimate_poisson_mle(data) == pytest.approx(1.5)
    expected_log = sum(x * math.log(1.5) - 1.5 - math.lgamma(x + 1) for x in data)
    assert poisson_log_likelihood(data, 1.5) == pytest.approx(expected_log)
    assert poisson_likelihood(data, 1.5) == pytest.approx(math.exp(expected_log))


def test_poisson_zero_rate_boundary():
    assert poisson_log_likelihood([0, 0], 0) == 0
    assert poisson_likelihood([0, 1], 0) == 0


def test_normal_mle_uses_n_denominator():
    mean, variance = estimate_normal_mle([1, 2, 3, 4])
    assert mean == pytest.approx(2.5)
    assert variance == pytest.approx(1.25)


@pytest.mark.parametrize("function, data", [
    (estimate_bernoulli_mle, []), (estimate_poisson_mle, []), (estimate_normal_mle, []),
])
def test_empty_data_rejected(function, data):
    with pytest.raises(ValueError):
        function(data)


@pytest.mark.parametrize("function, data", [
    (estimate_bernoulli_mle, [0, 2]), (estimate_poisson_mle, [1, -1]),
    (estimate_poisson_mle, [1, 1.5]),
])
def test_invalid_observations_rejected(function, data):
    with pytest.raises(ValueError):
        function(data)


def test_validation_precision_and_report():
    result = compare_estimates(1.0, 1.0 + 1e-10, tolerance=1e-8)
    assert result.is_valid
    assert validate_mle("normal", [1, 2, 3], [2.0, 2 / 3]).is_valid
    report = generate_mle_report("bernoulli", [1, 0], 0.5)
    assert report["validation_status"] == "valid"
    assert report["statistical_method"] == "Bernoulli MLE"


def test_validation_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        compare_estimates([1, 2], 1)
