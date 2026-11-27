"""Deterministic stress tests for independent MLE validation."""

import math
import random

import pytest

from estimators.mle import _utils
from estimators.mle import bernoulli as production_bernoulli
from estimators.mle import normal as production_normal
from estimators.mle import poisson as production_poisson
from reporting.report_generator import generate_independent_mle_report
from validation.numerical_validation import (
    ValidationStatus,
    compare_numeric_estimates,
    validate_independent_mle,
)
from validation.reference_calculations import (
    bernoulli_reference,
    normal_reference,
    poisson_reference,
)


@pytest.mark.parametrize("n", [1, 2, 17, 10_000])
def test_bernoulli_reference_matches_count_identity_for_varied_sizes(n):
    data = [1 if i % 7 == 0 else 0 for i in range(n)]
    expected = sum(data) / n
    reference = bernoulli_reference(data)
    assert reference.estimates["p"] == expected
    assert validate_independent_mle("bernoulli", data, {"p": expected})["p"].status is ValidationStatus.PASS


@pytest.mark.parametrize("data, expected", [([0], 0.0), ([1], 1.0), ([0, 1], 0.5), ([0] * 2000, 0.0)])
def test_bernoulli_boundary_and_sample_size(data, expected):
    assert bernoulli_reference(data).estimates["p"] == expected


@pytest.mark.parametrize("data, expected", [([0], 0.0), ([0, 0, 0], 0.0), ([3], 3.0), ([0, 2, 4], 2.0)])
def test_poisson_reference_small_and_zero_rate_samples(data, expected):
    assert poisson_reference(data).estimates["lambda"] == expected


def test_reference_handles_large_finite_poisson_observations():
    result = poisson_reference([1e300, 1e300])
    assert result.estimates["lambda"] == 1e300
    assert math.isfinite(result.estimates["lambda"])


@pytest.mark.parametrize("data, mean, variance", [([4], 4.0, 0.0), ([1, 3], 2.0, 1.0), ([1, 2, 3, 4], 2.5, 1.25)])
def test_normal_reference_small_samples_and_zero_variance(data, mean, variance):
    assert normal_reference(data).estimates == {"mean": mean, "variance": variance}


def test_normal_reference_reports_overflow_for_unrepresentable_variance():
    with pytest.raises(OverflowError):
        normal_reference([-1.7e308, 1.7e308])


def test_subnormal_difference_from_zero_is_not_accepted_at_zero_tolerance():
    result = compare_numeric_estimates(0.0, 5e-324, absolute_tolerance=0.0, relative_tolerance=0.0)
    assert result.status is ValidationStatus.FAIL
    assert result.is_valid is False

def test_normal_reference_large_finite_values_remains_finite():
    result = normal_reference([-1e150, 1e150])
    assert result.estimates["mean"] == 0.0
    assert result.estimates["variance"] == pytest.approx(1e300, rel=1e-15)
    assert all(math.isfinite(value) for value in result.estimates.values())


@pytest.mark.parametrize("method, data", [
    ("bernoulli", []), ("bernoulli", [0, 1.0000000000000002]),
    ("poisson", []), ("poisson", [-1]), ("poisson", [1.5]),
    ("normal", []),
])
def test_reference_rejects_empty_or_out_of_domain_data(method, data):
    reference = {"bernoulli": bernoulli_reference, "poisson": poisson_reference,
                 "normal": normal_reference}[method]
    with pytest.raises(ValueError):
        reference(data)


@pytest.mark.parametrize("bad_data", [None, "123", [None], ["x"], [float("nan")],
                                      [float("inf")], [-float("inf")]])
def test_reference_rejects_missing_malformed_and_nonfinite_data(bad_data):
    with pytest.raises((TypeError, ValueError)):
        bernoulli_reference(bad_data)


@pytest.mark.parametrize("method, data, estimate", [
    ("bernoulli", [0, 1], {"p": 0.5}),
    ("poisson", [0, 2, 4], {"lambda": 2.0}),
    ("normal", [-2, 0, 2], {"mean": 0.0, "variance": 8 / 3}),
])
def test_fixed_seed_randomized_integer_samples_against_independent_identities(method, data, estimate):
    rng = random.Random(20261128)
    for _ in range(30):
        size = rng.randint(1, 80)
        if method == "bernoulli":
            sample = [rng.randrange(2) for _ in range(size)]
            expected = {"p": sum(sample) / size}
        elif method == "poisson":
            sample = [rng.randrange(25) for _ in range(size)]
            expected = {"lambda": math.fsum(sample) / size}
        else:
            sample = [rng.randrange(-100, 101) for _ in range(size)]
            mean = math.fsum(sample) / size
            expected = {"mean": mean, "variance": math.fsum((x - mean) ** 2 for x in sample) / size}
        actual = {"bernoulli": bernoulli_reference, "poisson": poisson_reference,
                  "normal": normal_reference}[method](sample).estimates
        assert actual == pytest.approx(expected)
        outcomes = validate_independent_mle(method, sample, expected)
        assert all(item.status is ValidationStatus.PASS for item in outcomes.values())


def test_tolerance_policy_at_and_just_beyond_absolute_boundary():
    boundary = compare_numeric_estimates(0.125, 0.0, absolute_tolerance=0.125, relative_tolerance=0)
    outside = compare_numeric_estimates(math.nextafter(0.125, math.inf), 0.0,
                                        absolute_tolerance=0.125, relative_tolerance=0)
    assert boundary.status is ValidationStatus.PASS
    assert outside.status is ValidationStatus.FAIL


def test_tolerance_policy_uses_relative_tolerance_for_nonzero_reference():
    assert compare_numeric_estimates(110.0, 100.0, absolute_tolerance=0,
                                     relative_tolerance=0.1).status is ValidationStatus.PASS
    assert compare_numeric_estimates(math.nextafter(110.0, math.inf), 100.0, absolute_tolerance=0,
                                     relative_tolerance=0.1).status is ValidationStatus.FAIL


@pytest.mark.parametrize("estimate", [None, "bad", [], [1, 2], float("nan"), float("inf"), -float("inf")])
def test_invalid_estimates_never_pass_or_validate(estimate):
    result = compare_numeric_estimates(estimate, 1.0)
    assert result.status is ValidationStatus.INVALID
    assert result.is_valid is False


def test_validation_selects_distribution_specific_reference():
    bernoulli = validate_independent_mle("bernoulli", [1, 1, 0], {"p": 2 / 3})
    poisson = validate_independent_mle("poisson", [1, 1, 0], {"lambda": 2 / 3})
    assert set(bernoulli) == {"p"}
    assert set(poisson) == {"lambda"}
    assert bernoulli["p"].status is ValidationStatus.PASS
    assert poisson["lambda"].status is ValidationStatus.PASS

def test_invalid_reference_input_names_and_parameters_are_explicit():
    with pytest.raises(ValueError, match="method"):
        validate_independent_mle("gamma", [1, 2], 1.5)
    result = validate_independent_mle("normal", [1, 2], {"mean": 1.5})
    assert result["variance"].status is ValidationStatus.INVALID
    unexpected = validate_independent_mle("poisson", [1, 3], {"rate": 2})
    assert unexpected["lambda"].status is ValidationStatus.INVALID


def test_reference_remains_independent_when_estimators_and_shared_input_helper_are_disabled(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("reference path called production code")
    monkeypatch.setattr(production_bernoulli, "estimate_bernoulli_mle", fail)
    monkeypatch.setattr(production_poisson, "estimate_poisson_mle", fail)
    monkeypatch.setattr(production_normal, "estimate_normal_mle", fail)
    monkeypatch.setattr(_utils, "observations", fail)
    assert bernoulli_reference([1, 0]).estimates == {"p": 0.5}
    assert poisson_reference([1, 3]).estimates == {"lambda": 2.0}
    assert normal_reference([1, 3]).estimates == {"mean": 2.0, "variance": 1.0}


def test_parameter_and_aggregate_reports_preserve_invalid_status():
    report = generate_independent_mle_report("normal", [1, 2, 3], {"mean": 2})
    assert report["validation_status"] == "INVALID"
    assert {item["parameter"]: item["validation_status"] for item in report["parameters"]} == {
        "mean": "PASS", "variance": "INVALID"}



