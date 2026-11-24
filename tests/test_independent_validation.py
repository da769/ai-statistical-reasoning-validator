"""Tests for the independent numerical reference engine."""

import pytest

from estimators.mle import bernoulli as production_bernoulli
from estimators.mle import normal as production_normal
from estimators.mle import poisson as production_poisson
from reporting.report_generator import generate_independent_mle_report
from validation.numerical_validation import (ValidationStatus,
                                             compare_numeric_estimates,
                                             validate_independent_mle)
from validation.reference_calculations import (bernoulli_reference,
                                               normal_reference,
                                               poisson_reference)


def test_reference_calculations_cover_expected_results():
    assert bernoulli_reference([1, 1, 1, 1, 1, 1, 1, 0, 0, 0]).estimates == {"p": 0.7}
    assert poisson_reference([2, 3, 4, 5, 6]).estimates == {"lambda": 4.0}
    assert normal_reference([2, 4, 6, 8]).estimates == {"mean": 5.0, "variance": 5.0}


def test_reference_engine_does_not_call_production_estimators(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("production estimator called")
    monkeypatch.setattr(production_bernoulli, "estimate_bernoulli_mle", fail)
    monkeypatch.setattr(production_poisson, "estimate_poisson_mle", fail)
    monkeypatch.setattr(production_normal, "estimate_normal_mle", fail)
    assert validate_independent_mle("bernoulli", [1, 0], 0.5)["p"].status is ValidationStatus.PASS
    assert validate_independent_mle("poisson", [2, 3], 2.5)["lambda"].status is ValidationStatus.PASS
    assert all(result.status is ValidationStatus.PASS
               for result in validate_independent_mle("normal", [1, 2, 3], (2, 2 / 3)).values())


@pytest.mark.parametrize("method, data, estimate", [
    ("bernoulli", [0, 0], 0), ("bernoulli", [1, 1], 1),
    ("poisson", [0, 0, 0, 0], 0), ("normal", [4, 4, 4, 4], (4, 0)),
])
def test_boundary_estimates_pass(method, data, estimate):
    results = validate_independent_mle(method, data, estimate)
    assert all(result.status is ValidationStatus.PASS for result in results.values())


def test_failures_and_tolerance_controls():
    assert compare_numeric_estimates(0.65, 0.7).status is ValidationStatus.FAIL
    assert compare_numeric_estimates(0.70000001, 0.7,
                                     absolute_tolerance=1e-6, relative_tolerance=0).status is ValidationStatus.PASS
    assert compare_numeric_estimates(0.7001, 0.7,
                                     absolute_tolerance=1e-6, relative_tolerance=1e-6).status is ValidationStatus.FAIL


@pytest.mark.parametrize("ai, expected", [(None, 1), ("bad", 1), (float("nan"), 1), (float("inf"), 1)])
def test_invalid_numeric_inputs_are_invalid(ai, expected):
    assert compare_numeric_estimates(ai, expected).status is ValidationStatus.INVALID
    assert compare_numeric_estimates(1, 1, absolute_tolerance=-1).status is ValidationStatus.INVALID
    assert compare_numeric_estimates([1, 2], 1).status is ValidationStatus.INVALID


def test_missing_normal_parameter_and_unexpected_name_are_invalid():
    results = validate_independent_mle("normal", [1, 2, 3], {"mean": 2})
    assert results["variance"].status is ValidationStatus.INVALID
    assert validate_independent_mle("bernoulli", [1, 0], {"unexpected": 0.5})["p"].status is ValidationStatus.INVALID
    assert validate_independent_mle("bernoulli", [1, 0], {"p": 0.5, "extra": 1})["p"].status is ValidationStatus.INVALID


def test_report_contains_parameter_level_validation_details():
    report = generate_independent_mle_report("normal", [2, 4, 6, 8], {"mean": 5, "variance": 5})
    assert report["validation_status"] == "PASS"
    assert {item["parameter"] for item in report["parameters"]} == {"mean", "variance"}
    assert all(item["validation_status"] == "PASS" for item in report["parameters"])
