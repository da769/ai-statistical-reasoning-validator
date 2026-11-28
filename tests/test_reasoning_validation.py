"""Tests for deterministic statistical reasoning and methodology validation."""

import pytest

from estimators.mle import bernoulli as production_bernoulli
from reasoning_validation import (
    CheckStatus,
    OverallStatus,
    validate_assumptions,
    validate_formula_claims,
    validate_interpretation,
    validate_method_selection,
    validate_reasoning,
)


def V(name):
    return {"var": name}


def O(name, *args):
    return {"op": name, "args": list(args)}


def R(name, expression):
    return {"op": name, "expr": expression}


def bernoulli_formulas():
    x, p, n = V("x"), V("p"), V("n")
    successes = R("sum", x)
    return {
        "likelihood": R("product", O("mul", O("pow", p, x),
                                        O("pow", O("sub", 1, p), O("sub", 1, x)))),
        "log_likelihood": R("sum", O("add", O("mul", x, O("log", p)),
                                         O("mul", O("sub", 1, x), O("log", O("sub", 1, p))))),
        "score": {"p": O("sub", O("div", successes, p),
                          O("div", O("sub", n, successes), O("sub", 1, p)))},
        "estimator": {"p": O("div", successes, n)},
    }


def poisson_formulas():
    x, rate, n = V("x"), V("lambda"), V("n")
    total = R("sum", x)
    factorial = O("factorial", x)
    return {
        "likelihood": R("product", O("mul", O("exp", O("neg", rate)),
                                        O("div", O("pow", rate, x), factorial))),
        "log_likelihood": R("sum", O("sub", O("sub", O("mul", x, O("log", rate)), rate),
                                              O("log", factorial))),
        "score": {"lambda": O("sub", O("div", total, rate), n)},
        "estimator": {"lambda": O("div", total, n)},
    }


def normal_formulas(variance_denominator=None):
    x, mean, variance, n = V("x"), V("mean"), V("variance"), V("n")
    residual = O("sub", x, mean)
    squared = O("pow", residual, 2)
    normalizer = O("sqrt", O("mul", O("mul", 2, V("pi")), variance))
    log_normalizer = O("neg", O("mul", 0.5, O("log", O("mul", O("mul", 2, V("pi")), variance))))
    variance_score = O("add", O("div", O("neg", n), O("mul", 2, variance)),
                       O("div", R("sum", squared), O("mul", 2, O("pow", variance, 2))))
    denominator = variance_denominator if variance_denominator is not None else n
    return {
        "likelihood": R("product", O("div", O("exp", O("neg", O("div", squared, O("mul", 2, variance)))),
                                        normalizer)),
        "log_likelihood": R("sum", O("sub", log_normalizer,
                                         O("div", squared, O("mul", 2, variance)))),
        "score": {"mean": O("div", R("sum", residual), variance), "variance": variance_score},
        "estimator": {"mean": O("div", R("sum", x), n),
                      "variance": O("div", R("sum", O("pow", O("sub", x, V("mean")), 2)),
                                     denominator)},
    }


def test_method_selection_uses_context_and_validates_supported_methods():
    result = validate_method_selection(
        "bernoulli", [0, 1, 1],
        problem_context={"outcome_type": "binary", "distribution": "bernoulli", "parameters": ["p"]},
        assumptions={"independence": {"satisfied": True, "evidence": "separate trials"}},
    )
    assert result.status is CheckStatus.PASS
    assert all(check.status is CheckStatus.PASS for check in result.checks)


def test_method_selection_fails_for_wrong_distribution_and_binary_normal_context():
    result = validate_method_selection("normal", [0, 1, 0], problem_context={
        "outcome_type": "binary", "distribution": "normal", "parameters": ["mean", "variance"]})
    assert result.status is CheckStatus.FAIL
    assert next(c for c in result.checks if c.name == "outcome_suitability").status is CheckStatus.FAIL


def test_count_data_is_not_accepted_as_bernoulli_data():
    result = validate_method_selection("bernoulli", [0, 2, 4], problem_context={
        "outcome_type": "count", "distribution": "bernoulli", "parameters": ["p"]})
    assert next(c for c in result.checks if c.name == "data_format").status is CheckStatus.FAIL
    assert next(c for c in result.checks if c.name == "outcome_suitability").status is CheckStatus.FAIL


def test_method_selection_reports_insufficient_context_instead_of_assuming_fit():
    result = validate_method_selection("poisson", [0, 1, 2])
    assert result.status is CheckStatus.INSUFFICIENT_INFORMATION
    assert {c.status for c in result.checks} & {CheckStatus.INSUFFICIENT_INFORMATION}


def test_method_selection_rejects_wrong_parameter_identification():
    result = validate_method_selection("normal", [1, 2, 4], problem_context={
        "outcome_type": "continuous", "distribution": "normal", "parameters": ["mean", "standard_deviation"]})
    check = next(c for c in result.checks if c.name == "parameter_identification")
    assert check.status is CheckStatus.FAIL


def test_assumptions_distinguish_observable_binary_data_from_independence():
    result = validate_assumptions("bernoulli", [0, 1, 1])
    checks = {check.name: check for check in result.checks}
    assert checks["binary_outcomes"].status is CheckStatus.PASS
    assert checks["independence"].status is CheckStatus.INSUFFICIENT_INFORMATION
    assert checks["constant_success_probability"].status is CheckStatus.INSUFFICIENT_INFORMATION
    assert checks["independence"].requires_context


def test_stated_but_unsupported_assumption_is_warning_and_explicit_violation_fails():
    warning = validate_assumptions("bernoulli", [0, 1], assumptions={"independence": True})
    assert next(c for c in warning.checks if c.name == "independence").status is CheckStatus.WARNING
    failed = validate_assumptions("bernoulli", [0, 1], assumptions={"independence": False})
    assert next(c for c in failed.checks if c.name == "independence").status is CheckStatus.FAIL


def test_poisson_requires_rate_independence_and_equidispersion_context():
    result = validate_assumptions("poisson", [0, 1, 2, 3])
    checks = {check.name: check for check in result.checks}
    assert checks["count_data"].status is CheckStatus.PASS
    assert checks["constant_rate"].status is CheckStatus.INSUFFICIENT_INFORMATION
    assert checks["independence"].status is CheckStatus.INSUFFICIENT_INFORMATION
    assert checks["equidispersion"].status is CheckStatus.INSUFFICIENT_INFORMATION
    assert "sample_mean" in checks["equidispersion"].evidence
    assert "cannot establish" in checks["equidispersion"].explanation


def test_poisson_supplied_assumptions_are_recorded_with_context():
    assumptions = {name: {"satisfied": True, "evidence": "specified in study design"}
                   for name in ("constant_rate", "independence", "equidispersion")}
    result = validate_assumptions("poisson", [0, 2, 4], assumptions=assumptions)
    assert all(c.status is CheckStatus.PASS for c in result.checks)


def test_normality_is_not_inferred_and_zero_variance_is_flagged():
    result = validate_assumptions("normal", [5, 5, 5], problem_context={"outcome_type": "continuous"})
    checks = {check.name: check for check in result.checks}
    assert checks["continuous_data"].status is CheckStatus.PASS
    assert checks["normality"].status is CheckStatus.INSUFFICIENT_INFORMATION
    assert checks["variance_validity"].status is CheckStatus.FAIL


def test_normality_with_explicit_context_and_positive_variance():
    result = validate_assumptions("normal", [1, 2, 4],
                                 assumptions={"normality": {"satisfied": True, "evidence": "diagnostic review"},
                                              "independence": {"satisfied": True, "evidence": "study design"}},
                                 problem_context={"outcome_type": "continuous"})
    assert all(check.status is CheckStatus.PASS for check in result.checks)


def test_correct_bernoulli_likelihood_log_likelihood_score_and_mle():
    result = validate_formula_claims("bernoulli", bernoulli_formulas(), [1, 0, 1])
    assert result.status is CheckStatus.PASS


def test_incorrect_bernoulli_formula_is_detected():
    formulas = bernoulli_formulas()
    formulas["estimator"] = {"p": O("div", R("sum", V("x")), O("add", V("n"), 1))}
    result = validate_formula_claims("bernoulli", formulas, [1, 0, 1])
    assert next(c for c in result.checks if c.name == "estimator").status is CheckStatus.FAIL


def test_correct_poisson_likelihood_log_likelihood_score_and_mle():
    result = validate_formula_claims("poisson", poisson_formulas(), [0, 2, 3])
    assert result.status is CheckStatus.PASS


def test_incorrect_poisson_likelihood_is_detected():
    formulas = poisson_formulas()
    formulas["likelihood"] = R("product", O("mul", O("exp", O("neg", V("lambda"))),
                                               O("pow", V("lambda"), V("x"))))
    result = validate_formula_claims("poisson", formulas, [0, 2, 3])
    assert next(c for c in result.checks if c.name == "likelihood").status is CheckStatus.FAIL


def test_correct_normal_mle_and_equivalent_mean_expression_pass():
    formulas = normal_formulas()
    formulas["estimator"]["mean"] = O("mul", R("sum", V("x")), O("div", 1, V("n")))
    result = validate_formula_claims("normal", formulas, [1, 2, 5])
    assert result.status is CheckStatus.PASS


def test_normal_unbiased_variance_denominator_n_minus_one_is_detected():
    formulas = normal_formulas(O("sub", V("n"), 1))
    result = validate_formula_claims("normal", formulas, [1, 2, 5])
    assert next(c for c in result.checks if c.name == "estimator").status is CheckStatus.FAIL


def test_formula_validation_is_independent_of_production_estimator():
    formulas = bernoulli_formulas()
    formulas["estimator"] = {"p": O("div", R("sum", V("x")), O("add", V("n"), 3))}
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(production_bernoulli, "estimate_bernoulli_mle", lambda data: 0.5)
    try:
        result = validate_formula_claims("bernoulli", formulas, [1, 0, 1])
    finally:
        monkeypatch.undo()
    assert next(c for c in result.checks if c.name == "estimator").status is CheckStatus.FAIL


def test_missing_formula_claims_are_insufficient_not_pass():
    result = validate_formula_claims("normal", {}, [1, 2, 3])
    assert result.status is CheckStatus.INSUFFICIENT_INFORMATION
    assert all(check.status is CheckStatus.INSUFFICIENT_INFORMATION for check in result.checks)


def test_interpretation_accepts_structured_bernoulli_estimate():
    claim = {"p": {"concept": "probability_of_success", "quantity_type": "probability",
                    "population_relation": "estimate_of_population_parameter", "certainty": "estimate"}}
    assert validate_interpretation("bernoulli", claim).status is CheckStatus.PASS


def test_interpretation_detects_probability_frequency_and_guarantee_confusion():
    claim = {"p": {"concept": "probability_of_success", "quantity_type": "frequency",
                    "population_relation": "known_population_truth", "certainty": "guaranteed_future"}}
    result = validate_interpretation("bernoulli", claim)
    assert result.status is CheckStatus.FAIL
    assert any(check.status is CheckStatus.FAIL for check in result.checks)


def test_interpretation_detects_variance_standard_deviation_confusion():
    claims = {
        "mean": {"concept": "central_location", "population_relation": "estimate_of_population_parameter", "certainty": "estimate"},
        "variance": {"concept": "standard_deviation", "population_relation": "estimate_of_population_parameter", "certainty": "estimate"},
    }
    result = validate_interpretation("normal", claims)
    assert next(c for c in result.checks if c.name == "variance_interpretation:concept").status is CheckStatus.FAIL


def test_poisson_interpretation_requires_rate_interval_and_expected_count_meaning():
    missing = validate_interpretation("poisson", {"lambda": {
        "concept": "expected_count_per_interval", "population_relation": "estimate_of_population_parameter", "certainty": "estimate"}})
    check = next(c for c in missing.checks if c.name == "lambda_interpretation:observation_interval")
    assert check.status is CheckStatus.INSUFFICIENT_INFORMATION
    correct = validate_interpretation("poisson", {"lambda": {
        "concept": "expected_count_per_interval", "observation_interval": "per 1 hour",
        "population_relation": "estimate_of_population_parameter", "certainty": "estimate"}})
    assert correct.status is CheckStatus.PASS


def complete_normal_reasoning():
    return validate_reasoning(
        "normal", [1, 2, 5],
        problem_context={"outcome_type": "continuous", "distribution": "normal", "parameters": ["mean", "variance"]},
        assumptions={"normality": {"satisfied": True, "evidence": "diagnostics supplied"},
                     "independence": {"satisfied": True, "evidence": "study design supplied"}},
        formulas=normal_formulas(),
        interpretations={
            "mean": {"concept": "central_location", "population_relation": "estimate_of_population_parameter", "certainty": "estimate"},
            "variance": {"concept": "dispersion_variance", "population_relation": "estimate_of_population_parameter", "certainty": "estimate"},
        })


def test_orchestrator_emits_complete_consistent_report():
    report = complete_normal_reasoning()
    assert report["overall_status"] == OverallStatus.PASS.value
    assert report["distribution"] == "normal"
    assert {key for key in ("method_selection", "assumptions", "formula_validation", "interpretation")} <= report.keys()
    assert report["issues"] == []
    assert report["limitations"]


def test_numeric_pass_does_not_override_methodology_failure():
    report = validate_reasoning("normal", [0, 1, 0],
                                problem_context={"outcome_type": "binary", "distribution": "normal",
                                                 "parameters": ["mean", "variance"]},
                                numerical_validation={"validation_status": "PASS", "is_valid": True})
    assert report["numerical_validation"]["status"] == "PASS"
    assert report["overall_status"] == OverallStatus.FAIL.value
    assert report["issues"]


def test_missing_context_and_claims_produce_insufficient_information():
    report = validate_reasoning("normal", [1, 2, 3])
    assert report["overall_status"] == OverallStatus.INSUFFICIENT_INFORMATION.value
    assert report["formula_validation"]["status"] == CheckStatus.INSUFFICIENT_INFORMATION.value
    assert report["interpretation"]["status"] == CheckStatus.INSUFFICIENT_INFORMATION.value


def test_existing_validation_namespace_reexports_reasoning_components():
    from validation.interpretation_validation import validate_interpretation as validation_interpretation
    from validation.methodology_validation import (
        validate_assumptions as validation_assumptions,
        validate_formula_claims as validation_formulas,
        validate_method_selection as validation_method_selection,
    )
    from reasoning_validation import validate_assumptions as package_assumptions
    from reasoning_validation import validate_formula_claims as package_formulas
    from reasoning_validation import validate_interpretation as package_interpretation
    from reasoning_validation import validate_method_selection as package_method_selection

    assert validation_assumptions is package_assumptions
    assert validation_formulas is package_formulas
    assert validation_method_selection is package_method_selection
    assert validation_interpretation is package_interpretation

def test_malformed_structured_containers_are_explicit_failures():
    assert validate_formula_claims("bernoulli", "not a mapping", [0, 1]).status is CheckStatus.FAIL
    assert validate_interpretation("bernoulli", ["not", "a", "mapping"]).status is CheckStatus.FAIL
    assert validate_method_selection("normal", [0, 1], problem_context="free text").status is CheckStatus.FAIL
    assert validate_assumptions("normal", [0, 1], assumptions="independent").status is CheckStatus.FAIL
    assert validate_assumptions("normal", [0, 1], assumptions={"indepdence": True}).status is CheckStatus.FAIL

def test_invalid_distribution_and_malformed_formula_are_explicit():
    with pytest.raises(ValueError, match="distribution"):
        validate_reasoning("gamma", [1, 2])
    malformed = validate_formula_claims("bernoulli", {"likelihood": "p^x"}, [0, 1])
    assert next(c for c in malformed.checks if c.name == "likelihood").status is CheckStatus.FAIL
