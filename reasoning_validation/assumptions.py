"""Distribution-specific observable and contextual assumption checks."""

import math
from collections.abc import Mapping
from typing import Any

from ._shared import assumption_check, context_value, distribution_name, numeric_values, make_check
from .models import CheckStatus, ComponentResult, component_result


def validate_assumptions(distribution: str, data: Any = None, *,
                         assumptions: Mapping[str, Any] | None = None,
                         problem_context: Mapping[str, Any] | None = None) -> ComponentResult:
    """Check data-verifiable conditions and explicitly supplied assumptions."""
    name = distribution_name(distribution)
    values, error = numeric_values(data)
    checks = []
    allowed = {
        "bernoulli": {"independence", "constant_success_probability"},
        "poisson": {"constant_rate", "independence", "equidispersion"},
        "normal": {"normality", "independence"},
    }[name]
    if isinstance(assumptions, Mapping):
        unexpected = set(assumptions) - allowed
        if unexpected:
            checks.append(make_check("unexpected_assumptions", "Only assumptions for the selected model are supplied.",
                                     CheckStatus.FAIL, sorted(map(str, unexpected)),
                                     "Unknown or inapplicable assumption name(s) were supplied."))
    if assumptions is not None and not isinstance(assumptions, Mapping):
        checks.append(make_check("assumptions_structure", "Assumptions are supplied as a structured mapping.",
                                 CheckStatus.FAIL, type(assumptions).__name__,
                                 "assumptions must be a mapping keyed by assumption name."))
        assumptions = None

    def data_check(check_name, description, predicate, success, failure):
        if values is None:
            status = CheckStatus.INSUFFICIENT_INFORMATION if data is None else CheckStatus.FAIL
            checks.append(make_check(check_name, description, status, data,
                                      error or "Observations are unavailable.", data is None))
        else:
            valid = predicate(values)
            checks.append(make_check(check_name, description,
                                     CheckStatus.PASS if valid else CheckStatus.FAIL, values,
                                     success if valid else failure))

    if name == "bernoulli":
        data_check("binary_outcomes", "Each Bernoulli outcome is binary.",
                   lambda xs: all(x in (0.0, 1.0) for x in xs),
                   "The supplied observations are binary.", "At least one observation is not 0 or 1.")
        checks.extend([
            assumption_check("independence", "Bernoulli trials are independent.", assumptions),
            assumption_check("constant_success_probability", "The success probability is constant across trials.", assumptions),
        ])
    elif name == "poisson":
        data_check("count_data", "Observations are non-negative integer counts.",
                   lambda xs: all(x >= 0 and x.is_integer() for x in xs),
                   "The supplied observations are non-negative integer counts.",
                   "At least one observation is not a non-negative integer count.")
        checks.extend([
            assumption_check("constant_rate", "The event rate is constant over the observation interval.", assumptions),
            assumption_check("independence", "Counts from distinct observation units are independent.", assumptions),
        ])
        equidispersion = assumption_check(
            "equidispersion", "The Poisson model's population variance equals its population mean.", assumptions)
        if equidispersion.status is CheckStatus.INSUFFICIENT_INFORMATION and values:
            mean = math.fsum(values) / len(values)
            variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
            equidispersion = make_check(
                "equidispersion", equidispersion.description, equidispersion.status,
                {"sample_mean": mean, "sample_variance": variance},
                "Sample dispersion is shown as a diagnostic only; it cannot establish population equidispersion.", True)
        checks.append(equidispersion)
    else:
        outcome = context_value(problem_context, "outcome_type")
        if outcome is None:
            checks.append(make_check("continuous_data", "The response is meaningfully continuous.",
                                     CheckStatus.INSUFFICIENT_INFORMATION, None,
                                     "Numeric values alone cannot establish a continuous outcome; supply outcome_type.", True))
        elif str(outcome).lower() == "continuous":
            checks.append(make_check("continuous_data", "The response is meaningfully continuous.",
                                     CheckStatus.PASS, outcome,
                                     "The supplied context identifies a continuous response."))
        else:
            checks.append(make_check("continuous_data", "The response is meaningfully continuous.",
                                     CheckStatus.FAIL, outcome,
                                     "The supplied outcome type is not continuous."))
        checks.append(assumption_check("normality", "The Normal distribution is justified for the response.", assumptions))
        checks.append(assumption_check("independence", "Observations are independent where required.", assumptions))
        if values is None:
            checks.append(make_check("variance_validity", "The data support a positive Normal variance.",
                                     CheckStatus.INSUFFICIENT_INFORMATION if data is None else CheckStatus.FAIL,
                                     data, error or "Observations are unavailable.", data is None))
        else:
            mean = math.fsum(values) / len(values)
            variance = math.fsum((x - mean) ** 2 for x in values) / len(values)
            positive = variance > 0
            checks.append(make_check("variance_validity", "The data support a positive Normal variance.",
                                     CheckStatus.PASS if positive else CheckStatus.FAIL, variance,
                                     "Observed variance is positive." if positive else
                                     "Observed variance is zero; the ordinary Normal likelihood has no positive finite MLE variance."))
    return component_result(checks, limitations=(
        "Contextual assumptions recorded as true are declarations, not facts proven by the observations.",
        "Sample checks cannot establish independence, constant rates, or population distributional assumptions."))
