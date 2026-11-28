"""Deterministic checks of whether a claimed MLE family fits supplied context."""

from collections.abc import Mapping
from typing import Any

from ._shared import assumption_check, context_value, distribution_name, numeric_values
from .models import CheckStatus, ComponentResult, component_result
from ._shared import make_check


_EXPECTED_OUTCOME = {"bernoulli": "binary", "poisson": "count", "normal": "continuous"}
_EXPECTED_PARAMETERS = {"bernoulli": {"p"}, "poisson": {"lambda"},
                        "normal": {"mean", "variance"}}


def validate_method_selection(distribution: str, data: Any = None, *,
                              problem_context: Mapping[str, Any] | None = None,
                              assumptions: Mapping[str, Any] | None = None) -> ComponentResult:
    """Validate observable data constraints and structured problem context.

    Free-text descriptions are retained as evidence but are not interpreted.
    Use structured keys such as ``outcome_type``, ``distribution``,
    ``distribution_justification``, and ``parameters`` for deterministic checks.
    """
    name = distribution_name(distribution)
    values, error = numeric_values(data)
    checks = []
    if problem_context is not None and not isinstance(problem_context, Mapping):
        checks.append(make_check("problem_context_structure", "Problem context is a structured mapping.",
                                 CheckStatus.FAIL, type(problem_context).__name__,
                                 "problem_context must be a mapping of structured context fields."))
        problem_context = None
    if assumptions is not None and not isinstance(assumptions, Mapping):
        checks.append(make_check("assumptions_structure", "Assumptions are supplied as a structured mapping.",
                                 CheckStatus.FAIL, type(assumptions).__name__,
                                 "assumptions must be a mapping keyed by assumption name."))
        assumptions = None
    if values is None:
        status = CheckStatus.INSUFFICIENT_INFORMATION if data is None else CheckStatus.FAIL
        checks.append(make_check("data_format", "Observations satisfy the selected model's data requirements.",
                                 status, data, error or "Observations are unavailable.", data is None))
    elif name == "bernoulli":
        valid = all(value in (0.0, 1.0) for value in values)
        checks.append(make_check("data_format", "Bernoulli observations are binary.",
                                 CheckStatus.PASS if valid else CheckStatus.FAIL, values,
                                 "All values are binary." if valid else "Bernoulli observations must be 0 or 1."))
    elif name == "poisson":
        valid = all(value >= 0 and value.is_integer() for value in values)
        checks.append(make_check("data_format", "Poisson observations are non-negative integer counts.",
                                 CheckStatus.PASS if valid else CheckStatus.FAIL, values,
                                 "All values are non-negative integer counts." if valid else
                                 "Poisson observations must be non-negative integers."))
    else:
        checks.append(make_check("data_format", "Normal observations are finite numeric values.",
                                 CheckStatus.PASS, values,
                                 "Numeric format is valid; this alone does not establish continuous-data suitability."))

    outcome = context_value(problem_context, "outcome_type")
    if outcome is None:
        checks.append(make_check("outcome_suitability", "The outcome type suits the selected distribution.",
                                 CheckStatus.INSUFFICIENT_INFORMATION, None,
                                 "Supply outcome_type as binary, count, or continuous.", True))
    elif str(outcome).lower() != _EXPECTED_OUTCOME[name]:
        checks.append(make_check("outcome_suitability", "The outcome type suits the selected distribution.",
                                 CheckStatus.FAIL, outcome,
                                 f"{name} is not supported by the stated {outcome!r} outcome type."))
    else:
        checks.append(make_check("outcome_suitability", "The outcome type suits the selected distribution.",
                                 CheckStatus.PASS, outcome,
                                 "The supplied structured outcome type matches the model family."))

    stated_distribution = context_value(problem_context, "distribution", "stated_distribution")
    justification = context_value(problem_context, "distribution_justification")
    if stated_distribution is not None:
        matches = isinstance(stated_distribution, str) and stated_distribution.lower() == name
        checks.append(make_check("model_consistency", "The problem context supports the selected model.",
                                 CheckStatus.PASS if matches else CheckStatus.FAIL, stated_distribution,
                                 "The stated model matches." if matches else
                                 f"The problem states {stated_distribution!r}, not {name!r}."))
    elif isinstance(justification, str) and justification.strip():
        checks.append(make_check("model_consistency", "The problem context supports the selected model.",
                                 CheckStatus.WARNING, justification,
                                 "A justification was supplied but was not interpreted or independently verified.", True))
    else:
        checks.append(make_check("model_consistency", "The problem context supports the selected model.",
                                 CheckStatus.INSUFFICIENT_INFORMATION,
                                 context_value(problem_context, "problem_description"),
                                 "No structured model statement or justification was supplied. Free text is not parsed.", True))

    supplied_parameters = context_value(problem_context, "parameters")
    if supplied_parameters is None:
        checks.append(make_check("parameter_identification", "The model parameters are correctly identified.",
                                 CheckStatus.INSUFFICIENT_INFORMATION, None,
                                 "Supply the parameter names associated with the proposed method.", True))
    else:
        try:
            actual = {str(item).lower() for item in supplied_parameters}
        except TypeError:
            actual = set()
        expected = _EXPECTED_PARAMETERS[name]
        valid = actual == expected
        checks.append(make_check("parameter_identification", "The model parameters are correctly identified.",
                                 CheckStatus.PASS if valid else CheckStatus.FAIL, sorted(actual),
                                 f"Expected parameters: {sorted(expected)}."))

    checks.append(assumption_check("independence", "Observations are independent or independence is justified.", assumptions))
    limits = ("Data format alone cannot establish that a distribution is appropriate.",
              "Problem descriptions are not interpreted as natural language.")
    return component_result(checks, limitations=limits)
