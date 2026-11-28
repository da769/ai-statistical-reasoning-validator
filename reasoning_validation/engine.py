"""Reasoning orchestration across independent validation components."""

from collections.abc import Mapping
from typing import Any

from ._shared import distribution_name
from .assumptions import validate_assumptions
from .formulas import validate_formula_claims
from .interpretation import validate_interpretation
from .method_selection import validate_method_selection
from .reports import build_reasoning_report


def validate_reasoning(distribution: str, data: Any = None, *,
                       problem_context: Mapping[str, Any] | None = None,
                       assumptions: Mapping[str, Any] | None = None,
                       formulas: Mapping[str, Any] | None = None,
                       interpretations: Mapping[str, Any] | None = None,
                       numerical_validation: Any = None) -> dict[str, Any]:
    """Run deterministic reasoning checks and return an integrated report.

    ``problem_context`` uses structured fields (outcome_type, distribution,
    distribution_justification, parameters). Free-text descriptions are retained
    as context but are not evaluated. Formula claims use the documented AST.
    """
    name = distribution_name(distribution)
    method = validate_method_selection(name, data, problem_context=problem_context, assumptions=assumptions)
    assumption_results = validate_assumptions(name, data, assumptions=assumptions,
                                              problem_context=problem_context)
    formula_results = validate_formula_claims(name, formulas, data)
    interpretation_results = validate_interpretation(name, interpretations)
    return build_reasoning_report(name, method_selection=method, assumptions=assumption_results,
                                  formula_validation=formula_results,
                                  interpretation=interpretation_results,
                                  numerical_validation=numerical_validation)
