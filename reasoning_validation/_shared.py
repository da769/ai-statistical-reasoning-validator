"""Private deterministic helpers shared by reasoning components."""

import math
from collections.abc import Mapping
from typing import Any

from .models import CheckResult, CheckStatus

DISTRIBUTIONS = {"bernoulli", "poisson", "normal"}


def distribution_name(value: str) -> str:
    if not isinstance(value, str) or value.lower() not in DISTRIBUTIONS:
        raise ValueError("distribution must be 'bernoulli', 'poisson', or 'normal'")
    return value.lower()


def numeric_values(data: Any) -> tuple[list[float] | None, str | None]:
    if data is None:
        return None, "observations were not supplied"
    if isinstance(data, (str, bytes)):
        return None, "observations must be a sequence of numeric values"
    try:
        values = [float(value) for value in data]
    except (TypeError, ValueError, OverflowError):
        return None, "observations must contain only numeric values"
    if not values:
        return None, "observations must not be empty"
    if not all(math.isfinite(value) for value in values):
        return None, "observations must be finite"
    return values, None


def make_check(name: str, description: str, status: CheckStatus,
               evidence: Any, explanation: str, requires_context: bool = False) -> CheckResult:
    return CheckResult(name, description, status, evidence, explanation, requires_context)


def context_value(context: Mapping[str, Any] | None, *names: str) -> Any:
    if not isinstance(context, Mapping):
        return None
    for name in names:
        if name in context:
            return context[name]
    return None


def assumption_check(name: str, description: str, assumptions: Mapping[str, Any] | None) -> CheckResult:
    """Assess a declared contextual assumption without treating data as proof."""
    if not isinstance(assumptions, Mapping) or name not in assumptions or assumptions[name] is None:
        return make_check(name, description, CheckStatus.INSUFFICIENT_INFORMATION, None,
                          "This contextual assumption was not supplied.", True)
    supplied = assumptions[name]
    if isinstance(supplied, Mapping):
        value = supplied.get("satisfied", supplied.get("value"))
        evidence = supplied.get("evidence", supplied.get("justification"))
    else:
        value, evidence = supplied, None
    if not isinstance(value, bool):
        return make_check(name, description, CheckStatus.INSUFFICIENT_INFORMATION, supplied,
                          "Supply a Boolean assumption and, where possible, supporting context.", True)
    if not value:
        return make_check(name, description, CheckStatus.FAIL, evidence,
                          "The supplied information says this assumption is not satisfied.", False)
    if evidence:
        return make_check(name, description, CheckStatus.PASS, evidence,
                          "The assumption was stated with supporting context; this is not independent proof.", False)
    return make_check(name, description, CheckStatus.WARNING, "assumption stated true; no justification supplied",
                      "The assumption was stated but not supported. Sample values alone cannot establish it.", True)
