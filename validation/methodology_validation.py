"""Compatibility exports for reasoning checks in the validation namespace."""

from reasoning_validation.assumptions import validate_assumptions
from reasoning_validation.formulas import validate_formula_claims
from reasoning_validation.method_selection import validate_method_selection

__all__ = ["validate_assumptions", "validate_formula_claims", "validate_method_selection"]
