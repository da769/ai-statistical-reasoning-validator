"""Generation of structured MLE validation reports."""

from typing import Any

from validation.numerical_validation import ValidationResult, validate_mle


def generate_mle_report(method: str, data: Any, ai_estimate: Any, *, tolerance: float = 1e-8) -> dict[str, Any]:
    """Generate a structured report for an AI-generated MLE estimate."""
    result: ValidationResult = validate_mle(method, data, ai_estimate, tolerance=tolerance)
    return {"statistical_method": f"{method.title()} MLE", "ai_generated_estimate": result.ai_estimate,
            "independently_calculated_estimate": result.independent_estimate,
            "numerical_accuracy": {"absolute_error": result.absolute_error, "relative_error": result.relative_error,
                                    "tolerance": result.tolerance},
            "validation_status": "valid" if result.is_valid else "invalid"}
