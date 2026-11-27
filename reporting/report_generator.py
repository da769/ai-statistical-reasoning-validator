"""Generation of structured MLE validation reports."""

from typing import Any

from validation.numerical_validation import ValidationResult, validate_independent_mle, validate_mle


def generate_mle_report(method: str, data: Any, ai_estimate: Any, *, tolerance: float = 1e-8) -> dict[str, Any]:
    """Generate a structured report for an AI-generated MLE estimate."""
    result: ValidationResult = validate_mle(method, data, ai_estimate, tolerance=tolerance)
    return {"statistical_method": f"{method.title()} MLE", "ai_generated_estimate": result.ai_estimate,
            "independently_calculated_estimate": result.independent_estimate,
            "numerical_accuracy": {"absolute_error": result.absolute_error, "relative_error": result.relative_error,
                                    "tolerance": result.tolerance},
            "validation_status": "valid" if result.is_valid else "invalid"}


def generate_independent_mle_report(method: str, data: Any, ai_estimate: Any, *,
                                    absolute_tolerance: float = 1e-8,
                                    relative_tolerance: float = 1e-8) -> dict[str, Any]:
    """Generate parameter-level reports using independent reference calculations."""
    results = validate_independent_mle(method, data, ai_estimate,
                                       absolute_tolerance=absolute_tolerance,
                                       relative_tolerance=relative_tolerance)
    parameters = []
    for parameter, result in results.items():
        parameters.append({"distribution": method.lower(), "parameter": parameter,
                           "ai_estimate": result.ai_estimate,
                           "reference_estimate": result.independent_estimate,
                           "absolute_error": result.absolute_error,
                           "relative_error": result.relative_error,
                           "absolute_tolerance": absolute_tolerance,
                           "relative_tolerance": relative_tolerance,
                           "validation_status": result.status.value,
                           "explanation": result.explanation})
    statuses = {item["validation_status"] for item in parameters}
    overall = "INVALID" if "INVALID" in statuses else "PASS" if statuses == {"PASS"} else "FAIL"
    return {"distribution": method.lower(), "parameters": parameters,
            "validation_status": overall}
