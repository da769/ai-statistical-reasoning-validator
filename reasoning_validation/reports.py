"""Structured report construction for reasoning validation components."""

from collections.abc import Mapping
from typing import Any

from ._shared import distribution_name
from .models import CheckStatus, OverallStatus


def _numerical_section(result: Any) -> dict[str, Any]:
    if isinstance(result, Mapping):
        raw_status = result.get("validation_status", result.get("status"))
        valid = result.get("is_valid")
        evidence = dict(result)
    else:
        raw_status = getattr(result, "status", None)
        valid = getattr(result, "is_valid", None)
        evidence = {"result_type": type(result).__name__}
    if hasattr(raw_status, "value"):
        raw_status = raw_status.value
    if raw_status is None:
        status = "PASS" if valid is True else "FAIL" if valid is False else "INSUFFICIENT_INFORMATION"
    else:
        normalized = str(raw_status).upper()
        status = "PASS" if normalized in {"PASS", "VALID"} else \
                 "FAIL" if normalized in {"FAIL", "INVALID"} else "INSUFFICIENT_INFORMATION"
    return {"status": status, "evidence": evidence}


def _overall_status(statuses: list[str]) -> OverallStatus:
    if "FAIL" in statuses:
        return OverallStatus.FAIL
    if statuses and all(status in {"PASS", "NOT_APPLICABLE"} for status in statuses):
        return OverallStatus.PASS
    if not any(status in {"PASS", "WARNING"} for status in statuses):
        return OverallStatus.INSUFFICIENT_INFORMATION
    return OverallStatus.PARTIAL


def build_reasoning_report(distribution: str, *, method_selection, assumptions,
                           formula_validation, interpretation,
                           numerical_validation=None) -> dict[str, Any]:
    """Build a stable report from independently produced component results."""
    name = distribution_name(distribution)
    sections = {
        "method_selection": method_selection.as_dict(),
        "assumptions": assumptions.as_dict(),
        "formula_validation": formula_validation.as_dict(),
        "interpretation": interpretation.as_dict(),
    }
    statuses = [section["status"] for section in sections.values()]
    if numerical_validation is not None:
        sections["numerical_validation"] = _numerical_section(numerical_validation)
        statuses.append(sections["numerical_validation"]["status"])
    issues = []
    warnings = []
    limitations = [
        "Contextual assumptions are not established by sample observations alone.",
        "Formula checks use deterministic numerical probes and do not constitute symbolic proof.",
        "Natural-language descriptions and interpretations are not parsed.",
    ]
    for section_name, section in sections.items():
        for check in section.get("checks", []):
            item = {"component": section_name, **check}
            if check["status"] == CheckStatus.FAIL.value:
                issues.append(item)
            elif check["status"] == CheckStatus.WARNING.value:
                warnings.append(item)
        limitations.extend(section.get("limitations", []))
    return {"distribution": name, **sections,
            "overall_status": _overall_status(statuses).value,
            "issues": issues, "warnings": warnings,
            "limitations": list(dict.fromkeys(limitations))}
