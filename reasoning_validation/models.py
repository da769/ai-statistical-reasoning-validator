"""Shared structured results for statistical reasoning checks."""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CheckStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class OverallStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    PARTIAL = "PARTIAL"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"


@dataclass(frozen=True)
class CheckResult:
    name: str
    description: str
    status: CheckStatus
    evidence: Any
    explanation: str
    requires_context: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "status": self.status.value, "evidence": self.evidence,
                "explanation": self.explanation,
                "requires_context": self.requires_context}


@dataclass(frozen=True)
class ComponentResult:
    status: CheckStatus
    checks: tuple[CheckResult, ...]
    limitations: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status.value,
                "checks": [check.as_dict() for check in self.checks],
                "limitations": list(self.limitations)}


def component_result(checks, *, limitations=()) -> ComponentResult:
    checks = tuple(checks)
    statuses = {check.status for check in checks}
    if CheckStatus.FAIL in statuses:
        status = CheckStatus.FAIL
    elif CheckStatus.INSUFFICIENT_INFORMATION in statuses:
        status = CheckStatus.INSUFFICIENT_INFORMATION
    elif CheckStatus.WARNING in statuses:
        status = CheckStatus.WARNING
    elif statuses and statuses <= {CheckStatus.PASS, CheckStatus.NOT_APPLICABLE}:
        status = CheckStatus.PASS
    else:
        status = CheckStatus.INSUFFICIENT_INFORMATION
    return ComponentResult(status, checks, tuple(limitations))
