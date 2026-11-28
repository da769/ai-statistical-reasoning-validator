"""Run targeted pytest nodes against source mutations in temporary copies.

Only temporary copies are changed. The repository source and tests are never
modified by this runner, and TemporaryDirectory cleans up each mutant copy.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTATIONS = [
    {
        "name": "wrong estimator denominator",
        "path": "estimators/mle/bernoulli.py",
        "old": "return float(np.mean(values))",
        "new": "return float(np.sum(values) / (values.size + 1))",
        "test": "tests/test_mle.py::test_bernoulli_mle_and_likelihoods",
    },
    {
        "name": "incorrect boundary estimate",
        "path": "estimators/mle/bernoulli.py",
        "old": "return float(np.mean(values))",
        "new": "return float(np.mean(values)) if np.mean(values) != 1.0 else 0.999",
        "test": "tests/test_mle.py::test_bernoulli_boundary_mles",
    },
    {
        "name": "incorrect distribution/reference selection",
        "path": "validation/numerical_validation.py",
        "old": '"bernoulli": bernoulli_reference',
        "new": '"bernoulli": normal_reference',
        "test": "tests/test_validation_stress.py::test_validation_selects_distribution_specific_reference",
    },
    {
        "name": "bypassed input validation",
        "path": "estimators/mle/bernoulli.py",
        "old": 'if not np.all(np.isin(values, (0.0, 1.0))):\n        raise ValueError("Bernoulli observations must be 0 or 1")',
        "new": 'if False:\n        raise ValueError("Bernoulli observations must be 0 or 1")',
        "test": "tests/test_mle.py::test_invalid_observations_rejected",
    },
    {
        "name": "incorrect reference-value calculation",
        "path": "validation/reference_calculations.py",
        "old": 'estimate = sum(value == 1.0 for value in values) / len(values)',
        "new": 'estimate = (sum(value == 1.0 for value in values) + 1) / len(values)',
        "test": "tests/test_independent_validation.py::test_reference_calculations_cover_expected_results",
    },
    {
        "name": "tolerance accepts a value that should fail",
        "path": "validation/numerical_validation.py",
        "old": 'passed = bool(np.all(np.isclose(ai, expected, atol=absolute_tolerance, rtol=relative_tolerance)))',
        "new": 'passed = True',
        "test": "tests/test_validation_stress.py::test_tolerance_policy_at_and_just_beyond_absolute_boundary",
    },
    {
        "name": "nonfinite estimate is not rejected",
        "path": "validation/numerical_validation.py",
        "old": 'if not np.all(np.isfinite(ai)) or not np.all(np.isfinite(expected)):',
        "new": 'if False:',
        "test": "tests/test_validation_stress.py::test_invalid_estimates_never_pass_or_validate",
    },
    {
        "name": "report status detached from parameter outcome",
        "path": "reporting/report_generator.py",
        "old": 'overall = "INVALID" if "INVALID" in statuses else "PASS" if statuses == {"PASS"} else "FAIL"',
        "new": 'overall = "PASS"',
        "test": "tests/test_validation_stress.py::test_parameter_and_aggregate_reports_preserve_invalid_status",
    },
]


def make_copy(destination):
    for directory in ("estimators", "validation", "reporting", "tests"):
        shutil.copytree(ROOT / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))


def run_mutation(mutation):
    with tempfile.TemporaryDirectory(prefix="aisrv_mutation_") as temporary:
        project = Path(temporary)
        make_copy(project)
        source = project / mutation["path"]
        original = source.read_text(encoding="utf-8")
        if mutation["old"] not in original:
            return {"mutation": mutation["name"], "test": mutation["test"],
                    "result": "NOT APPLICABLE", "detail": "expected source fragment was not found"}
        source.write_text(original.replace(mutation["old"], mutation["new"], 1), encoding="utf-8")
        command = [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", mutation["test"]]
        completed = subprocess.run(command, cwd=project, capture_output=True, text=True)
        output = completed.stdout + completed.stderr
        summary = output.strip().splitlines()
        target_failure = next((line.strip() for line in summary
                               if line.strip().startswith(f"FAILED {mutation['test']}")), None)
        if completed.returncode == 0:
            result = "SURVIVED"
            detail = "mapped pytest node passed"
        elif target_failure:
            result = "KILLED"
            detail = target_failure
        else:
            result = "HARNESS ERROR"
            detail = summary[-1] if summary else f"pytest exit code {completed.returncode}"
        return {"mutation": mutation["name"], "test": mutation["test"], "result": result,
                "detail": detail}


def main():
    results = [run_mutation(mutation) for mutation in MUTATIONS]
    print(json.dumps(results, indent=2))
    return 1 if any(item["result"] != "KILLED" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
