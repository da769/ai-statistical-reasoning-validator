"""Numerical equivalence checks for structured formula expression trees.

This module does not parse formula strings or prove identities symbolically.
It evaluates a small, safe expression language over deterministic probe data.
"""

import math
from collections.abc import Mapping
from typing import Any

from ._shared import distribution_name, numeric_values, make_check
from .models import CheckStatus, ComponentResult, component_result

_REQUIRED = ("likelihood", "log_likelihood", "score", "estimator")
_TOLERANCE = 1e-8


class FormulaError(ValueError):
    """A submitted expression is malformed or outside the supported language."""


def _evaluate(expression: Any, data: list[float], env: dict[str, float], x: float | None = None) -> float:
    if isinstance(expression, bool):
        raise FormulaError("Boolean values are not formula expressions")
    if isinstance(expression, (int, float)):
        result = float(expression)
    elif isinstance(expression, Mapping):
        if "var" in expression:
            name = expression["var"]
            if name == "x":
                if x is None:
                    raise FormulaError("x is only available inside sum/product")
                result = x
            elif name == "pi":
                result = math.pi
            elif name in env:
                result = env[name]
            else:
                raise FormulaError(f"unknown variable {name!r}")
        elif "const" in expression:
            result = float(expression["const"])
        else:
            operation = expression.get("op")
            args = expression.get("args", [])
            if operation in ("sum", "product"):
                child = expression.get("expr")
                if child is None:
                    raise FormulaError(f"{operation} requires an expr")
                evaluated = [_evaluate(child, data, env, value) for value in data]
                result = math.fsum(evaluated) if operation == "sum" else math.prod(evaluated)
            else:
                values = [_evaluate(arg, data, env, x) for arg in args]
                if operation == "add" and len(values) >= 2:
                    result = math.fsum(values)
                elif operation == "sub" and len(values) == 2:
                    result = values[0] - values[1]
                elif operation == "mul" and len(values) >= 2:
                    result = math.prod(values)
                elif operation == "div" and len(values) == 2:
                    result = values[0] / values[1]
                elif operation == "pow" and len(values) == 2:
                    result = values[0] ** values[1]
                elif operation == "neg" and len(values) == 1:
                    result = -values[0]
                elif operation == "log" and len(values) == 1:
                    result = math.log(values[0])
                elif operation == "exp" and len(values) == 1:
                    result = math.exp(values[0])
                elif operation == "sqrt" and len(values) == 1:
                    result = math.sqrt(values[0])
                elif operation == "factorial" and len(values) == 1:
                    result = float(math.factorial(int(values[0])))
                else:
                    raise FormulaError(f"unsupported operation or argument count: {operation!r}")
    else:
        raise FormulaError("expressions must be numbers or structured mappings")
    if not math.isfinite(result):
        raise FormulaError("expression produced a non-finite value")
    return result


def _probe_data(name: str, supplied: list[float]) -> list[list[float]]:
    if name == "bernoulli":
        probes = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0]]
    elif name == "poisson":
        probes = [[0.0, 1.0, 2.0], [2.0, 3.0], [0.0, 0.0, 1.0, 4.0]]
    else:
        probes = [[-1.0, 0.0, 2.0], [1.0, 2.0, 5.0], [-3.0, 1.0, 4.0, 7.0]]
    if supplied not in probes:
        probes.insert(0, supplied)
    return probes


def _reference(name: str, part: str, data: list[float], parameter: str | None = None) -> float:
    n = len(data)
    total = math.fsum(data)
    if name == "bernoulli":
        p = 0.37
        successes = total
        if part == "likelihood":
            return p ** successes * (1 - p) ** (n - successes)
        if part == "log_likelihood":
            return successes * math.log(p) + (n - successes) * math.log1p(-p)
        if part == "score":
            return successes / p - (n - successes) / (1 - p)
        return total / n
    if name == "poisson":
        rate = 1.7
        if part == "likelihood":
            return math.prod(math.exp(-rate) * rate ** x / math.factorial(int(x)) for x in data)
        if part == "log_likelihood":
            return math.fsum(x * math.log(rate) - rate - math.lgamma(x + 1) for x in data)
        if part == "score":
            return total / rate - n
        return total / n
    mean = 0.4
    variance = 1.3
    if part == "likelihood":
        return math.prod(math.exp(-((x - mean) ** 2) / (2 * variance)) /
                         math.sqrt(2 * math.pi * variance) for x in data)
    if part == "log_likelihood":
        return math.fsum(-0.5 * math.log(2 * math.pi * variance) -
                         (x - mean) ** 2 / (2 * variance) for x in data)
    if part == "score" and parameter == "mean":
        return math.fsum(x - mean for x in data) / variance
    if part == "score":
        return -n / (2 * variance) + math.fsum((x - mean) ** 2 for x in data) / (2 * variance ** 2)
    fitted_mean = total / n
    if parameter == "mean":
        return fitted_mean
    return math.fsum((x - fitted_mean) ** 2 for x in data) / n


def _environment(name: str, part: str, data: list[float], parameter: str | None = None) -> dict[str, float]:
    env = {"n": float(len(data)), "sum_x": math.fsum(data)}
    if name == "bernoulli":
        env["p"] = 0.37
    elif name == "poisson":
        env["lambda"] = 1.7
    elif part == "score":
        env.update({"mean": 0.4, "variance": 1.3})
    elif part == "likelihood" or part == "log_likelihood":
        env.update({"mean": 0.4, "variance": 1.3})
    else:
        fitted_mean = math.fsum(data) / len(data)
        env.update({"mean": fitted_mean,
                    "variance": math.fsum((x - fitted_mean) ** 2 for x in data) / len(data)})
    return env


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=_TOLERANCE, abs_tol=_TOLERANCE)


def validate_formula_claims(distribution: str, formulas: Mapping[str, Any] | None,
                             data: Any = None) -> ComponentResult:
    """Evaluate likelihood, log-likelihood, score derivation, and MLE AST claims.

    Formula input uses mappings such as ``{"op": "div", "args": [...]}``,
    ``{"var": "x"}``, ``{"op": "sum", "expr": ...}``, and ``{"op": "product", "expr": ...}``.
    Each submitted claim is checked at the supplied observations and at three
    fixed analytic probes. Passing is numerical agreement, not symbolic proof.
    """
    name = distribution_name(distribution)
    values, error = numeric_values(data)
    if values is None:
        return component_result([make_check("input", "Finite non-empty observations are supplied.",
                                            CheckStatus.INSUFFICIENT_INFORMATION if data is None else CheckStatus.FAIL,
                                            data, error or "Observations are unavailable.", data is None)])
    if name == "bernoulli" and any(x not in (0.0, 1.0) for x in values):
        return component_result([make_check("input", "Bernoulli observations are binary.", CheckStatus.FAIL,
                                            values, "Formula validation requires binary observations.")])
    if name == "poisson" and any(x < 0 or not x.is_integer() for x in values):
        return component_result([make_check("input", "Poisson observations are counts.", CheckStatus.FAIL,
                                            values, "Formula validation requires non-negative integer counts.")])
    if formulas is not None and not isinstance(formulas, Mapping):
        return component_result([make_check("formula_structure", "Formula claims use a structured mapping.",
                                            CheckStatus.FAIL, type(formulas).__name__,
                                            "Formula claims must be supplied as a mapping.")])
    if formulas is None:
        formulas = {}
    checks = []
    unexpected = set(formulas) - set(_REQUIRED)
    if unexpected:
        checks.append(make_check("unexpected_claims", "Only supported formula claims are supplied.",
                                 CheckStatus.FAIL, sorted(map(str, unexpected)),
                                 "Unsupported formula claim name(s) were supplied."))
    datasets = _probe_data(name, values)
    for part in _REQUIRED:
        claim = formulas.get(part)
        if claim is None:
            checks.append(make_check(part, f"Submitted {part.replace('_', ' ')} claim is independently checked.",
                                     CheckStatus.INSUFFICIENT_INFORMATION, None,
                                     f"No structured {part} expression was supplied.", True))
            continue
        evaluations = []
        try:
            if part in ("score", "estimator"):
                parameter_names = (("p",) if name == "bernoulli" else
                                   ("lambda",) if name == "poisson" else ("mean", "variance"))
                if not isinstance(claim, Mapping) or set(claim) != set(parameter_names):
                    raise FormulaError(f"{part} must map exactly {parameter_names}")
                for sample in datasets:
                    for parameter in parameter_names:
                        env = _environment(name, part, sample, parameter)
                        if part == "score":
                            env["p"] = 0.37 if name == "bernoulli" else env.get("p", 0.37)
                            env["lambda"] = 1.7 if name == "poisson" else env.get("lambda", 1.7)
                            env["mean"] = 0.4 if name == "normal" else env.get("mean", 0.4)
                            env["variance"] = 1.3 if name == "normal" else env.get("variance", 1.3)
                        actual = _evaluate(claim[parameter], sample, env)
                        expected = _reference(name, part, sample, parameter)
                        evaluations.append({"data": sample, "parameter": parameter,
                                            "actual": actual, "expected": expected,
                                            "matches": _close(actual, expected)})
            else:
                for sample in datasets:
                    env = _environment(name, part, sample)
                    actual = _evaluate(claim, sample, env)
                    expected = _reference(name, part, sample)
                    evaluations.append({"data": sample, "actual": actual,
                                        "expected": expected, "matches": _close(actual, expected)})
            passed = all(item["matches"] for item in evaluations)
            checks.append(make_check(part, f"Submitted {part.replace('_', ' ')} agrees with independent identities.",
                                     CheckStatus.PASS if passed else CheckStatus.FAIL, evaluations,
                                     "All deterministic probes agree." if passed else
                                     "At least one deterministic probe disagrees with the independent identity."))
        except (FormulaError, ArithmeticError, KeyError, TypeError, ValueError, OverflowError) as exc:
            checks.append(make_check(part, f"Submitted {part.replace('_', ' ')} is a supported structured expression.",
                                     CheckStatus.FAIL, {"error": str(exc)},
                                     "The formula could not be evaluated in the supported expression language."))
    return component_result(checks, limitations=(
        "Formula claims are checked numerically over deterministic probes; this is not a symbolic proof.",
        "The supported formula language is limited to the operators documented in README.md."))
