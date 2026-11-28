"""Structured interpretation checks; arbitrary prose is deliberately not parsed."""

from collections.abc import Mapping
from typing import Any

from ._shared import distribution_name, make_check
from .models import CheckStatus, ComponentResult, component_result

_EXPECTED = {
    "bernoulli": {"p": "probability_of_success"},
    "poisson": {"lambda": "expected_count_per_interval"},
    "normal": {"mean": "central_location", "variance": "dispersion_variance"},
}


def validate_interpretation(distribution: str, interpretations: Mapping[str, Any] | None) -> ComponentResult:
    """Validate structured parameter claims without evaluating natural language.

    Each parameter claim may contain ``concept``, ``quantity_type``,
    ``population_relation``, ``certainty``, and (for Poisson) ``observation_interval``.
    """
    name = distribution_name(distribution)
    if interpretations is not None and not isinstance(interpretations, Mapping):
        return component_result([make_check("interpretation_structure", "Interpretations use a structured parameter mapping.",
                                            CheckStatus.FAIL, type(interpretations).__name__,
                                            "Interpretations must be a mapping keyed by parameter name.")])
    claims = interpretations if isinstance(interpretations, Mapping) else {}
    checks = []
    for parameter, expected_concept in _EXPECTED[name].items():
        claim = claims.get(parameter)
        prefix = f"{parameter}_interpretation"
        if not isinstance(claim, Mapping):
            checks.append(make_check(prefix, f"The {parameter} interpretation is supplied and structured.",
                                     CheckStatus.INSUFFICIENT_INFORMATION, claim,
                                     "A structured interpretation for this parameter is required.", True))
            continue
        supplied_parameter = claim.get("parameter", parameter)
        checks.append(make_check(f"{prefix}:parameter", "The interpreted parameter is correctly identified.",
                                 CheckStatus.PASS if supplied_parameter == parameter else CheckStatus.FAIL,
                                 supplied_parameter,
                                 "Parameter name matches." if supplied_parameter == parameter else
                                 f"Expected parameter {parameter!r}."))
        concept = claim.get("concept")
        if concept is None:
            checks.append(make_check(f"{prefix}:concept", "The parameter meaning is correctly described.",
                                     CheckStatus.INSUFFICIENT_INFORMATION, None,
                                     f"Supply concept={expected_concept!r}.", True))
        else:
            okay = concept == expected_concept
            checks.append(make_check(f"{prefix}:concept", "The parameter meaning is correctly described.",
                                     CheckStatus.PASS if okay else CheckStatus.FAIL, concept,
                                     "The concept matches the model parameter." if okay else
                                     f"Expected {expected_concept!r}; concepts such as standard deviation or frequency are distinct."))
        relation = claim.get("population_relation")
        if relation is None:
            checks.append(make_check(f"{prefix}:population_relation", "An estimate is not presented as a known population truth.",
                                     CheckStatus.INSUFFICIENT_INFORMATION, None,
                                     "Supply population_relation='estimate_of_population_parameter'.", True))
        else:
            okay = relation == "estimate_of_population_parameter"
            checks.append(make_check(f"{prefix}:population_relation", "An estimate is not presented as a known population truth.",
                                     CheckStatus.PASS if okay else CheckStatus.FAIL, relation,
                                     "The claim is marked as an estimate." if okay else
                                     "An estimated value must not be described as the known true parameter."))
        certainty = claim.get("certainty")
        if certainty is None:
            checks.append(make_check(f"{prefix}:certainty", "The claim does not guarantee an individual future result.",
                                     CheckStatus.INSUFFICIENT_INFORMATION, None,
                                     "State whether this is an estimate rather than a guaranteed outcome.", True))
        else:
            okay = certainty == "estimate"
            checks.append(make_check(f"{prefix}:certainty", "The claim does not guarantee an individual future result.",
                                     CheckStatus.PASS if okay else CheckStatus.FAIL, certainty,
                                     "The claim is described as an estimate." if okay else
                                     "A fitted parameter does not guarantee a future observation."))
        if name == "bernoulli":
            quantity_type = claim.get("quantity_type")
            if quantity_type is None:
                checks.append(make_check(f"{prefix}:quantity_type", "Probability is distinguished from observed frequency.",
                                         CheckStatus.INSUFFICIENT_INFORMATION, None,
                                         "Supply quantity_type='probability'.", True))
            else:
                okay = quantity_type == "probability"
                checks.append(make_check(f"{prefix}:quantity_type", "Probability is distinguished from observed frequency.",
                                         CheckStatus.PASS if okay else CheckStatus.FAIL, quantity_type,
                                         "The estimate is identified as a probability." if okay else
                                         "A probability parameter is not itself a guaranteed count or observed frequency."))
        if name == "poisson":
            interval = claim.get("observation_interval")
            valid_interval = isinstance(interval, str) and bool(interval.strip())
            checks.append(make_check(f"{prefix}:observation_interval", "The Poisson rate's observation interval is defined.",
                                     CheckStatus.PASS if valid_interval else CheckStatus.INSUFFICIENT_INFORMATION,
                                     interval, "The interval is stated." if valid_interval else
                                     "Lambda is a rate per interval; supply the interval from the problem context.",
                                     not valid_interval))
    extras = set(claims) - set(_EXPECTED[name])
    if extras:
        checks.append(make_check("unexpected_parameters", "Only supported parameters are interpreted.",
                                 CheckStatus.FAIL, sorted(map(str, extras)),
                                 "Unexpected interpretation parameter name(s) were supplied."))
    return component_result(checks, limitations=(
        "This component validates structured interpretation fields only; it does not read arbitrary prose.",
        "A Poisson rate can be interpreted only relative to the supplied observation interval."))
