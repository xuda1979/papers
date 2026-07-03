#!/usr/bin/env python3
"""Validate coefficient-robustness certificates for chart Carter densities.

The checker proves a small but useful implication: if a verified chart density
has coercivity delta and a second quadratic density differs from it by bounded
coefficients, then the second density keeps a certified lower coercivity
constant. Cross terms are bounded by the exact inequality
|uv| <= (u^2 + v^2)/2.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_chart_certificate as chart_certificate
import verify_carter_sos_identity as sos


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_keys(obj: dict[str, Any], keys: list[str], path: str) -> None:
    for key in keys:
        require(key in obj, f"{path}.{key} is missing")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def resolve_path(raw_path: Any, base_dir: Path | None, path_name: str) -> Path:
    require(isinstance(raw_path, str) and raw_path.strip(), f"{path_name} must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path


def format_fraction(value: Fraction) -> str:
    return str(value)


def validate_metadata(data: dict[str, Any]) -> None:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status"], "metadata")
    require(
        metadata["claim_scope"] == "chart_coefficient_robustness",
        "metadata.claim_scope must be chart_coefficient_robustness",
    )


def variable_losses(
    variables: tuple[str, ...],
    coefficient_bounds: dict[str, Any],
) -> dict[str, Fraction]:
    losses = {variable: Fraction(0) for variable in variables}
    for monomial_key, raw_bound in coefficient_bounds.items():
        bound = sos.parse_fraction(raw_bound)
        require(bound >= 0, f"coefficient bound for {monomial_key!r} must be nonnegative")
        monomial = sos.parse_monomial_key(monomial_key, variables)
        require(sum(monomial) == 2, f"coefficient bound {monomial_key!r} must be quadratic")
        active = [index for index, power in enumerate(monomial) if power]
        if len(active) == 1:
            index = active[0]
            require(monomial[index] == 2, f"coefficient bound {monomial_key!r} must be quadratic")
            losses[variables[index]] += bound
        elif len(active) == 2:
            left, right = active
            require(
                monomial[left] == 1 and monomial[right] == 1,
                f"coefficient bound {monomial_key!r} must be bilinear",
            )
            losses[variables[left]] += bound / 2
            losses[variables[right]] += bound / 2
        else:
            raise ValueError(f"unsupported quadratic monomial {monomial_key!r}")
    return losses


def validate_control_variables(data: dict[str, Any], variables: tuple[str, ...]) -> list[str]:
    control = data.get("control_variables")
    require(isinstance(control, list) and control, "control_variables must be a nonempty list")
    require(all(isinstance(variable, str) for variable in control), "control_variables must be strings")
    require(set(control).issubset(set(variables)), "control_variables must be contained in chart variables")
    return control


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    validate_metadata(data)
    require_keys(
        data,
        ["base_chart_certificate_path", "control_variables", "coefficient_bounds", "target_final_delta"],
        "certificate",
    )

    base_path = resolve_path(data["base_chart_certificate_path"], base_dir, "base_chart_certificate_path")
    base_certificate = load_json(base_path)
    base_result = chart_certificate.verify(base_certificate, base_dir=base_path.parent)
    require("coercivity_delta" in base_result, "base chart certificate must record coercivity_delta")

    variables = tuple(base_result["variables"])
    control_variables = validate_control_variables(data, variables)
    coefficient_bounds = data["coefficient_bounds"]
    require(isinstance(coefficient_bounds, dict) and coefficient_bounds, "coefficient_bounds must be a nonempty object")

    losses = variable_losses(variables, coefficient_bounds)
    uncontrolled = sorted(set(variables) - set(control_variables))
    for variable in uncontrolled:
        require(losses[variable] == 0, f"uncontrolled variable {variable!r} has perturbation loss")

    worst_loss = max(losses[variable] for variable in control_variables)
    base_delta = Fraction(base_result["coercivity_delta"])
    final_delta = sos.parse_fraction(data["target_final_delta"])
    require(final_delta > 0, "target_final_delta must be positive")
    require(
        base_delta - worst_loss >= final_delta,
        "target_final_delta plus worst coefficient loss must fit inside base coercivity delta",
    )

    return {
        "accepted": True,
        "claim_scope": data["metadata"]["claim_scope"],
        "base_chart": base_result["chart"],
        "variables": list(variables),
        "base_coercivity_delta": format_fraction(base_delta),
        "target_final_delta": format_fraction(final_delta),
        "worst_coefficient_loss": format_fraction(worst_loss),
        "remaining_delta": format_fraction(base_delta - worst_loss),
        "variable_losses": {variable: format_fraction(losses[variable]) for variable in sorted(losses)},
    }


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path), base_dir=path.parent)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter coefficient robustness certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
