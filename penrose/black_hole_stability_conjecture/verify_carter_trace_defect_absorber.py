#!/usr/bin/env python3
"""Validate positive/ideal defect absorbers for Carter trace densities.

The checker proves a useful relaxation of exact source matching.  A candidate
principal density may differ from the accepted trace density by a finite sum of
nonnegative square defects plus terms in a phase-free trapped/gauge constraint
ideal.  On the common zero set of those constraints the candidate is therefore
bounded below by the trace density, which already inherits the local Carter
cone margin.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_principal_symbol_trace as trace
import verify_carter_sos_identity as sos


CLAIM_SCOPE = "chart_trace_defect_absorber"
ALLOWED_DERIVATION_STATUSES = {
    "candidate_defect_absorber",
    "kerr_gravitational_proved",
}


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


def validate_metadata(data: dict[str, Any]) -> str:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status", "derivation_status"], "metadata")
    require(metadata["claim_scope"] == CLAIM_SCOPE, f"metadata.claim_scope must be {CLAIM_SCOPE}")
    derivation_status = metadata["derivation_status"]
    require(
        derivation_status in ALLOWED_DERIVATION_STATUSES,
        f"metadata.derivation_status must be one of {sorted(ALLOWED_DERIVATION_STATUSES)}",
    )
    if derivation_status == "kerr_gravitational_proved":
        raise ValueError(
            "metadata.derivation_status kerr_gravitational_proved requires an exact "
            "Kerr principal-symbol derivation; this checker verifies only the "
            "positive/ideal defect absorber algebra"
        )
    return str(derivation_status)


def validate_string_list(raw: Any, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    require(isinstance(raw, list), f"{name} must be a list")
    require(allow_empty or bool(raw), f"{name} must be nonempty")
    require(all(isinstance(item, str) and item for item in raw), f"{name} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{name} entries must be distinct")
    return tuple(raw)


def phase_degree(monomial: tuple[int, ...], phase_variables: tuple[str, ...]) -> int:
    return sum(monomial[index] for index in range(len(phase_variables)))


def validate_phase_homogeneous(
    poly: sos.Polynomial,
    phase_variables: tuple[str, ...],
    degree: int,
    name: str,
) -> None:
    for monomial in poly.terms:
        require(phase_degree(monomial, phase_variables) == degree, f"{name} must have phase degree {degree}")


def validate_phase_free(poly: sos.Polynomial, phase_variables: tuple[str, ...], name: str) -> None:
    for monomial in poly.terms:
        require(phase_degree(monomial, phase_variables) == 0, f"{name} must be phase-free")


def trace_variables(trace_data: dict[str, Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    raw = trace_data.get("trace_variables")
    require(isinstance(raw, dict), "trace certificate trace_variables must be an object")
    phase_variables = validate_string_list(raw.get("phase_variables"), "trace_variables.phase_variables")
    symbolic_variables = validate_string_list(
        raw.get("symbolic_coefficient_variables"),
        "trace_variables.symbolic_coefficient_variables",
    )
    source_auxiliary_variables = validate_string_list(
        raw.get("source_auxiliary_variables", []),
        "trace_variables.source_auxiliary_variables",
        allow_empty=True,
    )
    return phase_variables, symbolic_variables, source_auxiliary_variables


def trace_density_polynomial(
    trace_data: dict[str, Any],
    variables: tuple[str, ...],
    phase_variables: tuple[str, ...],
) -> sos.Polynomial:
    environment, _ = trace.load_formulas(trace_data, variables, phase_variables)
    density = trace.eval_trace_expr(trace_data.get("density_trace"), variables, environment)
    validate_phase_homogeneous(density, phase_variables, 2, "trace density")
    return density


def ideal_polynomial(
    raw_terms: Any,
    variables: tuple[str, ...],
    phase_variables: tuple[str, ...],
) -> tuple[sos.Polynomial, list[str]]:
    require(isinstance(raw_terms, list), "constraint_ideal_terms must be a list")
    total = sos.Polynomial.zero(variables)
    checked: list[str] = []
    for index, term in enumerate(raw_terms):
        require(isinstance(term, dict), f"constraint_ideal_terms[{index}] must be an object")
        require_keys(term, ["name", "constraint_expression", "multiplier_expression"], f"constraint_ideal_terms[{index}]")
        name = term["name"]
        require(isinstance(name, str) and name, f"constraint_ideal_terms[{index}].name must be nonempty")
        constraint = sos.eval_expr(term["constraint_expression"], variables)
        multiplier = sos.eval_expr(term["multiplier_expression"], variables)
        validate_phase_free(constraint, phase_variables, f"constraint {name!r}")
        total = total + constraint * multiplier
        checked.append(name)
    return total, checked


def nonnegative_defect_polynomial(
    raw_terms: Any,
    variables: tuple[str, ...],
    phase_variables: tuple[str, ...],
) -> tuple[sos.Polynomial, list[str]]:
    require(isinstance(raw_terms, list), "nonnegative_defect_terms must be a list")
    total = sos.Polynomial.zero(variables)
    checked: list[str] = []
    for index, term in enumerate(raw_terms):
        require(isinstance(term, dict), f"nonnegative_defect_terms[{index}] must be an object")
        require_keys(term, ["name", "coefficient", "square_expression"], f"nonnegative_defect_terms[{index}]")
        name = term["name"]
        require(isinstance(name, str) and name, f"nonnegative_defect_terms[{index}].name must be nonempty")
        coefficient = sos.parse_fraction(term["coefficient"])
        require(coefficient >= 0, f"nonnegative defect {name!r} coefficient must be nonnegative")
        linear = sos.eval_expr(term["square_expression"], variables)
        validate_phase_homogeneous(linear, phase_variables, 1, f"nonnegative defect {name!r} square_expression")
        total = total + (linear * linear).scale(coefficient)
        checked.append(name)
    return total, checked


def verify_defect_identity(
    data: dict[str, Any],
    trace_data: dict[str, Any],
    variables: tuple[str, ...],
    phase_variables: tuple[str, ...],
) -> tuple[list[str], list[str], list[str]]:
    require_keys(
        data,
        ["candidate_density_expression", "constraint_ideal_terms", "nonnegative_defect_terms"],
        "certificate",
    )
    candidate = sos.eval_expr(data["candidate_density_expression"], variables)
    validate_phase_homogeneous(candidate, phase_variables, 2, "candidate_density_expression")
    traced = trace_density_polynomial(trace_data, variables, phase_variables)
    ideal, checked_constraints = ideal_polynomial(data["constraint_ideal_terms"], variables, phase_variables)
    nonnegative, checked_nonnegative = nonnegative_defect_polynomial(
        data["nonnegative_defect_terms"],
        variables,
        phase_variables,
    )
    residual = candidate - traced - ideal - nonnegative
    require(not residual.terms, f"trace defect absorber identity failed with residual {residual.to_sparse_json()}")
    return (
        ["candidate_density_equals_trace_plus_nonnegative_defects_mod_constraint_ideal"],
        checked_constraints,
        checked_nonnegative,
    )


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(data, ["chart", "variables", "trace_certificate_path"], "certificate")

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    variables = validate_string_list(data["variables"], "variables")
    trace_path = resolve_path(data["trace_certificate_path"], base_dir, "trace_certificate_path")
    trace_data = load_json(trace_path)
    trace_result = trace.verify(trace_data, base_dir=trace_path.parent)
    require(chart == trace_result["chart"], "certificate chart must match trace chart")
    require(set(variables) == set(trace_result["variables"]), "variables must match trace variables")

    phase_variables, symbolic_variables, source_auxiliary_variables = trace_variables(trace_data)
    defect_auxiliary_variables = validate_string_list(
        data.get("defect_auxiliary_variables", []),
        "defect_auxiliary_variables",
        allow_empty=True,
    )
    all_variables = phase_variables + symbolic_variables + source_auxiliary_variables + defect_auxiliary_variables
    require(len(set(all_variables)) == len(all_variables), "all trace and defect variables must be disjoint")
    checked_identities, checked_constraints, checked_nonnegative = verify_defect_identity(
        data,
        trace_data,
        all_variables,
        phase_variables,
    )

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables),
        "trace_certificate": str(trace_path),
        "local_coercivity_delta": trace_result["local_coercivity_delta"],
        "checked_identities": checked_identities + trace_result["checked_identities"],
        "checked_constraint_terms": checked_constraints,
        "checked_nonnegative_defects": checked_nonnegative,
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
        print(f"Carter trace defect absorber rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
