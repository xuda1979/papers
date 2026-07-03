#!/usr/bin/env python3
"""Validate source-to-normal-form identities modulo trapped constraints.

The checker proves a quotient-algebra link: a proposed source density equals an
accepted symbolic normal form modulo an explicitly listed ideal of constraint
expressions.  It does not derive the source density from the Kerr equations;
that external derivation is still a separate backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_carter_sos_identity as sos
import verify_carter_symbolic_kerr_identification as symbolic


CLAIM_SCOPE = "chart_source_ideal_derivation"
ALLOWED_DERIVATION_STATUSES = {
    "model_probe",
    "candidate_source_ideal",
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
            "metadata.derivation_status kerr_gravitational_proved requires an exact Kerr derivation "
            "backend; no such backend is registered in this checker"
        )
    return str(derivation_status)


def validate_source_metadata(data: dict[str, Any], base_dir: Path | None) -> None:
    source = data.get("source_metadata")
    require(isinstance(source, dict), "source_metadata must be an object")
    require_keys(source, ["ansatz_path", "normalization", "derivation_note"], "source_metadata")
    ansatz_path = resolve_path(source["ansatz_path"], base_dir, "source_metadata.ansatz_path")
    require(ansatz_path.exists(), f"source_metadata.ansatz_path does not exist: {ansatz_path}")
    require(isinstance(source["normalization"], str) and source["normalization"], "source_metadata.normalization must be a nonempty string")
    require(isinstance(source["derivation_note"], str) and source["derivation_note"], "source_metadata.derivation_note must be a nonempty string")


def validate_variable_list(raw: Any, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    require(isinstance(raw, list), f"{name} must be a list")
    require(allow_empty or bool(raw), f"{name} must be nonempty")
    require(all(isinstance(variable, str) and variable for variable in raw), f"{name} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{name} entries must be distinct")
    return tuple(raw)


def phase_degree(poly: sos.Polynomial, monomial: tuple[int, ...], phase_variables: tuple[str, ...]) -> int:
    return sum(monomial[index] for index in range(len(phase_variables)))


def validate_phase_homogeneous(
    poly: sos.Polynomial,
    phase_variables: tuple[str, ...],
    degree: int,
    name: str,
) -> None:
    for monomial in poly.terms:
        require(
            phase_degree(poly, monomial, phase_variables) == degree,
            f"{name} must be homogeneous of phase degree {degree}",
        )


def validate_phase_free_constraint(
    poly: sos.Polynomial,
    phase_variables: tuple[str, ...],
    name: str,
) -> None:
    for monomial in poly.terms:
        require(phase_degree(poly, monomial, phase_variables) == 0, f"{name} must be phase-free")


def ideal_polynomial(
    raw_terms: Any,
    variables: tuple[str, ...],
    phase_variables: tuple[str, ...],
) -> tuple[sos.Polynomial, list[str]]:
    require(isinstance(raw_terms, list) and raw_terms, "constraint_ideal_terms must be a nonempty list")
    total = sos.Polynomial.zero(variables)
    checked: list[str] = []
    for index, term in enumerate(raw_terms):
        require(isinstance(term, dict), f"constraint_ideal_terms[{index}] must be an object")
        require_keys(term, ["name", "constraint_expression", "multiplier_expression"], f"constraint_ideal_terms[{index}]")
        name = str(term["name"])
        constraint = sos.eval_expr(term["constraint_expression"], variables)
        multiplier = sos.eval_expr(term["multiplier_expression"], variables)
        validate_phase_free_constraint(constraint, phase_variables, f"constraint {name!r}")
        total = total + constraint * multiplier
        checked.append(name)
    return total, checked


def verify_source_ideal_identity(
    data: dict[str, Any],
    symbolic_data: dict[str, Any],
) -> tuple[list[str], list[str]]:
    require_keys(data, ["source_density_expression", "constraint_ideal_terms"], "certificate")
    phase_variables = validate_variable_list(data["variables"], "variables")
    symbolic_variables = validate_variable_list(symbolic_data["symbolic_coefficient_variables"], "symbolic_coefficient_variables")
    source_auxiliary_variables = validate_variable_list(
        data.get("source_auxiliary_variables", []),
        "source_auxiliary_variables",
        allow_empty=True,
    )
    all_variables = phase_variables + symbolic_variables + source_auxiliary_variables
    require(len(set(all_variables)) == len(all_variables), "phase, symbolic, and source auxiliary variables must be disjoint")

    source_density = sos.eval_expr(data["source_density_expression"], all_variables)
    symbolic_density = sos.eval_expr(symbolic_data["density_expression"], all_variables)
    validate_phase_homogeneous(source_density, phase_variables, 2, "source_density_expression")
    ideal, checked_terms = ideal_polynomial(data["constraint_ideal_terms"], all_variables, phase_variables)
    residual = source_density - symbolic_density - ideal
    require(not residual.terms, f"source ideal identity failed with residual {residual.to_sparse_json()}")
    return ["source_density_equals_symbolic_normal_form_mod_constraint_ideal"], checked_terms


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(data, ["chart", "variables", "symbolic_certificate_path"], "certificate")
    validate_source_metadata(data, base_dir)

    symbolic_path = resolve_path(data["symbolic_certificate_path"], base_dir, "symbolic_certificate_path")
    symbolic_data = load_json(symbolic_path)
    symbolic_result = symbolic.verify(symbolic_data, base_dir=symbolic_path.parent)

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    require(chart == symbolic_result["chart"], "certificate chart must match symbolic certificate chart")
    variables = validate_variable_list(data["variables"], "variables")
    require(set(variables) == set(symbolic_result["variables"]), "variables must match symbolic certificate variables")

    checked_identities, checked_constraints = verify_source_ideal_identity(data, symbolic_data)
    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables),
        "symbolic_certificate": str(symbolic_path),
        "local_coercivity_delta": symbolic_result["local_coercivity_delta"],
        "checked_identities": checked_identities,
        "checked_constraint_terms": checked_constraints,
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
        print(f"Carter source ideal derivation rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
