#!/usr/bin/env python3
"""Validate Kerr source-backend interface certificates for a Carter chart.

The checker closes the formal interface between an eventual exact
linearized-Kerr principal-symbol derivation and the already verified
source-ideal certificate.  It verifies that the backend-exported source density
and trapped constraints are exactly the source-ideal inputs, then delegates the
coercivity chain to the source-ideal checker.  It is a contract checker, not yet
an independent derivation engine for the Einstein/Kerr principal symbol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_carter_sos_identity as sos
import verify_carter_source_ideal_derivation as source_ideal


CLAIM_SCOPE = "chart_kerr_source_backend"
ALLOWED_DERIVATION_STATUSES = {
    "candidate_backend_contract",
    "kerr_gravitational_proved",
}
REQUIRED_OPERATOR_BACKGROUND = "fixed_subextremal_kerr"
REQUIRED_OPERATOR_EQUATION = "linearized_vacuum_einstein_principal_symbol"


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
            "principal-symbol derivation engine; this checker currently verifies only "
            "the backend/source-ideal interface contract"
        )
    return str(derivation_status)


def validate_string_list(raw: Any, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    require(isinstance(raw, list), f"{name} must be a list")
    require(allow_empty or bool(raw), f"{name} must be nonempty")
    require(all(isinstance(item, str) and item for item in raw), f"{name} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{name} entries must be distinct")
    return tuple(raw)


def validate_string_map(raw: Any, expected_keys: set[str], name: str) -> None:
    require(isinstance(raw, dict), f"{name} must be an object")
    actual_keys = set(raw)
    require(actual_keys == expected_keys, f"{name} must contain exactly {sorted(expected_keys)}")
    for key, value in raw.items():
        require(isinstance(value, str) and value.strip(), f"{name}.{key} must be a nonempty string")


def load_symbolic_certificate(source_data: dict[str, Any], source_path: Path) -> dict[str, Any]:
    symbolic_path = resolve_path(
        source_data.get("symbolic_certificate_path"),
        source_path.parent,
        "source_ideal.symbolic_certificate_path",
    )
    return load_json(symbolic_path)


def constraint_expressions(source_data: dict[str, Any]) -> dict[str, Any]:
    raw_terms = source_data.get("constraint_ideal_terms")
    require(isinstance(raw_terms, list) and raw_terms, "source ideal constraint_ideal_terms must be a nonempty list")
    constraints: dict[str, Any] = {}
    for index, term in enumerate(raw_terms):
        require(isinstance(term, dict), f"source ideal constraint_ideal_terms[{index}] must be an object")
        require_keys(term, ["name", "constraint_expression"], f"source ideal constraint_ideal_terms[{index}]")
        name = term["name"]
        require(isinstance(name, str) and name, f"source ideal constraint_ideal_terms[{index}].name must be a nonempty string")
        require(name not in constraints, f"duplicate source ideal constraint name {name!r}")
        constraints[name] = term["constraint_expression"]
    return constraints


def validate_operator_family(contract: dict[str, Any], chart: str) -> str:
    operator = contract.get("operator_family")
    require(isinstance(operator, dict), "backend_contract.operator_family must be an object")
    require_keys(operator, ["name", "background", "equation", "chart", "gauge"], "backend_contract.operator_family")
    for key in ["name", "background", "equation", "chart", "gauge"]:
        require(isinstance(operator[key], str) and operator[key].strip(), f"backend_contract.operator_family.{key} must be a nonempty string")
    require(operator["background"] == REQUIRED_OPERATOR_BACKGROUND, f"backend_contract.operator_family.background must be {REQUIRED_OPERATOR_BACKGROUND}")
    require(operator["equation"] == REQUIRED_OPERATOR_EQUATION, f"backend_contract.operator_family.equation must be {REQUIRED_OPERATOR_EQUATION}")
    require(operator["chart"] == chart, "backend_contract.operator_family.chart must match certificate chart")
    return str(operator["name"])


def validate_normalization_map(
    contract: dict[str, Any],
    chart: str,
    phase_variables: tuple[str, ...],
    symbolic_variables: tuple[str, ...],
    constraint_names: tuple[str, ...],
) -> None:
    normalization = contract.get("normalization_map")
    require(isinstance(normalization, dict), "backend_contract.normalization_map must be an object")
    require_keys(
        normalization,
        ["coordinate_chart", "radial_normalization", "phase_variable_map", "symbolic_coefficient_map", "constraint_variable_map"],
        "backend_contract.normalization_map",
    )
    require(normalization["coordinate_chart"] == chart, "backend_contract.normalization_map.coordinate_chart must match chart")
    require(isinstance(normalization["radial_normalization"], str) and normalization["radial_normalization"].strip(), "backend_contract.normalization_map.radial_normalization must be a nonempty string")
    validate_string_map(normalization["phase_variable_map"], set(phase_variables), "backend_contract.normalization_map.phase_variable_map")
    validate_string_map(normalization["symbolic_coefficient_map"], set(symbolic_variables), "backend_contract.normalization_map.symbolic_coefficient_map")
    validate_string_map(normalization["constraint_variable_map"], set(constraint_names), "backend_contract.normalization_map.constraint_variable_map")


def polynomial_residual(
    left_expr: Any,
    right_expr: Any,
    variables: tuple[str, ...],
) -> sos.Polynomial:
    left = sos.eval_expr(left_expr, variables)
    right = sos.eval_expr(right_expr, variables)
    return left - right


def validate_interface(
    contract: dict[str, Any],
    source_data: dict[str, Any],
    phase_variables: tuple[str, ...],
    symbolic_variables: tuple[str, ...],
    source_auxiliary_variables: tuple[str, ...],
    constraints: dict[str, Any],
) -> list[str]:
    interface = contract.get("interface")
    require(isinstance(interface, dict), "backend_contract.interface must be an object")
    require_keys(
        interface,
        [
            "phase_variables",
            "symbolic_coefficient_variables",
            "source_auxiliary_variables",
            "exported_source_density_expression",
            "exported_constraint_expressions",
        ],
        "backend_contract.interface",
    )
    require(validate_string_list(interface["phase_variables"], "backend_contract.interface.phase_variables") == phase_variables, "backend_contract.interface.phase_variables must match certificate variables")
    require(
        validate_string_list(interface["symbolic_coefficient_variables"], "backend_contract.interface.symbolic_coefficient_variables")
        == symbolic_variables,
        "backend_contract.interface.symbolic_coefficient_variables must match symbolic certificate",
    )
    require(
        validate_string_list(
            interface["source_auxiliary_variables"],
            "backend_contract.interface.source_auxiliary_variables",
            allow_empty=True,
        )
        == source_auxiliary_variables,
        "backend_contract.interface.source_auxiliary_variables must match source ideal certificate",
    )

    all_variables = phase_variables + symbolic_variables + source_auxiliary_variables
    require(len(set(all_variables)) == len(all_variables), "interface variable families must be disjoint")
    source_residual = polynomial_residual(
        interface["exported_source_density_expression"],
        source_data["source_density_expression"],
        all_variables,
    )
    require(
        not source_residual.terms,
        f"backend source density interface mismatch: residual {source_residual.to_sparse_json()}",
    )

    exported_constraints_raw = interface["exported_constraint_expressions"]
    require(isinstance(exported_constraints_raw, list), "backend_contract.interface.exported_constraint_expressions must be a list")
    exported_constraints: dict[str, Any] = {}
    for index, item in enumerate(exported_constraints_raw):
        require(isinstance(item, dict), f"backend_contract.interface.exported_constraint_expressions[{index}] must be an object")
        require_keys(item, ["name", "expression"], f"backend_contract.interface.exported_constraint_expressions[{index}]")
        name = item["name"]
        require(isinstance(name, str) and name, f"backend_contract.interface.exported_constraint_expressions[{index}].name must be a nonempty string")
        require(name not in exported_constraints, f"duplicate exported constraint name {name!r}")
        exported_constraints[name] = item["expression"]
    require(set(exported_constraints) == set(constraints), "backend exported constraints must match source ideal constraint names")
    for name in sorted(constraints):
        residual = polynomial_residual(exported_constraints[name], constraints[name], all_variables)
        require(
            not residual.terms,
            f"backend constraint interface mismatch for {name}: residual {residual.to_sparse_json()}",
        )

    return [
        "backend_exported_source_density_matches_source_ideal",
        "backend_exported_constraints_match_source_ideal",
    ]


def validate_remaining_obligations(contract: dict[str, Any]) -> list[str]:
    obligations = validate_string_list(
        contract.get("remaining_exact_derivation_obligations"),
        "backend_contract.remaining_exact_derivation_obligations",
    )
    return list(obligations)


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(data, ["chart", "variables", "source_ideal_certificate_path", "backend_contract"], "certificate")

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    variables = validate_string_list(data["variables"], "variables")

    source_path = resolve_path(data["source_ideal_certificate_path"], base_dir, "source_ideal_certificate_path")
    source_data = load_json(source_path)
    source_result = source_ideal.verify(source_data, base_dir=source_path.parent)
    require(chart == source_result["chart"], "certificate chart must match source ideal chart")
    require(set(variables) == set(source_result["variables"]), "variables must match source ideal variables")

    symbolic_data = load_symbolic_certificate(source_data, source_path)
    symbolic_variables = validate_string_list(symbolic_data.get("symbolic_coefficient_variables"), "symbolic_coefficient_variables")
    source_auxiliary_variables = validate_string_list(
        source_data.get("source_auxiliary_variables", []),
        "source_auxiliary_variables",
        allow_empty=True,
    )
    constraints = constraint_expressions(source_data)
    constraint_names = tuple(constraints)

    contract = data["backend_contract"]
    require(isinstance(contract, dict), "backend_contract must be an object")
    operator_name = validate_operator_family(contract, chart)
    validate_normalization_map(contract, chart, variables, symbolic_variables, constraint_names)
    checked_interface_identities = validate_interface(
        contract,
        source_data,
        variables,
        symbolic_variables,
        source_auxiliary_variables,
        constraints,
    )
    remaining_obligations = validate_remaining_obligations(contract)

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables),
        "operator_family": operator_name,
        "source_ideal_certificate": str(source_path),
        "local_coercivity_delta": source_result["local_coercivity_delta"],
        "checked_identities": checked_interface_identities + source_result["checked_identities"],
        "checked_constraint_terms": source_result["checked_constraint_terms"],
        "remaining_exact_derivation_obligations": remaining_obligations,
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
        print(f"Carter Kerr source backend rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
