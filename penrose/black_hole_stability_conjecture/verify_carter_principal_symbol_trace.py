#!/usr/bin/env python3
"""Validate trace certificates for Carter principal-symbol source formulas.

The trace checker is a finite algebraic audit layer between a future exact
linearized-Kerr principal-symbol computation and the source-backend contract.
It expands named primitive packets and derived formulas, checks homogeneity and
phase-free trapped constraints, and verifies that the traced density and
constraints are exactly the backend interface.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_carter_kerr_source_backend as backend
import verify_carter_sos_identity as sos


CLAIM_SCOPE = "chart_principal_symbol_trace"
ALLOWED_DERIVATION_STATUSES = {
    "candidate_trace",
    "kerr_gravitational_proved",
}
ALLOWED_PRIMITIVE_ROLES = {
    "principal_density_packet",
    "trapped_constraint",
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
            "metadata.derivation_status kerr_gravitational_proved requires a registered "
            "principal-symbol differentiation engine; this checker verifies only exact "
            "trace algebra against the backend interface"
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


def validate_trace_variables(
    data: dict[str, Any],
    backend_data: dict[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    trace_variables = data.get("trace_variables")
    require(isinstance(trace_variables, dict), "trace_variables must be an object")
    require_keys(
        trace_variables,
        ["phase_variables", "symbolic_coefficient_variables", "source_auxiliary_variables"],
        "trace_variables",
    )
    phase_variables = validate_string_list(trace_variables["phase_variables"], "trace_variables.phase_variables")
    symbolic_variables = validate_string_list(
        trace_variables["symbolic_coefficient_variables"],
        "trace_variables.symbolic_coefficient_variables",
    )
    auxiliary_variables = validate_string_list(
        trace_variables["source_auxiliary_variables"],
        "trace_variables.source_auxiliary_variables",
        allow_empty=True,
    )

    interface = backend_data["backend_contract"]["interface"]
    require(phase_variables == tuple(interface["phase_variables"]), "trace phase variables must match backend interface")
    require(
        symbolic_variables == tuple(interface["symbolic_coefficient_variables"]),
        "trace symbolic variables must match backend interface",
    )
    require(
        auxiliary_variables == tuple(interface["source_auxiliary_variables"]),
        "trace source auxiliary variables must match backend interface",
    )
    require(
        len(set(phase_variables + symbolic_variables + auxiliary_variables))
        == len(phase_variables) + len(symbolic_variables) + len(auxiliary_variables),
        "trace variable families must be disjoint",
    )
    return phase_variables, symbolic_variables, auxiliary_variables


def eval_trace_expr(
    expr: Any,
    variables: tuple[str, ...],
    environment: dict[str, Any],
    stack: tuple[str, ...] = (),
) -> sos.Polynomial:
    if isinstance(expr, dict) and "ref" in expr:
        name = expr["ref"]
        require(isinstance(name, str) and name, "ref must be a nonempty string")
        require(name in environment, f"unknown trace formula reference {name!r}")
        require(name not in stack, f"cyclic trace formula reference involving {name!r}")
        return eval_trace_expr(environment[name], variables, environment, stack + (name,))
    if isinstance(expr, str):
        return sos.Polynomial.variable(variables, expr)
    if isinstance(expr, int):
        return sos.Polynomial.constant(variables, sos.parse_fraction(expr))
    if isinstance(expr, dict):
        if "const" in expr:
            return sos.Polynomial.constant(variables, sos.parse_fraction(expr["const"]))
        if "var" in expr:
            return sos.Polynomial.variable(variables, str(expr["var"]))
        if "poly" in expr:
            raw = expr["poly"]
            require(isinstance(raw, dict), "poly expression must be an object")
            return sos.parse_sparse_poly(raw, variables)
        if "neg" in expr:
            return -eval_trace_expr(expr["neg"], variables, environment, stack)
        if "sum" in expr:
            items = expr["sum"]
            require(isinstance(items, list), "sum expression must be a list")
            total = sos.Polynomial.zero(variables)
            for item in items:
                total = total + eval_trace_expr(item, variables, environment, stack)
            return total
        if "prod" in expr:
            items = expr["prod"]
            require(isinstance(items, list), "prod expression must be a list")
            total = sos.Polynomial.constant(variables, sos.parse_fraction(1))
            for item in items:
                total = total * eval_trace_expr(item, variables, environment, stack)
            return total
        if "square" in expr:
            poly = eval_trace_expr(expr["square"], variables, environment, stack)
            return poly * poly
        if "scale" in expr:
            spec = expr["scale"]
            require(isinstance(spec, dict) and "coeff" in spec and "expr" in spec, "scale expression must contain coeff and expr")
            return eval_trace_expr(spec["expr"], variables, environment, stack).scale(sos.parse_fraction(spec["coeff"]))
    raise ValueError(f"unsupported trace expression {expr!r}")


def load_formulas(data: dict[str, Any], variables: tuple[str, ...], phase_variables: tuple[str, ...]) -> tuple[dict[str, Any], list[str]]:
    environment: dict[str, Any] = {}
    checked_primitives: list[str] = []

    primitives = data.get("primitive_formulas")
    require(isinstance(primitives, list) and primitives, "primitive_formulas must be a nonempty list")
    for index, formula in enumerate(primitives):
        require(isinstance(formula, dict), f"primitive_formulas[{index}] must be an object")
        require_keys(formula, ["name", "role", "expression", "provenance"], f"primitive_formulas[{index}]")
        name = formula["name"]
        role = formula["role"]
        require(isinstance(name, str) and name, f"primitive_formulas[{index}].name must be a nonempty string")
        require(name not in environment, f"duplicate trace formula name {name!r}")
        require(role in ALLOWED_PRIMITIVE_ROLES, f"primitive formula {name!r} has unknown role {role!r}")
        require(isinstance(formula["provenance"], str) and formula["provenance"].strip(), f"primitive formula {name!r} provenance must be nonempty")
        environment[name] = formula["expression"]
        poly = eval_trace_expr(formula["expression"], variables, environment)
        if role == "principal_density_packet":
            validate_phase_homogeneous(poly, phase_variables, 2, f"primitive formula {name!r}")
        if role == "trapped_constraint":
            validate_phase_free(poly, phase_variables, f"primitive formula {name!r}")
        checked_primitives.append(name)

    derived = data.get("derived_formulas")
    require(isinstance(derived, list) and derived, "derived_formulas must be a nonempty list")
    derived_names: list[str] = []
    for index, formula in enumerate(derived):
        require(isinstance(formula, dict), f"derived_formulas[{index}] must be an object")
        require_keys(formula, ["name", "expression", "rule"], f"derived_formulas[{index}]")
        name = formula["name"]
        require(isinstance(name, str) and name, f"derived_formulas[{index}].name must be a nonempty string")
        require(name not in environment, f"duplicate trace formula name {name!r}")
        require(isinstance(formula["rule"], str) and formula["rule"].strip(), f"derived formula {name!r} rule must be nonempty")
        environment[name] = formula["expression"]
        derived_names.append(name)

    for name in derived_names:
        eval_trace_expr({"ref": name}, variables, environment)

    return environment, checked_primitives


def backend_constraint_map(backend_data: dict[str, Any]) -> dict[str, Any]:
    raw_constraints = backend_data["backend_contract"]["interface"]["exported_constraint_expressions"]
    require(isinstance(raw_constraints, list), "backend exported constraints must be a list")
    constraints: dict[str, Any] = {}
    for item in raw_constraints:
        require(isinstance(item, dict), "backend exported constraint entries must be objects")
        require_keys(item, ["name", "expression"], "backend exported constraint")
        name = item["name"]
        require(isinstance(name, str) and name, "backend exported constraint name must be nonempty")
        require(name not in constraints, f"duplicate backend constraint {name!r}")
        constraints[name] = item["expression"]
    return constraints


def polynomial_residual(left: sos.Polynomial, right_expr: Any, variables: tuple[str, ...]) -> sos.Polynomial:
    right = sos.eval_expr(right_expr, variables)
    return left - right


def verify_trace_matches_backend(
    data: dict[str, Any],
    backend_data: dict[str, Any],
    variables: tuple[str, ...],
    phase_variables: tuple[str, ...],
    environment: dict[str, Any],
) -> list[str]:
    require_keys(data, ["density_trace", "constraint_traces"], "certificate")
    density_poly = eval_trace_expr(data["density_trace"], variables, environment)
    validate_phase_homogeneous(density_poly, phase_variables, 2, "density_trace")
    backend_density = backend_data["backend_contract"]["interface"]["exported_source_density_expression"]
    residual = polynomial_residual(density_poly, backend_density, variables)
    require(not residual.terms, f"density trace does not match backend source density: residual {residual.to_sparse_json()}")

    trace_constraints_raw = data["constraint_traces"]
    require(isinstance(trace_constraints_raw, list) and trace_constraints_raw, "constraint_traces must be a nonempty list")
    trace_constraints: dict[str, sos.Polynomial] = {}
    for index, item in enumerate(trace_constraints_raw):
        require(isinstance(item, dict), f"constraint_traces[{index}] must be an object")
        require_keys(item, ["name", "expression"], f"constraint_traces[{index}]")
        name = item["name"]
        require(isinstance(name, str) and name, f"constraint_traces[{index}].name must be nonempty")
        require(name not in trace_constraints, f"duplicate constraint trace name {name!r}")
        poly = eval_trace_expr(item["expression"], variables, environment)
        validate_phase_free(poly, phase_variables, f"constraint trace {name!r}")
        trace_constraints[name] = poly

    backend_constraints = backend_constraint_map(backend_data)
    require(set(trace_constraints) == set(backend_constraints), "constraint trace names must match backend constraints")
    for name in sorted(trace_constraints):
        residual = polynomial_residual(trace_constraints[name], backend_constraints[name], variables)
        require(not residual.terms, f"constraint trace {name!r} does not match backend: residual {residual.to_sparse_json()}")

    return [
        "principal_symbol_density_trace_matches_backend_source",
        "principal_symbol_constraint_traces_match_backend_constraints",
    ]


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(data, ["chart", "variables", "backend_contract_certificate_path", "trace_variables"], "certificate")

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    variables = validate_string_list(data["variables"], "variables")
    backend_path = resolve_path(data["backend_contract_certificate_path"], base_dir, "backend_contract_certificate_path")
    backend_data = load_json(backend_path)
    backend_result = backend.verify(backend_data, base_dir=backend_path.parent)
    require(chart == backend_result["chart"], "certificate chart must match backend chart")
    require(set(variables) == set(backend_result["variables"]), "variables must match backend variables")

    phase_variables, symbolic_variables, auxiliary_variables = validate_trace_variables(data, backend_data)
    all_variables = phase_variables + symbolic_variables + auxiliary_variables
    environment, checked_primitives = load_formulas(data, all_variables, phase_variables)
    checked_trace_identities = verify_trace_matches_backend(
        data,
        backend_data,
        all_variables,
        phase_variables,
        environment,
    )

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables),
        "backend_contract_certificate": str(backend_path),
        "local_coercivity_delta": backend_result["local_coercivity_delta"],
        "checked_primitives": checked_primitives,
        "checked_identities": checked_trace_identities + backend_result["checked_identities"],
        "remaining_exact_derivation_obligations": backend_result["remaining_exact_derivation_obligations"],
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
        print(f"Carter principal-symbol trace rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
