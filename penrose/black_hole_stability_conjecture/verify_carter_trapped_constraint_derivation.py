#!/usr/bin/env python3
"""Verify exact trapped radial-constraint derivation for a Carter chart.

The checker proves the symbolic Kerr radial identities behind the normalized
trapped constraints and verifies that the backend interface exports exactly the
residual variables reserved for those constraints.  It does not derive the
full gravitational density q_grav.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sympy as sp

import verify_carter_kerr_source_backend as backend


CLAIM_SCOPE = "chart_trapped_constraint_derivation"
DERIVATION_STATUS = "exact_trapped_constraints"
REQUIRED_BACKEND_CONSTRAINTS = {
    "radial_constraint_R": "rho_R",
    "radial_constraint_dR": "rho_dR",
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


def validate_metadata(data: dict[str, Any]) -> None:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "claim_scope", "status", "derivation_status"], "metadata")
    require(metadata["claim_scope"] == CLAIM_SCOPE, f"metadata.claim_scope must be {CLAIM_SCOPE}")
    require(
        metadata["derivation_status"] == DERIVATION_STATUS,
        f"metadata.derivation_status must be {DERIVATION_STATUS}",
    )


def validate_string_list(raw: Any, name: str) -> tuple[str, ...]:
    require(isinstance(raw, list) and raw, f"{name} must be a nonempty list")
    require(all(isinstance(item, str) and item for item in raw), f"{name} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{name} entries must be distinct")
    return tuple(raw)


def backend_constraint_map(backend_data: dict[str, Any]) -> dict[str, str]:
    raw_constraints = backend_data["backend_contract"]["interface"]["exported_constraint_expressions"]
    require(isinstance(raw_constraints, list), "backend exported constraints must be a list")
    constraints: dict[str, str] = {}
    for index, item in enumerate(raw_constraints):
        require(isinstance(item, dict), f"backend exported constraint {index} must be an object")
        require_keys(item, ["name", "expression"], f"backend exported constraint {index}")
        name = item["name"]
        expr = item["expression"]
        require(isinstance(name, str) and name, f"backend exported constraint {index}.name must be nonempty")
        require(isinstance(expr, dict) and set(expr) == {"var"}, f"backend constraint {name} must export a single variable")
        variable = expr["var"]
        require(isinstance(variable, str) and variable, f"backend constraint {name} variable must be nonempty")
        require(name not in constraints, f"duplicate backend constraint {name!r}")
        constraints[name] = variable
    return constraints


def verify_radial_symbol_identities() -> list[str]:
    r, a, m = sp.symbols("r a m")
    lam_symbol, eta_symbol = sp.symbols("lambda eta")

    delta = r**2 - 2 * m * r + a**2
    lam = (r**2 + a**2) / a - (2 * r * delta) / (a * (r - m))
    eta = (4 * r**2 * delta) / (r - m) ** 2 - (lam - a) ** 2

    p = r**2 + a**2 - a * lam
    b = eta + (lam - a) ** 2
    radial = p**2 - delta * b
    radial_derivative = 4 * r * p - sp.diff(delta, r) * b

    symbolic_p = r**2 + a**2 - a * lam_symbol
    symbolic_b = eta_symbol + (lam_symbol - a) ** 2
    symbolic_radial = symbolic_p**2 - delta * symbolic_b
    symbolic_radial_derivative = sp.diff(symbolic_radial, r)

    checks = {
        "spherical_R_constraint_clears_to_zero": sp.factor(radial * (r - m) ** 2),
        "spherical_dR_constraint_clears_to_zero": sp.factor(radial_derivative * (r - m) ** 2),
        "formal_dR_matches_total_radial_derivative": sp.factor(symbolic_radial_derivative - (4 * r * symbolic_p - sp.diff(delta, r) * symbolic_b)),
        "P_normal_form_clears_to_zero": sp.factor((p - 2 * r * delta / (r - m)) * (r - m)),
        "B_normal_form_clears_to_zero": sp.factor((b - 4 * r**2 * delta / (r - m) ** 2) * (r - m) ** 2),
    }
    failed = {name: str(value) for name, value in checks.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"trapped radial constraint identity failed: {failed}")
    return sorted(checks)


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    validate_metadata(data)
    require_keys(data, ["chart", "variables", "backend_contract_certificate_path", "constraint_derivation"], "certificate")

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    variables = validate_string_list(data["variables"], "variables")

    backend_path = resolve_path(
        data["backend_contract_certificate_path"],
        base_dir,
        "backend_contract_certificate_path",
    )
    backend_data = load_json(backend_path)
    backend_result = backend.verify(backend_data, base_dir=backend_path.parent)
    require(backend_result["chart"] == chart, "certificate chart must match backend chart")
    require(set(backend_result["variables"]) == set(variables), "certificate variables must match backend variables")

    derivation = data["constraint_derivation"]
    require(isinstance(derivation, dict), "constraint_derivation must be an object")
    require_keys(
        derivation,
        ["radial_potential", "spherical_orbit_parameters", "normalized_residual_variables"],
        "constraint_derivation",
    )
    require(isinstance(derivation["radial_potential"], str) and derivation["radial_potential"].strip(), "radial_potential must be nonempty")
    require(isinstance(derivation["spherical_orbit_parameters"], str) and derivation["spherical_orbit_parameters"].strip(), "spherical_orbit_parameters must be nonempty")

    residual_variables = derivation["normalized_residual_variables"]
    require(isinstance(residual_variables, dict), "normalized_residual_variables must be an object")
    require(residual_variables == REQUIRED_BACKEND_CONSTRAINTS, "normalized residual variables must match required backend constraints")
    require(backend_constraint_map(backend_data) == REQUIRED_BACKEND_CONSTRAINTS, "backend exported trapped constraints must match normalized residual variables")

    checked_identities = verify_radial_symbol_identities()
    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": DERIVATION_STATUS,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables),
        "backend_contract_certificate": str(backend_path),
        "local_coercivity_delta": backend_result["local_coercivity_delta"],
        "derived_constraints": REQUIRED_BACKEND_CONSTRAINTS,
        "checked_identities": [
            "derived_constraints_match_backend_interface",
            "trapped_constraints_are_phase_free_backend_residuals",
        ]
        + checked_identities,
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
        print(f"Carter trapped-constraint derivation rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
