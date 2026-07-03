#!/usr/bin/env python3
"""Validate symbolic normal-form identities for local Carter densities.

This checker proves the exact algebraic bridge from a symbolic quadratic
density expression to the six coefficient expression trees already checked by
the rational extraction verifier.  It does not derive the density from the Kerr
equations; a genuine gravitational upgrade must be supplied by a separate exact
derivation backend.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_rational_coefficient_extraction as extraction
import verify_carter_sos_identity as sos


CLAIM_SCOPE = "chart_symbolic_kerr_identification"
ALLOWED_DERIVATION_STATUSES = {
    "model_probe",
    "candidate_symbolic_identity",
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


def monomial_polynomial(key: str, variables: tuple[str, ...]) -> sos.Polynomial:
    monomial = sos.parse_monomial_key(key, variables)
    return sos.Polynomial(variables, {monomial: Fraction(1)})


def reconstruct_density(
    coefficient_expressions: dict[str, Any],
    phase_variables: tuple[str, ...],
    symbolic_variables: tuple[str, ...],
) -> sos.Polynomial:
    variables = phase_variables + symbolic_variables
    total = sos.Polynomial.zero(variables)
    for monomial_key, coefficient_expression in coefficient_expressions.items():
        monomial = monomial_polynomial(monomial_key, variables)
        coefficient = sos.eval_expr(coefficient_expression, variables)
        total = total + monomial * coefficient
    return total


def validate_quadratic_phase_degree(poly: sos.Polynomial, phase_variables: tuple[str, ...]) -> None:
    for monomial in poly.terms:
        phase_degree = sum(monomial[index] for index in range(len(phase_variables)))
        require(phase_degree == 2, "symbolic density must be homogeneous quadratic in phase variables")


def verify_symbolic_identity(
    data: dict[str, Any],
    extraction_data: dict[str, Any],
) -> list[str]:
    require_keys(data, ["symbolic_coefficient_variables", "density_expression"], "certificate")
    phase_variables_raw = data["variables"]
    require(isinstance(phase_variables_raw, list) and phase_variables_raw, "variables must be a nonempty list")
    require(all(isinstance(variable, str) for variable in phase_variables_raw), "variables must be strings")
    phase_variables = tuple(phase_variables_raw)

    symbolic_raw = data["symbolic_coefficient_variables"]
    require(isinstance(symbolic_raw, list) and symbolic_raw, "symbolic_coefficient_variables must be a nonempty list")
    require(all(isinstance(variable, str) for variable in symbolic_raw), "symbolic coefficient variables must be strings")
    symbolic_variables = tuple(symbolic_raw)
    require(set(phase_variables).isdisjoint(set(symbolic_variables)), "phase and symbolic variables must be disjoint")

    interval_symbols = extraction_data.get("interval_symbols")
    require(isinstance(interval_symbols, dict), "extraction interval_symbols must be an object")
    require(set(interval_symbols).issubset(set(symbolic_variables)), "every extraction interval symbol must be symbolic")

    coefficient_expressions = extraction_data.get("coefficient_expressions")
    require(isinstance(coefficient_expressions, dict), "extraction coefficient_expressions must be an object")
    reconstructed = reconstruct_density(coefficient_expressions, phase_variables, symbolic_variables)
    density = sos.eval_expr(data["density_expression"], phase_variables + symbolic_variables)
    validate_quadratic_phase_degree(density, phase_variables)
    residual = density - reconstructed
    require(not residual.terms, f"symbolic density normal-form identity failed with residual {residual.to_sparse_json()}")
    return ["symbolic_density_equals_six_coefficient_normal_form"]


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(data, ["chart", "variables", "extraction_certificate_path"], "certificate")
    validate_source_metadata(data, base_dir)

    extraction_path = resolve_path(data["extraction_certificate_path"], base_dir, "extraction_certificate_path")
    extraction_data = load_json(extraction_path)
    extraction_result = extraction.verify(extraction_data, base_dir=extraction_path.parent)

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    require(chart == extraction_result["chart"], "certificate chart must match extraction chart")
    variables_raw = data["variables"]
    require(isinstance(variables_raw, list), "variables must be a list")
    require(set(variables_raw) == set(extraction_result["variables"]), "variables must match extraction variables")

    checked_identities = verify_symbolic_identity(data, extraction_data)
    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables_raw),
        "extraction_certificate": str(extraction_path),
        "local_coercivity_delta": extraction_result["local_coercivity_delta"],
        "checked_identities": checked_identities,
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
        print(f"Carter symbolic Kerr identification rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
