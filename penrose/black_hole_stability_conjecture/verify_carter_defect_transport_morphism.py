#!/usr/bin/env python3
"""Validate transport morphisms for Carter defect-absorber densities.

A transport morphism pulls an accepted trace-defect absorber through an exact
rational phase map.  The target density may exceed the pulled-back source
density by nonnegative square defects and phase-free ideal terms.  A separate
phase-norm lower certificate controls the loss in the coercivity margin.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_sos_identity as sos
import verify_carter_trace_defect_absorber as absorber


CLAIM_SCOPE = "chart_defect_transport_morphism"
ALLOWED_DERIVATION_STATUSES = {
    "candidate_transport_morphism",
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
            "Kerr derivation on the target chart; this checker verifies only the "
            "defect transport morphism"
        )
    return str(derivation_status)


def validate_string_list(raw: Any, name: str, allow_empty: bool = False) -> tuple[str, ...]:
    require(isinstance(raw, list), f"{name} must be a list")
    require(allow_empty or bool(raw), f"{name} must be nonempty")
    require(all(isinstance(item, str) and item for item in raw), f"{name} entries must be nonempty strings")
    require(len(set(raw)) == len(raw), f"{name} entries must be distinct")
    return tuple(raw)


def phase_degree(monomial: tuple[int, ...], phase_count: int) -> int:
    return sum(monomial[index] for index in range(phase_count))


def validate_phase_homogeneous(poly: sos.Polynomial, phase_count: int, degree: int, name: str) -> None:
    for monomial in poly.terms:
        require(phase_degree(monomial, phase_count) == degree, f"{name} must have target phase degree {degree}")


def validate_phase_free(poly: sos.Polynomial, phase_count: int, name: str) -> None:
    for monomial in poly.terms:
        require(phase_degree(monomial, phase_count) == 0, f"{name} must be target-phase-free")


def eval_expr_with_phase_pullback(
    expr: Any,
    variables: tuple[str, ...],
    phase_pullback_map: dict[str, Any],
    stack: tuple[str, ...] = (),
) -> sos.Polynomial:
    if isinstance(expr, str):
        if expr in phase_pullback_map:
            require(expr not in stack, f"cyclic phase pullback reference involving {expr!r}")
            return eval_expr_with_phase_pullback(phase_pullback_map[expr], variables, phase_pullback_map, stack + (expr,))
        return sos.Polynomial.variable(variables, expr)
    if isinstance(expr, int):
        return sos.Polynomial.constant(variables, sos.parse_fraction(expr))
    if isinstance(expr, dict):
        if "const" in expr:
            return sos.Polynomial.constant(variables, sos.parse_fraction(expr["const"]))
        if "var" in expr:
            name = str(expr["var"])
            if name in phase_pullback_map:
                require(name not in stack, f"cyclic phase pullback reference involving {name!r}")
                return eval_expr_with_phase_pullback(phase_pullback_map[name], variables, phase_pullback_map, stack + (name,))
            return sos.Polynomial.variable(variables, name)
        if "poly" in expr:
            raw = expr["poly"]
            require(isinstance(raw, dict), "poly expression must be an object")
            total = sos.Polynomial.zero(variables)
            for key, raw_coeff in raw.items():
                coeff = sos.parse_fraction(raw_coeff)
                monomial = sos.parse_monomial_key(key, tuple(phase_pullback_map) + variables)
                term = sos.Polynomial.constant(variables, coeff)
                for source_name, power in zip(tuple(phase_pullback_map) + variables, monomial):
                    if power == 0:
                        continue
                    base = eval_expr_with_phase_pullback({"var": source_name}, variables, phase_pullback_map, stack)
                    factor = sos.Polynomial.constant(variables, Fraction(1))
                    for _ in range(power):
                        factor = factor * base
                    term = term * factor
                total = total + term
            return total
        if "neg" in expr:
            return -eval_expr_with_phase_pullback(expr["neg"], variables, phase_pullback_map, stack)
        if "sum" in expr:
            items = expr["sum"]
            require(isinstance(items, list), "sum expression must be a list")
            total = sos.Polynomial.zero(variables)
            for item in items:
                total = total + eval_expr_with_phase_pullback(item, variables, phase_pullback_map, stack)
            return total
        if "prod" in expr:
            items = expr["prod"]
            require(isinstance(items, list), "prod expression must be a list")
            total = sos.Polynomial.constant(variables, Fraction(1))
            for item in items:
                total = total * eval_expr_with_phase_pullback(item, variables, phase_pullback_map, stack)
            return total
        if "square" in expr:
            poly = eval_expr_with_phase_pullback(expr["square"], variables, phase_pullback_map, stack)
            return poly * poly
        if "scale" in expr:
            spec = expr["scale"]
            require(isinstance(spec, dict) and "coeff" in spec and "expr" in spec, "scale expression must contain coeff and expr")
            return eval_expr_with_phase_pullback(spec["expr"], variables, phase_pullback_map, stack).scale(sos.parse_fraction(spec["coeff"]))
    raise ValueError(f"unsupported transport expression {expr!r}")


def source_variable_families_from_paths(
    source_data: dict[str, Any],
    source_path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    trace_path = resolve_path(source_data.get("trace_certificate_path"), source_path.parent, "source trace_certificate_path")
    trace_data = load_json(trace_path)
    phase_variables, symbolic_variables, source_auxiliary_variables = absorber.trace_variables(trace_data)
    defect_auxiliary_variables = validate_string_list(
        source_data.get("defect_auxiliary_variables", []),
        "source defect_auxiliary_variables",
        allow_empty=True,
    )
    return phase_variables, symbolic_variables, source_auxiliary_variables, defect_auxiliary_variables


def validate_phase_pullback_map(raw_map: Any, source_phase_variables: tuple[str, ...], variables: tuple[str, ...]) -> dict[str, Any]:
    require(isinstance(raw_map, dict), "phase_pullback_map must be an object")
    require(set(raw_map) == set(source_phase_variables), f"phase_pullback_map must contain exactly {list(source_phase_variables)}")
    pullback: dict[str, Any] = dict(raw_map)
    for source_name in source_phase_variables:
        poly = eval_expr_with_phase_pullback(pullback[source_name], variables, pullback)
        validate_phase_homogeneous(poly, len(source_phase_variables), 1, f"phase pullback {source_name!r}")
    return pullback


def ideal_polynomial(
    raw_terms: Any,
    variables: tuple[str, ...],
    phase_count: int,
    phase_pullback_map: dict[str, Any],
    name: str,
) -> tuple[sos.Polynomial, list[str]]:
    require(isinstance(raw_terms, list), f"{name} must be a list")
    total = sos.Polynomial.zero(variables)
    checked: list[str] = []
    for index, term in enumerate(raw_terms):
        require(isinstance(term, dict), f"{name}[{index}] must be an object")
        require_keys(term, ["name", "constraint_expression", "multiplier_expression"], f"{name}[{index}]")
        term_name = term["name"]
        require(isinstance(term_name, str) and term_name, f"{name}[{index}].name must be nonempty")
        constraint = eval_expr_with_phase_pullback(term["constraint_expression"], variables, phase_pullback_map)
        multiplier = eval_expr_with_phase_pullback(term["multiplier_expression"], variables, phase_pullback_map)
        validate_phase_free(constraint, phase_count, f"{name} constraint {term_name!r}")
        total = total + constraint * multiplier
        checked.append(term_name)
    return total, checked


def nonnegative_square_polynomial(
    raw_terms: Any,
    variables: tuple[str, ...],
    phase_count: int,
    phase_pullback_map: dict[str, Any],
    name: str,
) -> tuple[sos.Polynomial, list[str]]:
    require(isinstance(raw_terms, list), f"{name} must be a list")
    total = sos.Polynomial.zero(variables)
    checked: list[str] = []
    for index, term in enumerate(raw_terms):
        require(isinstance(term, dict), f"{name}[{index}] must be an object")
        require_keys(term, ["name", "coefficient", "square_expression"], f"{name}[{index}]")
        term_name = term["name"]
        require(isinstance(term_name, str) and term_name, f"{name}[{index}].name must be nonempty")
        coefficient = sos.parse_fraction(term["coefficient"])
        require(coefficient >= 0, f"{name} {term_name!r} coefficient must be nonnegative")
        linear = eval_expr_with_phase_pullback(term["square_expression"], variables, phase_pullback_map)
        validate_phase_homogeneous(linear, phase_count, 1, f"{name} {term_name!r} square_expression")
        total = total + (linear * linear).scale(coefficient)
        checked.append(term_name)
    return total, checked


def phase_norm(phase_variables: tuple[str, ...], variables: tuple[str, ...]) -> sos.Polynomial:
    total = sos.Polynomial.zero(variables)
    for variable in phase_variables:
        poly = sos.Polynomial.variable(variables, variable)
        total = total + poly * poly
    return total


def pulled_source_norm(
    source_phase_variables: tuple[str, ...],
    variables: tuple[str, ...],
    phase_pullback_map: dict[str, Any],
) -> sos.Polynomial:
    total = sos.Polynomial.zero(variables)
    for variable in source_phase_variables:
        poly = eval_expr_with_phase_pullback({"var": variable}, variables, phase_pullback_map)
        total = total + poly * poly
    return total


def verify_density_transport(
    data: dict[str, Any],
    source_data: dict[str, Any],
    variables: tuple[str, ...],
    phase_count: int,
    phase_pullback_map: dict[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    require_keys(
        data,
        ["target_density_expression", "nonnegative_transport_terms", "constraint_ideal_terms"],
        "certificate",
    )
    target_density = eval_expr_with_phase_pullback(data["target_density_expression"], variables, phase_pullback_map)
    validate_phase_homogeneous(target_density, phase_count, 2, "target_density_expression")
    source_density = eval_expr_with_phase_pullback(source_data["candidate_density_expression"], variables, phase_pullback_map)
    transport_nonnegative, checked_nonnegative = nonnegative_square_polynomial(
        data["nonnegative_transport_terms"],
        variables,
        phase_count,
        phase_pullback_map,
        "nonnegative_transport_terms",
    )
    transport_ideal, checked_constraints = ideal_polynomial(
        data["constraint_ideal_terms"],
        variables,
        phase_count,
        phase_pullback_map,
        "constraint_ideal_terms",
    )
    residual = target_density - source_density - transport_nonnegative - transport_ideal
    require(not residual.terms, f"transport density identity failed with residual {residual.to_sparse_json()}")
    return (
        ["target_density_equals_pulled_source_plus_nonnegative_transport_mod_ideal"],
        checked_constraints,
        checked_nonnegative,
    )


def verify_phase_norm_lower_bound(
    data: dict[str, Any],
    source_phase_variables: tuple[str, ...],
    target_phase_variables: tuple[str, ...],
    variables: tuple[str, ...],
    phase_pullback_map: dict[str, Any],
) -> tuple[Fraction, list[str]]:
    require("phase_norm_lower_bound" in data, "phase_norm_lower_bound is missing")
    lower_bound = sos.parse_fraction(data["phase_norm_lower_bound"])
    require(lower_bound > 0, "phase_norm_lower_bound must be positive")
    pulled_norm = pulled_source_norm(source_phase_variables, variables, phase_pullback_map)
    target_norm = phase_norm(target_phase_variables, variables)
    norm_nonnegative, checked_norm_terms = nonnegative_square_polynomial(
        data.get("phase_norm_defect_terms", []),
        variables,
        len(target_phase_variables),
        phase_pullback_map,
        "phase_norm_defect_terms",
    )
    residual = pulled_norm - target_norm.scale(lower_bound) - norm_nonnegative
    require(not residual.terms, f"phase norm lower-bound identity failed with residual {residual.to_sparse_json()}")
    return lower_bound, checked_norm_terms


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(
        data,
        ["source_chart", "target_chart", "variables", "source_defect_absorber_path", "phase_pullback_map"],
        "certificate",
    )

    source_path = resolve_path(data["source_defect_absorber_path"], base_dir, "source_defect_absorber_path")
    source_data = load_json(source_path)
    source_result = absorber.verify(source_data, base_dir=source_path.parent)
    source_chart = data["source_chart"]
    require(source_chart == source_result["chart"], "source_chart must match source defect absorber chart")
    target_chart = data["target_chart"]
    require(isinstance(target_chart, str) and target_chart, "target_chart must be a nonempty string")

    target_phase_variables = validate_string_list(data["variables"], "variables")
    source_phase_variables, symbolic_variables, source_auxiliary_variables, defect_auxiliary_variables = source_variable_families_from_paths(
        source_data,
        source_path,
    )
    transport_auxiliary_variables = validate_string_list(
        data.get("transport_auxiliary_variables", []),
        "transport_auxiliary_variables",
        allow_empty=True,
    )
    all_variables = (
        target_phase_variables
        + symbolic_variables
        + source_auxiliary_variables
        + defect_auxiliary_variables
        + transport_auxiliary_variables
    )
    require(len(set(all_variables)) == len(all_variables), "target, source, defect, and transport variables must be disjoint")
    phase_pullback_map = validate_phase_pullback_map(data["phase_pullback_map"], source_phase_variables, all_variables)

    checked_identities, checked_constraints, checked_nonnegative = verify_density_transport(
        data,
        source_data,
        all_variables,
        len(target_phase_variables),
        phase_pullback_map,
    )
    norm_lower_bound, checked_norm_terms = verify_phase_norm_lower_bound(
        data,
        source_phase_variables,
        target_phase_variables,
        all_variables,
        phase_pullback_map,
    )
    source_delta = sos.parse_fraction(source_result["local_coercivity_delta"])
    transported_delta = source_delta * norm_lower_bound

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "source_chart": source_chart,
        "target_chart": target_chart,
        "variables": list(target_phase_variables),
        "source_defect_absorber": str(source_path),
        "source_local_coercivity_delta": source_result["local_coercivity_delta"],
        "phase_norm_lower_bound": str(norm_lower_bound),
        "local_coercivity_delta": str(transported_delta),
        "checked_identities": checked_identities + source_result["checked_identities"],
        "checked_constraint_terms": checked_constraints,
        "checked_nonnegative_transport_terms": checked_nonnegative,
        "checked_phase_norm_defects": checked_norm_terms,
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
        print(f"Carter defect transport morphism rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
