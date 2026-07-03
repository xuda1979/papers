#!/usr/bin/env python3
"""Validate a chart-level Carter SOS-plus-transport certificate.

This checker connects the symbolic Carter chart ansatz to the exact polynomial
identity checker. It verifies that the chart metadata, variables, monomial
basis, exact SOS identity, and rational interval margins are mutually
consistent. It does not solve for Gram coefficients or prove the Kerr cone
lemma by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_carter_sos_identity as sos_identity
import verify_carter_gram_certificate as gram_certificate
from verify_carter_transport_certificate import parse_interval


REQUIRED_INTERVAL_FIELDS = [
    "lambda",
    "eta",
    "superradiant_factor",
    "radial_hessian",
    "trapped_scale",
]

CHART_CLAIM_SCOPE = "chart_level_carter_cone_certificate"

ALLOWED_TARGET_DENSITY_KINDS = {
    "protocol_toy",
    "chart_local_carter_normalized_model",
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


def load_ansatz(data: dict[str, Any], base_dir: Path | None) -> dict[str, Any]:
    has_embedded = "ansatz" in data
    has_path = "ansatz_path" in data
    require(has_embedded != has_path, "certificate must contain exactly one of ansatz or ansatz_path")
    if has_embedded:
        ansatz = data["ansatz"]
        require(isinstance(ansatz, dict), "ansatz must be an object")
        return ansatz

    raw_path = data["ansatz_path"]
    require(isinstance(raw_path, str) and raw_path.strip(), "ansatz_path must be a nonempty string")
    path = Path(raw_path)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return load_json(path)


def ansatz_variables(ansatz: dict[str, Any]) -> tuple[str, ...]:
    variables = ansatz.get("variables")
    require(isinstance(variables, dict) and variables, "ansatz.variables must be a nonempty object")
    names = tuple(variables.keys())
    require(all(isinstance(name, str) for name in names), "ansatz variable names must be strings")
    return names


def validate_ansatz_shape(ansatz: dict[str, Any], chart: str, variables: tuple[str, ...]) -> None:
    require(ansatz.get("chart") == chart, "certificate chart must match ansatz.chart")
    require(isinstance(ansatz.get("metadata"), dict), "ansatz.metadata must be an object")
    require_keys(ansatz["metadata"], ["name", "status"], "ansatz.metadata")
    require(isinstance(ansatz.get("target_identity"), str) and ansatz["target_identity"], "ansatz.target_identity must be a nonempty string")

    r_interval = parse_interval(ansatz.get("r_interval"), "ansatz.r_interval")
    require(r_interval.positive, "ansatz.r_interval must be positive")

    interval_data = ansatz.get("interval_data")
    require(isinstance(interval_data, dict), "ansatz.interval_data must be an object")
    require_keys(interval_data, REQUIRED_INTERVAL_FIELDS, "ansatz.interval_data")
    for field in REQUIRED_INTERVAL_FIELDS:
        interval = parse_interval(interval_data[field], f"ansatz.interval_data.{field}")
        if field == "trapped_scale":
            require(interval.positive, "ansatz.interval_data.trapped_scale must be positive")

    basis = ansatz.get("monomial_basis")
    require(isinstance(basis, list) and basis, "ansatz.monomial_basis must be a nonempty list")
    require(basis[0] == "1", "ansatz.monomial_basis must start with 1")
    seen = set()
    for index, monomial in enumerate(basis):
        path = f"ansatz.monomial_basis[{index}]"
        require(isinstance(monomial, str) and monomial.strip(), f"{path} must be a nonempty string")
        parsed = sos_identity.parse_monomial_key(monomial, variables)
        require(parsed not in seen, f"{path} duplicates an earlier monomial")
        seen.add(parsed)
    for variable in variables:
        require(variable in basis, f"ansatz.monomial_basis must include variable {variable!r}")


def validate_identity(data: dict[str, Any], variables: tuple[str, ...]) -> dict[str, Any]:
    identity = data.get("identity_certificate")
    require(isinstance(identity, dict), "identity_certificate must be an object")
    raw_variables = identity.get("variables")
    require(isinstance(raw_variables, list), "identity_certificate.variables must be a list")
    require(all(isinstance(variable, str) for variable in raw_variables), "identity variables must be strings")
    require(len(raw_variables) == len(set(raw_variables)), "identity variables must be distinct")
    require(set(raw_variables) == set(variables), "identity variables must match ansatz variables")
    return sos_identity.verify(identity)


def validate_margins(data: dict[str, Any]) -> tuple[Any, Any, Any]:
    margins = data.get("margins")
    require(isinstance(margins, dict), "margins must be an object")
    require_keys(margins, ["delta", "remainder_bound", "positive_margin"], "margins")
    delta = parse_interval(margins["delta"], "margins.delta")
    remainder = parse_interval(margins["remainder_bound"], "margins.remainder_bound")
    positive = parse_interval(margins["positive_margin"], "margins.positive_margin")
    require(delta.positive, "margins.delta must be positive")
    require(remainder.nonnegative, "margins.remainder_bound must be nonnegative")
    require(positive.positive, "margins.positive_margin must be positive")
    require(positive.dominates(remainder), "margins.positive_margin must dominate remainder_bound")
    return delta, remainder, positive


def validate_gram(data: dict[str, Any], variables: tuple[str, ...]) -> dict[str, Any] | None:
    gram = data.get("gram_certificate")
    if gram is None:
        return None
    require(isinstance(gram, dict), "gram_certificate must be an object")
    raw_variables = gram.get("variables")
    require(isinstance(raw_variables, list), "gram_certificate.variables must be a list")
    require(all(isinstance(variable, str) for variable in raw_variables), "Gram variables must be strings")
    require(set(raw_variables) == set(variables), "Gram variables must match ansatz variables")
    return gram_certificate.verify(gram)


def validate_metadata(data: dict[str, Any]) -> tuple[str, str]:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    require_keys(metadata, ["name", "status", "claim_scope", "target_density_kind"], "metadata")
    claim_scope = metadata["claim_scope"]
    target_kind = metadata["target_density_kind"]
    require(claim_scope == CHART_CLAIM_SCOPE, f"metadata.claim_scope must be {CHART_CLAIM_SCOPE}")
    require(isinstance(target_kind, str) and target_kind.strip(), "metadata.target_density_kind must be a nonempty string")
    if target_kind == "kerr_gravitational":
        raise ValueError(
            "metadata.target_density_kind kerr_gravitational requires an exact Kerr derivation backend; "
            "this checker only accepts explicit toy/model densities"
        )
    require(
        target_kind in ALLOWED_TARGET_DENSITY_KINDS,
        f"metadata.target_density_kind must be one of {sorted(ALLOWED_TARGET_DENSITY_KINDS)}",
    )
    return claim_scope, target_kind


def validate_coercivity(
    data: dict[str, Any],
    variables: tuple[str, ...],
    margin_delta: Any,
) -> dict[str, Any] | None:
    coercivity = data.get("coercivity_certificate")
    if coercivity is None:
        return None
    require(isinstance(coercivity, dict), "coercivity_certificate must be an object")
    require_keys(
        coercivity,
        [
            "variables",
            "target_polynomial",
            "control_polynomial",
            "delta",
            "residual_polynomial",
            "residual_gram_certificate",
        ],
        "coercivity_certificate",
    )
    raw_variables = coercivity["variables"]
    require(isinstance(raw_variables, list), "coercivity_certificate.variables must be a list")
    require(all(isinstance(variable, str) for variable in raw_variables), "coercivity variables must be strings")
    require(set(raw_variables) == set(variables), "coercivity variables must match ansatz variables")

    target = sos_identity.parse_sparse_poly(coercivity["target_polynomial"], variables)
    control = sos_identity.parse_sparse_poly(coercivity["control_polynomial"], variables)
    residual = sos_identity.parse_sparse_poly(coercivity["residual_polynomial"], variables)
    delta = sos_identity.parse_fraction(coercivity["delta"])
    require(delta > 0, "coercivity_certificate.delta must be positive")
    require(margin_delta.lo <= delta <= margin_delta.hi, "margins.delta must contain coercivity_certificate.delta")

    coercivity_residual = target - control.scale(delta) - residual
    require(
        not coercivity_residual.terms,
        f"coercivity residual identity failed with residual {coercivity_residual.to_sparse_json()}",
    )

    residual_gram = coercivity["residual_gram_certificate"]
    require(isinstance(residual_gram, dict), "coercivity_certificate.residual_gram_certificate must be an object")
    gram_target_raw = residual_gram.get("target_polynomial")
    require(isinstance(gram_target_raw, dict), "coercivity residual Gram target_polynomial must be an object")
    gram_target = sos_identity.parse_sparse_poly(gram_target_raw, variables)
    gram_target_residual = gram_target - residual
    require(
        not gram_target_residual.terms,
        f"coercivity residual Gram target mismatch {gram_target_residual.to_sparse_json()}",
    )
    gram_result = gram_certificate.verify(residual_gram)
    return {
        "coercivity_delta": str(delta),
        "coercivity_residual_gram_basis_size": gram_result["basis_size"],
        "coercivity_checked_principal_minors": gram_result["checked_principal_minors"],
    }


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    claim_scope, target_kind = validate_metadata(data)
    chart = data.get("chart")
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")

    ansatz = load_ansatz(data, base_dir)
    variables = ansatz_variables(ansatz)
    validate_ansatz_shape(ansatz, chart, variables)
    identity_result = validate_identity(data, variables)
    gram_result = validate_gram(data, variables)
    margin_delta, _, _ = validate_margins(data)
    coercivity_result = validate_coercivity(data, variables, margin_delta)
    result = {
        "accepted": True,
        "claim_scope": claim_scope,
        "target_density_kind": target_kind,
        "chart": chart,
        "variables": list(variables),
        "checked_identities": identity_result["checked_identities"],
    }
    if gram_result is not None:
        result["gram_basis_size"] = gram_result["basis_size"]
        result["checked_principal_minors"] = gram_result["checked_principal_minors"]
    if coercivity_result is not None:
        result.update(coercivity_result)
    return result


def verify_file(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return verify(data, base_dir=path.parent)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except ValueError as exc:
        print(f"chart certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
