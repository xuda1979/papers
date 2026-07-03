#!/usr/bin/env python3
"""Validate gravitational coefficient-enclosure candidates for a Carter chart.

This checker proves the transfer step from a coefficient enclosure to a local
coercive Carter density.  It deliberately does not derive the gravitational
principal density from the Kerr equations; a claimed gravitational proof must
come from a separate exact derivation backend, which is not registered here.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_coefficient_robustness as robustness
import verify_carter_sos_identity as sos
from verify_carter_transport_certificate import Interval, parse_interval


CLAIM_SCOPE = "chart_grav_coefficient_enclosure"
ALLOWED_DERIVATION_STATUSES = {
    "model_probe",
    "candidate_enclosure",
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


def format_fraction(value: Fraction) -> str:
    return str(value)


def interval_radius_around(interval: Interval, center: Fraction) -> Fraction:
    return max(abs(interval.lo - center), abs(interval.hi - center))


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


def load_base_chart_certificate(
    robust_data: dict[str, Any],
    robust_path: Path,
) -> tuple[dict[str, Any], Path]:
    require("base_chart_certificate_path" in robust_data, "robustness certificate lacks base_chart_certificate_path")
    base_path = resolve_path(
        robust_data["base_chart_certificate_path"],
        robust_path.parent,
        "base_chart_certificate_path",
    )
    return load_json(base_path), base_path


def base_polynomial(base_certificate: dict[str, Any], variables: tuple[str, ...]) -> sos.Polynomial:
    coercivity = base_certificate.get("coercivity_certificate")
    require(isinstance(coercivity, dict), "base chart certificate must contain coercivity_certificate")
    raw_target = coercivity.get("target_polynomial")
    require(isinstance(raw_target, dict), "base coercivity target_polynomial must be an object")
    return sos.parse_sparse_poly(raw_target, variables)


def validate_source_metadata(data: dict[str, Any], base_dir: Path | None) -> None:
    source = data.get("source_metadata")
    require(isinstance(source, dict), "source_metadata must be an object")
    require_keys(source, ["ansatz_path", "normalization", "derivation_note"], "source_metadata")
    ansatz_path = resolve_path(source["ansatz_path"], base_dir, "source_metadata.ansatz_path")
    require(ansatz_path.exists(), f"source_metadata.ansatz_path does not exist: {ansatz_path}")
    require(isinstance(source["normalization"], str) and source["normalization"], "source_metadata.normalization must be a nonempty string")
    require(isinstance(source["derivation_note"], str) and source["derivation_note"], "source_metadata.derivation_note must be a nonempty string")


def expected_quadratic_keys(variables: tuple[str, ...], coefficient_bounds: dict[str, Any]) -> set[str]:
    expected: set[str] = set()
    for key in coefficient_bounds:
        monomial = sos.parse_monomial_key(key, variables)
        require(sum(monomial) == 2, f"coefficient bound {key!r} must be quadratic")
        expected.add(key)
    return expected


def validate_coefficient_intervals(
    raw_intervals: Any,
    variables: tuple[str, ...],
    base_coefficients: sos.Polynomial,
    coefficient_bounds: dict[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    require(isinstance(raw_intervals, dict) and raw_intervals, "coefficient_intervals must be a nonempty object")
    expected = expected_quadratic_keys(variables, coefficient_bounds)
    actual = set(raw_intervals)
    require(actual == expected, f"coefficient_intervals must contain exactly {sorted(expected)}")

    deviations: dict[str, str] = {}
    bounds: dict[str, str] = {}
    for key in sorted(expected):
        interval = parse_interval(raw_intervals[key], f"coefficient_intervals.{key}")
        bound = sos.parse_fraction(coefficient_bounds[key])
        require(bound >= 0, f"coefficient bound for {key!r} must be nonnegative")
        center = base_coefficients.terms.get(sos.parse_monomial_key(key, variables), Fraction(0))
        deviation = interval_radius_around(interval, center)
        require(
            deviation <= bound,
            f"coefficient interval {key!r} leaves certified tube: deviation {deviation} exceeds bound {bound}",
        )
        deviations[key] = format_fraction(deviation)
        bounds[key] = format_fraction(bound)
    return deviations, bounds


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    derivation_status = validate_metadata(data)
    require_keys(data, ["chart", "variables", "robustness_certificate_path", "coefficient_intervals"], "certificate")
    validate_source_metadata(data, base_dir)

    robust_path = resolve_path(data["robustness_certificate_path"], base_dir, "robustness_certificate_path")
    robust_data = load_json(robust_path)
    robust_result = robustness.verify(robust_data, base_dir=robust_path.parent)
    base_certificate, _ = load_base_chart_certificate(robust_data, robust_path)

    variables_raw = data["variables"]
    require(isinstance(variables_raw, list) and variables_raw, "variables must be a nonempty list")
    require(all(isinstance(variable, str) for variable in variables_raw), "variables must be strings")
    variables = tuple(variables_raw)
    require(set(variables) == set(robust_result["variables"]), "variables must match robustness certificate variables")

    chart = data["chart"]
    require(isinstance(chart, str) and chart, "chart must be a nonempty string")
    require(chart == robust_result["base_chart"], "certificate chart must match robustness base_chart")

    coefficient_bounds = robust_data.get("coefficient_bounds")
    require(isinstance(coefficient_bounds, dict) and coefficient_bounds, "robustness coefficient_bounds must be a nonempty object")
    base_coefficients = base_polynomial(base_certificate, variables)
    deviations, bounds = validate_coefficient_intervals(
        data["coefficient_intervals"],
        variables,
        base_coefficients,
        coefficient_bounds,
    )

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": derivation_status,
        "proves_kerr_gravitational_density": False,
        "chart": chart,
        "variables": list(variables),
        "robustness_certificate": str(robust_path),
        "base_coercivity_delta": robust_result["base_coercivity_delta"],
        "local_coercivity_delta": robust_result["target_final_delta"],
        "coefficient_deviations": deviations,
        "tube_bounds": bounds,
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
        print(f"Carter gravitational coefficient enclosure rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
