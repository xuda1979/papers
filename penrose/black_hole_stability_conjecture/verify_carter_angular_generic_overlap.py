#!/usr/bin/env python3
"""Verify generic angular overlaps for the spin-half seed tubes.

This checker is deliberately local.  It certifies positive-length overlap
boxes where the polar/equatorial blow-up variables can be converted back to
ordinary Carter angular variables with all transition denominators bounded
away from zero.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from generate_carter_atlas import Interval, parse_interval


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def point(value: Fraction | int) -> Interval:
    return Interval(Fraction(value), Fraction(value))


def interval_json(interval: Interval) -> str:
    return interval.to_json()


def interval_contains(container: Interval, contained: Interval) -> bool:
    return container.lo <= contained.lo and contained.hi <= container.hi


def interval_strictly_positive(interval: Interval) -> bool:
    return interval.lo > 0


def positive_length(interval: Interval) -> bool:
    return interval.lo < interval.hi


def square(interval: Interval) -> Interval:
    return interval.square()


def one_minus(interval: Interval) -> Interval:
    return point(1) - interval


def required_interval(data: dict[str, Any], keys: list[str], name: str) -> Interval:
    current: Any = data
    for key in keys:
        require(isinstance(current, dict) and key in current, f"{name} missing {'.'.join(keys)}")
        current = current[key]
    return parse_interval(current)


def symbolic_identities() -> dict[str, str]:
    eta, a, lam, u, v, mu2, zeta = sp.symbols("eta a lambda u v mu2 zeta")
    polar_ordinary = eta + a**2 * (1 - v) - lam**2 * (1 - v) / v
    polar_regular = eta + (1 - v) * (a**2 - mu2)
    equatorial_ordinary = eta + a**2 * u - lam**2 * u / (1 - u)
    equatorial_factor = ((1 - eta * zeta) * (1 + a**2 * zeta) - lam**2 * zeta) / (
        1 - eta * zeta
    )
    identities = {
        "polar_generic_inverse": sp.factor(
            polar_ordinary - polar_regular.subs(mu2, lam**2 / v)
        ),
        "equatorial_generic_inverse": sp.factor(
            equatorial_ordinary - (eta * equatorial_factor).subs(zeta, u / eta)
        ),
    }
    failed = {name: str(value) for name, value in identities.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"generic angular overlap symbolic identity failed: {failed}")
    return {name: "0" for name in sorted(identities)}


def require_package(data: dict[str, Any], name: str) -> None:
    require(data.get("accepted") is True, f"{name} package is not accepted")
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), f"{name} metadata must be an object")
    require(metadata.get("spin") == "1/2", f"{name} package must target spin one-half")


def records_by_name(data: dict[str, Any], section: str) -> dict[str, dict[str, Any]]:
    records = data.get(section)
    require(isinstance(records, list), f"{section} must be a list")
    result: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"{section}[{index}] must be an object")
        name = record.get("name")
        require(isinstance(name, str) and name, f"{section}[{index}].name must be a nonempty string")
        require(name not in result, f"duplicate {section} record {name}")
        result[name] = record
    return result


def require_record(records: dict[str, dict[str, Any]], name: str, section: str) -> dict[str, Any]:
    require(name in records, f"{section} missing {name}")
    return records[name]


def verify_polar_generic_overlap(seed_tube: dict[str, Any]) -> dict[str, Any]:
    name = "polar_seed_to_generic_ordinary_overlap"
    a_squared = point(Fraction(1, 4))
    seed_lambda = required_interval(seed_tube, ["lambda_tube_interval"], name)
    bridge_lambda = required_interval(seed_tube, ["bridge_lambda_interval"], name)
    eta = required_interval(seed_tube, ["bridge_eta_interval"], name)
    seed_v = required_interval(seed_tube, ["sin_squared_theta_corridor"], name)
    seed_mu_squared = required_interval(seed_tube, ["mu_squared_interval"], name)
    ordinary_annulus = required_interval(seed_tube, ["ordinary_sin_squared_theta_annulus"], name)

    lambda_overlap = Interval(Fraction(-1, 12000), Fraction(1, 12000))
    v_overlap = Interval(Fraction(1, 64), Fraction(1, 16))
    inverse_mu_squared = square(lambda_overlap) / v_overlap
    one_minus_v = one_minus(v_overlap)
    ordinary_theta = eta + a_squared * one_minus_v - (square(lambda_overlap) / v_overlap) * one_minus_v
    regular_theta = eta + one_minus_v * (a_squared - inverse_mu_squared)

    checks = {
        "lambda_overlap_has_positive_length": positive_length(lambda_overlap),
        "sin_squared_theta_overlap_has_positive_length": positive_length(v_overlap),
        "generic_lambda_overlap_inside_seed_tube": interval_contains(seed_lambda, lambda_overlap),
        "generic_lambda_overlap_inside_radial_bridge": interval_contains(bridge_lambda, lambda_overlap),
        "generic_v_overlap_inside_regular_corridor": interval_contains(seed_v, v_overlap),
        "generic_v_overlap_inside_ordinary_annulus": interval_contains(ordinary_annulus, v_overlap),
        "inverse_mu_squared_inside_seed_tube": interval_contains(seed_mu_squared, inverse_mu_squared),
        "polar_transition_denominator_positive": interval_strictly_positive(v_overlap),
        "ordinary_polar_Theta_positive_on_generic_overlap": interval_strictly_positive(ordinary_theta),
        "regularized_polar_Theta_positive_on_generic_overlap": interval_strictly_positive(regular_theta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} checks failed: {failed}")

    return {
        "name": name,
        "seed_tube": seed_tube["name"],
        "seed_family": seed_tube["seed_family"],
        "overlap_type": "polar_regular_to_generic_ordinary",
        "generic_lambda_overlap_interval": interval_json(lambda_overlap),
        "generic_sin_squared_theta_overlap_interval": interval_json(v_overlap),
        "inverse_mu_squared_interval": interval_json(inverse_mu_squared),
        "transition_denominator_interval": interval_json(v_overlap),
        "bridge_eta_interval": interval_json(eta),
        "ordinary_Theta_interval": interval_json(ordinary_theta),
        "regularized_Theta_interval": interval_json(regular_theta),
        "checked_conditions": sorted(checks),
    }


def verify_equatorial_generic_overlap(seed_tube: dict[str, Any], name: str) -> dict[str, Any]:
    a_squared = point(Fraction(1, 4))
    seed_eta = required_interval(seed_tube, ["eta_tube_interval"], name)
    seed_zeta = required_interval(seed_tube, ["zeta_tube_interval"], name)
    ordinary_eta = required_interval(seed_tube, ["ordinary_eta_overlap_interval"], name)
    bridge_lambda = required_interval(seed_tube, ["bridge_lambda_interval"], name)

    eta_overlap = Interval(Fraction(1, 8192), Fraction(1, 1024))
    u_overlap = Interval(Fraction(0), Fraction(1, 393216))
    inverse_zeta = u_overlap / eta_overlap
    denominator = one_minus(u_overlap)
    lambda_squared = square(bridge_lambda)
    ordinary_theta = eta_overlap + a_squared * u_overlap - (lambda_squared * u_overlap / denominator)
    factor = (denominator * (point(1) + a_squared * inverse_zeta) - lambda_squared * inverse_zeta) / denominator
    regular_theta = eta_overlap * factor

    checks = {
        "eta_overlap_has_positive_length": positive_length(eta_overlap),
        "cos_squared_theta_overlap_has_positive_length": positive_length(u_overlap),
        "generic_eta_overlap_inside_seed_tube": interval_contains(seed_eta, eta_overlap),
        "generic_eta_overlap_inside_ordinary_overlap": interval_contains(ordinary_eta, eta_overlap),
        "inverse_zeta_inside_seed_tube": interval_contains(seed_zeta, inverse_zeta),
        "equatorial_transition_jacobian_positive": interval_strictly_positive(eta_overlap),
        "one_minus_cos_squared_theta_positive_on_generic_overlap": interval_strictly_positive(denominator),
        "ordinary_equatorial_Theta_positive_on_generic_overlap": interval_strictly_positive(ordinary_theta),
        "regularized_equatorial_factor_positive_on_generic_overlap": interval_strictly_positive(factor),
        "regularized_equatorial_Theta_positive_on_generic_overlap": interval_strictly_positive(regular_theta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} checks failed: {failed}")

    return {
        "name": name,
        "seed_tube": seed_tube["name"],
        "seed_family": seed_tube["seed_family"],
        "overlap_type": "equatorial_blowup_to_generic_ordinary",
        "generic_eta_overlap_interval": interval_json(eta_overlap),
        "generic_cos_squared_theta_overlap_interval": interval_json(u_overlap),
        "inverse_zeta_interval": interval_json(inverse_zeta),
        "transition_jacobian_interval": interval_json(eta_overlap),
        "bridge_lambda_interval": interval_json(bridge_lambda),
        "ordinary_Theta_interval": interval_json(ordinary_theta),
        "regularized_factor_interval": interval_json(factor),
        "regularized_Theta_interval": interval_json(regular_theta),
        "checked_conditions": sorted(checks),
    }


def compute_certificate(seed_package: dict[str, Any], source_path: str | None = None) -> dict[str, Any]:
    require_package(seed_package, "angular seed-tube")
    polar_seed_tubes = records_by_name(seed_package, "polar_seed_tubes")
    equatorial_seed_tubes = records_by_name(seed_package, "equatorial_seed_tubes")

    polar_overlaps = [
        verify_polar_generic_overlap(
            require_record(
                polar_seed_tubes,
                "polar_two_sided_parameter_seed_tube",
                "polar_seed_tubes",
            )
        )
    ]
    equatorial_overlaps = [
        verify_equatorial_generic_overlap(
            require_record(
                equatorial_seed_tubes,
                "equatorial_prograde_eta_zeta_seed_tube",
                "equatorial_seed_tubes",
            ),
            "equatorial_prograde_seed_to_generic_ordinary_overlap",
        ),
        verify_equatorial_generic_overlap(
            require_record(
                equatorial_seed_tubes,
                "equatorial_retrograde_eta_zeta_seed_tube",
                "equatorial_seed_tubes",
            ),
            "equatorial_retrograde_seed_to_generic_ordinary_overlap",
        ),
    ]
    metadata: dict[str, Any] = {
        "name": "spin-half angular generic-overlap certificate",
        "spin": "1/2",
        "mass_normalization": "M=1",
        "status": "local generic angular overlaps for seed tubes only; full Kerr angular atlas not proved",
        "warning": "This is not a Kerr stability proof certificate.",
    }
    if source_path is not None:
        metadata["angular_seed_tube_path"] = source_path
    return {
        "accepted": True,
        "metadata": metadata,
        "symbolic_identities": symbolic_identities(),
        "polar_generic_overlaps": polar_overlaps,
        "equatorial_generic_overlaps": equatorial_overlaps,
    }


def resolved_seed_path(package_path: Path, raw: Any) -> Path:
    require(isinstance(raw, str) and raw, "metadata.angular_seed_tube_path must be a nonempty path")
    path = Path(raw)
    if not path.is_absolute():
        path = package_path.parent / path
    return path


def verify_package(path: Path) -> dict[str, Any]:
    stored = load_json(path)
    metadata = stored.get("metadata")
    require(isinstance(metadata, dict), "angular generic-overlap metadata must be an object")
    source_path = str(metadata.get("angular_seed_tube_path", ""))
    seed_package = load_json(resolved_seed_path(path, source_path))
    computed = compute_certificate(seed_package, source_path=source_path)
    require(stored == computed, "angular generic-overlap certificate is stale or incorrect")
    return summary(computed)


def summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "polar_generic_overlap_count": len(certificate["polar_generic_overlaps"]),
        "equatorial_generic_overlap_count": len(certificate["equatorial_generic_overlaps"]),
        "polar_generic_overlaps": [
            overlap["name"] for overlap in certificate["polar_generic_overlaps"]
        ],
        "equatorial_generic_overlaps": [
            overlap["name"] for overlap in certificate["equatorial_generic_overlaps"]
        ],
    }


def relative_source_path(output: Path, source: Path) -> str:
    return os.path.relpath(source.resolve(), output.parent.resolve())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        if args.output is None:
            print(json.dumps(verify_package(args.input), indent=2, sort_keys=True))
            return 0
        seed_package = load_json(args.input)
        certificate = compute_certificate(
            seed_package,
            source_path=relative_source_path(args.output, args.input),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter angular generic-overlap rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
