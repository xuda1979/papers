#!/usr/bin/env python3
"""Verify a seed-level angular overlap policy for the spin-half Carter charts.

The policy upgrades the local rectangular generic-overlap doorways to dependent
coordinate transitions on the full seed overlap bands.  It still concerns only
the spin-half seed neighborhoods; it is not a full Kerr angular-torus atlas.
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


def interval_nonnegative(interval: Interval) -> bool:
    return interval.lo >= 0


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


def covers_target(intervals: list[Interval], target: Interval) -> bool:
    cursor = target.lo
    for interval in sorted(intervals, key=lambda item: (item.lo, item.hi)):
        if interval.hi < cursor:
            continue
        if interval.lo > cursor:
            return False
        if interval.hi > cursor:
            cursor = interval.hi
        if cursor >= target.hi:
            return True
    return cursor >= target.hi


def symbolic_identities() -> dict[str, str]:
    eta, a, lam, u, v, mu2, zeta = sp.symbols("eta a lambda u v mu2 zeta")
    polar_ordinary = eta + a**2 * (1 - v) - lam**2 * (1 - v) / v
    polar_regular = eta + (1 - v) * (a**2 - mu2)
    equatorial_ordinary = eta + a**2 * u - lam**2 * u / (1 - u)
    equatorial_factor = ((1 - eta * zeta) * (1 + a**2 * zeta) - lam**2 * zeta) / (
        1 - eta * zeta
    )
    identities = {
        "polar_dependent_overlap": sp.factor(
            polar_ordinary.subs(lam**2, v * mu2) - polar_regular
        ),
        "equatorial_dependent_overlap": sp.factor(
            equatorial_ordinary.subs(u, eta * zeta) - eta * equatorial_factor
        ),
    }
    failed = {name: str(value) for name, value in identities.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"angular overlap-policy identity failed: {failed}")
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


def verify_polar_policy(seed_tube: dict[str, Any], doorway: dict[str, Any]) -> dict[str, Any]:
    name = "polar_seed_dependent_overlap_policy"
    a_squared = point(Fraction(1, 4))
    eta = required_interval(seed_tube, ["bridge_eta_interval"], name)
    seed_lambda = required_interval(seed_tube, ["lambda_tube_interval"], name)
    bridge_lambda = required_interval(seed_tube, ["bridge_lambda_interval"], name)
    regular_v = required_interval(seed_tube, ["sin_squared_theta_corridor"], name)
    mu_squared = required_interval(seed_tube, ["mu_squared_interval"], name)
    ordinary_v = required_interval(seed_tube, ["ordinary_sin_squared_theta_annulus"], name)
    stored_regular_theta = required_interval(seed_tube, ["regularized_Theta_interval"], name)
    stored_ordinary_theta = required_interval(seed_tube, ["ordinary_Theta_full_bridge_lambda_interval"], name)

    target_v = Interval(Fraction(0), Fraction(1))
    overlap_v = Interval(Fraction(1, 64), Fraction(1, 16))
    dependent_lambda_squared = overlap_v * mu_squared
    dependent_lambda = Interval(Fraction(-1, 6000), Fraction(1, 6000))
    one_minus_v = one_minus(overlap_v)
    regular_theta = eta + one_minus_v * (a_squared - mu_squared)
    ordinary_theta = eta + a_squared * one_minus_v - (dependent_lambda_squared / overlap_v) * one_minus_v

    doorway_lambda = required_interval(doorway, ["generic_lambda_overlap_interval"], name)
    doorway_v = required_interval(doorway, ["generic_sin_squared_theta_overlap_interval"], name)
    doorway_mu_squared = required_interval(doorway, ["inverse_mu_squared_interval"], name)

    checks = {
        "polar_charts_cover_full_sin_squared_theta_interval": covers_target([regular_v, ordinary_v], target_v),
        "polar_overlap_band_has_positive_length": positive_length(overlap_v),
        "regular_chart_contains_overlap_band": interval_contains(regular_v, overlap_v),
        "ordinary_chart_contains_overlap_band": interval_contains(ordinary_v, overlap_v),
        "dependent_lambda_inside_seed_tube": interval_contains(seed_lambda, dependent_lambda),
        "dependent_lambda_inside_radial_bridge": interval_contains(bridge_lambda, dependent_lambda),
        "dependent_lambda_square_fits_coordinate_map": square(dependent_lambda).hi <= dependent_lambda_squared.hi,
        "polar_transition_denominator_positive": interval_strictly_positive(overlap_v),
        "dependent_regularized_Theta_positive": interval_strictly_positive(regular_theta),
        "dependent_ordinary_Theta_positive": interval_strictly_positive(ordinary_theta),
        "dependent_regularized_Theta_inside_seed_margin": interval_contains(stored_regular_theta, regular_theta),
        "ordinary_annulus_margin_remains_positive": interval_strictly_positive(stored_ordinary_theta),
        "rectangular_doorway_lambda_inside_policy": interval_contains(dependent_lambda, doorway_lambda),
        "rectangular_doorway_v_inside_policy": interval_contains(overlap_v, doorway_v),
        "rectangular_doorway_mu_squared_inside_policy": interval_contains(mu_squared, doorway_mu_squared),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} checks failed: {failed}")

    return {
        "name": name,
        "seed_family": seed_tube["seed_family"],
        "policy_type": "polar_regular_axis_plus_ordinary_annulus",
        "regular_sin_squared_theta_interval": interval_json(regular_v),
        "ordinary_sin_squared_theta_interval": interval_json(ordinary_v),
        "target_sin_squared_theta_interval": interval_json(target_v),
        "dependent_overlap_sin_squared_theta_interval": interval_json(overlap_v),
        "dependent_mu_squared_interval": interval_json(mu_squared),
        "dependent_lambda_interval": interval_json(dependent_lambda),
        "dependent_lambda_squared_interval": interval_json(dependent_lambda_squared),
        "transition_denominator_interval": interval_json(overlap_v),
        "dependent_regularized_Theta_interval": interval_json(regular_theta),
        "dependent_ordinary_Theta_interval": interval_json(ordinary_theta),
        "rectangular_doorway": doorway["name"],
        "checked_conditions": sorted(checks),
    }


def verify_equatorial_policy(
    seed_tube: dict[str, Any],
    doorway: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    a_squared = point(Fraction(1, 4))
    eta_tube = required_interval(seed_tube, ["eta_tube_interval"], name)
    zeta_tube = required_interval(seed_tube, ["zeta_tube_interval"], name)
    ordinary_eta = required_interval(seed_tube, ["ordinary_eta_overlap_interval"], name)
    bridge_lambda = required_interval(seed_tube, ["bridge_lambda_interval"], name)
    stored_factor = required_interval(seed_tube, ["ordinary_factor_interval"], name)
    stored_theta = required_interval(seed_tube, ["ordinary_Theta_interval"], name)

    eta_overlap = Interval(Fraction(1, 8192), Fraction(1, 1024))
    zeta_overlap = zeta_tube
    u_image = eta_overlap * zeta_overlap
    denominator = one_minus(u_image)
    lambda_squared = square(bridge_lambda)
    factor = (denominator * (point(1) + a_squared * zeta_overlap) - lambda_squared * zeta_overlap) / denominator
    regular_theta = eta_overlap * factor
    ordinary_theta = regular_theta

    doorway_eta = required_interval(doorway, ["generic_eta_overlap_interval"], name)
    doorway_u = required_interval(doorway, ["generic_cos_squared_theta_overlap_interval"], name)
    doorway_zeta = required_interval(doorway, ["inverse_zeta_interval"], name)

    checks = {
        "eta_overlap_has_positive_length": positive_length(eta_overlap),
        "zeta_overlap_has_positive_length": positive_length(zeta_overlap),
        "eta_tube_contains_overlap": interval_contains(eta_tube, eta_overlap),
        "ordinary_eta_record_contains_overlap": interval_contains(ordinary_eta, eta_overlap),
        "zeta_tube_contains_overlap": interval_contains(zeta_tube, zeta_overlap),
        "equatorial_transition_jacobian_positive": interval_strictly_positive(eta_overlap),
        "ordinary_cos_squared_theta_image_nonnegative": interval_nonnegative(u_image),
        "one_minus_cos_squared_theta_positive": interval_strictly_positive(denominator),
        "dependent_regularized_factor_positive": interval_strictly_positive(factor),
        "dependent_regularized_Theta_positive": interval_strictly_positive(regular_theta),
        "dependent_ordinary_Theta_positive": interval_strictly_positive(ordinary_theta),
        "dependent_factor_inside_seed_margin": interval_contains(stored_factor, factor),
        "dependent_Theta_inside_seed_margin": interval_contains(stored_theta, ordinary_theta),
        "rectangular_doorway_eta_inside_policy": interval_contains(eta_overlap, doorway_eta),
        "rectangular_doorway_u_inside_policy": interval_contains(u_image, doorway_u),
        "rectangular_doorway_zeta_inside_policy": interval_contains(zeta_overlap, doorway_zeta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} checks failed: {failed}")

    return {
        "name": name,
        "seed_family": seed_tube["seed_family"],
        "policy_type": "equatorial_blowup_with_ordinary_eta_collar",
        "eta_tube_interval": interval_json(eta_tube),
        "zeta_tube_interval": interval_json(zeta_tube),
        "ordinary_eta_collar_interval": interval_json(eta_overlap),
        "dependent_zeta_overlap_interval": interval_json(zeta_overlap),
        "ordinary_cos_squared_theta_image_interval": interval_json(u_image),
        "transition_jacobian_interval": interval_json(eta_overlap),
        "one_minus_cos_squared_theta_interval": interval_json(denominator),
        "dependent_regularized_factor_interval": interval_json(factor),
        "dependent_regularized_Theta_interval": interval_json(regular_theta),
        "dependent_ordinary_Theta_interval": interval_json(ordinary_theta),
        "rectangular_doorway": doorway["name"],
        "checked_conditions": sorted(checks),
    }


def compute_certificate(
    generic_package: dict[str, Any],
    seed_package: dict[str, Any],
    source_path: str | None = None,
) -> dict[str, Any]:
    require_package(generic_package, "angular generic-overlap")
    require_package(seed_package, "angular seed-tube")
    polar_seed_tubes = records_by_name(seed_package, "polar_seed_tubes")
    equatorial_seed_tubes = records_by_name(seed_package, "equatorial_seed_tubes")
    polar_doorways = records_by_name(generic_package, "polar_generic_overlaps")
    equatorial_doorways = records_by_name(generic_package, "equatorial_generic_overlaps")

    polar_policy = [
        verify_polar_policy(
            require_record(
                polar_seed_tubes,
                "polar_two_sided_parameter_seed_tube",
                "polar_seed_tubes",
            ),
            require_record(
                polar_doorways,
                "polar_seed_to_generic_ordinary_overlap",
                "polar_generic_overlaps",
            ),
        )
    ]
    equatorial_policy = [
        verify_equatorial_policy(
            require_record(
                equatorial_seed_tubes,
                "equatorial_prograde_eta_zeta_seed_tube",
                "equatorial_seed_tubes",
            ),
            require_record(
                equatorial_doorways,
                "equatorial_prograde_seed_to_generic_ordinary_overlap",
                "equatorial_generic_overlaps",
            ),
            "equatorial_prograde_dependent_overlap_policy",
        ),
        verify_equatorial_policy(
            require_record(
                equatorial_seed_tubes,
                "equatorial_retrograde_eta_zeta_seed_tube",
                "equatorial_seed_tubes",
            ),
            require_record(
                equatorial_doorways,
                "equatorial_retrograde_seed_to_generic_ordinary_overlap",
                "equatorial_generic_overlaps",
            ),
            "equatorial_retrograde_dependent_overlap_policy",
        ),
    ]
    metadata: dict[str, Any] = {
        "name": "spin-half angular seed overlap-policy certificate",
        "spin": "1/2",
        "mass_normalization": "M=1",
        "policy_scope": "polar and equatorial seed neighborhoods only",
        "status": "seed-level dependent angular overlap policy; full Kerr angular atlas not proved",
        "warning": "This is not a Kerr stability proof certificate.",
    }
    if source_path is not None:
        metadata["angular_generic_overlap_path"] = source_path
    return {
        "accepted": True,
        "metadata": metadata,
        "symbolic_identities": symbolic_identities(),
        "polar_overlap_policy": polar_policy,
        "equatorial_overlap_policy": equatorial_policy,
    }


def resolved_path(package_path: Path, raw: Any, field: str) -> Path:
    require(isinstance(raw, str) and raw, f"metadata.{field} must be a nonempty path")
    path = Path(raw)
    if not path.is_absolute():
        path = package_path.parent / path
    return path


def load_seed_for_generic(generic_path: Path, generic_package: dict[str, Any]) -> dict[str, Any]:
    metadata = generic_package.get("metadata")
    require(isinstance(metadata, dict), "angular generic-overlap metadata must be an object")
    seed_path = resolved_path(generic_path, metadata.get("angular_seed_tube_path"), "angular_seed_tube_path")
    return load_json(seed_path)


def verify_package(path: Path) -> dict[str, Any]:
    stored = load_json(path)
    metadata = stored.get("metadata")
    require(isinstance(metadata, dict), "angular overlap-policy metadata must be an object")
    source_path = str(metadata.get("angular_generic_overlap_path", ""))
    generic_path = resolved_path(path, source_path, "angular_generic_overlap_path")
    generic_package = load_json(generic_path)
    seed_package = load_seed_for_generic(generic_path, generic_package)
    computed = compute_certificate(generic_package, seed_package, source_path=source_path)
    require(stored == computed, "angular overlap-policy certificate is stale or incorrect")
    return summary(computed)


def summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "polar_policy_count": len(certificate["polar_overlap_policy"]),
        "equatorial_policy_count": len(certificate["equatorial_overlap_policy"]),
        "polar_overlap_policy": [policy["name"] for policy in certificate["polar_overlap_policy"]],
        "equatorial_overlap_policy": [
            policy["name"] for policy in certificate["equatorial_overlap_policy"]
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
        generic_package = load_json(args.input)
        seed_package = load_seed_for_generic(args.input, generic_package)
        certificate = compute_certificate(
            generic_package,
            seed_package,
            source_path=relative_source_path(args.output, args.input),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter angular overlap-policy rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
