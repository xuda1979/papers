#!/usr/bin/env python3
"""Verify a thicker local angular seed tube from the spin-half bridge data."""

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
        "polar_seed_tube_regularization": sp.factor(
            (v * polar_ordinary).subs(lam**2, v * mu2) - v * polar_regular
        ),
        "equatorial_seed_tube_factorization": sp.factor(
            equatorial_ordinary.subs(u, eta * zeta) - eta * equatorial_factor
        ),
    }
    failed = {name: str(value) for name, value in identities.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"angular seed-tube symbolic identity failed: {failed}")
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


def bridges_by_seed(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require_package(data, "degenerate seed bridge")
    bridges = data.get("bridges")
    require(isinstance(bridges, list) and bridges, "bridge package must contain bridges")
    result: dict[str, dict[str, Any]] = {}
    for index, bridge in enumerate(bridges):
        require(isinstance(bridge, dict), f"bridges[{index}] must be an object")
        seed = bridge.get("seed_family")
        require(isinstance(seed, str) and seed, f"bridges[{index}].seed_family must be nonempty")
        require(seed not in result, f"duplicate seed bridge {seed}")
        result[seed] = bridge
    return result


def verify_polar_seed_tube(bridge: dict[str, Any], prior_patch: dict[str, Any]) -> dict[str, Any]:
    name = "polar_two_sided_parameter_seed_tube"
    a_squared = point(Fraction(1, 4))
    bridge_eta = required_interval(bridge, ["generic_bridge_values", "eta_interval"], name)
    bridge_lambda = required_interval(bridge, ["generic_bridge_values", "lambda_interval"], name)
    prior_target = required_interval(prior_patch, ["target_sin_squared_theta_interval"], name)
    prior_ordinary = required_interval(prior_patch, ["ordinary_sin_squared_theta_interval"], name)

    lambda_tube = Interval(Fraction(-1, 6000), Fraction(1, 6000))
    v_corridor = Interval(Fraction(0), Fraction(1, 16))
    mu_squared = Interval(Fraction(0), Fraction(1, 2250000))
    ordinary_v = Interval(Fraction(1, 64), Fraction(1))
    require(interval_contains(bridge_lambda, lambda_tube), f"{name} lambda tube is not inside bridge data")
    require(interval_contains(prior_target, Interval(Fraction(0), Fraction(1))), f"{name} prior target changed")
    require(interval_contains(prior_ordinary, ordinary_v), f"{name} prior ordinary patch no longer contains annulus")

    lambda_squared_from_map = v_corridor * mu_squared
    gap = a_squared - mu_squared
    regular_theta = bridge_eta + one_minus(v_corridor) * gap
    ordinary_theta = bridge_eta + a_squared * one_minus(ordinary_v) - (square(bridge_lambda) / ordinary_v) * one_minus(ordinary_v)

    checks = {
        "two_sided_lambda_tube_inside_radial_bridge": interval_contains(bridge_lambda, lambda_tube),
        "lambda_square_fits_regular_coordinate_map": square(lambda_tube).hi <= lambda_squared_from_map.hi,
        "polar_regularized_gap_positive": interval_strictly_positive(gap),
        "polar_regularized_Theta_positive_on_thick_tube": interval_strictly_positive(regular_theta),
        "ordinary_annulus_denominator_positive": interval_strictly_positive(ordinary_v),
        "ordinary_annulus_Theta_positive_for_full_bridge_lambda": interval_strictly_positive(ordinary_theta),
        "prior_patch_covers_full_sin_squared_theta_interval": interval_contains(prior_target, Interval(Fraction(0), Fraction(1))),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} checks failed: {failed}")

    return {
        "name": name,
        "seed_family": bridge["seed_family"],
        "prior_patch": prior_patch["name"],
        "tube_type": "polar_two_sided_lambda_thickening",
        "lambda_tube_interval": interval_json(lambda_tube),
        "bridge_lambda_interval": interval_json(bridge_lambda),
        "bridge_eta_interval": interval_json(bridge_eta),
        "sin_squared_theta_corridor": interval_json(v_corridor),
        "mu_squared_interval": interval_json(mu_squared),
        "lambda_squared_from_coordinate_map": interval_json(lambda_squared_from_map),
        "regularized_gap_interval": interval_json(gap),
        "regularized_Theta_interval": interval_json(regular_theta),
        "ordinary_sin_squared_theta_annulus": interval_json(ordinary_v),
        "ordinary_Theta_full_bridge_lambda_interval": interval_json(ordinary_theta),
        "checked_conditions": sorted(checks),
    }


def verify_equatorial_seed_tube(
    bridge: dict[str, Any],
    prior_patch: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    a_squared = point(Fraction(1, 4))
    bridge_eta = required_interval(bridge, ["generic_bridge_values", "eta_interval"], name)
    bridge_lambda = required_interval(bridge, ["generic_bridge_values", "lambda_interval"], name)
    prior_zeta = required_interval(prior_patch, ["extended_zeta_interval"], name)

    eta_tube = Interval(Fraction(0), Fraction(1, 1024))
    ordinary_eta = Interval(Fraction(1, 8192), Fraction(1, 1024))
    zeta = Interval(Fraction(0), Fraction(1, 48))
    require(interval_contains(bridge_eta, eta_tube), f"{name} eta tube is not inside bridge data")
    require(interval_contains(zeta, prior_zeta), f"{name} zeta tube does not contain prior patch")

    u = eta_tube * zeta
    denominator = one_minus(u)
    lambda_squared = square(bridge_lambda)
    factor = (denominator * (point(1) + a_squared * zeta) - lambda_squared * zeta) / denominator
    theta = eta_tube * factor

    ordinary_u = ordinary_eta * zeta
    ordinary_denominator = one_minus(ordinary_u)
    ordinary_factor = (
        ordinary_denominator * (point(1) + a_squared * zeta) - lambda_squared * zeta
    ) / ordinary_denominator
    ordinary_theta = ordinary_eta * ordinary_factor

    checks = {
        "eta_tube_inside_radial_bridge": interval_contains(bridge_eta, eta_tube),
        "extended_zeta_contains_prior_patch": interval_contains(zeta, prior_zeta),
        "one_minus_cos_squared_theta_positive_on_eta_tube": interval_strictly_positive(denominator),
        "equatorial_factor_positive_on_eta_tube": interval_strictly_positive(factor),
        "Theta_nonnegative_on_eta_tube": interval_nonnegative(theta),
        "ordinary_eta_overlap_positive": interval_strictly_positive(ordinary_eta),
        "ordinary_overlap_denominator_positive": interval_strictly_positive(ordinary_denominator),
        "ordinary_overlap_factor_positive": interval_strictly_positive(ordinary_factor),
        "ordinary_overlap_Theta_positive": interval_strictly_positive(ordinary_theta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} checks failed: {failed}")

    return {
        "name": name,
        "seed_family": bridge["seed_family"],
        "prior_patch": prior_patch["name"],
        "tube_type": "equatorial_eta_zeta_thickening",
        "eta_tube_interval": interval_json(eta_tube),
        "ordinary_eta_overlap_interval": interval_json(ordinary_eta),
        "zeta_tube_interval": interval_json(zeta),
        "prior_zeta_interval": interval_json(prior_zeta),
        "bridge_eta_interval": interval_json(bridge_eta),
        "bridge_lambda_interval": interval_json(bridge_lambda),
        "bridge_lambda_squared_interval": interval_json(lambda_squared),
        "cos_squared_theta_interval": interval_json(u),
        "factor_interval": interval_json(factor),
        "Theta_interval": interval_json(theta),
        "ordinary_cos_squared_theta_interval": interval_json(ordinary_u),
        "ordinary_factor_interval": interval_json(ordinary_factor),
        "ordinary_Theta_interval": interval_json(ordinary_theta),
        "checked_conditions": sorted(checks),
    }


def compute_certificate(
    patch_package: dict[str, Any],
    bridge_package: dict[str, Any],
    source_path: str | None = None,
) -> dict[str, Any]:
    require_package(patch_package, "angular patch coverage")
    require_package(bridge_package, "degenerate seed bridge")
    patches = records_by_name(patch_package, "polar_patch_cover")
    equatorial_patches = records_by_name(patch_package, "equatorial_patch_cover")
    bridges = bridges_by_seed(bridge_package)

    polar_tubes = [
        verify_polar_seed_tube(
            require_record(bridges, "polar_L_equals_zero", "bridges"),
            require_record(patches, "polar_positive_L_full_v_patch", "polar_patch_cover"),
        )
    ]
    equatorial_tubes = [
        verify_equatorial_seed_tube(
            require_record(bridges, "equatorial_Q_equals_zero_prograde", "bridges"),
            require_record(equatorial_patches, "equatorial_prograde_extended_zeta_patch", "equatorial_patch_cover"),
            "equatorial_prograde_eta_zeta_seed_tube",
        ),
        verify_equatorial_seed_tube(
            require_record(bridges, "equatorial_Q_equals_zero_retrograde", "bridges"),
            require_record(equatorial_patches, "equatorial_retrograde_extended_zeta_patch", "equatorial_patch_cover"),
            "equatorial_retrograde_eta_zeta_seed_tube",
        ),
    ]
    metadata: dict[str, Any] = {
        "name": "spin-half angular seed-tube thickening certificate",
        "spin": "1/2",
        "mass_normalization": "M=1",
        "status": "local parameter-thickened seed tubes only; full Kerr angular atlas not proved",
        "warning": "This is not a Kerr stability proof certificate.",
    }
    if source_path is not None:
        metadata["angular_patch_coverage_path"] = source_path
    return {
        "accepted": True,
        "metadata": metadata,
        "symbolic_identities": symbolic_identities(),
        "polar_seed_tubes": polar_tubes,
        "equatorial_seed_tubes": equatorial_tubes,
    }


def resolved_path(package_path: Path, raw: Any, field: str) -> Path:
    require(isinstance(raw, str) and raw, f"metadata.{field} must be a nonempty path")
    path = Path(raw)
    if not path.is_absolute():
        path = package_path.parent / path
    return path


def load_bridge_for_patch(patch_path: Path, patch_package: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    patch_metadata = patch_package.get("metadata")
    require(isinstance(patch_metadata, dict), "angular patch coverage metadata must be an object")
    overlap_path = resolved_path(patch_path, patch_metadata.get("angular_overlap_path"), "angular_overlap_path")
    overlap = load_json(overlap_path)
    overlap_metadata = overlap.get("metadata")
    require(isinstance(overlap_metadata, dict), "angular overlap metadata must be an object")
    corridor_path = resolved_path(overlap_path, overlap_metadata.get("angular_corridor_path"), "angular_corridor_path")
    corridor = load_json(corridor_path)
    corridor_metadata = corridor.get("metadata")
    require(isinstance(corridor_metadata, dict), "angular corridor metadata must be an object")
    bridge_path = resolved_path(corridor_path, corridor_metadata.get("degenerate_seed_bridge_path"), "degenerate_seed_bridge_path")
    return bridge_path, load_json(bridge_path)


def verify_package(path: Path) -> dict[str, Any]:
    stored = load_json(path)
    metadata = stored.get("metadata")
    require(isinstance(metadata, dict), "angular seed-tube metadata must be an object")
    source_path = str(metadata.get("angular_patch_coverage_path", ""))
    patch_path = resolved_path(path, source_path, "angular_patch_coverage_path")
    patch_package = load_json(patch_path)
    _, bridge_package = load_bridge_for_patch(patch_path, patch_package)
    computed = compute_certificate(patch_package, bridge_package, source_path=source_path)
    require(stored == computed, "angular seed-tube certificate is stale or incorrect")
    return summary(computed)


def summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "polar_seed_tube_count": len(certificate["polar_seed_tubes"]),
        "equatorial_seed_tube_count": len(certificate["equatorial_seed_tubes"]),
        "polar_seed_tubes": [tube["name"] for tube in certificate["polar_seed_tubes"]],
        "equatorial_seed_tubes": [tube["name"] for tube in certificate["equatorial_seed_tubes"]],
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
        patch_package = load_json(args.input)
        _, bridge_package = load_bridge_for_patch(args.input, patch_package)
        certificate = compute_certificate(
            patch_package,
            bridge_package,
            source_path=relative_source_path(args.output, args.input),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter angular seed-tube rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
