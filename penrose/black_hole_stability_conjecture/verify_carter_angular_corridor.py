#!/usr/bin/env python3
"""Verify local angular Carter corridors near degenerate spin-half seeds.

The radial bridge certificate anchors the polar and equatorial seed radii inside
generic radial charts.  This checker adds a complementary local angular check:
it rewrites the null Carter angular potential in variables that stay regular at
the polar and equatorial degeneracies, then verifies rational corridor boxes.
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

from generate_carter_atlas import Interval, parse_fraction, parse_interval, point


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def interval_contains(container: Interval, contained: Interval) -> bool:
    return container.lo <= contained.lo and contained.hi <= container.hi


def interval_intersects_zero(interval: Interval) -> bool:
    return interval.lo <= 0 <= interval.hi


def interval_square(interval: Interval) -> Interval:
    return interval.square()


def interval_strictly_positive(interval: Interval) -> bool:
    return interval.lo > 0


def interval_nonnegative(interval: Interval) -> bool:
    return interval.lo >= 0


def point_interval(value: Fraction | int) -> Interval:
    return Interval(Fraction(value), Fraction(value))


def one_minus(interval: Interval) -> Interval:
    return point_interval(1) - interval


def interval_json(interval: Interval) -> str:
    return interval.to_json()


def require_bridge_package(data: dict[str, Any]) -> None:
    require(data.get("accepted") is True, "bridge certificate is not accepted")
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "bridge certificate metadata must be an object")
    require(metadata.get("spin") == "1/2", "angular corridor currently targets the spin-half bridge")
    bridges = data.get("bridges")
    require(isinstance(bridges, list) and bridges, "bridge certificate must contain bridges")


def bridge_by_seed(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require_bridge_package(data)
    result = {}
    for index, bridge in enumerate(data["bridges"]):
        require(isinstance(bridge, dict), f"bridges[{index}] must be an object")
        seed = bridge.get("seed_family")
        require(isinstance(seed, str) and seed, f"bridges[{index}].seed_family must be a nonempty string")
        require(seed not in result, f"duplicate seed bridge {seed}")
        result[seed] = bridge
    return result


def required_interval(data: dict[str, Any], keys: list[str], name: str) -> Interval:
    current: Any = data
    for key in keys:
        require(isinstance(current, dict) and key in current, f"{name} missing {'.'.join(keys)}")
        current = current[key]
    return parse_interval(current)


def symbolic_identities() -> dict[str, str]:
    eta, a, lam, u, v, zeta, mu2 = sp.symbols("eta a lambda u v zeta mu2")
    theta_u = eta + a**2 * u - lam**2 * u / (1 - u)
    equatorial_numerator = (1 - eta * zeta) * (1 + a**2 * zeta) - lam**2 * zeta
    polar_regular = eta + (1 - v) * (a**2 - mu2)
    identities = {
        "equatorial_blowup": sp.factor(
            theta_u.subs(u, eta * zeta) - eta * equatorial_numerator / (1 - eta * zeta)
        ),
        "polar_regularization": sp.factor(
            (v * (eta + a**2 * (1 - v) - lam**2 * (1 - v) / v)).subs(lam**2, v * mu2)
            - v * polar_regular
        ),
    }
    failed = {name: str(value) for name, value in identities.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"angular symbolic identity failed: {failed}")
    return {name: "0" for name in sorted(identities)}


def generic_interval(bridge: dict[str, Any], key: str, name: str) -> Interval:
    return required_interval(bridge, ["generic_bridge_values", key], name)


def regular_interval(bridge: dict[str, Any], key: str, name: str) -> Interval:
    return required_interval(bridge, ["regular_seed_values", key], name)


def verify_polar_corridor(
    bridge: dict[str, Any],
    name: str,
    lambda_interval: Interval,
    v_interval: Interval,
    mu_squared_interval: Interval,
) -> dict[str, Any]:
    a_squared = point_interval(Fraction(1, 4))
    eta = regular_interval(bridge, "eta_interval", name)
    generic_lambda = generic_interval(bridge, "lambda_interval", name)
    generic_eta = generic_interval(bridge, "eta_interval", name)
    require(interval_contains(generic_lambda, lambda_interval), f"{name} lambda corridor not contained in bridge")
    require(interval_contains(generic_eta, eta), f"{name} eta seed interval not contained in bridge")

    lambda_squared_from_map = v_interval * mu_squared_interval
    require(
        interval_square(lambda_interval).hi <= lambda_squared_from_map.hi,
        f"{name} lambda interval is wider than lambda^2=v*mu^2 corridor",
    )

    one_minus_v = one_minus(v_interval)
    angular_gap = a_squared - mu_squared_interval
    theta_regular = eta + one_minus_v * angular_gap

    checks = {
        "one_minus_sin_squared_theta_positive": interval_strictly_positive(one_minus_v),
        "polar_angular_gap_positive": interval_strictly_positive(angular_gap),
        "regularized_Theta_positive": interval_strictly_positive(theta_regular),
        "lambda_corridor_inside_radial_bridge": interval_contains(generic_lambda, lambda_interval),
        "eta_seed_inside_radial_bridge": interval_contains(generic_eta, eta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} angular corridor checks failed: {failed}")

    return {
        "name": name,
        "seed_family": bridge["seed_family"],
        "chart_type": "polar_regularized",
        "coordinate_relation": "lambda^2 = sin^2(theta)*mu^2",
        "sin_squared_theta_interval": interval_json(v_interval),
        "mu_squared_interval": interval_json(mu_squared_interval),
        "lambda_interval": interval_json(lambda_interval),
        "eta_interval": interval_json(eta),
        "angular_gap_interval": interval_json(angular_gap),
        "regularized_Theta_interval": interval_json(theta_regular),
        "lambda_squared_from_coordinate_map": interval_json(lambda_squared_from_map),
        "checked_conditions": sorted(checks),
    }


def verify_equatorial_corridor(bridge: dict[str, Any], name: str) -> dict[str, Any]:
    a_squared = point_interval(Fraction(1, 4))
    eta_thickness = Interval(Fraction(0), Fraction(1, 4096))
    zeta = Interval(Fraction(0), Fraction(1, 256))
    generic_eta = generic_interval(bridge, "eta_interval", name)
    lam = regular_interval(bridge, "lambda_interval", name)
    generic_lambda = generic_interval(bridge, "lambda_interval", name)
    require(interval_contains(generic_eta, eta_thickness), f"{name} eta corridor not contained in bridge")
    require(interval_contains(generic_lambda, lam), f"{name} lambda seed interval not contained in bridge")

    u = eta_thickness * zeta
    denominator = one_minus(u)
    lam_squared = interval_square(lam)
    numerator = denominator * (point_interval(1) + a_squared * zeta) - lam_squared * zeta
    factor = numerator / denominator
    theta_interval = eta_thickness * factor

    checks = {
        "cos_squared_theta_interval_small": interval_contains(Interval(Fraction(0), Fraction(1)), u),
        "one_minus_cos_squared_theta_positive": interval_strictly_positive(denominator),
        "equatorial_blowup_factor_positive": interval_strictly_positive(factor),
        "Theta_nonnegative_on_corridor": interval_nonnegative(theta_interval),
        "eta_corridor_inside_radial_bridge": interval_contains(generic_eta, eta_thickness),
        "lambda_seed_inside_radial_bridge": interval_contains(generic_lambda, lam),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} angular corridor checks failed: {failed}")

    return {
        "name": name,
        "seed_family": bridge["seed_family"],
        "chart_type": "equatorial_blowup",
        "coordinate_relation": "cos^2(theta)=eta*zeta",
        "eta_thickness_interval": interval_json(eta_thickness),
        "zeta_interval": interval_json(zeta),
        "cos_squared_theta_interval": interval_json(u),
        "lambda_interval": interval_json(lam),
        "lambda_squared_interval": interval_json(lam_squared),
        "factor_interval": interval_json(factor),
        "Theta_interval": interval_json(theta_interval),
        "checked_conditions": sorted(checks),
    }


def compute_certificate(bridge_package: dict[str, Any], source_path: str | None = None) -> dict[str, Any]:
    bridges = bridge_by_seed(bridge_package)
    required = [
        "polar_L_equals_zero",
        "equatorial_Q_equals_zero_prograde",
        "equatorial_Q_equals_zero_retrograde",
    ]
    missing = [name for name in required if name not in bridges]
    require(not missing, f"bridge package missing seed families {missing}")
    identities = symbolic_identities()
    polar = bridges["polar_L_equals_zero"]
    polar_corridors = [
        verify_polar_corridor(
            polar,
            "polar_positive_L_corridor",
            Interval(Fraction(0), Fraction(1, 8192)),
            Interval(Fraction(0), Fraction(1, 16)),
            Interval(Fraction(0), Fraction(1, 4194304)),
        ),
        verify_polar_corridor(
            polar,
            "polar_tiny_negative_L_corridor",
            Interval(Fraction(-1, 8192), Fraction(0)),
            Interval(Fraction(0), Fraction(1, 16)),
            Interval(Fraction(0), Fraction(1, 4194304)),
        ),
    ]
    equatorial_corridors = [
        verify_equatorial_corridor(
            bridges["equatorial_Q_equals_zero_prograde"],
            "equatorial_prograde_eta_corridor",
        ),
        verify_equatorial_corridor(
            bridges["equatorial_Q_equals_zero_retrograde"],
            "equatorial_retrograde_eta_corridor",
        ),
    ]
    metadata: dict[str, Any] = {
        "name": "spin-half angular corridor certificate",
        "spin": "1/2",
        "mass_normalization": "M=1",
        "status": "local angular corridors only; full angular overlap atlas not proved",
        "warning": "This is not a Kerr stability proof certificate.",
    }
    if source_path is not None:
        metadata["degenerate_seed_bridge_path"] = source_path
    return {
        "accepted": True,
        "metadata": metadata,
        "symbolic_identities": identities,
        "polar_corridors": polar_corridors,
        "equatorial_corridors": equatorial_corridors,
    }


def resolved_bridge_path(package_path: Path, raw: Any) -> Path:
    require(isinstance(raw, str) and raw, "metadata.degenerate_seed_bridge_path must be a nonempty path")
    path = Path(raw)
    if not path.is_absolute():
        path = package_path.parent / path
    return path


def verify_package(path: Path) -> dict[str, Any]:
    stored = load_json(path)
    metadata = stored.get("metadata")
    require(isinstance(metadata, dict), "angular corridor metadata must be an object")
    bridge_path = str(metadata.get("degenerate_seed_bridge_path", ""))
    bridge_package = load_json(resolved_bridge_path(path, bridge_path))
    computed = compute_certificate(bridge_package, source_path=bridge_path)
    require(stored == computed, "angular corridor certificate is stale or incorrect")
    return summary(computed)


def summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "polar_corridor_count": len(certificate["polar_corridors"]),
        "equatorial_corridor_count": len(certificate["equatorial_corridors"]),
        "polar_corridors": [corridor["name"] for corridor in certificate["polar_corridors"]],
        "equatorial_corridors": [corridor["name"] for corridor in certificate["equatorial_corridors"]],
    }


def relative_source_path(output: Path, bridge: Path) -> str:
    return os.path.relpath(bridge.resolve(), output.parent.resolve())


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        if args.output is None:
            print(json.dumps(verify_package(args.input), indent=2, sort_keys=True))
            return 0
        bridge_package = load_json(args.input)
        certificate = compute_certificate(
            bridge_package,
            source_path=relative_source_path(args.output, args.input),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter angular corridor rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
