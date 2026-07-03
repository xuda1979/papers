#!/usr/bin/env python3
"""Verify angular overlap strips between corridor and ordinary Carter charts."""

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


def strictly_positive(interval: Interval) -> bool:
    return interval.lo > 0


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
    ordinary_v = eta + a**2 * (1 - v) - lam**2 * (1 - v) / v
    polar_regular = eta + (1 - v) * (a**2 - mu2)
    ordinary_u = eta + a**2 * u - lam**2 * u / (1 - u)
    equatorial_regular = eta * ((1 - eta * zeta) * (1 + a**2 * zeta) - lam**2 * zeta) / (1 - eta * zeta)
    identities = {
        "polar_overlap_identity": sp.factor(ordinary_v.subs(lam**2, v * mu2) - polar_regular),
        "equatorial_overlap_identity": sp.factor(ordinary_u.subs(u, eta * zeta) - equatorial_regular),
    }
    failed = {key: str(value) for key, value in identities.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"angular overlap identity failed: {failed}")
    return {key: "0" for key in sorted(identities)}


def require_corridor_package(data: dict[str, Any]) -> None:
    require(data.get("accepted") is True, "angular corridor package is not accepted")
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "angular corridor metadata must be an object")
    require(metadata.get("spin") == "1/2", "angular overlap currently targets the spin-half corridors")


def corridors_by_name(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    require_corridor_package(data)
    result = {}
    for section in ["polar_corridors", "equatorial_corridors"]:
        corridors = data.get(section)
        require(isinstance(corridors, list), f"{section} must be a list")
        for index, corridor in enumerate(corridors):
            require(isinstance(corridor, dict), f"{section}[{index}] must be an object")
            name = corridor.get("name")
            require(isinstance(name, str) and name, f"{section}[{index}].name must be a nonempty string")
            require(name not in result, f"duplicate corridor name {name}")
            result[name] = corridor
    return result


def verify_polar_overlap(corridor: dict[str, Any], name: str) -> dict[str, Any]:
    a_squared = point(Fraction(1, 4))
    corridor_v = required_interval(corridor, ["sin_squared_theta_interval"], name)
    corridor_mu_squared = required_interval(corridor, ["mu_squared_interval"], name)
    corridor_eta = required_interval(corridor, ["eta_interval"], name)
    corridor_theta = required_interval(corridor, ["regularized_Theta_interval"], name)
    corridor_lambda = required_interval(corridor, ["lambda_interval"], name)

    v_strip = Interval(Fraction(1, 64), Fraction(1, 16))
    mu_squared = Interval(Fraction(0), Fraction(1, 4194304))
    eta = corridor_eta
    require(interval_contains(corridor_v, v_strip), f"{name} v strip is not contained in corridor")
    require(interval_contains(corridor_mu_squared, mu_squared), f"{name} mu strip is not contained in corridor")

    lambda_squared = v_strip * mu_squared
    lambda_abs = Interval(Fraction(0), Fraction(1, 8192))
    require(lambda_squared.hi <= square(lambda_abs).hi, f"{name} induced lambda is wider than corridor")
    require(
        interval_contains(Interval(Fraction(0), max(abs(corridor_lambda.lo), abs(corridor_lambda.hi))), lambda_abs),
        f"{name} lambda strip is not contained in one-sided corridor size",
    )
    one_minus_v = one_minus(v_strip)
    denominator = v_strip
    ordinary_theta = eta + a_squared * one_minus_v - (lambda_squared / denominator) * one_minus_v
    regular_theta = eta + one_minus_v * (a_squared - mu_squared)
    checks = {
        "sin_squared_theta_away_from_zero": strictly_positive(denominator),
        "ordinary_polar_Theta_positive": strictly_positive(ordinary_theta),
        "regularized_polar_Theta_positive": strictly_positive(regular_theta),
        "ordinary_Theta_inside_corridor_margin": interval_contains(corridor_theta, regular_theta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} polar overlap checks failed: {failed}")
    return {
        "name": name,
        "corridor": corridor["name"],
        "overlap_type": "polar_to_ordinary_angular",
        "sin_squared_theta_strip": interval_json(v_strip),
        "mu_squared_strip": interval_json(mu_squared),
        "lambda_squared_strip": interval_json(lambda_squared),
        "ordinary_Theta_interval": interval_json(ordinary_theta),
        "regularized_Theta_interval": interval_json(regular_theta),
        "checked_conditions": sorted(checks),
    }


def verify_equatorial_overlap(corridor: dict[str, Any], name: str) -> dict[str, Any]:
    a_squared = point(Fraction(1, 4))
    corridor_eta = required_interval(corridor, ["eta_thickness_interval"], name)
    corridor_zeta = required_interval(corridor, ["zeta_interval"], name)
    lam = required_interval(corridor, ["lambda_interval"], name)
    corridor_factor = required_interval(corridor, ["factor_interval"], name)

    eta = Interval(Fraction(1, 8192), Fraction(1, 4096))
    zeta = Interval(Fraction(0), Fraction(1, 256))
    require(interval_contains(corridor_eta, eta), f"{name} eta strip is not contained in corridor")
    require(interval_contains(corridor_zeta, zeta), f"{name} zeta strip is not contained in corridor")
    u = eta * zeta
    denominator = one_minus(u)
    lam_squared = square(lam)
    ordinary_theta = eta + a_squared * u - (lam_squared * u / denominator)
    factor = (denominator * (point(1) + a_squared * zeta) - lam_squared * zeta) / denominator
    regular_theta = eta * factor
    checks = {
        "eta_away_from_zero_on_overlap": strictly_positive(eta),
        "one_minus_cos_squared_theta_positive": strictly_positive(denominator),
        "ordinary_equatorial_Theta_positive": strictly_positive(ordinary_theta),
        "regularized_equatorial_factor_positive": strictly_positive(factor),
        "factor_inside_corridor_margin": interval_contains(corridor_factor, factor),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} equatorial overlap checks failed: {failed}")
    return {
        "name": name,
        "corridor": corridor["name"],
        "overlap_type": "equatorial_to_ordinary_angular",
        "eta_strip": interval_json(eta),
        "zeta_strip": interval_json(zeta),
        "cos_squared_theta_strip": interval_json(u),
        "ordinary_Theta_interval": interval_json(ordinary_theta),
        "regularized_factor_interval": interval_json(factor),
        "regularized_Theta_interval": interval_json(regular_theta),
        "checked_conditions": sorted(checks),
    }


def compute_certificate(corridor_package: dict[str, Any], source_path: str | None = None) -> dict[str, Any]:
    corridors = corridors_by_name(corridor_package)
    required = [
        "polar_positive_L_corridor",
        "polar_tiny_negative_L_corridor",
        "equatorial_prograde_eta_corridor",
        "equatorial_retrograde_eta_corridor",
    ]
    missing = [name for name in required if name not in corridors]
    require(not missing, f"angular corridor package missing {missing}")
    polar = [
        verify_polar_overlap(corridors["polar_positive_L_corridor"], "polar_positive_L_outer_overlap"),
        verify_polar_overlap(corridors["polar_tiny_negative_L_corridor"], "polar_negative_L_outer_overlap"),
    ]
    equatorial = [
        verify_equatorial_overlap(corridors["equatorial_prograde_eta_corridor"], "equatorial_prograde_eta_overlap"),
        verify_equatorial_overlap(corridors["equatorial_retrograde_eta_corridor"], "equatorial_retrograde_eta_overlap"),
    ]
    metadata: dict[str, Any] = {
        "name": "spin-half angular overlap strip certificate",
        "spin": "1/2",
        "mass_normalization": "M=1",
        "status": "local overlap strips only; full angular atlas not proved",
        "warning": "This is not a Kerr stability proof certificate.",
    }
    if source_path is not None:
        metadata["angular_corridor_path"] = source_path
    return {
        "accepted": True,
        "metadata": metadata,
        "symbolic_identities": symbolic_identities(),
        "polar_overlap_strips": polar,
        "equatorial_overlap_strips": equatorial,
    }


def resolved_source_path(package_path: Path, raw: Any) -> Path:
    require(isinstance(raw, str) and raw, "metadata.angular_corridor_path must be a nonempty path")
    path = Path(raw)
    if not path.is_absolute():
        path = package_path.parent / path
    return path


def verify_package(path: Path) -> dict[str, Any]:
    stored = load_json(path)
    metadata = stored.get("metadata")
    require(isinstance(metadata, dict), "angular overlap metadata must be an object")
    source_path = str(metadata.get("angular_corridor_path", ""))
    corridor_package = load_json(resolved_source_path(path, source_path))
    computed = compute_certificate(corridor_package, source_path=source_path)
    require(stored == computed, "angular overlap certificate is stale or incorrect")
    return summary(computed)


def summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "polar_overlap_count": len(certificate["polar_overlap_strips"]),
        "equatorial_overlap_count": len(certificate["equatorial_overlap_strips"]),
        "polar_overlap_strips": [strip["name"] for strip in certificate["polar_overlap_strips"]],
        "equatorial_overlap_strips": [strip["name"] for strip in certificate["equatorial_overlap_strips"]],
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
        corridor_package = load_json(args.input)
        certificate = compute_certificate(
            corridor_package,
            source_path=relative_source_path(args.output, args.input),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter angular overlap rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
