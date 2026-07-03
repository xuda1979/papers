#!/usr/bin/env python3
"""Verify local angular patch coverage beyond the first overlap strips."""

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
    eta, a, lam, u, v, zeta = sp.symbols("eta a lambda u v zeta")
    theta_u = eta + a**2 * u - lam**2 * u / (1 - u)
    theta_v = eta + a**2 * (1 - v) - lam**2 * (1 - v) / v
    equatorial_factor = ((1 - eta * zeta) * (1 + a**2 * zeta) - lam**2 * zeta) / (
        1 - eta * zeta
    )
    identities = {
        "ordinary_u_to_v": sp.factor(theta_u.subs(u, 1 - v) - theta_v),
        "equatorial_zeta_extension": sp.factor(theta_u.subs(u, eta * zeta) - eta * equatorial_factor),
    }
    failed = {name: str(value) for name, value in identities.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"angular patch symbolic identity failed: {failed}")
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


def verify_polar_patch(
    corridor: dict[str, Any],
    overlap: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    a_squared = point(Fraction(1, 4))
    corridor_v = required_interval(corridor, ["sin_squared_theta_interval"], name)
    eta = required_interval(corridor, ["eta_interval"], name)
    lam = required_interval(corridor, ["lambda_interval"], name)
    overlap_v = required_interval(overlap, ["sin_squared_theta_strip"], name)

    target_v = Interval(Fraction(0), Fraction(1))
    ordinary_v = Interval(Fraction(1, 64), Fraction(1))
    require(interval_contains(corridor_v, overlap_v), f"{name} overlap is not contained in corridor")
    require(interval_contains(ordinary_v, overlap_v), f"{name} overlap is not contained in ordinary annulus")
    require(covers_target([corridor_v, ordinary_v], target_v), f"{name} polar v coverage has a gap")

    denominator = ordinary_v
    lambda_squared = square(lam)
    ordinary_theta = eta + a_squared * one_minus(ordinary_v) - (lambda_squared / denominator) * one_minus(ordinary_v)

    checks = {
        "corridor_and_annulus_cover_full_v_interval": covers_target([corridor_v, ordinary_v], target_v),
        "overlap_strip_nonempty": overlap_v.lo < overlap_v.hi,
        "overlap_strip_inside_corridor": interval_contains(corridor_v, overlap_v),
        "overlap_strip_inside_ordinary_annulus": interval_contains(ordinary_v, overlap_v),
        "ordinary_sin_squared_theta_denominator_positive": interval_strictly_positive(denominator),
        "ordinary_annulus_Theta_positive": interval_strictly_positive(ordinary_theta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} polar patch checks failed: {failed}")

    return {
        "name": name,
        "corridor": corridor["name"],
        "overlap_strip": overlap["name"],
        "patch_type": "polar_corridor_plus_ordinary_annulus",
        "target_sin_squared_theta_interval": interval_json(target_v),
        "corridor_sin_squared_theta_interval": interval_json(corridor_v),
        "ordinary_sin_squared_theta_interval": interval_json(ordinary_v),
        "overlap_sin_squared_theta_interval": interval_json(overlap_v),
        "lambda_interval": interval_json(lam),
        "lambda_squared_interval": interval_json(lambda_squared),
        "eta_interval": interval_json(eta),
        "ordinary_Theta_interval": interval_json(ordinary_theta),
        "checked_conditions": sorted(checks),
    }


def verify_equatorial_patch(
    corridor: dict[str, Any],
    overlap: dict[str, Any],
    name: str,
) -> dict[str, Any]:
    a_squared = point(Fraction(1, 4))
    eta_thickness = required_interval(corridor, ["eta_thickness_interval"], name)
    old_zeta = required_interval(corridor, ["zeta_interval"], name)
    lam = required_interval(corridor, ["lambda_interval"], name)
    overlap_eta = required_interval(overlap, ["eta_strip"], name)
    overlap_zeta = required_interval(overlap, ["zeta_strip"], name)

    extended_zeta = Interval(Fraction(0), Fraction(1, 128))
    ordinary_eta = Interval(Fraction(1, 8192), Fraction(1, 4096))
    require(interval_contains(extended_zeta, old_zeta), f"{name} extended zeta does not contain corridor")
    require(interval_contains(extended_zeta, overlap_zeta), f"{name} extended zeta does not contain overlap")
    require(interval_contains(eta_thickness, ordinary_eta), f"{name} ordinary eta strip is not in corridor thickness")
    require(interval_contains(ordinary_eta, overlap_eta), f"{name} ordinary eta strip does not contain stored overlap")

    u = eta_thickness * extended_zeta
    denominator = one_minus(u)
    lambda_squared = square(lam)
    factor = (denominator * (point(1) + a_squared * extended_zeta) - lambda_squared * extended_zeta) / denominator
    theta = eta_thickness * factor

    ordinary_u = ordinary_eta * extended_zeta
    ordinary_denominator = one_minus(ordinary_u)
    ordinary_theta = ordinary_eta + a_squared * ordinary_u - (lambda_squared * ordinary_u / ordinary_denominator)
    ordinary_factor = (
        ordinary_denominator * (point(1) + a_squared * extended_zeta) - lambda_squared * extended_zeta
    ) / ordinary_denominator

    checks = {
        "extended_zeta_contains_old_corridor": interval_contains(extended_zeta, old_zeta),
        "extended_zeta_contains_stored_overlap": interval_contains(extended_zeta, overlap_zeta),
        "ordinary_eta_strip_inside_eta_thickness": interval_contains(eta_thickness, ordinary_eta),
        "ordinary_eta_strip_contains_stored_overlap": interval_contains(ordinary_eta, overlap_eta),
        "one_minus_cos_squared_theta_positive_on_extended_blowup": interval_strictly_positive(denominator),
        "extended_blowup_factor_positive": interval_strictly_positive(factor),
        "extended_blowup_Theta_nonnegative": interval_nonnegative(theta),
        "one_minus_cos_squared_theta_positive_on_ordinary_patch": interval_strictly_positive(ordinary_denominator),
        "ordinary_patch_factor_positive": interval_strictly_positive(ordinary_factor),
        "ordinary_patch_Theta_positive": interval_strictly_positive(ordinary_theta),
    }
    failed = [key for key, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"{name} equatorial patch checks failed: {failed}")

    return {
        "name": name,
        "corridor": corridor["name"],
        "overlap_strip": overlap["name"],
        "patch_type": "equatorial_extended_blowup_with_ordinary_eta_overlap",
        "eta_thickness_interval": interval_json(eta_thickness),
        "extended_zeta_interval": interval_json(extended_zeta),
        "stored_corridor_zeta_interval": interval_json(old_zeta),
        "ordinary_eta_overlap_interval": interval_json(ordinary_eta),
        "ordinary_zeta_overlap_interval": interval_json(extended_zeta),
        "lambda_interval": interval_json(lam),
        "lambda_squared_interval": interval_json(lambda_squared),
        "extended_cos_squared_theta_interval": interval_json(u),
        "extended_factor_interval": interval_json(factor),
        "extended_Theta_interval": interval_json(theta),
        "ordinary_cos_squared_theta_interval": interval_json(ordinary_u),
        "ordinary_factor_interval": interval_json(ordinary_factor),
        "ordinary_Theta_interval": interval_json(ordinary_theta),
        "checked_conditions": sorted(checks),
    }


def compute_certificate(
    overlap_package: dict[str, Any],
    corridor_package: dict[str, Any],
    source_path: str | None = None,
) -> dict[str, Any]:
    require_package(overlap_package, "angular overlap")
    require_package(corridor_package, "angular corridor")
    polar_corridors = records_by_name(corridor_package, "polar_corridors")
    equatorial_corridors = records_by_name(corridor_package, "equatorial_corridors")
    polar_overlaps = records_by_name(overlap_package, "polar_overlap_strips")
    equatorial_overlaps = records_by_name(overlap_package, "equatorial_overlap_strips")

    polar = [
        verify_polar_patch(
            require_record(polar_corridors, "polar_positive_L_corridor", "polar_corridors"),
            require_record(polar_overlaps, "polar_positive_L_outer_overlap", "polar_overlap_strips"),
            "polar_positive_L_full_v_patch",
        ),
        verify_polar_patch(
            require_record(polar_corridors, "polar_tiny_negative_L_corridor", "polar_corridors"),
            require_record(polar_overlaps, "polar_negative_L_outer_overlap", "polar_overlap_strips"),
            "polar_negative_L_full_v_patch",
        ),
    ]
    equatorial = [
        verify_equatorial_patch(
            require_record(equatorial_corridors, "equatorial_prograde_eta_corridor", "equatorial_corridors"),
            require_record(equatorial_overlaps, "equatorial_prograde_eta_overlap", "equatorial_overlap_strips"),
            "equatorial_prograde_extended_zeta_patch",
        ),
        verify_equatorial_patch(
            require_record(equatorial_corridors, "equatorial_retrograde_eta_corridor", "equatorial_corridors"),
            require_record(equatorial_overlaps, "equatorial_retrograde_eta_overlap", "equatorial_overlap_strips"),
            "equatorial_retrograde_extended_zeta_patch",
        ),
    ]
    metadata: dict[str, Any] = {
        "name": "spin-half angular patch coverage certificate",
        "spin": "1/2",
        "mass_normalization": "M=1",
        "status": "local patch coverage only; full Kerr angular atlas not proved",
        "warning": "This is not a Kerr stability proof certificate.",
    }
    if source_path is not None:
        metadata["angular_overlap_path"] = source_path
    return {
        "accepted": True,
        "metadata": metadata,
        "symbolic_identities": symbolic_identities(),
        "polar_patch_cover": polar,
        "equatorial_patch_cover": equatorial,
    }


def resolved_path(package_path: Path, raw: Any, field: str) -> Path:
    require(isinstance(raw, str) and raw, f"metadata.{field} must be a nonempty path")
    path = Path(raw)
    if not path.is_absolute():
        path = package_path.parent / path
    return path


def load_corridor_for_overlap(overlap_path: Path, overlap_package: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    metadata = overlap_package.get("metadata")
    require(isinstance(metadata, dict), "angular overlap metadata must be an object")
    corridor_path = resolved_path(overlap_path, metadata.get("angular_corridor_path"), "angular_corridor_path")
    return corridor_path, load_json(corridor_path)


def verify_package(path: Path) -> dict[str, Any]:
    stored = load_json(path)
    metadata = stored.get("metadata")
    require(isinstance(metadata, dict), "angular patch coverage metadata must be an object")
    source_path = str(metadata.get("angular_overlap_path", ""))
    overlap_path = resolved_path(path, source_path, "angular_overlap_path")
    overlap_package = load_json(overlap_path)
    _, corridor_package = load_corridor_for_overlap(overlap_path, overlap_package)
    computed = compute_certificate(overlap_package, corridor_package, source_path=source_path)
    require(stored == computed, "angular patch coverage certificate is stale or incorrect")
    return summary(computed)


def summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "polar_patch_count": len(certificate["polar_patch_cover"]),
        "equatorial_patch_count": len(certificate["equatorial_patch_cover"]),
        "polar_patch_cover": [patch["name"] for patch in certificate["polar_patch_cover"]],
        "equatorial_patch_cover": [patch["name"] for patch in certificate["equatorial_patch_cover"]],
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
        overlap_package = load_json(args.input)
        _, corridor_package = load_corridor_for_overlap(args.input, overlap_package)
        certificate = compute_certificate(
            overlap_package,
            corridor_package,
            source_path=relative_source_path(args.output, args.input),
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter angular patch coverage rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
