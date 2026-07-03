#!/usr/bin/env python3
"""Verify radial bridge data from degenerate Carter seeds to the generic atlas.

This checker does not prove the missing polar or equatorial angular-neighborhood
charts.  It proves a narrower compatibility statement: the regular polar and
equatorial seed intervals lie inside already generated nondegenerate radial
charts, and the generic spherical-orbit interval data on each seed interval
contain the regular seed data.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from generate_carter_atlas import (
    Interval,
    denominator_certificates,
    eta_interval,
    horizon_radius_interval,
    lambda_interval,
    omega_h_interval,
    parse_fraction,
    parse_interval,
    point,
    radial_hessian_interval,
)


INTERVAL_FIELDS = [
    "lambda_interval",
    "eta_interval",
    "radial_hessian_interval",
    "omega_h_interval",
    "superradiant_factor_interval",
    "trapped_scale_interval",
]

DENOMINATOR_FIELDS = [
    "a",
    "r_minus_M",
    "(r_minus_M)^2",
    "Delta",
    "r_plus",
]


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


def parse_interval_pair(raw: Any, name: str) -> Interval:
    try:
        return parse_interval(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an interval") from exc


def required_interval(data: dict[str, Any], keys: list[str], name: str) -> Interval:
    current: Any = data
    for key in keys:
        require(isinstance(current, dict) and key in current, f"{name} missing {'.'.join(keys)}")
        current = current[key]
    return parse_interval_pair(current, name)


def point_interval(value: Fraction | int) -> Interval:
    return Interval(Fraction(value), Fraction(value))


def interval_json(interval: Interval) -> str:
    return interval.to_json()


def denominator_interval(certificate: dict[str, Any]) -> Interval:
    interval = certificate.get("interval")
    require(isinstance(interval, str), "denominator certificate missing interval")
    return parse_interval(interval)


def require_denominator_contains(
    chart: dict[str, Any],
    chart_name: str,
    field: str,
    bridge_certificate: dict[str, Any],
) -> None:
    denominators = chart.get("denominator_certificates")
    require(isinstance(denominators, dict), f"{chart_name} missing denominator_certificates")
    stored = denominators.get(field)
    require(isinstance(stored, dict), f"{chart_name} missing denominator certificate {field}")
    require(stored.get("avoids_zero") is True, f"{chart_name} denominator {field} must avoid zero")
    require(
        interval_contains(denominator_interval(stored), denominator_interval(bridge_certificate)),
        f"{chart_name} denominator {field} misses seed bridge interval",
    )


def metadata_spin(atlas: dict[str, Any]) -> Fraction:
    metadata = atlas.get("metadata")
    require(isinstance(metadata, dict), "atlas must contain metadata")
    require("spin" in metadata, "atlas metadata missing spin")
    return parse_fraction(metadata["spin"])


def sqrt_digits(atlas: dict[str, Any]) -> int:
    metadata = atlas.get("metadata", {})
    require(isinstance(metadata, dict), "atlas metadata must be an object")
    return int(metadata.get("sqrt_enclosure_decimal_digits", 12))


def chart_records(atlas: dict[str, Any]) -> list[tuple[str, Interval, dict[str, Any]]]:
    charts = atlas.get("charts")
    require(isinstance(charts, list) and charts, "atlas must contain a nonempty charts list")
    records = []
    seen = set()
    for index, chart in enumerate(charts):
        require(isinstance(chart, dict), f"charts[{index}] must be an object")
        name = chart.get("chart")
        require(isinstance(name, str) and name, f"charts[{index}].chart must be a nonempty string")
        require(name not in seen, f"duplicate chart name {name}")
        seen.add(name)
        records.append((name, parse_interval_pair(chart.get("r_interval"), name), chart))
    return sorted(records, key=lambda item: (item[1].lo, item[1].hi, item[0]))


def containing_chart(atlas: dict[str, Any], seed_interval: Interval, seed_name: str) -> tuple[str, Interval, dict[str, Any]]:
    matches = [
        record
        for record in chart_records(atlas)
        if interval_contains(record[1], seed_interval)
    ]
    require(matches, f"no generic radial chart contains {seed_name} seed interval {seed_interval.to_json()}")
    require(len(matches) == 1, f"more than one generic radial chart contains {seed_name} seed interval")
    return matches[0]


def generic_bridge_values(spin: Fraction, r_interval: Interval, digits: int) -> dict[str, Any]:
    a = point(spin)
    lam = lambda_interval(r_interval, a)
    eta = eta_interval(r_interval, a, lam)
    hessian = radial_hessian_interval(r_interval, a, lam, eta)
    omega_h = omega_h_interval(spin, digits)
    superradiant = point_interval(1) - omega_h * lam
    trapped_scale = Interval(Fraction(1), Fraction(1) + hessian.abs_upper())
    r_plus = horizon_radius_interval(spin, digits)
    return {
        "lambda_interval": interval_json(lam),
        "eta_interval": interval_json(eta),
        "radial_hessian_interval": interval_json(hessian),
        "omega_h_interval": interval_json(omega_h),
        "superradiant_factor_interval": interval_json(superradiant),
        "trapped_scale_interval": interval_json(trapped_scale),
        "denominator_certificates": denominator_certificates(r_interval, a, r_plus),
    }


def require_chart_contains_bridge(chart: dict[str, Any], chart_name: str, bridge_values: dict[str, Any]) -> None:
    for field in INTERVAL_FIELDS:
        require(field in chart, f"{chart_name} missing {field}")
        require(
            interval_contains(parse_interval(chart[field]), parse_interval(bridge_values[field])),
            f"{chart_name} {field} misses seed bridge interval",
        )
    denominators = bridge_values["denominator_certificates"]
    require(isinstance(denominators, dict), "bridge denominator_certificates must be an object")
    for field in DENOMINATOR_FIELDS:
        require(field in denominators, f"bridge denominator_certificates missing {field}")
        require_denominator_contains(chart, chart_name, field, denominators[field])


def require_seed_contained_in_generic(seed_values: dict[str, Interval], bridge_values: dict[str, Any], seed_name: str) -> None:
    comparisons = {
        "lambda": "lambda_interval",
        "eta": "eta_interval",
        "radial_hessian": "radial_hessian_interval",
    }
    for seed_field, bridge_field in comparisons.items():
        require(
            interval_contains(parse_interval(bridge_values[bridge_field]), seed_values[seed_field]),
            f"{seed_name} regular {seed_field} interval is not contained in generic bridge data",
        )


def polar_seed_data(polar: dict[str, Any]) -> dict[str, Any]:
    require(polar.get("accepted") is True, "polar seed certificate is not accepted")
    require(polar.get("chart_family") == "polar_L_equals_zero_seed", "unexpected polar seed chart_family")
    r_interval = required_interval(polar, ["interval_bounds", "r_over_M"], "polar seed")
    return {
        "name": "polar_L_equals_zero",
        "seed_certificate_family": polar["chart_family"],
        "branch": "exterior",
        "r_interval": r_interval,
        "seed_values": {
            "lambda": point_interval(0),
            "eta": required_interval(polar, ["interval_bounds", "eta_over_M_squared"], "polar seed"),
            "radial_hessian": required_interval(
                polar,
                ["interval_bounds", "radial_hessian_over_M_squared"],
                "polar seed",
            ),
        },
    }


def equatorial_seed_data(equatorial: dict[str, Any]) -> list[dict[str, Any]]:
    require(equatorial.get("accepted") is True, "equatorial seed certificate is not accepted")
    require(
        equatorial.get("chart_family") == "equatorial_Q_equals_zero_seed",
        "unexpected equatorial seed chart_family",
    )
    branches = equatorial.get("branches")
    require(isinstance(branches, dict), "equatorial certificate missing branches")
    result = []
    for branch in ["prograde", "retrograde"]:
        require(branch in branches and isinstance(branches[branch], dict), f"equatorial missing {branch} branch")
        data = branches[branch]
        result.append(
            {
                "name": f"equatorial_Q_equals_zero_{branch}",
                "seed_certificate_family": equatorial["chart_family"],
                "branch": branch,
                "r_interval": required_interval(data, ["intervals", "r_over_M"], f"equatorial {branch}"),
                "seed_values": {
                    "lambda": required_interval(data, ["intervals", "lambda"], f"equatorial {branch}"),
                    "eta": required_interval(data, ["intervals", "eta"], f"equatorial {branch}"),
                    "radial_hessian": required_interval(
                        data,
                        ["intervals", "radial_hessian_over_M_squared"],
                        f"equatorial {branch}",
                    ),
                },
            }
        )
    return result


def bridge_record(atlas: dict[str, Any], spin: Fraction, digits: int, seed: dict[str, Any]) -> dict[str, Any]:
    chart_name, chart_interval, chart = containing_chart(atlas, seed["r_interval"], seed["name"])
    values = generic_bridge_values(spin, seed["r_interval"], digits)
    require_chart_contains_bridge(chart, chart_name, values)
    require_seed_contained_in_generic(seed["seed_values"], values, seed["name"])
    return {
        "seed_family": seed["name"],
        "seed_certificate_family": seed["seed_certificate_family"],
        "branch": seed["branch"],
        "seed_r_interval": interval_json(seed["r_interval"]),
        "containing_generic_chart": chart_name,
        "containing_generic_chart_r_interval": interval_json(chart_interval),
        "regular_seed_values": {
            "lambda_interval": interval_json(seed["seed_values"]["lambda"]),
            "eta_interval": interval_json(seed["seed_values"]["eta"]),
            "radial_hessian_interval": interval_json(seed["seed_values"]["radial_hessian"]),
        },
        "generic_bridge_values": values,
        "verified_containments": [
            "seed_r_interval_inside_containing_generic_chart",
            "generic_bridge_values_inside_containing_generic_chart",
            "regular_seed_lambda_eta_hessian_inside_generic_bridge_values",
            "seed_denominators_inside_containing_generic_chart_certificates",
        ],
    }


def compute_bridge_certificate(
    atlas: dict[str, Any],
    polar: dict[str, Any],
    equatorial: dict[str, Any],
    source_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
    spin = metadata_spin(atlas)
    require(spin == Fraction(1, 2), "degenerate seed bridge currently targets the spin-half atlas")
    digits = sqrt_digits(atlas)
    seeds = [polar_seed_data(polar), *equatorial_seed_data(equatorial)]
    bridges = [bridge_record(atlas, spin, digits, seed) for seed in seeds]
    metadata: dict[str, Any] = {
        "name": "spin-half degenerate seed radial bridge certificate",
        "spin": str(spin),
        "mass_normalization": "M=1",
        "status": "radial bridge only; angular-neighborhood charts and overlaps not proved",
        "warning": "This is not a Kerr stability proof certificate.",
    }
    if source_paths is not None:
        metadata.update(source_paths)
    return {
        "accepted": True,
        "metadata": metadata,
        "bridges": bridges,
    }


def resolved_source_path(package_path: Path, raw: Any, key: str) -> Path:
    require(isinstance(raw, str) and raw, f"metadata.{key} must be a nonempty path")
    path = Path(raw)
    if not path.is_absolute():
        path = package_path.parent / path
    return path


def verify_package(path: Path) -> dict[str, Any]:
    stored = load_json(path)
    metadata = stored.get("metadata")
    require(isinstance(metadata, dict), "bridge package metadata must be an object")
    source_paths = {
        "atlas_path": str(metadata.get("atlas_path", "")),
        "polar_seed_path": str(metadata.get("polar_seed_path", "")),
        "equatorial_seed_path": str(metadata.get("equatorial_seed_path", "")),
    }
    atlas = load_json(resolved_source_path(path, source_paths["atlas_path"], "atlas_path"))
    polar = load_json(resolved_source_path(path, source_paths["polar_seed_path"], "polar_seed_path"))
    equatorial = load_json(resolved_source_path(path, source_paths["equatorial_seed_path"], "equatorial_seed_path"))
    computed = compute_bridge_certificate(atlas, polar, equatorial, source_paths=source_paths)
    require(stored == computed, "degenerate seed bridge certificate is stale or incorrect")
    return summary(computed)


def summary(certificate: dict[str, Any]) -> dict[str, Any]:
    return {
        "accepted": True,
        "bridge_count": len(certificate["bridges"]),
        "bridges": [
            {
                "seed_family": bridge["seed_family"],
                "seed_r_interval": bridge["seed_r_interval"],
                "containing_generic_chart": bridge["containing_generic_chart"],
            }
            for bridge in certificate["bridges"]
        ],
    }


def relative_source_paths(output: Path, atlas: Path, polar: Path, equatorial: Path) -> dict[str, str]:
    base = output.parent.resolve()
    return {
        "atlas_path": os.path.relpath(atlas.resolve(), base),
        "polar_seed_path": os.path.relpath(polar.resolve(), base),
        "equatorial_seed_path": os.path.relpath(equatorial.resolve(), base),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])

    try:
        if len(args.inputs) == 1 and args.output is None:
            print(json.dumps(verify_package(args.inputs[0]), indent=2, sort_keys=True))
            return 0
        require(len(args.inputs) == 3, "provide either a bridge package or atlas, polar seed, and equatorial seed")
        atlas_path, polar_path, equatorial_path = args.inputs
        atlas = load_json(atlas_path)
        polar = load_json(polar_path)
        equatorial = load_json(equatorial_path)
        source_paths = None
        if args.output is not None:
            source_paths = relative_source_paths(args.output, atlas_path, polar_path, equatorial_path)
        certificate = compute_bridge_certificate(atlas, polar, equatorial, source_paths=source_paths)
        text = json.dumps(certificate, indent=2, sort_keys=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter degenerate seed bridge rejected: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
