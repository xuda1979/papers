#!/usr/bin/env python3
"""Verify exact Carter compatibility on overlapping radial charts."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from generate_carter_atlas import (
    Interval,
    ONE,
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


def metadata_spin(data: dict[str, Any]) -> Fraction:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "atlas must contain metadata")
    require("spin" in metadata, "atlas metadata must contain spin")
    return parse_fraction(metadata["spin"])


def sqrt_digits(data: dict[str, Any]) -> int:
    metadata = data.get("metadata", {})
    require(isinstance(metadata, dict), "atlas metadata must be an object")
    return int(metadata.get("sqrt_enclosure_decimal_digits", 12))


def chart_records(data: dict[str, Any]) -> list[tuple[str, Interval, dict[str, Any]]]:
    charts = data.get("charts")
    require(isinstance(charts, list) and charts, "atlas must contain a nonempty charts list")
    seen = set()
    result = []
    for index, chart in enumerate(charts):
        require(isinstance(chart, dict), f"charts[{index}] must be an object")
        name = chart.get("chart")
        require(isinstance(name, str) and name, f"charts[{index}].chart must be a nonempty string")
        require(name not in seen, f"duplicate chart name {name}")
        seen.add(name)
        result.append((name, parse_interval(chart.get("r_interval")), chart))
    return sorted(result, key=lambda item: (item[1].lo, item[1].hi, item[0]))


def proper_overlap(left: Interval, right: Interval) -> Interval | None:
    lo = max(left.lo, right.lo)
    hi = min(left.hi, right.hi)
    if lo < hi:
        return Interval(lo, hi)
    return None


def interval_contains(container: Interval, contained: Interval) -> bool:
    return container.lo <= contained.lo and contained.hi <= container.hi


def require_chart_contains_interval(chart: dict[str, Any], chart_name: str, field: str, interval: Interval) -> None:
    require(field in chart, f"{chart_name} missing {field}")
    stored = parse_interval(chart[field])
    require(interval_contains(stored, interval), f"{chart_name} {field} misses overlap interval {interval.to_json()}")


def require_denominator_contains_interval(
    chart: dict[str, Any],
    chart_name: str,
    field: str,
    certificate: dict[str, Any],
) -> None:
    denominators = chart.get("denominator_certificates")
    require(isinstance(denominators, dict), f"{chart_name} missing denominator certificates")
    stored = denominators.get(field)
    require(isinstance(stored, dict), f"{chart_name} missing denominator certificate {field}")
    require(stored.get("avoids_zero") is True, f"{chart_name} denominator certificate {field} must avoid zero")
    stored_interval = parse_interval(stored.get("interval"))
    overlap_interval = parse_interval(certificate["interval"])
    require(
        interval_contains(stored_interval, overlap_interval),
        f"{chart_name} denominator certificate {field} misses overlap interval {overlap_interval.to_json()}",
    )


def overlap_values(spin: Fraction, overlap: Interval, digits: int) -> dict[str, Any]:
    a = point(spin)
    lam = lambda_interval(overlap, a)
    eta = eta_interval(overlap, a, lam)
    hessian = radial_hessian_interval(overlap, a, lam, eta)
    omega_h = omega_h_interval(spin, digits)
    superradiant = ONE - omega_h * lam
    trapped_scale = Interval(Fraction(1), Fraction(1) + hessian.abs_upper())
    r_plus = horizon_radius_interval(spin, digits)
    return {
        "lambda_interval": lam.to_json(),
        "eta_interval": eta.to_json(),
        "radial_hessian_interval": hessian.to_json(),
        "omega_h_interval": omega_h.to_json(),
        "superradiant_factor_interval": superradiant.to_json(),
        "trapped_scale_interval": trapped_scale.to_json(),
        "denominator_certificates": denominator_certificates(overlap, a, r_plus),
    }


def overlap_certificate(
    left_name: str,
    left_interval: Interval,
    right_name: str,
    right_interval: Interval,
    spin: Fraction,
    digits: int,
) -> dict[str, Any] | None:
    overlap = proper_overlap(left_interval, right_interval)
    if overlap is None:
        return None
    return {
        "left": left_name,
        "right": right_name,
        "overlap_interval": overlap.to_json(),
        "overlap_values": overlap_values(spin, overlap, digits),
    }


def computed_overlap_certificates(data: dict[str, Any]) -> list[dict[str, Any]]:
    spin = metadata_spin(data)
    digits = sqrt_digits(data)
    records = chart_records(data)
    certificates = []
    for left_index, (left_name, left_interval, _) in enumerate(records):
        for right_name, right_interval, _ in records[left_index + 1 :]:
            certificate = overlap_certificate(left_name, left_interval, right_name, right_interval, spin, digits)
            if certificate is not None:
                certificates.append(certificate)
    return certificates


def verify_chart_pair_contains_certificate(
    chart_by_name: dict[str, dict[str, Any]],
    certificate: dict[str, Any],
) -> None:
    left = certificate["left"]
    right = certificate["right"]
    require(left in chart_by_name, f"overlap references missing chart {left}")
    require(right in chart_by_name, f"overlap references missing chart {right}")
    values = certificate["overlap_values"]
    for chart_name in [left, right]:
        chart = chart_by_name[chart_name]
        for field in INTERVAL_FIELDS:
            require_chart_contains_interval(chart, chart_name, field, parse_interval(values[field]))
        denominators = values["denominator_certificates"]
        require(isinstance(denominators, dict), "overlap denominator_certificates must be an object")
        for field in DENOMINATOR_FIELDS:
            require(field in denominators, f"overlap denominator_certificates missing {field}")
            require_denominator_contains_interval(chart, chart_name, field, denominators[field])


def verify_stored_certificates(data: dict[str, Any], computed: list[dict[str, Any]], require_certificates: bool) -> None:
    stored = data.get("overlap_certificates")
    if stored is None:
        require(not require_certificates, "atlas must contain overlap_certificates")
        return
    require(isinstance(stored, list), "overlap_certificates must be a list")
    require(stored == computed, "overlap_certificates are stale or incorrect")


def verify(data: dict[str, Any], require_certificates: bool = False) -> dict[str, Any]:
    records = chart_records(data)
    chart_by_name = {name: chart for name, _, chart in records}
    computed = computed_overlap_certificates(data)
    for certificate in computed:
        verify_chart_pair_contains_certificate(chart_by_name, certificate)
    verify_stored_certificates(data, computed, require_certificates)
    return {
        "accepted": True,
        "chart_count": len(records),
        "overlap_count": len(computed),
        "overlaps": [
            {
                "left": certificate["left"],
                "right": certificate["right"],
                "overlap_interval": certificate["overlap_interval"],
            }
            for certificate in computed
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--require-certificates", action="store_true")
    args = parser.parse_args(argv[1:])
    try:
        result = verify(load_json(args.atlas), require_certificates=args.require_certificates)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter overlap certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
