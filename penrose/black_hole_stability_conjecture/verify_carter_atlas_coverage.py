#!/usr/bin/env python3
"""Verify exact radial coverage and transition data for a Carter atlas."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from generate_carter_atlas import (
    Interval,
    eta_interval,
    lambda_interval,
    parse_fraction,
    parse_interval,
    radial_hessian_interval,
    transition_endpoint_data,
    point,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def chart_intervals(data: dict[str, Any]) -> list[tuple[str, Interval]]:
    charts = data.get("charts")
    require(isinstance(charts, list) and charts, "atlas must contain a nonempty charts list")
    result = []
    seen_names = set()
    for index, chart in enumerate(charts):
        require(isinstance(chart, dict), f"charts[{index}] must be an object")
        name = chart.get("chart", f"chart_{index}")
        require(isinstance(name, str) and name, f"charts[{index}].chart must be a nonempty string")
        require(name not in seen_names, f"duplicate chart name {name}")
        seen_names.add(name)
        result.append((name, parse_interval(chart.get("r_interval"))))
    return result


def metadata_spin(data: dict[str, Any]) -> Fraction:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "atlas must contain metadata")
    require("spin" in metadata, "atlas metadata must contain spin")
    return parse_fraction(metadata["spin"])


def parse_transition_endpoint_values(raw: Any, index: int) -> dict[str, str]:
    require(isinstance(raw, dict), f"transitions[{index}].endpoint_values must be an object")
    required = [
        "lambda",
        "eta",
        "radial_hessian",
        "r_minus_M",
        "(r_minus_M)^2",
        "Delta",
        "omega_h_interval",
        "superradiant_factor_interval",
        "trapped_scale_interval",
    ]
    values = {}
    for key in required:
        require(key in raw, f"transitions[{index}].endpoint_values missing {key}")
        values[key] = str(raw[key])
    return values


def transition_records(data: dict[str, Any], expected_count: int) -> list[dict[str, Any]]:
    records = data.get("transitions")
    require(isinstance(records, list), "atlas must contain a transitions list")
    require(len(records) == expected_count, f"atlas has {len(records)} transitions, expected {expected_count}")
    result = []
    for index, record in enumerate(records):
        require(isinstance(record, dict), f"transitions[{index}] must be an object")
        left = record.get("left")
        right = record.get("right")
        shared_endpoint = record.get("shared_endpoint")
        require(isinstance(left, str) and left, f"transitions[{index}].left must be a nonempty string")
        require(isinstance(right, str) and right, f"transitions[{index}].right must be a nonempty string")
        require(
            isinstance(shared_endpoint, str) and shared_endpoint,
            f"transitions[{index}].shared_endpoint must be a nonempty string",
        )
        result.append(
            {
                "left": left,
                "right": right,
                "shared_endpoint": Fraction(shared_endpoint),
                "endpoint_values": parse_transition_endpoint_values(record.get("endpoint_values"), index),
            }
        )
    return result


def interval_contains(interval_text: str, value: Fraction) -> bool:
    interval = parse_interval(interval_text)
    return interval.lo <= value <= interval.hi


def require_endpoint_inside_chart(
    chart_by_name: dict[str, dict[str, Any]],
    chart_name: str,
    exact_values: dict[str, Any],
) -> None:
    require(chart_name in chart_by_name, f"transition references missing chart {chart_name}")
    chart = chart_by_name[chart_name]
    require(interval_contains(chart["lambda_interval"], Fraction(exact_values["lambda"])), f"{chart_name} lambda interval misses endpoint value")
    require(interval_contains(chart["eta_interval"], Fraction(exact_values["eta"])), f"{chart_name} eta interval misses endpoint value")
    require(
        interval_contains(chart["radial_hessian_interval"], Fraction(exact_values["radial_hessian"])),
        f"{chart_name} radial Hessian interval misses endpoint value",
    )
    superradiant = parse_interval(exact_values["superradiant_factor_interval"])
    chart_superradiant = parse_interval(chart["superradiant_factor_interval"])
    require(
        chart_superradiant.lo <= superradiant.lo and superradiant.hi <= chart_superradiant.hi,
        f"{chart_name} superradiant interval misses endpoint interval",
    )
    trapped_scale = parse_interval(exact_values["trapped_scale_interval"])
    chart_trapped_scale = parse_interval(chart["trapped_scale_interval"])
    require(
        chart_trapped_scale.lo <= trapped_scale.lo and trapped_scale.hi <= chart_trapped_scale.hi,
        f"{chart_name} trapped-scale interval misses endpoint interval",
    )

    denominator_values = {
        "r_minus_M": exact_values["r_minus_M"],
        "(r_minus_M)^2": exact_values["(r_minus_M)^2"],
        "Delta": exact_values["Delta"],
    }
    denominator_certificates = chart.get("denominator_certificates")
    require(isinstance(denominator_certificates, dict), f"{chart_name} missing denominator certificates")
    for key, raw_value in denominator_values.items():
        certificate = denominator_certificates.get(key)
        require(isinstance(certificate, dict), f"{chart_name} missing denominator certificate {key}")
        certificate_interval = certificate.get("interval")
        require(isinstance(certificate_interval, str), f"{chart_name} denominator certificate {key} has no interval")
        require(interval_contains(certificate_interval, Fraction(raw_value)), f"{chart_name} denominator certificate {key} misses endpoint value")


def verify_transition_records(
    data: dict[str, Any],
    computed_transitions: list[dict[str, str]],
) -> list[dict[str, str]]:
    spin = metadata_spin(data)
    digits = int(data.get("metadata", {}).get("sqrt_enclosure_decimal_digits", 12))
    records = transition_records(data, len(computed_transitions))
    chart_by_name = {str(chart["chart"]): chart for chart in data["charts"]}
    expected_pairs = {(record["left"], record["right"]) for record in computed_transitions}
    seen_pairs = set()

    for index, (record, expected) in enumerate(zip(records, computed_transitions)):
        require(record["left"] == expected["left"], f"transitions[{index}].left does not match sorted atlas order")
        require(record["right"] == expected["right"], f"transitions[{index}].right does not match sorted atlas order")
        require(
            record["shared_endpoint"] == Fraction(expected["shared_endpoint"]),
            f"transitions[{index}].shared_endpoint does not match chart boundary",
        )
        pair = (record["left"], record["right"])
        require(pair in expected_pairs, f"unexpected transition pair {pair}")
        require(pair not in seen_pairs, f"duplicate transition pair {pair}")
        seen_pairs.add(pair)

        exact_values = transition_endpoint_data(spin, record["shared_endpoint"], digits)
        require(record["endpoint_values"] == exact_values, f"transitions[{index}].endpoint_values are stale or incorrect")
        require_endpoint_inside_chart(chart_by_name, record["left"], exact_values)
        require_endpoint_inside_chart(chart_by_name, record["right"], exact_values)

        r = point(record["shared_endpoint"])
        a = point(spin)
        lam = lambda_interval(r, a)
        eta = eta_interval(r, a, lam)
        hessian = radial_hessian_interval(r, a, lam, eta)
        require(lam.lo == lam.hi, f"transition {pair} lambda endpoint did not collapse to a point")
        require(eta.lo == eta.hi, f"transition {pair} eta endpoint did not collapse to a point")
        require(hessian.lo == hessian.hi, f"transition {pair} radial Hessian endpoint did not collapse to a point")

    require(seen_pairs == expected_pairs, "transition records do not match chart adjacency")
    return [
        {
            "left": record["left"],
            "right": record["right"],
            "shared_endpoint": str(record["shared_endpoint"]),
            "lambda": record["endpoint_values"]["lambda"],
            "eta": record["endpoint_values"]["eta"],
        }
        for record in records
    ]


def verify(
    data: dict[str, Any],
    expected_start: Fraction | None = None,
    expected_end: Fraction | None = None,
) -> dict[str, Any]:
    intervals = sorted(chart_intervals(data), key=lambda item: (item[1].lo, item[1].hi, item[0]))
    require(intervals[0][1].lo < intervals[0][1].hi, f"{intervals[0][0]} has empty interval")

    computed_transitions = []
    for (left_name, left), (right_name, right) in zip(intervals, intervals[1:]):
        require(left.lo < left.hi, f"{left_name} has empty interval")
        require(right.lo < right.hi, f"{right_name} has empty interval")
        if left.hi < right.lo:
            raise ValueError(f"gap between {left_name} and {right_name}: {left.hi} < {right.lo}")
        if left.hi > right.lo:
            raise ValueError(f"overlap between {left_name} and {right_name}: {left.hi} > {right.lo}")
        computed_transitions.append({"left": left_name, "right": right_name, "shared_endpoint": str(left.hi)})

    if expected_start is not None:
        require(intervals[0][1].lo == expected_start, f"coverage starts at {intervals[0][1].lo}, expected {expected_start}")
    if expected_end is not None:
        require(intervals[-1][1].hi == expected_end, f"coverage ends at {intervals[-1][1].hi}, expected {expected_end}")

    transitions = verify_transition_records(data, computed_transitions)

    return {
        "accepted": True,
        "chart_count": len(intervals),
        "coverage": [str(intervals[0][1].lo), str(intervals[-1][1].hi)],
        "transitions": transitions,
    }


def parse_optional_fraction(raw: str | None) -> Fraction | None:
    if raw is None:
        return None
    return Fraction(raw)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--expected-start")
    parser.add_argument("--expected-end")
    args = parser.parse_args(argv[1:])

    try:
        result = verify(
            load_json(args.atlas),
            expected_start=parse_optional_fraction(args.expected_start),
            expected_end=parse_optional_fraction(args.expected_end),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter atlas coverage rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
