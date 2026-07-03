#!/usr/bin/env python3
"""Rank Carter atlas charts by interval quality.

This script helps choose the first chart for an SOS-plus-transport attempt. It
does not prove positivity; it scores how sharp the interval enclosures are and
flags sign-definite superradiant factors.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

from generate_carter_atlas import Interval, parse_interval


QUALITY_FIELDS = [
    "lambda_interval",
    "eta_interval",
    "superradiant_factor_interval",
    "radial_hessian_interval",
]


def width(interval: Interval) -> Fraction:
    return interval.hi - interval.lo


def midpoint_abs_scale(interval: Interval) -> Fraction:
    midpoint_twice = interval.lo + interval.hi
    return max(Fraction(1), abs(midpoint_twice) / 2)


def relative_width(interval: Interval) -> Fraction:
    return width(interval) / midpoint_abs_scale(interval)


def sign_class(interval: Interval) -> str:
    if interval.lo > 0:
        return "positive"
    if interval.hi < 0:
        return "negative"
    if interval.lo == 0 and interval.hi == 0:
        return "zero"
    return "mixed"


def score_chart(chart: dict[str, Any]) -> dict[str, Any]:
    metrics = {}
    score = Fraction(0)
    for field in QUALITY_FIELDS:
        interval = parse_interval(chart[field])
        field_width = width(interval)
        field_relative = relative_width(interval)
        score += field_relative
        metrics[field] = {
            "width": str(field_width),
            "relative_width": str(field_relative),
            "sign": sign_class(interval),
        }

    superradiant = parse_interval(chart["superradiant_factor_interval"])
    hessian = parse_interval(chart["radial_hessian_interval"])
    flags = []
    if sign_class(superradiant) == "mixed":
        flags.append("superradiant_factor_crosses_zero")
    if sign_class(hessian) == "mixed":
        flags.append("radial_hessian_crosses_zero")

    return {
        "chart": chart["chart"],
        "score": str(score),
        "score_float": float(score),
        "r_interval": chart["r_interval"],
        "metrics": metrics,
        "flags": flags,
        "source": chart,
    }


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    charts = data.get("charts")
    if not isinstance(charts, list) or not charts:
        raise ValueError("atlas must contain a nonempty charts list")
    ranked = sorted((score_chart(chart) for chart in charts), key=lambda item: Fraction(item["score"]))
    return {
        "metadata": {
            **data.get("metadata", {}),
            "analysis_status": "quality ranking only; not a positivity proof",
            "quality_fields": QUALITY_FIELDS,
        },
        "best_chart": ranked[0]["chart"],
        "ranked_charts": ranked,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("atlas", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=0, help="print only top N charts to stdout; output file still gets all")
    args = parser.parse_args(argv[1:])

    with args.atlas.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        result = analyze(data)
    except ValueError as exc:
        print(f"Carter atlas analysis failed: {exc}", file=sys.stderr)
        return 1

    output_result = result
    if args.top > 0:
        output_result = {**result, "ranked_charts": result["ranked_charts"][: args.top]}

    text = json.dumps(output_result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
