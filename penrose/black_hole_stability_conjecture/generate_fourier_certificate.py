#!/usr/bin/env python3
"""Generate a toy Fourier-margin certificate block.

This is a scaffold for the trapped-superradiant route. It takes finite Fourier
coefficients for a real trigonometric polynomial and computes conservative
rational lower bounds for resonant averages by the triangle inequality:

    average >= p_0 - sum_{k in resonance, k != 0} |p_k|.

For a real Kerr certificate, the input coefficients must come from the actual
Kerr hidden-symmetry density on trapped torus charts, with interval arithmetic
and chart coverage. This script only supplies the plumbing and a checkable JSON
shape for that future data.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


def frac(raw: Any) -> Fraction:
    return Fraction(str(raw))


def interval(lo: Fraction, hi: Fraction) -> str:
    if lo > hi:
        raise ValueError("invalid interval")
    return f"[{lo}, {hi}]"


def coefficient_abs_bound(value: Any) -> Fraction:
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("complex coefficient lists must be [real, imag]")
        real = abs(frac(value[0]))
        imag = abs(frac(value[1]))
        # Safe l1 bound for |real + i imag|.
        return real + imag
    return abs(frac(value))


def lower_bound_for_resonance(coefficients: dict[str, Any], resonance: list[str]) -> Fraction:
    p0 = frac(coefficients.get("0", "0"))
    loss = Fraction(0)
    for mode in resonance:
        if mode == "0":
            continue
        if mode in coefficients:
            loss += coefficient_abs_bound(coefficients[mode])
    return p0 - loss


def generate(data: dict[str, Any]) -> dict[str, Any]:
    chart = data.get("chart", "unnamed_chart")
    coefficients = data.get("coefficients")
    resonances = data.get("resonances")
    approximation_error = frac(data.get("approximation_error", "0"))
    if not isinstance(coefficients, dict) or "0" not in coefficients:
        raise ValueError("coefficients must be an object containing the constant mode '0'")
    if not isinstance(resonances, list) or not resonances:
        raise ValueError("resonances must be a nonempty list")
    if approximation_error < 0:
        raise ValueError("approximation_error must be nonnegative")

    bounds = []
    global_lower = None
    for index, resonance in enumerate(resonances):
        if not isinstance(resonance, list):
            raise ValueError(f"resonance {index} must be a list of mode strings")
        lower = lower_bound_for_resonance(coefficients, resonance)
        global_lower = lower if global_lower is None else min(global_lower, lower)
        bounds.append(
            {
                "resonance": resonance,
                "triangle_lower_bound": interval(lower, lower),
                "dominates_approximation_error": lower > approximation_error,
            }
        )

    assert global_lower is not None
    return {
        "chart": chart,
        "fourier_polynomial": data.get("fourier_polynomial", "finite Fourier polynomial"),
        "approximation_error": interval(Fraction(0), approximation_error),
        "resonant_average_lower_bound": interval(global_lower, global_lower),
        "resonance_bounds": bounds,
        "accepted_by_margin_test": global_lower > approximation_error,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])

    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        result = generate(data)
    except (ValueError, KeyError) as exc:
        print(f"fourier certificate generation failed: {exc}", file=sys.stderr)
        return 1

    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
