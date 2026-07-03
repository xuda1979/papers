#!/usr/bin/env python3
"""Generate rational interval data for Carter trapped-orbit atlas charts.

The output is not a proof certificate. It supplies the Kerr-specific chart data
needed before an SOS-plus-transport Carter cone certificate can be attempted:
spherical-orbit lambda/eta intervals, normalized superradiant factor intervals,
and a trapped radial-Hessian scale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __mul__(self, other: "Interval") -> "Interval":
        values = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return Interval(min(values), max(values))

    def reciprocal(self) -> "Interval":
        if self.lo <= 0 <= self.hi:
            raise ValueError(f"cannot divide by interval containing zero: {self}")
        values = (Fraction(1, self.lo), Fraction(1, self.hi))
        return Interval(min(values), max(values))

    def __truediv__(self, other: "Interval") -> "Interval":
        return self * other.reciprocal()

    def square(self) -> "Interval":
        if self.lo <= 0 <= self.hi:
            return Interval(Fraction(0), max(self.lo * self.lo, self.hi * self.hi))
        values = (self.lo * self.lo, self.hi * self.hi)
        return Interval(min(values), max(values))

    def abs_upper(self) -> Fraction:
        return max(abs(self.lo), abs(self.hi))

    def to_json(self) -> str:
        return f"[{self.lo}, {self.hi}]"


ZERO = Interval(Fraction(0), Fraction(0))
ONE = Interval(Fraction(1), Fraction(1))
TWO = Interval(Fraction(2), Fraction(2))
FOUR = Interval(Fraction(4), Fraction(4))


def point(value: Fraction | int) -> Interval:
    value = Fraction(value)
    return Interval(value, value)


def parse_fraction(raw: Any) -> Fraction:
    return Fraction(str(raw).strip())


def parse_interval(raw: Any) -> Interval:
    if isinstance(raw, list) and len(raw) == 2:
        return Interval(parse_fraction(raw[0]), parse_fraction(raw[1]))
    if not isinstance(raw, str):
        raise ValueError(f"interval must be string or [lo, hi], got {raw!r}")
    text = raw.strip()
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"interval must look like [lo, hi], got {raw!r}")
    parts = text[1:-1].split(",")
    if len(parts) != 2:
        raise ValueError(f"interval must have two endpoints, got {raw!r}")
    return Interval(parse_fraction(parts[0]), parse_fraction(parts[1]))


def sqrt_fraction_interval(value: Fraction, digits: int = 12) -> Interval:
    """Return a decimal-rational interval enclosing sqrt(value)."""
    if value < 0:
        raise ValueError("cannot take square root of negative value")
    scale = 10**digits
    n = math.isqrt(value.numerator * scale * scale // value.denominator)
    lo = Fraction(n, scale)
    hi = Fraction(n, scale)
    if hi * hi < value:
        hi = Fraction(n + 1, scale)
    if lo * lo > value:
        lo = Fraction(n - 1, scale)
    return Interval(lo, hi)


def delta(r: Interval, a: Interval) -> Interval:
    return r.square() - TWO * r + a.square()


def lambda_interval(r: Interval, a: Interval) -> Interval:
    d = delta(r, a)
    first = (r.square() + a.square()) / a
    second = (TWO * r * d) / (a * (r - ONE))
    return first - second


def eta_interval(r: Interval, a: Interval, lam: Interval) -> Interval:
    d = delta(r, a)
    first = (FOUR * r.square() * d) / ((r - ONE).square())
    return first - (lam - a).square()


def radial_hessian_interval(r: Interval, a: Interval, lam: Interval, eta: Interval) -> Interval:
    # R/E^2 = A^2 - Delta B, A = r^2 + a^2 - a lambda,
    # B = eta + (lambda-a)^2. Holding lambda and eta fixed,
    # d_r^2(R/E^2) = 4A + 8r^2 - 2B.
    A = r.square() + a.square() - a * lam
    B = eta + (lam - a).square()
    return point(4) * A + point(8) * r.square() - TWO * B


def point_value(interval: Interval, name: str) -> Fraction:
    if interval.lo != interval.hi:
        raise ValueError(f"{name} is not a point interval: {interval}")
    return interval.lo


def omega_h_interval(a_value: Fraction, digits: int) -> Interval:
    if abs(a_value) >= 1:
        raise ValueError("spin must be strictly subextremal with M=1")
    a = point(a_value)
    sqrt_term = sqrt_fraction_interval(Fraction(1) - a_value * a_value, digits)
    r_plus = ONE + sqrt_term
    return a / (TWO * r_plus)


def sign_certificate(interval: Interval) -> dict[str, str | bool]:
    avoids_zero = not (interval.lo <= 0 <= interval.hi)
    result: dict[str, str | bool] = {
        "interval": interval.to_json(),
        "avoids_zero": avoids_zero,
    }
    if interval.lo > 0:
        result["sign"] = "positive"
        result["lower_abs_bound"] = str(interval.lo)
    elif interval.hi < 0:
        result["sign"] = "negative"
        result["lower_abs_bound"] = str(-interval.hi)
    elif interval.lo == 0 and interval.hi == 0:
        result["sign"] = "zero"
        result["lower_abs_bound"] = "0"
    else:
        result["sign"] = "mixed"
        result["lower_abs_bound"] = "0"
    return result


def denominator_certificates(r: Interval, a: Interval, r_plus: Interval) -> dict[str, dict[str, str | bool]]:
    r_minus_m = r - ONE
    return {
        "a": sign_certificate(a),
        "r_minus_M": sign_certificate(r_minus_m),
        "(r_minus_M)^2": sign_certificate(r_minus_m.square()),
        "Delta": sign_certificate(delta(r, a)),
        "r_plus": sign_certificate(r_plus),
    }


def transition_endpoint_data(spin: Fraction, r_value: Fraction, digits: int) -> dict[str, Any]:
    a = point(spin)
    r = point(r_value)
    lam = lambda_interval(r, a)
    eta = eta_interval(r, a, lam)
    hessian = radial_hessian_interval(r, a, lam, eta)
    omega_h = omega_h_interval(spin, digits)
    superradiant = ONE - omega_h * lam
    trapped_scale = Interval(Fraction(1), Fraction(1) + hessian.abs_upper())
    r_minus_m = r - ONE
    delta_value = delta(r, a)
    return {
        "lambda": str(point_value(lam, "lambda")),
        "eta": str(point_value(eta, "eta")),
        "radial_hessian": str(point_value(hessian, "radial_hessian")),
        "r_minus_M": str(point_value(r_minus_m, "r_minus_M")),
        "(r_minus_M)^2": str(point_value(r_minus_m.square(), "(r_minus_M)^2")),
        "Delta": str(point_value(delta_value, "Delta")),
        "omega_h_interval": omega_h.to_json(),
        "superradiant_factor_interval": superradiant.to_json(),
        "trapped_scale_interval": trapped_scale.to_json(),
    }


def transition_certificate(
    left: tuple[str, Interval],
    right: tuple[str, Interval],
    spin: Fraction,
    digits: int,
) -> dict[str, Any]:
    left_name, left_interval = left
    right_name, right_interval = right
    if left_interval.hi != right_interval.lo:
        raise ValueError(
            f"cannot build transition certificate for non-contiguous charts "
            f"{left_name} and {right_name}: {left_interval.hi} != {right_interval.lo}"
        )
    return {
        "left": left_name,
        "right": right_name,
        "shared_endpoint": str(left_interval.hi),
        "endpoint_values": transition_endpoint_data(spin, left_interval.hi, digits),
    }


def transition_certificates(charts: list[dict[str, Any]], spin: Fraction, digits: int) -> list[dict[str, Any]]:
    intervals = [
        (str(chart["chart"]), parse_interval(chart["r_interval"]))
        for chart in charts
    ]
    intervals.sort(key=lambda item: (item[1].lo, item[1].hi, item[0]))
    return [
        transition_certificate(left, right, spin, digits)
        for left, right in zip(intervals, intervals[1:])
    ]


def horizon_radius_interval(a_value: Fraction, digits: int) -> Interval:
    if abs(a_value) >= 1:
        raise ValueError("spin must be strictly subextremal with M=1")
    sqrt_term = sqrt_fraction_interval(Fraction(1) - a_value * a_value, digits)
    return ONE + sqrt_term


def split_interval(interval: Interval, parts: int) -> list[Interval]:
    if parts < 1:
        raise ValueError("subdivision count must be at least 1")
    width = (interval.hi - interval.lo) / parts
    result = []
    start = interval.lo
    for index in range(parts):
        end = interval.hi if index == parts - 1 else start + width
        result.append(Interval(start, end))
        start = end
    return result


def generate_chart(spin: Fraction, chart: dict[str, Any], digits: int, r_override: Interval | None = None, suffix: str = "") -> dict[str, Any]:
    a = point(spin)
    r = r_override if r_override is not None else parse_interval(chart["r_interval"])
    if a.lo == 0:
        raise ValueError("the current spherical-orbit chart formula requires nonzero spin")
    if (r - ONE).lo <= 0 <= (r - ONE).hi:
        raise ValueError(f"chart {chart.get('chart', '<unnamed>')} crosses r=M")

    lam = lambda_interval(r, a)
    eta = eta_interval(r, a, lam)
    r_plus = horizon_radius_interval(spin, digits)
    omega_h = omega_h_interval(spin, digits)
    superradiant = ONE - omega_h * lam
    hessian = radial_hessian_interval(r, a, lam, eta)
    trapped_scale = Interval(Fraction(1), Fraction(1) + hessian.abs_upper())

    return {
        "chart": f"{chart.get('chart', f'r_{r.lo}_to_{r.hi}')}{suffix}",
        "r_interval": r.to_json(),
        "lambda_interval": lam.to_json(),
        "eta_interval": eta.to_json(),
        "omega_h_interval": omega_h.to_json(),
        "superradiant_factor_interval": superradiant.to_json(),
        "radial_hessian_interval": hessian.to_json(),
        "trapped_scale_interval": trapped_scale.to_json(),
        "denominator_certificates": denominator_certificates(r, a, r_plus),
        "notes": chart.get("notes", "candidate interval chart; coverage and positivity not proved"),
    }


def generate(data: dict[str, Any], digits: int = 12, subdivisions: int = 1) -> dict[str, Any]:
    spin = parse_fraction(data.get("spin", data.get("metadata", {}).get("spin", "0")))
    charts = data.get("radial_charts")
    if not isinstance(charts, list) or not charts:
        raise ValueError("radial_charts must be a nonempty list")
    generated = []
    for chart in charts:
        r = parse_interval(chart["r_interval"])
        pieces = split_interval(r, subdivisions)
        for index, piece in enumerate(pieces):
            suffix = "" if subdivisions == 1 else f"_sub{index + 1:02d}"
            generated.append(generate_chart(spin, chart, digits, r_override=piece, suffix=suffix))
    transitions = transition_certificates(generated, spin, digits)
    return {
        "metadata": {
            "name": data.get("name", data.get("metadata", {}).get("name", "candidate Carter atlas")),
            "spin": str(spin),
            "mass_normalization": "M=1",
            "status": "atlas data only; not a proof certificate",
            "sqrt_enclosure_decimal_digits": digits,
            "radial_subdivisions_per_input_chart": subdivisions,
            "transition_certificate_status": "adjacent endpoint compatibility only; not a positivity proof",
        },
        "charts": generated,
        "transitions": transitions,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sqrt-digits", type=int, default=12)
    parser.add_argument("--subdivide", type=int, default=1)
    args = parser.parse_args(argv[1:])

    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    try:
        result = generate(data, args.sqrt_digits, args.subdivide)
    except ValueError as exc:
        print(f"Carter atlas generation failed: {exc}", file=sys.stderr)
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
