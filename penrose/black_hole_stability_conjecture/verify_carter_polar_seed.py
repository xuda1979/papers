#!/usr/bin/env python3
"""Verify a polar L=0 Carter seed chart for the spin-half atlas.

The generic nonpolar Carter atlas uses angular torus coordinates that degenerate
on the polar family L=0.  This checker keeps the radial spherical-orbit
equations but replaces the generic angular chart by the regular polar constraint
lambda=L/E=0.

For lambda=0, the equation R=dR/dr=0 is equivalent, away from r=M, to the
regular cubic

    r^3 - 3 M r^2 + a^2 r + M a^2 = 0.

The checker proves this algebraically and verifies a rational isolating interval
for the exterior polar root when a/M=1/2.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class Interval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"invalid interval [{self.lo}, {self.hi}]")

    @classmethod
    def point(cls, value: Fraction | int) -> "Interval":
        value = Fraction(value)
        return cls(value, value)

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self) -> "Interval":
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other: "Interval") -> "Interval":
        return self + (-other)

    def __mul__(self, other: "Interval") -> "Interval":
        values = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(min(values), max(values))

    def reciprocal(self) -> "Interval":
        if self.lo <= 0 <= self.hi:
            raise ZeroDivisionError(f"interval contains zero: {self}")
        return Interval(min(1 / self.lo, 1 / self.hi), max(1 / self.lo, 1 / self.hi))

    def __truediv__(self, other: "Interval") -> "Interval":
        return self * other.reciprocal()

    def scale(self, value: Fraction | int) -> "Interval":
        return self * Interval.point(value)

    def pow(self, exponent: int) -> "Interval":
        if exponent < 0:
            return self.pow(-exponent).reciprocal()
        if exponent == 0:
            return Interval.point(1)
        if exponent % 2 == 0:
            values = [self.lo**exponent, self.hi**exponent]
            if self.lo <= 0 <= self.hi:
                values.append(Fraction(0))
            return Interval(min(values), max(values))
        return Interval(self.lo**exponent, self.hi**exponent)

    def is_positive(self) -> bool:
        return self.lo > 0

    def as_list(self) -> list[str]:
        return [str(self.lo), str(self.hi)]


def i(value: Fraction | int) -> Interval:
    return Interval.point(value)


def display(expr: sp.Expr) -> str:
    return str(sp.factor(expr)).replace("**", "^")


def verify() -> dict[str, Any]:
    r, a, m = sp.symbols("r a M")
    delta = r**2 - 2 * m * r + a**2
    p = r**2 + a**2
    cubic = r**3 - 3 * m * r**2 + a**2 * r + m * a**2
    eta = 2 * r * p / (r - m) - a**2
    b = eta + a**2
    radial = p**2 - delta * b
    radial_derivative = 4 * r * p - sp.diff(delta, r) * b
    radial_hessian = 4 * p + 8 * r**2 - 2 * b

    checks = {
        "lambda_zero_cubic": sp.factor((p * (r - m) - 2 * r * delta) + cubic),
        "R_cleared_by_cubic": sp.factor(radial * (r - m) + p * cubic),
        "dR_dr_cleared": sp.factor(radial_derivative * (r - m)),
        "B_formula_cleared": sp.factor((b - 2 * r * p / (r - m)) * (r - m)),
        "radial_hessian_formula": sp.factor(radial_hessian - 4 * (-m * a**2 - 3 * m * r**2 + 2 * r**3) / (r - m)),
    }
    failed = {name: display(value) for name, value in checks.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"polar seed identity failed: {failed}")

    y = sp.symbols("y")
    spin = Fraction(1, 2)
    root_interval = (Fraction(576643, 200000), Fraction(144161, 50000))
    cubic_half = y**3 - 3 * y**2 + sp.Rational(1, 4) * y + sp.Rational(1, 4)
    cubic_derivative = sp.diff(cubic_half, y)
    left_value = sp.factor(cubic_half.subs(y, sp.Rational(root_interval[0].numerator, root_interval[0].denominator)))
    right_value = sp.factor(cubic_half.subs(y, sp.Rational(root_interval[1].numerator, root_interval[1].denominator)))
    derivative_left = sp.factor(cubic_derivative.subs(y, sp.Rational(root_interval[0].numerator, root_interval[0].denominator)))
    derivative_right = sp.factor(cubic_derivative.subs(y, sp.Rational(root_interval[1].numerator, root_interval[1].denominator)))
    if not (left_value < 0 < right_value and derivative_left > 0 and derivative_right > 0):
        raise ValueError("spin-half polar root interval does not isolate the exterior root")

    y_interval_box = Interval(root_interval[0], root_interval[1])
    r_minus_m = y_interval_box - i(1)
    delta_half = y_interval_box.pow(2) - y_interval_box.scale(2) + i(Fraction(1, 4))
    eta_half = (y_interval_box * (y_interval_box.pow(2) + i(Fraction(1, 4)))).scale(2) / r_minus_m - i(Fraction(1, 4))
    hessian_numerator = y_interval_box.pow(3).scale(2) - y_interval_box.pow(2).scale(3) - i(Fraction(1, 4))
    hessian_half = (hessian_numerator / r_minus_m).scale(4)

    interval_checks = {
        "r_minus_M_positive": r_minus_m.is_positive(),
        "Delta_positive": delta_half.is_positive(),
        "eta_positive": eta_half.is_positive(),
        "radial_hessian_positive": hessian_half.is_positive(),
    }
    failed_interval_checks = [name for name, accepted in interval_checks.items() if not accepted]
    if failed_interval_checks:
        raise ValueError(f"polar seed interval check failed: {failed_interval_checks}")

    y_interval = y_interval_box.as_list()
    interval_bounds = {
        "r_over_M": y_interval,
        "r_minus_M_over_M": r_minus_m.as_list(),
        "Delta_over_M_squared": delta_half.as_list(),
        "eta_over_M_squared": eta_half.as_list(),
        "radial_hessian_over_M_squared": hessian_half.as_list(),
    }

    return {
        "accepted": True,
        "chart_family": "polar_L_equals_zero_seed",
        "spin": "a/M=1/2",
        "regular_constraint": "lambda=L/E=0",
        "polar_radius_equation": "r^3 - 3*M*r^2 + a^2*r + M*a^2 = 0",
        "eta_formula": "2*r*(r^2+a^2)/(r-M) - a^2",
        "checked_identities": sorted(checks),
        "checked_interval_conditions": sorted(interval_checks),
        "root_isolation": {
            "variable": "y=r/M",
            "interval": y_interval,
            "cubic_at_left": str(left_value),
            "cubic_at_right": str(right_value),
            "derivative_positive_on_interval": True,
        },
        "interval_bounds": interval_bounds,
        "denominator_certificates": {
            "r_minus_M": {
                "value_interval_over_M": interval_bounds["r_minus_M_over_M"],
                "assumption": "M>0",
                "sign": "positive",
                "avoids_zero": True,
            },
            "Delta": {
                "value_interval_over_M_squared": interval_bounds["Delta_over_M_squared"],
                "assumption": "M>0",
                "sign": "positive",
                "avoids_zero": True,
            },
        },
        "status": "polar seed chart; angular-torus neighborhood and overlaps not proved",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
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
