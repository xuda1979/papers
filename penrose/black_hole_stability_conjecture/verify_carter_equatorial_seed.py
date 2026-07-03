#!/usr/bin/env python3
"""Verify equatorial Q=0 Carter seed charts for the spin-half atlas.

The generic Carter atlas has an angular degeneration on the equatorial family
Q=0, equivalently eta=Q/E^2=0.  On the nonzero-spin radial chart this condition
is regular after clearing the displayed denominators: eta=0 is equivalent to

    r^3 - 6 M r^2 + 9 M^2 r - 4 M a^2 = 0.

For a/M=1/2 this cubic has two exterior roots, the prograde and retrograde
equatorial photon radii.  This checker proves the exact factorization and
verifies rational isolating intervals and denominator/Hessian bounds for both
roots.
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

    def is_negative(self) -> bool:
        return self.hi < 0

    def as_list(self) -> list[str]:
        return [str(self.lo), str(self.hi)]


def i(value: Fraction | int) -> Interval:
    return Interval.point(value)


def display(expr: sp.Expr) -> str:
    return str(sp.factor(expr)).replace("**", "^")


def rational(value: Fraction) -> sp.Rational:
    return sp.Rational(value.numerator, value.denominator)


def interval_bounds(root_interval: tuple[Fraction, Fraction]) -> dict[str, Any]:
    y = Interval(*root_interval)
    x = i(Fraction(1, 2))
    x_squared = i(Fraction(1, 4))
    r_minus_m = y - i(1)
    delta = y.pow(2) - y.scale(2) + x_squared

    lambda_numerator = y.pow(3) - y.pow(2).scale(3) + x_squared * y + x_squared
    lam = -(lambda_numerator / (x * r_minus_m))

    hessian_numerator = y.pow(2).scale(4) - y.scale(9) + x_squared.scale(4) + x_squared * y
    hessian = (hessian_numerator / r_minus_m.pow(2)).scale(24)

    checks = {
        "r_minus_M_positive": r_minus_m.is_positive(),
        "Delta_positive": delta.is_positive(),
        "radial_hessian_positive": hessian.is_positive(),
    }
    failed = [name for name, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"equatorial interval check failed: {failed}")

    return {
        "intervals": {
            "r_over_M": y.as_list(),
            "r_minus_M_over_M": r_minus_m.as_list(),
            "Delta_over_M_squared": delta.as_list(),
            "lambda": lam.as_list(),
            "eta": ["0", "0"],
            "radial_hessian_over_M_squared": hessian.as_list(),
        },
        "checked_interval_conditions": sorted(checks),
    }


def verify() -> dict[str, Any]:
    r, a, m = sp.symbols("r a M")
    delta = r**2 - 2 * m * r + a**2
    cubic = r**3 - 6 * m * r**2 + 9 * m**2 * r - 4 * m * a**2
    lam = (r**2 + a**2) / a - 2 * r * delta / (a * (r - m))
    eta = 4 * r**2 * delta / (r - m) ** 2 - (lam - a) ** 2
    p = r**2 + a**2 - a * lam
    b = (lam - a) ** 2
    radial = p**2 - delta * b
    radial_derivative = 4 * r * p - sp.diff(delta, r) * b
    radial_hessian = 4 * p + 8 * r**2 - 2 * b
    branch_hessian = 24 * m * (-9 * m**2 * r + 4 * m * a**2 + 4 * m * r**2 + a**2 * r) / (r - m) ** 2
    hessian_reduction_numerator = sp.together((radial_hessian - branch_hessian) * a**2 * (r - m) ** 2).as_numer_denom()[0]

    checks = {
        "eta_cleared_by_cubic": sp.factor(eta * a**2 * (r - m) ** 2 + r**3 * cubic),
        "R_cleared_by_cubic": sp.factor(radial * a**2 * (r - m) ** 2 + r**3 * delta * cubic),
        "dR_dr_cleared_by_cubic": sp.factor(radial_derivative * a**2 * (r - m) ** 2 + 2 * r**3 * (r - m) * cubic),
        "radial_hessian_reduces_mod_cubic": sp.factor(sp.rem(hessian_reduction_numerator, cubic, r)),
    }
    failed = {name: display(value) for name, value in checks.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"equatorial seed identity failed: {failed}")

    y = sp.symbols("y")
    cubic_half = y**3 - 6 * y**2 + 9 * y - 1
    cubic_derivative = sp.diff(cubic_half, y)
    branch_specs = [
        ("prograde", (Fraction(2347, 1000), Fraction(587, 250)), "decreasing"),
        ("retrograde", (Fraction(883, 250), Fraction(3533, 1000)), "increasing"),
    ]

    branches: dict[str, Any] = {}
    for name, root_interval, monotonicity in branch_specs:
        left = cubic_half.subs(y, rational(root_interval[0]))
        right = cubic_half.subs(y, rational(root_interval[1]))
        derivative_left = cubic_derivative.subs(y, rational(root_interval[0]))
        derivative_right = cubic_derivative.subs(y, rational(root_interval[1]))
        if monotonicity == "decreasing":
            isolated = left > 0 > right and derivative_left < 0 and derivative_right < 0
        else:
            isolated = left < 0 < right and derivative_left > 0 and derivative_right > 0
        if not isolated:
            raise ValueError(f"{name} equatorial root interval does not isolate one root")

        bounds = interval_bounds(root_interval)
        branches[name] = {
            "root_isolation": {
                "variable": "y=r/M",
                "interval": [str(root_interval[0]), str(root_interval[1])],
                "cubic_at_left": str(sp.factor(left)),
                "cubic_at_right": str(sp.factor(right)),
                "monotonicity": monotonicity,
            },
            **bounds,
        }

    return {
        "accepted": True,
        "chart_family": "equatorial_Q_equals_zero_seed",
        "spin": "a/M=1/2",
        "regular_constraint": "eta=Q/E^2=0",
        "equatorial_radius_equation": "r^3 - 6*M*r^2 + 9*M^2*r - 4*M*a^2 = 0",
        "checked_identities": sorted(checks),
        "branches": branches,
        "denominator_certificates": {
            name: {
                "r_minus_M": {
                    "value_interval_over_M": branch["intervals"]["r_minus_M_over_M"],
                    "assumption": "M>0",
                    "sign": "positive",
                    "avoids_zero": True,
                },
                "Delta": {
                    "value_interval_over_M_squared": branch["intervals"]["Delta_over_M_squared"],
                    "assumption": "M>0",
                    "sign": "positive",
                    "avoids_zero": True,
                },
                "a": {
                    "value_over_M": "1/2",
                    "sign": "positive",
                    "avoids_zero": True,
                },
            }
            for name, branch in branches.items()
        },
        "status": "equatorial seed charts; angular-torus neighborhoods and overlaps not proved",
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify()
    except (ValueError, ZeroDivisionError) as exc:
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
