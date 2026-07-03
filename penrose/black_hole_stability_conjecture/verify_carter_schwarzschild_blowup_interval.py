#!/usr/bin/env python3
"""Verify a rational small-spin interval box for the Schwarzschild blow-up chart.

This is the first intervalized companion to the pointwise blow-up transition.
Use dimensionless variables

    x=a/M,  sigma=(r-3M)/a,  z=x*sigma,  y=r/M=3+z.

On the rational box |x|<=1/64, |sigma|<=1, the checker encloses the regularized
Carter invariants j/M^2, lambda/M, eta/M^2, Delta/M^2, and the normalized radial
Hessian using only Fraction interval arithmetic.  It proves that the old
division by a has been replaced by a regular small-spin neighborhood with
r-M and Delta uniformly away from zero.
"""

from __future__ import annotations

import argparse
import json
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

    def contains(self, value: Fraction | int) -> bool:
        value = Fraction(value)
        return self.lo <= value <= self.hi

    def is_positive(self) -> bool:
        return self.lo > 0

    def as_list(self) -> list[str]:
        return [str(self.lo), str(self.hi)]


def i(value: Fraction | int) -> Interval:
    return Interval.point(value)


def verify() -> dict[str, Any]:
    x = Interval(-Fraction(1, 64), Fraction(1, 64))
    sigma = Interval(-1, 1)
    x_squared = Interval(0, Fraction(1, 64) ** 2)
    z = x * sigma
    y = i(3) + z
    r_minus_m = i(2) + z
    r_minus_m_squared = r_minus_m.pow(2)

    delta = y.pow(2) - y.scale(2) + x_squared

    z_squared = z.pow(2)
    z_cubed = z.pow(3)
    z_fourth = z.pow(4)

    j_numerator = (
        i(108)
        + z.scale(180)
        + z_squared.scale(102)
        + x_squared.scale(16)
        + z_cubed.scale(24)
        + (x_squared * z).scale(8)
        + z_fourth.scale(2)
        + x_squared * z_squared
    )
    normalized_j = j_numerator / r_minus_m_squared

    hessian_middle = i(9) + z.scale(12) + z_squared.scale(6) - x_squared + z_cubed
    normalized_hessian = (y * hessian_middle / r_minus_m_squared).scale(8)

    sigma_squared = sigma.pow(2)
    sigma_cubed = sigma.pow(3)
    lambda_numerator = (
        sigma.scale(9)
        + (x * sigma_squared).scale(6)
        + x.scale(4)
        + x_squared * sigma_cubed
        + x_squared * sigma
    )
    normalized_lambda = -(lambda_numerator / r_minus_m)

    eta_factor = sigma_squared.scale(3) - i(4) + x * sigma_cubed
    normalized_eta = -(y.pow(3) * eta_factor / r_minus_m_squared)

    checks = {
        "r_minus_M_positive": r_minus_m.is_positive(),
        "Delta_positive": delta.is_positive(),
        "j_contains_schwarzschild_value": normalized_j.contains(27),
        "radial_hessian_contains_schwarzschild_value": normalized_hessian.contains(54),
        "lambda_interval_finite": normalized_lambda.lo > -5 and normalized_lambda.hi < 5,
        "eta_interval_positive": normalized_eta.is_positive(),
    }
    failed = [name for name, accepted in checks.items() if not accepted]
    if failed:
        raise ValueError(f"Schwarzschild blow-up interval check failed: {failed}")

    intervals = {
        "a_over_M": x.as_list(),
        "sigma": sigma.as_list(),
        "z": z.as_list(),
        "r_over_M": y.as_list(),
        "r_minus_M_over_M": r_minus_m.as_list(),
        "Delta_over_M_squared": delta.as_list(),
        "j_over_M_squared": normalized_j.as_list(),
        "radial_hessian_over_M_squared": normalized_hessian.as_list(),
        "lambda_over_M": normalized_lambda.as_list(),
        "eta_over_M_squared": normalized_eta.as_list(),
    }

    return {
        "accepted": True,
        "chart_family": "zero_spin_schwarzschild_blowup_interval",
        "parent_transition": "zero_spin_schwarzschild_transition_blowup",
        "box": {
            "abs_a_over_M_max": "1/64",
            "abs_sigma_max": "1",
            "radius_formula": "r/M=3+(a/M)*sigma",
        },
        "regular_variables": ["x=a/M", "sigma=(r-3M)/a", "j=lambda^2+eta"],
        "intervals": intervals,
        "checked_conditions": sorted(checks),
        "status": "rational interval blow-up seed; global small-spin atlas overlaps not proved",
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
