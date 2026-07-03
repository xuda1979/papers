#!/usr/bin/env python3
"""Verify the Schwarzschild blow-up transition for Carter spherical data.

The nonzero-spin formulas for lambda=L/E and eta=Q/E^2 divide by a.  Near
a=0, use the blow-up coordinate sigma=(r-3M)/a and the regular invariant
j=lambda^2+eta.  This checker proves that the pole in j cancels exactly and
that the blown-up Kerr trapped data limits to the Schwarzschild photon-sphere
seed r=3M, j=27M^2 with radial Hessian 54M^2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sympy as sp


def display(expr: sp.Expr) -> str:
    return str(sp.factor(expr)).replace("**", "^")


def verify() -> dict[str, Any]:
    r, a, m, sigma = sp.symbols("r a M sigma")
    delta = r**2 - 2 * m * r + a**2
    lam = (r**2 + a**2) / a - (2 * r * delta) / (a * (r - m))
    eta = 4 * r**2 * delta / (r - m) ** 2 - (lam - a) ** 2

    j = sp.cancel(lam**2 + eta)
    regular_j = sp.cancel(4 * r**2 * delta / (r - m) ** 2 + 2 * a * lam - a**2)
    j_numerator, j_denominator = sp.together(j).as_numer_denom()

    blowup_radius = 3 * m + a * sigma
    blowup_j = sp.cancel(j.subs(r, blowup_radius))
    lambda_limit = sp.limit(lam.subs(r, blowup_radius), a, 0)
    eta_limit = sp.limit(eta.subs(r, blowup_radius), a, 0)
    j_limit = sp.limit(blowup_j, a, 0)

    p = r**2 + a**2 - a * lam
    b = eta + (lam - a) ** 2
    radial_hessian = sp.cancel(4 * p + 8 * r**2 - 2 * b)
    hessian_limit = sp.limit(radial_hessian.subs(r, blowup_radius), a, 0)

    checks = {
        "j_pole_cancellation": sp.factor(j - regular_j),
        "j_has_no_a_denominator_after_cancellation": sp.factor(j_denominator - (r - m) ** 2),
        "lambda_blowup_limit": sp.factor(lambda_limit + sp.Rational(9, 2) * m * sigma),
        "eta_blowup_limit": sp.factor(eta_limit - (27 * m**2 - sp.Rational(81, 4) * m**2 * sigma**2)),
        "j_blowup_limit": sp.factor(j_limit - 27 * m**2),
        "lambda_eta_reconstruct_j_limit": sp.factor(lambda_limit**2 + eta_limit - 27 * m**2),
        "radial_hessian_blowup_limit": sp.factor(hessian_limit - 54 * m**2),
        "r_minus_M_denominator_limit": sp.factor(sp.limit((blowup_radius - m) - 2 * m, a, 0)),
        "Delta_denominator_limit": sp.factor(sp.limit(delta.subs(r, blowup_radius) - 3 * m**2, a, 0)),
    }
    failed = {name: display(value) for name, value in checks.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"Schwarzschild transition identity failed: {failed}")

    return {
        "accepted": True,
        "chart_family": "zero_spin_schwarzschild_transition_blowup",
        "blowup_coordinates": ["sigma=(r-3*M)/a", "j=lambda^2+eta"],
        "matched_seed_chart": "zero_spin_schwarzschild_limit",
        "regular_j_formula": display(j),
        "remaining_j_denominator": "(r-M)^2",
        "j_limit": "27*M^2",
        "radial_hessian_limit": "54*M^2",
        "limiting_old_variables": {
            "lambda": "-9*M*sigma/2",
            "eta": "27*M^2 - 81*M^2*sigma^2/4",
        },
        "checked_identities": sorted(checks),
        "denominator_certificates": {
            "M": {
                "assumption": "M>0",
                "sign": "positive",
                "avoids_zero": True,
            },
            "r_minus_M_limit": {
                "value": "2*M",
                "assumption": "M>0",
                "sign": "positive",
                "avoids_zero": True,
            },
            "Delta_limit": {
                "value": "3*M^2",
                "assumption": "M>0",
                "sign": "positive",
                "avoids_zero": True,
            },
            "old_a_denominator": {
                "limit": "0",
                "old_chart_valid_at_zero_spin": False,
                "regularized_by": "j=lambda^2+eta with sigma=(r-3*M)/a",
            },
        },
        "status": "blow-up transition seed; rational interval neighborhood and full overlap policy not proved",
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
