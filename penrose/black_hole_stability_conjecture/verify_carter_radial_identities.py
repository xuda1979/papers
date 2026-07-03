#!/usr/bin/env python3
"""Verify exact Carter radial identities for spherical null orbits.

The checker proves, by symbolic polynomial reduction, that the displayed
lambda(r) and eta(r) formulas make the radial Carter polynomial R and its
r-derivative vanish after clearing denominators. It also verifies the radial
Hessian formula used for the trapped nondegeneracy scale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp


def verify() -> dict[str, object]:
    r, a, m = sp.symbols("r a m")
    lam_symbol, eta_symbol = sp.symbols("lambda eta")
    delta = r**2 - 2 * m * r + a**2
    lam = (r**2 + a**2) / a - (2 * r * delta) / (a * (r - m))
    eta = (4 * r**2 * delta) / (r - m) ** 2 - (lam - a) ** 2
    p = r**2 + a**2 - a * lam
    b = eta + (lam - a) ** 2
    radial = p**2 - delta * b
    radial_derivative = 4 * r * p - sp.diff(delta, r) * b
    symbolic_p = r**2 + a**2 - a * lam_symbol
    symbolic_b = eta_symbol + (lam_symbol - a) ** 2
    symbolic_radial = symbolic_p**2 - delta * symbolic_b
    radial_hessian = sp.diff(symbolic_radial, r, 2)
    radial_hessian_formula = 4 * symbolic_p + 8 * r**2 - 2 * symbolic_b

    checks = {
        "R_cleared": sp.factor(radial * (r - m) ** 2),
        "dR_dr_cleared": sp.factor(radial_derivative * (r - m) ** 2),
        "P_formula_cleared": sp.factor((p - 2 * r * delta / (r - m)) * (r - m)),
        "B_formula_cleared": sp.factor((b - 4 * r**2 * delta / (r - m) ** 2) * (r - m) ** 2),
        "radial_hessian_formula": sp.factor(radial_hessian - radial_hessian_formula),
    }
    failed = {name: str(value) for name, value in checks.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"Carter radial identity failed: {failed}")
    return {
        "accepted": True,
        "checked_identities": sorted(checks),
        "generic_denominators": ["a", "r-M"],
    }


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print("usage: verify_carter_radial_identities.py", file=sys.stderr)
        return 2
    try:
        result = verify()
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
