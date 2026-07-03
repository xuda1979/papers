#!/usr/bin/env python3
"""Verify the exact Schwarzschild photon-sphere degeneration chart.

The nonzero-spin Carter atlas uses formulas for lambda=L/E and eta=Q/E^2 that
divide by the Kerr spin a.  At a=0 the regular variable is instead the total
angular momentum ratio j=(L^2+Q)/E^2.  This checker proves the circular null
orbit identities at r=3M without using the a-denominators.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sympy as sp


def verify() -> dict[str, Any]:
    r, m, j = sp.symbols("r M j")
    delta = r**2 - 2 * m * r
    radial = r**4 - delta * j
    radial_derivative = sp.diff(radial, r)
    radial_hessian = sp.diff(radial, r, 2)

    photon_radius = 3 * m
    angular_ratio = 27 * m**2

    checks = {
        "R_at_photon_sphere": sp.factor(radial.subs({r: photon_radius, j: angular_ratio})),
        "dR_dr_at_photon_sphere": sp.factor(radial_derivative.subs({r: photon_radius, j: angular_ratio})),
        "impact_parameter_formula": sp.factor((angular_ratio - r**4 / delta).subs(r, photon_radius) * delta.subs(r, photon_radius)),
        "radial_hessian_at_photon_sphere": sp.factor(radial_hessian.subs({r: photon_radius, j: angular_ratio}) - 54 * m**2),
        "photon_radius_equation": sp.factor((2 * delta - r * (r - m)).subs(r, photon_radius)),
    }
    failed = {name: str(value) for name, value in checks.items() if sp.simplify(value) != 0}
    if failed:
        raise ValueError(f"Schwarzschild photon-sphere identity failed: {failed}")

    denominator_certificates = {
        "M": {
            "assumption": "M>0",
            "sign": "positive",
            "avoids_zero": True,
        },
        "r_minus_2M_at_photon_sphere": {
            "value": "M",
            "assumption": "M>0",
            "sign": "positive",
            "avoids_zero": True,
        },
        "Delta_at_photon_sphere": {
            "value": "3*M^2",
            "assumption": "M>0",
            "sign": "positive",
            "avoids_zero": True,
        },
    }

    return {
        "accepted": True,
        "chart_family": "zero_spin_schwarzschild_limit",
        "regular_variables": ["r", "j=(L^2+Q)/E^2"],
        "photon_radius": "3*M",
        "normalized_total_angular_momentum": "27*M^2",
        "radial_hessian": "54*M^2",
        "checked_identities": sorted(checks),
        "denominator_certificates": denominator_certificates,
        "status": "exact Schwarzschild seed chart; transition to nonzero spin not proved",
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
