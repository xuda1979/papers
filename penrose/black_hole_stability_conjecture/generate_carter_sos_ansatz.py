#!/usr/bin/env python3
"""Generate a first chart-level SOS-plus-transport ansatz scaffold.

The ansatz is intentionally symbolic. It records the variables, monomial basis,
unknown coefficients, and interval data for a chosen chart. A later solver or
human calculation must fill in actual coefficients and identities.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


def monomials(variables: list[str], degree: int) -> list[str]:
    if degree < 0:
        raise ValueError("degree must be nonnegative")
    result = ["1"]
    if degree == 0:
        return result
    result.extend(variables)
    if degree >= 2:
        for i, left in enumerate(variables):
            for right in variables[i:]:
                result.append(f"{left}*{right}")
    if degree > 2:
        raise ValueError("only degrees 0, 1, and 2 are currently scaffolded")
    return result


def find_chart(analysis: dict[str, Any], chart_name: str | None) -> dict[str, Any]:
    ranked = analysis.get("ranked_charts")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("analysis must contain ranked_charts")
    if chart_name is None:
        return ranked[0]
    for chart in ranked:
        if chart.get("chart") == chart_name:
            return chart
    raise ValueError(f"chart {chart_name!r} not found")


def generate(analysis: dict[str, Any], chart_name: str | None = None, degree: int = 2) -> dict[str, Any]:
    chart = find_chart(analysis, chart_name)
    variables = ["x", "ell", "q"]
    basis = monomials(variables, degree)
    source = chart["source"]

    unknowns = {
        "transport_symbol_b": [f"b_{index}" for index, _ in enumerate(basis)],
        "sos_gram_upper_triangular": [f"G_{i}_{j}" for i in range(len(basis)) for j in range(i, len(basis))],
        "constraint_multipliers": ["mu_R", "mu_dR", "mu_chart"],
        "superradiant_multiplier": "lambda_sr",
        "delta": "delta_chart",
    }

    return {
        "metadata": {
            "name": f"SOS ansatz for {chart['chart']}",
            "status": "symbolic ansatz only; coefficients not solved",
            "source_analysis_status": analysis.get("metadata", {}).get("analysis_status", "unknown"),
        },
        "chart": chart["chart"],
        "r_interval": source["r_interval"],
        "interval_data": {
            "lambda": source["lambda_interval"],
            "eta": source["eta_interval"],
            "superradiant_factor": source["superradiant_factor_interval"],
            "radial_hessian": source["radial_hessian_interval"],
            "trapped_scale": source["trapped_scale_interval"],
        },
        "variables": {
            "x": "affine coordinate on the radial chart, x in [-1,1]",
            "ell": "renormalized superradiant variable E - Omega_H L",
            "q": "renormalized Carter variable",
        },
        "monomial_basis": basis,
        "unknowns": unknowns,
        "target_identity": (
            "H_p b + ell^2 + rho_R^2 + rho_dR^2 - R - delta*q_grav "
            "= v^T G v + mu_R*rho_R^2 + mu_dR*rho_dR^2 + mu_chart*(1-x^2)"
        ),
        "required_inequalities": [
            "G is positive semidefinite with rational/interval certificate",
            "delta_chart > 0",
            "positive margin dominates remainder and quantization losses",
            "all denominator intervals avoid zero on this chart",
        ],
        "analysis_score": chart["score"],
        "analysis_flags": chart["flags"],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--chart")
    parser.add_argument("--degree", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv[1:])

    with args.analysis.open("r", encoding="utf-8") as handle:
        analysis = json.load(handle)
    try:
        result = generate(analysis, args.chart, args.degree)
    except ValueError as exc:
        print(f"Carter SOS ansatz generation failed: {exc}", file=sys.stderr)
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
