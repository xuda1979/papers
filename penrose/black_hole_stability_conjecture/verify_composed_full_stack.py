#!/usr/bin/env python3
"""Verify the composed full stack: CWCC + CWQP-AB + RRBB + NBCT."""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_composed_cwcc_cwqp_rrbb as linear_triple
import verify_nonlinear_bootstrap_classification as nbct_mod


def verify_full_stack(
    cwcc_path: Path, cwqp_path: Path, rrbb_path: Path, nbct_path: Path
) -> dict[str, Any]:
    linear_result = linear_triple.verify_composed_triple(
        cwcc_path, cwqp_path, rrbb_path
    )
    nbct_result = nbct_mod.verify_file(nbct_path, rrbb_path=rrbb_path)

    nbct_b_lb = Fraction(nbct_result["linear_stack_reference"]["B_lower_bound"])
    rrbb_b_lb = Fraction(linear_result["rrbb_certificate"]["B_lower_bound"])
    if nbct_b_lb != rrbb_b_lb:
        raise ValueError(
            f"NBCT and RRBB B_lower_bound disagree: "
            f"NBCT={nbct_b_lb}, RRBB={rrbb_b_lb}"
        )

    b_nl = Fraction(nbct_result["nonlinear_budget"]["B_NL"])
    eps = Fraction(nbct_result["bootstrap_closure"]["epsilon_boot"])
    nonlinear_closure = b_nl * eps * eps <= rrbb_b_lb

    return {
        "accepted": bool(
            linear_result["accepted"]
            and nbct_result["accepted"]
            and nonlinear_closure
        ),
        "linear_stack": linear_result,
        "nbct_certificate": {
            "path": str(nbct_path),
            "accepted": nbct_result["accepted"],
            "B_NL": str(b_nl),
            "epsilon_boot": str(eps),
            "B_NL_eps_squared": str(b_nl * eps * eps),
        },
        "composed_nonlinear_closure": nonlinear_closure,
        "B_lower_bound": str(rrbb_b_lb),
        "warning": (
            "Full-stack arithmetic, structural-agreement, and nonlinear "
            "closure checks pass. This does NOT prove any of the four "
            "theorems (CWCC, CWQP-AB, RRBB, NBCT): the analytic content "
            "is not checked."
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cwcc_certificate", type=Path)
    parser.add_argument("cwqp_certificate", type=Path)
    parser.add_argument("rrbb_certificate", type=Path)
    parser.add_argument("nbct_certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_full_stack(
            args.cwcc_certificate, args.cwqp_certificate,
            args.rrbb_certificate, args.nbct_certificate,
        )
    except ValueError as exc:
        print(f"composed full-stack certificate rejected: {exc}",
              file=sys.stderr)
        return 1
    if not result["accepted"]:
        print("composed full-stack certificate rejected: nonlinear "
              "closure failed", file=sys.stderr)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
