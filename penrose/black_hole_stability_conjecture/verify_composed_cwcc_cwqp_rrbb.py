#!/usr/bin/env python3
"""Verify the composed CWCC + CWQP-AB + RRBB closure.

Given a CWCC certificate, a CWQP-AB certificate, and an RRBB certificate,
this verifier:
  1. independently verifies each certificate;
  2. checks that the CWCC and CWQP certificates agree on the chart names
     and bump constant;
  3. checks the cross-certificate inequality

        total_loss_v2(CWCC) <= B_lower_bound(RRBB)

     by exact rational arithmetic.

Acceptance means: all three certificates are internally consistent, the
chart data agrees across CWCC and CWQP, and the composition-aware
derivative-loss budget closes. This is the strongest form of the new
idea: the linearized trapped-superradiant sign obstruction is reduced
to a single rational-arithmetic inequality, with the calculus remainder
made explicit and absorbed into the loss budget.

The verifier does NOT check the analytic content of any of the three
theorems. It only checks the combinatorial, arithmetic, and
composed-closure content.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_carter_wick_composition as cwcc
import verify_quotient_positivity_assembly as cwqp
import verify_redshift_rp_buffer as rrbb


def _check_chart_name_agreement(
    cwcc_result: dict[str, Any], cwqp_result: dict[str, Any]
) -> None:
    """The CWCC and CWQP certificates must reference the same chart set."""
    cwcc_charts = {entry["name"] for entry in cwcc_result["composition_loss"]["per_chart"]}
    cwqp_charts = {entry["name"] for entry in cwqp_result["budget"]["per_chart"]}
    if cwcc_charts != cwqp_charts:
        raise ValueError(
            f"CWCC and CWQP chart name sets disagree: "
            f"CWCC={sorted(cwcc_charts)}, CWQP={sorted(cwqp_charts)}"
        )


def _check_bump_constant_agreement(
    cwcc_result: dict[str, Any], cwqp_result: dict[str, Any]
) -> None:
    cwcc_bump = Fraction(cwcc_result["bump_constant"])
    cwqp_bump = Fraction(cwqp_result["bump"]["bump_constant"])
    if cwcc_bump != cwqp_bump:
        raise ValueError(
            f"CWCC and CWQP bump constants disagree: "
            f"CWCC={cwcc_bump}, CWQP={cwqp_bump}"
        )


def verify_composed_triple(
    cwcc_path: Path, cwqp_path: Path, rrbb_path: Path
) -> dict[str, Any]:
    cwcc_result = cwcc.verify_file(cwcc_path)
    cwqp_result = cwqp.verify_file(cwqp_path)
    rrbb_result = rrbb.verify_file(rrbb_path)

    _check_chart_name_agreement(cwcc_result, cwqp_result)
    _check_bump_constant_agreement(cwcc_result, cwqp_result)

    total_loss_v2 = Fraction(cwcc_result["composition_loss"]["total_loss_v2"])
    b_lower_bound = Fraction(rrbb_result["buffer"]["B_lower_bound"])
    composed_closure = total_loss_v2 <= b_lower_bound

    # Also recover the v1 total loss from CWQP for comparison.
    total_loss_v1 = Fraction(cwqp_result["budget"]["total_loss"])

    return {
        "accepted": bool(
            cwcc_result["accepted"]
            and cwqp_result["accepted"]
            and rrbb_result["accepted"]
            and composed_closure
        ),
        "cwcc_certificate": {
            "path": str(cwcc_path),
            "accepted": cwcc_result["accepted"],
            "total_loss_v2": str(total_loss_v2),
        },
        "cwqp_certificate": {
            "path": str(cwqp_path),
            "accepted": cwqp_result["accepted"],
            "total_loss_v1": str(total_loss_v1),
        },
        "rrbb_certificate": {
            "path": str(rrbb_path),
            "accepted": rrbb_result["accepted"],
            "B_lower_bound": str(b_lower_bound),
        },
        "composed_closure": composed_closure,
        "margin": str(b_lower_bound - total_loss_v2),
        "v2_over_v1_excess": str(total_loss_v2 - total_loss_v1),
        "warning": (
            "Arithmetic, structural-agreement, and composed-closure checks "
            "pass. This does NOT prove any of the three theorems (CWCC, "
            "CWQP-AB, RRBB): the analytic content (diagonal symplectic form, "
            "Moyal expansion, Sobolev embedding, redshift calculation, r^p "
            "estimate, nontrapping of q_s, nonlinear bootstrap) is not "
            "checked."
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cwcc_certificate", type=Path)
    parser.add_argument("cwqp_certificate", type=Path)
    parser.add_argument("rrbb_certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_composed_triple(
            args.cwcc_certificate, args.cwqp_certificate, args.rrbb_certificate
        )
    except ValueError as exc:
        print(f"composed CWCC+CWQP+RRBB certificate rejected: {exc}",
              file=sys.stderr)
        return 1
    if not result["accepted"]:
        print("composed CWCC+CWQP+RRBB certificate rejected: composed "
              "closure failed", file=sys.stderr)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
