#!/usr/bin/env python3
"""Verify the composed CWQP-AB + RRBB closure.

Given a CWQP-AB certificate and an RRBB certificate, this verifier:
  1. independently verifies the CWQP-AB certificate (via
     verify_quotient_positivity_assembly.verify_file);
  2. independently verifies the RRBB certificate (via
     verify_redshift_rp_buffer.verify_file);
  3. checks the cross-certificate inequality

        total_loss(CWQP) <= B_lower_bound(RRBB)

     by exact rational arithmetic.

Acceptance means: both certificates are internally consistent AND the
composed derivative-loss budget closes. This is the strongest form of the
new idea: the linearized trapped-superradiant sign obstruction is reduced
to a single rational-arithmetic inequality, checkable by exact computation.

The verifier does NOT check the analytic content of either theorem
(Carter–Wick symbol calculus, redshift calculation, r^p estimate,
nontrapping of q_s on the bulk, nonlinear bootstrap). It only checks the
combinatorial, arithmetic, and composed-closure content.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import verify_quotient_positivity_assembly as cwqp
import verify_redshift_rp_buffer as rrbb


def verify_composed(cwqp_path: Path, rrbb_path: Path) -> dict[str, Any]:
    cwqp_result = cwqp.verify_file(cwqp_path)
    rrbb_result = rrbb.verify_file(rrbb_path)

    total_loss = Fraction(cwqp_result["budget"]["total_loss"])
    b_lower_bound = Fraction(rrbb_result["buffer"]["B_lower_bound"])

    composed_closure = total_loss <= b_lower_bound

    return {
        "accepted": bool(cwqp_result["accepted"] and rrbb_result["accepted"] and composed_closure),
        "cwqp_certificate": {
            "path": str(cwqp_path),
            "accepted": cwqp_result["accepted"],
            "total_loss": str(total_loss),
        },
        "rrbb_certificate": {
            "path": str(rrbb_path),
            "accepted": rrbb_result["accepted"],
            "B_lower_bound": str(b_lower_bound),
        },
        "composed_closure": composed_closure,
        "margin": str(b_lower_bound - total_loss),
        "warning": (
            "Combinatorial, arithmetic, and composed-closure checks pass. "
            "This does NOT prove the Carter–Wick Quotient Positivity Theorem "
            "or the Redshift–r^p Buffer Theorem: the analytic content "
            "(symbol calculus, redshift calculation, r^p estimate, "
            "nontrapping of q_s, nonlinear bootstrap) is not checked."
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cwqp_certificate", type=Path)
    parser.add_argument("rrbb_certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_composed(args.cwqp_certificate, args.rrbb_certificate)
    except ValueError as exc:
        print(f"composed CWQP+RRBB certificate rejected: {exc}", file=sys.stderr)
        return 1
    if not result["accepted"]:
        print(f"composed CWQP+RRBB certificate rejected: composed closure failed",
              file=sys.stderr)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
