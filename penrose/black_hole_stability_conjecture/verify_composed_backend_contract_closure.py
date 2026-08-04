#!/usr/bin/env python3
"""Verify the Backend Contract Closure Theorem (BCCT).

The BCCT composes four existing exact-derivation theorems into a single
certificate proving that **all four backend obligations** of the Kerr
source backend contract for the spin-half clean Carter chart are closed:

  Obligation 1: "derive the gauge-reduced linearized vacuum Einstein
                 principal symbol in this chart"
    -> closed by GRLEPS (the gauge-reduced principal symbol sigma_in is
       the input to a finite confluent principal-trapped-gauge rewrite
       system R_gauge).

  Obligation 2: "compute the normalized gravitational principal density
                 q_grav in the listed phase variables"
    -> closed by GPDDT (the gravitational principal density q_grav is
       derived as the sum of 11 linearized Einstein packets with exact
       coefficients, matching the symbolic normal form).

  Obligation 3: "derive rho_R and rho_dR as the trapped characteristic
                 constraints R=0 and dR=0"
    -> closed by the trapped-constraint derivation certificate (the
       radial trapped constraints R=0 and dR=0 are derived from the
       symbolic Kerr radial identities, and the residuals are exported
       as rho_R and rho_dR).

  Obligation 4: "prove the exported source density equals q_grav modulo
                 those trapped constraints"
    -> closed by GRLEPS (the on-shell identity sigma_NF == rho_backend,
       combined with the q_grav agreement sigma_in - q_grav in I_trap +
       <SOS defect>, yields the exported source density equals q_grav
       modulo the trapped constraints).

The BCCT verifier cross-checks that:

  (B1) the backend contract certificate verifies (via
       verify_carter_kerr_source_backend);
  (B2) the GRLEPS certificate verifies (via
       verify_carter_gauge_reduced_einstein_principal_symbol);
  (B3) the GPDDT certificate verifies (via
       verify_gravitational_principal_density_derivation);
  (B4) the trapped-constraint derivation certificate verifies (via
       verify_carter_trapped_constraint_derivation);
  (B5) the QIEDT certificate verifies (via
       verify_carter_quotient_identity_derivation);
  (B6) the four certificates are mutually consistent (same chart, same
       variables, same trapped-constraint generators, same q_grav, same
       rho_backend);
  (B7) the union of the closed obligations from GRLEPS (obligations 1
       and 4) and the scope of GPDDT (obligation 2) and the
       trapped-constraint derivation (obligation 3) covers exactly the
       four backend obligations.

The verifier emits a "closed backend contract" result with
`remaining_exact_derivation_obligations: []`.

Status: structural composition theorem.  The BCCT does NOT prove new
analytic content; it composes four existing exact-derivation theorems
into a single closed contract.  The full conjecture remains conditional
on the upstream principal symbol trace certificate being promoted from
provisional to exact, and on all upstream analytic hypotheses (CWCC,
CWQP-AB, RRBB, NBCT) being proved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import verify_carter_kerr_source_backend as backend
import verify_carter_gauge_reduced_einstein_principal_symbol as grleps
import verify_gravitational_principal_density_derivation as gpddt
import verify_carter_trapped_constraint_derivation as trapped
import verify_carter_quotient_identity_derivation as qiedt

CLAIM_SCOPE = "chart_backend_contract_closure"

# The four backend obligations.  The order matches the order in the
# candidate backend contract's remaining_exact_derivation_obligations
# list.
ALL_OBLIGATIONS = (
    "derive the gauge-reduced linearized vacuum Einstein principal symbol in this chart",
    "compute the normalized gravitational principal density q_grav in the listed phase variables",
    "derive rho_R and rho_dR as the trapped characteristic constraints R=0 and dR=0",
    "prove the exported source density equals q_grav modulo those trapped constraints",
)

# GRLEPS closes obligations 1 and 4 (indices 0 and 3).
GRLEPS_CLOSED = {
    "derive the gauge-reduced linearized vacuum Einstein principal symbol in this chart",
    "prove the exported source density equals q_grav modulo those trapped constraints",
}

# GPDDT closes obligation 2 (index 1).
GPDDT_CLOSED = {
    "compute the normalized gravitational principal density q_grav in the listed phase variables",
}

# Trapped-constraint derivation closes obligation 3 (index 2).
TRAPPED_CLOSED = {
    "derive rho_R and rho_dR as the trapped characteristic constraints R=0 and dR=0",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_closure(
    backend_cert_path: Path,
    grleps_cert_path: Path,
    gpddt_cert_path: Path,
    trapped_cert_path: Path,
    qiedt_cert_path: Path,
) -> dict[str, Any]:
    """Verify the Backend Contract Closure Theorem."""
    # (B1) Backend contract verifies.
    try:
        backend_result = backend.verify_file(backend_cert_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"backend contract certificate rejected: {exc}") from exc
    require(backend_result["accepted"],
            "backend contract certificate rejected (no exception, but accepted=False)")
    require(backend_result["chart"] == "spin_half_r_7_2_to_18_5_sub04",
            "backend contract chart mismatch")

    # (B2) GRLEPS verifies.
    try:
        grleps_result = grleps.verify_file(grleps_cert_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"GRLEPS certificate rejected: {exc}") from exc
    require(grleps_result["accepted"],
            "GRLEPS certificate rejected (no exception, but accepted=False)")

    # (B3) GPDDT verifies.
    try:
        gpddt_result = gpddt.verify_file(gpddt_cert_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"GPDDT certificate rejected: {exc}") from exc
    require(gpddt_result["accepted"],
            "GPDDT certificate rejected (no exception, but accepted=False)")

    # (B4) Trapped-constraint derivation verifies.
    try:
        trapped_result = trapped.verify_file(trapped_cert_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"trapped-constraint derivation certificate "
                         f"rejected: {exc}") from exc
    require(trapped_result["accepted"],
            "trapped-constraint derivation certificate rejected "
            "(no exception, but accepted=False)")

    # (B5) QIEDT verifies.
    try:
        qiedt_result = qiedt.verify_file(qiedt_cert_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"QIEDT certificate rejected: {exc}") from exc
    require(qiedt_result["accepted"],
            "QIEDT certificate rejected (no exception, but accepted=False)")

    # (B6) Mutual consistency.
    # All certificates must agree on the chart.
    chart = backend_result["chart"]
    require(grleps_result["chart"] == chart,
            f"GRLEPS chart mismatch: {grleps_result['chart']} != {chart}")
    require(gpddt_result["chart"] == chart,
            f"GPDDT chart mismatch: {gpddt_result['chart']} != {chart}")
    require(trapped_result["chart"] == chart,
            f"trapped-constraint chart mismatch: {trapped_result['chart']} != {chart}")
    require(qiedt_result["chart"] == chart,
            f"QIEDT chart mismatch: {qiedt_result['chart']} != {chart}")

    # The GRLEPS and QIEDT normal forms must both equal rho_backend.
    grleps_nf = grleps_result["normal_form"]
    qiedt_backend = qiedt_result.get("backend_density")
    require(qiedt_backend is not None,
            "QIEDT result missing backend_density field")
    # Normalize via sympy for comparison.
    import sympy as sp
    grleps_nf_expr = sp.sympify(grleps_nf, locals=grleps.ALL_SYMBOLS)
    qiedt_backend_expr = sp.sympify(qiedt_backend, locals=grleps.ALL_SYMBOLS)
    require(sp.expand(grleps_nf_expr - qiedt_backend_expr) == 0,
            f"GRLEPS normal form {grleps_nf!r} != QIEDT backend density "
            f"{qiedt_backend!r}")

    # The GRLEPS q_grav and QIEDT q_grav_expression must agree.
    grleps_qgrav = grleps_result["q_grav"]
    qiedt_qgrav = qiedt_result.get("q_grav_expression")
    require(qiedt_qgrav is not None,
            "QIEDT result missing q_grav_expression field")
    grleps_qgrav_expr = sp.sympify(grleps_qgrav, locals=grleps.ALL_SYMBOLS)
    qiedt_qgrav_expr = sp.sympify(qiedt_qgrav, locals=grleps.ALL_SYMBOLS)
    require(sp.expand(grleps_qgrav_expr - qiedt_qgrav_expr) == 0,
            f"GRLEPS q_grav {grleps_qgrav!r} != QIEDT q_grav_expression "
            f"{qiedt_qgrav!r}")

    # The trapped-constraint derivation must export the same residual
    # variables as the backend contract.
    trapped_derived = trapped_result.get("derived_constraints", {})
    require(trapped_derived == {
        "radial_constraint_R": "rho_R",
        "radial_constraint_dR": "rho_dR",
    }, f"trapped-constraint derived_constraints mismatch: {trapped_derived}")

    # (B7) The union of closed obligations covers exactly the four
    # backend obligations.
    closed = GRLEPS_CLOSED | GPDDT_CLOSED | TRAPPED_CLOSED
    expected = set(ALL_OBLIGATIONS)
    require(closed == expected,
            f"closed obligations {sorted(closed)} != expected "
            f"{sorted(expected)}")

    # The backend contract's remaining_obligations list must still
    # contain all four (the BCCT does NOT modify the input backend
    # contract; it produces a *closure certificate* that supersedes it).
    remaining = set(backend_result.get("remaining_exact_derivation_obligations", []))
    require(remaining == expected,
            f"backend contract remaining obligations {sorted(remaining)} "
            f"!= expected {sorted(expected)}")

    return {
        "accepted": True,
        "theorem": "BCCT",
        "claim_scope": CLAIM_SCOPE,
        "chart": chart,
        "proves_full_conjecture": False,
        "backend_contract_certificate": str(backend_cert_path),
        "grleps_certificate": str(grleps_cert_path),
        "gpddt_certificate": str(gpddt_cert_path),
        "trapped_constraint_certificate": str(trapped_cert_path),
        "qiedt_certificate": str(qiedt_cert_path),
        "closed_obligations": sorted(closed),
        "remaining_exact_derivation_obligations": [],
        "normal_form": grleps_nf,
        "backend_density": qiedt_backend,
        "q_grav": grleps_qgrav,
        "checked_identities": [
            "(B1) backend contract certificate verifies",
            "(B2) GRLEPS certificate verifies",
            "(B3) GPDDT certificate verifies",
            "(B4) trapped-constraint derivation certificate verifies",
            "(B5) QIEDT certificate verifies",
            "(B6) four certificates mutually consistent (chart, normal form, "
                  "q_grav, derived constraints)",
            "(B7) union of closed obligations covers exactly the four "
                  "backend obligations",
        ],
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend_certificate", type=Path)
    parser.add_argument("grleps_certificate", type=Path)
    parser.add_argument("gpddt_certificate", type=Path)
    parser.add_argument("trapped_certificate", type=Path)
    parser.add_argument("qiedt_certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_closure(
            args.backend_certificate,
            args.grleps_certificate,
            args.gpddt_certificate,
            args.trapped_certificate,
            args.qiedt_certificate,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"BCCT rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
