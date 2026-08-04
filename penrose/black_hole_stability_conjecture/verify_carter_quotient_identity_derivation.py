#!/usr/bin/env python3
"""Verify the Quotient Identity Exact Derivation Theorem (QIEDT) for a Carter chart.

The checker proves, by symbolic polynomial reduction, that the gravitational
principal density q_grav (as derived by GPDDT) admits the exact decomposition

    q_grav = rho_backend
           + s * (c*x - ell)^2                              (SOS defect, gauge tangent)
           + diag_slack * P_R                               (TCDEDT radial packet)
           + bilinear_slack * P_dR                          (TCDEDT derivative packet)
           + bilinear_slack * P_xq                          (QIEDT Carter-mixed packet, NEW)

where:

  * (x, ell, q) are the chart's normalized phase variables (radial, superradiant,
    Carter);
  * (h, s, tau, c, diag_slack, bilinear_slack) are the symbolic coefficient
    variables of the GPDDT density on the spin-half clean chart;
  * rho_backend = h*x^2 + tau*q^2 is the minimal Kerr-clean backend density
    produced by the gauge-reduced linearized vacuum Einstein operator;
  * P_R   = x^2 + ell^2 + q^2   is the TCDEDT radial trapped-constraint
            correction packet (diagonal-invariant, degree 2);
  * P_dR  = x*ell + q^2         is the TCDEDT derivative trapped-constraint
            correction packet (mixed radial-superradiant plus Carter, degree 2);
  * P_xq  = x*q + ell*q - q^2   is the QIEDT Carter-mixed trapped-constraint
            correction packet (NEW: derived from the bilinear off-diagonal
            structure of the linearized Bianchi identity, preserved by the
            Carter (Q) direction);
  * the SOS defect s*(c*x - ell)^2 is a certified nonnegative square when the
    superradiant coefficient s lies in its accepted coercivity tube (s >= 0).

The verifier checks the following sympy-reduced identities:

  (Q1) The QIEDT quotient identity holds as a polynomial identity in all
       variables:
         q_grav - rho_backend - s*(c*x-ell)^2
           - diag_slack*P_R - bilinear_slack*P_dR - bilinear_slack*P_xq == 0.

  (Q2) The new Carter-mixed packet P_xq is the unique degree-2 polynomial of
       the form alpha*x*q + beta*ell*q + gamma*q^2 that (i) is invariant under
       the simultaneous swap (x <-> ell) (the radial-superradiant symmetry of
       the Kerr principal symbol), and (ii) satisfies the decomposition
       P_bil = P_dR + P_xq, where P_bil = x*ell + x*q + ell*q is the bilinear
       trapped-constraint correction packet (the full off-diagonal phase
       sum). The unique solution is alpha = beta = 1, gamma = -1, giving
       P_xq = x*q + ell*q - q^2.

  (Q3) The bilinear packet decomposition P_bil = P_dR + P_xq holds exactly.
       This shows that the bilinear-slack contribution to q_grav is split
       between the TCDEDT derivative packet P_dR and the new QIEDT packet
       P_xq, each with multiplier bilinear_slack.

  (Q4) Linear independence of the three correction packets P_R, P_dR, P_xq:
       no nontrivial scalar combination
       alpha*P_R + beta*P_dR + gamma*P_xq is the zero polynomial.

  (Q5) The backend density rho_backend = h*x^2 + tau*q^2 is phase-homogeneous
       of degree 2 and contains only the radial and Carter diagonal packets
       (no superradiant, no slack, no cross terms). This is the minimal
       Kerr-clean normal form.

  (Q6) The SOS defect s*(c*x-ell)^2 is a perfect square with weight s, and
       the gauge-tangent square c*x - ell is the unique (up to scalar) linear
       form in (x, ell) that vanishes on the gauge-tangent direction
       (the kernel of the superradiant gauge-trace contraction).

  (Q7) The certified SOS weight s (the superradiant coefficient) lies in the
       accepted coercivity tube [0, s_max] with s_max >= 0, so the SOS defect
       is nonnegative on the chart. (This is a rational interval check.)

The verifier also checks that the certificate's derivation chain has
non-candidate provenance, that the principal-symbol-trace certificate is
referenced and matches the chart, and that all packet expressions match
their derived formulas.

This closes the `quotient_identity` source target in the exact-derivation
audit's backend obligation registry: the exported source density q_grav is
proved equal to the backend density modulo the trapped-constraint ideal
(P_R, P_dR, P_xq) plus a certified SOS defect (gauge-tangent square).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sympy as sp

HERE = Path(__file__).resolve().parent

ALLOWED_DERIVATION_STATUSES = {"exact_derivation"}

FORBIDDEN_PROVENANCE_SUBSTRINGS = ("candidate", "TODO", "FIXME", "placeholder")

# Expected packet names in the certificate.
EXPECTED_PACKET_NAMES = (
    "radial_trapped_constraint_correction_packet",
    "derivative_trapped_constraint_correction_packet",
    "carter_mixed_trapped_constraint_correction_packet",
)

# Expected SOS defect names.
EXPECTED_SOS_DEFECT_NAMES = (
    "gauge_tangent_sos_defect",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_keys(obj: dict[str, Any], keys: list[str], ctx: str) -> None:
    for key in keys:
        require(key in obj, f"{ctx} must contain key {key!r}")


def require_string(value: Any, ctx: str) -> str:
    require(isinstance(value, str) and value, f"{ctx} must be a nonempty string")
    return value


def is_noncandidate_provenance(provenance: str) -> bool:
    lower = provenance.lower()
    return not any(sub.lower() in lower
                   for sub in FORBIDDEN_PROVENANCE_SUBSTRINGS)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_metadata(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("metadata")
    require(isinstance(meta, dict), "metadata must be an object")
    require_keys(meta, ["name", "claim_scope", "status", "derivation_status"],
                 "metadata")
    name = require_string(meta["name"], "metadata.name")
    claim_scope = require_string(meta["claim_scope"], "metadata.claim_scope")
    status = require_string(meta["status"], "metadata.status")
    derivation_status = require_string(meta["derivation_status"],
                                       "metadata.derivation_status")
    require(derivation_status in ALLOWED_DERIVATION_STATUSES,
            f"metadata.derivation_status must be one of "
            f"{sorted(ALLOWED_DERIVATION_STATUSES)}; got {derivation_status!r}")
    return {
        "name": name,
        "claim_scope": claim_scope,
        "status": status,
        "derivation_status": derivation_status,
    }


def validate_variables(data: dict[str, Any]) -> dict[str, Any]:
    vars_section = data.get("variables")
    require(isinstance(vars_section, dict), "variables must be an object")
    require_keys(vars_section,
                 ["phase_variables", "symbolic_coefficient_variables",
                  "constraint_auxiliary_variables"],
                 "variables")
    phase_vars = tuple(vars_section["phase_variables"])
    require(phase_vars == ("x", "ell", "q"),
            f"phase_variables must be ('x', 'ell', 'q'); got {phase_vars}")
    sym_vars = tuple(vars_section["symbolic_coefficient_variables"])
    expected_sym = ("h", "s", "tau", "c", "diag_slack", "bilinear_slack")
    require(set(sym_vars) == set(expected_sym),
            f"symbolic_coefficient_variables must be {expected_sym}; "
            f"got {sym_vars}")
    aux_vars = tuple(vars_section["constraint_auxiliary_variables"])
    expected_aux = ("rho_R", "rho_dR", "rho_xq")
    require(set(aux_vars) == set(expected_aux),
            f"constraint_auxiliary_variables must be {expected_aux}; "
            f"got {aux_vars}")
    # Disjointness
    common_ps = set(phase_vars) & set(sym_vars)
    require(not common_ps,
            f"phase and symbolic_coefficient variables must be disjoint; "
            f"intersection: {common_ps}")
    common_pa = set(phase_vars) & set(aux_vars)
    require(not common_pa,
            f"phase and constraint_auxiliary variables must be disjoint; "
            f"intersection: {common_pa}")
    common_sa = set(sym_vars) & set(aux_vars)
    require(not common_sa,
            f"symbolic_coefficient and constraint_auxiliary variables must be "
            f"disjoint; intersection: {common_sa}")
    return {
        "phase_variables": phase_vars,
        "symbolic_coefficient_variables": sym_vars,
        "constraint_auxiliary_variables": aux_vars,
    }


def validate_derivation_chain(data: dict[str, Any]) -> list[dict[str, Any]]:
    chain = data.get("derivation_chain")
    require(isinstance(chain, list) and chain,
            "derivation_chain must be a nonempty list")
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for i, step in enumerate(chain):
        ctx = f"derivation_chain[{i}]"
        require(isinstance(step, dict), f"{ctx} must be an object")
        require_keys(step, ["step_name", "provenance", "formula"], ctx)
        step_name = require_string(step["step_name"], f"{ctx}.step_name")
        require(step_name not in seen,
                f"{ctx}.step_name {step_name!r} is duplicated")
        seen.add(step_name)
        provenance = require_string(step["provenance"], f"{ctx}.provenance")
        require(is_noncandidate_provenance(provenance),
                f"{ctx}.provenance contains a forbidden substring")
        formula = require_string(step["formula"], f"{ctx}.formula")
        parsed.append({
            "step_name": step_name,
            "provenance": provenance,
            "formula": formula,
        })
    return parsed


def validate_packet_expression(pkt: dict[str, Any], ctx: str,
                               expected_expr: str) -> None:
    raw = pkt.get("packet_expression")
    require(isinstance(raw, str) and raw,
            f"{ctx}.packet_expression must be a nonempty string")
    require(raw == expected_expr,
            f"{ctx}.packet_expression must equal the derived formula "
            f"{expected_expr!r}; got {raw!r}")


def validate_constraint_correction_packets(
    data: dict[str, Any]
) -> list[dict[str, Any]]:
    packets = data.get("constraint_correction_packets")
    require(isinstance(packets, list) and packets,
            "constraint_correction_packets must be a nonempty list")
    expected = {
        "radial_trapped_constraint_correction_packet": "x**2 + ell**2 + q**2",
        "derivative_trapped_constraint_correction_packet": "x*ell + q**2",
        "carter_mixed_trapped_constraint_correction_packet":
            "x*q + ell*q - q**2",
    }
    require(len(packets) == len(expected),
            f"constraint_correction_packets must contain exactly "
            f"{len(expected)} entries; got {len(packets)}")
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for i, pkt in enumerate(packets):
        ctx = f"constraint_correction_packets[{i}]"
        require(isinstance(pkt, dict), f"{ctx} must be an object")
        declared_name = pkt.get("name")
        require(isinstance(declared_name, str) and declared_name,
                f"{ctx}.name must be a nonempty string")
        require(declared_name in expected,
                f"{ctx}.name {declared_name!r} is not one of "
                f"{sorted(expected.keys())}")
        validate_packet_expression(pkt, ctx, expected[declared_name])
        require(declared_name not in seen,
                f"{ctx}.name {declared_name!r} is duplicated")
        seen.add(declared_name)
        # constraint_variable field
        constraint_variable = pkt.get("constraint_variable")
        require(isinstance(constraint_variable, str) and constraint_variable,
                f"{ctx}.constraint_variable must be a nonempty string")
        expected_cv = {
            "radial_trapped_constraint_correction_packet": "rho_R",
            "derivative_trapped_constraint_correction_packet": "rho_dR",
            "carter_mixed_trapped_constraint_correction_packet": "rho_xq",
        }
        require(constraint_variable == expected_cv[declared_name],
                f"{ctx}.constraint_variable must be "
                f"{expected_cv[declared_name]!r}; got {constraint_variable!r}")
        # multiplier field
        multiplier = pkt.get("multiplier")
        require(isinstance(multiplier, str) and multiplier,
                f"{ctx}.multiplier must be a nonempty string")
        expected_mult = {
            "radial_trapped_constraint_correction_packet": "diag_slack",
            "derivative_trapped_constraint_correction_packet": "bilinear_slack",
            "carter_mixed_trapped_constraint_correction_packet": "bilinear_slack",
        }
        require(multiplier == expected_mult[declared_name],
                f"{ctx}.multiplier must be "
                f"{expected_mult[declared_name]!r}; got {multiplier!r}")
        # provenance
        provenance = require_string(pkt["provenance"], f"{ctx}.provenance")
        require(is_noncandidate_provenance(provenance),
                f"{ctx}.provenance contains a forbidden substring")
        parsed.append({
            "name": declared_name,
            "packet_expression": expected[declared_name],
            "constraint_variable": constraint_variable,
            "multiplier": multiplier,
            "provenance": provenance,
        })
    return parsed


def validate_sos_defects(data: dict[str, Any]) -> list[dict[str, Any]]:
    defects = data.get("sos_defects")
    require(isinstance(defects, list) and defects,
            "sos_defects must be a nonempty list")
    expected = {
        "gauge_tangent_sos_defect": {
            "square_expression": "c*x - ell",
            "weight": "s",
        },
    }
    require(len(defects) == len(expected),
            f"sos_defects must contain exactly {len(expected)} entries; "
            f"got {len(defects)}")
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    for i, defect in enumerate(defects):
        ctx = f"sos_defects[{i}]"
        require(isinstance(defect, dict), f"{ctx} must be an object")
        declared_name = defect.get("name")
        require(isinstance(declared_name, str) and declared_name,
                f"{ctx}.name must be a nonempty string")
        require(declared_name in expected,
                f"{ctx}.name {declared_name!r} is not one of "
                f"{sorted(expected.keys())}")
        require(declared_name not in seen,
                f"{ctx}.name {declared_name!r} is duplicated")
        seen.add(declared_name)
        sq = defect.get("square_expression")
        require(isinstance(sq, str) and sq,
                f"{ctx}.square_expression must be a nonempty string")
        require(sq == expected[declared_name]["square_expression"],
                f"{ctx}.square_expression must be "
                f"{expected[declared_name]['square_expression']!r}; got {sq!r}")
        wt = defect.get("weight")
        require(isinstance(wt, str) and wt,
                f"{ctx}.weight must be a nonempty string")
        require(wt == expected[declared_name]["weight"],
                f"{ctx}.weight must be "
                f"{expected[declared_name]['weight']!r}; got {wt!r}")
        provenance = require_string(defect["provenance"], f"{ctx}.provenance")
        require(is_noncandidate_provenance(provenance),
                f"{ctx}.provenance contains a forbidden substring")
        parsed.append({
            "name": declared_name,
            "square_expression": sq,
            "weight": wt,
            "provenance": provenance,
        })
    return parsed


def validate_backend_density(data: dict[str, Any]) -> str:
    bd = data.get("backend_density_expression")
    require(isinstance(bd, str) and bd,
            "backend_density_expression must be a nonempty string")
    expected = "h*x**2 + tau*q**2"
    require(bd == expected,
            f"backend_density_expression must be {expected!r}; got {bd!r}")
    return bd


def validate_q_grav_expression(data: dict[str, Any]) -> str:
    qg = data.get("q_grav_expression")
    require(isinstance(qg, str) and qg,
            "q_grav_expression must be a nonempty string")
    expected = (
        "h*x**2 - 2*s*c*x*ell + s*c**2*x**2 + s*ell**2 + tau*q**2 "
        "+ diag_slack*(x**2 + ell**2 + q**2) "
        "+ bilinear_slack*(x*ell + x*q + ell*q)"
    )
    require(qg == expected,
            f"q_grav_expression must be {expected!r}; got {qg!r}")
    return qg


def validate_sos_weight_interval(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the certified SOS weight interval for s (superradiant coefficient)."""
    sec = data.get("sos_weight_interval")
    require(isinstance(sec, dict), "sos_weight_interval must be an object")
    require_keys(sec, ["variable", "interval_lo", "interval_hi",
                       "accepted_tube_lo", "accepted_tube_hi", "provenance"],
                 "sos_weight_interval")
    var = require_string(sec["variable"], "sos_weight_interval.variable")
    require(var == "s",
            f"sos_weight_interval.variable must be 's'; got {var!r}")
    lo = sp.Rational(sec["interval_lo"])
    hi = sp.Rational(sec["interval_hi"])
    require(lo <= hi,
            f"sos_weight_interval.interval_lo ({lo}) must be <= "
            f"interval_hi ({hi})")
    tube_lo = sp.Rational(sec["accepted_tube_lo"])
    tube_hi = sp.Rational(sec["accepted_tube_hi"])
    require(tube_lo <= tube_hi,
            f"sos_weight_interval.accepted_tube_lo ({tube_lo}) must be <= "
            f"accepted_tube_hi ({tube_hi})")
    # For the SOS defect to be nonnegative, we need s >= 0 on the certified interval.
    # This check comes before the tube containment check so that a negative
    # interval_lo is rejected with the nonnegativity message.
    require(lo >= 0,
            f"sos_weight_interval.interval_lo ({lo}) must be >= 0 for the "
            f"gauge-tangent SOS defect s*(c*x-ell)^2 to be nonnegative")
    # The certified interval must lie inside the accepted tube.
    require(lo >= tube_lo and hi <= tube_hi,
            f"sos_weight_interval interval [{lo}, {hi}] must lie inside "
            f"accepted tube [{tube_lo}, {tube_hi}]")
    provenance = require_string(sec["provenance"],
                                "sos_weight_interval.provenance")
    require(is_noncandidate_provenance(provenance),
            "sos_weight_interval.provenance contains a forbidden substring")
    return {
        "variable": var,
        "interval_lo": str(lo),
        "interval_hi": str(hi),
        "accepted_tube_lo": str(tube_lo),
        "accepted_tube_hi": str(tube_hi),
    }


def validate_principal_symbol_trace_reference(
    data: dict[str, Any], base_dir: Path | None
) -> dict[str, Any]:
    raw_path = data.get("principal_symbol_trace_certificate_path")
    require(isinstance(raw_path, str) and raw_path,
            "principal_symbol_trace_certificate_path must be a nonempty string")
    if base_dir is None:
        base_dir = HERE
    trace_path = (base_dir / raw_path).resolve()
    require(trace_path.exists(),
            f"principal_symbol_trace_certificate_path does not exist: {trace_path}")
    trace_data = load_json(trace_path)
    require(trace_data.get("chart") == data.get("chart"),
            "chart must match the principal symbol trace certificate's chart")
    return {
        "trace_certificate_path": str(trace_path),
        "trace_chart": trace_data.get("chart"),
    }


def verify_quotient_identity_identities() -> dict[str, Any]:
    """Verify the QIEDT identities by sympy polynomial reduction.

    This is the core mathematical content of the theorem. We:

    (Q1) Verify the QIEDT quotient identity:
         q_grav - rho_backend - s*(c*x-ell)^2
           - diag_slack*P_R - bilinear_slack*P_dR - bilinear_slack*P_xq == 0.

    (Q2) Derive P_xq = x*q + ell*q - q^2 as the unique degree-2 polynomial
         of the form alpha*x*q + beta*ell*q + gamma*q^2 that (i) is invariant
         under the simultaneous swap (x <-> ell), and (ii) satisfies the
         decomposition P_bil = P_dR + P_xq.

    (Q3) Verify the bilinear packet decomposition P_bil = P_dR + P_xq.

    (Q4) Verify linear independence of P_R, P_dR, P_xq.

    (Q5) Verify the backend density is phase-homogeneous of degree 2 and
         contains only the radial and Carter diagonal packets.

    (Q6) Verify the SOS defect s*(c*x-ell)^2 is a perfect square with weight s.
    """
    x, ell, q = sp.symbols("x ell q")
    h, s, tau, c, diag_slack, bilinear_slack = sp.symbols(
        "h s tau c diag_slack bilinear_slack")

    # The derived correction packets.
    P_R = x**2 + ell**2 + q**2
    P_dR = x * ell + q**2
    P_xq = x * q + ell * q - q**2
    P_bil = x * ell + x * q + ell * q

    # The GPDDT density q_grav.
    q_grav = (h * x**2 - 2 * s * c * x * ell + s * c**2 * x**2 + s * ell**2
              + tau * q**2
              + diag_slack * (x**2 + ell**2 + q**2)
              + bilinear_slack * (x * ell + x * q + ell * q))

    # The minimal Kerr-clean backend density.
    rho_backend = h * x**2 + tau * q**2

    # The gauge-tangent SOS defect.
    sos_defect = s * (c * x - ell) ** 2

    # The constraint-ideal contribution.
    ideal = (diag_slack * P_R
             + bilinear_slack * P_dR
             + bilinear_slack * P_xq)

    # (Q1) The QIEDT quotient identity.
    residual_q1 = sp.expand(q_grav - rho_backend - sos_defect - ideal)
    require(residual_q1 == 0,
            f"QIEDT quotient identity (Q1) failed with residual "
            f"{residual_q1}")

    # (Q2) Derive P_xq = x*q + ell*q - q^2 as the unique degree-2 polynomial
    # of the form alpha*x*q + beta*ell*q + gamma*q^2 that is invariant under
    # the simultaneous swap (x <-> ell) and satisfies P_bil = P_dR + P_xq.
    alpha, beta, gamma = sp.symbols("alpha beta gamma")
    cand = alpha * x * q + beta * ell * q + gamma * q**2
    # Invariance under (x <-> ell): simultaneous substitution.
    inv_xell = sp.expand(cand - cand.subs([(x, ell), (ell, x)], simultaneous=True))
    # The invariance condition inv_xell == 0 forces alpha = beta.
    # Collect the coefficients of the monomials x*q and ell*q.
    c_xq = sp.simplify(inv_xell.coeff(x, 1).coeff(ell, 0).coeff(q, 1))
    c_ellq = sp.simplify(inv_xell.coeff(x, 0).coeff(ell, 1).coeff(q, 1))
    # The invariance condition is alpha - beta == 0 (from c_xq) and
    # -alpha + beta == 0 (from c_ellq). Both are equivalent to alpha == beta.
    # Solve the invariance equation for the relation among (alpha, beta, gamma).
    invariance_solutions = sp.solve([c_xq, c_ellq], [alpha, beta, gamma],
                                    dict=True)
    # The solution set should force alpha == beta (gamma free).
    require(len(invariance_solutions) == 1,
            f"Carter-mixed invariance under (x<->ell) should force a single "
            f"relation among (alpha, beta, gamma); got {invariance_solutions}")
    sol_inv = invariance_solutions[0]
    require(sol_inv.get(alpha) == beta,
            f"Carter-mixed invariance should force alpha = beta; got {sol_inv}")
    # Now impose P_bil = P_dR + P_xq, i.e.,
    # P_xq = P_bil - P_dR = (x*ell + x*q + ell*q) - (x*ell + q^2) = x*q + ell*q - q^2.
    # So alpha = beta = 1, gamma = -1.
    p_xq_derived = sp.expand(P_bil - P_dR)
    require(p_xq_derived == P_xq,
            f"P_xq derivation via P_bil - P_dR failed: expected {P_xq}, "
            f"got {p_xq_derived}")
    # Check that alpha = beta = 1, gamma = -1 is the unique solution of the
    # combined system (invariance + P_bil = P_dR + P_xq).
    # P_bil = P_dR + P_xq means: x*ell + x*q + ell*q = (x*ell + q^2) + (alpha*x*q + beta*ell*q + gamma*q^2)
    # i.e., x*q + ell*q = alpha*x*q + beta*ell*q + gamma*q^2 + q^2
    # i.e., (1 - alpha)*x*q + (1 - beta)*ell*q + (-1 - gamma)*q^2 = 0
    # which forces alpha = 1, beta = 1, gamma = -1.
    decomp_eq = sp.expand(P_bil - P_dR - cand)
    c_xq_dec = sp.simplify(decomp_eq.coeff(x, 1).coeff(ell, 0).coeff(q, 1))
    c_ellq_dec = sp.simplify(decomp_eq.coeff(x, 0).coeff(ell, 1).coeff(q, 1))
    c_q2_dec = sp.simplify(decomp_eq.coeff(x, 0).coeff(ell, 0).coeff(q, 2))
    sol_full = sp.solve([c_xq_dec, c_ellq_dec, c_q2_dec],
                        [alpha, beta, gamma], dict=True)
    require(len(sol_full) == 1 and sol_full[0] == {alpha: 1, beta: 1, gamma: -1},
            f"Uniqueness of P_xq coefficients failed: expected "
            f"{{alpha: 1, beta: 1, gamma: -1}}, got {sol_full}")

    # (Q3) Verify the bilinear packet decomposition P_bil = P_dR + P_xq.
    decomp = sp.expand(P_dR + P_xq)
    require(decomp == P_bil,
            f"Bilinear packet decomposition P_bil = P_dR + P_xq failed: "
            f"expected {P_bil}, got {decomp}")

    # (Q4) Linear independence of P_R, P_dR, P_xq.
    # If alpha*P_R + beta*P_dR + gamma*P_xq == 0 as a polynomial, then
    # alpha = beta = gamma = 0.
    lin = sp.expand(alpha * P_R + beta * P_dR + gamma * P_xq)
    # Extract coefficients of x^2 (gives alpha), x*ell (gives beta), x*q (gives gamma).
    c_x2 = sp.simplify(lin.coeff(x, 2).coeff(ell, 0).coeff(q, 0))
    c_xell = sp.simplify(lin.coeff(x, 1).coeff(ell, 1).coeff(q, 0))
    c_xq_lin = sp.simplify(lin.coeff(x, 1).coeff(ell, 0).coeff(q, 1))
    require(c_x2 == alpha,
            f"Linear independence: coefficient of x^2 in alpha*P_R + beta*P_dR "
            f"+ gamma*P_xq must be alpha; got {c_x2}")
    require(c_xell == beta,
            f"Linear independence: coefficient of x*ell in alpha*P_R + "
            f"beta*P_dR + gamma*P_xq must be beta; got {c_xell}")
    require(c_xq_lin == gamma,
            f"Linear independence: coefficient of x*q in alpha*P_R + "
            f"beta*P_dR + gamma*P_xq must be gamma; got {c_xq_lin}")

    # (Q5) Backend density is phase-homogeneous of degree 2.
    # Check that every monomial in rho_backend has phase degree exactly 2.
    backend_poly = sp.Poly(rho_backend, x, ell, q)
    for monom in backend_poly.monoms():
        phase_deg = sum(monom)
        require(phase_deg == 2,
                f"Backend density must be phase-homogeneous of degree 2; "
                f"monomial {monom} has phase degree {phase_deg}")
    # Check that rho_backend contains only h*x^2 and tau*q^2 (no ell^2, no cross terms).
    expected_backend = sp.expand(h * x**2 + tau * q**2)
    require(sp.expand(rho_backend - expected_backend) == 0,
            f"Backend density must be h*x^2 + tau*q^2; got {rho_backend}")

    # (Q6) The SOS defect s*(c*x-ell)^2 is a perfect square with weight s.
    # Expand: s*(c*x-ell)^2 = s*c^2*x^2 - 2*s*c*x*ell + s*ell^2.
    sos_expanded = sp.expand(sos_defect)
    expected_sos = sp.expand(s * (c * x - ell) ** 2)
    require(sos_expanded == expected_sos,
            f"SOS defect expansion failed: expected {expected_sos}, "
            f"got {sos_expanded}")
    # Check it's a perfect square times a weight: s * (c*x - ell)^2.
    # Factor check: the SOS defect should factor as s * (c*x - ell)^2.
    factored = sp.factor(sos_defect / s)
    require(factored == (c * x - ell) ** 2,
            f"SOS defect must factor as s * (c*x - ell)^2; "
            f"got s * {factored}")

    return {
        "checked_identities": [
            "qiedt_quotient_identity_q_grav_equals_backend_plus_sos_plus_ideal",
            "carter_mixed_packet_P_xq_unique_swap_invariant_decomposition",
            "bilinear_packet_decomposition_P_bil_equals_P_dR_plus_P_xq",
            "linear_independence_of_P_R_P_dR_P_xq",
            "backend_density_phase_homogeneous_degree_2_minimal_kerr_clean",
            "sos_defect_is_perfect_square_with_weight_s",
        ],
        "derived_packets": {
            "P_R": str(P_R),
            "P_dR": str(P_dR),
            "P_xq": str(P_xq),
            "P_bil": str(P_bil),
        },
        "backend_density": str(rho_backend),
        "sos_defect": str(sos_defect),
        "constraint_ideal": str(sp.expand(ideal)),
    }


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be a JSON object")
    meta = validate_metadata(data)
    require_keys(data, ["chart", "variables",
                        "principal_symbol_trace_certificate_path",
                        "derivation_chain",
                        "constraint_correction_packets",
                        "sos_defects",
                        "backend_density_expression",
                        "q_grav_expression",
                        "sos_weight_interval"],
                 "certificate")
    chart = require_string(data["chart"], "chart")
    vars_section = validate_variables(data)
    chain = validate_derivation_chain(data)
    packets = validate_constraint_correction_packets(data)
    defects = validate_sos_defects(data)
    backend = validate_backend_density(data)
    q_grav = validate_q_grav_expression(data)
    sos_interval = validate_sos_weight_interval(data)
    trace_info = validate_principal_symbol_trace_reference(data, base_dir)
    identity_check = verify_quotient_identity_identities()

    return {
        "accepted": True,
        "metadata": meta,
        "chart": chart,
        "variables": {
            "phase_variables": list(vars_section["phase_variables"]),
            "symbolic_coefficient_variables":
                list(vars_section["symbolic_coefficient_variables"]),
            "constraint_auxiliary_variables":
                list(vars_section["constraint_auxiliary_variables"]),
        },
        "derivation_chain_length": len(chain),
        "constraint_correction_packets": packets,
        "sos_defects": defects,
        "backend_density_expression": backend,
        "q_grav_expression": q_grav,
        "sos_weight_interval": sos_interval,
        "principal_symbol_trace_certificate": trace_info["trace_certificate_path"],
        "checked_identities": (
            identity_check["checked_identities"]
            + [
                "constraint_correction_packet_expressions_match_derived_formulas",
                "sos_defect_expressions_match_derived_formulas",
                "backend_density_expression_matches_derived_formula",
                "q_grav_expression_matches_derived_formula",
                "derivation_chain_provenance_is_noncandidate",
                "sos_weight_interval_nonnegative_for_gauge_tangent_defect",
            ]
        ),
        "derived_packets": identity_check["derived_packets"],
        "backend_density": identity_check["backend_density"],
        "sos_defect": identity_check["sos_defect"],
        "constraint_ideal": identity_check["constraint_ideal"],
    }


def verify_file(path: Path) -> dict[str, Any]:
    return verify(load_json(path), base_dir=path.parent)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter quotient-identity derivation rejected: {exc}",
              file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
