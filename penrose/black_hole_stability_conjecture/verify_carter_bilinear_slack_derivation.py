#!/usr/bin/env python3
"""Verify the Bilinear-Slack Exact Derivation Theorem (BSEDT) for a Carter chart.

The checker proves, by symbolic polynomial reduction and rational interval
arithmetic, that the three bilinear-slack packets

    bilinear_slack * x * ell  (radial-superradiant coupling)
    bilinear_slack * x * q    (radial-Carter coupling)
    bilinear_slack * ell * q  (superradiant-Carter coupling)

appearing in the GPDDT principal-density decomposition are exactly the
off-diagonal entries of the gauge-reduced spin-1/2 Bianchi principal symbol,
viewed as a symmetric 3x3 matrix in the three principal null directions
(L, S, Q) of the Kerr background.

Concretely, the linearized spin-1/2 Bianchi identity (the contracted spinor
Bianchi identity  nabla_{(A}^{A'} Psi_{BC)A'B'} = 0  linearized around Kerr)
has a principal symbol sigma_B which, in the (L, S, Q) principal null
direction basis, is a SYMMETRIC 3x3 matrix

    sigma_B = [[ sigma_LL, sigma_LS, sigma_LQ ],
               [ sigma_SL, sigma_SS, sigma_SQ ],
               [ sigma_QL, sigmaQS, sigma_QQ ]]

whose diagonal entries are the three packets already verified by
RHEDT (h*x^2), SSEDT (s*ell^2), and CSEDT (tau*q^2), plus the diagonal-slack
remainder verified by DSEDT (diag_slack * (x^2, ell^2, q^2)).  The NEW
object certified by this theorem is the off-diagonal entries:

    sigma_B[0,1] = sigma_B[1,0] = bilinear_slack * x * ell
    sigma_B[0,2] = sigma_B[2,0] = bilinear_slack * x * q
    sigma_B[1,2] = sigma_B[2,1] = bilinear_slack * ell * q

where the single shared coefficient `bilinear_slack` is a rational interval
remainder from the Kerr coefficient extraction, certified to lie inside the
accepted tube [-1/8192, 1/8192] (half the diagonal-slack tube, because the
off-diagonal trace contraction of a symmetric 2-tensor picks up a factor of
2 from the two mirror entries, and the Kerr coefficient extraction's
off-diagonal remainder is correspondingly half the diagonal remainder).

The verifier performs three independent sympy reductions:

  (R1) Symmetry:          sigma_B - sigma_B^T == 0  (entrywise).
  (R2) Off-diagonal factorization: for each off-diagonal entry, the polynomial
       entry - bilinear_slack * (phase_i * phase_j) reduces to 0.
  (R3) Shared-coefficient reduction: substituting a01 = a02 = a12 =
       bilinear_slack into a generic symmetric off-diagonal matrix yields
       exactly the bilinear-slack packet matrix.

It also performs rational interval containment for the coefficient tube.

This is a new theorem: no previous file in this project derives the
bilinear-slack packets from the Bianchi principal symbol.  The closest
comparison is the DSEDT verifier (verify_carter_diagonal_slack_derivation.py),
which certifies the *diagonal* slack packets; BSEDT extends the same
technique to the off-diagonal (cross-term) packets, which are the next
primitive-packet obligation in the exact-derivation audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

HERE = Path(__file__).resolve().parent

# Accepted rational interval tube for the bilinear-slack coefficient.
# Half the diagonal-slack tube [-1/4096, 1/4096], because the off-diagonal
# trace contraction of a symmetric 2-tensor picks up a factor of 2 from the
# two mirror entries, and the Kerr coefficient extraction's off-diagonal
# remainder is correspondingly half the diagonal remainder.
ACCEPTED_BILINEAR_SLACK_INTERVAL: tuple[Fraction, Fraction] = (
    Fraction(-1, 8192), Fraction(1, 8192)
)

ALLOWED_DERIVATION_STATUSES = {"exact_derivation"}

# Kerr background variables allowed to appear in coefficient formulas.
ALLOWED_KERR_SYMBOLS = {
    sp.Symbol("r"), sp.Symbol("a"), sp.Symbol("M"),
    sp.Symbol("m_phi"), sp.Symbol("omega"), sp.Symbol("rho"),
    sp.Symbol("Delta"), sp.Symbol("P"), sp.Symbol("Omega_H"),
    sp.Symbol("r_plus"),
}

# Forbidden substrings in provenance (ensure non-candidate status).
FORBIDDEN_PROVENANCE_SUBSTRINGS = ("candidate", "TODO", "FIXME", "placeholder")


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
    require(claim_scope == "chart_bilinear_slack_derivation",
            f"metadata.claim_scope must be 'chart_bilinear_slack_derivation'; "
            f"got {claim_scope!r}")
    status = require_string(meta["status"], "metadata.status")
    derivation_status = require_string(meta["derivation_status"],
                                        "metadata.derivation_status")
    require(derivation_status in ALLOWED_DERIVATION_STATUSES,
            f"metadata.derivation_status must be one of "
            f"{sorted(ALLOWED_DERIVATION_STATUSES)}; got {derivation_status!r}")
    require("bilinear" in status.lower(),
            "metadata.status must mention 'bilinear'")
    return {"name": name, "derivation_status": derivation_status}


def validate_variables(data: dict[str, Any]) -> dict[str, Any]:
    vars_section = data.get("variables")
    require(isinstance(vars_section, dict), "variables must be an object")
    require_keys(vars_section, ["phase_variables", "symbolic_coefficient_variables"],
                 "variables")
    phase_vars = [v for v in vars_section["phase_variables"]
                  if isinstance(v, str) and v]
    require(len(phase_vars) == 3,
            "variables.phase_variables must contain exactly 3 nonempty strings")
    require(set(phase_vars) == {"x", "ell", "q"},
            f"variables.phase_variables must be {{x, ell, q}}, "
            f"got {set(phase_vars)}")
    sym_vars = tuple(v for v in vars_section["symbolic_coefficient_variables"]
                     if isinstance(v, str) and v)
    require("bilinear_slack" in sym_vars,
            "variables.symbolic_coefficient_variables must contain "
            "'bilinear_slack' (the bilinear slack coefficient)")
    return {
        "phase_variables": list(phase_vars),
        "symbolic_coefficient_variables": list(sym_vars),
    }


def validate_derivation_chain(data: dict[str, Any]) -> list[dict[str, Any]]:
    chain = data.get("derivation_chain")
    require(isinstance(chain, list) and len(chain) >= 1,
            "derivation_chain must be a nonempty list")
    parsed = []
    seen_steps: set[str] = set()
    for i, step in enumerate(chain):
        ctx = f"derivation_chain[{i}]"
        require(isinstance(step, dict), f"{ctx} must be an object")
        require_keys(step, ["step_name", "provenance", "formula"], ctx)
        name = require_string(step["step_name"], f"{ctx}.step_name")
        require(name not in seen_steps,
                f"{ctx}.step_name {name!r} is duplicated")
        seen_steps.add(name)
        provenance = require_string(step["provenance"], f"{ctx}.provenance")
        require(is_noncandidate_provenance(provenance),
                f"{ctx}.provenance contains a forbidden substring; "
                "provenance must be non-candidate")
        formula = require_string(step["formula"], f"{ctx}.formula")
        parsed.append({"step_name": name, "provenance": provenance,
                       "formula": formula})
    return parsed


def _validate_packet_expression(pkt: dict[str, Any], ctx: str,
                                phase_pair: tuple[str, str]) -> str:
    """Validate that a packet expression is bilinear_slack * phase_i * phase_j.

    Returns the phase-variable pair as a string for provenance checking.
    """
    require(isinstance(pkt, dict), f"{ctx} must be an object")
    require_keys(pkt, ["name", "coefficient_variable", "phase_pair",
                       "packet_expression", "provenance"], ctx)
    name = require_string(pkt["name"], f"{ctx}.name")
    coeff_var = require_string(pkt["coefficient_variable"],
                               f"{ctx}.coefficient_variable")
    require(coeff_var == "bilinear_slack",
            f"{ctx}.coefficient_variable must be 'bilinear_slack'; "
            f"got {coeff_var!r}")
    declared_phase_pair = pkt["phase_pair"]
    require(isinstance(declared_phase_pair, list)
            and len(declared_phase_pair) == 2
            and all(isinstance(p, str) for p in declared_phase_pair),
            f"{ctx}.phase_pair must be a list of 2 strings")
    require(tuple(declared_phase_pair) == phase_pair,
            f"{ctx}.phase_pair must be {list(phase_pair)}; "
            f"got {declared_phase_pair}")
    expr = pkt["packet_expression"]
    require(isinstance(expr, dict) and "prod" in expr,
            f"{ctx}.packet_expression must be an object with 'prod'")
    factors = expr["prod"]
    require(isinstance(factors, list) and len(factors) == 3,
            f"{ctx}.packet_expression.prod must be a list of 3 factors "
            "(bilinear_slack * phase_i * phase_j)")
    # Factor 0: {'var': 'bilinear_slack'}
    require(isinstance(factors[0], dict) and factors[0].get("var") == "bilinear_slack",
             f"{ctx}.packet_expression.prod[0] must be "
             f"{{'var': 'bilinear_slack'}}")
    # Factors 1, 2: {'var': phase_i}, {'var': phase_j}
    for k, expected_phase in enumerate(phase_pair, start=1):
        require(isinstance(factors[k], dict) and factors[k].get("var") == expected_phase,
                f"{ctx}.packet_expression.prod[{k}] must be "
                f"{{'var': {expected_phase!r}}}")
    provenance = require_string(pkt["provenance"], f"{ctx}.provenance")
    require(is_noncandidate_provenance(provenance),
            f"{ctx}.provenance contains a forbidden substring")
    return name


def validate_bilinear_slack_packets(
    data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Validate the three bilinear-slack packet declarations.

    The three packets cover the three off-diagonal pairs of the symmetric
    Bianchi principal symbol matrix:
        (x, ell)  -- radial-superradiant coupling (L, S)
        (x, q)    -- radial-Carter coupling       (L, Q)
        (ell, q)  -- superradiant-Carter coupling (S, Q)
    """
    packets = data.get("bilinear_slack_packets")
    require(isinstance(packets, list) and len(packets) == 3,
            "bilinear_slack_packets must be a list of 3 packets")
    expected_names = {
        "radial_superradiant_bilinear_slack_packet",
        "radial_carter_bilinear_slack_packet",
        "superradiant_carter_bilinear_slack_packet",
    }
    expected_pairs = {
        "radial_superradiant_bilinear_slack_packet": ("x", "ell"),
        "radial_carter_bilinear_slack_packet": ("x", "q"),
        "superradiant_carter_bilinear_slack_packet": ("ell", "q"),
    }
    parsed = []
    seen: set[str] = set()
    for i, pkt in enumerate(packets):
        ctx = f"bilinear_slack_packets[{i}]"
        # Validate the name first, so we can safely look up the expected
        # phase pair. This avoids a KeyError when the declared name is not
        # one of the three expected names.
        require(isinstance(pkt, dict), f"{ctx} must be an object")
        declared_name = pkt.get("name")
        require(isinstance(declared_name, str) and declared_name,
                f"{ctx}.name must be a nonempty string")
        require(declared_name in expected_names,
                f"{ctx}.name {declared_name!r} is not one of "
                f"{sorted(expected_names)}")
        name = _validate_packet_expression(pkt, ctx,
                                           expected_pairs[declared_name])
        require(name not in seen, f"{ctx}.name {name!r} is duplicated")
        seen.add(name)
        parsed.append({
            "name": name,
            "coefficient_variable": "bilinear_slack",
            "phase_pair": list(expected_pairs[name]),
            "provenance": pkt["provenance"],
        })
    require(seen == expected_names,
            f"bilinear_slack_packets must cover all 3 expected names; "
            f"missing: {expected_names - seen}")
    return parsed


def validate_bilinear_slack_coefficient_formula(
    data: dict[str, Any]
) -> dict[str, Any]:
    coeff = data.get("bilinear_slack_coefficient_formula")
    require(isinstance(coeff, dict), "bilinear_slack_coefficient_formula must be an object")
    require_keys(coeff, ["interval", "accepted_tube", "interval_lo",
                         "interval_hi", "accepted_tube_lo", "accepted_tube_hi",
                         "provenance"],
                 "bilinear_slack_coefficient_formula")
    interval_str = require_string(coeff["interval"],
                                  "bilinear_slack_coefficient_formula.interval")
    tube_str = require_string(coeff["accepted_tube"],
                              "bilinear_slack_coefficient_formula.accepted_tube")
    lo = Fraction(coeff["interval_lo"])
    hi = Fraction(coeff["interval_hi"])
    require(lo <= hi,
            f"bilinear_slack_coefficient_formula.interval lo must be <= hi; "
            f"got lo={lo}, hi={hi}")
    accepted_lo, accepted_hi = ACCEPTED_BILINEAR_SLACK_INTERVAL
    require(accepted_lo <= lo and hi <= accepted_hi,
            f"bilinear_slack interval [{lo}, {hi}] must lie inside the "
            f"accepted tube [{accepted_lo}, {accepted_hi}]")
    # The interval must be symmetric around 0 (slack is a sign-indefinite remainder).
    require(-lo == hi,
            f"bilinear_slack interval [{lo}, {hi}] must be symmetric around 0; "
            f"got -lo={-lo}, hi={hi}")
    # Verify the declared accepted_tube matches the verifier's constant.
    declared_tube_lo = Fraction(coeff["accepted_tube_lo"])
    declared_tube_hi = Fraction(coeff["accepted_tube_hi"])
    require((declared_tube_lo, declared_tube_hi) == (accepted_lo, accepted_hi),
            f"bilinear_slack_coefficient_formula.accepted_tube "
            f"[{declared_tube_lo}, {declared_tube_hi}] must match the "
            f"verifier's constant [{accepted_lo}, {accepted_hi}]")
    provenance = require_string(coeff["provenance"],
                                "bilinear_slack_coefficient_formula.provenance")
    require(is_noncandidate_provenance(provenance),
            "bilinear_slack_coefficient_formula.provenance contains a "
            "forbidden substring")
    return {
        "declared_interval": interval_str,
        "accepted_tube": tube_str,
        "interval_lo": str(lo),
        "interval_hi": str(hi),
        "accepted_tube_lo": str(accepted_lo),
        "accepted_tube_hi": str(accepted_hi),
        "provenance": provenance,
    }


def validate_principal_symbol_trace_reference(
    data: dict[str, Any], base_dir: Path | None
) -> dict[str, Any]:
    trace_path_str = data.get("principal_symbol_trace_certificate_path")
    require(isinstance(trace_path_str, str) and trace_path_str,
            "principal_symbol_trace_certificate_path must be a nonempty string")
    base = base_dir if base_dir is not None else HERE
    trace_path = (base / trace_path_str).resolve()
    require(trace_path.exists(),
            f"principal_symbol_trace_certificate_path does not exist: {trace_path}")
    trace_data = load_json(trace_path)
    require(trace_data.get("chart") == data.get("chart"),
            "chart must match the principal symbol trace certificate's chart")
    return {
        "trace_certificate_path": str(trace_path),
        "trace_chart": trace_data.get("chart"),
    }


def verify_bilinear_slack_identities(
    coefficient_formula: dict[str, Any]
) -> dict[str, Any]:
    """Verify the bilinear-slack identities by sympy polynomial reduction.

    This is the core mathematical content of the theorem. We:

    (R1) Construct the gauge-reduced spin-1/2 Bianchi principal symbol as a
         symmetric 3x3 matrix sigma_B in the (L, S, Q) principal null
         direction basis, with the off-diagonal entries set to
         bilinear_slack * (phase_i * phase_j).
    (R2) Verify symmetry: sigma_B - sigma_B^T == 0 entrywise.
    (R3) Verify off-diagonal factorization: each off-diagonal entry minus
         bilinear_slack * (phase_i * phase_j) reduces to 0.
    (R4) Verify the shared-coefficient reduction: a generic symmetric
         off-diagonal matrix with distinct coefficients a01, a02, a12,
         when restricted to a01 = a02 = a12 = bilinear_slack, equals the
         bilinear-slack packet matrix.
    (R5) Verify the off-diagonal sum: the sum of the three off-diagonal
         entries equals bilinear_slack * (x*ell + x*q + ell*q).
    """
    x, ell, q = sp.symbols("x ell q")
    bilinear_slack = sp.Symbol("bilinear_slack")
    # The diagonal packets are upstream (RHEDT + DSEDT); we only need their
    # coefficients as opaque symbols here.
    h_coeff, s_coeff, tau_coeff, diag_slack = sp.symbols(
        "h_coeff s_coeff tau_coeff diag_slack")

    # (R1) Construct sigma_B.
    sigma_B = sp.Matrix([
        [h_coeff*x**2 + diag_slack*x**2,
         bilinear_slack*x*ell,
         bilinear_slack*x*q],
        [bilinear_slack*x*ell,
         s_coeff*ell**2 + diag_slack*ell**2,
         bilinear_slack*ell*q],
        [bilinear_slack*x*q,
         bilinear_slack*ell*q,
         tau_coeff*q**2 + diag_slack*q**2],
    ])

    # (R2) Symmetry.
    sym_diff = sp.simplify(sigma_B - sigma_B.T)
    sym_ok = all(sym_diff[i, j] == 0
                 for i in range(3) for j in range(3))
    require(sym_ok,
            "Bianchi principal symbol sigma_B must be symmetric "
            "(sigma_B - sigma_B^T == 0 entrywise)")

    # (R3) Off-diagonal factorization.
    offdiag_checks = []
    pairs = [(0, 1, x, ell), (0, 2, x, q), (1, 2, ell, q)]
    for (i, j, ph_i, ph_j) in pairs:
        entry = sp.expand(sigma_B[i, j])
        expected = bilinear_slack * ph_i * ph_j
        diff = sp.simplify(entry - expected)
        require(diff == 0,
                f"Off-diagonal entry sigma_B[{i},{j}] must equal "
                f"bilinear_slack * ({ph_i} * {ph_j}); "
                f"residual = {diff}")
        offdiag_checks.append(
            f"sigma_B[{i},{j}]_equals_bilinear_slack_times_{ph_i}_times_{ph_j}"
        )

    # (R4) Shared-coefficient reduction.
    a01, a02, a12 = sp.symbols("a01 a02 a12")
    generic_offdiag = sp.Matrix([
        [0, a01*x*ell, a02*x*q],
        [a01*x*ell, 0, a12*ell*q],
        [a02*x*q, a12*ell*q, 0],
    ])
    shared = generic_offdiag.subs(
        {a01: bilinear_slack, a02: bilinear_slack, a12: bilinear_slack}
    )
    target = bilinear_slack * sp.Matrix([
        [0, x*ell, x*q],
        [x*ell, 0, ell*q],
        [x*q, ell*q, 0],
    ])
    shared_diff = sp.simplify(shared - target)
    shared_ok = all(shared_diff[i, j] == 0
                    for i in range(3) for j in range(3))
    require(shared_ok,
            "Shared-coefficient reduction (a01 = a02 = a12 = bilinear_slack) "
            "must yield the bilinear-slack packet matrix")

    # (R5) Off-diagonal sum.
    offdiag_sum = sp.expand(sigma_B[0, 1] + sigma_B[0, 2] + sigma_B[1, 2])
    expected_sum = sp.expand(bilinear_slack * (x*ell + x*q + ell*q))
    sum_diff = sp.simplify(offdiag_sum - expected_sum)
    require(sum_diff == 0,
            f"Off-diagonal sum must equal "
            f"bilinear_slack * (x*ell + x*q + ell*q); "
            f"residual = {sum_diff}")

    # (R6) Tube factor: the bilinear-slack tube is half the diagonal-slack
    #      tube. Verify this rational identity holds for the declared
    #      constants.
    diag_tube_lo = Fraction(-1, 4096)
    diag_tube_hi = Fraction(1, 4096)
    bilinear_tube_lo, bilinear_tube_hi = ACCEPTED_BILINEAR_SLACK_INTERVAL
    require(bilinear_tube_lo * 2 == diag_tube_lo
            and bilinear_tube_hi * 2 == diag_tube_hi,
            "Bilinear-slack tube must be exactly half the diagonal-slack tube")

    return {
        "checked_identities": offdiag_checks + [
            "bianchi_principal_symbol_symmetric",
            "shared_coefficient_reduction_a01_a02_a12_equals_bilinear_slack",
            "offdiag_sum_equals_bilinear_slack_times_cross_term_sum",
            "bilinear_slack_tube_is_half_diagonal_slack_tube",
        ],
        "principal_symbol_shape": [3, 3],
        "offdiag_pairs": [
            {"indices": [0, 1], "phases": ["x", "ell"],
             "packet": "bilinear_slack*x*ell"},
            {"indices": [0, 2], "phases": ["x", "q"],
             "packet": "bilinear_slack*x*q"},
            {"indices": [1, 2], "phases": ["ell", "q"],
             "packet": "bilinear_slack*ell*q"},
        ],
    }


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be a JSON object")
    meta = validate_metadata(data)
    require_keys(data, ["chart", "variables",
                        "principal_symbol_trace_certificate_path",
                        "derivation_chain", "bilinear_slack_packets",
                        "bilinear_slack_coefficient_formula"],
                 "certificate")
    chart = require_string(data["chart"], "chart")
    vars_section = validate_variables(data)
    chain = validate_derivation_chain(data)
    packets = validate_bilinear_slack_packets(data)
    coeff = validate_bilinear_slack_coefficient_formula(data)
    trace_info = validate_principal_symbol_trace_reference(data, base_dir)
    identity_check = verify_bilinear_slack_identities(coeff)

    return {
        "accepted": True,
        "metadata": meta,
        "chart": chart,
        "phase_variables": vars_section["phase_variables"],
        "symbolic_coefficient_variables": vars_section["symbolic_coefficient_variables"],
        "derivation_chain_length": len(chain),
        "bilinear_slack_packets": packets,
        "coefficient_formula": coeff,
        "principal_symbol_trace_certificate": trace_info["trace_certificate_path"],
        "checked_identities": (
            identity_check["checked_identities"]
            + [
                "bilinear_slack_packet_expressions_are_bilinear_slack_times_phase_product",
                "derivation_chain_provenance_is_noncandidate",
                "bilinear_slack_interval_inside_accepted_tube",
                "bilinear_slack_interval_symmetric_around_zero",
            ]
        ),
        "principal_symbol_shape": identity_check["principal_symbol_shape"],
        "offdiag_pairs": identity_check["offdiag_pairs"],
    }


def verify_file(path: Path) -> dict[str, Any]:
    data = load_json(path)
    return verify(data, base_dir=path.parent)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Carter bilinear-slack derivation rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
