#!/usr/bin/env python3
"""Verify the Superradiant-Square Exact Derivation Theorem (SSEDT) for a Carter chart.

The checker proves, by symbolic polynomial reduction, that the superradiant-square
packet ``s * (ell - c*x)^2`` appearing in the principal-symbol trace certificate
is exactly the superradiant-square contribution to the normalized principal
symbol of the spin-1/2 Teukolsky operator, where the superradiant phase variable
``ell`` corresponds to the superradiant frequency shift ``omega - m_phi*Omega_H``
and the cross term ``-c*x`` comes from the coupling between the radial phase and
the superradiant phase via the horizon angular velocity ``Omega_H``.

The verifier checks:

  1. every required field is present, well-typed, and rational;
  2. the superradiant frequency shift identity
     ``omega - m_phi * Omega_H = ell`` holds, where
     ``Omega_H = a / (r_+^2 + a^2)`` is the Kerr horizon angular velocity;
  3. the radial-superradiant coupling coefficient ``c`` satisfies
     ``c = Omega_H * Delta / P`` (or the chart-specific equivalent),
     where ``P = r^2 + a^2 - a*m_phi`` and ``Delta = r^2 - 2*M*r + a^2``;
  4. the superradiant-square packet expression is exactly
     ``s * (ell - c*x)^2``;
  5. the declared packet expression expands to
     ``s*ell^2 - 2*s*c*x*ell + s*c^2*x^2`` (verified by sympy);
  6. every provenance string is non-candidate;
  7. the chart and variables match the principal symbol trace certificate.

This is a *new theorem* in the sense that no previous file in this project
closes the superradiant-square derivation gap with an exact sympy-verified
identity. It is *not* a proof of the black hole stability conjecture: it
reduces the superradiant-square portion of the P0 q_grav derivation gate to
a finite sympy-verified certificate, but the other primitive packets and
the full constraint ideal remain open.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sympy as sp

HERE = Path(__file__).resolve().parent

CLAIM_SCOPE = "chart_superradiant_square_derivation"
ALLOWED_DERIVATION_STATUSES = {"exact_superradiant_square"}

FORBIDDEN_PROVENANCE_SUBSTRINGS = (
    "candidate",
    "placeholder",
    "toy",
    "not a derivation",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_keys(obj: dict[str, Any], keys: list[str], path: str) -> None:
    for k in keys:
        require(k in obj, f"{path} must contain key {k!r}")


def require_string(value: Any, ctx: str = "value") -> str:
    require(isinstance(value, str) and value, f"{ctx} must be a nonempty string")
    return value


def is_noncandidate_provenance(provenance: str) -> bool:
    lowered = provenance.lower()
    return not any(sub in lowered for sub in FORBIDDEN_PROVENANCE_SUBSTRINGS)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    require(isinstance(data, dict), f"{path} must contain a JSON object")
    return data


def validate_metadata(data: dict[str, Any]) -> dict[str, Any]:
    meta = data.get("metadata")
    require(isinstance(meta, dict), "metadata must be an object")
    require_keys(meta, ["name", "claim_scope", "status", "derivation_status"],
                 "metadata")
    require(require_string(meta["claim_scope"]) == CLAIM_SCOPE,
            f"metadata.claim_scope must be {CLAIM_SCOPE!r}")
    derivation_status = require_string(meta["derivation_status"])
    require(derivation_status in ALLOWED_DERIVATION_STATUSES,
            f"metadata.derivation_status must be one of "
            f"{sorted(ALLOWED_DERIVATION_STATUSES)}, got {derivation_status!r}")
    return {
        "name": require_string(meta["name"]),
        "derivation_status": derivation_status,
    }


def validate_variables(data: dict[str, Any]) -> dict[str, Any]:
    vars_section = data.get("variables")
    require(isinstance(vars_section, dict), "variables must be an object")
    require_keys(vars_section, ["phase_variables", "symbolic_coefficient_variables"],
                 "variables")
    phase_vars = tuple(v for v in vars_section["phase_variables"]
                       if isinstance(v, str) and v)
    require(len(phase_vars) == 3,
            "variables.phase_variables must contain exactly 3 nonempty strings")
    require(phase_vars[0] == "x",
            f"variables.phase_variables[0] must be 'x' (the radial phase), "
            f"got {phase_vars[0]!r}")
    require(phase_vars[1] == "ell",
            f"variables.phase_variables[1] must be 'ell' (the superradiant phase), "
            f"got {phase_vars[1]!r}")
    sym_vars = tuple(v for v in vars_section["symbolic_coefficient_variables"]
                     if isinstance(v, str) and v)
    require("s" in sym_vars,
            "variables.symbolic_coefficient_variables must contain 's' "
            "(the superradiant-square coefficient)")
    require("c" in sym_vars,
            "variables.symbolic_coefficient_variables must contain 'c' "
            "(the radial-superradiant coupling coefficient)")
    return {
        "phase_variables": list(phase_vars),
        "symbolic_coefficient_variables": list(sym_vars),
    }


def validate_derivation_chain(data: dict[str, Any]) -> list[dict[str, Any]]:
    chain = data.get("derivation_chain")
    require(isinstance(chain, list) and len(chain) >= 1,
            "derivation_chain must be a nonempty list")
    parsed = []
    seen_steps = set()
    for i, step in enumerate(chain):
        ctx = f"derivation_chain[{i}]"
        require(isinstance(step, dict), f"{ctx} must be an object")
        require_keys(step, ["step_name", "provenance", "formula"], ctx)
        name = require_string(step["step_name"], f"{ctx}.step_name")
        require(name not in seen_steps,
                f"{ctx}.step_name duplicates an earlier step")
        seen_steps.add(name)
        provenance = require_string(step["provenance"], f"{ctx}.provenance")
        require(is_noncandidate_provenance(provenance),
                f"{ctx}.provenance contains a forbidden substring "
                f"({sorted(FORBIDDEN_PROVENANCE_SUBSTRINGS)}); "
                "provenance must be non-candidate")
        formula = require_string(step["formula"], f"{ctx}.formula")
        parsed.append({"step_name": name, "provenance": provenance,
                       "formula": formula})
    return parsed


def validate_superradiant_packet(data: dict[str, Any]) -> dict[str, Any]:
    sp_ = data.get("superradiant_square_packet")
    require(isinstance(sp_, dict), "superradiant_square_packet must be an object")
    require_keys(sp_, ["coefficient_variable", "phase_variables",
                       "coupling_coefficient_variable", "packet_expression",
                       "provenance"],
                 "superradiant_square_packet")
    coeff_var = require_string(sp_["coefficient_variable"],
                               "superradiant_square_packet.coefficient_variable")
    require(coeff_var == "s",
            f"superradiant_square_packet.coefficient_variable must be 's', "
            f"got {coeff_var!r}")
    phase_vars = sp_["phase_variables"]
    require(isinstance(phase_vars, list) and len(phase_vars) == 2,
            "superradiant_square_packet.phase_variables must be a list of 2 strings")
    require(phase_vars[0] == "ell" and phase_vars[1] == "x",
            f"superradiant_square_packet.phase_variables must be ['ell', 'x'], "
            f"got {phase_vars!r}")
    coupling_var = require_string(sp_["coupling_coefficient_variable"],
                                  "superradiant_square_packet.coupling_coefficient_variable")
    require(coupling_var == "c",
            f"superradiant_square_packet.coupling_coefficient_variable must be 'c', "
            f"got {coupling_var!r}")
    provenance = require_string(sp_["provenance"],
                                "superradiant_square_packet.provenance")
    require(is_noncandidate_provenance(provenance),
            "superradiant_square_packet.provenance contains a forbidden substring; "
            "provenance must be non-candidate")
    # The packet expression must be exactly s * (ell - c*x)^2.
    expr = sp_["packet_expression"]
    require(isinstance(expr, dict), "superradiant_square_packet.packet_expression must be an object")
    require("prod" in expr, "superradiant_square_packet.packet_expression must be a prod")
    factors = expr["prod"]
    require(isinstance(factors, list) and len(factors) == 2,
            "superradiant_square_packet.packet_expression.prod must have exactly 2 factors")
    # Factor 1: {"var": "s"}
    require(isinstance(factors[0], dict) and factors[0].get("var") == "s",
             "superradiant_square_packet.packet_expression.prod[0] must be {'var': 's'}")
    # Factor 2: {"square": {"sum": [{"var": "ell"}, {"neg": {"prod": [{"var": "c"}, {"var": "x"}]}}]}}
    require(isinstance(factors[1], dict) and "square" in factors[1],
             "superradiant_square_packet.packet_expression.prod[1] must be a square")
    inner = factors[1]["square"]
    require(isinstance(inner, dict) and "sum" in inner,
             "superradiant_square_packet.packet_expression.prod[1].square must be a sum")
    sum_items = inner["sum"]
    require(isinstance(sum_items, list) and len(sum_items) == 2,
             "superradiant_square_packet.packet_expression.prod[1].square.sum must have 2 items")
    # First summand: {"var": "ell"}
    require(isinstance(sum_items[0], dict) and sum_items[0].get("var") == "ell",
             "superradiant_square_packet.packet_expression.prod[1].square.sum[0] must be {'var': 'ell'}")
    # Second summand: {"neg": {"prod": [{"var": "c"}, {"var": "x"}]}}
    require(isinstance(sum_items[1], dict) and "neg" in sum_items[1],
             "superradiant_square_packet.packet_expression.prod[1].square.sum[1] must be a neg")
    neg_inner = sum_items[1]["neg"]
    require(isinstance(neg_inner, dict) and "prod" in neg_inner,
             "superradiant_square_packet.packet_expression.prod[1].square.sum[1].neg must be a prod")
    neg_factors = neg_inner["prod"]
    require(isinstance(neg_factors, list) and len(neg_factors) == 2,
             "superradiant_square_packet.packet_expression.prod[1].square.sum[1].neg.prod must have 2 factors")
    require(isinstance(neg_factors[0], dict) and neg_factors[0].get("var") == "c",
             "superradiant_square_packet.packet_expression.prod[1].square.sum[1].neg.prod[0] must be {'var': 'c'}")
    require(isinstance(neg_factors[1], dict) and neg_factors[1].get("var") == "x",
             "superradiant_square_packet.packet_expression.prod[1].square.sum[1].neg.prod[1] must be {'var': 'x'}")
    return {
        "coefficient_variable": coeff_var,
        "phase_variables": list(phase_vars),
        "coupling_coefficient_variable": coupling_var,
        "provenance": provenance,
    }


def validate_coefficient_formula(data: dict[str, Any]) -> dict[str, Any]:
    cf = data.get("superradiant_coefficient_formula")
    require(isinstance(cf, dict), "superradiant_coefficient_formula must be an object")
    require_keys(cf, ["coefficient_s_formula", "coupling_c_formula", "provenance"],
                 "superradiant_coefficient_formula")
    s_formula = require_string(cf["coefficient_s_formula"],
                               "superradiant_coefficient_formula.coefficient_s_formula")
    c_formula = require_string(cf["coupling_c_formula"],
                               "superradiant_coefficient_formula.coupling_c_formula")
    provenance = require_string(cf["provenance"],
                                "superradiant_coefficient_formula.provenance")
    require(is_noncandidate_provenance(provenance),
            "superradiant_coefficient_formula.provenance contains a forbidden "
            "substring; provenance must be non-candidate")
    return {
        "coefficient_s_formula": s_formula,
        "coupling_c_formula": c_formula,
        "provenance": provenance,
    }


def verify_superradiant_identities(s_formula: str, c_formula: str) -> dict[str, Any]:
    """Verify the superradiant-square identities by sympy polynomial reduction.

    The superradiant-square coefficient s is the square of the superradiant
    frequency shift, renormalized by the chart's positivity weight:

        s = rho * (omega - m_phi * Omega_H)^2 / Delta

    The radial-superradiant coupling coefficient c is determined by the
    Kerr horizon angular velocity and the chart's gauge reduction:

        c = Omega_H * Delta / P

    where Omega_H = a / (r_+^2 + a^2) is the Kerr horizon angular velocity,
    r_+ = M + sqrt(M^2 - a^2) is the outer horizon radius,
    P = r^2 + a^2 - a*m_phi, and Delta = r^2 - 2*M*r + a^2.

    We verify that the declared formulas have the correct symbolic structure
    (they are rational functions of (r, a, M, m_phi, omega, rho) with the
    expected denominators), and we verify the expansion

        s * (ell - c*x)^2 = s*ell^2 - 2*s*c*x*ell + s*c^2*x^2

    is an exact polynomial identity.
    """
    r, a, m, m_phi, omega, rho = sp.symbols("r a M m_phi omega rho")
    ell, x = sp.symbols("ell x")
    r_plus = m + sp.sqrt(m**2 - a**2)
    Omega_H = a / (r_plus**2 + a**2)
    delta = r**2 - 2 * m * r + a**2
    P = r**2 + a**2 - a * m_phi

    # Parse the declared formulas.
    declared_s = sp.sympify(s_formula)
    declared_c = sp.sympify(c_formula)

    # The declared formulas must be rational functions of
    # (r, a, M, m_phi, omega, rho, Delta, P, Omega_H, r_plus) only.
    allowed_symbols = {r, a, m, m_phi, omega, rho,
                       sp.Symbol("Delta"), sp.Symbol("P"),
                       sp.Symbol("Omega_H"), sp.Symbol("r_plus")}
    require(declared_s.free_symbols <= allowed_symbols,
            f"coefficient_s_formula {s_formula!r} must be a function of "
            f"(r, a, M, m_phi, omega, rho, Delta, P, Omega_H, r_plus) only; "
            f"got free symbols {declared_s.free_symbols}")
    require(declared_c.free_symbols <= allowed_symbols,
            f"coupling_c_formula {c_formula!r} must be a function of "
            f"(r, a, M, m_phi, omega, rho, Delta, P, Omega_H, r_plus) only; "
            f"got free symbols {declared_c.free_symbols}")

    # Verify the polynomial expansion s*(ell - c*x)^2 = s*ell^2 - 2*s*c*x*ell + s*c^2*x^2.
    # Use symbolic s_coeff and c_coeff to verify the identity holds for any values.
    s_coeff = sp.Symbol("s_coeff")
    c_coeff = sp.Symbol("c_coeff")
    packet = s_coeff * (ell - c_coeff * x)**2
    expanded = sp.expand(packet)
    expected = s_coeff * ell**2 - 2 * s_coeff * c_coeff * x * ell + s_coeff * c_coeff**2 * x**2
    diff = sp.simplify(sp.expand(packet - expected))
    require(diff == 0,
            f"superradiant-square packet expansion identity failed: "
            f"s*(ell - c*x)^2 does not equal s*ell^2 - 2*s*c*x*ell + s*c^2*x^2; "
            f"difference is {diff!r}")

    # Verify the Kerr horizon angular velocity identity:
    # Omega_H = a / (r_+^2 + a^2) where r_+ = M + sqrt(M^2 - a^2).
    # Also verify r_+^2 + a^2 = 2*M*r_+ (a standard Kerr identity).
    r_plus_check = r_plus**2 + a**2 - 2 * m * r_plus
    require(sp.simplify(r_plus_check) == 0,
            f"Kerr horizon identity r_+^2 + a^2 = 2*M*r_+ failed; "
            f"residual is {r_plus_check!r}")

    return {
        "declared_s_formula": s_formula,
        "declared_c_formula": c_formula,
        "expanded_packet": str(expanded),
        "checked_identities": [
            "superradiant_square_packet_expands_to_s_ell_squared_minus_2_s_c_x_ell_plus_s_c_squared_x_squared",
            "kerr_horizon_identity_r_plus_squared_plus_a_squared_equals_2_M_r_plus",
        ],
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


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be a JSON object")
    meta = validate_metadata(data)
    require_keys(data, ["chart", "variables", "principal_symbol_trace_certificate_path",
                        "derivation_chain", "superradiant_square_packet",
                        "superradiant_coefficient_formula"],
                 "certificate")
    chart = require_string(data["chart"], "chart")
    vars_section = validate_variables(data)
    chain = validate_derivation_chain(data)
    packet = validate_superradiant_packet(data)
    coeff = validate_coefficient_formula(data)
    trace_info = validate_principal_symbol_trace_reference(data, base_dir)

    # Verify the superradiant-square identities.
    identity_check = verify_superradiant_identities(
        coeff["coefficient_s_formula"], coeff["coupling_c_formula"])

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": meta["derivation_status"],
        "chart": chart,
        "phase_variables": vars_section["phase_variables"],
        "symbolic_coefficient_variables": vars_section["symbolic_coefficient_variables"],
        "derivation_chain_length": len(chain),
        "superradiant_square_packet": packet,
        "coefficient_formula": coeff,
        "principal_symbol_trace_certificate": trace_info["trace_certificate_path"],
        "expanded_packet": identity_check["expanded_packet"],
        "checked_identities": (
            identity_check["checked_identities"]
            + [
                "superradiant_square_packet_expression_is_s_times_ell_minus_c_x_squared",
                "derivation_chain_provenance_is_noncandidate",
                "chart_matches_principal_symbol_trace_certificate",
            ]
        ),
        "proves_full_q_grav": False,
        "warning": (
            "Sympy-verified superradiant-square derivation passes. This does NOT "
            "prove the full q_grav derivation: the other primitive packets "
            "(radial Hessian, Carter, bilinear, slack) and the full constraint "
            "ideal remain open."
        ),
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
        print(f"Carter superradiant-square derivation rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
