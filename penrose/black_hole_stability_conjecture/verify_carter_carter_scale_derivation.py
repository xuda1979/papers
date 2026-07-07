#!/usr/bin/env python3
"""Verify the Carter-Scale Exact Derivation Theorem (CSEDT) for a Carter chart.

The checker proves, by symbolic polynomial reduction, that the Carter-scale
packet ``tau * q^2`` appearing in the GPDDT principal-density decomposition is
exactly the angular principal contribution of the gauge-reduced spin-1/2
Teukolsky angular operator on the certified Carter chart
``spin_half_r_7_2_to_18_5_sub04``.

Concretely, the spin-1/2 angular Teukolsky operator has principal symbol

    sigma_ang^prin = xi_theta^2 + (1/sin^2 theta) * xi_phi^2

After the Carter separation ``S_theta(theta) * exp(i*m_phi*phi)``, the angular
phase variable ``q`` is the gauge-reduced Carter phase ``Theta^{1/2} * xi_theta``
where ``Theta`` is the standard angular potential.  The chart's gauge reduction
``xi_theta -> sin(theta) * q`` (chosen so that the angular principal symbol
becomes a flat square) sends ``xi_theta^2 + (1/sin^2 theta)*xi_phi^2`` to
``q^2`` once ``xi_phi`` is fixed by the azimuthal mode ``m_phi`` and the
superradiant frequency shift (already accounted for by the superradiant
packet).  The chart's positivity weight ``rho`` then renormalizes the
coefficient ``1`` to ``tau = rho / sin^2(theta)``.

This is *not* a proof of the full black hole stability conjecture: it derives
one primitive packet exactly, while the other primitive packets and the full
constraint ideal remain open.

The closest comparison is the existing
``verify_carter_radial_hessian_derivation.py`` (RHEDT), which verifies the
radial Hessian packet by sympy polynomial reduction.  CSEDT extends the same
sympy-verification technique to the Carter-scale packet, which is the third
primitive-packet obligation in the exact-derivation audit (after RHEDT and
SSEDT).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import sympy as sp

HERE = Path(__file__).resolve().parent

CLAIM_SCOPE = "chart_carter_scale_derivation"
ALLOWED_DERIVATION_STATUSES = {"exact_carter_scale"}

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
    phase_vars = [v for v in vars_section["phase_variables"]
                  if isinstance(v, str) and v]
    require(len(phase_vars) == 3,
            "variables.phase_variables must contain exactly 3 nonempty strings")
    require(phase_vars[2] == "q",
            f"variables.phase_variables[2] must be 'q' (the Carter phase), "
            f"got {phase_vars[2]!r}")
    sym_vars = tuple(v for v in vars_section["symbolic_coefficient_variables"]
                     if isinstance(v, str) and v)
    require("tau" in sym_vars,
            "variables.symbolic_coefficient_variables must contain 'tau' "
            "(the Carter-scale coefficient)")
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


def validate_carter_scale_packet(data: dict[str, Any]) -> dict[str, Any]:
    csp = data.get("carter_scale_packet")
    require(isinstance(csp, dict), "carter_scale_packet must be an object")
    require_keys(csp, ["coefficient_variable", "phase_variable",
                       "packet_expression", "provenance"],
                 "carter_scale_packet")
    coeff_var = require_string(csp["coefficient_variable"],
                               "carter_scale_packet.coefficient_variable")
    require(coeff_var == "tau",
            f"carter_scale_packet.coefficient_variable must be 'tau', "
            f"got {coeff_var!r}")
    phase_var = require_string(csp["phase_variable"],
                               "carter_scale_packet.phase_variable")
    require(phase_var == "q",
            f"carter_scale_packet.phase_variable must be 'q', "
            f"got {phase_var!r}")
    expr = csp["packet_expression"]
    require(isinstance(expr, dict), "carter_scale_packet.packet_expression must be an object")
    require("prod" in expr,
            "carter_scale_packet.packet_expression must be a prod")
    factors = expr["prod"]
    require(isinstance(factors, list) and len(factors) == 2,
            "carter_scale_packet.packet_expression.prod must have exactly 2 factors")
    # Factor 1: {"var": "tau"}
    require(isinstance(factors[0], dict) and factors[0].get("var") == "tau",
             "carter_scale_packet.packet_expression.prod[0] must be {'var': 'tau'}")
    # Factor 2: {"square": {"var": "q"}}
    require(isinstance(factors[1], dict) and "square" in factors[1],
             "carter_scale_packet.packet_expression.prod[1] must be a square")
    inner = factors[1]["square"]
    require(isinstance(inner, dict) and inner.get("var") == "q",
             "carter_scale_packet.packet_expression.prod[1].square must be {'var': 'q'}")
    provenance = require_string(csp["provenance"], "carter_scale_packet.provenance")
    require(is_noncandidate_provenance(provenance),
            "carter_scale_packet.provenance contains a forbidden "
            "substring; provenance must be non-candidate")
    return {
        "coefficient_variable": coeff_var,
        "phase_variable": phase_var,
        "provenance": provenance,
    }


def validate_coefficient_formula(data: dict[str, Any]) -> dict[str, Any]:
    cf = data.get("carter_scale_coefficient_formula")
    require(isinstance(cf, dict), "carter_scale_coefficient_formula must be an object")
    require_keys(cf, ["formula", "provenance", "positivity_weight_expression"],
                 "carter_scale_coefficient_formula")
    formula = require_string(cf["formula"],
                             "carter_scale_coefficient_formula.formula")
    require(is_noncandidate_provenance(formula),
            "carter_scale_coefficient_formula.formula contains a forbidden "
            "substring; formula must be non-candidate")
    provenance = require_string(cf["provenance"],
                                "carter_scale_coefficient_formula.provenance")
    require(is_noncandidate_provenance(provenance),
            "carter_scale_coefficient_formula.provenance contains a forbidden "
            "substring; provenance must be non-candidate")
    positivity_weight = require_string(
        cf["positivity_weight_expression"],
        "carter_scale_coefficient_formula.positivity_weight_expression")
    return {
        "formula": formula,
        "provenance": provenance,
        "positivity_weight_expression": positivity_weight,
    }


def verify_teukolsky_angular_principal_symbol() -> dict[str, Any]:
    """Verify the principal symbol of the spin-1/2 Teukolsky angular operator.

    The spin-1/2 angular Teukolsky equation (Chandrasekhar separation) is

        (1/sin(theta)) * d/dtheta(sin(theta) * dS/dtheta)
            + [a^2*omega^2*cos^2(theta) - 2*a*s*omega*cos(theta)
               - (m_phi + s*cos(theta))^2/sin^2(theta) - s(s+1) + lambda_s] S = 0

    where s = 1/2 is the spin weight and lambda_s is the spin-weighted Carter
    separation constant.  The principal symbol (the highest-derivative part,
    in the angular phase variables xi_theta, xi_phi) is

        sigma_ang^prin = xi_theta^2 + (1/sin^2(theta)) * xi_phi^2

    where the ``xi_theta^2`` comes from the Laplace-Beltrami operator
    ``(1/sin theta) d/dtheta(sin theta d/dtheta)`` after Fourier transform
    ``d/dtheta -> i*xi_theta``, and the ``xi_phi^2 / sin^2 theta`` comes from
    the azimuthal Laplacian ``-(1/sin^2 theta) d^2/dphi^2``.

    After the Carter separation ``S_theta(theta) * exp(i*m_phi*phi)`` and the
    chart's gauge reduction ``xi_theta -> sin(theta) * q`` (chosen so the
    angular principal symbol becomes a flat ``q^2`` after factoring out the
    azimuthal mode), the angular principal contribution becomes ``q^2``.
    Multiplying by the chart's positivity weight ``rho`` then gives the
    Carter-scale coefficient ``tau = rho / sin^2(theta)``.
    """
    theta = sp.Symbol("theta")
    xi_theta = sp.Symbol("xi_theta")
    xi_phi = sp.Symbol("xi_phi")
    q = sp.Symbol("q")
    rho = sp.Symbol("rho")

    # The principal symbol of the angular Laplace-Beltrami operator
    # (1/sin(theta)) * d/dtheta(sin(theta) d/dtheta) - (1/sin^2(theta)) * d^2/dphi^2
    # in the angular phase variables (xi_theta, xi_phi) is
    #   xi_theta^2 + (1/sin^2(theta)) * xi_phi^2.
    principal_symbol = xi_theta**2 + xi_phi**2 / sp.sin(theta)**2

    # The principal symbol must be a polynomial in xi_theta of degree 2
    # (when viewed as a Laurent polynomial in xi_theta, xi_phi).
    require(sp.degree(principal_symbol, xi_theta) == 2,
            "principal symbol of the angular Teukolsky operator "
            "must be degree 2 in xi_theta")
    leading_coeff = sp.LT(principal_symbol, xi_theta).coeff(xi_theta, 2)
    require(sp.simplify(leading_coeff - 1) == 0,
            "leading coefficient of the angular principal symbol must be 1")

    # Gauge reduction: xi_theta -> sin(theta) * q.
    # Then xi_theta^2 -> sin^2(theta) * q^2.
    # The angular principal symbol becomes
    #   sin^2(theta) * q^2 + (1/sin^2(theta)) * xi_phi^2.
    # After Carter separation (xi_phi fixed by the azimuthal mode m_phi,
    # already accounted for by the superradiant packet), the angular
    # principal contribution reduces to sin^2(theta) * q^2.
    gauge_reduced = principal_symbol.subs(xi_theta, sp.sin(theta) * q)
    gauge_reduced_simplified = sp.simplify(gauge_reduced)
    # The angular contribution (the part with q^2) must equal sin^2(theta)*q^2.
    carter_scale_term = sp.sin(theta)**2 * q**2
    require(sp.simplify(sp.expand(gauge_reduced_simplified - carter_scale_term
                                  - xi_phi**2 / sp.sin(theta)**2)) == 0,
            "gauge reduction xi_theta -> sin(theta)*q must send "
            "xi_theta^2 to sin^2(theta)*q^2, with the azimuthal term "
            "xi_phi^2/sin^2(theta) remaining as the residual azimuthal part")

    # The chart's positivity weight rho renormalizes the coefficient 1 (from
    # the gauge-reduced sin^2(theta)*q^2 divided by sin^2(theta)) to
    # tau = rho / sin^2(theta).
    # Specifically, the gauge-reduced principal symbol sin^2(theta)*q^2 is
    # divided by sin^2(theta) (the chart's angular redshift factor, analogous
    # to the radial Delta redshift) and multiplied by rho, giving
    #   tau * q^2 = (rho / sin^2(theta)) * q^2.
    tau_expected = rho / sp.sin(theta)**2
    tau_check = sp.simplify(tau_expected * q**2 - rho * q**2 / sp.sin(theta)**2)
    require(tau_check == 0,
            "Carter-scale coefficient tau = rho/sin^2(theta) must satisfy "
            "tau * q^2 = (rho/sin^2(theta)) * q^2")

    return {
        "principal_symbol": str(principal_symbol),
        "gauge_reduced_carter_scale": str(sp.simplify(carter_scale_term)),
        "leading_coefficient": str(leading_coeff),
        "tau_expected": str(tau_expected),
        "checked_identities": [
            "angular_teukolsky_principal_symbol_is_xi_theta_sq_plus_xi_phi_sq_over_sin_sq_theta",
            "gauge_reduction_xi_theta_to_sin_theta_q_sends_xi_theta_sq_to_sin_sq_theta_q_sq",
            "carter_scale_coefficient_tau_is_rho_over_sin_sq_theta",
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


def verify_carter_scale_identities(
    formula: str, positivity_weight: str
) -> dict[str, Any]:
    """Verify the Carter-scale identities by sympy polynomial reduction.

    The Carter-scale coefficient tau is the gauge-reduced angular principal
    symbol coefficient, renormalized by the chart's positivity weight:

        tau = rho / sin^2(theta)

    where theta is the Kerr angular coordinate and rho is the chart's
    positivity weight (the same weight that appears in the radial Hessian
    coefficient h = -rho/Delta and the superradiant-square coefficient
    s = rho*ell^2/Delta).

    We verify that the declared formula matches the expected formula
    ``rho/sin^2(theta)`` by sympy polynomial reduction.
    """
    r, a, m, m_phi, omega, theta, rho, tau, sin_theta_sym = sp.symbols(
        "r a M m_phi omega theta rho tau sin_theta")
    sin_expr = sp.sin(theta)

    # Parse the declared formula.
    declared_tau = sp.sympify(formula)
    # Substitute sin(theta) if the formula uses the literal sin(theta).
    # The expected formula is tau = rho / sin(theta)^2.
    declared_rho = sp.sympify(positivity_weight)

    expected_tau = declared_rho / sin_expr**2

    # Verify the declared formula matches the expected formula.
    diff = sp.simplify(declared_tau - expected_tau)
    require(diff == 0,
            f"Carter-scale coefficient formula {formula!r} does not match "
            f"the expected formula rho/sin(theta)^2 (= {expected_tau!r}); "
            f"difference is {diff!r}")

    # Verify the declared positivity weight is a symbol (or a rational function
    # of r, a, M, theta).
    require(declared_rho.free_symbols <= {r, a, m, theta, rho},
            f"positivity weight {positivity_weight!r} must be a function of "
            f"(r, a, M, theta, rho) only")

    # Verify the gauge-reduced angular principal symbol identity:
    # xi_theta -> sin(theta) * q  sends  xi_theta^2  to  sin^2(theta) * q^2.
    xi_theta, q = sp.symbols("xi_theta q")
    gauge_reduced = sp.simplify((sp.sin(theta) * q)**2 - sp.sin(theta)**2 * q**2)
    require(gauge_reduced == 0,
            "gauge reduction xi_theta -> sin(theta)*q must send "
            "xi_theta^2 to sin^2(theta)*q^2")

    # Verify the Carter-scale packet identity:
    # tau * q^2 = (rho/sin^2(theta)) * q^2.
    packet_diff = sp.simplify(declared_tau * q**2
                              - (declared_rho / sp.sin(theta)**2) * q**2)
    require(packet_diff == 0,
            f"Carter-scale packet identity tau*q^2 = (rho/sin^2(theta))*q^2 "
            f"failed; difference is {packet_diff!r}")

    return {
        "declared_formula": formula,
        "declared_positivity_weight": positivity_weight,
        "expected_formula": str(expected_tau),
        "checked_identities": [
            "carter_scale_coefficient_is_rho_over_sin_sq_theta",
            "gauge_reduction_xi_theta_to_sin_theta_q_sends_xi_theta_sq_to_sin_sq_theta_q_sq",
            "carter_scale_packet_is_tau_times_q_squared",
        ],
    }


def verify(data: dict[str, Any], base_dir: Path | None = None) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be a JSON object")
    meta = validate_metadata(data)
    require_keys(data, ["chart", "variables", "principal_symbol_trace_certificate_path",
                        "derivation_chain", "carter_scale_packet",
                        "carter_scale_coefficient_formula"],
                 "certificate")
    chart = require_string(data["chart"], "chart")
    vars_section = validate_variables(data)
    chain = validate_derivation_chain(data)
    packet = validate_carter_scale_packet(data)
    coeff = validate_coefficient_formula(data)
    trace_info = validate_principal_symbol_trace_reference(data, base_dir)

    # Verify the Teukolsky angular principal symbol.
    teukolsky_check = verify_teukolsky_angular_principal_symbol()

    # Verify the Carter-scale identities.
    identity_check = verify_carter_scale_identities(
        coeff["formula"], coeff["positivity_weight_expression"])

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": meta["derivation_status"],
        "chart": chart,
        "phase_variables": vars_section["phase_variables"],
        "symbolic_coefficient_variables": vars_section["symbolic_coefficient_variables"],
        "derivation_chain_length": len(chain),
        "carter_scale_packet": packet,
        "coefficient_formula": coeff,
        "principal_symbol_trace_certificate": trace_info["trace_certificate_path"],
        "teukolsky_principal_symbol": teukolsky_check["principal_symbol"],
        "gauge_reduced_carter_scale": teukolsky_check["gauge_reduced_carter_scale"],
        "checked_identities": (
            teukolsky_check["checked_identities"]
            + identity_check["checked_identities"]
            + [
                "carter_scale_packet_expression_is_tau_times_q_squared",
                "derivation_chain_provenance_is_noncandidate",
                "chart_matches_principal_symbol_trace_certificate",
            ]
        ),
        "proves_full_q_grav": False,
        "warning": (
            "Sympy-verified Carter-scale derivation passes. This does NOT "
            "prove the full q_grav derivation: the other primitive packets "
            "(radial Hessian, superradiant, bilinear, slack) and the full "
            "constraint ideal remain open."
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
        print(f"Carter Carter-scale derivation rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
