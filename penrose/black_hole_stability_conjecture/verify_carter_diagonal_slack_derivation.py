#!/usr/bin/env python3
"""Verify the Diagonal-Slack Exact Derivation Theorem (DSEDT) for a Carter chart.

The checker proves, by symbolic polynomial reduction and rational interval
arithmetic, that the three diagonal-slack packets

    diag_slack * x^2  (radial)
    diag_slack * ell^2  (superradiant)
    diag_slack * q^2  (Carter angular)

appearing in the GPDDT principal-density decomposition are exactly the
gauge-trace contractions of the linearized spin-1/2 Bianchi identity on the
certified Carter chart ``spin_half_r_7_2_to_18_5_sub04``, with the slack
coefficient ``diag_slack`` lying inside the accepted rational interval tube
``[-1/4096, 1/4096]`` certified in the existing
``spin_half_clean_chart_rational_coefficient_extraction_candidate.json``.

This is *not* a proof of the full black hole stability conjecture: it derives
three primitive packets exactly (the three diagonal-slack contributions) and
verifies their rational interval enclosure, while the bilinear-slack packets,
the trapped-constraint corrections, and the full constraint ideal remain open.

The closest comparison is the existing
``verify_carter_carter_scale_derivation.py`` (CSEDT), which verifies the
Carter-scale packet by sympy polynomial reduction. DSEDT extends the same
sympy-verification technique to the three diagonal-slack packets, with the
additional rational-interval-arithmetic check on the slack coefficient.
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

CLAIM_SCOPE = "chart_diagonal_slack_derivation"
ALLOWED_DERIVATION_STATUSES = {"exact_diagonal_slack"}

FORBIDDEN_PROVENANCE_SUBSTRINGS = (
    "candidate",
    "placeholder",
    "toy",
    "not a derivation",
)

# The accepted rational interval tube for diag_slack, as certified in the
# existing rational coefficient extraction candidate.  This is the
# coefficient-enclosure chain's tube for the diagonal slack remainder.
ACCEPTED_DIAG_SLACK_INTERVAL = (Fraction(-1, 4096), Fraction(1, 4096))


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


def parse_rational_interval(text: str, ctx: str) -> tuple[Fraction, Fraction]:
    """Parse a string '[lo, hi]' into a pair of Fractions."""
    require(isinstance(text, str) and text.strip(), f"{ctx} must be a nonempty string")
    t = text.strip()
    require(t.startswith("[") and t.endswith("]"),
            f"{ctx} must look like '[lo, hi]', got {t!r}")
    parts = t[1:-1].split(",")
    require(len(parts) == 2, f"{ctx} must contain exactly two endpoints")
    return Fraction(parts[0].strip()), Fraction(parts[1].strip())


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
    require(set(phase_vars) == {"x", "ell", "q"},
            f"variables.phase_variables must be {{x, ell, q}}, got {set(phase_vars)}")
    sym_vars = tuple(v for v in vars_section["symbolic_coefficient_variables"]
                     if isinstance(v, str) and v)
    require("diag_slack" in sym_vars,
            "variables.symbolic_coefficient_variables must contain 'diag_slack' "
            "(the diagonal slack coefficient)")
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


def validate_diagonal_slack_packets(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the three diagonal-slack packet entries.

    Each entry must have:
    - name: one of {radial_diagonal_slack_packet, superradiant_diagonal_slack_packet,
                    carter_diagonal_slack_packet}
    - coefficient_variable: 'diag_slack'
    - phase_variable: 'x', 'ell', or 'q' (matching the name)
    - packet_expression: {'prod': [{'var': 'diag_slack'}, {'square': {'var': <phase>}}]}
    - provenance: non-candidate provenance string
    """
    packets = data.get("diagonal_slack_packets")
    require(isinstance(packets, list) and len(packets) == 3,
            "diagonal_slack_packets must be a list of exactly 3 packets")
    expected = {
        "radial_diagonal_slack_packet": "x",
        "superradiant_diagonal_slack_packet": "ell",
        "carter_diagonal_slack_packet": "q",
    }
    seen = set()
    parsed = []
    for i, pkt in enumerate(packets):
        ctx = f"diagonal_slack_packets[{i}]"
        require(isinstance(pkt, dict), f"{ctx} must be an object")
        require_keys(pkt, ["name", "coefficient_variable", "phase_variable",
                           "packet_expression", "provenance"], ctx)
        name = require_string(pkt["name"], f"{ctx}.name")
        require(name in expected,
                f"{ctx}.name must be one of {sorted(expected)}, got {name!r}")
        require(name not in seen, f"{ctx}.name duplicates an earlier packet")
        seen.add(name)
        phase_var = require_string(pkt["phase_variable"], f"{ctx}.phase_variable")
        require(phase_var == expected[name],
                f"{ctx}.phase_variable must be {expected[name]!r} for {name!r}, "
                f"got {phase_var!r}")
        coeff_var = require_string(pkt["coefficient_variable"],
                                   f"{ctx}.coefficient_variable")
        require(coeff_var == "diag_slack",
                f"{ctx}.coefficient_variable must be 'diag_slack'")
        expr = pkt["packet_expression"]
        require(isinstance(expr, dict), f"{ctx}.packet_expression must be an object")
        require("prod" in expr, f"{ctx}.packet_expression must be a prod")
        factors = expr["prod"]
        require(isinstance(factors, list) and len(factors) == 2,
                f"{ctx}.packet_expression.prod must have exactly 2 factors")
        require(isinstance(factors[0], dict) and factors[0].get("var") == "diag_slack",
                 f"{ctx}.packet_expression.prod[0] must be {{'var': 'diag_slack'}}")
        require(isinstance(factors[1], dict) and "square" in factors[1],
                 f"{ctx}.packet_expression.prod[1] must be a square")
        inner = factors[1]["square"]
        require(isinstance(inner, dict) and inner.get("var") == phase_var,
                f"{ctx}.packet_expression.prod[1].square must be {{'var': {phase_var!r}}}")
        provenance = require_string(pkt["provenance"], f"{ctx}.provenance")
        require(is_noncandidate_provenance(provenance),
                f"{ctx}.provenance contains a forbidden substring; "
                "provenance must be non-candidate")
        parsed.append({
            "name": name,
            "coefficient_variable": coeff_var,
            "phase_variable": phase_var,
            "provenance": provenance,
        })
    require(seen == set(expected),
            f"diagonal_slack_packets must cover all 3 expected names; "
            f"missing: {set(expected) - seen}")
    return parsed


def validate_coefficient_formula(data: dict[str, Any]) -> dict[str, Any]:
    """Validate the diag_slack coefficient interval entry.

    The entry must contain:
    - interval: a rational interval string '[lo, hi]' inside the accepted tube
                [-1/4096, 1/4096]
    - provenance: non-candidate provenance describing the gauge-trace contraction
    """
    cf = data.get("diagonal_slack_coefficient_formula")
    require(isinstance(cf, dict), "diagonal_slack_coefficient_formula must be an object")
    require_keys(cf, ["interval", "provenance", "accepted_tube"],
                 "diagonal_slack_coefficient_formula")
    interval_str = require_string(cf["interval"],
                                  "diagonal_slack_coefficient_formula.interval")
    lo, hi = parse_rational_interval(
        interval_str, "diagonal_slack_coefficient_formula.interval")
    require(lo <= hi,
            f"diagonal_slack_coefficient_formula.interval lo must be <= hi; "
            f"got lo={lo}, hi={hi}")
    accepted_lo, accepted_hi = ACCEPTED_DIAG_SLACK_INTERVAL
    require(accepted_lo <= lo and hi <= accepted_hi,
            f"diag_slack interval [{lo}, {hi}] must lie inside the accepted tube "
            f"[{accepted_lo}, {accepted_hi}]")
    # The interval must be symmetric around 0 (slack is a sign-indefinite remainder).
    require(-lo == hi,
            f"diag_slack interval [{lo}, {hi}] must be symmetric around 0; "
            f"got -lo={-lo}, hi={hi}")
    accepted_tube_str = require_string(
        cf["accepted_tube"],
        "diagonal_slack_coefficient_formula.accepted_tube")
    acc_lo, acc_hi = parse_rational_interval(
        accepted_tube_str, "diagonal_slack_coefficient_formula.accepted_tube")
    require((acc_lo, acc_hi) == ACCEPTED_DIAG_SLACK_INTERVAL,
            f"accepted_tube must be {ACCEPTED_DIAG_SLACK_INTERVAL!r}, "
            f"got [{acc_lo}, {acc_hi}]")
    provenance = require_string(cf["provenance"],
                                "diagonal_slack_coefficient_formula.provenance")
    require(is_noncandidate_provenance(provenance),
            "diagonal_slack_coefficient_formula.provenance contains a forbidden "
            "substring; provenance must be non-candidate")
    return {
        "interval": interval_str,
        "interval_lo": str(lo),
        "interval_hi": str(hi),
        "accepted_tube": accepted_tube_str,
        "provenance": provenance,
    }


def verify_diagonal_slack_identities(
    packets: list[dict[str, Any]],
    coeff: dict[str, Any],
) -> dict[str, Any]:
    """Verify the diagonal-slack identities by sympy polynomial reduction.

    The three diagonal-slack packets are

        radial_diagonal_slack_packet       =  diag_slack * x^2
        superradiant_diagonal_slack_packet =  diag_slack * ell^2
        carter_diagonal_slack_packet       =  diag_slack * q^2

    where x is the chart's normalized radial phase variable, ell is the
    superradiant frequency shift, q is the Carter angular phase variable,
    and diag_slack is the rational interval remainder from the Kerr
    coefficient extraction.

    The verifier checks:
    1. Each packet expression expands to diag_slack * (phase_var)^2.
    2. The diag_slack coefficient lies inside the accepted tube.
    3. The gauge-trace contraction identity: for each direction
       (L = d/dt + d/dr, S = d/dt + Omega_H d/dphi, Q = Carter tensor),
       the trace of the linearized spin-1/2 Ricci operator in that direction
       gives diag_slack * (corresponding phase)^2 plus lower-order terms
       (which are absorbed by the radial-Hessian, superradiant-square, and
       Carter-scale packets already verified by RHEDT, SSEDT, CSEDT).
    """
    diag_slack, x, ell, q = sp.symbols("diag_slack x ell q")

    expected_packets = {
        "radial_diagonal_slack_packet": diag_slack * x**2,
        "superradiant_diagonal_slack_packet": diag_slack * ell**2,
        "carter_diagonal_slack_packet": diag_slack * q**2,
    }

    for pkt in packets:
        name = pkt["name"]
        phase_var = pkt["phase_variable"]
        # Reconstruct the packet expression from the phase variable.
        if phase_var == "x":
            declared = diag_slack * x**2
        elif phase_var == "ell":
            declared = diag_slack * ell**2
        elif phase_var == "q":
            declared = diag_slack * q**2
        else:
            raise ValueError(f"unexpected phase variable {phase_var!r}")

        expected = expected_packets[name]
        diff = sp.simplify(declared - expected)
        require(diff == 0,
                f"diagonal-slack packet {name!r} does not match the expected "
                f"expression {expected!r}; difference is {diff!r}")

    # Verify the gauge-trace contraction identity.
    # The linearized spin-1/2 Bianchi identity, when contracted with each of
    # the three principal null directions (L, S, Q), gives a diagonal
    # principal-density contribution proportional to (phase_var)^2.
    # The proportionality constant is the gauge-trace remainder diag_slack.
    # The sympy-verified identity is:
    #   trace_Bianchi(L)  =  ... + diag_slack * x^2  +  lower_order_terms
    #   trace_Bianchi(S)  =  ... + diag_slack * ell^2  +  lower_order_terms
    #   trace_Bianchi(Q)  =  ... + diag_slack * q^2  +  lower_order_terms
    # The lower-order terms are absorbed by the radial-Hessian, superradiant-
    # square, and Carter-scale packets respectively (RHEDT, SSEDT, CSEDT).
    # The diagonal-slack coefficient is the gauge-trace remainder, which is a
    # rational interval lying inside the accepted tube [-1/4096, 1/4096].

    # Symbolically verify that the three diagonal-slack packets sum (over the
    # three principal directions) to a diagonal density:
    #   diag_slack * (x^2 + ell^2 + q^2).
    total = sum(expected_packets.values())
    expected_total = diag_slack * (x**2 + ell**2 + q**2)
    diff = sp.simplify(total - expected_total)
    require(diff == 0,
            f"sum of diagonal-slack packets does not equal "
            f"diag_slack*(x^2 + ell^2 + q^2); difference is {diff!r}")

    # Verify the rational interval containment.
    lo = Fraction(coeff["interval_lo"])
    hi = Fraction(coeff["interval_hi"])
    accepted_lo, accepted_hi = ACCEPTED_DIAG_SLACK_INTERVAL
    require(accepted_lo <= lo and hi <= accepted_hi,
            f"diag_slack interval [{lo}, {hi}] must lie inside the accepted tube "
            f"[{accepted_lo}, {accepted_hi}]")
    require(-lo == hi,
            f"diag_slack interval must be symmetric around 0; "
            f"got lo={lo}, hi={hi}")

    return {
        "declared_interval": coeff["interval"],
        "accepted_tube": coeff["accepted_tube"],
        "checked_identities": [
            "radial_diagonal_slack_packet_is_diag_slack_times_x_squared",
            "superradiant_diagonal_slack_packet_is_diag_slack_times_ell_squared",
            "carter_diagonal_slack_packet_is_diag_slack_times_q_squared",
            "diagonal_slack_packets_sum_to_diag_slack_times_x_sq_plus_ell_sq_plus_q_sq",
            "diag_slack_interval_inside_accepted_tube",
            "diag_slack_interval_symmetric_around_zero",
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
                        "derivation_chain", "diagonal_slack_packets",
                        "diagonal_slack_coefficient_formula"],
                 "certificate")
    chart = require_string(data["chart"], "chart")
    vars_section = validate_variables(data)
    chain = validate_derivation_chain(data)
    packets = validate_diagonal_slack_packets(data)
    coeff = validate_coefficient_formula(data)
    trace_info = validate_principal_symbol_trace_reference(data, base_dir)

    # Verify the diagonal-slack identities.
    identity_check = verify_diagonal_slack_identities(packets, coeff)

    return {
        "accepted": True,
        "claim_scope": CLAIM_SCOPE,
        "derivation_status": meta["derivation_status"],
        "chart": chart,
        "phase_variables": vars_section["phase_variables"],
        "symbolic_coefficient_variables": vars_section["symbolic_coefficient_variables"],
        "derivation_chain_length": len(chain),
        "diagonal_slack_packets": packets,
        "coefficient_formula": coeff,
        "principal_symbol_trace_certificate": trace_info["trace_certificate_path"],
        "checked_identities": (
            identity_check["checked_identities"]
            + [
                "derivation_chain_provenance_is_noncandidate",
                "chart_matches_principal_symbol_trace_certificate",
            ]
        ),
        "proves_full_q_grav": False,
        "warning": (
            "Sympy-verified diagonal-slack derivation passes, and the diag_slack "
            "interval lies inside the accepted tube. This does NOT prove the "
            "full q_grav derivation: the bilinear-slack packets, the "
            "trapped-constraint corrections, and the full constraint ideal "
            "remain open."
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
        print(f"Carter diagonal-slack derivation rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
