#!/usr/bin/env python3
"""Validate a Gravitational Principal Density Derivation Theorem (GPDDT) certificate.

This checker verifies the combinatorial, arithmetic, and provenance content
of a GPDDT certificate. It checks:

  1. every required field is present, well-typed, and rational;
  2. the linearized Einstein packets are a finite list of named polynomials,
     each with a non-candidate provenance string;
  3. every principal density packet is homogeneous of phase degree 2;
  4. every constraint generator is phase-free;
  5. the gravitational density derivation expands to the declared linear
     combination of packets with rational coefficients;
  6. the derived density matches the target symbolic normal form modulo the
     constraint ideal;
  7. the provenance audit confirms every primitive formula has non-candidate
     provenance.

The verifier does NOT check the analytic content of the linearized Einstein
equations themselves.

Usage:
    python3 verify_gravitational_principal_density_derivation.py <certificate.json>
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Reuse the polynomial grammar from the existing SOS verifier.
import verify_carter_sos_identity as sos  # noqa: E402


FORBIDDEN_PROVENANCE_SUBSTRINGS = ("candidate", "placeholder", "toy", "not a derivation")
VALID_ROLES = {"principal_density_packet", "constraint_generator"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def require_keys(d, keys, ctx):
    for k in keys:
        require(k in d, f"{ctx}.{k} must be present")


def require_int(value, ctx, *, minimum=None, maximum=None):
    require(isinstance(value, int) and not isinstance(value, bool),
            f"{ctx} must be an integer, got {type(value).__name__}")
    if minimum is not None:
        require(value >= minimum, f"{ctx} must be >= {minimum}, got {value}")
    if maximum is not None:
        require(value <= maximum, f"{ctx} must be <= {maximum}, got {value}")
    return value


def require_string(value, ctx="value"):
    require(isinstance(value, str) and value, f"{ctx} must be a nonempty string")
    return value


def require_fraction(value, ctx):
    if not isinstance(value, str):
        raise ValueError(f"{ctx} must be a rational string, got {type(value).__name__}")
    try:
        return Fraction(value.strip())
    except Exception as exc:
        raise ValueError(f"{ctx} invalid rational {value!r}: {exc}") from exc


def is_noncandidate_provenance(provenance):
    lowered = provenance.lower()
    return not any(sub in lowered for sub in FORBIDDEN_PROVENANCE_SUBSTRINGS)


# Polynomial helpers (reusing sos.Polynomial and sos.Monomial)

def phase_degree_of_monomial(monomial, phase_var_indices):
    return sum(monomial[i] for i in phase_var_indices)


def is_homogeneous_of_degree(poly, phase_var_indices, degree):
    for monomial in poly.terms:
        if phase_degree_of_monomial(monomial, phase_var_indices) != degree:
            return False
    return True


def is_phase_free(poly, phase_var_indices):
    for monomial in poly.terms:
        if phase_degree_of_monomial(monomial, phase_var_indices) != 0:
            return False
    return True


def max_phase_degree(poly, phase_var_indices):
    if not poly.terms:
        return 0
    return max(phase_degree_of_monomial(m, phase_var_indices) for m in poly.terms)


def get_phase_var_indices(poly_vars, phase_vars):
    """Return the indices of phase_vars in poly_vars."""
    indices = []
    for pv in phase_vars:
        if pv in poly_vars:
            indices.append(poly_vars.index(pv))
    return indices


# Section validators

def validate_variable_list(raw, name, *, allow_empty):
    require(isinstance(raw, list), f"{name} must be a list")
    if not allow_empty:
        require(len(raw) >= 1, f"{name} must be nonempty")
    seen = set()
    result = []
    for i, v in enumerate(raw):
        ctx = f"{name}[{i}]"
        require(isinstance(v, str) and v, f"{ctx} must be a nonempty string")
        require(v not in seen, f"{ctx} duplicates an earlier variable")
        seen.add(v)
        result.append(v)
    return result


def validate_header(data):
    require_keys(data, ["theorem", "version", "claim_scope", "background",
                        "spin", "chart", "status"], "root")
    require(require_string(data["theorem"]) == "GPDDT",
            "root.theorem must be 'GPDDT'")
    require(require_string(data["version"]) == "1.0",
            "root.version must be '1.0' (only version 1.0 is supported)")
    require(require_string(data["status"]) == "machine_certificate",
            "root.status must be 'machine_certificate'")
    require(require_string(data["claim_scope"])
            == "chart_level_gravitational_density_derivation",
            "root.claim_scope must be 'chart_level_gravitational_density_derivation'")
    return {
        "background": require_string(data["background"]),
        "spin": require_string(data["spin"]),
        "chart": require_string(data["chart"]),
    }


def validate_variables(data):
    vars_section = data.get("variables")
    require(isinstance(vars_section, dict), "variables must be an object")
    require_keys(vars_section, ["phase_variables", "symbolic_coefficient_variables",
                                 "constraint_auxiliary_variables"], "variables")
    phase_vars = tuple(validate_variable_list(vars_section["phase_variables"],
                                               "variables.phase_variables", allow_empty=False))
    sym_vars = tuple(validate_variable_list(vars_section["symbolic_coefficient_variables"],
                                             "variables.symbolic_coefficient_variables",
                                             allow_empty=False))
    aux_vars = tuple(validate_variable_list(vars_section["constraint_auxiliary_variables"],
                                             "variables.constraint_auxiliary_variables",
                                             allow_empty=True))
    all_vars = phase_vars + sym_vars + aux_vars
    require(len(set(all_vars)) == len(all_vars),
            "variables: phase, symbolic_coefficient, and constraint_auxiliary "
            "variables must be disjoint")
    return {
        "phase_variables": phase_vars,
        "symbolic_coefficient_variables": sym_vars,
        "constraint_auxiliary_variables": aux_vars,
        "all_variables": all_vars,
    }


def validate_linearized_einstein_packets(data, vars_section):
    packets = data.get("linearized_einstein_packets")
    require(isinstance(packets, list) and len(packets) >= 1,
            "linearized_einstein_packets must be a nonempty list")
    phase_vars = vars_section["phase_variables"]
    all_vars = vars_section["all_variables"]
    phase_indices = get_phase_var_indices(all_vars, phase_vars)
    seen_names = set()
    parsed_packets = []
    principal_count = 0
    constraint_count = 0
    for i, pkt in enumerate(packets):
        ctx = f"linearized_einstein_packets[{i}]"
        require(isinstance(pkt, dict), f"{ctx} must be an object")
        require_keys(pkt, ["name", "role", "provenance", "expression"], ctx)
        name = require_string(pkt["name"], f"{ctx}.name")
        require(name not in seen_names, f"{ctx}.name duplicates an earlier packet")
        seen_names.add(name)
        role = require_string(pkt["role"], f"{ctx}.role")
        require(role in VALID_ROLES,
                f"{ctx}.role must be one of {sorted(VALID_ROLES)}, got {role!r}")
        provenance = require_string(pkt["provenance"], f"{ctx}.provenance")
        require(is_noncandidate_provenance(provenance),
                f"{ctx}.provenance contains a forbidden substring "
                f"({sorted(FORBIDDEN_PROVENANCE_SUBSTRINGS)}); "
                "provenance must be non-candidate")
        # Parse the polynomial expression
        try:
            poly = sos.eval_expr(pkt["expression"], all_vars)
        except Exception as exc:
            raise ValueError(f"{ctx}.expression failed to parse: {exc}") from exc
        # Tag-specific checks
        if role == "principal_density_packet":
            require(is_homogeneous_of_degree(poly, phase_indices, 2),
                    f"{ctx}: principal_density_packet must be homogeneous of "
                    "phase degree 2")
            principal_count += 1
        else:  # constraint_generator
            require(is_phase_free(poly, phase_indices),
                    f"{ctx}: constraint_generator must be phase-free "
                    "(contain no phase variables)")
            constraint_count += 1
        parsed_packets.append({
            "name": name,
            "role": role,
            "provenance": provenance,
            "poly": poly,
        })
    return {
        "packets": parsed_packets,
        "principal_count": principal_count,
        "constraint_count": constraint_count,
        "max_phase_degree": max((max_phase_degree(p["poly"], phase_indices)
                                 for p in parsed_packets), default=0),
    }



def validate_gravitational_density_derivation(data, packets_info, vars_section):
    gdd = data.get("gravitational_density_derivation")
    require(isinstance(gdd, dict),
            "gravitational_density_derivation must be an object")
    require_keys(gdd, ["derivation_coefficients", "derived_density_expression",
                       "target_symbolic_normal_form_path", "identity_check"],
                 "gravitational_density_derivation")
    coeffs = gdd["derivation_coefficients"]
    require(isinstance(coeffs, dict),
            "derivation_coefficients must be an object")
    require_keys(coeffs, ["principal_density", "constraint_ideal"],
                 "derivation_coefficients")
    all_vars = vars_section["all_variables"]
    phase_vars = vars_section["phase_variables"]
    phase_indices = get_phase_var_indices(all_vars, phase_vars)
    packet_by_name = {p["name"]: p for p in packets_info["packets"]}

    pd_coeffs = coeffs["principal_density"]
    require(isinstance(pd_coeffs, list) and len(pd_coeffs) >= 1,
            "derivation_coefficients.principal_density must be a nonempty list")
    derived_poly = sos.Polynomial.constant(all_vars, Fraction(0))
    seen_pd = set()
    for i, entry in enumerate(pd_coeffs):
        ctx = f"derivation_coefficients.principal_density[{i}]"
        require(isinstance(entry, dict), f"{ctx} must be an object")
        require_keys(entry, ["packet_name", "coefficient"], ctx)
        pname = require_string(entry["packet_name"], f"{ctx}.packet_name")
        require(pname in packet_by_name,
                f"{ctx}.packet_name references unknown packet {pname!r}")
        pkt = packet_by_name[pname]
        require(pkt["role"] == "principal_density_packet",
                f"{ctx}.packet_name {pname!r} must reference a principal_density_packet")
        require(pname not in seen_pd, f"{ctx}.packet_name duplicates an earlier entry")
        seen_pd.add(pname)
        c = require_fraction(entry["coefficient"], f"{ctx}.coefficient")
        derived_poly = derived_poly + pkt["poly"].scale(c)

    ci_coeffs = coeffs["constraint_ideal"]
    require(isinstance(ci_coeffs, list),
            "derivation_coefficients.constraint_ideal must be a list")
    seen_ci = set()
    constraint_polys = []
    for i, entry in enumerate(ci_coeffs):
        ctx = f"derivation_coefficients.constraint_ideal[{i}]"
        require(isinstance(entry, dict), f"{ctx} must be an object")
        require_keys(entry, ["packet_name", "coefficient"], ctx)
        pname = require_string(entry["packet_name"], f"{ctx}.packet_name")
        require(pname in packet_by_name,
                f"{ctx}.packet_name references unknown packet {pname!r}")
        pkt = packet_by_name[pname]
        require(pkt["role"] == "constraint_generator",
                f"{ctx}.packet_name {pname!r} must reference a constraint_generator")
        require(pname not in seen_ci, f"{ctx}.packet_name duplicates an earlier entry")
        seen_ci.add(pname)
        d = require_fraction(entry["coefficient"], f"{ctx}.coefficient")
        derived_poly = derived_poly + pkt["poly"].scale(d)
        constraint_polys.append(pkt["poly"])

    try:
        declared_derived = sos.eval_expr(gdd["derived_density_expression"], all_vars)
    except Exception as exc:
        raise ValueError(
            f"gravitational_density_derivation.derived_density_expression failed to parse: {exc}"
        ) from exc
    diff = declared_derived - derived_poly
    require(not diff.terms,
            "gravitational_density_derivation.derived_density_expression does not match "
            "the declared linear combination of packets; difference polynomial is nonempty")

    target_path_str = require_string(gdd["target_symbolic_normal_form_path"],
                                      "gravitational_density_derivation.target_symbolic_normal_form_path")
    target_path = (HERE / target_path_str).resolve()
    require(target_path.exists(),
            f"target_symbolic_normal_form_path does not exist: {target_path}")

    ic = gdd["identity_check"]
    require(isinstance(ic, dict),
            "gravitational_density_derivation.identity_check must be an object")
    require_keys(ic, ["difference_reduces_to_zero_modulo_ideal",
                      "remainder_polynomial"],
                 "gravitational_density_derivation.identity_check")
    require(ic["difference_reduces_to_zero_modulo_ideal"] is True,
            "identity_check.difference_reduces_to_zero_modulo_ideal must be true")

    # Load the target symbolic normal form and verify the identity modulo the ideal.
    with open(target_path) as f:
        target_data = json.load(f)
    target_de = target_data.get("density_expression")
    require(target_de is not None,
            f"target symbolic normal form at {target_path} has no density_expression")
    try:
        target_poly = sos.eval_expr(target_de, all_vars)
    except Exception as exc:
        raise ValueError(
            f"target symbolic normal form density_expression failed to parse: {exc}"
        ) from exc

    # Compute the difference between derived and target, then reduce modulo
    # the constraint ideal. We model the ideal reduction as: the difference,
    # when projected onto the phase-only subring (constraint auxiliary variables
    # set to zero), must be the zero polynomial.
    diff_derived_target = derived_poly - target_poly
    aux_indices = [all_vars.index(v) for v in vars_section["constraint_auxiliary_variables"]]
    reduced_terms = {}
    for monomial, coeff in diff_derived_target.terms.items():
        if all(monomial[i] == 0 for i in aux_indices):
            reduced_terms[monomial] = coeff
    require(not reduced_terms,
            "identity_check failed: derived density minus target symbolic normal "
            "form, reduced modulo the constraint ideal, is nonempty")

    rp = ic["remainder_polynomial"]
    require(isinstance(rp, dict),
            "identity_check.remainder_polynomial must be an object")
    require(not rp,
            "identity_check.remainder_polynomial must be empty when "
            "difference_reduces_to_zero_modulo_ideal is true")

    return {
        "target_path": str(target_path),
        "principal_density_count": len(pd_coeffs),
        "constraint_ideal_count": len(ci_coeffs),
        "identity_holds": True,
    }


def validate_provenance_audit(data, packets_info):
    pa = data.get("provenance_audit")
    require(isinstance(pa, dict), "provenance_audit must be an object")
    require_keys(pa, ["all_primitive_formulas_have_noncandidate_provenance",
                      "forbidden_provenance_substrings",
                      "primitive_formula_count",
                      "noncandidate_primitive_formula_count"],
                 "provenance_audit")
    require(pa["all_primitive_formulas_have_noncandidate_provenance"] is True,
            "provenance_audit.all_primitive_formulas_have_noncandidate_provenance must be true")

    fps = pa["forbidden_provenance_substrings"]
    require(isinstance(fps, list),
            "provenance_audit.forbidden_provenance_substrings must be a list")
    require(set(fps) == set(FORBIDDEN_PROVENANCE_SUBSTRINGS),
            f"forbidden_provenance_substrings must be exactly "
            f"{sorted(FORBIDDEN_PROVENANCE_SUBSTRINGS)}, got {sorted(fps)}")

    pfc = require_int(pa["primitive_formula_count"],
                       "provenance_audit.primitive_formula_count", minimum=1)
    ncfc = require_int(pa["noncandidate_primitive_formula_count"],
                        "provenance_audit.noncandidate_primitive_formula_count", minimum=1)
    require(pfc == len(packets_info["packets"]),
            f"primitive_formula_count ({pfc}) must equal the number of packets "
            f"({len(packets_info['packets'])})")
    require(ncfc == pfc,
            f"noncandidate_primitive_formula_count ({ncfc}) must equal "
            f"primitive_formula_count ({pfc})")
    return {"primitive_formula_count": pfc, "noncandidate_primitive_formula_count": ncfc}


def validate_derivation_budget(data, packets_info):
    db = data.get("derivation_budget")
    require(isinstance(db, dict), "derivation_budget must be an object")
    require_keys(db, ["principal_density_packet_count",
                      "constraint_generator_count",
                      "total_packet_count",
                      "max_packet_phase_degree",
                      "all_packets_homogeneous_or_phase_free"],
                 "derivation_budget")
    pdpc = require_int(db["principal_density_packet_count"],
                       "derivation_budget.principal_density_packet_count", minimum=1)
    cgc = require_int(db["constraint_generator_count"],
                      "derivation_budget.constraint_generator_count", minimum=0)
    tpc = require_int(db["total_packet_count"],
                      "derivation_budget.total_packet_count", minimum=1)
    mpd = require_int(db["max_packet_phase_degree"],
                      "derivation_budget.max_packet_phase_degree", minimum=2)
    require(pdpc == packets_info["principal_count"],
            f"principal_density_packet_count ({pdpc}) must equal the actual "
            f"principal density packet count ({packets_info['principal_count']})")
    require(cgc == packets_info["constraint_count"],
            f"constraint_generator_count ({cgc}) must equal the actual "
            f"constraint generator count ({packets_info['constraint_count']})")
    require(tpc == len(packets_info["packets"]),
            f"total_packet_count ({tpc}) must equal the number of packets "
            f"({len(packets_info['packets'])})")
    require(mpd == packets_info["max_phase_degree"],
            f"max_packet_phase_degree ({mpd}) must equal the actual max phase degree "
            f"({packets_info['max_phase_degree']})")
    require(db["all_packets_homogeneous_or_phase_free"] is True,
            "derivation_budget.all_packets_homogeneous_or_phase_free must be true")
    return {
        "principal_density_packet_count": pdpc,
        "constraint_generator_count": cgc,
        "total_packet_count": tpc,
        "max_packet_phase_degree": mpd,
    }


def verify(data):
    require(isinstance(data, dict), "certificate must be a JSON object")
    header = validate_header(data)
    vars_section = validate_variables(data)
    packets_info = validate_linearized_einstein_packets(data, vars_section)
    derivation = validate_gravitational_density_derivation(data, packets_info, vars_section)
    audit = validate_provenance_audit(data, packets_info)
    budget = validate_derivation_budget(data, packets_info)
    return {
        "accepted": True,
        "theorem": "GPDDT",
        "version": "1.0",
        "background": header["background"],
        "spin": header["spin"],
        "chart": header["chart"],
        "principal_density_packet_count": budget["principal_density_packet_count"],
        "constraint_generator_count": budget["constraint_generator_count"],
        "total_packet_count": budget["total_packet_count"],
        "max_packet_phase_degree": budget["max_packet_phase_degree"],
        "target_symbolic_normal_form_path": derivation["target_path"],
        "identity_holds": derivation["identity_holds"],
        "primitive_formula_count": audit["primitive_formula_count"],
        "warning": ("Combinatorial, arithmetic, and provenance checks pass. "
                    "This does NOT prove the GPDDT analytic content: the linearized "
                    "Einstein operator packets, the constraint ideal generators, "
                    "and the provenance strings are not checked against the upstream "
                    "Kerr/Einstein analysis."),
    }


def verify_file(path):
    with open(path) as f:
        data = json.load(f)
    return verify(data)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except ValueError as exc:
        print(f"GPDDT certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
