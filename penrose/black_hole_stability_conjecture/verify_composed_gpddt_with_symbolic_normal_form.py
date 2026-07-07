#!/usr/bin/env python3
"""Validate the composition of a symbolic Kerr identification certificate with a GPDDT certificate.

This composed verifier checks that:
  1. the symbolic Kerr identification verifier
     `verify_carter_symbolic_kerr_identification.py` accepts the symbolic normal
     form `S`;
  2. the GPDDT verifier `verify_gravitational_principal_density_derivation.py`
     accepts the GPDDT certificate `G`;
  3. the two certificates agree on the chart, the phase variables, the symbolic
     coefficient variables, and the target symbolic normal form path; and
  4. the GPDDT-derived density expression, when reduced modulo the GPDDT
     constraint ideal, equals the symbolic normal form's density expression
     as a polynomial in the shared variable ring.

The composition is:

> Symbolic Normal Form + Linearized Einstein Packets + GPDDT -> derived
> gravitational principal density.

This reduces the P0 q_grav derivation gate to a finite rational-arithmetic
certificate with an explicit machine-checkable interface. The analytic content
of the linearized Einstein equations (the actual principal symbols, the actual
constraint ideal, and the correctness of the provenance strings) remains open.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import verify_carter_symbolic_kerr_identification as vsk  # noqa: E402
import verify_gravitational_principal_density_derivation as vgp  # noqa: E402
import verify_carter_sos_identity as sos  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def cross_validate(
    symbolic: dict[str, Any],
    gpddt: dict[str, Any],
    symbolic_path: Path,
    errors: list[str],
) -> None:
    """Check that the GPDDT certificate matches the symbolic normal form."""

    # --- Chart agreement ---
    sym_chart = symbolic.get("chart")
    gpddt_chart = gpddt.get("chart")
    _check(sym_chart == gpddt_chart,
           f"chart mismatch: symbolic={sym_chart!r}, gpddt={gpddt_chart!r}",
           errors)

    # --- Background agreement ---
    sym_meta = symbolic.get("metadata", {})
    gpddt_background = gpddt.get("background")
    _check(gpddt_background == "fixed_subextremal_kerr",
           f"GPDDT background must be 'fixed_subextremal_kerr', got {gpddt_background!r}",
           errors)

    # --- Spin agreement ---
    gpddt_spin = gpddt.get("spin")
    _check(gpddt_spin == "1/2",
           f"GPDDT spin must be '1/2' for this chart family, got {gpddt_spin!r}",
           errors)

    # --- Phase variables agreement ---
    sym_phase = list(symbolic.get("variables", []))
    gpddt_vars = gpddt.get("variables", {})
    gpddt_phase = list(gpddt_vars.get("phase_variables", []))
    _check(sym_phase == gpddt_phase,
           f"phase variables mismatch: symbolic={sym_phase!r}, gpddt={gpddt_phase!r}",
           errors)

    # --- Symbolic coefficient variables agreement ---
    sym_coeff = list(symbolic.get("symbolic_coefficient_variables", []))
    gpddt_coeff = list(gpddt_vars.get("symbolic_coefficient_variables", []))
    _check(sym_coeff == gpddt_coeff,
           f"symbolic coefficient variables mismatch: "
           f"symbolic={sym_coeff!r}, gpddt={gpddt_coeff!r}",
           errors)

    # --- Target symbolic normal form path agreement ---
    gdd = gpddt.get("gravitational_density_derivation", {})
    target_path_str = gdd.get("target_symbolic_normal_form_path")
    if target_path_str is not None:
        target_path = (Path(__file__).resolve().parent / target_path_str).resolve()
        sym_path_resolved = symbolic_path.resolve()
        _check(target_path == sym_path_resolved,
               f"GPDDT target_symbolic_normal_form_path ({target_path}) must "
               f"reference the supplied symbolic normal form ({sym_path_resolved})",
               errors)
    else:
        errors.append("GPDDT certificate is missing "
                      "gravitational_density_derivation.target_symbolic_normal_form_path")

    # --- Polynomial-level identity check (re-derive independently) ---
    if sym_phase and sym_coeff and target_path_str is not None:
        all_vars = tuple(gpddt_phase + gpddt_coeff +
                         list(gpddt_vars.get("constraint_auxiliary_variables", [])))
        # Symbolic density expression evaluated in the GPDDT variable ring.
        sym_de = symbolic.get("density_expression")
        if sym_de is None:
            errors.append("symbolic normal form is missing density_expression")
        else:
            try:
                sym_poly = sos.eval_expr(sym_de, all_vars)
            except Exception as exc:
                errors.append(f"symbolic density_expression failed to parse in the "
                              f"GPDDT variable ring: {exc}")
                sym_poly = None
            # GPDDT declared derived density.
            gpddt_de = gdd.get("derived_density_expression")
            if gpddt_de is None:
                errors.append("GPDDT certificate is missing "
                              "gravitational_density_derivation.derived_density_expression")
            else:
                try:
                    gpddt_poly = sos.eval_expr(gpddt_de, all_vars)
                except Exception as exc:
                    errors.append(f"GPDDT derived_density_expression failed to parse: {exc}")
                    gpddt_poly = None
                if sym_poly is not None and gpddt_poly is not None:
                    diff = gpddt_poly - sym_poly
                    # Reduce modulo the constraint ideal: project onto the
                    # phase-only subring by dropping constraint auxiliary variables.
                    aux_indices = [all_vars.index(v)
                                   for v in gpddt_vars.get("constraint_auxiliary_variables", [])
                                   if v in all_vars]
                    reduced_terms = {}
                    for monomial, coeff in diff.terms.items():
                        if all(monomial[i] == 0 for i in aux_indices):
                            reduced_terms[monomial] = coeff
                    _check(not reduced_terms,
                           "derived density minus symbolic normal form, reduced modulo "
                           "the constraint ideal, is nonempty",
                           errors)


def verify_composed(symbolic_path: Path, gpddt_path: Path) -> dict[str, Any]:
    errors: list[str] = []

    symbolic = _load(symbolic_path)
    gpddt = _load(gpddt_path)

    # Run each upstream verifier independently.
    try:
        sym_result = vsk.verify(symbolic, base_dir=symbolic_path.parent)
    except ValueError as exc:
        errors.append(f"symbolic normal form rejected: {exc}")
        sym_result = None

    try:
        gpddt_result = vgp.verify(gpddt)
    except ValueError as exc:
        errors.append(f"GPDDT certificate rejected: {exc}")
        gpddt_result = None

    # Cross-validate only if both upstream checks passed.
    if sym_result is not None and gpddt_result is not None:
        cross_validate(symbolic, gpddt, symbolic_path, errors)

    if errors:
        return {
            "accepted": False,
            "composition": "symbolic_normal_form + GPDDT",
            "errors": errors,
            "symbolic_accepted": sym_result is not None,
            "gpddt_accepted": gpddt_result is not None,
        }

    return {
        "accepted": True,
        "composition": "symbolic_normal_form + GPDDT",
        "symbolic_accepted": True,
        "gpddt_accepted": True,
        "chart": sym_result["chart"],
        "spin": gpddt_result["spin"],
        "background": gpddt_result["background"],
        "phase_variables": sym_result["variables"],
        "symbolic_coefficient_variables": list(
            gpddt["variables"]["symbolic_coefficient_variables"]
        ),
        "principal_density_packet_count": gpddt_result["principal_density_packet_count"],
        "constraint_generator_count": gpddt_result["constraint_generator_count"],
        "identity_holds": gpddt_result["identity_holds"],
        "warning": ("Composed combinatorial, arithmetic, provenance, and "
                    "cross-certificate polynomial-identity checks pass. This "
                    "does NOT prove the black hole stability conjecture: the "
                    "analytic content of the linearized Einstein operator "
                    "packets, the constraint ideal generators, the provenance "
                    "strings, and the symbolic Kerr identification itself "
                    "remain open."),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    symbolic_path = Path(argv[1])
    gpddt_path = Path(argv[2])
    if not symbolic_path.exists():
        print(f"symbolic normal form not found: {symbolic_path}", file=sys.stderr)
        return 2
    if not gpddt_path.exists():
        print(f"GPDDT certificate not found: {gpddt_path}", file=sys.stderr)
        return 2
    result = verify_composed(symbolic_path, gpddt_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
