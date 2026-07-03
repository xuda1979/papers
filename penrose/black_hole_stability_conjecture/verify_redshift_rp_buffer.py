#!/usr/bin/env python3
"""Validate a Redshift–r^p Buffer (RRBB) certificate.

This checker verifies the combinatorial and arithmetic content of the
Redshift–r^p Buffer Theorem (RRBB). It checks:

  1. every required field is present and well-typed;
  2. every rational is a valid Fraction representation and positive where
     required;
  3. B_redshift = kappa_red * mu_redshift;
  4. B_rp = C_p * r_far^{-eta_p}, with r_far taken from regions.rp_far_field;
  5. B_comm lower bound is nonnegative;
  6. B_lower_bound = B_redshift + B_rp + B_comm_lower_bound;
  7. the regions are ordered r_+ < r_red < r_far;
  8. if cwqp_compatibility is present, the CWQP-AB total loss is at most
     B_lower_bound (composed closure).

The verifier is deliberately conservative. It does NOT check the analytic
content of the theorem (Dafermos–Rodnianski redshift calculation, r^p
weighted estimate, nontrapping of q_s on the bulk). It only checks that a
proposed certificate is internally consistent and that the buffer
arithmetic closes.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def as_fraction(value: Any) -> Fraction:
    require(isinstance(value, str), f"rational must be a string, got {type(value).__name__}: {value!r}")
    try:
        return Fraction(value)
    except (ArithmeticError, ValueError) as exc:
        raise ValueError(f"invalid rational {value!r}: {exc}") from exc


def as_positive_fraction(value: Any, name: str) -> Fraction:
    f = as_fraction(value)
    require(f > 0, f"{name} must be positive, got {f}")
    return f


def as_nonnegative_fraction(value: Any, name: str) -> Fraction:
    f = as_fraction(value)
    require(f >= 0, f"{name} must be nonnegative, got {f}")
    return f


def validate_metadata(data: dict[str, Any]) -> dict[str, str]:
    metadata = data.get("metadata")
    require(isinstance(metadata, dict), "metadata must be an object")
    for key in ("name", "generated_on", "spin", "claim_scope", "status", "warning"):
        require(key in metadata, f"metadata.{key} must be present")
        require(isinstance(metadata[key], str), f"metadata.{key} must be a string")
    require(
        "Redshift" in metadata["name"] or "RRBB" in metadata["name"]
        or "redshift" in metadata["name"],
        "metadata.name must reference Redshift or RRBB",
    )
    return metadata


def validate_regions(regions: Any) -> dict[str, Any]:
    require(isinstance(regions, dict), "regions must be an object")
    for key in ("redshift_layer", "trapped_bulk", "rp_far_field"):
        require(key in regions, f"regions.{key} must be present")
        entry = regions[key]
        require(isinstance(entry, dict), f"regions.{key} must be an object")
        require("endpoints" in entry, f"regions.{key}.endpoints must be present")
        endpoints = entry["endpoints"]
        require(isinstance(endpoints, list) and len(endpoints) == 2,
                f"regions.{key}.endpoints must be a 2-element list")
        lo = as_fraction(endpoints[0])
        hi = as_fraction(endpoints[1])
        require(lo < hi, f"regions.{key}.endpoints must satisfy lo < hi")
        regions[key] = {**entry, "_lo": lo, "_hi": hi}
    # Ordering: r_+ < r_red < r_far.
    r_plus = regions["redshift_layer"]["_lo"]
    r_red_hi = regions["redshift_layer"]["_hi"]
    r_far_lo = regions["rp_far_field"]["_lo"]
    require(r_red_hi <= r_far_lo,
            f"regions must satisfy redshift_layer.hi <= rp_far_field.lo "
            f"(got {r_red_hi} and {r_far_lo})")
    require(r_plus < r_red_hi, "redshift_layer must have positive width")
    return regions


def validate_redshift_certificate(entry: Any) -> dict[str, Fraction]:
    require(isinstance(entry, dict), "redshift_certificate must be an object")
    kappa_red = as_positive_fraction(entry.get("kappa_red"), "redshift_certificate.kappa_red")
    mu_redshift = as_positive_fraction(entry.get("mu_redshift"), "redshift_certificate.mu_redshift")
    sgr = as_positive_fraction(
        entry.get("horizon_surface_gravity_lower_bound"),
        "redshift_certificate.horizon_surface_gravity_lower_bound",
    )
    return {
        "kappa_red": kappa_red,
        "mu_redshift": mu_redshift,
        "horizon_surface_gravity_lower_bound": sgr,
    }


def validate_rp_certificate(entry: Any) -> dict[str, Any]:
    require(isinstance(entry, dict), "rp_certificate must be an object")
    c_p = as_positive_fraction(entry.get("C_p"), "rp_certificate.C_p")
    eta_p = as_positive_fraction(entry.get("eta_p"), "rp_certificate.eta_p")
    p = as_positive_fraction(entry.get("p"), "rp_certificate.p")
    require(p > 2, f"rp_certificate.p must satisfy p > 2 for decay, got {p}")
    sobolev = entry.get("sobolev_index")
    require(isinstance(sobolev, int) and sobolev >= 1,
            f"rp_certificate.sobolev_index must be a positive integer, got {sobolev!r}")
    return {"C_p": c_p, "eta_p": eta_p, "p": p, "sobolev_index": sobolev}


def validate_commutator_slack(entry: Any) -> dict[str, Any]:
    require(isinstance(entry, dict), "commutator_slack_certificate must be an object")
    nonneg = entry.get("nonnegative")
    require(isinstance(nonneg, bool),
            "commutator_slack_certificate.nonnegative must be a boolean")
    require(nonneg, "commutator_slack_certificate.nonnegative must be true")
    esc_ref = entry.get("escape_symbol_reference")
    require(isinstance(esc_ref, str) and esc_ref,
            "commutator_slack_certificate.escape_symbol_reference must be a nonempty string")
    lb = entry.get("B_comm_lower_bound", "0/1")
    lb_frac = as_nonnegative_fraction(lb, "commutator_slack_certificate.B_comm_lower_bound")
    return {"nonnegative": nonneg, "escape_symbol_reference": esc_ref,
            "B_comm_lower_bound": lb_frac}


def compute_B_redshift(kappa_red: Fraction, mu_redshift: Fraction) -> Fraction:
    return kappa_red * mu_redshift


def compute_B_rp(c_p: Fraction, r_far: Fraction, eta_p: Fraction) -> Fraction:
    """B_rp = C_p * r_far^{-eta_p}. Requires eta_p to be a nonnegative integer
    so that r_far^{-eta_p} is a rational exact power. For non-integer eta_p,
    the certificate must declare an explicit rational B_rp instead."""
    if eta_p.denominator == 1:
        return c_p * (r_far ** (-int(eta_p)))
    raise ValueError(
        f"eta_p must be an integer for exact rational r_far^(-eta_p); "
        f"got eta_p = {eta_p}. For non-integer eta_p, declare B_rp explicitly "
        f"in the buffer object."
    )


def validate_buffer(
    data: dict[str, Any],
    b_red: Fraction,
    b_rp: Fraction,
    b_comm_lb: Fraction,
) -> dict[str, Fraction]:
    buffer = data.get("buffer")
    require(isinstance(buffer, dict), "buffer must be an object")
    for key in ("B_redshift", "B_rp", "B_comm_lower_bound", "B_lower_bound"):
        require(key in buffer, f"buffer.{key} must be present")
    declared_b_red = as_positive_fraction(buffer["B_redshift"], "buffer.B_redshift")
    declared_b_rp = as_positive_fraction(buffer["B_rp"], "buffer.B_rp")
    declared_b_comm_lb = as_nonnegative_fraction(
        buffer["B_comm_lower_bound"], "buffer.B_comm_lower_bound"
    )
    declared_b_lb = as_positive_fraction(buffer["B_lower_bound"], "buffer.B_lower_bound")
    require(declared_b_red == b_red,
            f"buffer.B_redshift mismatch: declared {declared_b_red} != computed {b_red}")
    require(declared_b_rp == b_rp,
            f"buffer.B_rp mismatch: declared {declared_b_rp} != computed {b_rp}")
    require(declared_b_comm_lb == b_comm_lb,
            f"buffer.B_comm_lower_bound mismatch: declared {declared_b_comm_lb} "
            f"!= computed {b_comm_lb}")
    computed_lb = b_red + b_rp + b_comm_lb
    require(declared_b_lb == computed_lb,
            f"buffer.B_lower_bound mismatch: declared {declared_b_lb} "
            f"!= computed {computed_lb}")
    return {
        "B_redshift": b_red,
        "B_rp": b_rp,
        "B_comm_lower_bound": b_comm_lb,
        "B_lower_bound": computed_lb,
    }


def validate_cwqp_compatibility(
    data: dict[str, Any], b_lower_bound: Fraction
) -> dict[str, Any] | None:
    cwqp = data.get("cwqp_compatibility")
    if cwqp is None:
        return None
    require(isinstance(cwqp, dict), "cwqp_compatibility must be an object")
    require("cwqp_certificate_ref" in cwqp,
            "cwqp_compatibility.cwqp_certificate_ref must be present")
    ref = cwqp["cwqp_certificate_ref"]
    require(isinstance(ref, str) and ref,
            "cwqp_compatibility.cwqp_certificate_ref must be a nonempty string")
    require("total_loss" in cwqp,
            "cwqp_compatibility.total_loss must be present")
    total_loss = as_positive_fraction(cwqp["total_loss"],
                                       "cwqp_compatibility.total_loss")
    closure = total_loss <= b_lower_bound
    require("closure" in cwqp, "cwqp_compatibility.closure must be present")
    require(isinstance(cwqp["closure"], bool),
            "cwqp_compatibility.closure must be a boolean")
    require(cwqp["closure"] == closure,
            f"cwqp_compatibility.closure mismatch: declared {cwqp['closure']} "
            f"!= computed {closure} (total_loss={total_loss}, "
            f"B_lower_bound={b_lower_bound})")
    return {
        "cwqp_certificate_ref": ref,
        "total_loss": total_loss,
        "B_lower_bound": b_lower_bound,
        "closure": closure,
    }


def _frac_to_str(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _frac_to_str(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_frac_to_str(v) for v in obj]
    return obj


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(isinstance(data, dict), "certificate must be an object")
    metadata = validate_metadata(data)
    regions = validate_regions(data.get("regions"))
    red_cert = validate_redshift_certificate(data.get("redshift_certificate"))
    rp_cert = validate_rp_certificate(data.get("rp_certificate"))
    slack_cert = validate_commutator_slack(data.get("commutator_slack_certificate"))

    b_red = compute_B_redshift(red_cert["kappa_red"], red_cert["mu_redshift"])
    r_far = regions["rp_far_field"]["_lo"]
    b_rp = compute_B_rp(rp_cert["C_p"], r_far, rp_cert["eta_p"])
    b_comm_lb = slack_cert["B_comm_lower_bound"]

    buffer = validate_buffer(data, b_red, b_rp, b_comm_lb)
    cwqp_compat = validate_cwqp_compatibility(data, buffer["B_lower_bound"])

    result = {
        "accepted": True,
        "metadata": metadata,
        "regions": {
            "redshift_layer": [str(regions["redshift_layer"]["_lo"]),
                               str(regions["redshift_layer"]["_hi"])],
            "trapped_bulk": [str(regions["trapped_bulk"]["_lo"]),
                             str(regions["trapped_bulk"]["_hi"])],
            "rp_far_field": [str(regions["rp_far_field"]["_lo"]),
                             str(regions["rp_far_field"]["_hi"])],
        },
        "redshift_certificate": red_cert,
        "rp_certificate": rp_cert,
        "commutator_slack_certificate": slack_cert,
        "buffer": buffer,
        "cwqp_compatibility": cwqp_compat,
        "warning": (
            "Combinatorial and arithmetic checks pass. This does NOT prove "
            "the Redshift–r^p Buffer Theorem: the analytic content "
            "(Dafermos–Rodnianski redshift, r^p weighted estimate, "
            "nontrapping of q_s on the bulk) is not checked."
        ),
    }
    return _frac_to_str(result)


def verify_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return verify(data)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("certificate", type=Path)
    args = parser.parse_args(argv[1:])
    try:
        result = verify_file(args.certificate)
    except ValueError as exc:
        print(f"RRBB certificate rejected: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
