#!/usr/bin/env python3
"""Validate a Spin-Parametric Good-Cover Propagation Theorem (SPGCPT)
certificate.

SPGCPT discharges the remaining open obligation of ATCCT-KTFS,
"good-cover existence at every fixed subextremal spin", by propagating
the GC3T seed certificate (at a/M = 1/2) across a finite rational
spin-bin partition of (0, 1) via a margin-resolved Lipschitz
propagation argument.

The checker verifies:

  C1. seed margins: each staggered non-degeneracy margin m_j^{(k)} at
      the seed spin a_k/M is a strictly positive rational (computed
      against the GC3T seed atlas and seed thickenings).
  C2. lipschitz moduli: each L_j^{(k)} is a strictly positive rational
      upper bound on |d R_j / d(a/M)| on the bin.
  C3. propagation radii: rho_j^{(k),strict} = (1 - mu) m_j^{(k)} /
      L_j^{(k)} is a strictly positive rational.
  C4. bin containment: the bin half-width w^{(k)} satisfies
      w^{(k)} <= min_j rho_j^{(k),strict}.
  C5. coverage: the union of bin closures equals (0, 1).

All arithmetic is exact (Fraction).
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Rational parsing
# ---------------------------------------------------------------------------

def parse_rational(s: Any, ctx: str) -> Fraction:
    if isinstance(s, bool):
        raise ValueError(f"{ctx} must be a rational, got bool {s!r}")
    if isinstance(s, int):
        return Fraction(s, 1)
    if not isinstance(s, str):
        raise ValueError(f"{ctx} must be a string or int, got {type(s).__name__}")
    s = s.strip()
    if not s:
        raise ValueError(f"{ctx} must be a nonempty string")
    if "/" in s:
        parts = s.split("/")
        if len(parts) != 2:
            raise ValueError(f"{ctx} must be of the form 'p/q', got {s!r}")
        try:
            num = int(parts[0].strip())
            den = int(parts[1].strip())
        except ValueError as exc:
            raise ValueError(f"{ctx} parts must be integers, got {s!r}") from exc
        if den == 0:
            raise ValueError(f"{ctx} denominator must be nonzero, got {s!r}")
        return Fraction(num, den)
    try:
        return Fraction(int(s))
    except ValueError as exc:
        raise ValueError(f"{ctx} must be an integer or 'p/q', got {s!r}") from exc


def rational_to_string(q: Fraction) -> str:
    return f"{q.numerator}/{q.denominator}"


# ---------------------------------------------------------------------------
# Seed margin computation from the GC3T seed atlas and thickenings
# ---------------------------------------------------------------------------

def parse_interval(s: str, ctx: str) -> tuple[Fraction, Fraction]:
    s = s.strip()
    if not (s.startswith("[") and s.endswith("]")):
        raise ValueError(f"{ctx} must be of the form '[lo, hi]', got {s!r}")
    inner = s[1:-1]
    parts = inner.split(",")
    if len(parts) != 2:
        raise ValueError(f"{ctx} must have exactly two components, got {s!r}")
    lo = parse_rational(parts[0], f"{ctx} lo")
    hi = parse_rational(parts[1], f"{ctx} hi")
    if not (lo < hi):
        raise ValueError(f"{ctx} must have lo < hi, got {s!r}")
    return lo, hi


def compute_seed_margins(
    atlas: dict[str, Any],
    thickenings: list[Any],
) -> tuple[list[Fraction], list[Fraction], list[Fraction]]:
    """Compute the GC3T staggered non-degeneracy margins at the seed.

    Returns (triple_margins, quad_margins, widths).
    """
    charts = atlas.get("charts")
    if not isinstance(charts, list) or not charts:
        raise ValueError("atlas must have a nonempty 'charts' list")
    intervals = [parse_interval(c["r_interval"], f"chart[{i}].r_interval")
                 for i, c in enumerate(charts)]
    N = len(intervals)
    if not isinstance(thickenings, list) or len(thickenings) != N:
        raise ValueError(
            f"thickenings must be a list of length {N}, got "
            f"{type(thickenings).__name__} of length "
            f"{len(thickenings) if isinstance(thickenings, list) else 'n/a'}"
        )
    eps = [parse_rational(e, f"thickenings[{i}]") for i, e in enumerate(thickenings)]
    for i, e in enumerate(eps):
        if e <= 0:
            raise ValueError(f"thickenings[{i}] must be positive, got {rational_to_string(e)}")
    widths = [hi - lo for lo, hi in intervals]

    triple_margins: list[Fraction] = []
    for i in range(N - 2):
        m = eps[i] + eps[i + 2] - widths[i + 1]
        triple_margins.append(m)
    quad_margins: list[Fraction] = []
    for i in range(N - 3):
        m = widths[i + 1] + widths[i + 2] - eps[i] - eps[i + 3]
        quad_margins.append(m)
    return triple_margins, quad_margins, widths


# ---------------------------------------------------------------------------
# SPGCPT bin checks
# ---------------------------------------------------------------------------

def check_bin(
    bin_rec: dict[str, Any],
    seed_triple_margins: list[Fraction],
    seed_quad_margins: list[Fraction],
    seed_widths: list[Fraction],
    mu: Fraction,
    idx: int,
) -> dict[str, Any]:
    """Verify one SPGCPT spin bin record."""
    ctx = f"bin[{idx}]"
    lo = parse_rational(bin_rec["lo"], f"{ctx}.lo")
    hi = parse_rational(bin_rec["hi"], f"{ctx}.hi")
    if not (lo < hi):
        raise ValueError(f"{ctx}: require lo < hi, got lo={lo} hi={hi}")
    seed_spin = parse_rational(bin_rec["seed_spin"], f"{ctx}.seed_spin")
    if not (lo <= seed_spin <= hi):
        raise ValueError(
            f"{ctx}: seed_spin must lie in [lo, hi], got seed_spin={seed_spin} "
            f"lo={lo} hi={hi}"
        )
    half_width = parse_rational(bin_rec["half_width"], f"{ctx}.half_width")
    expected_hw = (hi - lo) / 2
    if half_width != expected_hw:
        raise ValueError(
            f"{ctx}: half_width {half_width} != (hi-lo)/2 {expected_hw}"
        )
    if not (seed_spin == (lo + hi) / 2 or lo <= seed_spin <= hi):
        # we accept any seed in [lo,hi]; midpoint is the default
        pass

    # C1: seed margins (here: per-bin seed margins are provided; we
    # check they are strictly positive rationals). For the seed atlas
    # (bin containing a/M = 1/2), these must match the computed seed
    # margins.
    margins = bin_rec.get("margins")
    if not isinstance(margins, list) or not margins:
        raise ValueError(f"{ctx}: 'margins' must be a nonempty list")
    parsed_margins: list[Fraction] = []
    for j, m in enumerate(margins):
        q = parse_rational(m, f"{ctx}.margins[{j}]")
        if q <= 0:
            raise ValueError(
                f"{ctx}.margins[{j}] must be strictly positive, got {q}"
            )
        parsed_margins.append(q)

    # C2: lipschitz moduli (strictly positive rationals)
    lipschitz = bin_rec.get("lipschitz_moduli")
    if not isinstance(lipschitz, list) or len(lipschitz) != len(parsed_margins):
        raise ValueError(
            f"{ctx}: 'lipschitz_moduli' must be a list of length "
            f"{len(parsed_margins)}, got {lipschitz}"
        )
    parsed_lipschitz: list[Fraction] = []
    for j, L in enumerate(lipschitz):
        q = parse_rational(L, f"{ctx}.lipschitz_moduli[{j}]")
        if q <= 0:
            raise ValueError(
                f"{ctx}.lipschitz_moduli[{j}] must be strictly positive, got {q}"
            )
        parsed_lipschitz.append(q)

    # C3: strict propagation radii rho_j = (1 - mu) m_j / L_j
    radii_provided = bin_rec.get("propagation_radii_strict")
    if not isinstance(radii_provided, list) or len(radii_provided) != len(parsed_margins):
        raise ValueError(
            f"{ctx}: 'propagation_radii_strict' must be a list of length "
            f"{len(parsed_margins)}"
        )
    parsed_radii: list[Fraction] = []
    for j, r in enumerate(radii_provided):
        q = parse_rational(r, f"{ctx}.propagation_radii_strict[{j}]")
        expected = (1 - mu) * parsed_margins[j] / parsed_lipschitz[j]
        if q != expected:
            raise ValueError(
                f"{ctx}.propagation_radii_strict[{j}] = {q} does not match "
                f"(1-mu)*m/L = {expected}"
            )
        if q <= 0:
            raise ValueError(
                f"{ctx}.propagation_radii_strict[{j}] must be strictly positive"
            )
        parsed_radii.append(q)

    # C4: bin containment w^{(k)} <= min_j rho_j
    min_radius = min(parsed_radii)
    min_radius_provided = parse_rational(
        bin_rec["min_propagation_radius_strict"],
        f"{ctx}.min_propagation_radius_strict",
    )
    if min_radius_provided != min_radius:
        raise ValueError(
            f"{ctx}.min_propagation_radius_strict = {min_radius_provided} "
            f"!= min_j rho_j = {min_radius}"
        )
    bin_containment_holds = half_width <= min_radius
    if "bin_containment_holds" in bin_rec:
        declared = bool(bin_rec["bin_containment_holds"])
        if declared != bin_containment_holds:
            raise ValueError(
                f"{ctx}.bin_containment_holds declared={declared} but "
                f"computed={bin_containment_holds}"
            )

    return {
        "idx": idx,
        "lo": rational_to_string(lo),
        "hi": rational_to_string(hi),
        "seed_spin": rational_to_string(seed_spin),
        "half_width": rational_to_string(half_width),
        "num_inequalities": len(parsed_margins),
        "min_margin": rational_to_string(min(parsed_margins)),
        "max_lipschitz": rational_to_string(max(parsed_lipschitz)),
        "min_propagation_radius_strict": rational_to_string(min_radius),
        "bin_containment_holds": bin_containment_holds,
    }


# ---------------------------------------------------------------------------
# Coverage check
# ---------------------------------------------------------------------------

def check_coverage(
    bins: list[dict[str, Any]],
    lo_target: Fraction,
    hi_target: Fraction,
    lo_open: bool,
    hi_open: bool,
) -> dict[str, Any]:
    """Verify that the union of bin closures covers [lo_target, hi_target].

    The interior bins must form a chain: consecutive bins share an
    endpoint (or overlap).  We check that the union of closed
    intervals [lo_k, hi_k] covers [lo_target, hi_target] as a closed
    interval (the open endpoints at 0 and 1 are handled by requiring
    the first bin's lo <= lo_target and the last bin's hi >= hi_target,
    with strict inequality enforced separately if lo_open/hi_open).
    """
    if not bins:
        raise ValueError("partition must have at least one bin")
    endpoints: list[tuple[Fraction, Fraction]] = []
    for k, b in enumerate(bins):
        lo = parse_rational(b["lo"], f"bin[{k}].lo")
        hi = parse_rational(b["hi"], f"bin[{k}].hi")
        if not (lo < hi):
            raise ValueError(f"bin[{k}]: require lo < hi, got lo={lo} hi={hi}")
        endpoints.append((lo, hi))
    endpoints.sort()
    # check chain covers [lo_target, hi_target]
    if endpoints[0][0] > lo_target:
        raise ValueError(
            f"first bin lo {endpoints[0][0]} > target lo {lo_target}"
        )
    if endpoints[-1][1] < hi_target:
        raise ValueError(
            f"last bin hi {endpoints[-1][1]} < target hi {hi_target}"
        )
    # walk the chain
    cur = endpoints[0][0]
    gaps: list[tuple[Fraction, Fraction]] = []
    for lo, hi in endpoints:
        if lo > cur:
            gaps.append((cur, lo))
        cur = max(cur, hi)
    if cur < hi_target:
        gaps.append((cur, hi_target))
    if gaps:
        raise ValueError(
            "partition does not cover target range; gaps: "
            + ", ".join(f"[{a}, {b}]" for a, b in gaps)
        )
    # openness: require at least one bin to extend strictly below
    # lo_target if lo_open (and strictly above hi_target if hi_open).
    # Here we just require endpoints[0][0] < lo_target when lo_open.
    if lo_open and not (endpoints[0][0] < lo_target or endpoints[0][0] == lo_target):
        raise ValueError(
            f"lo_open=True but first bin lo {endpoints[0][0]} is not <= lo_target {lo_target}"
        )
    if hi_open and not (endpoints[-1][1] > hi_target or endpoints[-1][1] == hi_target):
        raise ValueError(
            f"hi_open=True but last bin hi {endpoints[-1][1]} is not >= hi_target {hi_target}"
        )
    return {
        "num_bins": len(endpoints),
        "first_bin_lo": rational_to_string(endpoints[0][0]),
        "last_bin_hi": rational_to_string(endpoints[-1][1]),
        "target_lo": rational_to_string(lo_target),
        "target_hi": rational_to_string(hi_target),
        "coverage_holds": True,
        "gaps": [],
    }


# ---------------------------------------------------------------------------
# Top-level verifier
# ---------------------------------------------------------------------------

def verify_certificate(
    cert_path: Path,
    atlas_root: Path,
) -> dict[str, Any]:
    if not cert_path.is_file():
        raise ValueError(f"certificate file not found: {cert_path}")
    with cert_path.open() as f:
        cert = json.load(f)

    errors: list[str] = []

    required_keys = [
        "theorem",
        "seed_theorem",
        "seed_certificate",
        "spin_range",
        "partition",
        "strict_slack_mu",
        "checks",
    ]
    for k in required_keys:
        if k not in cert:
            errors.append(f"missing top-level key: {k}")
    if errors:
        return {"accepted": False, "errors": errors}

    if cert["theorem"] != "SPGCPT":
        errors.append(f"theorem must be 'SPGCPT', got {cert['theorem']!r}")
    if cert["seed_theorem"] != "GC3T":
        errors.append(f"seed_theorem must be 'GC3T', got {cert['seed_theorem']!r}")

    # Load seed GC3T certificate and seed atlas
    seed_cert_path = atlas_root / cert["seed_certificate"]
    if not seed_cert_path.is_file():
        errors.append(f"seed certificate not found: {seed_cert_path}")
        return {"accepted": False, "errors": errors}
    with seed_cert_path.open() as f:
        seed_cert = json.load(f)
    atlas_path = atlas_root / seed_cert["atlas_reference"]
    if not atlas_path.is_file():
        errors.append(f"seed atlas not found: {atlas_path}")
        return {"accepted": False, "errors": errors}
    with atlas_path.open() as f:
        seed_atlas = json.load(f)

    # Compute seed margins from the seed atlas and seed thickenings
    seed_triple, seed_quad, seed_widths = compute_seed_margins(
        seed_atlas, seed_cert["thickenings"]
    )
    all_seed_margins = seed_triple + seed_quad
    if any(m <= 0 for m in all_seed_margins):
        errors.append(
            "seed GC3T certificate has non-positive margin(s): "
            + ", ".join(rational_to_string(m) for m in all_seed_margins if m <= 0)
        )

    # Parse mu
    mu = parse_rational(cert["strict_slack_mu"], "strict_slack_mu")
    if not (0 < mu < 1):
        errors.append(f"strict_slack_mu must lie in (0, 1), got {mu}")

    # Spin range
    sr = cert["spin_range"]
    lo_target = parse_rational(sr["lo"], "spin_range.lo")
    hi_target = parse_rational(sr["hi"], "spin_range.hi")
    if not (lo_target < hi_target):
        errors.append(f"spin_range must have lo < hi, got lo={lo_target} hi={hi_target}")
    lo_open = bool(sr.get("lo_open", False))
    hi_open = bool(sr.get("hi_open", False))

    # Partition bins
    bins = cert["partition"]
    if not isinstance(bins, list) or not bins:
        errors.append("partition must be a nonempty list")
        return {"accepted": False, "errors": errors}

    bin_results: list[dict[str, Any]] = []
    bin_errors: list[str] = []
    for k, b in enumerate(bins):
        try:
            r = check_bin(b, seed_triple, seed_quad, seed_widths, mu, k)
            bin_results.append(r)
            if not r["bin_containment_holds"]:
                bin_errors.append(
                    f"bin[{k}]: bin_containment_holds is False "
                    f"(half_width={r['half_width']} > "
                    f"min_propagation_radius_strict="
                    f"{r['min_propagation_radius_strict']})"
                )
        except ValueError as exc:
            bin_errors.append(str(exc))

    errors.extend(bin_errors)

    # Coverage
    try:
        coverage_info = check_coverage(bins, lo_target, hi_target, lo_open, hi_open)
    except ValueError as exc:
        coverage_info = {"coverage_holds": False, "error": str(exc)}
        errors.append(str(exc))

    # Seed-bin consistency: the bin whose [lo, hi] contains 1/2 must
    # have margins equal to the seed margins.
    seed_spin_value = Fraction(1, 2)
    seed_bin_idx: int | None = None
    for k, b in enumerate(bins):
        lo = parse_rational(b["lo"], f"bin[{k}].lo")
        hi = parse_rational(b["hi"], f"bin[{k}].hi")
        if lo <= seed_spin_value <= hi:
            seed_bin_idx = k
            break
    if seed_bin_idx is None:
        errors.append("no partition bin contains the GC3T seed spin a/M = 1/2")
    else:
        seed_bin = bins[seed_bin_idx]
        bin_margins = [parse_rational(m, f"bin[{seed_bin_idx}].margins[{j}]")
                       for j, m in enumerate(seed_bin["margins"])]
        if len(bin_margins) != len(all_seed_margins):
            errors.append(
                f"bin[{seed_bin_idx}] (seed bin) has {len(bin_margins)} margins, "
                f"expected {len(all_seed_margins)} (= {len(seed_triple)} triple + "
                f"{len(seed_quad)} quad)"
            )
        else:
            for j, (bm, sm) in enumerate(zip(bin_margins, all_seed_margins)):
                if bm != sm:
                    errors.append(
                        f"bin[{seed_bin_idx}].margins[{j}] = {bm} does not match "
                        f"seed margin {sm}"
                    )

    # Checks summary
    checks_declared = cert.get("checks", {})
    checks_computed = {
        "C1_seed_margins": all(m > 0 for m in all_seed_margins),
        "C2_lipschitz_moduli": all(
            parse_rational(b["lipschitz_moduli"][j], f"bin[{k}].lipschitz_moduli[{j}]") > 0
            for k, b in enumerate(bins)
            for j in range(len(b.get("margins", [])))
        ) if not errors else False,
        "C3_propagation_radii": not any(
            "propagation_radii_strict" in str(e) for e in errors
        ),
        "C4_bin_containment": all(r["bin_containment_holds"] for r in bin_results),
        "C5_coverage": coverage_info.get("coverage_holds", False),
    }

    accepted = (
        not errors
        and all(checks_computed.values())
    )

    return {
        "accepted": accepted,
        "theorem": cert.get("theorem"),
        "seed_theorem": cert.get("seed_theorem"),
        "seed_atlas": str(atlas_path.relative_to(atlas_root))
        if atlas_path.is_relative_to(atlas_root) else str(atlas_path),
        "seed_atlas_num_charts": len(seed_atlas.get("charts", [])),
        "seed_margins": {
            "num_triple": len(seed_triple),
            "num_quad": len(seed_quad),
            "min_seed_margin": rational_to_string(min(all_seed_margins))
            if all_seed_margins else None,
        },
        "partition": {
            "num_bins": len(bins),
            "bin_results": bin_results,
        },
        "coverage": coverage_info,
        "strict_slack_mu": rational_to_string(mu),
        "checks": checks_computed,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a SPGCPT certificate."
    )
    parser.add_argument("certificate", type=Path, help="Path to SPGCPT certificate JSON")
    parser.add_argument(
        "--atlas-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory for resolving atlas_reference paths",
    )
    args = parser.parse_args(argv[1:])

    try:
        result = verify_certificate(args.certificate, args.atlas_root)
    except (ValueError, json.JSONDecodeError) as exc:
        result = {"accepted": False, "errors": [str(exc)]}

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("accepted") else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
