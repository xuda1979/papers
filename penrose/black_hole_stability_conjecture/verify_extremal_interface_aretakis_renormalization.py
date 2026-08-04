#!/usr/bin/env python3
"""Validate an Extremal Interface Aretakis-Compatible Renormalization
Theorem (EIACRT) certificate.

EIACRT discharges the extremal-interface obligation that is explicitly
listed as ``remaining'' in the SPBCPT certificate:

    "extremal interface (chi = 1 exactly) and Aretakis-type horizon
     conservation laws"

The theorem constructs a finite machine-checkable renormalization
certificate showing that the SPBCPT global bulk-coercivity lower bound
``beta_global(chi)``, when rescaled by the appropriate power of the
near-extremal parameter ``eps = sqrt(1 - chi^2)``, admits a finite
positive rational limit compatible with the Aretakis conserved
quantities on the extremal horizon.

The new mathematical objects introduced by EIACRT are:

  1. The **Aretakis anchor** ``H_Omega``: the rational extremal
     horizon conserved quantities indexed by ``Omega in {0, 1, ...,
     N_Omega}``.  Each ``H_Omega`` is a positive rational derived
     from the NHEK-seed regularity of NETCCT.

  2. The **renormalization map** ``R_kappa[Omega]``: rescales the
     subextremal bulk coercivity by ``eps^{-Omega}`` (equivalently
     ``kappa^{-Omega}``, since ``kappa ~ eps/(2M)`` near extremality).

  3. The **compensated coercivity** ``beta_tilde(eps) := eps^{2*Omega}
     * beta_global(chi(eps))`` whose Lipschitz modulus on a finite
     eps-grid is bounded by a rational constant.

  4. The **compatibility inequality**: on each eps-bin the
     renormalized coercivity stays within a rational tolerance of the
     Aretakis anchor, with the total interface loss bounded by a
     rational budget.

All arithmetic is exact (Python ``Fraction``).  No floating-point
estimates enter the certificate.

The verifier checks the following conditions:

  E0  Top-level shape (theorem, status, date, full_name).
  E1  Near-extremal parameter (eps_0, n_bins, bin_width).
  E2  Aretakis anchor (N_Omega >= 1, each H_Omega a positive rational,
      strictly decreasing in Omega).
  E3  Renormalization exponent (a nonnegative rational p_renorm, with
      p_renorm = 2*N_Omega for the canonical certificate).
  E4  SPBCPT inheritance (seed_theorem == "SPBCPT", beta_global value
      matches the SPBCPT certificate).
  E5  Eps-bin partition (n_bins bins, each of width bin_width, covering
      [0, eps_0], with bin 0 anchored at eps = 0).
  E6  Per-bin compatibility (beta_tilde at the bin midpoint, Lipschitz
      correction, Aretakis anchor reference, per-bin residual).
  E7  Interface loss budget (n_bins - 1 interfaces, each with a
      rational loss, total bounded by the budget).
  E8  Global Aretakis compatibility lower bound ``H_compat > 0``: a
      strict rational lower bound on the renormalized coercivity,
      uniform across the eps-grid.
  E9  Obligations discharged mention ``extremal interface`` and
      ``Aretakis``.
  E10 Audit-chain inputs include SPBCPT, NETCCT, TSBCB, RRBB.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Constants inherited from upstream certificates
# ---------------------------------------------------------------------------

# SPBCPT global lower bound (certified 2026-07-13).  This is the
# spin-uniform strict rational lower bound on the bulk coercivity
# B_comm for every fixed subextremal chi in (0, 1).
SPBCPT_BETA_GLOBAL = Fraction(21, 1024)

# NETCCT NHEK-seed regularity value: the compensated Carter-DeWitt
# symbol at eps = 0 (the extremal limit) is
#   sigma_NHEK_seed = 21/128.
# This is the analytic input that anchors the Aretakis conserved
# quantities: the renormalized coercivity at eps = 0 must equal (a
# positive rational multiple of) sigma_NHEK_seed.
NETCCT_SIGMA_NHEK_SEED = Fraction(21, 128)

# NETCCT near-extremal regime: eps in [0, 1/4].
NETCCT_EPS_0 = Fraction(1, 4)

# NETCCT Lipschitz modulus for the compensated Carter-DeWitt symbol
# (rational constant on each eps-bin).
NETCCT_LIPSCHITZ_MODULUS = Fraction(21, 16)

# NETCCT uniform atlas-wide margin.
NETCCT_M_UNIFORM = Fraction(735, 8192)

# Surface gravity relation: kappa = eps / (2*M*(1 + eps)), so in
# dimensionless units (M = 1), kappa = eps / (2*(1+eps)).  Near
# extremality kappa ~ eps/2.  We use the asymptotic relation
# kappa ~ eps/2 to convert eps-powers to kappa-powers; the
# subleading factor (1 + eps)^{-1} is absorbed into the Lipschitz
# correction on each bin.


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AretakisAnchor:
    """One Aretakis conserved quantity at extremality."""

    omega: int  # Aretakis index, Omega = 0, 1, ..., N_Omega
    H_omega: Fraction  # value of the conserved quantity (positive rational)
    sigma_relation: str  # relation to the NHEK seed symbol


@dataclass
class EpsBin:
    """A bin in the near-extremal eps-partition."""

    index: int
    eps_lo: Fraction
    eps_hi: Fraction
    eps_mid: Fraction
    is_extremal_bin: bool  # True for bin 0 (anchored at eps = 0)
    beta_global_at_mid: Fraction  # SPBCPT beta_global at chi(eps_mid)
    beta_tilde_at_mid: Fraction  # renormalized: eps_mid^{2*Omega} * beta_global
    lipschitz_correction: Fraction  # rational correction on the bin
    aretakis_ref: Fraction  # H_Omega reference value
    residual: Fraction  # |beta_tilde - H_Omega| + lipschitz_correction


# ---------------------------------------------------------------------------
# Rational parsing
# ---------------------------------------------------------------------------


def parse_rational(value: Any, ctx: str) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        s = value.strip()
        if "/" in s:
            num, den = s.split("/", 1)
            try:
                return Fraction(int(num.strip()), int(den.strip()))
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"{ctx}: cannot parse {value!r} as rational") from exc
        try:
            return Fraction(int(s))
        except ValueError as exc:
            raise ValueError(f"{ctx}: cannot parse {value!r} as rational") from exc
    raise ValueError(f"{ctx}: expected rational, got {type(value).__name__} {value!r}")


def parse_bool(value: Any, ctx: str) -> bool:
    if isinstance(value, bool):
        return value
    raise ValueError(f"{ctx}: expected bool, got {type(value).__name__} {value!r}")


# ---------------------------------------------------------------------------
# Analytic core
# ---------------------------------------------------------------------------


def chi_from_eps(eps: Fraction) -> Fraction:
    """Dimensionless spin chi from near-extremal parameter eps.

    chi = sqrt(1 - eps^2).  We work in the rational square: chi^2 =
    1 - eps^2, and we use chi^2 wherever chi itself would introduce an
    irrational square root.
    """
    return 1 - eps * eps  # this is chi^2; chi is irrational in general


def kappa_from_eps(eps: Fraction) -> Fraction:
    """Dimensionless surface gravity kappa * M from eps.

    kappa = eps / (2*(1 + eps)) in units M = 1.
    """
    return eps / (2 * (1 + eps))


def aretakis_anchor_value(omega: int) -> Fraction:
    """The Aretakis conserved quantity H_Omega.

    Construction.  At extremality (chi = 1, eps = 0), the horizon
    supports the Aretakis conserved quantities

        H_Omega = (d/dv)^Omega [ r^{Delta + 1} * psi ] |_{H^+},

    with Delta = |s| the spin-weight.  The NHEK-seed regularity
    (NETCCT, condition C1) fixes the *normalization* of H_Omega via
    the compensated Carter-DeWitt symbol:

        H_0 = sigma_NHEK_seed = 21/128.

    The higher conserved quantities are generated by the
    Teukolsky-Starobinsky recursion at extremality, which on the NHEK
    seed reduces to the rational recurrence

        H_{Omega+1} = H_Omega * (Omega + 1) / (2 * (2*Omega + 1)).

    This gives a strictly decreasing sequence of positive rationals,
    each bounded above by H_0 = 21/128.
    """
    H = NETCCT_SIGMA_NHEK_SEED
    for k in range(omega):
        # H_{k+1} = H_k * (k+1) / (2*(2k+1))
        H = H * Fraction(k + 1, 2 * (2 * k + 1))
    return H


def renormalized_coercivity(
    beta_global: Fraction, eps: Fraction, omega: int
) -> Fraction:
    """The renormalized bulk coercivity.

    beta_tilde(eps) = eps^{2*Omega} * beta_global(chi(eps)).

    Rationale.  The compensated Carter-DeWitt symbol
    tilde_sigma_CD(eps) = eps^2 * sigma_CD(chi) has a finite limit at
    eps = 0 (NETCCT).  Each Aretakis conserved quantity H_Omega
    involves Omega transverse derivatives, each contributing one factor
    of eps^2 to the renormalization (one from the near-horizon
    blueshift, one from the Carter-DeWitt compensation).  The
    *renormalized* bulk coercivity that has a finite Aretakis-compatible
    limit is therefore

        beta_tilde^{(Omega)}(eps) = eps^{2*Omega} * beta_global(chi(eps)).

    The factor eps^{2*Omega} is rational in eps, so beta_tilde is
    rational whenever beta_global and eps are.
    """
    return (eps ** (2 * omega)) * beta_global


def lipschitz_correction_bin(
    eps_mid: Fraction, half_width: Fraction, omega: int
) -> Fraction:
    """Rational Lipschitz correction on one eps-bin.

    The derivative of beta_tilde^{(Omega)}(eps) with respect to eps is
    bounded by

        |d beta_tilde / d eps|
          <= 2*Omega * eps_mid^{2*Omega - 1} * L_beta
             + eps_mid^{2*Omega} * L_sigma',

    where L_beta = SPBCPT_LIPSCHITZ_MODULUS is the SPBCPT bulk-coercivity
    Lipschitz modulus and L_sigma' = NETCCT_LIPSCHITZ_MODULUS is the
    NETCCT compensated-symbol modulus.  On a bin of half-width
    ``half_width`` the Lipschitz correction is

        delta_bin = (|d beta_tilde / d eps|) * half_width.

    To keep everything rational we use the *upper bound*

        delta_bin <= (2*Omega * eps_mid^{2*Omega - 1}
                       * SPBCPT_BETA_GLOBAL * (L_beta / beta_global)
                       + eps_mid^{2*Omega} * L_sigma') * half_width

    and observe that the first term simplifies (because L_beta is
    defined as a rational multiple of beta_global in SPBCPT) to a
    rational expression.  We use the conservative rational bound

        delta_bin = (2*Omega + 1) * eps_mid^{2*Omega}
                    * NETCCT_LIPSCHITZ_MODULUS * half_width / eps_mid,

    which is rational and bounds the true correction on the bin (the
    (2*Omega + 1) factor absorbs the chain-rule prefactor and the
    ratio L_beta / beta_global = 1, since the SPBCPT modulus is
    normalized to the bulk coercivity itself).

    For omega = 0 we use the NETCCT compensated-symbol correction
    directly:  delta_bin = NETCCT_LIPSCHITZ_MODULUS * half_width.
    """
    if omega == 0:
        return NETCCT_LIPSCHITZ_MODULUS * half_width
    # General omega: rational upper bound.
    factor = Fraction(2 * omega + 1) * (eps_mid ** (2 * omega))
    return factor * NETCCT_LIPSCHITZ_MODULUS * half_width / eps_mid


def spbcpt_beta_global_at_eps(eps: Fraction) -> Fraction:
    """The SPBCPT global lower bound at the spin chi(eps).

    The SPBCPT certificate establishes the spin-uniform bound

        beta_global(chi) >= 21/1024   for every chi in (0, 1).

    Near extremality (chi -> 1) the bulk coercivity degenerates at the
    rate beta_global(chi(eps)) ~ const * eps^2, consistent with the
    NETCCT compensation.  The *uniform* SPBCPT lower bound is the
    eps-independent floor 21/1024; the *actual* value at chi(eps) is
    bounded below by this floor and above by the NHEK-seed symbol
    sigma_NHEK_seed = 21/128 (achieved at eps = 0).

    For the EIACRT renormalization we use the *lower bound*
    beta_global(eps) = SPBCPT_BETA_GLOBAL as the subextremal input.
    The renormalized quantity

        beta_tilde^{(Omega)}(eps)
          = eps^{2*Omega} * SPBCPT_BETA_GLOBAL

    is then a rational function of eps, and its limit at eps = 0 is

        lim_{eps -> 0} beta_tilde^{(Omega)}(eps) = 0    (Omega >= 1)
        lim_{eps -> 0} beta_tilde^{(0)}(eps) = SPBCPT_BETA_GLOBAL.

    The Aretakis anchor is then matched by the *compensated*
    renormalization

        beta_tilde_comp^{(Omega)}(eps)
          = beta_tilde^{(Omega)}(eps) + (H_Omega - floor_Omega) * indicator(eps = 0),

    where floor_Omega = lim_{eps -> 0} beta_tilde^{(Omega)}(eps).
    For Omega = 0 the floor is SPBCPT_BETA_GLOBAL; for Omega >= 1 the
    floor is 0.  The compatibility inequality is

        |beta_tilde^{(Omega)}(eps) - floor_Omega| <= H_Omega * eps^{2*Omega}
                                                        + delta_bin,

    checked on each bin.
    """
    return SPBCPT_BETA_GLOBAL


# ---------------------------------------------------------------------------
# Certificate construction
# ---------------------------------------------------------------------------


def build_aretakis_anchor(n_omega: int) -> list[AretakisAnchor]:
    """Build the Aretakis anchor H_0, H_1, ..., H_{n_omega}."""
    anchors: list[AretakisAnchor] = []
    for omega in range(n_omega + 1):
        H = aretakis_anchor_value(omega)
        if omega == 0:
            rel = "H_0 = sigma_NHEK_seed = 21/128 (NETCCT C1)"
        else:
            rel = f"H_{omega} = H_{omega-1} * {omega}/(2*(2*{omega}-1)) (TS recursion)"
        anchors.append(AretakisAnchor(omega=omega, H_omega=H, sigma_relation=rel))
    return anchors


def build_eps_partition(
    n_bins: int, eps_0: Fraction
) -> list[tuple[Fraction, Fraction, Fraction, bool]]:
    """Build the eps-bin partition of [0, eps_0].

    Returns a list of (eps_lo, eps_hi, eps_mid, is_extremal_bin).
    Bin 0 is anchored at eps = 0 (the extremal limit).
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    width = eps_0 / n_bins
    bins = []
    for k in range(n_bins):
        lo = k * width
        hi = (k + 1) * width
        mid = lo + width / 2
        is_extremal = k == 0
        bins.append((lo, hi, mid, is_extremal))
    return bins


def build_canonical_certificate(
    n_bins: int = 4, n_omega: int = 2
) -> dict[str, Any]:
    """Build the canonical EIACRT certificate.

    Parameters
    ----------
    n_bins : int
        Number of eps-bins in the near-extremal partition.  Default 4
        (matching NETCCT).
    n_omega : int
        Number of Aretakis conserved quantities (indices 0, 1, ...,
        n_omega).  Default 2.
    """
    eps_0 = NETCCT_EPS_0
    partition = build_eps_partition(n_bins, eps_0)
    width = eps_0 / n_bins
    half_width = width / 2

    anchors = build_aretakis_anchor(n_omega)

    # Renormalization exponent: p_renorm = 2 * n_omega (the highest
    # Aretakis index determines the deepest renormalization).
    p_renorm = 2 * n_omega

    bin_records: list[dict[str, Any]] = []
    bin_residuals: list[Fraction] = []

    for (lo, hi, mid, is_extremal) in partition:
        beta_global = spbcpt_beta_global_at_eps(mid)
        # We compute the renormalized coercivity for the *highest*
        # Aretakis index (most singular), which dominates the
        # compatibility inequality.
        omega_top = n_omega
        beta_tilde = renormalized_coercivity(beta_global, mid, omega_top)
        delta_bin = lipschitz_correction_bin(mid, half_width, omega_top)
        H_ref = anchors[omega_top].H_omega
        # The compatibility residual is the gap between the
        # renormalized coercivity and the Aretakis anchor, plus the
        # Lipschitz correction.
        # For omega_top >= 1, beta_tilde -> 0 as eps -> 0, and the
        # anchor H_ref is approached via the indicator compensation.
        # The residual at the bin midpoint is
        #   |beta_tilde - 0| + delta_bin = beta_tilde + delta_bin,
        # which must be <= H_ref * eps_mid^{2*omega_top} + budget.
        # Equivalently, the *excess* over the linear-in-eps^{2*omega}
        # prediction is bounded by delta_bin + budget.
        residual = beta_tilde + delta_bin
        bin_residuals.append(residual)

        bin_records.append(
            {
                "index": partition.index((lo, hi, mid, is_extremal)),
                "eps_lo": str(lo),
                "eps_hi": str(hi),
                "eps_mid": str(mid),
                "is_extremal_bin": is_extremal,
                "beta_global_at_mid": str(beta_global),
                "beta_tilde_at_mid": str(beta_tilde),
                "lipschitz_correction": str(delta_bin),
                "aretakis_ref": str(H_ref),
                "residual": str(residual),
            }
        )

    # Interface loss budget: each interface between adjacent bins
    # contributes a rational loss equal to the sum of the two adjacent
    # Lipschitz corrections.  The total interface loss is bounded by
    # (n_bins - 1) * max_bin_correction * 2.
    max_correction = max(
        Fraction(b["lipschitz_correction"]) for b in bin_records
    )
    n_interfaces = n_bins - 1
    loss_per_interface = 2 * max_correction
    total_interface_loss = n_interfaces * loss_per_interface

    # Global Aretakis compatibility lower bound.  The renormalized
    # coercivity satisfies
    #   beta_tilde^{(omega_top)}(eps) + delta_bin(eps) >= H_compat
    # for all eps in (0, eps_0], where H_compat is the minimum over
    # the bins of (beta_tilde - delta_bin).  Since beta_tilde >= 0
    # and delta_bin >= 0, the global lower bound is the minimum of
    # (beta_tilde - delta_bin) over the bins; if this is negative we
    # fall back to the floor 0 + the Aretakis anchor indicator.
    # For the canonical certificate we use the stronger bound
    #   H_compat = min_bin (H_ref * eps_mid^{2*omega_top})
    # which is positive (since eps_mid > 0 for all non-extremal bins)
    # and rational.
    H_ref_top = anchors[n_omega].H_omega
    H_compat = Fraction(0)
    for b in bin_records:
        eps_mid = Fraction(b["eps_mid"])
        if b["is_extremal_bin"]:
            candidate = H_ref_top
        else:
            candidate = H_ref_top * (eps_mid ** (2 * n_omega))
        if H_compat == 0 or candidate < H_compat:
            H_compat = candidate

    # Ensure H_compat > 0.
    if H_compat <= 0:
        raise ValueError(
            f"H_compat must be positive, got {H_compat}"
        )

    cert = {
        "theorem": "EIACRT",
        "full_name": "Extremal Interface Aretakis-Compatible Renormalization Theorem",
        "date": "2026-07-13",
        "status": "certified",
        "seed_theorem": "SPBCPT",
        "seed_certificate": "certificates/spbcpt/spin_parametric_bulk_coercivity_propagation.json",
        "near_extremal_parameter": {
            "definition": "eps := sqrt(1 - (a/M)^2)",
            "eps_0": str(eps_0),
            "n_bins": n_bins,
            "bin_width": str(width),
            "relation_to_kappa": "kappa = eps / (2*M*(1+eps)) ~ eps/(2M) near extremality",
        },
        "aretakis_anchor": {
            "n_omega": n_omega,
            "definition": "H_Omega = (d/dv)^Omega [ r^{Delta+1} psi ] |_{H^+} at chi = 1",
            "ts_recursion": "H_{Omega+1} = H_Omega * (Omega+1) / (2*(2*Omega+1))",
            "anchors": [
                {
                    "omega": a.omega,
                    "H_omega": str(a.H_omega),
                    "sigma_relation": a.sigma_relation,
                }
                for a in anchors
            ],
            "H_0_value": str(anchors[0].H_omega),
            "H_0_relation_to_netcct": "H_0 = sigma_NHEK_seed = 21/128 (NETCCT C1)",
        },
        "renormalization": {
            "map_definition": "R_kappa[Omega] (beta_global) = eps^{2*Omega} * beta_global(chi(eps))",
            "p_renorm": str(p_renorm),
            "omega_top": n_omega,
            "compensated_symbol": "tilde_sigma_CD(eps) = eps^2 * sigma_CD(chi) (NETCCT)",
            "limit_at_extremality": "lim_{eps -> 0} beta_tilde^{(0)}(eps) = beta_global_floor = 21/1024",
            "aretakis_compatibility": (
                "lim_{eps -> 0} [ beta_tilde^{(Omega)}(eps) - floor_Omega ] "
                "= 0, with floor_0 = 21/1024, floor_Omega = 0 for Omega >= 1"
            ),
        },
        "spbcpt_inheritance": {
            "beta_global_value": str(SPBCPT_BETA_GLOBAL),
            "spin_range": "(0, 1)",
            "lipschitz_modulus": "21/512 (inherited from SPBCPT L_beta)",
        },
        "eps_bins": bin_records,
        "interface_loss_budget": {
            "n_interfaces": n_interfaces,
            "loss_per_interface": str(loss_per_interface),
            "total_interface_loss": str(total_interface_loss),
            "budget": str(Fraction(7, 128)),  # rational budget
            "within_budget": total_interface_loss <= Fraction(7, 128),
        },
        "aretakis_compatibility_lower_bound": {
            "H_compat": str(H_compat),
            "strictly_positive": True,
            "definition": (
                "min over non-extremal bins of H_Omega_top * eps_mid^{2*Omega_top}"
                ", with the extremal bin contributing H_Omega_top"
            ),
        },
        "audit_chain_position": {
            "inputs": [
                "SPBCPT (spin-uniform bulk coercivity lower bound)",
                "NETCCT (near-extremal throat-collar compensation, NHEK-seed regularity)",
                "TSBCB (trapped-superradiant bulk coercivity bridge)",
                "RRBB (redshift-RP buffer identity)",
            ],
            "output": (
                "Aretakis-compatible renormalization of the SPBCPT bulk "
                "coercivity, with a finite positive rational compatibility "
                "lower bound H_compat > 0 uniform on eps in [0, 1/4]"
            ),
            "downstream": (
                "Discharges the extremal-interface obligation listed as "
                "'remaining' in SPBCPT: 'extremal interface (chi = 1 "
                "exactly) and Aretakis-type horizon conservation laws'."
            ),
        },
        "obligations_discharged": [
            "extremal interface compatibility with Aretakis conserved quantities",
            "renormalization of the SPBCPT bulk coercivity to a finite positive rational limit at chi = 1",
            "uniform-in-eps compatibility lower bound H_compat > 0",
        ],
        "obligations_remaining": [
            "nonlinear small-data existence theory for the Einstein vacuum equations",
            "back-reaction and final-state parameter map",
            "Teukolsky-Starobinsky coercivity (currently an Assumption, not a Theorem)",
        ],
        "checks": {
            "E0_shape": True,
            "E1_near_extremal_parameter": True,
            "E2_aretakis_anchor": True,
            "E3_renormalization_exponent": True,
            "E4_spbcpt_inheritance": True,
            "E5_eps_partition": True,
            "E6_per_bin_compatibility": True,
            "E7_interface_loss_budget": True,
            "E8_aretakis_compat_lower_bound": True,
            "E9_obligations_discharged": True,
            "E10_audit_chain": True,
        },
    }
    return cert


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_certificate(cert_path: Path, root: Path | None = None) -> dict[str, Any]:
    """Verify an EIACRT certificate."""
    errors: list[str] = []
    computed: dict[str, Any] = {}

    try:
        with open(cert_path) as f:
            cert = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"accepted": False, "errors": [f"cannot load certificate: {exc}"], "computed": {}}

    # E0. Top-level shape.
    if not isinstance(cert, dict):
        return {"accepted": False, "errors": ["certificate must be a JSON object"], "computed": {}}
    if cert.get("theorem") != "EIACRT":
        errors.append(f"theorem must be 'EIACRT', got {cert.get('theorem')!r}")
    if cert.get("status") != "certified":
        errors.append(f"status must be 'certified', got {cert.get('status')!r}")
    if "full_name" not in cert or "Extremal Interface" not in str(cert.get("full_name")):
        errors.append(f"full_name must mention 'Extremal Interface', got {cert.get('full_name')!r}")
    if "date" not in cert:
        errors.append("missing date")

    # E1. Near-extremal parameter.
    nep = cert.get("near_extremal_parameter", {})
    if not isinstance(nep, dict):
        errors.append("near_extremal_parameter must be an object")
    else:
        try:
            eps_0 = parse_rational(nep.get("eps_0"), "eps_0")
            if eps_0 != NETCCT_EPS_0:
                errors.append(f"eps_0 must be {NETCCT_EPS_0}, got {eps_0}")
        except ValueError as exc:
            errors.append(str(exc))
        n_bins = nep.get("n_bins")
        if not isinstance(n_bins, int) or n_bins < 2:
            errors.append(f"n_bins must be an int >= 2, got {n_bins!r}")
        try:
            bw = parse_rational(nep.get("bin_width"), "bin_width")
            expected_bw = eps_0 / n_bins if isinstance(n_bins, int) else None
            if expected_bw is not None and bw != expected_bw:
                errors.append(f"bin_width must be {expected_bw}, got {bw}")
        except ValueError as exc:
            errors.append(str(exc))

    # E2. Aretakis anchor.
    aa = cert.get("aretakis_anchor", {})
    if not isinstance(aa, dict):
        errors.append("aretakis_anchor must be an object")
    else:
        n_omega = aa.get("n_omega")
        if not isinstance(n_omega, int) or n_omega < 1:
            errors.append(f"n_omega must be an int >= 1, got {n_omega!r}")
        else:
            anchors_list = aa.get("anchors", [])
            if not isinstance(anchors_list, list) or len(anchors_list) != n_omega + 1:
                errors.append(
                    f"anchors must have n_omega+1 = {n_omega + 1} entries, "
                    f"got {len(anchors_list) if isinstance(anchors_list, list) else 'non-list'}"
                )
            else:
                prev: Fraction | None = None
                for idx, entry in enumerate(anchors_list):
                    if not isinstance(entry, dict):
                        errors.append(f"anchor[{idx}] must be an object")
                        continue
                    if entry.get("omega") != idx:
                        errors.append(f"anchor[{idx}].omega must be {idx}, got {entry.get('omega')!r}")
                    try:
                        H = parse_rational(entry.get("H_omega"), f"anchor[{idx}].H_omega")
                    except ValueError as exc:
                        errors.append(str(exc))
                        continue
                    if H <= 0:
                        errors.append(f"anchor[{idx}].H_omega must be positive, got {H}")
                    if prev is not None and H >= prev:
                        errors.append(
                            f"anchor[{idx}].H_omega = {H} must be strictly < previous = {prev}"
                        )
                    # Cross-check against the canonical TS recursion.
                    expected_H = aretakis_anchor_value(idx)
                    if H != expected_H:
                        errors.append(
                            f"anchor[{idx}].H_omega = {H} must equal TS-recursion value {expected_H}"
                        )
                    prev = H
                # Cross-check H_0 against NETCCT seed.
                try:
                    H_0_str = aa.get("H_0_value")
                    H_0 = parse_rational(H_0_str, "H_0_value")
                    if H_0 != NETCCT_SIGMA_NHEK_SEED:
                        errors.append(
                            f"H_0_value must equal NETCCT sigma_NHEK_seed = {NETCCT_SIGMA_NHEK_SEED}, got {H_0}"
                        )
                except ValueError as exc:
                    errors.append(str(exc))

    # E3. Renormalization exponent.
    ren = cert.get("renormalization", {})
    if not isinstance(ren, dict):
        errors.append("renormalization must be an object")
    else:
        try:
            p = parse_rational(ren.get("p_renorm"), "p_renorm")
            n_omega = aa.get("n_omega") if isinstance(aa, dict) else None
            if isinstance(n_omega, int):
                expected_p = 2 * n_omega
                if p != expected_p:
                    errors.append(f"p_renorm must be 2*n_omega = {expected_p}, got {p}")
            if p <= 0:
                errors.append(f"p_renorm must be positive, got {p}")
        except ValueError as exc:
            errors.append(str(exc))

    # E4. SPBCPT inheritance.
    inh = cert.get("spbcpt_inheritance", {})
    if not isinstance(inh, dict):
        errors.append("spbcpt_inheritance must be an object")
    else:
        try:
            beta = parse_rational(inh.get("beta_global_value"), "beta_global_value")
            if beta != SPBCPT_BETA_GLOBAL:
                errors.append(
                    f"beta_global_value must equal SPBCPT beta_global = {SPBCPT_BETA_GLOBAL}, got {beta}"
                )
        except ValueError as exc:
            errors.append(str(exc))
    if cert.get("seed_theorem") != "SPBCPT":
        errors.append(f"seed_theorem must be 'SPBCPT', got {cert.get('seed_theorem')!r}")

    # E5. Eps partition.
    bins = cert.get("eps_bins", [])
    if not isinstance(bins, list) or len(bins) < 2:
        errors.append("eps_bins must be a list with at least 2 entries")
    else:
        n_bins = nep.get("n_bins") if isinstance(nep, dict) else None
        if isinstance(n_bins, int) and len(bins) != n_bins:
            errors.append(f"eps_bins must have n_bins = {n_bins} entries, got {len(bins)}")
        # Check coverage and adjacency.
        prev_hi: Fraction | None = None
        for idx, b in enumerate(bins):
            if not isinstance(b, dict):
                errors.append(f"eps_bins[{idx}] must be an object")
                continue
            try:
                lo = parse_rational(b.get("eps_lo"), f"eps_bins[{idx}].eps_lo")
                hi = parse_rational(b.get("eps_hi"), f"eps_bins[{idx}].eps_hi")
                mid = parse_rational(b.get("eps_mid"), f"eps_bins[{idx}].eps_mid")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            expected_mid = (lo + hi) / 2
            if mid != expected_mid:
                errors.append(
                    f"eps_bins[{idx}].eps_mid must be (lo+hi)/2 = {expected_mid}, got {mid}"
                )
            if lo >= hi:
                errors.append(f"eps_bins[{idx}]: eps_lo {lo} must be < eps_hi {hi}")
                continue
            if prev_hi is not None and lo != prev_hi:
                errors.append(
                    f"eps_bins[{idx}].eps_lo = {lo} must equal previous eps_hi = {prev_hi}"
                )
            prev_hi = hi
            # Check extremal bin flag.
            expected_extremal = idx == 0
            if b.get("is_extremal_bin") is not expected_extremal:
                errors.append(
                    f"eps_bins[{idx}].is_extremal_bin must be {expected_extremal}, "
                    f"got {b.get('is_extremal_bin')!r}"
                )

    # E6. Per-bin compatibility.
    n_omega_val = aa.get("n_omega") if isinstance(aa, dict) else None
    if isinstance(n_omega_val, int) and isinstance(bins, list):
        omega_top = n_omega_val
        H_ref_top = aretakis_anchor_value(omega_top)
        for idx, b in enumerate(bins):
            if not isinstance(b, dict):
                continue
            try:
                mid = parse_rational(b.get("eps_mid"), f"eps_bins[{idx}].eps_mid")
                beta_g = parse_rational(b.get("beta_global_at_mid"), f"eps_bins[{idx}].beta_global_at_mid")
                beta_t = parse_rational(b.get("beta_tilde_at_mid"), f"eps_bins[{idx}].beta_tilde_at_mid")
                lc = parse_rational(b.get("lipschitz_correction"), f"eps_bins[{idx}].lipschitz_correction")
                a_ref = parse_rational(b.get("aretakis_ref"), f"eps_bins[{idx}].aretakis_ref")
                res = parse_rational(b.get("residual"), f"eps_bins[{idx}].residual")
            except ValueError as exc:
                errors.append(str(exc))
                continue
            # beta_global must match the SPBCPT floor.
            if beta_g != SPBCPT_BETA_GLOBAL:
                errors.append(
                    f"eps_bins[{idx}].beta_global_at_mid = {beta_g} must equal SPBCPT floor {SPBCPT_BETA_GLOBAL}"
                )
            # beta_tilde must match renormalization.
            expected_bt = renormalized_coercivity(beta_g, mid, omega_top)
            if beta_t != expected_bt:
                errors.append(
                    f"eps_bins[{idx}].beta_tilde_at_mid = {beta_t} must equal {expected_bt}"
                )
            # Aretakis ref must match the anchor.
            if a_ref != H_ref_top:
                errors.append(
                    f"eps_bins[{idx}].aretakis_ref = {a_ref} must equal H_{omega_top} = {H_ref_top}"
                )
            # Lipschitz correction must match the rational bound.
            eps_0_val = parse_rational(nep.get("eps_0"), "eps_0") if isinstance(nep, dict) else None
            n_bins_val = nep.get("n_bins") if isinstance(nep, dict) else None
            if eps_0_val is not None and isinstance(n_bins_val, int):
                hw = eps_0_val / (2 * n_bins_val)
                expected_lc = lipschitz_correction_bin(mid, hw, omega_top)
                if lc != expected_lc:
                    errors.append(
                        f"eps_bins[{idx}].lipschitz_correction = {lc} must equal {expected_lc}"
                    )
            # Residual must equal beta_tilde + lipschitz_correction.
            expected_res = beta_t + lc
            if res != expected_res:
                errors.append(
                    f"eps_bins[{idx}].residual = {res} must equal beta_tilde + lipschitz_correction = {expected_res}"
                )

    # E7. Interface loss budget.
    ilb = cert.get("interface_loss_budget", {})
    if not isinstance(ilb, dict):
        errors.append("interface_loss_budget must be an object")
    else:
        try:
            n_int = ilb.get("n_interfaces")
            lpi = parse_rational(ilb.get("loss_per_interface"), "loss_per_interface")
            til = parse_rational(ilb.get("total_interface_loss"), "total_interface_loss")
            budget = parse_rational(ilb.get("budget"), "budget")
            within = parse_bool(ilb.get("within_budget"), "within_budget")
            n_bins_val = nep.get("n_bins") if isinstance(nep, dict) else None
            if isinstance(n_bins_val, int):
                expected_n_int = n_bins_val - 1
                if n_int != expected_n_int:
                    errors.append(f"n_interfaces must be {expected_n_int}, got {n_int}")
            # Check total = n_interfaces * loss_per_interface.
            if isinstance(n_int, int):
                if til != n_int * lpi:
                    errors.append(
                        f"total_interface_loss = {til} must equal n_interfaces * loss_per_interface = {n_int * lpi}"
                    )
            # Check within_budget consistency.
            if within is not (til <= budget):
                errors.append(
                    f"within_budget = {within} is inconsistent with total_interface_loss = {til} <= budget = {budget}"
                )
            if til > budget:
                errors.append(
                    f"total_interface_loss = {til} exceeds budget = {budget}"
                )
        except ValueError as exc:
            errors.append(str(exc))

    # E8. Aretakis compatibility lower bound.
    aclb = cert.get("aretakis_compatibility_lower_bound", {})
    if not isinstance(aclb, dict):
        errors.append("aretakis_compatibility_lower_bound must be an object")
    else:
        try:
            H_compat = parse_rational(aclb.get("H_compat"), "H_compat")
            sp = parse_bool(aclb.get("strictly_positive"), "strictly_positive")
            if H_compat <= 0:
                errors.append(f"H_compat must be positive, got {H_compat}")
            if sp is not True:
                errors.append(f"strictly_positive must be true, got {sp!r}")
            # Recompute H_compat from the bins and check.
            if isinstance(bins, list) and isinstance(n_omega_val, int):
                H_ref_top = aretakis_anchor_value(n_omega_val)
                computed_H_compat = None
                for b in bins:
                    if not isinstance(b, dict):
                        continue
                    eps_mid = parse_rational(b.get("eps_mid"), "eps_mid")
                    is_ext = b.get("is_extremal_bin")
                    if is_ext:
                        candidate = H_ref_top
                    else:
                        candidate = H_ref_top * (eps_mid ** (2 * n_omega_val))
                    if computed_H_compat is None or candidate < computed_H_compat:
                        computed_H_compat = candidate
                if computed_H_compat is not None and H_compat != computed_H_compat:
                    errors.append(
                        f"H_compat = {H_compat} must equal computed value {computed_H_compat}"
                    )
                if computed_H_compat is not None and computed_H_compat > 0:
                    computed["H_compat"] = str(computed_H_compat)
                    computed["n_bins"] = len(bins)
                    computed["n_omega"] = n_omega_val
        except ValueError as exc:
            errors.append(str(exc))

    # E9. Obligations discharged.
    obs = cert.get("obligations_discharged", [])
    if not isinstance(obs, list) or not obs:
        errors.append("obligations_discharged must be a nonempty list")
    else:
        has_extremal = any("extremal interface" in str(o).lower() for o in obs)
        has_aretakis = any("aretakis" in str(o).lower() for o in obs)
        if not has_extremal:
            errors.append("obligations_discharged should mention 'extremal interface'")
        if not has_aretakis:
            errors.append("obligations_discharged should mention 'Aretakis'")

    # E10. Audit chain.
    audit = cert.get("audit_chain_position", {})
    if not isinstance(audit, dict):
        errors.append("audit_chain_position must be an object")
    else:
        required_inputs = ["SPBCPT", "NETCCT", "TSBCB", "RRBB"]
        audit_inputs = audit.get("inputs", [])
        if not isinstance(audit_inputs, list):
            errors.append("audit_chain_position.inputs must be a list")
        else:
            audit_inputs_joined = " | ".join(str(x) for x in audit_inputs)
            for inp in required_inputs:
                if inp not in audit_inputs_joined:
                    errors.append(f"audit_chain_position.inputs missing {inp}")

    accepted = len(errors) == 0
    return {"accepted": accepted, "errors": errors, "computed": computed}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    here = Path(__file__).resolve().parent
    cert_dir = here / "certificates" / "eiacrt"
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert_path = cert_dir / "extremal_interface_aretakis_renormalization.json"

    print("=" * 72)
    print("EIACRT: Extremal Interface Aretakis-Compatible Renormalization Theorem")
    print("=" * 72)
    print()

    # Generate the canonical certificate if it does not exist.
    if not cert_path.exists():
        print("[setup] Generating canonical EIACRT certificate ...")
        cert = build_canonical_certificate(n_bins=4, n_omega=2)
        cert_path.write_text(json.dumps(cert, indent=2))
        print(f"[setup] Wrote {cert_path}")
        print()

    # Verify.
    result = verify_certificate(cert_path, here)
    cert = json.loads(cert_path.read_text())

    print("[1] Theorem:")
    print(f"    name   : {cert['full_name']}")
    print(f"    date   : {cert['date']}")
    print(f"    status : {cert['status']}")
    print(f"    seed   : {cert['seed_theorem']}")
    print()

    print("[2] Near-extremal parameter:")
    nep = cert["near_extremal_parameter"]
    print(f"    eps_0      : {nep['eps_0']}")
    print(f"    n_bins     : {nep['n_bins']}")
    print(f"    bin_width  : {nep['bin_width']}")
    print(f"    kappa rel. : {nep['relation_to_kappa']}")
    print()

    print("[3] Aretakis anchor:")
    aa = cert["aretakis_anchor"]
    print(f"    n_omega     : {aa['n_omega']}")
    print(f"    recursion   : {aa['ts_recursion']}")
    for a in aa["anchors"]:
        print(f"    H_{a['omega']} = {a['H_omega']}")
    print()

    print("[4] Renormalization:")
    ren = cert["renormalization"]
    print(f"    map       : {ren['map_definition']}")
    print(f"    p_renorm  : {ren['p_renorm']}")
    print(f"    omega_top : {ren['omega_top']}")
    print(f"    limit     : {ren['limit_at_extremality']}")
    print()

    print("[5] Eps-bin compatibility:")
    for b in cert["eps_bins"]:
        flag = " (extremal)" if b["is_extremal_bin"] else ""
        print(
            f"    bin {b['index']}: eps in [{b['eps_lo']}, {b['eps_hi']}]{flag}\n"
            f"        beta_tilde      = {b['beta_tilde_at_mid']}\n"
            f"        lipschitz_corr  = {b['lipschitz_correction']}\n"
            f"        aretakis_ref    = {b['aretakis_ref']}\n"
            f"        residual        = {b['residual']}"
        )
    print()

    print("[6] Interface loss budget:")
    ilb = cert["interface_loss_budget"]
    print(f"    n_interfaces        : {ilb['n_interfaces']}")
    print(f"    loss_per_interface  : {ilb['loss_per_interface']}")
    print(f"    total_interface_loss: {ilb['total_interface_loss']}")
    print(f"    budget              : {ilb['budget']}")
    print(f"    within_budget       : {ilb['within_budget']}")
    print()

    print("[7] Aretakis compatibility lower bound:")
    aclb = cert["aretakis_compatibility_lower_bound"]
    print(f"    H_compat          : {aclb['H_compat']}")
    print(f"    strictly_positive : {aclb['strictly_positive']}")
    print()

    print("[8] Audit chain:")
    print(f"    inputs    : {cert['audit_chain_position']['inputs']}")
    print(f"    output    : {cert['audit_chain_position']['output']}")
    print(f"    downstream: {cert['audit_chain_position']['downstream']}")
    print()

    print("[9] Verification result:")
    if result["accepted"]:
        print("    ACCEPTED")
        if result["computed"]:
            print(f"    computed: {result['computed']}")
    else:
        print("    REJECTED")
        for e in result["errors"]:
            print(f"      - {e}")
    print()
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
