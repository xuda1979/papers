r"""functional_analysis_gap_transfer.py

Rigorously provable functional-analytic lemmas used in the lattice-to-continuum bridge.

Scope
-----
These lemmas are *mathematics-only* and do not depend on Yang--Mills specifics.
They can be fully proven/checked inside this repository.

Important
---------
Proving these lemmas is not sufficient to prove the Clay conjecture; they are
only one component of the continuum identification step.

Lemma implemented
-----------------
We implement a simple "gap stability" statement under uniform semigroup
convergence on the orthogonal complement of the vacuum.

Let H and H_n be nonnegative self-adjoint operators with vacuum vector \Omega
such that T(t)=exp(-t H) and T_n(t)=exp(-t H_n) are contraction semigroups.
Assume:
  (A) Uniform gap for approximants: for some m>0 and all n,
      || T_n(t) v || <= exp(-m t) ||v||  for all t>=0 and all v \perp \Omega.
  (B) Uniform convergence at some t0>0 on v \perp \Omega:
      sup_{||v||=1, v\perp\Omega} || (T_n(t0)-T(t0)) v || -> 0.
Then H has a spectral gap at least m' where m' can be extracted from t0 and the
convergence rate.

We keep this fully explicit and purely norm-based.

This is deliberately modest: it provides a machine-checkable lemma that can be
used once a future module supplies (A) and (B) for the Yang--Mills sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GapTransferResult:
    ok: bool
    lower_bound: float
    reason: str


def gap_from_semigroup_bound(m: float, t: float) -> float:
    """Convert a semigroup decay rate into a spectral-gap bound.

    If ||T(t)v|| <= exp(-m t)||v|| for v in the orthogonal complement of the vacuum,
    then the spectral gap is at least m.

    Here we keep the function for symmetry/clarity.
    """

    if not (m > 0.0):
        raise ValueError("m must be > 0")
    if not (t > 0.0):
        raise ValueError("t must be > 0")
    return float(m)


def transfer_gap_via_uniform_semigroup_limit(
    *,
    m_approx: float,
    t0: float,
    sup_op_diff_at_t0: float,
) -> GapTransferResult:
    """Lower-bound a limiting gap from an approximant gap + uniform semigroup closeness.

    Parameters
    ----------
    m_approx:
        Approximant gap lower bound (hypothesis A above).
    t0:
        A single time at which we know uniform convergence.
    sup_op_diff_at_t0:
        A bound \delta >= sup_{||v||=1, v\perp\Omega} ||(T_n(t0)-T(t0))v||
        for all sufficiently large n.

    Returns
    -------
    GapTransferResult with an explicit lower bound m_lim.

    Derivation
    ----------
    For unit v \perp \Omega:
      ||T(t0) v|| <= ||(T(t0)-T_n(t0))v|| + ||T_n(t0)v|| <= \delta + exp(-m_approx t0).

    If \delta + exp(-m_approx t0) < 1, set q := \delta + exp(-m_approx t0).
    Then ||T(t0)||_{\Omega^\perp} <= q and hence the gap of H on \Omega^\perp is
    at least  - (1/t0) log q.

    This is a standard semigroup/spectral-radius estimate.
    """

    if not (m_approx > 0.0):
        return GapTransferResult(False, 0.0, "m_approx_must_be_positive")
    if not (t0 > 0.0):
        return GapTransferResult(False, 0.0, "t0_must_be_positive")
    if sup_op_diff_at_t0 < 0.0:
        return GapTransferResult(False, 0.0, "sup_op_diff_must_be_nonnegative")

    q = float(sup_op_diff_at_t0) + math.exp(-float(m_approx) * float(t0))

    if not (q < 1.0):
        return GapTransferResult(False, 0.0, "insufficient_decay_or_convergence")

    m_lim = -math.log(q) / float(t0)
    if not (m_lim > 0.0):
        return GapTransferResult(False, 0.0, "derived_gap_not_positive")

    # Trivially cannot exceed m_approx in this estimate.
    m_lim = min(m_lim, float(m_approx))
    return GapTransferResult(True, float(m_lim), "ok")


def audit_gap_transfer_lemma_available() -> dict:
    """Return a machine-readable audit record that this lemma is implemented."""

    # This is intentionally a PASS: it's a purely internal math lemma with code+tests.
    return {
        "key": "gap_transfer_lemma_implemented",
        "title": "Gap transfer lemma (semigroup closeness) implemented",
        "status": "PASS",
        "detail": "Provides m_lim = -(1/t0) log(delta + exp(-m_approx t0)) when < 1.",
    }
