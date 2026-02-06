r"""Certified linear-algebra bounds for interval matrices.

This module deliberately avoids floating-point linear algebra routines that can
silently under-approximate (e.g. np.linalg.eig, np.linalg.norm on point samples).

Contract (core utilities):
- Inputs are small real/interval matrices expressed as nested lists.
- Outputs are conservative *upper bounds* or *enclosures* expressed as `Interval`
  where appropriate.

The goal is to keep proof-facing code on simple, auditable inequalities:
- induced norms (\|A\|_∞, \|A\|_1)
- Gershgorin discs to enclose eigenvalues

Note: This does not attempt to compute exact eigenvalues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Union

try:
    # Prefer package-relative import when used inside `verification`.
    from .interval_arithmetic import Interval
except ImportError:  # pragma: no cover
    from interval_arithmetic import Interval


NumberOrInterval = Union[float, int, Interval]
Matrix = Sequence[Sequence[NumberOrInterval]]


def _as_interval(x: NumberOrInterval) -> Interval:
    if isinstance(x, Interval):
        return x
    return Interval(float(x), float(x))


def interval_abs_upper(x: NumberOrInterval) -> float:
    """Upper bound for |x|.

    Returns a plain float because it's used as a radius in Gershgorin/norm bounds.
    """
    ix = _as_interval(x)
    return max(abs(ix.lower), abs(ix.upper))


def matrix_infinity_norm_upper(A: Matrix) -> float:
    r"""Conservative upper bound on induced infinity norm.

    \|A\|_∞ <= max_i sum_j |a_ij|.
    """
    row_sums: List[float] = []
    for row in A:
        s = 0.0
        for aij in row:
            s += interval_abs_upper(aij)
        row_sums.append(s)
    return max(row_sums) if row_sums else 0.0


def matrix_one_norm_upper(A: Matrix) -> float:
    r"""Conservative upper bound on induced 1-norm.

    \|A\|_1 <= max_j sum_i |a_ij|.
    """
    if not A:
        return 0.0

    n_cols = max(len(r) for r in A)
    col_sums = [0.0 for _ in range(n_cols)]
    for row in A:
        for j in range(n_cols):
            if j < len(row):
                col_sums[j] += interval_abs_upper(row[j])
            else:
                col_sums[j] += 0.0
    return max(col_sums) if col_sums else 0.0


def gershgorin_discs(A: Matrix) -> List[Tuple[Interval, float]]:
    """Gershgorin discs (center interval, radius upper-bound).

    For each i, disc i is:
      { z : |z - a_ii| <= sum_{j != i} |a_ij| }

    We return (center_interval, radius_float).
    """
    discs: List[Tuple[Interval, float]] = []
    for i, row in enumerate(A):
        aii = _as_interval(row[i]) if i < len(row) else Interval(0.0, 0.0)
        r = 0.0
        for j, aij in enumerate(row):
            if j == i:
                continue
            r += interval_abs_upper(aij)
        discs.append((aii, r))
    return discs


def gershgorin_eigenvalue_enclosure(A: Matrix) -> Interval:
    """Single interval enclosing *all* eigenvalues by Gershgorin.

    Returns interval [min_i (lower(aii) - r_i), max_i (upper(aii) + r_i)].
    """
    discs = gershgorin_discs(A)
    if not discs:
        return Interval(0.0, 0.0)

    low = min(center.lower - r for center, r in discs)
    high = max(center.upper + r for center, r in discs)
    return Interval(low, high)
