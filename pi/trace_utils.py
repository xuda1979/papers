"""Utilities for computing/estimating Tr(A^{-1}) for SPD sparse matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class HutchinsonResult:
    estimate: float
    std_error: float
    samples: int
    cg_iterations_failed: int


def dense_trace_inverse(A: sp.spmatrix) -> float:
    """Compute Tr(A^{-1}) exactly via dense eigendecomposition.

    This is O(n^3) and only feasible for small matrices.
    """
    eigvals = scipy.linalg.eigvalsh(A.todense())
    return float(np.sum(1.0 / eigvals))


def hutchinson_trace_inverse(
    A: sp.spmatrix,
    *,
    samples: int = 50,
    seed: int = 0,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: Optional[int] = None,
) -> HutchinsonResult:
    """Estimate Tr(A^{-1}) using Hutchinson with CG solves.

    Uses Rademacher vectors v (entries ±1). For each v, solves A x = v and
    accumulates v^T x. Assumes A is symmetric positive definite.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")

    n = int(A.shape[0])
    rng = np.random.default_rng(seed)

    estimates = np.empty(samples, dtype=np.float64)
    fails = 0

    # Ensure CSR/CSC for fast matvec in CG
    if not sp.isspmatrix(A):
        A = sp.csr_matrix(A)
    else:
        A = A.tocsr()

    for i in range(samples):
        v = rng.integers(0, 2, size=n, dtype=np.int8)
        v = (2.0 * v - 1.0).astype(np.float64, copy=False)  # ±1

        x, info = spla.cg(A, v, rtol=rtol, atol=atol, maxiter=maxiter)
        if info != 0:
            fails += 1
        estimates[i] = float(v @ x)

    est = float(estimates.mean())
    # Standard error of mean
    std_err = float(estimates.std(ddof=1) / np.sqrt(samples)) if samples >= 2 else 0.0

    return HutchinsonResult(
        estimate=est,
        std_error=std_err,
        samples=samples,
        cg_iterations_failed=fails,
    )
