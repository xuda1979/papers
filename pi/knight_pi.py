"""Knight walk (8 L-moves) π-identity numerical verification.

Default mode uses dense diagonalization (exact trace) and is only feasible for
moderate R. Use --hutch to estimate Tr(L^{-1}) with Hutchinson + CG for larger R.
"""

import argparse
import math
import sys
import time

import numpy as np
import scipy.sparse as sp

from trace_utils import dense_trace_inverse, hutchinson_trace_inverse


def build_knight_laplacian(R: int) -> sp.csr_matrix:
    """Return sparse Dirichlet Laplacian 𝓛_R = I - P_R for the knight walk."""
    N = R * R
    row: list[int] = []
    col: list[int] = []
    data: list[float] = []

    def idx(x: int, y: int) -> int:  # (1-based) lattice → 0-based index
        return (y - 1) * R + (x - 1)

    nbrs = [
        (2, 1), (2, -1), (-2, 1), (-2, -1),
        (1, 2), (1, -2), (-1, 2), (-1, -2),
    ]
    w = -0.125  # −1/8

    for y in range(1, R + 1):
        for x in range(1, R + 1):
            i = idx(x, y)
            for dx, dy in nbrs:
                xx, yy = x + dx, y + dy
                if 1 <= xx <= R and 1 <= yy <= R:
                    row.append(i)
                    col.append(idx(xx, yy))
                    data.append(w)

    off_diag = sp.csr_matrix((data, (row, col)), shape=(N, N))
    return sp.csr_matrix(sp.eye(N)) + off_diag


def trace_inverse(lap: sp.csr_matrix) -> float:
    """Trace(𝓛^{-1}) via full eigendecomposition (dense)."""
    return dense_trace_inverse(lap)


def knight_pi(R: int) -> float:
    lap = build_knight_laplacian(R)
    tr_inv = trace_inverse(lap)
    return (1.0 / 5.0) * (R**2 * math.log(R**2)) / tr_inv


def knight_pi_hutch(
    R: int,
    *,
    samples: int,
    seed: int,
    rtol: float,
    maxiter: int | None,
) -> tuple[float, float, float, int]:
    lap = build_knight_laplacian(R)
    res = hutchinson_trace_inverse(lap, samples=samples, seed=seed, rtol=rtol, maxiter=maxiter)
    pi_est = (1.0 / 5.0) * (R**2 * math.log(R**2)) / res.estimate
    return float(pi_est), float(res.estimate), res.std_error, res.cg_iterations_failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("R", type=int)
    parser.add_argument("--hutch", type=int, default=0, help="Use Hutchinson estimator with given number of samples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-8, help="CG relative tolerance")
    parser.add_argument("--maxiter", type=int, default=0, help="CG max iterations (0 means default)")
    args = parser.parse_args()

    R = args.R
    t0 = time.time()
    if args.hutch and args.hutch > 0:
        pi_R, tr_hat, se_tr, fails = knight_pi_hutch(
            R,
            samples=args.hutch,
            seed=args.seed,
            rtol=args.rtol,
            maxiter=(None if args.maxiter == 0 else args.maxiter),
        )
    else:
        pi_R = knight_pi(R)
        tr_hat, se_tr, fails = 0.0, 0.0, 0
    t1 = time.time()

    print(f"Knight-π (R={R}): {pi_R}  error={abs(pi_R - math.pi):.3e}  time={t1 - t0:.2f}s")
    if args.hutch and args.hutch > 0:
        rel = (se_tr / abs(tr_hat)) if tr_hat != 0.0 else float("nan")
        se_pi = abs(pi_R) * rel
        print(f"trace est (Hutch) = {tr_hat}")
        print(f"trace std.err (Hutch) = {se_tr}  (rel {rel:.3e})")
        print(f"π std.err (propagated) ≈ {se_pi}  (rel {rel:.3e})")
        print(f"CG nonconvergence count = {fails} / {args.hutch}")
