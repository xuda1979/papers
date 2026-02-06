"""King walk (8 neighbours) π-identity numerical verification.

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

def build_king_laplacian(R: int) -> sp.csr_matrix:
    """Return sparse Dirichlet Laplacian 𝕂_R for the 8-neighbour walk."""
    N = R * R
    diag_data   = np.ones(N, dtype=np.float64)
    row, col, data = [], [], []

    def idx(x, y):           # (1-based) lattice → 0-based index
        return (y - 1) * R + (x - 1)

    # offsets for the 8 neighbours
    nbrs = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]
    w = -0.125                # −1/8

    for y in range(1, R + 1):
        for x in range(1, R + 1):
            i = idx(x, y)
            for dx, dy in nbrs:
                xx, yy = x + dx, y + dy
                if 1 <= xx <= R and 1 <= yy <= R:
                    j = idx(xx, yy)
                    row.append(i)
                    col.append(j)
                    data.append(w)

    off_diag = sp.csr_matrix((data, (row, col)), shape=(N, N))
    return sp.csr_matrix(sp.eye(N)) + off_diag   # I − (1/8)A


def trace_inverse(lap: sp.csr_matrix) -> float:
    """Trace(𝕂^{-1}) via full eigendecomposition."""
    return dense_trace_inverse(lap)


def king_pi(R: int) -> float:
    lap = build_king_laplacian(R)
    tr_inv = trace_inverse(lap)
    return (2.0 / 3.0) * (R**2 * math.log(R**2)) / tr_inv


def king_pi_hutch(
    R: int,
    *,
    samples: int,
    seed: int,
    rtol: float,
    maxiter: int | None,
) -> tuple[float, float, float, int]:
    lap = build_king_laplacian(R)
    res = hutchinson_trace_inverse(lap, samples=samples, seed=seed, rtol=rtol, maxiter=maxiter)
    pi_est = (2.0 / 3.0) * (R**2 * math.log(R**2)) / res.estimate
    return pi_est, res.estimate, res.std_error, res.cg_iterations_failed


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
        pi_R, tr_hat, se_tr, fails = king_pi_hutch(
            R,
            samples=args.hutch,
            seed=args.seed,
            rtol=args.rtol,
            maxiter=(None if args.maxiter == 0 else args.maxiter),
        )
    else:
        pi_R = king_pi(R)
        tr_hat, se_tr, fails = 0.0, 0.0, 0
    t1 = time.time()

    print(f"π_R  (R={R}) = {pi_R}")
    print(f"actual π     = {math.pi}")
    print(f"abs. error   = {abs(pi_R - math.pi)}")
    if args.hutch and args.hutch > 0:
        rel = (se_tr / abs(tr_hat)) if tr_hat != 0.0 else float("nan")
        se_pi = abs(pi_R) * rel
        print(f"trace est (Hutch) = {tr_hat}")
        print(f"trace std.err (Hutch) = {se_tr}  (rel {rel:.3e})")
        print(f"π std.err (propagated) ≈ {se_pi}  (rel {rel:.3e})")
        print(f"CG nonconvergence count = {fails} / {args.hutch}")
    print(f"time         = {t1 - t0:.2f} s")
