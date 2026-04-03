"""Triangular walk (6 neighbours) π-identity numerical verification.

Default mode uses dense diagonalization (exact trace) and is only feasible for
moderate R. Use --hutch to estimate Tr(L^{-1}) with Hutchinson + CG for larger R.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np
import scipy.sparse as sp

from trace_utils import dense_trace_inverse, hutchinson_trace_inverse

# --- build triangular Laplacian ----------------------------------------------
def build_tri_lap(R:int) -> sp.csr_matrix:
    N = R*R
    main  = np.ones(N)
    row, col, data = [], [], []
    idx = lambda x,y: (y-1)*R + (x-1)

    nbrs = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for y in range(1,R+1):
        for x in range(1,R+1):
            i = idx(x,y)
            for dx,dy in nbrs:
                xx,yy = x+dx, y+dy
                if 1<=xx<=R and 1<=yy<=R:
                    row.append(i); col.append(idx(xx,yy)); data.append(-1/6)
    return sp.csr_matrix((data,(row,col)), shape=(N,N)) + sp.eye(N)

# --- trace inverse via eigvalsh --------------------------------------------------
def trace_inv(lap):   # full spectrum for small R
    if lap.shape[0] > 10000:
        print(
            "Warning: matrix is large; dense diagonalization may be slow/memory intensive. "
            "Consider using --hutch."
        )
    return dense_trace_inverse(lap)

def tri_pi(R:int):
    lap = build_tri_lap(R)
    tr  = trace_inv(lap)
    return (np.sqrt(3)/2) * (R**2 * math.log(R**2)) / tr


def tri_pi_hutch(
    R: int,
    *,
    samples: int,
    seed: int,
    rtol: float,
    maxiter: int | None,
) -> tuple[float, float, float, int]:
    lap = build_tri_lap(R)
    res = hutchinson_trace_inverse(lap, samples=samples, seed=seed, rtol=rtol, maxiter=maxiter)
    pi_est = (np.sqrt(3) / 2.0) * (R**2 * math.log(R**2)) / res.estimate
    return float(pi_est), float(res.estimate), res.std_error, res.cg_iterations_failed

# --- main ---------------------------------------------------------------------
if __name__=="__main__":
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
        val, tr_hat, se_tr, fails = tri_pi_hutch(
            R,
            samples=args.hutch,
            seed=args.seed,
            rtol=args.rtol,
            maxiter=(None if args.maxiter == 0 else args.maxiter),
        )
    else:
        val = tri_pi(R)
        tr_hat, se_tr, fails = 0.0, 0.0, 0
    t1 = time.time()
    print(f"Tri-π (R={R}): {val}  error={abs(val-math.pi):.3e}  time={t1-t0:.2f}s")
    if args.hutch and args.hutch > 0:
        rel = (se_tr / abs(tr_hat)) if tr_hat != 0.0 else float("nan")
        se_pi = abs(val) * rel
        print(f"trace est (Hutch) = {tr_hat}")
        print(f"trace std.err (Hutch) = {se_tr}  (rel {rel:.3e})")
        print(f"π std.err (propagated) ≈ {se_pi}  (rel {rel:.3e})")
        print(f"CG nonconvergence count = {fails} / {args.hutch}")
