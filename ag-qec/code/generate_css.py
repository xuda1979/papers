#!/usr/bin/env python3
"""
Generate dual-contained CSS parity-checks at n=255 with guaranteed commutation.
Saves Hx, Hz, and logs degrees, ranks, 4-cycles, and commutation check.
"""
from __future__ import annotations
import json, argparse, math, os, time
import numpy as np
from typing import Tuple, Dict

def set_seed(seed: int):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = "0"

def gf2_rank(A: np.ndarray) -> int:
    A = A.copy().astype(np.uint8)
    m, n = A.shape
    r = 0
    col = 0
    for row in range(m):
        while col < n and not A[row:, col].any():
            col += 1
        if col == n:
            break
        pivot = row + np.argmax(A[row:, col])
        if A[pivot, col] == 0:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        # eliminate below
        below = np.where(A[(row+1):, col] == 1)[0] + row + 1
        for rr in below:
            A[rr, :] ^= A[row, :]
        r += 1
        col += 1
    return r

def gf2_nullspace(A: np.ndarray) -> np.ndarray:
    """Return a basis for the nullspace over GF(2) as rows of a matrix."""
    m, n = A.shape
    # RREF with tracking
    A = A.copy().astype(np.uint8)
    pivots = []
    row = 0
    for col in range(n):
        if row >= m: break
        pivot = row + np.argmax(A[row:, col])
        if A[pivot, col] == 0: 
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        pivots.append(col)
        # eliminate other rows
        others = [r for r in range(m) if r != row and A[r, col] == 1]
        for r in others:
            A[r, :] ^= A[row, :]
        row += 1
    free = [j for j in range(n) if j not in pivots]
    if not free:
        return np.zeros((0, n), dtype=np.uint8)
    basis = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8)
        v[f] = 1
        # back-solve for pivot columns
        for r, c in reversed(list(enumerate(pivots))):
            # find row r: A[r, c] == 1 in RREF
            s = 0
            # sum of A[r, free]*v[free] over GF(2):
            for ff in free:
                if A[r, ff] and v[ff]:
                    s ^= 1
            v[c] = s
        basis.append(v)
    return np.vstack(basis).astype(np.uint8)

def random_ldpc_parity(m: int, n: int, col_w: int, max_tries=10) -> np.ndarray:
    """Simple near-regular LDPC parity-check with approximate column weight."""
    for _ in range(max_tries):
        H = np.zeros((m, n), dtype=np.uint8)
        for j in range(n):
            rows = np.random.choice(m, size=col_w, replace=False)
            H[rows, j] ^= 1
        # ensure full row rank (or close)
        if gf2_rank(H) >= min(m, n - 1):
            return H
    return H

def count_4cycles(H: np.ndarray) -> int:
    """Count length-4 cycles naively via pairwise column overlaps."""
    n = H.shape[1]
    cols = [set(np.where(H[:, j])[0]) for j in range(n)]
    cnt = 0
    for a in range(n):
        for b in range(a+1, n):
            if len(cols[a] & cols[b]) >= 2:
                # number of 4-cycles contributed by this pair
                k = len(cols[a] & cols[b])
                cnt += math.comb(k, 2)
    return cnt

def build_css(n: int, rX: int, rZ: int, wX: int = 3, seed: int = 1337) -> Tuple[np.ndarray, np.ndarray, Dict]:
    set_seed(seed)
    # Step 1: random sparse Hx
    Hx = random_ldpc_parity(m=rX, n=n, col_w=wX)
    # Step 2: rows for Hz from nullspace(Hx)
    N = gf2_nullspace(Hx)  # rows span Null(Hx)
    if N.shape[0] < rZ:
        raise ValueError(f"Nullspace too small ({N.shape[0]}) to draw rZ={rZ} rows; decrease rZ or rX.")
    idx = np.random.choice(N.shape[0], size=rZ, replace=False)
    Hz = N[idx, :]
    # Stats
    rankX = gf2_rank(Hx)
    rankZ = gf2_rank(Hz)
    k = n - rankX - rankZ
    # degree stats
    wX_cols = Hx.sum(axis=0).tolist()
    wZ_cols = Hz.sum(axis=0).tolist()
    c4x = count_4cycles(Hx)
    c4z = count_4cycles(Hz)
    # commutation check
    ok = ((Hx @ Hz.T) % 2 == 0).all()
    stats = {
        "n": n, "rX": int(rX), "rZ": int(rZ), "rankX": int(rankX), "rankZ": int(rankZ), "k": int(k),
        "wX_mean": float(np.mean(wX_cols)), "wZ_mean": float(np.mean(wZ_cols)),
        "cycles4_X": int(c4x), "cycles4_Z": int(c4z),
        "commutes": bool(ok), "seed": int(seed)
    }
    return Hx, Hz, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--rX", type=int, required=True)
    ap.add_argument("--rZ", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--wX", type=int, default=3)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--stats", type=str, required=True)
    args = ap.parse_args()

    t0 = time.time()
    Hx, Hz, stats = build_css(n=args.n, rX=args.rX, rZ=args.rZ, wX=args.wX, seed=args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, Hx=Hx.astype(np.uint8), Hz=Hz.astype(np.uint8))
    with open(args.stats, "w") as f:
        json.dump(stats, f, indent=2)
    dt = (time.time() - t0) * 1000
    print(f"[generate_css] saved to {args.out}; k={stats['k']} (rankX={stats['rankX']}, rankZ={stats['rankZ']}), "
          f"commutes={stats['commutes']}; {dt:.1f} ms")

if __name__ == "__main__":
    main()

