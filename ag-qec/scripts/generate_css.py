#!/usr/bin/env python3
import argparse, os
from pathlib import Path
import numpy as np

def set_seed(s: int):
    np.random.seed(s % (2**32-1))

def mod2(A): 
    return (A.astype(np.uint8) & 1)

def row_echelon_mod2(M):
    M = mod2(M.copy()); r, c = 0, 0
    m, n = M.shape; piv = []
    while r < m and c < n:
        pivot_rows = np.where(M[r:, c]==1)[0]
        if pivot_rows.size == 0:
            c += 1; continue
        i = r + pivot_rows[0]
        if i != r: M[[r,i]] = M[[i,r]]
        for j in range(m):
            if j!=r and M[j,c]==1: M[j] ^= M[r]
        piv.append(c); r += 1; c += 1
    return M, piv

def rank_mod2(M) -> int:
    return len(row_echelon_mod2(M)[1])

def nullspace_mod2(M):
    R, piv = row_echelon_mod2(M)
    m, n = R.shape
    free = [c for c in range(n) if c not in piv]
    B = []
    for f in free:
        v = np.zeros(n, dtype=np.uint8); v[f]=1
        for i in range(len(piv)-1, -1, -1):
            pc = piv[i]
            s = (R[i] & v).sum() & 1
            v[pc] = s
        B.append(v)
    return np.array(B, dtype=np.uint8) if B else np.zeros((0,n), dtype=np.uint8)

def gen_ldpc_hx(n: int, rx: int, col_w: int) -> np.ndarray:
    # Build sparse H with target column weight then trim to rank rx
    H = np.zeros((max(rx+8, rx), n), dtype=np.uint8)
    for j in range(n):
        rows = np.random.choice(H.shape[0], size=col_w, replace=False)
        H[rows, j] ^= 1
    # Grow/trim until rank >= rx, then select independent rows
    while rank_mod2(H) < rx:
        extra = np.zeros((1,n), dtype=np.uint8)
        cols = np.random.choice(n, size=col_w, replace=False)
        extra[0, cols] = 1; H = np.vstack([H, extra])
    # Select rx independent rows
    sel = []
    R = np.zeros((0,n), dtype=np.uint8)
    for i in range(H.shape[0]):
        cand = np.vstack([R, H[i:i+1]])
        if rank_mod2(cand) > rank_mod2(R):
            R = cand; sel.append(i)
        if R.shape[0]==rx: break
    return R

def pick_hz_from_nullspace(Hx: np.ndarray, rz: int, col_w_hint: int) -> np.ndarray:
    N = nullspace_mod2(Hx)  # basis vectors
    d, n = N.shape
    rows = []
    R = np.zeros((0,n), dtype=np.uint8)
    tries = 0
    while len(rows) < rz and tries < 10000:
        k = np.random.randint(1, min(col_w_hint+2, max(2,d)))
        coeff = (np.random.rand(d) < (k/d)).astype(np.uint8)
        v = mod2(coeff @ N)  # random combo
        if v.sum()==0: 
            tries += 1; continue
        cand = np.vstack([R, v])
        if rank_mod2(cand) > rank_mod2(R):
            R = cand; rows.append(v)
        tries += 1
    if len(rows) < rz:
        raise RuntimeError("Failed to assemble independent HZ rows from nullspace; relax rz or tweak seed.")
    return np.vstack(rows)

def save_csv(path: Path, A: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, A.astype(np.uint8), fmt="%d", delimiter=",")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20250825)
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--rx", type=int, default=111)
    ap.add_argument("--rz", type=int, default=111)
    ap.add_argument("--col_wx", type=int, default=3)
    ap.add_argument("--col_wz", type=int, default=3)
    ap.add_argument("--outdir", type=str, default="matrices")
    args = ap.parse_args()

    set_seed(args.seed)
    n = args.n
    Hx = gen_ldpc_hx(n, args.rx, args.col_wx)
    Hz = pick_hz_from_nullspace(Hx, args.rz, args.col_wz)
    assert Hx.shape[1]==Hz.shape[1]==n
    # Verify CSS commutation
    css_ok = np.all(((Hx @ Hz.T) & 1) == 0)
    rkx, rkz = rank_mod2(Hx), rank_mod2(Hz)
    k = n - rkx - rkz
    print(f"n={n}, rank(HX)={rkx}, rank(HZ)={rkz}, k={k}, CSS_commute={css_ok}")
    out = Path(args.outdir)
    save_csv(out/"HX.csv", Hx)
    save_csv(out/"HZ.csv", Hz)

if __name__ == "__main__":
    main()
