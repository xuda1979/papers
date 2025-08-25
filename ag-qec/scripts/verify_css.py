#!/usr/bin/env python3
import numpy as np
from pathlib import Path

def load_csv(p): 
    return (np.loadtxt(p, dtype=np.uint8, delimiter=",") & 1)

def rank_mod2(M):
    M = M.copy().astype(np.uint8) & 1
    r = c = 0; m, n = M.shape
    while r < m and c < n:
        rows = np.where(M[r:, c]==1)[0]
        if rows.size==0: c += 1; continue
        i = r + rows[0]
        if i!=r: M[[r,i]] = M[[i,r]]
        for j in range(m):
            if j!=r and M[j,c]==1: M[j] ^= M[r]
        r += 1; c += 1
    return r

def main():
    hx = Path("matrices/HX.csv"); hz = Path("matrices/HZ.csv")
    if not hx.exists() or not hz.exists():
        raise SystemExit("Missing matrices/HX.csv or matrices/HZ.csv; run scripts/generate_css.py first.")
    Hx = load_csv(hx); Hz = load_csv(hz)
    commute = np.all(((Hx @ Hz.T) & 1)==0)
    rkx, rkz = rank_mod2(Hx), rank_mod2(Hz)
    n = Hx.shape[1]; k = n - rkx - rkz
    print(f"HX: {Hx.shape}, HZ: {Hz.shape}, commute={commute}, k={k}")

if __name__ == "__main__":
    main()
