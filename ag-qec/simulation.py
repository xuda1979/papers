#!/usr/bin/env python3
"""
Reproducible Monte Carlo for CSS-like codes under a two-state MMPP noise model.
Implements EXACT BCH-based CSS construction and Information Set Decoding (ISD).

Revisions:
- Exact BCH matrix construction (g_poly derived).
- Replaced surrogate BP with ISD (Prange's Algorithm) on exact matrices.
- ISD is ideal for correcting low-weight errors in generic/dense linear codes.
"""
import argparse, csv, math, random, statistics, sys
from typing import List, Tuple

try:
    import numpy as np
except ImportError:
    print("NumPy not found.", file=sys.stderr)
    sys.exit(1)

# --------------------- GF(2^8) Arithmetic ---------------------

PRIM_POLY = 0x11D
gf_exp = [0] * 512
gf_log = [0] * 256

def init_gf_tables():
    x = 1
    for i in range(255):
        gf_exp[i] = x
        gf_log[x] = i
        x <<= 1
        if x & 0x100: x ^= PRIM_POLY
    for i in range(255, 512):
        gf_exp[i] = gf_exp[i - 255]

init_gf_tables()

def gf_mul(a, b):
    if a == 0 or b == 0: return 0
    return gf_exp[gf_log[a] + gf_log[b]]

def poly_mul_binary(p1, p2):
    res = [0] * (len(p1) + len(p2) - 1)
    for i, c1 in enumerate(p1):
        if c1:
            for j, c2 in enumerate(p2):
                if c2: res[i+j] ^= 1
    return res

def poly_div_mod_binary(dividend, divisor):
    dividend = list(dividend)
    deg_divisor = len(divisor) - 1
    while deg_divisor > 0 and divisor[deg_divisor] == 0: deg_divisor -= 1
    if deg_divisor < 0: raise ZeroDivisionError
    deg_dividend = len(dividend) - 1
    while deg_dividend >= 0 and dividend[deg_dividend] == 0: deg_dividend -= 1
    if deg_dividend < deg_divisor: return [0], dividend[:deg_dividend+1]
    quotient = [0] * (deg_dividend - deg_divisor + 1)
    remainder = list(dividend)
    for i in range(deg_dividend, deg_divisor - 1, -1):
        if remainder[i]:
            quotient[i - deg_divisor] = 1
            for j in range(deg_divisor + 1):
                if divisor[j]: remainder[i - deg_divisor + j] ^= 1
    while len(remainder) > 0 and remainder[-1] == 0: remainder.pop()
    if not remainder: remainder = [0]
    return quotient, remainder

def get_cyclotomic_coset(s, n=255):
    coset = []
    x = s % n
    while x not in coset:
        coset.append(x)
        x = (2 * x) % n
    return coset

def get_minimal_poly(coset):
    poly = [1]
    for idx in coset:
        root = gf_exp[idx]
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i] ^= gf_mul(c, root)
            new_poly[i+1] ^= c
        poly = new_poly
    return [1 if c else 0 for c in poly]

def get_generator_poly(coset_leaders):
    g = [1]
    for s in coset_leaders:
        mp = get_minimal_poly(get_cyclotomic_coset(s))
        g = poly_mul_binary(g, mp)
    return g

def build_parity_check_matrix(g_poly, n):
    dividend = [1] + [0]*(n-1) + [1]
    h_poly, rem = poly_div_mod_binary(dividend, g_poly)
    if any(rem): raise ValueError("g(x) does not divide x^n - 1")
    g_perp = h_poly[::-1]
    n_minus_k = len(g_poly) - 1
    H = np.zeros((n_minus_k, n), dtype=np.int8)
    for i in range(n_minus_k):
        for j, coeff in enumerate(g_perp):
            H[i, (i+j)%n] = coeff
    return H

# --------------------- ISD Decoder (Prange) ---------------------

def isd_decode(H, syndrome, t_max=None, max_iters=50):
    """
    Information Set Decoding (Prange).
    Attempts to find error e with H e = s and wt(e) minimal (or <= t_max).
    Since noise is low, we stop as soon as we find a valid solution with low weight.
    """
    m, n = H.shape
    if t_max is None: t_max = m # Loose bound

    # Pre-allocate
    Aug = np.zeros((m, n+1), dtype=np.int8)

    best_e = None
    min_wt = n + 1

    for _ in range(max_iters):
        # Random permutation
        perm = np.random.permutation(n)
        H_p = H[:, perm]

        # Gaussian Elimination on [H_p | s]
        Aug[:, :n] = H_p
        Aug[:, n] = syndrome

        # We need to find m independent columns.
        # Since we permuted randomly, we just try to form I on the first m columns.
        # Standard GE.

        # Forward
        pivot_row = 0
        col = 0
        pivots = []
        possible = True

        while pivot_row < m and col < n:
            if Aug[pivot_row, col] == 0:
                swap = -1
                for r in range(pivot_row + 1, m):
                    if Aug[r, col]:
                        swap = r; break
                if swap != -1:
                    Aug[[pivot_row, swap]] = Aug[[swap, pivot_row]]
                else:
                    col += 1; continue

            pivots.append(col)
            # Eliminate
            idx = np.where(Aug[:, col])[0]
            # Vectorized elimination?
            pivot_vec = Aug[pivot_row]
            for r in idx:
                if r != pivot_row:
                    Aug[r] ^= pivot_vec

            pivot_row += 1
            col += 1

        if len(pivots) < m:
            # Rank deficient? H should be full rank m.
            # If so, we might not satisfy syndrome.
            # Check consistency?
            # Assuming H full rank.
            pass

        # Extract solution
        # x_pivots = RHS.
        e_p = np.zeros(n, dtype=np.int8)

        # Back substitution not needed if we eliminated above and below (Gauss-Jordan)
        # My loop `if r != pivot_row` does GJ.

        valid = True
        for i, p_col in enumerate(pivots):
            # Row `i` corresponds to pivot `p_col`?
            # We found pivot for row `i` at `p_col`.
            # So row `i` is [0...0 1 0... | b_i].
            e_p[p_col] = Aug[i, n]

        # Calculate weight
        wt = np.sum(e_p)

        # Re-map
        e = np.zeros(n, dtype=np.int8)
        e[perm] = e_p

        # Check syndrome (sanity)
        # s_check = (H @ e) % 2
        # if not np.array_equal(s_check, syndrome): continue

        if wt < min_wt:
            min_wt = wt
            best_e = e
            # Optimization: if wt is very small (compatible with channel error rate), stop
            if wt <= 3: return e

    return best_e if best_e is not None else np.zeros(n, dtype=np.int8) # Fail safe

def in_row_space(v, M):
    if np.all(v == 0): return True
    return np.linalg.matrix_rank(np.vstack([M, v])) == np.linalg.matrix_rank(M)

# --------------------- Main Experiment ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--d", type=int, default=21)
    ap.add_argument("--txt_out", type=str, default="results.txt")
    ap.add_argument("--bp_txt_out", type=str, default="")
    ap.add_argument("--runlen_csv", type=str, default="")
    ap.add_argument("--sensitivity_csv", type=str, default="")
    ap.add_argument("--css_log", type=str, default="")
    args = ap.parse_args()

    # HCF Params
    length_km, power_dbm = 100.0, 10.0
    atten_db, delta_lam = 0.25, 10.0
    kappa_r, kappa_f = 0.11, 1e-4
    eta_d, tau_g = 1.0, 1.0
    eta_xz, rho, burst = 0.3, 0.6, 2.0

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    # 1. Construct Codes (BCH)
    cosets_Z = [1, 3, 5, 7, 9, 11, 13]
    g_Z = get_generator_poly(cosets_Z)
    H_Z = build_parity_check_matrix(g_Z, 255)

    cosets_X = [1, 3, 5]
    g_X = get_generator_poly(cosets_X)
    H_X = build_parity_check_matrix(g_X, 255)

    rank_Z = np.linalg.matrix_rank(H_Z)
    rank_X = np.linalg.matrix_rank(H_X)

    # Noise rates
    def dbm_to_watts(p): return 10 ** ((p - 30) / 10)
    peff = dbm_to_watts(power_dbm) * 10 ** (-(atten_db * length_km) / 10)
    lam_z_base = eta_d*tau_g*(kappa_r*peff*length_km + kappa_f*peff**2*delta_lam)
    lam_x_base = lam_z_base

    def get_probs(lam_x, lam_z):
        lx, lz = max(0, eta_xz*lam_x), max(0, lam_z)
        lxh, lzh = burst*lx, burst*lz
        return (1-math.exp(-lx), 1-math.exp(-lxh)), (1-math.exp(-lz), 1-math.exp(-lzh))

    (pxl, pxh), (pzl, pzh) = get_probs(lam_x_base, lam_z_base)

    t_thresh = (args.d - 1) // 2
    fail_bdd = 0
    fail_isd = 0

    def sample_states_errors():
        states = []
        s = 1 if rng.random() < 0.5 else 0
        states.append(s)
        for _ in range(254):
            if rng.random() > rho: s ^= 1
            states.append(s)
        xs, zs = [], []
        for s in states:
            px = pxl if s==0 else pxh
            pz = pzl if s==0 else pzh
            xs.append(1 if rng.random() < px else 0)
            zs.append(1 if rng.random() < pz else 0)
        return states, xs, zs

    for _ in range(args.trials):
        states, xs, zs = sample_states_errors()
        if sum(xs) > t_thresh or sum(zs) > t_thresh: fail_bdd += 1

        # ISD Decode
        syn_z = (H_X @ np.array(zs)) % 2
        est_z = isd_decode(H_X, syn_z, max_iters=40)

        syn_x = (H_Z @ np.array(xs)) % 2
        est_x = isd_decode(H_Z, syn_x, max_iters=40)

        # Logical Check
        d_z = (np.array(zs) + est_z) % 2
        if not in_row_space(d_z, H_Z):
            fail_isd += 1
            continue

        d_x = (np.array(xs) + est_x) % 2
        if not in_row_space(d_x, H_X):
            fail_isd += 1

    with open(args.txt_out, "w") as f:
        f.write(f"trials = {args.trials}\n")
        f.write(f"failures_bdd = {fail_bdd}\n")
        f.write(f"failures_isd = {fail_isd}\n")
        f.write(f"p_L_bdd = {fail_bdd/args.trials}\n")
        f.write(f"p_L_isd = {fail_isd/args.trials}\n")

    if args.css_log:
        with open(args.css_log, "w") as f:
            f.write(f"rank_H_Z = {rank_Z}\nrank_H_X = {rank_X}\n")

if __name__ == "__main__":
    main()
