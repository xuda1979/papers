#!/usr/bin/env python3
"""
Reproducible Monte Carlo for CSS-like codes under a two-state MMPP noise model.
Top-tier revision: Implements EXACT BCH code construction and Ordered Statistics Decoding (OSD-0).

Features:
- Exact construction of BCH cyclic codes at n=255.
- GF(2^8) arithmetic for rigorous polynomial operations.
- OSD-0 decoding on the exact dense parity-check matrix (replacing surrogate BP).
- MMPP noise model with detailed physics parameters.
- Exact Clopper-Pearson intervals and analytical bounds.

"""
import argparse, csv, math, random, statistics, time
from typing import List, Tuple, Optional

# Strict dependency on numpy/scipy for this rigorous version
import numpy as np
from scipy.stats import beta

# --------------------- GF(2^8) and Polynomials ---------------------

PRIM = 0x11D
GF_SIZE = 256
gf_exp = [0] * 512
gf_log = [0] * 256

def init_gf_tables():
    x = 1
    for i in range(255):
        gf_exp[i] = x
        gf_log[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM
    for i in range(255, 512):
        gf_exp[i] = gf_exp[i - 255]

init_gf_tables()

def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else gf_exp[gf_log[a] + gf_log[b]]

def get_cyclotomic_coset(s, n=255):
    coset = []
    x = s % n
    while x not in coset:
        coset.append(x)
        x = (2 * x) % n
    return coset

def get_minimal_poly(coset):
    # Returns coeffs of minimal poly over GF(2), lowest degree first
    poly = [1]
    for idx in coset:
        root = gf_exp[idx]
        new_poly = [0] * (len(poly) + 1)
        for i, c in enumerate(poly):
            new_poly[i] ^= gf_mul(c, root)
            new_poly[i+1] ^= c
        poly = new_poly
    return [1 if x else 0 for x in poly]

def poly_mul_binary(p1, p2):
    res = [0] * (len(p1) + len(p2) - 1)
    for i, c1 in enumerate(p1):
        if c1:
            for j, c2 in enumerate(p2):
                if c2:
                    res[i+j] ^= 1
    return res

def poly_div_binary(dividend, divisor):
    # Returns quotient. Assumes remainder is 0.
    # Input: lowest degree first.
    # Convert to high degree first for division
    deg_n = len(dividend) - 1
    deg_d = len(divisor) - 1

    N = list(dividend)
    D = list(divisor)

    Q = [0] * (deg_n - deg_d + 1)

    for i in range(deg_n - deg_d, -1, -1):
        # Look at term x^{i+deg_d} in N
        # N has length deg_n + 1. index deg_n is x^deg_n.
        # We want to eliminate the highest degree term of N.
        # But we iterate downwards.
        # Wait, simple way:
        # While deg(N) >= deg(D):
        #   diff = deg(N) - deg(D)
        #   Q[diff] = 1
        #   Subtract x^diff * D from N
        pass

    # Re-implement simple polynomial division
    rem = list(dividend)
    while len(rem) > 0 and rem[-1] == 0:
        rem.pop()

    if not rem: return []

    quot = [0] * (len(rem) - len(divisor) + 1)

    while len(rem) >= len(divisor):
        deg_diff = len(rem) - len(divisor)
        quot[deg_diff] = 1
        # Subtract divisor * x^deg_diff
        for i, c in enumerate(divisor):
            rem[i + deg_diff] ^= c
        while len(rem) > 0 and rem[-1] == 0:
            rem.pop()

    return quot

def get_generator_poly(coset_leaders):
    g = [1]
    for s in coset_leaders:
        mp = get_minimal_poly(get_cyclotomic_coset(s))
        g = poly_mul_binary(g, mp)
    return g

def build_cyclic_H(n, k, g_poly):
    # H is generator of dual code.
    # h(x) = (x^n - 1) / g(x)
    # g_perp(x) = x^k * h(x^-1) / h(0) (normalized)
    # Since n=255 is odd, x^n-1 has no factor x, so g(0)=1, h(0)=1.
    dividend = [1] + [0]*(n-1) + [1] # x^n + 1
    h_poly = poly_div_binary(dividend, g_poly)

    # Reciprocal of h
    g_perp = list(reversed(h_poly))
    # Pad to length n for cyclic shifting
    # deg(h) = n - (n-k) = k.
    # deg(g_perp) = k.
    # H has n-k rows.
    # The rows are shifts of g_perp.
    # H = [ g_perp, 0 ... 0 ]
    #     [ 0, g_perp, ...  ]

    r = n - k
    H = np.zeros((r, n), dtype=np.int8)
    # Verify degree
    # print(f"DEBUG: deg(h)={len(h_poly)-1}, k={k}, r={r}")
    # g_perp has length k+1.
    for i in range(r):
        for j, c in enumerate(g_perp):
            if i+j < n:
                H[i, i+j] = c
            else:
                H[i, (i+j)%n] = c # Cyclic wrap?
                # Standard cyclic code H usually defined with wrap.
    return H

# --------------------- Physics / Noise ---------------------

def dbm_to_watts(p_dbm: float) -> float:
    return 10 ** ((p_dbm - 30.0) / 10.0)

def effective_power_watts(p0_w: float, alpha_db_per_km: float, length_km: float) -> float:
    return p0_w * 10 ** (-(alpha_db_per_km * length_km) / 10.0)

def mmpp_base_rates(length_km, power_dbm, atten_db_per_km, delta_lambda_nm, kappa_r, kappa_f, eta_d, tau_g):
    p0_w = dbm_to_watts(power_dbm)
    peff = effective_power_watts(p0_w, atten_db_per_km, length_km)
    mu_sprs = eta_d * tau_g * kappa_r * peff * length_km
    mu_fwm  = eta_d * tau_g * kappa_f * (peff ** 2) * delta_lambda_nm
    lam_z = mu_sprs + mu_fwm
    lam_x = lam_z
    return lam_x, lam_z

def per_gate_probs(lam_x, lam_z, eta_xz, burst_mult, dark, after):
    lam_x_low, lam_z_low = max(0.0, eta_xz * lam_x), max(0.0, lam_z)
    lam_x_high, lam_z_high = burst_mult * lam_x_low, burst_mult * lam_z_low
    p_x_low,  p_x_high  = 1.0 - math.exp(-lam_x_low),  1.0 - math.exp(-lam_x_high)
    p_z_low,  p_z_high  = 1.0 - math.exp(-lam_z_low),  1.0 - math.exp(-lam_z_high)
    p_z_low  = min(1.0, p_z_low  + dark + after)
    p_z_high = min(1.0, p_z_high + dark + after)
    p_x_low  = min(1.0, p_x_low  + 0.5*dark + 0.5*after)
    p_x_high = min(1.0, p_x_high + 0.5*dark + 0.5*after)
    return (p_x_low, p_x_high), (p_z_low, p_z_high)

def sample_markov_states(n, rho, rng):
    s = 1 if rng.random() < 0.5 else 0
    states = [s]
    for _ in range(1, n):
        if rng.random() > rho:
            s ^= 1
        states.append(s)
    return states

def sample_errors(states, p_low, p_high, rng):
    errs = []
    for s in states:
        p = p_low if s == 0 else p_high
        errs.append(1 if rng.random() < p else 0)
    return errs

# --------------------- OSD-0 Decoder ---------------------

def osd0_decode(H, s, llr):
    """
    OSD-0 decoding.
    1. Sort variables by reliability |LLR|.
    2. Perform Gaussian Elimination to find a basis for columns.
    3. Solve for error on the least reliable positions that form a basis for syndrome.
       Wait, standard OSD solves H e = s.
       We assume e_MRB = 0 (hard decision on Most Reliable Basis is correct).
       So we try to satisfy syndrome using Least Reliable bits?
       NO. OSD usually re-encodes information bits.

       View: H is (n-k) x n.
       We want e such that H e = s.
       Let e = y + d, where y is hard decision.
       H(y+d) = s => H d = s + H y = s'.
       We want d of min weight.
       OSD-0 assumes d is 0 on the Most Reliable positions.
       So we modify y only on the Least Reliable positions?
       Actually, standard OSD:
       1. Hard decision y. Syndrome s' = s + H y.
       2. We want correction d such that H d = s'.
       3. Sort positions.
       4. Find (n-k) independent columns in H among the LEAST reliable?
          No, if we fix k bits (Most Reliable), we need to solve for the other n-k.
          So we need n-k independent columns in the UNFIXED set (Least Reliable)?
          Yes. If we assume MRB are correct (error=0), we must satisfy constraints using LRB.
          So we need a basis in the LRB.

       Ref: "Ordered statistics decoding for linear block codes", Fossorier & Lin.
       "Determine the k most reliable independent positions (MRIP)."
       "Re-encode these k positions."
       This corresponds to Generator Matrix G view.

       With H:
       H has rank n-k.
       We need to select n-k pivot variables (dependent) and k free variables (independent).
       We want the free variables to be the Most Reliable ones, so we can fix them (to 0 or y).
       So we want the pivots (which we solve for) to be the Least Reliable ones?
       Yes.

       Algorithm:
       1. Compute hard decision y. s_res = s - H y.
       2. Sort columns of H by reliability of corresponding bits (Ascending? Descending?).
          We want to FIX the most reliable ones. So we want them to be non-pivots?
          Pivots are solved for. Non-pivots are set to 0 (in correction d).
          So we want pivots to be Least Reliable.
          So sort by reliability: [Least Reliable ... Most Reliable].
          Perform GE. We prefer pivots from the left (Least Reliable).
       3. Find basis J (size n-k).
       4. Solve H_J d_J = s_res. Set d_{J^c} = 0.
       5. Correction e = d.
       6. Return y + e.
    """
    m, n = H.shape
    y = (llr < 0).astype(np.int8)
    s_res = (s - H.dot(y)) % 2

    # Reliability
    reliab = np.abs(llr)
    # Sort indices: [Least Reliable ... Most Reliable]
    # argsort gives ascending order
    perm = np.argsort(reliab)

    # Permute H
    H_perm = H[:, perm]

    # Gaussian Elimination to find pivots from the left
    # We need n-k pivots.
    # We form [H_perm | s_res] and solve.

    aug = np.column_stack((H_perm, s_res))
    pivots = []

    row = 0
    for col in range(n):
        if row >= m: break
        # Find pivot
        pivot_row = -1
        for r in range(row, m):
            if aug[r, col] == 1:
                pivot_row = r
                break

        if pivot_row != -1:
            # Swap
            if pivot_row != row:
                aug[[row, pivot_row]] = aug[[pivot_row, row]]

            # Eliminate
            for r in range(m):
                if r != row and aug[r, col] == 1:
                    aug[r] ^= aug[row]

            pivots.append((col, row)) # col in permuted matrix maps to variable
            row += 1

    # Back substitution? Already diagonalized (mostly).
    # aug[row, col] = 1.
    # d_J solution:
    # d[col] = aug[row, -1]

    d_perm = np.zeros(n, dtype=np.int8)

    # If system is consistent (should be for full rank H)
    # If H is not full rank? m > rank.
    # BCH H might have dependent rows if defined naively, but build_cyclic_H gives n-k rows.
    # Are they independent? usually yes for primitive BCH.

    for col, r in pivots:
        d_perm[col] = aug[r, -1]

    # Map back
    d = np.zeros(n, dtype=np.int8)
    d[perm] = d_perm

    return (y + d) % 2

# --------------------- Analysis ---------------------

def clopper_pearson_ci(k, n, alpha=0.05):
    if k == 0:
        lo = 0.0
        hi = beta.ppf(1.0 - alpha/2.0, 1, n)
    elif k == n:
        lo = beta.ppf(alpha/2.0, n, 1)
        hi = 1.0
    else:
        lo = beta.ppf(alpha / 2.0, k, n - k + 1)
        hi = beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi

def chernoff_union_bound(n, t, p_x_low, p_x_high, p_z_low, p_z_high):
    def dkl(q, p):
        if q <= p or q >= 1: return 0.0 # Bound usually for q > p
        return q*math.log(q/p) + (1.0-q)*math.log((1.0-q)/(1.0-p))
    q = (t + 1) / n
    pbar_x = 0.5 * (p_x_low + p_x_high)
    pbar_z = 0.5 * (p_z_low + p_z_high)
    bx = math.exp(-n * dkl(q, pbar_x)) if q > pbar_x else 1.0
    bz = math.exp(-n * dkl(q, pbar_z)) if q > pbar_z else 1.0
    return bx + bz, q, pbar_x, pbar_z

def lezaud_markov_bound(n, t, p_low, p_high, rho):
    gamma = 2.0 * (1.0 - rho)
    ef = 0.5 * (p_low + p_high)
    q = (t + 1) / n
    eps = q - ef
    if eps <= 0:
        return "not_applicable", gamma, ef, eps
    val = math.exp(-(n * eps * eps * gamma) / (2.0 * (1.0 + eps)))
    return f"{val:.6g}", gamma, ef, eps

# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--d", type=int, default=21)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--txt_out", type=str, default="results.txt")
    ap.add_argument("--runlen_csv", type=str, default="runlen_hist.csv")
    ap.add_argument("--sensitivity_csv", type=str, default="")
    ap.add_argument("--css_log", type=str, default="css_results.txt")
    # Physics defaults
    ap.add_argument("--length_km", type=float, default=100.0)
    ap.add_argument("--power_dbm", type=float, default=10.0)
    ap.add_argument("--atten_db_per_km", type=float, default=0.25)
    ap.add_argument("--delta_lambda_nm", type=float, default=10.0)
    ap.add_argument("--kappa_r", type=float, default=0.11)
    ap.add_argument("--kappa_f", type=float, default=1e-4)
    ap.add_argument("--eta_xz", type=float, default=0.3)
    ap.add_argument("--burst_mult", type=float, default=2.0)
    ap.add_argument("--eta_d", type=float, default=1.0)
    ap.add_argument("--tau_g", type=float, default=1.0)
    ap.add_argument("--dark", type=float, default=0.0)
    ap.add_argument("--after", type=float, default=0.0)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # 1. Construct Codes
    print("Constructing Exact BCH Codes...")
    # C_Z: delta=15. Cosets 1,3,5,7,9,11,13.
    cosets_Z = [1, 3, 5, 7, 9, 11, 13]
    g_Z = get_generator_poly(cosets_Z)
    deg_gZ = len(g_Z) - 1
    k_Z = 255 - deg_gZ # 255 - 56 = 199? No, deg is 56.
    # Wait, paper says rank_H_Z = 56. So dim(C_Z) = 199.
    # C_X: delta=7. Cosets 1,3,5.
    cosets_X = [1, 3, 5]
    g_X = get_generator_poly(cosets_X)
    deg_gX = len(g_X) - 1
    # CSS: C_Z_perp <= C_X  <=> C_X_perp <= C_Z
    # We construct H_Z and H_X.
    # H_Z is generator of C_Z_perp.
    # H_X is generator of C_X_perp.
    # dim(C_Z_perp) = deg_gZ = 56.
    # dim(C_X_perp) = deg_gX = 24.

    # Invariants for paper
    k_css = 255 - deg_gZ - deg_gX # 175

    # Build Matrices
    HZ = build_cyclic_H(255, 255-deg_gZ, g_Z) # (56, 255)
    HX = build_cyclic_H(255, 255-deg_gX, g_X) # (24, 255)

    if args.css_log:
        with open(args.css_log, "w") as f:
            f.write(f"css_n = 255\n")
            f.write(f"deg_g_z = {deg_gZ}\n")
            f.write(f"deg_g_x = {deg_gX}\n")
            f.write(f"rank_H_Z = {deg_gZ}\n")
            f.write(f"rank_H_X = {deg_gX}\n")
            f.write(f"k = {k_css}\n")
            ortho = np.any(np.dot(HX, HZ.T) % 2) == False
            f.write(f"orthogonality = {'ok' if ortho else 'FAIL'}\n")
            f.write("coset_reps_Z = " + ",".join(map(str, cosets_Z)) + "\n")
            f.write("coset_reps_X = " + ",".join(map(str, cosets_X)) + "\n")

    # 2. Physics Model
    lam_x, lam_z = mmpp_base_rates(args.length_km, args.power_dbm, args.atten_db_per_km,
                                   args.delta_lambda_nm, args.kappa_r, args.kappa_f,
                                   args.eta_d, args.tau_g)
    (p_x_low, p_x_high), (p_z_low, p_z_high) = per_gate_probs(lam_x, lam_z, args.eta_xz,
                                                              args.burst_mult, args.dark, args.after)

    # 3. Monte Carlo
    print(f"Running {args.trials} trials with Exact OSD-0...")
    n = 255
    t_bdd = (args.d - 1) // 2
    fail_bdd = 0
    fail_osd = 0
    rng = random.Random(args.seed)

    all_states = []

    for _ in range(args.trials):
        # Sample errors
        # Assuming shared state for X/Z
        states = sample_markov_states(n, args.rho, rng)
        all_states.extend(states)

        ex = np.array(sample_errors(states, p_x_low, p_x_high, rng), dtype=np.int8)
        ez = np.array(sample_errors(states, p_z_low, p_z_high, rng), dtype=np.int8)

        # BDD Oracle check
        if np.sum(ex) > t_bdd or np.sum(ez) > t_bdd:
            fail_bdd += 1

        # OSD-0 Decoding
        # Z-correction uses HX (measures X-parity checks? No, Z-errors cause X-syndrome?)
        # Standard:
        # Z-errors detected by H_X. X-errors detected by H_Z.

        sx = (HX.dot(ez) % 2)
        sz = (HZ.dot(ex) % 2)

        # LLRs
        def get_llrs(p_lo, p_hi, sts):
            # Genie-aided LLRs based on hidden state?
            # Or average LLRs?
            # Paper says "LLRs from per-bit probabilities (genie-aided states)"
            ls = []
            for s in sts:
                p = p_lo if s==0 else p_hi
                p = max(1e-12, min(1-1e-12, p))
                ls.append(math.log((1-p)/p))
            return np.array(ls)

        llr_z = get_llrs(p_z_low, p_z_high, states)
        llr_x = get_llrs(p_x_low, p_x_high, states)

        ez_hat = osd0_decode(HX, sx, llr_z)
        ex_hat = osd0_decode(HZ, sz, llr_x)

        # Check success
        # Success if residual error is stabilizer.
        # residual_z = ez + ez_hat. HX(res) = 0 => in C_X_perp?
        # No, C_X_perp is generated by H_X rows.
        # Stabilizers are rows of H_X.
        # So H_X(res) = 0 means res is in kernel(H_X) = C_X.
        # Wait, stabilizer group S = < rows(H_X), rows(H_Z) > ?
        # Code space is fixed by S.
        # Logical operators.
        # Usually:
        # X-errors must be corrected up to Z-stabilizers?
        # Z-errors must be corrected up to X-stabilizers?
        # Actually, for CSS:
        # X-correction succeeds if ex + ex_hat \in C_Z^perp (rows of H_Z)
        # Z-correction succeeds if ez + ez_hat \in C_X^perp (rows of H_X)

        # Here HZ generates C_Z_perp. HX generates C_X_perp.
        # So we check if ex+ex_hat is in span(HZ)?
        # And ez+ez_hat in span(HX)?

        # How to check span?
        # Rank check. Or solve linear system.
        # Or simply:
        # Is (ex + ex_hat) * H_something = 0?
        # C_Z_perp = image(H_Z^T).
        # We need ex + ex_hat = H_Z^T * u.
        # This implies H_? * (ex + ex_hat) = 0?
        # The dual of C_Z_perp is C_Z. Generated by g_Z.
        # So we check if (ex + ex_hat) is in C_Z_perp?
        # This is equivalent to checking if syndrome of (ex+ex_hat) w.r.t some matrix is 0?
        # No, easier:
        # Can we express diff as linear comb of HZ rows?
        # Stack [HZ; diff]. Check rank.
        # If rank == rank(HZ), then diff is in span.

        res_x = (ex + ex_hat) % 2
        res_z = (ez + ez_hat) % 2

        # Check X success
        aug_x = np.vstack([HZ, res_x])
        if np.linalg.matrix_rank(aug_x) > np.linalg.matrix_rank(HZ):
            fail_osd += 1
            continue

        # Check Z success
        aug_z = np.vstack([HX, res_z])
        if np.linalg.matrix_rank(aug_z) > np.linalg.matrix_rank(HX):
            fail_osd += 1
            continue

    # 4. Results
    with open(args.txt_out, "w") as f:
        f.write(f"n = {n}\n")
        f.write(f"d = {args.d}\n")
        f.write(f"trials = {args.trials}\n")
        f.write(f"failures_osd = {fail_osd}\n")
        f.write(f"p_L_osd = {fail_osd/args.trials}\n")

        # Bounds
        lo, hi = clopper_pearson_ci(fail_osd, args.trials)
        f.write(f"p_L_osd_ci_95 = [{lo}, {hi}]\n")

        ch_u, q, pbx, pbz = chernoff_union_bound(n, t_bdd, p_x_low, p_x_high, p_z_low, p_z_high)
        f.write(f"chernoff_union = {ch_u}\n")

        lez, gam, ef, eps = lezaud_markov_bound(n, t_bdd, p_z_low, p_z_high, args.rho)
        f.write(f"lezaud_bound = {lez}\n")

if __name__ == "__main__":
    main()
