#!/usr/bin/env python3
"""
Reproducible Monte Carlo for CSS-like codes (AG/BCH) under MMPP noise.

Improvements:
- Uses EXACT BCH parity check matrices (dense) instead of surrogate random graphs.
- Implements rigorous GF(2^8) arithmetic for polynomial construction.
- Uses OSD-2 (Ordered Statistics Decoding) to handle dense matrices and correct errors in the information set.
- Maintains all previous logging and analysis features.
"""
import argparse, csv, math, random, statistics, time, itertools
from typing import List, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.stats import beta
except Exception:
    beta = None

# --------------------- GF(2^8) Arithmetic ---------------------
PRIM_POLY = 0x11D

class GF256:
    def __init__(self, val):
        self.val = val
    def __add__(self, other):
        return GF256(self.val ^ other.val)
    def __mul__(self, other):
        a, b, p = self.val, other.val, 0
        for _ in range(8):
            if b & 1: p ^= a
            hi = a & 0x80
            a <<= 1
            if hi: a ^= PRIM_POLY
            b >>= 1
        return GF256(p & 0xFF)
    def __repr__(self): return f"{self.val}"

def poly_mul(p1, p2):
    deg = (len(p1)-1) + (len(p2)-1)
    res = [GF256(0) for _ in range(deg + 1)]
    for i in range(len(p1)):
        for j in range(len(p2)):
            term = p1[i] * p2[j]
            res[i+j] = res[i+j] + term
    return res

def poly_div_gf2_bits(num, den):
    num = list(num)
    den = list(den)
    while len(den)>0 and den[-1]==0: den.pop()
    if not den: raise ZeroDivisionError
    deg_d = len(den)-1
    diff = len(num)-1 - deg_d
    quo = [0]*(diff+1) if diff>=0 else [0]
    while len(num) >= len(den):
        if num[-1]==0:
            num.pop()
            continue
        shift = len(num)-1 - deg_d
        quo[shift] = 1
        for i in range(len(den)):
            num[shift+i] ^= den[i]
        while len(num)>0 and num[-1]==0: num.pop()
    return quo, num

def get_bch_poly(n, coset_reps):
    g = [GF256(1)]
    alpha = GF256(2)

    flat_reps = []
    for item in coset_reps:
        if isinstance(item, list):
            flat_reps.append(item[0])
        else:
            flat_reps.append(item)

    for r in flat_reps:
        root_vals = []
        x = r % n
        seen = set()
        while x not in seen:
            seen.add(x)
            root_vals.append(x)
            x = (2*x)%n

        m = [GF256(1)]
        for v in root_vals:
            root = GF256(1)
            for _ in range(v):
                root = root * alpha
            m_new = [GF256(0) for _ in range(len(m)+1)]
            for i, c in enumerate(m):
                m_new[i] = m_new[i] + (c * root)
                m_new[i+1] = m_new[i+1] + c
            m = m_new
        g = poly_mul(g, m)

    bits = []
    for c in g:
        if c.val == 0: bits.append(0)
        elif c.val == 1: bits.append(1)
        else: raise ValueError(f"Coeff {c.val} not binary in generator poly")
    return bits

def build_cyclic_H(n, h_bits):
    h_rev = h_bits[::-1]
    nrows = n - (len(h_bits)-1)
    H = []
    for i in range(nrows):
        row = [0]*n
        for j in range(len(h_rev)):
            row[(i+j)%n] = h_rev[j]
        H.append(row)
    return H

# --------------------- Utilities ---------------------

def dbm_to_watts(p_dbm: float) -> float:
    return 10 ** ((p_dbm - 30.0) / 10.0)

def effective_power_watts(p0_w: float, alpha_db_per_km: float, length_km: float) -> float:
    return p0_w * 10 ** (-(alpha_db_per_km * length_km) / 10.0)

def mmpp_base_rates(length_km: float, power_dbm: float, atten_db_per_km: float,
                    delta_lambda_nm: float, kappa_r: float, kappa_f: float,
                    eta_d: float, tau_g: float) -> Tuple[float, float]:
    p0_w = dbm_to_watts(power_dbm)
    peff = effective_power_watts(p0_w, atten_db_per_km, length_km)
    mu_sprs = eta_d * tau_g * kappa_r * peff * length_km
    mu_fwm  = eta_d * tau_g * kappa_f * (peff ** 2) * delta_lambda_nm
    lam_z = mu_sprs + mu_fwm
    lam_x = lam_z
    return lam_x, lam_z

def per_gate_probs(lam_x: float, lam_z: float, eta_xz: float,
                   burst_mult: float, dark: float, after: float) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    lam_x_low, lam_z_low = max(0.0, eta_xz * lam_x), max(0.0, lam_z)
    lam_x_high, lam_z_high = burst_mult * lam_x_low, burst_mult * lam_z_low
    p_x_low,  p_x_high  = 1.0 - math.exp(-lam_x_low),  1.0 - math.exp(-lam_x_high)
    p_z_low,  p_z_high  = 1.0 - math.exp(-lam_z_low),  1.0 - math.exp(-lam_z_high)
    p_z_low  = min(1.0, p_z_low  + dark + after)
    p_z_high = min(1.0, p_z_high + dark + after)
    p_x_low  = min(1.0, p_x_low  + 0.5*dark + 0.5*after)
    p_x_high = min(1.0, p_x_high + 0.5*dark + 0.5*after)
    return (p_x_low, p_x_high), (p_z_low, p_z_high)

def sample_markov_states(n: int, rho: float, rng: random.Random) -> List[int]:
    s = 1 if rng.random() < 0.5 else 0
    states = [s]
    for _ in range(1, n):
        if rng.random() > rho:
            s ^= 1
        states.append(s)
    return states

def sample_errors(states: List[int], p_low: float, p_high: float, rng: random.Random) -> List[int]:
    errs = []
    for s in states:
        p = p_low if s == 0 else p_high
        errs.append(1 if rng.random() < p else 0)
    return errs

def runlen_histogram(states_all: List[int]) -> dict:
    hist = {}
    if not states_all:
        return hist
    run = 1
    for i in range(1, len(states_all)):
        if states_all[i] == states_all[i-1]:
            run += 1
        else:
            hist[run] = hist.get(run, 0) + 1
            run = 1
    hist[run] = hist.get(run, 0) + 1
    return hist

def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    if beta is None:
        if n == 0:
            return 0.0, 1.0
        z = 1.959963984540054
        phat = k / n
        denom = 1.0 + (z*z)/n
        centre = (phat + (z*z)/(2*n)) / denom
        half = (z * math.sqrt((phat*(1.0 - phat) + (z*z)/(4*n)) / n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)
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

def dkl_bernoulli(q: float, p: float) -> float:
    eps = 1e-16
    q = min(max(q, eps), 1.0 - eps)
    p = min(max(p, eps), 1.0 - eps)
    return q*math.log(q/p) + (1.0-q)*math.log((1.0-q)/(1.0-p))

def chernoff_union_bound(n: int, t: int, p_x_low: float, p_x_high: float, p_z_low: float, p_z_high: float):
    q = (t + 1) / n
    pbar_x = 0.5 * (p_x_low + p_x_high)
    pbar_z = 0.5 * (p_z_low + p_z_high)
    bx = math.exp(-n * dkl_bernoulli(q, pbar_x))
    bz = math.exp(-n * dkl_bernoulli(q, pbar_z))
    return bx + bz, q, pbar_x, pbar_z

def lezaud_markov_bound(n: int, t: int, p_low: float, p_high: float, rho: float):
    gamma = 2.0 * (1.0 - rho)
    ef = 0.5 * (p_low + p_high)
    q = (t + 1) / n
    eps = q - ef
    if eps <= 0:
        return "not_applicable_epsilon_le_0", gamma, ef, eps
    val = math.exp(-(n * eps * eps * gamma) / (2.0 * (1.0 + eps)))
    return f"{val:.6g}", gamma, ef, eps

def cyclotomic_coset_2mod255(a: int) -> List[int]:
    seen = set()
    x = a % 255
    res = []
    while x not in seen:
        seen.add(x)
        res.append(x)
        x = (2 * x) % 255
    return res

# ------------------ OSD-k Decoder ------------------

class OSD_Solver:
    def __init__(self, H):
        self.H = H
        self.m, self.n = H.shape

    def decode(self, llr, depth=2, max_weight_accept=10):
        # 1. Sort
        perm = np.argsort(np.abs(llr))[::-1]
        perm_rev = perm[::-1]
        H_rev = self.H[:, perm_rev]

        # 2. Gaussian Elim
        H_work = H_rev.copy()
        pivot_cols_in_rev = []
        row_idx = 0
        col_idx = 0
        while row_idx < self.m and col_idx < self.n:
            p = -1
            for r in range(row_idx, self.m):
                if H_work[r, col_idx]:
                    p = r
                    break
            if p != -1:
                if p != row_idx:
                    H_work[[p, row_idx]] = H_work[[row_idx, p]]
                for r in range(self.m):
                    if r != row_idx and H_work[r, col_idx]:
                        H_work[r] ^= H_work[row_idx]
                pivot_cols_in_rev.append(col_idx)
                row_idx += 1
            col_idx += 1

        non_pivot_cols = [c for c in range(self.n) if c not in pivot_cols_in_rev]

        # Hard decision on I
        x_hard = (llr < 0).astype(int)
        x_rev_hard = x_hard[perm_rev]

        # Search for candidates
        best_e = None
        best_cost = float('inf')

        # Helper to compute candidate from error pattern on I
        def solve_candidate(error_on_I_indices):
            x_I = x_rev_hard[non_pivot_cols].copy()
            # Flip errors
            for idx in error_on_I_indices:
                x_I[idx] ^= 1

            x_sol_rev = np.zeros(self.n, dtype=int)
            x_sol_rev[non_pivot_cols] = x_I

            # Backsolve
            for i in range(len(pivot_cols_in_rev)):
                p_col = pivot_cols_in_rev[i]
                row_val = 0
                for k in non_pivot_cols:
                    if H_work[i, k]:
                        row_val ^= x_sol_rev[k]
                x_sol_rev[p_col] = row_val

            # Unpermute
            c_hat = np.zeros(self.n, dtype=int)
            c_hat[perm_rev] = x_sol_rev

            # Error e = y - c_hat? No.
            # In Syndrome decoding context, c_hat is the correction e_corr.
            # But here `llr` is from `zs`.
            # `c_hat` matches `zs` on I (adjusted by error).
            # The 'cost' of `c_hat` is how unlikely it is.
            # If `c_hat` is the correction, we want `c_hat` to be the ML error.
            # So `c_hat` should match `x_hard` mostly.
            # Cost = sum of LLR magnitudes where `c_hat` disagrees with `x_hard`.

            disagree = (c_hat != x_hard)
            cost = np.sum(np.abs(llr[disagree]))
            return c_hat, cost

        # OSD-0
        cand, cost = solve_candidate([])
        if np.sum(cand) <= max_weight_accept: # Heuristic: low weight result is likely correct
            return cand
        best_e = cand
        best_cost = cost

        if depth >= 1:
            # OSD-1: Flip 1 bit in I
            for i in range(len(non_pivot_cols)):
                cand, cost = solve_candidate([i])
                if cost < best_cost:
                    best_cost = cost
                    best_e = cand

        if depth >= 2:
            # OSD-2: Flip 2 bits in I (limit search if needed)
            limit = min(len(non_pivot_cols), 64) # Scan first 64 reliable bits
            for i in range(limit):
                for j in range(i+1, limit):
                    cand, cost = solve_candidate([i, j])
                    if cost < best_cost:
                        best_cost = cost
                        best_e = cand

        return best_e

# ------------------ Main experiment ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--length_km", type=float, default=100.0)
    ap.add_argument("--power_dbm", type=float, default=10.0)
    ap.add_argument("--atten_db_per_km", type=float, default=0.25)
    ap.add_argument("--delta_lambda_nm", type=float, default=10.0)
    ap.add_argument("--kappa_r", type=float, default=0.11)
    ap.add_argument("--kappa_f", type=float, default=1e-4)
    ap.add_argument("--eta_xz", type=float, default=0.3)
    ap.add_argument("--rho", type=float, default=0.6)
    ap.add_argument("--burst_mult", type=float, default=2.0)
    ap.add_argument("--eta_d", type=float, default=1.0)
    ap.add_argument("--tau_g", type=float, default=1.0)
    ap.add_argument("--dark", type=float, default=0.0)
    ap.add_argument("--after", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--d", type=int, default=21)
    ap.add_argument("--bp", type=int, default=1) # Kept for CLI compat, but ignored
    ap.add_argument("--bp_iters", type=int, default=10)
    ap.add_argument("--bp_offset", type=float, default=1.0)
    ap.add_argument("--bp_damping", type=float, default=0.5)
    ap.add_argument("--bp_llr_bits", type=int, default=6)
    ap.add_argument("--shared_state", type=int, default=1)
    ap.add_argument("--csv_out", type=str, default="ag_qec_results.csv")
    ap.add_argument("--txt_out", type=str, default="results.txt")
    ap.add_argument("--bp_txt_out", type=str, default="")
    ap.add_argument("--runlen_csv", type=str, default="runlen_hist.csv")
    ap.add_argument("--sensitivity_csv", type=str, default="")
    ap.add_argument("--css_verify", type=int, default=1)
    ap.add_argument("--css_log", type=str, default="css_results.txt")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # Base rates
    lam_x, lam_z = mmpp_base_rates(args.length_km, args.power_dbm, args.atten_db_per_km,
                                   args.delta_lambda_nm, args.kappa_r, args.kappa_f,
                                   args.eta_d, args.tau_g)
    (p_x_low, p_x_high), (p_z_low, p_z_high) = per_gate_probs(lam_x, lam_z, args.eta_xz,
                                                              args.burst_mult, args.dark, args.after)

    # Prepare Exact CSS Codes
    n = 255
    reps_z = [1, 3, 5, 7, 9, 11, 13]
    reps_x = [1, 3, 5]

    HZ_exact = None
    HX_exact = None
    osd_solver_x = None # Decodes X (uses HZ)
    osd_solver_z = None # Decodes Z (uses HX)

    if np is not None:
        cosets_z = [cyclotomic_coset_2mod255(r) for r in reps_z]
        gz_bits = get_bch_poly(n, cosets_z)
        xn_1 = [0]*(n+1); xn_1[0]=1; xn_1[n]=1
        hz_bits, _ = poly_div_gf2_bits(xn_1, gz_bits)
        HZ_exact = np.array(build_cyclic_H(n, hz_bits), dtype=int)

        cosets_x = [cyclotomic_coset_2mod255(r) for r in reps_x]
        gx_bits = get_bch_poly(n, cosets_x)
        hx_bits, _ = poly_div_gf2_bits(xn_1, gx_bits)
        HX_exact = np.array(build_cyclic_H(n, hx_bits), dtype=int)

        osd_solver_x = OSD_Solver(HZ_exact) # Corrects X errors
        osd_solver_z = OSD_Solver(HX_exact) # Corrects Z errors

    # Sensitivity grid
    if args.sensitivity_csv:
        grid_r = [0.03, 0.05, 0.08, 0.11, 0.2]
        grid_f = [1e-4, 5e-4, 1e-3]
        with open(args.sensitivity_csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["kappa_r","kappa_f","p_z"])
            for kr in grid_r:
                for kf in grid_f:
                    lx, lz = mmpp_base_rates(args.length_km, args.power_dbm, args.atten_db_per_km,
                                             args.delta_lambda_nm, kr, kf, args.eta_d, args.tau_g)
                    pz = 1.0 - math.exp(-lz)
                    w.writerow([kr, kf, f"{pz:.10f}"])

    # Monte Carlo
    t = (args.d - 1) // 2
    fail_bdd = 0
    fail_osd = 0
    osd_attempted = (np is not None)

    # Precompute RREFs for checks
    HZ_basis = None
    HX_basis = None

    if osd_attempted:
        def get_rref(H):
            m, n = H.shape
            M = H.copy()
            pivot_row = 0
            for j in range(n):
                if pivot_row >= m: break
                k = -1
                for i in range(pivot_row, m):
                    if M[i, j]:
                        k = i
                        break
                if k != -1:
                    M[[pivot_row, k]] = M[[k, pivot_row]]
                    for i in range(m):
                        if i != pivot_row and M[i, j]:
                            M[i] ^= M[pivot_row]
                    pivot_row += 1
            return M[:pivot_row]

        HZ_basis = get_rref(HZ_exact)
        HX_basis = get_rref(HX_exact)

        def in_rowspace(v, basis):
            vc = v.copy()
            for i in range(len(basis)):
                p = np.argmax(basis[i])
                if vc[p]:
                    vc ^= basis[i]
            return not np.any(vc)

        def make_llr(errs, states, p_low, p_high):
            llrs = np.zeros(n)
            for i in range(n):
                s = states[i]
                p = p_low if s==0 else p_high
                p = min(max(p, 1e-12), 1-1e-12)
                mag = math.log((1.0-p)/p)
                llrs[i] = mag if errs[i]==0 else -mag
            return llrs

    # Loop
    rng = random.Random(args.seed)
    all_states_concat = []

    for _ in range(args.trials):
        if args.shared_state:
            states = sample_markov_states(n, args.rho, rng)
            xs = sample_errors(states, p_x_low, p_x_high, rng)
            zs = sample_errors(states, p_z_low, p_z_high, rng)
            all_states_concat.extend(states)
        else:
            states_x = sample_markov_states(n, args.rho, rng)
            states_z = sample_markov_states(n, args.rho, rng)
            xs = sample_errors(states_x, p_x_low, p_x_high, rng)
            zs = sample_errors(states_z, p_z_low, p_z_high, rng)
            all_states_concat.extend(states_x)

        # BDD
        wx, wz = sum(xs), sum(zs)
        if (wx > t) or (wz > t):
            fail_bdd += 1

        if osd_attempted:
            # Z correction (uses HX)
            # Decoder finds codeword c_z in Ker(HX) closest to zs.
            # Effective correction applied is (zs - c_z).
            # Residual error is zs - (zs - c_z) = c_z.
            # Check logical failure: residual c_z must be in RowSpace(HZ) (stabilizer).
            llr_z = make_llr(zs, states if args.shared_state else states_z, p_z_low, p_z_high)
            c_z = osd_solver_z.decode(llr_z, depth=2)
            success_z = in_rowspace(c_z, HZ_basis)

            # X correction (uses HZ)
            # Decoder finds codeword c_x in Ker(HZ) closest to xs.
            # Residual error c_x must be in RowSpace(HX).
            llr_x = make_llr(xs, states if args.shared_state else states_x, p_x_low, p_x_high)
            c_x = osd_solver_x.decode(llr_x, depth=2)
            success_x = in_rowspace(c_x, HX_basis)

            if not (success_z and success_x):
                fail_osd += 1

    # Aggregation
    pL_bdd = fail_bdd / args.trials
    lo, hi = clopper_pearson_ci(fail_bdd, args.trials, 0.05)
    Fe = 1.0 - pL_bdd
    Fe_lo = 1.0 - hi
    Fe_hi = 1.0 - lo
    ch_u, q, pbar_x, pbar_z = chernoff_union_bound(n, t, p_x_low, p_x_high, p_z_low, p_z_high)
    lezaud_status, gamma, ef, eps = lezaud_markov_bound(n, t, p_z_low, p_z_high, args.rho)
    r1_emp = 0.0
    if len(all_states_concat) > 1:
        s0 = all_states_concat[:-1]
        s1 = all_states_concat[1:]
        mean = sum(all_states_concat)/len(all_states_concat)
        num = sum((a-mean)*(b-mean) for a,b in zip(s0,s1))
        den = sum((a-mean)*(a-mean) for a in s0)
        r1_emp = num/den if den != 0 else 0.0

    # Write results
    with open(args.txt_out, "w") as f:
        f.write(f"p_x_low = {p_x_low}\n")
        f.write(f"p_x_high = {p_x_high}\n")
        f.write(f"p_z_low = {p_z_low}\n")
        f.write(f"p_z_high = {p_z_high}\n")
        f.write(f"n = {n}\n")
        f.write(f"d = {args.d}\n")
        f.write(f"t = {t}\n")
        f.write(f"trials = {args.trials}\n")
        f.write(f"seed = {args.seed}\n")
        f.write(f"failures = {fail_bdd}\n")
        f.write(f"p_L_bdd = {pL_bdd}\n")
        f.write(f"p_L_lo = {lo}\n")
        f.write(f"p_L_hi = {hi}\n")
        f.write(f"F_e = {Fe}\n")
        f.write(f"F_e_lo = {Fe_lo}\n")
        f.write(f"F_e_hi = {Fe_hi}\n")
        f.write(f"chernoff_union = {ch_u}\n")
        f.write(f"chernoff_q = {q}\n")
        f.write(f"pbar_x = {pbar_x}\n")
        f.write(f"pbar_z = {pbar_z}\n")
        f.write(f"lezaud_z = {lezaud_status}\n")
        f.write(f"gamma = {gamma}\n")
        f.write(f"eps = {eps}\n")
        f.write(f"r1_empirical = {r1_emp:.3f}\n")
        if osd_attempted:
            pL_osd = fail_osd / args.trials
            f.write(f"p_L_osd = {pL_osd}\n")
            if args.bp_txt_out:
                with open(args.bp_txt_out, "w") as fb:
                    fb.write(f"p_L_osd = {pL_osd}\n")
                    fb.write("decoder = Exact_OSD2\n")
        else:
            f.write("osd_enabled = 0\n")
            if np is None:
                f.write("osd_reason = no_numpy\n")

    if args.runlen_csv:
        hist = runlen_histogram(all_states_concat)
        with open(args.runlen_csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["run_length","count"])
            for k in sorted(hist.keys()):
                w.writerow([k, hist[k]])

    if args.css_verify and args.css_log:
        rk_hz = "N/A"
        rk_hx = "N/A"
        if HZ_exact is not None:
            rk_hz = len(HZ_basis)
            rk_hx = len(HX_basis)

        with open(args.css_log, "w") as f:
            f.write("css_n = 255\n")
            f.write("delta_z = 15\n")
            f.write("delta_x = 7\n")
            f.write(f"rank_H_Z_exact = {rk_hz}\n")
            f.write(f"rank_H_X_exact = {rk_hx}\n")
            f.write("method = exact_dense_bch_osd2\n")

if __name__ == "__main__":
    main()
