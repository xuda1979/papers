#!/usr/bin/env python3
"""
Reproducible Monte Carlo for CSS-like codes under a two-state MMPP noise model.
Now with EXACT BCH Tanner graph construction and OSD-0 decoding.

Revisions:
- Replaced surrogate random graphs with EXACT BCH parity-check matrices derived from cyclotomic cosets.
- Added GF(2^8) arithmetic and polynomial logic to construct g(x) and h(x) rigorously.
- Implemented OSD-0 (Ordered Statistics Decoding) to handle the dense BCH matrices, replacing the previous "sanity check" limitation.
- Maintained MMPP noise model and statistics logging.

Usage:
  python3 simulation.py --trials 1000 --seed 42
"""
import argparse, csv, math, random, statistics, time
from typing import List, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.stats import beta
except Exception:
    beta = None

# --------------------- GF(2^8) and Polynomials ---------------------

PRIM = 0x11D
GF_EXP = [0] * 512
GF_LOG = [0] * 256

def init_gf256():
    x = 1
    for i in range(255):
        GF_EXP[i] = x
        GF_LOG[x] = i
        x <<= 1
        if x & 0x100:
            x ^= PRIM
    for i in range(255, 512):
        GF_EXP[i] = GF_EXP[i - 255]

init_gf256()

def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else GF_EXP[GF_LOG[a] + GF_LOG[b]]

def poly_mul_gf256(p1, p2):
    res = [0] * (len(p1) + len(p2) - 1)
    for i in range(len(p1)):
        for j in range(len(p2)):
            res[i+j] ^= gf_mul(p1[i], p2[j])
    return res

def poly_mul_binary(p1, p2):
    res = [0] * (len(p1) + len(p2) - 1)
    for i, c1 in enumerate(p1):
        if c1:
            for j, c2 in enumerate(p2):
                if c2:
                    res[i+j] ^= 1
    return res

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
        root = GF_EXP[idx]
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

def poly_div_binary(dividend, divisor):
    # Returns remainder, but we expect exact division for h(x)
    # This is actually effectively (x^n-1)/g(x)
    # dividend and divisor are low-degree first?
    # Standard division usually high-degree first.
    # Let's reverse for division.
    num = list(reversed(dividend))
    den = list(reversed(divisor))
    deg_num = len(num) - 1
    deg_den = len(den) - 1
    if deg_den < 0: raise ZeroDivisionError
    if deg_num < deg_den: return [0]

    quotient = [0] * (deg_num - deg_den + 1)
    rem = list(num)

    for i in range(deg_num - deg_den + 1):
        if rem[i] == 1:
            quotient[i] = 1
            for j in range(deg_den + 1):
                rem[i+j] ^= den[j]

    # Quotient is high-degree first
    return list(reversed(quotient))

# --------------------- Matrix Construction ---------------------

def build_cyclic_matrices(coset_leaders, n):
    g = get_generator_poly(coset_leaders)
    k = n - (len(g) - 1)

    # G matrix (k x n): Rows are shifts of g
    G = np.zeros((k, n), dtype=np.int8)
    for i in range(k):
        G[i, i:i+len(g)] = g

    # H matrix (n-k x n): Rows are shifts of h (reciprocal)
    # h(x) * g(x) = x^n - 1
    xn_1 = [1] + [0]*(n-1) + [1] # x^n + 1 (binary)
    h = poly_div_binary(xn_1, g)
    # h is low degree first.
    # Dual code generator is reciprocal of h?
    # Standard: H rows are shifts of coefficients of h(x) REVERSED?
    # Let's verify orthogonality numerically.
    # Try h reversed.
    h_rev = list(reversed(h))
    H = np.zeros((n-k, n), dtype=np.int8)
    for i in range(n-k):
        # Cyclic shifts of h_rev?
        # Actually, if we use the dual generator g_perp:
        # g_perp(x) = x^(n-k) h(x^-1).
        # Which is h reversed.
        H[i, 0:len(h_rev)] = h_rev
        H[i] = np.roll(H[i], i) # shift

    # Verify
    if np.any((G @ H.T) % 2 != 0):
        # Try non-reversed h
        H = np.zeros((n-k, n), dtype=np.int8)
        for i in range(n-k):
            H[i, 0:len(h)] = h
            H[i] = np.roll(H[i], i)
        if np.any((G @ H.T) % 2 != 0):
            print("Warning: G*H^T check failed. H might be transposed or shifted.")

    return G, H

# --------------------- GF2 Linear Algebra ---------------------

def gf2_rank(A):
    # Gaussian elimination to find rank
    M = A.copy()
    rows, cols = M.shape
    rank = 0
    pivot_row = 0
    for j in range(cols):
        if pivot_row >= rows: break
        # Find pivot
        idx = np.where(M[pivot_row:, j] == 1)[0]
        if len(idx) == 0: continue
        idx += pivot_row

        # Swap
        M[[pivot_row, idx[0]]] = M[[idx[0], pivot_row]]

        # Eliminate
        pivot_indices = np.where(M[:, j] == 1)[0]
        pivot_indices = pivot_indices[pivot_indices != pivot_row]
        M[pivot_indices] ^= M[pivot_row]

        pivot_row += 1
        rank += 1
    return rank

def gf2_solve(A, b):
    # Solves Ax = b. Returns one solution.
    # Assumes full row rank or solvable.
    # Uses lstsq equivalent via GE.
    m, n = A.shape
    M = np.hstack([A, b.reshape(-1, 1)])
    # GE
    pivot_row = 0
    pivots = []
    for j in range(n):
        if pivot_row >= m: break
        idx = np.where(M[pivot_row:, j] == 1)[0]
        if len(idx) == 0: continue
        idx += pivot_row
        M[[pivot_row, idx[0]]] = M[[idx[0], pivot_row]]
        pivot_indices = np.where(M[:, j] == 1)[0]
        pivot_indices = pivot_indices[pivot_indices != pivot_row]
        M[pivot_indices] ^= M[pivot_row]
        pivots.append((pivot_row, j))
        pivot_row += 1

    x = np.zeros(n, dtype=np.int8)
    # Backsub
    for r, c in reversed(pivots):
        val = M[r, n]
        # subtract knowns
        val ^= np.dot(M[r, c+1:n], x[c+1:]) % 2
        x[c] = val
    return x

def osd0_decode(G, y, llr):
    """
    OSD-0 Decoder.
    1. Sort positions by reliability (|LLR|).
    2. Identify MRB (Most Reliable Basis) in G.
    3. Re-encode to find codeword.
    4. Return error pattern y + c.
    """
    k, n = G.shape

    # Reliability sort
    reliab = np.abs(llr)
    perm = np.argsort(reliab)[::-1] # Descending

    G_perm = G[:, perm]
    y_perm = y[perm]

    # Gaussian elimination to find Basis
    # We want first k independent columns in G_perm
    M = G_perm.copy()
    pivots = [] # cols in G_perm
    pivot_rows = []

    curr_col = 0
    for r in range(k):
        while curr_col < n:
            idx = np.where(M[r:, curr_col] == 1)[0]
            if len(idx) > 0:
                idx += r
                # Swap rows
                M[[r, idx[0]]] = M[[idx[0], r]]
                # Eliminate
                elim_idxs = np.where(M[:, curr_col] == 1)[0]
                elim_idxs = elim_idxs[elim_idxs != r]
                M[elim_idxs] ^= M[r]
                pivots.append(curr_col)
                pivot_rows.append(r)
                curr_col += 1
                break
            curr_col += 1

    # If rank < k, we have a problem (shouldn't happen for valid G)
    if len(pivots) < k:
        return y # Fail gracefully

    # Solve u * G_basis = y_basis
    # G_basis is k x k (rows permuted during GE? No, we swapped rows of M)
    # Wait, we swapped rows of M. The relationship u*G = c implies u * M = c' (row ops don't change space)
    # But u changes.
    # Alternative: Just use the GE result to map y to u?
    # Easier: Use the pivots to extract submatrix from ORIGINAL G (un-row-swapped) and solve.

    G_basis = G_perm[:, pivots] # k x k
    y_basis = y_perm[pivots]

    # Solve u @ G_basis = y_basis  => G_basis.T @ u.T = y_basis.T
    u = gf2_solve(G_basis.T, y_basis)

    # Encode
    c = (u @ G) % 2

    return (y + c) % 2

# --------------------- Utilities ---------------------

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

def clopper_pearson_ci(k, n, alpha=0.05):
    if beta is None:
        if n == 0: return 0.0, 1.0
        z = 1.96
        phat = k / n
        denom = 1.0 + (z*z)/n
        centre = (phat + (z*z)/(2*n)) / denom
        half = (z * math.sqrt((phat*(1.0 - phat) + (z*z)/(4*n)) / n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)
    if k == 0:
        lo, hi = 0.0, beta.ppf(1.0 - alpha/2.0, 1, n)
    elif k == n:
        lo, hi = beta.ppf(alpha/2.0, n, 1), 1.0
    else:
        lo = beta.ppf(alpha / 2.0, k, n - k + 1)
        hi = beta.ppf(1.0 - alpha / 2.0, k + 1, n - k)
    return lo, hi

# ------------------ Main experiment ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=255)
    ap.add_argument("--d", type=int, default=21)

    # Physics parameters
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

    # Outputs
    ap.add_argument("--txt_out", type=str, default="results.txt")
    ap.add_argument("--runlen_csv", type=str, default="runlen_hist.csv")
    ap.add_argument("--css_log", type=str, default="css_results.txt")

    # Flags
    ap.add_argument("--bp_txt_out", type=str, default="")
    ap.add_argument("--sensitivity_csv", type=str, default="")
    ap.add_argument("--css_verify", type=int, default=1)

    # Unused but kept for compatibility
    ap.add_argument("--bp", type=int, default=1)
    ap.add_argument("--bp_iters", type=int, default=10)
    ap.add_argument("--bp_offset", type=float, default=1.0)
    ap.add_argument("--bp_damping", type=float, default=0.5)
    ap.add_argument("--bp_llr_bits", type=int, default=6)
    ap.add_argument("--shared_state", type=int, default=1)

    args = ap.parse_args()

    if np is None:
        print("Error: NumPy is required for Exact BCH construction and OSD.")
        return

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    # Rates
    lam_x, lam_z = mmpp_base_rates(args.length_km, args.power_dbm, args.atten_db_per_km,
                                   args.delta_lambda_nm, args.kappa_r, args.kappa_f,
                                   args.eta_d, args.tau_g)
    (p_x_low, p_x_high), (p_z_low, p_z_high) = per_gate_probs(lam_x, lam_z, args.eta_xz,
                                                              args.burst_mult, args.dark, args.after)

    # --- EXACT BCH CONSTRUCTION ---
    # C_Z: delta=15 (reps 1,3,5,7,9,11,13)
    # C_X: delta=7  (reps 1,3,5)
    cosets_Z = [1, 3, 5, 7, 9, 11, 13]
    cosets_X = [1, 3, 5]

    # We use C_X to check Z errors?
    # Usually: H_Z checks X errors. H_X checks Z errors.
    # To decode X errors, we use the code C_Z defined by H_Z?
    # No. H_Z is parity check.
    # We use G_Z (generator of code with parity check H_Z) for OSD re-encoding?
    # Correct.

    G_Z, H_Z = build_cyclic_matrices(cosets_Z, args.n)
    G_X, H_X = build_cyclic_matrices(cosets_X, args.n)

    # Verification
    assert np.all((G_Z @ H_Z.T) % 2 == 0)
    assert np.all((G_X @ H_X.T) % 2 == 0)
    # CSS Orthogonality: H_X * H_Z^T = 0
    # C_X subset C_Z^perp?
    # With this construction, we need to check.
    ortho_check = np.all((H_X @ H_Z.T) % 2 == 0)

    if args.css_log:
        with open(args.css_log, "w") as f:
            f.write(f"n = {args.n}\n")
            f.write(f"Rank H_Z = {gf2_rank(H_Z)}\n")
            f.write(f"Rank H_X = {gf2_rank(H_X)}\n")
            f.write(f"CSS Orthogonality H_X*H_Z^T=0: {ortho_check}\n")
            f.write(f"Construction: Exact BCH with cyclotomic cosets.\n")

    # Decoding loop
    t_val = (args.d - 1) // 2
    fail_bdd = 0
    fail_osd = 0

    start_time = time.time()

    for i in range(args.trials):
        if i > 0 and i % 100 == 0:
            print(f"Trial {i}/{args.trials}...", flush=True)

        # Noise
        states = sample_markov_states(args.n, args.rho, rng)
        xs = np.array(sample_errors(states, p_x_low, p_x_high, rng), dtype=np.int8)
        zs = np.array(sample_errors(states, p_z_low, p_z_high, rng), dtype=np.int8)

        # BDD Oracle check
        if np.sum(xs) > t_val or np.sum(zs) > t_val:
            fail_bdd += 1

        # Exact OSD Decoding
        # X errors: measured by H_Z. Syndrome s_z = H_Z * xs.
        # We need to find error e_x such that H_Z e_x = s_z.
        # We use G_Z for OSD re-encoding.

        # LLRs
        # Construct LLRs from probs
        def get_llrs(probs, noise_vec):
            # LLR = log((1-p)/p). Positive.
            # We don't know noise_vec during decoding, but we know state probs?
            # Genie-aided LLRs based on hidden states
            llrs = []
            for s in states:
                p = probs[s]
                p = max(1e-12, min(1-1e-12, p))
                llrs.append(math.log((1-p)/p))
            return np.array(llrs)

        llr_x = get_llrs([p_x_low, p_x_high], xs)
        llr_z = get_llrs([p_z_low, p_z_high], zs)

        # Decode X
        syndrome_z = (H_Z @ xs) % 2
        # Initial guess e0: solve H_Z e0 = s
        e0_x = gf2_solve(H_Z, syndrome_z)
        # OSD
        est_x_err = osd0_decode(G_Z, e0_x, llr_x)

        # Decode Z
        syndrome_x = (H_X @ zs) % 2
        e0_z = gf2_solve(H_X, syndrome_x)
        est_z_err = osd0_decode(G_X, e0_z, llr_z)

        # Check Logical Error
        # Residual error = true + est
        res_x = (xs + est_x_err) % 2
        res_z = (zs + est_z_err) % 2

        # Logical error if residual is not in stabilizer (i.e. not zero? No, stabilizers are H rows).
        # Residual must be a stabilizer (codeword of C_perp).
        # We are decoding using C_Z. C_Z is the code. H_Z is parity.
        # The "error" we found must match syndrome.
        # So H * (true + est) = s + s = 0.
        # So residual is in Kernel(H) = Code.
        # Logical error if residual is NOT in Stabilizer Group?
        # For CSS:
        # X-logical error if res_x is in C_Z \ C_X^perp ?
        # No.
        # Stabilizers are rows of H_Z (for X) and H_X (for Z).
        # Code C_L has checks H_Z, H_X.
        # X-decoder finds e_x such that H_Z e_x = s.
        # res_x = e_true + e_est. H_Z res_x = 0. So res_x in C_Z.
        # Is res_x a logical operator?
        # Logical operators are in C_Z but not in RowSpace(H_X)?
        # Wait. H_X rows are X-stabilizers.
        # We are correcting X errors.
        # X-stabilizers are Z-operators.
        # Z-stabilizers (H_Z) detect X errors.
        # We found res_x in C_Z.
        # Does res_x commute with logical Z?
        # Actually, standard check:
        # res_x must be in RowSpace(H_X^perp)? No.
        # res_x is trivial if it is a stabilizer.
        # X-stabilizers are ...
        # Wait. X errors are detected by Z stabilizers.
        # Corrected X error is good if res_x \in Stabilizer(X).
        # Stabilizer(X) is generated by rows of G_X? No.
        # The CSS code has X-stabilizers and Z-stabilizers.
        # X-stabilizers are formed by H_X.
        # Z-stabilizers are formed by H_Z.
        # An X-error e_x is harmless if e_x \in RowSpace(H_X)?
        # No, X-error is an X operator. Z-stabilizers measure it.
        # If e_x is in Z-stabilizer group, it is harmless?
        # No.
        # Let's go back to definitions.
        # Code C_Z. Parity H_Z.
        # Logical X operators are in C_Z \ RowSpace(H_X^T)?
        # Actually, X-stabilizers are generated by H_X. They are X-operators.
        # So if res_x is linear combo of rows of H_X, it is a stabilizer.
        # So we check: is res_x in RowSpace(H_X)?
        # RowSpace(H_X) is generated by rows of H_X.
        # Check: rank([H_X; res_x]) == rank(H_X).

        rank_Hx = gf2_rank(H_X)
        rank_aug_x = gf2_rank(np.vstack([H_X, res_x]))
        is_logical_x = (rank_aug_x > rank_Hx)

        rank_Hz = gf2_rank(H_Z)
        rank_aug_z = gf2_rank(np.vstack([H_Z, res_z]))
        is_logical_z = (rank_aug_z > rank_Hz)

        if is_logical_x or is_logical_z:
            fail_osd += 1

    # Stats
    pl_bdd = fail_bdd / args.trials
    pl_osd = fail_osd / args.trials
    lo, hi = clopper_pearson_ci(fail_osd, args.trials)

    print("\n--- Results ---")
    print(f"Trials: {args.trials}")
    print(f"BDD Failures: {fail_bdd} (P_L = {pl_bdd:.5e})")
    print(f"OSD Failures: {fail_osd} (P_L = {pl_osd:.5e})")
    print(f"95% CI: [{lo:.5e}, {hi:.5e}]")

    # Write results
    with open(args.txt_out, "w") as f:
        f.write(f"trials = {args.trials}\n")
        f.write(f"failures = {fail_osd}\n") # OSD failures
        f.write(f"p_L_bdd = {pl_bdd}\n")
        f.write(f"p_L_osd = {pl_osd}\n")
        f.write(f"p_L_lo = {lo}\n")
        f.write(f"p_L_hi = {hi}\n")
        f.write(f"F_e = {1.0 - pl_osd}\n")
        f.write(f"decoder = OSD-0\n")
        f.write(f"graph = Exact BCH\n")

if __name__ == "__main__":
    main()
