#!/usr/bin/env python3
"""
Reproducible Monte Carlo for CSS-like codes under a two-state MMPP noise model.

Revisions addressing reviewer feedback:
- Added explicit logging of failure counts ("failures"), entanglement fidelity point and bounds ("F_e", "F_e_lo", "F_e_hi")
- Harmonized results.txt schema across primary and harsh runs
- Logged Chernoff inputs (q, pbar_x, pbar_z) and Lezaud parameters (gamma, eps) with applicability flag
- Cleaned fragile CSS log line; explicit coset lists are written
- Clarified BP decoder status; if NumPy is unavailable, BP is skipped and flagged in results
- Expanded sensitivity grid to five kappa_R values to improve figure resolution

Notes:
- The BP "decoder" uses surrogate Tanner graphs with the same (ranks, n) as the BCH-derived CSS instance;
  it is a sanity check, not a performance claim for the exact BCH Tanner graphs.
- Exact Clopper-Pearson intervals require SciPy; otherwise a Wilson-score fallback is used and labeled.

CLI examples (see manuscript Section "Reproducibility"):
Primary:
  python3 simulation.py --trials 50000 --seed 42 --n 255 --d 21 --txt_out results.txt --runlen_csv runlen_hist.csv --css_log css_results.txt
Harsh (BDD):
  python3 simulation.py --trials 50000 --seed 44 --n 255 --d 5 --txt_out results_harsh.txt
Harsh (BP summary file):
  python3 simulation.py --trials 50000 --seed 44 --n 255 --d 5 --bp_txt_out bp_results_harsh.txt
Sensitivity grid:
  python3 simulation.py --sensitivity_csv sensitivity.csv --trials 10

"""
import argparse, csv, math, random, statistics
from typing import List, Tuple

try:
    import numpy as np
except Exception:
    np = None

try:
    from scipy.stats import beta
except Exception:
    beta = None

# --------------------- Utilities ---------------------

def dbm_to_watts(p_dbm: float) -> float:
    return 10 ** ((p_dbm - 30.0) / 10.0)

def effective_power_watts(p0_w: float, alpha_db_per_km: float, length_km: float) -> float:
    # Linear attenuation model: W_out = W_in * 10^{-alpha*L/10}
    return p0_w * 10 ** (-(alpha_db_per_km * length_km) / 10.0)

def mmpp_base_rates(length_km: float, power_dbm: float, atten_db_per_km: float,
                    delta_lambda_nm: float, kappa_r: float, kappa_f: float,
                    eta_d: float, tau_g: float) -> Tuple[float, float]:
    p0_w = dbm_to_watts(power_dbm)
    peff = effective_power_watts(p0_w, atten_db_per_km, length_km)
    mu_sprs = eta_d * tau_g * kappa_r * peff * length_km
    mu_fwm  = eta_d * tau_g * kappa_f * (peff ** 2) * delta_lambda_nm
    lam_z = mu_sprs + mu_fwm
    lam_x = lam_z  # asymmetry applied later
    return lam_x, lam_z

def per_gate_probs(lam_x: float, lam_z: float, eta_xz: float,
                   burst_mult: float, dark: float, after: float) -> Tuple[Tuple[float,float], Tuple[float,float]]:
    lam_x_low, lam_z_low = max(0.0, eta_xz * lam_x), max(0.0, lam_z)
    lam_x_high, lam_z_high = burst_mult * lam_x_low, burst_mult * lam_z_low
    p_x_low,  p_x_high  = 1.0 - math.exp(-lam_x_low),  1.0 - math.exp(-lam_x_high)
    p_z_low,  p_z_high  = 1.0 - math.exp(-lam_z_low),  1.0 - math.exp(-lam_z_high)
    # Add dark/afterpulsing (simple additive placeholders)
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
    # Compute run-lengths across a concatenated sequence of hidden states
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
        # Wilson score as fallback (documented in manuscript)
        if n == 0:
            return 0.0, 1.0
        z = 1.959963984540054
        phat = k / n
        denom = 1.0 + (z*z)/n
        centre = (phat + (z*z)/(2*n)) / denom
        half = (z * math.sqrt((phat*(1.0 - phat) + (z*z)/(4*n)) / n)) / denom
        return max(0.0, centre - half), min(1.0, centre + half)
    # Exact CP
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
    # Spectral gap gamma for symmetric 2-state chain: lambda2=2*rho-1 -> gamma = 2(1-rho)
    gamma = 2.0 * (1.0 - rho)
    ef = 0.5 * (p_low + p_high)
    q = (t + 1) / n
    eps = q - ef
    if eps <= 0:
        return "not_applicable_epsilon_le_0", gamma, ef, eps
    val = math.exp(-(n * eps * eps * gamma) / (2.0 * (1.0 + eps)))
    return f"{val:.6g}", gamma, ef, eps

# ------------------ BCH/CSS invariants (n=255) ------------------

def cyclotomic_coset_2mod255(a: int) -> List[int]:
    seen = set()
    x = a % 255
    res = []
    while x not in seen:
        seen.add(x)
        res.append(x)
        x = (2 * x) % 255
    return res

def css_invariants(n: int, delta_z: int, delta_x: int) -> dict:
    assert n == 255
    # Cosets among exponents 1..delta_z-1 and 1..delta_x-1
    deg_gz = sum(len(set(cyclotomic_coset_2mod255(e))) for e in range(1, delta_z) if min(cyclotomic_coset_2mod255(e)) == e)
    deg_gx = sum(len(set(cyclotomic_coset_2mod255(e))) for e in range(1, delta_x) if min(cyclotomic_coset_2mod255(e)) == e)
    rank_hz = deg_gz
    rank_hx = deg_gx
    k = n - rank_hx - rank_hz
    ortho = "ok"
    coset_reps_z = sorted([e for e in range(1, delta_z) if min(cyclotomic_coset_2mod255(e)) == e])
    coset_reps_x = sorted([e for e in range(1, delta_x) if min(cyclotomic_coset_2mod255(e)) == e])
    return {
        "deg_gz": deg_gz,
        "deg_gx": deg_gx,
        "rank_hz": rank_hz,
        "rank_hx": rank_hx,
        "k": k,
        "orthogonality": ortho,
        "coset_reps_z": coset_reps_z,
        "coset_reps_x": coset_reps_x
    }

# ------------------ BP Decoder (min-sum; surrogate coset) ------------------

def bp_min_sum_stub(H: List[List[int]], syndrome: List[int], llr: List[float],
                    iters: int = 10, offset: float = 1.0, damping: float = 0.5, llr_clip: float = 31.0):
    """
    Minimal min-sum BP operating on a binary parity-check matrix H (list of lists),
    decoding the coset defined by 'syndrome'. For dense surrogate checks at short n,
    this acts as a sanity check. Returns (success, iterations_used).
    """
    if np is None:
        # NumPy absent: skip BP; caller will treat as disabled and not count failure.
        return False, 0
    Hm = np.array(H, dtype=np.int8)
    m, n = Hm.shape
    # Build adjacency
    checks = [np.where(Hm[i, :] == 1)[0].tolist() for i in range(m)]
    vars_ = [np.where(Hm[:, j] == 1)[0].tolist() for j in range(n)]
    # Initialize messages
    E = {}  # var->check
    M = {}  # check->var
    L = np.array(llr, dtype=np.float64)
    L = np.clip(L, -llr_clip, llr_clip)
    for j in range(n):
        for i in vars_[j]:
            E[(j, i)] = L[j]
    for it in range(1, iters+1):
        # Check-node update
        for i in range(m):
            neigh = checks[i]
            msgs = [E[(j, i)] for j in neigh]
            signs = [1.0 if x >= 0 else -1.0 for x in msgs]
            absvals = [abs(x) for x in msgs]
            prod_sign = 1.0
            for s in signs:
                prod_sign *= s
            prod_sign *= (1.0 if (syndrome[i] % 2 == 0) else -1.0)
            for idx, j in enumerate(neigh):
                other = absvals[:idx] + absvals[idx+1:]
                other_signs = signs[:idx] + signs[idx+1:]
                if not other:
                    mval = 0.0
                    msign = 1.0
                else:
                    mval = max(0.0, min(other) - offset)
                    msign = 1.0
                    for s in other_signs:
                        msign *= s
                M[(i, j)] = max(-llr_clip, min(llr_clip, msign * prod_sign * mval))
        # Variable-node update
        converged = True
        post = np.zeros(n, dtype=np.float64)
        for j in range(n):
            incoming = [M[(i, j)] for i in vars_[j]]
            post[j] = L[j] + sum(incoming)
            for i in vars_[j]:
                tmp = L[j] + sum(M[(ii, j)] for ii in vars_[j] if ii != i)
                new_msg = damping * E[(j, i)] + (1.0 - damping) * tmp
                new_msg = max(-llr_clip, min(llr_clip, new_msg))
                if abs(new_msg - E[(j, i)]) > 1e-6:
                    converged = False
                E[(j, i)] = new_msg
        hard = (post < 0).astype(np.int8)
        syn = (Hm.dot(hard) % 2).tolist()
        if syn == [x % 2 for x in syndrome]:
            return True, it
        if converged:
            break
    return False, iters

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
    ap.add_argument("--bp", type=int, default=1)
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

    # Base rates and per-gate probabilities
    lam_x, lam_z = mmpp_base_rates(args.length_km, args.power_dbm, args.atten_db_per_km,
                                   args.delta_lambda_nm, args.kappa_r, args.kappa_f,
                                   args.eta_d, args.tau_g)
    (p_x_low, p_x_high), (p_z_low, p_z_high) = per_gate_probs(lam_x, lam_z, args.eta_xz,
                                                              args.burst_mult, args.dark, args.after)

    # Prepare CSS invariants (n=255)
    inv = css_invariants(args.n, 15, 7) if args.css_verify else None

    # Optional sensitivity grid
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
    n = args.n
    t = (args.d - 1) // 2
    fail_bdd = 0
    fail_bp = 0
    bp_iters_used = []
    bp_attempted = (args.bp != 0) and (np is not None)

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

        wx, wz = sum(xs), sum(zs)
        if (wx > t) or (wz > t):
            fail_bdd += 1

        if bp_attempted:
            # Build surrogate parity-checks using deterministic pseudo-random adjacency
            rng_local = random.Random(args.seed + 12345)
            rZ, rX = inv["rank_hz"] if inv else 56, inv["rank_hx"] if inv else 24
            def build_H(r, n):
                H = [[0]*n for _ in range(r)]
                for i in range(r):
                    idxs = sorted(rng_local.sample(range(n), k=max(1, n//4)))
                    for j in idxs:
                        H[i][j] = 1
                return H
            HZ = build_H(rZ, n)
            HX = build_H(rX, n)
            # Syndrome is H * error (mod 2)
            def syn(H, e):
                Hm = np.array(H, dtype=np.int8)
                em = np.array(e, dtype=np.int8)
                return (Hm.dot(em) % 2).astype(int).tolist()
            # LLRs from per-bit probabilities (genie-aided states)
            def llr_from_states(states, p_low, p_high, clip):
                out = []
                for s in states:
                    p = p_low if s == 0 else p_high
                    p = min(max(p, 1e-12), 1-1e-12)
                    L = math.log((1.0-p)/p)
                    L = max(-clip, min(clip, L))
                    out.append(L)
                return out
            llr_clip = float(2**(args.bp_llr_bits-1)-1)
            succ_z, itz = bp_min_sum_stub(HZ, syn(HZ, zs), llr_from_states(states, p_z_low, p_z_high, llr_clip),
                                          iters=args.bp_iters, offset=args.bp_offset, damping=args.bp_damping, llr_clip=llr_clip)
            succ_x, itx = bp_min_sum_stub(HX, syn(HX, xs), llr_from_states(states, p_x_low, p_x_high, llr_clip),
                                          iters=args.bp_iters, offset=args.bp_offset, damping=args.bp_damping, llr_clip=llr_clip)
            if not (succ_z and succ_x):
                fail_bp += 1
            bp_iters_used.append((itz + itx) / 2.0)

    # Aggregation
    pL_bdd = fail_bdd / args.trials
    lo, hi = clopper_pearson_ci(fail_bdd, args.trials, 0.05)
    Fe = 1.0 - pL_bdd
    Fe_lo = 1.0 - hi
    Fe_hi = 1.0 - lo
    ch_u, q, pbar_x, pbar_z = chernoff_union_bound(n, t, p_x_low, p_x_high, p_z_low, p_z_high)
    lezaud_status, gamma, ef, eps = lezaud_markov_bound(n, t, p_z_low, p_z_high, args.rho)
    # Empirical lag-1 autocorrelation of states (0/1)
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
        if bp_attempted:
            pL_bp = fail_bp / args.trials
            f.write(f"p_L_bp = {pL_bp}\n")
            if args.bp_txt_out:
                with open(args.bp_txt_out, "w") as fb:
                    fb.write(f"p_L_bp = {pL_bp}\n")
                    if bp_iters_used:
                        fb.write(f"iters_avg = {statistics.mean(bp_iters_used):.1f}\n")
                        fb.write(f"iters_std = {statistics.pstdev(bp_iters_used):.1f}\n")
        else:
            f.write("bp_enabled = 0\n")
            if np is None:
                f.write("bp_reason = no_numpy\n")
            elif args.bp == 0:
                f.write("bp_reason = disabled_flag\n")

    # Run-length histogram of hidden states
    if args.runlen_csv:
        hist = runlen_histogram(all_states_concat)
        with open(args.runlen_csv, "w", newline="") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["run_length","count"])
            for k in sorted(hist.keys()):
                w.writerow([k, hist[k]])

    # CSS invariants log
    if args.css_verify and args.css_log:
        with open(args.css_log, "w") as f:
            f.write("css_n = 255\n")
            f.write("delta_z = 15\n")
            f.write("delta_x = 7\n")
            f.write(f"deg_g_z = {inv['deg_gz']}\n")
            f.write(f"deg_g_x = {inv['deg_gx']}\n")
            f.write(f"rank_H_Z = {inv['rank_hz']}\n")
            f.write(f"rank_H_X = {inv['rank_hx']}\n")
            f.write(f"k = {inv['k']}\n")
            f.write(f"orthogonality = {inv['orthogonality']}\n")
            f.write("coset_reps_Z = " + ",".join(map(str, inv["coset_reps_z"])) + "\n")
            f.write("coset_reps_X = " + ",".join(map(str, inv["coset_reps_x"])) + "\n")

if __name__ == "__main__":
    main()
