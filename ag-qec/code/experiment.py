from __future__ import annotations
import argparse, csv, os, time, json
import numpy as np
from typing import List
from markov_pauli import pauli_markov_block, effective_rates
from decoder_bp_osd import css_decode
from stats import wilson_interval, env_info, dump_json

def llr_from_p(p: float):
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    return np.log((1 - p) / p)

def run_once(Hx, Hz, pX, pZ, pY, eta, iters, damping, osd_order, rng_seed):
    n = Hx.shape[1]
    eX, eZ = pauli_markov_block(n=n, pX=pX, pZ=pZ, pY=pY, eta=eta, seed=rng_seed)
    pX_eff, pZ_eff = effective_rates(pX, pZ, pY)
    ex_llr = np.full(n, llr_from_p(pX_eff), dtype=float) * (1 - 2*eX)  # sign flips encode received bits
    ez_llr = np.full(n, llr_from_p(pZ_eff), dtype=float) * (1 - 2*eZ)
    ex_hat, ez_hat, metrics = css_decode(Hx, Hz, ex_llr, ez_llr, iters=iters, damping=damping, osd_order=osd_order)
    # Block failure if any residual syndrome remains after correction
    fail_x = ((Hx @ ex_hat) % 2).any()
    fail_z = ((Hz @ ez_hat) % 2).any()
    fail = int(fail_x or fail_z)
    return fail, metrics

def parse_eta_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]

def write_header_if_new(path: str):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "seed","n","rX","rZ","k","wX_mean","wZ_mean","cycles4",
                "decoder","iters","damping","osd_order","osd_calls",
                "pX","pZ","pY","eta","rhoZY",
                "trials","failures","p_hat","wilson_low","wilson_high",
                "bp_time_ms","osd_time_ms","total_time_ms","notes","env_json"
            ])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--css", type=str, help="Path to npz with Hx,Hz")
    ap.add_argument("--results", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--eta-list", type=str, default="0,0.3,0.6")
    ap.add_argument("--pX", type=float, default=0.02)
    ap.add_argument("--pZ", type=float, default=0.04)
    ap.add_argument("--pY", type=float, default=0.0)
    ap.add_argument("--rhoZY", type=float, default=0.0)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--damping", type=float, default=0.5)
    ap.add_argument("--osd", type=int, default=2)
    ap.add_argument("--baseline-lp", action="store_true", help="Run small lifted-product toy baseline instead")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.results), exist_ok=True)
    write_header_if_new(args.results)
    env = env_info()
    env_path = os.path.join(os.path.dirname(args.results), "env.json")
    dump_json(env_path, env)

    if args.baseline_lp:
        # Tiny toy baseline (placeholder): simulate a smaller random code to demonstrate the pipeline.
        n = 128
        rX = rZ = 40
        rng = np.random.default_rng(args.seed)
        Hx = (rng.random((rX, n)) < 0.03).astype(np.uint8)
        Hz = (rng.random((rZ, n)) < 0.03).astype(np.uint8)
        note = "baseline_lp_toy"
        wX_mean = float(Hx.sum(axis=0).mean()); wZ_mean = float(Hz.sum(axis=0).mean())
        cycles4 = 0
        k = n - min(rX, n) - min(rZ, n)
    else:
        D = np.load(args.css)
        Hx = D["Hx"].astype(np.uint8)
        Hz = D["Hz"].astype(np.uint8)
        # derive stats from sidecar JSON if available
        stats_path = os.path.join(os.path.dirname(args.results), f"stats_seed{args.seed}.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f: S = json.load(f)
            wX_mean = float(S.get("wX_mean", 0.0)); wZ_mean = float(S.get("wZ_mean", 0.0))
            k = int(S.get("k", Hx.shape[1] - Hx.shape[0] - Hz.shape[0]))
            cycles4 = int(S.get("cycles4_X", 0) + S.get("cycles4_Z", 0))
            rX = int(S.get("rX", Hx.shape[0])); rZ = int(S.get("rZ", Hz.shape[0])); n = int(S.get("n", Hx.shape[1]))
        else:
            n = Hx.shape[1]; rX = Hx.shape[0]; rZ = Hz.shape[0]
            k = n - rX - rZ; wX_mean = float(Hx.sum(axis=0).mean()); wZ_mean = float(Hz.sum(axis=0).mean())
            cycles4 = 0
        note = ""

    rng = np.random.default_rng(args.seed)
    with open(args.results, "a", newline="") as f:
        w = csv.writer(f)
        for eta in parse_eta_list(args.eta_list):
            failures = 0
            bp_ms_acc = 0.0
            osd_ms_acc = 0.0
            osd_calls_acc = 0
            t0 = time.time()
            for t in range(args.trials):
                fail, metr = run_once(Hx, Hz, args.pX, args.pZ, args.pY, eta, args.iters, args.damping, args.osd, rng.integers(1<<31))
                failures += fail
                bp_ms_acc += metr["bp_time_ms"]
                osd_ms_acc += metr["osd_time_ms"]
                osd_calls_acc += metr["osd_calls"]
            p_hat = failures / max(1, args.trials)
            lo, hi = wilson_interval(failures, max(1, args.trials))
            total_ms = (time.time() - t0) * 1000.0
            w.writerow([
                args.seed, n, rX, rZ, k, wX_mean, wZ_mean, cycles4,
                "bp_minsum+osd" if args.osd>0 else "bp_minsum",
                args.iters, args.damping, args.osd, osd_calls_acc,
                args.pX, args.pZ, args.pY, eta, args.rhoZY,
                args.trials, failures, f"{p_hat:.6g}", f"{lo:.6g}", f"{hi:.6g}",
                f"{bp_ms_acc/args.trials:.3f}", f"{osd_ms_acc/args.trials:.3f}", f"{total_ms/args.trials:.3f}",
                note, os.path.basename(env_path)
            ])
            f.flush()
            print(f"[sweep] eta={eta:.2f} trials={args.trials} p_hat={p_hat:.3e} CI=({lo:.3e},{hi:.3e}) "
                  f"bp={bp_ms_acc/args.trials:.2f}ms osd={osd_ms_acc/args.trials:.2f}ms calls={osd_calls_acc}")

if __name__ == "__main__":
    main()

