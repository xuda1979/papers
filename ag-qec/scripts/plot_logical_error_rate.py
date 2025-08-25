#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=str, default="data/logical_error_rate.csv")
    ap.add_argument("--out", type=str, default="figs/logical_error_rate.png")
    args = ap.parse_args()
    X = np.genfromtxt(args.inp, delimiter=",", names=True, dtype=float, encoding=None)
    p = X["block_logical_rate"]; lo = X["ci_lo"]; hi = X["ci_hi"]
    fig = plt.figure(figsize=(6,4))
    x = np.arange(1, len(np.atleast_1d(p))+1)
    plt.errorbar(x, p, yerr=[p-lo, hi-p], fmt="o", capsize=3)
    plt.yscale("log")
    plt.xlabel("Configuration index")
    plt.ylabel("Block logical error rate")
    plt.title("Logical error rate (95% CI)")
    import os; os.makedirs("figs", exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=180)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
