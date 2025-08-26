#!/usr/bin/env python3
from __future__ import annotations
import argparse, pandas as pd, numpy as np, os
import matplotlib.pyplot as plt

def load(results_path: str) -> pd.DataFrame:
    return pd.read_csv(results_path)

def plot_ablation(df: pd.DataFrame, outdir: str):
    # single chart: p_hat vs eta for BP-only and BP+OSD (if both present)
    key = ["decoder","eta"]
    g = df.groupby(key)["p_hat"].mean().reset_index()
    plt.figure()
    for name, sub in g.groupby("decoder"):
        sub = sub.sort_values("eta")
        plt.plot(sub["eta"].values, sub["p_hat"].values, marker="o", label=name)
    plt.yscale("log")
    plt.xlabel("Markov persistence η")
    plt.ylabel("Block error rate (mean over grid)")
    plt.title("Ablation: BP vs BP+OSD-2")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "ablation.pdf"))
    plt.close()

def plot_complexity(df: pd.DataFrame, outdir: str):
    # bar chart of per-block time by decoder type
    agg = df.groupby("decoder")[["bp_time_ms","osd_time_ms","total_time_ms"]].mean()
    plt.figure()
    idx = np.arange(len(agg))
    width = 0.35
    plt.bar(idx, agg["bp_time_ms"].values, width, label="BP avg ms")
    plt.bar(idx, agg["osd_time_ms"].values, width, bottom=agg["bp_time_ms"].values, label="OSD avg ms")
    plt.xticks(idx, agg.index.tolist(), rotation=15)
    plt.ylabel("Per-block time (ms)")
    plt.title("Decoder complexity (average)")
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "complexity.pdf"))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    df = load(args.results)
    # coerce numerics (CSV written with strings for p_hat/CI)
    for c in ["p_hat","wilson_low","wilson_high","bp_time_ms","osd_time_ms","total_time_ms","eta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    plot_ablation(df, args.outdir)
    plot_complexity(df, args.outdir)

if __name__ == "__main__":
    main()

