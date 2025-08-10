"""
Plot latency and failure statistics from CSV logs.
"""
from __future__ import annotations
import csv, os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def load_csv(path):
    xs, lats, comps, fails = [], [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["episode"]))
            lats.append(float(row["latency_ms"]))
            comps.append(float(row["compute"]))
            fails.append(int(row["failures"]))
    return np.array(xs), np.array(lats), np.array(comps), np.array(fails)

def main(logdir: str):
    p = Path(logdir)
    base_csv = p / "baseline" / "baseline_log.csv"
    if not base_csv.exists():
        print(f"Baseline CSV not found at {base_csv}")
        return
    xs, lats, comps, fails = load_csv(base_csv)

    # Plot 1: latency per episode
    plt.figure()
    plt.plot(xs, lats, label="Baseline latency")
    plt.xlabel("Episode")
    plt.ylabel("Latency (ms)")
    plt.title("Latency per Episode (Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p / "latency_baseline.png")

    # Plot 2: failures per episode
    plt.figure()
    plt.plot(xs, fails, label="Baseline failures")
    plt.xlabel("Episode")
    plt.ylabel("Failures (count)")
    plt.title("Failures per Episode (Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p / "failures_baseline.png")

    print(f"Saved plots to {p}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", type=str, default="./data/logs")
    args = ap.parse_args()
    main(args.logdir)
