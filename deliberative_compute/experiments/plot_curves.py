#!/usr/bin/env python
import argparse
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    args = ap.parse_args()
    with open(args.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        by_policy = defaultdict(list)
        for row in reader:
            try:
                tokens = float(row["avg_gen_tokens"])
                acc = float(row["accuracy"])
            except (KeyError, TypeError, ValueError):
                continue
            by_policy[row.get("policy", "")].append((tokens, acc))

    # Plot accuracy vs. avg_gen_tokens
    for pol, pts in by_policy.items():
        pts.sort(key=lambda x: x[0])
        xs = [x for x, _ in pts]
        ys = [y for _, y in pts]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Average generated tokens")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs. Tokens — {pol}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(args.out_png.replace(".png", f"_{pol}.png"), dpi=200)
        plt.close()
