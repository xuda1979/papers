#!/usr/bin/env python
import argparse

import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_png", type=str, required=True)
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    # Plot accuracy vs. avg_gen_tokens
    for pol in df.policy.unique():
        sub = df[df.policy == pol]
        plt.figure()
        plt.plot(sub["avg_gen_tokens"], sub["accuracy"], marker="o")
        plt.xlabel("Average generated tokens")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs. Tokens — {pol}")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(args.out_png.replace(".png", f"_{pol}.png"), dpi=200)
        plt.close()
