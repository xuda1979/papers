"""Generate a plot showing the L1-contraction (adversarial stability) result.

Reads results/stability_{U,B,FULL}.json and produces a figure with three
panels showing the L1 ratio vs epsilon for each variant, for random and
FGSM perturbations.
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base, "results")
    fig_dir = os.path.join(base, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    variants = [
        ("DTFN-U", "stability_U.json", "#1f77b4"),
        ("DTFN-B", "stability_B.json", "#ff7f0e"),
        ("DTFN-FULL", "stability_FULL.json", "#2ca02c"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for attack, ax in [("random", axes[0]), ("fgsm", axes[1])]:
        for name, fname, color in variants:
            path = os.path.join(results_dir, fname)
            if not os.path.exists(path):
                continue
            d = json.load(open(path))
            epsilons = sorted(float(k.split("_")[1]) for k in d[attack])
            ratios = [d[attack][f"eps_{eps}"]["ratio_l1"] for eps in epsilons]
            ax.plot(epsilons, ratios, "o-", color=color, label=name, linewidth=2, markersize=7)
        ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Theorem 5 bound")
        ax.set_xlabel(r"Perturbation magnitude $\varepsilon$")
        ax.set_xlim(0, 0.55)
        ax.set_ylim(0.7, 1.4)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{'Random' if attack == 'random' else 'FGSM'} perturbation")

    axes[0].set_ylabel(r"$\ell_1$ contraction ratio $\|\delta_{\mathrm{out}}\|_1 / \|\delta_{\mathrm{in}}\|_1$")
    axes[0].legend(loc="lower right", framealpha=0.9)
    fig.suptitle("Adversarial stability: DTFN forward map is 1-Lipschitz (Theorem 5)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(fig_dir, "fig_stability.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    main()
