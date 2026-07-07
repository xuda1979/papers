"""Generate all paper figures from experiment results."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)

# Colour scheme for models
COLORS = {
    "LST": "#d62728",        # red — our method
    "Transformer": "#1f77b4", # blue
    "SSM": "#2ca02c",         # green
    "Linformer": "#9467bd",   # purple
    "Perceiver": "#8c564b",
}
MARKERS = {"LST": "o", "Transformer": "s", "SSM": "^", "Linformer": "D", "Perceiver": "P"}


def load_json(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fig_throughput():
    """Forward+backward time vs sequence length."""
    data = load_json("exp2_throughput.json") or load_json("exp2_throughput_partial.json")
    if data is None:
        print("No throughput data"); return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    # Left: ms/step vs N
    ax = axes[0]
    by_model = {}
    for d in data:
        if d.get("status") != "OK":
            continue
        by_model.setdefault(d["model"], []).append((d["seq_len"], d["avg_ms_per_step"]))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
        ax.plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                label=m, linewidth=2, markersize=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Sequence length $N$"); ax.set_ylabel("Time per step (ms)")
    ax.set_title("Throughput (forward+backward)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    # Right: peak memory vs N
    ax = axes[1]
    by_model = {}
    for d in data:
        if d.get("status") != "OK":
            continue
        by_model.setdefault(d["model"], []).append((d["seq_len"], d["peak_mem_mb"]))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
        ax.plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                label=m, linewidth=2, markersize=6)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Sequence length $N$"); ax.set_ylabel("Peak memory (MB)")
    ax.set_title("Memory footprint")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "throughput.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "throughput.png", bbox_inches="tight", dpi=150)
    print("Wrote throughput.pdf")


def fig_mqar_scaling():
    """MQAR accuracy vs n_kv_pairs."""
    data = load_json("mqar_scale.json")
    if data is None:
        print("No mqar_scale data"); return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    by_model = {}
    for d in data:
        by_model.setdefault(d["model"], []).append((d["n_kv"], d["final_acc"]))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
        ax.plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                label=m, linewidth=2, markersize=7)
    ax.set_xlabel("Number of $KV$ pairs"); ax.set_ylabel("MQAR accuracy")
    ax.set_title("MQAR scaling")
    ax.set_ylim(-0.02, 1.0)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "mqar_scaling.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "mqar_scaling.png", bbox_inches="tight", dpi=150)
    print("Wrote mqar_scaling.pdf")


def fig_k_ablation():
    """K (latent dimension) ablation."""
    data = load_json("exp4_k_ablation.json") or load_json("exp4_k_ablation_partial.json")
    if data is None:
        print("No k_ablation data"); return
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    ax = axes[0]
    ks = [d["k"] for d in data]
    accs = [d["final_acc"] for d in data]
    ax.plot(ks, accs, marker='o', color=COLORS["LST"], linewidth=2, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent dimension $K$"); ax.set_ylabel("MQAR accuracy")
    ax.set_title("Accuracy vs $K$")
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    times = [d["time_s"] for d in data]
    ax.plot(ks, times, marker='s', color=COLORS["LST"], linewidth=2, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Latent dimension $K$"); ax.set_ylabel("Training time (s)")
    ax.set_title("Cost vs $K$")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "k_ablation.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "k_ablation.png", bbox_inches="tight", dpi=150)
    print("Wrote k_ablation.pdf")


def fig_induction_copy():
    """Induction and CopyMemory results."""
    data = load_json("exp3_induction_copy.json") or load_json("exp3_partial.json")
    if data is None:
        print("No exp3 data"); return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, task in zip(axes, ["induction", "copy"]):
        by_model = {}
        for d in data:
            if d["task"] != task:
                continue
            by_model.setdefault(d["model"], []).append((d["seq_len"], d["final_acc"]))
        for m, vals in by_model.items():
            vals.sort()
            xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
            ax.plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                    label=m, linewidth=2, markersize=6)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Sequence length $N$"); ax.set_ylabel("Accuracy")
        ax.set_title(task.capitalize())
        ax.set_ylim(-0.02, 1.05)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "induction_copy.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "induction_copy.png", bbox_inches="tight", dpi=150)
    print("Wrote induction_copy.pdf")


def fig_complexity():
    """Theoretical complexity comparison (illustrative)."""
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    N = np.logspace(2, 5, 100)
    D = 128
    K = 64
    # Normalised so that N=1024 -> 1 for LST
    norm = 1024 * K * D
    lst = N * K * D / norm
    transformer = N * N * D / (1024 * 1024 * D)
    ssm = N * D / (1024 * D)
    linformer = N * K * D / norm
    ax.plot(N, transformer, label=f"Transformer $O(N^2 D)$", color=COLORS["Transformer"], linewidth=2)
    ax.plot(N, lst, label=f"LST $O(NK D)$", color=COLORS["LST"], linewidth=2, linestyle='--')
    ax.plot(N, linformer, label=f"Linformer $O(NK D)$", color=COLORS["Linformer"], linewidth=2, linestyle=':')
    ax.plot(N, ssm, label=f"SSM $O(ND)$", color=COLORS["SSM"], linewidth=2, linestyle='-.')
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Sequence length $N$"); ax.set_ylabel("Normalised FLOPs")
    ax.set_title("Theoretical complexity")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "complexity.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "complexity.png", bbox_inches="tight", dpi=150)
    print("Wrote complexity.pdf")


if __name__ == "__main__":
    fig_complexity()
    fig_throughput()
    fig_mqar_scaling()
    fig_k_ablation()
    fig_induction_copy()
    print("Done")
