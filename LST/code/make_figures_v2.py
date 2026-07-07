"""Generate all paper figures from experiment results (v2, includes LSTG)."""
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

COLORS = {
    "LST":  "#d62728",   # red
    "LSTG": "#9467bd",   # purple
    "Transformer": "#1f77b4",  # blue
    "SSM":  "#2ca02c",   # green
    "Linformer":  "#ff7f0e",  # orange
}
MARKERS = {
    "LST": "s", "LSTG": "D", "Transformer": "o",
    "SSM": "^", "Linformer": "v",
}


def load_json(name):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text())


def fig_mqar_scaling():
    """MQAR accuracy vs n_kv_pairs, with LSTG."""
    # Prefer v2, fall back to original
    data = load_json("exp1_mqar_scaling_v2.json") or load_json("exp1_v2_partial.json") \
           or load_json("mqar_scale.json")
    if data is None:
        print("No mqar data"); return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    by_model = {}
    for d in data:
        if "n_kv" in d:
            by_model.setdefault(d["model"], []).append((d["n_kv"], d["final_acc"]))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] * 100 for v in vals]
        ax.plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                label=m, linewidth=2, markersize=7)
    ax.set_xlabel("Number of $KV$ pairs $n_{kv}$")
    ax.set_ylabel("MQAR accuracy (\\%)")
    ax.set_title("MQAR scaling")
    ax.set_ylim(-2, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "mqar_scaling.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "mqar_scaling.png", bbox_inches="tight", dpi=150)
    print("Wrote mqar_scaling.pdf")


def fig_k_ablation():
    """K (latent dimension) ablation, with LSTG."""
    data = load_json("exp4_k_ablation_v2.json") or load_json("exp4_v2_partial.json") \
           or load_json("exp4_k_ablation.json")
    if data is None:
        print("No k-ablation data"); return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # Left: accuracy vs K
    by_model = {}
    for d in data:
        if "k" in d:
            by_model.setdefault(d["model"], []).append((d["k"], d["final_acc"]))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] * 100 for v in vals]
        axes[0].plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                     label=m, linewidth=2, markersize=7)
    axes[0].set_xlabel("Latent dimension $K$")
    axes[0].set_ylabel("MQAR accuracy (\\%)")
    axes[0].set_title("Accuracy vs $K$")
    axes[0].set_ylim(-2, 100)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(True, alpha=0.3)
    # Right: training time vs K
    by_model = {}
    for d in data:
        if "k" in d:
            by_model.setdefault(d["model"], []).append((d["k"], d["time_s"]))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
        axes[1].plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                     label=m, linewidth=2, markersize=7)
    axes[1].set_xlabel("Latent dimension $K$")
    axes[1].set_ylabel("Training time (s)")
    axes[1].set_title("Cost vs $K$")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "k_ablation.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "k_ablation.png", bbox_inches="tight", dpi=150)
    print("Wrote k_ablation.pdf")


def fig_induction_copy():
    """Induction and CopyMemory results, with LSTG."""
    data = load_json("exp3_induction_copy_v2.json") or load_json("exp3_v2_partial.json") \
           or load_json("exp3_induction_copy.json")
    if data is None:
        print("No exp3 data"); return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    for ax, task in zip(axes, ["induction", "copy"]):
        by_model = {}
        for d in data:
            if d.get("task") == task:
                by_model.setdefault(d["model"], []).append((d["seq_len"], d["final_acc"]))
        for m, vals in by_model.items():
            vals.sort()
            xs = [v[0] for v in vals]; ys = [v[1] * 100 for v in vals]
            ax.plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                    label=m, linewidth=2, markersize=7)
        ax.set_xlabel("Sequence length $N$")
        ax.set_ylabel(f"{task.capitalize()} accuracy (\\%)")
        ax.set_title(task.capitalize())
        ax.set_xscale("log")
        ax.set_ylim(-2, 100)
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "induction_copy.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "induction_copy.png", bbox_inches="tight", dpi=150)
    print("Wrote induction_copy.pdf")


def fig_gate_analysis():
    """Gate behaviour in LSTG: per-position-type mean gate value."""
    data = load_json("exp5_gate_analysis_v2.json") or load_json("exp5_gate_analysis.json")
    if data is None:
        print("No gate analysis data"); return
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # Left: bar chart of per-position-type mean gate, one bar group per layer
    pos_types = ["key", "value", "query"]
    pt_colors = {"key": "#1f77b4", "value": "#2ca02c", "query": "#d62728"}
    n_layers = len(data["per_position_type"].get("key", []))
    x = np.arange(n_layers)
    width = 0.25
    for i, pt in enumerate(pos_types):
        means = [e["mean"] for e in data["per_position_type"].get(pt, [])]
        stds = [e["std"] for e in data["per_position_type"].get(pt, [])]
        if means:
            axes[0].bar(x + i * width, means, width, yerr=stds, label=pt,
                        color=pt_colors[pt], alpha=0.8, capsize=3)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels([f"Layer {i}" for i in range(n_layers)])
    axes[0].set_ylabel("Mean gate $g_t$")
    axes[0].set_title("Gate by position type")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    # Right: per-position mean gate across the sequence
    ppm = data.get("per_position_mean", [])
    pos_type = data.get("pos_type", [])
    seq_len = len(ppm)
    if ppm:
        axes[1].plot(range(seq_len), ppm, color="#9467bd", linewidth=2)
        # Color the background by position type
        for t, pt in enumerate(pos_type):
            if isinstance(pt, str):
                pt_name = pt
            else:
                pt_name = ["key", "value", "query"][int(pt)]
            color = pt_colors.get(pt_name, "gray")
            axes[1].axvspan(t - 0.5, t + 0.5, alpha=0.15, color=color)
        # Legend
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor=pt_colors[pt], alpha=0.5, label=pt)
                        for pt in ["key", "value", "query"]]
        axes[1].legend(handles=legend_elems, frameon=False, fontsize=9, loc='upper left')
        # Mark key/value boundaries
        n_kv = data.get("n_kv", 32)
        n_q = data.get("n_q", 8)
        axes[1].axvline(x=2 * n_kv - 0.5, color='black', linestyle='--', alpha=0.5)
        axes[1].text(2 * n_kv - 0.5, 0.95, ' queries start', fontsize=8, va='top')
    axes[1].set_xlabel("Position $t$")
    axes[1].set_ylabel("Mean gate $g_t$")
    axes[1].set_title("Gate across sequence")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "gate_analysis.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "gate_analysis.png", bbox_inches="tight", dpi=150)
    print("Wrote gate_analysis.pdf")


def fig_throughput():
    """Throughput vs sequence length (uses existing exp2 results)."""
    data = load_json("exp2_throughput.json") or load_json("exp2_throughput_partial.json")
    if data is None:
        print("No throughput data"); return
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    by_model = {}
    for d in data:
        if "error" in d: continue
        by_model.setdefault(d["model"], []).append((d["seq_len"], d["throughput_tokens_per_s"]))
    for m, vals in by_model.items():
        vals.sort()
        xs = [v[0] for v in vals]; ys = [v[1] for v in vals]
        ax.plot(xs, ys, marker=MARKERS.get(m, 'o'), color=COLORS.get(m, 'gray'),
                label=m, linewidth=2, markersize=7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Sequence length $N$"); ax.set_ylabel("Throughput (tokens/s)")
    ax.set_title("Throughput scaling")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "throughput.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "throughput.png", bbox_inches="tight", dpi=150)
    print("Wrote throughput.pdf")


def fig_complexity():
    """Theoretical FLOP scaling."""
    N = np.logspace(2, 5, 50)
    D = 128; K = 64
    trans = N**2 * D
    ssm   = N * D
    lst   = N * K * D
    linf  = N * K * D
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.plot(N, trans / trans[0], label=f"Transformer $O(N^2 D)$", color=COLORS["Transformer"], linewidth=2)
    ax.plot(N, lst  / trans[0], label=f"LST $O(NKD)$", color=COLORS["LST"], linewidth=2, linestyle='--')
    ax.plot(N, linf / trans[0], label=f"Linformer $O(NKD)$", color=COLORS["Linformer"], linewidth=2, linestyle=':')
    ax.plot(N, ssm  / trans[0], label=f"SSM $O(ND)$", color=COLORS["SSM"], linewidth=2, linestyle='-.')
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Sequence length $N$"); ax.set_ylabel("Normalised FLOPs")
    ax.set_title("Theoretical complexity")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS / "complexity.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "complexity.png", bbox_inches="tight", dpi=150)
    print("Wrote complexity.pdf")


def fig_long_range():
    """Long-range retrieval: accuracy vs sequence length N."""
    data = load_json("exp6_long_range_v2.json") or load_json("exp6_v2_partial.json")
    if data is None:
        print("No exp6 data"); return
    models = ["LST", "LSTG", "Transformer", "SSM", "Linformer"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name in models:
        entries = [r for r in data if r.get("model") == name and not r.get("skipped") and not r.get("error")]
        if not entries:
            continue
        entries.sort(key=lambda r: r["seq_len"])
        seqs = [r["seq_len"] for r in entries]
        accs = [r["final_acc"] * 100 for r in entries]
        ax.plot(seqs, accs, marker=MARKERS.get(name, "o"), color=COLORS.get(name, "gray"),
                label=name, linewidth=1.5, markersize=6)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence length $N$")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_title("Long-range retrieval (fixed $n_{kv}=16$, $K=64$)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGS / "long_range.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "long_range.png", bbox_inches="tight", dpi=150)
    print("Wrote long_range.pdf")


def fig_noise_robustness():
    """Noise robustness: accuracy vs noise ratio."""
    data = load_json("exp7_noise_v2.json") or load_json("exp7_v2_partial.json")
    if data is None:
        print("No exp7 data"); return
    models = ["LST", "LSTG", "Transformer", "SSM", "Linformer"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for name in models:
        entries = [r for r in data if r.get("model") == name and not r.get("error")]
        if not entries:
            continue
        entries.sort(key=lambda r: r["noise_ratio"])
        noises = [r["noise_ratio"] for r in entries]
        accs = [r["final_acc"] * 100 for r in entries]
        ax.plot(noises, accs, marker=MARKERS.get(name, "o"), color=COLORS.get(name, "gray"),
                label=name, linewidth=1.5, markersize=6)
    ax.set_xlabel("Noise ratio")
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_title("Noise robustness ($N=256$, $n_{kv}=16$)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(FIGS / "noise_robustness.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "noise_robustness.png", bbox_inches="tight", dpi=150)
    print("Wrote noise_robustness.pdf")


def fig_k_vs_n():
    """K vs n_kv scaling frontier for LST vs Linformer."""
    data = load_json("exp8_k_vs_n.json") or load_json("exp8_partial.json")
    if data is None:
        print("No exp8 data"); return
    n_kvs = sorted(set(r["n_kv"] for r in data))
    k_lats = sorted(set(r["k_latent"] for r in data))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for idx, name in enumerate(["LST", "Linformer"]):
        ax = axes[idx]
        entries = [r for r in data if r.get("model") == name]
        # Build matrix: rows = n_kv, cols = k_latent
        mat = np.full((len(n_kvs), len(k_lats)), np.nan)
        for r in entries:
            i = n_kvs.index(r["n_kv"])
            j = k_lats.index(r["k_latent"])
            mat[i, j] = r["final_acc"] * 100
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100, origin="lower")
        ax.set_xticks(range(len(k_lats)))
        ax.set_xticklabels(k_lats)
        ax.set_yticks(range(len(n_kvs)))
        ax.set_yticklabels(n_kvs)
        ax.set_xlabel("Latent budget $K$")
        ax.set_ylabel("$n_{kv}$")
        ax.set_title(name)
        # Annotate cells
        for i in range(len(n_kvs)):
            for j in range(len(k_lats)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.0f}", ha="center", va="center",
                            fontsize=7, color="black" if mat[i,j] > 50 else "white")
    fig.colorbar(im, ax=axes, label="Accuracy (\\%)", shrink=0.8)
    plt.savefig(FIGS / "k_vs_n.pdf", bbox_inches="tight")
    plt.savefig(FIGS / "k_vs_n.png", bbox_inches="tight", dpi=150)
    print("Wrote k_vs_n.pdf")


if __name__ == "__main__":
    fig_complexity()
    fig_throughput()
    fig_mqar_scaling()
    fig_k_ablation()
    fig_induction_copy()
    fig_gate_analysis()
    fig_long_range()
    fig_noise_robustness()
    fig_k_vs_n()
    print("Done")
