"""Generate plots for the DTFN paper."""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtfn_core as dc
import torch


def plot_architecture(out_path: str) -> None:
    """Schematic of the DTFN architecture."""
    fig, ax = plt.subplots(figsize=(10, 4))
    L = 6
    d = 8
    np.random.seed(0)
    masses = np.random.rand(L, d) * 0.8 + 0.1

    # Plot mass vectors as colored cells
    for i in range(L):
        rect = plt.Rectangle((i, 0), 1, 1, facecolor='#4A90E2', alpha=0.3, edgecolor='black')
        ax.add_patch(rect)
        # individual "mass" bars
        for j in range(d):
            h = masses[i, j]
            rect = plt.Rectangle((i + 0.1 + j*0.1, 0.1), 0.08, h*0.8,
                                  facecolor='#2E5B8A', alpha=0.7, edgecolor='none')
            ax.add_patch(rect)
        ax.text(i + 0.5, -0.2, f'token {i}', ha='center', fontsize=10)
        ax.text(i + 0.5, 1.1, f'$p_{i}$', ha='center', fontsize=12)

    # Flux arrows
    for i in range(L-1):
        ax.annotate('', xy=(i+1.0, 0.5), xytext=(i+1.0, 0.5),
                    arrowprops=dict(arrowstyle='->', color='#D0021B', lw=2))
        ax.text(i + 1.0, 0.7, f'$F_{{{i}+1/2}}$', ha='center', fontsize=10, color='#D0021B')

    # K update label
    ax.annotate('', xy=(L+0.5, 0.5), xytext=(L+0.5, 0.5))
    ax.text(L+0.2, 0.5, r'$\times K$' + '\nupdate steps', fontsize=12, ha='left', va='center')

    ax.set_xlim(-0.5, L+2)
    ax.set_ylim(-0.5, 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Discrete Token-Flow Network: mass vectors and learned fluxes', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_conservation(out_path: str) -> None:
    """Mass conservation over many steps."""
    torch.manual_seed(0)
    L = 100
    d = 8
    p = torch.rand(1, L, d) + 0.1
    totals = [p.sum().item()]
    mins = [p.min().item()]
    maxs = [p.max().item()]

    # Pure upwind
    cfg = dc.DTFNConfig(vocab_size=4, d=d, K=1, scheme='upwind')
    layer = dc.DTFNLayer(cfg)
    with torch.no_grad():
        for _ in range(50):
            p = layer(p)
            totals.append(p.sum().item())
            mins.append(p.min().item())
            maxs.append(p.max().item())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    steps = np.arange(len(totals))
    axes[0].plot(steps, totals, 'b-', linewidth=2)
    axes[0].set_xlabel('Update step $k$')
    axes[0].set_ylabel('Total mass $\\sum_i p_i$')
    axes[0].set_title('Mass conservation')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, mins, 'g-', linewidth=2, label='min $p$')
    axes[1].plot(steps, maxs, 'r-', linewidth=2, label='max $p$')
    axes[1].set_xlabel('Update step $k$')
    axes[1].set_ylabel('Mass value')
    axes[1].set_title('Positivity & max principle')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # Conservation error vs steps
    rel_err = [abs(t - totals[0]) / totals[0] for t in totals]
    axes[2].plot(steps, rel_err, 'k-', linewidth=2)
    axes[2].set_xlabel('Update step $k$')
    axes[2].set_ylabel('Relative conservation error')
    axes[2].set_title('Conservation error (float precision)')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_needle(out_path: str) -> None:
    """Needle advection across sequence lengths."""
    fig, axes = plt.subplots(1, 4, figsize=(14, 3))
    for ax_idx, L in enumerate([64, 256, 1024, 4096]):
        m0 = torch.zeros(1, L, 1)
        m0[0, 0, 0] = 1.0
        v, dt = 0.8, 0.5
        m = m0.clone()
        for _ in range(L - 1):
            F = v * m[:, :-1, :]
            dm = torch.zeros_like(m)
            dm[:, :-1, :] -= F
            dm[:, 1:, :] += F
            m = torch.clamp(m + dt * dm, min=0.0)
        peak = int(m[0, :, 0].argmax().item())
        expected = int(round((L - 1) * v * dt))
        axes[ax_idx].plot(m[0, :, 0].numpy(), 'b-', linewidth=1.5)
        axes[ax_idx].axvline(expected, color='r', linestyle='--', label=f'expected $i^*$={expected}')
        axes[ax_idx].set_title(f'$L={L}$, peak={peak}')
        axes[ax_idx].set_xlabel('Position $i$')
        if ax_idx == 0:
            axes[ax_idx].set_ylabel('Mass $p_i$')
        axes[ax_idx].legend()
        axes[ax_idx].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_extrapolation(out_path: str, results_path: str) -> None:
    """Length extrapolation comparison."""
    with open(results_path) as f:
        r = json.load(f)
    extrap = r.get('extrap', {})
    if not extrap:
        print("No extrapolation results found")
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    colors = {'DTFN-U': '#4A90E2', 'DTFN-B': '#2E5B8A', 'LSTM': '#D0021B', 'Transformer': '#F5A623'}
    for name, res in extrap.items():
        Ls = sorted(int(L) for L in res['extrap'].keys())
        losses = [res['extrap'][str(L)]['loss'] for L in Ls]
        accs = [res['extrap'][str(L)]['token_acc'] for L in Ls]
        axes[0].plot(Ls, losses, 'o-', label=name, color=colors.get(name, None), linewidth=2)
        axes[1].plot(Ls, accs, 'o-', label=name, color=colors.get(name, None), linewidth=2)
    axes[0].set_xlabel('Test sequence length (pairs)')
    axes[0].set_ylabel('Cross-entropy loss')
    axes[0].set_title('Length extrapolation: loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axvspan(4, 20, alpha=0.1, color='gray', label='train range')
    axes[1].set_xlabel('Test sequence length (pairs)')
    axes[1].set_ylabel('Token accuracy')
    axes[1].set_title('Length extrapolation: accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axvspan(4, 20, alpha=0.1, color='gray')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_scaling(out_path: str, results_path: str) -> None:
    """Wall-clock scaling vs sequence length."""
    with open(results_path) as f:
        r = json.load(f)
    scaling = r.get('scaling', {})
    if not scaling:
        print("No scaling results found")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'DTFN-U': '#4A90E2', 'DTFN-B': '#2E5B8A', 'LSTM': '#D0021B', 'Transformer': '#F5A623'}
    markers = {'DTFN-U': 'o', 'DTFN-B': 's', 'LSTM': '^', 'Transformer': 'D'}
    for name, res in scaling.items():
        Ls = sorted(int(L) for L in res.keys())
        times = [res[str(L)] * 1000 for L in Ls]
        ax.plot(Ls, times, marker=markers.get(name, 'o'), label=name,
                color=colors.get(name, None), linewidth=2, markersize=8)
    ax.set_xlabel('Sequence length $L$')
    ax.set_ylabel('Forward pass time (ms)')
    ax.set_title('Wall-clock scaling vs sequence length')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_dyck_comparison(out_path: str, results_path: str) -> None:
    """Bar chart comparing models on Dyck-1."""
    with open(results_path) as f:
        r = json.load(f)
    if 'summary' in r:
        # multiseed
        models = list(r['summary'].keys())
        losses = [r['summary'][m]['loss_mean'] for m in models]
        loss_stds = [r['summary'][m]['loss_std'] for m in models]
        accs = [r['summary'][m]['tk_acc_mean'] for m in models]
        acc_stds = [r['summary'][m]['tk_acc_std'] for m in models]
        seqs = [r['summary'][m]['seq_acc_mean'] for m in models]
        seq_stds = [r['summary'][m]['seq_acc_std'] for m in models]
    else:
        # single-seed dyck
        models = list(r.keys())
        losses = [r[m]['test']['loss'] for m in models]
        loss_stds = [0]*len(models)
        accs = [r[m]['test']['token_acc'] for m in models]
        acc_stds = [0]*len(models)
        seqs = [r[m]['test']['seq_acc'] for m in models]
        seq_stds = [0]*len(models)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ['#4A90E2', '#2E5B8A', '#7ED321', '#9013FE', '#D0021B', '#F5A623']
    x = np.arange(len(models))
    axes[0].bar(x, losses, yerr=loss_stds, color=colors[:len(models)], capsize=5)
    axes[0].set_xticks(x); axes[0].set_xticklabels(models, rotation=30, ha='right')
    axes[0].set_ylabel('Cross-entropy loss')
    axes[0].set_title('Dyck-1: test loss (lower better)')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, accs, yerr=acc_stds, color=colors[:len(models)], capsize=5)
    axes[1].set_xticks(x); axes[1].set_xticklabels(models, rotation=30, ha='right')
    axes[1].set_ylabel('Token accuracy')
    axes[1].set_title('Dyck-1: token accuracy (higher better)')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(x, seqs, yerr=seq_stds, color=colors[:len(models)], capsize=5)
    axes[2].set_xticks(x); axes[2].set_xticklabels(models, rotation=30, ha='right')
    axes[2].set_ylabel('Sequence accuracy')
    axes[2].set_title('Dyck-1: exact-match (higher better)')
    axes[2].set_ylim(0, 1.05)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def plot_ablation_K(out_path: str, results_path: str) -> None:
    with open(results_path) as f:
        r = json.load(f)
    abl = r.get('ablation_K', {})
    if not abl:
        return
    Ks = sorted(int(k) for k in abl.keys())
    losses = [abl[str(k)]['test']['loss'] for k in Ks]
    accs = [abl[str(k)]['test']['token_acc'] for k in Ks]
    params = [abl[str(k)]['n_params'] for k in Ks]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(Ks, losses, 'o-', color='#4A90E2', linewidth=2, markersize=8, label='Test loss')
    ax2 = axes[0].twinx()
    ax2.plot(Ks, params, 's--', color='#D0021B', linewidth=2, markersize=6, label='Params')
    ax2.set_ylabel('# parameters', color='#D0021B')
    axes[0].set_xlabel('$K$ (number of update steps)')
    axes[0].set_ylabel('Cross-entropy loss', color='#4A90E2')
    axes[0].set_title('Effect of $K$ on Dyck-1 performance')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(Ks, accs, 'o-', color='#7ED321', linewidth=2, markersize=8)
    axes[1].set_xlabel('$K$ (number of update steps)')
    axes[1].set_ylabel('Token accuracy')
    axes[1].set_title('Token accuracy vs $K$')
    axes[1].set_ylim(0.99, 1.001)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="/Users/daxu/papers/dtfn/figures")
    parser.add_argument("--results", default="/Users/daxu/papers/dtfn/results/results.json")
    parser.add_argument("--multiseed", default="/Users/daxu/papers/dtfn/results/multiseed.json")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    plot_architecture(os.path.join(args.output_dir, 'fig_architecture.pdf'))
    plot_conservation(os.path.join(args.output_dir, 'fig_conservation.pdf'))
    plot_needle(os.path.join(args.output_dir, 'fig_needle.pdf'))
    plot_extrapolation(os.path.join(args.output_dir, 'fig_extrapolation.pdf'), args.results)
    plot_scaling(os.path.join(args.output_dir, 'fig_scaling.pdf'), args.results)
    plot_dyck_comparison(os.path.join(args.output_dir, 'fig_dyck_comparison.pdf'), args.multiseed)
    plot_ablation_K(os.path.join(args.output_dir, 'fig_ablation_K.pdf'), args.results)
    print("All plots generated.")


if __name__ == "__main__":
    main()
