"""
Generate publication-quality plots for the SSA paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set style for publication
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.figsize': (8, 5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

OUTPUT_DIR = '.'

# =============================================================================
# Plot 1: Scalability (exp1_scalability.png)
# =============================================================================

def plot_scalability():
    """Runtime scalability comparison."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Actual measured data from experiments
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    
    # Times in ms (from our experiments)
    dense_time = [1.8, 8.7, 26.3, 144.6, 389.4, 2051.3]
    ssa_time = [42.8, 209.5, 331.8, 1983.5, 6172.1, 25498.0]
    local_time = [20.7, 76.8, 299.8, 257.1, 607.7, 1187.1]
    
    ax.plot(seq_lengths, dense_time, 'o-', label='Dense $O(N^2)$', 
            linewidth=2.5, markersize=8, color='#1f77b4')
    ax.plot(seq_lengths, ssa_time, 's-', label='SSA (NumPy ref.)', 
            linewidth=2, markersize=7, color='#ff7f0e')
    ax.plot(seq_lengths, local_time, 'd-', label='Local-256', 
            linewidth=2, markersize=6, color='#d62728')
    
    ax.set_xlabel('Sequence Length $N$')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Attention Computation Time (CPU Reference Implementation)')
    ax.legend(loc='upper left')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([str(n) for n in seq_lengths])
    
    # Add annotation about NumPy overhead
    ax.annotate('NumPy overhead\\ndominates at small $N$', 
                xy=(512, 209), xytext=(200, 500),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp1_scalability.png'))
    plt.close()
    print("Saved exp1_scalability.png")


# =============================================================================
# Plot 2: Spectral Preservation (exp2_spectral.png)
# =============================================================================

def plot_spectral():
    """Spectral gap and eigenvalue preservation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Simulated eigenvalue data
    np.random.seed(42)
    k_values = [4, 8, 16, 32]
    
    # Top 10 eigenvalues for dense attention (sorted)
    dense_eigs = np.array([0.0, 0.012, 0.025, 0.041, 0.058, 0.078, 0.101, 0.127, 0.156, 0.188])
    
    # Plot eigenvalues for different k
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(k_values)))
    
    ax1.plot(range(10), dense_eigs, 'k-o', linewidth=2.5, markersize=8, 
             label='Dense', zorder=10)
    
    for i, k in enumerate(k_values):
        # SSA eigenvalues with increasing error for larger k
        noise = np.random.randn(10) * (0.005 + 0.002 * k)
        ssa_eigs = dense_eigs + noise
        ssa_eigs = np.sort(np.clip(ssa_eigs, 0, None))
        ax1.plot(range(10), ssa_eigs, 's--', color=colors[i], 
                linewidth=1.5, markersize=6, label=f'SSA ($k={k}$)', alpha=0.8)
    
    ax1.set_xlabel('Eigenvalue Index')
    ax1.set_ylabel('Eigenvalue $\\lambda_i$')
    ax1.set_title('Laplacian Eigenvalue Preservation')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Spectral gap preservation
    k_range = np.arange(4, 65, 4)
    gap_preservation = 1.0 - 0.008 * k_range + 0.0001 * k_range**2
    gap_preservation = np.clip(gap_preservation, 0.85, 1.0)
    
    ax2.plot(k_range, gap_preservation * 100, 'o-', linewidth=2.5, 
             markersize=8, color='#2ca02c')
    ax2.axhline(y=95, color='gray', linestyle='--', alpha=0.5, label='95% threshold')
    ax2.fill_between(k_range, 95, gap_preservation * 100, 
                      where=gap_preservation * 100 >= 95, alpha=0.2, color='green')
    
    ax2.set_xlabel('Number of Clusters $k$')
    ax2.set_ylabel('Spectral Gap Preservation (\\%)')
    ax2.set_title('Gap Preservation vs. Cluster Count')
    ax2.set_ylim(80, 102)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp2_spectral.png'))
    plt.close()
    print("Saved exp2_spectral.png")


# =============================================================================
# Plot 3: Long-Range Performance (exp3_long_range.png)
# =============================================================================

def plot_long_range():
    """Needle-in-haystack retrieval performance."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Actual data from experiments
    seq_lengths = [512, 1024, 2048, 4096]
    
    dense = [1.000, 1.000, 1.000, 1.000]
    ssa = [1.000, 1.000, 1.000, 1.000]
    routing = [1.000, 1.000, 1.000, 1.000]
    local = [0.016, 0.008, -0.007, -0.018]
    reformer = [1.000, 1.000, 1.000, 1.000]
    
    x = np.arange(len(seq_lengths))
    width = 0.15
    
    bars1 = ax.bar(x - 2*width, dense, width, label='Dense', color='#1f77b4')
    bars2 = ax.bar(x - width, ssa, width, label='SSA (Ours)', color='#2ca02c')
    bars3 = ax.bar(x, routing, width, label='Routing Trans.', color='#ff7f0e')
    bars4 = ax.bar(x + width, local, width, label='Local-256', color='#d62728')
    bars5 = ax.bar(x + 2*width, reformer, width, label='Reformer', color='#9467bd')
    
    ax.set_xlabel('Sequence Length $N$')
    ax.set_ylabel('Retrieval Accuracy (Cosine Similarity)')
    ax.set_title('Needle-in-Haystack Retrieval Task')
    ax.set_xticks(x)
    ax.set_xticklabels(seq_lengths)
    ax.legend(loc='lower left', ncol=2)
    ax.set_ylim(-0.1, 1.15)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Annotation
    ax.annotate('Local attention cannot\nreach distant needle', 
                xy=(2, -0.007), xytext=(2.5, 0.4),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp3_long_range.png'))
    plt.close()
    print("Saved exp3_long_range.png")


# =============================================================================
# Plot 4: Energy Scaling (exp4_energy.png)
# =============================================================================

def plot_energy():
    """Energy proxy scaling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    seq_lengths = [256, 512, 1024, 2048, 4096]
    
    # From Table 6 in paper
    dense_fp16 = [0.03, 0.10, 0.41, 1.62, 6.49]
    ssa_fp16 = [0.01, 0.02, 0.05, 0.12, 0.34]
    ssa_bitnet = [0.001, 0.003, 0.009, 0.024, 0.068]
    
    ax.plot(seq_lengths, dense_fp16, 'o-', label='Dense FP16', 
            linewidth=2.5, markersize=8, color='#1f77b4')
    ax.plot(seq_lengths, ssa_fp16, 's-', label='SSA FP16', 
            linewidth=2.5, markersize=8, color='#ff7f0e')
    ax.plot(seq_lengths, ssa_bitnet, '^-', label='SSA + BitNet', 
            linewidth=2.5, markersize=8, color='#2ca02c')
    
    ax.set_xlabel('Sequence Length $N$')
    ax.set_ylabel('Normalized Energy Proxy')
    ax.set_title('Energy Scaling Comparison')
    ax.legend(loc='upper left')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(seq_lengths)
    ax.set_xticklabels([str(n) for n in seq_lengths])
    
    # Add annotation for savings
    ax.annotate('95× savings\nat N=4096', 
                xy=(4096, 0.068), xytext=(2500, 0.3),
                fontsize=11, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp4_energy.png'))
    plt.close()
    print("Saved exp4_energy.png")


# =============================================================================
# Plot 5: Ablation (exp5_ablation.png)
# =============================================================================

def plot_ablation():
    """Ablation on number of clusters."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # From Table 7
    clusters = [2, 4, 8, 16, 32]
    density = [0.53, 0.28, 0.16, 0.09, 0.06]
    cos_sim = [0.807, 0.732, 0.629, 0.535, 0.449]
    time_s = [0.016, 0.017, 0.027, 0.059, 0.162]
    
    # Left: Density and Similarity vs k
    ax1_twin = ax1.twinx()
    
    line1, = ax1.plot(clusters, [d*100 for d in density], 'o-', 
                      color='#1f77b4', linewidth=2.5, markersize=8, label='Density (%)')
    line2, = ax1_twin.plot(clusters, cos_sim, 's-', 
                           color='#ff7f0e', linewidth=2.5, markersize=8, label='Cos. Similarity')
    
    ax1.set_xlabel('Number of Clusters $k$')
    ax1.set_ylabel('Edge Density (\\%)', color='#1f77b4')
    ax1_twin.set_ylabel('Cosine Similarity', color='#ff7f0e')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1_twin.tick_params(axis='y', labelcolor='#ff7f0e')
    ax1.set_title('Density vs. Approximation Quality')
    ax1.grid(True, alpha=0.3)
    
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')
    
    # Right: Pareto frontier
    # Trade-off between sparsity and quality
    sparsity = [1 - d for d in density]
    
    ax2.scatter(sparsity, cos_sim, c=clusters, cmap='viridis', 
                s=200, edgecolors='black', linewidths=1.5)
    
    for i, k in enumerate(clusters):
        ax2.annotate(f'$k$={k}', (sparsity[i], cos_sim[i]), 
                    textcoords='offset points', xytext=(10, 5), fontsize=10)
    
    ax2.set_xlabel('Sparsity (1 - Density)')
    ax2.set_ylabel('Cosine Similarity to Dense')
    ax2.set_title('Accuracy-Sparsity Trade-off')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.4, 1.0)
    ax2.set_ylim(0.4, 0.9)
    
    # Add optimal region annotation
    ax2.axvspan(0.65, 0.85, alpha=0.1, color='green', label='Optimal region')
    ax2.axhspan(0.6, 0.75, alpha=0.1, color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp5_ablation.png'))
    plt.close()
    print("Saved exp5_ablation.png")


# =============================================================================
# Plot 6: Pareto Frontier (exp6_pareto.png)
# =============================================================================

def plot_pareto():
    """Pareto frontier of accuracy vs energy."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Methods: (accuracy proxy, energy proxy)
    # Higher accuracy = better, Lower energy = better
    methods = {
        'Dense FP16': (1.0, 1.0),
        'SSA FP16': (0.95, 0.12),
        'SSA + BitNet': (0.93, 0.015),
        'Local-256': (0.65, 0.08),
        'Reformer': (0.85, 0.15),
        'Routing Trans.': (0.92, 0.13),
        'Linformer': (0.80, 0.10),
    }
    
    colors = {
        'Dense FP16': '#1f77b4',
        'SSA FP16': '#ff7f0e',
        'SSA + BitNet': '#2ca02c',
        'Local-256': '#d62728',
        'Reformer': '#9467bd',
        'Routing Trans.': '#8c564b',
        'Linformer': '#e377c2',
    }
    
    for method, (acc, energy) in methods.items():
        ax.scatter(energy, acc, s=200, c=colors[method], 
                  edgecolors='black', linewidths=1.5, label=method, zorder=10)
    
    # Draw Pareto frontier
    pareto_points = [(0.015, 0.93), (0.12, 0.95), (1.0, 1.0)]
    pareto_x = [p[0] for p in pareto_points]
    pareto_y = [p[1] for p in pareto_points]
    ax.plot(pareto_x, pareto_y, 'k--', linewidth=2, alpha=0.5, label='Pareto frontier')
    
    # Fill Pareto-dominated region
    ax.fill_between([0, 1.2], [0, 0], [1.05, 1.05], alpha=0.03, color='gray')
    
    ax.set_xlabel('Relative Energy Cost')
    ax.set_ylabel('Relative Accuracy')
    ax.set_title('Pareto Frontier: Accuracy vs. Energy')
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.5, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Annotation for SSA + BitNet
    ax.annotate('Best efficiency\n(SSA + BitNet)', 
                xy=(0.015, 0.93), xytext=(0.15, 0.7),
                fontsize=11, ha='center',
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exp6_pareto.png'))
    plt.close()
    print("Saved exp6_pareto.png")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("Generating publication plots...")
    plot_scalability()
    # plot_spectral()  # Removed: uses simulated data
    plot_long_range()
    # plot_energy()  # Removed: uses analytic proxy, not measured
    plot_ablation()
    # plot_pareto()  # Removed: uses made-up proxy values
    print("\nAll plots generated successfully!")

