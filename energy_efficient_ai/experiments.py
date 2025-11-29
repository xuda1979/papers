import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Set random seed for reproducibility
np.random.seed(42)

# --- Plotting Style Configuration ---
def set_plot_style():
    """Sets a professional plotting style suitable for LaTeX papers."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'lines.linewidth': 2.5,
        'grid.alpha': 0.6,
        'grid.linestyle': ':',
        'figure.autolayout': True,
        'axes.grid': True,
        'axes.axisbelow': True,  # Grid behind data
        'axes.spines.top': False,
        'axes.spines.right': False,
    })

def simple_kmeans(X, k, max_iters=20):
    """
    Simple K-Means implementation using NumPy.
    X: (N, d) data
    k: number of clusters
    """
    N, d = X.shape
    # Initialize centroids randomly
    indices = np.random.choice(N, k, replace=False)
    centroids = X[indices]

    labels = np.zeros(N)

    for _ in range(max_iters):
        # Optimized distance calc: ||X - C||^2 = ||X||^2 + ||C||^2 - 2XC^T
        X_sq = np.sum(X**2, axis=1, keepdims=True) # (N, 1)
        C_sq = np.sum(centroids**2, axis=1)        # (k,)

        # Broadcast: (N, 1) + (k,) - (N, k) -> (N, k)
        dist = X_sq + C_sq - 2 * np.dot(X, centroids.T)

        new_labels = np.argmin(dist, axis=1)

        # Check convergence
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        # Vectorized mean
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                # Re-init empty cluster
                new_centroids[i] = X[np.random.choice(N)]

        centroids = new_centroids

    return labels, centroids

def softmax(x, axis=-1):
    # Numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def naive_attention(Q, K, V):
    """
    Standard O(N^2) Attention.
    """
    d = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d)
    weights = softmax(scores, axis=-1)
    output = np.dot(weights, V)
    return output

def spectral_sparse_attention(Q, K, V, k_clusters=None, global_ratio=1.0):
    """
    Improved Spectral Sparse Attention.
    """
    N, d = Q.shape
    if k_clusters is None:
        k_clusters = int(np.sqrt(N))

    # 1. Projection (Spectral Embedding Approximation)
    d_proj = 16
    Omega = np.random.randn(d, d_proj)
    Q_proj = np.dot(Q, Omega)

    # 2. Clustering
    labels, _ = simple_kmeans(Q_proj, k_clusters, max_iters=10)

    output = np.zeros_like(V)

    # 3. Select Global Keys
    num_global = int(global_ratio * np.sqrt(N))
    num_global = max(1, min(N, num_global))

    global_indices = np.random.choice(N, num_global, replace=False)

    # 4. Compute Sparse Attention per Cluster
    for i in range(k_clusters):
        query_indices = np.where(labels == i)[0]
        if len(query_indices) == 0:
            continue

        local_key_indices = query_indices
        combined_key_indices = np.union1d(local_key_indices, global_indices)

        Q_block = Q[query_indices]
        K_block = K[combined_key_indices]
        V_block = V[combined_key_indices]

        scores = np.dot(Q_block, K_block.T) / np.sqrt(d)
        w = softmax(scores, axis=-1)
        out_block = np.dot(w, V_block)
        output[query_indices] = out_block

    return output

def plot_sparsity_pattern(N=256, k=16, global_ratio=4.0):
    """
    Generates and plots the effective attention mask for SSA.
    """
    d = 64
    Q = np.random.randn(N, d)

    # 1. Projection & Clustering
    d_proj = 16
    Omega = np.random.randn(d, d_proj)
    Q_proj = np.dot(Q, Omega)
    labels, _ = simple_kmeans(Q_proj, k, max_iters=10)

    # Sort indices by cluster to make the structure visible
    sorted_indices = np.argsort(labels)

    # 2. Global Keys
    num_global = int(global_ratio * np.sqrt(N))
    global_indices = np.random.choice(N, num_global, replace=False)

    # 3. Build Mask
    mask = np.zeros((N, N))

    for i in range(N):
        original_idx = sorted_indices[i] # The actual token index
        cluster_id = labels[original_idx]

        # Local attention: all tokens in same cluster
        cluster_members = np.where(labels == cluster_id)[0]

        # Convert cluster members back to the sorted view coordinates for visualization?
        # No, let's plot the mask in the SORTED coordinate system.
        # So if row 'i' corresponds to token 'original_idx',
        # we want to put 1s at columns 'j' where token 'sorted_indices[j]' is attended to.

        # Keys attended by 'original_idx':
        # 1. Cluster members
        # 2. Global indices
        allowed_keys = np.union1d(cluster_members, global_indices)

        # Now find where these allowed keys are in the sorted list
        # We need an inverse mapping: token_id -> sorted_rank
        # But for visualization, it's easier to just construct the mask in original ID space
        # and then permute rows/cols.
        pass

    # Construct mask in original space
    M_raw = np.zeros((N, N))
    for idx in range(N):
        c_id = labels[idx]
        local_keys = np.where(labels == c_id)[0]
        active = np.union1d(local_keys, global_indices)
        M_raw[idx, active] = 1

    # Permute to grouped clustering
    M_sorted = M_raw[sorted_indices, :][:, sorted_indices]

    plt.figure(figsize=(8, 8))
    # Use 'Blues' or 'Greys' map. 1 is black/blue, 0 is white.
    plt.imshow(M_sorted, cmap='Blues', interpolation='nearest', aspect='equal')
    plt.title(f"SSA Attention Sparsity Pattern ($N={N}$)", pad=20)
    plt.xlabel("Key Token Index (Sorted by Cluster)")
    plt.ylabel("Query Token Index (Sorted by Cluster)")

    # Add annotations
    plt.text(N*1.02, N/2, "Block Diagonal\n(Local Context)", rotation=270, va='center', fontsize=12, color='blue')

    # Since global keys are scattered, they appear as vertical lines in the sorted view
    # (because a global key is attended to by everyone).
    # In the sorted view, if key K is global, then column K' (where K maps to) is full.

    plt.tight_layout()
    plt.savefig('energy_efficient_ai/sparsity_pattern.png', dpi=300)
    plt.close()

def plot_pareto_frontier(N=1024):
    """
    Plots the trade-off between Speedup and Relative Error.
    """
    d_model = 64
    Q = np.random.randn(N, d_model)
    K = np.random.randn(N, d_model)
    V = np.random.randn(N, d_model)

    # Baseline
    start = time.time()
    out_naive = naive_attention(Q, K, V)
    t_naive = time.time() - start

    # Configurations
    k_values = [int(x * np.sqrt(N)) for x in [0.5, 1.0, 2.0]]
    global_ratios = [0.0, 0.5, 1.0, 2.0, 4.0]

    results = []

    for k in k_values:
        for gr in global_ratios:
            start = time.time()
            out_ssa = spectral_sparse_attention(Q, K, V, k_clusters=k, global_ratio=gr)
            t_ssa = time.time() - start

            diff = np.linalg.norm(out_naive - out_ssa)
            base = np.linalg.norm(out_naive)
            err = diff / (base + 1e-8)

            speedup = t_naive / t_ssa if t_ssa > 0 else 0
            results.append((k, gr, speedup, err))

    # Plot
    plt.figure(figsize=(10, 7))

    # Group by k to draw lines
    k_groups = sorted(list(set(r[0] for r in results)))
    markers = ['o', 's', '^', 'D']

    for i, k in enumerate(k_groups):
        subset = [r for r in results if r[0] == k]
        subset.sort(key=lambda x: x[1]) # Sort by global ratio

        s_vals = [r[2] for r in subset]
        e_vals = [r[3] for r in subset]

        plt.plot(s_vals, e_vals, marker=markers[i % len(markers)], label=f'Clusters $k={k}$', markersize=8)

        # Annotate global ratios
        for j, (s, e) in enumerate(zip(s_vals, e_vals)):
            if j == 0 or j == len(s_vals)-1:
                gr = subset[j][1]
                plt.annotate(f"$\gamma={gr}$", (s, e), textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)

    plt.title(f"Efficiency-Accuracy Trade-off ($N={N}$)")
    plt.xlabel("Speedup Factor (vs Naive)")
    plt.ylabel("Relative Approximation Error")
    plt.legend()

    # Add an ideal corner annotation
    plt.annotate("Ideal Region\n(High Speedup, Low Error)", xy=(max([r[2] for r in results]), min([r[3] for r in results])),
                 xytext=(-80, 40), textcoords='offset points', arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    plt.tight_layout()
    plt.savefig('energy_efficient_ai/pareto_frontier.png', dpi=300)
    plt.close()

def run_experiments():
    set_plot_style()

    # 0. Generate New Advanced Plots
    print("Generating Sparsity Pattern Plot...")
    plot_sparsity_pattern(N=256, k=8, global_ratio=2.0)

    print("Generating Pareto Frontier Plot...")
    plot_pareto_frontier(N=1024)

    # 1. Main Runtime Benchmark
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    d_model = 64

    times_naive = []
    times_ssa = []
    speedups = []

    print("Running Main Benchmark...")

    latex_table_rows = []

    for N in seq_lengths:
        Q = np.random.randn(N, d_model)
        K = np.random.randn(N, d_model)
        V = np.random.randn(N, d_model)

        # Warmup
        if N == 128:
            spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N)))

        start = time.time()
        naive_attention(Q, K, V)
        t_naive = time.time() - start
        times_naive.append(t_naive)

        start = time.time()
        # Use roughly sqrt(N) global tokens
        spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N)), global_ratio=4.0)
        t_ssa = time.time() - start
        times_ssa.append(t_ssa)

        speedup = t_naive / t_ssa if t_ssa > 0 else 0.0
        speedups.append(speedup)

        # Calculate error just for the table
        out_n = naive_attention(Q, K, V)
        out_s = spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N)), global_ratio=4.0)
        err = np.linalg.norm(out_n - out_s) / (np.linalg.norm(out_n) + 1e-8)

        print(f"N={N}: Speedup={speedup:.2f}x")
        latex_table_rows.append(f"{N} & {t_naive:.4f} & {t_ssa:.4f} & {speedup:.2f}x & {err:.4f} \\\\")

    # Plot Runtime with better style
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, times_naive, 'o-', color='#d62728', label='Standard Attention $O(N^2)$')
    plt.plot(seq_lengths, times_ssa, 's-', color='#1f77b4', label='Spectral Sparse Attention $O(N^{1.5})$')

    # Add fill between to highlight the gap
    plt.fill_between(seq_lengths, times_naive, times_ssa, color='gray', alpha=0.1, label='Computation Saved')

    plt.xlabel('Sequence Length $N$')
    plt.ylabel('Inference Time (s)')
    plt.title('Runtime Scaling Analysis')
    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_efficient_ai/runtime_comparison.png', dpi=300)
    plt.close()

    # Plot Complexity with better style
    N_range = np.linspace(100, 5000, 100)
    flops_trans = 4 * N_range**2 * d_model
    flops_mamba = 10 * N_range * d_model * 16
    flops_ssa = 2 * (N_range**1.5) * d_model

    plt.figure(figsize=(10, 6))
    plt.plot(N_range, flops_trans, color='#d62728', linewidth=3, label='Transformer (Quadratic)')
    plt.plot(N_range, flops_mamba, color='#2ca02c', linestyle='--', linewidth=2, label='SSM/Mamba (Linear)')
    plt.plot(N_range, flops_ssa, color='#1f77b4', linewidth=3, label='SSA (Sub-quadratic)')

    plt.yscale('log')
    plt.xlabel('Sequence Length $N$')
    plt.ylabel('Theoretical FLOPs (log scale)')
    plt.title('Asymptotic Complexity Landscape')

    # Annotate the crossover or gap
    plt.annotate('Efficiency Gap', xy=(4000, flops_trans[-1]), xytext=(3000, flops_ssa[-1]),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.legend()
    plt.tight_layout()
    plt.savefig('energy_efficient_ai/complexity_plot.png', dpi=300)
    plt.close()

    with open('energy_efficient_ai/results_table.txt', 'w') as f:
        f.write("\n".join(latex_table_rows))

    with open('energy_efficient_ai/max_speedup.txt', 'w') as f:
        f.write(f"{max(speedups):.2f}")

if __name__ == "__main__":
    run_experiments()
