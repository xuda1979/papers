import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

# Set random seed for reproducibility
np.random.seed(42)

# Professional Plotting Setup
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'grid.alpha': 0.3
})

def generate_structured_data(N, d, num_clusters=10, intra_cluster_std=0.01):
    """
    Generates Q and K with strictly orthogonal cluster structure.
    """
    if num_clusters <= d:
        H, _ = np.linalg.qr(np.random.randn(d, d))
        centers = H[:num_clusters] * 5.0
    else:
        centers = np.random.randn(num_clusters, d)
        centers = centers / np.linalg.norm(centers, axis=1, keepdims=True) * 5.0

    cluster_assignments = np.random.randint(0, num_clusters, size=N)

    Q = np.zeros((N, d))
    K = np.zeros((N, d))
    V = np.random.randn(N, d)

    for i in range(N):
        c_idx = cluster_assignments[i]
        center = centers[c_idx]

        Q[i] = center + np.random.randn(d) * intra_cluster_std
        K[i] = center + np.random.randn(d) * intra_cluster_std

    return Q, K, V, cluster_assignments

def simple_kmeans(X, k, max_iters=20):
    """
    K-means implementation with K-means++ style initialization for stability.
    """
    N, d = X.shape

    # K-means++ initialization
    centroids = np.empty((k, d))
    # 1. Choose first center uniformly at random
    centroids[0] = X[np.random.choice(N)]

    # 2. Choose remaining k-1 centers with probability proportional to D(x)^2
    for i in range(1, k):
        dist_sq = np.min(np.sum((X[:, None, :] - centroids[:i])**2, axis=2), axis=1)
        probs = dist_sq / np.sum(dist_sq)
        cumprobs = np.cumsum(probs)
        r = np.random.rand()

        # Find index where r falls
        idx = np.searchsorted(cumprobs, r)
        centroids[i] = X[idx]

    labels = np.zeros(N)

    for _ in range(max_iters):
        # Compute distances
        # dist[i, j] = ||X[i] - centroids[j]||^2
        X_sq = np.sum(X**2, axis=1, keepdims=True)
        C_sq = np.sum(centroids**2, axis=1)
        dist = X_sq + C_sq - 2 * np.dot(X, centroids.T)

        new_labels = np.argmin(dist, axis=1)

        if np.array_equal(labels, new_labels):
            break

        labels = new_labels

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = X[mask].mean(axis=0)
            else:
                # Handle empty cluster: re-initialize to a random point
                new_centroids[i] = X[np.random.choice(N)]
        centroids = new_centroids

    return labels, centroids

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def naive_attention(Q, K, V):
    d = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d)
    weights = softmax(scores, axis=-1)
    output = np.dot(weights, V)
    return output

def spectral_sparse_attention(Q, K, V, k_clusters=None, global_ratio=1.0, return_mask=False):
    N, d = Q.shape
    if k_clusters is None:
        k_clusters = int(np.sqrt(N))

    d_proj = 32
    Omega = np.random.randn(d, d_proj)
    Q_proj = np.dot(Q, Omega)

    labels, _ = simple_kmeans(Q_proj, k_clusters, max_iters=10)

    output = np.zeros_like(V)
    mask_vis = None
    if return_mask:
        mask_vis = np.zeros((N, N), dtype=int)

    num_global = int(global_ratio * np.sqrt(N))
    num_global = max(1, min(N, num_global))
    global_indices = np.random.choice(N, num_global, replace=False)

    for i in range(k_clusters):
        query_indices = np.where(labels == i)[0]
        if len(query_indices) == 0:
            continue

        local_key_indices = query_indices
        combined_key_indices = np.union1d(local_key_indices, global_indices)

        if return_mask:
             for q_idx in query_indices:
                 mask_vis[q_idx, combined_key_indices] = 1

        Q_block = Q[query_indices]
        K_block = K[combined_key_indices]
        V_block = V[combined_key_indices]

        scores = np.dot(Q_block, K_block.T) / np.sqrt(d)
        w = softmax(scores, axis=-1)
        out_block = np.dot(w, V_block)
        output[query_indices] = out_block

    if return_mask:
        return output, mask_vis
    return output

def cosine_similarity(A, B):
    dot = np.sum(A * B, axis=1)
    normA = np.linalg.norm(A, axis=1)
    normB = np.linalg.norm(B, axis=1)
    sim = dot / (normA * normB + 1e-8)
    return np.mean(sim)

def run_experiments():
    # Use relative path assuming running from repo root
    output_dir = 'energy_efficient_ai'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    d_model = 128

    times_naive = []
    times_ssa = []
    speedups = []
    errors = []
    cos_sims = []

    print(f"{'N':<6} | {'Naive (s)':<10} | {'SSA (s)':<10} | {'Speedup':<10} | {'Rel Err':<10} | {'Cos Sim':<10}")
    print("-" * 80)
    latex_table_rows = []

    for N in seq_lengths:
        Q, K, V, _ = generate_structured_data(N, d_model, num_clusters=max(2, int(np.sqrt(N))))

        # Warmup for small N
        if N == 128:
             spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N)))

        start = time.time()
        out_naive = naive_attention(Q, K, V)
        times_naive.append(time.time() - start)

        start = time.time()
        out_ssa = spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N)), global_ratio=2.0)
        t_ssa = time.time() - start
        times_ssa.append(t_ssa)

        diff = np.linalg.norm(out_naive - out_ssa)
        base = np.linalg.norm(out_naive)
        err = diff / (base + 1e-8)
        errors.append(err)

        sim = cosine_similarity(out_naive, out_ssa)
        cos_sims.append(sim)

        speedup = times_naive[-1] / t_ssa if t_ssa > 0 else 0.0
        speedups.append(speedup)

        print(f"{N:<6} | {times_naive[-1]:<10.4f} | {t_ssa:<10.4f} | {speedup:<10.2f} | {err:<10.2f} | {sim:<10.4f}")
        latex_table_rows.append(f"{N} & {times_naive[-1]:.4f} & {t_ssa:.4f} & {speedup:.2f}x & {err:.4f} & {sim:.4f} \\\\")

    with open(os.path.join(output_dir, 'results_table.txt'), 'w') as f:
        f.write("\n".join(latex_table_rows))
    with open(os.path.join(output_dir, 'max_speedup.txt'), 'w') as f:
        f.write(f"{max(speedups):.2f}")

    # --- Composite Figure: Runtime & Complexity ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Runtime
    sns.lineplot(x=seq_lengths, y=times_naive, marker='o', label='Standard Attention $O(N^2)$', ax=axes[0], color='#333333', linestyle='--')
    sns.lineplot(x=seq_lengths, y=times_ssa, marker='s', label='SSA (Ours) $O(N^{1.5})$', ax=axes[0], color='#d62728', linewidth=2.5)
    axes[0].set_xlabel('Sequence Length $N$')
    axes[0].set_ylabel('Inference Time (s)')
    axes[0].set_title('Runtime Comparison')
    axes[0].legend()
    axes[0].set_yscale('log') # Log scale helps differentiate at lower N

    # Plot 2: Complexity
    N_range = np.linspace(100, 5000, 100)
    flops_trans = 4 * N_range**2 * d_model
    flops_mamba = 10 * N_range * d_model * 16
    flops_ssa = 2 * (N_range**1.5) * d_model

    axes[1].plot(N_range, flops_trans, label='Transformer $O(N^2)$', color='#333333', linestyle=':', linewidth=1.5)
    axes[1].plot(N_range, flops_mamba, label='Mamba $O(N)$', color='#1f77b4', linestyle='--', linewidth=1.5)
    axes[1].plot(N_range, flops_ssa, label='SSA (Ours) $O(N \sqrt{N})$', color='#d62728', linewidth=2.5)
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Sequence Length $N$')
    axes[1].set_ylabel('Theoretical FLOPs')
    axes[1].set_title('Complexity Analysis')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_performance.png'), bbox_inches='tight')

    # Save individual plots for flexibility
    # Save the runtime plot separately
    extent1 = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(output_dir, 'runtime_comparison.png'), bbox_inches=extent1.expanded(1.1, 1.2))

    # Save the complexity plot separately
    extent2 = axes[1].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(output_dir, 'complexity_plot.png'), bbox_inches=extent2.expanded(1.1, 1.2))


    # --- Pareto Frontier ---
    N_pareto = 1024
    ratios = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    p_speedups = []
    p_cosines = []
    Q, K, V, _ = generate_structured_data(N_pareto, d_model)

    start = time.time()
    out_n = naive_attention(Q, K, V)
    t_base = time.time() - start

    for r in ratios:
        start = time.time()
        out_s = spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N_pareto)), global_ratio=r)
        t_s = time.time() - start

        sim = cosine_similarity(out_n, out_s)
        p_speedups.append(t_base/t_s)
        p_cosines.append(sim)

    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(p_speedups, p_cosines, c=np.log2(ratios), cmap='viridis', s=100, zorder=5)
    plt.plot(p_speedups, p_cosines, color='gray', linestyle='--', alpha=0.5, zorder=1)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Global Key Ratio ($log_2(r)$)')

    plt.xlabel('Speedup Factor ($t_{naive} / t_{ssa}$)')
    plt.ylabel('Cosine Similarity (Accuracy)')
    plt.title(f'Efficiency vs. Accuracy Frontier ($N={N_pareto}$)')

    # Annotate points
    for i, r in enumerate(ratios):
        if i % 2 == 0: # Annotate every other point to avoid clutter
            plt.annotate(f"r={r}", (p_speedups[i], p_cosines[i]), xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'pareto_frontier.png'), bbox_inches='tight')

    # --- Sparsity Pattern ---
    N_vis = 128
    Q, K, V, _ = generate_structured_data(N_vis, d_model, num_clusters=4)
    _, mask = spectral_sparse_attention(Q, K, V, k_clusters=4, global_ratio=1.0, return_mask=True)

    plt.figure(figsize=(6, 6))
    sns.heatmap(mask, cmap="Blues", cbar=False, square=True, xticklabels=False, yticklabels=False)
    plt.xlabel('Key Index $j$')
    plt.ylabel('Query Index $i$')
    plt.title(f'Attention Mask Pattern ($N={N_vis}$)\nBlock-Diagonal + Vertical Stripes')
    plt.savefig(os.path.join(output_dir, 'sparsity_pattern.png'), bbox_inches='tight')

if __name__ == "__main__":
    run_experiments()
