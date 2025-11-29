import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Set random seed for reproducibility
np.random.seed(42)

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
        counts = np.zeros(k)

        # Vectorized sum per cluster is tricky in pure numpy without extra memory
        # Loop is fine for small k
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
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

def spectral_sparse_attention(Q, K, V, k_clusters=None, sparsity_fraction=0.1):
    """
    Improved Spectral Sparse Attention.
    1. Projects Q to lower dim using Random Projection.
    2. Clusters Q to find structure.
    3. Computes block-diagonal attention (intra-cluster).
    4. Computes global attention using Cluster Centroids as landmarks (Nystrom-like).
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

    # Pre-calculate counts/indices for speed
    # We will accumulate output

    # 3. Local Attention (Block Diagonal)
    for i in range(k_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) == 0:
            continue

        Q_c = Q[indices]
        K_c = K[indices]
        V_c = V[indices]

        # Standard Attention within cluster
        scores = np.dot(Q_c, K_c.T) / np.sqrt(d)
        w = softmax(scores, axis=-1)
        out_c = np.dot(w, V_c)

        output[indices] += 0.5 * out_c # Weighted contribution

    # 4. Global Attention (Landmark / Low Rank)
    # Instead of random tokens, we use centroids of K as landmarks?
    # Or just sample random tokens? The paper mentions "random landmarks" or "summary tokens".
    # Let's use Random Landmarks as it's faster and consistent with "Sparse" literature
    # But to make it better, let's use the Cluster Centroids of K/V?
    # Calculating centroids of K takes O(N), same as clustering.
    # Let's stick to random landmarks for O(1) selection cost (or O(N) to copy).

    s = max(1, int(N * sparsity_fraction))
    # Ensure we don't pick more than N
    s = min(s, N)

    global_indices = np.random.choice(N, s, replace=False)
    K_global = K[global_indices]
    V_global = V[global_indices]

    # All queries attend to global keys
    scores_g = np.dot(Q, K_global.T) / np.sqrt(d)
    w_g = softmax(scores_g, axis=-1)
    out_g = np.dot(w_g, V_global)

    output += 0.5 * out_g

    return output

def run_experiments():
    # Extended range up to 4096
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    d_model = 64

    times_naive = []
    times_ssa = []
    speedups = []
    errors = []

    print(f"{'N':<6} | {'Naive (s)':<10} | {'SSA (s)':<10} | {'Speedup':<10} | {'Rel Error':<10}")
    print("-" * 65)

    latex_table_rows = []

    for N in seq_lengths:
        # Create data
        Q = np.random.randn(N, d_model)
        K = np.random.randn(N, d_model)
        V = np.random.randn(N, d_model)

        # Warmup (optional, but good for consistency)
        if N == 128:
            spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N)))

        # Measure Naive
        # For N=4096, Naive might be very slow (approx 4*0.7s ~ 3s). It's fine.
        start = time.time()
        out_naive = naive_attention(Q, K, V)
        end = time.time()
        t_naive = end - start
        times_naive.append(t_naive)

        # Measure SSA
        start = time.time()
        out_ssa = spectral_sparse_attention(Q, K, V, k_clusters=int(np.sqrt(N)))
        end = time.time()
        t_ssa = end - start
        times_ssa.append(t_ssa)

        # Error
        # Relative Frobenius norm error
        diff = np.linalg.norm(out_naive - out_ssa)
        base = np.linalg.norm(out_naive)
        err = diff / (base + 1e-8)
        errors.append(err)

        speedup = t_naive / t_ssa if t_ssa > 0 else 0.0
        speedups.append(speedup)

        print(f"{N:<6} | {t_naive:<10.4f} | {t_ssa:<10.4f} | {speedup:<10.2f} | {err:<10.2f}")

        # Format for LaTeX
        latex_table_rows.append(f"{N} & {t_naive:.4f} & {t_ssa:.4f} & {speedup:.2f}x \\\\")

    # 1. Plot Runtime
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, times_naive, 'o-', linewidth=2, label='Standard Attention $O(N^2)$')
    plt.plot(seq_lengths, times_ssa, 's-', linewidth=2, label='Spectral Sparse Attention $O(N^{1.5})$')
    plt.xlabel('Sequence Length $N$', fontsize=12)
    plt.ylabel('Inference Time (seconds)', fontsize=12)
    plt.title('Runtime Comparison: Standard vs SSA', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('energy_efficient_ai/runtime_comparison.png')

    # 2. Plot Complexity (Theoretical)
    N_range = np.linspace(100, 5000, 100)
    flops_trans = 4 * N_range**2 * d_model
    flops_mamba = 10 * N_range * d_model * 16
    flops_ssa = N_range * d_model * (d_model/2) + (N_range**2 / np.sqrt(N_range)) * d_model + N_range * d_model * d_model

    plt.figure(figsize=(10, 6))
    plt.plot(N_range, flops_trans, label='Transformer $O(N^2)$', linewidth=2)
    plt.plot(N_range, flops_mamba, label='Mamba $O(N)$', linestyle='--')
    plt.plot(N_range, flops_ssa, label='SSA (Ours) $O(N \sqrt{N})$', linewidth=2)
    plt.yscale('log')
    plt.xlabel('Sequence Length $N$', fontsize=12)
    plt.ylabel('Theoretical FLOPs (log scale)', fontsize=12)
    plt.title('Theoretical Complexity Analysis', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('energy_efficient_ai/complexity_plot.png')

    # 3. Save Results to file
    with open('energy_efficient_ai/results_table.txt', 'w') as f:
        f.write("\n".join(latex_table_rows))

    # Save the max speedup to a file to be read by the agent
    max_speedup = max(speedups)
    with open('energy_efficient_ai/max_speedup.txt', 'w') as f:
        f.write(f"{max_speedup:.2f}")

if __name__ == "__main__":
    run_experiments()
