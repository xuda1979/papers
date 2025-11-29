import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Set random seed for reproducibility
np.random.seed(42)

def simple_kmeans(X, k, max_iters=10):
    """
    Simple K-Means implementation using NumPy.
    X: (N, d) data
    k: number of clusters
    """
    N, d = X.shape
    # Initialize centroids randomly
    centroids = X[np.random.choice(N, k, replace=False)]

    for _ in range(max_iters):
        # Compute distances: ||X - C||^2 = ||X||^2 + ||C||^2 - 2XC^T
        # But for speed/simplicity, we can just expand dims or use broadcasting
        # Distance matrix: (N, k)

        # Expanding (N, 1, d) - (1, k, d) -> (N, k, d) -> norm -> (N, k)
        # This might be memory heavy for large N. Let's do batch or loop if needed.
        # For simulation sizes (up to 2000), it fits in memory.

        # Optimized distance calc:
        X_sq = np.sum(X**2, axis=1, keepdims=True) # (N, 1)
        C_sq = np.sum(centroids**2, axis=1)        # (k,)
        dist = X_sq + C_sq - 2 * np.dot(X, centroids.T)

        labels = np.argmin(dist, axis=1)

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Re-init empty cluster
                new_centroids[i] = X[np.random.choice(N)]

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return labels, centroids

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def naive_attention(Q, K, V):
    """
    Standard O(N^2) Attention.
    """
    d = Q.shape[-1]
    # QK^T
    scores = np.dot(Q, K.T) / np.sqrt(d)
    weights = softmax(scores, axis=-1)
    output = np.dot(weights, V)
    return output

def spectral_sparse_attention(Q, K, V, k_clusters=None, sparsity_fraction=0.1):
    """
    Mock implementation of Spectral Sparse Attention.
    1. Projects Q, K to lower dim.
    2. Clusters them.
    3. Computes block-diagonal attention (intra-cluster).
    4. Computes low-rank global attention (mocked by random sampling).
    """
    N, d = Q.shape
    if k_clusters is None:
        k_clusters = int(np.sqrt(N))

    # 1. Projection (Mocking 'Spectral Embedding' via Random Projection)
    # Project to d_proj = 16 for clustering speed
    d_proj = 16
    Omega = np.random.randn(d, d_proj)
    Q_proj = np.dot(Q, Omega)

    # 2. Clustering
    labels, _ = simple_kmeans(Q_proj, k_clusters, max_iters=5)

    output = np.zeros_like(V)

    # 3. Local Attention (Block Diagonal)
    # We iterate over clusters and attend only within the cluster
    for i in range(k_clusters):
        indices = np.where(labels == i)[0]
        if len(indices) == 0:
            continue

        Q_c = Q[indices]
        K_c = K[indices]
        V_c = V[indices]

        # Attend
        scores = np.dot(Q_c, K_c.T) / np.sqrt(d)
        w = softmax(scores, axis=-1)
        out_c = np.dot(w, V_c)

        output[indices] += out_c

    # 4. Global Attention (Sparse / Low Rank)
    # Select 's' global tokens. For simplicity, just pick random indices as 'landmarks'
    s = max(1, int(N * sparsity_fraction))
    global_indices = np.random.choice(N, s, replace=False)

    K_global = K[global_indices]
    V_global = V[global_indices]

    # All queries attend to global keys
    scores_g = np.dot(Q, K_global.T) / np.sqrt(d)
    w_g = softmax(scores_g, axis=-1)
    out_g = np.dot(w_g, V_global)

    # Combine (Weighted average or residual - simplistic here)
    # In real implementation, we'd handle the softmax normalization properly across both sets.
    # Here we just average them to simulate the 'cost' and 'flow'.
    output = 0.5 * output + 0.5 * out_g

    return output

def run_experiments():
    seq_lengths = [128, 256, 512, 1024, 2048]
    d_model = 64

    times_naive = []
    times_ssa = []
    errors = []

    print(f"{'N':<6} | {'Naive Time (s)':<15} | {'SSA Time (s)':<15} | {'Speedup':<10} | {'Rel Error':<10}")
    print("-" * 70)

    for N in seq_lengths:
        Q = np.random.randn(N, d_model)
        K = np.random.randn(N, d_model)
        V = np.random.randn(N, d_model)

        # Measure Naive
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

        # Error (just to see if it's somewhat stable, though meaningless without training)
        # Since the approximation is untrain, error will be high.
        # We check magnitude relative difference.
        err = np.linalg.norm(out_naive - out_ssa) / np.linalg.norm(out_naive)
        errors.append(err)

        speedup = t_naive / t_ssa if t_ssa > 0 else 0
        print(f"{N:<6} | {t_naive:<15.4f} | {t_ssa:<15.4f} | {speedup:<10.2f} | {err:<10.2f}")

    # Plot Runtime
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, times_naive, 'o-', label='Standard Attention O($N^2$)')
    plt.plot(seq_lengths, times_ssa, 's-', label='Spectral Sparse Attention O($N^{1.5}$)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Runtime (seconds)')
    plt.title('Runtime Comparison: Standard vs SSA')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_efficient_ai/runtime_comparison.png')

    # Theoretical FLOPs plot (re-creating the complexity plot from before but better)
    # We use the previous logic for the complexity plot but distinct file
    N_range = np.linspace(100, 5000, 50)
    flops_trans = 4 * N_range**2 * d_model
    flops_mamba = 10 * N_range * d_model * 16 # d_state=16
    flops_ssa = N_range * d_model * (d_model/2) + (N_range**2 / np.sqrt(N_range)) * d_model + N_range * d_model * d_model

    plt.figure(figsize=(10, 6))
    plt.plot(N_range, flops_trans, label='Transformer')
    plt.plot(N_range, flops_mamba, label='Mamba')
    plt.plot(N_range, flops_ssa, label='SSA (Ours)')
    plt.yscale('log')
    plt.xlabel('Sequence Length')
    plt.ylabel('Theoretical FLOPs')
    plt.title('Theoretical Complexity Analysis')
    plt.legend()
    plt.grid(True)
    plt.savefig('energy_efficient_ai/complexity_plot.png')

if __name__ == "__main__":
    run_experiments()
