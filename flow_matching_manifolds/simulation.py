import numpy as np
import matplotlib.pyplot as plt
import os

# Ensure we save to the correct directory relative to the script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def exp_map_S2(x, v):
    """Exponential map on S2: x \in S2, v \in T_x S2."""
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    # Avoid division by zero
    v_norm = np.maximum(v_norm, 1e-8)
    return x * np.cos(v_norm) + (v / v_norm) * np.sin(v_norm)

def log_map_S2(x, y):
    """Logarithm map on S2."""
    dot = np.sum(x * y, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    # Direction
    dir_vec = y - x * dot
    dir_norm = np.linalg.norm(dir_vec, axis=-1, keepdims=True)

    # Handle antipodal/close points
    mask = (dir_norm > 1e-8)
    v = np.zeros_like(x)

    # Use np.where for safe indexing or mask manually
    # Note: mask shape is (N, 1), v and dir_vec are (N, 3)
    # We broadcast the mask
    safe_mask = np.broadcast_to(mask, v.shape)

    if np.any(mask):
        # We need to be careful with broadcasting.
        # theta[mask] selects elements where mask is true, flattening them.
        # But we want to preserve structure or just do element-wise multiplication with zeroing.

        factor = np.zeros_like(theta)
        np.divide(theta, dir_norm, out=factor, where=mask)
        v = factor * dir_vec

    return v

def sample_wrapped_normal(mu, sigma, n_samples):
    """Sample from a wrapped normal distribution on S2."""
    # Tangent space samples
    v = np.random.normal(0, sigma, (n_samples, 3))
    # Project to tangent space of mu (ensure orthogonality)
    v = v - np.sum(v * mu, axis=1, keepdims=True) * mu
    return exp_map_S2(mu, v)

def simulate_flow():
    np.random.seed(42)

    # 1. Define Target Distribution (3 modes)
    mu1 = np.array([1.0, 0.0, 0.0])
    mu2 = np.array([-0.5, 0.866, 0.0])
    mu3 = np.array([-0.5, -0.866, 0.0])

    n_samples = 300
    samples1 = sample_wrapped_normal(mu1, 0.4, n_samples)
    samples2 = sample_wrapped_normal(mu2, 0.4, n_samples)
    samples3 = sample_wrapped_normal(mu3, 0.4, n_samples)

    x1_data = np.vstack([samples1, samples2, samples3])

    # 2. Source Distribution (Uniform on S2)
    # Rejection sampling for uniform sphere
    x0_data = np.random.randn(x1_data.shape[0], 3)
    x0_data /= np.linalg.norm(x0_data, axis=1, keepdims=True)

    # 3. Compute Trajectories (Geodesic Interpolation)
    ts = np.linspace(0, 1, 10)
    trajectories = []

    # We plot a subset of trajectories to avoid clutter
    subset_indices = np.random.choice(len(x1_data), 50, replace=False)

    for idx in subset_indices:
        p0 = x0_data[idx]
        p1 = x1_data[idx]
        v = log_map_S2(p0, p1)
        path = []
        for t in ts:
            pt = exp_map_S2(p0, t * v)
            path.append(pt)
        trajectories.append(np.array(path))

    # 4. Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw sphere wireframe
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.1)

    # Plot trajectories
    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='blue', alpha=0.3)

    # Plot target samples
    ax.scatter(x1_data[:, 0], x1_data[:, 1], x1_data[:, 2], color='red', s=5, label='Target Data')
    ax.scatter(x0_data[subset_indices, 0], x0_data[subset_indices, 1], x0_data[subset_indices, 2], color='green', s=10, label='Source Samples')

    ax.set_title("Riemannian Flow Matching on S2\n(Geodesic Interpolation)")
    ax.legend()
    ax.axis('off')

    output_path = os.path.join(OUTPUT_DIR, 'spherical_flow.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Simulation visualization saved to {output_path}")

if __name__ == "__main__":
    simulate_flow()
