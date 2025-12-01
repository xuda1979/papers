import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# Ensure we save to the correct directory relative to the script
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# Spherical Geometry (S2)
# -----------------------------------------------------------------------------

def exp_map_S2(x, v):
    """Exponential map on S2: x \in S2, v \in T_x S2."""
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.maximum(v_norm, 1e-8)
    return x * np.cos(v_norm) + (v / v_norm) * np.sin(v_norm)

def log_map_S2(x, y):
    """Logarithm map on S2."""
    dot = np.sum(x * y, axis=-1, keepdims=True)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)

    dir_vec = y - x * dot
    dir_norm = np.linalg.norm(dir_vec, axis=-1, keepdims=True)

    mask = (dir_norm > 1e-8)
    v = np.zeros_like(x)
    safe_mask = np.broadcast_to(mask, v.shape)

    if np.any(mask):
         factor = np.zeros_like(theta)
         np.divide(theta, dir_norm, out=factor, where=mask)
         v = factor * dir_vec

    return v

def project_tangent_S2(x, v):
    """Project vector v onto tangent space at x."""
    return v - np.sum(v * x, axis=-1, keepdims=True) * x

def sample_wrapped_normal_S2(mu, sigma, n_samples):
    v = np.random.normal(0, sigma, (n_samples, 3))
    v = project_tangent_S2(mu, v)
    return exp_map_S2(mu, v)

# -----------------------------------------------------------------------------
# Hyperbolic Geometry (H2 - Poincaré Disk)
# -----------------------------------------------------------------------------

def mobius_add(x, y):
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    y2 = np.sum(y * y, axis=-1, keepdims=True)
    xy = np.sum(x * y, axis=-1, keepdims=True)
    num = (1 + 2*xy + y2)*x + (1 - x2)*y
    den = 1 + 2*xy + x2*y2
    return num / den

def mobius_neg(x):
    return -x

def exp_map_H2(x, v):
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    lambda_x = 2.0 / (1.0 - x2)
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
    v_norm = np.maximum(v_norm, 1e-8)
    coeff = np.tanh(lambda_x * v_norm / 2.0)
    u = coeff * (v / v_norm)
    return mobius_add(x, u)

def log_map_H2(x, y):
    neg_x = mobius_neg(x)
    diff = mobius_add(neg_x, y)
    diff_norm = np.linalg.norm(diff, axis=-1, keepdims=True)
    diff_norm = np.maximum(diff_norm, 1e-8)
    x2 = np.sum(x * x, axis=-1, keepdims=True)
    lambda_x = 2.0 / (1.0 - x2)
    coeff = (2.0 / lambda_x) * np.arctanh(diff_norm)
    return coeff * (diff / diff_norm)

def project_tangent_H2(x, v):
    # Tangent space of Poincare ball is just R^n, but metric is scaled.
    # The vectors are Euclidean vectors. No projection needed for "conformal" model
    # if we represent them in ambient R^n coordinates?
    # Actually, T_x D^n is identified with R^n.
    return v

def sample_wrapped_normal_H2(mu, sigma, n_samples):
    v = np.random.normal(0, sigma, (n_samples, 2))
    return exp_map_H2(mu, v)

# -----------------------------------------------------------------------------
# Stability Analysis (Vector Field Error Sensitivity)
# -----------------------------------------------------------------------------

def simulate_trajectory(geometry, x0, x1, steps=20, error_magnitude=0.0):
    """
    Simulate a particle moving from x0 to x1 under the conditional flow,
    plus a constant error in the vector field.
    """
    dt = 1.0 / steps
    path = [x0]
    xt = x0

    if geometry == 'S2':
        log_map, exp_map, proj = log_map_S2, exp_map_S2, project_tangent_S2
        dim = 3
    else:
        log_map, exp_map, proj = log_map_H2, exp_map_H2, project_tangent_H2
        dim = 2

    # Generate a random error vector field (constant in ambient space for simplicity)
    # For fair comparison, we normalize it relative to the metric?
    # Or just a constant perturbation in tangent space?
    # Let's add a random perturbation at each step

    # We fix a specific random perturbation direction for this trajectory to simulate systematic error
    perturbation_base = np.random.randn(dim)
    perturbation_base /= np.linalg.norm(perturbation_base)

    for i in range(steps):
        t = i * dt
        # True velocity (conditional vector field)
        # u_t(x_t) is the tangent vector of the geodesic from x_t to x1?
        # Ideally: u_t(x) = v such that flow reaches x1 at t=1.
        # Geodesic connecting x0 to x1 is gamma(t). u(gamma(t)) = gamma'(t).
        # If we are off-path at x_t, we should point towards x1 such that we arrive at time 1?
        # The conditional flow usually assumes we follow the geodesic from x0.
        # But here we simulate "vector field matching", so we follow the field.
        # The field at x_t (if we are off track) is usually defined as the velocity of the geodesic
        # starting at x_t targeting x1 *with remaining time*.
        # Velocity v = Log_{x_t}(x_1) / (1-t).

        remaining_time = 1.0 - t
        if remaining_time < 1e-3: remaining_time = 1e-3

        target_v = log_map(xt, x1) / remaining_time

        # Add error
        # In H2, metric is lambda^2. So unit norm in tangent space means euclidean norm 1/lambda.
        # We want error_magnitude to be in terms of Manifold norm.
        # ||e||_g = lambda ||e||_e = error_magnitude
        # So ||e||_e = error_magnitude / lambda

        if geometry == 'H2':
            x2 = np.sum(xt * xt)
            lam = 2.0 / (1.0 - x2)
            pert_scale = error_magnitude / lam
        else:
            pert_scale = error_magnitude # metric is 1 on S2 (embedded)

        noise = perturbation_base * pert_scale
        if geometry == 'S2':
            noise = proj(xt, noise)

        applied_v = target_v + noise

        # Euler step (Exp map)
        xt = exp_map(xt, applied_v * dt)
        path.append(xt)

    return np.array(path)

def compute_endpoint_error(geometry, steps=20, error_mag=0.1):
    if geometry == 'S2':
        x0 = np.random.randn(3); x0/=np.linalg.norm(x0)
        x1 = np.random.randn(3); x1/=np.linalg.norm(x1)
    else:
        x0 = np.random.uniform(-0.5, 0.5, 2)
        x1 = np.random.uniform(-0.5, 0.5, 2)

    # Trajectory with error
    traj = simulate_trajectory(geometry, x0, x1, steps, error_mag)
    final_x = traj[-1]

    # Distance to true target x1
    if geometry == 'S2':
        dist = np.arccos(np.clip(np.dot(final_x, x1), -1, 1))
    else:
        # H2 dist
        neg_x = mobius_neg(final_x)
        diff = mobius_add(neg_x, x1)
        dn = np.linalg.norm(diff)
        dist = 2 * np.arctanh(dn)

    return dist

# -----------------------------------------------------------------------------
# Main Simulation
# -----------------------------------------------------------------------------

def simulate_flow():
    np.random.seed(42)

    # --- Visualization Code (Same as before) ---
    print("Generating visualizations...")
    # ... (omitted for brevity, assume previous visual code works or just focus on stability now)
    # Actually, let's keep the visual code but streamline it

    mu1 = np.array([1.0, 0.0, 0.0])
    mu2 = np.array([-0.5, 0.866, 0.0])
    mu3 = np.array([-0.5, -0.866, 0.0])
    n = 100
    target_S2 = np.vstack([sample_wrapped_normal_S2(m, 0.2, n) for m in [mu1, mu2, mu3]])
    source_S2 = np.random.randn(len(target_S2), 3); source_S2 /= np.linalg.norm(source_S2, axis=1, keepdims=True)

    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    indices = np.random.choice(len(target_S2), 30, replace=False)
    for idx in indices:
        p0, p1 = source_S2[idx], target_S2[idx]
        v_tan = log_map_S2(p0, p1)
        path = np.array([exp_map_S2(p0, t * v_tan) for t in np.linspace(0, 1, 20)])
        ax1.plot(path[:,0], path[:,1], path[:,2], color='blue', alpha=0.3)
    ax1.scatter(target_S2[:,0], target_S2[:,1], target_S2[:,2], c='red', s=5)
    ax1.set_title("Spherical Flow (S2)")
    ax1.axis('off')

    # H2 Visual
    mu_h = [np.array([0,0]), np.array([0.6,0]), np.array([-0.3,0.5])]
    target_H2 = np.vstack([sample_wrapped_normal_H2(m, 0.3, n) for m in mu_h])
    source_H2 = sample_wrapped_normal_H2(np.array([0,0]), 0.8, len(target_H2))

    ax2 = fig.add_subplot(122)
    circle = plt.Circle((0, 0), 1, color='black', fill=False)
    ax2.add_artist(circle)
    indices_h = np.random.choice(len(target_H2), 30, replace=False)
    for idx in indices_h:
        p0, p1 = source_H2[idx], target_H2[idx]
        v_tan = log_map_H2(p0, p1)
        path = np.array([exp_map_H2(p0, t * v_tan) for t in np.linspace(0, 1, 20)])
        ax2.plot(path[:,0], path[:,1], color='blue', alpha=0.3)
    ax2.scatter(target_H2[:,0], target_H2[:,1], c='red', s=5)
    ax2.set_xlim(-1.1, 1.1); ax2.set_ylim(-1.1, 1.1); ax2.set_aspect('equal')
    ax2.set_title("Hyperbolic Flow (H2)")
    ax2.axis('off')
    plt.savefig(os.path.join(OUTPUT_DIR, 'combined_flow.png'), dpi=300)

    # --- Stability Analysis (Error Propagation) ---
    print("Running Stability Analysis...")
    errors_S2 = []
    errors_H2 = []
    error_mag = 0.5  # Significant error to see effects

    for _ in range(500):
        errors_S2.append(compute_endpoint_error('S2', error_mag=error_mag))
        errors_H2.append(compute_endpoint_error('H2', error_mag=error_mag))

    avg_S2 = np.mean(errors_S2)
    avg_H2 = np.mean(errors_H2)

    print(f"Mean Endpoint Error (Input Noise={error_mag}):")
    print(f"  Sphere (K=+1): {avg_S2:.4f}")
    print(f"  Hyperbolic (K=-1): {avg_H2:.4f}")

    plt.figure(figsize=(6, 4))
    plt.hist(errors_S2, bins=30, alpha=0.5, label=f'Sphere (Mean={avg_S2:.2f})', color='blue')
    plt.hist(errors_H2, bins=30, alpha=0.5, label=f'Hyperbolic (Mean={avg_H2:.2f})', color='red')
    plt.xlabel("Endpoint Error (Geodesic Distance)")
    plt.ylabel("Count")
    plt.title("Impact of Vector Field Error on Generation")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, 'stability_analysis.png'), dpi=300)

if __name__ == "__main__":
    simulate_flow()
