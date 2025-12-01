import numpy as np
import matplotlib.pyplot as plt
import os

def generate_gue(n):
    """Generate a random Hermitian matrix from the Gaussian Unitary Ensemble."""
    A = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    H = (A + A.conj().T) / 2
    return H

def generate_ginibre(n):
    """Generate a random non-Hermitian matrix (Ginibre Ensemble)."""
    return np.random.randn(n, n) + 1j * np.random.randn(n, n)

def simulate_flow(n=50, t_steps=100, t_max=5.0):
    """
    Simulate the RG flow of the Hamiltonian:
    H(t) = H_0 + e^{-t} * V
    where H_0 is Hermitian (GUE) and V is non-Hermitian (Ginibre).
    """
    np.random.seed(42)
    H0 = generate_gue(n)
    V = generate_ginibre(n)

    # Normalize to comparable scales
    H0 = H0 / np.sqrt(n)
    V = V / np.sqrt(n)

    times = np.linspace(0, t_max, t_steps)
    eigenvalue_history = []
    order_parameter = []

    for t in times:
        coupling = np.exp(-t)
        H_t = H0 + coupling * V
        evals = np.linalg.eigvals(H_t)
        eigenvalue_history.append(evals)

        # Order parameter: sum of squared imaginary parts (measure of symmetry breaking)
        # We shift eigenvalues so the "real axis" is the target.
        # Actually GUE eigenvalues are real. The non-Hermitian perturbation moves them off-axis.
        off_axis = np.sum(np.abs(evals.imag)**2)
        order_parameter.append(off_axis)

    return times, eigenvalue_history, order_parameter

def plot_results(times, ev_history, order_param):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot 1: Eigenvalue flow in complex plane
    plt.figure(figsize=(10, 8))

    # Plot start (t=0, red) and end (t=max, blue)
    ev_start = ev_history[0]
    ev_end = ev_history[-1]

    plt.scatter(ev_start.real, ev_start.imag, c='red', alpha=0.5, label='UV (t=0)')
    plt.scatter(ev_end.real, ev_end.imag, c='blue', alpha=0.5, label='IR (t=max)')

    # Plot trajectories for a few eigenvalues
    n_evals = len(ev_start)
    # Transpose history to get trajectories: shape (steps, n) -> (n, steps)
    # Sorting eigenvalues is tricky due to continuity, but for visualization we can try just raw scatter
    # or simple matching. For now, just scatter clouds.

    plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
    plt.title('Eigenvalue Flow under RG (Symmetry Restoration)')
    plt.xlabel('Re(E)')
    plt.ylabel('Im(E)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'eigenvalue_flow.png'))
    plt.close()

    # Plot 2: Order parameter vs time
    plt.figure(figsize=(8, 6))
    plt.plot(times, order_param, linewidth=2)
    plt.title('Symmetry Restoration Order Parameter')
    plt.xlabel('Arithmetic Time t')
    plt.ylabel(r'$\sum |Im(E_i)|^2$')
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'symmetry_restoration.png'))
    plt.close()

    print(f"Plots saved to {output_dir}")

if __name__ == "__main__":
    times, ev_hist, ord_param = simulate_flow()
    plot_results(times, ev_hist, ord_param)
