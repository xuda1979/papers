import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def random_unitary(dim):
    """Generates a random unitary matrix of size dim x dim (Haar measure)."""
    # QR decomposition of a random complex matrix
    Z = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    Q, R = np.linalg.qr(Z)
    # Correct phases to ensure Haar measure
    D = np.diag(np.sign(np.diag(R)))
    U = np.dot(Q, D)
    return U

def simulate_rcs(num_qubits):
    """
    Simulates a Random Quantum Circuit by generating a random unitary U
    and computing output probabilities.
    Ideally, we simulate local gates, but a global random unitary
    gives the perfect Porter-Thomas distribution immediately.
    To be more realistic to RCS, we should simulate layers of gates.
    But for 8 qubits, global unitary is fine to demonstrate the statistics.
    """
    dim = 2**num_qubits
    print(f"Simulating Random Circuit on {num_qubits} qubits (dim={dim})...")

    # 1. Global Haar Random Unitary (Ideal Chaotic Limit)
    # This represents a deep random circuit.
    U = random_unitary(dim)

    # Initial state |0...0>
    psi_0 = np.zeros(dim)
    psi_0[0] = 1.0

    # Evolve
    psi_final = np.dot(U, psi_0)

    # Probabilities
    probs = np.abs(psi_final)**2

    return probs

def plot_porter_thomas(probs, num_qubits):
    dim = 2**num_qubits
    # Normalized probabilities x = N * p
    scaled_probs = dim * probs

    plt.figure(figsize=(8, 6))

    # Plot Histogram
    plt.hist(scaled_probs, bins=50, density=True, alpha=0.6, color='b', label='Simulation')

    # Theoretical Porter-Thomas: P(x) = e^{-x}
    x_vals = np.linspace(0, max(scaled_probs), 100)
    y_vals = np.exp(-x_vals)

    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Porter-Thomas $e^{-Np}$')

    plt.xlabel('Scaled Probability $Np$')
    plt.ylabel('Frequency')
    plt.title(f'Porter-Thomas Distribution (RCS Statistics) for N={num_qubits} Qubits')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "porter_thomas.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # N=10 -> 1024 states. Good statistics.
    # N=12 -> 4096 states. Even better.
    N = 12
    probabilities = simulate_rcs(N)
    plot_porter_thomas(probabilities, N)
