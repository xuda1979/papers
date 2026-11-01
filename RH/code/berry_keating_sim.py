import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh
import os

def create_bk_hamiltonian(N, L=10.0):
    """
    Creates a discrete approximation of H = (xp + px)/2 on a finite interval [1, L].
    """
    x = np.linspace(1.0, L, N)
    dx = x[1] - x[0]

    # Momentum operator p = -i d/dx (central difference)
    p = np.zeros((N, N), dtype=complex)
    for i in range(N):
        if i + 1 < N:
            p[i, i+1] = -1j
        if i - 1 >= 0:
            p[i, i-1] = 1j
    p /= (2 * dx)

    # Position operator X is diagonal
    X = np.diag(x)

    # H = (X p + p X) / 2
    H = 0.5 * (X @ p + p @ X)

    # Symmetrize
    H = 0.5 * (H + H.conj().T)

    return H

def run_simulation():
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))

    # Riemann Zeros (first few)
    zeros = [14.1347, 21.0220, 25.0108, 30.4248, 32.9350, 37.5862, 40.9187]

    # Let's focus on the "Staircase" N(E) comparison for the largest N
    H = create_bk_hamiltonian(1000, L=100.0)
    evals = eigvalsh(H)
    evals = evals[evals > 10] # Low energy cut-off
    evals = evals[evals < 60]

    plt.step(evals, np.arange(len(evals)), where='post', label='Discrete Berry-Keating')

    # Theoretical N(E) = (E/2pi) * log(E/2pi) - E/2pi + 7/8 (Berry-Keating asymptotic)
    E_vals = np.linspace(10, 60, 100)
    # The smooth part of Riemann counting function
    N_riemann = (E_vals / (2*np.pi)) * np.log(E_vals / (2*np.pi)) - (E_vals / (2*np.pi)) + 7.0/8.0

    # Plot Actual Zeros as vertical lines
    for z in zeros:
        plt.axvline(z, color='r', linestyle='--', alpha=0.5, label='Riemann Zero' if z == zeros[0] else "")

    plt.plot(E_vals, N_riemann, 'k--', label='Riemann Smooth Trend')

    plt.xlabel('Energy E')
    plt.ylabel('Integrated Density of States N(E)')
    plt.title('Spectrum of Discrete Berry-Keating Operator vs Riemann Zeros')
    plt.legend()
    plt.grid(True)

    output_path = os.path.join(data_dir, 'berry_keating_spectrum.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    run_simulation()
