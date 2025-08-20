import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def random_state(dim):
    """
    Generates a random pure quantum state vector in a Hilbert space of a given dimension.
    The state is normalized.

    Args:
        dim (int): The dimension of the Hilbert space.

    Returns:
        np.ndarray: A normalized complex vector representing the quantum state.
    """
    vec = np.random.randn(dim) + 1j * np.random.randn(dim)
    vec /= np.linalg.norm(vec)
    return vec


def calculate_entropy(psi, m, qubits):
    """
    Calculates the entanglement entropy of a subsystem.

    For a pure state `psi` of a total system of `qubits`, this function traces out
    `qubits - m` qubits to get the reduced density matrix of the first `m` qubits,
    and then computes its von Neumann entropy.

    Args:
        psi (np.ndarray): The state vector of the total system.
        m (int): The number of qubits in the subsystem of interest.
        qubits (int): The total number of qubits.

    Returns:
        float: The entanglement entropy in nats.
    """
    # Reshape the state vector into a matrix to perform the partial trace
    psi_matrix = psi.reshape((2 ** m, 2 ** (qubits - m)))

    # Compute the reduced density matrix for the subsystem of size m
    rho = psi_matrix @ psi_matrix.conj().T

    # Compute the eigenvalues of the density matrix. Since it's Hermitian, use eigvalsh.
    eigvals = np.linalg.eigvalsh(rho)

    # Filter out zero or near-zero eigenvalues to avoid log(0) errors
    eigvals = eigvals[eigvals > 1e-12]

    # Calculate von Neumann entropy: S = -Tr(rho * log(rho))
    entropy = -np.sum(eigvals * np.log(eigvals))
    return float(entropy)


def run_simulation(qubits, num_runs=10):
    """
    Runs the Page curve simulation multiple times to average the results.

    Args:
        qubits (int): The total number of qubits in the system.
        num_runs (int): The number of random states to average over for smoother results.

    Returns:
        A tuple containing:
        - A list of (m, avg_entropy, std_dev) tuples for the simulation.
        - A list of (m, theoretical_entropy) tuples for the theoretical Page curve.
    """
    dim = 2 ** qubits
    results = {m: [] for m in range(1, qubits + 1)}

    print(f"  Running {num_runs} simulations for {qubits} qubits...")
    for i in range(num_runs):
        psi = random_state(dim)
        for m in range(1, qubits + 1):
            entropy = calculate_entropy(psi, m, qubits)
            results[m].append(entropy)

    avg_results = []
    for m in range(1, qubits + 1):
        entropies = results[m]
        avg_entropy = np.mean(entropies)
        std_dev = np.std(entropies)
        avg_results.append((m, avg_entropy, std_dev))

    # For a random pure state, the theoretical entropy is min(log(d_A), log(d_B))
    theoretical_curve = [(m, min(m, qubits - m) * np.log(2)) for m in range(1, qubits + 1)]

    return avg_results, theoretical_curve


def plot_results(qubits, data, theoretical_curve, output_dir):
    """
    Plots the simulation results against the theoretical Page curve and saves the plot.
    """
    m_vals, avg_entropies, std_devs = zip(*data)
    m_theory, s_theory = zip(*theoretical_curve)

    plt.figure(figsize=(10, 6))
    # Plot simulation results with error bars
    plt.errorbar(m_vals, avg_entropies, yerr=std_devs, fmt='o-', capsize=5, label=f'Simulation (Avg over 10 runs)')
    # Plot theoretical curve
    plt.plot(m_theory, s_theory, 'r--', label='Theoretical Page Curve')

    plt.xlabel("Number of radiated qubits ($m$)")
    plt.ylabel("Entanglement Entropy $S(m)$ (nats)")
    plt.title(f"Page Curve for a System of {qubits} Qubits")
    plt.legend()
    plt.grid(True)

    # Save the plot to the results directory
    plot_file = output_dir / f'page_curve_{qubits}_qubits.pdf'
    plt.savefig(plot_file)
    print(f"  Plot saved to {plot_file}")
    plt.close()


def main():
    """
    Main function to run simulations for different qubit counts,
    save the resulting data, and generate plots.
    """
    np.random.seed(0)  # for reproducibility

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / 'data'
    results_dir = base_dir / 'results'
    data_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    # Run simulations for a range of qubit sizes
    for qubits in [8, 10, 12]:
        print(f"\nStarting simulation for {qubits} qubits...")

        # Run simulation to get averaged data and theoretical curve
        avg_data, theoretical_curve = run_simulation(qubits, num_runs=10)

        # Save the numerical data to a CSV file
        out_file = data_dir / f'page_curve_{qubits}_qubits.csv'
        with out_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['m', 'S_rad_avg', 'S_rad_std'])
            for m, avg, std in avg_data:
                writer.writerow([m, avg, std])
        print(f"  Data written to {out_file}")

        # Generate and save the plot
        plot_results(qubits, avg_data, theoretical_curve, results_dir)

    print("\nSimulations complete.")


if __name__ == '__main__':
    main()
