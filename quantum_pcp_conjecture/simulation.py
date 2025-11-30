import numpy as np
from scipy.sparse import csr_matrix, kron, identity
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time

def pauli_x():
    return np.array([[0, 1], [1, 0]])

def pauli_z():
    return np.array([[1, 0], [0, -1]])

def build_hamiltonian(n_spins, J, h):
    """
    Constructs the Hamiltonian for a 1D Transverse Field Ising Model.
    H = -J * sum(Z_i Z_{i+1}) - h * sum(X_i)
    """
    dim = 2 ** n_spins
    H = csr_matrix((dim, dim))

    # Interaction terms -J * Z_i Z_{i+1}
    Z = csr_matrix(pauli_z())
    I = csr_matrix(np.eye(2))

    for i in range(n_spins - 1):
        # Construct operator I x ... x Z x Z x ... x I
        # Z is at index i and i+1
        ops = [I] * n_spins
        ops[i] = Z
        ops[i+1] = Z

        term = ops[0]
        for op in ops[1:]:
            term = kron(term, op, format='csr')

        H -= J * term

    # Transverse field terms -h * X_i
    X = csr_matrix(pauli_x())
    for i in range(n_spins):
        ops = [I] * n_spins
        ops[i] = X

        term = ops[0]
        for op in ops[1:]:
            term = kron(term, op, format='csr')

        H -= h * term

    return H

def simulate_ground_state(max_n=12):
    results = []
    print(f"Simulating 1D TFIM Ground State Energy (J=1.0, h=0.5) up to N={max_n}")
    print(f"{'N':<5} | {'Energy':<10} | {'Time (s)':<10}")
    print("-" * 35)

    for n in range(2, max_n + 1):
        start_time = time.time()
        H = build_hamiltonian(n, J=1.0, h=0.5)

        # We only need the lowest eigenvalue (ground state energy)
        # k=1 for one eigenvalue, which='SA' for Smallest Algebraic
        eigvals, _ = eigsh(H, k=1, which='SA')
        energy = eigvals[0]

        elapsed = time.time() - start_time
        results.append((n, energy, elapsed))
        print(f"{n:<5} | {energy:.4f}     | {elapsed:.4f}")

    return results

if __name__ == "__main__":
    data = simulate_ground_state(max_n=10)

    # Save results to file for inclusion in paper
    with open("quantum_pcp_conjecture/simulation_results.txt", "w") as f:
        f.write("N,GroundStateEnergy,Time\n")
        for n, e, t in data:
            f.write(f"{n},{e},{t}\n")

    # Generate a plot
    ns = [row[0] for row in data]
    energies = [row[1] for row in data]
    times = [row[2] for row in data]

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Number of Spins (N)')
    ax1.set_ylabel('Ground State Energy', color=color)
    ax1.plot(ns, energies, color=color, marker='o')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Computation Time (s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(ns, times, color=color, linestyle='--', marker='x')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('1D TFIM Ground State Calculation Complexity')
    fig.tight_layout()
    plt.savefig('quantum_pcp_conjecture/complexity_plot.png')
    print("Plot saved to quantum_pcp_conjecture/complexity_plot.png")
