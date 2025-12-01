import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time
import os

# Ensure output directory exists
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def tensor_product(ops):
    """Computes the tensor product of a list of sparse matrices."""
    res = ops[0]
    for op in ops[1:]:
        res = sp.kron(res, op, format='csr')
    return res

def build_toric_code_hamiltonian(L):
    """
    Constructs the Toric Code Hamiltonian on an L x L lattice.
    """
    num_qubits = 2 * L * L
    dim = 2 ** num_qubits

    # Pauli matrices
    I = sp.eye(2, format='csr')
    X = sp.csr_matrix([[0, 1], [1, 0]])
    Z = sp.csr_matrix([[1, 0], [0, -1]])

    H = sp.csr_matrix((dim, dim))

    # Star Operators (Vertices)
    for x in range(L):
        for y in range(L):
            e1 = 2 * (x * L + y)
            e2 = 2 * (x * L + y) + 1
            e3 = 2 * (x * L + (y - 1) % L)
            e4 = 2 * (((x - 1) % L) * L + y) + 1
            edges = [e1, e2, e3, e4]
            ops = [I] * num_qubits
            for e in edges:
                ops[e] = X
            H -= tensor_product(ops)

    # Plaquette Operators (Faces)
    for x in range(L):
        for y in range(L):
            e1 = 2 * (x * L + y)
            e2 = 2 * (x * L + y) + 1
            e3 = 2 * (x * L + (y + 1) % L) + 1
            e4 = 2 * (((x + 1) % L) * L + y)
            edges = [e1, e2, e3, e4]
            ops = [I] * num_qubits
            for e in edges:
                ops[e] = Z
            H -= tensor_product(ops)

    return H

def get_partial_trace(rho_vec, keep_indices):
    """Computes the partial trace of a pure state vector |psi>."""
    N = int(np.log2(len(rho_vec)))
    keep_indices = list(keep_indices) # ensure list

    if len(keep_indices) == 0:
        return np.array([[1.0]])
    if len(keep_indices) >= N:
        return np.outer(rho_vec, rho_vec.conj())

    if len(keep_indices) > 13:
        print(f"Skipping region with {len(keep_indices)} qubits (too large for dense matrix).")
        return None

    all_indices = set(range(N))
    keep_set = set(keep_indices)
    trace_indices = list(all_indices - keep_set)

    shape = [2] * N
    psi_tensor = rho_vec.reshape(shape)

    perm = keep_indices + trace_indices
    psi_permuted = np.transpose(psi_tensor, perm)

    dim_keep = 2 ** len(keep_indices)
    dim_trace = 2 ** len(trace_indices)

    psi_mat = psi_permuted.reshape((dim_keep, dim_trace))
    rho_A = np.dot(psi_mat, psi_mat.conj().T)
    return rho_A

def von_neumann_entropy(rho):
    if rho is None: return 0
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

def get_edges_rect(L, w, h):
    """
    Returns indices of edges within a w x h rectangular block of plaquettes.
    Includes boundary edges of the block.
    """
    edges = set()
    for x in range(w):
        for y in range(h):
            e1 = 2 * (x * L + y)
            e2 = 2 * (x * L + y) + 1
            e3 = 2 * (x * L + (y + 1) % L) + 1
            e4 = 2 * (((x + 1) % L) * L + y)
            edges.add(e1)
            edges.add(e2)
            edges.add(e3)
            edges.add(e4)
    return list(edges)

def simulate_toric_code(L=3):
    print(f"Building Toric Code Hamiltonian for L={L} ({2*L**2} qubits)...")
    H = build_toric_code_hamiltonian(L)

    print("Finding Ground State...")
    eigvals, eigvecs = eigsh(H, k=1, which='SA')
    ground_state = eigvecs[:, 0]

    regions = []
    # Only use regions with size < N/2 (9 qubits) to avoid finite size effects
    # 1x1: 4 qubits. P=4.
    # 1x2: 7 qubits. P=6.
    regions.append((1, 1))
    regions.append((1, 2))
    regions.append((2, 1))

    perimeters = []
    entropies = []

    print("\nCalculating Entropies...")
    for w, h in regions:
        if w > L or h > L: continue

        edges = get_edges_rect(L, w, h)
        perimeter = 2*w + 2*h

        rho = get_partial_trace(ground_state, edges)
        if rho is None: continue

        S = von_neumann_entropy(rho)

        print(f"Region {w}x{h} (Perim {perimeter}, Qubits {len(edges)}): S = {S:.4f}")
        perimeters.append(perimeter)
        entropies.append(S)

    return perimeters, entropies

if __name__ == "__main__":
    L = 3
    perimeters, entropies = simulate_toric_code(L)

    if len(perimeters) > 1:
        coeffs = np.polyfit(perimeters, entropies, 1)
        alpha = coeffs[0]
        gamma = -coeffs[1]

        print(f"\nFit Results:")
        print(f"Slope (alpha): {alpha:.4f}")
        print(f"Intercept (-gamma): {coeffs[1]:.4f} => gamma = {gamma:.4f}")

        plt.figure(figsize=(8, 6))
        plt.plot(perimeters, entropies, 'bo', label='Simulation Data')

        x_range = np.linspace(min(perimeters)-1, max(perimeters)+1, 100)
        y_fit = alpha * x_range - gamma
        plt.plot(x_range, y_fit, 'r--', label=f'Fit: S = {alpha:.2f}L - {gamma:.2f}')

        plt.xlabel('Boundary Length L (Perimeter)')
        plt.ylabel('Entanglement Entropy S')
        # We manually label gamma as ln(2) for illustration if fit is noisy
        plt.title(f'Toric Code Area Law (L={L})')
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_path = os.path.join(OUTPUT_DIR, "toric_entropy.png")
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        print("Not enough data points for fit.")
