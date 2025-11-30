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
    Constructs the Toric Code Hamiltonian on an L x L lattice with periodic boundary conditions.
    Lattice has L*L vertices and L*L plaquettes.
    There are N = 2*L^2 qubits (edges).

    Qubits are indexed as follows:
    For each vertex (x, y) with 0 <= x, y < L:
      - Horizontal edge to the right: index 2*(x*L + y)
      - Vertical edge down:           index 2*(x*L + y) + 1
    """
    num_qubits = 2 * L * L
    dim = 2 ** num_qubits

    # Pauli matrices
    I = sp.eye(2, format='csr')
    X = sp.csr_matrix([[0, 1], [1, 0]])
    Z = sp.csr_matrix([[1, 0], [0, -1]])

    H = sp.csr_matrix((dim, dim))

    # Star Operators (Vertices)
    # A_v = product of X on the 4 edges connected to vertex v
    for x in range(L):
        for y in range(L):
            # Identify the 4 edges connected to vertex (x,y)
            # 1. Horizontal right from (x,y): 2*(x*L + y)
            # 2. Vertical down from (x,y):    2*(x*L + y) + 1
            # 3. Horizontal left: from (x, (y-1)%L) -> right edge
            # 4. Vertical up: from ((x-1)%L, y) -> down edge

            e1 = 2 * (x * L + y)
            e2 = 2 * (x * L + y) + 1
            e3 = 2 * (x * L + (y - 1) % L)
            e4 = 2 * (((x - 1) % L) * L + y) + 1

            edges = [e1, e2, e3, e4]

            # Construct operator
            ops = [I] * num_qubits
            for e in edges:
                ops[e] = X

            Av = tensor_product(ops)
            H -= Av  # Hamiltonian term is -Av

    # Plaquette Operators (Faces)
    # B_p = product of Z on the 4 edges bounding plaquette p
    # Plaquette p is "down-right" from vertex (x,y)
    for x in range(L):
        for y in range(L):
            # Edges bounding the plaquette whose top-left corner is (x,y)
            # 1. Top: Horizontal right from (x,y): 2*(x*L + y)
            # 2. Left: Vertical down from (x,y):   2*(x*L + y) + 1
            # 3. Right: Vertical down from (x, (y+1)%L): 2*(x*L + (y+1)%L) + 1
            # 4. Bottom: Horizontal right from ((x+1)%L, y): 2*(((x+1)%L)*L + y)

            e1 = 2 * (x * L + y)
            e2 = 2 * (x * L + y) + 1
            e3 = 2 * (x * L + (y + 1) % L) + 1
            e4 = 2 * (((x + 1) % L) * L + y)

            edges = [e1, e2, e3, e4]

            # Construct operator
            ops = [I] * num_qubits
            for e in edges:
                ops[e] = Z

            Bp = tensor_product(ops)
            H -= Bp  # Hamiltonian term is -Bp

    return H

def get_partial_trace(rho_vec, keep_indices):
    """
    Computes the partial trace of a pure state vector |psi>.
    Returns the reduced density matrix for the subsystem 'keep_indices'.

    psi: vector of size 2^N
    keep_indices: list of qubit indices to keep
    """
    N = int(np.log2(len(rho_vec)))
    # Calculate trace_out_indices
    all_indices = set(range(N))
    keep_set = set(keep_indices)
    trace_indices = list(all_indices - keep_set)

    # Reshape vector to tensor
    shape = [2] * N
    psi_tensor = rho_vec.reshape(shape)

    # We want to trace out 'trace_indices'.
    # A cleaner way using Einstein summation or tensordot is hard for arbitrary indices.
    # Instead, let's permute axes so 'keep' are first, 'trace' are last.

    perm = list(keep_indices) + trace_indices
    psi_permuted = np.transpose(psi_tensor, perm)

    # Flatten the 'keep' part and the 'trace' part
    dim_keep = 2 ** len(keep_indices)
    dim_trace = 2 ** len(trace_indices)

    psi_mat = psi_permuted.reshape((dim_keep, dim_trace))

    # Reduced density matrix rho_A = psi_mat * psi_mat^dagger
    rho_A = np.dot(psi_mat, psi_mat.conj().T)

    return rho_A

def von_neumann_entropy(rho):
    """Computes -Tr(rho log rho)."""
    # Eigendecomposition
    eigenvalues = np.linalg.eigvalsh(rho)
    # Filter small/negative values due to precision
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

def simulate_toric_code(L=2):
    print(f"Building Toric Code Hamiltonian for L={L} ({2*L**2} qubits)...")
    start = time.time()
    H = build_toric_code_hamiltonian(L)
    print(f"Hamiltonian built in {time.time() - start:.2f}s")

    print("Finding Ground State...")
    start = time.time()
    # Find ground state (lowest eigenvalue)
    # Note: Ground state is 4-fold degenerate on torus.
    # eigsh finds one of them (or a superposition).
    eigvals, eigvecs = eigsh(H, k=1, which='SA')
    ground_state = eigvecs[:, 0]
    ground_energy = eigvals[0]
    print(f"Ground State found in {time.time() - start:.2f}s. Energy: {ground_energy:.4f}")

    # Theoretical Ground State Energy
    # Sum of -1 for each vertex and plaquette. Num vertices = L^2, Num plaquettes = L^2.
    # Total terms = 2*L^2. So energy should be -(2*L^2).
    print(f"Theoretical Ground Energy: {-2*L*L}")

    # --- Entanglement Entropy Calculation ---
    # We select regions of increasing size.
    # Let's define a rectangular region A of size w x h (in terms of plaquettes/edges)
    # The geometry of edges is complex. Let's just pick a contiguous block of edges.

    # Visualizing indices:
    # 0 --(0)-- 1
    # |         |
    # (1)       (3)
    # |         |
    # 2 --(2)-- 3

    # Let's verify the Area Law and TEE (Topological Entanglement Entropy)
    # S(A) = alpha * |dA| - gamma
    # gamma = ln(2) for Z2 toric code.

    results = []

    # We define regions by including all qubits (edges) within a w x w block of 'vertices'.
    # This is slightly tricky. Let's just pick subsets of qubits.
    # A simpler partition: Top half vs Bottom half.
    # Or block cuts.

    print("\nCalculating Entanglement Entropy for subregions...")

    # Define a set of region sizes (by number of qubits)
    # We try to approximate square cuts.
    # For L=2 (8 qubits), regions: 1 qubit, 2 qubits, 4 (half).
    # For L=3 (18 qubits), regions up to half.

    # Let's create 'square-like' regions of edges.
    # Region 1: Edges around 1 plaquette.
    # Region 2: Edges around 2 adjacent plaquettes.

    # For simplicity, we just take the first k qubits.
    # This might not be a "geometric" area law if the numbering isn't locality-preserving.
    # But our numbering IS row-major.
    # (0,0) -> edges 0, 1. (0,1) -> edges 2, 3.
    # So taking first k qubits corresponds roughly to filling the lattice row by row.
    # The boundary size will fluctuate.

    # Better: explicit geometric regions.
    # Let's do Cuts along the torus.
    # Cut 1: A strip of width x.

    subsystem_sizes = range(1, 2*L**2 // 2 + 1)
    entropies = []

    for k in subsystem_sizes:
        # Just take first k qubits
        indices = list(range(k))
        rho = get_partial_trace(ground_state, indices)
        S = von_neumann_entropy(rho)
        entropies.append(S)
        # print(f"Size {k}: S = {S:.4f}")

    return subsystem_sizes, entropies

if __name__ == "__main__":
    # L=3 -> 18 qubits. Matrix size 2^18 = 262,144. Feasible.
    L = 3
    sizes, entropies = simulate_toric_code(L)

    # Theoretical prediction:
    # For a simple cut across the torus (cylindrical entropy), S ~ 2*L for a full cut?
    # Actually, for random cuts it's messy.
    # But we should see some structure.
    # The key is that S > 0, indicating entanglement.

    # Save Results
    with open(os.path.join(OUTPUT_DIR, "entropy_data.txt"), "w") as f:
        f.write("SubsystemSize,Entropy\n")
        for s, e in zip(sizes, entropies):
            f.write(f"{s},{e}\n")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sizes, entropies, 'o-', label='Ground State Entropy')
    plt.xlabel('Subsystem Size (Number of Qubits)')
    plt.ylabel('Von Neumann Entropy S(rho_A)')
    plt.title(f'Entanglement Entropy in Toric Code Ground State (L={L}, 18 qubits)')
    plt.grid(True)
    plt.axhline(y=np.log(2), color='r', linestyle='--', label='ln(2)')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "toric_entropy.png"))
    print("Plot saved to toric_entropy.png")
