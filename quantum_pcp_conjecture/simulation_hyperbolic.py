import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import time
import os
import networkx as nx

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def tensor_product(ops):
    """Computes the tensor product of a list of sparse matrices."""
    res = ops[0]
    for op in ops[1:]:
        res = sp.kron(res, op, format='csr')
    return res

class CodeGraph:
    def __init__(self):
        self.faces = [] # List of list of edge indices
        self.vertices = [] # List of list of edge indices
        self.edges = [] # List of pairs (u, v) - abstract
        self.num_edges = 0

    def add_face(self, edge_list):
        self.faces.append(edge_list)

    def add_vertex(self, edge_list):
        self.vertices.append(edge_list)

def generate_hyperbolic_tiling(p, q):
    """
    Generates a small patch of a hyperbolic {p, q} tiling.
    Specifically, constructs a central p-gon and its q neighbors.
    This is sufficient for small-scale entanglement verification (approx 16 qubits).
    Returns a CodeGraph object.
    """
    cg = CodeGraph()

    # We manually construct a {5,4} patch (Pentagons, 4 meeting at a vertex).
    # Structure:
    # 1 Central Pentagon (Face 0).
    # 4 Neighboring Pentagons (Faces 1-4) attached to its edges.
    # Note: In {5,4}, 4 faces meet at a vertex. This patch satisfies local constraints
    # for the inner vertices, though boundary vertices will have lower degree.

    # Vertices of the graph (not the code check vertices, but graph nodes):
    # Center face vertices: 0, 1, 2, 3, 4
    # Outer vertices layer: 5..12

    # Edges are the qubits.
    edge_map = {}
    next_edge_idx = 0

    def get_edge(u, v):
        nonlocal next_edge_idx
        pair = tuple(sorted((u, v)))
        if pair not in edge_map:
            edge_map[pair] = next_edge_idx
            next_edge_idx += 1
        return edge_map[pair]

    # Define Faces by their vertex sequences
    # F0: Center
    f0_verts = [0, 1, 2, 3, 4] # Not used as a check in this patch logic?
    # Actually, let's define the faces surrounding a vertex to ensure vertex degree 4.

    # Alternative Construction:
    # F1: 0-1-5-6-2-0
    f1_verts = [0, 1, 5, 6, 2]
    # F2: 0-2-7-8-3-0
    f2_verts = [0, 2, 7, 8, 3]
    # F3: 0-3-9-10-4-0
    f3_verts = [0, 3, 9, 10, 4]
    # F4: 0-4-11-12-1-0
    f4_verts = [0, 4, 11, 12, 1]

    # Note: This creates a central vertex 0 with degree 4 (edges to 1, 2, 3, 4).
    # Vertices 1, 2, 3, 4 have degree 3.
    # Outer vertices have degree 2.

    faces_verts = [f1_verts, f2_verts, f3_verts, f4_verts]

    # Register Faces (Z-checks)
    for fv in faces_verts:
        e_list = []
        for i in range(len(fv)):
            u, v = fv[i], fv[(i+1)%len(fv)]
            e_list.append(get_edge(u, v))
        cg.add_face(e_list)

    # Register Vertices (X-checks)
    # Based on the edges created above
    vertex_edges = {i: [] for i in range(13)}
    for (u, v), e_idx in edge_map.items():
        vertex_edges[u].append(e_idx)
        vertex_edges[v].append(e_idx)

    for i in range(13):
        cg.add_vertex(vertex_edges[i])

    cg.num_edges = next_edge_idx
    return cg

def build_hamiltonian(cg):
    num_qubits = cg.num_edges
    dim = 2 ** num_qubits

    I = sp.eye(2, format='csr')
    X = sp.csr_matrix([[0, 1], [1, 0]])
    Z = sp.csr_matrix([[1, 0], [0, -1]])

    H = sp.csr_matrix((dim, dim))

    # Vertex operators (Star) - X type
    for v_edges in cg.vertices:
        ops = [I] * num_qubits
        for e in v_edges:
            ops[e] = X
        Av = tensor_product(ops)
        H -= Av

    # Face operators (Plaquette) - Z type
    for f_edges in cg.faces:
        ops = [I] * num_qubits
        for e in f_edges:
            ops[e] = Z
        Bp = tensor_product(ops)
        H -= Bp

    return H

def run_simulation():
    print("Generating Hyperbolic Patch {5,4}...")
    cg = generate_hyperbolic_tiling(5, 4)
    print(f"Graph generated: {len(cg.vertices)} vertices, {len(cg.faces)} faces, {cg.num_edges} edges (qubits).")

    if cg.num_edges > 20:
        print("Warning: Too many qubits for dense simulation.")
        return

    print("Building Hamiltonian...")
    start = time.time()
    H = build_hamiltonian(cg)
    print(f"Hamiltonian built in {time.time() - start:.2f}s")

    print("Finding Ground State...")
    start = time.time()
    eigvals, eigvecs = eigsh(H, k=1, which='SA')
    ground_state = eigvecs[:, 0]
    print(f"Ground State Energy: {eigvals[0]:.4f} (Time: {time.time() - start:.2f}s)")

    subsystem_sizes = range(1, cg.num_edges // 2 + 1)
    entropies = []

    print("Calculating Entanglement Entropy...")
    for k in subsystem_sizes:
        indices = list(range(k))
        rho = get_partial_trace(ground_state, indices)
        S = von_neumann_entropy(rho)
        entropies.append(S)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(subsystem_sizes, entropies, 's-', label='Hyperbolic {5,4} Patch')
    plt.xlabel('Subsystem Size')
    plt.ylabel('Entanglement Entropy')
    plt.title(f'Entropy in Hyperbolic Code Patch ({cg.num_edges} qubits)')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "hyperbolic_entropy.png"))
    print("Saved plot to hyperbolic_entropy.png")

# --- Helpers ---
def get_partial_trace(rho_vec, keep_indices):
    N = int(np.log2(len(rho_vec)))
    all_indices = set(range(N))
    keep_set = set(keep_indices)
    trace_indices = list(all_indices - keep_set)
    shape = [2] * N
    psi_tensor = rho_vec.reshape(shape)
    perm = list(keep_indices) + trace_indices
    psi_permuted = np.transpose(psi_tensor, perm)
    dim_keep = 2 ** len(keep_indices)
    dim_trace = 2 ** len(trace_indices)
    psi_mat = psi_permuted.reshape((dim_keep, dim_trace))
    rho_A = np.dot(psi_mat, psi_mat.conj().T)
    return rho_A

def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]
    return -np.sum(eigenvalues * np.log(eigenvalues))

if __name__ == "__main__":
    run_simulation()
