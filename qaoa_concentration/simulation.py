import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def build_maxcut_hamiltonian_diagonal(G):
    """
    Returns the diagonal of the MaxCut Hamiltonian H_C for graph G.
    H_C = 0.5 * sum_{(u,v) in E} (I - Z_u Z_v)
    The eigenvalues are simply the cut values (number of edges crossing the cut).
    We return just the array of cut values for all 2^n states.
    """
    n = len(G.nodes)
    N = 2**n
    diagonal = np.zeros(N)

    # Iterate over all basis states |x> where x is an integer 0..2^n-1
    # This is inefficient (O(2^n * E)) but fine for small simulation (n<=12)
    for x in range(N):
        cut_value = 0
        # Convert integer x to bit string (nodes in partition 0 or 1)
        x_bits = [(x >> i) & 1 for i in range(n)]

        for u, v in G.edges:
            if x_bits[u] != x_bits[v]:
                cut_value += 1
        diagonal[x] = cut_value

    return diagonal

def qaoa_p1_energy(G, gamma, beta):
    """
    Calculates the expectation value <H_C> for QAOA with p=1 depth.
    Formula derived analytically for p=1 on MaxCut is:
    <H_C> = 0.5 * sum_{(u,v)} (1 + sin(4*beta) * sin(gamma) * (prod_{w in N(u) - {v}} cos(gamma) + prod_{w in N(v) - {u}} cos(gamma)))
    ...Actually, implementing the exact unitary simulation is safer and more general for verifying.
    """
    n = len(G.nodes)
    N = 2**n

    # Initial state |+>
    psi = np.ones(N) / np.sqrt(N)

    # H_C diagonal
    # H_C = sum C_{uv} -> We want to maximize C.
    # Usually Hamiltonian is MINIMIZED. So H = -C.
    # But let's stick to maximizing the Cut C.
    # The evolution is e^{-i gamma H_C}. If H_C is the cut operator (positive eigenvalues), we maximize <H_C>.
    # Wait, usually standard is e^{-i gamma H_C}.

    # Let's use the explicit diagonal calculated.
    diag_HC = build_maxcut_hamiltonian_diagonal(G)

    # Apply U_C(gamma) = e^{-i gamma H_C}
    psi = np.exp(-1j * gamma * diag_HC) * psi

    # Apply U_B(beta) = e^{-i beta H_B} where H_B = sum X_i
    # This is equivalent to applying R_x(2*beta) on each qubit.
    # R_x(theta) = [cos(t/2) -isin(t/2); -isin(t/2) cos(t/2)]
    # This tensor product is expensive to build.
    # Faster way: Apply H, then e^{-i beta Z}, then H.

    # Apply Hadamard to all
    # For simulation, let's just use a matrix-free approach or small n optimized loop?
    # For n=10, 2^10=1024, dense matrix is fine.

    # Construct H_B matrix (sum of X_i)
    # Actually, simpler: U_B = (e^{-i beta X})^{\otimes n}
    # e^{-i beta X} = [[cos(beta), -i sin(beta)], [-i sin(beta), cos(beta)]]
    c = np.cos(beta)
    s = -1j * np.sin(beta)
    u_local = np.array([[c, s], [s, c]])

    # Apply tensor product of u_local
    # We can do this iteratively
    psi = psi.reshape([2]*n)
    for i in range(n):
        # Apply u_local to dimension i
        # tensordot is tricky, let's use einsum or swapaxes
        # Move axis i to 0
        psi = np.moveaxis(psi, i, 0)
        # Contract
        # new_psi[j, ...] = sum_k u_local[j, k] * old_psi[k, ...]
        psi = np.tensordot(u_local, psi, axes=(1, 0))
        # Move axis 0 back to i
        psi = np.moveaxis(psi, 0, i)

    psi = psi.flatten()

    # Calculate expectation value <psi|H_C|psi>
    # <psi|H_C|psi> = sum_x |psi_x|^2 * diag_HC[x]
    expectation = np.sum(np.abs(psi)**2 * diag_HC)
    return expectation.real

def run_simulation():
    # Parameters
    gammas = np.linspace(0, np.pi, 20)
    betas = np.linspace(0, np.pi/2, 20)
    GG, BB = np.meshgrid(gammas, betas)

    n = 8

    # Generate 3 random 3-regular graphs
    graphs = []
    for _ in range(3):
        graphs.append(nx.random_regular_graph(3, n))

    fig = plt.figure(figsize=(15, 4))

    for idx, G in enumerate(graphs):
        landscape = np.zeros_like(GG)
        for i in range(GG.shape[0]):
            for j in range(GG.shape[1]):
                landscape[i, j] = qaoa_p1_energy(G, GG[i, j], BB[i, j])

        ax = fig.add_subplot(1, 3, idx+1)
        im = ax.imshow(landscape, extent=[0, np.pi, 0, np.pi/2], origin='lower', aspect='auto')
        ax.set_title(f'Graph {idx+1} (n={n})')
        ax.set_xlabel('Gamma')
        ax.set_ylabel('Beta')
        plt.colorbar(im, ax=ax)

    plt.suptitle('QAOA Energy Landscape Concentration (Similar Peaks across Instances)')
    plt.savefig('qaoa_concentration/landscape.png')
    print("Plot saved to qaoa_concentration/landscape.png")

if __name__ == "__main__":
    run_simulation()
