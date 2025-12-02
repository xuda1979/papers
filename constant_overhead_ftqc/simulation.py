import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import os

def create_cyclic_matrix(n, weight):
    """
    Creates a random cyclic matrix of size n x n with given row/col weight.
    This mimics a polynomial multiplication in the group ring F_2[Z_n].
    """
    # Create the first row with 'weight' ones
    indices = np.random.choice(n, weight, replace=False)
    row = np.zeros(n, dtype=int)
    row[indices] = 1

    # Create cyclic shifts
    rows = []
    for i in range(n):
        rows.append(np.roll(row, i))

    return np.array(rows)

def construct_lifted_product(n, weight):
    """
    Constructs a simplified Lifted Product Code from two cyclic matrices A and B.
    H_X = A \otimes I + I \otimes B^T
    H_Z = B \otimes I + I \otimes A^T

    Note: Real LP codes use a slightly more complex construction with A and B being
    matrices over the ring, but this captures the sparsity and structure.
    For rigorous LP, we need A and B to be members of a chain complex.

    Here we use a simplified version: Hypergraph Product Code structure which is a specific instance.
    Let H1, H2 be classical parity check matrices.
    H_X = [ H1 \otimes I_n2, I_n1 \otimes H2^T ]
    H_Z = [ I_n1 \otimes H2, H1^T \otimes I_n2 ]

    This is a CSS code.
    """
    # Create two random classical LDPC matrices
    H1 = create_cyclic_matrix(n, weight)
    H2 = create_cyclic_matrix(n, weight)

    n1, m1 = H1.shape # n x n
    n2, m2 = H2.shape # n x n

    I_n1 = sp.eye(n1, dtype=int)
    I_n2 = sp.eye(n2, dtype=int)

    # Construct blocks using Kronecker product
    # H_X = [ H1 \otimes I_n2 | I_n1 \otimes H2^T ]
    block1 = sp.kron(H1, I_n2)
    block2 = sp.kron(I_n1, H2.T)
    HX = sp.hstack([block1, block2])

    # H_Z = [ I_n1 \otimes H2 | H1^T \otimes I_n2 ]
    block3 = sp.kron(I_n1, H2)
    block4 = sp.kron(H1.T, I_n2)
    HZ = sp.hstack([block3, block4])

    return HX, HZ

def add_noise(codeword, p):
    noise = (np.random.rand(len(codeword)) < p).astype(int)
    return (codeword + noise) % 2, noise

def bp_decode(H, syndrome, max_iter=20, p_channel=0.01):
    """
    Simple Belief Propagation decoder (Log-Domain SPA).
    """
    m, n = H.shape

    # LLRs for channel (0 -> +infinity, 1 -> -infinity)
    # p0 = 1-p, p1 = p. LLR = log((1-p)/p)
    llr_init = np.log((1-p_channel)/p_channel)

    # Variable to Check messages (L_q->c)
    # Initialize with channel LLRs
    L_vc = sp.lil_matrix(H.T * llr_init) # This isn't quite right for sparse, doing dense loop for simplicity

    # Use dense for small codes for simplicity in this script
    H_dense = H.toarray()

    # L_vc[v, c] initialized to channel LLR
    L_vc = np.zeros((n, m))
    rows, cols = H.nonzero()
    for i in range(len(rows)):
        L_vc[cols[i], rows[i]] = llr_init

    L_cv = np.zeros((m, n))

    for iteration in range(max_iter):
        # Check to Variable Step
        for c in range(m):
            # Neighbors of check c
            neighbors = np.where(H_dense[c, :] == 1)[0]
            for v in neighbors:
                # tanh(L_cv/2) = product of tanh(L_vc/2) * (-1)^syndrome
                prod = 1.0
                for v_prime in neighbors:
                    if v_prime == v: continue
                    prod *= np.tanh(L_vc[v_prime, c] / 2.0)

                prod *= (1 - 2*syndrome[c])
                # Avoid numerical issues
                prod = np.clip(prod, -0.999999, 0.999999)
                L_cv[c, v] = 2 * np.arctanh(prod)

        # Variable to Check Step
        for v in range(n):
            neighbors = np.where(H_dense[:, v] == 1)[0]
            sum_llr = llr_init
            for c in neighbors:
                sum_llr += L_cv[c, v]

            # Update messages
            for c in neighbors:
                L_vc[v, c] = sum_llr - L_cv[c, v]

    # Hard decision
    correction = np.zeros(n, dtype=int)
    for v in range(n):
        neighbors = np.where(H_dense[:, v] == 1)[0]
        total_llr = llr_init + np.sum(L_cv[neighbors, v])
        if total_llr < 0:
            correction[v] = 1

    return correction

def simulate_ssec():
    print("Constructing Code...")
    n_base = 15 # Size of base cyclic code
    weight = 3
    HX, HZ = construct_lifted_product(n_base, weight)

    n_qubits = HX.shape[1]
    n_checks_X = HX.shape[0]

    print(f"Code parameters: n={n_qubits}, checks={n_checks_X}")

    p_phys_range = np.linspace(0.001, 0.05, 10)
    p_meas_range = np.linspace(0.001, 0.05, 10)

    residual_weights = []

    print("Starting Simulation...")
    # Fix physical error rate, vary measurement error rate
    p_phys = 0.01

    results = []

    for p_meas in p_meas_range:
        avg_residual = 0
        trials = 20
        for _ in range(trials):
            # Generate random error
            noise_data = (np.random.rand(n_qubits) < p_phys).astype(int)

            # Calculate perfect syndrome
            syndrome_perfect = HX @ noise_data % 2

            # Add measurement noise to syndrome
            noise_meas = (np.random.rand(n_checks_X) < p_meas).astype(int)
            syndrome_noisy = (syndrome_perfect + noise_meas) % 2

            # Decode using noisy syndrome
            # Note: Standard BP assumes perfect syndrome.
            # For SSEC, we treat syndrome bits as check nodes that can be satisfied or not?
            # Actually, robust BP or just treating it as a standard decoding problem works for demonstration.
            # If the decoder is good, it finds a correction 'c' such that HX*c is close to 'syndrome_noisy'.
            # Ideally we want c ~ noise_data.

            correction = bp_decode(HX, syndrome_noisy, max_iter=10, p_channel=p_phys)

            # Residual error: The actual error remaining on the state
            # e_residual = e_data + correction
            e_residual = (noise_data + correction) % 2

            avg_residual += np.sum(e_residual)

        avg_residual /= trials
        results.append(avg_residual)
        print(f"p_meas={p_meas:.3f}, Residual Weight={avg_residual:.2f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(p_meas_range, results, 'o-', linewidth=2, label=f'Physical Error p={p_phys}')
    plt.xlabel('Measurement Error Rate ($p_{meas}$)')
    plt.ylabel('Residual Data Error Weight')
    plt.title('Single-Shot Stability: Residual Error vs Measurement Noise')
    plt.grid(True)
    plt.legend()
    plt.savefig('constant_overhead_ftqc/ssec_stability.png')
    print("Plot saved to constant_overhead_ftqc/ssec_stability.png")

if __name__ == "__main__":
    simulate_ssec()
