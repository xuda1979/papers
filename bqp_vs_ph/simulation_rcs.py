import numpy as np
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

def random_su2():
    """Generates a random 2x2 unitary matrix (Haar measure on SU(2))."""
    # QR decomposition of a random complex matrix
    Z = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
    Q, R = np.linalg.qr(Z)
    # Correct phases to ensure Haar measure
    D = np.diag(np.sign(np.diag(R)))
    U = np.dot(Q, D)
    return U

def apply_single_qubit_layer(state, num_qubits):
    """Applies random SU(2) gates to all qubits."""
    # Reshape state to tensor
    tensor_shape = [2] * num_qubits
    state_tensor = state.reshape(tensor_shape)

    for i in range(num_qubits):
        U = random_su2()
        # Apply U to the i-th axis
        # tensordot: axes=1 contracts the last axis of U with the i-th axis of state
        # We need to be careful with indices.
        # Easier method: loop and multiply using np.tensordot
        state_tensor = np.tensordot(U, state_tensor, axes=([1], [i]))
        # tensordot puts the new axis at the front (index 0).
        # We need to move it back to position i to maintain qubit order?
        # Actually, let's just move it to position i.
        state_tensor = np.moveaxis(state_tensor, 0, i)

    return state_tensor.flatten()

def apply_cz_layer(state, num_qubits, pattern='even'):
    """
    Applies CZ gates to adjacent pairs.
    pattern 'even': (0,1), (2,3), ...
    pattern 'odd': (1,2), (3,4), ...
    """
    dim = 2**num_qubits
    # CZ is diagonal in Z-basis: diag(1, 1, 1, -1) for 2 qubits.
    # We can construct the full diagonal vector efficiently.

    # We will compute the phase mask.
    # Phase is -1 if both q1 and q2 are 1.

    indices = np.arange(dim)
    phase_mask = np.ones(dim, dtype=np.complex128)

    start = 0 if pattern == 'even' else 1

    for i in range(start, num_qubits - 1, 2):
        q1 = i
        q2 = i + 1

        # Check if bit q1 and bit q2 are set
        # Bitwise logic: (n >> q1) & 1
        bit1 = (indices >> q1) & 1
        bit2 = (indices >> q2) & 1

        # Identify where both are 1
        flip_indices = (bit1 == 1) & (bit2 == 1)
        phase_mask[flip_indices] *= -1.0

    return state * phase_mask

def simulate_rcs_local(num_qubits, depth):
    """
    Simulates a Random Quantum Circuit with local gates.
    Layers: Single Qubit Random -> CZ (even) -> Single Qubit Random -> CZ (odd) ...
    """
    dim = 2**num_qubits
    print(f"Simulating Local Random Circuit on {num_qubits} qubits, Depth {depth}...")

    # Initial state |0...0>
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    for d in range(depth):
        # Layer of 1Q gates
        state = apply_single_qubit_layer(state, num_qubits)

        # Layer of CZ gates
        # Alternate patterns to ensure connectivity
        pattern = 'even' if d % 2 == 0 else 'odd'
        state = apply_cz_layer(state, num_qubits, pattern)

    # Final layer of 1Q gates to obscure the basis
    state = apply_single_qubit_layer(state, num_qubits)

    # Probabilities
    probs = np.abs(state)**2
    return probs

def plot_porter_thomas(probs, num_qubits):
    dim = 2**num_qubits
    # Normalized probabilities x = N * p
    scaled_probs = dim * probs

    plt.figure(figsize=(8, 6))

    # Plot Histogram
    plt.hist(scaled_probs, bins=50, density=True, alpha=0.6, color='b', label='Simulation (Local Gates)')

    # Theoretical Porter-Thomas: P(x) = e^{-x}
    x_vals = np.linspace(0, max(scaled_probs), 100)
    y_vals = np.exp(-x_vals)

    plt.plot(x_vals, y_vals, 'r-', linewidth=2, label='Porter-Thomas $e^{-Np}$')

    plt.xlabel('Scaled Probability $Np$')
    plt.ylabel('Frequency')
    plt.title(f'Porter-Thomas Distribution for N={num_qubits}, Local Circuit')
    plt.legend()
    plt.grid(True, alpha=0.3)

    save_path = os.path.join(OUTPUT_DIR, "porter_thomas.png")
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    # N=12 -> 4096 states.
    N = 12
    # Depth must be roughly sqrt(N) or N to scramble enough.
    # For N=12, depth 15 should be plenty.
    D = 15
    probabilities = simulate_rcs_local(N, D)
    plot_porter_thomas(probabilities, N)
