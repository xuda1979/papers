import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.linalg import svd, eigh

# Configure plotting style
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams['font.family'] = 'serif'

np.random.seed(42)

# --- Quantum Simulator (State Vector) ---

def apply_gate(state, gate, target, n_qubits):
    """Applies a single qubit gate."""
    # gate: (2, 2)
    # state: (2**n_qubits)
    # We use tensor reshaping to apply the gate
    t = state.reshape((2**target, 2, 2**(n_qubits - target - 1)))
    # Contract: gate[i, j] * t[..., j, ...]
    new_t = np.einsum('ij,kjl->kil', gate, t)
    return new_t.reshape(-1)

def apply_crz(state, phi, control, target, n_qubits):
    """Applies a Controlled-RZ gate."""
    # CRZ(phi) = |0><0| I + |1><1| RZ(phi)
    # RZ(phi) = diag(e^{-i phi/2}, e^{i phi/2})
    N = n_qubits
    new_state = state.copy()

    # Vectorized application using masks
    indices = np.arange(2**N)
    control_mask = (indices >> (N - 1 - control)) & 1
    target_mask = (indices >> (N - 1 - target)) & 1

    # Phase shift is applied only where control is 1
    # If target is 0 -> -phi/2, if target is 1 -> phi/2
    phases = np.zeros(2**N)
    phases[control_mask == 1] = np.where(target_mask[control_mask == 1] == 0, -phi/2, phi/2)

    # Apply phase: exp(i * phase)
    # Note: RZ is usually exp(-i Z phi / 2).
    # Z|0> = |0>, Z|1> = -|1>.
    # So |0> -> exp(-i phi/2), |1> -> exp(i phi/2).
    # This matches the logic above.

    new_state *= np.exp(1j * phases)
    return new_state

def ry_gate(theta):
    return np.array([
        [np.cos(theta/2), -np.sin(theta/2)],
        [np.sin(theta/2),  np.cos(theta/2)]
    ])

def get_expectation_z(state, qubit_idx, n_qubits):
    """Calculates <psi| Z_i |psi>."""
    prob_dens = np.abs(state)**2
    indices = np.arange(2**n_qubits)
    mask = (indices >> (n_qubits - 1 - qubit_idx)) & 1
    # 0 -> +1, 1 -> -1
    vals = np.where(mask == 0, 1, -1)
    return np.sum(vals * prob_dens)

def get_entanglement_entropy(state, n_qubits):
    """
    Calculates Von Neumann entropy of the reduced density matrix
    for the first N/2 qubits.
    """
    half_n = n_qubits // 2
    psi_matrix = state.reshape((2**half_n, 2**(n_qubits - half_n)))
    try:
        _, s, _ = svd(psi_matrix, full_matrices=False)
        eigenvalues = s**2
        eigenvalues = eigenvalues[eigenvalues > 1e-15]
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        return entropy
    except:
        return 0.0

# --- VQC Circuit ---

class VQC:
    def __init__(self, n_qubits, n_layers, entanglement_phi):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.phi = entanglement_phi
        self.params = np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits))

    def forward(self, x, params=None):
        if params is None:
            params = self.params

        state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        state[0] = 1.0 # |00...0>

        # Encoding: Ry(arctan(x))
        angles = np.arctan(x)
        for i in range(self.n_qubits):
            feat_idx = i % len(x)
            gate = ry_gate(angles[feat_idx])
            state = apply_gate(state, gate, i, self.n_qubits)

        # Variational Layers
        reshaped_params = params.reshape(self.n_layers, self.n_qubits)

        for l in range(self.n_layers):
            # Rotation layer
            for q in range(self.n_qubits):
                theta = reshaped_params[l, q]
                state = apply_gate(state, ry_gate(theta), q, self.n_qubits)

            # Entangling layer (CRZ chain)
            # Circular topology
            for q in range(self.n_qubits):
                control = q
                target = (q + 1) % self.n_qubits
                state = apply_crz(state, self.phi, control, target, self.n_qubits)

        return state

    def predict(self, x, params=None):
        state = self.forward(x, params)
        return get_expectation_z(state, 0, self.n_qubits)

    def compute_gradients(self, x, params=None):
        """
        Computes the gradient of the state vector |psi> w.r.t parameters using finite differences.
        This is for FIM calculation.
        Returns: Matrix of shape (2^N, num_params)
        """
        if params is None:
            params = self.params

        flat_params = params.flatten()
        num_params = len(flat_params)
        dim = 2**self.n_qubits

        grads = np.zeros((dim, num_params), dtype=np.complex128)
        epsilon = 1e-4

        base_state = self.forward(x, params)

        for i in range(num_params):
            p_perturbed = flat_params.copy()
            p_perturbed[i] += epsilon
            state_perturbed = self.forward(x, p_perturbed.reshape(self.params.shape))

            grads[:, i] = (state_perturbed - base_state) / epsilon

        return base_state, grads

def compute_qfi_rank(model, X_sample, tol=1e-4):
    """
    Computes the effective dimension via the rank of the Quantum Fisher Information Matrix (QFIM).
    QFIM_ij = 4 * Re( <d_i psi | d_j psi> - <d_i psi | psi><psi | d_j psi> )
    We average the QFIM over inputs X_sample.
    """
    num_params = model.params.size
    qfim_sum = np.zeros((num_params, num_params))

    for x in X_sample:
        psi, grads = model.compute_gradients(x)
        # grads is (dim, P)
        # psi is (dim,)

        # Term 1: <d_i psi | d_j psi>
        # (P, dim) @ (dim, P) -> (P, P)
        term1 = np.dot(grads.conj().T, grads)

        # Term 2: <d_i psi | psi> <psi | d_j psi>
        # <d_i psi | psi> is a vector of size P
        overlap = np.dot(grads.conj().T, psi) # (P,)
        term2 = np.outer(overlap, overlap.conj())

        qfim = 4 * np.real(term1 - term2)
        qfim_sum += qfim

    avg_qfim = qfim_sum / len(X_sample)

    # Calculate Rank
    evals = np.linalg.eigvalsh(avg_qfim)
    # Normalize eigenvalues by number of parameters for consistent scale interpretation
    # effective dimension roughly sum of eigenvalues / max_eigenvalue or just count > tol
    # Abbas et al use the capacitance definition, but rank is a good proxy for capacity.

    rank = np.sum(evals > tol)

    # Return normalized effective dimension (0 to 1)
    return rank / num_params

# --- Training and Experiment ---

def generate_data(n_samples=50):
    X = np.random.uniform(-1, 1, size=(n_samples, 2))
    # Non-linear boundary: Concentric circles + some noise
    # y = 1 if r < 0.6, else -1
    y = np.array([1 if x[0]**2 + x[1]**2 < 0.6**2 else -1 for x in X])
    return X, y

def loss_fn(model, params, X, y):
    predictions = np.array([model.predict(x, params) for x in X])
    return np.mean((predictions - y)**2)

def train(model, X, y, iterations=100, eta=0.1):
    params = model.params.flatten()

    for i in range(iterations):
        c = 0.1 / ( (i+1)**0.11 )
        lr = eta / ( (i+1)**0.602 )
        delta = np.random.choice([-1, 1], size=params.shape)
        loss_plus = loss_fn(model, params + c*delta, X, y)
        loss_minus = loss_fn(model, params - c*delta, X, y)
        grad = (loss_plus - loss_minus) / (2 * c) * delta
        params -= lr * grad

    model.params = params.reshape(model.n_layers, model.n_qubits)
    return model

def run_experiment():
    print("Starting Advanced Simulation...")

    # Configuration
    N_QUBITS = 6  # Increased from 4
    N_LAYERS = 3  # Reduced depth slightly for speed
    ITERATIONS = 40 # Reduced iterations
    TRIALS = 3    # Reduced trials

    print(f"System: N={N_QUBITS}, Layers={N_LAYERS}")

    X_train, y_train = generate_data(80)
    X_test, y_test = generate_data(40)

    # For FIM calc, use a subset
    X_fim = X_train[:6]

    phis = np.linspace(0, 2.5, 8)

    final_errors = []
    final_entropies = []
    final_eff_dims = []

    for idx, phi in enumerate(phis):
        print(f"[{idx+1}/{len(phis)}] Testing Entanglement Strength phi = {phi:.2f}...")

        errors_run = []
        entropies_run = []
        eff_dims_run = []

        for trial in range(TRIALS):
            model = VQC(N_QUBITS, N_LAYERS, phi)
            model = train(model, X_train, y_train, iterations=ITERATIONS)

            # 1. Test Error
            test_loss = loss_fn(model, model.params, X_test, y_test)
            errors_run.append(test_loss)

            # 2. Entanglement Entropy
            # Average over a few random inputs
            ents = []
            for _ in range(5):
                sample_in = np.random.uniform(-1, 1, 2)
                sample_state = model.forward(sample_in)
                ents.append(get_entanglement_entropy(sample_state, N_QUBITS))
            entropies_run.append(np.mean(ents))

            # 3. Effective Dimension (QFIM Rank)
            eff_dim = compute_qfi_rank(model, X_fim)
            eff_dims_run.append(eff_dim)

        final_errors.append(np.mean(errors_run))
        final_entropies.append(np.mean(entropies_run))
        final_eff_dims.append(np.mean(eff_dims_run))

    # --- Plotting ---
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Plot 1: Generalization Error vs Entanglement
    plt.figure(figsize=(10, 6))

    # Scatter plot with color mapping
    sc = plt.scatter(final_entropies, final_errors, c=phis, cmap='plasma', s=120, edgecolor='k', alpha=0.9)

    # Polynomial trend line
    if len(final_entropies) > 3:
        z = np.polyfit(final_entropies, final_errors, 2)
        p = np.poly1d(z)
        x_sort = np.sort(final_entropies)
        plt.plot(x_sort, p(x_sort), 'k--', alpha=0.6, label='Quadratic Fit', linewidth=2)

    cbar = plt.colorbar(sc)
    cbar.set_label('Entangling Power $\phi$')

    plt.xlabel('Average Entanglement Entropy $S$', fontsize=12)
    plt.ylabel('Generalization Error (MSE)', fontsize=12)
    plt.title(f'Entanglement-Generalization Duality ($N={N_QUBITS}$)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'entanglement_vs_generalization.png'), dpi=300)
    plt.close()

    # Plot 2: Effective Dimension vs Entanglement
    # This shows "Capacity" vs "Connectivity"
    plt.figure(figsize=(10, 6))

    # Dual axis plot? No, just Eff Dim vs Entropy
    sns.regplot(x=np.array(final_entropies), y=np.array(final_eff_dims),
                scatter_kws={'s': 100, 'edgecolor':'w'},
                order=2, ci=None, line_kws={'color': 'darkred', 'linestyle':'--'})

    plt.xlabel('Average Entanglement Entropy $S$', fontsize=12)
    plt.ylabel('Effective Dimension Ratio $d_{eff} / d_{param}$', fontsize=12)
    plt.title('Capacity Phase Transition', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, 'effective_dimension.png'), dpi=300)
    plt.close()

    print("Simulation Complete. Results saved.")

if __name__ == "__main__":
    run_experiment()
