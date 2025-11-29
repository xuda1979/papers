import numpy as np
import matplotlib.pyplot as plt

def simulate_transformer_cost(seq_length, d_model):
    """
    Simulates computational cost for Transformer attention.
    Complexity: O(N^2 * d)
    """
    # 2*N^2*d for QK^T, N^2 for softmax, N^2*d for AV
    flops = 4 * (seq_length ** 2) * d_model
    return flops

def simulate_mamba_cost(seq_length, d_model, d_state):
    """
    Simulates computational cost for Mamba (SSM).
    Complexity: O(N * d * state)
    Detailed:
    - Input projection: N * d * 2d_inner
    - Convolution: N * d_inner * kernel
    - SSM: N * d_inner * d_state * 2
    - Output projection: N * d_inner * d
    """
    # Simplified approximation for relative scaling
    flops = 10 * seq_length * d_model * d_state
    return flops

def simulate_ssa_cost(seq_length, d_model, k_clusters=None):
    """
    Simulates computational cost for Spectral Sparse Attention (SSA).
    Complexity: O(N^1.5 * d) with single level clustering k=sqrt(N)
    """
    if k_clusters is None:
        k_clusters = int(np.sqrt(seq_length))

    # Projection: N * d * m (let m=d/2)
    flops_proj = seq_length * d_model * (d_model // 2)

    # Clustering (approx): N * k * iter (negligible compared to attention)

    # Local Attention: k * (N/k)^2 * d = N^2/k * d
    flops_local = (seq_length**2 / k_clusters) * d_model

    # Global Attention: N * s * d (let s=d)
    flops_global = seq_length * d_model * d_model

    return flops_proj + flops_local + flops_global

def main():
    print("Running numerical experiments simulation...")

    seq_lengths = np.linspace(128, 8192, 20).astype(int)
    d_model = 512
    d_state = 16

    t_costs = []
    m_costs = []
    ssa_costs = []

    print(f"{'Seq Length':<15} | {'Transformer':<15} | {'Mamba':<15} | {'SSA (Proposed)':<15}")
    print("-" * 65)

    for N in seq_lengths:
        t = simulate_transformer_cost(N, d_model)
        m = simulate_mamba_cost(N, d_model, d_state)
        s = simulate_ssa_cost(N, d_model)

        t_costs.append(t)
        m_costs.append(m)
        ssa_costs.append(s)

        print(f"{N:<15} | {t:<15.2e} | {m:<15.2e} | {s:<15.2e}")

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, t_costs, label='Transformer (O($N^2$))', marker='o')
    plt.plot(seq_lengths, m_costs, label='Mamba (O($N$))', marker='s')
    plt.plot(seq_lengths, ssa_costs, label='SSA (Proposed O($N^{1.5}$))', marker='^')

    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Estimated FLOPs')
    plt.title('Computational Complexity Analysis')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.yscale('log') # Log scale to see the differences better

    output_file = 'energy_efficient_ai/complexity_plot.png'
    plt.savefig(output_file)
    print(f"\nPlot saved to {output_file}")

if __name__ == "__main__":
    main()
