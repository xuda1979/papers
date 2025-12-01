import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

def simulate_reachability():
    # Settings
    N_range = range(10, 105, 5)
    Depths = [2, 4, 8, 12] # Fixed depths
    # We also simulate "Pointer Jumping" for one depth to illustrate the difference
    PJ_Depth = 4

    num_graphs = 100

    results_linear = {d: [] for d in Depths}
    results_pj = {d: [] for d in Depths}
    results_cot = []

    print("Starting simulation...")

    for N in N_range:
        print(f"Simulating N={N}...")

        # Stats for this N
        solved_counts_linear = {d: 0 for d in Depths}
        solved_counts_pj = {d: 0 for d in Depths}
        solved_cot = 0
        total_connected_pairs = 0

        for _ in range(num_graphs):
            # Generate random directed graph
            # p = 3/N for sparse graphs
            p = 3.0 / N
            G = nx.fast_gnp_random_graph(N, p, directed=True)

            # Select random start and end
            try:
                # Calculate all pairs shortest paths
                paths = dict(nx.all_pairs_shortest_path_length(G))
            except:
                continue

            # Sample 10 pairs per graph
            nodes = list(G.nodes())
            if len(nodes) < 2: continue

            for _ in range(10):
                u, v = np.random.choice(nodes, 2, replace=False)

                if v in paths[u]:
                    dist = paths[u][v]
                    total_connected_pairs += 1

                    # Check fixed depths
                    for d in Depths:
                        # Linear: Depth >= Path Length
                        if d >= dist:
                            solved_counts_linear[d] += 1

                        # Exponential (Pointer Jumping): 2^Depth >= Path Length
                        if (2**d) >= dist:
                            solved_counts_pj[d] += 1

                    # CoT always solves it if connected
                    solved_cot += 1

        # Calculate percentages
        for d in Depths:
            if total_connected_pairs > 0:
                results_linear[d].append(solved_counts_linear[d] / total_connected_pairs * 100)
                results_pj[d].append(solved_counts_pj[d] / total_connected_pairs * 100)
            else:
                results_linear[d].append(100.0)
                results_pj[d].append(100.0)

        if total_connected_pairs > 0:
            results_cot.append(100.0)
        else:
            results_cot.append(100.0)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Custom palette
    palette = sns.color_palette("viridis", len(Depths))

    # Plot Linear (Solid lines)
    for i, d in enumerate(Depths):
        plt.plot(N_range, results_linear[d], marker='o', label=f'Linear L={d}', color=palette[i], linewidth=2.5)

    # Plot Pointer Jumping for L=4 (example)
    # plt.plot(N_range, results_pj[4], marker='^', linestyle=':', label=f'Pointer Jumping L={PJ_Depth} ($2^L$)', color='orange', linewidth=2.5)

    plt.plot(N_range, results_cot, marker='*', linestyle='--', label='Chain-of-Thought (Adaptive)', color='red', linewidth=2.5)

    plt.title('The "Categorical Gap": Fixed-Depth Transformers vs. Problem Scale', fontsize=16)
    plt.xlabel('Problem Size (N) - Graph Nodes', fontsize=14)
    plt.ylabel('Reasoning Accuracy (% Solvable)', fontsize=14)
    plt.legend(fontsize=12)
    plt.ylim(0, 105)
    plt.tight_layout()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'reasoning_gap.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    simulate_reachability()
