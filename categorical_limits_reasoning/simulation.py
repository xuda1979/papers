import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os

"""
Reasoning Gap Simulation
========================
This script simulates the ability of fixed-depth Transformers to solve the Graph Reachability problem.
It validates the "No-Go Theorem" by showing that for any fixed depth L, there is a graph size N
beyond which the accuracy collapses.

Assumption:
    - Fixed Depth L: Can solve reachability if path length <= L (Linear Propagation).
    - CoT (Chain-of-Thought): Can solve reachability if path length <= N (Adaptive Depth).
"""

def simulate_reachability():
    # Settings
    N_range = range(10, 105, 5)
    Depths = [2, 4, 8, 12] # Fixed depths
    num_graphs = 100

    results = {d: [] for d in Depths}
    results['CoT'] = [] # "Infinite" depth

    print("Starting simulation...")

    for N in N_range:
        print(f"Simulating N={N}...")

        # Stats for this N
        solved_counts = {d: 0 for d in Depths}
        solved_cot = 0
        total_connected_pairs = 0

        for _ in range(num_graphs):
            # Generate random directed graph
            # p = 3/N for sparse graphs to ensure non-trivial structure
            p = 3.0 / N
            G = nx.fast_gnp_random_graph(N, p, directed=True)

            # Calculate all pairs shortest paths for ground truth
            try:
                paths = dict(nx.all_pairs_shortest_path_length(G))
            except:
                continue

            # Sample 10 random pairs to evaluate reasoning
            nodes = list(G.nodes())
            for _ in range(10):
                u, v = np.random.choice(nodes, 2, replace=False)

                # We only care about cases where a path actually exists (Ground Truth = True)
                if v in paths[u]:
                    dist = paths[u][v]
                    total_connected_pairs += 1

                    # Check fixed depths
                    for d in Depths:
                        # Success condition: Model depth >= Path length
                        if d >= dist:
                            solved_counts[d] += 1

                    # CoT always solves it if connected (assuming sufficient context)
                    solved_cot += 1

        # Calculate percentages
        for d in Depths:
            if total_connected_pairs > 0:
                results[d].append(solved_counts[d] / total_connected_pairs * 100)
            else:
                results[d].append(100.0) # Trivial case

        if total_connected_pairs > 0:
            results['CoT'].append(100.0)
        else:
            results['CoT'].append(100.0)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Custom palette
    palette = sns.color_palette("viridis", len(Depths))

    for i, d in enumerate(Depths):
        plt.plot(N_range, results[d], marker='o', label=f'Fixed Depth L={d}', color=palette[i], linewidth=2.5)

    plt.plot(N_range, results['CoT'], marker='*', linestyle='--', label='Chain-of-Thought (Adaptive)', color='red', linewidth=2.5)

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
