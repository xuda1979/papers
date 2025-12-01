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

    results = {d: [] for d in Depths}
    results['PJ_4'] = []
    results['CoT'] = [] # "Infinite" depth

    print("Starting simulation...")

    for N in N_range:
        print(f"Simulating N={N}...")

        # Stats for this N
        solved_counts = {d: 0 for d in Depths}
        solved_pj = 0
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

            # We'll sample 10 pairs per graph
            nodes = list(G.nodes())
            if len(nodes) < 2: continue

            for _ in range(10):
                u, v = np.random.choice(nodes, 2, replace=False)

                if v in paths[u]:
                    dist = paths[u][v]
                    total_connected_pairs += 1

                    # Check fixed depths (Linear Mixing Hypothesis)
                    for d in Depths:
                        if d >= dist:
                            solved_counts[d] += 1

                    # Check Pointer Jumping Hypothesis for Depth 4
                    # Effective Depth = 2^L
                    if (2**PJ_Depth) >= dist:
                        solved_pj += 1

                    # CoT always solves it if connected
                    solved_cot += 1

        # Calculate percentages
        for d in Depths:
            if total_connected_pairs > 0:
                results[d].append(solved_counts[d] / total_connected_pairs * 100)
            else:
                results[d].append(100.0) # Trivial

        if total_connected_pairs > 0:
            results['PJ_4'].append(solved_pj / total_connected_pairs * 100)
            results['CoT'].append(100.0)
        else:
            results['PJ_4'].append(100.0)
            results['CoT'].append(100.0)

    # Plotting
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Custom palette
    palette = sns.color_palette("viridis", len(Depths))

    for i, d in enumerate(Depths):
        plt.plot(N_range, results[d], marker='o', label=f'Linear Mixing L={d}', color=palette[i], linewidth=2.5)

    # Plot Pointer Jumping for L=4
    plt.plot(N_range, results['PJ_4'], marker='^', linestyle=':', label=f'Pointer Jumping L={PJ_Depth} ($2^L$)', color='orange', linewidth=2.5)

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
