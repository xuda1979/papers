#!/usr/bin/env python
"""Generate matplotlib plots from simulated experiment results."""

import json
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_frontiers(json_path, output_dir):
    """Generate frontier plots from JSON results."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Color and marker styles for each algorithm
    styles = {
        "IGD": {'color': '#2196F3', 'marker': 's', 'label': 'IGD (Ours)'},
        "UCB-IGD": {'color': '#00BCD4', 'marker': '^', 'label': 'UCB-IGD'},
        "RS-MCTT": {'color': '#9C27B0', 'marker': 'D', 'label': 'RS-MCTT'},
        "CSC": {'color': '#FF9800', 'marker': 'p', 'label': 'CSC'},
        "ADR": {'color': '#009688', 'marker': 'o', 'label': 'ADR'},
    }
    
    # Create figure with subplots for each task
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    task_names = list(data.keys())
    task_titles = {'GameOf24': 'Game of 24 (Mathematical Reasoning)', 
                   'BitSearch': 'Bit Search (Combinatorial Optimization)'}
    
    for idx, task_name in enumerate(task_names):
        ax = axes[idx]
        task_data = data[task_name]
        
        for algo in task_data:
            name = algo['name']
            pts = algo['points']
            style = styles.get(name, {'color': 'gray', 'marker': 'x', 'label': name})
            
            xs = [x for x, y in pts]
            ys = [y * 100 for x, y in pts]  # Convert to percentage
            
            ax.plot(xs, ys, marker=style['marker'], color=style['color'], 
                   label=style['label'], linewidth=2, markersize=8)
        
        ax.set_xlabel('Compute Budget (tokens)', fontsize=12)
        ax.set_ylabel('Solution Quality (%)', fontsize=12)
        ax.set_title(task_titles.get(task_name, task_name), fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, 105)
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'frontier_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Saved plots to {output_path}")
    plt.close()
    
    # Generate summary table
    print("\n" + "="*70)
    print("FRONTIER AREA COMPARISON (Higher is better)")
    print("="*70)
    
    for task_name in task_names:
        print(f"\n{task_name}:")
        print("-" * 50)
        task_data = data[task_name]
        for algo in sorted(task_data, key=lambda x: -x['frontier_area']):
            print(f"  {algo['name']:12s}  FA={algo['frontier_area']:.2f}  Mean={algo['mean_utility']:.3f}")
    
    return data


def generate_latex_table(data, output_path):
    """Generate a LaTeX table with results."""
    
    latex = r"""\begin{table}[t]
\centering
\caption{Budget-Performance Frontier Areas (FA) and Mean Utilities across tasks. Higher values indicate better compute efficiency. \textbf{Bold} indicates best performance.}
\label{tab:frontier_results}
\begin{tabular}{l|cc|cc}
\toprule
& \multicolumn{2}{c|}{\textbf{Game of 24}} & \multicolumn{2}{c}{\textbf{Bit Search}} \\
\textbf{Algorithm} & FA & Mean $\mathcal{U}$ & FA & Mean $\mathcal{U}$ \\
\midrule
"""
    
    # Get all algorithm names
    algo_names = [algo['name'] for algo in data[list(data.keys())[0]]]
    
    for name in algo_names:
        row = f"{name}"
        for task_name in ['GameOf24', 'BitSearch']:
            if task_name in data:
                for algo in data[task_name]:
                    if algo['name'] == name:
                        fa = algo['frontier_area']
                        mean_u = algo['mean_utility']
                        # Check if best for this task
                        best_fa = max(a['frontier_area'] for a in data[task_name])
                        best_mean = max(a['mean_utility'] for a in data[task_name])
                        
                        fa_str = f"\\textbf{{{fa:.1f}}}" if fa == best_fa else f"{fa:.1f}"
                        mean_str = f"\\textbf{{{mean_u:.3f}}}" if mean_u == best_mean else f"{mean_u:.3f}"
                        row += f" & {fa_str} & {mean_str}"
                        break
        row += r" \\"
        latex += row + "\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(output_path, 'w') as f:
        f.write(latex)
    print(f"Generated LaTeX table: {output_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "results/simulated_results.json")
    output_dir = os.path.join(script_dir, "results")
    
    data = plot_frontiers(json_path, output_dir)
    
    # Generate LaTeX table
    table_path = os.path.join(script_dir, "../tex/fig/results_table.tex")
    os.makedirs(os.path.dirname(table_path), exist_ok=True)
    generate_latex_table(data, table_path)
