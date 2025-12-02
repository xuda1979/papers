import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import pandas as pd
import csv

def main():
    # Load Real Data
    script_dir = Path(__file__).parent
    data_file = script_dir / 'simulation_results.csv'

    if not data_file.exists():
        print("Data file not found. Run simulation.py first.")
        return

    df = pd.read_csv(data_file)
    steps = df['step'].values
    arh_acc = df['ARH_Accuracy'].values
    base_acc = df['Baseline_Accuracy'].values
    dissonance = df['ARH_Dissonance'].values

    # Check if 'ARH_Nodes' is in df, otherwise use dummy or skip
    if 'ARH_Nodes' in df.columns:
        nodes = df['ARH_Nodes'].values
    else:
        nodes = np.zeros_like(steps)

    TOTAL_STEPS = len(steps)
    SHIFT_STEP = 5000
    NOISE_START = 2500
    NOISE_END = 3500
    THETA_SPAWN = 5.0 # Match simulation.py

    # Smooth the curves for better visualization
    window = 50
    arh_smooth = pd.Series(arh_acc).rolling(window=window, min_periods=1).mean().values
    base_smooth = pd.Series(base_acc).rolling(window=window, min_periods=1).mean().values

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 9),
        sharex=True, gridspec_kw={'height_ratios': [3, 1]}
    )

    # --- Plot 1: Accuracy ---
    ax1.plot(steps, base_smooth, label="Baseline (Static)", color="orange", linestyle="--", alpha=0.8)
    ax1.plot(steps, arh_smooth, label="ARH (Ours)", color="blue", linewidth=2.5)

    # Highlight the structural shift
    ax1.axvline(x=SHIFT_STEP, color='black', linestyle='--', linewidth=1.5, label='Concept Drift')

    # Highlight Noise Phase
    ax1.axvspan(NOISE_START, NOISE_END, color='gray', alpha=0.2, label='Noise Injection (Test Stability)')

    # Formatting for main plot
    ax1.set_ylabel('Task Accuracy (%)', fontsize=14)
    ax1.set_title('Model Performance on Hierarchical Concept Drift (HCD)', fontsize=16)
    ax1.legend(fontsize=12, loc="lower left")
    ax1.set_ylim(0, 105)

    # --- Plot 2: Dissonance and Nodes ---
    ax2.plot(steps, dissonance, color='purple', label='ARH Dissonance Level $D(t)$', linewidth=1)
    ax2.axhline(y=THETA_SPAWN, color='red', linestyle='--', label=f'Spawn Threshold $\\theta={THETA_SPAWN}$')
    ax2.axvline(x=SHIFT_STEP, color='black', linestyle='--', linewidth=1.5)

    # Highlight Noise Phase in subplot 2
    ax2.axvspan(NOISE_START, NOISE_END, color='gray', alpha=0.2)

    # Fill area where dissonance is high
    ax2.fill_between(steps, 0, dissonance, color='purple', alpha=0.2)

    # Add Node count as text annotation or secondary axis
    # Let's use a secondary axis for Node Count to show growth
    ax3 = ax2.twinx()
    ax3.plot(steps, nodes, color='green', linestyle=':', label='Node Count', linewidth=2)
    ax3.set_ylabel('Network Size (Nodes)', fontsize=14, color='green')
    ax3.tick_params(axis='y', labelcolor='green')
    ax3.grid(False) # Turn off grid for secondary axis to avoid clutter

    ax2.set_xlabel('Time Steps', fontsize=14)
    ax2.set_ylabel('Dissonance', fontsize=14)

    # Combine legends
    lines, labels = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(lines + lines3, labels + labels3, fontsize=10, loc='upper left')

    ax2.set_xlim(0, TOTAL_STEPS)
    ax2.set_ylim(bottom=0, top=THETA_SPAWN * 2.5) # Scale y-axis to see spikes

    plt.tight_layout()
    fig.align_ylabels()

    # --- Save outputs ---
    plot_file = script_dir / 'arh_hcd_visualization.pdf'
    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")

if __name__ == '__main__':
    main()
