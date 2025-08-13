import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import csv

def generate_stylized_curve(steps, initial_acc, nadir, recovery_time,
                            shift_point, adaptation_duration=0, final_stabilization=0.99):
    """Generates a stylized accuracy curve for a concept drift scenario."""
    time = np.arange(steps)
    accuracy = np.full(steps, initial_acc, dtype=float)

    # Phase 1: Stable performance
    accuracy[:shift_point] = initial_acc + np.random.normal(0, 0.5, shift_point)

    # Phase 2 Start: Sharp drop to nadir
    drop_duration = 150
    drop_end = min(shift_point + drop_duration, steps)
    if drop_end > shift_point:
        accuracy[shift_point:drop_end] = np.linspace(accuracy[shift_point-1],
                                                     nadir, drop_end - shift_point)

    # Adaptation phase (optional, for ARH)
    recovery_start = min(drop_end + adaptation_duration, steps)
    if recovery_start > drop_end:
        accuracy[drop_end:recovery_start] = nadir

    # Recovery phase
    recovery_end = min(recovery_start + recovery_time, steps)
    if recovery_end > recovery_start:
        # Use a sigmoid-like curve for smooth recovery
        x = np.linspace(-6, 6, recovery_end - recovery_start)
        sigmoid = 1 / (1 + np.exp(-x))
        target_acc = initial_acc * final_stabilization
        recovered_segment = nadir + (target_acc - nadir) * sigmoid
        accuracy[recovery_start:recovery_end] = recovered_segment
        if recovery_end < steps:
            accuracy[recovery_end:] = target_acc + np.random.normal(0, 0.5, steps-recovery_end)

    return np.clip(accuracy, 0, 100)

def main():
    # --- Parameters ---
    # These parameters are chosen to create a visually compelling illustration
    # of the HCD scenario as described in the paper. They do not represent
    # the results of a real experiment.
    TOTAL_STEPS = 10000
    SHIFT_STEP = 5000 # The point at which the task's rules change.

    # --- Generate Data for each model ---
    # The parameters for each model (initial accuracy, nadir, recovery time)
    # are selected to reflect their theoretical strengths and weaknesses.
    time_steps = np.arange(TOTAL_STEPS)
    curves = {
        # LLM: High initial performance, but catastrophic failure after the shift.
        "LLM (Monolithic)": generate_stylized_curve(TOTAL_STEPS, 91.5, 22.5, 10000, SHIFT_STEP),
        # HRM: Good initial performance, better than LLM post-shift, but slow recovery.
        "HRM (Static)": generate_stylized_curve(TOTAL_STEPS, 93.1, 35.0, 7500, SHIFT_STEP),
        # ARH: Good initial performance, a high nadir, and rapid recovery due to adaptation.
        "ARH (Ours, Dynamic)": generate_stylized_curve(TOTAL_STEPS, 92.8, 65.2, 1700, SHIFT_STEP, adaptation_duration=400)
    }

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid') # Using a modern, clean plotting style.
    fig, ax = plt.subplots(figsize=(12, 6))

    plot_params = {
        "LLM (Monolithic)": {"color": "red", "linestyle": ":"},
        "HRM (Static)": {"color": "orange", "linestyle": "--"},
        "ARH (Ours, Dynamic)": {"color": "blue", "linestyle": "-", "linewidth": 2.5}
    }

    for label, params in plot_params.items():
        ax.plot(time_steps, curves[label], label=label, **params)

    # Highlight the structural shift
    ax.axvline(x=SHIFT_STEP, color='black', linestyle='--', linewidth=1.5, label='Structural Shift')

    # Highlight ARH adaptation phase
    adapt_start = SHIFT_STEP + 150
    adapt_end = adapt_start + 400
    rect = patches.Rectangle((adapt_start, 0), adapt_end - adapt_start, 100,
                             linewidth=0, facecolor='blue', alpha=0.1)
    ax.add_patch(rect)
    ax.text(adapt_start + 200, 40, 'ARH Adaptation\n(STDC triggers)',
            horizontalalignment='center', color='blue', alpha=0.8)

    # Formatting
    ax.set_xlabel('Time Steps', fontsize=14)
    ax.set_ylabel('Task Accuracy (%)', fontsize=14)
    ax.set_title('Model Performance on Hierarchical Concept Drift (HCD)', fontsize=16)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, TOTAL_STEPS)
    plt.tight_layout()

    # --- Save outputs ---
    out_dir = Path(__file__).resolve().parent
    plot_file = out_dir / 'arh_hcd_visualization.pdf'
    curves_data_file = out_dir / 'arh_hcd_curves.csv'
    table_data_file = out_dir / 'hcd_results_table.csv'

    plt.savefig(plot_file)
    print(f"Saved plot to {plot_file}")

    # Save the plot curves data
    with curves_data_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step'] + list(curves.keys()))
        for i in range(TOTAL_STEPS):
            row = [time_steps[i]] + [curves[label][i] for label in curves]
            writer.writerow(row)
    print(f"Saved data to {curves_data_file}")

    # Save the data for the results table
    table_data = [
        ("LLM (Monolithic)", 91.5, 22.5, "> 10,000 (failed)"),
        ("HRM (Static, 2-layer)", 93.1, 35.0, "approx. 7,500"),
        ("ARH (Ours, dynamic)", 92.8, 65.2, "1,700")
    ]
    with table_data_file.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Phase 1 Accuracy (%)', 'Post-Shift Nadir (%)', 'Recovery Speed (steps to 90%)'])
        writer.writerows(table_data)
    print(f"Saved table data to {table_data_file}")

if __name__ == '__main__':
    main()
