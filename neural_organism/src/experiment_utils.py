import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_bandit_results(
    results: dict,
    out_dir: str,
    filename_prefix: str,
):
    """Plots and saves cumulative reward and regret for bandit experiments.

    Args:
        results (dict): A dictionary where keys are policy names and values are
                        dicts containing 'cum_rewards' and 'cum_regrets'.
        out_dir (str): Directory to save the plots.
        filename_prefix (str): Prefix for the output plot filenames.
    """
    T = len(next(iter(results.values()))['cum_rewards'])
    x = np.arange(T)
    steps_per_phase = T // 6  # Assuming 6 phases as in the experiments

    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        plt.plot(x, data['cum_rewards'], label=name)
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Contextual bandit: cumulative reward under drift")
    plt.xlabel("Step")
    plt.ylabel("Cumulative reward")
    plt.legend()
    cumr_path = os.path.join(out_dir, f"{filename_prefix}_cumreward.png")
    plt.tight_layout()
    plt.savefig(cumr_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    for name, data in results.items():
        plt.plot(x, data['cum_regrets'], label=name)
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1)
    plt.title("Contextual bandit: cumulative regret under drift")
    plt.xlabel("Step")
    plt.ylabel("Cumulative regret")
    plt.legend()
    regret_path = os.path.join(out_dir, f"{filename_prefix}_cumregret.png")
    plt.tight_layout()
    plt.savefig(regret_path)
    plt.close()

    return cumr_path, regret_path


def save_bandit_summary(
    summary_data: list,
    out_dir: str,
    filename: str,
):
    """Saves the bandit experiment summary to a CSV file.

    Args:
        summary_data (list): A list of dictionaries, where each dictionary
                             represents a row in the summary table.
        out_dir (str): Directory to save the CSV file.
        filename (str): Name of the output CSV file.
    """
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(out_dir, filename)
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def plot_classification_results(
    results: dict,
    out_dir: str,
    filename: str,
    window_size: int = 100,
):
    """Plots and saves classification accuracy over time.

    Args:
        results (dict): A dictionary where keys are model names and values
                        are arrays of correctness (1s and 0s).
        out_dir (str): Directory to save the plot.
        filename (str): Name of the output plot file.
        window_size (int): The size of the sliding window for smoothing accuracy.
    """
    plt.figure(figsize=(10, 6))
    for name, corrects in results.items():
        # Calculate sliding window accuracy
        smoothed_acc = pd.Series(corrects).rolling(window=window_size, min_periods=1).mean()
        plt.plot(smoothed_acc, label=name)

    T = len(next(iter(results.values())))
    steps_per_phase = T // 6  # Assuming 6 phases
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray', alpha=0.7)

    plt.title(f"Sliding Window Accuracy (size={window_size}) Under Drift")
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    acc_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()
    return acc_path
