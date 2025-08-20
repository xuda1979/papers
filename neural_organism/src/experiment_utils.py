import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --- General Plotting Settings ---
plt.style.use('seaborn-v0_8-whitegrid')
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FIG_SIZE_BANDIT = (8, 5)
FIG_SIZE_CLASSIFICATION = (10, 6)

def plot_bandit_results(
    results: dict,
    out_dir: str,
    plot_settings: dict,
):
    """Plots and saves cumulative reward and regret for bandit experiments."""
    filename_prefix = plot_settings['filename_prefix']
    T = len(next(iter(results.values()))['cum_rewards'])
    x = np.arange(T)
    steps_per_phase = T // 6

    # Plot Cumulative Reward
    plt.figure(figsize=FIG_SIZE_BANDIT)
    for name, data in results.items():
        plt.plot(x, data['cum_rewards'], label=name)
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray')
    plt.title("Cumulative Reward Under Drift", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Cumulative Reward", fontsize=FONT_SIZE_LABEL)
    plt.legend()
    cumr_path = os.path.join(out_dir, f"{filename_prefix}_cumreward.png")
    plt.tight_layout()
    plt.savefig(cumr_path)
    plt.close()

    # Plot Cumulative Regret
    plt.figure(figsize=FIG_SIZE_BANDIT)
    for name, data in results.items():
        plt.plot(x, data['cum_regrets'], label=name)
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray')
    plt.title("Cumulative Regret Under Drift", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Cumulative Regret", fontsize=FONT_SIZE_LABEL)
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
    """Saves the bandit experiment summary to a CSV file."""
    summary_df = pd.DataFrame(summary_data).sort_values("Policy")
    summary_path = os.path.join(out_dir, filename)
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    return summary_path


def plot_classification_results(
    results: dict,
    out_dir: str,
    plot_settings: dict,
    window_size: int = 100,
):
    """Plots and saves classification accuracy over time."""
    filename = plot_settings['acc_out_file']
    plt.figure(figsize=FIG_SIZE_CLASSIFICATION)
    for name, corrects in results.items():
        smoothed_acc = pd.Series(corrects).rolling(window=window_size, min_periods=1).mean()
        plt.plot(smoothed_acc, label=name)

    T = len(next(iter(results.values())))
    steps_per_phase = T // 6
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray', alpha=0.7)

    plt.title(f"Sliding Window Accuracy (size={window_size})", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("Accuracy", fontsize=FONT_SIZE_LABEL)
    plt.ylim(0, 1.05)
    plt.legend()
    acc_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(acc_path)
    plt.close()
    return acc_path

def plot_prototypes(
    counts: np.ndarray,
    model_name: str,
    out_dir: str,
    filename: str,
):
    """Plots the growth of prototypes over time for a single model."""
    plt.figure(figsize=(8, 3.8))
    plt.plot(counts, label=model_name)
    steps_per_phase = len(counts) // 6
    for dp in [i * steps_per_phase for i in range(1, 6)]:
        plt.axvline(dp, linestyle='--', linewidth=1, color='gray')
    plt.title("Prototype Growth Over Time", fontsize=FONT_SIZE_TITLE)
    plt.xlabel("Step", fontsize=FONT_SIZE_LABEL)
    plt.ylabel("# Prototypes", fontsize=FONT_SIZE_LABEL)
    plt.legend()
    proto_path = os.path.join(out_dir, filename)
    plt.tight_layout()
    plt.savefig(proto_path)
    plt.close()
    return proto_path
