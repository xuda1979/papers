import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_speedup_v_rank():
    df = pd.read_csv('results/ephemeral_weight_benchmark.csv')

    hw_name = 'Ascend_910B_like'
    d = 8192
    gm = 128.0

    subset = df[
        (df['hardware'] == hw_name)
        & (df['hidden_dim'] == d)
        & (df['generator_flop_multiplier'] == gm)
    ]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset, x='rank', y='speedup', hue='batch_tokens', marker='o')

    plt.yscale('log')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.title(
        f'Ephemeral Weight Speedup vs Rank ({hw_name})\n'
        f'$d={d}$, Generator Mult={gm}x'
    )
    plt.ylabel('Speedup (Log Scale)')
    plt.xlabel('Low Rank ($r$)')
    plt.grid(True, which='both', ls='-', alpha=0.5)
    plt.legend(title='Batch Tokens')

    plt.tight_layout()
    Path('figures').mkdir(exist_ok=True)
    plt.savefig('figures/speedup_vs_rank_910b.png', dpi=200)
    print('Saved figures/speedup_vs_rank_910b.png')


def plot_generator_sensitivity():
    df = pd.read_csv('results/ephemeral_weight_benchmark.csv')
    hw_name = 'Ascend_910B_like'
    d = 8192
    r = 128

    subset = df[
        (df['hardware'] == hw_name)
        & (df['hidden_dim'] == d)
        & (df['rank'] == r)
    ]

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset, x='generator_flop_multiplier', y='speedup', hue='batch_tokens', marker='s')

    plt.yscale('log')
    plt.axhline(y=1.0, color='r', linestyle='--', label='Break-even')
    plt.title(
        f'Sensitivity to Generator Cost ({hw_name})\n'
        f'$d={d}$, Rank $r={r}$'
    )
    plt.ylabel('Speedup (Log Scale)')
    plt.xlabel('Generator FLOP Multiplier')
    plt.grid(True, which='both', ls='-', alpha=0.5)
    plt.legend(title='Batch Tokens')

    plt.tight_layout()
    plt.savefig('figures/generator_sensitivity_910b.png', dpi=200)
    print('Saved figures/generator_sensitivity_910b.png')


if __name__ == '__main__':
    plot_speedup_v_rank()
    plot_generator_sensitivity()
