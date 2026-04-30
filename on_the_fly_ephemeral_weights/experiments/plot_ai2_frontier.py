#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


@dataclass
class FrontierBestRow:
    hidden_dim: int
    batch_tokens: int
    best_rank: int
    dense_mean_ms: float
    best_ephemeral_mean_ms: float
    speedup_dense_over_ephemeral: float


def save_csv(rows: list[FrontierBestRow], path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty frontier summary to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_best_rows(df: pd.DataFrame) -> list[FrontierBestRow]:
    dense = df[df['mode'] == 'dense'].copy()
    eph = df[df['mode'] == 'ephemeral'].copy()
    rows: list[FrontierBestRow] = []

    for _, dense_row in dense.iterrows():
        bt = int(dense_row['batch_tokens'])
        d = int(dense_row['hidden_dim'])
        m = int(dense_row['expansion_dim'])
        subset = eph[
            (eph['batch_tokens'] == bt)
            & (eph['hidden_dim'] == d)
            & (eph['expansion_dim'] == m)
        ]
        best = subset.loc[subset['avg_ms'].idxmin()]
        rows.append(
            FrontierBestRow(
                hidden_dim=d,
                batch_tokens=bt,
                best_rank=int(best['rank']),
                dense_mean_ms=float(dense_row['avg_ms']),
                best_ephemeral_mean_ms=float(best['avg_ms']),
                speedup_dense_over_ephemeral=float(dense_row['avg_ms']) / float(best['avg_ms']),
            )
        )
    return sorted(rows, key=lambda x: (x.hidden_dim, x.batch_tokens))


def plot(df: pd.DataFrame, path: Path) -> None:
    dense = df[df['mode'] == 'dense'].copy()
    eph = df[df['mode'] == 'ephemeral'].copy()
    hidden_dims = sorted(eph['hidden_dim'].unique())
    batch_tokens = sorted(eph['batch_tokens'].unique())

    fig, axes = plt.subplots(1, len(hidden_dims), figsize=(6.4 * len(hidden_dims), 4.8), sharey=True)
    if len(hidden_dims) == 1:
        axes = [axes]

    colors = {
        batch_tokens[i]: plt.cm.tab10(i) for i in range(len(batch_tokens))
    }

    for ax, hidden_dim in zip(axes, hidden_dims):
        sub = eph[eph['hidden_dim'] == hidden_dim]
        dense_sub = dense[dense['hidden_dim'] == hidden_dim]
        for bt in batch_tokens:
            bt_sub = sub[sub['batch_tokens'] == bt].sort_values('rank')
            dense_row = dense_sub[dense_sub['batch_tokens'] == bt].iloc[0]
            speedup = dense_row['avg_ms'] / bt_sub['avg_ms']
            ax.plot(bt_sub['rank'], speedup, marker='o', label=f'b={bt}', color=colors[bt])
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'd={hidden_dim}')
        ax.set_xlabel('Rank r')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Dense / Ephemeral Mean-Latency Speedup')
    axes[0].legend(title='Batch Tokens')
    fig.suptitle('Ascend 910B Measured Frontier: Speedup vs Rank')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    input_path = Path('results/ai2_frontier_repeat_20260325.csv')
    df = pd.read_csv(input_path)
    plot(df, Path('figures/ai2_frontier_speedup_vs_rank.png'))
    best_rows = build_best_rows(df)
    save_csv(best_rows, Path('results/ai2_frontier_best_20260325.csv'))
    print('wrote figures/ai2_frontier_speedup_vs_rank.png')
    print('wrote results/ai2_frontier_best_20260325.csv')
