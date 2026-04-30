#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class BestShapeRow:
    batch_tokens: int
    hidden_dim: int
    expansion_dim: int
    dense_mean_ms: float
    dense_std_ms: float
    best_ephemeral_mean_ms: float
    best_ephemeral_std_ms: float
    best_rank: int
    speedup_dense_over_ephemeral: float


def save_csv(rows: list[BestShapeRow], path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty repeatability summary to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_best_rows(summary: pd.DataFrame) -> list[BestShapeRow]:
    rows: list[BestShapeRow] = []
    dense_df = summary[summary['mode'] == 'dense']
    eph_df = summary[summary['mode'] == 'ephemeral']

    for _, dense_row in dense_df.iterrows():
        key = (
            int(dense_row['batch_tokens']),
            int(dense_row['hidden_dim']),
            int(dense_row['expansion_dim']),
        )
        candidates = eph_df[
            (eph_df['batch_tokens'] == key[0])
            & (eph_df['hidden_dim'] == key[1])
            & (eph_df['expansion_dim'] == key[2])
        ]
        best = candidates.loc[candidates['avg_ms'].idxmin()]
        rows.append(
            BestShapeRow(
                batch_tokens=key[0],
                hidden_dim=key[1],
                expansion_dim=key[2],
                dense_mean_ms=float(dense_row['avg_ms']),
                dense_std_ms=float(dense_row.get('std_ms', 0.0)),
                best_ephemeral_mean_ms=float(best['avg_ms']),
                best_ephemeral_std_ms=float(best.get('std_ms', 0.0)),
                best_rank=int(best['rank']),
                speedup_dense_over_ephemeral=float(dense_row['avg_ms']) / float(best['avg_ms']),
            )
        )
    return sorted(rows, key=lambda x: (x.batch_tokens, x.hidden_dim))


def write_markdown(best_rows: list[BestShapeRow], path: Path) -> None:
    lines = [
        '# Ascend 910B ai2 repeatability summary',
        '',
        'Repeated-run summary from `results/ai2_torch_npu_microbench.csv`.',
        '',
        '## Best dense vs best ephemeral by shape',
        '',
    ]
    for row in best_rows:
        lines.append(
            f"- bt={row.batch_tokens}, d={row.hidden_dim}, m={row.expansion_dim}: "
            f"dense={row.dense_mean_ms:.6f}±{row.dense_std_ms:.6f} ms, "
            f"best_ephemeral={row.best_ephemeral_mean_ms:.6f}±{row.best_ephemeral_std_ms:.6f} ms "
            f"at r={row.best_rank}, speedup(dense/eph)={row.speedup_dense_over_ephemeral:.3f}"
        )

    wins = [row for row in best_rows if row.speedup_dense_over_ephemeral > 1.0]
    losses = [row for row in best_rows if row.speedup_dense_over_ephemeral <= 1.0]
    lines.extend(
        [
            '',
            '## Interpretation',
            '',
            f'- Best-ephemeral wins in {len(wins)} of {len(best_rows)} tested shape buckets.',
            f'- Best-ephemeral losses remain in {len(losses)} of {len(best_rows)} tested shape buckets.',
        ]
    )
    if wins:
        strongest = max(wins, key=lambda x: x.speedup_dense_over_ephemeral)
        lines.append(
            f'- Strongest measured win: bt={strongest.batch_tokens}, d={strongest.hidden_dim}, '
            f'm={strongest.expansion_dim}, r={strongest.best_rank}, '
            f'speedup={strongest.speedup_dense_over_ephemeral:.3f}.'
        )

    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot_heatmap(best_rows: list[BestShapeRow], path: Path) -> None:
    df = pd.DataFrame([asdict(row) for row in best_rows])
    pivot = df.pivot(index='hidden_dim', columns='batch_tokens', values='speedup_dense_over_ephemeral')
    annot = df.pivot(index='hidden_dim', columns='batch_tokens', values='best_rank')

    pivot = pivot.sort_index()
    annot = annot.sort_index()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(pivot.to_numpy(), cmap='RdYlGn', aspect='auto', vmin=pivot.min().min(), vmax=pivot.max().max())
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Dense / Best-Ephemeral Speedup')

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([str(x) for x in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(x) for x in pivot.index])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            speedup = pivot.iloc[i, j]
            rank = int(annot.iloc[i, j])
            ax.text(j, i, f'{speedup:.2f}\nr={rank}', ha='center', va='center', fontsize=9)

    plt.title('Best Measured Ephemeral Speedup on Ascend 910B\nCell annotation = best rank')
    plt.xlabel('Batch Tokens')
    plt.ylabel('Hidden Dimension')
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


if __name__ == '__main__':
    summary_path = Path('results/ai2_torch_npu_microbench.csv')
    summary = pd.read_csv(summary_path)
    best_rows = build_best_rows(summary)

    save_csv(best_rows, Path('results/ai2_best_shape_summary.csv'))
    write_markdown(best_rows, Path('results/ai2_repeatability_summary.md'))
    plot_heatmap(best_rows, Path('figures/ai2_best_speedup_heatmap.png'))

    print('wrote results/ai2_best_shape_summary.csv')
    print('wrote results/ai2_repeatability_summary.md')
    print('wrote figures/ai2_best_speedup_heatmap.png')
