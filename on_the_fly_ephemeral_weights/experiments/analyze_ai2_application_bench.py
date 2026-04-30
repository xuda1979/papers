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
class ScenarioBestRow:
    scenario: str
    hidden_dim: int
    batch_tokens: int
    best_static_rank: int
    best_static_speedup: float
    best_dynamic_rank: int
    best_dynamic_speedup: float


def save_csv(rows: list[ScenarioBestRow], path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty scenario summary to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def build_best_rows(df: pd.DataFrame) -> list[ScenarioBestRow]:
    rows: list[ScenarioBestRow] = []
    for scenario in sorted(df['scenario'].unique()):
        for d in sorted(df['hidden_dim'].unique()):
            for bt in sorted(df['batch_tokens'].unique()):
                sub = df[
                    (df['scenario'] == scenario)
                    & (df['hidden_dim'] == d)
                    & (df['batch_tokens'] == bt)
                ]
                dense = sub[sub['mode'] == 'dense'].iloc[0]
                static_rows = sub[sub['mode'] == 'static_ephemeral']
                dynamic_rows = sub[sub['mode'] == 'dynamic_ephemeral']
                best_static = static_rows.loc[static_rows['avg_ms'].idxmin()]
                best_dynamic = dynamic_rows.loc[dynamic_rows['avg_ms'].idxmin()]
                rows.append(
                    ScenarioBestRow(
                        scenario=scenario,
                        hidden_dim=int(d),
                        batch_tokens=int(bt),
                        best_static_rank=int(best_static['rank']),
                        best_static_speedup=float(dense['avg_ms']) / float(best_static['avg_ms']),
                        best_dynamic_rank=int(best_dynamic['rank']),
                        best_dynamic_speedup=float(dense['avg_ms']) / float(best_dynamic['avg_ms']),
                    )
                )
    return rows


def write_markdown(rows: list[ScenarioBestRow], path: Path) -> None:
    static_wins = sum(row.best_static_speedup > 1.0 for row in rows)
    dynamic_wins = sum(row.best_dynamic_speedup > 1.0 for row in rows)
    lines = [
        '# Ascend 910B ai2 application-scenario summary',
        '',
        'Repeated scenario benchmark comparing dense, static ephemeral, and dynamic ephemeral variants.',
        '',
        f'- Static ephemeral beats dense in {static_wins}/{len(rows)} scenario-shape buckets.',
        f'- Dynamic ephemeral beats dense in {dynamic_wins}/{len(rows)} scenario-shape buckets.',
        '',
        '## Best static and dynamic speedups by shape',
        '',
    ]
    for row in rows:
        lines.append(
            f"- scenario={row.scenario}, d={row.hidden_dim}, bt={row.batch_tokens}: "
            f"best_static r={row.best_static_rank}, speedup={row.best_static_speedup:.3f}; "
            f"best_dynamic r={row.best_dynamic_rank}, speedup={row.best_dynamic_speedup:.3f}"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def plot(rows: list[ScenarioBestRow], path: Path) -> None:
    df = pd.DataFrame([asdict(row) for row in rows])
    scenarios = sorted(df['scenario'].unique())
    hidden_dims = sorted(df['hidden_dim'].unique())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6.2 * len(scenarios), 4.8), sharey=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        sub = df[df['scenario'] == scenario]
        for bt in sorted(sub['batch_tokens'].unique()):
            bt_sub = sub[sub['batch_tokens'] == bt].sort_values('hidden_dim')
            ax.plot(bt_sub['hidden_dim'], bt_sub['best_static_speedup'], marker='o', label=f'static, b={bt}')
            ax.plot(bt_sub['hidden_dim'], bt_sub['best_dynamic_speedup'], marker='s', linestyle='--', label=f'dynamic, b={bt}')
        ax.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax.set_title(scenario.replace('_', ' ').upper())
        ax.set_xlabel('Hidden Dimension')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Dense / Ephemeral Mean-Latency Speedup')
    axes[0].legend(fontsize=8)
    fig.suptitle('Ascend 910B Application Scenarios')
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    df = pd.read_csv('results/ai2_application_bench.csv')
    rows = build_best_rows(df)
    save_csv(rows, Path('results/ai2_application_best.csv'))
    write_markdown(rows, Path('results/ai2_application_summary.md'))
    plot(rows, Path('figures/ai2_application_scenarios.png'))
    print('wrote results/ai2_application_best.csv')
    print('wrote results/ai2_application_summary.md')
    print('wrote figures/ai2_application_scenarios.png')
