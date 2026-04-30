#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd


@dataclass
class CalibratedRow:
    batch_tokens: int
    hidden_dim: int
    expansion_dim: int
    rank: int
    dense_ms_sim: float
    eph_ms_sim: float
    dense_ms_meas: float
    eph_ms_meas: float
    dense_error_ratio: float
    eph_error_ratio: float
    measured_speedup_dense_over_eph: float
    simulated_speedup_dense_over_eph: float


def save(rows: list[CalibratedRow], path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty calibration table to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


if __name__ == '__main__':
    sim = pd.read_csv('results/ephemeral_weight_benchmark.csv')
    meas = pd.read_csv('results/ai2_torch_npu_microbench.csv')

    sim = sim[sim['hardware'] == 'Ascend_910B_like'].copy()
    meas = meas.copy()

    out: list[CalibratedRow] = []
    for _, dense in meas[meas['mode'] == 'dense'].iterrows():
        bt = int(dense['batch_tokens'])
        d = int(dense['hidden_dim'])
        m = int(dense['expansion_dim'])

        sim_dense = sim[
            (sim['batch_tokens'] == bt)
            & (sim['hidden_dim'] == d)
            & (sim['expansion_dim'] == m)
        ].iloc[0]

        eph_rows = meas[
            (meas['mode'] == 'ephemeral')
            & (meas['batch_tokens'] == bt)
            & (meas['hidden_dim'] == d)
            & (meas['expansion_dim'] == m)
        ]

        for _, eph in eph_rows.iterrows():
            r = int(eph['rank'])
            sim_eph = sim[
                (sim['batch_tokens'] == bt)
                & (sim['hidden_dim'] == d)
                & (sim['expansion_dim'] == m)
                & (sim['rank'] == r)
                & (sim['generator_flop_multiplier'] == 128.0)
            ].iloc[0]

            out.append(
                CalibratedRow(
                    batch_tokens=bt,
                    hidden_dim=d,
                    expansion_dim=m,
                    rank=r,
                    dense_ms_sim=float(sim_dense['dense_latency_ms']),
                    eph_ms_sim=float(sim_eph['ephemeral_latency_ms']),
                    dense_ms_meas=float(dense['avg_ms']),
                    eph_ms_meas=float(eph['avg_ms']),
                    dense_error_ratio=float(dense['avg_ms']) / float(sim_dense['dense_latency_ms']),
                    eph_error_ratio=float(eph['avg_ms']) / float(sim_eph['ephemeral_latency_ms']),
                    measured_speedup_dense_over_eph=float(dense['avg_ms']) / float(eph['avg_ms']),
                    simulated_speedup_dense_over_eph=float(sim_dense['dense_latency_ms']) / float(sim_eph['ephemeral_latency_ms']),
                )
            )

    save(out, Path('results/910b_calibration_table.csv'))
    df = pd.DataFrame([asdict(x) for x in out])
    print('wrote results/910b_calibration_table.csv')
    print(df.groupby(['batch_tokens','hidden_dim'])[['measured_speedup_dense_over_eph','simulated_speedup_dense_over_eph']].max().to_string())
