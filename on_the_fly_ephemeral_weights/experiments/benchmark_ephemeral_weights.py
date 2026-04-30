#!/usr/bin/env python3
from __future__ import annotations

import csv
import itertools
from dataclasses import dataclass, asdict
from pathlib import Path

BYTES_PER_PARAM = 2.0


@dataclass
class Hardware:
    name: str
    peak_tflops: float
    hbm_tbps: float


@dataclass
class Result:
    hardware: str
    hidden_dim: int
    expansion_dim: int
    batch_tokens: int
    rank: int
    generator_flop_multiplier: float
    dense_hbm_gb: float
    ephemeral_hbm_gb: float
    dense_tflops: float
    ephemeral_tflops: float
    dense_latency_ms: float
    ephemeral_latency_ms: float
    speedup: float


def roofline_latency_ms(flops: float, bytes_hbm: float, hw: Hardware) -> float:
    compute_ms = flops / (hw.peak_tflops * 1e12) * 1e3
    bandwidth_ms = bytes_hbm / (hw.hbm_tbps * 1e12) * 1e3
    return max(compute_ms, bandwidth_ms)


def dense_cost(batch_tokens: int, d: int, m: int) -> tuple[float, float]:
    flops = 2.0 * batch_tokens * d * m
    bytes_hbm = d * m * BYTES_PER_PARAM
    return flops, bytes_hbm


def ephemeral_cost(batch_tokens: int, d: int, m: int, r: int, generator_flop_multiplier: float) -> tuple[float, float]:
    core_flops = 2.0 * batch_tokens * d * r + 2.0 * batch_tokens * r * m
    generator_flops = generator_flop_multiplier * (d + m) * r
    total_flops = core_flops + generator_flops
    bytes_hbm = (d * r + r * m) * BYTES_PER_PARAM
    return total_flops, bytes_hbm


def run() -> list[Result]:
    hardwares = [
        Hardware('RTX4090_like', peak_tflops=330.0, hbm_tbps=1.0),
        Hardware('A100_like', peak_tflops=312.0, hbm_tbps=2.0),
        Hardware('H100_like', peak_tflops=990.0, hbm_tbps=3.35),
        Hardware('B200_like', peak_tflops=2250.0, hbm_tbps=8.0),
        Hardware('Ascend_910B_like', peak_tflops=312.0, hbm_tbps=1.6), # 910B approximation: ~312 TFLOPS BF16/FP16, ~1.6 TB/s HBM
    ]
    hidden_dims = [1024, 2048, 4096, 8192, 16384]
    expansion_factors = [4]
    batch_tokens_list = [1, 8, 32, 128]
    ranks = [16, 32, 64, 128, 256]
    generator_multipliers = [32.0, 128.0, 512.0]

    out: list[Result] = []
    for hw, d, ef, batch_tokens, r, gm in itertools.product(
        hardwares, hidden_dims, expansion_factors, batch_tokens_list, ranks, generator_multipliers
    ):
        m = d * ef
        dense_flops, dense_bytes = dense_cost(batch_tokens, d, m)
        eph_flops, eph_bytes = ephemeral_cost(batch_tokens, d, m, r, gm)
        dense_latency = roofline_latency_ms(dense_flops, dense_bytes, hw)
        eph_latency = roofline_latency_ms(eph_flops, eph_bytes, hw)
        out.append(
            Result(
                hardware=hw.name,
                hidden_dim=d,
                expansion_dim=m,
                batch_tokens=batch_tokens,
                rank=r,
                generator_flop_multiplier=gm,
                dense_hbm_gb=dense_bytes / 1e9,
                ephemeral_hbm_gb=eph_bytes / 1e9,
                dense_tflops=dense_flops / 1e12,
                ephemeral_tflops=eph_flops / 1e12,
                dense_latency_ms=dense_latency,
                ephemeral_latency_ms=eph_latency,
                speedup=dense_latency / eph_latency if eph_latency > 0 else 0.0,
            )
        )
    return out


def save_csv(rows: list[Result], path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty benchmark result set to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


if __name__ == '__main__':
    rows = run()
    save_csv(rows, Path('results/ephemeral_weight_benchmark.csv'))
    best = max(rows, key=lambda x: x.speedup)
    print('wrote results/ephemeral_weight_benchmark.csv')
    print('best case:', asdict(best))
