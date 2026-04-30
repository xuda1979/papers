#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


def read_device_clock_ghz() -> float:
    try:
        with os.popen("npu-smi info | grep -m1 -E 'AICore\(MHz\)|AICore\(mhz\)'") as f:
            text = f.read()
        nums = [float(tok) for tok in text.replace('|', ' ').replace(':', ' ').split() if tok.replace('.', '', 1).isdigit()]
        if nums:
            mhz = max(nums)
            return mhz / 1000.0
    except Exception:
        pass
    return 1.2


@dataclass
class CaseResult:
    mode: str
    batch_tokens: int
    hidden_dim: int
    expansion_dim: int
    rank: int
    dtype: str
    measured_ms: float
    approx_tflops: float
    bytes_moved_gb: float
    arithmetic_intensity: float
    notes: str


def benchmark_numpy_dense(bt: int, d: int, m: int, reps: int = 10) -> CaseResult:
    x = np.random.randn(bt, d).astype(np.float32)
    w = np.random.randn(d, m).astype(np.float32)
    _ = x @ w
    t0 = time.perf_counter()
    for _ in range(reps):
        y = x @ w
    dt = (time.perf_counter() - t0) / reps
    flops = 2.0 * bt * d * m
    bytes_moved = 4.0 * (bt * d + d * m + bt * m)
    return CaseResult(
        mode='dense_numpy_cpu',
        batch_tokens=bt,
        hidden_dim=d,
        expansion_dim=m,
        rank=0,
        dtype='fp32',
        measured_ms=dt * 1e3,
        approx_tflops=flops / dt / 1e12,
        bytes_moved_gb=bytes_moved / 1e9,
        arithmetic_intensity=flops / bytes_moved,
        notes='CPU fallback only; remote NPU package stack unavailable or not yet used',
    )


def benchmark_numpy_ephemeral(bt: int, d: int, m: int, r: int, reps: int = 10) -> CaseResult:
    x = np.random.randn(bt, d).astype(np.float32)
    u = np.random.randn(d, r).astype(np.float32)
    v = np.random.randn(r, m).astype(np.float32)
    _ = (x @ u) @ v
    t0 = time.perf_counter()
    for _ in range(reps):
        y = (x @ u) @ v
    dt = (time.perf_counter() - t0) / reps
    flops = 2.0 * bt * d * r + 2.0 * bt * r * m
    bytes_moved = 4.0 * (bt * d + d * r + r * m + bt * r + bt * m)
    return CaseResult(
        mode='ephemeral_numpy_cpu',
        batch_tokens=bt,
        hidden_dim=d,
        expansion_dim=m,
        rank=r,
        dtype='fp32',
        measured_ms=dt * 1e3,
        approx_tflops=flops / dt / 1e12,
        bytes_moved_gb=bytes_moved / 1e9,
        arithmetic_intensity=flops / bytes_moved,
        notes='CPU fallback only; remote NPU package stack unavailable or not yet used',
    )


def run() -> list[CaseResult]:
    rows: list[CaseResult] = []
    hidden_dims = [1024, 2048, 4096]
    batch_tokens_list = [1, 8, 32]
    ranks = [16, 32, 64, 128]
    expansion_factor = 4

    for d in hidden_dims:
        m = d * expansion_factor
        for bt in batch_tokens_list:
            rows.append(benchmark_numpy_dense(bt, d, m))
            for r in ranks:
                rows.append(benchmark_numpy_ephemeral(bt, d, m, r))
    return rows


def save(rows: list[CaseResult], out_path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty microbench result set to {out_path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


if __name__ == '__main__':
    rows = run()
    save(rows, Path('results/ai2_microbench.csv'))
    print(f'wrote results/ai2_microbench.csv with {len(rows)} rows')
    best = min(rows, key=lambda x: x.measured_ms)
    print('fastest case:', asdict(best))
