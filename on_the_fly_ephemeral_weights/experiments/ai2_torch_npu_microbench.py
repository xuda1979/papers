#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, stdev

import torch

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except Exception:
    HAS_NPU = False


@dataclass
class BenchRow:
    trial: int
    device: str
    mode: str
    batch_tokens: int
    hidden_dim: int
    expansion_dim: int
    rank: int
    dtype: str
    avg_ms: float
    tflops: float
    bytes_gb: float
    arithmetic_intensity: float


@dataclass
class SummaryRow:
    device: str
    mode: str
    batch_tokens: int
    hidden_dim: int
    expansion_dim: int
    rank: int
    dtype: str
    trials: int
    avg_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    ci95_ms: float
    tflops: float
    bytes_gb: float
    arithmetic_intensity: float


def sync_if_needed(device: str) -> None:
    if device.startswith('npu') and HAS_NPU:
        torch.npu.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def timed(fn, device: str, warmup: int = 5, reps: int = 20) -> float:
    for _ in range(warmup):
        fn()
    sync_if_needed(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync_if_needed(device)
    return (time.perf_counter() - t0) / reps


def dense_case(trial: int, device: str, bt: int, d: int, m: int, dtype: torch.dtype, warmup: int, reps: int) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    w = torch.randn(d, m, device=device, dtype=dtype)

    def fn():
        return x @ w

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    flops = 2.0 * bt * d * m
    bytes_moved = 2.0 * (bt * d + d * m + bt * m)
    return BenchRow(
        trial=trial,
        device=device,
        mode='dense',
        batch_tokens=bt,
        hidden_dim=d,
        expansion_dim=m,
        rank=0,
        dtype=str(dtype).replace('torch.', ''),
        avg_ms=dt * 1e3,
        tflops=flops / dt / 1e12,
        bytes_gb=bytes_moved / 1e9,
        arithmetic_intensity=flops / bytes_moved,
    )


def ephemeral_case(
    trial: int,
    device: str,
    bt: int,
    d: int,
    m: int,
    r: int,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    u = torch.randn(d, r, device=device, dtype=dtype)
    v = torch.randn(r, m, device=device, dtype=dtype)

    def fn():
        return (x @ u) @ v

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    flops = 2.0 * bt * d * r + 2.0 * bt * r * m
    bytes_moved = 2.0 * (bt * d + d * r + r * m + bt * r + bt * m)
    return BenchRow(
        trial=trial,
        device=device,
        mode='ephemeral',
        batch_tokens=bt,
        hidden_dim=d,
        expansion_dim=m,
        rank=r,
        dtype=str(dtype).replace('torch.', ''),
        avg_ms=dt * 1e3,
        tflops=flops / dt / 1e12,
        bytes_gb=bytes_moved / 1e9,
        arithmetic_intensity=flops / bytes_moved,
    )


def save(rows: list[object], path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty row set to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(',') if x]


def set_trial_seed(base_seed: int, trial: int) -> None:
    seed = base_seed + trial
    torch.manual_seed(seed)
    if HAS_NPU:
        torch.npu.manual_seed_all(seed)


def summarize_rows(rows: list[BenchRow]) -> list[SummaryRow]:
    grouped: dict[tuple[str, str, int, int, int, int, str], list[BenchRow]] = defaultdict(list)
    for row in rows:
        key = (
            row.device,
            row.mode,
            row.batch_tokens,
            row.hidden_dim,
            row.expansion_dim,
            row.rank,
            row.dtype,
        )
        grouped[key].append(row)

    out: list[SummaryRow] = []
    for key in sorted(grouped):
        items = grouped[key]
        ms_values = [x.avg_ms for x in items]
        avg_ms = mean(ms_values)
        std_ms = stdev(ms_values) if len(ms_values) > 1 else 0.0
        ci95_ms = 1.96 * std_ms / math.sqrt(len(ms_values)) if len(ms_values) > 1 else 0.0
        exemplar = items[0]
        avg_seconds = avg_ms / 1e3
        flops = exemplar.tflops * (exemplar.avg_ms / 1e3) * 1e12
        out.append(
            SummaryRow(
                device=exemplar.device,
                mode=exemplar.mode,
                batch_tokens=exemplar.batch_tokens,
                hidden_dim=exemplar.hidden_dim,
                expansion_dim=exemplar.expansion_dim,
                rank=exemplar.rank,
                dtype=exemplar.dtype,
                trials=len(items),
                avg_ms=avg_ms,
                std_ms=std_ms,
                min_ms=min(ms_values),
                max_ms=max(ms_values),
                ci95_ms=ci95_ms,
                tflops=flops / avg_seconds / 1e12,
                bytes_gb=exemplar.bytes_gb,
                arithmetic_intensity=exemplar.arithmetic_intensity,
            )
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dims', default='1024,2048,4096')
    parser.add_argument('--batch-tokens', default='1,8,32')
    parser.add_argument('--ranks', default='16,32,64,128')
    parser.add_argument('--expansion-factor', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--summary-output', default='results/ai2_torch_npu_microbench.csv')
    parser.add_argument('--raw-output', default='results/ai2_torch_npu_microbench_trials.csv')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    device = 'npu:0' if HAS_NPU else 'cpu'
    dtype = torch.float16 if HAS_NPU else torch.float32

    rows: list[BenchRow] = []
    hidden_dims = parse_int_list(args.hidden_dims)
    batch_tokens_list = parse_int_list(args.batch_tokens)
    ranks = parse_int_list(args.ranks)

    print('device =', device)
    print('dtype =', dtype)
    print('trials =', args.trials)
    print('warmup =', args.warmup)
    print('reps =', args.reps)

    for trial in range(args.trials):
        set_trial_seed(args.seed, trial)
        for d in hidden_dims:
            m = d * args.expansion_factor
            for bt in batch_tokens_list:
                rows.append(dense_case(trial, device, bt, d, m, dtype, args.warmup, args.reps))
                for r in ranks:
                    rows.append(ephemeral_case(trial, device, bt, d, m, r, dtype, args.warmup, args.reps))
                print('finished trial=', trial, 'd=', d, 'bt=', bt)

    raw_out = Path(args.raw_output)
    summary_out = Path(args.summary_output)
    save(rows, raw_out)
    print('wrote', raw_out)
    summary_rows = summarize_rows(rows)
    save(summary_rows, summary_out)
    print('wrote', summary_out)
    best = min(summary_rows, key=lambda x: x.avg_ms)
    print('fastest summary row:', asdict(best))
