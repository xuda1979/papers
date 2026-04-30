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
import torch.nn.functional as F

try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except Exception:
    HAS_NPU = False


@dataclass
class BenchRow:
    trial: int
    scenario: str
    mode: str
    batch_tokens: int
    hidden_dim: int
    expansion_dim: int
    rank: int
    basis_size: int
    dtype: str
    avg_ms: float


@dataclass
class SummaryRow:
    scenario: str
    mode: str
    batch_tokens: int
    hidden_dim: int
    expansion_dim: int
    rank: int
    basis_size: int
    dtype: str
    trials: int
    avg_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    ci95_ms: float


def parse_int_list(value: str) -> list[int]:
    return [int(x) for x in value.split(',') if x]


def sync_if_needed(device: str) -> None:
    if device.startswith('npu') and HAS_NPU:
        torch.npu.synchronize()
    elif device.startswith('cuda'):
        torch.cuda.synchronize()


def timed(fn, device: str, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    sync_if_needed(device)
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync_if_needed(device)
    return (time.perf_counter() - t0) / reps


def set_trial_seed(base_seed: int, trial: int) -> None:
    seed = base_seed + trial
    torch.manual_seed(seed)
    if HAS_NPU:
        torch.npu.manual_seed_all(seed)


def summarize_rows(rows: list[BenchRow]) -> list[SummaryRow]:
    grouped: dict[tuple[str, str, int, int, int, int, int, str], list[BenchRow]] = defaultdict(list)
    for row in rows:
        key = (
            row.scenario,
            row.mode,
            row.batch_tokens,
            row.hidden_dim,
            row.expansion_dim,
            row.rank,
            row.basis_size,
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
        out.append(
            SummaryRow(
                scenario=exemplar.scenario,
                mode=exemplar.mode,
                batch_tokens=exemplar.batch_tokens,
                hidden_dim=exemplar.hidden_dim,
                expansion_dim=exemplar.expansion_dim,
                rank=exemplar.rank,
                basis_size=exemplar.basis_size,
                dtype=exemplar.dtype,
                trials=len(items),
                avg_ms=avg_ms,
                std_ms=std_ms,
                min_ms=min(ms_values),
                max_ms=max(ms_values),
                ci95_ms=ci95_ms,
            )
        )
    return out


def save(rows: list[object], path: Path) -> None:
    if not rows:
        raise ValueError(f'cannot save empty row set to {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def dense_ffn_case(
    trial: int,
    device: str,
    bt: int,
    d: int,
    m: int,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    w_up = torch.randn(d, m, device=device, dtype=dtype)
    w_down = torch.randn(m, d, device=device, dtype=dtype)

    def fn():
        return F.gelu(x @ w_up) @ w_down

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'ffn', 'dense', bt, d, m, 0, 0, str(dtype).replace('torch.', ''), dt * 1e3)


def static_ephemeral_ffn_case(
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
    u_up = torch.randn(d, r, device=device, dtype=dtype)
    v_up = torch.randn(r, m, device=device, dtype=dtype)
    u_down = torch.randn(m, r, device=device, dtype=dtype)
    v_down = torch.randn(r, d, device=device, dtype=dtype)

    def fn():
        h = F.gelu((x @ u_up) @ v_up)
        return (h @ u_down) @ v_down

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'ffn', 'static_ephemeral', bt, d, m, r, 0, str(dtype).replace('torch.', ''), dt * 1e3)


def dynamic_ephemeral_ffn_case(
    trial: int,
    device: str,
    bt: int,
    d: int,
    m: int,
    r: int,
    basis_size: int,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    gen_up = torch.randn(d, basis_size, device=device, dtype=dtype)
    gen_down = torch.randn(d, basis_size, device=device, dtype=dtype)
    basis_u_up = torch.randn(basis_size, d * r, device=device, dtype=dtype)
    basis_v_up = torch.randn(basis_size, r * m, device=device, dtype=dtype)
    basis_u_down = torch.randn(basis_size, m * r, device=device, dtype=dtype)
    basis_v_down = torch.randn(basis_size, r * d, device=device, dtype=dtype)

    def fn():
        pooled = x.mean(dim=0)
        coeff_up = pooled @ gen_up
        coeff_down = pooled @ gen_down
        u_up = (coeff_up @ basis_u_up).reshape(d, r)
        v_up = (coeff_up @ basis_v_up).reshape(r, m)
        u_down = (coeff_down @ basis_u_down).reshape(m, r)
        v_down = (coeff_down @ basis_v_down).reshape(r, d)
        h = F.gelu((x @ u_up) @ v_up)
        return (h @ u_down) @ v_down

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'ffn', 'dynamic_ephemeral', bt, d, m, r, basis_size, str(dtype).replace('torch.', ''), dt * 1e3)


def dense_swiglu_case(
    trial: int,
    device: str,
    bt: int,
    d: int,
    m: int,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    w_up = torch.randn(d, m, device=device, dtype=dtype)
    w_gate = torch.randn(d, m, device=device, dtype=dtype)
    w_down = torch.randn(m, d, device=device, dtype=dtype)

    def fn():
        up = x @ w_up
        gate = x @ w_gate
        return (F.silu(gate) * up) @ w_down

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'swiglu', 'dense', bt, d, m, 0, 0, str(dtype).replace('torch.', ''), dt * 1e3)


def static_ephemeral_swiglu_case(
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
    u_up = torch.randn(d, r, device=device, dtype=dtype)
    v_up = torch.randn(r, m, device=device, dtype=dtype)
    u_gate = torch.randn(d, r, device=device, dtype=dtype)
    v_gate = torch.randn(r, m, device=device, dtype=dtype)
    u_down = torch.randn(m, r, device=device, dtype=dtype)
    v_down = torch.randn(r, d, device=device, dtype=dtype)

    def fn():
        up = (x @ u_up) @ v_up
        gate = (x @ u_gate) @ v_gate
        return ((F.silu(gate) * up) @ u_down) @ v_down

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'swiglu', 'static_ephemeral', bt, d, m, r, 0, str(dtype).replace('torch.', ''), dt * 1e3)


def dynamic_ephemeral_swiglu_case(
    trial: int,
    device: str,
    bt: int,
    d: int,
    m: int,
    r: int,
    basis_size: int,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    gen_up = torch.randn(d, basis_size, device=device, dtype=dtype)
    gen_gate = torch.randn(d, basis_size, device=device, dtype=dtype)
    gen_down = torch.randn(d, basis_size, device=device, dtype=dtype)

    basis_u_up = torch.randn(basis_size, d * r, device=device, dtype=dtype)
    basis_v_up = torch.randn(basis_size, r * m, device=device, dtype=dtype)
    basis_u_gate = torch.randn(basis_size, d * r, device=device, dtype=dtype)
    basis_v_gate = torch.randn(basis_size, r * m, device=device, dtype=dtype)
    basis_u_down = torch.randn(basis_size, m * r, device=device, dtype=dtype)
    basis_v_down = torch.randn(basis_size, r * d, device=device, dtype=dtype)

    def fn():
        pooled = x.mean(dim=0)
        coeff_up = pooled @ gen_up
        coeff_gate = pooled @ gen_gate
        coeff_down = pooled @ gen_down

        u_up = (coeff_up @ basis_u_up).reshape(d, r)
        v_up = (coeff_up @ basis_v_up).reshape(r, m)
        u_gate = (coeff_gate @ basis_u_gate).reshape(d, r)
        v_gate = (coeff_gate @ basis_v_gate).reshape(r, m)
        u_down = (coeff_down @ basis_u_down).reshape(m, r)
        v_down = (coeff_down @ basis_v_down).reshape(r, d)

        up = (x @ u_up) @ v_up
        gate = (x @ u_gate) @ v_gate
        return ((F.silu(gate) * up) @ u_down) @ v_down

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'swiglu', 'dynamic_ephemeral', bt, d, m, r, basis_size, str(dtype).replace('torch.', ''), dt * 1e3)


def dense_moe_top2_case(
    trial: int,
    device: str,
    bt: int,
    d: int,
    m: int,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    router = torch.randn(d, 2, device=device, dtype=dtype)
    w_up_1 = torch.randn(d, m, device=device, dtype=dtype)
    w_down_1 = torch.randn(m, d, device=device, dtype=dtype)
    w_up_2 = torch.randn(d, m, device=device, dtype=dtype)
    w_down_2 = torch.randn(m, d, device=device, dtype=dtype)

    def fn():
        probs = F.softmax(x @ router, dim=-1)
        y1 = F.gelu(x @ w_up_1) @ w_down_1
        y2 = F.gelu(x @ w_up_2) @ w_down_2
        return probs[:, :1] * y1 + probs[:, 1:2] * y2

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'moe_top2', 'dense', bt, d, m, 0, 0, str(dtype).replace('torch.', ''), dt * 1e3)


def static_ephemeral_moe_top2_case(
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
    router = torch.randn(d, 2, device=device, dtype=dtype)
    u_up_1 = torch.randn(d, r, device=device, dtype=dtype)
    v_up_1 = torch.randn(r, m, device=device, dtype=dtype)
    u_down_1 = torch.randn(m, r, device=device, dtype=dtype)
    v_down_1 = torch.randn(r, d, device=device, dtype=dtype)
    u_up_2 = torch.randn(d, r, device=device, dtype=dtype)
    v_up_2 = torch.randn(r, m, device=device, dtype=dtype)
    u_down_2 = torch.randn(m, r, device=device, dtype=dtype)
    v_down_2 = torch.randn(r, d, device=device, dtype=dtype)

    def fn():
        probs = F.softmax(x @ router, dim=-1)
        y1 = ((F.gelu((x @ u_up_1) @ v_up_1) @ u_down_1) @ v_down_1)
        y2 = ((F.gelu((x @ u_up_2) @ v_up_2) @ u_down_2) @ v_down_2)
        return probs[:, :1] * y1 + probs[:, 1:2] * y2

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'moe_top2', 'static_ephemeral', bt, d, m, r, 0, str(dtype).replace('torch.', ''), dt * 1e3)


def dynamic_ephemeral_moe_top2_case(
    trial: int,
    device: str,
    bt: int,
    d: int,
    m: int,
    r: int,
    basis_size: int,
    dtype: torch.dtype,
    warmup: int,
    reps: int,
) -> BenchRow:
    x = torch.randn(bt, d, device=device, dtype=dtype)
    router = torch.randn(d, 2, device=device, dtype=dtype)
    gen_up_1 = torch.randn(d, basis_size, device=device, dtype=dtype)
    gen_down_1 = torch.randn(d, basis_size, device=device, dtype=dtype)
    gen_up_2 = torch.randn(d, basis_size, device=device, dtype=dtype)
    gen_down_2 = torch.randn(d, basis_size, device=device, dtype=dtype)
    basis_u_up_1 = torch.randn(basis_size, d * r, device=device, dtype=dtype)
    basis_v_up_1 = torch.randn(basis_size, r * m, device=device, dtype=dtype)
    basis_u_down_1 = torch.randn(basis_size, m * r, device=device, dtype=dtype)
    basis_v_down_1 = torch.randn(basis_size, r * d, device=device, dtype=dtype)
    basis_u_up_2 = torch.randn(basis_size, d * r, device=device, dtype=dtype)
    basis_v_up_2 = torch.randn(basis_size, r * m, device=device, dtype=dtype)
    basis_u_down_2 = torch.randn(basis_size, m * r, device=device, dtype=dtype)
    basis_v_down_2 = torch.randn(basis_size, r * d, device=device, dtype=dtype)

    def fn():
        pooled = x.mean(dim=0)
        probs = F.softmax(x @ router, dim=-1)
        coeff_up_1 = pooled @ gen_up_1
        coeff_down_1 = pooled @ gen_down_1
        coeff_up_2 = pooled @ gen_up_2
        coeff_down_2 = pooled @ gen_down_2

        u_up_1 = (coeff_up_1 @ basis_u_up_1).reshape(d, r)
        v_up_1 = (coeff_up_1 @ basis_v_up_1).reshape(r, m)
        u_down_1 = (coeff_down_1 @ basis_u_down_1).reshape(m, r)
        v_down_1 = (coeff_down_1 @ basis_v_down_1).reshape(r, d)
        u_up_2 = (coeff_up_2 @ basis_u_up_2).reshape(d, r)
        v_up_2 = (coeff_up_2 @ basis_v_up_2).reshape(r, m)
        u_down_2 = (coeff_down_2 @ basis_u_down_2).reshape(m, r)
        v_down_2 = (coeff_down_2 @ basis_v_down_2).reshape(r, d)

        y1 = ((F.gelu((x @ u_up_1) @ v_up_1) @ u_down_1) @ v_down_1)
        y2 = ((F.gelu((x @ u_up_2) @ v_up_2) @ u_down_2) @ v_down_2)
        return probs[:, :1] * y1 + probs[:, 1:2] * y2

    dt = timed(fn, device=device, warmup=warmup, reps=reps)
    return BenchRow(trial, 'moe_top2', 'dynamic_ephemeral', bt, d, m, r, basis_size, str(dtype).replace('torch.', ''), dt * 1e3)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden-dims', default='2048,4096,8192')
    parser.add_argument('--batch-tokens', default='1,8,32')
    parser.add_argument('--ranks', default='16,32,64')
    parser.add_argument('--expansion-factor', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--trials', type=int, default=3)
    parser.add_argument('--seed', type=int, default=20260326)
    parser.add_argument('--basis-size', type=int, default=8)
    parser.add_argument('--summary-output', default='results/ai2_application_bench.csv')
    parser.add_argument('--raw-output', default='results/ai2_application_bench_trials.csv')
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    device = 'npu:0' if HAS_NPU else 'cpu'
    dtype = torch.float16 if HAS_NPU else torch.float32
    hidden_dims = parse_int_list(args.hidden_dims)
    batch_tokens_list = parse_int_list(args.batch_tokens)
    ranks = parse_int_list(args.ranks)

    print('device =', device)
    print('dtype =', dtype)
    print('trials =', args.trials)
    print('basis_size =', args.basis_size)

    rows: list[BenchRow] = []
    for trial in range(args.trials):
        set_trial_seed(args.seed, trial)
        for d in hidden_dims:
            m = d * args.expansion_factor
            for bt in batch_tokens_list:
                rows.append(dense_ffn_case(trial, device, bt, d, m, dtype, args.warmup, args.reps))
                rows.append(dense_swiglu_case(trial, device, bt, d, m, dtype, args.warmup, args.reps))
                rows.append(dense_moe_top2_case(trial, device, bt, d, m, dtype, args.warmup, args.reps))
                for r in ranks:
                    rows.append(static_ephemeral_ffn_case(trial, device, bt, d, m, r, dtype, args.warmup, args.reps))
                    rows.append(dynamic_ephemeral_ffn_case(trial, device, bt, d, m, r, args.basis_size, dtype, args.warmup, args.reps))
                    rows.append(static_ephemeral_swiglu_case(trial, device, bt, d, m, r, dtype, args.warmup, args.reps))
                    rows.append(dynamic_ephemeral_swiglu_case(trial, device, bt, d, m, r, args.basis_size, dtype, args.warmup, args.reps))
                    rows.append(static_ephemeral_moe_top2_case(trial, device, bt, d, m, r, dtype, args.warmup, args.reps))
                    rows.append(dynamic_ephemeral_moe_top2_case(trial, device, bt, d, m, r, args.basis_size, dtype, args.warmup, args.reps))
                print('finished trial=', trial, 'd=', d, 'bt=', bt)

    raw_out = Path(args.raw_output)
    summary_out = Path(args.summary_output)
    save(rows, raw_out)
    print('wrote', raw_out)
    summary_rows = summarize_rows(rows)
    save(summary_rows, summary_out)
    print('wrote', summary_out)
