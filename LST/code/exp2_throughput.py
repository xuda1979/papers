"""Experiment 2: Throughput and memory scaling with sequence length.

We measure wall-clock forward+backward time and peak memory for LST, vanilla
Transformer, pure SSM, and Linformer at sequence lengths N = 256, 512, 1024,
2048, 4096, 8192. We fix d_model=128, n_layers=2, and use batch sizes that
fit in memory.

This experiment directly demonstrates the sub-quadratic scaling of LST.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, TransformerLM, SSMLM, LinformerLM


def build_model(name: str, vocab: int, d_model: int, n_layers: int,
                k_latent: int, max_T: int):
    if name == "LST":
        return LST(vocab=vocab, d_model=d_model, n_layers=n_layers,
                   k_latent=k_latent, d_state=16, n_heads=4, dropout=0.0)
    if name == "Transformer":
        return TransformerLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                             n_heads=4, dropout=0.0)
    if name == "SSM":
        return SSMLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                     d_state=16, dropout=0.0)
    if name == "Linformer":
        return LinformerLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                           n_heads=4, k_latent=k_latent, max_T=max_T,
                           dropout=0.0)
    raise ValueError(name)


def benchmark_one(name: str, seq_len: int, d_model: int = 128, n_layers: int = 2,
                  k_latent: int = 64, batch_size: int = 16, n_warmup: int = 3,
                  n_runs: int = 8, device: str = "mps") -> dict:
    vocab = 257
    max_T = max(8192, seq_len * 2)
    model = build_model(name, vocab=vocab, d_model=d_model, n_layers=n_layers,
                        k_latent=k_latent, max_T=max_T).to(device)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab, (batch_size, seq_len), device=device)
    import torch.nn.functional as F
    # Warmup
    for _ in range(n_warmup):
        try:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {"model": name, "seq_len": seq_len, "batch_size": batch_size,
                        "status": "OOM"}
            raise
    if device == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize() if device.startswith("cuda") else None
    # Measure
    times = []
    peak_mem_mb = 0.0
    for _ in range(n_runs):
        t0 = time.time()
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if device == "mps":
            torch.mps.synchronize()
            try:
                peak_mem_mb = max(peak_mem_mb, torch.mps.current_allocated_memory() / 1e6)
            except Exception:
                pass
        elif device.startswith("cuda"):
            torch.cuda.synchronize()
            peak_mem_mb = max(peak_mem_mb, torch.cuda.max_memory_allocated() / 1e6)
        times.append(time.time() - t0)
    times.sort()
    # Discard slowest and fastest, average the middle
    times = times[1:-1] if len(times) > 2 else times
    avg_ms = sum(times) / len(times) * 1000
    n_params = sum(p.numel() for p in model.parameters())
    return {
        "model": name,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "params": n_params,
        "avg_ms_per_step": avg_ms,
        "peak_mem_mb": peak_mem_mb,
        "throughput_tokens_per_s": batch_size * seq_len / (avg_ms / 1000),
        "status": "OK",
    }


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    seq_lens = [256, 512, 1024, 2048, 4096, 8192]
    models = ["LST", "Transformer", "SSM", "Linformer"]
    d_model = 128
    n_layers = 2
    k_latent = 64
    # Use a modest batch size; reduce for long sequences
    batch_table = {256: 32, 512: 32, 1024: 16, 2048: 8, 4096: 4, 8192: 2}

    results = []
    for N in seq_lens:
        bs = batch_table[N]
        for name in models:
            print(f"\n=== {name} | N={N} | bs={bs} ===")
            try:
                r = benchmark_one(name, N, d_model, n_layers, k_latent,
                                  batch_size=bs, device=device)
                print(f"RESULT: {name} N={N} time={r['avg_ms_per_step']:.1f}ms "
                      f"mem={r['peak_mem_mb']:.0f}MB "
                      f"throughput={r['throughput_tokens_per_s']:.0f} tok/s")
                results.append(r)
                (out_dir / "exp2_throughput_partial.json").write_text(
                    json.dumps(results, indent=2)
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"FAIL: {name} N={N}: {e}")
                results.append({"model": name, "seq_len": N, "error": str(e)})
                (out_dir / "exp2_throughput_partial.json").write_text(
                    json.dumps(results, indent=2)
                )

    (out_dir / "exp2_throughput.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp2_throughput.json'}")


if __name__ == "__main__":
    main()
