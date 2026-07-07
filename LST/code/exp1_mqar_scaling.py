"""Experiment 1: MQAR scaling with sequence length.

The classic result (Olsson et al. 2022; Arora et al. 2024) is that pure SSMs
struggle on MQAR at long lengths while Transformers excel. We test whether LST
retains Transformer-level accuracy at SSM-like cost.

We train LST, vanilla Transformer, pure SSM, and Linformer baselines on MQAR
at N = 256, 512, 1024, 2048 and report final accuracy.

This script uses at most 3 worker processes (one per model) to respect the
"don't use more than 3 npus" constraint. On Apple Silicon MPS there is a
single shared MPS device, so we run models sequentially to avoid contention.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, TransformerLM, SSMLM, LinformerLM
from lst.tasks import MQARDataset
from lst.train import TrainConfig, train_model, evaluate_model, get_param_count


def make_iter(ds, batch_size):
    """Return a callable that yields a fresh batch from an iterable dataset."""
    it = iter(ds)
    def _next():
        nonlocal it
        batch = []
        for _ in range(batch_size):
            try:
                batch.append(next(it))
            except StopIteration:
                it = iter(ds)
                batch.append(next(it))
        seqs, tgts = zip(*batch)
        return torch.stack(seqs), torch.stack(tgts)
    return _next


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


def run_one(name: str, seq_len: int, d_model: int, n_layers: int,
            k_latent: int, steps: int, batch_size: int, device: str,
            seed: int) -> dict:
    torch.manual_seed(seed)
    # MQAR config: scale n_kv_pairs with seq_len so the task is non-trivial
    n_kv = max(4, seq_len // 32)
    n_q = max(2, seq_len // 128)
    vocab_keys = 64
    vocab_vals = 64
    vocab = 1 + vocab_keys + vocab_vals
    max_T = max(4096, seq_len * 2)
    train_ds = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q,
                           vocab_keys=vocab_keys, vocab_vals=vocab_vals,
                           seq_len=seq_len, seed=seed)
    eval_ds = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q,
                          vocab_keys=vocab_keys, vocab_vals=vocab_vals,
                          seq_len=seq_len, seed=seed + 99991)
    train_iter = make_iter(train_ds, batch_size)
    eval_iter = make_iter(eval_ds, batch_size)
    model = build_model(name, vocab=vocab, d_model=d_model, n_layers=n_layers,
                        k_latent=k_latent, max_T=max_T)
    n_params = get_param_count(model)
    cfg = TrainConfig(epochs=1, steps_per_epoch=steps, batch_size=batch_size,
                      lr=3e-4, weight_decay=0.01, warmup_steps=min(50, steps // 5),
                      grad_clip=1.0, eval_every=max(50, steps // 4),
                      eval_steps=50, log_every=max(25, steps // 8),
                      device=device, seed=seed)
    t0 = time.time()
    hist = train_model(model, train_iter, eval_iter, cfg, label=f"{name}/N={seq_len}")
    elapsed = time.time() - t0
    final_loss, final_acc = hist["eval_loss"][-1][1], hist["eval_acc"][-1][1]
    return {
        "model": name,
        "seq_len": seq_len,
        "n_kv_pairs": n_kv,
        "n_queries": n_q,
        "params": n_params,
        "final_eval_loss": final_loss,
        "final_eval_acc": final_acc,
        "elapsed_s": elapsed,
        "history": hist,
    }


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Sequence lengths to sweep
    seq_lens = [256, 512, 1024, 2048]
    # Models to test
    models = ["LST", "Transformer", "SSM", "Linformer"]
    # Hyperparameters — keep small for a fast but meaningful experiment
    d_model = 96
    n_layers = 2
    k_latent = 64
    # Steps: scale down for longer seqs to keep wall-clock manageable
    steps_table = {256: 400, 512: 400, 1024: 300, 2048: 200}
    batch_table = {256: 64, 512: 48, 1024: 32, 2048: 16}

    results = []
    for N in seq_lens:
        steps = steps_table[N]
        bs = batch_table[N]
        for name in models:
            print(f"\n=== {name} | N={N} | steps={steps} | bs={bs} ===")
            try:
                r = run_one(name, N, d_model, n_layers, k_latent,
                            steps, bs, device, seed=42)
                # Strip the heavy history before saving
                r_save = {k: v for k, v in r.items() if k != "history"}
                results.append(r_save)
                print(f"RESULT: {name} N={N} acc={r['final_eval_acc']:.4f} "
                      f"loss={r['final_eval_loss']:.4f} time={r['elapsed_s']:.1f}s")
                (out_dir / "exp1_mqar_partial.json").write_text(
                    json.dumps(results, indent=2)
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"FAIL: {name} N={N}: {e}")
                results.append({"model": name, "seq_len": N, "error": str(e)})
                (out_dir / "exp1_mqar_partial.json").write_text(
                    json.dumps(results, indent=2)
                )

    (out_dir / "exp1_mqar.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp1_mqar.json'}")


if __name__ == "__main__":
    main()
