"""Experiment 4 v2: K (latent dimension) ablation including LSTG.

Sweep K = 8, 16, 32, 64, 128 on MQAR with n_kv=64, n_q=16. Compare LST and LSTG.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, LSTG
from lst.tasks import MQARDataset
from lst.train import TrainConfig, train_model


def make_iter(ds, batch_size):
    it = iter(ds)
    def _next():
        nonlocal it
        batch = []
        for _ in range(batch_size):
            try: batch.append(next(it))
            except StopIteration:
                it = iter(ds); batch.append(next(it))
        seqs, tgts = zip(*batch)
        return torch.stack(seqs), torch.stack(tgts)
    return _next


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)

    n_kv, n_q = 64, 16
    vocab_keys, vocab_vals = 64, 64
    V = 1 + vocab_keys + vocab_vals
    seq_len = 2 * n_kv + n_q
    ds = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q,
                     vocab_keys=vocab_keys, vocab_vals=vocab_vals,
                     seq_len=seq_len, seed=0)
    ds_eval = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q,
                          vocab_keys=vocab_keys, vocab_vals=vocab_vals,
                          seq_len=seq_len, seed=1)
    train_iter = make_iter(ds, 64)
    eval_iter = make_iter(ds_eval, 64)

    results = []
    for k in [8, 16, 32, 64, 128]:
        for name, m_factory in [
            ("LST",  lambda k=k: LST(vocab=V, d_model=128, n_layers=2,
                                     k_latent=k, n_heads=4)),
            ("LSTG", lambda k=k: LSTG(vocab=V, d_model=128, n_layers=2,
                                      k_latent=k, n_heads=4)),
        ]:
            m = m_factory().to(device)
            n_params = sum(p.numel() for p in m.parameters())
            print(f"\n--- {name} K={k} ---", flush=True)
            cfg = TrainConfig(epochs=1, steps_per_epoch=500, batch_size=64,
                              lr=3e-4, weight_decay=0.0, warmup_steps=100,
                              grad_clip=1.0, eval_every=125, eval_steps=50,
                              log_every=125, device=device, seed=42)
            t0 = time.time()
            hist = train_model(m, train_iter, eval_iter, cfg, label=f"{name}/K={k}")
            dt = time.time() - t0
            acc = hist['eval_acc'][-1][1]
            loss = hist['eval_loss'][-1][1]
            print(f"RESULT: {name} K={k} acc={acc:.4f} loss={loss:.4f} time={dt:.1f}s params={n_params:,}", flush=True)
            results.append({"model": name, "k": k, "params": n_params,
                            "final_acc": acc, "final_loss": loss, "time_s": dt})
            (out_dir / "exp4_v2_partial.json").write_text(json.dumps(results, indent=2))
            del m
            if device == "mps":
                torch.mps.empty_cache()

    (out_dir / "exp4_k_ablation_v2.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp4_k_ablation_v2.json'}")


if __name__ == "__main__":
    main()
