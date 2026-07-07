"""Experiment 4: K (latent dimension) ablation.

How does the latent bottleneck size K affect accuracy and speed?
We sweep K = 8, 16, 32, 64, 128, 256 on MQAR with n_kv=64, n_q=16.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST
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
    print(f"Using device: {device}")

    n_kv = 64; n_q = 16; vk = 128; vv = 128
    ds = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q, vocab_keys=vk, vocab_vals=vv, seed=0)
    ds_eval = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q, vocab_keys=vk, vocab_vals=vv, seed=999)
    train_iter = make_iter(ds, 64); eval_iter = make_iter(ds_eval, 64)

    k_values = [8, 16, 32, 64, 128, 256]
    results = []
    for k in k_values:
        print(f"\n=== LST K={k} ===", flush=True)
        model = LST(vocab=ds.vocab, d_model=128, n_layers=2, k_latent=k, n_heads=4)
        n_params = sum(p.numel() for p in model.parameters())
        cfg = TrainConfig(epochs=1, steps_per_epoch=600, batch_size=64,
                          lr=3e-4, weight_decay=0.0, warmup_steps=100,
                          grad_clip=1.0, eval_every=150, eval_steps=50,
                          log_every=150, device=device, seed=42)
        t0 = time.time()
        hist = train_model(model, train_iter, eval_iter, cfg, label=f"LST/K={k}")
        dt = time.time() - t0
        acc = hist['eval_acc'][-1][1]
        loss = hist['eval_loss'][-1][1]
        print(f"RESULT: K={k} acc={acc:.4f} loss={loss:.4f} time={dt:.1f}s params={n_params:,}", flush=True)
        results.append({"k": k, "params": n_params, "final_acc": acc,
                        "final_loss": loss, "time_s": dt})
        (out_dir / "exp4_k_ablation_partial.json").write_text(json.dumps(results, indent=2))

    (out_dir / "exp4_k_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp4_k_ablation.json'}")


if __name__ == "__main__":
    main()
