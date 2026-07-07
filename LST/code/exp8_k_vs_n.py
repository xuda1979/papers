"""Experiment 8: Latent budget K vs sequence length N scaling.

Systematically varies both K (latent budget) and N (sequence length) to
map out the scaling frontier of LST. Tests the theoretical prediction
that LST needs K >= n_kv for exact retrieval, and measures how accuracy
degrades when K < n_kv.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, LSTG, LinformerLM
from lst.tasks import MQARDataset
from lst.train import TrainConfig, train_model


def make_iter(ds, batch_size):
    it = iter(ds)
    def _next():
        batch = []
        for _ in range(batch_size):
            batch.append(next(it))
        seqs, targets = zip(*batch)
        return torch.stack(seqs), torch.stack(targets)
    return _next


def run_one(name, model_cls, n_kv, k_latent, V, device, steps=600):
    n_q = max(4, n_kv // 4)
    ds = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q,
                     vocab_keys=128, vocab_vals=128, seed=42, score_kv=True)
    train_iter = make_iter(ds, batch_size=64)
    eval_iter = make_iter(ds, batch_size=64)

    m = model_cls().to(device)
    n_params = sum(p.numel() for p in m.parameters())
    cfg = TrainConfig(epochs=1, steps_per_epoch=steps, batch_size=64,
                      lr=5e-4, weight_decay=0.0, warmup_steps=100,
                      grad_clip=1.0, eval_every=150, eval_steps=50,
                      log_every=150, device=str(device), seed=42)
    t0 = time.time()
    hist = train_model(m, train_iter, eval_iter, cfg, label=f"{name}/K={k_latent}_nkv={n_kv}")
    dt = time.time() - t0
    acc = hist["eval_acc"][-1][1]
    loss = hist["eval_loss"][-1][1]
    del m
    if device.type == 'mps':
        torch.mps.empty_cache()
    return {"model": name, "n_kv": n_kv, "k_latent": k_latent, "vocab": V,
            "params": n_params, "final_acc": acc, "final_loss": loss, "time_s": dt}


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    V = 1 + 128 + 128

    results = []
    # Vary K and n_kv to find the scaling frontier
    for n_kv in [8, 16, 32, 48, 64]:
        for k_latent in [8, 16, 32, 64]:
            print(f"\n{'='*60}")
            print(f"n_kv = {n_kv}, K = {k_latent}")
            print(f"{'='*60}", flush=True)
            for name, model_cls in [
                ("LST", lambda: LST(vocab=V, d_model=128, n_layers=2,
                                    k_latent=k_latent, n_heads=4)),
                ("Linformer", lambda: LinformerLM(vocab=V, d_model=128, n_layers=2,
                                                   n_heads=4, k_latent=k_latent, max_T=4096)),
            ]:
                try:
                    r = run_one(name, model_cls, n_kv, k_latent, V, device, steps=600)
                    results.append(r)
                    print(f"RESULT: {name} n_kv={n_kv} K={k_latent} acc={r['final_acc']:.4f}", flush=True)
                    (out_dir / "exp8_partial.json").write_text(json.dumps(results, indent=2))
                except Exception as e:
                    print(f"ERROR {name} n_kv={n_kv} K={k_latent}: {e}")
                    results.append({"model": name, "n_kv": n_kv, "k_latent": k_latent,
                                    "final_acc": 0.0, "error": str(e)})
                    (out_dir / "exp8_partial.json").write_text(json.dumps(results, indent=2))

    (out_dir / "exp8_k_vs_n.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp8_k_vs_n.json'}")


if __name__ == "__main__":
    main()
