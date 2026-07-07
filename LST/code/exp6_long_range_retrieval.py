"""Experiment 6: Long-range retrieval scaling.

Tests LST's ability to retrieve from very long contexts (N up to 4096)
with a fixed latent budget K=64. Compares against:
  - Transformer (O(N^2) baseline, will be slow at large N)
  - SSM (linear but limited state)
  - Linformer (linear but static compression)
  - LST (linear with state-informed compression)
  - LST-G (linear with gated routing)

Task: MQAR with n_kv=64 key-value pairs, padded to sequence length N.
The KV pairs are placed at the START of the sequence, followed by noise,
then queries at the END. This tests pure long-range retrieval.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, LSTG, TransformerLM, SSMLM, LinformerLM
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


def run_one(name, model_cls, n_kv, seq_len, V, device, steps=400):
    """Train a single model on long-range MQAR."""
    n_q = max(4, n_kv // 4)
    ds = MQARDataset(n_kv_pairs=n_kv, n_queries=n_q,
                     vocab_keys=128, vocab_vals=128,
                     seq_len=seq_len, seed=42, score_kv=False)
    train_iter = make_iter(ds, batch_size=32)
    eval_iter = make_iter(ds, batch_size=32)

    m = model_cls().to(device)
    n_params = sum(p.numel() for p in m.parameters())
    cfg = TrainConfig(epochs=1, steps_per_epoch=steps, batch_size=32,
                      lr=5e-4, weight_decay=0.0, warmup_steps=50,
                      grad_clip=1.0, eval_every=100, eval_steps=30,
                      log_every=100, device=str(device), seed=42)
    t0 = time.time()
    hist = train_model(m, train_iter, eval_iter, cfg, label=f"{name}/N={seq_len}")
    dt = time.time() - t0
    acc = hist["eval_acc"][-1][1]
    loss = hist["eval_loss"][-1][1]
    del m
    if device.type == 'mps':
        torch.mps.empty_cache()
    return {"model": name, "n_kv": n_kv, "seq_len": seq_len, "vocab": V,
            "params": n_params, "final_acc": acc, "final_loss": loss, "time_s": dt}


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    V = 1 + 128 + 128  # 257
    n_kv = 64

    results = []
    # Test at increasing sequence lengths with fixed n_kv=64
    for seq_len in [256, 512, 1024, 2048, 4096]:
        print(f"\n{'='*60}")
        print(f"seq_len = {seq_len}, n_kv = {n_kv}")
        print(f"{'='*60}")
        for name, model_cls in [
            ("Transformer", lambda: TransformerLM(vocab=V, d_model=128, n_layers=2,
                                                   n_heads=4, max_seq_len=8192)),
            ("LST", lambda: LST(vocab=V, d_model=128, n_layers=2,
                                k_latent=64, n_heads=4)),
            ("LSTG", lambda: LSTG(vocab=V, d_model=128, n_layers=2,
                                  k_latent=64, n_heads=4)),
            ("SSM", lambda: SSMLM(vocab=V, d_model=128, n_layers=2, d_state=16)),
            ("Linformer", lambda: LinformerLM(vocab=V, d_model=128, n_layers=2,
                                               n_heads=4, k_latent=64, max_T=8192)),
        ]:
            # Skip transformer for very long sequences (too slow)
            if name == "Transformer" and seq_len > 1024:
                print(f"Skipping {name} at seq_len={seq_len} (too slow)")
                results.append({"model": name, "n_kv": n_kv, "seq_len": seq_len,
                                "vocab": V, "params": 0, "final_acc": 0.0,
                                "final_loss": 0.0, "time_s": 0.0, "skipped": True})
                continue
            try:
                r = run_one(name, model_cls, n_kv, seq_len, V, device, steps=400)
                results.append(r)
                print(f"RESULT: {name} seq={seq_len} acc={r['final_acc']:.4f} time={r['time_s']:.1f}s")
                (out_dir / "exp6_partial.json").write_text(json.dumps(results, indent=2))
            except Exception as e:
                print(f"ERROR {name} seq={seq_len}: {e}")
                results.append({"model": name, "n_kv": n_kv, "seq_len": seq_len,
                                "vocab": V, "params": 0, "final_acc": 0.0,
                                "final_loss": 0.0, "time_s": 0.0, "error": str(e)})
                (out_dir / "exp6_partial.json").write_text(json.dumps(results, indent=2))

    (out_dir / "exp6_long_range.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp6_long_range.json'}")


if __name__ == "__main__":
    main()
