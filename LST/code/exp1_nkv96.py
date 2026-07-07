"""Run just n_kv=96 for exp1 v2 (since the original crashed there)."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LST, LSTG, TransformerLM, SSMLM, LinformerLM
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

    # Load existing partial results
    partial_path = out_dir / "exp1_v2_partial.json"
    if partial_path.exists():
        results = json.loads(partial_path.read_text())
    else:
        results = []

    n_kv = 96
    n_q = max(4, n_kv // 4)
    vocab_keys = max(128, n_kv + 32)
    vocab_vals = 128
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
    for name, m_factory in [
        ("LST",  lambda V=V: LST(vocab=V, d_model=128, n_layers=2,
                                 k_latent=32, d_state=16, n_heads=4)),
        ("LSTG", lambda V=V: LSTG(vocab=V, d_model=128, n_layers=2,
                                  k_latent=32, d_state=16, n_heads=4)),
        ("Transformer", lambda V=V: TransformerLM(vocab=V, d_model=128,
                                                   n_layers=2, n_heads=4)),
        ("SSM",  lambda V=V: SSMLM(vocab=V, d_model=128, n_layers=2,
                                   d_state=16)),
        ("Linformer", lambda V=V: LinformerLM(vocab=V, d_model=128,
                                               n_layers=2, k_latent=32,
                                               n_heads=4)),
    ]:
        # Skip if already in results
        if any(r.get("model") == name and r.get("n_kv") == n_kv for r in results):
            print(f"Skipping {name} n_kv={n_kv} (already done)")
            continue
        m = m_factory().to(device)
        n_params = sum(p.numel() for p in m.parameters())
        print(f"\n--- {name} n_kv={n_kv} seq_len={seq_len} ---", flush=True)
        cfg = TrainConfig(epochs=1, steps_per_epoch=600, batch_size=64,
                          lr=5e-4, weight_decay=0.0, warmup_steps=100,
                          grad_clip=1.0, eval_every=150, eval_steps=50,
                          log_every=150, device=device, seed=42)
        t0 = time.time()
        hist = train_model(m, train_iter, eval_iter, cfg,
                           label=f"{name}/n_kv={n_kv}")
        dt = time.time() - t0
        acc = hist['eval_acc'][-1][1]
        loss = hist['eval_loss'][-1][1]
        print(f"RESULT: {name} n_kv={n_kv} acc={acc:.4f} loss={loss:.4f} time={dt:.1f}s params={n_params:,}", flush=True)
        results.append({"model": name, "n_kv": n_kv, "n_q": n_q,
                        "seq_len": seq_len, "vocab": V, "params": n_params,
                        "final_acc": acc, "final_loss": loss, "time_s": dt})
        partial_path.write_text(json.dumps(results, indent=2))
        del m
        if device == "mps":
            torch.mps.empty_cache()

    (out_dir / "exp1_mqar_scaling_v2.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp1_mqar_scaling_v2.json'}")


if __name__ == "__main__":
    main()
