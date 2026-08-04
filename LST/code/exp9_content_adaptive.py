"""Experiment 9: Content-Adaptive SIA — lifts LST's MQAR accuracy when K < n_kv.

Hypothesis (from Theorem 1): LST solves MQAR exactly when K >= n_kv.  When
K < n_kv, the original position-windowed SIA blends multiple key-value pairs
into each slot, causing retrieval failure.  The Content-Adaptive SIA (CA-SIA)
replaces positional windowing with a learned content router that selects the
K most salient tokens, so the latent holds task-relevant keys/values rather
than positional averages — recovering accuracy in the K < n_kv regime.

Protocol:
  * Fix d_model=128, n_layers=2, n_heads=4, d_state=16, K=32.
  * Vary n_kv in {32, 48, 64, 96, 128} so that n_kv > K in the hard regime.
  * Vocab: vocab_keys = n_kv + 32 (distinct keys), vocab_vals = 128.
  * Train 800 steps, batch 64, lr 5e-4, cosine, warmup 100.
  * Compare: LST, LSTG, LST-CA, LST-AA, Transformer (full attention upper bound).
  * Single NPU (MPS), 1 process, deterministic seeds.

Outputs: code/results/exp9_content_adaptive.json (+ partial).
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import (
    LST, LSTG, LSTCA, LSTCA2, LSTAA, TransformerLM,
    MQARDataset, TrainConfig, train_model, get_param_count,
)


def make_iter(ds, batch_size):
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


def build_model(name, vocab, d_model, n_layers, k_latent, n_heads, d_state, device):
    if name == "LST":
        m = LST(vocab=vocab, d_model=d_model, n_layers=n_layers,
                k_latent=k_latent, d_state=d_state, n_heads=n_heads)
    elif name == "LSTG":
        m = LSTG(vocab=vocab, d_model=d_model, n_layers=n_layers,
                 k_latent=k_latent, d_state=d_state, n_heads=n_heads)
    elif name == "LST-CA":
        m = LSTCA(vocab=vocab, d_model=d_model, n_layers=n_layers,
                  k_latent=k_latent, d_state=d_state, n_heads=n_heads,
                  router_temp=0.5, hard_top_k=True)
    elif name == "LST-CA2":
        m = LSTCA2(vocab=vocab, d_model=d_model, n_layers=n_layers,
                   k_latent=k_latent, d_state=d_state, n_heads=n_heads)
    elif name == "LST-AA":
        m = LSTAA(vocab=vocab, d_model=d_model, n_layers=n_layers,
                  k_max=k_latent, k_min=8, d_state=d_state, n_heads=n_heads,
                  router_temp=0.5)
    elif name == "Transformer":
        m = TransformerLM(vocab=vocab, d_model=d_model, n_layers=n_layers,
                          n_heads=n_heads)
    else:
        raise ValueError(name)
    return m.to(device)


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    try:
        from device_utils import get_device, empty_cache
        device = get_device()
    except ImportError:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        def empty_cache(_d):
            if _.type == "mps": torch.mps.empty_cache()
    print(f"Device: {device}", flush=True)

    # K fixed at 32 (the original config); n_kv grows beyond K to expose the
    # position-windowed bottleneck.
    K_fixed = 32
    n_kv_list = [32, 48, 64, 96, 128]
    models = ["LST", "LSTG", "LST-CA2", "LST-AA", "Transformer"]
    d_model, n_layers, n_heads, d_state = 128, 2, 4, 16

    results = []
    for n_kv in n_kv_list:
        n_q = max(4, n_kv // 4)
        # Distinct keys: vocab_keys must be >= n_kv.  Use n_kv + 32 for headroom.
        # Match the original exp1 config: vocab_keys=n_kv+32, vocab_vals=32
        # (gives V=1+n_kv+32+32=n_kv+65; e.g. V=97 at n_kv=32, V=161 at n_kv=96).
        vocab_keys = n_kv + 32
        vocab_vals = 32
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

        for name in models:
            torch.manual_seed(42)
            m = build_model(name, V, d_model, n_layers, K_fixed, n_heads, d_state, device)
            n_params = get_param_count(m)
            print(f"\n--- {name} n_kv={n_kv} K={K_fixed} seq_len={seq_len} V={V} params={n_params} ---", flush=True)
            cfg = TrainConfig(epochs=1, steps_per_epoch=800, batch_size=64,
                              lr=5e-4, weight_decay=0.0, warmup_steps=100,
                              grad_clip=1.0, eval_every=200, eval_steps=50,
                              log_every=200, device=device, seed=42)
            t0 = time.time()
            hist = train_model(m, train_iter, eval_iter, cfg,
                               label=f"{name}/n_kv={n_kv}/K={K_fixed}")
            dt = time.time() - t0
            eloss, eacc = hist["eval_loss"][-1][1], hist["eval_acc"][-1][1]
            results.append({"model": name, "n_kv": n_kv, "n_q": n_q,
                            "seq_len": seq_len, "K": K_fixed,
                            "vocab": V, "params": n_params,
                            "final_acc": eacc, "final_loss": eloss, "time_s": dt})
            (out_dir / "exp9_partial.json").write_text(json.dumps(results, indent=2))
            print(f"RESULT {name} n_kv={n_kv} K={K_fixed}: acc={eacc:.4f} loss={eloss:.4f} t={dt:.1f}s", flush=True)
            del m
            empty_cache(device)

    (out_dir / "exp9_content_adaptive.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp9_content_adaptive.json'}")


if __name__ == "__main__":
    main()
