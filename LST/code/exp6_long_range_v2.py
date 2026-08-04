"""Experiment 6 v2: Long-range retrieval with fixed n_kv=16, varying noise.

Uses n_kv=16 (which exp1 showed is learnable in 600 steps) and varies
the sequence length by inserting noise tokens between KV pairs and queries.
This tests whether models can retrieve from an increasing context window
with a fixed latent budget K=64.
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
from lst.train import TrainConfig, train_model


class LongRangeMQAR:
    """MQAR with n_kv=16 pairs, noise inserted between KV and queries.
    Tests retrieval at increasing sequence lengths with fixed recall burden.
    """
    def __init__(self, n_kv=16, n_q=4, vocab_keys=128, vocab_vals=128,
                 seq_len=64, seed=0):
        self.n_kv = n_kv
        self.n_q = n_q
        self.vocab_keys = vocab_keys
        self.vocab_vals = vocab_vals
        self.seq_len = seq_len
        self.seed = seed
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def __iter__(self):
        while True:
            yield self._sample()

    def _sample(self):
        n_kv = self.n_kv
        n_q = self.n_q
        keys = torch.randperm(self.vocab_keys, generator=self._gen)[:n_kv] + 1
        vals = torch.randint(self.vocab_keys + 1, self.vocab_keys + self.vocab_vals + 1,
                             (n_kv,), generator=self._gen)
        # Interleaved KV pairs
        kv = torch.stack([keys, vals], dim=1).reshape(-1)  # (2*n_kv,)
        # Queries (sample n_q distinct keys)
        q_idx = torch.randperm(n_kv, generator=self._gen)[:n_q]
        queries = keys[q_idx]
        answers = vals[q_idx]
        core = torch.cat([kv, queries])  # (2*n_kv + n_q,)
        # Insert noise between KV and queries
        n_noise = self.seq_len - core.numel()
        if n_noise > 0:
            # Noise from a separate vocab range (0 = padding/noise token)
            noise = torch.zeros(n_noise, dtype=torch.long)
            seq = torch.cat([kv, noise, queries])
        else:
            seq = core
        # Pad or truncate
        if seq.numel() < self.seq_len:
            pad = torch.zeros(self.seq_len - seq.numel(), dtype=torch.long)
            seq = torch.cat([seq, pad])
        seq = seq[:self.seq_len]
        # Targets: (1) score value positions in KV pairs (auxiliary signal)
        # (2) after each query, predict the answer
        target = torch.full((self.seq_len,), -100, dtype=torch.long)
        # Score value positions: at position 2*i+1, target is vals[i]
        for i in range(n_kv):
            pos = 2 * i + 1
            if pos < self.seq_len:
                target[pos] = vals[i].item()
        # Score answer positions: after each query
        q_start = 2 * n_kv + max(0, n_noise)
        for i in range(n_q):
            q_pos = q_start + i
            ans_pos = q_pos + 1
            if ans_pos < self.seq_len:
                target[ans_pos] = answers[i].item()
        return seq, target


def make_iter(ds, batch_size):
    it = iter(ds)
    def _next():
        batch = []
        for _ in range(batch_size):
            batch.append(next(it))
        seqs, targets = zip(*batch)
        return torch.stack(seqs), torch.stack(targets)
    return _next


def run_one(name, model_cls, seq_len, V, device, steps=600):
    ds = LongRangeMQAR(n_kv=16, n_q=4, vocab_keys=128, vocab_vals=128,
                       seq_len=seq_len, seed=42)
    train_iter = make_iter(ds, batch_size=64)
    eval_iter = make_iter(ds, batch_size=64)

    m = model_cls().to(device)
    n_params = sum(p.numel() for p in m.parameters())
    cfg = TrainConfig(epochs=1, steps_per_epoch=steps, batch_size=64,
                      lr=5e-4, weight_decay=0.0, warmup_steps=100,
                      grad_clip=1.0, eval_every=150, eval_steps=50,
                      log_every=150, device=str(device), seed=42)
    t0 = time.time()
    hist = train_model(m, train_iter, eval_iter, cfg, label=f"{name}/N={seq_len}")
    dt = time.time() - t0
    acc = hist["eval_acc"][-1][1]
    loss = hist["eval_loss"][-1][1]
    del m
    empty_cache(device)
    return {"model": name, "n_kv": 16, "seq_len": seq_len, "vocab": V,
            "params": n_params, "final_acc": acc, "final_loss": loss, "time_s": dt}


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

    V = 1 + 128 + 128

    results = []
    # n_kv=16, core length = 36, vary noise to get seq_lens
    for seq_len in [64, 128, 256, 512, 1024, 2048, 4096]:
        print(f"\n{'='*60}")
        print(f"seq_len = {seq_len}, n_kv = 16")
        print(f"{'='*60}", flush=True)
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
                print(f"Skipping {name} at seq_len={seq_len}")
                results.append({"model": name, "n_kv": 16, "seq_len": seq_len,
                                "vocab": V, "params": 0, "final_acc": 0.0,
                                "final_loss": 0.0, "time_s": 0.0, "skipped": True})
                continue
            try:
                r = run_one(name, model_cls, seq_len, V, device, steps=600)
                results.append(r)
                print(f"RESULT: {name} seq={seq_len} acc={r['final_acc']:.4f} time={r['time_s']:.1f}s", flush=True)
                (out_dir / "exp6_v2_partial.json").write_text(json.dumps(results, indent=2))
            except Exception as e:
                print(f"ERROR {name} seq={seq_len}: {e}")
                import traceback; traceback.print_exc()
                results.append({"model": name, "n_kv": 16, "seq_len": seq_len,
                                "vocab": V, "params": 0, "final_acc": 0.0,
                                "final_loss": 0.0, "time_s": 0.0, "error": str(e)})
                (out_dir / "exp6_v2_partial.json").write_text(json.dumps(results, indent=2))

    (out_dir / "exp6_long_range_v2.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp6_long_range_v2.json'}")


if __name__ == "__main__":
    main()
