"""Experiment 7: Noise robustness — retrieval under varying noise ratios.

Tests how well LST retrieves key-value associations when the context is
infiltrated by varying fractions of noise tokens. The latent bottleneck
must learn to ignore noise and retain only the (key, value) signal.

Task: MQAR with n_kv=32 pairs, sequence length 256, but with 0%, 25%,
50%, 75% noise tokens inserted between the KV pairs and the queries.
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


class NoisyMQAR:
    """MQAR with noise tokens inserted between KV pairs and queries."""
    def __init__(self, n_kv=32, n_q=8, vocab_keys=128, vocab_vals=128,
                 seq_len=256, noise_ratio=0.0, seed=0):
        self.n_kv = n_kv
        self.n_q = n_q
        self.vocab_keys = vocab_keys
        self.vocab_vals = vocab_vals
        self.seq_len = seq_len
        self.noise_ratio = noise_ratio
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
        answers = vals[torch.randperm(n_kv, generator=self._gen)[:n_q]]
        # Build the core sequence: [k1, v1, ..., kn, vn, q1, ..., qm]
        kv = torch.stack([keys, vals], dim=1).reshape(-1)
        queries = keys[torch.randperm(n_kv, generator=self._gen)[:n_q]]
        core = torch.cat([kv, queries])  # length 2*n_kv + n_q
        # Insert noise tokens
        n_noise = int(self.seq_len * self.noise_ratio)
        # Noise vocab is 0 (a special padding/noise token)
        noise = torch.zeros(n_noise, dtype=torch.long)
        # Place noise between KV pairs and queries
        kv_part = core[:2*n_kv]
        q_part = core[2*n_kv:]
        # Split noise and insert
        if n_noise > 0:
            seq = torch.cat([kv_part, noise, q_part])
        else:
            seq = core
        # Pad or truncate to seq_len
        if seq.numel() < self.seq_len:
            pad = torch.zeros(self.seq_len - seq.numel(), dtype=torch.long)
            seq = torch.cat([seq, pad])
        seq = seq[:self.seq_len]
        # Targets: after each query, predict the answer
        target = torch.full((self.seq_len,), -100, dtype=torch.long)
        # Query positions are after kv_part (2*n_kv) + noise (n_noise)
        q_start = 2 * n_kv + n_noise
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


def run_one(name, model_cls, noise_ratio, V, device, steps=400):
    ds = NoisyMQAR(n_kv=32, n_q=8, vocab_keys=128, vocab_vals=128,
                   seq_len=256, noise_ratio=noise_ratio, seed=42)
    train_iter = make_iter(ds, batch_size=32)
    eval_iter = make_iter(ds, batch_size=32)

    m = model_cls().to(device)
    n_params = sum(p.numel() for p in m.parameters())
    cfg = TrainConfig(epochs=1, steps_per_epoch=steps, batch_size=32,
                      lr=5e-4, weight_decay=0.0, warmup_steps=50,
                      grad_clip=1.0, eval_every=100, eval_steps=30,
                      log_every=100, device=str(device), seed=42)
    t0 = time.time()
    hist = train_model(m, train_iter, eval_iter, cfg, label=f"{name}/noise={noise_ratio}")
    dt = time.time() - t0
    acc = hist["eval_acc"][-1][1]
    loss = hist["eval_loss"][-1][1]
    del m
    if device.type == 'mps':
        torch.mps.empty_cache()
    return {"model": name, "noise_ratio": noise_ratio, "vocab": V,
            "params": n_params, "final_acc": acc, "final_loss": loss, "time_s": dt}


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    V = 1 + 128 + 128

    results = []
    for noise_ratio in [0.0, 0.25, 0.50, 0.75]:
        print(f"\n{'='*60}")
        print(f"noise_ratio = {noise_ratio}")
        print(f"{'='*60}")
        for name, model_cls in [
            ("Transformer", lambda: TransformerLM(vocab=V, d_model=128, n_layers=2,
                                                   n_heads=4, max_seq_len=4096)),
            ("LST", lambda: LST(vocab=V, d_model=128, n_layers=2,
                                k_latent=64, n_heads=4)),
            ("LSTG", lambda: LSTG(vocab=V, d_model=128, n_layers=2,
                                  k_latent=64, n_heads=4)),
            ("SSM", lambda: SSMLM(vocab=V, d_model=128, n_layers=2, d_state=16)),
            ("Linformer", lambda: LinformerLM(vocab=V, d_model=128, n_layers=2,
                                               n_heads=4, k_latent=64, max_T=4096)),
        ]:
            try:
                r = run_one(name, model_cls, noise_ratio, V, device, steps=400)
                results.append(r)
                print(f"RESULT: {name} noise={noise_ratio} acc={r['final_acc']:.4f}")
                (out_dir / "exp7_partial.json").write_text(json.dumps(results, indent=2))
            except Exception as e:
                print(f"ERROR {name} noise={noise_ratio}: {e}")
                results.append({"model": name, "noise_ratio": noise_ratio,
                                "final_acc": 0.0, "error": str(e)})
                (out_dir / "exp7_partial.json").write_text(json.dumps(results, indent=2))

    (out_dir / "exp7_noise_robustness.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out_dir / 'exp7_noise_robustness.json'}")


if __name__ == "__main__":
    main()
