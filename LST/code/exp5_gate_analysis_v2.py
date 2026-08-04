"""Experiment 5 v2: Gate behaviour analysis in LST-G at n_kv=64 (where the model learns well)."""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lst import LSTG
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


def collect_gate_stats(model, x):
    """Run a forward pass and collect per-position gate values from all GSIA blocks."""
    B, T = x.shape
    model.eval()
    gates_per_layer = []
    with torch.no_grad():
        h = model.embed(x)
        for blk in model.blocks:
            normed = blk.norm1(h)
            gsia = blk.gsia
            B_, T_, D_ = normed.shape
            H, Kh, Dh = gsia.n_heads, gsia.k_latent, gsia.d_head
            Q = gsia.latent_queries.unsqueeze(0).expand(B_, -1, -1)
            q_c = gsia.q_proj_c(Q).view(B_, Kh, H, Dh).transpose(1, 2)
            k_c = gsia.k_proj_c(normed).view(B_, T_, H, Dh).transpose(1, 2)
            v_c = gsia.v_proj_c(normed).view(B_, T_, H, Dh).transpose(1, 2)
            rows_k = torch.arange(Kh, device=normed.device).view(-1, 1)
            cols_t = torch.arange(T_, device=normed.device).view(1, -1)
            causal_c = (cols_t >= (rows_k * T_) // Kh) & (cols_t < ((rows_k + 1) * T_) // Kh)
            attn_c = (q_c @ k_c.transpose(-1, -2)) * gsia.scale
            attn_c = attn_c.masked_fill(~causal_c.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_c = F.softmax(attn_c, dim=-1)
            latent = attn_c @ v_c
            latent = latent.transpose(1, 2).contiguous().view(B_, Kh, D_)
            latent = gsia.latent_ssm(latent)
            q_r = gsia.q_proj_r(normed).view(B_, T_, H, Dh).transpose(1, 2)
            k_r = gsia.k_proj_r(latent).view(B_, Kh, H, Dh).transpose(1, 2)
            rows_t = torch.arange(T_, device=normed.device).view(-1, 1)
            cols_k = torch.arange(Kh, device=normed.device).view(1, -1)
            causal_r = cols_k <= ((rows_t + 1) * Kh - 1) // T_
            mask = causal_r.unsqueeze(0).unsqueeze(-1).float()
            latent_for_mu = latent.unsqueeze(1)
            mu_sum = (latent_for_mu * mask).sum(dim=2)
            mu_cnt = mask.sum(dim=2).clamp(min=1.0)
            mu = mu_sum / mu_cnt
            gate = torch.sigmoid(gsia.gate_x(normed) + gsia.gate_mu(mu))
            gates_per_layer.append(gate.mean(dim=-1).cpu())
            # continue forward
            v_r = gsia.v_proj_r(latent).view(B_, Kh, H, Dh).transpose(1, 2)
            attn_r = (q_r @ k_r.transpose(-1, -2)) * gsia.scale
            attn_r = attn_r.masked_fill(~causal_r.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_r = F.softmax(attn_r, dim=-1)
            h_sia = attn_r @ v_r
            h_sia = h_sia.transpose(1, 2).contiguous().view(B_, T_, D_)
            h_local = gsia.local_ssm(normed)
            y = gate * h_sia + (1.0 - gate) * h_local
            h = h + gsia.out_proj(y)
            h = h + blk.ffn(blk.norm2(h))
    return gates_per_layer


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
    print(f"Using device: {device}", flush=True)

    # Use n_kv=64 where all models learn well (~80% accuracy)
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

    m = LSTG(vocab=V, d_model=128, n_layers=2, k_latent=32, d_state=16, n_heads=4).to(device)
    print("Training LST-G for gate analysis (n_kv=64)...", flush=True)
    cfg = TrainConfig(epochs=1, steps_per_epoch=600, batch_size=64,
                      lr=5e-4, weight_decay=0.0, warmup_steps=100,
                      grad_clip=1.0, eval_every=150, eval_steps=50,
                      log_every=150, device=device, seed=42)
    t0 = time.time()
    hist = train_model(m, train_iter, eval_iter, cfg, label="LSTG/gate-analysis")
    dt = time.time() - t0
    acc = hist['eval_acc'][-1][1]
    print(f"Trained LST-G: acc={acc:.4f} time={dt:.1f}s", flush=True)

    # Collect gate stats on 256 eval sequences
    print("Collecting gate statistics...", flush=True)
    all_gates = []
    n_batches = 4
    for _ in range(n_batches):
        x, y = eval_iter()
        x = x.to(device)
        gates = collect_gate_stats(m, x)
        if not all_gates:
            all_gates = [g.cpu() for g in gates]
        else:
            for i, g in enumerate(gates):
                all_gates[i] = torch.cat([all_gates[i], g.cpu()], dim=0)

    # Compute per-position-type gate means
    pos_type = []
    for t in range(seq_len):
        if t < 2 * n_kv:
            if t % 2 == 0:
                pos_type.append("key")
            else:
                pos_type.append("value")
        else:
            pos_type.append("query")
    pos_type_idx = torch.tensor([{ "key": 0, "value": 1, "query": 2 }[p] for p in pos_type])

    gate_stats = {"per_layer": [], "per_position_type": {}}
    for layer, g in enumerate(all_gates):
        gate_stats["per_layer"].append({
            "layer": layer,
            "mean": float(g.mean()),
            "std": float(g.std()),
        })
        for ptype, pidx in [("key", 0), ("value", 1), ("query", 2)]:
            mask = (pos_type_idx == pidx)
            if mask.any():
                gate_stats["per_position_type"].setdefault(ptype, []).append({
                    "layer": layer,
                    "mean": float(g[:, mask].mean()),
                    "std": float(g[:, mask].std()),
                })
    avg_g = torch.stack(all_gates, dim=0).mean(dim=0)
    gate_stats["per_position_mean"] = avg_g.mean(dim=0).tolist()
    gate_stats["pos_type"] = pos_type
    gate_stats["acc"] = acc
    gate_stats["n_kv"] = n_kv
    gate_stats["n_q"] = n_q
    gate_stats["seq_len"] = seq_len

    (out_dir / "exp5_gate_analysis_v2.json").write_text(json.dumps(gate_stats, indent=2))
    print(f"\nWrote {out_dir / 'exp5_gate_analysis_v2.json'}")
    print("Per-position-type gate means:")
    for ptype in ["key", "value", "query"]:
        if ptype in gate_stats["per_position_type"]:
            for entry in gate_stats["per_position_type"][ptype]:
                print(f"  {ptype} layer={entry['layer']}: mean={entry['mean']:.3f} std={entry['std']:.3f}")


if __name__ == "__main__":
    main()
