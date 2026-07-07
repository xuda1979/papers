"""Comprehensive experiments for Discrete Token-Flow Networks (DTFNs).

This script reproduces and extends the experiments from the paper:

  Exp 1: Conservation and positivity across DTFN variants
  Exp 2: Needle advection across long sequences
  Exp 3: Dyck-1 language modeling with multiple baselines
  Exp 4: Length extrapolation (train short, test long)
  Exp 5: Copy task (memory-intensive)
  Exp 6: Ablation on K (number of update steps)
  Exp 7: Ablation on d (embedding dimension)
  Exp 8: Wall-clock scaling vs sequence length

All experiments are deterministic (fixed seeds) and run on CPU or MPS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dtfn_core as dc


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(name: str) -> torch.device:
    if name == "mps":
        return torch.device("mps")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    """Token-level accuracy ignoring padded positions."""
    preds = logits.argmax(dim=-1)
    mask = target != ignore_index
    correct = (preds == target) & mask
    return correct.sum().float().item() / mask.sum().float().clamp(min=1).item()


def sequence_accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    """Sequence-level accuracy: fraction of sequences exactly matched (ignoring pad)."""
    preds = logits.argmax(dim=-1)
    mask = target != ignore_index
    correct = (preds == target) | ~mask
    return correct.all(dim=1).float().mean().item()


# ----------------------------------------------------------------------
# Datasets
# ----------------------------------------------------------------------

class DyckDataset(Dataset):
    """Dyck-1 (single type of bracket) balanced parenthesis sequences.

    Each sample is (input, target) where input = [BOS] + seq and
    target = seq + [EOS].  Tokens: ( = 0, ) = 1, PAD = 2, BOS = 3, EOS = 4.
    For Dyck-k, k pairs of bracket types are used.
    """

    def __init__(self, n: int, min_pairs: int = 4, max_pairs: int = 32,
                 num_bracket_types: int = 1, seed: int = 0) -> None:
        self.tok2id = {'(' : 0, ')' : 1, 'PAD' : 2, 'BOS' : 3, 'EOS' : 4}
        # For multi-bracket Dyck, we add tokens 4+k... but keep simple here
        # For Dyck-1: ( = 0, ) = 1
        # For Dyck-2: ( = 0, ) = 1, [ = 4, ] = 5
        self.pad_id = 2
        self.bos_id = 3
        self.eos_id = 4
        self.num_bracket_types = num_bracket_types
        if num_bracket_types > 1:
            # add extra token ids for additional bracket types
            for k in range(1, num_bracket_types):
                # use ids 4 + 2*(k-1) and 4 + 2*(k-1) + 1
                pass
        vocab_size = 2 + 2 * num_bracket_types + 2  # base + brackets + BOS/EOS
        # Actually, simplest scheme: tokens 0..2k-1 are brackets (open then close)
        # token 2k = PAD, 2k+1 = BOS, 2k+2 = EOS
        self.vocab_size = 2 * num_bracket_types + 3
        self.pad_id = 2 * num_bracket_types
        self.bos_id = 2 * num_bracket_types + 1
        self.eos_id = 2 * num_bracket_types + 2

        rnd = random.Random(seed)
        self.samples: List[List[int]] = []
        for _ in range(n):
            p = rnd.randint(min_pairs, max_pairs)
            self.samples.append(self._generate(p, rnd))

    def _generate(self, p: int, rnd: random.Random) -> List[int]:
        res: List[int] = []
        stack: List[int] = []
        total_len = 2 * p
        for _ in range(total_len):
            # Decide open or close
            can_open = len(stack) < p - len(res) + len(stack)  # heuristic
            if not stack:
                # must open
                br = rnd.randrange(self.num_bracket_types)
                res.append(2 * br)
                stack.append(br)
            elif len(res) + len(stack) == total_len:
                # must close
                br = stack.pop()
                res.append(2 * br + 1)
            elif rnd.random() < 0.5:
                br = rnd.randrange(self.num_bracket_types)
                res.append(2 * br)
                stack.append(br)
            else:
                br = stack.pop()
                res.append(2 * br + 1)
        return res

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx]
        x = torch.tensor([self.bos_id] + seq, dtype=torch.long)
        y = torch.tensor(seq + [self.eos_id], dtype=torch.long)
        return x, y


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]],
            pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x = torch.full((len(xs), max_len), fill_value=pad_id, dtype=torch.long)
    y = torch.full((len(ys), max_len), fill_value=pad_id, dtype=torch.long)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        x[i, : xi.size(0)] = xi
        y[i, : yi.size(0)] = yi
    return x, y


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------

class LSTMModel(nn.Module):
    def __init__(self, vocab_size: int, d: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dec = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.emb(x)
        h, _ = self.lstm(h)
        return self.dec(h)


class TransformerModel(nn.Module):
    """A small causal-transformer baseline with quadratic attention."""

    def __init__(self, vocab_size: int, d: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 512) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=n_heads, dim_feedforward=4*d,
            dropout=0.0, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.dec = nn.Linear(d, vocab_size)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, L)
        h = self.emb(x) + self.pos(positions)
        # Causal mask
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        h = self.encoder(h, mask=mask)
        return self.dec(h)


# ----------------------------------------------------------------------
# Training utilities
# ----------------------------------------------------------------------

@dataclass
class TrainResult:
    final_loss: float
    final_token_acc: float
    final_seq_acc: float
    train_time_sec: float
    n_params: int
    history: List[Dict[str, float]]


def train_model(model: nn.Module, loader: DataLoader, epochs: int,
                device: torch.device, lr: float = 1e-3,
                pad_id: int = 2, seed: int = 0, verbose: bool = True,
                clip_grad: float = 1.0) -> TrainResult:
    set_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    history: List[Dict[str, float]] = []
    t0 = time.time()
    for ep in range(epochs):
        total_loss = 0.0
        total_tk_acc = 0.0
        total_seq_acc = 0.0
        n_seen = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            B, L, V = logits.shape
            loss = ce(logits.reshape(B * L, V), yb.reshape(B * L))
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            opt.step()
            with torch.no_grad():
                tk_acc = token_accuracy(logits, yb, ignore_index=pad_id)
                sq_acc = sequence_accuracy(logits, yb, ignore_index=pad_id)
            total_loss += loss.item() * xb.size(0)
            total_tk_acc += tk_acc * xb.size(0)
            total_seq_acc += sq_acc * xb.size(0)
            n_seen += xb.size(0)
        avg_loss = total_loss / n_seen
        avg_tk = total_tk_acc / n_seen
        avg_sq = total_seq_acc / n_seen
        history.append({'epoch': ep+1, 'loss': avg_loss, 'token_acc': avg_tk, 'seq_acc': avg_sq})
        if verbose:
            print(f"  [ep {ep+1:2d}] loss={avg_loss:.4f} tk_acc={avg_tk:.4f} seq_acc={avg_sq:.4f}")
    dt = time.time() - t0
    return TrainResult(
        final_loss=history[-1]['loss'],
        final_token_acc=history[-1]['token_acc'],
        final_seq_acc=history[-1]['seq_acc'],
        train_time_sec=dt,
        n_params=count_params(model),
        history=history,
    )


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device,
                   pad_id: int = 2) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tk = 0.0
    total_sq = 0.0
    n_seen = 0
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            B, L, V = logits.shape
            loss = ce(logits.reshape(B * L, V), yb.reshape(B * L))
            total_loss += loss.item() * xb.size(0)
            total_tk += token_accuracy(logits, yb, ignore_index=pad_id) * xb.size(0)
            total_sq += sequence_accuracy(logits, yb, ignore_index=pad_id) * xb.size(0)
            n_seen += xb.size(0)
    return {
        'loss': total_loss / n_seen,
        'token_acc': total_tk / n_seen,
        'seq_acc': total_sq / n_seen,
    }


# ----------------------------------------------------------------------
# Experiment 1: Conservation and positivity
# ----------------------------------------------------------------------

def experiment_conservation(seed: int = 0) -> Dict:
    """Test mass conservation across DTFN variants."""
    set_seed(seed)
    print("\n=== Experiment 1: Conservation and Positivity ===")
    variants = {
        'DTFN-U': dc.make_dtfn_unidirectional(4, d=64, K=3),
        'DTFN-B': dc.make_dtfn_bidirectional(4, d=64, K=3),
        'DTFN-MUSCL': dc.make_dtfn_muscl(4, d=64, K=3),
    }
    results = {}
    x = torch.randint(0, 4, (64, 100))
    for name, m in variants.items():
        err = m.conservation_error(x)
        # Min mass after forward pass
        m0 = m.to_mass(m.emb(x)) + m.cfg.epsilon
        m_out = m._run_stack(m0, m.layers) if not m.cfg.bidirectional else \
                m._run_stack(m0, m.layers_fwd)
        results[name] = {
            'mean_conservation_error': float(err.mean().item()),
            'max_conservation_error': float(err.max().item()),
            'min_mass_after': float(m_out.min().item()),
            'n_params': count_params(m),
        }
        print(f"  {name:15s} mean_err={err.mean().item():.3e} max_err={err.max().item():.3e} min_mass={m_out.min().item():.3e}")
    return results


# ----------------------------------------------------------------------
# Experiment 2: Needle advection
# ----------------------------------------------------------------------

def experiment_needle(seed: int = 0) -> Dict:
    """Test needle advection across sequence lengths."""
    set_seed(seed)
    print("\n=== Experiment 2: Needle Advection ===")
    lengths = [64, 128, 256, 512, 1024]
    results = {}
    for L in lengths:
        # Place unit mass at position 0
        m0 = torch.zeros(1, L, 1)
        m0[0, 0, 0] = 1.0
        # Run L-1 upwind steps with v=0.8, dt=0.5, lambda=0.4
        v, dt = 0.8, 0.5
        lam = v * dt
        m = m0.clone()
        for _ in range(L - 1):
            F = v * m[:, :-1, :]
            dm = torch.zeros_like(m)
            dm[:, :-1, :] -= F
            dm[:, 1:, :] += F
            m = torch.clamp(m + dt * dm, min=0.0)
        peak = int(m[0, :, 0].argmax().item())
        expected = int(round((L - 1) * lam))
        # Mass conservation
        rel_err = abs(m.sum().item() - m0.sum().item()) / m0.sum().item()
        results[L] = {
            'predicted_peak': peak,
            'expected_peak': expected,
            'match': abs(peak - expected) <= 1,
            'mass_rel_err': rel_err,
            'min_mass': float(m.min().item()),
        }
        print(f"  L={L:5d} peak={peak:5d} expected={expected:5d} match={abs(peak - expected) <= 1} mass_err={rel_err:.3e}")
    return results


# ----------------------------------------------------------------------
# Experiment 3: Dyck-1 language modeling
# ----------------------------------------------------------------------

def experiment_dyck(seed: int = 0, epochs: int = 5, device_name: str = "cpu") -> Dict:
    """Train and compare models on Dyck-1."""
    set_seed(seed)
    device = get_device(device_name)
    print(f"\n=== Experiment 3: Dyck-1 Language Modeling (device={device}) ===")

    # Dataset
    train_ds = DyckDataset(2000, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=0)
    test_ds = DyckDataset(500, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=42)
    pad_id = train_ds.pad_id
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             collate_fn=lambda b: collate(b, pad_id))

    models = {
        'DTFN-U': dc.make_dtfn_unidirectional(vocab_size, d=64, K=3),
        'DTFN-B': dc.make_dtfn_bidirectional(vocab_size, d=64, K=3),
        'DTFN-MUSCL': dc.make_dtfn_muscl(vocab_size, d=64, K=3),
        'DTFN-FULL': dc.make_dtfn_full(vocab_size, d=64, K=3),
        'LSTM': LSTMModel(vocab_size, d=64, hidden_dim=128),
        'Transformer': TransformerModel(vocab_size, d=64, n_heads=4, n_layers=2, max_len=256),
    }
    results = {}
    for name, model in models.items():
        print(f"\n  Training {name} (params={count_params(model)})")
        tr = train_model(model, train_loader, epochs=epochs, device=device,
                         lr=1e-3, pad_id=pad_id, seed=seed, verbose=True)
        eval_res = evaluate_model(model, test_loader, device, pad_id=pad_id)
        results[name] = {
            'train': asdict(tr),
            'test': eval_res,
            'n_params': count_params(model),
        }
        print(f"  -> test loss={eval_res['loss']:.4f} tk_acc={eval_res['token_acc']:.4f} seq_acc={eval_res['seq_acc']:.4f}")
    return results


# ----------------------------------------------------------------------
# Experiment 4: Length extrapolation
# ----------------------------------------------------------------------

def experiment_extrapolation(seed: int = 0, epochs: int = 5, device_name: str = "cpu") -> Dict:
    """Train on short sequences, evaluate on longer ones."""
    set_seed(seed)
    device = get_device(device_name)
    print(f"\n=== Experiment 4: Length Extrapolation (device={device}) ===")

    # Train on lengths 4-20 pairs, test on 30, 40, 50, 60 pairs
    train_ds = DyckDataset(2000, min_pairs=4, max_pairs=20, num_bracket_types=1, seed=0)
    pad_id = train_ds.pad_id
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))

    models = {
        'DTFN-U': dc.make_dtfn_unidirectional(vocab_size, d=64, K=3),
        'DTFN-B': dc.make_dtfn_bidirectional(vocab_size, d=64, K=3),
        'LSTM': LSTMModel(vocab_size, d=64, hidden_dim=128),
        'Transformer': TransformerModel(vocab_size, d=64, n_heads=4, n_layers=2, max_len=256),
    }
    test_lengths = [20, 30, 40, 50, 60, 80, 100]
    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}")
        tr = train_model(model, train_loader, epochs=epochs, device=device,
                         lr=1e-3, pad_id=pad_id, seed=seed, verbose=False)
        results[name] = {'train': asdict(tr), 'extrap': {}}
        for L in test_lengths:
            ds = DyckDataset(200, min_pairs=L, max_pairs=L, num_bracket_types=1, seed=100+L)
            loader = DataLoader(ds, batch_size=32, shuffle=False,
                                collate_fn=lambda b: collate(b, pad_id))
            ev = evaluate_model(model, loader, device, pad_id=pad_id)
            results[name]['extrap'][L] = ev
            print(f"    {name:12s} L={L:3d} pairs: loss={ev['loss']:.4f} tk_acc={ev['token_acc']:.4f}")
    return results


# ----------------------------------------------------------------------
# Experiment 5: Copy task
# ----------------------------------------------------------------------

class CopyDataset(Dataset):
    """Copy task: input is random tokens, target is the same tokens."""
    def __init__(self, n: int, seq_len: int, vocab_size: int = 4, seed: int = 0) -> None:
        rng = random.Random(seed)
        self.pad_id = vocab_size
        self.bos_id = vocab_size + 1
        self.eos_id = vocab_size + 2
        self.vocab_size = vocab_size + 3
        self.samples = []
        for _ in range(n):
            seq = [rng.randrange(vocab_size) for _ in range(seq_len)]
            self.samples.append(seq)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx]
        x = torch.tensor([self.bos_id] + seq, dtype=torch.long)
        y = torch.tensor(seq + [self.eos_id], dtype=torch.long)
        return x, y


def experiment_copy(seed: int = 0, epochs: int = 5, device_name: str = "cpu") -> Dict:
    set_seed(seed)
    device = get_device(device_name)
    print(f"\n=== Experiment 5: Copy Task (device={device}) ===")
    train_ds = CopyDataset(1000, seq_len=20, vocab_size=4, seed=0)
    pad_id = train_ds.pad_id
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))

    models = {
        'DTFN-U': dc.make_dtfn_unidirectional(vocab_size, d=64, K=3),
        'DTFN-B': dc.make_dtfn_bidirectional(vocab_size, d=64, K=3),
        'LSTM': LSTMModel(vocab_size, d=64, hidden_dim=128),
    }
    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}")
        tr = train_model(model, train_loader, epochs=epochs, device=device,
                         lr=1e-3, pad_id=pad_id, seed=seed, verbose=False)
        # Test on different lengths
        results[name] = {'train': asdict(tr), 'extrap': {}}
        for L in [10, 20, 30, 40, 60, 80]:
            ds = CopyDataset(100, seq_len=L, vocab_size=4, seed=200+L)
            loader = DataLoader(ds, batch_size=32, shuffle=False,
                                collate_fn=lambda b: collate(b, pad_id))
            ev = evaluate_model(model, loader, device, pad_id=pad_id)
            results[name]['extrap'][L] = ev
            print(f"    {name:12s} L={L:3d}: tk_acc={ev['token_acc']:.4f}")
    return results


# ----------------------------------------------------------------------
# Experiment 6: Ablation on K
# ----------------------------------------------------------------------

def experiment_ablation_K(seed: int = 0, epochs: int = 5, device_name: str = "cpu") -> Dict:
    set_seed(seed)
    device = get_device(device_name)
    print(f"\n=== Experiment 6: Ablation on K (number of update steps) ===")
    train_ds = DyckDataset(2000, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=0)
    pad_id = train_ds.pad_id
    vocab_size = train_ds.vocab_size
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))
    test_ds = DyckDataset(500, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=42)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False,
                             collate_fn=lambda b: collate(b, pad_id))

    Ks = [1, 2, 3, 5, 8, 12]
    results = {}
    for K in Ks:
        model = dc.make_dtfn_unidirectional(vocab_size, d=64, K=K)
        print(f"\n  K={K} (params={count_params(model)})")
        tr = train_model(model, train_loader, epochs=epochs, device=device,
                         lr=1e-3, pad_id=pad_id, seed=seed, verbose=False)
        ev = evaluate_model(model, test_loader, device, pad_id=pad_id)
        results[K] = {'train': asdict(tr), 'test': ev, 'n_params': count_params(model)}
        print(f"  -> test loss={ev['loss']:.4f} tk_acc={ev['token_acc']:.4f}")
    return results


# ----------------------------------------------------------------------
# Experiment 7: Wall-clock scaling
# ----------------------------------------------------------------------

def experiment_scaling(seed: int = 0, device_name: str = "cpu") -> Dict:
    set_seed(seed)
    device = get_device(device_name)
    print(f"\n=== Experiment 7: Wall-clock Scaling vs Sequence Length ===")
    lengths = [128, 256, 512, 1024, 2048, 4096]
    results = {}
    for L in lengths:
        x = torch.randint(0, 4, (8, L), device=device)
        for name, model_factory in [
            ('DTFN-U', lambda: dc.make_dtfn_unidirectional(4, d=64, K=3)),
            ('DTFN-B', lambda: dc.make_dtfn_bidirectional(4, d=64, K=3)),
            ('LSTM', lambda: LSTMModel(4, d=64, hidden_dim=128)),
            ('Transformer', lambda: TransformerModel(4, d=64, n_heads=4, n_layers=2, max_len=8192)),
        ]:
            model = model_factory().to(device)
            # Warmup
            for _ in range(2):
                with torch.no_grad():
                    _ = model(x)
            t0 = time.time()
            for _ in range(5):
                with torch.no_grad():
                    _ = model(x)
            dt = (time.time() - t0) / 5
            results.setdefault(name, {})[L] = dt
            print(f"  {name:12s} L={L:5d}: {dt*1000:.2f} ms")
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=[
        "conservation", "needle", "dyck", "extrap", "copy",
        "ablation_K", "scaling", "all",
    ], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="/Users/daxu/papers/dtfn/results/results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    all_results: Dict = {}

    def save():
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    if args.experiment in ("conservation", "all"):
        all_results['conservation'] = experiment_conservation(args.seed)
        save()
    if args.experiment in ("needle", "all"):
        all_results['needle'] = experiment_needle(args.seed)
        save()
    if args.experiment in ("dyck", "all"):
        all_results['dyck'] = experiment_dyck(args.seed, args.epochs, args.device)
        save()
    if args.experiment in ("extrap", "all"):
        all_results['extrap'] = experiment_extrapolation(args.seed, args.epochs, args.device)
        save()
    if args.experiment in ("copy", "all"):
        all_results['copy'] = experiment_copy(args.seed, args.epochs, args.device)
        save()
    if args.experiment in ("ablation_K", "all"):
        all_results['ablation_K'] = experiment_ablation_K(args.seed, args.epochs, args.device)
        save()
    if args.experiment in ("scaling", "all"):
        all_results['scaling'] = experiment_scaling(args.seed, args.device)
        save()

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
