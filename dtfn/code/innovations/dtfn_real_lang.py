"""Real-language evaluation: Tiny Shakespeare character-level LM.

This module provides:
  - A character-level tokenizer for Tiny Shakespeare
  - A DTFN language model wrapper
  - Training and evaluation loops
  - Perplexity reporting

The goal is to demonstrate that DTFNs can learn real language structure
(not just synthetic Dyck grammars), and to compare DTFN variants
(U, B, MUSCL, Adaptive, WENO5) on a real-language benchmark.
"""
from __future__ import annotations

import math
import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from dtfn_core import (
    DTFN, DTFNConfig, make_dtfn_unidirectional, make_dtfn_bidirectional,
    make_dtfn_muscl, make_dtfn_full,
)


# ----------------------------------------------------------------------
# Tiny Shakespeare dataset
# ----------------------------------------------------------------------

TINY_SHAKESPEARE_URLS = [
    "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
]
TINY_SHAKESPEARE_CACHE = Path(__file__).parent.parent / "data" / "tinyshakespeare.txt"


def download_tiny_shakespeare() -> str:
    """Download Tiny Shakespeare (1 MB) and cache it locally.

    Tries multiple mirrors to be robust against rate limiting.
    """
    if TINY_SHAKESPEARE_CACHE.exists():
        return TINY_SHAKESPEARE_CACHE.read_text()
    TINY_SHAKESPEARE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in TINY_SHAKESPEARE_URLS:
        try:
            print(f"Downloading Tiny Shakespeare from {url}...")
            urllib.request.urlretrieve(url, TINY_SHAKESPEARE_CACHE)
            return TINY_SHAKESPEARE_CACHE.read_text()
        except Exception as e:
            print(f"  Failed: {e}")
            last_err = e
    raise RuntimeError(f"All download mirrors failed. Last error: {last_err}")


class CharTokenizer:
    """Simple character-level tokenizer."""

    def __init__(self, text: str):
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for i, c in enumerate(self.chars)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, idx: List[int]) -> str:
        return "".join(self.itos.get(i, "") for i in idx)


def prepare_shakespeare(text: str, seq_len: int = 64, train_frac: float = 0.9):
    """Split Tiny Shakespeare into train/val character-level sequences."""
    tok = CharTokenizer(text)
    data = tok.encode(text)
    n = len(data)
    n_train = int(train_frac * n)
    train_data = data[:n_train]
    val_data = data[n_train:]

    # Build sequences of length seq_len + 1 (input = first seq_len, target = last seq_len)
    def make_seqs(d: List[int]) -> List[List[int]]:
        return [d[i:i + seq_len + 1] for i in range(0, len(d) - seq_len - 1, seq_len)]

    train_seqs = make_seqs(train_data)
    val_seqs = make_seqs(val_data)
    return tok, train_seqs, val_seqs


# ----------------------------------------------------------------------
# DTFN language model
# ----------------------------------------------------------------------

class DTFNLanguageModel(nn.Module):
    """Wrap a DTFN as a character-level language model."""

    def __init__(self, vocab_size: int, d: int = 64, K: int = 3, variant: str = "U"):
        super().__init__()
        if variant == "U":
            self.dtfn = make_dtfn_unidirectional(vocab_size=vocab_size, d=d, K=K)
        elif variant == "B":
            self.dtfn = make_dtfn_bidirectional(vocab_size=vocab_size, d=d, K=K)
        elif variant == "MUSCL":
            self.dtfn = make_dtfn_muscl(vocab_size=vocab_size, d=d, K=K)
        elif variant == "FULL":
            self.dtfn = make_dtfn_full(vocab_size=vocab_size, d=d, K=K)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dtfn(x)


# ----------------------------------------------------------------------
# Training and evaluation
# ----------------------------------------------------------------------

def train_lm(
    model: nn.Module,
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 0,
    log_every: int = 1,
) -> Dict:
    """Train a DTFN LM and return training history."""
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    def batch_loss(seqs: List[List[int]], training: bool = True) -> Tuple[float, float]:
        total_loss = 0.0
        total_tokens = 0
        # Mini-batch
        for i in range(0, len(seqs), batch_size):
            batch = seqs[i:i + batch_size]
            if not batch:
                continue
            x = torch.tensor([s[:-1] for s in batch], device=device, dtype=torch.long)
            y = torch.tensor([s[1:] for s in batch], device=device, dtype=torch.long)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            if training:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item() * y.numel()
            total_tokens += y.numel()
        return total_loss / max(total_tokens, 1), total_tokens

    for epoch in range(epochs):
        import random
        rng = random.Random(seed + epoch)
        rng.shuffle(train_seqs)
        train_loss, train_tokens = batch_loss(train_seqs, training=True)
        with torch.no_grad():
            val_loss, val_tokens = batch_loss(val_seqs, training=False)
        # Compute perplexity
        train_ppl = math.exp(min(train_loss, 20.0))
        val_ppl = math.exp(min(val_loss, 20.0))
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_ppl": train_ppl,
            "val_ppl": val_ppl,
        }
        history.append(entry)
        if log_every > 0 and (epoch % log_every == 0 or epoch == epochs - 1):
            print(f"  Epoch {epoch}: train_loss={train_loss:.4f} (ppl={train_ppl:.2f}), "
                  f"val_loss={val_loss:.4f} (ppl={val_ppl:.2f})")

    return {"history": history, "final_val_loss": history[-1]["val_loss"],
            "final_val_ppl": history[-1]["val_ppl"]}


def run_shakespeare_experiment(
    variants: List[str] = None,
    d: int = 64,
    K: int = 3,
    epochs: int = 5,
    batch_size: int = 32,
    seq_len: int = 64,
    seeds: List[int] = None,
    device: str = "cpu",
) -> Dict:
    """Run the Tiny Shakespeare experiment across DTFN variants and seeds."""
    if variants is None:
        variants = ["U", "B", "MUSCL", "FULL"]
    if seeds is None:
        seeds = [0, 1, 2]

    text = download_tiny_shakespeare()
    tok, train_seqs, val_seqs = prepare_shakespeare(text, seq_len=seq_len)
    print(f"Tiny Shakespeare: vocab_size={tok.vocab_size}, "
          f"train_seqs={len(train_seqs)}, val_seqs={len(val_seqs)}, seq_len={seq_len}")

    results = {}
    for variant in variants:
        print(f"\n=== DTFN-{variant} ===")
        runs = []
        for seed in seeds:
            print(f"  Seed {seed}:")
            model = DTFNLanguageModel(vocab_size=tok.vocab_size, d=d, K=K, variant=variant)
            n_params = sum(p.numel() for p in model.parameters())
            res = train_lm(model, train_seqs, val_seqs, epochs=epochs,
                          batch_size=batch_size, device=device, seed=seed, log_every=1)
            res["n_params"] = n_params
            res["seed"] = seed
            runs.append(res)
            print(f"  -> final val_ppl={res['final_val_ppl']:.2f}")
        # Aggregate
        val_ppls = [r["final_val_ppl"] for r in runs]
        val_losses = [r["final_val_loss"] for r in runs]
        results[variant] = {
            "runs": runs,
            "val_ppl_mean": sum(val_ppls) / len(val_ppls),
            "val_ppl_std": (sum((p - sum(val_ppls)/len(val_ppls))**2 for p in val_ppls) / len(val_ppls))**0.5,
            "val_loss_mean": sum(val_losses) / len(val_losses),
            "val_loss_std": (sum((l - sum(val_losses)/len(val_losses))**2 for l in val_losses) / len(val_losses))**0.5,
            "n_params": runs[0]["n_params"],
        }
        print(f"  Summary: val_ppl={results[variant]['val_ppl_mean']:.2f} "
              f"± {results[variant]['val_ppl_std']:.2f}")

    return {"variants": results, "vocab_size": tok.vocab_size,
            "seq_len": seq_len, "epochs": epochs, "seeds": seeds}


if __name__ == "__main__":
    # Quick smoke test
    res = run_shakespeare_experiment(
        variants=["U", "MUSCL"],
        d=32, K=2, epochs=2, batch_size=16, seq_len=32, seeds=[0],
    )
    print("\nDone.")
