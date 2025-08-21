"""Experiments for Discrete Token–Flow Networks (DTFNs).

This script implements three sets of experiments described in the
accompanying paper:

1. A NumPy-based conservation study verifying conservation of mass and
   correct propagation for a simple upwind advection scheme.  The helper
   functions ``upwind_advection`` and ``run_reference`` can be executed
   directly to reproduce the statistics reported in the paper.

2. A PyTorch implementation of a DTFN model trained on Dyck–1 bracket
   sequences.  The classes ``DyckDataset``, ``FluxNet``, and ``DTFN``, along
   with the training loop in ``train_dtfn``, provide a minimal example
   demonstrating how to train a DTFN backbone.  The ``train_lstm`` routine
   trains a baseline LSTM model for comparison.  Ablations on the number
   of update steps ``K`` can be performed via a command-line argument.

To run the conservation study, invoke ``python dtfn_experiments.py --experiment reference``.
To train the Dyck–1 models, use ``--experiment dtfn`` or ``--experiment lstm``.
All experiments fix a random seed for reproducibility.
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------
# NumPy mechanism experiment
# ----------------------------------------------------------------------

def upwind_advection(mass: np.ndarray, v: float = 0.8, dt: float = 0.5,
                     steps: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve a non‐negative mass array using a simple upwind advection scheme.

    Parameters
    ----------
    mass : np.ndarray
        An array of shape (L, D) representing the initial mass at each of L
        positions and D dimensions.
    v : float, optional
        The advection velocity. Defaults to 0.8.
    dt : float, optional
        The time step used for the finite-difference update. Defaults to 0.5.
    steps : int, optional
        Number of update steps to perform. Defaults to 200.

    Returns
    -------
    m : np.ndarray
        The evolved mass array after ``steps`` iterations.
    errs : np.ndarray
        A 1D array of relative conservation errors at each time step.
    """
    L, D = mass.shape
    m = mass.copy().astype(float)
    errs: List[float] = []
    total0 = m.sum()
    for _ in range(steps):
        # compute flux at each interior interface. Upwind: F[i+1/2] = v * m[i]
        F = v * m[:-1]  # shape (L-1, D)
        dm = np.zeros_like(m)
        dm[:-1] -= F
        dm[1:] += F
        m = np.maximum(m + dt * dm, 0.0)
        errs.append(abs(m.sum() - total0) / total0)
    return m, np.array(errs)


def run_reference() -> None:
    """Run the conservation and needle tests and print summary statistics."""
    rng = np.random.default_rng(7)
    mass0 = rng.random((256, 100))
    mfin, errs = upwind_advection(mass0)
    print(f"Mean conservation error: {errs.mean():.3e}")
    print(f"Max conservation error: {errs.max():.3e}")
    print(f"Minimum mass: {mfin.min():.3e}")

    lengths = [64, 128, 256, 512]
    accs: List[float] = []
    v = 0.8
    dt = 0.5
    for L in lengths:
        ok = 0
        expected = int(round((L - 1) * v * dt))
        for _ in range(50):
            m0 = np.zeros((L, 1))
            m0[0] = 1.0
            mend, _ = upwind_advection(m0, v=v, dt=dt, steps=L - 1)
            pred = int(np.argmax(mend[:, 0]))
            if abs(pred - expected) <= 1:
                ok += 1
        accs.append(ok / 50.0)
    print("Needle accuracy:", dict(zip(lengths, accs)))


# ----------------------------------------------------------------------
# PyTorch Dyck-1 Training
# ----------------------------------------------------------------------

class DyckDataset(Dataset):
    """Dataset of Dyck–1 balanced parenthesis sequences.

    Each sample is a pair of tensors (input, target), where the input
    sequence begins with a special padding token (ID=2) and the target
    sequence ends with an end-of-sequence token (ID=3). Padding tokens are
    placed at the end of sequences to achieve uniform length within a batch.
    """

    def __init__(self, n: int, min_pairs: int = 2, max_pairs: int = 32,
                 seed: int = 0) -> None:
        self.vocab = ['(', ')']
        self.tok2id = {c: i for i, c in enumerate(self.vocab)}
        self.pad_id = 2
        self.eos_id = 3
        rnd = random.Random(seed)
        self.samples: List[List[int]] = []
        for _ in range(n):
            p = rnd.randint(min_pairs, max_pairs)
            self.samples.append(self._generate(p, rnd))

    def _generate(self, p: int, rnd: random.Random) -> List[int]:
        """Generate a random balanced parenthesis string with p pairs."""
        res: List[str] = []
        open_count = 0
        total_len = 2 * p
        for _ in range(total_len):
            if open_count == 0:
                res.append('(')
                open_count += 1
            elif len(res) == total_len - open_count:
                res.append(')')
                open_count -= 1
            elif rnd.random() < 0.5:
                res.append('(')
                open_count += 1
            else:
                res.append(')')
                open_count -= 1
        return [self.tok2id[c] for c in res]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.samples[idx]
        x = torch.tensor([self.pad_id] + seq, dtype=torch.long)
        y = torch.tensor(seq + [self.eos_id], dtype=torch.long)
        return x, y


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (input, target) pairs into uniformly sized tensors."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    x = torch.full((len(xs), max_len), fill_value=2, dtype=torch.long)
    y = torch.full((len(ys), max_len), fill_value=2, dtype=torch.long)
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        x[i, : xi.size(0)] = xi
        y[i, : yi.size(0)] = yi
    return x, y


class FluxNet(nn.Module):
    """Learned non-negative flux function for DTFN.

    Given the mass at adjacent positions, predicts the flux flowing from the
    left cell to the right cell.  A small multilayer perceptron produces
    an output vector which is passed through a Softplus nonlinearity to
    enforce non-negativity.  We do not explicitly constrain this mapping
    to be monotone in its inputs; rather, it provides a flexible
    parameterization of a non-negative numerical flux.  Exploring
    isotonic or monotone network architectures is left for future work.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.Tanh(),
            nn.Linear(d, d),
        )
        self.sp = nn.Softplus()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.sp(self.net(torch.cat([left, right], dim=-1)))


class DTFN(nn.Module):
    """Discrete Token–Flow Network module.

    Embeds tokens into non-negative mass vectors, applies a fixed number of
    finite-volume updates using a learned flux network, and decodes back
    to logits.  A learnable scalar controls the time step (via a
    sigmoid), ensuring stability.
    """

    def __init__(self, vocab_size: int, d: int = 128, K: int = 3) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.to_mass = nn.Sequential(nn.Linear(d, d), nn.Softplus())
        self.flux = FluxNet(d)
        self.cfl_raw = nn.Parameter(torch.tensor(0.0))
        self.dec = nn.Linear(d, vocab_size)
        self.K = K

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.to_mass(self.emb(x)) + 1e-6
        dt = torch.sigmoid(self.cfl_raw)
        for _ in range(self.K):
            left = m[:, :-1]
            right = m[:, 1:]
            F = self.flux(left, right)
            dm = torch.zeros_like(m)
            dm[:, :-1] -= F
            dm[:, 1:] += F
            m = torch.clamp(m + dt * dm, min=1e-6)
        return self.dec(m)


def sequence_accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute sequence-level accuracy: fraction of sequences exactly matched."""
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == y).all(dim=1)
    return correct.float().mean().item()


def train_dtfn(epochs: int = 2, device: str = "cpu", k: int = 3) -> Tuple[float, float]:
    """Train a small DTFN on Dyck–1 bracket sequences.

    Returns the final cross-entropy loss and sequence-level accuracy.
    """
    dataset = DyckDataset(2000, seed=0)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
    net = DTFN(vocab_size=4, d=64, K=k).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(ignore_index=2)
    for ep in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            B, L, V = logits.shape
            loss = ce(logits.view(B * L, V), yb.view(B * L))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += sequence_accuracy(logits, yb) * xb.size(0)
        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(dataset)
        print(f"[DTFN] Epoch {ep+1} loss {avg_loss:.4f} acc {avg_acc:.4f}")
    return avg_loss, avg_acc


def train_lstm(epochs: int = 2, device: str = "cpu", hidden_dim: int = 128) -> Tuple[float, float]:
    """Train a baseline LSTM on Dyck–1 bracket sequences.

    Returns the final cross-entropy loss and sequence-level accuracy.
    """
    dataset = DyckDataset(2000, seed=0)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate)
    vocab_size = 4
    emb = nn.Embedding(vocab_size, hidden_dim).to(device)
    lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True).to(device)
    dec = nn.Linear(hidden_dim, vocab_size).to(device)
    opt = torch.optim.Adam(list(emb.parameters()) + list(lstm.parameters()) + list(dec.parameters()), lr=1e-3)
    ce = nn.CrossEntropyLoss(ignore_index=2)
    for ep in range(epochs):
        total_loss = 0.0
        total_acc = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            h = emb(xb)
            h, _ = lstm(h)
            logits = dec(h)
            B, L, V = logits.shape
            loss = ce(logits.view(B * L, V), yb.view(B * L))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            total_acc += sequence_accuracy(logits, yb) * xb.size(0)
        avg_loss = total_loss / len(loader)
        avg_acc = total_acc / len(dataset)
        print(f"[LSTM] Epoch {ep+1} loss {avg_loss:.4f} acc {avg_acc:.4f}")
    return avg_loss, avg_acc


def main() -> None:
    parser = argparse.ArgumentParser(description="DTFN experiments.")
    parser.add_argument("--experiment", choices=["reference", "dtfn", "lstm"], default="reference",
                        help="Which experiment to run: reference, dtfn, or lstm.")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs for training models.")
    parser.add_argument("--k", type=int, default=3, help="Number of update steps K for DTFN.")
    parser.add_argument("--device", default="cpu", help="Device to train on (cpu or cuda).")
    args = parser.parse_args()
    set_seed(0)
    if args.experiment == "reference":
        run_reference()
    elif args.experiment == "dtfn":
        train_dtfn(epochs=args.epochs, device=args.device, k=args.k)
    elif args.experiment == "lstm":
        train_lstm(epochs=args.epochs, device=args.device)


if __name__ == "__main__":
    main()
