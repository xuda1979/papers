"""
Experiments for Discrete Token–Flow Networks (DTFNs).

This module implements two sets of experiments described in the paper:

1. A NumPy-based mechanism study verifying conservation of mass and
   correct propagation for a simple upwind advection scheme. The helper
   functions `upwind_advection` and `run_reference` can be executed
   directly to reproduce the statistics reported in the paper.

2. A PyTorch implementation of a DTFN model trained on Dyck–1 bracket
   sequences. The classes `DyckDataset`, `FluxNet`, and `DTFN`, along
   with the training loop in `train_dftn`, provide a minimal example
   demonstrating how to train a DTFN backbone.

To run the conservation study, simply invoke ``python dtfn_experiments.py``.
To train the Dyck–1 model, uncomment the call to ``train_dftn()`` at
the bottom of this file. These experiments are intentionally kept small
and self-contained; more extensive evaluations can build upon this
codebase.
"""

from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# NumPy mechanism experiment
# -----------------------------------------------------------------------------

def upwind_advection(mass: np.ndarray, v: float = 0.8, dt: float = 0.5,
                     steps: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Evolve a non‑negative mass array using a simple upwind advection scheme.

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
        The evolved mass array after `steps` iterations.
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
        # interior updates
        dm[1:-1] = F[:-1] - F[1:]
        # boundary conditions: outflow only
        dm[0] = -F[0]
        dm[-1] = F[-1]
        m = np.maximum(m + dt * dm, 0.0)
        errs.append(abs(m.sum() - total0) / total0)
    return m, np.array(errs)


def run_reference() -> None:
    """Run the conservation and needle tests and print summary statistics."""
    rng = np.random.default_rng(7)
    mass0 = rng.random((256, 8))
    mfin, errs = upwind_advection(mass0)
    print("Mean conservation error:", errs.mean())
    print("Max conservation error:", errs.max())
    print("Minimum mass:", mfin.min())

    lengths = [64, 128, 256, 512]
    accs: List[float] = []
    v = 0.8
    dt = 0.5
    for L in lengths:
        ok = 0
        expected = int(round((L - 1) * v * dt))
        for _ in range(50):
            m0 = np.zeros((L, 4))
            m0[0, 0] = 1.0
            mend, _ = upwind_advection(m0, v=v, dt=dt, steps=L - 1)
            pred = int(np.argmax(mend[:, 0]))
            if abs(pred - expected) <= 1:
                ok += 1
        accs.append(ok / 50.0)
    print("Needle accuracies:", dict(zip(lengths, accs)))


# -----------------------------------------------------------------------------
# PyTorch Dyck–1 Training
# -----------------------------------------------------------------------------

class DyckDataset(Dataset):
    """Dataset of Dyck–1 balanced parenthesis sequences.

    Each sample is a pair of tensors (input, target), where the input
    sequence begins with a special padding token (ID=2) and the target
    sequence ends with an end-of-sequence token (ID=3). Padding tokens are
    placed at the end of both sequences to achieve uniform length within a
    batch.
    """

    def __init__(self, num: int, min_pairs: int = 2, max_pairs: int = 32,
                 seed: int = 0) -> None:
        self.vocab = ['(', ')']
        self.tok2id = {c: i for i, c in enumerate(self.vocab)}
        self.pad_id = 2
        self.eos_id = 3
        rnd = random.Random(seed)
        self.samples: List[List[int]] = []
        for _ in range(num):
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
            elif open_count == (total_len - len(res)):
                res.append(')')
                open_count -= 1
            else:
                if rnd.random() < 0.5:
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


def collate(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of (input, target) pairs into uniformly sized tensors."""
    xs, ys = zip(*batch)
    max_len = max(x.size(0) for x in xs)
    X = torch.full((len(xs), max_len), fill_value=2, dtype=torch.long)
    Y = torch.full((len(xs), max_len), fill_value=2, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        X[i, : x.size(0)] = x
        Y[i, : y.size(0)] = y
    return X, Y


class FluxNet(nn.Module):
    """Learned monotone flux function for DTFN.

    Given the mass at adjacent positions, predicts the flux flowing from the
    left cell to the right cell. The output is passed through a Softplus
    nonlinearity to guarantee non‑negativity.
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d, d), nn.Tanh(), nn.Linear(d, d)
        )
        self.sp = nn.Softplus()

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.sp(self.net(torch.cat([left, right], dim=-1)))


class DTFN(nn.Module):
    """Discrete Token–Flow Network module.

    Embeds tokens into positive mass vectors, applies a fixed number of
    finite–volume updates using a learned flux network, and decodes back
    to logits. A learnable sigmoid parameter controls the CFL number
    (time step), ensuring stability.
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
        for _ in range(self.K):
            left = m[:, :-1]
            right = m[:, 1:]
            F = self.flux(left, right)
            dt = torch.sigmoid(self.cfl_raw)
            dm = torch.zeros_like(m)
            dm[:, 1:-1] = F[:, :-1] - F[:, 1:]
            dm[:, 0] = -F[:, 0]
            dm[:, -1] = F[:, -1]
            m = torch.clamp(m + dt * dm, min=1e-6)
        return self.dec(m)


def train_dftn(epochs: int = 2, device: str = "cpu") -> None:
    """Train a small DTFN on Dyck–1 bracket sequences."""
    train_loader = DataLoader(
        DyckDataset(2000), batch_size=32, shuffle=True, collate_fn=collate
    )
    net = DTFN(vocab_size=4, d=64).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss(ignore_index=2)
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = net(xb)
            B, L, V = logits.shape
            loss = ce(logits.view(B * L, V), yb.view(B * L))
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {ep}: loss {total_loss / len(train_loader):.4f}")


if __name__ == "__main__":
    # Run the NumPy mechanism experiment. Comment this out if you only want
    # to train the PyTorch model.
    run_reference()
    # Uncomment the following line to train the DTFN on Dyck–1 sequences.
    # train_dftn()
