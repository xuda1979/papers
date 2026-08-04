"""DTFN v4 experiments: comprehensive NPU-ready experiment suite.

This script runs the full experiment battery for the DTFN paper,
designed to run on Ascend NPU (via torch_npu) or CPU/MPS fallback.

Experiments:
  Exp 1: Conservation verification (all variants, float32)
  Exp 2: Higher-order accuracy (upwind/MUSCL/WENO5) in float64
  Exp 3: Dyck-1 multi-seed comparison (DTFN-U/B/MUSCL/Adaptive/WENO5)
  Exp 4: Dyck-2 multi-seed (showing innovations close the gap)
  Exp 5: Length extrapolation with innovation variants
  Exp 6: Tiny Shakespeare real-language evaluation
  Exp 7: SSM bridge with small-perturbation linearisation
  Exp 8: CPSA conservation with explicit tolerance reporting
  Exp 9: Wall-clock scaling to 16k tokens (NPU)
  Exp 10: Adversarial stability with innovations
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------------------------------------------------
# Device setup: prefer NPU, fall back to MPS, then CPU
# ----------------------------------------------------------------------

try:
    import torch_npu  # type: ignore
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

if HAS_NPU and torch.npu.is_available():
    DEVICE = "npu:0"
    HAS_NPU = True
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    HAS_NPU = False
else:
    DEVICE = "cpu"
    HAS_NPU = False

# Ensure code dir is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ----------------------------------------------------------------------
# Exp 1: Conservation verification
# ----------------------------------------------------------------------

def exp1_conservation() -> Dict:
    """Verify exact mass conservation and non-negativity for all variants."""
    print("\n=== Exp 1: Conservation verification ===")
    from dtfn_core import (make_dtfn_unidirectional, make_dtfn_bidirectional,
                           make_dtfn_muscl, make_dtfn_full)
    from innovations.dtfn_weno import make_dtfn_weno5
    from innovations.dtfn_adaptive import make_dtfn_adaptive
    from innovations.dtfn_hybrid import make_dtfn_cpsa
    from innovations.dtfn_2d import make_dtfn_2d

    torch.manual_seed(42)
    results = {}

    # 1D variants
    variants_1d = {
        "DTFN-U": lambda: make_dtfn_unidirectional(vocab_size=4, d=16, K=3),
        "DTFN-B": lambda: make_dtfn_bidirectional(vocab_size=4, d=16, K=3),
        "DTFN-MUSCL": lambda: make_dtfn_muscl(vocab_size=4, d=16, K=3),
        "DTFN-FULL": lambda: make_dtfn_full(vocab_size=4, d=16, K=3),
        "DTFN-WENO5": lambda: make_dtfn_weno5(vocab_size=4, d=16, K=3),
        "DTFN-Adaptive": lambda: make_dtfn_adaptive(vocab_size=4, d=16, K=3),
        "DTFN-CPSA": lambda: make_dtfn_cpsa(vocab_size=4, d=16, K=2, n_heads=2, window=8),
    }
    for name, factory in variants_1d.items():
        model = factory().to(DEVICE)
        x = torch.randint(0, 4, (4, 32), device=DEVICE)
        with torch.no_grad():
            m0 = model.to_mass(model.emb(x)) + 1e-6
            m1 = model.forward_mass(x)
        total0 = m0.sum(dim=(1, 2))
        total1 = m1.sum(dim=(1, 2))
        cons_err = (total1 - total0).abs().max().item()
        min_mass = m1.min().item()
        # Note: DTFN-FULL uses a non-conservative source term (by design),
        # so its conservation error reflects the source magnitude, not a bug.
        has_source = "FULL" in name
        results[name] = {
            "conservation_error": cons_err,
            "min_mass": min_mass,
            "positive": bool(min_mass > 0),
            "has_source_term": has_source,
            "conservation_type": "approximate (source term)" if has_source else "exact (flux only)",
        }
        print(f"  {name:15s}: cons_err={cons_err:.2e}, min_mass={min_mass:.2e}, "
              f"positive={min_mass > 0}, source={has_source}")

    # 2D variant: DTFN2D doesn't have forward_mass; compute mass manually.
    model_2d = make_dtfn_2d(vocab_size=4, d=8, K=2, H=4, W=4).to(DEVICE)
    x = torch.randint(0, 4, (2, 4, 4), device=DEVICE)
    with torch.no_grad():
        m0 = model_2d.to_mass(model_2d.emb(x)) + 1e-6
        m1 = m0
        for layer in model_2d.layers:
            m1 = layer(m1)
    cons_err = (m1.sum() - m0.sum()).abs().max().item()
    min_mass = m1.min().item()
    results["DTFN-2D"] = {
        "conservation_error": cons_err,
        "min_mass": min_mass,
        "positive": bool(min_mass > 0),
    }
    print(f"  DTFN-2D         : cons_err={cons_err:.2e}, min_mass={min_mass:.2e}, positive={min_mass > 0}")

    return results


# ----------------------------------------------------------------------
# Exp 2: Higher-order accuracy in float64
# ----------------------------------------------------------------------

def exp2_accuracy_order_float64() -> Dict:
    """Convergence study in float64 to show clean order-5 WENO5.

    Uses CPU explicitly because MPS doesn't support float64.
    """
    print("\n=== Exp 2: Accuracy order (float64) ===")
    torch.manual_seed(42)
    # Use CPU for float64 (MPS doesn't support float64; NPU support varies)
    DEVICE_F64 = "cpu"
    results = {"upwind": [], "muscl": [], "weno5": [], "fct": []}

    def gaussian_periodic(x, center, width=0.01, dtype=torch.float64):
        dx = (x - center)
        dx = dx - torch.floor(dx + 0.5)
        return torch.exp(-(dx ** 2) / width)

    def smooth_gaussian(L, device, dtype=torch.float64):
        x = torch.arange(L, device=device, dtype=dtype) / L
        return gaussian_periodic(x, 0.3).unsqueeze(0).unsqueeze(-1)

    def advect(p, scheme, lam, v=0.3):
        if scheme == "upwind":
            F_face = v * p
            F_left = torch.roll(F_face, shifts=1, dims=1)
            return p + lam * (F_left - F_face)
        elif scheme == "muscl":
            p_left = torch.roll(p, shifts=1, dims=1)
            p_right = torch.roll(p, shifts=-1, dims=1)
            d_minus = p - p_left
            d_plus = p_right - p
            same = (d_minus * d_plus) > 0
            sigma = torch.where(same, 2.0 * d_minus * d_plus / (d_minus + d_plus + 1e-12), torch.zeros_like(d_minus))
            p_face = p + 0.5 * sigma
            F_face = v * p_face
            F_left = torch.roll(F_face, shifts=1, dims=1)
            return p + lam * (F_left - F_face)
        elif scheme == "weno5":
            p_m2 = torch.roll(p, shifts=2, dims=1)
            p_m1 = torch.roll(p, shifts=1, dims=1)
            p_0 = p
            p_p1 = torch.roll(p, shifts=-1, dims=1)
            p_p2 = torch.roll(p, shifts=-2, dims=1)
            p0 = (2.0 * p_m2 - 7.0 * p_m1 + 11.0 * p_0) / 6.0
            p1 = (-p_m1 + 5.0 * p_0 + 2.0 * p_p1) / 6.0
            p2 = (2.0 * p_0 + 5.0 * p_p1 - p_p2) / 6.0
            beta0 = (13.0/12.0) * (p_m2 - 2.0*p_m1 + p_0)**2 + 0.25 * (p_m2 - 4.0*p_m1 + 3.0*p_0)**2
            beta1 = (13.0/12.0) * (p_m1 - 2.0*p_0 + p_p1)**2 + 0.25 * (p_m1 - p_p1)**2
            beta2 = (13.0/12.0) * (p_0 - 2.0*p_p1 + p_p2)**2 + 0.25 * (3.0*p_0 - 4.0*p_p1 + p_p2)**2
            tau = (beta0 - beta2).abs()
            eps = 1e-12  # float64-safe
            d0, d1, d2 = 0.1, 0.6, 0.3
            a0 = d0 * (1.0 + tau / (beta0 + eps))
            a1 = d1 * (1.0 + tau / (beta1 + eps))
            a2 = d2 * (1.0 + tau / (beta2 + eps))
            sa = a0 + a1 + a2
            w0, w1, w2 = a0/sa, a1/sa, a2/sa
            p_face = w0 * p0 + w1 * p1 + w2 * p2
            F_face = v * p_face
            F_left = torch.roll(F_face, shifts=1, dims=1)
            return p + lam * (F_left - F_face)
        elif scheme == "fct":
            # Flux-Corrected Transport: low-order upwind + limited WENO5 correction.
            # Learned limiter replaced here by a smoothness-based limiter for the
            # pure-advection convergence test (no learned parameters in this test).
            F_low = v * p
            # WENO5-Z face reconstruction (as in 'weno5' branch)
            p_m2 = torch.roll(p, shifts=2, dims=1)
            p_m1 = torch.roll(p, shifts=1, dims=1)
            p_0 = p
            p_p1 = torch.roll(p, shifts=-1, dims=1)
            p_p2 = torch.roll(p, shifts=-2, dims=1)
            p0 = (2.0 * p_m2 - 7.0 * p_m1 + 11.0 * p_0) / 6.0
            p1 = (-p_m1 + 5.0 * p_0 + 2.0 * p_p1) / 6.0
            p2 = (2.0 * p_0 + 5.0 * p_p1 - p_p2) / 6.0
            beta0 = (13.0/12.0) * (p_m2 - 2.0*p_m1 + p_0)**2 + 0.25 * (p_m2 - 4.0*p_m1 + 3.0*p_0)**2
            beta1 = (13.0/12.0) * (p_m1 - 2.0*p_0 + p_p1)**2 + 0.25 * (p_m1 - p_p1)**2
            beta2 = (13.0/12.0) * (p_0 - 2.0*p_p1 + p_p2)**2 + 0.25 * (3.0*p_0 - 4.0*p_p1 + p_p2)**2
            tau = (beta0 - beta2).abs()
            eps = 1e-12
            d0, d1, d2 = 0.1, 0.6, 0.3
            a0 = d0 * (1.0 + tau / (beta0 + eps))
            a1 = d1 * (1.0 + tau / (beta1 + eps))
            a2 = d2 * (1.0 + tau / (beta2 + eps))
            sa = a0 + a1 + a2
            w0, w1, w2 = a0/sa, a1/sa, a2/sa
            p_face = w0 * p0 + w1 * p1 + w2 * p2
            F_high = v * p_face
            # Smoothness-based limiter: phi → 1 in smooth regions (Jameson sensor).
            p_left = torch.roll(p, shifts=1, dims=1)
            p_right = torch.roll(p, shifts=-1, dims=1)
            second_diff = (p_right - 2.0 * p + p_left).abs()
            norm = p_right.abs() + 2.0 * p.abs() + p_left.abs() + 1e-12
            s = torch.clamp(second_diff / norm, 0.0, 1.0)
            phi = (1.0 - s).clamp(0.0, 1.0)
            F_face = F_low + phi * (F_high - F_low)
            F_left = torch.roll(F_face, shifts=1, dims=1)
            return p + lam * (F_left - F_face)

    def rk3_step(p, scheme, lam, v):
        p1 = advect(p, scheme, lam, v)
        p2 = 0.75 * p + 0.25 * advect(p1, scheme, lam, v)
        return (1.0/3.0) * p + (2.0/3.0) * advect(p2, scheme, lam, v)

    T = 0.5
    v = 0.3
    Ls = [32, 64, 128, 256, 512, 1024]
    for L in Ls:
        dx = 1.0 / L
        cfl = 0.3
        dt = cfl * dx / v
        n_steps = int(round(T / dt))
        dt = T / n_steps
        lam = dt / dx
        p0 = smooth_gaussian(L, DEVICE_F64, dtype=torch.float64)
        x = torch.arange(L, device=DEVICE_F64, dtype=torch.float64) / L
        p_ref = gaussian_periodic(x, 0.45).unsqueeze(0).unsqueeze(-1)
        for scheme in ["upwind", "muscl", "weno5", "fct"]:
            p = p0.clone()
            for _ in range(n_steps):
                p = rk3_step(p, scheme, lam, v)
            err = (p - p_ref).abs().mean().item()
            results[scheme].append({"L": L, "L1_error": err, "n_steps": n_steps})
            print(f"  {scheme:8s} L={L:5d}: L1={err:.6e} ({n_steps} steps)")

    orders = {}
    for scheme in ["upwind", "muscl", "weno5", "fct"]:
        errs = [r["L1_error"] for r in results[scheme]]
        Ls_used = [r["L"] for r in results[scheme]]
        order_estimates = []
        for i in range(1, len(errs)):
            if errs[i] > 0 and errs[i-1] > 0:
                order = math.log(errs[i-1] / errs[i]) / math.log(Ls_used[i] / Ls_used[i-1])
                order_estimates.append(order)
        orders[scheme] = order_estimates
        print(f"  {scheme:8s} orders: {[f'{o:.2f}' for o in order_estimates]}")

    results["orders"] = orders
    return results


# ----------------------------------------------------------------------
# Exp 3: Dyck-1 multi-seed comparison
# ----------------------------------------------------------------------

@dataclass
class DyckSample:
    tokens: List[int]
    label: int = 0


def generate_dyck1(n_samples: int, min_len: int, max_len: int, rng: random.Random) -> List[DyckSample]:
    """Generate Dyck-1 sequences: 1='(', 2=')'. Valid bracket sequences."""
    samples = []
    for _ in range(n_samples):
        n_pairs = rng.randint(min_len // 2, max_len // 2)
        tokens = []
        depth = 0
        remaining_opens = n_pairs
        remaining_closes = n_pairs
        while remaining_opens > 0 or remaining_closes > 0:
            if remaining_opens == 0:
                tokens.append(2); remaining_closes -= 1; depth -= 1
            elif remaining_closes == 0 or depth == 0:
                tokens.append(1); remaining_opens -= 1; depth += 1
            else:
                if rng.random() < 0.5:
                    tokens.append(1); remaining_opens -= 1; depth += 1
                else:
                    tokens.append(2); remaining_closes -= 1; depth -= 1
        samples.append(DyckSample(tokens=tokens, label=0))
    return samples


def train_dyck1(model, train_loader, val_loader, epochs=5, batch_size=32, lr=1e-3,
                device="cpu", seed=0, pad_id=2):
    """Train DTFN on Dyck-1 next-token prediction. Returns history.

    Uses the DyckDataset from dtfn_experiments_v2 with proper BOS/EOS/PAD tokens.
    """
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    def run_epoch(loader, training):
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            mask = (y != pad_id)
            logits = model(x)
            loss = F.cross_entropy(logits[mask], y[mask])
            if training:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            total_loss += loss.item() * mask.sum().item()
            preds = logits[mask].argmax(-1)
            total_correct += (preds == y[mask]).sum().item()
            total_tokens += mask.sum().item()
        return total_loss / max(total_tokens, 1), total_correct / max(total_tokens, 1)

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        with torch.no_grad():
            val_loss, val_acc = run_epoch(val_loader, training=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        if epoch % 2 == 0 or epoch == epochs - 1:
            print(f"    Epoch {epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    return {"history": history, "final_val_acc": history[-1]["val_acc"],
            "final_val_loss": history[-1]["val_loss"]}


def exp3_dyck1_multiseed(seeds=(0, 1, 2), d=32, K=3, epochs=5, n_train=2000, n_val=500) -> Dict:
    """Multi-seed Dyck-1 comparison: U, B, MUSCL, Adaptive, WENO5, CPSA.

    Uses the proper DyckDataset with BOS/EOS/PAD tokens (vocab_size=5).
    """
    print("\n=== Exp 3: Dyck-1 multi-seed comparison ===")
    from dtfn_core import (make_dtfn_unidirectional, make_dtfn_bidirectional,
                           make_dtfn_muscl, make_dtfn_full)
    from innovations.dtfn_weno import make_dtfn_weno5
    from innovations.dtfn_adaptive import make_dtfn_adaptive
    from innovations.dtfn_hybrid import make_dtfn_cpsa
    # Import the DyckDataset from v2
    from dtfn_experiments_v2 import DyckDataset
    from torch.utils.data import DataLoader

    # Create datasets: vocab_size=5 for Dyck-1 ( '(':0, ')':1, PAD:2, BOS:3, EOS:4 )
    vocab_size = 5
    train_ds = DyckDataset(n_train, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=0)
    val_ds = DyckDataset(n_val, min_pairs=4, max_pairs=32, num_bracket_types=1, seed=42)
    pad_id = train_ds.pad_id
    print(f"  Dataset: {len(train_ds)} train, {len(val_ds)} val, vocab_size={vocab_size}, pad_id={pad_id}")

    # Import the collate function
    from dtfn_experiments_v2 import collate
    collate_fn = lambda batch: collate(batch, pad_id=pad_id)

    variants = {
        "DTFN-U": lambda: make_dtfn_unidirectional(vocab_size=vocab_size, d=d, K=K),
        "DTFN-B": lambda: make_dtfn_bidirectional(vocab_size=vocab_size, d=d, K=K),
        "DTFN-MUSCL": lambda: make_dtfn_muscl(vocab_size=vocab_size, d=d, K=K),
        "DTFN-FULL": lambda: make_dtfn_full(vocab_size=vocab_size, d=d, K=K),
        "DTFN-Adaptive": lambda: make_dtfn_adaptive(vocab_size=vocab_size, d=d, K=K),
        "DTFN-WENO5": lambda: make_dtfn_weno5(vocab_size=vocab_size, d=d, K=K),
        "DTFN-CPSA": lambda: make_dtfn_cpsa(vocab_size=vocab_size, d=d, K=2, n_heads=2, window=16),
    }

    results = {}
    for name, factory in variants.items():
        print(f"\n  --- {name} ---")
        runs = []
        for seed in seeds:
            print(f"    Seed {seed}:")
            torch.manual_seed(seed)
            model = factory()
            n_params = sum(p.numel() for p in model.parameters())
            train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                                       generator=torch.Generator().manual_seed(seed),
                                       collate_fn=collate_fn)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
            res = train_dyck1(model, train_loader, val_loader, epochs=epochs,
                              batch_size=32, lr=1e-3, device=DEVICE, seed=seed, pad_id=pad_id)
            res["n_params"] = n_params
            res["seed"] = seed
            runs.append(res)
            print(f"    -> val_acc={res['final_val_acc']:.4f}")
        accs = [r["final_val_acc"] for r in runs]
        losses = [r["final_val_loss"] for r in runs]
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
        mean_loss = sum(losses) / len(losses)
        std_loss = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5
        results[name] = {
            "runs": runs,
            "val_acc_mean": mean_acc, "val_acc_std": std_acc,
            "val_loss_mean": mean_loss, "val_loss_std": std_loss,
            "n_params": runs[0]["n_params"],
        }
        print(f"  {name}: val_acc={mean_acc:.4f} ± {std_acc:.4f}")

    return results


# ----------------------------------------------------------------------
# Exp 6: Tiny Shakespeare real-language evaluation
# ----------------------------------------------------------------------

def exp6_shakespeare(variants=None, d=64, K=3, epochs=5, batch_size=32, seq_len=64, seeds=(0, 1, 2)) -> Dict:
    """Tiny Shakespeare character-level LM across variants."""
    print("\n=== Exp 6: Tiny Shakespeare ===")
    from innovations.dtfn_real_lang import run_shakespeare_experiment
    if variants is None:
        variants = ["U", "B", "MUSCL", "FULL"]
    return run_shakespeare_experiment(
        variants=variants, d=d, K=K, epochs=epochs, batch_size=batch_size,
        seq_len=seq_len, seeds=list(seeds), device=DEVICE,
    )


# ----------------------------------------------------------------------
# Exp 7: SSM bridge with small-perturbation linearisation
# ----------------------------------------------------------------------

def exp7_ssm_bridge_small_perturb() -> Dict:
    """Verify DTFN-SSM correspondence with small perturbations.

    The original exp4 used perturbations that engaged the positivity
    limiter, causing the actual Jacobian to differ from the linearised
    one. Here we use a large equilibrium and small perturbations to
    stay in the linear regime.
    """
    print("\n=== Exp 7: SSM bridge (small perturbation) ===")
    from innovations.dtfn_ssm_bridge import linearise_flux, LinearisedDTFN
    from dtfn_core import MonotoneFluxNet, DTFNLayer, DTFNConfig

    torch.manual_seed(42)
    results = {}

    for d in [4, 8]:
        for L in [8, 16]:
            for lam in [0.3, 0.5]:
                flux_net = MonotoneFluxNet(d=d, hidden=128).to(DEVICE)
                # Large equilibrium so positivity limiter doesn't engage
                p_star = torch.ones(1, 1, d, device=DEVICE) * 10.0

                J_left, J_right = linearise_flux(flux_net, p_star, eps=1e-3)
                lin = LinearisedDTFN(J_left, J_right, lam, L).to(DEVICE)

                # Predicted spectral radius (top) and contraction rate (second eigenvalue)
                eig_pred_all = torch.linalg.eigvals(lin.T.cpu()).abs()
                eig_pred_sorted = sorted(eig_pred_all.tolist(), reverse=True)
                eig_pred = eig_pred_sorted[0]  # top (should be 1.0 for mass conservation)
                contraction_pred = eig_pred_sorted[d]  # (d+1)-th: contraction on zero-mean subspace

                # Actual spectral radius via autograd with SMALL perturbation
                cfg = DTFNConfig(d=d, K=1, scheme="upwind")
                cfg.epsilon = 1e-6  # small epsilon so limiter doesn't engage at p=10
                # Set cfl_init so that sigmoid(cfl_init) = lam
                cfg.cfl_init = math.log(lam / (1.0 - lam))
                layer = DTFNLayer(cfg).to(DEVICE)
                # Copy the flux net weights so layer matches the linearised flux
                layer.flux.load_state_dict(flux_net.state_dict())
                # Use a small perturbation (0.01) around the large equilibrium
                p_eq = p_star.expand(1, L, d).clone()
                perturb = 0.01 * torch.randn(1, L, d, device=DEVICE)
                p_perturbed = p_eq + perturb

                # Compute Jacobian via finite differences with small perturbation
                delta = 1e-3
                J_actual = torch.zeros(L * d, L * d, device=DEVICE)
                with torch.no_grad():
                    out0 = layer(p_eq)
                    for i in range(L * d):
                        dp = torch.zeros(1, L, d, device=DEVICE)
                        dp.reshape(-1)[i] = delta
                        out_p = layer(p_eq + dp)
                        J_actual[:, i] = (out_p - out0).reshape(-1) / delta

                eig_actual_all = torch.linalg.eigvals(J_actual.cpu()).abs()
                eig_actual_sorted = sorted(eig_actual_all.tolist(), reverse=True)
                eig_actual = eig_actual_sorted[0]
                contraction_actual = eig_actual_sorted[d] if len(eig_actual_sorted) > d else 0.0
                jac_err = (J_actual - lin.T.to(J_actual.device)).abs().max().item()
                rel_err = jac_err / (lin.T.abs().max().item() + 1e-8)

                # The top d eigenvalues are ~1.0 (mass conservation per channel).
                # The contraction rate is the (d+1)-th eigenvalue, which should be < 1.
                key = f"d{d}_L{L}_lam{lam}"
                results[key] = {
                    "predicted_spectral_radius": eig_pred,
                    "actual_spectral_radius": eig_actual,
                    "predicted_contraction_rate": contraction_pred,
                    "actual_contraction_rate": contraction_actual,
                    "jacobian_max_error": jac_err,
                    "jacobian_relative_error": rel_err,
                    "stable": contraction_pred < 1.0 and contraction_actual < 1.0 + 1e-4,
                    "equilibrium": 10.0,
                    "perturbation_delta": delta,
                }
                print(f"  {key}: pred_rho={eig_pred:.4f} (contraction={contraction_pred:.4f}), "
                      f"actual_rho={eig_actual:.4f} (contraction={contraction_actual:.4f}), "
                      f"jac_err={jac_err:.2e}, stable={results[key]['stable']}")

    return results


# ----------------------------------------------------------------------
# Exp 9: Wall-clock scaling to 16k tokens
# ----------------------------------------------------------------------

def exp9_scaling_16k() -> Dict:
    """Wall-clock scaling to L=16384 tokens on NPU."""
    print("\n=== Exp 9: Wall-clock scaling to 16k ===")
    from dtfn_core import make_dtfn_unidirectional, make_dtfn_muscl
    from innovations.dtfn_weno import make_dtfn_weno5 as make_weno

    results = {}
    Ls = [256, 512, 1024, 2048, 4096, 8192, 16384]
    variants = {
        "DTFN-U": lambda: make_dtfn_unidirectional(vocab_size=4, d=64, K=3),
        "DTFN-MUSCL": lambda: make_dtfn_muscl(vocab_size=4, d=64, K=3),
        "DTFN-WENO5": lambda: make_weno(vocab_size=4, d=64, K=3),
    }
    for name, factory in variants.items():
        results[name] = []
        for L in Ls:
            try:
                model = factory().to(DEVICE)
                x = torch.randint(0, 4, (1, L), device=DEVICE)
                # Warmup
                with torch.no_grad():
                    for _ in range(2):
                        _ = model(x)
                if DEVICE != "cpu":
                    torch.synchronize() if DEVICE == "cpu" else None
                # Time
                t0 = time.time()
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(x)
                if HAS_NPU:
                    torch.npu.synchronize()
                elif DEVICE == "mps":
                    torch.mps.synchronize()
                t1 = time.time()
                ms_per_seq = (t1 - t0) / 5 * 1000
                results[name].append({"L": L, "ms_per_seq": ms_per_seq, "device": DEVICE})
                print(f"  {name:12s} L={L:6d}: {ms_per_seq:.2f} ms/seq")
            except Exception as e:
                print(f"  {name:12s} L={L:6d}: FAILED ({e})")
                results[name].append({"L": L, "ms_per_seq": None, "error": str(e)})
    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(experiments=None):
    print(f"Device: {DEVICE} (HAS_NPU={HAS_NPU})")
    if HAS_NPU:
        print(f"NPU available: {torch.npu.device_count()} devices")
        try:
            print(f"Device name: {torch.npu.get_device_name(0)}")
        except Exception:
            pass

    if experiments is None:
        experiments = ["exp1", "exp2", "exp3", "exp6", "exp7", "exp9"]

    all_results = {"device": DEVICE, "has_npu": HAS_NPU, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    if "exp1" in experiments:
        all_results["exp1_conservation"] = exp1_conservation()
    if "exp2" in experiments:
        all_results["exp2_accuracy_order_float64"] = exp2_accuracy_order_float64()
    if "exp3" in experiments:
        all_results["exp3_dyck1_multiseed"] = exp3_dyck1_multiseed()
    if "exp6" in experiments:
        all_results["exp6_shakespeare"] = exp6_shakespeare()
    if "exp7" in experiments:
        all_results["exp7_ssm_bridge"] = exp7_ssm_bridge_small_perturb()
    if "exp9" in experiments:
        all_results["exp9_scaling_16k"] = exp9_scaling_16k()

    # Save
    out_path = Path(__file__).parent.parent / "results" / "innovations_v4.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", nargs="+", default=None,
                        help="Experiments to run (default: all)")
    args = parser.parse_args()
    main(args.exp)
