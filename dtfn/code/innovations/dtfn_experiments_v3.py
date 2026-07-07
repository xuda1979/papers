"""Comprehensive experiments for DTFN innovations on Ascend NPU.

This script runs the new architectural contributions:

  Exp 1: Conservation verification for all new variants (WENO5, CPSA, 2D, Adaptive)
  Exp 2: Higher-order accuracy comparison (upwind vs MUSCL vs WENO5)
  Exp 3: Dyck-1 with new variants (vs base DTFN, LSTM, Transformer)
  Exp 4: DTFN-SSM correspondence verification
  Exp 5: 2D DTFN on image-like patterns
  Exp 6: Adaptive vs uniform CFL on variable-difficulty Dyck
  Exp 7: CPSA-DTFN on long-range dependency tasks

All experiments run on a single Ascend 910B NPU (npu:0).
"""
from __future__ import annotations

# Set MPS fallback BEFORE importing torch (MPS lacks linalg_eig etc.).
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Detect NPU (Ascend) or MPS (Apple Silicon Neural Engine) or CPU
HAS_NPU = False
HAS_MPS = False
try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    pass
if not HAS_NPU and torch.backends.mps.is_available():
    HAS_MPS = True

# Honour the user constraint: at most 1 NPU. We pick the first available
# accelerator (Ascend NPU > Apple Silicon NE/GPU via MPS > CPU).
if HAS_NPU:
    DEVICE = torch.device("npu:0")
elif HAS_MPS:
    DEVICE = torch.device("mps:0")
    # MPS lacks some ops (e.g. linalg_eig); fall back to CPU for those.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}  (NPU={HAS_NPU}, MPS={HAS_MPS})")

# Add the parent directory to sys.path so we can import dtfn_core
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dtfn_core import (
    DTFN, DTFNConfig, make_dtfn_unidirectional, make_dtfn_bidirectional,
    make_dtfn_muscl, make_dtfn_full,
)
from innovations.dtfn_weno import WENOZLayer, make_dtfn_weno5
from innovations.dtfn_hybrid import CPSADTFNLayer, make_dtfn_cpsa
from innovations.dtfn_2d import DTFN2D, DTFNLayer2D, make_dtfn_2d
from innovations.dtfn_adaptive import AdaptiveDTFNLayer, make_dtfn_adaptive


# ----------------------------------------------------------------------
# Experiment 1: Conservation verification
# ----------------------------------------------------------------------

def exp1_conservation() -> Dict:
    """Verify exact mass conservation and positivity for all new variants."""
    print("\n=== Exp 1: Conservation verification ===")
    torch.manual_seed(42)
    results = {}
    variants = {
        "WENO5": make_dtfn_weno5(vocab_size=4, d=16, K=3),
        "CPSA": make_dtfn_cpsa(vocab_size=4, d=16, K=3, n_heads=2, window=8),
        "Adaptive": make_dtfn_adaptive(vocab_size=4, d=16, K=3),
    }
    for name, model in variants.items():
        model = model.to(DEVICE)
        x = torch.randint(0, 4, (4, 32), device=DEVICE)
        with torch.no_grad():
            # Use forward_mass to get the final mass distribution
            m0 = model.to_mass(model.emb(x)) + model.cfg.epsilon
            m1 = model.forward_mass(x)
        total0 = m0.sum(dim=(1, 2))
        total1 = m1.sum(dim=(1, 2))
        cons_err = (total1 - total0).abs().max().item()
        min_mass = m1.min().item()
        results[name] = {
            "conservation_error": cons_err,
            "min_mass": min_mass,
            "positive": min_mass > 0,
        }
        print(f"  {name:12s}: conservation_err={cons_err:.2e}, min_mass={min_mass:.2e}, positive={min_mass > 0}")

    # 2D DTFN
    model_2d = make_dtfn_2d(vocab_size=4, d=16, K=3, H=8, W=8).to(DEVICE)
    x = torch.randint(0, 4, (4, 8, 8), device=DEVICE)
    with torch.no_grad():
        m0 = model_2d.to_mass(model_2d.emb(x)) + 1e-6
        m1 = m0
        for layer in model_2d.layers:
            m1 = layer(m1)
    cons_err_2d = (m1.sum() - m0.sum()).abs().max().item()
    min_mass_2d = m1.min().item()
    results["2D-DTFN"] = {
        "conservation_error": cons_err_2d,
        "min_mass": min_mass_2d,
        "positive": min_mass_2d > 0,
    }
    print(f"  {'2D-DTFN':12s}: conservation_err={cons_err_2d:.2e}, min_mass={min_mass_2d:.2e}, positive={min_mass_2d > 0}")

    return results


# ----------------------------------------------------------------------
# Experiment 2: Higher-order accuracy comparison
# ----------------------------------------------------------------------

def exp2_accuracy_order() -> Dict:
    """Compare convergence order of upwind, MUSCL, and WENO5 on smooth advection.

    We evolve a smooth Gaussian pulse with the upwind, MUSCL, and WENO5
    schemes and measure the L1 error as a function of grid resolution L.
    Expected orders: upwind ~ 1, MUSCL ~ 2, WENO5 ~ 5.
    """
    print("\n=== Exp 2: Accuracy order comparison ===")
    torch.manual_seed(42)
    results = {"upwind": [], "muscl": [], "weno5": []}

    def smooth_gaussian(L: int, device) -> torch.Tensor:
        # Use endpoint=False for a proper periodic grid (linspace(0,1,L) puts
        # points at both 0 and 1, which are the same point for periodic BCs).
        x = torch.arange(L, device=device, dtype=torch.float32) / L
        return (torch.exp(-((x - 0.3) ** 2) / 0.01)).unsqueeze(0).unsqueeze(-1)  # (1, L, 1)

    # Use a constant-velocity advection: F(p) = v * p, v = 0.3, with PERIODIC BCs.
    # We implement this as a special flux (not learned) for clean order measurement.
    def advect(p: torch.Tensor, scheme: str, lam: float, v: float = 0.3) -> torch.Tensor:
        B, L, d = p.shape
        # Periodic padding: p_{-1} = p_{L-1}, p_L = p_0.
        if scheme == "upwind":
            # F_{i+1/2} = v * p_i for i = 0..L-1 (periodic: face L-1/2 -> between L-1 and 0 uses p_{L-1})
            # We need F at faces 0..L-1 (length L), then update uses (F_{i-1/2} - F_{i+1/2}) with periodic wrap.
            F_face = v * p  # (B, L, d) — F at face i+1/2 = v*p_i
            # dm[i] = F_{i-1/2} - F_{i+1/2} = F_face[(i-1) mod L] - F_face[i]
            F_left = torch.roll(F_face, shifts=1, dims=1)
            dm = F_left - F_face
            return p + lam * dm
        elif scheme == "muscl":
            # Van Leer limited slopes, periodic
            p_left = torch.roll(p, shifts=1, dims=1)
            p_right = torch.roll(p, shifts=-1, dims=1)
            d_minus = p - p_left
            d_plus = p_right - p
            same_sign = (d_minus * d_plus) > 0
            slopes = torch.where(same_sign, 2.0 * d_minus * d_plus / (d_minus + d_plus + 1e-12), torch.zeros_like(d_minus))
            p_face_right = p + 0.5 * slopes  # face value at i+1/2 reconstructed from cell i
            F_face = v * p_face_right  # (B, L, d)
            F_left = torch.roll(F_face, shifts=1, dims=1)
            dm = F_left - F_face
            return p + lam * dm
        elif scheme == "weno5":
            # WENO5 reconstruction with periodic BCs
            # Build 5-point stencils using roll
            p_m2 = torch.roll(p, shifts=2, dims=1)
            p_m1 = torch.roll(p, shifts=1, dims=1)
            p_0 = p
            p_p1 = torch.roll(p, shifts=-1, dims=1)
            p_p2 = torch.roll(p, shifts=-2, dims=1)
            # WENO-Z weights for face value at i+1/2 (left-biased reconstruction)
            # Three stencils of size 3:
            # S0: {p_{i-2}, p_{i-1}, p_i} -> p_{i+1/2}^{(0)} = (2 p_{i-2} - 7 p_{i-1} + 11 p_i)/6
            # S1: {p_{i-1}, p_i, p_{i+1}}  -> p_{i+1/2}^{(1)} = (-p_{i-1} + 5 p_i + 2 p_{i+1})/6
            # S2: {p_i, p_{i+1}, p_{i+2}}  -> p_{i+1/2}^{(2)} = (2 p_i + 5 p_{i+1} - p_{i+2})/6
            p0 = (2*p_m2 - 7*p_m1 + 11*p_0) / 6.0
            p1 = (-p_m1 + 5*p_0 + 2*p_p1) / 6.0
            p2 = (2*p_0 + 5*p_p1 - p_p2) / 6.0
            # Smoothness indicators
            beta0 = (13.0/12.0)*(p_m2 - 2*p_m1 + p_0)**2 + 0.25*(p_m2 - 4*p_m1 + 3*p_0)**2
            beta1 = (13.0/12.0)*(p_m1 - 2*p_0 + p_p1)**2 + 0.25*(p_m1 - p_p1)**2
            beta2 = (13.0/12.0)*(p_0 - 2*p_p1 + p_p2)**2 + 0.25*(3*p_0 - 4*p_p1 + p_p2)**2
            # WENO-Z (Borges et al. 2008): use tau = |beta0 - beta2|
            tau = (beta0 - beta2).abs()
            eps = 1e-6  # float32-safe (1e-40 flushes to 0 on MPS)
            d0, d1, d2 = 0.1, 0.6, 0.3
            alpha0 = d0 * (1.0 + tau / (beta0 + eps))
            alpha1 = d1 * (1.0 + tau / (beta1 + eps))
            alpha2 = d2 * (1.0 + tau / (beta2 + eps))
            sa = alpha0 + alpha1 + alpha2
            w0 = alpha0 / sa
            w1 = alpha1 / sa
            w2 = alpha2 / sa
            p_face = w0 * p0 + w1 * p1 + w2 * p2  # face value at i+1/2
            F_face = v * p_face
            F_left = torch.roll(F_face, shifts=1, dims=1)
            dm = F_left - F_face
            return p + lam * dm
        else:
            raise ValueError(f"Unknown scheme: {scheme}")

    # Run convergence study with SSP-RK3 time integration so that the spatial
    # convergence order of each scheme is not masked by 1st-order Euler time.
    # SSP-RK3 (Shu-Osher): p1 = p + dt*L(p); p2 = 0.75*p + 0.25*(p1 + dt*L(p1));
    # p_new = (1/3)*p + (2/3)*(p2 + dt*L(p2)).
    def rk3_step(p, scheme, lam, v):
        p1 = advect(p, scheme, lam, v)
        p2 = 0.75 * p + 0.25 * advect(p1, scheme, lam, v)
        return (1.0/3.0) * p + (2.0/3.0) * advect(p2, scheme, lam, v)

    Ls = [32, 64, 128, 256, 512]
    n_steps_factor = 4  # CFL ~ 0.15
    v = 0.3

    for L in Ls:
        T = 0.5  # total advection time
        dt = T / n_steps_factor / L  # CFL ~ 0.15
        n_steps = int(T / dt)
        p0 = smooth_gaussian(L, DEVICE)
        # Reference solution: periodic shift of the Gaussian by v*T to the right.
        # On a periodic grid of length 1 with v*T = 0.15, the Gaussian centred at 0.3
        # moves to 0.45 (mod 1).
        x = torch.arange(L, device=DEVICE, dtype=p0.dtype) / L  # periodic grid (no endpoint)
        shift = v * T
        # Periodic reference: x_ref = (x - shift) mod 1, p_ref = p0(x_ref).
        x_ref = (x - shift) % 1.0
        # Re-evaluate the smooth Gaussian at the shifted positions
        # The smooth_gaussian uses center 0.3 and width 0.1.
        p_ref = torch.exp(-((x_ref - 0.3) ** 2) / 0.01)
        p_ref = p_ref.unsqueeze(0).unsqueeze(-1).to(p0.dtype)

        for scheme in ["upwind", "muscl", "weno5"]:
            p = p0.clone()
            for _ in range(n_steps):
                # Conservative update: p_new = p + (dt/dx) * (F_{i-1/2} - F_{i+1/2})
                # Here lam = dt/dx = dt * L (since dx = 1/L). The velocity v is
                # already encoded in the flux F = v * p.
                # Use SSP-RK3 to avoid 1st-order time-accuracy masking spatial order.
                p = rk3_step(p, scheme, lam=dt * L, v=v)
            err = (p - p_ref).abs().mean().item()
            results[scheme].append({"L": L, "L1_error": err, "n_steps": n_steps})
            print(f"  {scheme:8s} L={L:4d}: L1 error = {err:.6e} ({n_steps} steps)")

    # Compute convergence orders
    orders = {}
    for scheme in ["upwind", "muscl", "weno5"]:
        errs = [r["L1_error"] for r in results[scheme]]
        Ls_used = [r["L"] for r in results[scheme]]
        order_estimates = []
        for i in range(1, len(errs)):
            if errs[i] > 0 and errs[i-1] > 0:
                order = math.log(errs[i-1] / errs[i]) / math.log(Ls_used[i] / Ls_used[i-1])
                order_estimates.append(order)
        orders[scheme] = order_estimates
        print(f"  {scheme:8s} order estimates: {[f'{o:.2f}' for o in order_estimates]}")

    results["orders"] = orders
    return results


# ----------------------------------------------------------------------
# Experiment 3: Dyck-1 with new variants
# ----------------------------------------------------------------------

@dataclass
class DyckSample:
    tokens: List[int]
    label: int

def generate_dyck1(n_samples: int, min_len: int, max_len: int, seed: int = 0) -> List[DyckSample]:
    """Generate Dyck-1 strings (balanced parentheses)."""
    rng = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(n_samples):
        n_pairs = int(torch.randint(min_len, max_len + 1, (1,), generator=rng).item())
        depth = 0
        tokens = []
        # Generate a balanced sequence
        opens = [1] * n_pairs
        closes = [2] * n_pairs
        # Random shuffle of opens and closes, keeping validity
        all_tokens = opens + closes
        # Use a valid Dyck generation: at each step, choose open if depth < remaining closes
        remaining_opens = n_pairs
        remaining_closes = n_pairs
        for _ in range(2 * n_pairs):
            if remaining_opens == 0:
                tokens.append(2); remaining_closes -= 1; depth -= 1
            elif remaining_closes == 0 or depth == 0:
                tokens.append(1); remaining_opens -= 1; depth += 1
            else:
                # Random choice
                if torch.rand(1, generator=rng).item() < 0.5:
                    tokens.append(1); remaining_opens -= 1; depth += 1
                else:
                    tokens.append(2); remaining_closes -= 1; depth -= 1
        samples.append(DyckSample(tokens=tokens, label=0))
    return samples


def collate_dyck(batch: List[DyckSample], pad_id: int = 0):
    """Collate Dyck samples.  Returns (input, target) for next-token prediction."""
    max_len = max(len(s.tokens) for s in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(batch):
        input_ids[i, :len(s.tokens)] = torch.tensor(s.tokens, dtype=torch.long)
    # Target: shifted input
    targets = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, s in enumerate(batch):
        targets[i, :len(s.tokens) - 1] = torch.tensor(s.tokens[1:], dtype=torch.long)
    return input_ids, targets


def train_model(model, train_loader, val_loader, n_epochs=5, lr=1e-3, device=DEVICE):
    """Train a model and return training history."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    history = []
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0; total_correct = 0; total_tokens = 0; n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item(); n_batches += 1
            preds = logits.argmax(-1)
            mask = y != -100
            total_correct += ((preds == y) & mask).sum().item()
            total_tokens += mask.sum().item()
        scheduler.step()
        train_loss = total_loss / n_batches
        train_acc = total_correct / max(total_tokens, 1)

        # Validation
        model.eval()
        val_loss = 0; val_correct = 0; val_tokens = 0; val_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                val_loss += loss.item(); val_batches += 1
                preds = logits.argmax(-1)
                mask = y != -100
                val_correct += ((preds == y) & mask).sum().item()
                val_tokens += mask.sum().item()
        val_loss /= val_batches
        val_acc = val_correct / max(val_tokens, 1)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
        })
        print(f"    Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    return history


def exp3_dyck1_variants() -> Dict:
    """Compare DTFN variants on Dyck-1."""
    print("\n=== Exp 3: Dyck-1 with new variants ===")
    torch.manual_seed(42)
    PAD_ID = 0

    # Generate data
    train_data = generate_dyck1(1000, 5, 20, seed=42)
    val_data = generate_dyck1(200, 5, 20, seed=43)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=lambda b: collate_dyck(b, PAD_ID))
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=lambda b: collate_dyck(b, PAD_ID))

    results = {}
    variants = {
        "DTFN-U (upwind)": make_dtfn_unidirectional(vocab_size=4, d=64, K=3),
        "DTFN-MUSCL": make_dtfn_muscl(vocab_size=4, d=64, K=3),
        "DTFN-WENO5": make_dtfn_weno5(vocab_size=4, d=64, K=3),
        "DTFN-CPSA": make_dtfn_cpsa(vocab_size=4, d=64, K=3, n_heads=4, window=16),
        "DTFN-Adaptive": make_dtfn_adaptive(vocab_size=4, d=64, K=3),
    }
    for name, model in variants.items():
        print(f"  Training {name}...")
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        t0 = time.time()
        history = train_model(model, train_loader, val_loader, n_epochs=5, lr=1e-3)
        train_time = time.time() - t0
        results[name] = {
            "n_params": n_params,
            "train_time_sec": train_time,
            "history": history,
            "final_val_loss": history[-1]["val_loss"],
            "final_val_acc": history[-1]["val_acc"],
        }
        print(f"    -> {name}: val_loss={history[-1]['val_loss']:.4f}, val_acc={history[-1]['val_acc']:.4f}, params={n_params}, time={train_time:.1f}s")

    return results


# ----------------------------------------------------------------------
# Experiment 4: DTFN-SSM correspondence
# ----------------------------------------------------------------------

def exp4_ssm_correspondence() -> Dict:
    """Verify the DTFN-SSM correspondence empirically."""
    print("\n=== Exp 4: DTFN-SSM correspondence ===")
    from innovations.dtfn_ssm_bridge import linearise_flux, LinearisedDTFN

    torch.manual_seed(42)
    results = {}

    # For several random DTFN configurations, compute the linearised
    # Jacobian and compare with the autograd Jacobian.
    for d in [4, 8]:
        for L in [8, 16]:
            for lam in [0.3, 0.5]:
                from dtfn_core import MonotoneFluxNet
                flux_net = MonotoneFluxNet(d=d, hidden=128).to(DEVICE)
                p_star = torch.ones(1, 1, d, device=DEVICE) * 2.0

                # Linearise at uniform equilibrium
                J_left, J_right = linearise_flux(flux_net, p_star, eps=1e-4)
                lin = LinearisedDTFN(J_left, J_right, lam, L).to(DEVICE)

                # Predicted spectral radius (compute on CPU for eig compatibility)
                eig_pred = torch.linalg.eigvals(lin.T.cpu()).abs().max().item()

                # Actual spectral radius via autograd at uniform equilibrium
                from dtfn_core import DTFNLayer, DTFNConfig
                cfg = DTFNConfig(d=d, K=1, scheme="upwind")
                layer = DTFNLayer(cfg).to(DEVICE)
                layer.flux.load_state_dict(flux_net.state_dict())
                with torch.no_grad():
                    layer.cfl_logit.fill_(math.log(lam / max(1 - lam, 1e-3)))

                p_eq = p_star.expand(1, L, d).clone().requires_grad_(True)
                # Compute Jacobian via autograd
                out = layer(p_eq)
                J_actual = torch.zeros(L * d, L * d, device=DEVICE)
                for i in range(L * d):
                    grad = torch.zeros(1, L, d, device=DEVICE)
                    grad.reshape(-1)[i] = 1.0
                    g = torch.autograd.grad(out, p_eq, grad_outputs=grad, retain_graph=True)[0]
                    J_actual[:, i] = g.reshape(-1)
                # eig on CPU (MPS lacks linalg_eig)
                eig_actual = torch.linalg.eigvals(J_actual.cpu()).abs().max().item()
                jac_err = (J_actual - lin.T.to(J_actual.device)).abs().max().item()

                key = f"d{d}_L{L}_lam{lam}"
                results[key] = {
                    "predicted_spectral_radius": eig_pred,
                    "actual_spectral_radius": eig_actual,
                    "jacobian_max_error": jac_err,
                    "stable": eig_pred < 1.0 and eig_actual <= 1.0 + 1e-4,
                }
                print(f"  {key}: pred_rho={eig_pred:.4f}, actual_rho={eig_actual:.4f}, err={jac_err:.2e}, stable={results[key]['stable']}")

    return results


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE} (HAS_NPU={HAS_NPU})")
    if HAS_NPU:
        print(f"NPU available: {torch.npu.device_count()} devices")
        print(f"Device name: {torch.npu.get_device_name(0)}")

    all_results = {}
    all_results["exp1_conservation"] = exp1_conservation()
    all_results["exp2_accuracy_order"] = exp2_accuracy_order()
    all_results["exp3_dyck1_variants"] = exp3_dyck1_variants()
    all_results["exp4_ssm_correspondence"] = exp4_ssm_correspondence()

    # Save results
    out_path = Path(__file__).parent.parent / "results" / "innovations_v3.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
