"""Empirical verification of the L1-contraction / adversarial stability theorem
(Theorem 5 / thm:l1) and the CFL spectral-radius stability (Theorem 6 / thm:cfl).

This experiment:
  1. Trains a DTFN-FULL model on Dyck-1 (small, fast).
  2. Constructs adversarial perturbations of the embedded input at increasing
     epsilon, both random and gradient-based (FGSM-style).
  3. Measures L1, L2, and Linf norms of the output perturbation vs input.
  4. Verifies ||delta_out||_p <= ||delta_in||_p for p in {1, 2, inf} -- the
     1-Lipschitz guarantee of Theorem 5.
  5. Computes the empirical spectral radius of the flux Jacobian and verifies
     the CFL condition lambda < 1/rho(J_F) (Theorem 6).
"""

import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dtfn_core import (
    DTFN, DTFNConfig, make_dtfn_full, make_dtfn_unidirectional,
    make_dtfn_bidirectional,
    FluxNet, MonotoneFluxNet, SourceNet,
)


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -- Dyck-1 data (minimal, inlined) ---------------------------------------

PAD, BOS, EOS = 2, 3, 4
OPEN, CLOSE = 0, 1
VOCAB = 5

def sample_dyck1(min_pairs: int = 4, max_pairs: int = 16, rng: torch.Generator = None) -> List[int]:
    n_pairs = int(torch.randint(min_pairs, max_pairs + 1, (1,), generator=rng).item())
    depth = 0
    seq = [BOS]
    remaining_open = n_pairs
    remaining_close = n_pairs
    while remaining_open > 0 or remaining_close > 0:
        can_open = remaining_open > 0
        can_close = depth > 0 and remaining_close > 0
        if can_open and can_close:
            p_open = 0.5
        elif can_open:
            p_open = 1.0
        else:
            p_open = 0.0
        if float(torch.rand(1, generator=rng).item()) < p_open:
            seq.append(OPEN); depth += 1; remaining_open -= 1
        else:
            seq.append(CLOSE); depth -= 1; remaining_close -= 1
    seq.append(EOS)
    return seq


def make_dataset(n: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    seqs = [sample_dyck1(4, 16, rng) for _ in range(n)]
    maxlen = max(len(s) for s in seqs)
    # Inputs: seqs[:-1], Targets: seqs[1:]
    x = torch.full((n, maxlen - 1), PAD, dtype=torch.long)
    y = torch.full((n, maxlen - 1), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)-1] = torch.tensor(s[:-1], dtype=torch.long)
        y[i, :len(s)-1] = torch.tensor(s[1:], dtype=torch.long)
    return x, y


# -- Adversarial perturbations --------------------------------------------

def random_linf_perturb(emb_input: torch.Tensor, eps: float, generator=None) -> torch.Tensor:
    """Uniform random perturbation in [-eps, eps] (Linfty ball of radius eps)."""
    if generator is not None:
        noise = (torch.rand(emb_input.shape, generator=generator, device=emb_input.device) * 2 - 1) * eps
    else:
        noise = (torch.rand_like(emb_input) * 2 - 1) * eps
    return emb_input + noise


def fgsm_perturb(model: nn.Module, emb_input: torch.Tensor, targets: torch.Tensor,
                 eps: float, criterion) -> torch.Tensor:
    """Fast Gradient Sign Method: perturb input to maximize loss."""
    emb_input = emb_input.clone().detach().requires_grad_(True)
    logits = model.decode_from_mass(emb_input)
    # logits: (B, L, V); targets: (B, L)
    loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    grad, = torch.autograd.grad(loss, emb_input)
    with torch.no_grad():
        perturbed = emb_input + eps * grad.sign()
    return perturbed.detach()


# -- Hook to expose DTFN intermediate mass --------------------------------

def forward_with_mass(model: DTFN, tokens: torch.Tensor) -> torch.Tensor:
    """Return the final mass distribution p^K (input to decoder)."""
    return model.forward_mass(tokens)


# -- Main experiment -------------------------------------------------------

def run_stability_experiment(seed: int = 0, n_train: int = 800, n_test: int = 200,
                              epochs: int = 3, epsilons=(0.01, 0.05, 0.1, 0.2, 0.5),
                              model_variant: str = "DTFN-FULL") -> Dict:
    set_seed(seed)
    device = get_device()
    print(f"Using device: {device}")
    print(f"Model variant: {model_variant}")

    # Build the requested model variant.
    if model_variant == "DTFN-U":
        model = make_dtfn_unidirectional(vocab_size=VOCAB, d=32, K=3)
    elif model_variant == "DTFN-B":
        model = make_dtfn_bidirectional(vocab_size=VOCAB, d=32, K=3)
    elif model_variant == "DTFN-FULL":
        model = make_dtfn_full(vocab_size=VOCAB, d=32, K=3)
    else:
        raise ValueError(f"unknown variant {model_variant}")

    # Train a small model on Dyck-1
    x_train, y_train = make_dataset(n_train, seed)
    x_test, y_test = make_dataset(n_test, seed + 1)

    model.to(device)
    x_train_d = x_train.to(device); y_train_d = y_train.to(device)
    x_test_d = x_test.to(device); y_test_d = y_test.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    batch = 32
    print(f"Training {model_variant} on Dyck-1...")
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch):
            idx = perm[i:i+batch]
            xb, yb = x_train_d[idx], y_train_d[idx]
            logits = model(xb)
            loss = crit(logits.reshape(-1, VOCAB), yb.reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
        # compute test acc
        with torch.no_grad():
            logits = model(x_test_d)
            pred = logits.argmax(-1)
            mask = y_test_d != PAD
            acc = (pred[mask] == y_test_d[mask]).float().mean().item()
        print(f"  epoch {ep+1}: train_loss={loss.item():.4f}  test_acc={acc:.4f}")

    # Get clean mass distributions on test set.
    # We need BOTH the input mass p^0 (right after embedding, before FV stack)
    # and the output mass p^K (after FV stack, before decoder).
    # Theorem 5 is stated on the FV map p^0 -> p^K.
    print("\nComputing mass distributions on clean test inputs...")
    # p^0: the initial mass (after embedding + positivity)
    p0 = (model.to_mass(model.emb(x_test_d)) + model.cfg.epsilon).detach()
    # p^K: run p0 through the FV stack
    if model.cfg.bidirectional:
        mf = model._run_stack(p0, model.layers_fwd)
        mb = model._run_stack(torch.flip(p0, dims=[1]), model.layers_bwd)
        mb = torch.flip(mb, dims=[1])
        alpha = torch.sigmoid(model.alpha_logit).view(1, 1, -1)
        pK_clean = (alpha * mf + (1.0 - alpha) * mb).detach()
    else:
        pK_clean = model._run_stack(p0, model.layers).detach()
    clean_logits = model.decode_from_mass(pK_clean)
    print(f"  p0 shape: {p0.shape}, min={p0.min().item():.4f}, max={p0.max().item():.4f}")
    print(f"  pK shape: {pK_clean.shape}, min={pK_clean.min().item():.4f}, max={pK_clean.max().item():.4f}")

    # Helper: run p0 through the FV stack and return pK
    def fv_forward(p0_in: torch.Tensor) -> torch.Tensor:
        if model.cfg.bidirectional:
            mf = model._run_stack(p0_in, model.layers_fwd)
            mb = model._run_stack(torch.flip(p0_in, dims=[1]), model.layers_bwd)
            mb = torch.flip(mb, dims=[1])
            alpha = torch.sigmoid(model.alpha_logit).view(1, 1, -1)
            return alpha * mf + (1.0 - alpha) * mb
        return model._run_stack(p0_in, model.layers)

    # Apply perturbations at each epsilon and measure output perturbation.
    # Theorem 5 is stated on the FV map p^0 -> p^K, so we perturb p^0 (input
    # mass) and measure the perturbation in p^K (output mass). We also record
    # the perturbation in logit space for downstream accuracy analysis.
    results = {"random": {}, "fgsm": {}, "n_test": n_test,
               "epsilons": list(epsilons), "model_variant": model_variant}
    rng = torch.Generator(device=device).manual_seed(seed)

    def norm_stats(delta: torch.Tensor) -> Tuple[float, float, float]:
        """Return (Linf, L1-per-sample-mean, L2-per-sample-mean)."""
        linf = delta.abs().max().item()
        l1 = delta.abs().sum(dim=(1, 2)).mean().item()
        l2 = delta.pow(2).sum(dim=(1, 2)).mean().sqrt().item()
        return linf, l1, l2

    for eps in epsilons:
        print(f"\nepsilon = {eps}")

        # ---- Random L_inf perturbation of p^0 ----
        noise = (torch.rand_like(p0) * 2 - 1) * eps
        p0_pert = (p0 + noise).clamp(min=1e-6)
        with torch.no_grad():
            pK_pert = fv_forward(p0_pert)
        delta_in = (p0_pert - p0)        # in mass space, p^0
        delta_out = (pK_pert - pK_clean) # in mass space, p^K
        din_linf, din_l1, din_l2 = norm_stats(delta_in)
        dout_linf, dout_l1, dout_l2 = norm_stats(delta_out)

        # Accuracy under perturbation (via decoder)
        with torch.no_grad():
            logits_pert = model.decode_from_mass(pK_pert)
        pred_clean = clean_logits.argmax(-1)
        pred_pert = logits_pert.argmax(-1)
        mask = y_test_d != PAD
        acc_clean = (pred_clean[mask] == y_test_d[mask]).float().mean().item()
        acc_pert = (pred_pert[mask] == y_test_d[mask]).float().mean().item()
        flip_rate = (pred_clean[mask] != pred_pert[mask]).float().mean().item()

        results["random"][f"eps_{eps}"] = {
            "delta_in_linf": din_linf, "delta_out_linf": dout_linf,
            "ratio_linf": dout_linf / max(din_linf, 1e-12),
            "delta_in_l1": din_l1, "delta_out_l1": dout_l1,
            "ratio_l1": dout_l1 / max(din_l1, 1e-12),
            "delta_in_l2": din_l2, "delta_out_l2": dout_l2,
            "ratio_l2": dout_l2 / max(din_l2, 1e-12),
            "acc_clean": acc_clean, "acc_perturbed": acc_pert,
            "pred_flip_rate": flip_rate,
        }
        print(f"  random:  Linf ratio={dout_linf/max(din_linf,1e-12):.4f}  "
              f"L1 ratio={dout_l1/max(din_l1,1e-12):.4f}  "
              f"L2 ratio={dout_l2/max(din_l2,1e-12):.4f}  "
              f"acc_drop={acc_clean-acc_pert:.4f}")

        # ---- FGSM perturbation of p^0 ----
        # Gradient of loss wrt p^0 (input mass), sign-step, then forward.
        fgsm_batch = min(64, n_test)
        p0_b = p0[:fgsm_batch].clone().detach().requires_grad_(True)
        pK_b = fv_forward(p0_b)
        logits_b = model.decode_from_mass(pK_b)
        yb = y_test_d[:fgsm_batch]
        loss = crit(logits_b.reshape(-1, VOCAB), yb.reshape(-1))
        grad, = torch.autograd.grad(loss, p0_b)
        with torch.no_grad():
            p0_pert_b = (p0_b + eps * grad.sign()).clamp(min=1e-6)
            pK_pert_b = fv_forward(p0_pert_b)
            logits_pert_b = model.decode_from_mass(pK_pert_b)
        din = (p0_pert_b - p0_b)
        dout = (pK_pert_b - pK_b)
        din_linf, din_l1, din_l2 = norm_stats(din)
        dout_linf, dout_l1, dout_l2 = norm_stats(dout)
        pred_c = logits_b.argmax(-1)
        pred_p = logits_pert_b.argmax(-1)
        msk = yb != PAD
        acc_c = (pred_c[msk] == yb[msk]).float().mean().item()
        acc_p = (pred_p[msk] == yb[msk]).float().mean().item()
        flip = (pred_c[msk] != pred_p[msk]).float().mean().item()

        results["fgsm"][f"eps_{eps}"] = {
            "delta_in_linf": din_linf, "delta_out_linf": dout_linf,
            "ratio_linf": dout_linf / max(din_linf, 1e-12),
            "delta_in_l1": din_l1, "delta_out_l1": dout_l1,
            "ratio_l1": dout_l1 / max(din_l1, 1e-12),
            "delta_in_l2": din_l2, "delta_out_l2": dout_l2,
            "ratio_l2": dout_l2 / max(din_l2, 1e-12),
            "acc_clean": acc_c, "acc_perturbed": acc_p,
            "pred_flip_rate": flip,
        }
        print(f"  fgsm:    Linf ratio={dout_linf/max(din_linf,1e-12):.4f}  "
              f"L1 ratio={dout_l1/max(din_l1,1e-12):.4f}  "
              f"L2 ratio={dout_l2/max(din_l2,1e-12):.4f}  "
              f"acc_drop={acc_c-acc_p:.4f}")

    # Spectral radius of flux Jacobian (Theorem 6 verification)
    print("\nComputing spectral radius of flux Jacobian (Theorem 6)...")
    model.eval()
    # Take a single batch
    xb = x_test_d[:8]
    mass = forward_with_mass(model, xb).detach()  # (8, L, d)
    B, L, d = mass.shape
    # The flux Jacobian J_F maps (d,) -> (d,) at each interface.
    # We estimate it via finite differences of the MonotoneFluxNet.
    # For bidirectional models, use layers_fwd; for unidirectional, use layers.
    if hasattr(model, 'layers_fwd'):
        first_layer = model.layers_fwd[0]
    else:
        first_layer = model.layers[0]
    flux = first_layer.flux
    # FluxNet forward: takes (left, right) -> scalar per channel (output shape (..., d))
    # Jacobian wrt left input has shape (d, d).
    # Estimate via torch.autograd.functional.jacobian on a single interface.
    left = mass[0, :d].clone().requires_grad_(True)  # (d,)
    right = mass[0, :d].clone().requires_grad_(True)  # (d,)
    F_out = flux(left, right)  # (d,)
    # J_F wrt left: (d, d)
    JF_left = torch.autograd.functional.jacobian(lambda l: flux(l, right), left)
    JF_right = torch.autograd.functional.jacobian(lambda r: flux(left, r), right)
    # Spectral radius = max abs eigenvalue
    eigs_l = torch.linalg.eigvals(JF_left.cpu())
    eigs_r = torch.linalg.eigvals(JF_right.cpu())
    rho_l = eigs_l.abs().max().item()
    rho_r = eigs_r.abs().max().item()
    rho = max(rho_l, rho_r)
    # CFL coefficient
    lam = torch.sigmoid(first_layer.cfl_logit).item()
    print(f"  spectral radius rho(J_F) = {rho:.4f}")
    print(f"  lambda = {lam:.4f}")
    print(f"  1/rho = {1/max(rho,1e-12):.4f}")
    print(f"  CFL satisfied: lambda < 1/rho? {lam < 1/max(rho, 1e-12)}")
    print(f"  lambda < 1? {lam < 1.0} (certified by sigmoid)")

    results["spectral_radius"] = {
        "rho_JF_left": rho_l,
        "rho_JF_right": rho_r,
        "rho_JF_max": rho,
        "lambda": lam,
        "inv_rho": 1.0 / max(rho, 1e-12),
        "cfl_satisfied": bool(lam < 1.0 / max(rho, 1e-12)),
        "sigmoid_certified": bool(lam < 1.0),
    }

    # Mass conservation under perturbation (Theorem 1 check).
    # We perturb p^0 and run through the FV stack; total mass should be
    # preserved by the flux-telescoping argument (Theorem 1).
    print("\nMass conservation under perturbation (Theorem 1)...")
    total_clean = p0.sum(dim=(1, 2))  # (n_test,)
    noise = (torch.rand_like(p0) * 2 - 1) * 0.1
    p0_pert = (p0 + noise).clamp(min=1e-6)
    with torch.no_grad():
        pK_pert_cons = fv_forward(p0_pert)
    total_after = pK_pert_cons.sum(dim=(1, 2))
    cons_err = (total_after - total_clean).abs().max().item()
    rel_err = ((total_after - total_clean).abs() / total_clean.abs().clamp(min=1e-8)).max().item()
    print(f"  max abs conservation error: {cons_err:.2e}")
    print(f"  max rel conservation error: {rel_err:.2e}")
    results["conservation_under_perturbation"] = {
        "max_abs_err": cons_err,
        "max_rel_err": rel_err,
    }

    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--variant", type=str, default="DTFN-FULL",
                   choices=["DTFN-U", "DTFN-B", "DTFN-FULL"])
    p.add_argument("--out", type=str, default="../results/stability.json")
    args = p.parse_args()

    t0 = time.time()
    results = run_stability_experiment(seed=args.seed, epochs=args.epochs,
                                        model_variant=args.variant)
    results["elapsed_sec"] = time.time() - t0
    results["model_variant"] = args.variant
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")
    print(f"Elapsed: {results['elapsed_sec']:.1f}s")
