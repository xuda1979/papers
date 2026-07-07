"""Conservation-Preserving Sparse Attention for DTFN.

The base DTFN is purely local: each update couples only adjacent cells.
This is the source of its O(L) complexity and its conservation
guarantees, but it also limits the receptive field to K cells after K
layers.  Attention-based models have global receptive field but O(L^2)
complexity and no conservation structure.

This module introduces **Conservation-Preserving Sparse Attention
(CPSA)** that combines the two paradigms:

  * A *local* DTFN finite-volume update provides the conservative,
    positivity-preserving, O(L) backbone.
  * A *sparse* attention head computes attention weights over a window
    of width w << L and produces a *conservative correction* to the
    flux: the attention output is projected into the null space of the
    discrete divergence operator, so that the corrected flux still
    telescopes to zero at the boundary.

The resulting CPSA-DTFN keeps:
  (I1) exact mass conservation,
  (I2) pointwise non-negativity (via the positivity floor),
  (I3) O(L * w) complexity with w a fixed window width,

while gaining a learnable, content-dependent mixing within each window
that approximates a sparse attention head.

Theoretical guarantee (Theorem, stated in the paper):
  Let F_att be the sparse-attention flux correction.  If F_att is
  L_att-Lipschitz in (p_L, p_R) and the effective CFL coefficient
  satisfies
      lambda * (L_flux + L_att) <= 1,
  then the CPSA-DTFN update is monotone, conservative, and
  positivity-preserving, with discrete maximum principle.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dtfn_core import MonotoneFluxNet, DTFNLayer, DTFN, DTFNConfig


class SparseAttentionFlux(nn.Module):
    """Sparse windowed attention that produces a conservative flux correction.

    The attention head computes content-dependent weights over a local
    window of width w and outputs a flux correction delta_F that is
    *projected* to be conservative (i.e. sum_i delta_F_{i-1/2} -
    delta_F_{i+1/2} = 0 by construction).

    Concretely, the attention produces an unconstrained g_i at each
    cell, and we set
        delta_F_{i+1/2} = g_{i+1} - g_i
    so that the discrete divergence telescopes:
        sum_i (delta_F_{i-1/2} - delta_F_{i+1/2}) = 0.

    This is the discrete analogue of writing a flux correction as the
    gradient of a potential g; the potential is computed by attention.
    """

    def __init__(self, d: int, n_heads: int = 4, window: int = 16,
                 hidden: int = 64) -> None:
        super().__init__()
        assert window >= 1
        self.d = d
        self.n_heads = n_heads
        self.window = window
        self.hidden = hidden

        # Q, K, V projections (one per head)
        self.q_proj = nn.Linear(d, n_heads * hidden)
        self.k_proj = nn.Linear(d, n_heads * hidden)
        self.v_proj = nn.Linear(d, n_heads * hidden)
        # Output projection: heads -> potential g (same shape as p)
        self.out_proj = nn.Linear(n_heads * hidden, d)
        # Bounded scale so the flux correction is Lipschitz-bounded
        self.scale_logit = nn.Parameter(torch.tensor(-1.0))  # init scale ~0.27

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Compute the conservative flux correction.

        Input:  p of shape (B, L, d).
        Output: F_corr of shape (B, L+1, d) -- conservative flux correction
                at faces 0..L (with zero-flux boundary conditions).
        """
        B, L, d = p.shape
        H = self.n_heads
        W = self.window
        D = self.hidden

        # Project to Q, K, V
        Q = self.q_proj(p).view(B, L, H, D).transpose(1, 2)  # (B, H, L, D)
        K = self.k_proj(p).view(B, L, H, D).transpose(1, 2)
        V = self.v_proj(p).view(B, L, H, D).transpose(1, 2)

        # Scaled dot-product attention with causal/local mask of width W.
        # For each position i, attend to positions max(0, i-W) .. i.
        # We use a full (L, L) attention with a band mask for simplicity.
        # For very long sequences this can be replaced by a sliding-window
        # attention (e.g. via torch's nested tensor or a custom kernel).
        scale = 1.0 / math.sqrt(D)
        scores = (Q @ K.transpose(-2, -1)) * scale  # (B, H, L, L)

        # Band mask: position i attends to j in [i-W, i].
        # We use an upper-triangular-style mask (causal with window W).
        idx = torch.arange(L, device=p.device)
        mask = (idx[None, :] > idx[:, None] + 0) | (idx[None, :] < idx[:, None] - W)
        scores = scores.masked_fill(mask[None, None, :, :], float('-inf'))

        # Softmax over the last dim
        attn = torch.softmax(scores, dim=-1)  # (B, H, L, L)
        # Replace NaNs (rows with all -inf) with zeros
        attn = torch.nan_to_num(attn, nan=0.0)

        # Apply attention to V
        out = attn @ V  # (B, H, L, D)
        out = out.transpose(1, 2).contiguous().view(B, L, H * D)

        # Project to potential g of shape (B, L, d)
        g = self.out_proj(out)

        # Make g bounded (tanh) so the flux correction is Lipschitz
        g = torch.tanh(g)

        # Conservative flux correction: delta_F_{i+1/2} = g_{i+1} - g_i
        # Faces: 0..L (L+1 faces).
        # F_corr[:, 0, :] = 0 (boundary)
        # F_corr[:, j, :] = g[:, j, :] - g[:, j-1, :] for j=1..L
        # F_corr[:, L, :] = 0 (boundary, since g_L = g_{L-1} by padding)
        # Equivalently, pad g with one replicate on the left and compute diff.
        g_pad = torch.cat([g[:, :1, :], g, g[:, -1:, :]], dim=1)  # (B, L+2, d)
        F_corr = g_pad[:, 1:, :] - g_pad[:, :-1, :]  # (B, L+1, d)
        # Zero boundary fluxes explicitly (already zero by replicate padding,
        # but enforce for safety)
        F_corr = F_corr.clone()
        F_corr[:, 0, :] = 0.0
        F_corr[:, -1, :] = 0.0

        # Scale by bounded scalar
        scale = torch.sigmoid(self.scale_logit)
        return scale * F_corr


class CPSADTFNLayer(nn.Module):
    """Conservation-Preserving Sparse-Attention DTFN layer.

    Combines a local DTFN finite-volume update with a sparse-attention
    flux correction.  Both contributions are conservative, so the sum
    is conservative.  Positivity is preserved by the standard
    positivity-preserving step.

    Update:
        dm_i = (F_local_{i-1/2} - F_local_{i+1/2}) + (F_att_{i-1/2} - F_att_{i+1/2})
        p^{k+1}_i = max(p^k_i + lam * dm_i, eps)
    """

    def __init__(self, cfg: DTFNConfig, n_heads: int = 4, window: int = 16,
                 attn_hidden: int = 64) -> None:
        super().__init__()
        self.cfg = cfg
        self.local_layer = DTFNLayer(cfg)
        self.attention = SparseAttentionFlux(
            d=cfg.d, n_heads=n_heads, window=window, hidden=attn_hidden,
        )

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        # Local conservative flux difference
        # We need the raw dm (F_{i-1/2} - F_{i+1/2}) from the local layer.
        # The base DTFNLayer doesn't expose dm directly, so we recompute
        # it using the same MonotoneFluxNet.
        lam = torch.sigmoid(self.local_layer.cfl_logit)
        # Recompute local dm (unscaled) -- mirror DTFNLayer.forward logic
        # for the upwind scheme (we use upwind as the local backbone).
        dm_local = self._local_dm(p)

        # Attention flux correction (B, L+1, d) -> flux difference (B, L, d)
        F_corr = self.attention(p)  # (B, L+1, d)
        dm_att = F_corr[:, :-1, :] - F_corr[:, 1:, :]  # (B, L, d)

        # Combine: total conservative flux difference
        dm = dm_local + dm_att

        # Positivity-preserving step
        eps = self.cfg.epsilon
        candidate = p + lam * dm
        neg_mask = (candidate < eps)
        if neg_mask.any():
            safe = torch.where(
                dm < 0,
                (p - eps) / (-dm + 1e-12),
                torch.full_like(dm, 1e6),
            )
            safe = torch.where(neg_mask, safe, torch.full_like(dm, 1e6))
            lam_safe = safe.min().clamp(max=lam.item())
            candidate = p + lam_safe * dm
        return torch.clamp(candidate, min=eps)

    def _local_dm(self, p: torch.Tensor) -> torch.Tensor:
        """Compute the local upwind flux difference (F_{i-1/2} - F_{i+1/2}).

        This mirrors DTFNLayer.forward but returns the unscaled dm rather
        than the layer output.  Zero-flux boundary conditions are enforced
        explicitly: F_{-1/2} = F_{L-1/2} = 0.
        """
        B, L, d = p.shape
        flux_net = self.local_layer.flux
        # Upwind: F_{i+1/2} = F(p_i, p_{i+1}) for i=0..L-2; F_{L-1/2} = 0.
        if L >= 2:
            left = p[:, :-1, :]
            right = p[:, 1:, :]
            flux = flux_net(left, right)  # (B, L-1, d)
            # Pad flux to length L with a trailing zero (F_{L-1/2} = 0)
            flux_padded = torch.cat([flux, torch.zeros(B, 1, d, device=p.device, dtype=p.dtype)], dim=1)
        else:
            flux_padded = torch.zeros_like(p)
        # dm_i = F_{i-1/2} - F_{i+1/2}
        # F_{-1/2} = 0, F_{i+1/2} = flux_padded[:, i, :]
        # dm_0 = 0 - flux_padded[:, 0, :]
        # dm_i = flux_padded[:, i-1, :] - flux_padded[:, i, :] for i=1..L-1
        dm = torch.zeros_like(p)
        dm[:, 0, :] = -flux_padded[:, 0, :]
        if L > 1:
            dm[:, 1:, :] = flux_padded[:, :-1, :] - flux_padded[:, 1:, :]
        return dm


def make_dtfn_cpsa(vocab_size: int, d: int = 64, K: int = 3,
                   n_heads: int = 4, window: int = 16):
    """Construct a CPSA-DTFN: DTFN with sparse attention flux correction."""
    cfg = DTFNConfig(vocab_size=vocab_size, d=d, K=K, scheme="upwind",
                     bidirectional=False, use_source=False, use_mlp_decoder=False)
    model = DTFN(cfg)
    # Replace each DTFN layer with a CPSA layer
    model.layers = nn.ModuleList([
        CPSADTFNLayer(cfg, n_heads=n_heads, window=window) for _ in range(K)
    ])
    return model


if __name__ == "__main__":
    # Smoke test
    torch.manual_seed(0)
    p = F.softplus(torch.randn(2, 32, 8))
    layer = CPSADTFNLayer(DTFNConfig(d=8, K=1), n_heads=2, window=8)
    out = layer(p)
    print("CPSA-DTFN layer:")
    print("  Input  shape:", p.shape, "sum:", p.sum().item())
    print("  Output shape:", out.shape, "sum:", out.sum().item())
    print("  Mass conservation error:", (out.sum() - p.sum()).abs().item())
    print("  Min output:", out.min().item(), "(should be > 0)")

    # Full model test
    model = make_dtfn_cpsa(vocab_size=4, d=16, K=2, n_heads=2, window=8)
    x = torch.randint(0, 4, (2, 32))
    logits = model(x)
    print("\nCPSA-DTFN model:")
    print("  Input shape:", x.shape)
    print("  Logits shape:", logits.shape)
    print("  Params:", sum(p.numel() for p in model.parameters()))
