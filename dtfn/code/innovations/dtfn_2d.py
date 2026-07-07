"""2D DTFN: dimensional-splitting extension to 2D conservation laws.

The base DTFN is 1D (sequence axis only).  For images, grid-structured
data, or 2D token layouts (e.g. document grids), we extend to 2D via
dimensional splitting (Strang splitting):

  p^{k+1} = A_x(dt/2) * A_y(dt) * A_x(dt/2) * p^k

where A_x and A_y are 1D DTFN updates along the x and y axes.  This
preserves all 2D conservation laws:
  - Total mass: sum_{i,j} p_{i,j} is invariant
  - Positivity: p_{i,j} >= epsilon
  - Discrete maximum principle (under CFL)

The 2D extension is natural for:
  - Document grids (text laid out in 2D)
  - Image-like token arrangements
  - Hierarchical token structures (paragraph -> sentence -> word)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dtfn_core import MonotoneFluxNet, DTFNConfig


class DTFNLayer2D(nn.Module):
    """2D DTFN layer via Strang splitting.

    Input:  p of shape (B, H, W, d) where H is the height (rows) and
            W is the width (columns).
    Output: (B, H, W, d) with 2D conservation guaranteed.
    """

    def __init__(self, cfg: DTFNConfig) -> None:
        super().__init__()
        self.cfg = cfg
        # Separate flux nets for x and y directions (different anisotropy)
        self.flux_x = MonotoneFluxNet(cfg.d, cfg.flux_hidden)
        self.flux_y = MonotoneFluxNet(cfg.d, cfg.flux_hidden)
        self.cfl_logit = nn.Parameter(torch.tensor(cfg.cfl_init))

    def _axis_update(self, p: torch.Tensor, flux_net: MonotoneFluxNet,
                     lam: float) -> torch.Tensor:
        """Apply a 1D DTFN update along the last axis of p.

        p has shape (..., L, d).  We treat the last axis as the spatial
        axis and apply the standard upwind flux with zero-flux BC.
        """
        # Compute flux at faces: F_{i+1/2} = flux_net(p_i, p_{i+1})
        left = p[..., :-1, :]
        right = p[..., 1:, :]
        flux = flux_net(left, right)  # (..., L-1, d)
        # Pad flux to length L with a trailing zero (F_{L-1/2} = 0)
        pad_shape = list(flux.shape)
        pad_shape[-2] = 1
        flux_padded = torch.cat([flux, torch.zeros(*pad_shape, device=p.device, dtype=p.dtype)], dim=-2)
        # dm_i = F_{i-1/2} - F_{i+1/2}
        dm = torch.zeros_like(p)
        dm[..., 0, :] = -flux_padded[..., 0, :]
        if p.shape[-2] > 1:
            dm[..., 1:, :] = flux_padded[..., :-1, :] - flux_padded[..., 1:, :]
        # Positivity-preserving step: shrink lambda so that no cell goes below eps.
        # Compute safe lambda per-batch (global min within each batch element)
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
            # Reduce over all axes except batch (axis 0)
            safe = safe.view(p.shape[0], -1).min(dim=1, keepdim=True).values
            safe = safe.clamp(max=lam)
            # Reshape back for broadcasting
            safe = safe.view(-1, *([1] * (p.dim() - 1)))
            candidate = p + safe * dm
        return torch.clamp(candidate, min=eps)

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        """Apply 2D DTFN update via Strang splitting.

        p: (B, H, W, d).
        """
        lam = torch.sigmoid(self.cfl_logit).item()
        # Strang splitting: A_x(dt/2) * A_y(dt) * A_x(dt/2)
        # x-axis: update along W (last axis)
        p = self._axis_update(p, self.flux_x, lam / 2.0)
        # y-axis: transpose H and W, update, transpose back
        p_t = p.transpose(-2, -3).contiguous()  # (B, W, H, d)
        p_t = self._axis_update(p_t, self.flux_y, lam)
        p = p_t.transpose(-2, -3).contiguous()  # (B, H, W, d)
        # x-axis: update along W again
        p = self._axis_update(p, self.flux_x, lam / 2.0)
        return p


class DTFN2D(nn.Module):
    """Full 2D DTFN model for grid-structured inputs.

    Pipeline:
      input (B, H, W) -> embedding (B, H, W, d) -> positive mass
            -> K stacked 2D DTFN layers
            -> decoder -> logits (B, H, W, vocab_size)
    """

    def __init__(self, vocab_size: int, d: int = 64, K: int = 3,
                 H: int = 16, W: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d = d
        self.K = K
        self.H = H
        self.W = W
        self.emb = nn.Embedding(vocab_size, d)
        self.to_mass = nn.Sequential(nn.Linear(d, d), nn.Softplus())
        cfg = DTFNConfig(d=d, K=K, scheme="upwind")
        self.layers = nn.ModuleList([DTFNLayer2D(cfg) for _ in range(K)])
        self.dec = nn.Linear(d, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, W) integer tokens.  Returns (B, H, W, vocab_size)."""
        m0 = self.to_mass(self.emb(x)) + self.cfg_eps()
        m = m0
        for layer in self.layers:
            m = layer(m)
        return self.dec(m)

    def cfg_eps(self) -> torch.Tensor:
        return 1e-6


def make_dtfn_2d(vocab_size: int, d: int = 64, K: int = 3,
                 H: int = 16, W: int = 16):
    return DTFN2D(vocab_size=vocab_size, d=d, K=K, H=H, W=W)


if __name__ == "__main__":
    torch.manual_seed(0)
    p = F.softplus(torch.randn(2, 8, 16, 4))
    layer = DTFNLayer2D(DTFNConfig(d=4, K=1))
    out = layer(p)
    print("2D DTFN layer:")
    print("  Input  shape:", p.shape, "sum:", p.sum().item())
    print("  Output shape:", out.shape, "sum:", out.sum().item())
    print("  Mass conservation error:", (out.sum() - p.sum()).abs().item())
    print("  Min output:", out.min().item(), "(should be > 0)")

    model = make_dtfn_2d(vocab_size=4, d=8, K=2, H=4, W=4)
    x = torch.randint(0, 4, (2, 4, 4))
    logits = model(x)
    print("\n2D DTFN model:")
    print("  Input shape:", x.shape)
    print("  Logits shape:", logits.shape)
    print("  Params:", sum(p.numel() for p in model.parameters()))
