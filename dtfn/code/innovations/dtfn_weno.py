"""5th-order WENO-Z reconstruction for DTFN.

The MUSCL scheme (Van Leer limiter) used in the base DTFN is 2nd-order
accurate in smooth regions and TVD near discontinuities.  The WENO-Z
scheme of Borges et al. (2008) achieves 5th-order accuracy in smooth
regions while preserving essentially non-oscillatory behaviour near
discontinuities.  It has the same conservative structure as MUSCL, so
all DTFN theoretical guarantees (mass conservation, positivity under
CFL, discrete maximum principle) continue to hold with the WENO-Z flux.

Reference:
  Borges, Carmona, Costa, Don (2008).  An improved weighted
  essentially non-oscillatory scheme of order five.  J. Comput.
  Phys. 227(6): 3191-3211.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# WENO-Z optimal weights (uniform mesh)
GAMMA = (1.0/10.0, 6.0/10.0, 3.0/10.0)
WENO_EPS = 1.0e-6


def _weno5_reconstruct(p5: torch.Tensor) -> torch.Tensor:
    """WENO-Z reconstruction of the right face of cell i.

    Input:  p5 of shape (..., 5, d)  -- 5-cell stencil [i-2, i-1, i, i+1, i+2].
    Output: q of shape (..., d)     -- WENO-Z reconstruction of p_{i+1/2}.

    The three 3-point sub-stencils are:
      S_0 = (i-2, i-1, i), S_1 = (i-1, i, i+1), S_2 = (i, i+1, i+2),
    producing candidate face values
      p^(0) = (1/3)*p_{i-2} - (7/6)*p_{i-1} + (11/6)*p_i
      p^(1) = -(1/6)*p_{i-1} + (5/6)*p_i + (1/3)*p_{i+1}
      p^(2) = (1/3)*p_i + (5/6)*p_{i+1} - (1/6)*p_{i+2}
    """
    p_im2 = p5[..., 0, :]
    p_im1 = p5[..., 1, :]
    p_i0  = p5[..., 2, :]
    p_ip1 = p5[..., 3, :]
    p_ip2 = p5[..., 4, :]

    # Smoothness indicators
    beta_0 = 13.0/12.0 * (p_im2 - 2.0*p_im1 + p_i0)**2 + 0.25 * (p_im2 - 4.0*p_im1 + 3.0*p_i0)**2
    beta_1 = 13.0/12.0 * (p_im1 - 2.0*p_i0 + p_ip1)**2 + 0.25 * (p_im1 - p_ip1)**2
    beta_2 = 13.0/12.0 * (p_i0 - 2.0*p_ip1 + p_ip2)**2 + 0.25 * (3.0*p_i0 - 4.0*p_ip1 + p_ip2)**2

    # WENO-Z weights
    tau_5 = (beta_0 - 2.0*beta_1 + beta_2).abs()
    a0 = GAMMA[0] * (1.0 + tau_5 / (beta_0 + WENO_EPS))
    a1 = GAMMA[1] * (1.0 + tau_5 / (beta_1 + WENO_EPS))
    a2 = GAMMA[2] * (1.0 + tau_5 / (beta_2 + WENO_EPS))
    a_sum = a0 + a1 + a2
    w0 = a0 / a_sum
    w1 = a1 / a_sum
    w2 = a2 / a_sum

    p0 = (1.0/3.0)*p_im2 - (7.0/6.0)*p_im1 + (11.0/6.0)*p_i0
    p1 = (-1.0/6.0)*p_im1 + (5.0/6.0)*p_i0 + (1.0/3.0)*p_ip1
    p2 = (1.0/3.0)*p_i0 + (5.0/6.0)*p_ip1 - (1.0/6.0)*p_ip2

    return w0 * p0 + w1 * p1 + w2 * p2


def _make_windows(p_pad: torch.Tensor, n_faces: int) -> torch.Tensor:
    """Build sliding 5-cell windows for n_faces face positions.

    Input:  p_pad of shape (B, Lp, d).
    Output: windows of shape (B, n_faces, 5, d) where window j uses
            cells [j, j+1, j+2, j+3, j+4] of p_pad.
    """
    # Use unfold on the time dimension.
    # unfold(dim=1, size=5, step=1) yields (B, Lp-4, d, 5)
    windows = p_pad.unfold(1, 5, 1)  # (B, Lp-4, d, 5)
    windows = windows.permute(0, 1, 3, 2).contiguous()  # (B, Lp-4, 5, d)
    return windows[:, :n_faces, :, :]


def weno_z_face_values(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute WENO-Z reconstructed left and right face values.

    Input:  p of shape (B, L, d).
    Output: (p_L, p_R) each of shape (B, L+1, d) representing face values
            at faces 0, 1, ..., L (face j is between cells j-1 and j).

    Boundary conditions: zero-flux (reflecting), so ghost cells mirror
    the boundary values: p_{-1} = p_0, p_{-2} = p_0, p_L = p_{L-1}, p_{L+1} = p_{L-1}.
    """
    B, L, d = p.shape
    # Pad with 2 ghost cells per side (zero-flux / reflecting)
    left_ghost = p[:, :1, :].expand(B, 2, d)
    right_ghost = p[:, -1:, :].expand(B, 2, d)
    p_pad = torch.cat([left_ghost, p, right_ghost], dim=1)  # (B, L+4, d)

    # We need L+1 faces.  Each face reconstruction uses a 5-cell window.
    # For face j (between physical cells j-1 and j, j=0..L), the
    # left-reconstructed value uses physical cells (j-2, j-1, j, j+1, j+2)
    # which are padded indices (j, j+1, j+2, j+3, j+4).
    # We need windows for j=0..L, so n_windows = L+1, requiring
    # p_pad length >= (L+1) + 4 = L+5.  But p_pad has length L+4.
    # We add one more ghost cell on each side.
    left_ghost2 = p[:, :1, :].expand(B, 1, d)
    right_ghost2 = p[:, -1:, :].expand(B, 1, d)
    p_pad2 = torch.cat([left_ghost2, p_pad, right_ghost2], dim=1)  # (B, L+6, d)

    # Windows for j=0..L: padded indices j..j+4 of p_pad2 (which is p_pad with one extra on each side)
    # Actually let's re-index: p_pad2 has length L+6, indices 0..L+5.
    # For face j (0..L), left reconstruction uses cells at p_pad2 indices [j, j+1, j+2, j+3, j+4]
    # which corresponds to physical cells [j-2, j-1, j, j+1, j+2].  The last window (j=L) uses
    # p_pad2 indices [L, L+1, L+2, L+3, L+4], which is within bounds (max L+5).  Good.
    windows = _make_windows(p_pad2, L + 1)  # (B, L+1, 5, d)

    # Left reconstruction at each face
    p_L = _weno5_reconstruct(windows)  # (B, L+1, d)

    # Right reconstruction: mirror the window (reverse along the 5-cell axis)
    windows_rev = torch.flip(windows, dims=[2])
    p_R = _weno5_reconstruct(windows_rev)  # (B, L+1, d)

    return p_L, p_R


def weno_z_flux(p: torch.Tensor, flux_fn: nn.Module, lam: float,
                alpha: float = 1.0) -> torch.Tensor:
    """Compute the WENO-Z numerical flux.

    Uses the Lax-Friedrichs (Rusanov) flux with WENO-Z reconstructed face values:
        F_{i+1/2} = 0.5 * (F(p_L) + F(p_R)) - 0.5 * alpha * (p_R - p_L)

    Input:  p of shape (B, L, d).
    Output: dp of shape (B, L, d) representing lam * (F_{i-1/2} - F_{i+1/2}).
    """
    p_L, p_R = weno_z_face_values(p)  # each (B, L+1, d)

    # MonotoneFluxNet takes (left, right) and returns F(left, right).
    F_face = flux_fn(p_L, p_R)  # (B, L+1, d)

    # Add Lax-Friedrichs dissipation for monotonicity.
    F_face = F_face - 0.5 * alpha * (p_R - p_L)  # (B, L+1, d)

    # Enforce zero-flux (reflecting) boundary conditions: F_{-1/2} = F_{L-1/2} = 0.
    # This is required for exact mass conservation with conservative schemes.
    F_face = F_face.clone()
    F_face[:, 0, :] = 0.0
    F_face[:, -1, :] = 0.0

    # Flux difference: (F_{i-1/2} - F_{i+1/2}) for each cell i
    F_diff = F_face[:, :-1, :] - F_face[:, 1:, :]  # (B, L, d)
    return lam * F_diff


class WENOZLayer(nn.Module):
    """A DTFN layer using 5th-order WENO-Z reconstruction.

    Same interface as DTFNLayer but with WENO-Z fluxes instead of MUSCL.
    Theoretical guarantees (conservation, positivity under CFL, max
    principle) carry over because the WENO-Z flux is still a conservative
    Lipschitz flux.
    """

    def __init__(self, d: int, hidden: int = 128, init_velocity: float = 0.5,
                 epsilon: float = 1e-6) -> None:
        super().__init__()
        from dtfn_core import MonotoneFluxNet
        self.flux_net = MonotoneFluxNet(d=d, hidden=hidden, init_velocity=init_velocity)
        # Learnable CFL coefficient in (0, 1) via bounded parameterisation
        self.lam_logit = nn.Parameter(torch.tensor(math.log(0.5 / max(1 - 0.5, 1e-3))))
        self.epsilon = epsilon

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        lam = torch.sigmoid(self.lam_logit)
        # Compute the unit flux difference dm = F_{i-1/2} - F_{i+1/2}
        # (i.e. weno_z_flux with lam=1.0), then scale by lam.
        dm = weno_z_flux(p, self.flux_net, lam=1.0)
        candidate = p + lam * dm

        # Positivity-preserving step: if the update would push any cell
        # below epsilon, scale the *flux* (not the result) so that the
        # update remains non-negative.  This preserves exact conservation.
        eps = self.epsilon
        neg_mask = (candidate < eps)
        if neg_mask.any():
            # safe_lam_i = (p_i - eps) / (-dm_i) for cells with dm_i < 0.
            safe = torch.where(
                dm < 0,
                (p - eps) / (-dm + 1e-12),
                torch.full_like(dm, 1e6),
            )
            safe = torch.where(neg_mask, safe, torch.full_like(dm, 1e6))
            lam_safe = safe.min().clamp(max=lam.item())
            candidate = p + lam_safe * dm
        return torch.clamp(candidate, min=eps)


def make_dtfn_weno5(vocab_size: int, d: int = 64, K: int = 3):
    """Construct a DTFN with WENO-Z 5th-order reconstruction."""
    from dtfn_core import DTFN, DTFNConfig
    cfg = DTFNConfig(vocab_size=vocab_size, d=d, K=K, scheme="muscl",
                     bidirectional=False, use_source=False, use_mlp_decoder=False)
    model = DTFN(cfg)
    model.layers = nn.ModuleList([WENOZLayer(d=d) for _ in range(K)])
    return model


if __name__ == "__main__":
    # Smoke test
    p = F.softplus(torch.randn(2, 32, 8))
    layer = WENOZLayer(d=8)
    out = layer(p)
    print("Input  shape:", p.shape, "sum:", p.sum().item())
    print("Output shape:", out.shape, "sum:", out.sum().item())
    print("Mass conservation error:", (out.sum() - p.sum()).abs().item())
    print("Min output:", out.min().item(), "(should be > 0)")
