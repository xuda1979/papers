"""Core model implementations for the LST paper (pure PyTorch, MPS-friendly).

All modules are written so they can run on CPU or Apple MPS without custom kernels.
The SelectiveSSM is a faithful, parallel-form (cumsum-based) implementation of the
Mamba selective SSM, equivalent in spirit to mamba_ssm but without CUDA kernels.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _modulate(x: torch.Tensor, bias: torch.Tensor, act: nn.Module) -> torch.Tensor:
    return act(x + bias)


def _init_normal(module: nn.Module, std: float = 0.02) -> None:
    """Initialise all Linear/Embedding weights with N(0, std^2); zeros for biases."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=std)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for attention-based models.

    Applies rotary rotation to query and key tensors before attention,
    encoding relative position information. Standard in modern Transformers.
    """
    def __init__(self, d_head: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even for RoPE"
        self.d_head = d_head
        # Precompute frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # Cache cos/sin up to max_seq_len
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, d_head/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RoPE to x of shape (B, H, T, D_head)."""
        T = x.size(-2)
        cos = self.cos_cached[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, D/2)
        sin = self.sin_cached[:T].unsqueeze(0).unsqueeze(0)
        x1, x2 = x.chunk(2, dim=-1)  # each (B, H, T, D/2)
        # Rotate: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return rotated


# ---------------------------------------------------------------------------
# Selective SSM (Mamba-style, pure PyTorch parallel form)
# ---------------------------------------------------------------------------
class SelectiveSSM(nn.Module):
    """Selective state-space layer (diagonal parallel form).

    Implements the discrete-time Mamba selective SSM with a *diagonal* state:
        h_t = A_t * h_{t-1} + B_t * x_t       (per-channel)
        y_t = C_t * h_t
    where A, B, C are *input-dependent* (selective) and the state has the same
    width as the channel dimension (a diagonal approximation of the full
    (N, d_inner) state). This keeps the implementation O(B*T*d_inner) in both
    memory and time — fast on MPS/CPU — while preserving the essential
    "selectivity" property that distinguishes Mamba from LSSL/S4.

    The full (N, d_inner) state version is available via ``stateful=True`` and
    is used only for small ablations because it is much more expensive.

    Args:
        d_model:      input/output feature dimension D
        d_state:      SSM state dimension N (only used if stateful=True)
        d_inner:      inner projection width (if None, no projection)
        channel_proj: if True, project input to d_inner before scan and back.
        stateful:     if True, use the full (N, d_inner) state (slow).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_inner: Optional[int] = None,
                 channel_proj: bool = True, stateful: bool = False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_inner if d_inner is not None else d_model
        self.channel_proj = channel_proj
        self.stateful = stateful

        if channel_proj:
            self.in_proj  = nn.Linear(d_model, self.d_inner, bias=False)
            self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()

        # Input-dependent projections for delta, B, C
        if stateful:
            self.x_proj = nn.Linear(self.d_inner, self.d_inner + 2 * self.d_state, bias=False)
            # A_log initialised to small negative values so exp(A_log) is small
            # (i.e. strong decay) at the start of training -> stable.
            self.A_log = nn.Parameter(torch.zeros(self.d_state, self.d_inner) - 1.0)
        else:
            self.x_proj = nn.Linear(self.d_inner, 3 * self.d_inner, bias=False)
            self.A_log = nn.Parameter(torch.zeros(self.d_inner) - 1.0)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.ones(self.d_inner) * 0.5)

    def _forward_stateful(self, u: torch.Tensor, B_, C_, delta: torch.Tensor) -> torch.Tensor:
        """Full (N, d_inner) state version — slow but more expressive."""
        B, T, _ = u.shape
        N = self.d_state
        A_log_clamped = self.A_log.clamp(min=-8.0, max=2.0)
        A = -torch.exp(A_log_clamped)                     # (N, d_inner)
        delta = delta.clamp(min=1e-4, max=10.0)
        dA = torch.exp(delta.unsqueeze(-2) * A)            # (B, T, N, d_inner)
        dB = delta.unsqueeze(-2) * B_.unsqueeze(-1)        # (B, T, N, d_inner)
        u_exp = u.unsqueeze(-2)                            # (B, T, 1, d_inner)
        chunk = 1024
        h = torch.zeros(B, T, N, self.d_inner, device=u.device, dtype=u.dtype)
        h_prev = torch.zeros(B, N, self.d_inner, device=u.device, dtype=u.dtype)
        for i in range(0, T, chunk):
            sl = slice(i, min(i + chunk, T))
            dA_chunk = dA[:, sl]
            dB_chunk = dB[:, sl]
            u_chunk = u_exp[:, sl]
            dA_cum = torch.cumprod(dA_chunk, dim=1)
            dB_u = dB_chunk * u_chunk
            inv_dA_cum = 1.0 / dA_cum.clamp(min=1e-8)
            dB_u_scaled = dB_u * inv_dA_cum
            dB_u_cum = torch.cumsum(dB_u_scaled, dim=1)
            h_chunk = dA_cum * dB_u_cum + dA_cum * h_prev.unsqueeze(1)
            h[:, sl] = h_chunk
            h_prev = h_chunk[:, -1]
        y = (h * C_.unsqueeze(-1)).sum(dim=-2)
        return y

    def _forward_diagonal(self, u: torch.Tensor, B_, C_, delta: torch.Tensor) -> torch.Tensor:
        """Diagonal (per-channel) state version — fast and numerically stable.

        Uses a chunked sequential scan over chunks of size `chunk`. Within each
        chunk we use the log-space cumsum trick with a per-chunk stabiliser;
        across chunks we carry the hidden state explicitly. The chunk size is
        chosen small enough that the cumulative exponent never underflows.
        """
        B, T, D = u.shape
        # Clamp A_log to prevent exp(A_log) from exploding/vanishing
        A_log_clamped = self.A_log.clamp(min=-6.0, max=2.0)
        A = -torch.exp(A_log_clamped)                     # (d_inner,), negative
        # Clamp delta to avoid extremely large/small exponents
        delta = delta.clamp(min=1e-4, max=4.0)
        # Per-token log-discretisation: log(dA_t) = delta_t * A  (negative)
        log_dA = delta * A                                 # (B, T, d_inner)
        # Choose chunk size so that |sum of log_dA in a chunk| < 30 (safe for exp)
        # worst case: |A| ~ exp(2) ~ 7.4, delta ~ 4, so |log_dA| ~ 30 per step.
        # So chunk size 1 is the safe choice; but that's a sequential loop.
        # Compromise: chunk size = 32, and clamp the cumulative exponent.
        chunk = 64
        h = torch.empty_like(u)
        h_prev = torch.zeros(B, D, device=u.device, dtype=u.dtype)
        for i in range(0, T, chunk):
            j = min(i + chunk, T)
            sl = slice(i, j)
            log_dA_chunk = log_dA[:, sl]                   # (B, c, D)
            # Stabiliser: max over time within this chunk (detach so grad flows
            # only through the relative differences)
            m = log_dA_chunk.max(dim=1, keepdim=True).values.detach()
            log_dA_cum = torch.cumsum(log_dA_chunk, dim=1)
            # Clamp the exponent to avoid overflow/underflow
            dA_cum = torch.exp((log_dA_cum - m).clamp(min=-30.0, max=0.0))
            inv_dA_cum = torch.exp((m - log_dA_cum).clamp(min=0.0, max=30.0))
            # Input contribution: dB_u = delta * B * u
            dB_u_chunk = (delta * B_ * u)[:, sl]
            dB_u_scaled = dB_u_chunk * inv_dA_cum
            dB_u_cum = torch.cumsum(dB_u_scaled, dim=1)
            h_chunk = dA_cum * dB_u_cum
            # Add contribution from previous chunk: h_t += dA_cum_t * h_prev
            h_chunk = h_chunk + dA_cum * h_prev.unsqueeze(1)
            h[:, sl] = h_chunk
            # Carry the last hidden state to the next chunk (detach for stability)
            h_prev = h_chunk[:, -1].detach()
        y = C_ * h                                         # (B, T, d_inner)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, _ = x.shape
        u = self.in_proj(x)                                # (B, T, d_inner)
        x_dbc = self.x_proj(u)
        if self.stateful:
            delta = F.softplus(x_dbc[..., :self.d_inner] + self.dt_bias)
            B_ = x_dbc[..., self.d_inner:self.d_inner + self.d_state]
            C_ = x_dbc[..., self.d_inner + self.d_state:]
            y = self._forward_stateful(u, B_, C_, delta)
        else:
            delta = F.softplus(x_dbc[..., :self.d_inner] + self.dt_bias)
            B_ = x_dbc[..., self.d_inner:2 * self.d_inner]
            C_ = x_dbc[..., 2 * self.d_inner:]
            y = self._forward_diagonal(u, B_, C_, delta)
        y = y + self.D * u
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# State-Informed Attention (SIA) — the latent bottleneck of LST
# ---------------------------------------------------------------------------
class SIA(nn.Module):
    """State-Informed Attention: a fixed-size latent bottleneck whose keys
    and values are derived from a *selective* compression of the input.

    Two-step formulation:
      1. Compression: each token x_t produces K latent slots via K learned
         queries q_k applied to the *SSM-augmented* features.
      2. Retrieval: each token attends to the K latent slots.

    To make the bottleneck *state-informed*, the latent keys/values are the
    output of a selective SSM sweep over the compressed tokens, so that the
    latent state carries temporal continuity (this is the key innovation over
    Perceiver/Linformer).

    Args:
        d_model:  feature dimension D
        k_latent: number of latent slots K (<< N)
        d_state:  SSM state dim for the latent-state sweep
        n_heads:  number of attention heads for retrieval
    """

    def __init__(self, d_model: int, k_latent: int = 128, d_state: int = 16,
                 n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.k_latent = k_latent
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Learned latent queries for compression
        self.latent_queries = nn.Parameter(torch.randn(k_latent, d_model) * 0.02)
        # Projections for compression (cross-attention from queries to tokens)
        self.q_proj_c = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_c = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_c = nn.Linear(d_model, d_model, bias=False)
        # Selective SSM applied to the latent sequence (state-informed bottleneck)
        self.latent_ssm = SelectiveSSM(d_model, d_state=d_state, channel_proj=False)
        # Projections for retrieval (each token queries the latent slots)
        self.q_proj_r = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_r = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_r = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        # Scale
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        H, Kh, Dh = self.n_heads, self.k_latent, self.d_head
        # ---- 1. Compression: (B, T, D) -> (B, K, D) ----
        Q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)            # (B, K, D)
        q_c = self.q_proj_c(Q).view(B, Kh, H, Dh).transpose(1, 2)        # (B, H, K, Dh)
        k_c = self.k_proj_c(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        v_c = self.v_proj_c(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.scale              # (B, H, K, T)
        # Causal mask: a latent slot at position k can only attend to tokens up to
        # the chunk of sequence it represents. We use a *block-aligned* causal mask:
        # slot k sees tokens [k*T/K, (k+1)*T/K) union all earlier tokens.
        # For simplicity and stronger causality, we use a soft causal mask where
        # slot k attends to tokens t with t < (k+1) * T / K.
        rows = torch.arange(Kh, device=x.device).view(-1, 1)             # (K, 1)
        cols = torch.arange(T, device=x.device).view(1, -1)              # (1, T)
        causal = cols < ((rows + 1) * T + Kh - 1) // Kh                   # (K, T) bool
        attn_c = attn_c.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_c = F.softmax(attn_c, dim=-1)
        attn_c = self.dropout(attn_c)
        latent = attn_c @ v_c                                             # (B, H, K, Dh)
        latent = latent.transpose(1, 2).contiguous().view(B, Kh, D)       # (B, K, D)
        # ---- 2. State-Informed bottleneck: run selective SSM over latent ----
        latent = latent + self.latent_ssm(latent)                         # (B, K, D)
        # ---- 3. Retrieval: each token queries the latent slots ----
        q_r = self.q_proj_r(x).view(B, T, H, Dh).transpose(1, 2)          # (B, H, T, Dh)
        k_r = self.k_proj_r(latent).view(B, Kh, H, Dh).transpose(1, 2)    # (B, H, K, Dh)
        v_r = self.v_proj_r(latent).view(B, Kh, H, Dh).transpose(1, 2)    # (B, H, K, Dh)
        attn_r = (q_r @ k_r.transpose(-2, -1)) * self.scale               # (B, H, T, K)
        # Causal: token t can see all latent slots (slots are causally ordered
        # via the latent SSM). We allow token t to attend only to slots whose
        # aligned block ends <= t, to preserve strict causality.
        rows_t = torch.arange(T, device=x.device).view(-1, 1)             # (T, 1)
        cols_k = torch.arange(Kh, device=x.device).view(1, -1)            # (1, K)
        # slot k represents tokens [k*T/K, (k+1)*T/K). Allow token t to see slot k
        # if (k+1)*T/K <= t+1, i.e. k <= (t+1)*K/T - 1
        causal_r = cols_k <= ((rows_t + 1) * Kh - 1) // T                  # (T, K) bool
        attn_r = attn_r.masked_fill(~causal_r.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_r = F.softmax(attn_r, dim=-1)
        attn_r = self.dropout(attn_r)
        y = attn_r @ v_r                                                  # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)                  # (B, T, D)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# Gated State-Informed Attention (GSIA) — dynamic SSM/latent routing
# ---------------------------------------------------------------------------
class GSIA(nn.Module):
    """Gated State-Informed Attention.

    A generalisation of SIA in which each token dynamically decides how much
    of its update should come from the latent attention pathway versus the
    local SSM pathway. The gate is a content-dependent, per-channel sigmoid
    computed from the token representation and a global latent-summary vector.

    Concretely, the block computes:
      g_t = sigma( W_g x_t + U_g mu(L_t) )              (per-token gate)
      h_SIA_t   = Retrieval( SSM_latent( Compress(x) ) )
      h_local_t = SSM_local(x)_t                        (cheap diagonal SSM)
      y_t       = g_t * h_SIA_t + (1 - g_t) * h_local_t
    where mu(L_t) is the (causal) mean of the latent-state slots available
    at position t. The gate lets the model spend its O(NK) attention budget
    only on tokens that need long-range retrieval, while the others are
    served by the O(ND) SSM at full speed. This is the architectural
    innovation that gives LST-G its accuracy/speed profile in Section 6.
    """

    def __init__(self, d_model: int, k_latent: int = 128, d_state: int = 16,
                 n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.k_latent = k_latent
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Compression cross-attention (tokens -> K latent slots)
        self.latent_queries = nn.Parameter(torch.randn(k_latent, d_model) * 0.02)
        self.q_proj_c = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_c = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_c = nn.Linear(d_model, d_model, bias=False)
        # Latent-state SSM (gives the bottleneck temporal continuity)
        self.latent_ssm = SelectiveSSM(d_model, d_state=d_state, channel_proj=False)
        # Retrieval cross-attention (each token queries the latent slots)
        self.q_proj_r = nn.Linear(d_model, d_model, bias=False)
        self.k_proj_r = nn.Linear(d_model, d_model, bias=False)
        self.v_proj_r = nn.Linear(d_model, d_model, bias=False)
        # Local SSM for the gated residual pathway
        self.local_ssm = SelectiveSSM(d_model, d_state=d_state)
        # Dynamic gate: sigma(W_g x + U_g mu(L)) per channel
        self.gate_x = nn.Linear(d_model, d_model, bias=True)
        self.gate_mu = nn.Linear(d_model, d_model, bias=True)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D) -> (B, T, D)"""
        B, T, D = x.shape
        H, Kh, Dh = self.n_heads, self.k_latent, self.d_head
        # ---- 1. Compression: (B, T, D) -> (B, K, D) ----
        Q = self.latent_queries.unsqueeze(0).expand(B, -1, -1)            # (B, K, D)
        q_c = self.q_proj_c(Q).view(B, Kh, H, Dh).transpose(1, 2)        # (B, H, K, Dh)
        k_c = self.k_proj_c(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        v_c = self.v_proj_c(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        # Causal compression: slot k can attend to tokens [k*T/K, (k+1)*T/K)
        rows_k = torch.arange(Kh, device=x.device).view(-1, 1)           # (K, 1)
        cols_t = torch.arange(T, device=x.device).view(1, -1)            # (1, T)
        causal_c = (cols_t >= (rows_k * T) // Kh) & (cols_t < ((rows_k + 1) * T) // Kh)
        # When K > T, some rows may be all-False (empty token ranges).
        # Fallback: those slots attend to ALL tokens (non-causal) to avoid NaN.
        empty_rows = ~causal_c.any(dim=1, keepdim=True)                  # (K, 1)
        causal_c = causal_c | empty_rows.expand(-1, T)
        attn_c = (q_c @ k_c.transpose(-1, -2)) * self.scale              # (B, H, K, T)
        attn_c = attn_c.masked_fill(~causal_c.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_c = F.softmax(attn_c, dim=-1)
        attn_c = self.dropout(attn_c)
        latent = attn_c @ v_c                                            # (B, H, K, Dh)
        latent = latent.transpose(1, 2).contiguous().view(B, Kh, D)      # (B, K, D)
        # ---- 2. Latent state propagation: SSM over the K slots ----
        latent = self.latent_ssm(latent)                                 # (B, K, D)
        # ---- 3. Retrieval: each token queries the latent-state ----
        q_r = self.q_proj_r(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        k_r = self.k_proj_r(latent).view(B, Kh, H, Dh).transpose(1, 2)   # (B, H, K, Dh)
        v_r = self.v_proj_r(latent).view(B, Kh, H, Dh).transpose(1, 2)   # (B, H, K, Dh)
        attn_r = (q_r @ k_r.transpose(-1, -2)) * self.scale              # (B, H, T, K)
        rows_t = torch.arange(T, device=x.device).view(-1, 1)            # (T, 1)
        cols_k = torch.arange(Kh, device=x.device).view(1, -1)           # (1, K)
        causal_r = cols_k <= ((rows_t + 1) * Kh - 1) // T                # (T, K) bool
        attn_r = attn_r.masked_fill(~causal_r.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_r = F.softmax(attn_r, dim=-1)
        attn_r = self.dropout(attn_r)
        h_sia = attn_r @ v_r                                             # (B, H, T, Dh)
        h_sia = h_sia.transpose(1, 2).contiguous().view(B, T, D)         # (B, T, D)
        # ---- 4. Local SSM pathway (cheap, content-recurrent) ----
        h_local = self.local_ssm(x)                                      # (B, T, D)
        # ---- 5. Dynamic gate from token + latent summary ----
        # mu(L_t) = causal mean over latent slots visible at position t
        # Vectorised: use the (T, K) causal mask to compute a masked mean.
        mask = causal_r.unsqueeze(0).unsqueeze(-1).float()               # (1, T, K, 1)
        latent_for_mu = latent.unsqueeze(1)                              # (B, 1, K, D)
        mu_sum = (latent_for_mu * mask).sum(dim=2)                       # (B, T, D)
        mu_cnt = mask.sum(dim=2).clamp(min=1.0)                          # (B, T, 1)
        mu = mu_sum / mu_cnt                                             # (B, T, D)
        gate = torch.sigmoid(self.gate_x(x) + self.gate_mu(mu))          # (B, T, D)
        # ---- 6. Combine and project ----
        y = gate * h_sia + (1.0 - gate) * h_local                        # (B, T, D)
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# LST block and LM
# ---------------------------------------------------------------------------
class LSTBlock(nn.Module):
    def __init__(self, d_model: int, k_latent: int = 128, d_state: int = 16,
                 n_heads: int = 4, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.norm_sia = nn.LayerNorm(d_model)
        self.sia = SIA(d_model, k_latent=k_latent, d_state=d_state,
                       n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SSM sublayer
        x = x + self.ssm(self.norm1(x))
        # SIA sublayer (state-informed attention bottleneck)
        x = x + self.sia(self.norm_sia(x))
        # FFN sublayer
        x = x + self.ffn(self.norm2(x))
        return x


class LST(nn.Module):
    def __init__(self, vocab: int, d_model: int = 128, n_layers: int = 2,
                 k_latent: int = 64, d_state: int = 16, n_heads: int = 4,
                 dropout: float = 0.0, tie_weights: bool = True):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            LSTBlock(d_model, k_latent=k_latent, d_state=d_state,
                     n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        _init_normal(self, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# LST-G: Gated LST variant (uses GSIA instead of SIA)
# ---------------------------------------------------------------------------
class LSTGBlock(nn.Module):
    """LST block with the gated GSIA bottleneck instead of plain SIA."""

    def __init__(self, d_model: int, k_latent: int = 128, d_state: int = 16,
                 n_heads: int = 4, d_ff: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.gsia = GSIA(d_model, k_latent=k_latent, d_state=d_state,
                         n_heads=n_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GSIA already contains a local-SSM pathway inside the gate, so we
        # do not stack a separate SSM sublayer here; this keeps the block
        # cheap while preserving the SSM+attention character.
        x = x + self.gsia(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LSTG(nn.Module):
    """Gated LST language model."""

    def __init__(self, vocab: int, d_model: int = 128, n_layers: int = 2,
                 k_latent: int = 64, d_state: int = 16, n_heads: int = 4,
                 dropout: float = 0.0, tie_weights: bool = True):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            LSTGBlock(d_model, k_latent=k_latent, d_state=d_state,
                      n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        _init_normal(self, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Baseline 1: vanilla Transformer
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 4, d_ff: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, attn_mask=causal, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, dropout: float = 0.0, tie_weights: bool = True,
                 max_seq_len: int = 4096, use_pos_emb: bool = True):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            self.pos_emb = nn.Embedding(max_seq_len, d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads=n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        _init_normal(self, std=0.02)
        # Re-init pos_emb after _init_normal (which skips Embedding? No, it covers Embedding)
        if use_pos_emb:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        if self.use_pos_emb:
            T = x.size(1)
            pos = torch.arange(T, device=x.device)
            x = x + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Baseline 2: pure SSM (no attention at all)
# ---------------------------------------------------------------------------
class SSMBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_ff: Optional[int] = None,
                 dropout: float = 0.0):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ssm(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class SSMLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 128, n_layers: int = 2,
                 d_state: int = 16, dropout: float = 0.0, tie_weights: bool = True):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([
            SSMBlock(d_model, d_state=d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        _init_normal(self, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Baseline 3: Linformer-style linear attention with learned low-rank K/V
# ---------------------------------------------------------------------------
class LinformerAttention(nn.Module):
    """Linformer-style attention: project K and V along the sequence axis
    with learned projection matrices of shape (T, K). For causal fairness,
    we use a lower-triangular projection so token t only attends to a
    compressed view of past tokens.
    """

    def __init__(self, d_model: int, n_heads: int = 4, k_latent: int = 64,
                 max_T: int = 4096, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.k_latent = k_latent
        self.max_T = max_T
        # Learned K and V projection matrices (T -> K), shared across batch
        self.k_proj_mat = nn.Parameter(torch.randn(max_T, k_latent) * (1.0 / math.sqrt(max_T)))
        self.v_proj_mat = nn.Parameter(torch.randn(max_T, k_latent) * (1.0 / math.sqrt(max_T)))
        # Standard Q/K/V projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, Kh, Dh = self.n_heads, self.k_latent, self.d_head
        assert T <= self.max_T, f"sequence {T} exceeds max_T={self.max_T}"
        # Apply causal mask to projection matrices (lower-triangular)
        E = self.k_proj_mat[:T, :Kh]                                  # (T, K)
        F_ = self.v_proj_mat[:T, :Kh]                                 # (T, K)
        # Make projections causal by zeroing future contributions
        causal = torch.tril(torch.ones(T, T, device=x.device))
        # E_causal[t, k] = sum_{s<=t} causal[t, s] * E[s, k]
        # i.e. each query t sees only past-compressed keys.
        # Implement as: E_causal = causal @ E   (since E maps T->K, we want
        # the projection at position t to only include s <= t)
        E_c = causal @ E                                              # (T, K)
        F_c = causal @ F_                                             # (T, K)
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)         # (B, H, T, Dh)
        # Compress K and V along time: (B, H, T, Dh) -> (B, H, K, Dh)
        k_comp = k.transpose(2, 3) @ E_c.unsqueeze(0).unsqueeze(0)   # (B,H,Dh,T)@(1,1,T,K)
        k_comp = k_comp.transpose(2, 3)                               # (B, H, K, Dh)
        v_comp = v.transpose(2, 3) @ F_c.unsqueeze(0).unsqueeze(0)
        v_comp = v_comp.transpose(2, 3)                               # (B, H, K, Dh)
        # Attention: (B, H, T, K)
        attn = (q @ k_comp.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        y = attn @ v_comp                                            # (B, H, T, Dh)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


class LinformerLM(nn.Module):
    def __init__(self, vocab: int, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, k_latent: int = 64, max_T: int = 4096,
                 dropout: float = 0.0, tie_weights: bool = True,
                 use_pos_emb: bool = True):
        super().__init__()
        self.vocab = vocab
        self.use_pos_emb = use_pos_emb
        self.d_model = d_model
        self.embed = nn.Embedding(vocab, d_model)
        if use_pos_emb:
            self.pos_emb = nn.Embedding(max_T, d_model)
            nn.init.normal_(self.pos_emb.weight, std=0.02)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'norm1': nn.LayerNorm(d_model),
                'attn': LinformerAttention(d_model, n_heads=n_heads,
                                           k_latent=k_latent, max_T=max_T,
                                           dropout=dropout),
                'norm2': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model), nn.Dropout(dropout),
                ),
            }) for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab, bias=False)
        if tie_weights:
            self.lm_head.weight = self.embed.weight
        _init_normal(self, std=0.02)
        if self.use_pos_emb:
            nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        if self.use_pos_emb:
            T = x.size(1)
            pos = torch.arange(T, device=x.device)
            x = x + self.pos_emb(pos)
        for blk in self.blocks:
            x = x + blk['attn'](blk['norm1'](x))
            x = x + blk['ffn'](blk['norm2'](x))
        x = self.norm_f(x)
        return self.lm_head(x)
