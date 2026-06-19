import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MinimalMamba(nn.Module):
    """
    A simplified, pure PyTorch implementation of a Selective SSM (Mamba-like).
    This is functionally similar to Mamba but uses a sequential loop,
    so it is much slower than the optimized CUDA kernel version.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        # Projects input to d_inner * 2 (for x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Activation
        self.activation = nn.SiLU()

        # Projection for Delta, B, C
        # dt_rank = math.ceil(self.d_model / 16)
        dt_rank = self.d_inner // 16 if self.d_inner // 16 > 0 else 1
        self.x_proj = nn.Linear(self.d_inner, dt_rank + self.d_state * 2, bias=False)

        # dt_proj maps dt_rank to d_inner
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # A parameter (structured state matrix), simplified to diagonal
        # A is typically initialized as range 1..d_state
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A)) # Learn log(A) for stability
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Out projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        x: (Batch, SeqLen, Dim)
        """
        batch, seq_len, d_model = x.shape

        # 1. Project to higher dimension
        xz = self.in_proj(x) # (B, L, 2*D_inner)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # 2. Convolution
        x_branch = x_branch.transpose(1, 2) # (B, D_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :seq_len]
        x_branch = x_branch.transpose(1, 2) # (B, L, D_inner)

        x_branch = self.activation(x_branch)
        z_branch = self.activation(z_branch)

        # 3. SSM (Sequential scan for simplicity/portability)
        # In a real scenario, this would be a parallel scan or CUDA kernel

        # Prepare parameters
        # A = -exp(A_log)
        A = -torch.exp(self.A_log.float()) # (D_inner, D_state)

        # x_proj projects to (dt, B, C)
        x_dbl = self.x_proj(x_branch) # (B, L, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_proj.in_features, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt)) # (B, L, D_inner)

        # Scan
        # h_t = (1 + A * dt) * h_{t-1} + (B * dt) * x
        # actually typically discretized:
        # dA = exp(A * dt)
        # dB = (1 - exp(A * dt)) / A * B  ~ dt * B (for small dt)

        # We'll use a simplified discretization:
        # h_t = h_{t-1} + dt * (A h_{t-1} + B x_t)
        # Euler discretization:
        # h_t = (1 + dt*A) h_{t-1} + (dt*B) x_t
        # But Mamba uses ZOH usually. Let's use the explicit decay form:
        # decay = exp(A * dt)

        # A is (D_inner, D_state)
        # dt is (B, L, D_inner)
        # dt.unsqueeze(-1) * A -> (B, L, D_inner, D_state)
        decay = torch.exp(dt.unsqueeze(-1) * A)

        # input term: (B, L, D_inner, D_state)
        # B: (B, L, D_state) -> unsqueeze -> (B, L, 1, D_state)
        # x_branch: (B, L, D_inner) -> unsqueeze -> (B, L, D_inner, 1)
        # term = dt * B * x
        # standard is dB * x, where dB is discretized B.
        # dB approx dt * B

        input_term = dt.unsqueeze(-1) * B.unsqueeze(2) * x_branch.unsqueeze(-1)

        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device)
        ys = []

        # Sequential loop (slow but correct)
        for t in range(seq_len):
            h = decay[:, t] * h + input_term[:, t]

            # y = C * h
            # C: (B, L, D_state)
            # h: (B, D_inner, D_state)
            # y_t = sum(h * C_t, dim=-1)
            y_t = torch.einsum('bds,bs->bd', h, C[:, t])
            ys.append(y_t)

        y = torch.stack(ys, dim=1) # (B, L, D_inner)

        # 4. Residual connection (D * x)
        y = y + x_branch * self.D

        # 5. Output gating
        out = y * z_branch
        out = self.out_proj(out)

        return out


class StateInformedAttention(nn.Module):
    """
    SIA Block:
    1. Compresses sequence H (N x D) -> Latent L (K x D)
    2. Retrieves info L (K x D) -> H (N x D)
    """
    def __init__(self, d_model, n_latent, num_heads=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_latent = n_latent
        self.num_heads = num_heads

        # Latent queries
        self.latent_queries = nn.Parameter(torch.randn(1, n_latent, d_model))

        # Compression: Attention(Q=Latent, K=Input, V=Input)
        self.compression_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # Retrieval: Attention(Q=Input, K=Latent, V=Latent)
        self.retrieval_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, N, D)
        """
        B, N, D = x.shape

        # 1. Compression
        # Q: Latent queries broadcasted to batch (B, K, D)
        q_latent = self.latent_queries.expand(B, -1, -1)

        # K, V: x
        # Attn output: (B, K, D)
        latent_state, _ = self.compression_attn(q_latent, x, x)
        latent_state = self.dropout(latent_state)
        latent_state = self.norm1(latent_state)

        # 2. Retrieval
        # Q: x (B, N, D)
        # K, V: latent_state (B, K, D)
        # Attn output: (B, N, D)
        out, _ = self.retrieval_attn(x, latent_state, latent_state)

        return out

class LSTBlock(nn.Module):
    def __init__(self, d_model, n_latent=64, d_state=16, expand=2, num_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = MinimalMamba(d_model, d_state=d_state, expand=expand)

        self.norm2 = nn.LayerNorm(d_model)
        self.sia = StateInformedAttention(d_model, n_latent, num_heads, dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 1. SSM Branch
        # Note: In the paper, it says H = X + SSM(LN(X))
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = residual + self.dropout(x)

        # 2. SIA Branch
        residual = x
        x = self.norm2(x)
        x = self.sia(x)
        x = residual + self.dropout(x)

        # 3. FFN Branch
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x # FFN usually has dropout inside

        return x

class LSTModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_latent=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            LSTBlock(d_model, n_latent=n_latent, num_heads=num_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.head(x)
        return logits
