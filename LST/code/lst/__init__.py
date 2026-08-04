"""Latent State Transformer (LST) — minimal PyTorch implementation.

This package implements:
  * SelectiveSSM     — a Mamba-style selective state-space layer (pure PyTorch).
  * SIA              — the State-Informed Attention latent bottleneck.
  * GSIA             — Gated State-Informed Attention (dynamic SSM/latent routing).
  * LSTBlock         — SSM -> SIA -> FFN with Pre-LN and residuals.
  * LST             — a stacked LST language model.
  * LSTGBlock / LSTG — gated LST variant using GSIA.
  * TransformerBlock / TransformerLM     — vanilla baseline.
  * SSMBlock / SSMLM                    — pure-SSM baseline.
  * LinformerAttention / LinformerLM     — linear-attention baseline.
"""
from .models import (
    SelectiveSSM,
    SIA,
    GSIA,
    LSTBlock,
    LST,
    LSTGBlock,
    LSTG,
    TransformerBlock,
    TransformerLM,
    SSMBlock,
    SSMLM,
    LinformerAttention,
    LinformerLM,
    RotaryPositionalEmbedding,
)
from .tasks import (
    MQARDataset,
    InductionDataset,
    CopyMemoryDataset,
    InductionPairsDataset,
    CopySegmentDataset,
    collate_fixed,
)
from .train import TrainConfig, train_model, evaluate_model, get_param_count
from .adaptive import (
    CASIA, CASIA2, AASIA, LSTCABlock, LSTCA, LSTCABlock2, LSTCA2,
    LSTAABlock, LSTAA, soft_top_k_mask, causal_top_k,
)
__all__ = [
    "SelectiveSSM", "SIA", "GSIA", "LSTBlock", "LST", "LSTGBlock", "LSTG",
    "TransformerBlock", "TransformerLM",
    "SSMBlock", "SSMLM",
    "LinformerAttention", "LinformerLM",
    "RotaryPositionalEmbedding",
    "MQARDataset", "InductionDataset", "CopyMemoryDataset",
    "InductionPairsDataset", "CopySegmentDataset",
    "collate_fixed",
    "TrainConfig", "train_model", "evaluate_model", "get_param_count",
    # Content-adaptive variants (this paper's innovation)
    "CASIA", "CASIA2", "AASIA",
    "LSTCABlock", "LSTCA", "LSTCABlock2", "LSTCA2",
    "LSTAABlock", "LSTAA",
    "soft_top_k_mask", "causal_top_k",
]
