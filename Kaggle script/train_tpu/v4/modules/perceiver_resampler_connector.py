#!/usr/bin/env python3
"""
================================================================================
ASL V4: MULTIMODAL PERCEIVER RESAMPLER LLM CONNECTOR
================================================================================
Compresses arbitrary-length continuous sign representations (e.g. 64 condensed sign tokens)
into a compact, fixed set of N_latents=16 high-density prompt prefix embeddings
projected into the Foundation LLM's embedding space (D_llm = 2048 or 3072).

Features:
- Alternating Cross-Attention (Latents -> Visual Sequence) and Self-Attention (Latents -> Latents)
- 100% TPU v5e systolic tile alignment (16, 32, 64, 128 latents)
- Zero-overhead static computation graph
================================================================================
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResamplerLayer(nn.Module):
    """A single layer of the Perceiver Resampler with cross-attention, self-attention, and FFN."""

    def __init__(self, dim: int = 512, nhead: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead

        # 1. Cross-attention (Latent Queries attend to Visual Keys/Values)
        self.cross_norm_latents = nn.LayerNorm(dim)
        self.cross_norm_context = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)

        # 2. Self-attention (Latents attend to Latents)
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)

        # 3. Feedforward network
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,                                      # [B, num_latents, dim]
        context: torch.Tensor,                                      # [B, seq_len, dim]
        context_mask: Optional[torch.Tensor] = None,                # [B, seq_len]
    ) -> torch.Tensor:
        # Cross-Attention
        q = self.cross_norm_latents(latents)
        k = self.cross_norm_context(context)
        v = k
        key_padding_mask = (~context_mask.bool()) if context_mask is not None else None
        cross_out, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask)
        latents = latents + cross_out

        # Self-Attention
        norm_lat = self.self_norm(latents)
        self_out, _ = self.self_attn(norm_lat, norm_lat, norm_lat)
        latents = latents + self_out

        # FFN
        latents = latents + self.ffn(self.ffn_norm(latents))
        return latents


class PerceiverResamplerConnector(nn.Module):
    """
    Multimodal Perceiver Resampler projecting sign tokens into LLM prompt prefix space.
    """

    def __init__(
        self,
        dim: int = 512,
        dim_llm: int = 2048,
        num_latents: int = 16,
        depth: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim = dim
        self.dim_llm = dim_llm
        self.num_latents = num_latents

        # Learned latent query prototypes
        self.latents = nn.Parameter(torch.randn(num_latents, dim) * (1.0 / math.sqrt(dim)))

        # Perceiver layers
        self.layers = nn.ModuleList([
            PerceiverResamplerLayer(dim, nhead, dim_feedforward, dropout)
            for _ in range(depth)
        ])

        # Final projection to LLM embedding dimension with LayerNorm
        self.final_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim, dim_llm)

    def forward(
        self,
        context: torch.Tensor,                                      # [B, T_condensed, dim]
        context_mask: Optional[torch.Tensor] = None,                # [B, T_condensed]
    ) -> torch.Tensor:
        """
        Compresses context [B, T, dim] -> LLM prefix [B, num_latents, dim_llm]
        """
        B = context.shape[0]
        # Expand latents to batch size
        lat = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, num_latents, dim]

        for layer in self.layers:
            lat = layer(lat, context, context_mask=context_mask)

        lat = self.final_norm(lat)
        prefix_tokens = self.out_proj(lat)  # [B, num_latents, dim_llm]
        return prefix_tokens
