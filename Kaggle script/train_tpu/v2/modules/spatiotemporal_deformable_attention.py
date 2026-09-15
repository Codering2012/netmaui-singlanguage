#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SPATIOTEMPORAL DEFORMABLE ATTENTION ENGINE
================================================================================
Implements Spatiotemporal Deformable Attention & Trajectory Correlation (DAT-SLT):
1. Continuous 2D/3D Spatiotemporal Offset Generation:
     For each query joint k at time t, predicts M attention heads x N_pts sampling offsets:
     (Delta t, Delta k) in [-R_t, +R_t] x [-R_k, +R_k]
2. Differentiable Bilinear Manifold Sampling:
     Samples features at continuous (t + Delta t, k + Delta k) coordinates using
     pure PyTorch vectorized differentiable bilinear interpolation (100% TPU/XLA native).
     x(t*, k*) = sum_{i,j in {0,1}} w_{ij} * x_{t_i, k_j}
3. Adaptive Sparse Feature Aggregation:
     Replaces O((T*K)^2) dense attention with O(T * K * M * N_pts) sparse content-aware
     aggregation focused on active signing articulators (hands, face, trajectory stroke).
4. Trajectory Offset Regularization Loss:
     L_offset = (1 / (B*T*K)) * sum ||Delta p||_2^2 preventing unconstrained drift.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformableAttentionOutput(NamedTuple):
    output: torch.Tensor                 # [B, T, K, D] or [B, T, D]
    offset_loss: torch.Tensor            # Scalar offset regularization loss
    sampling_offsets: torch.Tensor       # [B, T, K, M, N_pts, 2]
    attention_weights: torch.Tensor      # [B, T, K, M, N_pts]


class SpatiotemporalDeformableAttention(nn.Module):
    """
    Spatiotemporal Deformable Attention over (Time, Joint) Keypoint Manifolds.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_points: int = 4,
        max_time_offset: float = 4.0,
        max_joint_offset: float = 8.0,
        offset_loss_weight: float = 0.01,
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model {d_model} must be divisible by num_heads {num_heads}"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = d_model // num_heads
        self.max_time_offset = max_time_offset
        self.max_joint_offset = max_joint_offset
        self.offset_loss_weight = offset_loss_weight

        # Offset prediction head: predicts (Delta t, Delta k) for each head and point
        self.offset_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, num_heads * num_points * 2),
        )

        # Attention weight projection: Softmax over num_points
        self.attn_weight_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, num_heads * num_points),
        )

        # Value projection & Output projection
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize offset proj with small weights so initial sampling is localized
        nn.init.zeros_(self.offset_proj[-1].weight)
        nn.init.zeros_(self.offset_proj[-1].bias)
        nn.init.zeros_(self.attn_weight_proj[-1].weight)
        nn.init.zeros_(self.attn_weight_proj[-1].bias)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def _bilinear_interpolate(
        self,
        values: torch.Tensor,       # [B, T, K, M, head_dim]
        t_coords: torch.Tensor,     # [B, T, K, M, N_pts]
        k_coords: torch.Tensor,     # [B, T, K, M, N_pts]
    ) -> torch.Tensor:
        """
        Purely vectorized, 100% differentiable bilinear interpolation on (T, K) grid.
        Returns: [B, T, K, M, N_pts, head_dim]
        """
        B, T, K, M, D_head = values.shape
        device = values.device

        # Clamp continuous coordinates to grid boundaries
        t_clamped = torch.clamp(t_coords, 0.0, float(T - 1))
        k_clamped = torch.clamp(k_coords, 0.0, float(K - 1))

        t0 = torch.floor(t_clamped).long()
        t1 = torch.clamp(t0 + 1, max=T - 1)
        k0 = torch.floor(k_clamped).long()
        k1 = torch.clamp(k0 + 1, max=K - 1)

        alpha = (t_clamped - t0.float()).unsqueeze(-1)  # [B, T, K, M, N_pts, 1]
        beta = (k_clamped - k0.float()).unsqueeze(-1)   # [B, T, K, M, N_pts, 1]

        # Expand batch and head indices for advanced indexing
        b_idx = torch.arange(B, device=device).view(B, 1, 1, 1, 1).expand_as(t0)
        m_idx = torch.arange(M, device=device).view(1, 1, 1, M, 1).expand_as(t0)

        # Gather the 4 bounding corner values: values[b, t, k, m, :]
        v00 = values[b_idx, t0, k0, m_idx]  # [B, T, K, M, N_pts, D_head]
        v10 = values[b_idx, t1, k0, m_idx]  # [B, T, K, M, N_pts, D_head]
        v01 = values[b_idx, t0, k1, m_idx]  # [B, T, K, M, N_pts, D_head]
        v11 = values[b_idx, t1, k1, m_idx]  # [B, T, K, M, N_pts, D_head]

        # Bilinear combination
        w00 = (1.0 - alpha) * (1.0 - beta)
        w10 = alpha * (1.0 - beta)
        w01 = (1.0 - alpha) * beta
        w11 = alpha * beta

        sampled = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11
        return sampled

    def forward(
        self,
        x: torch.Tensor,  # [B, T, K, D] (or [B, T, D] which will be reshaped with K=1)
        pos_embed: Optional[torch.Tensor] = None,
    ) -> DeformableAttentionOutput:
        """
        Forward pass for Spatiotemporal Deformable Attention.
        """
        orig_shape = x.shape
        if x.dim() == 3:
            # [B, T, D] -> [B, T, 1, D]
            B, T, D = x.shape
            K = 1
            x = x.unsqueeze(2)
        elif x.dim() == 4:
            B, T, K, D = x.shape
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {orig_shape}")

        device = x.device
        residual = x

        # Add optional positional embedding
        query_input = x if pos_embed is None else x + pos_embed

        # 1. Project values and reshape into multi-head format: [B, T, K, M, head_dim]
        v = self.value_proj(x).view(B, T, K, self.num_heads, self.head_dim)

        # 2. Predict continuous spatiotemporal offsets: [B, T, K, M, N_pts, 2]
        raw_offsets = self.offset_proj(query_input).view(B, T, K, self.num_heads, self.num_points, 2)
        # Scale offsets by max allowed range via tanh
        scaled_offsets = torch.tanh(raw_offsets)
        delta_t = scaled_offsets[..., 0] * self.max_time_offset    # [B, T, K, M, N_pts]
        delta_k = scaled_offsets[..., 1] * self.max_joint_offset   # [B, T, K, M, N_pts]

        # 3. Reference grid coordinates (t_ref, k_ref)
        t_ref = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1, 1, 1).expand(B, T, K, self.num_heads, self.num_points)
        k_ref = torch.arange(K, device=device, dtype=torch.float32).view(1, 1, K, 1, 1).expand(B, T, K, self.num_heads, self.num_points)

        sample_t = t_ref + delta_t
        sample_k = k_ref + delta_k

        # 4. Predict attention weights: Softmax over N_pts -> [B, T, K, M, N_pts, 1]
        raw_attn = self.attn_weight_proj(query_input).view(B, T, K, self.num_heads, self.num_points)
        attn_weights = F.softmax(raw_attn, dim=-1).unsqueeze(-1)  # [B, T, K, M, N_pts, 1]

        # 5. Differentiable bilinear manifold sampling: [B, T, K, M, N_pts, head_dim]
        sampled_feats = self._bilinear_interpolate(v, sample_t, sample_k)

        # 6. Weighted aggregation over sampled points: [B, T, K, M, head_dim]
        aggregated_heads = (sampled_feats * attn_weights).sum(dim=4)

        # 7. Reshape and project output: [B, T, K, D]
        aggregated = aggregated_heads.view(B, T, K, self.d_model)
        out = self.out_proj(aggregated)
        out = self.norm(out + residual)

        # 8. Offset regularization loss (prevents erratic extreme drift)
        offset_loss = (delta_t.pow(2) + delta_k.pow(2)).mean() * self.offset_loss_weight

        # Restore original dimension if input was 3D
        if len(orig_shape) == 3:
            out = out.squeeze(2)

        return DeformableAttentionOutput(
            output=out,
            offset_loss=offset_loss,
            sampling_offsets=scaled_offsets,
            attention_weights=attn_weights.squeeze(-1),
        )


class SpatiotemporalDeformableTransformerBlock(nn.Module):
    """
    Complete Transformer block using Spatiotemporal Deformable Attention & MLP.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_heads: int = 4,
        num_points: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.attn = SpatiotemporalDeformableAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_points=num_points,
        )
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,  # [B, T, K, D]
        pos_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(x, pos_embed=pos_embed)
        out = attn_out.output + self.mlp(self.norm(attn_out.output))
        return out, attn_out.offset_loss
