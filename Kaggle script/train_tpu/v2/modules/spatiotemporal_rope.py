#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SPATIOTEMPORAL 3D AXIAL ROTARY POSITION EMBEDDING (3D-ROPE)
================================================================================
Implements 3D Axial Rotary Position Embedding (3D-RoPE) for Sign Language Kinematics:
1. Tri-Axial Channel Subspace Partitioning:
     Head dimension d is partitioned into 3 orthogonal frequency subspaces:
     d = d_t + d_k + d_r (e.g. 32 temporal + 16 spatial joint + 16 anatomical region = 64)
2. Pure Relative Invariance in Attention:
     < R(t1, k1, r1) q, R(t2, k2, r2) k > = g(q, k, Delta t, Delta k, Delta r)
     Eliminates additive positional distortion while preserving exact relative distances.
3. Long-Sequence Infinite Extrapolation:
     Enables smooth generalization from training lengths (T=32) to streaming lengths (T=256+).
4. High-Precision Float32 Angle Formulation:
     Computes cos/sin in float32 prior to casting to bfloat16/float16 to avoid
     low-precision rotation truncation errors.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatiotemporalRoPE(nn.Module):
    """
    3D Axial Rotary Position Embedding across (Time, Joint Index, Anatomical Region).
    """

    def __init__(
        self,
        head_dim: int = 64,
        d_t: Optional[int] = None,
        d_k: Optional[int] = None,
        d_r: Optional[int] = None,
        base_t: float = 10000.0,
        base_k: float = 1000.0,
        base_r: float = 100.0,
    ):
        super().__init__()
        assert head_dim % 2 == 0, f"head_dim {head_dim} must be even!"
        self.head_dim = head_dim

        # Default partition: 50% Time, 25% Joint, 25% Region
        if d_t is None or d_k is None or d_r is None:
            self.d_t = head_dim // 2
            remaining = head_dim - self.d_t
            self.d_k = remaining // 2
            self.d_r = remaining - self.d_k
        else:
            self.d_t = d_t
            self.d_k = d_k
            self.d_r = d_r

        assert self.d_t % 2 == 0 and self.d_k % 2 == 0 and self.d_r % 2 == 0, "All sub-dimensions must be even!"
        assert self.d_t + self.d_k + self.d_r == head_dim, "Sub-dimensions must sum to head_dim!"

        # Precompute base inverse frequencies
        inv_freq_t = 1.0 / (base_t ** (torch.arange(0, self.d_t, 2).float() / self.d_t))
        inv_freq_k = 1.0 / (base_k ** (torch.arange(0, self.d_k, 2).float() / self.d_k))
        inv_freq_r = 1.0 / (base_r ** (torch.arange(0, self.d_r, 2).float() / self.d_r))

        self.register_buffer("inv_freq_t", inv_freq_t)
        self.register_buffer("inv_freq_k", inv_freq_k)
        self.register_buffer("inv_freq_r", inv_freq_r)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half of the coordinates: [-x2, x1].
        x: [..., d] where d is even
        """
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def _compute_axial_cos_sin(
        self,
        coords: torch.Tensor,      # [...] arbitrary coordinates
        inv_freq: torch.Tensor,    # [d // 2]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes cos and sin in float32 for high precision.
        """
        # Outer product: coords [...] x inv_freq [d // 2] -> [..., d // 2]
        freqs = torch.einsum("... , d -> ... d", coords.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # [..., d]
        return emb.cos(), emb.sin()

    def apply_rope_to_subspace(
        self,
        x_sub: torch.Tensor,       # [..., d_sub]
        coords: torch.Tensor,      # [...]
        inv_freq: torch.Tensor,    # [d_sub // 2]
    ) -> torch.Tensor:
        orig_dtype = x_sub.dtype
        cos, sin = self._compute_axial_cos_sin(coords, inv_freq)
        cos = cos.to(orig_dtype)
        sin = sin.to(orig_dtype)
        return (x_sub * cos) + (self._rotate_half(x_sub) * sin)

    def forward(
        self,
        q: torch.Tensor,                          # [B, num_heads, T, K, head_dim] or [B, num_heads, T, head_dim]
        k: torch.Tensor,                          # [B, num_heads, T, K, head_dim] or [B, num_heads, T, head_dim]
        t_coords: Optional[torch.Tensor] = None,  # [T]
        k_coords: Optional[torch.Tensor] = None,  # [K]
        r_coords: Optional[torch.Tensor] = None,  # [K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies 3D Axial RoPE to query and key tensors.
        """
        device = q.device
        is_4d = (q.dim() == 4)  # [B, M, T, D]
        if is_4d:
            # Reshape [B, M, T, D] -> [B, M, T, 1, D]
            q = q.unsqueeze(3)
            k = k.unsqueeze(3)

        B, M, T, K, D = q.shape

        # Default coordinate grids if not provided
        if t_coords is None:
            t_coords = torch.arange(T, device=device, dtype=torch.float32)  # [T]
        if k_coords is None:
            k_coords = torch.arange(K, device=device, dtype=torch.float32)  # [K]
        if r_coords is None:
            # Partition K=60 into regions: Face (0..13 -> 0), Pose (14..17 -> 1), Hands (18..59 -> 2)
            r_coords = torch.zeros(K, device=device, dtype=torch.float32)
            if K >= 18:
                r_coords[14:18] = 1.0
                r_coords[18:] = 2.0

        # Expand coordinate grids to broadcast with [B, M, T, K]
        # t: [1, 1, T, 1]
        t_grid = t_coords.view(1, 1, T, 1).expand(B, M, T, K)
        # k: [1, 1, 1, K]
        k_grid = k_coords.view(1, 1, 1, K).expand(B, M, T, K)
        # r: [1, 1, 1, K]
        r_grid = r_coords.view(1, 1, 1, K).expand(B, M, T, K)

        # 1. Split q and k into 3 axial subspaces
        q_t = q[..., :self.d_t]
        q_k = q[..., self.d_t:self.d_t + self.d_k]
        q_r = q[..., self.d_t + self.d_k:]

        k_t = k[..., :self.d_t]
        k_k = k[..., self.d_t:self.d_t + self.d_k]
        k_r = k[..., self.d_t + self.d_k:]

        # 2. Apply axial rotations
        q_rot_t = self.apply_rope_to_subspace(q_t, t_grid, self.inv_freq_t)
        q_rot_k = self.apply_rope_to_subspace(q_k, k_grid, self.inv_freq_k)
        q_rot_r = self.apply_rope_to_subspace(q_r, r_grid, self.inv_freq_r)

        k_rot_t = self.apply_rope_to_subspace(k_t, t_grid, self.inv_freq_t)
        k_rot_k = self.apply_rope_to_subspace(k_k, k_grid, self.inv_freq_k)
        k_rot_r = self.apply_rope_to_subspace(k_r, r_grid, self.inv_freq_r)

        # 3. Concatenate back to full head_dim
        q_out = torch.cat([q_rot_t, q_rot_k, q_rot_r], dim=-1)
        k_out = torch.cat([k_rot_t, k_rot_k, k_rot_r], dim=-1)

        if is_4d:
            q_out = q_out.squeeze(3)
            k_out = k_out.squeeze(3)

        return q_out, k_out
