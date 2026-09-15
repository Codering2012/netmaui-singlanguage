#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — PRODUCT MANIFOLD GEOMETRY ENGINE (PRODUCTMANIFOLDSIGN)
================================================================================
Implements Mixed-Curvature Product Manifold M = E^{d_E} x H^{d_H} x S^{d_S} (GeoSign):
1. Euclidean Sub-Space E^{d_E} (Zero Curvature kappa = 0):
     d_E(u, v) = ||u - v||_2   (captures linear coordinates and flat motion)
2. Hyperbolic Sub-Space H^{d_H} (Negative Curvature kappa = -c < 0, Poincare Ball):
     x_H = Exp_0^c(v) = tanh(sqrt(c)*||v||) * (v / (sqrt(c)*||v||))
     d_H(u, v) = (1/sqrt(c)) * arcosh( 1 + 2*c*||u - v||^2 / ((1 - c*||u||^2)*(1 - c*||v||^2)) )
     (captures hierarchical taxonomic trees and phoneme parent-child categories)
3. Spherical Sub-Space S^{d_S} (Positive Curvature kappa = +1 > 0, Hypersphere):
     x_S = v / ||v||_2 in S^{d_S}
     d_S(u, v) = arccos( clamp( <u, v>, -1.0, 1.0 ) )
     (captures cyclic gestures, rotational orientation orbits, and SO(3) loops)
4. Overall Product Manifold Geodesic Metric:
     d_M^2(x, y) = d_E^2(x_E, y_E) + d_H^2(x_H, y_H) + d_S^2(x_S, y_S)
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductManifoldOutput(NamedTuple):
    manifold_features: torch.Tensor     # [B, T, d_model] Fused product manifold representations
    euclidean_coords: torch.Tensor      # [B, T, d_euc] Flat Euclidean embeddings
    hyperbolic_coords: torch.Tensor     # [B, T, d_hyp] Poincare ball embeddings in B_c
    spherical_coords: torch.Tensor      # [B, T, d_sph] Hypersphere embeddings on S^{d_sph}
    pairwise_distances: torch.Tensor    # [B, B] Geodesic distance matrix on M
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + manifold_features


class ASLProductManifoldGeometryEngine(nn.Module):
    """
    Mixed-Curvature Product Manifold (Euclidean x Hyperbolic x Spherical) Geometry Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_euc: int = 64,              # Euclidean dimension
        d_hyp: int = 32,              # Hyperbolic dimension
        d_sph: int = 32,              # Spherical dimension
        curvature: float = 1.0,       # Hyperbolic curvature parameter c
    ):
        super().__init__()
        self.d_model = d_model
        self.d_euc = d_euc
        self.d_hyp = d_hyp
        self.d_sph = d_sph
        self.c = curvature

        # 1. Projections from model dimension to manifold sub-spaces
        self.proj_euc = nn.Linear(d_model, d_euc)
        self.proj_hyp = nn.Linear(d_model, d_hyp)
        self.proj_sph = nn.Linear(d_model, d_sph)

        # 2. Output projection from manifold concatenation back to d_model
        total_dim = d_euc + d_hyp + d_sph
        self.out_proj = nn.Sequential(
            nn.Linear(total_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def exp_map_poincare(self, v: torch.Tensor) -> torch.Tensor:
        """
        Exponential map from tangent space at origin to Poincare ball B_c: Exp_0^c(v).
        Returns: x_H in B_c with ||x_H|| < 1/sqrt(c)
        """
        eps = 1e-6
        sqrt_c = math.sqrt(self.c)
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True) + eps  # [..., 1]
        # tanh(sqrt(c) * ||v||) / (sqrt(c) * ||v||) * v
        scale = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        x_hyp = scale * v
        # Ensure strict containment within radius (1/sqrt_c - 1e-4)
        max_rad = (1.0 / sqrt_c) - 1e-4
        x_hyp_norm = torch.norm(x_hyp, p=2, dim=-1, keepdim=True) + eps
        clamp_scale = torch.clamp(max_rad / x_hyp_norm, max=1.0)
        return x_hyp * clamp_scale

    def hyperbolic_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Poincare ball geodesic distance: d_H(u, v).
        u, v: [..., d_hyp]
        Returns: [...]
        """
        eps = 1e-6
        sqrt_c = math.sqrt(self.c)
        sq_diff = (u - v).pow(2).sum(dim=-1)                   # [...]
        u_sq = u.pow(2).sum(dim=-1)                           # [...]
        v_sq = v.pow(2).sum(dim=-1)                           # [...]

        denom = (1.0 - self.c * u_sq).clamp(min=eps) * (1.0 - self.c * v_sq).clamp(min=eps)
        delta = 2.0 * self.c * sq_diff / denom                # [...]
        arcosh_val = torch.acosh((1.0 + delta).clamp(min=1.0))
        arcosh_val = torch.where(sq_diff < 1e-12, torch.zeros_like(arcosh_val), arcosh_val)
        return (1.0 / sqrt_c) * arcosh_val

    def spherical_distance(self, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Great-circle geodesic distance on unit hypersphere: d_S(u, v) = arccos(<u, v>).
        u, v: [..., d_sph] (unit vectors)
        Returns: [...]
        """
        dot = (u * v).sum(dim=-1)
        dot_clamped = torch.clamp(dot, -1.0, 1.0)
        dist = torch.acos(dot_clamped)
        dist = torch.where((u - v).pow(2).sum(dim=-1) < 1e-12, torch.zeros_like(dist), dist)
        return dist

    def forward(
        self,
        h_seq: torch.Tensor,  # [B, T, d_model] Input feature sequence
    ) -> ProductManifoldOutput:
        """
        Projects inputs to mixed-curvature product manifold, computes geodesic distances, and fuses representations.
        """
        B, T, D = h_seq.shape
        device = h_seq.device
        eps = 1e-6

        # 1. Euclidean Sub-Space: x_E in R^{d_euc}
        x_euc = self.proj_euc(h_seq)  # [B, T, d_euc]

        # 2. Hyperbolic Sub-Space: x_H in B_c^{d_hyp}
        v_hyp = self.proj_hyp(h_seq)  # [B, T, d_hyp]
        x_hyp = self.exp_map_poincare(v_hyp)  # [B, T, d_hyp]

        # 3. Spherical Sub-Space: x_S in S^{d_sph}
        v_sph = self.proj_sph(h_seq)  # [B, T, d_sph]
        x_sph = F.normalize(v_sph, p=2, dim=-1, eps=eps)  # [B, T, d_sph]

        # 4. Fused Manifold Concatenation: [B, T, d_euc + d_hyp + d_sph]
        x_manifold_concat = torch.cat([x_euc, x_hyp, x_sph], dim=-1)  # [B, T, total_dim]
        m_out = self.out_proj(x_manifold_concat)  # [B, T, d_model]

        # 5. Compute Batch-Level Pairwise Geodesic Distance Matrix on M: [B, B]
        # Pool sequences to [B, d] for sequence-level distance matrix
        e_pool = x_euc.mean(dim=1)   # [B, d_euc]
        h_pool = x_hyp.mean(dim=1)   # [B, d_hyp]
        s_pool = F.normalize(x_sph.mean(dim=1), p=2, dim=-1)  # [B, d_sph]

        # Pairwise Euclidean distance squared
        e_i = e_pool.unsqueeze(1)    # [B, 1, d_euc]
        e_j = e_pool.unsqueeze(0)    # [1, B, d_euc]
        d_euc_sq = (e_i - e_j).pow(2).sum(dim=-1)  # [B, B]

        # Pairwise Hyperbolic distance squared
        h_i = h_pool.unsqueeze(1).expand(B, B, self.d_hyp)  # [B, B, d_hyp]
        h_j = h_pool.unsqueeze(0).expand(B, B, self.d_hyp)  # [B, B, d_hyp]
        d_hyp_sq = self.hyperbolic_distance(h_i, h_j).pow(2)  # [B, B]

        # Pairwise Spherical distance squared
        s_i = s_pool.unsqueeze(1).expand(B, B, self.d_sph)  # [B, B, d_sph]
        s_j = s_pool.unsqueeze(0).expand(B, B, self.d_sph)  # [B, B, d_sph]
        d_sph_sq = self.spherical_distance(s_i, s_j).pow(2)  # [B, B]

        # Combined product manifold distance: d_M = sqrt(d_E^2 + d_H^2 + d_S^2)
        dist_M = torch.sqrt((d_euc_sq + d_hyp_sq + d_sph_sq).clamp(min=0.0))  # [B, B]

        augmented = h_seq + m_out

        return ProductManifoldOutput(
            manifold_features=m_out,
            euclidean_coords=x_euc,
            hyperbolic_coords=x_hyp,
            spherical_coords=x_sph,
            pairwise_distances=dist_M,
            augmented_features=augmented,
        )
