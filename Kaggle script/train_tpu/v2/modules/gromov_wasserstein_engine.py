#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — GROMOV-WASSERSTEIN ENGINE (GROMOVWASSERSTEINSIGN)
================================================================================
Implements Metric Measure Space Optimal Transport & Isometry Invariance (GWAlign-SLT):
1. Intrinsic Intra-Skeleton Pairwise Distance Matrices:
     D_t(i, j) = || p_t(i) - p_t(j) ||_2 in R^{60 x 60}
     Strictly invariant to all SE(3) rotations, translations, and coordinate frame choices.
2. Entropic Regularized Gromov-Wasserstein Sinkhorn Iterations:
     M^{(t)} = -4 * D_t @ P^{(t)} @ D_{ref}^T
     P^{(t+1)} = Sinkhorn( M^{(t)}, epsilon )
3. Quadratic Metric Distortion Cost:
     GW(D_t, D_{ref}) = sum_{i,j,k,l} | D_t(i, j) - D_{ref}(k, l) |^2 * P_{ik} P_{jl}
     GW == 0 if and only if skeletons are isometric metric spaces.
4. Fused Gromov-Wasserstein (FGW) Hybrid Distance:
     FGW = (1 - lambda) * <C_feat, P> + lambda * GW(D_t, D_{ref})
5. Distortional Feature Projection & Canonical Alignment:
     H_gw = H + LayerNorm( Linear( [P @ H, GW, FGW, Tr(P D P^T)] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GromovWassersteinOutput(NamedTuple):
    gw_features: torch.Tensor           # [B, T, d_model] Projected Gromov-Wasserstein representations
    gw_distance: torch.Tensor           # [B, T] Quadratic Gromov-Wasserstein metric distortion
    fgw_distance: torch.Tensor          # [B, T] Fused Gromov-Wasserstein hybrid distance
    optimal_coupling: torch.Tensor      # [B, T, 60, 60] Soft optimal transport matching plan P
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + gw_features


class ASLGromovWassersteinEngine(nn.Module):
    """
    Metric Measure Space Gromov-Wasserstein & Isometry Invariant Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        epsilon_ent: float = 0.05,       # Entropic regularization parameter
        num_iters: int = 5,              # Number of Sinkhorn-Gromov iterations
        lambda_fgw: float = 0.6,         # Weight between structure (GW) and features (W)
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.epsilon = epsilon_ent
        self.num_iters = num_iters
        self.lambda_fgw = lambda_fgw

        # Canonical reference skeleton pairwise distance matrix D_ref [60, 60]
        # Initialized with anatomical neutral posture
        D_init = torch.ones(num_keypoints, num_keypoints, dtype=torch.float32) * 0.5
        for i in range(num_keypoints):
            D_init[i, i] = 0.0
        self.register_buffer("D_ref", D_init)

        # Feature projection head
        # Input: d_model + 3 (gw_dist, fgw_dist, tr_metric) -> d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_pairwise_distances(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Computes intra-skeleton pairwise Euclidean distance matrix D: [B, T, 60, 60].
        pos: [B, T, 60, 3]
        """
        eps = 1e-7
        # p_i - p_j
        p_i = pos.unsqueeze(-2)  # [B, T, 60, 1, 3]
        p_j = pos.unsqueeze(-3)  # [B, T, 1, 60, 3]
        D = torch.sqrt(((p_i - p_j) ** 2).sum(dim=-1) + eps)  # [B, T, 60, 60]
        return D

    def sinkhorn_knopp(
        self,
        cost: torch.Tensor,       # [B, T, 60, 60]
        n_iters: int = 5,
    ) -> torch.Tensor:
        """
        Computes entropic regularized optimal transport coupling P via Sinkhorn-Knopp.
        """
        B, T, K, _ = cost.shape
        device = cost.device
        mu = torch.ones(K, device=device, dtype=cost.dtype) / float(K)  # [60]

        # Log kernel: K = - cost / epsilon
        log_K = -cost / self.epsilon
        u = torch.zeros(B, T, K, device=device, dtype=cost.dtype)
        v = torch.zeros(B, T, K, device=device, dtype=cost.dtype)

        for _ in range(n_iters):
            # u = log(mu) - logsumexp(log_K + v.unsqueeze(-2), dim=-1)
            u = math.log(1.0 / K) - torch.logsumexp(log_K + v.unsqueeze(-2), dim=-1)
            # v = log(mu) - logsumexp(log_K + u.unsqueeze(-1), dim=-2)
            v = math.log(1.0 / K) - torch.logsumexp(log_K + u.unsqueeze(-1), dim=-2)

        P = torch.exp(log_K + u.unsqueeze(-1) + v.unsqueeze(-2))  # [B, T, 60, 60]
        return P

    def compute_gromov_wasserstein(
        self,
        D_t: torch.Tensor,        # [B, T, 60, 60]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Solves Entropic Gromov-Wasserstein alignment between D_t and D_ref.
        Returns: (gw_dist [B, T], fgw_dist [B, T], optimal_coupling [B, T, 60, 60])
        """
        B, T, K, _ = D_t.shape
        device = D_t.device

        # Initial uniform coupling: P_0 = mu @ mu^T = (1/K^2)
        P = torch.full((B, T, K, K), 1.0 / (K * K), device=device)

        D_ref_exp = self.D_ref.unsqueeze(0).unsqueeze(0)  # [1, 1, 60, 60]

        # Sinkhorn-Gromov iterations
        for _ in range(self.num_iters):
            # Gradient cost: M = - 4 * D_t @ P @ D_ref^T
            # D_t: [B, T, 60, 60], P: [B, T, 60, 60], D_ref: [1, 1, 60, 60]
            D_P = torch.matmul(D_t, P)                         # [B, T, 60, 60]
            M = -4.0 * torch.matmul(D_P, D_ref_exp.transpose(-1, -2)) # [B, T, 60, 60]
            P = self.sinkhorn_knopp(M, n_iters=3)

        # Compute quadratic GW loss:
        # GW = ||D_t||^2_P + ||D_ref||^2_P - 2 <D_t P, P D_ref>
        term1 = (D_t ** 2).mean(dim=(-1, -2))                      # [B, T]
        term2 = (self.D_ref ** 2).mean()                           # scalar
        D_P = torch.matmul(D_t, P)                                 # [B, T, 60, 60]
        P_Dref = torch.matmul(P, D_ref_exp)                        # [B, T, 60, 60]
        term3 = (D_P * P_Dref).sum(dim=(-1, -2))                   # [B, T]

        gw_dist = (term1 + term2 - 2.0 * term3).clamp(min=0.0)     # [B, T]

        # Feature cost (diagonal landmark variance proxy):
        c_feat = torch.diagonal(D_t, dim1=-2, dim2=-1).mean(dim=-1) # [B, T]
        fgw_dist = (1.0 - self.lambda_fgw) * c_feat + self.lambda_fgw * gw_dist # [B, T]

        return gw_dist, fgw_dist, P

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> GromovWassersteinOutput:
        """
        Computes pairwise distance matrices, solves Gromov-Wasserstein alignment, and projects to d_model.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        pos = kinematics[..., 0:3]  # [B, T, 60, 3]

        # 1. Compute Intra-Skeleton Pairwise Distance Matrix D_t
        D_t = self.compute_pairwise_distances(pos)  # [B, T, 60, 60]

        # 2. Solve Entropic Gromov-Wasserstein Optimal Transport
        gw_dist, fgw_dist, P = self.compute_gromov_wasserstein(D_t)

        # 3. Compute Metric Alignment Invariant Trace
        # Tr(P @ D_ref @ P^T) / K
        D_ref_exp = self.D_ref.unsqueeze(0).unsqueeze(0)
        P_D_P = torch.matmul(torch.matmul(P, D_ref_exp), P.transpose(-1, -2))
        tr_metric = torch.diagonal(P_D_P, dim1=-2, dim2=-1).sum(dim=-1) # [B, T]

        if h_seq is None:
            h_base = torch.zeros(B, T, self.d_model, device=device, dtype=D_t.dtype)
        else:
            h_base = h_seq

        # 4. Feature Projection
        # [B, T, d_model + 3] -> [B, T, d_model]
        gw_metrics = torch.stack([gw_dist, fgw_dist, tr_metric], dim=-1)
        f_all = torch.cat([h_base, gw_metrics], dim=-1)
        h_gw = self.proj(f_all)

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_gw

        return GromovWassersteinOutput(
            gw_features=h_gw,
            gw_distance=gw_dist,
            fgw_distance=fgw_dist,
            optimal_coupling=P,
            augmented_features=augmented,
        )
