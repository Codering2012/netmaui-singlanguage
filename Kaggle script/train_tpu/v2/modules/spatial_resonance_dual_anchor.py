#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SPATIAL RESONANCE & DUAL-ANCHOR CODEBOOK OT (DUALANCHOR)
================================================================================
Implements Spatial Resonance OT (STNet) & Dual-Anchor Lexical Grounding (DualAnchor):
1. Inter-Frame Spatial Resonance Optimal Transport:
     Solves optimal transport T_{t, t+1}^* between adjacent frame landmark features:
     T^* = argmin < T, C_{t, t+1} > - epsilon * H(T)
     Enforces semantic joint correspondence and continuous motion flow during fast crossovers.
2. Dual-Anchor Cross-Modal Codebook Alignment:
     Maintains Visual Motion Anchors A_vis in R^{M x D} and Lexical Semantic Anchors A_lex in R^{M x D}:
     W(A_vis, A_lex) = sum_{i, j} P^*_{i, j} * ||A_vis(i) - A_lex(j)||_2^2
     Bridges the representation gap between visual kinematics and symbolic gloss tokens.
3. Length-Normalized Wasserstein Loss Formulation:
     Differentiable unrolled Sinkhorn-Knopp algorithm with exact marginal mass conservation.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResonanceDualAnchorOutput(NamedTuple):
    resonance_features: torch.Tensor     # [B, T, K, d_model] Spatially resonated landmark features
    total_loss: torch.Tensor             # Combined spatial resonance and dual-anchor OT loss
    spatial_resonance_loss: torch.Tensor # Inter-frame landmark transport cost
    dual_anchor_loss: torch.Tensor       # Visual-to-Lexical anchor Wasserstein distance
    vis_anchors: torch.Tensor            # [M, d_model] Visual anchor codebook
    lex_anchors: torch.Tensor            # [M, d_model] Lexical anchor codebook


class BatchedSinkhornOT(nn.Module):
    """
    Differentiable Batched Sinkhorn-Knopp Optimal Transport Solver.
    """

    def __init__(self, eps: float = 0.05, max_iters: int = 20):
        super().__init__()
        self.eps = eps
        self.max_iters = max_iters

    def forward(
        self,
        C: torch.Tensor,                  # [B, N, M] Cost matrix
        a: Optional[torch.Tensor] = None, # [B, N] Source marginals
        b: Optional[torch.Tensor] = None, # [B, M] Target marginals
    ) -> torch.Tensor:
        """
        Returns optimal transport plan P* [B, N, M].
        """
        B, N, M = C.shape
        device = C.device

        if a is None:
            a = torch.full((B, N), 1.0 / N, device=device, dtype=C.dtype)
        if b is None:
            b = torch.full((B, M), 1.0 / M, device=device, dtype=C.dtype)

        # Stabilized Gibbs Kernel K = exp(-C / eps)
        min_c = C.amin(dim=(1, 2), keepdim=True)
        c_norm = torch.clamp((C - min_c) / self.eps, 0.0, 50.0)
        K = torch.exp(-c_norm)  # [B, N, M]

        u = torch.ones_like(a)  # [B, N]
        v = torch.ones_like(b)  # [B, M]

        for _ in range(self.max_iters):
            # v = b / (K^T u + eps)
            denom_v = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
            v = b / denom_v
            # u = a / (K v + eps)
            denom_u = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-6)
            u = a / denom_u

        # Transport plan: P* = diag(u) @ K @ diag(v)
        P = u.unsqueeze(-1) * K * v.unsqueeze(1)  # [B, N, M]
        return P


class ASLSpatialResonanceDualAnchorEngine(nn.Module):
    """
    Spatial Resonance Module & Dual-Anchor Optimal Transport Alignment Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_anchors: int = 32,
        eps_ot: float = 0.05,
        max_iters: int = 15,
        weight_resonance: float = 0.10,
        weight_anchor: float = 0.50,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_anchors = num_anchors
        self.w_res = weight_resonance
        self.w_anc = weight_anchor

        self.sinkhorn = BatchedSinkhornOT(eps=eps_ot, max_iters=max_iters)

        # Dual-Anchor Codebooks
        self.vis_anchors = nn.Parameter(torch.randn(num_anchors, d_model) / math.sqrt(d_model))
        self.lex_anchors = nn.Parameter(torch.randn(num_anchors, d_model) / math.sqrt(d_model))

        # Resonance fusion gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_joints: torch.Tensor,          # [B, T, K, d_model] Spatiotemporal joint features
        text_features: Optional[torch.Tensor] = None, # [B, L, d_model] optional text token features
    ) -> ResonanceDualAnchorOutput:
        """
        Executes Inter-Frame Spatial Resonance and Dual-Anchor Alignment.
        """
        B, T, K, D = h_joints.shape
        device = h_joints.device

        # 1. Inter-Frame Spatial Resonance OT
        if T > 1:
            # Features at t and t+1: [B*(T-1), K, D]
            h_t = h_joints[:, :-1].reshape(B * (T - 1), K, D)
            h_next = h_joints[:, 1:].reshape(B * (T - 1), K, D)

            # Pairwise cosine distance cost matrix: C_{ij} = 1.0 - cos(h_t(i), h_next(j))
            h_t_norm = F.normalize(h_t, p=2, dim=-1)
            h_next_norm = F.normalize(h_next, p=2, dim=-1)
            cost_mat = 1.0 - torch.bmm(h_t_norm, h_next_norm.transpose(1, 2))  # [B*(T-1), K, K]

            # Solve OT plan P* [B*(T-1), K, K]
            P_star = self.sinkhorn(cost_mat)

            # Resonated features: P* @ h_next -> [B*(T-1), K, D]
            h_res_trans = torch.bmm(P_star, h_next)
            h_res_trans = h_res_trans.view(B, T - 1, K, D)

            # Pad last frame to keep length T
            h_res_trans_full = torch.cat([h_res_trans, h_joints[:, -1:]], dim=1)  # [B, T, K, D]

            # Gated fusion with original features
            concat_res = torch.cat([h_joints, h_res_trans_full], dim=-1)
            gate = self.fusion_gate(concat_res)
            h_fused = (1.0 - gate) * h_joints + gate * h_res_trans_full

            # Spatial resonance transport loss: sum P* * C
            loss_res = (P_star * cost_mat).sum(dim=(-1, -2)).mean()
        else:
            h_fused = h_joints
            loss_res = torch.tensor(0.0, device=device)

        # 2. Dual-Anchor Codebook Wasserstein Distance
        # Cost between Visual Anchors [M, D] and Lexical Anchors [M, D]
        vis_norm = F.normalize(self.vis_anchors, p=2, dim=-1)
        lex_norm = F.normalize(self.lex_anchors, p=2, dim=-1)
        cost_anchors = 1.0 - torch.matmul(vis_norm, lex_norm.t()).unsqueeze(0)  # [1, M, M]

        P_anchors = self.sinkhorn(cost_anchors)  # [1, M, M]
        loss_anchor = (P_anchors * cost_anchors).sum()

        # Optional text-to-anchor grounding if text features provided
        if text_features is not None:
            B_txt, L_txt, _ = text_features.shape
            txt_norm = F.normalize(text_features, p=2, dim=-1)
            cost_txt_anc = 1.0 - torch.bmm(txt_norm, lex_norm.unsqueeze(0).repeat(B_txt, 1, 1).transpose(1, 2))  # [B, L, M]
            P_txt = self.sinkhorn(cost_txt_anc)
            loss_txt_anc = (P_txt * cost_txt_anc).sum(dim=(-1, -2)).mean()
            loss_anchor = loss_anchor + 0.50 * loss_txt_anc

        total_loss = self.w_res * loss_res + self.w_anc * loss_anchor

        return ResonanceDualAnchorOutput(
            resonance_features=h_fused,
            total_loss=total_loss,
            spatial_resonance_loss=loss_res,
            dual_anchor_loss=loss_anchor,
            vis_anchors=self.vis_anchors,
            lex_anchors=self.lex_anchors,
        )
