#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — OPTIMAL TRANSPORT ALIGNMENT (OTA / DUALANCHOR) ENGINE
================================================================================
Implements Gloss-Free Cross-Modal Optimal Transport Alignment & Grounding (OTA/DualAnchor):
1. Pairwise Cross-Modal Cost Matrix:
     Computes cosine distance matrix C between visual kinematics (V) and text tokens (U):
     C_{t, l} = 1 - (v_t^T u_l) / (||v_t|| * ||u_l||)
2. Differentiable Sinkhorn-Knopp Solver:
     Solves entropy-regularized optimal transport problem in O(K * T * L) unrolled iterations:
     T* = argmin_{T in U(a, b)} <T, C> - epsilon * H(T)
     u^{(k+1)} = a / (K v^{(k)}),  v^{(k+1)} = b / (K^T u^{(k+1)}), where K = exp(-C / epsilon)
3. Lexical Fidelity & Optimal Transport Loss:
     L_ot = sum_{t, l} T*_{t, l} * C_{t, l}
4. Cross-Modal Mutual Information / Contrastive Grounding:
     L_ground = - (1/L) * sum_l log ( exp( <T*_l, V>^T u_l / tau ) / sum_{l'} exp( <T*_l, V>^T u_{l'} / tau ) )
5. Eliminates gloss dependence while maintaining fine-grained word-level visual grounding.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class OTAlignmentOutput(NamedTuple):
    ot_loss: torch.Tensor                # Scalar optimal transport distance
    grounding_loss: torch.Tensor         # Scalar cross-modal grounding loss
    total_loss: torch.Tensor             # Combined alignment objective
    transport_plan: torch.Tensor         # [B, T, L] Optimal coupling matrix T*
    cost_matrix: torch.Tensor            # [B, T, L] Pairwise cosine cost matrix


class OptimalTransportAlignmentEngine(nn.Module):
    """
    Entropy-regularized Optimal Transport Alignment for Gloss-Free Cross-Modal Grounding.
    """

    def __init__(
        self,
        d_vis: int = 128,
        d_text: int = 128,
        d_proj: int = 128,
        epsilon: float = 0.05,
        max_iter: int = 10,
        tau: float = 0.10,
        lambda_ot: float = 1.0,
        lambda_ground: float = 0.50,
    ):
        super().__init__()
        self.d_vis = d_vis
        self.d_text = d_text
        self.d_proj = d_proj
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tau = tau
        self.lambda_ot = lambda_ot
        self.lambda_ground = lambda_ground

        # Projection heads into shared semantic metric space
        self.vis_proj = nn.Sequential(
            nn.Linear(d_vis, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
        )

        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_proj),
            nn.LayerNorm(d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj),
        )

    def sinkhorn(
        self,
        cost: torch.Tensor,              # [B, T, L]
        a: torch.Tensor,                 # [B, T]
        b: torch.Tensor,                 # [B, L]
    ) -> torch.Tensor:
        """
        Differentiable batched Sinkhorn-Knopp solver for entropy-regularized OT.
        Returns coupling matrix T* of shape [B, T, L].
        """
        B, T, L = cost.shape
        device = cost.device
        eps = self.epsilon
        # Gibbs kernel K = exp(-C / eps) with log-sum-exp stabilization
        cost_min = cost.min(dim=-1, keepdim=True)[0]
        K = torch.exp(-(cost - cost_min) / eps)  # [B, T, L]

        u = torch.ones_like(a)  # [B, T]
        v = torch.ones_like(b)  # [B, L]

        for _ in range(self.max_iter):
            Kv = torch.bmm(K, v.unsqueeze(-1)).squeeze(-1)  # [B, T]
            u = a / (Kv.clamp(min=1e-8))

            Ktu = torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)  # [B, L]
            v = b / (Ktu.clamp(min=1e-8))

        # Coupling matrix T* = diag(u) @ K @ diag(v)
        T_star = u.unsqueeze(-1) * K * v.unsqueeze(-2)  # [B, T, L]
        return T_star

    def forward(
        self,
        vis_feats: torch.Tensor,         # [B, T, d_vis]
        text_feats: torch.Tensor,        # [B, L, d_text]
        vis_lengths: Optional[torch.Tensor] = None,   # [B]
        text_lengths: Optional[torch.Tensor] = None,  # [B]
    ) -> OTAlignmentOutput:
        """
        Computes Optimal Transport Alignment and Cross-Modal Grounding loss.
        """
        B, T, _ = vis_feats.shape
        _, L, _ = text_feats.shape
        device = vis_feats.device

        # 1. Project into normalized shared space
        z_v = F.normalize(self.vis_proj(vis_feats), p=2, dim=-1)   # [B, T, D]
        z_u = F.normalize(self.text_proj(text_feats), p=2, dim=-1)  # [B, L, D]

        # 2. Pairwise Cosine Cost Matrix: C_{t, l} = 1 - <z_v(t), z_u(l)>
        sim = torch.bmm(z_v, z_u.transpose(1, 2))  # [B, T, L] in [-1, 1]
        cost = 1.0 - sim                           # [B, T, L] in [0, 2]

        # 3. Form Marginal Distributions (a and b) with padding mask
        if vis_lengths is not None:
            mask_v = torch.arange(T, device=device).unsqueeze(0) < vis_lengths.unsqueeze(1)  # [B, T]
            a = mask_v.float() / mask_v.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            a = torch.full((B, T), 1.0 / float(T), device=device)

        if text_lengths is not None:
            mask_u = torch.arange(L, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)  # [B, L]
            b = mask_u.float() / mask_u.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            b = torch.full((B, L), 1.0 / float(L), device=device)

        # 4. Solve Optimal Transport Plan T*
        T_star = self.sinkhorn(cost, a, b)  # [B, T, L]

        # 5. Optimal Transport Loss: <T*, C>
        ot_loss = (T_star * cost).sum(dim=(-1, -2)).mean()

        # 6. Cross-Modal Grounding Loss:
        # For each text token l, the visual aligned context is c_l = sum_t T*_{t, l} * z_v(t)
        # We enforce c_l to be most similar to z_u(l) compared to all other text tokens in the sequence
        c_vis = torch.bmm(T_star.transpose(1, 2), z_v)  # [B, L, D]
        c_vis_norm = F.normalize(c_vis, p=2, dim=-1)

        # Cross-token logits: [B, L, L]
        ground_logits = torch.bmm(c_vis_norm, z_u.transpose(1, 2)) / self.tau
        ground_labels = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]

        if text_lengths is not None:
            ground_loss = F.cross_entropy(
                ground_logits.view(B * L, L),
                ground_labels.reshape(B * L),
                reduction="none"
            ).view(B, L)
            ground_loss = (ground_loss * mask_u.float()).sum() / mask_u.float().sum().clamp(min=1.0)
        else:
            ground_loss = F.cross_entropy(
                ground_logits.view(B * L, L),
                ground_labels.reshape(B * L)
            )

        # 7. Total Combined Alignment Loss
        total_loss = self.lambda_ot * ot_loss + self.lambda_ground * ground_loss

        return OTAlignmentOutput(
            ot_loss=ot_loss,
            grounding_loss=ground_loss,
            total_loss=total_loss,
            transport_plan=T_star,
            cost_matrix=cost,
        )
