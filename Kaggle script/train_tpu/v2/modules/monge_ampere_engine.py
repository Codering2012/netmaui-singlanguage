#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — OPTIMAL TRANSPORT MONGE-AMPÈRE ENGINE (MONGEAMPERESIGN)
================================================================================
Implements Continuous Optimal Transport Monge-Ampère & Least Action Velocity Field (Monge-SLT):
1. Brenier's Theorem & Convex Potential:
     Transport map T(x) = grad_x psi(x), where psi is strictly convex.
     psi(x) = 0.5 * ||x||_2^2 + f_convex(x)
2. Least-Action Optimal Velocity Field:
     v_transport(x) = grad_x psi(x) - x
3. Hessian Divergence / Monge-Ampère Curvature:
     Delta psi = Tr( grad_x^2 psi(x) ) = div( T(x) )
4. Dynamic Benamou-Brenier Least-Action Loss:
     L_Monge = 0.5 * ||grad_x psi(P_t) - P_{t+1}||_F^2 + lambda * ||Delta psi - 3.0||^2
5. Feature Projection & Canonical Fusion:
     H_monge = H + LayerNorm( Linear( [v_transport, psi(x), Delta_psi] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MongeAmpereOutput(NamedTuple):
    monge_features: torch.Tensor        # [B, T, d_model] Projected Monge transport representations
    transport_velocity: torch.Tensor    # [B, T, 60, 3] Optimal transport velocity field v = grad(psi) - x
    monge_potential: torch.Tensor       # [B, T, 60, 1] Scalar convex potential values psi(x)
    hessian_trace: torch.Tensor         # [B, T, 60, 1] Monge-Ampère Laplacian divergence Tr(grad^2 psi)
    least_action_loss: torch.Tensor     # [1] Dynamic Benamou-Brenier least action kinetic loss
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + monge_features


class ASLMongeAmpereEngine(nn.Module):
    """
    Optimal Transport Monge-Ampère & Least-Action Kinetic Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        hidden_dim: int = 64,
        curv_reg: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.hidden_dim = hidden_dim
        self.curv_reg = curv_reg

        # Smooth scalar convex potential network f_convex: R^3 -> R
        self.fc1 = nn.Linear(3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1, bias=False)

        # Output feature projection: [v_transport (3) + psi (1) + div (1)] * 60 = 300 -> d_model
        in_feat_dim = num_keypoints * (3 + 1 + 1)
        self.out_proj = nn.Sequential(
            nn.Linear(in_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_potential(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Evaluates scalar convex potential: psi(x) = 0.5 * ||x||^2 + Softplus(fc3(GELU(fc2(GELU(fc1(x))))))
        pos: [..., 3]
        Returns: [..., 1]
        """
        base_quad = 0.5 * (pos ** 2).sum(dim=-1, keepdim=True)  # [..., 1]
        h = F.gelu(self.fc1(pos))
        h = F.gelu(self.fc2(h))
        # Use Softplus activation on output to enforce non-negative convex contribution
        nonlin = F.softplus(self.fc3(h))  # [..., 1]
        return base_quad + nonlin

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] or [B, T, 60, 3]
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> MongeAmpereOutput:
        """
        Computes Monge-Ampère potential, analytical transport gradient, divergence, and least action loss.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        pos = kinematics[..., 0:3]  # [B, T, 60, 3]

        # Enable autograd for potential gradient computation
        pos_var = pos.detach().requires_grad_(True)
        psi = self.compute_potential(pos_var)  # [B, T, 60, 1]

        # 1. Optimal Transport Velocity Field: T(x) = grad_x psi(x)
        grad_psi = torch.autograd.grad(
            outputs=psi.sum(),
            inputs=pos_var,
            create_graph=True,
            retain_graph=True,
        )[0]  # [B, T, 60, 3]

        v_trans = grad_psi - pos_var  # [B, T, 60, 3]

        # 2. Monge-Ampère Laplacian Divergence: div(T) = Tr(grad^2 psi)
        # Approximate divergence via directional gradient probe
        v1 = grad_psi[..., 0:1]
        v2 = grad_psi[..., 1:2]
        v3 = grad_psi[..., 2:3]

        d1 = torch.autograd.grad(v1.sum(), pos_var, create_graph=True, retain_graph=True)[0][..., 0:1]
        d2 = torch.autograd.grad(v2.sum(), pos_var, create_graph=True, retain_graph=True)[0][..., 1:2]
        d3 = torch.autograd.grad(v3.sum(), pos_var, create_graph=True, retain_graph=True)[0][..., 2:3]
        laplacian_psi = d1 + d2 + d3  # [B, T, 60, 1] (Nominal value ~ 3.0 for identity transport)

        # 3. Dynamic Benamou-Brenier Least-Action Loss across Consecutive Frames
        if T > 1:
            target_next = pos[:, 1:, :, :]  # [B, T-1, 60, 3]
            pred_next = grad_psi[:, :-1, :, :] # [B, T-1, 60, 3]
            action_loss = 0.5 * F.mse_loss(pred_next, target_next)
        else:
            action_loss = 0.5 * (v_trans ** 2).mean()

        curv_loss = F.mse_loss(laplacian_psi, torch.full_like(laplacian_psi, 3.0))
        total_loss = action_loss + self.curv_reg * curv_loss

        # 4. Feature Projection
        # [B, T, 60, 5] -> [B, T, 300]
        node_feats = torch.cat([v_trans, psi, laplacian_psi], dim=-1).reshape(B, T, K * 5)
        h_monge = self.out_proj(node_feats)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_monge

        return MongeAmpereOutput(
            monge_features=h_monge,
            transport_velocity=v_trans,
            monge_potential=psi,
            hessian_trace=laplacian_psi,
            least_action_loss=total_loss,
            augmented_features=augmented,
        )
