#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — PHASE SPACE RECONSTRUCTION & LYAPUNOV DYNAMICS (CHAOSSIGN)
================================================================================
Implements Non-Linear Dynamical Phase Space Embedding & Local Lyapunov Divergence:
1. Takens' Time-Delay Phase Space Embedding:
     Maps 1D/3D kinematic channels into m-dimensional attractor manifolds:
     y(t) = [x(t), x(t - tau), x(t - 2*tau), ..., x(t - (m-1)*tau)] in R^{m * C}
2. Local Lyapunov Exponent (LLE) Divergence:
     lambda(t) = (1 / dt) * ln ( ||y_i(t + dt) - y_j(t + dt)||_2 / (||y_i(t) - y_j(t)||_2 + eps) )
     - lambda < 0: Stable attractor hold / linguistic pause
     - lambda ~ 0: Periodic limit cycle gesture
     - lambda > 0: High-energy ballistic transition stroke / inflection point
3. Dynamic Attractor Feature Injection:
     H_phase = H + LayerNorm(Linear([y(t), lambda(t)]))
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseSpaceOutput(NamedTuple):
    phase_space_embeddings: torch.Tensor # [B, T, d_model] Projected phase space features
    reconstructed_attractor: torch.Tensor# [B, T, K, m * C] Takens phase space coordinates
    local_lyapunov_divergence: torch.Tensor # [B, T, 1] Local trajectory divergence rate
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + phase_emb
    stability_mask: torch.Tensor         # [B, T] Boolean mask (True = stable pause/hold)


class ASLPhaseSpaceDynamicsEngine(nn.Module):
    """
    Non-Linear Dynamical Phase Space Reconstruction & Local Lyapunov Dynamics Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        embedding_dim: int = 3,          # Takens embedding dimension m
        delay_tau: int = 2,              # Time delay stride tau
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.m = embedding_dim
        self.tau = delay_tau

        # Projection head: [num_keypoints * m * in_channels + 1] -> d_model (pooled keypoints: m * in_channels + 1)
        in_dim = self.m * in_channels + 1
        self.phase_proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def reconstruct_phase_space(self, kinematics: torch.Tensor) -> torch.Tensor:
        """
        Constructs Takens delay embedding: y(t) = [x(t), x(t-tau), ..., x(t-(m-1)tau)].
        kinematics: [B, T, K, C]
        Returns: [B, T, K, m * C]
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        delay_vectors = []
        for k in range(self.m):
            shift = k * self.tau
            if shift == 0:
                delayed = kinematics
            else:
                # Left-pad sequence with first frame
                pad = kinematics[:, :1].repeat(1, shift, 1, 1)
                delayed = torch.cat([pad, kinematics[:, :-shift]], dim=1)
            delay_vectors.append(delayed)

        # Concatenate along channel dimension: [B, T, K, m * C]
        attractor = torch.cat(delay_vectors, dim=-1)
        return attractor

    def compute_local_lyapunov(self, attractor: torch.Tensor) -> torch.Tensor:
        """
        Computes local trajectory divergence lambda(t) = ln(||d_{t+1}|| / (||d_t|| + eps)).
        attractor: [B, T, K, m*C]
        Returns: [B, T, 1]
        """
        B, T, K, D_att = attractor.shape
        device = attractor.device

        # Keypoint-pooled attractor: [B, T, D_att]
        y = attractor.mean(dim=2)  # [B, T, D_att]

        # Velocity in phase space: d_t = y_{t} - y_{t-1}
        pad = y[:, :1]
        y_prev = torch.cat([pad, y[:, :-1]], dim=1)  # [B, T, D_att]
        dist_t = torch.norm(y - y_prev, p=2, dim=-1, keepdim=True)  # [B, T, 1]

        # Next step velocity: d_{t+1}
        pad_next = y[:, -1:]
        y_next = torch.cat([y[:, 1:], pad_next], dim=1)  # [B, T, D_att]
        dist_t_plus_1 = torch.norm(y_next - y, p=2, dim=-1, keepdim=True)  # [B, T, 1]

        # Local Lyapunov exponent divergence rate: ln(d_{t+1} / (d_t + eps))
        eps = 1e-6
        lle = torch.log((dist_t_plus_1 + eps) / (dist_t + eps)).clamp(-5.0, 5.0)  # [B, T, 1]
        return lle

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] Kinematic inputs
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> PhaseSpaceOutput:
        """
        Executes Takens phase space embedding, Lyapunov divergence, and feature injection.
        """
        B, T, K, C = kinematics.shape

        # 1. Takens Phase Space Attractor Embedding
        attractor = self.reconstruct_phase_space(kinematics)  # [B, T, K, m * C]

        # 2. Local Lyapunov Divergence Rate
        lle = self.compute_local_lyapunov(attractor)  # [B, T, 1]

        # 3. Stability Mask (lambda < 0 -> stable pause / hold)
        stability_mask = (lle.squeeze(-1) < 0.0)  # [B, T]

        # 4. Projected Phase Space Features
        # Mean pool attractor over keypoints: [B, T, m * C]
        attractor_pooled = attractor.mean(dim=2)
        phase_in = torch.cat([attractor_pooled, lle], dim=-1)  # [B, T, m * C + 1]
        phase_emb = self.phase_proj(phase_in)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + phase_emb

        return PhaseSpaceOutput(
            phase_space_embeddings=phase_emb,
            reconstructed_attractor=attractor,
            local_lyapunov_divergence=lle,
            augmented_features=augmented,
            stability_mask=stability_mask,
        )
