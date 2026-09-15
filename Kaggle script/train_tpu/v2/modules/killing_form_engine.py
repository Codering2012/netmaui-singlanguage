#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — LIE-ALGEBRAIC KILLING FORM ENGINE (KILLINGFORMSIGN)
================================================================================
Implements Lie-Algebraic Killing Form Metric & Casimir Invariants (Killing-SLT):
1. Lie Algebra so(3) Generator Construction:
     Given 3D angular velocity omega = (w_x, w_y, w_z), forms skew-symmetric matrix:
     X = [[   0, -w_z,  w_y],
          [ w_z,    0, -w_x],
          [-w_y,  w_x,    0]] in so(3)
2. Exact Killing Bilinear Form B(X, Y):
     B(X, Y) = Tr( ad_X o ad_Y ) = 2 * Tr( X @ Y ) = -4 * <w_X, w_Y>
     Strictly Ad-invariant under SO(3) rotations: B(Ad_g(X), Ad_g(Y)) == B(X, Y).
3. Quadratic Casimir Invariant & Killing Energy:
     C_2(X) = -0.5 * Tr( X @ X ) = ||w_X||_2^2
     E_killing = -0.25 * B(X, X) = ||w_X||_2^2
4. Pairwise Keypoint Killing Gram Matrix G_killing in R^{K x K}:
     G_{ij} = B(X_i, X_j) / 4 = - <w_i, w_j>
5. Feature Projection & Canonical Fusion:
     H_killing = H + LayerNorm( Linear( [G_diag, C_2, G_summary] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class KillingFormOutput(NamedTuple):
    killing_features: torch.Tensor      # [B, T, d_model] Projected Killing form representations
    killing_energy: torch.Tensor        # [B, T, 60] Quadratic Casimir / Killing energy per joint
    killing_gram_matrix: torch.Tensor   # [B, T, 60, 60] Pairwise Killing form metric matrix
    casimir_invariants: torch.Tensor    # [B, T, 60] Second-order Casimir invariants
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + killing_features


class ASLKillingFormEngine(nn.Module):
    """
    Lie-Algebraic Killing Form Metric & Casimir Invariant Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints

        # Input dimension to projection: 60 (E_killing) + 60 (C_2) + 60 (Gram mean) = 180
        in_feat_dim = num_keypoints * 3
        self.proj = nn.Sequential(
            nn.Linear(in_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def hat_map(self, w: torch.Tensor) -> torch.Tensor:
        """
        Maps 3D vector w = (w_x, w_y, w_z) to skew-symmetric 3x3 matrix in so(3).
        w: [B, T, 60, 3] -> X: [B, T, 60, 3, 3]
        """
        B, T, K, _ = w.shape
        wx = w[..., 0]
        wy = w[..., 1]
        wz = w[..., 2]
        zeros = torch.zeros_like(wx)

        row0 = torch.stack([zeros, -wz, wy], dim=-1)
        row1 = torch.stack([wz, zeros, -wx], dim=-1)
        row2 = torch.stack([-wy, wx, zeros], dim=-1)

        X = torch.stack([row0, row1, row2], dim=-2)  # [B, T, 60, 3, 3]
        return X

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> KillingFormOutput:
        """
        Computes so(3) Lie algebra generators, Killing bilinear form, Casimir invariants, and projects to d_model.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        pos = kinematics[..., 0:3]  # [B, T, 60, 3]
        if C >= 6:
            vel = kinematics[..., 3:6]
        else:
            vel = torch.zeros_like(pos)
            if T > 1:
                vel[:, :-1, :, :] = pos[:, 1:, :, :] - pos[:, :-1, :, :]
                vel[:, -1, :, :] = vel[:, -2, :, :]

        # 1. Angular velocity proxy w = pos x vel (orbital angular momentum generator)
        w = torch.cross(pos, vel, dim=-1)  # [B, T, 60, 3]

        # 2. Skew-symmetric so(3) matrices: [B, T, 60, 3, 3]
        X = self.hat_map(w)

        # 3. Quadratic Casimir Invariant & Killing Kinetic Energy
        # C_2(X) = -0.5 * Tr(X @ X) = ||w||_2^2
        # E_killing = -0.25 * B(X, X) = ||w||_2^2
        E_killing = (w ** 2).sum(dim=-1)  # [B, T, 60]
        casimir = E_killing.clone()       # [B, T, 60]

        # 4. Pairwise Keypoint Killing Gram Matrix G_{ij} = - <w_i, w_j> = 0.5 * Tr(X_i @ X_j)
        # w: [B, T, 60, 3] -> Gram: w @ w^T: [B, T, 60, 60]
        G_killing = torch.matmul(w, w.transpose(-1, -2))  # [B, T, 60, 60]
        G_summary = G_killing.mean(dim=-1)                 # [B, T, 60]

        # 5. Feature Projection
        f_all = torch.cat([E_killing, casimir, G_summary], dim=-1)  # [B, T, 180]
        h_killing = self.proj(f_all)                                # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_killing

        return KillingFormOutput(
            killing_features=h_killing,
            killing_energy=E_killing,
            killing_gram_matrix=G_killing,
            casimir_invariants=casimir,
            augmented_features=augmented,
        )
