#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SPHERICAL HARMONIC POSE & ORIENTATION ENGINE (SPHERICALSIGN)
================================================================================
Implements Local Real Spherical Harmonics Expansion on Unit Sphere S^2 (SO3-SLT):
1. Unit Direction Vectors for 10 Canonical Bone Chains:
     u_bone = (p_tip - p_wrist) / (||p_tip - p_wrist||_2 + eps) in S^2
2. Real Spherical Harmonic Basis Functions (Degree l <= 2, 9 Channels per Bone):
     l = 0 (Monopole, 1 chan): Y_0^0 = 0.5 * sqrt(1/pi)
     l = 1 (Dipoles,  3 chan): Y_1^{-1} = c1 * y, Y_1^0 = c1 * z, Y_1^1 = c1 * x
     l = 2 (Quadrupoles, 5 chan):
         Y_2^{-2} = c2 * x * y, Y_2^{-1} = c2 * y * z, Y_2^0 = c3 * (2*z^2 - x^2 - y^2)
         Y_2^1 = c2 * x * z,    Y_2^2 = 0.5 * c2 * (x^2 - y^2)
3. SO(3) Equivariance & Tangent Projection:
     Transforms under Wigner D-matrices D^{(l)}(R) under 3D camera/subject rotations.
4. Multi-Bone Spherical Spectrum Feature Injection:
     H_spherical = H + LayerNorm(Linear(Y_{0..2}(u_bones)))
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalHarmonicsOutput(NamedTuple):
    spherical_features: torch.Tensor     # [B, T, d_model] Projected spherical harmonic features
    harmonic_coefficients: torch.Tensor  # [B, T, num_bones, 9] Real SH coefficients per bone
    bone_directions: torch.Tensor        # [B, T, num_bones, 3] Unit directional vectors on S^2
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + spherical_features


class ASLSphericalHarmonicsEngine(nn.Module):
    """
    Local Spherical Harmonics Pose & Orientation Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        max_degree: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.max_degree = max_degree

        # 10 Canonical Bone Chains: (Root Joint, Tip Joint)
        # Left Hand: Wrist 18 -> Thumb 19, Index 23, Middle 27, Ring 31, Pinky 35
        # Right Hand: Wrist 39 -> Thumb 40, Index 44, Middle 48, Ring 52, Pinky 56
        self.bone_pairs = [
            (18, 19), (18, 23), (18, 27), (18, 31), (18, 35),
            (39, 40), (39, 44), (39, 48), (39, 52), (39, 56),
        ]
        self.num_bones = len(self.bone_pairs)
        self.num_harmonics = 9  # (2*0+1) + (2*1+1) + (2*2+1) = 1 + 3 + 5 = 9

        # Spherical Normalization Constants
        self.c0 = 0.5 * math.sqrt(1.0 / math.pi)                    # 0.28209479
        self.c1 = math.sqrt(3.0 / (4.0 * math.pi))                  # 0.48860251
        self.c2 = 0.5 * math.sqrt(15.0 / math.pi)                   # 1.09254843
        self.c3 = 0.25 * math.sqrt(5.0 / math.pi)                   # 0.31539156

        # Projection head: [num_bones * 9] -> [d_model]
        in_dim = self.num_bones * self.num_harmonics  # 10 * 9 = 90
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_spherical_harmonics(
        self,
        u: torch.Tensor,  # [..., 3] Unit directional vectors on S^2 (x, y, z)
    ) -> torch.Tensor:
        """
        Evaluates 9 Real Spherical Harmonics up to degree l=2 on unit sphere S^2.
        Returns: [..., 9]
        """
        x = u[..., 0]
        y = u[..., 1]
        z = u[..., 2]

        # Degree l = 0 (1 channel)
        y0_0 = torch.full_like(x, self.c0)

        # Degree l = 1 (3 channels)
        y1_m1 = self.c1 * y
        y1_0  = self.c1 * z
        y1_p1 = self.c1 * x

        # Degree l = 2 (5 channels)
        y2_m2 = self.c2 * x * y
        y2_m1 = self.c2 * y * z
        y2_0  = self.c3 * (2.0 * z.pow(2) - x.pow(2) - y.pow(2))
        y2_p1 = self.c2 * x * z
        y2_p2 = 0.5 * self.c2 * (x.pow(2) - y.pow(2))

        # Stack into [..., 9]
        sh = torch.stack([
            y0_0,
            y1_m1, y1_0, y1_p1,
            y2_m2, y2_m1, y2_0, y2_p1, y2_p2
        ], dim=-1)

        return sh

    def extract_bone_directions(
        self,
        kinematics: torch.Tensor,  # [B, T, 60, 9] (coords at 0:3)
    ) -> torch.Tensor:
        """
        Computes unit direction vectors for canonical bone chains.
        Returns: [B, T, num_bones, 3]
        """
        pos = kinematics[..., :3]  # [B, T, 60, 3]
        eps = 1e-6

        bone_dirs = []
        for root_idx, tip_idx in self.bone_pairs:
            p_root = pos[..., root_idx, :]  # [B, T, 3]
            p_tip  = pos[..., tip_idx, :]   # [B, T, 3]
            diff = p_tip - p_root           # [B, T, 3]
            norm = torch.norm(diff, p=2, dim=-1, keepdim=True).clamp(min=eps)
            u = diff / norm                 # [B, T, 3] on S^2
            bone_dirs.append(u)

        bone_dirs_tensor = torch.stack(bone_dirs, dim=-2)  # [B, T, num_bones, 3]
        return bone_dirs_tensor

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> SphericalHarmonicsOutput:
        """
        Executes unit bone extraction, Real SH evaluation, and feature projection.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # 1. Extract Unit Bone Directions on S^2: [B, T, num_bones, 3]
        bone_dirs = self.extract_bone_directions(kinematics)

        # 2. Evaluate Real Spherical Harmonics: [B, T, num_bones, 9]
        sh_coeffs = self.compute_spherical_harmonics(bone_dirs)

        # 3. Project Flat Harmonic Spectrum to Model Dimension
        flat_sh = sh_coeffs.view(B, T, self.num_bones * self.num_harmonics)  # [B, T, 90]
        spherical_emb = self.proj(flat_sh)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + spherical_emb

        return SphericalHarmonicsOutput(
            spherical_features=spherical_emb,
            harmonic_coefficients=sh_coeffs,
            bone_directions=bone_dirs,
            augmented_features=augmented,
        )
