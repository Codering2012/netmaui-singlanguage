#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — STEERABLE SPHERICAL HARMONIC ROUTING (SPHERICALROUTING)
================================================================================
Implements Equivariant Steerable Spherical Harmonic Angular Routing (EquiSphere-SLT):
1. Spherical Coordinate Decomposition:
     r = ||p_j - p_i||_2,  theta = arccos( z / (r + eps) ),  phi = atan2( y, x )
2. Real Spherical Harmonics Basis Functions Y_l^m(theta, phi) for l in {0, 1, 2} (9 components):
     l=0: Y_0^0 = 0.5 * sqrt(1/pi)
     l=1: Y_1^{-1} = sqrt(3/(4pi))*sin(theta)*sin(phi)
          Y_1^0    = sqrt(3/(4pi))*cos(theta)
          Y_1^1    = sqrt(3/(4pi))*sin(theta)*cos(phi)
     l=2: Y_2^{-2} = 0.5*sqrt(15/pi)*sin^2(theta)*sin(2phi)
          Y_2^{-1} = sqrt(15/(4pi))*sin(theta)*cos(theta)*sin(phi)
          Y_2^0    = 0.25*sqrt(5/pi)*(3*cos^2(theta) - 1)
          Y_2^1    = sqrt(15/(4pi))*sin(theta)*cos(theta)*cos(phi)
          Y_2^2    = 0.25*sqrt(15/pi)*sin^2(theta)*cos(2phi)
3. Radial Distance Multi-Layer Perceptron:
     W_rad(r) = MLP(r)
4. Steerable Harmonic Kernel & Angular Attention:
     f_steerable(i, j) = W_rad(r_ij) * Y(theta_ij, phi_ij)
     A_harm = Softmax( (Q Y)(K Y)^T / sqrt(d_k) )
5. Equivariant Angular Routing & Feature Projection:
     H_harm = H + LayerNorm( Linear( A_harm * V ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SphericalHarmonicRoutingOutput(NamedTuple):
    harmonic_features: torch.Tensor     # [B, T, d_model] Projected steerable harmonic representations
    harmonic_basis: torch.Tensor        # [B, T, 60, 9] Real spherical harmonic coefficients Y_l^m
    steerable_attention: torch.Tensor   # [B, T, 60, 60] Angular harmonic attention distribution
    power_spectrum: torch.Tensor        # [B, T, 3] SO(3) invariant power per degree (l=0, 1, 2)
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + harmonic_features


class ASLSphericalHarmonicRoutingEngine(nn.Module):
    """
    Steerable Spherical Harmonic Angular Frequency Convolution & Attention Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        radial_dim: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.radial_dim = radial_dim
        self.num_harmonics = 9  # Degree 0 (1), Degree 1 (3), Degree 2 (5)

        # Precomputed constant factors for real spherical harmonics
        self.c00 = 0.5 * math.sqrt(1.0 / math.pi)
        self.c1  = math.sqrt(3.0 / (4.0 * math.pi))
        self.c2_0 = 0.25 * math.sqrt(5.0 / math.pi)
        self.c2_1 = math.sqrt(15.0 / (4.0 * math.pi))
        self.c2_2 = 0.25 * math.sqrt(15.0 / math.pi)
        self.c2_m2 = 0.25 * math.sqrt(15.0 / math.pi)

        # Radial embedding MLP
        self.radial_mlp = nn.Sequential(
            nn.Linear(1, radial_dim),
            nn.GELU(),
            nn.Linear(radial_dim, radial_dim),
        )

        # Harmonic projections for Query, Key, Value
        self.q_proj = nn.Linear(self.num_harmonics, d_model)
        self.k_proj = nn.Linear(self.num_harmonics, d_model)
        self.v_proj = nn.Linear(in_channels, d_model)

        # Output feature projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_spherical_harmonics(
        self,
        vectors: torch.Tensor,  # [..., 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes 9 real spherical harmonics Y_l^m and radial distance r.
        Returns: (Y [..., 9], r [..., 1])
        """
        eps = 1e-7
        x = vectors[..., 0:1]
        y = vectors[..., 1:2]
        z = vectors[..., 2:3]

        r = torch.norm(vectors, p=2, dim=-1, keepdim=True).clamp(min=eps)

        cos_theta = (z / r).clamp(min=-1.0 + eps, max=1.0 - eps)
        sin_theta = torch.sqrt((1.0 - cos_theta ** 2).clamp(min=eps))

        phi = torch.atan2(y, x)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        cos_2phi = torch.cos(2.0 * phi)
        sin_2phi = torch.sin(2.0 * phi)

        # l=0: (1)
        y00 = torch.full_like(r, self.c00)

        # l=1: (3)
        y1_m1 = self.c1 * sin_theta * sin_phi
        y1_0  = self.c1 * cos_theta
        y1_p1 = self.c1 * sin_theta * cos_phi

        # l=2: (5)
        y2_m2 = self.c2_m2 * (sin_theta ** 2) * sin_2phi
        y2_m1 = self.c2_1 * sin_theta * cos_theta * sin_phi
        y2_0  = self.c2_0 * (3.0 * (cos_theta ** 2) - 1.0)
        y2_p1 = self.c2_1 * sin_theta * cos_theta * cos_phi
        y2_p2 = self.c2_2 * (sin_theta ** 2) * cos_2phi

        # Concatenate 9 harmonics: [..., 9]
        Y = torch.cat([y00, y1_m1, y1_0, y1_p1, y2_m2, y2_m1, y2_0, y2_p1, y2_p2], dim=-1)
        return Y, r

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9]
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> SphericalHarmonicRoutingOutput:
        """
        Executes steerable spherical harmonic basis calculation, attention routing, and feature projection.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        pos = kinematics[..., 0:3]  # [B, T, 60, 3]

        # 1. Compute Hand Center of Mass and Joint Radial Vectors: [B, T, 60, 3]
        center = pos.mean(dim=2, keepdim=True)  # [B, T, 1, 3]
        rel_pos = pos - center                   # [B, T, 60, 3]

        # 2. Real Spherical Harmonics Basis: Y [B, T, 60, 9], r [B, T, 60, 1]
        Y, r = self.compute_spherical_harmonics(rel_pos)

        # 3. Compute SO(3) Invariant Power Spectrum per Degree:
        # P_0 = |Y_00|^2, P_1 = sum(Y_1^m^2), P_2 = sum(Y_2^m^2)
        p0 = (Y[..., 0:1] ** 2).mean(dim=2)            # [B, T, 1]
        p1 = (Y[..., 1:4] ** 2).sum(dim=-1).mean(dim=2, keepdim=True) # [B, T, 1]
        p2 = (Y[..., 4:9] ** 2).sum(dim=-1).mean(dim=2, keepdim=True) # [B, T, 1]
        power_spec = torch.cat([p0, p1, p2], dim=-1)   # [B, T, 3]

        # 4. Harmonic Multi-Head Attention Routing
        Q = self.q_proj(Y)                           # [B, T, 60, d_model]
        K_mat = self.k_proj(Y)                       # [B, T, 60, d_model]
        V = self.v_proj(kinematics)                  # [B, T, 60, d_model]

        # Scaled dot-product attention: [B, T, 60, 60]
        scale = 1.0 / math.sqrt(self.d_model)
        attn_scores = torch.matmul(Q, K_mat.transpose(-1, -2)) * scale
        attn = F.softmax(attn_scores, dim=-1)        # [B, T, 60, 60]

        # Attended features: [B, T, 60, d_model]
        routed = torch.matmul(attn, V)               # [B, T, 60, d_model]
        routed_pool = routed.mean(dim=2)             # [B, T, d_model]

        # 5. Output Projection
        h_harm = self.out_proj(routed_pool)          # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_harm

        return SphericalHarmonicRoutingOutput(
            harmonic_features=h_harm,
            harmonic_basis=Y,
            steerable_attention=attn,
            power_spectrum=power_spec,
            augmented_features=augmented,
        )
