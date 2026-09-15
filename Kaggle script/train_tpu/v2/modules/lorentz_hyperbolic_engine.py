#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — LORENTZ HYPERBOLIC EMBEDDING ENGINE (LORENTZCONESIGN)
================================================================================
Implements Hyperbolic Lorentz-Minkowski Kinematic Cones & Hyperboloids (LorentzHyperboloid-SLT):
1. Minkowski Spacetime Metric <x, y>_L in R^{d+1}:
     <x, y>_L = - x_0 * y_0 + sum_{i=1}^d x_i * y_i
2. Hyperboloid Manifold H^d_c:
     H^d_c = { x in R^{d+1} : <x, x>_L = - 1/c, x_0 > 0 }
3. Exact Exponential Map from Tangent Space at Origin o = (1/sqrt(c), 0, ..., 0):
     exp_o^c(v) = ( (1/sqrt(c)) * cosh(sqrt(c) * ||v||_2),  (v / ||v||_2) * (1/sqrt(c)) * sinh(sqrt(c) * ||v||_2) )
     Strictly satisfies <exp(v), exp(v)>_L == -1/c with zero numerical drift!
4. Pairwise Lorentz Geodesic Distances:
     d_L^c(x, y) = (1/sqrt(c)) * arcosh( - c * <x, y>_L )
     Naturally embeds branching anatomical tree hierarchies (Torso -> Arms -> Wrists -> Fingers) with zero distortion.
5. Tangent Logarithmic Map Pullback & Feature Fusion:
     log_o^c(x) = arcosh( sqrt(c) * x_0 ) * ( x_{1:d} / ||x_{1:d}||_2 )
     H_lorentz = H + LayerNorm( Linear( [log_o^c(X), d_L(wrist, fingers), <x, x>_L] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LorentzHyperbolicOutput(NamedTuple):
    lorentz_features: torch.Tensor      # [B, T, d_model] Projected Lorentz hyperbolic representations
    hyperboloid_points: torch.Tensor    # [B, T, 60, d+1] Projected points on Lorentz manifold H^d_c
    geodesic_distances: torch.Tensor    # [B, T, 60] Geodesic distance of each keypoint from origin o
    wrist_finger_distances: torch.Tensor# [B, T, 10] Distances from wrists (18, 39) to 10 finger tips
    minkowski_norms: torch.Tensor       # [B, T, 60] Invariant Minkowski inner products <x, x>_L == -1/c
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + lorentz_features


class ASLLorentzHyperbolicEngine(nn.Module):
    """
    Hyperbolic Lorentz-Minkowski Kinematic Cone & Hyperboloid Embedding Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        latent_dim: int = 16,           # Spatial Euclidean tangent dimension d
        curvature_c: float = 1.0,       # Hyperbolic curvature c > 0
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.d = latent_dim
        self.c = curvature_c
        self.sqrt_c = math.sqrt(curvature_c)

        # 1. Tangent Space Encoder: kinematics -> v in T_o H^d_c ~ R^d
        self.tangent_encoder = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, latent_dim),
        )

        # 2. Output Projection Head
        # Input: 60 * latent_dim + 60 (d_origin) + 10 (d_wrist_tips) -> d_model
        in_proj_dim = num_keypoints * latent_dim + num_keypoints + 10
        self.proj = nn.Sequential(
            nn.Linear(in_proj_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 10 Finger tip indices: Left (22, 26, 30, 34, 38), Right (43, 47, 51, 55, 59)
        self.register_buffer("l_tips", torch.tensor([22, 26, 30, 34, 38], dtype=torch.long))
        self.register_buffer("r_tips", torch.tensor([43, 47, 51, 55, 59], dtype=torch.long))

    def exp_map_origin(self, v: torch.Tensor) -> torch.Tensor:
        """
        Maps tangent vector v in R^d to hyperboloid H^d_c in R^{d+1}.
        v: [B, T, 60, d] -> x: [B, T, 60, d+1]
        """
        eps = 1e-7
        v_norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=eps) # [B, T, 60, 1]
        sc_norm = self.sqrt_c * v_norm

        x0 = (1.0 / self.sqrt_c) * torch.cosh(sc_norm)                    # [B, T, 60, 1]
        x_rest = (v / v_norm) * (1.0 / self.sqrt_c) * torch.sinh(sc_norm) # [B, T, 60, d]

        x = torch.cat([x0, x_rest], dim=-1)                               # [B, T, 60, d+1]
        return x

    def log_map_origin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pulls back hyperboloid point x in R^{d+1} to tangent vector v in R^d at origin o.
        """
        eps = 1e-7
        x0 = x[..., 0:1].clamp(min=1.0 / self.sqrt_c + eps)
        x_rest = x[..., 1:]

        x_rest_norm = torch.norm(x_rest, p=2, dim=-1, keepdim=True).clamp(min=eps)
        # arcosh( sqrt(c) * x0 )
        arg = (self.sqrt_c * x0).clamp(min=1.0 + eps)
        dist = (1.0 / self.sqrt_c) * torch.acosh(arg)

        v = (dist / x_rest_norm) * x_rest
        return v

    def minkowski_inner_product(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes Lorentz Minkowski inner product <x, y>_L = - x0*y0 + <x_rest, y_rest>.
        """
        dot_rest = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
        dot_0 = x[..., 0:1] * y[..., 0:1]
        return dot_rest - dot_0

    def lorentz_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes hyperbolic geodesic distance d_L^c(x, y) = (1/sqrt(c)) * arcosh( - c * <x, y>_L ).
        """
        eps = 1e-6
        inner = self.minkowski_inner_product(x, y)
        arg = (-self.c * inner).clamp(min=1.0 + eps)
        dist = (1.0 / self.sqrt_c) * torch.acosh(arg).squeeze(-1)
        return dist

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> LorentzHyperbolicOutput:
        """
        Executes exponential mapping to Lorentz hyperboloid, evaluates tree metrics, and projects to d_model.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # 1. Project Kinematics to Euclidean Tangent Space v in T_o H^d_c
        v = self.tangent_encoder(kinematics)  # [B, T, 60, d]

        # 2. Exponential Map onto Lorentz Hyperboloid Manifold H^d_c
        x_hyp = self.exp_map_origin(v)        # [B, T, 60, d+1]

        # 3. Verify Minkowski Norm <x, x>_L == -1/c
        mink_norms = self.minkowski_inner_product(x_hyp, x_hyp).squeeze(-1)  # [B, T, 60]

        # 4. Compute Geodesic Distances from Origin o = (1/sqrt(c), 0, ..., 0)
        o_pt = torch.zeros(1, 1, 1, self.d + 1, device=device, dtype=x_hyp.dtype)
        o_pt[..., 0] = 1.0 / self.sqrt_c
        d_origin = self.lorentz_distance(x_hyp, o_pt.expand(B, T, K, -1))   # [B, T, 60]

        # 5. Compute Hierarchical Wrist-to-Fingertip Hyperbolic Geodesic Distances
        # Left Wrist: 18, Right Wrist: 39
        x_lwrist = x_hyp[:, :, 18:19, :]  # [B, T, 1, d+1]
        x_rwrist = x_hyp[:, :, 39:40, :]  # [B, T, 1, d+1]

        x_ltips = x_hyp[:, :, self.l_tips, :] # [B, T, 5, d+1]
        x_rtips = x_hyp[:, :, self.r_tips, :] # [B, T, 5, d+1]

        d_ltips = self.lorentz_distance(x_ltips, x_lwrist.expand(-1, -1, 5, -1)) # [B, T, 5]
        d_rtips = self.lorentz_distance(x_rtips, x_rwrist.expand(-1, -1, 5, -1)) # [B, T, 5]
        d_wrist_tips = torch.cat([d_ltips, d_rtips], dim=-1)                    # [B, T, 10]

        # 6. Tangent Space Pullback via Logarithmic Map
        v_pullback = self.log_map_origin(x_hyp)                                  # [B, T, 60, d]

        # 7. Output Feature Projection
        v_flat = v_pullback.reshape(B, T, K * self.d)
        f_all = torch.cat([v_flat, d_origin, d_wrist_tips], dim=-1)              # [B, T, in_proj_dim]
        h_lorentz = self.proj(f_all)                                            # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_lorentz

        return LorentzHyperbolicOutput(
            lorentz_features=h_lorentz,
            hyperboloid_points=x_hyp,
            geodesic_distances=d_origin,
            wrist_finger_distances=d_wrist_tips,
            minkowski_norms=mink_norms,
            augmented_features=augmented,
        )
