#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — MEDIAL AXIS SKELETON & TOPOLOGICAL FLOW (MEDIALAXISSIGN)
================================================================================
Implements Medial Axis Transform (MAT) & Topological Centerline Flow (MAT-SLT):
1. Medial Centerline Geodesic Sampling:
     m_k(s) = (1 - s) * p_{root} + s * p_{tip},   s in {0.25, 0.50, 0.75, 1.0}
2. Differentiable Inscribed Sphere Radius Field:
     R_k(s) = min_{j not in bone} ||m_k(s) - p_j||_2
     Measures volumetric hand thickness and inter-finger proximity.
3. Medial Tangent Flow & Thickness Gradient:
     T_k = d m_k / ds = (p_{tip} - p_{root}) / ||p_{tip} - p_{root}||
     G_k = d R_k / ds (spatial thickness variation along bone axis)
4. Homotopy-Invariant Feature Projection:
     H_medial = H + LayerNorm(Linear([m_centers, R_radii, T_tangents, G_gradients]))
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MedialAxisOutput(NamedTuple):
    medial_features: torch.Tensor       # [B, T, d_model] Projected topological medial features
    centerline_points: torch.Tensor     # [B, T, num_bones, S, 3] Sampled medial axis coordinates
    inscribed_radii: torch.Tensor       # [B, T, num_bones, S] Inscribed radius field R_k(s)
    tangent_flow: torch.Tensor          # [B, T, num_bones, 3] Unit bone tangent vector T_k
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + medial_features


class ASLMedialAxisSkeletonEngine(nn.Module):
    """
    Medial Axis Transform (MAT) Topological Skeletonization Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        samples_per_bone: int = 4,      # S sampling points along each bone
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.S = samples_per_bone

        # 10 Canonical Bone Chains: (Root Joint, Tip Joint)
        # Left Hand: Wrist 18 -> Thumb 19, Index 23, Middle 27, Ring 31, Pinky 35
        # Right Hand: Wrist 39 -> Thumb 40, Index 44, Middle 48, Ring 52, Pinky 56
        self.bone_pairs = [
            (18, 19), (18, 23), (18, 27), (18, 31), (18, 35),
            (39, 40), (39, 44), (39, 48), (39, 52), (39, 56),
        ]
        self.num_bones = len(self.bone_pairs)

        # Sampling locations s in (0, 1]
        s_steps = torch.linspace(0.25, 1.0, samples_per_bone, dtype=torch.float32)
        self.register_buffer("s_steps", s_steps)

        # Feature dimension per bone: S * 3 (coords) + S * 1 (radii) + 3 (tangent) + S (gradient)
        # = S * 5 + 3 = 4 * 5 + 3 = 23 per bone -> 10 * 23 = 230
        in_dim = self.num_bones * (self.S * 4 + 3 + (self.S - 1))
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_medial_axis(
        self,
        pos: torch.Tensor,  # [B, T, 60, 3] (Cartesian coordinates)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes medial centerline points, inscribed radius field, and tangent flows.
        Returns: (m_centers [B,T,10,S,3], radii [B,T,10,S], tangents [B,T,10,3], rad_grad [B,T,10,S-1])
        """
        B, T, K, _ = pos.shape
        device = pos.device
        eps = 1e-6

        all_centers = []
        all_radii = []
        all_tangents = []
        all_rad_grads = []

        for root_idx, tip_idx in self.bone_pairs:
            p_root = pos[..., root_idx, :]  # [B, T, 3]
            p_tip  = pos[..., tip_idx, :]   # [B, T, 3]

            diff = p_tip - p_root           # [B, T, 3]
            bone_len = torch.norm(diff, p=2, dim=-1, keepdim=True) + eps  # [B, T, 1]
            tangent = diff / bone_len       # [B, T, 3]
            all_tangents.append(tangent)

            # Sample S points along the centerline: m(s) = (1 - s) * p_root + s * p_tip
            # s_steps: [S] -> [1, 1, S, 1]
            s_exp = self.s_steps.view(1, 1, self.S, 1)
            # m_pts: [B, T, S, 3]
            m_pts = (1.0 - s_exp) * p_root.unsqueeze(-2) + s_exp * p_tip.unsqueeze(-2)
            all_centers.append(m_pts)

            # Inscribed Sphere Radius R(s): min distance from m(s) to other non-bone keypoints
            # Exclude root and tip by taking minimum over all other joints
            # pos: [B, T, 1, K, 3], m_pts: [B, T, S, 1, 3]
            dist_to_all = torch.norm(m_pts.unsqueeze(-2) - pos.unsqueeze(-3), p=2, dim=-1)  # [B, T, S, K]
            # Zero-out root and tip distance to avoid self-radius = 0
            mask = torch.ones(K, device=device, dtype=torch.bool)
            mask[root_idx] = False
            mask[tip_idx] = False
            r_s = dist_to_all[..., mask].min(dim=-1).values  # [B, T, S]
            all_radii.append(r_s)

            # Radius gradient: d R / ds = R(s_{i+1}) - R(s_i)
            rad_grad = r_s[..., 1:] - r_s[..., :-1]  # [B, T, S-1]
            all_rad_grads.append(rad_grad)

        m_centers = torch.stack(all_centers, dim=2)   # [B, T, num_bones, S, 3]
        radii = torch.stack(all_radii, dim=2)         # [B, T, num_bones, S]
        tangents = torch.stack(all_tangents, dim=2)   # [B, T, num_bones, 3]
        rad_grads = torch.stack(all_rad_grads, dim=2) # [B, T, num_bones, S-1]

        return m_centers, radii, tangents, rad_grads

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords at 0:3)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional representations
    ) -> MedialAxisOutput:
        """
        Executes Medial Axis Transform extraction, radius field computation, and topological feature projection.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        pos = kinematics[..., :3]  # [B, T, 60, 3]

        # 1. Compute Medial Axis Centers, Radii, Tangents, and Thickness Gradients
        m_centers, radii, tangents, rad_grads = self.compute_medial_axis(pos)

        # 2. Flatten and Concatenate Topological Descriptors
        # m_centers: [B, T, 10, S*3], radii: [B, T, 10, S], tangents: [B, T, 10, 3], rad_grads: [B, T, 10, S-1]
        f_centers = m_centers.view(B, T, self.num_bones, self.S * 3)
        f_radii   = radii.view(B, T, self.num_bones, self.S)
        f_tangents= tangents.view(B, T, self.num_bones, 3)
        f_grads   = rad_grads.view(B, T, self.num_bones, self.S - 1)

        f_bone_all = torch.cat([f_centers, f_radii, f_tangents, f_grads], dim=-1)  # [B, T, 10, bone_dim]
        f_flat = f_bone_all.view(B, T, -1)  # [B, T, total_dim]

        # 3. Project to Model Feature Dimension
        medial_emb = self.proj(f_flat)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + medial_emb

        return MedialAxisOutput(
            medial_features=medial_emb,
            centerline_points=m_centers,
            inscribed_radii=radii,
            tangent_flow=tangents,
            augmented_features=augmented,
        )
