#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — RIEMANNIAN HOLONOMY ENGINE (HOLONOMYCURVATURESIGN)
================================================================================
Implements Riemannian Levi-Civita Parallel Transport & Holonomy Defect (Holonomy-SLT):
1. Unit Sphere Directional Projection:
     u_t = (p_t - c_torso) / ||p_t - c_torso||_2 in S^2
2. Levi-Civita Geodesic Parallel Transport on S^2:
     Gamma_{u_t -> u_{t+1}}( v ) = v - ( (u_t + u_{t+1}) / (1 + u_t . u_{t+1}) ) * (u_{t+1} . v)
     Strictly preserves tangent vector norm: ||Gamma(v)||_2 == ||v||_2.
3. Ambrose-Singer Closed-Loop Holonomy Rotation R_hol in SO(3):
     R_hol = Gamma_{t+2 -> t} o Gamma_{t+1 -> t+2} o Gamma_{t -> t+1}
     Measures enclosed Gaussian curvature flux: Tr(R_hol) = 1 + 2 * cos(theta_hol).
4. Holonomy Solid Angle Defect theta_hol:
     theta_hol = arccos( clamp( (Tr(R_hol) - 1) / 2, -1, 1 ) )
5. Feature Projection & Canonical Fusion:
     H_holonomy = H + LayerNorm( Linear( [v_transported, theta_hol, Tr(R_hol)] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RiemannianHolonomyOutput(NamedTuple):
    holonomy_features: torch.Tensor     # [B, T, d_model] Projected holonomy representations
    transported_velocities: torch.Tensor# [B, T, 60, 3] Levi-Civita parallel-transported velocity field
    holonomy_angles: torch.Tensor       # [B, T, 60, 1] Closed-loop solid angle defect theta_hol in [0, pi]
    holonomy_traces: torch.Tensor       # [B, T, 60, 1] Trace of holonomy rotation matrix Tr(R_hol)
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + holonomy_features


class ASLRiemannianHolonomyEngine(nn.Module):
    """
    Riemannian Levi-Civita Parallel Transport & Holonomy Curvature Loop Engine.
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

        # Input dimension: 60 * (3 (v_transported) + 1 (theta_hol) + 1 (Tr(R))) = 300
        in_feat_dim = num_keypoints * 5
        self.proj = nn.Sequential(
            nn.Linear(in_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def parallel_transport_s2(
        self,
        u_from: torch.Tensor,  # [B, T, 60, 3] (Unit vectors)
        u_to: torch.Tensor,    # [B, T, 60, 3] (Unit vectors)
        v: torch.Tensor,       # [B, T, 60, 3] (Tangent vectors)
    ) -> torch.Tensor:
        """
        Executes exact Levi-Civita parallel transport along the minimal geodesic on S^2:
        Gamma(v) = v - ((u_from + u_to) / (1 + u_from . u_to)) * (u_to . v)
        """
        eps = 1e-6
        dot_u = (u_from * u_to).sum(dim=-1, keepdim=True).clamp(min=-1.0 + eps, max=1.0)
        denom = 1.0 + dot_u  # [B, T, 60, 1]

        # Ensure strict tangency at u_from
        v_tan = v - (v * u_from).sum(dim=-1, keepdim=True) * u_from
        dot_to_v = (u_to * v_tan).sum(dim=-1, keepdim=True)  # [B, T, 60, 1]
        sum_u = u_from + u_to                             # [B, T, 60, 3]

        v_transported = v_tan - (sum_u / denom) * dot_to_v
        return v_transported

    def compute_holonomy_loop(
        self,
        u: torch.Tensor,  # [B, T, 60, 3] (Unit sphere directions)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes 3-step closed-loop holonomy rotation trace and solid angle defect theta_hol.
        """
        B, T, K, _ = u.shape
        device = u.device

        # If sequence is short (T < 3), return identity holonomy
        if T < 3:
            traces = torch.full((B, T, K, 1), 3.0, device=device, dtype=u.dtype)
            angles = torch.zeros((B, T, K, 1), device=device, dtype=u.dtype)
            return traces, angles

        # Form closed triangle: u_t -> u_{t+1} -> u_{t+2} -> u_t
        u0 = u[:, :-2, :, :]
        u1 = u[:, 1:-1, :, :]
        u2 = u[:, 2:, :, :]

        # Test canonical orthonormal basis vectors e1, e2, e3 through loop
        e1 = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=u.dtype).view(1, 1, 1, 3).expand(B, T - 2, K, 3)
        e2 = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=u.dtype).view(1, 1, 1, 3).expand(B, T - 2, K, 3)
        e3 = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=u.dtype).view(1, 1, 1, 3).expand(B, T - 2, K, 3)

        # Transport along triangle: 0 -> 1 -> 2 -> 0
        def transport_loop(v_in):
            v_1 = self.parallel_transport_s2(u0, u1, v_in)
            v_2 = self.parallel_transport_s2(u1, u2, v_1)
            v_0 = self.parallel_transport_s2(u2, u0, v_2)
            return v_0

        e1_rot = transport_loop(e1)
        e2_rot = transport_loop(e2)
        e3_rot = transport_loop(e3)

        # Holonomy 3D trace: Tr_{SO(3)}(R) = 1.0 (normal) + Tr_{2D}(R_tan)
        tr1 = (e1_rot * e1).sum(dim=-1, keepdim=True)
        tr2 = (e2_rot * e2).sum(dim=-1, keepdim=True)
        tr3 = (e3_rot * e3).sum(dim=-1, keepdim=True)
        tr_loop = (1.0 + tr1 + tr2 + tr3).clamp(min=-1.0, max=3.0)  # [B, T-2, 60, 1] in [-1, 3]

        # Holonomy solid angle defect: theta = arccos( (Tr(R) - 1) / 2 )
        cos_theta = ((tr_loop - 1.0) / 2.0).clamp(min=-1.0, max=1.0)
        theta_loop = torch.acos(cos_theta)  # [B, T-2, 60, 1]

        # Pad back to length T
        pad_t = T - theta_loop.shape[1]
        traces = F.pad(tr_loop, (0, 0, 0, 0, 0, pad_t), value=3.0)
        angles = F.pad(theta_loop, (0, 0, 0, 0, 0, pad_t), value=0.0)

        return traces, angles

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> RiemannianHolonomyOutput:
        """
        Computes Levi-Civita parallel transport, closed-loop holonomy metrics, and projects to d_model.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device
        eps = 1e-6

        pos = kinematics[..., 0:3]  # [B, T, 60, 3]
        if C >= 6:
            vel = kinematics[..., 3:6]
        else:
            vel = torch.zeros_like(pos)
            if T > 1:
                vel[:, :-1, :, :] = pos[:, 1:, :, :] - pos[:, :-1, :, :]
                vel[:, -1, :, :] = vel[:, -2, :, :]

        # 1. Project onto Unit Sphere S^2
        # Center relative to torso (Joints 14, 15)
        c_torso = 0.5 * (pos[:, :, 14:15, :] + pos[:, :, 15:16, :])  # [B, T, 1, 3]
        rel_pos = pos - c_torso                                       # [B, T, 60, 3]
        norm_r = torch.norm(rel_pos, p=2, dim=-1, keepdim=True).clamp(min=eps)
        u = rel_pos / norm_r                                         # [B, T, 60, 3] in S^2

        # 2. Sequential Parallel Transport of Velocity to Reference Frame
        if T > 1:
            u_t = u[:, :-1, :, :]
            u_next = u[:, 1:, :, :]
            v_t = vel[:, :-1, :, :]
            v_trans_step = self.parallel_transport_s2(u_t, u_next, v_t)
            v_trans = torch.cat([v_trans_step, vel[:, -1:, :, :]], dim=1)
        else:
            v_trans = vel

        # 3. Closed-Loop Holonomy Solid Angle Defect
        traces, angles = self.compute_holonomy_loop(u)

        # 4. Feature Projection
        # [B, T, 60, 5] -> [B, T, 300]
        node_feats = torch.cat([v_trans, angles, traces], dim=-1).reshape(B, T, K * 5)
        h_holonomy = self.proj(node_feats)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_holonomy

        return RiemannianHolonomyOutput(
            holonomy_features=h_holonomy,
            transported_velocities=v_trans,
            holonomy_angles=angles,
            holonomy_traces=traces,
            augmented_features=augmented,
        )
