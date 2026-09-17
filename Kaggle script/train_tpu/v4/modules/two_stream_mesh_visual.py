#!/usr/bin/env python3
"""
================================================================================
ASL V4: TWO-STREAM 3D HAND MESH & DENSE VISUAL EMBEDDING FUSION
================================================================================
Combines:
1. HaMeR / MANO 3D Hand Mesh Parameters (778 vertices / 45 articulation angles)
   + Mediapipe 60-keypoint kinematic skeletal landmarks.
2. Dense Visual Tokens from frozen Foundation Vision Backbones (DINOv2 / VideoMAE-v2 / SigLIP)
   + Multi-scale Upper-Body ROI crops.

Features:
- Micro-Macro Spatial Resonance Cross-Attention
- Stochastic Modality Dropout (p=0.15) preventing modality collapse
- Pure linear projections with LayerNorm (TPU v5e 128x128 tile aligned)
================================================================================
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoStreamMeshVisualFusion(nn.Module):
    """
    Two-Stream fusion combining 3D mesh kinematics with dense spatiotemporal vision tokens.
    """

    def __init__(
        self,
        d_model: int = 512,
        kinematic_in_dim: int = 540,
        mesh_in_dim: int = 1536,
        visual_in_dim: int = 1024,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Kinematic & 3D Mesh Stems
        self.kin_proj = nn.Linear(kinematic_in_dim, d_model)
        self.mesh_proj = nn.Linear(mesh_in_dim, d_model)
        self.kin_norm = nn.LayerNorm(d_model)

        # 2. Dense Visual Foundation Stem
        self.vis_proj = nn.Linear(visual_in_dim, d_model)
        self.vis_norm = nn.LayerNorm(d_model)

        # 3. Micro-Macro Spatial Resonance Cross-Attention
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(d_model)

        # Gated dynamic blend
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2),
        )

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        kinematics: torch.Tensor,                                   # [B, T, kinematic_in_dim]
        mesh_features: Optional[torch.Tensor] = None,               # [B, T, mesh_in_dim] (HaMeR MANO)
        dense_visual_tokens: Optional[torch.Tensor] = None,         # [B, T, visual_in_dim] (DINOv2)
        modality_dropout_prob: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            fused_features: [B, T, d_model]
            resonant_visual: [B, T, d_model] (detached teacher for VAC distillation)
        """
        B, T = kinematics.shape[:2]

        # Flatten kinematics if passed in 4D [B, T, K, C]
        if kinematics.dim() == 4:
            kin_flat = kinematics.view(B, T, -1)
        else:
            kin_flat = kinematics

        # 1. Kinematic projection
        if kin_flat.shape[-1] != self.kin_proj.in_features:
            # Flexible projection adapter
            h_kin = F.pad(kin_flat, (0, max(0, self.kin_proj.in_features - kin_flat.shape[-1])))[:, :, :self.kin_proj.in_features]
            h_kin = self.kin_proj(h_kin)
        else:
            h_kin = self.kin_proj(kin_flat)

        # Fuse 3D Mesh (HaMeR) if available
        if mesh_features is not None:
            if mesh_features.shape[-1] != self.mesh_proj.in_features:
                m_feat = F.pad(mesh_features, (0, max(0, self.mesh_proj.in_features - mesh_features.shape[-1])))[:, :, :self.mesh_proj.in_features]
            else:
                m_feat = mesh_features
            h_mesh = self.mesh_proj(m_feat)
            h_kin = h_kin + h_mesh

        h_kin = self.kin_norm(h_kin)

        # 2. Visual stream (DINOv2 / VideoMAE)
        if dense_visual_tokens is not None:
            if dense_visual_tokens.shape[-1] != self.vis_proj.in_features:
                v_feat = F.pad(dense_visual_tokens, (0, max(0, self.vis_proj.in_features - dense_visual_tokens.shape[-1])))[:, :, :self.vis_proj.in_features]
            else:
                v_feat = dense_visual_tokens
            h_vis = self.vis_norm(self.vis_proj(v_feat))

            # Stochastic modality dropout during training
            if self.training and modality_dropout_prob > 0.0:
                if torch.rand(1).item() < modality_dropout_prob:
                    h_vis = torch.zeros_like(h_vis)

            # Cross-modal resonance: Kinematics queries Visual tokens
            res_vis, _ = self.cross_attn(query=h_kin, key=h_vis, value=h_vis)
            res_vis = self.cross_norm(res_vis)

            # Gated dynamic blend
            gate_logits = self.fusion_gate(torch.cat([h_kin, res_vis], dim=-1))  # [B, T, 2]
            weights = F.softmax(gate_logits, dim=-1)
            fused = weights[..., 0:1] * h_kin + weights[..., 1:2] * res_vis
        else:
            fused = h_kin
            res_vis = None

        return self.out_norm(fused), res_vis
