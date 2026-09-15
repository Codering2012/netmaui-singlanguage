#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — REFERENCE-BASED PART NORMALIZATION (REFERENCENORMSIGN)
================================================================================
Implements Multi-Body Part Anatomical Normalization & Canonical Warping (PartNorm-SLT):
1. Part-Specific Anchors and Characteristic Anatomical Scales:
     Face (0..13):        c_face  = p_0 (nose tip),       s_face  = ||p_0 - p_1||_2
     Torso (14..17):      c_torso = (p_14 + p_15) / 2,     s_torso = ||p_14 - p_15||_2
     Left Hand (18..38):  c_lhand = p_18 (wrist),         s_lhand = ||p_18 - p_27||_2
     Right Hand (39..59): c_rhand = p_39 (wrist),         s_rhand = ||p_39 - p_48||_2
2. Canonical Local Coordinate Normalization:
     p_hat_i = (p_i - c_m) / s_m,   for all i in Part_m
     Strictly eliminates subject body scale alpha and camera translation t.
3. Cross-Part Spatial Interaction Vectors:
     d_rh_face = (p_39 - p_0) / s_torso
     d_lh_face = (p_18 - p_0) / s_torso
     d_hands   = (p_39 - p_18) / s_torso
4. Feature Projection & Canonical Frame Fusion:
     H_norm = H + LayerNorm( Linear( [p_hat, d_interactions, scales] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferencePartNormOutput(NamedTuple):
    normalized_features: torch.Tensor   # [B, T, d_model] Projected part-normalized representations
    canonical_coords: torch.Tensor      # [B, T, 60, 3] Scale- and position-normalized keypoints
    interaction_vectors: torch.Tensor   # [B, T, 9] Bilateral hand-face-hand relative spatial cues
    part_scales: torch.Tensor           # [B, T, 4] Extracted characteristic scales (Face, Torso, LH, RH)
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + normalized_features


class ASLReferencePartNormalizationEngine(nn.Module):
    """
    Reference-Based Multi-Body Part Anatomical Normalization Engine.
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

        # Part index ranges
        self.face_indices = list(range(0, 14))
        self.torso_indices = list(range(14, 18))
        self.lhand_indices = list(range(18, 39))
        self.rhand_indices = list(range(39, 60))

        # Input dimension: 60*3 (canonical coords) + 9 (interactions) + 4 (scales) = 193
        in_dim = num_keypoints * 3 + 9 + 4
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_part_normalization(
        self,
        pos: torch.Tensor,  # [B, T, 60, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts part anchors and scales, normalizes coordinates, and computes interaction cues.
        Returns: (canonical_pos [B, T, 60, 3], interactions [B, T, 9], scales [B, T, 4])
        """
        B, T, K, _ = pos.shape
        device = pos.device
        eps = 1e-6

        # 1. Extract Anchors: [B, T, 3]
        c_face  = pos[:, :, 0, :]                             # Nose tip
        c_torso = 0.5 * (pos[:, :, 14, :] + pos[:, :, 15, :]) # Mid-shoulder
        c_lhand = pos[:, :, 18, :]                            # Left wrist
        c_rhand = pos[:, :, 39, :]                            # Right wrist

        # 2. Extract Characteristic Scales: [B, T, 1]
        s_face  = torch.norm(pos[:, :, 0, :] - pos[:, :, 1, :], p=2, dim=-1, keepdim=True).clamp(min=eps)
        s_torso = torch.norm(pos[:, :, 14, :] - pos[:, :, 15, :], p=2, dim=-1, keepdim=True).clamp(min=eps)
        s_lhand = torch.norm(pos[:, :, 18, :] - pos[:, :, 27, :], p=2, dim=-1, keepdim=True).clamp(min=eps) # Wrist to Middle Tip
        s_rhand = torch.norm(pos[:, :, 39, :] - pos[:, :, 48, :], p=2, dim=-1, keepdim=True).clamp(min=eps) # Wrist to Middle Tip

        # 3. Canonical Local Part Normalization: [B, T, K, 3]
        canonical_pos = torch.zeros_like(pos)
        canonical_pos[:, :, self.face_indices, :]  = (pos[:, :, self.face_indices, :] - c_face.unsqueeze(2)) / s_face.unsqueeze(2)
        canonical_pos[:, :, self.torso_indices, :] = (pos[:, :, self.torso_indices, :] - c_torso.unsqueeze(2)) / s_torso.unsqueeze(2)
        canonical_pos[:, :, self.lhand_indices, :] = (pos[:, :, self.lhand_indices, :] - c_lhand.unsqueeze(2)) / s_lhand.unsqueeze(2)
        canonical_pos[:, :, self.rhand_indices, :] = (pos[:, :, self.rhand_indices, :] - c_rhand.unsqueeze(2)) / s_rhand.unsqueeze(2)

        # 4. Bilateral Spatial Interaction Vectors: [B, T, 3]
        d_rh_face = (pos[:, :, 39, :] - pos[:, :, 0, :]) / s_torso  # Right hand to face
        d_lh_face = (pos[:, :, 18, :] - pos[:, :, 0, :]) / s_torso  # Left hand to face
        d_hands   = (pos[:, :, 39, :] - pos[:, :, 18, :]) / s_torso # Right hand to left hand

        interactions = torch.cat([d_rh_face, d_lh_face, d_hands], dim=-1)  # [B, T, 9]
        scales = torch.cat([s_face, s_torso, s_lhand, s_rhand], dim=-1)     # [B, T, 4]

        return canonical_pos, interactions, scales

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] or [B, T, 60, 3]
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional representations
    ) -> ReferencePartNormOutput:
        """
        Executes reference part normalization, interaction calculation, and feature projection.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        pos = kinematics[..., 0:3]  # [B, T, 60, 3]

        # 1. Compute Part Normalization
        canonical_pos, interactions, scales = self.compute_part_normalization(pos)

        # 2. Flatten and Concatenate
        p_flat = canonical_pos.reshape(B, T, self.num_keypoints * 3)  # [B, T, 180]
        f_all = torch.cat([p_flat, interactions, scales], dim=-1)     # [B, T, in_dim]

        # 3. Project to Model Dimension
        h_norm = self.proj(f_all)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_norm

        return ReferencePartNormOutput(
            normalized_features=h_norm,
            canonical_coords=canonical_pos,
            interaction_vectors=interactions,
            part_scales=scales,
            augmented_features=augmented,
        )
