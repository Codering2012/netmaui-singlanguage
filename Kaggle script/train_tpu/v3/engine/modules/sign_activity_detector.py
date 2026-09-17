#!/usr/bin/env python3
"""
================================================================================
SIGN ACTIVITY DETECTOR (SAD / VVAD) FOR REAL-TIME DEPLOYMENT
================================================================================
Prevents false translation emissions caused by non-signing human activity
in live camera streams:
- Resting hands in lap / on hips (IDLE_REST)
- Incidental fidgeting, coughing, scratching nose, drinking water (INCIDENTAL_FIDGET)
- Active communicative signing strokes (ACTIVE_SIGNING)
- Turn-holding pauses with hands frozen in space (COGNITIVE_HOLD)

Acts as a high-speed, zero-overhead gate before invoking heavy neural translation.
================================================================================
"""

from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignActivityDetector(nn.Module):
    r"""
    Visual Voice Activity Detector (VVAD) / Sign Activity Detector (SAD).
    
    Args:
        in_channels: Kinematic channels per keypoint (default 9).
        num_keypoints: Number of body keypoints (default 60).
        d_model: Hidden feature dimension.
    """

    STATE_IDLE_REST = 0        # Hands down, resting on table or lap
    STATE_INCIDENTAL = 1       # Scratching face, adjusting glasses, touching hair
    STATE_ACTIVE_SIGNING = 2   # Communicative manual signing
    STATE_COGNITIVE_HOLD = 3   # Floor-holding pause while thinking

    def __init__(
        self,
        in_channels: int = 9,
        num_keypoints: int = 60,
        d_model: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.d_model = d_model

        # Lightweight 1D temporal convolution over kinematic velocities & positions
        # Input: [B, T, 60*9] -> [B, T, d_model]
        self.encoder = nn.Sequential(
            nn.Linear(num_keypoints * in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # 4-way Activity State Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 4),
        )

        # Explicit geometric envelope features:
        # 1. Height of hands relative to sternum
        # 2. Distance of hands to nose/eyes (face touch heuristic)
        # 3. Kinetic energy of both hands
        self.geo_proj = nn.Linear(6, d_model)

        # Zero-initialize final classifier layer to start smoothly from inductive prior logits
        nn.init.zeros_(self.classifier[-1].weight)
        nn.init.zeros_(self.classifier[-1].bias)

    def extract_geometric_cues(self, kinematics: torch.Tensor) -> torch.Tensor:
        """
        Derives rule-based physical signals from 60 keypoints:
        [B, T, 60, 9] -> [B, T, 6]
        Features:
        0: R-hand elevation above sternum
        1: L-hand elevation above sternum
        2: R-hand distance to nose
        3: L-hand distance to nose
        4: R-hand kinetic energy (velocity norm)
        5: L-hand kinetic energy (velocity norm)
        """
        B, T = kinematics.shape[:2]
        pts = kinematics.view(B, T, self.num_keypoints, -1)
        pos = pts[:, :, :, :3]  # [B, T, 60, 3]
        vel = pts[:, :, :, 3:6] if pts.shape[-1] >= 6 else torch.diff(pos, dim=1, prepend=pos[:, :1, :])

        # Indices:
        # Sternum: mid-point of shoulders (42, 43) or 0
        r_sh = pos[:, :, 43, :] if self.num_keypoints > 43 else pos[:, :, 1, :]
        l_sh = pos[:, :, 42, :] if self.num_keypoints > 42 else pos[:, :, 0, :]
        sternum = (r_sh + l_sh) * 0.5
        nose = pos[:, :, 48, :] if self.num_keypoints > 48 else pos[:, :, 0, :]

        r_wrist = pos[:, :, 21, :] if self.num_keypoints > 21 else pos[:, :, 12, :]
        l_wrist = pos[:, :, 0, :]

        # Hand elevations (positive = above sternum, negative = resting below)
        r_elev = r_wrist[:, :, 1] - sternum[:, :, 1]
        l_elev = l_wrist[:, :, 1] - sternum[:, :, 1]

        # Distance to nose (face touching / nose scratching heuristic)
        r_nose_dist = torch.norm(r_wrist - nose, dim=-1)
        l_nose_dist = torch.norm(l_wrist - nose, dim=-1)

        # Kinetic energy
        r_energy = torch.norm(vel[:, :, 21, :], dim=-1) if self.num_keypoints > 21 else torch.zeros_like(r_elev)
        l_energy = torch.norm(vel[:, :, 0, :], dim=-1)

        cues = torch.stack([r_elev, l_elev, r_nose_dist, l_nose_dist, r_energy, l_energy], dim=-1)
        return cues

    def compute_prior_logits(self, cues: torch.Tensor) -> torch.Tensor:
        r"""Computes physically grounded inductive prior logits from geometric cues."""
        r_elev = cues[..., 0]
        l_elev = cues[..., 1]
        r_nose_dist = cues[..., 2]
        l_nose_dist = cues[..., 3]
        r_energy = cues[..., 4]
        l_energy = cues[..., 5]

        e_max = torch.maximum(r_elev, l_elev)
        k_sum = r_energy + l_energy
        d_nose = torch.minimum(r_nose_dist, l_nose_dist)

        # Class 0: IDLE_REST (hands low, still)
        p0 = 4.0 * torch.sigmoid(-10.0 * e_max) * torch.sigmoid(10.0 * (0.15 - k_sum))
        # Class 1: INCIDENTAL_FIDGET (hand touching face, low velocity)
        p1 = 4.0 * torch.sigmoid(15.0 * (0.18 - d_nose)) * torch.sigmoid(10.0 * (0.25 - k_sum))
        # Class 2: ACTIVE_SIGNING (hands in signing space, dynamic movement)
        p2 = 4.0 * torch.sigmoid(10.0 * (e_max - 0.05)) * torch.sigmoid(10.0 * (k_sum - 0.08))
        # Class 3: COGNITIVE_HOLD (hands in signing space, stationary hold)
        p3 = 4.0 * torch.sigmoid(10.0 * (e_max - 0.05)) * torch.sigmoid(10.0 * (0.06 - k_sum))

        return torch.stack([p0, p1, p2, p3], dim=-1)

    def forward(
        self,
        kinematics: torch.Tensor,  # [B, T, 60*9] or [B, T, 60, 9]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            kinematics: [B, T, 60*9]
            
        Returns:
            activity_logits: [B, T, 4] 4-class probabilities
            is_active_signing: [B, T] Boolean mask (True for communicative frames)
            aux_info: Dictionary with state breakdown and energy metrics
        """
        B, T = kinematics.shape[:2]
        if kinematics.dim() == 4:
            kin_flat = kinematics.view(B, T, -1)
        else:
            kin_flat = kinematics

        # 1. Kinematic Temporal Convolution
        x = kin_flat
        h = self.encoder[0](x)  # [B, T, d_model]
        h = self.encoder[1](h)
        h = self.encoder[2](h)
        # Conv1d expects [B, d_model, T]
        h_conv = self.encoder[3](h.transpose(1, 2))
        h_conv = self.encoder[4](h_conv)
        h_conv = self.encoder[5](h_conv).transpose(1, 2)  # [B, T, d_model]

        # 2. Geometric heuristic injection & Prior Logits
        cues = self.extract_geometric_cues(kinematics)  # [B, T, 6]
        cues_embed = self.geo_proj(cues)                # [B, T, d_model]
        prior_logits = self.compute_prior_logits(cues)  # [B, T, 4]

        fused = h_conv + cues_embed
        logits = self.classifier(fused) + prior_logits  # [B, T, 4]

        # 3. Decision rule: Active signing = ACTIVE_SIGNING (class 2) or COGNITIVE_HOLD (class 3)
        probs = F.softmax(logits, dim=-1)
        signing_prob = probs[:, :, self.STATE_ACTIVE_SIGNING] + probs[:, :, self.STATE_COGNITIVE_HOLD]
        is_active = signing_prob > 0.45  # Gating threshold

        return logits, is_active, {
            "signing_prob": signing_prob,
            "idle_prob": probs[:, :, self.STATE_IDLE_REST],
            "incidental_prob": probs[:, :, self.STATE_INCIDENTAL],
            "hold_prob": probs[:, :, self.STATE_COGNITIVE_HOLD],
        }
