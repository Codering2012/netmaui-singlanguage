#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — TEMPORAL SPEED-VARIANCE CURRICULUM ENGINE
================================================================================
Implements Multi-Rate Kinematics Resampling & Speed Invariance (MCL-SLT / SignVTCL):
1. Kinematically-Consistent Speed Warping:
     x_warp(t) = Interp(x, s * T)
     v_warp(t) = v(t) * s
     a_warp(t) = a(t) * s^2
   Maintains strict Newtonian kinematic consistency across variable signing speeds.
2. Progressive Curriculum Scheduler:
     Expands speed scaling bounds from [0.90, 1.10] (Epoch 1) to [0.60, 1.60] (Epoch E_max).
3. Speed-Invariant Multi-Rate Contrastive Loss:
     Aligns latent representations of identical gesture sequences performed at
     different temporal velocities.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLSpeedVarianceCurriculum:
    """
    Manages continuous multi-rate temporal time-warping, curriculum speed bounds,
    and speed-invariant contrastive regularization.
    """

    def __init__(
        self,
        min_speed_limit: float = 0.60,
        max_speed_limit: float = 1.60,
        total_curriculum_epochs: int = 40,
        contrastive_temp: float = 0.10,
    ):
        self.min_speed_limit = min_speed_limit
        self.max_speed_limit = max_speed_limit
        self.total_curriculum_epochs = total_curriculum_epochs
        self.contrastive_temp = contrastive_temp

    def get_epoch_speed_bounds(self, current_epoch: int) -> Tuple[float, float]:
        """
        Calculates active speed scaling range [s_min, s_max] for current training epoch.
        """
        progress = min(1.0, max(0.0, (current_epoch - 1) / max(1, self.total_curriculum_epochs - 1)))
        # Cosine curriculum progression
        curve = 0.5 * (1.0 - math.cos(progress * math.pi))

        s_min = 1.0 - curve * (1.0 - self.min_speed_limit)
        s_max = 1.0 + curve * (self.max_speed_limit - 1.0)
        return s_min, s_max

    def warp_temporal_speed(
        self,
        kinematics: torch.Tensor,
        speed_factor: float,
    ) -> torch.Tensor:
        """
        Resamples kinematics temporally by speed_factor while preserving physical laws:
        kinematics: [B, T, K, C] (C >= 3)
        Returns: [B, T_new, K, C] where T_new = int(T / speed_factor)
        """
        B, T, K, C = kinematics.shape
        if T <= 4 or abs(speed_factor - 1.0) < 1e-3:
            return kinematics

        # Target sequence length: higher speed = fewer frames
        target_len = max(4, int(round(T / speed_factor)))

        # Reshape to [B * K, C, T] for 1D linear interpolation
        x_flat = kinematics.permute(0, 2, 3, 1).reshape(B * K, C, T)

        # 1D Temporal Resampling
        resampled_flat = F.interpolate(
            x_flat,
            size=target_len,
            mode="linear",
            align_corners=False,
        )

        warped = resampled_flat.view(B, K, C, target_len).permute(0, 3, 1, 2)  # [B, T_new, K, C]

        # Adjust velocity (v * s) and acceleration (a * s^2) if present
        if C >= 6:
            # Velocity channels 3:6
            warped[..., 3:6] = warped[..., 3:6] * speed_factor
        if C >= 9:
            # Acceleration channels 6:9
            warped[..., 6:9] = warped[..., 6:9] * (speed_factor ** 2)

        return warped

    def compute_speed_invariance_loss(
        self,
        z_standard: torch.Tensor,
        z_warped: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes InfoNCE contrastive alignment between standard and speed-warped latent vectors.
        z_standard, z_warped: [B, D]
        """
        z_std_norm = F.normalize(z_standard, p=2, dim=-1)
        z_warp_norm = F.normalize(z_warped, p=2, dim=-1)

        sim_matrix = torch.matmul(z_std_norm, z_warp_norm.t()) / self.contrastive_temp  # [B, B]
        labels = torch.arange(z_standard.size(0), device=z_standard.device)

        loss = 0.5 * (
            F.cross_entropy(sim_matrix, labels) +
            F.cross_entropy(sim_matrix.t(), labels)
        )
        return loss
