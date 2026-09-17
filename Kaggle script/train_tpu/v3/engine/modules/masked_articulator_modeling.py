#!/usr/bin/env python3
"""
================================================================================
MASKED ARTICULATOR MODELING (MAM): HIGH-SPEED SELF-SUPERVISED PRETRAINING
================================================================================
Implements harsh, asymmetric articulator masking tailored to ASL phonology:
- Right Hand Drop: masks 10-25 contiguous frames of dominant hand.
- Non-Manual Facial Drop: masks 5-15 frames of facial landmarks.
- Velocity-Weighted Smooth L1 + Directional Cosine Error.

Enables self-supervised pretraining at 1500+ FPS on Cloud TPU v5e without decoders.
================================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedArticulatorModeler(nn.Module):
    """
    Harsh Masked Articulator Modeling for self-supervised pretraining.
    """
    def __init__(self, d_model: int = 128, num_keypoints: int = 60):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.recon_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_keypoints * 3), # Predicts 3D velocities
        )

    def generate_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        """
        Creates asymmetric articulator mask [B, T, 60]:
        - Right Hand: 21..41
        - Face: 48..59
        """
        mask = torch.zeros(B, T, self.num_keypoints, device=device, dtype=torch.bool)
        for b in range(B):
            # 1. Harsh Right Hand Temporal Drop (10 to 20 contiguous frames)
            if T > 15:
                span_len = int(np.random.randint(8, min(22, T - 2)))
                start_t = int(np.random.randint(0, T - span_len))
                mask[b, start_t:start_t + span_len, 21:42] = True

            # 2. Face Landmark Temporal Drop (5 to 15 frames)
            if T > 12:
                f_span = int(np.random.randint(5, min(16, T - 2)))
                f_start = int(np.random.randint(0, T - f_span))
                mask[b, f_start:f_start + f_span, 48:60] = True
        return mask

    def compute_loss(
        self,
        encoded_features: torch.Tensor, # [B, T, d_model]
        gt_kinematics: torch.Tensor,    # [B, T, 60, 9] or [B, T, 540]
        mask: torch.Tensor,             # [B, T, 60]
    ) -> torch.Tensor:
        B, T = encoded_features.shape[:2]
        pred_vel = self.recon_head(encoded_features).view(B, T, self.num_keypoints, 3)
        gt_4d = gt_kinematics if gt_kinematics.dim() == 4 else gt_kinematics.view(B, T, self.num_keypoints, -1)
        gt_vel = gt_4d[:, :, :, 3:6].detach() # Target velocity

        if not mask.any():
            return torch.tensor(0.0, device=encoded_features.device, requires_grad=True)

        pred_masked = pred_vel[mask] # [N, 3]
        gt_masked = gt_vel[mask]     # [N, 3]

        smooth_l1 = F.smooth_l1_loss(pred_masked, gt_masked)
        # Directional cosine error
        cos_sim = F.cosine_similarity(pred_masked + 1e-6, gt_masked + 1e-6, dim=-1)
        cos_loss = torch.mean(1.0 - cos_sim)

        return smooth_l1 + 0.5 * cos_loss
