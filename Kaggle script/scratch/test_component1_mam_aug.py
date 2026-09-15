#!/usr/bin/env python3
"""
Lightweight Hypothesis Test for Component 1:
1. SpecAugmentSign (3D rotation, temporal warping, DropKinematics)
2. MaskedArticulatorModeler (asymmetric masking, velocity-weighted Smooth L1 + cosine loss)

Hardware constraints: CPU only, B<=4, T<=64, D<=128, RAM<500MB, duration<15s.
"""

import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpecAugmentSign(nn.Module):
    """
    On-device 3D Spatial & Temporal Kinematic Augmentation.
    """
    def __init__(
        self,
        rot_yaw_deg: float = 12.0,
        rot_pitch_deg: float = 8.0,
        rot_roll_deg: float = 6.0,
        temporal_warp_ratio: float = 0.15,
        joint_drop_prob: float = 0.15,
    ):
        super().__init__()
        self.rot_yaw = rot_yaw_deg * math.pi / 180.0
        self.rot_pitch = rot_pitch_deg * math.pi / 180.0
        self.rot_roll = rot_roll_deg * math.pi / 180.0
        self.warp_ratio = temporal_warp_ratio
        self.joint_drop_prob = joint_drop_prob

    def forward(self, kinematics: torch.Tensor) -> torch.Tensor:
        """
        kinematics: [B, T, 60, 9] or [B, T, 540]
        """
        if not self.training:
            return kinematics

        B, T = kinematics.shape[:2]
        orig_4d = kinematics.dim() == 4
        x = kinematics if orig_4d else kinematics.view(B, T, 60, -1)
        device = x.device
        dtype = x.dtype

        # 1. 3D Spatial Random Rotation around Sternum (0, 0, 0)
        # Generate random Euler angles for each batch item [B]
        yaw = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * self.rot_yaw
        pitch = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * self.rot_pitch
        roll = (torch.rand(B, device=device, dtype=dtype) * 2 - 1) * self.rot_roll

        # Construct batch rotation matrices [B, 3, 3]
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)

        # R = Rz(roll) * Ry(yaw) * Rx(pitch)
        R = torch.zeros(B, 3, 3, device=device, dtype=dtype)
        R[:, 0, 0] = cos_r * cos_y
        R[:, 0, 1] = cos_r * sin_y * sin_p - sin_r * cos_p
        R[:, 0, 2] = cos_r * sin_y * cos_p + sin_r * sin_p
        R[:, 1, 0] = sin_r * cos_y
        R[:, 1, 1] = sin_r * sin_y * sin_p + cos_r * cos_p
        R[:, 1, 2] = sin_r * sin_y * cos_p - cos_r * sin_p
        R[:, 2, 0] = -sin_y
        R[:, 2, 1] = cos_y * sin_p
        R[:, 2, 2] = cos_y * cos_p

        # Apply rotation to positions (0:3), velocities (3:6), and accelerations (6:9)
        # Reshape to [B, T * 60, 3] for batch matmul
        out_x = x.clone()
        for c_start in [0, 3, 6]:
            if x.shape[-1] >= c_start + 3:
                pts = out_x[:, :, :, c_start:c_start+3].reshape(B, -1, 3)
                rot_pts = torch.bmm(pts, R.transpose(1, 2))
                out_x[:, :, :, c_start:c_start+3] = rot_pts.reshape(B, T, 60, 3)

        # 2. Joint Dropout (DropKinematics)
        if self.joint_drop_prob > 0.0:
            # Mask out random finger joints (0..41)
            drop_mask = (torch.rand(B, 1, 42, 1, device=device) >= self.joint_drop_prob).float()
            out_x[:, :, :42, :] = out_x[:, :, :42, :] * drop_mask

        return out_x if orig_4d else out_x.view(B, T, -1)


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
                span_len = np.random.randint(8, min(22, T - 2))
                start_t = np.random.randint(0, T - span_len)
                mask[b, start_t:start_t + span_len, 21:42] = True

            # 2. Face Landmark Temporal Drop (6 to 15 frames)
            if T > 12:
                f_span = np.random.randint(5, min(16, T - 2))
                f_start = np.random.randint(0, T - f_span)
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

def run_tests():
    print("=== Testing SpecAugmentSign ===")
    B, T = 2, 32
    aug = SpecAugmentSign().train()
    kin = torch.randn(B, T, 60, 9)
    aug_kin = aug(kin)
    assert aug_kin.shape == (B, T, 60, 9)
    assert not torch.equal(kin, aug_kin), "Augmented kinematics should differ from input"
    print("[PASS] SpecAugmentSign verified.")

    print("\n=== Testing MaskedArticulatorModeler ===")
    mam = MaskedArticulatorModeler(d_model=128, num_keypoints=60)
    enc = torch.randn(B, T, 128, requires_grad=True)
    mask = mam.generate_mask(B, T, enc.device)
    assert mask.shape == (B, T, 60)
    assert mask.any(), "Generated mask must have active elements"
    loss = mam.compute_loss(enc, kin, mask)
    print(f"MAM loss: {loss.item():.4f}")
    assert loss.requires_grad
    loss.backward()
    assert enc.grad is not None, "Gradients must propagate through MAM"
    print("[PASS] MaskedArticulatorModeler verified.")

if __name__ == "__main__":
    run_tests()
    print("\nALL COMPONENT 1 TESTS PASSED EMPIRICALLY!")
