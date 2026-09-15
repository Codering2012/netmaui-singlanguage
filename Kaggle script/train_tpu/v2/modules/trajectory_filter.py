#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — KINEMATIC TRAJECTORY DENOISING & TEMPORAL FILTER
================================================================================
Implements Robust Temporal Smoothing for MediaPipe/RTMPose Landmark Streams:
1. Savitzky-Golay / Gaussian Trajectory Denoising:
     x_smooth(t, k) = sum_{w=-W}^W c_w * x(t+w, k)
   Eliminates high-frequency camera estimation jitter while preserving rapid
   linguistic articulation and inflection points.
2. Anomaly Jump Clamping & Linear Interpolation for Dropouts:
     Detects tracking teleportations (||Delta x|| > tau) and repairs corrupted frames.
3. Full 9-Channel Kinematics Generation:
     Recomputes smooth Velocity (v) and Acceleration (a) from denoised 3D trajectories:
       Channels = [x, y, z, vx, vy, vz, ax, ay, az]
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLKinematicTrajectoryFilter:
    """
    Temporal trajectory smoothing and kinematics enrichment filter for continuous sign language landmarks.
    """

    def __init__(
        self,
        window_size: int = 5,
        polynomial_order: int = 2,
        max_jump_threshold: float = 0.25,  # Max allowed normalized spatial jump per frame
        device: Union[str, torch.device] = "cpu",
    ):
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.half_window = self.window_size // 2
        self.polynomial_order = polynomial_order
        self.max_jump_threshold = max_jump_threshold
        self.device = torch.device(device)

        # Precompute 1D Gaussian smoothing kernel
        sigma = (self.window_size - 1) / 4.0
        kernel_1d = torch.tensor(
            [math.exp(-((i - self.half_window) ** 2) / (2.0 * sigma ** 2)) for i in range(self.window_size)],
            dtype=torch.float32,
            device=self.device,
        )
        self.kernel_1d = (kernel_1d / kernel_1d.sum()).view(1, 1, self.window_size)

    def clamp_and_interpolate_anomalies(self, features_3d: torch.Tensor) -> torch.Tensor:
        """
        Detects anomalous spatial teleportations (tracking glitches) and smooths them.
        features_3d: [B, T, K, 3]
        """
        B, T, K, _ = features_3d.shape
        if T <= 2:
            return features_3d

        clamped = features_3d.clone()
        for t in range(1, T):
            diff = torch.norm(clamped[:, t] - clamped[:, t - 1], dim=-1)  # [B, K]
            jump_mask = diff > self.max_jump_threshold
            if jump_mask.any():
                if t < T - 1:
                    # Linear interpolation between t-1 and t+1
                    interp_val = 0.5 * (clamped[:, t - 1] + clamped[:, t + 1])
                    clamped[:, t][jump_mask] = interp_val[jump_mask]
                else:
                    clamped[:, t][jump_mask] = clamped[:, t - 1][jump_mask]

        return clamped

    def smooth_trajectories(self, features_3d: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D temporal convolution smoothing across frame dimension.
        features_3d: [B, T, K, 3]
        """
        B, T, K, C = features_3d.shape
        if T < self.window_size:
            return features_3d

        # Reshape to [B * K * C, 1, T] for 1D convolution
        x_flat = features_3d.permute(0, 2, 3, 1).reshape(B * K * C, 1, T)

        # Replicate padding on time boundaries
        x_padded = F.pad(x_flat, (self.half_window, self.half_window), mode="replicate")
        smoothed_flat = F.conv1d(x_padded, self.kernel_1d)

        smoothed = smoothed_flat.view(B, K, C, T).permute(0, 3, 1, 2)
        return smoothed

    def filter_and_enrich_kinematics(
        self,
        raw_landmarks: torch.Tensor,
        fps: float = 30.0,
    ) -> torch.Tensor:
        """
        Takes raw landmarks [B, T, K, 3] or [B, T, K, C], denoises spatial coordinates,
        and derives high-fidelity 9-channel kinematics: [pos (3), vel (3), acc (3)].
        Returns: [B, T, K, 9]
        """
        raw_landmarks = raw_landmarks.to(self.device)
        pos_3d = raw_landmarks[..., :3]
        B, T, K, _ = pos_3d.shape

        # 1. Anomaly Jump Clamping
        pos_clean = self.clamp_and_interpolate_anomalies(pos_3d)

        # 2. Temporal Gaussian Trajectory Smoothing
        pos_smooth = self.smooth_trajectories(pos_clean)

        # 3. Derive First-Order Velocity (v)
        dt = 1.0 / max(1.0, fps)
        vel = torch.zeros_like(pos_smooth)
        if T > 1:
            vel[:, 1:] = (pos_smooth[:, 1:] - pos_smooth[:, :-1]) / dt
            vel[:, 0] = vel[:, 1]  # Initial boundary condition

        # 4. Derive Second-Order Acceleration (a)
        acc = torch.zeros_like(vel)
        if T > 2:
            acc[:, 1:-1] = (vel[:, 2:] - vel[:, :-2]) / (2.0 * dt)
            acc[:, 0] = acc[:, 1]
            acc[:, -1] = acc[:, -2]

        # 5. Concatenate into 9-channel kinematics [pos, vel, acc]
        kinematics_9ch = torch.cat([pos_smooth, vel, acc], dim=-1)
        return kinematics_9ch
