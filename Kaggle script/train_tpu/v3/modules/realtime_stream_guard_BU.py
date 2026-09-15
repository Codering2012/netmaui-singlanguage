#!/usr/bin/env python3
"""
================================================================================
REAL-TIME STREAM GUARD FOR IN-THE-WILD WEBCAM SIGNING DEPLOYMENT
================================================================================
Solves the 4 most prevalent failure modes when deploying continuous sign language
translation models to consumer webcams, smartphones, and video calls:

1. Bi-Acromial Metric Normalizer (BAMN):
   Eliminates distance/depth variations (signer leaning forward/back, changing zoom)
   by normalizing all keypoints by the biomechanically constant shoulder width.

2. Temporal-Timestamp Invariant Continuous Kinematics (T-TICK):
   Eliminates velocity spikes caused by webcam framerate jitter (15-30 FPS)
   and dropped frames using explicit delta-time scaling: v = (p_t - p_{t-1}) / dt.

3. Anatomical Continuity & Handedness Disambiguation Tracker (ACHD):
   Detects and repairs Left/Right hand label swaps caused by landmark tracker
   confusion during two-handed crossing signs ("CHANGE", "RELATIONSHIP", "WAR").

4. Conversational Backchannel & Turn-Holding Gate (BSTG):
   Distinguishes between listener backchannels (periodic head nods with hands idle)
   and active conversational turn-taking, preventing false caption emissions.
================================================================================
"""

from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BiAcromialMetricNormalizer:
    r"""
    Normalizes 3D skeletal landmarks to be invariant to camera distance, zoom,
    and signer physical stature using the bi-acromial shoulder diameter.
    
    Proof:
        x_pixel = f * X / Z
        D_shoulder = f * W_shoulder / Z
        x_norm = x_pixel / D_shoulder = X / W_shoulder (independent of f and Z).
    """

    def __init__(self, left_shoulder_idx: int = 42, right_shoulder_idx: int = 43, eps: float = 1e-6):
        self.l_idx = left_shoulder_idx
        self.r_idx = right_shoulder_idx
        self.eps = eps

    def normalize(self, landmarks_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            landmarks_3d: [..., num_kp, 3] or [..., num_kp * 3]
            
        Returns:
            normalized_landmarks: Same shape, centered at sternum and scaled by shoulder width.
            shoulder_width: [...] scale factor.
        """
        orig_shape = landmarks_3d.shape
        if landmarks_3d.shape[-1] != 3:
            reshaped = landmarks_3d.view(*orig_shape[:-1], -1, 3)
        else:
            reshaped = landmarks_3d

        # Extract shoulders
        l_sh = reshaped[..., self.l_idx, :] if reshaped.shape[-2] > self.l_idx else reshaped[..., 0, :]
        r_sh = reshaped[..., self.r_idx, :] if reshaped.shape[-2] > self.r_idx else reshaped[..., 1, :]

        sternum = (l_sh + r_sh) * 0.5  # [..., 1, 3]
        shoulder_width = torch.norm(r_sh - l_sh, dim=-1, keepdim=True).unsqueeze(-2)  # [..., 1, 1]
        shoulder_width = torch.clamp(shoulder_width, min=self.eps)

        norm_pts = (reshaped - sternum.unsqueeze(-2)) / shoulder_width

        if landmarks_3d.shape[-1] != 3:
            return norm_pts.view(orig_shape), shoulder_width.squeeze(-1).squeeze(-1)
        return norm_pts, shoulder_width.squeeze(-1).squeeze(-1)


class ContinuousKinematicsNormalizer:
    r"""
    Temporal-Timestamp Invariant Continuous Kinematics (T-TICK).
    Computes true physical velocity and acceleration from variable-rate webcam frames
    by dividing by actual timestamp deltas (dt):
        v_t = (p_t - p_{t-1}) / (dt + eps)
        a_t = (v_t - v_{t-1}) / (dt + eps)
    """

    def __init__(self, default_fps: float = 30.0, max_valid_dt: float = 0.5, eps: float = 1e-5):
        self.default_dt = 1.0 / default_fps
        self.max_valid_dt = max_valid_dt
        self.eps = eps
        self.prev_pos: Optional[torch.Tensor] = None
        self.prev_vel: Optional[torch.Tensor] = None
        self.prev_timestamp: Optional[float] = None

    def reset(self):
        self.prev_pos = None
        self.prev_vel = None
        self.prev_timestamp = None

    def step(
        self,
        current_pos: torch.Tensor,       # [num_kp, 3]
        timestamp: Optional[float] = None, # seconds
    ) -> torch.Tensor:
        r"""
        Returns 9D kinematic vector [pos(3), vel(3), accel(3)] for all keypoints: [num_kp, 9]
        """
        # Determine dt
        if timestamp is not None and self.prev_timestamp is not None:
            dt = timestamp - self.prev_timestamp
            if dt <= 0 or dt > self.max_valid_dt:
                dt = self.default_dt
        else:
            dt = self.default_dt

        self.prev_timestamp = timestamp

        # Compute velocity
        if self.prev_pos is None:
            vel = torch.zeros_like(current_pos)
            accel = torch.zeros_like(current_pos)
        else:
            # Mask out keypoints that are missing / zero
            valid_mask = (torch.norm(current_pos, dim=-1) > 1e-4) & (torch.norm(self.prev_pos, dim=-1) > 1e-4)
            vel = (current_pos - self.prev_pos) / (dt + self.eps)
            vel[~valid_mask] = 0.0

            if self.prev_vel is None:
                accel = torch.zeros_like(current_pos)
            else:
                accel = (vel - self.prev_vel) / (dt + self.eps)
                accel[~valid_mask] = 0.0

        self.prev_pos = current_pos.clone()
        self.prev_vel = vel.clone()

        # Combine into 9D kinematics [num_kp, 9]
        return torch.cat([current_pos, vel, accel], dim=-1)


class HandednessContinuityTracker:
    r"""
    Anatomical Continuity & Handedness Disambiguation Tracker (ACHD).
    Detects and repairs Left/Right hand label swaps in real time by tracking
    wrist trajectories against biomechanical velocity bounds (v <= 4.5 m/s).
    """

    def __init__(
        self,
        left_wrist_idx: int = 0,
        right_wrist_idx: int = 21,
        max_jump_threshold: float = 0.15,  # Relative to shoulder width
    ):
        self.l_wrist_idx = left_wrist_idx
        self.r_wrist_idx = right_wrist_idx
        self.threshold = max_jump_threshold
        self.prev_l_wrist: Optional[torch.Tensor] = None
        self.prev_r_wrist: Optional[torch.Tensor] = None
        self.is_swapped = False

    def reset(self):
        self.prev_l_wrist = None
        self.prev_r_wrist = None
        self.is_swapped = False

    def disambiguate_and_repair(
        self,
        landmarks: torch.Tensor,  # [num_kp, 3] or [..., num_kp, 3]
    ) -> Tuple[torch.Tensor, bool]:
        r"""
        Checks whether left and right hand keypoints have erroneously swapped identities,
        and returns the corrected landmarks along with a swap flag.
        """
        pts = landmarks.clone()
        l_curr = pts[..., self.l_wrist_idx, :3]
        r_curr = pts[..., self.r_wrist_idx, :3]

        if self.prev_l_wrist is None or self.prev_r_wrist is None:
            self.prev_l_wrist = l_curr.clone()
            self.prev_r_wrist = r_curr.clone()
            return pts, False

        # Compute displacement under hypothesis (normal vs swapped)
        dist_normal = torch.norm(l_curr - self.prev_l_wrist) + torch.norm(r_curr - self.prev_r_wrist)
        dist_swapped = torch.norm(l_curr - self.prev_r_wrist) + torch.norm(r_curr - self.prev_l_wrist)

        # If swapped assignment is significantly closer and normal jump exceeds threshold
        detected_swap = False
        if dist_swapped < dist_normal and (dist_normal - dist_swapped) > self.threshold:
            # Perform hand swap repair on hand landmark blocks
            # Left hand: [0:21], Right hand: [21:42]
            l_block = pts[..., 0:21, :].clone()
            r_block = pts[..., 21:42, :].clone()
            pts[..., 0:21, :] = r_block
            pts[..., 21:42, :] = l_block
            detected_swap = True

        self.prev_l_wrist = pts[..., self.l_wrist_idx, :3].clone()
        self.prev_r_wrist = pts[..., self.r_wrist_idx, :3].clone()
        return pts, detected_swap


class ConversationalBackchannelGate:
    r"""
    Backchannel Suppression & Turn-Holding Gate (BSTG).
    Distinguishes between conversational listener backchannels (periodic head nods at 1.5-3 Hz
    with hands resting) and active communicative signing turns.
    """

    def __init__(self, window_size: int = 15, nod_frequency_band: Tuple[float, float] = (1.2, 3.5)):
        self.window_size = window_size
        self.low_freq, self.high_freq = nod_frequency_band
        self.pitch_history: List[float] = []

    def reset(self):
        self.pitch_history.clear()

    def evaluate_backchannel(
        self,
        cranial_pitch_velocity: float,
        hand_elevation: float,
        hand_kinetic_energy: float,
    ) -> bool:
        r"""
        Returns True if the signer is merely listening and nodding ("backchanneling"),
        meaning text generation should NOT be triggered.
        """
        # If hands are active in signing space, it's definitely NOT a passive backchannel
        if hand_elevation > 0.05 or hand_kinetic_energy > 0.15:
            self.pitch_history.clear()
            return False

        self.pitch_history.append(cranial_pitch_velocity)
        if len(self.pitch_history) > self.window_size:
            self.pitch_history.pop(0)

        if len(self.pitch_history) < 8:
            return False

        # Count zero-crossings in cranial pitch velocity to estimate nod frequency
        arr = np.array(self.pitch_history)
        zero_crossings = np.where(np.diff(np.sign(arr)))[0]
        # At 30 FPS, window of 15 frames = 0.5 sec
        # 1-2 zero crossings in 0.5s = 1-2 Hz nod
        is_nodding = 1 <= len(zero_crossings) <= 4 and np.std(arr) > 0.05
        return is_nodding


from .edge_case_mitigators import (
    DominantHandClassifierAndMirror,
    OneEuroLandmarkFilter,
    PerspectivePitchNormalizer,
)


class RealtimeStreamGuard:
    r"""
    Unified Production Stream Guard.
    Wraps raw webcam frames, applies BAMN scale normalization, T-TICK kinematics,
    ACHD hand disambiguation, One-Euro jitter filtering, Perspective tilt correction,
    Left-handed parity mirroring, and BSTG backchannel suppression before model inference.
    """

    def __init__(self):
        self.bamn = BiAcromialMetricNormalizer()
        self.ttick = ContinuousKinematicsNormalizer()
        self.achd = HandednessContinuityTracker()
        self.bstg = ConversationalBackchannelGate()
        self.oe_filter = OneEuroLandmarkFilter(fc_min=1.0, beta=10.0)
        self.pitch_norm = PerspectivePitchNormalizer()
        self.hand_mirror = DominantHandClassifierAndMirror()

    def reset(self):
        self.ttick.reset()
        self.achd.reset()
        self.bstg.reset()
        self.oe_filter.reset()
        self.hand_mirror.reset()

    def process_frame(
        self,
        raw_landmarks: torch.Tensor,       # [60, 3]
        timestamp: Optional[float] = None, # seconds
        cranial_pitch_vel: float = 0.0,
    ) -> Dict[str, Any]:
        r"""
        Returns sanitized and normalized tensors ready for ASLV3FoundationModel.
        """
        # 0. One-Euro Adaptive Jitter Filter (Eliminates stationary sensor noise)
        filtered_raw = self.oe_filter.filter(raw_landmarks, timestamp)

        # 1. Perspective Pitch Normalization (Desk camera angle compensation)
        aligned_raw, pitch_deg = self.pitch_norm.normalize_pitch(filtered_raw)

        # 2. Handedness Disambiguation & Continuity Repair
        repaired_pts, was_swapped = self.achd.disambiguate_and_repair(aligned_raw)

        # 3. Bi-Acromial Metric Normalization (Depth & Zoom Invariance)
        norm_pts, shoulder_width = self.bamn.normalize(repaired_pts)

        # 4. Continuous Kinematics with Variable dt
        kinematics_9d = self.ttick.step(norm_pts, timestamp)  # [60, 9]

        # 5. Dominant Hand Dynamic Spatial Mirroring (Left-handed signer parity)
        norm_pts, kinematics_9d, is_left_dom = self.hand_mirror.update_and_mirror(norm_pts, kinematics_9d)

        # 6. Conversational Backchannel Check
        r_wrist_y = norm_pts[21, 1].item() if norm_pts.shape[0] > 21 else 0.0
        l_wrist_y = norm_pts[0, 1].item()
        hand_elev = max(r_wrist_y, l_wrist_y)
        hand_ke = torch.norm(kinematics_9d[21, 3:6]).item() + torch.norm(kinematics_9d[0, 3:6]).item()

        is_backchannel = self.bstg.evaluate_backchannel(
            cranial_pitch_velocity=cranial_pitch_vel,
            hand_elevation=hand_elev,
            hand_kinetic_energy=hand_ke,
        )

        return {
            "kinematics": kinematics_9d,         # [60, 9]
            "normalized_pts": norm_pts,          # [60, 3]
            "shoulder_width": shoulder_width,    # scalar float
            "was_hand_swapped": was_swapped,     # bool
            "is_left_dominant": is_left_dom,     # bool
            "camera_pitch_deg": pitch_deg,       # float
            "is_backchannel": is_backchannel,   # bool
        }
