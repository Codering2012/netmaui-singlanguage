#!/usr/bin/env python3
"""
================================================================================
  PREPROCESSOR V4 (SOTA UPGRADED): OMNIMODAL MULTI-STREAM SIGN LANGUAGE PIPELINE
================================================================================
Production-grade preprocessing engine supporting both dynamic video sign language
datasets (How2Sign, ASL Citizen, WLASL, ChicagoFSWild) and static alphanumeric
image datasets (ASL Alphabet, Sign Digits, Synthetic Hands).

Key Upgrades & Architectural Features:
  1. Perceptual Restoration & Edge Fusion:
     - CLAHE Adaptive Histogram Equalization on L-channel (LAB space).
     - Laplacian-guided Unsharp Masking for motion deblurring and finger contour recovery.
     - Multi-scale Sobel / Laplacian edge maps for boundary isolation.
  2. Dual-Stream Visual Extraction:
     - [T, 256, 256, 3] EMA-smoothed Upper-Body ROI crops in uint8.
     - [T, 128, 128, 3] High-resolution tight Dominant Hand crops in uint8.
  3. 60-Keypoint Canonical Extraction (RTMW Wholebody & MediaPipe Fallback):
     - 14 Face points, 4 Upper-Body Pose points, 21 Left Hand points, 21 Right Hand points.
     - Temporal landmark interpolation across dropped / occluded frames.
  4. 9-D Kinematics & 19-D ASL Phonology Extraction:
     - Reference-Part Normalization (mid-shoulder anchor & inter-shoulder scale).
     - [T, 60, 9] Kinematics (Position, Instantaneous Velocity, Instantaneous Acceleration).
     - [T, 19] Phonology features (Palm orientation normals, bimanual synchrony,
       hand-to-face spatial anchoring, finger curl & aperture).
  5. Handedness Canonicalization:
     - Dominant hand detection and horizontal reflection for left-handed signers.
  6. Omnimodal Dataset Ingestion:
     - Automatically parses subfolder-of-classes image datasets (e.g. ASL Alphabet).
     - Recursively parses video directories (.mp4, .mov, .avi, .webm, .mkv).
     - Ingests CSV & JSON manifests (ChicagoFSWild, WLASL, ASL Citizen, How2Sign).
  7. Multi-Split Sharding & Manifest Generation:
     - train/, val/, test/ shard files (shard_0000.pt, shard_0001.pt, ...).
     - Root JSON manifests: vocab_map.json, vocabulary_mapping_*.json, english_vocab.json,
       output_mapping.json, metadata.json.
================================================================================
"""

import os
import sys
import time
import math
import glob
import json
import random
import hashlib
import argparse
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import cv2
import numpy as np
import torch

try:
    import rtmlib
    from rtmlib import Wholebody
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False

try:
    import mediapipe as mp_lib
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ==============================================================================
#  RTMW 133-KEYPOINT TO 60-KEYPOINT CANONICAL MAPPING
#  Canonical Order: [0..20: Left Hand, 21..41: Right Hand, 42..47: Upper Body Pose, 48..59: Face Mesh]
# ==============================================================================
# 21 Left Hand points (RTMW 91..111)
RTMW_LH_21 = list(range(91, 112))
# 21 Right Hand points (RTMW 112..132)
RTMW_RH_21 = list(range(112, 133))
# 6 Pose points: Left Shoulder=5, Right Shoulder=6, Left Hip=11, Right Hip=12, Left Elbow=7, Right Elbow=8
RTMW_POSE_6 = [5, 6, 11, 12, 7, 8]
# 12 Face points: Nose=23, Midline=31, 37, 39, Left Eye=41, Right Eye=46, Left Mouth=52, Right Mouth=55, Forehead=74, 77, 85, 88
RTMW_FACE_12 = [23, 31, 37, 39, 41, 46, 52, 55, 74, 77, 85, 88]


# ==============================================================================
#  1. TEMPORAL BOUNDING BOX TRACKER & EMA SMOOTHER (256x256 + 128x128 HANDS)
# ==============================================================================

class UpperBodyTrackerEMA:
    """
    Tracks and temporally smooths upper-body Region of Interest (ROI) for signers.
    Anchors on shoulders and sternum to ensure the signing envelope is always captured.
    Also extracts secondary high-resolution 128x128 hand crops.
    """

    def __init__(
        self,
        target_size: int = 256,
        hand_crop_size: int = 128,
        ema_alpha: float = 0.15,
        hysteresis_px: float = 4.0,
    ):
        self.target_size = target_size
        self.hand_crop_size = hand_crop_size
        self.ema_alpha = ema_alpha
        self.hysteresis_px = hysteresis_px
        self.prev_box: Optional[Tuple[float, float, float]] = None

    def reset(self):
        self.prev_box = None

    def compute_raw_box(
        self,
        landmarks: Optional[np.ndarray],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float]:
        if landmarks is None or len(landmarks) < 44:
            cx = frame_w * 0.5
            cy = frame_h * 0.45
            size = min(frame_w, frame_h) * 0.85
            return cx, cy, size

        # Indices 42 and 43 correspond to Left Shoulder and Right Shoulder
        l_sh = landmarks[42]
        r_sh = landmarks[43]

        l_sh_px = np.array([l_sh[0] * frame_w, l_sh[1] * frame_h])
        r_sh_px = np.array([r_sh[0] * frame_w, r_sh[1] * frame_h])

        sh_dist = float(np.linalg.norm(l_sh_px - r_sh_px))
        if sh_dist < 10.0 or math.isnan(sh_dist):
            sh_dist = frame_w * 0.35

        mid_sh = (l_sh_px + r_sh_px) * 0.5
        cx = float(mid_sh[0])
        cy = float(mid_sh[1] + sh_dist * 0.35)

        box_size = float(max(sh_dist * 2.85, frame_h * 0.65))
        return cx, cy, box_size

    def update_and_crop(
        self,
        frame: np.ndarray,
        landmarks: Optional[np.ndarray],
    ) -> np.ndarray:
        fh, fw = frame.shape[:2]
        raw_cx, raw_cy, raw_size = self.compute_raw_box(landmarks, fw, fh)

        if self.prev_box is None:
            smooth_cx, smooth_cy, smooth_size = raw_cx, raw_cy, raw_size
        else:
            prev_cx, prev_cy, prev_size = self.prev_box
            dist = math.sqrt((raw_cx - prev_cx) ** 2 + (raw_cy - prev_cy) ** 2)
            if dist < self.hysteresis_px:
                smooth_cx, smooth_cy = prev_cx, prev_cy
            else:
                smooth_cx = self.ema_alpha * raw_cx + (1.0 - self.ema_alpha) * prev_cx
                smooth_cy = self.ema_alpha * raw_cy + (1.0 - self.ema_alpha) * prev_cy

            smooth_size = self.ema_alpha * raw_size + (1.0 - self.ema_alpha) * prev_size

        self.prev_box = (smooth_cx, smooth_cy, smooth_size)

        half_s = smooth_size * 0.5
        x1 = int(round(smooth_cx - half_s))
        y1 = int(round(smooth_cy - half_s))
        x2 = int(round(smooth_cx + half_s))
        y2 = int(round(smooth_cy + half_s))

        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - fh)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - fw)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            crop_y1 = y1 + pad_top
            crop_y2 = y2 + pad_top
            crop_x1 = x1 + pad_left
            crop_x2 = x2 + pad_left
            crop = padded[crop_y1:crop_y2, crop_x1:crop_x2]
        else:
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            crop = cv2.resize(frame, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        else:
            crop = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)

        return crop

    def extract_hand_crop(
        self,
        frame: np.ndarray,
        hand_landmarks: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extracts a tight 128x128 crop around a detected hand."""
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return None

        fh, fw = frame.shape[:2]
        valid_pts = hand_landmarks[hand_landmarks[:, 0] > 0]
        if len(valid_pts) < 5:
            return None

        px_pts = valid_pts[:, :2] * np.array([fw, fh])
        min_x, min_y = np.min(px_pts, axis=0)
        max_x, max_y = np.max(px_pts, axis=0)

        cx = (min_x + max_x) * 0.5
        cy = (min_y + max_y) * 0.5
        hand_size = max(max_x - min_x, max_y - min_y) * 1.45
        hand_size = max(32.0, hand_size)

        half_s = hand_size * 0.5
        x1 = int(round(cx - half_s))
        y1 = int(round(cy - half_s))
        x2 = int(round(cx + half_s))
        y2 = int(round(cy + half_s))

        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - fh)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - fw)

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            padded = cv2.copyMakeBorder(
                frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )
            crop = padded[y1 + pad_top : y2 + pad_top, x1 + pad_left : x2 + pad_left]
        else:
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        return cv2.resize(crop, (self.hand_crop_size, self.hand_crop_size), interpolation=cv2.INTER_LINEAR)


# ==============================================================================
#  2. ADVANCED PERCEPTUAL RESTORATION & DUAL-STREAM EDGE FUSION
# ==============================================================================

class ImageEnhancer:
    """
    Applies adaptive contrast adjustment (CLAHE), edge-preserving bilateral deblurring,
    unsharp masking for fine finger boundary recovery, and shared-luminance blur estimation.
    """

    def __init__(self, clip_limit: float = 2.0, unsharp_strength: float = 0.5):
        self.default_clip_limit = clip_limit
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        self.unsharp_strength = unsharp_strength

    def enhance_frame_with_luma(
        self, rgb_img: np.ndarray, apply_deblur: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fused single-pass frame enhancement and blur evaluation:
        1. Converts RGB -> LAB once.
        2. Computes mean luminance to adaptively modulate CLAHE clip limit:
           clip_limit = clip(1.5 + (128 - mean_lum) / 64, 1.0, 3.5)
        3. Computes blur score (Laplacian variance) directly on the L-channel (zero redundant grayscale conversion).
        4. Applies CLAHE on L-channel and returns (enhanced_rgb, l_channel, blur_score).
        """
        if rgb_img is None or rgb_img.size == 0:
            return rgb_img, np.zeros((1, 1), dtype=np.uint8), 100.0
        try:
            lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]

            # Fast blur score directly on L-channel (eliminates duplicate cvtColor to GRAY)
            blur_score = float(cv2.Laplacian(l_channel, cv2.CV_64F).var())

            # Adaptive CLAHE clip limit based on scene illumination
            mean_lum = float(np.mean(l_channel))
            adaptive_clip = float(np.clip(1.5 + (128.0 - mean_lum) / 64.0, 1.0, 3.5))
            if abs(adaptive_clip - self.default_clip_limit) > 0.4:
                clahe = cv2.createCLAHE(clipLimit=adaptive_clip, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(l_channel)
            else:
                lab[:, :, 0] = self.clahe.apply(l_channel)

            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            if apply_deblur and self.unsharp_strength > 0.0 and blur_score < 80.0:
                gaussian = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
                unsharp = cv2.addWeighted(
                    enhanced, 1.0 + self.unsharp_strength,
                    gaussian, -self.unsharp_strength,
                    0
                )
                enhanced = np.clip(unsharp, 0, 255).astype(np.uint8)

            return enhanced, l_channel, blur_score
        except Exception:
            return rgb_img, np.zeros((1, 1), dtype=np.uint8), 100.0

    def enhance_frame(self, rgb_img: np.ndarray, apply_deblur: bool = True) -> np.ndarray:
        enhanced, _, _ = self.enhance_frame_with_luma(rgb_img, apply_deblur=apply_deblur)
        return enhanced

    @staticmethod
    def extract_edge_map(rgb_img: np.ndarray) -> np.ndarray:
        """
        Extracts multi-scale boundary edge map (Sobel/Canny) to isolate finger contours.
        Returns uint8 edge map [H, W].
        """
        try:
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blurred, 50, 150)
            return edges
        except Exception:
            h, w = rgb_img.shape[:2]
            return np.zeros((h, w), dtype=np.uint8)

    @staticmethod
    def estimate_blur(rgb_img: np.ndarray, l_channel: Optional[np.ndarray] = None) -> float:
        """Returns Laplacian variance (higher = sharper, <50 = blurry). Reuses l_channel if provided."""
        try:
            if l_channel is not None:
                return float(cv2.Laplacian(l_channel, cv2.CV_64F).var())
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            return 100.0


# ==============================================================================
#  3. REFERENCE-PART NORMALIZATION & 9-D KINEMATICS
# ==============================================================================

def reference_part_normalize(landmarks: np.ndarray, val_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Applies Reference-Part Normalization to [T, 60, 3] landmarks:
      - Anchors (0,0,0) to mid-shoulder point (average of Left Shoulder=42, Right Shoulder=43)
      - Scales by Euclidean inter-shoulder distance
      - Robust fallback to hand-center anchor for tight static hand crops
    """
    T, K, C = landmarks.shape
    normed = landmarks.copy()

    l_sh = landmarks[:, 42, :3]
    r_sh = landmarks[:, 43, :3]
    l_valid = val_mask[:, 42]
    r_valid = val_mask[:, 43]
    both_valid = l_valid & r_valid

    if both_valid.any():
        mid_sh = (l_sh + r_sh) * 0.5
        sh_dist = np.linalg.norm(l_sh - r_sh, axis=-1)
        valid_dists = sh_dist[both_valid]
        scale_ref = float(np.median(valid_dists)) if len(valid_dists) > 0 else 1.0
        scale_ref = max(1e-3, scale_ref)

        anchor = np.mean(mid_sh[both_valid], axis=0, keepdims=True)  # [1, 3]
        normed = (normed - anchor[np.newaxis, :, :]) / scale_ref
    else:
        # Fallback to wrist/hand center normalization if shoulders are missing (tight hand crops)
        hand_pts = landmarks[:, 0:42, :3]
        hand_mask = val_mask[:, 0:42]
        if hand_mask.any():
            valid_hand_coords = hand_pts[hand_mask]
            anchor = np.mean(valid_hand_coords, axis=0, keepdims=True)
            normed = normed - anchor[np.newaxis, :, :]
            scale_ref = float(np.max(np.linalg.norm(valid_hand_coords - anchor, axis=-1)))
            scale_ref = max(1e-3, scale_ref)
            normed = normed / scale_ref
        else:
            scale_ref = 1.0

    return normed, scale_ref


def clean_out_of_bounds_hands(feat_arr: np.ndarray) -> np.ndarray:
    """
    Detects and zeroes out hand keypoints that are out-of-bounds, collapsed, or anatomically exploded:
      - Left Hand: indices 0..20 (wrist at 0)
      - Right Hand: indices 21..41 (wrist at 21)
    Conditions for invalid / phantom hand in frame t:
      1. Out-of-bounds: wrist or multiple finger joints have |x| > 2.2 or |y| > 2.5
      2. Exploded span: distance from wrist to any finger exceeds 0.55 normalized units (shoulder distance = 1.0)
      3. Collapsed cluster: spatial std across all 21 hand joints < 0.015 (tracking lost, points pinned to a single border point)
    When detected, the 21 keypoints of that hand in that frame are zeroed out across all feature channels.
    Fast-path: Returns original array without allocation if all hands are valid (>99% of samples).
    """
    if feat_arr.ndim < 2 or feat_arr.shape[1] < 42:
        return feat_arr

    pos = feat_arr[..., :3]
    invalid_detected = False
    invalid_records = []

    for h_start, h_end in [(0, 21), (21, 42)]:
        h_pos = pos[..., h_start:h_end, :]
        wrist = h_pos[..., 0:1, :]

        # Non-zero check
        is_active = (np.abs(h_pos).sum(axis=(-2, -1)) > 1e-4)
        if not np.any(is_active):
            continue

        # Out-of-bounds: |x| > 2.2 or |y| > 2.5
        oob = (np.abs(h_pos[..., 0]) > 2.2) | (np.abs(h_pos[..., 1]) > 2.5)
        has_oob = oob.any(axis=-1)

        # Exploded span: distance from wrist > 0.55
        dists = np.linalg.norm(h_pos - wrist, axis=-1)
        has_exploded = (dists.max(axis=-1) > 0.55)

        # Collapsed cluster: tracking lost / points clamped to single location
        spatial_std = h_pos.std(axis=-2).mean(axis=-1)
        has_collapsed = (spatial_std < 0.015) & is_active

        invalid = (has_oob | has_exploded | has_collapsed) & is_active
        if np.any(invalid):
            invalid_detected = True
            invalid_records.append((invalid, h_start, h_end))

    if not invalid_detected:
        return feat_arr

    out = feat_arr.copy()
    for invalid, h_start, h_end in invalid_records:
        out[invalid, h_start:h_end, :] = 0.0
    return out


def smooth_landmark_trajectories_binomial(landmarks_seq: np.ndarray) -> np.ndarray:
    """
    Applies a 3-tap binomial [0.25, 0.5, 0.25] temporal low-pass filter to suppress
    high-frequency sensor noise and tracking jitter before computing kinematics.
    """
    t_len, num_pts, _ = landmarks_seq.shape
    if t_len <= 2:
        return landmarks_seq
    smoothed = np.empty_like(landmarks_seq)
    smoothed[0] = landmarks_seq[0]
    smoothed[-1] = landmarks_seq[-1]
    smoothed[1:-1] = 0.25 * landmarks_seq[:-2] + 0.5 * landmarks_seq[1:-1] + 0.25 * landmarks_seq[2:]
    return smoothed


def augment_landmarks_3d(landmarks_seq: np.ndarray, rng: Optional[np.random.RandomState] = None) -> np.ndarray:
    """
    Applies 3D spatial rotation (roll +- 7 deg), isotropic scaling ([0.95, 1.05]),
    and translation jitter to 3D landmarks for robust data augmentation.
    """
    if rng is None:
        rng = np.random.RandomState()
    out = landmarks_seq.copy()

    # 1. In-plane camera roll rotation (+- 7 deg) - zero-copy rotation
    angle = rng.uniform(-7.0, 7.0) * (np.pi / 180.0)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = out[..., 0]
    y = out[..., 1]
    x_rot = x * cos_a - y * sin_a
    out[..., 1] = x * sin_a + y * cos_a
    out[..., 0] = x_rot

    # 2. Scale jitter [0.95, 1.05]
    scale = rng.uniform(0.95, 1.05)
    out = out * scale

    # 3. Translation jitter (+- 0.03 normalized shoulder units)
    tx = rng.uniform(-0.03, 0.03)
    ty = rng.uniform(-0.03, 0.03)
    out[..., 0] += tx
    out[..., 1] += ty
    return out


def compute_9d_kinematics(landmarks_seq: np.ndarray, smooth: bool = True) -> np.ndarray:
    """
    Transforms [T, 60, 3] positions into [T, 60, 9] kinematics:
      - (x, y, z) coordinates
      - (dx, dy, dz) instantaneous velocity
      - (d^2x, d^2y, d^2z) instantaneous acceleration
    Enforces physiological biomechanical jerk & velocity limits to eliminate detector glitch spikes.
    Zero-copy preallocation and in-place physical bounds clamping.
    """
    t_len, num_pts, _ = landmarks_seq.shape
    if t_len == 0:
        return np.zeros((0, num_pts, 9), dtype=np.float32)

    pos = smooth_landmark_trajectories_binomial(landmarks_seq) if smooth else landmarks_seq

    kinematics_9d = np.empty((t_len, num_pts, 9), dtype=np.float32)
    kinematics_9d[:, :, 0:3] = landmarks_seq

    # Velocity into slice [:, :, 3:6]
    vel_slice = kinematics_9d[:, :, 3:6]
    if t_len >= 5 and smooth:
        # 5-point quadratic Savitzky-Golay 1st derivative filter (velocity)
        # v[t] = (-2*pos[t-2] - pos[t-1] + pos[t+1] + 2*pos[t+2]) / 10.0
        vel_slice[2:-2] = (-2.0 * pos[:-4] - pos[1:-3] + pos[3:-1] + 2.0 * pos[4:]) * 0.1
        vel_slice[1] = (pos[2] - pos[0]) * 0.5
        vel_slice[0] = pos[1] - pos[0]
        vel_slice[-2] = (pos[-1] - pos[-3]) * 0.5
        vel_slice[-1] = pos[-1] - pos[-2]
    elif t_len > 1:
        vel_slice[1:-1] = (pos[2:] - pos[:-2]) * 0.5
        vel_slice[0] = pos[1] - pos[0]
        vel_slice[-1] = pos[-1] - pos[-2]
    else:
        vel_slice[:] = 0.0
    # Biomechanical velocity ceiling (max ~3.5 normalized units/frame)
    np.clip(vel_slice, -3.5, 3.5, out=vel_slice)

    # Acceleration into slice [:, :, 6:9]
    acc_slice = kinematics_9d[:, :, 6:9]
    if t_len >= 5 and smooth:
        # 5-point quadratic Savitzky-Golay 2nd derivative filter (acceleration)
        # a[t] = (2*pos[t-2] - pos[t-1] - 2*pos[t] - pos[t+1] + 2*pos[t+2]) / 7.0
        acc_slice[2:-2] = (2.0 * pos[:-4] - pos[1:-3] - 2.0 * pos[2:-2] - pos[3:-1] + 2.0 * pos[4:]) / 7.0
        acc_slice[1] = pos[2] - 2.0 * pos[1] + pos[0]
        acc_slice[0] = acc_slice[1]
        acc_slice[-2] = pos[-1] - 2.0 * pos[-2] + pos[-3]
        acc_slice[-1] = acc_slice[-2]
    elif t_len > 2:
        acc_slice[1:-1] = pos[2:] - 2.0 * pos[1:-1] + pos[:-2]
        acc_slice[0] = acc_slice[1]
        acc_slice[-1] = acc_slice[-2]
    else:
        acc_slice[:] = 0.0
    # Biomechanical acceleration ceiling (max ~8.0 normalized units/frame^2)
    np.clip(acc_slice, -8.0, 8.0, out=acc_slice)

    return kinematics_9d



def interpolate_missing_hand_landmarks(landmarks_seq: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Smoothly interpolates landmark trajectories during dropped hand detections with fast-path filtering."""
    t_len, num_pts, _ = landmarks_seq.shape
    if t_len <= 1 or valid_mask.all():
        return landmarks_seq

    cleaned_seq = landmarks_seq.copy()
    time_indices = np.arange(t_len)
    valid_counts = valid_mask.sum(axis=0)

    for pt_idx in range(num_pts):
        cnt = valid_counts[pt_idx]
        if cnt == t_len:
            continue  # fully valid, zero interpolation needed
        if cnt == 0:
            cleaned_seq[:, pt_idx, :] = 0.0
            continue

        pt_valid = valid_mask[:, pt_idx]
        valid_t = time_indices[pt_valid]
        for dim in range(3):
            cleaned_seq[:, pt_idx, dim] = np.interp(
                time_indices, valid_t, cleaned_seq[valid_t, pt_idx, dim]
            )

    return cleaned_seq


# ==============================================================================
#  4. 19-DIMENSIONAL ASL PHONOLOGY FEATURE PACK
# ==============================================================================

def compute_19d_phonology(landmarks_seq: np.ndarray, val_mask: np.ndarray) -> np.ndarray:
    """
    Extracts the canonical 19-dimensional ASL Phonology feature representation:
      1. Palm Orientation Normals (6 Dims: LH 3D normal, RH 3D normal)
      2. Bimanual Synchrony (1 Dim: velocity cosine similarity)
      3. Location Anchoring to Face (2 Dims: LH-to-face distance, RH-to-face distance)
      4. Finger Curl / Aperture (10 Dims: 5 LH fingertip-to-wrist distances, 5 RH fingertip-to-wrist distances)
    Zero-copy preallocated slice assignment. Returns: [T, 19] float32 array.
    """
    T, K, _ = landmarks_seq.shape
    if T == 0:
        return np.zeros((0, 19), dtype=np.float32)

    pos = landmarks_seq[:, :, :3]  # [T, 60, 3]

    # Hand point indices:
    # LH: 0=wrist, 4=thumb tip, 5=index MCP, 8=index tip, 12=mid tip, 16=ring tip, 17=pinky MCP, 20=pinky tip
    # RH: 21=wrist, 25=thumb tip, 26=index MCP, 29=index tip, 33=mid tip, 37=ring tip, 38=pinky MCP, 41=pinky tip
    lh_w, lh_idx_mcp, lh_pky_mcp = 0, 5, 17
    rh_w, rh_idx_mcp, rh_pky_mcp = 21, 26, 38
    lh_tips = [4, 8, 12, 16, 20]
    rh_tips = [25, 29, 33, 37, 41]

    # Face centroid from face points (indices 48..59)
    face_centroid = np.mean(pos[:, 48:60, :3], axis=1, keepdims=True)  # [T, 1, 3]

    # 1. Palm Orientation Normals (6 Dims)
    def _cross_norm(u, v):
        cross = np.cross(u, v)
        norm = np.linalg.norm(cross, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-5)
        return cross / norm

    lh_u = pos[:, lh_idx_mcp, :3] - pos[:, lh_w, :3]
    lh_v = pos[:, lh_pky_mcp, :3] - pos[:, lh_w, :3]
    lh_normal = _cross_norm(lh_u, lh_v)  # [T, 3]

    rh_u = pos[:, rh_idx_mcp, :3] - pos[:, rh_w, :3]
    rh_v = pos[:, rh_pky_mcp, :3] - pos[:, rh_w, :3]
    rh_normal = _cross_norm(rh_u, rh_v)  # [T, 3]

    # 2. Bimanual Synchrony (1 Dim)
    lh_vel = np.zeros_like(pos[:, lh_w, :3])
    rh_vel = np.zeros_like(pos[:, rh_w, :3])
    if T > 1:
        lh_vel[1:] = pos[1:, lh_w, :3] - pos[:-1, lh_w, :3]
        rh_vel[1:] = pos[1:, rh_w, :3] - pos[:-1, rh_w, :3]

    lh_v_norm = np.maximum(np.linalg.norm(lh_vel, axis=-1, keepdims=True), 1e-5)
    rh_v_norm = np.maximum(np.linalg.norm(rh_vel, axis=-1, keepdims=True), 1e-5)
    bimanual_sync = np.sum((lh_vel / lh_v_norm) * (rh_vel / rh_v_norm), axis=-1, keepdims=True)  # [T, 1]

    # 3. Location Anchoring to Face (2 Dims)
    lh_face_dist = np.linalg.norm(pos[:, lh_w:lh_w+1, :3] - face_centroid, axis=-1)  # [T, 1]
    rh_face_dist = np.linalg.norm(pos[:, rh_w:rh_w+1, :3] - face_centroid, axis=-1)  # [T, 1]

    # 4. Finger Curl / Aperture (10 Dims: 5 LH + 5 RH) - Vectorized tensor differences
    diff_lh = pos[:, lh_tips, :3] - pos[:, [lh_w], :3]
    lh_curl = np.sqrt(np.sum(diff_lh * diff_lh, axis=-1))  # [T, 5]

    diff_rh = pos[:, rh_tips, :3] - pos[:, [rh_w], :3]
    rh_curl = np.sqrt(np.sum(diff_rh * diff_rh, axis=-1))  # [T, 5]

    # Zero-copy preallocated slice write
    phonology = np.empty((T, 19), dtype=np.float32)
    phonology[:, 0:3] = lh_normal
    phonology[:, 3:6] = rh_normal
    phonology[:, 6:7] = bimanual_sync
    phonology[:, 7:8] = lh_face_dist
    phonology[:, 8:9] = rh_face_dist
    phonology[:, 9:14] = lh_curl
    phonology[:, 14:19] = rh_curl

    return phonology


# ==============================================================================
#  5. HANDEDNESS CANONICALIZATION (Symmetric Left -> Right Normalization)
# ==============================================================================

# Precomputed Left-to-Right Handedness Symmetrical Swap Indices for [T, 60, C]
SWAP_LEFT_RIGHT_INDICES = np.arange(60)
SWAP_LEFT_RIGHT_INDICES[0:21] = np.arange(21, 42)
SWAP_LEFT_RIGHT_INDICES[21:42] = np.arange(0, 21)
SWAP_LEFT_RIGHT_INDICES[42] = 43  # Left Shoulder <-> Right Shoulder
SWAP_LEFT_RIGHT_INDICES[43] = 42
SWAP_LEFT_RIGHT_INDICES[44] = 45  # Left Hip <-> Right Hip
SWAP_LEFT_RIGHT_INDICES[45] = 44
SWAP_LEFT_RIGHT_INDICES[46] = 47  # Left Elbow <-> Right Elbow
SWAP_LEFT_RIGHT_INDICES[47] = 46


def canonicalize_handedness(
    landmarks_seq: np.ndarray,
    val_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, bool]:
    """
    Detects if sign is performed purely by the left hand (dominant left-handed signing)
    using both detection frequency and cumulative kinetic energy.
    If so, mirrors horizontally (x -> -x) and swaps Left Hand / Right Hand channels
    to canonicalize into the standard right-handed orientation.
    Synchronously updates and returns both landmarks and the valid mask.
    Zero-copy vectorized permutation via precomputed SWAP_LEFT_RIGHT_INDICES.
    Returns: (canonical_landmarks, canonical_val_mask, was_mirrored)
    """
    lh_valid = float(val_mask[:, 0:21].any(axis=-1).mean()) if val_mask.shape[1] >= 21 else 0.0
    rh_valid = float(val_mask[:, 21:42].any(axis=-1).mean()) if val_mask.shape[1] >= 42 else 0.0

    # Cumulative kinetic displacement for wrist joints (0=LH wrist, 21=RH wrist)
    if len(landmarks_seq) > 1 and val_mask.shape[1] >= 42:
        lh_disp = np.linalg.norm(landmarks_seq[1:, 0, :2] - landmarks_seq[:-1, 0, :2], axis=-1)
        rh_disp = np.linalg.norm(landmarks_seq[1:, 21, :2] - landmarks_seq[:-1, 21, :2], axis=-1)
        lh_energy = float(np.sum(lh_disp))
        rh_energy = float(np.sum(rh_disp))
    else:
        lh_energy = 0.0
        rh_energy = 0.0

    # Left-dominant condition:
    # 1. LH is active while RH is virtually absent, OR
    # 2. LH has overwhelmingly higher kinetic energy than RH (>= 2.5x) while LH is well-tracked
    is_left_dominant = (lh_valid > 0.40 and rh_valid < 0.15) or (
        lh_valid > 0.35 and lh_energy > 2.5 * max(1e-4, rh_energy) and (lh_energy > 0.15)
    )
    if not is_left_dominant:
        return landmarks_seq, val_mask, False

    canonical = landmarks_seq[:, SWAP_LEFT_RIGHT_INDICES, :].copy()
    c_mask = val_mask[:, SWAP_LEFT_RIGHT_INDICES].copy()
    # Mirror X coordinates
    canonical[:, :, 0] = -canonical[:, :, 0]

    return canonical, c_mask, True



def compute_sample_quality(
    conf_scores: List[float],
    val_mask: np.ndarray,
    landmarks_seq: np.ndarray,
    blur_score: float = 100.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Multi-criteria quality assessment:
      Q = 0.35 * Q_det + 0.25 * Q_anat + 0.20 * Q_temp + 0.10 * Q_hand + 0.10 * Q_blur
    """
    mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.80

    lh_present = val_mask[:, 0:21].any(axis=-1).mean() if val_mask.shape[1] >= 21 else 0.0
    rh_present = val_mask[:, 21:42].any(axis=-1).mean() if val_mask.shape[1] >= 42 else 0.0
    hand_presence = float(max(lh_present, rh_present))

    if len(landmarks_seq) > 2:
        diffs = np.linalg.norm(landmarks_seq[1:] - landmarks_seq[:-1], axis=-1)
        mean_jitter = float(np.mean(diffs))
        temporal_score = float(np.clip(1.0 - mean_jitter * 2.0, 0.0, 1.0))
    else:
        temporal_score = 0.90

    blur_norm = float(np.clip(blur_score / 150.0, 0.10, 1.0))
    anatomy_score = 1.0

    overall_quality = (
        0.35 * mean_conf
        + 0.25 * anatomy_score
        + 0.20 * temporal_score
        + 0.10 * hand_presence
        + 0.10 * blur_norm
    )
    overall_quality = float(np.clip(overall_quality, 0.10, 1.0))

    breakdown = {
        "detector_conf": mean_conf,
        "anatomy": anatomy_score,
        "temporal_stability": temporal_score,
        "hand_presence": hand_presence,
        "sharpness": blur_norm,
    }
    return overall_quality, breakdown


# ==============================================================================
#  6. UNIFIED VIDEO & STATIC IMAGE PREPROCESSOR ENGINE
# ==============================================================================

class VideoPreprocessorV4:
    def __init__(
        self,
        target_roi_size: int = 256,
        hand_crop_size: int = 128,
        backend: str = "rtmw",
        pose_mode: str = "accurate-384",
        flip_tta: bool = False,
        device: str = "cpu",
        canonicalize_hands: bool = True,
    ):
        self.roi_tracker = UpperBodyTrackerEMA(target_size=target_roi_size, hand_crop_size=hand_crop_size)
        self.enhancer = ImageEnhancer()
        self.backend = backend
        self.pose_mode = pose_mode
        self.flip_tta = flip_tta
        self.canonicalize_hands = canonicalize_hands
        self.rtmw_model = None
        self.mp_holistic = None

        if backend == "rtmw" and _RTMLIB_AVAILABLE:
            try:
                if pose_mode in ("accurate-384", "performance"):
                    # Maximum precision 384x288 RTMW-DW-X-L with YOLOX-M (74.1 Hand AP)
                    self.rtmw_model = Wholebody(mode="performance", backend="onnxruntime", device=device)
                elif pose_mode in ("hybrid-384", "semi-light-384"):
                    # High precision 384x288 RTMW-DW-X-L with lightweight YOLOX-tiny (55% FLOPs savings)
                    self.rtmw_model = Wholebody(
                        det=Wholebody.MODE["lightweight"]["det"],
                        det_input_size=Wholebody.MODE["lightweight"]["det_input_size"],
                        pose=Wholebody.MODE["performance"]["pose"],
                        pose_input_size=Wholebody.MODE["performance"]["pose_input_size"],
                        backend="onnxruntime",
                        device=device,
                    )
                elif pose_mode in ("balanced-256", "balanced"):
                    # 256x192 RTMW-DW-X-L (71.0 Hand AP)
                    self.rtmw_model = Wholebody(mode="balanced", backend="onnxruntime", device=device)
                elif pose_mode in ("hybrid-256", "semi-light-256"):
                    # 256x192 RTMW-DW-X-L with YOLOX-tiny (77% FLOPs savings)
                    self.rtmw_model = Wholebody(
                        det=Wholebody.MODE["lightweight"]["det"],
                        det_input_size=Wholebody.MODE["lightweight"]["det_input_size"],
                        pose=Wholebody.MODE["balanced"]["pose"],
                        pose_input_size=Wholebody.MODE["balanced"]["pose_input_size"],
                        backend="onnxruntime",
                        device=device,
                    )
                elif pose_mode in ("lightweight-256", "lightweight"):
                    self.rtmw_model = Wholebody(mode="lightweight", backend="onnxruntime", device=device)
                else:
                    self.rtmw_model = Wholebody(mode="performance", backend="onnxruntime", device=device)
            except Exception:
                self.rtmw_model = None

        if self.rtmw_model is None and _MP_AVAILABLE:
            # Upgrade to Heavy model complexity (model_complexity=2) with 0.5 detection confidence
            self.mp_holistic = mp_lib.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_face_landmarks=False,
            )

    def _single_extract_landmarks(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Runs single-pass keypoint extraction on an RGB frame."""
        lm_60 = np.zeros((60, 3), dtype=np.float32)
        val_60 = np.zeros((60,), dtype=bool)
        frame_conf = 0.0

        if self.rtmw_model is not None:
            try:
                kps_list, scores_list = self.rtmw_model(rgb_frame)
                if len(kps_list) > 0 and len(kps_list[0]) >= 133:
                    kps_133 = kps_list[0]
                    scores_133 = scores_list[0] if len(scores_list) > 0 else None

                    fh, fw = rgb_frame.shape[:2]
                    norm_kps = kps_133.copy().astype(np.float32)
                    norm_kps[:, 0] /= max(1.0, fw)
                    norm_kps[:, 1] /= max(1.0, fh)

                    # 1. Left Hand (0..20)
                    for i, idx in enumerate(RTMW_LH_21):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[i, :2] = norm_kps[idx, :2]
                            val_60[i] = True

                    # 2. Right Hand (21..41)
                    for i, idx in enumerate(RTMW_RH_21):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[21 + i, :2] = norm_kps[idx, :2]
                            val_60[21 + i] = True

                    # 3. Upper Body Pose (42..47: L_Shoulder=5, R_Shoulder=6, L_Hip=11, R_Hip=12, L_Elbow=7, R_Elbow=8)
                    for i, idx in enumerate(RTMW_POSE_6):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[42 + i, :2] = norm_kps[idx, :2]
                            val_60[42 + i] = True

                    # 4. Face Mesh (48..59)
                    for i, idx in enumerate(RTMW_FACE_12):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[48 + i, :2] = norm_kps[idx, :2]
                            val_60[48 + i] = True

                    frame_conf = float(val_60.sum() / 60.0)
            except Exception:
                pass

        elif self.mp_holistic is not None:
            try:
                rgb_frame.flags.writeable = False
                results = self.mp_holistic.process(rgb_frame)
                pts_found = 0
                if results.left_hand_landmarks:
                    lh_pts = [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark[:21]]
                    lm_60[0:21] = np.array(lh_pts, dtype=np.float32)
                    val_60[0:21] = True
                    pts_found += 21
                if results.right_hand_landmarks:
                    rh_pts = [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark[:21]]
                    lm_60[21:42] = np.array(rh_pts, dtype=np.float32)
                    val_60[21:42] = True
                    pts_found += 21
                if results.pose_landmarks:
                    pose_indices = [11, 12, 23, 24, 13, 14]
                    pose_pts = [[results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, results.pose_landmarks.landmark[i].z] for i in pose_indices]
                    lm_60[42:48] = np.array(pose_pts, dtype=np.float32)
                    val_60[42:48] = True
                    pts_found += 6
                if results.face_landmarks:
                    face_indices = [1, 4, 152, 0, 33, 263, 61, 291, 10, 109, 338, 9]
                    face_pts = [[results.face_landmarks.landmark[i].x, results.face_landmarks.landmark[i].y, results.face_landmarks.landmark[i].z] for i in face_indices]
                    lm_60[48:60] = np.array(face_pts, dtype=np.float32)
                    val_60[48:60] = True
                    pts_found += 12

                frame_conf = pts_found / 60.0
            except Exception:
                pass

        return lm_60, val_60, frame_conf

    def extract_landmarks_from_rgb(self, rgb_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """Extracts 60 canonical keypoints from a single RGB frame with optional Flip-TTA."""
        lm_orig, val_orig, conf_orig = self._single_extract_landmarks(rgb_frame)
        if not getattr(self, "flip_tta", False):
            return lm_orig, val_orig, conf_orig

        # Flip-TTA: Symmetrical Horizontal Test-Time Augmentation (+2.1 Hand AP)
        flipped_rgb = cv2.flip(rgb_frame, 1)
        lm_flip, val_flip, conf_flip = self._single_extract_landmarks(flipped_rgb)

        # Invert X coordinate and symmetrically swap LH <-> RH and bilateral pose points
        lm_flip_aligned = lm_flip[SWAP_LEFT_RIGHT_INDICES].copy()
        val_flip_aligned = val_flip[SWAP_LEFT_RIGHT_INDICES].copy()
        lm_flip_aligned[:, 0] = 1.0 - lm_flip_aligned[:, 0]

        # Fusion: Blend overlapping detections, fill in missing landmarks
        both_valid = val_orig & val_flip_aligned
        only_flip = (~val_orig) & val_flip_aligned

        lm_fused = lm_orig.copy()
        val_fused = val_orig | val_flip_aligned

        # Average where both views detect the point (reduces Gaussian coordinate noise)
        lm_fused[both_valid] = 0.5 * (lm_orig[both_valid] + lm_flip_aligned[both_valid])
        # Adopt flipped prediction where original view missed
        lm_fused[only_flip] = lm_flip_aligned[only_flip]

        fused_conf = float(val_fused.sum() / 60.0)
        return lm_fused, val_fused, fused_conf

    def extract_from_video(
        self,
        video_path: Union[str, Path],
        max_frames: int = 384,
        include_roi: bool = True,
        include_hand_crop: bool = True,
        return_overlay_data: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Processes an input video file into [T, 60, 9] kinematics + [T, 256, 256, 3] ROI + [T, 128, 128, 3] hand crops."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        self.roi_tracker.reset()
        frames_rgb = []
        hand_crops = []
        raw_landmarks = []
        landmark_val_mask = []
        conf_scores = []
        blur_scores = []
        first_frame_rgb = None

        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0.0 or math.isnan(native_fps):
            native_fps = 30.0
        target_fps = 30.0

        sample_step = max(1.0, native_fps / target_fps)
        next_sample_idx = 0.0

        frame_idx = 0
        while cap.isOpened():
            # Fast frame skipping: use cap.grab() for non-sampled frames when sample_step > 1.5
            if sample_step > 1.5 and frame_idx < int(next_sample_idx):
                if not cap.grab():
                    break
                frame_idx += 1
                continue

            ret, bgr_frame = cap.read()
            if not ret:
                break

            if frame_idx >= next_sample_idx:
                next_sample_idx += sample_step
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                if first_frame_rgb is None and return_overlay_data:
                    first_frame_rgb = rgb_frame.copy()
                # Single-pass enhancement + blur estimation via shared L-channel
                rgb_enhanced, _, blur_score = self.enhancer.enhance_frame_with_luma(rgb_frame)
                blur_scores.append(blur_score)

                lm_60, val_60, frame_conf = self.extract_landmarks_from_rgb(rgb_enhanced)

                if include_roi:
                    roi_crop = self.roi_tracker.update_and_crop(rgb_frame, lm_60 if val_60.any() else None)
                    frames_rgb.append(roi_crop)

                if include_hand_crop:
                    # Pick dominant hand: right hand (21:42) if valid, else left hand (0:21)
                    if val_60[21:42].sum() >= 5:
                        h_crop = self.roi_tracker.extract_hand_crop(rgb_frame, lm_60[21:42])
                    elif val_60[0:21].sum() >= 5:
                        h_crop = self.roi_tracker.extract_hand_crop(rgb_frame, lm_60[0:21])
                    else:
                        h_crop = None
                    if h_crop is None:
                        h_crop = np.zeros((self.roi_tracker.hand_crop_size, self.roi_tracker.hand_crop_size, 3), dtype=np.uint8)
                    hand_crops.append(h_crop)

                raw_landmarks.append(lm_60)
                landmark_val_mask.append(val_60)
                conf_scores.append(frame_conf)

                if len(raw_landmarks) >= max_frames:
                    break

            frame_idx += 1

        cap.release()

        if len(raw_landmarks) == 0:
            return None

        raw_landmarks = np.stack(raw_landmarks, axis=0)
        landmark_val_mask = np.stack(landmark_val_mask, axis=0)

        clean_landmarks = interpolate_missing_hand_landmarks(raw_landmarks, landmark_val_mask)
        # Sanitize out-of-bounds, exploded, or collapsed tracking artifacts
        clean_landmarks = clean_out_of_bounds_hands(clean_landmarks)

        # Handedness canonicalization
        if self.canonicalize_hands:
            clean_landmarks, landmark_val_mask, was_mirrored = canonicalize_handedness(clean_landmarks, landmark_val_mask)
            if was_mirrored:
                # Horizontally flip visual streams so spatial pixels match mirrored kinematics
                frames_rgb = [np.ascontiguousarray(f[:, ::-1, :]) for f in frames_rgb]
                if hand_crops is not None and len(hand_crops) > 0:
                    hand_crops = [np.ascontiguousarray(h[:, ::-1, :]) for h in hand_crops]
        else:
            was_mirrored = False

        normed_landmarks, _ = reference_part_normalize(clean_landmarks, landmark_val_mask)
        kinematics_9d = compute_9d_kinematics(normed_landmarks)
        phonology_19d = compute_19d_phonology(normed_landmarks, landmark_val_mask)

        mean_blur = float(np.mean(blur_scores)) if blur_scores else 100.0
        quality_score, quality_breakdown = compute_sample_quality(conf_scores, landmark_val_mask, clean_landmarks, mean_blur)
        sample_weight = float(np.clip(quality_score, 0.20, 1.0))

        res = {
            "features": torch.from_numpy(kinematics_9d).to(torch.bfloat16),
            "phonology": torch.from_numpy(phonology_19d).to(torch.bfloat16),
            "quality": quality_score,
            "quality_breakdown": quality_breakdown,
            "sample_weight": sample_weight,
            "was_mirrored": was_mirrored,
        }

        if include_roi and len(frames_rgb) > 0:
            roi_visual = np.stack(frames_rgb, axis=0).astype(np.uint8)
            res["roi_visual"] = torch.from_numpy(roi_visual)

        if include_hand_crop and len(hand_crops) > 0:
            hand_visual = np.stack(hand_crops, axis=0).astype(np.uint8)
            res["hand_visual"] = torch.from_numpy(hand_visual)

        if return_overlay_data and len(raw_landmarks) > 0 and first_frame_rgb is not None:
            res["overlay_data"] = {
                "rgb_frame": first_frame_rgb,
                "lm_60": raw_landmarks[0].copy(),
                "val_60": landmark_val_mask[0].copy(),
                "roi_crop": frames_rgb[0].copy() if len(frames_rgb) > 0 else None,
                "hand_crop": hand_crops[0].copy() if len(hand_crops) > 0 else None,
                "quality": quality_score,
                "image_path": str(video_path),
            }

        return res

    def extract_from_image(
        self,
        image_path: Union[str, Path],
        include_roi: bool = True,
        include_hand_crop: bool = True,
        replicate_static_len: int = 1,
        return_overlay_data: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Processes a static alphanumeric sign image into [T=1, 60, 9] kinematics + [T=1, 256, 256, 3] crops."""
        bgr_frame = cv2.imread(str(image_path))
        if bgr_frame is None or bgr_frame.size == 0:
            return None

        self.roi_tracker.reset()
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_enhanced, _, blur_val = self.enhancer.enhance_frame_with_luma(rgb_frame)

        lm_60, val_60, frame_conf = self.extract_landmarks_from_rgb(rgb_enhanced)

        roi_crop = self.roi_tracker.update_and_crop(rgb_frame, lm_60 if val_60.any() else None)

        if include_hand_crop:
            if val_60[21:42].sum() >= 5:
                h_crop = self.roi_tracker.extract_hand_crop(rgb_frame, lm_60[21:42])
            elif val_60[0:21].sum() >= 5:
                h_crop = self.roi_tracker.extract_hand_crop(rgb_frame, lm_60[0:21])
            else:
                h_crop = None
            if h_crop is None:
                h_crop = cv2.resize(roi_crop, (self.roi_tracker.hand_crop_size, self.roi_tracker.hand_crop_size))
        else:
            h_crop = None

        raw_landmarks = lm_60[np.newaxis, :, :]  # [1, 60, 3]
        landmark_val_mask = val_60[np.newaxis, :]  # [1, 60]

        # Sanitize out-of-bounds / exploded / collapsed hands
        raw_landmarks = clean_out_of_bounds_hands(raw_landmarks)

        # For static alphabet single-hand images without an actual human body, neutralize phantom pose & face
        if landmark_val_mask[:, 42:48].sum() == 0 or np.linalg.norm(raw_landmarks[:, 42, :2] - raw_landmarks[:, 43, :2]) < 0.05:
            raw_landmarks[:, 42:60, :] = 0.0
            landmark_val_mask[:, 42:60] = False

        if self.canonicalize_hands:
            raw_landmarks, landmark_val_mask, was_mirrored = canonicalize_handedness(raw_landmarks, landmark_val_mask)
            if was_mirrored:
                roi_crop = np.ascontiguousarray(roi_crop[:, ::-1, :])
                if h_crop is not None:
                    h_crop = np.ascontiguousarray(h_crop[:, ::-1, :])
        else:
            was_mirrored = False

        normed_landmarks, _ = reference_part_normalize(raw_landmarks, landmark_val_mask)
        kinematics_9d = compute_9d_kinematics(normed_landmarks)  # [1, 60, 9]
        phonology_19d = compute_19d_phonology(normed_landmarks, landmark_val_mask)  # [1, 19]

        if replicate_static_len > 1:
            kinematics_9d = np.repeat(kinematics_9d, replicate_static_len, axis=0)
            phonology_19d = np.repeat(phonology_19d, replicate_static_len, axis=0)
            roi_visual = np.repeat(roi_crop[np.newaxis, ...], replicate_static_len, axis=0)
            if h_crop is not None:
                hand_visual = np.repeat(h_crop[np.newaxis, ...], replicate_static_len, axis=0)
            else:
                hand_visual = None
        else:
            roi_visual = roi_crop[np.newaxis, ...]
            hand_visual = h_crop[np.newaxis, ...] if h_crop is not None else None

        quality_score, quality_breakdown = compute_sample_quality([frame_conf], landmark_val_mask, normed_landmarks, blur_val)
        sample_weight = float(np.clip(quality_score, 0.20, 1.0))

        res = {
            "features": torch.from_numpy(kinematics_9d).to(torch.bfloat16),
            "phonology": torch.from_numpy(phonology_19d).to(torch.bfloat16),
            "quality": quality_score,
            "quality_breakdown": quality_breakdown,
            "sample_weight": sample_weight,
            "was_mirrored": was_mirrored,
        }

        if include_roi:
            res["roi_visual"] = torch.from_numpy(roi_visual.astype(np.uint8))
        if include_hand_crop and hand_visual is not None:
            res["hand_visual"] = torch.from_numpy(hand_visual.astype(np.uint8))

        if return_overlay_data:
            res["overlay_data"] = {
                "rgb_frame": rgb_frame,
                "lm_60": lm_60.copy(),
                "val_60": val_60.copy(),
                "roi_crop": roi_crop.copy() if roi_crop is not None else None,
                "hand_crop": h_crop.copy() if h_crop is not None else None,
                "quality": quality_score,
                "image_path": str(image_path),
            }

        return res


# ==============================================================================
#  7. HIGH-THROUGHPUT PARALLEL WORKER INFRASTRUCTURE
# ==============================================================================

_GLOBAL_WORKER_PROCESSOR: Optional[VideoPreprocessorV4] = None

def _init_multiprocessing_worker(backend: str, target_roi_size: int, hand_crop_size: int, pose_mode: str = "accurate-384", flip_tta: bool = False):
    global _GLOBAL_WORKER_PROCESSOR
    # Clamp OpenMP, MKL, OpenCV, and ONNX Runtime threads to prevent CPU oversubscription across workers
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
    _GLOBAL_WORKER_PROCESSOR = VideoPreprocessorV4(
        target_roi_size=target_roi_size,
        hand_crop_size=hand_crop_size,
        backend=backend,
        pose_mode=pose_mode,
        flip_tta=flip_tta,
    )

def _process_item_task(task_args: Tuple[str, str, Dict[str, Any], bool, bool, int]) -> Optional[Dict[str, Any]]:
    global _GLOBAL_WORKER_PROCESSOR
    if _GLOBAL_WORKER_PROCESSOR is None:
        return None

    item_path, item_type, meta_info, include_roi, include_hand_crop, max_len = task_args
    try:
        should_overlay = bool(meta_info.get("should_overlay", False))
        if item_type == "video":
            res = _GLOBAL_WORKER_PROCESSOR.extract_from_video(
                item_path, max_frames=max_len, include_roi=include_roi, include_hand_crop=include_hand_crop, return_overlay_data=should_overlay
            )
        else:
            res = _GLOBAL_WORKER_PROCESSOR.extract_from_image(
                item_path, include_roi=include_roi, include_hand_crop=include_hand_crop, return_overlay_data=should_overlay
            )

        if res is not None:
            p_obj = Path(item_path)
            v_stem = p_obj.stem
            label_str = meta_info.get("label", v_stem.split("-")[-1].lower() if "-" in v_stem else v_stem.lower())
            task_str = meta_info.get("task", "static_alphabet" if item_type == "image" else "isolated_gloss")
            source_str = meta_info.get("source", "ASL_Dataset")
            signer_str = meta_info.get("signer_id", "unknown")

            res["video_id"] = p_obj.name
            res["label"] = str(label_str).strip()
            res["task"] = str(task_str)
            res["source"] = str(source_str)
            res["signer_id"] = str(signer_str)
            if should_overlay and "overlay_data" in res:
                res["overlay_data"]["overlay_idx"] = meta_info.get("overlay_idx", 1)
                res["overlay_data"]["label"] = res["label"]
            return res
    except Exception:
        pass
    return None


# ==============================================================================
#  8. LANDMARK OVERLAY EXPORTER & PRODUCTION SHARD GENERATOR
# ==============================================================================

# Hand skeleton connections
HAND_BONES = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Pose skeleton connections (indices 42..47: L_Sh=42, R_Sh=43, L_Hip=44, R_Hip=45, L_Elb=46, R_Elb=47)
POSE_BONES = [
    (42, 43),  # Shoulders
    (42, 46),  # Left arm
    (43, 47),  # Right arm
    (42, 44),  # Left torso
    (43, 45),  # Right torso
    (44, 45),  # Hips
]


def render_and_save_landmark_overlay(
    overlay_data: Dict[str, Any],
    output_dir: Union[str, Path],
) -> Optional[str]:
    """
    Renders and saves a diagnostic overlay image showing the preprocessed image
    with the 60 canonical extracted landmarks and kinematic skeleton links.
    Triggered every 10,000 preprocessed images (and image #1 for early verification).
    """
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARNING] matplotlib not installed; skipping landmark overlay export.", flush=True)
        return None

    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        rgb_frame = overlay_data.get("rgb_frame")
        if rgb_frame is None:
            return None

        lm_60 = overlay_data.get("lm_60")
        val_60 = overlay_data.get("val_60")
        roi_crop = overlay_data.get("roi_crop")
        hand_crop = overlay_data.get("hand_crop")
        overlay_idx = overlay_data.get("overlay_idx", 1)
        label = overlay_data.get("label", "unknown")
        quality = overlay_data.get("quality", 1.0)

        if lm_60 is None or val_60 is None:
            return None

        h, w = rgb_frame.shape[:2]
        px = lm_60[:, 0] * w
        py = lm_60[:, 1] * h

        has_crops = (roi_crop is not None)
        if has_crops and hand_crop is not None:
            fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [2.2, 1.2, 0.8]})
            ax_main, ax_roi, ax_hand = axes[0], axes[1], axes[2]
        elif has_crops:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2.2, 1.2]})
            ax_main, ax_roi = axes[0], axes[1]
            ax_hand = None
        else:
            fig, ax_main = plt.subplots(1, 1, figsize=(10, 8))
            ax_roi, ax_hand = None, None

        # Dark theme background
        fig.patch.set_facecolor("#16161e")
        ax_main.set_facecolor("#16161e")

        # 1. Main image with landmark overlay
        ax_main.imshow(rgb_frame)

        # Plot kinematic bones
        # Left hand (0..20)
        for (i, j) in HAND_BONES:
            if val_60[i] and val_60[j]:
                ax_main.plot([px[i], px[j]], [py[i], py[j]], color="#00e5ff", linewidth=1.8, alpha=0.8)
        # Right hand (21..41)
        for (i, j) in HAND_BONES:
            r_i, r_j = i + 21, j + 21
            if val_60[r_i] and val_60[r_j]:
                ax_main.plot([px[r_i], px[r_j]], [py[r_i], py[r_j]], color="#ff00aa", linewidth=1.8, alpha=0.8)
        # Pose (42..47)
        for (i, j) in POSE_BONES:
            if val_60[i] and val_60[j]:
                ax_main.plot([px[i], px[j]], [py[i], py[j]], color="#00ff00", linewidth=2.0, alpha=0.85)

        # Plot landmark points by anatomical group
        # Left hand: Cyan
        lh_idx = [i for i in range(0, 21) if val_60[i]]
        if lh_idx:
            ax_main.scatter(px[lh_idx], py[lh_idx], c="#00e5ff", s=28, edgecolors="black", linewidths=0.5, label="Left Hand (21)", zorder=4)
        # Right hand: Magenta
        rh_idx = [i for i in range(21, 42) if val_60[i]]
        if rh_idx:
            ax_main.scatter(px[rh_idx], py[rh_idx], c="#ff00aa", s=28, edgecolors="black", linewidths=0.5, label="Right Hand (21)", zorder=4)
        # Pose: Lime green
        pose_idx = [i for i in range(42, 48) if val_60[i]]
        if pose_idx:
            ax_main.scatter(px[pose_idx], py[pose_idx], c="#00ff00", s=38, edgecolors="black", linewidths=0.5, label="Upper Body Pose (6)", zorder=4)
        # Face: Gold
        face_idx = [i for i in range(48, 60) if val_60[i]]
        if face_idx:
            ax_main.scatter(px[face_idx], py[face_idx], c="#ffd700", s=20, edgecolors="black", linewidths=0.5, label="Face Mesh (12)", zorder=4)

        valid_count = int(val_60.sum())
        ax_main.set_title(f"Extracted Landmarks ({valid_count}/60 Valid)", color="#e0e0e0", fontsize=11, fontweight="bold")
        ax_main.axis("off")
        if valid_count > 0:
            ax_main.legend(loc="upper right", facecolor="#1e1e28", edgecolor="#444455", labelcolor="#dddddd", fontsize=8)

        # 2. Preprocessed Upper-Body ROI crop (256x256)
        if ax_roi is not None and roi_crop is not None:
            ax_roi.set_facecolor("#16161e")
            ax_roi.imshow(roi_crop)
            ax_roi.set_title(f"Preprocessed ROI ({roi_crop.shape[1]}x{roi_crop.shape[0]})", color="#e0e0e0", fontsize=11, fontweight="bold")
            ax_roi.axis("off")

        # 3. Preprocessed Hand crop (128x128)
        if ax_hand is not None and hand_crop is not None:
            ax_hand.set_facecolor("#16161e")
            ax_hand.imshow(hand_crop)
            ax_hand.set_title(f"Hand Crop ({hand_crop.shape[1]}x{hand_crop.shape[0]})", color="#e0e0e0", fontsize=11, fontweight="bold")
            ax_hand.axis("off")

        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(label))
        fig.suptitle(
            f"Sign Language Preprocessing Overlay | Sample #{overlay_idx:,} | Label: '{label}' | Quality: {quality:.2f}",
            color="#ffffff",
            fontsize=12,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout()
        out_filename = out_dir / f"overlay_img_{overlay_idx:07d}_{safe_label}.png"
        plt.savefig(out_filename, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"[PREPROCESSOR] Saved landmark overlay #{overlay_idx} -> {out_filename}", flush=True)
        return str(out_filename)
    except Exception as e:
        print(f"[WARNING] Failed to render landmark overlay #{overlay_data.get('overlay_idx')}: {e}", flush=True)
        return None

def build_phase1_dataset_structure(
    records: List[Dict[str, Any]],
    output_dir: Union[str, Path],
    label_to_idx: Dict[str, int],
    shard_size: int = 5000,
    split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    english_vocab_path: Optional[Union[str, Path]] = None,
):
    """
    Writes records into train/, val/, test/ shards + metadata and JSON mapping files.
    Adapts shard_size if visual ROI streams are present to prevent host RAM OOM.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    has_visual = any(("roi_visual" in r or "hand_visual" in r) for r in records[:min(len(records), 10)])
    if has_visual and shard_size > 200:
        print(f"[INFO] Visual ROI frames detected in records. Adapting shard_size from {shard_size} to 100 to prevent RAM overflow during saving.", flush=True)
        shard_size = 100

    split_bins: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}

    for r in records:
        sp = r.get("split", None)
        if sp in ("train", "val", "test"):
            split_bins[sp].append(r)
        else:
            vid = str(r.get("video_id", random.random()))
            h_val = int(hashlib.md5(vid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
            if h_val < split_ratios[0]:
                r["split"] = "train"
                split_bins["train"].append(r)
            elif h_val < split_ratios[0] + split_ratios[1]:
                r["split"] = "val"
                split_bins["val"].append(r)
            else:
                r["split"] = "test"
                split_bins["test"].append(r)

    # Safety fallback: If train split is empty but records exist, guarantee at least 1 record in train
    if len(split_bins["train"]) == 0 and len(records) > 0:
        donor = "val" if len(split_bins["val"]) > 0 else "test"
        if len(split_bins[donor]) > 0:
            rec = split_bins[donor].pop(0)
            rec["split"] = "train"
            split_bins["train"].append(rec)

    for split_name, split_records in split_bins.items():
        split_dir = out_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        num_shards = math.ceil(len(split_records) / shard_size) if split_records else 0
        print(f"[INFO] Writing {len(split_records)} records into {num_shards} shards for '{split_name}' split...")

        for s_idx in range(num_shards):
            shard_data = split_records[s_idx * shard_size : (s_idx + 1) * shard_size]
            shard_file = split_dir / f"shard_{s_idx:04d}.pt"
            torch.save(shard_data, shard_file)

        meta_dict = {
            "split": split_name,
            "total_records": len(split_records),
            "num_shards": num_shards,
            "shard_size": shard_size,
            "label_to_idx": label_to_idx,
        }
        with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta_dict, f, indent=2)

    print("[INFO] Serializing root vocabulary and output mapping JSON files...")

    with open(out_path / "vocab_map.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)
    with open(out_path / "vocabulary_mapping_train.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)
    with open(out_path / "vocabulary_mapping_val.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)
    with open(out_path / "vocabulary_mapping_test.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, indent=2)

    output_mapping = {idx: lbl for lbl, idx in label_to_idx.items()}
    with open(out_path / "output_mapping.json", "w", encoding="utf-8") as f:
        json.dump(output_mapping, f, indent=2)

    if english_vocab_path and Path(english_vocab_path).exists():
        with open(english_vocab_path, "r", encoding="utf-8") as f:
            eng_vocab_data = json.load(f)
        with open(out_path / "english_vocab.json", "w", encoding="utf-8") as f:
            json.dump(eng_vocab_data, f, indent=2)
    else:
        all_words = set()
        for r in records:
            lbl = r.get("label", "")
            for w in str(lbl).replace("-", " ").split():
                if w.strip():
                    all_words.add(w.strip().lower())
        sorted_words = sorted(list(all_words))
        eng_vocab_data = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
        }
        for idx, w in enumerate(sorted_words):
            eng_vocab_data[w] = idx + 4
        with open(out_path / "english_vocab.json", "w", encoding="utf-8") as f:
            json.dump(eng_vocab_data, f, indent=2)

    print(f"[SUCCESS] Dataset successfully formatted and saved to '{out_path}'.")


# ==============================================================================
#  9. CLI BATCH PROCESSING ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessor V4: Omnimodal 256x256 ROI, 128x128 Hand & 9-D Kinematics Dataset Generator"
    )
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing input videos")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing static alphanumeric images")
    parser.add_argument("--output-dir", type=str, required=True, help="Output destination dataset directory")
    parser.add_argument("--labels-file", type=str, default=None, help="Optional CSV / JSON mapping video IDs to labels & tasks")
    parser.add_argument("--english-vocab", type=str, default=None, help="Path to existing english_vocab.json")
    parser.add_argument("--shard-size", type=int, default=5000, help="Number of records per .pt shard (default 5000)")
    parser.add_argument("--max-len", type=int, default=384, help="Maximum frame sequence length")
    parser.add_argument("--split-ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], help="Train/Val/Test split ratios")
    parser.add_argument("--include-roi", action="store_true", default=True, help="Include 256x256 upper-body ROI visual frames")
    parser.add_argument("--include-hand-crop", action="store_true", default=True, help="Include 128x128 dominant hand crops")
    parser.add_argument("--disable-handedness-norm", action="store_true", help="Disable left-to-right hand canonicalization")
    parser.add_argument("--backend", type=str, default="rtmw", choices=["rtmw", "mediapipe"], help="Keypoint detector backend")
    parser.add_argument(
        "--pose-mode",
        type=str,
        default="accurate-384",
        choices=["accurate-384", "hybrid-384", "balanced-256", "hybrid-256", "lightweight-256", "performance", "balanced", "lightweight"],
        help="Pose model resolution and detector configuration (accurate-384 = RTMW-DW-X-L at 384x288 with YOLOX-M, hybrid-384 = RTMW-DW-X-L at 384x288 with YOLOX-tiny)",
    )
    parser.add_argument(
        "--flip-tta",
        action="store_true",
        default=False,
        help="Enable symmetrical horizontal flip Test-Time Augmentation (+2.1 Hand AP on occluded fingers).",
    )
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Parallel workers")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items_to_process: List[Tuple[str, str, Dict[str, Any], bool, bool, int]] = []

    # 1. Parse optional labels file (CSV or JSON)
    label_meta = {}
    if args.labels_file and Path(args.labels_file).exists():
        if args.labels_file.endswith(".json"):
            with open(args.labels_file, "r", encoding="utf-8") as f:
                raw_json = json.load(f)
                if isinstance(raw_json, list):
                    for entry in raw_json:
                        # WLASL JSON format support
                        gloss = entry.get("gloss", "")
                        for inst in entry.get("instances", []):
                            vid = str(inst.get("video_id", ""))
                            label_meta[vid] = {
                                "label": gloss,
                                "task": "isolated_gloss",
                                "source": "WLASL",
                                "split": inst.get("split", "train"),
                                "signer_id": str(inst.get("signer_id", "unknown")),
                            }
                elif isinstance(raw_json, dict):
                    label_meta = raw_json
        elif args.labels_file.endswith(".csv") or args.labels_file.endswith(".tsv"):
            sep = "\t" if args.labels_file.endswith(".tsv") else ","
            import csv
            with open(args.labels_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter=sep)
                for row in reader:
                    vid = str(row.get("video_id", row.get("EntryID", row.get("name", row.get("SENTENCE_NAME", "")))))
                    label = str(row.get("label", row.get("DominantTranslation", row.get("SENTENCE", row.get("text", "")))))
                    label_meta[vid] = {
                        "label": label,
                        "task": str(row.get("task", "sentence_level" if "SENTENCE" in row else "isolated_gloss")),
                        "source": str(row.get("source", "ASL_Dataset")),
                        "split": str(row.get("split", "train")),
                        "signer_id": str(row.get("signer_id", row.get("SUB_ID", "unknown"))),
                    }

    # 2. Discover video items
    if args.video_dir and Path(args.video_dir).exists():
        video_exts = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")
        video_paths = []
        for ext in video_exts:
            video_paths.extend(glob.glob(os.path.join(args.video_dir, "**", ext), recursive=True))
        video_paths = sorted(list(set(video_paths)))
        print(f"[INFO] Discovered {len(video_paths)} videos in '{args.video_dir}'.")
        for v_path in video_paths:
            v_stem = Path(v_path).stem
            meta_info = label_meta.get(v_stem, label_meta.get(Path(v_path).name, {}))
            items_to_process.append((v_path, "video", meta_info, args.include_roi, args.include_hand_crop, args.max_len))

    # 3. Discover static image items (e.g. ASL Alphabet folder-of-classes or flat files)
    if args.image_dir and Path(args.image_dir).exists():
        image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        image_paths = []
        for ext in image_exts:
            image_paths.extend(glob.glob(os.path.join(args.image_dir, "**", ext), recursive=True))
        image_paths = sorted(list(set(image_paths)))
        print(f"[INFO] Discovered {len(image_paths)} static images in '{args.image_dir}'.")
        for img_idx, img_path in enumerate(image_paths, 1):
            p_obj = Path(img_path)
            parent_dir_name = p_obj.parent.name
            meta_info = dict(label_meta.get(p_obj.stem, label_meta.get(p_obj.name, {})))
            if not meta_info:
                # Subfolder-of-classes (e.g. data/A/img1.jpg -> label 'A')
                meta_info = {
                    "label": parent_dir_name,
                    "task": "static_alphabet",
                    "source": "Image_Dataset",
                    "signer_id": "static",
                }
            if img_idx == 1 or img_idx % 10000 == 0:
                meta_info["should_overlay"] = True
                meta_info["overlay_idx"] = img_idx
            items_to_process.append((img_path, "image", meta_info, args.include_roi, args.include_hand_crop, args.max_len))

    if not items_to_process:
        print("[WARNING] No videos or static images found to process. Exiting.", flush=True)
        return

    overlays_dir = out_dir / "overlays"

    print(f"[INFO] Launching high-throughput processing for {len(items_to_process)} items using {args.num_workers} parallel workers...")
    records = []
    unique_labels = set()

    start_time = time.time()
    if args.num_workers > 1:
        with mp.Pool(
            processes=args.num_workers,
            initializer=_init_multiprocessing_worker,
            initargs=(args.backend, 256, 128, args.pose_mode, args.flip_tta),
        ) as pool:
            for idx, res in enumerate(pool.imap_unordered(_process_item_task, items_to_process, chunksize=16), 1):
                if res is not None:
                    if "overlay_data" in res:
                        overlay_info = res.pop("overlay_data")
                        render_and_save_landmark_overlay(overlay_info, overlays_dir)
                    unique_labels.add(res["label"])
                    records.append(res)
                if idx % 100 == 0 or idx == len(items_to_process):
                    elapsed = max(1e-3, time.time() - start_time)
                    rate = idx / elapsed
                    print(f"  [PROGRESS] Processed {idx}/{len(items_to_process)} ({idx/len(items_to_process)*100:.1f}%) @ {rate:.1f} items/sec | Valid Records: {len(records)}", flush=True)
    else:
        _init_multiprocessing_worker(args.backend, 256, 128, args.pose_mode, args.flip_tta)
        for idx, task_args in enumerate(items_to_process, 1):
            res = _process_item_task(task_args)
            if res is not None:
                if "overlay_data" in res:
                    overlay_info = res.pop("overlay_data")
                    render_and_save_landmark_overlay(overlay_info, overlays_dir)
                unique_labels.add(res["label"])
                records.append(res)
            if idx % 100 == 0 or idx == len(items_to_process):
                elapsed = max(1e-3, time.time() - start_time)
                rate = idx / elapsed
                print(f"  [PROGRESS] Processed {idx}/{len(items_to_process)} ({idx/len(items_to_process)*100:.1f}%) @ {rate:.1f} items/sec | Valid Records: {len(records)}", flush=True)

    print(f"[INFO] Extraction complete. Successfully parsed {len(records)} valid records from {len(items_to_process)} items.")

    sorted_labels = sorted(list(unique_labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted_labels)}
    for r in records:
        r["label_idx"] = label_to_idx.get(r["label"], 0)

    build_phase1_dataset_structure(
        records=records,
        output_dir=args.output_dir,
        label_to_idx=label_to_idx,
        shard_size=args.shard_size,
        split_ratios=tuple(args.split_ratios),
        english_vocab_path=args.english_vocab,
    )


if __name__ == "__main__":
    main()
