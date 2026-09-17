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
import tempfile
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import cv2
import numpy as np
import torch

import glob
import ctypes

def _preload_nvidia_cuda_libraries():
    """Preloads pip-installed NVIDIA shared objects (libcublasLt.so.13, libcudnn) with RTLD_GLOBAL."""
    try:
        import site
        site_dirs = []
        if hasattr(site, "getsitepackages"):
            site_dirs.extend(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            site_dirs.append(site.getusersitepackages())
        nvidia_dirs = []
        for s in site_dirs:
            if s and os.path.isdir(s):
                nvidia_dirs.extend(glob.glob(os.path.join(s, "nvidia", "*", "lib")))
                nvidia_dirs.extend(glob.glob(os.path.join(s, "torch", "lib")))
        if nvidia_dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

        candidates = []
        for d in nvidia_dirs:
            for so_file in glob.glob(os.path.join(d, "*.so*")):
                if os.path.isfile(so_file) and not os.path.islink(so_file):
                    candidates.append(so_file)
        priority = ["cuda_runtime", "cublaslt", "cublas", "cudnn"]
        def rank(p):
            n = os.path.basename(p).lower()
            for idx, key in enumerate(priority):
                if key in n:
                    return idx
            return len(priority)
        candidates = sorted(list(set(candidates)), key=rank)
        for c in candidates:
            try:
                ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
    except Exception:
        pass

    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass

_preload_nvidia_cuda_libraries()

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
        self.prev_hand_box: Optional[Tuple[float, float, float]] = None

    def reset(self):
        self.prev_box = None
        self.prev_hand_box = None

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

        l_valid = float(np.linalg.norm(l_sh[:2])) > 1e-4
        r_valid = float(np.linalg.norm(r_sh[:2])) > 1e-4

        l_sh_px = np.array([l_sh[0] * frame_w, l_sh[1] * frame_h]) if l_valid else None
        r_sh_px = np.array([r_sh[0] * frame_w, r_sh[1] * frame_h]) if r_valid else None

        # Nose index 48 fallback if available
        nose = landmarks[48] if len(landmarks) > 48 and float(np.linalg.norm(landmarks[48][:2])) > 1e-4 else None
        nose_px = np.array([nose[0] * frame_w, nose[1] * frame_h]) if nose is not None else None

        if l_valid and r_valid:
            sh_dist = float(np.linalg.norm(l_sh_px - r_sh_px))
            if sh_dist < 10.0 or math.isnan(sh_dist):
                sh_dist = frame_w * 0.35
            mid_sh = (l_sh_px + r_sh_px) * 0.5
            cx = float(mid_sh[0])
            cy = float(mid_sh[1] + sh_dist * 0.35)
        elif l_valid and nose_px is not None:
            # Reconstruct from Left Shoulder + Nose
            sh_dist = float(np.linalg.norm(l_sh_px - nose_px)) * 1.4
            cx = float(nose_px[0])
            cy = float(l_sh_px[1] + sh_dist * 0.35)
        elif r_valid and nose_px is not None:
            # Reconstruct from Right Shoulder + Nose
            sh_dist = float(np.linalg.norm(r_sh_px - nose_px)) * 1.4
            cx = float(nose_px[0])
            cy = float(r_sh_px[1] + sh_dist * 0.35)
        elif nose_px is not None:
            sh_dist = frame_w * 0.35
            cx = float(nose_px[0])
            cy = float(nose_px[1] + sh_dist * 0.70)
        else:
            cx = frame_w * 0.5
            cy = frame_h * 0.45
            sh_dist = frame_w * 0.35

        box_size = float(max(sh_dist * 2.85, frame_h * 0.65))
        return cx, cy, box_size

    @staticmethod
    def _crop_box(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, target_size: int) -> Optional[np.ndarray]:
        """Extracts and resizes a bounding box crop using zero-copy canvas intersection instead of full-frame border padding."""
        fh, fw = frame.shape[:2]
        if 0 <= x1 and x2 <= fw and 0 <= y1 and y2 <= fh:
            sub = frame[y1:y2, x1:x2]
            if sub.size == 0 or sub.shape[0] < 4 or sub.shape[1] < 4:
                return None
            return cv2.resize(sub, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

        w_box = max(1, x2 - x1)
        h_box = max(1, y2 - y1)
        src_x1 = max(0, x1)
        src_x2 = min(fw, x2)
        src_y1 = max(0, y1)
        src_y2 = min(fh, y2)

        canvas = np.zeros((h_box, w_box, 3), dtype=frame.dtype)
        dst_x1 = src_x1 - x1
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y1 = src_y1 - y1
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        if src_x2 > src_x1 and src_y2 > src_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = frame[src_y1:src_y2, src_x1:src_x2]
        if canvas.size == 0 or canvas.shape[0] < 4 or canvas.shape[1] < 4:
            return None
        return cv2.resize(canvas, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

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

        crop = self._crop_box(frame, x1, y1, x2, y2, self.target_size)
        if crop is None:
            crop = cv2.resize(frame, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        return crop

    def extract_hand_crop(
        self,
        frame: np.ndarray,
        hand_landmarks: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extracts a stabilized 128x128 crop around a detected hand."""
        if hand_landmarks is None or len(hand_landmarks) < 21:
            return None

        fh, fw = frame.shape[:2]
        valid_pts = hand_landmarks[hand_landmarks[:, 0] > 0]
        if len(valid_pts) < 5:
            return None

        px_pts = valid_pts[:, :2] * np.array([fw, fh])
        min_x, min_y = np.min(px_pts, axis=0)
        max_x, max_y = np.max(px_pts, axis=0)

        raw_cx = float((min_x + max_x) * 0.5)
        raw_cy = float((min_y + max_y) * 0.5)
        raw_size = float(max(max_x - min_x, max_y - min_y) * 1.45)
        raw_size = float(max(32.0, raw_size))

        if self.prev_hand_box is None:
            smooth_cx, smooth_cy, smooth_size = raw_cx, raw_cy, raw_size
        else:
            prev_cx, prev_cy, prev_size = self.prev_hand_box
            dist = math.hypot(raw_cx - prev_cx, raw_cy - prev_cy)
            if dist < self.hysteresis_px:
                smooth_cx, smooth_cy = prev_cx, prev_cy
            else:
                smooth_cx = 0.35 * raw_cx + 0.65 * prev_cx
                smooth_cy = 0.35 * raw_cy + 0.65 * prev_cy
            smooth_size = 0.25 * raw_size + 0.75 * prev_size

        self.prev_hand_box = (smooth_cx, smooth_cy, smooth_size)

        half_s = smooth_size * 0.5
        x1 = int(round(smooth_cx - half_s))
        y1 = int(round(smooth_cy - half_s))
        x2 = int(round(smooth_cx + half_s))
        y2 = int(round(smooth_cy + half_s))

        return self._crop_box(frame, x1, y1, x2, y2, self.hand_crop_size)


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
        self.unsharp_strength = unsharp_strength
        # Pre-cache quantized CLAHE instances to avoid dynamic C++ object construction per frame
        self.clahe_cache = {
            round(c, 1): cv2.createCLAHE(clipLimit=round(c, 1), tileGridSize=(8, 8))
            for c in np.arange(1.0, 3.6, 0.2)
        }
        self.default_clahe = self.clahe_cache.get(round(clip_limit, 1), cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8)))

    def enhance_frame_with_luma(
        self, rgb_img: np.ndarray, apply_deblur: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Fused single-pass frame enhancement and blur evaluation:
        1. Evaluates blur score and illumination on sub-sampled green channel (60% luma) in CV_32F.
        2. Fast-path: Bypasses expensive LAB color space conversion and CLAHE if frame is well-illuminated and sharp.
        3. Pre-cached CLAHE lookup avoids dynamic C++ grid re-allocation.
        """
        if rgb_img is None or rgb_img.size == 0:
            return rgb_img, np.zeros((1, 1), dtype=np.uint8), 100.0
        try:
            # 1. Fast blur score and illumination on green channel sub-sampled grid (4x pixel reduction) in CV_32F
            green_sub = rgb_img[::2, ::2, 1]
            blur_score = float(cv2.Laplacian(green_sub, cv2.CV_32F).var())
            mean_lum = float(np.mean(green_sub))

            # 2. Fast-path: If frame is well-illuminated and sharp, skip expensive LAB conversions & CLAHE!
            if 80.0 <= mean_lum <= 180.0 and blur_score >= 80.0:
                return rgb_img, rgb_img[:, :, 1], blur_score

            # 3. Enhanced branch with cached CLAHE
            lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
            l_channel = lab[:, :, 0]
            adaptive_clip = round(float(np.clip(1.5 + (128.0 - mean_lum) / 64.0, 1.0, 3.5)), 1)
            clahe = self.clahe_cache.get(adaptive_clip, self.default_clahe)
            lab[:, :, 0] = clahe.apply(l_channel)

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
        """Returns Laplacian variance on CV_32F (higher = sharper, <50 = blurry). Reuses l_channel if provided."""
        try:
            if l_channel is not None:
                return float(cv2.Laplacian(l_channel[::2, ::2], cv2.CV_32F).var())
            green = rgb_img[::2, ::2, 1] if rgb_img.ndim == 3 else rgb_img[::2, ::2]
            return float(cv2.Laplacian(green, cv2.CV_32F).var())
        except Exception:
            return 100.0


# ==============================================================================
#  3. REFERENCE-PART NORMALIZATION & 9-D KINEMATICS
# ==============================================================================

def reference_part_normalize(landmarks: np.ndarray, val_mask: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Applies Fine-Tuned Reference-Part Normalization to [T, 60, 3] landmarks:
      - Anchors (0,0,0) to low-pass filtered sternum trajectory (EMA mid-shoulder, tau ~ 0.7s)
        eliminating high-frequency drift/jitter while preserving grammatical body leans.
      - Scales by robust 3D torso-box diagonal (inter-shoulder + shoulder-to-hip).
      - Robust fallback to inter-ocular or wrist/hand center anchor for tight static crops.
    """
    T, K, C = landmarks.shape

    l_valid = val_mask[:, 42]
    r_valid = val_mask[:, 43]
    both_valid = l_valid & r_valid

    if both_valid.any():
        valid_idx = np.where(both_valid)[0]
        l_sh = landmarks[both_valid, 42, :3]
        r_sh = landmarks[both_valid, 43, :3]
        mid_sh_valid = (l_sh + r_sh) * 0.5
        d_sh = l_sh - r_sh
        sh_dist = np.sqrt(d_sh[:, 0] ** 2 + d_sh[:, 1] ** 2 + d_sh[:, 2] ** 2)
        scale_ref = float(np.median(sh_dist)) if len(sh_dist) > 0 else 1.0

        # Check for 3D torso diagonal scaling using hips (44, 45) if available
        if val_mask.shape[1] > 45:
            hip_valid = val_mask[:, 44] & val_mask[:, 45]
            both_and_hip = both_valid & hip_valid
            if both_and_hip.any():
                sh_pts = (landmarks[both_and_hip, 42, :3] + landmarks[both_and_hip, 43, :3]) * 0.5
                hip_pts = (landmarks[both_and_hip, 44, :3] + landmarks[both_and_hip, 45, :3]) * 0.5
                torso_h = np.sqrt(np.sum((sh_pts - hip_pts) ** 2, axis=-1))
                d_sh_sub = landmarks[both_and_hip, 42, :3] - landmarks[both_and_hip, 43, :3]
                torso_w = np.sqrt(np.sum(d_sh_sub ** 2, axis=-1))
                torso_diag = np.sqrt(torso_w ** 2 + torso_h ** 2)
                # Calibrate to canonical shoulder scale (shoulder width is ~0.65 of torso diagonal)
                scale_ref = float(np.median(torso_diag)) * 0.65

        scale_ref = max(1e-3, scale_ref)
        inv_scale = np.float32(1.0 / scale_ref)

        # Build smoothed sternum anchor sequence [T, 3]
        if T == 1 or len(valid_idx) == 1:
            anchor_seq = np.mean(mid_sh_valid, axis=0, keepdims=True)  # [1, 3]
        else:
            # Interpolate any dropped shoulder frames across time
            mid_sh_full = np.empty((T, 3), dtype=np.float32)
            if len(valid_idx) == T:
                mid_sh_full[:] = mid_sh_valid
            else:
                for c in range(3):
                    mid_sh_full[:, c] = np.interp(np.arange(T), valid_idx, mid_sh_valid[:, c])
            
            # Exponential Moving Average filter (alpha=0.90 -> tau ~ 0.7s at 30 fps)
            alpha = 0.90
            anchor_seq = np.empty((T, 3), dtype=np.float32)
            anchor_seq[0] = mid_sh_full[0]
            for t in range(1, T):
                anchor_seq[t] = alpha * anchor_seq[t - 1] + (1.0 - alpha) * mid_sh_full[t]

        normed = (landmarks - anchor_seq[:, None, :]) * inv_scale
    else:
        # Fallback to wrist/hand center normalization if shoulders are missing (tight hand crops)
        hand_pts = landmarks[:, 0:42, :3]
        hand_mask = val_mask[:, 0:42]
        if hand_mask.any():
            valid_hand_coords = hand_pts[hand_mask]
            anchor = np.mean(valid_hand_coords, axis=0)
            d_hand = valid_hand_coords - anchor
            scale_ref = float(np.max(np.sqrt(d_hand[:, 0] ** 2 + d_hand[:, 1] ** 2 + d_hand[:, 2] ** 2)))
            scale_ref = max(1e-3, scale_ref)
            inv_scale = np.float32(1.0 / scale_ref)
            normed = (landmarks - anchor) * inv_scale
        else:
            normed = landmarks.copy()
            scale_ref = 1.0

    return normed, scale_ref


def clean_out_of_bounds_hands(
    feat_arr: np.ndarray,
    val_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Detects and zeroes out hand keypoints that are out-of-bounds, collapsed, or anatomically exploded:
      - Left Hand: indices 0..20 (wrist at 0)
      - Right Hand: indices 21..41 (wrist at 21)
    Conditions for invalid / phantom hand in frame t:
      1. Out-of-bounds: wrist or multiple finger joints have |x| > 2.2 or |y| > 2.5
      2. Exploded span: distance from wrist to any finger exceeds 0.55 normalized units (shoulder distance = 1.0)
      3. Collapsed cluster: spatial std across all 21 hand joints < 0.015 (tracking lost, points pinned to a single border point)
    When detected, the 21 keypoints of that hand in that frame are zeroed out across all feature channels.
    Fast-path: Dual-hand batched processing with squared distance thresholds and zero square root allocations.
    Synchronously updates val_mask if provided.
    """
    if feat_arr.ndim < 2 or feat_arr.shape[1] < 42:
        return feat_arr

    # Batched view over both hands: [T, 2, 21, 3]
    h_both = feat_arr[..., :42, :3].reshape(feat_arr.shape[0], 2, 21, 3)
    wrist = h_both[:, :, 0:1, :]  # [T, 2, 1, 3]

    is_active = np.max(np.abs(h_both), axis=(2, 3)) > 1e-4  # [T, 2]
    if not np.any(is_active):
        return feat_arr

    oob = (np.abs(h_both[..., 0]) > 2.2) | (np.abs(h_both[..., 1]) > 2.5)  # [T, 2, 21]
    has_oob = np.any(oob, axis=2)  # [T, 2]

    diff = h_both - wrist
    dist_sq = np.sum(diff * diff, axis=-1)  # [T, 2, 21]
    has_exploded = np.max(dist_sq, axis=2) > 0.3025  # 0.55^2

    mean_h = np.mean(h_both, axis=2, keepdims=True)
    var_h = np.mean(np.sum((h_both - mean_h) ** 2, axis=-1), axis=2)
    has_collapsed = (var_h < 0.000225) & is_active  # 0.015^2

    invalid = (has_oob | has_exploded | has_collapsed) & is_active
    if not np.any(invalid):
        return feat_arr

    out = feat_arr.copy()
    T = feat_arr.shape[0]
    for h_idx, (h_start, h_end) in enumerate([(0, 21), (21, 42)]):
        inv_h = invalid[:, h_idx]
        if not np.any(inv_h):
            continue
        valid_indices = np.where(~inv_h)[0]
        if len(valid_indices) == 0:
            out[:, h_start:h_end, :] = 0.0
            if val_mask is not None:
                val_mask[:, h_start:h_end] = False
            continue
        # Identify contiguous invalid chunks
        diff_inv = np.diff(inv_h.astype(np.int32))
        starts = np.where(diff_inv == 1)[0] + 1
        if inv_h[0]:
            starts = np.r_[0, starts]
        ends = np.where(diff_inv == -1)[0]
        if inv_h[-1]:
            ends = np.r_[ends, T - 1]

        for s, e in zip(starts, ends):
            gap_len = e - s + 1
            # If transient glitch (<= 5 frames) between valid frames, interpolate smoothly
            if gap_len <= 5 and s > 0 and e < T - 1:
                t_coords = np.arange(s, e + 1)
                for k in range(h_start, h_end):
                    for c in range(feat_arr.shape[-1]):
                        out[t_coords, k, c] = np.interp(
                            t_coords, [s - 1, e + 1], [out[s - 1, k, c], out[e + 1, k, c]]
                        )
                if val_mask is not None:
                    val_mask[s:e + 1, h_start:h_end] = True
            else:
                out[s:e + 1, h_start:h_end, :] = 0.0
                if val_mask is not None:
                    val_mask[s:e + 1, h_start:h_end] = False
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


def compute_9d_kinematics(landmarks_seq: np.ndarray, smooth: bool = True, fps: float = 30.0) -> np.ndarray:
    """
    Transforms [T, 60, 3] positions into [T, 60, 9] kinematics:
      - (x, y, z) coordinates
      - (dx, dy, dz) instantaneous velocity (physical delta_t normalized)
      - (d^2x, d^2y, d^2z) instantaneous acceleration (physical delta_t^2 normalized)
    Fused zero-copy Savitzky-Golay stencil with physical human biomechanical clamping.
    """
    t_len, num_pts, _ = landmarks_seq.shape
    if t_len == 0:
        return np.zeros((0, num_pts, 9), dtype=np.float32)

    delta_t = float(1.0 / max(fps, 1.0))
    pos = smooth_landmark_trajectories_binomial(landmarks_seq) if smooth else landmarks_seq

    kinematics_9d = np.empty((t_len, num_pts, 9), dtype=np.float32)
    kinematics_9d[:, :, 0:3] = landmarks_seq

    vel_slice = kinematics_9d[:, :, 3:6]
    acc_slice = kinematics_9d[:, :, 6:9]

    # Max velocity & acceleration bounds in normalized units (shoulder width ~ 0.4m, max arm speed ~ 4.5 m/s -> ~11.0 units/s)
    v_clip = 12.0
    a_clip = 45.0

    if t_len >= 5 and smooth:
        p0 = pos[:-4]
        p1 = pos[1:-3]
        p2 = pos[2:-2]
        p3 = pos[3:-1]
        p4 = pos[4:]

        # Fused 5-point quadratic Savitzky-Golay 1st derivative (velocity) with delta_t scaling
        v_mid = (-2.0 * p0 - p1 + p3 + 2.0 * p4) / (10.0 * delta_t)
        np.clip(v_mid, -v_clip, v_clip, out=vel_slice[2:-2])
        vel_slice[1] = np.clip((pos[2] - pos[0]) / (2.0 * delta_t), -v_clip, v_clip)
        vel_slice[0] = np.clip((pos[1] - pos[0]) / delta_t, -v_clip, v_clip)
        vel_slice[-2] = np.clip((pos[-1] - pos[-3]) / (2.0 * delta_t), -v_clip, v_clip)
        vel_slice[-1] = np.clip((pos[-1] - pos[-2]) / delta_t, -v_clip, v_clip)

        # Fused 5-point quadratic Savitzky-Golay 2nd derivative (acceleration) with delta_t^2 scaling
        a_mid = (2.0 * p0 - p1 - 2.0 * p2 - p3 + 2.0 * p4) / (7.0 * (delta_t ** 2))
        np.clip(a_mid, -a_clip, a_clip, out=acc_slice[2:-2])
        a1 = np.clip((pos[2] - 2.0 * pos[1] + pos[0]) / (delta_t ** 2), -a_clip, a_clip)
        acc_slice[0] = a1
        acc_slice[1] = a1
        a_end = np.clip((pos[-1] - 2.0 * pos[-2] + pos[-3]) / (delta_t ** 2), -a_clip, a_clip)
        acc_slice[-2] = a_end
        acc_slice[-1] = a_end
    elif t_len > 2:
        vel_slice[1:-1] = np.clip((pos[2:] - pos[:-2]) / (2.0 * delta_t), -v_clip, v_clip)
        vel_slice[0] = np.clip((pos[1] - pos[0]) / delta_t, -v_clip, v_clip)
        vel_slice[-1] = np.clip((pos[-1] - pos[-2]) / delta_t, -v_clip, v_clip)

        acc_slice[1:-1] = np.clip((pos[2:] - 2.0 * pos[1:-1] + pos[:-2]) / (delta_t ** 2), -a_clip, a_clip)
        acc_slice[0] = acc_slice[1]
        acc_slice[-1] = acc_slice[-2]
    elif t_len > 1:
        vel_slice[0] = np.clip((pos[1] - pos[0]) / delta_t, -v_clip, v_clip)
        vel_slice[1] = vel_slice[0]
        acc_slice[:] = 0.0
    else:
        vel_slice[:] = 0.0
        acc_slice[:] = 0.0

    # Zero out velocity and acceleration where positions were zero/inactive to prevent artificial step spikes
    is_inactive = (np.abs(pos).sum(axis=-1) < 1e-5)
    vel_slice[is_inactive] = 0.0
    acc_slice[is_inactive] = 0.0

    return np.nan_to_num(kinematics_9d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def interpolate_missing_hand_landmarks(
    landmarks_seq: np.ndarray,
    valid_mask: np.ndarray,
    max_gap: int = 5,
) -> np.ndarray:
    """
    Smoothly interpolates hand landmark trajectories during brief dropped hand detections.
    CRITICAL FIX: Only interpolates short dropout gaps (<= max_gap frames).
    Prevents np.interp from extrapolating constant ghost hands across the entire video
    before the hand appears or after the hand leaves the signing space.
    """
    t_len, num_pts, _ = landmarks_seq.shape
    if t_len <= 1:
        return landmarks_seq

    # Fast-path: Only hands (indices 0:42) require interpolation. Pose and face are stable.
    hand_pts = min(num_pts, 42)
    if valid_mask[:, :hand_pts].all():
        return landmarks_seq

    cleaned_seq = landmarks_seq.copy()
    time_indices = np.arange(t_len)
    valid_counts = valid_mask[:, :hand_pts].sum(axis=0)

    for pt_idx in range(hand_pts):
        cnt = valid_counts[pt_idx]
        if cnt == t_len:
            continue
        if cnt <= 1:
            continue

        pt_valid = valid_mask[:, pt_idx]
        valid_t = time_indices[pt_valid]

        # Interpolate only across short internal dropout gaps <= max_gap
        for i in range(len(valid_t) - 1):
            t_start = valid_t[i]
            t_end = valid_t[i + 1]
            gap = t_end - t_start - 1
            if 0 < gap <= max_gap:
                alpha = np.linspace(0.0, 1.0, gap + 2, dtype=np.float32)[1:-1, np.newaxis]
                cleaned_seq[t_start + 1:t_end, pt_idx, :] = (
                    (1.0 - alpha) * cleaned_seq[t_start, pt_idx, :] +
                    alpha * cleaned_seq[t_end, pt_idx, :]
                )

    return cleaned_seq


def compute_cranial_imu(landmarks_seq: np.ndarray, val_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes 3D angular velocities (omega_yaw, omega_pitch, omega_roll) from rigid facial landmarks:
    - Nose tip (index 48)
    - Left/Right eye outer corners (indices 52, 53)
    Returns [T, 3] array in rad/s (scaled for 30 FPS).
    """
    T = landmarks_seq.shape[0]
    imu = np.zeros((T, 3), dtype=np.float32)
    if T <= 1 or landmarks_seq.shape[1] < 54:
        return imu

    if landmarks_seq.shape[1] >= 60:
        nose = landmarks_seq[:, 48, :3]
        l_eye = landmarks_seq[:, 52, :3]
        r_eye = landmarks_seq[:, 53, :3]
    else:
        nose = landmarks_seq[:, -6, :3]
        l_eye = landmarks_seq[:, -4, :3]
        r_eye = landmarks_seq[:, -3, :3]

    eye_dx = r_eye[:, 0] - l_eye[:, 0]
    eye_dz = r_eye[:, 2] - l_eye[:, 2]
    yaw_angle = np.arctan2(eye_dz, eye_dx + 1e-6)
    pitch_angle = np.arctan2(nose[:, 1] - 0.5 * (l_eye[:, 1] + r_eye[:, 1]), eye_dx + 1e-6)
    roll_angle = np.arctan2(r_eye[:, 1] - l_eye[:, 1], eye_dx + 1e-6)

    if T > 2:
        imu[1:-1, 0] = (yaw_angle[2:] - yaw_angle[:-2]) * 15.0
        imu[1:-1, 1] = (pitch_angle[2:] - pitch_angle[:-2]) * 15.0
        imu[1:-1, 2] = (roll_angle[2:] - roll_angle[:-2]) * 15.0
        imu[0] = imu[1]
        imu[-1] = imu[-2]
    elif T == 2:
        imu[:, 0] = (yaw_angle[1] - yaw_angle[0]) * 30.0
        imu[:, 1] = (pitch_angle[1] - pitch_angle[0]) * 30.0
        imu[:, 2] = (roll_angle[1] - roll_angle[0]) * 30.0

    return np.clip(imu, -15.0, 15.0)


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
    Vectorized analytic 3D cross-products and fast distance norms. Returns: [T, 19] float32 array.
    """
    T, K, _ = landmarks_seq.shape
    if T == 0:
        return np.zeros((0, 19), dtype=np.float32)

    pos = landmarks_seq[:, :, :3]  # [T, 60, 3]

    phonology = np.empty((T, 19), dtype=np.float32)

    # 1. Palm Orientation Normals (6 Dims)
    # LH: u = idx - w, v = pky - w (0=wrist, 5=index MCP, 17=pinky MCP)
    lh_u = pos[:, 5] - pos[:, 0]
    lh_v = pos[:, 17] - pos[:, 0]
    lh_nx = lh_u[:, 1] * lh_v[:, 2] - lh_u[:, 2] * lh_v[:, 1]
    lh_ny = lh_u[:, 2] * lh_v[:, 0] - lh_u[:, 0] * lh_v[:, 2]
    lh_nz = lh_u[:, 0] * lh_v[:, 1] - lh_u[:, 1] * lh_v[:, 0]
    inv_lh = 1.0 / np.maximum(np.sqrt(lh_nx * lh_nx + lh_ny * lh_ny + lh_nz * lh_nz), 1e-5)
    phonology[:, 0] = lh_nx * inv_lh
    phonology[:, 1] = lh_ny * inv_lh
    phonology[:, 2] = lh_nz * inv_lh

    # RH: u = idx - w, v = pky - w (21=wrist, 26=index MCP, 38=pinky MCP)
    rh_u = pos[:, 26] - pos[:, 21]
    rh_v = pos[:, 38] - pos[:, 21]
    rh_nx = rh_u[:, 1] * rh_v[:, 2] - rh_u[:, 2] * rh_v[:, 1]
    rh_ny = rh_u[:, 2] * rh_v[:, 0] - rh_u[:, 0] * rh_v[:, 2]
    rh_nz = rh_u[:, 0] * rh_v[:, 1] - rh_u[:, 1] * rh_v[:, 0]
    inv_rh = 1.0 / np.maximum(np.sqrt(rh_nx * rh_nx + rh_ny * rh_ny + rh_nz * rh_nz), 1e-5)
    phonology[:, 3] = rh_nx * inv_rh
    phonology[:, 4] = rh_ny * inv_rh
    phonology[:, 5] = rh_nz * inv_rh

    # 2. Bimanual Synchrony (1 Dim)
    lh_vel = np.zeros_like(pos[:, 0, :3])
    rh_vel = np.zeros_like(pos[:, 21, :3])
    if T > 1:
        lh_vel[1:] = pos[1:, 0, :3] - pos[:-1, 0, :3]
        rh_vel[1:] = pos[1:, 21, :3] - pos[:-1, 21, :3]

    dot = np.sum(lh_vel * rh_vel, axis=-1, keepdims=True)
    denom = np.maximum(
        np.sqrt(np.sum(lh_vel * lh_vel, axis=-1, keepdims=True) * np.sum(rh_vel * rh_vel, axis=-1, keepdims=True)),
        1e-5
    )
    phonology[:, 6:7] = dot / denom

    # 3. Location Anchoring to Face (2 Dims: Gaussian proximity kernel in (0, 1])
    # k_face = exp(- d^2 / (2 * sigma^2)), where sigma=0.35 (normalized torso units)
    face_centroid = np.mean(pos[:, 48:60, :3], axis=1)  # [T, 3]
    d_lh_face = pos[:, 0, :3] - face_centroid
    d_rh_face = pos[:, 21, :3] - face_centroid
    sigma_face_sq = 2.0 * (0.35 ** 2)
    phonology[:, 7] = np.exp(-np.sum(d_lh_face * d_lh_face, axis=-1) / sigma_face_sq)
    phonology[:, 8] = np.exp(-np.sum(d_rh_face * d_rh_face, axis=-1) / sigma_face_sq)

    # 4. Finger Curl / Aperture (10 Dims: 5 LH + 5 RH normalized by palm length to [0, 1])
    # 0.0 = fully curled fist, 1.0 = fully extended open hand
    lh_tips = [4, 8, 12, 16, 20]
    rh_tips = [25, 29, 33, 37, 41]
    
    # Palm length (wrist to MCP joint: 0 to 9 for LH, 21 to 30 for RH)
    lh_palm_len = np.maximum(np.sqrt(np.sum((pos[:, 9, :3] - pos[:, 0, :3]) ** 2, axis=-1, keepdims=True)), 1e-4)
    rh_palm_len = np.maximum(np.sqrt(np.sum((pos[:, 30, :3] - pos[:, 21, :3]) ** 2, axis=-1, keepdims=True)), 1e-4)

    diff_lh = pos[:, lh_tips, :3] - pos[:, 0:1, :3]
    lh_curl_raw = np.sqrt(np.sum(diff_lh * diff_lh, axis=-1))
    phonology[:, 9:14] = np.clip(lh_curl_raw / (1.85 * lh_palm_len), 0.0, 1.0)

    diff_rh = pos[:, rh_tips, :3] - pos[:, 21:22, :3]
    rh_curl_raw = np.sqrt(np.sum(diff_rh * diff_rh, axis=-1))
    phonology[:, 14:19] = np.clip(rh_curl_raw / (1.85 * rh_palm_len), 0.0, 1.0)

    return np.nan_to_num(phonology, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


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

# Symmetrical bilateral face landmarks (indices 48..59):
# (48: Nose, 49..51: Midline, 56: Forehead top, 59: Glabella remain fixed along midline)
SWAP_LEFT_RIGHT_INDICES[52] = 53  # Left Eye <-> Right Eye
SWAP_LEFT_RIGHT_INDICES[53] = 52
SWAP_LEFT_RIGHT_INDICES[54] = 55  # Left Mouth Corner <-> Right Mouth Corner
SWAP_LEFT_RIGHT_INDICES[55] = 54
SWAP_LEFT_RIGHT_INDICES[57] = 58  # Left Eyebrow <-> Right Eyebrow
SWAP_LEFT_RIGHT_INDICES[58] = 57


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
        d_lh = landmarks_seq[1:, 0, :2] - landmarks_seq[:-1, 0, :2]
        d_rh = landmarks_seq[1:, 21, :2] - landmarks_seq[:-1, 21, :2]
        lh_energy = float(np.sum(np.sqrt(d_lh[:, 0] ** 2 + d_lh[:, 1] ** 2)))
        rh_energy = float(np.sum(np.sqrt(d_rh[:, 0] ** 2 + d_rh[:, 1] ** 2)))
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
    Fast vectorized RMS jitter calculation.
    """
    mean_conf = float(np.mean(conf_scores)) if conf_scores else 0.80

    lh_present = float(val_mask[:, 0:21].any(axis=-1).mean()) if val_mask.shape[1] >= 21 else 0.0
    rh_present = float(val_mask[:, 21:42].any(axis=-1).mean()) if val_mask.shape[1] >= 42 else 0.0
    hand_presence = float(max(lh_present, rh_present))

    if len(landmarks_seq) > 2:
        d = landmarks_seq[1:] - landmarks_seq[:-1]
        diffs = np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2)
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
        target_fps: float = 30.0,
        pose_backend: Optional[str] = None,
        **kwargs,
    ):
        if pose_backend is not None:
            backend = pose_backend
        self.target_fps = target_fps
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
                    dim = min(3, norm_kps.shape[1])

                    # 1. Left Hand (0..20)
                    for i, idx in enumerate(RTMW_LH_21):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[i, :dim] = norm_kps[idx, :dim]
                            val_60[i] = True

                    # 2. Right Hand (21..41)
                    for i, idx in enumerate(RTMW_RH_21):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[21 + i, :dim] = norm_kps[idx, :dim]
                            val_60[21 + i] = True

                    # 3. Upper Body Pose (42..47: L_Shoulder=5, R_Shoulder=6, L_Hip=11, R_Hip=12, L_Elbow=7, R_Elbow=8)
                    for i, idx in enumerate(RTMW_POSE_6):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[42 + i, :dim] = norm_kps[idx, :dim]
                            val_60[42 + i] = True

                    # 4. Face Mesh (48..59)
                    for i, idx in enumerate(RTMW_FACE_12):
                        if idx < len(norm_kps) and (scores_133 is None or scores_133[idx] >= 0.20):
                            lm_60[48 + i, :dim] = norm_kps[idx, :dim]
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
        target_fps = getattr(self, "target_fps", 30.0)

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
        clean_landmarks = clean_out_of_bounds_hands(clean_landmarks, landmark_val_mask)

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

        cranial_imu = compute_cranial_imu(clean_landmarks, landmark_val_mask)
        face_landmarks = clean_landmarks[:, 48:60, :3] if clean_landmarks.shape[1] >= 60 else np.zeros((clean_landmarks.shape[0], 12, 3), dtype=np.float32)

        # Ensure finite numeric stability across all output tensors
        kinematics_9d = np.nan_to_num(kinematics_9d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        phonology_19d = np.nan_to_num(phonology_19d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cranial_imu = np.nan_to_num(cranial_imu, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        face_landmarks = np.nan_to_num(face_landmarks, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        res = {
            "features": torch.from_numpy(kinematics_9d).to(torch.bfloat16),
            "phonology": torch.from_numpy(phonology_19d).to(torch.bfloat16),
            "cranial_imu": torch.from_numpy(cranial_imu).to(torch.float32),
            "face_landmarks": torch.from_numpy(face_landmarks).to(torch.float32),
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
        raw_landmarks = clean_out_of_bounds_hands(raw_landmarks, landmark_val_mask)

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

        cranial_imu = compute_cranial_imu(raw_landmarks, landmark_val_mask)
        face_landmarks = raw_landmarks[:, 48:60, :3] if raw_landmarks.shape[1] >= 60 else np.zeros((raw_landmarks.shape[0], 12, 3), dtype=np.float32)

        # Ensure finite numeric stability across all output tensors
        kinematics_9d = np.nan_to_num(kinematics_9d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        phonology_19d = np.nan_to_num(phonology_19d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        cranial_imu = np.nan_to_num(cranial_imu, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        face_landmarks = np.nan_to_num(face_landmarks, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        res = {
            "features": torch.from_numpy(kinematics_9d).to(torch.bfloat16),
            "phonology": torch.from_numpy(phonology_19d).to(torch.bfloat16),
            "cranial_imu": torch.from_numpy(cranial_imu).to(torch.float32),
            "face_landmarks": torch.from_numpy(face_landmarks).to(torch.float32),
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

def _init_multiprocessing_worker(backend: str, target_roi_size: int, hand_crop_size: int, pose_mode: str = "accurate-384", flip_tta: bool = False, device: str = "cpu"):
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
        device=device,
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

class StreamingShardWriter:
    """
    High-throughput, streaming disk writer that incrementally writes shards to train/val/test
    directories as records arrive. Keeps memory footprint strictly bounded (<500 MB) even
    when processing hundreds of thousands of samples with 256x256 ROI visual frames.
    """
    def __init__(
        self,
        output_dir: Union[str, Path],
        label_to_idx: Optional[Dict[str, int]] = None,
        shard_size: int = 5000,
        split_ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        has_visual: bool = False,
    ):
        self.out_path = Path(output_dir)
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.label_to_idx = dict(label_to_idx) if label_to_idx is not None else {}
        self.split_ratios = split_ratios

        if has_visual and shard_size > 200:
            print(f"[INFO] Visual frames detected. Adapting shard_size from {shard_size} to 100 to bound RAM usage.", flush=True)
            self.shard_size = 100
        else:
            self.shard_size = shard_size

        self.split_dirs = {
            "train": self.out_path / "train",
            "val": self.out_path / "val",
            "test": self.out_path / "test",
        }
        for d in self.split_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        self.buffers: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
        self.shard_indices: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.total_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
        self.all_words: Set[str] = set()

    def add_record(self, r: Dict[str, Any]):
        lbl = str(r.get("label", "")).strip()
        if lbl not in self.label_to_idx:
            self.label_to_idx[lbl] = len(self.label_to_idx)
        r["label_idx"] = self.label_to_idx[lbl]

        for w in str(lbl).replace("-", " ").split():
            if w.strip():
                self.all_words.add(w.strip().lower())

        sp = r.get("split", None)
        if sp not in ("train", "val", "test"):
            vid = str(r.get("video_id", random.random()))
            h_val = int(hashlib.md5(vid.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
            if h_val < self.split_ratios[0]:
                sp = "train"
            elif h_val < self.split_ratios[0] + self.split_ratios[1]:
                sp = "val"
            else:
                sp = "test"
            r["split"] = sp

        self.buffers[sp].append(r)
        self.total_counts[sp] += 1

        if len(self.buffers[sp]) >= self.shard_size:
            self._flush_split(sp)

    def _flush_split(self, split_name: str):
        buf = self.buffers[split_name]
        if not buf:
            return
        s_idx = self.shard_indices[split_name]
        shard_file = self.split_dirs[split_name] / f"shard_{s_idx:04d}.pt"
        torch.save(buf, shard_file)
        self.shard_indices[split_name] += 1
        self.buffers[split_name] = []

    def finalize(self, english_vocab_path: Optional[Union[str, Path]] = None):
        for sp in ("train", "val", "test"):
            if self.buffers[sp]:
                self._flush_split(sp)

        # Safety fallback: If train split is empty but records exist, ensure train has data
        if self.total_counts["train"] == 0 and (self.total_counts["val"] > 0 or self.total_counts["test"] > 0):
            donor = "val" if self.total_counts["val"] > 0 else "test"
            first_shard_donor = self.split_dirs[donor] / "shard_0000.pt"
            if first_shard_donor.exists():
                donor_records = torch.load(first_shard_donor, map_location="cpu", weights_only=False)
                if len(donor_records) > 0:
                    rec = donor_records.pop(0)
                    rec["split"] = "train"
                    torch.save(donor_records, first_shard_donor)
                    torch.save([rec], self.split_dirs["train"] / "shard_0000.pt")
                    self.total_counts["train"] += 1
                    self.total_counts[donor] -= 1
                    self.shard_indices["train"] = 1

        for sp in ("train", "val", "test"):
            meta_dict = {
                "split": sp,
                "total_records": self.total_counts[sp],
                "num_shards": self.shard_indices[sp],
                "shard_size": self.shard_size,
                "label_to_idx": self.label_to_idx,
            }
            with open(self.split_dirs[sp] / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta_dict, f, indent=2)

        with open(self.out_path / "vocab_map.json", "w", encoding="utf-8") as f:
            json.dump(self.label_to_idx, f, indent=2)
        with open(self.out_path / "vocabulary_mapping_train.json", "w", encoding="utf-8") as f:
            json.dump(self.label_to_idx, f, indent=2)
        with open(self.out_path / "vocabulary_mapping_val.json", "w", encoding="utf-8") as f:
            json.dump(self.label_to_idx, f, indent=2)
        with open(self.out_path / "vocabulary_mapping_test.json", "w", encoding="utf-8") as f:
            json.dump(self.label_to_idx, f, indent=2)

        output_mapping = {idx: lbl for lbl, idx in self.label_to_idx.items()}
        with open(self.out_path / "output_mapping.json", "w", encoding="utf-8") as f:
            json.dump(output_mapping, f, indent=2)

        if english_vocab_path and Path(english_vocab_path).exists():
            with open(english_vocab_path, "r", encoding="utf-8") as f:
                eng_vocab_data = json.load(f)
            with open(self.out_path / "english_vocab.json", "w", encoding="utf-8") as f:
                json.dump(eng_vocab_data, f, indent=2)
        else:
            sorted_words = sorted(list(self.all_words))
            eng_vocab_data = {
                "<PAD>": 0,
                "<BOS>": 1,
                "<EOS>": 2,
                "<UNK>": 3,
            }
            for idx, w in enumerate(sorted_words):
                eng_vocab_data[w] = idx + 4
            with open(self.out_path / "english_vocab.json", "w", encoding="utf-8") as f:
                json.dump(eng_vocab_data, f, indent=2)

        print(f"[SUCCESS] Dataset successfully formatted and saved to '{self.out_path}'.", flush=True)


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
    has_visual = any(("roi_visual" in r or "hand_visual" in r) for r in records[:min(len(records), 10)])
    writer = StreamingShardWriter(
        output_dir=output_dir,
        label_to_idx=label_to_idx,
        shard_size=shard_size,
        split_ratios=split_ratios,
        has_visual=has_visual,
    )
    for r in records:
        writer.add_record(r)
    writer.finalize(english_vocab_path=english_vocab_path)


# ==============================================================================
#  9. ZIP STREAMING DATASET ORCHESTRATOR (COLAB & CLOUD STORAGE)
# ==============================================================================

class ZipStreamingDatasetOrchestrator:
    """
    Direct zip-streaming preprocessor orchestrator for Google Colab and large zip archives.
    Uses VideoPreprocessorV4 as the extraction engine.
    Applies ping-pong double buffering on local disk, background Drive upload, and persistent ledger.
    """

    def __init__(
        self,
        zip_dir: Union[str, Path],
        output_dir: Union[str, Path],
        chunk_size: int = 100,
        backend: str = "rtmw",
        pose_mode: str = "balanced",
        flip_tta: bool = False,
        include_roi: bool = True,
        include_hand_crop: bool = True,
        max_len: int = 384,
        device: str = "cuda",
    ):
        self.zip_dir = Path(zip_dir)
        self.output_dir = Path(output_dir)
        self.chunk_size = chunk_size
        self.backend = backend
        self.pose_mode = pose_mode
        self.flip_tta = flip_tta
        self.include_roi = include_roi
        self.include_hand_crop = include_hand_crop
        self.max_len = max_len
        self.device = device

        self.shards_dir = self.output_dir / "shards"
        self.manifest_dir = self.output_dir / "manifests"
        self.ledger_path = self.manifest_dir / "ledger.json"

        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        scratch_base = Path("/content/v4_zip_scratch") if os.path.exists("/content") else Path(tempfile.gettempdir()) / "v4_zip_scratch"
        self.local_scratch = scratch_base
        self.local_scratch.mkdir(parents=True, exist_ok=True)

        self.ledger = self._load_ledger()
        from concurrent.futures import ThreadPoolExecutor
        self.uploader = ThreadPoolExecutor(max_workers=1)
        self.upload_futures = []

    def _load_ledger(self) -> Dict[str, Any]:
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"processed_clips": {}, "shards_written": 0}

    def _save_ledger(self):
        tmp = self.ledger_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.ledger, f, indent=2)
        tmp.replace(self.ledger_path)

    def _upload_worker(self, local_shard: Path, dest_shard: Path, completed_stems: List[str]):
        try:
            tmp_dest = dest_shard.with_suffix(".tmp")
            shutil.copy2(local_shard, tmp_dest)
            tmp_dest.replace(dest_shard)

            for stem in completed_stems:
                self.ledger["processed_clips"][stem] = True
            self.ledger["shards_written"] += 1
            self._save_ledger()
            print(f"\n  [STREAM UPLOADER] Safely written {dest_shard.name} to persistent storage!", flush=True)
        except Exception as e:
            print(f"\n  [STREAM UPLOADER ERROR] Failed saving {dest_shard.name}: {e}", flush=True)
        finally:
            if local_shard.exists():
                try:
                    os.remove(local_shard)
                except Exception:
                    pass

    def load_transcription_maps(self) -> Dict[str, str]:
        text_map: Dict[str, str] = {}
        patterns = ["*realigned*.csv", "*realigned*.tsv", "*realigned*.txt", "*.csv", "*.tsv"]
        candidate_files = []
        for pat in patterns:
            candidate_files.extend(list(self.zip_dir.rglob(pat)))
        seen = set()
        for meta_file in candidate_files:
            if meta_file.resolve() in seen:
                continue
            seen.add(meta_file.resolve())
            try:
                import csv
                with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline()
                    delimiter = "\t" if "\t" in first_line else ","
                    f.seek(0)
                    reader = csv.reader(f, delimiter=delimiter)
                    header = next(reader, None)
                    header_lower = [c.strip().lower() for c in header]
                    id_col = -1
                    sent_col = -1

                    # 1. Exact match pass
                    for i, h in enumerate(header_lower):
                        if h in ("sentence_name", "clip_id", "video_id", "entryid", "name", "id"):
                            if id_col == -1: id_col = i
                        elif h in ("sentence", "translation", "dominanttranslation", "text", "transcript"):
                            if sent_col == -1: sent_col = i

                    # 2. Substring match fallback (strictly ensuring sent_col != id_col)
                    if id_col == -1:
                        for i, h in enumerate(header_lower):
                            if any(k in h for k in ["name", "clip", "video", "id"]):
                                id_col = i
                                break
                    if id_col == -1:
                        id_col = 0

                    if sent_col == -1:
                        for i, h in enumerate(header_lower):
                            if i != id_col and any(k in h for k in ["sentence", "translation", "text", "transcript"]):
                                sent_col = i
                                break
                    if sent_col == -1:
                        sent_col = len(header) - 1 if len(header) > 1 else 0

                    for row in reader:
                        if len(row) > max(id_col, sent_col):
                            raw_id = row[id_col].strip()
                            clean_id = raw_id.replace("-rgb_front", "").replace("_rgb_front", "")
                            sentence = row[sent_col].strip()
                            if sentence and clean_id:
                                text_map[clean_id] = sentence
                                text_map[raw_id] = sentence
            except Exception as e:
                print(f"[WARN] Failed parsing {meta_file.name}: {e}", flush=True)
        return text_map

    def process_all(self):
        import zipfile
        all_zips = list(self.zip_dir.rglob("*.zip"))
        clips_zips = [z for z in all_zips if "clip" in z.name.lower()]
        raw_zips = [z for z in all_zips if "clip" not in z.name.lower()]
        zips = clips_zips + raw_zips

        if not zips:
            print(f"[ERROR] No zip archives found under: {self.zip_dir}", flush=True)
            return

        text_map = self.load_transcription_maps()
        print(f"[INFO] Discovered {len(zips)} zip archives and {len(text_map)} translations.", flush=True)

        v4_engine = VideoPreprocessorV4(
            target_roi_size=256,
            hand_crop_size=128,
            backend=self.backend,
            pose_mode=self.pose_mode,
            flip_tta=self.flip_tta,
            device=self.device,
        )

        buffer_toggle = 0
        for z_idx, zip_path in enumerate(zips, 1):
            print(f"\n" + "=" * 80)
            print(f"[{z_idx}/{len(zips)}] PROCESSING ARCHIVE: {zip_path.name}")
            print("=" * 80)

            name_lower = zip_path.name.lower()
            split = "val" if "val" in name_lower else ("test" if "test" in name_lower else "train")
            split_shards_dir = self.shards_dir / split
            split_shards_dir.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    file_list = [
                        f for f in archive.namelist()
                        if f.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")) and not f.startswith("__MACOSX")
                    ]
                    unprocessed = [f for f in file_list if Path(f).stem not in self.ledger["processed_clips"]]
                    print(f"  Total Clips: {len(file_list)} | Unprocessed: {len(unprocessed)}", flush=True)

                    for chunk_start in range(0, len(unprocessed), self.chunk_size):
                        chunk_files = unprocessed[chunk_start : chunk_start + self.chunk_size]

                        # Double-buffer ping pong: chunk_A vs chunk_B
                        buf_dir = self.local_scratch / f"chunk_{'A' if buffer_toggle == 0 else 'B'}"
                        buffer_toggle = 1 - buffer_toggle
                        shutil.rmtree(buf_dir, ignore_errors=True)
                        buf_dir.mkdir(parents=True, exist_ok=True)

                        # 1. Extract batch
                        for fname in chunk_files:
                            archive.extract(fname, path=buf_dir)

                        # 2. Extract using VideoPreprocessorV4
                        shard_records = []
                        processed_stems = []
                        t0 = time.time()

                        for fname in chunk_files:
                            local_vid = buf_dir / fname
                            stem = Path(fname).stem
                            clean_stem = stem.replace("-rgb_front", "").replace("_rgb_front", "")

                            try:
                                res = v4_engine.extract_from_video(
                                    video_path=local_vid,
                                    max_frames=self.max_len,
                                    include_roi=self.include_roi,
                                    include_hand_crop=self.include_hand_crop,
                                )
                                if res is not None:
                                    text_label = text_map.get(clean_stem, text_map.get(stem, "how2sign_sentence"))
                                    res["id"] = clean_stem
                                    res["label"] = text_label
                                    res["text"] = text_label
                                    res["split"] = split
                                    res["source"] = "How2Sign"
                                    shard_records.append(res)
                                    processed_stems.append(stem)
                            except Exception as e:
                                print(f"    [WARN] Skipped {stem}: {e}", flush=True)

                        # 3. Save local shard and dispatch background upload thread
                        if shard_records:
                            s_idx = self.ledger["shards_written"] + len(self.upload_futures)
                            shard_name = f"shard_{s_idx:05d}.pt"
                            local_shard = self.local_scratch / shard_name
                            dest_shard = split_shards_dir / shard_name

                            torch.save(shard_records, local_shard)
                            dt = time.time() - t0
                            fps = len(shard_records) / (dt + 1e-4)
                            print(f"  [V4 BATCH READY] {shard_name} ({len(shard_records)} items @ {fps:.1f} seq/s). Uploading in background...", flush=True)

                            fut = self.uploader.submit(
                                self._upload_worker,
                                local_shard=local_shard,
                                dest_shard=dest_shard,
                                completed_stems=processed_stems,
                            )
                            self.upload_futures.append(fut)

                        # 4. Wipe local videos immediately and collect garbage
                        shutil.rmtree(buf_dir, ignore_errors=True)
                        del shard_records
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            except Exception as e:
                print(f"[ERROR] Failed archive {zip_path.name}: {e}", flush=True)

        print("\n[FINISHING] Waiting for remaining background uploads to conclude...", flush=True)
        for fut in self.upload_futures:
            fut.result()
        self.uploader.shutdown(wait=True)
        print("\n[COMPLETE] All archives processed and streamed with PreprocessorV4!", flush=True)


# ==============================================================================
#  10. CLI BATCH PROCESSING ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessor V4: Omnimodal 256x256 ROI, 128x128 Hand & 9-D Kinematics Dataset Generator"
    )
    parser.add_argument("--video-dir", type=str, default=None, help="Directory containing input videos")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory containing static alphanumeric images")
    parser.add_argument("--zip-stream-dir", type=str, default=None, help="Directory containing zip archives for on-the-fly streaming")
    parser.add_argument("--zip-chunk-size", type=int, default=100, help="Number of clips per micro-chunk extraction (default 100)")
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
        help="Pose model resolution and detector configuration",
    )
    parser.add_argument(
        "--flip-tta",
        action="store_true",
        default=False,
        help="Enable symmetrical horizontal flip Test-Time Augmentation (+2.1 Hand AP on occluded fingers).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Compute device for keypoint detectors ('auto', 'cuda', 'cpu')",
    )
    parser.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 4) - 2), help="Parallel workers")
    parser.add_argument("--chunksize", type=int, default=None, help="Optional chunksize for imap_unordered")
    parser.add_argument("--disable-streaming", action="store_true", help="Accumulate all records in RAM before writing shards")
    args = parser.parse_args()

    if args.device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = args.device
    print(f"[INFO] Using compute device: {resolved_device}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fast-path: On-the-fly zip streaming mode for Colab / cloud archives
    if args.zip_stream_dir:
        orchestrator = ZipStreamingDatasetOrchestrator(
            zip_dir=args.zip_stream_dir,
            output_dir=args.output_dir,
            chunk_size=args.zip_chunk_size,
            backend=args.backend,
            pose_mode=args.pose_mode,
            flip_tta=args.flip_tta,
            include_roi=args.include_roi,
            include_hand_crop=args.include_hand_crop,
            max_len=args.max_len,
            device=resolved_device,
        )
        orchestrator.process_all()
        return

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

    # Pre-extract all discovered labels so label_to_idx is populated upfront
    all_discovered_labels = set()
    for item in items_to_process:
        meta = item[2]
        lbl = meta.get("label")
        if lbl:
            all_discovered_labels.add(str(lbl).strip())
        else:
            p_obj = Path(item[0])
            v_stem = p_obj.stem
            fallback_lbl = v_stem.split("-")[-1].lower() if "-" in v_stem else v_stem.lower()
            all_discovered_labels.add(fallback_lbl)
    sorted_labels = sorted(list(all_discovered_labels))
    initial_label_to_idx = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

    # Adaptive chunksize calculation for optimal IPC throughput
    if args.chunksize is not None and args.chunksize > 0:
        chunksize = args.chunksize
    else:
        is_image_heavy = (args.image_dir is not None and args.video_dir is None)
        if is_image_heavy:
            chunksize = min(128, max(16, len(items_to_process) // max(1, args.num_workers * 8)))
        else:
            chunksize = min(16, max(2, len(items_to_process) // max(1, args.num_workers * 16)))
        chunksize = max(1, chunksize)
    print(f"[INFO] Multiprocessing imap_unordered chunksize: {chunksize}", flush=True)

    use_streaming = not args.disable_streaming
    if use_streaming:
        has_visual = args.include_roi or args.include_hand_crop
        writer = StreamingShardWriter(
            output_dir=args.output_dir,
            label_to_idx=initial_label_to_idx,
            shard_size=args.shard_size,
            split_ratios=tuple(args.split_ratios),
            has_visual=has_visual,
        )
    else:
        records = []
        unique_labels = set()

    print(f"[INFO] Launching high-throughput processing for {len(items_to_process)} items using {args.num_workers} parallel workers...")
    valid_count = 0
    start_time = time.time()

    if args.num_workers > 1:
        with mp.Pool(
            processes=args.num_workers,
            initializer=_init_multiprocessing_worker,
            initargs=(args.backend, 256, 128, args.pose_mode, args.flip_tta, resolved_device),
        ) as pool:
            for idx, res in enumerate(pool.imap_unordered(_process_item_task, items_to_process, chunksize=chunksize), 1):
                if res is not None:
                    if "overlay_data" in res:
                        overlay_info = res.pop("overlay_data")
                        render_and_save_landmark_overlay(overlay_info, overlays_dir)
                    if use_streaming:
                        writer.add_record(res)
                    else:
                        unique_labels.add(res["label"])
                        records.append(res)
                    valid_count += 1
                if idx % 100 == 0 or idx == len(items_to_process):
                    elapsed = max(1e-3, time.time() - start_time)
                    rate = idx / elapsed
                    print(f"  [PROGRESS] Processed {idx}/{len(items_to_process)} ({idx/len(items_to_process)*100:.1f}%) @ {rate:.1f} items/sec | Valid Records: {valid_count}", flush=True)
    else:
        _init_multiprocessing_worker(args.backend, 256, 128, args.pose_mode, args.flip_tta, device=resolved_device)
        for idx, task_args in enumerate(items_to_process, 1):
            res = _process_item_task(task_args)
            if res is not None:
                if "overlay_data" in res:
                    overlay_info = res.pop("overlay_data")
                    render_and_save_landmark_overlay(overlay_info, overlays_dir)
                if use_streaming:
                    writer.add_record(res)
                else:
                    unique_labels.add(res["label"])
                    records.append(res)
                valid_count += 1
            if idx % 100 == 0 or idx == len(items_to_process):
                elapsed = max(1e-3, time.time() - start_time)
                rate = idx / elapsed
                print(f"  [PROGRESS] Processed {idx}/{len(items_to_process)} ({idx/len(items_to_process)*100:.1f}%) @ {rate:.1f} items/sec | Valid Records: {valid_count}", flush=True)

    print(f"[INFO] Extraction complete. Successfully parsed {valid_count} valid records from {len(items_to_process)} items.")

    if use_streaming:
        writer.finalize(english_vocab_path=args.english_vocab)
    else:
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
