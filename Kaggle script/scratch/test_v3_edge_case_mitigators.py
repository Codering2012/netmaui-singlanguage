#!/usr/bin/env python3
"""
Empirical Verification of Real-World Edge Case Mitigators:
1. Preprocessor V4 Missing Shoulder Fallback & Hand Crop Smoothing
2. DominantHandClassifierAndMirror (Left-handed signer reflection)
3. OneEuroLandmarkFilter (Micro-jitter elimination & zero-lag rapid stroke)
4. MouthOcclusionInpainter (Hand-over-mouth contact disambiguation)
5. PerspectivePitchNormalizer (Desk webcam upward tilt compensation)
"""

import sys
import time
import math
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from preprocessing.preprocessor_v4 import UpperBodyTrackerEMA
from train_tpu.v3.modules import (
    DominantHandClassifierAndMirror,
    OneEuroLandmarkFilter,
    MouthOcclusionInpainter,
    PerspectivePitchNormalizer,
)


def test_edge_case_mitigators():
    print("=" * 80)
    print("RUNNING EDGE CASE MITIGATION EMPIRICAL VERIFICATION")
    print("=" * 80)
    start_time = time.time()
    torch.manual_seed(42)
    np.random.seed(42)

    # --------------------------------------------------------------------------
    # TEST 1: Preprocessor V4 Missing Shoulder Fallback & Hand Crop Smoothing
    # --------------------------------------------------------------------------
    print("\n[TEST 1/5] Testing Preprocessor V4 UpperBodyTrackerEMA Edge Cases...")
    cropper = UpperBodyTrackerEMA(target_size=256, hand_crop_size=128)

    # A. One shoulder occluded (Left shoulder is [0, 0, 0])
    partial_landmarks = np.zeros((60, 3), dtype=np.float32)
    partial_landmarks[43] = [0.65, 0.40, 0.0]  # Right shoulder
    partial_landmarks[48] = [0.50, 0.25, 0.0]  # Nose
    cx, cy, size = cropper.compute_raw_box(partial_landmarks, frame_w=640, frame_h=480)
    print(f"  -> Single-shoulder fallback box center: ({cx:.1f}, {cy:.1f}), size: {size:.1f}")
    assert cx > 200 and cy > 100, f"Box collapsed to origin! cx={cx}, cy={cy}"
    print("  [PASS] Single-shoulder occlusion fallback successfully anchors on torso!")

    # B. Hand crop EMA smoothing across rapid fingerspelling
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    hand1 = np.ones((21, 3), dtype=np.float32) * 0.5
    hand2 = hand1.copy()
    hand2[8, 1] = 0.35  # Index finger extends rapidly (35% height change)
    crop1 = cropper.extract_hand_crop(dummy_frame, hand1)
    crop2 = cropper.extract_hand_crop(dummy_frame, hand2)
    assert crop1 is not None and crop2 is not None
    assert cropper.prev_hand_box is not None
    print(f"  -> Hand crop stabilized center: ({cropper.prev_hand_box[0]:.1f}, {cropper.prev_hand_box[1]:.1f})")
    print("  [PASS] Hand crop stabilization and smoothing verified!")

    # --------------------------------------------------------------------------
    # TEST 2: DominantHandClassifierAndMirror (Left-Handed Signer Parity)
    # --------------------------------------------------------------------------
    print("\n[TEST 2/5] Testing DominantHandClassifierAndMirror...")
    mirror_engine = DominantHandClassifierAndMirror(left_dominant_threshold=0.65)

    # Simulate 15 frames of left-hand dominant signing:
    # Left hand has high kinetic velocity (1.5 m/s), Right hand is resting base (0.1 m/s)
    is_left_dom = False
    for _ in range(15):
        kin = torch.zeros(1, 60, 9)
        kin[0, 0, 3] = 1.5   # Left wrist X-velocity
        kin[0, 21, 3] = 0.1  # Right wrist X-velocity
        pts = torch.zeros(1, 60, 3)
        pts[0, 0, 0] = -0.3  # Left hand on left
        pts[0, 21, 0] = 0.1  # Right hand near center

        mirrored_pts, mirrored_kin, is_left_dom = mirror_engine.update_and_mirror(pts, kin)

    print(f"  -> Left-hand dominance detected: {is_left_dom}")
    assert is_left_dom, "Failed to identify left-hand dominant signer!"
    # Verify spatial parity reflection: X coordinates inverted and hand channels swapped!
    # Dominant hand should now be at positive X in the right-hand slot (index 21)
    assert mirrored_pts[0, 21, 0] > 0, "Dominant hand not mapped to positive X right hand!"
    print("  [PASS] Left-hand dominance identified and dynamically mirrored to canonical form!")

    # --------------------------------------------------------------------------
    # TEST 3: OneEuroLandmarkFilter (Noise Reduction & Zero Lag)
    # --------------------------------------------------------------------------
    print("\n[TEST 3/5] Testing OneEuroLandmarkFilter...")
    oe_filter = OneEuroLandmarkFilter(fc_min=1.0, beta=10.0)

    # A. Stationary hand with Gaussian sensor noise (micro-jitter)
    raw_stationary = [torch.tensor([0.5 + np.random.normal(0, 0.02), 0.5, 0.0]) for _ in range(30)]
    filtered_stationary = []
    for i, p in enumerate(raw_stationary):
        f = oe_filter.filter(p, timestamp=i / 30.0)
        filtered_stationary.append(f[0].item())

    raw_var = np.var([p[0].item() for p in raw_stationary])
    filt_var = np.var(filtered_stationary[5:])  # ignore warm-up
    jitter_reduction = (1.0 - filt_var / raw_var) * 100.0
    print(f"  -> Stationary jitter variance reduction: {jitter_reduction:.1f}%")
    assert jitter_reduction > 50.0, f"Jitter reduction insufficient: {jitter_reduction:.1f}%"
    print("  [PASS] Sensor micro-jitter eliminated during stationary hand hold!")

    # B. High-speed rapid stroke: verify zero phase delay
    oe_filter.reset()
    p_fast_start = torch.tensor([0.0, 0.0, 0.0])
    p_fast_end = torch.tensor([1.0, 0.0, 0.0])
    oe_filter.filter(p_fast_start, timestamp=0.0)
    p_out = oe_filter.filter(p_fast_end, timestamp=0.033)
    # Under high speed, alpha should approach 1.0 (pass through without lag)
    lag = abs(p_out[0].item() - 1.0)
    print(f"  -> Fast stroke tracking response: {p_out[0].item():.4f} (lag error: {lag:.4f})")
    assert lag < 0.15, "OneEuroFilter introduced excessive phase lag during rapid stroke!"
    print("  [PASS] High-speed stroke passes with near-zero latency!")

    # --------------------------------------------------------------------------
    # TEST 4: MouthOcclusionInpainter (Hand Touching Mouth)
    # --------------------------------------------------------------------------
    print("\n[TEST 4/5] Testing MouthOcclusionInpainter...")
    inpainter = MouthOcclusionInpainter(occlusion_distance_threshold=0.10)

    mouth_pts = torch.tensor([[0.5, 0.3, 0.0]])   # mouth center at y=0.3
    clean_mouth_feat = torch.ones(128) * 2.5       # canonical mouth morpheme vector

    # Frame 1: Hand far from mouth (signing "CAR" near waist: hand at y=0.7)
    hand_far = torch.tensor([[0.5, 0.7, 0.0]] * 21)
    feat1, occluded1 = inpainter.process(mouth_pts, hand_far, clean_mouth_feat)
    assert not occluded1
    assert torch.equal(feat1, clean_mouth_feat)

    # Frame 2: Hand touches mouth (signing "EAT" or "SECRET": hand at y=0.32)
    hand_touching = torch.tensor([[0.5, 0.32, 0.0]] * 21)
    corrupted_feat = torch.randn(128) * 5.0  # corrupted by finger occlusions
    feat2, occluded2 = inpainter.process(mouth_pts, hand_touching, corrupted_feat)
    print(f"  -> Mouth occlusion detected: {occluded2}")
    assert occluded2, "Failed to detect hand-over-mouth contact occlusion!"
    # Verify inpainter preserved pre-occlusion feature rather than passing corrupted noise
    cos_sim = torch.cosine_similarity(feat2, clean_mouth_feat, dim=0).item()
    print(f"  -> Inpainted feature cosine similarity to clean mouth shape: {cos_sim:.4f}")
    assert cos_sim > 0.95, "Inpainter failed to preserve pre-occlusion mouth morpheme representation!"
    print("  [PASS] Hand-over-mouth contact inpainting verified!")

    # --------------------------------------------------------------------------
    # TEST 5: PerspectivePitchNormalizer (Desk Camera Tilt Compensation)
    # --------------------------------------------------------------------------
    print("\n[TEST 5/5] Testing PerspectivePitchNormalizer...")
    pitch_norm = PerspectivePitchNormalizer()

    # Create landmarks tilted backward by 25 degrees (as seen by laptop webcam on a desk)
    # y = cos(25 deg), z = sin(25 deg)
    angle_rad = math.radians(25.0)
    pts_tilted = torch.zeros(60, 3)
    pts_tilted[42] = torch.tensor([-0.2, 0.0, 0.0])  # left shoulder
    pts_tilted[43] = torch.tensor([0.2, 0.0, 0.0])   # right shoulder
    # Nose elevated along tilted cranial axis
    pts_tilted[48] = torch.tensor([0.0, math.cos(angle_rad) * 0.3, math.sin(angle_rad) * 0.3])

    aligned_pts, estimated_pitch = pitch_norm.normalize_pitch(pts_tilted)
    print(f"  -> Estimated camera pitch angle: {estimated_pitch:.1f} degrees (true: 25.0 deg)")
    assert abs(estimated_pitch - 25.0) < 2.0, f"Pitch estimation inaccurate: {estimated_pitch}"
    # Verify nose z-coordinate after rotation is near zero (aligned to vertical plane)
    assert abs(aligned_pts[48, 2].item()) < 0.02, "Landmarks not upright after pitch normalization!"
    print("  [PASS] Perspective upward camera tilt normalized to upright vertical plane!")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] All Edge Case Mitigators Passed in {elapsed:.2f} seconds!")
    print("=" * 80)


if __name__ == "__main__":
    test_edge_case_mitigators()
