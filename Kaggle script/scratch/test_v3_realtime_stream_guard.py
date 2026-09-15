#!/usr/bin/env python3
"""
Empirical Verification of Real-Time Stream Guard Innovations:
1. BiAcromialMetricNormalizer (Optical scale & zoom invariance)
2. ContinuousKinematicsNormalizer (Framerate jitter & dropped frame delta-t invariance)
3. HandednessContinuityTracker (Hand crossing swap detection & repair)
4. ConversationalBackchannelGate (Head nod backchannel suppression)
5. RealtimeStreamGuard Unified Pipeline
"""

import sys
import time
import math
from pathlib import Path
import torch
import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from train_tpu.v3.modules import (
    BiAcromialMetricNormalizer,
    ContinuousKinematicsNormalizer,
    HandednessContinuityTracker,
    ConversationalBackchannelGate,
    RealtimeStreamGuard,
)


def test_realtime_stream_guard():
    print("=" * 80)
    print("RUNNING REAL-TIME STREAM GUARD EMPIRICAL VERIFICATION")
    print("=" * 80)
    start_time = time.time()
    torch.manual_seed(42)

    # --------------------------------------------------------------------------
    # TEST 1: Bi-Acromial Metric Normalizer (Scale & Zoom Invariance)
    # --------------------------------------------------------------------------
    print("\n[TEST 1/5] Testing Bi-Acromial Metric Normalizer (BAMN)...")
    bamn = BiAcromialMetricNormalizer(left_shoulder_idx=42, right_shoulder_idx=43)

    # Base pose at distance Z = 1.0m: shoulders at (-0.2, 0.2, 0), (+0.2, 0.2, 0) -> width = 0.4m
    base_pose = torch.randn(60, 3) * 0.1
    base_pose[42] = torch.tensor([-0.2, 0.2, 0.0])
    base_pose[43] = torch.tensor([0.2, 0.2, 0.0])
    base_pose[21] = torch.tensor([0.15, 0.35, 0.05])  # right wrist

    # Same pose at distance Z = 2.5m (zoomed out: coordinates scaled by 1/2.5 = 0.4)
    zoomed_pose = base_pose * 0.4

    norm_base, width_base = bamn.normalize(base_pose)
    norm_zoomed, width_zoomed = bamn.normalize(zoomed_pose)

    diff = torch.norm(norm_base - norm_zoomed).item()
    print(f"  -> Raw zoom scale difference: {width_base.item():.4f} vs {width_zoomed.item():.4f}")
    print(f"  -> Max difference in normalized coordinates: {diff:.6e}")
    assert diff < 1e-4, f"BAMN failed scale invariance! Diff: {diff}"
    print("  [PASS] Perfect optical depth & zoom scale invariance verified!")

    # --------------------------------------------------------------------------
    # TEST 2: Continuous Kinematics with Variable Timestamps (T-TICK)
    # --------------------------------------------------------------------------
    print("\n[TEST 2/5] Testing Continuous Kinematics Normalizer (T-TICK)...")
    ttick = ContinuousKinematicsNormalizer(default_fps=30.0)

    # Simulate constant velocity hand movement: v = 1.0 m/s along X
    # Frames arrive with jitter and a dropped frame:
    # t = [0.0, 0.033, 0.066, 0.132 (dropped frame!), 0.165, 0.198]
    timestamps = [0.0, 0.033, 0.066, 0.132, 0.165, 0.198]
    velocities_measured = []

    pos = torch.ones(60, 3) * 0.5

    for t in timestamps:
        pos[21, 0] = 0.5 + 1.0 * t  # x = x0 + v * t
        kin_9d = ttick.step(pos, timestamp=t)
        v_x = kin_9d[21, 3].item()
        if t > 0.0:
            velocities_measured.append(v_x)

    print(f"  -> Measured X-velocities across jitter/dropped frames: {[round(v, 3) for v in velocities_measured]}")
    # Verify all measured velocities are close to 1.0 m/s
    for v in velocities_measured:
        assert abs(v - 1.0) < 0.05, f"Velocity calculation warped by jitter! Measured: {v}"
    print("  [PASS] Frame-drop and jitter-invariant continuous kinematics verified!")

    # --------------------------------------------------------------------------
    # TEST 3: Handedness Continuity & Identity Disambiguation (ACHD)
    # --------------------------------------------------------------------------
    print("\n[TEST 3/5] Testing Handedness Continuity Tracker (ACHD)...")
    achd = HandednessContinuityTracker(left_wrist_idx=0, right_wrist_idx=21)

    # Frame 1: Normal positions: Left at x = -0.3, Right at x = +0.3
    frame1 = torch.zeros(60, 3)
    frame1[0] = torch.tensor([-0.3, 0.0, 0.0])
    frame1[21] = torch.tensor([0.3, 0.0, 0.0])
    _, swapped1 = achd.disambiguate_and_repair(frame1)
    assert not swapped1

    # Frame 2: Hands approach near center: Left at x = -0.05, Right at x = +0.05
    frame2 = torch.zeros(60, 3)
    frame2[0] = torch.tensor([-0.05, 0.0, 0.0])
    frame2[21] = torch.tensor([0.05, 0.0, 0.0])
    _, swapped2 = achd.disambiguate_and_repair(frame2)
    assert not swapped2

    # Frame 3: TRACKER BUG: Tracker swaps IDs!
    # Hand on right (+0.3) is erroneously labeled as Left hand (index 0)
    # Hand on left (-0.3) is erroneously labeled as Right hand (index 21)
    frame3_buggy = torch.zeros(60, 3)
    frame3_buggy[0] = torch.tensor([0.3, 0.0, 0.0])
    frame3_buggy[21] = torch.tensor([-0.3, 0.0, 0.0])

    repaired_frame3, was_swapped = achd.disambiguate_and_repair(frame3_buggy)
    print(f"  -> Hand swap detected: {was_swapped}")
    assert was_swapped, "ACHD failed to detect tracker identity inversion!"
    # Verify that repaired frame has Left hand on left (negative x) and Right hand on right (positive x)
    assert repaired_frame3[0, 0] < 0, f"Left wrist should be at negative x! Got {repaired_frame3[0, 0]}"
    assert repaired_frame3[21, 0] > 0, f"Right wrist should be at positive x! Got {repaired_frame3[21, 0]}"
    print("  [PASS] Two-handed crossing swap detected and repaired seamlessly!")

    # --------------------------------------------------------------------------
    # TEST 4: Conversational Backchannel & Turn-Holding Gate (BSTG)
    # --------------------------------------------------------------------------
    print("\n[TEST 4/5] Testing Conversational Backchannel Gate (BSTG)...")
    bstg = ConversationalBackchannelGate(window_size=15)

    # A. Listener nodding head at 2 Hz with hands in lap
    # 15 frames at 30 FPS = 0.5s = 1 full cycle of a 2 Hz nod: sin(2 * pi * 2 * t)
    is_bc_detected = False
    for i in range(15):
        t = i / 30.0
        pitch_vel = math.sin(2.0 * math.pi * 2.0 * t) * 0.3
        is_bc = bstg.evaluate_backchannel(
            cranial_pitch_velocity=pitch_vel,
            hand_elevation=-0.3,  # hands resting in lap
            hand_kinetic_energy=0.01,  # hands still
        )
        if is_bc:
            is_bc_detected = True

    print(f"  -> Listener head-nod backchannel detected: {is_bc_detected}")
    assert is_bc_detected, "BSTG failed to identify passive conversational head-nod!"
    print("  [PASS] Listener backchannel correctly suppressed from emitting text.")

    # B. Active signer nodding for emphasis while signing actively
    bstg.reset()
    is_active_signing_blocked = False
    for i in range(15):
        t = i / 30.0
        pitch_vel = math.sin(2.0 * math.pi * 2.0 * t) * 0.3
        is_bc = bstg.evaluate_backchannel(
            cranial_pitch_velocity=pitch_vel,
            hand_elevation=+0.25,  # hands raised in signing space!
            hand_kinetic_energy=0.40,  # dynamic sign stroke!
        )
        if is_bc:
            is_active_signing_blocked = True

    assert not is_active_signing_blocked, "BSTG incorrectly blocked active signer emphatic nod!"
    print("  [PASS] Emphatic linguistic nods during active signing correctly permitted.")

    # --------------------------------------------------------------------------
    # TEST 5: RealtimeStreamGuard Unified Pipeline
    # --------------------------------------------------------------------------
    print("\n[TEST 5/5] Testing RealtimeStreamGuard End-to-End Pipeline...")
    guard = RealtimeStreamGuard()

    raw_pts = torch.randn(60, 3) * 0.05
    raw_pts[42] = torch.tensor([-0.25, 0.2, 0.0])
    raw_pts[43] = torch.tensor([0.25, 0.2, 0.0])
    raw_pts[21] = torch.tensor([0.15, 0.35, 0.05])

    out = guard.process_frame(raw_pts, timestamp=0.033, cranial_pitch_vel=0.0)
    assert out["kinematics"].shape == (60, 9)
    assert out["normalized_pts"].shape == (60, 3)
    assert "was_hand_swapped" in out
    assert "is_backchannel" in out
    print("  [PASS] Unified RealtimeStreamGuard pipeline verified!")

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] All Real-Time Stream Guard Tests Passed in {elapsed:.2f} seconds!")
    print("=" * 80)


if __name__ == "__main__":
    test_realtime_stream_guard()
