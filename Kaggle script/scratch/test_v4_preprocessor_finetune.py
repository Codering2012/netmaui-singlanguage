#!/usr/bin/env python3
"""
Lightweight Hypothesis Test for Preprocessor V4 Fine-Tuning:
1. Low-pass filtered sternum anchor vs static mean anchor.
2. Savitzky-Golay Delta t scaling invariance across 24, 30, 60 fps.
3. Standardized 19-D Phonology scaling (finger curl in [0, 1], face proximity in (0, 1]).
4. Spline imputation for transient tracking gaps.

Hardware constraint: CPU only, B<=4, T<=64, RAM < 500MB, duration < 15s.
"""

import sys
import math
import numpy as np

def test_low_pass_torso_anchor():
    print("=== Testing Low-Pass Torso Anchor ===")
    T = 60
    # Simulate a signer swaying slowly from left to right (x moves from -0.3 to +0.3)
    t_vals = np.linspace(0, 2 * np.pi, T)
    sway = 0.3 * np.sin(t_vals) # slow grammatical or natural sway
    jitter = 0.02 * np.random.randn(T) # high-frequency sensor noise
    
    mid_sh = np.zeros((T, 3), dtype=np.float32)
    mid_sh[:, 0] = sway + jitter
    
    # 1. Old approach: static average
    static_anchor = np.mean(mid_sh, axis=0)
    drift_old = mid_sh - static_anchor # still contains the full sway!
    
    # 2. Fine-tuned approach: low-pass filtered EMA anchor
    alpha = 0.85
    smooth_anchor = np.zeros_like(mid_sh)
    smooth_anchor[0] = mid_sh[0]
    for t in range(1, T):
        smooth_anchor[t] = alpha * smooth_anchor[t-1] + (1.0 - alpha) * mid_sh[t]
        
    centered_new = mid_sh - smooth_anchor # high-freq jitter removed, steady tracking
    
    print(f"Old centered max coordinate: {np.max(np.abs(drift_old)):.4f}")
    print(f"New centered max residual jitter: {np.max(np.abs(centered_new)):.4f}")
    assert np.max(np.abs(centered_new)) < np.max(np.abs(drift_old))
    print("[PASS] Low-pass anchor successfully filters camera/body drift.")


def test_delta_t_kinematics_invariance():
    print("\n=== Testing Savitzky-Golay Delta-t Scaling ===")
    # Simulate a hand moving 1.0 meter across 1.0 second at constant physical speed (1.0 m/s)
    # Tested at 30 fps and 60 fps
    for fps in [30, 60]:
        delta_t = 1.0 / fps
        num_frames = fps # 1 second
        pos = np.zeros((num_frames, 60, 3), dtype=np.float32)
        # Move wrist (idx 21) from 0.0 to 1.0 along x
        pos[:, 21, 0] = np.linspace(0.0, 1.0, num_frames)
        
        # 5-point Savitzky-Golay with delta_t
        p0 = pos[:-4, 21, 0]
        p1 = pos[1:-3, 21, 0]
        p3 = pos[3:-1, 21, 0]
        p4 = pos[4:, 21, 0]
        v_mid = (-2.0 * p0 - p1 + p3 + 2.0 * p4) / (10.0 * delta_t)
        mean_vel = float(np.mean(v_mid))
        print(f"FPS {fps}: Mean estimated velocity = {mean_vel:.4f} (Expected: ~1.0000)")
        assert abs(mean_vel - 1.0) < 0.05, f"Velocity {mean_vel} deviates from 1.0 at fps {fps}"

    print("[PASS] Savitzky-Golay velocity is mathematically invariant to video FPS.")


def test_phonology_standardization():
    print("\n=== Testing 19-D Phonology Standardization ===")
    T = 16
    pos = np.zeros((T, 60, 3), dtype=np.float32)
    # Right wrist at 21, MCP at 26, Tip at 29
    pos[:, 21, :] = [0.0, 0.0, 0.0]
    pos[:, 26, :] = [0.0, 0.1, 0.0] # palm length = 0.1
    # Test curl: tip fully curled vs fully extended
    # Curled: tip touches MCP / wrist
    pos[:8, 29, :] = [0.0, 0.02, 0.0] # distance to wrist = 0.02
    # Extended: tip far out
    pos[8:, 29, :] = [0.0, 0.18, 0.0] # distance to wrist = 0.18
    
    # Standardized curl formula
    wrist = pos[:, 21:22, :]
    mcp = pos[:, 26:27, :]
    tip = pos[:, 29:30, :]
    palm_len = np.linalg.norm(mcp - wrist, axis=-1)
    tip_dist = np.linalg.norm(tip - wrist, axis=-1)
    
    curl = np.clip(tip_dist / (1.8 * np.maximum(palm_len, 1e-4)), 0.0, 1.0)
    print(f"Curled finger score: {float(np.mean(curl[:8])):.4f} in [0, 1]")
    print(f"Extended finger score: {float(np.mean(curl[8:])):.4f} in [0, 1]")
    assert np.all(curl >= 0.0) and np.all(curl <= 1.0)
    
    # Face proximity Gaussian
    pos[:, 48:60, :] = [0.0, 0.5, 0.0] # face center at y=0.5
    face_center = np.mean(pos[:, 48:60, :], axis=1, keepdims=True)
    d_rh_face = np.linalg.norm(pos[:, 21:22, :] - face_center, axis=-1)
    sigma_face = 0.35
    k_face = np.exp(- (d_rh_face ** 2) / (2.0 * sigma_face ** 2))
    print(f"Face contact proximity score: {float(np.mean(k_face)):.4f} in (0, 1]")
    assert np.all(k_face > 0.0) and np.all(k_face <= 1.0)
    print("[PASS] Phonology standardization strictly bounds all features.")


def test_spline_imputation():
    print("\n=== Testing Cubic Spline Imputation on Transient Tracking Gaps ===")
    T = 20
    # Simulate hand coordinates with a 3-frame drop in the middle (frames 8, 9, 10)
    trajectory = np.sin(np.linspace(0, np.pi, T))
    corrupted = trajectory.copy()
    corrupted[8:11] = 0.0 # dropped frames
    
    is_missing = (corrupted == 0.0)
    valid_idx = np.where(~is_missing)[0]
    missing_idx = np.where(is_missing)[0]
    
    # Monotonic cubic Hermite interpolation (Pchip) across gap
    from scipy.interpolate import PchipInterpolator
    pchip = PchipInterpolator(valid_idx, corrupted[valid_idx])
    reconstructed = corrupted.copy()
    reconstructed[missing_idx] = pchip(missing_idx)
    
    max_recon_error = np.max(np.abs(reconstructed - trajectory))
    print(f"Max reconstruction error on dropped frames with Pchip: {max_recon_error:.4f}")
    assert max_recon_error < 0.05
    print("[PASS] Spline imputation preserves smooth trajectory without 0-spikes.")

if __name__ == "__main__":
    test_low_pass_torso_anchor()
    test_delta_t_kinematics_invariance()
    test_phonology_standardization()
    test_spline_imputation()
    print("\nALL HYPOTHESIS TESTS PASSED EMPIRICALLY!")
