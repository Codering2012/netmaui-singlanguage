#!/usr/bin/env python3
"""
Empirical Benchmark of Dual Visual Stems (256x256 ROI + 128x128 Hand Crop)
with DynamicComputeGovernor on Local CPU.
"""

import sys
import time
from pathlib import Path
import torch
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from train_tpu.v3.modules.asl_v3_foundation_model import VisualROI256Stem, VisualHandCrop128Stem
from train_tpu.v3.modules.dynamic_compute_governor import DynamicComputeGovernor


def benchmark_visual_stems():
    print("=" * 80)
    print("BENCHMARKING VISUAL CNN STEMS & DYNAMIC COMPUTE GOVERNOR")
    print("=" * 80)
    torch.manual_seed(42)

    roi_stem = VisualROI256Stem(d_model=128)
    hand_stem = VisualHandCrop128Stem(d_model=128)
    roi_stem.eval()
    hand_stem.eval()

    governor = DynamicComputeGovernor(d_model=128, velocity_threshold=0.10, idle_sleep_frames=5)

    # 1. Benchmark single frame forward pass of Depthwise-Separable Visual Stems
    dummy_roi = torch.randn(1, 1, 3, 256, 256)
    dummy_hand = torch.randn(1, 1, 3, 128, 128)

    # Warm-up
    for _ in range(3):
        with torch.no_grad():
            _ = roi_stem(dummy_roi)
            _ = hand_stem(dummy_hand)

    # Measure 30 active frames
    latencies_active = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            _ = roi_stem(dummy_roi)
            _ = hand_stem(dummy_hand)
            t1 = time.perf_counter()
            latencies_active.append((t1 - t0) * 1000.0)

    mean_active = np.mean(latencies_active)
    print(f"Active Full Dual Visual Stems (256x256 + 128x128) CPU Latency: {mean_active:.2f} ms")

    # 2. Measure Gated Mode (when hands are resting or static hold)
    dummy_kin_idle = torch.zeros(1, 1, 60, 9)
    latencies_gated = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            f_roi, f_hand, meta = governor(roi_stem, hand_stem, dummy_roi, dummy_hand, dummy_kin_idle)
            t1 = time.perf_counter()
            latencies_gated.append((t1 - t0) * 1000.0)

    mean_gated = np.mean(latencies_gated)
    print(f"Gated Deep Sleep Mode (Governor Active) CPU Latency:            {mean_gated:.4f} ms")
    print(f"FLOPs & Energy Saved in Sleep Mode:                            {meta['saved_flops_ratio'] * 100:.1f}%")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_visual_stems()
