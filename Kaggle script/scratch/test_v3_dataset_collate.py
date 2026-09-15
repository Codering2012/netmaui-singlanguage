#!/usr/bin/env python3
"""
Tests fast_vectorized_v3_collate_fn with all multi-tier annotations.
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from train_tpu.v3.engine.dataset import fast_vectorized_v3_collate_fn

def test_v3_collate():
    batch = [
        {
            "feature": torch.randn(16, 60, 9),
            "roi_visual": torch.randint(0, 255, (16, 3, 256, 256), dtype=torch.uint8),
            "hand_visual": torch.randint(0, 255, (16, 3, 128, 128), dtype=torch.uint8),
            "phonology": torch.randn(16, 19),
            "cranial_imu": torch.randn(16, 3),
            "face_landmarks": torch.randn(16, 12, 3),
            "label_idx": 4,
            "text": "I do not want to go",
        },
        {
            "feature": torch.randn(20, 60, 9),
            "roi_visual": torch.randint(0, 255, (20, 3, 256, 256), dtype=torch.uint8),
            "hand_visual": torch.randint(0, 255, (20, 3, 128, 128), dtype=torch.uint8),
            "phonology": torch.randn(20, 19),
            "cranial_imu": torch.randn(20, 3),
            "face_landmarks": torch.randn(20, 12, 3),
            "label_idx": 7,
            "text": "The dog is playing outside",
        },
    ]

    collated = fast_vectorized_v3_collate_fn(batch)
    print("V3 Collation Results:")
    print("  kinematics shape:", collated["kinematics"].shape)
    print("  roi_visual shape:", collated["roi_visual"].shape)
    print("  hand_visual shape:", collated["hand_visual"].shape)
    print("  cranial_imu shape:", collated["cranial_imu"].shape)
    print("  face_landmarks shape:", collated["face_landmarks"].shape)
    print("  text_is_negative:", collated["text_is_negative"])

    assert collated["kinematics"].shape == (2, 20, 60, 9)
    assert collated["cranial_imu"].shape == (2, 20, 3)
    assert collated["face_landmarks"].shape == (2, 20, 12, 3)
    assert collated["text_is_negative"][0] == True  # "do not"
    assert collated["text_is_negative"][1] == False
    print("[PASS] V3 collation successfully verified!")

if __name__ == "__main__":
    test_v3_collate()
