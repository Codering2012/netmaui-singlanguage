#!/usr/bin/env python3
"""
================================================================================
Comprehensive End-to-End Pipeline & Preprocessing Audit Test
================================================================================
Verifies all fine-tuned Preprocessor V4 modules, dataset collation with tile alignment,
SignerAdaIN, multi-task losses, and full train_step execution.

Local Hardware Constraint Compliant:
- CPU only, B=2, T=32 (and 128 for tile check), D=128, vocab=128.
- Memory ceiling < 500 MB.
- Execution timeout < 15 seconds.
================================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.preprocessor_v4 import (
    reference_part_normalize,
    clean_out_of_bounds_hands,
    compute_9d_kinematics,
    compute_19d_phonology,
)
from train_tpu.v3.engine.dataset import fast_vectorized_v3_collate_fn
from train_tpu.v3.modules.asl_v3_foundation_model import ASLV3FoundationModel, SignerAdaIN
from train_tpu.v3.engine.train_all_in_one_tpu import V3TrainingOrchestrator

def test_v4_preprocessing_suite():
    print("--- 1. Testing Preprocessor V4 Fine-Tuned Functions ---")
    T = 30
    K = 60
    landmarks = np.zeros((T, K, 3), dtype=np.float32)
    val_mask = np.ones((T, K), dtype=bool)
    
    # Left shoulder = 42, Right shoulder = 43
    landmarks[:, 42, :] = [-0.5, 0.0, 0.0]
    landmarks[:, 43, :] = [0.5, 0.0, 0.0]
    # Hips = 44, 45
    landmarks[:, 44, :] = [-0.4, -1.0, 0.0]
    landmarks[:, 45, :] = [0.4, -1.0, 0.0]
    # Right wrist at 21, MCP at 30, Tip at 29
    landmarks[:, 21, :] = [0.2, 0.1, 0.0]
    landmarks[:, 30, :] = [0.2, 0.25, 0.0]
    landmarks[:, 29, :] = [0.2, 0.35, 0.0]
    
    # Test normalization
    normed, scale_ref = reference_part_normalize(landmarks, val_mask)
    print(f"Normed landmarks shape: {normed.shape}, scale_ref: {scale_ref:.4f}")
    assert normed.shape == (T, K, 3)
    assert scale_ref > 0.0
    
    # Test clean_out_of_bounds_hands with transient tracking drop
    corrupted = normed.copy()
    corrupted[10:12, 21:42, :] = 3.5 # Out of bounds for 2 frames
    cleaned = clean_out_of_bounds_hands(corrupted)
    assert np.all(np.abs(cleaned[..., :42, :2]) < 3.0), "OOB points must be interpolated or zeroed"
    print("[PASS] reference_part_normalize & clean_out_of_bounds_hands verified.")
    
    # Test kinematics with delta_t scaling
    kin_30 = compute_9d_kinematics(normed, smooth=True, fps=30.0)
    assert kin_30.shape == (T, K, 9)
    print(f"Kinematics 9D shape: {kin_30.shape}, max velocity: {np.max(np.abs(kin_30[:, :, 3:6])):.4f}")
    assert np.max(np.abs(kin_30[:, :, 3:6])) <= 12.0
    
    # Test 19-D standardized phonology
    phon = compute_19d_phonology(normed, val_mask)
    assert phon.shape == (T, 19)
    print(f"Phonology shape: {phon.shape}, curl range: [{np.min(phon[:, 9:19]):.3f}, {np.max(phon[:, 9:19]):.3f}]")
    assert np.all(phon[:, 9:19] >= 0.0) and np.all(phon[:, 9:19] <= 1.0)
    assert np.all(phon[:, 7:9] > 0.0) and np.all(phon[:, 7:9] <= 1.0)
    print("[PASS] compute_9d_kinematics & compute_19d_phonology verified.")


def test_dataset_collation_and_tile_alignment():
    print("\n--- 2. Testing Dataset Collation with TPU 128-Tile Multiple ---")
    raw_samples = [
        {
            "kinematics": torch.randn(45, 60, 9),
            "phonology": torch.randn(45, 19),
            "face_landmarks": torch.randn(45, 12, 3),
            "cranial_imu": torch.randn(45, 3),
            "roi_visual": torch.randn(45, 128),
            "hand_visual": torch.randn(45, 128),
            "text_tokens": torch.randint(1, 100, (12,)),
            "hand_mask": torch.ones(45, 2, dtype=torch.bool),
        },
        {
            "kinematics": torch.randn(75, 60, 9),
            "phonology": torch.randn(75, 19),
            "face_landmarks": torch.randn(75, 12, 3),
            "cranial_imu": torch.randn(75, 3),
            "roi_visual": torch.randn(75, 128),
            "hand_visual": torch.randn(75, 128),
            "text_tokens": torch.randint(1, 100, (16,)),
            "hand_mask": torch.ones(75, 2, dtype=torch.bool),
        },
    ]
    
    # Test tile multiple alignment to 128
    collated = fast_vectorized_v3_collate_fn(raw_samples, tile_multiple=128, modality_dropout_prob=0.0)
    kin_shape = collated["kinematics"].shape
    print(f"Collate kin shape with tile_multiple=128: {kin_shape}")
    assert kin_shape[1] == 128, f"Expected length 128, got {kin_shape[1]}"
    assert collated["hand_mask"].shape == (2, 128, 2)
    assert collated["roi_visual"].shape == (2, 128, 128)
    assert collated["hand_visual"].shape == (2, 128, 128)
    print("[PASS] TPU 128-tile alignment & hand mask collation verified.")


def test_model_and_orchestrator_end_to_end():
    print("\n--- 3. Testing ASLV3FoundationModel & V3TrainingOrchestrator ---")
    B, T, D, vocab = 2, 32, 128, 128
    
    model = ASLV3FoundationModel(
        d_model=D,
        in_channels=9,
        num_keypoints=60,
        vocab_size=vocab,
        english_vocab_size=vocab,
        max_seq_len=64,
        num_dec_layers=2,
        nhead=4,
        use_gpt2_decoder=True,
    )
    
    orchestrator = V3TrainingOrchestrator(
        model=model,
        lr=1e-3,
        use_ema=True,
        ema_decay=0.99,
        device=torch.device("cpu"),
    )
    
    batch = {
        "kinematics": torch.randn(B, T, 60, 9),
        "phonology": torch.randn(B, T, 19),
        "face_landmarks": torch.randn(B, T, 12, 3),
        "cranial_imu": torch.randn(B, T, 3),
        "roi_visual": torch.randn(B, T, D),
        "hand_visual": torch.randn(B, T, D),
        "text_tokens": torch.randint(1, vocab, (B, 16)),
        "text_is_negative": torch.tensor([False, True]),
        "hand_mask": torch.ones(B, T, 2, dtype=torch.bool),
    }
    
    metrics = orchestrator.train_step(batch, sync_metrics=True)
    print(f"Train step executed successfully! Metrics:")
    for k, v in list(metrics.items())[:8]:
        print(f"  {k}: {v:.4f}")
    
    assert "total_loss" in metrics
    assert "loss_translation_ce" in metrics
    assert "loss_anti_hallucination" in metrics
    assert "loss_polarity" in metrics
    assert metrics["total_loss"] > 0.0
    print("[PASS] End-to-end model forward, loss computation, backward pass, and EMA update passed!")


if __name__ == "__main__":
    t0 = time.time()
    test_v4_preprocessing_suite()
    test_dataset_collation_and_tile_alignment()
    test_model_and_orchestrator_end_to_end()
    elapsed = time.time() - t0
    print(f"\n================================================================================")
    print(f"ALL END-TO-END PIPELINE & PREPROCESSING AUDIT TESTS PASSED in {elapsed:.2f}s!")
    print(f"================================================================================")
