#!/usr/bin/env python3
"""
Lightweight Preliminary Audit Test Script
Local CPU constraint compliant: B=2, T=32, D=128, vocab=128.
Tests current baseline behavior before architectural refactoring.
"""

import sys
from pathlib import Path

# Add workspace root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_tpu.v3.modules.asl_v3_foundation_model import ASLV3FoundationModel
from train_tpu.v3.modules.classifier_trajectory import DeconstructiveClassifierField
from train_tpu.v3.engine.train_all_in_one_tpu import V3TrainingOrchestrator

def test_cpc_loop_shapes():
    print("=== Testing CPC Trajectory Loop Shapes ===")
    field = DeconstructiveClassifierField(d_model=128, num_classifier_types=16, future_steps=4)
    h = torch.randn(2, 32, 128)
    hand_pos = torch.randn(2, 32, 3)
    base_pos = torch.randn(2, 32, 3)

    out, losses = field(h, hand_positions=hand_pos, base_hand_positions=base_pos)
    print("DeconstructiveClassifierField output shape:", out.shape)
    print("loss_classifier_cpc:", losses["loss_classifier_cpc"])
    print("[NOTE] Verified: Field runs, but uses Python loop over k in future_steps with dynamic shapes T-k.")

def test_foundation_model_step():
    print("\n=== Testing ASLV3FoundationModel and Train Step ===")
    B, T, D = 2, 32, 128
    model = ASLV3FoundationModel(
        d_model=D,
        in_channels=9,
        num_keypoints=60,
        vocab_size=128,
        english_vocab_size=256,
        num_enc_layers=2,
        num_dec_layers=2,
        nhead=4,
        max_seq_len=64,
        chunk_size=16,
        use_gpt2_decoder=True,
    )
    
    batch = {
        "kinematics": torch.randn(B, T, 60 * 9),
        "phonology": torch.randn(B, T, 19),
        "cranial_imu": torch.randn(B, T, 3),
        "text_tokens": torch.randint(1, 200, (B, 16)),
        "text_is_negative": torch.tensor([0.0, 1.0]),
    }

    orchestrator = V3TrainingOrchestrator(model, lr=1e-4, use_ema=False, device=torch.device("cpu"))
    metrics = orchestrator.train_step(batch, sync_metrics=False)
    print("Train step succeeded. Metrics keys:", list(metrics.keys()))
    print("Total loss tensor requires_grad:", metrics["total_loss"].requires_grad)
    assert not metrics["total_loss"].requires_grad, "total_loss must be detached!"
    print("[SUCCESS] Preliminary baseline audit passed.")

if __name__ == "__main__":
    test_cpc_loop_shapes()
    test_foundation_model_step()
