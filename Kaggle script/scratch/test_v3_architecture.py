#!/usr/bin/env python3
"""
================================================================================
ASL V3 FOUNDATION ARCHITECTURE EMPIRICAL VERIFICATION & SELF-TEST
================================================================================
Verifies end-to-end forward pass, multi-task losses, and gradient flow across:
1. Dynamic 3D Locus Neural Memory Bank
2. Non-Manual Feature Pyramid (NMM-FPN) & Polarity Guard
3. Deconstructive Classifier Trajectory Field (CPC)
4. Monotonic Chunk Permutation Transducer (OSV -> SVO reordering)
5. Cross-Attention Visual Grounding Shield
6. Dual Visual Stems (256x256 ROI + 128x128 Hand Crops)

Local constraints: B <= 4, T <= 32, D = 128, Memory < 500 MB, Runtime < 15s.
================================================================================
"""

import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import torch
import torch.nn as nn

from train_tpu.v3.modules.asl_v3_foundation_model import ASLV3FoundationModel, V3ModelOutput
from train_tpu.v3.engine.train_all_in_one_tpu import V3TrainingOrchestrator


def run_verification():
    print("=" * 80)
    print("RUNNING ASL V3 FOUNDATION ARCHITECTURE INTEGRATION & GRADIENT VERIFICATION")
    print("=" * 80)

    start_time = time.time()
    torch.manual_seed(42)

    # 1. Setup Lightweight Mock Dimensions
    B, T = 2, 32
    D = 128
    vocab_size = 128
    english_vocab = 256
    text_len = 16

    print(f"[TEST 1/4] Instantiating ASLV3FoundationModel (B={B}, T={T}, D={D})...")
    model = ASLV3FoundationModel(
        d_model=D,
        in_channels=9,
        num_keypoints=60,
        vocab_size=vocab_size,
        english_vocab_size=english_vocab,
        num_enc_layers=2,
        num_dec_layers=2,
        nhead=4,
        chunk_size=16,
    )
    print("  -> Model successfully initialized. Parameter count:", sum(p.numel() for p in model.parameters()))

    # 2. Synthesize Full Multimodal Batch
    print("[TEST 2/4] Synthesizing full multimodal tensor batch...")
    batch = {
        "kinematics": torch.randn(B, T, 60 * 9),
        "roi_visual": torch.randn(B, T, 3, 256, 256),
        "hand_visual": torch.randn(B, T, 3, 128, 128),
        "phonology": torch.randn(B, T, 19),
        "face_landmarks": torch.randn(B, T, 12, 3),
        "cranial_imu": torch.randn(B, T, 3),
        "text_tokens": torch.randint(1, english_vocab, (B, text_len)),
        "text_is_negative": torch.tensor([True, False]),
    }
    print("  -> Batch synthesized with 8 multimodal channels.")

    # 3. Test Forward Pass
    print("[TEST 3/4] Running full forward pass with V3 multi-task losses...")
    output: V3ModelOutput = model(
        kinematics=batch["kinematics"],
        roi_visual=batch["roi_visual"],
        hand_visual=batch["hand_visual"],
        phonology=batch["phonology"],
        face_landmarks=batch["face_landmarks"],
        cranial_imu=batch["cranial_imu"],
        text_tokens=batch["text_tokens"],
        text_is_negative=batch["text_is_negative"],
    )

    print("  -> Output CTC Logits shape:", output.ctc_logits.shape)
    print("  -> Output English CTC Logits shape:", output.english_ctc_logits.shape)
    print("  -> Output Decoder Logits shape:", output.decoder_logits.shape)
    print("  -> Active Multi-Task Losses returned:")
    for k, v in output.multi_task_losses.items():
        print(f"     * {k}: {v.item():.4f}")

    assert "loss_locus" in output.multi_task_losses, "Missing loss_locus!"
    assert "loss_nmm" in output.multi_task_losses, "Missing loss_nmm!"
    assert "loss_polarity" in output.multi_task_losses, "Missing loss_polarity!"
    assert "loss_classifier" in output.multi_task_losses, "Missing loss_classifier!"
    assert "loss_permutation" in output.multi_task_losses, "Missing loss_permutation!"
    assert "loss_phonology" in output.multi_task_losses, "Missing loss_phonology!"
    assert "loss_anti_hallucination" in output.multi_task_losses, "Missing loss_anti_hallucination!"
    print("  [PASS] All 7 architectural loss components verified active!")

    # 4. Test Backward Pass and Gradient Flow across all modules
    print("[TEST 4/4] Verifying backward pass & gradient flow across specialized engines...")
    orchestrator = V3TrainingOrchestrator(model, lr=1e-4)
    step_metrics = orchestrator.train_step(batch)

    print("  -> Train Step Metrics:")
    for k, v in list(step_metrics.items())[:6]:
        val = v.item() if isinstance(v, torch.Tensor) else float(v)
        print(f"     * {k}: {val:.4f}")

    # Inspect gradient norms
    locus_grad = torch.norm(model.locus_memory.write_gate[0].weight.grad).item()
    nmm_grad = torch.norm(model.nmm_pyramid.eyebrow_encoder[0].weight.grad).item()
    classifier_grad = torch.norm(model.classifier_field.trajectory_regressor[0].weight.grad).item()
    chunk_grad = torch.norm(model.chunk_transducer.swap_scorer[0].weight.grad).item()
    shield_grad = torch.norm(model.grounding_shield.is_content_token.grad).item()
    hand_stem_grad = torch.norm(model.hand_stem.conv1.weight.grad).item()
    vis_stem_grad = torch.norm(model.visual_stem.conv1.weight.grad).item()

    print(f"  -> Locus Memory Grad Norm:        {locus_grad:.6f}")
    print(f"  -> NMM Eyebrow Grad Norm:         {nmm_grad:.6f}")
    print(f"  -> Classifier Field Grad Norm:    {classifier_grad:.6f}")
    print(f"  -> Chunk Transducer Grad Norm:    {chunk_grad:.6f}")
    print(f"  -> Visual Grounding Shield Grad:  {shield_grad:.6f}")
    print(f"  -> Hand Crop Stem Grad Norm:      {hand_stem_grad:.6f}")
    print(f"  -> Visual 256 Stem Grad Norm:     {vis_stem_grad:.6f}")

    assert locus_grad > 0, "Zero gradient in Locus Memory!"
    assert nmm_grad > 0, "Zero gradient in NMM Pyramid!"
    assert classifier_grad > 0, "Zero gradient in Classifier Field!"
    assert chunk_grad > 0, "Zero gradient in Chunk Transducer!"
    assert shield_grad > 0, "Zero gradient in Grounding Shield!"
    assert hand_stem_grad > 0, "Zero gradient in Hand Stem!"
    assert vis_stem_grad > 0, "Zero gradient in Visual Stem!"

    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] ASL V3 Full Verification Passed in {elapsed:.2f} seconds!")
    print("=" * 80)


if __name__ == "__main__":
    run_verification()
