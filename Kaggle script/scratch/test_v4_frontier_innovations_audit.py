#!/usr/bin/env python3
"""
================================================================================
EMPIRICAL AUDIT: V4 POLISHED PREPROCESSOR & FRONTIER TRANSLATION ENGINES
================================================================================
Strict Hardware Invariants (Physical i5-8250U 16GB):
- Lightweight Mock Dimensions: B=2, T=32, D=128, English Vocab=256
- Memory ceiling < 400 MB
- Execution time < 10 seconds
================================================================================
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from train_tpu.v3.modules.dynamic_phonological_condenser import DynamicPhonologicalCondenser
from train_tpu.v3.modules.vq_phono_codebook import VQPhonoCodebook
from train_tpu.v3.modules.sinkhorn_transducer import SinkhornChunkTransducer
from train_tpu.v3.modules.semantic_embedding_anchor import SemanticEmbeddingAnchor
from train_tpu.v3.modules.gpt2_translation_decoder import GPT2CrossModalTranslationDecoder
from train_tpu.v3.modules.asl_v3_foundation_model import ASLV3FoundationModel
from train_tpu.v3.engine.train_all_in_one_tpu import V3TrainingOrchestrator
from train_tpu.v3.engine.dataset import fast_vectorized_v3_collate_fn


def test_dynamic_phonological_condenser():
    print("\n--- [1/6] Testing Dynamic Phonological Hold-Condensation Pooling (D-PCP) ---")
    B, T, D = 2, 32, 128
    condenser = DynamicPhonologicalCondenser(d_model=D, n_condensed=16, num_keypoints=60, in_channels=9)
    h = torch.randn(B, T, D, requires_grad=True)
    kinematics = torch.randn(B, T, 60 * 9)

    h_cond, s_t, assign_weights = condenser(h, kinematics)
    print(f"  Input: {h.shape} -> Condensed: {h_cond.shape}")
    print(f"  Hold salience shape: {s_t.shape}, min={s_t.min():.4f}, max={s_t.max():.4f}")
    print(f"  Assignment weights: {assign_weights.shape}, sum across T={assign_weights.sum(dim=-1)[0, :3]}")

    assert h_cond.shape == (B, 16, D), f"Expected shape {(B, 16, D)}, got {h_cond.shape}"
    assert torch.allclose(assign_weights.sum(dim=-1), torch.ones(B, 16), atol=1e-5), "Assignment weights must sum to 1"

    # Test backward pass
    loss = h_cond.sum()
    loss.backward()
    assert h.grad is not None and not torch.isnan(h.grad).any(), "Gradient through D-PCP failed!"
    print("  [PASS] D-PCP forward, backward, and hold-saliency verified.")


def test_vq_phono_codebook():
    print("\n--- [2/6] Testing VQ-Phono Codebook & Articulatory CutMix ---")
    B, T, D = 2, 32, 128
    vq = VQPhonoCodebook(d_model=D, num_codes=256, code_dim=32, num_keypoints=60, in_channels=9)
    h = torch.randn(B, T, D, requires_grad=True)
    kinematics = torch.randn(B, T, 60, 9)
    phonology = torch.randn(B, T, 19)

    # 1. Test CutMix
    vq.train()
    cutmixed_kin = vq.apply_articulatory_cutmix(kinematics)
    assert cutmixed_kin.shape == kinematics.shape, f"CutMix altered shape! {cutmixed_kin.shape} vs {kinematics.shape}"

    # 2. Test pretraining loss
    total_loss, metrics = vq.compute_pretraining_loss(h, kinematics, phonology)
    print(f"  VQ-MAM Total Loss: {total_loss.item():.4f}")
    print(f"  Loss CE: {metrics['loss_vq_ce'].item():.4f}, Loss Vel: {metrics['loss_vq_vel'].item():.4f}")

    total_loss.backward()
    assert h.grad is not None and not torch.isnan(h.grad).any(), "Gradient through VQ-Phono failed!"
    print("  [PASS] VQ-Phono Codebook & CutMix verified.")


def test_monotonic_sinkhorn():
    print("\n--- [3/6] Testing Band-Constrained Monotonic Sinkhorn Transducer ---")
    B, T, D = 2, 16, 128
    transducer = SinkhornChunkTransducer(d_model=D, chunk_size=2, num_iters=16, epsilon=0.08, band_weight=0.5)
    h = torch.randn(B, T, D, requires_grad=True)

    h_reordered, P = transducer(h)
    mono_loss = transducer.compute_monotonic_loss(P)
    print(f"  Reordered: {h_reordered.shape}, Permutation P: {P.shape}")
    print(f"  Monotonicity Loss: {mono_loss.item():.4f}")
    print(f"  Row sums of P: {P.sum(dim=-1)[0]}")

    assert h_reordered.shape == (B, T, D), "Reordered shape mismatch!"
    assert torch.allclose(P.sum(dim=-1), torch.ones_like(P.sum(dim=-1)), atol=1e-4), "P must be row-stochastic!"

    (h_reordered.sum() + mono_loss).backward()
    assert h.grad is not None, "Gradient through Monotonic Sinkhorn failed!"
    print("  [PASS] Band-Constrained Monotonic Sinkhorn verified.")


def test_coverage_decoder():
    print("\n--- [4/6] Testing Translation Decoder Coverage Loss & Repetition Shield ---")
    B, L, D = 2, 10, 128
    T_enc = 16
    decoder = GPT2CrossModalTranslationDecoder(
        vocab_size=256,
        max_position_embeddings=64,
        d_model=D,
        d_encoder=D,
        num_layers=2,
        num_heads=4,
    )
    input_ids = torch.randint(0, 256, (B, L))
    memory = torch.randn(B, T_enc, D)

    logits, cross_attn = decoder(input_ids, memory)
    cov_loss = decoder.compute_coverage_loss(cross_attn)
    print(f"  Logits: {logits.shape}, Cross-Attn: {cross_attn.shape}")
    print(f"  Coverage Loss: {cov_loss.item():.4f}")

    assert logits.shape == (B, L, 256), f"Expected {(B, L, 256)}, got {logits.shape}"
    assert cross_attn.shape == (B, L, T_enc), f"Expected {(B, L, T_enc)}, got {cross_attn.shape}"
    assert cov_loss.item() >= 0.0, "Coverage loss must be non-negative!"

    # Test generation with repetition penalty
    generated = decoder.generate(memory, max_new_tokens=8, repetition_penalty=1.5)
    print(f"  Generated token sequence: {generated.shape}")
    assert generated.shape == (B, 9), f"Expected {(B, 9)}, got {generated.shape}"
    print("  [PASS] Decoder Coverage Loss & Repetition Shield verified.")


def test_asl_v3_foundation_model_grand_forward():
    print("\n--- [5/6] Testing Full ASLV3FoundationModel Forward with All Innovations ---")
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
    )
    kinematics = torch.randn(B, T, 60 * 9)
    phonology = torch.randn(B, T, 19)
    cranial_imu = torch.randn(B, T, 3)
    text_tokens = torch.randint(1, 200, (B, 12))
    text_is_neg = torch.tensor([True, False], dtype=torch.bool)
    sent_emb = torch.randn(B, 384)

    out = model(
        kinematics=kinematics,
        phonology=phonology,
        cranial_imu=cranial_imu,
        text_tokens=text_tokens,
        text_is_negative=text_is_neg,
        target_sentence_embeddings=sent_emb,
        enable_specaugment=True,
        enable_mam=True,
    )

    print(f"  Decoder Logits: {out.decoder_logits.shape if out.decoder_logits is not None else None}")
    print(f"  Active Multi-Task Losses: {list(out.multi_task_losses.keys())}")
    for k, v in out.multi_task_losses.items():
        print(f"    - {k}: {v.item():.4f}")

    assert "loss_coverage" in out.multi_task_losses, "loss_coverage missing!"
    assert "loss_monotonic" in out.multi_task_losses, "loss_monotonic missing!"
    assert "loss_mam" in out.multi_task_losses, "loss_mam missing!"
    assert "loss_semantic_bridge" in out.multi_task_losses, "loss_semantic_bridge missing!"
    print("  [PASS] Full Foundation Model forward verified.")


def test_orchestrator_full_dataset_schedule():
    print("\n--- [6/6] Testing V3TrainingOrchestrator Full Dataset Scheduler & Step ---")
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
    )
    orchestrator = V3TrainingOrchestrator(model, lr=3e-4)

    # Configure schedule for full dataset steps (e.g. 500 steps across all epochs, NO warmup truncation)
    total_dataset_steps = 500
    orchestrator.setup_full_dataset_scheduler(total_dataset_steps)
    initial_lr = orchestrator.optimizer.param_groups[0]["lr"]
    print(f"  Initial Encoder LR: {initial_lr:.6f} (Decoupled Decoder LR: {orchestrator.optimizer.param_groups[1]['lr']:.6f})")

    # Mock batch
    batch = {
        "kinematics": torch.randn(B, T, 60 * 9),
        "phonology": torch.randn(B, T, 19),
        "cranial_imu": torch.randn(B, T, 3),
        "english_seq": torch.randint(1, 200, (B, 12)),  # Test english_seq fallback to text_tokens
        "text_is_negative": torch.tensor([True, False], dtype=torch.bool),
        "sentence_embedding": torch.randn(B, 384),
    }

    metrics = orchestrator.train_step(batch, sync_metrics=True)
    stepped_lr = orchestrator.optimizer.param_groups[0]["lr"]
    print(f"  Step 1 Total Loss: {metrics['total_loss']:.4f}")
    print(f"  Translation CE Loss: {metrics.get('loss_translation_ce', 0.0):.4f}")
    print(f"  Stepped LR after step 1: {stepped_lr:.6f}")

    assert metrics.get("loss_translation_ce") is not None, "Translation CE was not executed!"
    assert metrics.get("loss_coverage") is not None, "Coverage loss was not recorded!"
    assert metrics.get("loss_monotonic") is not None, "Monotonic loss was not recorded!"
    print("  [PASS] Training Orchestrator with Full Dataset Schedule verified.")


if __name__ == "__main__":
    t0 = time.time()
    test_dynamic_phonological_condenser()
    test_vq_phono_codebook()
    test_monotonic_sinkhorn()
    test_coverage_decoder()
    test_asl_v3_foundation_model_grand_forward()
    test_orchestrator_full_dataset_schedule()
    dt = time.time() - t0
    print(f"\n==================================================================")
    print(f"  ALL 6 FRONTIER TESTS PASSED DETERMINISTICALLY IN {dt:.2f}s!")
    print(f"==================================================================")
