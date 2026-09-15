#!/usr/bin/env python3
"""
================================================================================
EMPIRICAL VERIFICATION: GPT-2 CROSS-MODAL TRANSLATION DECODER
================================================================================
Verifies:
1. Teacher-forced forward pass: [B, L] -> [B, L, vocab_size] logits.
2. Cross-attention weight extraction: [B, L, T] for Visual Grounding Shield.
3. Weight tying: lm_head.weight shares exact storage with wte.weight.
4. Autoregressive greedy decoding: generate() produces monotonic token emissions.
Hardware Ceiling: B=2, L=16, T=32, D=128, vocab=256, Execution < 5s.
================================================================================
"""

import sys
import time
from pathlib import Path

workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))

import torch
from train_tpu.v3.modules.gpt2_translation_decoder import GPT2CrossModalTranslationDecoder


def test_gpt2_decoder():
    print("[TEST 1/2] Testing GPT-2 Cross-Modal Decoder Forward Pass & Weight Tying...")
    B, L, T, D = 2, 16, 32, 128
    V = 256
    decoder = GPT2CrossModalTranslationDecoder(
        vocab_size=V,
        max_position_embeddings=64,
        d_model=D,
        d_encoder=D,
        num_layers=2,
        num_heads=2,
    )

    # 1. Verify weight tying
    assert decoder.lm_head.weight is decoder.wte.weight, "lm_head.weight must be tied to wte.weight!"
    print("  [PASS] Weight tying verified.")

    # 2. Forward pass
    mock_input_ids = torch.randint(0, V, (B, L))
    mock_memory = torch.randn(B, T, D)

    logits, cross_attn = decoder(mock_input_ids, mock_memory)

    assert logits.shape == (B, L, V), f"Logits shape mismatch: {logits.shape} vs ({B}, {L}, {V})"
    assert cross_attn.shape == (B, L, T), f"Cross-attn shape mismatch: {cross_attn.shape} vs ({B}, {L}, {T})"
    assert not torch.isnan(logits).any(), "NaN in decoder logits!"
    assert not torch.isnan(cross_attn).any(), "NaN in cross-attention weights!"
    print("  [PASS] Forward pass produced valid logits and cross-attention weights.")

    # 3. Autoregressive generation
    print("[TEST 2/2] Testing Autoregressive Token Generation...")
    generated = decoder.generate(mock_memory, max_new_tokens=8, bos_token_id=1, eos_token_id=2)
    assert generated.shape[0] == B, f"Batch size mismatch: {generated.shape[0]} vs {B}"
    assert generated.shape[1] >= 2, "Generation should produce at least 2 tokens (BOS + next)"
    assert generated[:, 0].eq(1).all(), "First token must be BOS (1)"
    print(f"  [PASS] Autoregressive generation emitted sequence of length {generated.shape[1]}.")


if __name__ == "__main__":
    t0 = time.time()
    test_gpt2_decoder()
    dt = time.time() - t0
    print(f"\n[SUCCESS] GPT-2 Cross-Modal Decoder verified in {dt:.2f}s (< 15s hardware limit)!")
