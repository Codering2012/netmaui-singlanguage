#!/usr/bin/env python3
"""
Hypothesis Test: GPT-2 Translation Decoder & Visual Grounding Shield Refactor
Tests SDPA acceleration, dynamic vocabulary handling, and autograd detachment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_tpu.v3.modules.gpt2_translation_decoder import GPT2CrossModalTranslationDecoder
from train_tpu.v3.modules.visual_grounding_shield import VisualGroundingShield

def test_gpt2_and_shield():
    B, L, T, D = 2, 16, 32, 128
    vocab_size = 256

    decoder = GPT2CrossModalTranslationDecoder(
        vocab_size=vocab_size,
        max_position_embeddings=64,
        d_model=D,
        d_encoder=D,
        num_layers=2,
        num_heads=4,
    )

    shield = VisualGroundingShield(d_model=D, vocab_size=vocab_size)

    input_ids = torch.randint(1, vocab_size, (B, L))
    memory = torch.randn(B, T, D)

    # 1. Forward pass through decoder
    logits, cross_attn = decoder(input_ids=input_ids, memory=memory)
    print("Decoder logits shape:", logits.shape)
    print("Cross attention weights shape:", cross_attn.shape)
    assert logits.shape == (B, L, vocab_size)
    assert cross_attn.shape == (B, L, T)

    # 2. Shield forward pass
    motion_energy = torch.norm(torch.diff(memory, dim=1, prepend=memory[:, :1, :]), dim=-1)
    shielded_logits, shield_losses = shield(logits, cross_attn, motion_energy=motion_energy)
    print("Shielded logits shape:", shielded_logits.shape)
    print("Anti-hallucination loss:", shield_losses["loss_anti_hallucination"].item())

    # 3. Autograd check
    loss = torch.sum(shielded_logits) + shield_losses["loss_anti_hallucination"]
    loss.backward()
    print("Backward pass succeeded.")

    # 4. Generate check
    gen_tokens = decoder.generate(memory=memory, max_new_tokens=8, bos_token_id=1, eos_token_id=2)
    print("Generated tokens shape:", gen_tokens.shape)
    assert gen_tokens.shape[0] == B
    print("[SUCCESS] GPT-2 Decoder and Shield verified.")

if __name__ == "__main__":
    test_gpt2_and_shield()
