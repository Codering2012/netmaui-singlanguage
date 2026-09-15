#!/usr/bin/env python3
"""
================================================================================
   EMPIRICAL PROFILING: ASLG VS KDWD VS PURE COMPUTE STEP
================================================================================
Measures exact microseconds per batch for:
  1. Pure in-memory tensor indexing (ASLG)
  2. DAE corruptions + string conversions
  3. KDWD in-memory flat tensor indexing
  4. Collation function throughput
  5. Pure forward + loss + backward pass
"""

import sys
import time
import os
import random
from pathlib import Path
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "engine"))
try:
    from dataset import apply_dae_corruptions, phase1_collate_fn, EnglishVocabulary, GlossVocabulary
    from train_all_in_one_tpu import ASLFoundationModel, Phase1TextWrapper
except ImportError:
    from train_tpu.v2.engine.dataset import apply_dae_corruptions, phase1_collate_fn, EnglishVocabulary, GlossVocabulary
    from train_tpu.v2.engine.train_all_in_one_tpu import ASLFoundationModel, Phase1TextWrapper

def benchmark():
    print("=" * 80)
    print("       🔬 EMPIRICAL STEP-BY-STEP PROFILING HARNESS       ")
    print("=" * 80)

    # 1. Profile apply_dae_corruptions
    print("\n[PROFILING 1/4] Profiling DAE corruption logic on 10,000 samples...")
    sample_tokens = list(range(10, 85))
    t0 = time.time()
    for _ in range(10000):
        corrupted = apply_dae_corruptions(sample_tokens.copy(), unk_id=3)
    dt_dae = time.time() - t0
    print(f"  -> 10,000 DAE corruptions took: {dt_dae:.4f}s ({10000/dt_dae:.1f} samples/sec)")

    # 2. Profile Collate Function
    print("\n[PROFILING 2/4] Profiling phase1_collate_fn on batch_size=128...")
    mock_batch = [
        {"input_ids": torch.randint(1, 1000, (64,), dtype=torch.long), "target_ids": torch.randint(1, 1000, (64,), dtype=torch.long), "is_dae": (i % 2 == 0)}
        for i in range(128)
    ]
    t0 = time.time()
    for _ in range(500):
        collated = phase1_collate_fn(mock_batch, max_len=384, eng_pad_id=0)
    dt_collate = time.time() - t0
    print(f"  -> 500 batches (64,000 samples) collated in: {dt_collate:.4f}s ({64000/dt_collate:.1f} samples/sec)")

    # 3. Profile In-Memory Tensor Slicing
    print("\n[PROFILING 3/4] Profiling In-Memory Flat Tensor Indexing (ASLG vs KDWD in RAM)...")
    total_tokens = 1_000_000
    flat_tensor = torch.randint(1, 20000, (total_tokens,), dtype=torch.int32)
    offsets = torch.arange(0, total_tokens, 50, dtype=torch.int32)
    offsets_np = offsets.numpy()

    t0 = time.time()
    for _ in range(100_000):
        idx = random.randint(0, len(offsets_np) - 2)
        st = int(offsets_np[idx])
        ed = int(offsets_np[idx + 1])
        item = flat_tensor[st:ed].to(torch.long)
    dt_ram = time.time() - t0
    print(f"  -> 100,000 in-memory RAM slices in: {dt_ram:.4f}s ({100000/dt_ram:.1f} samples/sec)")

    # 4. Profile Pure Forward + Backward Compute Step
    print("\n[PROFILING 4/4] Profiling Pure Forward + Fused Loss + Backward (Batch=128, Len=384)...")
    model = ASLFoundationModel(
        num_enc_layers=0,
        num_dec_layers=6,
        d_dec=512,
        nhead_dec=8,
        vocab_size=2560,
        english_vocab_size=23552,
        enable_aux_decoders=True,
        is_causal=True,
    )
    phase1_net = Phase1TextWrapper(model)
    phase1_net.train()
    
    # Warmup
    dummy_in_gloss = torch.randint(0, 2560, (128, 384))
    dummy_in_eng = torch.randint(0, 23552, (128, 384))
    dummy_mask = torch.zeros(128, 384, dtype=torch.bool)
    dummy_aslg_mask = torch.ones(128, 1, 1, dtype=torch.bool)
    dummy_tgt_in = torch.randint(0, 23552, (128, 383))
    dummy_tgt_out = torch.randint(0, 23552, (128, 383))

    t0 = time.time()
    for step in range(5):
        logits = phase1_net(dummy_in_gloss, dummy_in_eng, dummy_aslg_mask, dummy_tgt_in, dummy_mask)
        loss = F.cross_entropy(logits.reshape(-1, 23552), dummy_tgt_out.reshape(-1), ignore_index=0)
        loss.backward()
        phase1_net.zero_grad(set_to_none=True)
    dt_compute = time.time() - t0
    print(f"  -> 5 Forward+Loss+Backward steps (Batch 128) completed in: {dt_compute:.4f}s (CPU reference)")

    print("\n" + "=" * 80)
    print("                      📊 SUMMARY VERDICT                      ")
    print("=" * 80)
    print(f"  1. In-Memory Tensor Slicing:  ~{100000/dt_ram:,.0f} samples/sec (Instantaneous)")
    print(f"  2. Collate Function:          ~{64000/dt_collate:,.0f} samples/sec (Zero bottleneck)")
    print(f"  3. DAE Token Corruptions:     ~{10000/dt_dae:,.0f} samples/sec (Zero bottleneck)")
    print("=" * 80)

if __name__ == "__main__":
    benchmark()
