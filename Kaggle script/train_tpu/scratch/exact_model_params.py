import sys
import os
from pathlib import Path

sys.path.insert(0, r"c:\Users\Windows 10 21H1\source\repos\Kaggle script\train_tpu")
import torch
import train_all_in_one_tpu as tm

print("=" * 70)
print("   EXACT EMPIRICAL PARAMETER COUNT CALCULATION FOR FOUNDATION MODEL   ")
print("=" * 70)

# Check vocabulary sizes on E:\datasets\asl_dataset\asl_preprocessed_phase1
vocab_size = 3586  # output_mapping / gloss vocab
english_vocab_size = 23473  # from english_vocab.json

eng_vocab_file = Path("E:/datasets/asl_dataset/asl_preprocessed_phase1/english_vocab.json")
if eng_vocab_file.exists():
    import json
    with open(eng_vocab_file, "r", encoding="utf-8") as f:
        ev = json.load(f)
        english_vocab_size = len(ev)
        print(f"Loaded exact English vocabulary size: {english_vocab_size:,} classes")

# Instantiate ASLFoundationModel with user's exact CLI configuration:
# --d-model 384 --nhead 6 --num-layers 10 --enable-aux-decoders --is-causal --disable-bpe --max-len 384
model = tm.ASLFoundationModel(
    vocab_size=vocab_size,
    english_vocab_size=english_vocab_size,
    d_enc=384,
    d_dec=384,
    nhead_enc=6,
    nhead_dec=6,
    num_enc_layers=10,
    num_dec_layers=10,
    dropout=0.1,
    max_enc_len=384,
    scale_embeddings=True,
    enable_aux_decoders=True,
    is_causal=True,
)

total_full_model_params = sum(p.numel() for p in model.parameters())
trainable_full_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\n1. TOTAL FULL ASL FOUNDATION MODEL (All Encoders + All 3 Decoders):")
print(f"   Total Parameters:     {total_full_model_params:,} ({total_full_model_params / 1e6:.2f} Million params)")
print(f"   Trainable Parameters: {trainable_full_params:,} ({trainable_full_params / 1e6:.2f} Million params)")

# Component breakdown of full model:
print("\n--- Submodule Breakdown (Full Model) ---")
comp_params = {}
for name, param in model.named_parameters():
    top_module = name.split(".")[0]
    comp_params[top_module] = comp_params.get(top_module, 0) + param.numel()

for comp, p_cnt in sorted(comp_params.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {comp:<28}: {p_cnt:>12,} ({p_cnt / 1e6:>6.2f}M)  [{p_cnt / total_full_model_params * 100:>5.1f}%]")

# 2. Phase 1 Text Pretraining Wrapper
phase1_wrapper = tm.Phase1TextWrapper(model)
p1_total = sum(p.numel() for p in phase1_wrapper.parameters())
p1_trainable = sum(p.numel() for p in phase1_wrapper.parameters() if p.requires_grad)

print(f"\n2. PHASE 1 ACTIVE TEXT PRETRAINING WRAPPER:")
print(f"   Total Phase 1 Params:     {p1_total:,} ({p1_total / 1e6:.2f} Million params)")
print(f"   Trainable Phase 1 Params: {p1_trainable:,} ({p1_trainable / 1e6:.2f} Million params)")

print("\n--- Submodule Breakdown (Phase 1 Wrapper) ---")
p1_comp_params = {}
for name, param in phase1_wrapper.named_parameters():
    top_module = name.split(".")[0]
    p1_comp_params[top_module] = p1_comp_params.get(top_module, 0) + param.numel()

for comp, p_cnt in sorted(p1_comp_params.items(), key=lambda x: x[1], reverse=True):
    print(f"  - {comp:<28}: {p_cnt:>12,} ({p_cnt / 1e6:>6.2f}M)  [{p_cnt / p1_total * 100:>5.1f}%]")

# What if d_model=512?
model_512 = tm.ASLFoundationModel(
    vocab_size=vocab_size,
    english_vocab_size=english_vocab_size,
    d_enc=512,
    d_dec=512,
    nhead_enc=8,
    nhead_dec=8,
    num_enc_layers=10,
    num_dec_layers=10,
    dropout=0.1,
    max_enc_len=384,
    scale_embeddings=True,
    enable_aux_decoders=True,
    is_causal=True,
)
total_512 = sum(p.numel() for p in model_512.parameters())
print(f"\n3. COMPARISON IF d_model=512 (10 layers):")
print(f"   Full Foundation Model (d=512): {total_512:,} ({total_512 / 1e6:.2f} Million params)")
