import torch
import math

# Model parameters
d_model = 512
nhead = 8
num_layers = 10
max_len = 384
vocab_size = 23473
samples_per_sec = 1327.5
batch_size = 512

# Peak Hardware TFLOPs for TPU v5e (8 chips)
# TPU v5e has 197 TFLOPs (BF16 dense) per chip
peak_tflops_per_chip = 197.0
total_peak_tflops = peak_tflops_per_chip * 8  # 1576 TFLOPs

# 1. Parameter counting for active text decoder
# Embedding / LM Head
p_emb = vocab_size * d_model  # Tied embedding: 23473 * 512 = 12,018,176

# Attention per layer (Q, K, V, Out)
p_qkv_out = 4 * d_model * d_model  # 4 * 512 * 512 = 1,048,576

# FFN per layer (SwiGLU with d_ff = 8/3 * d = 1365 or 4d = 2048)
# In our architecture, standard SwiGLU has 3 matrices of shape (512, 1365) = 2,096,640
d_ff = int(8 * d_model / 3)
p_ffn = 3 * d_model * d_ff

# Total params per layer
p_layer = p_qkv_out + p_ffn
total_non_emb_params = num_layers * p_layer
total_params = total_non_emb_params + p_emb

# 2. FLOPs calculation per token (Chinchilla / Kaplan formulation)
# Forward pass: 2 FLOPs per param for matrix muls + 4 * d_model * max_len for attention matrix
flops_fwd_per_token = 2 * total_params + 4 * num_layers * d_model * max_len
# Backward pass: 4 FLOPs per param (2 for input grad + 2 for weight grad) + 8 * d_model * max_len
flops_bwd_per_token = 4 * total_params + 8 * num_layers * d_model * max_len
total_flops_per_token = flops_fwd_per_token + flops_bwd_per_token  # 6 * params + 12 * L * d * T

# Total FLOPs per sample (max_len = 384 tokens)
total_flops_per_sample = total_flops_per_token * max_len

# Total FLOPs per second
total_flops_per_sec = total_flops_per_sample * samples_per_sec
realized_tflops = total_flops_per_sec / 1e12

# MFU (Model FLOPs Utilization)
mfu_dense = (realized_tflops / total_peak_tflops) * 100

print("="*60)
print("       EMPIRICAL TFLOPS & MFU PERFORMANCE AUDIT        ")
print("="*60)
print(f"Active Parameters:           {total_params / 1e6:.2f} Million")
print(f"  - Non-embedding params:    {total_non_emb_params / 1e6:.2f} Million")
print(f"  - Vocabulary embeddings:   {p_emb / 1e6:.2f} Million")
print(f"Sequence Length:             {max_len} tokens")
print(f"Throughput:                  {samples_per_sec:.1f} samples/sec ({samples_per_sec * max_len / 1e3:.1f}k tokens/sec)")
print(f"FLOPs per sample:            {total_flops_per_sample / 1e9:.2f} GFLOPs")
print("-" * 60)
print(f"REALIZED COMPUTE THROUGHPUT: {realized_tflops:.2f} TFLOPs")
print(f"Peak Cluster Compute (8x):   {total_peak_tflops:.1f} TFLOPs (TPU v5e-8)")
print(f"Model FLOPs Utilization:     {mfu_dense:.2f}% MFU")
print("="*60)
