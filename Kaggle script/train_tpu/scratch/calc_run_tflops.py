import math

d_model = 384
nhead = 6
num_layers = 10
max_len = 384
vocab_size = 23473
samples_per_sec = 1500.0
batch_size = 128  # 16 per core on 8 cores

# 1. Parameter breakdown for d_model=384, 10 layers, vocab=23473
# Tied Embedding / LM Head:
p_emb = vocab_size * d_model  # 23473 * 384 = 9,013,632

# Multi-head Self-Attention per layer: Q, K, V, Out
# 4 * d_model * d_model = 4 * 384 * 384 = 589,824
p_attn_self = 4 * d_model * d_model

# Cross-Attention per layer (in encoder-decoder Phase 1 wrapper): Q, K, V, Out
p_attn_cross = 4 * d_model * d_model

# SwiGLU FFN per layer: d_ff = int(8/3 * d_model) = 1024
d_ff = int(8 * d_model / 3)  # 1024
# 3 matrices of size (384, 1024): gate, up, down = 3 * 384 * 1024 = 1,179,648
p_ffn = 3 * d_model * d_ff

p_layer = p_attn_self + p_attn_cross + p_ffn
total_non_emb = num_layers * p_layer
total_params = total_non_emb + p_emb

# 2. FLOPs per token calculation (Forward + Backward)
# Self-attention FLOPs per token:
# Projections: 2 * (4 * d^2) = 8 * d^2
# Attention matrix (Q*K^T + A*V): 4 * d * T
flops_self_attn_fwd = 8 * (d_model**2) + 4 * d_model * max_len

# Cross-attention FLOPs per token:
flops_cross_attn_fwd = 8 * (d_model**2) + 4 * d_model * max_len

# FFN FLOPs per token: 2 * p_ffn = 6 * d * d_ff
flops_ffn_fwd = 2 * p_ffn

# LM Head projection FLOPs per token: 2 * V * d
flops_lm_head_fwd = 2 * vocab_size * d_model

# Total forward FLOPs per token:
flops_fwd_per_token = num_layers * (flops_self_attn_fwd + flops_cross_attn_fwd + flops_ffn_fwd) + flops_lm_head_fwd

# Total backward FLOPs per token is 2x forward (gradient wrt inputs + gradient wrt weights):
flops_bwd_per_token = 2 * flops_fwd_per_token
total_flops_per_token = flops_fwd_per_token + flops_bwd_per_token

# Total FLOPs per sample (T=384 tokens):
total_flops_per_sample = total_flops_per_token * max_len

# Total Realized TFLOPs at 1500 samples/sec:
total_flops_per_sec = total_flops_per_sample * samples_per_sec
realized_tflops = total_flops_per_sec / 1e12

# Peak Cluster TFLOPs (8x TPU v5e):
total_peak_tflops = 197.0 * 8  # 1576 TFLOPs
mfu = (realized_tflops / total_peak_tflops) * 100

print("="*65)
print("     EXACT TFLOPS AUDIT: d_model=384, 10 Layers @ 1,500 samp/s     ")
print("="*65)
print(f"Total Active Parameters:      {total_params / 1e6:.2f} Million")
print(f"  - Decoder Layers (10x):     {total_non_emb / 1e6:.2f} Million")
print(f"  - Vocabulary Embedding:     {p_emb / 1e6:.2f} Million")
print(f"Sequence Length:              {max_len} tokens")
print(f"Tokens Processed per Sec:     {samples_per_sec * max_len / 1e3:.1f}k tokens/sec")
print(f"Compute per Sample:           {total_flops_per_sample / 1e9:.2f} GFLOPs")
print("-" * 65)
print(f"REALIZED COMPUTE THROUGHPUT:  {realized_tflops:.2f} TFLOPs")
print(f"Peak Hardware Capacity (8x):  {total_peak_tflops:.1f} TFLOPs (TPU v5e-8)")
print(f"Model FLOPs Utilization (MFU):{mfu:.2f}% MFU")
print("="*65)
