import math

d_model = 384
nhead = 6
num_layers = 10
max_len = 384
vocab_size = 23473
samples_per_sec = 4343.2

# 1. Parameter counts
p_emb = vocab_size * d_model  # 9,013,632
p_attn_self = 4 * (d_model**2)  # 589,824
p_attn_cross = 4 * (d_model**2)  # 589,824
d_ff = int(8 * d_model / 3)  # 1024
p_ffn = 3 * d_model * d_ff  # 1,179,648
p_layer = p_attn_self + p_attn_cross + p_ffn
total_non_emb = num_layers * p_layer  # 23,592,960
total_params = total_non_emb + p_emb  # 32,606,592

# 2. FLOPs per token (Forward + Backward = 3x fwd)
flops_self = 8 * (d_model**2) + 4 * d_model * max_len
flops_cross = 8 * (d_model**2) + 4 * d_model * max_len
flops_ffn = 2 * p_ffn
flops_head = 2 * vocab_size * d_model

flops_fwd_token = num_layers * (flops_self + flops_cross + flops_ffn) + flops_head
flops_total_token = 3 * flops_fwd_token

flops_sample = flops_total_token * max_len

# 3. Realized TFLOPs & MFU
tokens_per_sec = samples_per_sec * max_len
total_tflops = (samples_per_sec * flops_sample) / 1e12
peak_cluster_tflops = 8 * 197.0  # 1576.0 TFLOPs
mfu = (total_tflops / peak_cluster_tflops) * 100

print(f"Total Active Parameters:     {total_params/1e6:.2f} Million")
print(f"Tokens Processed per Sec:    {tokens_per_sec:,.1f} tokens/s ({tokens_per_sec/1e6:.2f}M tokens/s)")
print(f"Compute per Sample:          {flops_sample/1e9:.2f} GFLOPs")
print(f"REALIZED COMPUTE THROUGHPUT: {total_tflops:.2f} TFLOPs")
print(f"Peak 8x TPU v5e Capacity:    {peak_cluster_tflops:.1f} TFLOPs")
print(f"MODEL FLOPs UTILIZATION MFU: {mfu:.2f}%")
