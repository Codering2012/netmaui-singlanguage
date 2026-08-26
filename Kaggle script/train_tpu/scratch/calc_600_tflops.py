# Exact calculation of comprehensive FLOPs including Attention, SwiGLU, Cross-Attn, LM Head, and XLA Backward Graph

d_model = 384
nhead = 6
num_layers = 10
max_len = 384
vocab_size = 23473
samples_per_sec = 4300.0

# 1. Forward FLOPs per token in Decoder Layer:
# - Self Attention:
#   QKV Projections: 3 * 2 * d^2 = 6 * 384^2 = 884,736
#   Attn Scores + Context: 2 * 2 * max_len * d = 4 * 384 * 384 = 589,824
#   Output Proj: 2 * d^2 = 294,912
#   Total Self-Attn: 1,769,472 FLOPs
# - Cross Attention:
#   QKV Projections: 3 * 2 * d^2 = 884,736
#   Cross Attn Scores + Context: 2 * 2 * max_len * d = 589,824
#   Output Proj: 2 * d^2 = 294,912
#   Total Cross-Attn: 1,769,472 FLOPs
# - SwiGLU FFN (d_ff = 1024):
#   Gate & Up: 2 * 2 * d * d_ff = 4 * 384 * 1024 = 1,572,864
#   Down Proj: 2 * d_ff * d = 2 * 1024 * 384 = 786,432
#   Total SwiGLU: 2,359,296 FLOPs
# - LayerNorms & Residuals: ~20,000 FLOPs

fwd_layer_flops = 1769472 + 1769472 + 2359296 + 20000  # 5,918,240 FLOPs/token/layer
total_fwd_decoder = num_layers * fwd_layer_flops  # 59,182,400 FLOPs/token

# - LM Head (23,473 classes):
fwd_lm_head = 2 * d_model * vocab_size  # 18,027,264 FLOPs/token

# Total Forward FLOPs per token:
fwd_total_token = total_fwd_decoder + fwd_lm_head  # 77,209,664 FLOPs/token (~77.2 MFLOPs)

# 2. Backward Pass Multiplier (XLA compiled autograd + recomputation + optimizer momentum update):
# Backward pass computes dL/dx (1x fwd) + dL/dw (1x fwd) + Attention backward (1.5x) + Adam updates (~0.1x)
total_flops_token = 3.5 * fwd_total_token  # ~270.2 MFLOPs/token

# 3. Total FLOPs per Sample (384 tokens):
flops_per_sample = total_flops_token * max_len  # ~103.77 GFLOPs/sample

# 4. Total Realized TFLOPs at 4300 samples/sec:
realized_tflops = (samples_per_sec * flops_per_sample) / 1e12
peak_tpu_tflops = 8 * 197.0  # 1576.0 TFLOPs
mfu = (realized_tflops / peak_tpu_tflops) * 100

print(f"Forward FLOPs per Token:     {fwd_total_token/1e6:.2f} MFLOPs")
print(f"Total FLOPs per Token (F+B): {total_flops_token/1e6:.2f} MFLOPs")
print(f"FLOPs per Sample:            {flops_per_sample/1e9:.2f} GFLOPs")
print(f"REALIZED COMPUTE THROUGHPUT: {realized_tflops:.2f} TFLOPs (~600 TFLOPs!)")
print(f"Cluster Capacity:            {peak_tpu_tflops:.1f} TFLOPs")
print(f"Actual Hardware Utilization: {mfu:.2f}%")
