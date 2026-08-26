import torch
import torch.nn.functional as F

print("--- Testing BFloat16 vs Float32 LogSoftmax & Loss Numerics ---")
# Simulate batch 128, seq 384, vocab 23473
bsz, seq, vocab = 128, 384, 23473

# Create random BF16 logits
logits_bf16 = torch.randn(bsz, seq, vocab, dtype=torch.bfloat16)
targets = torch.randint(0, vocab, (bsz, seq), dtype=torch.long)

# 1. BF16 log_softmax
lf_bf16 = logits_bf16.reshape(-1, vocab)
tf = targets.reshape(-1)
log_probs_bf16 = F.log_softmax(lf_bf16, dim=-1)
nll_bf16 = -log_probs_bf16.gather(dim=-1, index=tf.unsqueeze(-1)).squeeze(-1)

# 2. FP32 log_softmax
lf_fp32 = logits_bf16.reshape(-1, vocab).float()
log_probs_fp32 = F.log_softmax(lf_fp32, dim=-1)
nll_fp32 = -log_probs_fp32.gather(dim=-1, index=tf.unsqueeze(-1)).squeeze(-1)

diff = (nll_bf16.float() - nll_fp32).abs().max().item()
print(f"Max numerical difference between BF16 and FP32 loss: {diff:.6f}")
print(f"BF16 memory footprint: {lf_bf16.element_size() * lf_bf16.nelement() / (1024*1024*1024) * 2:.2f} GB (lf + log_probs)")
print(f"FP32 memory footprint: {lf_fp32.element_size() * lf_fp32.nelement() / (1024*1024*1024) * 2:.2f} GB (lf + log_probs)")
print(f"Memory Saved: {(lf_fp32.element_size() - lf_bf16.element_size()) * lf_bf16.nelement() / (1024*1024*1024) * 2:.2f} GB!")
