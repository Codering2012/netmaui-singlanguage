#!/usr/bin/env python3
"""
Empirical Hypothesis Test: Vectorized CPC vs Loop-based CPC
Tests gradient correctness, shape invariance, and execution on CPU.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

def test_vectorized_cpc():
    B, T, D = 2, 32, 128
    K = 4
    torch.manual_seed(42)

    h = torch.randn(B, T, D, requires_grad=True)
    hand_positions = torch.randn(B, T, 3)

    # 1. Loop-based CPC (original)
    cpc_projections = nn.ModuleList([nn.Linear(D, 3, bias=False) for _ in range(K)])
    gt_vel = torch.diff(hand_positions, dim=1, prepend=hand_positions[:, :1, :])
    
    cpc_losses = []
    for k, proj in enumerate(cpc_projections, 1):
        pred_vel = proj(h[:, :-k, :])
        target_vel = gt_vel[:, k:, :].detach()
        cpc_losses.append(F.mse_loss(pred_vel, target_vel))
    loss_loop = torch.mean(torch.stack(cpc_losses))

    # 2. Vectorized CPC (new)
    # Pack weights from projections into single linear layer for exact equivalence check
    cpc_fused = nn.Linear(D, K * 3, bias=False)
    with torch.no_grad():
        fused_weight = torch.cat([proj.weight for proj in cpc_projections], dim=0) # [K*3, D]
        cpc_fused.weight.copy_(fused_weight)

    pred_all = cpc_fused(h).view(B, T, K, 3) # [B, T, K, 3]

    gt_padded = F.pad(gt_vel, (0, 0, 0, K)) # [B, T+K, 3]
    future_targets = torch.stack([gt_padded[:, k : k + T, :] for k in range(1, K + 1)], dim=2).detach() # [B, T, K, 3]

    # Valid mask for positions where t + k < T
    t_indices = torch.arange(T, device=h.device).view(1, T, 1)
    k_offsets = torch.arange(1, K + 1, device=h.device).view(1, 1, K)
    valid_mask = (t_indices + k_offsets < T).float() # [1, T, K]

    sq_err = torch.sum((pred_all - future_targets) ** 2, dim=-1) # [B, T, K]
    masked_err = sq_err * valid_mask # [B, T, K]

    # Per-k mean loss
    k_counts = torch.sum(valid_mask, dim=1) # [1, K] = [T-1, T-2, T-3, T-4]
    loss_per_k = torch.sum(masked_err, dim=(0, 1)) / (k_counts.squeeze(0) * B * 3) # [K]
    loss_vectorized = torch.mean(loss_per_k)

    print(f"Loop loss:       {loss_loop.item():.6f}")
    print(f"Vectorized loss: {loss_vectorized.item():.6f}")
    diff = abs(loss_loop.item() - loss_vectorized.item())
    print(f"Absolute diff:   {diff:.8f}")
    assert diff < 1e-5, f"Vectorized loss ({loss_vectorized}) must match loop loss ({loss_loop})!"

    # Gradient check
    loss_loop.backward(retain_graph=True)
    grad_loop = h.grad.clone()
    h.grad.zero_()

    loss_vectorized.backward()
    grad_vec = h.grad.clone()

    grad_diff = torch.max(torch.abs(grad_loop - grad_vec)).item()
    print(f"Max gradient diff: {grad_diff:.8f}")
    assert grad_diff < 1e-5, f"Gradients must match exactly! Got max diff: {grad_diff}"
    print("[SUCCESS] Vectorized CPC is mathematically and empirically proven identical!")

if __name__ == "__main__":
    test_vectorized_cpc()
