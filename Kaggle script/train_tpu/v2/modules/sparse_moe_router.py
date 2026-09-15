#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SPARSE MIXTURE-OF-EXPERTS (MOE) ROUTING ENGINE
================================================================================
Implements Multi-Modal Sparse MoE with Load Balancing (Switch / ST-MoE / MultiStream):
1. Top-K Sparse Gating (k=2 out of E=4 experts):
     Expert 1: Fingerspelling & Rapid Micro-Hand Articulation
     Expert 2: Continuous Gesture & Dynamic Spatial Kinematics
     Expert 3: Facial Mouthing & Non-Manual Affective Cues
     Expert 4: Linguistic Syntax & Cross-Modal Context
2. Auxiliary Load-Balancing & Stability Losses:
     L_balance = E * sum_{e=1}^E f_e * P_e  (Prevents expert collapse)
     L_z = 1/T * sum ( log sum exp(h_gate) )^2  (Stabilizes BF16/FP16 router logits)
3. 4x Model Capacity with Constant Compute & Low Inference Latency.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEForwardOutput(NamedTuple):
    output: torch.Tensor                # [B, T, d_model]
    load_balance_loss: torch.Tensor     # Scalar auxiliary loss
    router_z_loss: torch.Tensor         # Scalar router stability loss
    expert_assignments: torch.Tensor    # [B, T, k]


class ASLSparseMoEExpert(nn.Module):
    """
    Specialized Feed-Forward Network Expert.
    """
    def __init__(self, d_model: int = 128, ffn_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASLSparseMoERouter(nn.Module):
    """
    Sparse Mixture-of-Experts Router with Load-Balancing and Z-Loss Regularization.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_experts: int = 4,
        top_k: int = 2,
        ffn_dim: int = 256,
        load_balance_coef: float = 0.01,
        router_z_coef: float = 0.001,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.load_balance_coef = load_balance_coef
        self.router_z_coef = router_z_coef

        # Gating Router
        self.gate = nn.Linear(d_model, num_experts, bias=False)

        # Parallel Specialized Experts
        self.experts = nn.ModuleList([
            ASLSparseMoEExpert(d_model=d_model, ffn_dim=ffn_dim, dropout=dropout)
            for _ in range(num_experts)
        ])

    def compute_auxiliary_losses(
        self,
        gate_logits: torch.Tensor,       # [N, num_experts]
        topk_indices: torch.Tensor,      # [N, top_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes auxiliary load balance loss and router z-loss.
        """
        N, E = gate_logits.shape
        gate_probs = F.softmax(gate_logits, dim=-1)  # [N, E]

        # 1. Load Balance Loss: L_balance = E * sum(f_e * P_e)
        # Fraction of tokens dispatched to expert e
        top1_idx = topk_indices[:, 0]  # [N]
        f_e = torch.zeros(E, device=gate_logits.device, dtype=gate_logits.dtype)
        for e in range(E):
            f_e[e] = (top1_idx == e).float().mean()

        P_e = gate_probs.mean(dim=0)  # [E]
        balance_loss = E * torch.sum(f_e * P_e)

        # 2. Router Z-Loss: L_z = 1/N * sum ( log sum exp(logits) )^2
        log_z = torch.logsumexp(gate_logits, dim=-1)  # [N]
        z_loss = torch.mean(log_z ** 2)

        return balance_loss, z_loss

    def forward(self, x: torch.Tensor) -> MoEForwardOutput:
        """
        Routes tokens to top-k experts and linearly combines their outputs:
        x: [B, T, d_model]
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)  # [N, D] where N = B * T
        N = B * T

        # 1. Compute Gating Logits & Probabilities
        gate_logits = self.gate(x_flat)  # [N, E]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # 2. Select Top-K Experts
        topk_weights, topk_indices = torch.topk(gate_probs, k=self.top_k, dim=-1)  # [N, k], [N, k]
        # Re-normalize top-k routing weights
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-5)

        # 3. Compute Auxiliary Losses
        balance_loss, z_loss = self.compute_auxiliary_losses(gate_logits, topk_indices)

        # 4. Dispatch Tokens to Experts & Aggregate
        out_flat = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_idx_k = topk_indices[:, k]  # [N]
            weight_k = topk_weights[:, k:k + 1]  # [N, 1]

            for e in range(self.num_experts):
                mask = expert_idx_k == e
                if mask.any():
                    tokens_e = x_flat[mask]
                    expert_out = self.experts[e](tokens_e)
                    out_flat[mask] += weight_k[mask] * expert_out

        out = out_flat.view(B, T, D)
        expert_assignments = topk_indices.view(B, T, self.top_k)

        return MoEForwardOutput(
            output=out,
            load_balance_loss=self.load_balance_coef * balance_loss,
            router_z_loss=self.router_z_coef * z_loss,
            expert_assignments=expert_assignments,
        )
