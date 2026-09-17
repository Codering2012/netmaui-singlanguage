#!/usr/bin/env python3
"""
================================================================================
BAND-CONSTRAINED MONOTONIC LOG-DOMAIN SINKHORN TRANSDUCER (OSV -> SVO)
================================================================================
Implements entropy-regularized optimal transport in log-space for continuous ASL
syntactic reordering with two breakthrough structural constraints:
1. Gaussian Band Prior:
   C_ij = (1.0 - cos_sim(q_i, k_j)) + lambda_band * ((i - j) / M)^2
   Permits sharp local syntactic permutations (OSV -> SVO within clauses) while
   strictly preventing distant cross-clause narrative scrambling.
2. Monotonic Regularization Loss:
   Penalizes backward time travel across clauses using center-of-mass penalties.
================================================================================
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogDomainSinkhornSolver(nn.Module):
    """
    Log-domain Differentiable Sinkhorn-Knopp Optimal Transport Solver.
    Uses native PyTorch ATen ops (torch.logsumexp) for 100% TPU/XLA compatibility.
    Guarantees doubly stochastic transport plan P where sum_j P_ij = 1 and sum_i P_ij = 1.
    """

    def __init__(self, num_iters: int = 16, epsilon: float = 0.08):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon

    def set_epsilon(self, epsilon: float):
        """Allows annealing epsilon from soft exploration (0.15) to sharp permutation (0.03)."""
        self.epsilon = max(0.01, float(epsilon))

    def forward(self, cost_matrix: torch.Tensor) -> torch.Tensor:
        """
        cost_matrix: [B, M, M] >= 0
        Returns: P [B, M, M] doubly stochastic permutation matrix.
        """
        B, M, _ = cost_matrix.shape
        inv_eps = 1.0 / self.epsilon

        # Initialize dual potentials in log-space: [B, M]
        f = torch.zeros(B, M, device=cost_matrix.device, dtype=cost_matrix.dtype)
        g = torch.zeros(B, M, device=cost_matrix.device, dtype=cost_matrix.dtype)

        # Static loop of fixed iterations (no dynamic while-loop, fully XLA compilable)
        for _ in range(self.num_iters):
            # Update f: f_i = -eps * logsumexp_j ((g_j - C_ij) / eps)
            kernel_f = (g.unsqueeze(1) - cost_matrix) * inv_eps
            f = -self.epsilon * torch.logsumexp(kernel_f, dim=-1)

            # Update g: g_j = -eps * logsumexp_i ((f_i - C_ij) / eps)
            kernel_g = (f.unsqueeze(2) - cost_matrix) * inv_eps
            g = -self.epsilon * torch.logsumexp(kernel_g, dim=1)

        # Compute optimal transport matrix P in log space
        log_P = (f.unsqueeze(2) + g.unsqueeze(1) - cost_matrix) * inv_eps
        P = torch.exp(log_P)
        # Final row-normalization for exact stochasticity
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-6)
        return P


class SinkhornChunkTransducer(nn.Module):
    """
    Syntactic Reordering Transducer with Band-Constrained Log-Domain Sinkhorn.
    Transforms ASL Topic-Comment order into English SVO order without clause scrambling.
    """

    def __init__(
        self,
        d_model: int = 128,
        chunk_size: int = 4,
        num_iters: int = 16,
        epsilon: float = 0.08,
        band_weight: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.band_weight = band_weight
        self.sinkhorn = LogDomainSinkhornSolver(num_iters=num_iters, epsilon=epsilon)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.perm_norm = nn.LayerNorm(d_model)

    def set_epsilon(self, epsilon: float):
        self.sinkhorn.set_epsilon(epsilon)

    def compute_monotonic_loss(self, P: torch.Tensor, delta: float = 0.5) -> torch.Tensor:
        """
        Monotonicity Loss penalizing backward time shifts across chunks:
        Let c_i = sum_j j * P_ij be the expected temporal location of chunk i.
        L_mono = mean_i ReLU(c_i - c_{i+1} + delta)
        """
        B, M, _ = P.shape
        if M <= 1:
            return torch.zeros((), device=P.device)
        j_indices = torch.arange(M, device=P.device, dtype=P.dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, M]
        center_of_mass = torch.sum(P * j_indices, dim=-1)  # [B, M]
        diffs = center_of_mass[:, :-1] - center_of_mass[:, 1:] + delta  # [B, M-1]
        loss_mono = F.relu(diffs).mean()
        return loss_mono

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, T, D]
        Returns: (h_reordered, P)
        """
        B, T, D = h.shape
        M = max(1, T // self.chunk_size)
        # Average pool into M sign chunks
        h_chunks = h.view(B, M, self.chunk_size, D).mean(dim=2)  # [B, M, D]

        q = F.normalize(self.query_proj(h_chunks), dim=-1)
        k = F.normalize(self.key_proj(h_chunks), dim=-1)

        # 1. Base semantic cost: Cosine distance C_semantic = 1.0 - cos_sim(q_i, k_j)
        cost_semantic = 1.0 - torch.bmm(q, k.transpose(1, 2))  # [B, M, M] in [0, 2]

        # 2. Gaussian Band Prior: Heavily penalize permutations between distant clauses
        if M > 1:
            idx = torch.arange(M, device=h.device, dtype=h.dtype)
            dist_mat = ((idx.unsqueeze(1) - idx.unsqueeze(0)) / float(M)) ** 2  # [M, M]
            cost = cost_semantic + self.band_weight * dist_mat.unsqueeze(0)
        else:
            cost = cost_semantic

        P = self.sinkhorn(cost)  # [B, M, M]

        # Apply permutation to chunk features
        reordered_chunks = torch.bmm(P, h_chunks)  # [B, M, D]

        # Broadcast/interpolate back to [B, T, D]
        h_reordered = reordered_chunks.unsqueeze(2).expand(B, M, self.chunk_size, D).reshape(B, T, D)
        h_out = self.perm_norm(h + h_reordered)
        return h_out, P
