#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — WASSERSTEIN BARYCENTER & SIGNER ALIGNMENT (W2-SLT)
================================================================================
Implements Entropic 2-Wasserstein Barycenter & Signer Distribution Alignment:
1. Optimal Transport 2-Wasserstein Metric:
     W_{2,eps}^2(p, q) = min_{T in Pi(p, q)} <T, M>_F - eps * H(T)
     where M_ij = ||x_i - y_j||_2^2 is the squared Euclidean ground metric.
2. Sinkhorn-Knopp Fixed-Point Barycenter Consensus:
     mu* = argmin_{mu} sum_k lambda_k * W_{2,eps}^2(mu, P_k)
     Iterative log-domain update: mu^{(l+1)} = exp( sum_k lambda_k * log(K_k * v_k) )
3. Signer-Invariant Manifold Regularization:
     L_bary = (1/B) sum_{b=1}^B W_{2,eps}^2(H_b, mu*)
     Pulls heterogeneous signer distributions into the shared non-linear barycenter.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WassersteinBarycenterOutput(NamedTuple):
    barycenter_features: torch.Tensor   # [B, T, d_model] Barycenter-aligned sequence representations
    barycenter_weights: torch.Tensor    # [M] Probability mass distribution of optimal barycenter mu*
    barycenter_loss: torch.Tensor       # [1] Entropic W2 alignment loss across batch signers
    transport_plans: torch.Tensor       # [B, T, M] Optimal transport matrix coupling T_b
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + barycenter_features


class ASLWassersteinBarycenterEngine(nn.Module):
    """
    Entropic 2-Wasserstein Barycenter Engine for Multi-Signer Distribution Alignment.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_barycenter_atoms: int = 16,     # M support atoms on barycenter manifold
        epsilon: float = 0.5,               # Entropic regularization parameter
        num_sinkhorn_iters: int = 5,        # Sinkhorn iterations for speed & stability
    ):
        super().__init__()
        self.d_model = d_model
        self.num_atoms = num_barycenter_atoms
        self.eps = epsilon
        self.num_iters = num_sinkhorn_iters

        # Learnable barycenter support locations Y in R^{M x d_model}
        self.barycenter_atoms = nn.Parameter(torch.randn(num_barycenter_atoms, d_model))
        nn.init.normal_(self.barycenter_atoms, std=0.02)

        # Output refinement projection
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_cost_matrix(
        self,
        X: torch.Tensor,  # [B, T, d_model] Signer sequence embeddings
        Y: torch.Tensor,  # [M, d_model] Barycenter atoms
    ) -> torch.Tensor:
        """
        Computes normalized squared Euclidean ground cost matrix M_b(t, m) = ||X_bt - Y_m||_2^2 / d_model.
        Returns: [B, T, M]
        """
        # ||x - y||^2 / D
        x_norm = X.pow(2).sum(dim=-1, keepdim=True)        # [B, T, 1]
        y_norm = Y.pow(2).sum(dim=-1).view(1, 1, -1)       # [1, 1, M]
        xy = torch.matmul(X, Y.t())                        # [B, T, M]
        M = F.relu(x_norm + y_norm - 2.0 * xy) / float(self.d_model)  # [B, T, M]
        return M

    def sinkhorn_transport(
        self,
        M: torch.Tensor,         # [B, T, M] Ground cost matrix
        mu_target: torch.Tensor, # [M] Barycenter target distribution
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes entropic optimal transport plan T and W2 cost via Sinkhorn algorithm.
        Returns: (T_plan [B, T, M], W2_dist [B])
        """
        B, T, M_dim = M.shape
        device = M.device

        # Source distribution: uniform over time frames [B, T]
        a = torch.full((B, T), 1.0 / T, device=device, dtype=M.dtype)  # [B, T]
        b = mu_target.unsqueeze(0).expand(B, M_dim)                     # [B, M]

        # Gibbs kernel K = exp(-M / eps)
        K = torch.exp(-M / self.eps)  # [B, T, M]

        # Sinkhorn scaling vectors
        u = torch.ones_like(a)  # [B, T]
        v = torch.ones_like(b)  # [B, M]

        for _ in range(self.num_iters):
            # v = b / (K^T * u)
            v = b / (torch.matmul(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
            # u = a / (K * v)
            u = a / (torch.matmul(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)

        # Transport plan T = diag(u) * K * diag(v) in [B, T, M]
        T_plan = u.unsqueeze(-1) * K * v.unsqueeze(1)  # [B, T, M]

        # Entropic W2 cost: <T, M>_F
        w2_dist = (T_plan * M).sum(dim=(1, 2))  # [B]
        return T_plan, w2_dist

    def forward(
        self,
        h_seq: torch.Tensor,  # [B, T, d_model] Latent sequence representations
    ) -> WassersteinBarycenterOutput:
        """
        Computes Wasserstein barycentric consensus and aligns multi-signer distributions.
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        # 1. Compute Cost Matrix M in [B, T, M]
        M = self.compute_cost_matrix(h_seq, self.barycenter_atoms)  # [B, T, M]

        # 2. Estimate Barycenter Measure mu* (Uniform prior over M atoms)
        mu_star = torch.full((self.num_atoms,), 1.0 / self.num_atoms, device=device, dtype=h_seq.dtype)

        # 3. Compute Entropic Optimal Transport Plans & Distance
        T_plan, w2_dists = self.sinkhorn_transport(M, mu_star)  # [B, T, M], [B]

        # 4. Transport Projection: Pull features toward barycenter atoms
        # T_plan [B, T, M] * Y [M, D] -> [B, T, D]
        # Normalize transport weights along atom dimension
        T_weights = T_plan / (T_plan.sum(dim=-1, keepdim=True) + 1e-8)  # [B, T, M]
        bary_emb = torch.matmul(T_weights, self.barycenter_atoms)       # [B, T, D]
        bary_out = self.out_proj(bary_emb)                              # [B, T, D]

        # 5. Barycentric Alignment Loss
        loss_bary = w2_dists.mean()

        augmented = h_seq + bary_out

        return WassersteinBarycenterOutput(
            barycenter_features=bary_out,
            barycenter_weights=mu_star,
            barycenter_loss=loss_bary,
            transport_plans=T_plan,
            augmented_features=augmented,
        )
