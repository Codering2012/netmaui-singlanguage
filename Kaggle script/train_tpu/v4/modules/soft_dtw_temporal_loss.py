#!/usr/bin/env python3
"""
================================================================================
ASL V4: DIFFERENTIABLE SOFT DYNAMIC TIME WARPING (SOFT-DTW) LOSS
================================================================================
Computes differentiable temporal alignment between continuous sign representations
and target text/gloss embeddings (Cuturi & Blondel, ICML 2017).

Mathematical Formulation:
min^gamma(a, b, c) = -gamma * log( exp(-a/gamma) + exp(-b/gamma) + exp(-c/gamma) )
R_{i, j} = D_{i, j} + min^gamma( R_{i-1, j}, R_{i, j-1}, R_{i-1, j-1} )

Guarantees smooth sub-frame temporal boundary gradients without trellis collapse.
================================================================================
"""

import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_min3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, gamma: float) -> torch.Tensor:
    """Computes differentiable soft minimum of three tensors with temperature gamma."""
    stacked = torch.stack([-a / gamma, -b / gamma, -c / gamma], dim=0)
    return -gamma * torch.logsumexp(stacked, dim=0)


class SoftDTWLoss(nn.Module):
    """
    Differentiable Soft-DTW alignment loss for continuous sign language.
    """

    def __init__(self, gamma: float = 0.1, normalize: bool = True):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def _calc_distance_matrix(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise squared Euclidean distance matrix between x [B, N, D] and y [B, M, D].
        Returns: [B, N, M]
        """
        # x: [B, N, D], y: [B, M, D]
        x_norm = (x ** 2).sum(dim=-1, keepdim=True)       # [B, N, 1]
        y_norm = (y ** 2).sum(dim=-1).unsqueeze(1)        # [B, 1, M]
        xy = torch.bmm(x, y.transpose(1, 2))              # [B, N, M]
        dist_sq = F.relu(x_norm + y_norm - 2.0 * xy)
        return dist_sq

    def _forward_soft_dtw(self, D_mat: torch.Tensor) -> torch.Tensor:
        """
        Computes forward pass of Soft-DTW DP matrix.
        D_mat: [B, N, M]
        Returns: [B] Soft-DTW discrepancy
        """
        B, N, M = D_mat.shape
        device = D_mat.device
        dtype = D_mat.dtype

        # Allocate DP table R with boundary padding [B, N+1, M+1]
        # Initialize with +infinity (large positive value)
        large_val = 1e6
        R = torch.full((B, N + 1, M + 1), large_val, device=device, dtype=dtype)
        R[:, 0, 0] = 0.0

        for i in range(1, N + 1):
            for j in range(1, M + 1):
                cost = D_mat[:, i - 1, j - 1]
                r_up = R[:, i - 1, j]
                r_left = R[:, i, j - 1]
                r_diag = R[:, i - 1, j - 1]
                soft_min = _soft_min3(r_up, r_left, r_diag, self.gamma)
                R[:, i, j] = cost + soft_min

        return R[:, N, M]

    def forward(
        self,
        visual_features: torch.Tensor,                              # [B, T_vis, D]
        text_features: torch.Tensor,                                # [B, L_text, D]
        vis_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes Normalized Soft-DTW Discrepancy:
        Loss = SoftDTW(X, Y) - 0.5 * (SoftDTW(X, X) + SoftDTW(Y, Y))
        """
        # Unit normalization for stability
        x = F.normalize(visual_features.float(), dim=-1)
        y = F.normalize(text_features.float(), dim=-1)

        D_xy = self._calc_distance_matrix(x, y)
        s_xy = self._forward_soft_dtw(D_xy)

        if self.normalize:
            D_xx = self._calc_distance_matrix(x, x)
            D_yy = self._calc_distance_matrix(y, y)
            s_xx = self._forward_soft_dtw(D_xx)
            s_yy = self._forward_soft_dtw(D_yy)
            # Divergence form guarantees non-negativity: d(X, Y) = S(X, Y) - 0.5 * (S(X, X) + S(Y, Y))
            loss = F.relu(s_xy - 0.5 * (s_xx + s_yy)).mean()
        else:
            loss = s_xy.mean()

        return loss
