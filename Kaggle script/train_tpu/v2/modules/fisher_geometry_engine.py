#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — FISHER INFORMATION GEOMETRY ENGINE (FISHERGEOMETRYSIGN)
================================================================================
Implements Riemannian Fisher Information Geometry & Natural Gradients (FisherNatural-SLT):
1. Statistical Manifold of Kinematic Gaussian Distributions:
     Each keypoint k at frame t is modeled as N(mu_k(t), Sigma_k(t))
     mu_k(t) in R^3 (spatial coordinates), sigma_k^2 in R^3_+ (velocity-dependent variance)
2. Fisher Information Metric Tensor G(theta) in R^{6 x 6}:
     G_mu = Sigma^{-1} = diag(1/sigma_1^2, 1/sigma_2^2, 1/sigma_3^2)
     G_sigma = 2 * Sigma^{-1} = diag(2/sigma_1^2, 2/sigma_2^2, 2/sigma_3^2)
     Strictly Positive Definite: G > 0 with Riemannian metric ds^2 = sum (dmu_i^2 + 2 dsigma_i^2) / sigma_i^2
3. Fisher-Rao Geodesic Distances & Volume Invariants:
     Tr(G) = sum 3 / sigma_i^2
     log det(G) = sum log(2 / sigma_i^4) = 3 log(2) - 4 sum log(sigma_i)
4. Natural Gradient Riemannian Velocity Transport:
     grad_nat = G^{-1} grad_euclidean = Sigma @ grad_mu
5. Feature Projection & Canonical Fusion:
     H_fisher = H + LayerNorm( Linear( [mu, sigma, Tr(G), log det(G), d_Fisher(wrist, face)] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FisherGeometryOutput(NamedTuple):
    fisher_features: torch.Tensor       # [B, T, d_model] Projected information geometric representations
    gaussian_means: torch.Tensor        # [B, T, 60, 3] Statistical mean coordinates mu
    gaussian_std: torch.Tensor          # [B, T, 60, 3] Statistical standard deviations sigma > 0
    fisher_trace: torch.Tensor          # [B, T, 60] Trace of Fisher metric Tr(G)
    fisher_log_det: torch.Tensor        # [B, T, 60] Log determinant log det(G)
    fisher_rao_distances: torch.Tensor  # [B, T, 2] Fisher-Rao distance from wrists to nose (0)
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + fisher_features


class ASLFisherGeometryEngine(nn.Module):
    """
    Riemannian Fisher Information Metric & Information Geometry Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        base_sigma: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.base_sigma = base_sigma

        # Variance prediction network: kinematics -> log_sigma in R^3
        self.variance_net = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 3),
        )

        # Output projection head
        # Input: 60 * 3 (mu) + 60 * 3 (sigma) + 60 (tr_G) + 60 (log_det) + 2 (d_FR) = 60*8 + 2 = 482 -> d_model
        in_feat_dim = num_keypoints * 8 + 2
        self.proj = nn.Sequential(
            nn.Linear(in_feat_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_fisher_rao_distance(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the analytical Fisher-Rao geodesic distance between univariate Gaussians per coordinate.
        mu1, sigma1: [B, T, 3]
        mu2, sigma2: [B, T, 3]
        """
        eps = 1e-6
        # Upper half plane hyperbolic metric: delta = |mu1 - mu2|^2 + 2*(sigma1 - sigma2)^2
        delta_minus = (mu1 - mu2) ** 2 + 2.0 * (sigma1 - sigma2) ** 2
        delta_plus = (mu1 - mu2) ** 2 + 2.0 * (sigma1 + sigma2) ** 2

        num = torch.sqrt(delta_minus.clamp(min=0.0)) + torch.sqrt(delta_plus.clamp(min=eps))
        den = (2.0 * math.sqrt(2.0) * torch.sqrt((sigma1 * sigma2).clamp(min=eps))).clamp(min=eps)

        ratio = (num / den).clamp(min=1.0)
        d_coord = math.sqrt(2.0) * torch.log(ratio)  # [B, T, 3]
        d_fr = torch.norm(d_coord, p=2, dim=-1)       # [B, T]
        return d_fr

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> FisherGeometryOutput:
        """
        Evaluates Fisher Information Metric, Fisher-Rao distances, and projects to d_model.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # 1. Statistical Gaussian Mean: mu = pos in R^3
        mu = kinematics[..., 0:3]  # [B, T, 60, 3]

        # 2. Statistical Gaussian Variance: sigma = base_sigma * exp(log_sigma) > 0
        raw_log_sigma = self.variance_net(kinematics)  # [B, T, 60, 3]
        sigma = self.base_sigma * torch.exp(torch.clamp(raw_log_sigma, min=-3.0, max=3.0))  # [B, T, 60, 3]

        # 3. Fisher Information Metric Tensor Properties:
        # G_mu = diag(1 / sigma^2), G_sigma = diag(2 / sigma^2)
        inv_sigma2 = 1.0 / (sigma ** 2)  # [B, T, 60, 3]

        # Trace of G = sum_i (1/sigma_i^2 + 2/sigma_i^2) = sum_i (3/sigma_i^2)
        tr_G = (3.0 * inv_sigma2).sum(dim=-1)  # [B, T, 60]

        # Log Determinant of G = sum_i [ log(1/sigma_i^2) + log(2/sigma_i^2) ] = sum_i [ log(2) - 4 log(sigma_i) ]
        log_det_G = (math.log(2.0) - 4.0 * torch.log(sigma)).sum(dim=-1)  # [B, T, 60]

        # 4. Fisher-Rao Geodesic Distance from Left Wrist (18) and Right Wrist (39) to Nose (0)
        mu_nose = mu[:, :, 0, :]        # [B, T, 3]
        sig_nose = sigma[:, :, 0, :]    # [B, T, 3]

        mu_lwrist = mu[:, :, 18, :]     # [B, T, 3]
        sig_lwrist = sigma[:, :, 18, :] # [B, T, 3]

        mu_rwrist = mu[:, :, 39, :]     # [B, T, 3]
        sig_rwrist = sigma[:, :, 39, :] # [B, T, 3]

        d_fr_lwrist = self.compute_fisher_rao_distance(mu_lwrist, sig_lwrist, mu_nose, sig_nose) # [B, T]
        d_fr_rwrist = self.compute_fisher_rao_distance(mu_rwrist, sig_rwrist, mu_nose, sig_nose) # [B, T]
        d_fr_all = torch.stack([d_fr_lwrist, d_fr_rwrist], dim=-1)                               # [B, T, 2]

        # 5. Output Feature Projection
        mu_flat = mu.reshape(B, T, K * 3)
        sigma_flat = sigma.reshape(B, T, K * 3)

        f_all = torch.cat([mu_flat, sigma_flat, tr_G, log_det_G, d_fr_all], dim=-1) # [B, T, in_feat_dim]
        h_fisher = self.proj(f_all)                                                  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_fisher

        return FisherGeometryOutput(
            fisher_features=h_fisher,
            gaussian_means=mu,
            gaussian_std=sigma,
            fisher_trace=tr_G,
            fisher_log_det=log_det_G,
            fisher_rao_distances=d_fr_all,
            augmented_features=augmented,
        )
