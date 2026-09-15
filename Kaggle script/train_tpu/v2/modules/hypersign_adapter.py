#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SIGNER-ADAPTIVE HYPERNETWORK ENGINE (HYPERSIGN)
================================================================================
Implements Dynamic Parameter Modulation & Low-Rank Hyper-LoRA for Signer Adaptation:
1. Unsupervised Signer Style Signature Extractor:
     Extracts signer-invariant anatomical and kinematic style representation s in R^{d_s}:
     s = MeanPool(TemporalConv(x_kinematics))
2. Dynamic Low-Rank Hyper-LoRA Parameter Generation:
     Predicts low-rank adapter matrices A(s) in R^{B x D x r}, B(s) in R^{B x r x D} (rank r=4)
     and FiLM modulation vectors gamma(s), beta(s) in R^{B x 1 x D}:
     W_adapted = W_0 + (alpha / r) * (A(s) @ B(s))
     H_adapted = gamma(s) * (H @ W_adapted) + beta(s)
3. Signer Separation Contrastive Loss:
     Encourages style embeddings from the same signer across different glosses to cluster
     while maximizing distance between different signers.
4. Parameter Efficiency:
     Low-rank formulation reduces parameter generation by 16x vs dense hypernetworks.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperSignOutput(NamedTuple):
    adapted_features: torch.Tensor       # [B, T, d_model]
    signer_style: torch.Tensor           # [B, d_style] Extracted style embedding
    style_contrastive_loss: torch.Tensor # Scalar contrastive style separation loss
    gamma: torch.Tensor                  # [B, 1, d_model] Dynamic scale
    beta: torch.Tensor                   # [B, 1, d_model] Dynamic shift


class SignerStyleExtractor(nn.Module):
    """
    Extracts time-invariant anatomical and dynamic style signatures from kinematics.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_keypoints: int = 60,
        d_style: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.d_style = d_style

        # Input [B, T, K, C] -> flatten joints [B, T, K*C]
        self.conv_net = nn.Sequential(
            nn.Conv1d(num_keypoints * in_channels, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, d_style, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_style),
            nn.GELU(),
        )
        self.style_proj = nn.Sequential(
            nn.Linear(d_style, d_style),
            nn.LayerNorm(d_style),
        )

    def forward(self, kinematics: torch.Tensor) -> torch.Tensor:
        """
        kinematics: [B, T, K, C]
        Returns: [B, d_style] normalized signer style vector
        """
        B, T, K, C = kinematics.shape
        flat_k = kinematics.view(B, T, K * C).transpose(1, 2)  # [B, K*C, T]

        feats = self.conv_net(flat_k)  # [B, d_style, T']
        pooled = feats.mean(dim=-1)     # [B, d_style]

        style = self.style_proj(pooled)
        return F.normalize(style, p=2, dim=-1)


class HyperSignDynamicLoRA(nn.Module):
    """
    Low-Rank Dynamic Parameter Generation & FiLM Modulation.
    """

    def __init__(
        self,
        d_in: int = 128,
        d_out: int = 128,
        d_style: int = 64,
        rank: int = 4,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_style = d_style
        self.rank = rank
        self.scale = alpha / rank

        # Static base linear projection
        self.base_linear = nn.Linear(d_in, d_out)

        # HyperNetworks for low-rank factors A [d_in, rank] and B [rank, d_out]
        self.hyper_A = nn.Sequential(
            nn.Linear(d_style, 64),
            nn.GELU(),
            nn.Linear(64, d_in * rank),
        )
        self.hyper_B = nn.Sequential(
            nn.Linear(d_style, 64),
            nn.GELU(),
            nn.Linear(64, rank * d_out),
        )

        # HyperNetwork for FiLM affine scale gamma and shift beta
        self.hyper_film = nn.Sequential(
            nn.Linear(d_style, 64),
            nn.GELU(),
            nn.Linear(64, d_out * 2),
        )

    def forward(
        self,
        x: torch.Tensor,             # [B, T, d_in]
        signer_style: torch.Tensor,  # [B, d_style]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape

        # 1. Base transformation
        h_base = self.base_linear(x)  # [B, T, d_out]

        # 2. Dynamic Low-Rank LoRA Adaptation
        A = self.hyper_A(signer_style).view(B, self.d_in, self.rank)    # [B, d_in, r]
        B_mat = self.hyper_B(signer_style).view(B, self.rank, self.d_out) # [B, r, d_out]

        # Low-rank forward: (x @ A) @ B
        # [B, T, d_in] @ [B, d_in, r] -> [B, T, r]
        x_A = torch.bmm(x, A)
        # [B, T, r] @ [B, r, d_out] -> [B, T, d_out]
        delta_h = torch.bmm(x_A, B_mat) * self.scale

        h_lora = h_base + delta_h  # [B, T, d_out]

        # 3. Dynamic FiLM Modulation
        film_params = self.hyper_film(signer_style)  # [B, 2 * d_out]
        gamma = 1.0 + film_params[:, :self.d_out].unsqueeze(1)  # [B, 1, d_out] (centered at 1.0)
        beta = film_params[:, self.d_out:].unsqueeze(1)         # [B, 1, d_out] (centered at 0.0)

        h_adapted = gamma * h_lora + beta  # [B, T, d_out]

        return h_adapted, gamma, beta


class ASLHyperSignAdapterEngine(nn.Module):
    """
    Unified Signer-Adaptive HyperNetwork Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        d_style: int = 64,
        rank: int = 4,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_style = d_style
        self.temperature = temperature

        self.style_extractor = SignerStyleExtractor(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            d_style=d_style,
        )

        self.dynamic_adapter = HyperSignDynamicLoRA(
            d_in=d_model,
            d_out=d_model,
            d_style=d_style,
            rank=rank,
        )

    def forward(
        self,
        h_seq: torch.Tensor,                     # [B, T, d_model]
        kinematics_9ch: torch.Tensor,           # [B, T, K, 9]
        signer_ids: Optional[torch.Tensor] = None, # [B] optional integer signer identities
    ) -> HyperSignOutput:
        """
        Executes signer style extraction, dynamic parameter modulation, and contrastive loss.
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        # 1. Extract Signer Style Embedding
        signer_style = self.style_extractor(kinematics_9ch)  # [B, d_style]

        # 2. Dynamic Low-Rank & FiLM Adaptation
        h_adapted, gamma, beta = self.dynamic_adapter(h_seq, signer_style)  # [B, T, d_model]

        # 3. Supervised / Self-Supervised Style Contrastive Loss
        if signer_ids is not None and B > 1:
            # Pairwise cosine similarity: [B, B]
            sim_mat = torch.matmul(signer_style, signer_style.t()) / self.temperature
            # Positive mask where signer_ids match: [B, B]
            pos_mask = (signer_ids.unsqueeze(0) == signer_ids.unsqueeze(1)).float()
            # Remove self-similarity from positive count
            diag_mask = torch.eye(B, device=device)
            pos_mask_no_diag = pos_mask * (1.0 - diag_mask)

            # Log-sum-exp denominator
            exp_sim = torch.exp(sim_mat) * (1.0 - diag_mask)
            log_denom = torch.log(exp_sim.sum(dim=-1, keepdim=True).clamp(min=1e-6))

            # InfoNCE style loss
            if pos_mask_no_diag.sum() > 0:
                pos_sim = (sim_mat * pos_mask_no_diag).sum(dim=-1, keepdim=True) / pos_mask_no_diag.sum(dim=-1, keepdim=True).clamp(min=1.0)
                loss_contrastive = (-pos_sim + log_denom).mean()
            else:
                loss_contrastive = torch.tensor(0.0, device=device)
        else:
            # Self-regularization: maximize style embedding entropy
            loss_contrastive = (signer_style.pow(2).mean() - 1.0).pow(2)

        return HyperSignOutput(
            adapted_features=h_adapted,
            signer_style=signer_style,
            style_contrastive_loss=loss_contrastive,
            gamma=gamma,
            beta=beta,
        )
