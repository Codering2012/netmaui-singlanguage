#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SIGNER-INVARIANT ADVERSARIAL NORMALIZATION ENGINE
================================================================================
Implements Morphological Skeleton Normalization & Adversarial Disentanglement:
1. Morphological Procrustes Normalization:
     x_norm = (x - c_nose) / ||x_{L_shoulder} - x_{R_shoulder}||_2
   Removes signer body size, arm length, and camera distance variances.
2. Adversarial Signer Disentanglement (GRL / TA3N):
     L_{signer, adv} = CrossEntropy( SignerClassifier( GRL_lambda(h_cls) ), y_signer )
   Forces latent representations to be invariant to signer identity.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (Ganin & Lempitsky).
    Forward: Identity mapping.
    Backward: Multiplies gradients by -lambda.
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_output.neg() * ctx.alpha, None


class ASLSignerInvariantAdapter(nn.Module):
    """
    Morphological skeleton normalizer and adversarial signer identity disentanglement engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_signers: int = 20,
        grl_alpha: float = 0.50,
        nose_idx: int = 0,
        l_shoulder_idx: int = 53,
        r_shoulder_idx: int = 54,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_signers = num_signers
        self.grl_alpha = grl_alpha
        self.nose_idx = nose_idx
        self.l_shoulder_idx = l_shoulder_idx
        self.r_shoulder_idx = r_shoulder_idx

        # Adversarial Signer Identity Classifier
        self.signer_discriminator = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(d_model, num_signers),
        )

    def normalize_morphology(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Applies morphological Procrustes normalization:
        landmarks: [B, T, K, C] (C >= 3)
        Returns: normalized landmarks of identical shape.
        """
        B, T, K, C = landmarks.shape
        pos = landmarks[..., :3]

        # 1. Center on Nose
        nose_center = pos[:, :, self.nose_idx:self.nose_idx + 1, :]  # [B, T, 1, 3]
        centered_pos = pos - nose_center

        # 2. Scale by Torso Shoulder-to-Shoulder Width
        l_sh = pos[:, :, self.l_shoulder_idx, :]  # [B, T, 3]
        r_sh = pos[:, :, self.r_shoulder_idx, :]  # [B, T, 3]
        torso_width = torch.norm(l_sh - r_sh, dim=-1, keepdim=True).unsqueeze(-1).clamp(min=1e-3)  # [B, T, 1, 1]

        norm_pos = centered_pos / torso_width

        # Re-assemble kinematics tensor
        if C > 3:
            kin_rest = landmarks[..., 3:] / torso_width  # Also scale velocities and accelerations
            return torch.cat([norm_pos, kin_rest], dim=-1)
        return norm_pos

    def compute_adversarial_signer_loss(
        self,
        h_cls: torch.Tensor,
        signer_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes domain adversarial loss to remove signer identity from latent embeddings.
        h_cls: [B, d_model]
        signer_labels: [B]
        """
        # Apply Gradient Reversal
        reversed_h = GradientReversalFunction.apply(h_cls, self.grl_alpha)
        signer_logits = self.signer_discriminator(reversed_h)  # [B, num_signers]

        loss_adv = F.cross_entropy(signer_logits, signer_labels)

        with torch.no_grad():
            preds = torch.argmax(signer_logits, dim=-1)
            acc = (preds == signer_labels).float().mean().item() * 100.0

        return {
            "loss_signer_adv": loss_adv,
            "signer_logits": signer_logits,
            "signer_accuracy": acc,
        }
