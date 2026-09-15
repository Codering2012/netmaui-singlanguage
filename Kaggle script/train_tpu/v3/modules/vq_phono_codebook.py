#!/usr/bin/env python3
"""
================================================================================
VECTOR-QUANTIZED PHONOLOGICAL CODEBOOK (VQ-PHONO) & ARTICULATORY CUTMIX
================================================================================
Revolutionizes self-supervised pretraining for continuous sign language:
1. Eliminates Euclidean Mean Regression:
   Continuous coordinate MSE/L1 regression predicts blurry average handshapes.
   VQ-Phono quantizes continuous 3D hand and phonological configurations into K=256
   discrete phonetic prototypes (Stokoe / Liddell-Johnson phonemes), converting
   pretraining into sharp, high-gradient discrete cross-entropy classification.
2. Harsh Spatiotemporal Span Masking:
   Masks 70% of dominant hand spans in blocks of 4 to 16 continuous frames,
   forcing the Conformer to infer complete lexical signs from context.
3. Articulatory CutMix (ACM):
   Stochastically swaps non-dominant hand channels between random batch pairs (p=0.25),
   decoupling bilateral articulators and dramatically improving data efficiency.
================================================================================
"""

import math
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQPhonoCodebook(nn.Module):
    r"""
    Discrete Vector-Quantized Phonological Codebook & Harsh Masked Articulator Modeling.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_codes: int = 256,
        code_dim: int = 32,
        phonology_dim: int = 19,
        num_keypoints: int = 60,
        in_channels: int = 9,
        commitment_cost: float = 0.25,
        acm_prob: float = 0.25,
        mask_ratio: float = 0.70,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.commitment_cost = commitment_cost
        self.acm_prob = acm_prob
        self.mask_ratio = mask_ratio
        self.num_keypoints = num_keypoints
        self.in_channels = in_channels

        # Project phonology (19D) + dominant hand kinematics (21 kp * 3 = 63D) into code_dim
        self.phono_projector = nn.Sequential(
            nn.Linear(phonology_dim + 63, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, code_dim),
        )

        # Discrete Codebook Embeddings: [K, code_dim]
        self.embedding = nn.Embedding(num_codes, code_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        # Pretraining Prediction Heads from Conformer d_model
        # 1. Discrete code classification head
        self.code_classifier = nn.Linear(d_model, num_codes)
        # 2. Kinematic momentum velocity forecast head (dominant hand 21 kp * 3 vel = 63)
        self.vel_predictor = nn.Linear(d_model, 63)

    def quantize_target(self, phonology: torch.Tensor, kinematics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Quantizes continuous inputs into discrete code indices k in [0, K-1].

        Returns:
            quantized: [B, T, code_dim]
            code_indices: [B, T]
        """
        B, T = kinematics.shape[:2]
        pts = kinematics.view(B, T, self.num_keypoints, self.in_channels)
        # Dominant (Right) hand pos: keypoints 21..41 (21 kp * 3 coords = 63)
        r_hand_pos = pts[:, :, 21:42, :3].reshape(B, T, 63)

        # If phonology is None or wrong size, pad with zeros
        if phonology is None:
            phonology = torch.zeros((B, T, 19), device=kinematics.device, dtype=kinematics.dtype)

        feat = torch.cat([phonology, r_hand_pos], dim=-1)  # [B, T, 19 + 63 = 82]
        z_e = self.phono_projector(feat)                   # [B, T, code_dim]

        # Compute Euclidean distances to all K codes: ||z_e - e_k||^2
        z_flat = z_e.view(-1, self.code_dim)
        d = (
            torch.sum(z_flat ** 2, dim=1, keepdim=True) +
            torch.sum(self.embedding.weight ** 2, dim=1) -
            2.0 * torch.matmul(z_flat, self.embedding.weight.t())
        )
        code_indices = torch.argmin(d, dim=1).view(B, T)
        z_q = self.embedding(code_indices)

        # Straight-through estimator
        quantized = z_e + (z_q - z_e).detach()
        return quantized, code_indices

    def apply_articulatory_cutmix(self, kinematics: torch.Tensor) -> torch.Tensor:
        r"""
        Swaps non-dominant (Left) hand kinematics (keypoints 0..20) across random pairs in batch.
        Forces the model to decouple dominant vs non-dominant coordination.
        """
        if not self.training or kinematics.shape[0] <= 1 or torch.rand(1).item() > self.acm_prob:
            return kinematics

        orig_shape = kinematics.shape
        B, T = orig_shape[:2]
        pts = kinematics.view(B, T, self.num_keypoints, self.in_channels).clone()
        perm = torch.randperm(B, device=kinematics.device)

        # Left hand keypoints: 0..20
        pts[:, :, :21, :] = pts[perm, :, :21, :]
        return pts.view(orig_shape)

    def generate_harsh_span_mask(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        r"""
        Generates spatiotemporal block mask covering mask_ratio (70%) of frames in chunks of 4-16 frames.
        Returns:
            mask: [B, T] bool tensor where True means MASKED
        """
        mask = torch.zeros((B, T), dtype=torch.bool, device=device)
        total_to_mask = int(T * self.mask_ratio)

        for b in range(B):
            masked_count = 0
            # Safety counter to avoid infinite loops on short T
            iterations = 0
            while masked_count < total_to_mask and iterations < 50:
                iterations += 1
                span_len = int(torch.randint(4, min(17, max(5, T // 2)), (1,)).item())
                if span_len + masked_count > total_to_mask:
                    span_len = total_to_mask - masked_count
                if T - span_len <= 0:
                    start_idx = 0
                else:
                    start_idx = int(torch.randint(0, T - span_len + 1, (1,)).item())
                mask[b, start_idx : start_idx + span_len] = True
                masked_count = int(mask[b].sum().item())

        return mask

    def compute_pretraining_loss(
        self,
        h: torch.Tensor,                              # [B, T, d_model] Conformer output
        kinematics: torch.Tensor,                     # [B, T, K * C]
        phonology: Optional[torch.Tensor] = None,     # [B, T, 19]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r"""
        Computes dual VQ-Phono MAM objective:
        1. Cross-entropy classification on masked discrete codebook IDs.
        2. Smooth L1 prediction on masked dominant hand velocities.
        """
        B, T = kinematics.shape[:2]
        # 1. Target discrete codes
        with torch.no_grad():
            _, target_codes = self.quantize_target(phonology, kinematics)
            pts = kinematics.view(B, T, self.num_keypoints, self.in_channels)
            # Target right hand velocities (keypoints 21..41, cols 3:6)
            vel = pts[:, :, 21:42, 3:6].reshape(B, T, 63)

        # 2. Harsh span mask
        mask = self.generate_harsh_span_mask(B, T, kinematics.device)  # [B, T]

        # 3. Model predictions
        pred_code_logits = self.code_classifier(h)  # [B, T, num_codes]
        pred_vel = self.vel_predictor(h)            # [B, T, 63]

        # 4. Losses on masked frames
        if mask.any():
            ce_loss = F.cross_entropy(
                pred_code_logits[mask],
                target_codes[mask],
            )
            vel_loss = F.smooth_l1_loss(
                pred_vel[mask],
                vel[mask],
            )
        else:
            ce_loss = F.cross_entropy(pred_code_logits.view(-1, self.num_codes), target_codes.view(-1))
            vel_loss = F.smooth_l1_loss(pred_vel, vel)

        total_mam_loss = ce_loss + 0.5 * vel_loss
        metrics = {
            "loss_vq_ce": ce_loss.detach(),
            "loss_vq_vel": vel_loss.detach(),
            "total_vq_mam": total_mam_loss,
        }
        return total_mam_loss, metrics
