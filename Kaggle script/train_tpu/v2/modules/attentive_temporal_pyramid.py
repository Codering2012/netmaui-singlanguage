#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — ATTENTIVE MULTI-SCALE TEMPORAL PYRAMID (APN / MSTP-SLT)
================================================================================
Implements Attentive Multi-Scale Temporal Pyramid Network & Hierarchical CTC:
1. Multi-Scale Temporal Striding:
     - Scale 1 (1x): Full resolution (rapid fingerspelling, transient micro-motions)
     - Scale 2 (2x): Half resolution (stride-2 Conv1D sign stroke units)
     - Scale 3 (4x): Quarter resolution (stride-4 Conv1D phrase & syntactic rhythm)
2. Attentive Cross-Scale Temporal Gating:
     Computes dynamic frame-by-frame scale selection weights alpha_t in Delta^2:
     H_fused(t) = sum_{s=1}^3 alpha_{t, s} * H_s_interpolated(t)
3. Length-Safe Hierarchical Multi-Scale CTC Supervision:
     Computes auxiliary CTC supervision across all 3 temporal resolutions with strict
     length validation (T_s >= L_target) to eliminate blank collapse and overflow.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MSTPOutput(NamedTuple):
    fused_features: torch.Tensor         # [B, T, d_model]
    fused_logits: torch.Tensor           # [B, T, vocab_size]
    gate_weights: torch.Tensor           # [B, T, 3] Attention weights per scale
    total_loss: torch.Tensor             # Combined hierarchical CTC loss
    fused_ctc_loss: torch.Tensor         # Scale 1 fused CTC loss
    scale2_ctc_loss: torch.Tensor        # Scale 2 auxiliary CTC loss
    scale3_ctc_loss: torch.Tensor        # Scale 3 auxiliary CTC loss


class AttentiveTemporalPyramidEngine(nn.Module):
    """
    Attentive Multi-Scale Temporal Pyramid Network (APN / MSTP-SLT).
    """

    def __init__(
        self,
        d_model: int = 128,
        vocab_size: int = 100,
        blank_idx: int = 0,
        weight_fused: float = 1.0,
        weight_scale2: float = 0.50,
        weight_scale3: float = 0.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.blank_idx = blank_idx
        self.w_fused = weight_fused
        self.w_s2 = weight_scale2
        self.w_s3 = weight_scale3

        # Scale 2: 2x temporal subsampling (Conv1D k=3, s=2)
        self.conv_scale2 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Scale 3: 4x temporal subsampling (Conv1D k=3, s=4)
        self.conv_scale3 = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=5, stride=4, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Attentive Scale Gating MLP: takes concatenated [H1, H2_up, H3_up] -> 3 weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 3),
        )

        # Output CTC Heads
        self.head_fused = nn.Linear(d_model, vocab_size)
        self.head_scale2 = nn.Linear(d_model, vocab_size)
        self.head_scale3 = nn.Linear(d_model, vocab_size)

        self.ctc_loss_fn = nn.CTCLoss(blank=blank_idx, reduction="none", zero_infinity=True)

    def forward(
        self,
        h_seq: torch.Tensor,                                # [B, T, d_model]
        targets: Optional[torch.Tensor] = None,             # [B, L]
        input_lengths: Optional[torch.Tensor] = None,       # [B]
        target_lengths: Optional[torch.Tensor] = None,      # [B]
    ) -> MSTPOutput:
        """
        Executes Attentive Multi-Scale Temporal Pyramid forward and hierarchical CTC losses.
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        # Scale 1: Full resolution [B, T, D]
        h1 = h_seq

        # Scale 2: 2x downsampled [B, T/2, D]
        h1_t = h1.transpose(1, 2)  # [B, D, T]
        h2 = self.conv_scale2(h1_t).transpose(1, 2)  # [B, T_s2, D]
        T_s2 = h2.shape[1]

        # Scale 3: 4x downsampled [B, T/4, D]
        h3 = self.conv_scale3(h1_t).transpose(1, 2)  # [B, T_s3, D]
        T_s3 = h3.shape[1]

        # Upsample H2 and H3 to length T via 1D linear interpolation
        h2_up = F.interpolate(h2.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)  # [B, T, D]
        h3_up = F.interpolate(h3.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)  # [B, T, D]

        # Compute dynamic scale gating weights: [B, T, 3]
        concat_scales = torch.cat([h1, h2_up, h3_up], dim=-1)  # [B, T, 3*D]
        gate_logits = self.gate_mlp(concat_scales)             # [B, T, 3]
        gate_weights = F.softmax(gate_logits, dim=-1)          # [B, T, 3]

        # Attentive weighted fusion
        alpha1 = gate_weights[..., 0:1]  # [B, T, 1]
        alpha2 = gate_weights[..., 1:2]  # [B, T, 1]
        alpha3 = gate_weights[..., 2:3]  # [B, T, 1]

        h_fused = alpha1 * h1 + alpha2 * h2_up + alpha3 * h3_up  # [B, T, D]
        logits_fused = self.head_fused(h_fused)  # [B, T, V]

        if targets is not None and input_lengths is not None and target_lengths is not None:
            # 1. Primary Fused CTC Loss (Scale 1)
            log_probs_fused = F.log_softmax(logits_fused, dim=-1).transpose(0, 1)  # [T, B, V]
            loss_fused = self.ctc_loss_fn(log_probs_fused, targets, input_lengths, target_lengths).mean()

            # 2. Auxiliary Scale 2 CTC Loss
            logits_s2 = self.head_scale2(h2)  # [B, T_s2, V]
            log_probs_s2 = F.log_softmax(logits_s2, dim=-1).transpose(0, 1)  # [T_s2, B, V]
            input_lens_s2 = (input_lengths // 2).clamp(min=1)

            valid_s2 = input_lens_s2 >= target_lengths
            if valid_s2.any():
                loss_s2_raw = self.ctc_loss_fn(
                    log_probs_s2[:, valid_s2],
                    targets[valid_s2],
                    input_lens_s2[valid_s2],
                    target_lengths[valid_s2],
                )
                loss_s2 = loss_s2_raw.mean()
            else:
                loss_s2 = torch.tensor(0.0, device=device)

            # 3. Auxiliary Scale 3 CTC Loss
            logits_s3 = self.head_scale3(h3)  # [B, T_s3, V]
            log_probs_s3 = F.log_softmax(logits_s3, dim=-1).transpose(0, 1)  # [T_s3, B, V]
            input_lens_s3 = (input_lengths // 4).clamp(min=1)

            valid_s3 = input_lens_s3 >= target_lengths
            if valid_s3.any():
                loss_s3_raw = self.ctc_loss_fn(
                    log_probs_s3[:, valid_s3],
                    targets[valid_s3],
                    input_lens_s3[valid_s3],
                    target_lengths[valid_s3],
                )
                loss_s3 = loss_s3_raw.mean()
            else:
                loss_s3 = torch.tensor(0.0, device=device)

            # Total Hierarchical Loss
            total_loss = (
                self.w_fused * loss_fused +
                self.w_s2 * loss_s2 +
                self.w_s3 * loss_s3
            )
        else:
            loss_fused = torch.tensor(0.0, device=device)
            loss_s2 = torch.tensor(0.0, device=device)
            loss_s3 = torch.tensor(0.0, device=device)
            total_loss = torch.tensor(0.0, device=device)

        return MSTPOutput(
            fused_features=h_fused,
            fused_logits=logits_fused,
            gate_weights=gate_weights,
            total_loss=total_loss,
            fused_ctc_loss=loss_fused,
            scale2_ctc_loss=loss_s2,
            scale3_ctc_loss=loss_s3,
        )
