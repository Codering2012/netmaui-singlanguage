#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — VISUAL ALIGNMENT CONSTRAINT & SELF-MUTUAL KD (VAC-SMKD)
================================================================================
Implements Visual Alignment Constraint (VAC) & Self-Mutual Knowledge Distillation:
1. Visual Alignment Auxiliary CTC Supervision:
     Attaches an auxiliary CTC classifier directly to the visual feature extractor (h_vis)
     to prevent short-term spatial representations from being bypassed by contextual layers.
     L_ctc_vis = CTC( h_vis * W_vis, Y )
     L_ctc_ctx = CTC( h_ctx * W_ctx, Y )
2. Detached Bidirectional Self-Mutual Knowledge Distillation:
     Enforces mutual distribution alignment between short-term visual and long-term contextual
     temporal representations with detached teacher targets to prevent degenerate collapse:
     L_distill_v2c = KL( Softmax( Z_vis.detach() / tau ) || Softmax( Z_ctx / tau ) ) * tau^2
     L_distill_c2v = KL( Softmax( Z_ctx.detach() / tau ) || Softmax( Z_vis / tau ) ) * tau^2
3. Anti-Spike Temporal Probability Smoothing:
     Mitigates CTC feature saturation and sharp impulse spikes by regularizing consecutive
     frame probability transitions:
     L_smooth = (1 / (T-1)) * sum || P_t - P_{t+1} ||_2^2
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VACSMKDOutput(NamedTuple):
    ctx_ctc_loss: torch.Tensor           # Primary contextual CTC loss
    vis_ctc_loss: torch.Tensor           # Auxiliary visual CTC loss
    distill_v2c_loss: torch.Tensor       # Visual -> Contextual KD loss
    distill_c2v_loss: torch.Tensor       # Contextual -> Visual KD loss
    smooth_loss: torch.Tensor            # Temporal anti-spike smoothing loss
    total_loss: torch.Tensor             # Combined weighted objective
    ctx_logits: torch.Tensor             # [B, T, V] Contextual logits
    vis_logits: torch.Tensor             # [B, T, V] Visual logits


class ASLVACSMKDEngine(nn.Module):
    """
    Visual Alignment Constraint & Self-Mutual Knowledge Distillation Engine.
    """

    def __init__(
        self,
        vocab_size: int = 100,
        d_vis: int = 128,
        d_ctx: int = 128,
        blank_idx: int = 0,
        temperature: float = 2.0,
        lambda_vis: float = 1.0,
        lambda_distill: float = 0.50,
        lambda_smooth: float = 0.05,
        share_classifier: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_vis = d_vis
        self.d_ctx = d_ctx
        self.blank_idx = blank_idx
        self.temperature = temperature
        self.lambda_vis = lambda_vis
        self.lambda_distill = lambda_distill
        self.lambda_smooth = lambda_smooth
        self.share_classifier = share_classifier

        # Visual and Contextual Projection Heads
        self.vis_head = nn.Sequential(
            nn.Linear(d_vis, d_vis),
            nn.LayerNorm(d_vis),
            nn.GELU(),
            nn.Linear(d_vis, vocab_size),
        )

        if share_classifier and d_vis == d_ctx:
            self.ctx_head = self.vis_head
        else:
            self.ctx_head = nn.Sequential(
                nn.Linear(d_ctx, d_ctx),
                nn.LayerNorm(d_ctx),
                nn.GELU(),
                nn.Linear(d_ctx, vocab_size),
            )

        self.ctc_loss_fn = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)

    def compute_mutual_distillation(
        self,
        vis_logits: torch.Tensor,        # [B, T, V]
        ctx_logits: torch.Tensor,        # [B, T, V]
        input_lengths: Optional[torch.Tensor] = None,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes bidirectional KL divergence between visual and contextual distributions
        with detached teacher targets.
        """
        B, T, V = vis_logits.shape
        tau = self.temperature

        # Compute log-probs and detached target probs
        log_p_ctx = F.log_softmax(ctx_logits / tau, dim=-1)
        log_p_vis = F.log_softmax(vis_logits / tau, dim=-1)

        p_vis_target = F.softmax(vis_logits.detach() / tau, dim=-1)
        p_ctx_target = F.softmax(ctx_logits.detach() / tau, dim=-1)

        # KL(P_vis || P_ctx) and KL(P_ctx || P_vis)
        # kl_div(input=log_p, target=p, reduction='none') = p * (log(p) - log_p)
        kl_v2c = F.kl_div(log_p_ctx, p_vis_target, reduction="none").sum(dim=-1)  # [B, T]
        kl_c2v = F.kl_div(log_p_vis, p_ctx_target, reduction="none").sum(dim=-1)  # [B, T]

        if input_lengths is not None:
            # Mask out padding frames
            mask = torch.arange(T, device=vis_logits.device).unsqueeze(0) < input_lengths.unsqueeze(1)  # [B, T]
            kl_v2c = (kl_v2c * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
            kl_c2v = (kl_c2v * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        else:
            kl_v2c = kl_v2c.mean()
            kl_c2v = kl_c2v.mean()

        # Scale by tau^2 according to KD standard formulation (Hinton et al.)
        loss_v2c = kl_v2c * (tau ** 2)
        loss_c2v = kl_c2v * (tau ** 2)

        return loss_v2c, loss_c2v

    def compute_temporal_smoothing(
        self,
        logits: torch.Tensor,            # [B, T, V]
        input_lengths: Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:
        """
        Anti-spike frame transition regularization: minimizes || P_t - P_{t+1} ||_2^2.
        """
        probs = F.softmax(logits, dim=-1)  # [B, T, V]
        diff = probs[:, 1:] - probs[:, :-1]  # [B, T-1, V]
        sq_diff = diff.pow(2).sum(dim=-1)    # [B, T-1]

        if input_lengths is not None:
            T_valid = (input_lengths - 1).clamp(min=1)
            mask = torch.arange(diff.shape[1], device=logits.device).unsqueeze(0) < T_valid.unsqueeze(1)
            smooth_loss = (sq_diff * mask.float()).sum() / mask.float().sum().clamp(min=1.0)
        else:
            smooth_loss = sq_diff.mean()

        return smooth_loss

    def forward(
        self,
        h_vis: torch.Tensor,             # [B, T, d_vis]
        h_ctx: torch.Tensor,             # [B, T, d_ctx]
        targets: torch.Tensor,           # [B, L_max] or flattened 1D targets
        input_lengths: torch.Tensor,     # [B]
        target_lengths: torch.Tensor,    # [B]
    ) -> VACSMKDOutput:
        """
        Executes full Visual Alignment Constraint & Self-Mutual Distillation pipeline.
        """
        vis_logits = self.vis_head(h_vis)  # [B, T, V]
        ctx_logits = self.ctx_head(h_ctx)  # [B, T, V]

        # PyTorch CTCLoss expects inputs of shape [T, B, V] in log-probabilities
        log_probs_ctx = F.log_softmax(ctx_logits, dim=-1).transpose(0, 1)  # [T, B, V]
        log_probs_vis = F.log_softmax(vis_logits, dim=-1).transpose(0, 1)  # [T, B, V]

        # 1. CTC Losses
        ctx_ctc_loss = self.ctc_loss_fn(log_probs_ctx, targets, input_lengths, target_lengths)
        vis_ctc_loss = self.ctc_loss_fn(log_probs_vis, targets, input_lengths, target_lengths)

        # 2. Bidirectional Self-Mutual Knowledge Distillation
        distill_v2c, distill_c2v = self.compute_mutual_distillation(
            vis_logits=vis_logits,
            ctx_logits=ctx_logits,
            input_lengths=input_lengths,
        )

        # 3. Anti-Spike Smoothing Loss
        smooth_loss = 0.5 * (
            self.compute_temporal_smoothing(ctx_logits, input_lengths) +
            self.compute_temporal_smoothing(vis_logits, input_lengths)
        )

        # 4. Total Combined Loss
        total_loss = (
            ctx_ctc_loss +
            self.lambda_vis * vis_ctc_loss +
            self.lambda_distill * (distill_v2c + distill_c2v) +
            self.lambda_smooth * smooth_loss
        )

        return VACSMKDOutput(
            ctx_ctc_loss=ctx_ctc_loss,
            vis_ctc_loss=vis_ctc_loss,
            distill_v2c_loss=distill_v2c,
            distill_c2v_loss=distill_c2v,
            smooth_loss=smooth_loss,
            total_loss=total_loss,
            ctx_logits=ctx_logits,
            vis_logits=vis_logits,
        )
