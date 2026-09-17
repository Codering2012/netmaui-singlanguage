#!/usr/bin/env python3
"""
================================================================================
DYNAMIC PHONOLOGICAL HOLD-CONDENSATION POOLING (D-PCP)
================================================================================
Solves the "Cross-Attention Entropy & Frame Dilution" bottleneck in continuous SLT:
- Continuous video spans T = 128 to 384 frames at 30 fps.
- Target English translation spans only L = 15 to 35 BPE tokens.
- Standard cross-attention diffuses weight (~0.004) across non-linguistic transition
  frames (movement epenthesis), triggering token hallucination.

D-PCP leverages the Flash & Hogan (1985) Minimum-Jerk kinematic hold salience:
    s_t = 1.0 - beta_t in [0, 1]
where s_t -> 1 at phonetic holds and s_t -> 0 at ballistic transition strokes.
It condenses the sequence into exactly N_condensed = 64 dense lexical sign tokens
via energy-weighted soft anchor assignment with a local Gaussian window prior.
================================================================================
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .movement_epenthesis_suppressor import MovementEpenthesisSuppressor


class DynamicPhonologicalCondenser(nn.Module):
    r"""
    Condenses continuous T-frame Conformer features into N_condensed dense semantic sign tokens.
    Guarantees strict TPU v5e tile alignment while suppressing epenthesis transition noise.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_condensed: int = 64,
        num_keypoints: int = 60,
        in_channels: int = 9,
        hold_weight: float = 1.0,
        positional_sharpness: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_condensed = n_condensed
        self.hold_weight = hold_weight
        self.positional_sharpness = positional_sharpness

        # Epenthesis detector for computing kinematic hold salience s_t = 1 - beta_t
        self.epenthesis_detector = MovementEpenthesisSuppressor(
            in_channels=in_channels,
            num_keypoints=num_keypoints,
            d_model=d_model,
        )

        # Learned anchor query prototypes for the condensed sequence
        self.anchor_queries = nn.Parameter(torch.randn(n_condensed, d_model) * (1.0 / math.sqrt(d_model)))

        # Multi-head projection layers
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def compute_hold_salience(
        self,
        kinematics: torch.Tensor,
        beta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        Computes kinematic hold salience s_t in [0, 1].
        s_t is high when hands are in a stable hold configuration (low speed, high finger stability).
        """
        if beta_t is None:
            # Flatten kinematics if needed
            if kinematics.dim() == 4:
                B, T = kinematics.shape[:2]
                kinematics = kinematics.view(B, T, -1)
            beta_t = self.epenthesis_detector.compute_epenthesis_probability(kinematics)
        # Hold salience is inverse of transition probability
        s_t = torch.clamp(1.0 - beta_t, min=1e-4, max=1.0)
        return s_t

    def forward(
        self,
        h: torch.Tensor,                                      # [B, T, d_model]
        kinematics: Optional[torch.Tensor] = None,            # [B, T, K * C]
        beta_t: Optional[torch.Tensor] = None,                # [B, T]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Condenses [B, T, d_model] -> [B, n_condensed, d_model]

        Returns:
            h_condensed: [B, n_condensed, d_model]
            s_t: [B, T] hold salience weights
            assign_weights: [B, n_condensed, T] soft assignment matrix
        """
        B, T, D = h.shape
        N = self.n_condensed

        # 1. Compute kinematic hold salience
        if kinematics is not None or beta_t is not None:
            s_t = self.compute_hold_salience(kinematics, beta_t)  # [B, T]
        else:
            s_t = torch.ones((B, T), device=h.device, dtype=h.dtype)

        # 2. Project Keys, Values from input sequence, and Queries from anchors
        # Anchors: [N, D] -> [B, N, D]
        queries = self.q_proj(self.anchor_queries).unsqueeze(0).expand(B, -1, -1)  # [B, N, D]
        keys = self.k_proj(h)                                                       # [B, T, D]
        values = self.v_proj(h)                                                     # [B, T, D]

        # 3. Scaled dot-product attention scores
        # [B, N, T]
        scores = torch.bmm(queries, keys.transpose(1, 2)) * (1.0 / math.sqrt(D))

        # 4. Inject Kinematic Hold Salience Bias
        # Boost attention to frames where hands are holding lexical sign configurations
        # s_t in [0, 1] -> log(s_t) in [-inf, 0]
        hold_bias = self.hold_weight * torch.log(s_t).unsqueeze(1)  # [B, 1, T]
        scores = scores + hold_bias

        # 5. Inject Local Temporal Gaussian Prior
        # Anchor n expects content around center t_n = (n + 0.5) / N * T
        anchor_indices = (torch.arange(N, device=h.device, dtype=torch.float32) + 0.5) / float(N)  # [N]
        frame_indices = (torch.arange(T, device=h.device, dtype=torch.float32) + 0.5) / float(T)    # [T]
        # Normalized temporal distance squared [N, T]
        dist_sq = ((anchor_indices.unsqueeze(1) - frame_indices.unsqueeze(0)) * float(N)) ** 2
        # Penalize distant frames beyond local receptive field
        temporal_prior = -0.5 * self.positional_sharpness * dist_sq.unsqueeze(0)  # [B, N, T]
        scores = scores + temporal_prior

        # 6. Soft assignment weights across time T
        assign_weights = F.softmax(scores, dim=-1)  # [B, N, T]

        # 7. Aggregate into condensed representations
        condensed = torch.bmm(assign_weights, values)  # [B, N, D]
        h_condensed = self.norm(self.out_proj(condensed) + queries)

        return h_condensed, s_t, assign_weights
