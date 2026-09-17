#!/usr/bin/env python3
"""
================================================================================
ASL V4: HIERARCHICAL PROSODIC GRAMMAR SCOPE PREDICTOR
================================================================================
Models continuous syntactic prosody in ASL:
- Wh-questions (furrowed brows, backward head tilt)
- Yes/No questions (raised brows, forward head tilt)
- Topicalization / Conditionals (raised brows, head tilt, pause)
- Negation scope (headshake span across verbal predicates)
- Clause boundary / Prosodic pause detection

Enforces piecewise-smooth temporal scopes using Total Variation (TV) regularization
and outputs a clause-boundary prior matrix for the Sinkhorn syntactic transducer.
================================================================================
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProsodicGrammarScopePredictor(nn.Module):
    """
    Predicts multi-tier prosodic grammar scopes and clause boundaries in continuous ASL.
    """

    NUM_PROSODIC_TIERS = 5
    TIER_NAMES = ["WH_QUESTION", "YN_QUESTION", "TOPICALIZATION", "NEGATION", "CLAUSE_BOUNDARY"]

    def __init__(self, d_model: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.d_model = d_model

        # Dilated 1D temporal convolution stack to capture multi-second syntactic prosody spans
        self.scope_conv = nn.Sequential(
            nn.Conv1d(d_model, hidden_dim, kernel_size=5, padding=2, dilation=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=4, dilation=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=8, dilation=4),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, self.NUM_PROSODIC_TIERS, kernel_size=1),
        )

        self.proj_out = nn.Linear(d_model + self.NUM_PROSODIC_TIERS, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,                                # [B, T, D]
        face_landmarks: Optional[torch.Tensor] = None,              # [B, T, 12, 3] or [B, T, 36]
        cranial_imu: Optional[torch.Tensor] = None,                 # [B, T, 3]
        target_clause_boundaries: Optional[torch.Tensor] = None,    # [B, T]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            enhanced_hidden: [B, T, D]
            losses: Dict containing TV smoothness and clause boundary loss
            scope_probs: [B, T, 5] Continuous grammatical scope activations
        """
        B, T, D = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Conv1d operates over [B, D, T]
        h_trans = hidden_states.transpose(1, 2)
        raw_logits = self.scope_conv(h_trans).transpose(1, 2)  # [B, T, 5]
        scope_probs = torch.sigmoid(raw_logits)               # [B, T, 5]

        # 1. Total Variation (TV) Regularization:
        # Grammatical prosody cannot flicker at 30Hz; it must be piecewise continuous
        tv_loss = torch.mean(torch.abs(scope_probs[:, 1:, :] - scope_probs[:, :-1, :]))

        # 2. Supervised Clause Boundary Loss (if annotations provided)
        clause_loss = torch.zeros((), device=device, dtype=dtype)
        if target_clause_boundaries is not None:
            pred_boundary = scope_probs[..., 4]  # CLAUSE_BOUNDARY channel
            clause_loss = F.binary_cross_entropy(pred_boundary, target_clause_boundaries.float())

        # 3. Construct Clause Affinity / Barrier Matrix for Sinkhorn Transducer
        # [B, T, 1] - [B, 1, T]: Cumulative boundary indicator prevents reordering across clauses
        clause_cum = torch.cumsum(scope_probs[..., 4], dim=-1)  # [B, T]
        clause_barrier = torch.abs(clause_cum.unsqueeze(2) - clause_cum.unsqueeze(1))  # [B, T, T]

        total_loss = tv_loss * 0.05 + clause_loss * 0.1

        # Residual integration into Conformer sequence
        fused = torch.cat([hidden_states, scope_probs], dim=-1)
        enhanced_hidden = self.norm(hidden_states + self.proj_out(fused))

        return enhanced_hidden, {
            "loss_prosodic_scope": total_loss,
            "prosodic_tv_loss": tv_loss.detach(),
        }, clause_barrier
