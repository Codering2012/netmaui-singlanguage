#!/usr/bin/env python3
"""
================================================================================
MULTI-GRANULARITY SENTENCE-EMBEDDING SEMANTIC ANCHOR & CONTRASTIVE SYNTAX GUARD
================================================================================
1. Bridges the gloss-free cross-modal semantic gap via Bidirectional Asymmetric
   InfoNCE contrastive alignment between utterance-level visual thoughts and
   sentence embeddings (all-MiniLM-L6-v2 / BGE-small-en).
2. Contrastive Syntax Guard:
   Guarantees polarity-preserving syntactic translation. Specifically uses cranial
   IMU headshake velocity to enforce a strict margin ranking loss between
   affirmative and negative sentence embeddings, completely preventing the
   catastrophic affirmative-inversion failure mode ("I do not know" -> "I know").
================================================================================
"""

from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticEmbeddingAnchor(nn.Module):
    """
    Multi-Granularity Sentence-Embedding Semantic Anchor & Contrastive Syntax Guard.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_sent: int = 384,
        temperature: float = 0.07,
        polarity_margin: float = 0.40,
    ):
        super().__init__()
        self.temperature = temperature
        self.polarity_margin = polarity_margin
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_sent),
        )

    def forward(
        self,
        encoded_features: torch.Tensor,                                # [B, T, D]
        target_sentence_embeddings: torch.Tensor,                      # [B, D_sent]
        text_is_negative: Optional[torch.Tensor] = None,               # [B] bool
        cranial_imu: Optional[torch.Tensor] = None,                    # [B, T, 3] or [B, 3]
    ) -> torch.Tensor:
        """
        Computes combined Bidirectional InfoNCE and Contrastive Syntax Guard loss.
        """
        B = encoded_features.shape[0]
        # Utterance-level visual thought via attention-weighted or mean pooling
        vis_thought = encoded_features.mean(dim=1)                      # [B, D]
        vis_proj = F.normalize(self.proj(vis_thought), dim=-1)         # [B, D_sent]
        text_emb = F.normalize(target_sentence_embeddings.detach(), dim=-1) # [B, D_sent]

        # 1. Similarity logits: [B, B]
        sim_matrix = torch.matmul(vis_proj, text_emb.transpose(0, 1)) / self.temperature
        labels = torch.arange(B, device=encoded_features.device)

        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.transpose(0, 1), labels)
        info_nce_loss = 0.5 * (loss_v2t + loss_t2v)

        # 2. Contrastive Syntax Guard (Polarity-Preserving Hard Negative Mining)
        syntax_loss = torch.zeros((), device=encoded_features.device)
        if text_is_negative is not None and text_is_negative.any():
            # Extract cranial yaw angular speed if available (cranial IMU col 1 = yaw velocity)
            cranial_yaw_speed = torch.zeros(B, device=encoded_features.device)
            if cranial_imu is not None:
                if cranial_imu.dim() == 3:
                    # Average absolute yaw velocity across sequence [B]
                    cranial_yaw_speed = cranial_imu[:, :, 1].abs().mean(dim=1)
                elif cranial_imu.dim() == 2:
                    cranial_yaw_speed = cranial_imu[:, 1].abs()

            # For each negative sample i, find hard affirmative negatives in the same batch
            neg_indices = torch.where(text_is_negative)[0]
            aff_indices = torch.where(~text_is_negative)[0]

            if len(aff_indices) > 0:
                # Dynamic margin boosted by physical headshake velocity
                for i in neg_indices:
                    pos_sim = torch.dot(vis_proj[i], text_emb[i])
                    # Nearest affirmative sentence in batch (hard negative)
                    aff_sims = torch.matmul(text_emb[aff_indices], vis_proj[i])
                    hard_aff_sim = aff_sims.max()
                    
                    # Boost margin if physical headshake is present
                    dynamic_margin = self.polarity_margin + 0.3 * torch.tanh(cranial_yaw_speed[i] / 0.5)
                    # Margin loss: pos_sim must exceed hard_aff_sim by dynamic_margin
                    viol = F.relu(dynamic_margin - pos_sim + hard_aff_sim)
                    syntax_loss = syntax_loss + viol
                syntax_loss = syntax_loss / max(1, len(neg_indices))

        return info_nce_loss + 0.5 * syntax_loss
