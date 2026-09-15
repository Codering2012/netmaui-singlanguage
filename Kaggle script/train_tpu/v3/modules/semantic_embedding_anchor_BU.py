#!/usr/bin/env python3
"""
================================================================================
MULTI-GRANULARITY SENTENCE-EMBEDDING SEMANTIC ANCHOR (GLOSS-FREE BRIDGE)
================================================================================
Bridges the gloss-free cross-modal semantic gap via Bidirectional Asymmetric InfoNCE
contrastive alignment between utterance-level visual thoughts and text sentence embeddings.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticEmbeddingAnchor(nn.Module):
    """
    Multi-Granularity Sentence-Embedding Semantic Anchor.
    Bridges the gloss-free modality gap via Bidirectional InfoNCE contrastive alignment.
    """
    def __init__(self, d_model: int = 128, d_sent: int = 384, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_sent),
        )

    def forward(
        self,
        encoded_features: torch.Tensor,     # [B, T, D]
        target_sentence_embeddings: torch.Tensor, # [B, D_sent]
    ) -> torch.Tensor:
        """
        Computes bidirectional InfoNCE loss.
        """
        B = encoded_features.shape[0]
        # Global mean pool across time to form the utterance-level visual thought
        vis_thought = encoded_features.mean(dim=1) # [B, D]
        vis_proj = F.normalize(self.proj(vis_thought), dim=-1) # [B, D_sent]
        text_emb = F.normalize(target_sentence_embeddings.detach(), dim=-1) # [B, D_sent]

        # Similarity logits: [B, B]
        sim_matrix = torch.matmul(vis_proj, text_emb.transpose(0, 1)) / self.temperature
        labels = torch.arange(B, device=encoded_features.device)

        loss_v2t = F.cross_entropy(sim_matrix, labels)
        loss_t2v = F.cross_entropy(sim_matrix.transpose(0, 1), labels)
        return 0.5 * (loss_v2t + loss_t2v)
