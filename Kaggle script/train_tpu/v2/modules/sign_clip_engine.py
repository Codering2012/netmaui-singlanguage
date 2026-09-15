#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SIGNCLIP CROSS-MODAL RETRIEVAL & ZERO-SHOT ENGINE
================================================================================
Implements SignCLIP (Contrastive Kinematics-Text Pretraining):
1. Dual L2-Normalized Projections:
     v = VisualProj(h_cls) / ||VisualProj(h_cls)||_2
     t = TextProj(h_text)   / ||TextProj(h_text)||_2
2. Symmetric InfoNCE Cross-Entropy:
     L_CLIP = 0.5 * ( CrossEntropy(S / tau, y) + CrossEntropy(S^T / tau, y) )
     S_ij = v_i . t_j
3. Zero-Shot Classification & Fast Dense Retrieval (Video <-> Text).
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrievalResult(NamedTuple):
    query_idx: int
    top_matched_indices: List[int]
    top_matched_scores: List[float]


class ASLSignCLIPEngine(nn.Module):
    """
    SignCLIP Dual-Encoder Contrastive Learning & Cross-Modal Retrieval Engine.
    """

    def __init__(
        self,
        visual_model: nn.Module,
        embed_dim: int = 256,
        init_temperature: float = 0.07,
        max_temperature: float = 100.0,
    ):
        super().__init__()
        self.visual_model = visual_model
        self.embed_dim = embed_dim

        d_enc = getattr(self.visual_model, "d_enc", 128)
        self.vis_proj = nn.Sequential(
            nn.Linear(d_enc, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Lightweight Text Encoder for Gloss / English prompts
        self.text_emb = nn.Embedding(500, embed_dim, padding_idx=0)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=2,
        )
        self.text_proj = nn.Linear(embed_dim, embed_dim)

        # Learnable logit scale (temperature parameter tau)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temperature))
        self.max_temperature = max_temperature

    def encode_visual(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes visual kinematics into L2-normalized unit vectors [B, embed_dim].
        """
        out = self.visual_model(input_x=features, mask=mask, frame_indices=frame_indices)
        h_cls = out["h_cls"]  # [B, d_enc]
        vis_feats = self.vis_proj(h_cls)
        return F.normalize(vis_feats, p=2, dim=-1)

    def encode_text(
        self,
        text_ids: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encodes text tokens into L2-normalized unit vectors [B, embed_dim].
        """
        x = self.text_emb(text_ids)
        src_key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        h = self.text_encoder(x, src_key_padding_mask=src_key_padding_mask)
        pooled = h.mean(dim=1)  # Mean pooling over token sequence
        text_feats = self.text_proj(pooled)
        return F.normalize(text_feats, p=2, dim=-1)

    def forward(
        self,
        features: torch.Tensor,
        text_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes symmetric SignCLIP InfoNCE loss for a batch of paired (video, text).
        """
        v_emb = self.encode_visual(features, mask=mask, frame_indices=frame_indices)  # [B, D]
        t_emb = self.encode_text(text_ids, text_mask=text_mask)                       # [B, D]

        logit_scale = self.logit_scale.exp().clamp(max=self.max_temperature)
        logits_per_video = logit_scale * torch.matmul(v_emb, t_emb.t())  # [B, B]
        logits_per_text = logits_per_video.t()                            # [B, B]

        B = features.size(0)
        labels = torch.arange(B, device=features.device)

        loss_v2t = F.cross_entropy(logits_per_video, labels)
        loss_t2v = F.cross_entropy(logits_per_text, labels)
        total_clip_loss = 0.5 * (loss_v2t + loss_t2v)

        # In-batch retrieval accuracy
        with torch.no_grad():
            pred_v2t = torch.argmax(logits_per_video, dim=-1)
            v2t_acc = (pred_v2t == labels).float().mean().item() * 100.0

        return {
            "loss": total_clip_loss,
            "loss_v2t": loss_v2t,
            "loss_t2v": loss_t2v,
            "v2t_accuracy": v2t_acc,
            "temperature": (1.0 / logit_scale).item(),
            "similarity_matrix": logits_per_video / logit_scale,
        }

    @torch.no_grad()
    def zero_shot_classify(
        self,
        features: torch.Tensor,
        candidate_text_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classifies video against candidate prompt texts via cosine similarity.
        Returns: (predicted_class_indices, softmax_probabilities)
        """
        v_emb = self.encode_visual(features, mask=mask, frame_indices=frame_indices)  # [B, D]
        cand_t_emb = self.encode_text(candidate_text_ids)                             # [Num_Classes, D]

        logit_scale = self.logit_scale.exp().clamp(max=self.max_temperature)
        logits = logit_scale * torch.matmul(v_emb, cand_t_emb.t())                    # [B, Num_Classes]
        probs = F.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)

        return preds, probs
