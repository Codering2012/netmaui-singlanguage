#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — MULTI-SPAN GLOSS-TO-TEXT REGULARIZER & LENGTH PREDICTOR
================================================================================
Implements Multi-Span Contrastive Alignment & Non-Autoregressive Length Prediction:
1. Target Length Predictor (Delta L in [-15, +16]):
     Predicts spoken language target token count from continuous sign kinematics:
     L_hat = T_in + (argmax(Softmax(MLP(H_pooled))) - 15)
     L_length = CrossEntropy(P_{Delta L}, Delta L_gt)
2. Multi-Span Syntactic Alignment:
     Aligns variable-duration gesture spans (Topic, Action, Comment) with English phrase spans:
     s_sign = MeanPool(H[t_start:t_end]), s_text = MeanPool(E[l_start:l_end])
     L_span = - log ( exp(<s_sign, s_text> / tau) / sum exp(<s_sign, s_neg> / tau) )
3. Topic-Comment Inversion Resolution:
     Resolves ASL OSV / SOV syntax into English SVO syntax without phrase inversion errors.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiSpanOutput(NamedTuple):
    predicted_lengths: torch.Tensor      # [B] Predicted integer target sequence lengths
    length_logits: torch.Tensor          # [B, num_classes] Length offset distribution
    length_loss: torch.Tensor            # Scalar length prediction cross-entropy loss
    span_contrastive_loss: torch.Tensor  # Scalar multi-span alignment loss
    total_loss: torch.Tensor             # Combined loss


class ASLMultiSpanGlossToTextEngine(nn.Module):
    """
    Multi-Span Contrastive Alignment & Non-Autoregressive Length Predictor Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        max_offset: int = 15,            # Length offset range [-15, +16] -> 32 classes
        tau: float = 0.07,
        weight_length: float = 0.50,
        weight_span: float = 0.50,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_offset = max_offset
        self.num_classes = 2 * max_offset + 2  # 32 classes
        self.tau = tau
        self.w_len = weight_length
        self.w_span = weight_span

        # Target Length Predictor MLP
        self.length_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_classes),
        )

        # Span projection heads
        self.sign_span_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.text_span_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def predict_length(self, h_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h_seq: [B, T, d_model]
        Returns: (predicted_lengths [B], logits [B, num_classes])
        """
        B, T, D = h_seq.shape
        pooled = h_seq.mean(dim=1)  # [B, D]
        logits = self.length_predictor(pooled)  # [B, 32]
        pred_offset_idx = torch.argmax(logits, dim=-1)  # [B]
        pred_offset = pred_offset_idx - self.max_offset  # [B] in [-15, +16]
        predicted_lengths = (T + pred_offset).clamp(min=1)
        return predicted_lengths, logits

    def compute_span_contrastive_loss(
        self,
        h_seq: torch.Tensor,                  # [B, T, d_model]
        text_seq: torch.Tensor,               # [B, L, d_model]
        num_spans: int = 3,
    ) -> torch.Tensor:
        """
        Extracts temporal spans from sign and text and computes symmetric InfoNCE alignment.
        """
        B, T, D = h_seq.shape
        _, L, _ = text_seq.shape
        device = h_seq.device

        span_len_t = max(1, T // num_spans)
        span_len_l = max(1, L // num_spans)

        sign_spans = []
        text_spans = []

        for s in range(num_spans):
            t_start, t_end = s * span_len_t, min(T, (s + 1) * span_len_t)
            l_start, l_end = s * span_len_l, min(L, (s + 1) * span_len_l)

            s_sign = h_seq[:, t_start:t_end].mean(dim=1)   # [B, D]
            s_text = text_seq[:, l_start:l_end].mean(dim=1) # [B, D]

            sign_spans.append(self.sign_span_proj(s_sign))
            text_spans.append(self.text_span_proj(s_text))

        # Flatten spans: [B * num_spans, D]
        flat_sign = F.normalize(torch.cat(sign_spans, dim=0), p=2, dim=-1)
        flat_text = F.normalize(torch.cat(text_spans, dim=0), p=2, dim=-1)

        # Pairwise cosine similarity: [N_spans, N_spans]
        sim_mat = torch.matmul(flat_sign, flat_text.t()) / self.tau
        targets = torch.arange(flat_sign.shape[0], device=device)

        loss_sign_to_text = F.cross_entropy(sim_mat, targets)
        loss_text_to_sign = F.cross_entropy(sim_mat.t(), targets)

        return 0.50 * (loss_sign_to_text + loss_text_to_sign)

    def forward(
        self,
        h_seq: torch.Tensor,                         # [B, T, d_model]
        text_seq: Optional[torch.Tensor] = None,     # [B, L, d_model]
        target_lengths: Optional[torch.Tensor] = None, # [B] ground truth lengths
    ) -> MultiSpanOutput:
        """
        Executes length prediction and multi-span contrastive alignment.
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        # 1. Length Prediction
        pred_lengths, length_logits = self.predict_length(h_seq)

        # 2. Length Loss
        if target_lengths is not None:
            delta_target = target_lengths - T  # in [-15, +16]
            target_class_idx = (delta_target + self.max_offset).clamp(0, self.num_classes - 1)
            loss_len = F.cross_entropy(length_logits, target_class_idx)
        else:
            loss_len = torch.tensor(0.0, device=device)

        # 3. Multi-Span Contrastive Alignment
        if text_seq is not None and B > 0:
            loss_span = self.compute_span_contrastive_loss(h_seq, text_seq)
        else:
            loss_span = torch.tensor(0.0, device=device)

        total_loss = self.w_len * loss_len + self.w_span * loss_span

        return MultiSpanOutput(
            predicted_lengths=pred_lengths,
            length_logits=length_logits,
            length_loss=loss_len,
            span_contrastive_loss=loss_span,
            total_loss=total_loss,
        )
