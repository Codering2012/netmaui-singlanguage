#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — CONTINUOUS GESTURE BOUNDARY SEGMENTER & BIO TAGGER
================================================================================
Performs temporal segmentation and boundary detection over continuous video:
1. Predicts frame-level BIO states:
     - 0: Outside / Pause / Movement Epenthesis (O)
     - 1: Begin Sign Gesture (B)
     - 2: Inside Active Sign (I)
2. Segments continuous streams into discrete Sign Spans: [t_start, t_end, gloss, confidence]
3. Eliminates transitional noise and movement artifacts between signs
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignSpan(NamedTuple):
    start_frame: int
    end_frame: int
    duration: int
    gloss_token: int
    confidence: float


class ASLGestureBoundarySegmenter(nn.Module):
    """
    Temporal boundary segmenter for Continuous Sign Language Recognition.
    """

    def __init__(self, d_model: int = 128, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        padding = kernel_size // 2

        # Temporal Context Convolution
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 3-Class BIO Classifier: 0=Outside, 1=Begin, 2=Inside
        self.bio_head = nn.Linear(d_model, 3)

    def forward(self, encoder_features: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        encoder_features: [B, T, D]
        Returns bio_logits: [B, T, 3]
        """
        feat_t = encoder_features.transpose(1, 2)  # [B, D, T]
        conv_out = self.conv(feat_t).transpose(1, 2)  # [B, T, D]
        h = self.dropout(self.act(self.norm(conv_out)))

        bio_logits = self.bio_head(h)  # [B, T, 3]
        if mask is not None:
            bio_logits = bio_logits * mask.unsqueeze(-1).to(bio_logits.dtype)
        return bio_logits

    @torch.no_grad()
    def segment_sequence(
        self,
        encoder_features: torch.Tensor,
        ctc_log_probs: torch.Tensor,
        min_duration: int = 3,
        threshold_b: float = 0.4,
    ) -> List[List[SignSpan]]:
        """
        Extracts structured SignSpan intervals from continuous sequence representations.
        """
        self.eval()
        B, T, D = encoder_features.shape
        bio_logits = self.forward(encoder_features)
        bio_probs = F.softmax(bio_logits, dim=-1)  # [B, T, 3]

        ctc_tokens = torch.argmax(ctc_log_probs, dim=-1)  # [B, T]
        ctc_confidences = torch.exp(torch.max(ctc_log_probs, dim=-1)[0])  # [B, T]

        batch_spans: List[List[SignSpan]] = []

        for b in range(B):
            spans: List[SignSpan] = []
            in_sign = False
            cur_start = 0
            cur_tokens = []
            cur_confs = []

            for t in range(T):
                p_o = bio_probs[b, t, 0].item()
                p_b = bio_probs[b, t, 1].item()
                p_i = bio_probs[b, t, 2].item()
                state = int(torch.argmax(bio_probs[b, t]).item())

                tok = int(ctc_tokens[b, t].item())
                conf = float(ctc_confidences[b, t].item())

                # Begin state or Transition into sign
                if (state == 1 or p_b >= threshold_b) and not in_sign:
                    in_sign = True
                    cur_start = t
                    cur_tokens = [tok] if tok != 0 else []
                    cur_confs = [conf]
                elif in_sign and (state == 0 and p_o > 0.6):
                    # End of active sign
                    duration = t - cur_start
                    if duration >= min_duration and len(cur_tokens) > 0:
                        # Most frequent non-blank token
                        best_tok = max(set(cur_tokens), key=cur_tokens.count)
                        avg_conf = sum(cur_confs) / max(1, len(cur_confs))
                        spans.append(SignSpan(cur_start, t - 1, duration, best_tok, avg_conf))
                    in_sign = False
                    cur_tokens.clear()
                    cur_confs.clear()
                elif in_sign:
                    if tok != 0:
                        cur_tokens.append(tok)
                    cur_confs.append(conf)

            # Flush final active span
            if in_sign and (T - cur_start) >= min_duration and len(cur_tokens) > 0:
                best_tok = max(set(cur_tokens), key=cur_tokens.count)
                avg_conf = sum(cur_confs) / max(1, len(cur_confs))
                spans.append(SignSpan(cur_start, T - 1, T - cur_start, best_tok, avg_conf))

            batch_spans.append(spans)

        return batch_spans
