#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — MULTI-MODAL HALLUCINATION DETECTOR & GROUNDING GUARD
================================================================================
Detects and mitigates ungrounded visual-to-text hallucinations:
1. Visual Faithfulness Sensitivity: Measures KL divergence between video-conditioned
   logits and unconditioned (masked-video) language priors:
     S_vis(y_t) = KL( P(y_t | X, y_<t) || P(y_t | 0, y_<t) )
2. Per-Token Hallucination Risk Index: Flags tokens generated purely from LM priors.
3. Hallucination Suppression: Penalizes ungrounded tokens during decoding.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenGroundingReport(NamedTuple):
    token_id: int
    token_str: Optional[str]
    confidence: float
    visual_sensitivity: float
    hallucination_risk: float  # 0.0 (fully grounded) to 1.0 (pure hallucination)


class ASLHallucinationDetector:
    """
    Evaluates visual grounding and detects hallucinations in generated sign translations.
    """

    def __init__(
        self,
        model: nn.Module,
        grounding_threshold: float = 0.5,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model = model
        self.grounding_threshold = grounding_threshold
        self.device = torch.device(device)
        self.model.eval()

    @torch.no_grad()
    def audit_translation(
        self,
        features: torch.Tensor,
        generated_tokens: List[int],
        vocab_list: Optional[List[str]] = None,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Audits a generated token sequence for visual faithfulness.
        """
        B = features.size(0)
        assert B == 1, "Hallucination detector operates on single sequence."

        features = features.to(self.device)
        T_len = len(generated_tokens)
        if T_len <= 1:
            return {"token_reports": [], "sentence_hallucination_score": 0.0, "is_hallucinated": False}

        # 1. Video-Conditioned Forward Pass
        cand_tensor = torch.tensor([generated_tokens + [0]], dtype=torch.long, device=self.device)
        out_vis = self.model(
            input_x=features,
            mask=mask,
            frame_indices=frame_indices,
            gloss_seq=cand_tensor,
        )
        logits_vis = out_vis["dec_logits"][0, :T_len]  # [T_len, V]
        probs_vis = F.softmax(logits_vis, dim=-1)

        # 2. Counterfactual (Zero-Video / Unconditioned) Forward Pass
        zero_features = torch.zeros_like(features)
        out_zero = self.model(
            input_x=zero_features,
            mask=mask,
            frame_indices=frame_indices,
            gloss_seq=cand_tensor,
        )
        logits_zero = out_zero["dec_logits"][0, :T_len]  # [T_len, V]
        probs_zero = F.softmax(logits_zero, dim=-1)

        token_reports: List[TokenGroundingReport] = []
        hallucination_scores: List[float] = []

        for t in range(T_len):
            tok_id = generated_tokens[t]
            tok_str = vocab_list[tok_id] if vocab_list and tok_id < len(vocab_list) else None

            p_v = probs_vis[t]
            p_z = probs_zero[t]

            # Compute KL Divergence: KL(P_vis || P_zero)
            kl_div = F.kl_div(torch.log(p_z.clamp(min=1e-8)), p_v, reduction="sum").item()
            conf = p_v[tok_id].item()

            # Hallucination risk: high if visual sensitivity is near 0
            # risk in [0, 1]
            risk = math.exp(-max(0.0, kl_div))

            token_reports.append(
                TokenGroundingReport(
                    token_id=tok_id,
                    token_str=tok_str,
                    confidence=conf,
                    visual_sensitivity=kl_div,
                    hallucination_risk=risk,
                )
            )
            hallucination_scores.append(risk)

        sentence_score = sum(hallucination_scores) / max(1, len(hallucination_scores))
        is_hallucinated = sentence_score > (1.0 - self.grounding_threshold)

        return {
            "token_reports": token_reports,
            "sentence_hallucination_score": sentence_score,
            "is_hallucinated": is_hallucinated,
        }
