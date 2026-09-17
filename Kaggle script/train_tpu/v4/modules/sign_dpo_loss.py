#!/usr/bin/env python3
"""
================================================================================
ASL V4: DIRECT PREFERENCE OPTIMIZATION (SIGN-DPO) LOSS
================================================================================
Aligns Sign Language Translation against hallucinated and polarity-inverted text.
Given:
- Visual Prefix x (from sign encoder)
- Winning translation y_w (ground truth English sentence)
- Losing translation y_l (adversarial hallucinated / polarity-inverted English sentence)

Mathematical Formulation (Rafailov et al., NeurIPS 2023):
L_DPO = -E [ log sigma( beta * log(pi_theta(y_w|x) / pi_ref(y_w|x))
                      - beta * log(pi_theta(y_l|x) / pi_ref(y_l|x)) ) ]
================================================================================
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignDPOLoss(nn.Module):
    """
    Direct Preference Optimization loss specialized for continuous sign language translation.
    """

    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def _get_batch_logps(
        self,
        logits: torch.Tensor,                                       # [B, L, V]
        labels: torch.Tensor,                                       # [B, L]
        pad_mask: Optional[torch.Tensor] = None,                    # [B, L]
    ) -> torch.Tensor:
        """
        Computes sum of token log-probabilities over valid positions.
        """
        log_probs = F.log_softmax(logits, dim=-1)
        safe_labels = labels.clamp(min=0, max=logits.shape[-1] - 1)
        token_logps = log_probs.gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)  # [B, L]

        if pad_mask is not None:
            token_logps = token_logps * pad_mask.float()
            denom = pad_mask.float().sum(dim=-1).clamp(min=1.0)
        else:
            denom = float(token_logps.shape[-1])

        return token_logps.sum(dim=-1) / denom  # [B]

    def forward(
        self,
        policy_chosen_logits: torch.Tensor,                         # [B, L_w, V]
        policy_rejected_logits: torch.Tensor,                       # [B, L_l, V]
        chosen_labels: torch.Tensor,                                # [B, L_w]
        rejected_labels: torch.Tensor,                              # [B, L_l]
        ref_chosen_logits: Optional[torch.Tensor] = None,           # [B, L_w, V] (detached reference)
        ref_rejected_logits: Optional[torch.Tensor] = None,         # [B, L_l, V] (detached reference)
        chosen_mask: Optional[torch.Tensor] = None,
        rejected_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Computes Sign-DPO loss and implicit reward metrics.
        """
        # 1. Evaluate log-probabilities under policy
        pi_logps_chosen = self._get_batch_logps(policy_chosen_logits, chosen_labels, chosen_mask)
        pi_logps_rejected = self._get_batch_logps(policy_rejected_logits, rejected_labels, rejected_mask)

        # 2. Evaluate under reference model (if None, assume uniform / detached initial policy)
        if ref_chosen_logits is not None and ref_rejected_logits is not None:
            with torch.no_grad():
                ref_logps_chosen = self._get_batch_logps(ref_chosen_logits, chosen_labels, chosen_mask)
                ref_logps_rejected = self._get_batch_logps(ref_rejected_logits, rejected_labels, rejected_mask)
        else:
            ref_logps_chosen = pi_logps_chosen.detach()
            ref_logps_rejected = pi_logps_rejected.detach()

        # 3. Log-ratios
        log_ratio_chosen = pi_logps_chosen - ref_logps_chosen
        log_ratio_rejected = pi_logps_rejected - ref_logps_rejected

        logits = self.beta * (log_ratio_chosen - log_ratio_rejected)

        # DPO loss with optional label smoothing
        if self.label_smoothing > 0:
            loss = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            ).mean()
        else:
            loss = -F.logsigmoid(logits).mean()

        # Metrics
        with torch.no_grad():
            implicit_reward_chosen = self.beta * log_ratio_chosen
            implicit_reward_rejected = self.beta * log_ratio_rejected
            reward_margin = (implicit_reward_chosen - implicit_reward_rejected).mean()
            accuracy = (logits > 0).float().mean()

        metrics = {
            "dpo_loss": loss.detach(),
            "reward_margin": reward_margin.detach(),
            "preference_accuracy": accuracy.detach(),
        }

        return loss, metrics
