#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — CONFORMAL UNCERTAINTY & PREDICTION SETS (CONFORMALSIGN)
================================================================================
Implements Token-Level Split Conformal Prediction (CP-SLT / ConformalSign):
1. Finite-Sample Distribution-Free Coverage Guarantee:
     P( y_t* in C_alpha(x_t) ) >= 1 - alpha   (e.g., alpha = 0.10 -> 90% coverage)
2. Adaptive Prediction Sets (APS) Nonconformity Scoring:
     Sort probabilities descending: p_{(1)} >= p_{(2)} >= ... >= p_{(V)}
     Score s_i = sum_{j=1}^{r_i} p_{(j)} + U * p_{(r_i)}   (where r_i is rank of y_i*)
3. Conformal Quantile Estimation:
     q_hat = Quantile( {s_1, ..., s_N}, ceil((N + 1)(1 - alpha)) / N )
4. Dynamic Prediction Set Inference:
     C_alpha(x_t) = { y in V | CumulativeProb(y) <= q_hat }
     Yields |C_alpha| = 1 on unambiguous signs and multi-token sets on ambiguous signs.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformalOutput(NamedTuple):
    prediction_sets_mask: torch.Tensor   # [B, T, V] Boolean mask of tokens in conformal set
    set_sizes: torch.Tensor              # [B, T] Number of candidate tokens in C_alpha
    point_predictions: torch.Tensor      # [B, T] Argmax greedy point predictions
    conformal_threshold: float           # Calibrated quantile threshold q_hat
    mean_set_size: float                 # Average prediction set cardinality |C_alpha|
    coverage_rate: Optional[float]       # Empirical test coverage if targets provided


class ASLConformalUncertaintyEngine(nn.Module):
    """
    Token-Level Split Conformal Prediction Engine for ASL Sequences.
    """

    def __init__(
        self,
        alpha: float = 0.10,          # Error rate (1 - alpha = 0.90 coverage guarantee)
        score_type: str = "aps",      # "aps" (Adaptive Prediction Sets) or "margin"
    ):
        super().__init__()
        self.alpha = alpha
        self.score_type = score_type

        # Register buffer for calibrated quantile threshold q_hat
        self.register_buffer("q_hat", torch.tensor(0.95, dtype=torch.float32))
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))

    def compute_nonconformity_scores(
        self,
        probs: torch.Tensor,     # [N, V] Probability distributions
        targets: torch.Tensor,   # [N] Ground truth token indices
    ) -> torch.Tensor:
        """
        Computes nonconformity scores for calibration examples.
        """
        N, V = probs.shape
        device = probs.device

        # Sort probabilities descending
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        # Cumulative probability mass
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # [N, V]

        # Find rank of ground-truth token
        # targets: [N, 1]
        targets_exp = targets.unsqueeze(-1)  # [N, 1]
        # Match sorted_indices == targets_exp
        ranks = (sorted_indices == targets_exp).nonzero()[:, 1]  # [N] rank index (0-indexed)

        if self.score_type == "aps":
            # APS Score: cumulative sum up to target token rank
            scores = cumsum_probs[torch.arange(N, device=device), ranks]
        else:
            # Margin score: 1.0 - prob(target)
            target_p = probs[torch.arange(N, device=device), targets]
            scores = 1.0 - target_p

        return scores  # [N]

    def calibrate(
        self,
        calibration_logits: torch.Tensor,   # [N, V] or [B, T, V] Logits on held-out calibration set
        calibration_targets: torch.Tensor,  # [N] or [B, T] Ground-truth targets
    ) -> float:
        """
        Calibrates the conformal quantile threshold q_hat on held-out validation data.
        """
        # Flatten sequence dimensions if needed
        if calibration_logits.dim() == 3:
            B, T, V = calibration_logits.shape
            cal_logits = calibration_logits.view(B * T, V)
            cal_targets = calibration_targets.view(B * T)
        else:
            cal_logits = calibration_logits
            cal_targets = calibration_targets

        probs = F.softmax(cal_logits, dim=-1)
        N = cal_targets.size(0)

        # 1. Compute Nonconformity Scores
        scores = self.compute_nonconformity_scores(probs, cal_targets)

        # 2. Conformal Quantile: ceil((N + 1) * (1 - alpha)) / N
        level = math.ceil((N + 1) * (1.0 - self.alpha)) / float(N)
        level = min(max(level, 0.0), 1.0)

        q_val = torch.quantile(scores, q=level, interpolation="higher").item()
        self.q_hat.fill_(q_val)
        self.is_calibrated.fill_(True)

        return q_val

    def predict_sets(
        self,
        logits: torch.Tensor,                       # [B, T, V] or [N, V] Inference logits
        targets: Optional[torch.Tensor] = None,     # Optional targets to evaluate empirical coverage
    ) -> ConformalOutput:
        """
        Constructs dynamic conformal prediction sets with mathematical coverage guarantees.
        """
        orig_shape = logits.shape
        is_seq = (logits.dim() == 3)
        if is_seq:
            B, T, V = orig_shape
            flat_logits = logits.view(B * T, V)
        else:
            flat_logits = logits
            B, T, V = flat_logits.size(0), 1, flat_logits.size(1)

        probs = F.softmax(flat_logits, dim=-1)  # [B*T, V]
        N = probs.size(0)
        device = probs.device

        # Sort probabilities descending
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)  # [B*T, V]

        q_thresh = self.q_hat.item()

        if self.score_type == "aps":
            # Include tokens until cumulative probability exceeds q_hat
            # (Shift cumsum by sorted_probs to include the boundary token)
            cumsum_prev = cumsum_probs - sorted_probs
            included_sorted = (cumsum_prev < q_thresh)
            # Guarantee at least 1 token is included (the top-1 token)
            included_sorted[:, 0] = True
        else:
            # Threshold probability >= 1.0 - q_hat
            included_sorted = (sorted_probs >= (1.0 - q_thresh))
            included_sorted[:, 0] = True

        # Scatter back to original token vocabulary space [B*T, V]
        pred_mask_flat = torch.zeros_like(probs, dtype=torch.bool)
        pred_mask_flat.scatter_(dim=-1, index=sorted_indices, src=included_sorted)

        set_sizes_flat = pred_mask_flat.sum(dim=-1).float()  # [B*T]
        point_preds_flat = sorted_indices[:, 0]              # [B*T] Top-1

        # Reshape back to original dimensions
        if is_seq:
            pred_mask = pred_mask_flat.view(B, T, V)
            set_sizes = set_sizes_flat.view(B, T)
            point_preds = point_preds_flat.view(B, T)
        else:
            pred_mask = pred_mask_flat
            set_sizes = set_sizes_flat
            point_preds = point_preds_flat

        mean_size = set_sizes_flat.mean().item()

        # Compute empirical coverage if targets provided
        emp_coverage = None
        if targets is not None:
            if is_seq:
                flat_targets = targets.view(B * T)
            else:
                flat_targets = targets
            covered = pred_mask_flat[torch.arange(N, device=device), flat_targets]
            emp_coverage = covered.float().mean().item()

        return ConformalOutput(
            prediction_sets_mask=pred_mask,
            set_sizes=set_sizes,
            point_predictions=point_preds,
            conformal_threshold=q_thresh,
            mean_set_size=mean_size,
            coverage_rate=emp_coverage,
        )
