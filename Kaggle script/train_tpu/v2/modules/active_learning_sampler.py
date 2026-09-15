#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — CONTINUOUS ACTIVE LEARNING & UNCERTAINTY SAMPLER
================================================================================
Implements Uncertainty-Aware Active Learning & Pseudo-Labeling (CUAL Framework):
1. Multi-Metric Uncertainty Scoring: Computes Normalized Predictive Entropy,
   Margin Difference, and Least Confidence over sequence predictions:
     H(X) = - 1/T sum_t sum_v P(y_t=v | X) log P(y_t=v | X)
2. Temporal Frame Ambiguity Localization: Identifies specific temporal frames
   with high visual ambiguity (occlusions, rapid fingerspelling, blurry transitions).
3. Dual-Queue Pool Partitioning:
   - High-Uncertainty Queue -> Human Oracle Review & Active Fine-Tuning.
   - Low-Uncertainty Queue  -> Self-Supervised Pseudo-Labeling.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameUncertaintyInfo(NamedTuple):
    frame_idx: int
    entropy: float
    margin: float
    least_confidence: float
    top_token: int
    runner_up_token: int


class SequenceUncertaintyReport(NamedTuple):
    mean_entropy: float
    mean_margin: float
    min_confidence: float
    composite_uncertainty_score: float  # Normalized in [0, 1]
    ambiguous_frame_indices: List[int]
    recommended_action: str  # "ORACLE_ANNOTATION", "PSEUDO_LABEL", "DISCARD"


class ASLActiveLearningSampler:
    """
    Evaluates prediction uncertainty across sign language sequences to prioritize
    active learning annotation and high-confidence pseudo-labeling.
    """

    def __init__(
        self,
        oracle_entropy_threshold: float = 1.20,
        pseudo_entropy_threshold: float = 0.35,
        margin_threshold: float = 0.25,
    ):
        self.oracle_entropy_threshold = oracle_entropy_threshold
        self.pseudo_entropy_threshold = pseudo_entropy_threshold
        self.margin_threshold = margin_threshold

    @torch.no_grad()
    def score_sequence(
        self,
        log_probs: torch.Tensor,
        top_k_ambiguous_frames: int = 5,
    ) -> SequenceUncertaintyReport:
        """
        Evaluates sequence uncertainty from model log-probabilities [T, V] or [B, T, V] (B=1).
        """
        if log_probs.dim() == 3:
            log_probs = log_probs[0]

        T, V = log_probs.shape
        probs = torch.exp(log_probs).clamp(min=1e-8, max=1.0)  # [T, V]

        # 1. Per-frame Shannon Entropy: - sum(p * log(p))
        entropies = -(probs * torch.log(probs)).sum(dim=-1)  # [T]
        norm_factor = math.log(max(2, V))
        norm_entropies = (entropies / norm_factor).tolist()

        # 2. Per-frame Margin: P(top1) - P(top2)
        top2_vals, top2_indices = torch.topk(probs, k=2, dim=-1)
        margins = (top2_vals[:, 0] - top2_vals[:, 1]).tolist()
        least_confs = (1.0 - top2_vals[:, 0]).tolist()

        frame_reports: List[FrameUncertaintyInfo] = []
        for t in range(T):
            frame_reports.append(
                FrameUncertaintyInfo(
                    frame_idx=t,
                    entropy=norm_entropies[t],
                    margin=margins[t],
                    least_confidence=least_confs[t],
                    top_token=top2_indices[t, 0].item(),
                    runner_up_token=top2_indices[t, 1].item(),
                )
            )

        # Rank frames by highest entropy / lowest margin
        ranked_frames = sorted(frame_reports, key=lambda f: f.entropy, reverse=True)
        ambiguous_frame_indices = [f.frame_idx for f in ranked_frames[:top_k_ambiguous_frames]]

        mean_ent = sum(norm_entropies) / max(1, T)
        mean_mar = sum(margins) / max(1, T)
        min_conf = min(top2_vals[:, 0].tolist())

        # Composite score: higher means more uncertain
        composite_score = (mean_ent + (1.0 - mean_mar) + (1.0 - min_conf)) / 3.0
        composite_score = max(0.0, min(1.0, composite_score))

        if mean_ent >= self.oracle_entropy_threshold / norm_factor or mean_mar <= self.margin_threshold:
            action = "ORACLE_ANNOTATION"
        elif mean_ent <= self.pseudo_entropy_threshold / norm_factor and mean_mar >= 0.70:
            action = "PSEUDO_LABEL"
        else:
            action = "NEUTRAL"

        return SequenceUncertaintyReport(
            mean_entropy=mean_ent,
            mean_margin=mean_mar,
            min_confidence=min_conf,
            composite_uncertainty_score=composite_score,
            ambiguous_frame_indices=ambiguous_frame_indices,
            recommended_action=action,
        )

    def prioritize_unlabeled_pool(
        self,
        pool_log_probs: List[torch.Tensor],
    ) -> Dict[str, List[int]]:
        """
        Sorts an unlabeled video pool into Oracle Queue and Pseudo-Label Queue.
        Returns dictionary of indices.
        """
        oracle_queue = []
        pseudo_queue = []
        scored_items = []

        for idx, lp in enumerate(pool_log_probs):
            rep = self.score_sequence(lp)
            scored_items.append((idx, rep))
            if rep.recommended_action == "ORACLE_ANNOTATION":
                oracle_queue.append((idx, rep.composite_uncertainty_score))
            elif rep.recommended_action == "PSEUDO_LABEL":
                pseudo_queue.append((idx, rep.composite_uncertainty_score))

        # Sort Oracle queue by highest uncertainty first
        oracle_queue.sort(key=lambda x: x[1], reverse=True)
        # Sort Pseudo queue by lowest uncertainty (highest confidence) first
        pseudo_queue.sort(key=lambda x: x[1])

        return {
            "oracle_indices": [x[0] for x in oracle_queue],
            "pseudo_label_indices": [x[0] for x in pseudo_queue],
        }
