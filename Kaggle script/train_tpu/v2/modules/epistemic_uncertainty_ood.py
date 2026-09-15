#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — EPISTEMIC UNCERTAINTY & OOD DETECTION ENGINE
================================================================================
Implements Monte Carlo (MC) Dropout & Energy-Based Out-of-Distribution Detection:
1. Test-Time MC Dropout Sampling (Gal & Ghahramani):
     P_mean = 1/M * sum_{m=1}^M P^(m)(y | X)
     Epistemic Uncertainty = Predictive_Entropy(P_mean) - Mean_Entropy(P^(m))
2. Multi-Modal Free Energy OOD Score:
     E(X) = -T_0 * log sum_c exp( z_c / T_0 )
3. Safe Abstention & Hallucination Suppression:
     Flags out-of-domain gestures, heavy occlusions, or corrupt camera feeds.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyOODReport(NamedTuple):
    mean_probabilities: torch.Tensor    # [B, V]
    epistemic_uncertainty: torch.Tensor # [B] (Mutual Information)
    predictive_entropy: torch.Tensor    # [B] (Total Uncertainty)
    energy_score: torch.Tensor          # [B] (OOD Metric)
    is_ood: torch.Tensor                # [B] Boolean flag
    confidence: torch.Tensor            # [B] Calibrated confidence score


class ASLEpistemicUncertaintyOODDetector:
    """
    Monte Carlo Dropout Uncertainty and Out-of-Distribution Detector.
    """

    def __init__(
        self,
        model: nn.Module,
        num_mc_samples: int = 10,
        energy_temperature: float = 1.0,
        epistemic_threshold: float = 0.45,
        energy_threshold: float = 2.50,
    ):
        self.model = model
        self.num_mc_samples = max(2, num_mc_samples)
        self.energy_temperature = energy_temperature
        self.epistemic_threshold = epistemic_threshold
        self.energy_threshold = energy_threshold

    def enable_mc_dropout(self):
        """
        Forces all Dropout layers into training mode for stochastic sampling.
        """
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
                module.train()

    @torch.no_grad()
    def estimate_uncertainty_and_ood(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> UncertaintyOODReport:
        """
        Performs M stochastic forward passes and computes epistemic uncertainty and OOD status.
        features: [B, T, K, C]
        """
        self.model.eval()
        self.enable_mc_dropout()

        B = features.size(0)
        mc_probs = []
        mc_entropies = []
        mc_energies = []

        for _ in range(self.num_mc_samples):
            out = self.model(input_x=features, mask=mask, frame_indices=frame_indices)
            aux_logits = out["aux_logits"]  # [B, vocab_size]

            probs_m = F.softmax(aux_logits, dim=-1)  # [B, V]
            entropy_m = -torch.sum(probs_m * torch.log(probs_m.clamp(min=1e-8)), dim=-1)  # [B]

            # Energy score: E(x) = -T * logsumexp(logits / T)
            energy_m = -self.energy_temperature * torch.logsumexp(aux_logits / self.energy_temperature, dim=-1)  # [B]

            mc_probs.append(probs_m)
            mc_entropies.append(entropy_m)
            mc_energies.append(energy_m)

        # Stack over MC samples [M, B, V]
        stacked_probs = torch.stack(mc_probs, dim=0)
        mean_probs = stacked_probs.mean(dim=0)  # [B, V]

        # Total Predictive Entropy
        total_entropy = -torch.sum(mean_probs * torch.log(mean_probs.clamp(min=1e-8)), dim=-1)  # [B]

        # Expected Aleatoric Entropy
        stacked_entropies = torch.stack(mc_entropies, dim=0)  # [M, B]
        expected_entropy = stacked_entropies.mean(dim=0)      # [B]

        # Epistemic Uncertainty (Mutual Information = Total - Aleatoric)
        epistemic_mi = (total_entropy - expected_entropy).clamp(min=0.0)  # [B]

        # Average Energy Score
        mean_energy = torch.stack(mc_energies, dim=0).mean(dim=0)  # [B]

        # Calibrated Confidence
        max_prob, _ = torch.max(mean_probs, dim=-1)
        calibrated_conf = (max_prob * torch.exp(-epistemic_mi)).clamp(0.0, 1.0)  # [B]

        # OOD Flag: High epistemic uncertainty OR high free energy
        is_ood = (epistemic_mi > self.epistemic_threshold) | (mean_energy > self.energy_threshold)

        return UncertaintyOODReport(
            mean_probabilities=mean_probs,
            epistemic_uncertainty=epistemic_mi,
            predictive_entropy=total_entropy,
            energy_score=mean_energy,
            is_ood=is_ood,
            confidence=calibrated_conf,
        )
