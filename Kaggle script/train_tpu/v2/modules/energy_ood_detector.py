#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — ENERGY-BASED OOD & OOV DETECTION ENGINE (ENERGY-EBM)
================================================================================
Implements Helmholtz Free Energy OOD Scoring & Energy Margin Regularization:
1. Free Energy Thermodynamic Scoring:
     E(x; T) = - T * LogSumExp(logits(x) / T)
     - In-Distribution (ID) valid signs: Sharp logit peaks -> Low Energy (E << 0)
     - Out-of-Distribution (OOD) noise / non-signs: Flat logits -> High Energy (E >> 0)
2. Energy Margin Loss Regularization:
     L_energy = E_in [ max(0, E(x_in) - m_in)^2 ] + lambda_noise * E_noise [ max(0, m_out - E(x_noise))^2 ]
     Pushes valid sign distributions into low-energy wells while repelling noise.
3. Post-Hoc OOD & Non-Sign Gesture Rejection:
     Replaces overconfident Softmax normalization with provable energy gating:
     Filters non-signing frames (resting hands, scratching, noise) before CTC / AR decoding.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyOODOutput(NamedTuple):
    energy: torch.Tensor                # [B, T] or [B] Helmholtz free energy scores
    ood_mask: torch.Tensor              # [B, T] or [B] Boolean mask of rejected OOD frames/samples
    filtered_logits: torch.Tensor       # [B, T, V] Logits with OOD frames suppressed/masked
    energy_loss: torch.Tensor           # Scalar energy margin regularization loss
    in_distribution_score: torch.Tensor # [B, T] Probability of valid in-distribution sign


class ASLEnergyOODDetector(nn.Module):
    """
    Helmholtz Free Energy Out-of-Distribution (OOD) Detector & Margin Regularizer.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        margin_in: float = -12.0,      # Desired energy upper bound for valid signs
        margin_out: float = -3.0,      # Desired energy lower bound for noise/OOD
        lambda_noise: float = 0.50,
        ood_threshold: float = -6.0,   # Decisions: E > ood_threshold => OOD rejected
    ):
        super().__init__()
        self.temperature = temperature
        self.margin_in = margin_in
        self.margin_out = margin_out
        self.lambda_noise = lambda_noise
        self.ood_threshold = ood_threshold

    def compute_energy(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes Helmholtz free energy: E(x) = - T * LogSumExp(logits / T).
        logits: [..., V]
        Returns: [...]
        """
        # Numerically stabilized LogSumExp
        return - self.temperature * torch.logsumexp(logits / self.temperature, dim=-1)

    def compute_energy_loss(
        self,
        in_logits: torch.Tensor,
        noise_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes two-sided energy margin regularization loss.
        """
        energy_in = self.compute_energy(in_logits)  # [B, T] or [B]
        loss_in = F.relu(energy_in - self.margin_in).pow(2).mean()

        if noise_logits is not None:
            energy_noise = self.compute_energy(noise_logits)
            loss_out = F.relu(self.margin_out - energy_noise).pow(2).mean()
            return loss_in + self.lambda_noise * loss_out
        else:
            return loss_in

    def forward(
        self,
        logits: torch.Tensor,                         # [B, T, V] or [B, V]
        noise_logits: Optional[torch.Tensor] = None,  # [B_noise, T, V] optional synthetic noise
    ) -> EnergyOODOutput:
        """
        Computes energy scores, OOD rejection masks, and energy regularization loss.
        """
        device = logits.device

        # 1. Compute Free Energy
        energy = self.compute_energy(logits)  # [B, T] or [B]

        # 2. Compute Energy Margin Loss
        energy_loss = self.compute_energy_loss(logits, noise_logits)

        # 3. Detect Out-of-Distribution (OOD) Frames (Energy > Threshold)
        ood_mask = (energy > self.ood_threshold)  # [B, T] or [B]

        # 4. In-Distribution Confidence Score (Sigmoid mapped from negative energy)
        # Higher score (near 1.0) = High confidence valid sign
        id_score = torch.sigmoid(-(energy - self.ood_threshold))

        # 5. Suppress / Mask OOD frames in logits
        filtered_logits = logits.clone()
        if logits.dim() == 3:
            # Set blank token or uniform low value for OOD frames
            ood_expanded = ood_mask.unsqueeze(-1).expand_as(logits)
            filtered_logits = torch.where(ood_expanded, torch.full_like(logits, -100.0), logits)
            # Blank token (index 0) preserved as active during non-signing holds
            filtered_logits[ood_mask, 0] = 0.0

        return EnergyOODOutput(
            energy=energy,
            ood_mask=ood_mask,
            filtered_logits=filtered_logits,
            energy_loss=energy_loss,
            in_distribution_score=id_score,
        )
