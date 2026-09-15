#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — CONFIDENCE CALIBRATION & TEMPERATURE SCALING
================================================================================
Implements post-hoc probability calibration (Guo et al.) and reliability metrics:
1. Expected Calibration Error (ECE) & Maximum Calibration Error (MCE)
2. L-BFGS Temperature Scaling Optimization for NLL minimization
3. Calibrated inference for CTC recognition and Autoregressive sequence generation
================================================================================
"""

from typing import Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ASLConfidenceCalibrator(nn.Module):
    """
    Confidence calibrator for ASL Foundation Models.
    Applies learned scalar temperature T to soften overconfident logits.
    """

    def __init__(self, initial_temperature: float = 1.0, num_bins: int = 15):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature, dtype=torch.float32))
        self.num_bins = num_bins

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Scales logits by temperature parameter T."""
        temp = self.temperature.clamp(min=0.01, max=10.0)
        return logits / temp

    @torch.no_grad()
    def compute_ece_mce(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        logits: [N, C] unnormalized logits
        targets: [N] ground-truth class labels
        """
        scaled_logits = self.forward(logits)
        softmaxes = F.softmax(scaled_logits, dim=-1)
        confidences, predictions = torch.max(softmaxes, dim=-1)
        accuracies = predictions.eq(targets)

        bin_boundaries = torch.linspace(0, 1, self.num_bins + 1, device=logits.device)
        ece = 0.0
        mce = 0.0
        n_total = float(len(targets))

        for i in range(self.num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean().item()

            if in_bin.sum() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean().item()
                avg_confidence_in_bin = confidences[in_bin].mean().item()
                diff = abs(accuracy_in_bin - avg_confidence_in_bin)
                ece += diff * (in_bin.sum().item() / n_total)
                mce = max(mce, diff)

        return ece, mce

    def fit_temperature(
        self,
        val_logits: torch.Tensor,
        val_targets: torch.Tensor,
        max_iter: int = 50,
        lr: float = 0.01,
    ) -> Dict[str, float]:
        """
        Optimizes temperature T using L-BFGS to minimize NLL on validation logits.
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        ece_before, mce_before = self.compute_ece_mce(val_logits, val_targets)

        def eval_step():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_targets)
            loss.backward()
            return loss

        optimizer.step(eval_step)

        ece_after, mce_after = self.compute_ece_mce(val_logits, val_targets)

        return {
            "temperature": self.temperature.item(),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "mce_before": mce_before,
            "mce_after": mce_after,
        }
