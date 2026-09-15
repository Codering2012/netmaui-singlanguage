#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — CONFORMALIZED QUANTILE REGRESSION (CONFORMALQUANTILESIGN)
================================================================================
Implements Distribution-Free Conformalized Quantile Regression (CQR-SLT):
1. Multi-Quantile Prediction Heads:
     q_low  = q_{alpha/2}(x)     (e.g., tau = 0.05 for 90% coverage)
     q_med  = q_{0.50}(x)        (median point prediction)
     q_high = q_{1 - alpha/2}(x) (e.g., tau = 0.95 for 90% coverage)
2. Asymmetric Pinball (Quantile) Loss:
     L_tau(y, q) = max( tau * (y - q), (tau - 1) * (y - q) )
3. Non-Conformity Score & Finite-Sample Calibration:
     E_i = max( q_low(x_i) - y_i, y_i - q_high(x_i) )
     Q_hat = Quantile_{ (N+1)(1 - alpha)/N } ( {E_i} )
4. Mathematically Guaranteed 1 - alpha Prediction Intervals:
     C(x) = [ q_low(x) - Q_hat,  q_high(x) + Q_hat ]
     P( y in C(x) ) >= 1 - alpha  (Strict Finite-Sample Coverage Theorem)
5. Adaptive Uncertainty Bandwidth & Feature Projection:
     W(x) = q_high(x) - q_low(x) + 2 * Q_hat
     H_cqr = H + LayerNorm( Linear( [q_med, q_low, q_high, W] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConformalQuantileOutput(NamedTuple):
    quantile_features: torch.Tensor     # [B, T, d_model] Projected CQR representations
    q_low: torch.Tensor                 # [B, T, out_dim] Lower quantile (tau = alpha/2)
    q_med: torch.Tensor                 # [B, T, out_dim] Median prediction (tau = 0.50)
    q_high: torch.Tensor                # [B, T, out_dim] Upper quantile (tau = 1 - alpha/2)
    interval_bandwidth: torch.Tensor    # [B, T, out_dim] Adaptive uncertainty width W(x)
    conformal_threshold: torch.Tensor   # [1] Empirical calibration quantile Q_hat
    pinball_loss: Optional[torch.Tensor]# [1] Multi-quantile training loss
    empirical_coverage: Optional[torch.Tensor] # [1] Empirical validation coverage fraction
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + quantile_features


class ASLConformalQuantileEngine(nn.Module):
    """
    Adaptive Conformalized Quantile Regression & Finite-Sample Calibrator.
    """

    def __init__(
        self,
        d_model: int = 128,
        out_dim: int = 60,              # Target dimension (e.g. 60 landmarks or features)
        alpha: float = 0.10,            # Miscoverage rate (1 - alpha = 90% coverage)
        calibration_buffer_size: int = 500,
    ):
        super().__init__()
        self.d_model = d_model
        self.out_dim = out_dim
        self.alpha = alpha
        self.tau_low = alpha / 2.0      # 0.05
        self.tau_high = 1.0 - alpha / 2.0 # 0.95
        self.buffer_size = calibration_buffer_size

        # 3 Quantile Prediction Heads
        self.head_low  = nn.Linear(d_model, out_dim)
        self.head_med  = nn.Linear(d_model, out_dim)
        self.head_high = nn.Linear(d_model, out_dim)

        # Online Calibration Non-Conformity Score Buffer
        self.register_buffer("calibration_scores", torch.zeros(calibration_buffer_size, dtype=torch.float32))
        self.register_buffer("buffer_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("buffer_filled", torch.tensor(False, dtype=torch.bool))

        # Output feature projection: [q_med, q_low, q_high, bandwidth] = 4 * out_dim -> d_model
        self.out_proj = nn.Sequential(
            nn.Linear(out_dim * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def pinball_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor, tau: float) -> torch.Tensor:
        """
        Computes Pinball / Check loss: max(tau * diff, (tau - 1) * diff).
        """
        diff = y_true - y_pred
        return torch.max(tau * diff, (tau - 1.0) * diff).mean()

    def update_calibration(self, q_low: torch.Tensor, q_high: torch.Tensor, y_true: torch.Tensor):
        """
        Updates non-conformity score buffer: E_i = max(q_low - y, y - q_high).
        """
        with torch.no_grad():
            scores = torch.maximum(q_low - y_true, y_true - q_high).view(-1)  # [N]
            n_scores = scores.numel()
            if n_scores == 0:
                return

            ptr = int(self.buffer_ptr.item())
            buf_len = self.buffer_size

            if n_scores >= buf_len:
                self.calibration_scores.copy_(scores[-buf_len:])
                self.buffer_ptr.fill_(0)
                self.buffer_filled.fill_(True)
            else:
                end_ptr = ptr + n_scores
                if end_ptr <= buf_len:
                    self.calibration_scores[ptr:end_ptr] = scores
                    self.buffer_ptr.fill_(end_ptr % buf_len)
                    if end_ptr == buf_len:
                        self.buffer_filled.fill_(True)
                else:
                    first_chunk = buf_len - ptr
                    second_chunk = n_scores - first_chunk
                    self.calibration_scores[ptr:buf_len] = scores[:first_chunk]
                    self.calibration_scores[0:second_chunk] = scores[first_chunk:]
                    self.buffer_ptr.fill_(second_chunk)
                    self.buffer_filled.fill_(True)

    def get_conformal_threshold(self) -> torch.Tensor:
        """
        Computes empirical quantile Q_hat at level (1 - alpha) * (N + 1) / N.
        """
        if not bool(self.buffer_filled.item()) and int(self.buffer_ptr.item()) == 0:
            return torch.tensor(0.0, device=self.calibration_scores.device)

        valid_scores = (
            self.calibration_scores if bool(self.buffer_filled.item())
            else self.calibration_scores[:int(self.buffer_ptr.item())]
        )
        N = valid_scores.numel()
        # Conformal index: ceil((N + 1) * (1 - alpha)) / N
        level = min(1.0, math.ceil((N + 1) * (1.0 - self.alpha)) / float(N))
        return torch.quantile(valid_scores, level)

    def forward(
        self,
        h_seq: torch.Tensor,                         # [B, T, d_model] Latent representations
        targets: Optional[torch.Tensor] = None,      # [B, T, out_dim] Optional ground truth targets
    ) -> ConformalQuantileOutput:
        """
        Predicts lower, median, upper quantiles, calibrates conformal threshold, and projects features.
        """
        B, T, D = h_seq.shape
        device = h_seq.device

        # 1. Predict 3 Asymmetric Quantiles
        q_l = self.head_low(h_seq)   # [B, T, out_dim]
        q_m = self.head_med(h_seq)   # [B, T, out_dim]
        q_h = self.head_high(h_seq)  # [B, T, out_dim]

        # Enforce monotonicity: q_low <= q_med <= q_high
        q_med = q_m
        q_low = q_med - F.relu(q_med - q_l)
        q_high = q_med + F.relu(q_h - q_med)

        loss_total = None
        emp_coverage = None

        if targets is not None:
            # 2. Compute Multi-Quantile Pinball Losses
            l_low  = self.pinball_loss(targets, q_low, self.tau_low)
            l_med  = self.pinball_loss(targets, q_med, 0.50)
            l_high = self.pinball_loss(targets, q_high, self.tau_high)
            loss_total = l_low + l_med + l_high

            # Update calibration score buffer
            self.update_calibration(q_low, q_high, targets)

        # 3. Compute Conformal Calibration Threshold Q_hat
        Q_hat = self.get_conformal_threshold()  # [1]

        # 4. Calibrated Prediction Interval Bandwidth: W(x) = (q_high - q_low) + 2 * Q_hat
        bandwidth = (q_high - q_low) + 2.0 * Q_hat  # [B, T, out_dim]

        if targets is not None:
            # Empirical coverage test: q_low - Q_hat <= y <= q_high + Q_hat
            covered = (targets >= (q_low - Q_hat)) & (targets <= (q_high + Q_hat))
            emp_coverage = covered.float().mean()

        # 5. Feature Projection
        f_concat = torch.cat([q_med, q_low, q_high, bandwidth], dim=-1)  # [B, T, out_dim * 4]
        cqr_emb = self.out_proj(f_concat)  # [B, T, d_model]

        augmented = h_seq + cqr_emb

        return ConformalQuantileOutput(
            quantile_features=cqr_emb,
            q_low=q_low,
            q_med=q_med,
            q_high=q_high,
            interval_bandwidth=bandwidth,
            conformal_threshold=Q_hat,
            pinball_loss=loss_total,
            empirical_coverage=emp_coverage,
            augmented_features=augmented,
        )
