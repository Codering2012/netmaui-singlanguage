#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — COVARIATE SHIFT CONFORMAL ENGINE (COVARIATESHIFTSIGN)
================================================================================
Implements Weighted Conformal Prediction under Temporal Covariate Shift (ShiftConformal-SLT):
1. Covariate Density Ratio Estimator w(X) = p_test(X) / p_calib(X):
     w(X) = exp( Linear( LayerNorm( Mean(X) ) ) ) in (0, +inf)
2. Normalized Conformal Importance Weights:
     p_tilde_i = w(X_i) / ( sum_{j=1}^n w(X_j) + w(X_{n+1}) )
3. Weighted Conformal Quantile Calibration q_{1-alpha}(X_{n+1}):
     Finds smallest non-conformity threshold s such that cumulative weighted mass >= 1 - alpha.
4. Finite-Sample Guaranteed 1 - alpha (e.g. 90%) Prediction Sets:
     C_{1-alpha}(X_{n+1}) = { y in Y : 1 - softmax(logits)_y <= q_{1-alpha}(X_{n+1}) }
5. Covariate Shift Invariant Feature Gating:
     H_shift = H * sigmoid( Linear( [w(X), q_{1-alpha}, set_size] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CovariateShiftOutput(NamedTuple):
    shift_features: torch.Tensor        # [B, T, d_model] Covariate-shift robust representations
    density_ratios: torch.Tensor        # [B] Estimated importance weights w(X)
    weighted_quantiles: torch.Tensor    # [B] Adaptive weighted conformal threshold q_{1-alpha}(X)
    prediction_sets: torch.Tensor       # [B, num_classes] Boolean mask of included candidate classes
    set_sizes: torch.Tensor             # [B] Card(C_{1-alpha}(X)) candidate class counts
    empirical_coverage: Optional[float] # Verified coverage probability on batch
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + shift_features


class ASLCovariateShiftConformalEngine(nn.Module):
    """
    Weighted Conformal Prediction & Temporal Covariate Shift Calibration Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        num_classes: int = 100,
        alpha: float = 0.10,            # 90% target coverage (alpha = 0.10)
        calib_size: int = 64,           # Calibration buffer size
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.num_classes = num_classes
        self.alpha = alpha
        self.calib_size = calib_size

        # 1. Covariate Density Ratio Network w(X) = exp( MLP( X ) )
        # Mean kinematics per sequence: num_keypoints * in_channels = 60 * 9 = 540
        in_dim = num_keypoints * in_channels
        self.density_mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # 2. Calibration Buffers
        self.register_buffer("calib_scores", torch.zeros(calib_size, dtype=torch.float32))
        self.register_buffer("calib_weights", torch.ones(calib_size, dtype=torch.float32))
        self.register_buffer("calib_ptr", torch.tensor(0, dtype=torch.long))
        self.register_buffer("calib_filled", torch.tensor(False, dtype=torch.bool))

        # 3. Shift Invariant Gating Projection
        self.gate_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.Sigmoid(),
        )

    def compute_density_ratio(self, kinematics: torch.Tensor) -> torch.Tensor:
        """
        Computes positive importance weight w(X) = exp( MLP( mean(kinematics) ) ).
        kinematics: [B, T, 60, 9] -> w: [B]
        """
        B, T, K, C = kinematics.shape
        x_mean = kinematics.mean(dim=1).reshape(B, K * C)  # [B, 540]
        log_w = self.density_mlp(x_mean).squeeze(-1)       # [B]
        # Clamp log_w to prevent numerical overflow/underflow
        w = torch.exp(log_w.clamp(min=-5.0, max=5.0))      # [B]
        return w

    @torch.no_grad()
    def update_calibration_buffer(self, scores: torch.Tensor, weights: torch.Tensor):
        """
        Updates circular calibration buffer with non-conformity scores and density weights.
        """
        num_new = scores.shape[0]
        for i in range(num_new):
            idx = self.calib_ptr.item()
            self.calib_scores[idx] = scores[i].item()
            self.calib_weights[idx] = weights[i].item()
            new_ptr = (idx + 1) % self.calib_size
            self.calib_ptr.copy_(torch.tensor(new_ptr, dtype=torch.long))
            if new_ptr == 0:
                self.calib_filled.copy_(torch.tensor(True, dtype=torch.bool))

    def compute_weighted_quantile(
        self,
        test_weight: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        """
        Calculates the weighted conformal quantile q_{1-alpha}(X_test) for each test sample.
        """
        B = test_weight.shape[0]
        device = test_weight.device

        n_calib = self.calib_size if self.calib_filled.item() else max(1, self.calib_ptr.item())
        calib_s = self.calib_scores[:n_calib]    # [N]
        calib_w = self.calib_weights[:n_calib]   # [N]

        # Sort calibration scores
        sorted_s, sort_idx = torch.sort(calib_s)
        sorted_w = calib_w[sort_idx]

        quantiles = torch.zeros(B, device=device, dtype=test_weight.dtype)

        target_prob = 1.0 - self.alpha

        for b in range(B):
            w_b = test_weight[b]
            # All weights: sorted calibration weights + test weight at +inf
            total_w = sorted_w.sum() + w_b
            norm_w = sorted_w / (total_w + 1e-6)
            cum_w = torch.cumsum(norm_w, dim=0)

            # Find index where cumulative weight >= 1 - alpha
            mask = cum_w >= target_prob
            if mask.any():
                idx_q = torch.nonzero(mask, as_tuple=False)[0].item()
                q_val = sorted_s[idx_q].item()
            else:
                q_val = 1.0

            quantiles[b] = q_val

        return quantiles

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        logits: torch.Tensor,                        # [B, num_classes] raw class logits
        labels: Optional[torch.Tensor] = None,       # [B] ground truth class indices (for calibration)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> CovariateShiftOutput:
        """
        Computes importance weights, weighted conformal quantiles, and covariate-shift gated features.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # 1. Estimate Covariate Density Ratio w(X)
        w_X = self.compute_density_ratio(kinematics)  # [B]

        # 2. Compute Softmax Probabilities and Non-Conformity Scores
        probs = F.softmax(logits, dim=-1)             # [B, num_classes]

        # If labels provided, update calibration buffer with true-class non-conformity: 1 - probs[y]
        empirical_cov = None
        if labels is not None:
            # 1 - pi_y
            true_probs = probs.gather(1, labels.unsqueeze(-1)).squeeze(-1)  # [B]
            calib_scores = (1.0 - true_probs).detach()                     # [B]
            self.update_calibration_buffer(calib_scores, w_X.detach())

        # 3. Compute Adaptive Weighted Quantiles q_{1-alpha}(X)
        q_weights = self.compute_weighted_quantile(w_X.detach())  # [B]

        # 4. Construct Conformal Prediction Sets: { y : 1 - probs[y] <= q }
        # non-conformity of all classes: [B, num_classes]
        all_nonconf = 1.0 - probs                                 # [B, num_classes]
        q_exp = q_weights.unsqueeze(-1)                           # [B, 1]
        pred_sets = all_nonconf <= q_exp                          # [B, num_classes]
        set_sizes = pred_sets.sum(dim=-1).float()                 # [B]

        if labels is not None:
            contained = pred_sets.gather(1, labels.unsqueeze(-1)).squeeze(-1) # [B]
            empirical_cov = contained.float().mean().item()

        # 5. Covariate Shift Invariant Gating
        # Summary vector: [w(X), q, set_size / num_classes] -> [B, 3]
        gate_in = torch.stack([w_X, q_weights, set_sizes / max(1, self.num_classes)], dim=-1) # [B, 3]
        gate = self.gate_proj(gate_in).unsqueeze(1)  # [B, 1, d_model]

        if h_seq is None:
            h_base = torch.zeros(B, T, self.d_model, device=device, dtype=gate.dtype)
        else:
            h_base = h_seq

        h_shift = h_base * gate

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_shift

        return CovariateShiftOutput(
            shift_features=h_shift,
            density_ratios=w_X,
            weighted_quantiles=q_weights,
            prediction_sets=pred_sets,
            set_sizes=set_sizes,
            empirical_coverage=empirical_cov,
            augmented_features=augmented,
        )
