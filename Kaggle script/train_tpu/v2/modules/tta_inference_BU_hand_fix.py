#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — TEST-TIME AUGMENTATION (TTA) INFERENCE ENGINE
================================================================================
Improves translation robustness and confidence calibration via:
1. Spatial Multi-Scale TTA (Camera distance / zoom invariance: 0.95x, 1.0x, 1.05x)
2. Bilateral Keypoint Mirroring (Left-hand / Right-hand dominant signer invariance)
3. Ensembled Logit Aggregation across all views for CTC & Autoregressive Decoding
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLTTAInferenceEngine:
    """
    Test-Time Augmentation (TTA) manager for ASL Foundation Models.
    """

    def __init__(
        self,
        model: nn.Module,
        scales: Tuple[float, ...] = (0.95, 1.0, 1.05),
        enable_mirroring: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model = model
        self.scales = scales
        self.enable_mirroring = enable_mirroring
        self.device = torch.device(device)
        self.model.eval()

    def generate_views(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]]:
        """
        Generates augmented spatio-temporal views for input tensor [B, T, K, C].
        """
        views = []
        B, T, K, C = x.shape

        for scale in self.scales:
            # 1. Spatial Scaling view
            x_scaled = x.clone()
            # Scale (x, y, z) position coordinates (first 3 channels)
            x_scaled[..., :3] = x_scaled[..., :3] * scale
            views.append((x_scaled, mask, frame_indices))

            # 2. Bilateral Mirroring View (if enabled)
            if self.enable_mirroring:
                x_mirrored = x_scaled.clone()
                # Invert X coordinates
                x_mirrored[..., 0] = -x_mirrored[..., 0]

                # Swap Left Hand (0-20) and Right Hand (21-41) keypoints
                if K >= 42:
                    lh = x_mirrored[:, :, 0:21, :].clone()
                    rh = x_mirrored[:, :, 21:42, :].clone()
                    x_mirrored[:, :, 0:21, :] = rh
                    x_mirrored[:, :, 21:42, :] = lh

                views.append((x_mirrored, mask, frame_indices))

        return views

    @torch.no_grad()
    def predict_ctc_tta(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Performs multi-view TTA evaluation for CTC recognition.
        Returns ensemble-averaged log-probabilities and collapsed token sequence.
        """
        views = self.generate_views(x, mask, frame_indices)
        all_log_probs = []

        for v_x, v_mask, v_fi in views:
            v_x = v_x.to(self.device)
            if v_mask is not None:
                v_mask = v_mask.to(self.device)
            if v_fi is not None:
                v_fi = v_fi.to(self.device)

            out = self.model(input_x=v_x, mask=v_mask, frame_indices=v_fi)
            all_log_probs.append(out["ctc_log_probs"])

        # Softmax average across views
        probs = [F.softmax(lp, dim=-1) for lp in all_log_probs]
        avg_probs = torch.stack(probs, dim=0).mean(dim=0)
        avg_log_probs = torch.log(avg_probs.clamp(min=1e-8))

        # Collapse tokens
        best_tokens = torch.argmax(avg_log_probs, dim=-1)[0].tolist()
        collapsed = []
        prev = None
        for tok in best_tokens:
            if tok != prev and tok != 0:
                collapsed.append(tok)
            prev = tok

        return {
            "ctc_log_probs": avg_log_probs,
            "ctc_tokens": collapsed,
            "num_views": len(views),
        }
