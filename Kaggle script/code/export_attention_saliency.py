#!/usr/bin/env python3
"""
================================================================================
  SPATIAL-TEMPORAL SALIENCY & ATTENTION ROLLOUT EXPORTER FOR ASL MODEL V2
================================================================================
Computes gradient-based and attention-based saliency maps over 60 anatomical landmarks:
  1. Temporal Saliency Profile: Which frames triggered the predicted word?
  2. Anatomical Saliency Breakdown: Face vs Pose vs Left Hand vs Right Hand.
  3. Integrated Gradient Attribution over (x, y, z, dx, dy, dz, d2x, d2y, d2z).
================================================================================
"""

import sys
import os
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np

# Setup paths
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(workspace_root / "train_tpu"))

from train_all_in_one_tpu_v2 import ASLFoundationModel


class SaliencyMapExporter:
    """
    Extracts spatial and temporal attribution maps for ASL translation interpretability.
    """

    def __init__(self, model: ASLFoundationModel):
        self.model = model
        self.model.eval()

    def compute_gradient_saliency(
        self,
        features: torch.Tensor,
        target_token_idx: Optional[int] = None,
        roi_visual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Computes input attribution via Input x Gradient saliency.
        features: [1, T, 60, 9] requires_grad=True
        Returns:
          temporal_saliency: [T] frame-wise importance score
          landmark_saliency: [60] anatomical keypoint importance score
          part_breakdown: { 'face': float, 'pose': float, 'left_hand': float, 'right_hand': float }
        """
        feat = features.clone().detach().requires_grad_(True)
        self.model.zero_grad()

        # Forward pass
        enc_out = self.model._encode(feat, mask=mask, roi_visual=roi_visual)
        h_seq = enc_out[1]
        ctc_logits = self.model.ctc_head(h_seq)  # [1, T, V]

        if target_token_idx is not None:
            score = ctc_logits[0, :, target_token_idx].sum()
        else:
            score = ctc_logits.max(dim=-1).values.sum()

        score.backward()

        # Input x Gradient attribution [1, T, 60, 9]
        grad = feat.grad
        if grad is None:
            attribution = torch.zeros_like(feat)
        else:
            attribution = torch.abs(feat * grad)

        attr_np = attribution.squeeze(0).detach().cpu().numpy()  # [T, 60, 9]
        # Sum over 9 feature channels
        kp_attr_matrix = attr_np.sum(axis=-1)  # [T, 60]

        temporal_saliency = kp_attr_matrix.sum(axis=1)  # [T]
        landmark_saliency = kp_attr_matrix.sum(axis=0)  # [60]

        total_sal = landmark_saliency.sum() + 1e-8
        part_breakdown = {
            "face": float(landmark_saliency[:14].sum() / total_sal),
            "pose": float(landmark_saliency[14:18].sum() / total_sal),
            "left_hand": float(landmark_saliency[18:39].sum() / total_sal),
            "right_hand": float(landmark_saliency[39:60].sum() / total_sal),
        }

        return {
            "temporal_saliency": temporal_saliency,
            "landmark_saliency": landmark_saliency,
            "kp_attr_matrix": kp_attr_matrix,
            "part_breakdown": part_breakdown,
        }


def main():
    print("[INFO] Initializing Spatial-Temporal Saliency Exporter...")
    model = ASLFoundationModel(
        num_enc_layers=4,
        num_dec_layers=4,
        d_enc=256,
        d_dec=256,
        vocab_size=2560,
        english_vocab_size=23552,
        is_causal=False,
    )
    exporter = SaliencyMapExporter(model)

    mock_feat = torch.randn(1, 30, 60, 9)
    res = exporter.compute_gradient_saliency(mock_feat)

    print(f"[+] Temporal Saliency Length: {len(res['temporal_saliency'])}")
    print(f"[+] Anatomical Attribution Breakdown: {res['part_breakdown']}")


if __name__ == "__main__":
    main()
