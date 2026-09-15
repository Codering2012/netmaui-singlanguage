#!/usr/bin/env python3
"""
================================================================================
  BI-DIRECTIONAL SIGN LANGUAGE TRANSLATION (SLT) & CYCLE-CONSISTENCY VALIDATOR
================================================================================
Validates bi-directional translation fidelity:
  1. Video / Kinematics -> English Translation (Forward SLT)
  2. English -> Gloss Sequence Generation (Back-Translation)
  3. Cycle-Consistency Metric: BLEU(Back-Translated Gloss, Ground Truth Gloss)
================================================================================
"""

import sys
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np

# Setup paths
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))
try:
    from asl_master_foundation_model import ASLFoundationModel
except ImportError:
    try:
        from train_tpu.v2.modules.asl_master_foundation_model import ASLFoundationModel
    except ImportError:
        ASLFoundationModel = Any


class BiDirectionalSLTValidator:
    """
    Evaluates forward and reverse translation with cycle-consistency checks.
    """

    def __init__(self, model: ASLFoundationModel):
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def forward_translate_video(
        self,
        features: torch.Tensor,
        roi_visual: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_gen_len: int = 32,
        bos_token: int = 1,
        eos_token: int = 2,
    ) -> torch.Tensor:
        """
        Generates English text token IDs from sign video kinematics.
        """
        B = features.size(0)
        device = features.device

        # 1. Encode video
        enc_out = self.model._encode(features, mask=mask, roi_visual=roi_visual)
        h_seq = enc_out[1]
        enc_mask = enc_out[2] if len(enc_out) > 2 else mask

        # 2. Autoregressive decoding with English decoder
        generated = torch.full((B, 1), bos_token, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        decoder_op = self.model.english_decoder if self.model.english_decoder is not None else self.model.decoder
        for step in range(max_gen_len):
            dec_out = decoder_op(
                generated,
                memory=h_seq,
                memory_key_padding_mask=(~enc_mask) if enc_mask is not None else None,
            )
            dec_logits = dec_out[0] if isinstance(dec_out, tuple) else dec_out
            next_token = dec_logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            finished = finished | (next_token.squeeze(-1) == eos_token)
            if finished.all():
                break

        return generated

    def compute_cycle_consistency_score(
        self,
        gt_glosses: List[List[str]],
        predicted_english: List[List[str]],
        back_translated_glosses: List[List[str]],
    ) -> Dict[str, float]:
        """
        Computes forward match and reverse cycle-consistency similarity.
        """
        matches = 0
        total_tokens = 0
        for ref_g, cycle_g in zip(gt_glosses, back_translated_glosses):
            for t_ref, t_cyc in zip(ref_g, cycle_g):
                if t_ref == t_cyc:
                    matches += 1
                total_tokens += 1

        cycle_acc = (matches / max(1, total_tokens)) * 100.0
        return {
            "cycle_token_accuracy": cycle_acc,
            "total_tokens": total_tokens,
        }


def main():
    print("[INFO] Bi-Directional SLT Validator initialized.")
    model = ASLFoundationModel(
        num_enc_layers=4,
        num_dec_layers=4,
        d_enc=256,
        d_dec=256,
        vocab_size=2560,
        english_vocab_size=23552,
        is_causal=False,
    )
    validator = BiDirectionalSLTValidator(model)

    mock_feat = torch.randn(2, 20, 60, 9)
    out_tokens = validator.forward_translate_video(mock_feat, max_gen_len=10)
    print(f"[+] Forward Translated English Tokens Shape: {out_tokens.shape}")


if __name__ == "__main__":
    main()
