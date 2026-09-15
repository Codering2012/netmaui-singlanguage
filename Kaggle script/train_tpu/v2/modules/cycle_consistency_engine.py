#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — BIDIRECTIONAL CYCLE-CONSISTENCY REGULARIZATION ENGINE
================================================================================
Implements Multi-Modal Video <-> Text Cycle Consistency (CCL-SLR / BeyondGloss):
1. Forward Cycle (Video -> Decoded Text -> Latent Reconstruction):
     L_{cycle, fwd} = || z_{vis}(X) - z_{text}(Decoder(X)) ||_2^2
   Guarantees that the generated gloss/English text contains sufficient semantic
   information to fully reconstruct the visual gesture embedding.
2. Reverse Cycle (Text -> Motion Generator -> Text Decoding):
     L_{cycle, rev} = CrossEntropy( Decoder( MotionGen(Y) ), Y )
   Enforces that synthesized sign kinematic trajectories faithfully reconstruct
   the originating sentence.
3. Joint Multi-Modal Regularization:
     L_{cycle} = lambda_{fwd} * L_{cycle, fwd} + lambda_{rev} * L_{cycle, rev}
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLCycleConsistencyEngine(nn.Module):
    """
    Bidirectional Video <-> Text Cycle Consistency Regularizer for ASL Foundation Models.
    """

    def __init__(
        self,
        model: nn.Module,
        d_model: int = 128,
        vocab_size: int = 80,
        lambda_fwd: float = 1.0,
        lambda_rev: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.lambda_fwd = lambda_fwd
        self.lambda_rev = lambda_rev

        # Latent Visual-Text Projection Bridges
        self.vis_to_text_proj = nn.Linear(d_model, d_model)
        self.text_to_vis_proj = nn.Linear(d_model, d_model)

        # Lightweight Motion Latent Synthesizer for Reverse Cycle
        self.motion_synthesizer = nn.Sequential(
            nn.Embedding(vocab_size, d_model, padding_idx=0),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_forward_cycle_loss(
        self,
        h_cls_vis: torch.Tensor,
        dec_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward Cycle: Video -> Decoded Text Softmax -> Text Latent -> Match Visual Latent.
        h_cls_vis: [B, D]
        dec_logits: [B, L, V]
        """
        probs = F.softmax(dec_logits, dim=-1)  # [B, L, V]
        # Soft token embedding expectation
        token_weight = getattr(self.model.decoder, "token_emb", None)
        if token_weight is not None and hasattr(token_weight, "weight"):
            W = token_weight.weight[:probs.size(-1)]  # [V, D]
            soft_text_emb = torch.matmul(probs, W).mean(dim=1)  # [B, D]
        else:
            # Fallback projection
            soft_text_emb = probs.mean(dim=1)  # [B, V]
            if soft_text_emb.size(-1) != self.d_model:
                W_proj = self.motion_synthesizer[0].weight[:probs.size(-1)]
                soft_text_emb = torch.matmul(probs, W_proj).mean(dim=1)

        z_text = self.text_to_vis_proj(soft_text_emb)
        z_vis = self.vis_to_text_proj(h_cls_vis)

        z_text_norm = F.normalize(z_text, p=2, dim=-1)
        z_vis_norm = F.normalize(z_vis, p=2, dim=-1)

        loss_fwd = F.mse_loss(z_text_norm, z_vis_norm)
        return loss_fwd

    def compute_reverse_cycle_loss(
        self,
        text_seq: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reverse Cycle: Ground-Truth Text -> Synthesized Motion Latents -> Model Decoder -> Reconstruct Text.
        text_seq: [B, L]
        """
        B, L = text_seq.shape
        # Synthesize pseudo-kinematics sequence in latent space
        synth_motion = self.motion_synthesizer(text_seq)  # [B, L, D]

        # Ingest synthesized motion through decoder
        dec_pad_id = 0
        dec_padding_mask = text_seq == dec_pad_id

        # Use model decoder forward
        dec_out = self.model.decoder(
            tgt_ids=text_seq[:, :-1] if L > 1 else text_seq,
            memory=synth_motion,
            memory_key_padding_mask=text_mask,
        )
        out_logits = dec_out[0] if isinstance(dec_out, tuple) else dec_out

        target = text_seq[:, 1:] if L > 1 else text_seq
        loss_rev = F.cross_entropy(
            out_logits.reshape(-1, out_logits.size(-1)),
            target.reshape(-1),
            ignore_index=dec_pad_id,
        )
        return loss_rev

    def forward(
        self,
        features: torch.Tensor,
        text_seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes joint bidirectional cycle consistency loss.
        """
        # Forward pass on visual model
        out = self.model(
            input_x=features,
            mask=mask,
            frame_indices=frame_indices,
            gloss_seq=text_seq,
        )

        h_cls = out["h_cls"]
        dec_logits = out["dec_logits"]

        loss_fwd = self.compute_forward_cycle_loss(h_cls, dec_logits)
        loss_rev = self.compute_reverse_cycle_loss(text_seq)

        total_cycle_loss = self.lambda_fwd * loss_fwd + self.lambda_rev * loss_rev

        return {
            "loss_cycle": total_cycle_loss,
            "loss_fwd_cycle": loss_fwd,
            "loss_rev_cycle": loss_rev,
            "dec_logits": dec_logits,
        }
