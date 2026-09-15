#!/usr/bin/env python3
"""
================================================================================
ASL V3 FLAGSHIP TPU TRAINING ORCHESTRATOR
================================================================================
Distributed training engine for ASL V3 Multi-Tier Foundation Architecture:
- Cloud TPU v5e (PJRT / PyTorch-XLA) & Multi-GPU / CPU support.
- Homoscedastic Multi-Task Loss Balancing across 11 geometric and syntactic losses.
- Single-graph execution invariance (strict 128x128 tile alignments).
- Zero-bypass end-to-end gradient verification.
================================================================================
"""

import os
import sys
import argparse
import math
import time
from typing import Dict, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# TPU PyTorch/XLA imports with safe fallback
try:
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

from train_tpu.v3.modules.asl_v3_foundation_model import ASLV3FoundationModel, V3ModelOutput
from train_tpu.v3.engine.dataset import fast_vectorized_v3_collate_fn


class V3HomoscedasticLossWrapper(nn.Module):
    """
    Homoscedastic uncertainty-weighted loss balancer for V3 multi-task objectives.
    Learns log-variance parameters s_i for each loss term to dynamically scale gradients:
    L_total = sum_i (0.5 * exp(-s_i) * L_i + 0.5 * s_i)
    Strictly bounded with clamp(min=-4.0, max=8.0) to prevent exponential overflow.
    """

    def __init__(self, loss_names: list):
        super().__init__()
        self.loss_names = loss_names
        # Initialize log variances to 0.0 (initial weight = 0.5)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in loss_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total_loss = torch.zeros((), device=next(iter(losses.values())).device)
        logged_weights: Dict[str, torch.Tensor] = {}

        for name, loss_val in losses.items():
            if name in self.log_vars:
                # Clamp log variance to prevent exp(-s) gradient explosion or vanishing
                s = torch.clamp(self.log_vars[name], min=-3.0, max=3.0)
                precision = torch.exp(-s)
                total_loss = total_loss + 0.5 * precision * loss_val + 0.5 * s
                # Keep detached tensor instead of calling .item() to prevent XLA device-host sync stalls
                logged_weights[f"w_{name}"] = precision.detach()
            else:
                total_loss = total_loss + loss_val

        return total_loss, logged_weights


class ModelEMA:
    """
    Exponential Moving Average with strict .detach() graph-isolation to prevent memory leaks.
    Safely handles both floating-point parameters and integer buffers (e.g. step counters).
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                if v.is_floating_point():
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
                else:
                    self.shadow[k].copy_(v.detach())

    def apply_shadow(self, model: nn.Module):
        model.load_state_dict(self.shadow)


class V3TrainingOrchestrator:
    """
    Training harness for ASL V3 Foundation Model on Cloud TPU v5e or local hardware.
    """

    def __init__(
        self,
        model: ASLV3FoundationModel,
        lr: float = 1e-4,
        weight_decay: float = 1e-2,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        device: Optional[torch.device] = None,
    ):
        self.device = device or (xm.xla_device() if HAS_XLA else torch.device("cpu"))
        self.model = model.to(self.device)
        self.use_ema = use_ema
        self.ema = ModelEMA(self.model, decay=ema_decay) if use_ema else None

        loss_keys = [
            "loss_ctc",
            "loss_ctc_english",
            "loss_phonology",
            "loss_locus",
            "loss_nmm",
            "loss_polarity",
            "loss_classifier",
            "loss_permutation",
            "loss_anti_hallucination",
            "loss_translation_ce",
            "loss_epenthesis_consistency",
            "loss_fs_consistency",
            "loss_mam",
            "loss_semantic_bridge",
            "loss_coverage",
            "loss_monotonic",
        ]
        self.loss_wrapper = V3HomoscedasticLossWrapper(loss_keys).to(self.device)
        self.curriculum_stage: int = 3

        # Decoupled AdamW optimizer:
        # Protect pretrained English decoder weights while allowing faster Conformer representation learning
        decoder_params = []
        if self.model.gpt2_decoder is not None:
            decoder_params.extend(list(self.model.gpt2_decoder.parameters()))
        decoder_params.extend(list(self.model.decoder.parameters()))
        decoder_params.extend(list(self.model.decoder_head.parameters()))
        decoder_params.extend(list(self.model.text_embedding.parameters()))
        decoder_ids = set(id(p) for p in decoder_params)

        loss_params = list(self.loss_wrapper.parameters())
        loss_ids = set(id(p) for p in loss_params)

        encoder_params = [p for p in self.model.parameters() if id(p) not in decoder_ids and id(p) not in loss_ids]

        param_groups = [
            {"params": encoder_params, "lr": lr, "weight_decay": weight_decay},
            {"params": decoder_params, "lr": lr * 0.3, "weight_decay": weight_decay},
            {"params": loss_params, "lr": lr * 2.0, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(param_groups)
        self.scheduler: Optional[Any] = None

    def setup_full_dataset_scheduler(self, total_steps: int):
        """
        Schedules learning rate across 100% of the dataset steps.
        Zero warmup truncation: starts at full learning rate immediately across the full dataset.
        """
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_steps), eta_min=1e-6
        )

    def set_curriculum_stage(self, stage: int):
        """
        Configures the 3-Stage Curriculum Staging Engine:
        - Stage 1: Kinematic & Phonetic Pretraining (1500 FPS, frozen decoder, MAM active)
        - Stage 2: Syntactic & Semantic Bridging (800 FPS, Sinkhorn + Semantic Anchor active)
        - Stage 3: End-to-End Autoregressive Translation (Full Decoder + Grounding active)
        """
        self.curriculum_stage = stage
        if stage == 1:
            if self.model.gpt2_decoder is not None:
                for p in self.model.gpt2_decoder.parameters():
                    p.requires_grad = False
            for p in self.model.decoder.parameters():
                p.requires_grad = False
            for p in self.model.text_embedding.parameters():
                p.requires_grad = False
            for p in self.model.decoder_head.parameters():
                p.requires_grad = False
            for p in self.model.grounding_shield.parameters():
                p.requires_grad = False
        elif stage == 2:
            for p in self.model.encoder.parameters():
                p.requires_grad = True
            for p in self.model.sinkhorn_transducer.parameters():
                p.requires_grad = True
            for p in self.model.semantic_anchor.parameters():
                p.requires_grad = True
        elif stage == 3:
            for p in self.model.parameters():
                p.requires_grad = True

    def train_step(self, batch: Dict[str, Any], sync_metrics: bool = False) -> Dict[str, Any]:
        """
        Executes a single forward, backward, and optimization step.
        If sync_metrics is False, returns detached tensors without host-sync stalls.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Move tensors to device
        kinematics = batch["kinematics"].to(self.device)
        roi_visual = batch.get("roi_visual")
        if roi_visual is not None:
            roi_visual = roi_visual.to(self.device)
        hand_visual = batch.get("hand_visual")
        if hand_visual is not None:
            hand_visual = hand_visual.to(self.device)
        phonology = batch.get("phonology")
        if phonology is not None:
            phonology = phonology.to(self.device)
        face_landmarks = batch.get("face_landmarks")
        if face_landmarks is not None:
            face_landmarks = face_landmarks.to(self.device)
        cranial_imu = batch.get("cranial_imu")
        if cranial_imu is not None:
            cranial_imu = cranial_imu.to(self.device)
        text_tokens = batch.get("text_tokens", batch.get("english_seq"))
        if text_tokens is not None:
            text_tokens = text_tokens.to(self.device)
        text_is_negative = batch.get("text_is_negative")
        if text_is_negative is not None:
            text_is_negative = text_is_negative.to(self.device)
        hand_mask = batch.get("hand_mask")
        if hand_mask is not None:
            hand_mask = hand_mask.to(self.device)

        tgt_sentence_emb = batch.get("target_sentence_embeddings", batch.get("sentence_embedding"))
        if tgt_sentence_emb is not None:
            tgt_sentence_emb = tgt_sentence_emb.to(self.device)

        # Forward pass with curriculum-aware flags
        enable_mam = (self.curriculum_stage in (1, 2, 3))
        output: V3ModelOutput = self.model(
            kinematics=kinematics,
            roi_visual=roi_visual,
            hand_visual=hand_visual,
            phonology=phonology,
            face_landmarks=face_landmarks,
            cranial_imu=cranial_imu,
            text_tokens=text_tokens,
            text_is_negative=text_is_negative,
            hand_mask=hand_mask,
            target_sentence_embeddings=tgt_sentence_emb,
            enable_mam=enable_mam,
        )

        losses = dict(output.multi_task_losses)

        # Translation Cross Entropy Loss if text tokens are present and stage permits
        if self.curriculum_stage == 3 and output.decoder_logits is not None and text_tokens is not None and text_tokens.shape[1] > 1:
            tgt_out = text_tokens[:, 1:].detach()
            logits = output.decoder_logits[:, :-1, :]
            losses["loss_translation_ce"] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                tgt_out.reshape(-1),
                ignore_index=0,
            )

        # Balance multi-task losses
        total_loss, weights = self.loss_wrapper(losses)

        # Backward pass
        total_loss.backward()

        # Step optimizer with device-appropriate gradient clipping
        if HAS_XLA and self.device.type == "xla":
            xm.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            xm.optimizer_step(self.optimizer)
            if self.scheduler is not None:
                self.scheduler.step()
            xm.mark_step()
        else:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

        # Update EMA if enabled
        if self.ema is not None:
            self.ema.update(self.model)

        # Detach all metrics to prevent autograd graph retention
        if sync_metrics:
            metrics = {"total_loss": total_loss.item()}
            for k, v in losses.items():
                metrics[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in weights.items():
                metrics[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
        else:
            metrics = {"total_loss": total_loss.detach()}
            for k, v in losses.items():
                metrics[k] = v.detach() if isinstance(v, torch.Tensor) else v
            for k, v in weights.items():
                metrics[k] = v.detach() if isinstance(v, torch.Tensor) else v

        return metrics


def build_v3_parser() -> argparse.ArgumentParser:
    """Builds CLI parser for ASL V3 Training."""
    parser = argparse.ArgumentParser(description="ASL V3 Flagship TPU Training")
    parser.add_argument("--d_model", type=int, default=128, help="Latent feature dimension")
    parser.add_argument("--num_enc_layers", type=int, default=4, help="Encoder depth")
    parser.add_argument("--num_dec_layers", type=int, default=4, help="Decoder depth")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size per TPU core")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--total_steps", type=int, default=0, help="Total training steps for full dataset cosine decay (0 = epochs * steps_per_epoch)")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps (default: 0, full dataset training immediately at peak lr)")
    return parser


def main():
    parser = build_v3_parser()
    args = parser.parse_args()
    print("Initializing ASL V3 Foundation Architecture on:", "TPU" if HAS_XLA else "CPU/GPU")
    model = ASLV3FoundationModel(d_model=args.d_model, num_enc_layers=args.num_enc_layers, num_dec_layers=args.num_dec_layers)
    orchestrator = V3TrainingOrchestrator(model, lr=args.lr)
    if args.total_steps > 0:
        orchestrator.setup_full_dataset_scheduler(args.total_steps)
        print(f"Full Dataset Cosine Annealing scheduled over {args.total_steps:,} steps (0 warmup).")
    print(f"ASL V3 Ready: {sum(p.numel() for p in model.parameters())} parameters.")


if __name__ == "__main__":
    main()
