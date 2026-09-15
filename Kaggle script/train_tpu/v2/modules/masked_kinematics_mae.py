#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SELF-SUPERVISED SPATIOTEMPORAL MASKED AUTOENCODER (MAE)
================================================================================
Implements Self-Supervised Pretraining for Landmark Foundation Models:
1. Spatiotemporal Kinematics Masking: High-ratio masking (60% - 85%) across
   frames and joints, with optional hand-salient prioritization (SignMAE).
2. Kinematics Reconstruction Head: Lightweight decoder reconstructing 3D joint
   coordinates and first-order temporal velocity vectors:
     L_MAE = MSE(X_hat_mask, X_mask) + lambda_vel * MSE(V_hat_mask, V_mask)
3. Zero-Label Foundation Pretraining: Enables unsupervised pretraining on
   unlabeled sign language video landmark archives.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLMaskedKinematicsMAE(nn.Module):
    """
    Self-Supervised Spatiotemporal Masked Autoencoder Engine for ASLFoundationModel.
    """

    def __init__(
        self,
        encoder_model: nn.Module,
        mask_ratio: float = 0.70,
        hand_mask_ratio: float = 0.85,
        hand_joint_indices: Optional[List[int]] = None,
        lambda_vel: float = 0.50,
        d_decoder: int = 128,
        num_decoder_layers: int = 2,
    ):
        super().__init__()
        self.encoder = encoder_model
        self.mask_ratio = mask_ratio
        self.hand_mask_ratio = hand_mask_ratio
        self.lambda_vel = lambda_vel

        # Default hand joint indices in 60-keypoint format (Left: 11..31, Right: 32..52)
        if hand_joint_indices is None:
            self.hand_joint_indices = list(range(11, 53))
        else:
            self.hand_joint_indices = list(hand_joint_indices)

        d_enc = getattr(self.encoder, "d_enc", 128)
        self.channels_per_kp = getattr(self.encoder, "channels_per_kp", 9)
        self.num_keypoints = 60

        # Lightweight MAE Reconstruction Head
        self.decoder_stem = nn.Linear(d_enc, d_decoder)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_decoder,
            nhead=4,
            dim_feedforward=d_decoder * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.decoder_transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_decoder_layers)
        self.reconstruction_head = nn.Linear(d_decoder, self.num_keypoints * self.channels_per_kp)

    def generate_spatiotemporal_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        salient_hand_masking: bool = True,
    ) -> torch.Tensor:
        """
        Generates binary spatiotemporal mask [B, T, K]:
        1 = Masked (to be reconstructed), 0 = Visible (fed to encoder).
        """
        if not salient_hand_masking:
            rand_tensor = torch.rand((batch_size, seq_len, self.num_keypoints), device=device)
            return rand_tensor < self.mask_ratio

        # Hand-salient masking: higher masking ratio on hands
        mask = torch.zeros((batch_size, seq_len, self.num_keypoints), dtype=torch.bool, device=device)
        rand_body = torch.rand((batch_size, seq_len, self.num_keypoints), device=device)
        rand_hand = torch.rand((batch_size, seq_len, self.num_keypoints), device=device)

        body_indices = [i for i in range(self.num_keypoints) if i not in self.hand_joint_indices]

        mask[:, :, body_indices] = rand_body[:, :, body_indices] < self.mask_ratio
        mask[:, :, self.hand_joint_indices] = rand_hand[:, :, self.hand_joint_indices] < self.hand_mask_ratio

        return mask

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        salient_hand_masking: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Self-Supervised MAE Forward Step:
        features: [B, T, K, C]
        """
        B, T, K, C = features.shape
        device = features.device

        # 1. Generate Spatiotemporal Binary Mask
        spatio_mask = self.generate_spatiotemporal_mask(
            batch_size=B,
            seq_len=T,
            device=device,
            salient_hand_masking=salient_hand_masking,
        )  # [B, T, K]

        # 2. Corrupt Visible Inputs (Zero-out masked keypoints)
        masked_features = features.clone()
        masked_features[spatio_mask.unsqueeze(-1).expand_as(features)] = 0.0

        # 3. Encoder Forward Pass over Corrupted Inputs
        enc_out = self.encoder(
            input_x=masked_features,
            mask=mask,
            frame_indices=frame_indices,
        )
        h_seq = enc_out["h_seq"]  # [B, T_enc, d_enc]

        # 4. Decoder Reconstruction Pass
        dec_h = self.decoder_stem(h_seq)
        dec_h = self.decoder_transformer(dec_h)
        recon_flat = self.reconstruction_head(dec_h)  # [B, T_enc, K * C]

        # Resize to original time resolution T if downsampled
        if recon_flat.size(1) != T:
            recon_flat = F.interpolate(
                recon_flat.transpose(1, 2),
                size=T,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)

        recon_features = recon_flat.view(B, T, K, self.channels_per_kp)

        # 5. Compute Kinematics & Coordinate Losses
        # A. Position Coordinate Loss (MSE on masked entries)
        pos_target = features[..., :3]
        pos_pred = recon_features[..., :3]
        pos_mask = spatio_mask.unsqueeze(-1).expand_as(pos_target)

        loss_pos = F.mse_loss(pos_pred[pos_mask], pos_target[pos_mask])

        # B. Velocity Kinematics Loss (first temporal difference)
        loss_vel = torch.tensor(0.0, device=device)
        if T > 1:
            vel_target = pos_target[:, 1:] - pos_target[:, :-1]
            vel_pred = pos_pred[:, 1:] - pos_pred[:, :-1]
            vel_mask = pos_mask[:, 1:]
            loss_vel = F.mse_loss(vel_pred[vel_mask], vel_target[vel_mask])

        total_mae_loss = loss_pos + self.lambda_vel * loss_vel

        # Fraction of spatiotemporal points masked
        actual_mask_ratio = spatio_mask.float().mean().item()

        return {
            "loss": total_mae_loss,
            "loss_pos": loss_pos,
            "loss_vel": loss_vel,
            "actual_mask_ratio": actual_mask_ratio,
            "reconstructed_features": recon_features,
        }
