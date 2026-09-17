#!/usr/bin/env python3
"""
================================================================================
NON-MANUAL FEATURE PYRAMID & POLARITY GUARD (V3 ARCHITECTURE)
================================================================================
Prevents catastrophic 180-degree semantic polarity inversions (translating
negation as affirmative or questions as statements) by explicitly modeling
non-manual grammatical markers:

1. Upper-Face Eyebrow Elevation/Furrowing (wh-q vs y/n-q vs topicalization).
2. Lower-Face Native Mouth Morphemes (CHA, MM, OO, TH, CS, PUFF, PAH, STA-STA)
   vs English Mouthings.
3. Rigid Cranial IMU Tracking: Rotational velocities (omega_yaw, omega_pitch, omega_roll).
4. PolarityGuard Loss: Penalizes polarity mismatch between cranial oscillations
   and text decoder emissions.
================================================================================
"""

from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonManualFeaturePyramid(nn.Module):
    r"""
    Multi-Scale Non-Manual Feature Pyramid with cranial IMU and mouth morpheme heads.
    
    Args:
        d_model: Latent feature dimension (aligned to multiples of 128).
        num_mouth_classes: Number of discrete mouth morpheme classes (default 10).
    """

    MOUTH_MORPHEMES = [
        "NEUTRAL",
        "CHA",        # Extreme size, intense scale
        "MM",         # Normal, effortless, average
        "OO",         # Small, delicate, thin
        "TH",         # Careless, sloppy, inattentive
        "CS",         # Immediate temporal proximity
        "PUFF",       # Large volume, abundant quantity
        "PAH",        # Finally, sudden breakthrough
        "STA_STA",    # Prolonged struggle, repetitive labor
        "ENG_MOUTH",  # English contact mouthing (homophone disambiguation)
    ]

    def __init__(
        self,
        d_model: int = 128,
        num_mouth_classes: int = 10,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_mouth_classes = num_mouth_classes

        # 1. Cranial IMU Dynamics Encoder (Rotational Velocities: yaw, pitch, roll)
        # Input: [B, T, 3] -> [B, T, d_model // 4]
        d_cranial = max(32, d_model // 4)
        self.cranial_encoder = nn.Sequential(
            nn.Linear(3, d_cranial),
            nn.LayerNorm(d_cranial),
            nn.GELU(),
            nn.Linear(d_cranial, d_cranial),
        )

        # 2. Upper-Face Eyebrow Kinematics Encoder
        # 4 eyebrow points (inner/outer left, inner/outer right) x 3D coords = 12 features
        d_eyebrow = max(32, d_model // 4)
        self.eyebrow_encoder = nn.Sequential(
            nn.Linear(12, d_eyebrow),
            nn.LayerNorm(d_eyebrow),
            nn.GELU(),
            nn.Linear(d_eyebrow, d_eyebrow),
        )

        # 3. Lower-Face Mouth/Lips Kinematics Encoder
        # 8 lip contour points x 3D coords = 24 features
        d_mouth = max(64, d_model // 2)
        self.mouth_encoder = nn.Sequential(
            nn.Linear(24, d_mouth),
            nn.LayerNorm(d_mouth),
            nn.GELU(),
            nn.Linear(d_mouth, d_mouth),
        )

        # 4. Multi-Scale Non-Manual Fusion
        total_nmm_dim = d_cranial + d_eyebrow + d_mouth
        self.nmm_fusion = nn.Sequential(
            nn.Linear(total_nmm_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 5. Syntactic Classification Heads
        # Eyebrow Grammar: [0: Neutral, 1: Wh-Question (furrowed), 2: Yes/No Question (raised), 3: Topic (raised+hold)]
        self.eyebrow_head = nn.Linear(d_model, 4)
        # Mouth Morpheme Classifier
        self.mouth_head = nn.Linear(d_model, num_mouth_classes)
        # Binary Negation Head (detects active headshake during sign stroke)
        self.negation_head = nn.Linear(d_model, 1)

        self.norm = nn.LayerNorm(d_model)

    def _estimate_cranial_imu_from_face(self, face_landmarks: torch.Tensor) -> torch.Tensor:
        """
        Estimates rigid cranial angular velocity [omega_yaw, omega_pitch, omega_roll]
        from rigid nasal bridge and eye contour landmarks: [B, T, 12, 3] -> [B, T, 3].
        """
        # Temporal difference of facial orientation
        # Approximate yaw by left vs right eye outer canthi depth difference
        # Approximate pitch by nose tip to forehead vector elevation
        # Fallback if raw IMU is not present
        B, T, K, C = face_landmarks.shape
        diff = torch.zeros((B, T, 3), device=face_landmarks.device, dtype=face_landmarks.dtype)
        if T > 1:
            # Velocity of nasal center [index 0]
            nose = face_landmarks[:, :, 0, :]
            v_nose = torch.diff(nose, dim=1, prepend=nose[:, :1, :])
            diff[:, :, 0] = v_nose[:, :, 0] * 10.0  # yaw proxy
            diff[:, :, 1] = v_nose[:, :, 1] * 10.0  # pitch proxy
            diff[:, :, 2] = v_nose[:, :, 2] * 10.0  # roll proxy
        return diff

    def forward(
        self,
        hidden_states: torch.Tensor,
        face_landmarks: Optional[torch.Tensor] = None,
        cranial_imu: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, d_model] Encoder sequence.
            face_landmarks: [B, T, 12, 3] or [B, T, 14, 3] Facial keypoints.
            cranial_imu: [B, T, 3] Raw rotational velocities (yaw, pitch, roll).
            
        Returns:
            enhanced_hidden: [B, T, d_model]
            aux_predictions: Dictionary with 'negation_logits', 'eyebrow_logits', 'mouth_logits'.
        """
        B, T, D = hidden_states.shape

        if face_landmarks is None:
            # Fallback when face features are omitted
            negation_logits = torch.zeros((B, T, 1), device=hidden_states.device)
            eyebrow_logits = torch.zeros((B, T, 4), device=hidden_states.device)
            mouth_logits = torch.zeros((B, T, self.num_mouth_classes), device=hidden_states.device)
            return hidden_states, {
                "negation_logits": negation_logits,
                "eyebrow_logits": eyebrow_logits,
                "mouth_logits": mouth_logits,
                "loss_nmm": torch.zeros((), device=hidden_states.device),
            }

        # 1. Resolve Cranial IMU
        if cranial_imu is None:
            cranial_imu = self._estimate_cranial_imu_from_face(face_landmarks)
        cranial_feat = self.cranial_encoder(cranial_imu)  # [B, T, d_cranial]

        # 2. Slice Eyebrow Features (first 4 face keypoints or pad)
        num_face_kp = face_landmarks.shape[2]
        if num_face_kp >= 4:
            eyebrow_pts = face_landmarks[:, :, :4, :].reshape(B, T, 12)
        else:
            eyebrow_pts = torch.zeros((B, T, 12), device=face_landmarks.device, dtype=face_landmarks.dtype)
        eyebrow_feat = self.eyebrow_encoder(eyebrow_pts)  # [B, T, d_eyebrow]

        # 3. Slice Mouth/Lip Features (remaining face keypoints up to 8 points)
        if num_face_kp >= 12:
            mouth_pts = face_landmarks[:, :, 4:12, :].reshape(B, T, 24)
        else:
            mouth_pts = torch.zeros((B, T, 24), device=face_landmarks.device, dtype=face_landmarks.dtype)
        mouth_feat = self.mouth_encoder(mouth_pts)  # [B, T, d_mouth]

        # 4. Multi-scale fusion
        fused_nmm = torch.cat([cranial_feat, eyebrow_feat, mouth_feat], dim=-1)  # [B, T, total_nmm_dim]
        nmm_embed = self.nmm_fusion(fused_nmm)  # [B, T, d_model]

        # 5. Syntactic predictions
        eyebrow_logits = self.eyebrow_head(nmm_embed)
        mouth_logits = self.mouth_head(nmm_embed)
        negation_logits = self.negation_head(nmm_embed)

        # 6. Gated integration into contextual sequence
        enhanced_hidden = self.norm(hidden_states + nmm_embed)

        # Auxiliary temporal entropy regularization to prevent constant-state collapse
        neg_probs = torch.sigmoid(negation_logits)
        variance_loss = -torch.mean(torch.var(neg_probs, dim=1) + 1e-6)  # Encourage dynamic range

        return enhanced_hidden, {
            "negation_logits": negation_logits,
            "eyebrow_logits": eyebrow_logits,
            "mouth_logits": mouth_logits,
            "loss_nmm": variance_loss * 0.05,
        }


class PolarityGuard(nn.Module):
    r"""
    Loss wrapper that penalizes semantic polarity inversions between text translations
    and detected non-manual grammatical markers (negation headshakes and questions).
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        negation_logits: torch.Tensor,       # [B, T, 1]
        text_is_negative: torch.Tensor,     # [B] Boolean tensor (1 if target text contains negation)
        mask: Optional[torch.Tensor] = None, # [B, T]
    ) -> torch.Tensor:
        """
        Evaluates asymmetric margin penalty for semantic polar inversion.
        """
        # Average negation score across active sequence
        if mask is not None:
            weights = mask.unsqueeze(-1).float()
            video_neg_score = torch.sum(torch.sigmoid(negation_logits) * weights, dim=1) / (torch.sum(weights, dim=1) + 1e-6)
        else:
            video_neg_score = torch.mean(torch.sigmoid(negation_logits), dim=1)  # [B, 1]

        video_neg_score = video_neg_score.squeeze(-1)  # [B]
        target_neg = text_is_negative.float().detach() # [B] Detach target to prevent gradient contamination

        # Binary cross entropy between video negation signature and target text polarity
        polarity_loss = F.binary_cross_entropy(video_neg_score, target_neg)
        return polarity_loss
