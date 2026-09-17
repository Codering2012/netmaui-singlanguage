#!/usr/bin/env python3
"""
================================================================================
KINEMATIC MOVEMENT EPENTHESIS SUPPRESSOR (KMES)
================================================================================
Solves the fundamental "Transition Hallucination" problem in continuous sign language:
Non-linguistic transitional hand repositioning movements between signs (Movement Epenthesis)
frequently look like signs, tricking CTC and autoregressive decoders into emitting false
spurious words ("NAME", "TELL", "COME").

Grounded in Flash & Hogan (1985) Minimum-Jerk Motor Control Law:
1. Ballistic repositioning movements exhibit a single bell-shaped velocity peak with
   zero finger internal articulation entropy (relaxed passive handshape).
2. Intentional lexical signs exhibit target holds, deceleration plateaus, and high
   finger articulation complexity.

KMES calculates a continuous Ballistic Transition Score beta_t and injects an adaptive
CTC Blank Token Bias, forcing decoders to emit <BLANK> during transitions.
================================================================================
"""

from typing import Tuple, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MovementEpenthesisSuppressor(nn.Module):
    r"""
    Detects non-gestural inter-sign movement epenthesis and suppresses transition hallucinations.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_keypoints: int = 60,
        d_model: int = 128,
        blank_bias_strength: float = 10.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.blank_bias_strength = blank_bias_strength

        # Lightweight kinematic trajectory encoder for transition discrimination
        self.temporal_gate = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Initialize gate with negative prior bias to suppress false background activations
        nn.init.normal_(self.temporal_gate[3].weight, std=0.01)
        nn.init.constant_(self.temporal_gate[3].bias, -2.5)

    def extract_transition_signatures(self, kinematics: torch.Tensor) -> torch.Tensor:
        r"""
        Extracts 6 physical transition signals per frame:
        0: Wrist translational speed ||v_wrist||
        1: Finger internal dispersion velocity sum ||v_fingers - v_wrist||
        2: Wrist jerk norm ||d3x/dt3|| (rate of change of acceleration)
        3: Ratio of wrist velocity to total finger velocity
        4: Curvature kappa of wrist trajectory
        5: Kinetic hold plateau indicator (1 if velocity < 0.05 m/s)
        """
        B, T = kinematics.shape[:2]
        pts = kinematics.view(B, T, self.num_keypoints, -1)
        pos = pts[..., :3]
        vel = pts[..., 3:6]
        acc = pts[..., 6:9] if pts.shape[-1] >= 9 else torch.diff(vel, dim=1, prepend=vel[:, :1])

        # Right wrist (21) and Left wrist (0)
        r_wrist_vel = torch.norm(vel[:, :, 21, :], dim=-1)
        l_wrist_vel = torch.norm(vel[:, :, 0, :], dim=-1)
        wrist_speed = torch.maximum(r_wrist_vel, l_wrist_vel)  # [B, T]

        # Finger internal motion (keypoints 1-20 for Left, 22-41 for Right)
        r_finger_vel = torch.norm(vel[:, :, 22:42, :] - vel[:, :, 21:22, :], dim=-1).mean(dim=-1)
        l_finger_vel = torch.norm(vel[:, :, 1:21, :] - vel[:, :, 0:1, :], dim=-1).mean(dim=-1)
        finger_speed = torch.maximum(r_finger_vel, l_finger_vel)  # [B, T]

        # Jerk norm (derivative of acceleration)
        jerk = torch.diff(acc, dim=1, prepend=acc[:, :1])
        r_wrist_jerk = torch.norm(jerk[:, :, 21, :], dim=-1)
        l_wrist_jerk = torch.norm(jerk[:, :, 0, :], dim=-1)
        wrist_jerk = torch.maximum(r_wrist_jerk, l_wrist_jerk)  # [B, T]

        # Ballistic Ratio: High when hand translates rapidly with passive fingers
        ballistic_ratio = wrist_speed / (wrist_speed + finger_speed + 1e-4)

        # Kinetic Presence Gate: Eliminates 0/0 indeterminate singularities during stationary holds
        kinetic_presence = torch.tanh(wrist_speed / 0.15)  # [B, T]

        # Curvature kappa: ||v x a|| / ||v||^3 on the active translating hand (bilateral support)
        use_rh = (r_wrist_vel >= l_wrist_vel).unsqueeze(-1)
        active_vel = torch.where(use_rh, vel[:, :, 21, :], vel[:, :, 0, :])
        active_acc = torch.where(use_rh, acc[:, :, 21, :], acc[:, :, 0, :])
        v_cross_a = torch.cross(active_vel, active_acc, dim=-1)
        curvature = torch.norm(v_cross_a, dim=-1) / (wrist_speed ** 3 + 1e-3)
        curvature = torch.clamp(curvature, max=10.0)

        # Hold plateau indicator: 1 if hands are holding pose
        hold_indicator = torch.sigmoid(20.0 * (0.06 - wrist_speed))

        signatures = torch.stack([
            wrist_speed,
            finger_speed,
            wrist_jerk,
            ballistic_ratio,
            curvature,
            hold_indicator,
        ], dim=1)  # [B, 6, T]

        return signatures

    def compute_epenthesis_probability(self, kinematics: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the frame-level Movement Epenthesis probability beta_t in [0, 1].
        High beta_t indicates a non-linguistic transition stroke.
        """
        signatures = self.extract_transition_signatures(kinematics)  # [B, 6, T]
        learned_gate = self.temporal_gate(signatures).squeeze(1)     # [B, T]

        # Combine with explicit physical heuristic prior:
        # High translational kinetic presence + passive fingers + NOT in a hold plateau
        wrist_speed = signatures[:, 0, :]
        kinetic_presence = torch.tanh(wrist_speed / 0.15)
        ballistic_ratio = signatures[:, 3, :]
        hold_indicator = signatures[:, 5, :]
        heuristic_prior = kinetic_presence * ballistic_ratio * (1.0 - hold_indicator)

        # Smooth combined transition score
        beta_t = 0.25 * learned_gate + 0.75 * heuristic_prior
        return beta_t

    def compute_consistency_loss(self, kinematics: torch.Tensor) -> torch.Tensor:
        r"""
        Self-supervised consistency loss aligning the learned neural temporal gate
        with the Flash & Hogan (1985) Minimum-Jerk kinematic heuristic prior:
            L_consistency = BCE(learned_gate, heuristic_prior.detach())
        """
        signatures = self.extract_transition_signatures(kinematics)
        learned_gate = self.temporal_gate(signatures).squeeze(1)
        wrist_speed = signatures[:, 0, :]
        kinetic_presence = torch.tanh(wrist_speed / 0.15)
        ballistic_ratio = signatures[:, 3, :]
        hold_indicator = signatures[:, 5, :]
        heuristic_prior = kinetic_presence * ballistic_ratio * (1.0 - hold_indicator)
        return F.binary_cross_entropy(learned_gate, heuristic_prior.detach())

    def apply_ctc_blank_bias(
        self,
        ctc_logits: torch.Tensor,     # [B, T, V] where 0 is BLANK
        kinematics: torch.Tensor,     # [B, T, 60*9]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Biases CTC logits during movement epenthesis transitions to force <BLANK> emission.
        
        Returns:
            biased_logits: [B, T, V]
            beta_t: [B, T] epenthesis probability
        """
        beta_t = self.compute_epenthesis_probability(kinematics)  # [B, T]
        biased_logits = ctc_logits.clone()
        # Full contrastive logit shift: boost BLANK and penalize non-blank tokens equally.
        # Crucial: detach beta_t so CTC loss does not backpropagate parasitic gradients into the kinematic gate.
        boost = (self.blank_bias_strength * beta_t.detach()).unsqueeze(-1)  # [B, T, 1]
        biased_logits[:, :, 0:1] += boost
        biased_logits[:, :, 1:] -= boost
        return biased_logits, beta_t
