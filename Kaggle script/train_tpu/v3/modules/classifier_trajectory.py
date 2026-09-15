#!/usr/bin/env python3
"""
================================================================================
DECONSTRUCTIVE CLASSIFIER TRAJECTORY FIELD (V3 ARCHITECTURE)
================================================================================
Resolves the inability of discrete vocabularies and CTC heads to capture
poly-morphemic classifier predicates (depicting signs of continuous motion,
speed, spatial orientation, and obstacle avoidance).

Decomposes depictions into three simultaneous linguistic streams:
1. Discrete Base Handshape Morpheme (CL:3 Vehicle, CL:1 Person, CL:C Container, etc.)
2. Continuous 3D Spatial Trajectory Field (Tangent Velocity, Curvature, Normal Vector)
3. Topological Interaction Manifold (Dual-hand distance, contact surface proximity)
================================================================================
"""

from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeconstructiveClassifierField(nn.Module):
    r"""
    Deconstructive Decomposition for Poly-morphemic Classifier Predicates.
    
    Args:
        d_model: Latent feature dimension (aligned to multiples of 128 for TPU v5e).
        num_classifier_types: Number of base canonical depicting handshapes (default 16).
        future_steps: Future trajectory prediction horizon for Contrastive Predictive Coding (CPC).
    """

    CLASSIFIER_TYPES = [
        "NONE",
        "CL_3_VEHICLE",         # Land/water vehicle
        "CL_1_PERSON_UPRIGHT",   # Individual upright person
        "CL_V_PERSON_BENT",     # Sitting person or small animal
        "CL_C_CONTAINER",       # Cylindrical object (cup, bottle, pipe)
        "CL_B_SURFACE_FLAT",    # Sheet, paper, tabletop, wall
        "CL_5_CLAW_BALL",       # Spherical object, clustered group
        "CL_F_SMALL_FLAT",      # Coin, button, small round mark
        "CL_G_THIN_STRIP",      # Thin dimension, small interval
        "CL_ILY_AIRPLANE",      # Airborne flight vehicle
        "CL_L_RECTANGLE",       # Picture frame, check, card
        "CL_O_COMPACT",         # Small dense package, pebble
        "CL_U_FLAT_STRIP",      # Ribbon, tongue, bandage
        "CL_4_PARALLEL",        # Line of people, flowing water, fence
        "CL_S_HEAD_HEAVY",      # Fixed heavy solid, fist, statue
        "CL_OPEN_A_BUILDING",   # House, structure, stationary landmark
    ]

    def __init__(
        self,
        d_model: int = 128,
        num_classifier_types: int = 16,
        future_steps: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classifier_types = num_classifier_types
        self.future_steps = future_steps

        # 1. Discrete Base Handshape Classifier Head
        self.handshape_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_classifier_types),
        )

        # 2. Continuous 3D Spatial Trajectory Estimator
        # Outputs: [Tangent Velocity (3), Curvature (1), Acceleration (3)] = 7 dims
        self.trajectory_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 7),
        )

        # 3. Topological Interaction Manifold Estimator
        # Outputs: [Dual-hand distance (1), Contact probability (1), Height relative to sternum (1)] = 3 dims
        self.topology_regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 3),
        )

        # 4. Dense Multimodal Continuous Fusion Projection
        # Projects all geometric and morphemic streams back into d_model
        total_stream_dim = num_classifier_types + 7 + 3
        self.deconstruction_fusion = nn.Sequential(
            nn.Linear(d_model + total_stream_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # 5. Vectorized Contrastive Predictive Coding (CPC) Projection for Future Trajectory
        # Fused projection predicts all future horizons simultaneously without unrolled Python loops
        self.cpc_head = nn.Linear(d_model, future_steps * 3, bias=False)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        hand_positions: Optional[torch.Tensor] = None,
        base_hand_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, d_model] Contextual sequence.
            hand_positions: [B, T, 3] Dominant hand 3D coordinates.
            base_hand_positions: [B, T, 3] Non-dominant hand 3D coordinates.
            
        Returns:
            enhanced_hidden: [B, T, d_model]
            aux_losses: Dictionary containing CPC trajectory loss and stream predictions.
        """
        B, T, D = hidden_states.shape

        # Stream 1: Base Handshape Morpheme Logits
        handshape_logits = self.handshape_classifier(hidden_states)  # [B, T, num_classifier_types]
        handshape_probs = F.softmax(handshape_logits, dim=-1)

        # Stream 2: Continuous 3D Spatial Trajectory Field
        traj_features = self.trajectory_regressor(hidden_states)  # [B, T, 7]

        # Stream 3: Topological Interaction Manifold
        topo_features = self.topology_regressor(hidden_states)  # [B, T, 3]

        # Dense Fusion
        concat_streams = torch.cat([hidden_states, handshape_probs, traj_features, topo_features], dim=-1)
        projected_stream = self.deconstruction_fusion(concat_streams)  # [B, T, d_model]
        enhanced_hidden = self.norm(hidden_states + projected_stream)

        # Vectorized CPC Auxiliary Trajectory Loss (Zero dynamic shapes for Cloud TPU v5e)
        loss_cpc = torch.zeros((), device=hidden_states.device)
        if hand_positions is not None and T > self.future_steps:
            # Predict all future velocities [B, T, K, 3]
            pred_all = self.cpc_head(hidden_states).view(B, T, self.future_steps, 3)

            # Velocity ground truth [B, T, 3]
            gt_vel = torch.diff(hand_positions, dim=1, prepend=hand_positions[:, :1, :])
            gt_padded = F.pad(gt_vel, (0, 0, 0, self.future_steps))  # [B, T + K, 3]

            # Vectorized static-shape future targets: [B, T, K, 3]
            future_targets = torch.stack(
                [gt_padded[:, k : k + T, :] for k in range(1, self.future_steps + 1)], dim=2
            ).detach()

            # Static valid mask where t + k < T: [1, T, K]
            t_indices = torch.arange(T, device=hidden_states.device).view(1, T, 1)
            k_offsets = torch.arange(1, self.future_steps + 1, device=hidden_states.device).view(1, 1, self.future_steps)
            valid_mask = (t_indices + k_offsets < T).float()

            sq_err = torch.sum((pred_all - future_targets) ** 2, dim=-1)  # [B, T, K]
            masked_err = sq_err * valid_mask                              # [B, T, K]
            k_counts = torch.sum(valid_mask, dim=1)                       # [1, K]
            loss_per_k = torch.sum(masked_err, dim=(0, 1)) / (k_counts.squeeze(0) * B * 3)  # [K]
            loss_cpc = torch.mean(loss_per_k)

        # Topological Reconstruction Loss if dual hands are present
        loss_topo = torch.zeros((), device=hidden_states.device)
        if hand_positions is not None and base_hand_positions is not None:
            actual_dist = torch.norm(hand_positions - base_hand_positions, dim=-1, keepdim=True).detach()  # [B, T, 1] Detach target distance
            pred_dist = F.softplus(topo_features[:, :, :1])  # Predicted distance must be positive
            loss_topo = F.mse_loss(pred_dist, actual_dist)

        total_aux_loss = loss_cpc * 0.1 + loss_topo * 0.05

        return enhanced_hidden, {
            "handshape_logits": handshape_logits,
            "traj_features": traj_features,
            "topo_features": topo_features,
            "loss_classifier_cpc": total_aux_loss,
        }
