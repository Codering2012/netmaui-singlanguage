#!/usr/bin/env python3
"""
================================================================================
DYNAMIC 3D LOCUS NEURAL MEMORY BANK (V3 ARCHITECTURE)
================================================================================
Resolves 3D spatial locus erasure in ASL discourse by decoupling anatomical
motion normalization from persistent global spatial referencing.

Maintains a differentiable memory bank M_locus in a signer-anchored
torso-oriented cylindrical coordinate frame (r, theta, y_rel) to track
established discourse entities (people, objects, locations) across multi-sentence
signing sequences, enabling accurate agreement verb querying (GIVE, SHOW, ASK).
================================================================================
"""

from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dynamic3DLocusMemoryBank(nn.Module):
    r"""
    Differentiable 3D Spatial Locus Memory Bank.
    
    Args:
        d_model: Latent feature dimension (aligned to multiples of 128 for TPU v5e).
        num_slots: Number of discrete spatial sectors in signing hemisphere (default 8).
        sigma: Gaussian bandwidth for spatial sector assignment.
        ema_decay: Momentum update rate for slot representations.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_slots: int = 8,
        sigma: float = 0.35,
        ema_decay: float = 0.90,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.sigma = sigma
        self.ema_decay = ema_decay

        # Predefined canonical spatial sector centroids in cylindrical space (r, theta, y_rel)
        # Angles span -120 to +120 degrees around the front torso hemisphere
        angles = torch.linspace(-2.0 * math.pi / 3.0, 2.0 * math.pi / 3.0, num_slots)
        radii = torch.full((num_slots,), 0.6)  # ~60cm reach
        heights = torch.tensor([-0.2, 0.0, 0.2, -0.2, 0.0, 0.2, -0.1, 0.1][:num_slots])
        canonical_centroids = torch.stack([radii, angles, heights], dim=-1)  # [K, 3]
        self.register_buffer("canonical_centroids", canonical_centroids)

        # Slot persistent entity vectors
        self.initial_slots = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)

        # Gated write projection
        self.write_gate = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        # Entity transformation projection
        self.entity_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Directional agreement verb query cross-attention
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.val_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _compute_torso_cylindrical_coordinates(
        self,
        hand_pos: torch.Tensor,
        shoulders: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Converts Cartesian hand coordinates to Torso-Anchored Cylindrical Coordinates:
        (r, theta, y_rel) relative to the bi-acromial shoulder orientation.
        """
        # hand_pos: [B, T, 3] (x, y, z)
        if shoulders is not None:
            # shoulders: [B, T, 2, 3] (0: left, 1: right)
            left_sh = shoulders[:, :, 0, :]
            right_sh = shoulders[:, :, 1, :]
            torso_center = (left_sh + right_sh) * 0.5
            dx = right_sh[:, :, 0] - left_sh[:, :, 0]
            dz = right_sh[:, :, 2] - left_sh[:, :, 2]
            theta_torso = torch.atan2(dz, dx + 1e-6)
        else:
            torso_center = torch.zeros_like(hand_pos)
            theta_torso = torch.zeros(hand_pos.shape[:2], device=hand_pos.device, dtype=hand_pos.dtype)

        rel_pos = hand_pos - torso_center
        rx = rel_pos[:, :, 0]
        ry = rel_pos[:, :, 1]
        rz = rel_pos[:, :, 2]

        r = torch.sqrt(rx ** 2 + rz ** 2 + 1e-6)
        theta = torch.atan2(rz, rx + 1e-6) - theta_torso
        # Wrap theta to [-pi, pi]
        theta = (theta + math.pi) % (2.0 * math.pi) - math.pi

        return torch.stack([r, theta, ry], dim=-1)  # [B, T, 3]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hand_coords: Optional[torch.Tensor] = None,
        shoulder_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with soft continuous slot writing and directional query reading.
        
        Args:
            hidden_states: [B, T, d_model] Encoder sequence.
            hand_coords: [B, T, 3] Cartesian coordinates of dominant hand/pointing index.
            shoulder_coords: [B, T, 2, 3] Left and right shoulder coordinates.
            
        Returns:
            enhanced_hidden: [B, T, d_model] Spatially grounded representations.
            aux_losses: Dictionary containing locus auxiliary consistency loss.
        """
        B, T, D = hidden_states.shape

        if hand_coords is None:
            # Fallback: Zero-gradient pass through identity
            return hidden_states, {"loss_locus": torch.zeros((), device=hidden_states.device)}

        # 1. Map hand coordinates into cylindrical torso-aligned space
        cyl_coords = self._compute_torso_cylindrical_coordinates(hand_coords, shoulder_coords)  # [B, T, 3]

        # 2. Compute soft Gaussian proximity to canonical spatial slots
        # cyl_coords: [B, T, 1, 3], canonical_centroids: [1, 1, K, 3]
        diff = cyl_coords.unsqueeze(2) - self.canonical_centroids.view(1, 1, self.num_slots, 3)
        dist_sq = torch.sum(diff ** 2, dim=-1)  # [B, T, K]
        spatial_affinity = F.softmax(-dist_sq / (2.0 * (self.sigma ** 2)), dim=-1)  # [B, T, K]

        # 3. Dynamic write gating: only write if gesture exhibits indexing or entity hold
        write_input = torch.cat([hidden_states, cyl_coords], dim=-1)  # [B, T, D + 3]
        write_prob = self.write_gate(write_input)  # [B, T, 1]
        effective_write = spatial_affinity * write_prob  # [B, T, K]

        # 4. Aggregate entity writes across temporal steps without unrolled loops
        # [B, K, T] @ [B, T, D] -> [B, K, D]
        projected_entity = self.entity_proj(hidden_states)  # [B, T, D]
        slot_updates = torch.bmm(effective_write.transpose(1, 2), projected_entity)  # [B, K, D]
        slot_normalizer = torch.sum(effective_write, dim=1, keepdim=True).transpose(1, 2) + 1e-5  # [B, K, 1]
        normalized_updates = slot_updates / slot_normalizer  # [B, K, D]

        # Initialize slots batch-wise
        current_slots = self.initial_slots.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        # Soft update with EMA blend
        updated_slots = self.ema_decay * current_slots + (1.0 - self.ema_decay) * normalized_updates  # [B, K, D]

        # 5. Cross-attention reading: directional verbs attend to persistent locus slots
        Q = self.query_proj(hidden_states)  # [B, T, D]
        K_mat = self.key_proj(updated_slots)  # [B, K, D]
        V_mat = self.val_proj(updated_slots)  # [B, K, D]

        scale = 1.0 / math.sqrt(D)
        attn_scores = torch.bmm(Q, K_mat.transpose(1, 2)) * scale  # [B, T, K]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, T, K]
        locus_context = torch.bmm(attn_weights, V_mat)  # [B, T, D]

        # 6. Residual integration
        enhanced_hidden = self.norm(hidden_states + self.out_proj(locus_context))

        # Auxiliary loss: spatial diversity penalty preventing slot representation collapse
        # Compute slot cosine similarity matrix: should be orthogonal
        norm_slots = F.normalize(updated_slots, p=2, dim=-1)
        sim_matrix = torch.bmm(norm_slots, norm_slots.transpose(1, 2))  # [B, K, K]
        eye = torch.eye(self.num_slots, device=hidden_states.device).unsqueeze(0)
        diversity_loss = torch.mean((sim_matrix - eye) ** 2)

        return enhanced_hidden, {"loss_locus": diversity_loss * 0.1}
