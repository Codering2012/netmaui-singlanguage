#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — PHONOLOGICAL TRIAD HARD NEGATIVE MINING ENGINE
================================================================================
Implements Stokoe/Battison Phonological Decomposition & Minimal-Pair Hard Mining:
1. Phonological Triad Subspaces:
     - Handshape Subspace: Finger joint angles and finger curl geometry.
     - Location Subspace:  Spatial coordinates relative to torso/head anchors.
     - Movement Subspace:  Kinematic velocity and trajectory inflections.
2. Minimal-Pair Hard Negative Mining:
     Mines negatives that are phonologically closest to the anchor (differing by
     only 1 parameter, e.g. "MOTHER" vs "FATHER") to prevent semantic collapse.
3. Triad Margin Contrastive Regularizer:
     L_triad = max(0, D(z_a, z_p) - D(z_a, z_hard_neg) + margin)
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriadMiningOutput(NamedTuple):
    triad_loss: torch.Tensor             # Scalar margin ranking loss
    mined_negative_indices: torch.Tensor # [B] Hardest negative index per anchor
    hard_neg_distance: torch.Tensor      # [B] Distance to hardest negative
    pos_distance: torch.Tensor           # [B] Distance to positive anchor


class ASLPhonologicalHardNegativeMiner(nn.Module):
    """
    Phonological Triad Decomposition and Hard Negative Mining Module.
    """

    def __init__(
        self,
        d_model: int = 128,
        triad_dim: int = 64,
        margin: float = 0.30,
        weight_shape: float = 0.40,
        weight_loc: float = 0.30,
        weight_move: float = 0.30,
    ):
        super().__init__()
        self.d_model = d_model
        self.triad_dim = triad_dim
        self.margin = margin
        self.w_shape = weight_shape
        self.w_loc = weight_loc
        self.w_move = weight_move

        # Subspace Projections for the Phonological Triad
        self.shape_proj = nn.Sequential(
            nn.Linear(d_model, triad_dim),
            nn.LayerNorm(triad_dim),
            nn.GELU(),
            nn.Linear(triad_dim, triad_dim),
        )
        self.loc_proj = nn.Sequential(
            nn.Linear(d_model, triad_dim),
            nn.LayerNorm(triad_dim),
            nn.GELU(),
            nn.Linear(triad_dim, triad_dim),
        )
        self.move_proj = nn.Sequential(
            nn.Linear(d_model, triad_dim),
            nn.LayerNorm(triad_dim),
            nn.GELU(),
            nn.Linear(triad_dim, triad_dim),
        )

    def decompose_triad(
        self,
        h: torch.Tensor,  # [B, d_model]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decomposes holistic token embedding into the 3 normalized phonological subspaces.
        """
        z_shape = F.normalize(self.shape_proj(h), p=2, dim=-1)
        z_loc = F.normalize(self.loc_proj(h), p=2, dim=-1)
        z_move = F.normalize(self.move_proj(h), p=2, dim=-1)
        return z_shape, z_loc, z_move

    def compute_pairwise_triad_distance(
        self,
        z_shape: torch.Tensor,  # [B, triad_dim]
        z_loc: torch.Tensor,    # [B, triad_dim]
        z_move: torch.Tensor,   # [B, triad_dim]
    ) -> torch.Tensor:
        """
        Computes pairwise phonological distance matrix D [B, B].
        """
        # Pairwise Euclidean distances: || u - v ||_2
        dist_shape = torch.cdist(z_shape, z_shape, p=2.0)
        dist_loc = torch.cdist(z_loc, z_loc, p=2.0)
        dist_move = torch.cdist(z_move, z_move, p=2.0)

        total_dist = self.w_shape * dist_shape + self.w_loc * dist_loc + self.w_move * dist_move
        return total_dist

    def forward(
        self,
        anchor_embeddings: torch.Tensor,   # [B, d_model]
        positive_embeddings: torch.Tensor, # [B, d_model]
        labels: torch.Tensor,              # [B] Discrete gloss IDs
    ) -> TriadMiningOutput:
        """
        Mines hardest phonological negatives and computes Triad Margin Loss.
        """
        B = anchor_embeddings.size(0)
        device = anchor_embeddings.device

        # 1. Decompose into Phonological Subspaces
        a_shape, a_loc, a_move = self.decompose_triad(anchor_embeddings)
        p_shape, p_loc, p_move = self.decompose_triad(positive_embeddings)

        # 2. Compute Positive Distances (Anchor to Augmented Positive)
        d_pos = (
            self.w_shape * torch.norm(a_shape - p_shape, p=2, dim=-1) +
            self.w_loc * torch.norm(a_loc - p_loc, p=2, dim=-1) +
            self.w_move * torch.norm(a_move - p_move, p=2, dim=-1)
        )  # [B]

        # 3. Pairwise Phonological Distance Matrix for Negative Mining
        dist_matrix = self.compute_pairwise_triad_distance(a_shape, a_loc, a_move)  # [B, B]

        # Mask out self (diagonal) and same-label positive samples
        label_match = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        masked_dist = dist_matrix.clone()
        masked_dist[label_match] = float("inf")

        # 4. Mine Hardest Negative (Minimal phonological distance with different label)
        hard_neg_dist, hard_neg_idx = torch.min(masked_dist, dim=-1)  # [B], [B]

        # Handle case where batch has no differing labels
        invalid_mask = torch.isinf(hard_neg_dist)
        if invalid_mask.any():
            hard_neg_dist[invalid_mask] = d_pos[invalid_mask] + self.margin

        # 5. Triplet Margin Ranking Loss
        # Loss = max(0, d(a, p) - d(a, n_hard) + margin)
        loss = F.relu(d_pos - hard_neg_dist + self.margin).mean()

        return TriadMiningOutput(
            triad_loss=loss,
            mined_negative_indices=hard_neg_idx,
            hard_neg_distance=hard_neg_dist,
            pos_distance=d_pos,
        )
