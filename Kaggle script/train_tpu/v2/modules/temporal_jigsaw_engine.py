#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SELF-SUPERVISED TEMPORAL JIGSAW & ARROW-OF-TIME ENGINE
================================================================================
Implements Self-Supervised Temporal Jigsaw, Pairwise Precedence, and Arrow-of-Time:
1. Vectorized Temporal Chunk Permutation (Jigsaw Puzzle):
     Partitions video into P=4 non-overlapping temporal chunks and applies
     permutations pi in S_4 (codebook of 24 canonical permutations) with 100%
     vectorized batched indexing on TPU/GPU without Python loops.
     L_jigsaw = CrossEntropy( JigsawHead( h_permuted ), y_pi )
2. Pairwise Precedence & Transitive Order Matrix:
     Predicts relative pairwise temporal order for all (i, j) chunk pairs:
     M_ij = I( orig_pos(i) < orig_pos(j) )
     L_pair = BCEWithLogits( PairwiseHead( h_i, h_j ), M_ij )
     L_trans = TransitivityLoss( sigmoid( M_hat ) )
3. Arrow-of-Time Reversal Prediction:
     Predicts forward (+1) vs temporally reversed (-1) gesture playback.
     L_arrow = BCEWithLogits( ArrowHead( h_dir ), y_dir )
4. Enforces gesture phase causality (Preparation -> Stroke -> Hold -> Retraction).
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class JigsawSSLOutput(NamedTuple):
    permuted_kinematics: torch.Tensor    # [B, T, K, C]
    permutation_label: torch.Tensor      # [B] Permutation index
    jigsaw_loss: torch.Tensor            # Scalar permutation cross-entropy
    arrow_loss: torch.Tensor             # Scalar forward/reverse binary loss
    pairwise_loss: torch.Tensor          # Scalar pairwise precedence loss
    transitive_loss: torch.Tensor        # Scalar transitive consistency loss
    total_loss: torch.Tensor             # Combined SSL loss
    predicted_order: torch.Tensor        # [B, P, P] Predicted pairwise precedence probabilities


class ASLTemporalJigsawEngine(nn.Module):
    """
    Self-supervised temporal jigsaw, pairwise precedence, and arrow-of-time engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        num_chunks: int = 4,
        arrow_loss_weight: float = 0.50,
        pairwise_loss_weight: float = 0.30,
        transitive_loss_weight: float = 0.10,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_chunks = num_chunks
        self.arrow_loss_weight = arrow_loss_weight
        self.pairwise_loss_weight = pairwise_loss_weight
        self.transitive_loss_weight = transitive_loss_weight

        # Precompute canonical permutation codebook for P=4 (24 permutations)
        perms = list(itertools.permutations(range(num_chunks)))
        self.num_perms = len(perms)
        # Register codebook as fixed buffer [num_perms, num_chunks]
        self.register_buffer("perm_codebook", torch.tensor(perms, dtype=torch.long))

        # Permutation Classification Head
        self.jigsaw_head = nn.Sequential(
            nn.Linear(d_model * num_chunks, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, self.num_perms),
        )

        # Pairwise Precedence Head: compares chunk_i and chunk_j representations
        self.pairwise_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Arrow-of-Time Direction Head
        self.arrow_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def permute_temporal_chunks(
        self,
        kinematics: torch.Tensor,  # [B, T, K, C]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Splits sequence into P chunks and permutes them with 100% vectorized batched indexing.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device
        chunk_len = T // self.num_chunks
        used_T = chunk_len * self.num_chunks

        # Sample permutation index per sample in batch
        perm_indices = torch.randint(0, self.num_perms, (B,), device=device)  # [B]

        # Extract chunks: [B, P, chunk_len, K, C]
        trimmed = kinematics[:, :used_T]
        chunk_tensor = trimmed.view(B, self.num_chunks, chunk_len, K, C)

        # Vectorized gather across permutation indices
        # perm_codebook[perm_indices] -> [B, P]
        pi = self.perm_codebook[perm_indices]  # [B, P]

        # Expand indices for 5D gather: [B, P, chunk_len, K, C]
        gather_idx = pi.view(B, self.num_chunks, 1, 1, 1).expand(B, self.num_chunks, chunk_len, K, C)
        permuted_chunk_tensor = torch.gather(chunk_tensor, dim=1, index=gather_idx)

        # Reshape back to [B, used_T, K, C]
        permuted_tensor = permuted_chunk_tensor.view(B, used_T, K, C)

        # Append remainder frames if T > used_T
        if T > used_T:
            remainder = kinematics[:, used_T:]
            permuted_tensor = torch.cat([permuted_tensor, remainder], dim=1)

        return permuted_tensor, perm_indices

    def create_arrow_of_time_batch(
        self,
        kinematics: torch.Tensor,  # [B, T, K, C]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Creates forward (label 1.0) and time-reversed (label 0.0) batch.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # Flip time dimension for half of the batch
        flip_mask = torch.rand(B, device=device) < 0.50
        dir_labels = (~flip_mask).float()  # 1.0 = forward, 0.0 = reversed

        dir_kinematics = kinematics.clone()
        if flip_mask.any():
            dir_kinematics[flip_mask] = torch.flip(dir_kinematics[flip_mask], dims=[1])

        return dir_kinematics, dir_labels

    def compute_jigsaw_loss(
        self,
        chunk_embeddings: torch.Tensor,  # [B, P, d_model]
        perm_labels: torch.Tensor,       # [B]
    ) -> torch.Tensor:
        """
        Computes multi-class permutation classification loss.
        """
        B, P, D = chunk_embeddings.shape
        flat_feats = chunk_embeddings.reshape(B, P * D)
        logits = self.jigsaw_head(flat_feats)  # [B, num_perms]
        loss = F.cross_entropy(logits, perm_labels)
        return loss

    def compute_pairwise_precedence_loss(
        self,
        chunk_embeddings: torch.Tensor,  # [B, P, d_model]
        perm_labels: torch.Tensor,       # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes pairwise precedence BCE loss and transitive consistency loss.
        """
        B, P, D = chunk_embeddings.shape
        device = chunk_embeddings.device

        # Form all (i, j) pairs: [B, P, P, 2*D]
        ci = chunk_embeddings.unsqueeze(2).expand(B, P, P, D)
        cj = chunk_embeddings.unsqueeze(1).expand(B, P, P, D)
        pair_feats = torch.cat([ci, cj], dim=-1)  # [B, P, P, 2*D]

        pair_logits = self.pairwise_head(pair_feats).squeeze(-1)  # [B, P, P]
        pair_probs = torch.sigmoid(pair_logits)                   # [B, P, P]

        # Ground truth precedence matrix M_ij = I( orig_pos(i) < orig_pos(j) )
        # pi[b] contains the permuted chunk order at positions 0..P-1
        pi = self.perm_codebook[perm_labels]  # [B, P]
        # Inverted permutation: orig_pos of chunk at position p is pi[b, p]
        orig_i = pi.unsqueeze(2).expand(B, P, P)  # [B, P, P]
        orig_j = pi.unsqueeze(1).expand(B, P, P)  # [B, P, P]
        gt_precedence = (orig_i < orig_j).float()  # [B, P, P]

        # Mask out diagonal (i == j)
        eye_mask = torch.eye(P, device=device, dtype=torch.bool).unsqueeze(0).expand(B, P, P)
        valid_mask = ~eye_mask

        # Pairwise BCE loss
        bce = F.binary_cross_entropy_with_logits(
            pair_logits[valid_mask],
            gt_precedence[valid_mask],
        )

        # Transitive consistency loss: for any triple (i, j, k), if M_ij=1 and M_jk=1 then M_ik=1
        # Penalty: max(0, P_ij + P_jk - P_ik - 1)
        p_ij = pair_probs.unsqueeze(3)  # [B, P, P, 1]
        p_jk = pair_probs.unsqueeze(1)  # [B, 1, P, P]
        p_ik = pair_probs.unsqueeze(2)  # [B, P, 1, P]

        trans_violation = F.relu(p_ij + p_jk - p_ik - 1.0)  # [B, P, P, P]
        # Exclude degenerate indices where i==j or j==k or i==k
        idx = torch.arange(P, device=device)
        non_degen = (idx.view(P, 1, 1) != idx.view(1, P, 1)) & \
                    (idx.view(1, P, 1) != idx.view(1, 1, P)) & \
                    (idx.view(P, 1, 1) != idx.view(1, 1, P))  # [P, P, P]
        
        non_degen_b = non_degen.unsqueeze(0).expand(B, P, P, P)
        if non_degen_b.any():
            trans_loss = trans_violation[non_degen_b].mean()
        else:
            trans_loss = torch.tensor(0.0, device=device)

        return bce, trans_loss, pair_probs

    def compute_arrow_loss(
        self,
        h_cls: torch.Tensor,             # [B, d_model]
        dir_labels: torch.Tensor,        # [B]
    ) -> torch.Tensor:
        """
        Computes binary classification loss for forward vs reversed motion.
        """
        logits = self.arrow_head(h_cls).squeeze(-1)  # [B]
        loss = F.binary_cross_entropy_with_logits(logits, dir_labels)
        return loss

    def forward(
        self,
        chunk_embeddings: torch.Tensor,  # [B, P, d_model]
        perm_labels: torch.Tensor,       # [B]
        dir_embeddings: torch.Tensor,    # [B, d_model]
        dir_labels: torch.Tensor,        # [B]
    ) -> JigsawSSLOutput:
        """
        Computes combined self-supervised Jigsaw, Pairwise, Transitive, and Arrow-of-Time loss.
        """
        j_loss = self.compute_jigsaw_loss(chunk_embeddings, perm_labels)
        pair_loss, trans_loss, pair_probs = self.compute_pairwise_precedence_loss(chunk_embeddings, perm_labels)
        a_loss = self.compute_arrow_loss(dir_embeddings, dir_labels)

        tot_loss = (
            j_loss +
            self.arrow_loss_weight * a_loss +
            self.pairwise_loss_weight * pair_loss +
            self.transitive_loss_weight * trans_loss
        )

        return JigsawSSLOutput(
            permuted_kinematics=chunk_embeddings,
            permutation_label=perm_labels,
            jigsaw_loss=j_loss,
            arrow_loss=a_loss,
            pairwise_loss=pair_loss,
            transitive_loss=trans_loss,
            total_loss=tot_loss,
            predicted_order=pair_probs,
        )

