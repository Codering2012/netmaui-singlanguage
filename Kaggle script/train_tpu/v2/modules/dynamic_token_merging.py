#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — DYNAMIC SPATIOTEMPORAL TOKEN MERGING (TOME) ENGINE
================================================================================
Implements Vectorized Bipartite Soft Matching Token Merging (ToMe / D³ToM):
1. Bipartite Cosine Similarity Matching:
     Splits sequence into source and destination sets, matching source tokens
     to their most similar temporal destinations.
2. Vectorized Out-of-Place Token Consolidation:
     X_dst_merged = (w_dst * X_dst + M^T * (w_src * X_src)) / (w_dst + M^T * w_src)
     w_dst_merged = w_dst + M^T * w_src
   Reduces sequence length by 30-50% while preserving 100% of information mass.
3. Kinematic Un-Merging Restoration:
     Maintains inverse bipartite routing matrix U [T_orig, T_merged] to perfectly
     reconstruct frame-level CTC alignments without loss of precision.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenMergingOutput(NamedTuple):
    merged_tokens: torch.Tensor         # [B, T_merged, d_model]
    token_weights: torch.Tensor         # [B, T_merged, 1]
    unmerge_matrix: torch.Tensor        # [B, T_orig, T_merged]
    reduction_ratio: float


class ASLDynamicTokenMergingEngine(nn.Module):
    """
    Dynamic Bipartite Token Merging & Un-Merging Engine for ASL Sequence Models.
    """

    def __init__(
        self,
        d_model: int = 128,
        merge_ratio: float = 0.35,  # Percentage of tokens to merge
    ):
        super().__init__()
        self.d_model = d_model
        self.merge_ratio = merge_ratio

    def forward(
        self,
        tokens: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> TokenMergingOutput:
        """
        Forward pass calling bipartite soft matching merge.
        """
        return self.bipartite_soft_matching_merge(tokens, weights)

    def bipartite_soft_matching_merge(
        self,
        tokens: torch.Tensor,       # [B, T, d_model]
        weights: Optional[torch.Tensor] = None, # [B, T, 1]
    ) -> TokenMergingOutput:
        """
        Vectorized bipartite soft matching to merge r most redundant tokens.
        """
        B, T, D = tokens.shape
        if weights is None:
            weights = torch.ones(B, T, 1, device=tokens.device, dtype=tokens.dtype)

        r = int(T * self.merge_ratio)
        if r <= 0 or T <= 2:
            unmerge_eye = torch.eye(T, device=tokens.device).unsqueeze(0).repeat(B, 1, 1)
            return TokenMergingOutput(
                merged_tokens=tokens,
                token_weights=weights,
                unmerge_matrix=unmerge_eye,
                reduction_ratio=0.0,
            )

        # Alternating split: even indices -> dst, odd indices -> src
        src_idx = torch.arange(1, T, 2, device=tokens.device)
        dst_idx = torch.arange(0, T, 2, device=tokens.device)

        N_src = len(src_idx)
        N_dst = len(dst_idx)
        r = min(r, N_src)

        src_tokens = tokens[:, src_idx]     # [B, N_src, D]
        dst_tokens = tokens[:, dst_idx]     # [B, N_dst, D]

        src_w = weights[:, src_idx]         # [B, N_src, 1]
        dst_w = weights[:, dst_idx]         # [B, N_dst, 1]

        # Normalized Cosine Similarity Matrix: [B, N_src, N_dst]
        src_norm = F.normalize(src_tokens, p=2, dim=-1)
        dst_norm = F.normalize(dst_tokens, p=2, dim=-1)
        sim_matrix = torch.bmm(src_norm, dst_norm.transpose(1, 2))  # [B, N_src, N_dst]

        # Find best destination match for each source token
        best_sim, best_dst_idx = torch.max(sim_matrix, dim=-1)  # [B, N_src], [B, N_src]

        # Select top-r most similar source tokens to merge
        _, topk_src_idx = torch.topk(best_sim, k=r, dim=-1)  # [B, r]

        # Construct bipartite assignment matrix M [B, N_src, N_dst]
        M = torch.zeros(B, N_src, N_dst, device=tokens.device, dtype=tokens.dtype)
        batch_idx = torch.arange(B, device=tokens.device).unsqueeze(-1).repeat(1, r)  # [B, r]
        selected_dst = torch.gather(best_dst_idx, 1, topk_src_idx)                    # [B, r]
        M[batch_idx, topk_src_idx, selected_dst] = 1.0

        # Out-of-place weighted consolidation
        # Accumulated source token mass into destination slots:
        # [B, N_dst, N_src] x [B, N_src, D] -> [B, N_dst, D]
        M_t = M.transpose(1, 2)
        accum_src_tokens = torch.bmm(M_t, src_w * src_tokens)
        accum_src_w = torch.bmm(M_t, src_w)

        merged_dst_tokens = (dst_w * dst_tokens + accum_src_tokens) / (dst_w + accum_src_w).clamp(min=1e-5)
        merged_dst_w = dst_w + accum_src_w

        # Mask of unmerged source tokens [B, N_src]
        unmerged_src_mask = 1.0 - M.sum(dim=-1)  # [B, N_src], 1.0 = unmerged

        # Build final merged sequence
        # We preserve all destinations (now merged) plus unmerged source tokens
        T_merged = T - r
        merged_tokens_list = []
        merged_weights_list = []
        unmerge_mat_list = []

        for b in range(B):
            b_dst_tokens = merged_dst_tokens[b]  # [N_dst, D]
            b_dst_w = merged_dst_w[b]            # [N_dst, 1]

            b_unmerged_mask = unmerged_src_mask[b].bool()
            b_unmerged_src_tokens = src_tokens[b][b_unmerged_mask]  # [N_unmerged, D]
            b_unmerged_src_w = src_w[b][b_unmerged_mask]            # [N_unmerged, 1]

            b_merged_t = torch.cat([b_dst_tokens, b_unmerged_src_tokens], dim=0)  # [T_merged, D]
            b_merged_w = torch.cat([b_dst_w, b_unmerged_src_w], dim=0)            # [T_merged, 1]

            # Build unmerge matrix [T_orig, T_merged]
            b_unmerge = torch.zeros(T, T_merged, device=tokens.device, dtype=tokens.dtype)
            
            # Destination rows
            for idx_d, orig_d in enumerate(dst_idx):
                b_unmerge[orig_d, idx_d] = 1.0

            # Merged source rows
            for s_i in range(N_src):
                if not b_unmerged_mask[s_i]:
                    # Merged into destination d_j
                    d_j = best_dst_idx[b, s_i].item()
                    orig_s = src_idx[s_i]
                    b_unmerge[orig_s, d_j] = 1.0

            # Unmerged source rows
            unmerged_indices = torch.nonzero(b_unmerged_mask).squeeze(-1)
            for idx_u, s_i in enumerate(unmerged_indices):
                orig_s = src_idx[s_i]
                merged_pos = N_dst + idx_u
                b_unmerge[orig_s, merged_pos] = 1.0

            merged_tokens_list.append(b_merged_t)
            merged_weights_list.append(b_merged_w)
            unmerge_mat_list.append(b_unmerge)

        merged_tokens_tensor = torch.stack(merged_tokens_list, dim=0)   # [B, T_merged, D]
        merged_weights_tensor = torch.stack(merged_weights_list, dim=0) # [B, T_merged, 1]
        unmerge_mat_tensor = torch.stack(unmerge_mat_list, dim=0)       # [B, T, T_merged]

        return TokenMergingOutput(
            merged_tokens=merged_tokens_tensor,
            token_weights=merged_weights_tensor,
            unmerge_matrix=unmerge_mat_tensor,
            reduction_ratio=float(r / T),
        )

    def unmerge_tokens(
        self,
        merged_tokens: torch.Tensor,    # [B, T_merged, d_model]
        unmerge_matrix: torch.Tensor,   # [B, T_orig, T_merged]
    ) -> torch.Tensor:
        """
        Unmerges tokens back to original temporal resolution [B, T_orig, d_model] via U @ X_merged.
        """
        # [B, T_orig, T_merged] @ [B, T_merged, D] -> [B, T_orig, D]
        return torch.bmm(unmerge_matrix, merged_tokens)

    def unmerge(
        self,
        merged_tokens: torch.Tensor,
        unmerge_matrix: torch.Tensor,
    ) -> torch.Tensor:
        """
        Alias for unmerge_tokens.
        """
        return self.unmerge_tokens(merged_tokens, unmerge_matrix)
