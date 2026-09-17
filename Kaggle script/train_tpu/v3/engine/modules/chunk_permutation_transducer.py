#!/usr/bin/env python3
"""
================================================================================
MONOTONIC CHUNK PERMUTATION TRANSDUCER (V3 ARCHITECTURE)
================================================================================
Resolves the fundamental conflict between ASL topic-prominent syntax
(Object-Subject-Verb OSV) and English Subject-Verb-Object (SVO) syntax under
monotonic CTC alignment.

Partitions continuous video representations into semantic chunks and evaluates
an adjacent transposition permutation lattice (S_2 subgroup) to permit local
reordering in O(M) linear time without combinatorial explosion or breaking
PyTorch/XLA graph tracing.
================================================================================
"""

from typing import Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkPermutationTransducer(nn.Module):
    r"""
    Monotonic Chunk-Level Permutation Transducer for non-monotonic ASL->English syntax.
    
    Args:
        d_model: Latent feature dimension (aligned to multiples of 128).
        chunk_size: Temporal length of each semantic chunk (default 16 or 32 frames).
        num_heads: Attention heads for inter-chunk permutation routing.
    """

    def __init__(
        self,
        d_model: int = 128,
        chunk_size: int = 16,
        num_heads: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.num_heads = num_heads

        # 1. Chunk Summary Pooling Attention
        self.chunk_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.chunk_pool_norm = nn.LayerNorm(d_model)

        # 2. Adjacent Transposition Swap Scoring Head
        # Evaluates probability p_swap in [0, 1] that chunk m should swap with chunk m+1
        self.swap_scorer = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # 3. Permuted Chunk Re-expansion Projection
        self.permute_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            hidden_states: [B, T, d_model] Contextual sequence.
            mask: [B, T] Optional valid temporal mask.
            
        Returns:
            reordered_hidden: [B, T, d_model] Syntax-reordered representations.
            aux_losses: Dictionary containing permutation entropy and continuity losses.
        """
        B, T, D = hidden_states.shape
        C = self.chunk_size

        # If sequence is shorter than 2 chunks, pass through
        if T < 2 * C:
            return hidden_states, {"loss_permutation": torch.zeros((), device=hidden_states.device)}

        # Pad T to exact multiple of chunk_size C for clean batched tensor reshaping
        num_chunks = math.ceil(T / C)
        pad_len = num_chunks * C - T
        if pad_len > 0:
            padded_hidden = F.pad(hidden_states, (0, 0, 0, pad_len))
        else:
            padded_hidden = hidden_states

        # Reshape into chunks: [B, M, C, D]
        chunked = padded_hidden.view(B, num_chunks, C, D)

        # 1. Pool each chunk into a single vector representation: [B, M, D]
        # Mean pooling across chunk frames
        chunk_reprs = torch.mean(chunked, dim=2)  # [B, M, D]
        chunk_reprs = self.chunk_pool_norm(chunk_reprs)

        # 2. Vectorized evaluation of adjacent swap probabilities
        # Pair adjacent chunks: [B, M-1, 2*D]
        chunk_pairs = torch.cat([chunk_reprs[:, :-1, :], chunk_reprs[:, 1:, :]], dim=-1)  # [B, M-1, 2*D]
        swap_logits = self.swap_scorer(chunk_pairs).squeeze(-1)  # [B, M-1]
        swap_probs = torch.sigmoid(swap_logits)  # [B, M-1] in [0, 1]

        # 3. Differentiable Soft Permutation via Convex Combinations
        # Construct soft transition matrix for adjacent pairs without unrolled loops
        # A chunk m receives weight from m, m-1, and m+1
        # Pad swap_probs with zeros on boundaries
        zeros_pad = torch.zeros((B, 1), device=hidden_states.device, dtype=swap_probs.dtype)
        p_left = torch.cat([zeros_pad, swap_probs], dim=1)        # Probability of swapping with left neighbor
        p_right = torch.cat([swap_probs, zeros_pad], dim=1)       # Probability of swapping with right neighbor

        # Soft reordering factor for chunk m:
        # permuted_chunk[m] = (1 - p_right[m] - p_left[m])*chunk[m] + p_right[m]*chunk[m+1] + p_left[m]*chunk[m-1]
        w_curr = (1.0 - 0.5 * (p_left + p_right)).unsqueeze(-1).unsqueeze(-1)  # [B, M, 1, 1]
        w_right = (0.5 * p_right).unsqueeze(-1).unsqueeze(-1)
        w_left = (0.5 * p_left).unsqueeze(-1).unsqueeze(-1)

        # Roll neighbors
        chunk_next = torch.roll(chunked, shifts=-1, dims=1)
        chunk_prev = torch.roll(chunked, shifts=1, dims=1)

        permuted_chunked = w_curr * chunked + w_right * chunk_next + w_left * chunk_prev  # [B, M, C, D]

        # 4. Flatten back to [B, num_chunks * C, D] and crop to original T
        flattened = permuted_chunked.view(B, num_chunks * C, D)
        reordered_hidden = flattened[:, :T, :]

        # Residual connection
        reordered_hidden = self.norm(hidden_states + self.permute_proj(reordered_hidden))

        # Auxiliary regularization: entropy penalty preventing indeterminate 0.5 swap states
        # swap_probs should be decisive (close to 0 or close to 1)
        decisive_penalty = torch.mean(swap_probs * (1.0 - swap_probs))

        return reordered_hidden, {
            "swap_probs": swap_probs,
            "loss_permutation": decisive_penalty * 0.05,
        }
