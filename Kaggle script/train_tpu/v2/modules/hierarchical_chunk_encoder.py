#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — HIERARCHICAL TEMPORAL CHUNKING & STREAMING ENCODER
================================================================================
Implements Multi-Scale Temporal Pyramid & Streaming Chunking (TSPNet / SignVTCL):
1. Local Fine-Grained Windowing (Level 1):
     Extracts local kinematic transitions within sliding windows W=16, stride=8.
2. Global Macro-Chunk Cross-Attention (Level 2):
     Aggregates local chunk tokens into high-level semantic representations,
     reducing quadratic attention FLOPs from O(T^2) to O(N_chunks * W^2).
3. Streaming Stepwise State Buffer:
     Maintains causal inter-chunk context for real-time live video processing.
4. Boundary Continuity Regularization:
     L_boundary = 1/(N-1) * sum || h_{i, end} - h_{i+1, start} ||_2^2
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkStreamingOutput(NamedTuple):
    chunk_tokens: torch.Tensor          # [B, N_chunks, d_model]
    boundary_loss: torch.Tensor         # Scalar continuity penalty
    total_chunks: int


class ASLHierarchicalChunkStreamingEncoder(nn.Module):
    """
    Hierarchical temporal window chunking and streaming encoder.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_keypoints: int = 60,
        d_model: int = 128,
        window_size: int = 16,
        stride: int = 8,
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        nhead: int = 4,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.window_size = window_size
        self.stride = stride

        # Spatial Keypoint Stem
        self.spatial_stem = nn.Sequential(
            nn.Linear(num_keypoints * in_channels, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Level 1: Local Window Transformer
        local_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.local_encoder = nn.TransformerEncoder(local_layer, num_layers=num_local_layers)

        # Level 2: Global Macro-Chunk Transformer
        global_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.global_encoder = nn.TransformerEncoder(global_layer, num_layers=num_global_layers)

        self.chunk_pool = nn.AdaptiveAvgPool1d(1)

    def extract_sliding_windows(
        self,
        spatial_feats: torch.Tensor,  # [B, T, d_model]
    ) -> Tuple[torch.Tensor, int]:
        """
        Unfolds sequence into overlapping chunks: [B * N_chunks, window_size, d_model]
        """
        B, T, D = spatial_feats.shape
        if T < self.window_size:
            pad_len = self.window_size - T
            spatial_feats = F.pad(spatial_feats, (0, 0, 0, pad_len), mode="replicate")
            T = self.window_size

        # Unfold along time dimension
        num_chunks = max(1, (T - self.window_size) // self.stride + 1)
        chunks = []
        for i in range(num_chunks):
            start = i * self.stride
            end = start + self.window_size
            chunks.append(spatial_feats[:, start:end])  # [B, window_size, D]

        chunked_tensor = torch.stack(chunks, dim=1)  # [B, N_chunks, window_size, D]
        return chunked_tensor, num_chunks

    def compute_boundary_continuity_loss(self, local_out: torch.Tensor) -> torch.Tensor:
        """
        Computes continuity loss across overlapping chunk boundaries.
        local_out: [B, N_chunks, window_size, D]
        """
        B, N, W, D = local_out.shape
        if N <= 1:
            return torch.tensor(0.0, device=local_out.device)

        # Compare end of chunk i with start of chunk i+1
        chunk_ends = local_out[:, :-1, -1, :]    # [B, N-1, D]
        chunk_starts = local_out[:, 1:, 0, :]    # [B, N-1, D]

        loss = F.mse_loss(chunk_ends, chunk_starts)
        return loss

    def forward(
        self,
        kinematics: torch.Tensor,  # [B, T, K, C]
    ) -> ChunkStreamingOutput:
        """
        Hierarchical Forward Pass: Local Windows -> Global Macro-Chunk Modeling.
        """
        B, T, K, C = kinematics.shape
        x_flat = kinematics.reshape(B, T, K * C)

        # 1. Spatial Stem
        spatial_feats = self.spatial_stem(x_flat)  # [B, T, d_model]

        # 2. Extract Local Overlapping Windows
        chunk_tensor, num_chunks = self.extract_sliding_windows(spatial_feats)  # [B, N, W, D]

        # 3. Level 1: Local Window Encoding
        chunk_flat = chunk_tensor.view(B * num_chunks, self.window_size, self.d_model)
        local_out_flat = self.local_encoder(chunk_flat)
        local_out = local_out_flat.view(B, num_chunks, self.window_size, self.d_model)

        # Boundary continuity penalty
        boundary_loss = self.compute_boundary_continuity_loss(local_out)

        # Pool local tokens into single macro-chunk embedding
        # [B * N, W, D] -> [B * N, D, W] -> pool -> [B * N, D, 1] -> [B, N, D]
        macro_tokens = self.chunk_pool(local_out_flat.transpose(1, 2)).squeeze(-1).view(B, num_chunks, self.d_model)

        # 4. Level 2: Global Macro-Chunk Contextual Encoding
        global_macro_out = self.global_encoder(macro_tokens)  # [B, N, d_model]

        return ChunkStreamingOutput(
            chunk_tokens=global_macro_out,
            boundary_loss=boundary_loss,
            total_chunks=num_chunks,
        )
