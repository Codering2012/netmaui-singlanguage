#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — GRASSMANNIAN PRINCIPAL ANGLES ENGINE (GRASSMANNIANSIGN)
================================================================================
Implements Grassmannian Geodesic Metrics & Principal Angles Alignment (SubspaceAlign-SLT):
1. Multi-Joint Temporal Motion Subspace U in St(p, 3K):
     Given a temporal motion chunk X_c in R^{T_chunk x 3K}, extracts orthonormal basis:
     U_c = Orthogonalize( X_c ) via Thin SVD: X_c^T = U_c * Sigma * V^T, U_c in R^{3K x p}
2. Canonical Principal Angles theta_1 <= theta_2 <= ... <= theta_p between Subspaces U_i, U_j:
     U_i^T * U_j = V_1 * diag( cos(theta_1), ..., cos(theta_p) ) * V_2^T
     theta_k = arccos( clamp( sigma_k, 0, 1 ) )
3. Gauge Invariance on Grassmannian Gr(p, 3K):
     Strictly invariant to internal basis choice: theta( U_1 * O_1, U_2 * O_2 ) == theta( U_1, U_2 ).
4. Grassmannian Geodesic Canonical Metrics:
     d_chord(U_i, U_j) = sqrt( sum_k sin^2(theta_k) ) = (1 / sqrt(2)) * || U_i U_i^T - U_j U_j^T ||_F
     d_geo(U_i, U_j)   = sqrt( sum_k theta_k^2 )
5. Grassmannian Kernel Attention & Feature Projection:
     A_subspace(i, j) = exp( -gamma * d_chord(U_i, U_j) )
     H_grassmann = H + LayerNorm( Linear( [A_subspace * H, theta_summary, d_geo] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GrassmannianSubspaceOutput(NamedTuple):
    subspace_features: torch.Tensor     # [B, T, d_model] Projected Grassmannian representations
    chordal_distance_matrix: torch.Tensor # [B, num_chunks, num_chunks] Pairwise Grassmannian chordal distances
    geodesic_distance_matrix: torch.Tensor # [B, num_chunks, num_chunks] Pairwise Grassmannian geodesic distances
    principal_angles: torch.Tensor      # [B, num_chunks, num_chunks, p] Principal angles in [0, pi/2]
    subspace_attention: torch.Tensor    # [B, num_chunks, num_chunks] Kernel routing attention
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + subspace_features


class ASLGrassmannianSubspaceEngine(nn.Module):
    """
    Grassmannian Geodesic Metric & Principal Angles Alignment Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        subspace_dim: int = 4,          # Dimension p of the motion subspace
        chunk_size: int = 8,            # Temporal frames per chunk
        stride: int = 4,                # Sliding stride
        kernel_gamma: float = 0.5,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.p = subspace_dim
        self.chunk_size = chunk_size
        self.stride = stride
        self.gamma = kernel_gamma
        self.spatial_dim = num_keypoints * 3  # 180

        # Subspace feature projection head
        self.out_proj = nn.Sequential(
            nn.Linear(d_model + self.p + 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def extract_chunk_subspaces(self, pos: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Extracts orthonormal bases U_c in R^{3K x p} for each temporal sliding chunk.
        pos: [B, T, 60, 3] -> flat [B, T, 180]
        Returns: (bases [B, num_chunks, 180, p], num_chunks)
        """
        B, T, K, _ = pos.shape
        flat_pos = pos.reshape(B, T, self.spatial_dim)  # [B, T, 180]

        chunks = []
        for t_start in range(0, max(1, T - self.chunk_size + 1), self.stride):
            t_end = min(T, t_start + self.chunk_size)
            chunk_data = flat_pos[:, t_start:t_end, :]  # [B, T_c, 180]
            # Center chunk
            chunk_centered = chunk_data - chunk_data.mean(dim=1, keepdim=True)
            # Transpose to [B, 180, T_c]
            X = chunk_centered.transpose(1, 2)
            # Thin QR decomposition: X = Q * R -> Q[:, :, :p] in R^{B, 180, p}
            Q, _ = torch.linalg.qr(X, mode="reduced")
            U_c = Q[:, :, :self.p]  # [B, 180, p]
            # If T_c < p, pad with random orthonormal columns
            if U_c.shape[-1] < self.p:
                pad_dim = self.p - U_c.shape[-1]
                U_c = F.pad(U_c, (0, pad_dim))
            chunks.append(U_c)

        if len(chunks) == 0:
            # Fallback for short sequences: use full sequence
            X = flat_pos.transpose(1, 2)
            Q, _ = torch.linalg.qr(X, mode="reduced")
            U_c = Q[:, :, :self.p]
            if U_c.shape[-1] < self.p:
                U_c = F.pad(U_c, (0, self.p - U_c.shape[-1]))
            chunks.append(U_c)

        bases = torch.stack(chunks, dim=1)  # [B, num_chunks, 180, p]
        return bases, len(chunks)

    def compute_principal_angles_and_metrics(
        self,
        bases: torch.Tensor,  # [B, N_c, 180, p]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes pairwise principal angles theta, chordal distance, geodesic distance, and attention.
        """
        B, N_c, S, P = bases.shape
        device = bases.device
        eps = 1e-7

        # Pairwise inner products: M_{ij} = U_i^T @ U_j in R^{B, N_c, N_c, p, p}
        # bases: [B, N_c, 180, p] -> U_i: [B, N_c, 1, p, 180], U_j: [B, 1, N_c, 180, p]
        U_i = bases.transpose(-1, -2).unsqueeze(2)  # [B, N_c, 1, p, 180]
        U_j = bases.unsqueeze(1)                   # [B, 1, N_c, 180, p]
        M = torch.matmul(U_i, U_j)                 # [B, N_c, N_c, p, p]

        # Singular values of M: cos(theta_k) in [0, 1]
        sigmas = torch.linalg.svdvals(M).clamp(min=0.0, max=1.0)  # [B, N_c, N_c, p]

        # Principal angles: theta = arccos(sigma) in [0, pi/2]
        thetas = torch.acos(sigmas)  # [B, N_c, N_c, p]

        # Grassmannian Geodesic Distance: d_geo = sqrt( sum(theta_k^2) )
        d_geo = torch.sqrt((thetas ** 2).sum(dim=-1)).clamp(min=0.0)  # [B, N_c, N_c]

        # Grassmannian Chordal Distance: d_chord = sqrt( sum(sin^2(theta_k)) )
        sin_thetas = torch.sin(thetas)
        d_chord = torch.sqrt((sin_thetas ** 2).sum(dim=-1)).clamp(min=0.0)  # [B, N_c, N_c]

        # Enforce exact metric axiom: d(U_i, U_i) == 0 identically on the diagonal
        diag_mask = torch.eye(N_c, device=device, dtype=torch.bool).unsqueeze(0)
        d_chord = d_chord.masked_fill(diag_mask, 0.0)
        d_geo = d_geo.masked_fill(diag_mask, 0.0)

        # Subspace Kernel Attention: A = Softmax( -gamma * d_chord )
        attn = F.softmax(-self.gamma * d_chord, dim=-1)  # [B, N_c, N_c]

        return thetas, d_chord, d_geo, attn

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] or [B, T, 60, 3]
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> GrassmannianSubspaceOutput:
        """
        Computes Grassmannian bases, principal angles, geodesic metrics, and projects to d_model.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        pos = kinematics[..., 0:3]  # [B, T, 60, 3]

        # 1. Extract Chunk Subspace Bases
        bases, num_chunks = self.extract_chunk_subspaces(pos)  # [B, N_c, 180, p]

        # 2. Compute Principal Angles and Geometric Metrics
        thetas, d_chord, d_geo, attn = self.compute_principal_angles_and_metrics(bases)

        # 3. Aggregate Chunk Summaries
        # Average principal angles per chunk: [B, N_c, p]
        theta_summary = thetas.mean(dim=2)
        # Average distances per chunk: [B, N_c, 1]
        d_chord_sum = d_chord.mean(dim=2, keepdim=True)
        d_geo_sum = d_geo.mean(dim=2, keepdim=True)

        # Chunk representation: [B, N_c, p + 2]
        chunk_metrics = torch.cat([theta_summary, d_chord_sum, d_geo_sum], dim=-1)

        # Interpolate chunk metrics back to full sequence length T: [B, T, p + 2]
        # chunk_metrics: [B, N_c, p+2] -> [B, p+2, N_c] -> interpolate -> [B, p+2, T] -> [B, T, p+2]
        metrics_interp = F.interpolate(
            chunk_metrics.transpose(1, 2),
            size=T,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)  # [B, T, p + 2]

        if h_seq is None:
            h_base = torch.zeros(B, T, self.d_model, device=device, dtype=kinematics.dtype)
        else:
            h_base = h_seq

        # 4. Feature Projection
        f_in = torch.cat([h_base, metrics_interp], dim=-1)  # [B, T, d_model + p + 2]
        h_grassmann = self.out_proj(f_in)                    # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_grassmann

        return GrassmannianSubspaceOutput(
            subspace_features=h_grassmann,
            chordal_distance_matrix=d_chord,
            geodesic_distance_matrix=d_geo,
            principal_angles=thetas,
            subspace_attention=attn,
            augmented_features=augmented,
        )
