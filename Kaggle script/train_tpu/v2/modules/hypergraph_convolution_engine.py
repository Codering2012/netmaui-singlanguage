#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — SPATIO-TEMPORAL HYPERGRAPH CONVOLUTION (HYPERSIGN)
================================================================================
Implements Spatio-Temporal Hypergraph Neural Networks (ST-HGNN / HyperSign):
1. High-Order Multi-Joint Hyperedges (connecting |e| >= 3 joints simultaneously):
     - e0: Facial Lip/Expression Contour (0..13)
     - e1: Upper Torso & Shoulders (14..17, 0)
     - e2: Left Fingertips Aperture (19, 23, 27, 31, 35)
     - e3: Left Palm & Thumb Web (18..22)
     - e4: Right Fingertips Aperture (40, 44, 48, 52, 56)
     - e5: Right Palm & Thumb Web (39..43)
     - e6: Bi-Manual Bilateral Coordination (18, 39, 14, 15)
     - e7: Face-Hand Signing Space Proximity (0, 18, 39)
2. Dynamic Learned Soft Hyperedge Incidence Matrix H_dyn in R^{K x E_dyn}:
     H_dyn(v, e) = Softmax( (q_v^T k_e) / sqrt(d) )
3. Hypergraph Laplacian Message Passing:
     X^{(l+1)} = sigma( D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2} X^{(l)} Theta ) + X^{(l)}
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HyperGraphOutput(NamedTuple):
    hypergraph_features: torch.Tensor    # [B, T, K, d_model] High-order hypergraph representations
    static_incidence_matrix: torch.Tensor# [K, num_static_edges] Binary incidence matrix H_static
    dynamic_incidence_matrix: torch.Tensor# [B, T, K, num_dyn_edges] Learned soft incidence matrix
    augmented_features: Optional[torch.Tensor] # [B, T, K, d_model] h_joints + hg_features


class ASLHypergraphConvolutionEngine(nn.Module):
    """
    Spatio-Temporal Hypergraph Convolutional Neural Network Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        num_dynamic_edges: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.num_dyn = num_dynamic_edges

        # 1. Define 8 Static Canonical Anatomical Hyperedges
        self.static_hyperedges = [
            list(range(0, 14)),                         # e0: Face (0..13)
            [14, 15, 16, 17, 0],                        # e1: Upper Torso & Head
            [19, 23, 27, 31, 35],                       # e2: Left Fingertips
            [18, 19, 20, 21, 22],                       # e3: Left Palm/Thumb
            [40, 44, 48, 52, 56],                       # e4: Right Fingertips
            [39, 40, 41, 42, 43],                       # e5: Right Palm/Thumb
            [18, 39, 14, 15],                           # e6: Bi-Manual Coordination
            [0, 18, 39],                                # e7: Face-Hand Signing Space
        ]
        self.num_static = len(self.static_hyperedges)
        self.total_edges = self.num_static + self.num_dyn

        # Construct binary static incidence matrix H_static in R^{K x num_static}
        H_stat = torch.zeros(num_keypoints, self.num_static, dtype=torch.float32)
        for e_idx, members in enumerate(self.static_hyperedges):
            for v_idx in members:
                if v_idx < num_keypoints:
                    H_stat[v_idx, e_idx] = 1.0
        self.register_buffer("H_static", H_stat)

        # Input feature embedding
        self.in_proj = nn.Linear(in_channels, d_model)

        # Dynamic Hyperedge Generator (Queries for vertices, Keys for hyperedges)
        self.q_proj = nn.Linear(d_model, d_model)
        self.hyperedge_keys = nn.Parameter(torch.randn(self.num_dyn, d_model))
        nn.init.normal_(self.hyperedge_keys, std=0.02)

        # Hyperedge learnable weights W_e
        self.edge_weights = nn.Parameter(torch.ones(self.total_edges))

        # Hypergraph convolution transform Theta
        self.conv_theta = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_hypergraph_laplacian(
        self,
        H: torch.Tensor,  # [..., K, E] Combined incidence matrix
    ) -> torch.Tensor:
        """
        Computes normalized Hypergraph Laplacian: L_H = D_v^{-1/2} H W_e D_e^{-1} H^T D_v^{-1/2}.
        Returns: [..., K, K]
        """
        eps = 1e-6
        # Edge degree D_e = sum_v H_ve in [..., E]
        D_e = H.sum(dim=-2) + eps  # [..., E]
        D_e_inv = 1.0 / D_e        # [..., E]

        # Scaled edge incidence: H_weighted = H * (W_e * D_e_inv) in [..., K, E]
        W_e_scaled = (self.edge_weights * D_e_inv).unsqueeze(-2)  # [..., 1, E]
        H_weighted = H * W_e_scaled  # [..., K, E]

        # Vertex degree D_v = sum_e (H_ve * W_e) in [..., K]
        D_v = (H * self.edge_weights.unsqueeze(-2)).sum(dim=-1) + eps  # [..., K]
        D_v_inv_sqrt = torch.rsqrt(D_v).unsqueeze(-1)  # [..., K, 1]

        # Message passing operator: M = H_weighted * H^T in [..., K, K]
        M = torch.matmul(H_weighted, H.transpose(-1, -2))  # [..., K, K]

        # Symmetrically normalize: L_H = D_v^{-1/2} * M * D_v^{-1/2}
        L_H = D_v_inv_sqrt * M * D_v_inv_sqrt.transpose(-1, -2)  # [..., K, K]
        return L_H

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_joints: Optional[torch.Tensor] = None,     # [B, T, 60, d_model] optional representations
    ) -> HyperGraphOutput:
        """
        Executes dynamic hypergraph construction, Laplacian message passing, and feature projection.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # 1. Project inputs
        x = self.in_proj(kinematics)  # [B, T, K, d_model]
        if h_joints is not None:
            x = x + h_joints

        # 2. Dynamic Learned Soft Hyperedges: H_dyn in [B, T, K, num_dyn]
        q = self.q_proj(x)  # [B, T, K, d_model]
        # Similarity against hyperedge prototype keys: [B, T, K, num_dyn]
        scores = torch.matmul(q, self.hyperedge_keys.t()) / math.sqrt(self.d_model)
        H_dyn = F.softmax(scores, dim=-2)  # Soft membership over vertices

        # 3. Combine Static & Dynamic Incidence Matrices
        # H_static: [K, num_static] -> expand to [B, T, K, num_static]
        H_stat_exp = self.H_static.view(1, 1, K, self.num_static).expand(B, T, K, self.num_static)
        H_total = torch.cat([H_stat_exp, H_dyn], dim=-1)  # [B, T, K, total_edges]

        # 4. Compute Hypergraph Laplacian L_H in [B, T, K, K]
        L_H = self.compute_hypergraph_laplacian(H_total)  # [B, T, K, K]

        # 5. Hypergraph Message Passing Convolution
        # High-order propagation: X_agg = L_H * X in [B, T, K, d_model]
        x_agg = torch.matmul(L_H, x)  # [B, T, K, d_model]
        x_out = self.conv_theta(x_agg) + x  # Residual connection

        augmented = None
        if h_joints is not None:
            augmented = h_joints + x_out

        return HyperGraphOutput(
            hypergraph_features=x_out,
            static_incidence_matrix=self.H_static,
            dynamic_incidence_matrix=H_dyn,
            augmented_features=augmented,
        )
