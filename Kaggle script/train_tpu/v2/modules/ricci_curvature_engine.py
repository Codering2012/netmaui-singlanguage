#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — OLLIVIER-RICCI GRAPH CURVATURE ENGINE (RICCICURVATURESIGN)
================================================================================
Implements Discrete Ricci Curvature Flow & Over-Squashing Mitigation (Ricci-SLT):
1. Anatomical Skeleton Graph Topology (60 Keypoints):
     Face (0..13), Torso (14..17), Left Hand (18..38), Right Hand (39..59)
2. Forman-Ricci Discrete Edge Curvature:
     Ric_F(u, v) = 4 - deg(u) - deg(v) + 3 * #Triangles(u, v) / deg_max
     Negative Curvature (Ric < 0) => Bottleneck / tree-like node (Over-squashing danger)
     Positive Curvature (Ric > 0) => Highly connected community (Fast mixing)
3. Dynamic Curvature-Guided Bandwidth Modulation:
     A_tilde_{u, v} = A_{u, v} * exp( -beta * Ric_F(u, v) )
     Mathematically widens bandwidth at bottleneck nodes (wrists: 18, 39) by exp(beta * |Ric|).
4. Curvature-Aware Graph Message Passing:
     Z = Softmax( A_tilde ) @ X @ W_msg
5. Curvature Invariant Fusion & Feature Projection:
     H_ricci = H + LayerNorm( Linear( [Z, Ric_node, Ric_edge_summary] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RicciCurvatureOutput(NamedTuple):
    ricci_features: torch.Tensor        # [B, T, d_model] Projected Ricci curvature representations
    modulated_adjacency: torch.Tensor   # [60, 60] Curvature-reweighted normalized adjacency matrix
    edge_curvature: torch.Tensor        # [num_edges] Forman-Ricci discrete curvature per edge
    node_curvature: torch.Tensor        # [60] Accumulated scalar Ricci curvature per node
    bottleneck_mask: torch.Tensor       # [60] Boolean indicator of negative-curvature bottleneck nodes
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + ricci_features


class ASLRicciCurvatureEngine(nn.Module):
    """
    Discrete Forman-Ricci Curvature & Over-Squashing Mitigation Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        beta_boost: float = 0.15,       # Curvature bandwidth boost strength
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.beta = beta_boost

        # 1. Construct Anatomical Skeleton Graph Adjacency
        A = torch.zeros(num_keypoints, num_keypoints, dtype=torch.float32)

        # Connect Face Contour (0..13)
        for i in range(13):
            A[i, i + 1] = A[i + 1, i] = 1.0
        A[13, 0] = A[0, 13] = 1.0

        # Connect Torso (14, 15: Shoulders, 16, 17: Elbows, 18, 39: Wrists)
        torso_edges = [
            (14, 15), (14, 16), (16, 18), (15, 17), (17, 39),
            (0, 14), (0, 15) # Face to shoulders
        ]
        for u, v in torso_edges:
            A[u, v] = A[v, u] = 1.0

        # Connect Left Hand (18 wrist to 5 finger chains)
        l_chains = [
            [18, 19, 20, 21, 22], [18, 23, 24, 25, 26], [18, 27, 28, 29, 30],
            [18, 31, 32, 33, 34], [18, 35, 36, 37, 38]
        ]
        for chain in l_chains:
            for i in range(len(chain) - 1):
                u, v = chain[i], chain[i + 1]
                A[u, v] = A[v, u] = 1.0

        # Connect Right Hand (39 wrist to 5 finger chains)
        r_chains = [
            [39, 40, 41, 42, 43], [39, 44, 45, 46, 47], [39, 48, 49, 50, 51],
            [39, 52, 53, 54, 55], [39, 56, 57, 58, 59]
        ]
        for chain in r_chains:
            for i in range(len(chain) - 1):
                u, v = chain[i], chain[i + 1]
                A[u, v] = A[v, u] = 1.0

        self.register_buffer("adj_orig", A)

        # 2. Compute Forman-Ricci Discrete Curvature across all Edges
        # Degrees
        deg = A.sum(dim=1) # [60]
        deg_max = deg.max().item()

        # Triangles: (A^3)_{u, v} / 2 for edge (u, v)
        A3 = torch.matmul(torch.matmul(A, A), A)

        # Edge list
        edges = []
        ric_edge_list = []
        ric_node = torch.zeros(num_keypoints, dtype=torch.float32)

        for u in range(num_keypoints):
            for v in range(u + 1, num_keypoints):
                if A[u, v] > 0.5:
                    edges.append((u, v))
                    # Number of common triangles = (A^2)_{u, v}
                    n_tri = (A[u, :] * A[v, :]).sum().item()
                    # Forman-Ricci formula: 4 - deg(u) - deg(v) + 3 * n_tri / deg_max
                    ric = 4.0 - deg[u].item() - deg[v].item() + 3.0 * (n_tri / max(deg_max, 1.0))
                    ric_edge_list.append(ric)
                    ric_node[u] += ric
                    ric_node[v] += ric

        self.edges = edges
        self.num_edges = len(edges)
        ric_edge_t = torch.tensor(ric_edge_list, dtype=torch.float32)
        self.register_buffer("edge_ricci", ric_edge_t)
        self.register_buffer("node_ricci", ric_node)

        # Identify bottleneck nodes (negative curvature: ric_node < 0)
        bottleneck_mask = ric_node < 0.0
        self.register_buffer("bottleneck_mask", bottleneck_mask)

        # 3. Compute Modulated Adjacency Matrix A_tilde = A * exp( -beta * Ric )
        A_tilde = A.clone()
        for idx, (u, v) in enumerate(edges):
            boost = math.exp(-self.beta * ric_edge_list[idx])
            A_tilde[u, v] = A[u, v] * boost
            A_tilde[v, u] = A[v, u] * boost

        # Normalize A_tilde: D^{-1/2} A_tilde D^{-1/2}
        d_tilde = A_tilde.sum(dim=1).clamp(min=1e-5)
        d_inv_sqrt = torch.diag(torch.rsqrt(d_tilde))
        A_norm = torch.matmul(torch.matmul(d_inv_sqrt, A_tilde), d_inv_sqrt)
        self.register_buffer("A_norm", A_norm)

        # 4. Message Passing Weights
        self.msg_fc = nn.Linear(in_channels, d_model)

        # 5. Output Projection
        # Input: d_model + 1 (node_ricci) + 1 (bottleneck) -> d_model
        self.out_proj = nn.Sequential(
            nn.Linear(num_keypoints * (d_model + 2), d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> RicciCurvatureOutput:
        """
        Executes Ricci curvature-modulated graph message passing and feature projection.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # 1. Project input node kinematics: [B, T, 60, d_model]
        X = self.msg_fc(kinematics)  # [B, T, 60, d_model]

        # 2. Curvature-Weighted Graph Message Passing: [B, T, 60, d_model]
        # A_norm is [60, 60]
        # Z = A_norm @ X -> [B, T, 60, d_model]
        A_mat = self.A_norm.unsqueeze(0).unsqueeze(0)  # [1, 1, 60, 60]
        Z = torch.matmul(A_mat, X)                     # [B, T, 60, d_model]

        # 3. Concatenate Node Ricci Invariants: [B, T, 60, d_model + 2]
        node_ric_exp = self.node_ricci.view(1, 1, K, 1).expand(B, T, K, 1)
        btn_mask_exp = self.bottleneck_mask.float().view(1, 1, K, 1).expand(B, T, K, 1)

        f_node = torch.cat([Z, node_ric_exp, btn_mask_exp], dim=-1).reshape(B, T, K * (self.d_model + 2))

        # 4. Output Projection
        h_ricci = self.out_proj(f_node)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_ricci

        return RicciCurvatureOutput(
            ricci_features=h_ricci,
            modulated_adjacency=self.A_norm,
            edge_curvature=self.edge_ricci,
            node_curvature=self.node_ricci,
            bottleneck_mask=self.bottleneck_mask,
            augmented_features=augmented,
        )
