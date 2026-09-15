#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — CELLULAR SHEAF DIFFUSION ENGINE (SHEAFDIFFUSIONSIGN)
================================================================================
Implements Cellular Sheaf Neural Diffusion & Sheaf Laplacian Cohomology (SheafLaplacian-SLT):
1. Cellular Sheaf F on Skeletal Graph G = (V, E) (60 Keypoints, 61 Edges):
     Node stalks F(v) in R^d, edge stalks F(e) in R^d (stalk_dim d = 4)
     Restriction maps F_{v |> e} in SO(d) (Orthogonal rotation matrices)
2. Block Discrete Sheaf Laplacian L_F in R^{60d x 60d}:
     Diagonal block:  L_F(u, u) = sum_{e ni u} F_{u |> e}^T F_{u |> e} = deg(u) * I_d
     Off-diagonal:    L_F(u, v) = - F_{u |> e}^T F_{v |> e}
     Guaranteed Positive Semi-Definite: <x, L_F x> = sum_e || F_{v |> e} x_v - F_{u |> e} x_u ||^2 >= 0
3. Sheaf Coboundary Operator & Cohomology Invariant:
     ||delta(x)||^2 directly measures morphological discrepancy across biological joints.
4. Neural Sheaf Continuous-Time Diffusion:
     X^{(k+1)} = X^{(k)} - dt * L_F X^{(k)} W_diff
     Mathematically prevents standard GNN over-smoothing via stalk rotations!
5. Feature Projection & Canonical Fusion:
     H_sheaf = H + LayerNorm( Linear( [X_diff, delta_energy, Tr(L_F)] ) )
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CellularSheafOutput(NamedTuple):
    sheaf_features: torch.Tensor        # [B, T, d_model] Projected sheaf diffusion representations
    diffused_stalks: torch.Tensor       # [B, T, 60, stalk_dim] Final diffused node stalk representations
    coboundary_energy: torch.Tensor     # [B, T] Global Dirichlet energy ||delta(x)||^2
    laplacian_trace: torch.Tensor       # [B] Trace of sheaf Laplacian Tr(L_F)
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + sheaf_features


class ASLCellularSheafEngine(nn.Module):
    """
    Cellular Sheaf Neural Diffusion & Discrete Sheaf Laplacian Cohomology Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        stalk_dim: int = 4,             # Dimension d of node/edge stalks
        dt: float = 0.2,                # Diffusion step size
        num_steps: int = 2,             # Discrete diffusion iterations
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.stalk_dim = stalk_dim
        self.dt = dt
        self.num_steps = num_steps

        # 1. Build Graph Edge Index
        # Torso & Face & Hands (61 undirected edges -> 61 bidirectional pairs)
        edges = []
        # Face contour (0..13)
        for i in range(13):
            edges.append((i, i + 1))
        edges.append((13, 0))

        # Torso
        torso_e = [(14, 15), (14, 16), (16, 18), (15, 17), (17, 39), (0, 14), (0, 15)]
        edges.extend(torso_e)

        # Left Hand (18 wrist to 5 fingers)
        for chain in [[18, 19, 20, 21, 22], [18, 23, 24, 25, 26], [18, 27, 28, 29, 30],
                      [18, 31, 32, 33, 34], [18, 35, 36, 37, 38]]:
            for i in range(len(chain) - 1):
                edges.append((chain[i], chain[i + 1]))

        # Right Hand (39 wrist to 5 fingers)
        for chain in [[39, 40, 41, 42, 43], [39, 44, 45, 46, 47], [39, 48, 49, 50, 51],
                      [39, 52, 53, 54, 55], [39, 56, 57, 58, 59]]:
            for i in range(len(chain) - 1):
                edges.append((chain[i], chain[i + 1]))

        self.edges = edges
        self.num_edges = len(edges)

        # 2. Stalk Mapping Network: Kinematics -> Initial Node Stalks x_0 in R^{60 x d}
        self.stalk_encoder = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, stalk_dim),
        )

        # 3. Restriction Map Predictor: predicts so(d) skew generators for each edge
        # Number of skew parameters in so(d): d*(d-1)/2 = 6 (for d=4)
        self.num_so_params = stalk_dim * (stalk_dim - 1) // 2
        self.restriction_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Linear(32, 2 * self.num_so_params), # for u->e and v->e
        )

        # 4. Output Projection Head
        # Input: 60 * stalk_dim + 1 (energy) + 1 (trace) -> d_model
        in_proj_dim = num_keypoints * stalk_dim + 2
        self.proj = nn.Sequential(
            nn.Linear(in_proj_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def params_to_so_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """
        Maps d*(d-1)/2 parameters to SO(d) via matrix exponential of skew-symmetric matrix.
        params: [B, T, num_edges, 6] -> R: [B, T, num_edges, 4, 4]
        """
        B, T, E, P = params.shape
        d = self.stalk_dim
        device = params.device

        # Form skew-symmetric matrix: A^T = -A
        A = torch.zeros(B, T, E, d, d, device=device, dtype=params.dtype)
        idx = 0
        for i in range(d):
            for j in range(i + 1, d):
                val = params[..., idx]
                A[..., i, j] = val
                A[..., j, i] = -val
                idx += 1

        # Matrix exponential: R = matrix_exp(A) in SO(d)
        R = torch.matrix_exp(A)  # [B, T, E, d, d]
        return R

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> CellularSheafOutput:
        """
        Executes cellular sheaf neural diffusion and computes sheaf Laplacian cohomology invariants.
        """
        B, T, K, C = kinematics.shape
        d = self.stalk_dim
        device = kinematics.device

        # 1. Initial Node Stalk Representations x_0: [B, T, 60, d]
        x = self.stalk_encoder(kinematics)  # [B, T, 60, d]

        # 2. Predict Orthogonal Restriction Maps F_{u |> e}, F_{v |> e} in SO(d)
        # Extract endpoint kinematics for each edge
        u_idx = torch.tensor([e[0] for e in self.edges], device=device)
        v_idx = torch.tensor([e[1] for e in self.edges], device=device)

        k_u = kinematics[:, :, u_idx, :]  # [B, T, num_edges, C]
        k_v = kinematics[:, :, v_idx, :]  # [B, T, num_edges, C]
        edge_kin = torch.cat([k_u, k_v], dim=-1)  # [B, T, num_edges, 2C]

        so_params = self.restriction_mlp(edge_kin)  # [B, T, num_edges, 2*num_so_params]
        params_u = so_params[..., :self.num_so_params]
        params_v = so_params[..., self.num_so_params:]

        F_u = self.params_to_so_matrix(params_u)  # [B, T, num_edges, d, d]
        F_v = self.params_to_so_matrix(params_v)  # [B, T, num_edges, d, d]

        # 3. Discrete Sheaf Diffusion Step: x_{k+1} = x_k - dt * L_F x_k
        # For each edge e = (u, v):
        # (delta x)_e = F_v x_v - F_u x_u in R^d
        # (L_F x)_u += F_u^T (delta x)_e = F_u^T (F_u x_u - F_v x_v) = x_u - F_u^T F_v x_v
        # (L_F x)_v -= F_v^T (delta x)_e = x_v - F_v^T F_u x_u
        for _ in range(self.num_steps):
            x_u = x[:, :, u_idx, :].unsqueeze(-1)  # [B, T, E, d, 1]
            x_v = x[:, :, v_idx, :].unsqueeze(-1)  # [B, T, E, d, 1]

            # F_u @ x_u, F_v @ x_v
            Fx_u = torch.matmul(F_u, x_u)  # [B, T, E, d, 1]
            Fx_v = torch.matmul(F_v, x_v)  # [B, T, E, d, 1]

            # Coboundary: delta_x = Fx_v - Fx_u
            delta_x = Fx_v - Fx_u          # [B, T, E, d, 1]

            # Pull back to nodes:
            # grad_u = - F_u^T @ delta_x
            # grad_v = + F_v^T @ delta_x
            flux_u = -torch.matmul(F_u.transpose(-1, -2), delta_x).squeeze(-1)  # [B, T, E, d]
            flux_v = torch.matmul(F_v.transpose(-1, -2), delta_x).squeeze(-1)   # [B, T, E, d]

            Lx = torch.zeros_like(x)  # [B, T, 60, d]
            Lx.index_add_(2, u_idx, -flux_u)
            Lx.index_add_(2, v_idx, -flux_v)

            # Update: x = x - dt * Lx
            x = x - self.dt * Lx

        # 4. Compute Global Dirichlet Energy ||delta(x)||^2
        x_u_final = x[:, :, u_idx, :].unsqueeze(-1)
        x_v_final = x[:, :, v_idx, :].unsqueeze(-1)
        delta_final = torch.matmul(F_v, x_v_final) - torch.matmul(F_u, x_u_final)
        dirichlet_energy = (delta_final ** 2).sum(dim=(-1, -2, -3))  # [B, T]

        # Sheaf Laplacian Trace = sum_v deg(v) * d = 2 * num_edges * d
        tr_LF = torch.full((B,), 2.0 * self.num_edges * self.stalk_dim, device=device)

        # 5. Feature Projection
        x_flat = x.reshape(B, T, K * d)  # [B, T, 60*d]
        tr_exp = tr_LF.view(B, 1, 1).expand(B, T, 1)
        dir_exp = dirichlet_energy.unsqueeze(-1)  # [B, T, 1]

        f_all = torch.cat([x_flat, dir_exp, tr_exp], dim=-1)  # [B, T, 60*d + 2]
        h_sheaf = self.proj(f_all)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + h_sheaf

        return CellularSheafOutput(
            sheaf_features=h_sheaf,
            diffused_stalks=x,
            coboundary_energy=dirichlet_energy,
            laplacian_trace=tr_LF,
            augmented_features=augmented,
        )
