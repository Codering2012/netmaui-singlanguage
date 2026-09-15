#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — THERMODYNAMIC ENTROPY PRODUCTION ENGINE (THERMODYNAMICSIGN)
================================================================================
Implements Non-Equilibrium Thermodynamic Entropy Production & Dissipation (TSSM):
1. Instantaneous Entropy Production Rate (EPR) sigma_dot(t):
     sigma_dot(t) = (gamma / T_temp) * ||v(t)||^2 + (m / T_temp) * (a(t) . v(t))
     Measures time-reversal symmetry breaking and kinetic energy dissipation.
2. Heat Conduction & Spatial Thermal Diffusion on Skeletal Graph:
     u_diff = (I - alpha * L_graph) * v_energy
     Propagates kinetic energy waves across the articulated skeletal graph.
3. Second Law of Thermodynamics Regularization:
     L_second_law = ReLU(-sigma_dot(t)).mean()  (enforces sigma_dot >= 0)
4. Thermodynamic Feature Projection:
     H_thermo = H + LayerNorm(Linear([sigma_dot, u_diff, kinetic_power]))
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ThermodynamicOutput(NamedTuple):
    thermodynamic_features: torch.Tensor # [B, T, d_model] Projected thermodynamic representations
    entropy_production_rate: torch.Tensor# [B, T, K] Joint-wise EPR sigma_dot(t)
    mean_sequence_entropy: torch.Tensor  # [B, T] Global temporal entropy dissipation
    diffused_thermal_energy: torch.Tensor# [B, T, K] Heat conduction diffused energy
    second_law_loss: torch.Tensor        # [1] Regularization penalty for negative entropy
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] h_seq + thermodynamic_features


class ASLThermodynamicEntropyEngine(nn.Module):
    """
    Non-Equilibrium Thermodynamic Entropy Production & Kinetic Dissipation Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 9,
        num_keypoints: int = 60,
        gamma_friction: float = 0.5,     # Damping friction coefficient gamma
        temperature: float = 1.0,        # Thermodynamic bath temperature T_temp
        alpha_diffusion: float = 0.1,    # Thermal graph diffusivity alpha
    ):
        super().__init__()
        self.d_model = d_model
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.gamma = gamma_friction
        self.temp = temperature
        self.alpha = alpha_diffusion

        # Construct normalized adjacency / Laplacian matrix for 60 keypoints
        # Connect consecutive joints within chains (face 0..13, body 14..17, left hand 18..38, right hand 39..59)
        A = torch.zeros(num_keypoints, num_keypoints, dtype=torch.float32)
        chains = [list(range(0, 14)), list(range(14, 18)), list(range(18, 39)), list(range(39, 60))]
        for chain in chains:
            for i in range(len(chain) - 1):
                u, v = chain[i], chain[i + 1]
                A[u, v] = 1.0
                A[v, u] = 1.0
        # Add bilateral wrist-shoulder connections
        A[14, 18] = 1.0; A[18, 14] = 1.0
        A[15, 39] = 1.0; A[39, 15] = 1.0

        # Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
        deg = A.sum(dim=-1).clamp(min=1.0)
        deg_inv_sqrt = torch.diag(torch.rsqrt(deg))
        L = torch.eye(num_keypoints) - torch.matmul(torch.matmul(deg_inv_sqrt, A), deg_inv_sqrt)
        self.register_buffer("L_graph", L)

        # Projection head: 3 features per joint (EPR, diffused_energy, kinetic_power) = 3 * K
        in_dim = num_keypoints * 3
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def compute_thermodynamic_metrics(
        self,
        kinematics: torch.Tensor,  # [B, T, 60, 9] (coords 0:3, vel 3:6, acc 6:9)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes joint-wise EPR sigma_dot, diffused thermal energy, and kinetic power.
        Returns: (epr [B, T, K], diffused [B, T, K], power [B, T, K])
        """
        vel = kinematics[..., 3:6]  # [B, T, K, 3]
        acc = kinematics[..., 6:9]  # [B, T, K, 3]

        # 1. Kinetic Power: P = m * (a . v)  (assuming unit mass m = 1.0)
        kinetic_power = (acc * vel).sum(dim=-1)  # [B, T, K]

        # 2. Velocity squared magnitude: ||v||^2
        vel_sq = vel.pow(2).sum(dim=-1)  # [B, T, K]

        # 3. Instantaneous Entropy Production Rate: sigma_dot = (gamma * ||v||^2 + a . v) / T_temp
        epr = (self.gamma * vel_sq + kinetic_power) / self.temp  # [B, T, K]

        # 4. Spatial Heat Conduction Graph Diffusion: u_diff = (I - alpha * L_graph) * vel_sq
        # L_graph: [K, K], vel_sq: [B, T, K]
        # (B*T, K) x (K, K) -> (B*T, K)
        B, T, K = vel_sq.shape
        vel_sq_flat = vel_sq.view(B * T, K)
        # Heat conduction step: u = vel_sq_flat - alpha * (vel_sq_flat * L^T)
        heat_diff = vel_sq_flat - self.alpha * torch.matmul(vel_sq_flat, self.L_graph.t())
        diffused_energy = heat_diff.view(B, T, K)

        return epr, diffused_energy, kinetic_power

    def forward(
        self,
        kinematics: torch.Tensor,                    # [B, T, 60, 9] (coords + vel + acc)
        h_seq: Optional[torch.Tensor] = None,        # [B, T, d_model] optional encoder features
    ) -> ThermodynamicOutput:
        """
        Computes non-equilibrium entropy production rate, heat diffusion, and feature projection.
        """
        B, T, K, C = kinematics.shape
        device = kinematics.device

        # 1. Compute Thermodynamic Metrics
        epr, diffused_energy, kinetic_power = self.compute_thermodynamic_metrics(kinematics)

        # 2. Global Mean Sequence Entropy
        mean_entropy = epr.mean(dim=-1)  # [B, T]

        # 3. Second Law Regularization Penalty: ReLU(-epr)
        second_law_loss = F.relu(-epr).mean()

        # 4. Feature Projection
        # Stack [epr, diffused_energy, kinetic_power] -> [B, T, K * 3]
        flat_thermo = torch.cat([epr, diffused_energy, kinetic_power], dim=-1)  # [B, T, K * 3]
        thermo_emb = self.proj(flat_thermo)  # [B, T, d_model]

        augmented = None
        if h_seq is not None:
            augmented = h_seq + thermo_emb

        return ThermodynamicOutput(
            thermodynamic_features=thermo_emb,
            entropy_production_rate=epr,
            mean_sequence_entropy=mean_entropy,
            diffused_thermal_energy=diffused_energy,
            second_law_loss=second_law_loss,
            augmented_features=augmented,
        )
