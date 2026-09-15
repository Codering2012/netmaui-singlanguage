#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — BIDIRECTIONAL SELECTIVE SCAN MAMBA ENGINE (MAMBASIGN)
================================================================================
Implements Pure PyTorch Hardware-Compatible Bidirectional Selective SSM (Bi-SSM):
1. Selective State-Space Parameters (S6 Formulation):
     h_t = bar{A}_t * h_{t-1} + bar{B}_t * x_t,   y_t = C_t * h_t + D * x_t
2. Input-Dependent Selective Discretization:
     Delta_t = softplus( Linear_Delta(x_t) + bias_Delta )
     B_t = Linear_B(x_t) in R^{B x T x N},  C_t = Linear_C(x_t) in R^{B x T x N}
     bar{A}_t = exp( Delta_t * A ) in R^{B x T x E x N}
     bar{B}_t = (Delta_t * B_t) in R^{B x T x E x N}
3. Bidirectional Temporal Scan:
     y_fwd = SelectiveScan(x)
     y_bwd = Reverse( SelectiveScan( Reverse(x) ) )
     y_out = Linear_out( (SiLU(y_fwd) + SiLU(y_bwd)) * SiLU(z) )
4. O(T) Linear Computational Complexity & O(1) Real-Time Streaming State.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaOutput(NamedTuple):
    mamba_features: torch.Tensor        # [B, T, d_model] Output sequence features
    forward_hidden_states: torch.Tensor # [B, T, E, N] Forward recurrent state trajectory
    backward_hidden_states: torch.Tensor# [B, T, E, N] Backward recurrent state trajectory
    delta_gate: torch.Tensor            # [B, T, E] Discretization step sizes Delta_t
    augmented_features: Optional[torch.Tensor] # [B, T, d_model] x + mamba_features


class ASLBidirectionalSelectiveMambaEngine(nn.Module):
    """
    Bidirectional Selective State-Space (Bi-SSM / Mamba) Engine.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,            # SSM state dimension N
        d_conv: int = 4,              # Local 1D depthwise convolution kernel
        expand: int = 2,              # Expansion factor E = expand * d_model
        dt_min: float = 0.001,
        dt_max: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model  # E (e.g. 256)

        # 1. In-projection: [d_model] -> [2 * d_inner] (x and z gate branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # 2. Local 1D Depthwise Convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        # 3. Parameterize HiPPO Structured Matrix A in R^{d_inner x d_state}
        # Initialized as log(1..N) to ensure negative real diagonal: A = -exp(log_A)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.log_A_fwd = nn.Parameter(torch.log(A))
        self.log_A_bwd = nn.Parameter(torch.log(A))

        # 4. Selective Projections: x_t -> B_t, C_t, Delta_t
        self.x_proj = nn.Linear(self.d_inner, d_state + d_state + self.d_inner, bias=False)

        # Delta parameter initialization
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        # Initialize dt_proj bias to log(uniform(dt_min, dt_max))
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # 5. Direct Skip Parameter D
        self.D_fwd = nn.Parameter(torch.ones(self.d_inner))
        self.D_bwd = nn.Parameter(torch.ones(self.d_inner))

        # 6. Out-projection: [d_inner] -> [d_model]
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def _selective_scan(
        self,
        u: torch.Tensor,       # [B, T, E]
        delta: torch.Tensor,   # [B, T, E]
        A: torch.Tensor,       # [E, N]
        B_mat: torch.Tensor,   # [B, T, N]
        C_mat: torch.Tensor,   # [B, T, N]
        D_param: torch.Tensor, # [E]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes sequential selective SSM scan.
        Returns: (y [B, T, E], h_seq [B, T, E, N])
        """
        B, T, E = u.shape
        N = self.d_state
        device = u.device

        # Continuous-to-Discrete Matrix Transformation
        # delta: [B, T, E, 1], A: [1, 1, E, N]
        # bar_A = exp(delta * A) in [B, T, E, N]
        bar_A = torch.exp(delta.unsqueeze(-1) * A.view(1, 1, E, N))  # [B, T, E, N]
        # bar_B = delta * B in [B, T, E, N]
        bar_B = delta.unsqueeze(-1) * B_mat.unsqueeze(-2)            # [B, T, E, N]

        # Recurrent Scan over Time T
        h_t = torch.zeros(B, E, N, device=device, dtype=u.dtype)
        y_list = []
        h_list = []

        for t in range(T):
            u_t = u[:, t, :].unsqueeze(-1)  # [B, E, 1]
            bar_A_t = bar_A[:, t, :, :]     # [B, E, N]
            bar_B_t = bar_B[:, t, :, :]     # [B, E, N]
            C_t = C_mat[:, t, :].unsqueeze(-2) # [B, 1, N]

            # h_t = bar_A_t * h_{t-1} + bar_B_t * u_t
            h_t = bar_A_t * h_t + bar_B_t * u_t  # [B, E, N]

            # y_t = (C_t * h_t).sum(dim=-1) + D * u_t
            y_t = (C_t * h_t).sum(dim=-1) + D_param * u[:, t, :]  # [B, E]

            y_list.append(y_t)
            h_list.append(h_t)

        y = torch.stack(y_list, dim=1)        # [B, T, E]
        h_seq = torch.stack(h_list, dim=1)    # [B, T, E, N]
        return y, h_seq

    def forward(
        self,
        x: torch.Tensor,                              # [B, T, d_model] Input sequence
        h_state: Optional[torch.Tensor] = None,       # Optional initial state
    ) -> MambaOutput:
        """
        Executes in-projection, depthwise convolution, bidirectional selective scan, and gated out-projection.
        """
        B, T, D = x.shape
        device = x.device

        # 1. In-projection & Branch Splitting
        xz = self.in_proj(x)  # [B, T, 2 * E]
        x_branch, z_branch = xz.chunk(2, dim=-1)  # [B, T, E], [B, T, E]

        # 2. Local 1D Depthwise Causal Convolution along time T
        # Transpose to [B, E, T] for conv1d
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)  # [B, T, E]
        x_act = F.silu(x_conv)  # [B, T, E]

        # 3. Dynamic Selective SSM Projections
        ssm_params = self.x_proj(x_act)  # [B, T, N + N + E]
        B_mat = ssm_params[:, :, :self.d_state]                        # [B, T, N]
        C_mat = ssm_params[:, :, self.d_state: 2 * self.d_state]       # [B, T, N]
        dt_raw = ssm_params[:, :, 2 * self.d_state:]                   # [B, T, E]
        delta = F.softplus(self.dt_proj(dt_raw))                       # [B, T, E]

        # HiPPO A matrices (negative real diagonal)
        A_fwd = -torch.exp(self.log_A_fwd)  # [E, N]
        A_bwd = -torch.exp(self.log_A_bwd)  # [E, N]

        # 4. Bidirectional Selective Scan
        # Forward Scan:
        y_fwd, h_fwd = self._selective_scan(x_act, delta, A_fwd, B_mat, C_mat, self.D_fwd)

        # Backward Scan (Reversed time):
        u_rev = torch.flip(x_act, dims=[1])
        delta_rev = torch.flip(delta, dims=[1])
        B_rev = torch.flip(B_mat, dims=[1])
        C_rev = torch.flip(C_mat, dims=[1])
        y_bwd_rev, h_bwd_rev = self._selective_scan(u_rev, delta_rev, A_bwd, B_rev, C_rev, self.D_bwd)
        y_bwd = torch.flip(y_bwd_rev, dims=[1])
        h_bwd = torch.flip(h_bwd_rev, dims=[1])

        # 5. Gated Gated SiLU Non-linear Fusion with z_branch
        z_gate = F.silu(z_branch)  # [B, T, E]
        y_combined = (F.silu(y_fwd) + F.silu(y_bwd)) * z_gate  # [B, T, E]

        # 6. Out-projection & Residual Norm
        out = self.out_proj(y_combined)  # [B, T, d_model]
        out_norm = self.norm(out + x)    # Residual connection

        return MambaOutput(
            mamba_features=out_norm,
            forward_hidden_states=h_fwd,
            backward_hidden_states=h_bwd,
            delta_gate=delta,
            augmented_features=out_norm,
        )
