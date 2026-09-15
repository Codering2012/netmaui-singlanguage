"""
Bidirectional State Space Model (BiMamba-2 Architecture) in Keras 3 (JAX Backend)
Uses State Space Duality (SSD) structured linear attention matrix multiplication.
Zero sequential Python loops, pure XLA fused matrix operations.
"""

import math
from typing import Optional
import keras
from keras import layers, ops
from .norm import RMSNorm
from .ffn import SwiGLUFFN


class BiMamba2SSMBlock(layers.Layer):
    """
    Bidirectional Mamba-2 State Space Model block using SSD.
    Captures deep past and future contextual sign kinematics without autograd loops.
    """
    def __init__(
        self,
        d_model: int = 512,
        expand: int = 2,
        headdim: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        ffn_dim: int = 1280,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.expand = expand
        self.d_inner = d_model * expand
        self.headdim = headdim
        self.d_state = d_state
        self.d_conv = d_conv
        self.ffn_dim = ffn_dim
        self.is_causal = is_causal
        self.nheads = self.d_inner // self.headdim
        self.supports_masking = True

    def build(self, input_shape):
        # 1. Input projections: [x, z, B, C, dt]
        # x_inner: d_inner, z_gate: d_inner, B_state: nheads * d_state, C_state: nheads * d_state, dt: nheads
        self.in_proj = layers.Dense(
            self.d_inner * 2 + 2 * self.nheads * self.d_state + self.nheads,
            use_bias=False,
            name="in_proj",
        )

        # 1D Depthwise Conv for local temporal continuity
        self.conv1d = layers.Conv1D(
            filters=self.d_inner + 2 * self.nheads * self.d_state,
            kernel_size=self.d_conv,
            padding="same",
            groups=self.d_inner + 2 * self.nheads * self.d_state,
            name="conv1d",
        )

        # Learnable log-A decay parameters
        self.a_log = self.add_weight(
            shape=(self.nheads,),
            initializer=keras.initializers.RandomUniform(minval=-3.0, maxval=-1.0),
            trainable=True,
            name="a_log",
        )

        # Output projection
        self.out_norm = RMSNorm(self.d_inner, name="out_norm")
        self.out_proj = layers.Dense(self.d_model, use_bias=False, name="out_proj")

        # Second FFN sublayer
        self.ffn_norm = RMSNorm(self.d_model, name="ffn_norm")
        self.ffn = SwiGLUFFN(self.d_model, self.ffn_dim, name="ffn")
        self.final_norm = RMSNorm(self.d_model, name="final_norm")

        super().build(input_shape)

    def _ssd_pass(self, x, dt_soft, log_decay, B, C, is_reverse: bool = False):
        """
        Computes structured SSD linear attention for one direction:
        y = (C (B * dt)^T * decay_mask) * x
        Pre-scaled B projection and precomputed point-wise log_decay.
        """
        # x: (B_sz, T, nheads, headdim)
        # dt_soft: (B_sz, T, nheads)
        # log_decay: (B_sz, T, nheads)
        # B: (B_sz, T, nheads, d_state)
        # C: (B_sz, T, nheads, d_state)
        if is_reverse:
            x = ops.flip(x, axis=1)
            dt_soft = ops.flip(dt_soft, axis=1)
            log_decay = ops.flip(log_decay, axis=1)
            B = ops.flip(B, axis=1)
            C = ops.flip(C, axis=1)

        b_sz = ops.shape(x)[0]
        t_sz = ops.shape(x)[1]

        scale = 1.0 / math.sqrt(self.d_state)
        x_t = ops.transpose(x, (0, 2, 1, 3))
        dt_t = ops.expand_dims(ops.transpose(dt_soft, (0, 2, 1)), -1)  # (B, H, T, 1)
        B_t = (ops.transpose(B, (0, 2, 1, 3)) * dt_t) * scale
        C_t = ops.transpose(C, (0, 2, 1, 3))

        # Semi-separable decay matrix M[i, j] = exp(cumsum(log_decay)[i] - cumsum(log_decay)[j]) for i >= j
        ld_t = ops.transpose(log_decay, (0, 2, 1))  # (B, H, T)
        cum_decay = ops.cumsum(ld_t, axis=-1)  # (B, H, T)
        decay_diff = ops.expand_dims(cum_decay, -1) - ops.expand_dims(cum_decay, -2)  # (B, H, T, T)

        causal_tril = ops.tril(ops.ones((t_sz, t_sz), dtype="bool"))
        decay_mask = ops.where(causal_tril, ops.exp(ops.clip(decay_diff, -40.0, 0.0)), 0.0)  # (B, H, T, T)

        # SSD Gram matrix: CB = (C @ B_scaled^T)
        cb = ops.matmul(C_t, ops.transpose(B_t, (0, 1, 3, 2)))  # (B, H, T, T)
        attn = cb * decay_mask  # (B, H, T, T)

        y = ops.matmul(attn, x_t)  # (B, H, T, headdim)
        y = ops.transpose(y, (0, 2, 1, 3))  # (B, T, nheads, headdim)

        if is_reverse:
            y = ops.flip(y, axis=1)

        return y

    def call(self, u, mask: Optional[any] = None):
        # u: (B, T, d_model)
        b_sz = ops.shape(u)[0]
        t_sz = ops.shape(u)[1]

        # 1. Project input
        proj = self.in_proj(u)

        # Split projections
        d_main = self.d_inner + 2 * self.nheads * self.d_state
        main_part = proj[..., :d_main]
        z_gate = proj[..., d_main : d_main + self.d_inner]
        dt_raw = proj[..., d_main + self.d_inner :]

        # 1D Convolution over sequence
        main_conv = ops.silu(self.conv1d(main_part))

        x_inner = main_conv[..., :self.d_inner]
        B_part = main_conv[..., self.d_inner : self.d_inner + self.nheads * self.d_state]
        C_part = main_conv[..., self.d_inner + self.nheads * self.d_state :]

        x_4d = ops.reshape(x_inner, (b_sz, t_sz, self.nheads, self.headdim))
        B_4d = ops.reshape(B_part, (b_sz, t_sz, self.nheads, self.d_state))
        C_4d = ops.reshape(C_part, (b_sz, t_sz, self.nheads, self.d_state))

        a_val = ops.exp(self.a_log)

        # Precompute point-wise dt_soft and log_decay once (point-wise operations commute with temporal flip)
        dt_soft = ops.clip(ops.softplus(dt_raw), 1e-4, 20.0)
        a_expanded = ops.reshape(a_val, (1, 1, self.nheads))
        log_decay = - ops.clip(dt_soft * a_expanded, 1e-4, 20.0)

        # 2. Forward SSD
        y_fwd = self._ssd_pass(x_4d, dt_soft, log_decay, B_4d, C_4d, is_reverse=False)

        # 3. Backward SSD (if not strictly causal)
        if not self.is_causal:
            y_bwd = self._ssd_pass(x_4d, dt_soft, log_decay, B_4d, C_4d, is_reverse=True)
            y = (y_fwd + y_bwd) * 0.5
        else:
            y = y_fwd

        y_flat = ops.reshape(y, (b_sz, t_sz, self.d_inner))

        # Gated multiplicative projection
        y_gated = y_flat * ops.silu(z_gate)
        out_ssm = self.out_proj(self.out_norm(y_gated))

        # First residual connection
        x = u + out_ssm

        # Second FFN sublayer
        x = x + self.ffn(self.ffn_norm(x))
        return self.final_norm(x)
