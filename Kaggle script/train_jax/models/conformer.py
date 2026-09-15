"""
Flax Linen Implementation of MobileConformer Architecture
Optimized for TPU v5e / TPU7x Systolic Matrix Multiply Units (128x128 / 256x256)
"""

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class RMSNorm(nn.Module):
    """Fused Root Mean Square Layer Normalization."""
    d_model: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale = self.param("scale", nn.initializers.ones, (self.d_model,))
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
        normed = (x_f32 * jax.lax.rsqrt(variance + self.eps)) * scale
        return normed.astype(x.dtype)


class SwiGLUFFN(nn.Module):
    """Fused SwiGLU Feed-Forward Network tiled for systolic MXUs."""
    d_model: int
    dim_feedforward: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Align hidden dimension to 128 for hardware tile efficiency
        hidden = ((int(self.dim_feedforward * 2 / 3) + 127) // 128) * 128
        gate_up = nn.Dense(2 * hidden, use_bias=False, name="w_gate_up")(x)
        gate, up = jnp.split(gate_up, 2, axis=-1)
        return nn.Dense(self.d_model, use_bias=False, name="w_down")(nn.silu(gate) * up)


class SpatialTemporalSE(nn.Module):
    """Squeeze-and-Excitation temporal attention module."""
    channels: int
    reduction: int = 4
    is_causal: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # x: (B, L, C)
        reduced = max(16, self.channels // self.reduction)
        if mask is not None:
            mask_f = mask[..., None].astype(x.dtype)
            mean_pooled = jnp.sum(x * mask_f, axis=1, keepdims=True) / jnp.maximum(1.0, jnp.sum(mask_f, axis=1, keepdims=True))
        else:
            mean_pooled = jnp.mean(x, axis=1, keepdims=True)

        fc1 = nn.Dense(reduced, use_bias=False, name="fc1")(mean_pooled)
        act = nn.gelu(fc1)
        fc2 = nn.Dense(self.channels, use_bias=False, name="fc2")(act)
        scale = nn.sigmoid(fc2)
        return x * scale


class ConvNeXtTemporalBlock(nn.Module):
    """1D Depthwise ConvNeXt block with SE gating."""
    channels: int
    kernel_size: int = 7
    expansion: int = 2
    is_causal: bool = False

    @nn.compact
    def __call__(self, x: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # x: (B, L, C)
        b, seq_len, c = x.shape
        pad_size = self.kernel_size - 1

        if self.is_causal:
            # Causal left padding
            padded_x = jnp.pad(x, ((0, 0), (pad_size, 0), (0, 0)))
        else:
            # Symmetric padding
            p_left = pad_size // 2
            p_right = pad_size - p_left
            padded_x = jnp.pad(x, ((0, 0), (p_left, p_right), (0, 0)))

        # Depthwise 1D Conv over temporal dimension
        dw = nn.Conv(
            features=c,
            kernel_size=(self.kernel_size,),
            feature_group_count=c,
            use_bias=False,
            padding="VALID",
            name="dw_conv",
        )(padded_x)

        normed = RMSNorm(c, name="norm")(dw)
        pw1 = nn.Dense(c * self.expansion, use_bias=True, name="pw_conv1")(normed)
        act = nn.gelu(pw1)
        pw2 = nn.Dense(c, use_bias=True, name="pw_conv2")(act)
        se = SpatialTemporalSE(c, is_causal=self.is_causal, name="se")(pw2, mask=mask)
        return se


def get_rotary_frequencies(head_dim: int, max_len: int = 512, theta: float = 10000.0) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generates precomputed cos/sin frequency tensors for Rotary Positional Encoding."""
    dim = head_dim // 2
    pos = jnp.arange(max_len, dtype=jnp.float32)
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    freqs = jnp.einsum("i,j->ij", pos, inv_freq)
    emb = jnp.concatenate([freqs, freqs], axis=-1)
    cos = jnp.cos(emb)
    sin = jnp.sin(emb)
    return cos, sin


def apply_rope(x: jnp.ndarray, cos: jnp.ndarray, sin: jnp.ndarray) -> jnp.ndarray:
    """Applies RoPE rotation to query or key tensor: x shape (B, N, L, D)."""
    # x shape: (B, N, L, D) where D is rope_dim
    seq_len = x.shape[2]
    c = cos[:seq_len, :][None, None, :, :]
    s = sin[:seq_len, :][None, None, :, :]
    d = x.shape[-1]
    half = d // 2
    x1, x2 = x[..., :half], x[..., half:]
    x_rot = jnp.concatenate([-x2, x1], axis=-1)
    return (x * c) + (x_rot * s)


class GroupedQueryEncoderAttention(nn.Module):
    """Multi-Head Grouped Query Attention with DeepSeek MLA Latent Compression and RoPE."""
    d_model: int = 512
    nhead: int = 8
    kv_heads: int = 2
    max_len: int = 512
    is_causal: bool = False

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        cos: Optional[jnp.ndarray] = None,
        sin: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        b, seq_len, _ = x.shape
        head_dim = self.d_model // self.nhead
        latent_dim = self.d_model // 4

        # Pre-LN
        q_in = RMSNorm(self.d_model, name="q_norm")(x)

        # DeepSeek V3 MLA Latent Compression
        kv_latent = nn.Dense(latent_dim, use_bias=False, name="kv_latent_proj")(x)
        kv_latent = RMSNorm(latent_dim, name="kv_latent_norm")(kv_latent)

        # Q and KV projections
        q = nn.Dense(self.d_model, use_bias=False, name="q_proj")(q_in)
        kv = nn.Dense(2 * self.kv_heads * head_dim, use_bias=False, name="kv_proj")(kv_latent)

        q = q.reshape((b, seq_len, self.nhead, head_dim)).swapaxes(1, 2)  # (B, H, L, D)
        kv = kv.reshape((b, seq_len, 2, self.kv_heads, head_dim))
        k = kv[:, :, 0].swapaxes(1, 2)  # (B, KV_H, L, D)
        v = kv[:, :, 1].swapaxes(1, 2)  # (B, KV_H, L, D)

        # RoPE on half of head dimension
        rope_dim = head_dim // 2
        q_rope, q_nop = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_nop = k[..., :rope_dim], k[..., rope_dim:]

        if cos is not None and sin is not None:
            q_rope = apply_rope(q_rope, cos, sin)
            k_rope = apply_rope(k_rope, cos, sin)

        q = jnp.concatenate([q_rope, q_nop], axis=-1)
        k = jnp.concatenate([k_rope, k_nop], axis=-1)

        # Repeat KV heads for Grouped Query Attention
        groups = self.nhead // self.kv_heads
        if groups > 1:
            k = jnp.repeat(k, groups, axis=1)
            v = jnp.repeat(v, groups, axis=1)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        if mask is not None:
            # mask: (B, L) -> (B, 1, 1, L)
            mask_expanded = mask[:, None, None, :]
            scores = jnp.where(mask_expanded, scores, -1e9)

        if self.is_causal:
            causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
            scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        context = context.swapaxes(1, 2).reshape((b, seq_len, self.d_model))
        return nn.Dense(self.d_model, use_bias=False, name="out_proj")(context)


class MobileConformerBlock(nn.Module):
    """Single MobileConformer Block combining Macaron FFNs, GQA Attention, and ConvNeXt."""
    d_model: int = 512
    nhead: int = 8
    kv_heads: int = 2
    dim_feedforward: int = 1280
    max_len: int = 384
    is_causal: bool = False
    init_values: float = 0.1

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        cos: Optional[jnp.ndarray] = None,
        sin: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # LayerScale parameters
        gamma_ffn1 = self.param("gamma_ffn1", lambda rng, shape: self.init_values * jnp.ones(shape), (self.d_model,))
        gamma_mha = self.param("gamma_mha", lambda rng, shape: self.init_values * jnp.ones(shape), (self.d_model,))
        gamma_conv = self.param("gamma_conv", lambda rng, shape: self.init_values * jnp.ones(shape), (self.d_model,))
        gamma_ffn2 = self.param("gamma_ffn2", lambda rng, shape: self.init_values * jnp.ones(shape), (self.d_model,))

        # 1. Macaron FFN1 (0.5 residual scale)
        ffn1_out = SwiGLUFFN(self.d_model, self.dim_feedforward, name="ffn1")(RMSNorm(self.d_model, name="ffn1_norm")(x))
        x = x + 0.5 * gamma_ffn1 * ffn1_out

        # 2. Grouped Query Attention with RoPE
        mha_out = GroupedQueryEncoderAttention(
            d_model=self.d_model, nhead=self.nhead, kv_heads=self.kv_heads, max_len=self.max_len, is_causal=self.is_causal, name="mha"
        )(x, mask=mask, cos=cos, sin=sin)
        x = x + gamma_mha * mha_out

        # 3. Depthwise ConvNeXt block
        conv_out = ConvNeXtTemporalBlock(self.d_model, is_causal=self.is_causal, name="conv_block")(
            RMSNorm(self.d_model, name="conv_norm")(x), mask=mask
        )
        x = x + gamma_conv * conv_out

        # 4. Macaron FFN2 (0.5 residual scale)
        ffn2_out = SwiGLUFFN(self.d_model, self.dim_feedforward, name="ffn2")(RMSNorm(self.d_model, name="ffn2_norm")(x))
        x = x + 0.5 * gamma_ffn2 * ffn2_out
        return x


class MobileConformerEncoder(nn.Module):
    """Stack of MobileConformer blocks preceded by a 1D landmark projection stem."""
    num_layers: int = 4
    d_model: int = 512
    nhead: int = 4
    kv_heads: int = 2
    dim_feedforward: int = 1280
    in_channels: int = 225  # 75 landmarks * (x, y, z)
    max_len: int = 384
    is_causal: bool = True

    @nn.compact
    def __call__(self, features: jnp.ndarray, mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        # features: (B, L, in_channels)
        x = nn.Dense(self.d_model, use_bias=True, name="input_stem")(features)
        x = RMSNorm(self.d_model, name="stem_norm")(x)

        # Precompute RoPE cos/sin frequencies
        head_dim = self.d_model // self.nhead
        cos, sin = get_rotary_frequencies(head_dim, max_len=self.max_len)

        for i in range(self.num_layers):
            x = MobileConformerBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                kv_heads=self.kv_heads,
                dim_feedforward=self.dim_feedforward,
                max_len=self.max_len,
                is_causal=self.is_causal,
                name=f"block_{i}",
            )(x, mask=mask, cos=cos, sin=sin)

        x = RMSNorm(self.d_model, name="final_norm")(x)
        return x
