"""
Grouped-Query Attention (GQA) with RoPE, Causal Masking, and Hardware-Fused Kernels
Optimized for Cloud TPU v5e & Keras 3 (JAX Backend)
"""

import math
from typing import Optional
import keras
from keras import layers, ops
from .rope import apply_rope


def _scaled_dot_product_attention_dispatch(
    q,
    k,
    v,
    mask: Optional[any] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
):
    """
    Hardware-accelerated fused attention dispatch.
    q: (B, T, N, H)
    k: (B, S, K, H)
    v: (B, S, K, H)
    mask: optional (B, S) boolean mask or 4D broadcastable mask
    """
    head_dim = ops.shape(q)[-1]
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    if keras.config.backend() == "jax":
        import jax.nn as jnn
        mask_4d = None
        if mask is not None:
            b = ops.shape(q)[0]
            t = ops.shape(q)[1]
            s = ops.shape(k)[1]
            m_ndim = len(ops.shape(mask))
            if m_ndim == 2:
                m0, m1 = ops.shape(mask)[0], ops.shape(mask)[1]
                if m0 == t and m1 == s and m0 != b:
                    mask_4d = ops.reshape(mask, (1, 1, t, s))
                else:
                    mask_4d = ops.reshape(mask, (m0, 1, 1, m1))
            elif m_ndim == 3:
                mask_4d = ops.expand_dims(mask, axis=1)
            else:
                mask_4d = mask
        return jnn.dot_product_attention(
            q, k, v, mask=mask_4d, is_causal=is_causal, scale=scale
        )
    else:
        # Fallback for CPU / other backends
        b = ops.shape(q)[0]
        t = ops.shape(q)[1]
        n = ops.shape(q)[2]
        s = ops.shape(k)[1]
        k_h = ops.shape(k)[2]

        q_t = ops.transpose(q, (0, 2, 1, 3))
        k_t = ops.transpose(k, (0, 2, 1, 3))
        v_t = ops.transpose(v, (0, 2, 1, 3))

        groups = n // k_h
        if groups > 1:
            k_t = ops.repeat(k_t, groups, axis=1)
            v_t = ops.repeat(v_t, groups, axis=1)

        scores = ops.matmul(q_t, ops.transpose(k_t, (0, 1, 3, 2))) * scale

        if is_causal:
            causal_mask = ops.tril(ops.ones((t, s), dtype="bool"))
            causal_mask = ops.expand_dims(ops.expand_dims(causal_mask, axis=0), axis=1)
            scores = ops.where(causal_mask, scores, -1e9)

        if mask is not None:
            m_ndim = len(ops.shape(mask))
            if m_ndim == 2:
                m0, m1 = ops.shape(mask)[0], ops.shape(mask)[1]
                if m0 == t and m1 == s and m0 != b:
                    m_exp = ops.reshape(mask, (1, 1, t, s))
                else:
                    m_exp = ops.reshape(mask, (m0, 1, 1, m1))
            elif m_ndim == 3:
                m_exp = ops.expand_dims(mask, axis=1)
            else:
                m_exp = mask
            scores = ops.where(m_exp, scores, -1e9)

        attn_weights = ops.softmax(scores, axis=-1)
        context = ops.matmul(attn_weights, v_t)
        return ops.transpose(context, (0, 2, 1, 3))


class GroupedQueryEncoderAttention(layers.Layer):
    """GQA Self-Attention with RoPE for Conformer Encoder."""
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        max_len: int = 384,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.max_len = max_len
        self.head_dim = d_model // nhead
        self.supports_masking = True
        assert self.nhead % self.kv_heads == 0, "nhead must be divisible by kv_heads"


    def build(self, input_shape):
        self.q_proj = layers.Dense(self.d_model, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(self.kv_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(self.kv_heads * self.head_dim, use_bias=False, name="v_proj")
        self.out_proj = layers.Dense(self.d_model, use_bias=False, name="out_proj")
        super().build(input_shape)

    def call(
        self,
        x,
        mask: Optional[any] = None,
        cos: Optional[any] = None,
        sin: Optional[any] = None,
    ):
        shape = ops.shape(x)
        b, seq_len = shape[0], shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape directly to native (B, L, H, D)
        q = ops.reshape(q, (b, seq_len, self.nhead, self.head_dim))
        k = ops.reshape(k, (b, seq_len, self.kv_heads, self.head_dim))
        v = ops.reshape(v, (b, seq_len, self.kv_heads, self.head_dim))

        # Apply RoPE on native (B, L, H, D)
        rope_dim = (self.head_dim // 2) - ((self.head_dim // 2) % 2)
        q_rope, q_nop = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_nop = k[..., :rope_dim], k[..., rope_dim:]

        if cos is not None and sin is not None:
            q_rope = apply_rope(q_rope, cos, sin, layout="BLHD")
            k_rope = apply_rope(k_rope, cos, sin, layout="BLHD")

        q = ops.concatenate([q_rope, q_nop], axis=-1)
        k = ops.concatenate([k_rope, k_nop], axis=-1)

        # Hardware-accelerated attention
        context = _scaled_dot_product_attention_dispatch(
            q, k, v, mask=mask, is_causal=False
        )

        # Zero-copy reshape back to (B, L, d_model)
        context = ops.reshape(context, (b, seq_len, self.d_model))
        return self.out_proj(context)


class CausalGroupedQueryAttention(layers.Layer):
    """Causal GQA with RoPE for Transformer Decoder."""
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        max_len: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.max_len = max_len
        self.head_dim = d_model // nhead
        self.supports_masking = True
        assert self.nhead % self.kv_heads == 0, "nhead must be divisible by kv_heads"


    def build(self, input_shape):
        self.q_proj = layers.Dense(self.d_model, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(self.kv_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(self.kv_heads * self.head_dim, use_bias=False, name="v_proj")
        self.out_proj = layers.Dense(self.d_model, use_bias=False, name="out_proj")
        super().build(input_shape)

    def call(
        self,
        x,
        mask: Optional[any] = None,
        cos: Optional[any] = None,
        sin: Optional[any] = None,
    ):
        shape = ops.shape(x)
        b, seq_len = shape[0], shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape directly to native (B, L, H, D)
        q = ops.reshape(q, (b, seq_len, self.nhead, self.head_dim))
        k = ops.reshape(k, (b, seq_len, self.kv_heads, self.head_dim))
        v = ops.reshape(v, (b, seq_len, self.kv_heads, self.head_dim))

        # Apply RoPE on native (B, L, H, D)
        rope_dim = (self.head_dim // 2) - ((self.head_dim // 2) % 2)
        q_rope, q_nop = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_nop = k[..., :rope_dim], k[..., rope_dim:]

        if cos is not None and sin is not None:
            q_rope = apply_rope(q_rope, cos, sin, layout="BLHD")
            k_rope = apply_rope(k_rope, cos, sin, layout="BLHD")

        q = ops.concatenate([q_rope, q_nop], axis=-1)
        k = ops.concatenate([k_rope, k_nop], axis=-1)

        # Hardware-accelerated Causal Attention
        context = _scaled_dot_product_attention_dispatch(
            q, k, v, mask=mask, is_causal=True
        )

        # Zero-copy reshape back to (B, L, d_model)
        context = ops.reshape(context, (b, seq_len, self.d_model))
        return self.out_proj(context)


class DecoderCrossAttention(layers.Layer):
    """Cross-Attention querying visual encoder representations."""
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.head_dim = d_model // nhead
        self.supports_masking = True
        assert self.nhead % self.kv_heads == 0, "nhead must be divisible by kv_heads"


    def build(self, input_shape):
        self.q_proj = layers.Dense(self.d_model, use_bias=False, name="q_proj")
        self.k_proj = layers.Dense(self.kv_heads * self.head_dim, use_bias=False, name="k_proj")
        self.v_proj = layers.Dense(self.kv_heads * self.head_dim, use_bias=False, name="v_proj")
        self.out_proj = layers.Dense(self.d_model, use_bias=False, name="out_proj")
        super().build(input_shape)

    def call(self, q_in, memory, memory_mask: Optional[any] = None):
        q_shape = ops.shape(q_in)
        m_shape = ops.shape(memory)
        b, q_len = q_shape[0], q_shape[1]
        kv_len = m_shape[1]

        q = self.q_proj(q_in)
        k = self.k_proj(memory)
        v = self.v_proj(memory)

        # Reshape directly to native (B, L, H, D)
        q = ops.reshape(q, (b, q_len, self.nhead, self.head_dim))
        k = ops.reshape(k, (b, kv_len, self.kv_heads, self.head_dim))
        v = ops.reshape(v, (b, kv_len, self.kv_heads, self.head_dim))

        # Hardware-accelerated Cross Attention
        context = _scaled_dot_product_attention_dispatch(
            q, k, v, mask=memory_mask, is_causal=False
        )

        # Zero-copy reshape back to (B, L, d_model)
        context = ops.reshape(context, (b, q_len, self.d_model))
        return self.out_proj(context)
