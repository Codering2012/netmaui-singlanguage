"""
Grouped-Query Attention (GQA) with RoPE and Causal Masking in Keras 3 (JAX Backend)
"""

import math
from typing import Optional
import keras
from keras import layers, ops
from .rope import apply_rope


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

        # Reshape to (B, H, L, D_head)
        q = ops.reshape(q, (b, seq_len, self.nhead, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        k = ops.reshape(k, (b, seq_len, self.kv_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))

        v = ops.reshape(v, (b, seq_len, self.kv_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))

        # Apply RoPE
        rope_dim = (self.head_dim // 2) - ((self.head_dim // 2) % 2)
        q_rope, q_nop = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_nop = k[..., :rope_dim], k[..., rope_dim:]

        if cos is not None and sin is not None:
            q_rope = apply_rope(q_rope, cos, sin)
            k_rope = apply_rope(k_rope, cos, sin)

        q = ops.concatenate([q_rope, q_nop], axis=-1)
        k = ops.concatenate([k_rope, k_nop], axis=-1)

        # GQA Repeat
        groups = self.nhead // self.kv_heads
        if groups > 1:
            k = ops.repeat(k, groups, axis=1)
            v = ops.repeat(v, groups, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

        if mask is not None:
            # mask: (B, L)
            mask_expanded = ops.expand_dims(ops.expand_dims(mask, axis=1), axis=2)
            scores = ops.where(mask_expanded, scores, -1e9)

        attn_weights = ops.softmax(scores, axis=-1)
        context = ops.matmul(attn_weights, v)

        # Reshape back to (B, L, d_model)
        context = ops.transpose(context, (0, 2, 1, 3))
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

        q = ops.reshape(q, (b, seq_len, self.nhead, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        k = ops.reshape(k, (b, seq_len, self.kv_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))

        v = ops.reshape(v, (b, seq_len, self.kv_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))

        # RoPE
        rope_dim = (self.head_dim // 2) - ((self.head_dim // 2) % 2)
        q_rope, q_nop = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_nop = k[..., :rope_dim], k[..., rope_dim:]

        if cos is not None and sin is not None:
            q_rope = apply_rope(q_rope, cos, sin)
            k_rope = apply_rope(k_rope, cos, sin)

        q = ops.concatenate([q_rope, q_nop], axis=-1)
        k = ops.concatenate([k_rope, k_nop], axis=-1)

        groups = self.nhead // self.kv_heads
        if groups > 1:
            k = ops.repeat(k, groups, axis=1)
            v = ops.repeat(v, groups, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

        # Apply Causal Mask
        causal_mask = ops.tril(ops.ones((seq_len, seq_len), dtype="bool"))
        causal_mask = ops.expand_dims(ops.expand_dims(causal_mask, axis=0), axis=1)
        scores = ops.where(causal_mask, scores, -1e9)

        if mask is not None:
            mask_expanded = ops.expand_dims(ops.expand_dims(mask, axis=1), axis=2)
            scores = ops.where(mask_expanded, scores, -1e9)

        attn_weights = ops.softmax(scores, axis=-1)
        context = ops.matmul(attn_weights, v)

        context = ops.transpose(context, (0, 2, 1, 3))
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

        q = ops.reshape(q, (b, q_len, self.nhead, self.head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        k = ops.reshape(k, (b, kv_len, self.kv_heads, self.head_dim))
        k = ops.transpose(k, (0, 2, 1, 3))

        v = ops.reshape(v, (b, kv_len, self.kv_heads, self.head_dim))
        v = ops.transpose(v, (0, 2, 1, 3))

        groups = self.nhead // self.kv_heads
        if groups > 1:
            k = ops.repeat(k, groups, axis=1)
            v = ops.repeat(v, groups, axis=1)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = ops.matmul(q, ops.transpose(k, (0, 1, 3, 2))) * scale

        if memory_mask is not None:
            mask_expanded = ops.expand_dims(ops.expand_dims(memory_mask, axis=1), axis=2)
            scores = ops.where(mask_expanded, scores, -1e9)

        attn_weights = ops.softmax(scores, axis=-1)
        context = ops.matmul(attn_weights, v)

        context = ops.transpose(context, (0, 2, 1, 3))
        context = ops.reshape(context, (b, q_len, self.d_model))
        return self.out_proj(context)
