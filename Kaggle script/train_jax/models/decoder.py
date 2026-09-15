"""
Flax Linen Implementation of ASLTransformerDecoder
Supports Causal Self-Attention, Cross-Attention over Visual Landmarks, and GQA
"""

import math
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from .conformer import RMSNorm, SwiGLUFFN, get_rotary_frequencies, apply_rope


class CausalGroupedQueryAttention(nn.Module):
    """Causal GQA with RoPE for Transformer Decoder."""
    d_model: int = 512
    nhead: int = 8
    kv_heads: int = 2
    max_len: int = 256

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

        q = nn.Dense(self.d_model, use_bias=False, name="q_proj")(x)
        k = nn.Dense(self.kv_heads * head_dim, use_bias=False, name="k_proj")(x)
        v = nn.Dense(self.kv_heads * head_dim, use_bias=False, name="v_proj")(x)

        q = q.reshape((b, seq_len, self.nhead, head_dim)).swapaxes(1, 2)
        k = k.reshape((b, seq_len, self.kv_heads, head_dim)).swapaxes(1, 2)
        v = v.reshape((b, seq_len, self.kv_heads, head_dim)).swapaxes(1, 2)

        # Apply RoPE
        rope_dim = head_dim // 2
        q_rope, q_nop = q[..., :rope_dim], q[..., rope_dim:]
        k_rope, k_nop = k[..., :rope_dim], k[..., rope_dim:]

        if cos is not None and sin is not None:
            q_rope = apply_rope(q_rope, cos, sin)
            k_rope = apply_rope(k_rope, cos, sin)

        q = jnp.concatenate([q_rope, q_nop], axis=-1)
        k = jnp.concatenate([k_rope, k_nop], axis=-1)

        # GQA Repeat
        groups = self.nhead // self.kv_heads
        if groups > 1:
            k = jnp.repeat(k, groups, axis=1)
            v = jnp.repeat(v, groups, axis=1)

        scale = 1.0 / math.sqrt(head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        # Apply causal mask
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=bool))
        scores = jnp.where(causal_mask[None, None, :, :], scores, -1e9)

        if mask is not None:
            scores = jnp.where(mask[:, None, None, :], scores, -1e9)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        context = context.swapaxes(1, 2).reshape((b, seq_len, self.d_model))
        return nn.Dense(self.d_model, use_bias=False, name="out_proj")(context)


class DecoderCrossAttention(nn.Module):
    """Multi-Head Cross Attention querying encoder visual representations."""
    d_model: int = 512
    nhead: int = 8
    kv_heads: int = 2

    @nn.compact
    def __call__(
        self,
        q_in: jnp.ndarray,
        memory: jnp.ndarray,
        memory_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        b, q_len, _ = q_in.shape
        _, kv_len, _ = memory.shape
        head_dim = self.d_model // self.nhead

        q = nn.Dense(self.d_model, use_bias=False, name="q_proj")(q_in)
        k = nn.Dense(self.kv_heads * head_dim, use_bias=False, name="k_proj")(memory)
        v = nn.Dense(self.kv_heads * head_dim, use_bias=False, name="v_proj")(memory)

        q = q.reshape((b, q_len, self.nhead, head_dim)).swapaxes(1, 2)
        k = k.reshape((b, kv_len, self.kv_heads, head_dim)).swapaxes(1, 2)
        v = v.reshape((b, kv_len, self.kv_heads, head_dim)).swapaxes(1, 2)

        groups = self.nhead // self.kv_heads
        if groups > 1:
            k = jnp.repeat(k, groups, axis=1)
            v = jnp.repeat(v, groups, axis=1)

        scale = 1.0 / math.sqrt(head_dim)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale

        if memory_mask is not None:
            scores = jnp.where(memory_mask[:, None, None, :], scores, -1e9)

        attn_weights = jax.nn.softmax(scores, axis=-1)
        context = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)
        context = context.swapaxes(1, 2).reshape((b, q_len, self.d_model))
        return nn.Dense(self.d_model, use_bias=False, name="out_proj")(context)


class ASLDecoderLayer(nn.Module):
    """Single ASL Decoder Layer with Causal Self-Attn, Cross-Attn, and SwiGLU FFN."""
    d_model: int = 512
    nhead: int = 8
    kv_heads: int = 2
    ffn_dim: int = 1280
    max_seq_len: int = 256
    init_values: float = 0.1

    @nn.compact
    def __call__(
        self,
        tgt: jnp.ndarray,
        memory: Optional[jnp.ndarray] = None,
        tgt_mask: Optional[jnp.ndarray] = None,
        memory_mask: Optional[jnp.ndarray] = None,
        cos: Optional[jnp.ndarray] = None,
        sin: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        gamma1 = self.param("gamma1", lambda rng, shape: self.init_values * jnp.ones(shape), (self.d_model,))
        gamma2 = self.param("gamma2", lambda rng, shape: self.init_values * jnp.ones(shape), (self.d_model,))
        gamma3 = self.param("gamma3", lambda rng, shape: self.init_values * jnp.ones(shape), (self.d_model,))

        # 1. Causal Self-Attention
        norm1 = RMSNorm(self.d_model, name="norm1")(tgt)
        sa_out = CausalGroupedQueryAttention(
            d_model=self.d_model, nhead=self.nhead, kv_heads=self.kv_heads, max_len=self.max_seq_len, name="self_attn"
        )(norm1, mask=tgt_mask, cos=cos, sin=sin)
        tgt = tgt + gamma1 * sa_out

        # 2. Cross-Attention over Visual Encoder Memory
        if memory is not None:
            norm2 = RMSNorm(self.d_model, name="norm2")(tgt)
            ca_out = DecoderCrossAttention(
                d_model=self.d_model, nhead=self.nhead, kv_heads=self.kv_heads, name="cross_attn"
            )(norm2, memory, memory_mask=memory_mask)
            tgt = tgt + gamma2 * ca_out

        # 3. SwiGLU Feed-Forward Network
        norm3 = RMSNorm(self.d_model, name="norm3")(tgt)
        ffn_out = SwiGLUFFN(self.d_model, self.ffn_dim, name="ffn")(norm3)
        tgt = tgt + gamma3 * ffn_out
        return tgt


class ASLTransformerDecoder(nn.Module):
    """Full Transformer Decoder for ASL Gloss Sequence Generation."""
    vocab_size: int
    d_model: int = 512
    nhead: int = 4
    kv_heads: int = 2
    num_layers: int = 4
    ffn_dim: int = 1280
    max_seq_len: int = 256

    @nn.compact
    def __call__(
        self,
        token_ids: jnp.ndarray,
        memory: Optional[jnp.ndarray] = None,
        memory_mask: Optional[jnp.ndarray] = None,
        token_mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # token_ids: (B, L_tgt)
        b, seq_len = token_ids.shape
        tok_emb = nn.Embed(self.vocab_size, self.d_model, name="token_embeddings")(token_ids)

        head_dim = self.d_model // self.nhead
        cos, sin = get_rotary_frequencies(head_dim, max_len=self.max_seq_len)

        x = tok_emb
        for i in range(self.num_layers):
            x = ASLDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                kv_heads=self.kv_heads,
                ffn_dim=self.ffn_dim,
                max_seq_len=self.max_seq_len,
                name=f"layer_{i}",
            )(x, memory=memory, tgt_mask=token_mask, memory_mask=memory_mask, cos=cos, sin=sin)

        x = RMSNorm(self.d_model, name="final_norm")(x)
        logits = nn.Dense(self.vocab_size, use_bias=False, name="lm_head")(x)
        return logits
