"""
ASLTransformerDecoder in Keras 3 (JAX Backend)
Supports Causal Self-Attention, Cross-Attention over Visual Representations, and GQA
"""

import math
from typing import Optional
import keras
from keras import layers, ops
from ..layers import (
    RMSNorm,
    SwiGLUFFN,
    get_rotary_frequencies,
    CausalGroupedQueryAttention,
    DecoderCrossAttention,
)


class ASLDecoderLayer(layers.Layer):
    """Transformer Decoder layer with Causal Self-Attention and Cross-Attention."""
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        dim_feedforward: int = 1280,
        max_len: int = 256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.dim_feedforward = dim_feedforward
        self.max_len = max_len
        self.supports_masking = True


    def build(self, input_shape):
        # 1. Causal Self-Attention
        self.self_attn_norm = RMSNorm(self.d_model, name="self_attn_norm")
        self.self_attn = CausalGroupedQueryAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            kv_heads=self.kv_heads,
            max_len=self.max_len,
            name="self_attn",
        )

        # 2. Cross-Attention over Visual Memory
        self.cross_attn_norm = RMSNorm(self.d_model, name="cross_attn_norm")
        self.cross_attn = DecoderCrossAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            kv_heads=self.kv_heads,
            name="cross_attn",
        )

        # 3. SwiGLU FFN
        self.ffn_norm = RMSNorm(self.d_model, name="ffn_norm")
        self.ffn = SwiGLUFFN(self.d_model, self.dim_feedforward, name="ffn")
        super().build(input_shape)

    def call(
        self,
        x,
        memory,
        self_mask: Optional[any] = None,
        memory_mask: Optional[any] = None,
        cos: Optional[any] = None,
        sin: Optional[any] = None,
    ):
        # 1. Causal Self-Attention with RoPE
        x = x + self.self_attn(self.self_attn_norm(x), mask=self_mask, cos=cos, sin=sin)

        # 2. Cross-Attention over visual/prompt memory (optional for Causal LM)
        if memory is not None:
            x = x + self.cross_attn(self.cross_attn_norm(x), memory, memory_mask=memory_mask)

        # 3. SwiGLU FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


class ASLTransformerDecoder(keras.Model):
    """Multi-layer ASL Transformer Decoder."""
    def __init__(
        self,
        vocab_size: int = 17800,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        num_layers: int = 4,
        dim_feedforward: int = 1280,
        max_seq_len: int = 256,
        tie_word_embeddings: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.max_seq_len = max_seq_len
        self.tie_word_embeddings = tie_word_embeddings
        self.embed_scale = math.sqrt(d_model)
        self.supports_masking = True


    def build(self, input_shape):
        self.embed_tokens = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.d_model,
            name="embed_tokens",
        )

        self.layers_list = [
            ASLDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                kv_heads=self.kv_heads,
                dim_feedforward=self.dim_feedforward,
                max_len=self.max_seq_len,
                name=f"layer_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = RMSNorm(self.d_model, name="final_norm")
        if not self.tie_word_embeddings:
            self.lm_head = layers.Dense(self.vocab_size, use_bias=False, name="lm_head")
        else:
            self.lm_head = None

        # RoPE tables
        head_dim = self.d_model // self.nhead
        rope_dim = (head_dim // 2) - ((head_dim // 2) % 2)
        cos, sin = get_rotary_frequencies(self.max_seq_len, rope_dim)
        self.cos = self.add_weight(shape=ops.shape(cos), initializer="zeros", trainable=False, name="rope_cos")
        self.sin = self.add_weight(shape=ops.shape(sin), initializer="zeros", trainable=False, name="rope_sin")
        self.cos.assign(cos)
        self.sin.assign(sin)

        super().build(input_shape)

    def call(
        self,
        tokens,
        memory: Optional[any] = None,
        self_mask: Optional[any] = None,
        memory_mask: Optional[any] = None,
    ):
        # tokens: (B, L_tgt)
        seq_len = ops.shape(tokens)[1]
        x = self.embed_tokens(tokens) * self.embed_scale

        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        for layer in self.layers_list:
            x = layer(
                x,
                memory=memory,
                self_mask=self_mask,
                memory_mask=memory_mask,
                cos=cos,
                sin=sin,
            )

        x = self.final_norm(x)
        if self.tie_word_embeddings:
            # Reuses embed_tokens.embeddings: (B, T, D) @ (D, V) -> (B, T, V)
            return ops.matmul(x, ops.transpose(self.embed_tokens.embeddings))
        return self.lm_head(x)
