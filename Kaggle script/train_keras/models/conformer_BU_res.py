"""
MobileConformer Architecture in Keras 3 (JAX Backend)
Optimized for Cloud TPU v5e Systolic Matrix Multiply Units (128x128)
"""

from typing import Optional, List
import keras
from keras import layers, ops
from ..layers import (
    RMSNorm,
    SwiGLUFFN,
    get_rotary_frequencies,
    GroupedQueryEncoderAttention,
    ConvNeXtTemporalBlock,
)


class ConformerBlock(layers.Layer):
    """Macaron-style MobileConformer Layer block."""
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        dim_feedforward: int = 1280,
        kernel_size: int = 31,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.dim_feedforward = dim_feedforward
        self.kernel_size = kernel_size
        self.is_causal = is_causal
        self.supports_masking = True


    def build(self, input_shape):
        # 1. FFN 1
        self.ffn1_norm = RMSNorm(self.d_model, name="ffn1_norm")
        self.ffn1 = SwiGLUFFN(self.d_model, self.dim_feedforward, name="ffn1")

        # 2. Self Attention
        self.self_attn_norm = RMSNorm(self.d_model, name="self_attn_norm")
        self.self_attn = GroupedQueryEncoderAttention(
            d_model=self.d_model,
            nhead=self.nhead,
            kv_heads=self.kv_heads,
            name="self_attn",
        )

        # 3. Convolution
        self.conv_norm = RMSNorm(self.d_model, name="conv_norm")
        self.conv = ConvNeXtTemporalBlock(
            channels=self.d_model,
            kernel_size=self.kernel_size,
            is_causal=self.is_causal,
            name="conv",
        )

        # 4. FFN 2
        self.ffn2_norm = RMSNorm(self.d_model, name="ffn2_norm")
        self.ffn2 = SwiGLUFFN(self.d_model, self.dim_feedforward, name="ffn2")

        # 5. Final Norm
        self.final_norm = RMSNorm(self.d_model, name="final_norm")
        super().build(input_shape)

    def call(
        self,
        x,
        mask: Optional[any] = None,
        cos: Optional[any] = None,
        sin: Optional[any] = None,
    ):
        # 1. Half-step FFN 1
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x))

        # 2. Self Attention with RoPE
        x = x + self.self_attn(self.self_attn_norm(x), mask=mask, cos=cos, sin=sin)

        # 3. ConvNeXt Temporal Block
        x = self.conv(self.conv_norm(x), mask=mask)

        # 4. Half-step FFN 2
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x))

        # 5. Final RMSNorm
        return self.final_norm(x)


class MobileConformerEncoder(keras.Model):
    """Full MobileConformer Visual Landmark Encoder."""
    def __init__(
        self,
        num_layers: int = 4,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        dim_feedforward: int = 1280,
        in_channels: int = 225,
        max_len: int = 384,
        kernel_size: int = 31,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.dim_feedforward = dim_feedforward
        self.in_channels = in_channels
        self.max_len = max_len
        self.kernel_size = kernel_size
        self.is_causal = is_causal
        self.supports_masking = True


    def build(self, input_shape):
        self.input_proj = layers.Dense(self.d_model, use_bias=True, name="input_proj")
        self.input_norm = RMSNorm(self.d_model, name="input_norm")

        self.blocks = [
            ConformerBlock(
                d_model=self.d_model,
                nhead=self.nhead,
                kv_heads=self.kv_heads,
                dim_feedforward=self.dim_feedforward,
                kernel_size=self.kernel_size,
                is_causal=self.is_causal,
                name=f"block_{i}",
            )
            for i in range(self.num_layers)
        ]
        self.final_norm = RMSNorm(self.d_model, name="encoder_final_norm")

        # Precompute rotary tables for head_dim
        head_dim = self.d_model // self.nhead
        rope_dim = (head_dim // 2) - ((head_dim // 2) % 2)
        cos, sin = get_rotary_frequencies(self.max_len, rope_dim)
        self.cos = self.add_weight(shape=ops.shape(cos), initializer="zeros", trainable=False, name="rope_cos")
        self.sin = self.add_weight(shape=ops.shape(sin), initializer="zeros", trainable=False, name="rope_sin")
        self.cos.assign(cos)
        self.sin.assign(sin)

        super().build(input_shape)

    def call(self, x, mask: Optional[any] = None):
        # x: (B, L, in_channels)
        x = self.input_proj(x)
        x = self.input_norm(x)

        seq_len = ops.shape(x)[1]
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]

        for block in self.blocks:
            x = block(x, mask=mask, cos=cos, sin=sin)

        return self.final_norm(x)
