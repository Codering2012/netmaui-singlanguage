"""
ConvNeXt Temporal Convolution & Squeeze-and-Excitation in Keras 3 (JAX Backend)
"""

from typing import Optional
import keras
from keras import layers, ops
from .norm import RMSNorm


class SpatialTemporalSE(layers.Layer):
    """Squeeze-and-Excitation temporal attention module."""
    def __init__(self, channels: int, reduction: int = 4, is_causal: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.is_causal = is_causal
        self.reduced = max(16, channels // reduction)

    def build(self, input_shape):
        self.fc1 = layers.Dense(self.reduced, use_bias=False, name="fc1")
        self.fc2 = layers.Dense(self.channels, use_bias=False, name="fc2")
        super().build(input_shape)

    def call(self, x, mask: Optional[any] = None):
        # x: (B, L, C)
        if mask is not None:
            # mask: (B, L)
            mask_f = ops.cast(ops.expand_dims(mask, axis=-1), ops.dtype(x))
            mean_pooled = ops.sum(x * mask_f, axis=1, keepdims=True) / ops.maximum(1.0, ops.sum(mask_f, axis=1, keepdims=True))
        else:
            mean_pooled = ops.mean(x, axis=1, keepdims=True)

        fc1_out = self.fc1(mean_pooled)
        act = ops.gelu(fc1_out)
        fc2_out = self.fc2(act)
        scale = ops.sigmoid(fc2_out)
        return x * scale


class ConvNeXtTemporalBlock(layers.Layer):
    """1D Depthwise ConvNeXt block with SE gating."""
    def __init__(
        self,
        channels: int,
        kernel_size: int = 7,
        expansion: int = 2,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.is_causal = is_causal
        self.pad_size = kernel_size - 1

    def build(self, input_shape):
        self.dw_conv = layers.DepthwiseConv1D(
            kernel_size=self.kernel_size,
            padding="valid",
            use_bias=False,
            name="dw_conv"
        )
        self.norm = RMSNorm(self.channels, name="norm")
        self.pw_conv1 = layers.Dense(self.channels * self.expansion, use_bias=True, name="pw_conv1")
        self.pw_conv2 = layers.Dense(self.channels, use_bias=True, name="pw_conv2")
        self.se = SpatialTemporalSE(self.channels, is_causal=self.is_causal, name="se")
        super().build(input_shape)

    def call(self, x, mask: Optional[any] = None):
        # x: (B, L, C)
        if self.is_causal:
            # Causal left padding: pad (pad_size, 0) on axis 1
            padded_x = ops.pad(x, ((0, 0), (self.pad_size, 0), (0, 0)))
        else:
            p_left = self.pad_size // 2
            p_right = self.pad_size - p_left
            padded_x = ops.pad(x, ((0, 0), (p_left, p_right), (0, 0)))

        dw = self.dw_conv(padded_x)
        normed = self.norm(dw)
        pw1 = self.pw_conv1(normed)
        act = ops.gelu(pw1)
        pw2 = self.pw_conv2(act)
        se_out = self.se(pw2, mask=mask)
        return x + se_out
