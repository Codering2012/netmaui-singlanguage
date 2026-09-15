"""
Temporal Strided Pooling Layer for ASL Foundation Architecture in Keras 3
Halves temporal token length midway through encoder to accelerate computation by 2x.
"""

from typing import Optional, Tuple, Dict, Any
import keras
from keras import layers, ops


class TemporalStridedPool(layers.Layer):
    """
    Strided 1D Temporal Pooling Layer.
    Physically downsamples sequence length from T -> ceil(T / 2).
    """
    def __init__(self, is_causal: bool = False, pool_size: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.is_causal = is_causal
        self.pool_size = pool_size
        self.supports_masking = True

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, mask: Optional[any] = None):
        # x: (B, T, D)
        t = ops.shape(x)[1]
        pad_len = t % self.pool_size
        if pad_len > 0:
            if self.is_causal:
                x = ops.concatenate([x[:, :1, :], x], axis=1)
                if mask is not None:
                    mask = ops.concatenate([mask[:, :1], mask], axis=1)
            else:
                x = ops.concatenate([x, x[:, -1:, :]], axis=1)
                if mask is not None:
                    mask = ops.concatenate([mask, mask[:, -1:]], axis=1)

        # Average pool adjacent tokens: (x[0::2] + x[1::2]) * 0.5
        x_even = x[:, 0::2, :]
        x_odd = x[:, 1::2, :]
        out = (x_even + x_odd) * 0.5

        if mask is not None:
            mask_even = mask[:, 0::2]
            mask_odd = mask[:, 1::2]
            out_mask = ops.logical_or(mask_even, mask_odd)
            return out, out_mask

        return out
