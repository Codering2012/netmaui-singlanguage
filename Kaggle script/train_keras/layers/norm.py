"""
Fused RMSNorm Layer in Keras 3 (JAX Backend)
"""

import keras
from keras import layers, ops


class RMSNorm(layers.Layer):
    """Fused Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.eps = eps
        self.supports_masking = True


    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.dim,),
            initializer="ones",
            trainable=True,
            name="scale"
        )
        super().build(input_shape)

    def call(self, x):
        in_dtype = ops.dtype(x)
        x_f32 = ops.cast(x, "float32")
        variance = ops.mean(ops.square(x_f32), axis=-1, keepdims=True)
        normed = (x_f32 * ops.rsqrt(variance + self.eps)) * self.scale
        return ops.cast(normed, in_dtype)

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "eps": self.eps,
        })
        return config
