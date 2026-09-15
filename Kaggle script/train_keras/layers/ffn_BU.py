"""
SwiGLU Feed-Forward Network in Keras 3 (JAX Backend)
Tiled for TPU v5e Systolic Matrix Multiply Units (128x128)
"""

import keras
from keras import layers, ops


class SwiGLUFFN(layers.Layer):
    """Fused SwiGLU Feed-Forward Network tiled for systolic MXUs."""
    def __init__(self, d_model: int, dim_feedforward: int, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        # Align hidden dimension to 128 for hardware tile efficiency
        self.hidden = ((int(dim_feedforward * 2 / 3) + 127) // 128) * 128

    def build(self, input_shape):
        self.w_gate_up = layers.Dense(2 * self.hidden, use_bias=False, name="w_gate_up")
        self.w_down = layers.Dense(self.d_model, use_bias=False, name="w_down")
        super().build(input_shape)

    def call(self, x):
        gate_up = self.w_gate_up(x)
        gate, up = ops.split(gate_up, 2, axis=-1)
        return self.w_down(ops.silu(gate) * up)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dim_feedforward": self.dim_feedforward,
        })
        return config
