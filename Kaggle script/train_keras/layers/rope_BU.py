"""
Rotary Position Embedding (RoPE) in Keras 3 (JAX Backend)
"""

import keras
from keras import ops


def get_rotary_frequencies(max_seq_len: int, dim: int, theta: float = 10000.0):
    """
    Precompute cos and sin rotary embedding tables.
    dim: total head dimension (must be even).
    cos, sin output shape: (max_seq_len, dim // 2)
    """
    half_dim = dim // 2
    inv_freq = 1.0 / (theta ** (ops.arange(0, half_dim, dtype="float32") / half_dim))
    t = ops.arange(max_seq_len, dtype="float32")
    freqs = ops.outer(t, inv_freq)
    cos = ops.cos(freqs)
    sin = ops.sin(freqs)
    return cos, sin


def apply_rope(x, cos, sin):
    """
    Apply Rotary Position Embedding to input tensor x.
    x: (B, H, L, D) or (B, L, D)
    cos: (L, D // 2)
    sin: (L, D // 2)
    """
    d = ops.shape(x)[-1]
    half = d // 2
    x1 = x[..., :half]
    x2 = x[..., half:]

    seq_len = ops.shape(x)[2] if len(ops.shape(x)) == 4 else ops.shape(x)[1]
    c = cos[:seq_len]
    s = sin[:seq_len]

    if len(ops.shape(x)) == 4:
        c = ops.reshape(c, (1, 1, seq_len, half))
        s = ops.reshape(s, (1, 1, seq_len, half))
    elif len(ops.shape(x)) == 3:
        c = ops.reshape(c, (1, seq_len, half))
        s = ops.reshape(s, (1, seq_len, half))

    c = ops.cast(c, ops.dtype(x))
    s = ops.cast(s, ops.dtype(x))

    # [x1 * cos - x2 * sin, x2 * cos + x1 * sin]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return ops.concatenate([out1, out2], axis=-1)
