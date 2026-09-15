"""
Continuous ASL Foundation Model in Keras 3 with JAX Backend
Optimized for Google Cloud TPU v5e Systolic Architecture
"""

import os
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "jax"
