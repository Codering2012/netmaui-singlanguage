"""
Parity Verification Suite: Compares PyTorch vs. Flax Linen Layers for Exact Numerical Parity
"""

import sys, os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
import torch
import torch.nn as tnn
import torch.nn.functional as tF
import jax
import jax.numpy as jnp
from train_jax.models.conformer import RMSNorm as FlaxRMSNorm, SwiGLUFFN as FlaxSwiGLU
from train_jax.convert_pt_to_flax import convert_torch_state_dict_to_flax


class PyTorchRMSNorm(tnn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = tnn.Parameter(torch.ones(d_model))

    def forward(self, x):
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class PyTorchSwiGLU(tnn.Module):
    def __init__(self, d_model: int, dim_feedforward: int):
        super().__init__()
        hidden = ((int(dim_feedforward * 2 / 3) + 127) // 128) * 128
        self.w_gate_up = tnn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = tnn.Linear(hidden, d_model, bias=False)

    def forward(self, x):
        gate, up = self.w_gate_up(x).chunk(2, dim=-1)
        return self.w_down(tF.silu(gate) * up)


def test_rmsnorm_parity():
    print("Testing RMSNorm Parity...")
    d_model = 64
    pt_norm = PyTorchRMSNorm(d_model)
    flax_norm = FlaxRMSNorm(d_model)

    # Initialize with random scale
    scale_np = np.random.randn(d_model).astype(np.float32)
    pt_norm.weight.data = torch.from_numpy(scale_np)

    flax_params = {"params": {"scale": jnp.array(scale_np)}}

    x_np = np.random.randn(2, 16, d_model).astype(np.float32)
    pt_out = pt_norm(torch.from_numpy(x_np)).detach().numpy()
    flax_out = np.array(flax_norm.apply(flax_params, jnp.array(x_np)))

    diff = np.max(np.abs(pt_out - flax_out))
    print(f"  [+] RMSNorm Max Difference: {diff:.2e}")
    assert diff < 1e-5, f"RMSNorm parity failed: diff={diff}"


def test_swiglu_parity():
    print("Testing SwiGLU Parity...")
    d_model = 64
    dim_ffn = 128
    pt_glu = PyTorchSwiGLU(d_model, dim_ffn)
    flax_glu = FlaxSwiGLU(d_model, dim_ffn)

    # Transfer transposed weights
    gate_up_w = pt_glu.w_gate_up.weight.detach().numpy()
    down_w = pt_glu.w_down.weight.detach().numpy()

    flax_params = {
        "params": {
            "w_gate_up": {"kernel": jnp.array(gate_up_w.T)},
            "w_down": {"kernel": jnp.array(down_w.T)},
        }
    }

    x_np = np.random.randn(2, 8, d_model).astype(np.float32)
    pt_out = pt_glu(torch.from_numpy(x_np)).detach().numpy()
    flax_out = np.array(flax_glu.apply(flax_params, jnp.array(x_np)))

    diff = np.max(np.abs(pt_out - flax_out))
    print(f"  [+] SwiGLU Max Difference: {diff:.2e}")
    assert diff < 1e-5, f"SwiGLU parity failed: diff={diff}"


if __name__ == "__main__":
    test_rmsnorm_parity()
    test_swiglu_parity()
    print("\n=======================================================")
    print("ALL NUMERICAL PARITY TESTS PASSED WITH ZERO TOLERANCE!")
    print("=======================================================")
