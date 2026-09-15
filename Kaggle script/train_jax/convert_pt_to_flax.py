"""
PyTorch to Flax Checkpoint Converter for Continuous ASL Foundation Architecture
Transfers pre-trained PyTorch weights (e.g. asl_llm_200, asl_model_epoch_1.pt) into Flax Linen parameters.
"""

from typing import Any, Dict
import numpy as np
import torch
import jax.numpy as jnp
from flax import serialization, traverse_util


def convert_torch_state_dict_to_flax(torch_state_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Translates a PyTorch state_dict into a Flax Linen nested parameter tree."""
    flat_flax_dict = {}

    for pt_key, tensor in torch_state_dict.items():
        val = tensor.detach().cpu().numpy()

        # Clean prefix names if wrapped in module or raw_model
        key = pt_key
        for prefix in ["module.", "model.", "raw_m."]:
            if key.startswith(prefix):
                key = key[len(prefix):]

        # 1. Linear weight transposition: PyTorch (out, in) -> Flax (in, out)
        if key.endswith(".weight") and val.ndim == 2:
            val = val.T
            flax_key = tuple(key[:-7].split(".") + ["kernel"])
            flat_flax_dict[flax_key] = jnp.array(val)

        # 2. Linear bias: PyTorch bias -> Flax bias
        elif key.endswith(".bias"):
            flax_key = tuple(key[:-5].split(".") + ["bias"])
            flat_flax_dict[flax_key] = jnp.array(val)

        # 3. RMSNorm scale parameter: PyTorch weight -> Flax scale
        elif "norm" in key and key.endswith(".weight") and val.ndim == 1:
            flax_key = tuple(key[:-7].split(".") + ["scale"])
            flat_flax_dict[flax_key] = jnp.array(val)

        # 4. 1D Conv weight transposition: PyTorch (out, in, k) -> Flax (k, in, out)
        elif key.endswith(".weight") and val.ndim == 3:
            val = np.transpose(val, (2, 1, 0))
            flax_key = tuple(key[:-7].split(".") + ["kernel"])
            flat_flax_dict[flax_key] = jnp.array(val)

        # 5. LayerScale gamma parameters
        elif "gamma" in key:
            flax_key = tuple(key.split("."))
            flat_flax_dict[flax_key] = jnp.array(val)

        # 6. Embeddings: PyTorch (vocab, dim) -> Flax (vocab, dim)
        elif "embedding" in key and key.endswith(".weight"):
            flax_key = tuple(key[:-7].split(".") + ["embedding"])
            flat_flax_dict[flax_key] = jnp.array(val)

        else:
            flax_key = tuple(key.split("."))
            flat_flax_dict[flax_key] = jnp.array(val)

    # Reconstruct nested dictionary tree
    nested_flax_params = traverse_util.unflatten_dict(flat_flax_dict)
    return nested_flax_params


def save_flax_checkpoint(flax_params: Dict[str, Any], output_path: str):
    """Serializes Flax parameters to a binary msgpack file."""
    bytes_data = serialization.to_bytes(flax_params)
    with open(output_path, "wb") as f:
        f.write(bytes_data)
    print(f"[+] Successfully saved Flax checkpoint ({len(bytes_data)} bytes) to: {output_path}")


def load_flax_checkpoint(input_path: str, target_template: Dict[str, Any]) -> Dict[str, Any]:
    """Deserializes Flax parameters from a binary msgpack file."""
    with open(input_path, "rb") as f:
        bytes_data = f.read()
    return serialization.from_bytes(target_template, bytes_data)
