"""
PyTorch to Keras 3 Checkpoint Converter for Continuous ASL Foundation Architecture
Transfers pre-trained PyTorch weights into Keras 3 Model parameters.
"""

from typing import Dict, Any
import re
import numpy as np
import torch
import keras


def _clean_name(name: str) -> str:
    """Normalize names for robust cross-framework matching."""
    s = name.lower()
    # Strip auto-generated numbers at the end (e.g. 'rms_norm_1' -> 'rms_norm')
    s = re.sub(r'_\d+$', '', s)
    s = re.sub(r'\d+$', '', s)
    for drop in ["rms_", "model.", "module.", "raw_m.", "_"]:
        s = s.replace(drop, "")
    return s


def load_torch_weights_into_keras(keras_model: keras.Model, torch_state_dict: Dict[str, torch.Tensor]):
    """
    Directly maps PyTorch state_dict tensors to Keras 3 model variables by clean name and shape alignment.
    """
    cleaned_dict = {}
    for k, v in torch_state_dict.items():
        key = k
        for pfx in ["module.", "model.", "raw_m."]:
            if key.startswith(pfx):
                key = key[len(pfx):]
        cleaned_dict[key] = v.detach().cpu().numpy()

    assigned_count = 0
    skipped_keys = []

    # Map variables by layer traversal
    for var in keras_model.weights:
        v_path = var.path.replace("/", ".")
        v_shape = tuple(var.shape)
        v_parts = [_clean_name(p) for p in v_path.split(".")]
        matched = False

        # Try to find corresponding key in PyTorch dict
        for pt_key, pt_arr in cleaned_dict.items():
            pt_parts = [_clean_name(p) for p in pt_key.split(".")]

            overlap = set(v_parts) & set(pt_parts)
            meaningful_overlap = {o for o in overlap if o not in ["kernel", "weight", "scale", "bias", ""]}
            if not meaningful_overlap:
                continue

            # 1. Linear weight transpose: (out, in) -> (in, out)
            if pt_arr.ndim == 2 and v_shape == (pt_arr.shape[1], pt_arr.shape[0]):
                var.assign(pt_arr.T)
                assigned_count += 1
                matched = True
                break

            # 2. 1D Conv transpose: (out, in, k) -> (k, in, out)
            elif pt_arr.ndim == 3 and pt_arr.shape[1] == 1 and v_shape == (pt_arr.shape[2], pt_arr.shape[0], 1):
                var.assign(np.transpose(pt_arr, (2, 0, 1)))
                assigned_count += 1
                matched = True
                break

            # 3. Exact shape match (biases, norms, embeddings)
            elif v_shape == tuple(pt_arr.shape):
                var.assign(pt_arr)
                assigned_count += 1
                matched = True
                break

        if not matched:
            skipped_keys.append((var.path, var.shape))

    print(f"[+] Assigned {assigned_count}/{len(keras_model.weights)} Keras 3 variables from PyTorch checkpoint.")
    if skipped_keys:
        print(f"[*] Unmatched variables ({len(skipped_keys)}): {skipped_keys[:5]}")
    return assigned_count
