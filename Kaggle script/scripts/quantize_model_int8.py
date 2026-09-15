#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — DYNAMIC INT8 QUANTIZATION & EDGE COMPRESSION TOOL
================================================================================
Applies PyTorch Dynamic INT8 Quantization to ASL Foundation Models:
  - Quantizes Linear layers to 8-bit integer weights (qint8)
  - Reduces memory footprint by ~3.5x - 4.0x
  - Delivers fast low-latency CPU / edge inference
================================================================================
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Union

import torch
import torch.nn as nn


def quantize_asl_model(model: nn.Module) -> nn.Module:
    """
    Applies dynamic INT8 quantization across all Linear layers in the model.
    """
    model.eval()
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model


def measure_model_size_mb(model: nn.Module) -> float:
    """
    Calculates in-memory parameter and buffer size in Megabytes.
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description="Dynamic INT8 quantization for ASL Foundation Models.")
    parser.add_argument("--src", type=str, required=True, help="Path to input checkpoint (.pt)")
    parser.add_argument("--dst", type=str, required=True, help="Path to output quantized model (.pt)")
    parser.add_argument("--arch", type=str, default="v2", choices=["v1", "v2"], help="Model architecture")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "train_tpu"))

    if args.arch == "v1":
        from train_all_in_one_tpu import ASLFoundationModel
    else:
        from train_all_in_one_tpu_v2 import ASLFoundationModel

    print(f"[*] Loading model for architecture: '{args.arch}'...")
    model = ASLFoundationModel()
    ckpt = torch.load(args.src, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)

    fp32_size = measure_model_size_mb(model)
    print(f"[*] Original FP32 Model Size: {fp32_size:.2f} MB")

    print("[*] Applying Dynamic INT8 Quantization...")
    quant_model = quantize_asl_model(model)
    int8_size = measure_model_size_mb(quant_model)
    print(f"[*] Quantized INT8 Model Size: {int8_size:.2f} MB ({fp32_size / max(1e-3, int8_size):.2f}x compression)")

    torch.save(quant_model, args.dst)
    print(f"[+] Saved quantized model to: {args.dst}")


if __name__ == "__main__":
    main()
