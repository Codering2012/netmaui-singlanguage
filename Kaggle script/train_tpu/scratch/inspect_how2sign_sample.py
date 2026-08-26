import os
import sys
from pathlib import Path
import json
import torch
import numpy as np

print("=" * 70, flush=True)
print("     HOW2SIGN DATASET SAMPLE INSPECTION & PIPELINE TRACE     ", flush=True)
print("=" * 70, flush=True)

base_dir = Path("E:/datasets/asl_dataset/asl_preprocessed_phase1")
train_shards = sorted(list((base_dir / "train").glob("*.pt")))

found_samples = []
for pf in train_shards:
    try:
        data = torch.load(pf, map_location="cpu", weights_only=False)
        for item in data:
            if isinstance(item, dict) and "how2sign" in str(item.get("source", "")).lower():
                found_samples.append(item)
                if len(found_samples) >= 3:
                    break
        if len(found_samples) >= 3:
            break
    except Exception:
        pass

if not found_samples:
    print("Searching other shards for any sentence-level continuous signing datasets...")
    for pf in train_shards:
        try:
            data = torch.load(pf, map_location="cpu", weights_only=False)
            for item in data:
                if isinstance(item, dict) and item.get("task") in ["continuous", "sentence", "translation", "cslr"]:
                    found_samples.append(item)
                    if len(found_samples) >= 3:
                        break
            if len(found_samples) >= 3:
                break
        except Exception:
            pass

if not found_samples:
    # Look at any sample with longest sequence length
    for pf in train_shards[:5]:
        try:
            data = torch.load(pf, map_location="cpu", weights_only=False)
            for item in data:
                if isinstance(item, dict) and "features" in item and hasattr(item["features"], "shape") and item["features"].shape[0] > 30:
                    found_samples.append(item)
                    if len(found_samples) >= 3:
                        break
            if len(found_samples) >= 3:
                break
        except Exception:
            pass

for idx, sample in enumerate(found_samples):
    print(f"\n--- SAMPLE #{idx+1} ---", flush=True)
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k:<20}: Tensor shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k:<20}: {v}")
        else:
            print(f"  {k:<20}: {v}")

    # Inspect features & duration
    if "features" in sample and hasattr(sample["features"], "shape"):
        n_frames = sample["features"].shape[0]
        fps = 30.0
        dur_sec = n_frames / fps
        print(f"\n  [Video Duration] {n_frames} frames @ 30 FPS = {dur_sec:.2f} seconds of signing")
