#!/usr/bin/env python3
"""
ASL Dataset Diagnostic & Health Inspection Script
Target Directory: E:\datasets\asl_preprocessed_phase1
"""

import os
import sys
import json
import glob
import torch
import numpy as np
from pathlib import Path

DATASET_DIR = Path(r"E:\datasets\asl_preprocessed_phase1")

def inspect_dataset():
    print("=" * 80)
    print(f"INSPECTING ASL DATASET AT: {DATASET_DIR}")
    print("=" * 80)

    if not DATASET_DIR.exists():
        print(f"❌ ERROR: Path '{DATASET_DIR}' does not exist!")
        return

    # 1. Output Mapping / Vocabulary Inspection
    out_map_file = DATASET_DIR / "output_mapping.json"
    print("\n[1] VOCABULARY & OUTPUT MAPPING")
    if out_map_file.exists():
        try:
            with open(out_map_file, "r", encoding="utf-8") as f:
                out_map = json.load(f)
            print(f"  ✓ Found 'output_mapping.json' with {len(out_map)} entries.")
            sample_keys = list(out_map.items())[:3]
            print(f"  ✓ Sample Mappings: {sample_keys}")
        except Exception as e:
            print(f"  ⚠️ Failed to read 'output_mapping.json': {e}")
    else:
        print("  ⚠️ 'output_mapping.json' not found at root.")

    # 2. Quality Logs Inspection
    q_dir = DATASET_DIR / "quality_logs"
    print("\n[2] QUALITY LOGS RECAP")
    if q_dir.exists():
        completed_files = list(q_dir.glob("completed_*.jsonl"))
        discarded_files = list(q_dir.glob("discarded_*.jsonl"))
        print(f"  ✓ Completed Log Files : {len(completed_files)}")
        print(f"  ✓ Discarded Log Files : {len(discarded_files)}")
        
        total_discarded = 0
        for df in discarded_files:
            try:
                with open(df, "r", encoding="utf-8") as f:
                    total_discarded += sum(1 for _ in f)
            except Exception: pass
        print(f"  ✓ Total Discarded Samples across logs: {total_discarded}")
    else:
        print("  ⚠️ No 'quality_logs' directory found.")

    # 3. Split Inspection (train, val, test)
    splits = ["train", "val", "test"]
    print("\n[3] SPLIT STRUCTURE & SHARD ANALYSIS")
    
    for split in splits:
        s_dir = DATASET_DIR / split
        if not s_dir.exists():
            print(f"  ❌ Split folder '{split}' missing!")
            continue

        shards = sorted(list(s_dir.glob("shard_*.pt")))
        meta_file = s_dir / "metadata.json"
        
        print(f"\n  📁 Split: '{split.upper()}' ({len(shards)} shards found)")
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                num_recs = meta.get("num_records", meta.get("total_samples", "N/A"))
                print(f"     └ Metadata summary: total_records = {num_recs}")
            except Exception: pass

        if not shards:
            continue

        # Inspect the very first shard in detail
        sample_shard = shards[0]
        try:
            data = torch.load(sample_shard, map_location="cpu", weights_only=False)
            data_type = type(data).__name__
            num_items = len(data) if hasattr(data, "__len__") else "Unknown"
            print(f"     └ Inspecting Sample Shard '{sample_shard.name}':")
            print(f"        • Data Structure Type : {data_type}")
            print(f"        • Records in Shard   : {num_items}")

            # Inspect first record
            first_key, first_rec = None, None
            if isinstance(data, dict):
                first_key = list(data.keys())[0]
                first_rec = data[first_key]
            elif isinstance(data, (list, tuple)):
                first_key = 0
                first_rec = data[0]

            if first_rec is not None:
                print(f"        • Sample Record Key  : {first_key}")
                if isinstance(first_rec, dict):
                    print(f"        • Record Keys        : {list(first_rec.keys())}")
                    feat = first_rec.get("features", first_rec.get("feature_array", None))
                    lbl = first_rec.get("label", first_rec.get("label_idx", "N/A"))
                    qual = first_rec.get("quality", first_rec.get("quality_score", 1.0))
                    print(f"        • Sample Label       : '{lbl}' | Quality: {qual}")
                else:
                    feat = first_rec

                # Feature tensor validation
                if feat is not None:
                    if isinstance(feat, torch.Tensor):
                        feat_np = feat.numpy()
                    elif isinstance(feat, np.ndarray):
                        feat_np = feat
                    else:
                        feat_np = np.array(feat)

                    print(f"        • Feature Array Shape: {feat_np.shape}")
                    print(f"        • Feature Data Type  : {feat_np.dtype}")
                    
                    has_nan = np.isnan(feat_np).any()
                    has_inf = np.isinf(feat_np).any()
                    min_val, max_val = feat_np.min(), feat_np.max()
                    print(f"        • NaNs / Infs Present: {has_nan} / {has_inf}")
                    print(f"        • Min / Max Values   : [{min_val:.4f}, {max_val:.4f}]")
                    
                    # Verify 3D WholeBody landmark channels
                    if feat_np.ndim == 3:
                        T, K, C = feat_np.shape
                        print(f"        • Temporal Frames (T): {T}")
                        print(f"        • Keypoints (K)      : {K} (Expected: 60)")
                        print(f"        • Feature Channels(C): {C} (Expected: 3, 6, or 9)")
                    elif feat_np.ndim == 2:
                        T, D = feat_np.shape
                        print(f"        • Flattened Shape    : T={T}, D={D} (Expected D=540 for 60x9)")

        except Exception as e:
            print(f"     ❌ Failed to parse sample shard '{sample_shard.name}': {e}")

    print("\n" + "=" * 80)
    print("✅ DIAGNOSTIC COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    inspect_dataset()
