#!/usr/bin/env python3
"""
Sidecar Dataset Manifest Builder
Pre-indexes sharded ASL dataset directories into a single lightweight JSON/Parquet sidecar file.
Eliminates runtime dataset scanning overhead and host RAM spikes on TPU/GPU clusters.
"""

import os
import sys
import glob
import json
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
import torch
import numpy as np


def index_dataset_dir(dataset_dir: Path, split: str = "train", max_len: int = 256) -> Path:
    target_dir = dataset_dir / split if (dataset_dir / split).exists() else dataset_dir
    shard_files = sorted(list(target_dir.glob("shard_*.pt")))
    if not shard_files:
        shard_files = sorted(list(target_dir.glob("*.pt")))
    if not shard_files:
        print(f"[ERROR] No .pt shards found in {target_dir}")
        return None

    print(f"[{split.upper()}] Found {len(shard_files)} shards in {target_dir}. Building manifest...")

    # Load master vocabulary mapping if available
    vocab_map = {}
    vocab_candidates = [
        target_dir / "vocab_map.json",
        target_dir / "vocabulary_mapping_global.json",
        target_dir.parent / "vocabulary_mapping_global.json",
        target_dir / f"vocabulary_mapping_{split}.json",
    ]
    for vc in vocab_candidates:
        if vc.exists():
            try:
                with open(vc, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    vocab_map = data.get("label_to_idx", data)
                    if vocab_map:
                        break
            except Exception:
                pass

    valid_label_set = set(int(v) for v in vocab_map.values()) if vocab_map else set()
    records = []
    class_counts = defaultdict(int)

    for shard_path in shard_files:
        try:
            shard_data = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
            items = shard_data.items() if isinstance(shard_data, dict) else enumerate(shard_data)
            
            for key_or_idx, rec in items:
                if not isinstance(rec, dict):
                    continue

                f_key = key_or_idx if isinstance(shard_data, dict) else None
                item_idx = key_or_idx if not isinstance(shard_data, dict) else None

                task_str = str(rec.get("task", "unknown"))
                source_str = str(rec.get("source", "unknown"))
                raw_label_str = str(rec.get("label", "")).strip().lower()
                raw_label_idx = rec.get("label_idx", -1)

                token_ids = []
                lbl_clean = -1

                if "gloss_seq" in rec:
                    gs = rec["gloss_seq"]
                    token_ids = gs.tolist() if isinstance(gs, torch.Tensor) else list(gs)
                
                if not token_ids and raw_label_idx is not None and int(raw_label_idx) >= 0:
                    lbl_clean = int(raw_label_idx)

                if token_ids:
                    lbl_clean = int(token_ids[0]) if token_ids else -1
                    for t in token_ids:
                        if t >= 0:
                            class_counts[int(t)] += 1
                elif lbl_clean >= 0:
                    class_counts[lbl_clean] += 1

                records.append({
                    "shard_path": str(shard_path.name),
                    "feature_key": f_key,
                    "item_idx": item_idx,
                    "label_idx": lbl_clean,
                    "raw_label": raw_label_str or str(lbl_clean),
                    "quality": float(rec.get("quality", rec.get("sample_weight", 1.0))),
                    "token_ids": token_ids,
                    "task": task_str,
                    "source": source_str,
                })
            del shard_data
        except Exception as e:
            print(f"[WARNING] Skipping problematic shard {shard_path.name}: {e}")

    # Compute global fingerprint
    shard_info = str([(p.name, p.stat().st_size) for p in shard_files]).encode()
    fingerprint = hashlib.md5(shard_info).hexdigest()[:12]
    
    manifest_path = target_dir / f"dataset_manifest_{split}_{fingerprint}.json"
    manifest_data = {
        "split": split,
        "total_records": len(records),
        "total_shards": len(shard_files),
        "fingerprint": fingerprint,
        "class_counts": dict(class_counts),
        "records": records,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    print(f"[{split.upper()}] Successfully saved manifest to {manifest_path} ({len(records)} records).")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Pre-index ASL dataset shards into a sidecar manifest.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to preprocessed dataset directory")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"[ERROR] Directory does not exist: {data_path}")
        sys.exit(1)

    for split in ["train", "val"]:
        index_dataset_dir(data_path, split=split)


if __name__ == "__main__":
    main()
