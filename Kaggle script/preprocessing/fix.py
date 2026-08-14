#!/usr/bin/env python3
"""
GLOBAL VOCABULARY RESCUE PATCH
Scans train, val, and test shards to build a unified global vocabulary mapping,
preventing validation/test OOV (Out-Of-Vocabulary) crashes.
"""

import os
import json
import torch
from pathlib import Path
from tqdm import tqdm


def rescue_global_vocabulary(data_dir: str):
    data_path = Path(data_dir)
    splits = ["train", "val", "test"]
    unique_labels = set()

    for split in splits:
        split_dir = data_path / split
        if not split_dir.exists():
            print(f"[*] Skipping {split} split (directory not found).")
            continue

        shard_files = sorted(split_dir.glob("shard_*.pt"))
        print(f"[*] Scanning {len(shard_files)} shards in '{split}' split...")

        for shard_path in tqdm(
            shard_files, desc=f"Parsing {split.capitalize()} Shards"
        ):
            try:
                records = torch.load(shard_path, map_location="cpu", weights_only=False)
                for r in records:
                    task = r.get("task", "")
                    raw_label = str(r.get("label", "")).strip().upper()

                    if not raw_label or raw_label == "<NONE>":
                        continue

                    if task in ("isolated_gloss", "static_alphabet", "isolated_number"):
                        unique_labels.add(raw_label)
                    elif task == "sentence_level":
                        for w in raw_label.split():
                            unique_labels.add(w)
                    elif task == "fingerspelling_sequence":
                        for c in list(raw_label.replace(" ", "")):
                            unique_labels.add(c)
            except Exception as e:
                print(f"[!] Failed to read {shard_path.name}: {e}")

    # Sort to ensure deterministic global ID mapping
    sorted_labels = sorted(list(unique_labels))
    label_to_idx_final = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

    # Save a master global vocabulary file alongside the train mapping
    global_vocab_path = data_path / "vocabulary_mapping_global.json"
    output_data = {
        "total_classes": len(label_to_idx_final),
        "label_to_idx": label_to_idx_final,
    }

    with open(global_vocab_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(
        f"\n[+] GLOBAL RESCUE COMPLETE: Extracted {len(label_to_idx_final)} unique tokens across all splits."
    )
    print(f"[+] Saved global vocabulary at: {global_vocab_path}")


if __name__ == "__main__":
    LOCAL_DATA_DIR = r"E:\datasets\results\asl_preprocessed_phase1"
    rescue_global_vocabulary(LOCAL_DATA_DIR)
