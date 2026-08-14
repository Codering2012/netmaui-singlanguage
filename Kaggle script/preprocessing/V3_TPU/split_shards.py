import os
import sys
import json
import torch
import random
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

def split_preprocessed_shards(
    input_dir: str,
    output_root: str,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    shard_size: int = 5000,
    seed: int = 42,
    num_workers: int = 8
):
    """
    Loads preprocessed PyTorch shards from `input_dir` (e.g. /kaggle/input/datasets/tranquocbao2012/frakenstein-asl),
    performs label-stratified shuffling, and splits records into canonical `train`, `val`, and `test` partitions.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    input_path = Path(input_dir)
    output_path = Path(output_root)
    
    print("======================================================================")
    print("       PREPROCESSED DATASET SPLITTER (TRAIN / VAL / TEST)")
    print("======================================================================")
    print(f"[*] Input Shard Path  : {input_path.resolve()}")
    print(f"[*] Output Root Path  : {output_path.resolve()}")
    print(f"[*] Split Ratios      : Train={1.0 - val_ratio - test_ratio:.2f}, Val={val_ratio:.2f}, Test={test_ratio:.2f}")

    shard_files = sorted(input_path.rglob("shard_*.pt")) + sorted(input_path.rglob("temp_shard_*.pt"))
    if not shard_files:
        shard_files = sorted(input_path.rglob("*.pt"))

    if not shard_files:
        print(f"[!] Error: No .pt shard files found under {input_path}")
        return

    print(f"[*] Discovered {len(shard_files)} PyTorch payload shards. Loading records into memory...")

    # Multi-threaded loading of shards
    def _load_shard(sp):
        try:
            return torch.load(sp, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[!] Warning loading shard {sp.name}: {e}")
            return []

    all_records = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_load_shard, sp) for sp in shard_files]
        for f in futures:
            recs = f.result()
            all_records.extend(recs)

    print(f"[+] Total loaded records: {len(all_records)}")

    # Group records by label for stratified splitting
    label_to_records = defaultdict(list)
    isolated_labels = set()

    for rec in all_records:
        lbl = rec.get("label", "<none>")
        task = rec.get("task", "")
        label_to_records[lbl].append(rec)
        if task in ("isolated_gloss", "static_alphabet", "isolated_number"):
            isolated_labels.add(lbl)

    unique_isolated_labels = sorted(list(isolated_labels))
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_isolated_labels)}
    print(f"[+] Total canonical isolated vocabulary classes: {len(unique_isolated_labels)}")

    train_recs, val_recs, test_recs = [], [], []

    # Perform stratified split per label class
    for lbl, recs in label_to_records.items():
        random.shuffle(recs)
        n = len(recs)
        n_val = max(1 if n >= 10 else 0, int(round(n * val_ratio)))
        n_test = max(1 if n >= 10 else 0, int(round(n * test_ratio)))
        n_train = n - n_val - n_test
        
        if n_train <= 0:
            n_train = n
            n_val, n_test = 0, 0
            
        train_recs.extend(recs[:n_train])
        val_recs.extend(recs[n_train : n_train + n_val])
        test_recs.extend(recs[n_train + n_val :])

    random.shuffle(train_recs)
    random.shuffle(val_recs)
    random.shuffle(test_recs)

    split_map = {
        "train": train_recs,
        "val": val_recs,
        "test": test_recs,
    }

    # Save sharded payloads per split
    for split_name, rec_list in split_map.items():
        if not rec_list:
            continue
        split_dir = output_path / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n[*] Saving split '{split_name}' ({len(rec_list)} records) under {split_dir}...")
        
        num_shards = (len(rec_list) + shard_size - 1) // shard_size
        for s_idx in range(num_shards):
            chunk = rec_list[s_idx * shard_size : (s_idx + 1) * shard_size]
            
            # Annotate records with integer label indices
            for r in chunk:
                r["label_idx"] = label_to_idx.get(r.get("label", "<none>"), 0)
                r["split"] = split_name

            shard_file = split_dir / f"shard_{s_idx:04d}.pt"
            torch.save(chunk, shard_file)
            print(f"  -> Saved {shard_file.name} ({len(chunk)} records)")

        # Save metadata and vocabulary mapping
        metadata = {
            "split": split_name,
            "total_records": len(rec_list),
            "num_shards": num_shards,
            "shard_size": shard_size,
            "label_to_idx": label_to_idx,
        }
        with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        with open(split_dir / f"vocabulary_mapping_{split_name}.json", "w", encoding="utf-8") as f:
            json.dump({"split": split_name, "label_to_idx": label_to_idx}, f, indent=2)

    print("\n======================================================================")
    print("      SUCCESS: DATASET PREPROCESSING SPLIT FINISHED COMPLETELY")
    print("======================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Split preprocessed PyTorch shards into train/val/test partitions.")
    parser.add_argument("--input-dir", type=str, default="/kaggle/input/datasets/tranquocbao2012/frakenstein-asl", help="Input shard directory")
    parser.add_argument("--output-dir", type=str, default="/kaggle/working/asl_preprocessed_phase1", help="Output split directory")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation set ratio (default 0.10)")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test set ratio (default 0.10)")
    parser.add_argument("--shard-size", type=int, default=5000, help="Shard size (default 5000 records)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads")
    args = parser.parse_args()

    split_preprocessed_shards(
        input_dir=args.input_dir,
        output_root=args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        shard_size=args.shard_size,
        seed=args.seed,
        num_workers=args.workers
    )
