import os
import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Union, Optional, List, Dict, Tuple, Any

def analyze_single_shard(shard_path: Path):
    """Parses a single PyTorch dataset shard and extracts quality & landmark statistics."""
    shard_name = shard_path.name
    results = []
    
    try:
        # Load shard (supports PyTorch 2.6+ dictionary unpickling)
        records = torch.load(shard_path, map_location="cpu", weights_only=False)
        for rec in records:
            source = rec.get("source", "Unknown")
            quality = float(rec.get("quality", 0.0))
            weight = float(rec.get("sample_weight", 0.0))
            
            breakdown = rec.get("quality_breakdown", {})
            best_hand = float(breakdown.get("best_hand", breakdown.get("best_hand_conf", quality)))
            left_conf = float(breakdown.get("left", breakdown.get("left_hand_conf", 0.0)))
            right_conf = float(breakdown.get("right", breakdown.get("right_hand_conf", 0.0)))
            pose_vis = float(breakdown.get("pose", breakdown.get("pose_vis", 0.0)))
            
            # Inspect features array (T, 60, 9)
            features = rec.get("features", None)
            feat_valid = False
            has_nan = False
            has_inf = False
            zero_ratio = 1.0
            
            if isinstance(features, torch.Tensor):
                features = features.numpy()
                
            if isinstance(features, np.ndarray) and features.size > 0:
                has_nan = bool(np.isnan(features).any())
                has_inf = bool(np.isinf(features).any())
                non_zero = np.abs(features[:, :, :2]) > 1e-4
                zero_ratio = float(1.0 - np.mean(non_zero))
                feat_valid = not has_nan and not has_inf
                
            results.append({
                "source": source,
                "quality": quality,
                "weight": weight,
                "best_hand": best_hand,
                "left_conf": left_conf,
                "right_conf": right_conf,
                "pose_vis": pose_vis,
                "feat_valid": feat_valid,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "zero_ratio": zero_ratio,
            })
    except Exception as e:
        print(f"[!] Warning reading {shard_name}: {e}")
        
    return results

def audit_dataset_shards(shard_dir: Union[str, Path] = None, num_workers: int = 8):
    """
    Scans a directory of payload shards and computes comprehensive hand landmark quality statistics.
    """
    if shard_dir is None:
        candidates = [
            Path("/kaggle/input/notebooks/tranquocbao2012/frakenstein-merger/asl_preprocessed_phase1/train"),
            Path("/kaggle/input/notebooks/tranquocbao2012/frakenstein-merger/asl_preprocessed_phase1"),
            Path("/kaggle/input/asl-preprocessed-phase1/train"),
            Path("/kaggle/working/asl_preprocessed_phase1/train"),
            Path("./asl_preprocessed_phase1/train"),
        ]
        shard_dir = next((c for c in candidates if c.exists()), Path("./output/train"))
    else:
        shard_dir = Path(shard_dir)
        
    if not shard_dir.exists():
        print(f"[!] Shard directory not found: {shard_dir}")
        return

    shard_files = sorted(shard_dir.glob("shard_*.pt")) + sorted(shard_dir.glob("temp_shard_*.pt"))
    if not shard_files:
        shard_files = sorted(shard_dir.rglob("shard_*.pt")) + sorted(shard_dir.rglob("temp_shard_*.pt"))
    if not shard_files:
        shard_files = sorted(shard_dir.rglob("*.pt"))

    if not shard_files:
        print(f"[!] No .pt shard files found under {shard_dir}")
        return

    print("==========================================================================================")
    print(f"               DATASET HAND QUALITY & LANDMARK AUDIT REPORT")
    print("==========================================================================================")
    print(f"[*] Target Directory : {shard_dir.resolve()}")
    print(f"[*] Total Shards     : {len(shard_files)}")
    print(f"[*] Workers          : {num_workers} threads\n")

    dataset_stats = defaultdict(lambda: {
        "count": 0,
        "qualities": [],
        "weights": [],
        "best_hands": [],
        "left_confs": [],
        "right_confs": [],
        "high_quality_hands": 0, # best_hand > 0.50
        "low_quality_hands": 0,  # best_hand < 0.25
        "invalid_features": 0,
        "zero_ratios": [],
    })

    total_records = 0

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(analyze_single_shard, sp) for sp in shard_files]
        for f in futures:
            recs = f.result()
            for r in recs:
                src = r["source"]
                ds = dataset_stats[src]
                ds["count"] += 1
                ds["qualities"].append(r["quality"])
                ds["weights"].append(r["weight"])
                ds["best_hands"].append(r["best_hand"])
                ds["left_confs"].append(r["left_conf"])
                ds["right_confs"].append(r["right_conf"])
                ds["zero_ratios"].append(r["zero_ratio"])
                
                if r["best_hand"] >= 0.50:
                    ds["high_quality_hands"] += 1
                if r["best_hand"] < 0.25:
                    ds["low_quality_hands"] += 1
                if not r["feat_valid"]:
                    ds["invalid_features"] += 1
                
                total_records += 1

    header = f"{'Dataset':<20} | {'Count':<8} | {'Avg Qual':<10} | {'Avg Hand Conf':<14} | {'High Conf (>0.5)':<16} | {'Low Conf (<0.25)':<16} | {'Feat Errors':<11}"
    print(header)
    print("-" * len(header))

    all_qual, all_hand, all_high, all_low, all_err = [], [], 0, 0, 0

    for ds_name in sorted(dataset_stats.keys()):
        st = dataset_stats[ds_name]
        cnt = st["count"]
        avg_q = float(np.mean(st["qualities"])) if cnt > 0 else 0.0
        avg_h = float(np.mean(st["best_hands"])) if cnt > 0 else 0.0
        pct_high = (st["high_quality_hands"] / max(1, cnt)) * 100.0
        pct_low = (st["low_quality_hands"] / max(1, cnt)) * 100.0
        err_cnt = st["invalid_features"]
        
        all_qual.extend(st["qualities"])
        all_hand.extend(st["best_hands"])
        all_high += st["high_quality_hands"]
        all_low += st["low_quality_hands"]
        all_err += err_cnt
        
        print(f"{ds_name:<20} | {cnt:<8} | {avg_q:<10.4f} | {avg_h:<14.4f} | {pct_high:<15.1f}% | {pct_low:<15.1f}% | {err_cnt:<11}")

    print("-" * len(header))
    overall_avg_q = float(np.mean(all_qual)) if all_qual else 0.0
    overall_avg_h = float(np.mean(all_hand)) if all_hand else 0.0
    overall_pct_high = (all_high / max(1, total_records)) * 100.0
    overall_pct_low = (all_low / max(1, total_records)) * 100.0
    print(f"{'OVERALL TOTAL':<20} | {total_records:<8} | {overall_avg_q:<10.4f} | {overall_avg_h:<14.4f} | {overall_pct_high:<15.1f}% | {overall_pct_low:<15.1f}% | {all_err:<11}")
    print("==========================================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Audit hand quality and landmark integrity across dataset shards.")
    parser.add_argument("--shard_dir", type=str, default="./asl_preprocessed_phase1/train", help="Path to shard directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads")
    args = parser.parse_args()
    
    audit_dataset_shards(args.shard_dir, num_workers=args.workers)
