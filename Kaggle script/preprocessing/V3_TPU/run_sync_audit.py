import os
import sys
import torch
import glob
import numpy as np
from collections import defaultdict, Counter

def run_sync_audit(target_dir: str):
    print(f"Scanning directory: {target_dir}")
    shard_files = sorted(glob.glob(os.path.join(target_dir, "**", "shard_*.pt"), recursive=True)) + \
                  sorted(glob.glob(os.path.join(target_dir, "**", "temp_shard_*.pt"), recursive=True))
    
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.pt"), recursive=True))
        
    print(f"Found {len(shard_files)} shard files.")
    
    dataset_stats = defaultdict(lambda: {
        "count": 0,
        "qualities": [],
        "weights": [],
        "best_hands": [],
        "high_quality_hands": 0,
        "low_quality_hands": 0,
        "invalid_features": 0,
    })

    total_records = 0

    for idx, sp in enumerate(shard_files):
        try:
            records = torch.load(sp, map_location="cpu", weights_only=False)
            for rec in records:
                source = rec.get("source", "Unknown")
                quality = float(rec.get("quality", 0.0))
                weight = float(rec.get("sample_weight", 0.0))
                breakdown = rec.get("quality_breakdown", {})
                best_hand = float(breakdown.get("best_hand", breakdown.get("best_hand_conf", quality)))
                
                features = rec.get("features", None)
                feat_valid = True
                if isinstance(features, torch.Tensor):
                    features = features.numpy()
                if isinstance(features, np.ndarray) and features.size > 0:
                    if np.isnan(features).any() or np.isinf(features).any():
                        feat_valid = False
                else:
                    feat_valid = False

                ds = dataset_stats[source]
                ds["count"] += 1
                ds["qualities"].append(quality)
                ds["weights"].append(weight)
                ds["best_hands"].append(best_hand)
                if best_hand >= 0.50:
                    ds["high_quality_hands"] += 1
                if best_hand < 0.25:
                    ds["low_quality_hands"] += 1
                if not feat_valid:
                    ds["invalid_features"] += 1
                total_records += 1
        except Exception as e:
            print(f"[!] Error loading {os.path.basename(sp)}: {e}")

    out_lines = []
    out_lines.append("==========================================================================================")
    out_lines.append("               DATASET HAND QUALITY & LANDMARK AUDIT REPORT")
    out_lines.append("==========================================================================================")
    out_lines.append(f"Target Directory : {target_dir}")
    out_lines.append(f"Total Shards     : {len(shard_files)}")
    out_lines.append(f"Total Records    : {total_records}\n")

    header = f"{'Dataset':<20} | {'Count':<8} | {'Avg Qual':<10} | {'Avg Hand Conf':<14} | {'High Conf (>0.5)':<16} | {'Low Conf (<0.25)':<16} | {'Feat Errors':<11}"
    out_lines.append(header)
    out_lines.append("-" * len(header))

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
        
        out_lines.append(f"{ds_name:<20} | {cnt:<8} | {avg_q:<10.4f} | {avg_h:<14.4f} | {pct_high:<15.1f}% | {pct_low:<15.1f}% | {err_cnt:<11}")

    out_lines.append("-" * len(header))
    overall_avg_q = float(np.mean(all_qual)) if all_qual else 0.0
    overall_avg_h = float(np.mean(all_hand)) if all_hand else 0.0
    overall_pct_high = (all_high / max(1, total_records)) * 100.0
    overall_pct_low = (all_low / max(1, total_records)) * 100.0
    out_lines.append(f"{'OVERALL TOTAL':<20} | {total_records:<8} | {overall_avg_q:<10.4f} | {overall_avg_h:<14.4f} | {overall_pct_high:<15.1f}% | {overall_pct_low:<15.1f}% | {all_err:<11}")
    out_lines.append("==========================================================================================")
    
    report_text = "\n".join(out_lines)
    print(report_text)
    with open("local_audit_result.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

if __name__ == "__main__":
    target = r"E:\datasets\results\asl_preprocessed_phase1"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    run_sync_audit(target)
