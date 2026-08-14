# %%writefile check_rejection_rate.py
#!/usr/bin/env python3
"""
Quality Log Analysis Script
Checks quality logs (summary JSONs and discarded JSONL files) and outputs:
- Total samples processed, kept, and discarded
- Average rejection rate
- Breakdown of rejection reasons/messages (the message being below_threshold, empty_sequence, etc.)
- Average quality of discarded samples
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze sign language pipeline quality logs.")
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Path to the quality logs directory (defaults to auto-detecting kaggle directories or current workspace)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["all", "train", "val", "test"],
        help="Analysis split"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output the results as JSON instead of formatted text"
    )
    return parser.parse_args()

def detect_log_dir(provided_dir=None):
    if provided_dir:
        path = Path(provided_dir)
        if path.exists() and path.is_dir():
            return path
        print(f"Error: Provided directory {provided_dir} does not exist.", file=sys.stderr)
        sys.exit(1)

    # Standard candidate paths
    candidates = [
        Path("quality_logs"),
        Path("/kaggle/working/asl_preprocessed_phase1/quality_logs"),
        Path("C:/kaggle/working/asl_preprocessed_phase1/quality_logs"),
        Path("../asl_preprocessed_phase1/quality_logs"),
    ]
    
    for cand in candidates:
        if cand.exists() and cand.is_dir():
            return cand
            
    # Fallback/last-resort search
    for root, dirs, files in os.walk("."):
        if "quality_logs" in dirs:
            return Path(root) / "quality_logs"
            
    print("Error: Could not automatically detect quality_logs directory.", file=sys.stderr)
    print("Please specify the path using: python check_rejection_rate.py --log-dir <path>", file=sys.stderr)
    sys.exit(1)

def analyze_split(log_dir, split):
    summary_path = log_dir / f"summary_{split}.json"
    discarded_path = log_dir / f"discarded_{split}.jsonl"

    stats = {
        "kept_total": 0,
        "discarded_total": 0,
        "reasons": Counter(),
        "discarded_qualities": [],
        "source_breakdown": defaultdict(lambda: {"kept": 0, "discarded": 0})
    }
    
    has_any_data = False

    # 1. Parse summary JSON if it exists
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
                has_any_data = True
                
                # Load stats from summary
                summary_stats = summary.get("stats", {})
                stats["kept_total"] = summary_stats.get("kept_total", 0)
                stats["discarded_total"] = summary_stats.get("discarded_total", 0)
                
                # Extract reasons and sources from summary stats
                for key, val in summary_stats.items():
                    if key.startswith("discarded_reason::"):
                        reason = key.split("::", 1)[1]
                        stats["reasons"][reason] += val
                    elif key.startswith("kept::"):
                        src = key.split("::", 1)[1]
                        stats["source_breakdown"][src]["kept"] += val
                    elif key.startswith("discarded::"):
                        src = key.split("::", 1)[1]
                        stats["source_breakdown"][src]["discarded"] += val
        except Exception as e:
            print(f"Warning: Failed to parse summary JSON {summary_path}: {e}", file=sys.stderr)

    # 2. Parse discarded JSONL if it exists
    if discarded_path.exists():
        try:
            with open(discarded_path, "r", encoding="utf-8") as f:
                lines_read = 0
                jsonl_reasons = Counter()
                jsonl_qualities = []
                jsonl_sources = Counter()
                
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        lines_read += 1
                        
                        # Extract metrics
                        reason = entry.get("reason", "unknown")
                        quality = entry.get("quality", 0.0)
                        source = entry.get("source", "unknown")
                        
                        jsonl_reasons[reason] += 1
                        jsonl_qualities.append(quality)
                        jsonl_sources[source] += 1
                    except Exception:
                        pass
                
                if lines_read > 0:
                    has_any_data = True
                    # If JSONL exists, prefer its fine-grained details for rejections
                    stats["discarded_total"] = max(stats["discarded_total"], lines_read)
                    stats["reasons"] = jsonl_reasons
                    stats["discarded_qualities"] = jsonl_qualities
                    for src, count in jsonl_sources.items():
                        stats["source_breakdown"][src]["discarded"] = count
        except Exception as e:
            print(f"Warning: Failed to parse discarded JSONL {discarded_path}: {e}", file=sys.stderr)

    if not has_any_data:
        return None

    # Calculate computed stats
    total = stats["kept_total"] + stats["discarded_total"]
    rejection_rate = stats["discarded_total"] / total if total > 0 else 0.0
    avg_discarded_quality = (
        sum(stats["discarded_qualities"]) / len(stats["discarded_qualities"])
        if stats["discarded_qualities"] else None
    )

    return {
        "split": split,
        "total_samples": total,
        "kept_total": stats["kept_total"],
        "discarded_total": stats["discarded_total"],
        "rejection_rate": rejection_rate,
        "average_discarded_quality": avg_discarded_quality,
        "reasons": dict(stats["reasons"]),
        "sources": {src: dict(metrics) for src, metrics in stats["source_breakdown"].items()}
    }

def print_text_report(results):
    print("=" * 60)
    print("                 QUALITY LOGS ANALYSIS REPORT                 ")
    print("=" * 60)
    
    for split_res in results:
        print(f"\nSplit: {split_res['split'].upper()}")
        print("-" * 30)
        print(f"Total Samples Processed : {split_res['total_samples']:,}")
        print(f"  - Kept Samples        : {split_res['kept_total']:,}")
        print(f"  - Discarded Samples   : {split_res['discarded_total']:,}")
        print(f"Average Rejection Rate  : {split_res['rejection_rate']:.2%}")
        
        if split_res['average_discarded_quality'] is not None:
            print(f"Avg Discarded Quality   : {split_res['average_discarded_quality']:.4f}")
        else:
            print("Avg Discarded Quality   : N/A (no individual sample records found)")
            
        print("\nRejection Breakdown (Why were samples discarded?):")
        if split_res['reasons']:
            # Sort reasons by descending count
            sorted_reasons = sorted(split_res['reasons'].items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_reasons:
                reason_pct = count / split_res['discarded_total'] if split_res['discarded_total'] > 0 else 0.0
                print(f"  - '{reason}': {count:,} ({reason_pct:.2%} of discards)")
        else:
            print("  - No reasons logged.")
            
        print("\nSource Dataset Performance:")
        if split_res['sources']:
            for src, counts in sorted(split_res['sources'].items()):
                src_kept = counts.get("kept", 0)
                src_disc = counts.get("discarded", 0)
                src_total = src_kept + src_disc
                src_rej_rate = src_disc / src_total if src_total > 0 else 0.0
                print(f"  - {src:20}: Total {src_total:5,} | Kept {src_kept:5,} | Discarded {src_disc:5,} | Rejection Rate {src_rej_rate:6.2%}")
        else:
            print("  - No source metrics logged.")
        print("-" * 60)

def main():
    args = parse_args()
    log_dir = detect_log_dir(args.log_dir)
    print(f"[*] Analyzing quality logs in directory: {log_dir.absolute()}")

    splits_to_analyze = [args.split] if args.split != "all" else ["train", "val", "test"]
    results = []
    
    for split in splits_to_analyze:
        split_res = analyze_split(log_dir, split)
        if split_res:
            results.append(split_res)
            
    if not results:
        print(f"[-] No quality log data found for split(s): {', '.join(splits_to_analyze)}")
        return

    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        print_text_report(results)

if __name__ == "__main__":
    main()
