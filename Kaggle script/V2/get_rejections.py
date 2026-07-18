#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Analyze ASL preprocessing quality rejection codes and statistics.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/kaggle/working/asl_preprocessed_phase1",
        help="Path to the output directory containing quality_logs (default: /kaggle/working/asl_preprocessed_phase1)"
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    quality_log_dir = output_path / "quality_logs"

    if not quality_log_dir.exists():
        print(f"Error: Quality log directory not found at {quality_log_dir}")
        print("Please verify the output directory path.")
        return

    # Find all summary and discarded files
    summary_files = list(quality_log_dir.glob("summary_*.json"))
    discarded_files = list(quality_log_dir.glob("discarded_*.jsonl"))

    if not summary_files and not discarded_files:
        print(f"No log files found in {quality_log_dir}")
        return

    print("=" * 60)
    print(f"ASL PIPELINE QUALITY & REJECTION ANALYSIS")
    print(f"Source Directory: {quality_log_dir}")
    print("=" * 60)

    # 1. Parse Summary Files
    total_kept = 0
    total_discarded = 0
    reason_counts = defaultdict(int)
    source_kept = defaultdict(int)
    source_discarded = defaultdict(int)

    for sf in summary_files:
        try:
            with open(sf, "r") as f:
                data = json.load(f)
            stats = data.get("stats", {})
            
            total_kept += stats.get("kept_total", 0)
            total_discarded += stats.get("discarded_total", 0)

            for key, val in stats.items():
                if key.startswith("discarded_reason::"):
                    reason = key.split("::", 1)[1]
                    reason_counts[reason] += val
                elif key.startswith("kept::"):
                    src = key.split("::", 1)[1]
                    source_kept[src] += val
                elif key.startswith("discarded::"):
                    src = key.split("::", 1)[1]
                    source_discarded[src] += val
        except Exception as e:
            print(f"Warning: Failed to parse summary file {sf.name}: {e}")

    # 2. Parse Discarded Files to calculate average quality of discarded items
    discard_quality_by_reason = defaultdict(list)
    for df in discarded_files:
        try:
            with open(df, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    reason = item.get("reason", "unknown")
                    quality = item.get("quality", 0.0)
                    discard_quality_by_reason[reason].append(quality)
        except Exception as e:
            print(f"Warning: Failed to parse discarded file {df.name}: {e}")

    total_processed = total_kept + total_discarded
    if total_processed == 0:
        print("No samples have been processed yet.")
        return

    # Print general stats
    print(f"{'Total Samples Processed:':<30} {total_processed:,}")
    print(f"{'  - Kept:':<30} {total_kept:,} ({total_kept/total_processed*100:.2f}%)")
    print(f"{'  - Discarded:':<30} {total_discarded:,} ({total_discarded/total_processed*100:.2f}%)")
    print("-" * 60)

    # Print Rejection Reasons breakdown
    print(f"{'REJECTION REASON':<32} {'COUNT':<10} {'PERCENTAGE':<12} {'AVG QUALITY':<12}")
    print("-" * 60)
    
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    for reason, count in sorted_reasons:
        pct = (count / total_discarded * 100) if total_discarded > 0 else 0.0
        qualities = discard_quality_by_reason.get(reason, [])
        avg_quality = (sum(qualities) / len(qualities)) if qualities else 0.0
        print(f"{reason:<32} {count:<10,} {pct:.2f}%{'':<5} {avg_quality:.3f}")

    print("-" * 60)

    # Print Dataset Source breakdown
    print(f"{'DATASET SOURCE':<32} {'KEPT':<10} {'DISCARDED':<12} {'DISCARD RATE':<12}")
    print("-" * 60)
    all_sources = set(source_kept.keys()).union(source_discarded.keys())
    for src in sorted(all_sources):
        k = source_kept[src]
        d = source_discarded[src]
        tot = k + d
        rate = (d / tot * 100) if tot > 0 else 0.0
        print(f"{src:<32} {k:<10,} {d:<12,} {rate:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
