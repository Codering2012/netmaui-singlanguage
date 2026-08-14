import torch
import glob
import numpy as np
from collections import defaultdict

files = sorted(glob.glob(r'E:\datasets\results\asl_preprocessed_phase1\**\shard_*.pt', recursive=True))
print(f"[*] Found {len(files)} shards under E:\\datasets\\results\\asl_preprocessed_phase1")

dataset_counts = defaultdict(int)
dataset_qualities = defaultdict(list)
dataset_hands = defaultdict(list)

# Sample across shards
sample_files = files[:15] + files[::3]
sample_files = sorted(list(set(sample_files)))

for f in sample_files:
    try:
        recs = torch.load(f, map_location="cpu", weights_only=False)
        for r in recs:
            src = r.get('source', 'Unknown')
            dataset_counts[src] += 1
            q = float(r.get('quality', 0.0))
            dataset_qualities[src].append(q)
            bh = float(r.get('quality_breakdown', {}).get('best_hand', q))
            dataset_hands[src].append(bh)
    except Exception as e:
        print(f"[!] Error reading {f}: {e}")

print("\n" + "="*85)
print(f"{'Dataset Source':<22} | {'Sample Count':<12} | {'Avg Quality':<12} | {'Avg Hand Conf':<14} | {'Status'}")
print("-" * 85)

for k in sorted(dataset_counts.keys()):
    cnt = dataset_counts[k]
    avg_q = float(np.mean(dataset_qualities[k]))
    avg_h = float(np.mean(dataset_hands[k]))
    status = "EXCELLENT" if avg_h >= 0.80 else ("GOOD" if avg_h >= 0.60 else "MARGINAL")
    print(f"{k:<22} | {cnt:<12} | {avg_q:<12.4f} | {avg_h:<14.4f} | {status}")

print("="*85)
