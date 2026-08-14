import glob
import torch
from collections import Counter, defaultdict

files = sorted(glob.glob(r'E:\datasets\results\asl_preprocessed_phase1\train\shard_*.pt'))[:5]
print(f"[*] Reading {len(files)} sample shards...")

task_counts = Counter()
task_labels = defaultdict(Counter)

for f in files:
    recs = torch.load(f, map_location="cpu", weights_only=False)
    for r in recs:
        t = r.get("task", "unknown")
        lbl = r.get("label", "<none>")
        task_counts[t] += 1
        task_labels[t][lbl] += 1

print("\nTask counts in sample:")
for task, cnt in task_counts.items():
    print(f"  - {task:<25}: {cnt:<6} records | {len(task_labels[task]):<5} unique labels")
    print(f"    Sample labels: {list(task_labels[task].keys())[:5]}")
