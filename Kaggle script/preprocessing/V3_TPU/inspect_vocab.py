import glob
import torch
from collections import Counter, defaultdict

files = sorted(glob.glob(r'E:\datasets\results\asl_preprocessed_phase1\train\shard_*.pt'))
print(f"[*] Inspecting {len(files)} shards under E:\\datasets\\results\\asl_preprocessed_phase1\\train...")

task_counts = Counter()
task_labels = defaultdict(Counter)
source_counts = Counter()

for f in files:
    try:
        recs = torch.load(f, map_location="cpu", weights_only=False)
        for r in recs:
            t = r.get("task", "unknown")
            src = r.get("source", "unknown")
            lbl = r.get("label", "<none>")
            task_counts[t] += 1
            source_counts[src] += 1
            task_labels[t][lbl] += 1
    except Exception as e:
        print(f"[!] Error reading {f}: {e}")

print("\n--- Summary by Source Dataset ---")
for src, cnt in source_counts.most_common():
    print(f"  {src:<22}: {cnt:<8} records")

print("\n--- Summary by Task Type ---")
for task, cnt in task_counts.most_common():
    unique_cnt = len(task_labels[task])
    print(f"  {task:<22}: {cnt:<8} records | {unique_cnt:<6} unique labels")

print("\n--- Sample Labels per Task Type ---")
for task, labels in task_labels.items():
    print(f"\nTask: '{task}' ({len(labels)} unique labels)")
    samples = list(labels.items())[:8]
    for lbl, freq in samples:
        print(f"   -> '{lbl}': {freq} occurrences")
