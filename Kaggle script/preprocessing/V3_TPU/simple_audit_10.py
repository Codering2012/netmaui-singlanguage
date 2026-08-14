import torch
import glob
import numpy as np
from collections import defaultdict

files = sorted(glob.glob(r"E:\datasets\results\asl_preprocessed_phase1\train\shard_*.pt"))
print(f"Total shards found: {len(files)}")

counts = defaultdict(int)
quals = defaultdict(list)
hands = defaultdict(list)

for f in files[:10]:
    recs = torch.load(f, map_location="cpu", weights_only=False)
    for r in recs:
        src = r.get("source", "Unknown")
        q = float(r.get("quality", 0.0))
        bd = r.get("quality_breakdown", {})
        h = float(bd.get("best_hand", bd.get("best_hand_conf", q)))
        counts[src] += 1
        quals[src].append(q)
        hands[src].append(h)

lines = []
lines.append("Dataset | Count | Avg Quality | Avg Hand Conf")
lines.append("-" * 50)
for k in sorted(counts.keys()):
    lines.append(f"{k:<20} | {counts[k]:<6} | {np.mean(quals[k]):.4f}      | {np.mean(hands[k]):.4f}")

result = "\n".join(lines)
print(result)
with open("audit_res.txt", "w") as out:
    out.write(result)
