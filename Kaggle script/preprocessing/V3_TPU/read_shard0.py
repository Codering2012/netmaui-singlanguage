import torch
import glob
import numpy as np

files = sorted(glob.glob(r"E:\datasets\results\asl_preprocessed_phase1\train\shard_000*.pt"))
print(f"Testing {len(files)} shards...")

records = torch.load(files[0], map_location="cpu", weights_only=False)
print(f"Shard 0 records: {len(records)}")

sources = {}
for r in records:
    src = r.get("source", "Unknown")
    q = float(r.get("quality", 0.0))
    bd = r.get("quality_breakdown", {})
    h = float(bd.get("best_hand", bd.get("best_hand_conf", q)))
    if src not in sources:
        sources[src] = {"q": [], "h": []}
    sources[src]["q"].append(q)
    sources[src]["h"].append(h)

out_text = f"Analyzed {len(records)} records from Shard 0:\n"
for s in sorted(sources.keys()):
    out_text += f"  - {s:<20}: Avg Qual = {np.mean(sources[s]['q']):.4f}, Avg Hand Conf = {np.mean(sources[s]['h']):.4f}\n"

print(out_text)
with open("shard0_summary.txt", "w") as f:
    f.write(out_text)
