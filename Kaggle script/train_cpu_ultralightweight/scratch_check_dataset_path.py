import os
from pathlib import Path

p = Path("E:/datasets/asl_dataset/asl_preprocessed_phase1")
print(f"Path {p} exists: {p.exists()}")
if p.exists():
    pts = list(p.glob("*.pt"))
    print(f"Found {len(pts)} .pt shards in {p}")
