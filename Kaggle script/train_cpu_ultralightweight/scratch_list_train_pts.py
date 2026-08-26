import os
from pathlib import Path

p = Path("E:/datasets/asl_dataset/asl_preprocessed_phase1/train")
files = list(p.glob("*.pt"))
print(f"Found {len(files)} pt files in {p}")
if len(files) > 0:
    print(f"First 5: {[f.name for f in files[:5]]}")
