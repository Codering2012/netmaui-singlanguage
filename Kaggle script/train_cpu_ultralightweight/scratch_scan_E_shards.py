import os
from pathlib import Path

p = Path("E:/datasets/asl_dataset/asl_preprocessed_phase1")
if p.exists():
    for root, dirs, files in os.walk(p):
        print(f"Dir: {root} -> {len(files)} files, {len(dirs)} dirs")
        if len(files) > 0:
            print(f"  Sample files: {files[:5]}")
