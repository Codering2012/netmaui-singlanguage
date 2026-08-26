import os
from pathlib import Path

for root_dir in ["D:/", "C:/Users/Windows 10 21H1/Downloads", "C:/Users/Windows 10 21H1/Desktop"]:
    p = Path(root_dir)
    if p.exists():
        print(f"Checking {p}...")
        try:
            for item in p.iterdir():
                if "dataset" in item.name.lower() or "asl" in item.name.lower() or "shard" in item.name.lower():
                    print(f"  Found match: {item}")
        except Exception as e:
            print(f"  Error reading {p}: {e}")
