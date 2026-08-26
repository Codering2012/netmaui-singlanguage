import sys
import os

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import glob
from pathlib import Path
from collections import Counter
import torch

def main():
    shards = sorted(glob.glob('E:/datasets/asl_dataset/asl_preprocessed_phase1/train/shard_*.pt'))
    print(f"[*] Quét toàn bộ {len(shards)} shards trong tập huấn luyện...")
    
    len_counts = Counter()
    task_counts = Counter()
    samples_by_len = {}
    
    for s_idx, s in enumerate(shards):
        try:
            data = torch.load(s, map_location='cpu', weights_only=False)
            for item in data:
                if not isinstance(item, dict) or 'features' not in item:
                    continue
                f = item['features']
                t_len = f.shape[0] if hasattr(f, 'shape') else len(f)
                len_counts[t_len] += 1
                
                task = item.get('task', 'unknown')
                task_counts[task] += 1
                
                if t_len not in samples_by_len and len(samples_by_len) < 10:
                    samples_by_len[t_len] = {
                        'label': item.get('label'),
                        'task': task,
                        'source': item.get('source'),
                        'shape': list(f.shape) if hasattr(f, 'shape') else None
                    }
        except Exception as e:
            print(f"Error on {s}: {e}")
            
    print("\n[+] Phân phối độ dài khung hình (Sequence Length Distribution):")
    for length, count in sorted(len_counts.items(), key=lambda x: x[0]):
        print(f"  Length T = {length:3d} frames: {count:7d} samples ({count/sum(len_counts.values())*100:.2f}%)")
        
    print("\n[+] Phân phối theo loại tác vụ (Task Types):")
    for task, count in task_counts.items():
        print(f"  Task '{task}': {count} samples")
        
    print("\n[+] Ví dụ mẫu theo các độ dài khác nhau:")
    for length, info in sorted(samples_by_len.items()):
        print(f"  T = {length:3d} -> Label: '{info['label']}', Task: '{info['task']}', Source: '{info['source']}', Shape: {info['shape']}")

if __name__ == "__main__":
    main()
