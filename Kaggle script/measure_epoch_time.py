import os
import time
from pathlib import Path
from datetime import datetime

ckpt_dir = Path("/home/binhhanh409/checkpoints")

def measure_epoch_times():
    if not ckpt_dir.exists():
        print(f"[!] Checkpoint directory {ckpt_dir} does not exist.")
        return

    files = sorted(list(ckpt_dir.glob("asl_model_epoch_*.pt")), key=lambda f: int(f.stem.split("_")[-1]))
    if not files:
        print("[*] No checkpoint files found yet in /home/binhhanh409/checkpoints.")
        return

    print("=" * 80)
    print(f"{'Epoch':<8} | {'Filename':<28} | {'Saved Timestamp (UTC)':<26} | {'Delta Time':<12}")
    print("=" * 80)

    prev_mtime = None
    deltas = []

    for f in files:
        epoch_num = int(f.stem.split("_")[-1])
        mtime = f.stat().st_mtime
        dt_str = datetime.utcfromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        
        if prev_mtime is not None:
            delta = mtime - prev_mtime
            deltas.append(delta)
            delta_str = f"{delta:.3f} sec"
        else:
            delta_str = "N/A (First)"

        print(f"Epoch {epoch_num:<2} | {f.name:<28} | {dt_str:<26} | {delta_str:<12}")
        prev_mtime = mtime

    print("=" * 80)
    if deltas:
        avg_delta = sum(deltas) / len(deltas)
        print(f"[+] Measured Average Real Epoch Time across {len(deltas)} epochs: {avg_delta:.3f} seconds / epoch")
    print("=" * 80)

if __name__ == "__main__":
    measure_epoch_times()
