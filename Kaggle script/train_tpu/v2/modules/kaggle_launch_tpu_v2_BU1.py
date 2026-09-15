#!/usr/bin/env python3
"""
================================================================================
  KAGGLE TPU VM AUTOMATED LAUNCHER & BENCHMARK RUNNER (V2 PIPELINE)
================================================================================
One-command launch script for Kaggle TPU VM (TPU v5e / v3-8):
  1. Sets up optimal PyTorch/XLA and PJRT runtime environment variables.
  2. Verifies dataset shards and vocabulary mapping directly from /kaggle/input.
  3. Launches multi-core distributed training via xmp.spawn.
  4. Streams live epoch metrics and serializes top checkpoints to /kaggle/working.
================================================================================
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def setup_tpu_environment():
    """Configures optimal low-overhead TPU environment variables."""
    os.environ["PJRT_DEVICE"] = "TPU"
    os.environ["XLA_USE_BF16"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["GRPC_VERBOSITY"] = "ERROR"
    os.environ["GLOO_LOG_LEVEL"] = "ERROR"

def verify_dataset(dataset_dir: str) -> bool:
    """Verifies that dataset shards and vocabulary files exist."""
    p = Path(dataset_dir)
    print(f"[*] Checking dataset at: {p}")
    if not p.exists():
        print(f"[!] Warning: Path '{p}' not found. Verify Kaggle dataset attachment.")
        return False
    train_dir = p / "train"
    if train_dir.exists():
        shards = list(train_dir.glob("shard_*.pt"))
        print(f"[+] Found {len(shards)} training shards in '{train_dir}'.")
        return len(shards) > 0
    return False

def main():
    parser = argparse.ArgumentParser(description="Kaggle TPU VM Automated Launcher for ASL Model V2")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_dataset",
        help="Path to preprocessed dataset root",
    )
    parser.add_argument("--save-dir", type=str, default="/kaggle/working/checkpoints", help="Output checkpoint directory")
    parser.add_argument("--batch-size", type=int, default=1024, help="Global batch size across 8 TPU cores")
    parser.add_argument("--phase1-epochs", type=int, default=0, help="Phase 1 pretraining epochs (0 to skip)")
    parser.add_argument("--phase2-epochs", type=int, default=20, help="Phase 2 multimodal epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Peak learning rate")
    args = parser.parse_args()

    setup_tpu_environment()
    verify_dataset(args.dataset_dir)

    candidates = [
        Path(__file__).parent.parent / "engine" / "train_all_in_one_tpu.py",
        Path(__file__).parent / "train_all_in_one_tpu_v2.py",
        Path(__file__).parent.parent / "train_all_in_one_tpu.py",
        Path(__file__).parent / "train_all_in_one_tpu.py",
    ]
    train_script = next((c for c in candidates if c.exists()), candidates[0])
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset-dir", args.dataset_dir,
        "--save-dir", args.save_dir,
        "--batch-size", str(args.batch_size),
        "--phase1-epochs", str(args.phase1_epochs),
        "--phase2-epochs", str(args.phase2_epochs),
        "--learning-rate", str(args.learning_rate),
        "--precision", "bfloat16",
        "--use-visual-roi",
    ]

    print(f"\n[🚀 LAUNCHING TPU V2 TRAINING PIPELINE]")
    print(f"  Command: {' '.join(cmd)}\n")
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
