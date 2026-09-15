#!/usr/bin/env python3
"""
================================================================================
HOW2SIGN PREPROCESSOR V4 DOUBLE-BUFFERED STREAMER FOR GOOGLE COLAB
================================================================================
Implements high-throughput, double-buffered producer-consumer video preprocessing:
1. Ping-Pong Buffers: Main thread runs PreprocessorV4 on GPU while background thread
   uploads previous shard to Google Drive.
2. Zero GPU Stall: Completely hides Google Drive FUSE latency behind GPU compute.
3. Crash-Resilient Ledger: Resumes automatically from interrupted shards.
4. Native V4 Integration: 9D kinematics, 19D phonology, 3D IMU, 12D NMM, quality weights.
================================================================================
"""

import os
import sys
import json
import time
import shutil
import zipfile
import argparse
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
import numpy as np

# Ensure local imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from preprocessor_v4 import VideoPreprocessorV4
except ImportError:
    # Try importing from current directory
    from preprocessing.preprocessor_v4 import VideoPreprocessorV4


class ShardUploaderThread:
    """
    Background worker thread that safely uploads completed shards to Google Drive
    and atomically updates the ledger without blocking GPU computation.
    """

    def __init__(self, target_drive_dir: Path, ledger_path: Path):
        self.target_drive_dir = target_drive_dir
        self.ledger_path = ledger_path
        self.thread: Optional[threading.Thread] = None
        self.error: Optional[Exception] = None
        self.lock = threading.Lock()

    def upload_async(self, local_shard_path: Path, processed_clip_ids: List[str]):
        """Starts asynchronous background upload of the completed shard."""
        self.wait_for_completion()
        self.thread = threading.Thread(
            target=self._upload_worker,
            args=(local_shard_path, processed_clip_ids),
            daemon=True,
        )
        self.thread.start()

    def wait_for_completion(self):
        """Blocks until the pending background upload completes."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join()
        if self.error is not None:
            err = self.error
            self.error = None
            raise RuntimeError(f"Background upload thread failed: {err}") from err

    def _upload_worker(self, local_shard_path: Path, processed_clip_ids: List[str]):
        try:
            drive_shard_path = self.target_drive_dir / local_shard_path.name
            temp_drive_path = self.target_drive_dir / f"{local_shard_path.name}.tmp"

            # 1. Copy shard to temporary file on Drive
            shutil.copy2(str(local_shard_path), str(temp_drive_path))

            # 2. Verify file size matches
            local_size = local_shard_path.stat().st_size
            drive_size = temp_drive_path.stat().st_size
            if local_size != drive_size:
                raise IOError(f"Size mismatch on Drive: {drive_size} vs local {local_size}")

            # 3. Atomic rename on Drive
            if drive_shard_path.exists():
                drive_shard_path.unlink()
            temp_drive_path.rename(drive_shard_path)

            # 4. Atomically update ledger
            with self.lock:
                ledger = {}
                if self.ledger_path.exists():
                    try:
                        with open(self.ledger_path, "r", encoding="utf-8") as f:
                            ledger = json.load(f)
                    except Exception:
                        ledger = {}

                completed_clips = set(ledger.get("completed_clips", []))
                completed_clips.update(processed_clip_ids)
                ledger["completed_clips"] = list(completed_clips)
                ledger["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

                temp_ledger = self.ledger_path.with_suffix(".tmp")
                with open(temp_ledger, "w", encoding="utf-8") as f:
                    json.dump(ledger, f, indent=2)
                temp_ledger.replace(self.ledger_path)

            # 5. Clean up local shard file
            if local_shard_path.exists():
                local_shard_path.unlink()

        except Exception as e:
            self.error = e


def parse_how2sign_csv(csv_path: Path) -> Dict[str, str]:
    """Parses How2Sign realigned transcription CSV file."""
    transcriptions = {}
    if not csv_path.exists():
        return transcriptions

    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
        delim = "\t" if "\t" in first_line else ","
        f.seek(0)
        header = f.readline().strip().split(delim)

        id_col = 0
        text_col = 1
        for i, col in enumerate(header):
            c_upper = col.upper().strip()
            if "SENTENCE_NAME" in c_upper or "VIDEO_NAME" in c_upper or "CLIP" in c_upper:
                id_col = i
            elif "SENTENCE" in c_upper or "TRANSCRIPTION" in c_upper or "TEXT" in c_upper:
                text_col = i

        for line in f:
            parts = line.strip().split(delim)
            if len(parts) > max(id_col, text_col):
                clip_id = parts[id_col].strip()
                text = parts[text_col].strip()
                transcriptions[clip_id] = text

    return transcriptions


def run_streaming_pipeline(args):
    """Executes the double-buffered How2Sign V4 streaming preprocessor."""
    drive_dir = Path(args.drive_dir)
    output_drive_dir = Path(args.output_drive_dir)
    output_drive_dir.mkdir(parents=True, exist_ok=True)

    local_root = Path(args.local_scratch)
    buf_a = local_root / "chunk_A"
    buf_b = local_root / "chunk_B"
    buf_a.mkdir(parents=True, exist_ok=True)
    buf_b.mkdir(parents=True, exist_ok=True)

    ledger_path = output_drive_dir / "ledger.json"
    completed_clips = set()
    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                completed_clips = set(json.load(f).get("completed_clips", []))
            print(f"Loaded ledger: {len(completed_clips)} clips previously completed.")
        except Exception:
            completed_clips = set()

    # Discover zip archives and CSVs
    zip_files = sorted(list(drive_dir.glob("*.zip")))
    if not zip_files:
        print(f"No .zip archives found in {drive_dir}")
        return

    print(f"Found {len(zip_files)} zip archives to stream.")

    # Load transcriptions from all CSV files found
    transcriptions = {}
    for csv_file in drive_dir.glob("*.csv"):
        print(f"Loading transcriptions from: {csv_file.name}")
        transcriptions.update(parse_how2sign_csv(csv_file))
    print(f"Loaded {len(transcriptions)} total sentence transcriptions.")

    # Initialize PreprocessorV4 on GPU
    preprocessor = VideoPreprocessorV4(
        target_fps=args.target_fps,
        pose_backend=args.backend,
        canonicalize_hands=True,
    )

    uploader = ShardUploaderThread(output_drive_dir, ledger_path)
    current_buf, next_buf = buf_a, buf_b
    shard_idx = len(list(output_drive_dir.glob("shard_*.pt")))

    for z_idx, zip_path in enumerate(zip_files):
        print(f"\n[{z_idx+1}/{len(zip_files)}] Inspecting archive: {zip_path.name}")
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            all_entries = [info for info in zf.infolist() if not info.is_dir() and info.filename.lower().endswith(".mp4")]
            unprocessed_entries = [info for info in all_entries if Path(info.filename).stem not in completed_clips]

        print(f"Archive contains {len(all_entries)} clips ({len(unprocessed_entries)} remaining).")
        if not unprocessed_entries:
            continue

        # Process in chunks of chunk_size
        chunk_size = args.chunk_size
        for chunk_start in range(0, len(unprocessed_entries), chunk_size):
            chunk_infos = unprocessed_entries[chunk_start : chunk_start + chunk_size]

            # 1. Clear current extraction buffer
            shutil.rmtree(str(current_buf), ignore_errors=True)
            current_buf.mkdir(parents=True, exist_ok=True)

            # 2. Extract batch into current buffer
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                for info in chunk_infos:
                    zf.extract(info, path=str(current_buf))

            # 3. Process each extracted clip on GPU
            processed_data = []
            chunk_clip_ids = []
            for info in chunk_infos:
                video_file = current_buf / info.filename
                if not video_file.exists():
                    continue

                clip_id = video_file.stem
                text = transcriptions.get(clip_id, "")

                try:
                    result = preprocessor.extract_from_video(
                        str(video_file),
                        include_roi=args.include_roi,
                        include_hand_crop=args.include_hand_crop,
                    )
                    if result is not None:
                        result["id"] = clip_id
                        result["label"] = text
                        processed_data.append(result)
                        chunk_clip_ids.append(clip_id)
                except Exception as e:
                    print(f"Error processing {clip_id}: {e}")

            if processed_data:
                shard_name = f"shard_{shard_idx:05d}.pt"
                local_shard_path = local_root / shard_name
                torch.save(processed_data, str(local_shard_path))
                print(f"Saved local shard {shard_name} ({len(processed_data)} clips). Triggering background upload...")

                # 4. Trigger asynchronous background upload to Google Drive
                uploader.upload_async(local_shard_path, chunk_clip_ids)
                shard_idx += 1

            # 5. Swap ping-pong buffers
            current_buf, next_buf = next_buf, current_buf

    # Finalize any pending upload
    uploader.wait_for_completion()
    print("\n[SUCCESS] Double-buffered How2Sign V4 streaming preprocessing fully complete!")


def main():
    parser = argparse.ArgumentParser(description="How2Sign Colab V4 Streamer")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/How2Sign", help="Input Drive path")
    parser.add_argument("--output-drive-dir", type=str, default="/content/drive/MyDrive/How2Sign_Preprocessed", help="Output Drive path")
    parser.add_argument("--local-scratch", type=str, default="/content/scratch", help="Local Colab scratch directory")
    parser.add_argument("--chunk-size", type=int, default=100, help="Clips per shard")
    parser.add_argument("--backend", type=str, default="rtmw", help="Pose backend (rtmw or mediapipe)")
    parser.add_argument("--target-fps", type=float, default=30.0, help="Uniform target FPS")
    parser.add_argument("--include-roi", action="store_true", help="Include 256x256 upper-body ROI crops")
    parser.add_argument("--include-hand-crop", action="store_true", help="Include 128x128 hand crops")
    args = parser.parse_args()

    run_streaming_pipeline(args)


if __name__ == "__main__":
    main()
