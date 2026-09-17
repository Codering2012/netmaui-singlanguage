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

import glob
import ctypes

def _preload_nvidia_cuda_libraries():
    """
    Automatically locates and preloads pip-installed NVIDIA shared objects
    (including libcublasLt.so.13 and libcudnn.so.9) using ctypes with RTLD_GLOBAL.
    This guarantees ONNX Runtime's dlopen() can resolve symbols even if LD_LIBRARY_PATH
    was not set prior to Python process launch.
    """
    try:
        import site
        site_dirs = []
        if hasattr(site, "getsitepackages"):
            site_dirs.extend(site.getsitepackages())
        if hasattr(site, "getusersitepackages"):
            site_dirs.append(site.getusersitepackages())
        
        nvidia_dirs = []
        for s in site_dirs:
            if s and os.path.isdir(s):
                nvidia_dirs.extend(glob.glob(os.path.join(s, "nvidia", "*", "lib")))
                nvidia_dirs.extend(glob.glob(os.path.join(s, "torch", "lib")))
        
        if nvidia_dirs:
            os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_dirs) + ":" + os.environ.get("LD_LIBRARY_PATH", "")

        candidates = []
        for d in nvidia_dirs:
            for so_file in glob.glob(os.path.join(d, "*.so*")):
                if os.path.isfile(so_file) and not os.path.islink(so_file):
                    candidates.append(so_file)
        
        priority = ["cuda_runtime", "cublaslt", "cublas", "cudnn"]
        def rank(p):
            n = os.path.basename(p).lower()
            for idx, key in enumerate(priority):
                if key in n:
                    return idx
            return len(priority)
        
        candidates = sorted(list(set(candidates)), key=rank)
        for c in candidates:
            try:
                ctypes.CDLL(c, mode=ctypes.RTLD_GLOBAL)
            except Exception:
                pass
    except Exception:
        pass

    try:
        import onnxruntime as ort
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:
        pass

_preload_nvidia_cuda_libraries()

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
                raw_clip_id = parts[id_col].strip()
                text = parts[text_col].strip()
                transcriptions[raw_clip_id] = text
                stem_id = Path(raw_clip_id).stem
                if stem_id not in transcriptions:
                    transcriptions[stem_id] = text

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

    # Discover zip archives and raw video files
    archive_pattern = getattr(args, "archive_pattern", None)
    if archive_pattern:
        zip_files = sorted(list(drive_dir.glob(archive_pattern)))
        print(f"Using archive pattern '{archive_pattern}': matched {len(zip_files)} archives.")
    else:
        # Prioritize segmented sentence clips over raw video archives
        all_zips = list(drive_dir.glob("*.zip"))
        # Sort so that 'clips' or 'front' come first, and raw videos come last
        zip_files = sorted(
            all_zips,
            key=lambda p: (0 if "clip" in p.name.lower() or "front" in p.name.lower() else (2 if "raw" in p.name.lower() else 1), p.name)
        )

    raw_video_files = []
    if not zip_files:
        raw_video_files = sorted([p for p in drive_dir.rglob("*") if p.suffix.lower() in [".mp4", ".mkv", ".avi", ".mov"] and not p.name.startswith(".")])
        if not raw_video_files:
            print(f"[!] Error: No .zip archives or video files (.mp4) found in {drive_dir}")
            return
        print(f"Found {len(raw_video_files)} raw video files to stream.")
    else:
        print(f"Found {len(zip_files)} zip archives to stream:")
        for z in zip_files:
            print(f"  - {z.name}")

    # Load transcriptions from all CSV files found
    transcriptions = {}
    for csv_file in drive_dir.glob("*.csv"):
        print(f"Loading transcriptions from: {csv_file.name}")
        transcriptions.update(parse_how2sign_csv(csv_file))
    print(f"Loaded {len(transcriptions)} total sentence transcriptions.")

    # Initialize Sentence Transformer only if explicitly requested by user (default False: pure fast video preprocessing)
    sent_model = None
    if getattr(args, "extract_sentence_embeddings", False):
        try:
            from sentence_transformers import SentenceTransformer
            print("[INFO] Loading 'all-MiniLM-L6-v2' for offline sentence embedding extraction...")
            sent_model = SentenceTransformer("all-MiniLM-L6-v2")
            print("[SUCCESS] SentenceTransformer loaded.")
        except Exception as e:
            print(f"[NOTE] sentence-transformers not active ({e}); skipping offline sentence embeddings.")

    # Initialize PreprocessorV4 on GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocessor = VideoPreprocessorV4(
        target_fps=args.target_fps,
        backend=args.backend,
        device=device,
        canonicalize_hands=True,
    )

    # Verify if CUDAExecutionProvider is actually active (prevents silent CPU fallback freeze)
    if args.backend == "rtmw" and device == "cuda":
        active_providers = []
        if preprocessor.rtmw_model is not None:
            for sub_attr in ("pose_model", "det_model"):
                sub_mod = getattr(preprocessor.rtmw_model, sub_attr, None)
                if sub_mod is not None and hasattr(sub_mod, "session"):
                    sess = getattr(sub_mod, "session", None)
                    if sess is not None and hasattr(sess, "get_providers"):
                        active_providers = sess.get_providers()
                        break
        print(f"[RTMW] Active ONNX Providers: {active_providers}", flush=True)
        if active_providers and "CUDAExecutionProvider" not in active_providers:
            print("\n" + "!" * 72, flush=True)
            print("[CRITICAL WARNING] CUDA was requested, but ONNX Runtime failed to load CUDAExecutionProvider!", flush=True)
            print(f"Active providers fallback: {active_providers}", flush=True)
            print("On CPU, RTMW WholeBody runs at ~1-2 FPS (~2-3 minutes per video).", flush=True)
            print("To fix this in Colab, install 'onnxruntime-gpu[cuda,cudnn]' and set LD_LIBRARY_PATH.", flush=True)
            print("!" * 72 + "\n", flush=True)
            if not getattr(args, "allow_cpu_fallback", False):
                raise RuntimeError(
                    "CUDAExecutionProvider failed to initialize. Aborting to prevent multi-hour silent CPU freeze. "
                    "Use --allow-cpu-fallback if you explicitly want slow CPU processing."
                )

    uploader = ShardUploaderThread(output_drive_dir, ledger_path)
    current_buf, next_buf = buf_a, buf_b
    shard_idx = len(list(output_drive_dir.glob("shard_*.pt")))

    if raw_video_files:
        unprocessed_videos = [p for p in raw_video_files if p.stem not in completed_clips]
        print(f"Streaming {len(unprocessed_videos)} remaining raw video files...", flush=True)
        chunk_size = args.chunk_size
        for chunk_start in range(0, len(unprocessed_videos), chunk_size):
            chunk_files = unprocessed_videos[chunk_start : chunk_start + chunk_size]
            processed_data = []
            chunk_clip_ids = []
            t_chunk_start = time.time()
            for v_idx, video_file in enumerate(chunk_files):
                clip_id = video_file.stem
                text = transcriptions.get(clip_id, "")
                t_clip_start = time.time()
                try:
                    result = preprocessor.extract_from_video(
                        str(video_file),
                        include_roi=args.include_roi,
                        include_hand_crop=args.include_hand_crop,
                    )
                    clip_time = time.time() - t_clip_start
                    if result is not None:
                        n_frames = (
                            result["landmarks"].shape[0]
                            if isinstance(result.get("landmarks"), (torch.Tensor, np.ndarray))
                            else len(result.get("landmarks", []))
                        )
                        fps = (n_frames / clip_time) if clip_time > 0 else 0.0
                        result["id"] = clip_id
                        result["label"] = text
                        result["text"] = text
                        if sent_model is not None and text:
                            emb = sent_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                            result["sentence_embedding"] = torch.from_numpy(emb).half()
                        processed_data.append(result)
                        chunk_clip_ids.append(clip_id)
                        text_preview = f'"{text[:32]}..."' if len(text) > 32 else (f'"{text}"' if text else "[No text]")
                        print(
                            f"  [{v_idx+1:3d}/{len(chunk_files):3d}] OK: {clip_id:<28} | "
                            f"{n_frames:3d} frames | {clip_time:5.2f}s ({fps:4.1f} fps) | {text_preview}",
                            flush=True,
                        )
                    else:
                        print(f"  [{v_idx+1:3d}/{len(chunk_files):3d}] SKIPPED: {clip_id} (No valid pose extracted)", flush=True)
                except Exception as e:
                    print(f"  [{v_idx+1:3d}/{len(chunk_files):3d}] ERROR: {clip_id}: {e}", flush=True)

            chunk_elapsed = time.time() - t_chunk_start
            if processed_data:
                shard_name = f"shard_{shard_idx:05d}.pt"
                local_shard_path = local_root / shard_name
                torch.save(processed_data, str(local_shard_path))
                print(
                    f"\n[+] Shard Saved: {shard_name} ({len(processed_data)} clips in {chunk_elapsed:.1f}s, "
                    f"{chunk_elapsed/max(1, len(processed_data)):.2f}s/clip). Uploading to Drive in background...",
                    flush=True,
                )
                uploader.upload_async(local_shard_path, chunk_clip_ids)
                shard_idx += 1
    else:
        for z_idx, zip_path in enumerate(zip_files):
            print(f"\n[{z_idx+1}/{len(zip_files)}] Inspecting archive: {zip_path.name}", flush=True)
            try:
                with zipfile.ZipFile(str(zip_path), "r") as zf:
                    all_entries = [info for info in zf.infolist() if not info.is_dir() and info.filename.lower().endswith(".mp4")]
                    unprocessed_entries = [info for info in all_entries if Path(info.filename).stem not in completed_clips]
            except zipfile.BadZipFile as bzf:
                split_parts = list(drive_dir.glob(f"*{zip_path.stem}*.z*")) + list(drive_dir.glob("*.z01"))
                if split_parts:
                    print(f"[!] Note: '{zip_path.name}' appears to be part of a split multi-part archive (.z01-.z09).", flush=True)
                    print(f"    Python standard zipfile cannot read split volumes directly. Skipping raw split archive.", flush=True)
                    print(f"    (If you have 'train_rgb_front_clips.zip', it will be processed next as the primary clips archive!)", flush=True)
                else:
                    print(f"[!] Skipping unreadable archive '{zip_path.name}': {bzf}", flush=True)
                continue
            except Exception as e:
                print(f"[!] Skipping '{zip_path.name}' due to error: {e}", flush=True)
                continue

            print(f"Archive contains {len(all_entries)} clips ({len(unprocessed_entries)} remaining).", flush=True)
            if not unprocessed_entries:
                continue

            # Process in chunks of chunk_size
            chunk_size = args.chunk_size
            total_chunks = (len(unprocessed_entries) + chunk_size - 1) // chunk_size
            for chunk_num, chunk_start in enumerate(range(0, len(unprocessed_entries), chunk_size)):
                chunk_infos = unprocessed_entries[chunk_start : chunk_start + chunk_size]
                print(f"\n--- [Archive {z_idx+1}/{len(zip_files)} | Chunk {chunk_num+1}/{total_chunks}] Extracting {len(chunk_infos)} clips ---", flush=True)

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
                t_chunk_start = time.time()
                for c_i, info in enumerate(chunk_infos):
                    video_file = current_buf / info.filename
                    if not video_file.exists():
                        continue

                    clip_id = video_file.stem
                    text = transcriptions.get(clip_id, "")
                    t_clip_start = time.time()

                    try:
                        result = preprocessor.extract_from_video(
                            str(video_file),
                            include_roi=args.include_roi,
                            include_hand_crop=args.include_hand_crop,
                        )
                        clip_time = time.time() - t_clip_start
                        if result is not None:
                            n_frames = (
                                result["landmarks"].shape[0]
                                if isinstance(result.get("landmarks"), (torch.Tensor, np.ndarray))
                                else len(result.get("landmarks", []))
                            )
                            fps = (n_frames / clip_time) if clip_time > 0 else 0.0
                            result["id"] = clip_id
                            result["label"] = text
                            result["text"] = text
                            if sent_model is not None and text:
                                emb = sent_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
                                result["sentence_embedding"] = torch.from_numpy(emb).half()
                            processed_data.append(result)
                            chunk_clip_ids.append(clip_id)
                            text_preview = f'"{text[:32]}..."' if len(text) > 32 else (f'"{text}"' if text else "[No text]")
                            print(
                                f"  [{c_i+1:3d}/{len(chunk_infos):3d}] OK: {clip_id:<28} | "
                                f"{n_frames:3d} frames | {clip_time:5.2f}s ({fps:4.1f} fps) | {text_preview}",
                                flush=True,
                            )
                        else:
                            print(f"  [{c_i+1:3d}/{len(chunk_infos):3d}] SKIPPED: {clip_id} (No valid pose extracted)", flush=True)
                    except Exception as e:
                        print(f"  [{c_i+1:3d}/{len(chunk_infos):3d}] ERROR: {clip_id}: {e}", flush=True)

                chunk_elapsed = time.time() - t_chunk_start
                if processed_data:
                    shard_name = f"shard_{shard_idx:05d}.pt"
                    local_shard_path = local_root / shard_name
                    torch.save(processed_data, str(local_shard_path))
                    print(
                        f"\n[+] Shard Saved: {shard_name} ({len(processed_data)} clips in {chunk_elapsed:.1f}s, "
                        f"{chunk_elapsed/max(1, len(processed_data)):.2f}s/clip). Uploading to Drive in background...",
                        flush=True,
                    )

                    # 4. Trigger asynchronous background upload to Google Drive
                    uploader.upload_async(local_shard_path, chunk_clip_ids)
                    shard_idx += 1

                # 5. Swap ping-pong buffers
                current_buf, next_buf = next_buf, current_buf

    # Finalize any pending upload
    uploader.wait_for_completion()
    print("\n[SUCCESS] Double-buffered How2Sign V4 streaming preprocessing fully complete!", flush=True)


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
    parser.add_argument("--extract-sentence-embeddings", action="store_true", default=False, help="Extract offline 384-D sentence embeddings via sentence-transformers (default: False)")
    parser.add_argument("--archive-pattern", type=str, default=None, help="Glob pattern or exact name of archive to process (e.g. '*clips*.zip' or 'train_rgb_front_clips.zip')")
    parser.add_argument("--allow-cpu-fallback", action="store_true", default=False, help="Allow continuing on CPU if CUDA provider fails")
    args = parser.parse_args()

    run_streaming_pipeline(args)


if __name__ == "__main__":
    main()
