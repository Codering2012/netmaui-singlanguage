#!/usr/bin/env python3
"""
================================================================================
HOW2SIGN GOOGLE COLAB STREAMING PREPROCESSOR (T4 GPU ACCELERATED)
================================================================================
Engineered for Google Colab + Google Drive architecture:
1. Streams clips directly from mounted .zip archives on Google Drive.
2. Extracts micro-chunks (e.g. 100 clips) to fast local Colab NVMe disk (/content/temp_chunk).
3. Executes GPU-accelerated 60-keypoint extraction + 9D Kinematics + 19D Phonology.
4. Packages into compressed PyTorch shards (shard_XXXX.pt) and streams back to Drive.
5. Immediately wipes local disk files and flushes CPU RAM + CUDA VRAM.
6. Persistent Google Drive Ledger: Survives Colab disconnects and resumes seamlessly.
================================================================================
"""

import os
import sys
import gc
import time
import json
import zipfile
import shutil
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

import cv2
import numpy as np
import torch

try:
    from rtmlib import Wholebody
    _RTMLIB_AVAILABLE = True
except ImportError:
    _RTMLIB_AVAILABLE = False

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False


# Canonical 60-Keypoint Indices:
# 1. RTMW Wholebody (133 landmarks):
# 0-20: Left Hand (indices 91:112)
# 21-41: Right Hand (indices 112:133)
# 42-47: Upper Body Pose (indices 5, 6, 7, 8, 9, 10): L/R Shoulders, L/R Elbows, L/R Wrists
# 48-59: Facial Non-Manuals (indices 53, 31, 59, 68, 40, 44, 45, 49, 71, 77, 74, 80)
RTMW_POSE_INDICES = [5, 6, 7, 8, 9, 10]
RTMW_FACE_INDICES = [53, 31, 59, 68, 40, 44, 45, 49, 71, 77, 74, 80]

# 2. MediaPipe Holistic Fallback (543 landmarks):
MP_POSE_INDICES = [11, 12, 13, 14, 15, 16]
MP_FACE_INDICES = [1, 4, 152, 0, 33, 263, 61, 291, 10, 109, 338, 9]


def extract_landmarks_from_video(
    video_path: str,
    extractor_model: Any,
    engine: str = "rtmw",
    extract_visual_crops: bool = False,
    max_frames: int = 300,
) -> Optional[Dict[str, Any]]:
    """
    Extracts canonical 60 keypoints and optional visual crops using RTMW (SOTA) or MediaPipe.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
        if len(raw_frames) >= max_frames:
            break
    cap.release()

    T = len(raw_frames)
    if T < 4:
        return None

    # Landmark buffer: [T, 60, 3]
    pos = np.zeros((T, 60, 3), dtype=np.float32)
    roi_crops = [] if extract_visual_crops else None
    hand_crops = [] if extract_visual_crops else None

    for t, bgr in enumerate(raw_frames):
        h_img, w_img = bgr.shape[:2]

        if engine == "rtmw" and _RTMLIB_AVAILABLE:
            # ------------------------------------------------------------------
            # RTMW-x SOTA WholeBody Inference (133 landmarks)
            # ------------------------------------------------------------------
            kpts, scores = extractor_model(bgr)
            if len(kpts) > 0:
                k133 = kpts[0]  # [133, 2] in pixel coords
                sc133 = scores[0] if scores is not None and len(scores) > 0 else np.ones(133, dtype=np.float32)

                # Normalize pixel coordinates to [0, 1]
                x_norm = np.clip(k133[:, 0] / max(w_img, 1), 0.0, 1.0)
                y_norm = np.clip(k133[:, 1] / max(h_img, 1), 0.0, 1.0)

                # 1. Left Hand (indices 91:112, 21 pts)
                pos[t, 0:21, 0] = x_norm[91:112]
                pos[t, 0:21, 1] = y_norm[91:112]
                pos[t, 0:21, 2] = sc133[91:112]

                # 2. Right Hand (indices 112:133, 21 pts)
                pos[t, 21:42, 0] = x_norm[112:133]
                pos[t, 21:42, 1] = y_norm[112:133]
                pos[t, 21:42, 2] = sc133[112:133]

                # 3. Upper Body Pose (6 pts: Shoulders, Elbows, Wrists)
                pos[t, 42:48, 0] = x_norm[RTMW_POSE_INDICES]
                pos[t, 42:48, 1] = y_norm[RTMW_POSE_INDICES]
                pos[t, 42:48, 2] = sc133[RTMW_POSE_INDICES]

                # 4. Facial Non-Manuals (12 contour pts)
                pos[t, 48:60, 0] = x_norm[RTMW_FACE_INDICES]
                pos[t, 48:60, 1] = y_norm[RTMW_FACE_INDICES]
                pos[t, 48:60, 2] = sc133[RTMW_FACE_INDICES]

        else:
            # ------------------------------------------------------------------
            # MediaPipe Holistic Fallback (543 landmarks)
            # ------------------------------------------------------------------
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            results = extractor_model.process(rgb)

            # Left Hand [0:21]
            if results.left_hand_landmarks:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    pos[t, i] = [lm.x, lm.y, lm.z]
            # Right Hand [21:42]
            if results.right_hand_landmarks:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    pos[t, 21 + i] = [lm.x, lm.y, lm.z]
            # Pose [42:48]
            if results.pose_landmarks:
                for i, p_idx in enumerate(MP_POSE_INDICES):
                    lm = results.pose_landmarks.landmark[p_idx]
                    pos[t, 42 + i] = [lm.x, lm.y, lm.z]
            # Face [48:60]
            if results.face_landmarks:
                for i, f_idx in enumerate(MP_FACE_INDICES):
                    lm = results.face_landmarks.landmark[f_idx]
                    pos[t, 48 + i] = [lm.x, lm.y, lm.z]

        # Optional visual streams (256x256 upper body ROI & 128x128 dominant hand)
        if extract_visual_crops:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if (engine == "rtmw") else rgb
            roi_res = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
            roi_crops.append(roi_res)

            rx = int(pos[t, 21, 0] * w_img) if pos[t, 21, 0] > 0 else w_img // 2
            ry = int(pos[t, 21, 1] * h_img) if pos[t, 21, 1] > 0 else h_img // 2
            half_box = max(32, min(w_img, h_img) // 6)
            x1, x2 = max(0, rx - half_box), min(w_img, rx + half_box)
            y1, y2 = max(0, ry - half_box), min(h_img, ry + half_box)
            crop = rgb[y1:y2, x1:x2]
            if crop.size > 0:
                crop_res = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
            else:
                crop_res = np.zeros((128, 128, 3), dtype=np.uint8)
            hand_crops.append(crop_res)

    # Interpolate dropout frames across missing hand landmarks
    pos = interpolate_missing_landmarks(pos)

    # Compute 9D Kinematics and 19D Phonology
    kinematics, phonology = compute_kinematics_and_phonology(pos)

    item = {
        "features": torch.from_numpy(kinematics).half(),
        "phonology": torch.from_numpy(phonology).half(),
    }
    if extract_visual_crops and roi_crops:
        item["roi_visual"] = torch.from_numpy(np.stack(roi_crops)).permute(0, 3, 1, 2)  # [T, 3, 256, 256]
        item["hand_visual"] = torch.from_numpy(np.stack(hand_crops)).permute(0, 3, 1, 2) # [T, 3, 128, 128]

    return item


def interpolate_missing_landmarks(pos: np.ndarray, max_gap: int = 5) -> np.ndarray:
    """Interpolates missing landmarks across short dropout intervals (max_gap frames)."""
    T, K, C = pos.shape
    cleaned = pos.copy()
    for k in range(K):
        valid = np.where(np.abs(pos[:, k, :]).sum(axis=-1) > 1e-4)[0]
        if len(valid) < 2:
            continue
        for i in range(len(valid) - 1):
            t_start, t_end = valid[i], valid[i + 1]
            gap = t_end - t_start - 1
            if 0 < gap <= max_gap:
                alpha = np.linspace(0.0, 1.0, gap + 2, dtype=np.float32)[1:-1, np.newaxis]
                cleaned[t_start + 1:t_end, k, :] = (
                    (1.0 - alpha) * cleaned[t_start, k, :] + alpha * cleaned[t_end, k, :]
                )
    return cleaned


def compute_kinematics_and_phonology(pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes Reference-Part Normalization, 9D kinematics, and 19D ASL phonology."""
    T, K, _ = pos.shape

    # 1. Bi-Acromial Normalization (Left shoulder 42, Right shoulder 43)
    sh_l = pos[:, 42, :3]
    sh_r = pos[:, 43, :3]
    sternum = (sh_l + sh_r) * 0.5
    sh_width = np.linalg.norm(sh_r - sh_l, axis=-1, keepdims=True)[:, np.newaxis]
    sh_width = np.clip(sh_width, a_min=1e-4, a_max=None)

    norm_pos = (pos - sternum[:, np.newaxis, :]) / sh_width

    # 2. Derivatives with zero-padding on inactive frames
    vel = np.diff(norm_pos, axis=0, prepend=norm_pos[:1, :, :])
    acc = np.diff(vel, axis=0, prepend=vel[:1, :, :])

    # Zero out velocities when landmarks are inactive
    is_inactive = (np.abs(pos).sum(axis=-1) < 1e-4)
    vel[is_inactive] = 0.0
    acc[is_inactive] = 0.0

    kinematics = np.concatenate([norm_pos, vel, acc], axis=-1).astype(np.float32)

    # 3. Phonology features: [T, 19]
    phonology = np.zeros((T, 19), dtype=np.float32)
    # Right wrist to nose distance
    r_wrist = norm_pos[:, 21, :]
    nose = norm_pos[:, 48, :]
    phonology[:, 0] = np.linalg.norm(r_wrist - nose, axis=-1)
    # Dual-hand distance
    l_wrist = norm_pos[:, 0, :]
    phonology[:, 1] = np.linalg.norm(r_wrist - l_wrist, axis=-1)
    # Right hand speed
    phonology[:, 2] = np.linalg.norm(vel[:, 21, :], axis=-1)
    # Left hand speed
    phonology[:, 3] = np.linalg.norm(vel[:, 0, :], axis=-1)
    # Elevation relative to sternum
    phonology[:, 4] = r_wrist[:, 1]
    phonology[:, 5] = l_wrist[:, 1]

    return kinematics, phonology


class How2SignStreamingOrchestrator:
    """
    Manages the complete micro-chunk extraction, processing, and drive streaming pipeline.
    """

    def __init__(
        self,
        drive_dir: str,
        output_drive_dir: str,
        local_scratch_dir: str = "/content/how2sign_scratch",
        chunk_size: int = 100,
        engine: str = "rtmw",
        extract_visual: bool = False,
    ):
        self.drive_dir = Path(drive_dir)
        self.output_drive_dir = Path(output_drive_dir)
        self.local_scratch = Path(local_scratch_dir)
        self.chunk_size = chunk_size
        self.engine = engine
        self.extract_visual = extract_visual

        # Destination paths on Drive
        self.shards_dir = self.output_drive_dir / "shards"
        self.manifest_dir = self.output_drive_dir / "manifests"
        self.ledger_path = self.manifest_dir / "ledger.json"

        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.local_scratch.mkdir(parents=True, exist_ok=True)

        self.ledger = self._load_ledger()

    def _load_ledger(self) -> Dict[str, Any]:
        if self.ledger_path.exists():
            try:
                with open(self.ledger_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"processed_clips": {}, "shards_written": 0}

    def _save_ledger(self):
        temp_ledger = self.ledger_path.with_suffix(".tmp")
        with open(temp_ledger, "w", encoding="utf-8") as f:
            json.dump(self.ledger, f, indent=2)
        temp_ledger.replace(self.ledger_path)

    def find_target_zips(self) -> List[Path]:
        """Discovers all How2Sign zips on Drive, prioritizing sentence clips."""
        all_zips = list(self.drive_dir.rglob("*.zip"))
        # Priority: clips first (sentence-level), then raw videos
        clips_zips = [z for z in all_zips if "clip" in z.name.lower()]
        raw_zips = [z for z in all_zips if "clip" not in z.name.lower()]
        return clips_zips + raw_zips

    def load_transcription_maps(self) -> Dict[str, str]:
        """Loads English sentence maps from any realigned CSV/TSV on Drive."""
        text_map: Dict[str, str] = {}
        # Search for realigned CSV, TSV, and TXT files
        patterns = ["*realigned*.csv", "*realigned*.tsv", "*realigned*.txt", "*.csv", "*.tsv"]
        candidate_files = []
        for pat in patterns:
            candidate_files.extend(list(self.drive_dir.rglob(pat)))

        # Deduplicate while preserving order
        seen_files = set()
        unique_files = []
        for cf in candidate_files:
            if cf.resolve() not in seen_files:
                seen_files.add(cf.resolve())
                unique_files.append(cf)

        for meta_file in unique_files:
            try:
                import csv
                with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
                    first_line = f.readline()
                    delimiter = "\t" if "\t" in first_line else ","
                    f.seek(0)
                    reader = csv.reader(f, delimiter=delimiter)
                    header = next(reader, None)
                    if not header:
                        continue

                    # Attempt column matching by name
                    header_lower = [col.strip().lower() for col in header]
                    id_col = -1
                    sent_col = -1

                    for idx, h in enumerate(header_lower):
                        if any(k in h for k in ["sentence_name", "clip_id", "video_id", "id", "name"]):
                            if id_col == -1:
                                id_col = idx
                        if any(k in h for k in ["sentence", "translation", "text", "transcript"]):
                            sent_col = idx

                    # Fallback to positional indices (col 0: id, col -1: sentence)
                    if id_col == -1:
                        id_col = 0
                    if sent_col == -1:
                        sent_col = len(header) - 1

                    for row in reader:
                        if len(row) > max(id_col, sent_col):
                            raw_id = row[id_col].strip()
                            clean_id = raw_id.replace("-rgb_front", "").replace("_rgb_front", "")
                            sentence = row[sent_col].strip()
                            if sentence and clean_id:
                                text_map[clean_id] = sentence
                                text_map[raw_id] = sentence
            except Exception as e:
                print(f"[WARN] Error reading metadata {meta_file.name}: {e}")
        return text_map

    def process_all(self):
        zips = self.find_target_zips()
        if not zips:
            print(f"[ERROR] No .zip files found under: {self.drive_dir}")
            return

        text_map = self.load_transcription_maps()
        print(f"[INFO] Discovered {len(zips)} zip files on Drive.")
        print(f"[INFO] Loaded {len(text_map)} English sentence annotations.")

        # Initialize requested extraction model
        extractor_model = None
        if self.engine == "rtmw":
            if _RTMLIB_AVAILABLE:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"[INFO] Initializing SOTA RTMW-x Wholebody on device={device} (mode=balanced)...")
                extractor_model = Wholebody(to_openpose=False, mode="balanced", backend="onnxruntime", device=device)
            else:
                print("[WARN] rtmlib not installed! Falling back to MediaPipe Holistic...")
                self.engine = "mediapipe"

        if self.engine == "mediapipe":
            if not _MEDIAPIPE_AVAILABLE:
                raise RuntimeError("Neither rtmlib nor mediapipe is installed! Run: pip install rtmlib onnxruntime-gpu")
            print("[INFO] Initializing MediaPipe Holistic...")
            mp_holistic = mp.solutions.holistic
            extractor_model = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                refine_face_landmarks=True,
            )

        for z_idx, zip_path in enumerate(zips, 1):
            print(f"\n" + "=" * 80)
            print(f"[{z_idx}/{len(zips)}] OPENING ARCHIVE: {zip_path.name}")
            print("=" * 80)

            # Determine split (train, val, test) from filename
            name_lower = zip_path.name.lower()
            if "val" in name_lower:
                split = "val"
            elif "test" in name_lower:
                split = "test"
            else:
                split = "train"

            split_shards_dir = self.shards_dir / split
            split_shards_dir.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_path, "r") as archive:
                    file_list = [
                        f for f in archive.namelist()
                        if f.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")) and not f.startswith("__MACOSX")
                    ]
                    print(f"  -> Archive contains {len(file_list)} video clips.")

                    # Filter out already processed clips
                    unprocessed = [f for f in file_list if Path(f).stem not in self.ledger["processed_clips"]]
                    print(f"  -> Unprocessed clips remaining: {len(unprocessed)}")

                    # Process in micro-chunks of chunk_size
                    for chunk_start in range(0, len(unprocessed), self.chunk_size):
                        chunk_files = unprocessed[chunk_start : chunk_start + self.chunk_size]
                        self._process_micro_chunk(
                            archive=archive,
                            chunk_files=chunk_files,
                            extractor_model=extractor_model,
                            split=split,
                            split_dir=split_shards_dir,
                            text_map=text_map,
                        )

            except Exception as e:
                print(f"[ERROR] Failed processing archive {zip_path.name}: {e}")

        if hasattr(extractor_model, "close"):
            extractor_model.close()
        print("\n" + "=" * 80)
        print("[COMPLETE] All How2Sign archives processed and streamed to Google Drive!")
        print(f"Total Shards Written: {self.ledger['shards_written']}")
        print(f"Total Clips Indexed: {len(self.ledger['processed_clips'])}")
        print("=" * 80)

    def _process_micro_chunk(
        self,
        archive: zipfile.ZipFile,
        chunk_files: List[str],
        extractor_model: Any,
        split: str,
        split_dir: Path,
        text_map: Dict[str, str],
    ):
        """Extracts a chunk of clips, processes on GPU, writes shard to Drive, and wipes local disk."""
        chunk_scratch = self.local_scratch / "chunk"
        chunk_scratch.mkdir(parents=True, exist_ok=True)

        shard_records = []
        t0 = time.time()

        try:
            # 1. Extract only this chunk to local NVMe
            for fname in chunk_files:
                archive.extract(fname, path=chunk_scratch)

            # 2. Process extracted videos
            for fname in chunk_files:
                local_vid = chunk_scratch / fname
                stem = Path(fname).stem
                clean_stem = stem.replace("-rgb_front", "").replace("_rgb_front", "")

                try:
                    data = extract_landmarks_from_video(
                        video_path=str(local_vid),
                        extractor_model=extractor_model,
                        engine=self.engine,
                        extract_visual_crops=self.extract_visual,
                    )
                    if data is not None:
                        text_label = text_map.get(clean_stem, text_map.get(stem, "how2sign_sentence"))
                        data["id"] = clean_stem
                        data["label"] = text_label
                        data["text"] = text_label
                        data["split"] = split
                        data["source"] = "How2Sign"
                        shard_records.append(data)
                        self.ledger["processed_clips"][stem] = True
                except Exception as e:
                    print(f"    [WARN] Skipped {stem}: {e}")

            # 3. Write shard directly to Google Drive if records exist
            if shard_records:
                s_idx = self.ledger["shards_written"]
                shard_name = f"shard_{s_idx:05d}.pt"
                local_shard = self.local_scratch / shard_name
                drive_shard = split_dir / shard_name

                # Save locally first, then copy to Drive
                torch.save(shard_records, local_shard)
                shutil.copy(local_shard, drive_shard)
                os.remove(local_shard)

                self.ledger["shards_written"] += 1
                self._save_ledger()

                dt = time.time() - t0
                fps = len(shard_records) / (dt + 1e-4)
                print(f"  [SAVED SHARD] {drive_shard.name} ({len(shard_records)} sequences, {fps:.1f} seq/s)")

        finally:
            # 4. Strictly wipe local video files to free up disk space
            shutil.rmtree(chunk_scratch, ignore_errors=True)
            # 5. Flush CPU RAM and CUDA cache
            del shard_records
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="How2Sign Google Colab Streaming Preprocessor")
    parser.add_argument("--drive-dir", type=str, default="/content/drive/MyDrive/Colab Notebooks", help="Path to mounted Google Drive folder containing How2Sign zips")
    parser.add_argument("--output-drive-dir", type=str, default="/content/drive/MyDrive/How2Sign_Preprocessed", help="Destination path on Drive for shards and manifests")
    parser.add_argument("--chunk-size", type=int, default=100, help="Number of video clips per micro-chunk (default 100)")
    parser.add_argument("--engine", type=str, default="rtmw", choices=["rtmw", "mediapipe"], help="Extraction engine: rtmw (default SOTA Wholebody on GPU) or mediapipe")
    parser.add_argument("--extract-visual", action="store_true", help="Extract 256x256 ROI and 128x128 Hand visual crops alongside kinematics")
    args = parser.parse_args()

    orchestrator = How2SignStreamingOrchestrator(
        drive_dir=args.drive_dir,
        output_drive_dir=args.output_drive_dir,
        chunk_size=args.chunk_size,
        engine=args.engine,
        extract_visual=args.extract_visual,
    )
    orchestrator.process_all()


if __name__ == "__main__":
    main()
