# %%writefile main.py
# ==============================================================================
# 0. INITIAL LOGGING & WARNING SUPPRESSION
# ==============================================================================
import os
import logging
import warnings

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OPENCV_FFMPEG_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_stderrthreshold"] = "3"

warnings.filterwarnings("ignore")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
try:
    from absl import logging as absl_logging

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

from tqdm.auto import tqdm
import time
import gc
import ctypes
import sys
import json
import math
import re
import tarfile
import argparse
import atexit
import threading
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import Value as MPValue, get_context
import numpy as np
import pandas as pd
Dataset = object
torch = None


def _init_torch():
    global torch, Dataset
    if torch is None:
        import torch as _torch
        from torch.utils.data import Dataset as _Dataset

        torch = _torch
        Dataset = _Dataset
        try:
            OnlineASLDataset.__bases__ = (Dataset,)
            FusedASLDataset.__bases__ = (Dataset,)
        except Exception:
            pass
from pathlib import Path
from scipy.interpolate import interp1d, CubicSpline
cv2 = None
mp = None


def _init_mediapipe():
    global mp, cv2
    if mp is None:
        import cv2 as _cv2
        import mediapipe as _mp

        cv2 = _cv2
        mp = _mp
        # Monkeypatch MediaPipe landmarker destructors to swallow shutdown errors safely
        try:

            def _safe_landmarker_del(self):
                try:
                    if hasattr(self, "close"):
                        self.close()
                except Exception:
                    pass

            mp.tasks.vision.HandLandmarker.__del__ = _safe_landmarker_del
            mp.tasks.vision.PoseLandmarker.__del__ = _safe_landmarker_del
            mp.tasks.vision.FaceLandmarker.__del__ = _safe_landmarker_del
        except Exception:
            pass

from collections import defaultdict
from datetime import datetime, timezone


def log_msg(msg: str):
    """Prints ISO/UTC timestamped debug messages [HH:MM:SS.mmm] to monitor dataset latency."""
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now_str}] {msg}")


def natural_sort_key(s):
    """Natural sorting key function for numerical filenames like frame_2.jpg vs frame_10.jpg."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def _force_gc(label: str = "") -> None:
    """Force Python GC cycles and ask the OS to reclaim freed C/C++ heap pages.

    MediaPipe's C++ runtime allocates large internal buffers (landmark tensors,
    GPU textures) via glibc malloc.  Python's GC only frees Python objects;
    freed C++ memory stays in the glibc malloc arena until explicitly released.
    Calling malloc_trim(0) after tearing down each dataset's MediaPipe pool
    returns those pages to the OS, preventing RSS from climbing monotonically
    across a multi-hour preprocessing run.
    """
    gc.collect()
    gc.collect()
    gc.collect()
    try:
        # glibc only (Linux). Harmless no-op on Windows / macOS.
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    if label:
        log_msg(f"[GC] gc forced after: {label}")


class DatasetProfiler:
    """Lightweight built-in profiler tracking top CPU heavy component tasks per dataset."""

    def __init__(self, dataset_name: str, split: str, num_workers: int = 1):
        self.dataset_name = dataset_name
        self.split = split
        self.num_workers = max(1, int(num_workers))
        self.t_start = time.perf_counter()
        self.timings = defaultdict(float)

    def add_timing(self, category: str, duration_sec: float):
        self.timings[category] += float(duration_sec)

    def ingest_task_result(self, record, discard):
        item_timings = None
        if record is not None and "timings" in record:
            item_timings = record["timings"]
        elif discard is not None:
            meta = discard.get("meta") or {}
            item_timings = meta.get("timings") or discard.get("timings")

        if item_timings:
            for cat, duration in item_timings.items():
                self.add_timing(cat, duration)

    def print_top5(self):
        t_wall = time.perf_counter() - self.t_start
        main_tasks_time = sum(
            v for k, v in self.timings.items()
            if k in ("fast_find_image_files", "metadata_parsing")
        )
        worker_wall_time = max(0.0, t_wall - main_tasks_time)
        worker_cpu_sum = sum(
            v for k, v in self.timings.items()
            if k not in ("fast_find_image_files", "metadata_parsing")
        )
        expected_worker_cpu = worker_wall_time * self.num_workers
        overhead = max(0.0, expected_worker_cpu - worker_cpu_sum)
        if overhead > 0.001:
            self.timings["pool_scheduling"] += overhead

        total_cpu = sum(self.timings.values())
        if total_cpu <= 0:
            return

        sorted_items = sorted(
            self.timings.items(), key=lambda x: x[1], reverse=True
        )[:5]
        log_msg(
            f"[+] [Profiler] Top 5 CPU Heavy Tasks for {self.dataset_name} ({self.split}):"
        )
        for i, (cat, d_sec) in enumerate(sorted_items, 1):
            pct = (d_sec / total_cpu) * 100.0
            print(f"    {i}:{cat} {pct:.0f}% ({d_sec:.1f}s)")


# ==============================================================================
# 1. FIXED KAGGLE PATH CONFIGURATIONS
# ==============================================================================


def resolve_dataset_dir(hardcoded_path: Path) -> Path:
    if hardcoded_path.exists():
        return hardcoded_path
    parts = hardcoded_path.parts
    if len(parts) >= 4:
        clean_parts = ["/kaggle/input"]
        start_idx = 3
        if len(parts) > 3 and parts[3] == "datasets":
            start_idx = 5
        if len(parts) > start_idx:
            clean_parts.extend(parts[start_idx:])
            direct_path = Path(*clean_parts)
            if direct_path.exists():
                return direct_path
            input_root = Path("/kaggle/input")
            if input_root.exists():
                try:
                    target_folder_name = parts[start_idx].lower()
                    for entry in input_root.iterdir():
                        e_name = entry.name.lower()
                        if entry.is_dir() and (
                            e_name == target_folder_name
                            or e_name.replace("-", "_")
                            == target_folder_name.replace("-", "_")
                        ):
                            resolved = entry
                            for sub in parts[start_idx + 1 :]:
                                resolved = resolved / sub
                            return resolved
                except Exception:
                    pass
    return hardcoded_path


ALPHABET_DIR = resolve_dataset_dir(
    Path("/kaggle/input/datasets/grassknoted/asl-alphabet")
)
ASL_CITIZEN_DIR = resolve_dataset_dir(
    Path("/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen")
)
HOW2SIGN_DIR = resolve_dataset_dir(
    Path(
        "/kaggle/input/datasets/psewmuthu/how2sign-holistic/how2sign_holistic_features"
    )
)
CHICAGO_FSWILD_DIR = resolve_dataset_dir(
    Path("/kaggle/input/datasets/joebeachcapital/chicagofswild")
)
NUMBER_DIR = resolve_dataset_dir(
    Path("/kaggle/input/datasets/lexset/synthetic-asl-numbers")
)
WLASL_DIR = resolve_dataset_dir(
    Path("/kaggle/input/datasets/risangbaskoro/wlasl-processed")
)
ASLEX_DIR = resolve_dataset_dir(Path("/kaggle/input/datasets/tranquocbao2012/asl-lex"))
ASLEX_SIGNDATA = ASLEX_DIR / "signdata.csv"

KAGGLE_OUTPUT_DIR = Path("/kaggle/working/asl_preprocessed_phase1")
KAGGLE_TEMP_DIR = Path("/kaggle/temp")

# Max workers for CPU-bound preprocessing (adjust based on CPU cores)
MAX_WORKERS = 4
# On Kaggle's headless P100 environment, EGL is unavailable and MediaPipe's GPU
# delegate always fails. Default to CPU so we skip the wasted GPU attempt.
# Override by setting MEDIAPIPE_USE_GPU=1 in the environment.
_NUM_AVAILABLE_GPUS = -1

def get_num_gpus() -> int:
    global _NUM_AVAILABLE_GPUS
    if _NUM_AVAILABLE_GPUS == -1:
        env_val = os.environ.get("NUM_AVAILABLE_GPUS")
        if env_val is not None:
            _NUM_AVAILABLE_GPUS = int(env_val)
        else:
            try:
                import torch
                _NUM_AVAILABLE_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except Exception:
                _NUM_AVAILABLE_GPUS = 0
            os.environ["NUM_AVAILABLE_GPUS"] = str(_NUM_AVAILABLE_GPUS)
    return _NUM_AVAILABLE_GPUS

_IS_KAGGLE = os.path.exists("/kaggle/working") and not os.path.exists("/workspace")
MEDIAPIPE_USE_GPU = os.environ.get(
    "MEDIAPIPE_USE_GPU", "1" if get_num_gpus() > 0 else ("0" if _IS_KAGGLE else "1")
).lower() not in ("0", "false", "no")

cpu_threads = os.cpu_count() or 8
if get_num_gpus() > 0:
    # Scale up workers when abundant CPU cores (e.g. 32 cores) are available to keep
    # the GPU fed, while staying within driver EGL context stability bounds.
    _workers_per_gpu = 6 if cpu_threads >= 32 else 4
    DEFAULT_GPU_WORKERS = min(12, max(2, get_num_gpus() * _workers_per_gpu))
else:
    DEFAULT_GPU_WORKERS = min(cpu_threads, max(4, cpu_threads - 2))

NUM_MP_GPU_WORKERS = int(os.environ.get("NUM_MP_GPU_WORKERS", str(DEFAULT_GPU_WORKERS)))

MEDIAPIPE_MODEL_DIR = Path(
    os.environ.get("MEDIAPIPE_MODEL_DIR", str(KAGGLE_TEMP_DIR / "mediapipe_models"))
)

# Quality management — global fallback; per-dataset thresholds below take precedence.
QUALITY_THRESHOLD = 0.60
QUALITY_LOG_DIRNAME = "quality_logs"
QUALITY_EPS = 1e-6

# Per-dataset quality thresholds, calibrated to each corpus's noise floor.
# Higher = stricter = fewer but cleaner samples fed to the transformer.
DATASET_QUALITY_THRESHOLDS: dict = {
    "ASL_Alphabet": 0.33,  # Static images
    "WLASL_v0.3": 0.20,  # Calibrated for web video noise & single-hand signs.
    "ChicagoFSWild": 0.22,  # Calibrated for fingerspelling in the wild.
    "ASL_Citizen": 0.20,  # Calibrated for crowdsourced webcams.
    "How2Sign_Holistic": 0.25,  # Pre-extracted holistic features.
    "Synthetic_Numbers": 0.35,
}

# Splits
SPLITS = ["train", "val", "test"]
SPLIT_ALIASES = {
    "ChicagoFSWild": {
        "val": "dev",
        "train": "train",
        "test": "test",
    },
    "ASL_Citizen": {
        "train": "train",
        "val": "val",
        "test": "test",
    },
    "WLASL_v0.3": {
        "train": "train",
        "val": "val",
        "test": "test",
    },
    "How2Sign_Holistic": {
        "train": "train",
        "val": "val",
        "test": "test",
    },
    "ASL_Alphabet": {
        "train": "train",
        "val": "val",
        "test": "test",
    },
    "Synthetic_Numbers": {
        "train": "train",
        "val": "val",
        "test": "test",
    },
}


def resolve_split(dataset_name, split):
    """Maps standard splits (train, val, test) to dataset-specific folder/file names."""
    return SPLIT_ALIASES.get(dataset_name, {}).get(split, split)


_DIR_FILE_CACHE = {}


def fast_find_image_files(
    root_dirs: list,
    valid_extensions=(".jpg", ".jpeg", ".png"),
    max_needed: int | None = None,
):
    """Blazing fast C-level scandir directory traversal with zero-pathlib inner loop overhead and memoization."""
    valid_set = {ext.lower() for ext in valid_extensions}
    existing = [p for p in root_dirs if p.exists()]
    if not existing:
        return []

    if max_needed is not None and max_needed > 0:
        found_quick = []
        for root_dir in existing:
            root_str = str(root_dir)
            stack = [root_str]
            while stack and len(found_quick) < max_needed:
                curr_dir = stack.pop()
                try:
                    with os.scandir(curr_dir) as it:
                        for entry in it:
                            if entry.is_file(follow_symlinks=False):
                                name = entry.name
                                dot_idx = name.rfind(".")
                                if dot_idx != -1 and name[dot_idx:].lower() in valid_set:
                                    found_quick.append((Path(entry.path), root_dir))
                                    if len(found_quick) >= max_needed:
                                        break
                            elif entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                except OSError:
                    pass
        return found_quick

    cache_key = tuple(sorted(str(p) for p in existing))
    if cache_key in _DIR_FILE_CACHE:
        return _DIR_FILE_CACHE[cache_key]

    found = []
    seen = set()
    for root_dir in existing:
        root_str = str(root_dir)
        stack = [root_str]
        while stack:
            curr_dir = stack.pop()
            try:
                with os.scandir(curr_dir) as it:
                    for entry in it:
                        if entry.is_file(follow_symlinks=False):
                            name = entry.name
                            dot_idx = name.rfind(".")
                            if dot_idx != -1 and name[dot_idx:].lower() in valid_set:
                                p_str = entry.path
                                if p_str not in seen:
                                    seen.add(p_str)
                                    found.append((Path(p_str), root_dir))
                        elif entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
            except OSError:
                pass

    _DIR_FILE_CACHE[cache_key] = found
    return found


import hashlib


def get_static_split_assignment(file_identifier: str) -> str:
    """Deterministically assigns a static image sample to train (80%), val (10%), or test (10%) based on MD5 hash.
    Guarantees zero dataset leakage between train, validation, and test splits.
    """
    h = int(hashlib.md5(file_identifier.encode("utf-8")).hexdigest(), 16)
    val_mod = h % 100
    if val_mod < 80:
        return "train"
    elif val_mod < 90:
        return "val"
    else:
        return "test"


def resolve_sequence_identity_swaps(raw_seq: np.ndarray) -> np.ndarray:
    """
    Frame-by-frame tracking identity swap resolution.
    Detects spatial trajectory discontinuities where MediaPipe accidentally swapped Left and Right hand labels
    mid-video, and swaps them back to maintain temporal hand continuity.
    """
    T, V, C = raw_seq.shape
    if T < 2 or V < 42:
        return raw_seq

    seq = raw_seq.copy()
    eps = 1e-5

    for t in range(T - 1):
        left_t = seq[t, 0:21, :3]
        right_t = seq[t, 21:42, :3]
        left_t1 = seq[t + 1, 0:21, :3]
        right_t1 = seq[t + 1, 21:42, :3]

        l_t_valid = np.any(np.abs(left_t[0]) > eps)
        r_t_valid = np.any(np.abs(right_t[0]) > eps)
        l_t1_valid = np.any(np.abs(left_t1[0]) > eps)
        r_t1_valid = np.any(np.abs(right_t1[0]) > eps)

        if not ((l_t_valid or r_t_valid) and (l_t1_valid or r_t1_valid)):
            continue

        normal_cost = 0.0
        swap_cost = 0.0

        if l_t_valid and l_t1_valid:
            normal_cost += np.linalg.norm(left_t[0] - left_t1[0])
        if r_t_valid and r_t1_valid:
            normal_cost += np.linalg.norm(right_t[0] - right_t1[0])

        if l_t_valid and r_t1_valid:
            swap_cost += np.linalg.norm(left_t[0] - right_t1[0])
        if r_t_valid and l_t1_valid:
            swap_cost += np.linalg.norm(right_t[0] - left_t1[0])

        if swap_cost < normal_cost * 0.60 and (normal_cost - swap_cost) > 0.05:
            tmp = seq[t + 1, 0:21, :].copy()
            seq[t + 1, 0:21, :] = seq[t + 1, 21:42, :]
            seq[t + 1, 21:42, :] = tmp

    return seq


# Preprocessing Constants
TARGET_FPS = 30  # Rate standardization target (T = 30)
TARGET_FRAMES = TARGET_FPS  # Backward-compatible alias used below
NUM_LANDMARKS = 60  # Active landmark subset (180 flat dimensions)
VOCAB_SIZE_TARGET = 2731

# Landmark Subsets (MediaPipe Tasks Vision layout — matches legacy Holistic ordering)
HAND_L_INDICES = list(range(0, 21))
HAND_R_INDICES = list(range(21, 42))
POSE_INDICES = [11, 12, 13, 14, 15, 16]
FACE_LIP_EYEBROW_INDICES = [0, 13, 17, 37, 267, 78, 308, 70, 107, 300, 336, 4]

# ==============================================================================
# 2. OPTIMIZED MATHEMATICAL SIGNAL ENGINES & HIGH-ACCURACY VISION FILTERS
# ==============================================================================


def enhance_frame_adaptive(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Sub-millisecond O(1) luminance check skips 80-85% of standard frames.
    Applies LAB CLAHE, Retinex log dynamic range compression, or Difference-of-Gaussians (DoG)
    high-pass edge maps exclusively to low-light or glare outlier frames.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    sample = (
        cv2.resize(frame_bgr, (64, 64), interpolation=cv2.INTER_NEAREST)
        if (h > 64 or w > 64)
        else frame_bgr
    )
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))

    # Fast path for clean frames (0% CPU cost)
    if 65.0 <= mean_val <= 185.0 and std_val > 25.0:
        return frame_bgr

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    if mean_val < 65.0:
        # Low-light recovery: Adaptive LAB CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_chan = clahe.apply(l_chan)
    elif mean_val > 185.0:
        # Glare / overexposure rescue: Logarithmic contrast compression
        l_float = l_chan.astype(np.float32)
        l_chan = np.uint8(np.log1p(l_float) * (255.0 / np.log1p(255.0)))

    if std_val <= 25.0:
        # Floating-point signed Difference of Gaussians (DoG) edge restoration
        l_f = l_chan.astype(np.float32)
        g1 = cv2.GaussianBlur(l_f, (3, 3), 0)
        g2 = cv2.GaussianBlur(l_f, (11, 11), 0)
        dog = g1 - g2
        l_chan = np.clip(l_f + 0.4 * dog, 0.0, 255.0).astype(np.uint8)

    enhanced_lab = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def normalize_riemannian_se3(normalized_seq: np.ndarray, V: int) -> np.ndarray:
    """
    SE(3) Riemannian Manifold Local Reference Alignment.
    Projects 3D spatial joint coordinates into the local torso-centric coordinate frame
    rendering coordinates 100% invariant to camera orientation and angle.
    Safely bypasses normalization if torso pose shoulders are missing/invalid.
    """
    if V < 44 or normalized_seq.shape[0] == 0:
        return normalized_seq

    seq = normalized_seq.copy()
    left_sh = seq[:, 42, :]
    right_sh = seq[:, 43, :]
    valid_left = np.any(np.abs(left_sh) > 1e-4, axis=-1)
    valid_right = np.any(np.abs(right_sh) > 1e-4, axis=-1)
    valid_both = valid_left & valid_right

    if not np.any(valid_both):
        return normalized_seq

    sh_vec = right_sh - left_sh
    sh_norm = np.linalg.norm(sh_vec, axis=-1, keepdims=True)

    valid_mask = valid_both & (sh_norm[:, 0] > 1e-3)
    if not np.any(valid_mask):
        return normalized_seq

    # Per-frame SO(3) alignment for frames with valid shoulders
    for t in np.where(valid_mask)[0]:
        u_x = sh_vec[t] / sh_norm[t]
        u_z = np.cross(u_x, np.array([0.0, 0.0, 1.0], dtype=np.float32))
        u_z_norm = np.linalg.norm(u_z) + 1e-6
        u_z = u_z / u_z_norm
        u_y = np.cross(u_z, u_x)

        R_t = np.stack([u_x, u_y, u_z], axis=-1)  # (3, 3)
        seq[t] = np.dot(seq[t], R_t)

    return seq


class CoordinateNormalizer:
    """
    Implements Phase 4 coordinate normalization + Riemannian SE(3) Gauge Normalization:
    - Handedness Invariance: Converts left-handed signers to right-handed topologies globally.
    - Translation Invariance: Origin centered on the dominant wrist.
    - Scale Invariance: Divided by dominant wrist-to-MCP distance.
    - Depth Normalization: Min-Max normalization of Z space.
    - SE(3) Dynamic Orientation Alignment: Torso-relative manifold alignment.
    """

    def __init__(self, target_frames=30, num_landmarks=60):
        self.target_frames = target_frames
        self.num_landmarks = num_landmarks

    def normalize(self, sequence):
        S, V, C = sequence.shape
        if S == 0:
            return np.zeros((1, self.num_landmarks, 3), dtype=np.float32)

        sequence = resolve_sequence_identity_swaps(sequence)
        sequence = sequence.copy()
        valid_mask = np.any(sequence != 0.0, axis=-1)  # Shape: (S, V)

        # --- GLOBAL HANDEDNESS SWAP & MIRROR ---
        l_hand = sequence[:, HAND_L_INDICES, :]
        r_hand = sequence[:, HAND_R_INDICES, :]
        l_valid = np.sum(np.any(l_hand != 0, axis=-1))
        r_valid = np.sum(np.any(r_hand != 0, axis=-1))

        if l_valid > r_valid:
            sequence[:, :, 0] = -sequence[:, :, 0]

            tmp_hand = sequence[:, HAND_L_INDICES, :].copy()
            sequence[:, HAND_L_INDICES, :] = sequence[:, HAND_R_INDICES, :]
            sequence[:, HAND_R_INDICES, :] = tmp_hand

            tmp_pose = sequence[:, [42, 44, 46], :].copy()
            sequence[:, [42, 44, 46], :] = sequence[:, [43, 45, 47], :]
            sequence[:, [43, 45, 47], :] = tmp_pose

            if V >= 60:
                face_l = [51, 53, 55, 56]
                face_r = [52, 54, 57, 58]
                tmp_face = sequence[:, face_l, :].copy()
                sequence[:, face_l, :] = sequence[:, face_r, :]
                sequence[:, face_r, :] = tmp_face

            tmp_mask = valid_mask[:, HAND_L_INDICES].copy()
            valid_mask[:, HAND_L_INDICES] = valid_mask[:, HAND_R_INDICES]
            valid_mask[:, HAND_R_INDICES] = tmp_mask

            tmp_pose_mask = valid_mask[:, [42, 44, 46]].copy()
            valid_mask[:, [42, 44, 46]] = valid_mask[:, [43, 45, 47]]
            valid_mask[:, [43, 45, 47]] = tmp_pose_mask

            if V >= 60:
                tmp_face_mask = valid_mask[:, face_l].copy()
                valid_mask[:, face_l] = valid_mask[:, face_r]
                valid_mask[:, face_r] = tmp_face_mask

        wrist_idx, mcp_idx = 21, 21 + 9

        r_wrist = sequence[:, 21, :]
        l_wrist = sequence[:, 0, :]
        r_valid = np.any(r_wrist != 0.0, axis=-1)
        l_valid_wrist = np.any(l_wrist != 0.0, axis=-1)

        origin = np.zeros_like(r_wrist)
        origin[r_valid] = r_wrist[r_valid]
        fallback_l = (~r_valid) & l_valid_wrist
        origin[fallback_l] = l_wrist[fallback_l]
        if V >= 44:
            shoulders = (sequence[:, 42, :] + sequence[:, 43, :]) * 0.5
            fallback_sh = (
                (~r_valid) & (~l_valid_wrist) & np.any(shoulders != 0.0, axis=-1)
            )
            origin[fallback_sh] = shoulders[fallback_sh]

        valid_origin_mask = r_valid | l_valid_wrist | (np.any(origin != 0.0, axis=-1))

        normalized_seq = sequence.copy()
        normalized_seq[valid_origin_mask] -= origin[valid_origin_mask, np.newaxis, :]

        # Sequence-Wide Rigid Scaling Calculation
        mcp_coords_norm = normalized_seq[:, mcp_idx, :]
        wrist_coords_norm = normalized_seq[:, wrist_idx, :]
        dists = np.linalg.norm(mcp_coords_norm - wrist_coords_norm, axis=-1)

        valid_dist_mask = (dists > 1e-5) & r_valid & valid_mask[:, mcp_idx]
        if np.any(valid_dist_mask):
            global_scale = np.median(dists[valid_dist_mask])
            if global_scale > 1e-5:
                normalized_seq /= global_scale
        elif V >= 44:
            sh_dists = np.linalg.norm(
                normalized_seq[:, 42, :] - normalized_seq[:, 43, :], axis=-1
            )
            valid_sh = sh_dists > 1e-5
            if np.any(valid_sh):
                sh_scale = np.median(sh_dists[valid_sh])
                if sh_scale > 1e-5:
                    normalized_seq /= sh_scale

        # Normalize Z-depth safely using ONLY valid spatial points
        z_coords = normalized_seq[:, :, 2]
        valid_z_elements = z_coords[valid_mask]
        if len(valid_z_elements) > 0:
            z_min, z_max = valid_z_elements.min(), valid_z_elements.max()
            if (z_max - z_min) > 1e-5:
                normalized_seq[:, :, 2] = (z_coords - z_min) / (z_max - z_min)

        normalized_seq[~valid_mask] = 0.0

        # Riemannian SE(3) dynamic orientation alignment
        normalized_seq = normalize_riemannian_se3(normalized_seq, V)

        return normalized_seq

    def apply_spatial_perturbations(self, sequence):
        sequence = np.asarray(sequence, dtype=np.float32)
        S, V, C = sequence.shape
        if S == 0:
            return sequence

        base_coords = sequence[:, :, :3]
        valid_mask = np.any(base_coords != 0.0, axis=-1)

        scale = np.random.uniform(0.85, 1.15)
        shear = np.random.uniform(-0.1, 0.1, size=(3, 3))
        np.fill_diagonal(shear, 1.0)
        trans = np.random.uniform(-0.1, 0.1, size=(3,))
        theta = np.random.uniform(-15 * (math.pi / 180), 15 * (math.pi / 180))
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        R = np.array(
            [[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32
        )

        augmented_seq = np.zeros_like(sequence, dtype=np.float32)

        if C == 9:
            for idx in range(0, 9, 3):
                sub_seq = sequence[:, :, idx : idx + 3]
                aug_sub = np.zeros_like(sub_seq)
                for t in range(S):
                    t_vector = trans if idx == 0 else 0.0
                    aug_sub[t] = (
                        np.dot(np.dot(sub_seq[t], R.T) * scale, shear.T) + t_vector
                    )
                augmented_seq[:, :, idx : idx + 3] = aug_sub
        else:
            for t in range(S):
                augmented_seq[t] = (
                    np.dot(np.dot(sequence[t], R.T) * scale, shear.T) + trans
                )

        for c_idx in range(0, C, 3):
            augmented_seq[:, :, c_idx : c_idx + 3][~valid_mask] = 0.0

        dropout_mask = np.random.rand(S, V, 1) > 0.05
        final_seq = np.where(dropout_mask, augmented_seq, 0.0)
        final_seq[~valid_mask] = 0.0

        return final_seq.astype(np.float32)


def append_kinematic_features(sequence):
    """Computes frame-to-frame velocity and acceleration vectors safely masked against missing frames."""
    S, V, C = sequence.shape
    if S < 2:
        return np.concatenate(
            [sequence, np.zeros_like(sequence), np.zeros_like(sequence)], axis=-1
        )

    valid_mask = np.any(sequence != 0.0, axis=-1, keepdims=True)
    velocity = np.diff(sequence, axis=0, prepend=sequence[:1]) * valid_mask
    acceleration = np.diff(velocity, axis=0, prepend=velocity[:1]) * valid_mask
    return np.concatenate([sequence, velocity, acceleration], axis=-1)


from scipy.interpolate import interp1d


def resample_sequence_to_fps(sequence, source_fps, target_fps=30.0):
    if sequence is None:
        return None
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[0] == 0:
        return None

    n_in, v, c = sequence.shape
    source_fps = (
        float(source_fps) if source_fps and source_fps > 0 else float(target_fps)
    )
    if n_in == 1:
        return sequence.copy()

    duration = (n_in - 1) / source_fps
    n_out = max(1, int(round(duration * target_fps)) + 1)

    if n_out == n_in and abs(source_fps - target_fps) < 1e-6:
        return sequence.astype(np.float32, copy=False)

    src_t = np.linspace(0.0, duration, num=n_in, endpoint=True)
    dst_t = np.linspace(0.0, duration, num=n_out, endpoint=True)

    f = interp1d(src_t, sequence, axis=0, kind="linear", fill_value="extrapolate")
    return f(dst_t).astype(np.float32)


def temporal_resample(sequence, target_frames=TARGET_FRAMES, source_fps=TARGET_FPS):
    return resample_sequence_to_fps(
        sequence, source_fps=source_fps, target_fps=target_frames
    )


def impute_missing_landmarks(sequence):
    """Imputes missing landmarks using stable bounded linear interpolation to avoid spline overshoot."""
    S, V, C = sequence.shape
    if S < 4:
        return sequence

    joint_missing = np.all(sequence == 0.0, axis=-1)
    if not np.any(joint_missing):
        return sequence

    flat_seq = sequence.reshape(S, V * C)
    imputed_flat = flat_seq.copy()

    masks = np.repeat(joint_missing, C, axis=-1).T
    unique_masks, inverse_indices = np.unique(masks, axis=0, return_inverse=True)

    for i, mask in enumerate(unique_masks):
        if not np.any(mask):
            continue
        col_indices = np.where(inverse_indices == i)[0]
        valid_indices = np.where(~mask)[0]

        if len(valid_indices) == 0:
            continue
        for col in col_indices:
            imputed_flat[:, col] = np.interp(
                np.arange(S), valid_indices, flat_seq[valid_indices, col]
            )

    return imputed_flat.reshape(S, V, C)


# MediaPipe hand bone connectivity — (parent_idx, child_idx) within a 21-landmark hand.
_HAND_BONE_PAIRS = np.array(
    [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),  # thumb
        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),  # index
        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),  # middle
        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),  # ring
        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),  # pinky
    ],
    dtype=np.int32,
)


def impute_anatomical_ik_landmarks(sequence: np.ndarray) -> np.ndarray:
    """
    Skeletal Inverse Kinematics (IK) Graph Auto-Imputer.
    Reconstructs damaged/missing joint landmarks using rigid anatomical bone length ratios (L_bone +/- 15%)
    and directional kinematics along finger bone topology with dynamic parent propagation.
    """
    S, V, C = sequence.shape
    if S < 2 or V < 42:
        return impute_missing_landmarks(sequence)

    seq = impute_missing_landmarks(sequence).copy()
    eps = 1e-6

    for hand_offset in (0, 21):
        hand = seq[:, hand_offset : hand_offset + 21, :]
        wrist_valid = np.any(np.abs(hand[:, 0, :]) > eps, axis=-1)
        if not np.any(wrist_valid):
            continue

        p_idx = _HAND_BONE_PAIRS[:, 0]
        c_idx = _HAND_BONE_PAIRS[:, 1]
        valid_frames = hand[wrist_valid]
        bone_vecs = valid_frames[:, c_idx, :] - valid_frames[:, p_idx, :]
        bone_lengths = np.linalg.norm(bone_vecs, axis=-1)
        median_lens = np.median(bone_lengths, axis=0)
        median_lens = np.where(median_lens > eps, median_lens, eps)

        for t in range(S):
            if np.all(np.abs(seq[t, hand_offset]) < eps):
                continue
            for b_i, (p, c) in enumerate(_HAND_BONE_PAIRS):
                parent_pos = seq[t, hand_offset + p]
                child_pos = seq[t, hand_offset + c]
                vec = child_pos - parent_pos
                curr_len = np.linalg.norm(vec)
                target_len = median_lens[b_i]
                if curr_len < eps or abs(curr_len - target_len) / target_len > 0.35:
                    unit_vec = (
                        vec / (curr_len + eps)
                        if curr_len > eps
                        else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                    )
                    seq[t, hand_offset + c] = parent_pos + unit_vec * target_len

    return seq


# ==============================================================================
# 2.9 MEDIAPIPE TASKS VISION (0.10.30+ — replaces legacy Holistic API)
# ==============================================================================
from scipy.signal import savgol_filter

_POSE_IDX = np.asarray(POSE_INDICES, dtype=np.int32)
_FACE_IDX = np.asarray(FACE_LIP_EYEBROW_INDICES, dtype=np.int32)

_MEDIAPIPE_MODEL_URLS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "pose_landmarker_heavy.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
    ),
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    ),
}
_mediapipe_model_paths: dict[str, Path] | None = None
_model_download_lock = threading.Lock()


def _ensure_mediapipe_models(model_dir: Path | None = None) -> dict[str, Path]:
    """Download .task model bundles once; thread-safe with lock and atomic rename."""
    global _mediapipe_model_paths
    # Fast path: already populated (no lock needed for read)
    if _mediapipe_model_paths is not None:
        return _mediapipe_model_paths
    with _model_download_lock:
        # Double-checked locking: re-check after acquiring
        if _mediapipe_model_paths is not None:
            return _mediapipe_model_paths
        model_dir = Path(model_dir or MEDIAPIPE_MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for filename, url in _MEDIAPIPE_MODEL_URLS.items():
            dest = model_dir / filename
            if not dest.exists() or dest.stat().st_size == 0:
                print(f"[*] Downloading MediaPipe model: {filename}")
                tmp = dest.with_suffix(".tmp")
                try:
                    urllib.request.urlretrieve(url, tmp)
                    tmp.replace(dest)
                except Exception as exc:
                    if tmp.exists():
                        try:
                            tmp.unlink()
                        except Exception:
                            pass
                    raise RuntimeError(
                        f"Failed to download MediaPipe model {filename} from {url}: {exc}"
                    )
            paths[filename] = dest
        _mediapipe_model_paths = paths
    return _mediapipe_model_paths


def smooth_mediapipe_sequence(sequence, window_length=5, polyorder=2):
    S, V, C = sequence.shape
    if S < 4:
        return sequence
    if window_length >= S:
        window_length = S if S % 2 != 0 else S - 1
        if window_length < 3:
            return sequence
    flat_seq = sequence.reshape(S, V * C)
    smoothed_flat = savgol_filter(
        flat_seq, window_length=window_length, polyorder=polyorder, axis=0
    )
    return smoothed_flat.reshape(S, V, C)


def _landmarks_to_xyz(landmarks) -> np.ndarray:
    """Unpack Task API NormalizedLandmark list into (N, 3) float32."""
    n = len(landmarks)
    out = np.empty((n, 3), dtype=np.float32)
    for i, lm in enumerate(landmarks):
        out[i, 0] = lm.x
        out[i, 1] = lm.y
        out[i, 2] = lm.z
    return out


def _indexed_landmarks(landmarks, indices: np.ndarray) -> np.ndarray:
    if not landmarks:
        return np.zeros((len(indices), 3), dtype=np.float32)
    return _landmarks_to_xyz(landmarks)[indices]


def _create_vision_task(
    landmarker_cls, options_cls, model_path: Path, running_mode, **extra
):
    """Create a Tasks Vision landmarker; try GPU delegate first, fall back to CPU."""
    delegates = [mp.tasks.BaseOptions.Delegate.GPU, mp.tasks.BaseOptions.Delegate.CPU]
    if not MEDIAPIPE_USE_GPU:
        delegates = [mp.tasks.BaseOptions.Delegate.CPU]
    last_exc = None

    devnull_fd = -1
    old_stderr = None
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull_fd, 2)
    except Exception:
        old_stderr = None

    try:
        for delegate in delegates:
            try:
                base_options = mp.tasks.BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=delegate,
                )
                options = options_cls(
                    base_options=base_options,
                    running_mode=running_mode,
                    **extra,
                )
                return landmarker_cls.create_from_options(options), delegate
            except Exception as exc:
                last_exc = exc
                if delegate == mp.tasks.BaseOptions.Delegate.CPU:
                    raise
    finally:
        if old_stderr is not None:
            try:
                os.dup2(old_stderr, 2)
                os.close(old_stderr)
            except Exception:
                pass
        if devnull_fd != -1 and devnull_fd != old_stderr:
            try:
                os.close(devnull_fd)
            except Exception:
                pass

    raise RuntimeError(f"Failed to create {landmarker_cls.__name__}: {last_exc}")


def resize_frame_to_max_dimension(frame, max_dim=720):
    """Resize input frame to max_dim while strictly preserving native image aspect ratio (no letterbox padding)."""
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if max(h, w) <= max_dim:
        return frame
    scale = max_dim / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (new_w, new_h), interpolation=interp)


_all_extractors = []
_extractors_lock = threading.Lock()


class MediaPipeExtractor:
    """Hand + Pose + Face landmark extraction via MediaPipe Tasks Vision API."""

    _ZERO_HAND = np.zeros((21, 3), dtype=np.float32)
    _ZERO_POSE = np.zeros((len(POSE_INDICES), 3), dtype=np.float32)
    _ZERO_FACE = np.zeros((len(FACE_LIP_EYEBROW_INDICES), 3), dtype=np.float32)

    def __init__(self, static_mode: bool = False):
        with _extractors_lock:
            _all_extractors.append(self)
        models = _ensure_mediapipe_models()
        running_mode = (
            mp.tasks.vision.RunningMode.IMAGE
            if static_mode
            else mp.tasks.vision.RunningMode.VIDEO
        )
        self._static_mode = static_mode
        self._closed = False
        self._rgb_buf: np.ndarray | None = None
        self._landmark_buf = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)
        self._clock_ms = 1
        self._last_ts = 0

        self._hand, _ = _create_vision_task(
            mp.tasks.vision.HandLandmarker,
            mp.tasks.vision.HandLandmarkerOptions,
            models["hand_landmarker.task"],
            running_mode,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._pose, _ = _create_vision_task(
            mp.tasks.vision.PoseLandmarker,
            mp.tasks.vision.PoseLandmarkerOptions,
            models["pose_landmarker_heavy.task"],
            running_mode,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._face, _ = _create_vision_task(
            mp.tasks.vision.FaceLandmarker,
            mp.tasks.vision.FaceLandmarkerOptions,
            models["face_landmarker.task"],
            running_mode,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

    def advance_video_clock(self, gap_ms: int = 1000):
        """Advance internal video clock to ensure clean monotonic progression between separate video files."""
        self._clock_ms = max(self._clock_ms + gap_ms, self._last_ts + gap_ms)

    def close(self):
        if self._closed:
            return
        self._closed = True
        with _extractors_lock:
            if self in _all_extractors:
                _all_extractors.remove(self)
        for attr in ("_hand", "_pose", "_face"):
            task = getattr(self, attr, None)
            if task is not None:
                try:
                    task.close()
                except Exception:
                    pass
                finally:
                    setattr(self, attr, None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _bgr_to_rgb(self, frame_bgr: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def _run_tasks(self, rgb: np.ndarray, timestamp_ms: int | None):
        # IMPORTANT: each model call gets its OWN mp.Image instance.
        #
        # With the GPU delegate, mp.Image wraps a GPU texture.  MediaPipe
        # tracks writes to that texture and raises a synchronisation error
        # (tensor.cc:404 "Tensors are designed for single writes") if the
        # SAME mp.Image object is passed to more than one model.  The GPU
        # driver then stalls to force re-synchronisation, causing 3-6× speed
        # drops that correlate exactly with the E0000 log lines.
        #
        # Creating a new wrapper for each model costs only a tiny Python
        # object allocation; the underlying RGB numpy buffer is NOT copied.
        if self._static_mode:
            hand_result = self._hand.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            pose_result = self._pose.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            if hand_result.hand_landmarks or pose_result.pose_landmarks:
                face_result = self._face.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )
            else:
                face_result = type("DummyFaceResult", (), {"face_landmarks": None})()
        else:
            rel_ts = 0 if timestamp_ms is None else int(timestamp_ms)
            ts = self._clock_ms + rel_ts
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            hand_result = self._hand.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )
            pose_result = self._pose.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )
            if hand_result.hand_landmarks or pose_result.pose_landmarks:
                face_result = self._face.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
                )
            else:
                face_result = type("DummyFaceResult", (), {"face_landmarks": None})()
        return hand_result, pose_result, face_result

    def _pack_hands(self, hand_result) -> tuple[np.ndarray, np.ndarray, float, float]:
        left = self._ZERO_HAND.copy()
        right = self._ZERO_HAND.copy()
        left_conf = right_conf = 0.0
        if not hand_result.hand_landmarks:
            return left, right, left_conf, right_conf
        for i, hand_lms in enumerate(hand_result.hand_landmarks):
            arr = _landmarks_to_xyz(hand_lms)
            score = 1.0
            if hand_result.handedness and i < len(hand_result.handedness):
                score = float(hand_result.handedness[i][0].score)
                label = hand_result.handedness[i][0].category_name
            else:
                label = "Left" if i == 0 else "Right"
            if label == "Left":
                left = arr
                left_conf = score
            else:
                right = arr
                right_conf = score
        return left, right, left_conf, right_conf

    def _pack_pose(self, pose_result) -> tuple[np.ndarray, float]:
        if not pose_result.pose_landmarks:
            return self._ZERO_POSE.copy(), 0.0
        pose_lms = pose_result.pose_landmarks[0]
        all_pose = _landmarks_to_xyz(pose_lms)
        block = all_pose[_POSE_IDX]
        vis_vals = [getattr(lm, "visibility", 0.0) for lm in pose_lms]
        pose_vis = float(np.mean(vis_vals)) if vis_vals else 0.0
        return block, pose_vis

    def _pack_face(self, face_result) -> np.ndarray:
        if not face_result.face_landmarks:
            return self._ZERO_FACE.copy()
        return _indexed_landmarks(face_result.face_landmarks[0], _FACE_IDX)

    def _frame_confidence(
        self, left_conf: float, right_conf: float, pose_vis: float
    ) -> dict:
        return {
            "pose_vis": pose_vis,
            "left_hand_conf": left_conf,
            "right_hand_conf": right_conf,
            "handedness_conf": float(np.mean([left_conf, right_conf])),
        }

    def extract_frame(
        self, frame, timestamp_ms: int | None = None, allow_cascade_retry: bool = True
    ):
        rgb = self._bgr_to_rgb(frame)
        hand_result, pose_result, face_result = self._run_tasks(rgb, timestamp_ms)

        left, right, left_conf, right_conf = self._pack_hands(hand_result)
        pose_block, pose_vis = self._pack_pose(pose_result)
        face_block = self._pack_face(face_result)

        buf = self._landmark_buf
        buf[0:21] = left
        buf[21:42] = right
        buf[42:48] = pose_block
        buf[48:60] = face_block

        confidence = self._frame_confidence(left_conf, right_conf, pose_vis)
        if left_conf > 0.1 and right_conf > 0.1:
            q_val = 0.40 * left_conf + 0.40 * right_conf + 0.20 * pose_vis
        else:
            q_val = 0.80 * max(left_conf, right_conf) + 0.20 * pose_vis
        quality = float(np.clip(q_val, 0.0, 1.0))

        # PASS 2: Opportunistic Cascade Retry using static mode extractor to protect video clock state
        if allow_cascade_retry and (
            quality < 0.35 or max(left_conf, right_conf) < 0.35
        ):
            enhanced_frame = enhance_frame_adaptive(frame)
            if enhanced_frame is not frame:
                try:
                    static_extractor = _get_process_extractor(static_mode=True)
                    buf_p2, q_p2, conf_p2 = static_extractor.extract_frame(
                        enhanced_frame, timestamp_ms=0, allow_cascade_retry=False
                    )
                    if q_p2 > quality:
                        return buf_p2, q_p2, conf_p2
                except Exception:
                    pass

        return buf.copy(), quality, confidence

    def extract_image(self, image_path, max_dim=320, enhance: bool = True):
        frame = cv2.imread(str(image_path))
        if frame is None:
            return None, 0.0, {}
        frame = resize_frame_to_max_dimension(frame, max_dim=max_dim)
        # Pre-enhance before inference so we only run the models once.
        # We never trigger the cascade retry here (that would call
        # _get_process_extractor(static_mode=True) which is *this* extractor,
        # doubling inference cost for every low-quality static image).
        if enhance:
            frame = enhance_frame_adaptive(frame)
        landmarks, quality, confidence = self.extract_frame(
            frame, timestamp_ms=0, allow_cascade_retry=False
        )
        return landmarks.reshape(1, NUM_LANDMARKS, 3), quality, confidence

    def extract_video(self, video_path, start_frame=None, end_frame=None, max_dim=320):
        """High-speed frame-selective video extraction at TARGET_FPS using cap.set() fast seeking and 320 resolution."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None, None, 0.0, {}
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not source_fps or source_fps <= 0:
            source_fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fast_mode_no_count = False
        if total_frames <= 0:
            total_frames = 100000
            fast_mode_no_count = True

        clip_start = int(start_frame or 0)
        clip_end = (
            int(end_frame)
            if (end_frame not in (None, -1) and not fast_mode_no_count)
            else total_frames - 1
        )
        clip_start = max(0, clip_start)

        if not fast_mode_no_count:
            clip_end = max(0, min(clip_end, total_frames - 1))
            if clip_start > clip_end:
                clip_start, clip_end = clip_end, clip_start

            clip_len = clip_end - clip_start + 1
            if clip_len > TARGET_FPS:
                target_indices = np.unique(
                    np.linspace(clip_start, clip_end, num=TARGET_FPS, dtype=int)
                )
            else:
                target_indices = np.arange(clip_start, clip_end + 1, dtype=np.int32)
        else:
            target_indices = None

        sequence = []
        qualities = []
        conf_acc = {
            "pose_vis": [],
            "left_hand_conf": [],
            "right_hand_conf": [],
            "handedness_conf": [],
        }

        signer_roi = None  # (roi_x0, roi_y0, roi_x1, roi_y1) in normalized coordinates

        if target_indices is not None:
            curr_frame_idx = 0
            for out_i, idx in enumerate(target_indices):
                if curr_frame_idx != idx:
                    diff = idx - curr_frame_idx
                    if 1 < diff <= 100:
                        while curr_frame_idx < idx:
                            if not cap.grab():
                                break
                            curr_frame_idx += 1
                    elif diff > 100:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        curr_frame_idx = idx
                success, frame = cap.read()
                if not success or frame is None:
                    break
                curr_frame_idx += 1

                fh, fw = frame.shape[:2]
                ts_ms = int(out_i * 1000 / TARGET_FPS)

                lm, q, conf = None, 0.0, {}
                # Attempt fast ROI extraction if signer bounding box is active
                if signer_roi is not None:
                    rx0, ry0, rx1, ry1 = signer_roi
                    px0, py0 = max(0, int(rx0 * fw)), max(0, int(ry0 * fh))
                    px1, py1 = min(fw, int(rx1 * fw)), min(fh, int(ry1 * fh))
                    if (px1 - px0) > 30 and (py1 - py0) > 30:
                        crop_frame = frame[py0:py1, px0:px1]
                        crop_resized = resize_frame_to_max_dimension(
                            crop_frame, max_dim=max_dim
                        )
                        c_lm, c_q, c_conf = self.extract_frame(
                            crop_resized, timestamp_ms=ts_ms
                        )
                        has_landmarks = (
                            c_conf.get("left_hand_conf", 0) > 0.05
                            or c_conf.get("right_hand_conf", 0) > 0.05
                            or c_conf.get("pose_vis", 0) > 0.3
                        )
                        if has_landmarks:
                            rw, rh = (px1 - px0) / fw, (py1 - py0) / fh
                            valid_m = np.any(c_lm != 0.0, axis=-1)
                            remapped_lm = c_lm.copy()
                            remapped_lm[valid_m, 0] = rx0 + c_lm[valid_m, 0] * rw
                            remapped_lm[valid_m, 1] = ry0 + c_lm[valid_m, 1] * rh
                            lm, q, conf = remapped_lm, c_q, c_conf

                if lm is None:
                    full_resized = resize_frame_to_max_dimension(frame, max_dim=max_dim)
                    lm, q, conf = self.extract_frame(full_resized, timestamp_ms=ts_ms)

                # Dynamically update signer ROI bounding envelope for subsequent frames
                valid_pts = lm[np.any(lm != 0.0, axis=-1)]
                if len(valid_pts) >= 5:
                    min_x, max_x = valid_pts[:, 0].min(), valid_pts[:, 0].max()
                    min_y, max_y = valid_pts[:, 1].min(), valid_pts[:, 1].max()
                    margin_x = max(0.12, 0.25 * (max_x - min_x))
                    margin_y = max(0.12, 0.25 * (max_y - min_y))
                    signer_roi = (
                        max(0.0, min_x - margin_x),
                        max(0.0, min_y - margin_y),
                        min(1.0, max_x + margin_x),
                        min(1.0, max_y + margin_y),
                    )

                sequence.append(lm)
                qualities.append(q)
                for k in conf_acc:
                    conf_acc[k].append(conf[k])
        else:
            if clip_start > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
            out_i = 0
            while True:
                success, frame = cap.read()
                if not success or frame is None:
                    break
                if max_dim is not None:
                    frame = resize_frame_to_max_dimension(frame, max_dim=max_dim)
                ts_ms = int(out_i * 1000 / TARGET_FPS)
                lm, q, conf = self.extract_frame(frame, timestamp_ms=ts_ms)
                sequence.append(lm)
                qualities.append(q)
                for k in conf_acc:
                    conf_acc[k].append(conf[k])
                out_i += 1
                if len(sequence) >= TARGET_FPS * 5:
                    break

        cap.release()
        self.advance_video_clock(gap_ms=1000)

        if not sequence:
            return None, source_fps, 0.0, {}
        seq_arr = np.array(sequence, dtype=np.float32)
        qual_arr = np.array(qualities, dtype=np.float32)
        agg_conf = {k: float(np.mean(conf_acc[k])) for k in conf_acc}
        return seq_arr, source_fps, float(np.mean(qual_arr)), agg_conf


# 2.5 ROBUST QUALITY SUB-SCORE HELPERS
# ==============================================================================


def _hand_valid_mask(hand_seq: np.ndarray, eps: float = QUALITY_EPS) -> np.ndarray:
    """(T,) bool: True if wrist AND at least 10 other landmarks are non-zero."""
    wrist_ok = np.any(np.abs(hand_seq[:, 0, :]) > eps, axis=-1)  # (T,)
    n_present = np.sum(np.any(np.abs(hand_seq) > eps, axis=-1), axis=-1)  # (T,)
    return wrist_ok & (n_present >= 10)


def compute_visibility_scores(seq: np.ndarray) -> dict:
    """
    Per-body-part visibility estimator — replaces the naive non-zero-coordinate check.

    For each hand the score aggregates:
      - Wrist-to-middle-MCP span (is the hand at a realistic scale?)
      - Landmark spatial spread (are points spread out, or degenerate?)
      - Unique landmark count  (>= 15 of 21 for full score)
      - Fraction of (x, y) landmarks inside image bounds [0, 1]

    Returns dict with keys: left, right, pose, lips — each in [0, 1].
    Must be called on the RAW (pre-normalization) sequence so coordinates
    are in image space where the [0,1] bound check is meaningful.
    """
    T, V, C = seq.shape
    eps = QUALITY_EPS

    def _score_hand(hand: np.ndarray, mcp_local: int = 9) -> float:
        valid = _hand_valid_mask(hand)
        if not np.any(valid):
            return 0.0
        h = hand[valid]  # (N, 21, 3)

        # 1. Wrist-to-middle-MCP span (calibrated for small/distant hands: >= 0.02 is 1.0)
        span = np.linalg.norm(h[:, mcp_local, :] - h[:, 0, :], axis=-1)
        span_score = float(np.clip(np.median(span) / 0.02, 0.0, 1.0))

        # 2. Landmark spread: std over all landmarks per valid frame (>= 0.008 is 1.0)
        spread = np.std(h.reshape(len(h), -1), axis=-1)
        spread_score = float(np.clip(np.mean(spread) / 0.008, 0.0, 1.0))

        # 3. Unique landmark count (max = 21)
        n_present = np.sum(np.any(np.abs(h) > eps, axis=-1), axis=-1)
        count_score = float(np.clip(np.mean(n_present) / 15.0, 0.0, 1.0))

        # 4. In-bounds fraction (x, y must be in [0, 1])
        inbounds = np.mean(
            (h[:, :, 0] >= 0.0)
            & (h[:, :, 0] <= 1.0)
            & (h[:, :, 1] >= 0.0)
            & (h[:, :, 1] <= 1.0)
        )
        return float(np.mean([span_score, spread_score, count_score, float(inbounds)]))

    left_score = _score_hand(seq[:, 0:21, :]) if V >= 21 else 0.0
    right_score = _score_hand(seq[:, 21:42, :]) if V >= 42 else 0.0

    pose_score = 0.0
    if V >= 48:
        present = np.any(np.abs(seq[:, 42:48, :]) > eps, axis=-1)  # (T, 6)
        pose_score = float(np.mean(present))

    lips_score = 0.0
    if V >= 60:
        present = np.any(np.abs(seq[:, 48:60, :]) > eps, axis=-1)  # (T, 12)
        lips_score = float(np.mean(present))

    return {
        "left": left_score,
        "right": right_score,
        "pose": pose_score,
        "lips": lips_score,
    }


def compute_anatomical_consistency(seq: np.ndarray) -> float:
    """
    Per-finger bone length consistency check.

    For each hand, computes every finger bone length each frame and compares
    against the per-bone sequence median.  Penalises frames where any bone
    deviates > 40% from its median (impossible stretching or shrinking).
    Returns score in [0, 1]: 1.0 = perfectly consistent anatomy.
    """
    T, V, C = seq.shape
    if T < 4 or V < 42:
        return 0.5

    eps = QUALITY_EPS
    scores = []
    for hand_start in (0, 21):
        hand = seq[:, hand_start : hand_start + 21, :]
        valid = _hand_valid_mask(hand)
        if np.sum(valid) < 4:
            continue
        h = hand[valid]  # (N, 21, 3)

        # Bone vectors and lengths: shape (N, num_bones)
        p_idx = _HAND_BONE_PAIRS[:, 0]
        c_idx = _HAND_BONE_PAIRS[:, 1]
        bone_vecs = h[:, c_idx, :] - h[:, p_idx, :]  # (N, B, 3)
        bone_lengths = np.linalg.norm(bone_vecs, axis=-1)  # (N, B)

        median_len = np.median(bone_lengths, axis=0)  # (B,)
        median_safe = np.where(median_len > eps, median_len, eps)

        deviation = (
            np.abs(bone_lengths - median_len[np.newaxis, :])
            / median_safe[np.newaxis, :]
        )
        violation_rate = float(np.mean(deviation > 0.40))
        scores.append(1.0 - violation_rate)

    return float(np.mean(scores)) if scores else 0.5


def compute_jerk_score(seq: np.ndarray) -> float:
    """
    Acceleration-based jitter detector (3rd-order derivative) computed over valid consecutive hand frames.
    """
    T, V, C = seq.shape
    if T < 5 or V < 42:
        return 0.5

    hands = seq[:, 0:42, :3].astype(np.float32)
    valid_mask = np.all(hands != 0.0, axis=(1, 2))  # shape (T,)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) < 5:
        return 0.5

    valid_hands = hands[valid_indices]
    velocity = np.diff(valid_hands, axis=0)
    acceleration = np.diff(velocity, axis=0)
    jerk = np.diff(acceleration, axis=0)

    rms_jerk = float(np.sqrt(np.mean(np.sum(jerk**2, axis=-1))))
    if math.isnan(rms_jerk) or math.isinf(rms_jerk):
        return 0.5
    return float(np.clip(np.exp(-30.0 * rms_jerk), 0.0, 1.0))


def compute_temporal_continuity(seq: np.ndarray) -> float:
    """
    Measures tracking stability over time.

    Rewards long continuous intervals where landmarks are visible and
    penalises frequent disappear/reappear transitions (tracker instability).
    Returns score in [0, 1].
    """
    T, V, C = seq.shape
    if T < 4:
        return 0.5

    eps = QUALITY_EPS
    key_lm = min(V, 48)  # both hands + pose
    present = np.any(np.abs(seq[:, :key_lm, :]) > eps, axis=(1, 2))  # (T,) bool

    if not np.any(present):
        return 0.0
    if np.all(present):
        return 1.0

    # Collect run lengths of consecutive valid frames
    run_lengths = []
    current = 0
    for p in present:
        if p:
            current += 1
        else:
            if current > 0:
                run_lengths.append(current)
            current = 0
    if current > 0:
        run_lengths.append(current)

    if not run_lengths:
        return 0.0

    mean_run = float(np.mean(run_lengths))
    n_gaps = len(run_lengths) - 1  # number of discontinuities

    continuity_score = float(np.clip(mean_run / T, 0.0, 1.0))
    transition_penalty = float(np.clip(n_gaps / max(T / 5.0, 1.0), 0.0, 1.0)) * 0.30
    return float(np.clip(continuity_score - transition_penalty, 0.0, 1.0))


def compute_identity_swap_score(raw_seq: np.ndarray) -> float:
    """
    Probabilistic left/right hand identity-swap detector.

    For each consecutive frame pair (t, t+1) where both wrists are visible:

        normal_cost = ||L_t - L_{t+1}|| + ||R_t - R_{t+1}||
        swap_cost   = ||L_t - R_{t+1}|| + ||R_t - L_{t+1}||

        swap_score_t = clip((normal_cost - swap_cost) / (normal_cost + eps), 0, 1)

    High swap_score_t means MediaPipe almost certainly swapped identities.
    The per-frame scores are averaged to give a sequence-level swap probability.

    MUST be called on the RAW (pre-normalization) sequence.  CoordinateNormalizer
    intentionally erases left/right identity information, making swap detection
    impossible after normalization.

    Returns mean swap probability in [0, 1]:
      0.0 = no swaps,  1.0 = constant identity swapping.
    """
    T, V, C = raw_seq.shape
    if T < 3 or V < 42:
        return 0.0

    eps = 1e-6
    left_wrist = raw_seq[:, 0, :]  # (T, 3)
    right_wrist = raw_seq[:, 21, :]  # (T, 3)

    swap_scores = []
    for t in range(T - 1):
        L_t = left_wrist[t]
        R_t = right_wrist[t]
        L_t1 = left_wrist[t + 1]
        R_t1 = right_wrist[t + 1]

        # Skip frames where either wrist is missing
        if (
            np.all(np.abs(L_t) < eps)
            or np.all(np.abs(R_t) < eps)
            or np.all(np.abs(L_t1) < eps)
            or np.all(np.abs(R_t1) < eps)
        ):
            continue

        normal_cost = np.linalg.norm(L_t - L_t1) + np.linalg.norm(R_t - R_t1)
        swap_cost = np.linalg.norm(L_t - R_t1) + np.linalg.norm(R_t - L_t1)

        score_t = float(
            np.clip((normal_cost - swap_cost) / (normal_cost + eps), 0.0, 1.0)
        )
        swap_scores.append(score_t)

    return float(np.mean(swap_scores)) if swap_scores else 0.0


def compute_occlusion_score(seq: np.ndarray) -> float:
    """
    Detects tracking failure masquerading as self-occlusion.

    For each frame, computes the median pairwise distance between all visible
    hand landmarks.  If this median falls below a collapse threshold (landmarks
    bunching into nearly the same position), the frame is flagged as degenerate.
    Genuine closed fists still spread landmarks over a small but non-zero area;
    tracker failure tends to collapse many landmarks onto a single point.

    Returns score in [0, 1]: 1.0 = no collapse detected.
    """
    T, V, C = seq.shape
    if T < 2 or V < 21:
        return 0.5

    eps = QUALITY_EPS
    COLLAPSE_THRESH = 0.003  # ~0.3% of image width in raw coords
    n_hands = min(V, 42)
    hands = seq[:, :n_hands, :3]

    collapse_flags = []
    for t in range(T):
        frame_lm = hands[t]  # (n_hands, 3)
        visible = np.any(np.abs(frame_lm) > eps, axis=-1)
        k = np.sum(visible)
        if k < 5:
            continue
        vis_lm = frame_lm[visible]  # (k, 3)
        diffs = vis_lm[:, None, :] - vis_lm[None, :, :]
        dists_sq = np.sum(diffs**2, axis=-1)
        iu = np.triu_indices(k, k=1)
        upper = np.sqrt(dists_sq[iu])
        if len(upper) == 0:
            continue
        collapse_flags.append(float(np.median(upper)) < COLLAPSE_THRESH)

    if not collapse_flags:
        return 0.5
    return float(1.0 - np.mean(collapse_flags))


# ==============================================================================
# 3.0  PROCESS-LOCAL EXTRACTOR + MODULE-LEVEL QUALITY & PIPELINE HELPERS
#      All symbols below are module-level so they are picklable and safe to
#      ship through ProcessPoolExecutor on any platform/start-method.
# ==============================================================================

# PID-keyed extractor cache: each worker holds one Tasks Vision bundle per mode.
_process_extractor_cache: dict = {}


def _suppress_worker_stderr():
    """Permanently redirect C++ stderr (fd 2) to /dev/null in this worker.

    glog and absl-log read their minimum-log-level configuration at library
    initialisation time.  When using ``fork``, the C++ MediaPipe layer is
    already initialised in the parent process before the workers are created,
    so environment variables like ``GLOG_minloglevel`` have no effect on the
    forked children.  The only reliable way to silence the C++ logging is to
    redirect the raw file descriptor.

    This is safe in workers because:
    - Worker results / exceptions travel through ``Future.result()``, not stderr.
    - Python's ``sys.stderr`` object is updated to point at /dev/null too so
      any Python-level write that goes through the high-level object also
      stays quiet.  Genuine tracebacks inside workers are wrapped and
      returned as error records by the ``try/except`` blocks in each _proc_*
      function.
    """
    try:
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)   # replace C++ fd 2
        os.close(devnull_fd)
    except Exception:
        pass
    try:
        import sys
        sys.stderr = open(os.devnull, "w")  # keep Python layer in sync
    except Exception:
        pass


def _ensure_egl_device_hook():
    """Ensure our EGL device enumeration interceptor exists so MediaPipe C++ respects ASSIGNED_GPU_ID."""
    if not sys.platform.startswith("linux"):
        return None
    hook_path = Path("/tmp/_egl_hook_v3.so")
    c_path = Path("/tmp/_egl_hook_v3.c")
    if hook_path.exists():
        return str(hook_path)
    try:
        c_code = """
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef void* EGLDeviceEXT;
typedef int (*eglQueryDevices_t)(int max_devices, EGLDeviceEXT *devices, int *num_devices);
typedef void* (*eglGetProcAddress_t)(const char *procname);

static eglQueryDevices_t real_eglQueryDevicesEXT = NULL;
static eglQueryDevices_t real_eglQueryDevicesKHR = NULL;
static eglGetProcAddress_t real_eglGetProcAddress = NULL;

static void init_real_funcs(void) {
    if (!real_eglQueryDevicesEXT) {
        real_eglQueryDevicesEXT = (eglQueryDevices_t)dlsym(RTLD_NEXT, "eglQueryDevicesEXT");
    }
    if (!real_eglQueryDevicesKHR) {
        real_eglQueryDevicesKHR = (eglQueryDevices_t)dlsym(RTLD_NEXT, "eglQueryDevicesKHR");
    }
    if (!real_eglGetProcAddress) {
        real_eglGetProcAddress = (eglGetProcAddress_t)dlsym(RTLD_NEXT, "eglGetProcAddress");
    }
}

static int query_devices_intercept(eglQueryDevices_t real_fn, int max_devices, EGLDeviceEXT *devices, int *num_devices) {
    if (!real_fn) return 0;

    char *assigned_gpu = getenv("ASSIGNED_GPU_ID");
    int target_id = -1;
    if (assigned_gpu) {
        target_id = atoi(assigned_gpu);
    } else {
        char *cuda_vis = getenv("CUDA_VISIBLE_DEVICES");
        if (cuda_vis) target_id = atoi(cuda_vis);
    }

    // Temporarily clear filtering environment variables so libEGL_nvidia enumerates
    // ALL physical EGL device handles (devices[0] = /dev/nvidia0, devices[1] = /dev/nvidia1).
    char *save_cuda = getenv("CUDA_VISIBLE_DEVICES") ? strdup(getenv("CUDA_VISIBLE_DEVICES")) : NULL;
    char *save_egl = getenv("EGL_VISIBLE_DEVICES") ? strdup(getenv("EGL_VISIBLE_DEVICES")) : NULL;
    char *save_nv = getenv("NVIDIA_VISIBLE_DEVICES") ? strdup(getenv("NVIDIA_VISIBLE_DEVICES")) : NULL;

    if (save_cuda) unsetenv("CUDA_VISIBLE_DEVICES");
    if (save_egl) unsetenv("EGL_VISIBLE_DEVICES");
    if (save_nv) unsetenv("NVIDIA_VISIBLE_DEVICES");

    int res = real_fn(max_devices, devices, num_devices);

    if (save_cuda) { setenv("CUDA_VISIBLE_DEVICES", save_cuda, 1); free(save_cuda); }
    if (save_egl) { setenv("EGL_VISIBLE_DEVICES", save_egl, 1); free(save_egl); }
    if (save_nv) { setenv("NVIDIA_VISIBLE_DEVICES", save_nv, 1); free(save_nv); }

    if (res && num_devices && *num_devices > 0 && devices && target_id >= 0) {
        if (target_id < *num_devices) {
            devices[0] = devices[target_id];
            *num_devices = 1;
        } else if (*num_devices == 1 && target_id > 0) {
            // If the driver still only returned 1 handle despite clearing env vars,
            // leave *num_devices=1 and devices[0] intact.
        }
    }
    return res;
}

int eglQueryDevicesEXT(int max_devices, EGLDeviceEXT *devices, int *num_devices) {
    init_real_funcs();
    return query_devices_intercept(real_eglQueryDevicesEXT, max_devices, devices, num_devices);
}

int eglQueryDevicesKHR(int max_devices, EGLDeviceEXT *devices, int *num_devices) {
    init_real_funcs();
    return query_devices_intercept(real_eglQueryDevicesKHR, max_devices, devices, num_devices);
}

void* eglGetProcAddress(const char *procname) {
    init_real_funcs();
    if (!procname) return NULL;
    if (strcmp(procname, "eglQueryDevicesEXT") == 0) {
        return (void*)eglQueryDevicesEXT;
    }
    if (strcmp(procname, "eglQueryDevicesKHR") == 0) {
        return (void*)eglQueryDevicesKHR;
    }
    if (real_eglGetProcAddress) {
        return real_eglGetProcAddress(procname);
    }
    return dlsym(RTLD_NEXT, procname);
}
"""
        c_path.write_text(c_code, encoding="utf-8")
        import subprocess
        subprocess.run(
            ["gcc", "-shared", "-fPIC", "-o", str(hook_path), str(c_path), "-ldl"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return str(hook_path)
    except Exception:
        return None


def _mp_pool_worker_init_gpu(gpu_id: int):
    """Worker initialiser pinned to a specific GPU.

    *gpu_id* is passed as an initarg so every worker in a pool gets an
    identical, deterministic assignment — no shared-counter races.
    Setting CUDA_VISIBLE_DEVICES here (before any EGL context is opened)
    makes NVIDIA's driver associate the new EGL display with that GPU.
    """
    _suppress_worker_stderr()

    assigned_gpu = os.environ.get("ASSIGNED_GPU_ID")
    if assigned_gpu is not None:
        gpu_id = int(assigned_gpu)

    hook_so = _ensure_egl_device_hook()
    if hook_so and os.path.exists(hook_so):
        try:
            ctypes.CDLL(hook_so, mode=ctypes.RTLD_GLOBAL)
            os.environ["LD_PRELOAD"] = hook_so + (":" + os.environ.get("LD_PRELOAD", "") if os.environ.get("LD_PRELOAD") else "")
        except Exception:
            pass

    # Do NOT set CUDA_VISIBLE_DEVICES, EGL_VISIBLE_DEVICES, or NVIDIA_VISIBLE_DEVICES here before
    # _init_mediapipe() runs! Setting them before library initialization restricts libEGL_nvidia to 1 device,
    # causing eglQueryDevicesEXT to return *num_devices=1 where devices[0] defaults to /dev/nvidia0.
    # Leaving them unset during _init_mediapipe() lets libEGL_nvidia see both physical GPUs (*num_devices=2),
    # allowing our LD_PRELOAD C hook (_egl_hook_v3.so) to read ASSIGNED_GPU_ID and swap devices[0] = devices[target_id].
    os.environ["DRI_PRIME"] = str(gpu_id)
    # DRM_DEVICE directs headless Mesa/DRM EGL to the exact render node (/dev/dri/renderD128 for GPU 0, renderD129 for GPU 1).
    os.environ["DRM_DEVICE"] = f"/dev/dri/renderD{128 + gpu_id}"
    # Isolate the child worker so any code calling get_num_gpus() inside the worker
    # sees exactly 1 GPU (its pinned device 0).
    os.environ["NUM_AVAILABLE_GPUS"] = "1"
    global _NUM_AVAILABLE_GPUS
    _NUM_AVAILABLE_GPUS = 1

    # Import CV2 / MediaPipe *before* setting CUDA_VISIBLE_DEVICES so our EGL interceptor
    # successfully selects ASSIGNED_GPU_ID from the full physical device list.
    _init_mediapipe()

    # Now that MediaPipe EGL is locked to ASSIGNED_GPU_ID (/dev/nvidia1 for GPU 1),
    # set CUDA_VISIBLE_DEVICES so any PyTorch or CUDA runtime calls are pinned to that GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    try:
        # Keep each worker single-threaded at the OpenCV/OpenMP/MKL layer.
        # MediaPipe already runs multi-threaded internal pipelines per image/video;
        # allowing 4-8 extra OpenCV/OpenMP threads per worker causes 250+ active OS
        # threads across dual-GPU sharded processes, hitting container thread limits
        # (Errno 11 Resource temporarily unavailable during fork_exec).
        cv2.setNumThreads(1)
        os.environ["OPENCV_FFMPEG_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    except Exception:
        pass

    # Pre-warm ONLY the static (IMAGE-mode) extractor.
    # Pre-warming both static AND video GPU delegates in the same process
    # shares MediaPipe's internal GPU tensor buffer pool, causing the
    # tensor.cc "multiple writes" synchronisation warning (E0000).  The
    # video extractor is lazily created on first use when video datasets
    # are processed, at which point it gets its own clean tensor pool.
    try:
        _get_process_extractor(static_mode=True)
    except Exception:
        pass


def _mp_pool_worker_init_cpu():
    """Worker initialiser for CPU-only machines (no GPU available)."""
    _suppress_worker_stderr()
    _init_mediapipe()
    try:
        cv2.setNumThreads(1)
        os.environ["OPENCV_FFMPEG_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
    except Exception:
        pass
    try:
        _get_process_extractor(static_mode=True)
    except Exception:
        pass


_SHARED_POOL = None
_POOL_LOCK = threading.Lock()

# Per-GPU pools: keyed by integer GPU index.
_GPU_POOLS: dict = {}
_GPU_POOL_LOCK = threading.Lock()


def _get_mp_context(gpu_isolated: bool = False):
    """Return the correct multiprocessing start-method context.

    GPU pools (gpu_isolated=True) — always 'spawn'
    -----------------------------------------------
    ``torch.cuda.is_available()`` (called from ``main()`` via ``_init_torch``)
    initialises the CUDA runtime in the *parent* process before any workers
    are created.  After ``fork()``, the child inherits an already-initialised
    CUDA context whose default device is GPU 0.  Setting
    ``CUDA_VISIBLE_DEVICES`` *inside* the forked worker has no effect because
    NVIDIA's driver selects the device at CUDA init time, not at env-var read
    time.

    ``spawn`` starts a *fresh* Python interpreter with zero CUDA state.  The
    initialiser then sets ``CUDA_VISIBLE_DEVICES=<gpu_id>`` **before** any
    CUDA/EGL code runs, so the driver sees only the intended GPU.

    Note: spawn children inherit the parent's environment variables.  The
    parent sets ``NUM_AVAILABLE_GPUS`` before any worker is created, so the
    worker's module-level ``get_num_gpus()`` reads the cached value instead
    of calling ``torch.cuda.device_count()`` again.  This avoids a second
    CUDA init in the worker before ``CUDA_VISIBLE_DEVICES`` takes effect.

    CPU pool (gpu_isolated=False) — 'fork' on Linux, 'spawn' elsewhere
    -------------------------------------------------------------------
    No CUDA context is used; fork is safe and much faster to start.
    """
    import platform as _platform
    if gpu_isolated:
        return get_context("spawn")   # CUDA isolation: must be spawn
    return get_context("fork" if _platform.system() == "Linux" else "spawn")


def get_or_create_gpu_pool(gpu_id: int) -> ProcessPoolExecutor:
    """Return (creating if needed) a ProcessPoolExecutor whose workers are
    all pinned to *gpu_id* via CUDA_VISIBLE_DEVICES.

    Uses 'spawn' (via gpu_isolated=True) so each worker starts with a clean
    CUDA runtime.  The initialiser sets CUDA_VISIBLE_DEVICES before importing
    MediaPipe, guaranteeing EGL device selection hits the right GPU.
    """
    with _GPU_POOL_LOCK:
        if gpu_id not in _GPU_POOLS:
            n_gpus = max(1, get_num_gpus())
            workers_per_gpu = max(1, NUM_MP_GPU_WORKERS // n_gpus)
            kwargs = {
                "max_workers": workers_per_gpu,
                "mp_context": _get_mp_context(gpu_isolated=True),  # spawn for CUDA isolation
                "initializer": _mp_pool_worker_init_gpu,
                "initargs": (gpu_id,),
            }
            if sys.version_info >= (3, 11):
                kwargs["max_tasks_per_child"] = 300
            _GPU_POOLS[gpu_id] = ProcessPoolExecutor(**kwargs)
        return _GPU_POOLS[gpu_id]


def reset_gpu_pools() -> "MultiGPUExecutorProxy":
    """Tear down all GPU-specific pools and return a fresh MultiGPUExecutorProxy."""
    with _GPU_POOL_LOCK:
        for pool in list(_GPU_POOLS.values()):
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        _GPU_POOLS.clear()
    return MultiGPUExecutorProxy()


def _close_local_mediapipe_extractors():
    """Close any extractor instances held by the current process/thread."""
    for attr in ("extractor_True", "extractor_False"):
        extractor = getattr(_thread_extractor_cache, attr, None)
        if extractor is not None:
            try:
                extractor.close()
            except Exception:
                pass
            finally:
                try:
                    delattr(_thread_extractor_cache, attr)
                except Exception:
                    pass
    with _extractors_lock:
        to_close = list(_all_extractors)
    for ext in to_close:
        try:
            ext.close()
        except Exception:
            pass


class SharedExecutorProxy:
    def __init__(self, executor: ProcessPoolExecutor):
        self._executor = executor

    def submit(self, fn, *args, **kwargs):
        return self._executor.submit(fn, *args, **kwargs)

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        return self._executor.map(fn, *iterables, timeout=timeout, chunksize=chunksize)

    def shutdown(self, wait=True, *, cancel_futures=False):
        # Deliberately no-op: the shared pool is closed once at program exit.
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def get_shared_pool() -> ProcessPoolExecutor:
    """CPU-only fallback pool (used when no GPUs are available)."""
    global _SHARED_POOL
    with _POOL_LOCK:
        if _SHARED_POOL is None:
            workers = max(1, int(NUM_MP_GPU_WORKERS))
            kwargs = {
                "max_workers": workers,
                "mp_context": _get_mp_context(),
                "initializer": _mp_pool_worker_init_cpu,
            }
            if sys.version_info >= (3, 11):
                kwargs["max_tasks_per_child"] = 300
            _SHARED_POOL = ProcessPoolExecutor(**kwargs)
        return _SHARED_POOL


def reset_shared_pool() -> ProcessPoolExecutor:
    global _SHARED_POOL
    with _POOL_LOCK:
        if _SHARED_POOL is not None:
            try:
                _SHARED_POOL.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            _SHARED_POOL = None
        return get_shared_pool()


def _reset_executor(executor):
    """Recreate whichever pool type backs *executor* after a submit failure."""
    if isinstance(executor, MultiGPUExecutorProxy):
        return reset_gpu_pools()
    return SharedExecutorProxy(reset_shared_pool())


def bounded_as_completed(executor, fn, tasks, max_in_flight=None):
    """Yields (future.result()) as tasks complete, maintaining at most max_in_flight in the queue."""
    if max_in_flight is None:
        max_workers = getattr(executor, "_max_workers", None) or getattr(
            getattr(executor, "_executor", None), "_max_workers", 4
        )
        # Maintain a deeper task queue buffer (3× workers) so CPU workers never sit idle
        # waiting for task dispatch latency between video files.
        max_in_flight = max(max_workers * 3, 16)

    task_iter = iter(tasks)
    futures = {}

    # Prime the queue
    for task in task_iter:
        try:
            fut = executor.submit(fn, *task)
        except Exception:
            executor = _reset_executor(executor)
            fut = executor.submit(fn, *task)
        futures[fut] = task
        if len(futures) >= max_in_flight:
            break

    from concurrent.futures import wait, FIRST_COMPLETED

    while futures:
        try:
            done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
        except Exception:
            done = [f for f in list(futures.keys()) if f.done()]

        for fut in done:
            try:
                res = fut.result()
            except Exception as e:
                res = (
                    None,
                    {
                        "reason": "exception",
                        "source": "worker",
                        "label": "unknown",
                        "quality": 0.0,
                        "meta": {"error": str(e)},
                    },
                )
            yield res
            if fut in futures:
                del futures[fut]

        for task in task_iter:
            try:
                fut = executor.submit(fn, *task)
            except Exception:
                executor = _reset_executor(executor)
                fut = executor.submit(fn, *task)
            futures[fut] = task
            if len(futures) >= max_in_flight:
                break


class MultiGPUExecutorProxy:
    """Dynamically distributes task submissions across one ProcessPoolExecutor
    per available GPU using least-busy load tracking (`in_flight_counts`).
    Falls back to the CPU shared pool when no GPUs exist.

    Each GPU's pool has its workers *pinned* to that GPU via initargs so
    CUDA_VISIBLE_DEVICES is set before any EGL context is created — the only
    reliable way to steer NVIDIA's driver to the right device on Linux.
    """

    def __init__(self):
        n_gpus = get_num_gpus()
        if n_gpus >= 1:
            self._pools = [get_or_create_gpu_pool(i) for i in range(n_gpus)]
        else:
            # CPU-only: delegate to the existing shared pool
            self._pools = [get_shared_pool()]
        self._n = len(self._pools)
        self._in_flight_counts = [0] * self._n
        self._rr = 0
        self._lock = threading.Lock()

    @property
    def _max_workers(self) -> int:
        """Total worker count across all pools (used by bounded_as_completed)."""
        return sum(getattr(p, "_max_workers", 1) for p in self._pools)

    def submit(self, fn, *args, **kwargs):
        with self._lock:
            # Dynamic least-busy allocation: pick the GPU pool holding the fewest pending/in-flight tasks.
            # If there's a tie, round-robin among tied candidates for even balancing.
            min_count = min(self._in_flight_counts)
            candidates = [i for i, count in enumerate(self._in_flight_counts) if count == min_count]
            pool_idx = candidates[self._rr % len(candidates)]
            self._rr += 1
            self._in_flight_counts[pool_idx] += 1
            pool = self._pools[pool_idx]

        fut = pool.submit(fn, *args, **kwargs)
        fut.add_done_callback(lambda f, idx=pool_idx: self._on_task_done(idx))
        return fut

    def _on_task_done(self, pool_idx: int):
        with self._lock:
            if self._in_flight_counts[pool_idx] > 0:
                self._in_flight_counts[pool_idx] -= 1

    def map(self, fn, *iterables, timeout=None, chunksize=1):
        # Dynamically distribute tasks across all GPU pools and merge results in order.
        futures = [self.submit(fn, *args) for args in zip(*iterables)]
        for fut in futures:
            yield fut.result(timeout=timeout)

    def shutdown(self, wait=True, *, cancel_futures=False):
        # Deliberately no-op: pools are torn down at exit via _shutdown_mediapipe_pool.
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _create_mediapipe_pool() -> MultiGPUExecutorProxy:
    """Return a proxy that distributes work across all available GPUs."""
    return MultiGPUExecutorProxy()


def _shutdown_mediapipe_pool():
    """Explicit cleanup at exit — tears down both per-GPU and shared CPU pools."""
    _close_local_mediapipe_extractors()
    with _GPU_POOL_LOCK:
        for pool in list(_GPU_POOLS.values()):
            try:
                pool.shutdown(wait=True, cancel_futures=False)
            except Exception:
                pass
        _GPU_POOLS.clear()
    global _SHARED_POOL
    with _POOL_LOCK:
        if _SHARED_POOL is not None:
            _SHARED_POOL.shutdown(wait=True, cancel_futures=False)
            _SHARED_POOL = None


atexit.register(_shutdown_mediapipe_pool)


def _release_mediapipe_worker_pool() -> None:
    """Tear down all MediaPipe worker pools and release their C++ heap.

    Called between datasets so that MediaPipe's internal C++ memory (landmark
    tensors, EGL contexts, GPU texture pools) is returned to the OS *before*
    the next dataset starts.  Without this, every dataset adds to a
    monotonically growing C++ heap that Python's GC cannot see or free.

    The pools are re-created lazily on the next _create_mediapipe_pool() call,
    so calling this function between datasets in build_split() is safe.
    """
    _shutdown_mediapipe_pool()
    _force_gc("mediapipe pool teardown")


_thread_extractor_cache = threading.local()


def _get_process_extractor(static_mode: bool = False) -> "MediaPipeExtractor":
    """Return a thread-local long-lived MediaPipeExtractor."""
    if mp is None:
        _init_mediapipe()
    key = f"extractor_{static_mode}"
    if not hasattr(_thread_extractor_cache, key):
        setattr(
            _thread_extractor_cache, key, MediaPipeExtractor(static_mode=static_mode)
        )
    return getattr(_thread_extractor_cache, key)


def _assess_quality(sequence, base_quality: float = 0.0, confidence=None):
    """Module-level, picklable quality scorer.
    Identical logic to FrankensteinDataProcessor.assess_quality — defined here
    so ProcessPoolExecutor workers can call it without pickling the processor.
    Adaptively handles cropped hand / single-modality inputs without artificial penalty.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[0] == 0:
        return 0.0, {"reason": "empty_sequence"}
    vis = compute_visibility_scores(seq)

    has_pose = vis["pose"] > 0.05
    has_lips = vis["lips"] > 0.05

    if vis["left"] > 0.1 and vis["right"] > 0.1:
        active_hands = (vis["left"] + vis["right"]) / 2.0
    else:
        active_hands = max(vis["left"], vis["right"])

    if has_pose and has_lips:
        geometry_score = float(
            0.80 * active_hands + 0.15 * vis["pose"] + 0.05 * vis["lips"]
        )
        vis_score = float(
            np.mean(
                [
                    vis["left"] if vis["left"] > 0.1 else active_hands,
                    vis["right"] if vis["right"] > 0.1 else active_hands,
                    vis["pose"],
                    vis["lips"],
                ]
            )
        )
    elif has_pose:
        geometry_score = float(0.85 * active_hands + 0.15 * vis["pose"])
        vis_score = float(np.mean([active_hands, vis["pose"]]))
    else:
        geometry_score = float(active_hands)
        vis_score = float(active_hands)

    temporal_score = compute_temporal_continuity(seq)
    if confidence:
        pose_vis = float(confidence.get("pose_vis", 0.0))
        left_conf = float(confidence.get("left_hand_conf", 0.0))
        right_conf = float(confidence.get("right_hand_conf", 0.0))
        if left_conf > 0.1 and right_conf > 0.1:
            hand_conf = (left_conf + right_conf) / 2.0
        else:
            hand_conf = max(left_conf, right_conf)

        if pose_vis > 0.05:
            detector_score = float(
                np.clip(0.70 * hand_conf + 0.30 * pose_vis, 0.0, 1.0)
            )
        else:
            detector_score = float(np.clip(hand_conf, 0.0, 1.0))
    else:
        pose_vis = 0.5
        hand_conf = active_hands
        detector_score = float(np.clip(active_hands, 0.0, 1.0))

    anatomy_score = compute_anatomical_consistency(seq)
    jerk_score = compute_jerk_score(seq)
    occlusion_score = compute_occlusion_score(seq)
    raw_quality = (
        0.20 * geometry_score
        + 0.20 * temporal_score
        + 0.20 * detector_score
        + 0.15 * anatomy_score
        + 0.10 * jerk_score
        + 0.10 * occlusion_score
        + 0.05 * vis_score
    )
    swap_penalty = compute_identity_swap_score(seq)
    raw_quality = raw_quality * (1.0 - 0.40 * swap_penalty)
    if base_quality is not None:
        raw_quality = 0.70 * raw_quality + 0.30 * float(base_quality)
    quality = float(np.clip(raw_quality, 0.0, 1.0))
    breakdown = {
        "geometry": float(geometry_score),
        "temporal": float(temporal_score),
        "detector_conf": float(detector_score),
        "anatomy": float(anatomy_score),
        "motion_realism": float(jerk_score),
        "occlusion": float(occlusion_score),
        "visibility": float(vis_score),
        "identity_swap": float(swap_penalty),
        "left": float(vis["left"]),
        "right": float(vis["right"]),
        "pose": float(vis["pose"]),
        "lips": float(vis["lips"]),
        "pose_vis": float(pose_vis),
        "hand_conf": float(hand_conf),
        "base_quality": float(base_quality) if base_quality is not None else None,
    }
    return quality, breakdown


def _assess_quality_static(sequence, base_quality: float = 0.0, confidence=None):
    """Quality scorer tuned for STATIC images (e.g. ASL Alphabet).

    The standard scorer weights left+right hands equally, penalising one-handed
    signs.  This variant uses the BEST of the two hands and adaptively evaluates
    modality presence so that static hand signs achieve accurate high scores.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[0] == 0:
        return 0.0, {"reason": "empty_sequence"}
    vis = compute_visibility_scores(seq[:1])
    best_hand = max(vis["left"], vis["right"])
    has_pose = vis["pose"] > 0.05
    has_lips = vis["lips"] > 0.05

    if has_pose and has_lips:
        geometry_score = float(
            0.80 * best_hand + 0.15 * vis["pose"] + 0.05 * vis["lips"]
        )
    elif has_pose:
        geometry_score = float(0.85 * best_hand + 0.15 * vis["pose"])
    else:
        geometry_score = float(best_hand)

    temporal_score = 1.0
    if confidence:
        pose_vis = float(confidence.get("pose_vis", 0.0))
        left_conf = float(confidence.get("left_hand_conf", 0.0))
        right_conf = float(confidence.get("right_hand_conf", 0.0))
        hand_conf = max(left_conf, right_conf)
        if pose_vis > 0.05:
            detector_score = float(
                np.clip(0.70 * hand_conf + 0.30 * pose_vis, 0.0, 1.0)
            )
        else:
            detector_score = float(np.clip(hand_conf, 0.0, 1.0))
    else:
        pose_vis = 0.5
        hand_conf = best_hand
        detector_score = float(np.clip(best_hand, 0.0, 1.0))

    anatomy_score = compute_anatomical_consistency(seq[:1])
    occlusion_score = compute_occlusion_score(seq[:1])
    raw_quality = (
        0.30 * geometry_score
        + 0.20 * temporal_score
        + 0.20 * detector_score
        + 0.20 * anatomy_score
        + 0.10 * occlusion_score
    )
    if base_quality is not None:
        raw_quality = 0.70 * raw_quality + 0.30 * float(base_quality)
    quality = float(np.clip(raw_quality, 0.0, 1.0))
    breakdown = {
        "geometry": float(geometry_score),
        "temporal": float(temporal_score),
        "detector_conf": float(detector_score),
        "anatomy": float(anatomy_score),
        "occlusion": float(occlusion_score),
        "left": float(vis["left"]),
        "right": float(vis["right"]),
        "best_hand": float(best_hand),
        "pose": float(vis["pose"]),
        "base_quality": float(base_quality) if base_quality is not None else None,
    }
    return quality, breakdown


def _process_static_image_features(raw_arr):
    """Dedicated fast-path feature extraction for static images.
    Bypasses temporal smoothing and interpolation which are CPU-expensive and redundant for constant/repeated inputs.
    """
    if raw_arr is None or len(raw_arr) == 0:
        return None
    normalizer = CoordinateNormalizer(num_landmarks=NUM_LANDMARKS)
    norm_arr = normalizer.normalize(raw_arr)
    feat_arr = append_kinematic_features(norm_arr)
    seq = np.repeat(feat_arr, 7, axis=0)
    return seq.astype(np.float16)


def _process_sequence(sequence, source_fps: float = 30.0, target_frames=None):
    """Module-level sequence processing pipeline for subprocess workers.
    Mirrors FrankensteinDataProcessor.process_real_sequence but is picklable.
    Applies temporal resampling to standardize frame rate across datasets.
    """
    if sequence is None or len(sequence) == 0:
        return None
    sequence = impute_anatomical_ik_landmarks(sequence)
    if sequence is None or len(sequence) == 0:
        return None
    sequence = smooth_mediapipe_sequence(sequence, window_length=5, polyorder=2)
    if target_frames is not None and target_frames > 0:
        sequence = temporal_resample(
            sequence, target_frames=target_frames, source_fps=source_fps
        )
    elif source_fps and source_fps > 0:
        sequence = resample_sequence_to_fps(
            sequence, source_fps=source_fps, target_fps=TARGET_FPS
        )
    if sequence is None or len(sequence) == 0:
        return None
    normalizer = CoordinateNormalizer(num_landmarks=NUM_LANDMARKS)
    sequence = normalizer.normalize(sequence)
    if sequence is None or len(sequence) == 0:
        return None
    sequence = append_kinematic_features(sequence)
    compressor = KinematicSaliencyCompressor(compression_ratio=0.28)
    sequence = compressor.compress(sequence)
    return sequence.astype(np.float16)


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL WORKER FUNCTIONS
# Each accepts only picklable primitive arguments and returns
#   (record_dict | None,  discard_info_dict | None)
# The calling process handles JSONL discard logging from the discard_info dict.
# ─────────────────────────────────────────────────────────────────────────────


def _proc_alphabet_image(img_path_str, label, split, quality_threshold):
    """Worker: extract one ASL Alphabet static image.

    Enhancement is done inside extract_image (pre-inference) so we run
    MediaPipe exactly once per image regardless of quality.  The old cascade
    retry path was effectively calling the static-mode extractor on itself,
    doubling inference time for every low-quality frame.
    """
    try:
        t0_mp = time.perf_counter()
        extractor = _get_process_extractor(static_mode=True)
        # enhance=True (default): CLAHE / log-compression / DoG applied BEFORE
        # the single MediaPipe inference call, not after.
        raw_arr, base_q, confidence = extractor.extract_image(
            img_path_str, enhance=True
        )
        t_mp = time.perf_counter() - t0_mp

        if raw_arr is None:
            return None, {
                "reason": "no_landmarks",
                "source": "ASL_Alphabet",
                "label": label,
                "quality": 0.0,
                "meta": {"image_path": img_path_str, "timings": {"mediapipe": t_mp}},
            }
        raw_seq = np.repeat(raw_arr, 7, axis=0)

        t0_q = time.perf_counter()
        quality, breakdown = _assess_quality_static(
            raw_seq, base_quality=base_q, confidence=confidence
        )
        t_q = time.perf_counter() - t0_q

        if quality < quality_threshold:
            return None, {
                "reason": "below_threshold",
                "source": "ASL_Alphabet",
                "label": label,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "image_path": img_path_str,
                    "timings": {"mediapipe": t_mp, "quality_assessment": t_q},
                },
            }

        t0_norm = time.perf_counter()
        seq = _process_static_image_features(raw_arr)
        t_norm = time.perf_counter() - t0_norm

        return {
            "task": "isolated_gloss",
            "label": label,
            "signer_id": "synthetic_alpha",
            "features": seq,
            "source": "ASL_Alphabet",
            "split": split,
            "quality": quality,
            "quality_breakdown": breakdown,
            "sample_weight": float(np.clip(quality, 0.25, 1.0)),
            "image_path": img_path_str,
            "timings": {
                "mediapipe": t_mp,
                "quality_assessment": t_q,
                "sequence_normalization": t_norm,
            },
        }, None
    except Exception as exc:
        return None, {
            "reason": "exception",
            "source": "ASL_Alphabet",
            "label": label,
            "quality": 0.0,
            "meta": {"image_path": img_path_str, "error": str(exc)},
        }


def _proc_citizen_row(video_path_str, gloss, participant_id, split, quality_threshold):
    """Worker: extract one ASL Citizen video."""
    try:
        t0_mp = time.perf_counter()
        extractor = _get_process_extractor(static_mode=False)
        sequence, detected_fps, base_q, confidence = extractor.extract_video(
            video_path_str, max_dim=320
        )
        t_mp = time.perf_counter() - t0_mp

        if sequence is None:
            return None, {
                "reason": "no_landmarks",
                "source": "ASL_Citizen",
                "label": gloss,
                "quality": 0.0,
                "meta": {"video_path": video_path_str, "timings": {"mediapipe": t_mp}},
            }

        t0_q = time.perf_counter()
        quality, breakdown = _assess_quality(
            sequence, base_quality=base_q, confidence=confidence
        )
        t_q = time.perf_counter() - t0_q

        if quality < quality_threshold:
            return None, {
                "reason": "below_threshold",
                "source": "ASL_Citizen",
                "label": gloss,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "video_path": video_path_str,
                    "timings": {"mediapipe": t_mp, "quality_assessment": t_q},
                },
            }

        t0_norm = time.perf_counter()
        features = _process_sequence(sequence, source_fps=detected_fps or 30.0)
        del sequence  # free raw landmark array — features already extracted
        t_norm = time.perf_counter() - t0_norm

        if features is None or len(features) < 5:
            return None, {
                "reason": "too_short_after_processing",
                "source": "ASL_Citizen",
                "label": gloss,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "video_path": video_path_str,
                    "timings": {
                        "mediapipe": t_mp,
                        "quality_assessment": t_q,
                        "sequence_normalization": t_norm,
                    },
                },
            }
        return {
            "task": "isolated_gloss",
            "label": gloss,
            "signer_id": str(participant_id),
            "features": features,
            "source": "ASL_Citizen",
            "split": split,
            "quality": quality,
            "quality_breakdown": breakdown,
            "sample_weight": float(np.clip(quality, 0.25, 1.0)),
            "video_path": video_path_str,
            "detected_fps": float(detected_fps or 30.0),
            "timings": {
                "mediapipe": t_mp,
                "quality_assessment": t_q,
                "sequence_normalization": t_norm,
            },
        }, None
    except Exception as exc:
        return None, {
            "reason": "exception",
            "source": "ASL_Citizen",
            "label": gloss,
            "quality": 0.0,
            "meta": {"video_path": video_path_str, "error": str(exc)},
        }


def _clean_chicago_label(label_proc, label_raw=None, label_notes=None, alias_map=None):
    """
    Cleans ChicagoFSWild labels following dataset annotator conventions:
    - Inline comments starting with '#' are stripped.
    - Two-handed indicators ('2:word' -> 'word') are stripped.
    - Spelling errors ('[spelled]*[intended]' -> '[intended]') are parsed to intended word.
    - Uncertainty flags ('word?') are cleaned.
    - Visible breaks ('!') are converted to spaces.
    """
    raw_str = str(
        label_proc
        if pd.notna(label_proc) and str(label_proc).strip()
        else (label_raw or "")
    ).strip()
    if not raw_str:
        return ""

    # Strip inline comments
    raw_str = raw_str.split("#")[0].strip()

    # Handle asterisk spelling corrections: [spelled]*[intended]
    if "*" in raw_str:
        parts = raw_str.split("*")
        raw_str = parts[-1].strip()

    # Strip two-handed signing prefix '2:'
    if raw_str.startswith("2:"):
        raw_str = raw_str[2:].strip()

    # Strip trailing uncertainty flags '?'
    raw_str = raw_str.rstrip("?").strip()

    # Replace visible break markers '!' with spaces
    raw_str = raw_str.replace("!", " ")

    return normalize_gloss(raw_str, alias_map)


def _load_chicago_bbox_map(bbox_root: Path) -> dict:
    """
    Loads bounding box annotations from BBox folder.
    Structure: BBox/[filename]/[xxxx].txt
    Format per line: x0, y0, x1, y1, L (L=1 indicates signing hand)
    Returns: dict[seq_name][frame_stem] = [x0, y0, x1, y1]
    """
    bbox_map = {}
    if not bbox_root or not bbox_root.exists():
        return bbox_map

    actual_root = bbox_root
    if (bbox_root / "BBox").exists() and (bbox_root / "BBox").is_dir():
        actual_root = bbox_root / "BBox"

    try:
        txt_files = list(actual_root.glob("**/*.txt"))
        for txt_file in txt_files:
            rel_path = txt_file.relative_to(actual_root)
            parts = rel_path.parts
            if len(parts) >= 2:
                seq_name = "/".join(parts[:-1]).replace("\\", "/")
                frame_stem = txt_file.stem

                boxes = []
                with open(txt_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        tokens = [
                            t.strip()
                            for t in line.replace(",", " ").split()
                            if t.strip()
                        ]
                        if len(tokens) >= 5:
                            try:
                                x0, y0, x1, y1, label = (
                                    float(tokens[0]),
                                    float(tokens[1]),
                                    float(tokens[2]),
                                    float(tokens[3]),
                                    int(float(tokens[4])),
                                )
                                if label == 1:
                                    boxes.append([x0, y0, x1, y1])
                            except ValueError:
                                pass
                if boxes:
                    if seq_name not in bbox_map:
                        bbox_map[seq_name] = {}
                    x0 = min(b[0] for b in boxes)
                    y0 = min(b[1] for b in boxes)
                    x1 = max(b[2] for b in boxes)
                    y1 = max(b[3] for b in boxes)
                    bbox_map[seq_name][frame_stem] = [x0, y0, x1, y1]
                    try:
                        bbox_map[seq_name][str(int(frame_stem))] = [x0, y0, x1, y1]
                    except ValueError:
                        pass
    except Exception as exc:
        print(f"[!] Warning reading ChicagoFSWild BBox annotations: {exc}")

    return bbox_map


def _proc_chicago_seq(
    frame_path_strs,
    label,
    signer,
    split,
    quality_threshold,
    num_frames_meta,
    seq_bboxes=None,
):
    """Worker: extract one ChicagoFSWild frame-directory sequence with optional BBox guidance."""
    try:
        n_total = len(frame_path_strs)
        if n_total > TARGET_FPS:
            indices = np.unique(np.linspace(0, n_total - 1, num=TARGET_FPS, dtype=int))
            frame_path_strs = [frame_path_strs[idx] for idx in indices]

        extractor = _get_process_extractor(static_mode=False)
        n_frames = len(frame_path_strs)
        seq_buf = np.zeros((n_frames, NUM_LANDMARKS, 3), dtype=np.float32)
        qualities = np.zeros(n_frames, dtype=np.float32)
        conf_keys = ("pose_vis", "left_hand_conf", "right_hand_conf", "handedness_conf")
        conf_bufs = {k: np.zeros(n_frames, dtype=np.float32) for k in conf_keys}
        filled = 0
        t0_mp = time.perf_counter()
        for fp_str in frame_path_strs:
            fp = Path(fp_str)
            frame_stem = fp.stem
            bbox = None
            if seq_bboxes is not None:
                bbox = seq_bboxes.get(frame_stem)
                if bbox is None and frame_stem.isdigit():
                    bbox = seq_bboxes.get(str(int(frame_stem)))

            frame = cv2.imread(fp_str)
            if frame is None:
                continue

            lm, q, conf = None, 0.0, {}
            if bbox is not None:
                fh, fw = frame.shape[:2]
                x0, y0, x1, y1 = bbox
                bw, bh = x1 - x0, y1 - y0
                px0 = max(0, int(x0 - 0.25 * bw))
                py0 = max(0, int(y0 - 0.25 * bh))
                px1 = min(fw, int(x1 + 0.25 * bw))
                py1 = min(fh, int(y1 + 0.25 * bh))
                if (px1 - px0) > 10 and (py1 - py0) > 10:
                    crop_img = frame[py0:py1, px0:px1]
                    c_lm, c_q, c_conf = extractor.extract_frame(
                        crop_img, timestamp_ms=0
                    )
                    c_lm = c_lm.reshape(1, NUM_LANDMARKS, 3)
                    cw, ch = px1 - px0, py1 - py0
                    has_hands = (
                        c_conf.get("left_hand_conf", 0) > 0.1
                        or c_conf.get("right_hand_conf", 0) > 0.1
                    )
                    if has_hands:
                        valid_mask = np.any(c_lm[0] != 0.0, axis=-1)
                        remapped_lm = c_lm.copy()
                        remapped_lm[0, valid_mask, 0] = (px0 + c_lm[0, valid_mask, 0] * cw) / fw
                        remapped_lm[0, valid_mask, 1] = (py0 + c_lm[0, valid_mask, 1] * ch) / fh
                        lm, q, conf = remapped_lm, c_q, c_conf

            if lm is None:
                resized_f = resize_frame_to_max_dimension(frame, max_dim=720)
                f_lm, f_q, f_conf = extractor.extract_frame(resized_f, timestamp_ms=0)
                lm = f_lm.reshape(1, NUM_LANDMARKS, 3)
                q = f_q
                conf = f_conf

            if lm is None:
                continue
            seq_buf[filled] = lm[0]
            qualities[filled] = q
            for k in conf_keys:
                conf_bufs[k][filled] = conf[k]
            filled += 1
        t_mp = time.perf_counter() - t0_mp

        if filled < 5:
            return None, {
                "reason": "no_landmarks",
                "source": "ChicagoFSWild",
                "label": label,
                "quality": 0.0,
                "meta": {"signer": signer, "timings": {"mediapipe": t_mp}},
            }
        seq_arr = seq_buf[:filled]
        base_q = float(np.mean(qualities[:filled]))
        agg_conf = {k: float(np.mean(conf_bufs[k][:filled])) for k in conf_keys}

        t0_q = time.perf_counter()
        quality, breakdown = _assess_quality(
            seq_arr, base_quality=base_q, confidence=agg_conf
        )
        t_q = time.perf_counter() - t0_q

        if quality < quality_threshold:
            return None, {
                "reason": "below_threshold",
                "source": "ChicagoFSWild",
                "label": label,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "signer": signer,
                    "timings": {"mediapipe": t_mp, "quality_assessment": t_q},
                },
            }
        if base_q < 0.35:
            return None, {
                "reason": "low_frame_quality",
                "source": "ChicagoFSWild",
                "label": label,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "signer": signer,
                    "timings": {"mediapipe": t_mp, "quality_assessment": t_q},
                },
            }

        t0_norm = time.perf_counter()
        features = _process_sequence(seq_arr, source_fps=TARGET_FPS)
        del seq_arr  # free raw landmark array — features already extracted
        t_norm = time.perf_counter() - t0_norm

        if features is None or len(features) < 5:
            return None, {
                "reason": "too_short_after_processing",
                "source": "ChicagoFSWild",
                "label": label,
                "quality": quality,
                "meta": {
                    "signer": signer,
                    "timings": {
                        "mediapipe": t_mp,
                        "quality_assessment": t_q,
                        "sequence_normalization": t_norm,
                    },
                },
            }
        return {
            "task": "isolated_gloss",
            "label": label,
            "signer_id": signer,
            "features": features,
            "source": "ChicagoFSWild",
            "split": split,
            "quality": quality,
            "quality_breakdown": breakdown,
            "sample_weight": float(np.clip(quality, 0.25, 1.0)),
            "number_of_frames": num_frames_meta,
            "timings": {
                "mediapipe": t_mp,
                "quality_assessment": t_q,
                "sequence_normalization": t_norm,
            },
        }, None
    except Exception as exc:
        return None, {
            "reason": "exception",
            "source": "ChicagoFSWild",
            "label": label,
            "quality": 0.0,
            "meta": {"signer": signer, "error": str(exc)},
        }


def _proc_numeric_image(img_path_str, label, split, quality_threshold):
    """Worker: extract one Synthetic Numbers static image."""
    try:
        t0_mp = time.perf_counter()
        extractor = _get_process_extractor(static_mode=True)
        raw_arr, base_q, confidence = extractor.extract_image(img_path_str)
        t_mp = time.perf_counter() - t0_mp

        if raw_arr is None:
            return None, {
                "reason": "no_landmarks",
                "source": "Synthetic_Numbers",
                "label": label,
                "quality": 0.0,
                "meta": {"image_path": img_path_str, "timings": {"mediapipe": t_mp}},
            }
        raw_seq = np.repeat(raw_arr, 7, axis=0)

        t0_q = time.perf_counter()
        quality, breakdown = _assess_quality_static(
            raw_seq, base_quality=base_q, confidence=confidence
        )
        t_q = time.perf_counter() - t0_q

        if quality < quality_threshold:
            return None, {
                "reason": "below_threshold",
                "source": "Synthetic_Numbers",
                "label": label,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "image_path": img_path_str,
                    "timings": {"mediapipe": t_mp, "quality_assessment": t_q},
                },
            }

        t0_norm = time.perf_counter()
        features = _process_static_image_features(raw_arr)
        t_norm = time.perf_counter() - t0_norm

        if features is None:
            return None, {
                "reason": "processing_failed",
                "source": "Synthetic_Numbers",
                "label": label,
                "quality": quality,
                "meta": {
                    "image_path": img_path_str,
                    "timings": {
                        "mediapipe": t_mp,
                        "quality_assessment": t_q,
                        "sequence_normalization": t_norm,
                    },
                },
            }
        return {
            "task": "isolated_gloss",
            "label": label,
            "signer_id": "synthetic_numeric",
            "features": features,
            "source": "Synthetic_Numbers",
            "split": split,
            "quality": quality,
            "quality_breakdown": breakdown,
            "sample_weight": float(np.clip(quality, 0.25, 1.0)),
            "image_path": img_path_str,
            "timings": {
                "mediapipe": t_mp,
                "quality_assessment": t_q,
                "sequence_normalization": t_norm,
            },
        }, None
    except Exception as exc:
        return None, {
            "reason": "exception",
            "source": "Synthetic_Numbers",
            "label": label,
            "quality": 0.0,
            "meta": {"image_path": img_path_str, "error": str(exc)},
        }


def _proc_wlasl_instance(
    video_path_str,
    gloss,
    signer_id,
    start_frame,
    end_frame,
    source_fps,
    split,
    quality_threshold,
):
    """Worker: extract one WLASL video clip with optional frame window."""
    try:
        t0_mp = time.perf_counter()
        extractor = _get_process_extractor(static_mode=False)
        sequence, detected_fps, base_q, confidence = extractor.extract_video(
            video_path_str, start_frame=start_frame, end_frame=end_frame, max_dim=320
        )
        t_mp = time.perf_counter() - t0_mp

        if sequence is None:
            return None, {
                "reason": "no_landmarks",
                "source": "WLASL_v0.3",
                "label": gloss,
                "quality": 0.0,
                "meta": {"video_path": video_path_str, "timings": {"mediapipe": t_mp}},
            }

        t0_q = time.perf_counter()
        quality, breakdown = _assess_quality(
            sequence, base_quality=base_q, confidence=confidence
        )
        t_q = time.perf_counter() - t0_q

        if quality < quality_threshold:
            return None, {
                "reason": "below_threshold",
                "source": "WLASL_v0.3",
                "label": gloss,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "video_path": video_path_str,
                    "timings": {"mediapipe": t_mp, "quality_assessment": t_q},
                },
            }
        eff_fps = source_fps if source_fps > 0 else (detected_fps or 30.0)

        t0_norm = time.perf_counter()
        features = _process_sequence(sequence, source_fps=eff_fps)
        del sequence  # free raw landmark array — features already extracted
        t_norm = time.perf_counter() - t0_norm

        if features is None:
            return None, {
                "reason": "processing_failed",
                "source": "WLASL_v0.3",
                "label": gloss,
                "quality": quality,
                "meta": {
                    "video_path": video_path_str,
                    "timings": {
                        "mediapipe": t_mp,
                        "quality_assessment": t_q,
                        "sequence_normalization": t_norm,
                    },
                },
            }
        return {
            "task": "isolated_gloss",
            "label": gloss,
            "signer_id": signer_id,
            "features": features,
            "source": "WLASL_v0.3",
            "split": split,
            "quality": quality,
            "quality_breakdown": breakdown,
            "sample_weight": float(np.clip(quality, 0.25, 1.0)),
            "video_path": video_path_str,
            "timings": {
                "mediapipe": t_mp,
                "quality_assessment": t_q,
                "sequence_normalization": t_norm,
            },
        }, None
    except Exception as exc:
        return None, {
            "reason": "exception",
            "source": "WLASL_v0.3",
            "label": gloss,
            "quality": 0.0,
            "meta": {"video_path": video_path_str, "error": str(exc)},
        }


def _proc_how2sign(npy_path_str, sentence, video_id, split, quality_threshold):
    """Worker: process one How2Sign pre-extracted holistic feature .npy file."""
    try:
        t0_load = time.perf_counter()
        raw_data = np.load(npy_path_str)
        if raw_data is None or raw_data.ndim != 3 or raw_data.shape[0] < 5:
            return None, {
                "reason": "invalid_npy_shape",
                "source": "How2Sign_Holistic",
                "label": sentence,
                "quality": 0.0,
                "meta": {"npy_path": npy_path_str},
            }
        if raw_data.shape[1] < 110:
            return None, {
                "reason": "insufficient_landmark_channels",
                "source": "How2Sign_Holistic",
                "label": sentence,
                "quality": 0.0,
                "meta": {"npy_path": npy_path_str, "channels": int(raw_data.shape[1])},
            }
        fused = np.zeros((raw_data.shape[0], 60, 3), dtype=np.float32)
        fused[:, 0:21, :] = raw_data[:, 25:46, :]
        fused[:, 21:42, :] = raw_data[:, 46:67, :]
        fused[:, 42:48, :] = raw_data[:, [11, 12, 13, 14, 15, 16], :]
        fused[:, 48:60, :] = raw_data[:, 67:79, :]
        t_load = time.perf_counter() - t0_load

        t0_q = time.perf_counter()
        quality, breakdown = _assess_quality(fused, base_quality=1.0)
        t_q = time.perf_counter() - t0_q

        if quality < quality_threshold:
            return None, {
                "reason": "below_threshold",
                "source": "How2Sign_Holistic",
                "label": sentence,
                "quality": quality,
                "breakdown": breakdown,
                "meta": {
                    "npy_path": npy_path_str,
                    "timings": {"npy_load_and_fuse": t_load, "quality_assessment": t_q},
                },
            }

        t0_norm = time.perf_counter()
        normalizer = CoordinateNormalizer(num_landmarks=NUM_LANDMARKS)
        features = normalizer.normalize(fused)
        del fused, raw_data  # free raw arrays — normalized features extracted
        if features is None or len(features) < 5:
            return None, {
                "reason": "too_short_after_processing",
                "source": "How2Sign_Holistic",
                "label": sentence,
                "quality": quality,
                "meta": {"npy_path": npy_path_str},
            }
        features = append_kinematic_features(features).astype(np.float16)
        t_norm = time.perf_counter() - t0_norm

        return {
            "task": "sentence_translation",
            "sentence": canonicalize_sentence(sentence),
            "label": canonicalize_sentence(sentence),
            "signer_id": video_id,
            "features": features,
            "source": "How2Sign_Holistic",
            "split": split,
            "quality": quality,
            "quality_breakdown": breakdown,
            "sample_weight": float(np.clip(quality, 0.25, 1.0)),
            "npy_path": npy_path_str,
            "timings": {
                "npy_load_and_fuse": t_load,
                "quality_assessment": t_q,
                "sequence_normalization": t_norm,
            },
        }, None
    except Exception as exc:
        return None, {
            "reason": "exception",
            "source": "How2Sign_Holistic",
            "label": sentence,
            "quality": 0.0,
            "meta": {"npy_path": npy_path_str, "error": str(exc)},
        }


class KinematicSaliencyCompressor:
    def __init__(self, compression_ratio=0.0, min_frames=8):
        self.target_reduction = compression_ratio
        self.min_frames = min_frames

    def compress(self, sequence):
        sequence_f32 = sequence.astype(np.float32, copy=False)
        S, V, C = sequence_f32.shape
        if S <= self.min_frames:
            return sequence
        num_keep = max(self.min_frames, int(round(S * (1.0 - self.target_reduction))))
        if num_keep >= S:
            return sequence
        hands = sequence_f32[:, 21:42, :3]
        velocities = np.diff(hands, axis=0)
        v_norms = np.linalg.norm(velocities, axis=-1)
        mean_v = v_norms.mean(axis=-1)

        v_t = velocities[:-1]
        v_next = velocities[1:]
        dot_product = np.sum(v_t * v_next, axis=-1)
        norm_product = np.linalg.norm(v_t, axis=-1) * np.linalg.norm(v_next, axis=-1)
        epsilon = 1e-6
        cos_theta = dot_product / (norm_product + epsilon)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        mean_curr = (1.0 - cos_theta).mean(axis=-1)

        # Topological Information Entropy scoring (preserves subtle sign hold frames)
        diff_hands = np.abs(np.diff(hands, axis=0))
        entropy_scores = np.zeros(max(1, S - 1), dtype=np.float32)
        n_dim = float(hands.shape[1] * hands.shape[2])  # 21 * 3 = 63
        max_entropy = math.log2(n_dim) if n_dim > 1 else 1.0
        for t in range(min(S - 1, len(diff_hands))):
            p = diff_hands[t].flatten() + epsilon
            p = p / np.sum(p)
            entropy_scores[t] = -np.sum(p * np.log2(p)) / max_entropy

        saliency_scores = np.zeros(S, dtype=np.float32)
        if S > 2:
            min_len = min(len(mean_v[:-1]), len(mean_curr), len(entropy_scores[:-1]))
            saliency_scores[1 : 1 + min_len] = mean_v[:min_len] * (
                1.0 + 1.5 * mean_curr[:min_len] + 0.5 * entropy_scores[:min_len]
            )
        saliency_scores[0] = mean_v[0] if len(mean_v) > 0 else 1.0
        saliency_scores[-1] = mean_v[-1] if len(mean_v) > 0 else 1.0

        critical_indices = {0, S - 1}
        sorted_indices = np.argsort(saliency_scores)[::-1]
        for idx in sorted_indices:
            if len(critical_indices) >= num_keep:
                break
            critical_indices.add(idx)

        keep_indices = sorted(list(critical_indices))
        return sequence[keep_indices]


# ==============================================================================
# 3. FRANKENSTEIN MULTI-SOURCE INGESTION ENGINES
# ==============================================================================


class OnlineASLDataset(Dataset):
    def __init__(self, pt_file_path, transform=None, is_training=False):
        """
        Loads the clean, preprocessed data payload dictionary and handles target mapping.
        """
        # --- FIX: Target the matching list within the structured payload ---
        self.payload = torch.load(pt_file_path, map_location="cpu")
        if isinstance(self.payload, dict):
            self.data = (
                self.payload.get("isolated_records")
                or self.payload.get("samples")
                or self.payload.get("records")
                or []
            )
        elif isinstance(self.payload, list):
            self.data = self.payload
            self.payload = {"isolated_records": self.data}
        else:
            self.data = []
            self.payload = {"isolated_records": []}
        self.transform = transform
        self.is_training = is_training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = np.asarray(item["features"], dtype=np.float32)

        if self.is_training and self.transform is not None:
            features = self.transform(features)

        # Map labels to their corresponding class indexes safely if downstream mapping is required
        label_to_idx = self.payload.get("label_to_idx", {})
        label_str = item["label"]
        label_idx = label_to_idx.get(label_str, 0)

        return {
            "features": torch.from_numpy(features.astype(np.float32)),
            "label": torch.tensor(label_idx, dtype=torch.long),
        }


def canonicalize_sentence(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sequence_quality(sequence: np.ndarray) -> float:
    if sequence is None or len(sequence) == 0:
        return 0.0
    return float(np.mean(np.any(sequence != 0.0, axis=-1)))


def load_sequence_npy(npy_path: Path, num_landmarks: int) -> np.ndarray | None:
    try:
        raw = np.load(npy_path)
        if raw is None or raw.ndim != 3 or raw.shape[0] < 5:
            return None
        if raw.shape[1] >= num_landmarks:
            raw = raw[:, :num_landmarks, :]
        else:
            padded = np.zeros((raw.shape[0], num_landmarks, 3), dtype=np.float32)
            padded[:, : raw.shape[1], :] = raw
            raw = padded
        raw = raw.astype(np.float32, copy=False)
        raw = impute_missing_landmarks(raw)
        return raw
    except Exception:
        return None


class FrankensteinDataProcessor:
    def __init__(self, normalizer=None, quality_threshold=QUALITY_THRESHOLD):
        self.normalizer = normalizer or CoordinateNormalizer()
        self.synthetic_counter = 0

        self.quality_threshold = float(quality_threshold)
        self.quality_log_dir = KAGGLE_OUTPUT_DIR / QUALITY_LOG_DIRNAME
        self.quality_log_dir.mkdir(parents=True, exist_ok=True)

        self._quality_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self.quality_stats = defaultdict(int)
        self.active_split = "unknown"
        self.is_test = False
        self.gpu_id = None
        self.num_gpus = 1
        self.phase = "all"

    def _shard_tasks(self, tasks: list) -> list:
        """Slice task list for multi-process GPU sharding if active."""
        if self.gpu_id is not None and self.num_gpus > 1 and tasks:
            sharded = [t for i, t in enumerate(tasks) if i % self.num_gpus == self.gpu_id]
            log_msg(f"[*] Shard {self.gpu_id}/{self.num_gpus}: processing {len(sharded)}/{len(tasks)} tasks.")
            return sharded
        return tasks

    def _get_quality_threshold(self, source: str) -> float:
        """Return the quality acceptance threshold for a given dataset source."""
        return DATASET_QUALITY_THRESHOLDS.get(source, QUALITY_THRESHOLD)

    def reset_quality_tracking(self, split):
        with self._quality_lock:
            self.active_split = str(split)
            self.quality_stats = defaultdict(int)

    def _jsonable(self, obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        return obj

    def _append_jsonl(self, path, payload):
        payload = {k: self._jsonable(v) for k, v in payload.items()}
        line = json.dumps(payload, ensure_ascii=False)
        with self._log_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def _update_quality_stat(self, key, amount=1):
        with self._quality_lock:
            self.quality_stats[key] += amount

    def save_quality_summary(self, split):
        out_path = self.quality_log_dir / f"summary_{split}.json"
        with self._quality_lock:
            payload = {
                "split": split,
                "quality_threshold": self.quality_threshold,
                "stats": dict(self.quality_stats),
            }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    def _discard(
        self, *, split, source, task, label, quality, reason, meta=None, breakdown=None
    ):
        payload = {
            "timestamp": datetime.now(timezone.utc),
            "split": split,
            "source": source,
            "task": task,
            "label": label,
            "quality": float(quality),
            "threshold": float(self.quality_threshold),
            "reason": reason,
            "meta": meta or {},
            "breakdown": breakdown or {},
        }
        self._append_jsonl(self.quality_log_dir / f"discarded_{split}.jsonl", payload)
        self._update_quality_stat("discarded_total", 1)
        self._update_quality_stat(f"discarded::{source}", 1)
        self._update_quality_stat(f"discarded_reason::{reason}", 1)

    def _keep(self, *, split, source, quality):
        self._update_quality_stat("kept_total", 1)
        self._update_quality_stat(f"kept::{source}", 1)

    def assess_quality(self, sequence, base_quality=0.0, confidence=None):
        """Delegates to the module-level _assess_quality.
        Defined at module level so ProcessPoolExecutor workers can call it
        without pickling the FrankensteinDataProcessor instance.
        """
        return _assess_quality(
            sequence, base_quality=base_quality, confidence=confidence
        )

    def finalize_sample(
        self,
        *,
        split,
        source,
        task,
        label,
        features,
        raw_sequence,
        base_quality=0.0,
        meta=None,
        confidence=None,
    ):
        quality, breakdown = self.assess_quality(
            raw_sequence, base_quality=base_quality, confidence=confidence
        )
        threshold = self._get_quality_threshold(source)
        if quality < threshold:
            self._discard(
                split=split,
                source=source,
                task=task,
                label=label,
                quality=quality,
                reason="below_threshold",
                meta=meta,
                breakdown=breakdown,
            )
            return None
        self._keep(split=split, source=source, quality=quality)
        return {
            "task": task,
            "label": label,
            "signer_id": meta.get("signer_id", "unknown") if meta else "unknown",
            "features": features,
            "source": source,
            "split": split,
            "quality": quality,
            "quality_breakdown": breakdown,
            "sample_weight": float(np.clip(quality, 0.25, 1.0)),
            **(meta or {}),
        }

    def process_real_sequence(self, sequence, source_fps=30.0):
        """Instance-method wrapper around the module-level processing pipeline.
        Temporal resampling is intentionally absent — native frame lengths are preserved.
        """
        if sequence is None or len(sequence) == 0:
            return None
        sequence = impute_anatomical_ik_landmarks(sequence)
        if sequence is None or len(sequence) == 0:
            return None
        sequence = smooth_mediapipe_sequence(sequence, window_length=5, polyorder=2)
        sequence = self.normalizer.normalize(sequence)
        if sequence is None or len(sequence) == 0:
            return None
        sequence = append_kinematic_features(sequence)
        compressor = KinematicSaliencyCompressor(compression_ratio=0.28)
        sequence = compressor.compress(sequence)
        return sequence.astype(np.float16)

    def process_video_file(
        self,
        video_path,
        source_fps=30.0,
        start_frame=None,
        end_frame=None,
        extractor=None,
    ):
        extractor = extractor or _get_process_extractor()
        sequence, detected_fps, _base_q, _confidence = extractor.extract_video(
            video_path, start_frame=start_frame, end_frame=end_frame
        )
        if sequence is None:
            self.synthetic_counter += 1
            return None
        fps = source_fps if source_fps and source_fps > 0 else detected_fps
        return self.process_real_sequence(sequence, source_fps=fps)

    def _generate_unified_features(self, S_len=45):
        raw = np.random.randn(max(5, S_len), NUM_LANDMARKS, 3).astype(np.float32)
        return self.process_real_sequence(raw, source_fps=TARGET_FPS)

    def process_asl_alphabet(self, split="train"):
        t0 = time.time()
        log_msg(f"[*] Processing ASL Alphabet ({split})...")
        profiler = DatasetProfiler("ASL Alphabet", split, num_workers=NUM_MP_GPU_WORKERS)
        candidates = [
            ALPHABET_DIR / "asl_alphabet_train" / "asl_alphabet_train",
            ALPHABET_DIR / "asl_alphabet_test" / "asl_alphabet_test",
        ]
        existing_dirs = [p for p in candidates if p.exists()]
        if not existing_dirs and ALPHABET_DIR.exists():
            existing_dirs = [ALPHABET_DIR]

        if not existing_dirs:
            log_msg(f"[!] Missing ASL Alphabet directory. Skipping.")
            return []
        threshold = self._get_quality_threshold("ASL_Alphabet")
        tasks = []
        alias_map = getattr(self, "aslex_alias_map", None)

        t0_find = time.perf_counter()
        max_needed = 400 if getattr(self, "is_test", False) else None
        found_files = fast_find_image_files(existing_dirs, max_needed=max_needed)
        profiler.add_timing("fast_find_image_files", time.perf_counter() - t0_find)

        for f, root_path in found_files:
            if get_static_split_assignment(f.name) == split:
                lbl = normalize_gloss(
                    (
                        f.parent.name
                        if f.parent.name != root_path.name
                        else f.stem.split("_")[0]
                    ),
                    alias_map,
                )
                tasks.append((str(f), lbl, split, threshold))
        # The file-scan cache was populated above. Clear it now so the
        # Path objects (potentially tens of thousands) can be GC'd.
        _DIR_FILE_CACHE.clear()
        tasks = self._shard_tasks(tasks)
        if getattr(self, "is_test", False):
            tasks = tasks[:100]
        records = []
        with _create_mediapipe_pool() as executor:
            results = bounded_as_completed(executor, _proc_alphabet_image, tasks)
            for record, discard in tqdm(
                results, total=len(tasks), desc=f"ASL Alphabet [{split}]"
            ):
                profiler.ingest_task_result(record, discard)
                if record is not None:
                    records.append(record)
                    self._keep(
                        split=split, source="ASL_Alphabet", quality=record["quality"]
                    )
                elif discard is not None:
                    self._discard(
                        split=split,
                        source=discard["source"],
                        task="isolated_gloss",
                        label=discard["label"],
                        quality=discard["quality"],
                        reason=discard["reason"],
                        meta=discard.get("meta"),
                        breakdown=discard.get("breakdown"),
                    )
        elapsed = time.time() - t0
        rate = len(records) / elapsed if elapsed > 0 else 0.0
        log_msg(
            f"[+] Loaded {len(records)} ASL Alphabet samples for {split} in {elapsed:.2f}s ({rate:.2f} samples/sec)."
        )
        profiler.print_top5()
        return records

    def process_asl_citizen(self, split="train"):
        t0 = time.time()
        log_msg(f"[*] Processing ASL Citizen ({split})...")
        profiler = DatasetProfiler("ASL Citizen", split, num_workers=NUM_MP_GPU_WORKERS)
        csv_path = (
            ASL_CITIZEN_DIR / "splits" / f"{resolve_split('ASL_Citizen', split)}.csv"
        )
        if not csv_path.exists():
            log_msg(f"[!] Missing: {csv_path}. Skipping.")
            return []
        t0_parse = time.perf_counter()
        df = pd.read_csv(csv_path)
        threshold = self._get_quality_threshold("ASL_Citizen")
        alias_map = getattr(self, "aslex_alias_map", None)

        def _parse_citizen_row(row_dict):
            video_name = (
                str(row_dict.get("Video file", ""))
                .strip()
                .replace("\\", "/")
                .lstrip("/")
            )
            video_path = ASL_CITIZEN_DIR / "videos" / video_name
            gloss_raw = str(row_dict.get("Gloss", "")).strip()
            if not video_path.exists():
                return None, {
                    "reason": "missing_video",
                    "label": gloss_raw,
                    "meta": {"video_path": str(video_path), "video_name": video_name},
                }
            gloss = normalize_gloss(gloss_raw, alias_map)
            if not gloss:
                return None, {
                    "reason": "empty_gloss",
                    "label": gloss_raw,
                    "meta": {"video_path": str(video_path)},
                }
            participant_id = str(row_dict.get("Participant ID", "unknown"))
            return (str(video_path), gloss, participant_id, split, threshold), None

        row_dicts = [r.to_dict() for _, r in df.iterrows()]
        tasks = []
        for row_dict in row_dicts:
            task, discard = _parse_citizen_row(row_dict)
            if task is not None:
                tasks.append(task)
            elif discard is not None:
                self._discard(
                    split=split,
                    source="ASL_Citizen",
                    task="isolated_gloss",
                    label=discard["label"],
                    quality=0.0,
                    reason=discard["reason"],
                    meta=discard.get("meta"),
                )
        profiler.add_timing("metadata_parsing", time.perf_counter() - t0_parse)

        tasks = self._shard_tasks(tasks)
        if getattr(self, "is_test", False):
            tasks = tasks[:100]
        records = []
        with _create_mediapipe_pool() as executor:
            results = bounded_as_completed(executor, _proc_citizen_row, tasks)
            for record, discard in tqdm(
                results, total=len(tasks), desc=f"ASL Citizen [{split}]"
            ):
                profiler.ingest_task_result(record, discard)
                if record is not None:
                    records.append(record)
                    self._keep(
                        split=split, source="ASL_Citizen", quality=record["quality"]
                    )
                elif discard is not None:
                    self._discard(
                        split=split,
                        source=discard["source"],
                        task="isolated_gloss",
                        label=discard["label"],
                        quality=discard["quality"],
                        reason=discard["reason"],
                        meta=discard.get("meta"),
                        breakdown=discard.get("breakdown"),
                    )
        elapsed = time.time() - t0
        rate = len(records) / elapsed if elapsed > 0 else 0.0
        log_msg(
            f"[+] Loaded {len(records)} ASL Citizen samples for {split} in {elapsed:.2f}s ({rate:.2f} samples/sec)."
        )
        profiler.print_top5()
        return records

    def process_how2sign_holistic(self, split="train", min_quality=0.20, limit=None):
        t0 = time.time()
        log_msg(f"[*] Processing How2Sign ({split})...")
        profiler = DatasetProfiler("How2Sign", split, num_workers=NUM_MP_GPU_WORKERS)
        t0_parse = time.perf_counter()
        csv_candidates = [
            HOW2SIGN_DIR / "metadata" / f"how2sign_{split}.csv",
            HOW2SIGN_DIR / "metadata" / f"how2sign_realigned_{split}.csv",
            HOW2SIGN_DIR / f"how2sign_{split}.csv",
            HOW2SIGN_DIR / f"how2sign_realigned_{split}.csv",
        ]
        df = None
        for csv_path in csv_candidates:
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, sep="\t", engine="python")
                    if "SENTENCE" not in [c.upper() for c in df.columns]:
                        df = pd.read_csv(csv_path, sep=None, engine="python")
                    break
                except Exception:
                    pass
        if df is None:
            log_msg(f"[!] Missing How2Sign metadata for {split}. Skipping.")
            return []
        col_map = {c.lower().strip(): c for c in df.columns}
        sentence_col = (
            col_map.get("sentence") or col_map.get("text") or col_map.get("english")
        )
        sentence_name_col = col_map.get("sentence_name") or col_map.get("sentencename")
        video_name_col = col_map.get("video_name") or col_map.get("videoname")
        video_id_col = col_map.get("video_id") or col_map.get("signer_id")

        feature_root_candidates = [
            HOW2SIGN_DIR / split / "frontal",
            HOW2SIGN_DIR / "frontal" / split,
            HOW2SIGN_DIR / split,
            HOW2SIGN_DIR,
        ]
        feature_root = next(
            (p for p in feature_root_candidates if p.exists() and p.is_dir()),
            HOW2SIGN_DIR / split / "frontal",
        )

        threshold = self._get_quality_threshold("How2Sign_Holistic")

        rows = [row.to_dict() for _, row in df.iterrows()]
        rows = self._shard_tasks(rows)
        if limit is not None:
            rows = rows[:limit]

        def _parse_how2sign_row(r_dict):
            sentence = str(r_dict.get(sentence_col, "")).strip() if sentence_col else ""
            sent_name = (
                str(r_dict.get(sentence_name_col, "")).strip()
                if sentence_name_col
                else ""
            )
            vid_name = (
                str(r_dict.get(video_name_col, "")).strip() if video_name_col else ""
            )
            name_stem = sent_name or vid_name

            if not sentence or not name_stem:
                return None, {
                    "reason": "missing_sentence_or_video",
                    "sentence": sentence,
                    "meta": {"sentence_name": name_stem},
                }

            npy_candidates = [
                feature_root / f"{name_stem}.npy",
                feature_root / f"{name_stem}_front_holistic.npy",
                feature_root / f"{name_stem}_holistic.npy",
            ]
            if vid_name and vid_name != name_stem:
                npy_candidates.extend(
                    [
                        feature_root / f"{vid_name}.npy",
                        feature_root / f"{vid_name}_front_holistic.npy",
                    ]
                )

            npy_target = next((c for c in npy_candidates if c.exists()), None)
            if npy_target is None:
                return None, {
                    "reason": "missing_feature_file",
                    "sentence": sentence,
                    "meta": {
                        "searched_stem": name_stem,
                        "feature_root": str(feature_root),
                    },
                }

            video_id = (
                str(r_dict.get(video_id_col, name_stem)) if video_id_col else name_stem
            )
            return (str(npy_target), sentence, video_id, split, threshold), None

        tasks = []
        for r_dict in rows:
            task, discard = _parse_how2sign_row(r_dict)
            if task is not None:
                tasks.append(task)
            elif discard is not None:
                self._discard(
                    split=split,
                    source="How2Sign_Holistic",
                    task="sentence_translation",
                    label=discard["sentence"],
                    quality=0.0,
                    reason=discard["reason"],
                    meta=discard.get("meta"),
                )
        profiler.add_timing("metadata_parsing", time.perf_counter() - t0_parse)

        tasks = self._shard_tasks(tasks)
        if getattr(self, "is_test", False):
            tasks = tasks[:100]
        records = []
        with _create_mediapipe_pool() as executor:
            results = bounded_as_completed(executor, _proc_how2sign, tasks)
            for record, discard in tqdm(
                results, total=len(tasks), desc=f"How2Sign [{split}]"
            ):
                profiler.ingest_task_result(record, discard)
                if record is not None:
                    records.append(record)
                    self._keep(
                        split=split,
                        source="How2Sign_Holistic",
                        quality=record["quality"],
                    )
                elif discard is not None:
                    self._discard(
                        split=split,
                        source=discard["source"],
                        task="sentence_translation",
                        label=discard["label"],
                        quality=discard["quality"],
                        reason=discard["reason"],
                        meta=discard.get("meta"),
                        breakdown=discard.get("breakdown"),
                    )
        elapsed = time.time() - t0
        rate = len(records) / elapsed if elapsed > 0 else 0.0
        log_msg(
            f"[+] Loaded {len(records)} How2Sign samples for {split} in {elapsed:.2f}s ({rate:.2f} samples/sec)."
        )
        profiler.print_top5()
        return records

    def process_chicago_fswild(self, split="train"):
        t0 = time.time()
        log_msg(f"[*] Processing ChicagoFSWild ({split})...")
        profiler = DatasetProfiler("ChicagoFSWild", split, num_workers=NUM_MP_GPU_WORKERS)
        t0_parse = time.perf_counter()
        csv_path = CHICAGO_FSWILD_DIR / "ChicagoFSWild.csv"
        unavailable_path = CHICAGO_FSWILD_DIR / "unavailable.csv"

        # Check if CSV is missing and try to extract it from archive
        if not csv_path.exists():
            archive_candidates = [
                CHICAGO_FSWILD_DIR / "ChicagoFSWild.tgz",
                CHICAGO_FSWILD_DIR / "ChicagoFSWild.tar.gz",
                CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild.tgz",
                CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild.tar.gz",
            ]
            archive_path = next((c for c in archive_candidates if c.exists()), None)

            if archive_path is not None:
                csv_extract_done = KAGGLE_TEMP_DIR / "chicago_csv_extract_done.txt"
                if not csv_extract_done.exists():
                    log_msg(
                        f"[*] Found ChicagoFSWild CSV archive: {archive_path.name}. Extracting to {KAGGLE_TEMP_DIR}..."
                    )
                    try:
                        KAGGLE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
                        with tarfile.open(archive_path, "r:gz") as tar:
                            members = []
                            for m in tar.getmembers():
                                m_name = Path(m.name).name
                                if m_name in ("ChicagoFSWild.csv", "unavailable.csv"):
                                    m.name = m_name  # extract flat
                                    members.append(m)
                            tar.extractall(path=KAGGLE_TEMP_DIR, members=members)
                        csv_extract_done.write_text("done")
                        log_msg(
                            f"[+] Successfully extracted ChicagoFSWild CSV and metadata."
                        )
                    except Exception as exc:
                        log_msg(f"[!] Failed to extract ChicagoFSWild CSV: {exc}")

                if (KAGGLE_TEMP_DIR / "ChicagoFSWild.csv").exists():
                    csv_path = KAGGLE_TEMP_DIR / "ChicagoFSWild.csv"
                if (KAGGLE_TEMP_DIR / "unavailable.csv").exists():
                    unavailable_path = KAGGLE_TEMP_DIR / "unavailable.csv"

        if not csv_path.exists():
            log_msg(f"[!] Missing: {csv_path}. Skipping.")
            return []
        df = pd.read_csv(csv_path)
        target_partition = "dev" if split == "val" else split
        partition_df = df[df["partition"] == target_partition].copy()
        unavailable = set()
        if unavailable_path.exists():
            try:
                udf = pd.read_csv(unavailable_path)
                if "filename" in udf.columns:
                    unavailable = set(udf["filename"].astype(str).str.strip().tolist())
                else:
                    unavailable = set(udf.iloc[:, 0].astype(str).str.strip().tolist())
            except Exception as exc:
                log_msg(f"[!] Could not read unavailable.csv: {exc}")
        frame_root = KAGGLE_TEMP_DIR / "ChicagoFSWild-Frames"
        if not frame_root.exists():
            # Check if there is an uncompressed directory inside the dataset dir
            dataset_frames = CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames"
            if dataset_frames.exists() and dataset_frames.is_dir():
                frame_root = dataset_frames
            else:
                archive_candidates = [
                    CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames.tgz",
                    CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames.tar.gz",
                    CHICAGO_FSWILD_DIR.parent
                    / "chicagofswild"
                    / "ChicagoFSWild-Frames.tgz",
                    CHICAGO_FSWILD_DIR.parent
                    / "chicagofswild"
                    / "ChicagoFSWild-Frames.tar.gz",
                ]
                archive_path = next((c for c in archive_candidates if c.exists()), None)

                if archive_path is not None:
                    extract_done_flag = KAGGLE_TEMP_DIR / "chicago_extract_done.txt"
                    if not extract_done_flag.exists():
                        log_msg(
                            f"[*] Found ChicagoFSWild archive: {archive_path.name}. Extracting to {KAGGLE_TEMP_DIR}..."
                        )
                        try:
                            KAGGLE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
                            lock_dir = KAGGLE_TEMP_DIR / "chicago_extract.lock"
                            try:
                                lock_dir.mkdir(exist_ok=False)
                                with tarfile.open(archive_path, "r:gz") as tar:
                                    tar.extractall(path=KAGGLE_TEMP_DIR)
                                extract_done_flag.write_text("done")
                                try:
                                    lock_dir.rmdir()
                                except Exception:
                                    pass
                                log_msg(
                                    f"[+] Successfully extracted ChicagoFSWild-Frames."
                                )
                            except FileExistsError:
                                # Another worker is extracting. Wait for it to complete.
                                start_time = time.time()
                                while (
                                    not extract_done_flag.exists()
                                    and (time.time() - start_time) < 180
                                ):
                                    time.sleep(2)
                        except Exception as exc:
                            log_msg(f"[!] Archive extraction failed: {exc}")
                            try:
                                lock_dir.rmdir()
                            except Exception:
                                pass

        if not frame_root.exists():
            candidates = [
                KAGGLE_TEMP_DIR / "ChicagoFSWild-Frames",
                KAGGLE_TEMP_DIR / "ChicagoFSWild",
                CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames",
                CHICAGO_FSWILD_DIR / "ChicagoFSWild",
                CHICAGO_FSWILD_DIR,
                KAGGLE_TEMP_DIR,
            ]
            for cand in candidates:
                if cand.exists() and (
                    cand.is_dir()
                    and (
                        len(list(cand.glob("signer_*"))) > 0
                        or len(list(cand.glob("*/*"))) > 0
                    )
                ):
                    frame_root = cand
                    break

        if not frame_root.exists():
            log_msg(f"[!] Missing frames root: {frame_root}. Skipping.")
            return []

        # Check and extract BBox annotations if available
        bbox_dir = KAGGLE_TEMP_DIR / "BBox"
        if not bbox_dir.exists():
            dataset_bbox = CHICAGO_FSWILD_DIR / "BBox"
            if dataset_bbox.exists() and dataset_bbox.is_dir():
                bbox_dir = dataset_bbox
            else:
                bbox_candidates = [
                    CHICAGO_FSWILD_DIR / "BBox.tgz",
                    CHICAGO_FSWILD_DIR / "BBox.tar.gz",
                    CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "BBox.tgz",
                    CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "BBox.tar.gz",
                ]
                bbox_archive = next((c for c in bbox_candidates if c.exists()), None)
                if bbox_archive is not None:
                    bbox_extract_done = (
                        KAGGLE_TEMP_DIR / "chicago_bbox_extract_done.txt"
                    )
                    if not bbox_extract_done.exists():
                        log_msg(
                            f"[*] Found BBox archive: {bbox_archive.name}. Extracting to {KAGGLE_TEMP_DIR}..."
                        )
                        try:
                            KAGGLE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
                            with tarfile.open(bbox_archive, "r:gz") as tar:
                                tar.extractall(path=KAGGLE_TEMP_DIR)
                            bbox_extract_done.write_text("done")
                            log_msg(
                                f"[+] Successfully extracted ChicagoFSWild BBox annotations."
                            )
                        except Exception as exc:
                            log_msg(f"[!] BBox archive extraction failed: {exc}")

        bbox_map = _load_chicago_bbox_map(
            bbox_dir if bbox_dir.exists() else CHICAGO_FSWILD_DIR
        )
        if bbox_map:
            log_msg(
                f"[*] Loaded bounding box annotations for {len(bbox_map)} ChicagoFSWild sequences."
            )

        threshold = self._get_quality_threshold("ChicagoFSWild")
        alias_map = getattr(self, "aslex_alias_map", None)

        def _parse_chicago_row(row_dict):
            filename = str(row_dict.get("filename", "")).strip()
            if filename in unavailable:
                return None, {
                    "reason": "unavailable_filename",
                    "filename": filename,
                    "label": str(row_dict.get("label_proc", "")),
                }
            seq_dir = frame_root / filename
            if seq_dir.is_dir():
                frame_paths = sorted(
                    list(seq_dir.glob("*.jpg"))
                    + list(seq_dir.glob("*.jpeg"))
                    + list(seq_dir.glob("*.png")),
                    key=natural_sort_key,
                )
            else:
                frame_paths = sorted(
                    list(frame_root.glob(f"{filename}*.jpg"))
                    + list(frame_root.glob(f"{filename}*.jpeg"))
                    + list(frame_root.glob(f"{filename}*.png")),
                    key=natural_sort_key,
                )
            if len(frame_paths) < 5:
                return None, {
                    "reason": "too_few_frames",
                    "filename": filename,
                    "label": str(row_dict.get("label_proc", "")),
                    "num_frames": len(frame_paths),
                }

            label = _clean_chicago_label(
                label_proc=row_dict.get("label_proc"),
                label_raw=row_dict.get("label_raw"),
                label_notes=row_dict.get("label_notes"),
                alias_map=alias_map,
            )
            if not label:
                return None, {
                    "reason": "empty_label",
                    "filename": filename,
                    "label": "",
                }

            signer = str(row_dict.get("signer", "unknown")).strip()
            nf_meta = (
                int(row_dict["number_of_frames"])
                if "number_of_frames" in row_dict
                and pd.notna(row_dict["number_of_frames"])
                else None
            )
            norm_filename = filename.replace("\\", "/")
            seq_bboxes = bbox_map.get(norm_filename)
            return (
                [str(p) for p in frame_paths],
                label,
                signer,
                split,
                threshold,
                nf_meta,
                seq_bboxes,
            ), None

        row_dicts = [r.to_dict() for _, r in partition_df.iterrows()]
        tasks = []
        for row_dict in row_dicts:
            task, discard = _parse_chicago_row(row_dict)
            if task is not None:
                tasks.append(task)
            elif discard is not None:
                self._discard(
                    split=split,
                    source="ChicagoFSWild",
                    task="isolated_gloss",
                    label=discard["label"],
                    quality=0.0,
                    reason=discard["reason"],
                    meta={
                        "filename": discard.get("filename"),
                        "num_frames": discard.get("num_frames"),
                    },
                )
        profiler.add_timing("metadata_parsing", time.perf_counter() - t0_parse)

        tasks = self._shard_tasks(tasks)
        if getattr(self, "is_test", False):
            tasks = tasks[:100]
        records = []
        with _create_mediapipe_pool() as executor:
            results = bounded_as_completed(executor, _proc_chicago_seq, tasks)
            for record, discard in tqdm(
                results, total=len(tasks), desc=f"ChicagoFSWild [{split}]"
            ):
                profiler.ingest_task_result(record, discard)
                if record is not None:
                    records.append(record)
                    self._keep(
                        split=split, source="ChicagoFSWild", quality=record["quality"]
                    )
                elif discard is not None:
                    self._discard(
                        split=split,
                        source=discard["source"],
                        task="isolated_gloss",
                        label=discard["label"],
                        quality=discard["quality"],
                        reason=discard["reason"],
                        meta=discard.get("meta"),
                        breakdown=discard.get("breakdown"),
                    )
        elapsed = time.time() - t0
        rate = len(records) / elapsed if elapsed > 0 else 0.0
        log_msg(
            f"[+] Loaded {len(records)} ChicagoFSWild samples for {split} in {elapsed:.2f}s ({rate:.2f} samples/sec)."
        )
        profiler.print_top5()
        return records

    def process_synthetic_numbers(self, split="train"):
        t0 = time.time()
        log_msg(f"[*] Processing Synthetic Numbers ({split})...")
        profiler = DatasetProfiler("Synthetic Numbers", split, num_workers=NUM_MP_GPU_WORKERS)
        candidates = [NUMBER_DIR / "Train_Nums", NUMBER_DIR / "Test_Nums"]
        existing_dirs = [p for p in candidates if p.exists()]
        if not existing_dirs and NUMBER_DIR.exists():
            existing_dirs = [NUMBER_DIR]

        if not existing_dirs:
            log_msg(f"[!] Missing Synthetic Numbers directory. Skipping.")
            return []
        threshold = self._get_quality_threshold("Synthetic_Numbers")
        alias_map = getattr(self, "aslex_alias_map", None)
        tasks = []

        t0_find = time.perf_counter()
        max_needed = 400 if getattr(self, "is_test", False) else None
        found_files = fast_find_image_files(existing_dirs, max_needed=max_needed)
        profiler.add_timing("fast_find_image_files", time.perf_counter() - t0_find)

        for p, root_path in found_files:
            if get_static_split_assignment(p.name) == split:
                lbl = normalize_gloss(
                    (
                        p.parent.name
                        if p.parent.name != root_path.name
                        else p.stem.split("_")[0]
                    ),
                    alias_map,
                )
                tasks.append((str(p), lbl, split, threshold))
        # Clear the file-scan cache — paths are now captured in `tasks`.
        _DIR_FILE_CACHE.clear()
        tasks = self._shard_tasks(tasks)
        if getattr(self, "is_test", False):
            tasks = tasks[:100]
        records = []
        with _create_mediapipe_pool() as executor:
            results = bounded_as_completed(executor, _proc_numeric_image, tasks)
            for record, discard in tqdm(
                results, total=len(tasks), desc=f"Synthetic Numbers [{split}]"
            ):
                profiler.ingest_task_result(record, discard)
                if record is not None:
                    records.append(record)
                    self._keep(
                        split=split,
                        source="Synthetic_Numbers",
                        quality=record["quality"],
                    )
                elif discard is not None:
                    self._discard(
                        split=split,
                        source=discard["source"],
                        task="isolated_gloss",
                        label=discard["label"],
                        quality=discard["quality"],
                        reason=discard["reason"],
                        meta=discard.get("meta"),
                        breakdown=discard.get("breakdown"),
                    )
        elapsed = time.time() - t0
        rate = len(records) / elapsed if elapsed > 0 else 0.0
        log_msg(
            f"[+] Loaded {len(records)} Synthetic Numbers samples for {split} in {elapsed:.2f}s ({rate:.2f} samples/sec)."
        )
        profiler.print_top5()
        return records

    def process_wlasl(self, split="train"):
        t0 = time.time()
        log_msg(f"[*] Processing WLASL v0.3 ({split})...")
        profiler = DatasetProfiler("WLASL v0.3", split, num_workers=NUM_MP_GPU_WORKERS)
        t0_parse = time.perf_counter()
        json_path = WLASL_DIR / "WLASL_v0.3.json"
        missing_path = WLASL_DIR / "missing.txt"
        video_root = WLASL_DIR / "videos"
        if not json_path.exists():
            log_msg(f"[!] Missing: {json_path}. Skipping.")
            return []
        missing_ids = set()
        if missing_path.exists():
            with open(missing_path, "r") as f:
                missing_ids = {line.strip() for line in f if line.strip()}
        with open(json_path, "r") as f:
            wlasl_data = json.load(f)
        threshold = self._get_quality_threshold("WLASL_v0.3")
        alias_map = getattr(self, "aslex_alias_map", None)

        def _parse_wlasl_entry(entry):
            gloss = normalize_gloss(entry.get("gloss", ""), alias_map)
            entry_tasks = []
            entry_discards = []
            for inst in entry.get("instances", []):
                if inst.get("split") != split:
                    continue
                video_id = str(inst.get("video_id", ""))
                if video_id in missing_ids:
                    entry_discards.append(
                        {
                            "reason": "missing_video_id",
                            "label": gloss,
                            "meta": {"video_id": video_id},
                        }
                    )
                    continue
                video_path = video_root / f"{video_id}.mp4"
                if not video_path.exists():
                    entry_discards.append(
                        {
                            "reason": "missing_video",
                            "label": gloss,
                            "meta": {
                                "video_id": video_id,
                                "video_path": str(video_path),
                            },
                        }
                    )
                    continue
                start_frame = int(inst.get("frame_start", 0) or 0)
                end_frame = inst.get("frame_end", -1)
                end_frame = None if end_frame in (-1, None) else int(end_frame)
                source_fps = float(inst.get("fps") or 30.0)
                signer_id = str(inst.get("signer_id", video_id))
                entry_tasks.append(
                    (
                        str(video_path),
                        gloss,
                        signer_id,
                        start_frame,
                        end_frame,
                        source_fps,
                        split,
                        threshold,
                    )
                )
            return entry_tasks, entry_discards

        tasks = []
        for entry in wlasl_data:
            e_tasks, e_discards = _parse_wlasl_entry(entry)
            tasks.extend(e_tasks)
            for discard in e_discards:
                self._discard(
                    split=split,
                    source="WLASL_v0.3",
                    task="isolated_gloss",
                    label=discard["label"],
                    quality=0.0,
                    reason=discard["reason"],
                    meta=discard["meta"],
                )
        profiler.add_timing("metadata_parsing", time.perf_counter() - t0_parse)

        tasks = self._shard_tasks(tasks)
        if getattr(self, "is_test", False):
            tasks = tasks[:100]
        records = []
        with _create_mediapipe_pool() as executor:
            results = bounded_as_completed(executor, _proc_wlasl_instance, tasks)
            for record, discard in tqdm(
                results, total=len(tasks), desc=f"WLASL [{split}]"
            ):
                profiler.ingest_task_result(record, discard)
                if record is not None:
                    records.append(record)
                    self._keep(
                        split=split, source="WLASL_v0.3", quality=record["quality"]
                    )
                elif discard is not None:
                    self._discard(
                        split=split,
                        source=discard["source"],
                        task="isolated_gloss",
                        label=discard["label"],
                        quality=discard["quality"],
                        reason=discard["reason"],
                        meta=discard.get("meta"),
                        breakdown=discard.get("breakdown"),
                    )
        elapsed = time.time() - t0
        rate = len(records) / elapsed if elapsed > 0 else 0.0
        log_msg(
            f"[+] Loaded {len(records)} WLASL samples for {split} in {elapsed:.2f}s ({rate:.2f} samples/sec)."
        )
        profiler.print_top5()
        return records


# ==============================================================================
# 4. PYTORCH DATASET BUILDER
# ==============================================================================
def canonicalize_text(text):
    text = str(text).strip().lower()
    text = re.sub(r"[-]+", "", text)
    text = re.sub(r"[_\/]+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_aslex_alias_map(signdata_csv=ASLEX_SIGNDATA):
    if not signdata_csv.exists():
        return {}
    df = pd.read_csv(signdata_csv, encoding="latin1")
    alias_map = {}
    for _, row in df.iterrows():
        entry = canonicalize_text(row.get("EntryID", ""))
        lemma = canonicalize_text(row.get("LemmaID", ""))
        item = canonicalize_text(row.get("Item", ""))
        canonical = lemma or item or entry
        if not canonical:
            continue
        for key in (entry, lemma, item):
            if key:
                alias_map[key] = canonical
    return alias_map


def normalize_gloss(label, alias_map=None):
    label = canonicalize_text(label)
    if not label:
        return label
    if alias_map:
        return alias_map.get(label, label)
    return label


class FusedASLDataset(Dataset):
    def __init__(self, records, label_to_idx=None, transform=None):
        self.records = [r for r in records if r.get("task") != "sentence_translation"]
        self.transform = transform

        if label_to_idx is not None:
            self.label_to_idx = label_to_idx
            self.idx_to_label = {v: k for k, v in label_to_idx.items()}
            self.records = [r for r in self.records if r["label"] in label_to_idx]
            self.classes = [
                self.idx_to_label[i] for i in sorted(self.idx_to_label.keys())
            ]
        else:
            self.classes = sorted(list(set(r["label"] for r in self.records)))
            self.label_to_idx = {label: idx for idx, label in enumerate(self.classes)}
            self.idx_to_label = {idx: label for idx, label in enumerate(self.classes)}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        features = record["features"]
        label_idx = self.label_to_idx[record["label"]]

        if self.transform:
            features = self.transform(features)

        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "signer_id": record["signer_id"],
            "source": record["source"],
            "length": torch.tensor(features.shape[0], dtype=torch.long),
            "split": record.get("split", "train"),
            "quality": torch.tensor(
                float(record.get("quality", 1.0)), dtype=torch.float32
            ),
            "sample_weight": torch.tensor(
                float(record.get("sample_weight", 1.0)), dtype=torch.float32
            ),
        }


def pad_collate_fn(batch):
    """
    Custom collate function for PyTorch DataLoader to batch variable-length sign sequences safely.
    Pads sequence features to batch maximum length (B, T_max, D) and returns lengths tensor.
    """
    features = [item["features"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    lengths = torch.tensor([f.shape[0] for f in features], dtype=torch.long)

    padded_features = torch.nn.utils.rnn.pad_sequence(
        features, batch_first=True, padding_value=0.0
    )

    return {
        "features": padded_features,  # Shape: (B, T_max, D)
        "labels": labels,  # Shape: (B,)
        "lengths": lengths,  # Shape: (B,)
        "sources": [item.get("source", "unknown") for item in batch],
        "signer_ids": [item.get("signer_id", "unknown") for item in batch],
        "sample_weights": torch.tensor(
            [item.get("sample_weight", 1.0) for item in batch], dtype=torch.float32
        ),
    }


# ==============================================================================
# 5. EXECUTION PIPELINE CONTROL
# ==============================================================================
def save_sharded_payload(payload, split, output_dir, shard_size=1000):
    """Write per-shard .pt files and a lightweight canonical header .pt.

    Fixes two previous RAM bugs:

    1. The old code did ``torch.save(payload, canonical.pt)`` *first*, which
       serialised the entire multi-GB payload in one shot and OOM'd at the
       save stage.

    2. ``shard_payload = dict(payload)`` duplicated sentence_records, sequences,
       vocabulary, and lengths into *every* shard, causing N× duplication.

    New behaviour:
      - Shards are written one-by-one (a slice of isolated_records + shard
        metadata only — no global lists).
      - Each shard is freed from RAM before the next is serialised.
      - The canonical .pt is written *last* and contains only vocabulary,
        label_to_idx, metadata, lengths (an int32 array, small), and
        sentence_records.  It does NOT embed isolated_records or sequences.
        Downstream code that reads shards should use the manifest JSON.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not isinstance(payload, dict):
        # Legacy path — non-dict payload, just save as-is.
        out_pt = output_dir / f"asl_frankenstein_{split}.pt"
        tmp_pt = output_dir / f"asl_frankenstein_{split}.pt.tmp"
        torch.save(payload, tmp_pt)
        tmp_pt.replace(out_pt)
        return out_pt

    records = payload.get("isolated_records") or []
    lengths = payload.get("lengths")
    # sentence_records, vocabulary, label_to_idx, metadata live ONLY in the
    # canonical .pt, not in individual shards.
    shard_manifest = []

    if len(records) > shard_size:
        shard_dir = output_dir / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        num_shards = math.ceil(len(records) / shard_size)
        print(
            f"[*] Writing {num_shards} shard files for {split} "
            f"({len(records)} records @ {shard_size}/shard)."
        )
        for i in range(num_shards):
            start = i * shard_size
            end = min(len(records), (i + 1) * shard_size)
            shard_lengths = (
                lengths[start:end].tolist()
                if isinstance(lengths, np.ndarray) and len(lengths) >= end
                else None
            )
            # Each shard contains ONLY its own records + lengths slice.
            # Global data (vocabulary, sentence_records …) lives in canonical.pt.
            shard_data = {
                "isolated_records": records[start:end],
                "lengths": shard_lengths,
                "metadata": {
                    **(payload.get("metadata") or {}),
                    "shard_index": i,
                    "total_shards": num_shards,
                    "shard_range": [start, end],
                },
            }
            shard_path = shard_dir / f"asl_frankenstein_{split}_shard_{i:03d}.pt"
            tmp_sp = shard_path.with_suffix(".pt.tmp")
            torch.save(shard_data, tmp_sp)
            tmp_sp.replace(shard_path)
            del shard_data  # free before next shard
            _force_gc(f"shard {i}")
            shard_manifest.append(
                {
                    "shard_index": i,
                    "path": str(shard_path),
                    "start": start,
                    "end": end,
                    "count": end - start,
                }
            )
            print(f"    -> Saved shard {i + 1}/{num_shards} to {shard_path}")

        manifest_path = output_dir / f"asl_frankenstein_{split}_shards.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "split": split,
                    "shard_size": shard_size,
                    "num_shards": num_shards,
                    "num_records": len(records),
                    "shards": shard_manifest,
                },
                f,
                indent=2,
            )

    # Canonical .pt — lightweight header (no inline feature arrays).
    # Written LAST so we never OOM before the shards are safely on disk.
    canonical = {
        "vocabulary": payload.get("vocabulary", []),
        "label_to_idx": payload.get("label_to_idx", {}),
        "sentence_records": payload.get("sentence_records") or [],
        "lengths": lengths,  # (N,) int32 — small
        "metadata": {
            **(payload.get("metadata") or {}),
            "total_shards": len(shard_manifest),
            "num_records": len(records),
        },
    }
    out_pt = output_dir / f"asl_frankenstein_{split}.pt"
    tmp_pt = output_dir / f"asl_frankenstein_{split}.pt.tmp"
    torch.save(canonical, tmp_pt)
    tmp_pt.replace(out_pt)

    return out_pt


def build_split(processor, split, normalizer, augment=False, label_to_idx=None):
    """Build a split with fully streaming I/O — no in-memory record accumulation.

    RAM profile (3 phases):

    Phase A — dataset processing
        One dataset at a time.  After each dataset the MediaPipe pool is torn
        down, glibc malloc_trim(0) is called, and only the resulting records
        are on disk (temp shard).  Peak ≈ largest single dataset.

    Phase B — label scan
        Temp shards are read one-by-one, only the string label fields are
        kept.  Peak ≈ set of unique label strings (tiny).

    Phase C — streaming serialisation
        Temp shards are processed one record at a time.  Output shards are
        flushed every SHARD_SIZE records and immediately freed.  Neither a
        full isolated_records list nor a processed_isolated list ever exists
        in RAM.  FusedASLDataset is bypassed entirely (it did a wasteful
        float16→float32→float16 round-trip with no other effect when
        transform=None).

    The returned payload contains vocabulary + metadata + lengths (small
    int32 array) but NO inline feature arrays.  save_sharded_payload receives
    this lightweight dict and writes only a small canonical header .pt.
    """
    processor.reset_quality_tracking(split)

    output_dir = KAGGLE_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    temp_shard_dir = output_dir / f"_tmp_shards_{split}"
    temp_shard_dir.mkdir(parents=True, exist_ok=True)
    temp_shard_paths: list = []

    def _flush_temp(records: list, tag: str) -> None:
        """Write *records* to a temp shard and log the flush."""
        if not records:
            return
        gpu_suffix = f"_gpu{processor.gpu_id}" if getattr(processor, "gpu_id", None) is not None else ""
        idx = len(temp_shard_paths)
        sp = temp_shard_dir / f"shard_{idx:03d}_{tag}{gpu_suffix}.pt"
        torch.save(records, sp)
        temp_shard_paths.append(sp)
        log_msg(f"[GC] Flushed {len(records)} {tag} records \u2192 {sp.name}")

    sentence_records: list = []

    try:
        # Phase A: process all datasets — one at a time —————————————
        if getattr(processor, "phase", "all") in ("all", "extract"):
            recs = processor.process_asl_alphabet(split=split)
            _flush_temp(recs, "alphabet"); del recs
            _release_mediapipe_worker_pool()

            recs = processor.process_asl_citizen(split=split)
            _flush_temp(recs, "citizen"); del recs
            _release_mediapipe_worker_pool()

            recs = processor.process_chicago_fswild(split=split)
            _flush_temp(recs, "chicago"); del recs
            _release_mediapipe_worker_pool()

            recs = processor.process_synthetic_numbers(split=split)
            _flush_temp(recs, "numbers"); del recs
            _release_mediapipe_worker_pool()

            recs = processor.process_wlasl(split=split)
            _flush_temp(recs, "wlasl"); del recs
            _release_mediapipe_worker_pool()

            sentence_records = processor.process_how2sign_holistic(split=split)
            _release_mediapipe_worker_pool()

            if getattr(processor, "phase", "all") == "extract":
                if sentence_records:
                    _flush_temp(sentence_records, "how2sign_holistic"); del sentence_records
                log_msg(f"[+] GPU Shard Process {processor.gpu_id} completed extraction for split {split}.")
                return None, None
        else:
            log_msg(f"[*] Phase A skipped (merge mode). Discovering temp shards across all GPUs...")
            temp_shard_paths = sorted(temp_shard_dir.glob("shard_*.pt"))
            log_msg(f"[*] Found {len(temp_shard_paths)} temp shards for Phase B & Phase C merge.")
            for sp in list(temp_shard_paths):
                if "how2sign_holistic" in sp.name:
                    sentence_records.extend(torch.load(sp, map_location="cpu"))
                    temp_shard_paths.remove(sp)

        # ── Phase B: label-only scan — build vocabulary ——————————————
        if label_to_idx is None:
            log_msg("[*] Phase B: scanning temp shards for label vocabulary...")
            all_labels: set = set()
            for sp in temp_shard_paths:
                shard_recs = torch.load(sp, map_location="cpu")
                for r in shard_recs:
                    all_labels.add(r["label"])
                del shard_recs
                gc.collect()
            classes = sorted(all_labels)
            label_to_idx_final: dict = {lbl: i for i, lbl in enumerate(classes)}
            idx_to_label: dict = dict(enumerate(classes))
            log_msg(f"[*] Vocabulary: {len(classes)} classes.")
        else:
            label_to_idx_final = label_to_idx
            classes = sorted(label_to_idx_final, key=label_to_idx_final.__getitem__)
            idx_to_label = dict(enumerate(classes))

        # ── Phase C: streaming serialisation — temp shards → output shards ——
        log_msg(
            f"[*] Phase C: streaming {len(temp_shard_paths)} temp shards "
            f"\u2192 output shards for '{split}'..."
        )
        out_shard_dir = output_dir / "shards"
        out_shard_dir.mkdir(parents=True, exist_ok=True)

        SHARD_SIZE = 1000
        cur_shard: list = []
        cur_shard_idx = 0
        out_shard_manifest: list = []
        all_lengths: list = []      # frame counts, one int per record (tiny)
        global_rec_idx = 0          # records written so far (across all shards)

        def _flush_out_shard() -> None:
            """Write the current output shard to disk and free its RAM."""
            nonlocal cur_shard_idx, cur_shard, global_rec_idx
            if not cur_shard:
                return
            sp = (
                out_shard_dir
                / f"asl_frankenstein_{split}_shard_{cur_shard_idx:03d}.pt"
            )
            tmp_sp = sp.with_suffix(".pt.tmp")
            shard_start = global_rec_idx - len(cur_shard)
            torch.save(
                {
                    "isolated_records": cur_shard,
                    "metadata": {
                        "split": split,
                        "shard_index": cur_shard_idx,
                        "shard_range": [shard_start, global_rec_idx],
                    },
                },
                tmp_sp,
            )
            tmp_sp.replace(sp)
            out_shard_manifest.append(
                {
                    "shard_index": cur_shard_idx,
                    "path": str(sp),
                    "start": shard_start,
                    "end": global_rec_idx,
                    "count": len(cur_shard),
                }
            )
            log_msg(
                f"[GC] Written output shard {cur_shard_idx} "
                f"({len(cur_shard)} records) \u2192 {sp.name}"
            )
            del cur_shard
            _force_gc(f"output shard {cur_shard_idx}")
            cur_shard_idx += 1
            cur_shard = []

        for temp_sp in tqdm(temp_shard_paths, desc=f"Streaming shards [{split}]"):
            shard_recs = torch.load(temp_sp, map_location="cpu")
            for r in shard_recs:
                lbl = r.get("label", "")
                if lbl not in label_to_idx_final:
                    continue
                # Features are already float16 from the _proc_* workers.
                # We skip the FusedASLDataset float16→float32→float16 round-trip.
                features = np.asarray(r["features"], dtype=np.float16)
                cur_shard.append(
                    {
                        "task": "isolated_gloss",
                        "label": lbl,
                        "signer_id": r.get("signer_id", "unknown"),
                        "features": features,
                        "source": r.get("source", "unknown"),
                        "split": r.get("split", split),
                        "quality": float(r.get("quality", 1.0)),
                        "sample_weight": float(r.get("sample_weight", 1.0)),
                    }
                )
                all_lengths.append(features.shape[0])
                global_rec_idx += 1
                if len(cur_shard) >= SHARD_SIZE:
                    _flush_out_shard()
            del shard_recs
            _force_gc(f"temp shard {temp_sp.name}")

        _flush_out_shard()  # flush any remainder

    finally:
        # Always remove temp shards, even on exception.
        for sp in list(temp_shard_paths):
            try:
                sp.unlink()
            except Exception:
                pass
        try:
            temp_shard_dir.rmdir()
        except Exception:
            pass

    lengths = np.array(all_lengths, dtype=np.int32)

    # Write the shard manifest (separate from the canonical .pt).
    manifest_path = output_dir / f"asl_frankenstein_{split}_shards.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "shard_size": SHARD_SIZE,
                "num_shards": len(out_shard_manifest),
                "num_records": global_rec_idx,
                "shards": out_shard_manifest,
            },
            f,
            indent=2,
        )

    # Lightweight canonical payload — no inline feature arrays.
    # save_sharded_payload will write this as a small header .pt.
    canonical_payload = {
        "vocabulary": classes,
        "label_to_idx": label_to_idx_final,
        "sentence_records": sentence_records,
        "lengths": lengths,          # (N,) int32 — small
        "shard_manifest_path": str(manifest_path),
        "metadata": {
            "split": split,
            "target_fps": TARGET_FPS,
            "num_landmarks": NUM_LANDMARKS,
            "num_records": global_rec_idx,
            "sequence_mode": "sharded_float16",
            "quality_threshold": processor.quality_threshold,
        },
    }

    # Proxy returned as 'fused_dataset' for API compatibility with main().
    class _VocabProxy:
        """Minimal stand-in for FusedASLDataset used only for vocab access."""
        def __init__(self):
            self.label_to_idx = label_to_idx_final
            self.idx_to_label = idx_to_label
            self.classes = classes

    processor.save_quality_summary(split)
    return canonical_payload, _VocabProxy()



def main(argv=None):
    global KAGGLE_OUTPUT_DIR
    parser = argparse.ArgumentParser(description="ASL Multi-Source Fusion Pipeline")
    parser.add_argument(
        "--split", type=str, default="all", choices=["all", "train", "val", "test"]
    )
    parser.add_argument(
        "--augment", action="store_true", help="Enable training data augmentation"
    )
    parser.add_argument(
        "--test",
        "-test",
        action="store_true",
        help="Enable fast testing mode (100 samples max)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KAGGLE_OUTPUT_DIR,
        help="Directory for saved artifacts",
    )
    parser.add_argument(
        "--shard-size", type=int, default=1000, help="Number of records per shard copy"
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=None,
        help="Number of multi-processing workers (overrides NUM_MP_GPU_WORKERS)",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=None, help="Assigned GPU ID for multi-process sharding"
    )
    parser.add_argument(
        "--num-gpus", type=int, default=1, help="Total number of GPU shard processes"
    )
    parser.add_argument(
        "--phase", type=str, default="all", choices=["all", "extract", "merge"], help="Execution phase when sharding"
    )
    args = parser.parse_args(argv)

    global NUM_MP_GPU_WORKERS
    if args.workers is not None and args.workers > 0:
        NUM_MP_GPU_WORKERS = int(args.workers)
        os.environ["NUM_MP_GPU_WORKERS"] = str(NUM_MP_GPU_WORKERS)

    n_gpus = get_num_gpus()
    if args.gpu_id is None and n_gpus > 1 and args.phase in ("all", "extract"):
        import subprocess
        log_msg(f"[*] Master Orchestrator: Detected {n_gpus} GPUs. Launching {n_gpus} independent OS processes (1 per GPU) for sector sharding...")
        total_workers = args.workers if args.workers is not None else NUM_MP_GPU_WORKERS
        workers_per_gpu = max(1, total_workers // n_gpus)

        hook_so = _ensure_egl_device_hook()
        procs = []
        for i in range(n_gpus):
            env = os.environ.copy()
            if hook_so and os.path.exists(hook_so):
                env["LD_PRELOAD"] = hook_so + (":" + env.get("LD_PRELOAD", "") if env.get("LD_PRELOAD") else "")
            env["ASSIGNED_GPU_ID"] = str(i)
            # Do NOT set CUDA_VISIBLE_DEVICES, EGL_VISIBLE_DEVICES, or NVIDIA_VISIBLE_DEVICES inside the
            # shard Popen environment! Setting them at process boot makes libEGL_nvidia cache 1 device only
            # (/dev/nvidia0), preventing our interceptor hook from swapping devices[0] = devices[1].
            # Leaving them unset here lets libEGL_nvidia see both GPUs (*num_devices=2), allowing
            # our interceptor to read ASSIGNED_GPU_ID and lock EGL to /dev/nvidia1.
            env["DRI_PRIME"] = str(i)
            env["DRM_DEVICE"] = f"/dev/dri/renderD{128 + i}"
            env["NUM_AVAILABLE_GPUS"] = "1"
            env["NUM_MP_GPU_WORKERS"] = str(workers_per_gpu)

            cmd = [
                sys.executable,
                sys.argv[0],
                "--phase", "extract",
                "--gpu-id", str(i),
                "--num-gpus", str(n_gpus),
                "--workers", str(workers_per_gpu),
            ]
            if args.split:
                cmd.extend(["--split", args.split])
            if args.test:
                cmd.append("--test")
            if args.augment:
                cmd.append("--augment")
            if args.output_dir:
                cmd.extend(["--output-dir", str(args.output_dir)])
            if args.shard_size:
                cmd.extend(["--shard-size", str(args.shard_size)])

            log_msg(f"[*] Launching GPU Shard Process {i}: {' '.join(cmd)}")
            procs.append((i, subprocess.Popen(cmd, env=env)))

        failed = False
        for i, p in procs:
            ret = p.wait()
            if ret != 0:
                log_msg(f"[!] Error: GPU Shard Process {i} exited with error code {ret}")
                failed = True
            else:
                log_msg(f"[+] GPU Shard Process {i} completed extraction successfully.")

        if failed:
            raise RuntimeError("One or more GPU shard processes failed during Phase A extraction.")

        args.phase = "merge"
        log_msg("[*] All GPU shard extractions finished. Master Orchestrator starting Phase B & C merge...")

    KAGGLE_OUTPUT_DIR = Path(args.output_dir)
    KAGGLE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _init_mediapipe()
    _init_torch()

    log_msg("======================================================================")
    log_msg("      ASL RECOGNITION: PHASE 1 COMPREHENSIVE FRANKENSTEIN ENGINE     ")
    log_msg("======================================================================")
    log_msg(
        f"[*] MediaPipe Tasks Vision (>=0.10.30) | GPU={MEDIAPIPE_USE_GPU} | "
        f"pool workers={NUM_MP_GPU_WORKERS}"
    )
    _ensure_mediapipe_models()

    normalizer = CoordinateNormalizer(
        target_frames=TARGET_FRAMES, num_landmarks=NUM_LANDMARKS
    )
    processor = FrankensteinDataProcessor(normalizer=normalizer)
    processor.is_test = args.test
    processor.gpu_id = args.gpu_id
    processor.num_gpus = args.num_gpus
    processor.phase = args.phase
    processor.aslex_alias_map = load_aslex_alias_map()
    if processor.aslex_alias_map:
        log_msg(f"[*] Loaded {len(processor.aslex_alias_map)} ASLEX alias entries.")
    else:
        log_msg(
            "[!] Warning: No global alias map found. Initialized with empty fallback."
        )

    master_label_to_idx = None
    train_map_path = KAGGLE_OUTPUT_DIR / "vocabulary_mapping_train.json"
    if train_map_path.exists():
        with open(train_map_path, "r", encoding="utf-8") as f:
            master_label_to_idx = json.load(f)
            log_msg("[*] Loaded existing training vocabulary mapping for consistency.")

    splits = SPLITS if args.split == "all" else [args.split]

    for split in splits:
        if split in ["val", "test"] and master_label_to_idx is None:
            raise FileNotFoundError(
                f"[!] Cannot process '{split}' split safely. "
                f"You must process the 'train' split first to lock in the master vocabulary mapping."
            )

        log_msg(
            f"==================== BUILDING SPLIT: {split.upper()} ===================="
        )
        payload, fused_dataset = build_split(
            processor,
            split,
            normalizer,
            augment=args.augment,
            label_to_idx=master_label_to_idx,
        )
        if payload is None:
            continue

        if split == "train" and master_label_to_idx is None:
            master_label_to_idx = fused_dataset.label_to_idx

        out_pt = save_sharded_payload(
            payload, split, KAGGLE_OUTPUT_DIR, shard_size=args.shard_size
        )
        out_map = KAGGLE_OUTPUT_DIR / f"vocabulary_mapping_{split}.json"

        with open(out_map, "w", encoding="utf-8") as f:
            json.dump(fused_dataset.label_to_idx, f, indent=4)

        log_msg(f"[+] Saved: {out_pt}")
        log_msg(f"[+] Saved: {out_map}")

        # Release the split's data from RAM before the next split starts.
        # `label_to_idx` is already captured in master_label_to_idx (a plain dict)
        # so deleting fused_dataset here is safe.
        del payload, fused_dataset
        _force_gc(f"split {split} complete")

    log_msg("[+] All requested splits completed.")
    _shutdown_mediapipe_pool()


if __name__ == "__main__":
    main()
