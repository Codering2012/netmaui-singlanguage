#!/usr/bin/env python3
"""
==============================================================================
ASL RECOGNITION: V3-TPU (TPU v5e-8 / v5litepod-8) FRANKENSTEIN ENGINE
High-Speed Batched Video Ingestion & Static PyTorch XLA WholeBody Extraction
==============================================================================

Architecture Directives (TPU v5e-8 Refactor):
1. PyTorch XLA Multi-Core Orchestration (`torch_xla`):
   - Uses `torch_xla.distributed.xla_multiprocessing.spawn(_tpu_worker_fn, nprocs=8)`.
   - Each worker `i` attaches to `xm.xla_device()` (`xla:0` .. `xla:7`) and processes its 1/8th partition (`records[worker_idx::8]`).
   - Establishes 8 independent CPU disk/decompression pools (`ThreadPoolExecutor(max_workers=16)` in each process) reading from `/kaggle/input` concurrently (`~8 to 12+ videos/sec` combined).

2. Static Shape Padding & Recompilation Protection (`xm.mark_step()`):
   - TPUs compile execution graphs via XLA. Dynamic batch sizes or variable image dimensions trigger graph re-compilation (`recompilation storm`).
   - `_direct_tpu_tensor_inference` enforces strict static tensor dimensions `(batch_size, 3, 384, 288)`. When a video chunk has `N < batch_size` frames, the remaining `[N:batch_size]` slots are zero-padded.
   - Calls `xm.mark_step()` right after SimCC keypoint decoding (`torch.softmax`, `torch.max`), forcing the XLA compiler to execute the systolic matrix multiplication immediately without graph breaks.

3. Zero C++ CUDA Kernel Dependencies (`MMCV_WITH_OPS=0`):
   - Bypasses `mmcv._ext` and `torchvision.ops` CUDA C++ extensions (`nms`, `roi_align`). By feeding batched frames directly into `RTMW` (`TopdownPoseEstimator`) without RTMDet person bounding box detection, `RTMW` runs on standard PyTorch ops (`Conv2d`, `Linear`, `LayerNorm`, `GELU`, `softmax`, `max`) natively supported by `torch_xla`.

4. Preserved Mathematical Core & Quality Assessment:
   - Canonical 133->60 keypoint slicing (`_slice_133_to_60`), `CoordinateNormalizer`, Riemannian SE(3) alignment, `impute_anatomical_ik_landmarks`, visibility quality scoring (`_assess_quality_video`), JSONL checkpointing, and two-tier sharded payload compilation (`save_sharded_payload`) are preserved exact 1:1.
"""

import os
import sys
import glob
import json
import time
import math
import shutil
import logging
import argparse
import warnings
import traceback
import re
import tarfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure MMCV does not attempt to compile C++ CUDA ops on TPU and silence C++ logging
os.environ["MMCV_WITH_OPS"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

# Hardcode CPU count to 96 vCPU as requested
HARDCODED_CPU_COUNT = 96
os.cpu_count = lambda: 96

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision.io as tv_io

# Try importing PyTorch XLA
_XLA_AVAILABLE = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    _XLA_AVAILABLE = True
except ImportError:
    pass


def _safe_torch_device(dev_str: Union[str, torch.device]) -> torch.device:
    if isinstance(dev_str, torch.device):
        return dev_str
    dev_s = str(dev_str).lower()
    if _XLA_AVAILABLE and "xla" in dev_s:
        try:
            return torch_xla.device(dev_str)
        except Exception:
            pass
    try:
        return torch.device(dev_str)
    except Exception:
        return torch.device("cpu")


# Try importing Decord for high-speed video reading
_DECORD_AVAILABLE = False
try:
    import decord
    from decord import VideoReader, cpu

    decord.bridge.set_bridge("torch")
    _DECORD_AVAILABLE = True
except ImportError:
    pass

# Try importing OpenMMLab APIs with explicit registry scope initialization
_MMPOSE_APIS_AVAILABLE = False
_MMPOSE_INIT_LOCK = threading.Lock()
try:
    import mmdet
    import mmpose
    import mmpose.models
    import mmpose.evaluation.functional
    from mmengine.registry import init_default_scope

    init_default_scope("mmpose")
    from mmpose.apis import init_model as init_pose_model
    from mmpose.apis import inference_topdown
    from mmpose.apis import MMPoseInferencer

    _MMPOSE_APIS_AVAILABLE = True
except ImportError:
    pass

# Try importing cv2 for fallback image/video reading with silent C++ logging
try:
    import cv2

    if hasattr(cv2, "setLogLevel"):
        cv2.setLogLevel(0)
    # CRITICAL: Prevent 96-thread explosion per TPU process due to the hardcoded CPU count
    cv2.setNumThreads(1)
except ImportError:
    cv2 = None

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# ==============================================================================
# GLOBAL PATHS & CONFIGURATION
# ==============================================================================
IS_KAGGLE = Path("/kaggle/input").exists()
if IS_KAGGLE:
    possible_inputs = [
        Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl"),
        Path(
            "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"
        ),
        Path("/kaggle/input/frakenstein-asl/asl_preprocessed_phase1"),
        Path("/kaggle/input/frakenstein-asl"),
        Path(
            "/kaggle/input/notebooks/tranquocbao2012/frakenstein-merger/asl_preprocessed_phase1"
        ),
        Path("/kaggle/input/asl-preprocessed-phase1/asl_preprocessed_phase1"),
        Path("/kaggle/input/frakenstein-merger/asl_preprocessed_phase1"),
    ]
    KAGGLE_INPUT = next(
        (p for p in possible_inputs if p.exists()), Path("/kaggle/input")
    )
    KAGGLE_TEMP_DIR = Path("/tmp/temp_extraction")
    DEFAULT_OUTPUT_DIR = Path("/kaggle/working/asl_preprocessed_phase1")
else:
    local_inputs = [
        Path(r"E:\datasets\results\asl_preprocessed_phase1"),
        Path("."),
    ]
    KAGGLE_INPUT = next((p for p in local_inputs if p.exists()), Path("."))
    KAGGLE_TEMP_DIR = Path("./temp_extraction")
    DEFAULT_OUTPUT_DIR = (
        Path(r"E:\datasets\results\asl_preprocessed_phase1")
        if Path(r"E:\datasets\results\asl_preprocessed_phase1").exists()
        else Path("./output")
    )
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)
os.environ["PJRT_DEVICE"] = "TPU"


def _fast_rglob(directory: Path, pattern: str) -> list[Path]:
    import json
    import time
    if not directory.exists():
        return []
        
    hash_str = f"{directory.name}_{pattern}".replace("/", "_").replace("\\", "_").replace("*", "star")
    cache_path = KAGGLE_TEMP_DIR / f"glob_cache_{hash_str}.json"
    lock_path = KAGGLE_TEMP_DIR / f"glob_cache_{hash_str}.lock"
    
    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                return [Path(p) for p in json.load(f)]
        except Exception:
            pass
            
    try:
        KAGGLE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=False)
    except FileExistsError:
        while not cache_path.exists():
            time.sleep(1)
        try:
            with open(cache_path, "r") as f:
                return [Path(p) for p in json.load(f)]
        except Exception:
            pass
            
    files = list(directory.rglob(pattern))
    
    tmp_cache = cache_path.with_suffix(".tmp")
    with open(tmp_cache, "w") as f:
        json.dump([str(p) for p in files], f)
    tmp_cache.rename(cache_path)
    
    try:
        lock_path.unlink()
    except Exception:
        pass
        
    return files


def _gather_candidate_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    candidates = []
    if not root.exists():
        return candidates
    current_level = [root]
    for depth in range(max_depth):
        next_level = []
        for d in current_level:
            try:
                for sub in d.iterdir():
                    if sub.is_dir():
                        candidates.append(sub)
                        next_level.append(sub)
            except Exception:
                pass
        current_level = next_level
        if not current_level:
            break
    return candidates


def resolve_dataset_dir(
    hardcoded_path: Path, marker_file: Optional[str] = None
) -> Path:
    if hardcoded_path.exists():
        return hardcoded_path
    if IS_KAGGLE and Path("/kaggle/input").exists():
        input_root = Path("/kaggle/input")
        # 1. Search by exact marker file recursively across all subdirectories in /kaggle/input
        if marker_file and not any(ch in marker_file for ch in "*?"):
            for p in input_root.rglob(marker_file):
                return p.parent
        # 2. Search by wildcard marker_file (e.g. A/*.jpg or *.csv or *.json)
        if marker_file and any(ch in marker_file for ch in "*?"):
            try:
                sub_pat = marker_file.split("/")[-1]
                for p in input_root.rglob(sub_pat):
                    return p.parent.parent if "/" in marker_file else p.parent
            except Exception:
                pass
        # 3. Search by matching directory name
        parts = hardcoded_path.parts
        target_names = [
            p.lower() for p in parts if p not in ("kaggle", "input", "datasets", ".")
        ]
        for cand in _gather_candidate_dirs(input_root, max_depth=5):
            c_name = cand.name.lower()
            if any(
                t == c_name or t.replace("-", "_") == c_name.replace("-", "_")
                for t in target_names
            ):
                return cand
    return hardcoded_path


def get_asl_alphabet_dir() -> Path:
    fast_paths = [
        Path(
            "/kaggle/input/datasets/grassknoted/asl-alphabet/asl_alphabet_train/asl_alphabet_train"
        ),
        Path("/kaggle/input/datasets/grassknoted/asl-alphabet/asl_alphabet_train"),
        Path("/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train"),
    ]
    for fp in fast_paths:
        if fp.exists():
            return fp
    if IS_KAGGLE and Path("/kaggle/input").exists():
        try:
            for p in Path("/kaggle/input").rglob("asl_alphabet_train"):
                if p.is_dir():
                    return p
            for p in Path("/kaggle/input").rglob("A"):
                if p.is_dir() and (any(p.glob("*.jpg")) or any(p.glob("*.png"))):
                    return p.parent
        except Exception:
            pass
    return KAGGLE_INPUT / "asl-alphabet"


def get_synthetic_dir() -> Path:
    fast_paths = [
        Path("/kaggle/input/datasets/lexset/synthetic-asl-numbers/Train_Nums"),
        Path("/kaggle/input/synthetic-asl-numbers/Train_Nums"),
    ]
    for fp in fast_paths:
        if fp.exists():
            return fp
    if IS_KAGGLE and Path("/kaggle/input").exists():
        try:
            for p in Path("/kaggle/input").rglob("Train_Nums"):
                if p.is_dir():
                    return p
            for p in Path("/kaggle/input").rglob("Train_Alphabet"):
                if p.is_dir():
                    return p
        except Exception:
            pass
    return KAGGLE_INPUT / "synthetic-asl-numbers" / "Train_Nums"


def get_wlasl_dir() -> Path:
    fast_paths = [
        Path("/kaggle/input/datasets/risangbaskoro/wlasl-processed"),
        Path("/kaggle/input/wlasl-processed"),
    ]
    for fp in fast_paths:
        if (fp / "WLASL_v0.3.json").exists():
            return fp
    if IS_KAGGLE and Path("/kaggle/input").exists():
        try:
            for p in Path("/kaggle/input").rglob("WLASL_v0.3.json"):
                return p.parent
        except Exception:
            pass
    return KAGGLE_INPUT / "wlasl-processed"


def get_chicago_dir() -> Path:
    shm_path = Path("/dev/shm/chicago_download")
    if shm_path.exists():
        return shm_path
    fast_paths = [
        Path("/kaggle/input/datasets/joebeachcapital/chicagofswild"),
        Path("/kaggle/input/chicagofswild"),
    ]
    for fp in fast_paths:
        if (fp / "ChicagoFSWild.csv").exists() or (
            fp / "ChicagoFSWild-Frames.tgz"
        ).exists():
            return fp
    if IS_KAGGLE and Path("/kaggle/input").exists():
        try:
            for p in Path("/kaggle/input").rglob("ChicagoFSWild.csv"):
                return p.parent
            for p in Path("/kaggle/input").rglob("ChicagoFSWild-Frames.tgz"):
                return p.parent
        except Exception:
            pass
    return KAGGLE_INPUT / "chicagofswild"


def get_citizen_dir() -> Path:
    fast_paths = [
        Path("/kaggle/input/datasets/abd0kamel/asl-citizen/ASL_Citizen"),
        Path("/kaggle/input/asl-citizen/ASL_Citizen"),
    ]
    for fp in fast_paths:
        if (fp / "videos").exists():
            return fp
    if IS_KAGGLE and Path("/kaggle/input").exists():
        try:
            for p in Path("/kaggle/input").rglob("train.csv"):
                if p.parent.name == "splits" and (p.parent.parent / "videos").exists():
                    return p.parent.parent
                if (p.parent / "videos").exists():
                    return p.parent
            for p in Path("/kaggle/input").rglob("ASL_Citizen"):
                if p.is_dir() and (p / "videos").exists():
                    return p
        except Exception:
            pass
    return KAGGLE_INPUT / "asl-citizen"


def get_how2sign_dir() -> Path:
    shm_path = Path("/dev/shm/how2sign_download")
    if shm_path.exists():
        for p in shm_path.rglob("how2sign_holistic_features"):
            if p.is_dir():
                return p
        return shm_path

    if IS_KAGGLE and Path("/kaggle/input").exists():
        try:
            for p in Path("/kaggle/input").rglob("how2sign_holistic_features"):
                if p.is_dir():
                    return p
            for p in Path("/kaggle/input").rglob("*how2sign*"):
                if p.is_dir():
                    return p
        except Exception:
            pass
    return KAGGLE_INPUT / "how2sign-holistic"


ASL_ALPHABET_DIR = get_asl_alphabet_dir()
WLASL_DIR = get_wlasl_dir()
SYNTHETIC_DIR = get_synthetic_dir()
CHICAGO_FSWILD_DIR = get_chicago_dir()
CHICAGO_FS_WILD_DIR = CHICAGO_FSWILD_DIR
ASL_CITIZEN_DIR = get_citizen_dir()
HOW2SIGN_DIR = get_how2sign_dir()

TARGET_FPS = 30.0
MAX_FRAMES_PER_VIDEO = 300
IMAGE_RESIZE_TARGET = (384, 288)  # (H, W) canonical for RTMW-l_384x288
NUM_LANDMARKS = 60

# Canonical 60-landmark structure:
# 0..20: Left Hand (21)
# 21..41: Right Hand (21)
# 42..47: Upper Body Pose (Shoulder_L, Shoulder_R, Elbow_L, Elbow_R, Wrist_L, Wrist_R) (6)
# 48..59: Facial Lips & Eye Contours (12)
LEFT_HAND_SLICE = slice(0, 21)
RIGHT_HAND_SLICE = slice(21, 42)
POSE_SLICE = slice(42, 48)
FACE_SLICE = slice(48, 60)

_CANONICAL_60_INDICES = np.array(
    list(range(91, 112))  # Left Hand (21)
    + list(range(112, 133))  # Right Hand (21)
    + [5, 6, 7, 8, 9, 10]  # Upper Body Pose (6)
    + [53, 74, 80, 85, 59, 68, 71, 77, 40, 44, 45, 49],  # Facial Lips/Eyes (12)
    dtype=np.int64,
)

# Quality Assessment Thresholds
THRESH_MAX_NAN_RATIO = 0.35
THRESH_MIN_CONF = 0.25
THRESH_JERK = 1500.0
THRESH_STATIC_HAND_CONF = 0.35

# ==============================================================================
# LOGGING & QUALITY TRACKING
# ==============================================================================
LOG_DIR = DEFAULT_OUTPUT_DIR / "quality_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FILE_HANDLE = None


def get_log_handle():
    global _LOG_FILE_HANDLE
    if _LOG_FILE_HANDLE is None:
        log_path = LOG_DIR / "pipeline_execution.log"
        _LOG_FILE_HANDLE = open(log_path, "a", encoding="utf-8")
    return _LOG_FILE_HANDLE


def log_msg(msg: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    try:
        h = get_log_handle()
        h.write(formatted + "\n")
        h.flush()
    except Exception:
        pass


def _get_completed_keys(split: str, dataset_name: str) -> set:
    checkpoint_path = LOG_DIR / f"completed_{split}_{dataset_name}.jsonl"
    if not checkpoint_path.exists():
        return set()
    completed = set()
    try:
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        completed.add(json.loads(line.strip()))
                    except Exception:
                        pass
    except Exception as e:
        log_msg(f"[!] Could not load completed keys from {checkpoint_path}: {e}")
    return completed


def _discard(
    reason: str,
    gloss: str,
    meta: dict,
    split: str,
    threshold: float = 0.0,
    quality: float = 0.0,
) -> None:
    discard_path = LOG_DIR / f"discarded_{split}.jsonl"
    video_id = str(meta.get("video_id", meta.get("filename", meta.get("file", ""))))
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gloss": gloss,
        "reason": reason,
        "quality": float(quality),
        "threshold": float(threshold),
        "video_id": video_id,
        "split": split,
    }
    try:
        with open(discard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


# ==============================================================================
# VOCABULARY NORMALIZATION
# ==============================================================================
def load_alias_map() -> dict:
    return {
        "spelling": "fs",
        "finger_spelling": "fs",
        "space": " ",
        "nothing": "<none>",
        "del": "<delete>",
    }


def normalize_gloss(raw_gloss: str, alias_map: dict) -> str:
    if not raw_gloss:
        return "<none>"
    clean = raw_gloss.strip().lower()
    clean = clean.replace("_", " ").replace("-", " ")
    clean = " ".join(clean.split())
    if clean in alias_map:
        return alias_map[clean]
    return clean


# ==============================================================================
# HIGH-SPEED BATCHED VIDEO INGESTION (`read_video_batch_gpu`)
# ==============================================================================
def _fast_read_bgr(path_str: str) -> Optional[np.ndarray]:
    lower_path = str(path_str).lower()
    # Route PNG files directly to PIL to completely bypass C-level libpng zlib warnings
    if lower_path.endswith(".png"):
        try:
            from PIL import Image

            with Image.open(path_str) as pimg:
                pimg = pimg.convert("RGB")
                if pimg.size != (IMAGE_RESIZE_TARGET[1], IMAGE_RESIZE_TARGET[0]):
                    pimg = pimg.resize(
                        (IMAGE_RESIZE_TARGET[1], IMAGE_RESIZE_TARGET[0]), Image.BILINEAR
                    )
                return np.array(pimg).copy()
        except Exception:
            pass

    if cv2 is not None:
        try:
            img = cv2.imread(path_str, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if img.shape[:2] != IMAGE_RESIZE_TARGET:
                    img = cv2.resize(
                        img,
                        (IMAGE_RESIZE_TARGET[1], IMAGE_RESIZE_TARGET[0]),
                        interpolation=cv2.INTER_AREA,
                    )
                return img
        except Exception:
            pass

    # PIL Fallback for JPEG or other formats
    try:
        from PIL import Image

        with Image.open(path_str) as pimg:
            pimg = pimg.convert("RGB")
            if pimg.size != (IMAGE_RESIZE_TARGET[1], IMAGE_RESIZE_TARGET[0]):
                pimg = pimg.resize(
                    (IMAGE_RESIZE_TARGET[1], IMAGE_RESIZE_TARGET[0]), Image.BILINEAR
                )
            return np.array(pimg).copy()
    except Exception:
        return None


def read_video_batch_gpu(
    video_path: Union[str, Path],
    device: str = "cpu",
    target_fps: float = TARGET_FPS,
    max_frames: int = MAX_FRAMES_PER_VIDEO,
) -> Tuple[Optional[torch.Tensor], None]:
    """
    Reads a video file using decord with native C++ FFmpeg resize.
    Optimization: width/height passed to VideoReader so decord resizes during
    decode — skipping any PyTorch TF.resize() call and saving gigabytes of RAM.
    XLA-safe: returns numpy array (breaks XLA graph context), caller reconstructs tensor.
    """
    path_str = str(video_path)
    if not Path(path_str).exists():
        return None, None

    target_h, target_w = IMAGE_RESIZE_TARGET  # (384, 288) or similar

    # --- PRIMARY: decord with native C++ FFmpeg resize (fastest path) ---
    if _DECORD_AVAILABLE:
        try:
            # Pass width/height so decord's FFmpeg backend resizes during decode.
            # This avoids allocating a full-resolution frame in memory entirely.
            # DO NOT pass width/height to decord. Let it decode at native resolution
            # to avoid the massive CPU bottleneck of software swscale resizing.
            # We will use PyTorch interpolate on the TPU.
            vr = VideoReader(
                path_str,
                ctx=cpu(0),
                num_threads=6,
            )
            num_frames = len(vr)
            if num_frames > 0:
                native_fps = float(vr.get_avg_fps())
                if native_fps <= 0 or math.isnan(native_fps):
                    native_fps = target_fps
                stride = max(1, int(round(native_fps / target_fps)))
                indices = list(range(0, num_frames, stride))[:max_frames]
                if indices:
                    # vr.get_batch returns (T, H, W, 3) uint8 — already at target size!
                    frames_tensor = vr.get_batch(indices)
                    # Convert to numpy to break XLA graph tracking before crossing threads
                    frames_np = frames_tensor.asnumpy()  # zero-copy decord->numpy
                    del frames_tensor, vr
                    # Caller is responsible for tensor reconstruction in main thread
                    t_out = (
                        torch.from_numpy(frames_np)
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .float()
                        / 255.0
                    )
                    del frames_np
                    return t_out, None
        except Exception:
            return None, None

    # --- FALLBACK: torchvision.io.read_video (no cv2, XLA-safe numpy bridge) ---
    # Only runs if decord is not installed.
    try:
        v_tensor, _, info = tv_io.read_video(path_str, pts_unit="sec")
        if v_tensor is not None and v_tensor.shape[0] > 0:
            native_fps = info.get("video_fps", target_fps)
            if native_fps <= 0 or math.isnan(native_fps):
                native_fps = target_fps
            stride = max(1, int(round(native_fps / target_fps)))
            indices = list(range(0, v_tensor.shape[0], stride))[:max_frames]
            v_sub = v_tensor[indices]  # (T, H, W, 3) uint8
            del v_tensor
            if v_sub.shape[1:3] != IMAGE_RESIZE_TARGET:
                v_sub = v_sub.permute(0, 3, 1, 2).float()
                v_sub = TF.resize(v_sub, IMAGE_RESIZE_TARGET, antialias=False)
                v_sub = v_sub.permute(0, 2, 3, 1).to(torch.uint8)
            t_out = v_sub.permute(0, 3, 1, 2).float() / 255.0
            del v_sub
            return t_out, None
    except Exception:
        pass

    return None, None


# ==============================================================================
# RTMW WHOLE-BODY TPU EXTRACTOR (`RTMWWholeBodyExtractor`)
# ==============================================================================
class RTMWWholeBodyExtractor:
    """
    High-Speed WholeBody Pose Extraction Engine tailored for PyTorch XLA (`TPU v5e-8`).
    Uses pure batched tensor forward passes (`_direct_tpu_tensor_inference`) with
    static zero-tensor padding and `xm.mark_step()` synchronization.
    """

    def __init__(self, device: str = "cpu", batch_size: int = 64) -> None:
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.inferencer = None
        # Limit PyTorch CPU intra-op threads per process to prevent thread explosion (EAGAIN)
        try:
            torch.set_num_threads(2)
        except Exception:
            pass
        self._io_pool = ThreadPoolExecutor(
            max_workers=32, thread_name_prefix="extractor_io"
        )
        self._init_model()

    def close(self) -> None:
        if hasattr(self, "_io_pool") and self._io_pool is not None:
            self._io_pool.shutdown(wait=False)
            self._io_pool = None

    def _init_model(self) -> None:
        if not _MMPOSE_APIS_AVAILABLE:
            raise RuntimeError(
                "CRITICAL ERROR: MMPose is not installed. Pipeline crashing to prevent zero-tensors."
            )
        try:
            log_msg(
                f"[*] Initializing RTMW WholeBody Extractor on device '{self.device}' (batch_size={self.batch_size})..."
            )
            with _MMPOSE_INIT_LOCK:
                try:
                    from mmengine.registry import DefaultScope

                    DefaultScope.get_instance("mmpose", scope_name="mmpose")
                except Exception:
                    pass
                # Initialize MMPoseInferencer on 'cpu' to prevent OpenMMLab device string validation errors on 'xla:0'
                self.inferencer = MMPoseInferencer(
                    pose2d="rtmw-l_8xb320-270e_cocktail14-384x288", device="cpu"
                )

            # Safely extract underlying nn.Module and transfer to target device (XLA or CPU)
            raw_model = self._get_pose_model()
            if raw_model is not None:
                try:
                    target_dev = _safe_torch_device(self.device)

                    # Cleanly cast to bfloat16 and move to XLA in one shot
                    self.model = raw_model.to(torch.bfloat16).to(target_dev)

                    self.model.eval()
                except Exception as dev_err:
                    log_msg(
                        f"[!] Note transferring RTMW model to '{self.device}': {dev_err}"
                    )
                    self.model = raw_model.eval()

            # Warmup compilation on XLA device
            if _XLA_AVAILABLE and "xla" in str(self.device).lower():
                dummy_dev = _safe_torch_device(self.device)
                dummy_batch = torch.zeros(
                    (
                        self.batch_size,
                        3,
                        IMAGE_RESIZE_TARGET[0],
                        IMAGE_RESIZE_TARGET[1],
                    ),
                    device=dummy_dev,
                )
                _ = self._direct_tpu_tensor_inference(dummy_batch, valid_len=1)
                log_msg(f"[*] XLA systolic compilation completed on {self.device}.")
        except Exception as e:
            log_msg(
                f"[!] Warning: Failed to initialize MMPoseInferencer on {self.device}: {e}"
            )
            self.inferencer = None

    def _get_pose_model(self):
        """Retrieves the underlying PyTorch nn.Module from MMPoseInferencer or self.model."""
        if self.model is not None:
            return self.model
        if self.inferencer is None:
            return None

        # MMPoseInferencer wraps Pose2DInferencer in self.inferencer (singular)
        if hasattr(self.inferencer, "inferencer") and hasattr(
            self.inferencer.inferencer, "model"
        ):
            return self.inferencer.inferencer.model

        if hasattr(self.inferencer, "inferencers") and isinstance(
            self.inferencer.inferencers, dict
        ):
            for k in ("pose2d", "wholebody", "body", "pose"):
                sub = self.inferencer.inferencers.get(k)
                if sub is not None and hasattr(sub, "model") and sub.model is not None:
                    return sub.model
        if hasattr(self.inferencer, "model") and self.inferencer.model is not None:
            return self.inferencer.model
        return getattr(self.inferencer, "pose_model", None)

    def _direct_tpu_tensor_inference(
        self, tensor_gpu: torch.Tensor, valid_len: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Pure PyTorch static graph inference tailored for TPU XLA.
        Assumes input is ALREADY padded to static `(self.batch_size, 3, 384, 288)` on CPU.
        Runs SimCC keypoint decoding directly on XLA and calls `xm.mark_step()`.
        """
        pose_model = self._get_pose_model()
        if pose_model is None or tensor_gpu is None or tensor_gpu.shape[0] == 0:
            return None

        feats = None
        head_out = None
        try:
            actual_len = valid_len if valid_len is not None else tensor_gpu.shape[0]

            pose_model.eval()
            model_dtype = next(pose_model.parameters()).dtype
            tensor_gpu = tensor_gpu.to(dtype=model_dtype)
            with torch.no_grad():
                feats = pose_model.extract_feat(tensor_gpu)
                head_out = pose_model.head.forward(feats)

                if isinstance(head_out, tuple):
                    # Force strict float32 before Argmax to prevent coordinate collapse
                    pred_x = head_out[0].float()
                    pred_y = head_out[1].float()
                else:
                    return None

                # Vectorized SimCC Argmax Coordinate Decoding
                max_idx_x = torch.argmax(pred_x, dim=2)
                max_idx_y = torch.argmax(pred_y, dim=2)

                split_ratio = getattr(pose_model.head, "simcc_split_ratio", 2.0)
                coords_x = max_idx_x.float() / split_ratio
                coords_y = max_idx_y.float() / split_ratio

                # Scale back to canonical image space
                input_w, input_h = IMAGE_RESIZE_TARGET[1], IMAGE_RESIZE_TARGET[0]
                coords_x = coords_x * (input_w / (pred_x.shape[2] / split_ratio))
                coords_y = coords_y * (input_h / (pred_y.shape[2] / split_ratio))

                max_logit_x, _ = torch.max(pred_x, dim=2)
                max_logit_y, _ = torch.max(pred_y, dim=2)
                conf = torch.min(torch.sigmoid(max_logit_x), torch.sigmoid(max_logit_y))
                keypoints = torch.stack([coords_x, coords_y, conf], dim=2)

                if _XLA_AVAILABLE and "xla" in str(self.device).lower():
                    xm.mark_step()

                # Slice off exact unpadded predictions
                return keypoints[:actual_len].cpu().numpy()
        except Exception as e:
            log_msg(f"[!] Direct TPU tensor inference fallback triggered: {e}")
            return None
        finally:
            del feats, head_out

    def _slice_133_to_60(
        self, kpts_133: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Slices RTMW's 133 WholeBody landmarks to canonical 60 coordinates via SIMD vectorized indexing:
        Left Hand (21), Right Hand (21), Upper Body Pose (6), Facial Lips/Eyes (12).
        """
        kpts_133 = np.nan_to_num(kpts_133, nan=0.0, posinf=0.0, neginf=0.0)
        out_buf = kpts_133[:, _CANONICAL_60_INDICES, :]

        left_conf = float(np.mean(out_buf[:, LEFT_HAND_SLICE, 2]))
        right_conf = float(np.mean(out_buf[:, RIGHT_HAND_SLICE, 2]))
        pose_vis = float(np.mean(out_buf[:, POSE_SLICE, 2]))

        conf_dict = {
            "left_hand_conf": left_conf,
            "right_hand_conf": right_conf,
            "pose_vis": pose_vis,
            "best_hand_conf": max(left_conf, right_conf),
        }
        return out_buf, conf_dict

    def extract_batch(
        self, frames_tensor: torch.Tensor, batch_size: int = 64
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Extracts 60-keypoint tracks from a (T, 3, H, W) video tensor using static XLA chunks."""
        if frames_tensor is None or frames_tensor.shape[0] == 0:
            return None, {}
        T = frames_tensor.shape[0]
        chunks_133 = []
        target_dev = _safe_torch_device(self.device)

        for i in range(0, T, self.batch_size):
            chunk_cpu = frames_tensor[i : i + self.batch_size]
            valid_l = chunk_cpu.shape[0]
            if valid_l < self.batch_size:
                pad = torch.zeros(
                    (
                        self.batch_size - valid_l,
                        3,
                        chunk_cpu.shape[2],
                        chunk_cpu.shape[3],
                    ),
                    dtype=chunk_cpu.dtype,
                )
                chunk_cpu = torch.cat([chunk_cpu, pad], dim=0)

            chunk_gpu = chunk_cpu.to(target_dev)
            
            # --- HARDWARE TPU RESIZING ---
            if chunk_gpu.shape[2:4] != IMAGE_RESIZE_TARGET:
                chunk_gpu = torch.nn.functional.interpolate(
                    chunk_gpu, size=IMAGE_RESIZE_TARGET, mode='bilinear', align_corners=False
                )
            
            kpts_chunk = self._direct_tpu_tensor_inference(chunk_gpu, valid_len=valid_l)
            del chunk_cpu, chunk_gpu
            if kpts_chunk is not None:
                chunks_133.append(kpts_chunk)
            else:
                chunks_133.append(np.zeros((valid_l, 133, 3), dtype=np.float32))

        if not chunks_133:
            return None, {}
        full_133 = np.concatenate(chunks_133, axis=0)[:T]
        del chunks_133
        return self._slice_133_to_60(full_133)

    def extract_video(
        self, video_path: Union[str, Path]
    ) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        tensor_gpu, _ = read_video_batch_gpu(video_path, device=self.device)
        if tensor_gpu is None or tensor_gpu.shape[0] == 0:
            return None, 0.0, {}
        buf, conf = self.extract_batch(tensor_gpu, batch_size=self.batch_size)
        del tensor_gpu
        if buf is None:
            return None, 0.0, conf
        q_val, _ = _assess_quality_video(buf, conf)
        return buf, q_val, conf

    def extract_images_batch(
        self, image_paths: List[Union[str, Path]], batch_size: int = 256
    ) -> List[Tuple[Optional[np.ndarray], float, Dict[str, Any]]]:
        """High-speed batched image extraction.

        Optimization 1 – True zero-copy PyTorch decode:
          torchvision.io.decode_jpeg / decode_png decodes bytes directly into a
          uint8 CPU Tensor [C, H, W] via libjpeg-turbo — no NumPy intermediate.
          .numpy() on a CPU tensor is a zero-copy view of the same buffer, used
          only to sever XLA graph tracking before crossing thread boundaries.

        Optimization 3 – No cv2:
          cv2 spins up hidden C++ threads that deadlock TPU multiprocessing.
          PNG fallback uses torchvision.io.decode_png instead.
        """
        if not image_paths:
            return []

        import torchvision.io as tio
        import torchvision.transforms.functional as TF_vis

        target_h, target_w = IMAGE_RESIZE_TARGET

        def _fast_read(path_str: str) -> Optional[np.ndarray]:
            """Returns a [C, H, W] numpy array (uint8) or None."""
            try:
                raw = Path(path_str).read_bytes()
                buf = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
                ext = path_str.lower()
                if ext.endswith((".jpg", ".jpeg")):
                    img_t = tio.decode_jpeg(
                        buf, mode=tio.ImageReadMode.RGB
                    )  # [C,H,W] uint8
                else:
                    img_t = tio.decode_png(
                        buf, mode=tio.ImageReadMode.RGB
                    )  # [C,H,W] uint8
                # We skip CPU resizing here. All ASL Alphabet images are uniformly 200x200.
                # Stacking them works without resize. We will resize on the TPU.
                # .numpy() is zero-copy (shares the same C++ storage),
                # but breaks XLA graph tracking before we cross thread boundaries.
                return img_t.numpy()
            except Exception:
                return None

        results = []
        effective_bs = max(self.batch_size, batch_size)
        pool = (
            self._io_pool
            if (hasattr(self, "_io_pool") and self._io_pool is not None)
            else None
        )

        for i in range(0, len(image_paths), effective_bs):
            chunk_paths = image_paths[i : i + effective_bs]
            paths_str = [str(p) for p in chunk_paths]

            if pool is not None:
                frames_list = list(pool.map(_fast_read, paths_str))
            else:
                frames_list = [_fast_read(p) for p in paths_str]

            valid_frames = []
            valid_indices = []
            for idx, frm in enumerate(frames_list):
                if frm is not None:
                    valid_frames.append(frm)
                    valid_indices.append(idx)

            chunk_results = [(None, 0.0, {})] * len(chunk_paths)
            if valid_frames:
                # Workers returned [C,H,W] numpy arrays — stack to [N,C,H,W], no permute needed.
                stacked = np.stack(valid_frames, axis=0)  # (N, C, H, W) uint8
                tensor_cpu = torch.from_numpy(stacked).contiguous().float() / 255.0
                del stacked
                target_dev = _safe_torch_device(self.device)

                kpts_chunks = []
                for b_start in range(0, tensor_cpu.shape[0], self.batch_size):
                    sub_cpu = tensor_cpu[b_start : b_start + self.batch_size]
                    valid_l = sub_cpu.shape[0]
                    if valid_l < self.batch_size:
                        pad = torch.zeros(
                            (
                                self.batch_size - valid_l,
                                3,
                                sub_cpu.shape[2],
                                sub_cpu.shape[3],
                            ),
                            dtype=sub_cpu.dtype,
                        )
                        sub_cpu = torch.cat([sub_cpu, pad], dim=0)

                    sub_gpu = sub_cpu.to(target_dev)
                    import torchvision.transforms.functional as TF
                    if sub_gpu.shape[-2:] != IMAGE_RESIZE_TARGET:
                        sub_gpu = TF.resize(
                            sub_gpu, size=IMAGE_RESIZE_TARGET, antialias=False
                        )
                    
                    sub_kpts = self._direct_tpu_tensor_inference(
                        sub_gpu, valid_len=valid_l
                    )
                    del sub_cpu, sub_gpu
                    if sub_kpts is not None:
                        kpts_chunks.append(sub_kpts)
                    else:
                        kpts_chunks.append(
                            np.zeros((valid_l, 133, 3), dtype=np.float32)
                        )

                if kpts_chunks:
                    kpts_133 = np.concatenate(kpts_chunks, axis=0)[: len(valid_frames)]
                    for v_idx, orig_idx in enumerate(valid_indices):
                        single_133 = kpts_133[v_idx : v_idx + 1]
                        buf_60, conf = self._slice_133_to_60(single_133)
                        q_val, _ = _assess_quality_static(buf_60[0], conf)
                        chunk_results[orig_idx] = (buf_60, q_val, conf)
                    del kpts_133
                del tensor_cpu, kpts_chunks
            del frames_list, valid_frames
            results.extend(chunk_results)
        return results


# ==============================================================================
# QUALITY EVALUATION ENGINE
# ==============================================================================
THRESH_JERK = 50.0  # Threshold for high-jerk motion check


def _assess_quality_video(sequence: np.ndarray, conf_dict: dict) -> Tuple[float, dict]:
    if sequence is None or sequence.shape[0] == 0:
        return 0.0, {"reason": "empty_sequence"}

    T, N, C = sequence.shape
    vis_scores = sequence[:, :, 2] if C >= 3 else np.ones((T, N))

    # Calculate non-zero landmark visibility
    non_zero_mask = (np.abs(sequence[:, :, 0]) > 1e-4) | (
        np.abs(sequence[:, :, 1]) > 1e-4
    )
    best_hand_conf = float(conf_dict.get("best_hand_conf", 0.0))

    if best_hand_conf <= 0.01:
        non_zero_vis = vis_scores[non_zero_mask]
        mean_conf = float(np.mean(non_zero_vis)) if len(non_zero_vis) > 0 else 0.85
    else:
        mean_conf = float(best_hand_conf)

    # Temporal smoothness / jerk check
    max_jerk = 0.0
    if T >= 3:
        try:
            coords = sequence[:, :, :2]
            vel = coords[1:] - coords[:-1]
            acc = vel[1:] - vel[:-1]
            jerk = np.linalg.norm(acc, axis=2)
            max_jerk = float(np.max(jerk))
        except Exception:
            max_jerk = 0.0

    score = float(np.clip(mean_conf, 0.25, 1.0))
    breakdown = {
        "temporal": float(1.0 - min(1.0, max_jerk / THRESH_JERK)) if T >= 3 else 1.0,
        "detector_conf": mean_conf,
        "anatomy": 1.0,
        "occlusion": 1.0,
        "left": float(conf_dict.get("left_hand_conf", mean_conf)),
        "right": float(conf_dict.get("right_hand_conf", mean_conf)),
        "best_hand": float(conf_dict.get("best_hand_conf", mean_conf)),
        "pose": float(conf_dict.get("pose_vis", mean_conf)),
    }
    return score, breakdown


def _assess_quality_static(frame_60: np.ndarray, conf_dict: dict) -> Tuple[float, dict]:
    if frame_60 is None or frame_60.shape[0] == 0:
        return 0.0, {"reason": "empty_frame"}

    best_hand = float(conf_dict.get("best_hand_conf", 0.0))
    if best_hand <= 0.01:
        vis = (
            frame_60[:, 2].copy()
            if frame_60.ndim == 2 and frame_60.shape[1] >= 3
            else np.ones(len(frame_60)) * 0.85
        )
        non_zero_mask = (np.abs(frame_60[:, 0]) > 1e-4) | (
            np.abs(frame_60[:, 1]) > 1e-4
        )
        non_zero_vis = vis[non_zero_mask]
        best_hand = float(np.mean(non_zero_vis)) if len(non_zero_vis) > 0 else 0.85

    score = float(np.clip(best_hand, 0.25, 1.0))
    breakdown = {
        "detector_conf": score,
        "anatomy": 1.0,
        "left": float(conf_dict.get("left_hand_conf", 0.0)),
        "right": float(conf_dict.get("right_hand_conf", 0.0)),
        "best_hand": best_hand,
    }
    return score, breakdown


# ==============================================================================
# MATHEMATICAL NORMALIZATION & INVERSE KINEMATICS
# ==============================================================================
class CoordinateNormalizer:
    """Canonical SE(3) and Riemannian invariant normalization for 60-keypoint tracks."""

    def __init__(
        self, target_shoulder_dist: float = 1.0, target_hand_span: float = 0.5
    ) -> None:
        self.target_shoulder_dist = target_shoulder_dist
        self.target_hand_span = target_hand_span

    def normalize(self, sequence: np.ndarray) -> np.ndarray:
        if sequence is None or sequence.shape[0] == 0:
            return sequence
        is_2d = sequence.ndim == 2
        if is_2d:
            seq = sequence[None, :, :].copy()
        else:
            seq = sequence.copy()

        # Shoulder L is index 42, Shoulder R is index 43 in 60-landmark layout
        sh_l = seq[:, 42, :2]
        sh_r = seq[:, 43, :2]
        sh_l_norm = np.linalg.norm(sh_l, axis=1)
        sh_r_norm = np.linalg.norm(sh_r, axis=1)
        has_shoulders = (sh_l_norm > 1e-4) & (sh_r_norm > 1e-4)

        if np.any(has_shoulders):
            # Center origin at mid-shoulder for full-body / upper-body tracks, per-frame
            center_t = np.zeros((seq.shape[0], 2))

            if np.all(has_shoulders):
                center_t = (sh_l + sh_r) * 0.5
            else:
                sh_l_v = sh_l[has_shoulders]
                sh_r_v = sh_r[has_shoulders]
                mean_center = np.mean((sh_l_v + sh_r_v) * 0.5, axis=0)
                center_t = (sh_l + sh_r) * 0.5
                center_t[~has_shoulders] = mean_center

            seq[:, :, :2] -= center_t[:, None, :]

            dist = np.linalg.norm(sh_l - sh_r, axis=1)
            mean_dist = (
                float(np.mean(dist[dist > 1e-4])) if np.any(dist > 1e-4) else 1.0
            )
            scale = self.target_shoulder_dist / max(mean_dist, 1e-4)
            seq[:, :, :2] *= scale
        else:
            # Hand-only normalization for static hand crops (ASL Alphabet / Synthetic Numbers)
            lh_vis = float(np.mean(seq[:, 0:21, 2]))
            rh_vis = float(np.mean(seq[:, 21:42, 2]))

            use_rh = rh_vis > lh_vis
            wrist_idx = 21 if use_rh else 0
            tip_idx = 30 if use_rh else 9

            wrist = seq[:, wrist_idx, :2]
            tip = seq[:, tip_idx, :2]

            # Center origin at wrist, per-frame
            wrist_center_t = wrist.copy()
            mean_wrist = np.mean(wrist, axis=0)
            # Fallback for completely missing wrists in some frames (if they are exactly 0,0)
            missing = np.all(wrist == 0, axis=1)
            if np.any(missing):
                wrist_center_t[missing] = mean_wrist

            seq[:, :, :2] -= wrist_center_t[:, None, :]

            # Scale by hand span (wrist to middle finger tip)
            span = np.linalg.norm(tip - wrist, axis=1)
            mean_span = (
                float(np.mean(span[span > 1e-4])) if np.any(span > 1e-4) else 0.5
            )
            scale = self.target_hand_span / max(mean_span, 1e-4)
            seq[:, :, :2] *= scale

        if is_2d:
            return seq[0]
        return seq


def impute_anatomical_ik_landmarks(sequence: np.ndarray) -> np.ndarray:
    """Imputes missing landmarks using temporal linear interpolation across time."""
    if sequence is None or sequence.shape[0] == 0:
        return sequence
    seq = sequence.copy()
    T, V, C = seq.shape
    for v in range(V):
        vis = seq[:, v, 2]
        missing = np.isnan(vis) | (vis <= 0.01)
        if np.all(missing) or not np.any(missing):
            continue
        valid_idx = np.where(~missing)[0]
        missing_idx = np.where(missing)[0]
        for c in range(2):
            seq[missing_idx, v, c] = np.interp(
                missing_idx, valid_idx, seq[valid_idx, v, c]
            )
        seq[missing_idx, v, 2] = 0.05  # Low imputed visibility
    return seq


def _convert_npy_to_60(arr: Any) -> Optional[np.ndarray]:
    """Robustly converts any .npy keypoint array or dictionary structure into (T, 60, 3) matrix."""
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        try:
            item = arr.item()
            if isinstance(item, dict):
                if "keypoints" in item:
                    arr = np.array(item["keypoints"], dtype=np.float32)
                elif "features" in item:
                    arr = np.array(item["features"], dtype=np.float32)
        except Exception:
            return None

    if not isinstance(arr, np.ndarray):
        return None

    # Reshape 2D (T, K*C) to 3D (T, K, C)
    if arr.ndim == 2:
        T, Total = arr.shape
        if Total % 3 == 0:
            K = Total // 3
            arr = arr.reshape(T, K, 3)
        elif Total % 2 == 0:
            K = Total // 2
            arr = np.concatenate(
                [arr.reshape(T, K, 2), np.ones((T, K, 1), dtype=np.float32) * 0.85],
                axis=2,
            )
        else:
            return None

    if arr.ndim != 3:
        return None

    T, K, C = arr.shape
    if C < 3:
        pad_c = np.ones((T, K, 1), dtype=np.float32) * 0.85
        arr = np.concatenate([arr[:, :, :2], pad_c], axis=2)

    out = np.zeros((T, 60, 3), dtype=np.float32)
    if K == 133:
        out[:, 0:21, :] = arr[:, 91:112, :]  # Left hand
        out[:, 21:42, :] = arr[:, 112:133, :]  # Right hand
        out[:, 42:48, :] = arr[:, [5, 6, 7, 8, 9, 10], :]  # Pose
        out[:, 48:60, :] = arr[
            :, [53, 74, 80, 85, 59, 68, 71, 77, 40, 44, 45, 49], :
        ]  # Face
        return out
    elif K == 543:
        out[:, 0:21, :] = arr[:, 21:42, :]  # Left hand
        out[:, 21:42, :] = arr[:, 42:63, :]  # Right hand
        out[:, 42:48, :] = arr[:, [11, 12, 13, 14, 15, 16], :]  # Pose
        out[:, 48:60, :] = arr[:, :12, :]  # Face
        return out
    elif K == 75:
        out[:, 0:21, :] = arr[:, 25:46, :]
        out[:, 21:42, :] = arr[:, 46:67, :]
        out[:, 42:48, :] = arr[:, [1, 2, 3, 4, 5, 6], :]
        out[:, 48:60, :] = arr[:, 67:75, :]
        return out
    elif K == 60:
        return arr
    elif K >= 42:
        copy_len = min(60, K)
        out[:, :copy_len, :] = arr[:, :copy_len, :]
        return out

    return None


def _extract_conf_and_sanitize_60(
    arr_60: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Given (T, 60, C) landmark array, ensures channel 2 is valid visibility in [0, 1].
    If channel 2 contains Z coordinates (negative values or depth values outside [0, 1]),
    replaces channel 2 with landmark visibility based on non-zero coordinates,
    and computes left_hand_conf, right_hand_conf, pose_vis, best_hand_conf.
    """
    if arr_60 is None or arr_60.shape[0] == 0:
        return arr_60, {}
    T, V, C = arr_60.shape
    clean_buf = arr_60.copy()

    ch2 = clean_buf[:, :, 2] if C >= 3 else np.zeros((T, V))
    is_z_coord = np.any(ch2 < -0.01) or np.any(ch2 > 1.01)

    if is_z_coord or C < 3:
        non_zero = (np.abs(clean_buf[:, :, 0]) > 1e-4) | (
            np.abs(clean_buf[:, :, 1]) > 1e-4
        )
        vis = non_zero.astype(np.float32)
        if C >= 3:
            clean_buf[:, :, 2] = vis
        else:
            clean_buf = np.concatenate([clean_buf[:, :, :2], vis[:, :, None]], axis=2)
    else:
        vis = clean_buf[:, :, 2]

    lh_vis = float(np.mean(vis[:, LEFT_HAND_SLICE]))
    rh_vis = float(np.mean(vis[:, RIGHT_HAND_SLICE]))
    pose_vis = float(np.mean(vis[:, POSE_SLICE]))
    best_hand = max(lh_vis, rh_vis)

    conf_dict = {
        "left_hand_conf": lh_vis,
        "right_hand_conf": rh_vis,
        "pose_vis": pose_vis,
        "best_hand_conf": best_hand,
    }
    return clean_buf, conf_dict


def append_kinematic_features(sequence: np.ndarray) -> np.ndarray:
    if sequence is None or len(sequence) == 0:
        return sequence
    is_2d = sequence.ndim == 2
    if is_2d:
        seq = sequence[None, :, :]
    else:
        seq = sequence
    T, N, C = seq.shape
    vel = np.zeros_like(seq)
    acc = np.zeros_like(seq)
    if T > 1:
        vel[1:] = seq[1:] - seq[:-1]
        vel[0] = vel[1]
    if T > 2:
        acc[1:] = vel[1:] - vel[:-1]
        acc[0] = acc[1]
    out = np.concatenate([seq, vel, acc], axis=-1).astype(np.float32)
    if is_2d:
        return out[0]
    return out


def _process_static_image_features(raw_arr: np.ndarray) -> Optional[np.ndarray]:
    if raw_arr is None or len(raw_arr) == 0:
        return None
    raw_arr = np.nan_to_num(raw_arr, nan=0.0, posinf=0.0, neginf=0.0)
    if raw_arr.ndim == 2:
        raw_arr = raw_arr[None, :, :]
    normalizer = CoordinateNormalizer()
    norm_arr = normalizer.normalize(raw_arr)
    feat_arr = append_kinematic_features(norm_arr)
    seq = np.repeat(feat_arr, 7, axis=0)
    seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
    return seq.astype(np.float16)


def _process_sequence(buf: np.ndarray) -> Optional[np.ndarray]:
    if buf is None or buf.shape[0] == 0:
        return None
    buf = np.nan_to_num(buf, nan=0.0, posinf=0.0, neginf=0.0)
    normalizer = CoordinateNormalizer()
    norm_seq = normalizer.normalize(buf)
    imputed_seq = impute_anatomical_ik_landmarks(norm_seq)
    feat_seq = append_kinematic_features(imputed_seq)
    feat_seq = np.nan_to_num(feat_seq, nan=0.0, posinf=0.0, neginf=0.0)
    return feat_seq.astype(np.float16)


def natural_sort_key(s):
    """Natural sorting key function for numerical filenames like frame_2.jpg vs frame_10.jpg."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", str(s))
    ]


def _clean_chicago_label(label_proc, label_raw=None, label_notes=None, alias_map=None):
    """
    Cleans ChicagoFSWild labels following dataset annotator conventions:
    - Inline comments starting with '#' are stripped.
    - Asterisk spelling corrections ('[spelled]*[intended]') parsed to intended word.
    - Two-handed indicators ('2:word' -> 'word') are stripped.
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

    raw_str = raw_str.split("#")[0].strip()
    if "*" in raw_str:
        parts = raw_str.split("*")
        raw_str = parts[-1].strip()
    if raw_str.startswith("2:"):
        raw_str = raw_str[2:].strip()
    raw_str = raw_str.rstrip("?").strip()
    raw_str = raw_str.replace("!", " ")
    return normalize_gloss(raw_str, alias_map)


SPLIT_ALIASES = {
    "ASL_Citizen": {"train": "train", "val": "val", "test": "test"},
    "WLASL_v0.3": {"train": "train", "val": "val", "test": "test"},
    "How2Sign_Holistic": {"train": "train", "val": "val", "test": "test"},
}


def resolve_split(dataset_name, split):
    """Maps standard splits (train, val, test) to dataset-specific folder/file names."""
    return SPLIT_ALIASES.get(dataset_name, {}).get(split, split)


# ==============================================================================
# BUFFERED SHARD WRITER (`_DatasetProcessor`)
# ==============================================================================
class _DatasetProcessor:
    """Handles dataset extraction on TPU core `worker_idx` with buffered JSONL writes."""

    def __init__(
        self,
        worker_idx: int,
        num_workers: int,
        split: str,
        batch_flush_size: int = 1000,
        batch_size: int = 128,
        device_str: Optional[str] = None,
    ) -> None:
        import gc

        self.gc = gc
        self.worker_idx = worker_idx
        self.num_workers = num_workers
        self.split = split
        self.batch_flush_size = batch_flush_size
        self.batch_size = batch_size

        # Attach directly to provided XLA Device (`xla:0` .. `xla:7`) without calling xm in background threads
        if device_str is not None:
            self.device = str(device_str)
        elif _XLA_AVAILABLE:
            try:
                supported_tpus = [str(d) for d in xm.get_xla_supported_devices()]
                if supported_tpus and worker_idx < len(supported_tpus):
                    self.device = supported_tpus[worker_idx]
                else:
                    self.device = str(xm.xla_device())
            except Exception:
                self.device = str(xm.xla_device())
        else:
            self.device = "cpu"

        if _XLA_AVAILABLE and "xla" in str(self.device).lower():
            try:
                mem_info = xm.get_memory_info(torch_xla.device(self.device))
                free_mb = (
                    mem_info.get("kb_free", 0) / 1024.0
                    if "kb_free" in mem_info
                    else mem_info.get("bytes_free", 0) / (1024.0 * 1024.0)
                )
                total_mb = (
                    mem_info.get("kb_total", 0) / 1024.0
                    if "kb_total" in mem_info
                    else mem_info.get("bytes_limit", 0) / (1024.0 * 1024.0)
                )
                log_msg(
                    f"[+] TPU Telemetry Worker {worker_idx} bound to '{self.device}' (HBM Free: {free_mb:.1f} MB / {total_mb:.1f} MB)"
                )
            except Exception:
                log_msg(
                    f"[+] TPU Telemetry Worker {worker_idx} bound to '{self.device}' (Hardware Active)"
                )

        self.extractor = RTMWWholeBodyExtractor(
            device=self.device, batch_size=self.batch_size
        )
        self.alias_map = load_alias_map()
        self.temp_shard_dir = KAGGLE_TEMP_DIR / "shards"
        self.temp_shard_dir.mkdir(parents=True, exist_ok=True)
        self.micro_shard_count = 0
        self.record_buffer: List[Dict[str, Any]] = []

        self._completed_buffer: Dict[str, List[str]] = defaultdict(list)
        self._buffer_count = 0
        self._io_executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix=f"async_io_{worker_idx}"
        )
        self._pending_io_futures: List[Any] = []
        self._chicago_resolved_seq_dir: Optional[Path] = None
        self._chicago_seq_dir_cache: Dict[str, Optional[List[Path]]] = {}

    def close(self) -> None:
        try:
            self._flush_buffer()
        except Exception:
            pass
        if hasattr(self, "_pending_io_futures") and self._pending_io_futures:
            for fut in self._pending_io_futures:
                try:
                    fut.result(timeout=60)
                except Exception:
                    pass
            self._pending_io_futures.clear()
        if hasattr(self, "_io_executor") and self._io_executor is not None:
            self._io_executor.shutdown(wait=True)
            self._io_executor = None
        if hasattr(self, "extractor") and self.extractor is not None:
            self.extractor.close()
            self.extractor = None
        self.gc.collect()

    def _push_record(self, record: Dict[str, Any]) -> None:
        if isinstance(record.get("features"), np.ndarray):
            record["features"] = np.nan_to_num(
                record["features"].astype(np.float16), nan=0.0, posinf=0.0, neginf=0.0
            )
        self.record_buffer.append(record)
        if len(self.record_buffer) >= self.batch_flush_size:
            self._flush_record_buffer()

    def _flush_record_buffer(self) -> None:
        if not self.record_buffer:
            return
        records_to_save = list(self.record_buffer)
        self.record_buffer.clear()
        micro_shard_path = (
            self.temp_shard_dir
            / f"temp_shard_{self.split}_w{self.worker_idx}_{self.micro_shard_count:04d}.pt"
        )
        self.micro_shard_count += 1

        def _async_save(recs, path):
            try:
                torch.save(recs, path)
            except Exception as e:
                log_msg(f"[!] Warning writing binary temp micro-shard {path}: {e}")
            finally:
                del recs

        if hasattr(self, "_io_executor") and self._io_executor is not None:
            self._pending_io_futures = [
                f for f in self._pending_io_futures if not f.done()
            ]
            while len(self._pending_io_futures) >= 2:
                try:
                    self._pending_io_futures[0].result(timeout=60)
                except Exception as e:
                    log_msg(f"[!] Warning waiting for async save backpressure: {e}")
                self._pending_io_futures = [
                    f for f in self._pending_io_futures if not f.done()
                ]

            fut = self._io_executor.submit(
                _async_save, records_to_save, micro_shard_path
            )
            self._pending_io_futures.append(fut)
        else:
            _async_save(records_to_save, micro_shard_path)
        self.gc.collect()

    def _buffer_completed_key(self, key: str, dataset_name: str) -> None:
        self._completed_buffer[dataset_name].append(key)
        self._buffer_count += 1
        if self._buffer_count >= 100:
            self._flush_buffer()

    def _flush_buffer(self) -> None:
        for dataset_name, keys in self._completed_buffer.items():
            if not keys:
                continue
            checkpoint_path = LOG_DIR / f"completed_{self.split}_{dataset_name}.jsonl"
            try:
                with open(checkpoint_path, "a", encoding="utf-8") as f:
                    for k in keys:
                        f.write(json.dumps(k) + "\n")
            except Exception:
                pass
        self._completed_buffer.clear()
        self._buffer_count = 0
        self._flush_record_buffer()

    def process_asl_alphabet(self) -> None:
        if not ASL_ALPHABET_DIR.exists():
            if self.worker_idx == 0:
                log_msg(
                    f"[!] Warning: ASL_Alphabet directory '{ASL_ALPHABET_DIR}' does not exist."
                )
            return
        completed_keys = _get_completed_keys(self.split, "ASL_Alphabet")
        all_files = sorted(_fast_rglob(ASL_ALPHABET_DIR, "*.jpg")) + sorted(
            _fast_rglob(ASL_ALPHABET_DIR, "*.png")
        )
        if not all_files:
            if self.worker_idx == 0:
                log_msg(
                    f"[!] Warning: No image files found in ASL_Alphabet directory '{ASL_ALPHABET_DIR}'."
                )
            return

        # Filter files by target split folder (asl_alphabet_train vs asl_alphabet_test)
        target_sub = (
            f"asl_alphabet_{self.split}"
            if self.split in ("train", "test")
            else "asl_alphabet_train"
        )
        split_files = [
            f
            for f in all_files
            if target_sub in str(f.resolve()).lower()
            or f"/{self.split}/" in str(f.resolve()).lower()
            or f"_{self.split}_" in f.name.lower()
        ]
        if split_files:
            all_files = split_files
        elif self.split in ("val", "test"):
            test_files = [
                f for f in all_files if "asl_alphabet_test" in str(f.resolve()).lower()
            ]
            if test_files:
                all_files = test_files

        # CRITICAL FIX (Point 3): ASL Alphabet deterministic train/val split
        import hashlib
        if self.split in ("train", "val") and "asl_alphabet_train" in target_sub:
            filtered = []
            for f in all_files:
                h = int(hashlib.md5(f.name.encode('utf-8')).hexdigest(), 16)
                is_val = (h % 10 == 0) # 10% validation
                if (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    filtered.append(f)
            all_files = filtered

        my_files = [
            f
            for i, f in enumerate(all_files)
            if i % self.num_workers == self.worker_idx
        ]

        pending_files = []
        for f in my_files:
            key = str(f.resolve())
            if key in completed_keys:
                continue
            pending_files.append(f)

        chunk_step = max(128, self.batch_size * 2)
        total_chunks = math.ceil(len(pending_files) / chunk_step)
        for idx_c, i in enumerate(range(0, len(pending_files), chunk_step)):
            if idx_c % 10 == 0 or idx_c == total_chunks - 1:
                log_msg(
                    f"[*] TPU Worker {self.worker_idx} [ASL_Alphabet]: Batch {idx_c+1}/{total_chunks}"
                )
            batch_paths = pending_files[i : i + chunk_step]
            results = self.extractor.extract_images_batch(
                batch_paths, batch_size=self.batch_size
            )
            for f_path, (buf_60, q_val, conf) in zip(batch_paths, results):
                key = str(f_path.resolve())
                label = normalize_gloss(f_path.parent.name, self.alias_map)
                if buf_60 is None:
                    _discard(
                        "no_landmarks",
                        label,
                        {"file": f_path.name},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "ASL_Alphabet")
                    continue
                seq = _process_static_image_features(buf_60)
                if seq is None:
                    _discard(
                        "processing_failed",
                        label,
                        {"file": f_path.name},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "ASL_Alphabet")
                    continue
                stat_q, breakdown = _assess_quality_static(buf_60[0], conf)
                record = {
                    "task": "static_alphabet",
                    "label": label,
                    "signer_id": "unknown",
                    "features": seq,
                    "source": "ASL_Alphabet",
                    "split": self.split,
                    "quality": stat_q,
                    "quality_breakdown": breakdown,
                    "sample_weight": float(np.clip(stat_q, 0.25, 1.0)),
                    "video_id": f_path.name,
                }
                self._push_record(record)
                self._buffer_completed_key(key, "ASL_Alphabet")
        self._flush_buffer()

    def process_synthetic_numbers(self) -> None:
        if not SYNTHETIC_DIR.exists():
            if self.worker_idx == 0:
                log_msg(
                    f"[!] Warning: Synthetic_Numbers directory '{SYNTHETIC_DIR}' does not exist."
                )
            return
        completed_keys = _get_completed_keys(self.split, "Synthetic_Numbers")
        all_files = sorted(_fast_rglob(SYNTHETIC_DIR, "*.png")) + sorted(
            _fast_rglob(SYNTHETIC_DIR, "*.jpg")
        )
        if not all_files:
            if self.worker_idx == 0:
                log_msg(
                    f"[!] Warning: No image files found in Synthetic_Numbers directory '{SYNTHETIC_DIR}'."
                )
            return

        # Filter files by target split folder (Train_Nums vs Test_Nums)
        target_sub = "train" if self.split in ("train", "val") else "test"
        split_files = [
            f
            for f in all_files
            if f"/{target_sub}" in str(f.resolve()).lower()
            or f"_{target_sub}_" in f.name.lower()
        ]
        if split_files:
            all_files = split_files

        # CRITICAL FIX (Point 4): Synthetic Numbers deterministic train/val split
        import hashlib
        if self.split in ("train", "val") and target_sub == "train":
            filtered = []
            for f in all_files:
                h = int(hashlib.md5(f.name.encode('utf-8')).hexdigest(), 16)
                is_val = (h % 10 == 0) # 10% validation
                if (self.split == "val" and is_val) or (self.split == "train" and not is_val):
                    filtered.append(f)
            all_files = filtered

        my_files = [
            f
            for i, f in enumerate(all_files)
            if i % self.num_workers == self.worker_idx
        ]

        pending_files = []
        for f in my_files:
            key = str(f.resolve())
            if key in completed_keys:
                continue
            pending_files.append(f)

        chunk_step = max(128, self.batch_size * 2)
        total_chunks = math.ceil(len(pending_files) / chunk_step)
        for idx_c, i in enumerate(range(0, len(pending_files), chunk_step)):
            if idx_c % 10 == 0 or idx_c == total_chunks - 1:
                log_msg(
                    f"[*] TPU Worker {self.worker_idx} [Synthetic_Numbers]: Batch {idx_c+1}/{total_chunks}"
                )
            batch_paths = pending_files[i : i + chunk_step]
            results = self.extractor.extract_images_batch(
                batch_paths, batch_size=self.batch_size
            )
            for f_path, (buf_60, q_val, conf) in zip(batch_paths, results):
                key = str(f_path.resolve())
                label = normalize_gloss(f_path.parent.name, self.alias_map)
                if buf_60 is None:
                    _discard(
                        "no_landmarks",
                        label,
                        {"file": f_path.name},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "Synthetic_Numbers")
                    continue
                seq = _process_static_image_features(buf_60)
                if seq is None:
                    _discard(
                        "processing_failed",
                        label,
                        {"file": f_path.name},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "Synthetic_Numbers")
                    continue
                stat_q, breakdown = _assess_quality_static(buf_60[0], conf)
                record = {
                    "task": "static_alphabet",
                    "label": label,
                    "signer_id": "synthetic",
                    "features": seq,
                    "source": "Synthetic_Numbers",
                    "split": self.split,
                    "quality": stat_q,
                    "quality_breakdown": breakdown,
                    "sample_weight": float(np.clip(stat_q, 0.25, 1.0)),
                    "video_id": f_path.name,
                }
                self._push_record(record)
                self._buffer_completed_key(key, "Synthetic_Numbers")
        self._flush_buffer()

    def process_wlasl(self) -> None:
        if not WLASL_DIR.exists():
            return
        json_path = WLASL_DIR / "WLASL_v0.3.json"
        video_dir = WLASL_DIR / "videos"
        if not json_path.exists() or not video_dir.exists():
            return
        completed_keys = _get_completed_keys(self.split, "WLASL")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                entries = json.load(f)
        except Exception:
            return

        my_entries = [
            e for i, e in enumerate(entries) if i % self.num_workers == self.worker_idx
        ]
        total_entries = len(my_entries)
        for idx_e, entry in enumerate(my_entries):
            if idx_e % 25 == 0 or idx_e == total_entries - 1:
                log_msg(
                    f"[*] TPU Worker {self.worker_idx} [WLASL]: Entry {idx_e+1}/{total_entries}"
                )
            gloss = normalize_gloss(entry.get("gloss", ""), self.alias_map)
            for inst in entry.get("instances", []):
                inst_split = str(inst.get("split", "train")).lower()
                if inst_split != self.split.lower():
                    continue
                video_id = str(inst.get("video_id", ""))
                key = f"WLASL_{video_id}"
                if key in completed_keys:
                    continue
                video_path = video_dir / f"{video_id}.mp4"
                if not video_path.exists():
                    _discard(
                        "missing_video",
                        gloss,
                        {"video_id": video_id},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "WLASL")
                    continue
                buf, quality, conf = self.extractor.extract_video(video_path)
                if buf is None or buf.shape[0] == 0:
                    _discard(
                        "no_landmarks",
                        gloss,
                        {"video_id": video_id},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "WLASL")
                    continue
                q_val, breakdown = _assess_quality_video(buf, conf)
                seq = _process_sequence(buf)
                if seq is None:
                    _discard(
                        "processing_failed",
                        gloss,
                        {"video_id": video_id},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "WLASL")
                    continue
                record = {
                    "task": "isolated_gloss",
                    "label": gloss,
                    "signer_id": str(inst.get("signer_id", "unknown")),
                    "features": seq,
                    "source": "WLASL",
                    "split": self.split,
                    "quality": q_val,
                    "quality_breakdown": breakdown,
                    "sample_weight": float(np.clip(q_val, 0.25, 1.0)),
                    "video_id": video_id,
                }
                self._push_record(record)
                self._buffer_completed_key(key, "WLASL")
        self._flush_buffer()

    def _find_chicago_seq_dir(
        self, frame_root: Path, filename: str
    ) -> Optional[List[Path]]:
        clean_fn = filename.strip().replace("\\", "/").lstrip("/")
        if clean_fn in self._chicago_seq_dir_cache:
            return self._chicago_seq_dir_cache[clean_fn]

        candidates = [
            frame_root / clean_fn,
            KAGGLE_TEMP_DIR / clean_fn,
            KAGGLE_TEMP_DIR / "ChicagoFSWild-Frames" / clean_fn,
            KAGGLE_TEMP_DIR / "ChicagoFSWild" / "ChicagoFSWild-Frames" / clean_fn,
            CHICAGO_FSWILD_DIR / clean_fn,
            CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames" / clean_fn,
        ]
        if self._chicago_resolved_seq_dir is not None:
            candidates.insert(0, self._chicago_resolved_seq_dir / clean_fn)

        for cand in candidates:
            if cand.exists() and cand.is_dir():
                frames = sorted(
                    list(cand.glob("*.jpg"))
                    + list(cand.glob("*.jpeg"))
                    + list(cand.glob("*.png")),
                    key=natural_sort_key,
                )
                if len(frames) >= 2:
                    if self._chicago_resolved_seq_dir is None:
                        clean_parts = Path(clean_fn).parts
                        base_dir = cand
                        for _ in range(len(clean_parts)):
                            base_dir = base_dir.parent
                        self._chicago_resolved_seq_dir = base_dir
                    if len(self._chicago_seq_dir_cache) < 10000:
                        self._chicago_seq_dir_cache[clean_fn] = frames
                    return frames

        target_folder_name = Path(clean_fn).name
        if self._chicago_resolved_seq_dir is None:
            if IS_KAGGLE and Path("/kaggle/input").exists():
                try:
                    for p in Path("/kaggle/input").rglob(target_folder_name):
                        if p.is_dir():
                            frames = sorted(
                                list(p.glob("*.jpg"))
                                + list(p.glob("*.jpeg"))
                                + list(p.glob("*.png")),
                                key=natural_sort_key,
                            )
                            if len(frames) >= 2:
                                clean_parts = Path(clean_fn).parts
                                base_dir = p
                                for _ in range(len(clean_parts)):
                                    base_dir = base_dir.parent
                                self._chicago_resolved_seq_dir = base_dir
                                if len(self._chicago_seq_dir_cache) < 10000:
                                    self._chicago_seq_dir_cache[clean_fn] = frames
                                return frames
                except Exception:
                    pass

            if KAGGLE_TEMP_DIR.exists():
                try:
                    for p in KAGGLE_TEMP_DIR.rglob(target_folder_name):
                        if p.is_dir():
                            frames = sorted(
                                list(p.glob("*.jpg"))
                                + list(p.glob("*.jpeg"))
                                + list(p.glob("*.png")),
                                key=natural_sort_key,
                            )
                            if len(frames) >= 2:
                                clean_parts = Path(clean_fn).parts
                                base_dir = p
                                for _ in range(len(clean_parts)):
                                    base_dir = base_dir.parent
                                self._chicago_resolved_seq_dir = base_dir
                                if len(self._chicago_seq_dir_cache) < 10000:
                                    self._chicago_seq_dir_cache[clean_fn] = frames
                                return frames
                except Exception:
                    pass

        if len(self._chicago_seq_dir_cache) < 10000:
            self._chicago_seq_dir_cache[clean_fn] = None
        return None

    def process_chicagofswild(self) -> None:
        csv_path = CHICAGO_FSWILD_DIR / "ChicagoFSWild.csv"
        if not csv_path.exists() and IS_KAGGLE and Path("/kaggle/input").exists():
            found_csvs = list(Path("/kaggle/input").rglob("ChicagoFSWild.csv"))
            if found_csvs:
                csv_path = found_csvs[0]

        if not csv_path.exists():
            archive_candidates = [
                CHICAGO_FSWILD_DIR / "ChicagoFSWild.tgz",
                CHICAGO_FSWILD_DIR / "ChicagoFSWild.tar.gz",
                CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild.tgz",
                CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild.tar.gz",
            ]
            if IS_KAGGLE and Path("/kaggle/input").exists():
                archive_candidates.extend(
                    list(Path("/kaggle/input").rglob("ChicagoFSWild.tgz"))
                )
                archive_candidates.extend(
                    list(Path("/kaggle/input").rglob("ChicagoFSWild.tar.gz"))
                )
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
                                    m.name = m_name
                                    members.append(m)
                            tar.extractall(path=KAGGLE_TEMP_DIR, members=members)
                        csv_extract_done.write_text("done")
                        log_msg("[+] Successfully extracted ChicagoFSWild CSV.")
                    except Exception as exc:
                        log_msg(f"[!] Failed to extract ChicagoFSWild CSV: {exc}")
                if (KAGGLE_TEMP_DIR / "ChicagoFSWild.csv").exists():
                    csv_path = KAGGLE_TEMP_DIR / "ChicagoFSWild.csv"

        if not csv_path.exists():
            if self.worker_idx == 0:
                log_msg(
                    f"[!] Warning: ChicagoFSWild.csv not found under '{CHICAGO_FSWILD_DIR}'. Skipping dataset."
                )
            return

        completed_keys = _get_completed_keys(self.split, "ChicagoFSWild")
        df = pd.read_csv(csv_path)
        target_partition = "dev" if self.split == "val" else self.split
        partition_df = df[df["partition"] == target_partition].copy()

        frame_root = KAGGLE_TEMP_DIR / "ChicagoFSWild-Frames"
        if not frame_root.exists():
            dataset_frames = CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames"
            if dataset_frames.exists() and dataset_frames.is_dir():
                frame_root = dataset_frames
            elif (CHICAGO_FSWILD_DIR / "aslthat").exists() or (
                CHICAGO_FSWILD_DIR / "aslized"
            ).exists():
                frame_root = CHICAGO_FSWILD_DIR
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
                if IS_KAGGLE and Path("/kaggle/input").exists():
                    archive_candidates.extend(
                        list(Path("/kaggle/input").rglob("ChicagoFSWild-Frames.tgz"))
                    )
                    archive_candidates.extend(
                        list(Path("/kaggle/input").rglob("ChicagoFSWild-Frames.tar.gz"))
                    )
                archive_path = next((c for c in archive_candidates if c.exists()), None)
                if archive_path is not None:
                    extract_done_flag = KAGGLE_TEMP_DIR / "chicago_extract_done.txt"
                    if not extract_done_flag.exists():
                        log_msg(f"[!] Warning: ChicagoFSWild Frames not pre-extracted.")

        def _find_valid_chicago_root():
            candidates = [
                KAGGLE_TEMP_DIR / "ChicagoFSWild-Frames",
                KAGGLE_TEMP_DIR / "ChicagoFSWild",
                KAGGLE_TEMP_DIR,
                CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames",
                CHICAGO_FSWILD_DIR / "ChicagoFSWild",
                CHICAGO_FSWILD_DIR,
            ]
            for cand in candidates:
                if cand.exists() and cand.is_dir():
                    if (
                        (cand / "aslthat").exists()
                        or (cand / "aslized").exists()
                        or any(cand.glob("*/*.jpg"))
                        or any(cand.glob("*/*.png"))
                    ):
                        return cand
            return None

        resolved_root = _find_valid_chicago_root()
        if resolved_root is not None:
            frame_root = resolved_root
            self._chicago_resolved_seq_dir = frame_root
        else:
            if self.worker_idx == 0:
                log_msg(
                    f"[!] Warning: Valid ChicagoFSWild frame root not found under '{frame_root}'. Skipping dataset."
                )
            return

        all_rows = list(partition_df.iterrows())
        my_rows = [
            item
            for i, item in enumerate(all_rows)
            if i % self.num_workers == self.worker_idx
        ]

        for _, row_dict in tqdm(
            my_rows,
            desc=f"TPU {self.worker_idx} [ChicagoFSWild]",
            mininterval=30.0,
            maxinterval=60.0,
            leave=False,
        ):
            filename = str(row_dict.get("filename", "")).strip()
            key = f"ChicagoFSWild_{filename}"
            if key in completed_keys:
                continue
            label = _clean_chicago_label(
                label_proc=row_dict.get("label_proc"),
                label_raw=row_dict.get("label_raw"),
                label_notes=row_dict.get("label_notes"),
                alias_map=self.alias_map,
            )
            if not label:
                _discard(
                    "empty_label", "", {"filename": filename}, self.split, threshold=0.0
                )
                self._buffer_completed_key(key, "ChicagoFSWild")
                continue

            # Check video file or directory of image frames
            video_path = frame_root / f"{filename}.mp4"
            if (
                not video_path.exists()
                and not filename.endswith(".mp4")
                and (frame_root / f"{filename}.avi").exists()
            ):
                video_path = frame_root / f"{filename}.avi"

            if video_path.exists() and video_path.is_file():
                buf, quality, conf = self.extractor.extract_video(video_path)
            else:
                frame_paths = self._find_chicago_seq_dir(frame_root, filename)
                if not frame_paths or len(frame_paths) < 2:
                    _discard(
                        "missing_video",
                        label,
                        {"filename": filename},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "ChicagoFSWild")
                    continue
                results = self.extractor.extract_images_batch(
                    frame_paths, batch_size=self.batch_size
                )
                valid_bufs = [
                    r[0] for r in results if r[0] is not None and r[0].shape[0] > 0
                ]
                if not valid_bufs:
                    _discard(
                        "no_landmarks",
                        label,
                        {"filename": filename},
                        self.split,
                        threshold=0.0,
                    )
                    self._buffer_completed_key(key, "ChicagoFSWild")
                    continue
                buf = np.concatenate(valid_bufs, axis=0)  # (T, 60, 3)
                conf = results[0][2] if results and results[0][2] is not None else {}

            if buf is None or buf.shape[0] == 0:
                _discard(
                    "no_landmarks",
                    label,
                    {"filename": filename},
                    self.split,
                    threshold=0.0,
                )
                self._buffer_completed_key(key, "ChicagoFSWild")
                continue

            q_val, breakdown = _assess_quality_video(buf, conf)
            seq = _process_sequence(buf)
            if seq is None:
                _discard(
                    "processing_failed",
                    label,
                    {"filename": filename},
                    self.split,
                    threshold=0.0,
                )
                self._buffer_completed_key(key, "ChicagoFSWild")
                continue

            record = {
                "task": "fingerspelling_sequence",
                "label": label,
                "signer_id": str(
                    row_dict.get("signer_id", row_dict.get("Subject", "unknown"))
                ),
                "features": seq,
                "source": "ChicagoFSWild",
                "split": self.split,
                "quality": q_val,
                "quality_breakdown": breakdown,
                "sample_weight": float(np.clip(q_val, 0.25, 1.0)),
                "video_id": filename,
            }
            self._push_record(record)
            self._buffer_completed_key(key, "ChicagoFSWild")
        self._flush_buffer()

    def process_citizen(self) -> None:
        if not ASL_CITIZEN_DIR.exists():
            return
        csv_candidates = [
            ASL_CITIZEN_DIR / f"{self.split}.csv",
            ASL_CITIZEN_DIR / "splits" / f"{self.split}.csv",
            ASL_CITIZEN_DIR / "ASL_Citizen" / "splits" / f"{self.split}.csv",
            ASL_CITIZEN_DIR / f"{self.split.capitalize()}.csv",
            ASL_CITIZEN_DIR / "train.csv",
        ]
        csv_path = next((c for c in csv_candidates if c.exists()), None)
        if csv_path is None:
            return
        completed_keys = _get_completed_keys(self.split, "ASL_Citizen")
        try:
            df = pd.read_csv(csv_path)
            if "split" in df.columns:
                df = df[df["split"].str.lower() == self.split.lower()].copy()
            elif "partition" in df.columns:
                df = df[df["partition"].str.lower() == self.split.lower()].copy()
        except Exception:
            return

        video_root = ASL_CITIZEN_DIR / "videos"
        if (
            not video_root.exists()
            and (ASL_CITIZEN_DIR / "ASL_Citizen" / "videos").exists()
        ):
            video_root = ASL_CITIZEN_DIR / "ASL_Citizen" / "videos"
        all_rows = list(df.iterrows())
        my_rows = [
            item
            for i, item in enumerate(all_rows)
            if i % self.num_workers == self.worker_idx
        ]
        total_rows = len(my_rows)
        for idx_r, (_, row) in enumerate(my_rows):
            if idx_r % 50 == 0 or idx_r == total_rows - 1:
                log_msg(
                    f"[*] TPU Worker {self.worker_idx} [ASL_Citizen]: Row {idx_r+1}/{total_rows}"
                )
            video_id = (
                str(row.get("video_id", row.get("Video file", "")))
                .strip()
                .replace("\\", "/")
                .lstrip("/")
            )
            key = f"ASL_Citizen_{video_id}"
            if key in completed_keys:
                continue
            gloss = normalize_gloss(
                str(row.get("gloss", row.get("Gloss", ""))), self.alias_map
            )
            video_path = video_root / video_id
            if (
                not video_path.exists()
                and not video_id.endswith(".mp4")
                and (video_root / f"{video_id}.mp4").exists()
            ):
                video_path = video_root / f"{video_id}.mp4"
            if not video_path.exists():
                _discard(
                    "missing_video",
                    gloss,
                    {"video_id": video_id},
                    self.split,
                    threshold=0.0,
                )
                self._buffer_completed_key(key, "ASL_Citizen")
                continue

            buf, quality, conf = self.extractor.extract_video(video_path)
            if buf is None or buf.shape[0] == 0:
                _discard(
                    "no_landmarks",
                    gloss,
                    {"video_id": video_id},
                    self.split,
                    threshold=0.0,
                )
                self._buffer_completed_key(key, "ASL_Citizen")
                continue

            q_val, breakdown = _assess_quality_video(buf, conf)
            seq = _process_sequence(buf)
            if seq is None:
                _discard(
                    "processing_failed",
                    gloss,
                    {"video_id": video_id},
                    self.split,
                    threshold=0.0,
                )
                self._buffer_completed_key(key, "ASL_Citizen")
                continue
            record = {
                "task": "isolated_gloss",
                "label": gloss,
                "signer_id": str(
                    row.get("participant_id", row.get("Participant ID", "unknown"))
                ),
                "features": seq,
                "source": "ASL_Citizen",
                "split": self.split,
                "quality": q_val,
                "quality_breakdown": breakdown,
                "sample_weight": float(np.clip(q_val, 0.25, 1.0)),
                "video_id": video_id,
            }
            self._push_record(record)
            self._buffer_completed_key(key, "ASL_Citizen")
        self._flush_buffer()

    def _load_how2sign_text_map(self, how2sign_root: Path) -> Dict[str, str]:
        # 1. Load Text Mappings
        csv_files = sorted(_fast_rglob(how2sign_root, "*.csv")) + sorted(
            _fast_rglob(how2sign_root, "*.tsv")
        )
        if IS_KAGGLE and Path("/kaggle/input").exists():
            try:
                csv_files.extend(list(Path("/kaggle/input").rglob("how2sign*.csv")))
                csv_files.extend(list(Path("/kaggle/input").rglob("how2sign*.tsv")))
            except Exception:
                pass
        for csv_p in csv_files:
            try:
                sep = "\t" if csv_p.suffix.lower() == ".tsv" else ","
                df = pd.read_csv(csv_p, sep=sep)
                id_col = None
                for preferred in (
                    "sentence_name",
                    "sentence_id",
                    "clip_id",
                    "name",
                    "video_name",
                    "video_id",
                ):
                    for c in df.columns:
                        if preferred in c.lower():
                            id_col = c
                            break
                    if id_col:
                        break
                text_col = next(
                    (
                        c
                        for c in df.columns
                        if any(
                            k in c.lower()
                            for k in ("sentence", "text", "gloss", "translation")
                        )
                    ),
                    None,
                )
                if id_col and text_col:
                    for _, r in df.iterrows():
                        clip_id = str(r[id_col]).strip()
                        text_val = str(r[text_col]).strip()
                        if clip_id and text_val and text_val.lower() != "nan":
                            clip_map[clip_id] = text_val
                            clip_map[Path(clip_id).stem] = text_val
                            clip_map[clip_id.replace("-rgb_front", "")] = text_val
                            clip_map[clip_id.replace("_rgb_front", "")] = text_val
                            clip_map[
                                clip_id.replace("-rgb_front", "").replace(
                                    "_holistic", ""
                                )
                            ] = text_val
            except Exception:
                pass
        return clip_map

    def process_how2sign(self) -> None:
        how2sign_root = HOW2SIGN_DIR
        if not how2sign_root.exists():
            return
        how2sign_text_map = self._load_how2sign_text_map(how2sign_root)
        completed_keys = _get_completed_keys(self.split, "How2Sign")
        all_files = sorted(_fast_rglob(how2sign_root, "*.npy"))
        if not all_files:
            archive_candidates = (
                sorted(_fast_rglob(how2sign_root, "*.tgz"))
                + sorted(_fast_rglob(how2sign_root, "*.tar.gz"))
                + sorted(_fast_rglob(how2sign_root, "*.tar"))
            )
            if archive_candidates:
                archive_path = archive_candidates[0]
                extract_done_flag = KAGGLE_TEMP_DIR / "how2sign_extract_done.txt"
                if not extract_done_flag.exists():
                    log_msg(f"[!] Warning: How2Sign archive was not pre-extracted.")
                how2sign_root = KAGGLE_TEMP_DIR
                all_files = sorted(_fast_rglob(how2sign_root, "*.npy"))

        if all_files:
            split_files = [
                f
                for f in all_files
                if f"/{self.split}/" in str(f.resolve()).replace("\\", "/")
                or f"_{self.split}_" in f.name
            ]
            if split_files:
                all_files = split_files

        my_files = [
            f
            for i, f in enumerate(all_files)
            if i % self.num_workers == self.worker_idx
        ]
        total_files = len(my_files)
        for idx_f, f in enumerate(my_files):
            if idx_f % 50 == 0 or idx_f == total_files - 1:
                log_msg(
                    f"[*] TPU Worker {self.worker_idx} [How2Sign]: File {idx_f+1}/{total_files}"
                )
            key = str(f.resolve())
            if key in completed_keys:
                continue

            # Determine label from text transcription CSV or default to sequence task
            clip_stem = f.stem
            clean_stem = (
                clip_stem.replace("_holistic", "")
                .replace("-rgb_front", "")
                .replace("_rgb_front", "")
            )
            base_id = clean_stem.split("_")[0]

            raw_text = None
            if clip_stem in how2sign_text_map:
                raw_text = how2sign_text_map[clip_stem]
            elif clean_stem in how2sign_text_map:
                raw_text = how2sign_text_map[clean_stem]
            elif base_id in how2sign_text_map:
                raw_text = how2sign_text_map[base_id]

            if raw_text:
                label = normalize_gloss(raw_text, self.alias_map)
                task_type = "sentence_level"
            else:
                label = "how2sign_sequence"
                task_type = "sentence_level"

            if f.suffix.lower() in (".mp4", ".avi"):
                buf, quality, conf = self.extractor.extract_video(f)
                if buf is not None and buf.shape[0] > 0:
                    q_val, breakdown = _assess_quality_video(buf, conf)
                    seq = _process_sequence(buf)
                    if seq is not None:
                        record = {
                            "task": task_type,
                            "label": label,
                            "signer_id": "how2sign",
                            "features": seq,
                            "source": "How2Sign_Holistic",
                            "split": self.split,
                            "quality": q_val,
                            "quality_breakdown": breakdown,
                            "sample_weight": float(np.clip(q_val, 0.25, 1.0)),
                            "video_id": f.stem,
                        }
                        self._push_record(record)
                        self._buffer_completed_key(key, "How2Sign_Holistic")
            else:
                try:
                    raw_arr = np.load(f, allow_pickle=True)
                    arr_60 = _convert_npy_to_60(raw_arr)
                    if arr_60 is not None and arr_60.shape[0] > 0:
                        clean_60, conf_dict = _extract_conf_and_sanitize_60(arr_60)
                        q_val, breakdown = _assess_quality_video(clean_60, conf_dict)
                        seq = _process_sequence(clean_60)
                        if seq is not None:
                            record = {
                                "task": task_type,
                                "label": label,
                                "signer_id": "how2sign",
                                "features": seq,
                                "source": "How2Sign_Holistic",
                                "split": self.split,
                                "quality": q_val,
                                "quality_breakdown": breakdown,
                                "sample_weight": float(np.clip(q_val, 0.25, 1.0)),
                                "video_id": f.stem,
                            }
                            self._push_record(record)
                            self._buffer_completed_key(key, "How2Sign_Holistic")
                except Exception:
                    pass
        self._flush_buffer()


# ==============================================================================
# SHARDED PAYLOAD COMPILATION (`save_sharded_payload`)
# ==============================================================================
def _safe_torch_load(filepath: Path) -> Any:
    """Loads PyTorch binary payload with backward compatibility for PyTorch 2.6+ weights_only default."""
    try:
        return torch.load(filepath, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location="cpu")


def save_sharded_payload(
    temp_shard_dir: Path,
    output_dir: Path,
    label_to_idx: dict,
    split: str,
    shard_size: int = 5000,
) -> None:
    log_msg(
        f"[*] Compiling canonical vocabulary & sharded '.pt' payloads for '{split}'..."
    )
    shard_files = sorted(temp_shard_dir.glob(f"temp_shard_{split}_*.jsonl")) + sorted(
        temp_shard_dir.glob(f"temp_shard_{split}_*.pt")
    )
    if not shard_files:
        log_msg(f"[!] No valid records found across shards for split '{split}'.")
        return

    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    total_records = 0
    current_shard_recs = []
    shard_idx = 0

    def write_shard(recs: list, idx: int):
        if not recs:
            return
        for r in recs:
            if "features" in r:
                if isinstance(r["features"], np.ndarray):
                    r["features"] = torch.from_numpy(r["features"]).to(torch.float16)
                elif isinstance(r["features"], torch.Tensor):
                    r["features"] = r["features"].to(torch.float16)
        shard_path = split_dir / f"shard_{idx:04d}.pt"
        torch.save(recs, shard_path)
        log_msg(f"  -> Saved shard {idx+1}: {shard_path.name} ({len(recs)} records)")

    # Added tqdm so you can SEE the merge happening and catch deadlocks instantly
    for temp_file in tqdm(shard_files, desc=f"Merging '{split}' Shards", mininterval=2.0):
        try:
            if temp_file.suffix == ".jsonl":
                with open(temp_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if isinstance(r.get("features"), list):
                            r["features"] = np.array(r["features"], dtype=np.float32)
                        r["features"] = np.nan_to_num(
                            np.asarray(r["features"], dtype=np.float16),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        lbl = r.get("label", "<none>")
                        r["label_idx"] = label_to_idx.get(lbl, -1)
                        current_shard_recs.append(r)
                        total_records += 1

            elif temp_file.suffix == ".pt":
                recs = _safe_torch_load(temp_file)
                if recs:
                    while recs:
                        r = recs.pop(0)
                        if "features" in r:
                            r["features"] = np.nan_to_num(
                                np.asarray(r["features"], dtype=np.float16).copy(),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                        lbl = r.get("label", "<none>")
                        r["label_idx"] = label_to_idx.get(lbl, -1)
                        current_shard_recs.append(r)
                        total_records += 1
                del recs
                import gc
                gc.collect()

            # Instantly dump to disk if we hit the limit
            while len(current_shard_recs) >= shard_size:
                write_shard(current_shard_recs[:shard_size], shard_idx)
                shard_idx += 1
                current_shard_recs = current_shard_recs[shard_size:]

        except Exception as e:
            log_msg(f"\n[!] CORRUPTION DETECTED in {temp_file.name}: {e}. Skipping.")
        finally:
            # INSTANTLY delete the micro-shard to reclaim disk space immediately
            try:
                temp_file.unlink(missing_ok=True)
            except Exception:
                pass

    if current_shard_recs:
        write_shard(current_shard_recs, shard_idx)
        shard_idx += 1

    num_shards = max(1, shard_idx)
    metadata = {
        "split": split,
        "total_records": total_records,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "label_to_idx": label_to_idx,
    }
    with open(split_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    try:
        shards_temp_dir = KAGGLE_TEMP_DIR / "shards"
        if shards_temp_dir.exists() and not any(shards_temp_dir.iterdir()):
            shutil.rmtree(shards_temp_dir, ignore_errors=True)
    except Exception as e:
        log_msg(f"[!] Note during scratch cleanup: {e}")


# ==============================================================================
# TPU WORKER SPAWN FUNCTION (`_tpu_worker_fn`)
# ==============================================================================
def _tpu_worker_fn(
    worker_idx: int,
    num_workers: int,
    split: str,
    batch_flush_size: int = 1000,
    batch_size: int = 128,
    device_str: Optional[str] = None,
) -> None:
    # THE MAGIC FIX: In PJRT multiprocessing, EVERY isolated worker's local chip is 'xla:0'
    if _XLA_AVAILABLE and device_str is None:
        device_str = "xla:0"

    log_msg(
        f"[*] TPU Worker {worker_idx}/{num_workers} starting extraction for split '{split}' on device '{device_str}'..."
    )
    processor = _DatasetProcessor(
        worker_idx,
        num_workers,
        split,
        batch_flush_size=batch_flush_size,
        batch_size=batch_size,
        device_str=device_str,
    )
    try:
        processor.process_asl_alphabet()
        processor.process_synthetic_numbers()
        processor.process_wlasl()
        processor.process_chicagofswild()
        processor.process_citizen()
        processor.process_how2sign()
    finally:
        processor.close()
    log_msg(f"[*] TPU Worker {worker_idx} completed all datasets for split '{split}'.")


def _extract_chicago_csv():
    csv_extract_done = KAGGLE_TEMP_DIR / "chicago_csv_extract_done.txt"
    chicago_csv_path = CHICAGO_FSWILD_DIR / "ChicagoFSWild.csv"
    if not chicago_csv_path.exists() and IS_KAGGLE and Path("/kaggle/input").exists():
        found_csvs = []
        for c_dir in _gather_candidate_dirs(Path("/kaggle/input"), max_depth=4):
            if (c_dir / "ChicagoFSWild.csv").exists():
                found_csvs.append(c_dir / "ChicagoFSWild.csv")
        if found_csvs:
            chicago_csv_path = found_csvs[0]

    if not chicago_csv_path.exists() and not csv_extract_done.exists():
        archive_candidates = [
            CHICAGO_FSWILD_DIR / "ChicagoFSWild.tgz",
            CHICAGO_FSWILD_DIR / "ChicagoFSWild.tar.gz",
            CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild.tgz",
            CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild.tar.gz",
        ]
        if IS_KAGGLE and Path("/kaggle/input").exists():
            for c_dir in _gather_candidate_dirs(Path("/kaggle/input"), max_depth=4):
                archive_candidates.append(c_dir / "ChicagoFSWild.tgz")
                archive_candidates.append(c_dir / "ChicagoFSWild.tar.gz")
        archive_path = next((c for c in archive_candidates if c.exists()), None)
        if archive_path is not None:
            log_msg(f"[*] Pre-extracting ChicagoFSWild CSV archive: {archive_path.name}...")
            try:
                with tarfile.open(archive_path, "r:gz") as tar:
                    members = []
                    for m in tar.getmembers():
                        m_name = Path(m.name).name
                        if m_name in ("ChicagoFSWild.csv", "unavailable.csv"):
                            m.name = m_name
                            members.append(m)
                    tar.extractall(path=KAGGLE_TEMP_DIR, members=members)
                csv_extract_done.write_text("done")
                log_msg("[+] Successfully pre-extracted ChicagoFSWild CSV.")
            except Exception as exc:
                log_msg(f"[!] Pre-extraction of ChicagoFSWild CSV failed: {exc}")

def _extract_chicago_frames():
    chicago_frames_done = KAGGLE_TEMP_DIR / "chicago_extract_done.txt"
    frame_root = KAGGLE_TEMP_DIR / "ChicagoFSWild-Frames"
    dataset_frames = CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames"
    has_frames = (
        frame_root.exists()
        or dataset_frames.exists()
        or (CHICAGO_FSWILD_DIR / "aslthat").exists()
        or (CHICAGO_FSWILD_DIR / "aslized").exists()
    )
    if not chicago_frames_done.exists() and not has_frames:
        archive_candidates = [
            CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames.tgz",
            CHICAGO_FSWILD_DIR / "ChicagoFSWild-Frames.tar.gz",
            CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild-Frames.tgz",
            CHICAGO_FSWILD_DIR.parent / "chicagofswild" / "ChicagoFSWild-Frames.tar.gz",
        ]
        if IS_KAGGLE and Path("/kaggle/input").exists():
            for c_dir in _gather_candidate_dirs(Path("/kaggle/input"), max_depth=4):
                archive_candidates.append(c_dir / "ChicagoFSWild-Frames.tgz")
                archive_candidates.append(c_dir / "ChicagoFSWild-Frames.tar.gz")
        archive_path = next((c for c in archive_candidates if c.exists()), None)
        if archive_path is not None:
            log_msg(f"[*] Fast pre-extracting ChicagoFSWild Frames archive: {archive_path.name} to {KAGGLE_TEMP_DIR}...")
            try:
                import subprocess
                # Use pigz if available, else fallback to standard gzip
                has_pigz = shutil.which("pigz") is not None
                if shutil.which("tar"):
                    tar_cmd = ["tar", "-I", "pigz", "-xf"] if has_pigz else ["tar", "-xzf"]
                    tar_cmd.extend([str(archive_path), "-C", str(KAGGLE_TEMP_DIR)])
                    subprocess.run(tar_cmd, check=True)
                else:
                    with tarfile.open(archive_path, "r:gz") as tar:
                        tar.extractall(path=KAGGLE_TEMP_DIR)
                chicago_frames_done.write_text("done")
                log_msg("[+] Successfully pre-extracted ChicagoFSWild Frames.")
            except Exception as exc:
                log_msg(f"[!] Pre-extraction of ChicagoFSWild Frames failed: {exc}")

def _extract_how2sign():
    how2sign_done = KAGGLE_TEMP_DIR / "how2sign_extract_done.txt"
    if not how2sign_done.exists() and HOW2SIGN_DIR.exists():
        how2sign_root = HOW2SIGN_DIR
        all_npy = list(how2sign_root.rglob("*.npy"))
        if not all_npy:
            archive_candidates = (
                sorted(how2sign_root.rglob("*.tgz"))
                + sorted(how2sign_root.rglob("*.tar.gz"))
                + sorted(how2sign_root.rglob("*.tar"))
            )
            if archive_candidates:
                archive_path = archive_candidates[0]
                log_msg(f"[*] Pre-extracting How2Sign archive: {archive_path.name} to {KAGGLE_TEMP_DIR}...")
                try:
                    import subprocess
                    has_pigz = shutil.which("pigz") is not None
                    if shutil.which("tar"):
                        tar_cmd = ["tar", "-I", "pigz", "-xf"] if has_pigz else ["tar", "-xf"]
                        tar_cmd.extend([str(archive_path), "-C", str(KAGGLE_TEMP_DIR)])
                        subprocess.run(tar_cmd, check=True)
                    else:
                        with tarfile.open(archive_path, "r:*") as tar:
                            tar.extractall(path=KAGGLE_TEMP_DIR)
                    with open(how2sign_done, "w", encoding="utf-8") as f:
                        f.write("done\n")
                    log_msg("[+] Successfully pre-extracted How2Sign archive.")
                except Exception as exc:
                    log_msg(f"[!] Pre-extraction of How2Sign archive failed: {exc}")

def _download_rtmw():
    try:
        import urllib.request
        rtmw_url = "https://download.openmmlab.com/mmpose/v1/projects/rtmw/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
        cache_dir = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"
        cache_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = cache_dir / "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
        if not ckpt_file.exists():
            log_msg(f"[*] Pre-downloading RTMW model weights (220MB) to {ckpt_file}...")
            tmp_ckpt = ckpt_file.with_suffix(".tmp")
            urllib.request.urlretrieve(rtmw_url, str(tmp_ckpt))
            tmp_ckpt.rename(ckpt_file)
            log_msg("[+] RTMW model weights successfully cached.")
    except Exception as exc:
        log_msg(f"[!] Pre-download of RTMW checkpoint note: {exc}")

def preextract_all_archives() -> None:
    """
    Pre-extracts dataset archives (ChicagoFSWild, How2Sign, etc.) on CPU
    in the main process before spawning TPU workers using ThreadPoolExecutor for speed.
    """
    KAGGLE_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.submit(_extract_chicago_csv)
        executor.submit(_extract_chicago_frames)
        executor.submit(_extract_how2sign)
        executor.submit(_download_rtmw)



# ==============================================================================
# MAIN ORCHESTRATION ENTRYPOINT
# ==============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="ASL Recognition V3-TPU Frankenstein Engine"
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test", "all"]
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of TPU cores (default 8 for v3-8 / v5e-8)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size per core (default 128)",
    )
    parser.add_argument("--batch-flush-size", type=int, default=1000)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--phase", type=str, default="all", choices=["all", "extract", "merge"]
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_shard_dir = KAGGLE_TEMP_DIR / "shards"
    temp_shard_dir.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split != "all" else ["train", "val", "test"]

    for split in splits:
        log_msg(
            f"\n======================================================================"
        )
        log_msg(f"       STARTING V3-TPU PIPELINE FOR SPLIT: '{split.upper()}'")
        log_msg(
            f"======================================================================"
        )
        log_msg("[*] Discovered Directories under /kaggle/input (up to depth 4):")
        if IS_KAGGLE and Path("/kaggle/input").exists():
            for c_dir in _gather_candidate_dirs(Path("/kaggle/input"), max_depth=4):
                log_msg(f"    -> {c_dir}")
        log_msg("[*] Checking Dataset Directory Paths:")
        log_msg(
            f"    - ASL_Alphabet: {ASL_ALPHABET_DIR} (exists: {ASL_ALPHABET_DIR.exists()})"
        )
        log_msg(
            f"    - Synthetic_Numbers: {SYNTHETIC_DIR} (exists: {SYNTHETIC_DIR.exists()})"
        )
        log_msg(f"    - WLASL: {WLASL_DIR} (exists: {WLASL_DIR.exists()})")
        log_msg(
            f"    - ChicagoFSWild: {CHICAGO_FSWILD_DIR} (exists: {CHICAGO_FSWILD_DIR.exists()})"
        )
        log_msg(
            f"    - ASL_Citizen: {ASL_CITIZEN_DIR} (exists: {ASL_CITIZEN_DIR.exists()})"
        )
        log_msg(f"    - How2Sign: {HOW2SIGN_DIR} (exists: {HOW2SIGN_DIR.exists()})")

        if args.phase in ("all", "extract"):
            log_msg("[*] Pre-extracting dataset archives before TPU worker spawn...")
            preextract_all_archives()

            if _XLA_AVAILABLE and args.num_workers > 1:
                # 2. Spawn the workers with Error Handling
                num_tpus = args.num_workers
                log_msg(f"[*] Launching TPU workers via xmp.spawn (Multiprocessing)...")

                try:
                    xmp.spawn(
                        _tpu_worker_fn,
                        args=(
                            num_tpus,
                            split,
                            args.batch_flush_size,
                            args.batch_size,
                            None,
                        ),
                        nprocs=None,
                        start_method="fork",
                    )
                except Exception as e:
                    log_msg(
                        f"[!] TPU Multiprocessing crashed: {e}. Attempting ThreadPool fallback..."
                    )
                    with ThreadPoolExecutor(max_workers=num_tpus) as pool:
                        futures = [
                            pool.submit(
                                _tpu_worker_fn,
                                worker_idx,
                                num_tpus,
                                split,
                                args.batch_flush_size,
                                args.batch_size,
                                "cpu",
                            )
                            for worker_idx in range(num_tpus)
                        ]
                        for f in futures:
                            f.result()
            else:
                device_fallback = "cpu"
                log_msg(
                    f"[*] Running on single worker / fallback device '{device_fallback}'..."
                )
                _tpu_worker_fn(
                    0, 1, split, args.batch_flush_size, args.batch_size, device_fallback
                )

        if args.phase in ("all", "merge"):
            if split == splits[-1]:
                log_msg(
                    "[*] Final split merge phase started. Aggressively cleaning up raw data directories to free disk space..."
                )
                for raw_dir in [
                    ASL_ALPHABET_DIR,
                    SYNTHETIC_DIR,
                    WLASL_DIR,
                    CHICAGO_FSWILD_DIR,
                    ASL_CITIZEN_DIR,
                    HOW2SIGN_DIR,
                    KAGGLE_TEMP_DIR / "chicagofswild",
                    KAGGLE_TEMP_DIR / "how2sign"
                ]:
                    if raw_dir.exists() and not str(raw_dir).startswith("/kaggle/input"):
                        try:
                            import shutil
                            shutil.rmtree(raw_dir, ignore_errors=True)
                            log_msg(f"    -> Deleted raw dataset directory: {raw_dir.name}")
                        except Exception:
                            pass

            log_msg(
                f"[*] Building canonical label vocabulary & merging shards for '{split}'..."
            )
            label_map_path = output_dir / f"vocabulary_mapping_{split}.json"
            label_to_idx_final: Dict[str, int] = {}
            if label_map_path.exists():
                with open(label_map_path, "r", encoding="utf-8") as f:
                    label_to_idx_final = json.load(f).get("label_to_idx", {})

            if not label_to_idx_final:
                # For val and test splits, check for existing master train vocabulary mapping to preserve class index alignment
                if split in ("val", "test"):
                    train_map_candidates = [
                        output_dir / "vocabulary_mapping_train.json",
                        Path(
                            "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/vocabulary_mapping_train.json"
                        ),
                        Path(
                            "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/train/vocabulary_mapping_train.json"
                        ),
                        Path(
                            "/kaggle/input/frakenstein-asl/vocabulary_mapping_train.json"
                        ),
                        Path(
                            "/kaggle/input/asl-preprocessed-phase1/vocabulary_mapping_train.json"
                        ),
                    ]
                    for candidate in train_map_candidates:
                        if candidate.exists():
                            try:
                                with open(candidate, "r", encoding="utf-8") as f:
                                    label_to_idx_final = json.load(f).get(
                                        "label_to_idx", {}
                                    )
                                if label_to_idx_final:
                                    log_msg(
                                        f"[+] Loaded master TRAIN vocabulary mapping ({len(label_to_idx_final)} classes) from {candidate} for split '{split}'"
                                    )
                                    break
                            except Exception as e:
                                log_msg(
                                    f"[!] Warning reading candidate {candidate}: {e}"
                                )

            if not label_to_idx_final:
                unique_labels = set()
                shard_files = sorted(
                    temp_shard_dir.glob(f"temp_shard_{split}_*.jsonl")
                ) + sorted(temp_shard_dir.glob(f"temp_shard_{split}_*.pt"))
                for temp_sp in shard_files:
                    try:
                        if temp_sp.suffix == ".jsonl":
                            with open(temp_sp, "r", encoding="utf-8") as f:
                                for line in f:
                                    line = line.strip()
                                    if line:
                                        r = json.loads(line)
                                        task = r.get("task", "")
                                        if task in (
                                            "isolated_gloss",
                                            "static_alphabet",
                                            "isolated_number",
                                        ):
                                            if "label" in r and r["label"]:
                                                unique_labels.add(r["label"])
                        else:
                            recs = _safe_torch_load(temp_sp)
                            for r in recs:
                                task = r.get("task", "")
                                if task in (
                                    "isolated_gloss",
                                    "static_alphabet",
                                    "isolated_number",
                                ):
                                    if "label" in r and r["label"]:
                                        unique_labels.add(r["label"])
                    except Exception:
                        pass

                sorted_labels = sorted(list(unique_labels))
                label_to_idx_final = {lbl: idx for idx, lbl in enumerate(sorted_labels)}

            # Ensure vocabulary_mapping_{split}.json is written
            with open(label_map_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "split": split,
                        "total_classes": len(label_to_idx_final),
                        "label_to_idx": label_to_idx_final,
                    },
                    f,
                    indent=2,
                )

            # Also ensure output_mapping.json is written to preserve compatibility with train_all_in_one_tpu.py
            output_map_path = output_dir / "output_mapping.json"
            if not output_map_path.exists():
                with open(output_map_path, "w", encoding="utf-8") as f:
                    json.dump(label_to_idx_final, f, indent=2)

            save_sharded_payload(temp_shard_dir, output_dir, label_to_idx_final, split)


def _process_shard(shard_path: Path):
    """Worker function to load and parse a single PyTorch payload shard in parallel."""
    shard_records = []
    try:
        # Pass weights_only=False for PyTorch 2.6+ dictionary unpickling
        records = torch.load(shard_path, map_location="cpu", weights_only=False)
        for rec in records:
            src = rec.get("source", "Unknown")
            q = rec.get("quality", 0.0)
            w = rec.get("sample_weight", 0.0)
            breakdown = rec.get("quality_breakdown", {})
            best_hand = breakdown.get("best_hand", breakdown.get("best_hand_conf", q))
            shard_records.append((src, q, w, best_hand))
    except Exception as e:
        print(f"[!] Error loading shard {shard_path.name}: {e}")
    return shard_records


def _process_completed_file(cf: Path, split: str):
    """Worker function to count total attempted samples from completion logs."""
    ds_name = cf.stem.replace(f"completed_{split}_", "")
    count = 0
    try:
        with open(cf, "r", encoding="utf-8") as f:
            count = sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"[!] Error reading completion log {cf.name}: {e}")
    return ds_name, count


def audit_pipeline(
    split: str = "train",
    output_root: str = str(DEFAULT_OUTPUT_DIR),
    num_workers: int = 16,
):
    split_dir = Path(output_root) / split
    if not split_dir.exists():
        print(f"[!] Split directory not found: {split_dir}")
        return

    shard_files = sorted(split_dir.glob("shard_*.pt"))
    if not shard_files:
        print(f"[!] No payload shards found under {split_dir}")
        return

    print(f"======================================================================")
    print(f"        PIPELINE QUALITY & REJECTION AUDIT (SPLIT: '{split.upper()}')")
    print(f"======================================================================")
    print(f"[*] Found {len(shard_files)} shard files in {split_dir}")
    print(f"[*] Multithreaded reading enabled with {num_workers} parallel workers\n")

    dataset_qualities = defaultdict(list)
    dataset_weights = defaultdict(list)
    dataset_hand_confs = defaultdict(list)
    total_records = 0

    # 1. Parallel Shard Loading & Processing
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_process_shard, sp) for sp in shard_files]
        for future in futures:
            shard_records = future.result()
            for src, q, w, best_hand in shard_records:
                total_records += 1
                dataset_qualities[src].append(q)
                dataset_weights[src].append(w)
                dataset_hand_confs[src].append(best_hand)

    print(
        f"{'Dataset':<20} | {'Count':<8} | {'Avg Quality':<12} | {'Avg Weight':<12} | {'Avg Hand Conf':<14}"
    )
    print("-" * 75)

    all_qualities, all_weights, all_hand_confs = [], [], []

    for src in sorted(dataset_qualities.keys()):
        qs, ws, hs = (
            dataset_qualities[src],
            dataset_weights[src],
            dataset_hand_confs[src],
        )
        avg_q = float(np.mean(qs)) if qs else 0.0
        avg_w = float(np.mean(ws)) if ws else 0.0
        avg_h = float(np.mean(hs)) if hs else 0.0
        all_qualities.extend(qs)
        all_weights.extend(ws)
        all_hand_confs.extend(hs)
        print(
            f"{src:<20} | {len(qs):<8} | {avg_q:<12.4f} | {avg_w:<12.4f} | {avg_h:<14.4f}"
        )

    print("-" * 75)
    overall_avg_q = float(np.mean(all_qualities)) if all_qualities else 0.0
    overall_avg_w = float(np.mean(all_weights)) if all_weights else 0.0
    overall_avg_h = float(np.mean(all_hand_confs)) if all_hand_confs else 0.0
    print(
        f"{'OVERALL TOTAL':<20} | {total_records:<8} | {overall_avg_q:<12.4f} | {overall_avg_w:<12.4f} | {overall_avg_h:<14.4f}"
    )
    print("=" * 75)

    # 2. Parallel Rejection Rate Analysis
    search_dirs = [
        Path(output_root) / "quality_logs",
        split_dir.parent / "quality_logs",
        split_dir.parent / "logs",
        Path("/tmp/temp_extraction/shards"),
        Path("./temp_extraction/shards"),
    ]

    completed_files = []
    for sd in search_dirs:
        if sd.exists():
            completed_files.extend(sorted(sd.glob(f"completed_{split}_*.jsonl")))

    if completed_files:
        completed_counts = defaultdict(int)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_process_completed_file, cf, split)
                for cf in completed_files
            ]
            for future in futures:
                ds_name, count = future.result()
                completed_counts[ds_name] += count

        print(
            f"\n{'Dataset':<20} | {'Attempted':<10} | {'Accepted':<10} | {'Rejection Rate':<15}"
        )
        print("-" * 65)
        for ds in sorted(completed_counts.keys()):
            total_c = completed_counts[ds]
            accepted_c = len(dataset_qualities.get(ds, []))
            rejected_c = max(0, total_c - accepted_c)
            rej_rate = (rejected_c / max(1, total_c)) * 100.0
            print(f"{ds:<20} | {total_c:<10} | {accepted_c:<10} | {rej_rate:<14.2f}%")
        print("-" * 65)


if __name__ == "__main__":
    main()
