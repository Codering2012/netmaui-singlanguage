#!/usr/bin/env python3
"""
================================================================================
ASL V3 HIGH-THROUGHPUT DATASET & MULTI-TIER COLLATION ENGINE
================================================================================
Upgrades the dataset pipeline to deliver full multi-tier multimodal samples:
- 60-Keypoint 9D Kinematics [B, T, 60, 9]
- Upper-Body Visual ROI [B, T, 3, 256, 256]
- Dual Hand Visual Crops [B, T, 3, 128, 128]
- 19D Phonology Descriptors [B, T, 19]
- Cranial IMU Rotational Velocities [B, T, 3]
- Non-Manual Facial Features [B, T, 12, 3]
- English & Gloss Tokens with Grammatical Negation Flags
================================================================================
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn.functional as F

# Resiliently import vocabulary and base loader classes from V2
try:
    from train_tpu.v2.engine.dataset import (
        GlossVocabulary,
        EnglishVocabulary,
        fast_vectorized_collate_fn as base_collate_fn,
        trim_host_memory,
    )
except ImportError:
    from dataset import (
        GlossVocabulary,
        EnglishVocabulary,
        fast_vectorized_collate_fn as base_collate_fn,
        trim_host_memory,
    )


NEGATION_WORDS = frozenset({"not", "n't", "no", "never", "none", "cannot", "cant", "wont", "dont", "didnt", "isnt"})


def fast_vectorized_v3_collate_fn(batch: List[Dict[str, Any]], tile_multiple: int = 1) -> Dict[str, Any]:
    """
    V3 Enhanced Vectorized Collation Function.
    Robustly pads and stacks all V3 multimodal streams cleanly into TPU-aligned batches.
    When tile_multiple == 128, enforces TPU v5e MXU 128x128 systolic tile invariants.
    """
    # 1. Ensure all features in batch have matching temporal length max_len
    lengths = []
    for b in batch:
        f = b.get("feature", b.get("features", b.get("kinematics", None)))
        if f is not None:
            lengths.append(f.shape[0])
    raw_max_len = max(lengths) if lengths else 1

    # Tile-align temporal length to multiple (e.g. 128 for TPU, 1 for local testing)
    if tile_multiple > 1:
        max_len = int(math.ceil(raw_max_len / float(tile_multiple)) * tile_multiple)
    else:
        max_len = raw_max_len

    padded_batch = []
    for b in batch:
        b_copy = dict(b)
        f = b.get("feature", b.get("features", b.get("kinematics", None)))
        if f is not None:
            if not isinstance(f, torch.Tensor):
                f = torch.tensor(f, dtype=torch.float32)
            if f.shape[0] < max_len:
                pad_amount = max_len - f.shape[0]
                # Flatten trailing dims to pad dim 0 easily
                orig_shape = f.shape
                f_flat = f.view(orig_shape[0], -1)
                f_padded = F.pad(f_flat, (0, 0, 0, pad_amount)).view(max_len, *orig_shape[1:])
                b_copy["feature"] = f_padded
            else:
                b_copy["feature"] = f[:max_len]

        if "mask" not in b_copy:
            b_copy["mask"] = torch.ones(max_len, dtype=torch.bool)
        elif isinstance(b_copy["mask"], torch.Tensor) and b_copy["mask"].shape[0] < max_len:
            b_copy["mask"] = F.pad(b_copy["mask"], (0, max_len - b_copy["mask"].shape[0]), value=False)
        elif isinstance(b_copy["mask"], torch.Tensor) and b_copy["mask"].shape[0] > max_len:
            b_copy["mask"] = b_copy["mask"][:max_len]

        if "label" not in b_copy:
            b_copy["label"] = b_copy.get("label_idx", 0)
        if "sample_weight" not in b_copy:
            b_copy["sample_weight"] = 1.0

        if "gloss_seq" not in b_copy:
            b_copy["gloss_seq"] = np.zeros(16, dtype=np.int64)
        if "gloss_len" not in b_copy:
            b_copy["gloss_len"] = 0
        if "has_valid_gloss" not in b_copy:
            b_copy["has_valid_gloss"] = False

        if "chicago_seq" not in b_copy:
            b_copy["chicago_seq"] = np.zeros(32, dtype=np.int64)
        if "chicago_len" not in b_copy:
            b_copy["chicago_len"] = 0
        if "has_valid_chicago" not in b_copy:
            b_copy["has_valid_chicago"] = False

        if "english_seq" not in b_copy:
            b_copy["english_seq"] = np.zeros(32, dtype=np.int64)
        if "english_len" not in b_copy:
            b_copy["english_len"] = 0
        if "has_valid_english" not in b_copy:
            b_copy["has_valid_english"] = False

        # Isolate visual streams from base_collate_fn to permit V3 1D compact tokens and heterogeneous None values
        b_copy.pop("roi_visual", None)
        b_copy.pop("hand_visual", None)

        padded_batch.append(b_copy)

    # Base collation on padded batch
    collated = base_collate_fn(padded_batch)

    # 2. Extract or Synthesize V3 Multi-Tier Streams
    B = len(batch)
    kin = collated.get("kinematics", collated.get("feature", collated.get("features")))
    collated["kinematics"] = kin
    actual_len = kin.shape[1]

    # A. 19D Phonology Stream [B, actual_len, 19]
    has_phon = any(b.get("phonology") is not None for b in batch)
    if has_phon:
        phon_list = []
        for item in batch:
            ph = item.get("phonology")
            if ph is None:
                phon_list.append(torch.zeros((actual_len, 19), dtype=torch.float32))
                continue
            if not isinstance(ph, torch.Tensor):
                ph = torch.tensor(ph, dtype=torch.float32)
            cur_len = ph.shape[0]
            if cur_len < actual_len:
                ph = F.pad(ph, (0, 0, 0, actual_len - cur_len))
            elif cur_len > actual_len:
                ph = ph[:actual_len]
            phon_list.append(ph)
        collated["phonology"] = torch.stack(phon_list, dim=0)
    else:
        collated["phonology"] = None

    # B. Cranial IMU [B, actual_len, 3]
    has_imu = any(b.get("cranial_imu") is not None for b in batch)
    if has_imu:
        cranial_list = []
        for item in batch:
            imu = item.get("cranial_imu")
            if imu is None:
                cranial_list.append(torch.zeros((actual_len, 3), dtype=torch.float32))
                continue
            if not isinstance(imu, torch.Tensor):
                imu = torch.tensor(imu, dtype=torch.float32)
            cur_len = imu.shape[0]
            if cur_len < actual_len:
                imu = F.pad(imu, (0, 0, 0, actual_len - cur_len))
            elif cur_len > actual_len:
                imu = imu[:actual_len]
            cranial_list.append(imu)
        collated["cranial_imu"] = torch.stack(cranial_list, dim=0)
    else:
        # Fallback: estimate from nasal tip landmark velocity (canonical landmark index 48)
        if kin.dim() == 4 and kin.shape[2] >= 60:
            nose_pos = kin[:, :, 48, :3].detach()  # Canonical nose index 48
            v_nose = torch.diff(nose_pos, dim=1, prepend=nose_pos[:, :1, :])
            collated["cranial_imu"] = v_nose * 5.0
        else:
            collated["cranial_imu"] = torch.zeros((B, actual_len, 3), dtype=torch.float32)

    # C. Non-Manual Facial Landmarks [B, actual_len, 12, 3]
    has_face = any(b.get("face_landmarks") is not None for b in batch)
    if has_face:
        face_list = []
        for item in batch:
            f = item.get("face_landmarks")
            if f is None:
                face_list.append(torch.zeros((actual_len, 12, 3), dtype=torch.float32))
                continue
            if not isinstance(f, torch.Tensor):
                f = torch.tensor(f, dtype=torch.float32)
            cur_len = f.shape[0]
            if cur_len < actual_len:
                f = F.pad(f, (0, 0, 0, 0, 0, actual_len - cur_len))
            elif cur_len > actual_len:
                f = f[:actual_len]
            face_list.append(f)
        collated["face_landmarks"] = torch.stack(face_list, dim=0)
    else:
        if kin.dim() == 4 and kin.shape[2] >= 60:
            collated["face_landmarks"] = kin[:, :, 48:60, :3].detach()
        else:
            collated["face_landmarks"] = None

    # D. Upper-Body ROI Visual Crops or Compact Tokens
    has_roi = any(b.get("roi_visual") is not None for b in batch)
    if has_roi:
        ref_roi = next(b["roi_visual"] for b in batch if b.get("roi_visual") is not None)
        if not isinstance(ref_roi, torch.Tensor):
            ref_roi = torch.tensor(ref_roi)
        is_4d_raw = (ref_roi.dim() == 4 and ref_roi.shape[-1] == 3) or (ref_roi.dim() == 4 and ref_roi.shape[1] == 3)
        trailing_shape = (3, 256, 256) if is_4d_raw else (ref_roi.shape[-1],)

        roi_list = []
        for item in batch:
            rv = item.get("roi_visual")
            if rv is None:
                roi_list.append(torch.zeros((actual_len, *trailing_shape), dtype=torch.float32))
                continue
            if not isinstance(rv, torch.Tensor):
                rv = torch.tensor(rv, dtype=torch.float32)
            if rv.dim() == 4 and rv.shape[-1] == 3:
                rv = rv.permute(0, 3, 1, 2)  # [T, H, W, 3] -> [T, 3, H, W]
            cur_len = rv.shape[0]
            if cur_len < actual_len:
                pad_shape = [0] * (2 * (rv.dim() - 1)) + [0, actual_len - cur_len]
                rv = F.pad(rv, pad_shape)
            elif cur_len > actual_len:
                rv = rv[:actual_len]
            roi_list.append(rv)
        collated["roi_visual"] = torch.stack(roi_list, dim=0)
    else:
        collated["roi_visual"] = None

    # E. Hand Visual Crops or Compact Tokens
    has_hand = any(b.get("hand_visual") is not None for b in batch)
    if has_hand:
        ref_hv = next(b["hand_visual"] for b in batch if b.get("hand_visual") is not None)
        if not isinstance(ref_hv, torch.Tensor):
            ref_hv = torch.tensor(ref_hv)
        is_4d_raw = (ref_hv.dim() == 4 and ref_hv.shape[-1] == 3) or (ref_hv.dim() == 4 and ref_hv.shape[1] == 3)
        trailing_shape = (3, 128, 128) if is_4d_raw else (ref_hv.shape[-1],)

        hand_list = []
        for item in batch:
            hv = item.get("hand_visual")
            if hv is None:
                hand_list.append(torch.zeros((actual_len, *trailing_shape), dtype=torch.float32))
                continue
            if not isinstance(hv, torch.Tensor):
                hv = torch.tensor(hv, dtype=torch.float32)
            if hv.dim() == 4 and hv.shape[-1] == 3:
                hv = hv.permute(0, 3, 1, 2)
            cur_len = hv.shape[0]
            if cur_len < actual_len:
                pad_shape = [0] * (2 * (hv.dim() - 1)) + [0, actual_len - cur_len]
                hv = F.pad(hv, pad_shape)
            elif cur_len > actual_len:
                hv = hv[:actual_len]
            hand_list.append(hv)
        collated["hand_visual"] = torch.stack(hand_list, dim=0)
    else:
        collated["hand_visual"] = None

    # F. Target Text Negation Polarity Flag [B]
    text_is_neg = torch.zeros(B, dtype=torch.bool)
    for idx, item in enumerate(batch):
        raw_text = item.get("text", "") or item.get("english_text", "")
        if isinstance(raw_text, str) and raw_text:
            tokens = raw_text.lower().split()
            if any(t in NEGATION_WORDS for t in tokens):
                text_is_neg[idx] = True
    collated["text_is_negative"] = text_is_neg

    return collated


class ASLV3Dataset:
    """
    V3 Dataset Factory providing unified dataset access with multi-tier collation.
    """

    @staticmethod
    def create_dataloader(
        dataset,
        batch_size: int = 16,
        shuffle: bool = True,
        num_workers: int = 2,
        drop_last: bool = True,
    ):
        from torch.utils.data import DataLoader
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=fast_vectorized_v3_collate_fn,
            drop_last=drop_last,
            pin_memory=False,
        )
