#!/usr/bin/env python3
"""
================================================================================
MONOLITHIC ALL-IN-ONE TPU/GPU ASL FOUNDATION MODEL — SENTENCE RECONSTRUCTION
Encoder: MobileConformer (8L × d=320, nhead=8, ffn=1280) — ~17.4M parameters
Decoder: ASLTransformerDecoder (8L × d=320, GQA 8Q/2KV, RoPE, ffn=1280) — ~12.9M parameters
Total:   ~31.0M parameters (High Efficiency & SOTA Accuracy via Extended Compute)

Task: Continuous Sign Language Understanding & Gloss Sentence Reconstruction
================================================================================
"""

import os
import sys
import time
import glob
import json
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Set up for Kaggle 2x T4 GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp

    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False


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


# ==============================================================================
# 1. LANDMARK AUGMENTER (REAL-WORLD CAMERA NOISE & PHYSIOLOGICAL STALLING)
# ==============================================================================


class LandmarkAugmenter:
    """
    Progressive Noise Curriculum Data Augmentation for 3D WholeBody landmark sequences (T, 60, 9).
    Calculates velocity and acceleration BEFORE applying coordinate dropouts to avoid velocity artifacts.
    """

    def __init__(
        self,
        base_jitter_std: float = 0.025,
        max_scale_range: Tuple[float, float] = (0.925, 1.075),
        max_shift_range: float = 0.025,
        max_rotation_range: float = 7.5,
        max_kp_drop_prob: float = 0.04,
        max_frame_drop_prob: float = 0.025,
        noise_level: float = 0.02,
    ):
        self.base_jitter_std = base_jitter_std
        self.max_scale_range = max_scale_range
        self.max_shift_range = max_shift_range
        self.max_rotation_range = max_rotation_range
        self.max_kp_drop_prob = max_kp_drop_prob
        self.max_frame_drop_prob = max_frame_drop_prob
        self.noise_level = max(0.001, min(1.0, noise_level))

    def set_noise_level(self, level: float) -> None:
        self.noise_level = max(0.001, min(1.0, level))

    def __call__(self, feat_arr: np.ndarray) -> np.ndarray:
        T, K, C = feat_arr.shape
        if T == 0:
            return feat_arr

        aug = feat_arr.copy()
        pos = aug[:, :, :3]

        jitter_std = self.base_jitter_std * self.noise_level
        rot_range = self.max_rotation_range * self.noise_level
        shift_range = self.max_shift_range * self.noise_level
        kp_drop_prob = self.max_kp_drop_prob * self.noise_level
        frame_drop_prob = self.max_frame_drop_prob * self.noise_level
        scale_delta = (self.max_scale_range[1] - 1.0) * self.noise_level

        choke_prob = 0.005 + 0.01 * self.noise_level
        finger_drop_prob = 0.01 + 0.025 * self.noise_level
        timestretch_prob = 0.01 + 0.04 * self.noise_level
        warping_prob = 0.01 + 0.05 * self.noise_level
        hand_occ_prob = 0.005 + 0.01 * self.noise_level

        if np.random.rand() > 0.5:
            pos[:, :, 0] = -pos[:, :, 0]
            pos_lh = pos[:, 18:39, :].copy()
            pos_rh = pos[:, 39:60, :].copy()
            pos[:, 18:39, :] = pos_rh
            pos[:, 39:60, :] = pos_lh

        scale = np.random.uniform(1.0 - scale_delta, 1.0 + scale_delta)
        pos = pos * scale

        if rot_range > 0 and np.random.rand() < 0.5:
            # MATH: Apply 3D rotation matrices for pitch, yaw, and roll
            pitch_deg = np.random.uniform(-rot_range, rot_range)
            yaw_deg = np.random.uniform(-rot_range, rot_range)
            roll_deg = np.random.uniform(-rot_range, rot_range)
            rad_p, rad_y, rad_r = (
                np.radians(pitch_deg),
                np.radians(yaw_deg),
                np.radians(roll_deg),
            )
            rx = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(rad_p), -np.sin(rad_p)],
                    [0, np.sin(rad_p), np.cos(rad_p)],
                ],
                dtype=np.float32,
            )
            ry = np.array(
                [
                    [np.cos(rad_y), 0, np.sin(rad_y)],
                    [0, 1, 0],
                    [-np.sin(rad_y), 0, np.cos(rad_y)],
                ],
                dtype=np.float32,
            )
            rz = np.array(
                [
                    [np.cos(rad_r), -np.sin(rad_r), 0],
                    [np.sin(rad_r), np.cos(rad_r), 0],
                    [0, 0, 1],
                ],
                dtype=np.float32,
            )
            rot_mat = np.dot(rz, np.dot(ry, rx))
            center = pos.mean(axis=(0, 1), keepdims=True)
            pos = pos - center
            pos = np.dot(pos.reshape(-1, 3), rot_mat.T).reshape(T, K, 3)
            pos = pos + center

        if shift_range > 0 and np.random.rand() < 0.5:
            # MATH: Random 3D spatial translation shift
            shift_val = np.random.uniform(-shift_range, shift_range, size=(1, 1, 3))
            pos = pos + shift_val

        if T > 10 and np.random.rand() < timestretch_prob:
            time_scale = np.random.uniform(0.75, 1.25)
            new_T = int(round(T * time_scale))
            if new_T > 4:
                idx = np.linspace(0, T - 1, num=new_T, dtype=int)
                pos = pos[idx]
                T = new_T

        if jitter_std > 0:
            pos += np.random.normal(0, jitter_std, size=pos.shape).astype(np.float32)

        if T > 20 and np.random.rand() < choke_prob:
            fz_len = min(int(T * 0.25), np.random.randint(10, 25))
            if fz_len > 1:
                fz_idx = np.random.randint(2, max(3, T - fz_len))
                base_f = pos[fz_idx].copy()
                t_steps = np.linspace(0, 2 * np.pi, num=fz_len, dtype=np.float32)
                dx = 0.015 * np.sin(t_steps) + np.random.normal(
                    0, 0.002, fz_len
                ).astype(np.float32)
                dy = 0.010 * np.cos(t_steps * 1.5) + np.random.normal(
                    0, 0.002, fz_len
                ).astype(np.float32)
                dz = 0.010 * np.sin(t_steps * 0.7) + np.random.normal(
                    0, 0.002, fz_len
                ).astype(np.float32)
                for si in range(fz_len):
                    ff = base_f.copy()
                    ff[:, 0] += dx[si]
                    ff[:, 1] += dy[si]
                    ff[:, 2] += dz[si]
                    scale_factor = 1.0 + 0.02 * np.sin(t_steps[si] * 3.0)
                    center_hands = ff[18:60, :].mean(axis=0, keepdims=True)
                    ff[18:60, :] = (
                        ff[18:60, :] - center_hands
                    ) * scale_factor + center_hands
                    pos[fz_idx + si] = ff

        if T > 25 and np.random.rand() < warping_prob:
            t_warped = np.power(np.linspace(0, 1, T), np.random.uniform(0.5, 1.5))
            w_idx = np.clip((t_warped * (T - 1)).astype(int), 0, T - 1)
            pos = pos[w_idx]

        disp_prob = 0.01 + 0.04 * self.noise_level
        if T > 15 and np.random.rand() < disp_prob:
            d_idx = np.random.randint(3, max(4, T - 10))
            d_len = np.random.randint(3, 8)
            h_idx = range(18, 39) if np.random.rand() > 0.5 else range(39, 60)
            off = np.random.normal(0, 1, 3).astype(np.float32)
            off = off / max(1e-6, np.linalg.norm(off)) * np.random.uniform(0.01, 0.15)
            arc = np.linspace(0, np.pi, num=d_len, dtype=np.float32)
            for si in range(d_len):
                if d_idx + si < T:
                    pos[d_idx + si, h_idx, :3] += np.sin(arc[si]) * off

        # ── Calculate velocities and accelerations AFTER ALL SPATIAL AND TEMPORAL DISTORTIONS ────
        vel = np.zeros_like(pos)
        if T > 2:
            vel[1:-1] = (pos[2:] - pos[:-2]) / 2.0
            vel[0] = pos[1] - pos[0]
            vel[-1] = pos[-1] - pos[-2]
        elif T == 2:
            vel[:] = pos[1] - pos[0]

        acc = np.zeros_like(pos)
        if T > 2:
            acc[1:-1] = pos[2:] - 2 * pos[1:-1] + pos[:-2]
            acc[0] = 2 * pos[0] - 5 * pos[1] + 4 * pos[2] - pos[3] if T > 3 else 0
            acc[-1] = 2 * pos[-1] - 5 * pos[-2] + 4 * pos[-3] - pos[-4] if T > 3 else 0

        # ── Apply dropout / occlusion masks to positions AND kinematic vectors ────
        drop_mask = np.ones((T, K, 1), dtype=np.float32)
        if kp_drop_prob > 0:
            drop_mask *= (np.random.rand(T, K, 1) > kp_drop_prob).astype(np.float32)

        if np.random.rand() < finger_drop_prob:
            lh_f = [
                list(range(18, 23)),
                list(range(23, 27)),
                list(range(27, 31)),
                list(range(31, 35)),
                list(range(35, 39)),
            ]
            rh_f = [
                list(range(39, 44)),
                list(range(44, 48)),
                list(range(48, 52)),
                list(range(52, 56)),
                list(range(56, 60)),
            ]
            for f_idx in random.sample(lh_f + rh_f, k=random.randint(1, 2)):
                drop_mask[:, f_idx, :] = 0.0

        if np.random.rand() < hand_occ_prob and T > 10:
            occ_s = np.random.randint(0, max(1, T - 8))
            occ_l = np.random.randint(4, max(5, T // 2))
            h_idx = range(18, 39) if np.random.rand() > 0.5 else range(39, 60)
            drop_mask[occ_s : occ_s + occ_l, h_idx, :] = 0.0

        pos = pos * drop_mask
        vel = vel * drop_mask
        acc = acc * drop_mask

        if frame_drop_prob > 0 and T > 8:
            keep_mask = np.random.rand(T) > frame_drop_prob
            if np.sum(keep_mask) >= 4:
                pos, vel, acc = pos[keep_mask], vel[keep_mask], acc[keep_mask]

        return np.concatenate([pos, vel, acc], axis=-1)


# ==============================================================================
# 2. GLOSS VOCABULARY — Sequence Vocabulary with Special Tokens
# ==============================================================================


class GlossVocabulary:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    OFFSET = 4

    def __init__(self, label_to_idx: Dict):
        clean_l2i = {}
        if isinstance(label_to_idx, dict):
            for k, v in label_to_idx.items():
                if isinstance(v, int):
                    clean_l2i[str(k)] = v
                elif isinstance(v, dict):
                    idx_val = v.get("id", v.get("idx", v.get("label_idx", 0)))
                    clean_l2i[str(k)] = int(idx_val)
                elif isinstance(v, str) and str(k).isdigit():
                    clean_l2i[str(v)] = int(k)
                else:
                    try:
                        clean_l2i[str(k)] = int(v)
                    except (ValueError, TypeError):
                        pass

        self.label_to_idx = clean_l2i
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        max_idx = max(clean_l2i.values()) if clean_l2i else 0
        self.vocab_size = max(len(self.label_to_idx), max_idx + 1) + self.OFFSET

        # Load translation mapping for decoded output strings (zoomin -> zoom-in)
        self.output_map = {}
        import json, os

        candidates = [
            "E:/datasets/results/asl_preprocessed_phase1/output_mapping.json",
            "./output_mapping.json",
            "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1/output_mapping.json",
            "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1/output_mapping.json",
            "/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1/output_mapping.json",
            "/kaggle/input/asl-preprocessed-phase1/output_mapping.json",
        ]

        for out_map_path in candidates:
            if os.path.exists(out_map_path):
                try:
                    with open(out_map_path, "r", encoding="utf-8") as f:
                        self.output_map = json.load(f)
                    break
                except Exception:
                    pass

    def __len__(self) -> int:
        return self.vocab_size

    def gloss_to_token(self, gloss: str) -> int:
        raw = self.label_to_idx.get(
            gloss.upper(), self.label_to_idx.get(gloss.lower(), None)
        )
        if raw is None:
            return self.UNK_ID
        return raw + self.OFFSET

    def token_to_gloss(self, tid: int) -> str:
        if tid == self.PAD_ID:
            return "<PAD>"
        if tid == self.BOS_ID:
            return "<BOS>"
        if tid == self.EOS_ID:
            return "<EOS>"
        if tid == self.UNK_ID:
            return "<UNK>"
        gloss = self.idx_to_label.get(tid - self.OFFSET, "<UNK>")
        return self.output_map.get(gloss, gloss)

    def encode(self, gloss_list: List[str]) -> List[int]:
        return (
            [self.BOS_ID] + [self.gloss_to_token(g) for g in gloss_list] + [self.EOS_ID]
        )

    def decode(self, token_ids: List[int]) -> List[str]:
        out = []
        for tid in token_ids:
            if tid == self.EOS_ID:
                break
            if tid in (self.PAD_ID, self.BOS_ID):
                continue
            out.append(self.token_to_gloss(tid))
        return out


# ==============================================================================
# 3. RMSNorm & SwiGLUFFN
# ==============================================================================


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MATH: RMSNorm normalizes by root mean square of features: x / sqrt(mean(x^2) + eps) * weight
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, num_layers: int = 8):
        super().__init__()
        hidden = (int(dim_feedforward * 2 / 3) + 7) // 8 * 8
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_down.weight, std=0.02 / math.sqrt(4.0 * num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # MATH: SwiGLU activation computes element-wise product of Swish-gated linear and up-projection: W_down * (swish(W_gate * x) * (W_up * x))
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


# ==============================================================================
# 4. RICH ASL-LEX MULTI-ATTRIBUTE EMBEDDING TABLE
# ==============================================================================


class RichASLLexEmbeddingTable(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 320,
        csv_path: Optional[Union[str, Path]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.lex_class_map = defaultdict(lambda: 0)
        self.signtype_map = defaultdict(lambda: 0)
        self.handshape_map = defaultdict(lambda: 0)
        self.location_map = defaultdict(lambda: 0)
        self.category_map = defaultdict(lambda: 0)

        self.emb_lexclass = nn.Embedding(20, 32)
        self.emb_signtype = nn.Embedding(16, 32)
        self.emb_handshape = nn.Embedding(48, 48)
        self.emb_location = nn.Embedding(24, 32)
        self.emb_category = nn.Embedding(36, 48)

        raw_dim = 32 + 32 + 48 + 32 + 48 + 3
        self.attr_proj = nn.Sequential(
            nn.Linear(raw_dim, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        attr_idx_matrix = torch.zeros(vocab_size, 5, dtype=torch.long)
        attr_scalars = torch.zeros(vocab_size, 3, dtype=torch.float32)
        if label_to_idx and csv_path and Path(csv_path).exists():
            self._populate_attr_matrix(
                attr_idx_matrix, attr_scalars, csv_path, label_to_idx
            )

        self.register_buffer("attr_idx_matrix", attr_idx_matrix)
        self.register_buffer("attr_scalars", attr_scalars)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.emb_lexclass.weight, std=0.02)
        nn.init.normal_(self.emb_signtype.weight, std=0.02)
        nn.init.normal_(self.emb_handshape.weight, std=0.02)
        nn.init.normal_(self.emb_location.weight, std=0.02)
        nn.init.normal_(self.emb_category.weight, std=0.02)

    def _populate_attr_matrix(
        self,
        attr_idx_matrix: torch.Tensor,
        attr_scalars: torch.Tensor,
        csv_path: Union[str, Path],
        label_to_idx: Dict[str, int],
    ):
        import csv as _csv

        try:
            with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                reader = _csv.DictReader(f)
                for row in reader:
                    lemma = (
                        (row.get("LemmaID") or row.get("EntryID") or "").strip().upper()
                    )
                    if not lemma:
                        continue
                    raw_id = label_to_idx.get(lemma)
                    if raw_id is None:
                        continue
                    tid = raw_id + GlossVocabulary.OFFSET
                    if tid >= self.vocab_size:
                        continue

                    lc = row.get("LexicalClass", "Other")
                    st = row.get("SignType.2.0", "Other")
                    hs = row.get("Handshape.2.0", "Other")
                    loc = row.get("MajorLocation.2.0", "Other")
                    cat = row.get("CDISemanticCategory", "Other")

                    def _get_id(val_str, mapping, max_n):
                        if val_str not in mapping:
                            mapping[val_str] = min(len(mapping) + 1, max_n - 1)
                        return mapping[val_str]

                    lc_id = _get_id(lc, self.lex_class_map, 20)
                    st_id = _get_id(st, self.signtype_map, 16)
                    hs_id = _get_id(hs, self.handshape_map, 48)
                    loc_id = _get_id(loc, self.location_map, 24)
                    cat_id = _get_id(cat, self.category_map, 36)

                    try:
                        freq_z = float(row.get("SignFrequency(Z)", 0.0) or 0.0)
                    except:
                        freq_z = 0.0
                    try:
                        icon_z = float(row.get("Iconicity(Z)", 0.0) or 0.0)
                    except:
                        icon_z = 0.0
                    try:
                        comp = float(row.get("Phonological Complexity", 0.0) or 0.0)
                    except:
                        comp = 0.0

                    attr_idx_matrix[tid] = torch.tensor(
                        [lc_id, st_id, hs_id, loc_id, cat_id], dtype=torch.long
                    )
                    attr_scalars[tid] = torch.tensor(
                        [freq_z, icon_z, comp], dtype=torch.float32
                    )
        except Exception:
            pass

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids.clamp(0, self.vocab_size - 1)
        attr_ids = self.attr_idx_matrix[ids]
        scalars = F.embedding(ids, self.attr_scalars)

        e_lc = self.emb_lexclass(attr_ids[:, :, 0])
        e_st = self.emb_signtype(attr_ids[:, :, 1])
        e_hs = self.emb_handshape(attr_ids[:, :, 2])
        e_loc = self.emb_location(attr_ids[:, :, 3])
        e_cat = self.emb_category(attr_ids[:, :, 4])

        raw_attrs = torch.cat([e_lc, e_st, e_hs, e_loc, e_cat, scalars], dim=-1)
        valid_lex_mask = (
            (token_ids >= GlossVocabulary.OFFSET).unsqueeze(-1).to(raw_attrs.dtype)
        )
        return self.attr_proj(raw_attrs) * valid_lex_mask


# ==============================================================================
# 5. TOKEN MERGING BLOCK (ToMe WITH CHRONOLOGICAL ARROW OF TIME RESTORATION)
# ==============================================================================


class TokenMergingBlock(nn.Module):
    """
    Bipartite Token Merging (ToMe) for temporal ASL sequence compression.
    Includes chronological interleaving to preserve temporal causality for Mamba/Conformer.
    Parameter-free heuristic similarity to eliminate non-differentiable dead weights.
    """

    def __init__(self, r: int = 80, d_model: int = 320):
        super().__init__()
        self.r = r
        self.d_model = d_model

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T, D = h.shape

        min_half = T // 2

        # Ensure we never compress below max_dec_len (e.g., 60 tokens)
        min_required_tokens = 60
        r_clamp = min(self.r, max(0, min_half - (min_required_tokens // 2)))

        if r_clamp <= 0:
            self._last_T = T
            self._last_sorted_routing = (
                torch.arange(T, device=h.device).unsqueeze(0).expand(B, -1)
            )
            self._last_unmerged_indices_a = torch.empty(
                B, 0, dtype=torch.long, device=h.device
            )
            self._last_unmerged_indices_b = torch.empty(
                B, 0, dtype=torch.long, device=h.device
            )
            self._last_merged_indices = torch.empty(
                B, 0, dtype=torch.long, device=h.device
            )
            self._last_matched_b_indices = torch.empty(
                B, 0, dtype=torch.long, device=h.device
            )
            return (
                h,
                mask,
                {"T_orig": T, "sorted_routing": torch.arange(T, device=h.device)},
            )

        if T > 1:
            kernel = (
                torch.tensor([0.25, 0.5, 0.25], device=h.device, dtype=h.dtype)
                .view(1, 1, 3)
                .expand(D, 1, 3)
            )
            h_pad = F.pad(h.transpose(1, 2), (1, 1), mode="replicate")
            h_smooth = F.conv1d(h_pad, kernel, groups=D).transpose(1, 2)
        else:
            h_smooth = h

        a = h[:, 0::2]
        b = h[:, 1::2]
        a_smooth = h_smooth[:, 0::2]
        b_smooth = h_smooth[:, 1::2]
        min_half = min(a.size(1), b.size(1))
        a, b = a[:, :min_half], b[:, :min_half]
        a_smooth, b_smooth = a_smooth[:, :min_half], b_smooth[:, :min_half]

        # MATH: Compute pairwise cosine similarity between consecutive temporal tokens
        # MATH: Bipartite matching maps A set (even indices) to B set (odd indices) for downsampling
        ka = F.normalize(a_smooth, p=2, dim=-1)
        kb = F.normalize(b_smooth, p=2, dim=-1)
        sim_matrix = torch.matmul(ka, kb.transpose(-1, -2))
        if mask is not None:
            ma = mask[:, 0::2][:, :min_half]  # True = Valid
            mb = mask[:, 1::2][:, :min_half]

            # Target cross-boundaries (Valid to Pad)
            cross_boundary = ma.unsqueeze(-1) ^ mb.unsqueeze(-2)
            # Target pad-to-pad (Pad to Pad)
            pad_to_pad = (~ma).unsqueeze(-1) & (~mb).unsqueeze(-2)

            sim_matrix = sim_matrix.masked_fill(pad_to_pad, -1e4)
            sim_matrix = sim_matrix.masked_fill(cross_boundary, -1e4)

        mlm_mask = kwargs.get("mlm_mask", None)
        if mlm_mask is not None:
            mlm_a = mlm_mask[:, 0::2][:, :min_half]
            mlm_b = mlm_mask[:, 1::2][:, :min_half]
            mlm_pair_mask = mlm_a.unsqueeze(-1) ^ mlm_b.unsqueeze(-2)
            sim_matrix = sim_matrix.masked_fill(mlm_pair_mask, -1e4)

        scores, dst_idx = sim_matrix.max(dim=-1)
        r_clamp = min(r, min_half)
        _, merge_idx = scores.topk(r_clamp, dim=-1, largest=True, sorted=False)

        matched_b_indices_local = dst_idx.gather(1, merge_idx)
        unmerged_scores = torch.zeros(B, min_half, device=h.device)
        unmerged_scores.scatter_(1, merge_idx, -1e4)
        _, kept_idx_a = unmerged_scores.topk(min_half - r_clamp, dim=-1, sorted=True)
        kept_idx_a, _ = kept_idx_a.sort(dim=-1)
        kept_a = a.gather(1, kept_idx_a.unsqueeze(-1).expand(-1, -1, D))

        a_merged = a.gather(1, merge_idx.unsqueeze(-1).expand(-1, -1, D))
        b_updated = b.clone()
        dst_idx_exp = matched_b_indices_local.unsqueeze(-1).expand(-1, -1, D)
        b_updated.scatter_add_(1, dst_idx_exp, a_merged)

        counts = torch.ones(B, min_half, 1, device=h.device, dtype=h.dtype)
        counts_add = torch.ones(B, r_clamp, 1, device=h.device, dtype=h.dtype)
        counts.scatter_add_(1, matched_b_indices_local.unsqueeze(-1), counts_add)
        b_updated = b_updated / counts

        unmerged_indices_a = kept_idx_a * 2
        unmerged_indices_b = (
            torch.arange(min_half, device=h.device).unsqueeze(0).expand(B, -1) * 2 + 1
        )
        if h.shape[1] % 2 != 0:
            tail_idx = torch.full((B, 1), h.shape[1] - 1, device=h.device)
            unmerged_indices_b = torch.cat([unmerged_indices_b, tail_idx], dim=1)

        all_out_indices = torch.cat([unmerged_indices_a, unmerged_indices_b], dim=1)
        _, sorted_routing = torch.sort(all_out_indices, dim=1)

        h_unordered = torch.cat([kept_a, b_updated], dim=1)
        if h.shape[1] % 2 != 0:
            h_unordered = torch.cat([h_unordered, h[:, -1:]], dim=1)
        h_out = h_unordered.gather(1, sorted_routing.unsqueeze(-1).expand(-1, -1, D))

        mask_out = None
        if mask is not None:
            ma = mask[:, 0::2][:, :min_half]
            mb = mask[:, 1::2][:, :min_half]
            kept_mask_a = ma.gather(1, kept_idx_a)
            merged_mask = ma.gather(1, merge_idx) | mb.gather(
                1, matched_b_indices_local
            )
            b_mask_updated = mb.clone()
            b_mask_updated.scatter_(1, matched_b_indices_local, merged_mask)
            mask_unordered = torch.cat([kept_mask_a, b_mask_updated], dim=1)
            if h.shape[1] % 2 != 0:
                mask_unordered = torch.cat([mask_unordered, mask[:, -1:]], dim=1)
            mask_out = mask_unordered.gather(1, sorted_routing)

        mlm_out = None
        if mlm_mask is not None:
            mlm_kept = mlm_a.gather(1, kept_idx_a)
            mlm_merged = mlm_a.gather(1, merge_idx) | mlm_b.gather(
                1, matched_b_indices_local
            )
            mlm_b_upd = mlm_b.clone()
            mlm_b_upd.scatter_(1, matched_b_indices_local, mlm_merged)
            mlm_unord = torch.cat([mlm_kept, mlm_b_upd], dim=1)
            if mlm_mask.shape[1] % 2 != 0:
                mlm_unord = torch.cat([mlm_unord, mlm_mask[:, -1:]], dim=1)
            mlm_out = mlm_unord.gather(1, sorted_routing)

        routing_info = {
            "kept_idx_a": kept_idx_a,
            "merge_idx": merge_idx,
            "matched_b_indices": matched_b_indices_local,
            "T_orig": T,
            "sorted_routing": sorted_routing,
            "mlm_out": mlm_out,
        }
        return h_out, mask_out, routing_info

    def unmerge(self, x_comp: torch.Tensor, routing_info: dict) -> torch.Tensor:
        B, T_comp, D = x_comp.shape
        T_orig = routing_info["T_orig"]
        x_full = torch.zeros(B, T_orig, D, device=x_comp.device, dtype=x_comp.dtype)
        x_unord = torch.zeros_like(x_comp)
        x_unord.scatter_(
            1, routing_info["sorted_routing"].unsqueeze(-1).expand(-1, -1, D), x_comp
        )
        ka_len = routing_info["kept_idx_a"].size(1)
        kb_len = T_comp - ka_len - (1 if T_orig % 2 != 0 else 0)
        x_ka = x_unord[:, :ka_len]
        x_b_updated = x_unord[:, ka_len : ka_len + kb_len]
        x_m = x_b_updated.gather(
            1, routing_info["matched_b_indices"].unsqueeze(-1).expand(-1, -1, D)
        )
        x_full.scatter_(
            1, (routing_info["kept_idx_a"] * 2).unsqueeze(-1).expand(-1, -1, D), x_ka
        )
        x_full.scatter_(
            1,
            (
                torch.arange(kb_len, device=x_comp.device).unsqueeze(0).expand(B, -1)
                * 2
                + 1
            )
            .unsqueeze(-1)
            .expand(-1, -1, D),
            x_b_updated,
        )
        x_full.scatter_(
            1, (routing_info["merge_idx"] * 2).unsqueeze(-1).expand(-1, -1, D), x_m
        )
        if T_orig % 2 != 0:
            x_full[:, -1:] = x_unord[:, -1:]
        return x_full


# ==============================================================================
# 6. ENCODER & DECODER ARCHITECTURE
# ==============================================================================


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class GroupedQueryEncoderAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 320,
        nhead: int = 8,
        kv_heads: int = 2,
        max_len: int = 320,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert nhead % kv_heads == 0
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.dropout_p = dropout_p
        self.groups = nhead // kv_heads
        self.head_dim = d_model // nhead
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.max_len = max_len
        self.max_relative_position = max_len - 1
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
        self.k_norm = RMSNorm(d_model)
        self.relative_position_bias = nn.Embedding(2 * max_len - 1, nhead)

    def _get_relative_position_bias(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # MATH: Calculate relative positional bias between queries and keys.
        # MATH: RelPos = q_pos - k_pos. Clamped to [-max_relative_position, max_relative_position]
        if frame_indices is not None:
            coords = frame_indices.to(device=device, dtype=torch.float32)
            rel_pos = coords.unsqueeze(-1) - coords.unsqueeze(-2)
        else:
            # Inject the missing batch dimension [1, T]
            coords = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(0)
            rel_pos = coords.unsqueeze(-1) - coords.unsqueeze(-2)

        rel_pos_clamped = torch.clamp(
            rel_pos, -self.max_relative_position, self.max_relative_position
        )
        rel_pos_idx = (rel_pos_clamped + self.max_relative_position).long()
        values = self.relative_position_bias(rel_pos_idx)
        # values shape: (1, T, T, kv_heads, groups)
        values = values.view(1, T, T, self.kv_heads, self.groups)
        return values.permute(0, 3, 4, 1, 2).to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q_in = self.q_norm(x)
        k_in = self.k_norm(x)
        q = (
            self.q_proj(q_in)
            .view(B, T, self.kv_heads, self.groups, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )
        k = (
            self.k_proj(k_in)
            .view(B, T, self.kv_heads, 1, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )
        v = (
            self.v_proj(x)
            .view(B, T, self.kv_heads, 1, self.head_dim)
            .permute(0, 2, 3, 1, 4)
        )

        attn_mask = (
            self._get_relative_position_bias(T, x.device, x.dtype, frame_indices)
            .expand(B, -1, -1, -1, -1)
            .contiguous()
        )
        if key_padding_mask is not None:
            pad = key_padding_mask.view(B, 1, 1, 1, T)
            attn_mask = attn_mask.masked_fill(pad, float("-inf"))

        q = q.reshape(B * self.kv_heads, self.groups, T, self.head_dim)
        k = k.repeat_interleave(self.groups, dim=1)
        v = v.repeat_interleave(self.groups, dim=1)
        k = k.reshape(B * self.kv_heads, self.groups, T, self.head_dim)
        v = v.reshape(B * self.kv_heads, self.groups, T, self.head_dim)
        attn_mask = attn_mask.reshape(B * self.kv_heads, self.groups, T, T)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        attn_out = (
            attn_out.view(B, self.kv_heads, self.groups, T, self.head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(B, T, -1)
        )
        return self.out_proj(attn_out)


class SpatialTemporalSE(nn.Module):
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.Linear(d_model, d_model // reduction, bias=False),
            nn.GELU(),
            nn.Linear(d_model // reduction, d_model, bias=False),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Linear(d_model, 1, bias=False), nn.Sigmoid())

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).to(x.dtype)
            x_valid = x * valid_mask
            mean_x = x_valid.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            mean_x = x.mean(dim=1)
        return x * torch.max(self.cSE(mean_x).unsqueeze(1), self.sSE(x))


class ConvNeXtTemporalBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        hidden = channels * expansion
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=0,
            groups=channels,
            padding_mode="reflect",
        )
        self.norm = RMSNorm(channels)
        self.pw_conv1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(hidden, channels)
        self.se = SpatialTemporalSE(channels)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        if x.size(1) < 4:
            xp = F.pad(x.transpose(1, 2), (3, 3), mode="replicate")
        else:
            xp = F.pad(x.transpose(1, 2), (3, 3), mode="reflect")

        y = F.conv1d(
            xp, self.dw_conv.weight, self.dw_conv.bias, groups=self.dw_conv.groups
        )

        # MATH: Manual padding added 6 to sequence length. F.conv1d(padding=0) with kernel=7 reduces it exactly back to T. No slicing needed.
        y = y.transpose(1, 2)

        if key_padding_mask is not None:
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        y = self.norm(y)
        y = self.pw_conv2(self.act(self.pw_conv1(y)))
        y = self.se(y, key_padding_mask=key_padding_mask)
        if key_padding_mask is not None:
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return y


class BiMamba2SSMBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 320,
        expand: int = 2,
        headdim: int = 80,
        d_state: int = 16,
        d_conv: int = 4,
        ffn_dim: int = 1280,
        drop_path_rate: float = 0.1,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        if self.d_inner % headdim == 0:
            self.headdim = headdim
            self.nheads = self.d_inner // headdim
        else:
            possible_heads = [
                h for h in range(1, self.d_inner + 1) if self.d_inner % h == 0
            ]
            target_heads = max(1, self.d_inner // headdim)
            self.nheads = min(possible_heads, key=lambda h: abs(h - target_heads))
            self.headdim = self.d_inner // self.nheads
        self.d_state = d_state

        self.norm1 = RMSNorm(d_model)
        self.d_in_proj = 2 * self.d_inner + 2 * self.nheads * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, self.d_in_proj, bias=False)
        self.bwd_proj = nn.Linear(
            d_model, self.d_inner + 2 * self.nheads * d_state + self.nheads, bias=False
        )

        self.fwd_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.bwd_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        A_init = torch.arange(1, self.nheads + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A_init))
        dt_init = torch.rand(self.nheads) * 0.099 + 0.001
        self.dt_bias = nn.Parameter(torch.log(torch.exp(dt_init) - 1))

        self.head_norm_fwd = RMSNorm(self.headdim)
        self.head_norm_bwd = RMSNorm(self.headdim)
        self.gated_norm = RMSNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.gamma_1 = nn.Parameter(init_values * torch.ones(d_model))
        self.drop_path1 = DropPath(drop_path_rate)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)
        self.gamma_2 = nn.Parameter(init_values * torch.ones(d_model))
        self.drop_path2 = DropPath(drop_path_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.in_proj.weight)
        nn.init.orthogonal_(self.bwd_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight)

        # Scale B and C initialization
        start_fwd = 2 * self.d_inner
        end_fwd = start_fwd + 2 * self.nheads * self.d_state
        self.in_proj.weight.data[start_fwd:end_fwd] /= math.sqrt(self.d_inner)

        start_bwd = self.d_inner
        end_bwd = start_bwd + 2 * self.nheads * self.d_state
        self.bwd_proj.weight.data[start_bwd:end_bwd] /= math.sqrt(self.d_inner)

    def _ssd_multihead_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        B_sz, T_sz, H_sz, P_sz = x.shape
        N_sz = B.shape[-1]
        device = x.device

        if reverse:
            x = x.flip(1)
            dt = dt.flip(1)
            B = B.flip(1)
            C = C.flip(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.flip(1)

        # MATH: Discretize continuous time scale: delta_t = softplus(dt)
        # MATH: Compute log decay for state transition: log(A_bar) = -delta_t * A
        dt_act = F.softplus(dt).clamp(max=20.0)
        decay_mag = (dt_act * A.view(1, 1, H_sz)).clamp(min=1e-4, max=20.0)
        log_decay = -decay_mag

        if key_padding_mask is not None:
            log_decay = log_decay.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        Q = min(chunk_size, T_sz)
        pad_len = (Q - (T_sz % Q)) % Q
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
            log_decay = F.pad(log_decay, (0, 0, 0, pad_len), value=-1e4)
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            dt_act = F.pad(dt_act, (0, 0, 0, pad_len), value=0.0)

        T_pad = x.shape[1]
        n_chunks = T_pad // Q

        x_chunk = x.view(B_sz, n_chunks, Q, H_sz, P_sz).permute(0, 3, 1, 2, 4)
        B_chunk = B.view(B_sz, n_chunks, Q, H_sz, N_sz).permute(0, 3, 1, 2, 4)
        C_chunk = C.view(B_sz, n_chunks, Q, H_sz, N_sz).permute(0, 3, 1, 2, 4)
        ld_chunk = log_decay.view(B_sz, n_chunks, Q, H_sz).permute(0, 3, 1, 2)
        dt_chunk = dt_act.view(B_sz, n_chunks, Q, H_sz).permute(0, 3, 1, 2)

        # MATH: Compute intra-chunk output using semi-separable matrix M = C * exp(log_decay) * B
        B_chunk_dt = B_chunk * dt_chunk.unsqueeze(-1)
        CB = torch.matmul(C_chunk, B_chunk_dt.transpose(-1, -2)) / math.sqrt(N_sz)
        cum_decay = (
            ld_chunk.to(torch.float32).cumsum(dim=-1).to(ld_chunk.dtype)
        )  # MATH: ld_chunk is [B, H, n, Q] after permute, so dim=-1 is Q
        decay_diff = cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)
        causal_mask = torch.tril(torch.ones(Q, Q, device=device, dtype=torch.bool))
        M = torch.exp(decay_diff.masked_fill(~causal_mask, -float("inf")))

        Y_intra = torch.matmul(M * CB, x_chunk)

        # MATH: Calculate total decay across the entire chunk to propagate hidden state to next chunk
        log_chunk_decay = ld_chunk.sum(
            dim=-1
        )  # MATH: sum over sequence length Q (dim=-1)
        decay_to_end = torch.exp(
            cum_decay[:, :, :, -1:] - cum_decay
        )  # FIX: proper indexing for Q dimension

        x_weighted = x_chunk * decay_to_end.unsqueeze(-1)
        state_gen = torch.einsum("bhcqp, bhcqn -> bhcpn", x_weighted, B_chunk_dt)

        # MATH: Compute inter-chunk exponential decay M_inter for passing states between chunks
        L = log_chunk_decay.cumsum(dim=2)
        zeros_L = torch.zeros_like(L[:, :, :1])
        L_shifted = torch.cat([zeros_L, L[:, :, :-1]], dim=2)

        diff_L = L_shifted.unsqueeze(-1) - L.unsqueeze(-2)
        mask_inter = torch.tril(
            torch.ones(n_chunks, n_chunks, device=device, dtype=torch.bool), diagonal=-1
        )
        M_inter = torch.exp(diff_L.masked_fill(~mask_inter, -float("inf")))

        P_sz, N_sz = state_gen.shape[-2], state_gen.shape[-1]
        state_gen_flat = state_gen.contiguous().reshape(
            B_sz, H_sz, n_chunks, P_sz * N_sz
        )
        state_stack_flat = torch.einsum("bhij, bhjk -> bhik", M_inter, state_gen_flat)
        state_stack = state_stack_flat.reshape(B_sz, H_sz, n_chunks, P_sz, N_sz)

        C_state = torch.einsum(
            "bhcqn, bhcpn -> bhcqp", C_chunk, state_stack
        ) / math.sqrt(N_sz)
        decay_from_start = torch.exp(cum_decay).unsqueeze(-1)
        Y_inter = C_state * decay_from_start

        Y = Y_intra + Y_inter
        Y_flat = Y.permute(0, 2, 3, 1, 4).reshape(B_sz, T_pad, H_sz, P_sz)

        if reverse:
            return Y_flat[:, :T_sz].flip(1)

        return Y_flat[:, :T_sz]

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        residual = x
        xn = self.norm1(x)
        B_sz, T_sz, _ = xn.shape

        projected = self.in_proj(xn)
        x_proj, z, B_ssm_fwd, C_ssm_fwd, dt_fwd = torch.split(
            projected,
            [
                self.d_inner,
                self.d_inner,
                self.nheads * self.d_state,
                self.nheads * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        projected_bwd = self.bwd_proj(xn)
        x_proj_bwd, B_ssm_bwd, C_ssm_bwd, dt_bwd = torch.split(
            projected_bwd,
            [
                self.d_inner,
                self.nheads * self.d_state,
                self.nheads * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        x_conv_fwd = F.silu(
            self.fwd_conv1d(x_proj.transpose(1, 2))[:, :, :T_sz].transpose(1, 2)
        )
        x_conv_bwd = F.silu(
            self.bwd_conv1d(x_proj_bwd.transpose(1, 2))[:, :, -T_sz:].transpose(1, 2)
        )

        x_fwd_h = x_conv_fwd.view(B_sz, T_sz, self.nheads, self.headdim)
        x_bwd_h = x_conv_bwd.view(B_sz, T_sz, self.nheads, self.headdim)
        B_h_fwd = B_ssm_fwd.view(B_sz, T_sz, self.nheads, self.d_state)
        C_h_fwd = C_ssm_fwd.view(B_sz, T_sz, self.nheads, self.d_state)

        B_h_bwd = B_ssm_bwd.view(B_sz, T_sz, self.nheads, self.d_state)
        C_h_bwd = C_ssm_bwd.view(B_sz, T_sz, self.nheads, self.d_state)

        A = F.softplus(self.A_log)

        # MATH: dt_bias is required before softplus discretization to initialize stable delta time steps
        y_fwd = self._ssd_multihead_scan(
            x_fwd_h,
            dt_fwd + self.dt_bias,
            A,
            B_h_fwd,
            C_h_fwd,
            key_padding_mask=key_padding_mask,
            reverse=False,
        )
        y_bwd = self._ssd_multihead_scan(
            x_bwd_h,
            dt_bwd + self.dt_bias,
            A,
            B_h_bwd,
            C_h_bwd,
            key_padding_mask=key_padding_mask,
            reverse=True,
        )

        y_normed = 0.5 * (self.head_norm_fwd(y_fwd) + self.head_norm_bwd(y_bwd))
        y_flat = y_normed.reshape(B_sz, T_sz, self.d_inner)

        y_gated = self.gated_norm(y_flat * F.silu(z))
        out = self.out_proj(y_gated)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        x = residual + self.drop_path1(self.gamma_1 * out)
        x2 = self.ffn(self.norm2(x))
        if key_padding_mask is not None:
            x2 = x2.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + self.drop_path2(self.gamma_2 * x2)

        return x


class MobileConformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int = 320,
        nhead: int = 8,
        dim_feedforward: int = 1280,
        dropout_p: float = 0.1,
        drop_path: float = 0.0,
        num_enc_layers: int = 8,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.ffn1_norm = RMSNorm(d_model)
        self.ffn1 = SwiGLUFFN(d_model, dim_feedforward, num_layers=num_enc_layers)
        self.drop_path_ffn1 = DropPath(drop_path)
        self.gamma_ffn1 = nn.Parameter(init_values * torch.ones(d_model))

        self.mha_norm = RMSNorm(d_model)
        self.mha = GroupedQueryEncoderAttention(
            d_model=d_model, nhead=nhead, kv_heads=2
        )
        self.drop_path_mha = DropPath(drop_path)
        self.gamma_mha = nn.Parameter(init_values * torch.ones(d_model))

        self.conv_norm = RMSNorm(d_model)
        self.conv_block = ConvNeXtTemporalBlock(d_model)
        self.drop_path_conv = DropPath(drop_path)
        self.gamma_conv = nn.Parameter(init_values * torch.ones(d_model))

        self.ffn2_norm = RMSNorm(d_model)
        self.ffn2 = SwiGLUFFN(d_model, dim_feedforward, num_layers=num_enc_layers)
        self.drop_path_ffn2 = DropPath(drop_path)
        self.gamma_ffn2 = nn.Parameter(init_values * torch.ones(d_model))

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + 0.5 * self.drop_path_ffn1(
            self.gamma_ffn1 * self.ffn1(self.ffn1_norm(x))
        )
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + self.drop_path_mha(
            self.gamma_mha
            * self.mha(self.mha_norm(x), key_padding_mask=key_padding_mask)
        )
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        # MATH: Split into class token and sequence for temporal convolution.
        # MATH: Apply ConvNeXt 1D depthwise convolution for local temporal modeling.
        cls_t = x[:, :1]
        x_seq = x[:, 1:]
        seq_mask = key_padding_mask[:, 1:] if key_padding_mask is not None else None
        xc_seq = self.conv_block(self.conv_norm(x_seq), key_padding_mask=seq_mask)
        xc = torch.cat([cls_t, xc_seq], dim=1)

        x = x + self.drop_path_conv(self.gamma_conv * xc)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        x = x + 0.5 * self.drop_path_ffn2(
            self.gamma_ffn2 * self.ffn2(self.ffn2_norm(x))
        )
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class LandmarkTrajectory1DStem(nn.Module):
    def __init__(
        self, in_channels: int = 9, num_keypoints: int = 60, out_dim: int = 128
    ):
        super().__init__()
        in_dim = num_keypoints * in_channels
        self.conv1d_stem = nn.Sequential(
            nn.Conv1d(in_dim, 256, kernel_size=7, padding=3, groups=1),
            nn.GroupNorm(8, 256),
            nn.GELU(),
            nn.Conv1d(256, 256, kernel_size=5, padding=2, groups=256),
            nn.Conv1d(256, out_dim, kernel_size=1),
            nn.GroupNorm(8, out_dim),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = x.size(0), x.size(1)
        x_flat = x.reshape(B, T, -1) if x.dim() == 4 else x
        x_t = x_flat.transpose(1, 2)
        if mask is not None:
            x_t = x_t * mask.unsqueeze(1).to(x_t.dtype)
        feat_seq = self.conv1d_stem(x_t).transpose(1, 2)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(-1).to(feat_seq.dtype)
        return self.out_proj(feat_seq)


# ==============================================================================
# 7. TRANSFORMER DECODER WITH EOS GRAMMAR PROTECTION
# ==============================================================================


class RoPEEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None], persistent=False)
        self._cache_len = seq_len

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        return torch.cat([-x[..., half:], x[..., :half]], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        S = q.shape[-2]
        total_len = S + offset
        if total_len <= self._cache_len:
            cos = self.cos_cache[:, :, offset:total_len, :].to(q.dtype)
            sin = self.sin_cache[:, :, offset:total_len, :].to(q.dtype)
        else:
            inv_freq = self.inv_freq.to(q.device)
            t = torch.arange(offset, total_len, device=q.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()[None, None].to(q.dtype)
            sin = emb.sin()[None, None].to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class GroupedQuerySelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 320,
        nhead: int = 8,
        kv_heads: int = 2,
        max_seq_len: int = 256,
    ):
        super().__init__()
        assert nhead % kv_heads == 0
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.groups = nhead // kv_heads
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
        self.k_norm = RMSNorm(d_model)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=max_seq_len)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = x.shape
        q_in = self.q_norm(x)
        k_in = self.k_norm(x)
        q = self.q_proj(q_in).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].size(2) if past_key_value is not None else 0
        q, k = self.rope(q, k, offset=past_len)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        current_key_value = (k, v) if use_cache else None

        k_exp = k.repeat_interleave(self.groups, dim=1)
        v_exp = v.repeat_interleave(self.groups, dim=1)
        total_len = k.size(2)

        if T == 1:
            out = F.scaled_dot_product_attention(q, k_exp, v_exp, scale=self.scale)
        else:
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, scale=self.scale, is_causal=True
            )

        out = self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        if use_cache:
            return out, current_key_value
        return out


class DecoderCrossAttention(nn.Module):
    def __init__(self, d_model: int = 320, nhead: int = 8, kv_heads: int = 2):
        super().__init__()
        assert nhead % kv_heads == 0
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.groups = nhead // kv_heads
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
        self.k_norm = RMSNorm(d_model)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.k_proj.weight, std=0.02)
        nn.init.normal_(self.v_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = tgt.shape
        q_in = self.q_norm(tgt)
        q = self.q_proj(q_in).view(B, T, self.nhead, self.head_dim).transpose(1, 2)

        if past_key_value is not None:
            k, v = past_key_value
        else:
            S = memory.size(1)
            k_in = self.k_norm(memory)
            k = (
                self.k_proj(k_in)
                .view(B, S, self.kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(memory)
                .view(B, S, self.kv_heads, self.head_dim)
                .transpose(1, 2)
            )

        # Cross-Attention keys/values are static from the encoder
        current_key_value = (k, v) if use_cache else None

        k_exp = k.repeat_interleave(self.groups, dim=1)
        v_exp = v.repeat_interleave(self.groups, dim=1)
        S = k.size(2)

        attn_mask = None
        if memory_key_padding_mask is not None:
            # MATH: enc_mask is True for Valid. SDPA boolean mask expects True for valid.
            attn_mask = memory_key_padding_mask.view(B, 1, 1, S).bool()

        # MATH: Scaled Dot-Product Attention: softmax(Q * K^T / sqrt(d)) * V
        attn_out = F.scaled_dot_product_attention(
            q,
            k_exp,
            v_exp,
            attn_mask=attn_mask,
        )
        out = self.o_proj(attn_out.transpose(1, 2).reshape(B, T, -1))

        if use_cache:
            return out, current_key_value
        return out


class ASLDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 320,
        nhead: int = 8,
        kv_heads: int = 2,
        ffn_dim: int = 1280,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        iv = 1e-4
        self.norm1 = RMSNorm(d_model)
        self.self_attn = GroupedQuerySelfAttention(
            d_model, nhead, kv_heads, max_seq_len
        )
        self.gamma1 = nn.Parameter(iv * torch.ones(d_model))

        self.norm2 = RMSNorm(d_model)
        self.cross_attn = DecoderCrossAttention(d_model, nhead, kv_heads)
        self.gamma2 = nn.Parameter(iv * torch.ones(d_model))

        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model=d_model, dim_feedforward=ffn_dim)
        self.gamma3 = nn.Parameter(iv * torch.ones(d_model))
        self.drop1 = DropPath(dropout)
        self.drop2 = DropPath(dropout)
        self.drop3 = DropPath(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ]:
        if use_cache:
            sa_out, new_self_kv = self.self_attn(
                self.norm1(tgt), past_key_value=past_self_kv, use_cache=True
            )
            tgt = tgt + self.gamma1 * sa_out
            ca_out, new_cross_kv = self.cross_attn(
                self.norm2(tgt),
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
                past_key_value=past_cross_kv,
                use_cache=True,
            )
            tgt = tgt + self.gamma2 * ca_out
            tgt = tgt + self.gamma3 * self.ffn(self.norm3(tgt))
            return tgt, new_self_kv, new_cross_kv
        else:
            tgt = tgt + self.drop1(self.gamma1 * self.self_attn(self.norm1(tgt)))
            tgt = tgt + self.drop2(
                self.gamma2
                * self.cross_attn(self.norm2(tgt), memory, memory_key_padding_mask)
            )
            tgt = tgt + self.drop3(self.gamma3 * self.ffn(self.norm3(tgt)))
            return tgt


class ASLTransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 320,
        nhead: int = 8,
        kv_heads: int = 2,
        num_layers: int = 8,
        ffn_dim: int = 1280,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        csv_path: Optional[Union[str, Path]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.input_token_dropout = 0.12

        self.token_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=GlossVocabulary.PAD_ID
        )

        self.asl_lex_emb = RichASLLexEmbeddingTable(
            vocab_size=vocab_size,
            d_model=d_model,
            csv_path=csv_path,
            label_to_idx=label_to_idx,
        )

        self.emb_drop = nn.Dropout(dropout * 0.5)
        self.emb_scale = math.sqrt(d_model)

        self.layers = nn.ModuleList(
            [
                ASLDecoderLayer(d_model, nhead, kv_heads, ffn_dim, dropout, max_seq_len)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, List[Tuple]],
    ]:
        B, S = tgt_ids.shape

        if self.training and self.input_token_dropout > 0:
            drop_mask = (
                torch.rand(tgt_ids.shape, device=tgt_ids.device)
                < self.input_token_dropout
            )
            drop_mask &= tgt_ids >= GlossVocabulary.OFFSET
            dropped_tgt_ids = torch.where(
                drop_mask,
                torch.tensor(GlossVocabulary.UNK_ID, device=tgt_ids.device),
                tgt_ids,
            )
        else:
            dropped_tgt_ids = tgt_ids

        lex_embs = self.asl_lex_emb(dropped_tgt_ids)
        valid_lex_mask = (
            (tgt_ids != GlossVocabulary.PAD_ID).unsqueeze(-1).to(lex_embs.dtype)
        )
        h = (
            self.token_emb(dropped_tgt_ids) * self.emb_scale
            + lex_embs * self.emb_scale * valid_lex_mask
        )
        h = self.emb_drop(h)

        new_key_values = [] if use_cache else None

        for idx, layer in enumerate(self.layers):
            past_self_kv = past_key_values[idx][0] if past_key_values else None
            past_cross_kv = past_key_values[idx][1] if past_key_values else None
            if use_cache:
                h, n_self_kv, n_cross_kv = layer(
                    h,
                    memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    past_self_kv=past_self_kv,
                    past_cross_kv=past_cross_kv,
                    use_cache=True,
                )
                new_key_values.append((n_self_kv, n_cross_kv))
            else:
                h = layer(h, memory, memory_key_padding_mask=memory_key_padding_mask)

        h = self.final_norm(h)
        logits = self.lm_head(h)
        if use_cache:
            return logits, h, new_key_values
        return logits, h


# ==============================================================================
# 8. AUXILIARY HEADS & HOMOSCEDASTIC LOSS WRAPPER WITH NULL-LOSS DETACH
# ==============================================================================


class HomoscedasticLossWrapper(nn.Module):
    """
    Homoscedastic Task Uncertainty Loss Weighting (Kendall & Gal, CVPR 2018).
    Bypasses gradient propagation for zero-valued or uncalculated losses to prevent divergence.
    """

    def __init__(self, num_losses: int = 7):
        super().__init__()
        self.num_losses = num_losses

        # Seq(CE) ~8.0, CTC ~10.0, DenseSem ~1.0, InfoNCE ~4.0, SupCon ~2.0, Domain ~1.0, MLM ~0.1
        init_vals = [
            math.log(8.0),  # Seq
            math.log(10.0),  # CTC
            math.log(1.0),  # Dense Sem
            math.log(4.0),  # InfoNCE
            math.log(2.0),  # SupCon
        ]
        # Pad with 0.0 just in case you pass more losses
        while len(init_vals) < num_losses:
            init_vals.append(0.0)

        self.log_vars = nn.Parameter(
            torch.tensor(init_vals[:num_losses], dtype=torch.float32)
        )

    def forward(self, losses: List[torch.Tensor], accum_steps: int = 1) -> torch.Tensor:
        # MATH: Homoscedastic uncertainty weighting: sum_i (0.5 * exp(-s_i) * L_i + 0.5 * s_i)
        total_loss = torch.tensor(0.0, device=losses[0].device, dtype=losses[0].dtype)
        for i, loss in enumerate(losses):
            if i < self.num_losses:
                s = self.log_vars[i].clamp(min=-4.0, max=2.0).to(loss.device)
                prec = torch.exp(-s)
                valid_mask = (loss.detach().abs() > 1e-6).to(loss.dtype)
                s_penalty = 0.5 * s * valid_mask
                total_loss = total_loss + 0.5 * prec * loss * valid_mask + s_penalty
        return total_loss


class DiffAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        import torch.distributed as dist

        ctx.rank = dist.get_rank()
        ctx.world_size = dist.get_world_size()
        local_size = torch.tensor(
            [tensor.size(0)], dtype=torch.long, device=tensor.device
        )
        size_list = [torch.zeros_like(local_size) for _ in range(ctx.world_size)]
        dist.all_gather(size_list, local_size)
        sizes = [s.item() for s in size_list]
        max_size = max(sizes)
        ctx.batch_sizes = sizes
        if tensor.size(0) < max_size:
            pad_tensor = torch.zeros(
                (max_size - tensor.size(0), *tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor_padded = torch.cat([tensor, pad_tensor], dim=0)
        else:
            tensor_padded = tensor
        gathered = [torch.zeros_like(tensor_padded) for _ in range(ctx.world_size)]
        dist.all_gather(gathered, tensor_padded)
        truncated = [g[:s] for g, s in zip(gathered, sizes)]
        return torch.cat(truncated, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        import torch.distributed as dist

        grad_output = grad_output.contiguous()
        grad_chunks = torch.split(grad_output, ctx.batch_sizes, dim=0)
        max_size = max(ctx.batch_sizes)
        padded_chunks = []
        for g, s in zip(grad_chunks, ctx.batch_sizes):
            if s < max_size:
                pad_tensor = torch.zeros(
                    (max_size - s, *g.shape[1:]), dtype=g.dtype, device=g.device
                )
                padded_chunks.append(torch.cat([g, pad_tensor], dim=0))
            else:
                padded_chunks.append(g)
        grad_tensor = torch.zeros_like(padded_chunks[0])
        dist.reduce_scatter(grad_tensor, padded_chunks)
        grad_tensor /= ctx.world_size
        return grad_tensor[: ctx.batch_sizes[ctx.rank]].contiguous()


def diff_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    import torch.distributed as dist

    if dist.is_initialized():
        return DiffAllGather.apply(tensor)
    return tensor


class CTCHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        return self.proj(enc_seq)


class CrossModalInfoNCE(nn.Module):
    def __init__(self, init_temp: float = 0.07):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, vis_emb: torch.Tensor, sent_emb: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(F.softplus(self.log_temp), min=0.05)
        device = vis_emb.device
        import torch.distributed as dist

        if _XLA_AVAILABLE and "xla" in str(device).lower():
            import torch_xla.core.xla_model as xm

            world_size = xm.xrt_world_size()
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1
        global_b = vis_emb.size(0) * world_size
        if global_b < 2:
            return torch.tensor(0.0, device=vis_emb.device, requires_grad=True)
        v = F.normalize(vis_emb.float(), p=2, dim=-1)
        s = F.normalize(sent_emb.float(), p=2, dim=-1)

        if _XLA_AVAILABLE and "xla" in str(vis_emb.device).lower():
            import torch_xla.core.functions as xf
            import torch_xla.core.xla_model as xm

            v_global = xf.all_gather(v, dim=0)
            s_global = xf.all_gather(s, dim=0)
            rank = xm.get_ordinal()
        elif dist.is_initialized():
            v_global = diff_all_gather(v)
            s_global = diff_all_gather(s)
            rank = dist.get_rank()
        else:
            v_global = v
            s_global = s
            rank = 0

        local_bs = torch.tensor([v.size(0)], device=v.device, dtype=torch.long)
        if _XLA_AVAILABLE and "xla" in str(v.device).lower():
            bs_global = xf.all_gather(local_bs, dim=0)
        elif dist.is_initialized():
            bs_global = diff_all_gather(local_bs)
        else:
            bs_global = local_bs

        offset = bs_global[:rank].sum()
        lbl = torch.arange(v.size(0), device=vis_emb.device) + offset

        # MATH: Symmetric InfoNCE loss with temperature scaling for cross-modal contrastive learning
        sim1 = torch.matmul(v, s_global.T) / temp
        sim2 = torch.matmul(s, v_global.T) / temp

        s_sim = torch.matmul(s, s_global.T)
        false_neg_mask = (s_sim > 0.99) & (
            torch.arange(s_global.size(0), device=s.device).unsqueeze(0)
            != lbl.unsqueeze(1)
        )
        sim1.masked_fill_(false_neg_mask, -1e9)
        sim2.masked_fill_(false_neg_mask, -1e9)
        return (F.cross_entropy(sim1, lbl) + F.cross_entropy(sim2, lbl)) * 0.5


class DenseSentenceSemanticLoss(nn.Module):
    def __init__(self, d_model: int = 320, embed_dim: int = 256):
        super().__init__()
        self.proj_pred = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            RMSNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.proj_gt = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            RMSNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        for p in self.proj_gt.parameters():
            p.requires_grad = False

    def update_momentum(self, m=0.01):
        with torch.no_grad():
            for p_target, p_online in zip(
                self.proj_gt.parameters(), self.proj_pred.parameters()
            ):
                p_target.lerp_(p_online.detach(), weight=m)

    def forward(
        self,
        last_hidden: torch.Tensor,
        gt_lex_embs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:

        m = valid_mask.unsqueeze(-1).float()
        valid_counts = m.sum(dim=1).clamp(min=1.0)
        has_tokens = (m.sum(dim=(1, 2)) > 0).float()

        gt_sent = (gt_lex_embs * m).sum(dim=1) / valid_counts

        p = F.normalize(self.proj_pred(last_hidden).float(), p=2, dim=-1, eps=1e-8)
        g = F.normalize(self.proj_gt(gt_sent).float(), p=2, dim=-1, eps=1e-8).detach()

        # MATH: Cosine similarity loss to align predicted sentence embedding with ground truth
        cos_sim = (p * g).sum(dim=-1)
        loss = (1.0 - cos_sim) * has_tokens
        return loss.sum() / has_tokens.sum().clamp(min=1.0)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, features: torch.Tensor, labels: torch.Tensor, enqueue: bool = True
    ) -> torch.Tensor:
        if labels is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        features = F.normalize(features, p=2, dim=-1)
        B, device = features.shape[0], features.device
        has_labels = (labels.abs().sum() > 0).float()
        import torch.distributed as dist

        if _XLA_AVAILABLE and "xla" in str(device).lower():
            import torch_xla.core.functions as xf
            import torch_xla.core.xla_model as xm

            all_f = xf.all_gather(features, dim=0)
            all_l = xf.all_gather(labels, dim=0)
            rank = xm.get_ordinal()
        elif dist.is_initialized():
            all_f = diff_all_gather(features)
            all_l = diff_all_gather(labels)
            rank = dist.get_rank()
        else:
            all_f = features
            all_l = labels
            rank = 0

        local_bs = torch.tensor([features.size(0)], device=device, dtype=torch.long)
        if _XLA_AVAILABLE and "xla" in str(device).lower():
            import torch_xla.core.functions as xf

            bs_global = xf.all_gather(local_bs, dim=0)
        elif dist.is_initialized():
            bs_global = diff_all_gather(local_bs)
        else:
            bs_global = local_bs

        offset = bs_global[:rank].sum()
        ids = torch.arange(features.size(0), device=device) + offset

        if _XLA_AVAILABLE and "xla" in str(device).lower():
            import torch_xla.core.functions as xf

            all_ids = xf.all_gather(ids, dim=0)
        elif dist.is_initialized():
            all_ids = diff_all_gather(ids)
        else:
            all_ids = ids

        # MATH: Supervised Contrastive Loss (SupCon): Groups positive examples and pushes away negatives.
        # MATH: L = -1/|P| * sum_{p in P} log( exp(sim_p) / sum_{all} exp(sim) )
        pos_mask = torch.eq(labels.view(-1, 1), all_l.view(1, -1)).float()
        self_m = torch.eq(ids.view(-1, 1), all_ids.view(1, -1)).float()
        pos_mask *= 1.0 - self_m

        cos_sim = torch.matmul(features, all_f.T)
        pos_logits = cos_sim / self.temperature

        exp_logits = torch.exp(pos_logits - pos_logits.max(dim=-1, keepdim=True)[0]) * (
            1.0 - self_m
        )
        denom = torch.clamp(exp_logits.sum(dim=-1, keepdim=True), min=1e-8)

        max_logits = pos_logits.max(dim=-1, keepdim=True)[0]
        log_prob = (pos_logits - max_logits) - torch.log(denom)

        pos_count = pos_mask.sum(dim=-1)
        valid_rows = (pos_count > 0).float()

        row_loss = -(log_prob * pos_mask).sum(dim=-1) / pos_count.clamp(min=1.0)
        loss = (row_loss * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss * has_labels


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: Union[float, torch.Tensor] = 1.0):
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(float(alpha), device=x.device, dtype=x.dtype)
        ctx.save_for_backward(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        return grad_output.neg() * alpha, None


class LandmarkReconstructionHead(nn.Module):
    def __init__(self, d_model: int = 320, out_dim: int = 540):
        super().__init__()
        self.recon = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        return self.recon(enc_seq)


# ==============================================================================
# 9. ASL FOUNDATION MODEL MAIN AGGREGATOR
# ==============================================================================


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class ASLFoundationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2484,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        d_enc: int = 320,
        nhead_enc: int = 8,
        num_enc_layers: int = 8,
        ffn_enc: int = 1280,
        d_dec: int = 320,
        nhead_dec: int = 8,
        kv_heads_dec: int = 2,
        num_dec_layers: int = 8,
        ffn_dec: int = 1280,
        dropout: float = 0.1,
        drop_path_rate: float = 0.25,
        max_enc_len: int = 320,
        max_dec_len: int = 196,
        num_domains: int = 4,
        csv_path: Optional[Union[str, Path]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        use_mamba: bool = True,
        tome_r: int = 80,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_enc = d_enc
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.use_mamba = use_mamba
        self.tome_r = tome_r
        input_dim = num_keypoints * channels_per_kp

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_enc) * 0.02)
        self.visual_encoder = LandmarkTrajectory1DStem(
            in_channels=channels_per_kp, num_keypoints=num_keypoints, out_dim=128
        )
        self.input_stem = nn.Sequential(
            nn.Linear(input_dim + 128, d_enc), RMSNorm(d_enc), nn.GELU()
        )
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, num_enc_layers)]

        self.blocks = nn.ModuleList()
        for i in range(num_enc_layers):
            if i >= 4 and i % 2 == 0:
                self.blocks.append(TokenMergingBlock(r=30, d_model=d_enc))
            if use_mamba and i >= 4:
                self.blocks.append(
                    BiMamba2SSMBlock(
                        d_model=d_enc, expand=2, ffn_dim=ffn_enc, drop_path_rate=dpr[i]
                    )
                )
            else:
                self.blocks.append(
                    MobileConformerBlock(
                        d_model=d_enc,
                        nhead=nhead_enc,
                        dim_feedforward=ffn_enc,
                        drop_path=dpr[i],
                    )
                )

        self.tome_early = TokenMergingBlock(r=tome_r, d_model=d_enc)
        self.tome_deep = TokenMergingBlock(r=tome_r, d_model=d_enc)
        self.enc_final_norm = RMSNorm(d_enc)

        self.decoder = ASLTransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_dec,
            nhead=nhead_dec,
            kv_heads=kv_heads_dec,
            num_layers=num_dec_layers,
            ffn_dim=ffn_dec,
            dropout=dropout,
            max_seq_len=max_dec_len,
            csv_path=csv_path,
            label_to_idx=label_to_idx,
        )

        self.pos_enc = PositionalEncoding1D(d_enc)

        self.ctc_head = CTCHead(d_enc, vocab_size)
        self.mlm_head = LandmarkReconstructionHead(d_enc, input_dim)
        self.visual_proj = nn.Sequential(
            nn.Linear(d_enc, 256), RMSNorm(256), nn.GELU(), nn.Linear(256, 256)
        )
        self.sentence_proj = nn.Sequential(
            nn.Linear(d_dec, 256), RMSNorm(256), nn.GELU(), nn.Linear(256, 256)
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_enc, 256), RMSNorm(256), nn.GELU(), nn.Linear(256, 256)
        )
        self.domain_head = nn.Sequential(
            nn.Linear(d_enc, 192), RMSNorm(192), nn.GELU(), nn.Linear(192, num_domains)
        )
        self.xmodal_loss_fn = CrossModalInfoNCE(init_temp=0.07)
        self.dense_sem_loss = DenseSentenceSemanticLoss(d_model=d_dec, embed_dim=256)

    def _encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mlm_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:

        B, T = x.size(0), x.size(1)

        if mlm_mask is not None:
            used_mlm_mask = mlm_mask
            mask_shape = [1] * (x.dim() - 2)
            x_in = x * (~mlm_mask).view(B, T, *mask_shape).to(x.dtype)
        else:
            x_in = x
            used_mlm_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        if x_in.dim() == 4 and x_in.size(2) == 60 and x_in.size(3) >= 3:
            xk = x_in.clone()
            lh_nz = (xk[:, :, 18:39, :3] != 0).to(xk.dtype)
            rh_nz = (xk[:, :, 39:60, :3] != 0).to(xk.dtype)
            xk[:, :, 18:39, :3] = (xk[:, :, 18:39, :3] - xk[:, :, 18:19, :3]) * lh_nz
            xk[:, :, 39:60, :3] = (xk[:, :, 39:60, :3] - xk[:, :, 39:40, :3]) * rh_nz
            x_flat = xk.reshape(B, T, -1)
            v_tokens = self.visual_encoder(xk, mask=mask)
        else:
            x_flat = x_in.reshape(B, T, -1) if x_in.dim() == 4 else x_in
            v_tokens = self.visual_encoder(x_in, mask=mask)

        h = self.input_stem(torch.cat([x_flat, v_tokens], dim=-1))
        h = self.pos_enc(h)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)

        cur_mask = mask
        if cur_mask is not None:
            kpm = torch.cat(
                [torch.zeros((B, 1), dtype=torch.bool, device=h.device), ~cur_mask],
                dim=1,
            )
        else:
            kpm = None

        h_pre_tome = None  # NEW: Capture features before merging

        for idx, block in enumerate(self.blocks):
            if isinstance(block, TokenMergingBlock):
                if h_pre_tome is None:
                    h_pre_tome = h[:, 1:]  # Capture full-res temporal sequence for MLM

                cls_t = h[:, :1]
                seq_t = h[:, 1:]
                seq_t, cur_mask, routing_info = block(
                    seq_t, cur_mask, mlm_mask=used_mlm_mask
                )
                if "mlm_out" in routing_info and routing_info["mlm_out"] is not None:
                    used_mlm_mask = routing_info["mlm_out"]
                h = torch.cat([cls_t, seq_t], dim=1)
                if cur_mask is not None:
                    kpm = torch.cat(
                        [
                            torch.zeros((B, 1), dtype=torch.bool, device=h.device),
                            ~cur_mask,
                        ],
                        dim=1,
                    )
                else:
                    kpm = None
            else:
                h = block(h, key_padding_mask=kpm)

        h = self.enc_final_norm(h)
        # If no ToMe blocks ran, fallback to the output sequence
        if h_pre_tome is None:
            h_pre_tome = h[:, 1:]

        return (
            h[:, 0],
            h[:, 1:],
            cur_mask,
            used_mlm_mask,
            h_pre_tome,
            mask,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gloss_seq: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        grl_alpha: float = 1.0,
    ) -> Union[Optional[torch.Tensor], Dict]:

        (
            h_cls,
            h_seq,
            enc_mask,
            used_mlm_mask,
            h_pre_tome,
            orig_enc_mask,
        ) = self._encode(x, mask, mlm_mask=mlm_mask)

        dec_logits, dec_hidden = None, None
        if gloss_seq is not None:
            dec_in = gloss_seq[:, :-1].contiguous()
            if self.training:
                # Target Word Dropout to mitigate exposure bias
                mask = (torch.rand_like(dec_in, dtype=torch.float) < 0.15) & (
                    dec_in != GlossVocabulary.BOS_ID
                )
                dec_in = dec_in.masked_fill(mask, GlossVocabulary.UNK_ID)

            dec_logits, dec_hidden = self.decoder(
                dec_in, h_seq, memory_key_padding_mask=enc_mask
            )

        if not return_aux:
            return dec_logits

        # CTC gets the MERGED sequence. No more unmerged duplicated vectors!
        ctc_log_probs = F.log_softmax(self.ctc_head(h_seq), dim=-1)

        # MLM gets the FULL-RES Pre-ToMe sequence where physical coordinates still make sense
        if self.training and h_pre_tome is not None and used_mlm_mask.sum() > 0:
            mlm_logits = self.mlm_head(h_pre_tome)
        else:
            mlm_logits = None

        vis_emb = F.normalize(self.visual_proj(h_cls), p=2, dim=-1)
        if dec_hidden is not None and gloss_seq is not None:
            gt_tokens = gloss_seq[:, 1:]
            valid_mask = gt_tokens != GlossVocabulary.PAD_ID
            last_idx = (
                (valid_mask.sum(dim=1) - 1)
                .clamp(min=0, max=dec_hidden.size(1) - 1)
                .view(-1, 1, 1)
                .expand(-1, 1, dec_hidden.shape[-1])
            )
            last_hidden = dec_hidden.gather(1, last_idx).squeeze(1)
            sent_emb = F.normalize(self.sentence_proj(last_hidden), p=2, dim=-1)
        else:
            sent_emb = None

        proj_feats = F.normalize(self.contrastive_head(h_cls), p=2, dim=-1)

        if grl_alpha > 0.0:
            rev_cls = GradientReversalFunction.apply(h_cls, grl_alpha)
            domain_logits = self.domain_head(rev_cls)
        else:
            domain_logits = None

        return {
            "dec_logits": dec_logits,
            "dec_hidden": dec_hidden,
            "ctc_log_probs": ctc_log_probs,
            "mlm_logits": mlm_logits,
            "mlm_mask": used_mlm_mask,
            "orig_x": x,
            "vis_emb": vis_emb,
            "sent_emb": sent_emb,
            "proj_feats": proj_feats,
            "domain_logits": domain_logits,
            "enc_seq": h_seq,
            "enc_mask": enc_mask,
            "orig_enc_mask": orig_enc_mask,
        }


# ==============================================================================
# 10. LOSS COMPUTATION & TRAINING LOGIC
# ==============================================================================


def _compute_ctc_loss_safe(
    ctc_log_probs: torch.Tensor,
    gloss_seq: torch.Tensor,
    gloss_len: torch.Tensor,
    enc_mask: Optional[torch.Tensor],
    has_valid: torch.Tensor,
) -> torch.Tensor:
    device = ctc_log_probs.device
    B, T_enc = ctc_log_probs.size(0), ctc_log_probs.size(1)

    if enc_mask is not None:
        enc_len = enc_mask.sum(dim=-1).long()
    else:
        enc_len = torch.full((B,), T_enc, dtype=torch.long, device=device)

    raw_targets = gloss_seq[:, 1:].contiguous()
    valid_mask = (
        (raw_targets >= GlossVocabulary.OFFSET)
        & (raw_targets < 2484)
        & has_valid.unsqueeze(1)
    )
    targets = raw_targets[valid_mask]
    tgt_lengths = valid_mask.sum(dim=-1).long()

    valid_ctc = (enc_len >= tgt_lengths) & (tgt_lengths > 0) & (enc_len > 0)

    # MATH: tgt_lengths includes EOS, so we don't subtract 1 to maintain dimension balance for 1D target array
    # MATH: CTC target cannot contain the blank index (PAD_ID), so length must be exact.
    tgt_len_for_ctc = tgt_lengths
    loss_vec = F.ctc_loss(
        ctc_log_probs.float().transpose(0, 1),
        targets,
        enc_len.clamp(min=1, max=T_enc),
        tgt_len_for_ctc,
        blank=GlossVocabulary.PAD_ID,
        reduction="none",
        zero_infinity=True,
    )
    loss_vec = torch.nan_to_num(loss_vec)
    valid_f = valid_ctc.float()
    return (loss_vec * valid_f).sum() / valid_f.sum().clamp(min=1.0)


def _compute_mlm_loss_safe(
    mlm_logits: torch.Tensor, orig_x: torch.Tensor, mlm_mask: torch.Tensor
) -> torch.Tensor:
    B, T = orig_x.size(0), orig_x.size(1)
    target = orig_x.clone()
    if target.dim() == 3 and target.size(-1) % 60 == 0:
        target = target.view(B, T, 60, -1)

    if target.dim() == 4 and target.size(2) == 60 and target.size(3) >= 3:
        lh_nz = (target[:, :, 18:39, :3] != 0).to(target.dtype)
        rh_nz = (target[:, :, 39:60, :3] != 0).to(target.dtype)
        target[:, :, 18:39, :3] = (
            target[:, :, 18:39, :3] - target[:, :, 18:19, :3]
        ) * lh_nz
        target[:, :, 39:60, :3] = (
            target[:, :, 39:60, :3] - target[:, :, 39:40, :3]
        ) * rh_nz

    target = target.reshape(B, T, -1)
    mask_f = mlm_mask.unsqueeze(-1).float()
    loss = F.smooth_l1_loss(mlm_logits, target, reduction="none")
    divisor = mask_f.sum().clamp(min=1.0) * target.size(-1)
    loss = (loss * mask_f).sum() / divisor
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def train_epoch_tpu(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    loss_wrapper: HomoscedasticLossWrapper,
    ema: Optional[Any],
    supcon_fn: SupervisedContrastiveLoss,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 150,
    prec_dtype: torch.dtype = torch.bfloat16,
    is_master: bool = True,
    accum_steps: int = 4,
) -> Tuple[float, float]:
    model.train()

    if loss_wrapper is not None and len(list(loss_wrapper.parameters())) > 0:
        found = any(
            p in group["params"]
            for group in optimizer.param_groups
            for p in loss_wrapper.parameters()
        )
        if not found:
            optimizer.add_param_group({"params": loss_wrapper.parameters()})

    total_loss_val = torch.tensor(0.0, device=device, dtype=torch.float32)
    correct_tok_val = torch.tensor(0.0, device=device, dtype=torch.float32)
    total_tok_val = torch.tensor(0.0, device=device, dtype=torch.float32)

    tracker = {"loss": 0.0, "corr": 0.0, "total": 0.0}

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
    use_autocast = (
        not is_xla and "cuda" in str(device).lower() and prec_dtype != torch.float32
    )
    scaler = (
        torch.cuda.amp.GradScaler()
        if use_autocast and prec_dtype == torch.float16
        else None
    )

    progress = float(max(0, epoch)) / float(max(1, total_epochs - 1))
    grl_alpha_val = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
    grl_alpha = torch.zeros((), device=device, dtype=prec_dtype)
    grl_alpha.fill_(grl_alpha_val)

    label_smoothing = max(0.05, 0.15 - 0.10 * progress)
    teacher_forcing_ratio = max(0.50, 1.0 - 0.50 * progress)
    mixup_active = np.random.rand() < min(
        0.30, max(0.0, 0.30 * float(epoch - 5) / float(max(1, total_epochs - 5)))
    )

    POLY1_EPS = 1.0

    def compute_seq_loss(
        logits_f: torch.Tensor, gt_ids: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = gt_ids.reshape(-1)
        vf = valid_mask.reshape(-1)
        log_p = F.log_softmax(lf, dim=-1)
        p = torch.exp(log_p)
        # MATH: Poly1 Loss expands Cross Entropy via Taylor series for better gradient scaling.
        # MATH: L_poly = CE + eps * (1 - pt), smoothed with uniform distribution.
        ce_unsmoothed = F.nll_loss(
            log_p, tf, ignore_index=GlossVocabulary.PAD_ID, reduction="none"
        )
        pt = torch.exp(-ce_unsmoothed)
        ce_uniform = -log_p[..., 1:].mean(dim=-1)
        ce_smoothed = (
            1.0 - label_smoothing
        ) * ce_unsmoothed + label_smoothing * ce_uniform
        poly1 = ce_smoothed + POLY1_EPS * (1.0 - pt)
        return (poly1 * vf.float()).sum() / vf.float().sum().clamp(min=1.0)

    if is_xla:
        loader = pl.MpDeviceLoader(loader, device)

    total_batches = len(loader)
    if is_xla:
        local_batches = torch.tensor(
            [total_batches], dtype=torch.float32, device=device
        )
        xm.all_reduce("min", local_batches)
        min_batches = int(local_batches.item())
    else:
        min_batches = total_batches
    if is_xla:
        xm.set_rng_state(42 + epoch * 10000 + xm.get_ordinal())

    for step_idx, batch in enumerate(loader, start=1):
        if step_idx > min_batches:
            continue
        features = batch["feature"].to(device)
        mask = batch["mask"].to(device)
        labels = batch.get("label", torch.zeros(features.size(0), dtype=torch.long)).to(
            device
        )
        domain_tgts = batch.get("domain_label", torch.zeros_like(labels)).to(device)
        has_domain = batch.get("has_domain_label", torch.zeros_like(labels)).to(device)
        gloss_seq = batch["gloss_seq"].to(device)
        gloss_len = batch["gloss_len"].to(device)
        has_valid = batch["has_valid_gloss"].to(device)
        mlm_mask = batch.get("mlm_mask", None)
        if mlm_mask is not None:
            mlm_mask = mlm_mask.to(device)

        if (step_idx - 1) % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        def forward_and_losses():
            out = model(
                features,
                mask=mask,
                gloss_seq=gloss_seq,
                mlm_mask=mlm_mask,
                return_aux=True,
                grl_alpha=grl_alpha,
            )

            dec_logits = out["dec_logits"]
            dec_hidden = out["dec_hidden"]
            ctc_log_probs = out["ctc_log_probs"]
            vis_emb = out["vis_emb"]
            sent_emb = out["sent_emb"]
            proj_feats = out["proj_feats"]
            domain_logits = out["domain_logits"]
            enc_mask = out["enc_mask"]
            orig_enc_mask = out.get("orig_enc_mask", enc_mask)

            gt_tokens = gloss_seq[:, 1:].contiguous()
            valid_mask = gt_tokens != GlossVocabulary.PAD_ID
            loss_seq = compute_seq_loss(dec_logits, gt_tokens, valid_mask)

            loss_ctc = _compute_ctc_loss_safe(
                ctc_log_probs, gloss_seq, gloss_len, enc_mask, has_valid
            )

            raw_model = model.module if hasattr(model, "module") else model
            valid_content_mask = gt_tokens >= GlossVocabulary.OFFSET
            gt_lex_embs = raw_model.decoder.asl_lex_emb(gt_tokens)
            loss_dense_sem = raw_model.dense_sem_loss(
                dec_hidden, gt_lex_embs, valid_content_mask
            )

            loss_xmodal = torch.tensor(0.0, device=device, requires_grad=True)
            if sent_emb is not None:
                xmodal_fn = raw_model.xmodal_loss_fn
                loss_xmodal = xmodal_fn(vis_emb, sent_emb)

            loss_supcon = supcon_fn(proj_feats.float(), labels)

            loss_terms = [loss_seq, loss_ctc, loss_dense_sem, loss_xmodal, loss_supcon]

            has_dom_f = has_domain.float()
            if domain_logits is not None and has_dom_f.sum() > 0:
                loss_domain = (
                    F.cross_entropy(
                        domain_logits.float(), domain_tgts, reduction="none"
                    )
                    * has_dom_f
                ).sum() / has_dom_f.sum().clamp(min=1.0)
            else:
                loss_domain = torch.tensor(0.0, device=device, requires_grad=True)
            loss_terms.append(loss_domain)

            if out["mlm_logits"] is not None and mlm_mask is not None:
                loss_mlm = _compute_mlm_loss_safe(
                    out["mlm_logits"], out["orig_x"], mlm_mask
                )
            else:
                loss_mlm = torch.tensor(0.0, device=device, requires_grad=True)

            loss_terms.extend([loss_mlm])
            raw_loss = loss_wrapper(loss_terms)

            with torch.no_grad():
                preds = dec_logits.argmax(dim=-1)
                valid_f = valid_mask.float()
                nc_t = ((preds == gt_tokens).float() * valid_f).sum()
                nt_t = valid_f.sum()

            return raw_loss, dec_logits, nc_t, nt_t

        if use_autocast:
            with torch.autocast(device_type, dtype=prec_dtype):
                raw_loss, dec_logits, nc_t, nt_t = forward_and_losses()
        else:
            raw_loss, dec_logits, nc_t, nt_t = forward_and_losses()

        loss = raw_loss / float(accum_steps)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx % accum_steps == 0) or (step_idx == min_batches):
            if is_xla:
                import torch_xla.utils.utils as xu

                xu.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0,
                )
                xm.optimizer_step(optimizer)
            else:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0,
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            if scheduler is not None:
                scheduler.step()
            raw_m = model.module if hasattr(model, "module") else model
            if ema is not None:
                ema.update(raw_m)
            if hasattr(raw_m, "dense_sem_loss"):
                raw_m.dense_sem_loss.update_momentum()

        if is_xla:
            xm.mark_step()

        def update_stats(tl, ct, tt, s_idx):
            tracker["loss"] += tl.item()
            tracker["corr"] += ct.item()
            tracker["total"] += tt.item()
            if (s_idx % 25 == 0) or (s_idx == min_batches):
                if is_master:
                    c_loss = tracker["loss"] / float(s_idx)
                    c_acc = (tracker["corr"] / max(1.0, tracker["total"])) * 100.0
                    print(
                        f"  [Step {s_idx:04d}/{min_batches:04d}] Loss: {c_loss:.4f} | TF-Acc: {c_acc:.2f}%",
                        flush=True,
                    )

        if is_xla:
            xm.add_step_closure(
                update_stats,
                args=(
                    raw_loss.detach(),
                    nc_t.detach(),
                    nt_t.detach(),
                    step_idx,
                ),
            )
        else:
            update_stats(raw_loss.detach(), nc_t.detach(), nt_t.detach(), step_idx)

    if is_xla:
        xm.mark_step()
        xm.rendezvous("end_of_epoch")

    avg_loss = tracker["loss"] / float(max(1, min_batches))
    token_acc = (tracker["corr"] / max(1.0, tracker["total"])) * 100.0

    return avg_loss, token_acc


print("[+] train_all_in_one_tpu.py module compiled successfully.")
