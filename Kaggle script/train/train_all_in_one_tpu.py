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

# Removed XLA_USE_BF16 to prevent conflicts with native PyTorch autocast
os.cpu_count = lambda: 96

import sys
import time
import glob
import json
import math
import random
import argparse
import re
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

# Force Local PJRT mode to avoid gRPC proxy concurrency limit and fork deadlocks
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)
os.environ["PJRT_DEVICE"] = "TPU"
os.environ["XLA_USE_BF16"] = "1"

try:
    import importlib.util

    _XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
except Exception:
    _XLA_AVAILABLE = False

train_dir = Path(__file__).resolve().parent
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))
try:
    from dataset import create_dataloader
except ImportError:
    pass


def _safe_torch_device(dev_str: Union[str, torch.device]) -> torch.device:
    if isinstance(dev_str, torch.device):
        return dev_str
    dev_s = str(dev_str).lower()
    if _XLA_AVAILABLE and "xla" in dev_s:
        try:
            import torch_xla

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
        self.output_map = {}

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


# ==============================================================================
# 3. RMSNorm & SwiGLUFFN
# ==============================================================================


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var.float() + self.eps).to(x.dtype) * self.weight


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, num_layers: int = 8):
        super().__init__()
        hidden = (int(dim_feedforward * 2 / 3) + 7) // 8 * 8
        self.w_gate, self.w_up, self.w_down = (
            nn.Linear(d_model, hidden, bias=False),
            nn.Linear(d_model, hidden, bias=False),
            nn.Linear(hidden, d_model, bias=False),
        )
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.w_up.weight, std=0.02)
        nn.init.normal_(self.w_down.weight, std=0.02 / math.sqrt(4.0 * num_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.emb_lexclass = nn.Embedding(20, 32)
        self.emb_signtype = nn.Embedding(16, 32)
        self.emb_handshape = nn.Embedding(48, 48)
        self.emb_location = nn.Embedding(24, 32)
        self.emb_category = nn.Embedding(36, 48)

        self.attr_proj = nn.Sequential(
            nn.Linear(32 + 32 + 48 + 32 + 48 + 3, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        attr_idx_matrix = torch.zeros(vocab_size, 5, dtype=torch.long)
        attr_scalars = torch.zeros(vocab_size, 3, dtype=torch.float32)
        self.register_buffer("attr_idx_matrix", attr_idx_matrix)
        self.register_buffer("attr_scalars", attr_scalars)

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
# 5. TOKEN MERGING BLOCK (ToMe)
# ==============================================================================


class TokenMergingBlock(nn.Module):
    def __init__(self, r: int = 80, d_model: int = 320):
        super().__init__()
        self.r, self.d_model = r, d_model
        self.register_buffer("_jl_proj", torch.randn(d_model, 16) / math.sqrt(16))

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        B, T, D = h.shape
        min_half, min_required_tokens = T // 2, 60
        r_clamp = min(self.r, max(0, min_half - (min_required_tokens // 2)))

        if r_clamp <= 0:
            return (
                h,
                mask,
                {
                    "T_orig": T,
                    "sorted_routing": torch.arange(T, device=h.device),
                    "mlm_out": kwargs.get("mlm_mask", None),
                },
            )

        if T > 1:
            h_smooth = F.conv1d(
                F.pad(h.transpose(1, 2), (1, 1), mode="replicate"),
                torch.tensor([0.25, 0.5, 0.25], device=h.device, dtype=h.dtype)
                .view(1, 1, 3)
                .expand(D, 1, 3),
                groups=D,
            ).transpose(1, 2)
        else:
            h_smooth = h

        a, b = h[:, 0::2], h[:, 1::2]
        a_smooth, b_smooth = h_smooth[:, 0::2], h_smooth[:, 1::2]
        min_half = min(a.size(1), b.size(1))
        a, b, a_smooth, b_smooth = (
            a[:, :min_half],
            b[:, :min_half],
            a_smooth[:, :min_half],
            b_smooth[:, :min_half],
        )

        a_proj = torch.matmul(a_smooth, self._jl_proj)
        b_proj = torch.matmul(b_smooth, self._jl_proj)
        ka = F.normalize(a_proj.float(), p=2, dim=-1, eps=1e-5).to(a_proj.dtype)
        kb = F.normalize(b_proj.float(), p=2, dim=-1, eps=1e-5).to(b_proj.dtype)
        sim_matrix = torch.matmul(ka, kb.transpose(-1, -2))
        if mask is not None:
            ma, mb = mask[:, 0::2][:, :min_half], mask[:, 1::2][:, :min_half]
            sim_matrix = sim_matrix.masked_fill(
                (~ma).unsqueeze(-1) & (~mb).unsqueeze(-2), -1e4
            ).masked_fill(ma.unsqueeze(-1) ^ mb.unsqueeze(-2), -1e4)

        mlm_mask = kwargs.get("mlm_mask", None)
        if mlm_mask is not None:
            sim_matrix = sim_matrix.masked_fill(
                mlm_mask[:, 0::2][:, :min_half].unsqueeze(-1)
                ^ mlm_mask[:, 1::2][:, :min_half].unsqueeze(-2),
                -1e4,
            )

        scores, dst_idx = sim_matrix.max(dim=-1)
        r_clamp = min(r_clamp, min_half)
        _, merge_idx = scores.topk(r_clamp, dim=-1, largest=True, sorted=False)

        matched_b_indices_local = dst_idx.gather(1, merge_idx)
        unmerged_scores = torch.zeros(B, min_half, device=h.device).scatter_(
            1, merge_idx, -1e4
        )
        _, kept_idx_a = unmerged_scores.topk(min_half - r_clamp, dim=-1, sorted=True)
        kept_idx_a, _ = kept_idx_a.sort(dim=-1)
        kept_a = a.gather(1, kept_idx_a.unsqueeze(-1).expand(-1, -1, D))

        b_updated = b.clone().scatter_add_(
            1,
            matched_b_indices_local.unsqueeze(-1).expand(-1, -1, D),
            a.gather(1, merge_idx.unsqueeze(-1).expand(-1, -1, D)),
        )
        counts = torch.ones(
            B, min_half, 1, device=h.device, dtype=h.dtype
        ).scatter_add_(
            1,
            matched_b_indices_local.unsqueeze(-1),
            torch.ones(B, r_clamp, 1, device=h.device, dtype=h.dtype),
        )
        b_updated = b_updated / counts

        unmerged_indices_b = (
            torch.cat(
                [
                    torch.arange(min_half, device=h.device).unsqueeze(0).expand(B, -1)
                    * 2
                    + 1,
                    torch.full((B, 1), h.shape[1] - 1, device=h.device),
                ],
                dim=1,
            )
            if h.shape[1] % 2 != 0
            else torch.arange(min_half, device=h.device).unsqueeze(0).expand(B, -1) * 2
            + 1
        )
        all_out_indices = torch.cat([kept_idx_a * 2, unmerged_indices_b], dim=1)
        _, sorted_routing = torch.sort(all_out_indices, dim=1)

        h_unordered = torch.cat(
            (
                [kept_a, b_updated, h[:, -1:]]
                if h.shape[1] % 2 != 0
                else [kept_a, b_updated]
            ),
            dim=1,
        )
        h_out = h_unordered.gather(1, sorted_routing.unsqueeze(-1).expand(-1, -1, D))

        mask_out, mlm_out = None, None
        if mask is not None:
            mask_unordered = torch.cat(
                (
                    [
                        ma.gather(1, kept_idx_a),
                        mb.clone().scatter_(
                            1,
                            matched_b_indices_local,
                            ma.gather(1, merge_idx)
                            | mb.gather(1, matched_b_indices_local),
                        ),
                        mask[:, -1:],
                    ]
                    if h.shape[1] % 2 != 0
                    else [
                        ma.gather(1, kept_idx_a),
                        mb.clone().scatter_(
                            1,
                            matched_b_indices_local,
                            ma.gather(1, merge_idx)
                            | mb.gather(1, matched_b_indices_local),
                        ),
                    ]
                ),
                dim=1,
            )
            mask_out = mask_unordered.gather(1, sorted_routing)

        if mlm_mask is not None:
            mlm_unordered = torch.cat(
                (
                    [
                        mlm_a.gather(1, kept_idx_a),
                        mlm_b.clone().scatter_(
                            1,
                            matched_b_indices_local,
                            mlm_a.gather(1, merge_idx)
                            | mlm_b.gather(1, matched_b_indices_local),
                        ),
                        mlm_mask[:, -1:],
                    ]
                    if mlm_mask.shape[1] % 2 != 0
                    else [
                        mlm_a.gather(1, kept_idx_a),
                        mlm_b.clone().scatter_(
                            1,
                            matched_b_indices_local,
                            mlm_a.gather(1, merge_idx)
                            | mlm_b.gather(1, matched_b_indices_local),
                        ),
                    ]
                ),
                dim=1,
            )
            mlm_out = mlm_unordered.gather(1, sorted_routing)

        return (
            h_out,
            mask_out,
            {"T_orig": T, "sorted_routing": sorted_routing, "mlm_out": mlm_out},
        )


# ==============================================================================
# 6. ENCODER ARCHITECTURE
# ==============================================================================


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    random_tensor = (
        keep_prob
        + torch.rand(
            (x.shape[0],) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device
        ).floor_()
    )
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
        self.nhead, self.kv_heads, self.groups, self.head_dim = (
            nhead,
            kv_heads,
            nhead // kv_heads,
            d_model // nhead,
        )
        self.scale, self.max_relative_position = (
            1.0 / np.sqrt(self.head_dim),
            max_len - 1,
        )
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (
            nn.Linear(d_model, d_model, bias=False),
            nn.Linear(d_model, kv_heads * self.head_dim, bias=False),
            nn.Linear(d_model, kv_heads * self.head_dim, bias=False),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.q_norm, self.k_norm, self.relative_position_bias = (
            RMSNorm(d_model),
            RMSNorm(d_model),
            nn.Embedding(2 * max_len - 1, nhead),
        )
        self.dropout_p = dropout_p

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = (
            self.q_proj(self.q_norm(x))
            .view(B, T, self.kv_heads, self.groups, self.head_dim)
            .permute(0, 2, 3, 1, 4),
            self.k_proj(self.k_norm(x))
            .view(B, T, self.kv_heads, 1, self.head_dim)
            .permute(0, 2, 3, 1, 4),
            self.v_proj(x)
            .view(B, T, self.kv_heads, 1, self.head_dim)
            .permute(0, 2, 3, 1, 4),
        )

        coords = (
            frame_indices.to(device=x.device, dtype=torch.float32)
            if frame_indices is not None
            else torch.arange(T, device=x.device, dtype=torch.float32).unsqueeze(0)
        )
        attn_mask = (
            self.relative_position_bias(
                (
                    torch.clamp(
                        coords.unsqueeze(-1) - coords.unsqueeze(-2),
                        -self.max_relative_position,
                        self.max_relative_position,
                    )
                    + self.max_relative_position
                ).long()
            )
            .view(1, T, T, self.kv_heads, self.groups)
            .permute(0, 3, 4, 1, 2)
            .to(dtype=x.dtype)
            .expand(B, -1, -1, -1, -1)
            .contiguous()
        )
        if key_padding_mask is not None:
            attn_mask = attn_mask.masked_fill(
                key_padding_mask.view(B, 1, 1, 1, T), float("-inf")
            )

        q = q.reshape(B * self.kv_heads, self.groups, T, self.head_dim)
        k = k.repeat_interleave(self.groups, dim=2).reshape(
            B * self.kv_heads, self.groups, T, self.head_dim
        )[:, :, ::2, :]
        v = v.repeat_interleave(self.groups, dim=2).reshape(
            B * self.kv_heads, self.groups, T, self.head_dim
        )[:, :, ::2, :]
        
        attn_mask = attn_mask.reshape(B * self.kv_heads, self.groups, T, T)[:, :, :, ::2]
        
        return self.out_proj(
            F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False,
            )
            .view(B, self.kv_heads, self.groups, T, self.head_dim)
            .permute(0, 3, 1, 2, 4)
            .reshape(B, T, -1)
        )


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
            mean_x = (x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            mean_x = x.mean(dim=1)
        return x * torch.max(self.cSE(mean_x).unsqueeze(1), self.sSE(x))


class ConvNeXtTemporalBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 2):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=0,
            groups=channels,
            padding_mode="reflect",
        )
        self.norm = RMSNorm(channels)
        self.pw_conv1, self.pw_conv2 = nn.Linear(
            channels, channels * expansion
        ), nn.Linear(channels * expansion, channels)
        self.act, self.se = nn.GELU(), SpatialTemporalSE(channels)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        y = self.norm(
            F.conv1d(
                F.pad(
                    x.transpose(1, 2),
                    (3, 3),
                    mode="replicate" if x.size(1) < 4 else "reflect",
                ),
                self.dw_conv.weight,
                self.dw_conv.bias,
                groups=self.dw_conv.groups,
            ).transpose(1, 2)
        )
        if key_padding_mask is not None:
            y = y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        y = self.se(
            self.pw_conv2(self.act(self.pw_conv1(y))), key_padding_mask=key_padding_mask
        )
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
        self.d_model, self.d_inner, self.d_state = d_model, d_model * expand, d_state
        self.nheads, self.headdim = (
            (self.d_inner // headdim)
            if self.d_inner % headdim == 0
            else min(
                [h for h in range(1, self.d_inner + 1) if self.d_inner % h == 0],
                key=lambda h: abs(h - max(1, self.d_inner // headdim)),
            )
        ), self.d_inner // (
            self.d_inner // headdim
            if self.d_inner % headdim == 0
            else min(
                [h for h in range(1, self.d_inner + 1) if self.d_inner % h == 0],
                key=lambda h: abs(h - max(1, self.d_inner // headdim)),
            )
        )

        self.norm1 = RMSNorm(d_model)
        self.in_proj = nn.Linear(
            d_model,
            2 * self.d_inner + 2 * self.nheads * d_state + self.nheads,
            bias=False,
        )
        self.bwd_proj = nn.Linear(
            d_model, self.d_inner + 2 * self.nheads * d_state + self.nheads, bias=False
        )

        self.fwd_conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )
        self.bwd_conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32))
        )
        self.dt_bias = nn.Parameter(
            torch.log(torch.exp(torch.rand(self.nheads) * 0.099 + 0.001) - 1)
        )

        self.head_norm_fwd, self.head_norm_bwd, self.gated_norm = (
            RMSNorm(self.headdim),
            RMSNorm(self.headdim),
            RMSNorm(self.d_inner),
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.gamma_1, self.drop_path1 = nn.Parameter(
            init_values * torch.ones(d_model)
        ), DropPath(drop_path_rate)
        self.norm2, self.ffn, self.gamma_2, self.drop_path2 = (
            RMSNorm(d_model),
            SwiGLUFFN(d_model, ffn_dim),
            nn.Parameter(init_values * torch.ones(d_model)),
            DropPath(drop_path_rate),
        )

        nn.init.orthogonal_(self.in_proj.weight)
        nn.init.orthogonal_(self.bwd_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight)

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
        N_sz, device = B.shape[-1], x.device
        if reverse:
            x, dt, B, C = x.flip(1), dt.flip(1), B.flip(1), C.flip(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.flip(1)

        dt_act = F.softplus(dt).clamp(max=20.0)
        log_decay = -(dt_act * A.view(1, 1, H_sz)).clamp(min=1e-4, max=20.0)
        if key_padding_mask is not None:
            log_decay = log_decay.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        Q = min(chunk_size, T_sz)
        pad_len = (Q - (T_sz % Q)) % Q
        if pad_len > 0:
            x, B, C, log_decay, dt_act = (
                F.pad(x, (0, 0, 0, 0, 0, pad_len)),
                F.pad(B, (0, 0, 0, 0, 0, pad_len)),
                F.pad(C, (0, 0, 0, 0, 0, pad_len)),
                F.pad(log_decay, (0, 0, 0, pad_len), value=-1e4),
                F.pad(dt_act, (0, 0, 0, pad_len), value=0.0),
            )

        T_pad, n_chunks = x.shape[1], x.shape[1] // Q

        x_chunk = x.view(B_sz, n_chunks, Q, H_sz, P_sz).permute(0, 3, 1, 2, 4)
        B_chunk = B.view(B_sz, n_chunks, Q, H_sz, N_sz).permute(0, 3, 1, 2, 4)
        C_chunk = C.view(B_sz, n_chunks, Q, H_sz, N_sz).permute(0, 3, 1, 2, 4)
        ld_chunk = log_decay.view(B_sz, n_chunks, Q, H_sz).permute(0, 3, 1, 2)

        B_chunk_dt = B_chunk * dt_act.view(B_sz, n_chunks, Q, H_sz).permute(
            0, 3, 1, 2
        ).unsqueeze(-1)
        CB = torch.matmul(C_chunk, B_chunk_dt.transpose(-1, -2)) / math.sqrt(N_sz)
        cum_decay = ld_chunk.to(torch.float32).cumsum(dim=-1).to(ld_chunk.dtype)
        M = torch.exp(
            (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).masked_fill(
                ~torch.tril(torch.ones(Q, Q, device=device, dtype=torch.bool)),
                -float("inf"),
            )
        )
        Y_intra = torch.matmul(M * CB, x_chunk)

        log_chunk_decay = ld_chunk.sum(dim=-1)
        decay_to_end = torch.exp(cum_decay[:, :, :, -1:] - cum_decay)
        state_gen = torch.einsum(
            "bhcqp, bhcqn -> bhcpn", x_chunk * decay_to_end.unsqueeze(-1), B_chunk_dt
        )

        L = log_chunk_decay.cumsum(dim=2)
        L_shifted = torch.cat([torch.zeros_like(L[:, :, :1]), L[:, :, :-1]], dim=2)
        M_inter = torch.exp(
            (L_shifted.unsqueeze(-1) - L.unsqueeze(-2)).masked_fill(
                ~torch.tril(
                    torch.ones(n_chunks, n_chunks, device=device, dtype=torch.bool),
                    diagonal=-1,
                ),
                -float("inf"),
            )
        )

        state_stack_flat = torch.einsum(
            "bhij, bhjk -> bhik",
            M_inter,
            state_gen.contiguous().reshape(
                B_sz, H_sz, n_chunks, state_gen.shape[-2] * state_gen.shape[-1]
            ),
        )
        state_stack = state_stack_flat.reshape(
            B_sz, H_sz, n_chunks, state_gen.shape[-2], state_gen.shape[-1]
        )

        C_state = torch.einsum(
            "bhcqn, bhcpn -> bhcqp", C_chunk, state_stack
        ) / math.sqrt(N_sz)
        Y_inter = C_state * torch.exp(cum_decay).unsqueeze(-1)

        Y_flat = (
            (Y_intra + Y_inter).permute(0, 2, 3, 1, 4).reshape(B_sz, T_pad, H_sz, P_sz)
        )
        return Y_flat[:, :T_sz].flip(1) if reverse else Y_flat[:, :T_sz]

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        xn = self.norm1(x)
        B_sz, T_sz, _ = xn.shape

        x_proj, z, B_ssm_fwd, C_ssm_fwd, dt_fwd = torch.split(
            self.in_proj(xn),
            [
                self.d_inner,
                self.d_inner,
                self.nheads * self.d_state,
                self.nheads * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )
        x_proj_bwd, B_ssm_bwd, C_ssm_bwd, dt_bwd = torch.split(
            self.bwd_proj(xn),
            [
                self.d_inner,
                self.nheads * self.d_state,
                self.nheads * self.d_state,
                self.nheads,
            ],
            dim=-1,
        )

        x_fwd_h = F.silu(
            self.fwd_conv1d(x_proj.transpose(1, 2))[:, :, :T_sz].transpose(1, 2)
        ).view(B_sz, T_sz, self.nheads, self.headdim)
        x_bwd_h = F.silu(
            self.bwd_conv1d(x_proj_bwd.transpose(1, 2))[:, :, -T_sz:].transpose(1, 2)
        ).view(B_sz, T_sz, self.nheads, self.headdim)
        B_h_fwd, C_h_fwd = B_ssm_fwd.view(
            B_sz, T_sz, self.nheads, self.d_state
        ), C_ssm_fwd.view(B_sz, T_sz, self.nheads, self.d_state)
        B_h_bwd, C_h_bwd = B_ssm_bwd.view(
            B_sz, T_sz, self.nheads, self.d_state
        ), C_ssm_bwd.view(B_sz, T_sz, self.nheads, self.d_state)

        A = F.softplus(self.A_log)
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
        out = self.out_proj(
            self.gated_norm(y_normed.reshape(B_sz, T_sz, self.d_inner) * F.silu(z))
        )
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        x = x + self.drop_path1(self.gamma_1 * out)
        x2 = self.ffn(self.norm2(x))
        if key_padding_mask is not None:
            x2 = x2.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x + self.drop_path2(self.gamma_2 * x2)


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
        self.nhead, self.kv_heads, self.groups, self.head_dim = (
            nhead,
            kv_heads,
            nhead // kv_heads,
            d_model // nhead,
        )
        self.scale = self.head_dim**-0.5
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(d_model, d_model, bias=False),
            nn.Linear(d_model, kv_heads * self.head_dim, bias=False),
            nn.Linear(d_model, kv_heads * self.head_dim, bias=False),
        )
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(d_model), RMSNorm(d_model)
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
    ):
        B, T, _ = x.shape
        q_in, k_in = self.q_norm(x), self.k_norm(x)
        q = self.q_proj(q_in).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(k_in).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].size(2) if past_key_value is not None else 0
        q, k = self.rope(q, k, offset=past_len)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        current_key_value = (k, v) if use_cache else None

        k_exp = k.repeat_interleave(self.groups, dim=1)
        v_exp = v.repeat_interleave(self.groups, dim=1)

        if T == 1:
            out = F.scaled_dot_product_attention(q, k_exp, v_exp, scale=self.scale)
        else:
            out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, scale=self.scale, is_causal=True
            )

        out = self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        return (out, current_key_value) if use_cache else out


class DecoderCrossAttention(nn.Module):
    def __init__(self, d_model: int = 320, nhead: int = 8, kv_heads: int = 2):
        super().__init__()
        assert nhead % kv_heads == 0
        self.nhead, self.kv_heads, self.groups, self.head_dim = (
            nhead,
            kv_heads,
            nhead // kv_heads,
            d_model // nhead,
        )
        self.q_proj, self.k_proj, self.v_proj = (
            nn.Linear(d_model, d_model, bias=False),
            nn.Linear(d_model, kv_heads * self.head_dim, bias=False),
            nn.Linear(d_model, kv_heads * self.head_dim, bias=False),
        )
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm, self.k_norm = RMSNorm(d_model), RMSNorm(d_model)
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
    ):
        B, T, _ = tgt.shape
        q = (
            self.q_proj(self.q_norm(tgt))
            .view(B, T, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        if past_key_value is not None:
            k, v = past_key_value
        else:
            S = memory.size(1)
            k = (
                self.k_proj(self.k_norm(memory))
                .view(B, S, self.kv_heads, self.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_proj(memory)
                .view(B, S, self.kv_heads, self.head_dim)
                .transpose(1, 2)
            )

        current_key_value = (k, v) if use_cache else None

        k_exp = k.repeat_interleave(self.groups, dim=1)
        v_exp = v.repeat_interleave(self.groups, dim=1)

        attn_mask = (
            memory_key_padding_mask.view(B, 1, 1, k.size(2)).bool()
            if memory_key_padding_mask is not None
            else None
        )
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask)
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        return (out, current_key_value) if use_cache else out


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
        self.drop1, self.drop2, self.drop3 = (
            DropPath(dropout),
            DropPath(dropout),
            DropPath(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
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
        self.d_model, self.vocab_size, self.max_seq_len, self.input_token_dropout = (
            d_model,
            vocab_size,
            max_seq_len,
            0.12,
        )

        self.token_emb = nn.Embedding(
            vocab_size, d_model, padding_idx=GlossVocabulary.PAD_ID
        )
        self.asl_lex_emb = RichASLLexEmbeddingTable(
            vocab_size=vocab_size,
            d_model=d_model,
            csv_path=csv_path,
            label_to_idx=label_to_idx,
        )
        self.emb_drop, self.emb_scale = nn.Dropout(dropout * 0.5), math.sqrt(d_model)

        self.layers = nn.ModuleList(
            [
                ASLDecoderLayer(d_model, nhead, kv_heads, ffn_dim, dropout, max_seq_len)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

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
    ):
        B, S = tgt_ids.shape
        if self.training and self.input_token_dropout > 0:
            drop_mask = (
                torch.rand(tgt_ids.shape, device=tgt_ids.device)
                < self.input_token_dropout
            ) & (tgt_ids >= GlossVocabulary.OFFSET)
            dropped_tgt_ids = torch.where(
                drop_mask,
                torch.tensor(GlossVocabulary.UNK_ID, device=tgt_ids.device),
                tgt_ids,
            )
        else:
            dropped_tgt_ids = tgt_ids

        dropped_tgt_ids = torch.clamp(dropped_tgt_ids, 0, self.vocab_size - 1)
        lex_embs = self.asl_lex_emb(dropped_tgt_ids)
        valid_lex_mask = (
            (tgt_ids != GlossVocabulary.PAD_ID).unsqueeze(-1).to(lex_embs.dtype)
        )
        h = self.emb_drop(
            self.token_emb(dropped_tgt_ids) * self.emb_scale
            + lex_embs * self.emb_scale * valid_lex_mask
        )

        new_key_values = [] if use_cache else None
        for idx, layer in enumerate(self.layers):
            if use_cache:
                h, n_self_kv, n_cross_kv = layer(
                    h,
                    memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    past_self_kv=past_key_values[idx][0] if past_key_values else None,
                    past_cross_kv=past_key_values[idx][1] if past_key_values else None,
                    use_cache=True,
                )
                new_key_values.append((n_self_kv, n_cross_kv))
            else:
                h = layer(h, memory, memory_key_padding_mask=memory_key_padding_mask)

        h = self.final_norm(h)
        logits = self.lm_head(h)
        return (logits, h, new_key_values) if use_cache else (logits, h)


# ==============================================================================
# 8. AUXILIARY HEADS & HOMOSCEDASTIC LOSS WRAPPER WITH NULL-LOSS DETACH
# ==============================================================================


class HomoscedasticLossWrapper(nn.Module):
    """
    Homoscedastic Task Uncertainty Loss Weighting (Kendall & Gal, CVPR 2018).
    Bypasses gradient propagation for zero-valued or uncalculated losses to prevent divergence.
    """

    # ─── MATH FIX: BUMPED TO 8 LOSSES TO PREVENT SILENT TRUNCATION ───
    def __init__(self, num_losses: int = 9): # <--- BUMP TO 9
        super().__init__()
        self.num_losses = num_losses

        # Seq ~8.0, CTC ~10.0, DenseSem ~1.0, InfoNCE ~4.0, SupCon ~2.0, Domain ~1.0, MLM ~0.1, Aux ~2.0, Length ~1.0
        init_vals = [
            math.log(8.0),  
            math.log(10.0), 
            math.log(1.0),  
            math.log(4.0),  
            math.log(2.0),  
            math.log(1.0),  
            math.log(0.1),  
            math.log(2.0),  
            math.log(1.0),  # <--- NEW: Length Regression Loss
        ]
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
                loss_safe = torch.nan_to_num(loss, nan=0.0)
                total_loss = total_loss + 0.5 * prec * loss_safe * valid_mask + s_penalty
        return total_loss


class CosineLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, init_tau: float = 20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # L2 normalize features and weights
        x_norm = F.normalize(x.float(), p=2, dim=-1, eps=1e-5).to(x.dtype)
        w_norm = F.normalize(self.weight.float(), p=2, dim=-1, eps=1e-5).to(self.weight.dtype)
        # Cosine similarity scaled by learnable temperature tau
        return F.linear(x_norm, w_norm) * self.tau


class CTCHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = CosineLinear(d_model, vocab_size)

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

            try:
                world_size = xm.xrt_world_size()
            except AttributeError:
                try:
                    import torch_xla.runtime as xr

                    world_size = xr.world_size()
                except Exception:
                    world_size = 4
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        v = F.normalize(vis_emb.float(), p=2, dim=-1)
        s = F.normalize(sent_emb.float(), p=2, dim=-1)

        lbl = torch.arange(v.size(0), device=vis_emb.device)

        sim1 = torch.matmul(v, s.T) / temp
        sim2 = torch.matmul(s, v.T) / temp

        s_sim = torch.matmul(s, s.T)
        false_neg_mask = (s_sim > 0.99) & (
            torch.arange(s.size(0), device=s.device).unsqueeze(0) != lbl.unsqueeze(1)
        )
        sim1.masked_fill_(false_neg_mask, -1e4)
        sim2.masked_fill_(false_neg_mask, -1e4)
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

        pred_sent = (
            (last_hidden * m).sum(dim=1) / valid_counts
            if last_hidden.dim() == 3
            else last_hidden
        )
        gt_sent = (
            (gt_lex_embs * m).sum(dim=1) / valid_counts
            if gt_lex_embs.dim() == 3
            else gt_lex_embs
        )

        p = F.normalize(self.proj_pred(pred_sent).float(), p=2, dim=-1, eps=1e-8)
        g = F.normalize(self.proj_gt(gt_sent).float(), p=2, dim=-1, eps=1e-8).detach()

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
        features = F.normalize(features.float(), p=2, dim=1, eps=1e-5).to(features.dtype)
        B, device = features.shape[0], features.device
        has_labels = (labels.abs().sum() > 0).float()

        ids = torch.arange(B, device=device)
        pos_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        self_m = torch.eq(ids.view(-1, 1), ids.view(1, -1)).float()
        pos_mask *= 1.0 - self_m

        pos_logits = torch.matmul(features, features.T) / self.temperature
        exp_logits = torch.exp(pos_logits - pos_logits.max(dim=1, keepdim=True)[0]) * (
            1.0 - self_m
        )
        denom = torch.clamp(exp_logits.sum(dim=1, keepdim=True).float(), min=1e-4)

        log_prob = (pos_logits - pos_logits.max(dim=1, keepdim=True)[0]) - torch.log(
            denom
        )
        pos_count = pos_mask.sum(dim=1)
        valid_rows = (pos_count > 0).float()

        row_loss = -(log_prob * pos_mask).sum(dim=1) / pos_count.clamp(min=1.0)
        loss = (row_loss * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0) * has_labels


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
        return x + self.pe[:, : x.size(1), :]


class ASLFoundationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2484,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        d_enc: int = 512,
        nhead_enc: int = 16,
        num_enc_layers: int = 12,
        ffn_enc: int = 2048,
        d_dec: int = 512,
        nhead_dec: int = 16,
        kv_heads_dec: int = 4,
        num_dec_layers: int = 12,
        ffn_dec: int = 2048,
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

        # ─── MATH FIX: Encoder Auxiliary Classification Head ───
        # Mathematically forces the Conformer to anchor the latent space into a discrete conceptual cluster
        # BEFORE giving the sequence to the decoder. Bypasses decoder hallucination drift.
        self.aux_gloss_head = CosineLinear(d_enc, vocab_size, init_tau=20.0)
        
        # ─── NEW: Sequence Length Prediction Head (Fertility) ───
        self.length_head = nn.Sequential(
            nn.Linear(d_enc, 128),
            RMSNorm(128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        # ────────────────────────────────────────────────────────

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

    def update_tome_r(self, epoch: int, max_epochs: int):
        progress = epoch / max(1, max_epochs - 1)
        new_r = int(10 + (80 - 10) * progress)
        self.tome_r = new_r
        if hasattr(self, 'tome_early'):
            self.tome_early.r = new_r
        if hasattr(self, 'tome_deep'):
            self.tome_deep.r = new_r

    def _encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mlm_mask: Optional[torch.Tensor] = None,
    ):
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
            lh_nz = (xk[:, :, 0:21, :3] != 0).to(xk.dtype)
            rh_nz = (xk[:, :, 21:42, :3] != 0).to(xk.dtype)
            xk[:, :, 0:21, :3] = (xk[:, :, 0:21, :3] - xk[:, :, 0:1, :3]) * lh_nz
            xk[:, :, 21:42, :3] = (xk[:, :, 21:42, :3] - xk[:, :, 21:22, :3]) * rh_nz
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

        h_pre_tome = None

        for idx, block in enumerate(self.blocks):
            if isinstance(block, TokenMergingBlock):
                if h_pre_tome is None:
                    h_pre_tome = h[:, 1:]
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
        if h_pre_tome is None:
            h_pre_tome = h[:, 1:]

        return h[:, 0], h[:, 1:], cur_mask, used_mlm_mask, h_pre_tome, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gloss_seq: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        grl_alpha: float = 1.0,
    ) -> Union[Optional[torch.Tensor], Dict]:

        h_cls, h_seq, enc_mask, used_mlm_mask, h_pre_tome, orig_enc_mask = self._encode(
            x, mask, mlm_mask=mlm_mask
        )

        dec_logits, dec_hidden = None, None
        if gloss_seq is not None:
            if self.training and gloss_seq.size(1) > 1:
                dec_in = gloss_seq[:, :-1].contiguous()
            else:
                dec_in = gloss_seq.contiguous()

            if self.training:
                mask_tgt = (torch.rand_like(dec_in, dtype=torch.float) < 0.15) & (
                    dec_in != GlossVocabulary.BOS_ID
                )
                dec_in = dec_in.masked_fill(mask_tgt, GlossVocabulary.UNK_ID)

            dec_logits, dec_hidden = self.decoder(
                dec_in, h_seq, memory_key_padding_mask=enc_mask
            )

        if not return_aux:
            return dec_logits

        # ─── MATH FIX: Auxiliary Average Pooling Head Computation ───
        if enc_mask is not None:
            mask_expanded = enc_mask.unsqueeze(-1).to(
                h_seq.dtype
            )  # enc_mask evaluates to True for Valid
            valid_lengths = mask_expanded.sum(dim=1).clamp(min=1.0)
            pooled_enc = (h_seq * mask_expanded).sum(dim=1) / valid_lengths
        else:
            pooled_enc = h_seq.mean(dim=1)
            
        aux_logits = self.aux_gloss_head(pooled_enc)
        
        # ─── NEW: Predict Sequence Length ───
        pred_len = self.length_head(pooled_enc).squeeze(-1) # Shape: (B,)
        # ────────────────────────────────────

        ctc_log_probs = F.log_softmax(self.ctc_head(h_seq), dim=-1)

        if self.training and h_pre_tome is not None and mlm_mask is not None:
            mlm_logits = self.mlm_head(h_pre_tome)
        else:
            mlm_logits = None

        vis_emb = F.normalize(self.visual_proj(h_cls).float(), p=2, dim=-1, eps=1e-5).to(h_cls.dtype)
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
            sent_emb = F.normalize(self.sentence_proj(last_hidden).float(), p=2, dim=-1, eps=1e-5).to(last_hidden.dtype)
        else:
            sent_emb = None

        proj_feats = F.normalize(self.contrastive_head(h_cls).float(), p=2, dim=-1, eps=1e-5).to(h_cls.dtype)

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
            "aux_logits": aux_logits,  # Passed out for grounding loss
            "pred_len": pred_len,
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
    valid_mask = (raw_targets != GlossVocabulary.PAD_ID) & has_valid.unsqueeze(1)
    targets = raw_targets
    tgt_lengths = valid_mask.sum(dim=-1).long()

    valid_ctc = (enc_len >= tgt_lengths) & (tgt_lengths > 0) & (enc_len > 0)

    try:
        loss_vec = F.ctc_loss(
            ctc_log_probs.float().transpose(0, 1),
            targets,
            enc_len.clamp(min=1, max=T_enc),
            tgt_lengths,
            blank=GlossVocabulary.PAD_ID,
            reduction="none",
            zero_infinity=True,
        )
        loss_vec = torch.nan_to_num(loss_vec)
        valid_f = valid_ctc.float()
        loss_ctc = (loss_vec * valid_f).sum() / valid_f.sum().clamp(min=1.0)
        return loss_ctc.clamp(max=15.0)
    except Exception:
        return torch.tensor(0.0, device=device)


def _compute_mlm_loss_safe(
    mlm_logits: torch.Tensor, orig_x: torch.Tensor, mlm_mask: torch.Tensor
) -> torch.Tensor:
    B, T = orig_x.size(0), orig_x.size(1)
    target = orig_x.clone()
    if target.dim() == 3 and target.size(-1) % 60 == 0:
        target = target.view(B, T, 60, -1)

    if target.dim() == 4 and target.size(2) == 60 and target.size(3) >= 3:
        lh_nz, rh_nz = (target[:, :, 0:21, :3] != 0).to(target.dtype), (
            target[:, :, 21:42, :3] != 0
        ).to(target.dtype)
        target[:, :, 0:21, :3] = (
            target[:, :, 0:21, :3] - target[:, :, 0:1, :3]
        ) * lh_nz
        target[:, :, 21:42, :3] = (
            target[:, :, 21:42, :3] - target[:, :, 21:22, :3]
        ) * rh_nz

    target = target.reshape(B, T, -1)
    mask_f = mlm_mask.unsqueeze(-1).float()
    loss = F.smooth_l1_loss(mlm_logits, target, reduction="none")
    return torch.nan_to_num(
        (loss * mask_f).sum() / (mask_f.sum().clamp(min=1.0) * target.size(-1)), nan=0.0
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay_base: float = 0.90, decay_max: float = 0.9999, device: str = "cpu"):
        self.decay_base = decay_base
        self.decay_max = decay_max
        self.device = device
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach().to(self.device)

    def update(self, model: nn.Module, progress: float = 1.0):
        with torch.no_grad():
            progress = min(1.0, max(0.0, progress))
            current_decay = self.decay_max - (self.decay_max - self.decay_base) * (1.0 + math.cos(math.pi * progress)) / 2.0
            
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Push updates to CPU asynchronously to save TPU memory
                    self.shadow[name].mul_(current_decay).add_(param.data.to(self.device), alpha=1.0 - current_decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone().detach().to(self.device)
                param.data.copy_(self.shadow[name].to(param.device))

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup.clear()

def _get_optimizer_groups(model: nn.Module, loss_wrapper: nn.Module, weight_decay: float):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    for param in loss_wrapper.parameters():
        if param.requires_grad:
            no_decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

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
    prec_dtype: torch.dtype = torch.float16,
    is_master: bool = True,
    accum_steps: int = 4,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    model.train()

    # ─── Dynamic Token Merging Scaling ───
    if hasattr(model, 'module'):
        if hasattr(model.module, 'update_tome_r'):
            model.module.update_tome_r(epoch, total_epochs)
    elif hasattr(model, 'update_tome_r'):
        model.update_tome_r(epoch, total_epochs)

    if loss_wrapper is not None and len(list(loss_wrapper.parameters())) > 0:
        found = any(
            any(p is param for param in group["params"])
            for group in optimizer.param_groups
            for p in loss_wrapper.parameters()
        )
        if not found:
            optimizer.add_param_group({"params": loss_wrapper.parameters()})

    tracker = {
        "loss": 0.0,
        "corr": 0.0,
        "total": 0.0,
        "seq": 0.0,
        "ctc": 0.0,
        "sem": 0.0,
        "supcon": 0.0,
        "dom": 0.0,
        "mlm": 0.0,
        "aux": 0.0,
    }
    epoch_start_time = time.time()

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    if is_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    is_master = xm.is_master_ordinal() if is_xla else True

    if is_xla:
        device_type, use_autocast = "xla", (prec_dtype in (torch.float16, torch.bfloat16))
    else:
        device_type, use_autocast = (
            "cuda" if "cuda" in str(device).lower() else "cpu"
        ), ("cuda" in str(device).lower() and prec_dtype != torch.float32)

    scaler = (
        torch.amp.GradScaler("cuda")
        if use_autocast and prec_dtype == torch.float16
        else None
    )

    progress = float(max(0, epoch)) / float(max(1, total_epochs - 1))
    grl_alpha = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
    label_smoothing = max(0.05, 0.15 - 0.10 * progress)
    POLY1_EPS = 1.0

    def compute_seq_loss(logits_f, gt_ids, valid_mask, class_weights=None, gamma=2.0):
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = torch.clamp(gt_ids.reshape(-1), 0, V - 1)
        vf = valid_mask.reshape(-1).float()

        if class_weights is not None:
            vf = vf * class_weights[tf]

        # ─── MATH FIX: THE HALLUCINATION KILLER (EOS GRADIENT FORCING) ───
        # Ensures the gradient for the stopping condition is never swallowed by Focal Loss
        eos_mask = (tf == GlossVocabulary.EOS_ID).to(vf.dtype)
        vf = vf * (1.0 + eos_mask * 9.0)  # 10.0x Multiplier for the EOS Token
        # ─────────────────────────────────────────────────────────────────

        log_p = F.log_softmax(lf, dim=-1)
        p_target = torch.exp(log_p.gather(1, tf.unsqueeze(1)).squeeze(1)).clamp(
            min=1e-6, max=1.0
        )
        focal_weight = torch.pow(1.0 - p_target, gamma)

        ce_unsmoothed = F.nll_loss(
            log_p, tf, ignore_index=GlossVocabulary.PAD_ID, reduction="none"
        )
        ce_uniform = -log_p[..., 1:].mean(dim=-1)
        ce_smoothed = (
            1.0 - label_smoothing
        ) * ce_unsmoothed + label_smoothing * ce_uniform

        poly1 = focal_weight * ce_smoothed + POLY1_EPS * (1.0 - p_target)
        return (poly1 * vf).sum() / vf.sum().clamp(min=1.0)

    if is_xla:
        loader = pl.MpDeviceLoader(loader, device)

    total_batches = len(loader)
    min_batches = total_batches
    if is_xla:
        min_batches = int(
            xm.mesh_reduce("min_batches", total_batches, lambda x: min(x))
        )
        try:
            ord_val = xm.get_ordinal()
        except AttributeError:
            try:
                import torch_xla.runtime as xr

                ord_val = xr.global_ordinal()
            except Exception:
                ord_val = 0
        xm.set_rng_state(42 + epoch * 10000 + ord_val)

    for step_idx, batch in enumerate(loader, start=1):
        if step_idx > min_batches:
            continue

        if isinstance(batch, (tuple, list)):
            features, mask, labels = (
                batch[0].to(device),
                batch[1].to(device),
                batch[2].to(device),
            )
            B, domain_tgts, has_domain = (
                features.size(0),
                torch.zeros_like(labels, device=device),
                torch.zeros_like(labels, dtype=torch.bool, device=device),
            )
            gloss_seq = torch.zeros((B, 3), dtype=torch.long, device=device)
            gloss_seq[:, 0], gloss_seq[:, 1], gloss_seq[:, 2] = (
                GlossVocabulary.BOS_ID,
                labels + GlossVocabulary.OFFSET,
                GlossVocabulary.EOS_ID,
            )
            gloss_len, has_valid, mlm_mask = (
                torch.full((B,), 3, dtype=torch.long, device=device),
                torch.ones(B, dtype=torch.bool, device=device),
                None,
            )
        else:
            features, mask, labels = (
                batch["feature"].to(device),
                batch["mask"].to(device),
                batch.get(
                    "label", torch.zeros(batch["feature"].size(0), dtype=torch.long)
                ).to(device),
            )
            domain_tgts, has_domain = batch.get(
                "domain_label", torch.zeros_like(labels)
            ).to(device), batch.get("has_domain_label", torch.zeros_like(labels)).to(
                device
            )
            gloss_seq, gloss_len, has_valid, mlm_mask = (
                batch["gloss_seq"].to(device),
                batch["gloss_len"].to(device),
                batch["has_valid_gloss"].to(device),
                batch.get("mlm_mask", None),
            )
            if mlm_mask is not None:
                mlm_mask = mlm_mask.to(device)

        # curriculum removed to prevent CTC NaN

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
            (
                dec_logits,
                dec_hidden,
                ctc_log_probs,
                vis_emb,
                sent_emb,
                proj_feats,
                domain_logits,
                aux_logits,
                enc_mask,
                pred_len,
            ) = (
                out["dec_logits"],
                out["dec_hidden"],
                out["ctc_log_probs"],
                out["vis_emb"],
                out["sent_emb"],
                out["proj_feats"],
                out["domain_logits"],
                out["aux_logits"],
                out["enc_mask"],
                out["pred_len"],
            )

            gt_tokens = gloss_seq[:, 1:].contiguous()
            valid_mask = gt_tokens != GlossVocabulary.PAD_ID
            
            # ─── NEW: SEQUENCE LENGTH LOSS (Huber Loss) ───
            # True length is the sum of valid tokens (excluding padding)
            target_len = valid_mask.sum(dim=1).float()
            loss_length = F.smooth_l1_loss(pred_len, target_len, reduction="mean")
            # ──────────────────────────────────────────────

            loss_seq = compute_seq_loss(
                dec_logits, gt_tokens, valid_mask, class_weights=class_weights
            )

            # ─── MATH FIX: AUXILIARY GROUNDING LOSS ───
            raw_model = model.module if hasattr(model, "module") else model
            aux_target = torch.clamp(
                labels + GlossVocabulary.OFFSET, 0, raw_model.vocab_size - 1
            )
            loss_aux = F.cross_entropy(
                aux_logits.float(), aux_target, reduction="mean", label_smoothing=0.1
            )

            loss_ctc = _compute_ctc_loss_safe(ctc_log_probs, gloss_seq, gloss_len, enc_mask, has_valid)
            loss_dense_sem = raw_model.dense_sem_loss(
                dec_hidden,
                raw_model.decoder.asl_lex_emb(gt_tokens),
                gt_tokens >= GlossVocabulary.OFFSET,
            )
            loss_xmodal = (
                raw_model.xmodal_loss_fn(vis_emb, sent_emb)
                if sent_emb is not None
                else torch.tensor(0.0, device=device, requires_grad=True)
            )
            loss_supcon = supcon_fn(proj_feats.float(), labels)

            loss_terms = [loss_seq, loss_ctc, loss_dense_sem, loss_xmodal, loss_supcon]

            has_dom_f = has_domain.float()
            loss_domain = (
                (
                    F.cross_entropy(
                        domain_logits.float(), domain_tgts, reduction="none"
                    )
                    * has_dom_f
                ).sum()
                / has_dom_f.sum().clamp(min=1.0)
                if domain_logits is not None
                else torch.tensor(0.0, device=device)
            )
            loss_terms.append(loss_domain)

            loss_mlm = (
                _compute_mlm_loss_safe(out["mlm_logits"], out["orig_x"], mlm_mask)
                if out["mlm_logits"] is not None and mlm_mask is not None
                else torch.tensor(0.0, device=device, requires_grad=True)
            )
            loss_terms.extend([loss_mlm])

            # Append the 8th and 9th losses for Homoscedastic balancing
            loss_terms.append(loss_aux)
            loss_terms.append(loss_length) # <--- APPEND LENGTH LOSS
            raw_loss = loss_wrapper(loss_terms)

            with torch.no_grad():
                preds = dec_logits.argmax(dim=-1)
                valid_f = valid_mask.float()
                nc_t = ((preds == gt_tokens).float() * valid_f).sum()
                nt_t = valid_f.sum()

            return (
                raw_loss,
                dec_logits,
                nc_t,
                nt_t,
                loss_seq.detach(),
                loss_ctc.detach(),
                loss_dense_sem.detach(),
                loss_supcon.detach(),
                loss_domain.detach(),
                loss_mlm.detach(),
                loss_aux.detach(),
                loss_length.detach(),
            )

        if use_autocast:
            with torch.autocast(device_type, dtype=prec_dtype):
                (
                    raw_loss,
                    dec_logits,
                    nc_t,
                    nt_t,
                    l_seq,
                    l_ctc,
                    l_sem,
                    l_sup,
                    l_dom,
                    l_mlm,
                    l_aux,
                    l_len,
                ) = forward_and_losses()
        else:
            (
                raw_loss,
                dec_logits,
                nc_t,
                nt_t,
                l_seq,
                l_ctc,
                l_sem,
                l_sup,
                l_dom,
                l_mlm,
                l_aux,
                l_len,
            ) = forward_and_losses()

        if torch.isnan(raw_loss) or torch.isinf(raw_loss):
            if is_master:
                print(
                    f"[!] Warning: NaN/Inf loss encountered at Epoch {epoch} step {step_idx}. Skipping batch backward.",
                    flush=True,
                )
            continue

        loss = raw_loss / float(accum_steps)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx % accum_steps == 0) or (step_idx == min_batches):
            if is_xla:
                # ─── SAM (Sharpness-Aware Minimization) for final 25% of epochs ───
                use_sam = epoch >= total_epochs * 0.75
                if use_sam:
                    xm.reduce_gradients(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()) + list(loss_wrapper.parameters()), max_norm=1.0
                    )
                    rho = 0.05
                    scale = rho / (grad_norm + 1e-6)
                    
                    e_w = {}
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if p.grad is not None:
                                e_w[n] = p.grad * scale
                                p.data.add_(e_w[n])
                                
                    optimizer.zero_grad(set_to_none=True)
                    if use_autocast:
                        with torch.autocast(device_type, dtype=prec_dtype):
                            raw_loss2 = forward_and_losses()[0]
                    else:
                        raw_loss2 = forward_and_losses()[0]
                        
                    loss2 = raw_loss2 / float(accum_steps)
                    if scaler is not None:
                        scaler.scale(loss2).backward()
                    else:
                        loss2.backward()
                        
                    with torch.no_grad():
                        for n, p in model.named_parameters():
                            if p.grad is not None and n in e_w:
                                p.data.sub_(e_w[n])

                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0,
                )
                xm.optimizer_step(optimizer)
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                optimizer.zero_grad(set_to_none=True)

                if is_master and ((step_idx % 25 == 0) or (step_idx == min_batches)):

                    def _async_step_print(
                        l_val,
                        s_val,
                        aux_val,
                        c_val,
                        sm_val,
                        nc_val,
                        nt_val,
                        st_idx,
                        m_batches,
                        ep,
                        tot_ep,
                        lr_val,
                        t_start,
                    ):
                        c_acc = (float(nc_val) / max(1.0, float(nt_val))) * 100.0
                        speed = float(st_idx) / max(0.001, time.time() - t_start)
                        msg = f"  [Epoch {ep:03d}/{tot_ep:03d} | Step {st_idx:04d}/{m_batches:04d}] Loss: {float(l_val):.4f} (Seq:{float(s_val):.4f} Aux:{float(aux_val):.4f} Sem:{float(sm_val):.4f}) | TF-Acc: {c_acc:.2f}% | LR: {lr_val:.2e} | {speed:.1f} it/s"
                        print(msg, flush=True)
                        try:
                            with open(
                                "/tmp/step_losses.txt", "a", encoding="utf-8"
                            ) as f_log:
                                f_log.write(msg + "\n")
                        except Exception:
                            pass

                    xm.add_step_closure(
                        _async_step_print,
                        args=(
                            raw_loss.detach(),
                            l_seq.detach(),
                            l_aux.detach(),
                            l_ctc.detach(),
                            l_sem.detach(),
                            nc_t.detach(),
                            nt_t.detach(),
                            step_idx,
                            min_batches,
                            epoch,
                            total_epochs,
                            optimizer.param_groups[0]["lr"],
                            epoch_start_time,
                        ),
                    )
            else:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                
                # Automatic Gradient Clipping (AGC) - No host sync
                for p in list(model.parameters()) + list(loss_wrapper.parameters()):
                    if p.grad is not None:
                        g = p.grad.detach()
                        eps = 1e-3
                        p_norm = p.detach().float().norm(2).clamp_(min=eps)
                        g_norm = g.float().norm(2).clamp_(min=eps)
                        max_norm = p_norm * 0.01  # clip_val = 0.01
                        clipped_g = g * (max_norm / g_norm).clamp_(max=1.0)
                        g.copy_(clipped_g)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception:
                        pass
                optimizer.zero_grad(set_to_none=True)

                if is_xla:
                    xm.mark_step()

                loss_cpu, seq_cpu, sem_cpu, nc_cpu, nt_cpu, aux_cpu = [
                    float(v)
                    for v in [
                        raw_loss.detach(),
                        l_seq.detach(),
                        l_sem.detach(),
                        nc_t.detach(),
                        nt_t.detach(),
                        l_aux.detach(),
                    ]
                ]
                tracker["loss"] += loss_cpu
                tracker["seq"] += seq_cpu
                tracker["sem"] += sem_cpu
                tracker["corr"] += nc_cpu
                tracker["total"] += nt_cpu
                tracker["aux"] += aux_cpu

            raw_m = model.module if hasattr(model, "module") else model
            if ema is not None:
                progress = (epoch * min_batches + step_idx) / max(1, total_epochs * min_batches)
                ema.update(raw_m, progress)
            if hasattr(raw_m, "dense_sem_loss"):
                raw_m.dense_sem_loss.update_momentum()

            if is_xla and step_idx >= min_batches:
                break

    if is_xla:
        xm.mark_step()
        xm.rendezvous("end_of_epoch")

    avg_loss = tracker["loss"] / float(max(1, min_batches))
    token_acc = (tracker["corr"] / max(1.0, tracker["total"])) * 100.0
    return avg_loss, token_acc


def _tpu_worker_fn(rank, args):
    if _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    try:
        device = (
            xm.xla_device()
            if _XLA_AVAILABLE
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    except Exception as e:
        print(f"FAILED TO INITIALIZE TPU OR GET DEVICE: {e}", flush=True)
        time.sleep(2)
        os._exit(1)
    is_master = (rank == 0) if _XLA_AVAILABLE else True

    data_dir = Path(args.data_dir)
    label_to_idx = {}
    for _mf in [
        data_dir / "output_mapping.json",
        data_dir / "vocabulary_mapping_global.json",
        data_dir / "vocabulary_mapping_train.json",
    ]:
        if _mf.exists():
            try:
                with open(_mf, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                if isinstance(raw_map, dict) and "label_to_idx" in raw_map:
                    label_to_idx = raw_map["label_to_idx"]
                elif isinstance(raw_map, dict):
                    non_meta_keys = {
                        k
                        for k in raw_map
                        if k
                        not in (
                            "task",
                            "total_classes",
                            "description",
                            "name",
                            "version",
                            "split",
                        )
                    }
                    if non_meta_keys:
                        label_to_idx = {
                            k: v for k, v in raw_map.items() if k in non_meta_keys
                        }
                if label_to_idx:
                    break
            except Exception as _e:
                if is_master:
                    print(
                        f"[!] Warning: Failed to load mapping from '{_mf}': {_e}",
                        flush=True,
                    )

    if not label_to_idx:
        if is_master:
            print(
                "[!] Warning: No vocabulary mapping found. Using default mapping.",
                flush=True,
            )
        label_to_idx = {str(i): i for i in range(6152)}

    vocab = GlossVocabulary(label_to_idx=label_to_idx)

    train_loader = create_dataloader(
        dataset_dir=data_dir,
        split="train",
        batch_size=args.batch_size,
        max_len=args.max_len,
        worker_idx=rank if _XLA_AVAILABLE else 0,
        num_workers=args.num_cores if _XLA_AVAILABLE else 1,
        num_dataloader_workers=args.num_dataloader_workers,
        shuffle=True,
    )

    class_weights_tensor = None
    try:
        raw_ds = getattr(train_loader, "dataset", None)
        c_counts = getattr(raw_ds, "class_counts", None) if raw_ds else None
        if c_counts:
            w_vec = torch.ones(len(vocab), dtype=torch.float32, device=device)
            max_c = max(c_counts.values()) if c_counts else 1
            for r_idx, cnt in c_counts.items():
                tok_id = r_idx + GlossVocabulary.OFFSET
                if tok_id < len(vocab):
                    w_vec[tok_id] = min(
                        10.0, max(1.0, (float(max_c) / float(max(1, cnt))) ** 0.35)
                    )
            class_weights_tensor = w_vec
    except Exception:
        pass

    asl_lex_csv = (
        Path(args.asl_lex_csv)
        if hasattr(args, "asl_lex_csv") and args.asl_lex_csv
        else (data_dir / "signdata.csv")
    )

    model = ASLFoundationModel(
        vocab_size=len(vocab),
        d_enc=args.d_model,
        d_dec=args.d_model,
        nhead_enc=args.nhead,
        nhead_dec=args.nhead,
        num_enc_layers=args.num_layers,
        num_dec_layers=args.num_layers,
        dropout=args.dropout,
        label_to_idx=label_to_idx,
        csv_path=asl_lex_csv if asl_lex_csv.exists() else None,
    ).to(device)

    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        if is_master:
            print("[*] JIT Compiling model with PyTorch Inductor (torch.compile)...", flush=True)
        try:
            model = torch.compile(model)
        except Exception as _e:
            if is_master:
                print(f"[!] Warning: torch.compile fallback: {_e}", flush=True)

    loss_wrapper = HomoscedasticLossWrapper(num_losses=9).to(device)
    supcon_fn = SupervisedContrastiveLoss().to(device)

    optimizer = torch.optim.AdamW(
        _get_optimizer_groups(model, loss_wrapper, args.weight_decay),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(
            1, len(train_loader) * (args.num_cores if _XLA_AVAILABLE else 1)
        ),
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    start_epoch = 1
    if hasattr(args, "resume") and args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "loss_wrapper_state_dict" in ckpt:
            try:
                loss_wrapper.load_state_dict(
                    ckpt["loss_wrapper_state_dict"], strict=False
                )
            except Exception:
                pass
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    save_dir = Path(args.save_dir)
    if is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 70, flush=True)
        print(
            f"       STARTING TPU MULTI-TASK FOUNDATION MODEL TRAINING ({args.epochs} EPOCHS)",
            flush=True,
        )
        print("=" * 70, flush=True)

    ema = ModelEMA(model) if is_master else None

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            
            # --- ADD THIS TO RAMP UP NOISE CURRICULUM ---
            if hasattr(train_loader.dataset, "set_noise_level"):
                # Ramps noise from 0.0 to 1.0 linearly over the epochs
                train_loader.dataset.set_noise_level(epoch / max(1, args.epochs))
            # --------------------------------------------
            
            train_loss, train_acc = train_epoch_tpu(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_wrapper=loss_wrapper,
                ema=ema,
                supcon_fn=supcon_fn,
                device=device,
                epoch=epoch,
                total_epochs=args.epochs,
                prec_dtype=(
                    torch.float16
                    if args.precision == "float16"
                    else (
                        torch.bfloat16
                        if args.precision == "bfloat16"
                        else torch.float32
                    )
                ),
                is_master=is_master,
                accum_steps=args.accum_steps,
                class_weights=class_weights_tensor,
            )
            ckpt_path = save_dir / f"asl_model_epoch_{epoch}.pt"
            raw_m = model.module if hasattr(model, "module") else model
            if ema is not None:
                ema.apply_shadow(raw_m)
                
            cpu_state = {
                "epoch": epoch,
                "model_state_dict": raw_m.state_dict(),
                "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            
            if ema is not None:
                ema.restore(raw_m)
            if _XLA_AVAILABLE:
                import torch_xla.core.xla_model as xm

                xm.save(cpu_state, str(ckpt_path), master_only=True)
            else:
                if is_master:
                    torch.save(cpu_state, str(ckpt_path))
            if is_master:
                print(f"[+] Saved checkpoint to {ckpt_path}", flush=True)
                try:
                    all_ckpts = sorted(
                        list(save_dir.glob("asl_model_epoch_*.pt")),
                        key=lambda p: int(p.stem.split("_")[-1]),
                    )
                    if len(all_ckpts) > 5:
                        for old_c in all_ckpts[:-5]:
                            ep_num = int(old_c.stem.split("_")[-1])
                            if ep_num % 10 != 0 and ep_num != epoch:
                                old_c.unlink(missing_ok=True)
                except Exception:
                    pass
            if scheduler is not None:
                try:
                    scheduler.step()
                except Exception:
                    pass
    except Exception as e:
        import traceback, sys

        print(f"CRITICAL PYTHON EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        time.sleep(2)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="ASL Foundation Model Multi-Task TPU Training Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/tmp/asl_dataset/results/asl_preprocessed_phase1",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-cores", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-dataloader-workers", type=int, default=8)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Enable PyTorch 2.0 torch.compile JIT acceleration")
    parser.add_argument("--save-dir", type=str, default="/tmp/checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--asl-lex-csv", type=str, default="/home/binhhanh409/signdata.csv"
    )
    args = parser.parse_args()

    if _XLA_AVAILABLE:
        import torch_xla.distributed.xla_multiprocessing as xmp

        if "LOCAL_RANK" in os.environ:
            _tpu_worker_fn(int(os.environ["LOCAL_RANK"]), args)
        else:
            xmp.spawn(_tpu_worker_fn, args=(args,), nprocs=None, start_method="fork")
    else:
        _tpu_worker_fn(0, args)


if __name__ == "__main__":
    main()
