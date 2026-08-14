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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# Removed XLA_USE_BF16 to prevent conflicts with native PyTorch autocast

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


# Force Local PJRT mode to avoid gRPC proxy concurrency limit and fork deadlocks
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)
# WARNING: Setting PJRT_DEVICE=TPU breaks training on Kaggle T4 GPUs.
# Uncomment the line below ONLY if running on a Kaggle TPU VM.
# os.environ["PJRT_DEVICE"] = "TPU"

try:
    import importlib.util

    _XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
except Exception:
    _XLA_AVAILABLE = False

def is_tpu_runtime() -> bool:
    return (
        _XLA_AVAILABLE
        and os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
    )

IS_TPU = is_tpu_runtime()

train_dir = Path(__file__).resolve().parent
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))
try:
    from dataset import create_dataloader
except ImportError:
    pass


def _distributed_normalize(
    local_sum: torch.Tensor, local_weight: torch.Tensor
) -> torch.Tensor:
    if _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm

        if getattr(xm, "xrt_world_size", lambda: 1)() > 1:
            global_weight = xm.all_reduce(xm.REDUCE_SUM, local_weight.clone())
            normed = (local_sum * xm.xrt_world_size()) / global_weight
            return torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)

    normed = local_sum / local_weight
    return torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)


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
                k_str = str(k).strip().lower()
                if isinstance(v, int):
                    clean_l2i[k_str] = v
                elif isinstance(v, dict):
                    idx_val = v.get("id", v.get("idx", v.get("label_idx", 0)))
                    clean_l2i[k_str] = int(idx_val)
                elif isinstance(v, str) and str(k).isdigit():
                    clean_l2i[str(v).strip().lower()] = int(k)
                else:
                    try:
                        clean_l2i[k_str] = int(v)
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
        raw = self.label_to_idx.get(gloss.strip().lower(), None)
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
        var = x.float().pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps).to(x.dtype) * self.weight.to(x.dtype)


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

        # Allow latent cluster learning instead of zero-collapse
        attr_idx_matrix = torch.zeros((vocab_size, 5), dtype=torch.long)
        attr_scalars = torch.zeros((vocab_size, 3), dtype=torch.float32)

        if (
            csv_path is not None
            and label_to_idx is not None
            and Path(csv_path).exists()
        ):
            try:
                import csv

                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)

                    # Maps for categorical features
                    lexclass_map = {"": 0}
                    signtype_map = {"": 0}
                    handshape_map = {"": 0}
                    location_map = {"": 0}
                    category_map = {"": 0}

                    for row in reader:
                        raw_word = (
                            (row.get("LemmaID") or row.get("EntryID") or "")
                            .strip()
                            .lower()
                        )
                        word_clean = raw_word.replace("_", "").replace("-", "")
                        import re

                        word_clean = re.sub(r"\d+$", "", word_clean)

                        # Check original or cleaned word
                        if raw_word in label_to_idx:
                            idx = label_to_idx[raw_word]
                        elif word_clean in label_to_idx:
                            idx = label_to_idx[word_clean]
                        else:
                            continue

                        # CRITICAL FIX (Point 5): Token IDs have an OFFSET of 4 (PAD, BOS, EOS, UNK)
                        idx += 4

                        # 1. Lexical Class
                        lc = row.get("LexicalClass", "").strip()
                        if lc not in lexclass_map:
                            lexclass_map[lc] = len(lexclass_map)
                        assert (
                            lexclass_map[lc] < 20
                        ), f"LexicalClass '{lc}' exceeds max 20 categories (got index {lexclass_map[lc]})"
                        attr_idx_matrix[idx, 0] = lexclass_map[lc]

                        # 2. Sign Type
                        st = row.get("SignType", "").strip()
                        if st not in signtype_map:
                            signtype_map[st] = len(signtype_map)
                        assert (
                            signtype_map[st] < 16
                        ), f"SignType '{st}' exceeds max 16 categories (got index {signtype_map[st]})"
                        attr_idx_matrix[idx, 1] = signtype_map[st]

                        # 3. Handshape
                        hs = row.get("SelectedHandshape", "").strip()
                        if hs not in handshape_map:
                            handshape_map[hs] = len(handshape_map)
                        assert (
                            handshape_map[hs] < 48
                        ), f"Handshape '{hs}' exceeds max 48 categories (got index {handshape_map[hs]})"
                        attr_idx_matrix[idx, 2] = handshape_map[hs]

                        # 4. Location
                        loc = row.get("MajorLocation", "").strip()
                        if loc not in location_map:
                            location_map[loc] = len(location_map)
                        assert (
                            location_map[loc] < 24
                        ), f"Location '{loc}' exceeds max 24 categories (got index {location_map[loc]})"
                        attr_idx_matrix[idx, 3] = location_map[loc]

                        # 5. Semantic Category
                        cat = row.get("SemanticCategory", "").strip()
                        if cat not in category_map:
                            category_map[cat] = len(category_map)
                        assert (
                            category_map[cat] < 36
                        ), f"SemanticCategory '{cat}' exceeds max 36 categories (got index {category_map[cat]})"
                        attr_idx_matrix[idx, 4] = category_map[cat]

                        # Scalars: Flexion, Transparency, Iconicity
                        try:
                            attr_scalars[idx, 0] = float(row.get("Flexion", 0.0) or 0.0)
                        except:
                            pass
                        try:
                            attr_scalars[idx, 1] = float(
                                row.get("Transparency", 0.0) or 0.0
                            )
                        except:
                            pass
                        try:
                            attr_scalars[idx, 2] = float(
                                row.get("Iconicity", 0.0) or 0.0
                            )
                        except:
                            pass
            except Exception as e:
                print(f"[!] Warning: Failed to parse ASL-LEX CSV: {e}", flush=True)

        self.register_buffer("attr_idx_matrix", attr_idx_matrix)
        self.register_buffer("attr_scalars", attr_scalars)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if (token_ids < 0).any() or (token_ids >= self.vocab_size).any():
            raise ValueError("Invalid token IDs passed to ASL-LEX embedding.")
        ids = token_ids
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
        self._jl_proj = nn.Parameter(
            torch.randn(d_model, 16) / math.sqrt(16), requires_grad=False
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        B, T, D = h.shape
        min_half = T // 2
        r_clamp = min(self.r, min_half)
        if T - r_clamp < 65:
            r_clamp = max(0, T - 65)

        fi = kwargs.get("frame_indices", None)
        if r_clamp <= 0:
            return (
                h,
                mask,
                {
                    "T_orig": T,
                    "sorted_routing": torch.arange(T, device=h.device),
                    "mlm_out": kwargs.get("mlm_mask", None),
                    "frame_indices": fi,
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
            if mask is not None:
                h_smooth = h_smooth.masked_fill(~mask.unsqueeze(-1), 0.0)
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
        mlm_mask = kwargs.get("mlm_mask", None)
        if mlm_mask is not None:
            mlm_a, mlm_b = (
                mlm_mask[:, 0::2][:, :min_half],
                mlm_mask[:, 1::2][:, :min_half],
            )
            sim_matrix = sim_matrix.masked_fill(
                mlm_a.unsqueeze(-1) ^ mlm_b.unsqueeze(-2),
                -1e4,
            )

        if mask is not None:
            ma, mb = mask[:, 0::2][:, :min_half], mask[:, 1::2][:, :min_half]
            invalid_a = ~ma.unsqueeze(-1)
            invalid_b = ~mb.unsqueeze(-2)
            cross_invalid = (invalid_a & mb.unsqueeze(-2)) | (
                ma.unsqueeze(-1) & invalid_b
            )
            sim_matrix = sim_matrix.masked_fill(cross_invalid, -1e4)
            pad_pad = invalid_a & invalid_b
            sim_matrix = sim_matrix.masked_fill(pad_pad, 1e4)

        scores, dst_idx = sim_matrix.max(dim=-1)
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

        fi_out = None
        if fi is not None:
            fi_a, fi_b = fi[:, 0::2][:, :min_half], fi[:, 1::2][:, :min_half]
            fi_unordered = torch.cat(
                (
                    [
                        fi_a.gather(1, kept_idx_a),
                        fi_b.clone().scatter_(
                            1, matched_b_indices_local, fi_a.gather(1, merge_idx)
                        ),
                        fi[:, -1:],
                    ]
                    if fi.shape[1] % 2 != 0
                    else [
                        fi_a.gather(1, kept_idx_a),
                        fi_b.clone().scatter_(
                            1, matched_b_indices_local, fi_a.gather(1, merge_idx)
                        ),
                    ]
                ),
                dim=1,
            )
            fi_out = fi_unordered.gather(1, sorted_routing)

        return (
            h_out,
            mask_out,
            {
                "T_orig": T,
                "sorted_routing": sorted_routing,
                "mlm_out": mlm_out,
                "frame_indices": fi_out,
            },
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
            .view(coords.shape[0], T, T, self.kv_heads, self.groups)
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
        )
        v = v.repeat_interleave(self.groups, dim=2).reshape(
            B * self.kv_heads, self.groups, T, self.head_dim
        )

        attn_mask = attn_mask.reshape(B * self.kv_heads, self.groups, T, T)

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
            log_decay = log_decay.masked_fill(key_padding_mask.unsqueeze(-1), -1e4)

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
        max_len: int = 320,
    ):
        super().__init__()
        self.ffn1_norm = RMSNorm(d_model)
        self.ffn1 = SwiGLUFFN(d_model, dim_feedforward, num_layers=num_enc_layers)
        self.drop_path_ffn1 = DropPath(drop_path)
        self.gamma_ffn1 = nn.Parameter(init_values * torch.ones(d_model))

        self.mha_norm = RMSNorm(d_model)
        self.mha = GroupedQueryEncoderAttention(
            d_model=d_model, nhead=nhead, kv_heads=2, max_len=max_len
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
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
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
            * self.mha(
                self.mha_norm(x),
                key_padding_mask=key_padding_mask,
                frame_indices=frame_indices,
            )
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
            (memory_key_padding_mask).view(B, 1, 1, k.size(2)).bool()
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
        use_asl_lex: bool = True,
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
        self.use_asl_lex = use_asl_lex
        if self.use_asl_lex:
            self.asl_lex_emb = RichASLLexEmbeddingTable(
                vocab_size=vocab_size,
                d_model=d_model,
                csv_path=csv_path,
                label_to_idx=label_to_idx,
            )
        else:
            self.asl_lex_emb = None
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

        if (dropped_tgt_ids < 0).any() or (dropped_tgt_ids >= self.vocab_size).any():
            raise ValueError("Invalid target IDs passed to decoder.")
        if getattr(self, "use_asl_lex", True) and self.asl_lex_emb is not None:
            lex_embs = self.asl_lex_emb(dropped_tgt_ids)
            valid_lex_mask = (
                (tgt_ids != GlossVocabulary.PAD_ID).unsqueeze(-1).to(lex_embs.dtype)
            )
            h = self.emb_drop(
                self.token_emb(dropped_tgt_ids) * self.emb_scale
                + lex_embs * self.emb_scale * valid_lex_mask
            )
        else:
            h = self.emb_drop(self.token_emb(dropped_tgt_ids) * self.emb_scale)

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

    def __init__(self, loss_config: Optional[Dict[str, float]] = None):
        super().__init__()
        if loss_config is None:
            loss_config = {
                "seq": 8.0,
                "eos": 2.0,
                "chicago": 8.0,
                "chicago_eos": 2.0,
                "chicago_len": 1.0,
                "english": 8.0,
                "english_eos": 2.0,
                "english_len": 1.0,
                "ctc": 10.0,
                "dense_sem": 1.0,
                "xmodal": 4.0,
                "supcon": 2.0,
                "clr": 0.1,  # Fix 3: 'mlm' renamed to 'clr' to match model output key
                "domain": 1.0,  # Fix 4: 'domain' added so it is not silently dropped
                "aux": 2.0,
                "length": 1.0,
            }

        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(-math.log(v), dtype=torch.float32))
                for name, v in loss_config.items()
            }
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = 0.0
        for name, loss in losses.items():
            # Fix 5: Prevent objective-key drift
            if name not in self.log_vars:
                raise ValueError(
                    f"Unregistered loss key '{name}' produced by model! Please add it to HomoscedasticLossWrapper config."
                )

            if name in self.log_vars:
                # Fix 6: Use exact Kendall & Gal formulation to preserve numeric initialization
                s = self.log_vars[name].to(loss.device)
                s = s.clamp(-5.0, 5.0)  # Bound to prevent objective-key drift exploding
                prec = torch.exp(-s)
                valid_mask = (loss.detach().abs() > 0.0).to(loss.dtype)

                # Sync valid mask so parameter penalty is symmetrically applied across replicas
                if _XLA_AVAILABLE:
                    import torch_xla.core.xla_model as xm

                    if getattr(xm, "xrt_world_size", lambda: 1)() > 1:
                        valid_mask = xm.all_reduce(xm.REDUCE_MAX, valid_mask.clone())

                # Only apply precision weighting and uncertainty penalty if the loss is active
                # This prevents log(sigma) from diverging to negative infinity for inactive tasks
                task_loss = valid_mask * (0.5 * prec * loss + 0.5 * s)
                total_loss += task_loss

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
        w_norm = F.normalize(self.weight.float(), p=2, dim=-1, eps=1e-5).to(
            self.weight.dtype
        )
        # Cosine similarity scaled by learnable temperature tau
        safe_tau = F.softplus(self.tau) + 1.0
        return F.linear(x_norm, w_norm) * safe_tau


class CTCHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = CosineLinear(d_model, vocab_size)

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        if torch.isnan(enc_seq).any():
            print("enc_seq has NaNs!")
        return F.log_softmax(self.proj(enc_seq), dim=-1)


class CrossModalInfoNCE(nn.Module):
    def __init__(self, init_temp: float = 0.07, **kwargs):
        super().__init__()
        target_sp = init_temp - 0.05
        # softplus(x) = log(1 + e^x) -> x = log(e^target_sp - 1)
        self.log_temp = nn.Parameter(torch.tensor(math.log(math.exp(target_sp) - 1.0)))

    def forward(
        self,
        vis_emb: torch.Tensor,
        sent_emb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
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

        v = F.normalize(vis_emb.float(), p=2, dim=-1, eps=1e-8).to(vis_emb.dtype)
        s = F.normalize(sent_emb.float(), p=2, dim=-1, eps=1e-8).to(sent_emb.dtype)

        if world_size > 1 and _XLA_AVAILABLE:
            import torch_xla.core.functions as xf

            v = xf.all_gather(v, dim=0)
            s = xf.all_gather(s, dim=0)
            if valid_mask is not None:
                valid_mask = xm.all_gather(valid_mask, dim=0)
            if sample_weights is not None:
                sample_weights = xm.all_gather(sample_weights, dim=0)

        if v.size(0) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        temp = F.softplus(self.log_temp) + 0.05
        logits = torch.matmul(v, s.transpose(-1, -2)) / temp

        if valid_mask is not None:
            invalid_mask = ~valid_mask
            logits = logits.masked_fill(invalid_mask.unsqueeze(0), -1e9)
            logits = logits.masked_fill(invalid_mask.unsqueeze(1), -1e9)

        labels = torch.arange(v.shape[0], device=v.device)
        loss_v = F.cross_entropy(logits, labels, reduction="none")
        loss_s = F.cross_entropy(logits.transpose(-1, -2), labels, reduction="none")

        loss = (loss_v + loss_s) * 0.5
        if sample_weights is not None:
            loss = loss * sample_weights

        weight_sum = torch.ones_like(loss)
        if valid_mask is not None:
            loss = loss * valid_mask.float()
            weight_sum = valid_mask.float()

        if sample_weights is not None:
            weight_sum = weight_sum * sample_weights

        res = _distributed_normalize(loss.sum(), weight_sum.sum())

        return res


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
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        m = valid_mask.unsqueeze(-1).float()
        valid_counts = m.sum(dim=1).clamp(min=1.0)
        has_tokens = (m.sum(dim=(1, 2)) > 0).float()

        if last_hidden.ndim == 2:
            pred_sent = last_hidden
        else:
            pred_sent = (last_hidden * m).sum(dim=1) / valid_counts

        gt_sent = (gt_lex_embs * m).sum(dim=1) / valid_counts

        p = F.normalize(self.proj_pred(pred_sent).float(), p=2, dim=-1, eps=1e-8)
        g = F.normalize(self.proj_gt(gt_sent).float(), p=2, dim=-1, eps=1e-8).detach()

        cos_sim = (p * g).sum(dim=-1)
        loss = (1.0 - cos_sim) * has_tokens

        weight_sum = has_tokens
        if sample_weights is not None:
            loss = loss * sample_weights
            weight_sum = weight_sum * sample_weights

        return _distributed_normalize(loss.sum(), weight_sum.sum())


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, **kwargs):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor = None,
        enqueue: bool = True,
    ) -> torch.Tensor:
        if labels is None:
            return torch.tensor(0.0, device=features.device, requires_grad=True)
        features = F.normalize(features.float(), p=2, dim=1, eps=1e-5).to(
            features.dtype
        )
        device = features.device

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

        if world_size > 1 and _XLA_AVAILABLE:
            import torch_xla.core.functions as xf

            features = xf.all_gather(features, dim=0)
            labels = xf.all_gather(labels, dim=0)
            if sample_weights is not None:
                sample_weights = xf.all_gather(sample_weights, dim=0)

        B = features.shape[0]
        has_labels = (labels >= 0).any()
        if not has_labels:
            return torch.tensor(0.0, device=device, requires_grad=True)

        ids = torch.arange(B, device=device)
        pos_mask = torch.eq(labels.view(-1, 1), labels.view(1, -1)).float()
        valid_labels = (labels.view(-1, 1) != -1).float()
        pos_mask = pos_mask * valid_labels
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
        if sample_weights is not None:
            loss = (row_loss * valid_rows * sample_weights).sum() / (
                valid_rows * sample_weights
            ).sum().clamp(min=1.0)
        else:
            loss = (row_loss * valid_rows).sum() / valid_rows.sum().clamp(min=1.0)
        if not torch.isfinite(loss):
            raise RuntimeError("SupCon loss resulted in NaN/Inf values.")
        if world_size > 1 and _XLA_AVAILABLE:
            loss = loss * world_size
        return loss


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
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self, x: torch.Tensor, frame_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if frame_indices is not None:
            idx = torch.clamp(frame_indices, max=self.pe.size(1) - 1)
            batch_pe = self.pe.squeeze(0)[idx]
            return x + batch_pe
        return x + self.pe[:, : x.size(1), :]


class ASLFoundationModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2484,
        english_vocab_size: int = 20005,
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
        max_dec_len: int = 512,
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
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
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
                        max_len=max_enc_len,
                    )
                )

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

        self.chicago_decoder = ASLTransformerDecoder(
            vocab_size=42,  # Chicago chars
            d_model=d_dec,
            nhead=nhead_dec,
            kv_heads=kv_heads_dec,
            num_layers=max(2, num_dec_layers // 2),
            ffn_dim=ffn_dec,
            dropout=dropout,
            max_seq_len=64,
            csv_path=None,
            label_to_idx=None,
            use_asl_lex=False,
        )
        self.english_decoder = ASLTransformerDecoder(
            vocab_size=max(4, english_vocab_size),  # English vocab
            d_model=d_dec,
            nhead=nhead_dec,
            kv_heads=kv_heads_dec,
            num_layers=num_dec_layers,
            ffn_dim=ffn_dec,
            dropout=dropout,
            max_seq_len=128,
            csv_path=None,
            label_to_idx=None,
            use_asl_lex=False,
        )
        self.chicago_length_head = nn.Sequential(
            nn.Linear(d_enc, 128), RMSNorm(128), nn.GELU(), nn.Linear(128, 1)
        )
        self.english_length_head = nn.Sequential(
            nn.Linear(d_enc, 128), RMSNorm(128), nn.GELU(), nn.Linear(128, 1)
        )

        self.pos_enc = PositionalEncoding1D(d_enc)

        self.ctc_head = CTCHead(d_enc, vocab_size)
        self.mlm_head = nn.Linear(d_enc, 540)
        self.domain_head = nn.Linear(d_enc, 2)

        # ─── MATH FIX: Encoder Auxiliary Classification Head ───
        # Mathematically forces the Conformer to anchor the latent space into a discrete conceptual cluster
        # BEFORE giving the sequence to the decoder. Bypasses decoder hallucination drift.
        self.aux_gloss_head = CosineLinear(d_enc, vocab_size, init_tau=20.0)

        # ─── NEW: Sequence Length Prediction Head (Fertility) ───
        self.length_head = nn.Sequential(
            nn.Linear(d_enc, 128), RMSNorm(128), nn.GELU(), nn.Linear(128, 1)
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
        self.xmodal_loss_fn = CrossModalInfoNCE(init_temp=0.07)
        self.dense_sem_loss = DenseSentenceSemanticLoss(d_model=d_dec, embed_dim=256)

        nn.init.normal_(self.decoder.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.decoder.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)

    def update_tome_r(self, epoch: int, max_epochs: int):
        # [TPU XLA HOTFIX] ToMe changes the tensor sequence length (e.g., N -> N-r),
        # which forces XLA to compile a brand new static graph in device memory.
        # Instead of increasing `r` every single epoch (which causes 70+ recompilations
        # and guaranteed HBM OOM), we "bucket" `r` into 4 distinct stages.
        # This gives us the dynamic ToMe effect but only compiles 4 graphs total!
        progress = epoch / max(1, max_epochs - 1)

        if progress < 0.25:
            new_r = 10
        elif progress < 0.50:
            new_r = 30
        elif progress < 0.75:
            new_r = 50
        else:
            new_r = 70

        if getattr(self, "tome_r", -1) == new_r:
            return  # No change, avoid unnecessary assignment

        self.tome_r = new_r
        for block in self.blocks:
            if isinstance(block, TokenMergingBlock):
                block.r = new_r

    def _encode(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        mlm_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
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

            lh_nz = (xk[:, :, 0:21, :2] != 0).to(xk.dtype)
            rh_nz = (xk[:, :, 21:42, :2] != 0).to(xk.dtype)

            # Avoid in-place modifications for XLA compatibility
            lh_norm = (xk[:, :, 0:21, :2] - xk[:, :, 0:1, :2]) * lh_nz
            rh_norm = (xk[:, :, 21:42, :2] - xk[:, :, 21:22, :2]) * rh_nz

            xk_list = [lh_norm, rh_norm, xk[:, :, 42:, :2]]
            xk_coords = torch.cat(xk_list, dim=2)

            # Retain non-coordinate channels unmodified
            xk = torch.cat([xk_coords, xk[:, :, :, 2:]], dim=3)

            # Use unmodified x_in for x_flat to retain global spatial context!
            x_flat = x_in.reshape(B, T, -1)
            v_tokens = self.visual_encoder(xk, mask=mask)
        else:
            x_flat = x_in.reshape(B, T, -1) if x_in.dim() == 4 else x_in
            v_tokens = self.visual_encoder(x_in, mask=mask)

        h = self.input_stem(torch.cat([x_flat, v_tokens], dim=-1))
        h = self.pos_enc(h, frame_indices=frame_indices)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)

        if frame_indices is not None:
            frame_indices = frame_indices.long() + 1
            cls_fi = torch.zeros(
                (B, 1), dtype=frame_indices.dtype, device=frame_indices.device
            )
            frame_indices = torch.cat([cls_fi, frame_indices], dim=1)

        cur_mask = mask
        if cur_mask is not None:
            kpm = torch.cat(
                [torch.zeros((B, 1), dtype=torch.bool, device=h.device), ~cur_mask],
                dim=1,
            )
        else:
            kpm = None

        h_pre_tome = None
        mlm_mask_pre_tome = used_mlm_mask

        for idx, block in enumerate(self.blocks):
            if isinstance(block, TokenMergingBlock):
                if h_pre_tome is None:
                    h_pre_tome = h[:, 1:]
                cls_t = h[:, :1]
                seq_t = h[:, 1:]
                fi_t = frame_indices[:, 1:] if frame_indices is not None else None
                seq_t, cur_mask, routing_info = block(
                    seq_t, cur_mask, mlm_mask=used_mlm_mask, frame_indices=fi_t
                )
                if "mlm_out" in routing_info and routing_info["mlm_out"] is not None:
                    used_mlm_mask = routing_info["mlm_out"]
                if (
                    "frame_indices" in routing_info
                    and routing_info["frame_indices"] is not None
                ):
                    frame_indices = torch.cat(
                        [frame_indices[:, :1], routing_info["frame_indices"]], dim=1
                    )
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
                h = block(h, key_padding_mask=kpm, frame_indices=frame_indices)

            if torch.isnan(h).any():
                print(f"NaN introduced at block {idx}!")
                break

        h = self.enc_final_norm(h)
        if torch.isnan(h).any():
            print("NaN introduced at enc_final_norm!")

        if h_pre_tome is None:
            h_pre_tome = h[:, 1:]

        return h[:, 0], h[:, 1:], cur_mask, mlm_mask_pre_tome, h_pre_tome, mask

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gloss_seq: Optional[torch.Tensor] = None,
        chicago_seq: Optional[torch.Tensor] = None,
        english_seq: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        grl_alpha: float = 1.0,
    ) -> Union[Optional[torch.Tensor], Dict]:

        h_cls, h_seq, enc_mask, used_mlm_mask, h_pre_tome, orig_enc_mask = self._encode(
            x, mask, mlm_mask=mlm_mask, frame_indices=frame_indices
        )

        dec_logits, dec_hidden = None, None
        chicago_logits, english_logits = None, None

        def decode_seq(decoder_module, seq_tensor):
            if seq_tensor is not None:
                if seq_tensor.size(1) > 1:
                    dec_in = seq_tensor[:, :-1].contiguous()
                else:
                    dec_in = seq_tensor.contiguous()
                return decoder_module(dec_in, h_seq, memory_key_padding_mask=enc_mask)
            return None, None

        if gloss_seq is not None:
            dec_logits, dec_hidden = decode_seq(self.decoder, gloss_seq)
        if chicago_seq is not None:
            chicago_logits, _ = decode_seq(self.chicago_decoder, chicago_seq)
        if english_seq is not None:
            english_logits, _ = decode_seq(self.english_decoder, english_seq)

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
        pred_len = self.length_head(pooled_enc).squeeze(-1)  # Shape: (B,)
        chicago_pred_len = self.chicago_length_head(pooled_enc).squeeze(-1)
        english_pred_len = self.english_length_head(pooled_enc).squeeze(-1)
        # ────────────────────────────────────

        ctc_log_probs = F.log_softmax(self.ctc_head(h_seq), dim=-1)

        if self.training and h_pre_tome is not None and mlm_mask is not None:
            mlm_logits = self.mlm_head(h_pre_tome)
        else:
            mlm_logits = None

        vis_emb = F.normalize(
            self.visual_proj(h_cls).float(), p=2, dim=-1, eps=1e-5
        ).to(h_cls.dtype)
        if dec_hidden is not None and gloss_seq is not None:
            gt_tokens = gloss_seq[:, 1:]
            valid_mask = (gt_tokens != GlossVocabulary.PAD_ID) & (
                gt_tokens != GlossVocabulary.EOS_ID
            )

            # Generate text embedding WITHOUT passing through the cross-attention decoder
            lex_features = self.decoder.asl_lex_emb(gt_tokens)
            valid_counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

            text_pooled = (
                lex_features * valid_mask.unsqueeze(-1).to(lex_features.dtype)
            ).sum(dim=1) / valid_counts
            sent_emb = F.normalize(
                self.sentence_proj(text_pooled).float(), p=2, dim=-1, eps=1e-5
            ).to(h_cls.dtype)
        else:
            sent_emb = None

        proj_feats = F.normalize(
            self.contrastive_head(h_cls).float(), p=2, dim=-1, eps=1e-5
        ).to(h_cls.dtype)

        domain_logits = None
        if grl_alpha is not None and grl_alpha > 0:
            h_grl = GradientReversalFunction.apply(h_cls, grl_alpha)
            domain_logits = self.domain_head(h_grl)

        return {
            "dec_logits": dec_logits,
            "chicago_logits": chicago_logits,
            "english_logits": english_logits,
            "dec_hidden": dec_hidden,
            "ctc_log_probs": ctc_log_probs,
            "mlm_logits": mlm_logits,
            "mlm_mask": used_mlm_mask,
            "orig_x": x,
            "vis_emb": vis_emb,
            "sent_emb": sent_emb,
            "proj_feats": proj_feats,
            "domain_logits": domain_logits,
            "aux_logits": aux_logits,
            "pred_len": pred_len,
            "chicago_pred_len": chicago_pred_len,
            "english_pred_len": english_pred_len,
            "enc_seq": h_seq,
            "pooled_enc": pooled_enc,
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
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = ctc_log_probs.device
    B, T_enc = ctc_log_probs.size(0), ctc_log_probs.size(1)

    if enc_mask is not None:
        enc_len = enc_mask.sum(dim=-1).long()
    else:
        enc_len = torch.full((B,), T_enc, dtype=torch.long, device=device)

    raw_targets = gloss_seq[:, 1:-1].contiguous()

    # CRITICAL FIX (Point 7): Exclude EOS from valid_mask so CTC doesn't try to predict it.
    valid_mask = (
        (raw_targets != GlossVocabulary.PAD_ID)
        & (raw_targets != GlossVocabulary.EOS_ID)
        & has_valid.unsqueeze(1)
    )
    targets = raw_targets.clone()
    tgt_lengths = valid_mask.sum(dim=-1).long()

    # P1: CTC dummy target insertion. We replace PAD_ID with UNK_ID for padding-only sequences
    # to avoid C++ core dumps from 0-length targets. Since valid_mask is 0 for these sequences,
    # their CTC loss contribution is completely masked out and won't affect gradients.
    zero_len_mask = tgt_lengths == 0
    targets[zero_len_mask, 0] = GlossVocabulary.UNK_ID

    # Vectorized check for consecutive identical tokens (to prevent XLA CPU sync bottleneck)
    same_as_prev = (targets[:, 1:] == targets[:, :-1]) & valid_mask[:, 1:]
    min_ctc_len = tgt_lengths + same_as_prev.sum(dim=-1).long()

    valid_ctc = (enc_len >= min_ctc_len) & (tgt_lengths > 0) & (enc_len > 0)

    # Force valid mathematical bounds via tensors to prevent C++ Core Dumps
    tgt_lengths = tgt_lengths.clamp(min=1)
    enc_len = enc_len.clamp(min=1, max=T_enc)

    loss_vec = F.ctc_loss(
        ctc_log_probs.float().transpose(0, 1),
        targets,
        enc_len,
        tgt_lengths,
        blank=GlossVocabulary.PAD_ID,
        reduction="none",
        zero_infinity=True,
    )
    if not torch.isfinite(loss_vec).all():
        print(
            f"CTC NaN Debug: ctc_log_probs has nans? {torch.isnan(ctc_log_probs).any().item()}"
        )
        print(f"enc_len={enc_len}, tgt_lengths={tgt_lengths}")
        print(
            "Warning: CTC loss resulted in NaN/Inf values. Using nan_to_num to recover."
        )
    loss_vec = torch.nan_to_num(loss_vec)
    loss_vec = loss_vec / tgt_lengths.float().clamp(min=1.0)
    valid_f = valid_ctc.float()
    if sample_weights is not None:
        valid_f = valid_f * sample_weights
    loss_ctc = _distributed_normalize((loss_vec * valid_f).sum(), valid_f.sum())

    with torch.no_grad():
        ctc_eligible = (tgt_lengths > 0).float().sum()
        ctc_used = valid_ctc.float().sum()
        ctc_dropped = ctc_eligible - ctc_used
        mean_enc_len = enc_len.float().mean()
        mean_tgt_len = tgt_lengths.float().mean()
        mean_min_ctc_len = min_ctc_len.float().mean()

    return (
        loss_ctc,
        ctc_eligible,
        ctc_used,
        ctc_dropped,
        mean_enc_len,
        mean_tgt_len,
        mean_min_ctc_len,
    )


def _compute_mlm_loss_safe(
    mlm_logits: torch.Tensor, orig_x: torch.Tensor, mlm_mask: torch.Tensor
) -> torch.Tensor:
    B, T = orig_x.size(0), orig_x.size(1)
    target = orig_x.clone()
    target = target.reshape(B, T, -1)
    mask_f = mlm_mask.unsqueeze(-1).float()
    loss = F.smooth_l1_loss(mlm_logits, target, reduction="none")
    if not torch.isfinite(loss).all():
        raise RuntimeError("MLM loss resulted in NaN/Inf values before nan_to_num.")
    local_loss_sum = (loss * mask_f).sum()
    local_weight = mask_f.sum() * target.size(-1)
    return torch.nan_to_num(
        _distributed_normalize(local_loss_sum, local_weight), nan=0.0
    )


class ModelEMA:
    def __init__(
        self, model: nn.Module, decay_base: float = 0.90, decay_max: float = 0.9999
    ):
        self.decay_base = decay_base
        self.decay_max = decay_max
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()

    def update(self, model: nn.Module, progress: float = 1.0):
        with torch.no_grad():
            # Fix XLA memory leak: Perform math as PyTorch tensors to prevent graph recompilation per step
            device = next(model.parameters()).device
            progress_t = torch.tensor(
                min(1.0, max(0.0, progress)), device=device, dtype=torch.float32
            )

            decay_t = (
                self.decay_max
                - (self.decay_max - self.decay_base)
                * (1.0 + torch.cos(math.pi * progress_t))
                / 2.0
            )

            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Pure TPU-to-TPU tensor math. Zero CPU bottlenecks.
                    decay_param_t = decay_t.to(param.dtype)
                    self.shadow[name].mul_(decay_param_t).add_(
                        param.data, alpha=1.0 - decay_param_t
                    )

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone().detach()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup.clear()


def _get_optimizer_groups(
    model: nn.Module, loss_wrapper: nn.Module, weight_decay: float
):
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
    scaler: getattr(torch.amp, "GradScaler", type(None)) = None,
    epoch: int = 0,
    total_epochs: int = 150,
    prec_dtype: torch.dtype = torch.float16,
    is_master: bool = True,
    accum_steps: int = 4,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    model.train()

    # ─── Dynamic Token Merging Scaling ───
    if hasattr(model, "module"):
        if hasattr(model.module, "update_tome_r"):
            model.module.update_tome_r(epoch, total_epochs)
    elif hasattr(model, "update_tome_r"):
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
        "gloss_trunc": 0.0,
        "chicago_trunc": 0.0,
        "english_trunc": 0.0,
        "ctc_eligible": 0.0,
        "ctc_used": 0.0,
        "ctc_dropped": 0.0,
        "sum_enc_len": 0.0,
        "sum_tgt_len": 0.0,
        "sum_min_ctc": 0.0,
    }
    epoch_start_time = time.time()

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    if is_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    is_master = xm.is_master_ordinal() if is_xla else True

    if is_xla:
        device_type, use_autocast = "xla", False
    else:
        device_type, use_autocast = (
            "cuda" if "cuda" in str(device).lower() else "cpu"
        ), ("cuda" in str(device).lower() and prec_dtype != torch.float32)

    # scaler passed in

    progress = float(max(0, epoch)) / float(max(1, total_epochs - 1))
    grl_alpha = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
    label_smoothing = max(0.05, 0.15 - 0.10 * progress)
    POLY1_EPS = 1.0

    def compute_seq_loss(
        logits_f, gt_ids, valid_mask, sample_weights=None, class_weights=None, gamma=2.0
    ):
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = gt_ids.reshape(-1)
        bad = (tf < 0) | (tf >= V)
        if bad.any():
            raise RuntimeError(
                f"Training Sequence ID out of bounds. Found values outside [0, {V-1}]."
            )
        vf = valid_mask.reshape(-1).float()

        # Exclude EOS from seq loss to separate concerns
        eos_mask = tf == GlossVocabulary.EOS_ID
        vf = vf * (~eos_mask).float()

        if class_weights is not None:
            vf = vf * class_weights[tf]

        if sample_weights is not None:
            sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
            vf = vf * sw

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
        return _distributed_normalize((poly1 * vf).sum(), vf.sum())

    def compute_eos_loss(logits_f, gt_ids, valid_mask, sample_weights=None, gamma=2.0):
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = gt_ids.reshape(-1)
        bad = (tf < 0) | (tf >= V)
        if bad.any():
            raise RuntimeError(
                f"Training Sequence ID out of bounds. Found values outside [0, {V-1}]."
            )
        vf = valid_mask.reshape(-1).float()

        # Only compute for EOS
        eos_mask = (tf == GlossVocabulary.EOS_ID).float()
        vf = vf * eos_mask

        if sample_weights is not None:
            sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
            vf = vf * sw

        log_p = F.log_softmax(lf, dim=-1)
        p_target = torch.exp(log_p.gather(1, tf.unsqueeze(1)).squeeze(1)).clamp(
            min=1e-6, max=1.0
        )
        focal_weight = torch.pow(1.0 - p_target, gamma)

        ce_unsmoothed = F.nll_loss(
            log_p, tf, ignore_index=GlossVocabulary.PAD_ID, reduction="none"
        )

        # We don't apply label smoothing to EOS to force hard termination
        poly1 = focal_weight * ce_unsmoothed + POLY1_EPS * (1.0 - p_target)
        return _distributed_normalize((poly1 * vf).sum(), vf.sum())

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

        features, mask, labels, frame_indices = (
            batch["feature"].to(device),
            batch["mask"].to(device),
            batch.get(
                "label", torch.zeros(batch["feature"].size(0), dtype=torch.long)
            ).to(device),
            batch["frame_indices"].to(device) if "frame_indices" in batch else None,
        )
        sample_weight = batch.get(
            "sample_weight", torch.ones_like(labels, dtype=torch.float32)
        ).to(device)
        domain_tgts, has_domain = batch.get(
            "domain_label", torch.zeros_like(labels)
        ).to(device), batch.get("has_domain_label", torch.zeros_like(labels)).to(device)
        gloss_seq, gloss_len, has_valid, mlm_mask = (
            batch["gloss_seq"].to(device),
            batch["gloss_len"].to(device),
            batch["has_valid_gloss"].to(device),
            batch.get("mlm_mask", None),
        )
        chicago_seq, chicago_len, has_valid_chicago = (
            batch["chicago_seq"].to(device),
            batch["chicago_len"].to(device),
            batch["has_valid_chicago"].to(device),
        )
        english_seq, english_len, has_valid_english = (
            batch["english_seq"].to(device),
            batch["english_len"].to(device),
            batch["has_valid_english"].to(device),
        )
        if mlm_mask is not None:
            mlm_mask = mlm_mask.to(device)

        # curriculum removed to prevent CTC NaN

        if (step_idx - 1) % accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)

        if mlm_mask is None:
            mlm_mask = (torch.rand(features.shape[:2], device=device) < 0.15) & mask

        def forward_and_losses():
            out = model(
                features,
                mask=mask,
                gloss_seq=gloss_seq,
                chicago_seq=chicago_seq,
                english_seq=english_seq,
                mlm_mask=mlm_mask,
                frame_indices=frame_indices,
                return_aux=True,
                grl_alpha=grl_alpha,
            )
            (
                dec_logits,
                chicago_logits,
                english_logits,
                dec_hidden,
                ctc_log_probs,
                vis_emb,
                sent_emb,
                proj_feats,
                domain_logits,
                aux_logits,
                enc_mask,
                pred_len,
                chicago_pred_len,
                english_pred_len,
            ) = (
                out["dec_logits"],
                out["chicago_logits"],
                out["english_logits"],
                out["dec_hidden"],
                out["ctc_log_probs"],
                out["vis_emb"],
                out["sent_emb"],
                out["proj_feats"],
                out["domain_logits"],
                out["aux_logits"],
                out["enc_mask"],
                out["pred_len"],
                out["chicago_pred_len"],
                out["english_pred_len"],
            )
            pooled_enc = out["pooled_enc"]

            gt_tokens = gloss_seq[:, 1:].contiguous()
            token_mask = (gt_tokens != GlossVocabulary.PAD_ID) & has_valid.bool().unsqueeze(-1)
            valid_gloss_mask = token_mask & (gt_tokens != GlossVocabulary.EOS_ID)

            # Gloss length & sequence loss masked strictly by has_valid_gloss
            if True:
                # Target length should match CTC length, excluding PAD and EOS
                valid_seq_mask = (gloss_seq[:, 1:] != GlossVocabulary.PAD_ID) & (
                    gloss_seq[:, 1:] != GlossVocabulary.EOS_ID
                )
                target_len = (
                    (valid_seq_mask * has_valid_gloss.unsqueeze(-1)).sum(dim=1).float()
                )
                loss_length = F.smooth_l1_loss(pred_len, target_len, reduction="none")
                loss_length = _distributed_normalize(
                    (loss_length * sample_weight * has_valid_gloss.float()).sum(),
                    (has_valid_gloss.float() * sample_weight).sum(),
                )

                loss_seq = (
                    compute_seq_loss(
                        dec_logits,
                        gt_tokens,
                        valid_gloss_mask,
                        sample_weights=sample_weight,
                        class_weights=class_weights,
                    )
                    if dec_logits is not None
                    else torch.tensor(0.0, device=device, requires_grad=True)
                )
                loss_eos = (
                    compute_eos_loss(
                        dec_logits,
                        gt_tokens,
                        token_mask,
                        sample_weights=sample_weight,
                    )
                    if dec_logits is not None
                    else torch.tensor(0.0, device=device, requires_grad=True)
                )
            else:
                loss_length = torch.tensor(0.0, device=device, requires_grad=True)
                loss_seq = torch.tensor(0.0, device=device, requires_grad=True)
                loss_eos = torch.tensor(0.0, device=device, requires_grad=True)

            # --- CHICAGO LOSS (Sample-wise Masking) ---
            chicago_gt = (
                chicago_seq[:, 1:].contiguous() if chicago_seq is not None else None
            )
            chicago_valid = (
                (chicago_gt != GlossVocabulary.PAD_ID)
                if chicago_gt is not None
                else None
            )

            if chicago_pred_len is not None and has_valid_chicago.any():
                c_valid_f = has_valid_chicago.float()
                c_target_len = (
                    (chicago_valid * has_valid_chicago.unsqueeze(-1)).sum(dim=1).float()
                )
                loss_chicago_len = F.smooth_l1_loss(
                    chicago_pred_len, c_target_len, reduction="none"
                )
                loss_chicago_len = _distributed_normalize(
                    (loss_chicago_len * sample_weight * c_valid_f).sum(),
                    (c_valid_f * sample_weight).sum(),
                )
            else:
                loss_chicago_len = torch.tensor(0.0, device=device, requires_grad=True)

            if chicago_logits is not None and has_valid_chicago.any():
                chi_token_mask = chicago_valid & has_valid_chicago.unsqueeze(-1)
                valid_chicago_mask = chi_token_mask & (chicago_gt != GlossVocabulary.EOS_ID)
                loss_chicago = compute_seq_loss(
                    chicago_logits,
                    chicago_gt,
                    valid_chicago_mask,
                    sample_weights=sample_weight,
                )
                loss_chicago_eos = compute_eos_loss(
                    chicago_logits,
                    chicago_gt,
                    chi_token_mask,
                    sample_weights=sample_weight,
                )
            else:
                loss_chicago = torch.tensor(0.0, device=device, requires_grad=True)
                loss_chicago_eos = torch.tensor(0.0, device=device, requires_grad=True)

            # --- ENGLISH LOSS (Sample-wise Masking) ---
            english_gt = (
                english_seq[:, 1:].contiguous() if english_seq is not None else None
            )
            english_valid = (
                (english_gt != GlossVocabulary.PAD_ID)
                if english_gt is not None
                else None
            )

            if english_pred_len is not None and has_valid_english.any():
                e_valid_f = has_valid_english.float()
                e_target_len = (
                    (english_valid * has_valid_english.unsqueeze(-1)).sum(dim=1).float()
                )
                loss_english_len = F.smooth_l1_loss(
                    english_pred_len, e_target_len, reduction="none"
                )
                loss_english_len = _distributed_normalize(
                    (loss_english_len * sample_weight * e_valid_f).sum(),
                    (e_valid_f * sample_weight).sum(),
                )
            else:
                loss_english_len = torch.tensor(0.0, device=device, requires_grad=True)

            if english_logits is not None and has_valid_english.any():
                eng_token_mask = english_valid & has_valid_english.unsqueeze(-1)
                valid_english_mask = eng_token_mask & (english_gt != GlossVocabulary.EOS_ID)
                loss_english = compute_seq_loss(
                    english_logits,
                    english_gt,
                    valid_english_mask,
                    sample_weights=sample_weight,
                )
                loss_english_eos = compute_eos_loss(
                    english_logits,
                    english_gt,
                    eng_token_mask,
                    sample_weights=sample_weight,
                )
            else:
                loss_english = torch.tensor(0.0, device=device, requires_grad=True)
                loss_english_eos = torch.tensor(0.0, device=device, requires_grad=True)

            # --- AUXILIARY GROUNDING & GLOSS AUX LOSSES ---
            raw_model = model.module if hasattr(model, "module") else model
            is_isolated = batch.get(
                "is_isolated", torch.ones_like(labels, dtype=torch.bool)
            ).to(device)
            isolated_f = is_isolated.float()

            aux_target = labels + GlossVocabulary.OFFSET
            bad_aux = (aux_target < 0) | (aux_target >= raw_model.vocab_size)
            if bad_aux.any():
                raise RuntimeError(
                    f"Auxiliary Target ID out of bounds. Vocab Size: {raw_model.vocab_size}"
                )
            loss_aux = F.cross_entropy(
                aux_logits.float(), aux_target, reduction="none", label_smoothing=0.1
            )
            loss_aux = _distributed_normalize(
                (loss_aux * sample_weight * isolated_f).sum(),
                (isolated_f * sample_weight).sum(),
            )

            if True:
                loss_ctc, c_elig, c_used, c_drop, m_enc, m_tgt, m_min = (
                    _compute_ctc_loss_safe(
                        ctc_log_probs,
                        gloss_seq,
                        gloss_len,
                        enc_mask,
                        has_valid_gloss,
                        sample_weights=sample_weight,
                    )
                )

                pooled_enc = out["pooled_enc"]
                gt_tokens = (
                    gloss_seq[:, 1:].contiguous() if gloss_seq is not None else None
                )
                valid_gloss_mask = (
                    (gt_tokens != GlossVocabulary.PAD_ID)
                    if gt_tokens is not None
                    else None
                )

                loss_dense_sem = raw_model.dense_sem_loss(
                    pooled_enc,
                    raw_model.decoder.asl_lex_emb(gt_tokens),
                    valid_gloss_mask & (gt_tokens != GlossVocabulary.EOS_ID),
                    sample_weights=sample_weight,
                )

                vis_emb = out.get("vis_emb", None)
                sent_emb = out.get("sent_emb", None)
                if sent_emb is not None and vis_emb is not None:
                    loss_xmodal = raw_model.xmodal_loss_fn(
                        vis_emb, sent_emb, has_valid_gloss, sample_weights=sample_weight
                    )
                else:
                    loss_xmodal = torch.tensor(0.0, device=device, requires_grad=True)
            else:
                loss_ctc = torch.tensor(0.0, device=device, requires_grad=True)
                c_elig = torch.tensor(0.0, device=device)
                c_used = torch.tensor(0.0, device=device)
                c_drop = torch.tensor(0.0, device=device)
                m_enc = torch.tensor(0.0, device=device)
                m_tgt = torch.tensor(0.0, device=device)
                m_min = torch.tensor(0.0, device=device)
                loss_dense_sem = torch.tensor(0.0, device=device, requires_grad=True)
                loss_xmodal = torch.tensor(0.0, device=device, requires_grad=True)

            if True:
                isolated_labels = torch.where(is_isolated, labels, -1)
                loss_supcon = supcon_fn(
                    proj_feats.float(), isolated_labels, sample_weight
                )
            else:
                loss_supcon = torch.tensor(0.0, device=device, requires_grad=True)

            # Domain adaptation is completely inactive in the current architecture.
            # has_dom_f = has_domain.float()
            loss_domain = torch.tensor(0.0, device=device, requires_grad=True)

            loss_clr = (
                _compute_mlm_loss_safe(out["mlm_logits"], out["orig_x"], mlm_mask)
                if out["mlm_logits"] is not None and mlm_mask is not None
                else torch.tensor(0.0, device=device, requires_grad=True)
            )

            loss_terms = {
                "seq": loss_seq,
                "eos": loss_eos,
                "chicago": loss_chicago,
                "chicago_eos": loss_chicago_eos,
                "chicago_len": loss_chicago_len,
                "english": loss_english,
                "english_eos": loss_english_eos,
                "english_len": loss_english_len,
                "ctc": loss_ctc,
                "dense_sem": loss_dense_sem,
                "xmodal": loss_xmodal,
                "supcon": loss_supcon,
                "domain": loss_domain,
                "clr": loss_clr,
                "aux": loss_aux,
                "length": loss_length,
            }
            raw_loss = loss_wrapper(loss_terms)

            with torch.no_grad():
                # Gloss Metrics
                preds = (
                    dec_logits.argmax(dim=-1) if dec_logits is not None else gt_tokens
                )
                vg_eval = valid_gloss_mask & (gt_tokens != GlossVocabulary.EOS_ID)
                vg_f = vg_eval.float()
                nc_t = ((preds == gt_tokens).float() * vg_f).sum()
                nt_t = vg_f.sum()

                # Chicago Metrics
                chicago_nc_t = torch.tensor(0.0, device=device)
                chicago_nt_t = torch.tensor(0.0, device=device)
                if chicago_logits is not None and has_valid_chicago.any():
                    c_preds = chicago_logits.argmax(dim=-1)
                    vc_eval = (
                        chicago_valid
                        & has_valid_chicago.unsqueeze(-1)
                        & (chicago_gt != GlossVocabulary.EOS_ID)
                    )
                    vc_f = vc_eval.float()
                    chicago_nc_t = ((c_preds == chicago_gt).float() * vc_f).sum()
                    chicago_nt_t = vc_f.sum()

                # English Metrics
                english_nc_t = torch.tensor(0.0, device=device)
                english_nt_t = torch.tensor(0.0, device=device)
                if english_logits is not None and has_valid_english.any():
                    e_preds = english_logits.argmax(dim=-1)
                    ve_eval = (
                        english_valid
                        & has_valid_english.unsqueeze(-1)
                        & (english_gt != GlossVocabulary.EOS_ID)
                    )
                    ve_f = ve_eval.float()
                    english_nc_t = ((e_preds == english_gt).float() * ve_f).sum()
                    english_nt_t = ve_f.sum()

            return (
                raw_loss,
                dec_logits,
                nc_t,
                nt_t,
                chicago_nc_t,
                chicago_nt_t,
                english_nc_t,
                english_nt_t,
                loss_seq.detach(),
                loss_aux.detach(),
                loss_ctc.detach(),
                loss_dense_sem.detach(),
                loss_chicago.detach(),
                loss_english.detach(),
                c_elig.detach(),
                c_used.detach(),
                c_drop.detach(),
                m_enc.detach(),
                m_tgt.detach(),
                m_min.detach(),
            )

        if use_autocast:
            with torch.autocast(device_type, dtype=prec_dtype):
                (
                    raw_loss,
                    dec_logits,
                    nc_t,
                    nt_t,
                    c_nc_t,
                    c_nt_t,
                    e_nc_t,
                    e_nt_t,
                    l_seq,
                    l_aux,
                    l_ctc,
                    l_sem,
                    l_chi,
                    l_eng,
                    c_elig,
                    c_used,
                    c_drop,
                    m_enc,
                    m_tgt,
                    m_min,
                ) = forward_and_losses()
        else:
            (
                raw_loss,
                dec_logits,
                nc_t,
                nt_t,
                c_nc_t,
                c_nt_t,
                e_nc_t,
                e_nt_t,
                l_seq,
                l_aux,
                l_ctc,
                l_sem,
                l_chi,
                l_eng,
                c_elig,
                c_used,
                c_drop,
                m_enc,
                m_tgt,
                m_min,
)

        if not torch.isfinite(raw_loss).all():
            raise RuntimeError(
                f"NaN/Inf loss encountered at Epoch {epoch} step {step_idx}."
            )

        loss = raw_loss / float(accum_steps)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx % accum_steps == 0) or (step_idx == min_batches):
            if is_xla:
                import torch_xla.core.xla_model as xm
                import torch_xla.utils.utils as xu

                xu.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0,
                )

                xm.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)

                raw_m = model.module if hasattr(model, "module") else model
                if ema is not None:
                    progress_ema = (epoch * min_batches + step_idx) / max(
                        1, total_epochs * min_batches
                    )
                    ema.update(raw_m, progress_ema)
                if hasattr(raw_m, "dense_sem_loss"):
                    raw_m.dense_sem_loss.update_momentum()

                if scheduler is not None:
                    try:
                        scheduler.step()
                    except Exception as e:
                        raise RuntimeError(
                            f"LR scheduler failed at epoch {epoch}, step {step_idx}: {e}"
                        ) from e

                log_freq = min(25, max(1, min_batches // 10))
                if is_master and (
                    (step_idx % log_freq == 0) or (step_idx == min_batches)
                ):

                    def _async_step_print(
                        l_val,
                        s_val,
                        aux_val,
                        c_val,
                        sm_val,
                        chi_val,
                        eng_val,
                        nc_val,
                        nt_val,
                        cnc_val,
                        cnt_val,
                        enc_val,
                        ent_val,
                        st_idx,
                        m_batches,
                        ep,
                        tot_ep,
                        lr_val,
                        t_start,
                    ):
                        g_acc = (float(nc_val) / max(1.0, float(nt_val))) * 100.0
                        c_acc = (float(cnc_val) / max(1.0, float(cnt_val))) * 100.0
                        e_acc = (float(enc_val) / max(1.0, float(ent_val))) * 100.0
                        speed = float(st_idx) / max(0.001, time.time() - t_start)
                        msg = (
                            f"  [Epoch {ep:03d}/{tot_ep:03d} | Step {st_idx:04d}/{m_batches:04d}] "
                            f"Loss: {float(l_val):.4f} (Gloss:{float(s_val):.4f} Chi:{float(chi_val):.4f} Eng:{float(eng_val):.4f}) | "
                            f"Acc (Gloss:{g_acc:.1f}% Chi:{c_acc:.1f}% Eng:{e_acc:.1f}%) | LR: {lr_val:.2e} | {speed:.1f} it/s"
                        )
                        print(msg, flush=True)

                    xm.add_step_closure(
                        _async_step_print,
                        args=(
                            raw_loss.detach(),
                            l_seq.detach(),
                            l_aux.detach(),
                            l_ctc.detach(),
                            l_sem.detach(),
                            l_chi.detach(),
                            l_eng.detach(),
                            nc_t.detach(),
                            nt_t.detach(),
                            c_nc_t.detach(),
                            c_nt_t.detach(),
                            e_nc_t.detach(),
                            e_nt_t.detach(),
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

                # Standard global gradient clipping for cross-device equivalence
                # On non-XLA devices, standard PyTorch clip is safe
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
                    try:
                        scheduler.step()
                    except Exception as e:
                        raise RuntimeError(
                            f"LR scheduler failed at epoch {epoch}, step {step_idx}: {e}"
                        ) from e
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
        tracker["gloss_trunc"] += (
            batch.get("gloss_trunc", torch.tensor([False])).float().sum().item()
        )
        tracker["chicago_trunc"] += (
            batch.get("chicago_trunc", torch.tensor([False])).float().sum().item()
        )
        tracker["english_trunc"] += (
            batch.get("english_trunc", torch.tensor([False])).float().sum().item()
        )
        tracker["ctc_eligible"] += float(c_elig)
        tracker["ctc_used"] += float(c_used)
        tracker["ctc_dropped"] += float(c_drop)
        tracker["sum_enc_len"] += float(m_enc)
        tracker["sum_tgt_len"] += float(m_tgt)
        tracker["sum_min_ctc"] += float(m_min)

        if is_xla and step_idx >= min_batches:
            break

    if is_xla:
        xm.mark_step()
        xm.rendezvous("end_of_epoch")
        g_tr = xm.mesh_reduce("g_tr", tracker["gloss_trunc"], sum)
        c_tr = xm.mesh_reduce("c_tr", tracker["chicago_trunc"], sum)
        e_tr = xm.mesh_reduce("e_tr", tracker["english_trunc"], sum)

        for k in [
            "ctc_eligible",
            "ctc_used",
            "ctc_dropped",
            "sum_enc_len",
            "sum_tgt_len",
            "sum_min_ctc",
        ]:
            t_val = xm.all_reduce(
                xm.REDUCE_SUM, torch.tensor(tracker[k], device=device)
            )
            tracker[k] = t_val.item()

        t_loss = torch.tensor(tracker["loss"], device=device)
        t_corr = torch.tensor(tracker["corr"], device=device)
        t_tot = torch.tensor(tracker["total"], device=device)
        t_step = torch.tensor(min_batches, device=device)

        t_loss = xm.all_reduce(xm.REDUCE_SUM, t_loss)
        t_corr = xm.all_reduce(xm.REDUCE_SUM, t_corr)
        t_tot = xm.all_reduce(xm.REDUCE_SUM, t_tot)
        t_step = xm.all_reduce(xm.REDUCE_SUM, t_step)

        tracker["loss"] = t_loss.item()
        tracker["corr"] = t_corr.item()
        tracker["total"] = t_tot.item()
        global_batches = int(t_step.item()) * world_size
    else:
        g_tr = tracker["gloss_trunc"]
        c_tr = tracker["chicago_trunc"]
        e_tr = tracker["english_trunc"]
        global_batches = min_batches

    if is_master:
        print(
            f"[Epoch {epoch} Truncation] Gloss: {int(g_tr)} | Chicago: {int(c_tr)} | English: {int(e_tr)}",
            flush=True,
        )

        drop_rate = (tracker["ctc_dropped"] / max(1.0, tracker["ctc_eligible"])) * 100.0
        print(
            f"[Epoch {epoch} CTC] Eligible: {int(tracker['ctc_eligible'])} | Used: {int(tracker['ctc_used'])} | Dropped: {int(tracker['ctc_dropped'])} ({drop_rate:.2f}%)"
        )
        print(
            f"[Epoch {epoch} CTC Lengths] Mean Enc: {tracker['sum_enc_len']/max(1, global_batches):.1f} | Mean Tgt: {tracker['sum_tgt_len']/max(1, global_batches):.1f} | Min CTC: {tracker['sum_min_ctc']/max(1, global_batches):.1f}"
        )

    avg_loss = tracker["loss"] / float(max(1, global_batches))
    token_acc = (tracker["corr"] / max(1.0, tracker["total"])) * 100.0
    return {
        "loss": avg_loss,
        "gloss_acc": token_acc,
    }


def validate_epoch_tpu(
    model: nn.Module,
    loader: DataLoader,
    loss_wrapper: HomoscedasticLossWrapper,
    device: torch.device,
    epoch: int = 0,
    total_epochs: int = 150,
    prec_dtype: torch.dtype = torch.float16,
    is_master: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
) -> Tuple[float, float]:
    model.eval()

    tracker = {
        "loss": 0.0,
        "chi_loss": 0.0,
        "eng_loss": 0.0,
        "corr": 0.0,
        "total": 0.0,
        "chi_corr": 0.0,
        "chi_total": 0.0,
        "eng_corr": 0.0,
        "eng_total": 0.0,
        "eng_trunc_count": 0.0,
        "eng_trunc_total": 0.0,
    }

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    if is_xla:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    is_master = xm.is_master_ordinal() if is_xla else True

    if is_xla:
        device_type, use_autocast = "xla", False
    else:
        device_type, use_autocast = (
            "cuda" if "cuda" in str(device).lower() else "cpu"
        ), ("cuda" in str(device).lower() and prec_dtype != torch.float32)

    def compute_seq_loss(
        logits_f, gt_ids, valid_mask, class_weights=None, sample_weights=None
    ):
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = gt_ids.reshape(-1)

        vf = valid_mask.reshape(-1).float()
        eos_mask = tf == GlossVocabulary.EOS_ID
        vf = vf * (~eos_mask).float()

        if class_weights is not None:
            vf = vf * class_weights[tf]

        ce_unsmoothed = F.cross_entropy(lf, tf, reduction="none")
        ce_uniform = -F.log_softmax(lf, dim=-1).mean(dim=-1)
        ce_smoothed = (
            1.0 - label_smoothing
        ) * ce_unsmoothed + label_smoothing * ce_uniform
        loss = ce_smoothed * vf
        if sample_weights is not None:
            sw = sample_weights.view(-1)
            loss = loss * sw
            vf = vf * sw

        return _distributed_normalize(loss.sum(), vf.sum())

    def compute_eos_loss(logits_f, gt_ids, valid_mask, sample_weights=None):
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = gt_ids.reshape(-1)
        vf = valid_mask.reshape(-1).float()
        eos_mask = tf == GlossVocabulary.EOS_ID
        vf = vf * eos_mask.float()

        ce_unsmoothed = F.cross_entropy(lf, tf, reduction="none")
        ce_uniform = -F.log_softmax(lf, dim=-1).mean(dim=-1)
        ce_smoothed = (
            1.0 - label_smoothing
        ) * ce_unsmoothed + label_smoothing * ce_uniform
        loss = ce_smoothed * vf
        if sample_weights is not None:
            sw = sample_weights.view(-1)
            loss = loss * sw
            vf = vf * sw

        return _distributed_normalize(loss.sum(), vf.sum())

    total_val_batches = len(loader)
    min_val_batches = total_val_batches
    if is_xla:
        min_val_batches = int(
            xm.mesh_reduce("min_val_batches", total_val_batches, lambda x: min(x))
        )

    with torch.no_grad():
        para_loader = (
            pl.ParallelLoader(loader, [device]).per_device_loader(device)
            if is_xla
            else loader
        )

        for step_idx, batch in enumerate(para_loader, 1):
            if step_idx > min_val_batches:
                break
            features = batch["feature"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            frame_indices = (
                batch["frame_indices"].to(device, non_blocking=True)
                if "frame_indices" in batch
                else None
            )
            gloss_seq = batch["gloss_seq"].to(device, non_blocking=True)
            has_valid_gloss = batch["has_valid_gloss"].to(device, non_blocking=True)
            chicago_seq = batch["chicago_seq"].to(device, non_blocking=True)
            has_valid_chicago = batch["has_valid_chicago"].to(device, non_blocking=True)
            english_seq = batch["english_seq"].to(device, non_blocking=True)
            has_valid_english = batch["has_valid_english"].to(device, non_blocking=True)

            def forward_and_losses():
                out = model(
                    features,
                    mask=mask,
                    gloss_seq=gloss_seq,
                    chicago_seq=chicago_seq,
                    english_seq=english_seq,
                    mlm_mask=None,
                    frame_indices=frame_indices,
                    return_aux=True,
                    grl_alpha=0.0,
                )
                dec_logits = out.get("dec_logits") if isinstance(out, dict) else out
                chicago_logits = (
                    out.get("chicago_logits") if isinstance(out, dict) else None
                )
                english_logits = (
                    out.get("english_logits") if isinstance(out, dict) else None
                )

                sample_weight = batch.get(
                    "sample_weight", torch.ones(batch["feature"].size(0), dtype=torch.float32, device=device)
                ).to(device)
                english_trunc_flag = batch.get(
                    "english_trunc", torch.zeros(has_valid_english.shape, dtype=torch.bool, device=device)
                ).to(device)
                gt_tokens = gloss_seq[:, 1:].contiguous()
                token_mask = (gt_tokens != GlossVocabulary.PAD_ID) & has_valid_gloss.unsqueeze(-1)
                valid_mask = token_mask & (gt_tokens != GlossVocabulary.EOS_ID)

                loss_seq = (
                    compute_seq_loss(
                        dec_logits,
                        gt_tokens,
                        valid_mask,
                        class_weights=class_weights,
                        sample_weights=sample_weight,
                    )
                    if dec_logits is not None
                    else torch.tensor(0.0, device=device)
                )
                loss_eos = (
                    compute_eos_loss(
                        dec_logits, gt_tokens, token_mask, sample_weights=sample_weight
                    )
                    if dec_logits is not None
                    else torch.tensor(0.0, device=device)
                )

                nc_t, nt_t = torch.tensor(0.0, device=device), torch.tensor(
                    0.0, device=device
                )
                if dec_logits is not None and True:
                    preds = dec_logits.argmax(dim=-1)
                    valid_f = valid_mask.float()
                    nc_t = ((preds == gt_tokens).float() * valid_f).sum()
                    nt_t = valid_f.sum()

                c_nc_t, c_nt_t = torch.tensor(0.0, device=device), torch.tensor(
                    0.0, device=device
                )
                loss_chi = torch.tensor(0.0, device=device)
                loss_chi_eos = torch.tensor(0.0, device=device)
                if chicago_logits is not None and True:
                    chicago_gt = chicago_seq[:, 1:].contiguous()
                    c_preds = chicago_logits.argmax(dim=-1)
                    chi_token_mask = (chicago_gt != GlossVocabulary.PAD_ID) & has_valid_chicago.unsqueeze(-1)
                    c_valid_mask = chi_token_mask & (chicago_gt != GlossVocabulary.EOS_ID)
                    c_valid = c_valid_mask.float()
                    loss_chi = compute_seq_loss(
                        chicago_logits,
                        chicago_gt,
                        c_valid_mask,
                        sample_weights=sample_weight,
                    )
                    loss_chi_eos = compute_eos_loss(
                        chicago_logits,
                        chicago_gt,
                        chi_token_mask,
                        sample_weights=sample_weight,
                    )
                    c_nc_t = ((c_preds == chicago_gt).float() * c_valid).sum()
                    c_nt_t = c_valid.sum()

                e_nc_t, e_nt_t = torch.tensor(0.0, device=device), torch.tensor(
                    0.0, device=device
                )
                loss_eng = torch.tensor(0.0, device=device)
                loss_eng_eos = torch.tensor(0.0, device=device)
                if english_logits is not None and True:
                    english_gt = english_seq[:, 1:].contiguous()
                    e_preds = english_logits.argmax(dim=-1)
                    eng_token_mask = (english_gt != GlossVocabulary.PAD_ID) & has_valid_english.unsqueeze(-1)
                    e_valid_mask = eng_token_mask & (english_gt != GlossVocabulary.EOS_ID)
                    e_valid = e_valid_mask.float()
                    loss_eng = compute_seq_loss(
                        english_logits,
                        english_gt,
                        e_valid_mask,
                        sample_weights=sample_weight,
                    )
                    loss_eng_eos = compute_eos_loss(
                        english_logits,
                        english_gt,
                        eng_token_mask,
                        sample_weights=sample_weight,
                    )
                    e_nc_t = ((e_preds == english_gt).float() * e_valid).sum()
                    e_nt_t = e_valid.sum()

                e_trunc_c, e_trunc_t = torch.tensor(0.0, device=device), torch.tensor(
                    0.0, device=device
                )
                if True:
                    eng_trunc_flag = batch.get(
                        "english_trunc", torch.zeros_like(has_valid_english)
                    )
                    e_trunc_c = eng_trunc_flag[has_valid_english].float().sum()
                    e_trunc_t = has_valid_english.float().sum()

                loss_terms = {
                    "seq": loss_seq,
                    "eos": loss_eos,
                    "chicago": loss_chi,
                    "chicago_eos": loss_chi_eos,
                    "english": loss_eng,
                    "english_eos": loss_eng_eos,
                    "ctc": torch.tensor(0.0, device=device),
                    "dense_sem": torch.tensor(0.0, device=device),
                    "xmodal": torch.tensor(0.0, device=device),
                    "supcon": torch.tensor(0.0, device=device),
                    "domain": torch.tensor(0.0, device=device),
                    "clr": torch.tensor(0.0, device=device),
                    "aux": torch.tensor(0.0, device=device),
                    "chicago_len": torch.tensor(0.0, device=device),
                    "english_len": torch.tensor(0.0, device=device),
                    "length": torch.tensor(0.0, device=device),
                }
                raw_loss = loss_wrapper(loss_terms)

                return (
                    raw_loss,
                    loss_chi,
                    loss_eng,
                    nc_t,
                    nt_t,
                    c_nc_t,
                    c_nt_t,
                    e_nc_t,
                    e_nt_t,
                    e_trunc_c,
                    e_trunc_t,
                )

            if use_autocast:
                with torch.autocast(device_type, dtype=prec_dtype):
                    (
                        raw_loss,
                        l_chi,
                        l_eng,
                        nc_t,
                        nt_t,
                        c_nc_t,
                        c_nt_t,
                        e_nc_t,
                        e_nt_t,
                        e_trunc_c,
                        e_trunc_t,
                    ) = forward_and_losses()
            else:
                (
                    raw_loss,
                    l_chi,
                    l_eng,
                    nc_t,
                    nt_t,
                    c_nc_t,
                    c_nt_t,
                    e_nc_t,
                    e_nt_t,
                    e_trunc_c,
                    e_trunc_t,
                ) = forward_and_losses()

            tracker["loss"] += raw_loss.detach()
            tracker["chi_loss"] += l_chi.detach()
            tracker["eng_loss"] += l_eng.detach()
            tracker["corr"] += nc_t.detach()
            tracker["total"] += nt_t.detach()
            tracker["chi_corr"] += c_nc_t.detach()
            tracker["chi_total"] += c_nt_t.detach()
            tracker["eng_corr"] += e_nc_t.detach()
            tracker["eng_total"] += e_nt_t.detach()
            tracker["eng_trunc_count"] += e_trunc_c.detach()
            tracker["eng_trunc_total"] += e_trunc_t.detach()

    if is_xla:
        xm.rendezvous("validate_metrics")
        t_loss = torch.tensor(tracker["loss"], device=device)
        t_chi_loss = torch.tensor(tracker["chi_loss"], device=device)
        t_eng_loss = torch.tensor(tracker["eng_loss"], device=device)
        t_corr = torch.tensor(tracker["corr"], device=device)
        t_tot = torch.tensor(tracker["total"], device=device)
        tc_corr = torch.tensor(tracker["chi_corr"], device=device)
        tc_tot = torch.tensor(tracker["chi_total"], device=device)
        te_corr = torch.tensor(tracker["eng_corr"], device=device)
        te_tot = torch.tensor(tracker["eng_total"], device=device)
        te_trunc_c = torch.tensor(tracker["eng_trunc_count"], device=device)
        te_trunc_t = torch.tensor(tracker["eng_trunc_total"], device=device)
        t_step = torch.tensor(step_idx, device=device)

        t_loss = xm.all_reduce(xm.REDUCE_SUM, t_loss)
        t_chi_loss = xm.all_reduce(xm.REDUCE_SUM, t_chi_loss)
        t_eng_loss = xm.all_reduce(xm.REDUCE_SUM, t_eng_loss)
        t_corr = xm.all_reduce(xm.REDUCE_SUM, t_corr)
        t_tot = xm.all_reduce(xm.REDUCE_SUM, t_tot)
        tc_corr = xm.all_reduce(xm.REDUCE_SUM, tc_corr)
        tc_tot = xm.all_reduce(xm.REDUCE_SUM, tc_tot)
        te_corr = xm.all_reduce(xm.REDUCE_SUM, te_corr)
        te_tot = xm.all_reduce(xm.REDUCE_SUM, te_tot)
        te_trunc_c = xm.all_reduce(xm.REDUCE_SUM, te_trunc_c)
        te_trunc_t = xm.all_reduce(xm.REDUCE_SUM, te_trunc_t)
        t_step = xm.all_reduce(xm.REDUCE_SUM, t_step)

        tracker["loss"] = t_loss.item()
        tracker["chi_loss"] = t_chi_loss.item()
        tracker["eng_loss"] = t_eng_loss.item()
        tracker["corr"] = t_corr.item()
        tracker["total"] = t_tot.item()
        tracker["chi_corr"] = tc_corr.item()
        tracker["chi_total"] = tc_tot.item()
        tracker["eng_corr"] = te_corr.item()
        tracker["eng_total"] = te_tot.item()
        tracker["eng_trunc_count"] = te_trunc_c.item()
        tracker["eng_trunc_total"] = te_trunc_t.item()
        step_idx = int(t_step.item())

    avg_loss = tracker["loss"] / float(max(1, step_idx))
    avg_chi_loss = tracker["chi_loss"] / float(max(1, step_idx))
    avg_eng_loss = tracker["eng_loss"] / float(max(1, step_idx))
    gloss_acc = (tracker["corr"] / max(1.0, tracker["total"])) * 100.0
    chicago_acc = (tracker["chi_corr"] / max(1.0, tracker["chi_total"])) * 100.0
    english_acc = (tracker["eng_corr"] / max(1.0, tracker["eng_total"])) * 100.0
    english_trunc_rate = (
        tracker["eng_trunc_count"] / max(1.0, tracker["eng_trunc_total"])
    ) * 100.0

    if is_master:
        print(
            f"[Validation Epoch {epoch}] GlossLoss: {avg_loss:.4f} | ChicagoLoss: {avg_chi_loss:.4f} | EnglishLoss: {avg_eng_loss:.4f} | GlossAcc: {gloss_acc:.2f}% | ChiAcc: {chicago_acc:.2f}% | EngAcc: {english_acc:.2f}% | EngTrunc: {english_trunc_rate:.2f}%",
            flush=True,
        )

    return {
        "loss": avg_loss,
        "gloss_acc": gloss_acc,
        "chicago_acc": chicago_acc,
        "english_acc": english_acc,
    }


def _tpu_worker_fn(rank, args):
    if IS_TPU:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl
    try:
        device = (
            xm.xla_device()
            if IS_TPU
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    except Exception as e:
        print(f"FAILED TO INITIALIZE TPU OR GET DEVICE: {e}", flush=True)
        time.sleep(2)
        os._exit(1)
    is_master = (rank == 0) if IS_TPU else True

    data_dir = Path(args.data_dir)
    vocab_path = data_dir / "vocabulary_mapping_train.json"
    if vocab_path.exists():
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
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
        except Exception as _e:
            if is_master:
                print(
                    f"[!] Warning: Failed to load mapping from '{vocab_path}': {_e}",
                    flush=True,
                )

    if not label_to_idx:
        raise ValueError(
            f"Failed to load vocabulary mapping from {vocab_path}. Will not hallucinate fictional classes."
        )

    vocab = GlossVocabulary(label_to_idx=label_to_idx)

    if _XLA_AVAILABLE:
        import torch_xla.runtime as xr

        world_size = xr.world_size()
        assert rank < world_size
        if is_master:
            print(f"TPU world size: {world_size}", flush=True)
    else:
        world_size = 1

    train_loader = create_dataloader(
        dataset_dir=data_dir,
        split="train",
        batch_size=args.batch_size,
        max_len=args.max_len,
        worker_idx=rank if _XLA_AVAILABLE else 0,
        num_workers=world_size,
        num_dataloader_workers=args.num_dataloader_workers,
        shuffle=True,
        augment=True,
    )

    val_loader = create_dataloader(
        dataset_dir=data_dir,
        split="val",
        batch_size=args.batch_size,
        max_len=args.max_len,
        worker_idx=rank if _XLA_AVAILABLE else 0,
        num_workers=world_size,
        num_dataloader_workers=args.num_dataloader_workers,
        shuffle=False,
        augment=False,
    )

    class_weights_tensor = None
    try:
        raw_ds = getattr(train_loader, "dataset", None)
        c_counts = getattr(raw_ds, "class_counts", None) if raw_ds else None
        if c_counts:
            # Aggregate counts globally across TPUs using RAW class space
            raw_counts = torch.zeros(len(vocab), dtype=torch.float32, device=device)
            for raw_idx, cnt in c_counts.items():
                if raw_idx < len(vocab):
                    raw_counts[raw_idx] = float(cnt)

            if _XLA_AVAILABLE:
                import torch_xla.core.xla_model as xm
                raw_counts = xm.all_reduce(xm.REDUCE_SUM, raw_counts)

            w_vec = torch.ones(len(vocab), dtype=torch.float32, device=device)
            max_c = max(1.0, float(raw_counts.max().item()))

            # Map raw counts into the offset model token space
            for tok_id in range(GlossVocabulary.OFFSET, len(vocab)):
                # The user's exact audit recommendation: cnt = local_counts[tok_id - GlossVocabulary.OFFSET]
                cnt = raw_counts[tok_id - GlossVocabulary.OFFSET].item()
                if cnt > 0:
                    w_vec[tok_id] = min(10.0, max(1.0, (max_c / cnt) ** 0.35))
            class_weights_tensor = w_vec
            if is_master:
                print(
                    f"[INFO] Class weighting: ENABLED (calculated from raw dataset class_counts)",
                    flush=True,
                )

            # Verifying dataset gloss_seq offset
            assert (
                tok_id >= GlossVocabulary.OFFSET
            ), "class_counts contains raw IDs, we properly added the OFFSET."
    except Exception as e:
        if is_master:
            print(f"[WARN] Class weighting: DISABLED due to exception: {e}", flush=True)
        class_weights_tensor = torch.ones(
            len(vocab), dtype=torch.float32, device=device
        )

    asl_lex_csv = (
        Path(args.asl_lex_csv)
        if hasattr(args, "asl_lex_csv") and args.asl_lex_csv
        else (data_dir / "signdata.csv")
    )

    eng_vsize = (
        len(train_loader.dataset.english_vocab)
        if hasattr(train_loader.dataset, "english_vocab")
        else 20005
    )

    import hashlib

    eng_hash = (
        int(
            hashlib.md5(
                str(
                    list(train_loader.dataset.english_vocab.token_to_id.items())
                ).encode()
            ).hexdigest()[:12],
            16,
        )
        if hasattr(train_loader.dataset, "english_vocab")
        else 0
    )
    if _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm

        local_hash = torch.tensor(eng_hash, dtype=torch.float64, device=device)
        global_min = int(
            xm.mesh_reduce("english_vocab_min", local_hash, lambda xs: min(xs)).item()
        )
        global_max = int(
            xm.mesh_reduce("english_vocab_max", local_hash, lambda xs: max(xs)).item()
        )

        if global_min != global_max:
            raise RuntimeError(
                f"English vocabulary differs across TPU ranks: "
                f"min_hash={global_min}, max_hash={global_max}"
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
        max_enc_len=args.max_len,
        english_vocab_size=eng_vsize,
        label_to_idx=label_to_idx,
        csv_path=asl_lex_csv if asl_lex_csv.exists() else None,
    ).to(device)

    if _XLA_AVAILABLE:
        xm.broadcast_master_param(model)

    if getattr(args, "compile", False) and hasattr(torch, "compile"):
        if is_master:
            print(
                "[*] JIT Compiling model with PyTorch Inductor (torch.compile)...",
                flush=True,
            )
        try:
            model = torch.compile(model)
        except Exception as _e:
            if is_master:
                print(f"[!] Warning: torch.compile fallback: {_e}", flush=True)

    loss_wrapper = HomoscedasticLossWrapper().to(device)
    if _XLA_AVAILABLE:
        xm.broadcast_master_param(loss_wrapper)

    supcon_fn = SupervisedContrastiveLoss().to(device)

    global_min_batches = len(train_loader)
    if _XLA_AVAILABLE:
        global_min_batches = int(
            xm.mesh_reduce("global_min_batches", len(train_loader), lambda x: min(x))
        )

    optimizer = torch.optim.AdamW(
        _get_optimizer_groups(model, loss_wrapper, args.weight_decay),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=max(
            1, math.ceil(global_min_batches / max(1, args.accum_steps))
        ),
        pct_start=0.1,
        div_factor=10.0,
        final_div_factor=100.0,
    )

    scaler = None
    if args.precision == "float16" and "cuda" in str(device).lower():
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    if hasattr(args, "resume") and args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu")

        missing, unexpected = model.load_state_dict(
            ckpt["model_state_dict"], strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint mismatch. Missing: {missing}, Unexpected: {unexpected}"
            )

        if "loss_wrapper_state_dict" in ckpt:
            missing_lw, unexpected_lw = loss_wrapper.load_state_dict(
                ckpt["loss_wrapper_state_dict"], strict=False
            )
            if missing_lw or unexpected_lw:
                raise RuntimeError(
                    f"Loss wrapper mismatch. Missing: {missing_lw}, Unexpected: {unexpected_lw}"
                )

        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as e:
                raise RuntimeError(f"Failed to load optimizer state: {e}")

        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                raise RuntimeError(f"Failed to load scheduler state: {e}")

        if (
            "scaler_state_dict" in ckpt
            and ckpt["scaler_state_dict"] is not None
            and "scaler" in locals()
            and scaler is not None
        ):
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        if "rng_state_torch" in ckpt:
            torch.set_rng_state(ckpt["rng_state_torch"])
        if "rng_state_numpy" in ckpt:
            np.random.set_state(ckpt["rng_state_numpy"])
        if "rng_state_random" in ckpt:
            import random

            random.setstate(ckpt["rng_state_random"])

        start_epoch = ckpt.get("epoch", 0) + 1

    save_dir = Path(args.save_dir)
    if is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 70, flush=True)
        print(
            f"       STARTING TPU MULTI-TASK FOUNDATION MODEL TRAINING ({args.epochs} EPOCHS)",
            flush=True,
        )
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"       Model: {args.num_layers} layers | d_model={args.d_model} | {total_params / 1e6:.1f}M params",
            flush=True,
        )
        print("=" * 70, flush=True)

    ema = ModelEMA(model)
    if "ema_state_dict" in locals().get("ckpt", {}):
        if ckpt["ema_state_dict"] is not None:
            for k, v in ckpt["ema_state_dict"].items():
                if k in ema.shadow:
                    ema.shadow[k].copy_(v.to(ema.shadow[k].device))
            if is_master:
                print("[+] Restored EMA state from checkpoint", flush=True)



    try:
        for epoch in range(start_epoch, args.epochs + 1):

            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

            # --- ADD THIS TO RAMP UP NOISE CURRICULUM ---
            if hasattr(train_loader.dataset, "set_noise_level"):
                # Ramps noise from 0.0 to 1.0 linearly over the epochs
                train_loader.dataset.set_noise_level(epoch / max(1, args.epochs))
            # --------------------------------------------

            train_metrics = train_epoch_tpu(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_wrapper=loss_wrapper,
                ema=ema,
                supcon_fn=supcon_fn,
                device=device,
                scaler=scaler,
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

            # --- VALIDATION LOOP ---
            raw_m = model.module if hasattr(model, "module") else model
            if ema is not None:
                ema.apply_shadow(raw_m)

            val_metrics = validate_epoch_tpu(
                model=raw_m,
                loader=val_loader,
                loss_wrapper=loss_wrapper,
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
                class_weights=class_weights_tensor,
            )

            if is_master:
                print(
                    f"[Validation Epoch {epoch}] SeqLoss: {val_metrics['loss']:.4f} | TokenAcc: {val_metrics['gloss_acc']:.2f}%",
                    flush=True,
                )

            if ema is not None:
                ema.restore(raw_m)
                if _XLA_AVAILABLE:
                    import torch_xla.core.xla_model as xm

                    xm.mark_step()

            ckpt_path = save_dir / f"asl_model_epoch_{epoch}.pt"
            import random

            cpu_state = None
            if is_master:
                cpu_state = {
                    "epoch": epoch,
                    "model_state_dict": raw_m.state_dict(),
                    "ema_state_dict": (
                        {k: v.cpu() for k, v in ema.shadow.items()}
                        if ema is not None
                        else None
                    ),
                    "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": (
                        scheduler.state_dict() if scheduler is not None else None
                    ),
                    "scaler_state_dict": (
                        scaler.state_dict()
                        if "scaler" in locals() and scaler is not None
                        else None
                    ),
                    "rng_state_torch": torch.get_rng_state(),
                    "rng_state_numpy": np.random.get_state(),
                    "rng_state_random": random.getstate(),
                }

            if _XLA_AVAILABLE:
                import torch_xla.core.xla_model as xm

                xm.save(cpu_state, str(ckpt_path), master_only=True)
            else:
                if is_master:
                    torch.save(cpu_state, str(ckpt_path))

            # Explicit garbage collection of the massive optimizer state dict
            if cpu_state is not None:
                del cpu_state
            import gc

            gc.collect()

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
        default=r"E:\datasets\asl_dataset\asl_preprocessed_phase1",
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
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-dataloader-workers", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable PyTorch 2.0 torch.compile JIT acceleration",
    )
    parser.add_argument("--save-dir", type=str, default="/tmp/checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--asl-lex-csv", type=str, default="/home/binhhanh409/signdata.csv"
    )
    args = parser.parse_args()

    if args.precision == "bfloat16":
        os.environ["XLA_USE_BF16"] = "1"
    elif args.precision == "float16":
        os.environ["XLA_USE_F16"] = "1"
    else:
        os.environ.pop("XLA_USE_BF16", None)
        os.environ.pop("XLA_USE_F16", None)

    if IS_TPU:
        import torch_xla.distributed.xla_multiprocessing as xmp

        if "LOCAL_RANK" in os.environ:
            _tpu_worker_fn(int(os.environ["LOCAL_RANK"]), args)
        else:
            xmp.spawn(_tpu_worker_fn, args=(args,), nprocs=None, start_method="fork")
    else:
        _tpu_worker_fn(0, args)


if __name__ == "__main__":
    main()
