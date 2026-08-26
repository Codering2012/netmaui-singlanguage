#!/usr/bin/env python3

"""
================================================================================
MONOLITHIC ALL-IN-ONE TPU/GPU ASL FOUNDATION MODEL — SENTENCE RECONSTRUCTION
Encoder: MobileConformer (8L × dim_d=320, nhead=8, ffn=1280) — ~17.4M parameters
Decoder: ASLTransformerDecoder (8L × dim_d=320, GQA 8Q/2KV, RoPE, ffn=1280) — ~12.9M parameters
Total:   ~31.0M parameters (High Efficiency & SOTA Accuracy via Extended Compute)

Task: Continuous Sign Language Understanding & Gloss Sentence Reconstruction
================================================================================
"""

import argparse
import itertools
import functools
import csv
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

# Cap PyTorch XLA PJRT C++ driver host memory reservation to prevent allocating 90% (297GB+) of system RAM
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.15"
os.environ["PJRT_ALLOCATOR_FRACTION"] = "0.15"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.15"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "100000"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "131072"

# Set balanced Python Garbage Collection threshold to prevent CPU micro-pauses during fast training loops
gc.set_threshold(50000, 500, 50)

# --- OpenXLA Hardware Fusion & Parallelism Flags ---
# Must be set before importing torch / torch_xla
if "LIBTPU_INIT_ARGS" not in os.environ:
    os.environ["LIBTPU_INIT_ARGS"] = (
        "--xla_tpu_enable_async_collective_fusion=true "
        "--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true "
        "--xla_tpu_enable_flash_attention=true "
        "--xla_enable_async_all_gather=true"
    )
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    import torch_xla.distributed.parallel_loader as pl
except ImportError:
    pl = None

from dataset import (
    phase1_collate_fn,
    phase2_collate_fn,
    create_dataloader,
    normalize_vocabulary,
    EnglishVocabulary,
    GlossVocabulary,
    ASLStreamedDataset,
    KDWDDataset,
    ASLGPC12Dataset,
)

print(
    "[DEBUG 1/8] Importing standard libraries & setting environment variables...",
    flush=True,
)

# [NEW] Persistent HLO caching to permanently bypass 13-minute XLA compilations

os.environ["XLA_PERSISTENT_CACHE_PATH"] = "./xla_cache"
os.makedirs("./xla_cache", exist_ok=True)


if "--tpu" in sys.argv:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

# XLA Native BFloat16 Compilation Flags (Preserving FP32 Precision for Sensitive Operations)
os.environ["XLA_USE_BF16"] = "1"
os.environ.pop("XLA_DOWNCAST_BF16", None)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Force Local PJRT mode to avoid gRPC proxy concurrency limit and fork deadlocks
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)
# PJRT_DEVICE initialization deferred to main() after argparse

try:
    import importlib.util
    _XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
except Exception:
    _XLA_AVAILABLE = False
IS_TPU = False


def get_xla_world_size() -> int:
    """Provides functionality for get_xla_world_size."""

    if IS_TPU:
        try:
            import torch_xla.runtime as xr

            return xr.world_size()
        except Exception:
            try:
                import torch_xla.core.xla_model as xm

                return getattr(
                    xm, "get_world_size", getattr(xm, "xrt_world_size", lambda: 1)
                )()
            except Exception:
                pass
    return 1


train_dir = Path(__file__).resolve().parent
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))

print("[DEBUG 2/8] Importing dataset module & vocabulary handlers...", flush=True)


def _distributed_normalize(
    local_sum: torch.Tensor, local_weight: torch.Tensor
) -> torch.Tensor:
    """Computes weighted loss mean globally across TPU ranks."""
    if IS_TPU and _XLA_AVAILABLE:
        import torch_xla.core.xla_model as xm

        # NEVER all_reduce local_sum inside the forward pass if we will backpropagate through it!
        # PyTorch XLA's autograd kernel for all_reduce crashes with SIGSEGV.
        # Instead, we all_reduce the detached weights, and scale the local sum.
        global_weight = xm.all_reduce(xm.REDUCE_SUM, local_weight.detach().clone())
        world_size = get_xla_world_size()

        # xm.optimizer_step() averages gradients across TPUs.
        # So if we multiply by world_size here, the final gradient is exactly sum(L_i) / global_weight
        normed = (local_sum * world_size) / global_weight.clamp_min(1e-8)
    else:
        normed = local_sum / local_weight.clamp_min(1e-8)

    return torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0) * (
        local_weight > 0
    ).to(normed.dtype)


def _safe_torch_device(dev_str: Union[str, torch.device]) -> torch.device:
    """Internal helper method _safe_torch_device."""

    if isinstance(dev_str, torch.device):
        return dev_str
    dev_s = str(dev_str).lower()
    if IS_TPU and "xla" in dev_s:
        try:
            import torch_xla

            return torch_xla.device(dev_str)
        except Exception:
            pass
    return torch.device(dev_str)


# ==============================================================================
# 1. LANDMARK AUGMENTER (REAL-WORLD CAMERA NOISE & PHYSIOLOGICAL STALLING)
# ==============================================================================

# ==============================================================================
# 2. GLOSS VOCABULARY — Sequence Vocabulary with Special Tokens
# ==============================================================================


# ==============================================================================
# 3. RMSNorm & SwiGLUFFN
# ==============================================================================


class RMSNorm(nn.Module):
    """Provides functionality for RMSNorm."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        """Initializes the module component."""

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for this module."""

        var = input_x.float().pow(2).mean(-1, keepdim=True)
        return (
            input_x
            * torch.rsqrt(var + self.eps).to(input_x.dtype)
            * self.weight.to(input_x.dtype)
        )


class SwiGLUFFN(nn.Module):
    """Provides functionality for SwiGLUFFN."""

    def __init__(self, d_model: int, dim_feedforward: int, num_layers: int = 8):
        """Initializes the module component."""

        super().__init__()
        # Align hidden dimension to 128 for TPU v5e MXU systolic arrays
        hidden = (int(dim_feedforward * 2 / 3) + 127) // 128 * 128
        self.w_gate, self.w_up, self.w_down = (
            nn.Linear(d_model, hidden, bias=False),
            nn.Linear(d_model, hidden, bias=False),
            nn.Linear(hidden, d_model, bias=False),
        )
        nn.init.normal_(self.w_gate.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.w_up.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.w_down.weight, std=1.0 / math.sqrt(2.0 * num_layers * d_model))

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for this module."""

        return self.w_down(F.silu(self.w_gate(input_x)) * self.w_up(input_x))


# ==============================================================================
# 4. RICH ASL-LEX MULTI-ATTRIBUTE EMBEDDING TABLE
# ==============================================================================


class RichASLLexEmbeddingTable(nn.Module):
    """Projects raw features into the embedding space using RichASLLexEmbeddingTable."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        csv_path: Optional[Union[str, Path]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        """Initializes the module component."""

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
                        if not (min(label_to_idx.values()) >= 4):
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
        # Removed .any() validation check to prevent XLA device-to-host syncs in the forward pass.
        """Forward pass for this module."""

        ids = token_ids
        attr_ids = self.attr_idx_matrix[ids]
        scalars = F.embedding(ids, self.attr_scalars)

        e_lc = self.emb_lexclass(attr_ids[:, :, 0])
        e_st = self.emb_signtype(attr_ids[:, :, 1])
        e_hs = self.emb_handshape(attr_ids[:, :, 2])
        e_loc = self.emb_location(attr_ids[:, :, 3])
        e_cat = self.emb_category(attr_ids[:, :, 4])

        raw_attrs = torch.cat([e_lc, e_st, e_hs, e_loc, e_cat, scalars], dim=-1)
        valid_lex_mask = (token_ids != 0).unsqueeze(-1).to(raw_attrs.dtype)
        return self.attr_proj(raw_attrs) * valid_lex_mask


# ==============================================================================
# 5. TOKEN MERGING BLOCK (ToMe)
# ==============================================================================


def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = (torch.rand(shape, device=x.device) < keep_prob).to(x.dtype)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class TemporalStridedPool(nn.Module):
    def __init__(self, is_causal=False, **kwargs):
        super().__init__()
        self.is_causal = is_causal
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2, ceil_mode=True)

    def forward(self, hidden_h, mask=None, **kwargs):
        r = getattr(self, "r", -1)
        if r == 0:
            return (
                hidden_h,
                mask,
                {
                    "T_orig": hidden_h.shape[1],
                    "sorted_routing": None,
                    "mlm_out": kwargs.get("mlm_mask", None),
                    "frame_indices": kwargs.get("frame_indices", None),
                    "token_sizes": kwargs.get("token_sizes", None),
                },
            )

        B, T, D = hidden_h.shape
        pad_len = T % 2

        if pad_len > 0:
            if self.is_causal:
                # Replicate-pad the first token to preserve its magnitude during averaging
                hidden_padded = torch.cat([hidden_h[:, :1, :], hidden_h], dim=1)
            else:
                hidden_padded = torch.cat([hidden_h, hidden_h[:, -1:, :]], dim=1)
        else:
            hidden_padded = hidden_h

        hidden_reshaped = hidden_padded.view(B, -1, 2, D)

        if mask is not None:
            if pad_len > 0:
                if self.is_causal:
                    mask_padded = torch.nn.functional.pad(mask, (pad_len, 0), value=False)
                else:
                    mask_padded = torch.nn.functional.pad(mask, (0, pad_len), value=False)
            else:
                mask_padded = mask

            mask_reshaped = mask_padded.view(B, -1, 2)

            # Mask out invalid frames before summing
            hidden_sum = (hidden_reshaped * mask_reshaped.unsqueeze(-1)).sum(dim=2)
            valid_count = mask_reshaped.sum(dim=2).unsqueeze(-1).clamp(min=1)

            hidden_h = hidden_sum / valid_count
            mask = mask_reshaped.any(dim=2)
        else:
            hidden_h = hidden_reshaped.mean(dim=2)
            mask = None

        fi = kwargs.get("frame_indices", None)
        if fi is not None:
            if pad_len > 0:
                if self.is_causal:
                    fi_padded = torch.nn.functional.pad(fi, (pad_len, 0))
                else:
                    fi_padded = torch.nn.functional.pad(fi, (0, pad_len))
            else:
                fi_padded = fi
            fi_reshaped = fi_padded.view(B, -1, 2)
            if mask is not None:
                fi_sum = (fi_reshaped * mask_reshaped).sum(dim=2)
                valid_count = mask_reshaped.sum(dim=2).clamp(min=1)
                fi = fi_sum / valid_count
            else:
                fi = fi_reshaped.float().mean(dim=2)

        token_sizes = kwargs.get("token_sizes", None)
        if token_sizes is not None:
            if token_sizes.ndim == 3:
                token_sizes = token_sizes.squeeze(-1)
            if pad_len > 0:
                if self.is_causal:
                    ts_padded = torch.nn.functional.pad(token_sizes, (pad_len, 0))
                else:
                    ts_padded = torch.nn.functional.pad(token_sizes, (0, pad_len))
            else:
                ts_padded = token_sizes
            token_sizes = ts_padded.view(B, -1, 2).sum(dim=2)

        mlm_in = kwargs.get("mlm_mask", None)
        if mlm_in is not None:
            if pad_len > 0:
                if self.is_causal:
                    mlm_p = torch.nn.functional.pad(mlm_in, (pad_len, 0), value=False)
                else:
                    mlm_p = torch.nn.functional.pad(mlm_in, (0, pad_len), value=False)
            else:
                mlm_p = mlm_in
            mlm_out = mlm_p.view(B, -1, 2).any(dim=2)
        else:
            mlm_out = None

        return (
            hidden_h,
            mask,
            {
                "T_orig": T,
                "sorted_routing": None,
                "mlm_out": mlm_out,
                "frame_indices": fi,
                "token_sizes": token_sizes,
            },
        )


class DropPath(nn.Module):
    """Provides functionality for DropPath."""

    def __init__(self, drop_prob: float = 0.0):
        """Initializes the module component."""

        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for this module."""

        is_training = self.training
        return drop_path(input_x, self.drop_prob, is_training)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        # Claim 92: Lower base for high-frequency coordinate tracking
        inv_freq = 1.0 / (500.0 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len

        # Pre-cache maximum possible sequence length to avoid XLA graph breaks
        t = torch.arange(max_len).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, q, k, frame_indices=None):

        def apply_rotary_emb(x, cos, sin):
            # x: [B, nhead, seq_len, head_dim]
            # cos, sin: [1, 1, max_len, dim]
            x_rot = torch.cat(
                [-x[..., x.shape[-1] // 2 :], x[..., : x.shape[-1] // 2]], dim=-1
            )
            
            if frame_indices is not None:
                # frame_indices: [B, seq_len]
                batch_sz = x.shape[0]
                seq_len = x.shape[2]
                
                # We need to index cos/sin with shape [B, 1, seq_len, dim]
                # Expand cos/sin to have batch dimension, then gather along sequence len
                # cos shape: [1, 1, max_len, dim] -> [1, max_len, dim]
                cos_flat = cos.squeeze(0).squeeze(0)  # [max_len, dim]
                sin_flat = sin.squeeze(0).squeeze(0)
                
                # Clamping just to be safe
                fi = frame_indices.long().clamp(0, self.max_len - 1) # [B, seq_len]
                
                cos_idx = cos_flat[fi].unsqueeze(1) # [B, 1, seq_len, dim]
                sin_idx = sin_flat[fi].unsqueeze(1)
                
                cos_dtype = cos_idx.to(x.dtype)
                sin_dtype = sin_idx.to(x.dtype)
            else:
                cos_dtype = cos[:, :, : x.shape[2], :].to(x.dtype)
                sin_dtype = sin[:, :, : x.shape[2], :].to(x.dtype)
                
            return (x * cos_dtype) + (x_rot * sin_dtype)

        return apply_rotary_emb(q, self.cos_cached, self.sin_cached), apply_rotary_emb(
            k, self.cos_cached, self.sin_cached
        )


class GroupedQueryEncoderAttention(nn.Module):
    """Implements the GroupedQueryEncoderAttention architecture for the sequence modeling pipeline."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        max_len: int = 512,
        dropout_p: float = 0.1,
        is_causal: bool = False,
        lookahead: int = 0,
    ):
        """Initializes the module component."""

        super().__init__()
        assert nhead % kv_heads == 0
        self.nhead, self.kv_heads, self.groups, self.head_dim = (
            nhead,
            kv_heads,
            nhead // kv_heads,
            d_model // nhead,
        )
        self.scale = 1.0 / np.sqrt(self.head_dim)
        self.is_causal = is_causal
        self.lookahead = lookahead

        # DeepSeek V3 MLA (Multi-Head Latent Attention) Compression
        self.latent_dim = d_model // 4
        self.kv_latent_proj = nn.Linear(d_model, self.latent_dim, bias=False)
        self.kv_latent_norm = RMSNorm(self.latent_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(
            self.latent_dim, 2 * kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(d_model)

        # RoPE only applies to half of the head dimension (head_dim // 2)
        self.rope = RotaryPositionalEncoding(self.head_dim // 2, max_len=max_len)
        self.dropout_p = dropout_p

    def forward(
        self,
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for this module."""

        batch_sz, seq_len, _ = input_x.shape
        q_in = self.q_norm(input_x)

        # DeepSeek V3 MLA Latent Compression
        kv_latent = self.kv_latent_proj(input_x)
        kv_latent = self.kv_latent_norm(kv_latent)

        query_q_lower = (
            self.q_proj(q_in)
            .view(batch_sz, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        kv = self.kv_proj(kv_latent)
        key_k_lower, val_v = torch.split(kv, kv.size(-1) // 2, dim=-1)
        key_k_lower = key_k_lower.view(
            batch_sz, seq_len, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        val_v = val_v.view(batch_sz, seq_len, self.kv_heads, self.head_dim).transpose(
            1, 2
        )
        rope_dim = self.head_dim // 2
        q_rope, q_nop = torch.split(query_q_lower, [rope_dim, query_q_lower.size(-1) - rope_dim], dim=-1) # Prevent XLA graph slicing detachment (Claim 71)
        k_rope, k_nop = torch.split(key_k_lower, [rope_dim, key_k_lower.size(-1) - rope_dim], dim=-1)
        q_rope, k_rope = self.rope(q_rope, k_rope, frame_indices=frame_indices)
        query_q_lower = torch.cat([q_rope, q_nop], dim=-1)
        key_k_lower = torch.cat([k_rope, k_nop], dim=-1)

        if attn_mask is not None and attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        if key_padding_mask is not None:
            # PyTorch SDPA expects True = attend (valid token), False = ignore (pad token).
            kpm = (~key_padding_mask.bool()).view(batch_sz, 1, 1, seq_len)
            if attn_mask is not None:
                attn_mask = attn_mask & kpm
            else:
                attn_mask = kpm

        sdpa_is_causal = self.is_causal
        current_lookahead = self.lookahead
        
        # Removed dynamic lookahead jitter because varying 'diagonal' recompiles the XLA graph
        if self.is_causal:
            if current_lookahead == 0 and attn_mask is None and key_padding_mask is None:
                # Highly-efficient FlashAttention path
                sdpa_is_causal = True
                attn_mask = None
            else:
                sdpa_is_causal = False
                # Create block-causal mask allowing `current_lookahead` future frames
                b_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_x.device)
                b_mask = torch.tril(b_mask, diagonal=current_lookahead)
                if attn_mask is not None:
                    attn_mask = attn_mask & b_mask.unsqueeze(0).unsqueeze(0)
                else:
                    attn_mask = b_mask.unsqueeze(0).unsqueeze(0)

        if self.groups > 1:
            key_k_lower = key_k_lower.unsqueeze(2).expand(batch_sz, self.kv_heads, self.groups, seq_len, self.head_dim).reshape(batch_sz, self.nhead, seq_len, self.head_dim)
            val_v = val_v.unsqueeze(2).expand(batch_sz, self.kv_heads, self.groups, seq_len, self.head_dim).reshape(batch_sz, self.nhead, seq_len, self.head_dim)

        out = F.scaled_dot_product_attention(
            query_q_lower,
            key_k_lower,
            val_v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=sdpa_is_causal,
            scale=self.scale,
        )
        out = self.out_proj(out.transpose(1, 2).reshape(batch_sz, seq_len, -1))
        return out


class Swin1DAttention(nn.Module):
    """Wraps an attention module to compute 1D Shifted Window Attention (Swin-1D)."""
    
    def __init__(self, mha_module: nn.Module, window_size: int = 128, shift_size: int = 0):
        super().__init__()
        self.mha = mha_module
        self.window_size = window_size
        # Disable shifting for causal modules to prevent future leakage from torch.roll
        self.shift_size = shift_size if not getattr(mha_module, "is_causal", False) else 0

    def forward(self, input_x, key_padding_mask=None, frame_indices=None):
        B, L, C = input_x.shape
        
        # Pad to multiple of window_size
        pad_l = (self.window_size - L % self.window_size) % self.window_size
        if pad_l > 0:
            input_x = F.pad(input_x, (0, 0, 0, pad_l))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, pad_l), value=True)
            else:
                key_padding_mask = torch.zeros((B, L), dtype=torch.bool, device=input_x.device)
                key_padding_mask = F.pad(key_padding_mask, (0, pad_l), value=True)

            if frame_indices is not None:
                # Pad frame_indices with max index to maintain monotonicity
                fi_pad_val = frame_indices.max() if L > 0 else 0
                frame_indices = F.pad(frame_indices, (0, pad_l), value=fi_pad_val)
                
        # Shift
        if self.shift_size > 0:
            shifted_x = torch.roll(input_x, shifts=-self.shift_size, dims=1)
            if key_padding_mask is not None:
                shifted_mask = torch.roll(key_padding_mask, shifts=-self.shift_size, dims=1)
            else:
                shifted_mask = None

            if frame_indices is not None:
                shifted_fi = torch.roll(frame_indices, shifts=-self.shift_size, dims=1)
            else:
                shifted_fi = None
        else:
            shifted_x = input_x
            shifted_mask = key_padding_mask
            shifted_fi = frame_indices

        # Partition windows
        num_windows = shifted_x.shape[1] // self.window_size
        x_windows = shifted_x.view(B * num_windows, self.window_size, C)
        
        if shifted_mask is not None:
            mask_windows = shifted_mask.view(B * num_windows, self.window_size)
        else:
            mask_windows = None

        if shifted_fi is not None:
            fi_windows = shifted_fi.view(B * num_windows, self.window_size)
        else:
            fi_windows = None

        # Masking for shifted windows to prevent cross-boundary attention
        attn_mask = None
        if self.shift_size > 0:
            L_feat = shifted_x.shape[1]
            s0 = max(0, L_feat - self.window_size)
            s1 = self.window_size - self.shift_size
            s2 = self.shift_size
            img_mask = torch.cat([
                torch.zeros((1, s0, 1), device=input_x.device),
                torch.ones((1, s1, 1), device=input_x.device),
                torch.full((1, s2, 1), 2.0, device=input_x.device),
            ], dim=1)
                
            mask_windows_attn = img_mask.view(1, num_windows, self.window_size, 1).view(num_windows, self.window_size)
            attn_mask = mask_windows_attn.unsqueeze(1) - mask_windows_attn.unsqueeze(2)
            attn_mask = (attn_mask == 0) # True means do attend
            attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_windows, self.window_size, self.window_size)

        attn_windows = self.mha(x_windows, key_padding_mask=mask_windows, frame_indices=fi_windows, attn_mask=attn_mask)
        
        # Reverse windows
        shifted_x = attn_windows.view(B, num_windows * self.window_size, C)
        
        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x
            
        # Unpad
        if pad_l > 0:
            x = x[:, :L, :]
            
        return x


class SpatialTemporalSE(nn.Module):
    """Provides functionality for SpatialTemporalSE."""

    def __init__(self, d_model: int, reduction: int = 4, is_causal: bool = False):
        """Initializes the module component."""

        super().__init__()
        self.is_causal = is_causal
        self.c_se = nn.Sequential(
            nn.Linear(d_model, d_model // reduction, bias=False),
            nn.GELU(),
            nn.Linear(d_model // reduction, d_model, bias=False),
            nn.Sigmoid(),
        )
        self.s_se = nn.Sequential(nn.Linear(d_model, 1, bias=False), nn.Sigmoid())

    def forward(
        self, input_x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for this module."""

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask.bool()).unsqueeze(-1).to(input_x.dtype)
        else:
            valid_mask = (input_x.abs().sum(dim=-1, keepdim=True) > 1e-5).to(
                input_x.dtype
            )

        if self.is_causal:
            cum_sum = (input_x * valid_mask).cumsum(dim=1)
            cum_count = valid_mask.cumsum(dim=1).clamp(min=1.0)
            mean_x = cum_sum / cum_count
            c_se_out = self.c_se(mean_x)
        else:
            mean_x = (input_x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(
                min=1.0
            )
            c_se_out = self.c_se(mean_x).unsqueeze(1)
            
        return input_x * c_se_out * self.s_se(input_x)


class ConvNeXtTemporalBlock(nn.Module):
    """Provides functionality for ConvNeXtTemporalBlock."""

    def __init__(self, channels: int, expansion: int = 2, is_causal: bool = False):
        """Initializes the module component."""

        super().__init__()
        self.is_causal = is_causal
        self.dw_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=7,
            padding=0,
            groups=channels,
        )
        self.norm = RMSNorm(channels)
        self.pw_conv1, self.pw_conv2 = nn.Linear(
            channels, channels * expansion
        ), nn.Linear(channels * expansion, channels)
        self.act, self.se = nn.GELU(), SpatialTemporalSE(channels, is_causal=is_causal)

    def forward(
        self,
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for this module."""

        valid_mask = (~key_padding_mask.bool()).unsqueeze(-1).to(input_x.dtype) if key_padding_mask is not None else None
        if valid_mask is not None:
            input_x = input_x * valid_mask
            
        pad_tuple = (6, 0) if self.is_causal else (3, 3)
        if cache is not None and self.is_causal:
            cache_key = str(id(self))
            x_t = input_x.transpose(1, 2)
            if cache_key not in cache:
                cache[cache_key] = torch.zeros((x_t.size(0), x_t.size(1), 6), device=x_t.device, dtype=x_t.dtype)
            cached_x = torch.cat([cache[cache_key], x_t], dim=2)
            if x_t.size(2) >= 6:
                cache[cache_key] = x_t[:, :, -6:]
            else:
                cache[cache_key] = cached_x[:, :, -6:]
            padded_x = cached_x
        else:
            padded_x = F.pad(
                input_x.transpose(1, 2),
                pad_tuple,
                mode="constant",
            )
            
        target_y = self.norm(
            F.conv1d(
                padded_x,
                self.dw_conv.weight,
                self.dw_conv.bias,
                groups=self.dw_conv.groups,
            ).transpose(1, 2)
        )
        if valid_mask is not None:
            target_y = target_y * valid_mask
        target_y = self.se(
            self.pw_conv2(self.act(self.pw_conv1(target_y))),
            key_padding_mask=key_padding_mask,
        )
        if valid_mask is not None:
            target_y = target_y * valid_mask
        return target_y


class BiMamba2SSMBlock(nn.Module):
    r"""
    Bidirectional State Space Model (Mamba-2 Architecture).

    Architecture:
    This block implements a parallelized scan algorithm over the continuous-time state-space differential equation:
        h'(t) = A h(t) + B x(t)
        y(t)  = C h(t)

    Discretization (Zero-Order Hold):
    Using a step size $\\Delta_t$, the system is discretized as:
        $\bar{A} = \exp(\\Delta_t A)$
        $\bar{B} = (\\Delta_t A)^{-1} (\exp(\\Delta_t A) - I) \\cdot \\Delta_t B \approx \\Delta_t B$
        $h_t = \bar{A} h_{t-1} + \bar{B} x_t$
        $y_t = C h_t$

    Bidirectional Formulation:
    To capture future context in non-causal sequence encoding tasks (like video/audio processing),
    we evaluate the state-space formulation independently in both the forward ($t=0 \dots T$)
    and backward ($t=T \dots 0$) directions, summing the resulting $y_t$ vectors.
    """

    def __init__(
        self,
        d_model: int = 512,
        expand: int = 2,
        headdim: int = 80,
        d_state: int = 16,
        d_conv: int = 4,
        ffn_dim: int = 1280,
        drop_path: float = 0.0,
        init_values: float = 0.1,
        max_len: int = 320,
        is_causal: bool = False,
    ):
        """Initializes the module component."""

        super().__init__()
        self.d_model, self.d_inner, self.d_state, self.d_conv = d_model, d_model * expand, d_state, d_conv
        self.nheads, self.headdim = (
            (self.d_inner // headdim)
            if self.d_inner % headdim == 0
            else min(
                [
                    hidden_h
                    for hidden_h in range(1, self.d_inner + 1)
                    if self.d_inner % hidden_h == 0
                ],
                key=lambda hidden_h: abs(hidden_h - max(1, self.d_inner // headdim)),
            )
        ), self.d_inner // (
            self.d_inner // headdim
            if self.d_inner % headdim == 0
            else min(
                [
                    hidden_h
                    for hidden_h in range(1, self.d_inner + 1)
                    if self.d_inner % hidden_h == 0
                ],
                key=lambda hidden_h: abs(hidden_h - max(1, self.d_inner // headdim)),
            )
        )

        self.norm1 = RMSNorm(d_model)
        self.is_causal = is_causal

        self.in_proj = nn.Linear(
            d_model,
            self.d_inner * 2 + self.nheads * d_state * 2 + self.nheads,
            bias=False,
        )

        self.fwd_conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=0 if is_causal else d_conv // 2,
        )

        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, self.nheads + 1, dtype=torch.float32))
        )
        self.dt_bias = nn.Parameter(
            torch.log(torch.exp(torch.rand(self.nheads) * 0.099 + 0.001) - 1)
        )

        self.head_norm_fwd, self.gated_norm = (
            RMSNorm(self.headdim),
            RMSNorm(self.d_inner),
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.gamma_1, self.drop_path1 = nn.Parameter(
            init_values * torch.ones(d_model)
        ), DropPath(drop_path)
        self.norm2, self.ffn, self.gamma_2, self.drop_path2 = (
            RMSNorm(d_model),
            SwiGLUFFN(d_model, ffn_dim),
            nn.Parameter(init_values * torch.ones(d_model)),
            DropPath(drop_path),
        )

        nn.init.orthogonal_(self.in_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight)
        
        self.register_buffer("tril_mask_q", torch.tril(torch.ones(64, 64, dtype=torch.bool)), persistent=False)
        self.register_buffer("tril_mask_c", torch.tril(torch.ones(256, 256, dtype=torch.bool), diagonal=-1), persistent=False)

    def _ssd_multihead_scan(
        self,
        input_x: torch.Tensor,
        dt: torch.Tensor,
        state_a: torch.Tensor,
        batch_sz: torch.Tensor,
        channels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Internal helper method _ssd_multihead_scan."""

        b_sz, t_sz, H_sz, P_sz = input_x.shape
        N_sz = batch_sz.shape[-1]
        if reverse:
            input_x, dt, batch_sz, channels = (
                input_x.flip(1),
                dt.flip(1),
                batch_sz.flip(1),
                channels.flip(1),
            )
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.flip(1)

        dt_act = F.softplus(dt).clamp(max=20.0)
        if key_padding_mask is not None:
            kpm_b = key_padding_mask.unsqueeze(-1)
            dt_act = dt_act.masked_fill(kpm_b, 0.0)
            input_x = input_x.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            batch_sz = batch_sz.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            channels = channels.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            log_decay = -((dt_act * state_a.view(1, 1, H_sz)).clamp(min=0.0, max=20.0))
            log_decay = log_decay.masked_fill(kpm_b, -10.0)
        else:
            # The formula is log_decay = -(dt * A), so A should be exp(a_log) not -exp(a_log).
            log_decay = -((dt_act * state_a.view(1, 1, H_sz)).clamp(min=1e-4, max=20.0))

        query_q = min(chunk_size, t_sz)
        pad_len = (query_q - (t_sz % query_q)) % query_q
        if pad_len > 0:
            input_x, batch_sz, channels, log_decay, dt_act = (
                F.pad(input_x, (0, 0, 0, 0, 0, pad_len)),
                F.pad(batch_sz, (0, 0, 0, 0, 0, pad_len)),
                F.pad(channels, (0, 0, 0, 0, 0, pad_len)),
                F.pad(log_decay, (0, 0, 0, pad_len), value=-10.0),
                F.pad(dt_act, (0, 0, 0, pad_len), value=0),
            )

        T_pad, n_chunks = input_x.shape[1], input_x.shape[1] // query_q
        x_chunk = input_x.reshape(b_sz, n_chunks, query_q, H_sz, P_sz).permute(
            0, 3, 1, 2, 4
        )
        B_chunk = batch_sz.reshape(b_sz, n_chunks, query_q, H_sz, N_sz).permute(
            0, 3, 1, 2, 4
        )
        C_chunk = channels.reshape(b_sz, n_chunks, query_q, H_sz, N_sz).permute(
            0, 3, 1, 2, 4
        )
        ld_chunk = log_decay.reshape(b_sz, n_chunks, query_q, H_sz).permute(0, 3, 1, 2)

        B_chunk_dt = B_chunk * dt_act.view(b_sz, n_chunks, query_q, H_sz).permute(
            0, 3, 1, 2
        ).unsqueeze(-1)
        CB = torch.matmul(C_chunk, B_chunk_dt.transpose(-1, -2)) / math.sqrt(N_sz)
        cum_decay = ld_chunk.to(torch.float32).cumsum(dim=-1).to(ld_chunk.dtype)
        M = torch.exp(
            (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).masked_fill(
                ~self.tril_mask_q[:query_q, :query_q],
                -65500.0,
            )
        )
        Y_intra = torch.matmul(M * CB, x_chunk)

        log_chunk_decay = ld_chunk.sum(dim=-1)
        decay_to_end = torch.exp(cum_decay[:, :, :, -1:] - cum_decay)
        state_gen = torch.einsum(
            "bhcqp, bhcqn -> bhcpn", x_chunk * decay_to_end.unsqueeze(-1), B_chunk_dt
        )

        length_l = log_chunk_decay.cumsum(dim=2)
        L_shifted = torch.cat(
            [torch.zeros_like(length_l[:, :, :1]), length_l[:, :, :-1]], dim=2
        )
        M_inter = torch.exp(
            (L_shifted.unsqueeze(-1) - length_l.unsqueeze(-2)).masked_fill(
                ~torch.tril(torch.ones(n_chunks, n_chunks, dtype=torch.bool, device=input_x.device), diagonal=-1),
                -65500.0,
            )
        )

        state_stack_flat = torch.einsum(
            "bhij, bhjk -> bhik",
            M_inter,
            state_gen.contiguous().reshape(
                b_sz, H_sz, n_chunks, state_gen.shape[-2] * state_gen.shape[-1]
            ),
        )
        state_stack = state_stack_flat.reshape(
            b_sz, H_sz, n_chunks, state_gen.shape[-2], state_gen.shape[-1]
        )

        C_state = torch.einsum(
            "bhcqn, bhcpn -> bhcqp", C_chunk, state_stack
        ) / math.sqrt(N_sz)
        Y_inter = C_state * torch.exp(cum_decay).unsqueeze(-1)

        Y_flat = (
            (Y_intra + Y_inter).permute(0, 2, 3, 1, 4).reshape(b_sz, T_pad, H_sz, P_sz)
        )
        return Y_flat[:, :t_sz].flip(1) if reverse else Y_flat[:, :t_sz]

    def forward(
        self,
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for this module."""

        if key_padding_mask is not None:
            input_x = input_x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        xn = self.norm1(input_x)
        b_sz, t_sz, _ = xn.shape

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
        x_conv_in = x_proj.transpose(1, 2)
        if self.is_causal:
            x_conv_in = F.pad(x_conv_in, (self.d_conv - 1, 0))
        x_fwd_h_padded = self.fwd_conv1d(x_conv_in)
        if x_fwd_h_padded.shape[-1] > t_sz:
            x_fwd_h_padded = x_fwd_h_padded[..., :t_sz]
        x_fwd_h = F.silu(x_fwd_h_padded.transpose(1, 2)).view(
            b_sz, t_sz, self.nheads, self.headdim
        )
        B_h_fwd, C_h_fwd = B_ssm_fwd.view(
            b_sz, t_sz, self.nheads, self.d_state
        ), C_ssm_fwd.view(b_sz, t_sz, self.nheads, self.d_state)
        state_a = torch.exp(self.a_log)
        y_fwd = self._ssd_multihead_scan(
            x_fwd_h,
            dt_fwd + self.dt_bias,
            state_a,
            B_h_fwd,
            C_h_fwd,
            key_padding_mask=key_padding_mask,
            reverse=False,
        )
        if self.is_causal:
            y_normed = self.head_norm_fwd(y_fwd)
        else:
            y_bwd = self._ssd_multihead_scan(
                x_fwd_h,
                dt_fwd + self.dt_bias,
                state_a,
                B_h_fwd,
                C_h_fwd,
                key_padding_mask=key_padding_mask,
                reverse=True,
            )
            y_normed = self.head_norm_fwd(y_fwd + y_bwd)
        out = self.out_proj(
            self.gated_norm(y_normed.reshape(b_sz, t_sz, self.d_inner) * F.silu(z))
        )
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        input_x = input_x + self.drop_path1(self.gamma_1 * out)
        x2 = self.ffn(self.norm2(input_x))
        if key_padding_mask is not None:
            x2 = x2.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return input_x + self.drop_path2(self.gamma_2 * x2)


class MobileConformerBlock(nn.Module):
    """
    MobileConformerBlock: A lightweight variant of the Conformer architecture designed for sign language recognition.

    Architecture:
    Combines Transformer self-attention with depthwise convolutions to capture both global context and local feature correlations.

    Mathematical Formulation:
    1. FeedForward Module 1 (FFN1): $x_1 = x_0 + \frac{1}{2} \text{FFN}(x_0)$
    2. Grouped-Query Attention (GQA): $x_2 = x_1 + \text{GQA}(x_1)$
    3. Convolution Module: $x_3 = x_2 + \text{Conv}(x_2)$
    4. FeedForward Module 2 (FFN2): $y = \text{LayerNorm}(x_3 + \frac{1}{2} \text{FFN}(x_3))$

    The convolution block utilizes a point-wise convolution followed by a GLU activation, a 1D depthwise convolution,
    and a final point-wise convolution.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 1280,
        dropout_p: float = 0.1,
        drop_path: float = 0.0,
        num_enc_layers: int = 8,
        init_values: float = 0.1,
        max_len: int = 320,
        use_swin: bool = False,
        window_size: int = 128,
        shift_size: int = 0,
        is_causal: bool = False,
        lookahead: int = 4,
    ):
        """Initializes the module component."""

        super().__init__()
        self.is_causal = is_causal
        self.ffn1_norm = RMSNorm(d_model)
        self.ffn1 = SwiGLUFFN(d_model, dim_feedforward, num_layers=num_enc_layers)
        self.drop_path_ffn1 = DropPath(drop_path)
        self.gamma_ffn1 = nn.Parameter(init_values * torch.ones(d_model))

        self.mha_norm = RMSNorm(d_model)
        mha_base = GroupedQueryEncoderAttention(
            d_model=d_model, nhead=nhead, kv_heads=2, max_len=max_len if not use_swin else window_size, is_causal=is_causal, lookahead=lookahead
        )
        if use_swin:
            self.mha = Swin1DAttention(mha_base, window_size=window_size, shift_size=shift_size)
        else:
            self.mha = mha_base
        self.drop_path_mha = DropPath(drop_path)
        self.gamma_mha = nn.Parameter(init_values * torch.ones(d_model))

        self.conv_norm = RMSNorm(d_model)
        self.conv_block = ConvNeXtTemporalBlock(d_model, is_causal=is_causal)
        self.drop_path_conv = DropPath(drop_path)
        self.gamma_conv = nn.Parameter(init_values * torch.ones(d_model))

        self.ffn2_norm = RMSNorm(d_model)
        self.ffn2 = SwiGLUFFN(d_model, dim_feedforward, num_layers=num_enc_layers)
        self.drop_path_ffn2 = DropPath(drop_path)
        self.gamma_ffn2 = nn.Parameter(init_values * torch.ones(d_model))

    def forward(
        self,
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass for this module."""

        from torch.utils.checkpoint import checkpoint

        def _inner_forward(x_in, kpm_in, fi_in):
            """Forward pass for this module."""

            if kpm_in is not None:
                x_in = x_in.masked_fill(kpm_in.unsqueeze(-1), 0.0)
            x_in = x_in + 0.5 * self.drop_path_ffn1(
                self.gamma_ffn1 * self.ffn1(self.ffn1_norm(x_in))
            )
            if kpm_in is not None:
                x_in = x_in.masked_fill(kpm_in.unsqueeze(-1), 0.0)
            x_in = x_in + self.drop_path_mha(
                self.gamma_mha
                * self.mha(
                    self.mha_norm(x_in),
                    key_padding_mask=kpm_in,
                    frame_indices=fi_in,
                )
            )
            if kpm_in is not None:
                x_in = x_in.masked_fill(kpm_in.unsqueeze(-1), 0.0)

            if not self.is_causal:
                cls_t = x_in[:, :1]
                x_seq = x_in[:, 1:]
                seq_mask = kpm_in[:, 1:] if kpm_in is not None else None
                xc_seq = self.conv_block(self.conv_norm(x_seq), key_padding_mask=seq_mask, cache=cache)
                xc = torch.cat([cls_t, xc_seq], dim=1)
            else:
                x_seq = x_in
                seq_mask = kpm_in
                xc = self.conv_block(self.conv_norm(x_seq), key_padding_mask=seq_mask, cache=cache)

            x_in = x_in + self.drop_path_conv(self.gamma_conv * xc)
            if kpm_in is not None:
                x_in = x_in.masked_fill(kpm_in.unsqueeze(-1), 0.0)
            x_in = x_in + 0.5 * self.drop_path_ffn2(
                self.gamma_ffn2 * self.ffn2(self.ffn2_norm(x_in))
            )
            if kpm_in is not None:
                x_in = x_in.masked_fill(kpm_in.unsqueeze(-1), 0.0)
            return x_in

        # If we are training and input_x requires gradients, we checkpoint.
        # But input_x might not require gradients yet, so we dummy-require it.
        if self.training and input_x.requires_grad:
            return checkpoint(
                _inner_forward,
                input_x,
                key_padding_mask,
                frame_indices,
                use_reentrant=False,
            )
        else:
            return _inner_forward(input_x, key_padding_mask, frame_indices)


class LandmarkTrajectory1DStem(nn.Module):
    """Projects raw features into the embedding space using LandmarkTrajectory1DStem."""

    def __init__(
        self, in_channels: int = 9, num_keypoints: int = 60, out_dim: int = 128, is_causal: bool = False
    ):
        """Initializes the module component."""

        super().__init__()
        self.is_causal = is_causal
        in_dim = num_keypoints * in_channels
        self.conv1 = nn.Conv1d(in_dim, 256, kernel_size=7, padding=0, groups=1)
        self.norm1 = RMSNorm(256) if is_causal else nn.GroupNorm(8, 256)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=0, groups=256)
        self.conv3 = nn.Conv1d(256, out_dim, kernel_size=1)
        self.norm2 = RMSNorm(out_dim) if is_causal else nn.GroupNorm(8, out_dim)
        self.act2 = nn.GELU()
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(
        self, input_x: torch.Tensor, mask: Optional[torch.Tensor] = None, cache: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass for this module."""

        batch_sz, seq_len = input_x.size(0), input_x.size(1)
        if input_x.is_floating_point() and input_x.dtype != self.conv1.weight.dtype:
            input_x = input_x.to(self.conv1.weight.dtype)
        x_flat = (
            input_x.reshape(batch_sz, seq_len, -1) if input_x.dim() == 4 else input_x
        )
        x_t = x_flat.transpose(1, 2)
        if mask is not None:
            x_t = x_t * mask.unsqueeze(1).to(x_t.dtype)

        feat_seq = x_t
        pad1 = (6, 0) if self.is_causal else (3, 3)
        if cache is not None and self.is_causal:
            k1 = str(id(self)) + "_1"
            if k1 not in cache:
                cache[k1] = torch.zeros((feat_seq.size(0), feat_seq.size(1), 6), device=feat_seq.device, dtype=feat_seq.dtype)
            cached_x1 = torch.cat([cache[k1], feat_seq], dim=2)
            if feat_seq.size(2) >= 6: cache[k1] = feat_seq[:, :, -6:]
            else: cache[k1] = cached_x1[:, :, -6:]
            padded_x1 = cached_x1
        else:
            padded_x1 = F.pad(feat_seq, pad1, mode="constant", value=0)
            
        feat_seq = self.act1(
            self.norm1(self.conv1(padded_x1).transpose(1, 2) if self.is_causal else self.conv1(padded_x1)).transpose(1, 2) if self.is_causal else self.norm1(self.conv1(padded_x1))
        )
        if feat_seq.dtype != self.conv2.weight.dtype:
            feat_seq = feat_seq.to(self.conv2.weight.dtype)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(1).to(feat_seq.dtype)

        pad2 = (4, 0) if self.is_causal else (2, 2)
        if cache is not None and self.is_causal:
            k2 = str(id(self)) + "_2"
            if k2 not in cache:
                cache[k2] = torch.zeros((feat_seq.size(0), feat_seq.size(1), 4), device=feat_seq.device, dtype=feat_seq.dtype)
            cached_x2 = torch.cat([cache[k2], feat_seq], dim=2)
            if feat_seq.size(2) >= 4: cache[k2] = feat_seq[:, :, -4:]
            else: cache[k2] = cached_x2[:, :, -4:]
            padded_x2 = cached_x2
        else:
            padded_x2 = F.pad(feat_seq, pad2, mode="constant", value=0)
            
        feat_seq = self.conv2(padded_x2)
        if feat_seq.dtype != self.conv3.weight.dtype:
            feat_seq = feat_seq.to(self.conv3.weight.dtype)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(1).to(feat_seq.dtype)

        feat_seq = self.act2(self.norm2(self.conv3(feat_seq).transpose(1, 2) if self.is_causal else self.conv3(feat_seq)).transpose(1, 2) if self.is_causal else self.norm2(self.conv3(feat_seq)))

        feat_seq = feat_seq.transpose(1, 2)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(-1).to(feat_seq.dtype)
        if feat_seq.dtype != self.out_proj.weight.dtype:
            feat_seq = feat_seq.to(self.out_proj.weight.dtype)
        return self.out_proj(feat_seq)


class MaskedGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x, mask=None):
        if mask is None:
            out = F.group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
            if out.dtype != x.dtype:
                out = out.to(x.dtype)
            return out

        B, C, T = x.shape
        G = self.num_groups
        D = C // G

        mask_ = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
        x_g = x.view(B, G, D, T)  # [B, G, D, T]

        valid_count = mask.sum(dim=1).view(B, 1, 1, 1).clamp(min=1) * D

        mean = (x_g * mask_).sum(dim=(2, 3), keepdim=True) / valid_count
        var = (((x_g - mean) ** 2) * mask_).sum(dim=(2, 3), keepdim=True) / valid_count

        x_normed = (x_g - mean) / torch.sqrt(var + self.eps)
        x_normed = x_normed.view(B, C, T)

        if x_normed.dtype != self.weight.dtype:
            x_normed = x_normed.to(self.weight.dtype)

        x_normed = x_normed * self.weight.view(1, C, 1)
        x_normed = x_normed + self.bias.view(1, C, 1)

        out = x_normed * mask.unsqueeze(1).to(x_normed.dtype)
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out


class VisualStem(nn.Module):
    """Projects raw features into the embedding space using VisualStem."""

    def __init__(
        self, in_channels: int = 9, num_keypoints: int = 60, out_dim: int = 128, is_causal: bool = False
    ):
        """Initializes the module component."""

        super().__init__()
        self.is_causal = is_causal
        in_dim = num_keypoints * in_channels
        self.conv1 = nn.Conv1d(in_dim, 256, kernel_size=7, padding=0, groups=1)
        self.norm1 = RMSNorm(256) if is_causal else MaskedGroupNorm(8, 256)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=0, groups=256)
        self.conv3 = nn.Conv1d(256, out_dim, kernel_size=1)
        self.norm2 = RMSNorm(out_dim) if is_causal else MaskedGroupNorm(8, out_dim)
        self.act2 = nn.GELU()
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(
        self, input_x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for this module."""

        batch_sz, seq_len = input_x.size(0), input_x.size(1)
        if input_x.is_floating_point() and input_x.dtype != self.conv1.weight.dtype:
            input_x = input_x.to(self.conv1.weight.dtype)
        x_flat = (
            input_x.reshape(batch_sz, seq_len, -1) if input_x.dim() == 4 else input_x
        )
        x_t = x_flat.transpose(1, 2)
        if mask is not None:
            x_t = x_t * mask.unsqueeze(1).to(x_t.dtype)

        feat_seq = x_t
        pad1 = (6, 0) if self.is_causal else (3, 3)
        if self.is_causal:
            feat_seq = self.act1(
                self.norm1(self.conv1(F.pad(feat_seq, pad1, mode="constant", value=0)).transpose(1, 2)).transpose(1, 2)
            )
        else:
            feat_seq = self.act1(
                self.norm1(self.conv1(F.pad(feat_seq, pad1, mode="constant", value=0)), mask)
            )
        if feat_seq.dtype != self.conv2.weight.dtype:
            feat_seq = feat_seq.to(self.conv2.weight.dtype)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(1).to(feat_seq.dtype)

        pad2 = (4, 0) if self.is_causal else (2, 2)
        feat_seq = self.conv2(F.pad(feat_seq, pad2, mode="constant", value=0))
        if feat_seq.dtype != self.conv3.weight.dtype:
            feat_seq = feat_seq.to(self.conv3.weight.dtype)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(1).to(feat_seq.dtype)

        if self.is_causal:
            feat_seq = self.act2(self.norm2(self.conv3(feat_seq).transpose(1, 2)).transpose(1, 2))
        else:
            feat_seq = self.act2(self.norm2(self.conv3(feat_seq), mask))

        feat_seq = feat_seq.transpose(1, 2)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(-1).to(feat_seq.dtype)
        if feat_seq.dtype != self.out_proj.weight.dtype:
            feat_seq = feat_seq.to(self.out_proj.weight.dtype)
        return self.out_proj(feat_seq)


# ==============================================================================
# 7. TRANSFORMER DECODER WITH EOS GRAMMAR PROTECTION
# ==============================================================================


class RoPEEmbedding(nn.Module):
    """Projects raw features into the embedding space using RoPEEmbedding."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        """Initializes the module component."""

        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Internal helper method _build_cache."""

        t = torch.arange(
            seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None], persistent=False)
        self._cache_len = seq_len

    @staticmethod
    def _rotate_half(input_x: torch.Tensor) -> torch.Tensor:
        """Internal helper method _rotate_half."""

        half = input_x.shape[-1] // 2
        return torch.cat([-input_x[..., half:], input_x[..., :half]], dim=-1)

    def forward(
        self, query_q_lower: torch.Tensor, key_k_lower: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for this module."""

        seq_s = query_q_lower.shape[-2]
        if isinstance(offset, int):
            offset_t = torch.tensor([offset], device=query_q_lower.device, dtype=torch.long)
        else:
            offset_t = offset.view(-1).to(torch.long)

        positions = torch.arange(seq_s, device=query_q_lower.device, dtype=torch.long) + offset_t
        cos_flat = self.cos_cache.squeeze(0).squeeze(0)  # [L, D]
        sin_flat = self.sin_cache.squeeze(0).squeeze(0)  # [L, D]

        cos = F.embedding(positions, cos_flat).unsqueeze(0).unsqueeze(0).to(query_q_lower.dtype)
        sin = F.embedding(positions, sin_flat).unsqueeze(0).unsqueeze(0).to(query_q_lower.dtype)

        query_q_lower = query_q_lower * cos + self._rotate_half(query_q_lower) * sin
        key_k_lower = key_k_lower * cos + self._rotate_half(key_k_lower) * sin
        return query_q_lower, key_k_lower


class GroupedQueryAttention(nn.Module):
    r"""
    Grouped-Query Attention (GQA) with Rotary Position Embeddings (RoPE).

    Architecture:
    Standard Multi-Head Attention (MHA) maintains $H$ key and value heads, requiring $O(T \cdot H \cdot D)$ memory caching.
    Multi-Query Attention (MQA) uses 1 key/value head, reducing memory but heavily degrading representational capacity.
    GQA interpolates between MHA and MQA by clustering $H$ query heads into $G$ groups, where each group shares a single Key/Value head.

    Mathematical Formulation:
    Let $Q \in \mathbb{R}^{B \times T \times H \times d}$, $K, V \in \mathbb{R}^{B \times S \times G \times d}$.
    For a given query head $h \in [1, H]$, its corresponding KV group is $g = \lfloor h \times G / H \rfloor$.

    The attention mechanism is computed as:
    $A_{h} = \text{Softmax}\left(\frac{Q_h K_g^T}{\sqrt{d}}\right) V_g$

    Rotary Position Embeddings (RoPE):
    Before computing the dot product, the first $d_{rope}$ dimensions of $Q$ and $K$ are rotated.
    Let $x_{m}^{(1)}, x_{m}^{(2)}$ be a feature pair at temporal position $m$. The rotation matrix $R_{\Theta, m}^d$ applies:
    $\begin{bmatrix} q_m^{(1)} \\ q_m^{(2)} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_m^{(1)} \\ x_m^{(2)} \end{bmatrix}$
    where $\theta_i = 10000^{-2i/d}$.
    This ensures that the inner product $\langle q_m, k_n \rangle$ depends strictly on the relative distance $(m - n)$.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        max_seq_len: int = 256,
    ):
        super().__init__()
        kv_heads = min(kv_heads, nhead)
        if nhead % kv_heads != 0:
            kv_heads = 1
        assert nhead % kv_heads == 0
        self.nhead, self.kv_heads, self.groups, self.head_dim = (
            nhead,
            kv_heads,
            nhead // kv_heads,
            d_model // nhead,
        )
        self.scale = self.head_dim**-0.5

        # DeepSeek V3 MLA Latent Compression
        self.latent_dim = d_model // 4
        self.kv_latent_proj = nn.Linear(d_model, self.latent_dim, bias=False)
        self.kv_latent_norm = RMSNorm(self.latent_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(
            self.latent_dim, 2 * kv_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=max_seq_len)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.kv_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        input_x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """Forward pass for this module."""

        batch_sz, seq_len, _ = input_x.shape
        q_in = self.q_norm(input_x)

        # MLA Latent Projection
        kv_latent = self.kv_latent_proj(input_x)
        kv_latent = self.kv_latent_norm(kv_latent)

        query_q_lower = (
            self.q_proj(q_in)
            .view(batch_sz, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )
        kv = self.kv_proj(kv_latent)
        key_k_lower, val_v = torch.split(kv, kv.size(-1) // 2, dim=-1)
        key_k_lower = key_k_lower.view(
            batch_sz, seq_len, self.kv_heads, self.head_dim
        ).transpose(1, 2)
        val_v = val_v.view(batch_sz, seq_len, self.kv_heads, self.head_dim).transpose(
            1, 2
        )

        if past_key_value is not None:
            if len(past_key_value) == 3:
                k_cache, v_cache, past_len = past_key_value
                query_q_lower, key_k_lower = self.rope(
                    query_q_lower, key_k_lower, offset=past_len
                )
                if seq_len == 1:
                    k_cache = k_cache.index_copy(2, past_len, key_k_lower)
                    v_cache = v_cache.index_copy(2, past_len, val_v)
                    key_k_lower = k_cache
                    val_v = v_cache
                else:
                    k_cache[:, :, past_len : past_len + seq_len, :] = key_k_lower
                    v_cache[:, :, past_len : past_len + seq_len, :] = val_v
                    key_k_lower = k_cache[:, :, : past_len + seq_len, :]
                    val_v = v_cache[:, :, : past_len + seq_len, :]
                    
                current_key_value = (
                    (k_cache, v_cache, past_len + seq_len) if use_cache else None
                )
            else:
                past_len = past_key_value[0].size(2)
                query_q_lower, key_k_lower = self.rope(
                    query_q_lower, key_k_lower, offset=past_len
                )
                key_k_lower = torch.cat([past_key_value[0], key_k_lower], dim=2)
                val_v = torch.cat([past_key_value[1], val_v], dim=2)
                current_key_value = (key_k_lower, val_v) if use_cache else None
        else:
            past_len = torch.tensor([0], device=input_x.device, dtype=torch.long) if use_cache else 0
            query_q_lower, key_k_lower = self.rope(
                query_q_lower, key_k_lower, offset=past_len
            )
            current_key_value = (key_k_lower, val_v) if use_cache else None

        if self.groups > 1:
            key_k_lower = key_k_lower.unsqueeze(2).expand(batch_sz, self.kv_heads, self.groups, key_k_lower.size(2), self.head_dim).reshape(batch_sz, self.nhead, key_k_lower.size(2), self.head_dim)
            val_v = val_v.unsqueeze(2).expand(batch_sz, self.kv_heads, self.groups, val_v.size(2), self.head_dim).reshape(batch_sz, self.nhead, val_v.size(2), self.head_dim)

        if past_len > 0:
            if seq_len == 1:
                max_len = key_k_lower.size(2)
                total_len = int(past_len) + 1 if not isinstance(past_len, torch.Tensor) else int(past_len) + 1
                causal_mask = (torch.arange(max_len, device=input_x.device) < total_len).view(1, 1, 1, max_len)
                
                if padding_mask is not None:
                    attn_mask = (~padding_mask).unsqueeze(1).unsqueeze(2) & causal_mask
                    out = F.scaled_dot_product_attention(
                        query_q_lower,
                        key_k_lower,
                        val_v,
                        attn_mask=attn_mask,
                        scale=self.scale,
                    )
                else:
                    out = F.scaled_dot_product_attention(
                        query_q_lower,
                        key_k_lower,
                        val_v,
                        attn_mask=causal_mask,
                        scale=self.scale,
                    )
            else:
                max_len = key_k_lower.size(2)
                total_len = int(past_len) + seq_len if not isinstance(past_len, torch.Tensor) else int(past_len) + seq_len
                
                # Standard causal mask for the new tokens
                causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_x.device))
                # Full mask: past tokens are all True, future padding is False
                full_mask = torch.zeros(seq_len, max_len, dtype=torch.bool, device=input_x.device)
                past_l = int(past_len) if not isinstance(past_len, torch.Tensor) else int(past_len)
                if past_l > 0:
                    full_mask[:, :past_l] = True
                full_mask[:, past_l:total_len] = causal
                
                attn_mask = full_mask.unsqueeze(0).unsqueeze(1)
                if padding_mask is not None:
                    attn_mask = attn_mask & (~padding_mask).unsqueeze(1).unsqueeze(2)
                out = F.scaled_dot_product_attention(
                    query_q_lower,
                    key_k_lower,
                    val_v,
                    attn_mask=attn_mask,
                    scale=self.scale,
                )
        else:
            if padding_mask is not None:
                max_len = key_k_lower.size(2)
                total_len = past_len + seq_len if not isinstance(past_len, torch.Tensor) else int(past_len) + seq_len
                if seq_len == 1:
                    causal_mask = (torch.arange(max_len, device=input_x.device) < total_len).view(1, 1, 1, max_len)
                    attn_mask = (~padding_mask).unsqueeze(1).unsqueeze(2) & causal_mask
                else:
                    causal = torch.tril(torch.ones(seq_len, total_len, dtype=torch.bool, device=input_x.device))
                    full_mask = torch.zeros(seq_len, max_len, dtype=torch.bool, device=input_x.device)
                    full_mask[:, :total_len] = causal
                    attn_mask = full_mask.unsqueeze(0).unsqueeze(1) & (~padding_mask).unsqueeze(1).unsqueeze(2)
                    
                out = F.scaled_dot_product_attention(
                    query_q_lower,
                    key_k_lower,
                    val_v,
                    attn_mask=attn_mask,
                    scale=self.scale,
                )
            else:
                max_len = key_k_lower.size(2)
                total_len = past_len + seq_len if not isinstance(past_len, torch.Tensor) else int(past_len) + seq_len
                if seq_len == 1:
                    causal_mask = (torch.arange(max_len, device=input_x.device) < total_len).view(1, 1, 1, max_len)
                    out = F.scaled_dot_product_attention(
                        query_q_lower, key_k_lower, val_v, attn_mask=causal_mask, scale=self.scale,
                    )
                else:
                    causal = torch.tril(torch.ones(seq_len, total_len, dtype=torch.bool, device=input_x.device))
                    full_mask = torch.zeros(seq_len, max_len, dtype=torch.bool, device=input_x.device)
                    full_mask[:, :total_len] = causal
                    out = F.scaled_dot_product_attention(
                        query_q_lower, key_k_lower, val_v, attn_mask=full_mask.unsqueeze(0).unsqueeze(1), scale=self.scale,
                    )

        out = self.o_proj(out.transpose(1, 2).reshape(batch_sz, seq_len, -1))
        return (out, current_key_value) if use_cache else out


class DecoderCrossAttention(nn.Module):
    """Implements the DecoderCrossAttention architecture for the sequence modeling pipeline."""

    def __init__(self, d_model: int = 512, nhead: int = 8, kv_heads: int = 2):
        """Initializes the module component."""

        super().__init__()
        kv_heads = min(kv_heads, nhead)
        if nhead % kv_heads != 0:
            kv_heads = 1
        assert nhead % kv_heads == 0
        self.nhead, self.kv_heads, self.groups, self.head_dim = (
            nhead,
            kv_heads,
            nhead // kv_heads,
            d_model // nhead,
        )
        # DeepSeek V3 MLA Latent Compression
        self.latent_dim = d_model // 4
        self.kv_latent_proj = nn.Linear(d_model, self.latent_dim, bias=False)
        self.kv_latent_norm = RMSNorm(self.latent_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(
            self.latent_dim, 2 * kv_heads * self.head_dim, bias=False
        )

        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=2048)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.kv_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """Forward pass for this module."""

        batch_sz, seq_len, _ = tgt.shape
        query_q = (
            self.q_proj(self.q_norm(tgt))
            .view(batch_sz, seq_len, self.nhead, self.head_dim)
            .transpose(1, 2)
        )

        if past_key_value is not None:
            key_k_lower, val_v = past_key_value
        else:
            seq_s = memory.size(1)
            # MLA Latent Projection for Cross Attention
            kv_latent = self.kv_latent_proj(memory)
            kv_latent = self.kv_latent_norm(kv_latent)

            kv = self.kv_proj(kv_latent)
            key_k_lower, val_v = torch.split(kv, kv.size(-1) // 2, dim=-1)
            key_k_lower = key_k_lower.view(
                batch_sz, seq_s, self.kv_heads, self.head_dim
            ).transpose(1, 2)
            val_v = val_v.view(batch_sz, seq_s, self.kv_heads, self.head_dim).transpose(
                1, 2
            )
        # Query (decoder) and key (encoder) have different sequence lengths;
        # applying shared RoPE causes shape broadcast errors and positional aliasing.
        # RoPE is only valid in self-attention where Q and K share temporal positions.

        current_key_value = (key_k_lower, val_v) if use_cache else None

        if self.groups > 1:
            key_k_lower = key_k_lower.unsqueeze(2).expand(batch_sz, self.kv_heads, self.groups, key_k_lower.size(2), self.head_dim).reshape(batch_sz, self.nhead, key_k_lower.size(2), self.head_dim)
            val_v = val_v.unsqueeze(2).expand(batch_sz, self.kv_heads, self.groups, val_v.size(2), self.head_dim).reshape(batch_sz, self.nhead, val_v.size(2), self.head_dim)

        if memory_key_padding_mask is not None:
            # PyTorch SDPA treats True = attend (valid token), False = ignore (pad token).
            # Callers pass memory_key_padding_mask where True = PAD token.
            attn_mask = (~memory_key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)
            out = F.scaled_dot_product_attention(
                query_q,
                key_k_lower,
                val_v,
                attn_mask=attn_mask,
            )
        else:
            out = F.scaled_dot_product_attention(
                query_q,
                key_k_lower,
                val_v,
            )
        out = self.o_proj(out.transpose(1, 2).reshape(batch_sz, seq_len, -1))
        return (out, current_key_value) if use_cache else out


class ASLDecoderLayer(nn.Module):
    """Implements the ASLDecoderLayer architecture for the sequence modeling pipeline."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        ffn_dim: int = 1280,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        num_layers: int = 8,
    ):
        """Initializes the module component."""

        super().__init__()
        iv = 0.1
        self.norm1 = RMSNorm(d_model)
        self.self_attn = GroupedQueryAttention(d_model, nhead, kv_heads, max_seq_len)
        self.gamma1 = nn.Parameter(iv * torch.ones(d_model))

        self.norm2 = RMSNorm(d_model)
        self.cross_attn = DecoderCrossAttention(d_model, nhead, kv_heads)
        self.gamma2 = nn.Parameter(iv * torch.ones(d_model))

        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(
            d_model=d_model, dim_feedforward=ffn_dim, num_layers=num_layers
        )
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
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_self_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        past_cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        """Forward pass for this module."""

        if use_cache:
            sa_out, new_self_kv = self.self_attn(
                self.norm1(tgt),
                padding_mask=tgt_key_padding_mask,
                past_key_value=past_self_kv,
                use_cache=True,
            )
            tgt = tgt + self.gamma1 * sa_out
            if memory is not None:
                ca_out, new_cross_kv = self.cross_attn(
                    self.norm2(tgt),
                    memory,
                    memory_key_padding_mask=memory_key_padding_mask,
                    past_key_value=past_cross_kv,
                    use_cache=True,
                )
                tgt = tgt + self.gamma2 * ca_out
            else:
                new_cross_kv = None
            tgt = tgt + self.gamma3 * self.ffn(self.norm3(tgt))
            return tgt, new_self_kv, new_cross_kv
        else:
            tgt = tgt + self.drop1(
                self.gamma1
                * self.self_attn(self.norm1(tgt), padding_mask=tgt_key_padding_mask)
            )
            if memory is not None:
                tgt = tgt + self.drop2(
                    self.gamma2
                    * self.cross_attn(self.norm2(tgt), memory, memory_key_padding_mask)
                )
            tgt = tgt + self.drop3(self.gamma3 * self.ffn(self.norm3(tgt)))
            return tgt, None, None


class ASLTransformerDecoder(nn.Module):
    """Implements the ASLTransformerDecoder architecture for the sequence modeling pipeline."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        num_layers: int = 8,
        ffn_dim: int = 1280,
        dropout: float = 0.1,
        max_seq_len: int = 256,
        csv_path: Optional[Union[str, Path]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        use_asl_lex: bool = True,
        pad_id: int = 0,
        bos_id: int = 1,
        eos_id: int = 2,
        unk_id: int = 3,
    ):
        """Initializes the module component."""

        super().__init__()
        self.d_model, self.vocab_size, self.max_seq_len, self.input_token_dropout = (
            d_model,
            vocab_size,
            max_seq_len,
            0.12,
        )

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
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
                ASLDecoderLayer(
                    d_model,
                    nhead,
                    kv_heads,
                    ffn_dim,
                    dropout,
                    max_seq_len,
                    num_layers=num_layers,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        self.mtp_layer = ASLDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            kv_heads=kv_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.mtp_layer2 = ASLDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            kv_heads=kv_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )

        nn.init.normal_(self.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[pad_id].fill_(0)
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id

    def forward(
        self,
        tgt_ids: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple]] = None,
        use_cache: bool = False,
    ):
        """Forward pass for this module."""

        batch_sz, seq_s = tgt_ids.shape
        if self.training and self.input_token_dropout > 0:
            drop_mask = (
                (
                    torch.rand(tgt_ids.shape, device=tgt_ids.device)
                    < self.input_token_dropout
                )
                & (tgt_ids != self.pad_id)
                & (tgt_ids != self.bos_id)
                & (tgt_ids != self.eos_id)
            )
            dropped_tgt_ids = torch.where(
                drop_mask,
                torch.full_like(tgt_ids, self.unk_id),
                tgt_ids,
            )
        else:
            dropped_tgt_ids = tgt_ids

        # Removed .any() validation check to prevent XLA device-to-host syncs
        if getattr(self, "use_asl_lex", True) and self.asl_lex_emb is not None:
            lex_embs = self.asl_lex_emb(dropped_tgt_ids)
            valid_lex_mask = (tgt_ids != self.pad_id).unsqueeze(-1).to(lex_embs.dtype)
            hidden_h = self.emb_drop(
                self.token_emb(dropped_tgt_ids) * self.emb_scale
                + lex_embs * self.emb_scale * valid_lex_mask
            )
        else:
            hidden_h = self.emb_drop(self.token_emb(dropped_tgt_ids) * self.emb_scale)

        new_key_values = [] if use_cache else None
        tgt_key_padding_mask = (
            (dropped_tgt_ids == self.pad_id) if dropped_tgt_ids is not None else None
        )

        for idx, layer in enumerate(self.layers):
            if use_cache:
                hidden_h, n_self_kv, n_cross_kv = layer(
                    hidden_h,
                    memory,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    past_self_kv=past_key_values[idx][0] if past_key_values else None,
                    past_cross_kv=past_key_values[idx][1] if past_key_values else None,
                    use_cache=True,
                )
                new_key_values.append((n_self_kv, n_cross_kv))
            else:
                hidden_h = layer(
                    hidden_h,
                    memory,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )[0]

        hidden_h = self.final_norm(hidden_h)
        logits = self.lm_head(hidden_h)

        if use_cache:
            if getattr(self, "training", False):
                # We do not need MTP representations during pure AR inference
                past_mtp1_kv = (
                    past_key_values[-2]
                    if past_key_values and len(past_key_values) > len(self.layers)
                    else (None, None)
                )
                past_mtp2_kv = (
                    past_key_values[-1]
                    if past_key_values and len(past_key_values) > len(self.layers)
                    else (None, None)
                )

                h_mtp, mtp1_self_kv, mtp1_cross_kv = self.mtp_layer(
                    hidden_h,
                    memory,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    past_self_kv=past_mtp1_kv[0],
                    past_cross_kv=past_mtp1_kv[1],
                    use_cache=True,
                )
                logits_2 = self.lm_head(h_mtp)

                h_mtp_2, mtp2_self_kv, mtp2_cross_kv = self.mtp_layer2(
                    h_mtp,
                    memory,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    past_self_kv=past_mtp2_kv[0],
                    past_cross_kv=past_mtp2_kv[1],
                    use_cache=True,
                )
                logits_3 = self.lm_head(h_mtp_2)

                new_key_values.append((mtp1_self_kv, mtp1_cross_kv))
                new_key_values.append((mtp2_self_kv, mtp2_cross_kv))
            else:
                logits_2 = None
                logits_3 = None
        else:
            h_mtp = self.mtp_layer(
                hidden_h,
                memory,
                memory_key_padding_mask=memory_key_padding_mask,
            )[0]
            logits_2 = self.lm_head(h_mtp)
            h_mtp_2 = self.mtp_layer2(
                h_mtp, memory, memory_key_padding_mask=memory_key_padding_mask
            )[0]
            logits_3 = self.lm_head(h_mtp_2)
        extra_logits = {"logits_2": logits_2, "logits_3": logits_3}

        return (
            (logits, hidden_h, extra_logits, new_key_values)
            if use_cache
            else (logits, hidden_h, extra_logits)
        )


# ==============================================================================
# 8. AUXILIARY HEADS & HOMOSCEDASTIC LOSS WRAPPER WITH NULL-LOSS DETACH
# ==============================================================================


class HomoscedasticLossWrapper(nn.Module):
    r"""
    Homoscedastic Task Uncertainty Loss Weighting (Kendall & Gal, CVPR 2018).

    Architecture:
    In multi-task learning, balancing loss magnitudes (e.g. CTC vs. CrossEntropy vs. InfoNCE) is notoriously difficult.
    Instead of fixed scalar weights, we learn a parameter $s_i = \log(\sigma_i^2)$ representing the log-variance (uncertainty) of task $i$.

    Mathematical Formulation:
    $\mathcal{L}_{total} = \sum_i \left( \frac{\mathcal{L}_i}{e^{s_i}} + \frac{s_i}{2} \right)$

    As the network trains, it can dynamically down-weight "noisy" or "difficult" tasks by increasing $s_i$.
    The $+ \frac{s_i}{2}$ regularizer prevents the model from ignoring all tasks by setting $s_i \to \infty$.

    Additionally, this wrapper bypasses gradient propagation for zero-valued or uncalculated losses to prevent divergence.
    """

    def __init__(self, loss_config: Optional[Dict[str, float]] = None):
        """Initializes the module component."""

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
                "clr": 0.1,
                "domain": 1.0,
                "aux": 2.0,
                "length": 1.0,
                "mtp2": 4.0,
                "mtp3": 2.0,
                "inter_ctc": 1.0,
                "lpc": 1.0,
            }

        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(-math.log(2.0 * val_v), dtype=torch.float32))
                for name, val_v in loss_config.items()
            }
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for this module."""

        # Vectorized loss computation (1 fused kernel on MXU)
        keys = list(losses.keys())
        for name in keys:
            if name not in self.log_vars:
                raise ValueError(f"Unregistered loss key '{name}' produced by model!")

        loss_vec = torch.stack([losses[k].mean() for k in keys])
        s_vec = torch.stack([self.log_vars[k] for k in keys]).to(loss_vec.device)

        # Removed hard clamp to fix boundary gradient saturation (Claim 63)
        prec_vec = torch.exp(-s_vec)
        is_active = (loss_vec > 1e-7).float()  # Fix zero detection (Claim 64)

        # Added a small L2 penalty on s_vec for inactive tasks so they drift to 0 (Claim 65)
        s_val = s_vec
        prec_val = prec_vec

        task_loss = torch.where(
            is_active.bool(),
            (0.5 * prec_val * loss_vec + 0.5 * s_val),
            0.01 * (s_vec ** 2),  # Inactive tasks drift log_var towards 0 (Claim 65)
        )
        return task_loss.sum()


class CosineLinear(nn.Module):
    """Provides functionality for CosineLinear."""

    def __init__(self, in_features: int, out_features: int, init_tau: float = 20.0, actual_vocab_size: int = None):
        """Initializes the module component."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        import math
        inv_softplus_tau = math.log(max(1e-5, math.exp(max(1.001, init_tau) - 1.0) - 1.0))
        self.tau = nn.Parameter(torch.tensor(inv_softplus_tau))
        self.actual_vocab_size = actual_vocab_size

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        # L2 normalize features and weights
        """Forward pass for this module."""

        x_norm = F.normalize(input_x.float(), p=2, dim=-1, eps=1e-5).to(input_x.dtype)
        w_norm = F.normalize(self.weight.float(), p=2, dim=-1, eps=1e-5).to(
            input_x.dtype
        )
        # Cosine similarity scaled by learnable temperature tau
        safe_tau = (F.softplus(self.tau) + 1.0).to(input_x.dtype)
        logits = F.linear(x_norm, w_norm) * safe_tau
        if self.actual_vocab_size is not None and logits.shape[-1] > self.actual_vocab_size:
            logits[..., self.actual_vocab_size:] = -65500.0
        return logits


class CTCHead(nn.Module):
    """Prediction head for CTCHead."""

    def __init__(self, d_model: int, vocab_size: int, actual_vocab_size: int = None):
        """Initializes the module component."""

        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        self.actual_vocab_size = actual_vocab_size

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass for this module."""

        if getattr(self, "debug_xla", False) and torch.isnan(enc_seq).any():
            print("enc_seq has NaNs!")
        logits = self.proj(enc_seq)
        if self.actual_vocab_size is not None and logits.shape[-1] > self.actual_vocab_size:
            logits[..., self.actual_vocab_size:] = -65500.0
        return F.log_softmax(logits, dim=-1)


class CrossModalInfoNCELoss(nn.Module):
    r"""
    CrossModalInfoNCE: Alignment of visual (sign language) and textual (gloss/sentence) representations.

    Architecture:
    Projects visual embeddings $v \in \mathbb{R}^D$ and textual embeddings $t \in \mathbb{R}^D$ into a shared latent space.
    Computes a scaled symmetric contrastive loss to maximize mutual information between aligned pairs.

    Mathematical Formulation:
    Let $V$ be a batch of visual vectors and $T$ be a batch of text vectors, both $L_2$-normalized.
    The similarity matrix is $S = V T^\top \\cdot \exp(\tau)$, where $\tau$ is a learnable log-temperature parameter.

    The symmetric loss is:
    $\\mathcal{L}_{V2T} = -\frac{1}{N} \sum_{i=1}^N \\log \frac{\exp(S_{i,i})}{\sum_{j=1}^N \exp(S_{i,j})}$
    $\\mathcal{L}_{T2V} = -\frac{1}{N} \sum_{i=1}^N \\log \frac{\exp(S_{i,i})}{\sum_{j=1}^N \exp(S_{j,i})}$
    $\\mathcal{L}_{InfoNCE} = \frac{1}{2} (\\mathcal{L}_{V2T} + \\mathcal{L}_{T2V})$
    """

    def __init__(self, init_temp: float = 0.07, **kwargs):
        """Initializes the module component."""

        super().__init__()
        target_sp = init_temp - 0.05
        # Clamp to avoid math domain errors: log(exp(x) - 1) requires exp(x) > 1, i.e. x > 0
        self.log_temp = nn.Parameter(
            torch.tensor(math.log(math.exp(max(1e-5, target_sp)) - 1.0))
        )
        self.pad_val = -10.0

    def forward(
        self,
        vis_emb: torch.Tensor,
        sent_emb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        gt_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for this module."""

        device = vis_emb.device
        import torch.distributed as dist

        if IS_TPU and "xla" in str(device).lower():
            import torch_xla.core.xla_model as xm

            world_size = get_xla_world_size()
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        val_v = F.normalize(vis_emb.float(), p=2, dim=-1, eps=1e-8)
        s = F.normalize(sent_emb.float(), p=2, dim=-1, eps=1e-8)

        if world_size > 1:
            to_gather = [s, val_v]
            if valid_mask is not None:
                to_gather.append(valid_mask.float().unsqueeze(1))
            if gt_tokens is not None:
                to_gather.append(gt_tokens.float())
                
            fused_all = torch.cat(to_gather, dim=1)
            
            if IS_TPU and "xla" in str(device).lower():
                fused_gathered = xm.all_gather(fused_all)
            elif dist.is_initialized():
                gathered_list = [torch.zeros_like(fused_all) for _ in range(world_size)]
                dist.all_gather(gathered_list, fused_all)
                fused_gathered = torch.cat(gathered_list, dim=0)
            else:
                fused_gathered = fused_all

            s_all = fused_gathered[:, : s.shape[1]]
            offset = s.shape[1]
            v_all = fused_gathered[:, offset : offset + val_v.shape[1]]
            offset += val_v.shape[1]
            
            if valid_mask is not None:
                valid_mask_all = fused_gathered[:, offset].bool()
                offset += 1
            else:
                valid_mask_all = None
                
            if gt_tokens is not None:
                gt_tokens_all = fused_gathered[:, offset:].long()
            else:
                gt_tokens_all = None

        else:
            v_all = val_v
            s_all = s
            valid_mask_all = valid_mask.bool() if valid_mask is not None else None
            gt_tokens_all = gt_tokens

        if val_v.size(0) == 0:
            return torch.zeros((), device=device)

        temp = F.softplus(self.log_temp) + 0.05
        # logits_v2s shape: [batch_sz, batch_sz]
        logits_v2s = torch.matmul(val_v, s_all.transpose(-1, -2)) / temp
        logits_s2v = torch.matmul(s, v_all.transpose(-1, -2)) / temp

        if valid_mask_all is not None:
            # fused_mask_gt nor valid_mask is provided. Guard ~None with bool check.
            invalid_candidate_mask = (
                ~valid_mask_all
            )  # safe: wrapped under `if valid_mask_all is not None:`
            invalid_row_mask = (
                ~valid_mask
                if valid_mask is not None
                else ~valid_mask_all[: val_v.size(0)]
            )

            logits_v2s = logits_v2s.masked_fill(
                invalid_candidate_mask.unsqueeze(0), -1e9
            ).masked_fill(invalid_row_mask.unsqueeze(1), -1e9)
            logits_s2v = logits_s2v.masked_fill(
                invalid_candidate_mask.unsqueeze(0), -1e9
            ).masked_fill(invalid_row_mask.unsqueeze(1), -1e9)

        rank_val = 0
        if IS_TPU and "xla" in str(device).lower():
            try:
                import torch_xla.runtime as xr

                rank_val = xr.global_ordinal()
            except Exception:
                try:
                    import torch_xla.core.xla_model as xm

                    rank_val = getattr(xm, "get_ordinal", lambda: 0)()
                except Exception:
                    rank_val = 0
        elif dist.is_initialized():
            rank_val = dist.get_rank()

        global_local_rows = rank_val * val_v.size(0) + torch.arange(
            val_v.size(0), device=val_v.device
        )
        labels_all = torch.arange(s_all.size(0), device=val_v.device)
        self_mask = global_local_rows.unsqueeze(1) == labels_all.unsqueeze(0)

        if gt_tokens_all is not None:
            # Ignore 100% padded rows from matching as false positives across replicas
            valid_text_rows = (gt_tokens != 0).any(dim=-1).unsqueeze(1)
            valid_text_all = (gt_tokens_all != 0).any(dim=-1).unsqueeze(0)
            pos_mask = (
                (gt_tokens.unsqueeze(1) == gt_tokens_all.unsqueeze(0))
                .all(dim=-1)
                .float()
                * valid_text_rows.float()
                * valid_text_all.float()
            )
        else:
            pos_mask = self_mask.float()

        if valid_mask is not None:
            valid_rows = valid_mask.float()
            pos_mask = pos_mask * valid_rows.unsqueeze(1)
            if valid_mask_all is not None:
                pos_mask = pos_mask * valid_mask_all.float().unsqueeze(0)
        else:
            valid_rows = torch.ones(val_v.shape[0], device=val_v.device)

        # Fused log_softmax on TPU MXU (1 fused kernel instead of 6 un-fused ops)
        log_prob_v2s = F.log_softmax(logits_v2s, dim=-1)
        log_prob_s2v = F.log_softmax(logits_s2v, dim=-1)

        actual_pos_count = pos_mask.sum(dim=-1)
        pos_count = actual_pos_count.clamp(min=1.0)
        loss_v2s = -(log_prob_v2s * pos_mask).sum(dim=-1) / pos_count
        loss_s2v = -(log_prob_s2v * pos_mask).sum(dim=-1) / pos_count

        loss = 0.5 * (loss_v2s + loss_s2v)

        # E64/E65: Mask out rows without any valid positives so they don't artificially drag down the batch mean
        has_positives = (actual_pos_count > 0).float()
        valid_rows = valid_rows * has_positives

        if sample_weights is not None:
            loss = loss * sample_weights

        weight_sum = valid_rows
        loss = loss * valid_rows

        if sample_weights is not None:
            weight_sum = weight_sum * sample_weights

        res = _distributed_normalize(loss.float().sum(), weight_sum.float().sum())

        return res


class DenseSentenceSemanticLoss(nn.Module):
    """Computes the DenseSentenceSemanticLoss criterion."""

    def __init__(self, d_model: int = 512, embed_dim: int = 256):
        """Initializes the module component."""

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
        for prob_p in self.proj_gt.parameters():
            prob_p.requires_grad = False
        self.proj_gt.load_state_dict(self.proj_pred.state_dict())

    def update_momentum(self, mask_m=0.01):
        """Provides functionality for update_momentum."""

        with torch.no_grad():
            for p_target, p_online in zip(
                self.proj_gt.parameters(), self.proj_pred.parameters()
            ):
                p_target.lerp_(p_online.detach(), weight=mask_m)

    def forward(
        self,
        last_hidden: torch.Tensor,
        gt_lex_embs: torch.Tensor,
        valid_mask: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for this module."""

        mask_m = valid_mask.unsqueeze(-1).float()
        valid_counts = mask_m.sum(dim=1).clamp(min=1.0)
        has_tokens = (mask_m.sum(dim=(1, 2)) > 0).float()

        if last_hidden.ndim == 2:
            pred_sent = last_hidden
        else:
            pred_sent = (last_hidden * mask_m).sum(dim=1) / valid_counts

        gt_sent = (gt_lex_embs * mask_m).sum(dim=1) / valid_counts

        prob_p = F.normalize(self.proj_pred(pred_sent).float(), p=2, dim=-1, eps=1e-8)
        with torch.no_grad():
            g = F.normalize(
                self.proj_gt(gt_sent).float(), p=2, dim=-1, eps=1e-8
            ).detach()

        cos_sim = (prob_p * g).sum(dim=-1)
        cos_loss = (1.0 - cos_sim) * has_tokens

        # Static masked variance (No boolean indexing, zero graph breaks)
        token_weights = has_tokens.unsqueeze(-1)  # [B, 1]
        denom = token_weights.sum(dim=0).clamp(min=1.0)
        mean_p = (prob_p * token_weights).sum(dim=0, keepdim=True) / denom
        var_p = (((prob_p - mean_p) ** 2) * token_weights).sum(dim=0) / denom
        std_p = torch.sqrt(var_p + 1e-4)

        target_std = 1.0 / math.sqrt(prob_p.shape[-1])
        # Only apply std_loss if there are at least 2 valid samples in the batch to compute meaningful variance
        std_loss = torch.mean(F.relu(target_std - std_p)) * (denom.squeeze() > 1.5).float()
        loss = (cos_loss + 0.5 * std_loss) * has_tokens

        weight_sum = has_tokens
        if sample_weights is not None:
            loss = loss * sample_weights
            weight_sum = weight_sum * sample_weights

        return _distributed_normalize(loss.float().sum(), weight_sum.float().sum())


class SupervisedContrastiveLoss(nn.Module):
    r"""
    Supervised Contrastive Loss (InfoNCE formulation).

    Architecture:
    This loss encourages embeddings of the same class (or domain) to cluster tightly in a unit-hypersphere,
    while repelling embeddings of different classes.

    Mathematical Formulation:
    Let $z_i \\in \\mathbb{R}^D$ be an $L_2$-normalized feature vector for anchor $i$, and $y_i$ be its label.
    Let $A(i) \equiv \{j : y_j = y_i, j \neq i\}$ be the set of indices for positive samples.
    Let $P(i) \equiv \{k : k \neq i\}$ be the set of all other indices (positives + negatives).

    $\\mathcal{L}_{sup}^{out} = \sum_{i \\in I} \frac{-1}{|A(i)|} \sum_{j \\in A(i)} \\log \frac{\exp(z_i \\cdot z_j / \tau)}{\sum_{k \\in P(i)} \exp(z_i \\cdot z_k / \tau)}$

    The temperature scaling parameter $\tau$ (typically 0.1) controls the penalty sharpness for hard negatives.
    """

    def __init__(self, temperature: float = 0.07, **kwargs):
        """Initializes the module component."""

        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor = None,
        enqueue: bool = True,
        sample_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for this module."""

        if labels is None:
            return torch.zeros((), device=features.device)
        features = F.normalize(features.float(), p=2, dim=1, eps=1e-5)
        device = features.device

        import torch.distributed as dist

        if IS_TPU and "xla" in str(device).lower():
            import torch_xla.core.xla_model as xm

            world_size = get_xla_world_size()
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        sample_weight = (
            sample_weights
            if sample_weights is not None
            else torch.ones_like(labels, dtype=features.dtype)
        )

        if world_size > 1 and IS_TPU and "xla" in str(device).lower():
            if sample_ids is not None:
                fused = torch.cat(
                    [
                        features,
                        labels.unsqueeze(1).float(),
                        sample_ids.unsqueeze(1).float(),
                        sample_weight.unsqueeze(1).float(),
                    ],
                    dim=1,
                )
                fused_all = xm.all_gather(fused)
                all_feats = fused_all[:, :-3]
                all_labels = fused_all[:, -3].long()
                all_sample_ids = fused_all[:, -2].long()
                all_weights = fused_all[:, -1].float()
            else:
                fused = torch.cat([features, labels.unsqueeze(1).float(), sample_weight.unsqueeze(1).float()], dim=1)
                fused_all = xm.all_gather(fused)
                all_feats = fused_all[:, :-2]
                all_labels = fused_all[:, -2].long()
                all_weights = fused_all[:, -1].float()
                all_sample_ids = None
        else:
            all_feats = features
            all_labels = labels
            all_sample_ids = sample_ids
            all_weights = sample_weight

        batch_sz = features.shape[0]
        pos_mask = torch.eq(labels.view(-1, 1), all_labels.view(1, -1)).float()
        valid_labels = (all_labels.view(1, -1) != -1).float()
        valid_weights = (all_weights.view(1, -1) > 0.0).float()
        pos_mask = pos_mask * valid_labels * valid_weights

        # Zero out self-pair matches so sample is not its own positive across all TPU ranks
        rank_val = 0
        if IS_TPU and "xla" in str(device).lower():
            try:
                import torch_xla.runtime as xr

                rank_val = xr.global_ordinal()
            except Exception:
                try:
                    import torch_xla.core.xla_model as xm

                    rank_val = getattr(xm, "get_ordinal", lambda: 0)()
                except Exception:
                    rank_val = 0
        elif dist.is_initialized():
            rank_val = dist.get_rank()

        # Fully static XLA-friendly self-masking
        if all_sample_ids is not None:
            is_self = sample_ids.unsqueeze(1) == all_sample_ids.unsqueeze(0)
        else:
            global_indices = torch.arange(all_feats.shape[0], device=device)
            local_indices = rank_val * batch_sz + torch.arange(batch_sz, device=device)
            # Broadcast to create a [batch_sz, global_B] boolean mask
            is_self = local_indices.unsqueeze(1) == global_indices.unsqueeze(0)

        pos_mask = torch.where(is_self, torch.zeros_like(pos_mask), pos_mask)

        pos_logits = torch.matmul(features.float(), all_feats.float().T) / float(
            self.temperature
        )
        # Mask self-similarity in denominator so exp(1.0/tau) = exp(14.28) does not suppress negative gradients
        pos_logits = torch.where(
            is_self, torch.full_like(pos_logits, -65500.0), pos_logits
        )
        exp_logits = torch.exp(pos_logits - pos_logits.max(dim=1, keepdim=True)[0])
        denom = torch.clamp(exp_logits.sum(dim=1, keepdim=True).float(), min=1e-4)

        log_prob = (pos_logits - pos_logits.max(dim=1, keepdim=True)[0]) - torch.log(
            denom
        )
        pos_count = pos_mask.sum(dim=1)
        valid_rows = (pos_count > 0).float()

        row_loss = -(log_prob * pos_mask).sum(dim=1) / pos_count.clamp(min=1.0)
        weight_sum = valid_rows
        loss_unweighted = row_loss * valid_rows
        if sample_weights is not None:
            loss_unweighted = loss_unweighted * sample_weights
            weight_sum = weight_sum * sample_weights

        return _distributed_normalize(
            loss_unweighted.float().sum(), weight_sum.float().sum()
        )


class GradientReversalFunction(torch.autograd.Function):
    """Provides functionality for GradientReversalFunction."""

    @staticmethod
    def forward(ctx, input_x: torch.Tensor, alpha: float = 1.0):
        """Forward pass for this module."""
        # Do not use float(alpha) as it triggers host-to-device sync when alpha is a tensor
        ctx.alpha = alpha
        return input_x.view_as(input_x)

    @staticmethod
    def backward(ctx, grad_output):
        """Provides functionality for backward."""

        return grad_output.neg() * ctx.alpha, None


class LandmarkReconstructionHead(nn.Module):
    """Prediction head for LandmarkReconstructionHead."""

    def __init__(self, d_model: int = 512, out_dim: int = 540):
        """Initializes the module component."""

        super().__init__()
        self.recon = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        """Forward pass for this module."""

        return self.recon(enc_seq)


# ==============================================================================
# 9. ASL FOUNDATION MODEL MAIN AGGREGATOR
# ==============================================================================


class PositionalEncoding1D(nn.Module):
    r"""
    PositionalEncoding1D: Adds absolute temporal position information to the sequence embeddings.

    Architecture:
    Uses sinusoidal functions of varying frequencies to encode sequence positions, allowing the model
    to extrapolate to sequence lengths longer than those encountered during training.

    Mathematical Formulation:
    For position $pos$ and dimension $i \in [0, d_{model}/2)$:
    $PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$
    $PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$

    The resulting embedding $PE \in \mathbb{R}^{T \times d_{model}}$ is added to the input sequence $X$.
    """

    def __init__(self, d_model: int, max_len: int = 4096):
        """Initializes the module component."""

        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

        # Precompute fallback for seq_len indexing
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(
        self, input_x: torch.Tensor, frame_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for this module."""

        if getattr(self, "scale_embeddings", False):
            input_x = input_x * math.sqrt(input_x.size(-1))

        if frame_indices is not None:
            position = frame_indices.float().unsqueeze(-1)
            pe_sin = torch.sin(position * self.div_term)
            pe_cos = torch.cos(position * self.div_term)
            pe = torch.stack([pe_sin, pe_cos], dim=-1).view(
                *position.shape[:-1], self.d_model
            )
            return input_x + pe.to(dtype=input_x.dtype)
        else:
            seq_len = input_x.size(1)
            return input_x + self.pe[:, :seq_len, :].to(dtype=input_x.dtype)

    def set_scale_embeddings(self, val: bool):
        self.scale_embeddings = val


def safe_norm(tensor, dim=-1, keepdim=False, eps=1e-6):
    """Provides functionality for safe_norm."""

    sq_norm = torch.sum(tensor**2, dim=dim, keepdim=keepdim)
    return torch.sqrt(sq_norm + eps) * (sq_norm > 0).to(tensor.dtype)

def fast_cross(a, b):
    return torch.stack([
        a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
        a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
        a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    ], dim=-1)


def safe_cosine_sim(v1, v2, eps=1e-5):
    """Provides functionality for safe_cosine_sim."""

    n1 = torch.norm(v1, dim=-1, keepdim=True).clamp(min=eps)
    n2 = torch.norm(v2, dim=-1, keepdim=True).clamp(min=eps)
    return (v1 * v2).sum(dim=-1) / (n1 * n2).squeeze(-1)

def xla_clip_grad_norm_(parameters, max_norm, norm_type=2.0):
    # (Claim 87 & Claim 15) Custom grad clipping without device-to-host sync on TPU, standard clip_grad_norm_ on GPU/CPU
    if IS_TPU:
        parameters = [p for p in parameters if p.grad is not None]
        if not parameters:
            return torch.tensor(0.0)
        
        device = parameters[0].grad.device
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        for p in parameters:
            p.grad.detach().mul_(clip_coef_clamped)
        return total_norm
    else:
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

class ASLFoundationModel(nn.Module):
    r"""
    ASLFoundationModel: The main model orchestrating the sequence processing pipeline for ASL translation.

    Architecture:
    1. Landmark Stem: Projects raw 3D coordinate inputs ($T \times K \times C$) into a continuous embedding sequence.
    2. Encoder (MobileConformer / Mamba): Captures deep temporal and spatial dynamics of the sign features.
    3. Multi-task Heads:
       - Connectionist Temporal Classification (CTC): For gloss-level temporal alignment.
       - Transformer Decoder: For sequence-to-sequence translation (Sign to Text).
       - CrossModal InfoNCE: To enforce semantic alignment between the encoder representation and target text.
       - Multi-Token Prediction (MTP): Auxiliary prediction heads to encourage forward planning.

    Mathematical Objective:
    $\\mathcal{L} = w_{ctc}\\mathcal{L}_{CTC} + w_{ce}\\mathcal{L}_{CE} + w_{nce}\\mathcal{L}_{InfoNCE} + \sum_{k} w_{mtp,k}\\mathcal{L}_{MTP,k}$
    Optimized dynamically using the HomoscedasticLossWrapper.
    """

    def __init__(
        self,
        vocab_size: int = 2484,
        english_vocab_size: int = 20005,
        enable_aux_decoders: bool = False,
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
        eng_pad_id: int = 0,
        eng_bos_id: int = 1,
        eng_eos_id: int = 2,
        use_mamba: bool = True,
        tome_r: int = 80,
        scale_embeddings: bool = True,
        use_swin_1d: bool = False,
        swin_window_size: int = 128,
        is_causal: bool = False,
    ):
        """Initializes the module component."""

        super().__init__()
        self.use_swin_1d = use_swin_1d
        self.swin_window_size = swin_window_size
        self.actual_vocab_size = vocab_size
        self.actual_english_vocab_size = english_vocab_size
        # Pad vocabularies to multiples of 128 for TPU MXU alignment (dramatically improves TFLOPs utilization)
        self.vocab_size = (vocab_size + 127) // 128 * 128
        english_vocab_size = (english_vocab_size + 127) // 128 * 128

        self.d_enc = d_enc
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.use_mamba = use_mamba
        self.tome_r = tome_r
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.scale_embeddings = scale_embeddings
        self.enable_aux_decoders = enable_aux_decoders

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_enc) * 0.02)
        self.is_causal = is_causal
        self.visual_encoder = LandmarkTrajectory1DStem(
            in_channels=channels_per_kp, num_keypoints=num_keypoints, out_dim=128, is_causal=is_causal
        )
        self.phonology_norm = RMSNorm(19)
        # Change input stem to expect 768 perfectly aligned dimensions (640 padded + 128)
        self.input_stem = nn.Sequential(
            nn.Linear(768, d_enc), RMSNorm(d_enc), nn.GELU()
        )
        dpr = (
            [
                input_x.item()
                for input_x in torch.linspace(0.0, drop_path_rate, num_enc_layers)
            ]
            if num_enc_layers > 0
            else []
        )

        self.blocks = nn.ModuleList()
        for i in range(num_enc_layers):
            if i == num_enc_layers // 2:
                # Physically halve the tensor midway through the network
                self.blocks.append(TemporalStridedPool(is_causal=self.is_causal))

            if use_mamba and i >= 4:
                self.blocks.append(
                    BiMamba2SSMBlock(
                        d_model=d_enc, expand=2, ffn_dim=ffn_enc, drop_path=dpr[i], is_causal=self.is_causal
                    )
                )
            else:
                self.blocks.append(
                    MobileConformerBlock(
                        d_model=d_enc,
                        nhead=nhead_enc,
                        dim_feedforward=ffn_enc,
                        drop_path=dpr[i],
                        max_len=max_enc_len + 1,
                        use_swin=getattr(self, "use_swin_1d", False),
                        window_size=getattr(self, "swin_window_size", 128),
                        shift_size=(getattr(self, "swin_window_size", 128) // 2) if (i % 2 == 1) else 0,
                        is_causal=self.is_causal,
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

        if getattr(self, "enable_aux_decoders", False):
            self.chicago_decoder = ASLTransformerDecoder(
                vocab_size=42,  # Chicago chars
                d_model=d_dec,
                nhead=nhead_dec,
                kv_heads=kv_heads_dec,
                num_layers=max(2, num_dec_layers // 2),
                ffn_dim=ffn_dec,
                dropout=dropout,
                max_seq_len=256,
                csv_path=None,
                label_to_idx=None,
                use_asl_lex=False,
            )
            english_max_len = max(max_dec_len, 384)
            self.english_decoder = ASLTransformerDecoder(
                vocab_size=max(4, english_vocab_size),  # English vocab
                d_model=d_dec,
                nhead=nhead_dec,
                kv_heads=kv_heads_dec,
                num_layers=num_dec_layers,
                ffn_dim=ffn_dec,
                dropout=dropout,
                max_seq_len=english_max_len,
                csv_path=None,
                label_to_idx=None,
                use_asl_lex=False,
            )
        else:
            self.chicago_decoder = None
            self.english_decoder = None

        self.chicago_length_head = nn.Sequential(
            nn.Linear(d_enc, 128), RMSNorm(128), nn.GELU(), nn.Linear(128, 1)
        )
        self.english_length_head = nn.Sequential(
            nn.Linear(d_enc, 128), RMSNorm(128), nn.GELU(), nn.Linear(128, 1)
        )

        self.time_emb = PositionalEncoding1D(d_enc, max_len=4096)
        self.time_emb.set_scale_embeddings(self.scale_embeddings)

        self.ctc_head = CTCHead(d_enc, self.vocab_size, actual_vocab_size=self.actual_vocab_size)
        self.inter_ctc_head = CTCHead(d_enc, self.vocab_size, actual_vocab_size=self.actual_vocab_size)
        self.lpc_proj = nn.Sequential(nn.Linear(d_enc, d_enc), RMSNorm(d_enc))
        self.mlm_head = LandmarkReconstructionHead(d_enc, 540)
        self.domain_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(d_enc, num_domains)
        )

        # ─── MATH FIX: Encoder Auxiliary Classification Head ───
        # Mathematically forces the Conformer to anchor the latent space into a discrete conceptual cluster
        # BEFORE giving the sequence to the decoder. Bypasses decoder hallucination drift.
        self.aux_gloss_head = CosineLinear(d_enc, self.vocab_size, init_tau=2.65, actual_vocab_size=self.actual_vocab_size)

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
        self.xmodal_loss_fn = CrossModalInfoNCELoss(init_temp=0.07)
        self.dense_sem_loss = DenseSentenceSemanticLoss(d_model=d_dec, embed_dim=256)

        nn.init.normal_(self.decoder.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.decoder.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)
            if self.chicago_decoder is not None:
                self.chicago_decoder.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)
            if self.english_decoder is not None:
                self.english_decoder.token_emb.weight[eng_pad_id].fill_(0)

    def update_tome_r(self, epoch: int, max_epochs: int):
        # [TPU XLA HOTFIX] ToMe changes the tensor sequence length (e.g., num_n -> num_n-r),
        # which forces XLA to compile a brand new static graph in device memory.
        # Instead of increasing `r` every single epoch (which causes 70+ recompilations
        # and guaranteed HBM OOM), we "bucket" `r` into 4 distinct stages.
        # This gives us the dynamic ToMe effect but only compiles 4 graphs total!
        # [NEW HOTFIX] On TPU, we fix the ratio to 30 completely to avoid graph breaks.
        """Provides functionality for update_tome_r."""

        new_r = 30  # Locked to 30 for static graph stability (Claims 51-53)

        if getattr(self, "tome_r", -1) == new_r:
            return  # No change, avoid unnecessary assignment

        self.tome_r = new_r
        for block in self.blocks:
            if isinstance(block, TemporalStridedPool):
                block.r = new_r

    def _encode(
        self,
        input_x: torch.Tensor,
        phonology_features: torch.Tensor,
        mask: Optional[torch.Tensor],
        mlm_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """Internal helper method _encode."""

        if (
            input_x.is_floating_point()
            and hasattr(self, "visual_encoder")
            and hasattr(self.visual_encoder, "conv1")
            and input_x.dtype != self.visual_encoder.conv1.weight.dtype
        ):
            input_x = input_x.to(self.visual_encoder.conv1.weight.dtype)
        if (
            phonology_features is not None
            and phonology_features.is_floating_point()
            and hasattr(self, "visual_encoder")
            and hasattr(self.visual_encoder, "conv1")
            and phonology_features.dtype != self.visual_encoder.conv1.weight.dtype
        ):
            phonology_features = phonology_features.to(
                self.visual_encoder.conv1.weight.dtype
            )

        batch_sz, seq_len = input_x.size(0), input_x.size(1)
        inter_h = None
        inter_idx = (
            next(
                (
                    i
                    for i, b in enumerate(self.blocks)
                    if isinstance(b, TemporalStridedPool)
                ),
                len(self.blocks) // 2,
            )
            + 1  # Capture *after* pooling (Claims 54-55)
        )
        if inter_idx < 0:
            inter_idx = 0

        if mlm_mask is not None:
            used_mlm_mask = mlm_mask
            mask_shape = [1] * (input_x.dim() - 2)
            x_in = input_x * (~mlm_mask).view(batch_sz, seq_len, *mask_shape).to(
                input_x.dtype
            )
        else:
            x_in = input_x
            used_mlm_mask = None

        if x_in.dim() == 4 and x_in.size(2) == 60 and x_in.size(3) >= 3:
            xk = x_in
            x_flat = x_in.reshape(batch_sz, seq_len, -1)
            v_tokens = self.visual_encoder(xk, mask=mask, cache=cache)
        else:
            x_flat = x_in.reshape(batch_sz, seq_len, -1) if x_in.dim() == 4 else x_in
            v_tokens = self.visual_encoder(x_in, mask=mask, cache=cache)

        phonology_features = self.phonology_norm(phonology_features)
        x_enriched = torch.cat([x_flat, phonology_features], dim=-1)
        padding_needed = 640 - x_enriched.size(-1)
        x_padded = F.pad(x_enriched, (0, padding_needed), value=0)

        hidden_h = self.input_stem(torch.cat([x_padded, v_tokens], dim=-1))
        if not self.is_causal:
            hidden_h = torch.cat([self.cls_token.expand(batch_sz, -1, -1), hidden_h], dim=1)
            
            if frame_indices is not None:
                cls_fi = torch.zeros(
                    (batch_sz, 1), dtype=frame_indices.dtype, device=frame_indices.device
                )
                fi_padded = torch.cat([cls_fi, frame_indices], dim=1)
            else:
                fi_padded = None
        else:
            fi_padded = frame_indices
            
        hidden_h = self.time_emb(hidden_h, frame_indices=fi_padded)

        routing_fi = frame_indices.long() if frame_indices is not None else None

        cur_mask = mask
        if cur_mask is not None:
            if not self.is_causal:
                kpm = torch.cat(
                    [
                        torch.zeros(
                            (batch_sz, 1), dtype=torch.bool, device=hidden_h.device
                        ),
                        ~cur_mask.bool(),
                    ],
                    dim=1,
                )
            else:
                kpm = ~cur_mask.bool()
        else:
            kpm = None

        token_sizes = torch.ones(
            batch_sz, seq_len, 1, device=hidden_h.device, dtype=hidden_h.dtype
        )

        for idx, block in enumerate(self.blocks):
            if isinstance(block, TemporalStridedPool):
                if not self.is_causal:
                    cls_t = hidden_h[:, :1]
                    seq_t = hidden_h[:, 1:]
                else:
                    seq_t = hidden_h
                    
                seq_t, cur_mask, routing_info = block(
                    seq_t,
                    cur_mask,
                    token_sizes=token_sizes,
                    mlm_mask=used_mlm_mask,
                    frame_indices=routing_fi,
                )
                if (
                    "token_sizes" in routing_info
                    and routing_info["token_sizes"] is not None
                ):
                    token_sizes = routing_info["token_sizes"]
                if "mlm_out" in routing_info and routing_info["mlm_out"] is not None:
                    used_mlm_mask = routing_info["mlm_out"]
                if (
                    "frame_indices" in routing_info
                    and routing_info["frame_indices"] is not None
                ):
                    routing_fi = routing_info["frame_indices"]
                    
                if not self.is_causal:
                    hidden_h = torch.cat([cls_t, seq_t], dim=1)
                    if cur_mask is not None:
                        kpm = torch.cat(
                            [
                                torch.zeros(
                                    (batch_sz, 1), dtype=torch.bool, device=hidden_h.device
                                ),
                                ~cur_mask.bool(),
                            ],
                            dim=1,
                        )
                    else:
                        kpm = None
                else:
                    hidden_h = seq_t
                    if cur_mask is not None:
                        kpm = ~cur_mask.bool()
                    else:
                        kpm = None
            else:
                if routing_fi is not None:
                    if not self.is_causal:
                        cls_fi = torch.zeros(
                            (batch_sz, 1), dtype=routing_fi.dtype, device=routing_fi.device
                        )
                        pos_fi = torch.cat([cls_fi, routing_fi + 1], dim=1)
                    else:
                        pos_fi = routing_fi + 1
                else:
                    pos_fi = None
                hidden_h = block(hidden_h, key_padding_mask=kpm, frame_indices=pos_fi, cache=cache)
                if idx == inter_idx:
                    if not self.is_causal:
                        inter_h = hidden_h[:, 1:]
                    else:
                        inter_h = hidden_h

            if getattr(self, "debug_xla", False) and torch.isnan(hidden_h).any():
                print(f"NaN introduced at block {idx}!")
                break

        hidden_h = self.enc_final_norm(hidden_h)
        if getattr(self, "debug_xla", False) and torch.isnan(hidden_h).any():
            print("NaN introduced at enc_final_norm!")

        # Extract cls_out
        if not self.is_causal:
            cls_out = hidden_h[:, 0]
            seq_out = hidden_h[:, 1:]
        else:
            seq_out = hidden_h
            if cur_mask is not None:
                # cur_mask is True for valid, False for pad. Get last valid index.
                # Shape is (batch_sz, seq_len)
                valid_lens = cur_mask.sum(dim=1).long() - 1 # 0-indexed
                valid_lens = valid_lens.clamp(min=0)
                # Gather last valid frame for each item in batch
                cls_out = seq_out[torch.arange(batch_sz, device=seq_out.device), valid_lens]
            else:
                cls_out = seq_out[:, -1]

        return (
            cls_out,
            seq_out,
            cur_mask,
            used_mlm_mask,
            routing_fi,
            mask,
            inter_h,
            token_sizes,
        )

    def forward(
        self,
        input_x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        gloss_seq: Optional[torch.Tensor] = None,
        chicago_seq: Optional[torch.Tensor] = None,
        english_seq: Optional[torch.Tensor] = None,
        mlm_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        grl_alpha: float = 1.0,
        compute_mlm: bool = True,
        compute_lpc: bool = True,
        cache: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Union[Optional[torch.Tensor], Dict]:
        # Always compute kinematics and augmentations on TPU to avoid Host CPU bottleneck
        """Forward pass for this module."""

        if (
            input_x.is_floating_point()
            and hasattr(self, "visual_encoder")
            and hasattr(self.visual_encoder, "conv1")
            and input_x.dtype != self.visual_encoder.conv1.weight.dtype
        ):
            input_x = input_x.to(self.visual_encoder.conv1.weight.dtype)

        if True:
            # E67: Prevent in-place modifications corrupting autograd graphs
            x = input_x
            batch_sz, seq_len, key_k, channels = x.shape

            # E68: Removed redundant spatial augmentations here since ASLDataset already handles
            # robust scale, shift, and rotate in its transform pipeline on CPU.

            dt = torch.ones(batch_sz, seq_len, 1, 1, device=x.device)
            if frame_indices is not None and seq_len > 1:
                actual_dt = (
                    (frame_indices[:, 1:] - frame_indices[:, :-1])
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                dt[:, 1:] = torch.where(
                    actual_dt == 0, torch.ones_like(actual_dt), actual_dt
                )

            # E69: Ensure kinematics are only derived from the [x, y, z] spatial coordinates
            pos = x[..., :3]
            if channels >= 9:
                vel = x[..., 3:6]
                acc = x[..., 6:9]
            else:
                vel = torch.zeros_like(pos)
                acc = torch.zeros_like(pos)

                if seq_len > 1:
                    vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / dt[:, 1:]
                    vel[:, 0] = vel[:, 1]
                    acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]
                    acc[:, 0] = acc[:, 1]

        if True:
            # Ensure downstream phonology features have access to the base coordinates

            # ─── MATH FIX: Preserve Unmasked Input for MLM Target (Defect #52) ───
            orig_input_x = torch.cat([pos, vel, acc], dim=-1)
            if orig_input_x.shape[-1] < self.channels_per_kp:
                pad = torch.zeros(
                    batch_sz,
                    seq_len,
                    key_k,
                    max(0, self.channels_per_kp - orig_input_x.shape[-1]),
                    device=orig_input_x.device,
                )
                orig_input_x = torch.cat([orig_input_x, pad], dim=-1)
            orig_input_x = orig_input_x[..., : self.channels_per_kp]

            # --- NEW: ASL Phonology Feature Pack (19 Dims) ---
            if mlm_mask is not None:
                mask_shape = [1] * (pos.dim() - 2)
                bool_mlm = ~mlm_mask
                bool_mlm_expanded = bool_mlm.view(batch_sz, seq_len, *mask_shape).to(
                    pos.dtype
                )
                pos = pos * bool_mlm_expanded
                vel = vel * bool_mlm_expanded
                acc = acc * bool_mlm_expanded

            # 1. Palm Orientation Normals (6 Dims)
            lh_u = pos[:, :, 5, :3] - pos[:, :, 0, :3]
            lh_v = pos[:, :, 17, :3] - pos[:, :, 0, :3]
            lh_normal = F.normalize(
                fast_cross(lh_u, lh_v), p=2, dim=-1, eps=1e-5
            )

            rh_u = pos[:, :, 26, :3] - pos[:, :, 21, :3]
            rh_v = pos[:, :, 38, :3] - pos[:, :, 21, :3]
            rh_normal = F.normalize(
                fast_cross(rh_u, rh_v), p=2, dim=-1, eps=1e-5
            )

            # 2. Bimanual Synchrony (1 Dim)
            lh_vel = vel[:, :, 0]
            rh_vel = vel[:, :, 21]
            bimanual_sync = safe_cosine_sim(lh_vel, rh_vel).unsqueeze(-1)

            # 3. Location Anchoring to Face (2 Dims)
            # 0-20=Left Hand, 21-41=Right Hand, 42-47=Pose, 48-59=Face.
            # Use face centroid (mean of lips+eyes landmarks 48-59) for
            # a semantically meaningful hand-to-face distance (ASL phonology).
            if pos.shape[2] > 48:
                face_centroid = pos[:, :, 48:60].mean(dim=2)  # [B, T, 3]
            else:
                face_centroid = pos[:, :, 0:1].mean(dim=2) * 0.0
            lh_face_dist = safe_norm(pos[:, :, 0] - face_centroid, dim=-1, keepdim=True)
            rh_face_dist = safe_norm(
                pos[:, :, 21] - face_centroid, dim=-1, keepdim=True
            )

            # 4. Finger Curl / Aperture (10 Dims)
            lh_curl = safe_norm(
                pos[:, :, [4, 8, 12, 16, 20]] - pos[:, :, 0].unsqueeze(2), dim=-1
            )
            rh_curl = safe_norm(
                pos[:, :, [25, 29, 33, 37, 41]] - pos[:, :, 21].unsqueeze(2), dim=-1
            )

            phonology_features = torch.cat(
                [
                    lh_normal,
                    rh_normal,
                    bimanual_sync,
                    lh_face_dist,
                    rh_face_dist,
                    lh_curl,
                    rh_curl,
                ],
                dim=-1,
            )  # Shape: [batch_sz, seq_len, 19]

            input_x = torch.cat([pos, vel, acc], dim=-1)

            if input_x.shape[-1] < self.channels_per_kp:
                pad = torch.zeros(
                    batch_sz,
                    seq_len,
                    key_k,
                    max(0, self.channels_per_kp - input_x.shape[-1]),
                    device=input_x.device,
                )
                input_x = torch.cat([input_x, pad], dim=-1)
            input_x = input_x[..., : self.channels_per_kp]

        h_cls, h_seq, enc_mask, used_mlm_mask, fi_out, orig_enc_mask, inter_h, token_sizes = (
            self._encode(
                input_x,
                phonology_features,
                mask,
                mlm_mask=mlm_mask,
                frame_indices=frame_indices,
                cache=cache,
            )
        )

        dec_logits, dec_hidden = None, None
        chicago_logits, english_logits = None, None

        if gloss_seq is not None and self.decoder is not None:
            dec_pad_id = getattr(
                self.decoder.token_emb, "padding_idx", GlossVocabulary.PAD_ID
            )
            dec_padding_mask = gloss_seq == dec_pad_id
            dec_logits, dec_hidden, _, _ = decode_seq(
                self.decoder, gloss_seq, h_seq, enc_mask, dec_padding_mask
            )

        if chicago_seq is not None and self.chicago_decoder is not None:
            chicago_padding_mask = chicago_seq == GlossVocabulary.PAD_ID
            chicago_logits, _, _, _ = decode_seq(
                self.chicago_decoder, chicago_seq, h_seq, enc_mask, chicago_padding_mask
            )

        if english_seq is not None and self.english_decoder is not None:
            eng_pad_id = getattr(
                self.english_decoder.token_emb, "padding_idx", GlossVocabulary.PAD_ID
            )
            english_padding_mask = english_seq == eng_pad_id
            # scheduled sampling doesn't mask BPE special tokens using Gloss IDs.
            eng_bos = getattr(self.english_decoder, "bos_id", None)
            eng_eos = getattr(self.english_decoder, "eos_id", None)
            english_logits, _, _, _ = decode_seq(
                self.english_decoder,
                english_seq,
                h_seq,
                enc_mask,
                english_padding_mask,
                bos_idx=eng_bos,
                eos_idx=eng_eos,
            )

        # Compute output heads
        ctc_log_probs = self.ctc_head(h_seq)
        inter_ctc_log_probs = (
            self.inter_ctc_head(inter_h) if inter_h is not None else None
        )
        aux_logits = self.aux_gloss_head(h_cls)
        pred_len = self.length_head(h_cls).squeeze(-1)
        chicago_pred_len = self.chicago_length_head(h_cls).squeeze(-1)
        english_pred_len = self.english_length_head(h_cls).squeeze(-1)

        vis_emb = self.visual_proj(h_cls)
        proj_feats = self.contrastive_head(h_cls)
        domain_logits = self.domain_head(
            GradientReversalFunction.apply(
                h_cls, torch.tensor(grl_alpha, device=h_cls.device)
            )
        )
        if compute_mlm:
            mlm_logits = self.mlm_head(inter_h if inter_h is not None else h_seq)
        else:
            mlm_logits = None

        if compute_lpc:
            lpc_feats = self.lpc_proj(h_seq)
            if h_seq.shape[1] > 1:
                diff = lpc_feats[:, 1:] - lpc_feats[:, :-1]
                if enc_mask is not None:
                    # enc_mask: [B, T] -> True where valid
                    valid_mask_f = enc_mask.unsqueeze(-1).float()
                    valid_count = valid_mask_f.sum(dim=(0, 1)).clamp(min=2.0)
                    lpc_mean = (lpc_feats * valid_mask_f).sum(dim=(0, 1)) / valid_count
                    lpc_var = (((lpc_feats - lpc_mean) * valid_mask_f) ** 2).sum(
                        dim=(0, 1)
                    ) / valid_count
                    lpc_std = torch.sqrt(lpc_var + 1e-8).mean()
                else:
                    lpc_std = lpc_feats.std(dim=1).mean()
                if enc_mask is not None:
                    mask_valid = (
                        (enc_mask[:, :-1] & enc_mask[:, 1:]).unsqueeze(-1).float()
                    )
                    diff = diff * mask_valid
                    valid_diff_count = mask_valid.sum(dim=1).clamp(min=1.0)
                    diff_mean = ((diff**2).sum(dim=(1, 2)) / (valid_diff_count.squeeze(-1) * diff.shape[-1])).mean()
                else:
                    diff_mean = (diff**2).mean()
                loss_lpc = diff_mean  # Removed artificial std penalty (Claim 61)
                if enc_mask is not None:
                    seq_has_valid = (enc_mask.sum(dim=1) > 1).float().mean()
                    loss_lpc = loss_lpc * seq_has_valid
            else:
                loss_lpc = torch.tensor(0.0, device=h_seq.device)
        else:
            loss_lpc = torch.tensor(0.0, device=h_seq.device)

        sent_emb = None
        if gloss_seq is not None and self.decoder is not None:
            tgt = gloss_seq[:, 1:]  # [B, T]
            non_pad = (
                (tgt != GlossVocabulary.PAD_ID) & (tgt != GlossVocabulary.EOS_ID)
            ).long()
            valid_lens = non_pad.sum(dim=1).clamp(min=1)

            embedded_text = self.decoder.token_emb(tgt)  # [B, T, D]
            text_mask = non_pad.unsqueeze(-1)
            pooled_text = (embedded_text * text_mask).sum(dim=1) / valid_lens.unsqueeze(
                -1
            )
            sent_emb = self.sentence_proj(pooled_text)

        return {
            "h_cls": h_cls,
            "h_seq": h_seq,
            "enc_mask": enc_mask,
            "used_mlm_mask": used_mlm_mask,
            "mlm_mask": used_mlm_mask,
            "dec_logits": dec_logits,
            "dec_hidden": dec_hidden,
            "chicago_logits": chicago_logits,
            "english_logits": english_logits,
            "inter_h": inter_h,
            "fi_out": fi_out,
            "orig_enc_mask": orig_enc_mask,
            "orig_x": orig_input_x,
            "ctc_log_probs": ctc_log_probs,
            "inter_ctc_log_probs": inter_ctc_log_probs,
            "aux_logits": aux_logits,
            "pred_len": pred_len,
            "chicago_pred_len": chicago_pred_len,
            "english_pred_len": english_pred_len,
            "vis_emb": vis_emb,
            "sent_emb": sent_emb,
            "proj_feats": proj_feats,
            "domain_logits": domain_logits,
            "mlm_logits": mlm_logits,
            "loss_lpc": loss_lpc,
            "token_sizes": token_sizes,
        }

    @torch.no_grad()
    def generate(
        self,
        features: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        task: str = "gloss",
        frame_indices: Optional[torch.Tensor] = None,
        h_seq: Optional[torch.Tensor] = None,
        enc_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation method for inference/validation."""

        if task == "chicago":
            decoder_mod = self.chicago_decoder
        elif task == "english":
            decoder_mod = self.english_decoder
        else:
            decoder_mod = self.decoder
        # uses the correct vocabulary boundaries, not hardcoded Gloss IDs.
        if task == "english" and self.english_decoder is not None:
            bos_id = getattr(self.english_decoder, "bos_id", GlossVocabulary.BOS_ID)
            eos_id = getattr(self.english_decoder, "eos_id", GlossVocabulary.EOS_ID)
        elif task == "chicago" and self.chicago_decoder is not None:
            bos_id = getattr(self.chicago_decoder, "bos_id", GlossVocabulary.BOS_ID)
            eos_id = getattr(self.chicago_decoder, "eos_id", GlossVocabulary.EOS_ID)
        else:
            bos_id = GlossVocabulary.BOS_ID
            eos_id = GlossVocabulary.EOS_ID

        if h_seq is not None:
            batch_sz = h_seq.size(0)
            device = h_seq.device
        else:
            batch_sz = features.size(0)
            device = features.device

            features_3d = features[..., :3]
            batch_sz, seq_len, key_k, _ = features_3d.shape

            vel = torch.zeros_like(features_3d)
            acc = torch.zeros_like(features_3d)
            dt = torch.ones(batch_sz, seq_len, 1, 1, device=device)
            if frame_indices is not None and seq_len > 1:
                actual_dt = (
                    (frame_indices[:, 1:] - frame_indices[:, :-1])
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                dt[:, 1:] = torch.where(
                    actual_dt == 0, torch.ones_like(actual_dt), actual_dt
                )

            if seq_len > 1:
                vel[:, 1:] = (features_3d[:, 1:] - features_3d[:, :-1]) / dt[:, 1:]
                vel[:, 0] = vel[:, 1]
                acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]
                acc[:, 0] = acc[:, 1]

            lh_u = features_3d[:, :, 5] - features_3d[:, :, 0]
            lh_v = features_3d[:, :, 17] - features_3d[:, :, 0]
            lh_normal = F.normalize(
                fast_cross(lh_u, lh_v), p=2, dim=-1, eps=1e-5
            )

            rh_u = features_3d[:, :, 26] - features_3d[:, :, 21]
            rh_v = features_3d[:, :, 38] - features_3d[:, :, 21]
            rh_normal = F.normalize(
                fast_cross(rh_u, rh_v), p=2, dim=-1, eps=1e-5
            )
            bimanual_sync = safe_cosine_sim(vel[:, :, 0], vel[:, :, 21]).unsqueeze(-1)
            if features_3d.shape[2] > 48:
                face_centroid = features_3d[:, :, 48:60].mean(dim=2)  # [B, T, 3]
            else:
                face_centroid = features_3d[:, :, 0:1].mean(dim=2) * 0.0
            lh_face_dist = safe_norm(
                features_3d[:, :, 0] - face_centroid, dim=-1, keepdim=True
            )
            rh_face_dist = safe_norm(
                features_3d[:, :, 21] - face_centroid, dim=-1, keepdim=True
            )

            lh_curl = safe_norm(
                features_3d[:, :, [4, 8, 12, 16, 20]]
                - features_3d[:, :, 0].unsqueeze(2),
                dim=-1,
            )
            rh_curl = safe_norm(
                features_3d[:, :, [25, 29, 33, 37, 41]]
                - features_3d[:, :, 21].unsqueeze(2),
                dim=-1,
            )

            phonology_features = torch.cat(
                [
                    lh_normal,
                    rh_normal,
                    bimanual_sync,
                    lh_face_dist,
                    rh_face_dist,
                    lh_curl,
                    rh_curl,
                ],
                dim=-1,
            )

            input_x = torch.cat([features_3d, vel, acc], dim=-1)
            if input_x.shape[-1] < self.channels_per_kp:
                pad = torch.zeros(
                    batch_sz,
                    seq_len,
                    key_k,
                    max(0, self.channels_per_kp - input_x.shape[-1]),
                    device=device,
                )
                input_x = torch.cat([input_x, pad], dim=-1)
            input_x = input_x[..., : self.channels_per_kp]

            _, h_seq, enc_mask, _, _, _, _, _ = self._encode(
                input_x,
                phonology_features,
                mask,
                mlm_mask=None,
                frame_indices=frame_indices,
            )
        generated = torch.full(
            (batch_sz, max_new_tokens + 1),
            GlossVocabulary.PAD_ID,
            dtype=torch.long,
            device=device,
        )
        generated[:, 0] = bos_id
        finished = torch.zeros(batch_sz, dtype=torch.bool, device=device)

        # Pre-allocate static KV Caches
        kv_heads = decoder_mod.layers[0].self_attn.kv_heads
        head_dim = decoder_mod.layers[0].self_attn.head_dim
        num_layers = len(decoder_mod.layers)
        cache_dtype = h_seq.dtype if h_seq is not None else input_x.dtype
        # Allocate static caches if generating on XLA to prevent graph recompilation
        kv_caches = []
        for _ in range(num_layers):
            # Self-attention caches: (k_cache, v_cache, past_len)
            self_k = torch.zeros(
                (batch_sz, kv_heads, max_new_tokens, head_dim),
                dtype=cache_dtype,
                device=device,
            )
            self_v = torch.zeros(
                (batch_sz, kv_heads, max_new_tokens, head_dim),
                dtype=cache_dtype,
                device=device,
            )
            # Cross-attention computes statically once per sequence, no need to pre-allocate iteratively here
            kv_caches.append(((self_k, self_v, torch.tensor([0], device=device, dtype=torch.long)), None))

        # MTP layer caches removed for generation to prevent XLA cache-size-shrink graph recompilations

        for step in range(max_new_tokens):
            # Evaluate finished mask out-of-place (Bug 4 fix removes any `.all().item()` exit condition)
            if finished.all().item() if not getattr(self, "is_xla", False) else False:
                break

            tgt_in = generated[:, step : step + 1]
            dec_pad_id = getattr(
                decoder_mod.token_emb, "padding_idx", GlossVocabulary.PAD_ID
            )
            dec_padding_mask = tgt_in == dec_pad_id
            logits, _, _, kv_caches = decode_seq(
                decoder_mod,
                tgt_in,
                h_seq,
                enc_mask,
                dec_padding_mask,
                kv_caches=kv_caches,
                use_cache=True,
                shift_target=False,
            )
            next_token_logits = logits[:, -1, :]

            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, dec_pad_id), next_tokens
            )
            generated[:, step + 1] = next_tokens

            finished = finished | (next_tokens == eos_id)

        # Slice off BOS correctly if needed, or return raw
        return generated[:, : max_new_tokens + 1]


class ModelEMA:
    """Exponential Moving Average wrapper for model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay_base: float = 0.999,
        decay_max: float = 0.9999,
    ):
        """Initializes the module component."""

        self.decay_base = decay_base
        self.decay_max = decay_max
        self.shadow = {}
        self.backup = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.clone().detach()

    def update(self, model: nn.Module, progress: float = 0.0):
        """Provides functionality for update."""
        first_param = next(iter(model.parameters()))
        progress_t = torch.tensor(
            max(0.0, min(1.0, float(progress))),
            dtype=torch.float32,
            device=first_param.device,
        )
        decay = self.decay_base + (self.decay_max - self.decay_base) * progress_t

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Avoid .data which causes XLA memory leaks!
                    # Execute EMA natively on TPU
                    self.shadow[name].lerp_(param, 1.0 - decay)

    def apply_shadow(self, model: nn.Module):
        """Provides functionality for apply_shadow."""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Swap current parameters with EMA safely in place
                    temp = param.clone().detach()
                    param.copy_(self.shadow[name])
                    self.shadow[name].copy_(temp)

    def restore(self, model: nn.Module):
        """Restores original model parameters by swapping back from shadow storage."""
        self.apply_shadow(model)


def _get_optimizer_groups(
    model: nn.Module, loss_wrapper: nn.Module, weight_decay: float
):
    """Internal helper method _get_optimizer_groups."""

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


POLY1_EPS = 0.001  # Used in Poly1 focal loss computation


def _compute_poly1_loss(
    focal_weight,
    ce,
    p_target,
    valid_mask,
    tf,
    eos_id,
    is_seq_loss,
    sw=None,
    class_weights=None,
    punct_ids=None,
):
    # The polynomial term must be scaled by the focal weight so it doesn't dominate easy examples
    poly1 = focal_weight * (ce + POLY1_EPS * (1.0 - p_target))
    if valid_mask is not None:
        vf = valid_mask.reshape(-1).float()
    else:
        vf = torch.ones_like(tf, dtype=torch.float32)

    if is_seq_loss:
        vf = vf * (tf != eos_id).float()
    else:
        vf = vf * (tf == eos_id).float()

    if punct_ids is not None and len(punct_ids) > 0:
        is_punct = torch.zeros_like(tf, dtype=torch.bool)
        for pid in punct_ids:
            is_punct = is_punct | (tf == pid)
        # Apply 2.0x loss penalty weight on punctuation tokens (. ? !) to enforce period placement at boundaries
        vf = vf * torch.where(is_punct, 2.0, 1.0)

    if class_weights is not None:
        vf = vf * class_weights[tf]

    if sw is not None:
        vf = vf * sw

    return _distributed_normalize((poly1 * vf).float().sum(), vf.float().sum())


def compute_seq_and_eos_loss(
    logits_f,
    gt_ids,
    valid_mask_seq,
    valid_mask_eos,
    class_weights=None,
    sample_weights=None,
    gamma=2.0,
    label_smoothing=0.1,
    pad_id=GlossVocabulary.PAD_ID,
    eos_id=GlossVocabulary.EOS_ID,
    punct_ids=None,
):
    """Computes both sequence and EOS focal loss efficiently in one pass."""
    vocab_v = logits_f.shape[-1]
    lf = logits_f.reshape(-1, vocab_v)
    tf = gt_ids.reshape(-1)

    # 1. Compute single-pass Log Softmax
    log_probs = F.log_softmax(lf, dim=-1)
    
    # 2. Extract NLL for the target tokens
    nll = -log_probs.gather(dim=-1, index=tf.unsqueeze(-1)).squeeze(-1)
    nll = nll.masked_fill(tf == pad_id, 0.0)
    
    # 3. Compute unsmoothed (EOS) and smoothed (SEQ) analytically
    ce_unsmoothed = nll
    mean_log_prob = log_probs.mean(dim=-1)
    ce_smoothed_seq = (1.0 - label_smoothing) * nll - label_smoothing * mean_log_prob
    ce_smoothed_seq = ce_smoothed_seq.masked_fill(tf == pad_id, 0.0)

    # 3. Compute p_target for focal weights and Poly-1 regularization
    p_target = torch.exp(-ce_unsmoothed).clamp(min=1e-6, max=1.0)
    focal_weight = torch.pow(1.0 - p_target.detach(), gamma)

    sw = (
        sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
        if sample_weights is not None
        else None
    )

    loss_seq = _compute_poly1_loss(
        focal_weight,
        ce_smoothed_seq,
        p_target,
        valid_mask_seq,
        tf,
        eos_id,
        is_seq_loss=True,
        sw=sw,
        class_weights=class_weights,
        punct_ids=punct_ids,
    )

    loss_eos = _compute_poly1_loss(
        focal_weight,
        ce_unsmoothed,
        p_target,
        valid_mask_eos,
        tf,
        eos_id,
        is_seq_loss=False,
        sw=sw,
        class_weights=None,
        punct_ids=punct_ids,
    )

    return loss_seq, loss_eos


def decode_seq(
    decoder_module,
    gt_seq: torch.Tensor,
    encoder_out: torch.Tensor,
    encoder_padding_mask: Optional[torch.Tensor] = None,
    decoder_padding_mask: Optional[torch.Tensor] = None,
    kv_caches=None,
    use_cache: bool = False,
    shift_target: bool = True,
    bos_idx: Optional[int] = None,
    eos_idx: Optional[int] = None,
):
    """
    Safely executes decoder forward pass.
    """
    if shift_target:
        target_in = gt_seq[:, :-1]
    else:
        target_in = gt_seq

    kpm = encoder_padding_mask
    out = decoder_module(
        target_in,
        encoder_out,
        memory_key_padding_mask=kpm,
        past_key_values=kv_caches,
        use_cache=use_cache,
    )

    if isinstance(out, tuple):
        logits = out[0]
        hidden = out[1] if len(out) > 1 else None
        extra_logits = out[2] if len(out) > 2 else None
        new_kv_caches = out[3] if len(out) > 3 else None
    else:
        logits, hidden, extra_logits, new_kv_caches = out, None, None, None

    return logits, hidden, extra_logits, new_kv_caches


def _compute_ctc_loss_safe(
    ctc_log_probs: torch.Tensor,
    gloss_seq: torch.Tensor,
    gloss_len: torch.Tensor,
    enc_mask: torch.Tensor,
    has_valid_gloss: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Computes CTC Loss with XLA padding safety masks."""
    batch_sz, seq_len = ctc_log_probs.shape[:2]
    if enc_mask is not None:
        input_lengths = enc_mask.sum(dim=1).to(torch.int32)
    else:
        input_lengths = torch.full(
            (batch_sz,), seq_len, dtype=torch.int32, device=ctc_log_probs.device
        )

    # True target length without BOS/EOS
    target_lengths = torch.clamp(gloss_len.to(torch.int32) - 2, min=0)
    # Convert log_probs to [seq_len, batch_size, num_classes] format required by CTC
    ctc_log_probs_t = ctc_log_probs.transpose(0, 1)

    # Correct CTC minimum length: need extra blank between adjacent identical labels
    # min_len = L + number_of_adjacent_duplicate_pairs (standard CTC constraint)
    
    # Shift gloss_seq left by 1 to remove BOS_ID at index 0
    clean_targets = torch.roll(gloss_seq, shifts=-1, dims=1)
    
    gloss_flat = clean_targets  # [B, T]
    valid_shifted = torch.arange(clean_targets.size(1), device=clean_targets.device).unsqueeze(0) < target_lengths.unsqueeze(1)
    
    adjacent_dups = (
        ((gloss_flat[:, 1:] == gloss_flat[:, :-1]) & valid_shifted[:, :-1])
        .sum(dim=1)
        .to(torch.int32)
    )
    min_ctc_len = target_lengths + adjacent_dups
    valid_ctc = (
        (input_lengths >= min_ctc_len) & (target_lengths > 0) & has_valid_gloss.bool()
    )

    # Sanitize targets so positions beyond target_lengths do not contain invalid data
    # PyTorch CTC loss requires the 2D tensor to be padded with the blank index (PAD_ID = 0)
    clean_targets = torch.where(valid_shifted, clean_targets, torch.tensor(GlossVocabulary.PAD_ID, dtype=clean_targets.dtype, device=clean_targets.device))

    # Only clamp to actual sequence length — never inflate beyond what the encoder produced.
    # Invalid pairs (input_lengths < min_ctc_len) are already excluded by valid_ctc above.
    # Pre-filter invalid samples to save CTC compute (Claims 68-69)
    target_lengths = target_lengths * valid_ctc.to(torch.int32)
    input_lengths = input_lengths * valid_ctc.to(torch.int32)
    actual_seq_len = torch.full(
        (batch_sz,), seq_len, dtype=torch.int32, device=ctc_log_probs.device
    )
    input_lengths = torch.minimum(input_lengths, actual_seq_len)

    # Compute CTC loss safely
    loss_raw = F.ctc_loss(
        ctc_log_probs_t,
        clean_targets,
        input_lengths,
        target_lengths,
        blank=GlossVocabulary.PAD_ID,
        reduction="none",
        zero_infinity=True,
    )

    # FastEmit Regularization: penalize non-blank tokens to encourage earlier emission (masked by valid frames)
    fastemit_lambda = 0.001
    prob_blank = torch.exp(ctc_log_probs[:, :, GlossVocabulary.PAD_ID])
    prob_non_blank = 1.0 - prob_blank
    
    frame_idx_tensor = torch.arange(ctc_log_probs.size(1), device=ctc_log_probs.device).unsqueeze(0)
    valid_frame_mask = (frame_idx_tensor < input_lengths.unsqueeze(1)).float()
    
    fastemit_penalty = fastemit_lambda * (prob_non_blank * valid_frame_mask).sum(dim=1) / valid_frame_mask.sum(dim=1).clamp_min(1.0)
    loss_raw = loss_raw + fastemit_penalty

    valid_f = valid_ctc.float()
    if sample_weights is not None:
        valid_f = valid_f * sample_weights

    loss_ctc = _distributed_normalize(
        (loss_raw * valid_f).float().sum(), valid_f.float().sum()
    )
    return (
        loss_ctc,
        has_valid_gloss.float().sum(),
        valid_f.float().sum(),
        (has_valid_gloss.float() - valid_f).clamp_min(0.0).sum(),
        input_lengths.float().mean(),
        target_lengths.float().mean(),
        min_ctc_len.float().mean(),
    )


def _compute_mlm_loss_safe(
    mlm_logits: torch.Tensor,
    mlm_labels: torch.Tensor,
    mlm_mask: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Computes Continuous Masked Landmark Reconstruction Loss with safety padding."""
    batch_size_mlm, seq_len_mlm, channels_mlm = mlm_logits.shape
    target = mlm_labels.float()
    if target.dim() == 4:
        target = target.view(batch_size_mlm, target.shape[1], -1)
    elif target.dim() == 2:
        target = target.view(batch_size_mlm, -1, channels_mlm)

    if target.shape[1] != seq_len_mlm:
        ratio = target.shape[1] // max(1, seq_len_mlm)
        if ratio >= 2 and target.shape[1] % seq_len_mlm == 0:
            # Stride-average target frames to align pooled MLM tokens with original frames
            target = target.permute(0, 2, 1)  # [B, C, T]
            target = F.avg_pool1d(target, kernel_size=ratio, stride=ratio)
            target = target.permute(0, 2, 1)  # [B, seq_len_mlm, C]
        else:
            min_len = min(int(target.shape[1]), seq_len_mlm)
            mlm_logits = mlm_logits[:, :min_len, :]
            target = target[:, :min_len, :]
            mlm_mask = mlm_mask[:, :min_len]
            seq_len_mlm = min_len

    target = target.reshape(batch_size_mlm, seq_len_mlm, channels_mlm)
    mask_flat = mlm_mask.unsqueeze(-1).float()
    loss_raw = (
        F.smooth_l1_loss(mlm_logits.float(), target, reduction="none") * mask_flat
    )

    if sample_weights is not None:
        weighted_mask = mask_flat * sample_weights.view(-1, 1, 1)
        loss_raw = loss_raw * sample_weights.view(-1, 1, 1)
    else:
        weighted_mask = mask_flat

    # Normalizer must account for the channel dimension since loss_raw sums over channels
    normalizer = weighted_mask.float().sum() * channels_mlm
    return _distributed_normalize(loss_raw.float().sum(), normalizer)


def _move_batch_to_device(batch, device, prec_dtype, args, is_train=True):
    feat_dtype = prec_dtype if prec_dtype in (torch.float16, torch.bfloat16) else None

    features = batch["feature"].to(device, dtype=feat_dtype, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    labels = batch.get(
        "label", torch.zeros(batch["feature"].size(0), dtype=torch.long, device=device)
    ).to(device, non_blocking=True)

    frame_indices = (
        batch["frame_indices"].to(device, non_blocking=True)
        if "frame_indices" in batch
        else None
    )

    sample_weight = batch.get(
        "sample_weight", torch.ones_like(labels, dtype=torch.float32, device=device)
    ).to(device, non_blocking=True)

    domain_tgts = batch.get(
        "domain_label", batch.get("source_id", torch.zeros_like(labels))
    ).to(device, non_blocking=True)

    sample_ids = None
    has_domain = batch.get(
        "has_domain_label", torch.ones_like(domain_tgts, dtype=torch.bool)
    ).to(device, non_blocking=True)

    gloss_seq = batch["gloss_seq"].to(device, non_blocking=True)
    gloss_len = batch["gloss_len"].to(device, non_blocking=True)
    has_valid_gloss = batch["has_valid_gloss"].to(device, non_blocking=True)
    mlm_mask = (
        batch["mlm_mask"].to(device, non_blocking=True)
        if "mlm_mask" in batch and batch["mlm_mask"] is not None
        else None
    )

    if getattr(args, "enable_aux_decoders", True):
        chicago_seq = batch["chicago_seq"].to(device, non_blocking=True)
        chicago_len = batch["chicago_len"].to(device, non_blocking=True)
        has_valid_chicago = batch["has_valid_chicago"].to(device, non_blocking=True)
        english_seq = batch["english_seq"].to(device, non_blocking=True)
        english_len = batch["english_len"].to(device, non_blocking=True)
        has_valid_english = batch["has_valid_english"].to(device, non_blocking=True)
    else:
        chicago_seq = chicago_len = english_seq = english_len = None
        has_valid_chicago = torch.zeros_like(gloss_len, dtype=torch.bool)
        has_valid_english = torch.zeros_like(gloss_len, dtype=torch.bool)

    is_isolated = batch.get(
        "is_isolated", torch.ones_like(labels, dtype=torch.bool)
    ).to(device, non_blocking=True)
    eng_trunc_flag = batch.get(
        "english_trunc",
        torch.zeros(has_valid_english.shape, dtype=torch.bool, device=device),
    ).to(device, non_blocking=True)

    return (
        features,
        mask,
        labels,
        frame_indices,
        sample_weight,
        domain_tgts,
        sample_ids,
        has_domain,
        gloss_seq,
        gloss_len,
        has_valid_gloss,
        mlm_mask,
        chicago_seq,
        chicago_len,
        has_valid_chicago,
        english_seq,
        english_len,
        has_valid_english,
        is_isolated,
        eng_trunc_flag,
    )


def train_epoch_tpu(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    loss_wrapper: HomoscedasticLossWrapper,
    ema: Optional[Any],
    supcon_fn: SupervisedContrastiveLoss,
    device: torch.device,
    scaler: Optional[Any] = None,
    epoch: int = 0,
    total_epochs: int = 120,
    prec_dtype: torch.dtype = torch.bfloat16,
    is_master: bool = True,
    is_xla: bool = True,
    accum_steps: int = 4,
    class_weights: Optional[torch.Tensor] = None,
    args: Optional[Any] = None,
    loss_ema: Optional[Any] = None,
) -> Dict[str, float]:
    """Executes a single training epoch across TPU cores."""

    model.train()

    # ─── Dynamic Token Merging Scaling ───
    # Disabled dynamic Token Merging to enforce static graph shapes and prevent XLA compilation thrashing

    # Removed dynamic optimizer.add_param_group to prevent TorchDynamo graph invalidation

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

        is_master = xm.is_master_ordinal() if is_xla else True

    if is_xla:
        if prec_dtype == torch.float16:
            raise ValueError(
                "TPU natively supports bfloat16 or float32 precision. Float16 is not supported on TPU."
            )
        device_type = "xla"
        use_autocast = prec_dtype == torch.bfloat16
    else:
        device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
        use_autocast = "cuda" in str(device).lower() and prec_dtype != torch.float32

    # scaler passed in

    progress = float(max(0, epoch)) / float(max(1, total_epochs - 1))
    grl_alpha = max(0.01, round(float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0), 2))
    label_smoothing = max(0.05, 0.15 - 0.10 * progress)

    try:
        total_batches = len(loader)
    except TypeError:
        total_batches = 2500
    min_batches = total_batches
    if is_xla:
        bpe = getattr(args, "batches_per_execution", 16)
        para_loader = pl.MpDeviceLoader(loader, device, batches_per_execution=bpe)
        min_batches = int(
            xm.mesh_reduce(
                "min_batches", total_batches, lambda input_x: min(input_x)
            )
        )
    else:
        para_loader = loader

    raw_model = model.module if hasattr(model, "module") else model
    if hasattr(raw_model, "update_tome_r") and args is not None:
        raw_model.update_tome_r(epoch, args.epochs)

    if is_xla:
        try:
            ord_val = xm.get_ordinal()
        except AttributeError:
            try:
                import torch_xla.runtime as xr

                ord_val = xr.global_ordinal()
            except Exception:
                ord_val = 0

    step_start_time = time.time()
    last_log_time_box = [step_start_time]

    if args is not None and getattr(args, "max_steps", 0) > 0:
        min_batches = min(min_batches, args.max_steps)

    TRAIN_METRIC_KEYS = [
        "loss",
        "seq",
        "sem",
        "corr",
        "total",
        "aux",
        "ctc_eligible",
        "ctc_used",
        "ctc_dropped",
        "sum_enc_len",
        "sum_tgt_len",
        "sum_min_ctc",
        "chi_corr",
        "chi_total",
        "eng_corr",
        "eng_total",
        "lpc_loss",
    ]
    running_metrics = torch.zeros(len(TRAIN_METRIC_KEYS), device="cpu")
    running_truncs = torch.zeros(3, device="cpu")
    optimizer.zero_grad(set_to_none=True)
    for step_idx, batch in enumerate(para_loader, start=1):
        if step_idx == 1:
            step_start_time = time.time()
        if step_idx > min_batches:
            if is_xla:
                if "para_loader" in locals():
                    del para_loader

                gc.collect()
            break

        (
            features,
            mask,
            labels,
            frame_indices,
            sample_weight,
            domain_tgts,
            sample_ids,
            has_domain,
            gloss_seq,
            gloss_len,
            has_valid_gloss,
            mlm_mask,
            chicago_seq,
            chicago_len,
            has_valid_chicago,
            english_seq,
            english_len,
            has_valid_english,
            is_isolated,
            _,
        ) = _move_batch_to_device(batch, device, prec_dtype, args, is_train=True)

        def forward_and_losses(
            features=features, mask=mask, labels=labels, frame_indices=frame_indices,
            sample_weight=sample_weight, domain_tgts=domain_tgts, sample_ids=sample_ids,
            has_domain=has_domain, gloss_seq=gloss_seq, gloss_len=gloss_len,
            has_valid_gloss=has_valid_gloss, mlm_mask=mlm_mask, chicago_seq=chicago_seq,
            chicago_len=chicago_len, has_valid_chicago=has_valid_chicago, english_seq=english_seq,
            english_len=english_len, has_valid_english=has_valid_english, is_isolated=is_isolated
        ):
            """Forward pass for this module."""
            eff_english_seq = english_seq

            out = model(
                features,
                mask=mask,
                gloss_seq=gloss_seq,
                chicago_seq=chicago_seq,
                english_seq=eff_english_seq,
                mlm_mask=mlm_mask,
                frame_indices=frame_indices,
                return_aux=True,
                grl_alpha=grl_alpha,
            )
            (
                dec_logits,
                mtp_logits,
                chicago_logits,
                english_logits,
                _,
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
                out.get("mtp_logits", None),
                out.get("chicago_logits"),
                out.get("english_logits"),
                out["dec_hidden"],
                out["ctc_log_probs"],
                out["vis_emb"],
                out["sent_emb"],
                out["proj_feats"],
                out.get("domain_logits", None),
                out["aux_logits"],
                out["enc_mask"],
                out["pred_len"],
                out.get("chicago_pred_len"),
                out.get("english_pred_len"),
            )
            h_cls = out.get("h_cls")

            gt_tokens = gloss_seq[:, 1:].contiguous()
            token_mask = (
                gt_tokens != GlossVocabulary.PAD_ID
            ) & has_valid_gloss.bool().unsqueeze(-1)
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
                    (loss_length * sample_weight * has_valid_gloss.float())
                    .float()
                    .sum(),
                    (has_valid_gloss.float() * sample_weight).float().sum(),
                )

                if dec_logits is not None:
                    loss_seq, loss_eos = compute_seq_and_eos_loss(
                        dec_logits,
                        gt_tokens,
                        valid_gloss_mask,
                        token_mask,
                        class_weights=class_weights,
                        sample_weights=sample_weight,
                        label_smoothing=label_smoothing,
                    )
                else:
                    loss_seq = torch.zeros((), device=device)
                    loss_eos = torch.zeros((), device=device)
            else:
                loss_length = torch.zeros((), device=device)
                loss_seq = torch.zeros((), device=device)
                loss_eos = torch.zeros((), device=device)

            # --- CHICAGO LOSS (Sample-wise Masking) ---
            c_valid = has_valid_chicago.float()
            if chicago_pred_len is not None and chicago_seq is not None:
                c_valid_seq_mask = (
                    (chicago_seq[:, 1:] != GlossVocabulary.PAD_ID)
                    & (chicago_seq[:, 1:] != GlossVocabulary.EOS_ID)
                    & has_valid_chicago.unsqueeze(-1)
                )
                c_target_len = (c_valid_seq_mask).sum(dim=1).float()
                loss_chicago_len = F.smooth_l1_loss(
                    chicago_pred_len, c_target_len, reduction="none"
                )
                loss_chicago_len = _distributed_normalize(
                    (loss_chicago_len * sample_weight * c_valid).float().sum(),
                    (c_valid * sample_weight).float().sum(),
                )
            else:
                loss_chicago_len = torch.zeros((), device=device)

            if chicago_logits is not None:
                loss_chicago, loss_chicago_eos = compute_seq_and_eos_loss(
                    chicago_logits,
                    chicago_seq[:, 1:],
                    (chicago_seq[:, 1:] != GlossVocabulary.PAD_ID)
                    & (chicago_seq[:, 1:] != GlossVocabulary.EOS_ID)
                    & has_valid_chicago.unsqueeze(-1),
                    (chicago_seq[:, 1:] != GlossVocabulary.PAD_ID)
                    & has_valid_chicago.unsqueeze(-1),
                    sample_weights=sample_weight,
                    label_smoothing=0.1,
                    pad_id=GlossVocabulary.PAD_ID,
                )
            else:
                loss_chicago = torch.zeros((), device=device)
                loss_chicago_eos = torch.zeros((), device=device)

            # --- ENGLISH LOSS (Sample-wise Masking) ---
            e_valid = has_valid_english.float()
            if english_pred_len is not None and english_seq is not None:
                e_valid_seq_mask = (
                    (english_seq[:, 1:] != EnglishVocabulary.PAD_ID)
                    & (english_seq[:, 1:] != EnglishVocabulary.EOS_ID)
                & (english_seq[:, 1:] != EnglishVocabulary.UNK_ID)
                    & has_valid_english.unsqueeze(-1)
                )
                e_target_len = (e_valid_seq_mask).sum(dim=1).float()
                loss_english_len = F.smooth_l1_loss(
                    english_pred_len, e_target_len, reduction="none"
                )
                loss_english_len = _distributed_normalize(
                    (loss_english_len * sample_weight * e_valid).float().sum(),
                    (e_valid * sample_weight).float().sum(),
                )
            else:
                loss_english_len = torch.zeros((), device=device)

            if english_logits is not None:
                loss_english, loss_english_eos = compute_seq_and_eos_loss(
                    english_logits,
                    english_seq[:, 1:],
                    (english_seq[:, 1:] != EnglishVocabulary.PAD_ID)
                    & (english_seq[:, 1:] != EnglishVocabulary.EOS_ID)
                & (english_seq[:, 1:] != EnglishVocabulary.UNK_ID)
                    & has_valid_english.unsqueeze(-1),
                    (english_seq[:, 1:] != EnglishVocabulary.PAD_ID)
                    & has_valid_english.unsqueeze(-1),
                    sample_weights=sample_weight,
                    label_smoothing=0.1,
                    pad_id=EnglishVocabulary.PAD_ID,
                    eos_id=EnglishVocabulary.EOS_ID,
                )
            else:
                loss_english = torch.zeros((), device=device)
                loss_english_eos = torch.zeros((), device=device)

            # --- AUXILIARY GROUNDING & GLOSS AUX LOSSES ---
            raw_model = model.module if hasattr(model, "module") else model
            isolated_f = is_isolated.float()
            if aux_logits is not None:
                raw_target = labels + GlossVocabulary.OFFSET
                mask_valid = (labels != -1) & (raw_target >= 0) & (raw_target < raw_model.vocab_size)
                aux_target = torch.where(mask_valid, raw_target, torch.zeros_like(raw_target))
                loss_aux = F.cross_entropy(
                    aux_logits.float(),
                    aux_target.long(),
                    reduction="none",
                    label_smoothing=0.1,
                )
                loss_aux = loss_aux * mask_valid.float()
            else:
                loss_aux = torch.zeros((), device=device)
            valid_isolated = isolated_f * (labels != -1).float()
            loss_aux = _distributed_normalize(
                (loss_aux * sample_weight * valid_isolated).float().sum(),
                (sample_weight * valid_isolated).float().sum(),
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
                inter_ctc_logits = out.get("inter_ctc_log_probs", None)
                if inter_ctc_logits is not None:
                    loss_inter_ctc, _, _, _, _, _, _ = _compute_ctc_loss_safe(
                        inter_ctc_logits,
                        gloss_seq,
                        gloss_len,
                        out.get("orig_enc_mask", enc_mask),
                        has_valid_gloss,
                        sample_weights=sample_weight,
                    )
                else:
                    loss_inter_ctc = torch.tensor(0.0, device=device)

                loss_lpc = out.get("loss_lpc", torch.tensor(0.0, device=device))

                h_cls = out.get("h_cls")
                gt_tokens = (
                    gloss_seq[:, 1:].contiguous() if gloss_seq is not None else None
                )
                valid_gloss_mask = (
                    (gt_tokens != GlossVocabulary.PAD_ID)
                    & has_valid_gloss.unsqueeze(-1)
                    if gt_tokens is not None
                    else None
                )

                loss_dense_sem = raw_model.dense_sem_loss(
                    h_cls,
                    raw_model.decoder.asl_lex_emb(gt_tokens),
                    valid_gloss_mask & (gt_tokens != GlossVocabulary.EOS_ID),
                    sample_weights=sample_weight,
                )

                vis_emb = out.get("vis_emb", None)
                sent_emb = out.get("sent_emb", None)
                if sent_emb is not None and vis_emb is not None:
                    loss_xmodal = raw_model.xmodal_loss_fn(
                        vis_emb,
                        sent_emb,
                        has_valid_gloss,
                        sample_weights=sample_weight,
                        gt_tokens=gt_tokens,
                    )
                else:
                    loss_xmodal = torch.zeros((), device=device)
            else:
                loss_ctc = torch.zeros((), device=device)
                c_elig = torch.zeros((), device=device)
                c_used = torch.zeros((), device=device)
                c_drop = torch.zeros((), device=device)
                m_enc = torch.zeros((), device=device)
                m_tgt = torch.zeros((), device=device)
                m_min = torch.zeros((), device=device)
                loss_dense_sem = torch.zeros((), device=device)
                loss_xmodal = torch.zeros((), device=device)
                loss_inter_ctc = torch.zeros((), device=device)
                loss_lpc = torch.zeros((), device=device)

            if True:
                isolated_labels = torch.where(is_isolated, labels, -1)
                loss_supcon = supcon_fn(
                    proj_feats.float(),
                    isolated_labels,
                    sample_weight,
                    sample_ids=sample_ids,
                )
            else:
                loss_supcon = torch.zeros((), device=device)
            # Mask out domain loss using the explicit has_domain flag, which is False for unknown sources.
            loss_domain = torch.zeros((), device=device)
            if domain_logits is not None:
                domain_loss = F.cross_entropy(
                    domain_logits, domain_tgts.long(), reduction="none"
                )
                valid_domain_f = has_domain.float()
                loss_domain = _distributed_normalize(
                    (domain_loss * valid_domain_f).float().sum(),
                    valid_domain_f.float().sum(),
                )

            loss_clr = (
                _compute_mlm_loss_safe(
                    out["mlm_logits"],
                    out["orig_x"],
                    out["mlm_mask"],
                    sample_weights=sample_weight,
                )
                if out.get("mlm_logits") is not None
                and out.get("orig_x") is not None
                and out.get("mlm_mask") is not None
                else torch.zeros((), device=device)
            )

            mtp_logits = out.get("extra_logits")
            loss_mtp2 = torch.zeros((), device=device)
            loss_mtp3 = torch.zeros((), device=device)

            if mtp_logits is not None and out.get("dec_logits") is not None:
                target_mtp2 = gloss_seq[:, 2:]
                target_mtp2 = F.pad(target_mtp2, (0, 1), value=GlossVocabulary.PAD_ID)
                v_g_mask_shifted = valid_gloss_mask[:, 2:] if valid_gloss_mask.shape[1] > 2 else valid_gloss_mask # Claim 90 fix
                if v_g_mask_shifted.shape[1] < target_mtp2.shape[1]:
                    v_g_mask_shifted = F.pad(v_g_mask_shifted, (0, target_mtp2.shape[1] - v_g_mask_shifted.shape[1]), value=False)
                valid_mtp2_mask = (
                    v_g_mask_shifted
                    & (target_mtp2 != GlossVocabulary.PAD_ID)
                    & (target_mtp2 != GlossVocabulary.EOS_ID)
                & (target_mtp2 != GlossVocabulary.UNK_ID)
                )
                loss_mtp2, _ = compute_seq_and_eos_loss(
                    mtp_logits["logits_2"],
                    target_mtp2,
                    valid_mtp2_mask,
                    torch.zeros_like(valid_mtp2_mask),
                )

                target_mtp3 = gloss_seq[:, 3:]
                target_mtp3 = F.pad(target_mtp3, (0, 2), value=GlossVocabulary.PAD_ID)
                v_g_mask_shifted_3 = valid_gloss_mask[:, 3:] if valid_gloss_mask.shape[1] > 3 else valid_gloss_mask # Claim 90 fix
                if v_g_mask_shifted_3.shape[1] < target_mtp3.shape[1]:
                    v_g_mask_shifted_3 = F.pad(v_g_mask_shifted_3, (0, target_mtp3.shape[1] - v_g_mask_shifted_3.shape[1]), value=False)
                valid_mtp3_mask = (
                    v_g_mask_shifted_3
                    & (target_mtp3 != GlossVocabulary.PAD_ID)
                    & (target_mtp3 != GlossVocabulary.EOS_ID)
                & (target_mtp3 != GlossVocabulary.UNK_ID)
                )
                loss_mtp3, _ = compute_seq_and_eos_loss(
                    mtp_logits["logits_3"],
                    target_mtp3,
                    valid_mtp3_mask,
                    torch.zeros_like(valid_mtp3_mask),
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
                "mtp2": loss_mtp2,
                "mtp3": loss_mtp3,
                "inter_ctc": loss_inter_ctc,
                "lpc": loss_lpc,
            }
            raw_loss = loss_wrapper(loss_terms)

            with torch.no_grad():
                # Gloss Metrics
                preds = (
                    dec_logits.argmax(dim=-1) if dec_logits is not None else gt_tokens
                )
                vg_eval = valid_gloss_mask & (gt_tokens != GlossVocabulary.EOS_ID)
                vg_f = vg_eval.float()
                nc_t = ((preds == gt_tokens).float() * vg_f).float().sum()
                nt_t = vg_f.float().sum()

                # Chicago Metrics
                chicago_nc_t = torch.zeros((), device=device)
                chicago_nt_t = torch.zeros((), device=device)
                if chicago_logits is not None:
                    c_preds = chicago_logits.argmax(dim=-1)
                    vc_eval = (
                        (chicago_seq[:, 1:] != GlossVocabulary.PAD_ID)
                        & (chicago_seq[:, 1:] != GlossVocabulary.EOS_ID)
                        & has_valid_chicago.unsqueeze(-1)
                    )
                    vc_f = vc_eval.float()
                    chicago_nc_t = (
                        (c_preds == chicago_seq[:, 1:]).float() * vc_f
                    ).sum()
                    chicago_nt_t = vc_f.float().sum()

                # English Metrics
                english_nc_t = torch.zeros((), device=device)
                english_nt_t = torch.zeros((), device=device)
                if english_logits is not None:
                    e_preds = english_logits.argmax(dim=-1)
                    ve_eval = (
                        (english_seq[:, 1:] != EnglishVocabulary.PAD_ID)
                        & (english_seq[:, 1:] != EnglishVocabulary.EOS_ID)
                & (english_seq[:, 1:] != EnglishVocabulary.UNK_ID)
                        & has_valid_english.unsqueeze(-1)
                    )
                    ve_f = ve_eval.float()
                    english_nc_t = (
                        (e_preds == english_seq[:, 1:]).float() * ve_f
                    ).sum()
                    english_nt_t = ve_f.float().sum()

            return (
                raw_loss,
                dec_logits,
                nc_t,
                nt_t,
                chicago_nc_t,
                chicago_nt_t,
                english_nc_t,
                english_nt_t,
                (
                    loss_seq.detach()
                    if loss_seq is not None
                    else torch.tensor(0.0, device=device)
                ),
                loss_aux.detach(),
                loss_ctc.detach(),
                loss_dense_sem.detach(),
                loss_chicago.detach(),
                loss_english.detach(),
                # calling .detach() on them throws AttributeError.
                # Furthermore, torch.stack requires Tensors, not native floats. Cast them here.
                c_elig,
                c_used,
                c_drop,
                m_enc,
                m_tgt,
                m_min,
                loss_lpc.detach(),
            )

        if use_autocast and not is_xla:
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
                    loss_lpc,
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
                loss_lpc,
            ) = forward_and_losses()

        loss = raw_loss

        effective_accum = min(
            args.accum_steps,
            min_batches
            - ((step_idx - 1) // max(1, args.accum_steps)) * max(1, args.accum_steps),
        )

        if scaler is not None:
            scaler.scale((loss * args.bwd_weight) / effective_accum).backward()
        else:
            ((loss * args.bwd_weight) / effective_accum).backward()

        if is_xla:
            import torch_xla.core.xla_model as xm

            do_update = (step_idx % max(1, args.accum_steps) == 0) or (
                step_idx == min_batches
            )
            if do_update:
                # Upcast gradients to float32 before clipping and optimizer step (Claims 72-73)
                for p in list(model.parameters()) + list(loss_wrapper.parameters()):
                    if p.grad is not None and p.grad.dtype != torch.float32:
                        p.grad.data = p.grad.data.to(torch.float32)
                xla_clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0,
                )
                xm.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
        else:
            do_update = (step_idx % max(1, args.accum_steps) == 0) or (
                step_idx == min_batches
            )
            if do_update:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    # Upcast gradients to float32 before clipping and optimizer step (Claims 72-73)
                    for p in list(model.parameters()) + list(loss_wrapper.parameters()):
                        if p.grad is not None and p.grad.dtype != torch.float32:
                            p.grad.data = p.grad.data.to(torch.float32)
                    xla_clip_grad_norm_(
                        list(model.parameters()) + list(loss_wrapper.parameters()),
                        max_norm=1.0,
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Upcast gradients to float32 before clipping and optimizer step (Claims 72-73)
                    for p in list(model.parameters()) + list(loss_wrapper.parameters()):
                        if p.grad is not None and p.grad.dtype != torch.float32:
                            p.grad.data = p.grad.data.to(torch.float32)
                    xla_clip_grad_norm_(
                        list(model.parameters()) + list(loss_wrapper.parameters()),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        if scheduler is not None and do_update:
            if (
                not hasattr(scheduler, "total_steps")
                or scheduler.last_epoch < scheduler.total_steps
            ):
                scheduler.step()
        if ema is not None and do_update:
            raw_m = model.module if hasattr(model, 'module') else model
            ema.update(raw_m, float(epoch) / float(total_epochs))
            if loss_ema is not None:
                loss_ema.update(loss_wrapper, float(epoch) / float(total_epochs))

        raw_m = model.module if hasattr(model, "module") else model
        if hasattr(raw_m, "dense_sem_loss") and (
            step_idx % 4 == 0 or step_idx == min_batches
        ):
            raw_m.dense_sem_loss.update_momentum()

        log_freq = min(25, max(1, min_batches // 10))
        if is_master and (
            (step_idx <= 10) or (step_idx % log_freq == 0) or (step_idx == min_batches)
        ):
            current_log_time = time.time()

            # Pack 13 scalar metrics into a single 1D tensor for 1-pass batched DMA host transfer
            log_vec = torch.stack(
                [
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
                ]
            )

            def _async_step_print(
                log_vec,
                st_idx,
                m_batches,
                ep,
                tot_ep,
                lr_val,
                t_start,
                t_prev_box,
                b_sz,
                l_freq,
            ):
                """Internal helper method _async_step_print."""
                vals = log_vec.cpu().tolist()
                (
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
                ) = vals
                g_acc = (float(nc_val) / max(1.0, float(nt_val))) * 100.0
                c_acc = (float(cnc_val) / max(1.0, float(cnt_val))) * 100.0
                e_acc = (float(enc_val) / max(1.0, float(ent_val))) * 100.0
                t_now = time.time()
                elapsed_win = max(0.001, t_now - t_prev_box[0])
                t_prev_box[0] = t_now
                win_steps = l_freq if st_idx >= l_freq else st_idx
                instant_it_s = float(win_steps) / elapsed_win
                samples_per_s = instant_it_s * b_sz
                import math

                eng_ppl_val = (
                    math.exp(min(float(eng_val), 20.0)) if float(eng_val) > 0 else 0.0
                )
                pct = (float(st_idx) / max(1.0, float(m_batches))) * 100.0
                e_ppl_str = (
                    f" PPL(Eng:{eng_ppl_val:.1f}) |" if float(eng_val) > 0 else ""
                )
                msg = (
                    f"  [Epoch {ep:03d}/{tot_ep:03d} | Step {st_idx:04d}/{m_batches:04d} ({pct:5.1f}%)] "
                    f"Loss: {float(l_val):.4f} [Seq:{float(s_val):.4f} CTC:{float(c_val):.4f} Sem:{float(sm_val):.4f} Chi:{float(chi_val):.4f} Eng:{float(eng_val):.4f}] | "
                    f"Acc (Gloss:{g_acc:.1f}% Chi:{c_acc:.1f}% Eng:{e_acc:.1f}%){e_ppl_str} "
                    f"LR: {lr_val:.2e} | Speed: {samples_per_s:.1f} seq/s ({instant_it_s:.2f} it/s)"
                )
                print(msg, flush=True)

            batch_sz_val = getattr(loader, "batch_size", 64)
            if not isinstance(batch_sz_val, int):
                batch_sz_val = 64

            args_tuple = (
                log_vec,
                step_idx,
                min_batches,
                epoch,
                total_epochs,
                optimizer.param_groups[0]["lr"],
                step_start_time,
                last_log_time_box,
                batch_sz_val,
                log_freq,
            )

            if is_xla:
                import torch_xla.core.xla_model as xm

                xm.add_step_closure(_async_step_print, args=args_tuple)
            else:
                _async_step_print(*args_tuple)
        with torch.no_grad():
            metrics_vec = torch.stack(
                [
                    raw_loss.detach(),
                    l_seq.detach(),
                    l_sem.detach(),
                    nc_t.detach(),
                    nt_t.detach(),
                    l_aux.detach(),
                    c_elig,
                    c_used,
                    c_drop,
                    m_enc,
                    m_tgt,
                    m_min,
                    c_nc_t.detach(),
                    c_nt_t.detach(),
                    e_nc_t.detach(),
                    e_nt_t.detach(),
                    (
                        loss_lpc.detach()
                        if "loss_lpc" in locals()
                        else torch.tensor(0.0, device=device)
                    ),
                ]
            )

            gt_flag = batch.get(
                "gloss_trunc", torch.zeros((1,), dtype=torch.bool, device=device)
            )
            ct_flag = batch.get(
                "chicago_trunc", torch.zeros((1,), dtype=torch.bool, device=device)
            )
            et_flag = batch.get(
                "english_trunc", torch.zeros((1,), dtype=torch.bool, device=device)
            )

            truncs_vec = torch.stack(
                [
                    gt_flag.float().sum().detach(),
                    ct_flag.float().sum().detach(),
                    et_flag.float().sum().detach(),
                ]
            )

            def _update_metrics(m_vec, t_vec):
                """Internal helper method _update_metrics."""
                running_metrics.add_(m_vec.cpu())
                running_truncs.add_(t_vec.cpu())

            if is_xla:
                import torch_xla.core.xla_model as xm
                xm.add_step_closure(_update_metrics, args=(metrics_vec, truncs_vec))
            else:
                _update_metrics(metrics_vec, truncs_vec)
            
        del batch
        del l_seq, l_aux, l_ctc, l_sem, l_chi, l_eng
        del features, mask, labels, gloss_seq, chicago_seq, english_seq
        del frame_indices, domain_tgts, has_domain, mlm_mask, loss


        # Removed gc.collect() to prevent massive CPU stalling at high iteration speeds
        if is_xla and step_idx >= min_batches:
            if "para_loader" in locals():
                del para_loader

            gc.collect()
            break

    if is_xla:
        xm.mark_step()
        xm.rendezvous("end_of_epoch")

        # Combine the running metrics and the truncation flags into one tensor
        if "running_metrics" not in locals():
            running_metrics = torch.zeros(
                len(TRAIN_METRIC_KEYS), dtype=torch.float32, device=device
            )
        if "running_truncs" not in locals():
            running_truncs = torch.zeros(3, dtype=torch.float32, device=device)

        final_vec = torch.cat(
            [
                running_metrics.to(device),
                torch.full(
                    (1,), float(min_batches), dtype=torch.float32, device=device
                ),
                running_truncs.to(device),
            ]
        )

        final_vec = xm.all_reduce(xm.REDUCE_SUM, final_vec)
        m_np = final_vec.detach().cpu().numpy()

        for idx, key in enumerate(TRAIN_METRIC_KEYS):
            tracker[key] = float(m_np[idx])

        global_batches = float(m_np[17])
        g_tr = float(m_np[18])
        c_tr = float(m_np[19])
        e_tr = float(m_np[20])
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
        if drop_rate > 20.0:
            print(
                f"WARNING: High CTC drop rate ({drop_rate:.2f}%) due to unalignable lengths. Training objective may be compromised!"
            )
        print(
            f"[Epoch {epoch} CTC Lengths] Mean Enc: {tracker['sum_enc_len']/max(1, global_batches):.1f} | Mean Tgt: {tracker['sum_tgt_len']/max(1, global_batches):.1f} | Min CTC: {tracker['sum_min_ctc']/max(1, global_batches):.1f}"
        )

    avg_loss = tracker["loss"] / float(max(1, global_batches))
    token_acc = (tracker["corr"] / max(1.0, tracker["total"])) * 100.0
    if "para_loader" in locals():
        del para_loader

    gc.collect()

    if is_xla:
        import torch_xla.core.xla_model as xm

        xm.wait_device_ops()

    if is_master and is_xla:
        try:
            mem_info = xm.get_memory_info(device)
            if mem_info:
                free_mb = mem_info.get("kb_free", 0) / 1024
                total_mb = mem_info.get("kb_total", 0) / 1024
                used_mb = total_mb - free_mb
                print(
                    f"[TPU Memory] {used_mb:.1f} MB used / {total_mb:.1f} MB total",
                    flush=True,
                )
            print("\n" + "=" * 80)
            print(f"🚀 XLA METRICS REPORT (END OF EPOCH {epoch}) 🚀")
            print("=" * 80)
            import torch_xla.debug.metrics as met

            print(met.metrics_report())
            print("=" * 80 + "\n", flush=True)
        except Exception:
            pass

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
    args: Optional[Any] = None,
) -> Tuple[float, float]:
    """Evaluates the model on the validation dataset."""

    model.eval()

    # [TPU XLA HOTFIX] Disable ToMe during validation to prevent dynamic shape graph recompilations
    # Removed validation temporal resolution discrepancy (Claims 51, 77)
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
        "ar_corr": 0.0,
        "ar_exact": 0.0,
        "ar_total": 0.0,
        "ar_seq_total": 0.0,
    }

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    if is_xla:
        import torch_xla.core.xla_model as xm

        is_master = xm.is_master_ordinal() if is_xla else True

    if is_xla:
        if prec_dtype == torch.float16:
            raise ValueError(
                "TPU natively supports bfloat16 or float32 precision. Float16 is not supported on TPU."
            )
        device_type = "xla"
        use_autocast = prec_dtype == torch.bfloat16
    else:
        device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
        use_autocast = "cuda" in str(device).lower() and prec_dtype != torch.float32

    try:
        total_val_batches = len(loader)
    except TypeError:
        total_val_batches = 500
    min_val_batches = total_val_batches
    if is_xla:
        min_val_batches = int(
            xm.mesh_reduce(
                "min_val_batches", total_val_batches, lambda input_x: min(input_x)
            )
        )

        if args is not None and getattr(args, "val_check_steps", 0) > 0:
            min_val_batches = min(min_val_batches, args.val_check_steps)

        sliced_loader = itertools.islice(loader, min_val_batches)
        bpe = getattr(args, "batches_per_execution", 16)
        para_loader = pl.MpDeviceLoader(sliced_loader, device, batches_per_execution=bpe)
    else:
        para_loader = loader
        min_val_batches = total_val_batches
        if args is not None and getattr(args, "val_check_steps", 0) > 0:
            min_val_batches = min(min_val_batches, args.val_check_steps)

    with torch.no_grad():

        for step_idx, batch in enumerate(para_loader, 1):
            if step_idx > min_val_batches:
                if is_xla:
                    if "para_loader" in locals():
                        del para_loader

                    gc.collect()
                break
            (
                features,
                mask,
                labels,
                frame_indices,
                sample_weight,
                domain_tgts,
                sample_ids,
                has_domain,
                gloss_seq,
                gloss_len,
                has_valid_gloss,
                mlm_mask,
                chicago_seq,
                chicago_len,
                has_valid_chicago,
                english_seq,
                english_len,
                has_valid_english,
                is_isolated,
                eng_trunc_flag,
            ) = _move_batch_to_device(batch, device, prec_dtype, args, is_train=False)

            def forward_and_losses(
                features=features, mask=mask, labels=labels, frame_indices=frame_indices,
                sample_weight=sample_weight, domain_tgts=domain_tgts, sample_ids=sample_ids,
                has_domain=has_domain, gloss_seq=gloss_seq, gloss_len=gloss_len,
                has_valid_gloss=has_valid_gloss, mlm_mask=mlm_mask, chicago_seq=chicago_seq,
                chicago_len=chicago_len, has_valid_chicago=has_valid_chicago, english_seq=english_seq,
                english_len=english_len, has_valid_english=has_valid_english, is_isolated=is_isolated
            ):
                """Forward pass for this module."""

                eff_english_seq = english_seq

                out = model(
                    features,
                    mask=mask,
                    gloss_seq=gloss_seq,
                    chicago_seq=chicago_seq,
                    english_seq=eff_english_seq,
                    mlm_mask=None,
                    frame_indices=frame_indices,
                    return_aux=True,
                    grl_alpha=0.0,
                    compute_mlm=False,
                    compute_lpc=False,
                )
                dec_logits = out.get("dec_logits") if isinstance(out, dict) else out
                chicago_logits = (
                    out.get("chicago_logits") if isinstance(out, dict) else None
                )
                english_logits = (
                    out.get("english_logits") if isinstance(out, dict) else None
                )
                gt_tokens = gloss_seq[:, 1:].contiguous()
                token_mask = (
                    gt_tokens != GlossVocabulary.PAD_ID
                ) & has_valid_gloss.unsqueeze(-1)
                valid_mask = token_mask & (gt_tokens != GlossVocabulary.EOS_ID)

                if dec_logits is not None:
                    loss_seq, loss_eos = compute_seq_and_eos_loss(
                        dec_logits,
                        gt_tokens,
                        valid_mask,
                        token_mask,
                        class_weights=class_weights,
                        sample_weights=sample_weight,
                    )
                else:
                    loss_seq = torch.zeros((), device=device)
                    loss_eos = torch.zeros((), device=device)

                nc_t, nt_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                if dec_logits is not None:
                    preds = dec_logits.argmax(dim=-1)
                    valid_f = valid_mask.float()
                    nc_t = ((preds == gt_tokens).float() * valid_f).float().sum()
                    nt_t = valid_f.float().sum()

                c_nc_t, c_nt_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                loss_chi = torch.zeros((), device=device)
                loss_chi_eos = torch.zeros((), device=device)
                if chicago_logits is not None:
                    loss_chi, loss_chi_eos = compute_seq_and_eos_loss(
                        chicago_logits,
                        chicago_seq[:, 1:],
                        (chicago_seq[:, 1:] != GlossVocabulary.PAD_ID)
                        & (chicago_seq[:, 1:] != GlossVocabulary.EOS_ID)
                        & has_valid_chicago.unsqueeze(-1),
                        (chicago_seq[:, 1:] != GlossVocabulary.PAD_ID)
                        & has_valid_chicago.unsqueeze(-1),
                        sample_weights=sample_weight,
                        label_smoothing=0.1,
                        pad_id=GlossVocabulary.PAD_ID,
                    )
                    c_valid = (
                        (chicago_seq[:, 1:] != GlossVocabulary.PAD_ID)
                        & (chicago_seq[:, 1:] != GlossVocabulary.EOS_ID)
                        & has_valid_chicago.unsqueeze(-1)
                    ).float()
                    c_nc_t = (
                        (chicago_logits.argmax(dim=-1) == chicago_seq[:, 1:]).float()
                        * c_valid
                    ).sum()
                    c_nt_t = c_valid.float().sum()

                e_nc_t, e_nt_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                loss_eng = torch.zeros((), device=device)
                loss_eng_eos = torch.zeros((), device=device)
                if english_logits is not None:
                    loss_eng, loss_eng_eos = compute_seq_and_eos_loss(
                        english_logits,
                        english_seq[:, 1:],
                        (english_seq[:, 1:] != EnglishVocabulary.PAD_ID)
                        & (english_seq[:, 1:] != EnglishVocabulary.EOS_ID)
                & (english_seq[:, 1:] != EnglishVocabulary.UNK_ID)
                        & has_valid_english.unsqueeze(-1),
                        (english_seq[:, 1:] != EnglishVocabulary.PAD_ID)
                        & has_valid_english.unsqueeze(-1),
                        sample_weights=sample_weight,
                        label_smoothing=0.1,
                        pad_id=EnglishVocabulary.PAD_ID,
                        eos_id=EnglishVocabulary.EOS_ID,
                    )
                    e_valid = (
                        (english_seq[:, 1:] != EnglishVocabulary.PAD_ID)
                        & (english_seq[:, 1:] != EnglishVocabulary.EOS_ID)
                & (english_seq[:, 1:] != EnglishVocabulary.UNK_ID)
                        & has_valid_english.unsqueeze(-1)
                    ).float()
                    e_nc_t = (
                        (english_logits.argmax(dim=-1) == english_seq[:, 1:]).float()
                        * e_valid
                    ).sum()
                    e_nt_t = e_valid.float().sum()

                _ = _ = _ = _ = _ = _ = _ = _ = torch.zeros((), device=device)

                e_trunc_c, e_trunc_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                if True:
                    e_trunc_c = eng_trunc_flag.float().sum()
                    e_trunc_t = (has_valid_english | eng_trunc_flag).float().sum()

                loss_terms = {
                    "seq": loss_seq,
                    "eos": loss_eos,
                    "chicago": loss_chi,
                    "chicago_eos": loss_chi_eos,
                    "english": loss_eng,
                    "english_eos": loss_eng_eos,
                    "ctc": torch.zeros((), device=device),
                    "dense_sem": torch.zeros((), device=device),
                    "xmodal": torch.zeros((), device=device),
                    "supcon": torch.zeros((), device=device),
                    "aux": torch.zeros((), device=device),
                    "chicago_len": torch.zeros((), device=device),
                    "english_len": torch.zeros((), device=device),
                    "length": torch.zeros((), device=device),
                    "mtp2": torch.zeros((), device=device),
                    "mtp3": torch.zeros((), device=device),
                    "inter_ctc": torch.zeros((), device=device),
                    "lpc": torch.zeros((), device=device),
                    "domain": torch.zeros((), device=device),
                    "clr": torch.zeros((), device=device),
                }
                raw_loss = loss_wrapper(loss_terms)
                # --- Autoregressive Generation ---
                ar_nc_t = torch.zeros((), device=device)
                ar_exact = torch.zeros((), device=device)
                ar_total = torch.zeros((), device=device)
                ar_seq_total = torch.zeros((), device=device)

                skip_gen = args is not None and getattr(
                    args, "skip_val_generation", False
                )
                if not skip_gen and isinstance(out, dict) and "h_seq" in out:
                    # Free heavy intermediate outputs and local references before 64-step AR loop to conserve HBM
                    for key_k_lower in [
                        "english_logits",
                        "chicago_logits",
                        "ctc_log_probs",
                        "dec_logits",
                        "dec_hidden",
                    ]:
                        if key_k_lower in out:
                            del out[key_k_lower]
                    dec_logits = None
                    chicago_logits = None
                    english_logits = None

                    gen_ids = model.generate(
                        features,
                        mask=mask,
                        max_new_tokens=64,
                        task="gloss",
                        frame_indices=frame_indices,
                        h_seq=out.get("h_seq"),
                        enc_mask=out.get("enc_mask"),
                    )
                    dec_preds = gen_ids[:, 1:]
                    gt_len = gt_tokens.size(1)
                    if dec_preds.size(1) < gt_len:
                        dec_preds = F.pad(
                            dec_preds,
                            (0, gt_len - dec_preds.size(1)),
                            value=GlossVocabulary.PAD_ID,
                        )
                    else:
                        dec_preds = dec_preds[:, :gt_len]

                    valid_f = valid_mask.float()
                    ar_nc_t = ((dec_preds == gt_tokens).float() * valid_f).float().sum()
                    ar_total = valid_f.float().sum()

                    eval_mask = valid_mask | (gt_tokens == GlossVocabulary.EOS_ID)
                    match_mask = (dec_preds == gt_tokens) | (~eval_mask)
                    ar_exact = (
                        match_mask.all(dim=1).float() * has_valid_gloss.float()
                    ).sum()
                    ar_seq_total = has_valid_gloss.float().sum()

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
                    ar_nc_t,
                    ar_exact,
                    ar_total,
                    ar_seq_total,
                    loss_seq,
                )

            if use_autocast and not is_xla:
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
                        ar_nc_t,
                        ar_exact,
                        ar_total,
                        ar_seq_total,
                        l_seq,
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
                    ar_nc_t,
                    ar_exact,
                    ar_total,
                    ar_seq_total,
                    l_seq,
                ) = forward_and_losses()

            with torch.no_grad():
                metrics_vec = torch.stack(
                    [
                        raw_loss.detach(),
                        l_chi.detach(),
                        l_eng.detach(),
                        nc_t.detach(),
                        nt_t.detach(),
                        c_nc_t.detach(),
                        c_nt_t.detach(),
                        e_nc_t.detach(),
                        e_nt_t.detach(),
                        e_trunc_c.detach(),
                        e_trunc_t.detach(),
                        ar_nc_t.detach(),
                        ar_exact.detach(),
                        ar_total.detach(),
                        ar_seq_total.detach(),
                        # l_seq: sequence (translation) loss for tracking
                        l_seq.detach() if l_seq is not None else raw_loss.detach(),
                    ]
                )

                def _val_async_step_print(r_loss, ep, st_idx, m_batches, mdl_dir):
                    if step_idx % 50 == 0 or step_idx == min_val_batches:
                        print(
                            f"  [Val Step {st_idx:04d}/{m_batches:04d}] Loss: {float(r_loss.cpu()):.4f}",
                            flush=True,
                        )
                    metrics_csv_path = os.path.join(
                        mdl_dir if mdl_dir else ".",
                        "training_metrics.csv",
                    )
                    with open(metrics_csv_path, "a", newline="") as f:
                        import csv
                        csv.writer(f).writerow(
                            [ep + 1, st_idx, "val_intra", float(r_loss.cpu())]
                            + [0.0] * 12
                        )

                if is_master:
                    mdl_dir = args.model_dir if args and hasattr(args, "model_dir") else "."
                    if is_xla:
                        import torch_xla.core.xla_model as xm
                        xm.add_step_closure(_val_async_step_print, args=(raw_loss, epoch, step_idx, min_val_batches, mdl_dir))
                    else:
                        _val_async_step_print(raw_loss, epoch, step_idx, min_val_batches, mdl_dir)

                if "running_val_metrics" not in locals():
                    running_val_metrics = torch.zeros_like(metrics_vec)
                running_val_metrics.add_(metrics_vec)

            if is_xla:
                import torch_xla.core.xla_model as xm
                # MpDeviceLoader manages batches natively
                pass

            del batch

            # Free validation tensors
            if "forward_and_losses" in locals():
                del forward_and_losses
            if "raw_loss" in locals():
                del raw_loss
            if "l_chi" in locals():
                del l_chi
            if "l_eng" in locals():
                del l_eng
            if "features" in locals():
                del features
            if "mask" in locals():
                del mask
            if "frame_indices" in locals():
                del frame_indices
            if "gloss_seq" in locals():
                del gloss_seq
            if "chicago_seq" in locals():
                del chicago_seq
            if "english_seq" in locals():
                del english_seq
            if "has_valid_gloss" in locals():
                del has_valid_gloss
            if "has_valid_chicago" in locals():
                del has_valid_chicago
            if "has_valid_english" in locals():
                del has_valid_english
            if "dec_preds" in locals():
                del dec_preds

    if is_xla:
        xm.rendezvous("validate_metrics")

        if "running_val_metrics" not in locals():
            running_val_metrics = torch.zeros(16, dtype=torch.float32, device=device)

        val_vec = torch.cat(
            [
                running_val_metrics,
                torch.tensor(
                    [float(min_val_batches)], dtype=torch.float32, device=device
                ),
            ]
        )

        val_vec = xm.all_reduce(xm.REDUCE_SUM, val_vec)
        v_np = val_vec.detach().cpu().numpy()

        tracker["loss"] = float(v_np[0])
        tracker["chi_loss"] = float(v_np[1])
        tracker["eng_loss"] = float(v_np[2])
        tracker["corr"] = float(v_np[3])
        tracker["total"] = float(v_np[4])
        tracker["chi_corr"] = float(v_np[5])
        tracker["chi_total"] = float(v_np[6])
        tracker["eng_corr"] = float(v_np[7])
        tracker["eng_total"] = float(v_np[8])
        tracker["eng_trunc_count"] = float(v_np[9])
        tracker["eng_trunc_total"] = float(v_np[10])
        tracker["ar_corr"] = float(v_np[11])
        tracker["ar_exact"] = float(v_np[12])
        tracker["ar_total"] = float(v_np[13])
        tracker["ar_seq_total"] = float(v_np[14])
        tracker["l_seq"] = float(v_np[15])
        step_idx = float(v_np[-1])
    elif torch.distributed.is_initialized():
        import torch.distributed as dist

        if "running_val_metrics" not in locals():
            running_val_metrics = torch.zeros(16, dtype=torch.float32, device=device)

        val_vec = torch.cat(
            [
                running_val_metrics,
                torch.tensor(
                    [float(min_val_batches)], dtype=torch.float32, device=device
                ),
            ]
        )
        dist.all_reduce(val_vec, op=dist.ReduceOp.SUM)
        v_np = val_vec.detach().cpu().numpy()

        tracker["loss"] = float(v_np[0])
        tracker["chi_loss"] = float(v_np[1])
        tracker["eng_loss"] = float(v_np[2])
        tracker["corr"] = float(v_np[3])
        tracker["total"] = float(v_np[4])
        tracker["chi_corr"] = float(v_np[5])
        tracker["chi_total"] = float(v_np[6])
        tracker["eng_corr"] = float(v_np[7])
        tracker["eng_total"] = float(v_np[8])
        tracker["eng_trunc_count"] = float(v_np[9])
        tracker["eng_trunc_total"] = float(v_np[10])
        tracker["ar_corr"] = float(v_np[11])
        tracker["ar_exact"] = float(v_np[12])
        tracker["ar_total"] = float(v_np[13])
        tracker["ar_seq_total"] = float(v_np[14])
        tracker["l_seq"] = float(v_np[15])
        step_idx = float(v_np[-1])
    else:
        step_idx = float(min_val_batches)
        if "running_val_metrics" in locals():
            v_np = running_val_metrics.detach().cpu().numpy()
            tracker["loss"] = float(v_np[0])
            tracker["chi_loss"] = float(v_np[1])
            tracker["eng_loss"] = float(v_np[2])
            tracker["corr"] = float(v_np[3])
            tracker["total"] = float(v_np[4])
            tracker["chi_corr"] = float(v_np[5])
            tracker["chi_total"] = float(v_np[6])
            tracker["eng_corr"] = float(v_np[7])
            tracker["eng_total"] = float(v_np[8])
            tracker["eng_trunc_count"] = float(v_np[9])
            tracker["eng_trunc_total"] = float(v_np[10])
            tracker["ar_corr"] = float(v_np[11])
            tracker["ar_exact"] = float(v_np[12])
            tracker["ar_total"] = float(v_np[13])
            tracker["ar_seq_total"] = float(v_np[14])
            tracker["l_seq"] = float(v_np[15])

    val_loss = tracker["loss"] / float(max(1, step_idx))
    val_acc = tracker["corr"] / max(1.0, tracker["total"])
    val_chi_acc = tracker["chi_corr"] / max(1.0, tracker["chi_total"])
    val_eng_acc = tracker["eng_corr"] / max(1.0, tracker["eng_total"])
    val_ar_acc = tracker["ar_corr"] / max(tracker["ar_total"], 1.0)
    val_ar_exact = tracker["ar_exact"] / max(tracker["ar_seq_total"], 1.0)

    val_eng_loss = (
        tracker["eng_loss"] / float(max(1, step_idx)) if "eng_loss" in tracker else 0.0
    )
    eng_ppl_str = (
        f" | EngPPL: {math.exp(min(val_eng_loss, 20.0)):.1f}"
        if val_eng_loss > 0
        else ""
    )

    if is_master:
        print(
            f"[Validation Epoch {epoch}] TotalLoss: {val_loss:.4f} | "
            f"GlossAcc(TF): {val_acc*100:.2f}% | GlossAcc(AR): {val_ar_acc*100:.2f}% | ExactMatch(AR): {val_ar_exact*100:.2f}% | "
            f"ChiAcc: {val_chi_acc*100:.2f}% | EngAcc: {val_eng_acc*100:.2f}%{eng_ppl_str}",
            flush=True,
        )

    if "para_loader" in locals():
        del para_loader

    gc.collect()

    try:
        import torch_xla.core.xla_model as xm

        xm.wait_device_ops()
    except Exception:
        pass

    # Removed validation temporal resolution discrepancy (Claims 51, 77)

    return {
        "loss": val_loss,
        "gloss_acc": val_acc * 100.0,
        "ar_acc": val_ar_acc * 100.0,
        "ar_exact": val_ar_exact * 100.0,
        "chicago_acc": val_chi_acc * 100.0,
        "english_acc": val_eng_acc * 100.0,
    }


def _tpu_worker_fn(rank, args):
    """Main entrypoint for each TPU worker process."""
    global IS_TPU
    if args.tpu and _XLA_AVAILABLE:
        IS_TPU = True

    if IS_TPU:
        try:
            import torch_xla.debug.profiler as xp

            # Start the profiler server on master rank only (Bug V: was started on all ranks)
            if rank == 0:
                server = xp.start_server(9012)
        except Exception:
            pass

        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm

        world_size = xr.world_size()
        assert rank < world_size
        is_master = rank == 0
        if is_master:
            print(
                f"[DEBUG 5/8] PJRT TPU runtime initialized. World size: {world_size}",
                flush=True,
            )
    else:
        world_size = 1
        is_master = True
        if is_master:
            print(
                f"[DEBUG 5/8] Non-TPU Worker initialized. Device: {'cuda' if torch.cuda.is_available() else 'cpu'}",
                flush=True,
            )

    try:
        if IS_TPU:
            try:
                import torch_xla

                device = torch_xla.device()
            except Exception:
                device = xm.xla_device()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"FAILED TO INITIALIZE TPU OR GET DEVICE: {e}", flush=True)
        time.sleep(2)
        os._exit(1)

    # Auto Memory Guard: Cap per-core micro-batch size to 16 for large sequence lengths / dim_d-model to guarantee fitting inside 16GB HBM
    requested_batch_size = args.batch_size
    target_per_core_batch = max(1, requested_batch_size // world_size)
    if IS_TPU:
        max_safe_batch = 16 if (args.max_len >= 384 or args.d_model >= 384) else 32
        if target_per_core_batch > max_safe_batch:
            safe_per_core_batch = max_safe_batch
            auto_accum = max(1, math.ceil(target_per_core_batch / safe_per_core_batch))
            args.accum_steps = max(args.accum_steps, auto_accum)
            effective_loader_batch = safe_per_core_batch
            if is_master:
                print(
                    f"[INFO] TPU Memory Guard Active: Capping per-core loader batch to {safe_per_core_batch} "
                    f"(Requested total: {requested_batch_size}, World size: {world_size}, MaxLen: {args.max_len}). "
                    f"Auto-setting accum_steps={args.accum_steps} for 100% OOM safety & identical gradient updates.",
                    flush=True,
                )
        else:
            effective_loader_batch = target_per_core_batch
    else:
        effective_loader_batch = requested_batch_size

    # TPU System RAM & Deadlock Guard: Force num_dataloader_workers=0 on multi-core TPU to eliminate nested fork queue deadlocks
    effective_num_dl_workers = 0 if IS_TPU else args.num_dataloader_workers
    if IS_TPU and args.num_dataloader_workers > 0 and is_master:
        print(
            "[INFO] TPU Deadlock Guard: Set num_dataloader_workers to 0 on multi-core TPU. "
            "This eliminates nested multiprocessing fork queue deadlocks while maintaining maximum streaming performance.",
            flush=True,
        )

    if hasattr(args, "phase1_epochs") and args.phase1_epochs > 0:
        text_pretrain_loop(args, device, is_master, per_core_batch=effective_loader_batch)

    if is_master:
        print(
            f"[DEBUG 6/8] Resolving dataset paths & loading vocabulary map from '{args.data_dir}'...",
            flush=True,
        )

    data_dir = Path(args.data_dir)
    has_pt_files = data_dir.exists() and (
        next(data_dir.glob("*.pt"), None) is not None
        or next(data_dir.rglob("*.pt"), None) is not None
    )
    if not data_dir.exists() or not has_pt_files:
        candidate_dirs = [
            Path("./asl_preprocessed_phase1"),
            Path("/kaggle/input/asl-preprocessed-phase1"),
            Path("/kaggle/input/frakenstein-asl/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"
            ),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"
            ),
            Path("/kaggle/input/frakenstein-asl"),
            Path.cwd(),
        ]
        candidate_dirs = [
            cd
            for cd in candidate_dirs
            if os.name != "nt" or not str(cd).startswith("/kaggle/")
        ]
        for cd in candidate_dirs:
            if cd.exists() and (
                next(cd.glob("*.pt"), None) is not None
                or next(cd.glob("shard_*.pt"), None) is not None
            ):
                data_dir = cd
                if is_master:
                    print(
                        f"[INFO] Auto-resolved dataset directory to: {data_dir}",
                        flush=True,
                    )
                break
        if (
            (not data_dir.exists() or next(data_dir.rglob("*.pt"), None) is None)
            and os.name != "nt"
            and Path("/kaggle/input").exists()
        ):
            try:
                # E47: Sort rglob results to ensure deterministic dataset resolution across XLA processes
                for pt_file in sorted(list(Path("/kaggle/input").rglob("*.pt"))):
                    candidate = pt_file.parent
                    data_dir = candidate
                    if is_master:
                        print(
                            f"[INFO] Auto-resolved dataset directory via rglob to: {data_dir}",
                            flush=True,
                        )
                    break
            except Exception:
                pass

    train_loader = create_dataloader(
        dataset_dir=data_dir,
        split="train",
        batch_size=effective_loader_batch,
        max_len=args.max_len,
        worker_idx=rank if _XLA_AVAILABLE else 0,
        num_workers=world_size,
        num_dataloader_workers=effective_num_dl_workers,
        shuffle=True,
        augment=True,
        streamed=getattr(args, "streamed_dataset", False),
        use_bpe=not getattr(args, "disable_bpe", False),
    )

    try:
        if hasattr(train_loader, "__len__") and len(train_loader) == 0:
            raise RuntimeError(
                f"[FATAL DATALOADER ERROR] Training dataloader initialized with 0 batches! "
                f"Dataset directory '{data_dir}' contains no valid training records."
            )
    except TypeError:
        pass

    label_to_idx = getattr(train_loader.dataset, "label_to_idx", {})
    if not label_to_idx:
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        possible_dirs = [
            data_dir,
            data_dir.parent,
            Path("./asl_preprocessed_phase1"),
            Path.cwd(),
        ]
        possible_filenames = [
            "vocabulary_mapping_global.json",
            "vocabulary_mapping_train.json",
            "vocab_map.json",
            "metadata.json",
        ]
        possible_dirs = [
            d
            for d in possible_dirs
            if os.name != "nt" or not str(d).startswith("/kaggle/")
        ]
        for dim_d in possible_dirs:
            if dim_d.exists():
                for fn in possible_filenames:
                    vp = dim_d / fn
                    if vp.exists():
                        try:
                            with open(vp, "r", encoding="utf-8") as f:
                                raw_map = json.load(f)
                            if "normalize_vocabulary" in globals():
                                label_to_idx = normalize_vocabulary(raw_map)
                            elif (
                                isinstance(raw_map, dict) and "label_to_idx" in raw_map
                            ):
                                label_to_idx = raw_map["label_to_idx"]
                            elif isinstance(raw_map, dict):
                                label_to_idx = raw_map
                            if label_to_idx:
                                break
                        except Exception:
                            pass
                if label_to_idx:
                    break

    if not label_to_idx:
        raise ValueError(
            f"Failed to load vocabulary mapping from dataset directory '{data_dir}' or candidate Kaggle input paths."
        )

    vocab = GlossVocabulary(label_to_idx=label_to_idx)

    val_loader = create_dataloader(
        dataset_dir=data_dir,
        split="val",
        batch_size=effective_loader_batch,
        max_len=args.max_len,
        worker_idx=rank if _XLA_AVAILABLE else 0,
        num_workers=world_size,
        num_dataloader_workers=effective_num_dl_workers,
        shuffle=False,
        augment=False,
        streamed=getattr(args, "streamed_dataset", False),
        use_bpe=not getattr(args, "disable_bpe", False),
    )

    class_weights_tensor = None
    try:
        raw_ds = getattr(train_loader, "dataset", None)
        c_counts = getattr(raw_ds, "class_counts", {}) if raw_ds else {}
        
        offset = GlossVocabulary.OFFSET
        num_classes = len(vocab) - offset

        # Aggregate counts globally across TPUs using RAW class space
        raw_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
        for raw_idx, cnt in c_counts.items():
            # Skip any negative or out-of-range raw indices.
            if isinstance(raw_idx, int) and 0 <= raw_idx < num_classes:
                raw_counts[raw_idx] = float(cnt)

        if IS_TPU:
            import torch_xla.core.xla_model as xm
            raw_counts = xm.all_reduce(xm.REDUCE_SUM, raw_counts)

        # Move raw_counts to CPU memory ONCE to avoid synchronous TPU .item() host transfers
        raw_counts_np = raw_counts.detach().cpu().numpy()
        w_vec_np = np.ones(len(vocab), dtype=np.float32)

        nz_mask = raw_counts_np[:num_classes] > 0
        if np.any(nz_mask):
            max_c = float(raw_counts_np[:num_classes][nz_mask].max())
            idxs = offset + np.flatnonzero(nz_mask)
            valid_counts = raw_counts_np[:num_classes][nz_mask]
            w_vec_np[idxs] = np.clip(
                (max_c / valid_counts) ** 0.35, 1.0, 10.0
            )

        class_weights_tensor = torch.from_numpy(w_vec_np).to(device)
        if is_master:
            print(
                "[INFO] Class weighting: ENABLED (calculated from raw dataset class_counts)",
                flush=True,
            )

        assert len(class_weights_tensor) == len(
            vocab
        ), "class_counts contains raw IDs, we properly added the OFFSET."
    except Exception as e:
        if is_master:
            print(f"[WARN] Class weighting: DISABLED due to exception: {e}", flush=True)
        class_weights_tensor = torch.ones(
            len(vocab), dtype=torch.float32, device=device
        )

    import json as _json_cv

    _chicago_vocab_path = Path(args.data_dir) / "chicago_vocab.json"
    if _chicago_vocab_path.exists():
        chicago_vocab = GlossVocabulary(
            label_to_idx=_json_cv.load(open(_chicago_vocab_path, encoding="utf-8"))
        )
    else:
        chicago_vocab = GlossVocabulary(label_to_idx={})
    chicago_vocab_size = chicago_vocab.vocab_size
    english_vocab = EnglishVocabulary(
        vocab_path=os.path.join(data_dir, "english_vocab.json")
    )
    english_vocab_size = len(english_vocab)

    asl_lex_csv = data_dir / "signdata.csv"
    if hasattr(args, "asl_lex_csv") and args.asl_lex_csv:
        if os.name != "nt" or not args.asl_lex_csv.startswith("/home/"):
            cand_p = Path(args.asl_lex_csv)
            if cand_p.exists():
                asl_lex_csv = cand_p

    eng_v_obj = getattr(
        train_loader.dataset,
        "eng_vocab",
        getattr(train_loader.dataset, "english_vocab", english_vocab),
    )
    eng_vsize = len(eng_v_obj) if eng_v_obj is not None else 20005

    import hashlib

    if eng_v_obj is not None:
        if getattr(eng_v_obj, "use_bpe", False):
            hash_str = str(getattr(eng_v_obj, "model_name", "bpe")) + str(
                len(eng_v_obj)
            )
        else:
            token_map = getattr(eng_v_obj, "token_to_id", {})
            hash_str = f"{len(token_map)}_" + "_".join(list(token_map.keys()))
        eng_hash = int(hashlib.md5(hash_str.encode()).hexdigest()[:12], 16)
    else:
        eng_hash = 0
    if IS_TPU:
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

    assert (
        GlossVocabulary.PAD_ID == 0
    ), "CTC requires PAD_ID == 0 for blank token mapping"
    assert GlossVocabulary.BOS_ID == 1, "Expected BOS_ID == 1"
    assert GlossVocabulary.EOS_ID == 2, "Expected EOS_ID == 2"
    assert GlossVocabulary.UNK_ID == 3, "Expected UNK_ID == 3"
    CTC_BLANK_ID = GlossVocabulary.PAD_ID

    model = ASLFoundationModel(
        vocab_size=vocab.vocab_size,
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
        scale_embeddings=True,
        enable_aux_decoders=getattr(args, "enable_aux_decoders", False),
        is_causal=getattr(args, "is_causal", False),
    ).to(device, dtype=torch.bfloat16 if IS_TPU else None)

    # Automatically load Phase 1 text pretraining weights if they were just trained sequentially
    phase1_ep = getattr(args, 'phase1_epochs', 0)
    llm_ckpt_path = os.path.join(
        args.save_dir, f"asl_llm_{phase1_ep}.pt"
    )
    if phase1_ep > 0:
        if os.path.exists(llm_ckpt_path):
            if is_master:
                print(
                    f"[INFO] Phase 1 Pre-training detected. Loading weights from {llm_ckpt_path}...",
                    flush=True,
                )
            ckpt = torch.load(llm_ckpt_path, map_location="cpu", weights_only=False)
            
            # If the checkpoint contains full model keys (e.g. english_decoder.token_emb.weight)
            if any(k.startswith("english_decoder.") or k.startswith("decoder.") for k in ckpt.keys()):
                missing, unexpected = model.load_state_dict(ckpt, strict=False)
                if is_master:
                    print(f"[INFO] Loaded full Phase 1 model. Missing keys: {len(missing)}")
            else:
                # Legacy checkpoint: contains only the english_decoder weights directly
                target_decoder = (
                    model.english_decoder
                    if model.english_decoder is not None
                    else model.decoder
                )
                filtered_ckpt = {}
                target_state = target_decoder.state_dict()
                for k, v in ckpt.items():
                    if k in target_state and target_state[k].shape == v.shape:
                        filtered_ckpt[k] = v
                    elif k.startswith("decoder."):
                        stripped_k = k[8:]
                        if (
                            stripped_k in target_state
                            and target_state[stripped_k].shape == v.shape
                        ):
                            filtered_ckpt[stripped_k] = v

                if len(filtered_ckpt) == 0:
                    raise RuntimeError(
                        "FATAL: Phase-1 checkpoint matched no keys in the target decoder. This would cause a silent partial transfer."
                    )

                target_decoder.load_state_dict(filtered_ckpt, strict=False)
                if is_master:
                    print(f"[INFO] Loaded legacy Phase 1 checkpoint into English Decoder.")

            del ckpt
            gc.collect()

    if IS_TPU:
        xm.broadcast_master_param(model)

    if args.compile and hasattr(torch, "compile"):
        if IS_TPU:
            if is_master:
                print(
                    "[!] WARNING: --compile is disabled on TPU because PJRT natively compiles XLA graphs.",
                    flush=True,
                )
        else:
            if is_master:
                print(
                    "[*] JIT Compiling model with PyTorch Inductor (torch.compile)...",
                    flush=True,
                )
            try:
                model = torch.compile(model)
            except Exception as _e:
                if is_master:
                    print(f"[!] Fatal Error during torch.compile: {_e}", flush=True)
                raise _e

    loss_wrapper = HomoscedasticLossWrapper().to(device)
    if IS_TPU:
        xm.broadcast_master_param(loss_wrapper)

    supcon_fn = SupervisedContrastiveLoss().to(device)

    try:
        train_loader_len = len(train_loader)
    except TypeError:
        train_loader_len = 2500
        
    global_min_batches = train_loader_len
    if IS_TPU:
        global_min_batches = int(
            xm.mesh_reduce(
                "global_min_batches", train_loader_len, lambda input_x: min(input_x)
            )
        )

    optimizer = torch.optim.AdamW(
        _get_optimizer_groups(model, loss_wrapper, args.weight_decay),
        lr=args.lr,
    )
    # The scheduler must be configured for the number of *optimizer* steps,
    # not raw batch steps. OneCycleLR.step() is called once per optimizer update.
    effective_steps_per_epoch = max(
        1, math.ceil(global_min_batches / max(1, args.accum_steps))
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=effective_steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=5.0,
    )

    scaler = None
    if args.precision == "float16" and "cuda" in str(device).lower():
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    if hasattr(args, "resume") and args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(f"FATAL: Checkpoint mismatch during resume: {e}")

        if "loss_wrapper_state_dict" in ckpt:
            try:
                loss_wrapper.load_state_dict(
                    ckpt["loss_wrapper_state_dict"], strict=True
                )
            except RuntimeError as e:
                raise RuntimeError(f"FATAL: loss_wrapper mismatch during resume: {e}")

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
        if "rng_state_xla" in ckpt and ckpt["rng_state_xla"] is not None and IS_TPU:
            try:
                import torch_xla.core.xla_model as xm
                xm.set_rng_state(ckpt["rng_state_xla"])
            except:
                pass

        start_epoch = ckpt.get("epoch", 0) + 1
        ema_state_dict_to_load = ckpt.get("ema_state_dict", None)
        del ckpt

        gc.collect()

    save_dir = Path(args.save_dir)
    if is_master:
        save_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 70, flush=True)
        print(
            f"       STARTING TPU MULTI-TASK FOUNDATION MODEL TRAINING ({args.epochs} EPOCHS)",
            flush=True,
        )
        total_params = sum(prob_p.numel() for prob_p in model.parameters())
        print(
            f"       Model: {args.num_layers} layers | d_model={args.d_model} | {total_params / 1e6:.1f}M params",
            flush=True,
        )
        print("=" * 70, flush=True)

    ema = ModelEMA(model)
    loss_ema = ModelEMA(loss_wrapper)
    if "ema_state_dict_to_load" in locals() and ema_state_dict_to_load is not None:
        for key_k_lower, val_v in ema_state_dict_to_load.items():
            if key_k_lower in ema.shadow:
                ema.shadow[key_k_lower].copy_(val_v.to(ema.shadow[key_k_lower].device))
        if is_master:
            print("[+] Restored EMA state from checkpoint", flush=True)
        del ema_state_dict_to_load
        gc.collect()

    if IS_TPU:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
        xm.rendezvous("model_ema_init_complete")

    if is_master:
        print(
            "[DEBUG 7/8] DataLoaders and ASLFoundationModel initialized successfully!",
            flush=True,
        )
        print(f"[DEBUG 8/8] Starting main training loop (Epoch {start_epoch})...", flush=True)

    try:
        for epoch in range(start_epoch, args.epochs + 1):

            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if (
                hasattr(train_loader.dataset, "shared_epoch")
                and train_loader.dataset.shared_epoch is not None
            ):
                train_loader.dataset.shared_epoch.value = epoch
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

            # --- ADD THIS TO RAMP UP NOISE CURRICULUM ---
            if hasattr(train_loader.dataset, "set_noise_level"):
                # Ensure curriculum actually starts at 0.0 on epoch 1
                train_loader.dataset.set_noise_level((epoch - 1) / max(1, args.epochs - 1))
            # --------------------------------------------

            train_metrics = train_epoch_tpu(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                loss_wrapper=loss_wrapper,
                ema=ema,
                loss_ema=loss_ema,
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
                args=args,
            )

            # --- VALIDATION LOOP ---
            raw_m = model.module if hasattr(model, "module") else model
            if ema is not None:
                ema.apply_shadow(raw_m)
                loss_ema.apply_shadow(loss_wrapper)

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
                args=args,
            )

            if is_master:
                print(
                    f"[Validation Epoch {epoch}] SeqLoss: {val_metrics['loss']:.4f} | TokenAcc(TF): {val_metrics['gloss_acc']:.2f}% | TokenAcc(AR): {val_metrics['ar_acc']:.2f}% | ExactMatch(AR): {val_metrics['ar_exact']:.2f}%",
                    flush=True,
                )
                try:
                    print_console_line_charts(epoch)
                    save_epoch_loss_curves_png(epoch)
                except Exception:
                    pass

            if ema is not None:
                ema.restore(raw_m)
                loss_ema.restore(loss_wrapper)
                if IS_TPU:
                    import torch_xla.core.xla_model as xm

            if IS_TPU:
                import torch_xla.core.xla_model as xm

                xm.mark_step()
                xm.rendezvous("pre_checkpoint_save")

            # Save checkpoint every 5 epochs or on final epoch to save memory and disk quota
            if epoch % getattr(args, "save_every_epoch", 1) == 0 or epoch == args.epochs:
                import random as py_random

                ckpt_path = save_dir / f"asl_model_epoch_{epoch}.pt"
                latest_path = save_dir / "asl_model_latest.pt"

                if IS_TPU:
                    import torch_xla.core.xla_model as xm

                    xm.mark_step()
                    xm.rendezvous("pre_checkpoint_build")

                    # All 8 TPU ranks construct cpu_state together so state_dict XLA graph sync executes across all ranks simultaneously
                    cpu_state = {
                        "epoch": epoch,
                        "model_state_dict": raw_m.state_dict(),
                        "ema_state_dict": (
                            {
                                k: v.cpu().detach()
                                for k, v in ema.shadow.items()
                            }
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
                        "rng_state_random": py_random.getstate(),
                        "rng_state_xla": xm.get_rng_state(),
                    }
                    gc.collect()
                    import torch_xla.core.xla_model as xm

                    xm.mark_step()
                    tmp_ckpt_path = str(ckpt_path) + ".tmp"
                    tmp_latest_path = str(latest_path) + ".tmp"
                    xm.save(cpu_state, tmp_ckpt_path)
                    if is_master:
                        os.replace(tmp_ckpt_path, str(ckpt_path))
                    xm.save(cpu_state, tmp_latest_path)
                    if is_master:
                        os.replace(tmp_latest_path, str(latest_path))
                    del cpu_state
                else:
                    if is_master:
                        cpu_state = {
                            "epoch": epoch,
                            "model_state_dict": raw_m.state_dict(),
                            "ema_state_dict": (
                                {k: v.detach().cpu() for k, v in ema.shadow.items()}
                                if ema is not None
                                else None
                            ),
                            "loss_wrapper_state_dict": loss_wrapper.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": (
                                scheduler.state_dict()
                                if scheduler is not None
                                else None
                            ),
                            "scaler_state_dict": (
                                scaler.state_dict()
                                if "scaler" in locals() and scaler is not None
                                else None
                            ),
                            "rng_state_torch": torch.get_rng_state(),
                            "rng_state_numpy": np.random.get_state(),
                            "rng_state_random": py_random.getstate(),
                            "rng_state_xla": None,
                        }
                        gc.collect()

                        torch.save(cpu_state, str(ckpt_path))
                        torch.save(cpu_state, str(latest_path))
                        del cpu_state

                if is_master:
                    print(f"[+] Saved checkpoint to {ckpt_path}", flush=True)
                    try:
                        all_ckpts = sorted(
                            list(save_dir.glob("asl_model_epoch_*.pt")),
                            key=lambda prob_p: int(prob_p.stem.split("_")[-1]),
                        )
                        if len(all_ckpts) > 5:
                            for old_c in all_ckpts[:-5]:
                                ep_num = int(old_c.stem.split("_")[-1])
                                if ep_num % 10 != 0 and ep_num != epoch:
                                    old_c.unlink(missing_ok=True)
                    except Exception:
                        pass

            # Explicit garbage collection of massive dicts & flush XLA IR graph
            if "cpu_state" in locals() and cpu_state is not None:
                del cpu_state
                cpu_state = None

            if IS_TPU:
                # pyrefly: ignore [missing-import]
                import torch_xla.core.xla_model as xm

                xm.mark_step()
                xm.rendezvous("post_checkpoint_save")

            gc.collect()
            gc.collect()

    except Exception as e:
        import traceback
        import sys

        print(f"CRITICAL PYTHON EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        time.sleep(2)
        sys.exit(1)


def inverted_gloss_pretrain_loop(args, device, is_master):
    if IS_TPU:
        import torch_xla.core.xla_model as xm
    from dataset import (
        ASLGPC12Dataset,
        EnglishVocabulary,
        GlossVocabulary,
    )

    if is_master:
        print(
            f"Starting Phase 2 Inverted Gloss Training (English -> Gloss) for {args.epochs} epochs..."
        )
        os.makedirs(args.save_dir, exist_ok=True)

    eng_vocab = EnglishVocabulary(
        vocab_path=os.path.join(args.data_dir, "english_vocab.json")
    )
    eng_pad_id = eng_vocab.PAD_ID
    import json

    gloss_vocab = (
        GlossVocabulary(
            label_to_idx=json.load(
                open(os.path.join(args.data_dir, "vocab_map.json"), encoding="utf-8")
            )
        )
        if os.path.exists(os.path.join(args.data_dir, "vocab_map.json"))
        else GlossVocabulary(label_to_idx={})
    )
    bpe_max_len = max(args.max_len, 384)
    dataset = ASLGPC12Dataset(
        csv_path=(
            args.data_dir + "/train.csv"
            if not args.data_dir.endswith(".csv")
            else args.data_dir
        ),
        eng_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=bpe_max_len,
        reverse=True,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=xm.get_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True,
        drop_last=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_dataloader_workers,
        collate_fn=functools.partial(phase2_collate_fn, max_len=args.max_len, eng_pad_id=eng_pad_id),
        drop_last=True,
    )
    
    val_dataset = ASLGPC12Dataset(
        csv_path=(args.data_dir + "/val.csv" if not args.data_dir.endswith(".csv") else args.data_dir.replace("train.csv", "val.csv")),
        eng_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=bpe_max_len,
        reverse=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_dataloader_workers,
        collate_fn=functools.partial(phase2_collate_fn, max_len=args.max_len, eng_pad_id=eng_pad_id),
        drop_last=False,
    )

    # Model (we use the decoder but with gloss vocab instead of english vocab)
    model = ASLFoundationModel(
        enable_aux_decoders=args.enable_aux_decoders,
        num_keypoints=60,
        d_enc=args.d_model,
        nhead_enc=args.nhead,
        num_enc_layers=args.num_layers,
        ffn_enc=args.d_model * 4,
        d_dec=args.d_model,
        nhead_dec=args.nhead,
        num_dec_layers=args.num_layers,
        ffn_dec=args.d_model * 4,
        dropout=args.dropout,
        max_enc_len=args.max_len,
        max_dec_len=args.max_len,
        csv_path=args.aslg_csv,
        use_mamba=getattr(args, "use_mamba", True),
        use_swin_1d=getattr(args, "use_swin", False),
        swin_window_size=getattr(args, "swin_window", 128),
        vocab_size=len(gloss_vocab),
    )

    # Using the gloss decoder (which uses GlossVocabulary logic)
    decoder = model.decoder
    # Tie embeddings for gloss decoder BEFORE moving to device
    decoder.token_emb.weight = decoder.lm_head.weight

    model = model.to(device, dtype=torch.bfloat16 if IS_TPU else None)
    decoder = model.decoder
    time_emb = model.time_emb

    # We also need an english embedding layer for the cross-attention
    eng_vocab_size = len(eng_vocab)
    english_emb = nn.Embedding(eng_vocab_size, args.d_model, padding_idx=0).to(
        device, dtype=torch.bfloat16 if IS_TPU else None
    )

    if IS_TPU:
        import torch_xla.core.xla_model as xm

        xm.broadcast_master_param(model)
        xm.broadcast_master_param(english_emb)

    params = list(decoder.parameters()) + list(english_emb.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.05))
    from torch.optim.lr_scheduler import CosineAnnealingLR
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(dataloader))

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        decoder.train()
        english_emb.train()
        if IS_TPU:
            loader = pl.MpDeviceLoader(dataloader, device, batches_per_execution=getattr(args, "batches_per_execution", 16))
        else:
            loader = dataloader

        for step, batch in enumerate(loader):
            # input is english, target is gloss
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]

            mask = input_ids == 0  # Qwen PAD
            with torch.autocast(
                device_type="cuda" if "cuda" in device.type else "xla",
                dtype=getattr(args, "precision", torch.bfloat16),
                enabled=not getattr(args, "disable_amp", False),
            ):
                memory = english_emb(input_ids)
                memory = time_emb(memory)

                tgt_in = target_ids[:, :-1]
                tgt_out = target_ids[:, 1:]

                out = decoder(
                    tgt_in,
                    memory,
                    tgt_key_padding_mask=(tgt_in == GlossVocabulary.PAD_ID),
                    memory_key_padding_mask=mask,
                )

                logits = out[0] if isinstance(out, tuple) else out

                valid_mask = (tgt_out != GlossVocabulary.PAD_ID) & (
                    tgt_out != GlossVocabulary.EOS_ID
                )
                loss, _ = compute_seq_and_eos_loss(
                    logits,
                    tgt_out,
                    valid_mask,
                    torch.zeros_like(valid_mask),
                    label_smoothing=0.1,
                )

            effective_accum = min(
                max(1, args.accum_steps),
                len(dataloader)
                - (step // max(1, args.accum_steps)) * max(1, args.accum_steps),
            )
            ((loss * args.bwd_weight) / effective_accum).backward()
            if (step + 1) % effective_accum == 0 or (step + 1) == len(dataloader):
                if getattr(args, "grad_clip", 0.0) > 0.0:
                    xm.clip_grad_norm_(params, args.grad_clip)
                xm.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % args.log_freq == 0:
                if is_master:
                    def _print_loss(ep, s, l_t):
                        print(f"Phase 2 Train | Epoch {ep:03d} | Step {s:04d} | Loss: {l_t.item():.4f}", flush=True)
                    xm.add_step_closure(_print_loss, args=(epoch, step, loss.detach().cpu()))

        if "para_loader" in locals():
            del para_loader

        # Validation loop
        decoder.eval()
        english_emb.eval()
        val_loss_sum = torch.tensor(0.0, device=device)
        val_steps = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(device)
                target_ids = batch["target_ids"].to(device)
                mask = input_ids == 0
                with torch.autocast(
                    device_type="cuda" if "cuda" in device.type else "xla",
                    dtype=getattr(args, "precision", torch.bfloat16),
                    enabled=not getattr(args, "disable_amp", False),
                ):
                    memory = english_emb(input_ids)
                    tgt_in = target_ids[:, :-1]
                    tgt_out = target_ids[:, 1:]
                    out = decoder(
                        tgt_in, memory,
                        tgt_key_padding_mask=(tgt_in == GlossVocabulary.PAD_ID),
                        memory_key_padding_mask=mask,
                    )
                    logits = out[0] if isinstance(out, tuple) else out
                    valid_mask = (tgt_out != GlossVocabulary.PAD_ID) & (tgt_out != GlossVocabulary.EOS_ID)
                    loss, _ = compute_seq_and_eos_loss(logits, tgt_out, valid_mask, torch.zeros_like(valid_mask), label_smoothing=0.0)
                    val_loss_sum += loss
                    val_steps += 1.0
        
        if IS_TPU:
            import torch_xla.core.xla_model as xm
            val_loss_sum = xm.all_reduce(xm.REDUCE_SUM, val_loss_sum)
            val_steps = xm.all_reduce(xm.REDUCE_SUM, val_steps)
        
        avg_val_loss = float((val_loss_sum / max(1.0, float(val_steps))).cpu())
        
        if is_master:
            xm.master_print(f"Phase 2 Val Epoch {epoch} | Loss: {avg_val_loss:.4f}")
            xm.master_print(f"Phase 2 Train Epoch {epoch} finished.")
            save_dict = {
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "english_emb": english_emb.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            tmp_save = os.path.join(args.save_dir, f"inverted_gloss_model_epoch_{epoch}.pt.tmp")
            final_save = os.path.join(args.save_dir, f"inverted_gloss_model_epoch_{epoch}.pt")
            xm.save(save_dict, tmp_save)
            os.replace(tmp_save, final_save)
            
            tmp_latest = os.path.join(args.save_dir, "inverted_gloss_model_latest.pt.tmp")
            final_latest = os.path.join(args.save_dir, "inverted_gloss_model_latest.pt")
            xm.save(save_dict, tmp_latest)
            os.replace(tmp_latest, final_latest)


def pseudo_gloss_gen_loop(args, device, is_master):
    """
    Phase 2 Pseudo-Gloss Generation Loop.

    Uses the Phase 1 pre-trained model to generate pseudo-gloss labels for the video dataset,
    bridging the modality gap for subsequent end-to-end training.

    Args:
        args (argparse.Namespace): Command line arguments.
        device (torch.device): The device to run generation on.
        is_master (bool): True if this process is the master node (rank 0).
    """
    if IS_TPU:
        import torch_xla.core.xla_model as xm

    if is_master:
        print("Starting Phase 2 Pseudo-Gloss Generation on TPU...")
        out_dir = os.path.join(args.save_dir, "pseudo_gloss_data")
        os.makedirs(out_dir, exist_ok=True)
        # Copy metadata.json
        meta_src = os.path.join(args.data_dir, "metadata.json")
        if os.path.exists(meta_src):
            with open(meta_src, "r", encoding="utf-8") as f:
                meta = json.load(f)
            # Flag has_valid_gloss as true globally
            for key in meta:
                meta[key]["has_valid_gloss"] = True
            with open(
                os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(meta, f)

    eng_vocab = EnglishVocabulary(
        vocab_path=os.path.join(args.data_dir, "english_vocab.json")
    )

    gloss_vocab = None
    candidate_vocab_names = ["vocab_map.json", "vocabulary_mapping_train.json", "vocabulary_mapping_global.json", "metadata.json"]
    candidate_vocab_dirs = [
        args.data_dir,
        os.path.dirname(args.data_dir),
        "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_dataset",
        "/dev/shm/dataset",
        "/dev/shm",
    ]
    for d in candidate_vocab_dirs:
        if d and os.path.exists(d):
            for fname in candidate_vocab_names:
                vpath = os.path.join(d, fname)
                if os.path.exists(vpath):
                    try:
                        with open(vpath, "r", encoding="utf-8") as f:
                            raw_map = json.load(f)
                        if isinstance(raw_map, dict) and "label_to_idx" in raw_map:
                            gloss_vocab = GlossVocabulary(label_to_idx=raw_map["label_to_idx"])
                        elif isinstance(raw_map, dict):
                            gloss_vocab = GlossVocabulary(label_to_idx=raw_map)
                        if gloss_vocab is not None:
                            break
                    except Exception:
                        pass
            if gloss_vocab is not None:
                break
    if gloss_vocab is None:
        gloss_vocab = GlossVocabulary(label_to_idx={})

    dataset = ASLStreamedDataset(
        dataset_dir=args.data_dir,
        split="val",  # Prevent infinite train shuffle loop
        english_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=args.max_len,
        worker_idx=xm.get_ordinal(),
        num_workers=get_xla_world_size(),
        shuffle_buffer_size=1,
    )

    # We will manually load shards to process them sequentially and save them back
    # Distribute shards among workers
    # Since dataset filters shard_files in __init__ based on world_size, dataset.shard_files is already local to this TPU core
    local_shards = dataset.shard_files

    model = ASLFoundationModel(
        channels_per_kp=9,
        num_enc_layers=0,
        d_enc=args.d_model,
        english_vocab_size=len(eng_vocab),
        eng_pad_id=eng_vocab.PAD_ID,
        eng_bos_id=eng_vocab.BOS_ID,
        eng_eos_id=eng_vocab.EOS_ID,
        d_dec=args.d_model,
        nhead_enc=args.nhead,
        nhead_dec=args.nhead,
        num_dec_layers=args.num_layers,
        max_enc_len=args.max_len,
        max_dec_len=args.max_len,
        drop_path_rate=0.0,
    ).to(device, dtype=torch.bfloat16 if IS_TPU else None)

    decoder = model.decoder
    time_emb = model.time_emb
    eng_vocab_size = len(eng_vocab)
    english_emb = nn.Embedding(eng_vocab_size, args.d_model, padding_idx=0).to(
        device, dtype=torch.bfloat16 if IS_TPU else None
    )

    if IS_TPU:
        xm.broadcast_master_param(model)
        xm.broadcast_master_param(english_emb)

    # Load weights
    candidate_ckpts = [
        os.path.join(args.save_dir, "inverted_gloss_model_latest.pt"),
        os.path.join(args.save_dir, "inverted_gloss_model.pt"),
    ]
    ckpt_path = None
    for p in candidate_ckpts:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        decoder.load_state_dict(ckpt["decoder"] if "decoder" in ckpt else ckpt)
        if "english_emb" in ckpt:
            english_emb.load_state_dict(ckpt["english_emb"])
    else:
        raise ValueError(
            f"FATAL: Inverted gloss model checkpoint not found in {args.save_dir}! Aborting pseudo-label generation to prevent generating garbage labels."
        )

    decoder.eval()
    english_emb.eval()

    out_dir = os.path.join(args.save_dir, "pseudo_gloss_data")

    with torch.no_grad():
        for shard_path in local_shards:
            print(f"[Core {xm.get_ordinal()}] Processing {shard_path}...")
            items = torch.load(shard_path, map_location="cpu", weights_only=False)
            if isinstance(items, dict):
                items_list = list(items.values())
            else:
                items_list = items

            # Batch items for faster generation
            batch_size = 64
            for i in range(0, len(items_list), batch_size):
                batch = items_list[i : i + batch_size]
                text_ids_list = []
                for rec in batch:
                    # Some files have 'english', some have 'english_seq'
                    if "english_seq" in rec and len(rec["english_seq"]) > 0:
                        t_ids = [tid for tid in rec["english_seq"] if tid not in (eng_vocab.BOS_ID, eng_vocab.EOS_ID, getattr(eng_vocab, "PAD_ID", 0))]
                    else:
                        eng_str = rec.get("english", "")
                        t_ids = eng_vocab.encode(eng_str)
                    
                    t_ids = (
                        [eng_vocab.BOS_ID]
                        + t_ids[: args.max_len - 2]
                        + [eng_vocab.EOS_ID]
                    )
                    text_ids_list.append(t_ids)

                max_len = args.max_len
                text_padded = torch.full((len(batch), max_len), 0, dtype=torch.long)
                for j, t_ids in enumerate(text_ids_list):
                    text_padded[j, : len(t_ids)] = torch.tensor(t_ids)

                text_padded = text_padded.to(device)
                mask = text_padded == 0

                with torch.autocast(
                    device_type="cuda" if "cuda" in device.type else "xla",
                    dtype=torch.bfloat16,
                    enabled=not getattr(args, "disable_amp", False),
                ):
                    memory = english_emb(text_padded)
                    memory = time_emb(memory)

                    # Autoregressive generation
                    bsz = memory.size(0)
                    gen_ids = torch.full(
                        (bsz, args.max_len + 1),
                        GlossVocabulary.PAD_ID,
                        dtype=torch.long,
                        device=device,
                    )
                    gen_ids[:, 0] = GlossVocabulary.BOS_ID

                    # Pre-allocate static KV Caches
                    kv_heads = decoder.layers[0].self_attn.kv_heads
                    head_dim = decoder.layers[0].self_attn.head_dim
                    num_layers = len(decoder.layers)
                    kv_caches = []
                    for _ in range(num_layers):
                        self_k = torch.zeros(
                            (bsz, kv_heads, args.max_len, head_dim),
                            dtype=memory.dtype,
                            device=device,
                        )
                        self_v = torch.zeros(
                            (bsz, kv_heads, args.max_len, head_dim),
                            dtype=memory.dtype,
                            device=device,
                        )
                        past_zero = torch.tensor([0], device=device, dtype=torch.long)
                        kv_caches.append(((self_k, self_v, past_zero), None))



                    for step in range(args.max_len):
                        tgt_in = gen_ids[:, step : step + 1]

                        out = decoder(
                            tgt_in,
                            memory,
                            memory_key_padding_mask=mask,
                            past_key_values=kv_caches,
                            use_cache=True,
                        )
                        logits = out[0]
                        kv_caches = out[3] if len(out) > 3 else None

                        next_token = logits[:, -1].argmax(dim=-1)
                        gen_ids[:, step + 1] = next_token

                # Move to CPU and extract
                gen_ids = gen_ids.cpu().tolist()

                for j, rec in enumerate(batch):
                    seq = gen_ids[j]
                    if GlossVocabulary.EOS_ID in seq:
                        trimmed_seq = seq[: seq.index(GlossVocabulary.EOS_ID) + 1]
                        rec["has_valid_gloss"] = len(trimmed_seq) > 2
                        rec["gloss_seq"] = trimmed_seq
                    else:
                        rec["has_valid_gloss"] = False
                        rec["gloss_seq"] = seq

            # Save updated shard atomically
            shard_name = os.path.basename(shard_path)
            tmp_shard_path = os.path.join(out_dir, shard_name + ".tmp")
            final_shard_path = os.path.join(out_dir, shard_name)
            torch.save(items, tmp_shard_path)
            os.replace(tmp_shard_path, final_shard_path)

    if IS_TPU:
        xm.rendezvous("pseudo_gloss_done")
    if is_master:
        with open(os.path.join(out_dir, "_SUCCESS"), "w") as f:
            f.write("OK\n")
        xm.master_print("Phase 2 Pseudo-Gloss Generation finished!")


def text_pretrain_loop(args, device, is_master, per_core_batch=None):
    """
    Phase 1 Text Pre-training Loop.

    This function trains the model exclusively on textual datasets (like KDWD and ASLG-PC12)
    to build a robust language model and semantic representations before introducing video inputs.
    """
    if IS_TPU:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
    import functools
    from dataset import Phase1MixedIterable

    if per_core_batch is None:
        world_size = xr.world_size() if IS_TPU else 1
        per_core_batch = max(1, args.batch_size // world_size) if IS_TPU else args.batch_size

    if is_master:
        print(f"Starting Phase 1 Text Pre-training for {args.phase1_epochs} epochs (Per-Core Batch: {per_core_batch})...", flush=True)
        os.makedirs(args.save_dir, exist_ok=True)

    eng_vocab = EnglishVocabulary(
        vocab_path=os.path.join(args.data_dir, "english_vocab.json"),
        use_bpe=not getattr(args, "disable_bpe", False)
    )
    eng_pad_id = eng_vocab.PAD_ID

    gloss_vocab = None
    candidate_vocab_names = ["vocab_map.json", "vocabulary_mapping_train.json", "vocabulary_mapping_global.json", "metadata.json"]
    candidate_vocab_dirs = [
        args.data_dir,
        os.path.dirname(args.data_dir),
        "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_dataset",
        "/dev/shm/dataset",
        "/dev/shm",
    ]
    for d in candidate_vocab_dirs:
        if d and os.path.exists(d):
            for fname in candidate_vocab_names:
                vpath = os.path.join(d, fname)
                if os.path.exists(vpath):
                    try:
                        with open(vpath, "r", encoding="utf-8") as f:
                            raw_map = json.load(f)
                        if isinstance(raw_map, dict) and "label_to_idx" in raw_map:
                            gloss_vocab = GlossVocabulary(label_to_idx=raw_map["label_to_idx"])
                        elif isinstance(raw_map, dict):
                            gloss_vocab = GlossVocabulary(label_to_idx=raw_map)
                        if gloss_vocab is not None:
                            print(f"[INFO] Phase 1 GlossVocabulary loaded from '{vpath}'", flush=True)
                            break
                    except Exception:
                        pass
            if gloss_vocab is not None:
                break

    if gloss_vocab is None:
        raise FileNotFoundError(
            f"[FATAL VOCAB ERROR] Gloss vocabulary map missing in '{args.data_dir}' or candidate directories. "
            f"Phase 1 text pre-training requires valid gloss vocabulary map!"
        )

    bpe_max_len = args.max_len
    dataset = Phase1MixedIterable(
        kdwd_dir=args.kdwd_dir,
        aslg_csv=(
            args.aslg_csv
            if getattr(args, "aslg_csv", "")
            else (
                args.data_dir + "/train.csv"
                if not args.data_dir.endswith(".csv")
                else args.data_dir
            )
        ),
        eng_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=bpe_max_len,
    )

    num_workers = getattr(args, "num_dataloader_workers", 0)
    dl_kwargs = {
        "dataset": dataset,
        "batch_size": per_core_batch,
        "num_workers": num_workers,
        "collate_fn": functools.partial(phase1_collate_fn, max_len=bpe_max_len, eng_pad_id=eng_pad_id),
        "drop_last": True if IS_TPU else False,
    }
    if num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = 4

    dataloader = torch.utils.data.DataLoader(**dl_kwargs)

    # Model instantiation
    model = ASLFoundationModel(
        channels_per_kp=9,
        num_enc_layers=0,
        d_enc=args.d_model,
        vocab_size=len(gloss_vocab),
        d_dec=args.d_model,
        nhead_enc=args.nhead,
        nhead_dec=args.nhead,
        num_dec_layers=args.num_layers,
        max_enc_len=args.max_len,
        max_dec_len=args.max_len,
        english_vocab_size=len(eng_vocab),
        eng_pad_id=eng_vocab.PAD_ID,
        eng_bos_id=eng_vocab.BOS_ID,
        eng_eos_id=eng_vocab.EOS_ID,
        drop_path_rate=0.0,
        enable_aux_decoders=True,
        is_causal=getattr(args, "is_causal", False),
    )

    # Tie embeddings
    model.english_decoder.token_emb.weight = model.english_decoder.lm_head.weight

    # Disable gradients on unused video encoder layers during Phase 1 to conserve TPU HBM
    if hasattr(model, "encoder"):
        model.encoder.requires_grad_(False)

    target_dtype = torch.bfloat16 if getattr(args, "precision", "bfloat16") == "bfloat16" else (
        torch.float16 if getattr(args, "precision", "bfloat16") == "float16" else torch.float32
    ) if (IS_TPU or device.type == "cuda") else torch.float32

    model = model.to(device, dtype=target_dtype)

    if getattr(args, "compile", False):
        if IS_TPU:
            if is_master:
                print("[!] WARNING: --compile is disabled on TPU because PJRT natively compiles XLA graphs. Using torch.compile(backend='openxla') causes nested compilation OOMs!", flush=True)
        else:
            try:
                model = torch.compile(model, backend="inductor")
                if is_master:
                    print("[INFO] Phase 1 model compiled successfully with torch.compile.", flush=True)
            except Exception as e:
                if is_master:
                    print(f"[!] Phase 1 model compilation warning: {e}", flush=True)

    if IS_TPU:
        xm.broadcast_master_param(model)

    # OPTIMIZER BUG FIX (Finding 97 & Claim 29): Include model.decoder parameters so decoder layers are actively trained!
    trainable_params = list(model.english_decoder.parameters()) + list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.phase1_epochs), eta_min=1e-5
    )

    if IS_TPU:
        bpe = getattr(args, "batches_per_execution", 16)
        loader = pl.MpDeviceLoader(dataloader, device, batches_per_execution=bpe)
    else:
        loader = dataloader

    for epoch in range(args.phase1_epochs):
        model.english_decoder.train()
        model.decoder.train()
        epoch_start_time = time.time()
        last_step_time = time.time()
        total_steps = (
            len(dataset) // max(1, per_core_batch)
            if hasattr(dataset, "__len__")
            else 0
        )

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            is_dae = batch["is_dae"] if "is_dae" in batch else torch.ones(input_ids.shape[0], dtype=torch.bool, device=device)
            mask = (input_ids == 0) | (input_ids == GlossVocabulary.PAD_ID)

            dae_mask_2d = is_dae.unsqueeze(1) if is_dae.dim() == 1 else is_dae
            safe_input_ids_gloss = input_ids.masked_fill(dae_mask_2d, 0)
            safe_input_ids_eng = input_ids.masked_fill(~dae_mask_2d, 0)

            with torch.autocast(
                device_type="cuda" if "cuda" in device.type else "xla",
                dtype=target_dtype,
                enabled=not getattr(args, "disable_amp", False),
            ):
                tgt_in = target_ids[:, :-1]
                tgt_out = target_ids[:, 1:]

                dae_mask_b = (
                    is_dae.squeeze(-1) if is_dae.dim() > 1 else is_dae
                ).unsqueeze(-1).unsqueeze(-1)
                aslg_mask_b = ~dae_mask_b

                # Unified single-graph embedding computation (0 CPU barriers, 0 dynamic graph branches)
                embedded_gloss = model.decoder.token_emb(safe_input_ids_gloss)
                embedded_eng = model.english_decoder.token_emb(safe_input_ids_eng)
                memory = torch.where(aslg_mask_b, embedded_gloss, embedded_eng)

                if hasattr(model, "time_emb"):
                    memory = model.time_emb(memory)

                out = model.english_decoder(
                    tgt_in,
                    memory=memory,
                    memory_key_padding_mask=mask,
                )
                logits = out[0] if isinstance(out, tuple) else out

                valid_mask = (tgt_out != eng_vocab.PAD_ID) & (
                    tgt_out != eng_vocab.EOS_ID
                )
                loss, _ = compute_seq_and_eos_loss(
                    logits,
                    tgt_out,
                    valid_mask,
                    None,
                    label_smoothing=0.1,
                    pad_id=eng_vocab.PAD_ID,
                )

            try:
                dloader_len = len(dataloader)
            except TypeError:
                dloader_len = 2500

            effective_accum = min(
                max(1, args.accum_steps),
                dloader_len
                - (step // max(1, args.accum_steps)) * max(1, args.accum_steps),
            )
            ((loss * args.bwd_weight) / effective_accum).backward()
            if (step + 1) % effective_accum == 0 or (step + 1) == dloader_len:
                if IS_TPU:
                    xm.optimizer_step(optimizer, barrier=False)
                    xm.mark_step()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % args.log_freq == 0:
                with torch.no_grad():
                    preds = logits.argmax(dim=-1)
                    correct = (preds == tgt_out) & valid_mask
                    correct_sum = correct.float().sum()
                    tot_tokens = valid_mask.float().sum()
                    
                    log_vec = torch.stack([
                        loss.detach(),
                        correct_sum.detach(),
                        tot_tokens.detach()
                    ])

                if is_master:
                    now = time.time()
                    elapsed = now - epoch_start_time
                    dt_window = max(1e-5, now - last_step_time)
                    steps_per_sec = (args.log_freq / dt_window) if step > 0 else 0.0
                    world_sz = xr.world_size() if IS_TPU else 1
                    samples_per_sec = (
                        steps_per_sec * per_core_batch * world_sz
                    )
                    lr = optimizer.param_groups[0]["lr"]
                    step_fmt = (
                        f"{step:04d}/{total_steps:04d}"
                        if total_steps > 0
                        else f"{step:04d}"
                    )

                    def _async_step_print(log_vec_cpu, step_idx, step_fmt_str, lr_val, elapsed_val, samples_per_sec_val, steps_per_sec_val, ep, tot_ep):
                        loss_val, c_sum, t_tokens = log_vec_cpu.cpu().tolist()
                        
                        acc_str = ""
                        token_acc = 0.0
                        if t_tokens > 0:
                            token_acc = (c_sum / t_tokens) * 100.0
                            acc_str = f" | Acc: {token_acc:.2f}%"

                        ppl = math.exp(min(loss_val, 20.0))

                        print(
                            f"Phase 1 | Epoch {ep:03d}/{tot_ep:03d} | "
                            f"Step {step_fmt_str} | Loss: {loss_val:.4f}{acc_str} | PPL: {ppl:.1f} | "
                            f"Speed: {samples_per_sec_val:.1f} samp/s ({steps_per_sec_val:.2f} step/s) | "
                            f"LR: {lr_val:.2e} | Elapsed: {elapsed_val:.1f}s",
                            flush=True,
                        )
                        try:
                            metrics_csv_path = "training_metrics.csv"
                            write_header = not os.path.exists(metrics_csv_path)
                            with open(metrics_csv_path, "a", encoding="utf-8") as f:
                                if write_header:
                                    f.write("phase,epoch,step,loss,acc,ppl\n")
                                f.write(f"phase1,{ep},{step_fmt_str},{loss_val:.4f},{token_acc:.2f},{ppl:.1f}\n")
                        except Exception:
                            pass

                    args_tuple = (
                        log_vec,
                        step,
                        step_fmt,
                        lr,
                        elapsed,
                        samples_per_sec,
                        steps_per_sec,
                        epoch + 1,
                        args.phase1_epochs
                    )

                    if IS_TPU:
                        import torch_xla.core.xla_model as xm
                        xm.add_step_closure(_async_step_print, args=args_tuple)
                    else:
                        _async_step_print(*args_tuple)

                    last_step_time = now

            steps_limit = (
                getattr(args, "max_steps", 0)
                if getattr(args, "max_steps", 0) > 0
                else total_steps
            )
            if steps_limit > 0 and step + 1 >= steps_limit:
                break

        scheduler.step()
        if is_master:
            print(f"Phase 1 Epoch {epoch+1}/{args.phase1_epochs} finished.", flush=True)

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.phase1_epochs:
            ckpt_path = os.path.join(args.save_dir, f"asl_llm_{epoch+1}.pt")
            if IS_TPU:
                xm.save(
                    model.state_dict(),
                    ckpt_path,
                    master_only=True,
                )
            elif is_master:
                print(f"Saving model to {ckpt_path}...", flush=True)
                torch.save(
                    model.state_dict(),
                    ckpt_path,
                )
                print("Model saved.", flush=True)

    print("Stopping Phase 1 background thread...", flush=True)
    dataset.stop()
    print("Flushing loader queue...", flush=True)
    if "loader" in locals():
        # Consume any remaining batches to unblock the background thread's queue.put()
        try:
            for _ in loader.per_device_loader(device):
                pass
        except Exception:
            pass
        del loader
    print("Deleting model...", flush=True)
    del model
    print("Deleting optimizer...", flush=True)
    del optimizer
    print("Deleting dataloader...", flush=True)
    del dataloader
    print("Deleting dataset...", flush=True)
    del dataset
    print("Running GC...", flush=True)
    gc.collect()
    print("Phase 1 cleanup complete.", flush=True)


def main():
    """Main CLI entrypoint for the training script."""

    print("[DEBUG 3/8] Executing main() entry point...", flush=True)

    parser = argparse.ArgumentParser(
        description="ASL Foundation Model Multi-Task TPU Training Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/kaggle/input/asl-shards",
    )
    parser.add_argument(
        "--kdwd-dir",
        type=str,
        default="",
        help="Path to KDWD dataset directory",
    )
    parser.add_argument(
        "--aslg-csv",
        type=str,
        default="",
        help="Path to ASLG-PC12 train.csv",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Stop epoch early after this many steps for testing.",
    )
    parser.add_argument(
        "--skip-val-generation",
        action="store_true",
        help="Skip autoregressive generation in validation loop",
    )
    parser.add_argument(
        "--val-check-steps",
        type=int,
        default=0,
        help="Stop validation early after this many steps for testing.",
    )
    parser.add_argument(
        "--phase1-epochs",
        type=int,
        default=0,
        help="Number of text pretraining epochs to run sequentially before Phase 2",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", "--d_model", dest="d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--is-causal", dest="is_causal", action="store_true", help="Enable causal masking")
    
    parser.add_argument("--use-swin", action="store_true", help="Enable Swin-1D shifted window attention")
    parser.add_argument("--swin-window", type=int, default=128, help="Window size for Swin-1D attention")

    parser.add_argument("--num-dataloader-workers", type=int, default=1)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--bwd-weight", type=float, default=1.0)
    parser.add_argument(
        "--enable-aux-decoders",
        action="store_true",
        default=False,
        help="Enable auxiliary Chicago/English decoders for multi-task learning",
    )
    parser.add_argument(
        "--disable-bpe",
        action="store_true",
        help="Disable HuggingFace BPE Tokenizer and fall back to custom EnglishVocabulary",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable PyTorch 2.0 torch.compile JIT acceleration",
    )
    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Force TPU initialization for PyTorch XLA",
    )
    parser.add_argument(
        "--streamed-dataset",
        "--stream-dataset",
        dest="streamed_dataset",
        action="store_true",
        help="Use ASLStreamedDataset (IterableDataset) for zero-RAM startup",
    )
    parser.add_argument(
        "--batches-per-execution",
        type=int,
        default=16,
        help="Number of batches per execution step for MpDeviceLoader on TPU",
    )
    parser.add_argument("--save-dir", type=str, default="/tmp/checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--asl-lex-csv", type=str, default="/home/binhhanh409/signdata.csv"
    )
    args = parser.parse_args()
    print(
        f"[DEBUG 4/8] Command line arguments parsed. Mode: TPU={args.tpu}, Precision={args.precision}, BatchSize={args.batch_size}",
        flush=True,
    )

    # Read datasets directly from /kaggle/input (NVMe SSD) to keep RAM usage near 0GB
    print("[*] Direct NVMe SSD Reading Enabled: Reading datasets directly from /kaggle/input without copying to /dev/shm RAM.", flush=True)

    global IS_TPU
    if args.tpu and _XLA_AVAILABLE:
        os.environ["PJRT_DEVICE"] = "TPU"
        IS_TPU = True

    # Scale learning rate to simulate accumulation step batch sizes if maintaining unrolled mega-graph
    if args.accum_steps > 1:

        print(
            f"[*] Simulating accum_steps={args.accum_steps} with unrolled mega-graph. Scaling LR to {args.lr:.2e}"
        )

    # Removed global environment variables for precision

    if not IS_TPU and args.precision == "bfloat16":
        print(
            "[*] CPU mode detected. Bypassing bfloat16 emulation by falling back to native float32.",
            flush=True,
        )
        args.precision = "float32"

    # Pre-cache KDWD and ASLG tokens to /dev/shm in single-process mode before xmp.spawn
    try:
        eng_vocab = EnglishVocabulary()
        candidate_vocab_dirs = [
            args.data_dir,
            os.path.dirname(args.data_dir),
            "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_dataset",
            "/dev/shm/dataset",
            "/dev/shm",
        ]
        gloss_vocab = None
        for d in candidate_vocab_dirs:
            if d and os.path.exists(d):
                for fname in ["vocab_map.json", "vocabulary_mapping_train.json", "vocabulary_mapping_global.json", "metadata.json"]:
                    vpath = os.path.join(d, fname)
                    if os.path.exists(vpath):
                        try:
                            with open(vpath, "r", encoding="utf-8") as f:
                                raw_map = json.load(f)
                            if isinstance(raw_map, dict) and "label_to_idx" in raw_map:
                                gloss_vocab = GlossVocabulary(label_to_idx=raw_map["label_to_idx"])
                                break
                        except Exception:
                            pass
                if gloss_vocab is not None:
                    break
        if gloss_vocab is None:
            gloss_vocab = GlossVocabulary(label_to_idx={})

        # Pre-cache KDWD and ASLG tokens to local disk in single-process mode before xmp.spawn
        from pathlib import Path
        
        args_kdwd_dir = getattr(args, "kdwd_dir", None)
        args_aslg_csv = getattr(args, "aslg_csv", None)
        
        should_cache_text = False
        if args_kdwd_dir is not None and args_kdwd_dir not in ["", "None", "none"]:
            if Path(args_kdwd_dir).exists():
                should_cache_text = True
                
        if args_aslg_csv is not None and args_aslg_csv not in ["", "None", "none"]:
            if Path(args_aslg_csv).exists():
                should_cache_text = True

        if should_cache_text:
            print("[*] Fast Single-Process Pre-caching: Initializing shared local disk dataset cache...", flush=True)
            if args_kdwd_dir:
                try:
                    KDWDDataset(args_kdwd_dir, eng_vocab, max_len=args.max_len)
                except Exception:
                    pass
            if args_aslg_csv:
                try:
                    ASLGPC12Dataset(args_aslg_csv, eng_vocab, gloss_vocab, max_len=args.max_len)
                except Exception:
                    pass
            print("[*] Fast Single-Process Pre-caching: Shared local disk dataset cache COMPLETE!", flush=True)
    except Exception as e:
        print(f"[*] Pre-caching warning: {e}", flush=True)

    if IS_TPU:
        if "LOCAL_RANK" in os.environ:
            # Launched via torchrun / PyTorch distributed launcher
            import torch.distributed as dist

            dist.init_process_group("xla")
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            _tpu_worker_fn(rank, args)
        else:
            # Direct python execution (e.g. Kaggle TPU notebook cell `python train_all_in_one_tpu.py --tpu`)
            # Spawns 8 processes (one per TPU core) to prevent PJRT barrier deadlock
            import torch_xla.distributed.xla_multiprocessing as xmp

            print(
                "[INFO] Kaggle TPU VM detected. Spawning 8 TPU core worker processes via xmp.spawn...",
                flush=True,
            )
            xmp.spawn(_tpu_worker_fn, args=(args,), nprocs=None, start_method="fork")
    else:
        _tpu_worker_fn(0, args)

    try:
        generate_training_report(
            args, torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    except Exception as e:
        print(f"[!] Note: Automatic report generation skipped: {e}")


def print_console_line_charts(epoch, csv_path="training_metrics.csv"):
    if not os.path.exists(csv_path):
        return

    import pandas as pd

    try:
        df = pd.read_csv(csv_path)
        if "mode" in df.columns:
            df = df.rename(
                columns={"mode": "phase", "loss_total": "loss", "acc_gloss": "acc"}
            )
        elif "phase" not in df.columns:
            df = pd.read_csv(csv_path, names=["epoch", "step", "phase", "loss", "acc"])
    except Exception:
        return

    if df.empty:
        return

    def render_ascii(title, series_dict, height=8, width=60):
        print("\n" + "=" * width)
        print(f" {title.center(width - 2)}")
        print("=" * width)

        all_vals = []
        for name, vals in series_dict.items():
            all_vals.extend(
                [float(v) for v in vals if v is not None and not math.isnan(v)]
            )

        if not all_vals:
            print(" [No metrics data available yet]")
            print("=" * width + "\n")
            return

        v_min, v_max = min(all_vals), max(all_vals)
        if v_min == v_max:
            v_min -= 0.1
            v_max += 0.1

        w = width - 12
        grid = [[" " for _ in range(w)] for _ in range(height)]

        markers = {"train": "*", "val": "o"}

        for s_name, vals in series_dict.items():
            if not vals:
                continue
            marker = markers.get(s_name, "x")
            n = len(vals)
            for col in range(w):
                idx = int(col * (n - 1) / max(1, w - 1))
                val = float(vals[idx])
                row = int((height - 1) * (1.0 - (val - v_min) / (v_max - v_min)))
                row = max(0, min(height - 1, row))
                grid[row][col] = marker if grid[row][col] == " " else "@"

        for r in range(height):
            val_at_row = v_max - r * (v_max - v_min) / max(1, height - 1)
            row_str = "".join(grid[r])
            print(f"{val_at_row:8.2f} |{row_str}")

        print(" " * 9 + "+" + "-" * w)
        print(" " * 9 + "Legend: [*] Train  [o] Validation  [@] Overlap\n")

    train_df = df[df["phase"] == "train"]
    val_df = df[df["phase"].str.startswith("val")]

    # 1. Loss Chart
    if not train_df.empty or not val_df.empty:
        render_ascii(
            f"EPOCH {epoch} LOSS TRAJECTORY (CONSOLE)",
            {"train": train_df["loss"].tolist(), "val": val_df["loss"].tolist()},
        )

    # 2. Accuracy Chart
    if "acc" in df.columns and not df["acc"].isnull().all():
        render_ascii(
            f"EPOCH {epoch} ACCURACY (%) TRAJECTORY (CONSOLE)",
            {"train": train_df["acc"].tolist(), "val": val_df["acc"].tolist()},
        )


def save_epoch_loss_curves_png(
    epoch, csv_path="training_metrics.csv", out_path="loss_curves.png"
):
    if not os.path.exists(csv_path):
        return

    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        df = pd.read_csv(csv_path)
        if "phase" not in df.columns:
            df = pd.read_csv(csv_path, names=["epoch", "step", "phase", "loss", "acc"])
    except Exception:
        return

    if df.empty:
        return

    has_acc = "acc" in df.columns and not df["acc"].isnull().all()
    fig, axes = plt.subplots(1, 2 if has_acc else 1, figsize=(14 if has_acc else 8, 5))
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    train_df = df[df["phase"] == "train"]
    val_df = df[df["phase"].str.startswith("val")]

    # Plot 1: Loss
    ax1 = axes[0]
    if not train_df.empty:
        ax1.plot(
            range(len(train_df)),
            train_df["loss"],
            label="Train Loss",
            alpha=0.35,
            color="dodgerblue",
        )
        ax1.plot(
            range(len(train_df)),
            train_df["loss"].rolling(10, min_periods=1).mean(),
            label="Train Loss (Smoothed)",
            color="blue",
            linewidth=2,
        )
    if not val_df.empty:
        ax1.plot(
            range(len(val_df)),
            val_df["loss"],
            "o-",
            label="Val Loss",
            color="orange",
            linewidth=1.5,
        )

    ax1.set_title(f"Loss Trajectory (Epoch {epoch})", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Step / Evaluation Index", fontsize=10)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Accuracy (if present)
    if has_acc and len(axes) > 1:
        ax2 = axes[1]
        if not train_df.empty and "acc" in train_df.columns:
            ax2.plot(
                range(len(train_df)),
                train_df["acc"],
                label="Train Acc %",
                alpha=0.35,
                color="mediumseagreen",
            )
            ax2.plot(
                range(len(train_df)),
                train_df["acc"].rolling(10, min_periods=1).mean(),
                label="Train Acc (Smoothed)",
                color="green",
                linewidth=2,
            )
        if not val_df.empty and "acc" in val_df.columns:
            ax2.plot(
                range(len(val_df)),
                val_df["acc"],
                "s-",
                label="Val Acc %",
                color="purple",
                linewidth=1.5,
            )

        ax2.set_title(
            f"Accuracy Trajectory (Epoch {epoch})", fontsize=12, fontweight="bold"
        )
        ax2.set_xlabel("Step / Evaluation Index", fontsize=10)
        ax2.set_ylabel("Accuracy (%)", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(
        f"[+] Saved updated epoch {epoch} loss/accuracy image plot to: {out_path}",
        flush=True,
    )

    try:
        from IPython.display import display, Image

        display(Image(filename=out_path))
    except Exception:
        pass


def generate_training_report(args, device):
    """
    Generates training reports and visualizations, specifically drawing loss/accuracy plots
    from the training metrics CSV and validating exact match scores.

    Args:
        args (argparse.Namespace): Arguments including data_dir.
        device (torch.device): Evaluation device.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import collections

    print("\n[INFO] Generating Training Report and Visualizations...")

    # 1. Plot Metrics
    if os.path.exists("training_metrics.csv"):
        df = pd.read_csv("training_metrics.csv")
        if "mode" in df.columns:
            df = df.rename(
                columns={"mode": "phase", "loss_total": "loss", "acc_gloss": "acc"}
            )
        elif "phase" not in df.columns:
            df = pd.read_csv(
                "training_metrics.csv", names=["epoch", "step", "phase", "loss", "acc"]
            )
        plt.figure(figsize=(10, 6))

        train_df = df[df["phase"] == "train"]
        val_intra = df[df["phase"] == "val_intra"]

        if not train_df.empty:
            plt.plot(train_df.index, train_df["loss"], label="Train Loss", alpha=0.3)
            # Smooth train loss
            plt.plot(
                train_df.index,
                train_df["loss"].rolling(50, min_periods=1).mean(),
                color="blue",
                label="Train Loss (Smoothed)",
            )
        if not val_intra.empty:
            plt.plot(
                val_intra.index,
                val_intra["loss"],
                "x",
                label="Intra-Epoch Val",
                color="orange",
            )

        plt.title("Training and Validation Loss")
        plt.xlabel("Step Index")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curves.png")
        print("Saved loss curves to loss_curves.png")

    # 2. Word Distribution (Non-How2Sign)
    print("[INFO] Calculating word distribution for non-How2Sign validation subset...")
    eng_vocab = EnglishVocabulary(
        vocab_path=os.path.join(args.data_dir, "english_vocab.json")
    )
    val_loader = create_dataloader(
        dataset_dir=args.data_dir,
        split="val",
        batch_size=getattr(args, "batch_size", 8),
        max_len=getattr(args, "max_len", 256),
        num_dataloader_workers=0,
        shuffle=False,
    )

    actual_vocab_size = 200
    if hasattr(val_loader.dataset, "label_to_idx") and val_loader.dataset.label_to_idx:
        actual_vocab_size = len(GlossVocabulary(label_to_idx=val_loader.dataset.label_to_idx))

    model = ASLFoundationModel(
        vocab_size=actual_vocab_size,
        d_enc=args.d_model,
        d_dec=args.d_model,
        nhead_enc=args.nhead,
        nhead_dec=args.nhead,
        num_enc_layers=args.num_layers,
        num_dec_layers=args.num_layers,
        dropout=args.dropout,
        english_vocab_size=len(eng_vocab),
        max_enc_len=getattr(args, "max_enc_len", 512),
        max_dec_len=getattr(args, "max_dec_len", 256),
        use_mamba=getattr(args, "use_mamba", True),
        tome_r=getattr(args, "tome_r", 0),
        scale_embeddings=getattr(args, "scale_embeddings", True),
        use_swin_1d=getattr(args, "use_swin_1d", False),
        swin_window_size=getattr(args, "swin_window_size", 128),
        is_causal=getattr(args, "is_causal", False),
        enable_aux_decoders=getattr(args, "enable_aux_decoders", True),
    ).to(device)

    save_dir = getattr(args, "save_dir", ".")
    ckpt_path = os.path.join(save_dir, "asl_model_latest.pt")
    
    if os.path.exists(ckpt_path):
        loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = loaded["model_state_dict"] if isinstance(loaded, dict) and "model_state_dict" in loaded else (loaded["model"] if isinstance(loaded, dict) and "model" in loaded else loaded)
        model.load_state_dict(state_dict, strict=getattr(args, "strict_load", True))
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}. Using current model weights.")

    model.eval()

    word_counts = collections.Counter()
    max_eval_batches = getattr(args, "val_batches", 50)
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_eval_batches:
                break

            valid_mask = batch.get("valid_mask", torch.ones(batch["feature"].shape[0], dtype=torch.bool))
            if not valid_mask.any():
                continue

            features = batch["feature"][valid_mask].to(device)
            mask = batch["mask"][valid_mask].to(device)
            eng_seq = batch.get("english_seq", batch.get("english", None))
            if eng_seq is None:
                continue
            eng_seq = eng_seq[valid_mask].to(device)
            domain = batch.get("domain_label", batch.get("source_id", torch.zeros_like(eng_seq[:, 0])))
            if isinstance(domain, torch.Tensor):
                domain = domain[valid_mask].to(device)

            # Forward pass
            out = model(
                input_x=features,
                mask=mask,
                english_seq=eng_seq,
            )

            logits = out.get("english_logits")
            if logits is not None:
                preds = logits.argmax(dim=-1)  # (B, S)

                # Filter non-How2Sign (domain_label != 2)
                non_how2sign_mask = domain != 2

                if non_how2sign_mask.any():
                    valid_preds = preds[non_how2sign_mask]
                    valid_gt = eng_seq[non_how2sign_mask]

                    for b_idx in range(valid_preds.shape[0]):
                        for s_idx in range(valid_preds.shape[1]):
                            if s_idx == 0:
                                continue  # Skip BOS
                            
                            # GT target for prediction at s_idx is at s_idx + 1 (Teacher Forcing alignment)
                            if s_idx + 1 >= valid_gt.shape[1]:
                                break
                            
                            gt_token = valid_gt[b_idx, s_idx + 1].item()
                            pred_token = valid_preds[b_idx, s_idx].item()

                            if gt_token in [0, 1, 2]:  # PAD, BOS, EOS
                                continue

                            # Count ALL predicted tokens to find model bias/distribution
                            word_counts[pred_token] += 1

    print("\n=======================================================")
    print("   TOP 50 PREDICTED WORDS (NON-HOW2SIGN)   ")
    print("=======================================================\n")
    print(f"{'Rank':<5} | {'Word':<20} | {'Count':<10}")
    print("-" * 40)
    for rank, (token_id, count) in enumerate(word_counts.most_common(50), 1):
        word = eng_vocab.decode([token_id])
        print(f"{rank:<5} | {word:<20} | {count:<10}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
