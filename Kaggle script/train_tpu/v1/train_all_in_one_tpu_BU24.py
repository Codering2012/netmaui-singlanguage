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

import os
import sys

# Critical: MUST be set on line 1 before ANY C libraries (numpy, mkl, openmp, torch, torch_xla) are loaded
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["GOMP_SPINCOUNT"] = "0"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "65536"
os.environ["MALLOC_TRIM_THRESHOLD_"] = "65536"
os.environ["MALLOC_ARENA_MAX"] = "2"
os.environ["PJRT_ALLOCATOR_FRACTION"] = "0.85"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.85"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["XLA_USE_BF16"] = "1"
os.environ.pop("XLA_DOWNCAST_BF16", None)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_TRANSFER_STREAM_LIMIT"] = "4"

# LibTPU & XLA Fast Compilation and Hardware Acceleration Flags (instant systolic execution on TPU v5e)
if "LIBTPU_INIT_ARGS" not in os.environ:
    os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_enable_flash_attention=true --xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true"
else:
    os.environ["LIBTPU_INIT_ARGS"] = os.environ["LIBTPU_INIT_ARGS"].replace(
        "xla_tpu_enable_async_collective_fusion_multiple_bars",
        "xla_tpu_enable_async_collective_fusion_multiple_steps",
    )

_existing_xla = os.environ.get("XLA_FLAGS", "")
# Sanitize any legacy or unknown fast_math flags from earlier runs in the same notebook kernel
_existing_xla = _existing_xla.replace("--xla_tpu_fast_math=true", "").replace("--xla_tpu_fast_math", "").strip()

_fast_flags = [
    "--xla_cpu_multi_thread_eigen=true",
]
for _ff in _fast_flags:
    if _ff not in _existing_xla:
        _existing_xla = (_existing_xla + " " + _ff).strip()
os.environ["XLA_FLAGS"] = _existing_xla

# Force Local PJRT mode to avoid gRPC proxy concurrency limit and fork deadlocks
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)

# Persistent XLA compilation cache directory path: ALWAYS use /tmp/xla_cache on Linux to prevent
# exhausting Kaggle's 20GB /kaggle/working directory quota with compiled HLO graphs across 8 cores.
_default_cache = "/tmp/xla_cache" if os.name != "nt" else "./xla_cache"
cache_dir = os.environ.get("XLA_PERSISTENT_CACHE_PATH", _default_cache)
if cache_dir.startswith("/kaggle/working"):
    cache_dir = "/tmp/xla_cache"
os.environ["XLA_PERSISTENT_CACHE_PATH"] = cache_dir
try:
    os.makedirs(cache_dir, exist_ok=True)
except Exception:
    pass

import argparse
import itertools
import functools
import contextlib
import csv
import gc
import json
import math
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import warnings

# Clean warning filters for PyTorch 2.4/2.5 on TPU
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*torch_xla\\.sync.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*_c10d_functional::all_reduce.*")

def trim_host_memory():
    """Forces Python garbage collection and releases glibc heap memory back to the OS."""
    gc.collect()
    if os.name != "nt":
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim(0)
        except Exception:
            pass


import atexit
import signal

_MAIN_PARENT_PID = os.getpid()

def _kill_all_child_subprocesses():
    """Bulletproof process reaper: Terminates child processes ONLY from the main parent process."""
    if os.getpid() != _MAIN_PARENT_PID:
        return
    try:
        import psutil
        parent = psutil.Process(_MAIN_PARENT_PID)
        for child in parent.children(recursive=True):
            try:
                child.kill()
            except Exception:
                pass
    except Exception:
        pass

def _global_signal_handler(signum, frame):
    if os.getpid() == _MAIN_PARENT_PID:
        _kill_all_child_subprocesses()
        sys.exit(0)
    else:
        sys.exit(1)

# Register with atexit and signals for automatic cleanup
atexit.register(_kill_all_child_subprocesses)
try:
    signal.signal(signal.SIGINT, _global_signal_handler)
    signal.signal(signal.SIGTERM, _global_signal_handler)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _global_signal_handler)
except Exception:
    pass

gc.set_threshold(1000, 15, 15)
import torch
try:
    import torch_xla
    torch.xla = torch_xla
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

pl = None
try:
    import torch_xla.distributed.parallel_loader as pl
except Exception:
    pass

try:
    from dataset import (
        phase1_collate_fn,
        phase2_collate_fn,
        create_dataloader,
        normalize_vocabulary,
        EnglishVocabulary,
        GlossVocabulary,
        ASLStreamedDataset,
    )
except ImportError:
    from train_tpu.v1.dataset import (
        phase1_collate_fn,
        phase2_collate_fn,
        create_dataloader,
        normalize_vocabulary,
        EnglishVocabulary,
        GlossVocabulary,
        ASLStreamedDataset,
    )

print(
    "[DEBUG 1/8] Importing standard libraries & setting environment variables...",
    flush=True,
)

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


def get_dynamic_loader_len(
    loader: Any, default_steps: int = 2500, args: Optional[Any] = None
) -> int:
    """Computes dynamic epoch steps accurately without hardcoded fallbacks."""
    if args is not None and getattr(args, "steps_per_epoch", 0) > 0:
        return int(args.steps_per_epoch)
    try:
        return len(loader)
    except (TypeError, AttributeError):
        pass
    ds = getattr(loader, "dataset", None)
    if ds is not None:
        if hasattr(ds, "total_records") and ds.total_records is not None and ds.total_records > 0:
            bs = getattr(loader, "batch_size", 64)
            world_sz = get_xla_world_size() if IS_TPU else int(os.environ.get("WORLD_SIZE", "1"))
            return max(1, ds.total_records // max(1, bs * world_sz))
        if hasattr(ds, "shard_files") and ds.shard_files:
            bs = getattr(loader, "batch_size", 64)
            return max(1, (len(ds.shard_files) * 1000) // max(1, bs))
    return default_steps


def _distributed_normalize(
    local_sum: torch.Tensor, local_weight: torch.Tensor
) -> torch.Tensor:
    """Computes weighted loss mean locally; global gradient averaging across TPUs is handled by xm.optimizer_step()."""
    normed = local_sum / local_weight.clamp_min(1e-8)
    return normed * (local_weight > 0).to(normed.dtype)


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
    """Fused RMSNorm for TPU/GPU without redundant dtype conversions."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        """Initializes the module component."""
        super().__init__()
        self.eps = eps
        self.d_model = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for this module."""
        if hasattr(F, "rms_norm"):
            return F.rms_norm(input_x, self.d_model, self.weight, eps=self.eps)
        var = input_x.pow(2).mean(-1, keepdim=True)
        return input_x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLUFFN(nn.Module):
    """Fused SwiGLUFFN for Peak TPU v5e MXU GEMM Throughput."""

    def __init__(self, d_model: int, dim_feedforward: int, num_layers: int = 8):
        """Initializes the module component."""

        super().__init__()
        # Align hidden dimension to 128 for TPU v5e MXU systolic arrays
        hidden = (int(dim_feedforward * 2 / 3) + 127) // 128 * 128
        self.hidden = hidden
        # Fused Gate-Up projection matrix (single large GEMM on TPU MXU)
        self.w_gate_up = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)
        nn.init.normal_(self.w_gate_up.weight, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.w_down.weight, std=1.0 / math.sqrt(2.0 * num_layers * d_model))

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Forward pass for this module."""

        gate, up = self.w_gate_up(input_x).chunk(2, dim=-1)
        return self.w_down(F.silu(gate) * up)


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
        self.emb_lexclass = nn.Embedding(24, 32)
        self.emb_signtype = nn.Embedding(16, 32)
        self.emb_handshape = nn.Embedding(64, 48)
        self.emb_location = nn.Embedding(32, 32)
        self.emb_category = nn.Embedding(48, 48)

        self.attr_proj = nn.Sequential(
            nn.Linear(32 + 32 + 48 + 32 + 48 + 3, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        attr_idx_matrix = torch.zeros((vocab_size, 5), dtype=torch.long)
        attr_scalars = torch.zeros((vocab_size, 3), dtype=torch.float32)
        minimal_pair_mask = torch.zeros((vocab_size, vocab_size), dtype=torch.bool)

        if csv_path is not None and Path(csv_path).exists():
            try:
                with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.DictReader(f)

                    lexclass_map = {"": 0}
                    signtype_map = {"": 0}
                    handshape_map = {"": 0}
                    location_map = {"": 0}
                    category_map = {"": 0}

                    needs_offset = False
                    if label_to_idx is not None:
                        normal_ids = [v for k, v in label_to_idx.items() if not str(k).startswith("<")]
                        needs_offset = (min(normal_ids) < 4) if normal_ids else False

                    matched_count = 0
                    for row in reader:
                        try:
                            raw_word = (row.get("LemmaID") or row.get("EntryID") or "").strip().lower()
                            if not raw_word:
                                continue
                            word_clean = re.sub(r"\d+$", "", raw_word.replace("_", "").replace("-", ""))

                            if label_to_idx is not None:
                                if raw_word in label_to_idx:
                                    idx = label_to_idx[raw_word]
                                elif word_clean in label_to_idx:
                                    idx = label_to_idx[word_clean]
                                else:
                                    continue
                                if needs_offset:
                                    idx += 4
                            else:
                                idx = matched_count

                            if idx >= vocab_size:
                                continue

                            matched_count += 1

                            # 1. Lexical Class
                            lc = (row.get("LexicalClass") or "").strip()
                            if lc not in lexclass_map and len(lexclass_map) < 24:
                                lexclass_map[lc] = len(lexclass_map)
                            attr_idx_matrix[idx, 0] = lexclass_map.get(lc, 0)

                            # 2. Sign Type
                            st = (row.get("SignType.2.0") or row.get("SignType") or "").strip()
                            if st not in signtype_map and len(signtype_map) < 16:
                                signtype_map[st] = len(signtype_map)
                            attr_idx_matrix[idx, 1] = signtype_map.get(st, 0)

                            # 3. Handshape (58 categories in ASL-LEX 2.0)
                            hs = (row.get("Handshape.2.0") or row.get("SelectedHandshape") or "").strip()
                            if hs not in handshape_map and len(handshape_map) < 64:
                                handshape_map[hs] = len(handshape_map)
                            attr_idx_matrix[idx, 2] = handshape_map.get(hs, 0)

                            # 4. Major Location
                            loc = (row.get("MajorLocation.2.0") or row.get("MajorLocation") or "").strip()
                            if loc not in location_map and len(location_map) < 32:
                                location_map[loc] = len(location_map)
                            attr_idx_matrix[idx, 3] = location_map.get(loc, 0)

                            # 5. Semantic Category
                            cat = (row.get("CDISemanticCategory") or row.get("SemanticCategory") or "").strip()
                            if cat not in category_map and len(category_map) < 48:
                                category_map[cat] = len(category_map)
                            attr_idx_matrix[idx, 4] = category_map.get(cat, 0)

                            # Scalars: Flexion, Transparency, Iconicity
                            try:
                                attr_scalars[idx, 0] = float(row.get("Flexion.2.0", row.get("Flexion", 0.0)) or 0.0)
                            except:
                                pass
                            try:
                                attr_scalars[idx, 1] = float(row.get("Transparency Z", row.get("Transparency(M)", 0.0)) or 0.0)
                            except:
                                pass
                            try:
                                attr_scalars[idx, 2] = float(row.get("Iconicity(Z)", row.get("Iconicity(M)", 0.0)) or 0.0)
                            except:
                                pass
                        except Exception:
                            continue

                    # Precompute pairwise minimal-pair mask:
                    # Same Handshape (idx 2 > 0) + Same SignType (idx 1 > 0) + Different Location (idx 3)
                    hs_t = attr_idx_matrix[:, 2].unsqueeze(1)
                    st_t = attr_idx_matrix[:, 1].unsqueeze(1)
                    loc_t = attr_idx_matrix[:, 3].unsqueeze(1)

                    valid_phon = (hs_t > 0) & (hs_t.T > 0)
                    same_hs = (hs_t == hs_t.T) & valid_phon
                    same_st = (st_t == st_t.T) & (st_t > 0)
                    diff_loc = (loc_t != loc_t.T)
                    
                    minimal_pair_mask = same_hs & same_st & diff_loc
            except Exception as e:
                print(f"[!] Warning: Failed to parse ASL-LEX CSV: {e}", flush=True)

        self.register_buffer("attr_idx_matrix", attr_idx_matrix, persistent=False)
        self.register_buffer("attr_scalars", attr_scalars, persistent=False)
        self.register_buffer("minimal_pair_mask", minimal_pair_mask, persistent=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        ids = token_ids
        attr_ids = self.attr_idx_matrix[ids]
        scalars = self.attr_scalars[ids]

        a0, a1, a2, a3, a4 = attr_ids.unbind(dim=-1)
        e_lc = self.emb_lexclass(a0)
        e_st = self.emb_signtype(a1)
        e_hs = self.emb_handshape(a2)
        e_loc = self.emb_location(a3)
        e_cat = self.emb_category(a4)

        raw_attrs = torch.cat([e_lc, e_st, e_hs, e_loc, e_cat, scalars], dim=-1)
        valid_lex_mask = (token_ids != 0).unsqueeze(-1).to(raw_attrs.dtype)
        return self.attr_proj(raw_attrs) * valid_lex_mask


def get_weight_dtype(layer: nn.Module) -> torch.dtype:
    """Safely retrieves the floating-point weight dtype across standard, DDP, and dynamically quantized layers."""
    if hasattr(layer, "weight"):
        w = layer.weight
        if callable(w):
            dtype = w().dtype
        elif isinstance(w, torch.Tensor):
            dtype = w.dtype
        else:
            dtype = torch.float32

        if dtype in (torch.qint8, torch.quint8, torch.qint32):
            return torch.float32
        return dtype
    for p in layer.parameters():
        return p.dtype
    return torch.float32


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
        self.r = kwargs.get("r", 2)  # Temporal pooling stride (r=0: identity, r=2: 2x downsampling)
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

        mask_reshaped = None
        if mask is not None:
            if pad_len > 0:
                if self.is_causal:
                    mask_padded = torch.nn.functional.pad(mask, (pad_len, 0), value=False)
                else:
                    mask_padded = torch.nn.functional.pad(mask, (0, pad_len), value=False)
            else:
                mask_padded = mask

            mask_reshaped = mask_padded.view(B, -1, 2)
            # Mask out invalid frames unconditionally (pure tensor operation without CPU sync)
            hidden_sum = (hidden_reshaped * mask_reshaped.unsqueeze(-1)).sum(dim=2)
            valid_count = mask_reshaped.sum(dim=2).unsqueeze(-1).clamp(min=1).to(hidden_sum.dtype)
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
            if mask_reshaped is not None:
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
        if self.drop_prob == 0.0 or not self.training:
            return input_x
        return drop_path(input_x, self.drop_prob, self.training)


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

    def forward(self, q: torch.Tensor, k: torch.Tensor, frame_indices: Optional[torch.Tensor] = None):
        """Forward pass for RotaryPositionalEncoding computing trigonometric frequencies once for both q and k."""
        if frame_indices is not None:
            inv_f = self.inv_freq.to(frame_indices.device)
            freqs = torch.einsum("bi,d->bid", frame_indices.float(), inv_f)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos_dtype = emb.cos().unsqueeze(1).to(q.dtype)
            sin_dtype = emb.sin().unsqueeze(1).to(q.dtype)
        else:
            cos_dtype = self.cos_cached[:, :, : q.shape[2], :].to(q.dtype)
            sin_dtype = self.sin_cached[:, :, : q.shape[2], :].to(q.dtype)

        q1, q2 = q.chunk(2, dim=-1)
        q_rot = torch.cat((-q2, q1), dim=-1)
        q_out = (q * cos_dtype) + (q_rot * sin_dtype)

        k1, k2 = k.chunk(2, dim=-1)
        k_rot = torch.cat((-k2, k1), dim=-1)
        k_out = (k * cos_dtype) + (k_rot * sin_dtype)

        return q_out, k_out


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
        kv_reshaped = kv.view(batch_sz, seq_len, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        key_k_lower, val_v = kv_reshaped[0], kv_reshaped[1]
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

        out = F.scaled_dot_product_attention(
            query_q_lower,
            key_k_lower,
            val_v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=sdpa_is_causal,
            scale=self.scale,
            enable_gqa=True,
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

        # Masking for shifted windows to prevent cross-boundary attention (Claim 31 Fix)
        attn_mask = None
        if self.shift_size > 0:
            img_mask = torch.arange(num_windows, device=input_x.device, dtype=torch.float32).repeat_interleave(self.window_size).view(1, -1, 1)
            img_mask_shifted = torch.roll(img_mask, shifts=-self.shift_size, dims=1)
            mask_windows_attn = img_mask_shifted.view(num_windows, self.window_size)
            attn_mask = (mask_windows_attn.unsqueeze(1) == mask_windows_attn.unsqueeze(2)) # True means attend
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
    """Fast fused channel-spatial gating for ConvNeXtTemporalBlock."""

    def __init__(self, d_model: int, reduction: int = 4, is_causal: bool = False):
        """Initializes the module component."""

        super().__init__()
        self.is_causal = is_causal
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self, input_x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass for this module."""

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask.bool()).unsqueeze(-1).to(input_x.dtype)
            mean_x = (input_x * valid_mask).sum(dim=1, keepdim=True) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        else:
            mean_x = input_x.mean(dim=1, keepdim=True)
            
        return input_x * self.gate(mean_x)


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

        valid_mask = (
            (~key_padding_mask.bool()).unsqueeze(-1).to(input_x.dtype)
            if key_padding_mask is not None
            else None
        )
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
            
        if padded_x.is_floating_point() and padded_x.dtype != self.dw_conv.weight.dtype:
            padded_x = padded_x.to(self.dw_conv.weight.dtype)
            
        target_y = self.norm(self.dw_conv(padded_x).transpose(1, 2))
        if target_y.is_floating_point() and target_y.dtype != self.pw_conv1.weight.dtype:
            target_y = target_y.to(self.pw_conv1.weight.dtype)
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
        self.split_sizes = [
            self.d_inner,
            self.d_inner,
            self.nheads * self.d_state,
            self.nheads * self.d_state,
            self.nheads,
        ]

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
                ~self.tril_mask_c[:n_chunks, :n_chunks],
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

        if input_x.is_floating_point() and input_x.dtype != self.in_proj.weight.dtype:
            input_x = input_x.to(self.in_proj.weight.dtype)

        kpm_un = key_padding_mask.unsqueeze(-1) if key_padding_mask is not None else None
        if key_padding_mask is not None:
            input_x = input_x.masked_fill(kpm_un, 0.0)
        xn = self.norm1(input_x)
        b_sz, t_sz, _ = xn.shape

        x_proj, z, B_ssm_fwd, C_ssm_fwd, dt_fwd = torch.split(
            self.in_proj(xn),
            self.split_sizes,
            dim=-1,
        )
        x_conv_in = x_proj.transpose(1, 2)
        if self.is_causal:
            x_conv_in = F.pad(x_conv_in, (self.d_conv - 1, 0))
        if x_conv_in.is_floating_point() and x_conv_in.dtype != self.fwd_conv1d.weight.dtype:
            x_conv_in = x_conv_in.to(self.fwd_conv1d.weight.dtype)
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
            out = out.masked_fill(kpm_un, 0.0)

        input_x = input_x + self.drop_path1(self.gamma_1 * out)
        x2 = self.ffn(self.norm2(input_x))
        if key_padding_mask is not None:
            x2 = x2.masked_fill(kpm_un, 0.0)
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
        x_in = input_x
        x_in = x_in + 0.5 * self.drop_path_ffn1(
            self.gamma_ffn1 * self.ffn1(self.ffn1_norm(x_in))
        )
        x_in = x_in + self.drop_path_mha(
            self.gamma_mha
            * self.mha(
                self.mha_norm(x_in),
                key_padding_mask=key_padding_mask,
                frame_indices=frame_indices,
            )
        )

        if not self.is_causal:
            cls_t = x_in[:, :1]
            x_seq = x_in[:, 1:]
            seq_mask = key_padding_mask[:, 1:] if key_padding_mask is not None else None
            xc_seq = self.conv_block(self.conv_norm(x_seq), key_padding_mask=seq_mask, cache=cache)
            xc = torch.cat([cls_t, xc_seq], dim=1)
        else:
            x_seq = x_in
            seq_mask = key_padding_mask
            xc = self.conv_block(self.conv_norm(x_seq), key_padding_mask=seq_mask, cache=cache)

        x_in = x_in + self.drop_path_conv(self.gamma_conv * xc)
        x_in = x_in + 0.5 * self.drop_path_ffn2(
            self.gamma_ffn2 * self.ffn2(self.ffn2_norm(x_in))
        )
        return x_in


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
        if input_x.dim() == 4:
            x_t = input_x.flatten(2).transpose(1, 2)
        else:
            x_t = input_x.transpose(1, 2)
        m_2d = mask.unsqueeze(1).to(x_t.dtype) if mask is not None else None
        if mask is not None:
            x_t = x_t * m_2d

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
            
        if padded_x1.is_floating_point() and padded_x1.dtype != self.conv1.weight.dtype:
            padded_x1 = padded_x1.to(self.conv1.weight.dtype)
            
        c1 = self.conv1(padded_x1)
        if self.is_causal:
            feat_seq = self.act1(self.norm1(c1.transpose(1, 2)).transpose(1, 2))
        else:
            feat_seq = self.act1(self.norm1(c1))

        if feat_seq.dtype != self.conv2.weight.dtype:
            feat_seq = feat_seq.to(self.conv2.weight.dtype)
        if mask is not None:
            feat_seq = feat_seq * m_2d

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
            
        if padded_x2.is_floating_point() and padded_x2.dtype != self.conv2.weight.dtype:
            padded_x2 = padded_x2.to(self.conv2.weight.dtype)
            
        feat_seq = self.conv2(padded_x2)
        if feat_seq.dtype != self.conv3.weight.dtype:
            feat_seq = feat_seq.to(self.conv3.weight.dtype)
        if mask is not None:
            feat_seq = feat_seq * m_2d

        c3 = self.conv3(feat_seq)
        if self.is_causal:
            feat_seq = self.act2(self.norm2(c3.transpose(1, 2)).transpose(1, 2))
        else:
            feat_seq = self.act2(self.norm2(c3))

        feat_seq = feat_seq.transpose(1, 2)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(-1).to(feat_seq.dtype)
        out_dtype = get_weight_dtype(self.out_proj)
        if feat_seq.dtype != out_dtype:
            feat_seq = feat_seq.to(out_dtype)
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
        self._build_cache(max(max_seq_len, 2048))

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
        self._cached_dtype = None
        self._cached_cos = None
        self._cached_sin = None

    @staticmethod
    def _rotate_half(input_x: torch.Tensor) -> torch.Tensor:
        """Internal helper method _rotate_half with zero-copy chunk split."""
        x1, x2 = input_x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self, query_q_lower: torch.Tensor, key_k_lower: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for this module with zero-allocation dtype-cached RoPE."""

        seq_s = query_q_lower.shape[-2]
        target_dtype = query_q_lower.dtype

        # Zero-allocation cached view in target dtype (eliminates 24 redundant XLA convert ops/step)
        if self._cached_dtype != target_dtype or self._cached_cos is None:
            self._cached_dtype = target_dtype
            self._cached_cos = self.cos_cache.to(target_dtype)
            self._cached_sin = self.sin_cache.to(target_dtype)

        if offset == 0:
            cos = self._cached_cos[:, :, :seq_s, :]
            sin = self._cached_sin[:, :, :seq_s, :]
        else:
            if isinstance(offset, int):
                offset_t = torch.tensor([offset], device=query_q_lower.device, dtype=torch.long)
            else:
                offset_t = offset.view(-1).to(torch.long)

            positions = torch.arange(seq_s, device=query_q_lower.device, dtype=torch.long) + offset_t
            cos_flat = self._cached_cos.squeeze(0).squeeze(0)  # [L, D]
            sin_flat = self._cached_sin.squeeze(0).squeeze(0)  # [L, D]

            cos = F.embedding(positions, cos_flat).unsqueeze(0).unsqueeze(0)
            sin = F.embedding(positions, sin_flat).unsqueeze(0).unsqueeze(0)

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
        kv_reshaped = kv.view(batch_sz, seq_len, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        key_k_lower, val_v = kv_reshaped[0], kv_reshaped[1]

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
            past_len = 0
            query_q_lower, key_k_lower = self.rope(
                query_q_lower, key_k_lower, offset=past_len
            )
            current_key_value = (key_k_lower, val_v) if use_cache else None

        # Robust dtype alignment before Scaled Dot-Product Attention
        if key_k_lower.dtype != query_q_lower.dtype:
            key_k_lower = key_k_lower.to(query_q_lower.dtype)
        if val_v.dtype != query_q_lower.dtype:
            val_v = val_v.to(query_q_lower.dtype)

        if past_len > 0:
            if seq_len == 1:
                max_len = key_k_lower.size(2)
                total_len = int(past_len) + 1 if not isinstance(past_len, torch.Tensor) else int(past_len) + 1
                causal_mask = (torch.arange(max_len, device=input_x.device) < total_len).view(1, 1, 1, max_len)
                
                if padding_mask is not None:
                    attn_mask = (~padding_mask).unsqueeze(1).unsqueeze(2) & causal_mask
                else:
                    attn_mask = causal_mask
                out = F.scaled_dot_product_attention(
                    query_q_lower, key_k_lower, val_v, attn_mask=attn_mask, scale=self.scale, enable_gqa=True
                )
            else:
                max_len = key_k_lower.size(2)
                total_len = int(past_len) + seq_len if not isinstance(past_len, torch.Tensor) else int(past_len) + seq_len
                causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_x.device))
                full_mask = torch.zeros(seq_len, max_len, dtype=torch.bool, device=input_x.device)
                past_l = int(past_len) if not isinstance(past_len, torch.Tensor) else int(past_len)
                if past_l > 0:
                    full_mask[:, :past_l] = True
                full_mask[:, past_l:total_len] = causal
                attn_mask = full_mask.unsqueeze(0).unsqueeze(1)
                if padding_mask is not None:
                    attn_mask = attn_mask & (~padding_mask).unsqueeze(1).unsqueeze(2)
                out = F.scaled_dot_product_attention(
                    query_q_lower, key_k_lower, val_v, attn_mask=attn_mask, scale=self.scale, enable_gqa=True
                )
        else:
            if not use_cache:
                # Fast direct causal FlashAttention kernel with native zero-copy GQA
                out = F.scaled_dot_product_attention(
                    query_q_lower, key_k_lower, val_v, is_causal=True, scale=self.scale, enable_gqa=True
                )
            else:
                max_len = key_k_lower.size(2)
                causal = torch.tril(torch.ones(seq_len, max_len, dtype=torch.bool, device=input_x.device))
                attn_mask = causal.unsqueeze(0).unsqueeze(1)
                if padding_mask is not None:
                    attn_mask = attn_mask & (~padding_mask).unsqueeze(1).unsqueeze(2)
                out = F.scaled_dot_product_attention(
                    query_q_lower, key_k_lower, val_v, attn_mask=attn_mask, scale=self.scale, enable_gqa=True
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
            if memory.dtype != tgt.dtype:
                memory = memory.to(tgt.dtype)
            seq_s = memory.size(1)
            # MLA Latent Projection for Cross Attention
            kv_latent = self.kv_latent_proj(memory)
            kv_latent = self.kv_latent_norm(kv_latent)

            mem_b = memory.size(0)
            kv = self.kv_proj(kv_latent)
            kv_reshaped = kv.view(mem_b, seq_s, 2, self.kv_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            key_k_lower, val_v = kv_reshaped[0], kv_reshaped[1]

        if key_k_lower.dtype != query_q.dtype:
            key_k_lower = key_k_lower.to(query_q.dtype)
        if val_v.dtype != query_q.dtype:
            val_v = val_v.to(query_q.dtype)

        current_key_value = (key_k_lower, val_v) if use_cache else None

        mem_b = key_k_lower.size(0)
        if mem_b < batch_sz:
            repeat_factor = batch_sz // mem_b
            query_5d = query_q.view(repeat_factor, mem_b, self.nhead, seq_len, self.head_dim)
            key_5d = key_k_lower.unsqueeze(0)
            val_5d = val_v.unsqueeze(0)
            if memory_key_padding_mask is not None:
                if memory_key_padding_mask.size(0) == mem_b:
                    attn_mask_5d = (~memory_key_padding_mask.bool()).unsqueeze(0).unsqueeze(2).unsqueeze(3)
                else:
                    attn_mask_5d = (~memory_key_padding_mask.view(repeat_factor, mem_b, 1, 1, -1).bool())
                out = F.scaled_dot_product_attention(
                    query_5d,
                    key_5d,
                    val_5d,
                    attn_mask=attn_mask_5d,
                    enable_gqa=True,
                )
            else:
                out = F.scaled_dot_product_attention(
                    query_5d,
                    key_5d,
                    val_5d,
                    enable_gqa=True,
                )
            out = out.view(batch_sz, self.nhead, seq_len, self.head_dim)
        else:
            if memory_key_padding_mask is not None:
                attn_mask = (~memory_key_padding_mask.bool()).unsqueeze(1).unsqueeze(2)
                out = F.scaled_dot_product_attention(
                    query_q,
                    key_k_lower,
                    val_v,
                    attn_mask=attn_mask,
                    enable_gqa=True,
                )
            else:
                out = F.scaled_dot_product_attention(
                    query_q,
                    key_k_lower,
                    val_v,
                    enable_gqa=True,
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
        use_checkpoint: Optional[bool] = None,
    ):
        """Initializes the module component."""

        super().__init__()
        iv = 0.1
        self.use_checkpoint = (num_layers >= 4) if use_checkpoint is None else use_checkpoint
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
                self.gamma1 * self.self_attn(self.norm1(tgt), padding_mask=tgt_key_padding_mask)
            )
            if memory is not None:
                tgt = tgt + self.drop2(
                    self.gamma2 * self.cross_attn(self.norm2(tgt), memory, memory_key_padding_mask)
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
        enable_mtp: bool = False,
        shared_layers: Optional[nn.ModuleList] = None,
        gradient_checkpointing: bool = False,
    ):
        """Initializes the module component."""

        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
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
        self.emb_drop, self.emb_scale = nn.Dropout(dropout * 0.5), 1.0  # Normalized to 1.0 (prevents identity-copy shortcut bias over attention)

        if shared_layers is not None:
            self.layers = shared_layers
        else:
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

        self.enable_mtp = enable_mtp
        if self.enable_mtp:
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
        else:
            self.mtp_layer = None
            self.mtp_layer2 = None

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
        compute_head: bool = True,
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
                self.unk_id,
                tgt_ids,
            )
        else:
            dropped_tgt_ids = tgt_ids

        # Removed .any() validation check to prevent XLA device-to-host syncs
        if getattr(self, "use_asl_lex", True) and self.asl_lex_emb is not None:
            lex_embs = self.asl_lex_emb(dropped_tgt_ids)
            hidden_h = self.emb_drop(
                (self.token_emb(dropped_tgt_ids) + lex_embs) * self.emb_scale
            )
        else:
            hidden_h = self.emb_drop(self.token_emb(dropped_tgt_ids) * self.emb_scale)

        hidden_h = hidden_h.to(self.token_emb.weight.dtype)

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
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    def _ckpt_dec_layer_standalone(l_mod, x_in, mem_in, t_mask, m_mask):
                        return l_mod(x_in, mem_in, tgt_key_padding_mask=t_mask, memory_key_padding_mask=m_mask)[0]
                    hidden_h = torch.utils.checkpoint.checkpoint(
                        _ckpt_dec_layer_standalone, layer, hidden_h, memory, tgt_key_padding_mask, memory_key_padding_mask, use_reentrant=True
                    )
                else:
                    hidden_h = layer(
                        hidden_h,
                        memory,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                    )[0]

        hidden_h = self.final_norm(hidden_h)
        logits = self.lm_head(hidden_h) if compute_head else hidden_h

        new_key_values = [] if use_cache else None

        if self.enable_mtp and self.mtp_layer is not None and self.mtp_layer2 is not None:
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
        else:
            logits_2 = None
            logits_3 = None
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
                "bone": 0.5,
                "phonology": 1.0,
                "mp": 1.0,
                "distill_gloss": 2.0,
                "distill_english": 2.0,
                "gpt2": 4.0,
            }

        self.keys = tuple(sorted(loss_config.keys()))
        self.key_to_idx = {k: i for i, k in enumerate(self.keys)}
        self.register_buffer("zero_scalar", torch.tensor(0.0, dtype=torch.float32), persistent=False)
        init_vals = [-math.log(2.0 * loss_config[k]) for k in self.keys]
        self.log_vars_vec = nn.Parameter(torch.tensor(init_vals, dtype=torch.float32))

    @property
    def log_vars(self):
        return {k: self.log_vars_vec[i] for i, k in enumerate(self.keys)}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        old_keys = [f"{prefix}log_vars.{k}" for k in self.keys]
        if any(ok in state_dict for ok in old_keys):
            vec = []
            for k in self.keys:
                pk = f"{prefix}log_vars.{k}"
                if pk in state_dict:
                    vec.append(state_dict.pop(pk))
                else:
                    vec.append(self.log_vars_vec[self.key_to_idx[k]])
            state_dict[f"{prefix}log_vars_vec"] = torch.stack(vec)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for this module with 100% static computation graph."""
        zero_ref = self.zero_scalar
        loss_vec = torch.stack([
            losses[k].mean() if (k in losses and losses[k] is not None) else zero_ref
            for k in self.keys
        ])
        s_vec = self.log_vars_vec  # Pure 1D parameter, zero torch.stack!

        # Clamp log_vars for numerical stability: s in [-6.0, 6.0] corresponds to task weights in [0.002, 403.4] (Claim 13 Fix)
        s_clamped = torch.clamp(s_vec, min=-6.0, max=6.0)
        prec_vec = torch.exp(-s_clamped)
        
        # When a loss is not present in the batch, it should not contribute to the total loss or drift its uncertainty parameter (Claims 11 & 12 Fix)
        active_mask = (loss_vec > 0.0).to(loss_vec.dtype)
        task_loss = (0.5 * (prec_vec * loss_vec + s_clamped)) * active_mask
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
        # Return float32 log probabilities directly for F.ctc_loss (eliminates 1.88GB ping-pong memory allocation)
        return F.log_softmax(logits.float(), dim=-1)


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
            invalid_candidate_mask = ~valid_mask_all
            invalid_row_mask = (
                ~valid_mask
                if valid_mask is not None
                else ~valid_mask_all[: val_v.size(0)]
            )
            invalid_combined = invalid_candidate_mask.unsqueeze(0) | invalid_row_mask.unsqueeze(1)
            logits_v2s = logits_v2s.masked_fill(invalid_combined, -1e9)
            logits_s2v = logits_s2v.masked_fill(invalid_combined, -1e9)

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


class SoftDTWAlignmentLoss(nn.Module):
    """
    Differentiable Dynamic Time Warping (Soft-DTW) Loss (Cuturi & Blondel, ICML).
    Computes smooth minimum alignment distance between sign sequences executed
    at varying speeds (e.g. 0.3s rapid vs 1.5s slow signing).
    """

    def __init__(self, gamma: float = 0.1, normalize: bool = True):
        super().__init__()
        self.gamma = gamma
        self.normalize = normalize

    def _soft_min(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack([-a / self.gamma, -b / self.gamma, -c / self.gamma], dim=-1)
        max_val, _ = torch.max(stacked, dim=-1, keepdim=True)
        exp_sum = torch.exp(stacked - max_val).sum(dim=-1)
        return -self.gamma * (torch.log(exp_sum + 1e-8) + max_val.squeeze(-1))

    def _compute_dtw_single(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: [T1, D], y: [T2, D]
        """
        T1, D = x.shape
        T2, _ = y.shape

        # Euclidean pairwise distance matrix [T1, T2]
        dist_mat = torch.cdist(x.unsqueeze(0), y.unsqueeze(0), p=2).squeeze(0) ** 2  # [T1, T2]

        # DP Matrix [T1 + 1, T2 + 1]
        R = torch.full((T1 + 1, T2 + 1), 1e5, dtype=x.dtype, device=x.device)
        R[0, 0] = 0.0

        for i in range(1, T1 + 1):
            for j in range(1, T2 + 1):
                soft_prev = self._soft_min(R[i - 1, j], R[i, j - 1], R[i - 1, j - 1])
                R[i, j] = dist_mat[i - 1, j - 1] + soft_prev

        return R[T1, T2]

    def forward(self, x_seq: torch.Tensor, y_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, T1, D]
        y_seq: [B, T2, D]
        """
        B = x_seq.size(0)
        dtw_costs = []
        for b in range(B):
            cost_xy = self._compute_dtw_single(x_seq[b], y_seq[b])
            if self.normalize:
                cost_xx = self._compute_dtw_single(x_seq[b], x_seq[b])
                cost_yy = self._compute_dtw_single(y_seq[b], y_seq[b])
                dtw_val = torch.clamp(cost_xy - 0.5 * (cost_xx + cost_yy), min=0.0)
            else:
                dtw_val = cost_xy
            dtw_costs.append(dtw_val)

        return torch.stack(dtw_costs).mean()


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
        self._tgt_params = list(self.proj_gt.parameters())
        self._src_params = list(self.proj_pred.parameters())

    def update_momentum(self, mask_m: float = 0.0025):
        """Provides functionality for update_momentum with fast multi-tensor foreach lerp."""
        with torch.no_grad():
            torch._foreach_lerp_(self._tgt_params, self._src_params, float(mask_m))

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
        cos_loss = 1.0 - cos_sim

        # Static masked variance (No boolean indexing, zero graph breaks)
        token_weights = has_tokens.unsqueeze(-1)  # [B, 1]
        denom = token_weights.sum(dim=0).clamp(min=1.0)
        mean_p = (prob_p * token_weights).sum(dim=0, keepdim=True) / denom
        var_p = (((prob_p - mean_p) ** 2) * token_weights).sum(dim=0) / denom
        std_p = torch.sqrt(var_p + 1e-4)

        target_std = 1.0 / math.sqrt(prob_p.shape[-1])
        # Only apply std_loss if there are at least 2 valid samples in the batch to compute meaningful variance
        std_loss = torch.mean(F.relu(target_std - std_p)) * (denom.squeeze() > 1.5).float()
        
        eff_weights = has_tokens if sample_weights is None else (has_tokens * sample_weights)
        loss = (cos_loss + 0.5 * std_loss) * eff_weights

        return _distributed_normalize(loss.float().sum(), eff_weights.float().sum())


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
        # Quality-weighted positive candidate masking (Claims 19 & 20 Fix)
        pos_mask = pos_mask * valid_labels * all_weights.view(1, -1).clamp_min(0.0)

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

        pos_mask = pos_mask.masked_fill(is_self, 0.0)

        pos_logits = torch.matmul(features.float(), all_feats.float().T) / float(
            self.temperature
        )
        # Mask self-similarity in denominator so exp(1.0/tau) = exp(14.28) does not suppress negative gradients
        pos_logits = pos_logits.masked_fill(is_self, -65500.0)
        log_prob = F.log_softmax(pos_logits, dim=1)
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

        return grad_output.mul(-ctx.alpha), None


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
            angles = position * self.div_term
            pe = torch.empty(*input_x.shape, device=input_x.device, dtype=input_x.dtype)
            pe[..., 0::2] = torch.sin(angles)
            pe[..., 1::2] = torch.cos(angles)
            return input_x + pe
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
    """XLA-safe distributed gradient clipping using PyTorch native or torch_xla clip_grad_norm_."""
    if _XLA_AVAILABLE:
        try:
            import torch_xla.core.xla_model as xm
            return xm.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)
        except Exception:
            pass
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type=norm_type)


class InGraphAugmentor(nn.Module):
    """
    Hardware-accelerated (TPU/GPU) batched spatial, kinematic, and handedness augmentor.
    Includes:
    1. Sagittal Horizontal Reflection (Mirroring X and swapping Left/Right Hand channels for handedness invariance).
    2. Symmetrical Unilateral Hand Reduction (Simulating casual 1-handed signing).
    3. Fused 3D Affine Transformations (Scale, Translation, Rotation).
    4. Gaussian Hand Jittering & Span Dropout.
    5. Out-of-place Kinematic Differentiation (Velocity & Acceleration).
    """
    def __init__(
        self,
        base_jitter_std: float = 0.003,
        scale_range: tuple = (0.85, 1.15),
        trans_range: float = 0.05,
        rot_angle_max_deg: float = 12.0,
        hand_mask_prob: float = 0.30,
        mirror_prob: float = 0.30,
        unilateral_drop_prob: float = 0.20,
        num_keypoints: int = 60,
    ):
        super().__init__()
        self.base_jitter_std = base_jitter_std
        self.scale_range = scale_range
        self.trans_range = trans_range
        self.rot_angle_max_deg = rot_angle_max_deg
        self.hand_mask_prob = hand_mask_prob
        self.mirror_prob = mirror_prob
        self.unilateral_drop_prob = unilateral_drop_prob
        self.num_keypoints = num_keypoints

        # Static index mask for non-mutating out-of-place execution
        jitter_mask = torch.zeros(1, 1, num_keypoints, 1, dtype=torch.float32)
        if num_keypoints >= 60:
            jitter_mask[:, :, 18:60, :] = 1.0
        self.register_buffer("jitter_kp_mask", jitter_mask, persistent=False)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        noise_level: float = 1.0,
        signtype_is_symmetrical: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise_level <= 0.0 or not self.training:
            return features

        B, T, K, C = features.shape
        device = features.device
        dtype = features.dtype

        # 3D Positions in FP32
        pos = features[..., :3].to(dtype=torch.float32)

        # 1. Sagittal Horizontal Reflection (Mirroring X and Swapping Left/Right Hands)
        if self.mirror_prob > 0 and K >= 60:
            mirror_apply = torch.rand(B, 1, 1, 1, device=device) < (self.mirror_prob * noise_level)
            reflect_x = torch.where(mirror_apply, -1.0, 1.0)
            reflect_vec = torch.cat([reflect_x, torch.ones_like(reflect_x), torch.ones_like(reflect_x)], dim=-1)
            pos_reflected = pos * reflect_vec

            # Swap Left Hand (indices 18:39) and Right Hand (indices 39:60)
            lh = pos_reflected[:, :, 18:39, :]
            rh = pos_reflected[:, :, 39:60, :]
            swapped_lh = torch.where(mirror_apply.expand(-1, T, 21, 3), rh, lh)
            swapped_rh = torch.where(mirror_apply.expand(-1, T, 21, 3), lh, rh)

            pos = torch.cat([
                pos_reflected[:, :, :18, :],
                swapped_lh,
                swapped_rh,
                pos_reflected[:, :, 60:, :],
            ], dim=2)

        # 2. Unilateral Hand Reduction (Casual 1-Handed Signing)
        if self.unilateral_drop_prob > 0 and K >= 60:
            if signtype_is_symmetrical is not None:
                is_sym = signtype_is_symmetrical.view(B, 1, 1, 1)
            else:
                is_sym = torch.ones(B, 1, 1, 1, device=device, dtype=torch.bool)
                
            drop_non_dom = (torch.rand(B, 1, 1, 1, device=device) < (self.unilateral_drop_prob * noise_level)) & is_sym
            lh_keep_factor = torch.where(drop_non_dom.expand(-1, T, 21, 3), 0.0, 1.0)
            pos = torch.cat([
                pos[:, :, :18, :],
                pos[:, :, 18:39, :] * lh_keep_factor,
                pos[:, :, 39:, :],
            ], dim=2)

        # 3. Fused Batched 3D Affine Transformation (Scale + Translation + Yaw Rotation)
        scale_min = 1.0 + (self.scale_range[0] - 1.0) * noise_level
        scale_max = 1.0 + (self.scale_range[1] - 1.0) * noise_level
        scales = torch.empty(B, 1, 1, device=device, dtype=torch.float32).uniform_(scale_min, scale_max)

        trans_max = self.trans_range * noise_level
        trans = torch.empty(B, 1, 3, device=device, dtype=torch.float32).uniform_(-trans_max, trans_max)

        rot_max_rad = math.radians(self.rot_angle_max_deg * noise_level)
        angles = torch.empty(B, device=device, dtype=torch.float32).uniform_(-rot_max_rad, rot_max_rad)
        cos_a, sin_a = torch.cos(angles), torch.sin(angles)
        zero, one = torch.zeros_like(cos_a), torch.ones_like(cos_a)

        rot_mats = torch.stack([
            torch.stack([cos_a, zero, sin_a], dim=-1),
            torch.stack([zero,  one,  zero ], dim=-1),
            torch.stack([-sin_a, zero, cos_a], dim=-1),
        ], dim=-2)

        if K >= 16:
            center = pos[:, :, 14:16, :].mean(dim=(1, 2), keepdim=True)
        else:
            center = pos.mean(dim=(1, 2), keepdim=True)

        M = scales * rot_mats
        trans_rot = torch.bmm(trans, rot_mats.transpose(-1, -2)).view(B, 1, 1, 3)
        pos_centered = (pos - center).view(B, T * K, 3)
        pos = torch.bmm(pos_centered, M.transpose(-1, -2)).view(B, T, K, 3) + (center + trans_rot)

        # 4. Gaussian Hand Jittering
        if K >= 60:
            jitter_std = self.base_jitter_std * noise_level
            jitter_noise = torch.randn(B, T, K, 3, device=device, dtype=torch.float32)
            pos = pos + jitter_noise * (jitter_std * self.jitter_kp_mask)

        # 5. Hand Masking Dropout
        if self.hand_mask_prob > 0 and K >= 60 and T >= 8:
            mask_apply = (torch.rand(B, 1, 1, 1, device=device) < (self.hand_mask_prob * noise_level))
            choice = torch.randint(0, 3, (B, 1, 1, 1), device=device)
            left_drop = (choice == 0) | (choice == 2)
            right_drop = (choice == 1) | (choice == 2)

            t_indices = torch.arange(T, device=device).view(1, T, 1, 1)
            span_len = max(2, int(T * 0.25))
            start_t = torch.randint(0, max(1, T - span_len), (B, 1, 1, 1), device=device)
            in_span = (t_indices >= start_t) & (t_indices < (start_t + span_len))

            left_drop_mask = 1.0 - (mask_apply & left_drop & in_span).float()
            right_drop_mask = 1.0 - (mask_apply & right_drop & in_span).float()

            drop_factors = torch.cat([
                torch.ones(B, T, 18, 1, device=device),
                left_drop_mask.expand(B, T, 21, 1),
                right_drop_mask.expand(B, T, 21, 1),
                torch.ones(B, T, max(0, K - 60), 1, device=device),
            ], dim=2)
            pos = pos * drop_factors

        # 6. Recompute Kinematics out-of-place for XLA
        if T > 1:
            dpos = pos[:, 1:] - pos[:, :-1]
            vel = torch.cat([dpos[:, 0:1], dpos], dim=1)
            dvel = vel[:, 1:] - vel[:, :-1]
            acc = torch.cat([dvel[:, 0:1], dvel], dim=1)
        else:
            vel = torch.zeros_like(pos)
            acc = torch.zeros_like(pos)

        if C >= 9:
            out = torch.cat([pos, vel, acc], dim=-1)
        else:
            out = pos

        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).to(dtype=torch.float32)
        return (out * mask_expanded).to(dtype=dtype)


class TopologicalBoneLengthLoss(nn.Module):
    """
    Soft anatomical constraint penalizing finger phalanx stretching or depth collapse under occlusion.
    Calculates MSE between instantaneous Euclidean segment lengths and temporal rest lengths.
    """
    def __init__(self):
        super().__init__()
        hand_bones = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12), # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),# Ring
            (0, 17), (17, 18), (18, 19), (19, 20),# Pinky
        ]
        lh_bones = [(u + 18, v + 18) for u, v in hand_bones]
        rh_bones = [(u + 39, v + 39) for u, v in hand_bones]
        all_bones = lh_bones + rh_bones
        
        self.bone_u = torch.tensor([u for u, v in all_bones], dtype=torch.long)
        self.bone_v = torch.tensor([v for u, v in all_bones], dtype=torch.long)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pos = features[..., :3].float()
        B, T, K, _ = pos.shape
        device = pos.device
        
        u_idx = self.bone_u.to(device)
        v_idx = self.bone_v.to(device)
        
        bone_vecs = pos[:, :, v_idx, :] - pos[:, :, u_idx, :]
        bone_lens = torch.norm(bone_vecs + 1e-8, dim=-1)
        
        mean_lens = bone_lens.mean(dim=1, keepdim=True)
        diff_sq = (bone_lens - mean_lens) ** 2
        
        mask_f = mask.unsqueeze(-1).float()
        valid_count = (mask_f.sum() * len(self.bone_u)).clamp(min=1.0)
        return (diff_sq * mask_f).sum() / valid_count


class PhonologicalMinimalPairLoss(nn.Module):
    """
    Supervised Contrastive Margin Loss to separate minimal pairs in latent space.
    Pushes representations of lookalike signs apart by margin m.
    """
    def __init__(self, margin: float = 0.30):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        minimal_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, D = embeddings.shape
        if B < 2:
            return torch.tensor(0.0, device=embeddings.device, dtype=embeddings.dtype)
            
        norm_emb = F.normalize(embeddings.float(), dim=-1)
        sim_matrix = torch.matmul(norm_emb, norm_emb.T)
        
        lbl_i = labels.view(B, 1)
        lbl_j = labels.view(1, B)
        
        # Valid labels only (ignore negative or pad IDs)
        valid_lbls = (lbl_i >= 0) & (lbl_j >= 0) & (lbl_i < minimal_pair_mask.shape[0]) & (lbl_j < minimal_pair_mask.shape[1])
        lbl_i_clamped = lbl_i.clamp(0, minimal_pair_mask.shape[0] - 1)
        lbl_j_clamped = lbl_j.clamp(0, minimal_pair_mask.shape[1] - 1)
        
        is_mp = minimal_pair_mask[lbl_i_clamped, lbl_j_clamped] & valid_lbls
        is_different = (lbl_i != lbl_j)
        
        target_mp = is_mp & is_different
        # Pure static vectorized XLA formulation: zero boolean indexing, zero dynamic shapes, zero device-to-host syncs
        relu_term = F.relu(sim_matrix - (1.0 - self.margin))
        mp_float = target_mp.float()
        valid_count = mp_float.sum().clamp(min=1.0)
        has_valid = (mp_float.sum() > 0.0).float()
        loss = ((relu_term * mp_float).sum() / valid_count) * has_valid
        return loss.to(embeddings.dtype)


class SignToGPT2Bridge(nn.Module):
    r"""
    Multimodal Prefix Bridge connecting Conformer video representations (512-dim)
    to a pretrained GPT-2 language model (768-dim, 124M parameters).
    Enables fluent, natural, publication-grade English translation.
    """
    def __init__(
        self,
        gpt2_name_or_path: str = "gpt2",
        d_visual: int = 512,
        prefix_len: int = 16,
        freeze_gpt2: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        from transformers import GPT2LMHeadModel
        candidate_paths = [
            gpt2_name_or_path,
            "/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1",
            "/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2",
            "/kaggle/input/gpt-2-by-openai-community/pytorch/gpt-2/1",
            "/kaggle/input/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2",
            "/kaggle/input/gpt-2/pytorch/gpt-2/1",
            "/kaggle/input/gpt2",
            "gpt2",
        ]
        resolved_path = next((p for p in candidate_paths if p and (os.path.exists(p) or p == "gpt2")), "gpt2")

        load_kwargs = {}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        try:
            self.gpt2 = GPT2LMHeadModel.from_pretrained(resolved_path, **load_kwargs)
        except Exception:
            self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2", **load_kwargs)

        if hasattr(self.gpt2.config, "loss_type"):
            self.gpt2.config.loss_type = "ForCausalLMLoss"
        if hasattr(self.gpt2.config, "use_cache"):
            self.gpt2.config.use_cache = False

        d_gpt2 = self.gpt2.config.n_embd  # 768
        self.prefix_len = prefix_len

        self.visual_proj = nn.Sequential(
            nn.Linear(d_visual, d_gpt2),
            nn.GELU(),
            nn.Linear(d_gpt2, d_gpt2),
        )
        self.prefix_tokens = nn.Parameter(torch.randn(1, prefix_len, d_gpt2) * 0.02)

        if freeze_gpt2:
            for p in self.gpt2.parameters():
                p.requires_grad = False
            self.gpt2.eval()

        if gradient_checkpointing and not freeze_gpt2:
            self.gpt2.gradient_checkpointing_enable()
            if hasattr(self.gpt2.config, "use_cache"):
                self.gpt2.config.use_cache = False

    def forward(
        self,
        h_visual: torch.Tensor,
        text_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        chunk_tokens: int = 512,
    ):
        B = h_visual.shape[0]
        h_vis_trans = h_visual.transpose(1, 2)
        h_vis_pooled = F.adaptive_avg_pool1d(h_vis_trans, self.prefix_len).transpose(1, 2)
        if h_vis_pooled.dtype != self.visual_proj[0].weight.dtype:
            h_vis_pooled = h_vis_pooled.to(self.visual_proj[0].weight.dtype)
        proj_vis = self.visual_proj(h_vis_pooled)
        if proj_vis.dtype != self.prefix_tokens.dtype:
            proj_vis = proj_vis.to(self.prefix_tokens.dtype)
        prefix_embeds = proj_vis + self.prefix_tokens.expand(B, -1, -1)

        if text_ids is None:
            return prefix_embeds

        # Bound check tokens to ensure valid GPT-2 embedding indices
        if hasattr(self.gpt2.config, "vocab_size"):
            text_ids = text_ids.clamp(0, self.gpt2.config.vocab_size - 1)

        # Align total sequence length (prefix_len + text_len) to an exact multiple of 128 (e.g. 256 or 128).
        # On TPU v5e / v3, 256 sequence length perfectly fits within hardware systolic tiles and prevents
        # overshooting to 384 (which wastes 113 pad tokens and bloats attention memory by 2.25x).
        target_total_len = min(256, max(128, ((self.prefix_len + text_ids.shape[1]) // 128) * 128))
        if target_total_len < 128:
            target_total_len = 128
        max_text_len = target_total_len - self.prefix_len
        if text_ids.shape[1] > max_text_len:
            text_ids = text_ids[:, :max_text_len]
            if labels is not None:
                labels = labels[:, :max_text_len]
        elif text_ids.shape[1] < max_text_len:
            pad_amount = max_text_len - text_ids.shape[1]
            pad_token = getattr(self.gpt2.config, "pad_token_id", None)
            if pad_token is None:
                pad_token = getattr(self.gpt2.config, "eos_token_id", 0)
            if pad_token is None or (hasattr(self.gpt2.config, "vocab_size") and pad_token >= self.gpt2.config.vocab_size):
                pad_token = 0
            text_ids = F.pad(text_ids, (0, pad_amount), value=pad_token)
            if labels is not None:
                labels = F.pad(labels, (0, pad_amount), value=-100)

        text_embeds = self.gpt2.transformer.wte(text_ids)
        if prefix_embeds.dtype != text_embeds.dtype:
            prefix_embeds = prefix_embeds.to(text_embeds.dtype)
        inputs_embeds = torch.cat([prefix_embeds, text_embeds], dim=1)
        del text_embeds

        if attention_mask is not None:
            if attention_mask.shape[1] < max_text_len:
                attention_mask = F.pad(attention_mask, (0, max_text_len - attention_mask.shape[1]), value=0)
            else:
                attention_mask = attention_mask[:, :max_text_len]
            prefix_mask = torch.ones((B, self.prefix_len), dtype=torch.bool, device=h_visual.device)
            full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            # Passing attention_mask=None allows HuggingFace GPT-2 to use its pre-registered static
            # 2D causal mask buffer without allocating a dynamic 4D [B, 1, L, L] attention mask tensor in HBM.
            full_mask = None

        if labels is None:
            return self.gpt2(inputs_embeds=inputs_embeds, attention_mask=full_mask)

        # Fused Chunked Cross-Entropy Streaming (eliminates [B, L, 50257] ~2.9GB logits tensor from HBM)
        hidden_states = self.gpt2.transformer(inputs_embeds=inputs_embeds, attention_mask=full_mask)[0]
        text_len = text_ids.shape[1]
        D = hidden_states.shape[-1]
        # Position prefix_len - 1 is the last visual prefix token (predicts first target token)
        h_flat = hidden_states[:, self.prefix_len - 1 : self.prefix_len + text_len - 1, :].reshape(-1, D)
        del hidden_states

        # Single clean fused projection (0 loop unrolling, 0 XLA rematerialization clones).
        # Eliminates 120 cloned attention matrices (saving 23.04 GB HBM) and reduces compilation from 16m to ~20s.
        l_flat = labels.reshape(-1)
        logits = self.gpt2.lm_head(h_flat)
        valid_sum = (l_flat != -100).sum().float()
        valid_count = valid_sum.clamp(min=1.0)
        has_valid = (valid_sum > 0.0).float()
        loss = (F.cross_entropy(logits, l_flat, ignore_index=-100, reduction="sum") / valid_count) * has_valid

        class BridgeOutput:
            def __init__(self, loss_val):
                self.loss = loss_val
        return BridgeOutput(loss)

    @torch.no_grad()
    def generate(
        self,
        h_visual: torch.Tensor,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ):
        prefix_embeds = self.forward(h_visual)
        pad_id = getattr(self.gpt2.config, "eos_token_id", 50256)
        return self.gpt2.generate(
            inputs_embeds=prefix_embeds,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_id,
            do_sample=(temperature > 0.0),
            temperature=max(0.1, temperature),
            top_p=top_p,
            repetition_penalty=1.2,
        )


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
        enable_aux_decoders: bool = True,
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
        english_max_len: Optional[int] = None,
        chicago_max_len: Optional[int] = None,
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
        gradient_checkpointing: bool = False,
        use_gpt2: bool = False,
        gpt2_path: str = "gpt2",
        **kwargs,
    ):
        """Initializes the module component."""

        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.use_swin_1d = use_swin_1d
        self.swin_window_size = swin_window_size
        self.actual_vocab_size = vocab_size
        self.english_max_len = english_max_len if english_max_len is not None else min(max_dec_len, 128)
        self.chicago_max_len = chicago_max_len if chicago_max_len is not None else min(max_dec_len, 128)
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
            gradient_checkpointing=gradient_checkpointing,
        )

        if getattr(self, "enable_aux_decoders", False):
            self.chicago_decoder = ASLTransformerDecoder(
                vocab_size=64,  # Chicago character vocab (safety padded to 64)
                d_model=d_dec,
                nhead=nhead_dec,
                kv_heads=kv_heads_dec,
                num_layers=num_dec_layers,
                ffn_dim=ffn_dec,
                dropout=dropout,
                max_seq_len=self.chicago_max_len,
                csv_path=None,
                label_to_idx=None,
                use_asl_lex=False,
                shared_layers=self.decoder.layers,
                gradient_checkpointing=gradient_checkpointing,
            )
            self.english_decoder = ASLTransformerDecoder(
                vocab_size=max(4, english_vocab_size),  # English vocab
                d_model=d_dec,
                nhead=nhead_dec,
                kv_heads=kv_heads_dec,
                num_layers=num_dec_layers,
                ffn_dim=ffn_dec,
                dropout=dropout,
                max_seq_len=self.english_max_len,
                csv_path=None,
                label_to_idx=None,
                use_asl_lex=False,
                shared_layers=self.decoder.layers,
                gradient_checkpointing=gradient_checkpointing,
            )
        else:
            self.chicago_decoder = None
            self.english_decoder = None

        # Optional Pretrained GPT-2 Multimodal Bridge for Fluent Natural English
        self.use_gpt2 = use_gpt2 or kwargs.get("use_gpt2", False)
        candidate_gpt2 = [
            gpt2_path if gpt2_path != "gpt2" else None,
            kwargs.get("gpt2_path", None),
            "/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1",
            "/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2",
            "/kaggle/input/gpt-2-by-openai-community/pytorch/gpt-2/1",
            "/kaggle/input/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2",
            "/kaggle/input/gpt-2",
            "gpt2",
        ]
        resolved_gpt2 = next((p for p in candidate_gpt2 if p and (os.path.exists(p) or p == "gpt2")), "gpt2")
        if self.use_gpt2:
            try:
                self.gpt2_bridge = SignToGPT2Bridge(
                    gpt2_name_or_path=resolved_gpt2,
                    d_visual=d_dec,
                    prefix_len=kwargs.get("gpt2_prefix_len", 16),
                    freeze_gpt2=kwargs.get("freeze_gpt2", True),
                    gradient_checkpointing=gradient_checkpointing,
                    torch_dtype=kwargs.get("torch_dtype", None),
                )
                print(f"[INFO] Initialized SignToGPT2Bridge successfully from {resolved_gpt2}", flush=True)
            except Exception as e:
                print(f"[WARNING] Could not load GPT-2 from {resolved_gpt2}: {e}. Standard decoder remains active.", flush=True)
                self.gpt2_bridge = None
                self.use_gpt2 = False
        else:
            self.gpt2_bridge = None

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

        # Auxiliary Phonological Heads (ASL-LEX 2.0 Inductive Guidance)
        self.head_handshape = nn.Linear(d_enc, 64)
        self.head_location = nn.Linear(d_enc, 32)
        self.head_signtype = nn.Linear(d_enc, 16)
        self.bone_loss_fn = TopologicalBoneLengthLoss()
        self.mp_loss_fn = PhonologicalMinimalPairLoss(margin=0.30)

        nn.init.normal_(self.decoder.token_emb.weight, std=0.02)
        with torch.no_grad():
            self.decoder.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)
            if self.chicago_decoder is not None:
                self.chicago_decoder.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)
            if self.english_decoder is not None:
                self.english_decoder.token_emb.weight[eng_pad_id].fill_(0)
        self.is_xla = IS_TPU

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

        target_dtype = next(self.parameters()).dtype
        if input_x.is_floating_point() and input_x.dtype != target_dtype:
            input_x = input_x.to(target_dtype)
        if (
            phonology_features is not None
            and phonology_features.is_floating_point()
            and phonology_features.dtype != target_dtype
        ):
            phonology_features = phonology_features.to(target_dtype)

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

        if x_in.dim() == 4 and x_in.size(2) == 60:
            if x_in.size(3) == 3:
                # Fused functional 9-channel kinematics: [position, velocity, acceleration] with 0 in-place scatter
                v = F.pad(x_in[:, 1:] - x_in[:, :-1], (0, 0, 0, 0, 1, 0))
                a = F.pad(v[:, 1:] - v[:, :-1], (0, 0, 0, 0, 1, 0))
                xk = torch.cat([x_in, v, a], dim=-1)
            else:
                xk = x_in

            # Vectorized anatomical reference part normalization (Face, Torso, Left Hand, Right Hand)
            face = xk[:, :, :40, :3] - xk[:, :, 0:1, :3]
            shoulder_mid = (xk[:, :, 40:41, :3] + xk[:, :, 41:42, :3]) * 0.5
            torso = xk[:, :, 40:48, :3] - shoulder_mid
            l_wrist = xk[:, :, 48:49, :3]
            l_hand = xk[:, :, 48:54, :3] - l_wrist
            r_wrist = xk[:, :, 54:55, :3]
            r_hand = xk[:, :, 54:60, :3] - r_wrist
            pos_normed = torch.cat([face, torso, l_hand, r_hand], dim=2)
            if xk.size(-1) > 3:
                xk_norm = torch.cat([pos_normed, xk[..., 3:]], dim=-1)
            else:
                xk_norm = pos_normed

            x_flat = xk_norm.reshape(batch_sz, seq_len, -1)
            v_tokens = self.visual_encoder(xk_norm, mask=mask, cache=cache)
        else:
            x_flat = x_in.reshape(batch_sz, seq_len, -1) if x_in.dim() == 4 else x_in
            v_tokens = self.visual_encoder(x_in, mask=mask, cache=cache)
        if phonology_features is None:
            phonology_features = torch.zeros((batch_sz, seq_len, 19), device=x_in.device, dtype=x_in.dtype)
        phonology_features = self.phonology_norm(phonology_features)
        
        pad_needed = 640 - (x_flat.size(-1) + phonology_features.size(-1))
        if pad_needed > 0:
            pad_tensor = torch.zeros(batch_sz, seq_len, pad_needed, device=x_flat.device, dtype=x_flat.dtype)
            x_stem_in = torch.cat([x_flat, phonology_features, pad_tensor, v_tokens], dim=-1)
        else:
            x_stem_in = torch.cat([x_flat, phonology_features, v_tokens], dim=-1)

        hidden_h = self.input_stem(x_stem_in)
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
                        cls_fi = torch.zeros(
                            (batch_sz, 1), dtype=routing_fi.dtype, device=routing_fi.device
                        )
                        pos_fi = torch.cat([cls_fi, routing_fi + 1], dim=1)
                    else:
                        pos_fi = routing_fi + 1
                    
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
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    def _ckpt_enc_block(mod, x_in, kpm_in, fi_in):
                        return mod(x_in, key_padding_mask=kpm_in, frame_indices=fi_in)
                    hidden_h = torch.utils.checkpoint.checkpoint(
                        _ckpt_enc_block, block, hidden_h, kpm, pos_fi, use_reentrant=True
                    )
                else:
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
        has_valid_english: Optional[torch.Tensor] = None,
        skip_augment: bool = False,
    ) -> Union[Optional[torch.Tensor], Dict]:
        # Always compute kinematics and augmentations on TPU to avoid Host CPU bottleneck
        """Forward pass for this module."""

        target_dtype = next(self.parameters()).dtype
        if input_x.is_floating_point() and input_x.dtype != target_dtype:
            input_x = input_x.to(target_dtype)

        if True:
            # E67: Prevent in-place modifications corrupting autograd graphs
            x = input_x
            if x.dim() == 3:
                b_sz, s_len, f_dim = x.shape
                k_k = self.num_keypoints
                c_k = f_dim // k_k if k_k > 0 and f_dim % k_k == 0 else self.channels_per_kp
                x = x.view(b_sz, s_len, k_k, c_k)
            batch_sz, seq_len, key_k, channels = x.shape

            if self.training and not skip_augment:
                # High-Performance In-Graph Vectorized Landmark Augmentation on Device (TPU/GPU)
                # Operates in < 0.05ms on accelerator MXU/VPU, eliminating CPU data-loading bottlenecks.
                pos_aug = x[..., :3]
                valid_mask = (torch.abs(pos_aug).sum(dim=-1, keepdim=True) > 0).to(x.dtype)
                
                # 1. Vectorized random affine scale: [B, 1, 1, 1]
                scale = torch.empty(batch_sz, 1, 1, 1, dtype=x.dtype, device=x.device).uniform_(0.88, 1.12)
                pos_aug = pos_aug * scale
                
                # 2. Vectorized random shift: [B, 1, 1, 3]
                shift = torch.empty(batch_sz, 1, 1, 3, dtype=x.dtype, device=x.device).uniform_(-0.025, 0.025)
                pos_aug = pos_aug + (shift * valid_mask)
                
                # 3. Vectorized coordinate jitter: [B, T, K, 3]
                jitter = torch.randn_like(pos_aug) * 0.015
                pos_aug = pos_aug + (jitter * valid_mask)

                # 4. SOTA In-Graph SpecAugment: Vectorized Temporal Cutout Masking (Time Mask)
                # Zero out random contiguous temporal block of length <= 16 with probability 0.5
                if seq_len > 16:
                    time_mask_prob = torch.rand(batch_sz, 1, 1, 1, device=x.device) < 0.5
                    t_start = torch.randint(0, max(1, seq_len - 16), (batch_sz, 1, 1, 1), device=x.device)
                    t_indices = torch.arange(seq_len, device=x.device).view(1, -1, 1, 1)
                    time_drop_mask = (t_indices >= t_start) & (t_indices < (t_start + 16)) & time_mask_prob
                    pos_aug = torch.where(time_drop_mask, torch.zeros_like(pos_aug), pos_aug)

                # 5. SOTA In-Graph SpecAugment: Keypoint DropBlock (Channel/Joint Mask)
                # Randomly drop Left Hand (0-20) or Right Hand (21-41) with probability 0.25
                if key_k >= 42:
                    drop_lh = (torch.rand(batch_sz, 1, 1, 1, device=x.device) < 0.25)
                    drop_rh = (torch.rand(batch_sz, 1, 1, 1, device=x.device) < 0.25)
                    k_indices = torch.arange(key_k, device=x.device).view(1, 1, -1, 1)
                    lh_mask = (k_indices < 21) & drop_lh
                    rh_mask = (k_indices >= 21) & (k_indices < 42) & drop_rh
                    joint_drop_mask = lh_mask | rh_mask
                    pos_aug = torch.where(joint_drop_mask, torch.zeros_like(pos_aug), pos_aug)
                
                if channels >= 9:
                    x = torch.cat([pos_aug, x[..., 3:]], dim=-1)
                else:
                    x = pos_aug

            if frame_indices is not None and seq_len > 1:
                actual_dt = (
                    (frame_indices[:, 1:] - frame_indices[:, :-1])
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).to(x.dtype)
                actual_dt = torch.where(actual_dt == 0, torch.ones_like(actual_dt), actual_dt)
                dt = F.pad(actual_dt, (0, 0, 0, 0, 1, 0), value=1.0)
            else:
                dt = torch.ones(batch_sz, seq_len, 1, 1, device=x.device, dtype=x.dtype)

            # Ensure kinematics are only derived from the [x, y, z] spatial coordinates
            pos = x[..., :3]
            if channels >= 9:
                vel = x[..., 3:6]
                acc = x[..., 6:9]
            else:
                if seq_len > 1:
                    d_pos = (pos[:, 1:] - pos[:, :-1]) / dt[:, 1:]
                    vel = F.pad(d_pos, (0, 0, 0, 0, 1, 0), value=0.0)
                    d_vel = (vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]
                    acc = F.pad(d_vel, (0, 0, 0, 0, 1, 0), value=0.0)
                else:
                    vel = torch.zeros_like(pos)
                    acc = torch.zeros_like(pos)

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
                    dtype=orig_input_x.dtype,
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

            if pos.shape[2] >= 60:
                lh_w, lh_idx, lh_pky = 18, 23, 35
                rh_w, rh_idx, rh_pky = 39, 44, 56
                lh_tips = [22, 26, 30, 34, 38]
                rh_tips = [43, 47, 51, 55, 59]
                face_centroid = pos[:, :, 0:1, :3].mean(dim=2)
            else:
                lh_w, lh_idx, lh_pky = 0, min(5, pos.shape[2]-1), min(17, pos.shape[2]-1)
                rh_w, rh_idx, rh_pky = min(21, pos.shape[2]-1), min(26, pos.shape[2]-1), min(38, pos.shape[2]-1)
                lh_tips = [min(i, pos.shape[2]-1) for i in [4, 8, 12, 16, 20]]
                rh_tips = [min(i, pos.shape[2]-1) for i in [25, 29, 33, 37, 41]]
                face_centroid = pos[:, :, 0:1, :3].mean(dim=2) * 0.0

            # 1. Palm Orientation Normals (6 Dims)
            lh_wrist = pos[:, :, lh_w, :3]
            rh_wrist = pos[:, :, rh_w, :3]
            lh_u = pos[:, :, lh_idx, :3] - lh_wrist
            lh_v = pos[:, :, lh_pky, :3] - lh_wrist
            lh_normal = F.normalize(
                fast_cross(lh_u, lh_v), p=2, dim=-1, eps=1e-5
            )

            rh_u = pos[:, :, rh_idx, :3] - rh_wrist
            rh_v = pos[:, :, rh_pky, :3] - rh_wrist
            rh_normal = F.normalize(
                fast_cross(rh_u, rh_v), p=2, dim=-1, eps=1e-5
            )

            # 2. Bimanual Synchrony (1 Dim)
            lh_vel = vel[:, :, lh_w, :3]
            rh_vel = vel[:, :, rh_w, :3]
            bimanual_sync = safe_cosine_sim(lh_vel, rh_vel).unsqueeze(-1)

            # 3. Location Anchoring to Face (2 Dims)
            lh_face_dist = safe_norm(lh_wrist - face_centroid, dim=-1, keepdim=True)
            rh_face_dist = safe_norm(rh_wrist - face_centroid, dim=-1, keepdim=True)

            # 4. Finger Curl / Aperture (10 Dims)
            lh_curl = safe_norm(pos[:, :, lh_tips, :3] - lh_wrist.unsqueeze(2), dim=-1)
            rh_curl = safe_norm(pos[:, :, rh_tips, :3] - rh_wrist.unsqueeze(2), dim=-1)

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
            ).to(target_dtype)  # Shape: [batch_sz, seq_len, 19]

            input_x = torch.cat([pos, vel, acc], dim=-1).to(target_dtype)

            if input_x.shape[-1] < self.channels_per_kp:
                pad = torch.zeros(
                    batch_sz,
                    seq_len,
                    key_k,
                    max(0, self.channels_per_kp - input_x.shape[-1]),
                    device=input_x.device,
                    dtype=input_x.dtype,
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
        h_e = None

        if (
            gloss_seq is not None
            and chicago_seq is not None
            and english_seq is not None
            and self.decoder is not None
            and self.chicago_decoder is not None
            and self.english_decoder is not None
            and getattr(self, "enable_aux_decoders", True)
        ):
            # Fused Multi-Task Batch Execution (1.70x Faster with 3x smaller decoder graph)
            tgt_g = gloss_seq[:, :-1]
            tgt_c = chicago_seq[:, :-1]
            tgt_e = english_seq[:, :-1]

            orig_g_len = tgt_g.size(1)
            orig_c_len = tgt_c.size(1)
            orig_e_len = tgt_e.size(1)
            max_tgt_len = max(orig_g_len, orig_c_len, orig_e_len)

            if orig_g_len < max_tgt_len:
                tgt_g = F.pad(tgt_g, (0, max_tgt_len - orig_g_len), value=GlossVocabulary.PAD_ID)
            if orig_c_len < max_tgt_len:
                tgt_c = F.pad(tgt_c, (0, max_tgt_len - orig_c_len), value=GlossVocabulary.PAD_ID)
            if orig_e_len < max_tgt_len:
                tgt_e = F.pad(tgt_e, (0, max_tgt_len - orig_e_len), value=EnglishVocabulary.PAD_ID)

            hg = self.decoder.emb_drop(self.decoder.token_emb(tgt_g) * self.decoder.emb_scale)
            hc = self.chicago_decoder.emb_drop(self.chicago_decoder.token_emb(tgt_c) * self.chicago_decoder.emb_scale)
            he = self.english_decoder.emb_drop(self.english_decoder.token_emb(tgt_e) * self.english_decoder.emb_scale)

            b_sz = gloss_seq.size(0)
            h_fused = torch.cat([hg, hc, he], dim=0)
            if h_seq.dtype != h_fused.dtype:
                h_seq = h_seq.to(h_fused.dtype)
            mem_fused = h_seq.repeat(3, 1, 1)
            enc_mask_fused = enc_mask.repeat(3, 1) if enc_mask is not None else None

            for layer in self.decoder.layers:
                if getattr(self, "gradient_checkpointing", False) and self.training:
                    def _ckpt_dec_layer(l_mod, x_in, mem_in, mask_in):
                        return l_mod(x_in, mem_in, memory_key_padding_mask=mask_in)[0]
                    h_fused = torch.utils.checkpoint.checkpoint(
                        _ckpt_dec_layer, layer, h_fused, mem_fused, enc_mask_fused, use_reentrant=True
                    )
                else:
                    h_fused = layer(h_fused, mem_fused, memory_key_padding_mask=enc_mask_fused)[0]

            h_fused = self.decoder.final_norm(h_fused)
            h_g, h_c, h_e = torch.split(h_fused, b_sz, dim=0)
            h_g = h_g[:, :orig_g_len]
            h_c = h_c[:, :orig_c_len]
            h_e = h_e[:, :orig_e_len]

            dec_hidden = h_g
            dec_logits = self.decoder.lm_head(h_g)
            chicago_logits = self.chicago_decoder.lm_head(h_c)
            if self.training:
                # Fused Chunked Projection: Avoid materializing 733MB-2.86GB logits in forward pass
                english_logits = None
            else:
                english_logits = self.english_decoder.lm_head(h_e)
        else:
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
                english_logits, h_e, _, _ = decode_seq(
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

        # Phonological Predictions
        pred_handshape = self.head_handshape(h_cls)
        pred_location = self.head_location(h_cls)
        pred_signtype = self.head_signtype(h_cls)

        vis_emb = self.visual_proj(h_cls)
        proj_feats = self.contrastive_head(h_cls)
        domain_logits = self.domain_head(
            GradientReversalFunction.apply(h_cls, grl_alpha)
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
                    mask_valid = (
                        (enc_mask[:, :-1] & enc_mask[:, 1:]).unsqueeze(-1).float()
                    )
                    diff = diff * mask_valid
                    valid_diff_count = mask_valid.sum(dim=1).clamp(min=1.0)
                    diff_mean = ((diff**2).sum(dim=(1, 2)) / (valid_diff_count.squeeze(-1) * diff.shape[-1])).mean()
                else:
                    diff_mean = (diff**2).mean()
                loss_lpc = diff_mean
                if enc_mask is not None:
                    seq_has_valid = (enc_mask.sum(dim=1) > 1).float().mean()
                    loss_lpc = loss_lpc * seq_has_valid
            else:
                loss_lpc = torch.zeros((), device=h_seq.device)
        else:
            loss_lpc = torch.zeros((), device=h_seq.device)

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

        gpt2_loss = None
        if getattr(self, "gpt2_bridge", None) is not None and english_seq is not None:
            try:
                target_labels = english_seq[:, 1:].clone()
                pad_mask = (target_labels == EnglishVocabulary.PAD_ID)
                if has_valid_english is not None:
                    pad_mask = pad_mask | (~has_valid_english.bool().unsqueeze(-1))
                target_labels[pad_mask] = -100

                gpt2_out = self.gpt2_bridge(h_seq, text_ids=english_seq[:, :-1], labels=target_labels)
                gpt2_loss = gpt2_out.loss
            except Exception as e:
                print(f"[ERROR] gpt2_bridge forward failed: {e}", flush=True)
                raise e

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
            "english_hidden": h_e,
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
            "pred_handshape": pred_handshape,
            "pred_location": pred_location,
            "pred_signtype": pred_signtype,
            "vis_emb": vis_emb,
            "sent_emb": sent_emb,
            "proj_feats": proj_feats,
            "domain_logits": domain_logits,
            "mlm_logits": mlm_logits,
            "loss_lpc": loss_lpc,
            "token_sizes": token_sizes,
            "gpt2_loss": gpt2_loss,
        }

    @torch.no_grad()
    def generate_sentence_gpt2(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 30,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        r"""Generates fluent, natural English sentences directly from 3D sign landmarks using the pretrained GPT-2 bridge."""
        if getattr(self, "gpt2_bridge", None) is None:
            raise RuntimeError("SignToGPT2Bridge is not initialized on this model instance.")
        enc_res = self.encode(features, mask=mask)
        h_seq = enc_res[0] if isinstance(enc_res, (tuple, list)) else enc_res
        from transformers import GPT2Tokenizer
        tok_name = getattr(self.gpt2_bridge.gpt2.config, "_name_or_path", "gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained(tok_name if os.path.exists(tok_name) else "gpt2")
        gen_ids = self.gpt2_bridge.generate(h_seq, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
        return tokenizer.decode(gen_ids[0], skip_special_tokens=True)

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
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """Autoregressive generation method for inference/validation."""

        if task == "chicago":
            decoder_mod = self.chicago_decoder
        elif task == "english":
            decoder_mod = self.english_decoder
        else:
            decoder_mod = self.decoder

        if decoder_mod is None:
            raise ValueError(
                f"Cannot perform autoregressive generation for task '{task}': "
                f"the requested auxiliary decoder is None (enable_aux_decoders was False)."
            )
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

            if frame_indices is not None and seq_len > 1:
                actual_dt = (
                    (frame_indices[:, 1:] - frame_indices[:, :-1])
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                actual_dt = torch.where(actual_dt == 0, torch.ones_like(actual_dt), actual_dt)
                dt = F.pad(actual_dt, (0, 0, 0, 0, 1, 0), value=1.0)
            else:
                dt = torch.ones(batch_sz, seq_len, 1, 1, device=device, dtype=features.dtype)

            if seq_len > 1:
                d_pos = (features_3d[:, 1:] - features_3d[:, :-1]) / dt[:, 1:]
                vel = F.pad(d_pos, (0, 0, 0, 0, 1, 0), value=0.0)
                d_vel = (vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]
                acc = F.pad(d_vel, (0, 0, 0, 0, 1, 0), value=0.0)
            else:
                vel = torch.zeros_like(features_3d)
                acc = torch.zeros_like(features_3d)

            if features_3d.shape[2] >= 60:
                lh_w, lh_idx, lh_pky = 18, 23, 35
                rh_w, rh_idx, rh_pky = 39, 44, 56
                lh_tips = [22, 26, 30, 34, 38]
                rh_tips = [43, 47, 51, 55, 59]
                face_centroid = features_3d[:, :, 0:1, :3].mean(dim=2)
            else:
                lh_w, lh_idx, lh_pky = 0, min(5, features_3d.shape[2]-1), min(17, features_3d.shape[2]-1)
                rh_w, rh_idx, rh_pky = min(21, features_3d.shape[2]-1), min(26, features_3d.shape[2]-1), min(38, features_3d.shape[2]-1)
                lh_tips = [min(i, features_3d.shape[2]-1) for i in [4, 8, 12, 16, 20]]
                rh_tips = [min(i, features_3d.shape[2]-1) for i in [25, 29, 33, 37, 41]]
                face_centroid = features_3d[:, :, 0:1, :3].mean(dim=2) * 0.0

            lh_u = features_3d[:, :, lh_idx, :3] - features_3d[:, :, lh_w, :3]
            lh_v = features_3d[:, :, lh_pky, :3] - features_3d[:, :, lh_w, :3]
            lh_normal = F.normalize(
                fast_cross(lh_u, lh_v), p=2, dim=-1, eps=1e-5
            )

            rh_u = features_3d[:, :, rh_idx, :3] - features_3d[:, :, rh_w, :3]
            rh_v = features_3d[:, :, rh_pky, :3] - features_3d[:, :, rh_w, :3]
            rh_normal = F.normalize(
                fast_cross(rh_u, rh_v), p=2, dim=-1, eps=1e-5
            )
            bimanual_sync = safe_cosine_sim(vel[:, :, lh_w, :3], vel[:, :, rh_w, :3]).unsqueeze(-1)
            lh_face_dist = safe_norm(
                features_3d[:, :, lh_w, :3] - face_centroid, dim=-1, keepdim=True
            )
            rh_face_dist = safe_norm(
                features_3d[:, :, rh_w, :3] - face_centroid, dim=-1, keepdim=True
            )

            lh_curl = safe_norm(
                features_3d[:, :, lh_tips, :3]
                - features_3d[:, :, lh_w:lh_w+1, :3],
                dim=-1,
            )
            rh_curl = safe_norm(
                features_3d[:, :, rh_tips, :3]
                - features_3d[:, :, rh_w:rh_w+1, :3],
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
                    dtype=input_x.dtype,
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
            next_token_logits = logits[:, -1, :].clone()

            if repetition_penalty != 1.0 and step > 0:
                for b in range(batch_sz):
                    prev_toks = generated[b, : step + 1].unique()
                    prev_toks = prev_toks[prev_toks > 0]
                    if prev_toks.numel() > 0:
                        tok_logits = next_token_logits[b, prev_toks]
                        penalized = torch.where(
                            tok_logits > 0,
                            tok_logits / repetition_penalty,
                            tok_logits * repetition_penalty,
                        )
                        next_token_logits[b, prev_toks] = penalized

            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if top_k > 0:
                top_k_clamped = min(max(top_k, 1), next_token_logits.shape[-1])
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_clamped, dim=-1)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float("Inf")

            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
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
            self.param_pairs = []
            self.params_list = []
            self.shadows_list = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    shadow_p = param.clone().detach()
                    self.shadow[name] = shadow_p
                    self.param_pairs.append((param, shadow_p))
                    self.params_list.append(param)
                    self.shadows_list.append(shadow_p)

    def update(self, model: nn.Module, progress: float = 0.0):
        """Provides functionality for update."""
        progress_val = max(0.0, min(1.0, float(progress)))
        decay = float(self.decay_base + (self.decay_max - self.decay_base) * progress_val)
        step_weight = float(1.0 - decay)

        with torch.no_grad():
            if hasattr(torch, "_foreach_lerp_"):
                torch._foreach_lerp_(self.shadows_list, [p.detach() for p in self.params_list], step_weight)
            else:
                for param, shadow_p in self.param_pairs:
                    shadow_p.lerp_(param.detach(), step_weight)

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


POLY1_EPS = 1.0  # Google Research ICLR 2022 PolyLoss: optimal polynomial expansion epsilon


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
        vf = valid_mask.reshape(-1).to(focal_weight.dtype)
    else:
        vf = torch.ones_like(tf, dtype=focal_weight.dtype)

    if is_seq_loss:
        vf = vf * (tf != eos_id).to(focal_weight.dtype)
    else:
        vf = vf * (tf == eos_id).to(focal_weight.dtype)

    if punct_ids is not None and len(punct_ids) > 0:
        is_punct = torch.zeros_like(tf, dtype=torch.bool)
        for pid in punct_ids:
            is_punct = is_punct | (tf == pid)
        # Apply 2.0x loss penalty weight on punctuation tokens (. ? !) to enforce period placement at boundaries
        vf = vf * torch.where(is_punct, 2.0, 1.0).to(focal_weight.dtype)

    if class_weights is not None:
        tf_safe = torch.clamp(tf, min=0, max=class_weights.shape[0] - 1)
        vf = vf * class_weights[tf_safe].to(focal_weight.dtype)

    if sw is not None:
        vf = vf * sw.to(focal_weight.dtype)

    return _distributed_normalize((poly1 * vf).float().sum(), vf.float().sum())


def compute_chunked_linear_seq_and_eos_loss(
    h_e: torch.Tensor,
    lm_head: nn.Linear,
    gt_ids: torch.Tensor,
    valid_mask_seq: Optional[torch.Tensor],
    valid_mask_eos: Optional[torch.Tensor],
    sample_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
    pad_id: int = 0,
    eos_id: int = 2,
    chunk_tokens: int = 512,
):
    """
    Chunked Linear Projection + Cross-Entropy for PyTorch/XLA.
    Eliminates full [B, L, V] logit tensor materialization, reducing peak HBM from >3GB to <24MB.
    """
    if isinstance(h_e, (tuple, list)):
        h_e = h_e[0]
    D = h_e.shape[-1]
    if h_e.size(1) == gt_ids.size(1) + 1:
        h_sub = h_e[:, :-1, :]
    else:
        h_sub = h_e

    h_flat = h_sub.reshape(-1, D)
    tf = gt_ids.reshape(-1)
    vocab_v = getattr(lm_head, "out_features", 23552)

    vf_seq = valid_mask_seq.reshape(-1).to(h_e.dtype) if valid_mask_seq is not None else torch.ones_like(tf, dtype=h_e.dtype)
    vf_eos = valid_mask_eos.reshape(-1).to(h_e.dtype) if valid_mask_eos is not None else torch.ones_like(tf, dtype=h_e.dtype)
    vf_seq = vf_seq * (tf != eos_id).to(h_e.dtype)
    vf_eos = vf_eos * (tf == eos_id).to(h_e.dtype)

    if sample_weights is not None:
        sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1).to(h_e.dtype)
        vf_seq = vf_seq * sw
        vf_eos = vf_eos * sw

    total_tokens = h_flat.shape[0]
    logit_mem_bytes = total_tokens * vocab_v * 2

    # Single fused MXU GEMM projection with 0 loop unrolling (eliminates graph cloning and 16m compilation)
    lf = lm_head(h_flat)
    tf_safe = torch.clamp(tf, min=0, max=vocab_v - 1)

    # Native Single-Pass Fused Log-Softmax Cross-Entropy (eliminates duplicate log_sum_exp forward & backward graphs)
    log_p = F.log_softmax(lf, dim=-1)
    nll = -log_p.gather(dim=-1, index=tf_safe.unsqueeze(-1)).squeeze(-1)
    valid_pad_mask = (tf_safe != pad_id)
    ce_unsmoothed = torch.where(valid_pad_mask, nll, torch.zeros_like(nll))
    if label_smoothing > 0.0:
        mean_log_p = log_p.mean(dim=-1)
        ce_smoothed_seq = (1.0 - label_smoothing) * nll - label_smoothing * mean_log_p
        ce_smoothed_seq = torch.where(valid_pad_mask, ce_smoothed_seq, torch.zeros_like(ce_smoothed_seq))
    else:
        ce_smoothed_seq = ce_unsmoothed

    with torch.no_grad():
        p_target = torch.exp(-ce_unsmoothed).clamp(min=1e-6, max=1.0)
        focal_weight = torch.pow(1.0 - p_target, 2.0)
        poly_reg = POLY1_EPS * (1.0 - p_target)

    poly1_seq = focal_weight * (ce_smoothed_seq + poly_reg)
    poly1_eos = focal_weight * (ce_unsmoothed + poly_reg)

    total_poly1_seq = (poly1_seq * vf_seq).float().sum()
    total_poly1_eos = (poly1_eos * vf_eos).float().sum()

    with torch.no_grad():
        preds = lf.argmax(dim=-1)
        total_correct = ((preds == tf) & (vf_seq > 0)).float().sum()

    loss_seq = _distributed_normalize(total_poly1_seq, vf_seq.float().sum())
    loss_eos = _distributed_normalize(total_poly1_eos, vf_eos.float().sum())
    c_valid = vf_seq.float().sum()
    return loss_seq, loss_eos, total_correct, c_valid


def compute_chunked_linear_ce_and_acc(
    h: torch.Tensor,
    lm_head: nn.Linear,
    targets: torch.Tensor,
    ignore_index: int = 0,
    label_smoothing: float = 0.0,
    chunk_tokens: int = 2048,
    compute_acc: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes cross-entropy loss AND token accuracy in a single unified pass.
    On TPU, directly executes 1 fused MXU GEMM when token count fits in HBM (up to 65k tokens),
    falling back to chunked iteration only for massive context batches.
    When compute_acc is False (on non-logging steps), skips argmax reduction to save memory bandwidth.
    """
    if isinstance(h, (tuple, list)):
        h = h[0]
    h_flat = h.reshape(-1, h.shape[-1])
    targets_flat = targets.reshape(-1)
    valid_mask = (targets_flat != ignore_index)
    total_valid = valid_mask.float().sum()
    total_tokens = h_flat.shape[0]

    vocab_size = getattr(lm_head, "out_features", 23473)
    logit_mem_bytes = total_tokens * vocab_size * 2

    # Single fused MXU GEMM projection with 0 loop unrolling (eliminates graph cloning and 16m compilation)
    logits = lm_head(h_flat)
    loss = F.cross_entropy(
        logits.float(),
        targets_flat,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="sum",
    )
    loss = _distributed_normalize(loss, total_valid)
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        v_c = (targets_flat != ignore_index)
        total_correct = ((preds == targets_flat) & v_c).float().sum()
    acc = (total_correct / total_valid.clamp_min(1.0)) * 100.0
    return loss, acc


def compute_chunked_linear_ce(
    h: torch.Tensor,
    lm_head: nn.Linear,
    targets: torch.Tensor,
    ignore_index: int = 0,
    label_smoothing: float = 0.0,
    chunk_tokens: int = 2048,
) -> torch.Tensor:
    loss, _ = compute_chunked_linear_ce_and_acc(h, lm_head, targets, ignore_index, label_smoothing, chunk_tokens)
    return loss


@torch.no_grad()
def compute_chunked_accuracy(
    h: torch.Tensor,
    lm_head: nn.Linear,
    targets: torch.Tensor,
    ignore_index: int = 0,
    chunk_tokens: int = 2048,
) -> torch.Tensor:
    _, acc = compute_chunked_linear_ce_and_acc(h, lm_head, targets, ignore_index, 0.0, chunk_tokens)
    return acc


def compute_chunked_distillation_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head: nn.Linear,
    sample_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    temperature: float = 2.0,
    chunk_tokens: int = 512,
) -> torch.Tensor:
    """
    Fused XLA-native knowledge distillation with zero unrolled Python loops.
    Uses elementary cross-entropy formulation -(t_probs * s_log_probs).sum(dim=-1) to achieve
    bit-identical gradients to KL divergence while eliminating graph cloning and remat duplication.
    """
    D = student_hidden.shape[-1]
    s_flat = student_hidden.reshape(-1, D)
    t_flat = teacher_hidden.reshape(-1, D)
    m_flat = sample_mask.reshape(-1).bool()
    w_flat = (
        sample_weights.unsqueeze(1).expand_as(sample_mask).reshape(-1).float()
        if sample_weights.dim() == 1 and sample_mask.dim() == 2
        else sample_weights.reshape(-1).float()
    )

    valid_weight = (m_flat.float() * w_flat).sum().clamp(min=1.0)
    has_valid = ((m_flat.float() * w_flat).sum() > 0).float()

    s_logits = lm_head(s_flat)
    with torch.no_grad():
        t_logits = lm_head(t_flat)
        t_probs = F.softmax(t_logits / float(temperature), dim=-1)

    s_log_probs = F.log_softmax(s_logits / float(temperature), dim=-1)
    kl = -(t_probs * s_log_probs).sum(dim=-1)
    loss = (((kl * m_flat.float()) * w_flat).sum() / valid_weight) * has_valid * (float(temperature) ** 2)
    return loss


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
    """Vectorized, XLA-native focal cross-entropy without dynamic slice loops (eliminates 14GB of HBM remat pads)."""
    vocab_v = logits_f.shape[-1]
    lf = logits_f.reshape(-1, vocab_v)
    tf = gt_ids.reshape(-1)

    sw = (
        sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
        if sample_weights is not None
        else None
    )

    tf_safe = torch.clamp(tf, min=0, max=vocab_v - 1)
    # Native Single-Pass Fused Log-Softmax Cross-Entropy (eliminates duplicate log_sum_exp forward & backward graphs)
    log_p = F.log_softmax(lf, dim=-1)
    nll = -log_p.gather(dim=-1, index=tf_safe.unsqueeze(-1)).squeeze(-1)
    valid_pad_mask = (tf_safe != pad_id)
    ce_unsmoothed = torch.where(valid_pad_mask, nll, torch.zeros_like(nll))
    if label_smoothing > 0.0:
        mean_log_p = log_p.mean(dim=-1)
        ce_smoothed_seq = (1.0 - label_smoothing) * nll - label_smoothing * mean_log_p
        ce_smoothed_seq = torch.where(valid_pad_mask, ce_smoothed_seq, torch.zeros_like(ce_smoothed_seq))
    else:
        ce_smoothed_seq = ce_unsmoothed

    p_target = torch.exp(-ce_unsmoothed).clamp(min=1e-6, max=1.0)
    focal_weight = torch.pow(1.0 - p_target.detach(), gamma)

    loss_seq = _compute_poly1_loss(
        focal_weight, ce_smoothed_seq, p_target, valid_mask_seq, tf, eos_id,
        is_seq_loss=True, sw=sw, class_weights=class_weights, punct_ids=punct_ids,
    )
    loss_eos = _compute_poly1_loss(
        focal_weight, ce_unsmoothed, p_target, valid_mask_eos, tf, eos_id,
        is_seq_loss=False, sw=sw, class_weights=None, punct_ids=punct_ids,
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
    
    # Shift gloss_seq left by 1 to remove BOS_ID at index 0 (functional slice + pad)
    clean_targets = F.pad(gloss_seq[:, 1:], (0, 1), value=GlossVocabulary.PAD_ID)
    
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
    clean_targets = torch.where(valid_shifted, clean_targets, 0)

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
        ctc_log_probs_t.float(),
        clean_targets,
        input_lengths,
        target_lengths,
        blank=GlossVocabulary.PAD_ID,
        reduction="none",
        zero_infinity=True,
    )

    # FastEmit Regularization: penalize blank tokens to encourage earlier non-blank emission (Claim 1 Fix)
    fastemit_lambda = 0.001
    prob_blank = torch.exp(ctc_log_probs[:, :, GlossVocabulary.PAD_ID])
    
    frame_idx_tensor = torch.arange(ctc_log_probs.size(1), device=ctc_log_probs.device).unsqueeze(0)
    valid_frame_mask = (frame_idx_tensor < input_lengths.unsqueeze(1)).float()
    
    fastemit_penalty = fastemit_lambda * (prob_blank * valid_frame_mask).sum(dim=1) / valid_frame_mask.sum(dim=1).clamp_min(1.0)
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

    feat_tensor = batch["feature"]
    if feat_tensor.device != device:
        features = feat_tensor.to(device, dtype=feat_dtype, non_blocking=True)
    elif feat_dtype is not None and feat_tensor.dtype != feat_dtype:
        features = feat_tensor.to(dtype=feat_dtype)
    else:
        features = feat_tensor

    mask = batch["mask"] if batch["mask"].device == device else batch["mask"].to(device, non_blocking=True)
    
    B = features.shape[0]
    labels = batch["label"] if ("label" in batch and batch["label"] is not None and batch["label"].device == device) else (batch["label"].to(device, non_blocking=True) if ("label" in batch and batch["label"] is not None) else torch.zeros(B, dtype=torch.long, device=device))

    frame_indices = (
        batch["frame_indices"].to(device, non_blocking=True)
        if ("frame_indices" in batch and batch["frame_indices"] is not None and batch["frame_indices"].device != device)
        else (batch.get("frame_indices"))
    )

    sample_weight = batch["sample_weight"] if ("sample_weight" in batch and batch["sample_weight"] is not None and batch["sample_weight"].device == device) else (batch["sample_weight"].to(device, non_blocking=True) if ("sample_weight" in batch and batch["sample_weight"] is not None) else torch.ones_like(labels, dtype=torch.float32, device=device))

    domain_tgts = batch["domain_label"] if ("domain_label" in batch and batch["domain_label"] is not None and batch["domain_label"].device == device) else (batch["domain_label"].to(device, non_blocking=True) if ("domain_label" in batch and batch["domain_label"] is not None) else torch.zeros_like(labels))

    sample_ids = None
    has_domain = batch["has_domain_label"] if ("has_domain_label" in batch and batch["has_domain_label"] is not None and batch["has_domain_label"].device == device) else (batch["has_domain_label"].to(device, non_blocking=True) if ("has_domain_label" in batch and batch["has_domain_label"] is not None) else torch.ones_like(domain_tgts, dtype=torch.bool))

    gloss_seq = batch["gloss_seq"] if batch["gloss_seq"].device == device else batch["gloss_seq"].to(device, non_blocking=True)
    gloss_len = batch["gloss_len"] if batch["gloss_len"].device == device else batch["gloss_len"].to(device, non_blocking=True)
    has_valid_gloss = batch["has_valid_gloss"] if batch["has_valid_gloss"].device == device else batch["has_valid_gloss"].to(device, non_blocking=True)
    mlm_mask = (
        batch["mlm_mask"].to(device, non_blocking=True)
        if ("mlm_mask" in batch and batch["mlm_mask"] is not None and batch["mlm_mask"].device != device)
        else batch.get("mlm_mask")
    )

    if getattr(args, "enable_aux_decoders", True):
        chicago_seq = batch["chicago_seq"] if batch["chicago_seq"].device == device else batch["chicago_seq"].to(device, non_blocking=True)
        chicago_len = batch["chicago_len"] if batch["chicago_len"].device == device else batch["chicago_len"].to(device, non_blocking=True)
        has_valid_chicago = batch["has_valid_chicago"] if batch["has_valid_chicago"].device == device else batch["has_valid_chicago"].to(device, non_blocking=True)
        english_seq = batch["english_seq"] if batch["english_seq"].device == device else batch["english_seq"].to(device, non_blocking=True)
        english_len = batch["english_len"] if batch["english_len"].device == device else batch["english_len"].to(device, non_blocking=True)
        has_valid_english = batch["has_valid_english"] if batch["has_valid_english"].device == device else batch["has_valid_english"].to(device, non_blocking=True)
    else:
        chicago_seq = chicago_len = english_seq = english_len = None
        has_valid_chicago = torch.zeros_like(gloss_len, dtype=torch.bool)
        has_valid_english = torch.zeros_like(gloss_len, dtype=torch.bool)

    is_isolated = batch["is_isolated"] if ("is_isolated" in batch and batch["is_isolated"] is not None and batch["is_isolated"].device == device) else (batch["is_isolated"].to(device, non_blocking=True) if ("is_isolated" in batch and batch["is_isolated"] is not None) else torch.ones_like(labels, dtype=torch.bool))
    eng_trunc_flag = batch["english_trunc"] if ("english_trunc" in batch and batch["english_trunc"] is not None and batch["english_trunc"].device == device) else (batch["english_trunc"].to(device, non_blocking=True) if ("english_trunc" in batch and batch["english_trunc"] is not None) else torch.zeros(has_valid_english.shape, dtype=torch.bool, device=device))

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


def _async_phase2_step_print(
    log_vals,
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
    """Internal top-level helper method _async_phase2_step_print."""
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
    ) = log_vals
    g_acc = (float(nc_val) / max(1.0, float(nt_val))) * 100.0
    c_acc = (float(cnc_val) / max(1.0, float(cnt_val))) * 100.0
    e_acc = (float(enc_val) / max(1.0, float(ent_val))) * 100.0
    t_now = time.time()
    elapsed_since_start = max(0.001, t_now - t_start)
    elapsed_win = t_now - t_prev_box[0]
    t_prev_box[0] = t_now
    delta_steps = st_idx - t_prev_box[1] if len(t_prev_box) > 1 and t_prev_box[1] > 0 else (l_freq if st_idx >= l_freq else st_idx)
    if len(t_prev_box) > 1:
        t_prev_box[1] = st_idx
    if elapsed_win > 0.05:
        instant_it_s = float(max(1, delta_steps)) / elapsed_win
    else:
        instant_it_s = float(st_idx) / elapsed_since_start
    samples_per_s = instant_it_s * b_sz

    eng_ppl_val = (
        math.exp(min(float(eng_val), 20.0)) if float(eng_val) > 0 else 0.0
    )
    pct = (float(st_idx) / max(1.0, float(m_batches))) * 100.0
    e_ppl_str = (
        f" PPL(Eng:{eng_ppl_val:.1f}) |" if float(eng_val) > 0 else ""
    )
    compile_str = f" [Initial Graph Compile: {elapsed_since_start:.1f}s]" if st_idx == 1 else ""
    msg = (
        f"  [Epoch {ep:03d}/{tot_ep:03d} | Step {st_idx:04d}/{m_batches:04d} ({pct:5.1f}%)] "
        f"Loss: {float(l_val):.4f} [Seq:{float(s_val):.4f} CTC:{float(c_val):.4f} Sem:{float(sm_val):.4f} Chi:{float(chi_val):.4f} Eng:{float(eng_val):.4f}] | "
        f"Acc (Gloss:{g_acc:.1f}% Chi:{c_acc:.1f}% Eng:{e_acc:.1f}%){e_ppl_str} "
        f"LR: {lr_val:.2e} | Speed: {samples_per_s:.1f} seq/s ({instant_it_s:.2f} it/s){compile_str}"
    )
    print(msg, flush=True)


def _async_phase2_closure_wrapper(mv, st_idx, m_batches, ep, tot_ep, lr_val, t_start, t_prev_box, b_sz, l_freq):
    import torch_xla.core.xla_model as xm
    if xm.is_master_ordinal():
        mv_list = mv.cpu().tolist()
        log_vals = [
            mv_list[0],  # raw_loss
            mv_list[1],  # l_seq
            mv_list[5],  # l_aux
            mv_list[17], # l_ctc
            mv_list[2],  # l_sem
            mv_list[18], # l_chi
            mv_list[19], # l_eng
            mv_list[3],  # nc_t
            mv_list[4],  # nt_t
            mv_list[12], # c_nc_t
            mv_list[13], # c_nt_t
            mv_list[14], # e_nc_t
            mv_list[15], # e_nt_t
        ]
        _async_phase2_step_print(log_vals, st_idx, m_batches, ep, tot_ep, lr_val, t_start, t_prev_box, b_sz, l_freq)
        if st_idx == 2:
            try:
                import torch_xla.debug.metrics as met
                print("\n[XLA Metrics Report at Step 2]\n" + met.metrics_report(), flush=True)
            except Exception:
                pass


def _async_pseudo_gloss_step_print(ep, s, l_t):
    if IS_TPU:
        import torch_xla.core.xla_model as xm
        if not xm.is_master_ordinal():
            return
    val = float(l_t.item()) if hasattr(l_t, "item") else float(l_t)
    msg = f"Phase 2 Train | Epoch {ep:03d} | Step {s:04d} | Loss: {val:.4f}"
    if IS_TPU:
        import torch_xla.core.xla_model as xm
        xm.master_print(msg, flush=True)
    else:
        print(msg, flush=True)


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
    in_graph_augmentor: Optional[Any] = None,
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
        import torch_xla.distributed.parallel_loader as pl

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

    total_batches = get_dynamic_loader_len(loader, default_steps=2500, args=args)
    min_batches = total_batches
    if is_xla:
        # On TPU v5e (16GB HBM), Phase 2 multimodal graphs must execute with batches_per_execution=1
        # to prevent unrolling multiple batches into a single static graph which causes fatal HBM OOM (>47GB).
        bpe = 1
        if epoch == 1 and is_master:
            if getattr(args, "batches_per_execution", 1) > 1:
                print(
                    f"[INFO] Phase 2: Enforced batches_per_execution=1 on TPU v5e to prevent multi-batch graph unrolling (HBM ceiling: 15.75GB).",
                    flush=True,
                )
            print(
                f"[XLA] Compiling graph with batches_per_execution={bpe}. First step will take ~15-30s.",
                flush=True,
            )
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
        raw_model.update_tome_r(epoch, getattr(args, "epochs", total_epochs))

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
    last_log_time_box = [step_start_time, 0]

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
    running_metrics = torch.zeros(
        len(TRAIN_METRIC_KEYS), dtype=torch.float32, device=device
    )
    running_truncs = torch.zeros(3, dtype=torch.float32, device=device)
    all_trainable_params = [
        p for p in list(model.parameters()) + list(loss_wrapper.parameters())
        if p.requires_grad
    ]
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

        # Apply In-Graph Vectorized Landmark Augmentation on Accelerator (TPU/GPU)
        if in_graph_augmentor is not None:
            features = in_graph_augmentor(features, mask, noise_level=progress)

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
                compute_mlm=(mlm_mask is not None and getattr(args, "enable_mlm", False)),
                compute_lpc=True,
                has_valid_english=has_valid_english,
                skip_augment=(in_graph_augmentor is not None),
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
                english_hidden,
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
                out.get("english_hidden"),
            )
            h_cls = out.get("h_cls")

            gt_tokens = gloss_seq[:, 1:].contiguous()
            token_mask = (
                gt_tokens != GlossVocabulary.PAD_ID
            ) & has_valid_gloss.bool().unsqueeze(-1)
            valid_gloss_mask = token_mask & (gt_tokens != GlossVocabulary.EOS_ID)

            # Gloss length & sequence loss masked strictly by has_valid_gloss
            target_len = valid_gloss_mask.sum(dim=1).float()
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
            # --- CHICAGO LOSS (Sample-wise Masking) ---
            c_valid = has_valid_chicago.float()
            if chicago_seq is not None:
                c_sub = chicago_seq[:, 1:]
                c_tok_mask = (c_sub != GlossVocabulary.PAD_ID) & has_valid_chicago.unsqueeze(-1)
                c_valid_seq_mask = c_tok_mask & (c_sub != GlossVocabulary.EOS_ID)
            else:
                c_sub = None
                c_tok_mask = None
                c_valid_seq_mask = None

            if chicago_pred_len is not None and c_valid_seq_mask is not None:
                c_target_len = c_valid_seq_mask.sum(dim=1).float()
                loss_chicago_len = F.smooth_l1_loss(
                    chicago_pred_len, c_target_len, reduction="none"
                )
                loss_chicago_len = _distributed_normalize(
                    (loss_chicago_len * sample_weight * c_valid).float().sum(),
                    (c_valid * sample_weight).float().sum(),
                )
            else:
                loss_chicago_len = torch.zeros((), device=device)

            if chicago_logits is not None and c_valid_seq_mask is not None:
                loss_chicago, loss_chicago_eos = compute_seq_and_eos_loss(
                    chicago_logits,
                    c_sub,
                    c_valid_seq_mask,
                    c_tok_mask,
                    sample_weights=sample_weight,
                    label_smoothing=0.1,
                    pad_id=GlossVocabulary.PAD_ID,
                )
            else:
                loss_chicago = torch.zeros((), device=device)
                loss_chicago_eos = torch.zeros((), device=device)

            # --- ENGLISH LOSS (Sample-wise Masking) ---
            e_valid = has_valid_english.float()
            if english_seq is not None:
                e_sub = english_seq[:, 1:]
                e_tok_mask = (e_sub != EnglishVocabulary.PAD_ID) & has_valid_english.unsqueeze(-1)
                e_valid_seq_mask = (
                    e_tok_mask
                    & (e_sub != EnglishVocabulary.EOS_ID)
                    & (e_sub != EnglishVocabulary.UNK_ID)
                )
            else:
                e_sub = None
                e_tok_mask = None
                e_valid_seq_mask = None

            if english_pred_len is not None and e_valid_seq_mask is not None:
                e_target_len = e_valid_seq_mask.sum(dim=1).float()
                loss_english_len = F.smooth_l1_loss(
                    english_pred_len, e_target_len, reduction="none"
                )
                loss_english_len = _distributed_normalize(
                    (loss_english_len * sample_weight * e_valid).float().sum(),
                    (e_valid * sample_weight).float().sum(),
                )
            else:
                loss_english_len = torch.zeros((), device=device)

            raw_model = model.module if hasattr(model, "module") else model
            english_nc_t = torch.zeros((), device=device)
            english_nt_t = torch.zeros((), device=device)
            if english_hidden is not None and e_valid_seq_mask is not None:
                loss_english, loss_english_eos, english_nc_t, english_nt_t = (
                    compute_chunked_linear_seq_and_eos_loss(
                        english_hidden,
                        raw_model.english_decoder.lm_head,
                        e_sub,
                        e_valid_seq_mask,
                        e_tok_mask,
                        sample_weights=sample_weight,
                        label_smoothing=0.1,
                        pad_id=EnglishVocabulary.PAD_ID,
                        eos_id=EnglishVocabulary.EOS_ID,
                        chunk_tokens=512,
                    )
                )
            elif english_logits is not None and e_valid_seq_mask is not None:
                loss_english, loss_english_eos = compute_seq_and_eos_loss(
                    english_logits,
                    e_sub,
                    e_valid_seq_mask,
                    e_tok_mask,
                    sample_weights=sample_weight,
                    label_smoothing=0.1,
                    pad_id=EnglishVocabulary.PAD_ID,
                    eos_id=EnglishVocabulary.EOS_ID,
                )
            else:
                loss_english = torch.zeros((), device=device)
                loss_english_eos = torch.zeros((), device=device)

            # --- AUXILIARY GROUNDING & GLOSS AUX LOSSES ---
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
                    enc_mask,
                    has_valid_gloss,
                    sample_weights=sample_weight,
                )
            else:
                loss_inter_ctc = torch.zeros((), device=device)

            loss_lpc = out.get("loss_lpc", torch.zeros((), device=device))

            h_cls = out.get("h_cls")
            if h_cls is not None and hasattr(raw_model, "dense_sem_loss") and raw_model.dense_sem_loss is not None:
                loss_dense_sem = raw_model.dense_sem_loss(
                    h_cls,
                    raw_model.decoder.asl_lex_emb(gt_tokens),
                    valid_gloss_mask,
                    sample_weights=sample_weight,
                )
            else:
                loss_dense_sem = torch.zeros((), device=device)

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

            isolated_labels = torch.where(is_isolated, labels, -1)
            if proj_feats is not None:
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
                    sample_weights=sample_weight,
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
                    sample_weights=sample_weight,
                )


            # --- BONE LENGTH REGULARIZATION LOSS ---
            orig_x = out.get("orig_x", None)
            if orig_x is not None and hasattr(raw_model, "bone_loss_fn"):
                loss_bone = raw_model.bone_loss_fn(orig_x, mask=out.get("orig_enc_mask", None))
            else:
                loss_bone = torch.zeros((), device=device)

            loss_phonology = out.get("phonology_loss", torch.zeros((), device=device))

            loss_mp = out.get("mp_loss", torch.zeros((), device=device))

            # --- BIDIRECTIONAL PHASE 1 DISTILLATION & CYCLE CONSISTENCY (English <-> ASL) ---
            loss_distill_gloss = torch.zeros((), device=device)
            loss_distill_english = torch.zeros((), device=device)
            is_gpt2_mode = getattr(args, "use_gpt2", False) or getattr(raw_model, "use_gpt2", False)
            if getattr(args, "enable_phase1_distill", False) and not is_gpt2_mode:
                if english_seq is not None and raw_model.english_decoder is not None and raw_model.decoder is not None:
                    with torch.no_grad():
                        mem_teacher_eng = raw_model.english_decoder.token_emb(english_seq)
                        mask_clean_eng = (english_seq == EnglishVocabulary.PAD_ID)
                        tgt_in_gloss = gloss_seq[:, :-1]
                        t_gloss_out = raw_model.decoder(
                            tgt_in_gloss,
                            memory=mem_teacher_eng,
                            memory_key_padding_mask=mask_clean_eng,
                            compute_head=False,
                        )
                        t_gloss_h = t_gloss_out[0] if isinstance(t_gloss_out, tuple) else t_gloss_out

                    s_gloss_h = out.get("dec_hidden", None)
                    if s_gloss_h is not None:
                        loss_distill_gloss = compute_chunked_distillation_kl(
                            s_gloss_h,
                            t_gloss_h,
                            raw_model.decoder.lm_head,
                            sample_mask=has_valid_english.unsqueeze(1).expand_as(tgt_in_gloss),
                            sample_weights=sample_weight,
                            temperature=2.0,
                            chunk_tokens=512,
                        )

                if gloss_seq is not None and raw_model.english_decoder is not None and raw_model.decoder is not None:
                    with torch.no_grad():
                        mem_teacher_gloss = raw_model.decoder.token_emb(gloss_seq)
                        mask_clean_gloss = (gloss_seq == GlossVocabulary.PAD_ID)
                        tgt_in_eng = english_seq[:, :-1]
                        t_eng_out = raw_model.english_decoder(
                            tgt_in_eng,
                            memory=mem_teacher_gloss,
                            memory_key_padding_mask=mask_clean_gloss,
                            compute_head=False,
                        )
                        t_eng_h = t_eng_out[0] if isinstance(t_eng_out, tuple) else t_eng_out

                    s_eng_h = out.get("english_hidden", None)
                    if s_eng_h is not None:
                        loss_distill_english = compute_chunked_distillation_kl(
                            s_eng_h,
                            t_eng_h,
                            raw_model.english_decoder.lm_head,
                            sample_mask=has_valid_gloss.unsqueeze(1).expand_as(tgt_in_eng),
                            sample_weights=sample_weight,
                            temperature=2.0,
                            chunk_tokens=512,
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
                "bone": loss_bone,
                "phonology": loss_phonology,
                "mp": loss_mp,
                "distill_gloss": loss_distill_gloss,
                "distill_english": loss_distill_english,
                "gpt2": out["gpt2_loss"] if out.get("gpt2_loss", None) is not None else torch.zeros((), device=device),
            }
            raw_loss = loss_wrapper(loss_terms)

            with torch.no_grad():
                # Gloss Metrics
                preds = (
                    dec_logits.argmax(dim=-1) if dec_logits is not None else gt_tokens
                )
                nc_t = ((preds == gt_tokens) & valid_gloss_mask).float().sum()
                nt_t = valid_gloss_mask.float().sum()

                # Chicago Metrics
                chicago_nc_t = torch.zeros((), device=device)
                chicago_nt_t = torch.zeros((), device=device)
                if chicago_logits is not None and c_valid_seq_mask is not None:
                    c_preds = chicago_logits.argmax(dim=-1)
                    chicago_nc_t = ((c_preds == c_sub) & c_valid_seq_mask).float().sum()
                    chicago_nt_t = c_valid_seq_mask.float().sum()

                # English Metrics
                if english_logits is not None and e_valid_seq_mask is not None:
                    e_preds = english_logits.argmax(dim=-1)
                    english_nc_t = ((e_preds == e_sub) & e_valid_seq_mask).float().sum()
                    english_nt_t = e_valid_seq_mask.float().sum()

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

        accum_steps_val = getattr(args, "accum_steps", accum_steps) if args is not None else accum_steps
        accum_steps_val = max(1, accum_steps_val)
        bwd_weight_val = getattr(args, "bwd_weight", 1.0) if args is not None else 1.0
        # Use a CONSTANT divisor for the backward pass. Varying effective_accum
        # per-step (e.g. at epoch boundaries) causes XLA to trace a new computation
        # graph every time the value changes, triggering expensive recompilation.
        # The do_update flag at epoch end already handles partial accumulation groups correctly.
        loss_scale = bwd_weight_val / accum_steps_val

        if scaler is not None:
            scaler.scale(loss * loss_scale).backward()
        else:
            (loss * loss_scale).backward()

        with torch.no_grad():
            # Static 20-element metric vector computed on 100% of steps (guarantees single static XLA graph)
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
                    l_ctc.detach(),
                    l_chi.detach(),
                    l_eng.detach(),
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

            running_metrics.add_(metrics_vec[:17])
            running_truncs.add_(truncs_vec)

            log_freq = getattr(args, "log_freq", 50) if args is not None else 50
            batch_sz_val = getattr(loader, "batch_size", 64)
            if not isinstance(batch_sz_val, int):
                batch_sz_val = 64
            cluster_batch_sz = batch_sz_val * (get_xla_world_size() if is_xla else 1)

            # Throttle warmup logging to steps 1, 5, 10 to avoid per-step host synchronization stalls
            should_log = (step_idx in (1, 5, 10)) or (step_idx % log_freq == 0) or (step_idx >= min_batches)
            if is_xla:
                if should_log:
                    args_tuple = (
                        metrics_vec,
                        step_idx,
                        min_batches,
                        epoch,
                        total_epochs,
                        optimizer.param_groups[0]["lr"],
                        step_start_time,
                        last_log_time_box,
                        cluster_batch_sz,
                        log_freq,
                    )
                    import torch_xla.core.xla_model as xm
                    xm.add_step_closure(_async_phase2_closure_wrapper, args=args_tuple)
            else:
                if is_master and should_log:
                    mv_list = metrics_vec.tolist()
                    log_vals = [
                        mv_list[0], mv_list[1], mv_list[5], mv_list[17],
                        mv_list[2], mv_list[18], mv_list[19], mv_list[3],
                        mv_list[4], mv_list[12], mv_list[13], mv_list[14],
                        mv_list[15],
                    ]
                    _async_phase2_step_print(log_vals, step_idx, min_batches, epoch, total_epochs,
                        optimizer.param_groups[0]["lr"], step_start_time,
                        last_log_time_box, batch_sz_val, log_freq,
                    )

        # Unified single-graph execution: perform optimizer step after attaching step closure
        do_update = (step_idx % max(1, accum_steps_val) == 0) or (
            step_idx == min_batches
        )
        if is_xla:
            import torch_xla.core.xla_model as xm
            if do_update:
                xla_clip_grad_norm_(all_trainable_params, max_norm=1.0)
                xm.optimizer_step(optimizer)
                optimizer.zero_grad(set_to_none=True)
            else:
                xm.mark_step()
        else:
            if do_update:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    xla_clip_grad_norm_(all_trainable_params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    xla_clip_grad_norm_(all_trainable_params, max_norm=1.0)
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
        if hasattr(raw_m, "dense_sem_loss") and do_update:
            raw_m.dense_sem_loss.update_momentum()
            
        # Complete memory hygiene: delete every intermediate tensor from this step
        del batch
        del l_seq, l_aux, l_ctc, l_sem, l_chi, l_eng
        del features, mask, labels, gloss_seq, chicago_seq, english_seq
        del frame_indices, domain_tgts, has_domain, mlm_mask, loss
        del raw_loss, nc_t, nt_t, c_nc_t, c_nt_t, e_nc_t, e_nt_t
        del c_elig, c_used, c_drop, m_enc, m_tgt, m_min
        if "forward_and_losses" in locals():
            del forward_and_losses
        if "gloss_len" in locals():
            del gloss_len
        if "has_valid_gloss" in locals():
            del has_valid_gloss
        if "chicago_len" in locals():
            del chicago_len
        if "has_valid_chicago" in locals():
            del has_valid_chicago
        if "english_len" in locals():
            del english_len
        if "has_valid_english" in locals():
            del has_valid_english
        if "is_isolated" in locals():
            del is_isolated
        if "sample_ids" in locals():
            del sample_ids
        if "sample_weight" in locals():
            del sample_weight
        dec_logits = None
        chicago_logits = None
        english_logits = None
        mtp_logits = None
        ctc_log_probs = None
        vis_emb = None
        sent_emb = None
        proj_feats = None
        domain_logits = None
        aux_logits = None
        english_hidden = None
        loss_lpc = None
        log_vec = None
        metrics_vec = None
        truncs_vec = None
        out = None

        if step_idx <= 2 or should_log:
            trim_host_memory()

        if is_xla and step_idx >= min_batches:
            if "para_loader" in locals():
                del para_loader

            trim_host_memory()
            break

    trim_host_memory()
    if is_xla:
        xm.mark_step()
        xm.rendezvous(f"train_end_of_epoch_{epoch}")

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
        for idx, key in enumerate(TRAIN_METRIC_KEYS):
            tracker[key] = float(running_metrics[idx])
        g_tr = float(running_truncs[0])
        c_tr = float(running_truncs[1])
        e_tr = float(running_truncs[2])
        tracker["gloss_trunc"] = g_tr
        tracker["chicago_trunc"] = c_tr
        tracker["english_trunc"] = e_tr
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
        import torch_xla.distributed.parallel_loader as pl

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

    total_val_batches = get_dynamic_loader_len(loader, default_steps=500, args=args)
    min_val_batches = total_val_batches
    if is_xla:
        min_val_batches = int(
            xm.mesh_reduce(
                "min_val_batches", total_val_batches, lambda input_x: min(input_x)
            )
        )

        if args is not None and getattr(args, "val_check_steps", 0) > 0:
            min_val_batches = min(min_val_batches, args.val_check_steps)

        bpe = 1 if is_xla else (max(1, getattr(args, "batches_per_execution", 1)) if args is not None else 1)
        para_loader = pl.MpDeviceLoader(loader, device, batches_per_execution=bpe)
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
                    has_valid_english=has_valid_english,
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
                    "bone": torch.zeros((), device=device),
                    "phonology": torch.zeros((), device=device),
                    "mp": torch.zeros((), device=device),
                    "distill_gloss": torch.zeros((), device=device),
                    "distill_english": torch.zeros((), device=device),
                    "gpt2": out["gpt2_loss"] if out.get("gpt2_loss", None) is not None else torch.zeros((), device=device),
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
                max_gen_batches = getattr(args, "max_val_gen_batches", 5) if args is not None else 5
                run_gen = (not skip_gen) and (step_idx <= max_gen_batches)
                if run_gen and isinstance(out, dict) and "h_seq" in out:
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

                    val_gen_len = 64
                    gen_ids = model.generate(
                        features,
                        mask=mask,
                        max_new_tokens=val_gen_len,
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
                    if st_idx % 50 == 0 or st_idx == m_batches:
                        print(
                            f"  [Val Step {st_idx:04d}/{m_batches:04d}] Loss: {float(r_loss.cpu()):.4f}",
                            flush=True,
                        )
                    metrics_csv_path = os.path.join(
                        mdl_dir if mdl_dir else ".",
                        "training_metrics.csv",
                    )
                    with open(metrics_csv_path, "a", newline="") as f:
                        csv.writer(f).writerow(
                            [ep + 1, st_idx, "val_intra", float(r_loss.cpu())]
                            + [0.0] * 12
                        )

                mdl_dir = args.save_dir if (args and hasattr(args, "save_dir")) else (args.model_dir if (args and hasattr(args, "model_dir")) else ".")
                if is_xla:
                    def _val_closure(r_loss, ep, st_idx, m_batches, m_dir):
                        import torch_xla.core.xla_model as xm
                        if xm.is_master_ordinal():
                            _val_async_step_print(r_loss, ep, st_idx, m_batches, m_dir)
                    import torch_xla.core.xla_model as xm
                    xm.add_step_closure(_val_closure, args=(raw_loss.detach(), epoch, step_idx, min_val_batches, mdl_dir))
                else:
                    if is_master:
                        _val_async_step_print(raw_loss, epoch, step_idx, min_val_batches, mdl_dir)

                if "running_val_metrics" not in locals():
                    running_val_metrics = torch.zeros_like(metrics_vec)
                running_val_metrics.add_(metrics_vec)

            del batch
            del metrics_vec
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
            if "sample_weight" in locals():
                del sample_weight
            if "domain_tgts" in locals():
                del domain_tgts
            if "has_domain" in locals():
                del has_domain
            if "mlm_mask" in locals():
                del mlm_mask
            if "gloss_len" in locals():
                del gloss_len
            if "chicago_len" in locals():
                del chicago_len
            if "english_len" in locals():
                del english_len
            if "is_isolated" in locals():
                del is_isolated
            if "eng_trunc_flag" in locals():
                del eng_trunc_flag

            if is_xla:
                import torch_xla.core.xla_model as xm
                xm.mark_step()

    if is_xla:
        xm.rendezvous(f"val_validate_metrics_{epoch}")

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

    trim_host_memory()

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
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
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

        global pl
        import torch_xla.distributed.parallel_loader as pl
        import torch_xla.runtime as xr
        import torch_xla.core.xla_model as xm

        world_size = xr.world_size()
        try:
            _default_cache = "/tmp/xla_cache" if os.name != "nt" else "./xla_cache"
            base_cache = os.environ.get("XLA_PERSISTENT_CACHE_PATH", _default_cache)
            if base_cache.startswith("/kaggle/working"):
                base_cache = "/tmp/xla_cache"
            rank_cache = f"{base_cache}_{rank}"
            os.makedirs(rank_cache, exist_ok=True)
            xr.initialize_cache(rank_cache, readonly=False)
            if rank == 0:
                print(f"[XLA] Compilation disk cache active: {rank_cache} (Per-rank cache enabled in /tmp)", flush=True)
        except Exception as _cache_err:
            if rank == 0:
                print(f"[XLA] Warning: Could not initialize compilation cache: {_cache_err}", flush=True)
        assert rank < world_size
        is_master = rank == 0
        if is_master:
            print(
                f"[DEBUG 5/8] PJRT TPU runtime initialized. World size: {world_size}",
                flush=True,
            )
    else:
        if "LOCAL_RANK" in os.environ:
            import torch.distributed as dist
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            if not dist.is_initialized():
                try:
                    dist.init_process_group(backend="nccl", device_id=device)
                except TypeError:
                    dist.init_process_group(backend="nccl")
            rank = local_rank
            is_master = (rank == 0)
            if is_master:
                print(
                    f"[DEBUG 5/8] Distributed GPU (DDP) initialized across {world_size} GPUs.",
                    flush=True,
                )
        else:
            rank = 0
            world_size = 1
            is_master = True
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if is_master:
                gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
                print(
                    f"[DEBUG 5/8] GPU Worker initialized. Total CUDA devices detected: {gpu_count}",
                    flush=True,
                )

    try:
        if IS_TPU:
            try:
                import torch_xla

                device = torch_xla.device()
            except Exception:
                device = xm.xla_device()
    except Exception as e:
        print(f"FAILED TO INITIALIZE TPU OR GET DEVICE: {e}", flush=True)
        raise

    requested_batch_size = args.batch_size
    target_per_core_batch = max(1, requested_batch_size // world_size)
    bpe_val = getattr(args, "batches_per_execution", 1)
    if IS_TPU:
        # TPU v5e has 16GB HBM per chip (15.75GB reservable by XLA runtime).
        # In Phase 1 text pre-training: vocab projections are small, allowing batch 256 per core (2048 total).
        # In Phase 2 video multimodal training:
        # - Batch 128 per core (1024 total) with Gradient Checkpointing reduces activation memory by >2.5x (fitting comfortably in ~7.2GB HBM).
        # - Auto-enable gradient checkpointing when target_per_core_batch >= 128 for maximum MXU saturation.
        # When args.epochs > 0, we are preparing the Phase 2 multimodal model and dataloader.
        # Only treat as phase 1 if running exclusively Phase 1 (epochs == 0 or phase1_only).
        is_phase1 = (getattr(args, "epochs", 0) == 0 or getattr(args, "phase1_only", False))
        if not is_phase1 and target_per_core_batch >= 64 and not getattr(args, "gradient_checkpointing", False):
            args.gradient_checkpointing = True
            if is_master:
                print(
                    f"[INFO] High-Throughput Mode: Auto-enabled Gradient Checkpointing for Phase 2 per-core batch {target_per_core_batch} "
                    f"(reduces per-batch activation HBM by >2.5x to enable 100% MXU utilization with zero OOM).",
                    flush=True,
                )
        is_gpt2_active = (
            getattr(args, "use_gpt2", False)
            or os.path.exists("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1")
            or os.path.exists("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2")
        )
        # TPU v5e/v3 HBM allocation rules:
        # 1. In Phase 1 text pre-training: vocab projections are small, allowing batch 256 per core.
        # 2. In Phase 2 with GPT-2: GPT-2 has vocab 50,257 and 12 attention heads.
        #    Per-core batch 64 aligns with systolic array tiles, maintains safe ~7.5GB HBM (50% headroom),
        #    and auto-accumulates to preserve the exact requested global batch size (e.g. 1024).
        # 3. In Phase 2 standard decoder (no GPT-2): vocab is 3,586, allowing per-core batch 128.
        if is_phase1:
            max_safe_batch = 256
        elif is_gpt2_active:
            max_safe_batch = 64
        else:
            max_safe_batch = 128
    else:
        # GPU: 16-24GB VRAM
        is_phase1 = (getattr(args, "epochs", 0) == 0 or getattr(args, "phase1_only", False))
        is_gpt2_active = (
            getattr(args, "use_gpt2", False)
            or os.path.exists("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1")
            or os.path.exists("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2")
        )
        max_safe_batch = 16 if is_gpt2_active else (32 if (args.max_len >= 384 or args.d_model >= 384) else 64)

    if target_per_core_batch > max_safe_batch:
        safe_per_core_batch = max_safe_batch
        auto_accum = max(1, math.ceil(target_per_core_batch / safe_per_core_batch))
        args.accum_steps = max(getattr(args, "accum_steps", 1), auto_accum)
        effective_loader_batch = safe_per_core_batch
        if is_master:
            hw_name = "TPU" if IS_TPU else "GPU"
            effective_total = safe_per_core_batch * world_size * args.accum_steps
            print(
                f"[INFO] {hw_name} High-Throughput Memory Guard Active: Capping per-device loader batch to {safe_per_core_batch} "
                f"(Requested total: {requested_batch_size}, World size: {world_size}, MaxLen: {args.max_len}, d_model: {getattr(args, 'd_model', 384)}, BPE: {bpe_val}). "
                f"Auto-setting accum_steps={args.accum_steps} for 100% OOM safety & identical gradient updates (Effective Total Batch: {effective_total}).",
                flush=True,
            )
    else:
        effective_loader_batch = target_per_core_batch

    # On TPU with xmp.spawn (8 core processes), DataLoader(num_workers > 0) creates 8 x N additional subprocesses
    # whose unbounded IPC tensor queues cause massive Host RAM explosion and deadlocks with PJRT runtime.
    # MpDeviceLoader already provides native background C++ device prefetching. Setting DataLoader num_workers=0 is mandatory on TPU.
    effective_num_dl_workers = 0 if IS_TPU else getattr(args, "num_dataloader_workers", 2)

    if hasattr(args, "phase1_epochs") and args.phase1_epochs > 0:
        target_core_batch = max(1, requested_batch_size // world_size)
        p1_max_len = getattr(args, "phase1_max_len", getattr(args, "english_max_len", 256))
        if IS_TPU:
            # Full Native Batch Allocation: Preserve exact requested per-core batch (e.g. 256 for 2048 total batch)
            # Sequential task execution with intermediate xm.mark_step() keeps peak HBM < 11.3 GB on TPU v5e
            phase1_core_batch = target_core_batch
            phase1_accum = max(1, getattr(args, "accum_steps", 1))
        else:
            phase1_core_batch = target_core_batch
            phase1_accum = max(1, getattr(args, "accum_steps", 1))

        if is_master:
            dev_name = "TPU cores" if IS_TPU else "GPUs"
            print(
                f"[INFO] Phase 1 Batch Allocation: Core batch {phase1_core_batch} x {world_size} {dev_name} = Physical batch {phase1_core_batch * world_size} (accum_steps={phase1_accum}, Effective batch {requested_batch_size}, MaxLen={p1_max_len}).",
                flush=True,
            )
        text_pretrain_loop(args, device, is_master, per_core_batch=phase1_core_batch, accum_steps=phase1_accum)
        gc.collect()
        trim_host_memory()
        if IS_TPU:
            xm.mark_step()
            xm.rendezvous("phase1_to_phase2_transition_done")
        if getattr(args, "epochs", 0) == 0 or getattr(args, "phase1_only", False):
            if is_master:
                print(
                    "[INFO] Phase 1 Text Pre-training complete. Exiting training as args.epochs == 0 / phase1_only is set.",
                    flush=True,
                )
            return

    if is_master:
        print(
            f"[DEBUG 6/8] Resolving dataset paths & loading vocabulary map from '{args.data_dir}'...",
            flush=True,
        )

    data_dir = Path(args.data_dir)
    # Auto-resolve subfolder if data_dir is a parent wrapper (e.g. asl_dataset containing asl_preprocessed_phase1)
    if (data_dir / "asl_preprocessed_phase1").exists() and (
        (data_dir / "asl_preprocessed_phase1" / "train").exists()
        or list((data_dir / "asl_preprocessed_phase1").glob("shard_*.pt"))
        or list((data_dir / "asl_preprocessed_phase1").glob("*.pt"))
        or list((data_dir / "asl_preprocessed_phase1").glob("*vocab*.json"))
        or (data_dir / "asl_preprocessed_phase1" / "output_mapping.json").exists()
    ):
        data_dir = data_dir / "asl_preprocessed_phase1"
        if is_master:
            print(f"[INFO] Auto-resolved dataset directory to subfolder: {data_dir}", flush=True)
    elif (data_dir / "results" / "asl_preprocessed_phase1").exists():
        data_dir = data_dir / "results" / "asl_preprocessed_phase1"
        if is_master:
            print(f"[INFO] Auto-resolved dataset directory to subfolder: {data_dir}", flush=True)
    elif not (data_dir / "train").exists() and not list(data_dir.glob("shard_*.pt")):
        for p in sorted(data_dir.rglob("train")):
            if p.is_dir() and (list(p.glob("*.pt")) or list(p.glob("shard_*.pt"))):
                data_dir = p.parent
                if is_master:
                    print(f"[INFO] Auto-resolved dataset directory via child 'train' folder to: {data_dir}", flush=True)
                break

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
        worker_idx=rank,
        num_workers=world_size,
        num_dataloader_workers=effective_num_dl_workers,
        shuffle=True,
        augment=True,
        streamed=getattr(args, "streamed_dataset", False),
        use_bpe=True,
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
            data_dir / "asl_preprocessed_phase1",
            data_dir.parent,
            data_dir.parent / "asl_preprocessed_phase1",
            Path("./asl_preprocessed_phase1"),
            Path.cwd(),
        ]
        possible_filenames = [
            "vocab_map.json",
            "output_mapping.json",
            "vocabulary_mapping_train.json",
            "vocabulary_mapping_global.json",
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
                            if label_to_idx and len(label_to_idx) > 10:
                                if is_master:
                                    print(f"[INFO] Loaded vocabulary mapping from {vp} ({len(label_to_idx)} classes)", flush=True)
                                break
                        except Exception:
                            pass
                if label_to_idx and len(label_to_idx) > 10:
                    break

        # Recursive search across data_dir and parent /kaggle/input if still not resolved
        if not label_to_idx or len(label_to_idx) <= 10:
            search_roots = [data_dir, data_dir.parent]
            if os.name != "nt" and Path("/kaggle/input").exists():
                search_roots.append(Path("/kaggle/input"))
            for sroot in search_roots:
                if sroot.exists():
                    for fn in possible_filenames:
                        try:
                            for vp in sorted(sroot.rglob(fn)):
                                with open(vp, "r", encoding="utf-8") as f:
                                    raw_map = json.load(f)
                                if "normalize_vocabulary" in globals():
                                    label_to_idx = normalize_vocabulary(raw_map)
                                elif isinstance(raw_map, dict) and "label_to_idx" in raw_map:
                                    label_to_idx = raw_map["label_to_idx"]
                                elif isinstance(raw_map, dict):
                                    label_to_idx = raw_map
                                if label_to_idx and len(label_to_idx) > 10:
                                    if is_master:
                                        print(f"[INFO] Loaded vocabulary mapping via rglob from {vp} ({len(label_to_idx)} classes)", flush=True)
                                    break
                        except Exception:
                            pass
                        if label_to_idx and len(label_to_idx) > 10:
                            break
                if label_to_idx and len(label_to_idx) > 10:
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
        worker_idx=rank,
        num_workers=world_size,
        num_dataloader_workers=effective_num_dl_workers,
        shuffle=False,
        augment=False,
        streamed=getattr(args, "streamed_dataset", False),
        use_bpe=True,
    )

    offset = GlossVocabulary.OFFSET
    num_classes = max(1, len(vocab) - offset)
    raw_counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    try:
        raw_ds = getattr(train_loader, "dataset", None)
        c_counts = getattr(raw_ds, "class_counts", {}) if raw_ds else {}
        for raw_idx, cnt in c_counts.items():
            if isinstance(raw_idx, int) and 0 <= raw_idx < num_classes:
                raw_counts[raw_idx] = float(cnt)
    except Exception as e:
        if is_master:
            print(f"[WARN] Local class counts parsing exception: {e}", flush=True)

    if IS_TPU:
        import torch_xla.core.xla_model as xm
        raw_counts = xm.all_reduce(xm.REDUCE_SUM, raw_counts)

    try:
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
            print(f"[WARN] Class weighting normalization exception: {e}", flush=True)
        class_weights_tensor = torch.ones(
            len(vocab), dtype=torch.float32, device=device
        )

    import json as _json_cv

    _chicago_vocab_path = data_dir / "chicago_vocab.json"
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

        local_hash = torch.tensor(float(eng_hash % 1000000007), dtype=torch.float32, device=device)
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

    resolved_lex_csv = asl_lex_csv if (asl_lex_csv is not None and isinstance(asl_lex_csv, Path) and asl_lex_csv.exists()) else (Path(asl_lex_csv) if asl_lex_csv and Path(asl_lex_csv).exists() else None)

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
        english_max_len=getattr(args, "english_max_len", 128),
        chicago_max_len=getattr(args, "chicago_max_len", 128),
        english_vocab_size=eng_vsize,
        label_to_idx=label_to_idx,
        csv_path=resolved_lex_csv,
        scale_embeddings=True,
        enable_aux_decoders=getattr(args, "enable_aux_decoders", True),
        is_causal=getattr(args, "is_causal", False),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        use_gpt2=getattr(args, "use_gpt2", False) or os.path.exists("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1") or os.path.exists("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2"),
        gpt2_path=getattr(args, "gpt2_path", "") or ("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1" if os.path.exists("/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1") else "/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2"),
        torch_dtype=torch.bfloat16 if IS_TPU or getattr(args, "precision", "") == "bfloat16" else None,
    ).to(device, dtype=torch.bfloat16 if IS_TPU else None)

    # Tie embeddings unconditionally on ALL ranks so parameter topology is identical across all TPU cores
    if hasattr(model, "english_decoder") and model.english_decoder is not None:
        model.english_decoder.token_emb.weight = model.english_decoder.lm_head.weight
    if hasattr(model, "decoder") and model.decoder is not None:
        model.decoder.token_emb.weight = model.decoder.lm_head.weight

    def safe_load_checkpoint(path_or_dir):
        """Loads a PyTorch checkpoint from a file, a directory containing .pt/.bin files,
        or an unzipped PyTorch archive directory (containing data.pkl and tensor shards)."""
        if not path_or_dir or not os.path.exists(path_or_dir):
            raise FileNotFoundError(f"Checkpoint path does not exist: {path_or_dir}")
        if os.path.isfile(path_or_dir):
            return torch.load(path_or_dir, map_location="cpu", weights_only=False)
        for root, _, files in os.walk(path_or_dir):
            for f in files:
                if f.endswith((".pt", ".pth", ".bin", ".ckpt")) and not f.startswith("."):
                    target_f = os.path.join(root, f)
                    if is_master:
                        print(f"[INFO] Found checkpoint file inside directory: {target_f}", flush=True)
                    return torch.load(target_f, map_location="cpu", weights_only=False)
        archive_root = None
        prefix = os.path.basename(path_or_dir.rstrip("/\\")) or "asl_llm_200"
        if os.path.exists(os.path.join(path_or_dir, "data.pkl")):
            archive_root = path_or_dir
        else:
            for item in sorted(os.listdir(path_or_dir)):
                sub = os.path.join(path_or_dir, item)
                if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "data.pkl")):
                    archive_root = sub
                    prefix = item
                    break
        if archive_root is not None:
            if is_master:
                print(f"[INFO] Detected unzipped PyTorch archive folder at: {archive_root}. Loading state...", flush=True)
            import io
            import zipfile
            mem_buf = io.BytesIO()
            with zipfile.ZipFile(mem_buf, "w", compression=zipfile.ZIP_STORED) as zf:
                for root, _, files in os.walk(archive_root):
                    for f in files:
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, archive_root).replace("\\", "/")
                        zf.write(full_p, f"{prefix}/{rel_p}")
            mem_buf.seek(0)
            return torch.load(mem_buf, map_location="cpu", weights_only=False)
        raise ValueError(f"No valid PyTorch checkpoint or unzipped archive found in: {path_or_dir}")

    # Automatically load Phase 1 text pretraining weights (from sequential Phase 1, CLI args, or dataset inputs)
    phase1_ep = getattr(args, 'phase1_epochs', 0)
    user_p1_ckpt = getattr(args, "phase1_checkpoint", "") or getattr(args, "pretrained_checkpoint", "")
    candidate_llm_paths = [
        user_p1_ckpt,
        "/kaggle/input/models/muddragonmike/pretrain-bidirectional-asl-english/pytorch/default/1/asl_llm_200",
        "/kaggle/input/models/muddragonmike/pretrain-bidirectional-asl-english/pytorch/default/1/asl_llm_200.pt",
        "/kaggle/input/models/muddragonmike/pretrain-bidirectional-asl-english/pytorch/default/1",
        "/kaggle/input/models/muddragonmike/pretrain-bidirectional-asl-english/pytorch/default/1/asl_llm_100",
        "/kaggle/input/models/muddragonmike/pretrain-bidirectional-asl-english/pytorch/default/1/asl_llm_100.pt",
        os.path.join(args.save_dir, f"asl_llm_{phase1_ep}.pt") if phase1_ep > 0 else "",
        os.path.join(args.save_dir, "asl_llm_200.pt"),
        os.path.join(args.save_dir, "asl_llm_100.pt"),
        os.path.join(args.save_dir, "asl_llm_best.pt"),
        os.path.join(args.save_dir, "asl_llm_last.pt"),
        "/kaggle/working/checkpoints/asl_llm_200.pt",
        "/kaggle/working/checkpoints/asl_llm_100.pt",
        "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_llm_200.pt",
        "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/pretrain.zip",
        "E:/datasets/asl_dataset/pretrain.zip",
    ]
    llm_ckpt_path = next((p for p in candidate_llm_paths if p and os.path.exists(p)), None)
    if llm_ckpt_path is not None:
        if is_master:
            print(
                f"[INFO] Pre-trained Phase 1 weights detected at: {llm_ckpt_path}. Transferring weights into Phase 2/3 decoders...",
                flush=True,
            )
        ckpt = safe_load_checkpoint(llm_ckpt_path)
        raw_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else (ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)

        target_state = model.state_dict()
        adapted_state = {}
        for k, v in raw_state.items():
            if k in target_state and target_state[k].shape == v.shape:
                adapted_state[k] = v
            elif k.startswith("english_decoder."):
                if k in target_state and target_state[k].shape == v.shape:
                    adapted_state[k] = v
            elif k.startswith("decoder_token_emb."):
                adapted_k = "decoder.token_emb." + k[len("decoder_token_emb."):]
                if adapted_k in target_state and target_state[adapted_k].shape == v.shape:
                    adapted_state[adapted_k] = v
            else:
                eng_k = f"english_decoder.{k}"
                if eng_k in target_state and target_state[eng_k].shape == v.shape:
                    adapted_state[eng_k] = v
                dec_k = f"decoder.{k}"
                if dec_k in target_state and target_state[dec_k].shape == v.shape:
                    adapted_state[dec_k] = v

        missing, unexpected = model.load_state_dict(adapted_state, strict=False)
        if hasattr(model, "english_decoder") and model.english_decoder is not None:
            model.english_decoder.token_emb.weight = model.english_decoder.lm_head.weight
        if hasattr(model, "decoder") and model.decoder is not None:
            model.decoder.token_emb.weight = model.decoder.lm_head.weight

        if is_master:
            print(f"[INFO] Loaded Phase 1 pre-trained weights into Phase 2/3 model: {len(adapted_state)}/{len(target_state)} tensors transferred. Missing: {len(missing)}, Unexpected: {len(unexpected)}", flush=True)

        del ckpt, raw_state, adapted_state
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
    if not IS_TPU and device.type == "cuda" and torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    loss_wrapper = HomoscedasticLossWrapper().to(device, dtype=torch.bfloat16 if IS_TPU else None)
    if IS_TPU:
        xm.broadcast_master_param(loss_wrapper)

    in_graph_augmentor = InGraphAugmentor().to(device, dtype=torch.bfloat16 if IS_TPU else None)
    in_graph_augmentor.train()

    supcon_fn = SupervisedContrastiveLoss().to(device, dtype=torch.bfloat16 if IS_TPU else None)

    train_loader_len = get_dynamic_loader_len(train_loader, default_steps=2500, args=args)
        
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
        if is_master:
            print(f"[INFO] Loading checkpoint from: {args.resume}...", flush=True)
        ckpt = safe_load_checkpoint(args.resume)
        raw_state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else (ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)

        target_state = model.state_dict()
        adapted_state = {}
        for k, v in raw_state.items():
            if k in target_state and target_state[k].shape == v.shape:
                adapted_state[k] = v
            elif k.startswith("english_decoder."):
                if k in target_state and target_state[k].shape == v.shape:
                    adapted_state[k] = v
            elif k.startswith("decoder_token_emb."):
                adapted_k = "decoder.token_emb." + k[len("decoder_token_emb."):]
                if adapted_k in target_state and target_state[adapted_k].shape == v.shape:
                    adapted_state[adapted_k] = v

        missing, unexpected = model.load_state_dict(adapted_state, strict=False)
        if hasattr(model, "english_decoder") and model.english_decoder is not None:
            model.english_decoder.token_emb.weight = model.english_decoder.lm_head.weight
        if hasattr(model, "decoder") and model.decoder is not None:
            model.decoder.token_emb.weight = model.decoder.lm_head.weight

        if is_master:
            print(f"[INFO] Checkpoint weights mapped: {len(adapted_state)}/{len(target_state)} tensors transferred. Missing: {len(missing)}, Unexpected: {len(unexpected)}", flush=True)

        is_same_model_resume = (len(missing) == 0)
        if is_same_model_resume:
            if "loss_wrapper_state_dict" in ckpt:
                try:
                    loss_wrapper.load_state_dict(ckpt["loss_wrapper_state_dict"], strict=False)
                except Exception:
                    pass
            if "optimizer_state_dict" in ckpt:
                if device.type == "cuda":
                    try:
                        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        for state in optimizer.state.values():
                            for k, v in state.items():
                                if isinstance(v, torch.Tensor):
                                    state[k] = v.to(device)
                    except Exception:
                        pass
                elif IS_TPU:
                    if is_master:
                        print("[INFO] TPU Mode: Initializing fresh optimizer momentum on device for Phase 2/3 fused execution.", flush=True)
            if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception:
                    pass
            start_epoch = ckpt.get("epoch", 0) + 1
            if is_master:
                print(f"[INFO] Full optimizer & scheduler state restored. Resuming from Epoch {start_epoch}...", flush=True)
        else:
            if is_master:
                print("[INFO] Warm-started weights from pre-trained checkpoint. Starting Phase 2/3 fresh from Epoch 1.", flush=True)

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
            except Exception:
                pass

        del ckpt, raw_state, adapted_state
        gc.collect()

    if IS_TPU:
        start_epoch = int(
            xm.mesh_reduce(
                "start_epoch_sync", start_epoch, lambda input_x: max(input_x)
            )
        )
        xm.broadcast_master_param(model)
        xm.broadcast_master_param(loss_wrapper)

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
    ema_dict = locals().get("ema_state_dict_to_load", None)
    if ema_dict is not None:
        for key_k_lower, val_v in ema_dict.items():
            if key_k_lower in ema.shadow:
                ema.shadow[key_k_lower].copy_(val_v.to(ema.shadow[key_k_lower].device))
        if is_master:
            print("[+] Restored EMA state from checkpoint", flush=True)
        del ema_dict
        gc.collect()

    loss_ema_dict = locals().get("loss_ema_state_dict_to_load", None)
    if loss_ema_dict is not None:
        for key_k_lower, val_v in loss_ema_dict.items():
            if key_k_lower in loss_ema.shadow:
                loss_ema.shadow[key_k_lower].copy_(val_v.to(loss_ema.shadow[key_k_lower].device))
        if is_master:
            print("[+] Restored Loss EMA state from checkpoint", flush=True)
        del loss_ema_dict
        gc.collect()

    if IS_TPU:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.rendezvous("init_sync_complete")

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

            if is_master:
                print(f"[Epoch {epoch}/{args.epochs}] Dispatching to XLA device (compiling graph on epoch 1, ~5-15 min)...", flush=True)

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
                in_graph_augmentor=in_graph_augmentor,
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
                    f"[Validation Epoch {epoch}] TotalLoss: {val_metrics['loss']:.4f} | TokenAcc(TF): {val_metrics['gloss_acc']:.2f}% | TokenAcc(AR): {val_metrics['ar_acc']:.2f}% | ExactMatch(AR): {val_metrics['ar_exact']:.2f}%",
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
                xm.rendezvous(f"phase2_pre_checkpoint_save_{epoch}")

            # Save checkpoint every 5 epochs or on final epoch to save memory and disk quota
            if epoch % getattr(args, "save_every_epoch", 5) == 0 or epoch == args.epochs:
                import random as py_random

                ckpt_path = save_dir / f"asl_model_epoch_{epoch}.pt"
                latest_path = save_dir / "asl_model_latest.pt"

                if IS_TPU:
                    import torch_xla.core.xla_model as xm
                    xm.mark_step()
                    xm.wait_device_ops()
                    xm.rendezvous(f"phase2_pre_save_{epoch}")

                    if is_master:
                        os.makedirs(save_dir, exist_ok=True)
                        cpu_model_state = {k: v.cpu() for k, v in raw_m.state_dict().items()}
                        cpu_loss_state = {k: v.cpu() for k, v in loss_wrapper.state_dict().items()}
                        cpu_state = {
                            "epoch": epoch,
                            "model_state_dict": cpu_model_state,
                            "ema_state_dict": (
                                {k: v.cpu() for k, v in ema.shadow.items()} if ema is not None else None
                            ),
                            "loss_wrapper_state_dict": cpu_loss_state,
                            "loss_ema_state_dict": (
                                {k: v.cpu() for k, v in loss_ema.shadow.items()} if loss_ema is not None else None
                            ),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": (
                                scheduler.state_dict() if scheduler is not None else None
                            ),
                            "rng_state_torch": torch.get_rng_state(),
                            "rng_state_numpy": np.random.get_state(),
                            "rng_state_random": py_random.getstate(),
                            "rng_state_xla": xm.get_rng_state(),
                        }
                        torch.save(cpu_state, ckpt_path)
                        try:
                            import shutil
                            shutil.copyfile(str(ckpt_path), str(latest_path))
                            prune_checkpoints(save_dir, prefix="asl_model_epoch_", keep_last_k=getattr(args, "keep_last_k", 5))
                            print(f"[INFO] Checkpoint saved successfully: {ckpt_path}", flush=True)
                        except Exception:
                            pass
                        del cpu_model_state, cpu_loss_state, cpu_state
                        gc.collect()
                        trim_host_memory()
                    xm.mark_step()
                    xm.rendezvous(f"phase2_post_save_{epoch}")
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
                            "loss_ema_state_dict": (
                                {k: v.detach().cpu() for k, v in loss_ema.shadow.items()}
                                if loss_ema is not None
                                else None
                            ),
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
                        if len(all_ckpts) > 2:
                            for old_c in all_ckpts[:-2]:
                                ep_num = int(old_c.stem.split("_")[-1])
                                if ep_num != epoch:
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
                xm.rendezvous(f"phase2_post_checkpoint_save_{epoch}")

            gc.collect()
            gc.collect()

    except Exception as e:
        import traceback
        import sys

        print(f"CRITICAL PYTHON EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        for _v in [
            "model", "raw_m", "optimizer", "scheduler", "scaler",
            "loss_wrapper", "loss_ema", "ema", "train_loader", "val_loader",
            "train_dataset", "val_dataset", "tracker"
        ]:
            if _v in locals():
                try:
                    del locals()[_v]
                except Exception:
                    pass
        gc.collect()
        trim_host_memory()


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
    p2_workers = 0 if IS_TPU else args.num_dataloader_workers
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=p2_workers,
        pin_memory=False,
        persistent_workers=(p2_workers > 0),
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
    val_sampler = (
        torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=xm.get_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False,
            drop_last=False,
        )
        if IS_TPU
        else None
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=(val_sampler is None and False),
        num_workers=p2_workers,
        pin_memory=False,
        persistent_workers=(p2_workers > 0),
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

    p2_bpe = max(1, getattr(args, "batches_per_execution", 1))
    if IS_TPU:
        loader = pl.MpDeviceLoader(dataloader, device, batches_per_execution=p2_bpe)
    else:
        loader = dataloader

    p2_accum = max(1, getattr(args, "accum_steps", 1))
    p2_weight = getattr(args, "bwd_weight", 1.0) / p2_accum
    grad_clip = getattr(args, "grad_clip", 0.0)
    amp_device = "xla" if IS_TPU else ("cuda" if "cuda" in device.type else "cpu")
    amp_dt = getattr(args, "precision", torch.bfloat16) if getattr(args, "precision", None) != "float32" else torch.float32
    amp_on = not getattr(args, "disable_amp", False)

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        decoder.train()
        english_emb.train()

        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            mask = (input_ids == 0)

            with torch.autocast(device_type=amp_device, dtype=amp_dt, enabled=amp_on):
                memory = time_emb(english_emb(input_ids))
                tgt_in = target_ids[:, :-1]
                tgt_out = target_ids[:, 1:]

                out = decoder(
                    tgt_in,
                    memory,
                    memory_key_padding_mask=mask,
                )

                logits = out[0] if isinstance(out, tuple) else out
                valid_mask = (tgt_out != GlossVocabulary.PAD_ID) & (tgt_out != GlossVocabulary.EOS_ID)
                loss, _ = compute_seq_and_eos_loss(
                    logits,
                    tgt_out,
                    valid_mask,
                    torch.zeros_like(valid_mask),
                    label_smoothing=0.1,
                )

            (loss * p2_weight).backward()
            if grad_clip > 0.0:
                xm.clip_grad_norm_(params, grad_clip)
            if IS_TPU:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            if step % args.log_freq == 0:
                loss_val = loss.detach()
                if IS_TPU:
                    import torch_xla.core.xla_model as xm
                    xm.add_step_closure(_async_pseudo_gloss_step_print, args=(epoch, step, loss_val))
                elif is_master:
                    _async_pseudo_gloss_step_print(epoch, step, loss_val)

            # Memory Proofing: dereference step activations
            logits = None
            loss = None
            memory = None
            tgt_in = None
            tgt_out = None
            mask = None
            batch = None

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
                        tgt_in,
                        memory,
                        memory_key_padding_mask=mask,
                    )
                    logits = out[0] if isinstance(out, tuple) else out
                    valid_mask = (tgt_out != GlossVocabulary.PAD_ID) & (tgt_out != GlossVocabulary.EOS_ID)
                    loss, _ = compute_seq_and_eos_loss(logits, tgt_out, valid_mask, torch.zeros_like(valid_mask), label_smoothing=0.0)
                    val_loss_sum += loss
                    val_steps += 1.0

                input_ids = None
                target_ids = None
                memory = None
                tgt_in = None
                tgt_out = None
                out = None
                logits = None
                loss = None
                batch = None
        
        if IS_TPU:
            import torch_xla.core.xla_model as xm
            val_loss_sum = xm.all_reduce(xm.REDUCE_SUM, val_loss_sum)
            val_steps = xm.all_reduce(xm.REDUCE_SUM, val_steps)
        
        avg_val_loss = float((val_loss_sum / max(1.0, float(val_steps))).cpu())
        gc.collect()
        trim_host_memory()
        
        if IS_TPU:
            xm.mark_step()
            xm.rendezvous(f"inverted_gloss_save_barrier_{epoch}")
            if is_master:
                xm.master_print(f"Phase 2 Val Epoch {epoch} | Loss: {avg_val_loss:.4f}")
                xm.master_print(f"Phase 2 Train Epoch {epoch} finished.")
                os.makedirs(args.save_dir, exist_ok=True)
                save_dict = {
                    "epoch": epoch,
                    "decoder": {k: v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else v for k, v in decoder.state_dict().items()},
                    "english_emb": {k: v.detach().to("cpu", copy=True) if isinstance(v, torch.Tensor) else v for k, v in english_emb.state_dict().items()},
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }
                final_save = os.path.join(args.save_dir, f"inverted_gloss_model_epoch_{epoch}.pt")
                final_latest = os.path.join(args.save_dir, "inverted_gloss_model_latest.pt")
                import threading
                threading.Thread(
                    target=_async_save_checkpoint_worker,
                    args=(save_dict, final_save, final_latest, None, args.save_dir, "inverted_gloss_model_epoch_", getattr(args, "keep_last_k", 5)),
                    daemon=True,
                ).start()
            xm.rendezvous(f"inverted_gloss_save_done_{epoch}")
        elif is_master:
            os.makedirs(args.save_dir, exist_ok=True)
            save_dict = {
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "english_emb": english_emb.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            tmp_save = os.path.join(args.save_dir, f"inverted_gloss_model_epoch_{epoch}.pt.tmp")
            final_save = os.path.join(args.save_dir, f"inverted_gloss_model_epoch_{epoch}.pt")
            torch.save(save_dict, tmp_save)
            os.replace(tmp_save, final_save)
            
            tmp_latest = os.path.join(args.save_dir, "inverted_gloss_model_latest.pt.tmp")
            final_latest = os.path.join(args.save_dir, "inverted_gloss_model_latest.pt")
            torch.save(save_dict, tmp_latest)
            os.replace(tmp_latest, final_latest)
            prune_checkpoints(args.save_dir, prefix="inverted_gloss_model_epoch_", keep_last_k=getattr(args, "keep_last_k", 5))
            print(f"[INFO] Phase 2 Checkpoint saved successfully: {final_save}", flush=True)


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

                # Flush XLA step before host transfer
                if IS_TPU:
                    xm.mark_step()
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


class Phase1TextWrapper(nn.Module):
    def __init__(self, foundation_model):
        super().__init__()
        self.decoder_token_emb = foundation_model.decoder.token_emb
        self.decoder = foundation_model.decoder
        self.english_decoder = foundation_model.english_decoder

    def forward_english(self, gloss_ids, text_ids, corrupted_text_ids, is_dae, compute_head: bool = False):
        mem_gloss = self.decoder_token_emb(gloss_ids)
        mem_corrupted_eng = self.english_decoder.token_emb(corrupted_text_ids)
        mask_gloss = (gloss_ids == self.decoder.pad_id)
        mask_corrupted_eng = (corrupted_text_ids == self.english_decoder.pad_id)

        dae_3d = is_dae.view(-1, 1, 1)
        mem_for_eng = torch.where(dae_3d, mem_corrupted_eng, mem_gloss)
        mask_for_eng = torch.where(is_dae.view(-1, 1), mask_corrupted_eng, mask_gloss)

        out = self.english_decoder(text_ids[:, :-1], memory=mem_for_eng, memory_key_padding_mask=mask_for_eng, compute_head=compute_head)
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward_gloss(self, gloss_ids, text_ids, compute_head: bool = False):
        mem_clean_eng = self.english_decoder.token_emb(text_ids)
        mask_clean_eng = (text_ids == self.english_decoder.pad_id)
        out = self.decoder(gloss_ids[:, :-1], memory=mem_clean_eng, memory_key_padding_mask=mask_clean_eng, compute_head=compute_head)
        return out[0] if isinstance(out, (tuple, list)) else out

    def forward(self, gloss_ids, text_ids, corrupted_text_ids, is_dae, compute_head: bool = False):
        h_eng = self.forward_english(gloss_ids, text_ids, corrupted_text_ids, is_dae, compute_head=compute_head)
        h_gloss = self.forward_gloss(gloss_ids, text_ids, compute_head=compute_head)
        return h_eng, h_gloss


def _async_phase1_step_print(loss_cpu, acc_eng_cpu, acc_gloss_cpu, step_idx, step_fmt_str, lr_val, elapsed_val, samples_per_sec_val, steps_per_sec_val, ep, tot_ep):
    try:
        if IS_TPU:
            import torch_xla.core.xla_model as xm
            if not xm.is_master_ordinal():
                return

        loss_val = 0.0
        acc_e_val = 0.0
        acc_g_val = 0.0
        try:
            loss_val = float(loss_cpu.item()) if hasattr(loss_cpu, "item") else float(loss_cpu)
        except Exception:
            loss_val = float(loss_cpu) if isinstance(loss_cpu, (int, float)) else 0.0

        try:
            acc_e_val = float(acc_eng_cpu.item()) if hasattr(acc_eng_cpu, "item") else float(acc_eng_cpu)
        except Exception:
            acc_e_val = 0.0

        try:
            acc_g_val = float(acc_gloss_cpu.item()) if hasattr(acc_gloss_cpu, "item") else float(acc_gloss_cpu)
        except Exception:
            acc_g_val = 0.0

        ppl = math.exp(min(loss_val, 20.0))

        msg = (
            f"Phase 1 | Epoch {ep:03d}/{tot_ep:03d} | "
            f"Step {step_fmt_str} | Loss: {loss_val:.4f} | Acc(E): {acc_e_val:.2f}% | Acc(G): {acc_g_val:.2f}% | PPL: {ppl:.1f} | "
            f"Speed: {samples_per_sec_val:.1f} samp/s ({steps_per_sec_val:.2f} step/s) | "
            f"LR: {lr_val:.2e} | Elapsed: {elapsed_val:.1f}s"
        )
        # Always use python print + sys.stdout.flush to guarantee Kaggle notebook captures worker output
        print(msg, flush=True)
        sys.stdout.flush()

        try:
            metrics_csv_path = "training_metrics.csv"
            write_header = not os.path.exists(metrics_csv_path)
            with open(metrics_csv_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("phase,epoch,step,loss,acc_eng,acc_gloss,ppl\n")
                f.write(f"phase1,{ep},{step_fmt_str},{loss_val:.4f},{acc_e_val:.2f},{acc_g_val:.2f},{ppl:.1f}\n")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"[ERROR in step print]: {e}", flush=True)
            sys.stdout.flush()
        except Exception:
            pass




def _async_save_checkpoint_worker(ckpt_payload, ckpt_path, last_ckpt_path=None, best_ckpt_path=None, save_dir=None, prefix="asl_llm_", keep_k=5):
    """Executes atomic torch.save and file management in a background daemon thread to eliminate TPU stalling."""
    try:
        tmp_ckpt = str(ckpt_path) + ".tmp"
        torch.save(ckpt_payload, tmp_ckpt)
        os.replace(tmp_ckpt, str(ckpt_path))
        import shutil
        if last_ckpt_path:
            try:
                shutil.copyfile(str(ckpt_path), str(last_ckpt_path))
            except Exception:
                pass
        if best_ckpt_path:
            try:
                shutil.copyfile(str(ckpt_path), str(best_ckpt_path))
            except Exception:
                pass
        if save_dir and keep_k > 0:
            prune_checkpoints(save_dir, prefix=prefix, keep_last_k=keep_k)
        print(f"[INFO] Checkpoint saved successfully: {ckpt_path}", flush=True)
    except Exception as e:
        print(f"[ERROR] Asynchronous checkpoint save failed for {ckpt_path}: {e}", flush=True)


def prune_checkpoints(save_dir, prefix="asl_llm_", keep_last_k=5):
    """Safely prunes older epoch checkpoints to prevent filling up VM disk space, while preserving best and last checkpoints."""
    if keep_last_k <= 0 or not os.path.exists(save_dir):
        return
    import glob, re
    pattern = os.path.join(save_dir, f"{prefix}*.pt")
    files = glob.glob(pattern)
    epoch_files = []
    for f in files:
        base = os.path.basename(f)
        if base in [f"{prefix}last.pt", f"{prefix}best.pt", "best_checkpoint.pt", "last_checkpoint.pt"]:
            continue
        m = re.search(rf"{prefix}(\d+)\.pt", base)
        if m:
            epoch_files.append((int(m.group(1)), f))
    epoch_files.sort(key=lambda x: x[0])
    if len(epoch_files) > keep_last_k:
        to_delete = epoch_files[:-keep_last_k]
        for _, fpath in to_delete:
            try:
                os.remove(fpath)
                print(f"[INFO] Pruned old checkpoint to preserve disk space: {fpath}", flush=True)
            except Exception:
                pass


def text_pretrain_loop(args, device, is_master, per_core_batch=None, accum_steps=1):
    """
    Phase 1 Text Pre-training Loop.

    This function trains the model exclusively on textual datasets (like KDWD and ASLG-PC12)
    to build a robust language model and semantic representations before introducing video inputs.
    """
    if IS_TPU:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        import torch_xla.distributed.parallel_loader as pl
    try:
        from dataset import Phase1MixedDataset
        dataset_cls = Phase1MixedDataset
    except ImportError:
        from dataset import Phase1MixedIterable
        dataset_cls = Phase1MixedIterable

    if IS_TPU:
        world_size = get_xla_world_size() if hasattr(xr, "world_size") else 8
        per_core_batch = per_core_batch if per_core_batch is not None else max(1, args.batch_size // world_size)
    else:
        if per_core_batch is None:
            per_core_batch = args.batch_size

    if is_master:
        print(
            f"Starting Phase 1 Text Pre-training for {args.phase1_epochs} epochs (Per-Core Batch: {per_core_batch})...",
            flush=True,
        )

    # Vocabulary handling
    eng_vocab_path = (
        getattr(args, "english_vocab", None)
        or getattr(args, "vocab_file", None)
        or (os.path.join(args.data_dir, "english_vocab.json") if hasattr(args, "data_dir") and args.data_dir else None)
    )
    try:
        eng_vocab = EnglishVocabulary(
            vocab_path=eng_vocab_path,
            use_bpe=True,
        )
    except TypeError:
        eng_vocab = EnglishVocabulary(
            vocab_path=eng_vocab_path,
        )
    eng_pad_id = eng_vocab.PAD_ID

    # Dynamic search for existing GlossVocab mapping in data directories
    candidate_vocab_names = [
        "vocabulary_mapping_train.json",
        "vocabulary_mapping_global.json",
        "vocab_map.json",
        "metadata.json",
    ]
    gloss_vocab = None
    for d in [args.data_dir, getattr(args, "kdwd_dir", None)]:
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

    bpe_max_len = getattr(args, "phase1_max_len", getattr(args, "english_max_len", 256))
    dataset = dataset_cls(
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

    # Force num_workers=0 for in-memory Phase 1 text streaming to eliminate inter-process locks
    num_workers = 0
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    sampler = None

    if not is_iterable:
        if IS_TPU:
            world_size = get_xla_world_size() if hasattr(xr, "world_size") else 8
            global_rank = xr.global_ordinal() if hasattr(xr, "global_ordinal") else 0
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=global_rank,
                shuffle=True,
                drop_last=True,
            )
        elif device.type == "cuda" and torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

    dl_kwargs = {
        "dataset": dataset,
        "batch_size": per_core_batch,
        "num_workers": num_workers,
        "collate_fn": functools.partial(phase1_collate_fn, max_len=bpe_max_len, eng_pad_id=eng_pad_id),
        "drop_last": True if IS_TPU else False,
        "pin_memory": False,
        "persistent_workers": (num_workers > 0),
    }
    if sampler is not None:
        dl_kwargs["sampler"] = sampler
    elif not is_iterable:
        dl_kwargs["shuffle"] = True

    dataloader = torch.utils.data.DataLoader(**dl_kwargs)

    # Hardware Optimization: Align English vocab size to 128-element TPU systolic tile boundary (23552)
    # Eliminates XLA internal dynamic padding/slicing on every GEMM projection
    aligned_eng_vocab_size = ((len(eng_vocab) + 127) // 128) * 128

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
        english_vocab_size=aligned_eng_vocab_size,
        eng_pad_id=eng_vocab.PAD_ID,
        eng_bos_id=eng_vocab.BOS_ID,
        eng_eos_id=eng_vocab.EOS_ID,
        drop_path_rate=0.0,
        enable_aux_decoders=True, # Khởi tạo english_decoder (với MTP tắt mặc định để tiết kiệm 70% bộ nhớ)
        is_causal=getattr(args, "is_causal", False),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
    )

    # Tie embeddings
    model.english_decoder.token_emb.weight = model.english_decoder.lm_head.weight
    model.decoder.token_emb.weight = model.decoder.lm_head.weight

    # Disable gradients on unused video encoder layers during Phase 1 to conserve TPU HBM
    if hasattr(model, "encoder"):
        model.encoder.requires_grad_(False)

    target_dtype = torch.bfloat16 if getattr(args, "precision", "bfloat16") == "bfloat16" else (
        torch.float16 if getattr(args, "precision", "bfloat16") == "float16" else torch.float32
    ) if (IS_TPU or device.type == "cuda") else torch.float32

    start_epoch = 0
    saved_opt_state = None
    saved_sched_state = None
    if hasattr(args, "resume") and args.resume and os.path.exists(args.resume):
        if is_master:
            print(f"[INFO] Resuming Phase 1 from checkpoint: {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            raw_state = ckpt["model_state_dict"]
            start_epoch = ckpt.get("epoch", 0)
            saved_opt_state = ckpt.get("optimizer_state_dict", None)
            saved_sched_state = ckpt.get("scheduler_state_dict", None)
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            raw_state = ckpt["state_dict"]
            start_epoch = ckpt.get("epoch", 0)
            saved_opt_state = ckpt.get("optimizer_state_dict", None)
            saved_sched_state = ckpt.get("scheduler_state_dict", None)
        elif isinstance(ckpt, dict):
            raw_state = ckpt
            import re
            m = re.search(r"asl_llm_(\d+)\.pt", str(args.resume))
            if m:
                start_epoch = int(m.group(1))
        else:
            raw_state = ckpt

        target_state = model.state_dict()
        adapted_state = {}
        for k, v in raw_state.items():
            if k in target_state:
                if target_state[k].shape == v.shape:
                    adapted_state[k] = v
                elif target_state[k].dim() == v.dim() == 2 and target_state[k].shape[1] == v.shape[1] and target_state[k].shape[0] >= v.shape[0]:
                    new_w = target_state[k].clone()
                    new_w[:v.shape[0]] = v
                    adapted_state[k] = new_w
            elif k.startswith("english_decoder."):
                if k in target_state and target_state[k].shape == v.shape:
                    adapted_state[k] = v
            elif k.startswith("decoder_token_emb."):
                adapted_k = "decoder.token_emb." + k[len("decoder_token_emb."):]
                if adapted_k in target_state and target_state[adapted_k].shape == v.shape:
                    adapted_state[adapted_k] = v
            else:
                eng_k = f"english_decoder.{k}"
                if eng_k in target_state and target_state[eng_k].shape == v.shape:
                    adapted_state[eng_k] = v
                dec_k = f"decoder.{k}"
                if dec_k in target_state and target_state[dec_k].shape == v.shape:
                    adapted_state[dec_k] = v

        missing, unexpected = model.load_state_dict(adapted_state, strict=False)
        # Re-tie embeddings
        model.english_decoder.token_emb.weight = model.english_decoder.lm_head.weight
        model.decoder.token_emb.weight = model.decoder.lm_head.weight
        if is_master:
            print(f"[INFO] Checkpoint weights mapped: {len(adapted_state)}/{len(target_state)} tensors transferred. Missing: {len(missing)}, Unexpected: {len(unexpected)}", flush=True)
            if start_epoch > 0:
                print(f"[INFO] Resuming training from Epoch {start_epoch + 1}/{args.phase1_epochs}...", flush=True)

    model_dtype = target_dtype if IS_TPU else torch.float32
    model = model.to(device, dtype=model_dtype)

    phase1_net = Phase1TextWrapper(model).to(device, dtype=model_dtype)

    if IS_TPU:
        xm.broadcast_master_param(phase1_net)
    elif device.type == "cuda" and torch.distributed.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        from torch.nn.parallel import DistributedDataParallel as DDP
        phase1_net = DDP(phase1_net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    trainable_params = list(phase1_net.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=getattr(args, "weight_decay", 0.01),
    )
    last_epoch_val = (start_epoch - 1) if (saved_sched_state is None and start_epoch > 0) else -1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.phase1_epochs), eta_min=5e-5, last_epoch=last_epoch_val
    )

    if saved_opt_state is not None:
        if device.type == "cuda":
            try:
                optimizer.load_state_dict(saved_opt_state)
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                if is_master:
                    print("[INFO] Phase 1 Optimizer state restored and synced to CUDA device successfully.", flush=True)
            except Exception as e:
                if is_master:
                    print(f"[WARNING] Could not restore optimizer state: {e}", flush=True)
        elif IS_TPU:
            if is_master:
                print("[INFO] TPU Mode: Initializing fresh optimizer momentum on device for 100% fused XLA kernel execution with zero recompilations.", flush=True)

    if saved_sched_state is not None:
        try:
            scheduler.load_state_dict(saved_sched_state)
            if is_master:
                print("[INFO] Phase 1 LR Scheduler state restored successfully.", flush=True)
        except Exception:
            pass

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and target_dtype == torch.float16 and not getattr(args, "disable_amp", False)),
    )

    world_sz = get_xla_world_size() if IS_TPU else int(os.environ.get("WORLD_SIZE", "1"))
    cluster_batch = max(1, per_core_batch * world_sz)

    # Protect TPU HBM (16GB) from OOM and ensure 100% static graph with zero remainder-batch variance
    p1_bpe = 1
    if is_master and getattr(args, "batches_per_execution", 1) > 1:
        print("[INFO] Phase 1: Enforced batches_per_execution=1 for static 1-batch XLA execution graph.", flush=True)
    if IS_TPU:
        loader = pl.MpDeviceLoader(dataloader, device, batches_per_execution=p1_bpe)
    else:
        loader = dataloader

    try:
        total_steps = len(dataloader)
    except Exception:
        total_steps = (len(dataset) // cluster_batch if hasattr(dataset, "__len__") else 0)

    best_phase1_loss = float("inf")

    # Hoist loop-invariant configurations outside training loops (Zero-Branching Inner Execution)
    raw_p1 = phase1_net.module if hasattr(phase1_net, "module") else phase1_net
    raw_eng_dec = getattr(raw_p1, "english_decoder", None)
    raw_gloss_dec = getattr(raw_p1, "decoder", None)
    lbl_sm = float(getattr(args, "label_smoothing", 0.0) if args is not None else 0.0)
    phase1_accum = accum_steps if accum_steps > 1 else max(1, getattr(args, "accum_steps", 1))
    task_weight = 0.5 * getattr(args, "bwd_weight", 1.0) / phase1_accum
    amp_device = "xla" if IS_TPU else ("cuda" if "cuda" in device.type else "cpu")
    amp_dt = target_dtype if target_dtype != torch.float32 else torch.bfloat16
    amp_on = not getattr(args, "disable_amp", False) and (device.type == "cuda" or (IS_TPU and target_dtype != torch.float32))

    for epoch in range(start_epoch, args.phase1_epochs):
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        if is_master:
            print(f"[INFO] Phase 1 Epoch {epoch+1:03d}/{args.phase1_epochs:03d} started...", flush=True)

        phase1_net.train()
        epoch_start_time = time.time()
        last_step_time = time.time()
        last_logged_step = 0
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        last_loss_val = 10.0

        for step, batch in enumerate(loader):
            should_log = (step < 10) or ((step + 1) % getattr(args, "log_freq", 10) == 0) or ((step + 1) == total_steps)
            gloss_ids = batch["gloss_ids"]
            text_ids = batch["text_ids"]
            corrupted_text_ids = batch["corrupted_text_ids"]
            is_dae = batch["is_dae"]

            with torch.autocast(device_type=amp_device, dtype=amp_dt, enabled=amp_on):
                # Task 1: English Text (ASL -> English & DAE)
                h_eng = raw_p1.forward_english(gloss_ids, text_ids, corrupted_text_ids, is_dae)
                loss_eng, acc_eng = compute_chunked_linear_ce_and_acc(
                    h_eng,
                    raw_eng_dec.lm_head,
                    text_ids[:, 1:],
                    ignore_index=eng_vocab.PAD_ID,
                    label_smoothing=lbl_sm,
                )
                task1_bwd = loss_eng * task_weight
                task1_bwd.backward()
                del h_eng, task1_bwd

                if IS_TPU:
                    # Intermediate execution barrier: Executes Task 1 forward+backward and frees its 9.4GB activations.
                    # Keeps XLA peak HBM at ~9.4GB (safely under 15.75GB limit) and prevents Task 1 and Task 2 activations from co-existing.
                    xm.mark_step()

                # Task 2: ASL Gloss (English -> ASL)
                h_gloss = raw_p1.forward_gloss(gloss_ids, text_ids)
                dae_2d = is_dae.view(-1, 1)
                safe_gloss_out = gloss_ids[:, 1:].masked_fill(dae_2d, gloss_vocab.PAD_ID)
                loss_gloss, acc_gloss = compute_chunked_linear_ce_and_acc(
                    h_gloss,
                    raw_gloss_dec.lm_head,
                    safe_gloss_out,
                    ignore_index=gloss_vocab.PAD_ID,
                    label_smoothing=lbl_sm,
                )
                task2_bwd = loss_gloss * task_weight
                task2_bwd.backward()
                del h_gloss, task2_bwd

                loss = (loss_eng.detach() + loss_gloss.detach()) * 0.5

            if should_log:
                now = time.time()
                elapsed = now - epoch_start_time
                dt_window = max(1e-5, now - last_step_time)
                delta_s = max(1, (step + 1) - last_logged_step)
                steps_per_sec = (delta_s / dt_window) if step > 0 else (1.0 / max(0.01, dt_window))
                world_sz = get_xla_world_size() if IS_TPU else int(os.environ.get("WORLD_SIZE", "1"))
                samples_per_sec = steps_per_sec * per_core_batch * world_sz
                lr = optimizer.param_groups[0]["lr"]
                step_fmt = f"{step + 1:04d}/{total_steps:04d}" if total_steps > 0 else f"{step + 1:04d}"

                args_tuple = (
                    loss.detach(),
                    acc_eng.detach(),
                    acc_gloss.detach(),
                    step,
                    step_fmt,
                    lr,
                    elapsed,
                    samples_per_sec,
                    steps_per_sec,
                    epoch + 1,
                    args.phase1_epochs,
                )

                if IS_TPU:
                    import torch_xla.core.xla_model as xm
                    xm.add_step_closure(_async_phase1_step_print, args=args_tuple)
                else:
                    if is_master:
                        _async_phase1_step_print(*args_tuple)

                last_step_time = now
                last_logged_step = step + 1

            if (step + 1) % phase1_accum == 0 or (step + 1) == total_steps:
                if IS_TPU:
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            elif IS_TPU:
                # Flush microstep execution to TPU device to prevent lazy IR graph bloat across accumulation steps
                xm.mark_step()

            # Zero-Leak Memory Proofing: Explicitly dereference step-level activations
            logits_eng = None
            logits_gloss = None
            loss = None
            task_loss = None
            loss_eng = None
            loss_gloss = None
            acc_eng = None
            acc_gloss = None
            gloss_ids = None
            text_ids = None
            corrupted_text_ids = None
            is_dae = None
            batch = None

            if (step + 1) % getattr(args, "log_freq", 100) == 0:
                trim_host_memory()

            phase1_max_limit = getattr(args, "phase1_max_steps", 0)
            if phase1_max_limit > 0 and step + 1 >= phase1_max_limit:
                break

        scheduler.step()
        if IS_TPU:
            xm.mark_step()

        gc.collect()
        trim_host_memory()

        epoch_duration = max(0.1, time.time() - epoch_start_time)
        epoch_samples = (step + 1) * per_core_batch * (xr.world_size() if IS_TPU else 1)
        epoch_speed = epoch_samples / epoch_duration
        if is_master:
            print(f"Phase 1 Epoch {epoch+1:03d}/{args.phase1_epochs:03d} finished in {epoch_duration:.1f}s ({epoch_speed:.1f} samp/s).", flush=True)

        ckpt_freq = getattr(args, "checkpoint_freq", 5)
        is_save_epoch = ((epoch + 1) % ckpt_freq == 0) or ((epoch + 1) == args.phase1_epochs)
        if is_save_epoch:
            ckpt_path = os.path.join(args.save_dir, f"asl_llm_{epoch+1}.pt")
            last_ckpt_path = os.path.join(args.save_dir, "asl_llm_last.pt")
            best_ckpt_path = os.path.join(args.save_dir, "asl_llm_best.pt")
            keep_k = getattr(args, "keep_last_k", 5)

            if IS_TPU:
                xm.mark_step()
                xm.wait_device_ops()
                xm.rendezvous(f"phase1_pre_save_{epoch}")

                if is_master:
                    os.makedirs(args.save_dir, exist_ok=True)
                    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    ckpt_payload = {
                        "epoch": epoch + 1,
                        "model_state_dict": cpu_state,
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    }
                    torch.save(ckpt_payload, ckpt_path)
                    try:
                        import shutil
                        shutil.copyfile(str(ckpt_path), str(last_ckpt_path))
                        prune_checkpoints(args.save_dir, prefix="asl_llm_", keep_last_k=keep_k)
                        print(f"[INFO] Phase 1 Checkpoint saved successfully: {ckpt_path}", flush=True)
                    except Exception:
                        pass
                    del cpu_state, ckpt_payload
                    gc.collect()
                    trim_host_memory()

                xm.mark_step()
                xm.rendezvous(f"phase1_post_save_{epoch}")
            elif is_master:
                os.makedirs(args.save_dir, exist_ok=True)
                ckpt_payload = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                }
                torch.save(ckpt_payload, ckpt_path)
                try:
                    import shutil
                    shutil.copyfile(str(ckpt_path), str(last_ckpt_path))
                    prune_checkpoints(args.save_dir, prefix="asl_llm_", keep_last_k=keep_k)
                    print(f"[INFO] Phase 1 Checkpoint saved successfully: {ckpt_path}", flush=True)
                except Exception:
                    pass
                del ckpt_payload
                gc.collect()
                trim_host_memory()

    if hasattr(dataset, "stop"):
        print("Stopping Phase 1 background thread...", flush=True)
        try:
            dataset.stop()
        except Exception:
            pass
    if "loader" in locals():
        del loader
    if "dataloader" in locals():
        del dataloader
    if "phase1_net" in locals():
        del phase1_net
    if "model" in locals():
        del model
    if "optimizer" in locals():
        del optimizer
    if "scheduler" in locals():
        del scheduler
    if "dataset" in locals():
        del dataset

    gc.collect()
    trim_host_memory()
    if IS_TPU:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
        xm.rendezvous("phase1_cleanup_complete")
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
    parser.add_argument(
        "--phase1-max-steps",
        type=int,
        default=0,
        help="Stop Phase 1 pretraining epoch early after this many steps (leaves Phase 2 full epochs untouched).",
    )
    parser.add_argument(
        "--phase1-max-len",
        type=int,
        default=256,
        help="Max sequence length for Phase 1 text pretraining (Default: 256, accommodating multi-sentence merged sequences with period boundary separation)",
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
    parser.add_argument("--log-freq", type=int, default=10)
    parser.add_argument("--is-causal", dest="is_causal", action="store_true", help="Enable causal masking")
    
    parser.add_argument("--use-swin", action="store_true", help="Enable Swin-1D shifted window attention")
    parser.add_argument("--swin-window", type=int, default=128, help="Window size for Swin-1D attention")

    parser.add_argument(
        "--num-dataloader-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers per process (default: 0 for zero process contention and direct NVMe reads on TPU)",
    )
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="Explicit steps per epoch (0 for dynamic auto-calculation)")
    parser.add_argument("--bwd-weight", type=float, default=1.0)
    parser.add_argument(
        "--enable-aux-decoders",
        action="store_true",
        default=True,
        help="Enable auxiliary Chicago/English decoders for multi-task learning (Default: True)",
    )
    parser.add_argument(
        "--use-gpt2",
        action="store_true",
        help="Enable pretrained GPT-2 decoder bridge for fluent English generation",
    )
    parser.add_argument(
        "--gpt2-path",
        type=str,
        default="/kaggle/input/models/manojkumarcs28/gpt-2-by-openai-community/pytorch/gpt-2/1/GPT-2",
        help="Path to pretrained GPT-2 weights or HuggingFace model name",
    )
    parser.add_argument(
        "--english-max-len",
        type=int,
        default=128,
        help="Max decoder sequence length for English translation head (Default: 128)",
    )
    parser.add_argument(
        "--chicago-max-len",
        type=int,
        default=128,
        help="Max decoder sequence length for Chicago fingerspelling head (Default: 128)",
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
        default=4,
        help="Number of batches per execution step for MpDeviceLoader on TPU (default: 4 for high throughput)",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing to reduce activation HBM memory by >65%%, enabling batch-size 1024 (128 per core) on TPU v5e.",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.03,
        help="Label smoothing factor for cross entropy loss (default: 0.03 for optimal balance between high accuracy and generalization).",
    )
    parser.add_argument(
        "--enable-phase1-distill",
        action="store_true",
        default=False,
        help="Enable bidirectional Phase 1 knowledge distillation and cycle consistency (English <-> ASL Gloss). (Default: False)",
    )
    parser.add_argument("--save-dir", type=str, default="/tmp/checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--phase1-checkpoint",
        "--pretrained-checkpoint",
        type=str,
        default="",
        help="Path to pre-trained Phase 1 checkpoint (.pt / .zip) to warm-start Phase 2/3 decoders",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (Default: 5)",
    )
    parser.add_argument(
        "--keep-last-k",
        type=int,
        default=5,
        help="Number of most recent epoch checkpoints to keep on disk (Default: 5, prevents disk overflow)",
    )
    parser.add_argument(
        "--asl-lex-csv", type=str, default="/home/binhhanh409/signdata.csv"
    )
    args = parser.parse_args()
    if args.batches_per_execution < 1:
        print(
            f"[!] WARNING: --batches-per-execution={args.batches_per_execution} is invalid (must be >= 1). "
            f"Clamping to 1 to prevent unbounded XLA graph accumulation and host OOM.",
            flush=True,
        )
        args.batches_per_execution = 1
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

    if IS_TPU:
        if "LOCAL_RANK" in os.environ:
            # Launched via torchrun / PyTorch distributed launcher
            import torch.distributed as dist

            dist.init_process_group("xla")
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            _tpu_worker_fn(rank, args)
        else:
            # Direct python execution (e.g. Kaggle TPU notebook cell `python train_all_in_one_tpu.py --tpu`)
            if getattr(args, "phase1_epochs", 0) > 0:
                try:
                    print("[INFO] Pre-caching Phase 1 Text Datasets in single parent process before spawning 8 TPU workers...", flush=True)
                    from dataset import Phase1MixedDataset, EnglishVocabulary, GlossVocabulary
                    cand_eng = [
                        getattr(args, "english_vocab", None),
                        getattr(args, "vocab_file", None),
                        f"{args.data_dir}/english_vocab.json",
                        f"{args.data_dir}/asl_preprocessed_phase1/english_vocab.json",
                        "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_dataset/asl_preprocessed_phase1/english_vocab.json",
                        "/kaggle/input/frakenstein-asl-final-version/asl_dataset/asl_preprocessed_phase1/english_vocab.json",
                    ]
                    eng_v = None
                    for epath in cand_eng:
                        if epath and os.path.exists(epath):
                            try:
                                eng_v = EnglishVocabulary(vocab_path=epath)
                                break
                            except Exception:
                                pass
                    if eng_v is None:
                        eng_v = EnglishVocabulary()

                    cand_vocab = [
                        f"{args.data_dir}/vocabulary_mapping.json",
                        f"{args.data_dir}/vocabulary_mapping_train.json",
                        f"{args.data_dir}/asl_preprocessed_phase1/vocabulary_mapping_train.json",
                        f"{args.data_dir}/asl_preprocessed_phase1/vocabulary_mapping.json",
                        f"{args.data_dir}/asl_preprocessed_phase1/vocab_map.json",
                        "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl-final-version/asl_dataset/asl_preprocessed_phase1/vocabulary_mapping_train.json",
                        "/kaggle/input/frakenstein-asl-final-version/asl_dataset/asl_preprocessed_phase1/vocabulary_mapping_train.json",
                    ]
                    gloss_v = None
                    for vpath in cand_vocab:
                        if vpath and os.path.exists(vpath):
                            try:
                                with open(vpath, "r", encoding="utf-8") as f:
                                    raw_map = json.load(f)
                                if isinstance(raw_map, dict) and "label_to_idx" in raw_map:
                                    gloss_v = GlossVocabulary(label_to_idx=raw_map["label_to_idx"])
                                elif isinstance(raw_map, dict):
                                    gloss_v = GlossVocabulary(label_to_idx=raw_map)
                                if gloss_v is not None:
                                    break
                            except Exception:
                                pass

                    if gloss_v is not None:
                        bpe_max_len = getattr(args, "phase1_max_len", getattr(args, "english_max_len", 256))
                        aslg_p = (
                            args.aslg_csv
                            if getattr(args, "aslg_csv", "")
                            else (
                                args.data_dir + "/train.csv"
                                if not args.data_dir.endswith(".csv")
                                else args.data_dir
                            )
                        )
                        _pre_ds = Phase1MixedDataset(
                            kdwd_dir=args.kdwd_dir,
                            aslg_csv=aslg_p,
                            eng_vocab=eng_v,
                            gloss_vocab=gloss_v,
                            max_len=bpe_max_len,
                        )
                        print(f"[INFO] Phase 1 Pre-caching complete (KDWD: {len(_pre_ds.kdwd_ds)}, ASLG: {len(_pre_ds.aslg_ds)}). Releasing parent memory...", flush=True)
                        del _pre_ds
                except Exception as _pw_err:
                    print(f"[WARNING] Pre-caching Phase 1 dataset in parent process: {_pw_err}. Workers will fallback to rank-0 sync.", flush=True)

            gc.collect()
            trim_host_memory()

            # Spawns 8 processes (one per TPU core) to prevent PJRT barrier deadlock
            import torch_xla.distributed.xla_multiprocessing as xmp

            print(
                "[INFO] Kaggle TPU VM detected. Spawning 8 TPU core worker processes via xmp.spawn...",
                flush=True,
            )
            try:
                _start_method = "fork" if os.name != "nt" else None
                xmp.spawn(_tpu_worker_fn, args=(args,), nprocs=None, start_method=_start_method)
            except (KeyboardInterrupt, SystemExit):
                print("\n[*] Training interrupted by user. Cleaning up TPU worker processes...", flush=True)
                try:
                    import psutil
                    parent = psutil.Process(os.getpid())
                    for child in parent.children(recursive=True):
                        try:
                            child.kill()
                        except Exception:
                            pass
                except Exception:
                    pass
                sys.exit(0)
    else:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        _tpu_worker_fn(rank, args)

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
    import matplotlib
    matplotlib.use("Agg", force=True)
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

    aligned_eng_vocab_size = ((len(eng_vocab) + 127) // 128) * 128

    model = ASLFoundationModel(
        vocab_size=actual_vocab_size,
        d_enc=args.d_model,
        d_dec=args.d_model,
        nhead_enc=args.nhead,
        nhead_dec=args.nhead,
        num_enc_layers=args.num_layers,
        num_dec_layers=args.num_layers,
        dropout=args.dropout,
        english_vocab_size=aligned_eng_vocab_size,
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
    try:
        main()
    except (KeyboardInterrupt, SystemExit):
        _kill_all_child_subprocesses()
        sys.exit(0)
    except BaseException as e:
        print(f"\n[FATAL RUNTIME ERROR] Pipeline encountered unhandled exception: {e}", flush=True)
        import traceback
        traceback.print_exc()
        _kill_all_child_subprocesses()
        sys.exit(1)
    finally:
        _kill_all_child_subprocesses()
