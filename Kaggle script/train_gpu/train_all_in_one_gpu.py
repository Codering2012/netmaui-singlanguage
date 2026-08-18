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

# [NEW] Persistent HLO caching to permanently bypass 13-minute XLA compilations
os.environ['XLA_PERSISTENT_CACHE_PATH'] = '/kaggle/working/xla_cache'

# [NEW] Experimental v5e Asynchronous Collective Pipelining & DP Overlap
USE_EXPERIMENTAL_XLA_FLAGS = True
USE_DYNAMO_COMPILE = True

if USE_EXPERIMENTAL_XLA_FLAGS:
    xla_flags = os.environ.get('XLA_FLAGS', '')
    xla_flags += " --xla_enable_async_all_gather=true"
    xla_flags += " --xla_tpu_enable_async_collective_fusion=true"
    xla_flags += " --xla_tpu_enable_ici_ag_pipelining=true"
    xla_flags += " --xla_should_allow_loop_variant_parameter_in_chain=kEnabled"
    xla_flags += " --xla_should_add_loop_invariant_op_in_chain=kEnabled"
    xla_flags += " --xla_tpu_enable_data_parallel_all_reduce_opt=true"
    xla_flags += " --xla_tpu_data_parallel_opt_different_sized_ops=true"
    xla_flags += " --xla_tpu_enable_flash_attention=true"
    os.environ['XLA_FLAGS'] = xla_flags

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# XLA Native BFloat16 Compilation Flags (Preserving FP32 Precision for Sensitive Operations)
os.environ["XLA_USE_BF16"] = "1"
os.environ.pop("XLA_DOWNCAST_BF16", None)

# XLA Ultra-Performance Memory Optimization Flags (Fused Optimizers & Dispatch)
os.environ["XLA_USE_FUSED_ADAMW"] = "1"
os.environ["XLA_USE_FUSED_ADAM"] = "1"
os.environ["XLA_EXPERIMENTAL_FUSED_ADAM"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

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

if "--tpu" in sys.argv and _XLA_AVAILABLE:
    os.environ["PJRT_DEVICE"] = "TPU"


def is_tpu_runtime() -> bool:
    return _XLA_AVAILABLE and os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


IS_TPU = False


def get_xla_world_size() -> int:
    if IS_TPU:
        try:
            
            return xr.world_size()
        except Exception:
            try:
                
                return getattr(xm, "xrt_world_size", lambda: 1)()
            except Exception:
                pass
    return 1


train_dir = Path(__file__).resolve().parent
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))
from dataset import (
    create_dataloader,
    normalize_vocabulary,
)


def _distributed_normalize(
    local_sum: torch.Tensor, local_weight: torch.Tensor
) -> torch.Tensor:
    normed = local_sum / local_weight.clamp_min(1e-8)
    return torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_torch_device(dev_str: Union[str, torch.device]) -> torch.device:
    if isinstance(dev_str, torch.device):
        return dev_str
    dev_s = str(dev_str).lower()
    if IS_TPU and "xla" in dev_s:
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
    def __init__(self, d_model: int, eps: float = 1e-5):
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
        d_model: int = 512,
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
        # Removed .any() validation check to prevent XLA device-to-host syncs in the forward pass.
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
    def __init__(self, r: int = 80, d_model: int = 512):
        super().__init__()
        self.r, self.d_model = r, d_model
        import math
        self._jl_proj = nn.Parameter(
            torch.randn(d_model, 16) / math.sqrt(16), requires_grad=False
        )
        self.register_buffer(
            "kernel", torch.tensor([0.25, 0.5, 0.25]).view(1, 1, 3), persistent=False
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor = None,
        key_padding_mask: torch.Tensor = None,
        **kwargs,
    ) -> tuple:
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
                    "token_sizes": kwargs.get("token_sizes", None),
                },
            )

        if T > 1:
            import torch.nn.functional as F
            h_smooth = F.conv1d(
                F.pad(h.transpose(1, 2), (1, 1), mode="constant"),
                self.kernel.to(dtype=h.dtype).expand(D, 1, 3),
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
        import torch.nn.functional as F
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
            cross_invalid = invalid_a | invalid_b
            sim_matrix = sim_matrix.masked_fill(cross_invalid, -1e4)

        scores, dst_idx = sim_matrix.max(dim=-1)
        _, merge_idx = scores.topk(r_clamp, dim=-1, largest=True, sorted=False)

        matched_b_indices_local = dst_idx.gather(1, merge_idx)

        # Drop the merged tokens from 'a'
        a_unmerged_mask = torch.ones(
            B, min_half, device=h.device, dtype=torch.bool
        ).scatter_(1, merge_idx, False)
        
        # Keep non-merged elements
        a_idx = torch.arange(min_half, device=h.device)
        unmerged_a_indices = a_idx.unsqueeze(0).expand(B, -1)[a_unmerged_mask].view(B, -1)
        
        token_sizes = kwargs.get("token_sizes", None)
        if token_sizes is None:
            token_sizes = torch.ones(B, T, 1, device=h.device, dtype=h.dtype)

        sa, sb = token_sizes[:, 0::2][:, :min_half], token_sizes[:, 1::2][:, :min_half]
        a_weighted = a * sa
        b_weighted = b * sb

        b_weighted_updated = b_weighted.clone().scatter_add_(
            1,
            matched_b_indices_local.unsqueeze(-1).expand(-1, -1, D),
            a_weighted.gather(1, merge_idx.unsqueeze(-1).expand(-1, -1, D)),
        )
        sb_updated = sb.clone().scatter_add_(
            1,
            matched_b_indices_local.unsqueeze(-1),
            sa.gather(1, merge_idx.unsqueeze(-1)),
        )
        b_updated = b_weighted_updated / sb_updated.clamp(min=1e-5)

        # Assemble new sequence dynamically
        a_kept = a.gather(1, unmerged_a_indices.unsqueeze(-1).expand(-1, -1, D))
        sa_kept = sa.gather(1, unmerged_a_indices.unsqueeze(-1))
        
        h_out = torch.cat([a_kept, b_updated], dim=1)
        sizes_out = torch.cat([sa_kept, sb_updated], dim=1)
        
        mask_out = None
        if mask is not None:
            ma, mb = mask[:, 0::2][:, :min_half], mask[:, 1::2][:, :min_half]
            ma_kept = ma.gather(1, unmerged_a_indices)
            mask_out = torch.cat([ma_kept, mb], dim=1)
            
        mlm_out = None
        if mlm_mask is not None:
            mla, mlb = mlm_mask[:, 0::2][:, :min_half], mlm_mask[:, 1::2][:, :min_half]
            mla_kept = mla.gather(1, unmerged_a_indices)
            mlm_out = torch.cat([mla_kept, mlb], dim=1)
            
        fi_out = None
        if fi is not None:
            fia, fib = fi[:, 0::2][:, :min_half], fi[:, 1::2][:, :min_half]
            fia_kept = fia.gather(1, unmerged_a_indices)
            fi_out = torch.cat([fia_kept, fib], dim=1)
            
        return h_out, mask_out, {
            "T_orig": T,
            "sorted_routing": torch.arange(T, device=h.device),
            "mlm_out": mlm_out,
            "frame_indices": fi_out,
            "token_sizes": sizes_out,
        }


# ==============================================================================
# 6. ENCODER ARCHITECTURE
# ==============================================================================


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = (torch.rand(shape, dtype=x.dtype, device=x.device) > drop_prob).to(x.dtype)
    random_tensor = random_tensor / (1.0 - drop_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class GroupedQueryEncoderAttention(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        max_len: int = 512,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert nhead % kv_heads == 0
        self.nhead, self.kv_heads, self.groups, self.head_dim = (
            nhead,
            kv_heads,
            nhead // kv_heads,
            d_model // nhead,
        )
        self.scale = 1.0 / np.sqrt(self.head_dim)
        
        # DeepSeek V3 MLA (Multi-Head Latent Attention) Compression
        self.latent_dim = d_model // 4
        self.kv_latent_proj = nn.Linear(d_model, self.latent_dim, bias=False)
        self.kv_latent_norm = RMSNorm(self.latent_dim)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(self.latent_dim, 2 * kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.q_norm = RMSNorm(d_model)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=max_len)
        self.dropout_p = dropout_p

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape
        q_in = self.q_norm(x)
        
        # DeepSeek V3 MLA Latent Compression
        kv_latent = self.kv_latent_proj(x)
        kv_latent = self.kv_latent_norm(kv_latent)
        
        q = self.q_proj(q_in).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_latent)
        k, v = torch.split(kv, kv.size(-1) // 2, dim=-1)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Positional Encoding (RoPE) directly to Q and K
        q, k = self.rope(q, k)

        if self.groups > 1:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)

        if key_padding_mask is not None:
            attn_mask = ~(key_padding_mask.view(B, 1, 1, T).bool())
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = self.out_proj(out.transpose(1, 2).reshape(B, T, -1))
        return out


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
        )
        self.norm = RMSNorm(channels)
        self.pw_conv1, self.pw_conv2 = nn.Linear(
            channels, channels * expansion
        ), nn.Linear(channels * expansion, channels)
        self.act, self.se = nn.GELU(), SpatialTemporalSE(channels)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        y = self.norm(
            F.conv1d(
                F.pad(
                    x.transpose(1, 2),
                    (6, 0),
                    mode="replicate",
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
        d_model: int = 512,
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

        self.fwd_conv1d = nn.Conv1d(
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

        self.head_norm_fwd, self.gated_norm = (
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
        if key_padding_mask is not None:
            kpm_b = key_padding_mask.unsqueeze(-1)
            dt_act = dt_act.masked_fill(kpm_b, 0.0)
            x = x.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            B = B.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            C = C.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            log_decay = -((dt_act * A.view(1, 1, H_sz)).clamp(min=0.0, max=20.0))
        else:
            log_decay = -((dt_act * A.view(1, 1, H_sz)).clamp(min=1e-4, max=20.0))

        Q = min(chunk_size, T_sz)
        pad_len = (Q - (T_sz % Q)) % Q
        if pad_len > 0:
            x, B, C, log_decay, dt_act = (
                F.pad(x, (0, 0, 0, 0, 0, pad_len)),
                F.pad(B, (0, 0, 0, 0, 0, pad_len)),
                F.pad(C, (0, 0, 0, 0, 0, pad_len)),
                F.pad(log_decay, (0, 0, 0, pad_len), value=0.0),
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
                -1e9,
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
                -1e9,
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
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        **kwargs,
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
        x_fwd_h = F.silu(
            self.fwd_conv1d(x_proj.transpose(1, 2))[:, :, :T_sz].transpose(1, 2)
        ).view(B_sz, T_sz, self.nheads, self.headdim)
        B_h_fwd, C_h_fwd = B_ssm_fwd.view(
            B_sz, T_sz, self.nheads, self.d_state
        ), C_ssm_fwd.view(B_sz, T_sz, self.nheads, self.d_state)

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
            x_fwd_h,
            dt_fwd + self.dt_bias,
            A,
            B_h_fwd,
            C_h_fwd,
            key_padding_mask=key_padding_mask,
            reverse=True,
        )

        y_normed = self.head_norm_fwd(y_fwd + y_bwd)
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
        d_model: int = 512,
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
        self.conv1 = nn.Conv1d(in_dim, 256, kernel_size=7, padding=0, groups=1)
        self.norm1 = nn.GroupNorm(8, 256)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv1d(256, 256, kernel_size=5, padding=0, groups=256)
        self.conv3 = nn.Conv1d(256, out_dim, kernel_size=1)
        self.norm2 = nn.GroupNorm(8, out_dim)
        self.act2 = nn.GELU()
        self.out_proj = nn.Linear(out_dim, out_dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = x.size(0), x.size(1)
        x_flat = x.reshape(B, T, -1) if x.dim() == 4 else x
        x_t = x_flat.transpose(1, 2)
        if mask is not None:
            x_t = x_t * mask.unsqueeze(1).to(x_t.dtype)
        
        feat_seq = x_t
        feat_seq = self.act1(self.norm1(self.conv1(F.pad(feat_seq, (6, 0), mode="constant"))))
        feat_seq = self.conv2(F.pad(feat_seq, (4, 0), mode="constant"))
        feat_seq = self.act2(self.norm2(self.conv3(feat_seq)))
        
        feat_seq = feat_seq.transpose(1, 2)
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
            indices = torch.arange(S, device=q.device) + offset
            cos = torch.index_select(self.cos_cache, 2, indices).to(q.dtype)
            sin = torch.index_select(self.sin_cache, 2, indices).to(q.dtype)
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
        d_model: int = 512,
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
        
        # DeepSeek V3 MLA Latent Compression
        self.latent_dim = d_model // 4
        self.kv_latent_proj = nn.Linear(d_model, self.latent_dim, bias=False)
        self.kv_latent_norm = RMSNorm(self.latent_dim)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = nn.Linear(self.latent_dim, 2 * kv_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len=max_seq_len)
        nn.init.normal_(self.q_proj.weight, std=0.02)
        nn.init.normal_(self.kv_proj.weight, std=0.02)
        nn.init.normal_(self.o_proj.weight, std=0.02 / math.sqrt(2.0))

    def forward(
        self,
        x: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ):
        B, T, _ = x.shape
        q_in = self.q_norm(x)
        
        # MLA Latent Projection
        kv_latent = self.kv_latent_proj(x)
        kv_latent = self.kv_latent_norm(kv_latent)
        
        q = self.q_proj(q_in).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_latent)
        k, v = torch.split(kv, kv.size(-1) // 2, dim=-1)
        k = k.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.kv_heads, self.head_dim).transpose(1, 2)

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
    def __init__(self, d_model: int = 512, nhead: int = 8, kv_heads: int = 2):
        super().__init__()
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
        self.kv_proj = nn.Linear(self.latent_dim, 2 * kv_heads * self.head_dim, bias=False)
        
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.q_norm = RMSNorm(d_model)
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
            # MLA Latent Projection for Cross Attention
            kv_latent = self.kv_latent_proj(memory)
            kv_latent = self.kv_latent_norm(kv_latent)
            
            kv = self.kv_proj(kv_latent)
            k, v = torch.split(kv, kv.size(-1) // 2, dim=-1)
            k = k.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, S, self.kv_heads, self.head_dim).transpose(1, 2)

        current_key_value = (k, v) if use_cache else None

        k_exp = k.repeat_interleave(self.groups, dim=1)
        v_exp = v.repeat_interleave(self.groups, dim=1)

        if memory_key_padding_mask is not None:
            attn_mask = memory_key_padding_mask.view(B, 1, 1, k.size(2)).bool()
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask)
        out = self.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        return (out, current_key_value) if use_cache else out


class ASLDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
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
        
        self.mtp_proj_2 = nn.Sequential(nn.Linear(d_model, d_model), RMSNorm(d_model), nn.SiLU())
        self.mtp_proj_3 = nn.Sequential(nn.Linear(d_model, d_model), RMSNorm(d_model), nn.SiLU())

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
                torch.full_like(tgt_ids, GlossVocabulary.UNK_ID),
                tgt_ids,
            )
        else:
            dropped_tgt_ids = tgt_ids

        # Removed .any() validation check to prevent XLA device-to-host syncs
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
        
        logits_2 = self.lm_head(self.mtp_proj_2(h))
        logits_3 = self.lm_head(self.mtp_proj_3(h))
        extra_logits = {"logits_2": logits_2, "logits_3": logits_3}
        
        return (logits, h, extra_logits, new_key_values) if use_cache else (logits, h, extra_logits)


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
                "mtp2": 4.0,
                "mtp3": 2.0,
            }

        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(-math.log(v), dtype=torch.float32))
                for name, v in loss_config.items()
            }
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss = torch.zeros((), device=next(iter(losses.values())).device)
        for name, loss in losses.items():
            if name not in self.log_vars:
                raise ValueError(
                    f"Unregistered loss key '{name}' produced by model! Please add it to HomoscedasticLossWrapper config."
                )

            raw_s = self.log_vars[name].to(loss.device)
            # Smooth parameterization using tanh instead of hard clamping
            s = torch.tanh(raw_s / 5.0) * 5.0
            prec = torch.exp(-s)
            
            # Null-loss masking: ONLY update log-variance parameter s and loss when task is active (loss != 0)
            is_active = (loss != 0).to(loss.dtype)
            task_loss = (0.5 * prec * loss + 0.5 * s) * is_active
            total_loss = total_loss + task_loss

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
        w_norm = F.normalize(self.weight.float(), p=2, dim=-1, eps=1e-5).to(x.dtype)
        # Cosine similarity scaled by learnable temperature tau
        safe_tau = (F.softplus(self.tau) + 1.0).to(x.dtype)
        return F.linear(x_norm, w_norm) * safe_tau


class CTCHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = CosineLinear(d_model, vocab_size)

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        if getattr(self, "debug_xla", False) and torch.isnan(enc_seq).any():
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
        gt_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = vis_emb.device
        import torch.distributed as dist

        if IS_TPU and "xla" in str(device).lower():
            
            world_size = get_xla_world_size()
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        v = F.normalize(vis_emb.float(), p=2, dim=-1, eps=1e-8)
        s = F.normalize(sent_emb.float(), p=2, dim=-1, eps=1e-8)

        if world_size > 1 and IS_TPU and "xla" in str(device).lower():
            fused_sv = torch.cat([s, v], dim=1)
            fused_sv_all = xm.all_gather(fused_sv)
            s_all = fused_sv_all[:, :s.shape[1]]
            v_all = fused_sv_all[:, s.shape[1]:]
            
            if valid_mask is not None and gt_tokens is not None:
                gt_float = gt_tokens.float()
                fused_mask_gt = torch.cat([valid_mask.float().unsqueeze(1), gt_float], dim=1)
                fused_mask_gt_all = xm.all_gather(fused_mask_gt)
                valid_mask_all = fused_mask_gt_all[:, 0].bool()
                gt_tokens_all = fused_mask_gt_all[:, 1:].long()
            elif valid_mask is not None:
                valid_mask_all = xm.all_gather(valid_mask.bool()).bool()
                gt_tokens_all = None
            elif gt_tokens is not None:
                valid_mask_all = None
                gt_tokens_all = xm.all_gather(gt_tokens)
            else:
                valid_mask_all = None
                gt_tokens_all = None
        else:
            v_all = v
            s_all = s
            valid_mask_all = valid_mask.bool() if valid_mask is not None else None
            gt_tokens_all = gt_tokens

        if v.size(0) == 0:
            return torch.zeros((), device=device)

        temp = F.softplus(self.log_temp) + 0.05
        # logits_v2s shape: [B, W*B]
        logits_v2s = torch.matmul(v, s_all.transpose(-1, -2)) / temp
        # logits_s2v shape: [B, W*B]
        logits_s2v = torch.matmul(s, v_all.transpose(-1, -2)) / temp

        if valid_mask_all is not None:
            # Mask out invalid candidate columns in [B, 8B] logits matrix
            invalid_candidate_mask = ~valid_mask_all
            logits_v2s = logits_v2s.masked_fill(invalid_candidate_mask.unsqueeze(0), -1e9)
            logits_s2v = logits_s2v.masked_fill(invalid_candidate_mask.unsqueeze(0), -1e9)

        if gt_tokens_all is not None:
            # Treat identically padded sequences as positives across all gathered replicas [B, 8B]
            pos_mask = (
                (gt_tokens.unsqueeze(1) == gt_tokens_all.unsqueeze(0)).all(dim=-1).float()
            )
        else:
            rank_val = 0
            if IS_TPU and "xla" in str(device).lower():
                try:
                                        rank_val = xr.global_ordinal()
                except Exception:
                    try:
                                                rank_val = getattr(xm, "get_ordinal", lambda: 0)()
                    except Exception:
                        rank_val = 0
            elif dist.is_initialized():
                rank_val = dist.get_rank()

            global_local_rows = rank_val * v.size(0) + torch.arange(v.size(0), device=v.device)
            labels_all = torch.arange(s_all.size(0), device=v.device)
            pos_mask = (global_local_rows.unsqueeze(1) == labels_all.unsqueeze(0)).float()

        if valid_mask is not None:
            valid_rows = valid_mask.float()
            pos_mask = pos_mask * valid_rows.unsqueeze(1)
            if valid_mask_all is not None:
                pos_mask = pos_mask * valid_mask_all.float().unsqueeze(0)
        else:
            valid_rows = torch.ones(v.shape[0], device=v.device)

        # v2s loss computation
        exp_logits_v2s = torch.exp(logits_v2s - logits_v2s.max(dim=-1, keepdim=True)[0])
        denom_v2s = torch.clamp(exp_logits_v2s.sum(dim=-1, keepdim=True).float(), min=1e-4)
        log_prob_v2s = (logits_v2s - logits_v2s.max(dim=-1, keepdim=True)[0]) - torch.log(denom_v2s)

        # s2v loss computation
        exp_logits_s2v = torch.exp(logits_s2v - logits_s2v.max(dim=-1, keepdim=True)[0])
        denom_s2v = torch.clamp(exp_logits_s2v.sum(dim=-1, keepdim=True).float(), min=1e-4)
        log_prob_s2v = (logits_s2v - logits_s2v.max(dim=-1, keepdim=True)[0]) - torch.log(denom_s2v)

        pos_count = pos_mask.sum(dim=-1).clamp(min=1.0)
        loss_v2s = -(log_prob_v2s * pos_mask).sum(dim=-1) / pos_count
        loss_s2v = -(log_prob_s2v * pos_mask).sum(dim=-1) / pos_count

        loss = 0.5 * (loss_v2s + loss_s2v)

        if sample_weights is not None:
            loss = loss * sample_weights

        weight_sum = valid_rows
        loss = loss * valid_rows

        if sample_weights is not None:
            weight_sum = weight_sum * sample_weights

        res = _distributed_normalize(loss.sum(), weight_sum.sum())

        return res


class DenseSentenceSemanticLoss(nn.Module):
    def __init__(self, d_model: int = 512, embed_dim: int = 256):
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
        cos_loss = (1.0 - cos_sim) * has_tokens

        global_mean = p.mean()
        global_var = p.var(unbiased=False)
        std_loss = torch.mean(F.relu(1.0 - std_p))

        loss = (cos_loss + 0.5 * std_loss) * has_tokens

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
            return torch.zeros((), device=features.device)
        features = F.normalize(features.float(), p=2, dim=1, eps=1e-5)
        device = features.device

        import torch.distributed as dist

        if IS_TPU and "xla" in str(device).lower():
            
            world_size = get_xla_world_size()
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if world_size > 1 and IS_TPU and "xla" in str(device).lower():
            fused = torch.cat([features, labels.unsqueeze(1).float()], dim=1)
            fused_all = xm.all_gather(fused)
            all_feats = fused_all[:, :-1]
            all_labels = fused_all[:, -1].long()
        else:
            all_feats = features
            all_labels = labels

        B = features.shape[0]
        pos_mask = torch.eq(labels.view(-1, 1), all_labels.view(1, -1)).float()
        valid_labels = (all_labels.view(1, -1) != -1).float()
        pos_mask = pos_mask * valid_labels

        # Zero out self-pair matches so sample is not its own positive across all TPU ranks
        rank_val = 0
        if IS_TPU and "xla" in str(device).lower():
            try:
                                rank_val = xr.global_ordinal()
            except Exception:
                try:
                                        rank_val = getattr(xm, "get_ordinal", lambda: 0)()
                except Exception:
                    rank_val = 0
        elif dist.is_initialized():
            rank_val = dist.get_rank()

        # Fully static XLA-friendly self-masking
        global_indices = torch.arange(all_feats.shape[0], device=device)
        local_indices = rank_val * B + torch.arange(B, device=device)
        # Broadcast to create a [B, global_B] boolean mask
        is_self = local_indices.unsqueeze(1) == global_indices.unsqueeze(0)
        pos_mask = torch.where(is_self, torch.zeros_like(pos_mask), pos_mask)

        pos_logits = torch.matmul(features.float(), all_feats.float().T) / float(self.temperature)
        # Mask self-similarity in denominator so exp(1.0/tau) = exp(14.28) does not suppress negative gradients
        pos_logits[local_rows[valid_cols], global_self_cols[valid_cols]] = -1e9
        exp_logits = torch.exp(pos_logits - pos_logits.max(dim=1, keepdim=True)[0])
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
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        return loss


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float = 1.0):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class LandmarkReconstructionHead(nn.Module):
    def __init__(self, d_model: int = 512, out_dim: int = 540):
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
            idx = torch.clamp(frame_indices.long(), min=0, max=self.pe.size(1) - 1)
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
        # Change input stem to expect 768 perfectly aligned dimensions (640 padded + 128)
        self.input_stem = nn.Sequential(
            nn.Linear(768, d_enc), RMSNorm(d_enc), nn.GELU()
        )
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, num_enc_layers)]

        self.blocks = nn.ModuleList()
        for i in range(num_enc_layers):
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
        self.domain_head = nn.Linear(d_enc, 4)

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
        # [NEW HOTFIX] On TPU, we fix the ratio to 30 completely to avoid graph breaks.
        progress = epoch / max(1, max_epochs - 1)

        if IS_TPU:
            new_r = 30
        elif progress < 0.25:
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
            used_mlm_mask = None

        if x_in.dim() == 4 and x_in.size(2) == 60 and x_in.size(3) >= 3:
            xk = x_in
            x_flat = x_in.reshape(B, T, -1)
            v_tokens = self.visual_encoder(xk, mask=mask)
        else:
            x_flat = x_in.reshape(B, T, -1) if x_in.dim() == 4 else x_in
            v_tokens = self.visual_encoder(x_in, mask=mask)

        # Pad 540 to 640 with zeros to hit the magic 768 total when concatenated
        if x_flat.size(-1) == 540:
            x_flat = F.pad(x_flat, (0, 100), value=0.0)

        h = self.input_stem(torch.cat([x_flat, v_tokens], dim=-1))
        h = self.pos_enc(h, frame_indices=frame_indices)
        h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)

        routing_fi = frame_indices.long() if frame_indices is not None else None

        cur_mask = mask
        if cur_mask is not None:
            kpm = torch.cat(
                [torch.zeros((B, 1), dtype=torch.bool, device=h.device), ~cur_mask],
                dim=1,
            )
        else:
            kpm = None

        token_sizes = torch.ones(B, T, 1, device=h.device, dtype=h.dtype)

        for idx, block in enumerate(self.blocks):
            if isinstance(block, TokenMergingBlock):
                cls_t = h[:, :1]
                seq_t = h[:, 1:]
                seq_t, cur_mask, routing_info = block(
                    seq_t,
                    cur_mask,
                    token_sizes=token_sizes,
                    mlm_mask=used_mlm_mask,
                    frame_indices=routing_fi,
                )
                if "token_sizes" in routing_info and routing_info["token_sizes"] is not None:
                    token_sizes = routing_info["token_sizes"]
                if "mlm_out" in routing_info and routing_info["mlm_out"] is not None:
                    used_mlm_mask = routing_info["mlm_out"]
                if (
                    "frame_indices" in routing_info
                    and routing_info["frame_indices"] is not None
                ):
                    routing_fi = routing_info["frame_indices"]
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
                if routing_fi is not None:
                    cls_fi = torch.zeros(
                        (B, 1), dtype=routing_fi.dtype, device=routing_fi.device
                    )
                    pos_fi = torch.cat([cls_fi, routing_fi + 1], dim=1)
                else:
                    pos_fi = None
                h = block(h, key_padding_mask=kpm, frame_indices=pos_fi)

            if getattr(self, "debug_xla", False) and torch.isnan(h).any():
                print(f"NaN introduced at block {idx}!")
                break

        h = self.enc_final_norm(h)
        if getattr(self, "debug_xla", False) and torch.isnan(h).any():
            print("NaN introduced at enc_final_norm!")

        # Return routing_fi directly (strictly 0-indexed original frame indices)
        return h[:, 0], h[:, 1:], cur_mask, used_mlm_mask, routing_fi, mask

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
        # Always compute kinematics and augmentations on TPU to avoid Host CPU bottleneck
        x = x[..., :3]
        if True:
            B, T, K, C = x.shape
            if self.training:
                scale = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.85, 1.15)
                x = x * scale

                shift = torch.empty(B, 1, 1, 2, device=x.device).uniform_(-0.035, 0.035)
                x[..., :2] = x[..., :2] + shift

                x = x + torch.randn_like(x) * 0.035

                roll = torch.empty(B, device=x.device).uniform_(-10.0, 10.0) * (3.14159 / 180.0)
                cos_r = torch.cos(roll).view(B, 1, 1, 1)
                sin_r = torch.sin(roll).view(B, 1, 1, 1)

                center = x[..., :2].mean(dim=(1, 2), keepdim=True)
                x_centered = x[..., :2] - center
                x_rot_0 = x_centered[..., 0:1] * cos_r - x_centered[..., 1:2] * sin_r
                x_rot_1 = x_centered[..., 0:1] * sin_r + x_centered[..., 1:2] * cos_r
                x[..., :2] = torch.cat([x_rot_0, x_rot_1], dim=-1) + center

                kp_mask = (torch.rand(B, T, K, 1, device=x.device) > 0.05).float()
                x = x * kp_mask

            dt = torch.ones(B, T, 1, 1, device=x.device)
            if frame_indices is not None and T > 1:
                actual_dt = (frame_indices[:, 1:] - frame_indices[:, :-1]).unsqueeze(-1).unsqueeze(-1)
                dt[:, 1:] = torch.where(actual_dt == 0, torch.ones_like(actual_dt), actual_dt)

            vel = torch.zeros_like(x)
            acc = torch.zeros_like(x)

            if T > 1:
                vel[:, 1:] = (x[:, 1:] - x[:, :-1]) / dt[:, 1:]
                vel[:, 0] = vel[:, 1]
                acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]
                acc[:, 0] = acc[:, 1]

            x = torch.cat([x, vel, acc], dim=-1)
            
            if x.shape[-1] < self.channels_per_kp:
                pad = torch.zeros(B, T, K, self.channels_per_kp - x.shape[-1], device=x.device)
                x = torch.cat([x, pad], dim=-1)
            x = x[..., :self.channels_per_kp]

        h_cls, h_seq, enc_mask, used_mlm_mask, fi_out, orig_enc_mask = self._encode(
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
                out = decoder_module(dec_in, h_seq, memory_key_padding_mask=enc_mask)
                if len(out) == 2:
                    return out[0], out[1], None
                elif len(out) == 3:
                    return out[0], out[1], out[2]
                return out
            return None, None, None

        mtp_logits = None
        if gloss_seq is not None:
            dec_logits, dec_hidden, mtp_logits = decode_seq(self.decoder, gloss_seq)
        if chicago_seq is not None and (chicago_seq != GlossVocabulary.PAD_ID).sum().item() > 0:
            chicago_logits, _, _ = decode_seq(self.chicago_decoder, chicago_seq)
        if english_seq is not None and (english_seq != GlossVocabulary.PAD_ID).sum().item() > 0:
            english_logits, _, _ = decode_seq(self.english_decoder, english_seq)

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

        ctc_log_probs = self.ctc_head(h_seq)

        if self.training and used_mlm_mask is not None:
            mlm_logits = self.mlm_head(h_seq)
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
            "orig_x": x,
            "enc_seq": h_seq,
            "dec_logits": dec_logits,
            "mtp_logits": mtp_logits,
            "chicago_logits": chicago_logits,
            "english_logits": english_logits,
            "dec_hidden": dec_hidden,
            "ctc_log_probs": ctc_log_probs,
            "mlm_logits": mlm_logits,
            "mlm_mask": used_mlm_mask,
            "vis_emb": vis_emb,
            "sent_emb": sent_emb,
            "proj_feats": proj_feats,
            "domain_logits": domain_logits,
            "aux_logits": aux_logits,
            "pred_len": pred_len,
            "chicago_pred_len": chicago_pred_len,
            "english_pred_len": english_pred_len,
            "pooled_enc": pooled_enc,
            "enc_mask": enc_mask,
            "orig_enc_mask": orig_enc_mask,
        }

    @torch.no_grad()
    def generate(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        task: str = "gloss",
        temperature: float = 1.0,
        top_k: int = 50,
        use_mtp_speculative: bool = True,
    ) -> torch.Tensor:
        """
        Real-Time Streaming Autoregressive Inference with KV Caching & DeepSeek MTP Speculative Decoding.
        Achieves real-time translation parity (Google Translate-like low latency).
        """
        self.eval()
        B = x.size(0)
        device = x.device

        # 1. Encode visual landmarks ONCE
        _, h_seq, enc_mask, _, _, _ = self._encode(x, mask)

        # Select target decoder
        if task == "chicago":
            decoder_module = self.chicago_decoder
            eos_id = GlossVocabulary.EOS_ID
        elif task == "english":
            decoder_module = self.english_decoder
            eos_id = GlossVocabulary.EOS_ID
        else:
            decoder_module = self.decoder
            eos_id = GlossVocabulary.EOS_ID

        # Start sequence with BOS (B, 1)
        generated_ids = torch.full((B, 1), GlossVocabulary.BOS_ID, dtype=torch.long, device=device)
        past_key_values = None
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        step = 0
        while step < max_new_tokens:
            current_input = generated_ids[:, -1:]

            # Forward pass through decoder with KV cache
            logits, h_out, extra_logits, past_key_values = decoder_module(
                current_input,
                h_seq,
                memory_key_padding_mask=enc_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            # Extract next token logits (B, 1, V) -> (B, V)
            do_sample = temperature != 1.0 or top_k > 0
            if do_sample:
                next_logits = logits[:, -1, :] / max(1e-4, temperature)
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits = torch.where(
                        next_logits < v[:, [-1]], 
                        torch.full_like(next_logits, -1e9), 
                        next_logits
                    )
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # (B, 1)

            # Retain EOS_ID in generated output before padding subsequent steps
            is_eos = (next_token.squeeze(-1) == eos_id)
            step_token = torch.where(finished.unsqueeze(-1), GlossVocabulary.PAD_ID, next_token)
            finished = finished | is_eos
            generated_ids = torch.cat([generated_ids, step_token], dim=1)
            step += 1

            # DeepSeek MTP Speculative Decoding (Draft Verification with KV cache alignment)
            if use_mtp_speculative and B == 1 and extra_logits and "logits_2" in extra_logits:
                draft_logits_2 = extra_logits["logits_2"][:, -1, :] / max(1e-4, temperature)
                if top_k > 0:
                    v2, _ = torch.topk(draft_logits_2, min(top_k, draft_logits_2.size(-1)))
                    draft_logits_2[draft_logits_2 < v2[:, [-1]]] = -1e9
                draft_token_2 = torch.argmax(F.softmax(draft_logits_2, dim=-1), dim=-1, keepdim=True)

                saved_kv_cache = [
                    ((s_k.clone(), s_v.clone()), (c_k.clone(), c_v.clone()))
                    for (s_k, s_v), (c_k, c_v) in past_key_values
                ] if past_key_values is not None else None

                v_logits, _, _, test_kv = decoder_module(
                    step_token,
                    h_seq,
                    memory_key_padding_mask=enc_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                verifier_token = torch.argmax(v_logits[:, -1, :], dim=-1, keepdim=True)
                is_matched = bool((verifier_token == draft_token_2).all().item())
                if is_matched:
                    past_key_values = test_kv
                    step_token_2 = torch.where(finished.unsqueeze(-1), GlossVocabulary.PAD_ID, draft_token_2)
                    is_eos_2 = (draft_token_2.squeeze(-1) == eos_id)
                    finished = finished | is_eos_2
                    generated_ids = torch.cat([generated_ids, step_token_2], dim=1)
                    step += 1
                else:
                    past_key_values = saved_kv_cache

        return generated_ids


def _xla_ctc_loss(
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    blank: int = 0,
) -> torch.Tensor:
        
    B, T, V = log_probs.shape
    S = targets.shape[1]
    
    U = 2 * S + 1
    ext_targets = torch.full((B, U), blank, dtype=targets.dtype, device=log_probs.device)
    ext_targets[:, 1::2] = targets
    
    ext_targets_expanded = ext_targets.unsqueeze(1).expand(B, T, U)
    log_probs_ext = torch.gather(log_probs, 2, ext_targets_expanded)
    
    log_probs_ext = log_probs_ext.transpose(0, 1) # [T, B, U]
    
    can_skip = torch.zeros((B, U), dtype=torch.bool, device=log_probs.device)
    if U > 2:
        can_skip[:, 2:] = (ext_targets[:, 2:] != ext_targets[:, :-2]) & (ext_targets[:, 2:] != blank)
    
    alpha = torch.full((B, U), -1e9, device=log_probs.device)
    alpha[:, 0] = log_probs_ext[0, :, 0]
    if U > 1:
        alpha[:, 1] = torch.where(target_lengths > 0, log_probs_ext[0, :, 1], torch.full_like(alpha[:, 1], -1e9))
    
    def scan_step(alpha_prev, xs):
        step_log_probs, t = xs
        
        a1 = alpha_prev
        
        a2 = torch.full_like(alpha_prev, -1e9)
        if U > 1:
            a2[:, 1:] = alpha_prev[:, :-1]
        
        a3 = torch.full_like(alpha_prev, -1e9)
        if U > 2:
            a3[:, 2:] = alpha_prev[:, :-2]
            a3 = torch.where(can_skip, a3, torch.full_like(a3, -1e9))
        
        max_a12 = torch.logaddexp(a1, a2)
        alpha_next = torch.logaddexp(max_a12, a3) + step_log_probs
        
        mask = (t < input_lengths).unsqueeze(1)
        alpha_next = torch.where(mask, alpha_next, alpha_prev)
        
        return alpha_next, alpha_next

    if T > 1:
        step_log_probs = log_probs_ext[1:]
        t = torch.arange(1, T, device=log_probs.device).unsqueeze(1).expand(T - 1, B)
        final_alpha, _ = xla_exp.scan(scan_step, alpha, (step_log_probs, t))
    else:
        final_alpha = alpha
        
    u1 = 2 * target_lengths
    u2 = 2 * target_lengths - 1
    
    alpha_u1 = torch.gather(final_alpha, 1, u1.unsqueeze(1)).squeeze(1)
    
    valid_u2 = target_lengths > 0
    clamped_u2 = torch.clamp(u2, min=0)
    alpha_u2 = torch.where(valid_u2, 
                           torch.gather(final_alpha, 1, clamped_u2.unsqueeze(1)).squeeze(1),
                           torch.full_like(alpha_u1, -1e9))
    
    final_log_prob = torch.logaddexp(alpha_u1, alpha_u2)
    return -final_log_prob


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

    # Vectorized check for consecutive identical tokens (to prevent XLA CPU sync bottleneck)
    same_as_prev = (targets[:, 1:] == targets[:, :-1]) & valid_mask[:, 1:] & valid_mask[:, :-1]
    min_ctc_len = tgt_lengths + same_as_prev.sum(dim=-1).long()

    valid_ctc = (enc_len >= min_ctc_len) & (tgt_lengths > 0) & (enc_len > 0)

    raw_tgt_lengths = tgt_lengths.clone()

    invalid_mask = ~valid_ctc
    safe_tgt_lengths = tgt_lengths.masked_fill(invalid_mask, 1)
    safe_enc_len = enc_len.masked_fill(invalid_mask, 1)

    # Mask out invalid padding from targets using UNK_ID (3) so target != blank (PAD_ID=0)
    targets = targets.masked_fill(~valid_mask, GlossVocabulary.UNK_ID)
    targets = targets.masked_fill(invalid_mask.unsqueeze(-1), GlossVocabulary.UNK_ID)

    valid_f = valid_ctc.float()
    if sample_weights is not None:
        valid_f = valid_f * sample_weights

    if IS_TPU:
        # Static XLA execution via scan, no dynamic shape recompilations
        loss_vec = _xla_ctc_loss(
            ctc_log_probs.float(),
            targets,
            safe_enc_len,
            safe_tgt_lengths,
            blank=GlossVocabulary.PAD_ID,
        )
    else:
        # Native PyTorch CTC Loss (XLA dynamically lowers to TPU natively)
        loss_vec = F.ctc_loss(
            ctc_log_probs.float().transpose(0, 1),
            targets,
            safe_enc_len,
            safe_tgt_lengths,
            blank=GlossVocabulary.PAD_ID,
            reduction='none',
            zero_infinity=True,
        )
    
    loss_vec = loss_vec.masked_fill(invalid_mask, 0.0)
    loss_vec = loss_vec / safe_tgt_lengths.float()
    local_sum = (loss_vec * valid_f).sum()
    local_weight = valid_f.sum()

    loss_ctc = _distributed_normalize(local_sum, local_weight)

    with torch.no_grad():
        ctc_eligible = (raw_tgt_lengths > 0).float().sum()
        ctc_used = valid_ctc.float().sum()
        ctc_dropped = ctc_eligible - ctc_used
        mean_enc_len = enc_len.float().mean()
        mean_tgt_len = raw_tgt_lengths.float().mean()
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
    mlm_logits: torch.Tensor,
    orig_x: torch.Tensor,
    mlm_mask: torch.Tensor,
    frame_indices: torch.Tensor = None,
) -> torch.Tensor:
    B, T = orig_x.size(0), orig_x.size(1)
    target = orig_x.reshape(B, T, -1)
    if frame_indices is not None:
        target = target.gather(
            1, frame_indices.long().unsqueeze(-1).expand(-1, -1, target.size(-1))
        )

    diff = F.smooth_l1_loss(mlm_logits, target, reduction="none")
    m = mlm_mask.unsqueeze(-1).to(diff.dtype)
    loss_sum = (diff * m).sum()
    count = m.sum() * diff.size(-1)
    return _distributed_normalize(loss_sum, count)


class ModelEMA:
    def __init__(
        self, model: nn.Module, decay_base: float = 0.90, decay_max: float = 0.9999
    ):
        self.decay_base = decay_base
        self.decay_max = decay_max
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.clone().detach()

    def update(self, model: nn.Module, progress: float = 1.0):
        with torch.no_grad():
            try:
                first_param = next(iter(model.parameters()))
                IS_XLA = "xla" in str(first_param.device).lower()
            except:
                IS_XLA = False

            if IS_XLA:
                # XLA cache breaker prevention: using dynamic python scalars in in-place ops 
                # bakes them as constants and forces a new graph compilation.
                decay = self.decay_max
            else:
                decay = self.decay_base + (self.decay_max - self.decay_base) * progress

            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Avoid .data which causes XLA memory leaks!
                    self.shadow[name].mul_(decay).add_(
                        param, alpha=1.0 - decay
                    )

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Backup the current parameters safely
                    self.backup[name] = param.clone().detach()
                    # Apply EMA parameters safely using .copy_ to preserve Dynamo tracing
                    param.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and hasattr(self, 'backup') and name in self.backup:
                    # Restore original parameters
                    param.copy_(self.backup[name])
        self.backup = {}


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
    scaler: Optional[Any] = None,
    epoch: int = 0,
    total_epochs: int = 150,
    prec_dtype: torch.dtype = torch.float16,
    is_master: bool = True,
    accum_steps: int = 4,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[float, float]:
    model.train()

    # ─── Dynamic Token Merging Scaling ───
    # Disabled dynamic Token Merging to enforce static graph shapes and prevent XLA compilation thrashing
    pass

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
    if False:
                    is_master = (int(os.environ.get("RANK", 0)) == 0) if is_xla else True

    if False:
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
    grl_alpha = float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
    label_smoothing = max(0.05, 0.15 - 0.10 * progress)
    POLY1_EPS = 1.0

    def compute_seq_loss(
        logits_f, gt_ids, valid_mask, sample_weights=None, class_weights=None, gamma=2.0
    ):
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = gt_ids.reshape(-1).clamp(0, V - 1)
        vf = valid_mask.reshape(-1).float()

        # Exclude EOS from seq loss to separate concerns
        eos_mask = tf == GlossVocabulary.EOS_ID
        vf = vf * (~eos_mask).float()

        if class_weights is not None:
            vf = vf * class_weights[tf]

        if sample_weights is not None:
            sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
            vf = vf * sw

        ce_unsmoothed = F.cross_entropy(lf, tf, reduction="none")
        p_target = torch.exp(-ce_unsmoothed).clamp(min=1e-6, max=1.0)
        focal_weight = torch.pow(1.0 - p_target, gamma)
        ce_smoothed = F.cross_entropy(lf, tf, reduction="none", label_smoothing=label_smoothing)

        poly1 = focal_weight * ce_smoothed + POLY1_EPS * (1.0 - p_target)
        return _distributed_normalize((poly1 * vf).sum(), vf.sum())

    def compute_eos_loss(logits_f, gt_ids, valid_mask, sample_weights=None, gamma=2.0):
        V = logits_f.shape[-1]
        lf = logits_f.reshape(-1, V).float()
        tf = gt_ids.reshape(-1).clamp(0, V - 1)
        vf = valid_mask.reshape(-1).float()

        # Only compute for EOS
        eos_mask = (tf == GlossVocabulary.EOS_ID).float()
        vf = vf * eos_mask

        if sample_weights is not None:
            sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
            vf = vf * sw

        ce_unsmoothed = F.cross_entropy(lf, tf, reduction="none")
        p_target = torch.exp(-ce_unsmoothed).clamp(min=1e-6, max=1.0)
        focal_weight = torch.pow(1.0 - p_target, gamma)

        # We don't apply label smoothing to EOS to force hard termination
        poly1 = focal_weight * ce_unsmoothed + POLY1_EPS * (1.0 - p_target)
        return _distributed_normalize((poly1 * vf).sum(), vf.sum())

    para_loader = loader

    total_batches = len(loader)
    min_batches = total_batches
    if False:
        min_batches = int(
            xm.mesh_reduce("min_batches", total_batches, lambda x: min(x))
        )
        try:
            ord_val = xm.get_ordinal()
        except AttributeError:
            try:
                
                ord_val = xr.global_ordinal()
            except Exception:
                ord_val = 0
        xm.set_rng_state(42 + epoch * 10000 + ord_val)

    step_start_time = time.time()
    for step_idx, batch in enumerate(para_loader, start=1):
        if step_idx == 1:
            step_start_time = time.time()
        if step_idx > min_batches:
            continue

        features, mask, labels, frame_indices = (
            batch["feature"].to(device),
            batch["mask"].to(device),
            batch.get(
                "label", torch.zeros(batch["feature"].size(0), dtype=torch.long, device=device)
            ).to(device),
            batch["frame_indices"].to(device) if "frame_indices" in batch else None,
        )
        sample_weight = batch.get(
            "sample_weight", torch.ones_like(labels, dtype=torch.float32, device=device)
        ).to(device)
        domain_tgts = batch.get(
            "domain_label", batch.get("source_id", torch.zeros_like(labels))
        ).to(device)
        has_domain = batch.get(
            "has_domain_label", torch.ones_like(domain_tgts, dtype=torch.bool)
        ).to(device)
        gloss_seq, gloss_len, has_valid_gloss, mlm_mask = (
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
        is_isolated = batch.get(
            "is_isolated", torch.ones_like(labels, dtype=torch.bool)
        ).to(device)
        if mlm_mask is not None:
            mlm_mask = mlm_mask.to(device)

        optimizer.zero_grad(set_to_none=True)

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
                mtp_logits,
                chicago_logits,
                english_logits,
                _,
                ctc_log_probs,
                vis_emb,
                sent_emb,
                proj_feats,
                _,
                aux_logits,
                enc_mask,
                pred_len,
                chicago_pred_len,
                english_pred_len,
            ) = (
                out["dec_logits"],
                out.get("mtp_logits", None),
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
                    else torch.zeros((), device=device)
                )
                


                loss_eos = (
                    compute_eos_loss(
                        dec_logits,
                        gt_tokens,
                        token_mask,
                        sample_weights=sample_weight,
                    )
                    if dec_logits is not None
                    else torch.zeros((), device=device)
                )
            else:
                loss_length = torch.zeros((), device=device)
                loss_seq = torch.zeros((), device=device)
                loss_eos = torch.zeros((), device=device)

            # --- CHICAGO LOSS (Sample-wise Masking) ---
            chicago_gt = (
                chicago_seq[:, 1:].contiguous() if chicago_seq is not None else None
            )
            chicago_valid = (
                (chicago_gt != GlossVocabulary.PAD_ID)
                if chicago_gt is not None
                else None
            )

            if chicago_pred_len is not None:
                c_valid_f = has_valid_chicago.float()
                c_valid_seq_mask = chicago_valid & (
                    chicago_gt != GlossVocabulary.EOS_ID
                )
                c_target_len = (
                    (c_valid_seq_mask * has_valid_chicago.unsqueeze(-1))
                    .sum(dim=1)
                    .float()
                )
                loss_chicago_len = F.smooth_l1_loss(
                    chicago_pred_len, c_target_len, reduction="none"
                )
                loss_chicago_len = _distributed_normalize(
                    (loss_chicago_len * sample_weight * c_valid_f).sum(),
                    (c_valid_f * sample_weight).sum(),
                )
            else:
                loss_chicago_len = torch.zeros((), device=device)

            if chicago_logits is not None:
                chi_token_mask = chicago_valid & has_valid_chicago.unsqueeze(-1)
                valid_chicago_mask = chi_token_mask & (
                    chicago_gt != GlossVocabulary.EOS_ID
                )
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
                loss_chicago = torch.zeros((), device=device)
                loss_chicago_eos = torch.zeros((), device=device)

            # --- ENGLISH LOSS (Sample-wise Masking) ---
            english_gt = (
                english_seq[:, 1:].contiguous() if english_seq is not None else None
            )
            english_valid = (
                (english_gt != GlossVocabulary.PAD_ID)
                if english_gt is not None
                else None
            )

            if english_pred_len is not None:
                e_valid_f = has_valid_english.float()
                e_valid_seq_mask = english_valid & (
                    english_gt != GlossVocabulary.EOS_ID
                )
                e_target_len = (
                    (e_valid_seq_mask * has_valid_english.unsqueeze(-1))
                    .sum(dim=1)
                    .float()
                )
                loss_english_len = F.smooth_l1_loss(
                    english_pred_len, e_target_len, reduction="none"
                )
                loss_english_len = _distributed_normalize(
                    (loss_english_len * sample_weight * e_valid_f).sum(),
                    (e_valid_f * sample_weight).sum(),
                )
            else:
                loss_english_len = torch.zeros((), device=device)

            if english_logits is not None:
                eng_token_mask = english_valid & has_valid_english.unsqueeze(-1)
                valid_english_mask = eng_token_mask & (
                    english_gt != GlossVocabulary.EOS_ID
                )
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
                loss_english = torch.zeros((), device=device)
                loss_english_eos = torch.zeros((), device=device)

            # --- AUXILIARY GROUNDING & GLOSS AUX LOSSES ---
            raw_model = model.module if hasattr(model, "module") else model
            isolated_f = is_isolated.float()

            aux_target = (labels + GlossVocabulary.OFFSET).clamp(
                0, raw_model.vocab_size - 1
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
                    & has_valid_gloss.unsqueeze(-1)
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

            if True:
                isolated_labels = torch.where(is_isolated, labels, -1)
                loss_supcon = supcon_fn(
                    proj_feats.float(), isolated_labels, sample_weight
                )
            else:
                loss_supcon = torch.zeros((), device=device)

            if out.get("domain_logits") is not None:
                has_dom_f = has_domain.float()
                d_loss_raw = F.cross_entropy(
                    out["domain_logits"], domain_tgts.long(), reduction="none"
                )
                loss_domain = torch.nan_to_num(
                    _distributed_normalize(
                        (d_loss_raw * has_dom_f).sum(), has_dom_f.sum()
                    ),
                    nan=0.0,
                )
            else:
                loss_domain = torch.zeros((), device=device)

            loss_clr = (
                _compute_mlm_loss_safe(
                    out["mlm_logits"],
                    out["orig_x"],
                    out["mlm_mask"],
                    out.get("frame_indices"),
                )
                if out["mlm_logits"] is not None
                and ("mlm_mask" in out and out["mlm_mask"] is not None)
                else torch.zeros((), device=device)
            )

            mtp_logits = out.get("mtp_logits")
            loss_mtp2 = torch.zeros((), device=device)
            loss_mtp3 = torch.zeros((), device=device)

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
                chicago_nc_t = torch.zeros((), device=device)
                chicago_nt_t = torch.zeros((), device=device)
                if chicago_logits is not None:
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
                english_nc_t = torch.zeros((), device=device)
                english_nt_t = torch.zeros((), device=device)
                if english_logits is not None:
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
            ) = forward_and_losses()

        # Ensure correct scaling for the final partial accumulation step across all microbatches in window
        rem = min_batches % accum_steps
        if rem != 0 and step_idx > (min_batches - rem):
            current_microbatches = rem
        else:
            current_microbatches = accum_steps

        loss = raw_loss / float(current_microbatches)

        # Gradient Accumulation
        rem = min_batches % accum_steps
        if rem != 0 and step_idx > (min_batches - rem):
            current_microbatches = rem
        else:
            current_microbatches = accum_steps
            
        loss = raw_loss / float(current_microbatches)

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step_idx % accum_steps == 0) or (step_idx == min_batches):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                try:
                    if (
                        not hasattr(scheduler, "total_steps")
                        or scheduler.last_epoch < scheduler.total_steps
                    ):
                        scheduler.step()
                except Exception:
                    pass

                last_log_time = time.time()
        with torch.no_grad():
            metrics_vec = torch.stack([
                raw_loss.detach(),
                l_seq.detach(),
                l_sem.detach(),
                nc_t.detach(),
                nt_t.detach(),
                l_aux.detach(),
                c_elig.detach(),
                c_used.detach(),
                c_drop.detach(),
                m_enc.detach(),
                m_tgt.detach(),
                m_min.detach(),
            ])
            
            gt_flag = batch.get("gloss_trunc", torch.zeros((1,), dtype=torch.bool, device=device))
            ct_flag = batch.get("chicago_trunc", torch.zeros((1,), dtype=torch.bool, device=device))
            et_flag = batch.get("english_trunc", torch.zeros((1,), dtype=torch.bool, device=device))

            truncs_vec = torch.stack([
                gt_flag.float().sum().detach(),
                ct_flag.float().sum().detach(),
                et_flag.float().sum().detach()
            ])

            if "tpu_metrics" not in locals():
                tpu_metrics = torch.zeros_like(metrics_vec)
                tpu_truncs = torch.zeros_like(truncs_vec)
                
            tpu_metrics += metrics_vec
            tpu_truncs += truncs_vec

            if "running_metrics" not in locals():
                running_metrics = torch.zeros(14, device="cpu")
                running_truncs = torch.zeros(3, device="cpu")

            def _update_metrics(m_vec, t_vec):
                running_metrics.add_(m_vec.cpu())
                running_truncs.add_(t_vec.cpu())

            if step_idx % 10 == 0 or step_idx == min_batches:
                if False:
                    xm.add_step_closure(_update_metrics, args=(tpu_metrics, tpu_truncs))
                else:
                    _update_metrics(tpu_metrics, tpu_truncs)
                tpu_metrics = torch.zeros_like(metrics_vec)
                tpu_truncs = torch.zeros_like(truncs_vec)
        del batch
        if "out" in locals():
            del out
            
        # Free XLA tensors to prevent OoM
        if "dec_logits" in locals(): del dec_logits
        if "raw_loss" in locals(): del raw_loss
        if "l_seq" in locals(): del l_seq
        if "l_aux" in locals(): del l_aux
        if "l_ctc" in locals(): del l_ctc
        if "l_sem" in locals(): del l_sem
        if "l_chi" in locals(): del l_chi
        if "l_eng" in locals(): del l_eng
        if "features" in locals(): del features
        if "mask" in locals(): del mask
        if "labels" in locals(): del labels
        if "gloss_seq" in locals(): del gloss_seq
        if "chicago_seq" in locals(): del chicago_seq
        if "english_seq" in locals(): del english_seq
        if "frame_indices" in locals(): del frame_indices
        if "domain_tgts" in locals(): del domain_tgts
        if "has_domain" in locals(): del has_domain
        if "mlm_mask" in locals(): del mlm_mask
        if "loss" in locals(): del loss
        if "loss_terms" in locals(): del loss_terms

        if False:
                        xm.mark_step()

        # Removed gc.collect() to prevent massive CPU stalling at high iteration speeds
        if is_xla and step_idx >= min_batches:
            if "para_loader" in locals():
                del para_loader
            import gc
            gc.collect()
            break

    if False:
        xm.mark_step()
        xm.rendezvous("end_of_epoch")
        
        # Combine the running metrics and the truncation flags into one tensor
        if "running_metrics" not in locals():
            running_metrics = torch.zeros(12, dtype=torch.float32, device=device)
        if "running_truncs" not in locals():
            running_truncs = torch.zeros(3, dtype=torch.float32, device=device)
            
        final_vec = torch.cat([
            running_metrics, 
            torch.full((1,), float(min_batches), dtype=torch.float32, device=device),
            running_truncs
        ])
        
        final_vec = xm.all_reduce(xm.REDUCE_SUM, final_vec)
        m_np = final_vec.detach().cpu().numpy()

        tracker["loss"] = float(m_np[0])
        tracker["seq"] = float(m_np[1])
        tracker["sem"] = float(m_np[2])
        tracker["corr"] = float(m_np[3])
        tracker["total"] = float(m_np[4])
        tracker["aux"] = float(m_np[5])
        tracker["ctc_eligible"] = float(m_np[6])
        tracker["ctc_used"] = float(m_np[7])
        tracker["ctc_dropped"] = float(m_np[8])
        tracker["sum_enc_len"] = float(m_np[9])
        tracker["sum_tgt_len"] = float(m_np[10])
        tracker["sum_min_ctc"] = float(m_np[11])
        
        global_batches = float(m_np[12])
        g_tr = float(m_np[13])
        c_tr = float(m_np[14])
        e_tr = float(m_np[15])
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
    import gc

    gc.collect()

    return {
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
        "ar_corr": 0.0,
        "ar_exact": 0.0,
        "ar_total": 0.0,
        "ar_seq_total": 0.0,
    }

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    if False:
                    is_master = (int(os.environ.get("RANK", 0)) == 0) if is_xla else True

    if False:
        if prec_dtype == torch.float16:
            raise ValueError(
                "TPU natively supports bfloat16 or float32 precision. Float16 is not supported on TPU."
            )
        device_type = "xla"
        use_autocast = prec_dtype == torch.bfloat16
    else:
        device_type = "cuda" if "cuda" in str(device).lower() else "cpu"
        use_autocast = "cuda" in str(device).lower() and prec_dtype != torch.float32

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

        ce_smoothed = F.cross_entropy(lf, tf, reduction="none", label_smoothing=label_smoothing)
        loss = ce_smoothed * vf
        if sample_weights is not None:
            sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
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

        ce_smoothed = F.cross_entropy(lf, tf, reduction="none", label_smoothing=label_smoothing)
        loss = ce_smoothed * vf
        if sample_weights is not None:
            sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
            loss = loss * sw
            vf = vf * sw

        return _distributed_normalize(loss.sum(), vf.sum())

    total_val_batches = len(loader)
    para_loader = loader
    import torch.distributed as dist
    if dist.is_initialized():
        _t = torch.tensor([total_val_batches], device=device)
        dist.all_reduce(_t, op=dist.ReduceOp.MIN)
        min_val_batches = int(_t.item())
    else:
        min_val_batches = total_val_batches

    with torch.no_grad():
        para_loader = loader

        for step_idx, batch in enumerate(para_loader, 1):
            if step_idx > min_val_batches:
                if False:
                    if "para_loader" in locals():
                        del para_loader
                    import gc
                    gc.collect()
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
            sample_weight = batch.get(
                "sample_weight",
                torch.ones(features.size(0), dtype=torch.float32, device=device),
            ).to(device)
            eng_trunc_flag = batch.get(
                "english_trunc",
                torch.zeros(has_valid_english.shape, dtype=torch.bool, device=device),
            ).to(device)

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
                gt_tokens = gloss_seq[:, 1:].contiguous()
                token_mask = (
                    gt_tokens != GlossVocabulary.PAD_ID
                ) & has_valid_gloss.unsqueeze(-1)
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
                    else torch.zeros((), device=device)
                )
                loss_eos = (
                    compute_eos_loss(
                        dec_logits, gt_tokens, token_mask, sample_weights=sample_weight
                    )
                    if dec_logits is not None
                    else torch.zeros((), device=device)
                )

                nc_t, nt_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                if dec_logits is not None and True:
                    preds = dec_logits.argmax(dim=-1)
                    valid_f = valid_mask.float()
                    nc_t = ((preds == gt_tokens).float() * valid_f).sum()
                    nt_t = valid_f.sum()

                c_nc_t, c_nt_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                loss_chi = torch.zeros((), device=device)
                loss_chi_eos = torch.zeros((), device=device)
                if chicago_logits is not None and True:
                    chicago_gt = chicago_seq[:, 1:].contiguous()
                    c_preds = chicago_logits.argmax(dim=-1)
                    chi_token_mask = (
                        chicago_gt != GlossVocabulary.PAD_ID
                    ) & has_valid_chicago.unsqueeze(-1)
                    c_valid_mask = chi_token_mask & (
                        chicago_gt != GlossVocabulary.EOS_ID
                    )
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

                e_nc_t, e_nt_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                loss_eng = torch.zeros((), device=device)
                loss_eng_eos = torch.zeros((), device=device)
                if english_logits is not None and True:
                    english_gt = english_seq[:, 1:].contiguous()
                    e_preds = english_logits.argmax(dim=-1)
                    eng_token_mask = (
                        english_gt != GlossVocabulary.PAD_ID
                    ) & has_valid_english.unsqueeze(-1)
                    e_valid_mask = eng_token_mask & (
                        english_gt != GlossVocabulary.EOS_ID
                    )
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

                e_trunc_c, e_trunc_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                if True:
                    e_trunc_c = (
                        eng_trunc_flag.float() * has_valid_english.float()
                    ).sum()
                    e_trunc_t = has_valid_english.float().sum()

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
                }
                raw_loss = loss_wrapper(loss_terms)
                # --- Autoregressive Generation ---
                ar_nc_t = torch.zeros((), device=device)
                ar_exact = torch.zeros((), device=device)
                ar_total = torch.zeros((), device=device)
                ar_seq_total = torch.zeros((), device=device)

                if step_idx <= 2 and isinstance(out, dict) and "enc_seq" in out:
                    # Free heavy intermediate outputs and local references before 64-step AR loop to conserve HBM
                    for k in [
                        "english_logits",
                        "chicago_logits",
                        "ctc_log_probs",
                        "dec_logits",
                        "dec_hidden",
                    ]:
                        if k in out:
                            del out[k]
                    dec_logits = None
                    chicago_logits = None
                    english_logits = None

                    gen_ids = model.generate(
                        features,
                        mask=mask,
                        max_new_tokens=64,
                        task="gloss",
                        temperature=1.0,
                        top_k=0,
                        use_mtp_speculative=False,
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
                    ar_nc_t = ((dec_preds == gt_tokens).float() * valid_f).sum()
                    ar_total = valid_f.sum()

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
                ) = forward_and_losses()

            with torch.no_grad():
                metrics_vec = torch.stack([
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
                ])
                
                if "running_val_metrics" not in locals():
                    running_val_metrics = torch.zeros_like(metrics_vec)
                running_val_metrics.add_(metrics_vec)

            if is_master and (step_idx % 50 == 0 or step_idx == min_val_batches):
                print(
                    f"  [Val Step {step_idx:04d}/{min_val_batches:04d}] Loss: {float(raw_loss.detach()):.4f}",
                    flush=True,
                )

            if False:
                                xm.mark_step()

            del batch
            
            # Free validation tensors
            if "forward_and_losses" in locals(): del forward_and_losses
            if "raw_loss" in locals(): del raw_loss
            if "l_chi" in locals(): del l_chi
            if "l_eng" in locals(): del l_eng
            if "features" in locals(): del features
            if "mask" in locals(): del mask
            if "frame_indices" in locals(): del frame_indices
            if "gloss_seq" in locals(): del gloss_seq
            if "chicago_seq" in locals(): del chicago_seq
            if "english_seq" in locals(): del english_seq
            if "has_valid_gloss" in locals(): del has_valid_gloss
            if "has_valid_chicago" in locals(): del has_valid_chicago
            if "has_valid_english" in locals(): del has_valid_english
            if "dec_preds" in locals(): del dec_preds

    if False:
        xm.rendezvous("validate_metrics")
        
        if "running_val_metrics" not in locals():
            running_val_metrics = torch.zeros(15, dtype=torch.float32, device=device)
            
        val_vec = torch.cat([
            running_val_metrics,
            torch.tensor([float(min_val_batches)], dtype=torch.float32, device=device)
        ])
        
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
        step_idx = float(v_np[15])
    else:
        step_idx = float(min_val_batches)

    val_loss = tracker["loss"] / float(max(1, step_idx))
    val_acc = tracker["corr"] / max(1.0, tracker["total"])
    val_chi_acc = tracker["chi_corr"] / max(1.0, tracker["chi_total"])
    val_eng_acc = tracker["eng_corr"] / max(1.0, tracker["eng_total"])
    val_ar_acc = tracker["ar_corr"] / max(tracker["ar_total"], 1.0)
    val_ar_exact = tracker["ar_exact"] / max(tracker["ar_seq_total"], 1.0)

    if is_master:
        print(
            f"[Validation Epoch {epoch}] TotalLoss: {val_loss:.4f} | "
            f"GlossAcc(TF): {val_acc*100:.2f}% | GlossAcc(AR): {val_ar_acc*100:.2f}% | ExactMatch(AR): {val_ar_exact*100:.2f}% | "
            f"ChiAcc: {val_chi_acc*100:.2f}% | EngAcc: {val_eng_acc*100:.2f}%",
            flush=True,
        )

    if "para_loader" in locals():
        del para_loader
    import gc

    gc.collect()

    return {
        "loss": val_loss,
        "gloss_acc": val_acc * 100.0,
        "ar_acc": val_ar_acc * 100.0,
        "ar_exact": val_ar_exact * 100.0,
        "chicago_acc": val_chi_acc * 100.0,
        "english_acc": val_eng_acc * 100.0,
    }


def _tpu_worker_fn(rank, args):
    if IS_TPU:
        try:
                        # Start the profiler server on port 9012 (Master only)
            server = xp.start_server(9012)
        except Exception:
            pass

                
        world_size = xr.world_size()
        assert rank < world_size
        is_master = rank == 0
        if is_master:
            print(f"PJRT TPU runtime initialized. World size: {world_size}", flush=True)
    else:
        world_size = 1
        is_master = True

    try:
        if IS_TPU:
            try:
                
                device = torch_xla.device()
            except Exception:
                device = torch.device(f"cuda:{index}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        print(f"FAILED TO INITIALIZE TPU OR GET DEVICE: {e}", flush=True)
        time.sleep(2)
        os._exit(1)

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not list(data_dir.glob("*.pt")):
        candidate_dirs = [
            Path("./asl_preprocessed_phase1"),
            Path("/kaggle/input/asl-preprocessed-phase1"),
            Path("/kaggle/input/frakenstein-asl/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl"),
            Path.cwd(),
        ]
        for cd in candidate_dirs:
            if cd.exists() and list(cd.glob("*.pt")):
                data_dir = cd
                if is_master:
                    print(f"[INFO] Auto-resolved dataset directory to: {data_dir}", flush=True)
                break

    # Auto Memory Guard: Cap per-core micro-batch size to 16 for large sequence lengths / d-model to guarantee fitting inside 16GB HBM
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

    # TPU System RAM Guard: Cap num_dataloader_workers to 1 on multi-core TPU to prevent PyTorch IPC subprocess duplication RAM blowouts
    effective_num_dl_workers = min(1, args.num_dataloader_workers) if IS_TPU else args.num_dataloader_workers
    if IS_TPU and args.num_dataloader_workers > 1 and is_master:
        print(
            "[INFO] TPU System RAM Guard: Capped num_dataloader_workers to 1 per core on TPU. "
            "This provides asynchronous background pre-fetching without exceeding Host RAM limits.",
            flush=True,
        )

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
    )

    label_to_idx = getattr(train_loader.dataset, "label_to_idx", {})
    if not label_to_idx:
        possible_dirs = [
            data_dir,
            Path("./asl_preprocessed_phase1"),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"
            ),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"
            ),
            Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl"),
            Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl/asl_preprocessed_phase1"),
            Path("/kaggle/input/frakenstein-asl"),
            Path("/kaggle/input/asl-preprocessed-phase1"),
            Path.cwd(),
        ]
        possible_filenames = [
            "vocabulary_mapping_global.json",
            "vocabulary_mapping_train.json",
            "vocab_map.json",
            "metadata.json",
        ]
        for d in possible_dirs:
            if d.exists():
                for fn in possible_filenames:
                    vp = d / fn
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

            if IS_TPU:
                
                raw_counts = xm.all_reduce(xm.REDUCE_SUM, raw_counts)

            # Move raw_counts to CPU memory ONCE to avoid 23,000+ synchronous TPU .item() host transfers
            raw_counts_np = raw_counts.detach().cpu().numpy()
            w_vec_np = np.ones(len(vocab), dtype=np.float32)

            offset = GlossVocabulary.OFFSET
            num_classes = len(vocab) - offset
            valid_slice = raw_counts_np[:num_classes]
            max_c = max(1.0, float(raw_counts_np.max()))

            nz_mask = valid_slice > 0
            if np.any(nz_mask):
                w_vec_np[offset : offset + num_classes][nz_mask] = np.clip(
                    (max_c / valid_slice[nz_mask]) ** 0.35, 1.0, 10.0
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
    if IS_TPU:
        
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
        vocab_size=GlossVocabulary.VOCAB_SIZE,
        d_enc=args.d_model,
        d_dec=args.d_model,
        nhead_enc=args.nhead,
        nhead_dec=args.nhead,
        num_enc_layers=args.num_layers,
        num_dec_layers=args.num_layers,
        dropout=args.dropout,
        max_enc_len=args.max_len,
        english_vocab_size=len(english_vocab),
        label_to_idx=label_to_idx,
        csv_path=asl_lex_csv if asl_lex_csv.exists() else None,
    ).to(device)

    if IS_TPU:
        xm.broadcast_master_param(model)

    if USE_DYNAMO_COMPILE and hasattr(torch, "compile"):
        if is_master:
            print(
                "[*] JIT Compiling model with PyTorch Inductor (torch.compile)...",
                flush=True,
            )
        try:
            model = (
                torch.compile(model, backend="openxla")
                if IS_TPU
                else torch.compile(model)
            )
        except Exception as _e:
            if is_master:
                print(f"[!] Warning: torch.compile fallback: {_e}", flush=True)

    loss_wrapper = HomoscedasticLossWrapper().to(device)
    if IS_TPU:
        xm.broadcast_master_param(loss_wrapper)

    supcon_fn = SupervisedContrastiveLoss().to(device)

    global_min_batches = len(train_loader)
    if IS_TPU:
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
        div_factor=25.0,
        final_div_factor=5.0,
    )

    scaler = None
    if args.precision == "float16" and "cuda" in str(device).lower():
        scaler = torch.amp.GradScaler("cuda")

    start_epoch = 1
    if hasattr(args, "resume") and args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)

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
        ema_state_dict_to_load = ckpt.get("ema_state_dict", None)
        del ckpt
        import gc

        gc.collect()

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
    if "ema_state_dict_to_load" in locals() and ema_state_dict_to_load is not None:
        for k, v in ema_state_dict_to_load.items():
            if k in ema.shadow:
                ema.shadow[k].copy_(v.to(ema.shadow[k].device))
        if is_master:
            print("[+] Restored EMA state from checkpoint", flush=True)
        del ema_state_dict_to_load
        gc.collect()

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if hasattr(train_loader.dataset, "set_epoch"):
                train_loader.dataset.set_epoch(epoch)
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
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
                    f"[Validation Epoch {epoch}] SeqLoss: {val_metrics['loss']:.4f} | TokenAcc(TF): {val_metrics['gloss_acc']:.2f}% | TokenAcc(AR): {val_metrics['ar_acc']:.2f}% | ExactMatch(AR): {val_metrics['ar_exact']:.2f}%",
                    flush=True,
                )

            # Save checkpoint every 5 epochs or on final epoch to save memory and disk quota
            if (epoch % 5 == 0 or epoch == args.epochs):
                import random as py_random

                ckpt_path = save_dir / f"asl_model_epoch_{epoch}.pt"
                latest_path = save_dir / "asl_model_latest.pt"

                if is_master:
                    cpu_state = {
                        "epoch": epoch,
                        "model_state_dict": raw_m.state_dict(),
                        "ema_state_dict": ema.shadow if ema is not None else None,
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
                    }
                    torch.save(cpu_state, str(ckpt_path))
                    torch.save(cpu_state, str(latest_path))
                    del cpu_state
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

            # Explicit garbage collection of massive dicts & flush XLA IR graph
            if cpu_state is not None:
                del cpu_state
                cpu_state = None

            gc.collect()
            gc.collect()

    except Exception as e:
        import traceback
        import sys

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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-dataloader-workers", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument(
        "--enable-aux-decoders",
        action="store_true",
        default=True,
        help="Enable auxiliary Chicago/English decoders for multi-task learning",
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
        action="store_true",
        help="Use ASLStreamedDataset (IterableDataset) for zero-RAM startup",
    )
    parser.add_argument("--save-dir", type=str, default="/tmp/checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--asl-lex-csv", type=str, default="/home/binhhanh409/signdata.csv"
    )
    args = parser.parse_args()

    # Scale learning rate to simulate accumulation step batch sizes if maintaining unrolled mega-graph
    if args.accum_steps > 1:
        args.lr = args.lr / float(args.accum_steps)
        print(f"[*] Simulating accum_steps={args.accum_steps} with unrolled mega-graph. Scaling LR to {args.lr:.2e}")

    # Removed global environment variables for precision
    pass

    if "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        _tpu_worker_fn(rank, args)
    else:
        _tpu_worker_fn(0, args)


if __name__ == "__main__":
    main()
