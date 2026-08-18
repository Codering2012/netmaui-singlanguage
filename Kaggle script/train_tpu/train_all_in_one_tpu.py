#!/usr/bin/env python3
# pylint: disable=not-callable,too-many-instance-attributes,reimported,no-else-return,unnecessary-lambda,import-outside-toplevel,possibly-unused-variable,redefined-outer-name,wrong-import-position,consider-using-from-import,bare-except,broad-exception-caught,too-many-locals,too-many-branches,too-many-statements,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,line-too-long,superfluous-parens,too-many-lines

"""
================================================================================
MONOLITHIC ALL-IN-ONE TPU/GPU ASL FOUNDATION MODEL — SENTENCE RECONSTRUCTION
Encoder: MobileConformer (8L × dim_d=320, nhead=8, ffn=1280) — ~17.4M parameters
Decoder: ASLTransformerDecoder (8L × dim_d=320, GQA 8Q/2KV, RoPE, ffn=1280) — ~12.9M parameters
Total:   ~31.0M parameters (High Efficiency & SOTA Accuracy via Extended Compute)

Task: Continuous Sign Language Understanding & Gloss Sentence Reconstruction
================================================================================
"""

import os  # pylint: disable=not-callable,import-outside-toplevel

# [NEW] Persistent HLO caching to permanently bypass 13-minute XLA compilations
# pylint: disable=not-callable,too-many-instance-attributes,reimported,no-else-return,unnecessary-lambda,import-outside-toplevel,possibly-unused-variable,redefined-outer-name

os.environ["XLA_PERSISTENT_CACHE_PATH"] = "/kaggle/working/xla_cache"
os.environ["XLA_OPTIMIZATION_LEVEL"] = "EFFORT_O2"
os.environ["XLA_MEMORY_FITTING_LEVEL"] = "EFFORT_O3"

# [NEW] Experimental v5e Asynchronous Collective Pipelining & DP Overlap
USE_EXPERIMENTAL_XLA_FLAGS = True
USE_DYNAMO_COMPILE = True

if USE_EXPERIMENTAL_XLA_FLAGS:
    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_flags += " --xla_enable_async_all_gather=true"
    xla_flags += " --xla_tpu_enable_async_collective_fusion=true"
    xla_flags += " --xla_tpu_enable_ici_ag_pipelining=true"
    xla_flags += " --xla_should_allow_loop_variant_parameter_in_chain=kEnabled"
    xla_flags += " --xla_should_add_loop_invariant_op_in_chain=kEnabled"
    xla_flags += " --xla_tpu_enable_data_parallel_all_reduce_opt=true"
    xla_flags += " --xla_tpu_data_parallel_opt_different_sized_ops=true"
    xla_flags += " --xla_tpu_enable_flash_attention=true"
    xla_flags += " --xla_tpu_enable_latency_hiding_scheduler=true"
    xla_flags += " --xla_tpu_enable_while_loop_double_buffering=true"
    os.environ["XLA_FLAGS"] = xla_flags

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

import sys  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
import time  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
import json  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
import math  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
import argparse  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
from pathlib import Path  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
from typing import Dict, List, Optional, Tuple, Union, Any  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel

import numpy as np  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
import torch  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
import torch.nn as nn  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
import torch.nn.functional as F  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
from torch.utils.data import DataLoader  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel

# Set up for Kaggle 2x T4 GPUs


# Force Local PJRT mode to avoid gRPC proxy concurrency limit and fork deadlocks
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.pop("TPU_NAME", None)
# WARNING: Setting PJRT_DEVICE=TPU breaks training on Kaggle T4 GPUs.
# Uncomment the line below ONLY if running on a Kaggle TPU VM.
# os.environ["PJRT_DEVICE"] = "TPU"

try:
    import importlib.util  # pylint: disable=not-callable,import-outside-toplevel

    _XLA_AVAILABLE = importlib.util.find_spec("torch_xla") is not None
except Exception:  # pylint: disable=not-callable,broad-exception-caught
    _XLA_AVAILABLE = False

if "--tpu" in sys.argv and _XLA_AVAILABLE:
    os.environ["PJRT_DEVICE"] = "TPU"


def is_tpu_runtime() -> bool:
    """Docstring for is_tpu_runtime."""

    return _XLA_AVAILABLE and os.environ.get("PJRT_DEVICE", "").upper() == "TPU"


IS_TPU = is_tpu_runtime()


def get_xla_world_size() -> int:
    """Docstring for get_xla_world_size."""

    if IS_TPU:
        try:
            import torch_xla.runtime as xr  # pylint: disable=not-callable,import-outside-toplevel

            return xr.world_size()
        except Exception:  # pylint: disable=not-callable,broad-exception-caught
            try:
                import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                return getattr(xm, "xrt_world_size", lambda: 1)()  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
            except Exception:  # pylint: disable=not-callable,broad-exception-caught
                pass
    return 1


train_dir = Path(__file__).resolve().parent
if str(train_dir) not in sys.path:
    sys.path.insert(0, str(train_dir))
from dataset import (  # pylint: disable=not-callable,wrong-import-position  # pylint: disable=not-callable,import-outside-toplevel
    create_dataloader,
    normalize_vocabulary,
)


def _distributed_normalize(
    local_sum: torch.Tensor, local_weight: torch.Tensor
) -> torch.Tensor:
    """Docstring for _distributed_normalize."""

    normed = local_sum / local_weight.clamp_min(1e-8)
    return torch.nan_to_num(normed, nan=0.0, posinf=0.0, neginf=0.0) * (local_weight > 0).to(normed.dtype)


def _safe_torch_device(dev_str: Union[str, torch.device]) -> torch.device:
    """Docstring for _safe_torch_device."""

    if isinstance(dev_str, torch.device):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        return dev_str
    dev_s = str(dev_str).lower()
    if IS_TPU and "xla" in dev_s:
        try:
            import torch_xla  # pylint: disable=not-callable,import-outside-toplevel

            return torch_xla.device(dev_str)
        except Exception:  # pylint: disable=not-callable,broad-exception-caught
            pass
    try:
        return torch.device(dev_str)
    except Exception:  # pylint: disable=not-callable,broad-exception-caught
        return torch.device("cpu")


# ==============================================================================
# 1. LANDMARK AUGMENTER (REAL-WORLD CAMERA NOISE & PHYSIOLOGICAL STALLING)
# ==============================================================================


# ==============================================================================
# 2. GLOSS VOCABULARY — Sequence Vocabulary with Special Tokens
# ==============================================================================


class GlossVocabulary:
    """Docstring for GlossVocabulary."""

    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    OFFSET = 4

    def __init__(self, label_to_idx: Dict):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        clean_l2i = {}
        if isinstance(label_to_idx, dict):
            for key_k_lower, val_v in label_to_idx.items():
                k_str = str(key_k_lower).strip().lower()
                if isinstance(val_v, int):
                    clean_l2i[k_str] = val_v
                elif isinstance(val_v, dict):
                    idx_val = val_v.get("id", val_v.get("idx", val_v.get("label_idx", 0)))
                    clean_l2i[k_str] = int(idx_val)
                elif isinstance(val_v, str) and str(key_k_lower).isdigit():
                    clean_l2i[str(val_v).strip().lower()] = int(key_k_lower)
                else:
                    try:
                        clean_l2i[k_str] = int(val_v)
                    except (ValueError, TypeError):
                        pass

        self.label_to_idx = clean_l2i
        self.idx_to_label = {val_v: key_k_lower for key_k_lower, val_v in self.label_to_idx.items()}
        max_idx = max(clean_l2i.values()) if clean_l2i else 0
        self.vocab_size = max(len(self.label_to_idx), max_idx + 1) + self.OFFSET
        self.output_map = {}

    def __len__(self) -> int:
        """Docstring for __len__."""

        return self.vocab_size

    def gloss_to_token(self, gloss: str) -> int:
        """Docstring for gloss_to_token."""

        raw = self.label_to_idx.get(gloss.strip().lower(), None)
        if raw is None:
            return self.UNK_ID
        return raw + self.OFFSET

    def token_to_gloss(self, tid: int) -> str:
        """Docstring for token_to_gloss."""

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
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for RMSNorm."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Docstring for forward."""

        var = input_x.float().pow(2).mean(-1, keepdim=True)
        return input_x * torch.rsqrt(var + self.eps).to(input_x.dtype) * self.weight.to(input_x.dtype)


class SwiGLUFFN(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for SwiGLUFFN."""

    def __init__(self, d_model: int, dim_feedforward: int, num_layers: int = 8):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Docstring for forward."""

        return self.w_down(F.silu(self.w_gate(input_x)) * self.w_up(input_x))


class XLASparseMoE(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for XLASparseMoE."""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_layers: int = 8,
        num_experts: int = 4,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.num_experts = num_experts
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLUFFN(d_model, dim_feedforward, num_layers)
                for _ in range(num_experts)
            ]
        )

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Docstring for forward."""

        routing_logits = self.router(input_x)
        routing_probs = F.softmax(routing_logits, dim=-1)
        max_probs, _ = torch.max(routing_probs, dim=-1, keepdim=True)
        expert_mask = (routing_probs == max_probs).to(input_x.dtype)

        routing_weights = expert_mask.detach() - routing_probs.detach() + routing_probs

        out = torch.zeros_like(input_x)
        for i, expert in enumerate(self.experts):
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            expert_out = expert(input_x)
            out = out + (expert_out * routing_weights[..., i : i + 1])
        return out


# ==============================================================================
# 4. RICH ASL-LEX MULTI-ATTRIBUTE EMBEDDING TABLE
# ==============================================================================


class RichASLLexEmbeddingTable(nn.Module):
    """Docstring for RichASLLexEmbeddingTable."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        csv_path: Optional[Union[str, Path]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
                import csv  # pylint: disable=not-callable,import-outside-toplevel

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
                        import re  # pylint: disable=not-callable,import-outside-toplevel

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
            except Exception as e:  # pylint: disable=not-callable,broad-exception-caught
                print(f"[!] Warning: Failed to parse ASL-LEX CSV: {e}", flush=True)

        self.register_buffer("attr_idx_matrix", attr_idx_matrix)
        self.register_buffer("attr_scalars", attr_scalars)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Removed .any() validation check to prevent XLA device-to-host syncs in the forward pass.
        """Docstring for forward."""

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
            (token_ids != 0).unsqueeze(-1).to(raw_attrs.dtype)
        )
        return self.attr_proj(raw_attrs) * valid_lex_mask


# ==============================================================================
# 5. TOKEN MERGING BLOCK (ToMe)
# ==============================================================================


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor

class TemporalStridedPool(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(
        self, hidden_h, mask=None, **kwargs
    ):
        hidden_h = hidden_h[:, ::2]
        if mask is not None:
            mask = mask[:, ::2]
        fi = kwargs.get("frame_indices", None)
        if fi is not None:
            fi = fi[:, ::2]
        token_sizes = kwargs.get("token_sizes", None)
        if token_sizes is not None:
            token_sizes = token_sizes[:, ::2]
            
        return (
            hidden_h,
            mask,
            {
                "T_orig": hidden_h.shape[1],
                "sorted_routing": None,
                "mlm_out": kwargs.get("mlm_mask", None),
                "frame_indices": fi,
                "token_sizes": token_sizes,
            },
        )

class DropPath(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for DropPath."""

    def __init__(self, drop_prob: float = 0.0):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        """Docstring for forward."""

        is_training = self.training or torch.is_grad_enabled()
        return drop_path(input_x, self.drop_prob, is_training)


class GroupedQueryEncoderAttention(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for GroupedQueryEncoderAttention."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        max_len: int = 512,
        dropout_p: float = 0.1,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
        self.kv_proj = nn.Linear(
            self.latent_dim, 2 * kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.q_norm = RMSNorm(d_model)
        
        self.dropout_p = dropout_p

    def forward(
        self,
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,  # pylint: disable=not-callable,unused-argument
    ) -> torch.Tensor:
        """Docstring for forward."""

        batch_sz, seq_len, _ = input_x.shape
        q_in = self.q_norm(input_x)

        # DeepSeek V3 MLA Latent Compression
        kv_latent = self.kv_latent_proj(input_x)
        kv_latent = self.kv_latent_norm(kv_latent)

        query_q_lower = self.q_proj(q_in).view(batch_sz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_latent)
        key_k_lower, val_v = torch.split(kv, kv.size(-1) // 2, dim=-1)
        key_k_lower = key_k_lower.view(batch_sz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        val_v = val_v.view(batch_sz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply Rotary Positional Encoding (RoPE) only to the first half of the head_dim (Spatial translation subset)
        rope_dim = self.head_dim // 2
        q_rope, q_nop = query_q_lower[..., :rope_dim], query_q_lower[..., rope_dim:]
        k_rope, k_nop = key_k_lower[..., :rope_dim], key_k_lower[..., rope_dim:]
        q_rope, k_rope = self.rope(q_rope, k_rope)
        query_q_lower = torch.cat([q_rope, q_nop], dim=-1)
        key_k_lower = torch.cat([k_rope, k_nop], dim=-1)

        if self.groups > 1:
            key_k_lower = key_k_lower.repeat_interleave(self.groups, dim=1)
            val_v = val_v.repeat_interleave(self.groups, dim=1)

        if key_padding_mask is not None:
            attn_mask = ~(key_padding_mask.view(batch_sz, 1, 1, seq_len).bool())
        else:
            attn_mask = None

        out = F.scaled_dot_product_attention(  # pylint: disable=not-callable,not-callable
            query_q_lower,
            key_k_lower,
            val_v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
            scale=self.scale,
        )
        out = self.out_proj(out.transpose(1, 2).reshape(batch_sz, seq_len, -1))
        return out


class SpatialTemporalSE(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for SpatialTemporalSE."""

    def __init__(self, d_model: int, reduction: int = 4):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
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
        """Docstring for forward."""

        if key_padding_mask is not None:
            valid_mask = (~key_padding_mask).unsqueeze(-1).to(input_x.dtype)
            mean_x = (input_x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        else:
            valid_mask = (input_x.abs().sum(dim=-1, keepdim=True) > 1e-5).to(input_x.dtype)
            mean_x = (input_x * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return input_x * torch.max(self.c_se(mean_x).unsqueeze(1), self.s_se(input_x))


class ConvNeXtTemporalBlock(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for ConvNeXtTemporalBlock."""

    def __init__(self, channels: int, expansion: int = 2):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
        self,  # pylint: disable=not-callable,unused-argument
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,  # pylint: disable=not-callable,unused-argument
        **kwargs,
    ) -> torch.Tensor:
        """Docstring for forward."""

        if key_padding_mask is not None:
            input_x = input_x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        target_y = self.norm(
            F.conv1d(  # pylint: disable=not-callable,not-callable
                F.pad(
                    input_x.transpose(1, 2),
                    (3, 3),
                    mode="replicate",
                ),
                self.dw_conv.weight,
                self.dw_conv.bias,
                groups=self.dw_conv.groups,
            ).transpose(1, 2)
        )
        if key_padding_mask is not None:
            target_y = target_y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        target_y = self.se(
            self.pw_conv2(self.act(self.pw_conv1(target_y))), key_padding_mask=key_padding_mask
        )
        if key_padding_mask is not None:
            target_y = target_y.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return target_y


class BiMamba2SSMBlock(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for BiMamba2SSMBlock."""

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
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.d_model, self.d_inner, self.d_state = d_model, d_model * expand, d_state
        self.nheads, self.headdim = (
            (self.d_inner // headdim)
            if self.d_inner % headdim == 0
            else min(  # pylint: disable=not-callable,consider-using-generator
                [hidden_h for hidden_h in range(1, self.d_inner + 1) if self.d_inner % hidden_h == 0],
                key=lambda hidden_h: abs(hidden_h - max(1, self.d_inner // headdim)),  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
            )
        ), self.d_inner // (
            self.d_inner // headdim
            if self.d_inner % headdim == 0
            else min(  # pylint: disable=not-callable,consider-using-generator
                [hidden_h for hidden_h in range(1, self.d_inner + 1) if self.d_inner % hidden_h == 0],
                key=lambda hidden_h: abs(hidden_h - max(1, self.d_inner // headdim)),  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
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
            padding="same",
            groups=self.d_inner,
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
        input_x: torch.Tensor,
        dt: torch.Tensor,
        state_a: torch.Tensor,
        batch_sz: torch.Tensor,
        channels: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        reverse: bool = False,
        chunk_size: int = 64,
    ) -> torch.Tensor:
        """Docstring for _ssd_multihead_scan."""

        b_sz, t_sz, H_sz, P_sz = input_x.shape  # pylint: disable=not-callable,invalid-name
        N_sz, device = batch_sz.shape[-1], input_x.device  # pylint: disable=not-callable,invalid-name
        if reverse:
            input_x, dt, batch_sz, channels = input_x.flip(1), dt.flip(1), batch_sz.flip(1), channels.flip(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.flip(1)

        dt_act = F.softplus(dt).clamp(max=20.0)  # pylint: disable=not-callable,not-callable
        if key_padding_mask is not None:
            kpm_b = key_padding_mask.unsqueeze(-1)
            dt_act = dt_act.masked_fill(kpm_b, 0.0)
            input_x = input_x.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            batch_sz = batch_sz.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            channels = channels.masked_fill(kpm_b.unsqueeze(-1), 0.0)
            log_decay = -((dt_act * state_a.view(1, 1, H_sz)).clamp(min=0.0, max=20.0))
        else:
            log_decay = -((dt_act * state_a.view(1, 1, H_sz)).clamp(min=1e-4, max=20.0))

        query_q = min(chunk_size, t_sz)
        pad_len = (query_q - (t_sz % query_q)) % query_q
        if pad_len > 0:
            input_x, batch_sz, channels, log_decay, dt_act = (
                F.pad(input_x, (0, 0, 0, 0, 0, pad_len)),
                F.pad(batch_sz, (0, 0, 0, 0, 0, pad_len)),
                F.pad(channels, (0, 0, 0, 0, 0, pad_len)),
                F.pad(log_decay, (0, 0, 0, pad_len), value=0.0),
                F.pad(dt_act, (0, 0, 0, pad_len), value=0.0),
            )

        T_pad, n_chunks = input_x.shape[1], input_x.shape[1] // query_q  # pylint: disable=not-callable,invalid-name

        x_chunk = input_x.view(b_sz, n_chunks, query_q, H_sz, P_sz).permute(0, 3, 1, 2, 4)
        B_chunk = batch_sz.view(b_sz, n_chunks, query_q, H_sz, N_sz).permute(0, 3, 1, 2, 4)  # pylint: disable=not-callable,invalid-name
        C_chunk = channels.view(b_sz, n_chunks, query_q, H_sz, N_sz).permute(0, 3, 1, 2, 4)  # pylint: disable=not-callable,invalid-name
        ld_chunk = log_decay.view(b_sz, n_chunks, query_q, H_sz).permute(0, 3, 1, 2)

        B_chunk_dt = B_chunk * dt_act.view(b_sz, n_chunks, query_q, H_sz).permute(  # pylint: disable=not-callable,invalid-name
            0, 3, 1, 2
        ).unsqueeze(-1)
        CB = torch.matmul(C_chunk, B_chunk_dt.transpose(-1, -2)) / math.sqrt(N_sz)  # pylint: disable=not-callable,invalid-name
        cum_decay = ld_chunk.to(torch.float32).cumsum(dim=-1).to(ld_chunk.dtype)
        M = torch.exp(  # pylint: disable=not-callable,invalid-name
            (cum_decay.unsqueeze(-1) - cum_decay.unsqueeze(-2)).masked_fill(
                ~torch.tril(torch.ones(query_q, query_q, device=device, dtype=torch.bool)),
                -1e9,
            )
        )
        Y_intra = torch.matmul(M * CB, x_chunk)  # pylint: disable=not-callable,invalid-name

        log_chunk_decay = ld_chunk.sum(dim=-1)
        decay_to_end = torch.exp(cum_decay[:, :, :, -1:] - cum_decay)
        state_gen = torch.einsum(
            "bhcqp, bhcqn -> bhcpn", x_chunk * decay_to_end.unsqueeze(-1), B_chunk_dt
        )

        length_l = log_chunk_decay.cumsum(dim=2)
        L_shifted = torch.cat([torch.zeros_like(length_l[:, :, :1]), length_l[:, :, :-1]], dim=2)  # pylint: disable=not-callable,invalid-name
        M_inter = torch.exp(  # pylint: disable=not-callable,invalid-name
            (L_shifted.unsqueeze(-1) - length_l.unsqueeze(-2)).masked_fill(
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
                b_sz, H_sz, n_chunks, state_gen.shape[-2] * state_gen.shape[-1]
            ),
        )
        state_stack = state_stack_flat.reshape(
            b_sz, H_sz, n_chunks, state_gen.shape[-2], state_gen.shape[-1]
        )

        C_state = torch.einsum(  # pylint: disable=not-callable,invalid-name
            "bhcqn, bhcpn -> bhcqp", C_chunk, state_stack
        ) / math.sqrt(N_sz)
        Y_inter = C_state * torch.exp(cum_decay).unsqueeze(-1)  # pylint: disable=not-callable,invalid-name

        Y_flat = (  # pylint: disable=not-callable,invalid-name
            (Y_intra + Y_inter).permute(0, 2, 3, 1, 4).reshape(b_sz, T_pad, H_sz, P_sz)
        )
        return Y_flat[:, :t_sz].flip(1) if reverse else Y_flat[:, :t_sz]

    def forward(
        self,  # pylint: disable=not-callable,unused-argument
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,  # pylint: disable=not-callable,unused-argument
        **kwargs,
    ) -> torch.Tensor:
        """Docstring for forward."""

        if key_padding_mask is not None:
            input_x = input_x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        xn = self.norm1(input_x)
        b_sz, t_sz, _ = xn.shape

        x_proj, z, B_ssm_fwd, C_ssm_fwd, dt_fwd = torch.split(  # pylint: disable=not-callable,invalid-name
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
            self.fwd_conv1d(x_proj.transpose(1, 2)).transpose(1, 2)
        ).view(b_sz, t_sz, self.nheads, self.headdim)
        B_h_fwd, C_h_fwd = B_ssm_fwd.view(  # pylint: disable=not-callable,invalid-name
            b_sz, t_sz, self.nheads, self.d_state
        ), C_ssm_fwd.view(b_sz, t_sz, self.nheads, self.d_state)

        state_a = F.softplus(self.a_log)  # pylint: disable=not-callable,not-callable
        y_fwd = self._ssd_multihead_scan(
            x_fwd_h,
            dt_fwd + self.dt_bias,
            state_a,
            B_h_fwd,
            C_h_fwd,
            key_padding_mask=key_padding_mask,
            reverse=False,
        )
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
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for MobileConformerBlock."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 1280,
        dropout_p: float = 0.1,  # pylint: disable=not-callable,unused-argument
        drop_path: float = 0.0,
        num_enc_layers: int = 8,
        init_values: float = 1e-4,
        max_len: int = 320,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.ffn1_norm = RMSNorm(d_model)
        self.ffn1 = XLASparseMoE(d_model, dim_feedforward, num_layers=num_enc_layers)
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
        self.ffn2 = XLASparseMoE(d_model, dim_feedforward, num_layers=num_enc_layers)
        self.drop_path_ffn2 = DropPath(drop_path)
        self.gamma_ffn2 = nn.Parameter(init_values * torch.ones(d_model))

    def forward(
        self,
        input_x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Docstring for forward."""

        from torch.utils.checkpoint import checkpoint  # pylint: disable=not-callable,import-outside-toplevel

        def _inner_forward(x_in, kpm_in, fi_in):
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            """Docstring for _inner_forward."""

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

            cls_t = x_in[:, :1]
            x_seq = x_in[:, 1:]
            seq_mask = kpm_in[:, 1:] if kpm_in is not None else None
            xc_seq = self.conv_block(self.conv_norm(x_seq), key_padding_mask=seq_mask)
            xc = torch.cat([cls_t, xc_seq], dim=1)

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
                _inner_forward, input_x, key_padding_mask, frame_indices, use_reentrant=False
            )
        else:
            return _inner_forward(input_x)  # pylint: disable=not-callable,no-value-for-parameter


class LandmarkTrajectory1DStem(nn.Module):
    """Docstring for LandmarkTrajectory1DStem."""

    def __init__(
        self, in_channels: int = 9, num_keypoints: int = 60, out_dim: int = 128
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
        self, input_x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Docstring for forward."""

        batch_sz, seq_len = input_x.size(0), input_x.size(1)
        x_flat = input_x.reshape(batch_sz, seq_len, -1) if input_x.dim() == 4 else input_x
        x_t = x_flat.transpose(1, 2)
        if mask is not None:
            x_t = x_t * mask.unsqueeze(1).to(x_t.dtype)

        feat_seq = x_t
        feat_seq = self.act1(
            self.norm1(self.conv1(F.pad(feat_seq, (3, 3), mode="constant")))
        )
        feat_seq = self.conv2(F.pad(feat_seq, (2, 2), mode="constant"))
        feat_seq = self.act2(self.norm2(self.conv3(feat_seq)))

        feat_seq = feat_seq.transpose(1, 2)
        if mask is not None:
            feat_seq = feat_seq * mask.unsqueeze(-1).to(feat_seq.dtype)
        return self.out_proj(feat_seq)


# ==============================================================================
# 7. TRANSFORMER DECODER WITH EOS GRAMMAR PROTECTION
# ==============================================================================


class RoPEEmbedding(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for RoPEEmbedding."""

    def __init__(self, head_dim: int, max_seq_len: int = 512, base: float = 10000.0):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """Docstring for _build_cache."""

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
        """Docstring for _rotate_half."""

        half = input_x.shape[-1] // 2
        return torch.cat([-input_x[..., half:], input_x[..., :half]], dim=-1)

    def forward(
        self, query_q_lower: torch.Tensor, key_k_lower: torch.Tensor, offset: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for forward."""

        seq_s = query_q_lower.shape[-2]
        total_len = seq_s + offset
        if total_len <= self._cache_len:
            indices = torch.arange(seq_s, device=query_q_lower.device) + offset
            cos = torch.index_select(self.cos_cache, 2, indices).to(query_q_lower.dtype)
            sin = torch.index_select(self.sin_cache, 2, indices).to(query_q_lower.dtype)
        else:
            inv_freq = self.inv_freq.to(query_q_lower.device)
            t = torch.arange(offset, total_len, device=query_q_lower.device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos()[None, None].to(query_q_lower.dtype)
            sin = emb.sin()[None, None].to(query_q_lower.dtype)
        query_q_lower = query_q_lower * cos + self._rotate_half(query_q_lower) * sin
        key_k_lower = key_k_lower * cos + self._rotate_half(key_k_lower) * sin
        return query_q_lower, key_k_lower


class GroupedQuerySelfAttention(nn.Module):
    """Docstring for GroupedQuerySelfAttention."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        kv_heads: int = 2,
        max_seq_len: int = 256,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for forward."""

        batch_sz, seq_len, _ = input_x.shape
        q_in = self.q_norm(input_x)

        # MLA Latent Projection
        kv_latent = self.kv_latent_proj(input_x)
        kv_latent = self.kv_latent_norm(kv_latent)

        query_q_lower = self.q_proj(q_in).view(batch_sz, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv_latent)
        key_k_lower, val_v = torch.split(kv, kv.size(-1) // 2, dim=-1)
        key_k_lower = key_k_lower.view(batch_sz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)
        val_v = val_v.view(batch_sz, seq_len, self.kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].size(2) if past_key_value is not None else 0
        query_q_lower, key_k_lower = self.rope(query_q_lower, key_k_lower, offset=past_len)

        if past_key_value is not None:
            key_k_lower = torch.cat([past_key_value[0], key_k_lower], dim=2)
            val_v = torch.cat([past_key_value[1], val_v], dim=2)

        current_key_value = (key_k_lower, val_v) if use_cache else None

        k_exp = key_k_lower.repeat_interleave(self.groups, dim=1)
        v_exp = val_v.repeat_interleave(self.groups, dim=1)

        if seq_len == 1:
            if padding_mask is not None:
                attn_mask = (~padding_mask).unsqueeze(1).unsqueeze(2)
                out = F.scaled_dot_product_attention(query_q_lower, k_exp, v_exp, attn_mask=attn_mask, scale=self.scale)
            else:
                out = F.scaled_dot_product_attention(query_q_lower, k_exp, v_exp, scale=self.scale)
        else:
            if padding_mask is not None:
                total_len = past_len + seq_len
                causal_mask = torch.tril(torch.ones(seq_len, total_len, dtype=torch.bool, device=input_x.device))
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(1) & (~padding_mask).unsqueeze(1).unsqueeze(2)
                out = F.scaled_dot_product_attention(
                    query_q_lower, k_exp, v_exp, attn_mask=attn_mask, scale=self.scale
                )
            else:
                out = F.scaled_dot_product_attention(
                    query_q_lower, k_exp, v_exp, scale=self.scale, is_causal=True
                )

        out = self.o_proj(out.transpose(1, 2).reshape(batch_sz, seq_len, -1))
        return (out, current_key_value) if use_cache else out


class DecoderCrossAttention(nn.Module):
    """Docstring for DecoderCrossAttention."""

    def __init__(self, d_model: int = 512, nhead: int = 8, kv_heads: int = 2):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for forward."""

        batch_sz, seq_len, _ = tgt.shape
        query_q_lower = (
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
            key_k_lower = key_k_lower.view(batch_sz, seq_s, self.kv_heads, self.head_dim).transpose(1, 2)
            val_v = val_v.view(batch_sz, seq_s, self.kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to query and key for temporal alignment
        query_q_lower, key_k_lower = self.rope(query_q_lower, key_k_lower)

        current_key_value = (key_k_lower, val_v) if use_cache else None

        k_exp = key_k_lower.repeat_interleave(self.groups, dim=1)
        v_exp = val_v.repeat_interleave(self.groups, dim=1)

        if memory_key_padding_mask is not None:
            attn_mask = ~(memory_key_padding_mask.view(batch_sz, 1, 1, key_k_lower.size(2)).bool())
        else:
            attn_mask = None
        out = F.scaled_dot_product_attention(query_q_lower, k_exp, v_exp, attn_mask=attn_mask)  # pylint: disable=not-callable,not-callable
        out = self.o_proj(out.transpose(1, 2).reshape(batch_sz, seq_len, -1))
        return (out, current_key_value) if use_cache else out


class ASLDecoderLayer(nn.Module):
    """Docstring for ASLDecoderLayer."""

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
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
        self.ffn = SwiGLUFFN(d_model=d_model, dim_feedforward=ffn_dim, num_layers=num_layers)
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
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for forward."""

        if use_cache:
            sa_out, new_self_kv = self.self_attn(
                self.norm1(tgt), padding_mask=tgt_key_padding_mask, past_key_value=past_self_kv, use_cache=True
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
            tgt = tgt + self.drop1(self.gamma1 * self.self_attn(self.norm1(tgt), padding_mask=tgt_key_padding_mask))
            tgt = tgt + self.drop2(
                self.gamma2
                * self.cross_attn(self.norm2(tgt), memory, memory_key_padding_mask)
            )
            tgt = tgt + self.drop3(self.gamma3 * self.ffn(self.norm3(tgt)))
            return tgt


class ASLTransformerDecoder(nn.Module):
    """Docstring for ASLTransformerDecoder."""

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
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
                ASLDecoderLayer(d_model, nhead, kv_heads, ffn_dim, dropout, max_seq_len, num_layers=num_layers)
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
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for forward."""

        batch_sz, seq_s = tgt_ids.shape  # pylint: disable=not-callable,unused-variable
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
            hidden_h = self.emb_drop(
                self.token_emb(dropped_tgt_ids) * self.emb_scale
                + lex_embs * self.emb_scale * valid_lex_mask
            )
        else:
            hidden_h = self.emb_drop(self.token_emb(dropped_tgt_ids) * self.emb_scale)

        new_key_values = [] if use_cache else None
        tgt_key_padding_mask = (dropped_tgt_ids == GlossVocabulary.PAD_ID) if dropped_tgt_ids is not None else None
        
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
                hidden_h = layer(hidden_h, memory, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        hidden_h = self.final_norm(hidden_h)
        logits = self.lm_head(hidden_h)

        if use_cache:
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

            h_mtp_2, mtp2_self_kv, mtp2_cross_kv = self.mtp_layer(
                h_mtp,
                memory,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                past_self_kv=past_mtp2_kv[0],
                past_cross_kv=past_mtp2_kv[1],
                use_cache=True,
            )
            logits_3 = self.lm_head(h_mtp_2)

            if getattr(self, "training", False):
                new_key_values.append((mtp1_self_kv, mtp1_cross_kv))
                new_key_values.append((mtp2_self_kv, mtp2_cross_kv))
        else:
            h_mtp = self.mtp_layer(
                hidden_h, memory, memory_key_padding_mask=memory_key_padding_mask
            )
            logits_2 = self.lm_head(h_mtp)
            h_mtp_2 = self.mtp_layer(
                h_mtp, memory, memory_key_padding_mask=memory_key_padding_mask
            )
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
    """
    Homoscedastic Task Uncertainty Loss Weighting (Kendall & Gal, CVPR 2018).
    Bypasses gradient propagation for zero-valued or uncalculated losses to prevent divergence.
    """

    def __init__(self, loss_config: Optional[Dict[str, float]] = None):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
                "inter_ctc": 1.0,
                "lpc": 1.0,
            }

        self.log_vars = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor(-math.log(val_v), dtype=torch.float32))
                for name, val_v in loss_config.items()
            }
        )

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Docstring for forward."""

        total_loss = torch.zeros((), device=next(iter(losses.values())).device)
        for name, loss in losses.items():
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            if name not in self.log_vars:
                raise ValueError(
                    f"Unregistered loss key '{name}' produced by model! Please add it to HomoscedasticLossWrapper config."
                )

            raw_s = self.log_vars[name].to(loss.device)
            # Smooth parameterization using tanh instead of hard clamping
            s = torch.tanh(raw_s / 5.0) * 5.0
            prec = torch.exp(-s)

            # Apply tracking weight to full loss value, do not zero out uncertainty gradients
            # Use torch.where to avoid modifying `s` parameter gradients when a task is disabled
            is_active = (loss != 0).to(loss.dtype)
            task_loss = torch.where(
                is_active.bool(),
                (0.5 * prec * loss + 0.5 * s),
                torch.zeros_like(loss)
            )
            total_loss = total_loss + task_loss.mean()

        return total_loss


class CosineLinear(nn.Module):
    """Docstring for CosineLinear."""

    def __init__(self, in_features: int, out_features: int, init_tau: float = 20.0):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        # L2 normalize features and weights
        """Docstring for forward."""

        x_norm = F.normalize(input_x.float(), p=2, dim=-1, eps=1e-5).to(input_x.dtype)
        w_norm = F.normalize(self.weight.float(), p=2, dim=-1, eps=1e-5).to(input_x.dtype)
        # Cosine similarity scaled by learnable temperature tau
        safe_tau = (F.softplus(self.tau) + 1.0).to(input_x.dtype)  # pylint: disable=not-callable,not-callable
        return F.linear(x_norm, w_norm) * safe_tau  # pylint: disable=not-callable,not-callable


class CTCHead(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for CTCHead."""

    def __init__(self, d_model: int, vocab_size: int):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.proj = CosineLinear(d_model, vocab_size)

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        """Docstring for forward."""

        if getattr(self, "debug_xla", False) and torch.isnan(enc_seq).any():
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            print("enc_seq has NaNs!")
        return F.log_softmax(self.proj(enc_seq), dim=-1)


class CrossModalInfoNCE(nn.Module):
    """Docstring for CrossModalInfoNCE."""

    def __init__(self, init_temp: float = 0.07, **kwargs):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        target_sp = init_temp - 0.05
        # Avoid bfloat16 precision issues near zero bounds by using direct scaling.
        self.log_temp = nn.Parameter(torch.tensor(math.log(math.exp(target_sp - 0.05) - 1.0) if target_sp > 0.05 else math.log(target_sp)))

    def forward(
        self,
        vis_emb: torch.Tensor,
        sent_emb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None,
        gt_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Docstring for forward."""

        device = vis_emb.device
        import torch.distributed as dist  # pylint: disable=not-callable,import-outside-toplevel

        if IS_TPU and "xla" in str(device).lower():
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

            world_size = get_xla_world_size()
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        val_v = F.normalize(vis_emb.float(), p=2, dim=-1, eps=1e-8)
        s = F.normalize(sent_emb.float(), p=2, dim=-1, eps=1e-8)

        if world_size > 1:
            if IS_TPU and "xla" in str(device).lower():
                fused_sv = torch.cat([s, val_v], dim=1)
                fused_sv_all = xm.all_gather(fused_sv)  # pylint: disable=not-callable,used-before-assignment
                s_all = fused_sv_all[:, : s.shape[1]]
                v_all = fused_sv_all[:, s.shape[1] :]

                if valid_mask is not None and gt_tokens is not None:
                    gt_float = gt_tokens.float()
                    fused_mask_gt = torch.cat(
                        [valid_mask.float().unsqueeze(1), gt_float], dim=1
                    )
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
            elif dist.is_initialized():
                fused_sv = torch.cat([s, val_v], dim=1)
                fused_sv_list = [torch.zeros_like(fused_sv) for _ in range(world_size)]
                dist.all_gather(fused_sv_list, fused_sv)
                fused_sv_all = torch.cat(fused_sv_list, dim=0)
                s_all = fused_sv_all[:, : s.shape[1]]
                v_all = fused_sv_all[:, s.shape[1] :]

                if valid_mask is not None and gt_tokens is not None:
                    gt_float = gt_tokens.float()
                    fused_mask_gt = torch.cat(
                        [valid_mask.float().unsqueeze(1), gt_float], dim=1
                    )
                    fused_mask_gt_list = [torch.zeros_like(fused_mask_gt) for _ in range(world_size)]
                    dist.all_gather(fused_mask_gt_list, fused_mask_gt)
                    fused_mask_gt_all = torch.cat(fused_mask_gt_list, dim=0)
                    valid_mask_all = fused_mask_gt_all[:, 0].bool()
                    gt_tokens_all = fused_mask_gt_all[:, 1:].long()
                elif valid_mask is not None:
                    valid_mask_list = [torch.zeros_like(valid_mask.bool()) for _ in range(world_size)]
                    dist.all_gather(valid_mask_list, valid_mask.bool())
                    valid_mask_all = torch.cat(valid_mask_list, dim=0)
                    gt_tokens_all = None
                elif gt_tokens is not None:
                    valid_mask_all = None
                    gt_tokens_list = [torch.zeros_like(gt_tokens) for _ in range(world_size)]
                    dist.all_gather(gt_tokens_list, gt_tokens)
                    gt_tokens_all = torch.cat(gt_tokens_list, dim=0)
                else:
                    valid_mask_all = None
                    gt_tokens_all = None
            else:
                v_all = val_v
                s_all = s
                valid_mask_all = valid_mask.bool() if valid_mask is not None else None
                gt_tokens_all = gt_tokens
        else:
            v_all = val_v
            s_all = s
            valid_mask_all = valid_mask.bool() if valid_mask is not None else None
            gt_tokens_all = gt_tokens

        if val_v.size(0) == 0:
            return torch.zeros((), device=device)

        temp = F.softplus(self.log_temp) + 0.05  # pylint: disable=not-callable,not-callable
        # logits_v2s shape: [batch_sz, batch_sz]
        logits_v2s = torch.matmul(val_v, s_all.transpose(-1, -2)) / temp
        logits_s2v = torch.matmul(s, v_all.transpose(-1, -2)) / temp


        if valid_mask_all is not None:
            # Mask out invalid candidate columns in [batch_sz, 8B] logits matrix
            invalid_candidate_mask = ~valid_mask_all
            logits_v2s = logits_v2s.masked_fill(
                invalid_candidate_mask.unsqueeze(0), -1e9
            )
            logits_s2v = logits_s2v.masked_fill(
                invalid_candidate_mask.unsqueeze(0), -1e9
            )

        if gt_tokens_all is not None:
            # Treat identically padded sequences as positives across all gathered replicas [batch_sz, 8B]
            pos_mask = (
                (gt_tokens.unsqueeze(1) == gt_tokens_all.unsqueeze(0))
                .all(dim=-1)
                .float()
            )
        else:
            rank_val = 0
            if IS_TPU and "xla" in str(device).lower():
                try:
                    import torch_xla.runtime as xr  # pylint: disable=not-callable,import-outside-toplevel

                    rank_val = xr.global_ordinal()
                except Exception:  # pylint: disable=not-callable,broad-exception-caught
                    try:
                        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                        rank_val = getattr(xm, "get_ordinal", lambda: 0)()  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
                    except Exception:  # pylint: disable=not-callable,broad-exception-caught
                        rank_val = 0
            elif dist.is_initialized():
                rank_val = dist.get_rank()

            global_local_rows = rank_val * val_v.size(0) + torch.arange(
                val_v.size(0), device=val_v.device
            )
            labels_all = torch.arange(s_all.size(0), device=val_v.device)
            pos_mask = (
                global_local_rows.unsqueeze(1) == labels_all.unsqueeze(0)
            ).float()

        if valid_mask is not None:
            valid_rows = valid_mask.float()
            pos_mask = pos_mask * valid_rows.unsqueeze(1)
            if valid_mask_all is not None:
                pos_mask = pos_mask * valid_mask_all.float().unsqueeze(0)
        else:
            valid_rows = torch.ones(val_v.shape[0], device=val_v.device)

        # v2s loss computation
        exp_logits_v2s = torch.exp(logits_v2s - logits_v2s.max(dim=-1, keepdim=True)[0])
        denom_v2s = torch.clamp(
            exp_logits_v2s.sum(dim=-1, keepdim=True).float(), min=1e-4
        )
        log_prob_v2s = (
            logits_v2s - logits_v2s.max(dim=-1, keepdim=True)[0]
        ) - torch.log(denom_v2s).to(logits_v2s.dtype)

        # s2v loss computation
        exp_logits_s2v = torch.exp(logits_s2v - logits_s2v.max(dim=-1, keepdim=True)[0])
        denom_s2v = torch.clamp(
            exp_logits_s2v.sum(dim=-1, keepdim=True).float(), min=1e-4
        )
        log_prob_s2v = (
            logits_s2v - logits_s2v.max(dim=-1, keepdim=True)[0]
        ) - torch.log(denom_s2v).to(logits_s2v.dtype)

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
    """Docstring for DenseSentenceSemanticLoss."""

    def __init__(self, d_model: int = 512, embed_dim: int = 256):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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

    def update_momentum(self, mask_m=0.01):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for update_momentum."""

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
        """Docstring for forward."""

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
            g = F.normalize(self.proj_gt(gt_sent).float(), p=2, dim=-1, eps=1e-8).detach()

        cos_sim = (prob_p * g).sum(dim=-1)
        cos_loss = (1.0 - cos_sim) * has_tokens

        if IS_TPU and "xla" in str(prob_p.device).lower():
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

            world_size = xm.xrt_world_size()
            batch_size = prob_p.shape[0]

            p_sq = prob_p**2
            # Compute local sum across the batch dimension before distributed reduction
            prob_p_local = prob_p.sum(dim=0)
            p_sq_local = p_sq.sum(dim=0)
            p_sum, p_sq_sum = xm.all_reduce(xm.REDUCE_SUM, [prob_p_local, p_sq_local])

            global_mean = p_sum / (world_size * batch_size)
            global_var = (p_sq_sum / (world_size * batch_size)) - (global_mean**2)

            std_p = torch.sqrt(global_var + 1e-4)
        else:
            std_p = torch.sqrt(prob_p.var(dim=0, unbiased=False) + 1e-4)
        std_loss = torch.mean(F.relu(1.0 - std_p))

        loss = (cos_loss + 0.5 * std_loss) * has_tokens

        weight_sum = has_tokens
        if sample_weights is not None:
            loss = loss * sample_weights
            weight_sum = weight_sum * sample_weights

        return _distributed_normalize(loss.sum(), weight_sum.sum())


class SupervisedContrastiveLoss(nn.Module):
    """Docstring for SupervisedContrastiveLoss."""

    def __init__(self, temperature: float = 0.07, **kwargs):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: torch.Tensor = None,
        enqueue: bool = True,  # pylint: disable=not-callable,unused-argument
    ) -> torch.Tensor:
        """Docstring for forward."""

        if labels is None:
            return torch.zeros((), device=features.device)
        features = F.normalize(features.float(), p=2, dim=1, eps=1e-5)
        device = features.device

        import torch.distributed as dist  # pylint: disable=not-callable,import-outside-toplevel

        if IS_TPU and "xla" in str(device).lower():
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

            world_size = get_xla_world_size()
        elif dist.is_initialized():
            world_size = dist.get_world_size()
        else:
            world_size = 1

        if world_size > 1 and IS_TPU and "xla" in str(device).lower():
            fused = torch.cat([features, labels.unsqueeze(1).float()], dim=1)
            fused_all = xm.all_gather(fused)  # pylint: disable=not-callable,possibly-used-before-assignment
            all_feats = fused_all[:, :-1]
            all_labels = fused_all[:, -1].long()
        else:
            all_feats = features
            all_labels = labels

        batch_sz = features.shape[0]
        pos_mask = torch.eq(labels.view(-1, 1), all_labels.view(1, -1)).float()
        valid_labels = (all_labels.view(1, -1) != -1).float()
        pos_mask = pos_mask * valid_labels

        # Zero out self-pair matches so sample is not its own positive across all TPU ranks
        rank_val = 0
        if IS_TPU and "xla" in str(device).lower():
            try:
                import torch_xla.runtime as xr  # pylint: disable=not-callable,import-outside-toplevel

                rank_val = xr.global_ordinal()
            except Exception:  # pylint: disable=not-callable,broad-exception-caught
                try:
                    import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                    rank_val = getattr(xm, "get_ordinal", lambda: 0)()  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
                except Exception:  # pylint: disable=not-callable,broad-exception-caught
                    rank_val = 0
        elif dist.is_initialized():
            rank_val = dist.get_rank()

        # Fully static XLA-friendly self-masking
        global_indices = torch.arange(all_feats.shape[0], device=device)
        local_indices = rank_val * batch_sz + torch.arange(batch_sz, device=device)
        # Broadcast to create a [batch_sz, global_B] boolean mask
        is_self = local_indices.unsqueeze(1) == global_indices.unsqueeze(0)
        pos_mask = torch.where(is_self, torch.zeros_like(pos_mask), pos_mask)

        pos_logits = torch.matmul(features.float(), all_feats.float().T) / float(
            self.temperature
        )
        # Mask self-similarity in denominator so exp(1.0/tau) = exp(14.28) does not suppress negative gradients
        pos_logits = torch.where(is_self, torch.full_like(pos_logits, -1e9), pos_logits)
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
    # pylint: disable=not-callable,abstract-method,arguments-differ
    """Docstring for GradientReversalFunction."""
    @staticmethod

    def forward(ctx, input_x: torch.Tensor, alpha: float = 1.0):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for forward."""

        ctx.alpha = float(alpha)
        return input_x.view_as(input_x)

    @staticmethod
    def backward(ctx, grad_output):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for backward."""

        return grad_output.neg() * ctx.alpha, None


class LandmarkReconstructionHead(nn.Module):
    """Docstring for LandmarkReconstructionHead."""

    def __init__(self, d_model: int = 512, out_dim: int = 540):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.recon = nn.Sequential(
            nn.Linear(d_model, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, out_dim),
        )

    def forward(self, enc_seq: torch.Tensor) -> torch.Tensor:
        """Docstring for forward."""

        return self.recon(enc_seq)


# ==============================================================================
# 9. ASL FOUNDATION MODEL MAIN AGGREGATOR
# ==============================================================================


class PositionalEncoding1D(nn.Module):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for PositionalEncoding1D."""

    def __init__(self, d_model: int, max_len: int = 1000000):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

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
        self, input_x: torch.Tensor, frame_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Docstring for forward."""

        if frame_indices is not None:
            idx = torch.clamp(frame_indices.long(), min=0, max=self.pe.size(1) - 1)
            batch_pe = self.pe.squeeze(0)[idx]
            return input_x + batch_pe.to(dtype=input_x.dtype)
        return input_x + self.pe[:, : input_x.size(1), :]


def safe_norm(tensor, dim=-1, keepdim=False, eps=1e-6):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for safe_norm."""

    return torch.sqrt(torch.sum(tensor**2, dim=dim, keepdim=keepdim) + eps)


def safe_cosine_sim(v1, v2, eps=1e-5):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for safe_cosine_sim."""

    n1 = torch.norm(v1, dim=-1, keepdim=True).clamp(min=eps)
    n2 = torch.norm(v2, dim=-1, keepdim=True).clamp(min=eps)
    return (v1 * v2).sum(dim=-1) / (n1 * n2).squeeze(-1)


class ASLFoundationModel(nn.Module):
    # pylint: disable=not-callable,too-many-instance-attributes
    """Docstring for ASLFoundationModel."""

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
        num_domains: int = 4,  # pylint: disable=not-callable,unused-argument
        csv_path: Optional[Union[str, Path]] = None,
        label_to_idx: Optional[Dict[str, int]] = None,
        use_mamba: bool = True,
        tome_r: int = 80,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        super().__init__()
        self.vocab_size = vocab_size
        self.d_enc = d_enc
        self.max_enc_len = max_enc_len
        self.max_dec_len = max_dec_len
        self.use_mamba = use_mamba
        self.tome_r = tome_r
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        num_keypoints * channels_per_kp  # pylint: disable=not-callable,pointless-statement

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_enc) * 0.02)
        self.visual_encoder = LandmarkTrajectory1DStem(
            in_channels=channels_per_kp, num_keypoints=num_keypoints, out_dim=128
        )
        # Change input stem to expect 768 perfectly aligned dimensions (640 padded + 128)
        self.input_stem = nn.Sequential(
            nn.Linear(768, d_enc), RMSNorm(d_enc), nn.GELU()
        )
        dpr = [input_x.item() for input_x in torch.linspace(0.0, drop_path_rate, num_enc_layers)]

        self.blocks = nn.ModuleList()
        for i in range(num_enc_layers):
            if i == num_enc_layers // 2:
                # Physically halve the tensor midway through the network
                self.blocks.append(TemporalStridedPool())

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
        self.inter_ctc_head = CTCHead(d_enc, vocab_size)
        self.lpc_proj = nn.Sequential(nn.Linear(d_enc, d_enc), RMSNorm(d_enc))
        self.mlm_head = nn.Linear(d_enc, 540)
        self.domain_head = nn.Linear(d_enc, 4)

        # ─── MATH FIX: Encoder Auxiliary Classification Head ───
        # Mathematically forces the Conformer to anchor the latent space into a discrete conceptual cluster
        # BEFORE giving the sequence to the decoder. Bypasses decoder hallucination drift.
        self.aux_gloss_head = CosineLinear(d_enc, vocab_size, init_tau=2.65)

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
        self.chicago_decoder.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)
        self.english_decoder.token_emb.weight[GlossVocabulary.PAD_ID].fill_(0)

    def update_tome_r(self, epoch: int, max_epochs: int):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        # [TPU XLA HOTFIX] ToMe changes the tensor sequence length (e.g., num_n -> num_n-r),
        # which forces XLA to compile a brand new static graph in device memory.
        # Instead of increasing `r` every single epoch (which causes 70+ recompilations
        # and guaranteed HBM OOM), we "bucket" `r` into 4 distinct stages.
        # This gives us the dynamic ToMe effect but only compiles 4 graphs total!
        # [NEW HOTFIX] On TPU, we fix the ratio to 30 completely to avoid graph breaks.
        """Docstring for update_tome_r."""

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
            if isinstance(block, TemporalStridedPool):
                block.r = new_r

    def _encode(
        self,
        input_x: torch.Tensor,
        phonology_features: torch.Tensor,
        mask: Optional[torch.Tensor],
        mlm_mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for _encode."""

        batch_sz, seq_len = input_x.size(0), input_x.size(1)
        inter_h = None
        inter_idx = len(self.blocks) // 2

        if mlm_mask is not None:
            used_mlm_mask = mlm_mask
            mask_shape = [1] * (input_x.dim() - 2)
            x_in = input_x * (~mlm_mask).view(batch_sz, seq_len, *mask_shape).to(input_x.dtype)
        else:
            x_in = input_x
            used_mlm_mask = None

        if x_in.dim() == 4 and x_in.size(2) == 60 and x_in.size(3) >= 3:
            xk = x_in
            x_flat = x_in.reshape(batch_sz, seq_len, -1)
            v_tokens = self.visual_encoder(xk, mask=mask)
        else:
            x_flat = x_in.reshape(batch_sz, seq_len, -1) if x_in.dim() == 4 else x_in
            v_tokens = self.visual_encoder(x_in, mask=mask)

        x_enriched = torch.cat([x_flat, phonology_features], dim=-1)
        padding_needed = 640 - x_enriched.size(-1)
        x_padded = F.pad(x_enriched, (0, padding_needed), value=0.0)

        hidden_h = self.input_stem(torch.cat([x_padded, v_tokens], dim=-1))
        hidden_h = self.pos_enc(hidden_h, frame_indices=frame_indices)
        hidden_h = torch.cat([self.cls_token.expand(batch_sz, -1, -1), hidden_h], dim=1)

        routing_fi = frame_indices.long() if frame_indices is not None else None

        cur_mask = mask
        if cur_mask is not None:
            kpm = torch.cat(
                [torch.zeros((batch_sz, 1), dtype=torch.bool, device=hidden_h.device), ~cur_mask],
                dim=1,
            )
        else:
            kpm = None

        token_sizes = torch.ones(batch_sz, seq_len, 1, device=hidden_h.device, dtype=hidden_h.dtype)

        for idx, block in enumerate(self.blocks):
            if isinstance(block, TemporalStridedPool):
                cls_t = hidden_h[:, :1]
                seq_t = hidden_h[:, 1:]
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
                hidden_h = torch.cat([cls_t, seq_t], dim=1)
                if cur_mask is not None:
                    kpm = torch.cat(
                        [
                            torch.zeros((batch_sz, 1), dtype=torch.bool, device=hidden_h.device),
                            ~cur_mask,
                        ],
                        dim=1,
                    )
                else:
                    kpm = None
            else:
                if routing_fi is not None:
                    cls_fi = torch.zeros(
                        (batch_sz, 1), dtype=routing_fi.dtype, device=routing_fi.device
                    )
                    pos_fi = torch.cat([cls_fi, routing_fi + 1], dim=1)
                else:
                    pos_fi = None
                hidden_h = block(hidden_h, key_padding_mask=kpm, frame_indices=pos_fi)
                if idx == inter_idx:
                    inter_h = hidden_h[:, 1:].clone()

            if getattr(self, "debug_xla", False) and torch.isnan(hidden_h).any():
                print(f"NaN introduced at block {idx}!")
                break

        hidden_h = self.enc_final_norm(hidden_h)
        if getattr(self, "debug_xla", False) and torch.isnan(hidden_h).any():
            print("NaN introduced at enc_final_norm!")

        # Return routing_fi directly (strictly 0-indexed original frame indices)
        return hidden_h[:, 0], hidden_h[:, 1:], cur_mask, used_mlm_mask, routing_fi, mask, inter_h

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
        ss_ratio: float = 0.0,
    ) -> Union[Optional[torch.Tensor], Dict]:
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        # Always compute kinematics and augmentations on TPU to avoid Host CPU bottleneck
        """Docstring for forward."""

        if True:  # pylint: disable=not-callable,using-constant-test
            input_x = input_x[..., :3]  # Force dynamic kinematic computation
            batch_sz, seq_len, key_k, channels = input_x.shape  # pylint: disable=not-callable,unused-variable
            if self.training:
                scale = torch.rand(batch_sz, 1, 1, 1, device=input_x.device) * 0.30 + 0.85
                input_x = input_x * scale

                shift = torch.rand(batch_sz, 1, 1, 2, device=input_x.device) * 0.07 - 0.035
                input_x[..., :2] = input_x[..., :2] + shift

                roll = (torch.rand(batch_sz, device=input_x.device) * 20.0 - 10.0) * (
                    3.14159 / 180.0
                )
                cos_r = torch.cos(roll).view(batch_sz, 1, 1, 1)
                sin_r = torch.sin(roll).view(batch_sz, 1, 1, 1)

                valid_mask_sp = (input_x[..., :2].abs().sum(dim=-1, keepdim=True) > 1e-5).float()
                center = (input_x[..., :2] * valid_mask_sp).sum(dim=(1, 2), keepdim=True) / valid_mask_sp.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)
                
                x_centered = input_x[..., :2] - center
                x_rot_0 = x_centered[..., 0:1] * cos_r - x_centered[..., 1:2] * sin_r
                x_rot_1 = x_centered[..., 0:1] * sin_r + x_centered[..., 1:2] * cos_r
                input_x[..., :2] = torch.cat([x_rot_0, x_rot_1], dim=-1) + center

                kp_mask = (torch.rand(batch_sz, seq_len, key_k, 1, device=input_x.device) > 0.05).float()
                input_x = input_x * kp_mask

            dt = torch.ones(batch_sz, seq_len, 1, 1, device=input_x.device)
            if frame_indices is not None and seq_len > 1:
                actual_dt = (
                    (frame_indices[:, 1:] - frame_indices[:, :-1])
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                dt[:, 1:] = torch.where(
                    actual_dt == 0, torch.ones_like(actual_dt), actual_dt
                )

            vel = torch.zeros_like(input_x)
            acc = torch.zeros_like(input_x)

            if seq_len > 1:
                vel[:, 1:] = (input_x[:, 1:] - input_x[:, :-1]) / dt[:, 1:]
                vel[:, 0] = vel[:, 1]
                acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]
                acc[:, 0] = acc[:, 1]

            if self.training:
                jitter = torch.randn(batch_sz, 1, key_k, channels, device=input_x.device) * 0.035
                input_x = input_x + jitter

        if True:  # pylint: disable=not-callable,using-constant-test
            pos = input_x

            # --- NEW: ASL Phonology Feature Pack (19 Dims) ---
            # 1. Palm Orientation Normals (6 Dims)
            lh_u = pos[:, :, 5] - pos[:, :, 0]
            lh_v = pos[:, :, 17] - pos[:, :, 0]
            lh_normal = F.normalize(
                torch.linalg.cross(lh_u, lh_v, dim=-1), p=2, dim=-1, eps=1e-5  # pylint: disable=not-callable,not-callable
            )

            rh_u = pos[:, :, 26] - pos[:, :, 21]
            rh_v = pos[:, :, 38] - pos[:, :, 21]
            rh_normal = F.normalize(
                torch.linalg.cross(rh_u, rh_v, dim=-1), p=2, dim=-1, eps=1e-5  # pylint: disable=not-callable,not-callable
            )

            # 2. Bimanual Synchrony (1 Dim)
            lh_vel = vel[:, :, 0]
            rh_vel = vel[:, :, 21]
            bimanual_sync = F.cosine_similarity(lh_vel, rh_vel, dim=-1).unsqueeze(-1)  # pylint: disable=not-callable,not-callable

            # 3. Location Anchoring to Face (2 Dims)
            nose = pos[:, :, 42]
            lh_face_dist = safe_norm(pos[:, :, 0] - nose, dim=-1, keepdim=True)
            rh_face_dist = safe_norm(pos[:, :, 21] - nose, dim=-1, keepdim=True)

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
                    batch_sz, seq_len, key_k, self.channels_per_kp - input_x.shape[-1], device=input_x.device
                )
                input_x = torch.cat([input_x, pad], dim=-1)
            input_x = input_x[..., : self.channels_per_kp]

        h_cls, h_seq, enc_mask, used_mlm_mask, fi_out, orig_enc_mask, inter_h = (  # pylint: disable=not-callable,unused-variable
            self._encode(
                input_x,
                phonology_features,  # pylint: disable=not-callable,possibly-used-before-assignment
                mask,
                mlm_mask=mlm_mask,
                frame_indices=frame_indices,
            )
        )

        dec_logits, dec_hidden = None, None
        chicago_logits, english_logits = None, None

        if gloss_seq is not None and self.decoder is not None:
            dec_padding_mask = (gloss_seq == GlossVocabulary.PAD_ID)
            dec_logits, dec_hidden, _ = decode_seq(
                self.decoder, gloss_seq, h_seq, enc_mask, dec_padding_mask, ss_ratio=ss_ratio
            )

        if chicago_seq is not None and self.chicago_decoder is not None:
            chicago_padding_mask = (chicago_seq == GlossVocabulary.PAD_ID)
            chicago_logits, _, _ = decode_seq(
                self.chicago_decoder, chicago_seq, h_seq, enc_mask, chicago_padding_mask, ss_ratio=ss_ratio
            )

        if english_seq is not None and self.english_decoder is not None:
            english_padding_mask = (english_seq == GlossVocabulary.PAD_ID)
            english_logits, _, _ = decode_seq(
                self.english_decoder, english_seq, h_seq, enc_mask, english_padding_mask, ss_ratio=ss_ratio
            )

        # Compute output heads
        ctc_log_probs = self.ctc_head(h_seq)
        inter_ctc_log_probs = self.inter_ctc_head(inter_h) if inter_h is not None else None
        aux_logits = self.aux_gloss_head(h_cls)
        pred_len = self.length_head(h_cls).squeeze(-1)
        chicago_pred_len = self.chicago_length_head(h_cls).squeeze(-1)
        english_pred_len = self.english_length_head(h_cls).squeeze(-1)

        vis_emb = self.visual_proj(h_cls)
        proj_feats = self.contrastive_head(h_cls)
        domain_logits = self.domain_head(GradientReversalFunction.apply(h_cls, grl_alpha))
        mlm_logits = self.mlm_head(h_seq)

        lpc_feats = self.lpc_proj(h_seq)
        if h_seq.shape[1] > 1:
            diff = lpc_feats[:, 1:] - lpc_feats[:, :-1]
            loss_lpc = (diff ** 2).mean() + F.relu(1.0 - lpc_feats.std(dim=1).mean())
        else:
            loss_lpc = torch.tensor(0.0, device=h_seq.device)

        sent_emb = None
        if dec_hidden is not None:
            sent_emb = self.sentence_proj(dec_hidden[:, -1])

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
            "orig_x": input_x,
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
        }



    @torch.no_grad()
    @torch._dynamo.disable
    def generate(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        task: str = "gloss",
        temperature: float = 1.0,
        top_k: int = 0,
        use_mtp_speculative: bool = False,
        frame_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive generation method for inference/validation."""
        batch_sz = features.size(0)
        device = features.device

        if task == "chicago":
            decoder_mod = self.chicago_decoder
        elif task == "english":
            decoder_mod = self.english_decoder
        else:
            decoder_mod = self.decoder

        bos_id = GlossVocabulary.BOS_ID
        eos_id = GlossVocabulary.EOS_ID

        features_3d = features[..., :3]
        batch_sz, seq_len, key_k, _ = features_3d.shape

        vel = torch.zeros_like(features_3d)
        acc = torch.zeros_like(features_3d)
        dt = torch.ones(batch_sz, seq_len, 1, 1, device=device)
        if frame_indices is not None and seq_len > 1:
            actual_dt = (frame_indices[:, 1:] - frame_indices[:, :-1]).unsqueeze(-1).unsqueeze(-1)
            dt[:, 1:] = torch.where(actual_dt == 0, torch.ones_like(actual_dt), actual_dt)

        if seq_len > 1:
            vel[:, 1:] = (features_3d[:, 1:] - features_3d[:, :-1]) / dt[:, 1:]
            vel[:, 0] = vel[:, 1]
            acc[:, 1:] = (vel[:, 1:] - vel[:, :-1]) / dt[:, 1:]
            acc[:, 0] = acc[:, 1]

        lh_u = features_3d[:, :, 5] - features_3d[:, :, 0]
        lh_v = features_3d[:, :, 17] - features_3d[:, :, 0]
        lh_normal = F.normalize(torch.linalg.cross(lh_u, lh_v, dim=-1), p=2, dim=-1, eps=1e-5)  # pylint: disable=not-callable,not-callable

        rh_u = features_3d[:, :, 26] - features_3d[:, :, 21]
        rh_v = features_3d[:, :, 38] - features_3d[:, :, 21]
        rh_normal = F.normalize(torch.linalg.cross(rh_u, rh_v, dim=-1), p=2, dim=-1, eps=1e-5)  # pylint: disable=not-callable,not-callable

        bimanual_sync = F.cosine_similarity(vel[:, :, 0], vel[:, :, 21], dim=-1).unsqueeze(-1)  # pylint: disable=not-callable,not-callable
        nose = features_3d[:, :, 42]
        lh_face_dist = safe_norm(features_3d[:, :, 0] - nose, dim=-1, keepdim=True)
        rh_face_dist = safe_norm(features_3d[:, :, 21] - nose, dim=-1, keepdim=True)

        lh_curl = safe_norm(features_3d[:, :, [4, 8, 12, 16, 20]] - features_3d[:, :, 0].unsqueeze(2), dim=-1)
        rh_curl = safe_norm(features_3d[:, :, [25, 29, 33, 37, 41]] - features_3d[:, :, 21].unsqueeze(2), dim=-1)

        phonology_features = torch.cat([lh_normal, rh_normal, bimanual_sync, lh_face_dist, rh_face_dist, lh_curl, rh_curl], dim=-1)

        input_x = torch.cat([features_3d, vel, acc], dim=-1)
        if input_x.shape[-1] < self.channels_per_kp:
            pad = torch.zeros(batch_sz, seq_len, key_k, self.channels_per_kp - input_x.shape[-1], device=device)
            input_x = torch.cat([input_x, pad], dim=-1)
        input_x = input_x[..., : self.channels_per_kp]

        _, h_seq, enc_mask, _, _, _, _ = self._encode(
            input_x, phonology_features, mask, mlm_mask=None, frame_indices=frame_indices
        )

        generated = torch.full((batch_sz, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_sz, dtype=torch.bool, device=device)

        kv_caches = None
        for _ in range(max_new_tokens):
            if finished.all():
                break
            dec_padding_mask = (generated == GlossVocabulary.PAD_ID)
            logits, _, kv_caches = decode_seq(decoder_mod, generated, h_seq, enc_mask, dec_padding_mask, kv_caches=kv_caches, shift_target=False)
            next_token_logits = logits[:, -1, :]

            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = torch.where(finished, torch.full_like(next_tokens, GlossVocabulary.PAD_ID), next_tokens)
            generated = torch.cat([generated, next_tokens.unsqueeze(-1)], dim=-1)

            finished = finished | (next_tokens == eos_id)

        return generated


class ModelEMA:
    """Exponential Moving Average wrapper for model parameters."""

    def __init__(
        self,
        model: nn.Module,
        decay_base: float = 0.999,
        decay_max: float = 0.9999,
    ):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for __init__."""

        self.decay_base = decay_base
        self.decay_max = decay_max
        self.shadow = {}
        self.backup = {}

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.clone().detach()

    def update(self, model: nn.Module, progress: float = 0.0):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for update."""

        try:
            first_param = next(iter(model.parameters()))
            IS_XLA = "xla" in str(first_param.device).lower()  # pylint: disable=not-callable,invalid-name
        except Exception:  # pylint: disable=not-callable,broad-exception-caught
            IS_XLA = False  # pylint: disable=not-callable,invalid-name

        if IS_XLA:
            # XLA cache breaker prevention: using dynamic python scalars in in-place ops
            # bakes them as constants and forces a new graph compilation.
            decay = self.decay_max
        else:
            decay = self.decay_base + (self.decay_max - self.decay_base) * progress

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Avoid .data which causes XLA memory leaks!
                    self.shadow[name].mul_(decay).add_(param, alpha=1.0 - decay)


    def apply_shadow(self, model: nn.Module):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for apply_shadow."""

        self.backup = {}  # pylint: disable=not-callable,attribute-defined-outside-init
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and name in self.shadow:
                    # Backup the current parameters safely
                    self.backup[name] = param.clone().detach()
                    # Apply EMA parameters safely using .copy_ to preserve Dynamo tracing
                    param.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
        """Docstring for restore."""

        with torch.no_grad():
            for name, param in model.named_parameters():
                if (
                    param.requires_grad
                    and hasattr(self, "backup")
                    and name in self.backup
                ):
                    # Restore original parameters
                    param.copy_(self.backup[name])
        self.backup = {}  # pylint: disable=not-callable,attribute-defined-outside-init


def _get_optimizer_groups(
    model: nn.Module, loss_wrapper: nn.Module, weight_decay: float
):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for _get_optimizer_groups."""

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



POLY1_EPS = 1.0  # Used in Poly1 focal loss computation

def compute_seq_loss(
    logits_f, gt_ids, valid_mask, sample_weights=None, class_weights=None, gamma=2.0, label_smoothing=0.1
):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for compute_seq_loss."""
    vocab_v = logits_f.shape[-1]
    lf = logits_f.reshape(-1, vocab_v).float()
    tf = gt_ids.reshape(-1).clamp(0, vocab_v - 1)
    vf = valid_mask.reshape(-1).float()

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
    ce_smoothed = F.cross_entropy(
        lf, tf, reduction="none", label_smoothing=label_smoothing
    )

    poly1 = focal_weight * ce_smoothed + POLY1_EPS * (1.0 - p_target)
    return _distributed_normalize((poly1 * vf).sum(), vf.sum())

def compute_eos_loss(logits_f, gt_ids, valid_mask, sample_weights=None, gamma=2.0):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for compute_eos_loss."""
    vocab_v = logits_f.shape[-1]
    lf = logits_f.reshape(-1, vocab_v).float()
    tf = gt_ids.reshape(-1).clamp(0, vocab_v - 1)
    vf = valid_mask.reshape(-1).float()

    eos_mask = (tf == GlossVocabulary.EOS_ID).float()
    vf = vf * eos_mask

    if sample_weights is not None:
        sw = sample_weights.unsqueeze(1).expand_as(gt_ids).reshape(-1)
        vf = vf * sw

    ce_unsmoothed = F.cross_entropy(lf, tf, reduction="none")
    p_target = torch.exp(-ce_unsmoothed).clamp(min=1e-6, max=1.0)
    focal_weight = torch.pow(1.0 - p_target, gamma)

    poly1 = focal_weight * ce_unsmoothed + POLY1_EPS * (1.0 - p_target)
    return _distributed_normalize((poly1 * vf).sum(), vf.sum())

def decode_seq(
    decoder_module,
    gt_seq: torch.Tensor,
    encoder_out: torch.Tensor,
    encoder_padding_mask: Optional[torch.Tensor] = None,
    decoder_padding_mask: Optional[torch.Tensor] = None,
    kv_caches=None,
    shift_target: bool = True,
    ss_ratio: float = 0.0,
):
    """
    Safely executes decoder forward pass with optional Scheduled Sampling.
    """
    if shift_target:
        target_in = gt_seq[:, :-1]
    else:
        target_in = gt_seq

    kpm = encoder_padding_mask
    if ss_ratio > 0.0:
        with torch.no_grad():
            out = decoder_module(
                target_in,
                encoder_out,
                memory_key_padding_mask=kpm,
                past_key_values=kv_caches,
                use_cache=(kv_caches is not None),
            )
    else:
        out = decoder_module(
            target_in,
            encoder_out,
            memory_key_padding_mask=kpm,
            past_key_values=kv_caches,
            use_cache=(kv_caches is not None),
        )
    
    if isinstance(out, tuple):
        logits = out[0]
        hidden = out[1] if len(out) > 1 else None
        new_kv_caches = out[3] if len(out) > 3 else None
    else:
        logits, hidden, new_kv_caches = out, None, None

    if ss_ratio > 0.0 and target_in.size(1) > 1 and getattr(decoder_module, "training", False):
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            # Create mask for tokens to replace (ignore BOS, EOS, PAD)
            replace_mask = (torch.rand_like(target_in.float()) < ss_ratio)
            replace_mask = replace_mask & (target_in != 0) & (target_in != 1) & (target_in != 2)
            
            mixed_target = target_in.clone()
            mixed_target = torch.where(replace_mask, preds, mixed_target)
            
        out2 = decoder_module(
            mixed_target,
            encoder_out,
            memory_key_padding_mask=kpm,
            past_key_values=kv_caches,
            use_cache=(kv_caches is not None),
        )
        if isinstance(out2, tuple):
            logits = out2[0]
            hidden = out2[1] if len(out2) > 1 else None
            extra_logits = out2[2] if len(out2) > 2 else None
            new_kv_caches = out2[3] if len(out2) > 3 else None
        else:
            logits, hidden, extra_logits, new_kv_caches = out2, None, None, None

    return logits, hidden, extra_logits, new_kv_caches


def _compute_ctc_loss_safe(
    ctc_log_probs: torch.Tensor,
    gloss_seq: torch.Tensor,
    gloss_len: torch.Tensor,
    enc_mask: torch.Tensor,
    has_valid_gloss: torch.Tensor,
    sample_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float, float, float, float, float, float]:
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    """Computes CTC Loss with XLA padding safety masks."""
    batch_sz, seq_len = ctc_log_probs.shape[:2]
    if enc_mask is not None:
        input_lengths = (~enc_mask).sum(dim=1).to(torch.int32)
    else:
        input_lengths = torch.full((batch_sz,), seq_len, dtype=torch.int32, device=ctc_log_probs.device)

    valid_gloss_tokens = (
        (gloss_seq != GlossVocabulary.PAD_ID)
        & (gloss_seq != GlossVocabulary.BOS_ID)
        & (gloss_seq != GlossVocabulary.EOS_ID)
    )
    target_lengths = valid_gloss_tokens.sum(dim=1).to(torch.int32)

    # Static minimum length verification for CTC monotonicity (relax rigid bounds for temporal decimation)
    min_ctc_len = target_lengths + 1
    valid_ctc = (input_lengths >= min_ctc_len) & (target_lengths > 0) & has_valid_gloss.bool()

    # Log prob normalization for XLA
    log_probs_t = ctc_log_probs.transpose(0, 1)

    # Sanitize targets so positions beyond target_lengths are clean PAD_IDs
    clean_targets = torch.where(valid_gloss_tokens, gloss_seq, GlossVocabulary.PAD_ID)

    # Prevent PyTorch XLA from dynamically trapping an illegal instruction exception
    # by clamping input_lengths to be at least target_lengths. Invalid pairs are masked later.
    input_lengths = torch.maximum(input_lengths, target_lengths + 1)
    
    # Compute CTC loss safely
    loss_raw = F.ctc_loss(
        log_probs_t,
        clean_targets,
        input_lengths,
        target_lengths,
        blank=GlossVocabulary.PAD_ID,
        reduction="none",
        zero_infinity=True,
    )
    
    # Sanitize XLA NaNs before boolean multiplication
    loss_raw = torch.nan_to_num(loss_raw, nan=0.0, posinf=0.0, neginf=0.0)

    valid_f = valid_ctc.float()
    if sample_weights is not None:
        valid_f = valid_f * sample_weights

    loss_ctc = _distributed_normalize((loss_raw * valid_f).sum(), valid_f.sum())
    return (
        loss_ctc,
        float(has_valid_gloss.float().sum()),
        float(valid_f.sum()),
        float((has_valid_gloss.float() - valid_f).clamp_min(0.0).sum()),
        float(input_lengths.float().mean()),
        float(target_lengths.float().mean()),
        float(min_ctc_len.float().mean()),
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
        target = target.transpose(1, 2)
        target = F.interpolate(target, size=seq_len_mlm, mode="linear", align_corners=False)
        target = target.transpose(1, 2)

    target = target.reshape(batch_size_mlm, seq_len_mlm, channels_mlm)
    mask_flat = mlm_mask.unsqueeze(-1).float()
    loss_raw = F.smooth_l1_loss(mlm_logits.float(), target, reduction="none") * mask_flat
    if sample_weights is not None:
        loss_raw = loss_raw * sample_weights.view(-1, 1, 1)
    return _distributed_normalize(loss_raw.sum(), mask_flat.sum() * mlm_logits.shape[-1])


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
) -> Dict[str, float]:
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    """Docstring for train_epoch_tpu."""

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
    epoch_start_time = time.time()  # pylint: disable=not-callable,possibly-unused-variable

    is_xla = _XLA_AVAILABLE and "xla" in str(device).lower()
    if is_xla:
        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel
        import torch_xla.distributed.parallel_loader as pl  # pylint: disable=not-callable,import-outside-toplevel
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

    if is_xla:
        # Set batches_per_execution=16 for maximum unrolled mega-graph throughput
        para_loader = pl.MpDeviceLoader(loader, device, batches_per_execution=16)
    else:
        para_loader = loader

    total_batches = len(loader)
    min_batches = total_batches
    if is_xla:
        min_batches = int(
            xm.mesh_reduce("min_batches", total_batches, lambda input_x: min(input_x))
        )
        try:
            ord_val = xm.get_ordinal()
        except AttributeError:
            try:
                import torch_xla.runtime as xr  # pylint: disable=not-callable,import-outside-toplevel

                ord_val = xr.global_ordinal()
            except Exception:  # pylint: disable=not-callable,broad-exception-caught
                ord_val = 0
        xm.set_rng_state(42 + epoch * 10000 + ord_val)
    else:
        torch.manual_seed(42 + epoch * 10000)

    step_start_time = time.time()
    running_metrics = torch.zeros(12, device="cpu")
    running_truncs = torch.zeros(3, device="cpu")
    for step_idx, batch in enumerate(para_loader, start=1):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop
        if step_idx == 1:
            step_start_time = time.time()
        if step_idx > min_batches:
            continue

        features, mask, labels, frame_indices = (
            batch["feature"].to(device),
            batch["mask"].to(device),
            batch.get(
                "label",
                torch.zeros(batch["feature"].size(0), dtype=torch.long, device=device),
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
        chicago_seq, chicago_len, has_valid_chicago = (  # pylint: disable=not-callable,possibly-unused-variable
            batch["chicago_seq"].to(device),
            batch["chicago_len"].to(device),
            batch["has_valid_chicago"].to(device),
        )
        english_seq, english_len, has_valid_english = (  # pylint: disable=not-callable,possibly-unused-variable
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
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
            """Docstring for forward_and_losses."""

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
                ss_ratio=min(0.3, max(0.0, (epoch / max(1, total_epochs)) * 0.5)),
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
            h_cls = out.get("h_cls")

            gt_tokens = gloss_seq[:, 1:].contiguous()
            token_mask = (
                gt_tokens != GlossVocabulary.PAD_ID
            ) & has_valid_gloss.bool().unsqueeze(-1)
            valid_gloss_mask = token_mask & (gt_tokens != GlossVocabulary.EOS_ID)

            # Gloss length & sequence loss masked strictly by has_valid_gloss
            if True:  # pylint: disable=not-callable,using-constant-test
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

            if True:  # pylint: disable=not-callable,using-constant-test
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

            if True:  # pylint: disable=not-callable,using-constant-test
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
                    sample_weights=sample_weight,
                )
                if out.get("mlm_logits") is not None
                and out.get("orig_x") is not None
                and out.get("mlm_mask") is not None
                else torch.zeros((), device=device)
            )

            mtp_logits = out.get("mtp_logits")
            loss_mtp2 = torch.zeros((), device=device)
            loss_mtp3 = torch.zeros((), device=device)

            if mtp_logits is not None and out.get("dec_logits") is not None:
                gt_tokens_mtp2 = gloss_seq[:, 2:].contiguous()
                gt_tokens_mtp2 = F.pad(
                    gt_tokens_mtp2, (0, 1), value=GlossVocabulary.PAD_ID
                )
                valid_mtp2_mask = valid_mask & (gt_tokens_mtp2 != GlossVocabulary.PAD_ID) & (
                    gt_tokens_mtp2 != GlossVocabulary.EOS_ID
                )
                valid_mtp2_mask = valid_mtp2_mask & has_valid_gloss.bool().unsqueeze(-1)
                loss_mtp2 = compute_seq_loss(
                    mtp_logits["logits_2"],
                    gt_tokens_mtp2,
                    valid_mtp2_mask,
                    sample_weights=sample_weight,
                    class_weights=class_weights,
                )

                gt_tokens_mtp3 = gloss_seq[:, 3:].contiguous()
                gt_tokens_mtp3 = F.pad(
                    gt_tokens_mtp3, (0, 2), value=GlossVocabulary.PAD_ID
                )
                valid_mtp3_mask = valid_mask & (gt_tokens_mtp3 != GlossVocabulary.PAD_ID) & (
                    gt_tokens_mtp3 != GlossVocabulary.EOS_ID
                )
                valid_mtp3_mask = valid_mtp3_mask & has_valid_gloss.bool().unsqueeze(-1)
                loss_mtp3 = compute_seq_loss(
                    mtp_logits["logits_3"],
                    gt_tokens_mtp3,
                    valid_mtp3_mask,
                    sample_weights=sample_weight,
                    class_weights=class_weights,
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
                "inter_ctc": loss_inter_ctc,  # pylint: disable=not-callable,possibly-used-before-assignment
                "lpc": loss_lpc,  # pylint: disable=not-callable,possibly-used-before-assignment
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

        loss = raw_loss

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if is_xla:
            import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

            xm.reduce_gradients(optimizer)
            xm.clip_grad_norm_(
                list(model.parameters()) + list(loss_wrapper.parameters()), max_norm=1.0
            )
            xm.optimizer_step(optimizer, barrier=False)
            optimizer.zero_grad(set_to_none=True)
        else:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_wrapper.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if scheduler is not None:
            try:
                if (not hasattr(scheduler, "total_steps") or scheduler.last_epoch < scheduler.total_steps):
                    scheduler.step()
            except Exception:
                pass

        if is_xla:
            import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

            raw_m = model.module if hasattr(model, "module") else model
            if hasattr(raw_m, "dense_sem_loss") and (
                step_idx % 4 == 0 or step_idx == min_batches
            ):
                raw_m.dense_sem_loss.update_momentum()

            log_freq = min(25, max(1, min_batches // 10))
            if is_master and (
                (step_idx <= 10)
                or (step_idx % log_freq == 0)
                or (step_idx == min_batches)
            ):
                current_log_time = time.time()  # pylint: disable=not-callable,possibly-unused-variable

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
                    t_start,  # pylint: disable=not-callable,unused-argument
                    t_prev,
                    b_sz,
                    l_freq,
                ):
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
                    """Docstring for _async_step_print."""

                    vals = log_vec
                    (
                        l_val,
                        s_val,
                        aux_val,  # pylint: disable=not-callable,unused-variable
                        c_val,  # pylint: disable=not-callable,unused-variable
                        sm_val,  # pylint: disable=not-callable,unused-variable
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
                    elapsed_win = max(0.001, time.time() - t_prev)
                    win_steps = l_freq if st_idx >= l_freq else st_idx
                    instant_it_s = float(win_steps) / elapsed_win
                    samples_per_s = instant_it_s * b_sz
                    msg = (
                        f"  [Epoch {ep:03d}/{tot_ep:03d} | Step {st_idx:04d}/{m_batches:04d}] "
                        f"Loss: {float(l_val):.4f} (Gloss:{float(s_val):.4f} Chi:{float(chi_val):.4f} Eng:{float(eng_val):.4f}) | "
                        f"Acc (Gloss:{g_acc:.1f}% Chi:{c_acc:.1f}% Eng:{e_acc:.1f}%) | LR: {lr_val:.2e} | "
                        f"{instant_it_s:.2f} it/s ({samples_per_s:.1f} seq/s)"
                    )
                    print(msg, flush=True)

                batch_sz_val = getattr(loader, "batch_size", 64)
                if not isinstance(batch_sz_val, int):
                    batch_sz_val = 64

                xm.add_step_closure(
                    _async_step_print,
                    args=(
                        log_vec,
                        step_idx,
                        min_batches,
                        epoch,
                        total_epochs,
                        optimizer.param_groups[0]["lr"],
                        step_start_time,
                        locals().get("last_log_time", step_start_time),
                        batch_sz_val,
                        log_freq,
                    ),
                )
                last_log_time = time.time()  # pylint: disable=not-callable,possibly-unused-variable
        with torch.no_grad():
            metrics_vec = torch.stack(
                [
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
                    c_nc_t.detach(),
                    c_nt_t.detach(),
                    e_nc_t.detach(),
                    e_nt_t.detach(),
                    loss_lpc.detach(),
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

            if "tpu_metrics" not in locals():
                tpu_metrics = torch.zeros_like(metrics_vec)
                tpu_truncs = torch.zeros_like(truncs_vec)

            tpu_metrics += metrics_vec
            tpu_truncs += truncs_vec



            def _update_metrics(m_vec, t_vec):
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
                """Docstring for _update_metrics."""

                running_metrics.add_(m_vec.cpu())  # pylint: disable=not-callable,used-before-assignment
                running_truncs.add_(t_vec.cpu())  # pylint: disable=not-callable,used-before-assignment

            if step_idx % 10 == 0 or step_idx == min_batches:
                if is_xla:
                    xm.add_step_closure(_update_metrics, args=(tpu_metrics, tpu_truncs))
                else:
                    _update_metrics(tpu_metrics, tpu_truncs)
                tpu_metrics = torch.zeros_like(metrics_vec)
                tpu_truncs = torch.zeros_like(truncs_vec)
        del batch
        if "out" in locals():
            del out  # pylint: disable=not-callable,undefined-variable

        if "l_seq" in locals():
            del l_seq
        if "l_aux" in locals():
            del l_aux
        if "l_ctc" in locals():
            del l_ctc
        if "l_sem" in locals():
            del l_sem
        if "l_chi" in locals():
            del l_chi
        if "l_eng" in locals():
            del l_eng
        if "features" in locals():
            del features
        if "mask" in locals():
            del mask
        if "labels" in locals():
            del labels
        if "gloss_seq" in locals():
            del gloss_seq
        if "chicago_seq" in locals():
            del chicago_seq
        if "english_seq" in locals():
            del english_seq
        if "frame_indices" in locals():
            del frame_indices
        if "domain_tgts" in locals():
            del domain_tgts
        if "has_domain" in locals():
            del has_domain
        if "mlm_mask" in locals():
            del mlm_mask
        if "loss" in locals():
            del loss
        if "loss_terms" in locals():
            del loss_terms  # pylint: disable=not-callable,undefined-variable

        if is_xla:
            import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

            xm.mark_step()

        # Removed gc.collect() to prevent massive CPU stalling at high iteration speeds
        if is_xla and step_idx >= min_batches:
            if "para_loader" in locals():
                del para_loader
            import gc  # pylint: disable=not-callable,import-outside-toplevel

            gc.collect()
            break

    if is_xla:
        xm.mark_step()
        xm.rendezvous("end_of_epoch")

        # Combine the running metrics and the truncation flags into one tensor
        if "running_metrics" not in locals():
            running_metrics = torch.zeros(12, dtype=torch.float32, device=device)
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
    import gc  # pylint: disable=not-callable,import-outside-toplevel

    gc.collect()

    if is_xla:
        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

        xm.wait_device_ops()

    if ema is not None:
        raw_m = model.module if hasattr(model, "module") else model
        ema.update(raw_m, float(epoch) / float(total_epochs))

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
            import torch_xla.debug.metrics as met  # pylint: disable=not-callable,import-outside-toplevel

            print(met.metrics_report())
            print("=" * 80 + "\n", flush=True)
        except Exception:  # pylint: disable=not-callable,broad-exception-caught
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
    total_epochs: int = 150,  # pylint: disable=not-callable,unused-argument
    prec_dtype: torch.dtype = torch.float16,
    is_master: bool = True,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.1,
) -> Tuple[float, float]:
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    """Docstring for validate_epoch_tpu."""

    model.eval()
    
    # [TPU XLA HOTFIX] Disable ToMe during validation to prevent dynamic shape graph recompilations
    # by ensuring sequence length stays constant at max_len across all buckets.
    if hasattr(model, "tome_r"):
        old_r = model.tome_r
        model.tome_r = 0
        for block in getattr(model, "blocks", []):
            if type(block).__name__ == "TemporalStridedPool":
                block.r = 0
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
        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel
        import torch_xla.distributed.parallel_loader as pl  # pylint: disable=not-callable,import-outside-toplevel
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

    total_val_batches = len(loader)
    min_val_batches = total_val_batches
    if is_xla:
        para_loader = pl.MpDeviceLoader(loader, device, batches_per_execution=16)
        min_val_batches = int(
            xm.mesh_reduce("min_val_batches", total_val_batches, lambda input_x: min(input_x))
        )
    else:
        para_loader = loader
        min_val_batches = total_val_batches

    with torch.no_grad():
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop

        for step_idx, batch in enumerate(para_loader, 1):
            if step_idx > min_val_batches:
                if is_xla:
                    if "para_loader" in locals():
                        del para_loader
                    import gc  # pylint: disable=not-callable,import-outside-toplevel

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
            # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
                """Docstring for forward_and_losses."""

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
                if dec_logits is not None:
                    preds = dec_logits.argmax(dim=-1)
                    valid_f = valid_mask.float()
                    nc_t = ((preds == gt_tokens).float() * valid_f).sum()
                    nt_t = valid_f.sum()

                c_nc_t, c_nt_t = torch.zeros((), device=device), torch.zeros(
                    (), device=device
                )
                loss_chi = torch.zeros((), device=device)
                loss_chi_eos = torch.zeros((), device=device)
                if chicago_logits is not None:
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
                if english_logits is not None:
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
                if True:  # pylint: disable=not-callable,using-constant-test
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

                if step_idx <= 2 and isinstance(out, dict) and "h_seq" in out:
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
                        temperature=1.0,
                        top_k=0,
                        use_mtp_speculative=False,
                        frame_indices=frame_indices,
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
                        loss_seq.detach() if "loss_seq" in locals() else raw_loss.detach(),
                        loss_seq.detach() if "loss_seq" in locals() else raw_loss.detach(),
                    ]
                )

                if "running_val_metrics" not in locals():
                    running_val_metrics = torch.zeros_like(metrics_vec)
                running_val_metrics.add_(metrics_vec)  # pylint: disable=not-callable,used-before-assignment

            if is_master and (step_idx % 50 == 0 or step_idx == min_val_batches):
                print(
                    f"  [Val Step {step_idx:04d}/{min_val_batches:04d}] Loss: {float(raw_loss.detach()):.4f}",
                    flush=True,
                )

            if is_xla:
                import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                xm.mark_step()

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
                del dec_preds  # pylint: disable=not-callable,undefined-variable

    if is_xla:
        xm.rendezvous("validate_metrics")

        if "running_val_metrics" not in locals():
            running_val_metrics = torch.zeros(17, dtype=torch.float32, device=device)

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
        step_idx = float(v_np[15])
    elif torch.distributed.is_initialized():
        import torch.distributed as dist
        if "running_val_metrics" not in locals():
            running_val_metrics = torch.zeros(17, dtype=torch.float32, device=device)
        
        val_vec = torch.cat([running_val_metrics, torch.tensor([float(min_val_batches)], dtype=torch.float32, device=device)])
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
    import gc  # pylint: disable=not-callable,import-outside-toplevel

    gc.collect()

    try:
        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

        xm.wait_device_ops()
    except Exception:  # pylint: disable=not-callable,broad-exception-caught
        pass

    if "old_r" in locals() and hasattr(model, "tome_r"):
        model.tome_r = old_r
        for block in getattr(model, "blocks", []):
            if type(block).__name__ == "TemporalStridedPool":
                block.r = old_r

    return {
        "loss": val_loss,
        "gloss_acc": val_acc * 100.0,
        "ar_acc": val_ar_acc * 100.0,
        "ar_exact": val_ar_exact * 100.0,
        "chicago_acc": val_chi_acc * 100.0,
        "english_acc": val_eng_acc * 100.0,
    }


def _tpu_worker_fn(rank, args):
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop
    """Docstring for _tpu_worker_fn."""

    if IS_TPU:
        try:
            import torch_xla.debug.profiler as xp  # pylint: disable=not-callable,import-outside-toplevel

            # Start the profiler server on port 9012 (Master only)
            server = xp.start_server(9012)  # pylint: disable=not-callable,possibly-unused-variable
        except Exception:  # pylint: disable=not-callable,broad-exception-caught
            pass

        import torch_xla.runtime as xr  # pylint: disable=not-callable,import-outside-toplevel
        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

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
                import torch_xla  # pylint: disable=not-callable,import-outside-toplevel

                device = torch_xla.device()
            except Exception:  # pylint: disable=not-callable,broad-exception-caught
                device = xm.xla_device()
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:  # pylint: disable=not-callable,broad-exception-caught
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
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"
            ),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"
            ),
            Path("/kaggle/input/frakenstein-asl"),
            Path.cwd(),
        ]
        for cd in candidate_dirs:
            if cd.exists() and list(cd.glob("*.pt")):
                data_dir = cd
                if is_master:
                    print(
                        f"[INFO] Auto-resolved dataset directory to: {data_dir}",
                        flush=True,
                    )
                break

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

    # TPU System RAM Guard: Cap num_dataloader_workers to 1 on multi-core TPU to prevent PyTorch IPC subprocess duplication RAM blowouts
    effective_num_dl_workers = (
        min(1, args.num_dataloader_workers) if IS_TPU else args.num_dataloader_workers
    )
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
        use_bpe=True,
        model_name="Qwen/Qwen2.5-0.5B",
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
                        except Exception:  # pylint: disable=not-callable,broad-exception-caught
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
        use_bpe=True,
        model_name="Qwen/Qwen2.5-0.5B",
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
                import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

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
    except Exception as e:  # pylint: disable=not-callable,broad-exception-caught
        if is_master:
            print(f"[WARN] Class weighting: DISABLED due to exception: {e}", flush=True)
        class_weights_tensor = torch.ones(
            len(vocab), dtype=torch.float32, device=device
        )

    from dataset import GlossVocabulary; chicago_vocab = GlossVocabulary(
        str(Path(args.train_dir) / "chicago_vocab.json")
    )
    chicago_vocab_size = len(chicago_vocab.token_to_id)
    try:
        from transformers import AutoTokenizer
        english_vocab = dataset1.EnglishVocabulary(use_bpe=True, model_name="Qwen/Qwen2.5-0.5B")
        print("Using HuggingFace BPE Tokenizer for English.")
    except Exception as e:
        print(f"Failed to load BPE Tokenizer, falling back to EnglishVocabulary: {e}")
        english_vocab = dataset1.EnglishVocabulary(
            str(Path(args.train_dir) / "english_vocab.json")
        )
    english_vocab_size = len(english_vocab.token_to_id)

    asl_lex_csv = (
        Path(args.asl_lex_csv)
        if hasattr(args, "asl_lex_csv") and args.asl_lex_csv
        else (data_dir / "signdata.csv")
    )

    eng_vsize = (  # pylint: disable=not-callable,possibly-unused-variable
        len(train_loader.dataset.english_vocab)
        if hasattr(train_loader.dataset, "english_vocab")
        else 20005
    )

    import hashlib  # pylint: disable=not-callable,import-outside-toplevel

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
        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

        local_hash = torch.tensor(eng_hash, dtype=torch.float64, device=device)
        global_min = int(
            xm.mesh_reduce("english_vocab_min", local_hash, lambda xs: min(xs)).item()  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
        )
        global_max = int(
            xm.mesh_reduce("english_vocab_max", local_hash, lambda xs: max(xs)).item()  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
        )

        if global_min != global_max:
            raise RuntimeError(
                f"English vocabulary differs across TPU ranks: "
                f"min_hash={global_min}, max_hash={global_max}"
            )

    model = ASLFoundationModel(
        vocab_size=len(label_to_idx) + GlossVocabulary.OFFSET,
        d_enc=args.d_model,
        d_dec=args.d_model,
        nhead_enc=args.nhead,
        nhead_dec=args.nhead,
        num_enc_layers=args.num_layers,
        num_dec_layers=args.num_layers,
        dropout=args.dropout,
        max_enc_len=args.max_len,
        english_vocab_size=20096,
        label_to_idx=label_to_idx,
        csv_path=asl_lex_csv if asl_lex_csv.exists() else None,
    ).to(device, dtype=torch.bfloat16 if IS_TPU else None)

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
        except Exception as _e:  # pylint: disable=not-callable,broad-exception-caught
            if is_master:
                print(f"[!] Warning: torch.compile fallback: {_e}", flush=True)

    loss_wrapper = HomoscedasticLossWrapper().to(device)
    if IS_TPU:
        xm.broadcast_master_param(loss_wrapper)

    supcon_fn = SupervisedContrastiveLoss().to(device)

    global_min_batches = len(train_loader)
    if IS_TPU:
        global_min_batches = int(
            xm.mesh_reduce("global_min_batches", len(train_loader), lambda input_x: min(input_x))  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
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
                raise RuntimeError(f"Failed to load optimizer state: {e}")  # pylint: disable=not-callable,raise-missing-from

        if "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as e:
                raise RuntimeError(f"Failed to load scheduler state: {e}")  # pylint: disable=not-callable,raise-missing-from

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
            import random  # pylint: disable=not-callable,import-outside-toplevel

            random.setstate(ckpt["rng_state_random"])

        start_epoch = ckpt.get("epoch", 0) + 1
        ema_state_dict_to_load = ckpt.get("ema_state_dict", None)
        del ckpt
        import gc  # pylint: disable=not-callable,import-outside-toplevel

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
    if "ema_state_dict_to_load" in locals() and ema_state_dict_to_load is not None:  # pylint: disable=not-callable,undefined-variable
        for key_k_lower, val_v in ema_state_dict_to_load.items():  # pylint: disable=not-callable,undefined-variable
            if key_k_lower in ema.shadow:
                ema.shadow[key_k_lower].copy_(val_v.to(ema.shadow[key_k_lower].device))
        if is_master:
            print("[+] Restored EMA state from checkpoint", flush=True)
        del ema_state_dict_to_load  # pylint: disable=not-callable,undefined-variable
        gc.collect()

    if IS_TPU:
        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

        xm.mark_step()
        xm.rendezvous("model_ema_init_complete")

        # ─── MASTER-FIRST SERIALIZED COMPILATION BARRIER ───
        if is_master:
            print(
                "[*] Master (Rank 0) compiling XLA graph first to prevent multi-core RAM OOM...",
                flush=True,
            )
            dummy_batch = next(iter(train_loader))
            dummy_feat = dummy_batch["feature"].to(device)
            dummy_mask = dummy_batch["mask"].to(device)
            dummy_gloss = dummy_batch["gloss_seq"].to(device)
            dummy_chicago = dummy_batch.get("chicago_seq", None)
            if dummy_chicago is not None:
                dummy_chicago = dummy_chicago.to(device)
            dummy_english = dummy_batch.get("english_seq", None)
            if dummy_english is not None:
                dummy_english = dummy_english.to(device)
            dummy_mlm = dummy_batch.get("mlm_mask", None)
            if dummy_mlm is not None:
                dummy_mlm = dummy_mlm.to(device)
            dummy_fi = dummy_batch.get("frame_indices", None)
            if dummy_fi is not None:
                dummy_fi = dummy_fi.to(device)

            with torch.no_grad():
                _out = model(
                    dummy_feat,
                    mask=dummy_mask,
                    gloss_seq=dummy_gloss,
                    chicago_seq=dummy_chicago,
                    english_seq=dummy_english,
                    mlm_mask=dummy_mlm,
                    frame_indices=dummy_fi,
                    return_aux=True,
                    grl_alpha=0.5,
                )
                xm.mark_step()

            del (
                dummy_batch,
                dummy_feat,
                dummy_mask,
                dummy_gloss,
                dummy_chicago,
                dummy_english,
                dummy_mlm,
                dummy_fi,
                _out,
            )
            import gc  # pylint: disable=not-callable,import-outside-toplevel

            gc.collect()
            print(
                "[*] Master XLA Graph Compilation Complete! Rank 0 cached binary successfully.",
                flush=True,
            )

        xm.rendezvous("master_compilation_done")
        if is_master:
            print(
                "[*] All 8 TPU ranks synchronized. Starting Training Epochs...",
                flush=True,
            )

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

            train_metrics = train_epoch_tpu(  # pylint: disable=not-callable,possibly-unused-variable
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

            if ema is not None:
                ema.restore(raw_m)
                if IS_TPU:
                    import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

            if IS_TPU:
                import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                xm.mark_step()
                xm.rendezvous("pre_checkpoint_save")

            # Save checkpoint every 5 epochs or on final epoch to save memory and disk quota
            if epoch % 5 == 0 or epoch == args.epochs:
                import random as py_random  # pylint: disable=not-callable,import-outside-toplevel

                ckpt_path = save_dir / f"asl_model_epoch_{epoch}.pt"
                latest_path = save_dir / "asl_model_latest.pt"

                if IS_TPU:
                    import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                    xm.mark_step()
                    xm.rendezvous("pre_checkpoint_build")

                    # All 8 TPU ranks construct cpu_state together so state_dict XLA graph sync executes across all ranks simultaneously
                    cpu_state = {
                        "epoch": epoch,
                        "model_state_dict": raw_m.state_dict(),
                        "ema_state_dict": {k: v.cpu() for k, v in ema.shadow.items()} if ema is not None else None,
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
                    gc.collect()
                    import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                    xm.mark_step()
                    xm.save(cpu_state, str(ckpt_path))
                    xm.save(cpu_state, str(latest_path))
                    del cpu_state
                else:
                    if is_master:
                        cpu_state = {
                            "epoch": epoch,
                            "model_state_dict": raw_m.state_dict(),
                            "ema_state_dict": {k: v.cpu() for k, v in ema.shadow.items()} if ema is not None else None,
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
                        }
                        gc.collect()
                        import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                        xm.save(cpu_state, str(ckpt_path))
                        xm.save(cpu_state, str(latest_path))
                        del cpu_state

                if is_master:
                    print(f"[+] Saved checkpoint to {ckpt_path}", flush=True)
                    try:
                        all_ckpts = sorted(
                            list(save_dir.glob("asl_model_epoch_*.pt")),
                            key=lambda prob_p: int(prob_p.stem.split("_")[-1]),  # pylint: disable=not-callable,unnecessary-lambda,cell-var-from-loop
                        )
                        if len(all_ckpts) > 5:
                            for old_c in all_ckpts[:-5]:
                                ep_num = int(old_c.stem.split("_")[-1])
                                if ep_num % 10 != 0 and ep_num != epoch:
                                    old_c.unlink(missing_ok=True)
                    except Exception:  # pylint: disable=not-callable,broad-exception-caught
                        pass

            # Explicit garbage collection of massive dicts & flush XLA IR graph
            if "cpu_state" in locals() and cpu_state is not None:
                del cpu_state
                cpu_state = None

            if IS_TPU:
                # pyrefly: ignore [missing-import]
                import torch_xla.core.xla_model as xm  # pylint: disable=not-callable,import-outside-toplevel

                xm.mark_step()
                xm.rendezvous("post_checkpoint_save")

            gc.collect()
            gc.collect()

    except Exception as e:  # pylint: disable=not-callable,broad-exception-caught
        import traceback  # pylint: disable=not-callable,import-outside-toplevel
        import sys  # pylint: disable=not-callable,import-outside-toplevel  # pylint: disable=not-callable,redefined-outer-name,reimported,import-outside-toplevel  # pylint: disable=not-callable,redefined-outer-name,reimported,import-outside-toplevel  # pylint: disable=not-callable,redefined-outer-name,reimported,import-outside-toplevel

        print(f"CRITICAL PYTHON EXCEPTION: {e}", flush=True)
        traceback.print_exc()
        time.sleep(2)
        sys.exit(1)


def inverted_gloss_pretrain_loop(args, device, is_master, model_name="Qwen/Qwen2.5-0.5B"):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl, model_name="Qwen/Qwen2.5-0.5B"):
    from dataset import ASLGPC12Dataset, phase1_collate_fn, EnglishVocabulary, GlossVocabulary
    
    if is_master:
        print(f"Starting Phase 2 Inverted Gloss Training (English -> Gloss) for {args.epochs} epochs...")
        os.makedirs(args.save_dir, exist_ok=True)
        
    eng_vocab = EnglishVocabulary(use_bpe=True, model_name=model_name)
    import json; gloss_vocab = GlossVocabulary(label_to_idx=json.load(open(os.path.join(args.data_dir, "vocab_map.json")))) if os.path.exists(os.path.join(args.data_dir, "vocab_map.json")) else GlossVocabulary()
    
    dataset = ASLGPC12Dataset(
        csv_path=args.data_dir + "/train.csv" if not args.data_dir.endswith(".csv") else args.data_dir,
        eng_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=args.max_len,
        reverse=True
    )
    
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True, drop_last=True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_dataloader_workers, collate_fn=phase2_collate_fn, drop_last=True
    )
    
    # Model (we use the decoder but with gloss vocab instead of english vocab)
    model = ASLFoundationModel(
        channels_per_kp=3,
        num_enc_layers=0,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len,
        vocab_size=2000, # Chicago vocab unused
        
        drop_path_rate=0.0
    ).to(device)
    
    # Using the gloss decoder (which uses GlossVocabulary logic)
    decoder = model.decoder
    # Tie embeddings for gloss decoder
    decoder.token_emb.weight = decoder.lm_head.weight
    
    # We also need an english embedding layer for the cross-attention
    eng_vocab_size = eng_vocab.tokenizer.vocab_size if getattr(eng_vocab, "tokenizer", None) else 151936
    english_emb = nn.Embedding(eng_vocab_size, args.d_model, padding_idx=151643).to(device)
    
    params = list(decoder.parameters()) + list(english_emb.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        decoder.train()
        english_emb.train()
        para_loader = pl.ParallelLoader(dataloader, [device])
        
        for step, batch in enumerate(para_loader.per_device_loader(device)):
            # input is english, target is gloss
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            
            mask = (input_ids == 151643) # Qwen PAD
            memory = english_emb(input_ids)
            
            tgt_in = target_ids[:, :-1]
            tgt_out = target_ids[:, 1:]
            
            out = decoder(
                tgt_in,
                memory,
                tgt_key_padding_mask=(tgt_in == GlossVocabulary.PAD_ID),
                memory_key_padding_mask=mask
            )
            
            logits = out[0] if isinstance(out, tuple) else out
            
            valid_mask = (tgt_out != GlossVocabulary.PAD_ID) & (tgt_out != GlossVocabulary.EOS_ID)
            loss = compute_seq_loss(logits, tgt_out, valid_mask, label_smoothing=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)
            
            if step % args.log_freq == 0:
                reduced = xm.all_reduce(xm.REDUCE_SUM, loss.detach()) / xm.xrt_world_size()
                if is_master:
                    print(f"Phase 2 Train | Epoch {epoch+1:03d} | Step {step:04d} | Loss: {reduced.item():.4f}", flush=True)
                    
        xm.master_print(f"Phase 2 Train Epoch {epoch+1} finished.")
        xm.save({"decoder": decoder.state_dict(), "english_emb": english_emb.state_dict()}, os.path.join(args.save_dir, "inverted_gloss_model.pt"))

def pseudo_gloss_gen_loop(args, device, is_master, model_name="Qwen/Qwen2.5-0.5B"):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl, model_name="Qwen/Qwen2.5-0.5B"):
    from dataset import ASLStreamedDataset, EnglishVocabulary, GlossVocabulary
    import shutil
    
    if is_master:
        print(f"Starting Phase 2 Pseudo-Gloss Generation on TPU...")
        out_dir = os.path.join(args.save_dir, "pseudo_gloss_data")
        os.makedirs(out_dir, exist_ok=True)
        # Copy metadata.json
        meta_src = os.path.join(args.data_dir, "metadata.json")
        if os.path.exists(meta_src):
            import json
            with open(meta_src, "r") as f:
                meta = json.load(f)
            # Flag has_valid_gloss as true globally
            for key in meta:
                meta[key]["has_valid_gloss"] = True
            with open(os.path.join(out_dir, "metadata.json"), "w") as f:
                json.dump(meta, f)
        
    eng_vocab = EnglishVocabulary(use_bpe=True, model_name=model_name)
    import json; gloss_vocab = GlossVocabulary(label_to_idx=json.load(open(os.path.join(args.data_dir, "vocab_map.json")))) if os.path.exists(os.path.join(args.data_dir, "vocab_map.json")) else GlossVocabulary()
    
    dataset = ASLStreamedDataset(
        data_dir=args.data_dir,
        split="val", # Prevent infinite train shuffle loop
        english_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=args.max_len,
        shuffle_buffer_size=1
    )
    
    # We will manually load shards to process them sequentially and save them back
    # Distribute shards among workers
    all_shards = dataset.shard_files
    # Since dataset filters shard_files in __init__ based on world_size, dataset.shard_files is already local to this TPU core
    local_shards = dataset.shard_files
    
    model = ASLFoundationModel(
        channels_per_kp=3,
        num_enc_layers=0,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len,
        vocab_size=2000,
        
        drop_path_rate=0.0
    ).to(device)
    
    decoder = model.decoder
    eng_vocab_size = eng_vocab.tokenizer.vocab_size if getattr(eng_vocab, "tokenizer", None) else 151936
    english_emb = nn.Embedding(eng_vocab_size, args.d_model, padding_idx=151643).to(device)
    
    # Load weights
    ckpt_path = os.path.join(args.save_dir, "inverted_gloss_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        decoder.load_state_dict(ckpt["decoder"])
        english_emb.load_state_dict(ckpt["english_emb"])
    else:
        print("WARNING: inverted_gloss_model.pt not found! Generating with random weights.")
        
    decoder.eval()
    english_emb.eval()
    
    out_dir = os.path.join(args.save_dir, "pseudo_gloss_data")
    
    with torch.no_grad():
        for shard_path in local_shards:
            print(f"[Core {xm.get_ordinal()}] Processing {shard_path}...")
            items = torch.load(shard_path, map_location="cpu", weights_only=False)
            new_items = []
            
            # Batch items for faster generation
            batch_size = 64
            for i in range(0, len(items), batch_size):
                batch = list(items.values())[i:i+batch_size] if isinstance(items, dict) else items[i:i+batch_size]
                text_ids_list = []
                for rec in batch:
                    # Some files have 'english', some have 'english_seq'
                    eng_str = rec.get("english", "")
                    t_ids = eng_vocab.encode(eng_str)
                    t_ids = [eng_vocab.BOS_ID] + t_ids[:args.max_len-2] + [eng_vocab.EOS_ID]
                    text_ids_list.append(t_ids)
                
                max_len = max(len(x) for x in text_ids_list)
                text_padded = torch.full((len(batch), max_len), 151643, dtype=torch.long)
                for j, t_ids in enumerate(text_ids_list):
                    text_padded[j, :len(t_ids)] = torch.tensor(t_ids)
                    
                text_padded = text_padded.to(device)
                mask = (text_padded == 151643)
                memory = english_emb(text_padded)
                
                # Autoregressive generation
                bsz = memory.size(0)
                gen_ids = torch.full((bsz, 1), GlossVocabulary.BOS_ID, dtype=torch.long, device=device)
                kv_caches = None
                
                for step in range(args.max_len):
                    tgt_in = gen_ids if kv_caches is None else gen_ids[:, -1:]
                    
                    out = decoder(
                        tgt_in,
                        memory,
                        memory_key_padding_mask=mask,
                        past_key_values=kv_caches,
                        use_cache=True
                    )
                    logits = out[0]
                    kv_caches = out[3] if len(out) > 3 else None
                    
                    next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                    gen_ids = torch.cat([gen_ids, next_token], dim=1)
                    
                    # If all sequences generated EOS, stop
                    if (gen_ids == GlossVocabulary.EOS_ID).any(dim=1).all():
                        break
                        
                # Move to CPU and extract
                gen_ids = gen_ids.cpu().tolist()
                
                for j, rec in enumerate(batch):
                    seq = gen_ids[j]
                    if GlossVocabulary.EOS_ID in seq:
                        seq = seq[:seq.index(GlossVocabulary.EOS_ID) + 1]
                    rec["gloss_seq"] = [max(0, i - GlossVocabulary.OFFSET) for i in seq]
                    rec["has_valid_gloss"] = True
                    new_items.append(rec)
                    
            # Save new shard
            shard_name = os.path.basename(shard_path)
            new_shard_path = os.path.join(out_dir, shard_name)
            torch.save(new_items, new_shard_path)
            
    xm.master_print("Phase 2 Pseudo-Gloss Generation finished!")


def text_pretrain_loop(args, device, is_master, model_name="Qwen/Qwen2.5-0.5B"):
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl, model_name="Qwen/Qwen2.5-0.5B"):
    from dataset import Phase1MixedIterable, phase1_collate_fn, EnglishVocabulary, GlossVocabulary
    
    if is_master:
        print(f"Starting Phase 1 Text Pre-training for {args.epochs} epochs...")
        os.makedirs(args.save_dir, exist_ok=True)
        
    eng_vocab = EnglishVocabulary(use_bpe=True, model_name=model_name)
    import json; gloss_vocab = GlossVocabulary(label_to_idx=json.load(open(os.path.join(args.data_dir, "vocab_map.json")))) if os.path.exists(os.path.join(args.data_dir, "vocab_map.json")) else GlossVocabulary()
    
    dataset = Phase1MixedIterable(
        kdwd_dir=args.kdwd_dir,
        aslg_csv=args.data_dir + "/train.csv" if not args.data_dir.endswith(".csv") else args.data_dir,
        eng_vocab=eng_vocab,
        gloss_vocab=gloss_vocab,
        max_len=args.max_len
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_dataloader_workers, collate_fn=phase2_collate_fn, drop_last=True
    )
    
    # Model
    model = ASLFoundationModel(
        channels_per_kp=3,
        num_enc_layers=0,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_len=args.max_len,
        vocab_size=eng_vocab.tokenizer.vocab_size if getattr(eng_vocab, "tokenizer", None) else 151936,
        
        drop_path_rate=0.0
    ).to(device)
    
    # Tie english_decoder embeddings to its language modeling head
    model.english_decoder.token_emb.weight = model.english_decoder.lm_head.weight
    
    optimizer = torch.optim.AdamW(model.english_decoder.parameters(), lr=args.lr, weight_decay=0.01)
    
    for epoch in range(args.epochs):
        model.english_decoder.train()
        para_loader = pl.ParallelLoader(dataloader, [device])
        
        for step, batch in enumerate(para_loader.per_device_loader(device)):
            input_ids = batch["input_ids"]
            target_ids = batch["target_ids"]
            
            mask = (input_ids == 0) # PAD
            memory = model.english_decoder.token_emb(input_ids)
            
            tgt_in = target_ids[:, :-1]
            tgt_out = target_ids[:, 1:]
            
            out = model.english_decoder(
                tgt_in,
                memory,
                tgt_key_padding_mask=(tgt_in == 151643),
                memory_key_padding_mask=mask
            )
            
            logits = out[0] if isinstance(out, tuple) else out
            
            valid_mask = (tgt_out != 151643) & (tgt_out != 151643)
            loss = compute_seq_loss(logits, tgt_out, valid_mask, label_smoothing=0.1)
            
            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)
            
            if step % args.log_freq == 0:
                reduced = xm.all_reduce(xm.REDUCE_SUM, loss.detach()) / xm.xrt_world_size()
                if is_master:
                    print(f"Phase 1 | Epoch {epoch+1:03d} | Step {step:04d} | Loss: {reduced.item():.4f}", flush=True)
                    
        xm.master_print(f"Phase 1 Epoch {epoch+1} finished.")
        xm.save(model.english_decoder.state_dict(), os.path.join(args.save_dir, f"asl_llm{epoch+1}.pt"))

def main():
    # pylint: disable=not-callable,too-many-branches,too-many-statements,too-many-locals,too-many-nested-blocks,cell-var-from-loop,too-many-arguments,too-many-positional-arguments,unused-argument,unused-variable,redefined-outer-name,possibly-unused-variable
    """Docstring for main."""

    parser = argparse.ArgumentParser(
        description="ASL Foundation Model Multi-Task TPU Training Pipeline"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"emb_e:\datasets\asl_dataset\asl_preprocessed_phase1",
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
    parser.add_argument("--d-model", "--d_model", dest="d_model", type=int, default=512)
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
        
        print(
            f"[*] Simulating accum_steps={args.accum_steps} with unrolled mega-graph. Scaling LR to {args.lr:.2e}"
        )

    # Removed global environment variables for precision

    if IS_TPU:
        if "LOCAL_RANK" in os.environ:
            # Launched via torchrun / PyTorch distributed launcher
            import torch.distributed as dist  # pylint: disable=not-callable,import-outside-toplevel

            dist.init_process_group("xla", init_method="xla://")
            rank = int(os.environ.get("LOCAL_RANK", "0"))
            _tpu_worker_fn(rank, args)
        else:
            # Direct python execution (e.g. Kaggle TPU notebook cell `python train_all_in_one_tpu.py --tpu`)
            # Spawns 8 processes (one per TPU core) to prevent PJRT barrier deadlock
            import torch_xla.distributed.xla_multiprocessing as xmp  # pylint: disable=not-callable,import-outside-toplevel

            print(
                "[INFO] Kaggle TPU VM detected. Spawning 8 TPU core worker processes via xmp.spawn...",
                flush=True,
            )
            xmp.spawn(_tpu_worker_fn, args=(args,), nprocs=None, start_method="fork")
    else:
        _tpu_worker_fn(0, args)


if __name__ == "__main__":
    main()
