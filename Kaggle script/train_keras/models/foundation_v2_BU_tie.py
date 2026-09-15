"""
Unified SOTA ASL Foundation Model Architecture V2 in Keras 3 (JAX Backend)
Dual-Stream Multimodal (Kinematics 540 + Phonology 19 + Visual ROI 256x256),
Hybrid Conformer + BiMamba-2 SSM with TemporalStridedPool, and Multi-Task Heads.
Optimized for Cloud TPU v5e Systolic MXU Alignment & Instant JIT Execution.
"""

import math
from typing import Dict, Optional, Any
import keras
from keras import layers, ops

from ..layers import (
    RMSNorm,
    SwiGLUFFN,
    get_rotary_frequencies,
    TemporalStridedPool,
    BiMamba2SSMBlock,
    LandmarkTrajectoryStem,
    VisualROI256Stem,
    GatedCrossModalFusion,
)
from .conformer import ConformerBlock
from .decoder import ASLTransformerDecoder


class ASLFoundationModelV2(keras.Model):
    """
    SOTA V2 Dual-Stream Multimodal Continuous ASL Foundation Model.
    Supports:
      - 60-keypoint 9-D kinematics (540 channels)
      - 19-D ASL phonology feature normalization
      - 256x256 upper-body visual ROI crops
      - Hybrid Conformer + BiMamba-2 SSM sequence modeling
      - TemporalStridedPool sequence halving
      - Multi-task heads: CTC, Inter-CTC, Cosine Aux Gloss, Sequence Length
      - Multi-task decoders: Gloss, Chicago Fingerspelling, English Translation
    """
    def __init__(
        self,
        vocab_size: int = 17800,
        chicago_vocab_size: int = 128,
        english_vocab_size: int = 50257,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        d_enc: int = 512,
        nhead_enc: int = 8,
        num_enc_layers: int = 6,
        ffn_enc: int = 1280,
        d_dec: int = 512,
        nhead_dec: int = 8,
        kv_heads_dec: int = 2,
        num_dec_layers: int = 4,
        ffn_dec: int = 1280,
        max_enc_len: int = 384,
        text_max_len: int = 128,
        english_max_len: int = 128,
        chicago_max_len: int = 128,
        kernel_size: int = 31,
        use_mamba: bool = True,
        enable_aux_decoders: bool = True,
        is_causal: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.actual_vocab_size = vocab_size
        self.actual_english_vocab_size = english_vocab_size
        self.actual_chicago_vocab_size = chicago_vocab_size

        # Pad vocabularies to multiples of 128 for TPU v5e systolic tile alignment
        self.vocab_size = (vocab_size + 127) // 128 * 128
        self.english_vocab_size = (english_vocab_size + 127) // 128 * 128
        self.chicago_vocab_size = (chicago_vocab_size + 127) // 128 * 128

        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.kinematics_dim = num_keypoints * channels_per_kp  # 60 * 9 = 540
        self.d_enc = d_enc
        self.nhead_enc = nhead_enc
        self.num_enc_layers = num_enc_layers
        self.ffn_enc = ffn_enc

        self.d_dec = d_dec
        self.nhead_dec = nhead_dec
        self.kv_heads_dec = kv_heads_dec
        self.num_dec_layers = num_dec_layers
        self.ffn_dec = ffn_dec

        self.max_enc_len = max_enc_len
        self.text_max_len = text_max_len
        self.english_max_len = english_max_len
        self.chicago_max_len = chicago_max_len
        self.kernel_size = kernel_size
        self.use_mamba = use_mamba
        self.enable_aux_decoders = enable_aux_decoders
        self.is_causal = is_causal
        self.supports_masking = True

    def build(self, input_shape):
        # 1. Multi-Modal Input Stems
        self.landmark_stem = LandmarkTrajectoryStem(
            in_channels=self.channels_per_kp,
            num_keypoints=self.num_keypoints,
            out_dim=128,
            is_causal=self.is_causal,
            name="landmark_stem",
        )
        self.phonology_norm = RMSNorm(19, name="phonology_norm")

        # 768-aligned input stem projection: 540 (kinematics) + 19 (phonology) + 81 (pad) + 128 (v_tokens) = 768
        self.input_proj = layers.Dense(self.d_enc, use_bias=False, name="input_proj")
        self.input_norm = RMSNorm(self.d_enc, name="input_norm")

        # Visual ROI Stem & Cross-Modal Gated Fusion
        self.visual_stem = VisualROI256Stem(d_model=self.d_enc, name="visual_stem")
        self.cross_modal_fusion = GatedCrossModalFusion(d_model=self.d_enc, name="cross_modal_fusion")

        # 2. Hybrid Conformer + BiMamba-2 Encoder Stack with Strided Pooling
        self.blocks = []
        mid_idx = self.num_enc_layers // 2
        for i in range(self.num_enc_layers):
            if i == mid_idx:
                self.blocks.append(TemporalStridedPool(is_causal=self.is_causal, name=f"strided_pool_{i}"))

            if self.use_mamba and i >= mid_idx:
                self.blocks.append(
                    BiMamba2SSMBlock(
                        d_model=self.d_enc,
                        expand=2,
                        headdim=max(16, self.d_enc // self.nhead_enc),
                        d_state=16,
                        d_conv=4,
                        ffn_dim=self.ffn_enc,
                        is_causal=self.is_causal,
                        name=f"mamba_block_{i}",
                    )
                )
            else:
                self.blocks.append(
                    ConformerBlock(
                        d_model=self.d_enc,
                        nhead=self.nhead_enc,
                        kv_heads=self.kv_heads_dec,
                        dim_feedforward=self.ffn_enc,
                        kernel_size=self.kernel_size,
                        is_causal=self.is_causal,
                        name=f"conformer_block_{i}",
                    )
                )

        self.enc_final_norm = RMSNorm(self.d_enc, name="enc_final_norm")

        # Precompute RoPE tables for encoder
        head_dim = self.d_enc // self.nhead_enc
        rope_dim = (head_dim // 2) - ((head_dim // 2) % 2)
        cos_enc, sin_enc = get_rotary_frequencies(self.max_enc_len + 1, rope_dim)
        self.cos_enc = self.add_weight(shape=ops.shape(cos_enc), initializer="zeros", trainable=False, name="cos_enc")
        self.sin_enc = self.add_weight(shape=ops.shape(sin_enc), initializer="zeros", trainable=False, name="sin_enc")
        self.cos_enc.assign(cos_enc)
        self.sin_enc.assign(sin_enc)

        # 3. Multi-Task Heads
        self.ctc_norm = RMSNorm(self.d_enc, name="ctc_norm")
        self.ctc_head = layers.Dense(self.vocab_size, use_bias=True, name="ctc_head")
        self.inter_ctc_head = layers.Dense(self.vocab_size, use_bias=True, name="inter_ctc_head")
        self.aux_gloss_head = layers.Dense(self.vocab_size, use_bias=False, name="aux_gloss_head")

        # Sequence Length Predictors (Fertility)
        self.len_head = layers.Dense(1, use_bias=True, name="len_head")
        self.chicago_len_head = layers.Dense(1, use_bias=True, name="chicago_len_head")
        self.english_len_head = layers.Dense(1, use_bias=True, name="english_len_head")

        # 4. Multi-Task Decoders
        self.gloss_decoder = ASLTransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.d_dec,
            nhead=self.nhead_dec,
            kv_heads=self.kv_heads_dec,
            num_layers=self.num_dec_layers,
            dim_feedforward=self.ffn_dec,
            max_seq_len=max(self.max_enc_len, self.text_max_len, self.english_max_len, self.chicago_max_len),
            name="gloss_decoder",
        )

        if self.enable_aux_decoders:
            self.chicago_decoder = ASLTransformerDecoder(
                vocab_size=self.chicago_vocab_size,
                d_model=self.d_dec,
                nhead=self.nhead_dec,
                kv_heads=self.kv_heads_dec,
                num_layers=self.num_dec_layers,
                dim_feedforward=self.ffn_dec,
                max_seq_len=self.chicago_max_len,
                name="chicago_decoder",
            )
            self.english_decoder = ASLTransformerDecoder(
                vocab_size=self.english_vocab_size,
                d_model=self.d_dec,
                nhead=self.nhead_dec,
                kv_heads=self.kv_heads_dec,
                num_layers=self.num_dec_layers,
                dim_feedforward=self.ffn_dec,
                max_seq_len=self.english_max_len,
                name="english_decoder",
            )
        else:
            self.chicago_decoder = None
            self.english_decoder = None

        super().build(input_shape)

    def _pool_length_mean(self, enc_feats, mask=None):
        """Mask-aware global average pooling over sequence length."""
        if mask is not None:
            mask_f = ops.cast(ops.expand_dims(mask, axis=-1), ops.dtype(enc_feats))
            enc_sum = ops.sum(enc_feats * mask_f, axis=1)
            valid_counts = ops.maximum(1.0, ops.sum(mask_f, axis=1))
            return enc_sum / valid_counts
        return ops.mean(enc_feats, axis=1)

    def call(
        self,
        features,
        phonology: Optional[any] = None,
        roi_visual: Optional[any] = None,
        gloss_seq: Optional[any] = None,
        chicago_seq: Optional[any] = None,
        english_seq: Optional[any] = None,
        mask: Optional[any] = None,
    ) -> Dict[str, any]:
        b = ops.shape(features)[0]
        t = ops.shape(features)[1]

        # 1. Kinematics flattening
        if ops.ndim(features) == 4:
            x_flat = ops.reshape(features, (b, t, -1))
        else:
            x_flat = features

        f_dim = ops.shape(x_flat)[-1]
        v_tokens = self.landmark_stem(x_flat, mask=mask)

        # 2. Phonology feature fusion
        if phonology is not None:
            phonology_normed = self.phonology_norm(phonology)
        else:
            phonology_normed = ops.zeros((b, t, 19), dtype=ops.dtype(x_flat))

        # Pad to 768 aligned stem input: x_flat + phonology_normed + padding + v_tokens (128)
        combined_core = ops.shape(x_flat)[-1] + ops.shape(phonology_normed)[-1]
        pad_needed = max(0, 640 - combined_core)
        if pad_needed > 0:
            pad_tensor = ops.zeros((b, t, pad_needed), dtype=ops.dtype(x_flat))
            x_stem_in = ops.concatenate([x_flat, phonology_normed, pad_tensor, v_tokens], axis=-1)
        else:
            x_stem_in = ops.concatenate([x_flat, phonology_normed, v_tokens], axis=-1)

        # Ensure exact 768 or pad if legacy 225
        cur_dim = ops.shape(x_stem_in)[-1]
        if cur_dim < 768:
            x_stem_in = ops.pad(x_stem_in, [[0, 0], [0, 0], [0, 768 - cur_dim]])
        elif cur_dim > 768:
            x_stem_in = x_stem_in[..., :768]

        hidden_h = ops.gelu(self.input_norm(self.input_proj(x_stem_in)))

        # 3. Dual-Stream Visual ROI Gated Cross-Modal Fusion
        if roi_visual is not None:
            v_roi = self.visual_stem(roi_visual)
            hidden_h = self.cross_modal_fusion(hidden_h, v_roi)

        # 4. Hybrid Conformer + BiMamba-2 Encoder Forward Pass
        seq_len = ops.shape(hidden_h)[1]
        cos = self.cos_enc[:seq_len]
        sin = self.sin_enc[:seq_len]

        inter_h = None
        cur_mask = mask
        for i, block in enumerate(self.blocks):
            if isinstance(block, TemporalStridedPool):
                if cur_mask is not None:
                    hidden_h, cur_mask = block(hidden_h, mask=cur_mask)
                else:
                    hidden_h = block(hidden_h)
                inter_h = hidden_h
                # Update RoPE slice after strided pooling
                s_pooled = ops.shape(hidden_h)[1]
                cos = self.cos_enc[:s_pooled]
                sin = self.sin_enc[:s_pooled]
            elif isinstance(block, BiMamba2SSMBlock):
                hidden_h = block(hidden_h, mask=cur_mask)
            else:
                hidden_h = block(hidden_h, mask=cur_mask, cos=cos, sin=sin)

        enc_out = self.enc_final_norm(hidden_h)

        # 5. Multi-Task Heads
        ctc_logits = self.ctc_head(self.ctc_norm(enc_out))
        inter_ctc_logits = self.inter_ctc_head(inter_h) if inter_h is not None else None

        # Conceptual clustering aux head
        enc_pooled = self._pool_length_mean(enc_out, mask=cur_mask)
        aux_gloss_logits = self.aux_gloss_head(enc_pooled)
        pred_len = ops.squeeze(self.len_head(enc_pooled), axis=-1)

        # 6. Multi-Task Decoders
        dec_logits = None
        if gloss_seq is not None:
            dec_logits = self.gloss_decoder(gloss_seq, memory=enc_out, memory_mask=cur_mask)

        chicago_logits = None
        chicago_len = None
        if chicago_seq is not None and self.chicago_decoder is not None:
            chicago_logits = self.chicago_decoder(chicago_seq, memory=enc_out, memory_mask=cur_mask)
            chicago_len = ops.squeeze(self.chicago_len_head(enc_pooled), axis=-1)

        english_logits = None
        english_len = None
        if english_seq is not None and self.english_decoder is not None:
            english_logits = self.english_decoder(english_seq, memory=enc_out, memory_mask=cur_mask)
            english_len = ops.squeeze(self.english_len_head(enc_pooled), axis=-1)

        out = {
            "enc_out": enc_out,
            "ctc_logits": ctc_logits,
            "inter_ctc_logits": inter_ctc_logits,
            "aux_gloss_logits": aux_gloss_logits,
            "pred_len": pred_len,
            "dec_logits": dec_logits,
            "chicago_logits": chicago_logits,
            "chicago_len": chicago_len,
            "english_logits": english_logits,
            "english_len": english_len,
        }
        # Filter None keys to guarantee safe Keras 3 mask metadata handling
        return {k: v for k, v in out.items() if v is not None}
