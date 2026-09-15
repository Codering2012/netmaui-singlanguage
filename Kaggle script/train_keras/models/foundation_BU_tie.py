"""
Unified Multi-Task Continuous ASL Foundation Model in Keras 3 (JAX Backend)
"""

from typing import Dict, Optional, Any
import keras
from keras import layers, ops
from .conformer import MobileConformerEncoder
from .decoder import ASLTransformerDecoder
from ..layers import RMSNorm


class ASLFoundationModel(keras.Model):
    """Complete Multi-Task Continuous ASL Foundation Model."""
    def __init__(
        self,
        vocab_size: int = 17800,
        chicago_vocab_size: int = 128,
        english_vocab_size: int = 50257,
        d_model: int = 512,
        nhead: int = 4,
        kv_heads: int = 2,
        num_enc_layers: int = 4,
        num_dec_layers: int = 4,
        dim_feedforward: int = 1280,
        in_channels: int = 225,
        max_len: int = 384,
        text_max_len: int = 128,
        english_max_len: int = 128,
        chicago_max_len: int = 128,
        kernel_size: int = 31,
        is_causal: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.chicago_vocab_size = chicago_vocab_size
        self.english_vocab_size = english_vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.kv_heads = kv_heads
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dim_feedforward = dim_feedforward
        self.in_channels = in_channels
        self.max_len = max_len
        self.text_max_len = text_max_len
        self.english_max_len = english_max_len
        self.chicago_max_len = chicago_max_len
        self.kernel_size = kernel_size
        self.is_causal = is_causal
        self.supports_masking = True


    def build(self, input_shape):
        # 1. Visual Landmark Encoder
        self.encoder = MobileConformerEncoder(
            num_layers=self.num_enc_layers,
            d_model=self.d_model,
            nhead=self.nhead,
            kv_heads=self.kv_heads,
            dim_feedforward=self.dim_feedforward,
            in_channels=self.in_channels,
            max_len=self.max_len,
            kernel_size=self.kernel_size,
            is_causal=self.is_causal,
            name="encoder",
        )

        # 2. Frame-level CTC Head
        self.ctc_norm = RMSNorm(self.d_model, name="ctc_norm")
        self.ctc_head = layers.Dense(self.vocab_size, use_bias=True, name="ctc_head")

        # 3. Length Predictor Head
        self.len_head = layers.Dense(1, use_bias=True, name="len_head")

        # 4. Multi-Task ASL Gloss Decoder
        self.gloss_decoder = ASLTransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            kv_heads=self.kv_heads,
            num_layers=self.num_dec_layers,
            dim_feedforward=self.dim_feedforward,
            max_seq_len=max(self.max_len, self.text_max_len, self.english_max_len, self.chicago_max_len),
            name="gloss_decoder",
        )

        # 5. Chicago Fingerspelling Embedding & Projection Head
        self.chicago_embed = layers.Embedding(self.chicago_vocab_size, self.d_model, name="chicago_embed")
        self.chicago_head = layers.Dense(self.chicago_vocab_size, use_bias=False, name="chicago_head")

        # 6. English BPE Translation Embedding & Projection Head
        self.english_embed = layers.Embedding(self.english_vocab_size, self.d_model, name="english_embed")
        self.english_head = layers.Dense(self.english_vocab_size, use_bias=False, name="english_head")
        super().build(input_shape)

    def _decode_tokens(self, embed_layer, tokens, memory, memory_mask=None):
        seq_len = ops.shape(tokens)[1]
        x = embed_layer(tokens) * self.gloss_decoder.embed_scale
        cos = self.gloss_decoder.cos[:seq_len]
        sin = self.gloss_decoder.sin[:seq_len]
        for layer in self.gloss_decoder.layers_list:
            x = layer(
                x,
                memory=memory,
                memory_mask=memory_mask,
                cos=cos,
                sin=sin,
            )
        return self.gloss_decoder.final_norm(x)

    def call(
        self,
        features,
        gloss_seq: Optional[any] = None,
        chicago_seq: Optional[any] = None,
        english_seq: Optional[any] = None,
        mask: Optional[any] = None,
    ) -> Dict[str, any]:
        # Flatten 4D landmark coordinates (B, L, 75, 3) to 3D (B, L, 225) if needed
        if ops.ndim(features) == 4:
            features = ops.reshape(features, (ops.shape(features)[0], ops.shape(features)[1], -1))

        # 1. Encode visual trajectory
        enc_out = self.encoder(features, mask=mask)

        # 2. CTC Logits
        ctc_feats = self.ctc_norm(enc_out)
        ctc_logits = self.ctc_head(ctc_feats)

        # 3. Sequence Length Prediction (Mask-Aware Global Pooling)
        if mask is not None:
            mask_f = ops.cast(ops.expand_dims(mask, axis=-1), ops.dtype(enc_out))
            enc_sum = ops.sum(enc_out * mask_f, axis=1)
            valid_counts = ops.maximum(1.0, ops.sum(mask_f, axis=1))
            enc_mean = enc_sum / valid_counts
        else:
            enc_mean = ops.mean(enc_out, axis=1)
        pred_len = ops.squeeze(self.len_head(enc_mean), axis=-1)

        # 4. Gloss Decoder Logits
        dec_logits = None
        if gloss_seq is not None:
            dec_logits = self.gloss_decoder(gloss_seq, memory=enc_out, memory_mask=mask)

        # 5. Chicago Fingerspelling Logits
        chicago_logits = None
        if chicago_seq is not None:
            h_chi = self._decode_tokens(self.chicago_embed, chicago_seq, enc_out, mask)
            chicago_logits = self.chicago_head(h_chi)

        # 6. English Translation Logits
        english_logits = None
        if english_seq is not None:
            h_eng = self._decode_tokens(self.english_embed, english_seq, enc_out, mask)
            english_logits = self.english_head(h_eng)

        out = {
            "enc_out": enc_out,
            "ctc_logits": ctc_logits,
            "pred_len": pred_len,
            "dec_logits": dec_logits,
            "chicago_logits": chicago_logits,
            "english_logits": english_logits,
        }
        return {k: v for k, v in out.items() if v is not None}
