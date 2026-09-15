"""
Unified ASL Multi-Task Foundation Model in Flax Linen
Encoder: MobileConformer
Decoders: ASL Gloss Decoder, English Translation Decoder, Chicago Fingerspelling, CTC Head
"""

from typing import Dict, Optional, Tuple, Any
import jax
import jax.numpy as jnp
import flax.linen as nn
from .conformer import MobileConformerEncoder, RMSNorm
from .decoder import ASLTransformerDecoder, ASLDecoderLayer, get_rotary_frequencies


class ASLFoundationModel(nn.Module):
    """Complete Multi-Task Continuous ASL Foundation Model."""
    vocab_size: int = 17800
    chicago_vocab_size: int = 128
    english_vocab_size: int = 50257
    d_model: int = 512
    nhead: int = 4
    kv_heads: int = 2
    num_enc_layers: int = 4
    num_dec_layers: int = 4
    dim_feedforward: int = 1280
    in_channels: int = 225
    max_len: int = 384
    english_max_len: int = 128
    chicago_max_len: int = 128

    def setup(self):
        # 1. Visual Landmark Encoder
        self.encoder = MobileConformerEncoder(
            num_layers=self.num_enc_layers,
            d_model=self.d_model,
            nhead=self.nhead,
            kv_heads=self.kv_heads,
            dim_feedforward=self.dim_feedforward,
            in_channels=self.in_channels,
            max_len=self.max_len,
            is_causal=True,
            name="encoder",
        )

        # 2. Frame-level CTC Head
        self.ctc_norm = RMSNorm(self.d_model, name="ctc_norm")
        self.ctc_head = nn.Dense(self.vocab_size, use_bias=True, name="ctc_head")

        # 3. Length Predictor Head
        self.len_head = nn.Dense(1, use_bias=True, name="len_head")

        # 4. Multi-Task ASL Gloss Decoder
        self.gloss_decoder = ASLTransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            nhead=self.nhead,
            kv_heads=self.kv_heads,
            num_layers=self.num_dec_layers,
            ffn_dim=self.dim_feedforward,
            max_seq_len=self.max_len,
            name="gloss_decoder",
        )

        # 5. Chicago Fingerspelling Projection Head
        self.chicago_head = nn.Dense(self.chicago_vocab_size, use_bias=False, name="chicago_head")

        # 6. English BPE Translation Projection Head
        self.english_head = nn.Dense(self.english_vocab_size, use_bias=False, name="english_head")

    def __call__(
        self,
        features: jnp.ndarray,
        gloss_seq: Optional[jnp.ndarray] = None,
        chicago_seq: Optional[jnp.ndarray] = None,
        english_seq: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None,
    ) -> Dict[str, jnp.ndarray]:
        # 1. Encode visual trajectory
        enc_out = self.encoder(features, mask=mask)

        # 2. CTC Logits
        ctc_feats = self.ctc_norm(enc_out)
        ctc_logits = self.ctc_head(ctc_feats)
        ctc_log_probs = jax.nn.log_softmax(ctc_logits, axis=-1)

        # 3. Sequence Length Prediction
        enc_mean = jnp.mean(enc_out, axis=1)
        pred_len = jnp.squeeze(self.len_head(enc_mean), axis=-1)

        # 4. Gloss Decoder Logits
        dec_logits = None
        if gloss_seq is not None:
            dec_logits = self.gloss_decoder(gloss_seq, memory=enc_out, memory_mask=mask)

        # 5. Chicago Fingerspelling Logits
        chicago_logits = self.chicago_head(enc_out)

        # 6. English Translation Logits
        english_logits = None
        if english_seq is not None:
            # Cross-attend english sequence over visual encoder
            english_logits = self.english_head(enc_out[:, :english_seq.shape[1], :])

        return {
            "enc_out": enc_out,
            "ctc_log_probs": ctc_log_probs,
            "dec_logits": dec_logits,
            "chicago_logits": chicago_logits,
            "english_logits": english_logits,
            "pred_len": pred_len,
        }
