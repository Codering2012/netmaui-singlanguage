#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — NON-AUTOREGRESSIVE MASK-CTC ITERATIVE REFINEMENT DECODER
================================================================================
Implements Mask-CTC / CASS-NAT Parallel Iterative Translation:
1. Fast O(1) CTC Prototype Initialization:
     Y^(0) = CTC_Collapse( argmax P_CTC(y_t | X) )
2. Confidence-Guided Selective Masking:
     Masks tokens where P(y_i) < tau_conf with [MASK] tokens.
3. Parallel Bidirectional Transformer Refinement:
     Y^(k+1) = NAR_Decoder( Y^(k)_masked, H_enc ) for K iterations.
4. Ultra-Low Latency (4x-6x speedup over sequential autoregressive beam search).
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NARTranslationOutput(NamedTuple):
    tokens: List[int]
    confidence_scores: List[float]
    iterations_run: int
    raw_ctc_tokens: List[int]


class ASLMaskCTCIterativeDecoder(nn.Module):
    """
    Non-autoregressive iterative refinement decoder guided by CTC alignments.
    """

    def __init__(
        self,
        d_model: int = 128,
        vocab_size: int = 80,
        num_refine_layers: int = 2,
        nhead: int = 4,
        mask_token_id: int = 3,  # Usually UNK/MASK
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        confidence_threshold: float = 0.65,
        max_refine_iterations: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.confidence_threshold = confidence_threshold
        self.max_refine_iterations = max_refine_iterations

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_emb = nn.Embedding(128, d_model)

        # Bidirectional Non-Autoregressive Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.nar_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_refine_layers)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def ctc_greedy_initialize(
        self,
        ctc_log_probs: torch.Tensor,  # [T, V] or [1, T, V]
    ) -> Tuple[List[int], List[float]]:
        """
        Collapses CTC frame predictions into initial discrete token sequence with confidences.
        """
        if ctc_log_probs.dim() == 3:
            ctc_log_probs = ctc_log_probs[0]

        probs = torch.softmax(ctc_log_probs, dim=-1)
        best_scores, best_ids = torch.max(probs, dim=-1)

        raw_ids = best_ids.tolist()
        raw_scores = best_scores.tolist()

        collapsed_tokens = []
        collapsed_confidences = []

        prev_id = -1
        for token_id, conf in zip(raw_ids, raw_scores):
            if token_id != 0 and token_id != prev_id:  # 0 is blank/pad
                collapsed_tokens.append(token_id)
                collapsed_confidences.append(conf)
            prev_id = token_id

        return collapsed_tokens, collapsed_confidences

    @torch.no_grad()
    def decode(
        self,
        ctc_log_probs: torch.Tensor,
        encoder_hidden: torch.Tensor,  # [1, T_enc, d_model]
        num_iterations: Optional[int] = None,
    ) -> NARTranslationOutput:
        """
        Executes Mask-CTC non-autoregressive iterative parallel decoding.
        """
        device = encoder_hidden.device
        iters = num_iterations if num_iterations is not None else self.max_refine_iterations

        # 1. CTC Prototype Initialization
        ctc_tokens, ctc_confs = self.ctc_greedy_initialize(ctc_log_probs)

        if len(ctc_tokens) == 0:
            return NARTranslationOutput(
                tokens=[],
                confidence_scores=[],
                iterations_run=0,
                raw_ctc_tokens=[],
            )

        current_tokens = list(ctc_tokens)
        current_confs = list(ctc_confs)
        L = len(current_tokens)

        # 2. Iterative Refinement Loop
        for it in range(iters):
            # Identify low-confidence tokens to mask
            masked_token_ids = []
            mask_positions = []

            for idx, (tok, conf) in enumerate(zip(current_tokens, current_confs)):
                if conf < self.confidence_threshold:
                    masked_token_ids.append(self.mask_token_id)
                    mask_positions.append(idx)
                else:
                    masked_token_ids.append(tok)

            if len(mask_positions) == 0:
                # Early stop if all tokens are confident (zero refinement needed)
                return NARTranslationOutput(
                    tokens=current_tokens,
                    confidence_scores=current_confs,
                    iterations_run=it,
                    raw_ctc_tokens=ctc_tokens,
                )

            # Forward pass through parallel bidirectional decoder
            tgt_tensor = torch.tensor([masked_token_ids], device=device, dtype=torch.long)
            pos_ids = torch.arange(L, device=device).unsqueeze(0)

            x = self.token_emb(tgt_tensor) + self.pos_emb(pos_ids)
            h_dec = self.nar_decoder(tgt=x, memory=encoder_hidden)
            logits = self.lm_head(h_dec)  # [1, L, V]

            refine_probs = F.softmax(logits[0], dim=-1)
            refine_scores, refine_ids = torch.max(refine_probs, dim=-1)

            # Update masked positions with refined predictions
            for idx in mask_positions:
                new_tok = refine_ids[idx].item()
                new_conf = refine_scores[idx].item()
                if new_tok != self.pad_token_id:
                    current_tokens[idx] = new_tok
                    current_confs[idx] = new_conf

        return NARTranslationOutput(
            tokens=current_tokens,
            confidence_scores=current_confs,
            iterations_run=it + 1,
            raw_ctc_tokens=ctc_tokens,
        )
