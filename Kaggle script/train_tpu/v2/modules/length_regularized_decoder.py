#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — DYNAMIC LENGTH-REGULARIZED SEQUENCE DECODER
================================================================================
Implements Length-Normalized Autoregressive Beam Search (Wu et al., Vaswani et al.):
1. Dynamic Length Penalty Normalization:
     Score(Y | X) = sum_t log P(y_t | X, y_<t) / LP(L)
     LP(L) = ((5 + L) / 6)^alpha
2. Visual-Duration-Aware EOS Gating: Prevents premature termination on long
   video sequences and prevents runaway generation on short sign gestures.
3. N-gram Repetition Blocking: Eliminates degenerate autoregressive loops.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BeamHypothesis(NamedTuple):
    tokens: List[int]
    log_prob: float
    length_normalized_score: float
    is_finished: bool


class ASLLengthRegularizedDecoder:
    """
    Length-regularized autoregressive beam search decoder for ASL Foundation Models.
    """

    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        beam_width: int = 4,
        alpha_length: float = 0.80,
        no_repeat_ngram_size: int = 3,
        min_length_ratio: float = 0.10,  # Min tokens per 10 video frames
        max_length_ratio: float = 0.80,  # Max tokens per 10 video frames
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        device: Union[str, torch.device] = "cpu",
    ):
        self.model = model
        self.vocab_size = vocab_size
        self.beam_width = beam_width
        self.alpha_length = alpha_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.min_length_ratio = min_length_ratio
        self.max_length_ratio = max_length_ratio
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.device = torch.device(device)
        self.model.eval()

    def calculate_length_penalty(self, length: int) -> float:
        """
        Wu et al. Length Penalty: LP(L) = ((5 + L) / 6)^alpha
        """
        return math.pow((5.0 + length) / 6.0, self.alpha_length)

    def is_ngram_repeated(self, tokens: List[int], next_token: int, n: int) -> bool:
        """
        Checks if appending next_token creates a duplicate n-gram.
        """
        if len(tokens) < n - 1 or n <= 1:
            return False
        cand_ngram = tuple(tokens[-(n - 1):] + [next_token])
        # Check against existing n-grams in history
        for i in range(len(tokens) - n + 1):
            if tuple(tokens[i : i + n]) == cand_ngram:
                return True
        return False

    @torch.no_grad()
    def decode(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        frame_indices: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
    ) -> List[BeamHypothesis]:
        """
        Executes dynamic length-regularized beam search decoding.
        features: [B, T, K, C] (B=1)
        """
        B, T, K, C = features.shape
        assert B == 1, "Length regularized decoder operates on single video stream."

        features = features.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        if frame_indices is not None:
            frame_indices = frame_indices.to(self.device)

        min_allowed_len = max(2, int(T * self.min_length_ratio))
        max_allowed_len = min(max_new_tokens, max(min_allowed_len + 4, int(T * self.max_length_ratio)))

        # Initial hypothesis containing [BOS]
        active_hypotheses: List[BeamHypothesis] = [
            BeamHypothesis(
                tokens=[self.bos_id],
                log_prob=0.0,
                length_normalized_score=0.0,
                is_finished=False,
            )
        ]
        completed_hypotheses: List[BeamHypothesis] = []

        for step in range(max_allowed_len):
            if not active_hypotheses:
                break

            candidates: List[BeamHypothesis] = []

            for hyp in active_hypotheses:
                if hyp.is_finished:
                    completed_hypotheses.append(hyp)
                    continue

                curr_seq = hyp.tokens
                cand_tensor = torch.tensor([curr_seq + [0]], dtype=torch.long, device=self.device)

                # Query decoder logits for next token
                out = self.model(
                    input_x=features,
                    mask=mask,
                    frame_indices=frame_indices,
                    gloss_seq=cand_tensor,
                )
                logits = out["dec_logits"][0, len(curr_seq) - 1].clone()  # [V]

                # 1. Suppress EOS if under min_allowed_len
                if len(curr_seq) < min_allowed_len:
                    logits[self.eos_id] = -1e9

                # 2. Block repeated n-grams
                if self.no_repeat_ngram_size > 0:
                    for v in range(self.vocab_size):
                        if self.is_ngram_repeated(curr_seq, v, self.no_repeat_ngram_size):
                            logits[v] = -1e9

                log_probs = F.log_softmax(logits, dim=-1)
                topk_log_probs, topk_tokens = torch.topk(log_probs, k=self.beam_width)

                for k in range(self.beam_width):
                    tok = topk_tokens[k].item()
                    tok_lp = topk_log_probs[k].item()

                    new_tokens = curr_seq + [tok]
                    new_log_prob = hyp.log_prob + tok_lp
                    is_eos = (tok == self.eos_id)

                    effective_len = len(new_tokens) - 1  # Excluding BOS
                    lp_factor = self.calculate_length_penalty(max(1, effective_len))
                    norm_score = new_log_prob / lp_factor

                    candidates.append(
                        BeamHypothesis(
                            tokens=new_tokens,
                            log_prob=new_log_prob,
                            length_normalized_score=norm_score,
                            is_finished=is_eos,
                        )
                    )

            # Prune to top-K hypotheses by length-normalized score
            candidates.sort(key=lambda h: h.length_normalized_score, reverse=True)
            active_hypotheses = candidates[: self.beam_width]

            # Separate newly finished hypotheses
            new_active = []
            for h in active_hypotheses:
                if h.is_finished:
                    completed_hypotheses.append(h)
                else:
                    new_active.append(h)
            active_hypotheses = new_active

        # Combine all candidates and rank by length-normalized score
        all_results = completed_hypotheses + active_hypotheses
        all_results.sort(key=lambda h: h.length_normalized_score, reverse=True)

        return all_results[: self.beam_width]
