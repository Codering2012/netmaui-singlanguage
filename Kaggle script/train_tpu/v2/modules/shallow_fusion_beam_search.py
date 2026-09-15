#!/usr/bin/env python3
"""
================================================================================
  SOTA SHALLOW FUSION CTC BEAM SEARCH DECODER WITH TRIE-BASED LM RESCORING
================================================================================
Implements prefix-tree constrained CTC beam search with n-gram / neural LM shallow fusion:
  Score(W) = log P_CTC(W | X) + alpha * log P_LM(W) + beta * |W|
================================================================================
"""

import sys
import math
import heapq
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np


class TrieNode:
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word: bool = False
        self.score: float = 0.0


class FastVocabularyTrie:
    """
    Prefix tree for dictionary-constrained decoding and language model lookup.
    """

    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, score: float = 0.0):
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_word = True
        node.score = score

    def search_prefix(self, prefix: str) -> Optional[TrieNode]:
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return node


class ShallowFusionBeamSearchDecoder:
    """
    Continuous Sign Language Beam Search Decoder with CTC + LM Shallow Fusion.
    """

    def __init__(
        self,
        vocab_list: List[str],
        blank_idx: int = 0,
        beam_width: int = 16,
        alpha_lm: float = 0.35,
        beta_len: float = 1.20,
        lm_unigram_probs: Optional[Dict[str, float]] = None,
    ):
        self.vocab_list = vocab_list
        self.blank_idx = blank_idx
        self.beam_width = beam_width
        self.alpha_lm = alpha_lm
        self.beta_len = beta_len
        self.lm_probs = lm_unigram_probs or {}
        self.trie = FastVocabularyTrie()

        for word in self.vocab_list:
            score = self.lm_probs.get(word, -10.0)
            self.trie.insert(word, score)

    def decode(
        self,
        ctc_log_probs: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> List[List[str]]:
        """
        ctc_log_probs: [B, T, V] normalized log-probabilities
        Returns: List of top-1 decoded word token strings per batch item.
        """
        B, T, V = ctc_log_probs.shape
        log_probs_np = ctc_log_probs.detach().cpu().numpy()
        results = []

        for b in range(B):
            actual_T = int(seq_lens[b].item()) if seq_lens is not None else T
            # Map: prefix -> (prob_blank, prob_non_blank)
            beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, -1e9)}

            for t in range(actual_T):
                step_log_probs = log_probs_np[b, t]
                new_beams: Dict[Tuple[int, ...], Tuple[float, float]] = defaultdict(lambda: (-1e9, -1e9))

                for prefix, (p_b, p_nb) in beams.items():
                    p_total = np.logaddexp(p_b, p_nb)

                    # 1. Blank transition
                    p_blank = step_log_probs[self.blank_idx]
                    cur_p_b, cur_p_nb = new_beams[prefix]
                    new_beams[prefix] = (np.logaddexp(cur_p_b, p_total + p_blank), cur_p_nb)

                    # 2. Non-blank extensions
                    top_k_indices = np.argsort(step_log_probs)[-self.beam_width:]
                    for c in top_k_indices:
                        if c == self.blank_idx:
                            continue

                        p_token = step_log_probs[c]
                        last_c = prefix[-1] if len(prefix) > 0 else -1

                        if c == last_c:
                            # Repeated token without intervening blank
                            cur_b, cur_nb = new_beams[prefix]
                            new_beams[prefix] = (cur_b, np.logaddexp(cur_nb, p_nb + p_token))

                            new_p = prefix + (c,)
                            cur_b, cur_nb = new_beams[new_p]
                            new_beams[new_p] = (cur_b, np.logaddexp(cur_nb, p_b + p_token))
                        else:
                            new_p = prefix + (c,)
                            cur_b, cur_nb = new_beams[new_p]
                            new_beams[new_p] = (cur_b, np.logaddexp(cur_nb, p_total + p_token))

                # Beam Pruning with LM and Length Bonus
                scored_beams = []
                for p, (b_val, nb_val) in new_beams.items():
                    ctc_score = np.logaddexp(b_val, nb_val)
                    # LM shallow fusion bonus
                    lm_score = 0.0
                    if len(p) > 0 and self.alpha_lm > 0:
                        for token_idx in p:
                            if token_idx < len(self.vocab_list):
                                w = self.vocab_list[token_idx]
                                lm_score += self.lm_probs.get(w, -5.0)

                    length_bonus = self.beta_len * len(p)
                    total_score = ctc_score + self.alpha_lm * lm_score + length_bonus
                    scored_beams.append((total_score, p, (b_val, nb_val)))

                # Keep top beam_width
                scored_beams.sort(key=lambda x: x[0], reverse=True)
                beams = {p: probs for _, p, probs in scored_beams[: self.beam_width]}

            # Get best hypothesis
            best_prefix = max(
                beams.keys(),
                key=lambda p: np.logaddexp(beams[p][0], beams[p][1]) + self.beta_len * len(p),
            )
            decoded_tokens = [self.vocab_list[idx] for idx in best_prefix if idx < len(self.vocab_list)]
            results.append(decoded_tokens)

        return results


def main():
    print("[INFO] Shallow Fusion Beam Search Decoder initialized.")
    vocab = ["<blank>", "hello", "world", "thank", "you", "sign", "language"]
    decoder = ShallowFusionBeamSearchDecoder(vocab_list=vocab, beam_width=8, alpha_lm=0.3)

    mock_logits = torch.randn(2, 20, len(vocab)).log_softmax(dim=-1)
    results = decoder.decode(mock_logits)
    print(f"[+] Decoded Sample Sentences: {results}")


if __name__ == "__main__":
    main()
