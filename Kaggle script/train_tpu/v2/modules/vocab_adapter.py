#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — DYNAMIC VOCABULARY ADAPTATION & OOV SUBWORD ENGINE
================================================================================
Enables runtime vocabulary expansion and robust Out-Of-Vocabulary (OOV) handling:
1. Dynamic Token Expansion: Expands model embedding tables and classification heads
   (V -> V') while strictly preserving existing pre-trained parameters.
2. Subword & Character Decomposition: Automatically maps OOV words into subwords
   or fingerspelled character sequences.
3. Statistical Initialization: Initializes new token embeddings using empirical
   lexical covariance to prevent catastrophic gradient shocks.
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASLDynamicVocabAdapter:
    """
    Manages dynamic vocabulary expansion and OOV subword fallbacks for ASL Foundation Models.
    """

    def __init__(
        self,
        vocab_list: List[str],
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ):
        self.vocab_list = list(vocab_list)
        self.word2id = {w: i for i, w in enumerate(self.vocab_list)}
        self.unk_token = unk_token
        self.unk_id = self.word2id.get(unk_token, 3)
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token

    def tokenize_with_subword_fallback(self, text: str) -> List[int]:
        """
        Tokenizes text words. If a word is OOV, decomposes it into character
        or subword tokens (e.g. 'ALICE' -> ['A', 'L', 'I', 'C', 'E']).
        """
        tokens = []
        words = text.strip().split()

        for w in words:
            if w in self.word2id:
                tokens.append(self.word2id[w])
            elif w.upper() in self.word2id:
                tokens.append(self.word2id[w.upper()])
            else:
                # Subword / character-level fingerspelling fallback
                chars_matched = 0
                for ch in w.upper():
                    ch_token = f"CHAR_{ch}"
                    if ch_token in self.word2id:
                        tokens.append(self.word2id[ch_token])
                        chars_matched += 1
                    elif ch in self.word2id:
                        tokens.append(self.word2id[ch])
                        chars_matched += 1

                if chars_matched == 0:
                    tokens.append(self.unk_id)

        return tokens

    def add_new_tokens(self, new_tokens: List[str]) -> List[str]:
        """
        Adds new tokens to the vocabulary dictionary.
        Returns list of actually added tokens.
        """
        added = []
        for tok in new_tokens:
            if tok not in self.word2id:
                self.word2id[tok] = len(self.vocab_list)
                self.vocab_list.append(tok)
                added.append(tok)
        return added

    def expand_model_vocabulary(
        self,
        model: nn.Module,
        new_vocab_size: int,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Dynamically resizes model's embedding layers and output heads to new_vocab_size.
        """
        device = torch.device(device)

        # 1. Expand CTC Head if present
        if hasattr(model, "ctc_head"):
            ctc_mod = model.ctc_head
            target_linear = ctc_mod.proj if hasattr(ctc_mod, "proj") else (ctc_mod if isinstance(ctc_mod, nn.Linear) else None)
            if target_linear is not None:
                old_out, in_features = target_linear.weight.shape
                if hasattr(ctc_mod, "actual_vocab_size"):
                    ctc_mod.actual_vocab_size = new_vocab_size

                if new_vocab_size > old_out:
                    # Allocate with 128 TPU alignment if needed
                    alloc_size = ((new_vocab_size + 127) // 128) * 128
                    new_linear = nn.Linear(in_features, alloc_size, bias=target_linear.bias is not None).to(device)
                    with torch.no_grad():
                        new_linear.weight[:old_out] = target_linear.weight
                        if target_linear.bias is not None:
                            new_linear.bias[:old_out] = target_linear.bias
                        std = target_linear.weight.std().item()
                        mean = target_linear.weight.mean().item()
                        new_linear.weight[old_out:].normal_(mean, std * 0.5)
                        if new_linear.bias is not None:
                            new_linear.bias[old_out:].zero_()

                    if hasattr(ctc_mod, "proj"):
                        ctc_mod.proj = new_linear
                    else:
                        model.ctc_head = new_linear

        # 2. Expand Decoder Embedding & Projection if present
        if hasattr(model, "decoder") and hasattr(model.decoder, "token_emb"):
            old_emb = model.decoder.token_emb
            if isinstance(old_emb, nn.Embedding) and new_vocab_size > old_emb.num_embeddings:
                alloc_size = ((new_vocab_size + 127) // 128) * 128
                new_emb = nn.Embedding(alloc_size, old_emb.embedding_dim, padding_idx=old_emb.padding_idx).to(device)
                with torch.no_grad():
                    new_emb.weight[:old_emb.num_embeddings] = old_emb.weight
                    std = old_emb.weight.std().item()
                    mean = old_emb.weight.mean().item()
                    new_emb.weight[old_emb.num_embeddings:].normal_(mean, std * 0.5)
                model.decoder.token_emb = new_emb

        if hasattr(model, "decoder") and hasattr(model.decoder, "out_proj"):
            old_proj = model.decoder.out_proj
            if isinstance(old_proj, nn.Linear) and new_vocab_size > old_proj.out_features:
                alloc_size = ((new_vocab_size + 127) // 128) * 128
                new_proj = nn.Linear(old_proj.in_features, alloc_size, bias=old_proj.bias is not None).to(device)
                with torch.no_grad():
                    new_proj.weight[:old_proj.out_features] = old_proj.weight
                    if old_proj.bias is not None:
                        new_proj.bias[:old_proj.out_features] = old_proj.bias
                    std = old_proj.weight.std().item()
                    mean = old_proj.weight.mean().item()
                    new_proj.weight[old_proj.out_features:].normal_(mean, std * 0.5)
                    if new_proj.bias is not None:
                        new_proj.bias[old_proj.out_features:].zero_()
                model.decoder.out_proj = new_proj
