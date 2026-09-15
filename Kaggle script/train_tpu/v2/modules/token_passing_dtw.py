#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — CONTINUOUS TOKEN PASSING DTW DECODER ENGINE
================================================================================
Implements dictionary-constrained Token Passing Dynamic Programming over
continuous Conformer encoder feature trajectories:
1. Dynamic alignment against canonical gesture exemplars E_w in feature space
2. Global optimal sequence parsing: W* = argmin sum DTW(H_k, E_w) + penalty
3. Recovers both recognized gloss sequence and exact frame alignments
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, NamedTuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenParseResult(NamedTuple):
    tokens: List[int]
    alignments: List[Tuple[int, int]]  # (start_frame, end_frame)
    total_cost: float


class ASLTokenPassingDTWDecoder:
    """
    Token Passing Dynamic Time Warping Decoder for Continuous Sign Language.
    """

    def __init__(
        self,
        exemplar_dict: Dict[int, torch.Tensor],
        transition_penalty: float = 2.5,
        distance_metric: str = "cosine",
    ):
        """
        exemplar_dict: Mapping token_id -> canonical feature tensor [L_w, D]
        """
        self.exemplar_dict = exemplar_dict
        self.transition_penalty = transition_penalty
        self.distance_metric = distance_metric

    def _frame_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Computes pairwise distance between input frame x [D] and exemplar state y [D].
        """
        if self.distance_metric == "cosine":
            # Cosine distance: 1 - cos_sim
            sim = F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0), dim=-1)[0]
            return 1.0 - sim
        else:
            # Euclidean distance
            return torch.norm(x - y, p=2)

    def decode_sequence(
        self,
        encoder_features: torch.Tensor,
    ) -> TokenParseResult:
        """
        encoder_features: [T, D] feature trajectory from Conformer encoder
        """
        T, D = encoder_features.shape
        device = encoder_features.device

        # Token state grids: for each word w, grid has length L_w
        # We maintain: token_cost[w][state_s], token_history[w][state_s]
        # At each frame t:
        # 1. Intra-word transitions: state_s can come from state_s (stay) or state_s - 1 (advance)
        # 2. Inter-word transitions: state 0 of any word can start from the best terminating state of any word at t-1

        word_ids = list(self.exemplar_dict.keys())
        if len(word_ids) == 0:
            return TokenParseResult(tokens=[], alignments=[], total_cost=0.0)

        # Initialize costs: [W, L_max]
        costs: Dict[int, torch.Tensor] = {}
        histories: Dict[int, List[List[Tuple[int, int]]]] = {}  # [w][s] -> list of (word_id, start_t)

        for w in word_ids:
            L_w = self.exemplar_dict[w].size(0)
            costs[w] = torch.full((L_w,), float("inf"), device=device)
            costs[w][0] = 0.0  # Initial word can start at t=0
            histories[w] = [[(w, 0)] for _ in range(L_w)]

        for t in range(T):
            x_t = encoder_features[t]  # [D]

            # Find best terminating token from previous frame t-1
            best_prev_cost = float("inf")
            best_prev_history: Optional[List[Tuple[int, int]]] = None

            for w in word_ids:
                L_w = self.exemplar_dict[w].size(0)
                term_cost = costs[w][L_w - 1].item()
                if term_cost < best_prev_cost:
                    best_prev_cost = term_cost
                    best_prev_history = histories[w][L_w - 1]

            new_costs: Dict[int, torch.Tensor] = {}
            new_histories: Dict[int, List[List[Tuple[int, int]]]] = {}

            for w in word_ids:
                exemplar = self.exemplar_dict[w]  # [L_w, D]
                L_w = exemplar.size(0)
                new_costs[w] = torch.full((L_w,), float("inf"), device=device)
                new_histories[w] = [[] for _ in range(L_w)]

                # Compute distance to all states in exemplar w
                if self.distance_metric == "cosine":
                    dists = 1.0 - F.cosine_similarity(x_t.unsqueeze(0), exemplar, dim=-1)  # [L_w]
                else:
                    dists = torch.norm(x_t.unsqueeze(0) - exemplar, dim=-1)

                for s in range(L_w):
                    d_s = dists[s]

                    # Option 1: Stay at state s
                    c_stay = costs[w][s] + d_s
                    best_c = c_stay
                    best_h = histories[w][s]

                    # Option 2: Advance from state s - 1
                    if s > 0:
                        c_adv = costs[w][s - 1] + d_s
                        if c_adv < best_c:
                            best_c = c_adv
                            best_h = histories[w][s - 1]

                    # Option 3: Start state (s == 0) from best terminating word at t-1
                    if s == 0 and best_prev_history is not None:
                        c_trans = best_prev_cost + self.transition_penalty + d_s
                        if c_trans < best_c:
                            best_c = c_trans
                            best_h = best_prev_history + [(w, t)]

                    new_costs[w][s] = best_c
                    new_histories[w][s] = best_h

            costs = new_costs
            histories = new_histories

        # Extract final best sequence
        final_best_cost = float("inf")
        final_best_history: List[Tuple[int, int]] = []

        for w in word_ids:
            L_w = self.exemplar_dict[w].size(0)
            term_cost = costs[w][L_w - 1].item()
            if term_cost < final_best_cost:
                final_best_cost = term_cost
                final_best_history = histories[w][L_w - 1]

        tokens = [h[0] for h in final_best_history]
        alignments = []
        for i in range(len(final_best_history)):
            start_f = final_best_history[i][1]
            end_f = final_best_history[i + 1][1] - 1 if i + 1 < len(final_best_history) else T - 1
            alignments.append((start_f, end_f))

        return TokenParseResult(tokens=tokens, alignments=alignments, total_cost=final_best_cost)
