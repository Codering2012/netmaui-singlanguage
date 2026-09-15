#!/usr/bin/env python3
"""
================================================================================
ASL FOUNDATION MODEL — MODEL SOUPS & GREEDY WEIGHT AVERAGING ENGINE
================================================================================
Implements Model Soups & Stochastic Weight Averaging (Wortsman et al. / SWA):
1. Uniform & Weighted Model Soups:
     theta_soup = sum_{k=1}^K alpha_k * theta_k,  sum alpha_k = 1
   Achieves ensemble-level generalization with ZERO extra latency & 1x VRAM.
2. Greedy Soup Search Algorithm:
     Iteratively tests candidate model checkpoints and accumulates those that
     improve validation accuracy / reduce validation loss on calibration sets.
3. Online SWA / EMA Weight Accumulator:
     theta_SWA = (n * theta_SWA + theta_curr) / (n + 1)
================================================================================
"""

from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import copy
import torch
import torch.nn as nn


class ASLModelSoupsWeightAverager:
    """
    Model Soups and Greedy Weight Averaging Engine for ASL Foundation Models.
    """

    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.swa_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.swa_count = 0

    @staticmethod
    def create_uniform_soup(
        models_or_state_dicts: List[Union[nn.Module, Dict[str, torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes uniform average of weights across K models: theta_soup = 1/K * sum theta_k.
        """
        if len(models_or_state_dicts) == 0:
            raise ValueError("Cannot create soup from empty model list!")

        num_models = len(models_or_state_dicts)
        state_dicts = [
            m.state_dict() if isinstance(m, nn.Module) else m
            for m in models_or_state_dicts
        ]

        soup_dict: Dict[str, torch.Tensor] = {}
        for key in state_dicts[0].keys():
            if state_dicts[0][key].dtype.is_floating_point:
                soup_dict[key] = sum(sd[key].float() for sd in state_dicts) / float(num_models)
                soup_dict[key] = soup_dict[key].to(state_dicts[0][key].dtype)
            else:
                # Non-floating point tensors (e.g. step counters, bool buffers) take first model's value
                soup_dict[key] = state_dicts[0][key].clone()

        return soup_dict

    @staticmethod
    def create_weighted_soup(
        models_or_state_dicts: List[Union[nn.Module, Dict[str, torch.Tensor]]],
        weights: List[float],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes weighted average of weights: theta_soup = sum alpha_k * theta_k.
        """
        assert len(models_or_state_dicts) == len(weights), "Weights and models count must match!"
        total_w = sum(weights)
        norm_weights = [w / total_w for w in weights]

        state_dicts = [
            m.state_dict() if isinstance(m, nn.Module) else m
            for m in models_or_state_dicts
        ]

        soup_dict: Dict[str, torch.Tensor] = {}
        for key in state_dicts[0].keys():
            if state_dicts[0][key].dtype.is_floating_point:
                accum = sum(w * sd[key].float() for w, sd in zip(norm_weights, state_dicts))
                soup_dict[key] = accum.to(state_dicts[0][key].dtype)
            else:
                soup_dict[key] = state_dicts[0][key].clone()

        return soup_dict

    def greedy_soup_search(
        self,
        candidate_models: List[Union[nn.Module, Dict[str, torch.Tensor]]],
        eval_fn: Callable[[nn.Module], float],
        higher_is_better: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], float]:
        """
        Executes Greedy Soup Algorithm (Wortsman et al.):
        Sorts candidates, iteratively adds checkpoints that improve validation metric.
        Returns: (best_soup_state_dict, selected_indices, best_score)
        """
        if len(candidate_models) == 0:
            raise ValueError("No candidate models provided!")

        # 1. Evaluate individual candidates
        scored_candidates = []
        for idx, candidate in enumerate(candidate_models):
            if isinstance(candidate, dict):
                self.base_model.load_state_dict(candidate)
                score = eval_fn(self.base_model)
            else:
                score = eval_fn(candidate)
            scored_candidates.append((idx, candidate, score))

        # 2. Sort candidates by initial performance
        scored_candidates.sort(key=lambda x: x[2], reverse=higher_is_better)

        # 3. Initialize soup with best individual model
        best_idx, best_cand, best_score = scored_candidates[0]
        soup_models = [best_cand]
        selected_indices = [best_idx]

        # 4. Iteratively evaluate additions
        for idx, candidate, score in scored_candidates[1:]:
            potential_soup_models = soup_models + [candidate]
            potential_soup_dict = self.create_uniform_soup(potential_soup_models)

            self.base_model.load_state_dict(potential_soup_dict)
            potential_score = eval_fn(self.base_model)

            is_improvement = (potential_score > best_score) if higher_is_better else (potential_score < best_score)
            if is_improvement:
                soup_models = potential_soup_models
                selected_indices.append(idx)
                best_score = potential_score

        final_soup_dict = self.create_uniform_soup(soup_models)
        return final_soup_dict, selected_indices, best_score

    def accumulate_swa_step(self, current_model: nn.Module):
        """
        Updates running Stochastic Weight Average (SWA) state dict:
        theta_SWA = (n * theta_SWA + theta_curr) / (n + 1)
        """
        curr_dict = current_model.state_dict()
        if self.swa_state_dict is None:
            self.swa_state_dict = copy.deepcopy(curr_dict)
            self.swa_count = 1
        else:
            self.swa_count += 1
            n = self.swa_count
            for key in self.swa_state_dict.keys():
                if self.swa_state_dict[key].dtype.is_floating_point:
                    self.swa_state_dict[key] = (
                        (n - 1) * self.swa_state_dict[key].float() + curr_dict[key].float()
                    ) / float(n)
                    self.swa_state_dict[key] = self.swa_state_dict[key].to(curr_dict[key].dtype)
                else:
                    self.swa_state_dict[key] = curr_dict[key].clone()
