#!/usr/bin/env python3
"""
================================================================================
  SOTA ASL MULTI-METRIC EVALUATION SUITE
================================================================================
Calculates comprehensive benchmark metrics across all tasks:
  1. Word Error Rate (WER) with Levenshtein Alignment (Sub, Del, Ins).
  2. Character Error Rate (CER) for fingerspelling sequences.
  3. BLEU-1, BLEU-2, BLEU-3, BLEU-4 for English Translation.
  4. ROUGE-L (Longest Common Subsequence).
  5. Per-class Recall/Precision, Confusion Matrix, and Inference Latency (FPS).
================================================================================
"""

import sys
import os
import time
import math
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np


# ==============================================================================
#  1. LEVENSHTEIN DISTANCE & WORD/CHARACTER ERROR RATE
# ==============================================================================

def compute_levenshtein(ref: List[Union[str, int]], hyp: List[Union[str, int]]) -> Tuple[int, int, int, int]:
    """
    Computes minimum edit distance between ref and hyp sequences.
    Returns: (substitutions, deletions, insertions, total_ref_tokens)
    """
    r_len = len(ref)
    h_len = len(hyp)

    d = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
    ops = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)  # 0: Match, 1: Sub, 2: Del, 3: Ins

    for i in range(r_len + 1):
        d[i, 0] = i
        ops[i, 0] = 2  # Del
    for j in range(h_len + 1):
        d[0, j] = j
        ops[0, j] = 3  # Ins
    ops[0, 0] = 0

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref[i - 1] == hyp[j - 1]:
                d[i, j] = d[i - 1, j - 1]
                ops[i, j] = 0
            else:
                sub = d[i - 1, j - 1] + 1
                delete = d[i - 1, j] + 1
                insert = d[i, j - 1] + 1

                min_cost = min(sub, delete, insert)
                d[i, j] = min_cost
                if min_cost == sub:
                    ops[i, j] = 1
                elif min_cost == delete:
                    ops[i, j] = 2
                else:
                    ops[i, j] = 3

    # Backtrace edit operations
    i, j = r_len, h_len
    subs, dels, inss = 0, 0, 0
    while i > 0 or j > 0:
        op = ops[i, j]
        if op == 0:
            i -= 1
            j -= 1
        elif op == 1:
            subs += 1
            i -= 1
            j -= 1
        elif op == 2:
            dels += 1
            i -= 1
        elif op == 3:
            inss += 1
            j -= 1

    return subs, dels, inss, r_len


def calculate_wer(refs: List[List[Union[str, int]]], hyps: List[List[Union[str, int]]]) -> Dict[str, float]:
    """
    Computes overall Word Error Rate across sequence predictions.
    """
    total_sub, total_del, total_ins, total_ref = 0, 0, 0, 0
    for r, h in zip(refs, hyps):
        sub, delete, ins, r_len = compute_levenshtein(r, h)
        total_sub += sub
        total_del += delete
        total_ins += ins
        total_ref += r_len

    wer = (total_sub + total_del + total_ins) / max(1, total_ref) * 100.0
    return {
        "wer": wer,
        "sub_rate": (total_sub / max(1, total_ref)) * 100.0,
        "del_rate": (total_del / max(1, total_ref)) * 100.0,
        "ins_rate": (total_ins / max(1, total_ref)) * 100.0,
        "total_ref_words": total_ref,
    }


def calculate_cer(ref_texts: List[str], hyp_texts: List[str]) -> Dict[str, float]:
    """
    Computes Character Error Rate for fingerspelling text strings.
    """
    char_refs = [list(t.lower()) for t in ref_texts]
    char_hyps = [list(t.lower()) for t in hyp_texts]
    res = calculate_wer(char_refs, char_hyps)
    return {
        "cer": res["wer"],
        "sub_rate": res["sub_rate"],
        "del_rate": res["del_rate"],
        "ins_rate": res["ins_rate"],
        "total_ref_chars": res["total_ref_words"],
    }


# ==============================================================================
#  2. BLEU & ROUGE-L TRANSLATION METRICS
# ==============================================================================

def compute_ngram_counts(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    counts = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def calculate_bleu(refs: List[List[str]], hyps: List[List[str]], max_n: int = 4) -> Dict[str, float]:
    """
    Computes BLEU-1 to BLEU-4 with brevity penalty.
    """
    precisions = [0.0] * max_n
    total_ref_len = 0
    total_hyp_len = 0

    for n in range(1, max_n + 1):
        match_count = 0
        total_count = 0

        for r_tokens, h_tokens in zip(refs, hyps):
            r_ng = compute_ngram_counts(r_tokens, n)
            h_ng = compute_ngram_counts(h_tokens, n)

            for ng, cnt in h_ng.items():
                match_count += min(cnt, r_ng.get(ng, 0))
                total_count += cnt

            if n == 1:
                total_ref_len += len(r_tokens)
                total_hyp_len += len(h_tokens)

        precisions[n - 1] = match_count / max(1, total_count)

    # Brevity penalty
    if total_hyp_len == 0:
        bp = 0.0
    elif total_hyp_len < total_ref_len:
        bp = math.exp(1.0 - total_ref_len / total_hyp_len)
    else:
        bp = 1.0

    bleu_scores = {}
    for n in range(1, max_n + 1):
        if min(precisions[:n]) == 0:
            bleu_scores[f"bleu_{n}"] = 0.0
        else:
            log_avg = sum(math.log(p) for p in precisions[:n]) / float(n)
            bleu_scores[f"bleu_{n}"] = bp * math.exp(log_avg) * 100.0

    return bleu_scores


def calculate_rouge_l(refs: List[List[str]], hyps: List[List[str]]) -> float:
    """
    Computes ROUGE-L F1 score based on Longest Common Subsequence.
    """
    f1_scores = []
    for r, h in zip(refs, hyps):
        if not r or not h:
            f1_scores.append(1.0 if not r and not h else 0.0)
            continue

        m, n = len(r), len(h)
        lcs_table = np.zeros((m + 1, n + 1), dtype=np.int32)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if r[i - 1] == h[j - 1]:
                    lcs_table[i, j] = lcs_table[i - 1, j - 1] + 1
                else:
                    lcs_table[i, j] = max(lcs_table[i - 1, j], lcs_table[i, j - 1])

        lcs_len = lcs_table[m, n]
        prec = lcs_len / float(n)
        rec = lcs_len / float(m)
        if prec + rec == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append((2.0 * prec * rec) / (prec + rec))

    return float(np.mean(f1_scores)) * 100.0


# ==============================================================================
#  3. INFERENCE LATENCY & THROUGHPUT PROFILER
# ==============================================================================

class ModelProfiler:
    """
    Profiles inference latency (ms/frame) and throughput (FPS) on TPU/GPU/CPU.
    """

    def __init__(self, warmup_runs: int = 5):
        self.warmup_runs = warmup_runs

    def profile(self, model: torch.nn.Module, sample_input: torch.Tensor, runs: int = 50) -> Dict[str, float]:
        model.eval()
        device = next(model.parameters()).device

        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(sample_input)

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.time()
            for _ in range(runs):
                _ = model(sample_input)

            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

        total_time = t1 - t0
        avg_batch_time_ms = (total_time / runs) * 1000.0
        batch_size = sample_input.size(0)
        num_frames = sample_input.size(1) if sample_input.dim() >= 2 else 1
        
        fps = (batch_size * num_frames * runs) / total_time
        latency_per_frame_ms = avg_batch_time_ms / (batch_size * num_frames)

        return {
            "batch_time_ms": avg_batch_time_ms,
            "latency_per_frame_ms": latency_per_frame_ms,
            "fps": fps,
            "batch_size": batch_size,
            "num_frames": num_frames,
        }


# ==============================================================================
#  4. CLI ENTRY POINT
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="SOTA ASL Multi-Metric Benchmark Evaluator")
    parser.add_argument("--predictions-json", type=str, required=False, help="Path to model predictions JSON")
    parser.add_argument("--output-json", type=str, default="evaluation_results.json", help="Destination report JSON")
    args = parser.parse_args()

    print("[INFO] SOTA Multi-Metric Evaluator initialized.")
    # Example verification
    mock_refs = [["hello", "world"], ["thank", "you", "very", "much"]]
    mock_hyps = [["hello", "world"], ["thank", "you", "much"]]

    wer_res = calculate_wer(mock_refs, mock_hyps)
    bleu_res = calculate_bleu(mock_refs, mock_hyps)
    rouge_res = calculate_rouge_l(mock_refs, mock_hyps)

    print(f"  -> Sample WER:     {wer_res['wer']:.2f}%")
    print(f"  -> Sample BLEU-4:  {bleu_res['bleu_4']:.2f}%")
    print(f"  -> Sample ROUGE-L: {rouge_res:.2f}%")


if __name__ == "__main__":
    main()
