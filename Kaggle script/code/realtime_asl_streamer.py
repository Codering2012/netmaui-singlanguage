#!/usr/bin/env python3
"""
================================================================================
  REAL-TIME ASL STREAMING INFERENCE ENGINE FOR LIVE WEBCAM & EDGE DEPLOYMENT
================================================================================
Implements low-latency sliding-window streaming inference:
  1. Circular frame buffer (T=30 sliding window with hop_size=5 frames).
  2. Speculative CTC token streaming with confidence filtering & debouncing.
  3. Real-time token stabilization & partial sentence assembly.
================================================================================
"""

import sys
import os
import time
import collections
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import numpy as np

# Setup paths
workspace_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(workspace_root / "train_tpu"))

from train_all_in_one_tpu_v2 import ASLFoundationModel


class RealtimeASLStreamer:
    """
    Low-latency streaming engine for continuous sign language recognition.
    """

    def __init__(
        self,
        model: ASLFoundationModel,
        vocab_list: List[str],
        window_size: int = 30,
        hop_size: int = 5,
        min_confidence: float = 0.40,
        blank_idx: int = 0,
    ):
        self.model = model
        self.vocab_list = vocab_list
        self.window_size = window_size
        self.hop_size = hop_size
        self.min_confidence = min_confidence
        self.blank_idx = blank_idx

        self.model.eval()
        self.buffer = collections.deque(maxlen=window_size)
        self.emitted_tokens: List[str] = []
        self.last_token_idx: int = blank_idx
        self.frame_count: int = 0

    def reset(self):
        """Clears buffer and token history."""
        self.buffer.clear()
        self.emitted_tokens.clear()
        self.last_token_idx = self.blank_idx
        self.frame_count = 0

    @torch.no_grad()
    def push_frame(
        self,
        frame_keypoints_60x9: np.ndarray,
        roi_crop_256x256: Optional[np.ndarray] = None,
    ) -> Optional[str]:
        """
        Pushes a single frame (60 keypoints x 9 kinematics) into the circular buffer.
        Returns newly emitted word/gloss if a confident sign gesture was detected.
        """
        self.buffer.append((frame_keypoints_60x9, roi_crop_256x256))
        self.frame_count += 1

        if len(self.buffer) < self.window_size or (self.frame_count % self.hop_size != 0):
            return None

        # Build tensor batch [1, T, 60, 9]
        feat_list = [f[0] for f in self.buffer]
        feat_tensor = torch.from_numpy(np.stack(feat_list, axis=0)).unsqueeze(0).float()

        # Visual ROI tensor [1, T, 256, 256, 3] if available
        if roi_crop_256x256 is not None:
            roi_list = [f[1] if f[1] is not None else np.zeros((256, 256, 3), dtype=np.uint8) for f in self.buffer]
            roi_tensor = torch.from_numpy(np.stack(roi_list, axis=0)).unsqueeze(0)
        else:
            roi_tensor = None

        # Forward pass through encoder + CTC head
        enc_out = self.model._encode(feat_tensor, roi_visual=roi_tensor)
        h_seq = enc_out[1]
        ctc_logits = self.model.ctc_head(h_seq)  # [1, T, V]
        probs = torch.softmax(ctc_logits[0, -1, :], dim=-1)  # Focus on latest frame

        top_prob, top_idx = torch.max(probs, dim=-1)
        prob_val = float(top_prob.item())
        idx_val = int(top_idx.item())

        # Debounce and confidence check
        if prob_val >= self.min_confidence and idx_val != self.blank_idx:
            if idx_val != self.last_token_idx:
                self.last_token_idx = idx_val
                if idx_val < len(self.vocab_list):
                    word = self.vocab_list[idx_val]
                    self.emitted_tokens.append(word)
                    return word
        elif idx_val == self.blank_idx:
            self.last_token_idx = self.blank_idx

        return None

    def get_accumulated_transcript(self) -> str:
        """Returns the assembled English sentence."""
        return " ".join(self.emitted_tokens)


def main():
    print("[INFO] Initializing Real-Time ASL Streamer...")
    vocab = ["<blank>", "HELLO", "HOW", "ARE", "YOU", "THANK-YOU"]
    model = ASLFoundationModel(
        num_enc_layers=4,
        num_dec_layers=4,
        d_enc=256,
        d_dec=256,
        vocab_size=2560,
        english_vocab_size=23552,
        is_causal=True,
    )
    streamer = RealtimeASLStreamer(model, vocab_list=vocab, window_size=20, hop_size=4)

    print("[+] Pushing 50 mock streaming frames...")
    for f in range(50):
        mock_kp = np.random.randn(60, 9).astype(np.float32)
        new_token = streamer.push_frame(mock_kp)
        if new_token is not None:
            print(f"  [Frame {f}] Emitted: {new_token}")

    print(f"[+] Final Transcript: '{streamer.get_accumulated_transcript()}'")


if __name__ == "__main__":
    main()
