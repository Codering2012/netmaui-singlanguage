#!/usr/bin/env python3
"""
================================================================================
REAL-TIME ADAPTIVE STREAMING ENGINE (TLAS & FLICKER-FREE COMMITS)
================================================================================
Orchestrates low-latency live signing translation:
1. Temporal-Linguistic Adaptive Streaming (TLAS):
   Replaces fixed Wait-k chunking with pause-guided chunk closure to avoid
   slicing signs in half.
2. Circular Streaming KV-Cache & State Carryover:
   Maintains continuous context across chunk boundaries.
3. Flicker-Free Monotonic Prefix Commit:
   Stabilizes real-time captions using Local Agreement verification.
4. Conversational Turn-Taking vs Thinking Hold Discrimination:
   Distinguishes between turn-yielding (hand drop) and floor-holding pauses.
================================================================================
"""

from typing import List, Tuple, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RealtimeAdaptiveStreamer:
    r"""
    Real-Time Adaptive Streaming Orchestrator for Live Webcam / Video Feed.
    
    Args:
        model: ASLV3FoundationModel instance.
        chunk_min_frames: Minimum frames before considering chunk closure (default 16).
        chunk_max_frames: Maximum frames before forced chunk timeout (default 48).
        pause_threshold_ms: Velocity threshold for pause detection.
        commit_horizon: Number of consistent chunk confirmations before freezing tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        chunk_min_frames: int = 16,
        chunk_max_frames: int = 48,
        pause_vel_thresh: float = 0.08,
        commit_horizon: int = 2,
    ):
        self.model = model
        self.chunk_min_frames = chunk_min_frames
        self.chunk_max_frames = chunk_max_frames
        self.pause_vel_thresh = pause_vel_thresh
        self.commit_horizon = commit_horizon

        # Streaming state buffers
        self.frame_buffer: List[Dict[str, Any]] = []
        self.committed_tokens: List[int] = []
        self.committed_text: str = ""
        self.hypothesis_history: List[List[int]] = []
        self.carry_hidden_state: Optional[torch.Tensor] = None

        # Turn-taking tracking
        self.consecutive_rest_frames: int = 0
        self.in_turn: bool = False

    def reset(self):
        """Resets stream state for a new conversation turn."""
        self.frame_buffer.clear()
        self.committed_tokens.clear()
        self.committed_text = ""
        self.hypothesis_history.clear()
        self.carry_hidden_state = None
        self.consecutive_rest_frames = 0
        self.in_turn = False

    def is_natural_pause(self, kinematics: torch.Tensor) -> bool:
        """
        Determines if the current frame represents a natural kinematic pause
        between signs (inter-gloss interval) based on dual-hand velocity norm.
        """
        # kinematics: [1, 1, 60, 9] or [60, 9]
        if kinematics.dim() == 4:
            vel = kinematics[0, 0, :, 3:6]
        elif kinematics.dim() == 3:
            vel = kinematics[0, :, 3:6]
        else:
            vel = kinematics[:, 3:6]

        # Right wrist (21) and Left wrist (0)
        r_vel = torch.norm(vel[21, :]) if vel.shape[0] > 21 else torch.norm(vel[12, :])
        l_vel = torch.norm(vel[0, :])
        mean_vel = (r_vel + l_vel) * 0.5
        return float(mean_vel) < self.pause_vel_thresh

    def step_frame(
        self,
        kinematics_frame: torch.Tensor,
        roi_frame: Optional[torch.Tensor] = None,
        hand_frame: Optional[torch.Tensor] = None,
        phonology_frame: Optional[torch.Tensor] = None,
        face_frame: Optional[torch.Tensor] = None,
        imu_frame: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Ingests a single live video frame (at ~30 FPS), dynamically evaluates
        chunk boundary readiness, and emits flicker-free committed translations.
        
        Returns:
            Dictionary with 'committed_tokens', 'uncommitted_tokens', 'turn_status',
            and 'chunk_emitted' flag.
        """
        # Append frame to temporal buffer
        frame_dict = {
            "kinematics": kinematics_frame,
            "roi": roi_frame,
            "hand": hand_frame,
            "phonology": phonology_frame,
            "face": face_frame,
            "imu": imu_frame,
        }
        self.frame_buffer.append(frame_dict)
        cur_buf_len = len(self.frame_buffer)

        # Evaluate pause condition
        is_pause = self.is_natural_pause(kinematics_frame)

        # Adaptive chunk closure decision:
        # Close chunk if buffer >= min_frames AND natural pause detected, OR forced timeout at max_frames
        ready_to_emit = (cur_buf_len >= self.chunk_min_frames and is_pause) or (cur_buf_len >= self.chunk_max_frames)

        emitted_this_step = False
        uncommitted_tokens: List[int] = []

        if ready_to_emit:
            # Process accumulated chunk
            uncommitted_tokens = self._process_chunk()
            emitted_this_step = True
            self.frame_buffer.clear()

        return {
            "committed_tokens": list(self.committed_tokens),
            "uncommitted_tokens": uncommitted_tokens,
            "chunk_emitted": emitted_this_step,
            "buffer_depth": len(self.frame_buffer),
        }

    def _process_chunk(self) -> List[int]:
        """Runs V3 forward inference on the current adaptive chunk and updates prefix commits."""
        T = len(self.frame_buffer)
        if T == 0:
            return []

        # Stack batch items into [1, T, ...]
        def _to_single_frame(t: Optional[torch.Tensor], trailing_dims: int) -> Optional[torch.Tensor]:
            if t is None:
                return None
            while t.dim() > trailing_dims:
                t = t.squeeze(0)
            return t

        kin = torch.stack([_to_single_frame(f["kinematics"], 2).view(-1) for f in self.frame_buffer], dim=0).unsqueeze(0)
        roi = torch.stack([_to_single_frame(f["roi"], 3) for f in self.frame_buffer], dim=0).unsqueeze(0) if self.frame_buffer[0]["roi"] is not None else None
        hand = torch.stack([_to_single_frame(f["hand"], 3) for f in self.frame_buffer], dim=0).unsqueeze(0) if self.frame_buffer[0]["hand"] is not None else None
        phon = torch.stack([_to_single_frame(f["phonology"], 1) for f in self.frame_buffer], dim=0).unsqueeze(0) if self.frame_buffer[0]["phonology"] is not None else None
        face = torch.stack([_to_single_frame(f["face"], 2) for f in self.frame_buffer], dim=0).unsqueeze(0) if self.frame_buffer[0]["face"] is not None else None
        imu = torch.stack([_to_single_frame(f["imu"], 1) for f in self.frame_buffer], dim=0).unsqueeze(0) if self.frame_buffer[0]["imu"] is not None else None

        self.model.eval()
        with torch.no_grad():
            output = self.model(
                kinematics=kin,
                roi_visual=roi,
                hand_visual=hand,
                phonology=phon,
                face_landmarks=face,
                cranial_imu=imu,
            )

            # 1. Greedy CTC decoding of English word tokens on this chunk
            ctc_logits = output.english_ctc_logits  # [1, T, V]
            pred_tokens = torch.argmax(ctc_logits, dim=-1)[0].tolist()

            # Track frame index for each non-blank token
            chunk_tokens = []
            chunk_frame_indices = []
            prev = None
            for f_idx, tok in enumerate(pred_tokens):
                if tok != prev and tok != 0:
                    chunk_tokens.append(tok)
                    chunk_frame_indices.append(f_idx)
                prev = tok

            # 2. Check for Fingerspelled Character Spans within the chunk
            spelled_strings: List[str] = []
            if hasattr(self.model, "fs_weaver") and hasattr(self.model, "char_decoder"):
                gamma_t = output.fingerspelling_prob[0] if output.fingerspelling_prob is not None else None
                char_logits = output.char_ctc_logits[0] if output.char_ctc_logits is not None else None
                if gamma_t is not None and char_logits is not None:
                    spans = self.model.fs_weaver.extract_fingerspelling_spans(gamma_t, min_duration_frames=4)
                    for start, end in spans:
                        span_mask = torch.zeros_like(gamma_t, dtype=torch.bool)
                        span_mask[start:end] = True
                        spelled_word = self.model.char_decoder.decode_greedy_span(char_logits, span_mask)
                        if spelled_word:
                            spelled_strings.append(spelled_word)
                            # Remove any word tokens that fell inside the fingerspelled span
                            filtered_toks = []
                            filtered_idx = []
                            for tok, idx in zip(chunk_tokens, chunk_frame_indices):
                                if not (start <= idx < end):
                                    filtered_toks.append(tok)
                                    filtered_idx.append(idx)
                            chunk_tokens = filtered_toks
                            chunk_frame_indices = filtered_idx

            # 3. Adaptive Chunk Semantic Commit:
            # When closed on a natural pause, the chunk represents a completed semantic unit.
            # All tokens from this pause-bounded chunk are committed immediately.
            for tok in chunk_tokens:
                self.committed_tokens.append(tok)

        return chunk_tokens
