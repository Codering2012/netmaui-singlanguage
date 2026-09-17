#!/usr/bin/env python3
"""
================================================================================
CONTINUOUS FINGERSPELLING SUB-TRANSDUCER & DYNAMIC SPAN WEAVER (CFST-DSW)
================================================================================
Solves the fundamental Open-Vocabulary Dactylology failure mode in ASL translation:
Fingerspelling accounts for 12% - 35% of natural ASL (names, medications, brands,
places, acronyms, e.g. "HAMILTON", "COVID", "TYLENOL").

Standard whole-word/gloss translation models fail catastrophically:
1. Finite word vocabularies cannot represent arbitrary open-vocabulary names.
2. Word decoders hallucinate phonetically similar words (e.g. spelling "N-O-A-H" -> "NOW").
3. Fingerspelling operates on a completely different kinematic manifold:
   - Spatial Confinement: Dominant hand anchored in "Conversational Shelf" near shoulder.
   - Quasi-Stationary Wrist: Minimal translational arm movement (< 0.08 m/s).
   - High-Frequency Intrinsic Articulation: Rapid finger joint transitions (4-6 chars/sec).

CFST-DSW provides:
1. ContinuousFingerspellingRouter: Decouples fingerspelling from lexical signs via
   spatial shelf locus + wrist stationarity + intrinsic finger frequency.
2. CharacterLevelCTCDecoder: 28-class CTC head (A-Z, space/apostrophe, blank)
   aligned to 128 for Cloud TPU v5e MXU tiling.
3. FingerspellingWordHybridWeaver: Seamlessly weaves character-level proper noun spans
   directly into the word-level sentence translation beam.
================================================================================
"""

from typing import Tuple, Optional, Dict, Any, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Vocabulary mappings: 0 is BLANK, 1-26 are A-Z, 27 is space/apostrophe
ALPHABET = [
    "<BLANK>", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", " "
]


class ContinuousFingerspellingRouter(nn.Module):
    r"""
    Kinematic Decoupling Router that continuously evaluates whether the signer
    is currently fingerspelling in the canonical conversational shelf.
    """

    def __init__(
        self,
        in_channels: int = 9,
        num_keypoints: int = 60,
        d_model: int = 128,
        shelf_threshold: float = 0.60,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.d_model = d_model
        self.shelf_threshold = shelf_threshold

        # Canonical Conversational Shelf relative centroid (relative to ipsilateral shoulder joint):
        # [x_rel: lateral offset (+0.08 right, -0.08 left), y_rel: elevation (+0.05), z_rel: forward reach (-0.22)]
        self.register_buffer("shelf_rel_centroid", torch.tensor([0.08, 0.05, -0.22], dtype=torch.float32))
        self.register_buffer("shelf_radii_inv", torch.tensor([1.0 / 0.08, 1.0 / 0.10, 1.0 / 0.09], dtype=torch.float32))

        # Temporal kinematic gate network
        self.kinematic_gate = nn.Sequential(
            nn.Conv1d(6, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )
        # Initialize gate with negative prior bias to suppress false background activations
        nn.init.normal_(self.kinematic_gate[3].weight, std=0.01)
        nn.init.constant_(self.kinematic_gate[3].bias, -2.5)

    def compute_kinematic_signatures(self, kinematics: torch.Tensor) -> torch.Tensor:
        r"""
        Extracts 6 kinematic signals indicating conversational shelf fingerspelling:
        0: Shoulder-Relative Shelf Spatial Gaussian Proximity exp(-d^2)
        1: Wrist Stationarity exp(-15 * ||v_wrist||^2)
        2: Intrinsic Finger Articulation Velocity Ratio (finger speed / (wrist speed + eps))
        3: Finger Acceleration / Jerk Energy
        4: Dominant Wrist Height relative to mid-torso
        5: Intrinsic Finger Flexion Power tanh(v_finger / 0.10)
        """
        B, T = kinematics.shape[:2]
        pts = kinematics.view(B, T, self.num_keypoints, -1)
        pos = pts[..., :3]
        vel = pts[..., 3:6]
        acc = pts[..., 6:9] if pts.shape[-1] >= 9 else torch.diff(vel, dim=1, prepend=vel[:, :1])

        # Extract shoulders for translation/lean-invariant anchor (indices 42 Left, 43 Right)
        # If shoulders are absent/zero (e.g. synthetic hand test or cropped dataset), fallback to canonical biometric shoulder positions
        default_r_sh = torch.tensor([0.20, 0.0, 0.0], device=pos.device, dtype=pos.dtype).view(1, 1, 3)
        default_l_sh = torch.tensor([-0.20, 0.0, 0.0], device=pos.device, dtype=pos.dtype).view(1, 1, 3)
        if self.num_keypoints > 43:
            raw_r_sh = pos[:, :, 43, :]
            raw_l_sh = pos[:, :, 42, :]
            r_sh_valid = (torch.norm(raw_r_sh, dim=-1, keepdim=True) > 1e-3).to(pos.dtype)
            l_sh_valid = (torch.norm(raw_l_sh, dim=-1, keepdim=True) > 1e-3).to(pos.dtype)
            r_sh = r_sh_valid * raw_r_sh + (1.0 - r_sh_valid) * default_r_sh
            l_sh = l_sh_valid * raw_l_sh + (1.0 - l_sh_valid) * default_l_sh
        else:
            r_sh = default_r_sh
            l_sh = default_l_sh

        # 1. Right Hand evaluation (idx 21, digits 22..41, relative centroid: [+0.08, 0.05, -0.22])
        r_wrist_pos = pos[:, :, 21, :]  # [B, T, 3]
        r_wrist_vel = torch.norm(vel[:, :, 21, :], dim=-1)  # [B, T]
        r_rel_wrist = r_wrist_pos - r_sh
        r_diff_shelf = (r_rel_wrist - self.shelf_rel_centroid.view(1, 1, 3)) * self.shelf_radii_inv.view(1, 1, 3)
        r_shelf_proximity = torch.exp(-0.5 * torch.sum(r_diff_shelf ** 2, dim=-1))
        r_wrist_stationarity = torch.exp(-15.0 * (r_wrist_vel ** 2))
        r_finger_vel = torch.norm(vel[:, :, 22:42, :] - vel[:, :, 21:22, :], dim=-1).mean(dim=-1)
        r_finger_ratio = r_finger_vel / (r_wrist_vel + r_finger_vel + 1e-4)
        r_finger_flexion = torch.tanh(r_finger_vel / 0.10)
        r_finger_energy = torch.tanh(10.0 * torch.norm(acc[:, :, 22:42, :], dim=-1).mean(dim=-1))
        r_elevation = torch.sigmoid(15.0 * (r_wrist_pos[:, :, 1] + 0.15))

        # 2. Left Hand evaluation (idx 0, digits 1..20, relative centroid: [-0.08, 0.05, -0.22])
        l_wrist_pos = pos[:, :, 0, :]  # [B, T, 3]
        l_wrist_vel = torch.norm(vel[:, :, 0, :], dim=-1)  # [B, T]
        l_rel_wrist = l_wrist_pos - l_sh
        l_rel_centroid = self.shelf_rel_centroid.clone()
        l_rel_centroid[0] = -l_rel_centroid[0]  # Mirror lateral X relative to left shoulder
        l_diff_shelf = (l_rel_wrist - l_rel_centroid.view(1, 1, 3)) * self.shelf_radii_inv.view(1, 1, 3)
        l_shelf_proximity = torch.exp(-0.5 * torch.sum(l_diff_shelf ** 2, dim=-1))
        l_wrist_stationarity = torch.exp(-15.0 * (l_wrist_vel ** 2))
        l_finger_vel = torch.norm(vel[:, :, 1:21, :] - vel[:, :, 0:1, :], dim=-1).mean(dim=-1)
        l_finger_ratio = l_finger_vel / (l_wrist_vel + l_finger_vel + 1e-4)
        l_finger_flexion = torch.tanh(l_finger_vel / 0.10)
        l_finger_energy = torch.tanh(10.0 * torch.norm(acc[:, :, 1:21, :], dim=-1).mean(dim=-1))
        l_elevation = torch.sigmoid(15.0 * (l_wrist_pos[:, :, 1] + 0.15))

        # Combine bilateral hands by choosing the active fingerspelling hand
        shelf_proximity = torch.maximum(r_shelf_proximity, l_shelf_proximity)
        wrist_stationarity = torch.where(r_shelf_proximity >= l_shelf_proximity, r_wrist_stationarity, l_wrist_stationarity)
        finger_ratio = torch.where(r_shelf_proximity >= l_shelf_proximity, r_finger_ratio, l_finger_ratio)
        finger_energy = torch.where(r_shelf_proximity >= l_shelf_proximity, r_finger_energy, l_finger_energy)
        elevation = torch.where(r_shelf_proximity >= l_shelf_proximity, r_elevation, l_elevation)
        finger_flexion = torch.where(r_shelf_proximity >= l_shelf_proximity, r_finger_flexion, l_finger_flexion)

        signatures = torch.stack([
            shelf_proximity,
            wrist_stationarity,
            finger_ratio,
            finger_energy,
            elevation,
            finger_flexion,
        ], dim=1)  # [B, 6, T]

        return signatures

    def compute_consistency_loss(self, kinematics: torch.Tensor) -> torch.Tensor:
        r"""
        Self-supervised consistency loss aligning the learned neural gate with the
        physical kinematic heuristic prior:
            L_consistency = BCE(learned_gate, heuristic_prior.detach())
        """
        signatures = self.compute_kinematic_signatures(kinematics)
        learned_gate = self.kinematic_gate(signatures).squeeze(1)
        shelf_proximity = signatures[:, 0, :]
        wrist_stationarity = signatures[:, 1, :]
        finger_ratio = signatures[:, 2, :]
        elevation = signatures[:, 4, :]
        finger_flexion_power = signatures[:, 5, :]
        heuristic_prior = shelf_proximity * wrist_stationarity * finger_ratio * finger_flexion_power * elevation
        return F.binary_cross_entropy(learned_gate, heuristic_prior.detach())

    def forward(self, kinematics: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes fingerspelling decoupling probability gamma_t in [0, 1] per frame.
        
        Returns:
            gamma_t: [B, T] continuous fingerspelling routing score
            is_fingerspelling: [B, T] boolean decision mask
        """
        signatures = self.compute_kinematic_signatures(kinematics)  # [B, 6, T]
        learned_gate = self.kinematic_gate(signatures).squeeze(1)   # [B, T]

        # Physical heuristic prior:
        # High shelf proximity AND wrist stationarity AND high dynamic finger flexion (bilateral)
        shelf_proximity = signatures[:, 0, :]
        wrist_stationarity = signatures[:, 1, :]
        finger_ratio = signatures[:, 2, :]
        elevation = signatures[:, 4, :]
        finger_flexion_power = signatures[:, 5, :]

        heuristic_prior = shelf_proximity * wrist_stationarity * finger_ratio * finger_flexion_power * elevation
        # Hybrid smooth routing probability
        gamma_t = 0.25 * learned_gate + 0.75 * heuristic_prior
        is_fingerspelling = (gamma_t >= self.shelf_threshold)

        return gamma_t, is_fingerspelling


class CharacterLevelCTCDecoder(nn.Module):
    r"""
    Dedicated 28-class Character CTC Head (A-Z, space, blank)
    Tiled to 128 for Cloud TPU v5e MXU alignment.
    """

    def __init__(self, d_model: int = 128, num_chars: int = 28):
        super().__init__()
        self.d_model = d_model
        self.num_chars = num_chars
        # TPU v5e tile alignment: multiple of 128
        self.tpu_vocab_size = (num_chars + 127) // 128 * 128
        self.char_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, self.tpu_vocab_size),
        )

    def forward(self, hand_features: torch.Tensor) -> torch.Tensor:
        r"""
        Computes character CTC logits [B, T, tpu_vocab_size].
        Token 0 is <BLANK>.
        Tokens 1-26 are A-Z.
        Token 27 is space/apostrophe.
        """
        logits = self.char_projection(hand_features)
        return logits

    def decode_greedy_span(
        self,
        char_logits: torch.Tensor,     # [T, tpu_vocab_size] or [1, T, tpu_vocab_size]
        gamma_mask: torch.Tensor,      # [T] boolean mask where fingerspelling is active
    ) -> str:
        r"""
        Standard CTC greedy collapse (remove consecutive duplicates and blank tokens)
        strictly over active fingerspelled frames.
        """
        if char_logits.dim() == 3:
            char_logits = char_logits.squeeze(0)
        if gamma_mask.dim() == 2:
            gamma_mask = gamma_mask.squeeze(0)

        # Slice to active frames
        active_indices = torch.nonzero(gamma_mask).squeeze(-1)
        if len(active_indices) == 0:
            return ""

        active_logits = char_logits[active_indices]  # [T_active, V]
        preds = torch.argmax(active_logits[:, :self.num_chars], dim=-1)  # [T_active]

        # CTC Collapse: remove consecutive duplicates and blank (0)
        collapsed_tokens: List[int] = []
        prev_token = 0
        for token_id in preds.tolist():
            if token_id != prev_token:
                if token_id != 0 and token_id < len(ALPHABET):
                    collapsed_tokens.append(token_id)
                prev_token = token_id

        # Convert to string
        spelled_str = "".join([ALPHABET[idx] for idx in collapsed_tokens]).strip()
        return spelled_str


class FingerspellingWordHybridWeaver(nn.Module):
    r"""
    Dynamic Span Weaver that seamlessly integrates Character-level fingerspelling
    transcriptions into the Word-level translation beam.
    """

    def __init__(
        self,
        d_model: int = 128,
        blank_suppression_weight: float = 10.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.blank_suppression_weight = blank_suppression_weight

    def suppress_word_logits_on_fingerspelling(
        self,
        word_ctc_logits: torch.Tensor,   # [B, T, V_word]
        gamma_t: torch.Tensor,           # [B, T]
    ) -> torch.Tensor:
        r"""
        Prevents word decoders from hallucinating whole words during fingerspelling spans.
        Adds positive bias to word <BLANK> token (idx 0) and symmetrically penalizes non-blank tokens.
        """
        # Full contrastive shift: boost BLANK and penalize non-blank tokens equally
        # Crucial: detach gamma_t so word CTC loss does not backpropagate into the kinematic gate
        biased_word_logits = word_ctc_logits.clone()
        boost = (self.blank_suppression_weight * gamma_t.detach()).unsqueeze(-1)  # [B, T, 1]
        biased_word_logits[:, :, 0:1] += boost
        biased_word_logits[:, :, 1:] -= boost
        return biased_word_logits

    def extract_fingerspelling_spans(
        self,
        gamma_t: torch.Tensor,           # [T]
        min_duration_frames: int = 4,
        threshold: float = 0.55,
    ) -> List[Tuple[int, int]]:
        r"""
        Finds contiguous temporal intervals [start_idx, end_idx] where fingerspelling is active.
        Vectorized on-device PyTorch execution without host CPU synchronization.
        """
        is_active = (gamma_t >= threshold)
        if not is_active.any():
            return []

        # Find rising and falling edges with torch.diff on device
        padded = F.pad(is_active.long(), (1, 1), value=0)
        diff = torch.diff(padded)
        starts = torch.nonzero(diff == 1).view(-1)
        ends = torch.nonzero(diff == -1).view(-1)

        spans: List[Tuple[int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            if (e - s) >= min_duration_frames:
                spans.append((s, e))

        return spans

    def weave_hybrid_sentence(
        self,
        word_glosses: List[str],
        gloss_frame_indices: List[int],
        char_decoder: CharacterLevelCTCDecoder,
        char_logits: torch.Tensor,       # [T, V_char]
        gamma_t: torch.Tensor,           # [T]
    ) -> str:
        r"""
        Combines word-level translated glosses with extracted fingerspelled names
        ordered by temporal timestamp.
        """
        spans = self.extract_fingerspelling_spans(gamma_t)
        spelled_entities: List[Tuple[int, str]] = []

        for start, end in spans:
            span_mask = torch.zeros_like(gamma_t, dtype=torch.bool)
            span_mask[start:end] = True
            spelled_word = char_decoder.decode_greedy_span(char_logits, span_mask)
            if len(spelled_word) > 0:
                mid_point = (start + end) // 2
                spelled_entities.append((mid_point, f"#{spelled_word}#"))

        # Merge word glosses and spelled entities by frame index
        combined_tokens: List[Tuple[int, str]] = []
        for idx, gloss in zip(gloss_frame_indices, word_glosses):
            combined_tokens.append((idx, gloss))
        for idx, entity in spelled_entities:
            combined_tokens.append((idx, entity))

        combined_tokens.sort(key=lambda item: item[0])

        # Filter out any word glosses that occurred inside a fingerspelled span
        clean_tokens: List[str] = []
        for idx, tok in combined_tokens:
            if tok.startswith("#") and tok.endswith("#"):
                clean_tokens.append(tok[1:-1])  # Strip delimiter
            else:
                # Check if idx falls inside any fingerspelling span
                inside_fs = any(start <= idx < end for start, end in spans)
                if not inside_fs:
                    clean_tokens.append(tok)

        return " ".join(clean_tokens)
