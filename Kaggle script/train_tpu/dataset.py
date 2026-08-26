import os
import re
import json
import math
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

import torch
import torch.nn.functional as F_torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

os.environ["TPU_PREMAPPED_BUFFER_SIZE"] = "1073741824"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.08"
os.environ["PJRT_ALLOCATOR_FRACTION"] = "0.08"
os.environ["XLA_CLIENT_MEM_FRACTION"] = "0.08"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Global multiprocessing caches
_GLOBAL_RECORDS_CACHE: Dict[str, Any] = {}
_GLOBAL_ACTIVE_RECORDS_CACHE: Dict[str, Any] = {}
_GLOBAL_SHARD_GROUPS_CACHE: Dict[str, Any] = {}

# Global Task Routing Constants
TASK_ISOLATED = 0
TASK_FINGERSPELLING = 1
TASK_SENTENCE = 2

# Labels that indicate unlabeled/placeholder data — skip during indexing
_SKIP_LABELS = frozenset(
    {
        "",  # empty label
        "unknown",  # generic fallback
        "none",  # bare 'none' without angle brackets
    }
)


def normalize_vocabulary(label_to_idx: Dict) -> Dict:
    if (
        isinstance(label_to_idx, dict)
        and "label_to_idx" in label_to_idx
        and isinstance(label_to_idx["label_to_idx"], dict)
    ):
        label_to_idx = label_to_idx["label_to_idx"]
    clean_l2i = {}
    if isinstance(label_to_idx, dict):
        for k, v in label_to_idx.items():
            if isinstance(v, bool):
                raise TypeError(f"Invalid boolean value for vocabulary ID: '{k}' -> {v}")
            k_str = str(k).strip().lower()
            if isinstance(v, int):
                val_int = v
            elif isinstance(v, dict):
                if "id" in v:
                    idx_val = v["id"]
                elif "idx" in v:
                    idx_val = v["idx"]
                elif "label_idx" in v:
                    idx_val = v["label_idx"]
                else:
                    raise ValueError(f"Malformed vocabulary entry for '{k_str}': {v} (missing 'id' or 'idx')")
                if isinstance(idx_val, bool):
                    raise TypeError(f"Invalid boolean value for vocabulary ID in dict: '{k}' -> {idx_val}")
                val_int = int(idx_val)
            elif isinstance(v, str) and str(k).isdigit():
                val_int = int(k)
                k_str = str(v).strip().lower()
            else:
                try:
                    val_int = int(v)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Malformed non-integer vocabulary value for entry '{k_str}': {v}") from exc

            if k_str in clean_l2i and clean_l2i[k_str] != val_int:
                raise ValueError(f"Vocabulary case-insensitivity collision: '{k_str}' mapped to both {clean_l2i[k_str]} and {val_int}")
            clean_l2i[k_str] = val_int

    for k, v in list(clean_l2i.items()):
        if v < 0:
            raise ValueError(f"Negative vocabulary ID found: {k} -> {v}")

    return clean_l2i


class GlossVocabulary:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    OFFSET = 4

    def __init__(self, label_to_idx: Dict):
        clean_l2i = normalize_vocabulary(label_to_idx)
        self.label_to_idx = clean_l2i
        self.idx_to_label = {}
        for k, v in self.label_to_idx.items():
            if v in self.idx_to_label:
                raise ValueError(
                    f"[FATAL VOCAB ERROR] Duplicate ID {v} found in vocabulary! "
                    f"'{k}' conflicts with '{self.idx_to_label[v]}'."
                )
            self.idx_to_label[v] = k
        special_names = {"<pad>": 0, "[pad]": 0, "pad": 0, "<bos>": 1, "[bos]": 1, "bos": 1, "<eos>": 2, "[eos]": 2, "eos": 2, "<unk>": 3, "[unk]": 3, "unk": 3}
        has_special = any(k in clean_l2i for k in special_names)
        regular_ids = [v for k, v in clean_l2i.items() if k not in special_names]
        min_regular = min(regular_ids, default=self.OFFSET)
        
        self.already_offset = (min_regular >= self.OFFSET) or has_special
        if self.already_offset:
            # Validate that reserved IDs 0-3 match special tokens
            for tok_name, tok_id in [(k, v) for k, v in clean_l2i.items() if v < self.OFFSET]:
                if tok_name not in special_names or special_names[tok_name] != tok_id:
                    raise ValueError(f"[FATAL VOCAB ERROR] Reserved token ID {tok_id} mapped to non-special token '{tok_name}'.")
        max_idx = max(clean_l2i.values()) if clean_l2i else 0
        self.vocab_size = (max_idx + 1) if self.already_offset else max(len(self.label_to_idx), max_idx + 1) + self.OFFSET
        self.output_map = {}

    def __len__(self) -> int:
        return self.vocab_size

    def gloss_to_token(self, gloss: str) -> int:
        raw = self.label_to_idx.get(gloss.strip().lower(), None)
        if raw is None:
            return self.UNK_ID
        if self.already_offset:
            return raw
        return raw + self.OFFSET

    def token_to_gloss(self, tid: int) -> str:
        if tid == self.PAD_ID:
            return "<PAD>"
        if tid == self.BOS_ID:
            return "<BOS>"
        if tid == self.EOS_ID:
            return "<EOS>"
        if tid == self.UNK_ID:
            return "<UNK>"
        if self.already_offset:
            gloss = self.idx_to_label.get(tid, "<UNK>")
        else:
            gloss = self.idx_to_label.get(tid - self.OFFSET, "<UNK>")
        return self.output_map.get(gloss, gloss)

    def encode(self, text: str, allow_unk: bool = True, **kwargs) -> list:
        """Encode a gloss string to a list of token IDs (with OFFSET applied)."""
        res = []
        for w in text.split():
            idx = self.label_to_idx.get(w.strip().lower(), None)
            if idx is not None:
                if self.already_offset:
                    res.append(idx)
                else:
                    res.append(idx + self.OFFSET)
            elif allow_unk:
                res.append(self.UNK_ID)
        return res


class EnglishVocabulary:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(
        self,
        vocab_path: Optional[Union[str, Path]] = None,
        use_bpe: bool = False,
        **kwargs,
    ):
        self.token_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.frozen = True
        self.is_valid = True

        resolved_path = None
        if vocab_path:
            vp = Path(vocab_path)
            if vp.is_file() and vp.exists():
                resolved_path = vp
            elif vp.is_dir() and (vp / "english_vocab.json").exists():
                resolved_path = vp / "english_vocab.json"

        if not resolved_path:
            candidates = [
                Path("./english_vocab.json"),
                Path("../english_vocab.json"),
                Path("/kaggle/working/english_vocab.json"),
            ]
            if vocab_path:
                vp = Path(vocab_path)
                candidates.extend([
                    vp / "english_vocab.json",
                    vp.parent / "english_vocab.json",
                    vp.parent.parent / "english_vocab.json",
                    vp.parent.parent.parent / "english_vocab.json",
                ])

            # Search under /kaggle/input if available
            kaggle_input = Path("/kaggle/input")
            if kaggle_input.exists():
                for p in kaggle_input.glob("**/english_vocab.json"):
                    candidates.append(p)
                    break

            for cand in candidates:
                if cand and cand.is_file() and cand.exists():
                    resolved_path = cand
                    break

        if resolved_path:
            try:
                with open(resolved_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, dict):
                    if "token_to_id" in data and isinstance(data["token_to_id"], dict):
                        mapping = data["token_to_id"]
                    else:
                        mapping = data

                    self.token_to_id = {}
                    self.id_to_token = {}
                    for k, v in mapping.items():
                        if int(v) in self.id_to_token:
                            raise ValueError(f"[FATAL VOCAB ERROR] Duplicate ID {v} in English vocabulary for token '{k}' (conflicts with '{self.id_to_token[int(v)]}')")
                        self.token_to_id[str(k)] = int(v)
                        self.id_to_token[int(v)] = str(k)
                elif isinstance(data, list):
                    self.token_to_id = {str(w): i for i, w in enumerate(data)}
                    self.id_to_token = {i: str(w) for i, w in enumerate(data)}
                
                # Enforce special token contract
                if (
                    self.token_to_id.get("<PAD>") != self.PAD_ID or
                    self.token_to_id.get("<BOS>") != self.BOS_ID or
                    self.token_to_id.get("<EOS>") != self.EOS_ID or
                    self.token_to_id.get("<UNK>") != self.UNK_ID
                ):
                    raise ValueError(
                        f"[FATAL VOCAB ERROR] 'english_vocab.json' violates special token contract. "
                        f"Expected PAD=0, BOS=1, EOS=2, UNK=3. "
                        f"Found PAD={self.token_to_id.get('<PAD>')}, BOS={self.token_to_id.get('<BOS>')}, "
                        f"EOS={self.token_to_id.get('<EOS>')}, UNK={self.token_to_id.get('<UNK>')}."
                    )

                print(f"[INFO] EnglishVocabulary loaded {len(self.token_to_id)} tokens from {resolved_path}")
            except Exception as e:
                raise RuntimeError(f"[FATAL VOCAB ERROR] Failed to load 'english_vocab.json' from {resolved_path}: {e}") from e
        elif not use_bpe:
            # Fallback to standard 128-token ASCII character vocabulary for --disable-bpe mode
            import string
            chars = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", " "] + list(string.ascii_lowercase) + list(string.digits) + list(".,!?;:'\"-()/")
            self.token_to_id = {c: i for i, c in enumerate(chars)}
            self.id_to_token = {i: c for i, c in enumerate(chars)}
            print(f"[INFO] 'english_vocab.json' not found; initialized default {len(self.token_to_id)}-token character vocabulary for --disable-bpe mode.")
        else:
            raise FileNotFoundError(
                f"[FATAL VOCAB ERROR] Required 'english_vocab.json' not found at '{vocab_path}' or candidate paths under /kaggle/input/. "
                f"Please ensure english_vocab.json is included in your dataset or pass --disable-bpe!"
            )

    def freeze(self):
        pass

    def encode(self, text: str, allow_unk: bool = True) -> List[int]:
        clean_text = re.sub(r"([.?!,;:—\-\(\)\[\]\"\'])", r" \1 ", text.strip().lower())
        words = clean_text.split()
        res = []
        for w in words:
            if w in self.token_to_id:
                res.append(self.token_to_id[w])
            elif allow_unk:
                res.append(self.UNK_ID)
        return res

    def decode(self, ids: List[int]) -> str:
        if not ids:
            return ""
        
        return " ".join(
            [
                self.id_to_token.get(int(i), "<UNK>")
                for i in ids
                if int(i) not in (0, 1, 2)
            ]
        )

    def __len__(self) -> int:
        if not self.id_to_token:
            return 0
        return max(self.id_to_token.keys()) + 1


class LandmarkAugmenter:
    r"""
    Progressive Noise Curriculum Data Augmentation for 3D WholeBody landmark sequences.

    Given a sequence of 3D spatial coordinates $X \\in \\mathbb{R}^{T \times K \times C}$,
    this module sequentially applies affine transformations and stochastically drops nodes
    to robustify models against missing/noisy pose estimation inputs.

    Mathematical Formulation:
    1. Scaling: $X' = s \\cdot X$ where $s \\sim U(1 - \\alpha, 1 + \\alpha)$
    2. Translation: $X'' = X' + \\Delta x$ where $\\Delta x \\sim U(-\\beta, \\beta)$
    3. Rotation (2D Spatial):
       Let $R(\\theta) = \\begin{bmatrix} \\cos(\\theta) & -\\sin(\\theta) \\\\ \\sin(\\theta) & \\cos(\\theta) \\end{bmatrix}$
       $X'''_{xy} = (X''_{xy} - \\mu_{xy}) R(\\theta)^T + \\mu_{xy}$ where $\\mu_{xy}$ is the valid centroid.
    4. Gaussian Jitter: $X'''' = X''' + \\epsilon$ where $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2)$
    5. Node Dropout: Independent Bernoulli masking on spatial dimension $K$ and temporal dimension $T$.

    The noise scale $\\gamma \\in (0, 1]$ parameterizes the intensity of $\\alpha, \\beta, \\theta,$ and $\\sigma$ progressively
    over the training epochs (Curriculum Learning).
    """

    def __init__(
        self,
        base_jitter_std: float = 0.035,
        max_scale_range: Tuple[float, float] = (0.85, 1.15),
        max_shift_range: float = 0.035,
        max_rotation_range: float = 10.0,
        max_kp_drop_prob: float = 0.05,
        max_frame_drop_prob: float = 0.035,
        noise_level: float = 0.02,
        max_len: int = 256,
    ):
        self.base_jitter_std = base_jitter_std
        self.max_scale_range = max_scale_range
        self.max_shift_range = max_shift_range
        self.max_rotation_range = max_rotation_range
        self.max_kp_drop_prob = max_kp_drop_prob
        self.max_frame_drop_prob = max_frame_drop_prob
        self.noise_level = max(0.0, min(1.0, noise_level))
        self.max_len = max_len

    def set_noise_level(self, level: float) -> None:
        """Sets progressive noise level ratio (0.0 to 1.0)."""
        effective_level = float(level)
        self.noise_level = max(0.0, min(1.0, effective_level))

    def __call__(
        self,
        feat_arr: np.ndarray,
        noise_level: Optional[float] = None,
        frame_indices: Optional[np.ndarray] = None,
    ):
        if noise_level is not None:
            self.set_noise_level(noise_level)
        T, K, C = feat_arr.shape
        if frame_indices is None:
            frame_indices = np.arange(T, dtype=np.int64)

        if T == 0 or (noise_level is not None and noise_level <= 0.0) or self.noise_level <= 0.0:
            return feat_arr if frame_indices is None else (feat_arr, frame_indices)

        aug = feat_arr.copy()

        # Extract XYZ for spatial transforms.
        xyz = aug[:, :, :3]

        jitter_std = self.base_jitter_std * self.noise_level
        rot_range = self.max_rotation_range * self.noise_level
        shift_range = self.max_shift_range * self.noise_level
        kp_drop_prob = self.max_kp_drop_prob * self.noise_level
        frame_drop_prob = self.max_frame_drop_prob * self.noise_level

        finger_drop_prob = (0.025 * self.noise_level) * 1.80
        timestretch_prob = (0.04 * self.noise_level) * 1.80
        warping_prob = (0.05 * self.noise_level) * 1.80
        hand_occ_prob = (0.01 * self.noise_level) * 1.80

        # 1. Scaling (Supports asymmetric max_scale_range)
        min_s = 1.0 + (self.max_scale_range[0] - 1.0) * self.noise_level
        max_s = 1.0 + (self.max_scale_range[1] - 1.0) * self.noise_level
        xyz = xyz * np.random.uniform(min_s, max_s)

        # 2. Shift
        if shift_range > 0:
            xyz[:, :, 0] = xyz[:, :, 0] + np.random.uniform(-shift_range, shift_range)
            xyz[:, :, 1] = xyz[:, :, 1] + np.random.uniform(-shift_range, shift_range)

        # 3. Rotation (2D only)
        if rot_range > 0:
            roll_deg = np.random.uniform(-rot_range, rot_range)
            rad_r = np.radians(roll_deg)
            rot_mat = np.array(
                [
                    [np.cos(rad_r), -np.sin(rad_r)],
                    [np.sin(rad_r), np.cos(rad_r)],
                ],
                dtype=np.float32,
            )
            valid_xyz = xyz[:, :, :2]
            valid_mask = np.abs(valid_xyz).sum(axis=-1, keepdims=True) > 0
            center = (valid_xyz * valid_mask).sum(
                axis=(0, 1), keepdims=True
            ) / np.maximum(valid_mask.sum(axis=(0, 1), keepdims=True), 1.0)
            xyz[:, :, :2] = (
                np.dot(xyz[:, :, :2] - center, rot_mat.T).reshape((T, K, 2)) + center
            )

        # 4. Jitter (Masked to preserve valid/missing coordinate distinction - Claim 37 Fix)
        if jitter_std > 0:
            valid_coords_mask = np.abs(xyz).sum(axis=-1, keepdims=True) > 0
            jitter = np.random.normal(0, jitter_std, size=xyz.shape).astype(np.float32)
            xyz = np.where(valid_coords_mask, xyz + jitter, 0.0)

        # Reconstruct pos
        pos = xyz

        if T > 20 and np.random.rand() < timestretch_prob:
            rate = np.random.uniform(0.8, 1.2)
            new_T = min(int(T * rate), self.max_len)
            old_t = np.linspace(0, 1, T)
            new_t = np.linspace(0, 1, new_T)
            pos_t = torch.from_numpy(pos).permute(1, 2, 0).unsqueeze(0)  # [1, K, C, T]
            B_1, K_k, C_c, T_t = pos_t.shape
            pos_t_flat = pos_t.view(B_1, K_k * C_c, T_t)
            pos_resampled_flat = F_torch.interpolate(
                pos_t_flat, size=new_T, mode="linear", align_corners=True
            )
            pos_resampled = pos_resampled_flat.view(B_1, K_k, C_c, new_T)
            pos = pos_resampled.squeeze(0).permute(2, 0, 1).numpy()
            T = new_T
            frame_indices = np.interp(new_t, old_t, frame_indices.astype(np.float32))

        if T > 25 and np.random.rand() < warping_prob:
            warp_idx = np.clip(
                (
                    np.power(np.linspace(0, 1, T), np.random.uniform(0.5, 1.5))
                    * (T - 1)
                ).astype(int),
                0,
                T - 1,
            )
            pos = pos[warp_idx]
            frame_indices = frame_indices[warp_idx]

        # ====================================================================
        # 6. CREATE UNIFIED MASK AND APPLY TO POS FIRST
        # ====================================================================
        unified_mask = np.ones((T, K, 1), dtype=np.float32)

        if kp_drop_prob > 0:
            unified_mask *= (np.random.rand(T, K, 1) > kp_drop_prob).astype(np.float32)

        if np.random.rand() < finger_drop_prob:
            all_finger_groups = [
                list(range(1, 5)),
                list(range(5, 9)),
                list(range(9, 13)),
                list(range(13, 17)),
                list(range(17, 21)),
                list(range(22, 26)),
                list(range(26, 30)),
                list(range(30, 34)),
                list(range(34, 38)),
                list(range(38, 42)),
            ]
            n_drop = np.random.randint(1, 3)
            chosen_indices = np.random.choice(
                len(all_finger_groups), size=n_drop, replace=False
            )
            for idx_c in chosen_indices:
                unified_mask[:, all_finger_groups[idx_c], :] = 0.0

        if np.random.rand() < hand_occ_prob and T > 10:
            occ_l = np.random.randint(4, max(5, min(T, T // 2 + 1)))
            occ_s = np.random.randint(0, max(1, T - occ_l + 1))
            unified_mask[
                occ_s : occ_s + occ_l,
                range(0, 21) if np.random.rand() > 0.5 else range(21, 42),
                :,
            ] = 0.0

        if np.random.rand() < 0.02 and T > 5:
            unified_mask[
                np.random.randint(0, T),
                range(0, 21) if np.random.rand() > 0.5 else range(21, 42),
                :,
            ] = 0.0

        # 7. Progressive Temporal Frame Dropout
        if frame_drop_prob > 0 and T > 8:
            keep_mask = np.random.rand(T) > frame_drop_prob
            if np.sum(keep_mask) >= 4:
                pos = pos[keep_mask]
                unified_mask = unified_mask[keep_mask]
                frame_indices = frame_indices[keep_mask]
                T = pos.shape[0]

        # ====================================================================
        # 🚨 FIX: CALCULATE KINEMATICS ON CONTINUOUS POS AFTER DROPOUT
        # using actual time elapsed (actual_dt)
        # ====================================================================
        vel = np.zeros_like(pos)
        acc = np.zeros_like(pos)
        if T > 1:
            actual_dt = (
                (frame_indices[1:] - frame_indices[:-1])
                .astype(np.float32)
                .reshape(-1, 1, 1)
            )
            actual_dt[actual_dt == 0] = 1.0  # Safe guard
            vel[1:] = (pos[1:] - pos[:-1]) / actual_dt
            vel[0] = vel[1]  # Strict causal boundary condition (Claim 60)
            acc[1:] = (vel[1:] - vel[:-1]) / actual_dt
            acc[0] = acc[1]

        pos = pos * unified_mask
        vel = vel * unified_mask
        acc = acc * unified_mask

        features = np.concatenate([pos, vel, acc], axis=-1)

        return features, frame_indices


def motion_aware_sample_indices(feat_arr: np.ndarray, max_len: int) -> np.ndarray:
    """
    Downsamples a sequence of length T > max_len to max_len frames in O(T) linear time.
    Uses fast L1 motion energy and np.argpartition (O(N) vs O(N log N) full sort).
    """
    T = feat_arr.shape[0]
    if T <= max_len:
        return np.arange(T)

    # 1. Compute fast L1 frame-to-frame motion magnitude (O(T) linear time)
    if feat_arr.ndim >= 2:
        xyz = feat_arr[..., :3]
        flat_feat = xyz.reshape(T, -1)
        motion_energy = np.abs(flat_feat[1:] - flat_feat[:-1]).sum(axis=-1)
        motion_energy = np.pad(motion_energy, (0, 1), mode="edge")
    else:
        motion_energy = np.ones(T, dtype=np.float32)

    # 2. Hybrid sampling: 70% uniform grid, 30% top motion frames in O(N) time
    uniform_count = int(max_len * 0.70)
    motion_count = max_len - uniform_count

    uniform_idx = np.linspace(0, T - 1, num=uniform_count, dtype=int)

    mask = np.ones(T, dtype=bool)
    mask[uniform_idx] = False
    remaining_idx = np.where(mask)[0]

    if len(remaining_idx) > 0 and motion_count > 0:
        n_rem = len(remaining_idx)
        k_top = min(motion_count, n_rem)
        sub_energy = motion_energy[remaining_idx]
        if k_top < n_rem:
            top_partition = np.argpartition(sub_energy, -k_top)[-k_top:]
        else:
            top_partition = np.arange(n_rem)
        motion_idx = remaining_idx[top_partition]
        selected_idx = np.concatenate([uniform_idx, motion_idx])
    else:
        selected_idx = np.linspace(0, T - 1, num=max_len, dtype=int)

    selected_idx.sort()
    return selected_idx


def clear_global_dataset_caches():
    """Clears module-level dataset metadata caches and invokes gc.collect() to prevent RAM accumulation."""
    _GLOBAL_RECORDS_CACHE.clear()
    _GLOBAL_ACTIVE_RECORDS_CACHE.clear()
    _GLOBAL_SHARD_GROUPS_CACHE.clear()
    import gc
    gc.collect()


# Global metadata & shard data caches to avoid massive IPC transfer and repeated disk reads
_GLOBAL_RECORDS_CACHE = {}
_GLOBAL_ACTIVE_RECORDS_CACHE = {}
_GLOBAL_SHARD_GROUPS_CACHE = {}


class ASLShardedDataset(Dataset):
    """
    PyTorch Dataset for reading sharded ASL landmark records from preprocessed phase 1 directory.
    Enforces strict static sequence padding and static batch shapes for PyTorch XLA TPU execution.
    Integrates ASL-LEX lexical features and real-world camera noise augmentation.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path] = r"E:\datasets\results\asl_preprocessed_phase1",
        split: str = "train",
        stride_length: int = 1,
        max_len: int = 256,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        worker_idx: int = 0,
        num_workers: int = 1,
        shuffle_shards: bool = True,
        stage: str = "full_mixture",
        augment: bool = False,
        shared_progress=None,
        shared_epoch=None,
        use_bpe: bool = False,
        model_name: str = "Qwen/Qwen2.5-0.5B",
    ):
        super().__init__()
        self.use_bpe = use_bpe
        self.model_name = model_name
        self.shared_progress = shared_progress
        self.shared_epoch = shared_epoch

        # Auto-discover candidate directories if specified directory doesn't exist
        input_dir = Path(dataset_dir)
        if not input_dir.exists():
            candidates = [
                Path(
                    "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"
                ),
                Path(
                    "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"
                ),
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl"),
                Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl"),
                Path("./asl_preprocessed_phase1"),
            ]
            candidates = [c for c in candidates if os.name != "nt" or not str(c).startswith("/kaggle/")]
            input_dir = next((c for c in candidates if c.exists()), input_dir)

        if (input_dir / split).exists():
            self.dataset_dir = input_dir / split
        else:
            self.dataset_dir = input_dir

        self.split = split
        self.max_len = max_len
        self.stride_length = stride_length
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.feature_dim = num_keypoints * channels_per_kp
        self.worker_idx = worker_idx
        self.num_workers = num_workers
        self.shuffle_shards = shuffle_shards
        self.stage = stage
        self.augmenter = LandmarkAugmenter(max_len=self.max_len) if augment else None

        # Resolve Master Vocabulary Mapping
        self.label_to_idx = {}
        vocab_candidates = [
            self.dataset_dir / "vocab_map.json",
            self.dataset_dir / "vocabulary_mapping_global.json",
            self.dataset_dir / "output_mapping.json",
            self.dataset_dir / f"vocabulary_mapping_{split}.json",
            self.dataset_dir / "vocabulary_mapping_train.json",
            self.dataset_dir.parent / "output_mapping.json",
            self.dataset_dir.parent / "vocab_map.json",
            self.dataset_dir.parent / "vocabulary_mapping_train.json",
            self.dataset_dir.parent / f"vocabulary_mapping_{split}.json",
            input_dir / "vocabulary_mapping_global.json",
            input_dir / "vocabulary_mapping_train.json",
            input_dir / "output_mapping.json",
            input_dir / f"vocabulary_mapping_{split}.json",
            self.dataset_dir.parent / "sign_to_prediction_index_map.json",
            self.dataset_dir.parent.parent / "sign_to_prediction_index_map.json",
            self.dataset_dir.parent.parent / "vocabulary_mapping_train.json",
            Path(
                "/kaggle/input/frakenstein-asl-final-version/sign_to_prediction_index_map.json"
            ),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1/vocabulary_mapping_global.json"
            ),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1/vocabulary_mapping_train.json"
            ),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1/vocabulary_mapping_train.json"
            ),
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/vocabulary_mapping_train.json"
            ),
        ]
        vocab_candidates = [p for p in vocab_candidates if os.name != "nt" or not str(p).startswith("/kaggle/")]
        
        # Add any vocabulary json under dataset_dir or dataset_dir.parent
        for search_p in [self.dataset_dir, self.dataset_dir.parent, self.dataset_dir.parent.parent]:
            if search_p.exists():
                for pat in ["*vocab*.json", "*mapping*.json"]:
                    vocab_candidates.extend(sorted(list(search_p.glob(pat))))

        vocab_candidates = [
            c for c in vocab_candidates
            if os.name != "nt" or not str(c).startswith("/kaggle/")
        ]
        for vc in vocab_candidates:
            if vc.exists():
                try:
                    with open(vc, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, dict) and "label_to_idx" in data:
                            self.label_to_idx = data["label_to_idx"]
                        elif isinstance(data, dict):
                            self.label_to_idx = data
                        if self.label_to_idx:
                            break
                except Exception:
                    pass

        metadata_path = self.dataset_dir / "metadata.json"
        if metadata_path.exists() and not self.label_to_idx:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    self.label_to_idx = meta.get("label_to_idx", {})
            except Exception:
                pass

        if not self.label_to_idx:
            raise FileNotFoundError(
                f"[FATAL VOCAB ERROR] Could not find 'vocab_map.json' or 'vocabulary_mapping_global.json' at '{self.dataset_dir}' or under /kaggle/input/. "
                f"Please ensure vocab_map.json is included in your dataset!"
            )

        # Normalize vocabulary to lowercase
        normalized_vocab = {}
        for key, value in self.label_to_idx.items():
            if isinstance(value, dict):
                value = int(value.get("id", value.get("idx", -1)))
            normalized_vocab[str(key).strip().lower()] = int(value)
        self.label_to_idx = normalized_vocab
        assert all(k == k.lower() for k in self.label_to_idx)
        # English Vocabulary & How2Sign Sentence Sidecar Loader
        english_vocab_file = self.dataset_dir / "english_vocab.json"
        if not english_vocab_file.exists():
            english_vocab_file = input_dir / "english_vocab.json"
        self.english_vocab = EnglishVocabulary(
            vocab_path=english_vocab_file
        )

        # Removed redundant how2sign_sentence_map sidecar loading, as the
        # physical .pt shards now natively store their English sentences.

        # ASL-LEX Lexical Grammatical Map Initialization
        self.asl_lex_map = {}
        grammar_candidates = [
            Path(__file__).resolve().parent.parent
            / "preprocessing"
            / "grammar_logic.json",
            input_dir / "grammar_logic.json",
            Path(
                "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/grammar_logic.json"
            ),
        ]
        csv_candidates = [
            Path("/kaggle/input/datasets/tranquocbao2012/asl-lex/signdata.csv"),
            Path("/kaggle/input/asl-lex/signdata.csv"),
            Path(__file__).resolve().parent.parent / "preprocessing" / "signdata.csv",
        ]

        grammar_candidates = [
            c for c in grammar_candidates
            if os.name != "nt" or not str(c).startswith("/kaggle/")
        ]
        csv_candidates = [
            c for c in csv_candidates
            if os.name != "nt" or not str(c).startswith("/kaggle/")
        ]
        pos_categories = {"Noun": 0, "Verb": 1, "Adjective": 2, "Adverb": 3}
        for gc in grammar_candidates:
            if gc.exists():
                try:
                    with open(gc, "r", encoding="utf-8") as f:
                        g_data = json.load(f)
                        for word_key, val_dict in g_data.items():
                            cls_str = (
                                val_dict.get("class", "Other")
                                if isinstance(val_dict, dict)
                                else str(val_dict)
                            )
                            word_clean = word_key.strip().lower()
                            self.asl_lex_map[word_clean] = pos_categories.get(
                                cls_str, 4
                            )
                        if self.asl_lex_map:
                            break
                except Exception:
                    pass

        if not self.asl_lex_map:
            for cc in csv_candidates:
                if cc.exists():
                    try:
                        import csv

                        with open(cc, "r", encoding="utf-8", errors="ignore") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                word_clean = (
                                    (row.get("LemmaID") or row.get("EntryID") or "")
                                    .strip()
                                    .lower()
                                    .replace("_", "")
                                    .replace("-", "")
                                )
                                word_clean = re.sub(r"\d+$", "", word_clean)
                                cls_str = (row.get("LexicalClass") or "Other").strip()
                                if word_clean:
                                    self.asl_lex_map[word_clean] = pos_categories.get(
                                        cls_str, 4
                                    )
                        if self.asl_lex_map:
                            break
                    except Exception:
                        pass

        # Collect shard files and partition among workers if distributed
        if self.dataset_dir.name == self.split:
            split_dir = self.dataset_dir
        else:
            split_dir = self.dataset_dir / self.split
        all_shard_files = []
        if split_dir.exists():
            all_shard_files = sorted(list(set(split_dir.glob("*.pt")).union(set(split_dir.rglob("*.pt")))))
        else:
             raise FileNotFoundError(
                f"[FATAL DATASET ERROR] Split directory '{split_dir}' does not exist. "
                f"Validation split separation requires explicit train/val subdirectories to prevent data contamination."
            )

        if not all_shard_files:
            raise FileNotFoundError(
                f"[FATAL DATASET ERROR] No '.pt' preprocessed dataset files found in '{split_dir}'. "
                f"Please verify that your preprocessed ASL dataset is correctly attached in Kaggle and --data-dir path is accurate!"
            )

        _n_workers = self.num_workers if self.num_workers > 0 else 1
        if len(all_shard_files) > 0:
            if self.worker_idx < len(all_shard_files):
                self.shard_files = all_shard_files[self.worker_idx :: _n_workers]
            else:
                self.shard_files = [all_shard_files[self.worker_idx % len(all_shard_files)]]
        else:
            self.shard_files = []

        self.dataset_name = f"dataset_{self.split}_w{self.worker_idx}_of_{self.num_workers}_{len(self.shard_files)}_{self.max_len}_{self.stride_length}"

        # Load records metadata from allocated shards
        import hashlib
        self.dataset_id = f"{self.dataset_name}_{self.stage}_{self.max_len}_{self.stride_length}_{self.channels_per_kp}_v2_{hashlib.md5(str(self.dataset_dir).encode()).hexdigest()[:8]}"
        self.cached_shard_path: Optional[Path] = None
        self.cached_shard_data: Optional[List] = None

        if self.dataset_id not in _GLOBAL_RECORDS_CACHE:
            self._load_records_metadata()

    def _load_records_metadata(self) -> None:
        """Loads metadata from manifest JSONL files or fallback shard files."""
        cache_key = self.dataset_id

        temp_metadata = []
        class_counts = defaultdict(int)

        # CRITICAL FIX (Point 8): Disable stale manifest reading.
        # Manifests can point to missing shards and silently inject zero-tensors.
        # We now force the dataset to read shards directly to build the index.
        if not temp_metadata:

            def _index_shard(args):
                shard_idx, shard_path = args
                local_metas = []
                local_counts = defaultdict(int)
                valid_label_set = set(int(v) for v in self.label_to_idx.values())
                try:
                    try:
                        shard_data = torch.load(
                            shard_path,
                            map_location="cpu",
                            weights_only=False,
                            mmap=True,
                        )
                    except Exception:
                        shard_data = torch.load(
                            shard_path, map_location="cpu", weights_only=False
                        )

                    items = (
                        shard_data.items()
                        if isinstance(shard_data, dict)
                        else enumerate(shard_data)
                    )
                    for key_or_idx, rec in items:
                        if not isinstance(rec, dict):
                            continue

                        f_key = key_or_idx if isinstance(shard_data, dict) else None
                        item_idx = (
                            key_or_idx if not isinstance(shard_data, dict) else None
                        )

                        task_str = (
                            str(rec.get("task", rec.get("task_str", "unknown")))
                            .strip()
                            .lower()
                        )
                        source_str = str(rec.get("source", "unknown")).strip().lower()
                        raw_label_str = (
                            str(
                                rec.get(
                                    "raw_label_str",
                                    rec.get("text", rec.get("label", "")),
                                )
                            )
                            .strip()
                            .lower()
                        )
                        raw_label_idx = rec.get("label_idx", -1)

                        token_ids = []
                        # SOURCE / TASK AWARE ROUTING FIRST
                        lbl_clean = -1
                        if (
                            task_str == "fingerspelling_sequence"
                            or "chicago" in source_str
                        ):
                            if raw_label_str in _SKIP_LABELS:
                                continue
                            raw_label_str = (
                                str(
                                    rec.get(
                                        "label_proc", rec.get("label", raw_label_str)
                                    )
                                )
                                .strip()
                                .lower()
                                .replace("<sp>", " ")
                            )
                        elif (
                            task_str == "sentence_level"
                            or source_str.startswith("how2sign")
                            or raw_label_str == "how2sign_sequence"
                        ):

                            if raw_label_str and raw_label_str != "how2sign_sequence":
                                pass
                            else:
                                continue  # Reject record lacking matching sentence metadata

                        # Unconditionally extract token_ids for all records if available
                        if "gloss_seq" in rec:
                            gs = rec["gloss_seq"]
                            token_ids = (
                                gs.tolist()
                                if isinstance(gs, torch.Tensor)
                                else list(gs)
                            )
                        elif raw_label_str and not (
                            task_str == "sentence_level"
                            or source_str.startswith("how2sign")
                        ):
                            idx = self.label_to_idx.get(
                                raw_label_str.strip().lower(), None
                            )
                            if idx is not None:
                                if isinstance(idx, dict):
                                    idx = idx.get("id", idx.get("idx", -1))
                                token_ids.append(max(-1, int(idx)))
                            else:
                                parts = raw_label_str.split()
                                for p in parts:
                                    idx = self.label_to_idx.get(p.strip().lower(), None)
                                    if isinstance(idx, dict):
                                        idx = idx.get("id", idx.get("idx", -1))
                                    if idx is not None:
                                        token_ids.append(max(-1, int(idx)))
                                    else:
                                        token_ids.append(-1)

                        if (
                            not token_ids
                            and raw_label_str.strip().lower() not in self.label_to_idx
                            and (raw_label_idx is None or int(raw_label_idx) < 0)
                            and not (
                                task_str == "sentence_level"
                                or source_str.startswith("how2sign")
                                or task_str == "fingerspelling_sequence"
                                or "chicago" in source_str
                            )
                        ):
                            continue

                        if not token_ids:
                            if raw_label_idx is None or int(raw_label_idx) < 0:
                                if task_str == "sentence_level" or source_str.startswith("how2sign") or task_str == "fingerspelling_sequence" or "chicago" in source_str:
                                    lbl_clean = -1
                                else:
                                    continue
                            else:
                                lbl_idx = int(raw_label_idx)
                                if lbl_idx not in valid_label_set:
                                    raise RuntimeError(
                                        f"DATASET CORRUPTION: label_idx {lbl_idx} out of bounds."
                                    )
                                lbl_clean = lbl_idx
                        else:
                            lbl_clean = int(token_ids[0]) if token_ids else -1
                        if token_ids:
                            for t in token_ids:
                                if t >= 0:
                                    local_counts[int(t)] += 1
                        elif lbl_clean >= 0:
                            local_counts[lbl_clean] += 1

                        source_id = 0
                        if "chicago" in source_str:
                            source_id = 1
                        elif "how2sign" in source_str:
                            source_id = 2
                        elif "citizen" in source_str:
                            source_id = 3

                        task_id = TASK_ISOLATED
                        if (
                            task_str == "fingerspelling_sequence"
                            or "chicago" in source_str
                        ):
                            task_id = TASK_FINGERSPELLING
                        elif task_str == "sentence_level" or source_str.startswith(
                            "how2sign"
                        ):
                            task_id = TASK_SENTENCE

                        local_metas.append(
                            (
                                shard_idx,
                                f_key,
                                item_idx,
                                lbl_clean,
                                float(
                                    rec.get("quality", rec.get("sample_weight", 1.0))
                                ),
                                token_ids,
                                task_id,
                                source_id,
                                raw_label_str,
                            )
                        )
                    del shard_data
                except RuntimeError:
                    raise
                except Exception as _shard_e:
                    raise RuntimeError(
                        f"Failed indexing shard {shard_path}: {_shard_e}"
                    ) from _shard_e
                return local_metas, local_counts

            import os
            import hashlib

            # Use local directory to avoid tmpfs RAM consumption on Kaggle
            cache_dir = Path("./dataset_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)

            def safe_mtime(sf):
                try:
                    return int(os.path.getmtime(sf)) if os.path.exists(sf) else 0
                except Exception:
                    return 0

            mtime_sum = (
                sum(safe_mtime(sf) + os.path.getsize(sf) for sf in self.shard_files)
                if self.shard_files
                else 0
            )
            vocab_hash = hashlib.md5(str(len(self.english_vocab)).encode()).hexdigest()[
                :8
            ]
            label_hash = hashlib.md5(
                str(sorted(self.label_to_idx.items())).encode()
            ).hexdigest()[:8]
            cache_hash = hashlib.sha256(
                f"{self.split}_{self.worker_idx}_{len(self.shard_files)}_{self.max_len}_{mtime_sum}_{vocab_hash}_{label_hash}_chicago_{min(self.max_len, 256)}_english_{min(self.max_len, 256)}".encode()
            ).hexdigest()[:12]
            cache_name = f"asl_metadata_{self.split}_w{self.worker_idx}_{cache_hash}.pt"
            cache_path = cache_dir / cache_name
            tmp_cache_path = cache_dir / f"{cache_name}.tmp_{os.getpid()}"

            if not cache_path.exists():
                try:
                    print(
                        f"[Worker {self.worker_idx}] Building dataset metadata cache...",
                        flush=True,
                    )
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(2, os.cpu_count() or 1)
                    ) as executor:
                        futures = [
                            executor.submit(_index_shard, (shard_idx, shard_path))
                            for shard_idx, shard_path in enumerate(self.shard_files)
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            local_metas, local_counts = future.result()
                            temp_metadata.extend(local_metas)
                            for k, v in local_counts.items():
                                class_counts[k] += v

                    if temp_metadata:
                        torch.save(
                            {
                                "temp_metadata": temp_metadata,
                                "class_counts": dict(class_counts),
                            },
                            tmp_cache_path,
                        )
                        os.replace(tmp_cache_path, cache_path)
                        print(
                            f"[Worker {self.worker_idx}] Atomically saved metadata cache ({len(temp_metadata)} records) to {cache_path}.",
                            flush=True,
                        )
                except Exception as e:
                    print(
                        f"[Worker {self.worker_idx}] Failed to save metadata cache: {e}",
                        flush=True,
                    )
                    if tmp_cache_path.exists():
                        try:
                            tmp_cache_path.unlink(missing_ok=True)
                        except Exception:
                            pass

            if cache_path.exists() and not temp_metadata:
                try:
                    cached = torch.load(
                        cache_path, map_location="cpu", weights_only=False
                    )
                    temp_metadata = cached["temp_metadata"]
                    class_counts = defaultdict(int, cached["class_counts"])
                    del cached
                    import gc
                    gc.collect()
                    print(
                        f"[Worker {self.worker_idx}] Loaded {len(temp_metadata)} records from cache.",
                        flush=True,
                    )
                except Exception as e:
                    print(
                        f"[Worker {self.worker_idx}] Failed to load cache: {e}. Falling back to manual parse.",
                        flush=True,
                    )
                    if not temp_metadata:
                        from concurrent.futures import ThreadPoolExecutor

                        with ThreadPoolExecutor(max_workers=1) as executor:
                            results = executor.map(
                                _index_shard, enumerate(self.shard_files)
                            )
                            for local_metas, local_counts in results:
                                temp_metadata.extend(local_metas)
                                for k, v in local_counts.items():
                                    class_counts[k] += v

            if self.worker_idx == 0:
                print(
                    f"[Worker {self.worker_idx}] Loaded {len(temp_metadata)} records. English vocabulary size: {len(self.english_vocab)}.",
                    flush=True,
                )

            if len(temp_metadata) == 0:
                raise RuntimeError(
                    f"[Worker {self.worker_idx}] DATASET INITIALIZATION FAILED: 0 valid records found in '{self.dataset_dir}'. Please check dataset_dir path and shard files."
                )

        records_by_shard = defaultdict(list)
        for r in temp_metadata:
            records_by_shard[r[0]].append(r)

        rng = random.Random(42 + getattr(self, "epoch", 0))
        shard_keys = list(records_by_shard.keys())
        if self.shuffle_shards:
            rng.shuffle(shard_keys)

        grouped_active = []
        shard_indices_map = defaultdict(list)
        for sk in shard_keys:
            s_recs = records_by_shard[sk]
            if self.shuffle_shards:
                rng.shuffle(s_recs)
            for rec in s_recs:
                shard_indices_map[sk].append(len(grouped_active))
                grouped_active.append(rec)

        shard_groups_list = [
            np.array(indices, dtype=np.int32)
            for indices in shard_indices_map.values()
            if indices
        ]

        self.class_counts = class_counts
        self.valid_label_ids = set(int(v) for v in self.label_to_idx.values())
        _GLOBAL_RECORDS_CACHE[cache_key] = temp_metadata
        _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = grouped_active
        _GLOBAL_SHARD_GROUPS_CACHE[cache_key] = shard_groups_list

    def set_noise_level(self, level: float) -> None:
        """Dynamically adjusts augmentation noise level and active Curriculum by Difficulty subset."""
        # Store directly on the instance so persistent_workers can read it even when
        # shared_progress is None (fixes disconnected curriculum bug).
        self._noise_level = float(level)
        if self.shared_progress is not None:
            self.shared_progress.value = float(level)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

        # Reshuffle shards block-by-block dynamically per epoch to prevent LRU RAM OoM
        # while keeping the data sufficiently randomized for training.
        if self.dataset_id in _GLOBAL_RECORDS_CACHE and self.shuffle_shards:
            import random

            rng = random.Random(epoch)

            temp_metadata = _GLOBAL_RECORDS_CACHE[self.dataset_id]
            records_by_shard = defaultdict(list)
            for r in temp_metadata:
                records_by_shard[r[0]].append(r)

            shard_keys = list(records_by_shard.keys())
            rng.shuffle(shard_keys)

            grouped_active = []
            shard_indices_map = defaultdict(list)
            for sk in shard_keys:
                s_recs = records_by_shard[sk]
                rng.shuffle(s_recs)
                for rec in s_recs:
                    shard_indices_map[sk].append(len(grouped_active))
                    grouped_active.append(rec)

            _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id] = grouped_active
            
            shard_groups_list = [
                np.array(indices, dtype=np.int32)
                for indices in shard_indices_map.values()
                if indices
            ]
            _GLOBAL_SHARD_GROUPS_CACHE[self.dataset_id] = shard_groups_list

    def __len__(self) -> int:
        if self.dataset_id not in _GLOBAL_RECORDS_CACHE:
            self._load_records_metadata()
        return len(_GLOBAL_RECORDS_CACHE[self.dataset_id])

    def _get_record_feature(self, shard_path: Path, item_key: Any) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        str_path = str(shard_path)

        # Multiprocessing RAM OoM Fix: Do not cache shards globally!
        # Read the shard directly using an LRU worker-local cache of size 1.
        if not hasattr(self, "_worker_shard_cache"):
            self._worker_shard_cache = {}
            self._worker_last_shard = None

        if self._worker_last_shard != str_path:
            old_shard = self._worker_shard_cache.pop(self._worker_last_shard, None)
            if old_shard is not None:
                del old_shard
            self._worker_shard_cache.clear()
            import gc

            gc.collect()

            self._worker_shard_cache[str_path] = torch.load(
                shard_path, map_location="cpu", weights_only=False, mmap=True
            )

            self._worker_last_shard = str_path

        shard_data = self._worker_shard_cache[str_path]

        if isinstance(shard_data, dict):
            rec = shard_data.get(item_key, None)
            raw_feat = (
                rec.get("features", rec.get("feature_array", rec))
                if isinstance(rec, dict)
                else rec
            )
        elif isinstance(shard_data, (list, tuple)):
            rec = shard_data[int(item_key)]
            raw_feat = (
                rec.get("features", rec.get("feature_array", rec))
                if isinstance(rec, dict)
                else rec
            )
        else:
            raw_feat = shard_data

        if raw_feat is None:
            raise ValueError(f"Feature '{item_key}' missing from shard '{shard_path}'.")

        frame_indices = None
        if isinstance(shard_data, dict):
            raw_rec = shard_data.get(item_key, None)
            if isinstance(raw_rec, dict):
                frame_indices = raw_rec.get("frame_index", None)
        elif isinstance(shard_data, (list, tuple)):
            rec = shard_data[int(item_key)]
            if isinstance(rec, dict):
                frame_indices = rec.get("frame_index", None)

        if isinstance(raw_feat, torch.Tensor):
            raw_feat = raw_feat.detach().cpu().numpy().copy()
        elif isinstance(raw_feat, np.ndarray):
            pass
        else:
            raw_feat = np.asarray(raw_feat, dtype=np.float32)

        return raw_feat, frame_indices

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.dataset_id not in _GLOBAL_RECORDS_CACHE:
            self._load_records_metadata()

        full_records = _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id]

        # We rely on set_epoch() dynamically reshuffling the _GLOBAL_ACTIVE_RECORDS_CACHE
        # block-by-block to guarantee shard locality and prevent RAM OOM, so we DO NOT permute globally here.

        if not (0 <= idx < len(full_records)):
            raise IndexError(
                f"Index {idx} out of bounds for dataset of size {len(full_records)}"
            )

        meta = full_records[idx]
        raw_label_str = ""
        if len(meta) >= 9:
            (
                shard_idx,
                feature_key,
                item_idx,
                label_idx,
                sample_weight,
                token_ids,
                task_id,
                source_id,
                raw_label_str,
            ) = meta[:9]
        elif len(meta) == 8:
            (
                shard_idx,
                feature_key,
                item_idx,
                label_idx,
                sample_weight,
                token_ids,
                task_id,
                source_id,
            ) = meta
        else:
            (
                shard_idx,
                feature_key,
                item_idx,
                label_idx,
                sample_weight,
                token_ids,
            ) = meta
            task_id = 0
            source_id = 0

        if task_id == TASK_FINGERSPELLING:
            task_str = "fingerspelling_sequence"
        elif task_id == TASK_SENTENCE:
            task_str = "sentence_level"
        elif task_id == TASK_ISOLATED:
            task_str = "isolated_gloss"
        else:
            raise ValueError(f"Unknown task_id: {task_id}")
            
        if source_id == 1:
            source_str = "ChicagoFSWild"
        elif source_id == 2:
            source_str = "How2Sign"
        elif source_id == 3:
            source_str = "ASLCitizen"
        elif source_id == 0:
            source_str = "unknown"
        else:
            raise ValueError(f"Unknown source_id: {source_id}")

        item_key = (
            feature_key
            if feature_key is not None
            else (item_idx if item_idx is not None else idx)
        )
        shard_path = self.shard_files[shard_idx]
        feat_arr, frame_indices_orig = self._get_record_feature(shard_path, item_key)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

        if feat_arr.ndim == 2:
            T, D = feat_arr.shape
            if D >= self.feature_dim:
                feat_arr = feat_arr[:, : self.feature_dim].reshape(
                    (T, self.num_keypoints, self.channels_per_kp)
                )
            else:
                pad_d = np.zeros((T, self.feature_dim - D), dtype=np.float32)
                feat_arr = np.concatenate([feat_arr, pad_d], axis=1).reshape(
                    (T, self.num_keypoints, self.channels_per_kp)
                )
            T = feat_arr.shape[0]
        elif feat_arr.ndim == 3:
            T, K, C = feat_arr.shape
            assert (
                K == self.num_keypoints
            ), f"Expected {self.num_keypoints} keypoints, got {K} in {shard_path}"
            if C < self.channels_per_kp:
                pad_c = np.zeros((T, K, self.channels_per_kp - C), dtype=np.float32)
                feat_arr = np.concatenate([feat_arr, pad_c], axis=-1)
            elif C > self.channels_per_kp:
                feat_arr = feat_arr[:, :, : self.channels_per_kp]
            T = feat_arr.shape[0]
        else:
            T = 0
            feat_arr = np.zeros(
                (0, self.num_keypoints, self.channels_per_kp), dtype=np.float32
            )

        # Real-World Camera Noise Data Augmentation during training
        # Enforce static sequence length (max_len) with Motion-Aware Priority Sampling FIRST
        features = np.zeros(
            (self.max_len, self.num_keypoints, self.channels_per_kp), dtype=np.float32
        )
        mask = np.zeros((self.max_len,), dtype=bool)
        # Use full_records[idx] (which corresponds to meta)
        # Note: 'rec' was implicitly available from meta logic; we re-retrieve
        # frame indices from the record metadata if available in shards
        # For simplicity here, we assume standard record structure.

        if frame_indices_orig is not None and len(frame_indices_orig) == T:
            frame_indices = np.asarray(frame_indices_orig, dtype=np.float32)
        else:
            frame_indices = np.arange(T, dtype=np.float32)

        if T > 0:
            if getattr(self, "stride_length", 1) > 1:
                feat_arr = feat_arr[::self.stride_length]
                frame_indices = frame_indices[::self.stride_length]
                T = feat_arr.shape[0]
            if T > self.max_len:
                idx = motion_aware_sample_indices(feat_arr, self.max_len)
                feat_arr = feat_arr[idx]
                frame_indices = frame_indices[idx]
            T = feat_arr.shape[0]

        if (
            getattr(self, "augment", False)
            and getattr(self, "augmenter", None) is not None
            and T > 0
        ):
            level = (
                self.shared_progress.value
                if self.shared_progress is not None
                else getattr(self, "_noise_level", 0.0)
            )
            aug_res = self.augmenter(
                feat_arr, noise_level=level, frame_indices=frame_indices[:T]
            )
            if isinstance(aug_res, tuple):
                feat_arr, aug_indices = aug_res
                if aug_indices is not None and len(aug_indices) > 0:
                    frame_indices = aug_indices
            else:
                feat_arr = aug_res
            if feat_arr.shape[-1] > self.channels_per_kp:
                feat_arr = feat_arr[..., : self.channels_per_kp]
            elif feat_arr.shape[-1] < self.channels_per_kp:
                pad_c = np.zeros(
                    (
                        feat_arr.shape[0],
                        feat_arr.shape[1],
                        self.channels_per_kp - feat_arr.shape[-1],
                    ),
                    dtype=np.float32,
                )
                feat_arr = np.concatenate([feat_arr, pad_c], axis=-1)
            T = feat_arr.shape[0]
        else:
            if T > 1 and feat_arr.shape[-1] >= 9:
                pos = feat_arr[:, :, :3]
                actual_dt = (frame_indices[1:] - frame_indices[:-1]).reshape(-1, 1, 1)
                actual_dt[actual_dt == 0] = 1.0
                feat_arr[1:, :, 3:6] = (pos[1:] - pos[:-1]) / actual_dt
                feat_arr[0, :, 3:6] = feat_arr[1, :, 3:6]
                feat_arr[1:, :, 6:9] = (feat_arr[1:, :, 3:6] - feat_arr[:-1, :, 3:6]) / actual_dt
                feat_arr[0, :, 6:9] = feat_arr[1, :, 6:9]
        
        padded_frame_indices = np.arange(self.max_len, dtype=np.float32)
        if T > 0:
            if T > self.max_len:
                idx_aug = motion_aware_sample_indices(feat_arr, self.max_len)
                feat_arr = feat_arr[idx_aug]
                if len(frame_indices) == T:
                    frame_indices = frame_indices[idx_aug]
                T = feat_arr.shape[0]

            if feat_arr.shape[-1] >= 9:
                # Recompute kinematics to ensure they are not stale after subsampling
                vel = np.zeros_like(feat_arr[..., :3])
                acc = np.zeros_like(feat_arr[..., :3])
                if feat_arr.shape[0] > 1:
                    dt = np.maximum(frame_indices[1:] - frame_indices[:-1], 1.0).astype(np.float32).reshape(-1, 1, 1)
                    vel[1:] = (feat_arr[1:, :, :3] - feat_arr[:-1, :, :3]) / dt
                    vel[0] = vel[1]
                    acc[1:] = (vel[1:] - vel[:-1]) / dt
                    acc[0] = acc[1]
                feat_arr[..., 3:6] = vel
                feat_arr[..., 6:9] = acc
                
            T_cap = min(T, self.max_len)
            features[:T_cap] = feat_arr[:T_cap]
            mask[:T_cap] = True
            padded_frame_indices[:T_cap] = frame_indices[:T_cap]

        label_idx = int(label_idx)
        raw_label_str = str(raw_label_str).strip().lower()
        sample_weight = float(sample_weight)
        import math

        if not math.isfinite(sample_weight) or sample_weight < 0:
            sample_weight = 0.0  # Safe fallback for invalid weights

        # Resolve ASL-LEX Grammatical Class (Only meaningful for glosses)
        lbl_str = re.sub(r"[^a-z0-9]", "", raw_label_str.lower())
        lbl_str = re.sub(r"\d+$", "", lbl_str)
        lex_class_idx = self.asl_lex_map.get(lbl_str, 4)

        # Common IDs
        _, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
        GLOSS_OFFSET = 4
        CHICAGO_OFFSET = 5

        MAX_GLOSS_LEN = min(self.max_len, 256)
        MAX_CHICAGO_LEN = min(self.max_len, 256)
        MAX_ENGLISH_LEN = min(self.max_len, 256)

        # Initialize defaults
        has_valid_gloss = False
        has_valid_chicago = False
        has_valid_english = False
        is_isolated = False

        ENG_BOS_ID = (
            getattr(self.english_vocab, "BOS_ID", 1)
            if hasattr(self, "english_vocab")
            else 1
        )
        ENG_EOS_ID = (
            getattr(self.english_vocab, "EOS_ID", 2)
            if hasattr(self, "english_vocab")
            else 2
        )

        raw_gloss_seq = [BOS_ID, EOS_ID]
        raw_chicago_seq = [BOS_ID, EOS_ID]
        raw_english_seq = [ENG_BOS_ID, ENG_EOS_ID]

        # 1. Routing based on source/task
        isolated_tasks = ("isolated_gloss", "static_alphabet", "isolated_number")

        if task_str in isolated_tasks:
            if raw_label_str in _SKIP_LABELS:
                has_valid_gloss = False
            else:
                has_valid_gloss = True
                is_isolated = True
                if not token_ids:
                    token_ids = [label_idx]
                max_valid_id = max(self.label_to_idx.values()) if self.label_to_idx else 30000
                if token_ids:
                    if np.max(token_ids) > max_valid_id:
                        raise ValueError(f"Token ID out of bounds. Max found {np.max(token_ids)}, max valid {max_valid_id}")
                raw_gloss_seq = (
                    [BOS_ID]
                    + np.where((np.array(token_ids) < 0) | (np.array(token_ids) == UNK_ID), UNK_ID, np.array(token_ids) + GLOSS_OFFSET).tolist()
                    + [EOS_ID]
                )

        elif task_str == "fingerspelling_sequence" or source_id == 1:
            if raw_label_str in _SKIP_LABELS:
                has_valid_chicago = False
            else:
                has_valid_chicago = True
            is_isolated = False
            # Tokenize chicago string: PAD=0, BOS=1, EOS=2, UNK=3, SP=4, a-z=5-30, 0-9=31-40
            SP_ID = 4
            raw_chicago_seq = [BOS_ID]
            clean_chicago_str = re.sub(
                r"[^a-z0-9\s]", "", raw_label_str.replace("<sp>", " ")
            )
            for c in clean_chicago_str:
                oc = ord(c)
                if oc == 32:
                    raw_chicago_seq.append(SP_ID)
                elif 97 <= oc <= 122:
                    raw_chicago_seq.append(oc - 97 + CHICAGO_OFFSET)
                elif 48 <= oc <= 57:
                    raw_chicago_seq.append(oc - 48 + 26 + CHICAGO_OFFSET)
                else:
                    raw_chicago_seq.append(UNK_ID)
            raw_chicago_seq.append(EOS_ID)

        elif task_str == "sentence_level" or source_id == 2:
            # How2Sign English Sentence
            is_isolated = False
            if raw_label_str != "how2sign_sequence" and len(raw_label_str) > 0:
                has_valid_english = True
                enc_ids = self.english_vocab.encode(raw_label_str, allow_unk=True)
                raw_english_seq = [ENG_BOS_ID] + enc_ids + [ENG_EOS_ID]
            else:
                has_valid_english = False  # Unrecoverable sentence

            if token_ids:
                has_valid_gloss = True
                max_valid_id = max(self.label_to_idx.values()) if self.label_to_idx else 30000
                if np.max(token_ids) > max_valid_id:
                    raise ValueError(f"Token ID out of bounds. Max found {np.max(token_ids)}, max valid {max_valid_id}")
                raw_gloss_seq = (
                    [BOS_ID]
                    + np.where((np.array(token_ids) < 0) | (np.array(token_ids) == UNK_ID), UNK_ID, np.array(token_ids) + GLOSS_OFFSET).tolist()
                    + [EOS_ID]
                )

        else:
            # Fallback for completely unknown tasks, treat as gloss sequence if token_ids exist
            if token_ids and label_idx != -1:
                has_valid_gloss = True
                is_isolated = len(token_ids) <= 1
                max_valid_id = max(self.label_to_idx.values()) if self.label_to_idx else 30000
                if np.max(token_ids) > max_valid_id:
                    raise ValueError(f"Token ID out of bounds. Max found {np.max(token_ids)}, max valid {max_valid_id}")
                raw_gloss_seq = (
                    [BOS_ID]
                    + np.where((np.array(token_ids) < 0) | (np.array(token_ids) == UNK_ID), UNK_ID, np.array(token_ids) + GLOSS_OFFSET).tolist()
                    + [EOS_ID]
                )

        # Pad sequences
        def pad_seq(raw_seq, max_len, pad_id=0):
            actual_len = min(len(raw_seq), max_len)
            is_truncated = len(raw_seq) > max_len
            padded = np.full(max_len, pad_id, dtype=np.int64)
            if is_truncated and actual_len > 0:
                padded[:actual_len - 1] = raw_seq[:actual_len - 1]
                padded[actual_len - 1] = raw_seq[-1]
            else:
                padded[:actual_len] = raw_seq[:actual_len]
            return padded, actual_len, is_truncated

        padded_gloss_seq, gloss_len, gloss_trunc = pad_seq(raw_gloss_seq, MAX_GLOSS_LEN)
        padded_chicago_seq, chicago_len, chicago_trunc = pad_seq(
            raw_chicago_seq, MAX_CHICAGO_LEN
        )
        # Use dynamic pad_id from the english vocabulary (e.g. 0 for Qwen)
        eng_pad_id = (
            getattr(self.english_vocab, "PAD_ID", 0)
            if hasattr(self, "english_vocab")
            else 0
        )
        padded_english_seq, english_len, english_trunc = pad_seq(
            raw_english_seq, MAX_ENGLISH_LEN, pad_id=eng_pad_id
        )


        # Source routing pseudo-IDs: 0 = Unknown/Default, 1 = ChicagoFSWild, 2 = How2Sign, 3 = ASLCitizen
        source_id = 0
        if "chicago" in source_str:
            source_id = 1
        elif "how2sign" in source_str:
            source_id = 2
        elif "citizen" in source_str:
            source_id = 3

        return {
            "feature": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "lex_class_idx": torch.tensor(lex_class_idx, dtype=torch.long),
            "domain_label": torch.tensor(source_id, dtype=torch.long),
            "has_domain_label": torch.tensor(source_id > 0, dtype=torch.bool),
            "frame_indices": torch.from_numpy(padded_frame_indices).float(),
            "gloss_seq": torch.tensor(padded_gloss_seq, dtype=torch.long),
            "gloss_len": torch.tensor(gloss_len, dtype=torch.long),
            "has_valid_gloss": torch.tensor(has_valid_gloss, dtype=torch.bool),
            "chicago_seq": torch.tensor(padded_chicago_seq, dtype=torch.long),
            "chicago_len": torch.tensor(chicago_len, dtype=torch.long),
            "has_valid_chicago": torch.tensor(has_valid_chicago, dtype=torch.bool),
            "english_seq": torch.tensor(padded_english_seq, dtype=torch.long),
            "english_len": torch.tensor(english_len, dtype=torch.long),
            "has_valid_english": torch.tensor(has_valid_english, dtype=torch.bool),
            "gloss_trunc": torch.tensor(gloss_trunc, dtype=torch.bool),
            "chicago_trunc": torch.tensor(chicago_trunc, dtype=torch.bool),
            "english_trunc": torch.tensor(english_trunc, dtype=torch.bool),
            "mlm_mask": (torch.rand(mask.shape[0]) < 0.15) & torch.from_numpy(mask),
            "is_isolated": torch.tensor(is_isolated, dtype=torch.bool),
            "source_id": torch.tensor(source_id, dtype=torch.long),
        }


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

    # CRITICAL FIX: Ensure worker threads do not explode when dataloader fetches a batch
    import os

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


class ShardPreservingSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, shuffle=True, seed=0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        import random

        dataset_id = self.dataset.dataset_id
        if dataset_id not in _GLOBAL_SHARD_GROUPS_CACHE:
            self.dataset._load_records_metadata()

        shard_groups = _GLOBAL_SHARD_GROUPS_CACHE[dataset_id]

        blocks = []
        for sg in shard_groups:
            blocks.append(sg.tolist())

        if self.shuffle:
            rng = random.Random(self.seed + self.epoch)
            np_rng = np.random.default_rng(self.seed + self.epoch)

            rng.shuffle(blocks)

            active_indices = []
            for b in blocks:
                b_copy = np.array(b, dtype=np.int32)
                np_rng.shuffle(b_copy)
                active_indices.append(b_copy)

            if active_indices:
                indices = np.concatenate(active_indices)
            else:
                indices = np.array([], dtype=np.int32)
        else:
            indices = (
                np.concatenate([np.array(b, dtype=np.int32) for b in blocks])
                if blocks
                else np.array([], dtype=np.int32)
            )

        dataset_len = len(self)
        if len(indices) == 0:
            return iter([])

        def gen():
            count = 0
            while count < dataset_len:
                for idx in indices:
                    yield int(idx)
                    count += 1
                    if count >= dataset_len:
                        return

        return gen()

    def __len__(self):
        return len(self.dataset)


class ASLStreamedDataset(IterableDataset):
    r"""
    Zero-RAM Streamed IterableDataset for distributed PyTorch XLA execution on massive sharded databases.

    Architecture:
    Let the entire dataset $\\mathcal{D}$ be partitioned into $S$ distinct non-overlapping shards $\{\\mathcal{S}_1, \dots, \\mathcal{S}_S\}$.
    In an $N$-worker cluster environment (e.g. TPU pods with 8-32 processes), each worker $w_i \\in \{0, \dots, N-1\}$
    will exclusively stream a subset of shards where `shard_idx % N == i`.

    Buffer Dynamics:
    To maintain temporal randomness without loading the entire $O(10^9)$ elements into host RAM,
    we maintain a worker-local shuffle buffer $\\mathcal{B}$ of size $K = 4096$. Elements are continuously
    yielded by selecting uniformly from $\\mathcal{B}$, and the empty slot is populated by the next
    incoming record from the IO-stream.

    This guarantees bounded memory utilization $M = O(K)$ and zero upfront serialization cost.
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path] = r"E:\datasets\results\asl_preprocessed_phase1",
        split: str = "train",
        max_len: int = 256,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        worker_idx: int = 0,
        num_workers: int = 1,
        shuffle_buffer_size: int = 64,
        stage: str = "full_mixture",
        augment: bool = False,
        shared_progress=None,
        shared_epoch=None,
        use_bpe: bool = False,
        model_name: str = "Qwen/Qwen2.5-0.5B",
    ):
        super().__init__()
        input_dir = Path(dataset_dir)
        if not input_dir.exists():
            candidates = [
                Path(
                    "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"
                ),
                Path(
                    "/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"
                ),
                Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl"),
                Path("/kaggle/input/asl-preprocessed-phase1"),
                Path("./asl_preprocessed_phase1"),
            ]
            candidates = [c for c in candidates if os.name != "nt" or not str(c).startswith("/kaggle/")]
            input_dir = next((c for c in candidates if c.exists()), input_dir)

        self.dataset_dir = (
            input_dir / split if (input_dir / split).exists() else input_dir
        )
        self.split = split
        self.max_len = max_len
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.worker_idx = worker_idx
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment = augment and (split == "train")
        self.shared_progress = shared_progress
        self.shared_epoch = shared_epoch
        # to prevent augmentation being applied on validation data
        self.augmenter = (
            LandmarkAugmenter(max_len=self.max_len) if self.augment else None
        )
        self.use_bpe = use_bpe
        self.model_name = model_name

        english_vocab_file = self.dataset_dir / "english_vocab.json"
        if not english_vocab_file.exists():
            english_vocab_file = input_dir / "english_vocab.json"
        if english_vocab_file.exists():
            self.english_vocab = EnglishVocabulary(
                vocab_path=english_vocab_file
            )
        else:
            self.english_vocab = EnglishVocabulary.__new__(EnglishVocabulary)
            self.english_vocab.token_to_id = {}
            self.english_vocab.id_to_token = {}
            self.english_vocab.PAD_ID = 0
            self.english_vocab.BOS_ID = 1
            self.english_vocab.EOS_ID = 2
            self.english_vocab.UNK_ID = 3
            self.english_vocab.encode = lambda text, allow_unk=True: []
            self.english_vocab.decode = lambda ids: ""
            self.english_vocab.is_valid = False

        all_shard_files = sorted(list(self.dataset_dir.glob("shard_*.pt")))
        if not all_shard_files:
            all_shard_files = sorted(list(self.dataset_dir.glob("*.pt")))
        # from processing every shard (duplicated writes & deadlock).
        _n_workers = self.num_workers if self.num_workers > 0 else 1
        self.shard_files = all_shard_files[self.worker_idx :: _n_workers]
        if not self.shard_files and all_shard_files:
            # Fallback to cyclic shard assignment to prevent worker starvation (Claim 42 Fix)
            self.shard_files = [all_shard_files[self.worker_idx % len(all_shard_files)]]

        total_records = 0
        manifest_path = self.dataset_dir / "manifest.json"
        if manifest_path.exists():
            try:
                import json

                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for s in self.shard_files:
                    total_records += manifest.get(s.name, 0)
            except Exception:
                pass

        if total_records == 0 and self.shard_files:
            self.total_records = None
        else:
            self.total_records = total_records
        self.class_counts = {}
        class_counts_path = self.dataset_dir / "class_counts.json"
        if class_counts_path.exists():
            try:
                with open(class_counts_path, "r", encoding="utf-8") as f:
                    self.class_counts = {int(k): v for k, v in json.load(f).items()}
            except Exception:
                pass
        else:
            for cache_file in self.dataset_dir.glob("metadata_cache_*.pt"):
                try:
                    cached = torch.load(cache_file, map_location="cpu", weights_only=False)
                    if "class_counts" in cached:
                        self.class_counts = dict(cached["class_counts"])
                        break
                except Exception:
                    pass

    def __len__(self) -> int:
        if getattr(self, "total_records", None) is None:
            raise TypeError("Length of this streamed dataset is unknown")
        return self.total_records
        
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.shared_epoch is not None:
            try:
                self.shared_epoch.value = int(epoch)
            except Exception:
                pass

    def set_noise_level(self, level: float) -> None:
        """Dynamically adjusts augmentation noise level for the dataset stream."""
        self._noise_level = float(level)
        if self.shared_progress is not None:
            self.shared_progress.value = float(level)

    def __call__(self, epoch: int):
        self.set_epoch(epoch)

    def __iter__(self):
        import random

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shards = self.shard_files[worker_info.id :: worker_info.num_workers]
        else:
            shards = list(self.shard_files)

        cur_epoch = (
            self.shared_epoch.value
            if self.shared_epoch is not None
            else getattr(self, "epoch", 0)
        )

        if self.split == "train":
            worker_seed = (
                worker_info.seed + cur_epoch
                if worker_info
                else 42 + cur_epoch + (self.worker_idx * 100)
            )
            random.seed(worker_seed)
            np.random.seed(worker_seed % (2**32 - 1))
            random.shuffle(shards)

        records_yielded = 0
        last_exception = None

        import concurrent.futures

        def load_shard(path):
            try:
                data = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
                if isinstance(data, dict):
                    keys = list(data.keys())
                    if self.split == "train":
                        random.shuffle(keys)
                    return [data[k] for k in keys]
                return data
            except Exception:
                try:
                    str_path = str(Path(path).absolute())
                    data = torch.load(str_path, map_location="cpu", weights_only=False)
                    if isinstance(data, dict):
                        keys = list(data.keys())
                        if self.split == "train":
                            random.shuffle(keys)
                        return [data[k] for k in keys]
                    return data
                except Exception as final_err:
                    return final_err

        buffer = []
        prev_raw_rec = None
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future_to_path = {executor.submit(load_shard, shards[0]): shards[0]} if shards else {}
            
            for i in range(len(shards)):
                shard_path = shards[i]
                future = list(future_to_path.keys())[0]
                del future_to_path[future]
                
                if i + 1 < len(shards):
                    next_path = shards[i + 1]
                    future_to_path[executor.submit(load_shard, next_path)] = next_path
                
                items = future.result()
                if isinstance(items, Exception):
                    last_exception = items
                    print(f"Failed to load shard {shard_path}: {items}")
                    raise items
                
                for rec in items:
                    if isinstance(rec, dict):
                        rec_to_process = rec
                        if self.augment and random.random() < 0.25:
                            if prev_raw_rec is not None:
                                stitched = self._stitch_raw_records(prev_raw_rec, rec)
                                prev_raw_rec = None
                                if stitched is not None:
                                    rec_to_process = stitched
                            else:
                                prev_raw_rec = rec
                                continue  # Do not yield prev_raw_rec individually if saved for stitching

                        processed = self._process_record(shard_path, rec_to_process)
                        if processed is not None:
                            if len(buffer) >= self.shuffle_buffer_size:
                                # True reservoir sliding shuffle replacement
                                idx = random.randint(0, len(buffer) - 1)
                                yield buffer[idx]
                                buffer[idx] = processed
                                records_yielded += 1
                            else:
                                buffer.append(processed)
                del items

        if buffer:
            if self.split == "train":
                random.shuffle(buffer)
            for processed in buffer:
                if processed is not None:
                    yield processed
                    records_yielded += 1
            buffer.clear()
        if records_yielded == 0:
            raise RuntimeError(
                f"ASLStreamedDataset fatal error: yielded zero records. "
                f"Last error: {last_exception}"
            )

    def _stitch_raw_records(self, rec1: dict, rec2: dict) -> Optional[dict]:
        """Stitches two raw records with an inserted idle pause to simulate continuous multi-sentence video streams."""
        try:
            feat1 = rec1.get("features", rec1.get("feature_array"))
            feat2 = rec2.get("features", rec2.get("feature_array"))
            if feat1 is None or feat2 is None:
                return None

            if isinstance(feat1, torch.Tensor):
                feat1 = feat1.detach().cpu().numpy()
            else:
                feat1 = np.asarray(feat1, dtype=np.float32)
            if isinstance(feat2, torch.Tensor):
                feat2 = feat2.detach().cpu().numpy()
            else:
                feat2 = np.asarray(feat2, dtype=np.float32)

            T1 = feat1.shape[0] if feat1.ndim >= 2 else 0
            T2 = feat2.shape[0] if feat2.ndim >= 2 else 0
            if T1 == 0 or T2 == 0:
                return None

            pause_len = random.randint(8, 16)
            if feat1.ndim == 2:
                pause_feat = np.zeros((pause_len, feat1.shape[1]), dtype=np.float32)
            elif feat1.ndim == 3:
                pause_feat = np.zeros((pause_len, feat1.shape[1], feat1.shape[2]), dtype=np.float32)
            else:
                return None

            stitched_feat = np.concatenate([feat1, pause_feat, feat2], axis=0)

            t1_str = str(rec1.get("raw_label_str", rec1.get("text", rec1.get("label", "")))).strip()
            t2_str = str(rec2.get("raw_label_str", rec2.get("text", rec2.get("label", "")))).strip()

            if t1_str and not t1_str.endswith((".", "?", "!")):
                t1_str += "."
            if t2_str and not t2_str.endswith((".", "?", "!")):
                t2_str += "."

            stitched_text = f"{t1_str} {t2_str}".strip()

            g1 = rec1.get("gloss_seq", rec1.get("token_ids", []))
            g2 = rec2.get("gloss_seq", rec2.get("token_ids", []))
            if isinstance(g1, torch.Tensor):
                g1 = g1.tolist()
            if isinstance(g2, torch.Tensor):
                g2 = g2.tolist()

            PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
            g1_clean = [t for t in g1 if t not in (PAD_ID, BOS_ID, EOS_ID)] if g1 else []
            g2_clean = [t for t in g2 if t not in (PAD_ID, BOS_ID, EOS_ID)] if g2 else []
            stitched_gloss = ([BOS_ID] + g1_clean + g2_clean + [EOS_ID]) if (g1_clean or g2_clean) else None

            stitched_rec = dict(rec1)
            stitched_rec["features"] = stitched_feat
            stitched_rec["raw_label_str"] = stitched_text
            stitched_rec["text"] = stitched_text
            stitched_rec["label_idx"] = -1
            stitched_rec["source"] = f"{rec1.get('source', 'unknown')}_{rec2.get('source', 'unknown')}"
            if stitched_gloss:
                stitched_rec["gloss_seq"] = stitched_gloss

            c1, c2 = rec1.get("chicago_seq"), rec2.get("chicago_seq")
            if c1 is not None and c2 is not None:
                c1_list = c1.tolist() if isinstance(c1, torch.Tensor) else list(c1)
                c2_list = c2.tolist() if isinstance(c2, torch.Tensor) else list(c2)
                SP_ID = 4  # Space token in chicago tokenizer (PAD=0,BOS=1,EOS=2,UNK=3,SP=4)
                c1_clean = [t for t in c1_list if t not in (PAD_ID, BOS_ID, EOS_ID)]
                c2_clean = [t for t in c2_list if t not in (PAD_ID, BOS_ID, EOS_ID)]
                chicago_body = (c1_clean + [SP_ID] + c2_clean) if (c1_clean and c2_clean) else (c1_clean or c2_clean)
                stitched_rec["chicago_seq"] = [BOS_ID] + chicago_body + [EOS_ID]

            e1, e2 = rec1.get("english_seq"), rec2.get("english_seq")
            if e1 is not None and e2 is not None:
                e1_list = e1.tolist() if isinstance(e1, torch.Tensor) else list(e1)
                e2_list = e2.tolist() if isinstance(e2, torch.Tensor) else list(e2)
                e1_clean = [t for t in e1_list if t not in (PAD_ID, BOS_ID, EOS_ID)]
                e2_clean = [t for t in e2_list if t not in (PAD_ID, BOS_ID, EOS_ID)]
                stitched_rec["english_seq"] = [BOS_ID] + e1_clean + e2_clean + [EOS_ID]

            f1, f2 = rec1.get("frame_index"), rec2.get("frame_index")
            if f1 is None:
                f1 = np.arange(T1, dtype=np.float32)
            else:
                f1 = f1.cpu().numpy() if isinstance(f1, torch.Tensor) else np.asarray(f1, dtype=np.float32)
            if f2 is None:
                f2 = np.arange(T2, dtype=np.float32)
            else:
                f2 = f2.cpu().numpy() if isinstance(f2, torch.Tensor) else np.asarray(f2, dtype=np.float32)

            f_pause = np.arange(pause_len, dtype=np.float32) + (f1[-1] + 1 if len(f1) > 0 else 0)
            f2 = f2 - (f2[0] if len(f2) > 0 else 0) + (f_pause[-1] + 1 if len(f_pause) > 0 else 0)
            stitched_rec["frame_index"] = np.concatenate([f1, f_pause, f2])

            stitched_rec["task"] = "sentence_level"
            return stitched_rec
        except Exception:
            return None

    def _process_record(
        self, shard_path: Path, rec: dict
    ) -> Optional[Dict[str, torch.Tensor]]:
        feat_arr = rec.get("features", rec.get("feature_array"))
        if feat_arr is None:
            return None
        if isinstance(feat_arr, torch.Tensor):
            feat_arr = feat_arr.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            feat_arr = np.asarray(feat_arr, dtype=np.float32)

        np.nan_to_num(feat_arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        T = feat_arr.shape[0] if feat_arr.ndim >= 2 else 0

        features = np.zeros(
            (self.max_len, self.num_keypoints, self.channels_per_kp), dtype=np.float16
        )
        mask = np.zeros((self.max_len,), dtype=bool)
        padded_frame_indices = np.arange(self.max_len, dtype=np.int64)

        raw_frame_indices = rec.get("frame_index", None)
        if raw_frame_indices is None:
            raw_frame_indices = np.arange(T, dtype=np.float32)
        else:
            if isinstance(raw_frame_indices, torch.Tensor):
                raw_frame_indices = raw_frame_indices.cpu().numpy()
            raw_frame_indices = np.asarray(raw_frame_indices, dtype=np.float32)
            if len(raw_frame_indices) != T:
                raise ValueError(f"Frame index length {len(raw_frame_indices)} != features length {T}")
            if T > 1 and not np.all(raw_frame_indices[1:] >= raw_frame_indices[:-1]):
                raise ValueError("Frame indices are not monotonically increasing")

        if self.augmenter is not None:
            level = (
                self.shared_progress.value
                if self.shared_progress is not None
                else getattr(self, "_noise_level", 0.0)
            )
            aug_res = self.augmenter(
                feat_arr, noise_level=level, frame_indices=raw_frame_indices
            )
            # Accessing aug_res["features"] raises TypeError: tuple indices must be integers.
            if isinstance(aug_res, dict):
                feat_arr = aug_res["features"]
                raw_frame_indices = aug_res.get("frame_indices", raw_frame_indices)
            else:
                feat_arr, raw_frame_indices = aug_res
            T = feat_arr.shape[0] if feat_arr.ndim >= 2 else 0

        if T > 0:
            if feat_arr.ndim == 2:
                T_raw, D = feat_arr.shape
                feat_dim = self.num_keypoints * self.channels_per_kp
                if D != feat_dim:
                    raise ValueError(f"Feature dimension {D} != expected {feat_dim}")
                feat_arr = feat_arr.reshape(
                    (T_raw, self.num_keypoints, self.channels_per_kp)
                )
            elif feat_arr.ndim == 3:
                T_raw, num_kp, C = feat_arr.shape
                if num_kp != self.num_keypoints or C != self.channels_per_kp:
                    raise ValueError(f"Feature shape {num_kp}x{C} != expected {self.num_keypoints}x{self.channels_per_kp}")

            if T > self.max_len:
                idx = motion_aware_sample_indices(feat_arr, self.max_len)
                feat_arr = feat_arr[idx]
                raw_frame_indices = raw_frame_indices[idx]
                padded_frame_indices[: self.max_len] = raw_frame_indices[: self.max_len]

                if feat_arr.shape[0] > 1 and feat_arr.shape[-1] >= 9:
                    actual_dt = (
                        (raw_frame_indices[1:] - raw_frame_indices[:-1])
                        .astype(np.float32)
                        .reshape(-1, 1, 1)
                    )
                    actual_dt[actual_dt == 0] = 1.0
                    vel = np.zeros_like(feat_arr[:, :, :3])
                    acc = np.zeros_like(feat_arr[:, :, :3])
                    pos = feat_arr[:, :, :3]
                    vel[1:] = (pos[1:] - pos[:-1]) / actual_dt
                    vel[0] = 0.0
                    acc[1:] = (vel[1:] - vel[:-1]) / actual_dt
                    acc[0] = 0.0
                    feat_arr[:, :, 3:6] = vel
                    feat_arr[:, :, 6:9] = acc

            T_cap = min(T, self.max_len)
            feat_slice = feat_arr[:T_cap]

            features[:T_cap] = feat_slice
            mask[:T_cap] = True
            if T <= self.max_len:
                padded_frame_indices[:T_cap] = raw_frame_indices[:T_cap]

        # Standardized Supervision Constants
        _, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
        GLOSS_OFFSET = 4
        CHICAGO_OFFSET = 5

        MAX_GLOSS_LEN = min(self.max_len, 256)
        MAX_CHICAGO_LEN = min(self.max_len, 256)
        MAX_ENGLISH_LEN = min(self.max_len, 256)

        raw_label_str = (
            str(rec.get("raw_label_str", rec.get("text", rec.get("label", ""))))
            .strip()
            .lower()
        )
        task_str = str(rec.get("task", rec.get("task_str", ""))).strip().lower()
        source_str = str(rec.get("source", "unknown")).strip().lower()

        source_id = 0
        if "chicago" in source_str:
            source_id = 1
        elif "how2sign" in source_str:
            source_id = 2
        elif "citizen" in source_str:
            source_id = 3
        else:
            source_id = 0

        label_idx = int(rec.get("label_idx", -1))
        raw_sw = rec.get("quality", rec.get("sample_weight", 1.0))
        try:
            sample_weight = float(raw_sw)
            if not math.isfinite(sample_weight) or sample_weight < 0.0:
                sample_weight = 0.0
        except (ValueError, TypeError):
            sample_weight = 0.0

        has_valid_gloss = False
        has_valid_chicago = False
        has_valid_english = False
        is_isolated = False

        ENG_BOS_ID = (
            getattr(self.english_vocab, "BOS_ID", 1)
            if hasattr(self, "english_vocab")
            else 1
        )
        ENG_EOS_ID = (
            getattr(self.english_vocab, "EOS_ID", 2)
            if hasattr(self, "english_vocab")
            else 2
        )

        raw_gloss_seq = [BOS_ID, EOS_ID]
        raw_chicago_seq = [BOS_ID, EOS_ID]
        raw_english_seq = [ENG_BOS_ID, ENG_EOS_ID]

        token_ids = rec.get("gloss_seq", rec.get("token_ids", None))
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        isolated_tasks = ("isolated_gloss", "static_alphabet", "isolated_number")

        if task_str in isolated_tasks or (label_idx != -1 and not task_str):
            if label_idx != -1 or token_ids:
                has_valid_gloss = True
                is_isolated = True
                if not token_ids and label_idx >= 0:
                    token_ids = [label_idx]
                if token_ids:
                    vocab_obj = getattr(self, "vocab", getattr(self, "gloss_vocab", None))
                    max_gloss_id = getattr(vocab_obj, "vocab_size", 999999) - 1
                    raw_gloss_seq = [BOS_ID] + [UNK_ID if (tid < 0 or tid == UNK_ID) else min(tid + GLOSS_OFFSET, max_gloss_id) for tid in token_ids] + [EOS_ID]

        elif task_str == "fingerspelling_sequence" or "chicago" in source_str:
            if raw_label_str and raw_label_str not in _SKIP_LABELS:
                has_valid_chicago = True
                is_isolated = False
                SP_ID = 4
                raw_chicago_seq = [BOS_ID]
                clean_chicago_str = re.sub(
                    r"[^a-z0-9\s]", "", raw_label_str.replace("<sp>", " ")
                )
                for c in clean_chicago_str:
                    if c == " ":
                        raw_chicago_seq.append(SP_ID)
                    elif "a" <= c <= "z":
                        raw_chicago_seq.append(ord(c) - ord("a") + CHICAGO_OFFSET)
                    elif "0" <= c <= "9":
                        raw_chicago_seq.append(ord(c) - ord("0") + 26 + CHICAGO_OFFSET)
                    else:
                        raw_chicago_seq.append(UNK_ID)
                raw_chicago_seq.append(EOS_ID)

        elif task_str == "sentence_level" or "how2sign" in source_str:
            is_isolated = False
            if raw_label_str and raw_label_str != "how2sign_sequence":
                enc_ids = self.english_vocab.encode(raw_label_str, allow_unk=True)
                has_valid_english = getattr(self.english_vocab, "is_valid", True) and len(enc_ids) > 0
                raw_english_seq = [ENG_BOS_ID] + enc_ids + [ENG_EOS_ID]
            if token_ids:
                has_valid_gloss = True
                max_gloss_id = getattr(self.vocab, "vocab_size", 999999) - 1
                raw_gloss_seq = [BOS_ID] + [UNK_ID if (tid < 0 or tid == UNK_ID) else min(tid + GLOSS_OFFSET, max_gloss_id) for tid in token_ids] + [EOS_ID]
        else:
            if token_ids and label_idx != -1:
                has_valid_gloss = True
                is_isolated = len(token_ids) <= 1
                max_gloss_id = getattr(self.vocab, "vocab_size", 999999) - 1
                raw_gloss_seq = [BOS_ID] + [UNK_ID if (tid < 0 or tid == UNK_ID) else min(tid + GLOSS_OFFSET, max_gloss_id) for tid in token_ids] + [EOS_ID]

        def pad_seq(raw_seq, max_len, pad_id=0):
            actual_len = min(len(raw_seq), max_len)
            is_truncated = len(raw_seq) > max_len
            padded = np.full(max_len, pad_id, dtype=np.int64)
            if is_truncated and actual_len > 0:
                padded[:actual_len - 1] = raw_seq[:actual_len - 1]
                padded[actual_len - 1] = raw_seq[-1]
            else:
                padded[:actual_len] = raw_seq[:actual_len]
            return padded, actual_len, is_truncated

        padded_gloss_seq, gloss_len, gloss_trunc = pad_seq(raw_gloss_seq, MAX_GLOSS_LEN)
        padded_chicago_seq, chicago_len, chicago_trunc = pad_seq(
            raw_chicago_seq, MAX_CHICAGO_LEN
        )
        eng_pad_id = (
            getattr(self.english_vocab, "PAD_ID", 0)
            if hasattr(self, "english_vocab")
            else 0
        )
        padded_english_seq, english_len, english_trunc = pad_seq(
            raw_english_seq, MAX_ENGLISH_LEN, pad_id=eng_pad_id
        )


        if feat_arr.ndim not in {2, 3} or T == 0:
            has_valid_gloss = False
            has_valid_chicago = False
            has_valid_english = False

        return {
            "feature": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label_idx, dtype=torch.int32),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "lex_class_idx": torch.tensor(4, dtype=torch.int32),
            "domain_label": torch.tensor(source_id, dtype=torch.int32),
            "has_domain_label": torch.tensor(source_id > 0, dtype=torch.bool),
            "frame_indices": torch.from_numpy(padded_frame_indices).float(),
            "gloss_seq": torch.tensor(padded_gloss_seq, dtype=torch.int32),
            "gloss_len": torch.tensor(gloss_len, dtype=torch.int32),
            "has_valid_gloss": torch.tensor(has_valid_gloss, dtype=torch.bool),
            "chicago_seq": torch.tensor(padded_chicago_seq, dtype=torch.int32),
            "chicago_len": torch.tensor(chicago_len, dtype=torch.int32),
            "has_valid_chicago": torch.tensor(has_valid_chicago, dtype=torch.bool),
            "english_seq": torch.tensor(padded_english_seq, dtype=torch.int32),
            "english_len": torch.tensor(english_len, dtype=torch.int32),
            "has_valid_english": torch.tensor(has_valid_english, dtype=torch.bool),
            "is_isolated": torch.tensor(is_isolated, dtype=torch.bool),
            "mlm_mask": (torch.rand(mask.shape[0]) < 0.15) & torch.from_numpy(mask),
            "english_trunc": torch.tensor(english_trunc, dtype=torch.bool),
        }


def create_dataloader(
    dataset_dir: Union[str, Path] = r"E:\datasets\asl_dataset\asl_preprocessed_phase1",
    split: str = "train",
    batch_size: int = 64,
    max_len: int = 256,
    worker_idx: int = 0,
    num_workers: int = 1,
    num_dataloader_workers: int = 0,
    shuffle: bool = True,
    stage: str = "full_mixture",
    augment: bool = False,
    streamed: bool = True,
    **kwargs,
) -> DataLoader:
    """Creates a PyTorch DataLoader wrapping ASLShardedDataset or ASLStreamedDataset."""
    shared_progress = None
    shared_epoch = None
    if num_dataloader_workers > 0:
        import torch.multiprocessing as mp

        try:
            mp_context = mp.get_context("fork")
        except ValueError:
            mp_context = mp.get_context()
        shared_progress = mp_context.Value("d", 0.0)
        shared_epoch = mp_context.Value("i", 0)

    if streamed:
        dataset = ASLStreamedDataset(
            dataset_dir=dataset_dir,
            split=split,
            max_len=max_len,
            worker_idx=worker_idx,
            num_workers=num_workers,
            shuffle_buffer_size=1000 if shuffle else 1,
            stage=stage,
            augment=augment,
            shared_progress=shared_progress,
            shared_epoch=shared_epoch,
            **kwargs,
        )
        sampler = None
    else:
        dataset = ASLShardedDataset(
            dataset_dir=dataset_dir,
            split=split,
            max_len=max_len,
            worker_idx=worker_idx,
            num_workers=num_workers,
            shuffle_shards=shuffle,
            stage=stage,
            augment=augment,
            shared_progress=shared_progress,
            shared_epoch=shared_epoch,
            **kwargs,
        )
        sampler = ShardPreservingSampler(dataset, shuffle=shuffle) if shuffle else None

    collate_fn = None

    dl_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "sampler": sampler,
        "num_workers": num_dataloader_workers,
        "pin_memory": False,
        "collate_fn": collate_fn,
        "drop_last": kwargs.get("drop_last", (split == "train" or kwargs.get("is_tpu", False))),
        "worker_init_fn": _seed_worker,
    }
    if num_dataloader_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = True
        if mp_context is not None:
            dl_kwargs["multiprocessing_context"] = mp_context

    return DataLoader(**dl_kwargs)


# ==============================================================================
# PHASE 1: TEXT PRE-TRAINING DATASETS (DAE & ASLG-PC12)
# ==============================================================================


def apply_dae_corruptions(
    tokens: list,
    unk_id: int,
    mask_prob: float = 0.15,
    drop_prob: float = 0.10,
    shuffle_prob: float = 0.10,
):
    """Applies Denoising Autoencoder (DAE) corruptions to a list of token IDs."""
    import random

    if len(tokens) <= 3:
        return tokens

    # 1. N-Gram Shuffling (local permutations)
    if random.random() < shuffle_prob:
        span_len = random.randint(2, 4)
        if len(tokens) > span_len:
            start_idx = random.randint(0, len(tokens) - span_len)
            span = tokens[start_idx : start_idx + span_len]
            random.shuffle(span)
            tokens = tokens[:start_idx] + span + tokens[start_idx + span_len :]

    # 2. Token Deletion & 3. Token Masking
    corrupted = []
    for t in tokens:
        if random.random() < drop_prob:
            continue
        if random.random() < mask_prob:
            corrupted.append(unk_id)
        else:
            corrupted.append(t)

    return corrupted if len(corrupted) > 0 else tokens


class KDWDDataset(torch.utils.data.IterableDataset):
    """Streams and filters the KDWD Wikipedia dataset for high-traffic articles."""

    def __init__(
        self,
        kdwd_dir: str,
        eng_vocab: EnglishVocabulary,
        max_len: int = 128,
        views_threshold: int = 5000,
    ):
        super().__init__()
        self.kdwd_dir = Path(kdwd_dir)
        self.eng_vocab = eng_vocab
        self.max_len = max_len
        self.views_threshold = views_threshold

        import os
        # Load and filter page_id by views
        self.valid_page_ids = set()
        page_csv = self.kdwd_dir / "page.csv"
        valid_ids_cache = Path("/dev/shm/valid_page_ids.pt") if os.path.exists("/dev/shm") else Path("./valid_page_ids.pt")
        
        try:
            import torch_xla.runtime as xr
            rank = xr.global_ordinal()
        except Exception:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))

        if valid_ids_cache.exists():
            try:
                self.valid_page_ids = torch.load(valid_ids_cache)
            except Exception:
                pass
        
        if not self.valid_page_ids and page_csv.exists():
            if rank == 0:
                import pandas as pd
                try:
                    df = pd.read_csv(page_csv, usecols=["page_id", "views"])
                    self.valid_page_ids = set(
                        df[df["views"] > self.views_threshold]["page_id"].tolist()
                    )
                    torch.save(self.valid_page_ids, valid_ids_cache)
                    del df
                    import gc
                    gc.collect()
                except Exception:
                    pass
            else:
                import time
                for _ in range(600):
                    if valid_ids_cache.exists():
                        try:
                            self.valid_page_ids = torch.load(valid_ids_cache)
                            break
                        except Exception:
                            pass
                    time.sleep(0.5)

        self.cached_tokens = []
        self._load_and_cache_tokens()

    def _load_and_cache_tokens(self, max_samples: int = 30000):
        jsonl_path = self.kdwd_dir / "link_annotated_text.jsonl"
        if not jsonl_path.exists():
            return

        import os
        import gc
        import time
        import json
        import re
        import hashlib

        cache_key_str = f"{jsonl_path}_{len(self.eng_vocab)}_{self.max_len}"
        cache_hash = hashlib.sha256(cache_key_str.encode('utf-8')).hexdigest()[:12]
        cache_fname = f"kdwd_cached_tokens_{cache_hash}.pt"

        # 1. Check if pre-cached file exists in input directory (read-only) or /dev/shm (writable RAM)
        input_cache_path = self.kdwd_dir / "kdwd_cached_tokens.pt"
        if input_cache_path.exists():
            shm_cache_path = input_cache_path
        elif os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
            shm_cache_path = Path(f"/dev/shm/{cache_fname}")
        else:
            shm_cache_path = Path(f"./{cache_fname}").absolute()

        # 1. If RAM cache already exists, load shared copy
        if shm_cache_path.exists():
            try:
                data = torch.load(shm_cache_path)
                self.cached_flat = data["flat"]
                self.cached_offsets = data["offsets"]
                return
            except Exception:
                pass

        rank = 0
        try:
            import torch_xla.runtime as xr
            if hasattr(xr, "is_runtime_initialized") and xr.is_runtime_initialized():
                rank = xr.global_ordinal()
            else:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        except Exception:
            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        if rank != 0:
            # Non-master ranks wait up to 300s (5 mins) for Rank 0 to finish writing RAM cache
            for _ in range(600):
                if shm_cache_path.exists():
                    try:
                        data = torch.load(shm_cache_path)
                        self.cached_flat = data["flat"]
                        self.cached_offsets = data["offsets"]
                        return
                    except Exception:
                        pass
                time.sleep(0.5)
            # CRITICAL: Non-master ranks MUST NEVER fall through to parse the 16GB JSONL file in parallel!
            return

        page_id_re = re.compile(r'"page_id":\s*(\d+)')
        print(f"[INFO] Rank {rank}: Pre-tokenizing KDWD dataset into compact int16 RAM disk ({shm_cache_path})...", flush=True)
        count = 0
        raw_token_sequences = []
        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if self.valid_page_ids:
                        match = page_id_re.search(line)
                        if match:
                            page_id = int(match.group(1))
                            if page_id not in self.valid_page_ids:
                                continue
                    try:
                        data = json.loads(line)
                        sections = data.get("sections", [])
                        for sec in sections:
                            text = sec.get("text", "").strip()
                            if len(text) < 20:
                                continue
                            text_ids = self.eng_vocab.encode(text)
                            if len(text_ids) >= 4:
                                raw_token_sequences.append(text_ids[: self.max_len - 2])
                                count += 1
                                if count >= max_samples:
                                    break
                    except Exception:
                        pass
                    if count >= max_samples:
                        break

            # Pack into ultra-compact 1D int16 flat tensor (takes ~3.5MB total RAM!)
            # Pack into ultra-compact 1D int32 flat tensor (takes ~7MB total RAM!)
            flat_ids = []
            offsets = [0]
            for seq in raw_token_sequences:
                flat_ids.extend(seq)
                offsets.append(len(flat_ids))
            
            self.cached_flat = torch.tensor(flat_ids, dtype=torch.int32)
            self.cached_offsets = torch.tensor(offsets, dtype=torch.int32)
            del raw_token_sequences, flat_ids, offsets
            gc.collect()

            print(f"[INFO] Successfully pre-tokenized {len(self.cached_offsets)-1} KDWD sentences into compact {self.cached_flat.element_size() * self.cached_flat.nelement() / (1024*1024):.2f}MB int32 RAM disk!", flush=True)
            if rank == 0:
                try:
                    tmp_cache_path = shm_cache_path.with_suffix(".tmp")
                    torch.save({"flat": self.cached_flat, "offsets": self.cached_offsets}, tmp_cache_path)
                    os.replace(tmp_cache_path, shm_cache_path)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARNING] Pre-tokenization of KDWD failed: {e}. Falling back to disk streaming.", flush=True)

    def __len__(self) -> int:
        if getattr(self, "cached_offsets", None) is not None and len(self.cached_offsets) > 1:
            return len(self.cached_offsets) - 1
        return 30000

    def __iter__(self):
        import os
        global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            total_workers = world_size * worker_info.num_workers
            worker_id = global_rank * worker_info.num_workers + worker_info.id
        else:
            total_workers = world_size
            worker_id = global_rank

        if getattr(self, "cached_flat", None) is not None and len(self.cached_offsets) > 1:
            total_items = len(self.cached_offsets) - 1
            start_idx = worker_id % total_workers
            for idx in range(start_idx, total_items, total_workers):
                st = self.cached_offsets[idx].item()
                ed = self.cached_offsets[idx + 1].item()
                text_ids = self.cached_flat[st:ed].to(torch.long).tolist()
                corrupted_ids = apply_dae_corruptions(
                    text_ids.copy(), self.eng_vocab.UNK_ID
                )
                corrupted_ids = (
                    [self.eng_vocab.BOS_ID]
                    + corrupted_ids[: self.max_len - 2]
                    + [self.eng_vocab.EOS_ID]
                )
                tgt_ids = (
                    [self.eng_vocab.BOS_ID]
                    + text_ids[: self.max_len - 2]
                    + [self.eng_vocab.EOS_ID]
                )
                yield {
                    "input_ids": torch.clamp(torch.tensor(corrupted_ids, dtype=torch.long), min=0, max=len(self.eng_vocab)-1),
                    "target_ids": torch.clamp(torch.tensor(tgt_ids, dtype=torch.long), min=0, max=len(self.eng_vocab)-1),
                    "is_dae": True,
                }
            return

        jsonl_path = self.kdwd_dir / "link_annotated_text.jsonl"
        if not jsonl_path.exists():
            return

        with open(jsonl_path, "r", encoding="utf-8") as f:
            from itertools import islice
            self.failures = 0
            import re
            page_id_re = re.compile(r'"page_id":\s*(\d+)')
            for line in islice(f, worker_id, None, total_workers):
                try:
                    if self.valid_page_ids:
                        match = page_id_re.search(line)
                        if match:
                            page_id = int(match.group(1))
                            if page_id not in self.valid_page_ids:
                                continue
                    
                    data = json.loads(line)

                    sections = data.get("sections", [])
                    for sec in sections:
                        text = sec.get("text", "").strip()
                        if len(text) < 20:
                            continue

                        # Encode target
                        text_ids = self.eng_vocab.encode(text)

                        # Apply DAE for input
                        corrupted_ids = apply_dae_corruptions(
                            text_ids.copy(), self.eng_vocab.UNK_ID
                        )

                        corrupted_ids = (
                            [self.eng_vocab.BOS_ID]
                            + corrupted_ids[: self.max_len - 2]
                            + [self.eng_vocab.EOS_ID]
                        )
                        text_ids = (
                            [self.eng_vocab.BOS_ID]
                            + text_ids[: self.max_len - 2]
                            + [self.eng_vocab.EOS_ID]
                        )

                        yield {
                            "input_ids": torch.tensor(corrupted_ids, dtype=torch.long),
                            "target_ids": torch.tensor(text_ids, dtype=torch.long),
                            "is_dae": True,
                        }
                except Exception as e:
                    self.failures += 1
                    if self.failures % 100 == 0:
                        print(f"[Worker {worker_id}] KDWD parsing failures: {self.failures}, last error: {e}")

class ASLGPC12Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        eng_vocab: EnglishVocabulary,
        gloss_vocab: GlossVocabulary,
        max_len: int = 128,
        reverse: bool = False,
    ):
        self.max_len = max_len
        self.eng_vocab = eng_vocab
        self.gloss_vocab = gloss_vocab
        self.reverse = reverse

        import pandas as pd
    
        import os, hashlib
        cache_key_str = f"{csv_path}_{len(self.gloss_vocab)}_{len(self.eng_vocab)}_{self.max_len}"
        cache_hash = hashlib.sha256(cache_key_str.encode('utf-8')).hexdigest()[:12]
        cache_fname = f"aslg_cached_tokens_{cache_hash}.pt"
        
        if os.path.exists("/dev/shm") and os.access("/dev/shm", os.W_OK):
            shm_aslg_cache = Path(f"/dev/shm/{cache_fname}")
        else:
            shm_aslg_cache = Path(f"./{cache_fname}").absolute()
        if shm_aslg_cache.exists():
            try:
                cached_data = torch.load(shm_aslg_cache)
                self.gloss_flat = cached_data["gloss_flat"]
                self.gloss_offsets = cached_data["gloss_offsets"]
                self.text_flat = cached_data["text_flat"]
                self.text_offsets = cached_data["text_offsets"]
                self.gloss_offsets_np = self.gloss_offsets.numpy()
                self.text_offsets_np = self.text_offsets.numpy()
                return
            except Exception:
                pass

        if Path(csv_path).exists():
            rank = 0
            try:
                import torch_xla.runtime as xr
                if hasattr(xr, "is_runtime_initialized") and xr.is_runtime_initialized():
                    rank = xr.global_ordinal()
                else:
                    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            except Exception:
                rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            
            if rank != 0:
                # Non-master ranks wait for master to generate the cache
                import time
                for _ in range(2400):  # 20 minutes timeout
                    if shm_aslg_cache.exists():
                        try:
                            cached_data = torch.load(shm_aslg_cache)
                            self.gloss_flat = cached_data["gloss_flat"]
                            self.gloss_offsets = cached_data["gloss_offsets"]
                            self.text_flat = cached_data["text_flat"]
                            self.text_offsets = cached_data["text_offsets"]
                            self.gloss_offsets_np = self.gloss_offsets.numpy()
                            self.text_offsets_np = self.text_offsets.numpy()
                            return
                        except Exception:
                            pass
                    time.sleep(0.5)
                raise TimeoutError("Non-master rank timed out waiting for ASLG cache generation from rank 0.")

            # Master rank loads the DataFrame
            df = pd.read_csv(csv_path)
            gloss_col = next((c for c in ["gloss", "sent.gloss", "sent_gloss"] if c in df.columns), None)
            text_col = next((c for c in ["text", "sent.eng", "english", "sent_eng"] if c in df.columns), None)
            if not gloss_col or not text_col:
                raise ValueError(f"[FATAL ASLG ERROR] ASLG-PC12 CSV missing required gloss/text columns. Found: {list(df.columns)}")
            df = df.dropna(subset=[gloss_col, text_col])
            
            import gc
            import os
            raw_gloss_list = df[gloss_col].astype(str).tolist()
            raw_text_list = df[text_col].astype(str).tolist()
            del df
            gc.collect()

            from concurrent.futures import ThreadPoolExecutor

            gloss_bos = self.gloss_vocab.BOS_ID
            gloss_eos = self.gloss_vocab.EOS_ID
            gloss_enc = self.gloss_vocab.encode
            max_sub = self.max_len - 2

            def _proc_gloss(s):
                return [gloss_bos] + gloss_enc(s, is_chicago=False)[:max_sub] + [gloss_eos]

            eng_bos = self.eng_vocab.BOS_ID
            eng_eos = self.eng_vocab.EOS_ID
            eng_enc = self.eng_vocab.encode

            def _proc_eng(s):
                return [eng_bos] + eng_enc(s)[:max_sub] + [eng_eos]

            num_threads = min(32, os.cpu_count() or 4)
            with ThreadPoolExecutor(max_workers=num_threads) as pool:
                encoded_gloss_list = list(pool.map(_proc_gloss, raw_gloss_list, chunksize=2000))
                encoded_text_list = list(pool.map(_proc_eng, raw_text_list, chunksize=2000))

            gloss_flat = []
            gloss_offsets = [0]
            for seq in encoded_gloss_list:
                gloss_flat.extend(seq)
                gloss_offsets.append(len(gloss_flat))

            text_flat = []
            text_offsets = [0]
            for seq in encoded_text_list:
                text_flat.extend(seq)
                text_offsets.append(len(text_flat))

            del raw_gloss_list, raw_text_list, encoded_gloss_list, encoded_text_list
            gc.collect()

            self.gloss_flat = torch.tensor(gloss_flat, dtype=torch.int32)
            self.gloss_offsets = torch.tensor(gloss_offsets, dtype=torch.int32)
            self.text_flat = torch.tensor(text_flat, dtype=torch.int32)
            self.text_offsets = torch.tensor(text_offsets, dtype=torch.int32)
            self.gloss_offsets_np = self.gloss_offsets.numpy()
            self.text_offsets_np = self.text_offsets.numpy()
            del gloss_flat, gloss_offsets, text_flat, text_offsets
            gc.collect()

            if rank == 0:
                try:
                    tmp_aslg_cache = shm_aslg_cache.with_suffix(".tmp")
                    torch.save({
                        "gloss_flat": self.gloss_flat,
                        "gloss_offsets": self.gloss_offsets,
                        "text_flat": self.text_flat,
                        "text_offsets": self.text_offsets,
                    }, tmp_aslg_cache)
                    os.replace(tmp_aslg_cache, shm_aslg_cache)
                except Exception:
                    pass
        else:
            raise FileNotFoundError(
                f"[FATAL ASLG ERROR] ASLG-PC12 CSV file not found at '{csv_path}'. "
                f"Phase 1 text pre-training requires valid ASLG-PC12 dataset!"
            )

    def __len__(self):
        return len(self.gloss_offsets) - 1 if getattr(self, "gloss_offsets", None) is not None else 0

    def __getitem__(self, idx: int):
        g_st = int(self.gloss_offsets_np[idx])
        g_ed = int(self.gloss_offsets_np[idx + 1])
        t_st = int(self.text_offsets_np[idx])
        t_ed = int(self.text_offsets_np[idx + 1])

        gloss_ids = self.gloss_flat[g_st:g_ed].to(torch.long)
        text_ids = self.text_flat[t_st:t_ed].to(torch.long)

        if self.reverse:
            return {
                "input_ids": text_ids,
                "target_ids": gloss_ids,
                "is_dae": False,
            }
        else:
            return {
                "input_ids": gloss_ids,
                "target_ids": text_ids,
                "is_dae": False,
            }


class Phase1MixedIterable(torch.utils.data.IterableDataset):
    """Mixes KDWD DAE and ASLG-PC12 Gloss-to-English 50/50 continuously."""

    def __init__(
        self, kdwd_dir: str, aslg_csv: str, eng_vocab, gloss_vocab, max_len=128
    ):
        self.kdwd_ds = KDWDDataset(kdwd_dir, eng_vocab, max_len)
        self.aslg_ds = ASLGPC12Dataset(aslg_csv, eng_vocab, gloss_vocab, max_len)
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def __len__(self) -> int:
        """Full epoch boundary covering combined ASLG-PC12 and KDWD datasets."""
        aslg_count = 0
        try:
            aslg_count = len(self.aslg_ds)
        except (TypeError, AttributeError):
            aslg_count = 87710

        kdwd_count = 0
        try:
            kdwd_count = len(self.kdwd_ds)
        except (TypeError, AttributeError):
            kdwd_count = 30000

        total = aslg_count + kdwd_count
        return total if total > 0 else 100000

    def __iter__(self):
        import random

        worker_info = torch.utils.data.get_worker_info()
        aslg_len = len(self.aslg_ds)
        if aslg_len == 0:
            raise RuntimeError("[FATAL ASLG ERROR] ASLG dataset length is 0 during iterator creation.")

        import os
        global_rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        try:
            import torch_xla.runtime as xr
            if hasattr(xr, "is_runtime_initialized") and xr.is_runtime_initialized() and xr.world_size() > 1:
                global_rank = xr.global_ordinal()
                world_size = xr.world_size()
        except Exception:
            pass

        if worker_info is not None:
            total_workers = world_size * worker_info.num_workers
            worker_id = global_rank * worker_info.num_workers + worker_info.id
            random.seed((worker_info.seed + worker_id) % (2**31))
        else:
            total_workers = world_size
            worker_id = global_rank
            random.seed((int(torch.initial_seed()) + worker_id) % (2**31))

        # Shuffled epoch permutation indexing for ASLG (Claims 3 & 4 Fix: 100% unique coverage per epoch)
        def _aslg_gen():
            local_indices = list(range(worker_id, aslg_len, total_workers))
            if not local_indices:
                local_indices = list(range(aslg_len))
            while True:
                random.shuffle(local_indices)
                for idx in local_indices:
                    yield self.aslg_ds[idx]

        aslg_iter = _aslg_gen()
        has_kdwd = self.kdwd_ds.kdwd_dir.exists() and (self.kdwd_ds.kdwd_dir / "link_annotated_text.jsonl").exists()
        kdwd_iter = iter(self.kdwd_ds) if has_kdwd else None

        while not self._stop_flag:
            use_dae = (random.random() < 0.5) if has_kdwd else False
            try:
                if use_dae and kdwd_iter is not None:
                    yield next(kdwd_iter)
                else:
                    yield next(aslg_iter)
            except StopIteration:
                if use_dae and has_kdwd:
                    kdwd_iter = iter(self.kdwd_ds)
                    try:
                        yield next(kdwd_iter)
                    except StopIteration:
                        yield next(aslg_iter)
                else:
                    aslg_iter = _aslg_gen()
                    yield next(aslg_iter)


def phase1_collate_fn(batch, max_len=256, eng_pad_id=0):
    # Enforce strict static tensor padding for PyTorch XLA TPU execution to compile EXACTLY 1 graph
    max_in = max_len
    max_tgt = max_len

    bsz = len(batch)
    input_padded = torch.full((bsz, max_in), eng_pad_id, dtype=torch.long)
    target_padded = torch.full((bsz, max_tgt), eng_pad_id, dtype=torch.long)
    is_dae_mask = torch.zeros(bsz, dtype=torch.bool)

    for i, x in enumerate(batch):
        in_seq = x["input_ids"]
        tgt_seq = x["target_ids"]
        in_len = min(max_in, len(in_seq))
        tgt_len = min(max_tgt, len(tgt_seq))
        if in_len > 0:
            input_padded[i, :in_len] = in_seq[:in_len] if isinstance(in_seq, torch.Tensor) else torch.as_tensor(in_seq[:in_len], dtype=torch.long)
        if tgt_len > 0:
            target_padded[i, :tgt_len] = tgt_seq[:tgt_len] if isinstance(tgt_seq, torch.Tensor) else torch.as_tensor(tgt_seq[:tgt_len], dtype=torch.long)
        if x.get("is_dae", False):
            is_dae_mask[i] = True

    return {
        "input_ids": input_padded,
        "target_ids": target_padded,
        "is_dae": is_dae_mask,
    }


def phase2_collate_fn(batch, max_len=256, eng_pad_id=0):
    max_in = max_len
    max_tgt = max_len

    bsz = len(batch)
    input_padded = torch.full((bsz, max_in), eng_pad_id, dtype=torch.long)  # Qwen/English PAD
    target_padded = torch.full((bsz, max_tgt), 0, dtype=torch.long)  # Gloss PAD

    for i, x in enumerate(batch):
        in_len = min(max_in, len(x["input_ids"]))
        tgt_len = min(max_tgt, len(x["target_ids"]))
        input_padded[i, :in_len] = (
            x["input_ids"][:in_len]
            if isinstance(x["input_ids"], torch.Tensor)
            else torch.as_tensor(x["input_ids"][:in_len], dtype=torch.long)
        )
        target_padded[i, :tgt_len] = (
            x["target_ids"][:tgt_len]
            if isinstance(x["target_ids"], torch.Tensor)
            else torch.as_tensor(x["target_ids"][:tgt_len], dtype=torch.long)
        )

    return {
        "input_ids": input_padded,
        "target_ids": target_padded,
    }
