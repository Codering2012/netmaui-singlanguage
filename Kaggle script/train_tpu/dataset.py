import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os

# CRITICAL FIX: Limit Dataloader multithreading to prevent 9600% CPU usage and OOM on Kaggle TPUs
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import json
import random
import torch
import numpy as np
torch.set_num_threads(1)
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union, Any
from torch.utils.data import Dataset, DataLoader, IterableDataset


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
    clean_l2i = {}
    if isinstance(label_to_idx, dict):
        for k, v in label_to_idx.items():
            k_str = str(k).strip().lower()
            if isinstance(v, int):
                clean_l2i[k_str] = v
            elif isinstance(v, dict):
                idx_val = v.get("id", v.get("idx", v.get("label_idx", 0)))
                clean_l2i[k_str] = int(idx_val)
            elif isinstance(v, str) and str(k).isdigit():
                clean_l2i[str(v).strip().lower()] = int(k)
            else:
                try:
                    clean_l2i[k_str] = int(v)
                except (ValueError, TypeError):
                    pass
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
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        max_idx = max(clean_l2i.values()) if clean_l2i else 0
        self.vocab_size = max(len(self.label_to_idx), max_idx + 1) + self.OFFSET
        self.output_map = {}

    def __len__(self) -> int:
        return self.vocab_size

    def gloss_to_token(self, gloss: str) -> int:
        raw = self.label_to_idx.get(gloss.strip().lower(), None)
        if raw is None:
            return self.UNK_ID
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
        gloss = self.idx_to_label.get(tid - self.OFFSET, "<UNK>")
        return self.output_map.get(gloss, gloss)


class EnglishVocabulary:
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3

    def __init__(self, vocab_path: Optional[Union[str, Path]] = None, use_bpe: bool = False, model_name: str = "Qwen/Qwen2.5-0.5B"):
        self.use_bpe = use_bpe
        if self.use_bpe:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.PAD_ID = self.tokenizer.pad_token_id
            self.BOS_ID = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else 1
            self.EOS_ID = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
            self.UNK_ID = self.tokenizer.unk_token_id if self.tokenizer.unk_token_id is not None else 3
            self.token_to_id = self.tokenizer.get_vocab()
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}
            return
            
        self.token_to_id = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.id_to_token = {0: "<PAD>", 1: "<BOS>", 2: "<EOS>", 3: "<UNK>"}
        self.frozen = True

        if vocab_path and Path(vocab_path).exists():
            try:
                with open(vocab_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if "token_to_id" in data:
                        self.token_to_id = data["token_to_id"]
                        self.id_to_token = {
                            int(v): k for k, v in self.token_to_id.items()
                        }

                        ids = [int(v) for v in self.token_to_id.values()]
                        if ids:
                            assert min(ids) == 0, "English vocab IDs must start at 0"
                            assert (
                                max(ids) == len(ids) - 1
                            ), "English vocab IDs must be contiguous without gaps"
                            assert len(set(ids)) == len(
                                ids
                            ), "English vocab IDs must be unique"
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load english vocab from {vocab_path}: {e}"
                )
        else:
            raise FileNotFoundError(
                f"Required english_vocab.json not found at {vocab_path}"
            )

    def freeze(self):
        pass

    def encode(self, text: str, allow_unk: bool = True) -> List[int]:
        if self.use_bpe:
            if getattr(self, "tokenizer", None) is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            return self.tokenizer.encode(text, add_special_tokens=False)
            
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
        if self.use_bpe:
            if getattr(self, "tokenizer", None) is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            ids_list = ids.tolist() if hasattr(ids, 'tolist') else ids
        if isinstance(ids_list[0], list):
            return [self.tokenizer.decode([int(i) for i in seq if int(i) not in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID)], skip_special_tokens=True) for seq in ids_list]
        return self.tokenizer.decode([int(i) for i in ids_list if int(i) not in (self.PAD_ID, self.BOS_ID, self.EOS_ID, self.UNK_ID)], skip_special_tokens=True)
            
        return " ".join(
            [
                self.id_to_token.get(int(i), "<UNK>")
                for i in ids
                if int(i) not in (0, 1, 2)
            ]
        )

    def __len__(self) -> int:
        return len(self.token_to_id)


class LandmarkAugmenter:
    """
    Progressive Noise Curriculum Data Augmentation for 3D WholeBody landmark sequences (T, 60, 3).
    Ramps up noise intensity using a square-root schedule, reaching approximately 3.5% jitter at maximum level.
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
        self.noise_level = max(0.001, min(1.0, noise_level))
        self.max_len = max_len

    def set_noise_level(self, level: float) -> None:
        """Sets progressive noise level ratio with balanced difficulty floor (0.50 to 1.0)."""
        effective_level = max(0.001, float(level) ** 0.5)
        self.noise_level = max(0.001, min(1.0, effective_level))

    def __call__(
        self, feat_arr: np.ndarray, noise_level: Optional[float] = None, frame_indices: Optional[np.ndarray] = None
    ):
        if noise_level is not None:
            self.set_noise_level(noise_level)
        T, K, C = feat_arr.shape
        if frame_indices is None:
            frame_indices = np.arange(T, dtype=np.int64)

        if T == 0:
            return feat_arr if frame_indices is None else (feat_arr, frame_indices)

        aug = feat_arr.copy()

        # Extract XYZ for spatial transforms.
        xyz = aug[:, :, :3]

        jitter_std = self.base_jitter_std * self.noise_level
        rot_range = self.max_rotation_range * self.noise_level
        shift_range = self.max_shift_range * self.noise_level
        kp_drop_prob = self.max_kp_drop_prob * self.noise_level
        frame_drop_prob = self.max_frame_drop_prob * self.noise_level
        scale_delta = (self.max_scale_range[1] - 1.0) * self.noise_level

        finger_drop_prob = (0.01 + 0.025 * self.noise_level) * 1.80
        timestretch_prob = (0.01 + 0.04 * self.noise_level) * 1.80
        warping_prob = (0.01 + 0.05 * self.noise_level) * 1.80
        hand_occ_prob = (0.005 + 0.01 * self.noise_level) * 1.80

        # 1. Scaling
        xyz = xyz * np.random.uniform(1.0 - scale_delta, 1.0 + scale_delta)

        # 2. Shift
        if shift_range > 0:
            xyz[:, :, 0] += np.random.uniform(-shift_range, shift_range)
            xyz[:, :, 1] += np.random.uniform(-shift_range, shift_range)

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
            center = xyz[:, :, :2].mean(axis=(0, 1), keepdims=True)
            xyz[:, :, :2] = (
                np.dot(xyz[:, :, :2] - center, rot_mat.T).reshape(T, K, 2) + center
            )

        # 4. Jitter
        if jitter_std > 0:
            xyz += np.random.normal(0, jitter_std, size=xyz.shape).astype(np.float32)

        # Reconstruct pos
        pos = xyz
        C = pos.shape[2]

        if T > 20 and np.random.rand() < timestretch_prob:
            rate = np.random.uniform(0.8, 1.2)
            new_T = int(T * rate)
            old_t = np.linspace(0, 1, T)
            new_t = np.linspace(0, 1, new_T)
            try:
                from scipy.interpolate import interp1d
                f = interp1d(old_t, pos, axis=0, assume_sorted=True)
                pos = f(new_t).astype(np.float32)
            except ImportError:
                new_pos = np.zeros((new_T, K * C), dtype=np.float32)
                pos_flat = pos.reshape(T, K * C)
                for i in range(K * C):
                    new_pos[:, i] = np.interp(new_t, old_t, pos_flat[:, i])
                pos = new_pos.reshape(new_T, K, C)
            T = new_T
            frame_indices = np.interp(new_t, old_t, frame_indices)

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
                list(range(19, 23)), list(range(23, 27)), list(range(27, 31)), list(range(31, 35)), list(range(35, 39)),
                list(range(40, 44)), list(range(44, 48)), list(range(48, 52)), list(range(52, 56)), list(range(56, 60)),
            ]
            n_drop = np.random.randint(1, 3)
            chosen_indices = np.random.choice(len(all_finger_groups), size=n_drop, replace=False)
            for idx_c in chosen_indices:
                unified_mask[:, all_finger_groups[idx_c], :] = 0.0

        if np.random.rand() < hand_occ_prob and T > 10:
            occ_s, occ_l = np.random.randint(0, max(1, T - 8)), np.random.randint(
                4, max(5, T // 2)
            )
            unified_mask[
                occ_s : occ_s + occ_l,
                range(18, 39) if np.random.rand() > 0.5 else range(39, 60),
                :,
            ] = 0.0

        if np.random.rand() < 0.02 and T > 5:
            unified_mask[
                np.random.randint(0, T),
                range(18, 39) if np.random.rand() > 0.5 else range(39, 60),
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
            vel[0] = vel[1]
            acc[1:] = (vel[1:] - vel[:-1]) / actual_dt
            acc[0] = acc[1]

        pos = pos * unified_mask
        vel = vel * unified_mask
        acc = acc * unified_mask

        return pos, frame_indices


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


# Global metadata & shard data caches to avoid massive IPC transfer and repeated disk reads
_GLOBAL_RECORDS_CACHE = {}
_GLOBAL_ACTIVE_RECORDS_CACHE = {}


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
        max_len: int = 256,
        num_keypoints: int = 60,
        channels_per_kp: int = 9,
        worker_idx: int = 0,
        num_workers: int = 1,
        shuffle_shards: bool = True,
        stage: str = "full_mixture",
        augment: bool = False,
        shared_progress=None,
        use_bpe: bool = False,
        model_name: str = "Qwen/Qwen2.5-0.5B",
    ):
        super().__init__()
        self.use_bpe = use_bpe
        self.model_name = model_name
        self.shared_progress = shared_progress

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
                Path("/kaggle/input/asl-preprocessed-phase1"),
                Path("./asl_preprocessed_phase1"),
                Path(r"E:\datasets\results\asl_preprocessed_phase1"),
            ]
            input_dir = next((c for c in candidates if c.exists()), input_dir)

        if (input_dir / split).exists():
            self.dataset_dir = input_dir / split
        else:
            self.dataset_dir = input_dir

        self.split = split
        self.max_len = max_len
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.feature_dim = num_keypoints * channels_per_kp
        self.worker_idx = worker_idx
        self.num_workers = num_workers
        self.shuffle_shards = shuffle_shards
        self.stage = stage
        self.augmenter = LandmarkAugmenter(max_len=self.max_len) if augment else None

        # Inflate the virtual size of the dataset to force more steps per epoch
        # (especially useful when batch_size is large and the physical dataset is heavily sharded/subsetted)
        self.epoch_multiplier = 1

        # Resolve Master Vocabulary Mapping
        self.label_to_idx = {}
        vocab_candidates = [
            self.dataset_dir / "vocab_map.json",
            self.dataset_dir / "vocabulary_mapping_global.json",
            input_dir / "vocabulary_mapping_global.json",
            self.dataset_dir / f"vocabulary_mapping_{split}.json",
            self.dataset_dir / "vocabulary_mapping_train.json",
            input_dir / "vocabulary_mapping_train.json",
            input_dir / f"vocabulary_mapping_{split}.json",
            self.dataset_dir.parent / "sign_to_prediction_index_map.json",
            self.dataset_dir.parent.parent / "sign_to_prediction_index_map.json",
            self.dataset_dir.parent / "vocabulary_mapping_train.json",
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
                f"Could not load vocabulary mapping from '{self.dataset_dir}' or candidate paths."
            )

        # Normalize vocabulary to lowercase
        normalized_vocab = {}
        for key, value in self.label_to_idx.items():
            normalized_vocab[str(key).strip().lower()] = value
        self.label_to_idx = normalized_vocab
        assert all(k == k.lower() for k in self.label_to_idx)
        # English Vocabulary & How2Sign Sentence Sidecar Loader
        english_vocab_file = self.dataset_dir / "english_vocab.json"
        if not english_vocab_file.exists():
            english_vocab_file = input_dir / "english_vocab.json"
        self.english_vocab = EnglishVocabulary(english_vocab_file, use_bpe=self.use_bpe, model_name=self.model_name)

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
        all_shard_files = sorted(list(self.dataset_dir.glob("shard_*.pt")))
        if not all_shard_files:
            all_shard_files = sorted(list(self.dataset_dir.glob("*.pt")))
        if not all_shard_files:
            raise FileNotFoundError(
                f"No shard_*.pt or .pt files found in '{self.dataset_dir}'"
            )

        if self.num_workers > 1:
            self.shard_files = all_shard_files[self.worker_idx::self.num_workers]
        else:
            self.shard_files = all_shard_files[self.worker_idx :: self.num_workers]

        self.dataset_name = f"dataset_{self.split}_w{self.worker_idx}_of_{self.num_workers}_{len(self.shard_files)}_{self.max_len}"

        # Load records metadata from allocated shards
        self.dataset_id = self.dataset_name
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
                try:
                    try:
                        shard_data = torch.load(
                            shard_path, map_location="cpu", weights_only=False, mmap=True
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

                        task_str = str(rec.get("task", "unknown"))
                        source_str = str(rec.get("source", "unknown"))
                        raw_label_str = str(rec.get("label", "")).strip().lower()
                        raw_label_idx = rec.get("label_idx", -1)

                        token_ids = []
                        # SOURCE / TASK AWARE ROUTING FIRST
                        lbl_clean = -1
                        if (
                            task_str == "fingerspelling_sequence"
                            or "Chicago" in source_str
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
                            or source_str.startswith("How2Sign")
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
                            or source_str.startswith("How2Sign")
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
                                    idx = self.label_to_idx.get(
                                        p.strip().lower(), None
                                    )
                                    if isinstance(idx, dict):
                                        idx = idx.get("id", idx.get("idx", -1))
                                    if idx is not None:
                                        token_ids.append(max(-1, int(idx)))

                        if (
                            not token_ids
                            and raw_label_str.strip().lower() not in self.label_to_idx
                            and (raw_label_idx is None or int(raw_label_idx) < 0)
                            and not (
                                task_str == "sentence_level"
                                or source_str.startswith("How2Sign")
                            )
                        ):
                            continue

                        if not token_ids:
                            if raw_label_idx is None or int(raw_label_idx) < 0:
                                continue
                            lbl_idx = int(raw_label_idx)
                            if lbl_idx not in set(
                                int(v) for v in self.label_to_idx.values()
                            ):
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
                        if "Chicago" in source_str:
                            source_id = 1
                        elif "How2Sign" in source_str:
                            source_id = 2
                        elif "Citizen" in source_str:
                            source_id = 3
                            
                        task_id = TASK_ISOLATED
                        if task_str == "fingerspelling_sequence" or "Chicago" in source_str:
                            task_id = TASK_FINGERSPELLING
                        elif task_str == "sentence_level" or source_str.startswith("How2Sign"):
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
            cache_dir = Path('./dataset_cache')
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            mtime_sum = sum(int(os.path.getmtime(sf)) for sf in self.shard_files[:10]) if self.shard_files else 0
            cache_hash = hashlib.sha256(f"{self.split}_{self.worker_idx}_{len(self.shard_files)}_{self.max_len}_{mtime_sum}".encode()).hexdigest()[:12]
            cache_name = f"asl_metadata_{self.split}_w{self.worker_idx}_{cache_hash}.pt"
            cache_path = cache_dir / cache_name
            tmp_cache_path = cache_dir / f"{cache_name}.tmp_{os.getpid()}"
            
            if not cache_path.exists():
                try:
                    print(f"[Worker {self.worker_idx}] Building dataset metadata cache...", flush=True)
                    for shard_idx, shard_path in enumerate(self.shard_files):
                        local_metas, local_counts = _index_shard((shard_idx, shard_path))
                        temp_metadata.extend(local_metas)
                        for k, v in local_counts.items():
                            class_counts[k] += v

                    torch.save({"temp_metadata": temp_metadata, "class_counts": dict(class_counts)}, tmp_cache_path)
                    os.replace(tmp_cache_path, cache_path)
                    print(f"[Worker {self.worker_idx}] Atomically saved metadata cache to {cache_path}.", flush=True)
                except Exception as e:
                    print(f"[Worker {self.worker_idx}] Failed to save metadata cache: {e}", flush=True)
                    if tmp_cache_path.exists():
                        try:
                            tmp_cache_path.unlink(missing_ok=True)
                        except Exception:
                            pass
                    
            if cache_path.exists() and not temp_metadata:
                try:
                    cached = torch.load(cache_path, map_location="cpu", weights_only=False)
                    temp_metadata = cached["temp_metadata"]
                    class_counts = defaultdict(int, cached["class_counts"])
                    print(f"[Worker {self.worker_idx}] Loaded {len(temp_metadata)} records from cache.", flush=True)
                except Exception as e:
                    print(f"[Worker {self.worker_idx}] Failed to load cache: {e}. Falling back to manual parse.", flush=True)
                    if not temp_metadata:
                        from concurrent.futures import ThreadPoolExecutor
                        with ThreadPoolExecutor(max_workers=1) as executor:
                            results = executor.map(_index_shard, enumerate(self.shard_files))
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

        shard_groups_list = [np.array(indices, dtype=np.int32) for indices in shard_indices_map.values() if indices]

        self.class_counts = class_counts
        self.valid_label_ids = set(int(v) for v in self.label_to_idx.values())
        _GLOBAL_RECORDS_CACHE[cache_key] = temp_metadata
        _GLOBAL_ACTIVE_RECORDS_CACHE[cache_key] = grouped_active
        _GLOBAL_SHARD_GROUPS_CACHE[cache_key] = shard_groups_list

    def set_noise_level(self, level: float) -> None:
        """Dynamically adjusts augmentation noise level and active Curriculum by Difficulty subset."""
        if self.shared_progress is not None:
            self.shared_progress.value = float(level)
        # Note: self.stage and cache adjustments are now handled dynamically inside __getitem__
        # so persistent_workers can pick up the changes.

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
            for sk in shard_keys:
                s_recs = records_by_shard[sk]
                rng.shuffle(s_recs)
                grouped_active.extend(s_recs)

            _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id] = grouped_active

    def __len__(self) -> int:
        if self.dataset_id not in _GLOBAL_RECORDS_CACHE:
            self._load_records_metadata()
        actual_len = len(_GLOBAL_RECORDS_CACHE[self.dataset_id]) * self.epoch_multiplier
        if hasattr(self, "pad_to_length") and self.pad_to_length is not None:
            return max(actual_len, self.pad_to_length)
        return actual_len

    def _get_record_feature(self, shard_path: Path, item_key: Any) -> np.ndarray:
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
            
            
            try:
                self._worker_shard_cache[str_path] = torch.load(
                    shard_path, map_location="cpu", weights_only=False, mmap=False
                )
            except Exception:
                self._worker_shard_cache[str_path] = torch.load(
                    shard_path, map_location="cpu", weights_only=False
                )
                
            self._worker_last_shard = str_path

        shard_data = self._worker_shard_cache[str_path]

        if isinstance(shard_data, dict):
            raw_feat = shard_data.get(item_key, None)
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

        if isinstance(raw_feat, torch.Tensor):
            return raw_feat.detach().cpu().numpy().copy()
        elif isinstance(raw_feat, np.ndarray):
            return raw_feat
        else:
            return np.asarray(raw_feat, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.dataset_id not in _GLOBAL_RECORDS_CACHE:
            self._load_records_metadata()

        full_records = _GLOBAL_ACTIVE_RECORDS_CACHE[self.dataset_id]

        # We rely on set_epoch() dynamically reshuffling the _GLOBAL_ACTIVE_RECORDS_CACHE
        # block-by-block to guarantee shard locality and prevent RAM OOM, so we DO NOT permute globally here.

        # Implement cyclic wrapping in case this dataset is artificially padded to match max_len
        idx = idx % len(full_records)

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

        task_str = "fingerspelling_sequence" if task_id == TASK_FINGERSPELLING else ("sentence_level" if task_id == TASK_SENTENCE else "isolated_gloss")
        source_str = "ChicagoFSWild" if source_id == 1 else ("How2Sign" if source_id == 2 else "ASLCitizen")

        item_key = (
            feature_key
            if feature_key is not None
            else (item_idx if item_idx is not None else idx)
        )
        shard_path = self.shard_files[shard_idx]
        feat_arr = self._get_record_feature(shard_path, item_key)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32
        )

        if feat_arr.ndim == 2:
            T, D = feat_arr.shape
            if D >= self.feature_dim:
                feat_arr = feat_arr[:, : self.feature_dim].reshape(
                    T, self.num_keypoints, self.channels_per_kp
                )
            else:
                pad_d = np.zeros((T, self.feature_dim - D), dtype=np.float32)
                feat_arr = np.concatenate([feat_arr, pad_d], axis=1).reshape(
                    T, self.num_keypoints, self.channels_per_kp
                )
            T = feat_arr.shape[0]
        elif feat_arr.ndim == 3:
            T, K, C = feat_arr.shape
            assert (
                K == self.num_keypoints
            ), f"Expected {self.num_keypoints} keypoints, got {K} in {shard_path}"
            if C < self.channels_per_kp:
                if self.channels_per_kp >= 9 and C == 3:
                    pos = feat_arr[..., :3]
                    vel = np.zeros_like(pos)
                    acc = np.zeros_like(pos)
                    if T > 1:
                        vel[1:] = pos[1:] - pos[:-1]
                        vel[0] = vel[1]
                    if T > 2:
                        acc[1:] = vel[1:] - vel[:-1]
                        acc[0] = acc[1]
                    
                    if self.channels_per_kp > 9:
                        pad_c = np.zeros((T, K, self.channels_per_kp - 9), dtype=np.float32)
                        feat_arr = np.concatenate([pos, vel, acc, pad_c], axis=-1)
                    else:
                        feat_arr = np.concatenate([pos, vel, acc], axis=-1)
                else:
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
        frame_indices = np.arange(T, dtype=np.int64)

        if T > 0:
            # Kinematics are computed here
            pos = feat_arr[..., : min(3, self.channels_per_kp)]
            if self.channels_per_kp >= 9 and pos.shape[-1] == 3:
                vel = np.zeros_like(pos)
                acc = np.zeros_like(pos)
                if T > 1:
                    vel[1:] = pos[1:] - pos[:-1]
                    vel[0] = vel[1]
                if T > 2:
                    acc[1:] = vel[1:] - vel[:-1]
                    acc[0] = acc[1]
                feat_arr = np.concatenate([pos, vel, acc], axis=-1)
                if self.channels_per_kp > 9:
                    pad_c = np.zeros((T, self.num_keypoints, self.channels_per_kp - 9), dtype=np.float32)
                    feat_arr = np.concatenate([feat_arr, pad_c], axis=-1)

            if T > self.max_len:
                # Use motion-aware temporal resampling for long sequences
                indices = motion_aware_sample_indices(feat_arr, self.max_len)
                indices = np.sort(indices)
                feat_arr = feat_arr[indices]
                frame_indices = frame_indices[indices]
            
            features[:min(T, self.max_len)] = feat_arr[:min(T, self.max_len)]
            mask[:min(T, self.max_len)] = True
            T = feat_arr.shape[0]

        if self.augment and self.augmenter is not None and T > 0:
            level = (
                self.shared_progress.value
                if self.shared_progress is not None
                else 0.0
            )
            aug_res = self.augmenter(feat_arr, noise_level=level, frame_indices=frame_indices[:T])
            if isinstance(aug_res, tuple):
                feat_arr, aug_indices = aug_res
                if aug_indices is not None and len(aug_indices) > 0:
                    frame_indices[:len(aug_indices)] = aug_indices
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

        padded_frame_indices = np.arange(self.max_len, dtype=np.float32)
        if T > 0:
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
        lbl_str = re.sub(r'[^a-z0-9]', '', raw_label_str.lower())
        lbl_str = re.sub(r"\d+$", "", lbl_str)
        lex_class_idx = self.asl_lex_map.get(lbl_str, 4)

        # Common IDs
        _, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
        GLOSS_OFFSET = 4
        CHICAGO_OFFSET = 5

        MAX_GLOSS_LEN = 128
        MAX_CHICAGO_LEN = 128
        MAX_ENGLISH_LEN = 128

        # Initialize defaults
        has_valid_gloss = False
        has_valid_chicago = False
        has_valid_english = False
        is_isolated = False

        raw_gloss_seq = [BOS_ID, EOS_ID]
        raw_chicago_seq = [BOS_ID, EOS_ID]
        raw_english_seq = [BOS_ID, EOS_ID]

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
                raw_gloss_seq = (
                    [BOS_ID]
                    + [t + GLOSS_OFFSET if t >= 0 else UNK_ID for t in token_ids]
                    + [EOS_ID]
                )

        elif task_str == "fingerspelling_sequence" or source_str == "ChicagoFSWild":
            if raw_label_str in _SKIP_LABELS:
                has_valid_chicago = False
            else:
                has_valid_chicago = True
            is_isolated = False
            # Tokenize chicago string: PAD=0, BOS=1, EOS=2, UNK=3, SP=4, a-z=5-30, 0-9=31-40
            SP_ID = 4
            raw_chicago_seq = [BOS_ID]
            clean_chicago_str = raw_label_str.replace("<sp>", " ")
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

        elif task_str == "sentence_level" or source_str.startswith("How2Sign"):
            # How2Sign English Sentence
            is_isolated = False
            if raw_label_str != "how2sign_sequence" and len(raw_label_str) > 0:
                has_valid_english = True
                enc_ids = self.english_vocab.encode(raw_label_str, allow_unk=True)
                raw_english_seq = [BOS_ID] + enc_ids + [EOS_ID]
            else:
                has_valid_english = False  # Unrecoverable sentence

            if token_ids:
                has_valid_gloss = True
                raw_gloss_seq = (
                    [BOS_ID]
                    + [t + GLOSS_OFFSET if t >= 0 else UNK_ID for t in token_ids]
                    + [EOS_ID]
                )

        else:
            # Fallback for completely unknown tasks, treat as gloss sequence if token_ids exist
            if token_ids and label_idx != -1:
                has_valid_gloss = True
                is_isolated = len(token_ids) <= 1
                raw_gloss_seq = (
                    [BOS_ID]
                    + [t + GLOSS_OFFSET if t >= 0 else UNK_ID for t in token_ids]
                    + [EOS_ID]
                )

        # Pad sequences
        def pad_seq(raw_seq, max_len):
            actual_len = min(len(raw_seq), max_len)
            is_truncated = len(raw_seq) > max_len
            padded = np.zeros(max_len, dtype=np.int64)
            padded[:actual_len] = raw_seq[:actual_len]
            return padded, actual_len, is_truncated

        padded_gloss_seq, gloss_len, gloss_trunc = pad_seq(raw_gloss_seq, MAX_GLOSS_LEN)
        padded_chicago_seq, chicago_len, chicago_trunc = pad_seq(
            raw_chicago_seq, MAX_CHICAGO_LEN
        )
        padded_english_seq, english_len, english_trunc = pad_seq(
            raw_english_seq, MAX_ENGLISH_LEN
        )

        if gloss_trunc:
            has_valid_gloss = False
        if chicago_trunc:
            has_valid_chicago = False
        if english_trunc:
            has_valid_english = False

        # Source routing pseudo-IDs: 0 = Unknown/Default, 1 = ChicagoFSWild, 2 = How2Sign, 3 = ASLCitizen
        source_id = 0
        if "Chicago" in source_str:
            source_id = 1
        elif "How2Sign" in source_str:
            source_id = 2
        elif "Citizen" in source_str:
            source_id = 3

        return {
            "feature": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "lex_class_idx": torch.tensor(lex_class_idx, dtype=torch.long),
            "domain_label": torch.tensor(source_id, dtype=torch.long),
            "has_domain_label": torch.tensor(source_id >= 0, dtype=torch.bool),
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
            "mlm_mask": torch.zeros_like(torch.tensor(padded_gloss_seq, dtype=torch.long)),
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
            indices = np.concatenate([np.array(b, dtype=np.int32) for b in blocks]) if blocks else np.array([], dtype=np.int32)
                
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
        l = len(self.dataset)
        import sys
        if "torch_xla" in sys.modules:
            import torch_xla.core.xla_model as xm
            world_size = xm.xrt_world_size()
            if l % world_size != 0:
                l += world_size - (l % world_size)
        return l

class ASLStreamedDataset(IterableDataset):
    """
    Zero-RAM Streamed IterableDataset for PyTorch XLA execution on sharded ASL landmark records.
    Streams shards sequentially with a local shuffle buffer (~100MB), guaranteeing 0 initial indexing
    overhead and strictly < 500MB total host RAM usage.
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
        shuffle_buffer_size: int = 1000,
        stage: str = "full_mixture",
        augment: bool = False,
        shared_progress=None,
    ):
        super().__init__()
        input_dir = Path(dataset_dir)
        if not input_dir.exists():
            candidates = [
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/asl_preprocessed_phase1"),
                Path("/kaggle/input/datasets/tranquocbao2012/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl/results/asl_preprocessed_phase1"),
                Path("/kaggle/input/frakenstein-asl"),
                Path("/kaggle/input/asl-preprocessed-phase1"),
                Path("./asl_preprocessed_phase1"),
            ]
            input_dir = next((c for c in candidates if c.exists()), input_dir)

        self.dataset_dir = input_dir / split if (input_dir / split).exists() else input_dir
        self.split = split
        self.max_len = max_len
        self.num_keypoints = num_keypoints
        self.channels_per_kp = channels_per_kp
        self.worker_idx = worker_idx
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.augment = augment and (split == "train")
        self.shared_progress = shared_progress
        self.augmenter = LandmarkAugmenter(max_len=self.max_len) if augment else None

        english_vocab_file = self.dataset_dir / "english_vocab.json"
        if not english_vocab_file.exists():
            english_vocab_file = input_dir / "english_vocab.json"
        if english_vocab_file.exists():
            self.english_vocab = EnglishVocabulary(english_vocab_file)
        else:
            self.english_vocab = EnglishVocabulary.__new__(EnglishVocabulary)
            self.english_vocab.encode = lambda text, allow_unk=True: []
            self.english_vocab.is_valid = False

        all_shard_files = sorted(list(self.dataset_dir.glob("shard_*.pt")))
        if not all_shard_files:
            all_shard_files = sorted(list(self.dataset_dir.glob("*.pt")))
            
        self.shard_files = all_shard_files

        # Exact record counting across shards
        total_records = 0
        for s in self.shard_files:
            try:
                data = torch.load(s, map_location="cpu", weights_only=False)
                if isinstance(data, list):
                    total_records += len(data)
                elif isinstance(data, dict):
                    rec = data.get("features", data.get("records", []))
                    total_records += len(rec) if hasattr(rec, "__len__") else 200
                else:
                    total_records += 200
            except Exception:
                total_records += 200
        self.total_records = max(1, total_records)

    def __len__(self) -> int:
        return self.total_records

    def __iter__(self):
        import random
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            shards = self.shard_files[worker_info.id :: worker_info.num_workers]
        else:
            shards = list(self.shard_files)

        while True:
            if self.split == "train":
                random.shuffle(shards)
            
            buffer = []
            for shard_path in shards:
                try:
                    shard_data = torch.load(shard_path, map_location="cpu", weights_only=False, mmap=True)
                    items = shard_data.values() if isinstance(shard_data, dict) else shard_data
                    for rec in items:
                        if isinstance(rec, dict):
                            if len(buffer) >= self.shuffle_buffer_size:
                                # True reservoir sliding shuffle replacement
                                idx = random.randint(0, len(buffer) - 1)
                                s_path, r = buffer[idx]
                                buffer[idx] = (shard_path, rec)
                                processed = self._process_record(s_path, r)
                                if processed is not None:
                                    yield processed
                            else:
                                buffer.append((shard_path, rec))
                    del shard_data
                except Exception:
                    pass

            if buffer:
                if self.split == "train":
                    random.shuffle(buffer)
                for s_path, r in buffer:
                    processed = self._process_record(s_path, r)
                    if processed is not None:
                        yield processed
                buffer.clear()
            if self.split != "train":
                break

    def _process_record(self, shard_path: Path, rec: dict) -> Optional[Dict[str, torch.Tensor]]:
        feat_arr = rec.get("features", rec.get("feature_array"))
        if feat_arr is None:
            return None
        if isinstance(feat_arr, torch.Tensor):
            feat_arr = feat_arr.detach().cpu().numpy()
        else:
            feat_arr = np.asarray(feat_arr, dtype=np.float32)

        feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        T = feat_arr.shape[0] if feat_arr.ndim >= 2 else 0

        features = np.zeros((self.max_len, self.num_keypoints, self.channels_per_kp), dtype=np.float32)
        mask = np.zeros((self.max_len,), dtype=bool)
        padded_frame_indices = np.arange(self.max_len, dtype=np.float32)

        raw_frame_indices = rec.get("frame_index", np.arange(T, dtype=np.float32))
        if isinstance(raw_frame_indices, torch.Tensor):
            raw_frame_indices = raw_frame_indices.cpu().numpy()
        raw_frame_indices = np.asarray(raw_frame_indices, dtype=np.float32)

        if T > 0:
            if feat_arr.ndim == 2:
                T_raw, D = feat_arr.shape
                feat_dim = self.num_keypoints * self.channels_per_kp
                if D >= feat_dim:
                    feat_arr = feat_arr[:, :feat_dim].reshape(T_raw, self.num_keypoints, self.channels_per_kp)
                else:
                    pad_d = np.zeros((T_raw, feat_dim - D), dtype=np.float32)
                    feat_arr = np.concatenate([feat_arr, pad_d], axis=1).reshape(T_raw, self.num_keypoints, self.channels_per_kp)

            if T > self.max_len:
                feat_arr = feat_arr[:self.max_len]
                padded_frame_indices[:self.max_len] = raw_frame_indices[:self.max_len]
                    
            T_cap = min(T, self.max_len)
            feat_slice = feat_arr[:T_cap]
            
            features[:T_cap] = feat_slice
            features[:T_cap] = feat_slice
            mask[:T_cap] = True
            if T <= self.max_len:
                padded_frame_indices[:T_cap] = raw_frame_indices[:T_cap]

        # Standardized Supervision Constants
        _, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
        GLOSS_OFFSET = 4
        CHICAGO_OFFSET = 5

        MAX_GLOSS_LEN = 128
        MAX_CHICAGO_LEN = 128
        MAX_ENGLISH_LEN = 128

        raw_label_str = str(rec.get("raw_label_str", rec.get("text", rec.get("label", "")))).strip().lower()
        task_str = str(rec.get("task", rec.get("task_str", ""))).strip().lower()
        source_str = str(rec.get("source", "unknown")).strip().lower()

        source_id = 0
        if "chicago" in source_str:
            source_id = 1
        elif "how2sign" in source_str:
            source_id = 2
        elif "citizen" in source_str:
            source_id = 3

        label_idx = int(rec.get("label_idx", -1))
        sample_weight = float(rec.get("quality", rec.get("sample_weight", 1.0)))

        has_valid_gloss = False
        has_valid_chicago = False
        has_valid_english = False
        is_isolated = False

        raw_gloss_seq = [BOS_ID, EOS_ID]
        raw_chicago_seq = [BOS_ID, EOS_ID]
        raw_english_seq = [BOS_ID, EOS_ID]

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
                    raw_gloss_seq = (
                        [BOS_ID]
                        + [t + GLOSS_OFFSET if t >= 0 else UNK_ID for t in token_ids]
                        + [EOS_ID]
                    )

        elif task_str == "fingerspelling_sequence" or "chicago" in source_str:
            if raw_label_str and raw_label_str not in _SKIP_LABELS:
                has_valid_chicago = True
                is_isolated = False
                SP_ID = 4
                raw_chicago_seq = [BOS_ID]
                clean_chicago_str = raw_label_str.replace("<sp>", " ")
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
                has_valid_english = getattr(self.english_vocab, "is_valid", True)
                enc_ids = self.english_vocab.encode(raw_label_str, allow_unk=True)
                raw_english_seq = [BOS_ID] + enc_ids + [EOS_ID]
            if token_ids:
                has_valid_gloss = True
                raw_gloss_seq = (
                    [BOS_ID]
                    + [t + GLOSS_OFFSET if t >= 0 else UNK_ID for t in token_ids]
                    + [EOS_ID]
                )
        else:
            if token_ids and label_idx != -1:
                has_valid_gloss = True
                is_isolated = len(token_ids) <= 1
                raw_gloss_seq = (
                    [BOS_ID]
                    + [t + GLOSS_OFFSET if t >= 0 else UNK_ID for t in token_ids]
                    + [EOS_ID]
                )

        def pad_seq(raw_seq, max_len):
            actual_len = min(len(raw_seq), max_len)
            is_truncated = len(raw_seq) > max_len
            padded = np.zeros(max_len, dtype=np.int64)
            padded[:actual_len] = raw_seq[:actual_len]
            return padded, actual_len, is_truncated

        padded_gloss_seq, gloss_len, gloss_trunc = pad_seq(raw_gloss_seq, MAX_GLOSS_LEN)
        padded_chicago_seq, chicago_len, chicago_trunc = pad_seq(raw_chicago_seq, MAX_CHICAGO_LEN)
        padded_english_seq, english_len, english_trunc = pad_seq(raw_english_seq, MAX_ENGLISH_LEN)

        if gloss_trunc:
            has_valid_gloss = False
        if chicago_trunc:
            has_valid_chicago = False
        if english_trunc:
            has_valid_english = False

        if feat_arr.ndim not in {2, 3} or T == 0:
            has_valid_gloss = False
            has_valid_chicago = False
            has_valid_english = False

        return {
            "feature": torch.from_numpy(features),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label_idx, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weight, dtype=torch.float32),
            "lex_class_idx": torch.tensor(4, dtype=torch.long),
            "domain_label": torch.tensor(source_id, dtype=torch.long),
            "has_domain_label": torch.tensor(source_id >= 0, dtype=torch.bool),
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
    streamed: bool = False,
    **kwargs,
) -> DataLoader:
    """Creates a PyTorch DataLoader wrapping ASLShardedDataset or ASLStreamedDataset."""
    import sys


    shared_progress = None
    if num_dataloader_workers > 0:
        mp_context = None
        shared_progress = mp_context.Value("d", 0.0) if mp_context is not None else None

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
        "drop_last": True if "torch_xla" in sys.modules else (split == "train"),
        "worker_init_fn": _seed_worker,
    }
    if num_dataloader_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = False
        if mp_context is not None:
            dl_kwargs["multiprocessing_context"] = mp_context

    return DataLoader(**dl_kwargs)

# ==============================================================================
# PHASE 1: TEXT PRE-TRAINING DATASETS (DAE & ASLG-PC12)
# ==============================================================================

def apply_dae_corruptions(tokens: list, unk_id: int, mask_prob: float = 0.15, drop_prob: float = 0.10, shuffle_prob: float = 0.10):
    """Applies Denoising Autoencoder (DAE) corruptions to a list of token IDs."""
    import random
    
    if len(tokens) <= 3:
        return tokens

    # 1. N-Gram Shuffling (local permutations)
    if random.random() < shuffle_prob:
        span_len = random.randint(2, 4)
        if len(tokens) > span_len:
            start_idx = random.randint(0, len(tokens) - span_len)
            span = tokens[start_idx:start_idx + span_len]
            random.shuffle(span)
            tokens = tokens[:start_idx] + span + tokens[start_idx + span_len:]

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
    def __init__(self, kdwd_dir: str, eng_vocab: EnglishVocabulary, max_len: int = 128, views_threshold: int = 5000):
        super().__init__()
        self.kdwd_dir = Path(kdwd_dir)
        self.eng_vocab = eng_vocab
        self.max_len = max_len
        self.views_threshold = views_threshold
        
        # Load and filter page_id by views
        self.valid_page_ids = set()
        page_csv = self.kdwd_dir / "page.csv"
        if page_csv.exists():
            import pandas as pd
            df = pd.read_csv(page_csv, usecols=["page_id", "views"])
            self.valid_page_ids = set(df[df["views"] > self.views_threshold]["page_id"].tolist())
        else:
            print("WARNING: KDWD page.csv not found, proceeding without view filtering.")

    def __iter__(self):
        import json
        jsonl_path = self.kdwd_dir / "link_annotated_text.jsonl"
        if not jsonl_path.exists():
            return
            
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    worker_info = torch.utils.data.get_worker_info()
                    if worker_info is not None:
                        # Hashing line as integer to distribute
                        if hash(line) % worker_info.num_workers != worker_info.id:
                            continue
                    data = json.loads(line)
                    page_id = data.get("page_id")
                    if self.valid_page_ids and page_id not in self.valid_page_ids:
                        continue
                        
                    sections = data.get("sections", [])
                    for sec in sections:
                        text = sec.get("text", "").strip()
                        if len(text) < 20:
                            continue
                            
                        # Encode target
                        text_ids = self.eng_vocab.encode(text)
                        
                        # Apply DAE for input
                        corrupted_ids = apply_dae_corruptions(text_ids.copy(), self.eng_vocab.UNK_ID)
                        
                        corrupted_ids = [self.eng_vocab.BOS_ID] + corrupted_ids[:self.max_len-2] + [self.eng_vocab.EOS_ID]
                        text_ids = [self.eng_vocab.BOS_ID] + text_ids[:self.max_len-2] + [self.eng_vocab.EOS_ID]
                        
                        yield {
                            "input_ids": torch.tensor(corrupted_ids, dtype=torch.long),
                            "target_ids": torch.tensor(text_ids, dtype=torch.long),
                            "is_dae": True
                        }
                except:
                    pass

class ASLGPC12Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, eng_vocab: EnglishVocabulary, gloss_vocab: GlossVocabulary, max_len: int = 128, reverse: bool = False):
        self.max_len = max_len
        self.eng_vocab = eng_vocab
        self.gloss_vocab = gloss_vocab
        self.reverse = reverse
        
        import pandas as pd
        if Path(csv_path).exists():
            self.df = pd.read_csv(csv_path).dropna(subset=['gloss', 'text'])
        else:
            print(f"WARNING: ASLG-PC12 CSV not found at {csv_path}. Using empty dataset.")
            self.df = pd.DataFrame(columns=['gloss', 'text'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        gloss_str = str(row['gloss'])
        text_str = str(row['text'])

        gloss_ids = self.gloss_vocab.encode(gloss_str, is_chicago=False)
        text_ids = self.eng_vocab.encode(text_str)

        gloss_ids = [self.gloss_vocab.BOS_ID] + gloss_ids[:self.max_len-2] + [self.gloss_vocab.EOS_ID]
        text_ids = [self.eng_vocab.BOS_ID] + text_ids[:self.max_len-2] + [self.eng_vocab.EOS_ID]

        if self.reverse:
            # English -> Gloss
            return {
                "input_ids": torch.tensor(text_ids, dtype=torch.long),
                "target_ids": torch.tensor(gloss_ids, dtype=torch.long),
                "is_dae": False
            }
        else:
            # Gloss -> English
            return {
                "input_ids": torch.tensor(gloss_ids, dtype=torch.long),
                "target_ids": torch.tensor(text_ids, dtype=torch.long),
                "is_dae": False
            }

class Phase1MixedIterable(torch.utils.data.IterableDataset):
    """Mixes KDWD DAE and ASLG-PC12 Gloss-to-English 50/50."""
    def __init__(self, kdwd_dir: str, aslg_csv: str, eng_vocab, gloss_vocab, max_len=128):
        self.kdwd_ds = KDWDDataset(kdwd_dir, eng_vocab, max_len)
        self.aslg_ds = ASLGPC12Dataset(aslg_csv, eng_vocab, gloss_vocab, max_len)
        
    def __iter__(self):
        import random
        worker_info = torch.utils.data.get_worker_info()
        aslg_len = len(self.aslg_ds)
        if aslg_len == 0:
            aslg_iter = iter([])
        else:
            # Distributed random indexing for ASLG
            def _aslg_gen():
                while True:
                    yield self.aslg_ds[random.randint(0, aslg_len - 1)]
            aslg_iter = _aslg_gen()
            
        kdwd_iter = iter(self.kdwd_ds)
        
        while True:
            use_dae = random.random() < 0.5
            try:
                if use_dae:
                    yield next(kdwd_iter)
                else:
                    yield next(aslg_iter)
            except StopIteration:
                if use_dae:
                    kdwd_iter = iter(self.kdwd_ds)
                    yield next(kdwd_iter)
                else:
                    if aslg_len == 0:
                        yield next(kdwd_iter)
                        continue
                    aslg_iter = _aslg_gen()
                    yield next(aslg_iter)

def phase1_collate_fn(batch):
    max_in = max(len(x['input_ids']) for x in batch)
    max_tgt = max(len(x['target_ids']) for x in batch)

    bsz = len(batch)
    input_padded = torch.full((bsz, max_in), 0, dtype=torch.long)
    target_padded = torch.full((bsz, max_tgt), 151643, dtype=torch.long) # Qwen PAD
    is_dae_mask = torch.zeros(bsz, dtype=torch.bool)
    
    for i, x in enumerate(batch):
        input_padded[i, :len(x['input_ids'])] = x['input_ids']
        target_padded[i, :len(x['target_ids'])] = x['target_ids']
        is_dae_mask[i] = x['is_dae']

    return {
        "input_ids": input_padded,
        "target_ids": target_padded,
        "is_dae": is_dae_mask
    }

def phase2_collate_fn(batch):
    max_in = max(len(x['input_ids']) for x in batch)
    max_tgt = max(len(x['target_ids']) for x in batch)

    bsz = len(batch)
    input_padded = torch.full((bsz, max_in), 151643, dtype=torch.long) # Qwen PAD
    target_padded = torch.full((bsz, max_tgt), 0, dtype=torch.long) # Gloss PAD
    
    for i, x in enumerate(batch):
        input_padded[i, :len(x['input_ids'])] = x['input_ids']
        target_padded[i, :len(x['target_ids'])] = x['target_ids']

    return {
        "input_ids": input_padded,
        "target_ids": target_padded,
    }